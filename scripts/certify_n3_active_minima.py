"""Certify the five active observer minima of the symmetric ``N=3`` root."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from square_riesz.certify import Box, decimal_string
from square_riesz.local_proof import (
    CoverResult,
    DerivativeCoverResult,
    certify_complement_gap,
    certify_derivative_region,
    symmetric_source_enclosure,
)


ACTIVE_REGIONS: dict[str, Box] = {
    "bottom_left": (Q(0), Q(1, 8), Q(0), Q(1, 8)),
    "bottom_right": (Q(7, 8), Q(1), Q(0), Q(1, 8)),
    "top_left": (Q(0), Q(1, 8), Q(7, 8), Q(1)),
    "top_right": (Q(7, 8), Q(1), Q(7, 8), Q(1)),
    "bottom_midpoint": (Q(3, 8), Q(5, 8), Q(0), Q(1, 8)),
}
DERIVATIVE_CONDITIONS = {
    "bottom_left": ("ux_positive", "uy_positive"),
    "bottom_right": ("ux_negative", "uy_positive"),
    "top_left": ("ux_positive", "uy_negative"),
    "top_right": ("ux_negative", "uy_negative"),
    "bottom_midpoint": ("uy_positive", "uxx_positive"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _box_json(box: Box) -> list[str]:
    return list(map(str, box))


def _exact(value: Q) -> dict[str, object]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal": decimal_string(value, 50),
    }


def _fraction_string(value: object, field: str) -> Q:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact rational string")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} is not a valid rational string") from error


def _cover_json(result: CoverResult) -> dict[str, object]:
    return {
        "certified": result.certified,
        "splits": result.splits,
        "certified_leaves": result.certified_leaves,
        "excluded_leaves": result.excluded_leaves,
        "maximum_depth": result.maximum_depth,
        "peak_stack_size": result.peak_stack_size,
        "minimum_certified_lower_bound": (
            _exact(result.minimum_lower_bound)
            if result.minimum_lower_bound is not None
            else None
        ),
        "failed_box": _box_json(result.failed_box) if result.failed_box else None,
        "failed_lower_bound": (
            _exact(result.failed_lower_bound)
            if result.failed_lower_bound is not None
            else None
        ),
    }


def _derivative_json(result: DerivativeCoverResult) -> dict[str, object]:
    return {
        "certified": result.certified,
        "splits": result.splits,
        "leaves": result.leaves,
        "maximum_depth": result.maximum_depth,
        "peak_stack_size": result.peak_stack_size,
        "minimum_margins": {
            name: _exact(value) for name, value in result.minimum_margins.items()
        },
        "failed_box": _box_json(result.failed_box) if result.failed_box else None,
        "failed_margins": (
            {name: _exact(value) for name, value in result.failed_margins.items()}
            if result.failed_margins
            else None
        ),
    }


def build_certificate(
    kkt_certificate_path: Path,
    *,
    gap: Q = Q(1, 10),
    max_splits: int = 1_000_000,
) -> dict[str, object]:
    if gap <= 0:
        raise ValueError("gap must be positive")
    if max_splits < 0:
        raise ValueError("max_splits must be nonnegative")
    kkt = json.loads(kkt_certificate_path.read_text(encoding="utf-8"))
    if type(kkt) is not dict:
        raise ValueError("the prerequisite KKT certificate must be a JSON object")
    if not (
        kkt.get("status") == "VERIFIED"
        and kkt.get("strictly_inside") is True
        and kkt.get("weights_positive") is True
    ):
        raise ValueError("the prerequisite KKT certificate is not verified")
    center = kkt.get("center")
    if type(center) is not dict:
        raise ValueError("the prerequisite KKT center must be a JSON object")
    radius = _fraction_string(kkt.get("radius"), "radius")
    a, b, c, level = (
        _fraction_string(center.get(name), f"center.{name}")
        for name in ("a", "b", "c", "level")
    )
    center_sources, source_boxes = symmetric_source_enclosure(a, b, c, radius)
    target = level + radius + gap

    started = perf_counter()
    derivative_results = {
        name: certify_derivative_region(
            ACTIVE_REGIONS[name],
            source_boxes,
            conditions,
            max_splits=max_splits,
        )
        for name, conditions in DERIVATIVE_CONDITIONS.items()
    }
    complement = certify_complement_gap(
        center_sources,
        source_boxes,
        radius,
        ACTIVE_REGIONS.values(),
        target,
        max_splits=max_splits,
    )
    verified = complement.certified and all(
        result.certified for result in derivative_results.values()
    )
    if not verified:
        raise RuntimeError("active-minimum isolation did not close")
    elapsed = perf_counter() - started
    try:
        prerequisite_display_path = kkt_certificate_path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        prerequisite_display_path = str(kkt_certificate_path.resolve())
    return {
        "schema_version": 1,
        "claim": (
            "Combined with the prerequisite Krawczyk certificate, the unique "
            "symmetric N=3 KKT root has exactly five global observer minima: "
            "the four corners and the bottom-edge midpoint."
        ),
        "method": (
            "exact-rational derivative interval covers plus an exact-rational "
            "adaptive complement lower-bound cover"
        ),
        "status": "VERIFIED",
        "kkt_prerequisite": {
            "path": prerequisite_display_path,
            "sha256": _sha256(kkt_certificate_path),
            "root_box_radius": str(radius),
        },
        "source_parameter_center": {name: center[name] for name in ("a", "b", "c")},
        "level_center": center["level"],
        "level_upper_bound": _exact(level + radius),
        "strict_complement_gap": _exact(gap),
        "complement_target": _exact(target),
        "active_regions": {
            name: _box_json(box) for name, box in ACTIVE_REGIONS.items()
        },
        "derivative_conditions": {
            name: list(conditions)
            for name, conditions in DERIVATIVE_CONDITIONS.items()
        },
        "derivative_covers": {
            name: _derivative_json(result)
            for name, result in derivative_results.items()
        },
        "complement_cover": _cover_json(complement),
        "symmetry_identity": (
            "For every exactly symmetric source triple in the root box, "
            "U_x(1/2,0)=0 identically."
        ),
        "logical_summary": [
            "Each corner rectangle is coordinate-monotone toward its corner, so its unique minimum is that corner.",
            "In the bottom-midpoint rectangle U_y>0, while U_xx>0 on every horizontal line; symmetry makes (1/2,0) its unique minimum.",
            "Outside those five rectangles U exceeds the KKT level by at least 1/10.",
            "The KKT equations make the four corner values and bottom-midpoint value exactly equal at the enclosed root.",
        ],
        "elapsed_seconds": elapsed,
        "scope_warning": (
            "This certifies the observer-space active set and strict isolation "
            "at the symmetric KKT root. It is not yet a proof that the root is "
            "a local or global maximum over all source configurations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument("--gap", default="1/10")
    parser.add_argument("--max-splits", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_active_minima_isolation.json",
    )
    args = parser.parse_args()
    result = build_certificate(
        args.kkt_certificate,
        gap=Q(args.gap),
        max_splits=args.max_splits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "gap": result["strict_complement_gap"]["decimal"],
                "complement_splits": result["complement_cover"]["splits"],
                "derivative_splits": {
                    name: cover["splits"]
                    for name, cover in result["derivative_covers"].items()
                },
                "elapsed_seconds": result["elapsed_seconds"],
                "scope_warning": result["scope_warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

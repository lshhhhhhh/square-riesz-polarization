"""Certify an explicit full six-dimensional active-branch neighborhood."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from scripts.certify_n3_active_minima import (
    ACTIVE_REGIONS,
    DERIVATIVE_CONDITIONS,
    _canonical_json_sha256,
)
from square_riesz.certify import Box, decimal_string
from square_riesz.local_proof import (
    certify_complement_gap,
    certify_derivative_region,
    full_source_enclosure,
    point_potential_upper_bound,
    potential_derivative_intervals,
)


def _fraction_string(value: object, field: str) -> Q:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact rational string")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} is not a valid rational string") from error


def _exact(value: Q) -> dict[str, object]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal": decimal_string(value, 50),
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _box_json(box: Box) -> list[str]:
    return list(map(str, box))


def build_certificate(
    kkt_certificate_path: Path,
    *,
    source_radius: Q,
    isolation_gap: Q = Q(1, 100),
    max_splits: int = 1_000_000,
) -> dict[str, object]:
    if source_radius <= 0 or isolation_gap <= 0 or max_splits < 0:
        raise ValueError("radius/gap must be positive and split limit nonnegative")
    kkt = json.loads(kkt_certificate_path.read_text(encoding="utf-8"))
    if not (
        type(kkt) is dict
        and kkt.get("status") == "VERIFIED"
        and kkt.get("strictly_inside") is True
    ):
        raise ValueError("KKT prerequisite is not verified")
    center = kkt.get("center")
    if type(center) is not dict:
        raise ValueError("KKT center must be a JSON object")
    a, b, c = (
        _fraction_string(center.get(name), f"center.{name}")
        for name in ("a", "b", "c")
    )
    kkt_radius = _fraction_string(kkt.get("radius"), "radius")
    if source_radius < kkt_radius:
        raise ValueError("full source radius must contain the KKT root box")
    center_sources = ((a, b), (Q(1) - a, b), (Q(1, 2), c))
    source_boxes = full_source_enclosure(center_sources, source_radius)

    corner_upper = point_potential_upper_bound((Q(0), Q(0)), source_boxes)
    complement_target = corner_upper + isolation_gap
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
    left_edge = potential_derivative_intervals(
        (Q(3, 8), Q(3, 8), Q(0), Q(0)), source_boxes
    )["ux"]
    right_edge = potential_derivative_intervals(
        (Q(5, 8), Q(5, 8), Q(0), Q(0)), source_boxes
    )["ux"]
    tangent_bracket = left_edge.upper < 0 < right_edge.lower
    complement = certify_complement_gap(
        center_sources,
        source_boxes,
        source_radius,
        ACTIVE_REGIONS.values(),
        complement_target,
        max_splits=max_splits,
    )
    verified = (
        tangent_bracket
        and complement.certified
        and all(result.certified for result in derivative_results.values())
    )
    if not verified:
        failures = {
            "derivative_regions": {
                name: {
                    "splits": result.splits,
                    "failed_box": (
                        list(map(str, result.failed_box))
                        if result.failed_box is not None
                        else None
                    ),
                    "failed_margins_decimal": (
                        {
                            condition: decimal_string(value, 20)
                            for condition, value in result.failed_margins.items()
                        }
                        if result.failed_margins is not None
                        else None
                    ),
                }
                for name, result in derivative_results.items()
                if not result.certified
            },
            "tangent_bracket": tangent_bracket,
            "left_ux_upper": decimal_string(left_edge.upper, 20),
            "right_ux_lower": decimal_string(right_edge.lower, 20),
            "complement_certified": complement.certified,
            "complement_splits": complement.splits,
            "complement_failed_lower": (
                decimal_string(complement.failed_lower_bound, 20)
                if complement.failed_lower_bound is not None
                else None
            ),
        }
        raise RuntimeError(
            "full-source active-branch neighborhood did not close: "
            + json.dumps(failures, sort_keys=True)
        )

    def derivative_result_json(result) -> dict[str, object]:
        return {
            "certified": result.certified,
            "splits": result.splits,
            "leaves": result.leaves,
            "maximum_depth": result.maximum_depth,
            "minimum_margins": {
                name: _exact(value) for name, value in result.minimum_margins.items()
            },
        }

    return {
        "schema_version": 1,
        "claim": (
            "Every source configuration in the declared full six-dimensional "
            "L-infinity box has a continuous minimum represented by the four "
            "corner branches and one unique smooth bottom-edge branch."
        ),
        "status": "VERIFIED",
        "method": (
            "exact-rational full-source derivative covers, tangential root "
            "bracketing, and complement lower-bound cover"
        ),
        "kkt_prerequisite": {
            "path": _display_path(kkt_certificate_path),
            "canonical_json_sha256": _canonical_json_sha256(
                kkt_certificate_path
            ),
            "root_box_radius": str(kkt_radius),
        },
        "full_source_linf_radius": _exact(source_radius),
        "source_box_dimension": 6,
        "active_regions": {
            name: _box_json(region) for name, region in ACTIVE_REGIONS.items()
        },
        "corner_branch_uniform_upper_bound": _exact(corner_upper),
        "isolation_gap": _exact(isolation_gap),
        "complement_target": _exact(complement_target),
        "derivative_covers": {
            name: derivative_result_json(result)
            for name, result in derivative_results.items()
        },
        "bottom_tangent_bracket": {
            "certified": tangent_bracket,
            "left_x": "3/8",
            "left_ux_upper": _exact(left_edge.upper),
            "right_x": "5/8",
            "right_ux_lower": _exact(right_edge.lower),
        },
        "complement_cover": {
            "certified": complement.certified,
            "splits": complement.splits,
            "certified_leaves": complement.certified_leaves,
            "excluded_leaves": complement.excluded_leaves,
            "maximum_depth": complement.maximum_depth,
            "minimum_certified_lower_bound": _exact(
                complement.minimum_lower_bound
            ),
        },
        "logical_summary": [
            "Each corner rectangle is strictly coordinate-monotone to its corner.",
            "On the bottom rectangle U_y>0 and U_xx>0, while endpoint U_x signs bracket one unique bottom-edge stationary point.",
            "The remaining square lies above a uniform upper bound for the bottom-left corner branch by the declared gap.",
            "Therefore the continuous minimum throughout the full source box is exactly the minimum of five smooth local branches.",
        ],
        "elapsed_seconds": perf_counter() - started,
        "scope_warning": (
            "This gives an explicit active-branch neighborhood. It does not "
            "by itself prove that the KKT root dominates every other source "
            "configuration in that box."
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
    parser.add_argument("--source-radius", default="3e-3")
    parser.add_argument("--isolation-gap", default="1/100")
    parser.add_argument("--max-splits", type=int, default=1_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_full_source_neighborhood.json",
    )
    args = parser.parse_args()
    result = build_certificate(
        args.kkt_certificate,
        source_radius=Q(args.source_radius),
        isolation_gap=Q(args.isolation_gap),
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
                "source_radius": result["full_source_linf_radius"]["decimal"],
                "corner_upper": result["corner_branch_uniform_upper_bound"][
                    "decimal"
                ],
                "complement_minimum": result["complement_cover"][
                    "minimum_certified_lower_bound"
                ]["decimal"],
                "complement_splits": result["complement_cover"]["splits"],
                "elapsed_seconds": result["elapsed_seconds"],
                "scope_warning": result["scope_warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

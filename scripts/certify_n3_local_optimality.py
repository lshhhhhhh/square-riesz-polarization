"""Certify the full six-dimensional strict local optimum for symmetric N=3."""

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

from square_riesz.certify import decimal_string
from square_riesz.exact_interval import Interval
from square_riesz.local_optimality import (
    choose_pivot_columns,
    critical_basis,
    fixed_observer_branch,
    moving_bottom_branch,
    projected_symmetric_2x2,
    weighted_sum_hessian,
)
from square_riesz.local_proof import symmetric_source_enclosure


POINTS = (
    (Q(0), Q(1)),
    (Q(1), Q(1)),
    (Q(0), Q(0)),
    (Q(1), Q(0)),
)
COORDINATE_NAMES = (
    "source_1_x",
    "source_1_y",
    "source_2_x",
    "source_2_y",
    "source_3_x",
    "source_3_y",
)


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def _fraction_string(value: object, field: str) -> Q:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact rational string")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} is not a valid rational string") from error


def _interval_json(value: Interval) -> dict[str, object]:
    return {
        "lower": str(value.lower),
        "upper": str(value.upper),
        "lower_decimal": decimal_string(value.lower, 40),
        "upper_decimal": decimal_string(value.upper, 40),
    }


def _matrix_json(matrix: list[list[Interval]]) -> list[list[dict[str, object]]]:
    return [[_interval_json(value) for value in row] for row in matrix]


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_verified(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict or payload.get("status") != "VERIFIED":
        raise ValueError(f"{label} is not a verified JSON certificate")
    return payload


def build_certificate(
    kkt_certificate_path: Path,
    active_minima_certificate_path: Path,
) -> dict[str, object]:
    kkt = _load_verified(kkt_certificate_path, "KKT prerequisite")
    active_minima = _load_verified(
        active_minima_certificate_path, "active-minimum prerequisite"
    )
    complement_cover = active_minima.get("complement_cover")
    derivative_covers = active_minima.get("derivative_covers")
    if not (
        kkt.get("strictly_inside") is True
        and kkt.get("weights_positive") is True
        and type(complement_cover) is dict
        and complement_cover.get("certified") is True
        and type(derivative_covers) is dict
        and set(derivative_covers)
        == {
            "bottom_left",
            "bottom_right",
            "top_left",
            "top_right",
            "bottom_midpoint",
        }
        and all(
            type(cover) is dict and cover.get("certified") is True
            for cover in derivative_covers.values()
        )
    ):
        raise ValueError("a prerequisite certificate is incomplete")
    active_kkt_prerequisite = active_minima.get("kkt_prerequisite")
    if type(active_kkt_prerequisite) is not dict:
        raise ValueError("active-minimum prerequisite has no KKT binding")
    expected_kkt_hash = active_kkt_prerequisite.get("canonical_json_sha256")
    if expected_kkt_hash != _canonical_json_sha256(kkt_certificate_path):
        raise ValueError("active-minimum prerequisite names a different KKT artifact")

    center = kkt.get("center")
    if type(center) is not dict:
        raise ValueError("KKT center must be a JSON object")
    names = (
        "a",
        "b",
        "c",
        "top_weight",
        "corner_weight",
        "midpoint_weight",
        "level",
    )
    center_values = {
        name: _fraction_string(center.get(name), f"center.{name}") for name in names
    }
    radius = _fraction_string(kkt.get("radius"), "radius")
    if radius <= 0:
        raise ValueError("KKT radius must be positive")
    source_centers, source_boxes = symmetric_source_enclosure(
        center_values["a"], center_values["b"], center_values["c"], radius
    )

    started = perf_counter()
    fixed_branches = [fixed_observer_branch(point, source_boxes) for point in POINTS]
    moving_gradient, moving_hessian, observer_xx = moving_bottom_branch(source_boxes)
    gradients = [gradient for gradient, _ in fixed_branches] + [moving_gradient]
    hessians = [hessian for _, hessian in fixed_branches] + [moving_hessian]
    difference_matrix = [
        [gradient[column] - moving_gradient[column] for column in range(6)]
        for gradient in gradients[:-1]
    ]

    _, center_source_boxes = symmetric_source_enclosure(
        center_values["a"], center_values["b"], center_values["c"], Q(0)
    )
    center_gradients = [
        fixed_observer_branch(point, center_source_boxes)[0] for point in POINTS
    ]
    center_moving_gradient = moving_bottom_branch(center_source_boxes)[0]
    center_difference_matrix = [
        [
            gradient[column].lower - center_moving_gradient[column].lower
            for column in range(6)
        ]
        for gradient in center_gradients
    ]
    pivot_columns = choose_pivot_columns(center_difference_matrix)
    basis, pivot_determinant = critical_basis(difference_matrix, pivot_columns)

    top_weight = Interval(
        center_values["top_weight"] - radius,
        center_values["top_weight"] + radius,
    ) / 2
    corner_weight = Interval(
        center_values["corner_weight"] - radius,
        center_values["corner_weight"] + radius,
    ) / 2
    midpoint_weight = Interval(
        center_values["midpoint_weight"] - radius,
        center_values["midpoint_weight"] + radius,
    )
    weights = (
        top_weight,
        top_weight,
        corner_weight,
        corner_weight,
        midpoint_weight,
    )
    if any(weight.lower <= 0 for weight in weights):
        raise RuntimeError("strict complementarity was not certified")
    weighted_hessian = weighted_sum_hessian(weights, hessians)
    b00, b01, b11, projected_determinant = projected_symmetric_2x2(
        basis, weighted_hessian
    )
    negative_definite = (
        b00.upper < 0 and b11.upper < 0 and projected_determinant.lower > 0
    )
    licq = not (pivot_determinant.lower <= 0 <= pivot_determinant.upper)
    if not licq or not negative_definite:
        raise RuntimeError("LICQ or the interval second-order test did not close")
    elapsed = perf_counter() - started

    free_columns = tuple(column for column in range(6) if column not in pivot_columns)
    return {
        "schema_version": 1,
        "claim": (
            "Combined with the KKT and active-observer prerequisites, the "
            "enclosed symmetric N=3 source configuration is a strict local "
            "maximizer of the continuous unit-square polarization objective "
            "in the full six-dimensional ordered source space."
        ),
        "method": (
            "exact-rational interval LICQ and second-order sufficient-condition "
            "test with the moving bottom-edge observer Schur complement"
        ),
        "status": "VERIFIED",
        "prerequisites": {
            "kkt": {
                "path": _display_path(kkt_certificate_path),
                "canonical_json_sha256": _canonical_json_sha256(
                    kkt_certificate_path
                ),
            },
            "active_minima": {
                "path": _display_path(active_minima_certificate_path),
                "canonical_json_sha256": _canonical_json_sha256(
                    active_minima_certificate_path
                ),
            },
        },
        "root_box_radius": str(radius),
        "strict_complementarity": {
            "certified": True,
            "individual_weight_intervals": [_interval_json(weight) for weight in weights],
        },
        "licq": {
            "certified": licq,
            "difference_matrix_shape": [4, 6],
            "pivot_columns": list(pivot_columns),
            "pivot_coordinate_names": [COORDINATE_NAMES[item] for item in pivot_columns],
            "free_columns": list(free_columns),
            "free_coordinate_names": [COORDINATE_NAMES[item] for item in free_columns],
            "pivot_determinant": _interval_json(pivot_determinant),
        },
        "moving_bottom_branch": {
            "observer_xx": _interval_json(observer_xx),
            "envelope_formula": "U_AA - U_Ax * U_xx^(-1) * U_xA",
        },
        "critical_basis_interval": _matrix_json(basis),
        "projected_weighted_envelope_hessian": {
            "b00": _interval_json(b00),
            "b01": _interval_json(b01),
            "b11": _interval_json(b11),
            "determinant": _interval_json(projected_determinant),
            "negative_definite": negative_definite,
            "criterion": "b00<0, b11<0, determinant>0",
        },
        "logical_summary": [
            "The Krawczyk root equations and reflection symmetry give full "
            "six-dimensional weighted stationarity and five positive "
            "multipliers summing to one.",
            "A nonzero 4x4 minor of the four active-gradient differences gives "
            "rank four; adding the epigraph t-column gives LICQ rank five.",
            "The moving bottom observer is eliminated by the implicit-function "
            "Schur complement.",
            "The weighted envelope Hessian is negative definite on the exact "
            "two-dimensional critical subspace.",
            "The active-minimum certificate makes these five smooth branches "
            "an exact local representation of the continuous minimum.",
            "Standard nonlinear-programming second-order sufficiency therefore "
            "yields a strict local maximum in ordered source space.",
        ],
        "elapsed_seconds": elapsed,
        "scope_warning": (
            "This is a strict local-optimality theorem, not a proof that the "
            "configuration is the global maximizer P_3 or that P_3 equals its "
            "value."
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
    parser.add_argument(
        "--active-minima-certificate",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_active_minima_isolation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_strict_local_optimality.json",
    )
    args = parser.parse_args()
    result = build_certificate(
        args.kkt_certificate, args.active_minima_certificate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    projected = result["projected_weighted_envelope_hessian"]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "pivot_coordinates": result["licq"]["pivot_coordinate_names"],
                "observer_xx": result["moving_bottom_branch"]["observer_xx"][
                    "lower_decimal"
                ],
                "projected_b00_upper": projected["b00"]["upper_decimal"],
                "projected_b11_upper": projected["b11"]["upper_decimal"],
                "projected_determinant_lower": projected["determinant"][
                    "lower_decimal"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "scope_warning": result["scope_warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

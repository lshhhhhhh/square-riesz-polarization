"""Certify an explicit source-space radius dominated by the symmetric N=3 root."""

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
    _canonical_json_sha256,
)
from square_riesz.certify import decimal_string
from square_riesz.exact_interval import Interval
from square_riesz.local_optimality import (
    choose_pivot_columns,
    critical_basis,
    fixed_observer_branch,
    interval_matrix_inverse,
    moving_bottom_branch,
    moving_bottom_branch_interval,
    projected_symmetric_2x2,
    weighted_sum_hessian,
)
from square_riesz.local_proof import (
    certify_derivative_region,
    full_source_enclosure,
    potential_derivative_intervals,
    symmetric_source_enclosure,
)


POINTS = (
    (Q(0), Q(1)),
    (Q(1), Q(1)),
    (Q(0), Q(0)),
    (Q(1), Q(0)),
)


def _fraction_string(value: object, field: str) -> Q:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact rational string")
    try:
        return Q(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{field} is not a valid rational string") from error


def _exact_object(value: object, field: str) -> Q:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact-value object")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not str or type(denominator) is not str:
        raise ValueError(f"{field} numerator/denominator must be strings")
    return Q(int(numerator), int(denominator))


def _abs_upper(value: Interval) -> Q:
    return max(abs(value.lower), abs(value.upper))


def _exact(value: Q) -> dict[str, object]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal": decimal_string(value, 50),
    }


def _interval(value: Interval) -> dict[str, object]:
    return {
        "lower": _exact(value.lower),
        "upper": _exact(value.upper),
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _near_margin(
    eta: Q, mu: Q, critical_graph_norm: Q, inverse_pivot_norm: Q, hessian_sum: Q
) -> Q:
    rho = inverse_pivot_norm * eta
    if rho >= 1:
        return -Q(1)
    negative = mu * ((1 - rho) / critical_graph_norm) ** 2
    cross = hessian_sum * (2 * (1 + rho) * rho + rho * rho)
    return negative - cross


def _choose_eta(
    mu: Q, critical_graph_norm: Q, inverse_pivot_norm: Q, hessian_sum: Q
) -> Q:
    lower = Q(0)
    upper = Q(1, 2) / inverse_pivot_norm
    for _ in range(48):
        midpoint = (lower + upper) / 2
        if _near_margin(
            midpoint,
            mu,
            critical_graph_norm,
            inverse_pivot_norm,
            hessian_sum,
        ) > 0:
            lower = midpoint
        else:
            upper = midpoint
    # Retain a factor-two cushion from the largest bisection-certified value.
    return lower / 2


def build_certificate(
    kkt_certificate_path: Path,
    active_neighborhood_path: Path,
    *,
    hessian_radius: Q = Q("1e-5"),
) -> dict[str, object]:
    if hessian_radius <= 0:
        raise ValueError("hessian radius must be positive")
    kkt = json.loads(kkt_certificate_path.read_text(encoding="utf-8"))
    neighborhood = json.loads(active_neighborhood_path.read_text(encoding="utf-8"))
    if not (
        type(kkt) is dict
        and kkt.get("status") == "VERIFIED"
        and kkt.get("strictly_inside") is True
        and kkt.get("weights_positive") is True
    ):
        raise ValueError("KKT prerequisite is not verified")
    if not (type(neighborhood) is dict and neighborhood.get("status") == "VERIFIED"):
        raise ValueError("active-neighborhood prerequisite is not verified")
    neighborhood_kkt = neighborhood.get("kkt_prerequisite")
    if not (
        type(neighborhood_kkt) is dict
        and neighborhood_kkt.get("canonical_json_sha256")
        == _canonical_json_sha256(kkt_certificate_path)
    ):
        raise ValueError("active neighborhood is bound to another KKT artifact")
    active_radius = _exact_object(
        neighborhood.get("full_source_linf_radius"), "full_source_linf_radius"
    )
    if hessian_radius > active_radius:
        raise ValueError("hessian box leaves the certified active neighborhood")

    center = kkt.get("center")
    if type(center) is not dict:
        raise ValueError("KKT center must be a JSON object")
    values = {
        name: _fraction_string(center.get(name), f"center.{name}")
        for name in (
            "a",
            "b",
            "c",
            "top_weight",
            "corner_weight",
            "midpoint_weight",
        )
    }
    root_radius = _fraction_string(kkt.get("radius"), "radius")
    if hessian_radius <= root_radius:
        raise ValueError("hessian radius must strictly contain the root box")
    center_sources = (
        (values["a"], values["b"]),
        (Q(1) - values["a"], values["b"]),
        (Q(1, 2), values["c"]),
    )
    _, root_source_boxes = symmetric_source_enclosure(
        values["a"], values["b"], values["c"], root_radius
    )
    neighborhood_source_boxes = full_source_enclosure(
        center_sources, hessian_radius
    )
    started = perf_counter()

    root_fixed = [fixed_observer_branch(point, root_source_boxes) for point in POINTS]
    root_moving_gradient, _, _ = moving_bottom_branch(root_source_boxes)
    root_gradients = [gradient for gradient, _ in root_fixed]
    difference_matrix = [
        [gradient[column] - root_moving_gradient[column] for column in range(6)]
        for gradient in root_gradients
    ]
    _, center_source_boxes = symmetric_source_enclosure(
        values["a"], values["b"], values["c"], Q(0)
    )
    center_gradients = [
        fixed_observer_branch(point, center_source_boxes)[0] for point in POINTS
    ]
    center_moving_gradient = moving_bottom_branch(center_source_boxes)[0]
    center_difference = [
        [
            gradient[column].lower - center_moving_gradient[column].lower
            for column in range(6)
        ]
        for gradient in center_gradients
    ]
    pivot_columns = choose_pivot_columns(center_difference)
    basis, _ = critical_basis(difference_matrix, pivot_columns)
    pivot_matrix = [
        [row[column] for column in pivot_columns] for row in difference_matrix
    ]
    inverse_pivot, pivot_determinant = interval_matrix_inverse(pivot_matrix)
    inverse_pivot_norm = max(
        sum((_abs_upper(value) for value in row), start=Q(0))
        for row in inverse_pivot
    )
    critical_graph_norm = max(
        sum((_abs_upper(value) for value in row), start=Q(0)) for row in basis
    )

    bottom_curvature = certify_derivative_region(
        ACTIVE_REGIONS["bottom_midpoint"],
        neighborhood_source_boxes,
        ("uxx_positive",),
    )
    if not bottom_curvature.certified:
        raise RuntimeError("bottom curvature did not close on the Hessian box")
    curvature_lower = bottom_curvature.minimum_margins["uxx_positive"]
    midpoint_ux = potential_derivative_intervals(
        (Q(1, 2), Q(1, 2), Q(0), Q(0)), neighborhood_source_boxes
    )["ux"]
    midpoint_ux_magnitude = _abs_upper(midpoint_ux)
    observer_shift = midpoint_ux_magnitude / curvature_lower
    if observer_shift >= Q(1, 8):
        raise RuntimeError("moving bottom observer escaped its active rectangle")
    observer_x = Interval(Q(1, 2) - observer_shift, Q(1, 2) + observer_shift)

    neighborhood_fixed = [
        fixed_observer_branch(point, neighborhood_source_boxes) for point in POINTS
    ]
    _, neighborhood_moving_hessian, observer_xx = moving_bottom_branch_interval(
        observer_x, neighborhood_source_boxes
    )
    hessians = [hessian for _, hessian in neighborhood_fixed] + [
        neighborhood_moving_hessian
    ]
    top_weight = Interval(
        values["top_weight"] - root_radius,
        values["top_weight"] + root_radius,
    ) / 2
    corner_weight = Interval(
        values["corner_weight"] - root_radius,
        values["corner_weight"] + root_radius,
    ) / 2
    midpoint_weight = Interval(
        values["midpoint_weight"] - root_radius,
        values["midpoint_weight"] + root_radius,
    )
    weights = (
        top_weight,
        top_weight,
        corner_weight,
        corner_weight,
        midpoint_weight,
    )
    lambda_min = min(weight.lower for weight in weights)
    weighted_hessian = weighted_sum_hessian(weights, hessians)
    b00, b01, b11, projected_determinant = projected_symmetric_2x2(
        basis, weighted_hessian
    )
    off_diagonal = _abs_upper(b01)
    mu = min(-b00.upper - off_diagonal, -b11.upper - off_diagonal)
    if mu <= 0 or projected_determinant.lower <= 0:
        raise RuntimeError(
            "uniform critical Hessian negativity did not close: "
            + json.dumps(
                {
                    "observer_shift": decimal_string(observer_shift, 20),
                    "b00_upper": decimal_string(b00.upper, 20),
                    "b01_abs_upper": decimal_string(off_diagonal, 20),
                    "b11_upper": decimal_string(b11.upper, 20),
                    "determinant_lower": decimal_string(
                        projected_determinant.lower, 20
                    ),
                    "mu": decimal_string(mu, 20),
                },
                sort_keys=True,
            )
        )

    hessian_sum = sum(
        (_abs_upper(value) for row in weighted_hessian for value in row),
        start=Q(0),
    )
    individual_hessian_bound = max(
        sum((_abs_upper(value) for row in hessian for value in row), start=Q(0))
        for hessian in hessians
    )
    eta = _choose_eta(
        mu,
        critical_graph_norm,
        inverse_pivot_norm,
        hessian_sum,
    )
    near_margin = _near_margin(
        eta,
        mu,
        critical_graph_norm,
        inverse_pivot_norm,
        hessian_sum,
    )
    if eta <= 0 or near_margin <= 0:
        raise RuntimeError("near-critical cone inequality did not close")
    # The linear case only requires r < 2*lambda_min*eta/M.  Use half of
    # that threshold, and reserve the Krawczyk radius when recentering the box.
    root_centered_cap_radius = min(
        hessian_radius,
        lambda_min * eta / individual_hessian_bound,
    )
    center_box_radius = root_centered_cap_radius - root_radius
    if center_box_radius <= 0:
        raise RuntimeError("quantitative cap is smaller than the root enclosure")

    return {
        "schema_version": 1,
        "claim": (
            "Every non-root source configuration within the declared full "
            "six-dimensional center box has polarization strictly below the "
            "enclosed symmetric KKT root value."
        ),
        "status": "VERIFIED",
        "method": (
            "exact-rational two-cone Taylor bound: active-gradient linear "
            "separation away from the critical space and uniform negative "
            "weighted envelope curvature near it"
        ),
        "prerequisites": {
            "kkt": {
                "path": _display_path(kkt_certificate_path),
                "canonical_json_sha256": _canonical_json_sha256(
                    kkt_certificate_path
                ),
            },
            "active_neighborhood": {
                "path": _display_path(active_neighborhood_path),
                "canonical_json_sha256": _canonical_json_sha256(
                    active_neighborhood_path
                ),
                "radius": _exact(active_radius),
            },
        },
        "hessian_enclosure_radius": _exact(hessian_radius),
        "moving_observer": {
            "midpoint_ux": _interval(midpoint_ux),
            "bottom_curvature_lower": _exact(curvature_lower),
            "observer_shift_bound": _exact(observer_shift),
            "observer_xx": _interval(observer_xx),
        },
        "critical_coordinates": {
            "pivot_columns": list(pivot_columns),
            "pivot_determinant": _interval(pivot_determinant),
            "inverse_pivot_infinity_norm_bound": _exact(inverse_pivot_norm),
            "critical_graph_infinity_norm_bound": _exact(critical_graph_norm),
        },
        "uniform_hessian_bounds": {
            "projected_b00": _interval(b00),
            "projected_b01": _interval(b01),
            "projected_b11": _interval(b11),
            "projected_determinant": _interval(projected_determinant),
            "critical_negative_margin_mu": _exact(mu),
            "weighted_entrywise_sum": _exact(hessian_sum),
            "maximum_individual_entrywise_sum": _exact(
                individual_hessian_bound
            ),
        },
        "two_cone_constants": {
            "minimum_multiplier": _exact(lambda_min),
            "gradient_difference_threshold_eta": _exact(eta),
            "near_critical_negative_margin": _exact(near_margin),
            "linear_case_strict_threshold": _exact(
                2 * lambda_min * eta / individual_hessian_bound
            ),
        },
        "root_centered_cap_radius": _exact(root_centered_cap_radius),
        "declared_center_box_linf_radius": _exact(center_box_radius),
        "logical_summary": [
            "If an active-gradient difference is large, one branch decreases linearly and dominates every quadratic Taylor remainder.",
            "If all differences are small, pivot elimination places the direction close to the critical graph, where the weighted envelope Hessian is uniformly negative.",
            "The two cases cover every nonzero source displacement in the declared box.",
        ],
        "elapsed_seconds": perf_counter() - started,
        "scope_warning": (
            "This is a quantitative local cap for global branch-and-bound. "
            "It still does not exclude better configurations outside the box."
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
        "--active-neighborhood",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_full_source_neighborhood.json",
    )
    parser.add_argument("--hessian-radius", default="1e-5")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_quantitative_local_cap.json",
    )
    args = parser.parse_args()
    result = build_certificate(
        args.kkt_certificate,
        args.active_neighborhood,
        hessian_radius=Q(args.hessian_radius),
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
                "hessian_radius": result["hessian_enclosure_radius"]["decimal"],
                "observer_shift": result["moving_observer"][
                    "observer_shift_bound"
                ]["decimal"],
                "eta": result["two_cone_constants"][
                    "gradient_difference_threshold_eta"
                ]["decimal"],
                "center_box_radius": result["declared_center_box_linf_radius"][
                    "decimal"
                ],
                "elapsed_seconds": result["elapsed_seconds"],
                "scope_warning": result["scope_warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

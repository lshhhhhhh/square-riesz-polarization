"""Numerical ROI analysis for a structured quantitative N=3 local cap.

This script separates source directions into the two-dimensional critical
space and its four-dimensional orthogonal complement at the certified KKT
center.  It estimates two constants used by a future exact proof:

* the first-order descent inradius away from the critical space; and
* a block spectral bound for the weighted Hessian near the critical space.

All calculations here are float64 diagnostics.  They do not extend the
certified local radius and are not a proof.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from square_riesz.local_optimality import (
    fixed_observer_branch,
    moving_bottom_branch,
    moving_bottom_branch_interval,
    weighted_sum_hessian,
)
from square_riesz.exact_interval import Interval
from square_riesz.local_proof import (
    potential_derivative_intervals,
    symmetric_source_enclosure,
)


POINTS = ((Q(0), Q(1)), (Q(1), Q(1)), (Q(0), Q(0)), (Q(1), Q(0)))


def _point_vector(values: list[object]) -> NDArray[np.float64]:
    return np.asarray([float(value.lower) for value in values], dtype=np.float64)


def _point_matrix(values: list[list[object]]) -> NDArray[np.float64]:
    return np.asarray(
        [[float(value.lower) for value in row] for row in values],
        dtype=np.float64,
    )


def _interval_midpoint_radius_matrix(
    values: list[list[Interval]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower = np.asarray(
        [[float(value.lower) for value in row] for row in values],
        dtype=np.float64,
    )
    upper = np.asarray(
        [[float(value.upper) for value in row] for row in values],
        dtype=np.float64,
    )
    return (lower + upper) / 2.0, (upper - lower) / 2.0


def _simplex_origin_inradius(points: NDArray[np.float64]) -> tuple[float, list[float]]:
    """Return the origin-centered inradius of a 4-simplex in R^4."""

    if points.shape != (5, 4):
        raise ValueError("expected five projected gradients in four dimensions")
    distances = []
    for omitted in range(5):
        facet = np.delete(points, omitted, axis=0)
        differences = facet[1:] - facet[0]
        _, _, right = np.linalg.svd(differences)
        normal = right[-1]
        normal /= np.linalg.norm(normal)
        distances.append(abs(float(normal @ facet[0])))
    return min(distances), distances


def _s_lemma_cone_upper_bound(
    hessian: NDArray[np.float64],
    transverse_projector: NDArray[np.float64],
    transverse_ratio: float,
) -> tuple[float, float]:
    """Bound ``u.T H u`` for unit ``u`` with transverse norm <= ratio.

    For every multiplier ``mu >= 0``, the bound is

    ``lambda_max(H - mu P) + mu * ratio**2``.

    Minimizing over ``mu`` is the one-quadratic-constraint S-lemma dual.  This
    float64 routine is only an ROI diagnostic; an exact certificate would use
    a stored rational multiplier and an interval negative-definiteness test.
    """

    ratio_squared = transverse_ratio * transverse_ratio

    def at_log_multiplier(log_multiplier: float) -> float:
        multiplier = float(np.exp(log_multiplier))
        maximum = float(
            np.linalg.eigvalsh(hessian - multiplier * transverse_projector)[-1]
        )
        return maximum + multiplier * ratio_squared

    optimized = minimize_scalar(
        at_log_multiplier,
        bounds=(-30.0, 30.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    zero_multiplier_bound = float(np.linalg.eigvalsh(hessian)[-1])
    if zero_multiplier_bound <= optimized.fun:
        return zero_multiplier_bound, 0.0
    multiplier = float(np.exp(optimized.x))
    return float(optimized.fun), multiplier


def _radius_interval_diagnostic(
    centers: tuple[tuple[Interval, Interval], ...],
    weights: NDArray[np.float64],
    transverse_projector: NDArray[np.float64],
    descent_inradius: float,
    radius: Q,
) -> dict[str, object]:
    source_boxes = tuple(
        (
            Interval(source_x.lower - radius, source_x.upper + radius),
            Interval(source_y.lower - radius, source_y.upper + radius),
        )
        for source_x, source_y in centers
    )
    fixed = [fixed_observer_branch(point, source_boxes) for point in POINTS]
    midpoint_derivative = potential_derivative_intervals(
        (Q(1, 2), Q(1, 2), Q(0), Q(0)), source_boxes
    )["ux"]
    observer_shift = max(
        abs(midpoint_derivative.lower), abs(midpoint_derivative.upper)
    ) / 3
    observer = Interval(Q(1, 2) - observer_shift, Q(1, 2) + observer_shift)
    try:
        _, moving_hessian, _ = moving_bottom_branch_interval(observer, source_boxes)
    except ValueError as error:
        return {
            "linf_radius": float(radius),
            "status": "full_cube_interval_failed",
            "bottom_observer_shift_bound": float(observer_shift),
            "reason": str(error),
        }
    hessians = [hessian for _, hessian in fixed] + [moving_hessian]
    interval_weights = tuple(Interval.point(Q(str(value))) for value in weights)
    weighted = weighted_sum_hessian(interval_weights, hessians)
    midpoint, entry_radii = _interval_midpoint_radius_matrix(weighted)
    spectral_error = float(np.max(np.sum(entry_radii, axis=1)))

    def robust_cone_bound(ratio: float) -> tuple[float, float]:
        bound, multiplier = _s_lemma_cone_upper_bound(
            midpoint, transverse_projector, ratio
        )
        return bound + spectral_error, multiplier

    lower_ratio = 0.0
    upper_ratio = 1.0
    for _ in range(60):
        middle = (lower_ratio + upper_ratio) / 2.0
        bound, _ = robust_cone_bound(middle)
        if bound < 0.0:
            lower_ratio = middle
        else:
            upper_ratio = middle

    individual_maximum = -np.inf
    for hessian in hessians:
        hessian_midpoint, hessian_radii = _interval_midpoint_radius_matrix(hessian)
        hessian_error = float(np.max(np.sum(hessian_radii, axis=1)))
        individual_maximum = max(
            individual_maximum,
            float(np.linalg.eigvalsh(hessian_midpoint)[-1]) + hessian_error,
        )
    safe_ratio = lower_ratio * (1.0 - 1.0e-8)
    robust_bound, multiplier = robust_cone_bound(safe_ratio)
    euclidean_far_capacity = (
        2.0 * descent_inradius * safe_ratio / individual_maximum
        if individual_maximum > 0.0
        else float("inf")
    )
    linf_far_capacity = euclidean_far_capacity / np.sqrt(6.0)
    return {
        "linf_radius": float(radius),
        "bottom_observer_shift_bound": float(observer_shift),
        "weighted_hessian_interval_spectral_error": spectral_error,
        "near_cone_transverse_ratio": lower_ratio,
        "near_cone_safe_bound": robust_bound,
        "near_cone_multiplier": multiplier,
        "individual_hessian_robust_maximum": individual_maximum,
        "far_cone_linf_radius_capacity": linf_far_capacity,
        "combined_root_constant_test_passes": bool(
            float(radius) < linf_far_capacity
        ),
    }


def analyze(kkt_path: Path) -> dict[str, object]:
    kkt = json.loads(kkt_path.read_text(encoding="utf-8"))
    if kkt.get("status") != "VERIFIED":
        raise ValueError("KKT artifact is not verified")
    center = kkt["center"]
    a, b, c = (Q(center[name]) for name in ("a", "b", "c"))
    _, source_boxes = symmetric_source_enclosure(a, b, c, Q(0))

    fixed = [fixed_observer_branch(point, source_boxes) for point in POINTS]
    moving_gradient, moving_hessian, _ = moving_bottom_branch(source_boxes)
    gradients = np.stack(
        [_point_vector(gradient) for gradient, _ in fixed]
        + [_point_vector(moving_gradient)]
    )
    hessians = np.stack(
        [_point_matrix(hessian) for _, hessian in fixed]
        + [_point_matrix(moving_hessian)]
    )
    weights = np.asarray(
        [
            float(Q(center["top_weight"]) / 2),
            float(Q(center["top_weight"]) / 2),
            float(Q(center["corner_weight"]) / 2),
            float(Q(center["corner_weight"]) / 2),
            float(Q(center["midpoint_weight"])),
        ],
        dtype=np.float64,
    )
    stationarity_residual = weights @ gradients

    _, singular_values, right = np.linalg.svd(gradients, full_matrices=True)
    row_basis = right[:4].T
    critical_basis = right[4:].T
    projected_gradients = gradients @ row_basis
    descent_inradius, facet_distances = _simplex_origin_inradius(
        projected_gradients
    )

    weighted_hessian = np.tensordot(weights, hessians, axes=(0, 0))
    critical_block = critical_basis.T @ weighted_hessian @ critical_basis
    cross_block = critical_basis.T @ weighted_hessian @ row_basis
    transverse_block = row_basis.T @ weighted_hessian @ row_basis
    critical_maximum = float(np.linalg.eigvalsh(critical_block)[-1])
    cross_norm = float(np.linalg.svd(cross_block, compute_uv=False)[0])
    transverse_maximum = float(np.linalg.eigvalsh(transverse_block)[-1])
    individual_curvature_maximum = float(
        max(np.linalg.eigvalsh(hessian)[-1] for hessian in hessians)
    )

    rho_grid = np.linspace(0.0, 1.0, 1_000_001)
    z_norm = np.sqrt(np.maximum(0.0, 1.0 - rho_grid * rho_grid))
    near_quadratic_upper = (
        critical_maximum * z_norm * z_norm
        + 2.0 * cross_norm * z_norm * rho_grid
        + transverse_maximum * rho_grid * rho_grid
    )
    failing = np.flatnonzero(near_quadratic_upper >= 0.0)
    near_rho = float(rho_grid[failing[0]]) if len(failing) else 1.0
    safe_rho = max(0.0, near_rho - 1.0e-6)
    far_radius_at_root = (
        2.0 * descent_inradius * safe_rho / individual_curvature_maximum
        if individual_curvature_maximum > 0.0
        else float("inf")
    )

    transverse_projector = row_basis @ row_basis.T
    lower_ratio = 0.0
    upper_ratio = 1.0
    for _ in range(60):
        middle_ratio = (lower_ratio + upper_ratio) / 2.0
        cone_bound, _ = _s_lemma_cone_upper_bound(
            weighted_hessian, transverse_projector, middle_ratio
        )
        if cone_bound < 0.0:
            lower_ratio = middle_ratio
        else:
            upper_ratio = middle_ratio
    s_lemma_ratio = lower_ratio
    s_lemma_bound, s_lemma_multiplier = _s_lemma_cone_upper_bound(
        weighted_hessian, transverse_projector, s_lemma_ratio * (1.0 - 1.0e-8)
    )
    s_lemma_far_radius = (
        2.0
        * descent_inradius
        * s_lemma_ratio
        * (1.0 - 1.0e-8)
        / individual_curvature_maximum
    )
    difference_matrix = gradients[:4] - gradients[4]
    gradient_metric = difference_matrix.T @ difference_matrix
    lower_difference_ratio = 0.0
    upper_difference_ratio = 100.0
    for _ in range(70):
        middle_ratio = (lower_difference_ratio + upper_difference_ratio) / 2.0
        cone_bound, _ = _s_lemma_cone_upper_bound(
            weighted_hessian, gradient_metric, middle_ratio
        )
        if cone_bound < 0.0:
            lower_difference_ratio = middle_ratio
        else:
            upper_difference_ratio = middle_ratio
    difference_ratio = lower_difference_ratio
    difference_bound, difference_multiplier = _s_lemma_cone_upper_bound(
        weighted_hessian,
        gradient_metric,
        difference_ratio * (1.0 - 1.0e-8),
    )
    difference_tradeoff = []
    for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85):
        bound, multiplier = _s_lemma_cone_upper_bound(
            weighted_hessian, gradient_metric, ratio
        )
        difference_tradeoff.append(
            {"ratio": ratio, "root_upper": bound, "multiplier": multiplier}
        )
    radius_diagnostics = [
        _radius_interval_diagnostic(
            source_boxes,
            weights,
            transverse_projector,
            descent_inradius,
            radius,
        )
        for radius in (Q(1, 10_000), Q(1, 2_000), Q(1, 1_000), Q(3, 1_000))
    ]

    return {
        "schema_version": 1,
        "rigor": "not_a_proof_float64_roi_analysis",
        "kkt_path": str(kkt_path),
        "gradient_singular_values": singular_values.tolist(),
        "weighted_stationarity_residual_linf": float(
            np.max(np.abs(stationarity_residual))
        ),
        "transverse_first_order_inradius": descent_inradius,
        "transverse_simplex_facet_distances": facet_distances,
        "critical_weighted_hessian_eigenvalues": np.linalg.eigvalsh(
            critical_block
        ).tolist(),
        "critical_block_maximum": critical_maximum,
        "critical_transverse_cross_norm": cross_norm,
        "transverse_block_maximum": transverse_maximum,
        "individual_branch_curvature_maximum": individual_curvature_maximum,
        "near_cone_transverse_ratio_root_bound": near_rho,
        "far_cone_radius_root_hessian_only": far_radius_at_root,
        "s_lemma_near_cone_transverse_ratio_root_bound": s_lemma_ratio,
        "s_lemma_boundary_diagnostic_upper": s_lemma_bound,
        "s_lemma_boundary_multiplier": s_lemma_multiplier,
        "s_lemma_far_cone_radius_root_hessian_only": s_lemma_far_radius,
        "s_lemma_far_cone_linf_radius_root_hessian_only": (
            s_lemma_far_radius / np.sqrt(6.0)
        ),
        "gradient_difference_cone_ratio_root_bound": difference_ratio,
        "gradient_difference_cone_boundary_upper": difference_bound,
        "gradient_difference_cone_multiplier": difference_multiplier,
        "gradient_difference_cone_tradeoff": difference_tradeoff,
        "full_cube_interval_diagnostics": radius_diagnostics,
        "interpretation": (
            "The near-cone ratio uses only the KKT-center weighted Hessian. "
            "The far-cone radius uses only KKT-center individual Hessians. "
            "A proof must add exact interval variation over the claimed radius."
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.kkt_certificate)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)


if __name__ == "__main__":
    main()

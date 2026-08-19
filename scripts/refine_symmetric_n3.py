"""High-precision KKT refinement for the reflection-symmetric N=3 family.

The family has sources ``(a,b)``, ``(1-a,b)``, and ``(1/2,c)``.  At the
candidate optimum, the top corners, bottom corners, and bottom-edge midpoint
have equal potential.  The remaining KKT equations determine the stationary
max-min point inside this three-parameter family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mpmath as mp
import numpy as np

from square_riesz.evaluate import evaluate_continuous
from square_riesz.potential import hessian_at_point


def kkt_system(a, b, c, top_weight, corner_weight, midpoint_weight, level):
    """Return the three active constraints and four KKT equations."""

    quarter = mp.mpf("0.25")
    half = mp.mpf("0.5")

    top_left = a * a + (1 - b) ** 2
    top_right = (1 - a) ** 2 + (1 - b) ** 2
    top_axis = quarter + (1 - c) ** 2
    top = 1 / top_left + 1 / top_right + 1 / top_axis
    top_gradient = (
        -2 * a / top_left**2 + 2 * (1 - a) / top_right**2,
        2 * (1 - b) * (1 / top_left**2 + 1 / top_right**2),
        2 * (1 - c) / top_axis**2,
    )

    bottom_left = a * a + b * b
    bottom_right = (1 - a) ** 2 + b * b
    bottom_axis = quarter + c * c
    corner = 1 / bottom_left + 1 / bottom_right + 1 / bottom_axis
    corner_gradient = (
        -2 * a / bottom_left**2 + 2 * (1 - a) / bottom_right**2,
        -2 * b * (1 / bottom_left**2 + 1 / bottom_right**2),
        -2 * c / bottom_axis**2,
    )

    bottom_midpoint_distance = (half - a) ** 2 + b * b
    midpoint = 2 / bottom_midpoint_distance + 1 / c**2
    midpoint_gradient = (
        4 * (half - a) / bottom_midpoint_distance**2,
        -4 * b / bottom_midpoint_distance**2,
        -2 / c**3,
    )

    stationarity = tuple(
        top_weight * top_gradient[index]
        + corner_weight * corner_gradient[index]
        + midpoint_weight * midpoint_gradient[index]
        for index in range(3)
    )
    return (
        top - level,
        corner - level,
        midpoint - level,
        *stationarity,
        top_weight + corner_weight + midpoint_weight - 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--digits",
        type=int,
        default=19,
        help="significant digits retained in each nontrivial coordinate",
    )
    parser.add_argument("--precision", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.digits < 12:
        raise SystemExit("--digits must be at least 12")
    if args.precision < args.digits + 30:
        raise SystemExit("--precision must exceed --digits by at least 30")

    mp.mp.dps = args.precision
    initial = tuple(
        mp.mpf(value)
        for value in (
            "0.11982679",
            "0.40231923",
            "0.98019746",
            "0.7137",
            "0.1999",
            "0.0864",
            "7.56838964",
        )
    )
    root = mp.findroot(
        kkt_system,
        initial,
        solver="mdnewton",
        tol=mp.mpf(10) ** (-(args.precision - 20)),
        verify=True,
    )
    a, b, c, top_weight, corner_weight, midpoint_weight, level = root

    def rounded(value) -> str:
        return mp.nstr(value, args.digits, strip_zeros=False)

    coordinate_strings = [
        [rounded(a), rounded(b)],
        [rounded(1 - a), rounded(b)],
        ["0.5", rounded(c)],
    ]
    coordinates = np.asarray(coordinate_strings, dtype=np.float64)
    evaluation = evaluate_continuous(
        coordinates,
        grid_size=257,
        low_seed_count=96,
        active_tolerance=1e-9,
    )
    residuals = kkt_system(*root)
    active_points = np.array(
        [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]
    )
    active_weights = np.array(
        [
            float(top_weight) / 2,
            float(top_weight) / 2,
            float(corner_weight) / 2,
            float(corner_weight) / 2,
            float(midpoint_weight),
        ]
    )
    source_diagnostics = []
    full_weighted_hessian = np.zeros((6, 6))
    for source in coordinates:
        delta = active_points - source
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        gradient = np.sum(
            active_weights[:, None]
            * 2.0
            * delta
            / np.square(squared_distance)[:, None],
            axis=0,
        )
        weighted_hessian = np.sum(
            active_weights[:, None, None]
            * np.stack(
                [hessian_at_point(source, [point]) for point in active_points]
            ),
            axis=0,
        )
        source_diagnostics.append(
            {
                "gradient": gradient.tolist(),
                "weighted_hessian_eigenvalues": np.linalg.eigvalsh(
                    weighted_hessian
                ).tolist(),
            }
        )
        source_index = len(source_diagnostics) - 1
        coordinate_slice = slice(2 * source_index, 2 * source_index + 2)
        full_weighted_hessian[coordinate_slice, coordinate_slice] = weighted_hessian

    active_constraint_gradients = []
    for point in active_points:
        pieces = []
        for source in coordinates:
            delta = point - source
            squared_distance = float(np.dot(delta, delta))
            pieces.extend((2.0 * delta / squared_distance**2).tolist())
        active_constraint_gradients.append(pieces)
    active_constraint_gradients = np.asarray(active_constraint_gradients)
    _, singular_values, right_vectors = np.linalg.svd(
        active_constraint_gradients, full_matrices=True
    )
    rank_tolerance = singular_values[0] * 1e-12
    constraint_rank = int(np.sum(singular_values > rank_tolerance))
    critical_basis = right_vectors[constraint_rank:].T
    projected_hessian = critical_basis.T @ full_weighted_hessian @ critical_basis

    # The bottom midpoint is not a fixed observation point: under source
    # perturbations its one-dimensional edge minimum moves.  The Hessian of
    # that value function is the fixed-point Hessian minus the usual envelope
    # (Schur-complement) correction H_Ax H_xx^{-1} H_xA.
    midpoint = active_points[-1]
    midpoint_delta = midpoint - coordinates
    midpoint_r2 = np.einsum("ij,ij->i", midpoint_delta, midpoint_delta)
    source_x_cross = np.concatenate(
        [
            2.0 * np.array([1.0, 0.0]) / radius**2
            - 8.0 * delta * delta[0] / radius**3
            for delta, radius in zip(midpoint_delta, midpoint_r2)
        ]
    )
    midpoint_observer_hessian = hessian_at_point(midpoint, coordinates)
    envelope_weighted_hessian = full_weighted_hessian.copy()
    envelope_weighted_hessian -= active_weights[-1] * np.outer(
        source_x_cross, source_x_cross
    ) / midpoint_observer_hessian[0, 0]
    projected_envelope_hessian = (
        critical_basis.T @ envelope_weighted_hessian @ critical_basis
    )
    result = {
        "schema_version": 1,
        "n": 3,
        "family": "vertical_reflection_pair_plus_axis_source",
        "coordinate_significant_digits": args.digits,
        "coordinates": coordinate_strings,
        "high_precision_parameters": {
            "a": mp.nstr(a, args.precision),
            "b": mp.nstr(b, args.precision),
            "c": mp.nstr(c, args.precision),
            "level": mp.nstr(level, args.precision),
            "active_weights": [
                mp.nstr(top_weight, args.precision),
                mp.nstr(corner_weight, args.precision),
                mp.nstr(midpoint_weight, args.precision),
            ],
            "maximum_absolute_kkt_residual": mp.nstr(
                max(abs(item) for item in residuals), args.precision
            ),
        },
        "continuous_evaluation": evaluation.to_dict(),
        "full_six_dimensional_kkt_diagnostic": {
            "active_points": active_points.tolist(),
            "active_point_weights": active_weights.tolist(),
            "per_source": source_diagnostics,
            "active_constraint_gradient_rank": constraint_rank,
            "critical_subspace_dimension": int(critical_basis.shape[1]),
            "projected_weighted_hessian_eigenvalues": np.linalg.eigvalsh(
                projected_hessian
            ).tolist(),
            "moving_edge_minimum_projected_hessian_eigenvalues": np.linalg.eigvalsh(
                projected_envelope_hessian
            ).tolist(),
            "bottom_midpoint_observer_xx_curvature": float(
                midpoint_observer_hessian[0, 0]
            ),
            "maximum_absolute_source_gradient": max(
                abs(component)
                for item in source_diagnostics
                for component in item["gradient"]
            ),
            "interpretation": (
                "zero weighted gradients establish first-order stationarity; "
                "the continuous objective's second-order test also requires "
                "the moving edge minimum's Schur-complement correction"
            ),
        },
        "rigor": "numerical_candidate_only_until_exact_certificate_is_run",
    }

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "data" / "candidates" / "n03_symmetric.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()

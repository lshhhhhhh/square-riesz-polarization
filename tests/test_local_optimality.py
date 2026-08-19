from __future__ import annotations

from fractions import Fraction as Q
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import eigvalsh

from scripts.certify_n3_local_optimality import build_certificate
from square_riesz.exact_interval import Interval
from square_riesz.local_optimality import (
    choose_pivot_columns,
    critical_basis,
    fixed_observer_branch,
    interval_matrix_inverse,
    moving_bottom_branch,
    weighted_sum_hessian,
)
from square_riesz.local_proof import symmetric_source_enclosure


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LocalOptimalityTests(unittest.TestCase):
    def test_interval_matrix_inverse_on_point_matrix(self) -> None:
        matrix = [
            [Interval.point(2), Interval.point(1)],
            [Interval.point(1), Interval.point(1)],
        ]
        inverse, determinant = interval_matrix_inverse(matrix)
        self.assertEqual((determinant.lower, determinant.upper), (Q(1), Q(1)))
        self.assertEqual(
            [[entry.lower for entry in row] for row in inverse],
            [[Q(1), Q(-1)], [Q(-1), Q(2)]],
        )
        self.assertTrue(all(entry.lower == entry.upper for row in inverse for entry in row))

    def test_replays_strict_local_optimality_certificate(self) -> None:
        result = build_certificate(
            PROJECT_ROOT
            / "data"
            / "certificates"
            / "n03_symmetric_kkt_krawczyk.json",
            PROJECT_ROOT
            / "data"
            / "certificates"
            / "n03_active_minima_isolation.json",
        )
        self.assertEqual(result["status"], "VERIFIED")
        self.assertTrue(result["licq"]["certified"])
        self.assertTrue(
            result["projected_weighted_envelope_hessian"]["negative_definite"]
        )

    def test_center_inertia_matches_independent_numerical_diagnostic(self) -> None:
        candidate = json.loads(
            (PROJECT_ROOT / "data" / "candidates" / "n03_symmetric.json").read_text(
                encoding="utf-8"
            )
        )
        parameters = candidate["high_precision_parameters"]
        a, b, c = map(Q, (parameters["a"], parameters["b"], parameters["c"]))
        _, source_boxes = symmetric_source_enclosure(a, b, c, Q(0))
        points = ((Q(0), Q(1)), (Q(1), Q(1)), (Q(0), Q(0)), (Q(1), Q(0)))
        fixed = [fixed_observer_branch(point, source_boxes) for point in points]
        moving_gradient, moving_hessian, _ = moving_bottom_branch(source_boxes)
        _, fixed_midpoint_hessian = fixed_observer_branch(
            (Q(1, 2), Q(0)), source_boxes
        )
        gradients = [item[0] for item in fixed]
        differences = [
            [gradient[column] - moving_gradient[column] for column in range(6)]
            for gradient in gradients
        ]
        center_differences = [
            [entry.lower for entry in row] for row in differences
        ]
        pivots = choose_pivot_columns(center_differences)
        basis, _ = critical_basis(differences, pivots)
        top, corner, midpoint = map(Q, parameters["active_weights"])
        weights = tuple(
            Interval.point(value)
            for value in (top / 2, top / 2, corner / 2, corner / 2, midpoint)
        )
        hessian = weighted_sum_hessian(
            weights, [item[1] for item in fixed] + [moving_hessian]
        )
        fixed_witness_hessian = weighted_sum_hessian(
            weights, [item[1] for item in fixed] + [fixed_midpoint_hessian]
        )
        z = np.asarray([[float(entry.lower) for entry in row] for row in basis])
        h = np.asarray([[float(entry.lower) for entry in row] for row in hessian])
        fixed_h = np.asarray(
            [[float(entry.lower) for entry in row] for row in fixed_witness_hessian]
        )
        metric = z.T @ z
        eigenvalues = eigvalsh(z.T @ h @ z, metric)
        fixed_witness_eigenvalues = eigvalsh(z.T @ fixed_h @ z, metric)
        expected = np.asarray(
            candidate["full_six_dimensional_kkt_diagnostic"][
                "moving_edge_minimum_projected_hessian_eigenvalues"
            ]
        )
        np.testing.assert_allclose(eigenvalues, expected, rtol=1e-10, atol=1e-10)
        self.assertGreater(fixed_witness_eigenvalues[-1], 3.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from square_riesz.evaluate import evaluate_continuous
from square_riesz.io import load_coordinates_csv
from square_riesz.potential import hessian_at_point, intensity_and_gradient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PotentialTests(unittest.TestCase):
    def test_gradient_and_hessian_against_finite_differences(self) -> None:
        sources = np.array([[0.1, 0.2], [0.8, 0.7], [0.3, 0.9]])
        point = np.array([0.41, 0.53])
        value, gradient = intensity_and_gradient(point, sources)
        hessian = hessian_at_point(point, sources)
        step = 1e-6

        numerical_gradient = np.empty(2)
        numerical_hessian = np.empty((2, 2))
        for axis in range(2):
            shift = np.zeros(2)
            shift[axis] = step
            plus_value, plus_gradient = intensity_and_gradient(point + shift, sources)
            minus_value, minus_gradient = intensity_and_gradient(point - shift, sources)
            numerical_gradient[axis] = (plus_value - minus_value) / (2.0 * step)
            numerical_hessian[:, axis] = (plus_gradient - minus_gradient) / (2.0 * step)

        self.assertTrue(np.isfinite(value))
        np.testing.assert_allclose(gradient, numerical_gradient, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(hessian, numerical_hessian, rtol=2e-8, atol=2e-8)


class ContinuousEvaluationTests(unittest.TestCase):
    def test_single_center_source_has_four_dark_corners(self) -> None:
        result = evaluate_continuous([[0.5, 0.5]], grid_size=65, low_seed_count=16)
        self.assertAlmostEqual(result.minimum, 2.0, places=12)
        dark_corners = [item for item in result.active_minima if item.location == "corner"]
        self.assertEqual(len(dark_corners), 4)

    def test_upstream_n07_numerical_witness_matches_certified_interval(self) -> None:
        path = (
            PROJECT_ROOT
            / "upstream"
            / "certificates"
            / "data"
            / "configurations"
            / "n07"
            / "coordinates.csv"
        )
        sources = load_coordinates_csv(path)
        result = evaluate_continuous(sources, grid_size=129, low_seed_count=64)
        self.assertGreaterEqual(result.minimum, 35.9098 - 2e-8)
        self.assertLessEqual(result.minimum, 35.90989421275112 + 2e-7)
        self.assertGreaterEqual(len(result.active_minima), 2)


if __name__ == "__main__":
    unittest.main()

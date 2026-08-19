from __future__ import annotations

import unittest

import numpy as np

from square_riesz.potential import intensity

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from square_riesz.torch_engine import (
        hard_minimum,
        intensity_block,
        optimize_population_softmin,
        soft_minimum,
        unit_square_grid,
    )


@unittest.skipUnless(torch is not None, "PyTorch is an optional GPU-search dependency")
class TorchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configurations = torch.tensor(
            [
                [[0.2, 0.3], [0.8, 0.6]],
                [[0.1, 0.9], [0.4, 0.4]],
            ],
            dtype=torch.float64,
        )
        self.points = torch.tensor(
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.25, 0.75]],
            dtype=torch.float64,
        )

    def test_intensity_block_matches_numpy(self) -> None:
        actual = intensity_block(self.configurations, self.points).numpy()
        expected = np.stack(
            [intensity(self.points.numpy(), item.numpy()) for item in self.configurations]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_chunked_hard_minimum_matches_full_block(self) -> None:
        expected = intensity_block(self.configurations, self.points).min(dim=1).values
        actual = hard_minimum(self.configurations, self.points, point_chunk_size=2)
        torch.testing.assert_close(actual, expected)

    def test_soft_minimum_converges_to_hard_minimum(self) -> None:
        hard = hard_minimum(self.configurations, self.points)
        soft = soft_minimum(self.configurations, self.points, 1e-5, normalize=False)
        torch.testing.assert_close(soft, hard, rtol=2e-5, atol=2e-5)

    def test_optimizer_never_forgets_a_better_initial_configuration(self) -> None:
        initial = torch.tensor(
            [[[0.5, 0.5]], [[0.1, 0.1]]], dtype=torch.float32
        )
        points = unit_square_grid(9, device="cpu")
        initial_values = hard_minimum(initial, points)
        result = optimize_population_softmin(
            initial,
            points,
            steps=3,
            learning_rate=0.5,
            point_chunk_size=32,
            report_every=1,
        )
        self.assertTrue(torch.all(result.sampled_minima >= initial_values))


if __name__ == "__main__":
    unittest.main()

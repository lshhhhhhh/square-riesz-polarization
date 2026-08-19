from __future__ import annotations

import unittest

import numpy as np

from scripts.global_upper_prototype import (
    box_witness_upper_bounds,
    initial_unordered_cell_boxes,
    split_nodes,
    witness_grid,
)


class GlobalUpperPrototypeTests(unittest.TestCase):
    def test_initial_cover_count_modulo_permutation(self) -> None:
        # Four cells and three sources: combinations with replacement C(6,3)=20.
        nodes = initial_unordered_cell_boxes(2)
        self.assertEqual(nodes.shape, (20, 3, 4))

    def test_box_witness_bound_for_fixed_center_sources(self) -> None:
        node = np.array([[[0.5, 0.5, 0.5, 0.5]] * 3])
        witnesses = np.array([[0.0, 0.0], [1.0, 1.0]])
        bounds = box_witness_upper_bounds(node, witnesses)
        self.assertAlmostEqual(bounds[0], 6.0)

    def test_split_preserves_union_and_halves_one_axis(self) -> None:
        nodes = initial_unordered_cell_boxes(1)
        children = split_nodes(nodes, 0)
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0, 0, 1], 0.5)
        self.assertEqual(children[1, 0, 0], 0.5)

    def test_witness_grid_includes_corners(self) -> None:
        points = witness_grid(3)
        self.assertTrue(any(np.array_equal(point, [0.0, 0.0]) for point in points))
        self.assertTrue(any(np.array_equal(point, [1.0, 1.0]) for point in points))


if __name__ == "__main__":
    unittest.main()

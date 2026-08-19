from __future__ import annotations

import unittest

import numpy as np

from scripts.global_upper_prototype import (
    adaptive_boundary_witness_upper_bounds,
    affine_boundary_witness_upper_bounds,
    boxes_inside_local_cap,
    boxes_closed_by_cap_outside_slabs,
    box_witness_upper_bounds,
    initial_unordered_cell_boxes,
    lookahead_split_coordinates,
    node_specific_witness_upper_bounds,
    split_nodes,
    split_nodes_by_coordinate,
    symmetric_candidate_copies,
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

    def test_per_node_split_uses_each_selected_coordinate(self) -> None:
        nodes = np.repeat(initial_unordered_cell_boxes(1), 2, axis=0)
        children = split_nodes_by_coordinate(nodes, np.asarray([0, 5]))
        self.assertEqual(children[0, 0, 1], 0.5)
        self.assertEqual(children[1, 0, 0], 0.5)
        self.assertEqual(children[2, 2, 3], 0.5)
        self.assertEqual(children[3, 2, 2], 0.5)

    def test_node_specific_witness_and_lookahead_shapes(self) -> None:
        nodes = initial_unordered_cell_boxes(2)[:3]
        witnesses = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]])
        bounds = node_specific_witness_upper_bounds(nodes, witnesses)
        coordinates = lookahead_split_coordinates(nodes, witnesses, 7.57)
        self.assertEqual(bounds.shape, (3,))
        self.assertEqual(coordinates.shape, (3,))
        self.assertTrue(np.all((0 <= coordinates) & (coordinates < 6)))

    def test_witness_grid_includes_corners(self) -> None:
        points = witness_grid(3)
        self.assertTrue(any(np.array_equal(point, [0.0, 0.0]) for point in points))
        self.assertTrue(any(np.array_equal(point, [1.0, 1.0]) for point in points))

    def test_local_cap_recognizes_permutations_and_d4_copies(self) -> None:
        copies = symmetric_candidate_copies(0.12, 0.4, 0.98)
        # The candidate itself has one reflection symmetry, so 8*6 copies
        # collapse to 24 distinct ordered configurations.
        self.assertEqual(len(copies), 24)
        center = copies[17]
        small = np.empty((1, 3, 4), dtype=np.float64)
        small[0, :, 0] = center[:, 0] - 0.001
        small[0, :, 1] = center[:, 0] + 0.001
        small[0, :, 2] = center[:, 1] - 0.001
        small[0, :, 3] = center[:, 1] + 0.001
        self.assertTrue(boxes_inside_local_cap(small, copies, 0.002)[0])
        self.assertFalse(boxes_inside_local_cap(small, copies, 0.0005)[0])

    def test_cap_outside_slab_union_handles_a_straddling_box(self) -> None:
        center = np.asarray([[[0.25, 0.5], [0.75, 0.5], [0.5, 0.8]]])
        node = np.asarray(
            [
                [
                    [0.23, 0.27, 0.48, 0.52],
                    [0.73, 0.77, 0.48, 0.52],
                    [0.48, 0.52, 0.78, 0.82],
                ]
            ]
        )
        witnesses = witness_grid(5)
        self.assertTrue(
            boxes_closed_by_cap_outside_slabs(
                node, center, 0.01, 1000.0, witnesses
            )[0]
        )
        self.assertFalse(
            boxes_closed_by_cap_outside_slabs(
                node, center, 0.01, 0.0, witnesses
            )[0]
        )

    def test_adaptive_boundary_witness_closes_fixed_candidate_at_757(self) -> None:
        a = 0.11982678825639196
        b = 0.40231923201634923
        c = 0.9801974656079522
        sources = np.asarray([[a, b], [1.0 - a, b], [0.5, c]])
        node = np.empty((1, 3, 4), dtype=np.float64)
        node[0, :, 0] = sources[:, 0]
        node[0, :, 1] = sources[:, 0]
        node[0, :, 2] = sources[:, 1]
        node[0, :, 3] = sources[:, 1]
        self.assertLess(adaptive_boundary_witness_upper_bounds(node)[0], 7.57)
        self.assertLess(affine_boundary_witness_upper_bounds(node)[0], 7.57)


if __name__ == "__main__":
    unittest.main()

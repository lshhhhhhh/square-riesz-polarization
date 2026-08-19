from __future__ import annotations

from fractions import Fraction as Q
import unittest

from scripts.certify_n3_directional_local_cap import (
    _audit_candidate_coverage,
    _direction_from_root_path,
    _fixed_observer_directional_quadratic,
    _quadratic_form,
)
from square_riesz.exact_interval import Interval
from square_riesz.local_optimality import fixed_observer_branch


class DirectionalLocalCapTests(unittest.TestCase):
    def test_root_path_reconstructs_cyclic_free_coordinate_splits(self) -> None:
        direction = _direction_from_root_path(0, "01")
        self.assertEqual((direction[0].lower, direction[0].upper), (Q(-1), Q(-1)))
        self.assertEqual((direction[1].lower, direction[1].upper), (Q(-1), Q(0)))
        self.assertEqual((direction[2].lower, direction[2].upper), (Q(0), Q(1)))

    def test_root_path_reconstructs_transverse_width_splits(self) -> None:
        sensitivities = tuple(Q(index) for index in range(1, 7))
        direction = _direction_from_root_path(
            0, "01", "transverse-width", sensitivities
        )
        self.assertEqual((direction[5].lower, direction[5].upper), (Q(-1), Q(0)))
        self.assertEqual((direction[4].lower, direction[4].upper), (Q(0), Q(1)))

    def test_adaptive_trivial_cover_passes_exact_kraft_audit(self) -> None:
        _audit_candidate_coverage(
            [{"root_id": root_id, "path": ""} for root_id in range(12)],
            "transverse-width",
            tuple(Q(index) for index in range(1, 7)),
        )

    def test_sparse_fixed_quadratic_is_contained_in_generic_form(self) -> None:
        boxes = (
            (Interval(Q(1, 4), Q(3, 10)), Interval(Q(1, 3), Q(2, 5))),
            (Interval(Q(7, 10), Q(3, 4)), Interval(Q(1, 3), Q(2, 5))),
            (Interval(Q(49, 100), Q(51, 100)), Interval(Q(1, 2), Q(3, 5))),
        )
        direction = [
            Interval(Q(-1), Q(-1, 2)),
            Interval(Q(-1, 3), Q(1, 4)),
            Interval(Q(1, 5), Q(4, 5)),
            Interval(Q(-2, 3), Q(1, 3)),
            Interval(Q(-1, 2), Q(1, 2)),
            Interval(Q(2, 5), Q(1)),
        ]
        _, hessian = fixed_observer_branch((Q(0), Q(0)), boxes)
        generic = _quadratic_form(direction, hessian)
        sparse = _fixed_observer_directional_quadratic(
            (Q(0), Q(0)), boxes, direction
        )
        self.assertGreaterEqual(sparse.lower, generic.lower)
        self.assertLessEqual(sparse.upper, generic.upper)

    def test_trivial_twelve_face_cover_passes_exact_kraft_audit(self) -> None:
        _audit_candidate_coverage(
            [{"root_id": root_id, "path": ""} for root_id in range(12)]
        )

    def test_missing_half_face_fails_exact_kraft_audit(self) -> None:
        leaves = [{"root_id": root_id, "path": ""} for root_id in range(1, 12)]
        leaves.extend(
            [
                {"root_id": 0, "path": "00"},
                {"root_id": 0, "path": "01"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "Kraft sum"):
            _audit_candidate_coverage(leaves)

    def test_prefix_collision_is_rejected(self) -> None:
        leaves = [{"root_id": root_id, "path": ""} for root_id in range(1, 12)]
        leaves.extend(
            [
                {"root_id": 0, "path": "0"},
                {"root_id": 0, "path": "00"},
                {"root_id": 0, "path": "1"},
            ]
        )
        with self.assertRaisesRegex(ValueError, "prefix-free"):
            _audit_candidate_coverage(leaves)


if __name__ == "__main__":
    unittest.main()

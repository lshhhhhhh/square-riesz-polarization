from __future__ import annotations

from fractions import Fraction as Q
import unittest

from scripts.certify_n3_directional_local_cap import (
    _audit_candidate_coverage,
    _direction_from_root_path,
)


class DirectionalLocalCapTests(unittest.TestCase):
    def test_root_path_reconstructs_cyclic_free_coordinate_splits(self) -> None:
        direction = _direction_from_root_path(0, "01")
        self.assertEqual((direction[0].lower, direction[0].upper), (Q(-1), Q(-1)))
        self.assertEqual((direction[1].lower, direction[1].upper), (Q(-1), Q(0)))
        self.assertEqual((direction[2].lower, direction[2].upper), (Q(0), Q(1)))

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

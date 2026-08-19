from __future__ import annotations

from fractions import Fraction as Q
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_global_upper_certificate import (
    _apply_path,
    _verify_prefix_cover,
    _witness_bound,
    verify,
)


class GlobalUpperCertificateTests(unittest.TestCase):
    def test_path_reconstructs_dyadic_children(self) -> None:
        boxes = _apply_path((0, 0, 0), "01", 1)
        self.assertEqual(boxes[0][0], Q(0))
        self.assertEqual(boxes[0][1], Q(1, 2))
        self.assertEqual(boxes[0][2], Q(1, 2))
        self.assertEqual(boxes[0][3], Q(1))

    def test_prefix_cover_accepts_complete_tree(self) -> None:
        _verify_prefix_cover({"0", "10", "11"})
        with self.assertRaises(ValueError):
            _verify_prefix_cover({"0", "10"})
        with self.assertRaises(ValueError):
            _verify_prefix_cover({"0", "00", "1"})

    def test_exact_witness_bound_for_three_fixed_center_sources(self) -> None:
        boxes = [[Q(1, 2), Q(1, 2), Q(1, 2), Q(1, 2)] for _ in range(3)]
        self.assertEqual(_witness_bound(boxes, 0, 3), Q(6))

    def test_verifier_rejects_coercible_but_noncanonical_fields(self) -> None:
        payload = {
            "n": 3,
            "target": "7.7",
            "initial_divisions": 1,
            "witness_grid": 3,
            "leaf_count": 1,
            "leaves": [{"root": [0, 0, 0], "path": "", "witness": 0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            for key, bad_value in (("target", 7.7), ("initial_divisions", 1.0)):
                corrupted = dict(payload)
                corrupted[key] = bad_value
                path.write_text(json.dumps(corrupted), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify(path)

            corrupted = dict(payload)
            corrupted["leaves"] = [
                {"root": [0.0, 0, 0], "path": "", "witness": 0}
            ]
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(path)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.verify_global_upper_certificate_cleanroom import _greater, verify


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CleanRoomGlobalUpperTests(unittest.TestCase):
    def test_exact_comparison_does_not_round_through_float(self) -> None:
        denominator = 10**40
        self.assertTrue(_greater((denominator + 1, denominator), (1, 1)))
        self.assertFalse(_greater((1, 1), (denominator + 1, denominator)))

    def test_replays_compact_n3_upper_tree(self) -> None:
        result = verify(
            PROJECT_ROOT / "data" / "certificates" / "n03_global_upper_7_7.json"
        )
        self.assertEqual(result["status"], "VERIFIED (clean-room)")
        self.assertEqual(result["target"], "77/10")
        self.assertEqual(result["root_count"], 816)
        self.assertEqual(result["leaf_count"], 41078)
        self.assertEqual(
            result["maximum_leaf_bound_exact"],
            "833239296/108213097",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import unittest

from scripts.record_hunt import (
    BASELINES,
    _evaluation_id,
    _job_id,
    _load_incumbent,
    _parse_n_values,
)


class RecordHuntTests(unittest.TestCase):
    def test_target_parser_and_ids(self) -> None:
        self.assertEqual(_parse_n_values("4, 5,13"), [4, 5, 13])
        self.assertEqual(_job_id(4, 2026081900), "n04_seed2026081900")
        self.assertEqual(
            _evaluation_id("n04_seed2026081900", 7),
            "n04_seed2026081900_rank007",
        )
        self.assertIn(9, BASELINES)
        self.assertGreater(BASELINES[29], 282.85)
        self.assertGreater(BASELINES[30], 285.34)
        self.assertGreater(BASELINES[31], 305.29)
        self.assertGreater(BASELINES[32], 317.20)
        self.assertGreater(BASELINES[33], 330.59)
        self.assertGreater(BASELINES[34], 337.90)
        self.assertGreater(BASELINES[35], 347.19)

    def test_loads_best_available_incumbent_coordinates(self) -> None:
        root = (
            Path(__file__).resolve().parents[1]
            / "upstream"
            / "certificates"
            / "data"
            / "configurations"
        )
        coordinates = _load_incumbent(35, root)
        self.assertEqual(len(coordinates), 35)
        self.assertTrue(all(len(point) == 2 for point in coordinates))

    def test_target_parser_rejects_unknown_or_duplicate(self) -> None:
        with self.assertRaises(Exception):
            _parse_n_values("4,4")
        with self.assertRaises(Exception):
            _parse_n_values("52")


if __name__ == "__main__":
    unittest.main()

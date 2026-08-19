from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _potential(coordinates: list[list[str]], x: Fraction, y: Fraction) -> Fraction:
    total = Fraction(0)
    for source_x_text, source_y_text in coordinates:
        source_x = Fraction(source_x_text)
        source_y = Fraction(source_y_text)
        total += 1 / ((x - source_x) ** 2 + (y - source_y) ** 2)
    return total


class N3LiteralWitnessTests(unittest.TestCase):
    def test_exact_corner_is_darker_than_truncated_bottom_midpoint(self) -> None:
        candidate = json.loads(
            (PROJECT_ROOT / "data" / "candidates" / "n03_symmetric.json").read_text(
                encoding="utf-8"
            )
        )
        coordinates = candidate["coordinates"]
        corner = _potential(coordinates, Fraction(0), Fraction(0))
        midpoint = _potential(coordinates, Fraction(1, 2), Fraction(0))
        self.assertLess(corner, midpoint)

        for filename in (
            "n03_target_7_56838963.json",
            "n03_componentwise_target_7_56838963.json",
        ):
            certificate = json.loads(
                (PROJECT_ROOT / "data" / "certificates" / filename).read_text(
                    encoding="utf-8"
                )
            )
            witness = certificate["upper_witness"]
            self.assertEqual(
                [(item["numerator"], item["denominator"]) for item in witness["point"]],
                [("0", "1"), ("0", "1")],
            )
            value = witness["value"]
            exact_value = Fraction(
                int(value["numerator"]), int(value["denominator"])
            )
            self.assertEqual(exact_value, corner)


if __name__ == "__main__":
    unittest.main()

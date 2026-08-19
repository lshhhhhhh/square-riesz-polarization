from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
import unittest

from square_riesz.certify import (
    certify_fixed_configuration,
    certify_fixed_configuration_componentwise,
    parse_decimal_points,
    potential_at,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExactCertificateTests(unittest.TestCase):
    def test_center_source_certifies_two(self) -> None:
        sources = parse_decimal_points([["0.5", "0.5"]])
        result = certify_fixed_configuration(sources, Fraction("2"), max_splits=5000)
        self.assertTrue(result.certified)
        self.assertGreaterEqual(result.minimum_leaf_lower_bound, Fraction("2"))
        self.assertEqual(potential_at((Fraction(0), Fraction(0)), sources), 2)

    def test_generic_verifier_replays_upstream_n07(self) -> None:
        path = (
            PROJECT_ROOT
            / "upstream"
            / "certificates"
            / "data"
            / "configurations"
            / "n07"
            / "coordinates.csv"
        )
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sources = parse_decimal_points([[row["x"], row["y"]] for row in rows])
        result = certify_fixed_configuration(
            sources, Fraction("35.9098"), max_splits=10_000
        )
        self.assertTrue(result.certified)
        self.assertEqual(result.splits, 499)

        componentwise = certify_fixed_configuration_componentwise(
            sources, Fraction("35.9098"), max_splits=10_000
        )
        self.assertTrue(componentwise.certified)
        self.assertEqual(componentwise.splits, 489)


if __name__ == "__main__":
    unittest.main()

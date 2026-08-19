from __future__ import annotations

from fractions import Fraction as Q
from pathlib import Path
import unittest

from scripts.certify_n3_active_minima import build_certificate
from square_riesz.certify import potential_at
from square_riesz.exact_interval import Interval
from square_riesz.local_proof import varying_source_box_lower_bound


class LocalProofTests(unittest.TestCase):
    def test_replays_n3_active_minimum_isolation(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = build_certificate(
            project_root
            / "data"
            / "certificates"
            / "n03_symmetric_kkt_krawczyk.json"
        )
        self.assertEqual(result["status"], "VERIFIED")
        self.assertEqual(result["complement_cover"]["splits"], 35)
        self.assertEqual(
            {
                name: cover["splits"]
                for name, cover in result["derivative_covers"].items()
            },
            {
                "bottom_left": 0,
                "bottom_right": 0,
                "top_left": 2,
                "top_right": 2,
                "bottom_midpoint": 1,
            },
        )

    def test_varying_source_lower_bound_contains_exact_samples(self) -> None:
        observer = (Q(0), Q(1, 16), Q(3, 4), Q(13, 16))
        centers = ((Q(1, 4), Q(1, 2)), (Q(3, 4), Q(1, 2)))
        radius = Q(1, 100)
        source_boxes = (
            (Interval(Q(6, 25), Q(13, 50)), Interval(Q(49, 100), Q(51, 100))),
            (Interval(Q(37, 50), Q(19, 25)), Interval(Q(49, 100), Q(51, 100))),
        )
        lower = varying_source_box_lower_bound(
            observer, centers, source_boxes, radius
        )
        observer_samples = (
            (observer[0], observer[2]),
            (observer[1], observer[2]),
            (observer[0], observer[3]),
            (observer[1], observer[3]),
            ((observer[0] + observer[1]) / 2, (observer[2] + observer[3]) / 2),
        )
        source_samples = (
            ((Q(6, 25), Q(49, 100)), (Q(37, 50), Q(49, 100))),
            ((Q(13, 50), Q(51, 100)), (Q(19, 25), Q(51, 100))),
            centers,
        )
        for sources in source_samples:
            for point in observer_samples:
                self.assertLessEqual(lower, potential_at(point, sources))


if __name__ == "__main__":
    unittest.main()

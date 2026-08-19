from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_friedman_submission import PROJECT_ROOT, RECORDS, build


class FriedmanSubmissionTests(unittest.TestCase):
    def test_builder_packages_all_certified_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            metadata = build(output)
            self.assertEqual(
                metadata["external_status"], "USER_REPORTED_EMAIL_SUBMITTED"
            )
            self.assertEqual(
                metadata["sent_materials_match"], "UNKNOWN_NOT_ARCHIVED"
            )
            self.assertEqual(metadata["records"], [3, 5, 29, 30, 31, 32, 33, 34, 35])
            self.assertEqual(metadata["coordinate_count"], 232)

            with (output / "records.csv").open(newline="", encoding="utf-8") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 9)
            self.assertTrue(all(row["rigorously_certified_lower_bound"] for row in records))

            with (output / "coordinates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                coordinates = list(csv.DictReader(handle))
            self.assertEqual(len(coordinates), 232)
            self.assertEqual(json.loads((output / "metadata.json").read_text())["record_count"], 9)
            reply_blocks = []
            for record in RECORDS:
                if not 29 <= record.n <= 35:
                    continue
                candidate = json.loads(
                    (PROJECT_ROOT / record.candidate).read_text(encoding="utf-8")
                )
                lines = [f"{{{x},{y}}}" for x, y in candidate["coordinates"]]
                reply_blocks.append(f"n={record.n}\n" + ",\n".join(lines))
            expected_reply = (
                "Dear Erich,\n\n"
                "Certainly. Here are the coordinates for 29 <= n <= 35 in the "
                "requested format.\n\n"
                + "\n\n".join(reply_blocks)
                + "\n\nBest regards,\n[YOUR NAME]\n"
            )
            self.assertEqual(
                (output / "ERICH_COORDINATES_29_35_REPLY.txt").read_text(
                    encoding="utf-8"
                ),
                expected_reply,
            )
            self.assertTrue((output / "SHA256SUMS.txt").read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()

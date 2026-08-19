"""Build deterministic, reviewable attachments for the Friedman record board.

This script never sends email or changes external state. It only extracts the
published literal coordinates and already-certified bounds into small CSV/JSON
artifacts that are convenient for a record-board maintainer to inspect.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "submission" / "friedman_20260819"


@dataclass(frozen=True)
class Record:
    n: int
    board_display: str
    candidate: str
    spectral: str
    componentwise: str
    prior_public_literal_upper: str = ""


RECORDS = (
    Record(
        3,
        "7.507+",
        "data/candidates/n03_symmetric.json",
        "data/certificates/n03_target_7_56838963.json",
        "data/certificates/n03_componentwise_target_7_56838963.json",
    ),
    Record(
        5,
        "21.342+",
        "data/candidates/n05_hunt_best.json",
        "data/certificates/n05_spectral_target_22_06.json",
        "data/certificates/n05_componentwise_target_22_06.json",
    ),
    Record(
        29,
        "259.948+",
        "data/candidates/n29_hunt_best.json",
        "data/certificates/n29_spectral_target_282_8.json",
        "data/certificates/n29_componentwise_target_282_8.json",
        "272.495973646473",
    ),
    Record(
        30,
        "268.319+",
        "data/candidates/n30_hunt_best.json",
        "data/certificates/n30_spectral_target_285_34.json",
        "data/certificates/n30_componentwise_target_285_34.json",
        "285.326742383550",
    ),
    Record(
        31,
        "274.727+",
        "data/candidates/n31_hunt_best.json",
        "data/certificates/n31_spectral_target_305_2.json",
        "data/certificates/n31_componentwise_target_305_2.json",
        "294.208893270426",
    ),
    Record(
        32,
        "279.851+",
        "data/candidates/n32_hunt_best.json",
        "data/certificates/n32_spectral_target_317_1.json",
        "data/certificates/n32_componentwise_target_317_1.json",
        "304.280799129879",
    ),
    Record(
        33,
        "287.670+",
        "data/candidates/n33_hunt_best.json",
        "data/certificates/n33_spectral_target_330_5.json",
        "data/certificates/n33_componentwise_target_330_5.json",
        "311.665641329667",
    ),
    Record(
        34,
        "297.004+",
        "data/candidates/n34_hunt_best.json",
        "data/certificates/n34_spectral_target_337_8.json",
        "data/certificates/n34_componentwise_target_337_8.json",
        "323.409928097309",
    ),
    Record(
        35,
        "303.346+",
        "data/candidates/n35_hunt_best.json",
        "data/certificates/n35_spectral_target_347_1.json",
        "data/certificates/n35_componentwise_target_347_1.json",
        "329.708966808636",
    ),
)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _csv_text(rows: list[list[object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _exact_value(payload: dict[str, object], field: str) -> Fraction:
    value = payload[field]
    if not isinstance(value, dict):
        raise ValueError(f"{field} is not an exact-value object")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def build(output_dir: Path, project_root: Path = PROJECT_ROOT) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_rows: list[list[object]] = [[
        "n",
        "friedman_board_display_2026-08-19",
        "reported_numerical_intensity",
        "reported_intensity_5dp",
        "rigorously_certified_lower_bound",
        "exact_point_witness_upper_bound",
        "prior_public_literal_upper_bound",
        "candidate_file",
        "spectral_certificate",
        "componentwise_certificate",
    ]]
    coordinate_rows: list[list[object]] = [["n", "source_index", "x", "y"]]
    manifest_paths: set[Path] = set()

    for record in RECORDS:
        candidate_path = project_root / record.candidate
        spectral_path = project_root / record.spectral
        componentwise_path = project_root / record.componentwise
        candidate = _load_json(candidate_path)
        spectral = _load_json(spectral_path)
        componentwise = _load_json(componentwise_path)

        if candidate.get("n") != record.n:
            raise ValueError(f"candidate N mismatch: {candidate_path}")
        coordinates = candidate.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != record.n:
            raise ValueError(f"candidate coordinate count mismatch: {candidate_path}")
        for certificate_path, certificate in (
            (spectral_path, spectral),
            (componentwise_path, componentwise),
        ):
            if certificate.get("certified") is not True:
                raise ValueError(f"certificate did not succeed: {certificate_path}")
            if certificate.get("n") != record.n:
                raise ValueError(f"certificate N mismatch: {certificate_path}")
            if certificate.get("coordinates") != coordinates:
                raise ValueError(f"certificate coordinates differ: {certificate_path}")

        spectral_target = _exact_value(spectral, "target_exact")
        componentwise_target = _exact_value(componentwise, "target_exact")
        if spectral_target != componentwise_target:
            raise ValueError(f"certificate targets differ for N={record.n}")
        spectral_upper = _exact_value(spectral["upper_witness"], "value")
        componentwise_upper = _exact_value(componentwise["upper_witness"], "value")
        if spectral_upper != componentwise_upper:
            raise ValueError(f"certificate point witnesses differ for N={record.n}")

        target_decimal = spectral["target_exact"]["decimal"]
        upper_decimal = spectral["upper_witness"]["value"]["decimal"]
        evaluation = candidate.get("continuous_evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError(f"candidate has no continuous evaluation: {candidate_path}")
        numerical_minimum = float(evaluation["minimum"])
        if not float(spectral_target) <= numerical_minimum <= float(spectral_upper):
            raise ValueError(f"numerical minimum falls outside bracket for N={record.n}")
        records_rows.append([
            record.n,
            record.board_display,
            repr(numerical_minimum),
            f"{numerical_minimum:.5f}",
            target_decimal,
            upper_decimal,
            record.prior_public_literal_upper,
            record.candidate,
            record.spectral,
            record.componentwise,
        ])
        for index, point in enumerate(coordinates, start=1):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"malformed coordinate N={record.n}, index={index}")
            coordinate_rows.append([record.n, index, point[0], point[1]])
        manifest_paths.update((candidate_path, spectral_path, componentwise_path))

    records_path = output_dir / "records.csv"
    coordinates_path = output_dir / "coordinates.csv"
    records_path.write_text(_csv_text(records_rows), encoding="utf-8", newline="\n")
    coordinates_path.write_text(
        _csv_text(coordinate_rows), encoding="utf-8", newline="\n"
    )

    metadata = {
        "schema_version": 1,
        "board_checked_date": "2026-08-19",
        "board_url": "https://erich-friedman.github.io/packing/light/",
        "submission_guidelines_url": "https://erich-friedman.github.io/packing/submit.html",
        "public_contact": "erichfriedman68@gmail.com",
        "record_count": len(RECORDS),
        "coordinate_count": len(coordinate_rows) - 1,
        "records": [record.n for record in RECORDS],
        "claim_scope": "fixed_literal_decimal_configurations_only",
        "external_status": "DRAFT_NOT_SENT",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    generated_paths = (records_path, coordinates_path, metadata_path)
    optional_package_paths = [
        output_dir / "README.md",
        output_dir / "EMAIL_DRAFT.txt",
        output_dir / "ATTACHMENT_CHECKLIST.txt",
        *(output_dir / "figures").glob("*.gif"),
    ]
    support_paths = [
        project_root / "requirements-submission.txt",
        project_root / "scripts" / "build_friedman_submission.py",
        project_root / "scripts" / "build_friedman_figures.py",
    ]
    included_paths = [
        *manifest_paths,
        *generated_paths,
        *(path for path in optional_package_paths if path.is_file()),
        *(path for path in support_paths if path.is_file()),
    ]
    manifest_lines = []
    for path in sorted(included_paths, key=lambda item: str(item)):
        if path.is_relative_to(project_root):
            relative = path.relative_to(project_root).as_posix()
        else:
            relative = f"generated/{path.relative_to(output_dir).as_posix()}"
        manifest_lines.append(f"{_sha256(path)}  {relative}")
    manifest_path = output_dir / "SHA256SUMS.txt"
    manifest_path.write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii", newline="\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))


if __name__ == "__main__":
    main()

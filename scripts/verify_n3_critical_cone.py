"""Independently recompute an exact N=3 critical-cone certificate."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_n3_hessian_lipschitz_exact import analyze
from scripts.certify_n3_active_minima import _canonical_json_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    args = parser.parse_args()

    artifact = json.loads(args.certificate.read_text(encoding="utf-8"))
    if artifact.get("status") != "CLOSED":
        raise ValueError("critical-cone artifact does not claim CLOSED")
    required_integer_fields = ("coarse_digits", "subdivisions_per_source_coordinate")
    for field in required_integer_fields:
        if type(artifact.get(field)) is not int:
            raise ValueError(f"{field} must be an integer")
    prerequisites = artifact.get("prerequisites")
    if prerequisites is not None:
        if type(prerequisites) is not dict or type(prerequisites.get("kkt")) is not dict:
            raise ValueError("KKT prerequisite metadata is malformed")
        if prerequisites["kkt"].get("canonical_json_sha256") != _canonical_json_sha256(
            args.kkt_certificate
        ):
            raise ValueError("KKT prerequisite hash mismatch")
    started = perf_counter()
    recomputed = analyze(
        args.kkt_certificate,
        Q(artifact["radius"]),
        Q(artifact["eta"]),
        Q(artifact["multiplier"]),
        artifact["coarse_digits"],
        artifact["subdivisions_per_source_coordinate"],
    )
    if recomputed.get("status") != "CLOSED":
        raise RuntimeError("independent exact recomputation did not close")
    exact_fields = (
        "radius",
        "eta",
        "multiplier",
        "source_center_coarsening_error",
        "root_source_enclosure_radius",
        "observer_shift_bound",
        "maximum_subbox_observer_width",
        "spectral_error_row_sum_bound",
        "negative_adjusted_matrix_certified",
        "negative_matrix_leading_minors",
    )
    mismatches = [
        field for field in exact_fields if recomputed.get(field) != artifact.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            "artifact differs from exact recomputation in: " + ", ".join(mismatches)
        )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "certificate": str(args.certificate),
                "radius": recomputed["radius"],
                "eta": recomputed["eta"],
                "source_subbox_count": recomputed["source_subbox_count"],
                "spectral_error_row_sum_decimal": recomputed[
                    "spectral_error_row_sum_decimal"
                ],
                "elapsed_seconds": perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

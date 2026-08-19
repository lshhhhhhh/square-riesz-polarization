"""Independently replay an N=3 exact directional local-cap certificate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from scripts.certify_n3_active_minima import _canonical_json_sha256
from scripts.certify_n3_directional_local_cap import (
    _audit_candidate_coverage,
    _initialize_worker,
    _validated_split_sensitivities,
    _verify_candidate_leaf,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "certificate",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT
        / "data/certificates/n03_directional_local_cap_0_0001.json",
    )
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument(
        "--active-neighborhood",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_full_source_neighborhood.json",
    )
    parser.add_argument("--critical-cone-certificate", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    artifact = json.loads(args.certificate.read_text(encoding="utf-8"))
    if artifact.get("status") != "VERIFIED":
        raise ValueError("certificate does not claim VERIFIED status")
    prerequisites = artifact.get("prerequisites")
    if type(prerequisites) is not dict:
        raise ValueError("prerequisites are missing")
    expected_hashes = {
        "kkt": _canonical_json_sha256(args.kkt_certificate),
        "active_neighborhood": _canonical_json_sha256(args.active_neighborhood),
    }
    critical_cone_path = args.critical_cone_certificate
    critical_entry = prerequisites.get("critical_cone")
    if critical_entry is not None:
        if type(critical_entry) is not dict:
            raise ValueError("critical_cone prerequisite is malformed")
        if critical_cone_path is None:
            stored_path = critical_entry.get("path")
            if type(stored_path) is not str:
                raise ValueError("critical_cone prerequisite path is missing")
            critical_cone_path = PROJECT_ROOT / stored_path
        expected_hashes["critical_cone"] = _canonical_json_sha256(
            critical_cone_path
        )
    for name, expected in expected_hashes.items():
        entry = prerequisites.get(name)
        if type(entry) is not dict or entry.get("canonical_json_sha256") != expected:
            raise ValueError(f"{name} prerequisite hash mismatch")
    leaves = artifact.get("leaves")
    if type(leaves) is not list or len(leaves) != artifact.get("checked_leaf_count"):
        raise ValueError("leaf count is inconsistent")
    if artifact.get("candidate_leaf_count") != len(leaves):
        raise ValueError("candidate leaf count is inconsistent")
    if artifact.get("failure_count") != 0 or artifact.get("failures") != []:
        raise ValueError("published certificate records failed leaves")
    split_policy = artifact.get("split_policy", "cyclic")
    split_sensitivities = (
        _validated_split_sensitivities(artifact.get("split_sensitivities"))
        if split_policy == "transverse-width"
        else None
    )
    _audit_candidate_coverage(leaves, split_policy, split_sensitivities)
    selected = leaves if args.limit is None else leaves[: args.limit]
    radius = Q(artifact["root_centered_linf_radius"])
    digits = artifact.get("coarse_enclosure_digits")
    if type(digits) is not int or digits < 1:
        raise ValueError("coarse_enclosure_digits is invalid")

    initargs = (
        str(args.kkt_certificate),
        str(args.active_neighborhood),
        str(radius),
        digits,
        str(critical_cone_path) if critical_cone_path is not None else None,
        split_policy,
        (
            tuple(str(value) for value in split_sensitivities)
            if split_sensitivities is not None
            else None
        ),
    )
    started = perf_counter()
    if args.workers == 1:
        _initialize_worker(*initargs)
        results = map(_verify_candidate_leaf, selected)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=initargs,
        )
        results = executor.map(_verify_candidate_leaf, selected, chunksize=8)
    failures = []
    criterion_mismatches = []
    try:
        for index, (leaf, result) in enumerate(zip(selected, results), start=1):
            passed, criterion, branch, _ = result
            if not passed:
                failures.append((leaf["root_id"], leaf["path"]))
            elif criterion != leaf.get("criterion") or branch != leaf.get("branch"):
                criterion_mismatches.append((leaf["root_id"], leaf["path"]))
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "replayed": index,
                            "total": len(selected),
                            "failures": len(failures),
                            "criterion_mismatches": len(criterion_mismatches),
                        }
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    if failures or criterion_mismatches:
        raise RuntimeError(
            f"replay failed: {len(failures)} open leaves and "
            f"{len(criterion_mismatches)} criterion mismatches"
        )
    print(
        json.dumps(
            {
                "status": "VERIFIED" if args.limit is None else "SAMPLE_VERIFIED",
                "certificate": str(args.certificate),
                "replayed_leaves": len(selected),
                "coverage": "exact prefix-free and Kraft checks passed",
                "elapsed_seconds": perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

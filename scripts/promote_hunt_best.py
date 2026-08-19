"""Freeze a record-hunt best candidate as literal decimals and re-evaluate it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from square_riesz.evaluate import evaluate_continuous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--digits", type=int, default=17)
    parser.add_argument("--evaluation-grid", type=int, default=257)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.digits < 12:
        raise SystemExit("--digits must be at least 12")

    best_payload = json.loads(args.best.read_text(encoding="utf-8"))
    source = best_payload[str(args.n)]
    coordinate_strings = [
        [format(float(x), f".{args.digits}g"), format(float(y), f".{args.digits}g")]
        for x, y in source["coordinates"]
    ]
    coordinates = np.asarray(coordinate_strings, dtype=np.float64)
    evaluation = evaluate_continuous(
        coordinates,
        grid_size=args.evaluation_grid,
        low_seed_count=max(96, 12 * args.n),
        active_tolerance=2e-6,
    )
    baseline = float(source["baseline"])
    result = {
        "schema_version": 1,
        "n": args.n,
        "coordinates": coordinate_strings,
        "coordinate_significant_digits": args.digits,
        "provenance": {
            "best_file": str(args.best.resolve()),
            "job_id": source["job_id"],
            "evaluation_id": source["evaluation_id"],
            "seed": source["seed"],
            "method": "free_cuda_softmin_then_literal_decimal_continuous_evaluation",
        },
        "baseline": baseline,
        "continuous_evaluation": evaluation.to_dict(),
        "margin_over_baseline": evaluation.minimum - baseline,
        "rigor": "numerical_candidate_only_until_both_exact_certifiers_pass",
    }
    output = args.output
    if output is None:
        output = PROJECT_ROOT / "data" / "candidates" / f"n{args.n:02d}_hunt_best.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "n": args.n,
                "minimum": evaluation.minimum,
                "baseline": baseline,
                "margin": evaluation.minimum - baseline,
                "coordinates": coordinate_strings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

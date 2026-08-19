"""Exploratory batched CUDA search followed by float64 continuous evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from square_riesz.evaluate import evaluate_continuous
from square_riesz.torch_engine import (
    optimize_population_softmin,
    random_configurations,
    unit_square_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--grid-size", type=int, default=65)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--start-temperature", type=float, default=1.0)
    parser.add_argument("--end-temperature", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--evaluation-grid", type=int, default=129)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.n < 1:
        raise SystemExit("--n must be positive")

    device = torch.device("cuda")
    initial = random_configurations(
        args.batch,
        args.n,
        seed=args.seed,
        device=device,
        dtype=torch.float32,
    )
    grid = unit_square_grid(args.grid_size, device=device, dtype=torch.float32)

    started = perf_counter()
    search = optimize_population_softmin(
        initial,
        grid,
        steps=args.steps,
        learning_rate=args.learning_rate,
        start_temperature=args.start_temperature,
        end_temperature=args.end_temperature,
        point_chunk_size=args.chunk,
        report_every=max(1, args.steps // 8),
    )
    order = torch.argsort(search.sampled_minima, descending=True)
    top_indices = order[: min(args.top_k, len(order))].cpu().tolist()

    candidates: list[dict[str, object]] = []
    for rank, index in enumerate(top_indices, start=1):
        coordinates = search.configurations[index].cpu().numpy().astype(np.float64)
        evaluation = evaluate_continuous(
            coordinates,
            grid_size=args.evaluation_grid,
            low_seed_count=max(48, 8 * args.n),
        )
        candidates.append(
            {
                "rank": rank,
                "population_index": index,
                "sampled_minimum": float(search.sampled_minima[index].item()),
                "coordinates": coordinates.tolist(),
                "continuous_evaluation": evaluation.to_dict(),
            }
        )

    candidates.sort(
        key=lambda item: item["continuous_evaluation"]["minimum"], reverse=True
    )
    result = {
        "schema_version": 1,
        "method": "batched_cuda_softmin_then_float64_continuous_evaluation",
        "n": args.n,
        "seed": args.seed,
        "batch": args.batch,
        "steps": args.steps,
        "grid_size": args.grid_size,
        "evaluation_grid": args.evaluation_grid,
        "learning_rate": args.learning_rate,
        "start_temperature": args.start_temperature,
        "end_temperature": args.end_temperature,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": perf_counter() - started,
        "history": list(search.history),
        "candidates": candidates,
    }

    output = args.output
    if output is None:
        output = PROJECT_ROOT / "runs" / f"softmin_n{args.n:02d}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)

    summary = {
        "output": str(output),
        "elapsed_seconds": result["elapsed_seconds"],
        "best_sampled_minimum": max(item["sampled_minimum"] for item in candidates),
        "best_continuous_minimum": candidates[0]["continuous_evaluation"]["minimum"],
        "best_coordinates": candidates[0]["coordinates"],
        "history": result["history"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Verify CUDA correctness and measure batched grid-evaluation throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from square_riesz.torch_engine import hard_minimum, random_configurations, unit_square_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--sources", type=int, default=13)
    parser.add_argument("--grid-size", type=int, default=129)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda")
    configurations = random_configurations(
        args.batch,
        args.sources,
        seed=20260818,
        device=device,
        dtype=torch.float32,
    )
    grid = unit_square_grid(args.grid_size, device=device, dtype=torch.float32)

    hard_minimum(configurations, grid, point_chunk_size=args.chunk)
    torch.cuda.synchronize()
    started = perf_counter()
    values = None
    for _ in range(args.repeats):
        values = hard_minimum(configurations, grid, point_chunk_size=args.chunk)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started

    evaluated_pairs = args.batch * len(grid) * args.sources * args.repeats
    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "sources": args.sources,
        "grid_size": args.grid_size,
        "test_points": len(grid),
        "chunk": args.chunk,
        "repeats": args.repeats,
        "elapsed_seconds": elapsed,
        "source_point_pairs_per_second": evaluated_pairs / elapsed,
        "best_sampled_minimum": float(values.max().item()),
        "allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Generate an exactly checked finite-witness global upper certificate for N=3.

Float64 vectorization chooses a promising witness for each source-position box.
The pruning decision itself is then recomputed with ``fractions.Fraction``.
The output contains every pruned leaf's root cell, split path, and witness; the
separate standard-library verifier does not trust this generator's decisions.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.global_upper_prototype import (
    box_witness_upper_bounds_and_argmin,
    initial_unordered_cell_boxes,
    split_nodes,
    witness_grid,
)


Q = Fraction


def _exact_dyadic(value: float) -> Q:
    numerator, denominator = float(value).as_integer_ratio()
    if denominator & (denominator - 1):
        raise ValueError("source boxes must have dyadic float coordinates")
    return Q(numerator, denominator)


def _distance_1d(point: Q, lower: Q, upper: Q) -> Q:
    if point < lower:
        return lower - point
    if point > upper:
        return point - upper
    return Q(0)


def exact_witness_upper_bound(
    node: np.ndarray, witness_index: int, grid_size: int
) -> Q | None:
    """Return the exact uniform box bound, or None for an invalid witness."""

    denominator = grid_size - 1
    point = (Q(witness_index % grid_size, denominator), Q(witness_index // grid_size, denominator))
    total = Q(0)
    for source_box in node:
        x0, x1, y0, y1 = (_exact_dyadic(value) for value in source_box)
        dx = _distance_1d(point[0], x0, x1)
        dy = _distance_1d(point[1], y0, y1)
        squared_distance = dx * dx + dy * dy
        if squared_distance == 0:
            return None
        total += 1 / squared_distance
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="7.7")
    parser.add_argument("--initial-divisions", type=int, default=4)
    parser.add_argument("--witness-grid", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-levels", type=int, default=40)
    parser.add_argument("--max-active", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.initial_divisions < 1 or args.initial_divisions & (args.initial_divisions - 1):
        raise SystemExit("--initial-divisions must be a positive power of two")
    if args.witness_grid < 2:
        raise SystemExit("--witness-grid must be at least 2")
    if args.batch_size < 1 or args.max_levels < 0 or args.max_active < 1:
        raise SystemExit("batch/level/active limits are invalid")
    target = Q(args.target)
    if target <= 0:
        raise SystemExit("--target must be positive")
    witnesses = witness_grid(args.witness_grid)
    active = initial_unordered_cell_boxes(args.initial_divisions)
    cell_count = args.initial_divisions**2
    roots = np.asarray(
        list(combinations_with_replacement(range(cell_count), 3)), dtype=np.int16
    )
    paths = [""] * len(active)
    leaves: list[dict[str, object]] = []
    reports = []
    started = perf_counter()
    status = "max_levels"

    for level in range(args.max_levels + 1):
        level_started = perf_counter()
        bounds, witness_indices = box_witness_upper_bounds_and_argmin(
            active, witnesses, batch_size=args.batch_size
        )
        exact_pruned = np.zeros(len(active), dtype=bool)
        for index, witness_index in enumerate(witness_indices):
            exact_bound = exact_witness_upper_bound(
                active[index], int(witness_index), args.witness_grid
            )
            if exact_bound is not None and exact_bound <= target:
                exact_pruned[index] = True
                leaves.append(
                    {
                        "root": roots[index].astype(int).tolist(),
                        "path": paths[index],
                        "witness": int(witness_index),
                    }
                )

        before = len(active)
        keep = ~exact_pruned
        active = active[keep]
        roots = roots[keep]
        paths = [path for path, retained in zip(paths, keep) if retained]
        report = {
            "level": level,
            "boxes_before_prune": before,
            "boxes_exactly_pruned": int(np.sum(exact_pruned)),
            "boxes_active": int(len(active)),
            "float_maximum": float(np.max(bounds)),
            "level_seconds": perf_counter() - level_started,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
        if len(active) == 0:
            status = "closed_exact_pruning_tree"
            break
        if level == args.max_levels:
            break
        if 2 * len(active) > args.max_active:
            status = "active_box_cap"
            break
        active = split_nodes(active, level % 6)
        roots = np.repeat(roots, 2, axis=0)
        paths = [child for path in paths for child in (path + "0", path + "1")]

    result = {
        "schema_version": 1,
        "claim": "For every three-source configuration in the unit square, the continuous minimum reciprocal-square potential is at most target.",
        "n": 3,
        "target": args.target,
        "initial_divisions": args.initial_divisions,
        "witness_grid": args.witness_grid,
        "split_schedule": "coordinate_at_depth_is_depth_mod_6; bit_0_lower_half_bit_1_upper_half",
        "source_permutation_reduction": "initial root cells are combinations with replacement",
        "status": status,
        "leaf_count": len(leaves),
        "remaining_active_boxes": int(len(active)),
        "generator_elapsed_seconds": perf_counter() - started,
        "reports": reports,
        "leaves": leaves,
    }
    output = args.output
    if output is None:
        output = (
            PROJECT_ROOT
            / "data"
            / "certificates"
            / f"n03_global_upper_{args.target.replace('.', '_')}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": status,
                "leaf_count": len(leaves),
                "remaining_active_boxes": len(active),
                "elapsed_seconds": result["generator_elapsed_seconds"],
            }
        ),
        flush=True,
    )
    if status != "closed_exact_pruning_tree":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

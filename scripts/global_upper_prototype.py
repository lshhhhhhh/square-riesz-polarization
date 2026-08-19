"""Float64 pruning-rate prototype for a global N=3 upper certificate.

For a product of three source-position rectangles and a finite witness set W,
``min_p sum_i 1/dist(p, B_i)^2`` is a uniform upper bound on the continuous
minimum illumination of every configuration in that product box.  This script
tests whether repeated source-box subdivision can close a relaxed target.

Float64 decisions here are diagnostics only.  They are not a proof; a successful
tree must later be replayed with exact rational or outward-rounded arithmetic.
"""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from numpy.typing import NDArray


def witness_grid(size: int) -> NDArray[np.float64]:
    if size < 3:
        raise ValueError("witness grid size must be at least 3")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def initial_unordered_cell_boxes(divisions: int) -> NDArray[np.float64]:
    """Cover configurations modulo source permutation by cell multisets."""

    if divisions < 1:
        raise ValueError("divisions must be positive")
    cells = []
    for y_index in range(divisions):
        for x_index in range(divisions):
            cells.append(
                [
                    x_index / divisions,
                    (x_index + 1) / divisions,
                    y_index / divisions,
                    (y_index + 1) / divisions,
                ]
            )
    nodes = [
        [cells[first], cells[second], cells[third]]
        for first, second, third in combinations_with_replacement(
            range(len(cells)), 3
        )
    ]
    return np.asarray(nodes, dtype=np.float64)


def box_witness_upper_bounds(
    nodes: NDArray[np.float64],
    witnesses: NDArray[np.float64],
    *,
    batch_size: int = 256,
) -> NDArray[np.float64]:
    """Compute non-rigorous float64 witness upper bounds for source boxes."""

    result = np.empty(len(nodes), dtype=np.float64)
    point_x = witnesses[:, 0][None, :]
    point_y = witnesses[:, 1][None, :]
    for start in range(0, len(nodes), batch_size):
        block = nodes[start : start + batch_size]
        total = np.zeros((len(block), len(witnesses)), dtype=np.float64)
        for source_index in range(3):
            boxes = block[:, source_index, :]
            x0 = boxes[:, 0][:, None]
            x1 = boxes[:, 1][:, None]
            y0 = boxes[:, 2][:, None]
            y1 = boxes[:, 3][:, None]
            dx = np.maximum(np.maximum(x0 - point_x, 0.0), point_x - x1)
            dy = np.maximum(np.maximum(y0 - point_y, 0.0), point_y - y1)
            squared_distance = dx * dx + dy * dy
            with np.errstate(divide="ignore"):
                total += np.reciprocal(squared_distance)
        result[start : start + len(block)] = np.min(total, axis=1)
    return result


def box_witness_upper_bounds_and_argmin(
    nodes: NDArray[np.float64],
    witnesses: NDArray[np.float64],
    *,
    batch_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return float64 bounds and one minimizing witness index per node."""

    bounds = np.empty(len(nodes), dtype=np.float64)
    indices = np.empty(len(nodes), dtype=np.int64)
    point_x = witnesses[:, 0][None, :]
    point_y = witnesses[:, 1][None, :]
    for start in range(0, len(nodes), batch_size):
        block = nodes[start : start + batch_size]
        total = np.zeros((len(block), len(witnesses)), dtype=np.float64)
        for source_index in range(3):
            boxes = block[:, source_index, :]
            x0 = boxes[:, 0][:, None]
            x1 = boxes[:, 1][:, None]
            y0 = boxes[:, 2][:, None]
            y1 = boxes[:, 3][:, None]
            dx = np.maximum(np.maximum(x0 - point_x, 0.0), point_x - x1)
            dy = np.maximum(np.maximum(y0 - point_y, 0.0), point_y - y1)
            squared_distance = dx * dx + dy * dy
            with np.errstate(divide="ignore"):
                total += np.reciprocal(squared_distance)
        local_indices = np.argmin(total, axis=1)
        indices[start : start + len(block)] = local_indices
        bounds[start : start + len(block)] = total[
            np.arange(len(block)), local_indices
        ]
    return bounds, indices


def split_nodes(
    nodes: NDArray[np.float64], coordinate_index: int
) -> NDArray[np.float64]:
    """Bisect every node in one of the six source coordinates."""

    source_index, axis = divmod(coordinate_index, 2)
    lower_index, upper_index = (0, 1) if axis == 0 else (2, 3)
    midpoint = (
        nodes[:, source_index, lower_index]
        + nodes[:, source_index, upper_index]
    ) / 2.0
    children = np.repeat(nodes, 2, axis=0)
    children[0::2, source_index, upper_index] = midpoint
    children[1::2, source_index, lower_index] = midpoint
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=7.7)
    parser.add_argument("--initial-divisions", type=int, default=4)
    parser.add_argument("--witness-grid", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-levels", type=int, default=36)
    parser.add_argument("--max-active", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    witnesses = witness_grid(args.witness_grid)
    active = initial_unordered_cell_boxes(args.initial_divisions)
    started = perf_counter()
    levels = []
    status = "max_levels"

    for level in range(args.max_levels + 1):
        level_started = perf_counter()
        before = len(active)
        upper_bounds = box_witness_upper_bounds(
            active, witnesses, batch_size=args.batch_size
        )
        keep = upper_bounds > args.target
        active = active[keep]
        finite = np.isfinite(upper_bounds)
        report = {
            "level": level,
            "coordinate_split": None if level == args.max_levels else level % 6,
            "boxes_before_prune": before,
            "boxes_pruned": int(before - len(active)),
            "boxes_active": int(len(active)),
            "pruned_fraction": float((before - len(active)) / before),
            "finite_fraction": float(np.mean(finite)),
            "minimum_upper_bound": float(np.min(upper_bounds)),
            "maximum_finite_upper_bound": (
                float(np.max(upper_bounds[finite])) if np.any(finite) else None
            ),
            "level_seconds": perf_counter() - level_started,
        }
        levels.append(report)
        print(json.dumps(report), flush=True)
        if len(active) == 0:
            status = "closed_in_float64"
            break
        if level == args.max_levels:
            break
        if 2 * len(active) > args.max_active:
            status = "active_box_cap"
            break
        active = split_nodes(active, level % 6)

    result = {
        "schema_version": 1,
        "method": "float64_finite_witness_source_box_pruning_prototype",
        "rigor": "not_a_proof",
        "target": args.target,
        "initial_divisions": args.initial_divisions,
        "witness_grid": args.witness_grid,
        "witness_count": len(witnesses),
        "max_levels": args.max_levels,
        "max_active": args.max_active,
        "status": status,
        "remaining_active_boxes": int(len(active)),
        "elapsed_seconds": perf_counter() - started,
        "levels": levels,
    }
    output = args.output
    if output is None:
        output = (
            PROJECT_ROOT
            / "runs"
            / f"global_upper_prototype_target_{str(args.target).replace('.', '_')}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **result}), flush=True)


if __name__ == "__main__":
    main()

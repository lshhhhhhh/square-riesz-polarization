"""Float64 ROI diagnostic for the local-cap/outside-slab union terminal."""

from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.global_upper_prototype import (
    adaptive_boundary_witness_upper_bounds,
    box_witness_upper_bounds,
    boxes_closed_by_cap_outside_slabs,
    boxes_inside_local_cap,
    witness_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=7.57)
    parser.add_argument("--radius", type=float, default=0.001)
    parser.add_argument("--dyadic-denominator", type=int, default=1024)
    parser.add_argument("--neighbor-offset", type=int, default=2)
    parser.add_argument("--witness-grid", type=int, default=33)
    parser.add_argument("--adaptive-boundary-witnesses", action="store_true")
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    args = parser.parse_args()
    kkt = json.loads(args.kkt_certificate.read_text(encoding="utf-8"))
    center = kkt["center"]
    a = float(Fraction(center["a"]))
    b = float(Fraction(center["b"]))
    c = float(Fraction(center["c"]))
    source_center = np.asarray([[a, b], [1.0 - a, b], [0.5, c]])
    flat_center = source_center.ravel()
    denominator = args.dyadic_denominator
    base_indices = np.floor(flat_center * denominator).astype(np.int64)
    offsets = range(-args.neighbor_offset, args.neighbor_offset + 1)
    selections = np.asarray(list(product(offsets, repeat=6)), dtype=np.int64)
    lower = (base_indices[None, :] + selections) / denominator
    upper = lower + 1.0 / denominator
    valid = np.all((lower >= 0.0) & (upper <= 1.0), axis=1)
    lower = lower[valid]
    upper = upper[valid]
    nodes = np.empty((len(lower), 3, 4), dtype=np.float64)
    nodes[:, :, 0] = lower[:, 0::2]
    nodes[:, :, 1] = upper[:, 0::2]
    nodes[:, :, 2] = lower[:, 1::2]
    nodes[:, :, 3] = upper[:, 1::2]
    copies = source_center[None, :, :]
    witnesses = witness_grid(args.witness_grid)
    started = perf_counter()
    baseline_bounds = box_witness_upper_bounds(nodes, witnesses)
    if args.adaptive_boundary_witnesses:
        baseline_bounds = np.minimum(
            baseline_bounds, adaptive_boundary_witness_upper_bounds(nodes)
        )
    baseline_closed = baseline_bounds <= args.target
    contained = boxes_inside_local_cap(nodes, copies, args.radius)
    hybrid = boxes_closed_by_cap_outside_slabs(
        nodes,
        copies,
        args.radius,
        args.target,
        witnesses,
        adaptive_boundary_witnesses=args.adaptive_boundary_witnesses,
    )
    print(
        json.dumps(
            {
                "rigor": "not_a_proof_float64_roi_diagnostic",
                "target": args.target,
                "radius": args.radius,
                "dyadic_width": 1.0 / denominator,
                "adaptive_boundary_witnesses": args.adaptive_boundary_witnesses,
                "boxes": len(nodes),
                "baseline_witness_closed": int(np.count_nonzero(baseline_closed)),
                "contained_cap_closed": int(np.count_nonzero(contained)),
                "hybrid_union_closed": int(np.count_nonzero(hybrid)),
                "additional_contained_after_witness": int(
                    np.count_nonzero(contained & ~baseline_closed)
                ),
                "additional_hybrid_after_witness_and_contained": int(
                    np.count_nonzero(hybrid & ~baseline_closed & ~contained)
                ),
                "elapsed_seconds": perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

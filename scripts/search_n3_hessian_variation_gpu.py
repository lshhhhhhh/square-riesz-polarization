"""GPU stress test for N=3 weighted envelope-Hessian variation.

This samples arbitrary full six-dimensional source perturbations, solves for
the moving bottom-edge observer, and compares the weighted five-branch
envelope Hessian with its KKT-center value.  It is a float64 adversarial ROI
test, not a proof or an interval bound.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.search_n3_direction_cover_gpu import (
    CORNERS,
    Classifier,
    moving_bottom_hessian,
    observer_branch,
)


def stationary_bottom_observer(sources: torch.Tensor, steps: int = 8) -> torch.Tensor:
    observer = torch.full(
        (len(sources),), 0.5, device=sources.device, dtype=sources.dtype
    )
    for _ in range(steps):
        difference = observer[:, None] - sources[:, :, 0]
        normal_squared = sources[:, :, 1].square()
        squared_distance = difference.square() + normal_squared
        gradient = torch.sum(-2.0 * difference / squared_distance.square(), dim=1)
        curvature = torch.sum(
            -2.0 / squared_distance.square()
            + 8.0 * difference.square() / squared_distance.pow(3),
            dim=1,
        )
        observer = torch.clamp(observer - gradient / curvature, 0.375, 0.625)
    return observer


def point_weighted_hessian(
    sources: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    point_sources = torch.stack((sources, sources), dim=-1)
    hessians = []
    for x, y in CORNERS:
        observer_x = torch.full(
            (len(sources), 2), x, device=sources.device, dtype=sources.dtype
        )
        observer_y = torch.full(
            (len(sources), 2), y, device=sources.device, dtype=sources.dtype
        )
        _, hessian = observer_branch(observer_x, observer_y, point_sources)
        hessians.append(hessian[..., 0])
    bottom = stationary_bottom_observer(sources)
    observer_x = torch.stack((bottom, bottom), dim=-1)
    _, moving = moving_bottom_hessian(observer_x, point_sources)
    hessians.append(moving[..., 0])
    result = torch.zeros_like(hessians[0])
    for weight, hessian in zip(weights, hessians):
        result += weight * hessian
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument("--radii", nargs="+", default=["0.0001", "0.0005", "0.001"])
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026081903)
    parser.add_argument("--cone-ratio", type=float, default=0.7)
    parser.add_argument("--cone-multiplier", type=float, default=1.3230840091475613)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/n03_hessian_variation_gpu.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("sample and batch counts must be positive")

    device = torch.device("cuda")
    dtype = torch.float64
    kkt = json.loads(args.kkt_certificate.read_text(encoding="utf-8"))
    center_data = kkt["center"]
    a = float(Fraction(center_data["a"]))
    b = float(Fraction(center_data["b"]))
    c = float(Fraction(center_data["c"]))
    center = torch.tensor(
        [[a, b], [1.0 - a, b], [0.5, c]], device=device, dtype=dtype
    )
    weights = torch.tensor(
        [
            float(Fraction(center_data["top_weight"])) / 2,
            float(Fraction(center_data["top_weight"])) / 2,
            float(Fraction(center_data["corner_weight"])) / 2,
            float(Fraction(center_data["corner_weight"])) / 2,
            float(Fraction(center_data["midpoint_weight"])),
        ],
        device=device,
        dtype=dtype,
    )
    root_hessian = point_weighted_hessian(center[None, ...], weights)[0]
    classifier = Classifier(center, weights, 0.0)
    difference = classifier.root_gradients[:4] - classifier.root_gradients[4]
    metric = difference.T @ difference
    identity = torch.eye(6, device=device, dtype=dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    started = perf_counter()
    reports = []
    for radius_text in args.radii:
        radius = float(radius_text)
        maximum_variation = -1.0
        maximum_adjusted = torch.tensor(-torch.inf, device=device, dtype=dtype)
        maximum_sources = None
        processed = 0
        while processed < args.samples:
            count = min(args.batch_size, args.samples - processed)
            offsets = 2.0 * torch.rand(
                (count, 3, 2), device=device, dtype=dtype, generator=generator
            ) - 1.0
            sources = center[None, ...] + radius * offsets
            hessian = point_weighted_hessian(sources, weights)
            difference_hessian = hessian - root_hessian
            eigenvalues = torch.linalg.eigvalsh(difference_hessian)
            variation = torch.maximum(
                torch.abs(eigenvalues[:, 0]), torch.abs(eigenvalues[:, -1])
            )
            local_maximum, local_index = torch.max(variation, dim=0)
            if float(local_maximum) > maximum_variation:
                maximum_variation = float(local_maximum)
                maximum_sources = sources[int(local_index)].detach().cpu().tolist()
            adjusted = (
                hessian
                - args.cone_multiplier * metric
                + args.cone_multiplier * args.cone_ratio**2 * identity
            )
            maximum_adjusted = torch.maximum(
                maximum_adjusted, torch.linalg.eigvalsh(adjusted)[:, -1].max()
            )
            processed += count

        # Include all 64 corners of the six-dimensional perturbation cube.
        corner_bits = torch.arange(64, device=device, dtype=torch.int64)
        corner_signs = torch.stack(
            [2.0 * ((corner_bits >> bit) & 1).to(dtype) - 1.0 for bit in range(6)],
            dim=1,
        ).reshape(64, 3, 2)
        corner_sources = center[None, ...] + radius * corner_signs
        corner_hessian = point_weighted_hessian(corner_sources, weights)
        corner_eigenvalues = torch.linalg.eigvalsh(corner_hessian - root_hessian)
        corner_variation = torch.maximum(
            torch.abs(corner_eigenvalues[:, 0]),
            torch.abs(corner_eigenvalues[:, -1]),
        )
        corner_maximum, corner_index = torch.max(corner_variation, dim=0)
        if float(corner_maximum) > maximum_variation:
            maximum_variation = float(corner_maximum)
            maximum_sources = corner_sources[int(corner_index)].cpu().tolist()
        corner_adjusted = (
            corner_hessian
            - args.cone_multiplier * metric
            + args.cone_multiplier * args.cone_ratio**2 * identity
        )
        maximum_adjusted = torch.maximum(
            maximum_adjusted,
            torch.linalg.eigvalsh(corner_adjusted)[:, -1].max(),
        )
        report = {
            "radius": radius,
            "random_samples": args.samples,
            "maximum_observed_hessian_spectral_variation": maximum_variation,
            "maximum_observed_adjusted_cone_eigenvalue": float(maximum_adjusted),
            "maximum_variation_sources": maximum_sources,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)

    result = {
        "schema_version": 1,
        "rigor": "not_a_proof_float64_gpu_stress_test",
        "device": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "cone_ratio": args.cone_ratio,
        "cone_multiplier": args.cone_multiplier,
        "reports": reports,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result}), flush=True)


if __name__ == "__main__":
    main()

"""GPU diagnostic for the quantitative basin of the symmetric N=3 root.

This is deliberately not a proof.  It searches the L-infinity boundary of
source-space boxes and helps decide whether a direction-subdivision interval
certificate is worth implementing.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from time import perf_counter

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORNERS = ((0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0))


def active_values(sources: torch.Tensor, *, newton_steps: int = 8) -> torch.Tensor:
    """Four corner branches and the locally minimized bottom-edge branch."""

    values = []
    for x, y in CORNERS:
        dx = x - sources[..., :, 0]
        dy = y - sources[..., :, 1]
        values.append(torch.reciprocal(dx * dx + dy * dy).sum(dim=-1))

    observer_x = torch.full_like(sources[..., 0, 0], 0.5)
    for _ in range(newton_steps):
        dx = observer_x[..., None] - sources[..., :, 0]
        dy = -sources[..., :, 1]
        r2 = dx * dx + dy * dy
        ux = (-2.0 * dx / r2.square()).sum(dim=-1)
        uxx = (-2.0 / r2.square() + 8.0 * dx.square() / r2.pow(3)).sum(
            dim=-1
        )
        observer_x = torch.clamp(observer_x - ux / uxx, 0.375, 0.625)
    dx = observer_x[..., None] - sources[..., :, 0]
    dy = -sources[..., :, 1]
    values.append(torch.reciprocal(dx * dx + dy * dy).sum(dim=-1))
    return torch.stack(values, dim=-1)


def critical_basis(center: torch.Tensor) -> torch.Tensor:
    variable = center.detach().clone().requires_grad_(True)
    values = active_values(variable[None, ...])[0]
    gradients = []
    for branch in range(5):
        gradient = torch.autograd.grad(
            values[branch], variable, retain_graph=True
        )[0]
        gradients.append(gradient.reshape(-1))
    gradient_matrix = torch.stack(gradients)
    differences = gradient_matrix[:4] - gradient_matrix[4]
    _, _, vh = torch.linalg.svd(differences, full_matrices=True)
    return vh[-2:].T


def best_sampled_direction(
    center: torch.Tensor,
    root_level: float,
    radius: float,
    basis: torch.Tensor,
    *,
    random_samples: int,
    chunk_size: int,
) -> tuple[float, torch.Tensor, torch.Tensor, str]:
    device = center.device
    dtype = center.dtype
    best_value = -torch.inf
    best_direction = torch.zeros(6, device=device, dtype=dtype)
    best_branches = torch.zeros(5, device=device, dtype=dtype)
    best_family = ""

    def consider(directions: torch.Tensor, family: str) -> None:
        nonlocal best_value, best_direction, best_branches, best_family
        for start in range(0, len(directions), chunk_size):
            block = directions[start : start + chunk_size]
            sources = center[None, ...] + radius * block.reshape(-1, 3, 2)
            branches = active_values(sources)
            minima = branches.min(dim=1).values - root_level
            value, index = minima.max(dim=0)
            if value > best_value:
                best_value = value.detach()
                best_direction = block[index].detach().clone()
                best_branches = branches[index].detach().clone()
                best_family = family

    generator = torch.Generator(device=device)
    generator.manual_seed(50903)
    remaining = random_samples
    while remaining:
        count = min(remaining, chunk_size)
        directions = 2.0 * torch.rand(
            (count, 6), generator=generator, device=device, dtype=dtype
        ) - 1.0
        directions /= directions.abs().amax(dim=1, keepdim=True)
        consider(directions, "random")
        remaining -= count

    angles = torch.linspace(
        0.0, 2.0 * torch.pi, 131073, device=device, dtype=dtype
    )[:-1]
    critical = (
        torch.cos(angles)[:, None] * basis[:, 0][None, :]
        + torch.sin(angles)[:, None] * basis[:, 1][None, :]
    )
    critical /= critical.abs().amax(dim=1, keepdim=True)
    consider(critical, "critical_circle")
    return (
        float(best_value.cpu()),
        best_direction.cpu(),
        best_branches.cpu(),
        best_family,
    )


def optimize_faces(
    center: torch.Tensor,
    root_level: float,
    radius: float,
    *,
    starts_per_face: int,
    steps: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Subgradient ascent on all 12 faces of the L-infinity unit box."""

    device = center.device
    dtype = center.dtype
    face_coordinates = torch.arange(6, device=device).repeat_interleave(
        2 * starts_per_face
    )
    face_signs = torch.tensor(
        [-1.0, 1.0], device=device, dtype=dtype
    ).repeat(6).repeat_interleave(starts_per_face)
    count = len(face_coordinates)
    generator = torch.Generator(device=device)
    generator.manual_seed(77003)
    parameter = (
        2.0
        * torch.rand((count, 6), generator=generator, device=device, dtype=dtype)
        - 1.0
    ).requires_grad_(True)
    row = torch.arange(count, device=device)
    with torch.no_grad():
        parameter[row, face_coordinates] = face_signs
    optimizer = torch.optim.Adam([parameter], lr=0.025)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        sources = center[None, ...] + radius * parameter.reshape(-1, 3, 2)
        objective = active_values(sources).min(dim=1).values
        (-objective.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(-1.0, 1.0)
            parameter[row, face_coordinates] = face_signs
    with torch.no_grad():
        branches = active_values(
            center[None, ...] + radius * parameter.reshape(-1, 3, 2)
        )
        minima = branches.min(dim=1).values - root_level
        value, index = minima.max(dim=0)
        return (
            float(value.cpu()),
            parameter[index].detach().cpu(),
            branches[index].detach().cpu(),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument(
        "--radii", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3"
    )
    parser.add_argument("--random-samples", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--starts-per-face", type=int, default=128)
    parser.add_argument("--optimization-steps", type=int, default=800)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "n03_local_basin_gpu.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    device = torch.device("cuda")
    dtype = torch.float64
    artifact = json.loads(args.kkt_certificate.read_text(encoding="utf-8"))
    center_data = artifact["center"]
    a = float(Fraction(center_data["a"]))
    b = float(Fraction(center_data["b"]))
    c = float(Fraction(center_data["c"]))
    root_level = float(Fraction(center_data["level"]))
    center = torch.tensor(
        [[a, b], [1.0 - a, b], [0.5, c]], device=device, dtype=dtype
    )
    basis = critical_basis(center)
    started = perf_counter()
    reports = []
    for radius in (float(item) for item in args.radii.split(",")):
        sample_value, sample_direction, sample_branches, family = (
            best_sampled_direction(
                center,
                root_level,
                radius,
                basis,
                random_samples=args.random_samples,
                chunk_size=args.chunk_size,
            )
        )
        optimized_value, optimized_direction, optimized_branches = optimize_faces(
            center,
            root_level,
            radius,
            starts_per_face=args.starts_per_face,
            steps=args.optimization_steps,
        )
        if optimized_value >= sample_value:
            value = optimized_value
            direction = optimized_direction
            branches = optimized_branches
            method = "face_subgradient_ascent"
        else:
            value = sample_value
            direction = sample_direction
            branches = sample_branches
            method = family
        report = {
            "radius": radius,
            "largest_found_min_branch_minus_root": value,
            "scaled_by_radius_squared": value / (radius * radius),
            "method": method,
            "direction": direction.tolist(),
            "branch_values": branches.tolist(),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)

    result = {
        "schema_version": 1,
        "rigor": "not_a_proof",
        "purpose": "ROI diagnostic for a direction-subdivision local certificate",
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "root_level": root_level,
        "random_samples_per_radius": args.random_samples,
        "starts_per_face": args.starts_per_face,
        "optimization_steps": args.optimization_steps,
        "critical_basis": basis.detach().cpu().tolist(),
        "reports": reports,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), **result}), flush=True)


if __name__ == "__main__":
    main()

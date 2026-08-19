"""GPU float-interval prototype for the N=3 directional local proof tree.

The formulas mirror ``certify_n3_directional_local_cap.py``, but binary64
interval endpoints are diagnostic only.  A successful tree must be replayed
with exact rational arithmetic.
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


def iadd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.stack((left[..., 0] + right[..., 0], left[..., 1] + right[..., 1]), -1)


def ineg(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((-value[..., 1], -value[..., 0]), -1)


def isub(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return iadd(left, ineg(right))


def imul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    products = torch.stack(
        (
            left[..., 0] * right[..., 0],
            left[..., 0] * right[..., 1],
            left[..., 1] * right[..., 0],
            left[..., 1] * right[..., 1],
        ),
        -1,
    )
    return torch.stack((products.amin(-1), products.amax(-1)), -1)


def ireciprocal(value: torch.Tensor) -> torch.Tensor:
    return torch.stack((1.0 / value[..., 1], 1.0 / value[..., 0]), -1)


def idiv(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return imul(left, ireciprocal(right))


def isquare(value: torch.Tensor) -> torch.Tensor:
    lower = torch.where(
        (value[..., 0] <= 0.0) & (value[..., 1] >= 0.0),
        torch.zeros_like(value[..., 0]),
        torch.minimum(value[..., 0].square(), value[..., 1].square()),
    )
    upper = torch.maximum(value[..., 0].square(), value[..., 1].square())
    return torch.stack((lower, upper), -1)


def scale(value: torch.Tensor, factor: float) -> torch.Tensor:
    if factor >= 0:
        return value * factor
    return torch.stack((value[..., 1] * factor, value[..., 0] * factor), -1)


def observer_branch(
    observer_x: torch.Tensor,
    observer_y: torch.Tensor,
    sources: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(sources)
    gradient = torch.zeros((count, 6, 2), device=sources.device, dtype=sources.dtype)
    hessian = torch.zeros((count, 6, 6, 2), device=sources.device, dtype=sources.dtype)
    for source in range(3):
        dx = isub(observer_x, sources[:, source, 0])
        dy = isub(observer_y, sources[:, source, 1])
        dx2 = isquare(dx)
        dy2 = isquare(dy)
        r2 = iadd(dx2, dy2)
        r4 = isquare(r2)
        r6 = imul(r4, r2)
        coordinate = 2 * source
        gradient[:, coordinate] = idiv(scale(dx, 2.0), r4)
        gradient[:, coordinate + 1] = idiv(scale(dy, 2.0), r4)
        hessian[:, coordinate, coordinate] = idiv(
            isub(scale(dx2, 6.0), scale(dy2, 2.0)), r6
        )
        hessian[:, coordinate + 1, coordinate + 1] = idiv(
            isub(scale(dy2, 6.0), scale(dx2, 2.0)), r6
        )
        mixed = idiv(scale(imul(dx, dy), 8.0), r6)
        hessian[:, coordinate, coordinate + 1] = mixed
        hessian[:, coordinate + 1, coordinate] = mixed
    return gradient, hessian


def moving_bottom_hessian(
    observer_x: torch.Tensor, sources: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    observer_y = torch.zeros_like(observer_x)
    gradient, fixed = observer_branch(observer_x, observer_y, sources)
    count = len(sources)
    cross = torch.zeros((count, 6, 2), device=sources.device, dtype=sources.dtype)
    observer_xx = torch.zeros((count, 2), device=sources.device, dtype=sources.dtype)
    for source in range(3):
        dx = isub(observer_x, sources[:, source, 0])
        dy = ineg(sources[:, source, 1])
        dx2 = isquare(dx)
        dy2 = isquare(dy)
        r2 = iadd(dx2, dy2)
        r4 = isquare(r2)
        r6 = imul(r4, r2)
        coordinate = 2 * source
        cross[:, coordinate] = isub(
            scale(ireciprocal(r4), 2.0),
            scale(idiv(dx2, r6), 8.0),
        )
        cross[:, coordinate + 1] = scale(idiv(imul(dx, dy), r6), -8.0)
        observer_xx = iadd(observer_xx, fixed[:, coordinate, coordinate])
    envelope = torch.empty_like(fixed)
    for row in range(6):
        for column in range(6):
            correction = idiv(
                imul(cross[:, row], cross[:, column]), observer_xx
            )
            envelope[:, row, column] = isub(fixed[:, row, column], correction)
    return gradient, envelope


def linear_form(direction: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((len(direction), 2), device=direction.device, dtype=direction.dtype)
    for index in range(6):
        result = iadd(result, imul(direction[:, index], gradient[:, index]))
    return result


def quadratic_form(direction: torch.Tensor, hessian: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((len(direction), 2), device=direction.device, dtype=direction.dtype)
    for row in range(6):
        for column in range(6):
            result = iadd(
                result,
                imul(imul(direction[:, row], hessian[:, row, column]), direction[:, column]),
            )
    return result


class Classifier:
    def __init__(self, center: torch.Tensor, weights: torch.Tensor, radius: float) -> None:
        self.center = center
        self.weights = weights
        self.radius = radius
        point_sources = torch.stack((center, center), -1)[None, ...]
        gradients = []
        for x, y in CORNERS:
            point_x = torch.tensor([[x, x]], device=center.device, dtype=center.dtype)
            point_y = torch.tensor([[y, y]], device=center.device, dtype=center.dtype)
            gradient, _ = observer_branch(point_x, point_y, point_sources)
            gradients.append(gradient[0, :, 0])
        midpoint = torch.tensor([[0.5, 0.5]], device=center.device, dtype=center.dtype)
        bottom, _ = moving_bottom_hessian(midpoint, point_sources)
        gradients.append(bottom[0, :, 0])
        self.root_gradients = torch.stack(gradients)

    def classify(self, direction: torch.Tensor) -> torch.Tensor:
        count = len(direction)
        offset_lower = torch.minimum(
            torch.zeros_like(direction[..., 0]), self.radius * direction[..., 0]
        )
        offset_upper = torch.maximum(
            torch.zeros_like(direction[..., 1]), self.radius * direction[..., 1]
        )
        offsets = torch.stack((offset_lower, offset_upper), -1)
        sources = self.center.reshape(1, 3, 2, 1) + offsets.reshape(count, 3, 2, 2)
        hessians = []
        for x, y in CORNERS:
            point_x = torch.full((count, 2), x, device=direction.device, dtype=direction.dtype)
            point_y = torch.full((count, 2), y, device=direction.device, dtype=direction.dtype)
            _, hessian = observer_branch(point_x, point_y, sources)
            hessians.append(hessian)

        midpoint = torch.full((count, 2), 0.5, device=direction.device, dtype=direction.dtype)
        bottom_y = torch.zeros((count, 2), device=direction.device, dtype=direction.dtype)
        midpoint_gradient, _ = observer_branch(midpoint, bottom_y, sources)
        ux_sum = torch.zeros((count, 2), device=direction.device, dtype=direction.dtype)
        for source in range(3):
            ux_sum = iadd(ux_sum, midpoint_gradient[:, 2 * source])
        shift = torch.maximum(ux_sum[:, 0].abs(), ux_sum[:, 1].abs()) / 3.0
        observer_x = torch.stack((0.5 - shift, 0.5 + shift), -1)
        _, moving = moving_bottom_hessian(observer_x, sources)
        hessians.append(moving)

        terminal = torch.zeros(count, device=direction.device, dtype=torch.int8)
        valid = shift < 0.125
        for branch in range(5):
            gradient = self.root_gradients[branch].reshape(1, 6, 1).expand(count, -1, 2)
            linear = linear_form(direction, gradient)
            quadratic = quadratic_form(direction, hessians[branch])
            ray_upper = linear[:, 1] + 0.5 * self.radius * torch.clamp_min(
                quadratic[:, 1], 0.0
            )
            terminal[(terminal == 0) & valid & (ray_upper < 0.0)] = 1

        weighted = torch.zeros_like(hessians[0])
        for weight, hessian in zip(self.weights, hessians):
            weighted = iadd(weighted, scale(hessian, float(weight)))
        weighted_quadratic = quadratic_form(direction, weighted)
        terminal[(terminal == 0) & valid & (weighted_quadratic[:, 1] < 0.0)] = 2
        return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--max-depth", type=int, default=35)
    parser.add_argument("--max-active", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/n03_direction_cover_gpu.json",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
    classifier = Classifier(center, weights, args.radius)
    directions = torch.empty((12, 6, 2), device=device, dtype=dtype)
    directions[..., 0] = -1.0
    directions[..., 1] = 1.0
    faces = torch.arange(6, device=device).repeat_interleave(2)
    signs = torch.tensor([-1.0, 1.0], device=device, dtype=dtype).repeat(6)
    rows = torch.arange(12, device=device)
    root_ids = list(range(12))
    paths = [""] * 12
    leaves: list[dict[str, object]] = []
    directions[rows, faces, 0] = signs
    directions[rows, faces, 1] = signs
    free_columns = torch.tensor(
        [[item for item in range(6) if item != face] for face in range(6)],
        device=device,
    )
    started = perf_counter()
    reports = []
    status = "max_depth"
    for depth in range(args.max_depth + 1):
        terminal_parts = []
        for start in range(0, len(directions), args.batch_size):
            terminal_parts.append(
                classifier.classify(directions[start : start + args.batch_size])
            )
        terminal = torch.cat(terminal_parts)
        branch_count = int((terminal == 1).sum().cpu())
        weighted_count = int((terminal == 2).sum().cpu())
        terminal_cpu = terminal.cpu().tolist()
        for root_id, path, criterion in zip(root_ids, paths, terminal_cpu):
            if criterion:
                leaves.append(
                    {
                        "root_id": root_id,
                        "path": path,
                        "float_criterion": (
                            "branch" if criterion == 1 else "weighted"
                        ),
                    }
                )
        unresolved_mask = terminal == 0
        keep = [index for index, criterion in enumerate(terminal_cpu) if criterion == 0]
        root_ids = [root_ids[index] for index in keep]
        paths = [paths[index] for index in keep]
        directions = directions[unresolved_mask]
        faces = faces[unresolved_mask]
        report = {
            "depth": depth,
            "before": len(terminal),
            "branch_closed": branch_count,
            "weighted_closed": weighted_count,
            "unresolved": len(directions),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
        if not len(directions):
            status = "closed_in_float_interval"
            break
        if depth == args.max_depth:
            break
        if 2 * len(directions) > args.max_active:
            status = "active_cap"
            break
        split_coordinates = free_columns[faces, depth % 5]
        parent_count = len(directions)
        children = directions.repeat_interleave(2, dim=0)
        child_faces = faces.repeat_interleave(2)
        child_rows = torch.arange(2 * parent_count, device=device)
        child_coordinates = split_coordinates.repeat_interleave(2)
        parent_midpoints = (
            directions[torch.arange(parent_count, device=device), split_coordinates, 0]
            + directions[torch.arange(parent_count, device=device), split_coordinates, 1]
        ) / 2.0
        midpoints = parent_midpoints.repeat_interleave(2)
        lower_children = child_rows % 2 == 0
        children[child_rows[lower_children], child_coordinates[lower_children], 1] = midpoints[
            lower_children
        ]
        children[child_rows[~lower_children], child_coordinates[~lower_children], 0] = midpoints[
            ~lower_children
        ]
        directions = children
        faces = child_faces
        root_ids = [root_id for root_id in root_ids for _ in range(2)]
        paths = [path + bit for path in paths for bit in ("0", "1")]

    result = {
        "schema_version": 1,
        "rigor": "not_a_proof",
        "method": "binary64 interval-formula direction-cover prototype",
        "device": torch.cuda.get_device_name(0),
        "radius": args.radius,
        "status": status,
        "remaining": len(directions),
        "leaf_count": len(leaves),
        "leaves": leaves,
        "elapsed_seconds": perf_counter() - started,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": status,
                "radius": args.radius,
                "remaining": len(directions),
                "leaf_count": len(leaves),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

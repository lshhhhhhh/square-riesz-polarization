"""Batched PyTorch kernels for exploratory source-configuration search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import torch
from torch import Tensor


def unit_square_grid(
    grid_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    axis = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def _validate_shapes(configurations: Tensor, test_points: Tensor) -> None:
    if configurations.ndim != 3 or configurations.shape[-1] != 2:
        raise ValueError("configurations must have shape (B, N, 2)")
    if test_points.ndim != 2 or test_points.shape[-1] != 2:
        raise ValueError("test_points must have shape (M, 2)")
    if configurations.device != test_points.device:
        raise ValueError("configurations and test_points must share a device")
    if configurations.dtype != test_points.dtype:
        raise ValueError("configurations and test_points must share a dtype")


def intensity_block(
    configurations: Tensor,
    test_points: Tensor,
    *,
    squared_distance_floor: float | None = None,
) -> Tensor:
    """Return intensities with shape ``(batch, test_points)``."""

    _validate_shapes(configurations, test_points)
    delta = test_points[None, :, None, :] - configurations[:, None, :, :]
    squared_distance = torch.sum(delta * delta, dim=-1)
    floor = squared_distance_floor or torch.finfo(configurations.dtype).tiny
    return torch.reciprocal(torch.clamp_min(squared_distance, floor)).sum(dim=-1)


def hard_minimum(
    configurations: Tensor,
    test_points: Tensor,
    *,
    point_chunk_size: int = 4096,
) -> Tensor:
    """Minimum sampled intensity for each configuration."""

    _validate_shapes(configurations, test_points)
    if point_chunk_size < 1:
        raise ValueError("point_chunk_size must be positive")
    result: Tensor | None = None
    for start in range(0, len(test_points), point_chunk_size):
        values = intensity_block(
            configurations, test_points[start : start + point_chunk_size]
        )
        chunk_minimum = values.min(dim=1).values
        result = chunk_minimum if result is None else torch.minimum(result, chunk_minimum)
    if result is None:
        raise ValueError("test_points must be non-empty")
    return result


def soft_minimum(
    configurations: Tensor,
    test_points: Tensor,
    temperature: float,
    *,
    point_chunk_size: int = 4096,
    normalize: bool = True,
) -> Tensor:
    """Differentiable log-sum-exp approximation to the sampled minimum.

    Normalization adds the configuration-independent ``temperature*log(M)``
    term. It changes reported values but not gradients or optimizers.
    """

    _validate_shapes(configurations, test_points)
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    running_logsum: Tensor | None = None
    for start in range(0, len(test_points), point_chunk_size):
        values = intensity_block(
            configurations, test_points[start : start + point_chunk_size]
        )
        chunk_logsum = torch.logsumexp(-values / temperature, dim=1)
        running_logsum = (
            chunk_logsum
            if running_logsum is None
            else torch.logaddexp(running_logsum, chunk_logsum)
        )
    if running_logsum is None:
        raise ValueError("test_points must be non-empty")
    result = -temperature * running_logsum
    if normalize:
        result = result + temperature * math.log(len(test_points))
    return result


def random_configurations(
    batch_size: int,
    source_count: int,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if batch_size < 1 or source_count < 1:
        raise ValueError("batch_size and source_count must be positive")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand(
        (batch_size, source_count, 2),
        generator=generator,
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class PopulationSearchResult:
    configurations: Tensor
    sampled_minima: Tensor
    history: tuple[dict[str, float], ...]


def optimize_population_softmin(
    initial_configurations: Tensor,
    test_points: Tensor,
    *,
    steps: int = 500,
    learning_rate: float = 0.01,
    start_temperature: float = 1.0,
    end_temperature: float = 0.02,
    point_chunk_size: int = 4096,
    report_every: int = 50,
) -> PopulationSearchResult:
    """Independently optimize a batch of configurations on a sampled grid."""

    if steps < 1:
        raise ValueError("steps must be positive")
    if start_temperature <= 0.0 or end_temperature <= 0.0:
        raise ValueError("temperatures must be positive")

    configurations = initial_configurations.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([configurations], lr=learning_rate)
    history: list[dict[str, float]] = []
    start_time = perf_counter()
    with torch.no_grad():
        best_configurations = configurations.detach().clone()
        best_values = hard_minimum(
            configurations,
            test_points,
            point_chunk_size=point_chunk_size,
        )

    for step in range(steps):
        fraction = 0.0 if steps == 1 else step / (steps - 1)
        temperature = start_temperature * (
            end_temperature / start_temperature
        ) ** fraction
        optimizer.zero_grad(set_to_none=True)
        smooth_values = soft_minimum(
            configurations,
            test_points,
            temperature,
            point_chunk_size=point_chunk_size,
        )
        loss = -smooth_values.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            configurations.clamp_(0.0, 1.0)

        should_report = step == 0 or step + 1 == steps or (step + 1) % report_every == 0
        if should_report:
            with torch.no_grad():
                hard_values = hard_minimum(
                    configurations,
                    test_points,
                    point_chunk_size=point_chunk_size,
                )
                improved = hard_values > best_values
                best_values = torch.where(improved, hard_values, best_values)
                best_configurations[improved] = configurations.detach()[improved]
            history.append(
                {
                    "step": float(step + 1),
                    "temperature": float(temperature),
                    "best_sampled_minimum": float(hard_values.max().item()),
                    "median_sampled_minimum": float(hard_values.median().item()),
                    "elapsed_seconds": float(perf_counter() - start_time),
                }
            )

    with torch.no_grad():
        final_values = hard_minimum(
            configurations, test_points, point_chunk_size=point_chunk_size
        )
        improved = final_values > best_values
        best_values = torch.where(improved, final_values, best_values)
        best_configurations[improved] = configurations.detach()[improved]
    return PopulationSearchResult(
        configurations=best_configurations.detach(),
        sampled_minima=best_values.detach(),
        history=tuple(history),
    )

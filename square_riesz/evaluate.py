"""Numerical continuous-domain evaluation of a fixed source configuration.

This module is deliberately independent of the exact-rational upstream
certifiers. It finds numerical upper witnesses and active minima; it never
claims a rigorous lower bound over the square.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import minimum_filter
from scipy.optimize import minimize, minimize_scalar

from .potential import as_sources, hessian_at_point, intensity, intensity_and_gradient


@dataclass(frozen=True)
class LocalMinimum:
    point: tuple[float, float]
    value: float
    location: str
    hessian_eigenvalues: tuple[float, float]


@dataclass(frozen=True)
class ContinuousEvaluation:
    source_count: int
    minimum: float
    darkest_point: tuple[float, float]
    grid_size: int
    local_minima: tuple[LocalMinimum, ...]
    active_minima: tuple[LocalMinimum, ...]
    active_tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _location(point: NDArray[np.float64], tolerance: float = 2e-7) -> str:
    x, y = point
    on_x = abs(x) <= tolerance or abs(x - 1.0) <= tolerance
    on_y = abs(y) <= tolerance or abs(y - 1.0) <= tolerance
    if on_x and on_y:
        return "corner"
    if on_x or on_y:
        return "edge"
    return "interior"


def _grid_points(grid_size: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return axis, np.column_stack((xx.ravel(), yy.ravel()))


def _spatially_distinct_low_points(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    count: int,
    separation: float,
) -> list[NDArray[np.float64]]:
    chosen: list[NDArray[np.float64]] = []
    for index in np.argsort(values):
        point = points[index]
        if all(np.linalg.norm(point - other) >= separation for other in chosen):
            chosen.append(point.copy())
        if len(chosen) >= count:
            break
    return chosen


def _grid_local_seeds(
    grid_points: NDArray[np.float64],
    grid_values: NDArray[np.float64],
    grid_size: int,
    limit: int,
) -> list[NDArray[np.float64]]:
    image = grid_values.reshape(grid_size, grid_size)
    local_mask = image <= minimum_filter(image, size=3, mode="nearest")
    indices = np.flatnonzero(local_mask.ravel())
    if indices.size > limit:
        order = np.argsort(grid_values[indices])[:limit]
        indices = indices[order]
    return [grid_points[index].copy() for index in indices]


def _polish_2d(
    seed: NDArray[np.float64], sources: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float] | None:
    def objective(point: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        value, gradient = intensity_and_gradient(point, sources)
        if not np.isfinite(value):
            return 1e300, gradient
        return value, gradient

    result = minimize(
        objective,
        seed,
        method="L-BFGS-B",
        jac=True,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 1000, "maxls": 50},
    )
    point = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    value = float(intensity(point, sources)[0])
    if not np.isfinite(value):
        return None
    return point, value


def _edge_candidates(
    axis: NDArray[np.float64], sources: NDArray[np.float64]
) -> Iterable[tuple[NDArray[np.float64], float]]:
    edge_maps = (
        lambda t: np.array([t, 0.0], dtype=np.float64),
        lambda t: np.array([t, 1.0], dtype=np.float64),
        lambda t: np.array([0.0, t], dtype=np.float64),
        lambda t: np.array([1.0, t], dtype=np.float64),
    )
    for edge_map in edge_maps:
        points = np.stack([edge_map(value) for value in axis])
        values = intensity(points, sources)
        local_indices = np.flatnonzero(
            (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
        ) + 1
        for index in local_indices:
            lower = float(axis[index - 1])
            upper = float(axis[index + 1])
            result = minimize_scalar(
                lambda t: float(intensity(edge_map(float(t)), sources)[0]),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-15, "maxiter": 500},
            )
            point = edge_map(float(result.x))
            yield point, float(intensity(point, sources)[0])


def _cluster_candidates(
    candidates: Iterable[tuple[NDArray[np.float64], float]],
    sources: NDArray[np.float64],
    tolerance: float,
) -> tuple[LocalMinimum, ...]:
    clusters: list[tuple[NDArray[np.float64], float]] = []
    for point, value in sorted(candidates, key=lambda item: item[1]):
        matched = None
        for index, (old_point, old_value) in enumerate(clusters):
            if np.linalg.norm(point - old_point) <= tolerance:
                matched = index
                if value < old_value:
                    clusters[index] = (point, value)
                break
        if matched is None:
            clusters.append((point, value))

    minima: list[LocalMinimum] = []
    for point, value in sorted(clusters, key=lambda item: item[1]):
        eigenvalues = np.linalg.eigvalsh(hessian_at_point(point, sources))
        minima.append(
            LocalMinimum(
                point=(float(point[0]), float(point[1])),
                value=float(value),
                location=_location(point),
                hessian_eigenvalues=(float(eigenvalues[0]), float(eigenvalues[1])),
            )
        )
    return tuple(minima)


def evaluate_continuous(
    sources: ArrayLike,
    *,
    grid_size: int = 129,
    low_seed_count: int = 64,
    max_grid_local_seeds: int | None = None,
    cluster_tolerance: float = 2e-7,
    active_tolerance: float = 2e-6,
) -> ContinuousEvaluation:
    """Find numerical local minima of a fixed configuration on the unit square.

    The returned minimum is an upper witness for the true continuous minimum,
    up to floating-point evaluation error. It is not a rigorous certificate.
    """

    source_array = as_sources(sources)
    if grid_size < 9 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer >= 9")
    if low_seed_count < 1:
        raise ValueError("low_seed_count must be positive")

    axis, grid_points = _grid_points(grid_size)
    grid_values = intensity(grid_points, source_array)
    grid_local_limit = max_grid_local_seeds or max(64, 8 * len(source_array))

    seeds = _grid_local_seeds(
        grid_points, grid_values, grid_size, grid_local_limit
    )
    seeds.extend(
        _spatially_distinct_low_points(
            grid_points,
            grid_values,
            low_seed_count,
            separation=1.5 / (grid_size - 1),
        )
    )

    candidates: list[tuple[NDArray[np.float64], float]] = []
    for seed in seeds:
        polished = _polish_2d(seed, source_array)
        if polished is not None:
            candidates.append(polished)

    candidates.extend(_edge_candidates(axis, source_array))
    corners = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64
    )
    candidates.extend((point, float(value)) for point, value in zip(corners, intensity(corners, source_array)))

    local_minima = _cluster_candidates(
        candidates, source_array, tolerance=cluster_tolerance
    )
    if not local_minima:
        raise RuntimeError("no finite minimum candidate was found")

    minimum = local_minima[0].value
    scale = max(1.0, abs(minimum))
    active = tuple(
        item for item in local_minima if item.value <= minimum + active_tolerance * scale
    )
    return ContinuousEvaluation(
        source_count=len(source_array),
        minimum=minimum,
        darkest_point=local_minima[0].point,
        grid_size=grid_size,
        local_minima=local_minima,
        active_minima=active,
        active_tolerance=active_tolerance,
    )

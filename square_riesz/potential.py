"""Riesz s=2 potential and analytic derivatives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_sources(sources: ArrayLike) -> NDArray[np.float64]:
    """Validate and return an ``(N, 2)`` float64 source array."""

    result = np.asarray(sources, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or result.shape[0] == 0:
        raise ValueError("sources must have shape (N, 2) with N >= 1")
    if not np.isfinite(result).all():
        raise ValueError("sources must be finite")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("sources must lie in the unit square")
    return result


def intensity(points: ArrayLike, sources: ArrayLike) -> NDArray[np.float64]:
    """Evaluate ``sum_i 1 / ||point-source_i||^2`` at one or more points.

    A point exactly equal to a source has infinite intensity, as required by the
    mathematical model. The returned array always has one dimension.
    """

    source_array = as_sources(sources)
    point_array = np.asarray(points, dtype=np.float64)
    point_array = np.atleast_2d(point_array)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (M, 2) or (2,)")
    if not np.isfinite(point_array).all():
        raise ValueError("points must be finite")

    delta = point_array[:, None, :] - source_array[None, :, :]
    squared_distance = np.einsum("mni,mni->mn", delta, delta)
    with np.errstate(divide="ignore"):
        return np.reciprocal(squared_distance).sum(axis=1)


def intensity_and_gradient(
    point: ArrayLike,
    sources: ArrayLike,
    *,
    singularity_floor: float = 1e-30,
) -> tuple[float, NDArray[np.float64]]:
    """Return the potential and its gradient with respect to ``point``.

    The floor is only a numerical guard for optimizer trial points. It does not
    regularize values used for scoring configurations.
    """

    source_array = as_sources(sources)
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (2,):
        raise ValueError("point must have shape (2,)")

    delta = point_array[None, :] - source_array
    squared_distance = np.einsum("ni,ni->n", delta, delta)
    if np.any(squared_distance <= singularity_floor):
        return float("inf"), np.zeros(2, dtype=np.float64)

    inverse_r2 = np.reciprocal(squared_distance)
    value = float(inverse_r2.sum())
    gradient = -2.0 * np.sum(delta * np.square(inverse_r2)[:, None], axis=0)
    return value, gradient


def hessian_at_point(point: ArrayLike, sources: ArrayLike) -> NDArray[np.float64]:
    """Return the 2x2 Hessian of the potential with respect to ``point``."""

    source_array = as_sources(sources)
    point_array = np.asarray(point, dtype=np.float64)
    if point_array.shape != (2,):
        raise ValueError("point must have shape (2,)")

    delta = point_array[None, :] - source_array
    squared_distance = np.einsum("ni,ni->n", delta, delta)
    if np.any(squared_distance == 0.0):
        return np.full((2, 2), np.nan, dtype=np.float64)

    inverse_r4 = np.reciprocal(np.square(squared_distance))
    inverse_r6 = inverse_r4 / squared_distance
    hessian = -2.0 * inverse_r4.sum() * np.eye(2, dtype=np.float64)
    hessian += 8.0 * np.einsum("n,ni,nj->ij", inverse_r6, delta, delta)
    return hessian

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
from fractions import Fraction
from itertools import combinations_with_replacement, permutations
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


def adaptive_boundary_witness_upper_bounds(
    nodes: NDArray[np.float64],
    *,
    boundary_grid_size: int = 33,
    newton_steps: int = 6,
    batch_size: int = 4096,
) -> NDArray[np.float64]:
    """Try node-specific edge witnesses derived from the box-center sources.

    The witness search itself is heuristic, but the returned value still uses
    the source-rectangle distance bound.  Thus every selected point is a valid
    witness for a later exact replay; only these float64 decisions are
    non-rigorous here.
    """

    if boundary_grid_size < 3:
        raise ValueError("boundary grid size must be at least 3")
    result = np.full(len(nodes), np.inf, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, boundary_grid_size, dtype=np.float64)
    grid_step = 1.0 / (boundary_grid_size - 1)

    for start in range(0, len(nodes), batch_size):
        block = nodes[start : start + batch_size]
        centers_x = (block[:, :, 0] + block[:, :, 1]) / 2.0
        centers_y = (block[:, :, 2] + block[:, :, 3]) / 2.0
        best = np.full(len(block), np.inf, dtype=np.float64)

        # (variable axis, fixed boundary coordinate)
        for variable_axis, fixed_coordinate in (
            (0, 0.0),
            (0, 1.0),
            (1, 0.0),
            (1, 1.0),
        ):
            variable_sources = centers_x if variable_axis == 0 else centers_y
            normal_sources = centers_y if variable_axis == 0 else centers_x
            normal_squared = (fixed_coordinate - normal_sources) ** 2
            difference = grid[None, :, None] - variable_sources[:, None, :]
            squared_distance = difference * difference + normal_squared[:, None, :]
            with np.errstate(divide="ignore"):
                center_values = np.sum(
                    np.reciprocal(squared_distance), axis=2
                )
            initial_indices = np.argmin(center_values, axis=1)
            parameter = grid[initial_indices]
            parameter_value = center_values[
                np.arange(len(block)), initial_indices
            ]

            for _ in range(newton_steps):
                difference = parameter[:, None] - variable_sources
                squared_distance = difference * difference + normal_squared
                with np.errstate(divide="ignore", invalid="ignore"):
                    gradient = np.sum(
                        -2.0 * difference / squared_distance**2, axis=1
                    )
                    curvature = np.sum(
                        -2.0 / squared_distance**2
                        + 8.0 * difference * difference / squared_distance**3,
                        axis=1,
                    )
                    proposal = parameter - gradient / curvature
                proposal = np.clip(
                    proposal,
                    np.maximum(0.0, parameter - grid_step),
                    np.minimum(1.0, parameter + grid_step),
                )
                proposal_difference = proposal[:, None] - variable_sources
                proposal_squared = (
                    proposal_difference * proposal_difference + normal_squared
                )
                with np.errstate(divide="ignore"):
                    proposal_value = np.sum(
                        np.reciprocal(proposal_squared), axis=1
                    )
                improve = proposal_value < parameter_value
                parameter[improve] = proposal[improve]
                parameter_value[improve] = proposal_value[improve]

            point_x = (
                parameter
                if variable_axis == 0
                else np.full(len(block), fixed_coordinate)
            )
            point_y = (
                np.full(len(block), fixed_coordinate)
                if variable_axis == 0
                else parameter
            )
            bound = np.zeros(len(block), dtype=np.float64)
            for source_index in range(3):
                boxes = block[:, source_index, :]
                dx = np.maximum(
                    np.maximum(boxes[:, 0] - point_x, 0.0),
                    point_x - boxes[:, 1],
                )
                dy = np.maximum(
                    np.maximum(boxes[:, 2] - point_y, 0.0),
                    point_y - boxes[:, 3],
                )
                with np.errstate(divide="ignore"):
                    bound += np.reciprocal(dx * dx + dy * dy)
            best = np.minimum(best, bound)
        result[start : start + len(block)] = best
    return result


def affine_boundary_witness_upper_bounds(
    nodes: NDArray[np.float64],
    *,
    boundary_grid_size: int = 33,
    newton_steps: int = 6,
    minimum_curvature: float = 1.0e-8,
    batch_size: int = 4096,
) -> NDArray[np.float64]:
    """Try edge witnesses that move affinely with all source coordinates.

    At the center of each source box, Newton iteration finds a stationary
    observer on each of the four edges.  Implicit differentiation of
    ``U_t(t, A) = 0`` then supplies an affine observer ``t(A)``.  The returned
    diagnostic bound encloses each affine source-observer difference by
    interval form.  Float64 is used only to estimate pruning rates; a proof
    certificate must quantize the affine coefficients and replay the same
    enclosure with exact arithmetic.
    """

    if boundary_grid_size < 3:
        raise ValueError("boundary grid size must be at least 3")
    if newton_steps < 0 or minimum_curvature <= 0.0 or batch_size < 1:
        raise ValueError("invalid affine boundary witness parameters")
    result = np.full(len(nodes), np.inf, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, boundary_grid_size, dtype=np.float64)
    grid_step = 1.0 / (boundary_grid_size - 1)

    for start in range(0, len(nodes), batch_size):
        block = nodes[start : start + batch_size]
        centers = np.empty((len(block), 3, 2), dtype=np.float64)
        radii = np.empty_like(centers)
        centers[:, :, 0] = (block[:, :, 0] + block[:, :, 1]) / 2.0
        centers[:, :, 1] = (block[:, :, 2] + block[:, :, 3]) / 2.0
        radii[:, :, 0] = (block[:, :, 1] - block[:, :, 0]) / 2.0
        radii[:, :, 1] = (block[:, :, 3] - block[:, :, 2]) / 2.0
        center_flat = centers.reshape(len(block), 6)
        radius_flat = radii.reshape(len(block), 6)
        best = np.full(len(block), np.inf, dtype=np.float64)

        # variable_axis 0 means (t, fixed) and 1 means (fixed, t).
        for variable_axis, fixed_coordinate in (
            (0, 0.0),
            (0, 1.0),
            (1, 0.0),
            (1, 1.0),
        ):
            normal_axis = 1 - variable_axis
            variable_sources = centers[:, :, variable_axis]
            normal_difference = fixed_coordinate - centers[:, :, normal_axis]
            normal_squared = normal_difference * normal_difference
            difference = grid[None, :, None] - variable_sources[:, None, :]
            squared_distance = difference * difference + normal_squared[:, None, :]
            with np.errstate(divide="ignore"):
                center_values = np.sum(np.reciprocal(squared_distance), axis=2)
            initial_indices = np.argmin(center_values, axis=1)
            parameter = grid[initial_indices]
            parameter_value = center_values[
                np.arange(len(block)), initial_indices
            ]

            for _ in range(newton_steps):
                difference = parameter[:, None] - variable_sources
                squared_distance = difference * difference + normal_squared
                with np.errstate(divide="ignore", invalid="ignore"):
                    gradient = np.sum(
                        -2.0 * difference / squared_distance**2, axis=1
                    )
                    curvature = np.sum(
                        -2.0 / squared_distance**2
                        + 8.0 * difference * difference / squared_distance**3,
                        axis=1,
                    )
                    proposal = parameter - gradient / curvature
                proposal = np.clip(
                    proposal,
                    np.maximum(0.0, parameter - grid_step),
                    np.minimum(1.0, parameter + grid_step),
                )
                proposal_difference = proposal[:, None] - variable_sources
                proposal_squared = (
                    proposal_difference * proposal_difference + normal_squared
                )
                with np.errstate(divide="ignore"):
                    proposal_value = np.sum(
                        np.reciprocal(proposal_squared), axis=1
                    )
                improve = proposal_value < parameter_value
                parameter[improve] = proposal[improve]
                parameter_value[improve] = proposal_value[improve]

            difference = parameter[:, None] - variable_sources
            squared_distance = difference * difference + normal_squared
            with np.errstate(divide="ignore", invalid="ignore"):
                per_source_curvature = (
                    -2.0 / squared_distance**2
                    + 8.0 * difference * difference / squared_distance**3
                )
                curvature = np.sum(per_source_curvature, axis=1)
                coefficients = np.zeros((len(block), 6), dtype=np.float64)
                for source_index in range(3):
                    coefficients[:, 2 * source_index + variable_axis] = (
                        per_source_curvature[:, source_index] / curvature
                    )
                    coefficients[:, 2 * source_index + normal_axis] = (
                        8.0
                        * difference[:, source_index]
                        * normal_difference[:, source_index]
                        / squared_distance[:, source_index] ** 3
                        / curvature
                    )

            parameter_radius = np.sum(
                np.abs(coefficients) * radius_flat, axis=1
            )
            valid = (
                np.isfinite(parameter)
                & np.isfinite(curvature)
                & np.all(np.isfinite(coefficients), axis=1)
                & (curvature >= minimum_curvature)
                & (parameter - parameter_radius >= 0.0)
                & (parameter + parameter_radius <= 1.0)
            )
            bound = np.zeros(len(block), dtype=np.float64)
            for source_index in range(3):
                difference_coefficients = -coefficients.copy()
                difference_coefficients[:, 2 * source_index + variable_axis] += 1.0
                difference_center = (
                    center_flat[:, 2 * source_index + variable_axis] - parameter
                )
                difference_radius = np.sum(
                    np.abs(difference_coefficients) * radius_flat, axis=1
                )
                variable_distance = np.maximum(
                    np.abs(difference_center) - difference_radius, 0.0
                )
                normal_center = (
                    center_flat[:, 2 * source_index + normal_axis]
                    - fixed_coordinate
                )
                normal_distance = np.maximum(
                    np.abs(normal_center)
                    - radius_flat[:, 2 * source_index + normal_axis],
                    0.0,
                )
                denominator = (
                    variable_distance * variable_distance
                    + normal_distance * normal_distance
                )
                with np.errstate(divide="ignore"):
                    bound += np.reciprocal(denominator)
            bound[~valid] = np.inf
            best = np.minimum(best, bound)
        result[start : start + len(block)] = best
    return result


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


def split_nodes_by_coordinate(
    nodes: NDArray[np.float64], coordinate_indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Bisect each node along its own selected source coordinate."""

    if coordinate_indices.shape != (len(nodes),):
        raise ValueError("one split coordinate is required per node")
    if np.any((coordinate_indices < 0) | (coordinate_indices >= 6)):
        raise ValueError("split coordinates must lie in 0..5")
    children = np.repeat(nodes, 2, axis=0)
    repeated_coordinates = np.repeat(coordinate_indices, 2)
    rows = np.arange(len(children))
    sources = repeated_coordinates // 2
    axes = repeated_coordinates % 2
    lower_indices = np.where(axes == 0, 0, 2)
    upper_indices = lower_indices + 1
    midpoints = (
        children[rows, sources, lower_indices]
        + children[rows, sources, upper_indices]
    ) / 2.0
    lower_children = rows % 2 == 0
    children[rows[lower_children], sources[lower_children], upper_indices[lower_children]] = (
        midpoints[lower_children]
    )
    upper_children = ~lower_children
    children[rows[upper_children], sources[upper_children], lower_indices[upper_children]] = (
        midpoints[upper_children]
    )
    return children


def node_specific_witness_upper_bounds(
    nodes: NDArray[np.float64], witnesses: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate one fixed witness against each corresponding source box."""

    if witnesses.shape != (len(nodes), 2):
        raise ValueError("one two-dimensional witness is required per node")
    total = np.zeros(len(nodes), dtype=np.float64)
    point_x = witnesses[:, 0]
    point_y = witnesses[:, 1]
    for source_index in range(3):
        boxes = nodes[:, source_index, :]
        dx = np.maximum(
            np.maximum(boxes[:, 0] - point_x, 0.0), point_x - boxes[:, 1]
        )
        dy = np.maximum(
            np.maximum(boxes[:, 2] - point_y, 0.0), point_y - boxes[:, 3]
        )
        with np.errstate(divide="ignore"):
            total += np.reciprocal(dx * dx + dy * dy)
    return total


def lookahead_split_coordinates(
    nodes: NDArray[np.float64],
    witnesses: NDArray[np.float64],
    target: float,
) -> NDArray[np.int64]:
    """Choose per-node splits by a cheap one-witness, one-level lookahead.

    Only currently widest coordinates are eligible, so the greedy lookahead
    cannot starve another dimension.  Among them, the primary score is the
    number of children that remain above ``target``; ties minimize the larger
    child bound.  This is an untrusted tree-shape heuristic: either child
    remains a complete dyadic half-box regardless of how the coordinate was
    selected.
    """

    if len(nodes) == 0:
        return np.empty(0, dtype=np.int64)
    best_coordinates = np.zeros(len(nodes), dtype=np.int64)
    best_survivor_counts = np.full(len(nodes), 3, dtype=np.int8)
    best_maximum_bounds = np.full(len(nodes), np.inf, dtype=np.float64)
    widths = np.empty((len(nodes), 6), dtype=np.float64)
    widths[:, 0::2] = nodes[:, :, 1] - nodes[:, :, 0]
    widths[:, 1::2] = nodes[:, :, 3] - nodes[:, :, 2]
    maximum_widths = np.max(widths, axis=1)
    repeated_witnesses = np.repeat(witnesses, 2, axis=0)
    for coordinate in range(6):
        children = split_nodes(nodes, coordinate)
        child_bounds = node_specific_witness_upper_bounds(
            children, repeated_witnesses
        ).reshape(len(nodes), 2)
        survivor_counts = np.sum(child_bounds > target, axis=1).astype(np.int8)
        maximum_bounds = np.max(child_bounds, axis=1)
        eligible = widths[:, coordinate] >= maximum_widths * (1.0 - 1.0e-12)
        better = eligible & (
            (survivor_counts < best_survivor_counts)
            | (
                (survivor_counts == best_survivor_counts)
                & (maximum_bounds < best_maximum_bounds)
            )
        )
        best_coordinates[better] = coordinate
        best_survivor_counts[better] = survivor_counts[better]
        best_maximum_bounds[better] = maximum_bounds[better]
    return best_coordinates


def symmetric_candidate_copies(a: float, b: float, c: float) -> NDArray[np.float64]:
    """Return all ordered source copies under D4 and source permutations."""

    base = np.asarray([[a, b], [1.0 - a, b], [0.5, c]], dtype=np.float64)
    transforms = (
        lambda point: (point[0], point[1]),
        lambda point: (1.0 - point[0], point[1]),
        lambda point: (point[0], 1.0 - point[1]),
        lambda point: (1.0 - point[0], 1.0 - point[1]),
        lambda point: (point[1], point[0]),
        lambda point: (1.0 - point[1], point[0]),
        lambda point: (point[1], 1.0 - point[0]),
        lambda point: (1.0 - point[1], 1.0 - point[0]),
    )
    copies: list[NDArray[np.float64]] = []
    seen: set[tuple[float, ...]] = set()
    for transform in transforms:
        transformed = np.asarray([transform(point) for point in base])
        for order in permutations(range(3)):
            copy = transformed[list(order)]
            # D4 maps that agree mathematically can differ by one float ulp
            # after expressions such as 1-(1-a).  Stable diagnostic deduping
            # avoids doing the same containment test twice.
            key = tuple(round(float(value), 15) for value in copy.ravel())
            if key not in seen:
                seen.add(key)
                copies.append(copy)
    return np.asarray(copies, dtype=np.float64)


def boxes_inside_local_cap(
    nodes: NDArray[np.float64],
    candidate_copies: NDArray[np.float64],
    radius: float,
) -> NDArray[np.bool_]:
    """Flag product boxes contained in a permutation/D4 copy of a local cap."""

    if radius <= 0.0:
        raise ValueError("local cap radius must be positive")
    inside = np.zeros(len(nodes), dtype=np.bool_)
    for center in candidate_copies:
        inside |= np.all(
            (nodes[:, :, 0] >= center[None, :, 0] - radius)
            & (nodes[:, :, 1] <= center[None, :, 0] + radius)
            & (nodes[:, :, 2] >= center[None, :, 1] - radius)
            & (nodes[:, :, 3] <= center[None, :, 1] + radius),
            axis=1,
        )
    return inside


def boxes_closed_by_cap_outside_slabs(
    nodes: NDArray[np.float64],
    candidate_copies: NDArray[np.float64],
    radius: float,
    target: float,
    witnesses: NDArray[np.float64],
    *,
    batch_size: int = 256,
    adaptive_boundary_witnesses: bool = False,
) -> NDArray[np.bool_]:
    """Close boxes by a local cap plus witness bounds on its outside slabs.

    For one cap center, a source box is the union of its cap intersection and
    at most twelve pieces where one of six coordinates lies below or above
    the cap interval.  The local theorem handles the first piece.  Every
    nonempty outside piece must have its own finite-witness upper bound.
    """

    if radius <= 0.0:
        raise ValueError("local cap radius must be positive")
    closed = np.zeros(len(nodes), dtype=np.bool_)
    for center in candidate_copies:
        center_coordinates = center.ravel()
        coordinate_lower = nodes[:, :, (0, 2)].reshape(len(nodes), 6)
        coordinate_upper = nodes[:, :, (1, 3)].reshape(len(nodes), 6)
        intersects = np.all(
            (coordinate_upper >= center_coordinates[None, :] - radius)
            & (coordinate_lower <= center_coordinates[None, :] + radius),
            axis=1,
        )
        candidate_indices = np.flatnonzero(intersects & ~closed)
        if not len(candidate_indices):
            continue
        candidate_nodes = nodes[candidate_indices]
        candidate_closed = np.ones(len(candidate_nodes), dtype=np.bool_)
        for coordinate in range(6):
            source = coordinate // 2
            lower_field, upper_field = ((0, 1) if coordinate % 2 == 0 else (2, 3))
            cap_lower = center_coordinates[coordinate] - radius
            cap_upper = center_coordinates[coordinate] + radius
            for side in ("lower", "upper"):
                if side == "lower":
                    nonempty = candidate_nodes[:, source, lower_field] < cap_lower
                else:
                    nonempty = candidate_nodes[:, source, upper_field] > cap_upper
                selected = np.flatnonzero(nonempty & candidate_closed)
                if not len(selected):
                    continue
                slabs = candidate_nodes[selected].copy()
                if side == "lower":
                    slabs[:, source, upper_field] = np.minimum(
                        slabs[:, source, upper_field], cap_lower
                    )
                else:
                    slabs[:, source, lower_field] = np.maximum(
                        slabs[:, source, lower_field], cap_upper
                    )
                slab_bounds = box_witness_upper_bounds(
                    slabs, witnesses, batch_size=batch_size
                )
                if adaptive_boundary_witnesses:
                    slab_bounds = np.minimum(
                        slab_bounds,
                        adaptive_boundary_witness_upper_bounds(slabs),
                    )
                candidate_closed[selected] &= slab_bounds <= target
        closed[candidate_indices[candidate_closed]] = True
    return closed


def load_local_cap(
    cap_path: Path, kkt_path: Path, target: float
) -> tuple[NDArray[np.float64], float, dict[str, object]]:
    """Load the proved cap for prototype use and check that it implies target."""

    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    kkt = json.loads(kkt_path.read_text(encoding="utf-8"))
    if cap.get("status") != "VERIFIED" or kkt.get("status") != "VERIFIED":
        raise ValueError("local cap and KKT artifacts must both be VERIFIED")
    if "center_box_linf_radius" in cap:
        radius = float(Fraction(cap["center_box_linf_radius"]))
    else:
        radius_entry = cap["declared_center_box_linf_radius"]
        # This is explicitly a float64 ROI prototype.  The older exact cap
        # radius has very large rational terms and records a conservative
        # decimal rendering for diagnostics.
        radius = float(radius_entry["decimal"])
    root_radius = Fraction(kkt["radius"])
    level_upper = Fraction(kkt["center"]["level"]) + root_radius
    if level_upper > Fraction(str(target)):
        raise ValueError(
            f"local cap only bounds boxes by root level <= {float(level_upper):.17g}, "
            f"which does not imply target {target:.17g}"
        )
    a = float(Fraction(kkt["center"]["a"]))
    b = float(Fraction(kkt["center"]["b"]))
    c = float(Fraction(kkt["center"]["c"]))
    copies = symmetric_candidate_copies(a, b, c)
    metadata: dict[str, object] = {
        "cap_path": str(cap_path),
        "kkt_path": str(kkt_path),
        "radius": radius,
        "root_level_upper": float(level_upper),
        "candidate_copy_count": len(copies),
    }
    return copies, radius, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=7.7)
    parser.add_argument("--initial-divisions", type=int, default=4)
    parser.add_argument("--witness-grid", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-levels", type=int, default=36)
    parser.add_argument("--max-active", type=int, default=2_000_000)
    parser.add_argument("--adaptive-boundary-witnesses", action="store_true")
    parser.add_argument("--affine-boundary-witnesses", action="store_true")
    parser.add_argument("--adaptive-split-lookahead", action="store_true")
    parser.add_argument("--local-cap", type=Path)
    parser.add_argument("--hybrid-local-cap-slabs", action="store_true")
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    witnesses = witness_grid(args.witness_grid)
    local_cap = None
    if args.local_cap is not None:
        local_cap = load_local_cap(
            args.local_cap, args.kkt_certificate, args.target
        )
    active = initial_unordered_cell_boxes(args.initial_divisions)
    started = perf_counter()
    levels = []
    status = "max_levels"

    for level in range(args.max_levels + 1):
        level_started = perf_counter()
        before = len(active)
        grid_witness_indices = None
        if args.adaptive_split_lookahead:
            upper_bounds, grid_witness_indices = box_witness_upper_bounds_and_argmin(
                active, witnesses, batch_size=args.batch_size
            )
        else:
            upper_bounds = box_witness_upper_bounds(
                active, witnesses, batch_size=args.batch_size
            )
        boundary_improvements = 0
        if args.adaptive_boundary_witnesses:
            boundary_bounds = adaptive_boundary_witness_upper_bounds(active)
            boundary_improvements = int(
                np.count_nonzero(boundary_bounds < upper_bounds)
            )
            upper_bounds = np.minimum(upper_bounds, boundary_bounds)
        affine_improvements = 0
        if args.affine_boundary_witnesses:
            affine_bounds = affine_boundary_witness_upper_bounds(active)
            affine_improvements = int(np.count_nonzero(affine_bounds < upper_bounds))
            upper_bounds = np.minimum(upper_bounds, affine_bounds)
        keep = upper_bounds > args.target
        active = active[keep]
        if grid_witness_indices is not None:
            grid_witness_indices = grid_witness_indices[keep]
        witness_pruned = int(before - len(active))
        local_cap_pruned = 0
        hybrid_local_cap_pruned = 0
        if local_cap is not None and len(active):
            candidate_copies, cap_radius, _ = local_cap
            inside_cap = boxes_inside_local_cap(
                active, candidate_copies, cap_radius
            )
            local_cap_pruned = int(np.count_nonzero(inside_cap))
            active = active[~inside_cap]
            if grid_witness_indices is not None:
                grid_witness_indices = grid_witness_indices[~inside_cap]
            if args.hybrid_local_cap_slabs and len(active):
                hybrid_closed = boxes_closed_by_cap_outside_slabs(
                    active,
                    candidate_copies,
                    cap_radius,
                    args.target,
                    witnesses,
                    batch_size=args.batch_size,
                    adaptive_boundary_witnesses=args.adaptive_boundary_witnesses,
                )
                hybrid_local_cap_pruned = int(np.count_nonzero(hybrid_closed))
                active = active[~hybrid_closed]
                if grid_witness_indices is not None:
                    grid_witness_indices = grid_witness_indices[~hybrid_closed]
        finite = np.isfinite(upper_bounds)
        report = {
            "level": level,
            "coordinate_split": (
                None
                if level == args.max_levels
                else (
                    "per_node_fixed_witness_lookahead"
                    if args.adaptive_split_lookahead
                    else level % 6
                )
            ),
            "boxes_before_prune": before,
            "boxes_pruned": int(before - len(active)),
            "boxes_witness_pruned": witness_pruned,
            "boxes_local_cap_pruned": local_cap_pruned,
            "boxes_hybrid_local_cap_pruned": hybrid_local_cap_pruned,
            "boxes_improved_by_adaptive_boundary_witness": boundary_improvements,
            "boxes_improved_by_affine_boundary_witness": affine_improvements,
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
        if args.adaptive_split_lookahead:
            if grid_witness_indices is None:
                raise AssertionError("lookahead witness indices were not computed")
            selected_witnesses = witnesses[grid_witness_indices]
            split_coordinates = lookahead_split_coordinates(
                active, selected_witnesses, args.target
            )
            active = split_nodes_by_coordinate(active, split_coordinates)
        else:
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
        "adaptive_boundary_witnesses": args.adaptive_boundary_witnesses,
        "affine_boundary_witnesses": args.affine_boundary_witnesses,
        "adaptive_split_lookahead": args.adaptive_split_lookahead,
        "local_cap": None if local_cap is None else local_cap[2],
        "hybrid_local_cap_slabs": args.hybrid_local_cap_slabs,
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

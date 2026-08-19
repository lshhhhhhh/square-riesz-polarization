"""Exact fixed-configuration lower certificates on the unit square.

This is a generic adaptation of the spectral exact-rational verifier in
``square-riesz-polarization-certificates`` by Satoshi Kishimoto and
contributors.  The branch-and-bound arithmetic is entirely
``fractions.Fraction`` based.  See ``THIRD_PARTY_NOTICES.md`` for the retained
BSD-3-Clause notice.

The certificate proves a lower bound for one literal finite-decimal source
configuration.  It does not prove global optimality over source positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from heapq import heappop, heappush
from typing import Callable, Iterable, Sequence


Q = Fraction
ZERO = Q(0)
ONE = Q(1)
Point = tuple[Q, Q]
Box = tuple[Q, Q, Q, Q]


@dataclass(frozen=True)
class CertificateResult:
    certified: bool
    target: Q
    splits: int
    leaves: int
    maximum_depth: int
    peak_heap_size: int
    minimum_leaf_lower_bound: Q | None
    failed_lower_bound: Q | None
    failed_box: Box | None


def parse_decimal_points(points: Iterable[Sequence[str]]) -> tuple[Point, ...]:
    """Convert literal decimal coordinate strings to exact rational points."""

    result = tuple((Q(point[0]), Q(point[1])) for point in points)
    if not result:
        raise ValueError("at least one source is required")
    if any(len(point) != 2 for point in result):
        raise ValueError("each source must have two coordinates")
    if any(not (ZERO <= coordinate <= ONE) for point in result for coordinate in point):
        raise ValueError("sources must lie in the unit square")
    if len(set(result)) != len(result):
        raise ValueError("source coordinates must be distinct")
    return result


def decimal_string(value: Q, precision: int = 70) -> str:
    """Render a rational as a non-rigorous diagnostic decimal string."""

    with localcontext() as context:
        context.prec = precision
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def potential_at(point: Point, sources: Sequence[Point]) -> Q:
    """Return the exact potential at a rational witness point."""

    x, y = point
    total = ZERO
    for source_x, source_y in sources:
        squared_distance = (x - source_x) ** 2 + (y - source_y) ** 2
        if squared_distance == 0:
            raise ValueError("potential is infinite at a source")
        total += ONE / squared_distance
    return total


def _minimum_squared_1d(source: Q, lower: Q, upper: Q) -> Q:
    if source < lower:
        return (lower - source) ** 2
    if source > upper:
        return (source - upper) ** 2
    return ZERO


def _maximum_squared_1d(source: Q, lower: Q, upper: Q) -> Q:
    return max((lower - source) ** 2, (upper - source) ** 2)


def box_lower_bound(box: Box, sources: Sequence[Point]) -> Q:
    """Return an exact lower bound for the potential throughout ``box``.

    The direct bound uses each source's maximum possible distance from the
    box.  Away from sources, a second bound uses the value and gradient at the
    box midpoint plus the global Hessian eigenvalue bound
    ``lambda_min >= -2 / r^4`` for a reciprocal-square term.
    """

    x0, x1, y0, y1 = box
    midpoint_x = (x0 + x1) / 2
    midpoint_y = (y0 + y1) / 2
    half_width = (x1 - x0) / 2
    half_height = (y1 - y0) / 2

    direct = ZERO
    source_in_box = False
    midpoint_value = ZERO
    gradient_x = ZERO
    gradient_y = ZERO
    curvature = ZERO

    for source_x, source_y in sources:
        maximum_distance_squared = _maximum_squared_1d(source_x, x0, x1)
        maximum_distance_squared += _maximum_squared_1d(source_y, y0, y1)
        direct += ONE / maximum_distance_squared

        minimum_distance_squared = _minimum_squared_1d(source_x, x0, x1)
        minimum_distance_squared += _minimum_squared_1d(source_y, y0, y1)
        if minimum_distance_squared == 0:
            source_in_box = True

        dx = midpoint_x - source_x
        dy = midpoint_y - source_y
        midpoint_distance_squared = dx * dx + dy * dy
        if midpoint_distance_squared != 0:
            midpoint_value += ONE / midpoint_distance_squared
            gradient_x += -2 * dx / midpoint_distance_squared**2
            gradient_y += -2 * dy / midpoint_distance_squared**2
        if minimum_distance_squared != 0:
            curvature += ONE / minimum_distance_squared**2

    if source_in_box:
        return direct

    taylor = midpoint_value
    taylor -= abs(gradient_x) * half_width + abs(gradient_y) * half_height
    taylor -= curvature * (half_width**2 + half_height**2)
    return max(direct, taylor)


def _square_interval(lower: Q, upper: Q) -> tuple[Q, Q]:
    minimum = ZERO if lower <= ZERO <= upper else min(lower * lower, upper * upper)
    return minimum, max(lower * lower, upper * upper)


def _multiply_intervals(a: Q, b: Q, c: Q, d: Q) -> tuple[Q, Q]:
    values = (a * c, a * d, b * c, b * d)
    return min(values), max(values)


def _divide_by_positive_interval(a: Q, b: Q, c: Q, d: Q) -> tuple[Q, Q]:
    if not (a <= b and ZERO < c <= d):
        raise ValueError("invalid interval division")
    values = (a / c, a / d, b / c, b / d)
    return min(values), max(values)


def _potential_and_gradient(point: Point, sources: Sequence[Point]) -> tuple[Q, Q, Q]:
    x, y = point
    value = gradient_x = gradient_y = ZERO
    for source_x, source_y in sources:
        dx = x - source_x
        dy = y - source_y
        squared_distance = dx * dx + dy * dy
        if squared_distance == 0:
            raise ValueError("potential is infinite at a source")
        value += ONE / squared_distance
        gradient_x += -2 * dx / squared_distance**2
        gradient_y += -2 * dy / squared_distance**2
    return value, gradient_x, gradient_y


def _componentwise_hessian_intervals(
    box: Box, sources: Sequence[Point]
) -> tuple[tuple[Q, Q], tuple[Q, Q], tuple[Q, Q]] | None:
    x0, x1, y0, y1 = box
    hessian_xx = (ZERO, ZERO)
    hessian_yy = (ZERO, ZERO)
    hessian_xy = (ZERO, ZERO)
    for source_x, source_y in sources:
        dx0, dx1 = x0 - source_x, x1 - source_x
        dy0, dy1 = y0 - source_y, y1 - source_y
        dx2_lower, dx2_upper = _square_interval(dx0, dx1)
        dy2_lower, dy2_upper = _square_interval(dy0, dy1)
        r2_lower = dx2_lower + dy2_lower
        r2_upper = dx2_upper + dy2_upper
        if r2_lower == 0:
            return None
        r6_lower = r2_lower**3
        r6_upper = r2_upper**3
        numerator_xx = (
            6 * dx2_lower - 2 * dy2_upper,
            6 * dx2_upper - 2 * dy2_lower,
        )
        numerator_yy = (
            6 * dy2_lower - 2 * dx2_upper,
            6 * dy2_upper - 2 * dx2_lower,
        )
        dxdy = _multiply_intervals(dx0, dx1, dy0, dy1)
        numerator_xy = (8 * dxdy[0], 8 * dxdy[1])
        interval_xx = _divide_by_positive_interval(
            *numerator_xx, r6_lower, r6_upper
        )
        interval_yy = _divide_by_positive_interval(
            *numerator_yy, r6_lower, r6_upper
        )
        interval_xy = _divide_by_positive_interval(
            *numerator_xy, r6_lower, r6_upper
        )
        hessian_xx = (
            hessian_xx[0] + interval_xx[0],
            hessian_xx[1] + interval_xx[1],
        )
        hessian_yy = (
            hessian_yy[0] + interval_yy[0],
            hessian_yy[1] + interval_yy[1],
        )
        hessian_xy = (
            hessian_xy[0] + interval_xy[0],
            hessian_xy[1] + interval_xy[1],
        )
    return hessian_xx, hessian_yy, hessian_xy


def box_lower_bound_componentwise(box: Box, sources: Sequence[Point]) -> Q:
    """Independent exact bound using componentwise Hessian intervals."""

    x0, x1, y0, y1 = box
    direct = ZERO
    for source_x, source_y in sources:
        _, maximum_x_squared = _square_interval(x0 - source_x, x1 - source_x)
        _, maximum_y_squared = _square_interval(y0 - source_y, y1 - source_y)
        direct += ONE / (maximum_x_squared + maximum_y_squared)

    intervals = _componentwise_hessian_intervals(box, sources)
    if intervals is None:
        return direct
    midpoint = ((x0 + x1) / 2, (y0 + y1) / 2)
    half_width = (x1 - x0) / 2
    half_height = (y1 - y0) / 2
    value, gradient_x, gradient_y = _potential_and_gradient(midpoint, sources)
    hessian_xx, hessian_yy, hessian_xy = intervals
    remainder = (
        min(ZERO, hessian_xx[0]) * half_width**2
        + min(ZERO, hessian_yy[0]) * half_height**2
    ) / 2
    remainder -= max(abs(hessian_xy[0]), abs(hessian_xy[1])) * half_width * half_height
    taylor = value - abs(gradient_x) * half_width - abs(gradient_y) * half_height
    taylor += remainder
    return max(direct, taylor)


def _certify_with_bound(
    sources: Sequence[Point],
    target: Q,
    max_splits: int,
    lower_bound: Callable[[Box, Sequence[Point]], Q],
) -> CertificateResult:
    root = (ZERO, ONE, ZERO, ONE)
    heap: list[tuple[Q, int, int, Box]] = []
    serial = 0
    heappush(heap, (lower_bound(root, sources), 0, serial, root))
    serial += 1
    splits = 0
    leaves = 0
    maximum_depth = 0
    peak_heap_size = 1
    minimum_leaf_lower_bound: Q | None = None

    while heap:
        lower, depth, _, box = heappop(heap)
        if lower >= target:
            leaves += 1
            maximum_depth = max(maximum_depth, depth)
            if minimum_leaf_lower_bound is None:
                minimum_leaf_lower_bound = lower
            else:
                minimum_leaf_lower_bound = min(minimum_leaf_lower_bound, lower)
            continue

        if splits >= max_splits:
            return CertificateResult(
                False,
                target,
                splits,
                leaves,
                maximum_depth,
                peak_heap_size,
                minimum_leaf_lower_bound,
                lower,
                box,
            )

        x0, x1, y0, y1 = box
        if x1 - x0 >= y1 - y0:
            midpoint = (x0 + x1) / 2
            children = ((x0, midpoint, y0, y1), (midpoint, x1, y0, y1))
        else:
            midpoint = (y0 + y1) / 2
            children = ((x0, x1, y0, midpoint), (x0, x1, midpoint, y1))
        for child in children:
            child_lower = lower_bound(child, sources)
            heappush(heap, (child_lower, depth + 1, serial, child))
            serial += 1
        splits += 1
        peak_heap_size = max(peak_heap_size, len(heap))

    return CertificateResult(
        True,
        target,
        splits,
        leaves,
        maximum_depth,
        peak_heap_size,
        minimum_leaf_lower_bound,
        None,
        None,
    )


def certify_fixed_configuration(
    sources: Sequence[Point],
    target: Q,
    *,
    max_splits: int = 2_000_000,
) -> CertificateResult:
    """Prove ``potential >= target`` on the complete unit square."""

    if target <= 0:
        raise ValueError("target must be positive")
    if max_splits < 0:
        raise ValueError("max_splits must be nonnegative")

    return _certify_with_bound(sources, target, max_splits, box_lower_bound)


def certify_fixed_configuration_componentwise(
    sources: Sequence[Point],
    target: Q,
    *,
    max_splits: int = 2_000_000,
) -> CertificateResult:
    """Prove the same bound with componentwise exact Hessian intervals."""

    if target <= 0:
        raise ValueError("target must be positive")
    if max_splits < 0:
        raise ValueError("max_splits must be nonnegative")
    return _certify_with_bound(
        sources, target, max_splits, box_lower_bound_componentwise
    )

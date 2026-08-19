"""Exact building blocks for the symmetric ``N=3`` local proof.

The routines in this module enclose the source coordinates as rational boxes.
They are deliberately independent of floating-point arithmetic: every accepted
leaf is justified with :class:`fractions.Fraction` interval calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Iterable, Sequence

from square_riesz.certify import Box, Point, box_lower_bound
from square_riesz.exact_interval import Interval


ZERO = Q(0)
ONE = Q(1)
SourceBox = tuple[Interval, Interval]


@dataclass(frozen=True)
class CoverResult:
    certified: bool
    splits: int
    certified_leaves: int
    excluded_leaves: int
    maximum_depth: int
    peak_stack_size: int
    minimum_lower_bound: Q | None
    failed_box: Box | None
    failed_lower_bound: Q | None


@dataclass(frozen=True)
class DerivativeCoverResult:
    certified: bool
    splits: int
    leaves: int
    maximum_depth: int
    peak_stack_size: int
    minimum_margins: dict[str, Q]
    failed_box: Box | None
    failed_margins: dict[str, Q] | None


def symmetric_source_enclosure(
    a: Q, b: Q, c: Q, radius: Q
) -> tuple[tuple[Point, ...], tuple[SourceBox, ...]]:
    """Return center sources and a box enclosing the symmetric family.

    The interval boxes forget the correlation between ``a`` and ``1-a``.  This
    only enlarges the set handled by the bounds, so every resulting inequality
    remains valid for the correlated symmetric family.
    """

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    if not (radius < a < ONE - radius):
        raise ValueError("a interval leaves the unit square")
    if not (radius < b < ONE - radius):
        raise ValueError("b interval leaves the unit square")
    if not (radius < c < ONE - radius):
        raise ValueError("c interval leaves the unit square")
    centers = ((a, b), (ONE - a, b), (Q(1, 2), c))
    boxes = (
        (Interval(a - radius, a + radius), Interval(b - radius, b + radius)),
        (
            Interval(ONE - a - radius, ONE - a + radius),
            Interval(b - radius, b + radius),
        ),
        (Interval.point(Q(1, 2)), Interval(c - radius, c + radius)),
    )
    return centers, boxes


def full_source_enclosure(
    centers: Sequence[Point], radius: Q
) -> tuple[SourceBox, ...]:
    """Enclose independent motion of every source coordinate."""

    if radius < 0:
        raise ValueError("radius must be nonnegative")
    boxes: list[SourceBox] = []
    for source_x, source_y in centers:
        if not (
            radius < source_x < ONE - radius
            and radius < source_y < ONE - radius
        ):
            raise ValueError("source box leaves the unit square")
        boxes.append(
            (
                Interval(source_x - radius, source_x + radius),
                Interval(source_y - radius, source_y + radius),
            )
        )
    return tuple(boxes)


def _square(interval: Interval) -> Interval:
    lower = interval.lower
    upper = interval.upper
    minimum = ZERO if lower <= ZERO <= upper else min(lower * lower, upper * upper)
    return Interval(minimum, max(lower * lower, upper * upper))


def _minimum_distance(first: Interval, second: Interval) -> Q:
    if first.upper < second.lower:
        return second.lower - first.upper
    if second.upper < first.lower:
        return first.lower - second.upper
    return ZERO


def _maximum_absolute_difference(first: Interval, second: Interval) -> Q:
    return max(
        abs(first.lower - second.lower),
        abs(first.lower - second.upper),
        abs(first.upper - second.lower),
        abs(first.upper - second.upper),
    )


def varying_source_box_lower_bound(
    box: Box,
    center_sources: Sequence[Point],
    source_boxes: Sequence[SourceBox],
    source_radius: Q,
) -> Q:
    """Bound the potential uniformly over an observer and source box.

    A direct maximum-distance bound is always valid.  Away from every source
    box, it is combined with the fixed-center spectral Taylor bound and an
    exact Lipschitz allowance for moving each source by ``source_radius`` in
    each coordinate.
    """

    if len(center_sources) != len(source_boxes) or not center_sources:
        raise ValueError("source centers and boxes must be nonempty and aligned")
    if source_radius < 0:
        raise ValueError("source radius must be nonnegative")
    observer_x = Interval(box[0], box[1])
    observer_y = Interval(box[2], box[3])
    direct = ZERO
    error = ZERO
    separated = True
    for (source_x, source_y), (center_x, center_y) in zip(
        source_boxes, center_sources
    ):
        max_dx = _maximum_absolute_difference(observer_x, source_x)
        max_dy = _maximum_absolute_difference(observer_y, source_y)
        maximum_r2 = max_dx * max_dx + max_dy * max_dy
        if maximum_r2 == 0:
            raise ValueError("degenerate observer and source boxes")
        direct += ONE / maximum_r2

        min_dx = _minimum_distance(observer_x, source_x)
        min_dy = _minimum_distance(observer_y, source_y)
        minimum_r2 = min_dx * min_dx + min_dy * min_dy
        if minimum_r2 == 0:
            separated = False
            continue
        if not (
            source_x.lower <= center_x <= source_x.upper
            and source_y.lower <= center_y <= source_y.upper
        ):
            raise ValueError("source center lies outside its source box")
        error += (
            2
            * source_radius
            * (max_dx + max_dy)
            / (minimum_r2 * minimum_r2)
        )
    if not separated:
        return direct
    return max(direct, box_lower_bound(box, center_sources) - error)


def potential_derivative_intervals(
    box: Box, source_boxes: Sequence[SourceBox]
) -> dict[str, Interval]:
    """Enclose ``U_x``, ``U_y``, and ``U_xx`` on a joint rational box."""

    observer_x = Interval(box[0], box[1])
    observer_y = Interval(box[2], box[3])
    ux = Interval.point(0)
    uy = Interval.point(0)
    uxx = Interval.point(0)
    for source_x, source_y in source_boxes:
        dx = observer_x - source_x
        dy = observer_y - source_y
        dx2 = _square(dx)
        dy2 = _square(dy)
        r2 = dx2 + dy2
        if r2.lower <= 0:
            raise ValueError("derivative box intersects a source box")
        r4 = Interval(r2.lower**2, r2.upper**2)
        r6 = Interval(r2.lower**3, r2.upper**3)
        ux += -2 * dx / r4
        uy += -2 * dy / r4
        uxx += (6 * dx2 - 2 * dy2) / r6
    return {"ux": ux, "uy": uy, "uxx": uxx}


def point_potential_upper_bound(
    point: Point, source_boxes: Sequence[SourceBox]
) -> Q:
    """Bound a fixed observer potential above over independent source boxes."""

    point_x = Interval.point(point[0])
    point_y = Interval.point(point[1])
    result = ZERO
    for source_x, source_y in source_boxes:
        min_dx = _minimum_distance(point_x, source_x)
        min_dy = _minimum_distance(point_y, source_y)
        minimum_r2 = min_dx * min_dx + min_dy * min_dy
        if minimum_r2 == 0:
            raise ValueError("observer point intersects a source box")
        result += ONE / minimum_r2
    return result


def _split(box: Box) -> tuple[Box, Box]:
    x0, x1, y0, y1 = box
    if x1 - x0 >= y1 - y0:
        midpoint = (x0 + x1) / 2
        return (x0, midpoint, y0, y1), (midpoint, x1, y0, y1)
    midpoint = (y0 + y1) / 2
    return (x0, x1, y0, midpoint), (x0, x1, midpoint, y1)


def _inside(box: Box, region: Box) -> bool:
    return (
        region[0] <= box[0]
        and box[1] <= region[1]
        and region[2] <= box[2]
        and box[3] <= region[3]
    )


def certify_complement_gap(
    center_sources: Sequence[Point],
    source_boxes: Sequence[SourceBox],
    source_radius: Q,
    excluded_regions: Iterable[Box],
    target: Q,
    *,
    max_splits: int = 1_000_000,
) -> CoverResult:
    """Prove ``U >= target`` outside a union of rational rectangles."""

    regions = tuple(excluded_regions)
    if target <= 0 or max_splits < 0:
        raise ValueError("invalid target or split limit")
    stack: list[tuple[Box, int]] = [((ZERO, ONE, ZERO, ONE), 0)]
    splits = certified_leaves = excluded_leaves = maximum_depth = 0
    peak_stack_size = 1
    minimum_lower_bound: Q | None = None
    while stack:
        box, depth = stack.pop()
        maximum_depth = max(maximum_depth, depth)
        if any(_inside(box, region) for region in regions):
            excluded_leaves += 1
            continue
        lower = varying_source_box_lower_bound(
            box, center_sources, source_boxes, source_radius
        )
        if lower >= target:
            certified_leaves += 1
            minimum_lower_bound = (
                lower
                if minimum_lower_bound is None
                else min(minimum_lower_bound, lower)
            )
            continue
        if splits >= max_splits:
            return CoverResult(
                False,
                splits,
                certified_leaves,
                excluded_leaves,
                maximum_depth,
                peak_stack_size,
                minimum_lower_bound,
                box,
                lower,
            )
        first, second = _split(box)
        stack.append((second, depth + 1))
        stack.append((first, depth + 1))
        splits += 1
        peak_stack_size = max(peak_stack_size, len(stack))
    return CoverResult(
        True,
        splits,
        certified_leaves,
        excluded_leaves,
        maximum_depth,
        peak_stack_size,
        minimum_lower_bound,
        None,
        None,
    )


def _condition_margins(
    derivatives: dict[str, Interval], conditions: Sequence[str]
) -> dict[str, Q]:
    available = {
        "ux_positive": derivatives["ux"].lower,
        "ux_negative": -derivatives["ux"].upper,
        "uy_positive": derivatives["uy"].lower,
        "uy_negative": -derivatives["uy"].upper,
        "uxx_positive": derivatives["uxx"].lower,
    }
    try:
        return {condition: available[condition] for condition in conditions}
    except KeyError as error:
        raise ValueError(f"unknown derivative condition: {error.args[0]}") from error


def certify_derivative_region(
    region: Box,
    source_boxes: Sequence[SourceBox],
    conditions: Sequence[str],
    *,
    max_splits: int = 1_000_000,
) -> DerivativeCoverResult:
    """Prove strict derivative signs throughout one rational rectangle."""

    if not conditions or max_splits < 0:
        raise ValueError("conditions must be nonempty and split limit nonnegative")
    stack: list[tuple[Box, int]] = [(region, 0)]
    splits = leaves = maximum_depth = 0
    peak_stack_size = 1
    minimum_margins: dict[str, Q] = {}
    while stack:
        box, depth = stack.pop()
        maximum_depth = max(maximum_depth, depth)
        margins = _condition_margins(
            potential_derivative_intervals(box, source_boxes), conditions
        )
        if all(margin > 0 for margin in margins.values()):
            leaves += 1
            for name, margin in margins.items():
                minimum_margins[name] = min(
                    margin, minimum_margins.get(name, margin)
                )
            continue
        if splits >= max_splits:
            return DerivativeCoverResult(
                False,
                splits,
                leaves,
                maximum_depth,
                peak_stack_size,
                minimum_margins,
                box,
                margins,
            )
        first, second = _split(box)
        stack.append((second, depth + 1))
        stack.append((first, depth + 1))
        splits += 1
        peak_stack_size = max(peak_stack_size, len(stack))
    return DerivativeCoverResult(
        True,
        splits,
        leaves,
        maximum_depth,
        peak_stack_size,
        minimum_margins,
        None,
        None,
    )

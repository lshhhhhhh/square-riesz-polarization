"""Exact interval second-order test for the symmetric ``N=3`` KKT root."""

from __future__ import annotations

from fractions import Fraction as Q
from itertools import combinations
from typing import Sequence

from square_riesz.exact_interval import Interval
from square_riesz.local_proof import SourceBox


ZERO = Q(0)
IntervalMatrix = list[list[Interval]]


def square_interval(value: Interval) -> Interval:
    minimum = (
        ZERO
        if value.lower <= ZERO <= value.upper
        else min(value.lower**2, value.upper**2)
    )
    return Interval(minimum, max(value.lower**2, value.upper**2))


def interval_determinant(matrix: Sequence[Sequence[Interval]]) -> Interval:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    if size == 1:
        return matrix[0][0]
    result = Interval.point(0)
    for column in range(size):
        minor = [
            [matrix[row][item] for item in range(size) if item != column]
            for row in range(1, size)
        ]
        term = matrix[0][column] * interval_determinant(minor)
        result += term if column % 2 == 0 else -term
    return result


def rational_determinant(matrix: Sequence[Sequence[Q]]) -> Q:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    if size == 1:
        return matrix[0][0]
    result = ZERO
    for column in range(size):
        minor = [
            [matrix[row][item] for item in range(size) if item != column]
            for row in range(1, size)
        ]
        term = matrix[0][column] * rational_determinant(minor)
        result += term if column % 2 == 0 else -term
    return result


def interval_matrix_inverse(
    matrix: Sequence[Sequence[Interval]],
) -> tuple[IntervalMatrix, Interval]:
    determinant = interval_determinant(matrix)
    if determinant.lower <= 0 <= determinant.upper:
        raise ValueError("interval determinant contains zero")
    size = len(matrix)
    inverse: IntervalMatrix = []
    for row in range(size):
        inverse_row: list[Interval] = []
        for column in range(size):
            minor = [
                [
                    matrix[source_row][source_column]
                    for source_column in range(size)
                    if source_column != row
                ]
                for source_row in range(size)
                if source_row != column
            ]
            cofactor = interval_determinant(minor)
            if (row + column) % 2:
                cofactor = -cofactor
            inverse_row.append(cofactor / determinant)
        inverse.append(inverse_row)
    return inverse, determinant


def fixed_observer_branch(
    point: tuple[Q, Q], source_boxes: Sequence[SourceBox]
) -> tuple[list[Interval], IntervalMatrix]:
    """Return source gradient and Hessian for one fixed observer point."""

    dimension = 2 * len(source_boxes)
    gradient = [Interval.point(0) for _ in range(dimension)]
    hessian = [
        [Interval.point(0) for _ in range(dimension)] for _ in range(dimension)
    ]
    for source_index, (source_x, source_y) in enumerate(source_boxes):
        dx = Interval.point(point[0]) - source_x
        dy = Interval.point(point[1]) - source_y
        dx2 = square_interval(dx)
        dy2 = square_interval(dy)
        r2 = dx2 + dy2
        if r2.lower <= 0:
            raise ValueError("active observer intersects a source box")
        r4 = Interval(r2.lower**2, r2.upper**2)
        r6 = Interval(r2.lower**3, r2.upper**3)
        coordinate = 2 * source_index
        gradient[coordinate] = 2 * dx / r4
        gradient[coordinate + 1] = 2 * dy / r4
        hessian[coordinate][coordinate] = (6 * dx2 - 2 * dy2) / r6
        hessian[coordinate + 1][coordinate + 1] = (6 * dy2 - 2 * dx2) / r6
        mixed = 8 * dx * dy / r6
        hessian[coordinate][coordinate + 1] = mixed
        hessian[coordinate + 1][coordinate] = mixed
    return gradient, hessian


def moving_bottom_branch(
    source_boxes: Sequence[SourceBox],
) -> tuple[list[Interval], IntervalMatrix, Interval]:
    """Return gradient and envelope Hessian of the moving bottom-edge minimum.

    The expression is evaluated at ``x=1/2``.  The enclosed KKT root is exactly
    reflection symmetric, so this is its stationary bottom-edge observer.
    """

    midpoint = (Q(1, 2), Q(0))
    gradient, fixed_hessian = fixed_observer_branch(midpoint, source_boxes)
    cross: list[Interval] = []
    observer_xx = Interval.point(0)
    for source_index, (source_x, source_y) in enumerate(source_boxes):
        dx = Interval.point(midpoint[0]) - source_x
        dy = Interval.point(midpoint[1]) - source_y
        dx2 = square_interval(dx)
        dy2 = square_interval(dy)
        r2 = dx2 + dy2
        r4 = Interval(r2.lower**2, r2.upper**2)
        r6 = Interval(r2.lower**3, r2.upper**3)
        cross.extend((2 / r4 - 8 * dx2 / r6, -8 * dx * dy / r6))
        coordinate = 2 * source_index
        observer_xx += fixed_hessian[coordinate][coordinate]
    if observer_xx.lower <= 0:
        raise ValueError("bottom observer curvature is not strictly positive")
    dimension = len(gradient)
    envelope = [
        [
            fixed_hessian[row][column]
            - cross[row] * cross[column] / observer_xx
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]
    return gradient, envelope, observer_xx


def choose_pivot_columns(center_difference_matrix: Sequence[Sequence[Q]]) -> tuple[int, ...]:
    """Choose the nonsingular 4-column minor with largest exact determinant."""

    row_count = len(center_difference_matrix)
    if row_count == 0 or any(
        len(row) != len(center_difference_matrix[0])
        for row in center_difference_matrix
    ):
        raise ValueError("difference matrix is empty or ragged")
    best_columns: tuple[int, ...] | None = None
    best_magnitude = ZERO
    for columns in combinations(range(len(center_difference_matrix[0])), row_count):
        minor = [[row[column] for column in columns] for row in center_difference_matrix]
        magnitude = abs(rational_determinant(minor))
        if magnitude > best_magnitude:
            best_magnitude = magnitude
            best_columns = columns
    if best_columns is None or best_magnitude == 0:
        raise ValueError("difference matrix does not have full row rank")
    return best_columns


def critical_basis(
    difference_matrix: IntervalMatrix, pivot_columns: Sequence[int]
) -> tuple[IntervalMatrix, Interval]:
    """Enclose a graph basis for the kernel of a full-row-rank matrix."""

    row_count = len(difference_matrix)
    column_count = len(difference_matrix[0])
    pivot_columns = tuple(pivot_columns)
    if len(pivot_columns) != row_count or len(set(pivot_columns)) != row_count:
        raise ValueError("invalid pivot columns")
    free_columns = tuple(
        column for column in range(column_count) if column not in pivot_columns
    )
    pivot_matrix = [
        [row[column] for column in pivot_columns] for row in difference_matrix
    ]
    inverse, determinant = interval_matrix_inverse(pivot_matrix)
    basis = [
        [Interval.point(0) for _ in free_columns] for _ in range(column_count)
    ]
    for basis_column, free_column in enumerate(free_columns):
        basis[free_column][basis_column] = Interval.point(1)
        right_hand_side = [
            -difference_matrix[row][free_column] for row in range(row_count)
        ]
        for pivot_row, pivot_column in enumerate(pivot_columns):
            basis[pivot_column][basis_column] = sum(
                (
                    inverse[pivot_row][row] * right_hand_side[row]
                    for row in range(row_count)
                ),
                start=Interval.point(0),
            )
    return basis, determinant


def weighted_sum_hessian(
    weights: Sequence[Interval], hessians: Sequence[IntervalMatrix]
) -> IntervalMatrix:
    if len(weights) != len(hessians) or not weights:
        raise ValueError("weights and Hessians must be nonempty and aligned")
    dimension = len(hessians[0])
    return [
        [
            sum(
                (
                    weight * hessian[row][column]
                    for weight, hessian in zip(weights, hessians)
                ),
                start=Interval.point(0),
            )
            for column in range(dimension)
        ]
        for row in range(dimension)
    ]


def projected_symmetric_2x2(
    basis: IntervalMatrix, hessian: IntervalMatrix
) -> tuple[Interval, Interval, Interval, Interval]:
    if not basis or len(basis[0]) != 2:
        raise ValueError("critical basis must have two columns")

    def entry(left: int, right: int) -> Interval:
        return sum(
            (
                basis[row][left]
                * hessian[row][column]
                * basis[column][right]
                for row in range(len(basis))
                for column in range(len(basis))
            ),
            start=Interval.point(0),
        )

    b00 = entry(0, 0)
    b01 = entry(0, 1)
    b11 = entry(1, 1)
    determinant = b00 * b11 - square_interval(b01)
    return b00, b01, b11, determinant

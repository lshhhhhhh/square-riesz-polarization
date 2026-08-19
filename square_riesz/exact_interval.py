"""Small exact-rational interval and first-order automatic differentiation tools.

This module deliberately uses only :class:`fractions.Fraction`.  It is slow,
but every endpoint operation is exact, which makes it suitable for compact
computer-assisted proofs rather than the numerical search loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    lower: Q
    upper: Q

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("interval endpoints are reversed")

    @classmethod
    def point(cls, value: Q | int) -> "Interval":
        value = Q(value)
        return cls(value, value)

    def __add__(self, other: object) -> "Interval":
        other = as_interval(other)
        return Interval(self.lower + other.lower, self.upper + other.upper)

    __radd__ = __add__

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __sub__(self, other: object) -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other: object) -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other: object) -> "Interval":
        other = as_interval(other)
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return Interval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.lower <= 0 <= self.upper:
            raise ZeroDivisionError("interval contains zero")
        return Interval(1 / self.upper, 1 / self.lower)

    def __truediv__(self, other: object) -> "Interval":
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other: object) -> "Interval":
        return as_interval(other) / self

    def __pow__(self, exponent: int) -> "Interval":
        if type(exponent) is not int or exponent < 0:
            raise ValueError("only nonnegative integer powers are supported")
        result = Interval.point(1)
        for _ in range(exponent):
            result *= self
        return result

    @property
    def width(self) -> Q:
        return self.upper - self.lower


def as_interval(value: object) -> Interval:
    if isinstance(value, Interval):
        return value
    if isinstance(value, (Q, int)):
        return Interval.point(value)
    return NotImplemented  # type: ignore[return-value]


@dataclass(frozen=True)
class IntervalAD:
    value: Interval
    derivative: tuple[Interval, ...]

    @classmethod
    def constant(cls, value: Q | int, dimension: int) -> "IntervalAD":
        return cls(Interval.point(value), (Interval.point(0),) * dimension)

    @classmethod
    def variable(
        cls, lower: Q, upper: Q, index: int, dimension: int
    ) -> "IntervalAD":
        derivative = [Interval.point(0) for _ in range(dimension)]
        derivative[index] = Interval.point(1)
        return cls(Interval(lower, upper), tuple(derivative))

    def _coerce(self, other: object) -> "IntervalAD":
        if isinstance(other, IntervalAD):
            if len(other.derivative) != len(self.derivative):
                raise ValueError("automatic-differentiation dimensions differ")
            return other
        if isinstance(other, (Q, int)):
            return IntervalAD.constant(other, len(self.derivative))
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: object) -> "IntervalAD":
        other = self._coerce(other)
        return IntervalAD(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.derivative, other.derivative)),
        )

    __radd__ = __add__

    def __neg__(self) -> "IntervalAD":
        return IntervalAD(-self.value, tuple(-item for item in self.derivative))

    def __sub__(self, other: object) -> "IntervalAD":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "IntervalAD":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "IntervalAD":
        other = self._coerce(other)
        return IntervalAD(
            self.value * other.value,
            tuple(
                left * other.value + self.value * right
                for left, right in zip(self.derivative, other.derivative)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "IntervalAD":
        inverse = self.value.reciprocal()
        return IntervalAD(
            inverse,
            tuple(-item * inverse * inverse for item in self.derivative),
        )

    def __truediv__(self, other: object) -> "IntervalAD":
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: object) -> "IntervalAD":
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> "IntervalAD":
        if type(exponent) is not int or exponent < 0:
            raise ValueError("only nonnegative integer powers are supported")
        result = IntervalAD.constant(1, len(self.derivative))
        for _ in range(exponent):
            result *= self
        return result


def invert_rational_matrix(matrix: Iterable[Iterable[Q]]) -> list[list[Q]]:
    rows = [list(map(Q, row)) for row in matrix]
    size = len(rows)
    if size == 0 or any(len(row) != size for row in rows):
        raise ValueError("matrix must be nonempty and square")
    augmented = [
        row + [Q(int(i == j)) for j in range(size)]
        for i, row in enumerate(rows)
    ]
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column]), None
        )
        if pivot is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_item
                    for value, pivot_item in zip(augmented[row], augmented[column])
                ]
    return [row[size:] for row in augmented]

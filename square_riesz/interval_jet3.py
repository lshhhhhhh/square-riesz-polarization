"""Exact-rational interval automatic differentiation through third order."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from square_riesz.exact_interval import Interval


def _zeros(dimension: int) -> tuple[Interval, ...]:
    return (Interval.point(0),) * dimension


@dataclass(frozen=True)
class IntervalJet3:
    value: Interval
    gradient: tuple[Interval, ...]
    hessian: tuple[tuple[Interval, ...], ...]
    third: tuple[tuple[tuple[Interval, ...], ...], ...]

    @property
    def dimension(self) -> int:
        return len(self.gradient)

    @classmethod
    def constant(
        cls, value: Interval | Q | int, dimension: int
    ) -> "IntervalJet3":
        interval = value if isinstance(value, Interval) else Interval.point(value)
        zero = _zeros(dimension)
        zero_matrix = tuple(zero for _ in range(dimension))
        zero_third = tuple(zero_matrix for _ in range(dimension))
        return cls(interval, zero, zero_matrix, zero_third)

    @classmethod
    def variable(
        cls, lower: Q, upper: Q, index: int, dimension: int
    ) -> "IntervalJet3":
        if not 0 <= index < dimension:
            raise ValueError("variable index is outside the jet dimension")
        result = cls.constant(Interval(lower, upper), dimension)
        gradient = list(result.gradient)
        gradient[index] = Interval.point(1)
        return cls(result.value, tuple(gradient), result.hessian, result.third)

    def _coerce(self, other: object) -> "IntervalJet3":
        if isinstance(other, IntervalJet3):
            if other.dimension != self.dimension:
                raise ValueError("jet dimensions differ")
            return other
        if isinstance(other, (Interval, Q, int)):
            return IntervalJet3.constant(other, self.dimension)
        return NotImplemented  # type: ignore[return-value]

    def __add__(self, other: object) -> "IntervalJet3":
        other = self._coerce(other)
        dimension = self.dimension
        return IntervalJet3(
            self.value + other.value,
            tuple(self.gradient[i] + other.gradient[i] for i in range(dimension)),
            tuple(
                tuple(
                    self.hessian[i][j] + other.hessian[i][j]
                    for j in range(dimension)
                )
                for i in range(dimension)
            ),
            tuple(
                tuple(
                    tuple(
                        self.third[i][j][k] + other.third[i][j][k]
                        for k in range(dimension)
                    )
                    for j in range(dimension)
                )
                for i in range(dimension)
            ),
        )

    __radd__ = __add__

    def __neg__(self) -> "IntervalJet3":
        dimension = self.dimension
        return IntervalJet3(
            -self.value,
            tuple(-self.gradient[i] for i in range(dimension)),
            tuple(
                tuple(-self.hessian[i][j] for j in range(dimension))
                for i in range(dimension)
            ),
            tuple(
                tuple(
                    tuple(-self.third[i][j][k] for k in range(dimension))
                    for j in range(dimension)
                )
                for i in range(dimension)
            ),
        )

    def __sub__(self, other: object) -> "IntervalJet3":
        return self + (-self._coerce(other))

    def __rsub__(self, other: object) -> "IntervalJet3":
        return self._coerce(other) - self

    def __mul__(self, other: object) -> "IntervalJet3":
        other = self._coerce(other)
        dimension = self.dimension
        gradient = tuple(
            self.gradient[i] * other.value + self.value * other.gradient[i]
            for i in range(dimension)
        )
        hessian = tuple(
            tuple(
                self.hessian[i][j] * other.value
                + self.gradient[i] * other.gradient[j]
                + self.gradient[j] * other.gradient[i]
                + self.value * other.hessian[i][j]
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        third = tuple(
            tuple(
                tuple(
                    self.third[i][j][k] * other.value
                    + self.hessian[i][j] * other.gradient[k]
                    + self.hessian[i][k] * other.gradient[j]
                    + self.gradient[i] * other.hessian[j][k]
                    + self.hessian[j][k] * other.gradient[i]
                    + self.gradient[j] * other.hessian[i][k]
                    + self.gradient[k] * other.hessian[i][j]
                    + self.value * other.third[i][j][k]
                    for k in range(dimension)
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        return IntervalJet3(self.value * other.value, gradient, hessian, third)

    __rmul__ = __mul__

    def reciprocal(self) -> "IntervalJet3":
        dimension = self.dimension
        inverse = self.value.reciprocal()
        inverse2 = inverse * inverse
        inverse3 = inverse2 * inverse
        inverse4 = inverse3 * inverse
        gradient = tuple(
            -self.gradient[i] * inverse2 for i in range(dimension)
        )
        hessian = tuple(
            tuple(
                2 * self.gradient[i] * self.gradient[j] * inverse3
                - self.hessian[i][j] * inverse2
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        third = tuple(
            tuple(
                tuple(
                    -6
                    * self.gradient[i]
                    * self.gradient[j]
                    * self.gradient[k]
                    * inverse4
                    + 2
                    * (
                        self.hessian[i][j] * self.gradient[k]
                        + self.hessian[i][k] * self.gradient[j]
                        + self.hessian[j][k] * self.gradient[i]
                    )
                    * inverse3
                    - self.third[i][j][k] * inverse2
                    for k in range(dimension)
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        return IntervalJet3(inverse, gradient, hessian, third)

    def __truediv__(self, other: object) -> "IntervalJet3":
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other: object) -> "IntervalJet3":
        return self._coerce(other) / self

    def __pow__(self, exponent: int) -> "IntervalJet3":
        if type(exponent) is not int or exponent < 0:
            raise ValueError("only nonnegative integer powers are supported")
        result = IntervalJet3.constant(1, self.dimension)
        for _ in range(exponent):
            result *= self
        return result

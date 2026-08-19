from __future__ import annotations

from fractions import Fraction as Q
import unittest

from square_riesz.exact_interval import Interval, IntervalAD, invert_rational_matrix


class ExactIntervalTests(unittest.TestCase):
    def test_interval_arithmetic_contains_expected_range(self) -> None:
        left = Interval(Q(-1), Q(2))
        right = Interval(Q(3), Q(4))
        self.assertEqual(left * right, Interval(Q(-4), Q(8)))
        self.assertEqual(right.reciprocal(), Interval(Q(1, 4), Q(1, 3)))

    def test_interval_ad_derivative(self) -> None:
        x = IntervalAD.variable(Q(2), Q(3), 0, 1)
        value = 1 / (x * x)
        self.assertLessEqual(value.derivative[0].lower, Q(-2, 27))
        self.assertGreaterEqual(value.derivative[0].upper, Q(-1, 4))

    def test_exact_matrix_inverse(self) -> None:
        inverse = invert_rational_matrix([[Q(2), Q(1)], [Q(1), Q(1)]])
        self.assertEqual(inverse, [[Q(1), Q(-1)], [Q(-1), Q(2)]])


if __name__ == "__main__":
    unittest.main()

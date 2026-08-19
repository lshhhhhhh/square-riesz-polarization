from __future__ import annotations

from fractions import Fraction as Q
import unittest

from square_riesz.interval_jet3 import IntervalJet3


class IntervalJet3Tests(unittest.TestCase):
    def test_reciprocal_derivatives_at_a_point(self) -> None:
        x = IntervalJet3.variable(Q(2), Q(2), 0, 1)
        inverse = x.reciprocal()
        self.assertEqual(inverse.value.lower, Q(1, 2))
        self.assertEqual(inverse.gradient[0].lower, Q(-1, 4))
        self.assertEqual(inverse.hessian[0][0].lower, Q(1, 4))
        self.assertEqual(inverse.third[0][0][0].lower, Q(-3, 8))

    def test_product_rule_through_third_order(self) -> None:
        x = IntervalJet3.variable(Q(2), Q(2), 0, 2)
        y = IntervalJet3.variable(Q(3), Q(3), 1, 2)
        product = x * y
        self.assertEqual(product.value.lower, Q(6))
        self.assertEqual(product.gradient[0].lower, Q(3))
        self.assertEqual(product.gradient[1].lower, Q(2))
        self.assertEqual(product.hessian[0][1].lower, Q(1))
        self.assertEqual(product.hessian[1][0].lower, Q(1))
        self.assertTrue(
            all(
                product.third[i][j][k].lower == 0
                and product.third[i][j][k].upper == 0
                for i in range(2)
                for j in range(2)
                for k in range(2)
            )
        )


if __name__ == "__main__":
    unittest.main()

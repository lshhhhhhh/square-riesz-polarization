"""Numerical tools for unit-square Riesz polarization."""

from .evaluate import ContinuousEvaluation, LocalMinimum, evaluate_continuous
from .potential import hessian_at_point, intensity, intensity_and_gradient

__all__ = [
    "ContinuousEvaluation",
    "LocalMinimum",
    "evaluate_continuous",
    "hessian_at_point",
    "intensity",
    "intensity_and_gradient",
]

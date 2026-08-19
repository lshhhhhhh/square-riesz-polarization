"""Exact third-derivative bound for the N=3 critical-cone Hessian.

The calculation uses exact-rational interval third-order automatic
differentiation.  It bounds the change of the five-branch weighted envelope
Hessian from the certified KKT root across a full six-dimensional source
cube, then tests the gradient-difference S-lemma matrix by interval
Sylvester minors.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from square_riesz.exact_interval import Interval
from square_riesz.interval_jet3 import IntervalJet3
from scripts.certify_n3_active_minima import _canonical_json_sha256
from square_riesz.local_optimality import (
    fixed_observer_branch,
    interval_determinant,
    moving_bottom_branch,
    moving_bottom_branch_interval,
    square_interval,
    weighted_sum_hessian,
)
from square_riesz.local_proof import full_source_enclosure, symmetric_source_enclosure


POINTS = ((Q(0), Q(1)), (Q(1), Q(1)), (Q(0), Q(0)), (Q(1), Q(0)))
DIMENSION = 7
SOURCE_INDICES = tuple(range(1, 7))


def _abs_upper(value: Interval) -> Q:
    return max(abs(value.lower), abs(value.upper))


def _potential_jet(
    observer_x: IntervalJet3,
    observer_y: IntervalJet3,
    source_coordinates: tuple[IntervalJet3, ...],
) -> IntervalJet3:
    total = IntervalJet3.constant(0, DIMENSION)
    for source in range(3):
        dx = observer_x - source_coordinates[2 * source]
        dy = observer_y - source_coordinates[2 * source + 1]
        total += (dx * dx + dy * dy).reciprocal()
    return total


def _source_variables(
    centers: tuple[Q, ...], radius: Q
) -> tuple[IntervalJet3, ...]:
    return tuple(
        IntervalJet3.variable(
            center - radius, center + radius, 1 + index, DIMENSION
        )
        for index, center in enumerate(centers)
    )


def _source_variables_from_bounds(
    bounds: tuple[Interval, ...],
) -> tuple[IntervalJet3, ...]:
    return tuple(
        IntervalJet3.variable(bound.lower, bound.upper, 1 + index, DIMENSION)
        for index, bound in enumerate(bounds)
    )


def _source_boxes_from_bounds(
    bounds: tuple[Interval, ...],
) -> tuple[tuple[Interval, Interval], ...]:
    return tuple((bounds[2 * source], bounds[2 * source + 1]) for source in range(3))


def _observer_x_gradient(
    observer_x: Q,
    source_boxes: tuple[tuple[Interval, Interval], ...],
) -> Interval:
    result = Interval.point(0)
    point_x = Interval.point(observer_x)
    point_y = Interval.point(0)
    for source_x, source_y in source_boxes:
        dx = point_x - source_x
        dy = point_y - source_y
        radius_squared = square_interval(dx) + square_interval(dy)
        radius_fourth = Interval(
            radius_squared.lower**2, radius_squared.upper**2
        )
        result += -2 * dx / radius_fourth
    return result


def _observer_root_bracket(
    source_boxes: tuple[tuple[Interval, Interval], ...],
    initial_shift: Q,
    digits: int,
) -> Interval:
    """Bracket every stationary bottom observer for one source subbox."""

    midpoint = Q(1, 2)
    shift = initial_shift
    for _ in range(8):
        lower = midpoint - shift
        upper = midpoint + shift
        lower_gradient = _observer_x_gradient(lower, source_boxes)
        upper_gradient = _observer_x_gradient(upper, source_boxes)
        if lower_gradient.upper <= 0 and upper_gradient.lower >= 0:
            break
        shift *= 2
        if shift >= Q(1, 8):
            shift = Q(1, 8)
    else:
        raise RuntimeError("could not bracket bottom-observer stationary point")
    if lower_gradient.upper > 0 or upper_gradient.lower < 0:
        raise RuntimeError("wide bottom-observer bracket does not certify signs")

    negative = lower
    positive = upper
    search_upper = upper
    for _ in range(48):
        trial = (negative + search_upper) / 2
        if _observer_x_gradient(trial, source_boxes).upper <= 0:
            negative = trial
        else:
            search_upper = trial
    search_lower = negative
    for _ in range(48):
        trial = (search_lower + positive) / 2
        if _observer_x_gradient(trial, source_boxes).lower >= 0:
            positive = trial
        else:
            search_lower = trial
    lower = _coarsen_lower(negative, digits)[0]
    upper = _round_up_nonnegative(positive, digits)
    return Interval(lower, upper)


def _fixed_third_tensor(
    point: tuple[Q, Q], source_variables: tuple[IntervalJet3, ...]
) -> list[list[list[Interval]]]:
    observer_x = IntervalJet3.constant(point[0], DIMENSION)
    observer_y = IntervalJet3.constant(point[1], DIMENSION)
    jet = _potential_jet(observer_x, observer_y, source_variables)
    return [
        [
            [jet.third[i][j][k] for k in SOURCE_INDICES]
            for j in SOURCE_INDICES
        ]
        for i in SOURCE_INDICES
    ]


def _moving_third_tensor(
    jet: IntervalJet3,
) -> list[list[list[Interval]]]:
    observer_xx = jet.hessian[0][0]
    if observer_xx.lower <= 0:
        raise ValueError("moving observer curvature interval contains zero")
    observer_derivatives = {
        k: -jet.hessian[0][k] / observer_xx for k in SOURCE_INDICES
    }
    result = []
    for i in SOURCE_INDICES:
        result_i = []
        for j in SOURCE_INDICES:
            a = jet.hessian[i][0]
            b = jet.hessian[0][j]
            result_ij = []
            for k in SOURCE_INDICES:
                xk = observer_derivatives[k]
                derivative_a = jet.third[i][0][k] + jet.third[i][0][0] * xk
                derivative_b = jet.third[0][j][k] + jet.third[0][j][0] * xk
                derivative_c = jet.third[0][0][k] + jet.third[0][0][0] * xk
                quotient_derivative = (
                    (derivative_a * b + a * derivative_b) / observer_xx
                    - a * b * derivative_c / (observer_xx * observer_xx)
                )
                result_ij.append(
                    jet.third[i][j][k]
                    + jet.third[i][j][0] * xk
                    - quotient_derivative
                )
            result_i.append(result_ij)
        result.append(result_i)
    return result


def _fraction_decimal(value: Q, digits: int = 18) -> str:
    scale = 10**digits
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    scaled = magnitude.numerator * scale // magnitude.denominator
    return f"{sign}{scaled // scale}.{scaled % scale:0{digits}d}"


def _coarsen_lower(value: Q, digits: int) -> tuple[Q, Q]:
    """Return a short decimal rational below value and its exact error."""

    scale = 10**digits
    numerator = value.numerator * scale // value.denominator
    coarse = Q(numerator, scale)
    return coarse, value - coarse


def _round_up_nonnegative(value: Q, digits: int) -> Q:
    """Round a nonnegative rational upward to a short decimal rational."""

    if value < 0:
        raise ValueError("value must be nonnegative")
    scale = 10**digits
    numerator = (value.numerator * scale + value.denominator - 1) // value.denominator
    return Q(numerator, scale)


def analyze(
    kkt_path: Path,
    radius: Q,
    eta: Q,
    multiplier: Q,
    coarse_digits: int = 18,
    subdivisions: int = 1,
) -> dict[str, object]:
    if radius <= 0 or eta <= 0 or multiplier <= 0:
        raise ValueError("radius, eta, and multiplier must be positive")
    kkt = json.loads(kkt_path.read_text(encoding="utf-8"))
    if kkt.get("status") != "VERIFIED":
        raise ValueError("KKT prerequisite is not verified")
    center = kkt["center"]
    root_radius = Q(kkt["radius"])
    if coarse_digits < 8:
        raise ValueError("coarse_digits must be at least 8")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    exact_abc = tuple(Q(center[name]) for name in ("a", "b", "c"))
    coarse_abc = tuple(_coarsen_lower(value, coarse_digits) for value in exact_abc)
    (a, a_error), (b, b_error), (c, c_error) = coarse_abc
    source_center_error = max(a_error, b_error, c_error)
    centers = (a, b, Q(1) - a, b, Q(1, 2), c)
    bound_digits = coarse_digits + 3
    root_source_radius = _round_up_nonnegative(
        root_radius + source_center_error, bound_digits
    )
    enclosure_radius = _round_up_nonnegative(
        radius + root_source_radius, bound_digits
    )
    source_variables = _source_variables(centers, enclosure_radius)
    started = perf_counter()

    def progress(label: str) -> None:
        print(f"[{perf_counter() - started:8.3f}s] {label}", flush=True)

    progress(
        "coarsened KKT source center "
        f"({coarse_digits} digits, max error "
        f"{_fraction_decimal(source_center_error, coarse_digits + 2)})"
    )

    midpoint_x = IntervalJet3.variable(Q(1, 2), Q(1, 2), 0, DIMENSION)
    zero_y = IntervalJet3.constant(0, DIMENSION)
    midpoint_jet = _potential_jet(midpoint_x, zero_y, source_variables)
    center_source_variables = _source_variables(centers, Q(0))
    center_midpoint_jet = _potential_jet(
        midpoint_x, zero_y, center_source_variables
    )
    midpoint_linear_bound = enclosure_radius * sum(
        (_abs_upper(center_midpoint_jet.hessian[0][k]) for k in SOURCE_INDICES),
        start=Q(0),
    )
    midpoint_quadratic_bound = enclosure_radius * enclosure_radius / 2 * sum(
        (
            _abs_upper(midpoint_jet.third[0][i][j])
            for i in SOURCE_INDICES
            for j in SOURCE_INDICES
        ),
        start=Q(0),
    )
    midpoint_gradient_bound = midpoint_linear_bound + midpoint_quadratic_bound
    progress("bounded centered moving-observer gradient")
    root_curvature_lower = center_midpoint_jet.hessian[0][0].lower
    if root_curvature_lower <= 0:
        raise RuntimeError("root bottom-observer curvature is not positive")
    observer_shift = _round_up_nonnegative(
        2 * midpoint_gradient_bound / root_curvature_lower, bound_digits
    )
    if observer_shift >= Q(1, 8):
        raise RuntimeError("initial centered observer enclosure left its active rectangle")
    enclosed = False
    source_boxes = full_source_enclosure(
        ((a, b), (Q(1) - a, b), (Q(1, 2), c)),
        enclosure_radius,
    )
    for _ in range(12):
        observer_x_interval = Interval(
            Q(1, 2) - observer_shift,
            Q(1, 2) + observer_shift,
        )
        _, _, observer_curvature = moving_bottom_branch_interval(
            observer_x_interval, source_boxes
        )
        curvature_lower = observer_curvature.lower
        if curvature_lower <= 0:
            break
        required_shift = _round_up_nonnegative(
            midpoint_gradient_bound / curvature_lower, bound_digits
        )
        if required_shift > observer_shift:
            observer_shift = _round_up_nonnegative(
                observer_shift * 2, bound_digits
            )
            if observer_shift >= Q(1, 8):
                break
            continue
        enclosed = True
        if observer_shift - required_shift <= Q(1, 10**30):
            break
        observer_shift = required_shift
    if not enclosed:
        raise RuntimeError(
            "centered moving-observer curvature did not close: "
            f"midpoint_gradient_bound={_fraction_decimal(midpoint_gradient_bound)}, "
            f"observer_shift={_fraction_decimal(observer_shift)}, "
            f"curvature_lower={_fraction_decimal(curvature_lower)}"
        )
    progress(
        "closed moving-observer enclosure at shift "
        f"{_fraction_decimal(observer_shift)}"
    )

    coarse_weights = {
        name: _coarsen_lower(Q(center[name]), coarse_digits)
        for name in ("top_weight", "corner_weight", "midpoint_weight")
    }
    top_weight, top_weight_error = coarse_weights["top_weight"]
    corner_weight, corner_weight_error = coarse_weights["corner_weight"]
    midpoint_weight, midpoint_weight_error = coarse_weights["midpoint_weight"]
    top_weight_radius = _round_up_nonnegative(
        root_radius + top_weight_error, bound_digits
    )
    corner_weight_radius = _round_up_nonnegative(
        root_radius + corner_weight_error, bound_digits
    )
    midpoint_weight_radius = _round_up_nonnegative(
        root_radius + midpoint_weight_error, bound_digits
    )
    weights = (
        Interval(
            top_weight / 2 - top_weight_radius / 2,
            top_weight / 2 + top_weight_radius / 2,
        ),
        Interval(
            top_weight / 2 - top_weight_radius / 2,
            top_weight / 2 + top_weight_radius / 2,
        ),
        Interval(
            corner_weight / 2 - corner_weight_radius / 2,
            corner_weight / 2 + corner_weight_radius / 2,
        ),
        Interval(
            corner_weight / 2 - corner_weight_radius / 2,
            corner_weight / 2 + corner_weight_radius / 2,
        ),
        Interval(
            midpoint_weight - midpoint_weight_radius,
            midpoint_weight + midpoint_weight_radius,
        ),
    )
    weighted_third_abs = [
        [[Q(0) for _ in range(6)] for _ in range(6)] for _ in range(6)
    ]
    max_observer_width = Q(0)
    partition_width = 2 * enclosure_radius / subdivisions
    coordinate_partitions = [
        tuple(
            Interval(
                center - enclosure_radius + part * partition_width,
                center - enclosure_radius + (part + 1) * partition_width,
            )
            for part in range(subdivisions)
        )
        for center in centers
    ]
    subbox_count = subdivisions**6
    for subbox_index, selection in enumerate(
        product(range(subdivisions), repeat=6), start=1
    ):
        coordinate_bounds = tuple(
            coordinate_partitions[coordinate][part]
            for coordinate, part in enumerate(selection)
        )
        subbox_source_variables = _source_variables_from_bounds(coordinate_bounds)
        subbox_source_boxes = _source_boxes_from_bounds(coordinate_bounds)
        observer_bracket = _observer_root_bracket(
            subbox_source_boxes, observer_shift, bound_digits
        )
        max_observer_width = max(
            max_observer_width,
            observer_bracket.upper - observer_bracket.lower,
        )
        observer_x = IntervalJet3.variable(
            observer_bracket.lower,
            observer_bracket.upper,
            0,
            DIMENSION,
        )
        moving_jet = _potential_jet(
            observer_x, zero_y, subbox_source_variables
        )
        branch_thirds = [
            _fixed_third_tensor(point, subbox_source_variables)
            for point in POINTS
        ] + [_moving_third_tensor(moving_jet)]
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    weighted_entry = sum(
                        (
                            weight * tensor[i][j][k]
                            for weight, tensor in zip(weights, branch_thirds)
                        ),
                        start=Interval.point(0),
                    )
                    weighted_third_abs[i][j][k] = max(
                        weighted_third_abs[i][j][k],
                        _abs_upper(weighted_entry),
                    )
        if subbox_index == subbox_count or subbox_index % max(1, subbox_count // 8) == 0:
            progress(
                f"bounded third tensors on source subbox "
                f"{subbox_index}/{subbox_count}"
            )
    error_matrix = [
        [
            radius
            * sum(
                (weighted_third_abs[i][j][k] for k in range(6)),
                start=Q(0),
            )
            for j in range(6)
        ]
        for i in range(6)
    ]
    spectral_error_bound = max(
        sum(row, start=Q(0)) for row in error_matrix
    )
    progress(
        "bounded weighted Hessian variation by "
        f"{_fraction_decimal(spectral_error_bound)}"
    )

    _, center_source_boxes = symmetric_source_enclosure(a, b, c, Q(0))
    center_fixed = [fixed_observer_branch(point, center_source_boxes) for point in POINTS]
    center_moving_gradient, _, _ = moving_bottom_branch(center_source_boxes)
    center_gradients = [gradient for gradient, _ in center_fixed]
    difference_matrix = [
        [
            center_gradients[row][column].lower
            - center_moving_gradient[column].lower
            for column in range(6)
        ]
        for row in range(4)
    ]
    metric = [
        [
            sum(
                (difference_matrix[row][i] * difference_matrix[row][j] for row in range(4)),
                start=Q(0),
            )
            for j in range(6)
        ]
        for i in range(6)
    ]
    progress("constructed exact gradient-difference metric")
    _, root_source_boxes = symmetric_source_enclosure(
        a, b, c, root_source_radius
    )
    root_fixed = [fixed_observer_branch(point, root_source_boxes) for point in POINTS]
    _, root_moving_hessian, _ = moving_bottom_branch(root_source_boxes)
    root_hessians = [hessian for _, hessian in root_fixed] + [root_moving_hessian]
    root_weighted = weighted_sum_hessian(weights, root_hessians)
    adjusted = [
        [
            root_weighted[i][j]
            - multiplier * metric[i][j]
            + (
                multiplier * eta * eta + spectral_error_bound
                if i == j
                else 0
            )
            for j in range(6)
        ]
        for i in range(6)
    ]
    leading_minors = []
    negative_definite = True
    for size in range(1, 7):
        progress(f"starting Sylvester minor {size}/6")
        negative_leading = [
            [-adjusted[i][j] for j in range(size)] for i in range(size)
        ]
        determinant = interval_determinant(negative_leading)
        leading_minors.append(determinant)
        negative_definite &= determinant.lower > 0
        progress(
            f"finished Sylvester minor {size}/6: lower "
            f"{_fraction_decimal(determinant.lower)}"
        )

    return {
        "schema_version": 1,
        "rigor": "exact_rational_interval_analysis",
        "status": "CLOSED" if negative_definite else "NOT_CLOSED",
        "prerequisites": {
            "kkt": {
                "path": str(kkt_path),
                "canonical_json_sha256": _canonical_json_sha256(kkt_path),
            }
        },
        "radius": str(radius),
        "eta": str(eta),
        "multiplier": str(multiplier),
        "coarse_digits": coarse_digits,
        "subdivisions_per_source_coordinate": subdivisions,
        "source_subbox_count": subbox_count,
        "source_center_coarsening_error": str(source_center_error),
        "root_source_enclosure_radius": str(root_source_radius),
        "observer_shift_bound": str(observer_shift),
        "observer_shift_decimal": _fraction_decimal(observer_shift),
        "maximum_subbox_observer_width": str(max_observer_width),
        "maximum_subbox_observer_width_decimal": _fraction_decimal(
            max_observer_width
        ),
        "spectral_error_row_sum_bound": str(spectral_error_bound),
        "spectral_error_row_sum_decimal": _fraction_decimal(spectral_error_bound),
        "matrix_perturbation_method": (
            "symmetric Hessian variation has spectral norm at most its "
            "componentwise absolute row-sum bound; certify the root "
            "S-lemma matrix plus that scalar times the identity"
        ),
        "negative_adjusted_matrix_certified": negative_definite,
        "negative_matrix_leading_minors": [
            {
                "lower": str(value.lower),
                "upper": str(value.upper),
                "lower_decimal": _fraction_decimal(value.lower),
                "upper_decimal": _fraction_decimal(value.upper),
            }
            for value in leading_minors
        ],
        "elapsed_seconds": perf_counter() - started,
        "scope": (
            "This closes only the Hessian-variation matrix used by the "
            "critical-cone terminal. A complete local theorem also needs a "
            "direction-tree cover for the complementary directions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument("--radius", default="0.001")
    parser.add_argument("--eta", default="0.7")
    parser.add_argument("--multiplier", default="1.3231")
    parser.add_argument("--coarse-digits", type=int, default=18)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--print-full-json", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/n03_hessian_lipschitz_exact.json",
    )
    args = parser.parse_args()
    result = analyze(
        args.kkt_certificate,
        Q(args.radius),
        Q(args.eta),
        Q(args.multiplier),
        args.coarse_digits,
        args.subdivisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.print_full_json:
        print(json.dumps(result, indent=2))
    else:
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "status",
                        "radius",
                        "eta",
                        "multiplier",
                        "source_subbox_count",
                        "maximum_subbox_observer_width_decimal",
                        "spectral_error_row_sum_decimal",
                        "negative_adjusted_matrix_certified",
                        "elapsed_seconds",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

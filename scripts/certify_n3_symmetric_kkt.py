"""Exact-rational Krawczyk certificate for the symmetric N=3 KKT root."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    # Exact Krawczyk endpoints can have several thousand decimal digits after
    # rational Gaussian elimination.  This is intentional certificate data.
    sys.set_int_max_str_digits(0)

from square_riesz.exact_interval import Interval, IntervalAD, invert_rational_matrix


def kkt_system(values: list[IntervalAD]) -> list[IntervalAD]:
    a, b, c, top_weight, corner_weight, midpoint_weight, level = values
    quarter = Q(1, 4)
    half = Q(1, 2)

    top_left = a * a + (1 - b) ** 2
    top_right = (1 - a) ** 2 + (1 - b) ** 2
    top_axis = quarter + (1 - c) ** 2
    top = 1 / top_left + 1 / top_right + 1 / top_axis
    top_gradient = (
        -2 * a / top_left**2 + 2 * (1 - a) / top_right**2,
        2 * (1 - b) * (1 / top_left**2 + 1 / top_right**2),
        2 * (1 - c) / top_axis**2,
    )

    bottom_left = a * a + b * b
    bottom_right = (1 - a) ** 2 + b * b
    bottom_axis = quarter + c * c
    corner = 1 / bottom_left + 1 / bottom_right + 1 / bottom_axis
    corner_gradient = (
        -2 * a / bottom_left**2 + 2 * (1 - a) / bottom_right**2,
        -2 * b * (1 / bottom_left**2 + 1 / bottom_right**2),
        -2 * c / bottom_axis**2,
    )

    midpoint_distance = (half - a) ** 2 + b * b
    midpoint = 2 / midpoint_distance + 1 / c**2
    midpoint_gradient = (
        4 * (half - a) / midpoint_distance**2,
        -4 * b / midpoint_distance**2,
        -2 / c**3,
    )
    stationarity = [
        top_weight * top_gradient[index]
        + corner_weight * corner_gradient[index]
        + midpoint_weight * midpoint_gradient[index]
        for index in range(3)
    ]
    return [
        top - level,
        corner - level,
        midpoint - level,
        *stationarity,
        top_weight + corner_weight + midpoint_weight - 1,
    ]


def evaluate(center: list[Q], radius: Q) -> list[IntervalAD]:
    dimension = len(center)
    variables = [
        IntervalAD.variable(value - radius, value + radius, index, dimension)
        for index, value in enumerate(center)
    ]
    return kkt_system(variables)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_ROOT / "data" / "candidates" / "n03_symmetric.json",
    )
    parser.add_argument("--radius", default="1e-40")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "certificates"
        / "n03_symmetric_kkt_krawczyk.json",
    )
    args = parser.parse_args()
    radius = Q(args.radius)
    if radius <= 0:
        raise SystemExit("--radius must be positive")

    payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    parameters = payload["high_precision_parameters"]
    center = [
        Q(parameters["a"]),
        Q(parameters["b"]),
        Q(parameters["c"]),
        *(Q(value) for value in parameters["active_weights"]),
        Q(parameters["level"]),
    ]
    started = perf_counter()
    point_values = evaluate(center, Q(0))
    box_values = evaluate(center, radius)
    point_jacobian = [
        [entry.lower for entry in item.derivative] for item in point_values
    ]
    inverse = invert_rational_matrix(point_jacobian)

    residual = [item.value.lower for item in point_values]
    base = [
        center[row]
        - sum(inverse[row][column] * residual[column] for column in range(7))
        for row in range(7)
    ]
    krawczyk_image: list[Interval] = []
    displacement = Interval(-radius, radius)
    for row in range(7):
        image = Interval.point(base[row])
        for column in range(7):
            coefficient = Interval.point(int(row == column))
            for inner in range(7):
                coefficient -= (
                    inverse[row][inner]
                    * box_values[inner].derivative[column]
                )
            image += coefficient * displacement
        krawczyk_image.append(image)

    strictly_inside = all(
        image.lower > midpoint - radius and image.upper < midpoint + radius
        for midpoint, image in zip(center, krawczyk_image)
    )
    weights_positive = all(center[index] - radius > 0 for index in (3, 4, 5))
    if not strictly_inside or not weights_positive:
        raise SystemExit("Krawczyk inclusion or positive-weight check failed")

    names = ["a", "b", "c", "top_weight", "corner_weight", "midpoint_weight", "level"]
    maximum_relative_displacement = max(
        max(abs(image.lower - midpoint), abs(image.upper - midpoint)) / radius
        for midpoint, image in zip(center, krawczyk_image)
    )
    result = {
        "schema_version": 1,
        "claim": (
            "The seven-equation symmetric N=3 epigraph KKT system has a "
            "unique root in the declared rational box, and all three orbit "
            "weights are positive."
        ),
        "method": "exact-rational Krawczyk inclusion",
        "status": "VERIFIED",
        "center": dict(zip(names, map(str, center))),
        "radius": str(radius),
        "krawczyk_image": {
            name: {"lower": str(image.lower), "upper": str(image.upper)}
            for name, image in zip(names, krawczyk_image)
        },
        "maximum_residual": str(max(map(abs, residual))),
        "strictly_inside": strictly_inside,
        "weights_positive": weights_positive,
        "maximum_relative_displacement_float": float(maximum_relative_displacement),
        "elapsed_seconds": perf_counter() - started,
        "scope_warning": (
            "This proves existence and uniqueness of the symmetric KKT root "
            "inside this box, not local or global optimality of P_3."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "radius": result["radius"],
                "maximum_residual": result["maximum_residual"],
                "maximum_relative_displacement_float": result[
                    "maximum_relative_displacement_float"
                ],
                "weights_positive": result["weights_positive"],
                "elapsed_seconds": result["elapsed_seconds"],
                "scope_warning": result["scope_warning"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

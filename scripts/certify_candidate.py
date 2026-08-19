"""Run an exact full-square lower certificate for a JSON candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    # Large-N exact Hessian bounds can legitimately produce rational
    # endpoints with more than Python's default 4300 decimal digits.
    sys.set_int_max_str_digits(0)

from square_riesz.certify import (
    certify_fixed_configuration_componentwise,
    decimal_string,
    parse_decimal_points,
    potential_at,
    certify_fixed_configuration,
)


def _fraction_json(value):
    if value is None:
        return None
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
            "decimal": decimal_string(value),
        }
    if isinstance(value, tuple):
        return [_fraction_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--witness-x", default="0.5")
    parser.add_argument("--witness-y", default="0")
    parser.add_argument("--max-splits", type=int, default=2_000_000)
    parser.add_argument(
        "--method", choices=("spectral", "componentwise"), default="spectral"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate = json.loads(args.input.read_text(encoding="utf-8"))
    sources = parse_decimal_points(candidate["coordinates"])
    target = Fraction(args.target)
    witness = (Fraction(args.witness_x), Fraction(args.witness_y))

    started = perf_counter()
    certificate_function = (
        certify_fixed_configuration
        if args.method == "spectral"
        else certify_fixed_configuration_componentwise
    )
    certificate = certificate_function(
        sources, target, max_splits=args.max_splits
    )
    elapsed = perf_counter() - started
    witness_value = potential_at(witness, sources)

    result = {
        "schema_version": 1,
        "method": f"exact_rational_{args.method}_hessian_full_unit_square",
        "candidate": str(args.input.resolve()),
        "n": len(sources),
        "coordinates": candidate["coordinates"],
        "target_exact": _fraction_json(target),
        "certified": certificate.certified,
        "splits": certificate.splits,
        "leaf_count": certificate.leaves,
        "maximum_depth": certificate.maximum_depth,
        "peak_heap_size": certificate.peak_heap_size,
        "minimum_leaf_lower_bound": _fraction_json(
            certificate.minimum_leaf_lower_bound
        ),
        "failed_lower_bound": _fraction_json(certificate.failed_lower_bound),
        "failed_box": _fraction_json(certificate.failed_box),
        "upper_witness": {
            "point": [_fraction_json(item) for item in witness],
            "value": _fraction_json(witness_value),
        },
        "elapsed_seconds": elapsed,
        "claim_scope": "fixed_literal_decimal_configuration_only",
        "implementation_origin": (
            "generic adaptation of the two Satoshi Kishimoto et al. "
            "BSD-3-Clause exact-rational certifiers"
        ),
    }

    output = args.output
    if output is None:
        output = (
            PROJECT_ROOT
            / "data"
            / "certificates"
            / (
                f"n{len(sources):02d}_{args.method}_target_"
                f"{args.target.replace('.', '_')}.json"
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), **result}, indent=2))
    if not certificate.certified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

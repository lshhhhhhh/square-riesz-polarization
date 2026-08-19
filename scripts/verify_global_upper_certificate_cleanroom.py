"""Clean-room verifier for the N=3 finite-witness global upper tree.

This implementation was contributed by an independent audit and then hardened
without changing its proof strategy. It deliberately avoids
``fractions.Fraction`` and the repository verifier's recursive coverage check:
all arithmetic uses integer numerator/denominator pairs, while coverage uses
prefix-freeness plus exact Kraft equality.

Unlike the original audit snapshot preserved in commit ``2ef5eb9``, this
version remains sound under ``python -O`` and selects the maximum leaf bound by
integer cross multiplication rather than a float comparison.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from itertools import combinations_with_replacement
import json
from math import gcd
from pathlib import Path
import re
from typing import Any


DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
Pair = tuple[int, int]
AxisBox = list[int]
SourceBox = list[AxisBox]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_json,
        )
    _require(type(payload) is dict, "certificate root must be a JSON object")
    return payload


def _strict_int(value: object, name: str) -> int:
    _require(type(value) is int, f"{name} must be a JSON integer")
    return value  # type: ignore[return-value]


def _parse_target(value: object) -> Pair:
    _require(
        type(value) is str and DECIMAL.fullmatch(value) is not None,
        "target must be a canonical nonnegative decimal string",
    )
    text = value  # type: ignore[assignment]
    if "." in text:
        integer, fractional = text.split(".")
        numerator = int(integer + fractional)
        denominator = 10 ** len(fractional)
    else:
        numerator, denominator = int(text), 1
    _require(numerator > 0, "target must be positive")
    common = gcd(numerator, denominator)
    return numerator // common, denominator // common


def _root_boxes(root: tuple[int, int, int], divisions: int) -> list[SourceBox]:
    boxes: list[SourceBox] = []
    for cell in root:
        _require(0 <= cell < divisions * divisions, "root cell outside grid")
        x_index, y_index = cell % divisions, cell // divisions
        boxes.append(
            [
                [x_index, x_index + 1, divisions],
                [y_index, y_index + 1, divisions],
            ]
        )
    return boxes


def _apply_path(
    root: tuple[int, int, int], path: str, divisions: int
) -> list[SourceBox]:
    boxes = _root_boxes(root, divisions)
    for depth, bit in enumerate(path):
        coordinate = depth % 6
        source, axis = divmod(coordinate, 2)
        lower, upper, denominator = boxes[source][axis]
        lower, upper, denominator = (
            2 * lower,
            2 * upper,
            2 * denominator,
        )
        _require((lower + upper) % 2 == 0, "nonintegral dyadic midpoint")
        midpoint = (lower + upper) // 2
        if bit == "0":
            upper = midpoint
        elif bit == "1":
            lower = midpoint
        else:
            raise ValueError("path contains a character other than 0 or 1")
        boxes[source][axis] = [lower, upper, denominator]
    return boxes


def _distance_numerator(
    point_numerator: int,
    point_denominator: int,
    lower: int,
    upper: int,
    box_denominator: int,
) -> int:
    """Distance numerator over ``point_denominator * box_denominator``."""

    point = point_numerator * box_denominator
    scaled_lower = lower * point_denominator
    scaled_upper = upper * point_denominator
    if point < scaled_lower:
        return scaled_lower - point
    if point > scaled_upper:
        return point - scaled_upper
    return 0


def _leaf_bound(
    root: tuple[int, int, int],
    path: str,
    witness: int,
    divisions: int,
    grid_size: int,
) -> Pair | None:
    _require(0 <= witness < grid_size * grid_size, "witness index outside grid")
    witness_denominator = grid_size - 1
    witness_x = witness % grid_size
    witness_y = witness // grid_size
    boxes = _apply_path(root, path, divisions)

    total_numerator, total_denominator = 0, 1
    for (x0, x1, x_denominator), (y0, y1, y_denominator) in boxes:
        dx = _distance_numerator(
            witness_x, witness_denominator, x0, x1, x_denominator
        )
        dy = _distance_numerator(
            witness_y, witness_denominator, y0, y1, y_denominator
        )
        x_scale = witness_denominator * x_denominator
        y_scale = witness_denominator * y_denominator
        distance_numerator = (
            dx * dx * y_scale * y_scale + dy * dy * x_scale * x_scale
        )
        distance_denominator = (x_scale * y_scale) ** 2
        if distance_numerator == 0:
            return None
        total_numerator = (
            total_numerator * distance_numerator
            + total_denominator * distance_denominator
        )
        total_denominator *= distance_numerator
    return total_numerator, total_denominator


def _verify_prefix_cover(root: tuple[int, int, int], paths: set[str]) -> None:
    _require(bool(paths), f"root {root} has no leaves")
    maximum_length = max(map(len, paths))
    kraft_numerator = sum(1 << (maximum_length - len(path)) for path in paths)
    _require(
        kraft_numerator == 1 << maximum_length,
        f"root {root} fails exact Kraft equality",
    )

    trie: dict[str, Any] = {}
    for path in sorted(paths, key=len):
        node = trie
        for bit in path:
            _require("$" not in node, f"root {root}: path extends an existing leaf")
            node = node.setdefault(bit, {})
        _require(not node, f"root {root}: path is a prefix of another leaf")
        node["$"] = True


def _greater(left: Pair, right: Pair) -> bool:
    return left[0] * right[1] > right[0] * left[1]


def _reduced(value: Pair) -> Pair:
    common = gcd(value[0], value[1])
    return value[0] // common, value[1] // common


def verify(path: Path) -> dict[str, object]:
    payload = _load_strict_json(path)
    _require(_strict_int(payload.get("n"), "n") == 3, "certificate is not for N=3")
    divisions = _strict_int(payload.get("initial_divisions"), "initial_divisions")
    grid_size = _strict_int(payload.get("witness_grid"), "witness_grid")
    _require(divisions >= 1, "initial_divisions must be positive")
    _require(grid_size >= 2, "witness_grid must be at least two")
    target = _parse_target(payload.get("target"))

    leaves = payload.get("leaves")
    _require(type(leaves) is list, "leaves must be a JSON array")
    expected_roots = set(
        combinations_with_replacement(range(divisions * divisions), 3)
    )
    by_root: dict[tuple[int, int, int], set[str]] = {}
    maximum_bound: Pair | None = None

    for index, leaf in enumerate(leaves):
        _require(type(leaf) is dict, f"leaf {index} must be a JSON object")
        root_values = leaf.get("root")
        _require(
            type(root_values) is list and len(root_values) == 3,
            f"leaf {index} root must contain three integers",
        )
        root = tuple(
            _strict_int(value, f"leaf {index} root cell") for value in root_values
        )
        _require(root in expected_roots, f"leaf {index} has a noncanonical root")
        path_bits = leaf.get("path")
        _require(type(path_bits) is str, f"leaf {index} path must be a string")
        witness = _strict_int(leaf.get("witness"), f"leaf {index} witness")
        bound = _leaf_bound(root, path_bits, witness, divisions, grid_size)
        _require(bound is not None, f"leaf {index} witness intersects a source box")
        _require(
            bound[0] * target[1] <= target[0] * bound[1],
            f"leaf {index} bound exceeds target",
        )
        if maximum_bound is None or _greater(bound, maximum_bound):
            maximum_bound = bound

        root_paths = by_root.setdefault(root, set())
        _require(path_bits not in root_paths, f"leaf {index} duplicates a path")
        root_paths.add(path_bits)

    _require(set(by_root) == expected_roots, "certificate omits an initial root")
    for root, paths in by_root.items():
        _verify_prefix_cover(root, paths)

    declared_leaf_count = _strict_int(payload.get("leaf_count"), "leaf_count")
    _require(len(leaves) == declared_leaf_count, "leaf_count metadata mismatch")
    _require(maximum_bound is not None, "certificate contains no leaves")
    maximum_bound = _reduced(maximum_bound)
    with localcontext() as context:
        context.prec = 40
        decimal = str(Decimal(maximum_bound[0]) / Decimal(maximum_bound[1]))
    return {
        "status": "VERIFIED (clean-room)",
        "target": f"{target[0]}/{target[1]}",
        "root_count": len(expected_roots),
        "leaf_count": len(leaves),
        "maximum_leaf_bound_exact": f"{maximum_bound[0]}/{maximum_bound[1]}",
        "maximum_leaf_bound_decimal": decimal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.certificate), indent=2))


if __name__ == "__main__":
    main()

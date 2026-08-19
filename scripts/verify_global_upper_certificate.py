"""Standard-library verifier for an N=3 global finite-witness upper certificate."""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
from itertools import combinations_with_replacement
import json
from pathlib import Path
import re
from time import perf_counter


Box = list[list[Q]]
DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    return value


def _strict_decimal(value: object, name: str) -> Q:
    if type(value) is not str or DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{name} must be a nonnegative decimal string")
    return Q(value)


def _root_boxes(root: tuple[int, int, int], divisions: int) -> Box:
    boxes = []
    for cell in root:
        x_index = cell % divisions
        y_index = cell // divisions
        boxes.append(
            [
                Q(x_index, divisions),
                Q(x_index + 1, divisions),
                Q(y_index, divisions),
                Q(y_index + 1, divisions),
            ]
        )
    return boxes


def _apply_path(root: tuple[int, int, int], path: str, divisions: int) -> Box:
    boxes = _root_boxes(root, divisions)
    for depth, bit in enumerate(path):
        coordinate = depth % 6
        source, axis = divmod(coordinate, 2)
        lower_index, upper_index = (0, 1) if axis == 0 else (2, 3)
        midpoint = (boxes[source][lower_index] + boxes[source][upper_index]) / 2
        if bit == "0":
            boxes[source][upper_index] = midpoint
        elif bit == "1":
            boxes[source][lower_index] = midpoint
        else:
            raise ValueError("path contains a character other than 0 or 1")
    return boxes


def _distance_1d(point: Q, lower: Q, upper: Q) -> Q:
    if point < lower:
        return lower - point
    if point > upper:
        return point - upper
    return Q(0)


def _witness_bound(boxes: Box, witness: int, grid_size: int) -> Q | None:
    if not 0 <= witness < grid_size**2:
        raise ValueError("witness index is outside the declared grid")
    denominator = grid_size - 1
    point_x = Q(witness % grid_size, denominator)
    point_y = Q(witness // grid_size, denominator)
    total = Q(0)
    for x0, x1, y0, y1 in boxes:
        dx = _distance_1d(point_x, x0, x1)
        dy = _distance_1d(point_y, y0, y1)
        squared_distance = dx * dx + dy * dy
        if squared_distance == 0:
            return None
        total += 1 / squared_distance
    return total


def _verify_prefix_cover(paths: set[str]) -> None:
    if not paths:
        raise ValueError("root has no leaves")
    for path in paths:
        if any(path[:length] in paths for length in range(len(path))):
            raise ValueError("leaf paths are not prefix-free")
    maximum_depth = max(map(len, paths))

    def covered(prefix: str) -> bool:
        if prefix in paths:
            return True
        if len(prefix) >= maximum_depth:
            return False
        return covered(prefix + "0") and covered(prefix + "1")

    if not covered(""):
        raise ValueError("leaf paths do not cover their complete root box")


def verify(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("certificate root must be a JSON object")
    if payload.get("n") != 3:
        raise ValueError("certificate is not for N=3")
    target = _strict_decimal(payload.get("target"), "target")
    divisions = _strict_int(payload.get("initial_divisions"), "initial_divisions")
    grid_size = _strict_int(payload.get("witness_grid"), "witness_grid")
    if divisions < 1 or grid_size < 2 or target <= 0:
        raise ValueError("invalid certificate parameters")

    leaves = payload.get("leaves")
    if type(leaves) is not list:
        raise ValueError("leaves must be a JSON array")
    expected_roots = set(
        combinations_with_replacement(range(divisions**2), 3)
    )
    leaves_by_root: dict[tuple[int, int, int], dict[str, int]] = {}
    maximum_bound = Q(0)
    for leaf in leaves:
        if type(leaf) is not dict:
            raise ValueError("each leaf must be a JSON object")
        root_values = leaf.get("root")
        if type(root_values) is not list or len(root_values) != 3:
            raise ValueError("leaf root must be an array of three integers")
        root = tuple(_strict_int(value, "root cell") for value in root_values)
        path_bits = leaf.get("path")
        if type(path_bits) is not str:
            raise ValueError("leaf path must be a JSON string")
        witness = _strict_int(leaf.get("witness"), "witness")
        if root not in expected_roots:
            raise ValueError("leaf has an invalid or unsorted root")
        root_leaves = leaves_by_root.setdefault(root, {})
        if path_bits in root_leaves:
            raise ValueError("duplicate leaf")
        boxes = _apply_path(root, path_bits, divisions)
        bound = _witness_bound(boxes, witness, grid_size)
        if bound is None or bound > target:
            raise ValueError("leaf witness does not prove the declared target")
        maximum_bound = max(maximum_bound, bound)
        root_leaves[path_bits] = witness

    if set(leaves_by_root) != expected_roots:
        raise ValueError("certificate omits one or more initial root boxes")
    for root_leaves in leaves_by_root.values():
        _verify_prefix_cover(set(root_leaves))
    leaf_count = _strict_int(payload.get("leaf_count"), "leaf_count")
    if len(leaves) != leaf_count:
        raise ValueError("leaf_count metadata mismatch")

    return {
        "status": "VERIFIED",
        "target": str(target),
        "root_count": len(expected_roots),
        "leaf_count": len(leaves),
        "maximum_leaf_bound_exact": str(maximum_bound),
        "maximum_leaf_bound_decimal": float(maximum_bound),
        "claim": "P_3 is at most target",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    result = verify(args.certificate)
    result["elapsed_seconds"] = perf_counter() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

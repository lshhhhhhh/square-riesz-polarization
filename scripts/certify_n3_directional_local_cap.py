"""Exact directional-cover certificate for a quantitative N=3 local cap.

Each L-infinity direction face is subdivided into rational boxes.  A box closes
if either one active observer branch decreases along every represented ray, or
the KKT-multiplier weighted average has uniformly negative ray curvature.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

from scripts.certify_n3_active_minima import _canonical_json_sha256
from square_riesz.exact_interval import Interval
from square_riesz.local_optimality import (
    fixed_observer_branch,
    moving_bottom_branch,
    moving_bottom_branch_interval,
    weighted_sum_hessian,
)
from square_riesz.local_proof import potential_derivative_intervals


POINTS = ((Q(0), Q(1)), (Q(1), Q(1)), (Q(0), Q(0)), (Q(1), Q(0)))
ZERO = Q(0)


def _fraction_string(value: object, field: str) -> Q:
    if type(value) is not str:
        raise ValueError(f"{field} must be an exact rational string")
    return Q(value)


def _exact_object(value: object, field: str) -> Q:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact object")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if type(numerator) is not str or type(denominator) is not str:
        raise ValueError(f"{field} endpoints must be strings")
    return Q(int(numerator), int(denominator))


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _coarse_enclosure(value: Q, error: Q, digits: int) -> Interval:
    scale = 10**digits
    lower_value = value - error
    upper_value = value + error
    lower = (lower_value.numerator * scale) // lower_value.denominator
    upper = -((-upper_value.numerator * scale) // upper_value.denominator)
    return Interval(Q(lower, scale), Q(upper, scale))


def _quadratic_form(
    direction: list[Interval], matrix: list[list[Interval]]
) -> Interval:
    return sum(
        (
            direction[row] * matrix[row][column] * direction[column]
            for row in range(6)
            for column in range(6)
        ),
        start=Interval.point(0),
    )


def _linear_form(
    direction: list[Interval], gradient: list[Interval]
) -> Interval:
    return sum(
        (direction[index] * gradient[index] for index in range(6)),
        start=Interval.point(0),
    )


def _offset_interval(direction: Interval, radius: Q) -> Interval:
    if direction.lower >= 0:
        return Interval(0, radius * direction.upper)
    if direction.upper <= 0:
        return Interval(radius * direction.lower, 0)
    return Interval(radius * direction.lower, radius * direction.upper)


def _ray_source_boxes(
    root_coordinates: list[Interval], direction: list[Interval], radius: Q
) -> tuple[tuple[Interval, Interval], ...]:
    coordinates = [
        root_coordinates[index] + _offset_interval(direction[index], radius)
        for index in range(6)
    ]
    return tuple(
        (coordinates[2 * source], coordinates[2 * source + 1])
        for source in range(3)
    )


@dataclass(frozen=True)
class DirectionNode:
    face_coordinate: int
    face_sign: int
    path: str
    direction: list[Interval]


class DirectionClassifier:
    def __init__(
        self,
        root_coordinates: list[Interval],
        root_source_boxes: tuple[tuple[Interval, Interval], ...],
        weights: tuple[Interval, ...],
        radius: Q,
        bottom_curvature_lower: Q,
    ) -> None:
        self.root_coordinates = root_coordinates
        self.weights = weights
        self.radius = radius
        self.bottom_curvature_lower = bottom_curvature_lower
        fixed = [fixed_observer_branch(point, root_source_boxes) for point in POINTS]
        moving_gradient, _, _ = moving_bottom_branch(root_source_boxes)
        self.root_gradients = [gradient for gradient, _ in fixed] + [moving_gradient]

    def classify(self, direction: list[Interval]) -> tuple[str, int | None, Q] | None:
        source_boxes = _ray_source_boxes(
            self.root_coordinates, direction, self.radius
        )
        fixed = [fixed_observer_branch(point, source_boxes) for point in POINTS]
        midpoint_ux = potential_derivative_intervals(
            (Q(1, 2), Q(1, 2), Q(0), Q(0)), source_boxes
        )["ux"]
        shift = max(abs(midpoint_ux.lower), abs(midpoint_ux.upper)) / (
            self.bottom_curvature_lower
        )
        if shift >= Q(1, 8):
            return None
        observer_x = Interval(Q(1, 2) - shift, Q(1, 2) + shift)
        _, moving_hessian, _ = moving_bottom_branch_interval(
            observer_x, source_boxes
        )
        hessians = [hessian for _, hessian in fixed] + [moving_hessian]

        best_branch: tuple[int, Q] | None = None
        for index, (gradient, hessian) in enumerate(
            zip(self.root_gradients, hessians)
        ):
            linear = _linear_form(direction, gradient)
            quadratic = _quadratic_form(direction, hessian)
            margin = -(
                linear.upper
                + self.radius * max(quadratic.upper, Q(0)) / 2
            )
            if margin > 0 and (best_branch is None or margin > best_branch[1]):
                best_branch = (index, margin)
        if best_branch is not None:
            return "branch", best_branch[0], best_branch[1]

        weighted_hessian = weighted_sum_hessian(self.weights, hessians)
        weighted_quadratic = _quadratic_form(direction, weighted_hessian)
        if weighted_quadratic.upper < 0:
            return "weighted", None, -weighted_quadratic.upper
        return None


def _prepare_classifier(
    kkt_path: Path,
    neighborhood_path: Path,
    radius: Q,
    coarse_digits: int,
) -> tuple[DirectionClassifier, Q]:
    kkt = json.loads(kkt_path.read_text(encoding="utf-8"))
    neighborhood = json.loads(neighborhood_path.read_text(encoding="utf-8"))
    if kkt.get("status") != "VERIFIED" or neighborhood.get("status") != "VERIFIED":
        raise ValueError("prerequisite artifacts must be VERIFIED")
    active_radius = _exact_object(
        neighborhood["full_source_linf_radius"], "full_source_linf_radius"
    )
    root_radius = _fraction_string(kkt["radius"], "radius")
    if radius <= 0 or radius + root_radius >= active_radius:
        raise ValueError("directional radius must lie inside the active neighborhood")
    curvature_artifact = _exact_object(
        neighborhood["derivative_covers"]["bottom_midpoint"]["minimum_margins"][
            "uxx_positive"
        ],
        "bottom uxx margin",
    )
    bottom_curvature_lower = Q(3)
    if curvature_artifact <= bottom_curvature_lower:
        raise ValueError("the simple bottom-curvature bound is not justified")

    center = kkt["center"]
    a = _coarse_enclosure(Q(center["a"]), root_radius, coarse_digits)
    b = _coarse_enclosure(Q(center["b"]), root_radius, coarse_digits)
    c = _coarse_enclosure(Q(center["c"]), root_radius, coarse_digits)
    half = Interval.point(Q(1, 2))
    root_coordinates = [a, b, Interval.point(1) - a, b, half, c]
    root_source_boxes = tuple(
        (root_coordinates[2 * source], root_coordinates[2 * source + 1])
        for source in range(3)
    )
    top_weight = _coarse_enclosure(
        Q(center["top_weight"]) / 2, root_radius, coarse_digits
    )
    corner_weight = _coarse_enclosure(
        Q(center["corner_weight"]) / 2, root_radius, coarse_digits
    )
    midpoint_weight = _coarse_enclosure(
        Q(center["midpoint_weight"]), root_radius, coarse_digits
    )
    weights = (
        top_weight,
        top_weight,
        corner_weight,
        corner_weight,
        midpoint_weight,
    )
    return (
        DirectionClassifier(
            root_coordinates,
            root_source_boxes,
            weights,
            radius,
            bottom_curvature_lower,
        ),
        root_radius,
    )


def _direction_from_root_path(root_id: int, path: str) -> list[Interval]:
    if type(root_id) is not int or not 0 <= root_id < 12:
        raise ValueError("root_id must be an integer from 0 through 11")
    if type(path) is not str or any(bit not in "01" for bit in path):
        raise ValueError("path must be a binary string")
    face_coordinate = root_id // 2
    face_sign = -1 if root_id % 2 == 0 else 1
    direction = [Interval(Q(-1), Q(1)) for _ in range(6)]
    direction[face_coordinate] = Interval.point(face_sign)
    free = [index for index in range(6) if index != face_coordinate]
    for depth, bit in enumerate(path):
        split_coordinate = free[depth % 5]
        value = direction[split_coordinate]
        midpoint = (value.lower + value.upper) / 2
        direction[split_coordinate] = (
            Interval(value.lower, midpoint)
            if bit == "0"
            else Interval(midpoint, value.upper)
        )
    return direction


def _audit_candidate_coverage(leaves: list[object]) -> None:
    paths_by_root: list[list[str]] = [[] for _ in range(12)]
    seen: set[tuple[int, str]] = set()
    for raw_leaf in leaves:
        if type(raw_leaf) is not dict:
            raise ValueError("candidate leaves must be objects")
        root_id = raw_leaf.get("root_id")
        path = raw_leaf.get("path")
        _direction_from_root_path(root_id, path)
        key = (root_id, path)
        if key in seen:
            raise ValueError("candidate tree contains a duplicate leaf")
        seen.add(key)
        paths_by_root[root_id].append(path)
    for root_id, paths in enumerate(paths_by_root):
        ordered = sorted(paths)
        for index, path in enumerate(ordered[:-1]):
            if ordered[index + 1].startswith(path):
                raise ValueError(f"root {root_id} is not prefix-free")
        kraft = sum((Q(1, 2 ** len(path)) for path in paths), start=Q(0))
        if kraft != 1:
            raise ValueError(f"root {root_id} has exact Kraft sum {kraft}, not 1")


_WORKER_CLASSIFIER: DirectionClassifier | None = None


def _initialize_worker(
    kkt_path: str, neighborhood_path: str, radius: str, coarse_digits: int
) -> None:
    global _WORKER_CLASSIFIER
    _WORKER_CLASSIFIER = _prepare_classifier(
        Path(kkt_path), Path(neighborhood_path), Q(radius), coarse_digits
    )[0]


def _verify_candidate_leaf(raw_leaf: dict[str, object]) -> tuple[bool, str, int | None, float]:
    if _WORKER_CLASSIFIER is None:
        raise RuntimeError("worker classifier was not initialized")
    direction = _direction_from_root_path(raw_leaf["root_id"], raw_leaf["path"])
    result = _WORKER_CLASSIFIER.classify(direction)
    if result is None:
        return False, "", None, 0.0
    criterion, branch, margin = result
    return True, criterion, branch, float(margin)


def verify_candidate_tree(
    candidate_path: Path,
    kkt_path: Path,
    neighborhood_path: Path,
    *,
    coarse_digits: int,
    workers: int,
    candidate_limit: int | None,
) -> dict[str, object]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "closed_in_float_interval":
        raise ValueError("candidate tree did not close in its diagnostic generator")
    radius = Q(str(candidate["radius"]))
    raw_leaves = candidate.get("leaves")
    if type(raw_leaves) is not list:
        raise ValueError("candidate leaves are missing")
    _audit_candidate_coverage(raw_leaves)
    selected = raw_leaves if candidate_limit is None else raw_leaves[:candidate_limit]
    started = perf_counter()
    actual_leaves = []
    failures = []
    minimum_margin: float | None = None
    initializer_arguments = (
        str(kkt_path),
        str(neighborhood_path),
        str(radius),
        coarse_digits,
    )
    if workers == 1:
        _initialize_worker(*initializer_arguments)
        results = map(_verify_candidate_leaf, selected)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=initializer_arguments,
        )
        results = executor.map(_verify_candidate_leaf, selected, chunksize=8)
    try:
        for index, (raw_leaf, result) in enumerate(zip(selected, results), start=1):
            passed, criterion, branch, margin = result
            if passed:
                minimum_margin = margin if minimum_margin is None else min(
                    minimum_margin, margin
                )
                actual_leaves.append(
                    {
                        "root_id": raw_leaf["root_id"],
                        "path": raw_leaf["path"],
                        "criterion": criterion,
                        "branch": branch,
                    }
                )
            else:
                failures.append(
                    {"root_id": raw_leaf["root_id"], "path": raw_leaf["path"]}
                )
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "verified": index,
                            "total": len(selected),
                            "failures": len(failures),
                        }
                    ),
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()
    complete = candidate_limit is None
    status = "VERIFIED" if complete and not failures else "INCOMPLETE"
    _, root_radius = _prepare_classifier(
        kkt_path, neighborhood_path, radius, coarse_digits
    )
    return {
        "schema_version": 1,
        "claim": (
            "Every non-root source configuration within the declared root-centered "
            "L-infinity radius has polarization strictly below the symmetric KKT root."
        ),
        "status": status,
        "method": "independent exact-rational replay of a GPU-proposed L-infinity direction tree",
        "prerequisites": {
            "kkt": {
                "path": _display_path(kkt_path),
                "canonical_json_sha256": _canonical_json_sha256(kkt_path),
            },
            "active_neighborhood": {
                "path": _display_path(neighborhood_path),
                "canonical_json_sha256": _canonical_json_sha256(neighborhood_path),
            },
        },
        "candidate_provenance": {
            "method": "untrusted GPU binary64 interval-formula tree proposal",
            "required_for_verification": False,
        },
        "root_centered_linf_radius": str(radius),
        "center_box_linf_radius": str(radius - root_radius),
        "coarse_enclosure_digits": coarse_digits,
        "candidate_leaf_count": len(raw_leaves),
        "checked_leaf_count": len(selected),
        "failure_count": len(failures),
        "failures": failures[:100],
        "minimum_float_rendered_exact_margin": minimum_margin,
        "leaves": actual_leaves if status == "VERIFIED" else [],
        "elapsed_seconds": perf_counter() - started,
        "scope_warning": (
            "The GPU candidate decisions are untrusted; status is VERIFIED only "
            "after exact replay and exact prefix/Kraft coverage checks."
        ),
    }


def build_certificate(
    kkt_path: Path,
    neighborhood_path: Path,
    *,
    radius: Q,
    coarse_digits: int,
    max_depth: int,
    max_leaves: int,
) -> dict[str, object]:
    kkt = json.loads(kkt_path.read_text(encoding="utf-8"))
    neighborhood = json.loads(neighborhood_path.read_text(encoding="utf-8"))
    if kkt.get("status") != "VERIFIED" or neighborhood.get("status") != "VERIFIED":
        raise ValueError("prerequisite artifacts must be VERIFIED")
    active_radius = _exact_object(
        neighborhood["full_source_linf_radius"], "full_source_linf_radius"
    )
    root_radius = _fraction_string(kkt["radius"], "radius")
    if radius <= 0 or radius + root_radius >= active_radius:
        raise ValueError("directional radius must lie inside the active neighborhood")
    curvature_artifact = _exact_object(
        neighborhood["derivative_covers"]["bottom_midpoint"]["minimum_margins"][
            "uxx_positive"
        ],
        "bottom uxx margin",
    )
    bottom_curvature_lower = Q(3)
    if curvature_artifact <= bottom_curvature_lower:
        raise ValueError("the simple bottom-curvature bound is not justified")

    center = kkt["center"]
    a = _coarse_enclosure(Q(center["a"]), root_radius, coarse_digits)
    b = _coarse_enclosure(Q(center["b"]), root_radius, coarse_digits)
    c = _coarse_enclosure(Q(center["c"]), root_radius, coarse_digits)
    half = Interval.point(Q(1, 2))
    root_coordinates = [a, b, Interval.point(1) - a, b, half, c]
    root_source_boxes = tuple(
        (root_coordinates[2 * source], root_coordinates[2 * source + 1])
        for source in range(3)
    )
    top_weight = _coarse_enclosure(
        Q(center["top_weight"]) / 2, root_radius, coarse_digits
    )
    corner_weight = _coarse_enclosure(
        Q(center["corner_weight"]) / 2, root_radius, coarse_digits
    )
    midpoint_weight = _coarse_enclosure(
        Q(center["midpoint_weight"]), root_radius, coarse_digits
    )
    weights = (
        top_weight,
        top_weight,
        corner_weight,
        corner_weight,
        midpoint_weight,
    )
    classifier = DirectionClassifier(
        root_coordinates,
        root_source_boxes,
        weights,
        radius,
        bottom_curvature_lower,
    )

    started = perf_counter()
    stack: list[DirectionNode] = []
    for face_coordinate in range(6):
        for face_sign in (-1, 1):
            direction = [Interval(Q(-1), Q(1)) for _ in range(6)]
            direction[face_coordinate] = Interval.point(face_sign)
            stack.append(DirectionNode(face_coordinate, face_sign, "", direction))
    leaves: list[dict[str, object]] = []
    unresolved = 0
    maximum_reached_depth = 0
    criterion_counts = {"branch": 0, "weighted": 0}
    while stack:
        node = stack.pop()
        maximum_reached_depth = max(maximum_reached_depth, len(node.path))
        result = classifier.classify(node.direction)
        if result is not None:
            criterion, branch, margin = result
            criterion_counts[criterion] += 1
            leaves.append(
                {
                    "face_coordinate": node.face_coordinate,
                    "face_sign": node.face_sign,
                    "path": node.path,
                    "criterion": criterion,
                    "branch": branch,
                    "margin": str(margin),
                }
            )
            if len(leaves) >= max_leaves:
                unresolved += len(stack)
                break
            continue
        if len(node.path) >= max_depth:
            unresolved += 1
            continue
        free = [index for index in range(6) if index != node.face_coordinate]
        split_coordinate = free[len(node.path) % 5]
        interval = node.direction[split_coordinate]
        midpoint = (interval.lower + interval.upper) / 2
        lower_direction = list(node.direction)
        upper_direction = list(node.direction)
        lower_direction[split_coordinate] = Interval(interval.lower, midpoint)
        upper_direction[split_coordinate] = Interval(midpoint, interval.upper)
        stack.append(
            DirectionNode(
                node.face_coordinate,
                node.face_sign,
                node.path + "1",
                upper_direction,
            )
        )
        stack.append(
            DirectionNode(
                node.face_coordinate,
                node.face_sign,
                node.path + "0",
                lower_direction,
            )
        )

    status = "VERIFIED" if unresolved == 0 and not stack else "INCOMPLETE"
    return {
        "schema_version": 1,
        "claim": (
            "Every non-root source configuration within the declared root-centered "
            "L-infinity radius has polarization strictly below the symmetric KKT root."
        ),
        "status": status,
        "method": "exact-rational L-infinity direction cover with branch-linear or weighted-curvature ray certificates",
        "prerequisites": {
            "kkt": {
                "path": _display_path(kkt_path),
                "canonical_json_sha256": _canonical_json_sha256(kkt_path),
            },
            "active_neighborhood": {
                "path": _display_path(neighborhood_path),
                "canonical_json_sha256": _canonical_json_sha256(neighborhood_path),
            },
        },
        "root_centered_linf_radius": str(radius),
        "center_box_linf_radius": str(radius - root_radius),
        "coarse_enclosure_digits": coarse_digits,
        "bottom_curvature_lower": str(bottom_curvature_lower),
        "max_depth": max_depth,
        "max_leaves": max_leaves,
        "maximum_reached_depth": maximum_reached_depth,
        "leaf_count": len(leaves),
        "unresolved_count": unresolved + len(stack),
        "criterion_counts": criterion_counts,
        "leaves": leaves,
        "elapsed_seconds": perf_counter() - started,
        "scope_warning": "The direction tree must be independently replayed before publication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kkt-certificate",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_symmetric_kkt_krawczyk.json",
    )
    parser.add_argument(
        "--active-neighborhood",
        type=Path,
        default=PROJECT_ROOT / "data/certificates/n03_full_source_neighborhood.json",
    )
    parser.add_argument("--radius", default="1e-4")
    parser.add_argument("--coarse-digits", type=int, default=18)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--max-leaves", type=int, default=100000)
    parser.add_argument("--candidate-tree", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs/n03_directional_local_cap.json",
    )
    args = parser.parse_args()
    if args.candidate_tree is None:
        result = build_certificate(
            args.kkt_certificate,
            args.active_neighborhood,
            radius=Q(args.radius),
            coarse_digits=args.coarse_digits,
            max_depth=args.max_depth,
            max_leaves=args.max_leaves,
        )
    else:
        result = verify_candidate_tree(
            args.candidate_tree,
            args.kkt_certificate,
            args.active_neighborhood,
            coarse_digits=args.coarse_digits,
            workers=args.workers,
            candidate_limit=args.candidate_limit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "radius": result["root_centered_linf_radius"],
                "leaf_count": result.get("leaf_count", result.get("checked_leaf_count")),
                "unresolved_count": result.get("unresolved_count", result.get("failure_count")),
                "maximum_reached_depth": result.get("maximum_reached_depth"),
                "criterion_counts": result.get("criterion_counts"),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

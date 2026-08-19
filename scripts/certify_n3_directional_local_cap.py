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
    square_interval,
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


def _coarse_lower_point(value: Q, digits: int) -> Q:
    scale = 10**digits
    return Q(value.numerator * scale // value.denominator, scale)


def _critical_difference_matrix(
    center: dict[str, object], coarse_digits: int
) -> tuple[tuple[Q, ...], ...]:
    a = _coarse_lower_point(Q(center["a"]), coarse_digits)
    b = _coarse_lower_point(Q(center["b"]), coarse_digits)
    c = _coarse_lower_point(Q(center["c"]), coarse_digits)
    coordinates = (a, b, Q(1) - a, b, Q(1, 2), c)
    source_boxes = tuple(
        (
            Interval.point(coordinates[2 * source]),
            Interval.point(coordinates[2 * source + 1]),
        )
        for source in range(3)
    )
    fixed = [fixed_observer_branch(point, source_boxes) for point in POINTS]
    moving_gradient, _, _ = moving_bottom_branch(source_boxes)
    gradients = [gradient for gradient, _ in fixed]
    return tuple(
        tuple(
            gradients[row][column].lower - moving_gradient[column].lower
            for column in range(6)
        )
        for row in range(4)
    )


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


def _fixed_observer_directional_quadratic(
    point: tuple[Q, Q],
    source_boxes: tuple[tuple[Interval, Interval], ...],
    direction: list[Interval],
) -> Interval:
    """Direct sparse fixed-observer quadratic form for one direction box."""

    result = Interval.point(0)
    point_x = Interval.point(point[0])
    point_y = Interval.point(point[1])
    for source, (source_x, source_y) in enumerate(source_boxes):
        dx = point_x - source_x
        dy = point_y - source_y
        dx2 = square_interval(dx)
        dy2 = square_interval(dy)
        radius_squared = dx2 + dy2
        radius_sixth = Interval(
            radius_squared.lower**3, radius_squared.upper**3
        )
        hxx = (6 * dx2 - 2 * dy2) / radius_sixth
        hyy = (6 * dy2 - 2 * dx2) / radius_sixth
        mixed = 8 * dx * dy / radius_sixth
        ux = direction[2 * source]
        uy = direction[2 * source + 1]
        result += (
            square_interval(ux) * hxx
            + 2 * ux * uy * mixed
            + square_interval(uy) * hyy
        )
    return result


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
        critical_difference_matrix: tuple[tuple[Q, ...], ...] | None = None,
        critical_cone_ratio: Q | None = None,
    ) -> None:
        self.root_coordinates = root_coordinates
        self.weights = weights
        self.radius = radius
        self.bottom_curvature_lower = bottom_curvature_lower
        self.critical_difference_matrix = critical_difference_matrix
        self.critical_cone_ratio = critical_cone_ratio
        fixed = [fixed_observer_branch(point, root_source_boxes) for point in POINTS]
        moving_gradient, _, _ = moving_bottom_branch(root_source_boxes)
        self.root_gradients = [gradient for gradient, _ in fixed] + [moving_gradient]

    def classify(
        self, direction: list[Interval], preferred_branch: int | None = None
    ) -> tuple[str, int | None, Q] | None:
        if self.critical_difference_matrix is not None:
            if self.critical_cone_ratio is None:
                raise RuntimeError("critical-cone ratio is missing")
            transverse_norm_squared = sum(
                (
                    square_interval(
                        sum(
                            (
                                coefficient * direction[column]
                                for column, coefficient in enumerate(row)
                            ),
                            start=Interval.point(0),
                        )
                    )
                    for row in self.critical_difference_matrix
                ),
                start=Interval.point(0),
            )
            direction_norm_squared = sum(
                (square_interval(value) for value in direction),
                start=Interval.point(0),
            )
            cone_margin = (
                self.critical_cone_ratio**2 * direction_norm_squared.lower
                - transverse_norm_squared.upper
            )
            if cone_margin >= 0:
                return "critical_cone", None, cone_margin
        source_boxes = _ray_source_boxes(
            self.root_coordinates, direction, self.radius
        )
        if preferred_branch is not None:
            if type(preferred_branch) is not int or not 0 <= preferred_branch < 5:
                raise ValueError("preferred branch must be an integer from 0 through 4")
            preferred_quadratic = None
            if preferred_branch < 4:
                preferred_quadratic = _fixed_observer_directional_quadratic(
                    POINTS[preferred_branch], source_boxes, direction
                )
            else:
                midpoint_ux = potential_derivative_intervals(
                    (Q(1, 2), Q(1, 2), Q(0), Q(0)), source_boxes
                )["ux"]
                shift = max(abs(midpoint_ux.lower), abs(midpoint_ux.upper)) / (
                    self.bottom_curvature_lower
                )
                if shift < Q(1, 8):
                    observer_x = Interval(Q(1, 2) - shift, Q(1, 2) + shift)
                    _, preferred_hessian, _ = moving_bottom_branch_interval(
                        observer_x, source_boxes
                    )
                    preferred_quadratic = _quadratic_form(
                        direction, preferred_hessian
                    )
            if preferred_quadratic is not None:
                linear = _linear_form(
                    direction, self.root_gradients[preferred_branch]
                )
                margin = -(
                    linear.upper
                    + self.radius * max(preferred_quadratic.upper, Q(0)) / 2
                )
                if margin > 0:
                    return "branch", preferred_branch, margin
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
    critical_cone_path: Path | None = None,
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
    critical_ratio = None
    critical_matrix = None
    if critical_cone_path is not None:
        critical = json.loads(critical_cone_path.read_text(encoding="utf-8"))
        if (
            critical.get("status") != "CLOSED"
            or critical.get("rigor") != "exact_rational_interval_analysis"
            or critical.get("negative_adjusted_matrix_certified") is not True
        ):
            raise ValueError("critical-cone prerequisite is not CLOSED")
        if _fraction_string(critical.get("radius"), "critical radius") != radius:
            raise ValueError("critical-cone radius does not match candidate radius")
        if (
            type(critical.get("coarse_digits")) is not int
            or critical.get("coarse_digits") != coarse_digits
        ):
            raise ValueError("critical-cone coarse digits do not match verifier")
        critical_ratio = _fraction_string(critical.get("eta"), "critical eta")
        critical_matrix = _critical_difference_matrix(center, coarse_digits)
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
            critical_matrix,
            critical_ratio,
        ),
        root_radius,
    )


def _validated_split_sensitivities(value: object) -> tuple[Q, ...]:
    if type(value) is not list or len(value) != 6:
        raise ValueError("split_sensitivities must contain six exact strings")
    result = tuple(_fraction_string(item, "split sensitivity") for item in value)
    if any(item <= 0 for item in result):
        raise ValueError("split sensitivities must be positive")
    return result


def _direction_from_root_path(
    root_id: int,
    path: str,
    split_policy: str = "cyclic",
    split_sensitivities: tuple[Q, ...] | None = None,
) -> list[Interval]:
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
        if split_policy == "cyclic":
            split_coordinate = free[depth % 5]
        elif split_policy == "transverse-width":
            if split_sensitivities is None:
                raise ValueError("transverse-width policy needs sensitivities")
            split_coordinate = free[0]
            best_score = (
                direction[split_coordinate].upper
                - direction[split_coordinate].lower
            ) * split_sensitivities[split_coordinate]
            for coordinate in free[1:]:
                score = (
                    direction[coordinate].upper - direction[coordinate].lower
                ) * split_sensitivities[coordinate]
                if score > best_score:
                    best_score = score
                    split_coordinate = coordinate
        else:
            raise ValueError("unsupported split policy")
        value = direction[split_coordinate]
        midpoint = (value.lower + value.upper) / 2
        direction[split_coordinate] = (
            Interval(value.lower, midpoint)
            if bit == "0"
            else Interval(midpoint, value.upper)
        )
    return direction


def _audit_candidate_coverage(
    leaves: list[object],
    split_policy: str = "cyclic",
    split_sensitivities: tuple[Q, ...] | None = None,
) -> None:
    if split_policy not in ("cyclic", "transverse-width"):
        raise ValueError("unsupported split policy")
    if split_policy == "transverse-width" and split_sensitivities is None:
        raise ValueError("transverse-width policy needs sensitivities")
    paths_by_root: list[list[str]] = [[] for _ in range(12)]
    seen: set[tuple[int, str]] = set()
    for raw_leaf in leaves:
        if type(raw_leaf) is not dict:
            raise ValueError("candidate leaves must be objects")
        root_id = raw_leaf.get("root_id")
        path = raw_leaf.get("path")
        if type(root_id) is not int or not 0 <= root_id < 12:
            raise ValueError("root_id must be an integer from 0 through 11")
        if type(path) is not str or any(bit not in "01" for bit in path):
            raise ValueError("path must be a binary string")
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
_WORKER_SPLIT_POLICY = "cyclic"
_WORKER_SPLIT_SENSITIVITIES: tuple[Q, ...] | None = None


def _initialize_worker(
    kkt_path: str,
    neighborhood_path: str,
    radius: str,
    coarse_digits: int,
    critical_cone_path: str | None = None,
    split_policy: str = "cyclic",
    split_sensitivities: tuple[str, ...] | None = None,
) -> None:
    global _WORKER_CLASSIFIER, _WORKER_SPLIT_POLICY, _WORKER_SPLIT_SENSITIVITIES
    _WORKER_CLASSIFIER = _prepare_classifier(
        Path(kkt_path),
        Path(neighborhood_path),
        Q(radius),
        coarse_digits,
        Path(critical_cone_path) if critical_cone_path is not None else None,
    )[0]
    _WORKER_SPLIT_POLICY = split_policy
    _WORKER_SPLIT_SENSITIVITIES = (
        tuple(Q(item) for item in split_sensitivities)
        if split_sensitivities is not None
        else None
    )


def _verify_candidate_leaf(raw_leaf: dict[str, object]) -> tuple[bool, str, int | None, float]:
    if _WORKER_CLASSIFIER is None:
        raise RuntimeError("worker classifier was not initialized")
    direction = _direction_from_root_path(
        raw_leaf["root_id"],
        raw_leaf["path"],
        _WORKER_SPLIT_POLICY,
        _WORKER_SPLIT_SENSITIVITIES,
    )
    preferred_branch = raw_leaf.get("branch_hint", raw_leaf.get("float_branch"))
    if preferred_branch is not None and (
        type(preferred_branch) is not int or not 0 <= preferred_branch < 5
    ):
        raise ValueError("float_branch hint must be an integer from 0 through 4")
    result = _WORKER_CLASSIFIER.classify(direction, preferred_branch)
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
    candidate_offset: int = 0,
    candidate_criterion: str | None = None,
    critical_cone_path: Path | None = None,
) -> dict[str, object]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "closed_in_float_interval":
        raise ValueError("candidate tree did not close in its diagnostic generator")
    radius = Q(str(candidate["radius"]))
    raw_leaves = candidate.get("leaves")
    if type(raw_leaves) is not list:
        raise ValueError("candidate leaves are missing")
    split_policy = candidate.get("split_policy", "cyclic")
    if split_policy not in ("cyclic", "transverse-width"):
        raise ValueError("candidate split policy is unsupported")
    split_sensitivities = (
        _validated_split_sensitivities(candidate.get("split_sensitivities"))
        if split_policy == "transverse-width"
        else None
    )
    if critical_cone_path is None and any(
        leaf.get("float_criterion") == "critical_cone"
        for leaf in raw_leaves
        if type(leaf) is dict
    ):
        raise ValueError("critical-cone leaves require an exact cone prerequisite")
    _audit_candidate_coverage(
        raw_leaves, split_policy, split_sensitivities
    )
    selectable_leaves = (
        [
            leaf
            for leaf in raw_leaves
            if type(leaf) is dict
            and leaf.get("float_criterion") == candidate_criterion
        ]
        if candidate_criterion is not None
        else raw_leaves
    )
    if candidate_offset < 0 or candidate_offset > len(selectable_leaves):
        raise ValueError("candidate offset is outside the leaf list")
    selected = selectable_leaves[candidate_offset:]
    if candidate_limit is not None:
        selected = selected[:candidate_limit]
    started = perf_counter()
    actual_leaves = []
    failures = []
    minimum_margin: float | None = None
    initializer_arguments = (
        str(kkt_path),
        str(neighborhood_path),
        str(radius),
        coarse_digits,
        str(critical_cone_path) if critical_cone_path is not None else None,
        split_policy,
        (
            tuple(str(value) for value in split_sensitivities)
            if split_sensitivities is not None
            else None
        ),
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
                        **(
                            {"branch_hint": raw_leaf["float_branch"]}
                            if "float_branch" in raw_leaf
                            else {}
                        ),
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
    complete = (
        candidate_limit is None
        and candidate_offset == 0
        and candidate_criterion is None
    )
    status = "VERIFIED" if complete and not failures else "INCOMPLETE"
    _, root_radius = _prepare_classifier(
        kkt_path,
        neighborhood_path,
        radius,
        coarse_digits,
        critical_cone_path,
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
            **(
                {
                    "critical_cone": {
                        "path": _display_path(critical_cone_path),
                        "canonical_json_sha256": _canonical_json_sha256(
                            critical_cone_path
                        ),
                    }
                }
                if critical_cone_path is not None
                else {}
            ),
        },
        "candidate_provenance": {
            "method": "untrusted GPU binary64 interval-formula tree proposal",
            "required_for_verification": False,
        },
        "root_centered_linf_radius": str(radius),
        "center_box_linf_radius": str(radius - root_radius),
        "coarse_enclosure_digits": coarse_digits,
        "split_policy": split_policy,
        "split_sensitivities": (
            [str(value) for value in split_sensitivities]
            if split_sensitivities is not None
            else None
        ),
        "candidate_leaf_count": len(raw_leaves),
        "candidate_offset": candidate_offset,
        "candidate_criterion_filter": candidate_criterion,
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
    parser.add_argument("--critical-cone-certificate", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument(
        "--candidate-criterion",
        choices=("branch", "weighted", "critical_cone"),
    )
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
            candidate_offset=args.candidate_offset,
            candidate_criterion=args.candidate_criterion,
            critical_cone_path=args.critical_cone_certificate,
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

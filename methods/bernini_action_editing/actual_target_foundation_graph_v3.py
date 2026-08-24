#!/usr/bin/env python3
"""Pure-CPU anonymous object, track-membership and edge primitives for V3.

The functions in this file intentionally know nothing about models or GPUs.
They make the geometry/control semantics independently testable with small
arrays before any foundation checkpoint is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local contract-only Python
    np = None  # type: ignore[assignment]


PHASES = 8
TRACK_STATES = (
    "ABSENT",
    "VISIBLE_MEMBER",
    "OCCLUDED",
    "VISIBLE_OUTSIDE_MASK",
)
TRANSITION_EVENTS = {
    ("ABSENT", "ABSENT"): (),
    ("ABSENT", "VISIBLE_MEMBER"): ("appearance",),
    ("ABSENT", "OCCLUDED"): ("appearance", "occlusion"),
    ("ABSENT", "VISIBLE_OUTSIDE_MASK"): ("appearance", "membership_loss"),
    ("VISIBLE_MEMBER", "ABSENT"): ("death",),
    ("VISIBLE_MEMBER", "VISIBLE_MEMBER"): (),
    ("VISIBLE_MEMBER", "OCCLUDED"): ("occlusion",),
    ("VISIBLE_MEMBER", "VISIBLE_OUTSIDE_MASK"): ("membership_loss",),
    ("OCCLUDED", "ABSENT"): ("death",),
    ("OCCLUDED", "VISIBLE_MEMBER"): ("reentry",),
    ("OCCLUDED", "OCCLUDED"): (),
    ("OCCLUDED", "VISIBLE_OUTSIDE_MASK"): ("membership_loss",),
    ("VISIBLE_OUTSIDE_MASK", "ABSENT"): ("death",),
    ("VISIBLE_OUTSIDE_MASK", "VISIBLE_MEMBER"): ("reentry",),
    ("VISIBLE_OUTSIDE_MASK", "OCCLUDED"): ("occlusion",),
    ("VISIBLE_OUTSIDE_MASK", "VISIBLE_OUTSIDE_MASK"): (),
}


class GraphV3Error(RuntimeError):
    pass


@dataclass(frozen=True)
class AnonymousNodeV3:
    mask: Any
    descriptor: Any
    area_fraction: float
    centroid_xy: tuple[float, float]
    track_id: int = -1


@dataclass(frozen=True)
class TrackMembershipV3:
    track_id: int
    point_indices: tuple[int, ...]
    member_phase_counts: tuple[int, ...]
    phase_member_counts: tuple[int, ...]
    phase_visible_counts: tuple[int, ...]
    phase_states: tuple[str, ...]
    centers_xy: Any
    center_valid: Any
    velocities_xy: Any
    velocity_valid: Any
    lifecycle: Mapping[str, int]


@dataclass(frozen=True)
class TrackAssignmentV3:
    memberships: tuple[TrackMembershipV3, ...]
    ambiguous_overlap_observation_count: int
    out_of_bounds_observation_count: int
    nonfinite_observation_count: int
    vote_tie_abstain_count: int
    insufficient_membership_abstain_count: int


@dataclass(frozen=True)
class EdgeSketchV3:
    signature: tuple[float, ...]
    dropped_signature: tuple[float, ...]
    per_phase_active_counts: tuple[int, ...]
    per_phase_birth_counts: tuple[int, ...]
    per_phase_persist_counts: tuple[int, ...]
    per_phase_death_counts: tuple[int, ...]
    per_phase_valid_velocity_counts: tuple[int, ...]
    per_phase_qualified_lifecycle_counts: tuple[int, ...]
    removed_edge_count: int


def _require_numpy() -> Any:
    if np is None:
        raise GraphV3Error("numpy is required for geometry primitives")
    return np


def _descriptor_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value)


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    numpy = _require_numpy()
    a = numpy.asarray(left, dtype=numpy.float64)
    b = numpy.asarray(right, dtype=numpy.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise GraphV3Error("descriptor shapes differ")
    denominator = float(numpy.linalg.norm(a) * numpy.linalg.norm(b))
    return 1.0 - float(a @ b / denominator) if denominator > 1e-12 else 1.0


def unbalanced_sinkhorn_dustbin(
    left: Sequence[AnonymousNodeV3],
    right: Sequence[AnonymousNodeV3],
    *,
    epsilon: float = 0.08,
    rho: float = 0.35,
    dustbin_cost: float = 0.42,
    iterations: int = 80,
    left_gap_phases: Sequence[int] | None = None,
    gap_penalty: float = 0.08,
) -> Any:
    """KL-relaxed unbalanced OT with an explicit final row/column dustbin."""

    numpy = _require_numpy()
    n, m = len(left), len(right)
    cost = numpy.full((n + 1, m + 1), dustbin_cost, dtype=numpy.float64)
    cost[-1, -1] = 0.0
    gaps = tuple(left_gap_phases or (0,) * n)
    if len(gaps) != n or any(not isinstance(value, int) or value < 0 for value in gaps):
        raise GraphV3Error("dormant-track gap geometry differs")
    for i, lhs in enumerate(left):
        for j, rhs in enumerate(right):
            geometry = math.dist(lhs.centroid_xy, rhs.centroid_xy)
            cost[i, j] = (
                0.8 * cosine_distance(lhs.descriptor, rhs.descriptor)
                + 0.2 * min(geometry, 1.0)
                + gap_penalty * gaps[i]
            )
    kernel = numpy.exp(-cost / epsilon).clip(1e-30, None)
    scale = max(n, m, 1)
    left_mass = numpy.full(n + 1, 1.0 / scale)
    right_mass = numpy.full(m + 1, 1.0 / scale)
    left_mass[-1] = max(m - n, 0) / scale + 1.0 / scale
    right_mass[-1] = max(n - m, 0) / scale + 1.0 / scale
    tau = rho / (rho + epsilon)
    u = numpy.ones(n + 1)
    v = numpy.ones(m + 1)
    for _ in range(iterations):
        u = numpy.power(left_mass / (kernel @ v).clip(1e-30, None), tau)
        v = numpy.power(right_mass / (kernel.T @ u).clip(1e-30, None), tau)
    plan = (u[:, None] * kernel) * v[None, :]
    if not numpy.isfinite(plan).all():
        raise GraphV3Error("nonfinite unbalanced transport")
    return plan


def hard_matches_with_dustbin(plan: Any) -> tuple[tuple[int, int], ...]:
    numpy = _require_numpy()
    value = numpy.asarray(plan)
    if value.ndim != 2 or min(value.shape) < 1:
        raise GraphV3Error("transport plan geometry differs")
    n, m = value.shape[0] - 1, value.shape[1] - 1
    matches = []
    for i in range(n):
        if m == 0:
            continue
        j = int(numpy.argmax(value[i, :m]))
        if int(numpy.argmax(value[:n, j])) != i:
            continue
        if value[i, j] <= value[i, m] or value[i, j] <= value[n, j]:
            continue
        matches.append((i, j))
    return tuple(matches)


def _node_key(node: AnonymousNodeV3) -> bytes:
    numpy = _require_numpy()
    if (
        not isinstance(node.mask, numpy.ndarray)
        or node.mask.ndim != 2
        or node.mask.dtype != numpy.bool_
        or not node.mask.flags.c_contiguous
    ):
        raise GraphV3Error(
            "node mask must be an already-owned C-contiguous bool ndarray"
        )
    values = tuple(
        round(float(item), 8)
        for item in (*_descriptor_tuple(node.descriptor), node.area_fraction, *node.centroid_xy)
    )
    digest = hashlib.sha256(repr(values).encode("ascii"))
    digest.update(memoryview(node.mask).cast("B"))
    return digest.digest()


def assign_anonymous_tracks(
    phases: Sequence[Sequence[AnonymousNodeV3]],
    *,
    max_absent_gap_phases: int = 2,
    gap_penalty: float = 0.08,
) -> tuple[tuple[AnonymousNodeV3, ...], ...]:
    output: list[tuple[AnonymousNodeV3, ...]] = []
    next_id = 0
    dormant: dict[int, tuple[AnonymousNodeV3, int]] = {}
    for phase_index, phase in enumerate(phases):
        current = [replace(node, track_id=-1) for node in sorted(phase, key=_node_key)]
        eligible = sorted(
            (
                (track_id, node, phase_index - last_phase - 1)
                for track_id, (node, last_phase) in dormant.items()
                if phase_index - last_phase - 1 <= max_absent_gap_phases
            ),
            key=lambda row: row[0],
        )
        if eligible and current:
            previous = tuple(row[1] for row in eligible)
            gaps = tuple(row[2] for row in eligible)
            plan = unbalanced_sinkhorn_dustbin(
                previous,
                current,
                left_gap_phases=gaps,
                gap_penalty=gap_penalty,
            )
            for i, j in hard_matches_with_dustbin(plan):
                current[j] = replace(current[j], track_id=eligible[i][0])
        for index, node in enumerate(current):
            if node.track_id < 0:
                current[index] = replace(node, track_id=next_id)
                next_id += 1
        dormant = {
            track_id: row
            for track_id, row in dormant.items()
            if phase_index - row[1] <= max_absent_gap_phases
        }
        for node in current:
            dormant[node.track_id] = (node, phase_index)
        output.append(tuple(current))
    return tuple(output)


def unbalanced_matching_diagnostics(
    phases: Sequence[Sequence[AnonymousNodeV3]],
) -> Mapping[str, Any]:
    unmatched = 0
    dustbin_mass = 0.0
    pairs = 0
    for left, right in zip(phases, phases[1:]):
        plan = unbalanced_sinkhorn_dustbin(left, right)
        matches = hard_matches_with_dustbin(plan)
        pairs += 1
        unmatched += len(left) + len(right) - 2 * len(matches)
        dustbin_mass += float(plan[-1, :-1].sum() + plan[:-1, -1].sum())
    return {
        "phase_pair_count": pairs,
        "explicit_dustbin": True,
        "unmatched_count": unmatched,
        "dustbin_transport_mass": dustbin_mass,
    }


def canonical_node_signature(
    phases: Sequence[Sequence[AnonymousNodeV3]],
    *,
    slots: int = 12,
    descriptor_width: int = 8,
) -> tuple[float, ...]:
    signature: list[float] = []
    for phase in phases:
        ordered = sorted(phase, key=_node_key)[:slots]
        for node in ordered:
            descriptor = _descriptor_tuple(node.descriptor)[:descriptor_width]
            descriptor += (0.0,) * (descriptor_width - len(descriptor))
            signature.extend((*descriptor, node.area_fraction, *node.centroid_xy))
        signature.extend((0.0,) * ((slots - len(ordered)) * (descriptor_width + 3)))
    return tuple(signature)


def break_mask_descriptor_binding(
    phases: Sequence[Sequence[AnonymousNodeV3]],
) -> tuple[tuple[AnonymousNodeV3, ...], ...]:
    broken = []
    for phase in phases:
        if len(phase) < 2:
            broken.append(tuple())
            continue
        descriptors = [node.descriptor for node in phase]
        broken.append(
            tuple(
                replace(node, descriptor=descriptors[(index + 1) % len(phase)])
                for index, node in enumerate(phase)
            )
        )
    return tuple(broken)


def relabel_slots(
    phases: Sequence[Sequence[AnonymousNodeV3]],
) -> tuple[tuple[AnonymousNodeV3, ...], ...]:
    return tuple(tuple(reversed(phase)) for phase in phases)


def boundary_gap_overlap(left: Any, right: Any) -> tuple[float, float]:
    numpy = _require_numpy()
    lhs = numpy.asarray(left, dtype=bool)
    rhs = numpy.asarray(right, dtype=bool)
    if lhs.shape != rhs.shape or lhs.ndim != 2:
        raise GraphV3Error("mask geometry differs")
    union = numpy.logical_or(lhs, rhs).sum()
    overlap = float(numpy.logical_and(lhs, rhs).sum() / union) if union else 0.0

    def boundary(mask: Any) -> Any:
        eroded = mask.copy()
        eroded[1:, :] &= mask[:-1, :]
        eroded[:-1, :] &= mask[1:, :]
        eroded[:, 1:] &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        return numpy.argwhere(mask & ~eroded)

    lhs_points, rhs_points = boundary(lhs), boundary(rhs)
    if not len(lhs_points) or not len(rhs_points):
        return 1.0, overlap
    minimum = min(
        float(
            (
                (lhs_points[start : start + 512, None, :] - rhs_points[None, :, :])
                ** 2
            )
            .sum(-1)
            .min()
        )
        for start in range(0, len(lhs_points), 512)
    )
    return math.sqrt(minimum) / math.hypot(*lhs.shape), overlap


def patch_area_pool(mask_weights: Any, patch_tokens: Any) -> tuple[float, ...]:
    numpy = _require_numpy()
    weights = numpy.asarray(mask_weights, dtype=numpy.float64).reshape(-1)
    tokens = numpy.asarray(patch_tokens, dtype=numpy.float64)
    if tokens.ndim != 2 or tokens.shape[0] != weights.size:
        raise GraphV3Error("patch/token geometry differs")
    support = float(weights.sum())
    if not numpy.isfinite(weights).all() or support <= 1e-6:
        raise GraphV3Error("zero/nonfinite DINO patch support abstains")
    return tuple((tokens * weights[:, None]).sum(0) / support)


def tubelet2_eight_blocks(hidden: Any, spatial_tokens: int) -> Any:
    numpy = _require_numpy()
    value = numpy.asarray(hidden)
    if value.ndim != 2 or spatial_tokens <= 0 or value.shape[0] != PHASES * spatial_tokens:
        raise GraphV3Error("V-JEPA output is not exactly eight tubelet2 temporal blocks")
    return value.reshape(PHASES, spatial_tokens, value.shape[-1]).mean(1)


def _mask_for_track(
    phase: Sequence[AnonymousNodeV3], track_id: int
) -> Any | None:
    matches = [node.mask for node in phase if node.track_id == track_id]
    if len(matches) > 1:
        raise GraphV3Error("one anonymous track occurs more than once in a phase")
    return matches[0] if matches else None


def canonical_track_signature(
    memberships: Sequence[TrackMembershipV3],
    descriptor_by_track: Mapping[int, Sequence[float]],
    *,
    maximum_worldlines: int = 96,
) -> tuple[float, ...]:
    """Label-free canonical worldline blocks; numeric track IDs never order slots."""

    numpy = _require_numpy()
    if len(memberships) > maximum_worldlines:
        raise GraphV3Error("assigned worldlines exceed the fixed mechanical maximum")
    rows = []
    for row in memberships:
        if row.track_id not in descriptor_by_track:
            raise GraphV3Error("assigned worldline lacks a descriptor")
        descriptor = tuple(float(value) for value in descriptor_by_track[row.track_id][:8])
        descriptor += (0.0,) * (8 - len(descriptor))
        valid_velocities = numpy.asarray(row.velocities_xy)[numpy.asarray(row.velocity_valid, dtype=bool)]
        mean_velocity = valid_velocities.mean(0) if len(valid_velocities) else numpy.zeros(2)
        member_total = sum(row.phase_member_counts)
        denominator = PHASES * len(row.point_indices)
        block = (
            *descriptor,
            float(mean_velocity[0]),
            float(mean_velocity[1]),
            member_total / denominator,
            sum(count > 0 for count in row.phase_member_counts) / PHASES,
        )
        label_free = (
            tuple(round(float(value), 8) for value in block),
            tuple(row.phase_states),
            tuple(row.phase_member_counts),
        )
        rows.append((hashlib.sha256(repr(label_free).encode("ascii")).digest(), block))
    values = [value for _key, block in sorted(rows, key=lambda item: item[0]) for value in block]
    values.extend([0.0] * ((maximum_worldlines - len(rows)) * 12))
    return tuple(values)


def assign_points_with_same_track_membership(
    phases: Sequence[Sequence[AnonymousNodeV3]],
    coordinates_xy: Any,
    visible: Any,
    *,
    minimum_member_phases: int = 3,
) -> TrackAssignmentV3:
    """Assign points only when visible inside the same anonymous mask track.

    A point votes for a track only on phases where CoTracker marks it visible
    *and* its coordinate lies inside that exact track's automatic mask.  The
    winning track must receive at least ``minimum_member_phases`` votes.
    Velocities exist only when both adjacent phases are visible members.
    """

    numpy = _require_numpy()
    if (
        not isinstance(coordinates_xy, numpy.ndarray)
        or coordinates_xy.dtype != numpy.float64
        or not coordinates_xy.flags.c_contiguous
    ):
        raise GraphV3Error(
            "CoTracker coordinates must enter as pre-owned C-contiguous float64"
        )
    if (
        not isinstance(visible, numpy.ndarray)
        or visible.dtype != numpy.bool_
        or not visible.flags.c_contiguous
    ):
        raise GraphV3Error(
            "CoTracker visibility must enter as pre-owned C-contiguous bool"
        )
    xy = coordinates_xy
    vis = visible
    if len(phases) != PHASES or xy.ndim != 3 or xy.shape[0] != PHASES or xy.shape[2] != 2:
        raise GraphV3Error("CoTracker coordinate geometry differs")
    if vis.shape != xy.shape[:2]:
        raise GraphV3Error("CoTracker visibility geometry differs")
    if minimum_member_phases < 3:
        raise GraphV3Error("same-track membership floor must be at least three phases")

    votes_by_point: list[dict[int, int]] = []
    ambiguous_overlap_count = 0
    out_of_bounds_count = 0
    nonfinite_count = 0
    for point in range(xy.shape[1]):
        votes: dict[int, int] = {}
        for phase_index, phase in enumerate(phases):
            if not bool(vis[phase_index, point]):
                continue
            raw_x = float(xy[phase_index, point, 0])
            raw_y = float(xy[phase_index, point, 1])
            if not math.isfinite(raw_x) or not math.isfinite(raw_y):
                nonfinite_count += 1
                continue
            x = int(round(raw_x))
            y = int(round(raw_y))
            containing: list[int] = []
            for node in sorted(phase, key=_node_key):
                mask = numpy.asarray(node.mask, dtype=bool)
                if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
                    continue
                if bool(mask[y, x]):
                    containing.append(node.track_id)
            if phase and not any(
                0 <= x < numpy.asarray(node.mask).shape[1]
                and 0 <= y < numpy.asarray(node.mask).shape[0]
                for node in phase
            ):
                out_of_bounds_count += 1
            if len(set(containing)) > 1:
                ambiguous_overlap_count += 1
                continue
            if len(containing) == 1:
                chosen = containing[0]
                votes[chosen] = votes.get(chosen, 0) + 1
        votes_by_point.append(votes)

    grouped: dict[int, list[int]] = {}
    member_counts: dict[tuple[int, int], int] = {}
    tie_abstain_count = 0
    insufficient_count = 0
    for point, votes in enumerate(votes_by_point):
        if not votes:
            insufficient_count += 1
            continue
        best_count = max(votes.values())
        winners = [track for track, count in votes.items() if count == best_count]
        if len(winners) != 1:
            tie_abstain_count += 1
            continue
        best_track = winners[0]
        if best_count < minimum_member_phases:
            insufficient_count += 1
            continue
        grouped.setdefault(best_track, []).append(point)
        member_counts[(best_track, point)] = best_count

    output: list[TrackMembershipV3] = []
    for track_id, points in sorted(grouped.items()):
        phase_members: list[int] = []
        phase_visible: list[int] = []
        states: list[str] = []
        centers: list[tuple[float, float]] = []
        center_valid: list[bool] = []
        for phase_index, phase in enumerate(phases):
            mask = _mask_for_track(phase, track_id)
            visible_count = 0
            member_points = []
            for point in points:
                if not bool(vis[phase_index, point]):
                    continue
                visible_count += 1
                if mask is None:
                    continue
                mask_value = numpy.asarray(mask, dtype=bool)
                raw_x = float(xy[phase_index, point, 0])
                raw_y = float(xy[phase_index, point, 1])
                if not math.isfinite(raw_x) or not math.isfinite(raw_y):
                    continue
                x = int(round(raw_x))
                y = int(round(raw_y))
                if not (0 <= x < mask_value.shape[1] and 0 <= y < mask_value.shape[0]):
                    continue
                containing = []
                for candidate in phase:
                    candidate_mask = numpy.asarray(candidate.mask, dtype=bool)
                    if 0 <= x < candidate_mask.shape[1] and 0 <= y < candidate_mask.shape[0] and bool(candidate_mask[y, x]):
                        containing.append(candidate.track_id)
                if containing == [track_id] or set(containing) == {track_id}:
                    member_points.append(point)
            phase_members.append(len(member_points))
            phase_visible.append(visible_count)
            if mask is None:
                state = "ABSENT"
            elif member_points:
                state = "VISIBLE_MEMBER"
            elif visible_count == 0:
                state = "OCCLUDED"
            else:
                state = "VISIBLE_OUTSIDE_MASK"
            states.append(state)
            if member_points:
                inverse = 1.0 / len(member_points)
                centers.append(
                    (
                        sum(
                            float(xy[phase_index, point, 0])
                            for point in member_points
                        )
                        * inverse,
                        sum(
                            float(xy[phase_index, point, 1])
                            for point in member_points
                        )
                        * inverse,
                    )
                )
                center_valid.append(True)
            else:
                centers.append((0.0, 0.0))
                center_valid.append(False)

        velocities: list[tuple[float, float]] = [(0.0, 0.0)]
        velocity_valid: list[bool] = [False]
        for phase_index in range(1, PHASES):
            previous, current = centers[phase_index - 1], centers[phase_index]
            if (
                center_valid[phase_index - 1]
                and center_valid[phase_index]
                and states[phase_index - 1] == "VISIBLE_MEMBER"
                and states[phase_index] == "VISIBLE_MEMBER"
            ):
                velocities.append((current[0] - previous[0], current[1] - previous[1]))
                velocity_valid.append(True)
            else:
                velocities.append((0.0, 0.0))
                velocity_valid.append(False)

        lifecycle = {"entry": 0, "occlusion": 0, "membership_loss": 0, "reentry": 0, "death": 0}
        previous = "ABSENT"
        ever_present = False
        for state in states:
            for event in TRANSITION_EVENTS[(previous, state)]:
                if event == "appearance":
                    lifecycle["reentry" if ever_present else "entry"] += 1
                else:
                    lifecycle[event] += 1
            if state != "ABSENT":
                ever_present = True
            previous = state

        output.append(
            TrackMembershipV3(
                track_id=track_id,
                point_indices=tuple(points),
                member_phase_counts=tuple(member_counts[(track_id, point)] for point in points),
                phase_member_counts=tuple(phase_members),
                phase_visible_counts=tuple(phase_visible),
                phase_states=tuple(states),
                centers_xy=numpy.asarray(centers, dtype=numpy.float64),
                center_valid=numpy.asarray(center_valid, dtype=bool),
                velocities_xy=numpy.asarray(velocities, dtype=numpy.float64),
                velocity_valid=numpy.asarray(velocity_valid, dtype=bool),
                lifecycle=lifecycle,
            )
        )
    return TrackAssignmentV3(
        memberships=tuple(output),
        ambiguous_overlap_observation_count=ambiguous_overlap_count,
        out_of_bounds_observation_count=out_of_bounds_count,
        nonfinite_observation_count=nonfinite_count,
        vote_tie_abstain_count=tie_abstain_count,
        insufficient_membership_abstain_count=insufficient_count,
    )


def per_phase_edge_signatures(
    phases: Sequence[Sequence[AnonymousNodeV3]],
    memberships: Sequence[TrackMembershipV3],
    *,
    overlap_iou_threshold: float = 0.01,
    boundary_gap_threshold: float = 0.04,
    predictive_gap_threshold: float = 0.08,
    converging_speed_threshold: float = 0.005,
) -> EdgeSketchV3:
    """Build five real channels per phase and a non-zero drop-edge control.

    Channels are boundary gap, overlap IoU, valid relative velocity, birth
    fraction and death fraction.  The universe is every canonical pair of
    assigned tracks.  Dropping deterministically removes the first active
    canonical edge per phase; absent phases retain a non-zero gap sentinel, so
    the control is never the old all-zero algebraic shortcut.
    """

    if len(phases) != PHASES:
        raise GraphV3Error("edge phases differ")
    by_track = {row.track_id: row for row in memberships}
    def membership_key(row: TrackMembershipV3) -> bytes:
        worldline_nodes = []
        for phase in phases:
            matches = [node for node in phase if node.track_id == row.track_id]
            if len(matches) > 1:
                raise GraphV3Error("one anonymous track occurs more than once in a phase")
            worldline_nodes.append(_node_key(matches[0]).hex() if matches else "ABSENT")
        values = (
            tuple(round(float(value), 8) for value in np.asarray(row.centers_xy).reshape(-1)),
            tuple(bool(value) for value in np.asarray(row.center_valid).reshape(-1)),
            tuple(row.phase_states),
            tuple(row.phase_member_counts),
            tuple(worldline_nodes),
        )
        return hashlib.sha256(repr(values).encode("ascii")).digest()
    track_ids = [row.track_id for row in sorted(memberships, key=membership_key)]
    pairs = [(lhs, rhs) for index, lhs in enumerate(track_ids) for rhs in track_ids[index + 1 :]]
    if not pairs:
        return EdgeSketchV3(
            signature=tuple([1.0, 0.0, 0.0, 0.0, 0.0] * PHASES),
            dropped_signature=tuple([1.0, 0.0, 0.0, 0.0, 0.0] * PHASES),
            per_phase_active_counts=(0,) * PHASES,
            per_phase_birth_counts=(0,) * PHASES,
            per_phase_persist_counts=(0,) * PHASES,
            per_phase_death_counts=(0,) * PHASES,
            per_phase_valid_velocity_counts=(0,) * PHASES,
            per_phase_qualified_lifecycle_counts=(0,) * PHASES,
            removed_edge_count=0,
        )

    signature: list[float] = []
    dropped: list[float] = []
    active_counts: list[int] = []
    birth_counts: list[int] = []
    persist_counts: list[int] = []
    death_counts: list[int] = []
    velocity_counts: list[int] = []
    qualified_lifecycle_counts: list[int] = []
    removed_total = 0
    previous_active: set[tuple[int, int]] = set()
    previous_velocity_valid: set[tuple[int, int]] = set()
    dropped_previous_active: set[tuple[int, int]] = set()
    for phase_index, phase in enumerate(phases):
        active_rows: list[tuple[int, int, float, float, float | None]] = []
        current_active: set[tuple[int, int]] = set()
        current_velocity_valid: set[tuple[int, int]] = set()
        for lhs, rhs in pairs:
            lhs_mask = _mask_for_track(phase, lhs)
            rhs_mask = _mask_for_track(phase, rhs)
            if lhs_mask is None or rhs_mask is None:
                continue
            gap, overlap = boundary_gap_overlap(lhs_mask, rhs_mask)
            lhs_row = by_track[lhs]
            rhs_row = by_track[rhs]
            lhs_velocity = lhs_row.velocities_xy[phase_index]
            rhs_velocity = rhs_row.velocities_xy[phase_index]
            velocity_valid = bool(lhs_row.velocity_valid[phase_index]) and bool(rhs_row.velocity_valid[phase_index])
            relative = math.dist(lhs_velocity, rhs_velocity) if velocity_valid else None
            converging = 0.0
            if velocity_valid and bool(lhs_row.center_valid[phase_index]) and bool(rhs_row.center_valid[phase_index]):
                relative_position = rhs_row.centers_xy[phase_index] - lhs_row.centers_xy[phase_index]
                relative_velocity = rhs_velocity - lhs_velocity
                distance = float(np.linalg.norm(relative_position))
                if distance > 1e-12:
                    converging = max(0.0, -float(relative_position @ relative_velocity) / distance)
            active = (
                overlap >= overlap_iou_threshold
                or gap <= boundary_gap_threshold
                or (
                    velocity_valid
                    and gap <= predictive_gap_threshold
                    and converging >= converging_speed_threshold
                )
            )
            if not active:
                continue
            active_rows.append((lhs, rhs, gap, overlap, relative))
            current_active.add((lhs, rhs))
            if relative is not None:
                current_velocity_valid.add((lhs, rhs))
        births = current_active - previous_active
        persists = current_active & previous_active
        deaths = previous_active - current_active

        def summarize(
            rows: Sequence[tuple[int, int, float, float, float | None]],
            row_births: set[tuple[int, int]],
            row_deaths: set[tuple[int, int]],
        ) -> list[float]:
            if rows:
                gap = sum(row[2] for row in rows) / len(rows)
                overlap = sum(row[3] for row in rows) / len(rows)
            else:
                gap, overlap = 1.0, 0.0
            velocities = [row[4] for row in rows if row[4] is not None]
            relative = sum(velocities) / len(velocities) if velocities else 0.0
            return [
                float(gap),
                float(overlap),
                float(relative),
                len(row_births) / max(len(pairs), 1),
                len(row_deaths) / max(len(pairs), 1),
            ]

        signature.extend(summarize(active_rows, births, deaths))
        drop_rows = active_rows[1:] if active_rows else active_rows
        dropped_current_active = {(row[0], row[1]) for row in drop_rows}
        dropped_births = dropped_current_active - dropped_previous_active
        dropped_deaths = dropped_previous_active - dropped_current_active
        if active_rows:
            removed_total += 1
        dropped.extend(summarize(drop_rows, dropped_births, dropped_deaths))
        active_counts.append(len(active_rows))
        birth_counts.append(len(births))
        persist_counts.append(len(persists))
        death_counts.append(len(deaths))
        velocity_counts.append(sum(row[4] is not None for row in active_rows))
        qualified_lifecycle_counts.append(
            len((persists & current_velocity_valid) | (deaths & previous_velocity_valid))
        )
        previous_active = current_active
        previous_velocity_valid = current_velocity_valid
        dropped_previous_active = dropped_current_active
    return EdgeSketchV3(
        signature=tuple(signature),
        dropped_signature=tuple(dropped),
        per_phase_active_counts=tuple(active_counts),
        per_phase_birth_counts=tuple(birth_counts),
        per_phase_persist_counts=tuple(persist_counts),
        per_phase_death_counts=tuple(death_counts),
        per_phase_valid_velocity_counts=tuple(velocity_counts),
        per_phase_qualified_lifecycle_counts=tuple(qualified_lifecycle_counts),
        removed_edge_count=removed_total,
    )


__all__ = [
    "AnonymousNodeV3",
    "EdgeSketchV3",
    "GraphV3Error",
    "PHASES",
    "TrackAssignmentV3",
    "TrackMembershipV3",
    "TRACK_STATES",
    "TRANSITION_EVENTS",
    "assign_anonymous_tracks",
    "assign_points_with_same_track_membership",
    "boundary_gap_overlap",
    "break_mask_descriptor_binding",
    "canonical_node_signature",
    "canonical_track_signature",
    "cosine_distance",
    "hard_matches_with_dustbin",
    "patch_area_pool",
    "per_phase_edge_signatures",
    "relabel_slots",
    "tubelet2_eight_blocks",
    "unbalanced_matching_diagnostics",
    "unbalanced_sinkhorn_dustbin",
]

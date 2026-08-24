#!/usr/bin/env python3
"""Deterministic V7 factorial-compatibility space-time tube reducer.

The reducer consumes the V6 projected visual-capture ABI.  Its proposal is a
closed-form, same-caption cross-state interaction residual.  Prompt-neutral
visual sketches alone define correspondence and public tube descriptors.  No
decoder, optimizer, semantic role inventory, reward, or generator route is
present in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

import torch

import self_generated_anonymous_object_observer_v6 as v6_observer
import self_generated_factorial_compatibility_registry_v7 as registry


METHOD = "bernini-self-generated-factorial-compatibility-tube-observer-v7"
SCHEMA_VERSION = "bernini-self-generated-factorial-compatibility-tube-observer-v7"
BRANCHES = registry.BRANCHES
LAYER_FOLDS = {"A": (6, 18), "B": (12, 24)}
TIME_FOLDS = {
    "A": tuple(range(0, registry.PHASES, 2)),
    "B": tuple(range(1, registry.PHASES, 2)),
}
CROSS_FIT_PHASE_PAIRS = {
    "A_to_B": tuple((phase, phase + 1) for phase in range(0, 20, 2)),
    "B_to_A": tuple((phase, phase + 1) for phase in range(1, 20, 2)),
}
CONTROL_NAMES = (
    "noop",
    "static",
    "reverse",
    "phase_shuffle",
    "paraphrase",
    "lexical_placebo",
    "source_swap",
)
_EPS = 1.0e-8


class FactorialCompatibilityTubeV7Error(RuntimeError):
    """A factorial identity, tube, ownership, or hard gate differs."""


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise FactorialCompatibilityTubeV7Error(
            "receipt is not canonical finite JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _zeroize(values: Sequence[torch.Tensor]) -> None:
    with torch.inference_mode():
        for value in values:
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                value.zero_()


def _validate_feature(value: torch.Tensor, *, phases: int, patches: int) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or tuple(value.shape[:2]) != (phases, patches)
        or value.shape[-1] <= 0
        or value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise FactorialCompatibilityTubeV7Error(
            "projected feature ABI differs"
        )


def _mean_blocks(
    rows: Mapping[int, torch.Tensor], blocks: Sequence[int]
) -> torch.Tensor:
    if set(rows) != set(registry.BLOCKS) or any(block not in rows for block in blocks):
        raise FactorialCompatibilityTubeV7Error("block feature matrix differs")
    result = torch.stack([rows[block] for block in blocks], dim=0).mean(dim=0)
    return result.detach().float()


def _branch_folds(branch: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if branch == "A_to_B":
        return LAYER_FOLDS["A"], LAYER_FOLDS["B"]
    if branch == "B_to_A":
        return LAYER_FOLDS["B"], LAYER_FOLDS["A"]
    raise FactorialCompatibilityTubeV7Error("cross-fit branch differs")


def _validate_factorial_feature_matrix(
    factorial: Mapping[str, Mapping[str, torch.Tensor]],
    neutral: Mapping[str, torch.Tensor],
) -> None:
    if set(factorial) != set(registry.APPEARANCE_IDS) or set(neutral) != set(
        registry.APPEARANCE_IDS
    ):
        raise FactorialCompatibilityTubeV7Error(
            "factorial state matrix differs"
        )
    shapes: set[tuple[int, ...]] = set()
    for state_id in registry.APPEARANCE_IDS:
        if set(factorial[state_id]) != set(registry.APPEARANCE_IDS):
            raise FactorialCompatibilityTubeV7Error(
                "full three by three factorial is required"
            )
        for caption_id in registry.APPEARANCE_IDS:
            value = factorial[state_id][caption_id]
            _validate_feature(
                value,
                phases=registry.PHASES,
                patches=registry.PATCHES,
            )
            shapes.add(tuple(value.shape))
        _validate_feature(
            neutral[state_id],
            phases=registry.PHASES,
            patches=registry.PATCHES,
        )
        shapes.add(tuple(neutral[state_id].shape))
    if len(shapes) != 1:
        raise FactorialCompatibilityTubeV7Error(
            "factorial feature widths differ"
        )


def factorial_interaction_residual_v7(
    factorial: Mapping[str, Mapping[str, torch.Tensor]],
    neutral: Mapping[str, torch.Tensor],
    *,
    state_id: str,
    branch: str,
) -> torch.Tensor:
    """Closed-form difference-in-differences for one matched state/caption.

    Within-state subtraction of the byte-identical neutral caption removes the
    visual-state/appearance main effect.  Subtraction of the same action
    caption on the branch's nuisance off-diagonal state removes the caption
    main effect.  The opposite off-diagonal remains untouched for evaluation.
    """

    _validate_factorial_feature_matrix(factorial, neutral)
    if state_id not in registry.APPEARANCE_IDS or branch not in BRANCHES:
        raise FactorialCompatibilityTubeV7Error(
            "factorial interaction identity differs"
        )
    nuisance_state = registry.nuisance_state_for_caption(branch, state_id)
    matched = factorial[state_id][state_id] - neutral[state_id]
    nuisance = factorial[nuisance_state][state_id] - neutral[nuisance_state]
    return (matched - nuisance).detach().float()


def factorial_heldout_residual_v7(
    factorial: Mapping[str, Mapping[str, torch.Tensor]],
    neutral: Mapping[str, torch.Tensor],
    *,
    state_id: str,
    branch: str,
) -> tuple[torch.Tensor, str]:
    """Interaction residual for the disjoint held-out source-swap caption."""

    _validate_factorial_feature_matrix(factorial, neutral)
    caption_id = registry.heldout_caption_for_state(branch, state_id)
    nuisance_state = registry.nuisance_state_for_caption(branch, caption_id)
    heldout = factorial[state_id][caption_id] - neutral[state_id]
    nuisance = factorial[nuisance_state][caption_id] - neutral[nuisance_state]
    return (heldout - nuisance).detach().float(), caption_id


def _grid(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def _spatial_neighbors(index: int, height: int, width: int) -> tuple[int, ...]:
    y, x = divmod(index, width)
    rows: list[int] = []
    if y > 0:
        rows.append(index - width)
    if y + 1 < height:
        rows.append(index + width)
    if x > 0:
        rows.append(index - 1)
    if x + 1 < width:
        rows.append(index + 1)
    return tuple(rows)


@dataclass(frozen=True)
class SpaceTimeTubeV7:
    tube_id: int
    components: tuple[v6_observer.ProposalComponentV6, ...]
    total_soft_mass: float

    @property
    def observed_proposal_phases(self) -> tuple[int, ...]:
        return tuple(row.phase for row in self.components)

    def public_row(self) -> Mapping[str, Any]:
        value = {
            "tube_id": self.tube_id,
            "observed_proposal_phases": list(self.observed_proposal_phases),
            "component_count": len(self.components),
            "total_soft_mass": self.total_soft_mass,
            "components": [row.public_row() for row in self.components],
            "semantic_role": None,
        }
        return {**value, "digest": _digest(value)}


@dataclass(frozen=True)
class SpaceTimeTubeConstructionV7:
    active_phases: tuple[int, ...]
    tubes: tuple[SpaceTimeTubeV7, ...]
    eligible_voxel_count: int
    dustbin_voxel_count: int
    temporal_edge_count: int
    temporal_dustbin_assignment_count: int = 0

    def public_row(self) -> Mapping[str, Any]:
        value = {
            "construction_domain": [
                registry.PHASES,
                registry.PATCH_HEIGHT,
                registry.PATCH_WIDTH,
            ],
            "active_phases": list(self.active_phases),
            "joint_space_time_connected_components": True,
            "independent_per_phase_slot_finalization_permitted": False,
            "tube_count": len(self.tubes),
            "eligible_voxel_count": self.eligible_voxel_count,
            "dustbin_voxel_count": self.dustbin_voxel_count,
            "temporal_neutral_correspondence_edge_count": self.temporal_edge_count,
            "temporal_unbalanced_ot_dustbin_assignment_count": (
                self.temporal_dustbin_assignment_count
            ),
            "temporal_correspondence": (
                "V6 unbalanced OT over prompt-neutral descriptors, centroid, and mass"
            ),
            "variable_cardinality": True,
            "unrestricted_dustbin": True,
            "tubes": [row.public_row() for row in self.tubes],
        }
        return {**value, "digest": _digest(value)}


def construct_space_time_tubes_v7(
    proposal_delta: torch.Tensor,
    neutral_visual: torch.Tensor,
    *,
    active_phases: Sequence[int],
    height: int,
    width: int,
    prereg: Optional[Mapping[str, Any]] = None,
) -> SpaceTimeTubeConstructionV7:
    """One joint connected-component pass over a sparse T x H x W graph."""

    spec = dict(prereg or registry.load_preregistration())
    discovery = spec["discovery"]
    tube_spec = spec["space_time_tubes"]
    patches = height * width
    _validate_feature(proposal_delta, phases=registry.PHASES, patches=patches)
    _validate_feature(neutral_visual, phases=registry.PHASES, patches=patches)
    active = tuple(int(phase) for phase in active_phases)
    if (
        not active
        or len(active) != len(set(active))
        or tuple(sorted(active)) != active
        or any(phase < 0 or phase >= registry.PHASES for phase in active)
    ):
        raise FactorialCompatibilityTubeV7Error("active phase fold differs")
    delta = proposal_delta.detach().float()
    neutral = neutral_visual.detach().float()
    eligible: set[tuple[int, int]] = set()
    soft_weight: dict[tuple[int, int], float] = {}
    for phase in active:
        energy = torch.linalg.vector_norm(delta[phase], dim=-1) / math.sqrt(
            float(delta.shape[-1])
        )
        if float(energy.max().item()) < float(discovery["absolute_energy_floor"]):
            continue
        top_count = max(
            1,
            int(
                math.ceil(
                    patches * float(discovery["spatial_concentration_top_fraction"])
                )
            ),
        )
        concentration = float(torch.topk(energy, top_count).values.sum().item()) / max(
            float(energy.sum().item()), _EPS
        )
        if concentration < float(discovery["spatial_concentration_min"]):
            continue
        median = energy.median()
        mad = (energy - median).abs().median().clamp_min(_EPS)
        z = (energy - median) / (1.4826 * mad)
        soft = torch.sigmoid(
            (z - float(discovery["component_seed_z"]))
            / float(discovery["component_soft_temperature_z"])
        )
        for patch in torch.nonzero(soft >= 0.5, as_tuple=False).reshape(-1).tolist():
            key = (phase, int(patch))
            eligible.add(key)
            soft_weight[key] = float(soft[int(patch)].item())
    if not eligible:
        return SpaceTimeTubeConstructionV7(active, (), 0, 0, 0)

    coords = _grid(height, width)
    per_phase_cap = int(discovery["maximum_components_per_phase_computational_cap"])
    spatial_rows: list[
        tuple[v6_observer.ProposalComponentV6, set[tuple[int, int]]]
    ] = []
    for phase in active:
        phase_eligible = {
            patch for item_phase, patch in eligible if item_phase == phase
        }
        visited: set[int] = set()
        phase_candidates: list[
            tuple[v6_observer.ProposalComponentV6, set[tuple[int, int]]]
        ] = []
        for start in sorted(phase_eligible):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            support_set: set[int] = set()
            while stack:
                patch = stack.pop()
                support_set.add(patch)
                for candidate in _spatial_neighbors(patch, height, width):
                    if candidate in phase_eligible and candidate not in visited:
                        visited.add(candidate)
                        stack.append(candidate)
            support = sorted(support_set)
            if len(support) < int(discovery["minimum_component_support_patches"]):
                continue
            index = torch.tensor(support, dtype=torch.long)
            weights = torch.tensor(
                [soft_weight[(phase, patch)] for patch in support],
                dtype=torch.float32,
            )
            mass = float(weights.sum().item())
            if mass < float(discovery["minimum_component_soft_mass"]):
                continue
            normalized = weights / weights.sum().clamp_min(_EPS)
            centroid = (coords[index] * normalized[:, None]).sum(dim=0)
            descriptor = (neutral[phase, index] * normalized[:, None]).sum(dim=0)
            descriptor = descriptor / torch.linalg.vector_norm(descriptor).clamp_min(_EPS)
            phase_candidates.append(
                (
                    v6_observer.ProposalComponentV6(
                        phase,
                        0,
                        tuple(support),
                        tuple(float(item) for item in weights.tolist()),
                        mass,
                        (float(centroid[0].item()), float(centroid[1].item())),
                        tuple(float(item) for item in descriptor.tolist()),
                    ),
                    {(phase, patch) for patch in support},
                )
            )
        phase_candidates.sort(
            key=lambda item: (-item[0].soft_mass, item[0].support)
        )
        for local_id, (row, selected) in enumerate(phase_candidates[:per_phase_cap]):
            spatial_rows.append(
                (
                    v6_observer.ProposalComponentV6(
                        row.phase,
                        local_id,
                        row.support,
                        row.weights,
                        row.soft_mass,
                        row.centroid,
                        row.neutral_descriptor,
                    ),
                    selected,
                )
            )

    if not spatial_rows:
        return SpaceTimeTubeConstructionV7(
            active, (), len(eligible), len(eligible), 0, 0
        )

    parent = list(range(len(spatial_rows)))
    group_phases = [{spatial_rows[index][0].phase} for index in range(len(spatial_rows))]

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> bool:
        a, b = find(left), find(right)
        if a == b:
            return True
        if group_phases[a] & group_phases[b]:
            return False
        low, high = min(a, b), max(a, b)
        parent[high] = low
        group_phases[low].update(group_phases[high])
        group_phases[high].clear()
        return True

    by_phase: dict[int, list[int]] = {}
    for index, (row, _selected) in enumerate(spatial_rows):
        by_phase.setdefault(row.phase, []).append(index)

    def ot_item(row: v6_observer.ProposalComponentV6) -> v6_observer.EvaluatedComponentV6:
        return v6_observer.EvaluatedComponentV6(
            row.phase,
            row.phase,
            row.local_id,
            row.soft_mass,
            row.centroid,
            row.neutral_descriptor,
            1.0,
            1.0,
            1.0,
        )

    maximum_gap = int(tube_spec["maximum_occlusion_gap_in_active_steps"])
    ot_spec = spec["unbalanced_ot"]
    temporal_edge_count = 0
    temporal_dustbin_count = 0
    for position, phase in enumerate(active):
        left_indices = by_phase.get(phase, [])
        if not left_indices:
            continue
        for offset in range(1, maximum_gap + 2):
            target_position = position + offset
            if target_position >= len(active):
                break
            right_indices = by_phase.get(active[target_position], [])
            if not right_indices:
                temporal_dustbin_count += len(left_indices)
                continue
            plan, cost = v6_observer.unbalanced_ot_with_dustbin_v6(
                [ot_item(spatial_rows[index][0]) for index in left_indices],
                [ot_item(spatial_rows[index][0]) for index in right_indices],
                prereg=spec,
            )
            scored = []
            for left_local, left_index in enumerate(left_indices):
                real_total = float(
                    plan[left_local, : len(right_indices)].sum().item()
                )
                dust = float(plan[left_local, len(right_indices)].item())
                denominator = max(real_total + dust, _EPS)
                for right_local, right_index in enumerate(right_indices):
                    fraction = float(plan[left_local, right_local].item()) / denominator
                    scored.append(
                        (
                            fraction,
                            -float(cost[left_local, right_local].item()),
                            -left_index,
                            -right_index,
                            left_index,
                            right_index,
                        )
                    )
            used_left: set[int] = set()
            used_right: set[int] = set()
            for fraction, negative_cost, _a, _b, left_index, right_index in sorted(
                scored, reverse=True
            ):
                if left_index in used_left or right_index in used_right:
                    continue
                if (
                    fraction < float(ot_spec["minimum_real_transport_fraction"])
                    or -negative_cost > float(ot_spec["maximum_match_cost"])
                ):
                    continue
                if not union(left_index, right_index):
                    continue
                used_left.add(left_index)
                used_right.add(right_index)
                temporal_edge_count += 1
            temporal_dustbin_count += (
                len(left_indices) - len(used_left)
                + len(right_indices) - len(used_right)
            )

    groups: dict[int, list[int]] = {}
    for index in range(len(spatial_rows)):
        groups.setdefault(find(index), []).append(index)
    candidates: list[
        tuple[
            float,
            tuple[v6_observer.ProposalComponentV6, ...],
            set[tuple[int, int]],
        ]
    ] = []
    for indices in groups.values():
        slices = tuple(
            spatial_rows[index][0]
            for index in sorted(
                indices,
                key=lambda item: (
                    spatial_rows[item][0].phase,
                    spatial_rows[item][0].local_id,
                ),
            )
        )
        if len({row.phase for row in slices}) != len(slices):
            raise FactorialCompatibilityTubeV7Error(
                "temporal OT merged multiple components in one phase"
            )
        if len(slices) < int(tube_spec["minimum_tube_observed_proposal_phases"]):
            continue
        selected = set().union(*(spatial_rows[index][1] for index in indices))
        candidates.append((sum(row.soft_mass for row in slices), slices, selected))
    candidates.sort(
        key=lambda row: (
            -row[0],
            tuple((item.phase, item.support) for item in row[1]),
        )
    )
    cap = int(tube_spec["maximum_tubes_computational_cap"])
    tubes: list[SpaceTimeTubeV7] = []
    selected_voxels: set[tuple[int, int]] = set()
    for tube_id, (mass, slices, selected) in enumerate(candidates[:cap]):
        normalized_slices = tuple(
            v6_observer.ProposalComponentV6(
                row.phase,
                tube_id,
                row.support,
                row.weights,
                row.soft_mass,
                row.centroid,
                row.neutral_descriptor,
            )
            for row in slices
        )
        tubes.append(SpaceTimeTubeV7(tube_id, normalized_slices, mass))
        selected_voxels.update(selected)
    return SpaceTimeTubeConstructionV7(
        active,
        tuple(tubes),
        len(eligible),
        len(eligible - selected_voxels),
        temporal_edge_count,
        temporal_dustbin_count,
    )


def _evaluate_tubes(
    construction: SpaceTimeTubeConstructionV7,
    evaluator_neutral: torch.Tensor,
    *,
    phase_pairs: Sequence[tuple[int, int]],
    height: int,
    width: int,
    prereg: Mapping[str, Any],
) -> tuple[
    tuple[v6_observer.TrackStateV6, ...],
    tuple[Mapping[str, Any], ...],
    int,
]:
    pair_map = dict(phase_pairs)
    evaluation_schedule = tuple(right for _left, right in phase_pairs)
    tracks: list[v6_observer.TrackStateV6] = []
    all_events: list[Mapping[str, Any]] = []
    dustbin = 0
    for tube in construction.tubes:
        observations = []
        for component in tube.components:
            if component.phase not in pair_map:
                dustbin += 1
                continue
            row = v6_observer.evaluate_component_with_neutral_tokens_v6(
                component,
                evaluator_neutral[pair_map[component.phase]],
                evaluation_phase=pair_map[component.phase],
                height=height,
                width=width,
                prereg=prereg,
            )
            if row is None:
                dustbin += 1
            else:
                observations.append(row)
        observations.sort(key=lambda row: row.evaluation_phase)
        if not observations:
            continue
        observed_phases = {row.evaluation_phase for row in observations}
        first_index = min(
            index
            for index, phase in enumerate(evaluation_schedule)
            if phase in observed_phases
        )
        last_index = max(
            index
            for index, phase in enumerate(evaluation_schedule)
            if phase in observed_phases
        )
        events: list[Mapping[str, Any]] = [
            {
                "event": "birth",
                "track_id": tube.tube_id,
                "phase": evaluation_schedule[first_index],
            }
        ]
        missing = False
        for phase in evaluation_schedule[first_index + 1 : last_index + 1]:
            if phase not in observed_phases:
                events.append(
                    {"event": "occlusion", "track_id": tube.tube_id, "phase": phase}
                )
                missing = True
            elif missing:
                events.append(
                    {"event": "reentry", "track_id": tube.tube_id, "phase": phase}
                )
                missing = False
        events.append(
            {
                "event": "death",
                "track_id": tube.tube_id,
                "phase": evaluation_schedule[last_index] + 1,
            }
        )
        track = v6_observer.TrackStateV6(
            tube.tube_id,
            observations,
            events,
            0,
            False,
        )
        tracks.append(track)
        all_events.extend(events)
    return tuple(tracks), tuple(all_events), dustbin


def _dominant_track(
    tracks: Sequence[v6_observer.TrackStateV6],
) -> Optional[v6_observer.TrackStateV6]:
    if not tracks:
        return None
    return max(tracks, key=lambda row: (len(row.observations), -row.track_id))


def _track_displacement(
    track: Optional[v6_observer.TrackStateV6],
) -> Optional[tuple[float, float]]:
    if track is None or len(track.observations) < 2:
        return None
    first, last = track.observations[0], track.observations[-1]
    return (
        last.centroid[0] - first.centroid[0],
        last.centroid[1] - first.centroid[1],
    )


def _direction_cosine(
    left: Optional[tuple[float, float]], right: Optional[tuple[float, float]]
) -> Optional[float]:
    if left is None or right is None:
        return None
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm <= _EPS or right_norm <= _EPS:
        return None
    return (left[0] * right[0] + left[1] * right[1]) / (
        left_norm * right_norm
    )


def _path_acceleration(
    track: Optional[v6_observer.TrackStateV6],
) -> Optional[float]:
    if track is None or len(track.observations) < 3:
        return None
    points = [row.centroid for row in track.observations]
    values = [
        math.hypot(
            points[index + 1][0] - 2 * points[index][0] + points[index - 1][0],
            points[index + 1][1] - 2 * points[index][1] + points[index - 1][1],
        )
        for index in range(1, len(points) - 1)
    ]
    return sum(values) / len(values)


def _tube_support_iou(
    left: SpaceTimeTubeConstructionV7,
    right: SpaceTimeTubeConstructionV7,
) -> Optional[float]:
    left_by_phase: dict[int, set[int]] = {}
    right_by_phase: dict[int, set[int]] = {}
    for tube in left.tubes:
        for row in tube.components:
            left_by_phase.setdefault(row.phase, set()).update(row.support)
    for tube in right.tubes:
        for row in tube.components:
            right_by_phase.setdefault(row.phase, set()).update(row.support)
    values = []
    for phase in sorted(set(left_by_phase) & set(right_by_phase)):
        a, b = left_by_phase[phase], right_by_phase[phase]
        if a or b:
            values.append(len(a & b) / float(len(a | b)))
    return None if not values else sum(values) / len(values)


def _component_count(construction: SpaceTimeTubeConstructionV7) -> int:
    return sum(len(tube.components) for tube in construction.tubes)


def evaluate_factorial_crossfit_branch_v7(
    factorial_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    controls_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    *,
    state_id: str,
    branch: str,
    height: int = registry.PATCH_HEIGHT,
    width: int = registry.PATCH_WIDTH,
    prereg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Evaluate one state and one branch using global tubes and all V6 gates."""

    spec = dict(prereg or registry.load_preregistration())
    if state_id not in registry.APPEARANCE_IDS or branch not in BRANCHES:
        raise FactorialCompatibilityTubeV7Error("branch state identity differs")
    if set(factorial_by_block) != set(registry.APPEARANCE_IDS) or set(
        controls_by_block
    ) != set(registry.APPEARANCE_IDS):
        raise FactorialCompatibilityTubeV7Error("factorial slab differs")
    proposer_blocks, evaluator_blocks = _branch_folds(branch)
    phase_pairs = CROSS_FIT_PHASE_PAIRS[branch]
    proposal_phases = tuple(left for left, _right in phase_pairs)
    evaluation_phases = tuple(right for _left, right in phase_pairs)
    if set(proposer_blocks) & set(evaluator_blocks) or set(proposal_phases) & set(
        evaluation_phases
    ):
        raise FactorialCompatibilityTubeV7Error("cross-fit folds overlap")

    factorial = {
        state: {
            caption: _mean_blocks(
                factorial_by_block[state][caption], proposer_blocks
            )
            for caption in registry.APPEARANCE_IDS
        }
        for state in registry.APPEARANCE_IDS
    }
    controls = {
        state: {
            arm: _mean_blocks(controls_by_block[state][arm], proposer_blocks)
            for arm in registry.CONTROL_ARMS
        }
        for state in registry.APPEARANCE_IDS
    }
    evaluator_neutral = _mean_blocks(
        controls_by_block[state_id]["neutral"], evaluator_blocks
    )
    interaction = factorial_interaction_residual_v7(
        factorial,
        {state: controls[state]["neutral"] for state in registry.APPEARANCE_IDS},
        state_id=state_id,
        branch=branch,
    )
    heldout, heldout_caption_id = factorial_heldout_residual_v7(
        factorial,
        {state: controls[state]["neutral"] for state in registry.APPEARANCE_IDS},
        state_id=state_id,
        branch=branch,
    )
    local = controls[state_id]
    deltas = {
        "action": interaction,
        "noop": torch.zeros_like(local["noop"]),
        "static": (local["static"] - local["noop"]).detach(),
        "reverse": (local["reverse"] - local["noop"]).detach(),
        "paraphrase": (local["paraphrase"] - local["noop"]).detach(),
        "lexical_placebo": (
            local["lexical_placebo"] - local["noop"]
        ).detach(),
        "source_swap": heldout,
    }
    permutation = tuple(int(item) for item in spec["controls"]["phase_shuffle"])
    if sorted(permutation) != list(range(registry.PHASES)):
        raise FactorialCompatibilityTubeV7Error(
            "phase shuffle is not a permutation"
        )
    deltas["phase_shuffle"] = interaction[
        torch.tensor(permutation, dtype=torch.long)
    ].detach()

    constructions: dict[str, SpaceTimeTubeConstructionV7] = {}
    tracks: dict[str, tuple[v6_observer.TrackStateV6, ...]] = {}
    events: dict[str, tuple[Mapping[str, Any], ...]] = {}
    evaluation_dustbin: dict[str, int] = {}
    for name, delta in deltas.items():
        construction = construct_space_time_tubes_v7(
            delta,
            local["neutral"],
            active_phases=proposal_phases,
            height=height,
            width=width,
            prereg=spec,
        )
        construction_tracks, construction_events, dustbin = _evaluate_tubes(
            construction,
            evaluator_neutral,
            phase_pairs=phase_pairs,
            height=height,
            width=width,
            prereg=spec,
        )
        constructions[name] = construction
        tracks[name] = construction_tracks
        events[name] = construction_events
        evaluation_dustbin[name] = dustbin

    qualified = {
        name: v6_observer.qualified_tracks_v6(rows, prereg=spec)
        for name, rows in tracks.items()
    }
    primary_tracks = qualified["action"]
    primary = _dominant_track(primary_tracks)
    primary_displacement = _track_displacement(primary)
    primary_norm = (
        None
        if primary_displacement is None
        else math.hypot(*primary_displacement)
    )
    static = _dominant_track(qualified["static"])
    static_displacement = _track_displacement(static)
    static_norm = 0.0 if static is None else (
        None if static_displacement is None else math.hypot(*static_displacement)
    )
    static_ratio = (
        None
        if primary_norm is None or primary_norm <= _EPS or static_norm is None
        else static_norm / primary_norm
    )
    reverse_cosine = _direction_cosine(
        primary_displacement,
        _track_displacement(_dominant_track(qualified["reverse"])),
    )
    paraphrase_cosine = _direction_cosine(
        primary_displacement,
        _track_displacement(_dominant_track(qualified["paraphrase"])),
    )
    primary_acceleration = _path_acceleration(primary)
    shuffle_acceleration = _path_acceleration(
        _dominant_track(qualified["phase_shuffle"])
    )
    shuffle_pass, shuffle_ratio = v6_observer.phase_shuffle_gate_v6(
        primary_acceleration, shuffle_acceleration, prereg=spec
    )
    component_counts = {
        name: _component_count(row) for name, row in constructions.items()
    }
    primary_component_count = component_counts["action"]
    lexical_ratio = (
        None
        if primary_component_count <= 0
        else component_counts["lexical_placebo"]
        / float(primary_component_count)
    )
    primary_coverage = (
        0.0
        if primary is None
        else len(primary.observations) / float(len(evaluation_phases))
    )
    primary_edges = v6_observer.dynamic_edge_lifecycle_v6(
        primary_tracks, prereg=spec
    )
    primary_lifecycle_count = sum(
        row["event"] in {"activate", "deactivate", "endpoint_death"}
        for row in primary_edges
    )
    source_track = _dominant_track(qualified["source_swap"])
    source_coverage = (
        0.0
        if source_track is None
        else len(source_track.observations) / float(len(evaluation_phases))
    )
    source_edges = v6_observer.dynamic_edge_lifecycle_v6(
        qualified["source_swap"], prereg=spec
    )
    source_lifecycle_count = sum(
        row["event"] in {"activate", "deactivate", "endpoint_death"}
        for row in source_edges
    )
    gates = v6_observer.branch_gate_decision_v6(
        primary_track_count=len(primary_tracks),
        primary_coverage=primary_coverage,
        primary_lifecycle_count=primary_lifecycle_count,
        component_counts=component_counts,
        static_ratio=static_ratio,
        reverse_cosine=reverse_cosine,
        phase_shuffle_pass=shuffle_pass,
        paraphrase_iou=_tube_support_iou(
            constructions["action"], constructions["paraphrase"]
        ),
        paraphrase_cosine=paraphrase_cosine,
        lexical_ratio=lexical_ratio,
        source_swap_iou=_tube_support_iou(
            constructions["action"], constructions["source_swap"]
        ),
        source_swap_coverage=source_coverage,
        source_swap_lifecycle_count=source_lifecycle_count,
        prereg=spec,
    )
    metrics = {
        "primary_track_coverage": primary_coverage,
        "primary_dynamic_edge_lifecycle_count": primary_lifecycle_count,
        "static_to_primary_displacement_ratio": static_ratio,
        "reverse_endpoint_direction_cosine": reverse_cosine,
        "phase_shuffle_to_primary_acceleration_ratio": shuffle_ratio,
        "paraphrase_support_iou": _tube_support_iou(
            constructions["action"], constructions["paraphrase"]
        ),
        "paraphrase_endpoint_direction_cosine": paraphrase_cosine,
        "lexical_placebo_to_primary_component_ratio": lexical_ratio,
        "source_swap_to_primary_support_iou": _tube_support_iou(
            constructions["action"], constructions["source_swap"]
        ),
        "source_swap_evaluated_track_coverage": source_coverage,
        "source_swap_dynamic_edge_lifecycle_count": source_lifecycle_count,
        "primary_path_acceleration": primary_acceleration,
        "phase_shuffle_path_acceleration": shuffle_acceleration,
    }
    branch_pass = all(gates.values())
    value = {
        "branch": branch,
        "state_appearance_id": state_id,
        "proposal_layer_fold": list(proposer_blocks),
        "evaluation_layer_fold": list(evaluator_blocks),
        "proposal_time_fold": list(proposal_phases),
        "evaluation_time_fold": list(evaluation_phases),
        "actual_phase_pairs": [list(row) for row in phase_pairs],
        "layers_disjoint": True,
        "times_disjoint": True,
        "nuisance_off_diagonal": [
            list(row)
            for row in registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["nuisance"]
        ],
        "heldout_source_swap_off_diagonal": [
            list(row)
            for row in registry.BRANCH_OFF_DIAGONAL_FOLDS[branch]["heldout"]
        ],
        "off_diagonal_folds_disjoint": True,
        "heldout_source_swap_caption_appearance_id": heldout_caption_id,
        "interaction_formula": (
            "(matched_action-identical_neutral)-"
            "(same_caption_nuisance_state-identical_neutral)"
        ),
        "appearance_and_caption_main_effects_removed": True,
        "interaction_residual_stop_gradient_proposal_only": True,
        "interaction_residual_used_as_descriptor": False,
        "interaction_residual_used_as_reward": False,
        "correspondence_descriptor": (
            "prompt_neutral_visual_query_hidden_sketch"
        ),
        "tube_construction": {
            name: row.public_row() for name, row in constructions.items()
        },
        "component_counts": component_counts,
        "track_counts": {name: len(row) for name, row in tracks.items()},
        "qualified_track_counts": {
            name: len(row) for name, row in qualified.items()
        },
        "evaluation_dustbin_counts": evaluation_dustbin,
        "control_executed": {name: True for name in CONTROL_NAMES},
        "metrics": metrics,
        "gates": gates,
        "graph_abstained": not gates["primary_graph_valid"],
        "R0_compensation_applied": False,
        "branch_pass": branch_pass,
        "primary_tracks": [
            {
                "track_id": track.track_id,
                "observed_phases": [
                    row.evaluation_phase for row in track.observations
                ],
                "event_kinds": [row["event"] for row in track.events],
                "semantic_role": None,
            }
            for track in primary_tracks
        ],
        "primary_dynamic_edge_lifecycle": list(primary_edges),
    }
    return {**value, "digest": _digest(value)}


@dataclass(frozen=True)
class ReducedFactorialTubeCellV7:
    appearance_id: str
    sigma_band: str
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    branch_receipts: Mapping[str, Mapping[str, Any]]
    branchwise_diagnostic_admitted: bool
    projected_inputs_zeroized: bool

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_id": self.appearance_id,
            "sigma_band": self.sigma_band,
            "state_sha256": self.state_sha256,
            "timestep_sha256": self.timestep_sha256,
            "rotary_sha256": self.rotary_sha256,
            "same_nontext_identity_within_state": True,
            "full_three_by_three_factorial_consumed": True,
            "branch_receipts": {
                branch: dict(self.branch_receipts[branch]) for branch in BRANCHES
            },
            "branchwise_formula": "A_to_B AND B_to_A",
            "branchwise_diagnostic_admitted": self.branchwise_diagnostic_admitted,
            "projected_inputs_zeroized": self.projected_inputs_zeroized,
            "representation_admitted": False,
            "stable_transferable_action_representation_claimed": False,
            "scientific_claim_authorized": False,
        }
        return {**value, "digest": _digest(value)}


def _validate_feature_slab(
    factorial_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    controls_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
) -> None:
    if set(factorial_by_block) != set(registry.APPEARANCE_IDS) or set(
        controls_by_block
    ) != set(registry.APPEARANCE_IDS):
        raise FactorialCompatibilityTubeV7Error("feature slab states differ")
    for state_id in registry.APPEARANCE_IDS:
        if set(factorial_by_block[state_id]) != set(registry.APPEARANCE_IDS):
            raise FactorialCompatibilityTubeV7Error(
                "full three by three factorial is required"
            )
        if set(controls_by_block[state_id]) != set(registry.CONTROL_ARMS):
            raise FactorialCompatibilityTubeV7Error("control arm slab differs")
        for rows in (
            list(factorial_by_block[state_id].values())
            + list(controls_by_block[state_id].values())
        ):
            if set(rows) != set(registry.BLOCKS):
                raise FactorialCompatibilityTubeV7Error(
                    "feature slab block set differs"
                )
            for value in rows.values():
                _validate_feature(
                    value,
                    phases=registry.PHASES,
                    patches=registry.PATCHES,
                )


def reduce_factorial_feature_slab_v7(
    factorial_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    controls_by_block: Mapping[str, Mapping[str, Mapping[int, torch.Tensor]]],
    *,
    sigma_band: str,
    identity_by_state: Optional[Mapping[str, Mapping[str, str]]] = None,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[ReducedFactorialTubeCellV7, ...]:
    """Pure CPU reduction of one complete three-state sigma slab."""

    spec = dict(prereg or registry.load_preregistration())
    _validate_feature_slab(factorial_by_block, controls_by_block)
    if sigma_band not in registry.SIGMA_CELL_INDICES:
        raise FactorialCompatibilityTubeV7Error("sigma band differs")
    identities = identity_by_state or {
        state_id: {
            "state_sha256": hashlib.sha256(
                f"V7-test-state:{state_id}:{sigma_band}".encode("ascii")
            ).hexdigest(),
            "timestep_sha256": hashlib.sha256(
                f"V7-test-timestep:{sigma_band}".encode("ascii")
            ).hexdigest(),
            "rotary_sha256": hashlib.sha256(
                f"V7-test-rotary:{sigma_band}".encode("ascii")
            ).hexdigest(),
        }
        for state_id in registry.APPEARANCE_IDS
    }
    if set(identities) != set(registry.APPEARANCE_IDS):
        raise FactorialCompatibilityTubeV7Error("state identity slab differs")
    rows = []
    for state_id in registry.APPEARANCE_IDS:
        identity = identities[state_id]
        if set(identity) != {"state_sha256", "timestep_sha256", "rotary_sha256"} or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in identity.values()
        ):
            raise FactorialCompatibilityTubeV7Error(
                "state identity digest differs"
            )
        branches = {
            branch: evaluate_factorial_crossfit_branch_v7(
                factorial_by_block,
                controls_by_block,
                state_id=state_id,
                branch=branch,
                prereg=spec,
            )
            for branch in BRANCHES
        }
        admitted = all(branches[branch]["branch_pass"] for branch in BRANCHES)
        rows.append(
            ReducedFactorialTubeCellV7(
                state_id,
                sigma_band,
                identity["state_sha256"],
                identity["timestep_sha256"],
                identity["rotary_sha256"],
                branches,
                admitted,
                False,
            )
        )
    return tuple(rows)


def reduce_factorial_capture_slab_v7(
    factorial_captures: Mapping[
        str,
        Mapping[str, Mapping[int, v6_observer.AnonymousProjectedArmV6]],
    ],
    control_captures: Mapping[
        str,
        Mapping[str, Mapping[int, v6_observer.AnonymousProjectedArmV6]],
    ],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[ReducedFactorialTubeCellV7, ...]:
    """Consume one V6-ABI capture slab and always scrub projected ownership."""

    # Ownership transfers at invocation, before structural validation.  Gather
    # every recognizable V6 projected row first so malformed outer mappings
    # cannot leave an unvisited capture resident on an exception path.
    owned: list[v6_observer.AnonymousProjectedArmV6] = []
    for outer in (factorial_captures, control_captures):
        if not isinstance(outer, Mapping):
            continue
        for middle in outer.values():
            if not isinstance(middle, Mapping):
                continue
            for inner in middle.values():
                if not isinstance(inner, Mapping):
                    continue
                for row in inner.values():
                    if isinstance(row, v6_observer.AnonymousProjectedArmV6):
                        owned.append(row)
    cpu: list[torch.Tensor] = []
    try:
        if set(factorial_captures) != set(registry.APPEARANCE_IDS) or set(
            control_captures
        ) != set(registry.APPEARANCE_IDS):
            raise FactorialCompatibilityTubeV7Error("capture slab states differ")
        for state_id in registry.APPEARANCE_IDS:
            if set(factorial_captures[state_id]) != set(registry.APPEARANCE_IDS):
                raise FactorialCompatibilityTubeV7Error(
                    "full three by three factorial is required"
                )
            if set(control_captures[state_id]) != set(registry.CONTROL_ARMS):
                raise FactorialCompatibilityTubeV7Error(
                    "control capture slab differs"
                )
            for caption_id in registry.APPEARANCE_IDS:
                rows = factorial_captures[state_id][caption_id]
                if set(rows) != set(registry.BLOCKS):
                    raise FactorialCompatibilityTubeV7Error(
                        "factorial capture blocks differ"
                )
                for block in registry.BLOCKS:
                    row = rows[block]
                    if not isinstance(row, v6_observer.AnonymousProjectedArmV6):
                        raise FactorialCompatibilityTubeV7Error(
                            "factorial capture type differs"
                        )
                    row.validate()
                    expected_arm = "action" if state_id == caption_id else "source_swap"
                    if (
                        row.appearance_id != state_id
                        or row.arm != expected_arm
                        or row.block_index != block
                    ):
                        raise FactorialCompatibilityTubeV7Error(
                            "factorial V6 capture identity differs"
                        )
            for arm in registry.CONTROL_ARMS:
                rows = control_captures[state_id][arm]
                if set(rows) != set(registry.BLOCKS):
                    raise FactorialCompatibilityTubeV7Error(
                        "control capture blocks differ"
                )
                for block in registry.BLOCKS:
                    row = rows[block]
                    if not isinstance(row, v6_observer.AnonymousProjectedArmV6):
                        raise FactorialCompatibilityTubeV7Error(
                            "control capture type differs"
                        )
                    row.validate()
                    if (
                        row.appearance_id != state_id
                        or row.arm != arm
                        or row.block_index != block
                    ):
                        raise FactorialCompatibilityTubeV7Error(
                            "control V6 capture identity differs"
                        )
        if len({id(row) for row in owned}) != len(owned):
            raise FactorialCompatibilityTubeV7Error(
                "capture ownership is aliased"
            )
        storages = []
        for row in owned:
            storages.extend(
                (
                    (row.query_sketch.device, row.query_sketch.untyped_storage().data_ptr()),
                    (row.hidden_sketch.device, row.hidden_sketch.untyped_storage().data_ptr()),
                )
            )
        if len(set(storages)) != len(storages):
            raise FactorialCompatibilityTubeV7Error(
                "capture tensor storage ownership is aliased"
            )
        sigma_values = {row.sigma_band for row in owned}
        projection_values = {row.projection_digest for row in owned}
        if len(sigma_values) != 1 or len(projection_values) != 1:
            raise FactorialCompatibilityTubeV7Error(
                "factorial slab sigma or projection differs"
            )
        identities: dict[str, Mapping[str, str]] = {}
        for state_id in registry.APPEARANCE_IDS:
            state_rows = [row for row in owned if row.appearance_id == state_id]
            for field_name in (
                "state_sha256",
                "timestep_sha256",
                "rotary_sha256",
            ):
                if len({getattr(row, field_name) for row in state_rows}) != 1:
                    raise FactorialCompatibilityTubeV7Error(
                        f"same-state {field_name} differs"
                    )
            identities[state_id] = {
                "state_sha256": state_rows[0].state_sha256,
                "timestep_sha256": state_rows[0].timestep_sha256,
                "rotary_sha256": state_rows[0].rotary_sha256,
            }
        factorial_features: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        control_features: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        for state_id in registry.APPEARANCE_IDS:
            factorial_features[state_id] = {}
            for caption_id in registry.APPEARANCE_IDS:
                factorial_features[state_id][caption_id] = {}
                for block, row in factorial_captures[state_id][caption_id].items():
                    query = row.query_sketch.detach().to(
                        device="cpu", dtype=torch.float32
                    )[0].contiguous()
                    cpu.append(query)
                    hidden = row.hidden_sketch.detach().to(
                        device="cpu", dtype=torch.float32
                    )[0].contiguous()
                    cpu.append(hidden)
                    combined = torch.cat((query, hidden), dim=-1).contiguous()
                    cpu.append(combined)
                    factorial_features[state_id][caption_id][block] = combined
            control_features[state_id] = {}
            for arm in registry.CONTROL_ARMS:
                control_features[state_id][arm] = {}
                for block, row in control_captures[state_id][arm].items():
                    query = row.query_sketch.detach().to(
                        device="cpu", dtype=torch.float32
                    )[0].contiguous()
                    cpu.append(query)
                    hidden = row.hidden_sketch.detach().to(
                        device="cpu", dtype=torch.float32
                    )[0].contiguous()
                    cpu.append(hidden)
                    combined = torch.cat((query, hidden), dim=-1).contiguous()
                    cpu.append(combined)
                    control_features[state_id][arm][block] = combined
        reduced = reduce_factorial_feature_slab_v7(
            factorial_features,
            control_features,
            sigma_band=next(iter(sigma_values)),
            identity_by_state=identities,
            prereg=prereg,
        )
        return tuple(
            ReducedFactorialTubeCellV7(
                row.appearance_id,
                row.sigma_band,
                row.state_sha256,
                row.timestep_sha256,
                row.rotary_sha256,
                row.branch_receipts,
                row.branchwise_diagnostic_admitted,
                True,
            )
            for row in reduced
        )
    finally:
        unique_owned = {id(row): row for row in owned}
        for row in unique_owned.values():
            row.zeroize()
        _zeroize(cpu)


class FactorialCompatibilityTubeObserverV7:
    """Nine-cell fail-closed accumulator; never admits a representation."""

    def __init__(self, prereg: Optional[Mapping[str, Any]] = None) -> None:
        self.prereg = dict(prereg or registry.load_preregistration())
        self._cells: dict[tuple[str, str], ReducedFactorialTubeCellV7] = {}
        self._closed = False

    def add(self, cell: ReducedFactorialTubeCellV7) -> None:
        if self._closed or not isinstance(cell, ReducedFactorialTubeCellV7):
            raise FactorialCompatibilityTubeV7Error(
                "V7 stream is closed or cell differs"
            )
        key = (cell.appearance_id, cell.sigma_band)
        if (
            cell.appearance_id not in registry.APPEARANCE_IDS
            or cell.sigma_band not in registry.SIGMA_CELL_INDICES
            or key in self._cells
        ):
            raise FactorialCompatibilityTubeV7Error(
                "duplicate or unknown V7 cell"
            )
        self._cells[key] = cell

    def abort(self) -> None:
        self._cells.clear()
        self._closed = True

    def finalize(self) -> Mapping[str, Any]:
        if self._closed:
            raise FactorialCompatibilityTubeV7Error("V7 stream is closed")
        expected = {
            (appearance, sigma)
            for appearance in registry.APPEARANCE_IDS
            for sigma in registry.SIGMA_CELL_INDICES
        }
        if set(self._cells) != expected:
            raise FactorialCompatibilityTubeV7Error(
                "V7 capture matrix is incomplete"
            )
        self._closed = True
        rows = [self._cells[key].receipt() for key in sorted(self._cells)]
        count = sum(row["branchwise_diagnostic_admitted"] for row in rows)
        all_controls = all(
            all(
                all(receipt["control_executed"].values())
                for receipt in row["branch_receipts"].values()
            )
            for row in rows
        )
        value = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_count": len(rows),
            "cells": rows,
            "branchwise_diagnostic_admitted_cell_count": count,
            "diagnostic_component_admitted": count == len(expected),
            "diagnostic_component_status": (
                "ADMITTED_9_OF_9" if count == len(expected) else "REJECTED"
            ),
            "overall_aggregation_formula": (
                "all 9/9 cells pass A_to_B AND B_to_A"
            ),
            "cell_selection_or_compensation_permitted": False,
            "all_controls_executed": all_controls,
            "full_three_by_three_factorial_per_sigma": True,
            "clockwise_anti_clockwise_crossfit_disjoint": True,
            "appearance_and_caption_main_effects_removed_closed_form": True,
            "joint_space_time_tube_domain": [
                registry.PHASES,
                registry.PATCH_HEIGHT,
                registry.PATCH_WIDTH,
            ],
            "anonymous_variable_cardinality_tubes": True,
            "unrestricted_dustbin": True,
            "slot_lifecycle_events": ["birth", "occlusion", "reentry", "death"],
            "caption_role_token_localization_used": False,
            "fixed_semantic_role_inventory_used": False,
            "prompt_shuffle_control_executed": False,
            "heldout_transfer_control_executed": False,
            "representation_admitted": False,
            "stable_transferable_action_representation_claimed": False,
            "scientific_claim_authorized": False,
            "training_or_parameter_updates_authorized": False,
            "renderer_or_decoder_authorized": False,
            "route_or_injection_authorized": False,
            "renderer_called": False,
            "decoder_called": False,
            "optimizer_created": False,
            "parameter_updates": 0,
            "route_or_injection_called": False,
            "gpu_launch_authorized": False,
            "gpu_runner_implemented": False,
            "factorial_prompt_embedding_runtime_binding_implemented": False,
            "launch_blocked_pending_independent_audit": True,
        }
        return {**value, "digest": _digest(value)}


__all__ = [
    "BRANCHES",
    "CONTROL_NAMES",
    "CROSS_FIT_PHASE_PAIRS",
    "FactorialCompatibilityTubeObserverV7",
    "FactorialCompatibilityTubeV7Error",
    "LAYER_FOLDS",
    "METHOD",
    "ReducedFactorialTubeCellV7",
    "SCHEMA_VERSION",
    "SpaceTimeTubeConstructionV7",
    "SpaceTimeTubeV7",
    "TIME_FOLDS",
    "construct_space_time_tubes_v7",
    "evaluate_factorial_crossfit_branch_v7",
    "factorial_heldout_residual_v7",
    "factorial_interaction_residual_v7",
    "reduce_factorial_capture_slab_v7",
    "reduce_factorial_feature_slab_v7",
]

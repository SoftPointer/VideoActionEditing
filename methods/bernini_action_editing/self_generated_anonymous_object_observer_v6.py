#!/usr/bin/env python3
"""Anonymous, cross-fitted object discovery for the V6 diagnostic probe.

The action-minus-noop residual is used only as a detached proposal field.
Descriptors, correspondence, trajectories and graph measurements are formed
from the separate prompt-neutral visual arm at disjoint layer/time folds.
There is no semantic role inventory and no fixed slot cardinality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence

import torch

import anonymous_visual_projection_hook_v6 as hook
import self_generated_anonymous_object_registry_v6 as registry


METHOD = "bernini-self-generated-anonymous-object-observer-v6"
SCHEMA_VERSION = "bernini-self-generated-anonymous-object-observer-v6"
BRANCHES = ("A_to_B", "B_to_A")
LAYER_FOLDS = {"A": (6, 18), "B": (12, 24)}
TIME_FOLDS = {
    "A": tuple(range(0, registry.PHASES, 2)),
    "B": tuple(range(1, registry.PHASES, 2)),
}
CROSS_FIT_PHASE_PAIRS = {
    "A_to_B": tuple((phase, phase + 1) for phase in range(0, 20, 2)),
    "B_to_A": tuple((phase, phase + 1) for phase in range(1, 20, 2)),
}
CONTROL_ARMS = (
    "noop",
    "static",
    "reverse",
    "phase_shuffle",
    "paraphrase",
    "lexical_placebo",
    "source_swap",
)
_EPS = 1.0e-8


class AnonymousObjectObserverV6Error(RuntimeError):
    """An anonymous discovery, cross-fit, ownership, or claim gate failed."""


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
        raise AnonymousObjectObserverV6Error(
            "receipt is not canonical finite JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _zeroize(values: Sequence[torch.Tensor]) -> None:
    with torch.inference_mode():
        for value in values:
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                value.zero_()


@dataclass
class AnonymousProjectedArmV6:
    appearance_id: str
    arm: str
    sigma_band: str
    block_index: int
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    projection_digest: str
    query_sketch: torch.Tensor
    hidden_sketch: torch.Tensor
    consumed: bool = False

    @classmethod
    def from_capture(
        cls, capture: hook.ProjectedVisualCaptureV6
    ) -> "AnonymousProjectedArmV6":
        capture.validate()
        result = cls(
            capture.identity.appearance_id,
            capture.identity.arm,
            capture.identity.sigma_band,
            capture.block_index,
            capture.identity.state_sha256,
            capture.identity.timestep_sha256,
            capture.identity.rotary_sha256,
            capture.projection_digest,
            capture.query_sketch,
            capture.hidden_sketch,
        )
        # Atomic ownership transfer: detach the source handle from the live
        # storage so even an accidental source.zeroize() cannot mutate the new
        # owner.  Allocate both tombstones before changing either source field,
        # so an allocation failure leaves the original capture as sole owner.
        result.validate()
        empty_query = torch.empty(
            (0,), dtype=result.query_sketch.dtype, device=result.query_sketch.device
        )
        empty_hidden = torch.empty(
            (0,), dtype=result.hidden_sketch.dtype, device=result.hidden_sketch.device
        )
        capture.query_sketch = empty_query
        capture.hidden_sketch = empty_hidden
        capture.consumed = True
        return result

    def validate(self) -> None:
        if (
            self.appearance_id not in registry.APPEARANCE_IDS
            or self.arm not in registry.ARMS
            or self.sigma_band not in registry.SIGMA_CELL_INDICES
            or self.block_index not in registry.BLOCKS
            or self.consumed
        ):
            raise AnonymousObjectObserverV6Error("projected arm identity differs")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                self.state_sha256,
                self.timestep_sha256,
                self.rotary_sha256,
                self.projection_digest,
            )
        ):
            raise AnonymousObjectObserverV6Error("projected arm digest differs")
        expected_q = (
            1,
            registry.PHASES,
            registry.PATCHES,
            hook.QUERY_SKETCH_DIM,
        )
        expected_h = (
            1,
            registry.PHASES,
            registry.PATCHES,
            hook.HIDDEN_SKETCH_DIM,
        )
        if (
            tuple(self.query_sketch.shape) != expected_q
            or tuple(self.hidden_sketch.shape) != expected_h
            or self.query_sketch.dtype != torch.float32
            or self.hidden_sketch.dtype != torch.float32
            or self.query_sketch.device != self.hidden_sketch.device
            or not bool(torch.isfinite(self.query_sketch).all().item())
            or not bool(torch.isfinite(self.hidden_sketch).all().item())
        ):
            raise AnonymousObjectObserverV6Error("projected arm geometry differs")

    def zeroize(self) -> None:
        _zeroize((self.query_sketch, self.hidden_sketch))
        self.consumed = True


@dataclass(frozen=True)
class ProposalComponentV6:
    phase: int
    local_id: int
    support: tuple[int, ...]
    weights: tuple[float, ...]
    soft_mass: float
    centroid: tuple[float, float]
    neutral_descriptor: tuple[float, ...]

    def public_row(self) -> Mapping[str, Any]:
        value = {
            "phase": self.phase,
            "local_id": self.local_id,
            "support_size": len(self.support),
            "soft_mass": self.soft_mass,
            "centroid": list(self.centroid),
            "descriptor_digest": _digest(list(self.neutral_descriptor)),
            "semantic_role": None,
        }
        return {**value, "digest": _digest(value)}


@dataclass(frozen=True)
class EvaluatedComponentV6:
    proposal_phase: int
    evaluation_phase: int
    local_id: int
    mass: float
    centroid: tuple[float, float]
    descriptor: tuple[float, ...]
    neutral_visual_cosine_margin: float
    top_vs_median_margin: float
    top10_mass_fraction: float


@dataclass
class TrackStateV6:
    track_id: int
    observations: list[EvaluatedComponentV6] = field(default_factory=list)
    events: list[Mapping[str, Any]] = field(default_factory=list)
    missing_evaluation_steps: int = 0
    alive: bool = True

    @property
    def last(self) -> EvaluatedComponentV6:
        return self.observations[-1]


def _normalized_grid(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)


def _neighbors(index: int, height: int, width: int) -> tuple[int, ...]:
    y, x = divmod(index, width)
    rows = []
    if y > 0:
        rows.append(index - width)
    if y + 1 < height:
        rows.append(index + width)
    if x > 0:
        rows.append(index - 1)
    if x + 1 < width:
        rows.append(index + 1)
    return tuple(rows)


def discover_soft_components_v6(
    proposal_delta: torch.Tensor,
    neutral_visual: torch.Tensor,
    *,
    phase: int,
    height: int,
    width: int,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[ProposalComponentV6, ...]:
    """Connected proposals from a detached residual; neutral descriptors only."""

    spec = dict(prereg or registry.load_preregistration())
    config = spec["discovery"]
    patches = height * width
    if (
        proposal_delta.ndim != 2
        or neutral_visual.ndim != 2
        or proposal_delta.shape[0] != patches
        or neutral_visual.shape[0] != patches
        or proposal_delta.requires_grad
        or neutral_visual.requires_grad
        or not bool(torch.isfinite(proposal_delta).all().item())
        or not bool(torch.isfinite(neutral_visual).all().item())
    ):
        raise AnonymousObjectObserverV6Error("component proposal ABI differs")
    # The residual is deliberately detached and never copied into a descriptor.
    delta = proposal_delta.detach().float()
    energy = torch.linalg.vector_norm(delta, dim=-1) / math.sqrt(float(delta.shape[-1]))
    if float(energy.max().item()) < float(config["absolute_energy_floor"]):
        return ()
    median = energy.median()
    mad = (energy - median).abs().median().clamp_min(_EPS)
    z = (energy - median) / (1.4826 * mad)
    temperature = float(config["component_soft_temperature_z"])
    seed_z = float(config["component_seed_z"])
    soft = torch.sigmoid((z - seed_z) / temperature)
    seed = soft >= 0.5
    top_count = max(1, int(math.ceil(patches * float(config["spatial_concentration_top_fraction"]))))
    concentration = float(torch.topk(energy, top_count).values.sum().item()) / max(
        float(energy.sum().item()), _EPS
    )
    if concentration < float(config["spatial_concentration_min"]):
        return ()

    coords = _normalized_grid(height, width)
    visited: set[int] = set()
    rows: list[ProposalComponentV6] = []
    for start in torch.nonzero(seed, as_tuple=False).reshape(-1).tolist():
        if start in visited:
            continue
        stack = [int(start)]
        support: list[int] = []
        visited.add(int(start))
        while stack:
            item = stack.pop()
            support.append(item)
            for neighbor in _neighbors(item, height, width):
                if neighbor not in visited and bool(seed[neighbor].item()):
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(support) < int(config["minimum_component_support_patches"]):
            continue
        index = torch.tensor(sorted(support), dtype=torch.long)
        weights = soft[index]
        mass = float(weights.sum().item())
        if mass < float(config["minimum_component_soft_mass"]):
            continue
        normalized = weights / weights.sum().clamp_min(_EPS)
        centroid = (coords[index] * normalized[:, None]).sum(dim=0)
        descriptor = (
            neutral_visual.detach().float()[index] * normalized[:, None]
        ).sum(dim=0)
        descriptor = descriptor / torch.linalg.vector_norm(descriptor).clamp_min(_EPS)
        rows.append(
            ProposalComponentV6(
                phase=int(phase),
                local_id=len(rows),
                support=tuple(int(item) for item in index.tolist()),
                weights=tuple(float(item) for item in weights.tolist()),
                soft_mass=mass,
                centroid=(float(centroid[0].item()), float(centroid[1].item())),
                neutral_descriptor=tuple(float(item) for item in descriptor.tolist()),
            )
        )
    rows.sort(key=lambda row: (-row.soft_mass, row.support))
    cap = int(config["maximum_components_per_phase_computational_cap"])
    return tuple(
        ProposalComponentV6(
            row.phase,
            index,
            row.support,
            row.weights,
            row.soft_mass,
            row.centroid,
            row.neutral_descriptor,
        )
        for index, row in enumerate(rows[:cap])
    )


def evaluate_component_with_neutral_tokens_v6(
    component: ProposalComponentV6,
    neutral_evaluation: torch.Tensor,
    *,
    evaluation_phase: int,
    height: int,
    width: int,
    prereg: Optional[Mapping[str, Any]] = None,
) -> Optional[EvaluatedComponentV6]:
    spec = dict(prereg or registry.load_preregistration())
    config = spec["cross_fit"]
    if (
        neutral_evaluation.ndim != 2
        or neutral_evaluation.shape[0] != height * width
        or neutral_evaluation.requires_grad
        or not bool(torch.isfinite(neutral_evaluation).all().item())
    ):
        raise AnonymousObjectObserverV6Error("neutral evaluation ABI differs")
    tokens = neutral_evaluation.detach().float()
    tokens = tokens / torch.linalg.vector_norm(tokens, dim=-1, keepdim=True).clamp_min(_EPS)
    descriptor = torch.tensor(component.neutral_descriptor, dtype=torch.float32)
    if descriptor.numel() != tokens.shape[-1]:
        raise AnonymousObjectObserverV6Error("neutral descriptor width differs")
    coords = _normalized_grid(height, width)
    source = torch.tensor(component.centroid, dtype=torch.float32)
    cosine = tokens @ descriptor
    visual_margin = float(cosine.max().item() - cosine.median().item())
    if visual_margin < float(
        config["neutral_visual_cosine_top_vs_median_margin_min"]
    ):
        # Geometry is forbidden from making an uninformative neutral visual
        # field look correspondable.
        return None
    squared_distance = ((coords - source) ** 2).sum(dim=-1)
    sigma = float(config["correspondence_spatial_sigma"])
    logits = cosine - squared_distance / (2.0 * sigma * sigma)
    top = float(logits.max().item())
    median = float(logits.median().item())
    margin = top - median
    probability = torch.softmax(
        logits / float(config["correspondence_softmax_temperature"]), dim=0
    )
    top_count = max(1, int(math.ceil(probability.numel() * 0.1)))
    concentration = float(torch.topk(probability, top_count).values.sum().item())
    if (
        margin < float(config["correspondence_top_vs_median_margin_min"])
        or concentration < float(config["correspondence_top10_mass_fraction_min"])
    ):
        return None
    centroid = (coords * probability[:, None]).sum(dim=0)
    evaluated_descriptor = (tokens * probability[:, None]).sum(dim=0)
    evaluated_descriptor = evaluated_descriptor / torch.linalg.vector_norm(
        evaluated_descriptor
    ).clamp_min(_EPS)
    return EvaluatedComponentV6(
        proposal_phase=component.phase,
        evaluation_phase=int(evaluation_phase),
        local_id=component.local_id,
        mass=float(probability.max().item()),
        centroid=(float(centroid[0].item()), float(centroid[1].item())),
        descriptor=tuple(float(item) for item in evaluated_descriptor.tolist()),
        neutral_visual_cosine_margin=visual_margin,
        top_vs_median_margin=margin,
        top10_mass_fraction=concentration,
    )


def unbalanced_ot_with_dustbin_v6(
    previous: Sequence[EvaluatedComponentV6],
    current: Sequence[EvaluatedComponentV6],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augmented Sinkhorn plan with explicit source/target dustbins."""

    spec = dict(prereg or registry.load_preregistration())
    config = spec["unbalanced_ot"]
    n, m = len(previous), len(current)
    cost = torch.full((n + 1, m + 1), float(config["dustbin_cost"]), dtype=torch.float64)
    cost[n, m] = 0.0
    for i, left in enumerate(previous):
        left_descriptor = torch.tensor(left.descriptor, dtype=torch.float64)
        for j, right in enumerate(current):
            right_descriptor = torch.tensor(right.descriptor, dtype=torch.float64)
            spatial = math.dist(left.centroid, right.centroid) / math.sqrt(8.0)
            cosine = float(
                torch.dot(left_descriptor, right_descriptor)
                / (
                    torch.linalg.vector_norm(left_descriptor)
                    * torch.linalg.vector_norm(right_descriptor)
                ).clamp_min(_EPS)
            )
            mass = abs(math.log(max(left.mass, _EPS) / max(right.mass, _EPS)))
            cost[i, j] = (
                float(config["spatial_cost_weight"]) * spatial
                + float(config["embedding_cost_weight"]) * (1.0 - cosine)
                + float(config["log_mass_cost_weight"]) * mass
            )
    left_mass = torch.tensor([max(item.mass, _EPS) for item in previous], dtype=torch.float64)
    right_mass = torch.tensor([max(item.mass, _EPS) for item in current], dtype=torch.float64)
    left_total, right_total = float(left_mass.sum()), float(right_mass.sum())
    supply = torch.cat((left_mass, torch.tensor([right_total], dtype=torch.float64)))
    demand = torch.cat((right_mass, torch.tensor([left_total], dtype=torch.float64)))
    total = max(float(supply.sum()), _EPS)
    supply, demand = supply / total, demand / total
    kernel = torch.exp(-cost / float(config["epsilon"])).clamp_min(1.0e-300)
    u = torch.ones_like(supply)
    v = torch.ones_like(demand)
    for _ in range(int(config["iterations"])):
        u = supply / (kernel @ v).clamp_min(1.0e-300)
        v = demand / (kernel.transpose(0, 1) @ u).clamp_min(1.0e-300)
    plan = u[:, None] * kernel * v[None, :]
    if not bool(torch.isfinite(plan).all().item()):
        raise AnonymousObjectObserverV6Error("unbalanced OT is non-finite")
    return plan.float(), cost.float()


def track_hypotheses_v6(
    components_by_phase: Mapping[int, Sequence[EvaluatedComponentV6]],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[tuple[TrackStateV6, ...], tuple[Mapping[str, Any], ...]]:
    spec = dict(prereg or registry.load_preregistration())
    ot = spec["unbalanced_ot"]
    tracking = spec["tracking"]
    tracks: list[TrackStateV6] = []
    events: list[Mapping[str, Any]] = []
    next_id = 0
    for phase in sorted(components_by_phase):
        current = list(components_by_phase[phase])
        candidates = [
            track
            for track in tracks
            if track.alive
            and track.missing_evaluation_steps <= int(tracking["maximum_occlusion_gap"])
        ]
        assignments: dict[int, int] = {}
        if candidates and current:
            plan, cost = unbalanced_ot_with_dustbin_v6(
                [track.last for track in candidates], current, prereg=spec
            )
            scored = []
            for i, track in enumerate(candidates):
                real_total = float(plan[i, : len(current)].sum().item())
                dust = float(plan[i, len(current)].item())
                denominator = max(real_total + dust, _EPS)
                for j in range(len(current)):
                    fraction = float(plan[i, j].item()) / denominator
                    scored.append((fraction, -float(cost[i, j].item()), track.track_id, j, i))
            used_tracks: set[int] = set()
            used_current: set[int] = set()
            for fraction, negative_cost, track_id, j, i in sorted(scored, reverse=True):
                if track_id in used_tracks or j in used_current:
                    continue
                if (
                    fraction < float(ot["minimum_real_transport_fraction"])
                    or -negative_cost > float(ot["maximum_match_cost"])
                ):
                    continue
                assignments[j] = i
                used_tracks.add(track_id)
                used_current.add(j)
        matched_ids = set()
        for j, component in enumerate(current):
            if j in assignments:
                track = candidates[assignments[j]]
                if track.missing_evaluation_steps:
                    row = {
                        "event": "reentry",
                        "track_id": track.track_id,
                        "phase": phase,
                        "gap": track.missing_evaluation_steps,
                    }
                    track.events.append(row)
                    events.append(row)
                track.observations.append(component)
                track.missing_evaluation_steps = 0
                matched_ids.add(track.track_id)
            else:
                track = TrackStateV6(next_id, [component])
                row = {"event": "birth", "track_id": next_id, "phase": phase}
                track.events.append(row)
                events.append(row)
                tracks.append(track)
                matched_ids.add(next_id)
                next_id += 1
        for track in tracks:
            if not track.alive or track.track_id in matched_ids:
                continue
            track.missing_evaluation_steps += 1
            if track.missing_evaluation_steps <= int(tracking["maximum_occlusion_gap"]):
                row = {"event": "occlusion", "track_id": track.track_id, "phase": phase}
            else:
                track.alive = False
                row = {"event": "death", "track_id": track.track_id, "phase": phase}
            track.events.append(row)
            events.append(row)
    final_phase = max(components_by_phase, default=-1) + 1
    for track in tracks:
        if track.alive:
            track.alive = False
            row = {"event": "death", "track_id": track.track_id, "phase": final_phase}
            track.events.append(row)
            events.append(row)
    return tuple(tracks), tuple(events)


def qualified_tracks_v6(
    tracks: Sequence[TrackStateV6],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[TrackStateV6, ...]:
    """Return tracks authorized to contribute to graph/control metrics."""

    spec = dict(prereg or registry.load_preregistration())
    tracking = spec["tracking"]
    minimum = int(tracking["minimum_track_observed_phases"])
    if (
        minimum != 3
        or tracking.get(
            "only_qualified_tracks_contribute_to_graph_or_control_metrics"
        )
        is not True
    ):
        raise AnonymousObjectObserverV6Error("qualified-track authority differs")
    return tuple(
        track
        for track in tracks
        if len({row.evaluation_phase for row in track.observations}) >= minimum
    )


def dynamic_edge_lifecycle_v6(
    tracks: Sequence[TrackStateV6],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[Mapping[str, Any], ...]:
    spec = dict(prereg or registry.load_preregistration())
    config = spec["dynamic_edges"]
    if config.get("qualified_tracks_only") is not True:
        raise AnonymousObjectObserverV6Error("dynamic-edge track authority differs")
    tracks = qualified_tracks_v6(tracks, prereg=spec)
    by_phase: dict[int, list[tuple[int, EvaluatedComponentV6]]] = {}
    for track in tracks:
        for observation in track.observations:
            by_phase.setdefault(observation.evaluation_phase, []).append(
                (track.track_id, observation)
            )
    active: set[tuple[int, int]] = set()
    rows: list[Mapping[str, Any]] = []
    for phase in sorted(by_phase):
        current: set[tuple[int, int]] = set()
        observations = sorted(by_phase[phase])
        for left_index in range(len(observations)):
            left_id, left = observations[left_index]
            for right_id, right in observations[left_index + 1 :]:
                pair = (min(left_id, right_id), max(left_id, right_id))
                distance = math.dist(left.centroid, right.centroid)
                affinity = math.exp(
                    -distance / float(config["soft_distance_temperature"])
                )
                if affinity >= float(config["activation_affinity"]):
                    current.add(pair)
                    event = "persist" if pair in active else "activate"
                    rows.append(
                        {
                            "event": event,
                            "phase": phase,
                            "anonymous_track_pair": list(pair),
                            "soft_affinity": affinity,
                            "physical_contact_truth_claimed": False,
                        }
                    )
        for pair in sorted(active - current):
            rows.append(
                {
                    "event": "deactivate",
                    "phase": phase,
                    "anonymous_track_pair": list(pair),
                }
            )
        active = current
    endpoint = max(by_phase, default=-1) + 1
    for pair in sorted(active):
        rows.append(
            {
                "event": "endpoint_death",
                "phase": endpoint,
                "anonymous_track_pair": list(pair),
            }
        )
    return tuple(rows)


def _nearest_phase(source: int, candidates: Sequence[int]) -> int:
    return min(candidates, key=lambda value: (abs(value - source), value))


def _mean_block_feature(
    tensors: Mapping[str, Mapping[int, torch.Tensor]],
    arm: str,
    blocks: Sequence[int],
) -> torch.Tensor:
    return torch.stack([tensors[arm][block] for block in blocks], dim=0).mean(dim=0)


def _support_iou(
    left: Mapping[int, Sequence[ProposalComponentV6]],
    right: Mapping[int, Sequence[ProposalComponentV6]],
) -> Optional[float]:
    rows = []
    for phase in sorted(set(left) & set(right)):
        a = set(item for row in left[phase] for item in row.support)
        b = set(item for row in right[phase] for item in row.support)
        if a or b:
            rows.append(len(a & b) / float(len(a | b)))
    return None if not rows else sum(rows) / len(rows)


def _track_displacement(track: TrackStateV6) -> Optional[tuple[float, float]]:
    if len(track.observations) < 2:
        return None
    first, last = track.observations[0], track.observations[-1]
    return (
        last.centroid[0] - first.centroid[0],
        last.centroid[1] - first.centroid[1],
    )


def _dominant_track(tracks: Sequence[TrackStateV6]) -> Optional[TrackStateV6]:
    if not tracks:
        return None
    return max(tracks, key=lambda row: (len(row.observations), -row.track_id))


def _direction_cosine(
    left: Optional[tuple[float, float]], right: Optional[tuple[float, float]]
) -> Optional[float]:
    if left is None or right is None:
        return None
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    if left_norm <= _EPS or right_norm <= _EPS:
        return None
    return (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)


def _path_acceleration(track: Optional[TrackStateV6]) -> Optional[float]:
    if track is None or len(track.observations) < 3:
        return None
    points = [item.centroid for item in track.observations]
    values = []
    for index in range(1, len(points) - 1):
        values.append(
            math.hypot(
                points[index + 1][0] - 2 * points[index][0] + points[index - 1][0],
                points[index + 1][1] - 2 * points[index][1] + points[index - 1][1],
            )
        )
    return sum(values) / len(values)


def _branch_direction(branch: str) -> tuple[str, str]:
    return ("A", "B") if branch == "A_to_B" else ("B", "A")


def _control_delta(
    features: Mapping[str, torch.Tensor], arm: str
) -> torch.Tensor:
    if arm == "noop":
        return torch.zeros_like(features["noop"])
    return (features[arm] - features["noop"]).detach()


def branch_gate_decision_v6(
    *,
    primary_track_count: int,
    primary_coverage: float,
    primary_lifecycle_count: int,
    component_counts: Mapping[str, int],
    static_ratio: Optional[float],
    reverse_cosine: Optional[float],
    phase_shuffle_pass: bool,
    paraphrase_iou: Optional[float],
    paraphrase_cosine: Optional[float],
    lexical_ratio: Optional[float],
    source_swap_iou: Optional[float],
    source_swap_coverage: float,
    source_swap_lifecycle_count: int,
    prereg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, bool]:
    """Exact preregistered AND gate; no score compensation is possible."""

    spec = dict(prereg or registry.load_preregistration())
    gate = spec["branchwise_diagnostic_gates"]
    values = {
        "primary_graph_valid": (
            primary_track_count >= int(gate["primary_minimum_track_count"])
            and primary_coverage >= float(gate["primary_minimum_track_coverage"])
            and primary_lifecycle_count
            >= int(gate["primary_minimum_dynamic_edge_lifecycle_events"])
        ),
        "noop_pass": component_counts.get("noop", -1)
        <= int(gate["noop_maximum_component_count"]),
        "static_pass": static_ratio is not None
        and static_ratio <= float(gate["static_to_primary_displacement_ratio_max"]),
        "reverse_pass": reverse_cosine is not None
        and reverse_cosine <= float(gate["reverse_endpoint_direction_cosine_max"]),
        "phase_shuffle_pass": phase_shuffle_pass is True,
        "paraphrase_pass": paraphrase_iou is not None
        and paraphrase_iou >= float(gate["paraphrase_support_iou_min"])
        and paraphrase_cosine is not None
        and paraphrase_cosine
        >= float(gate["paraphrase_endpoint_direction_cosine_min"]),
        "lexical_placebo_pass": lexical_ratio is not None
        and lexical_ratio
        <= float(gate["lexical_placebo_to_primary_component_ratio_max"]),
        "source_swap_pass": source_swap_iou is not None
        and source_swap_iou
        <= float(gate["source_swap_to_primary_support_iou_max"])
        and source_swap_coverage
        <= float(gate["source_swap_evaluated_track_coverage_max"])
        and source_swap_lifecycle_count
        <= int(gate["source_swap_dynamic_edge_lifecycle_max"]),
    }
    return values


def phase_shuffle_gate_v6(
    primary_acceleration: Optional[float],
    shuffle_acceleration: Optional[float],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, Optional[float]]:
    spec = dict(prereg or registry.load_preregistration())
    gate = spec["branchwise_diagnostic_gates"]
    floor = float(gate["phase_shuffle_absolute_acceleration_floor"])
    if primary_acceleration is None or shuffle_acceleration is None:
        return False, None
    if not math.isfinite(primary_acceleration) or not math.isfinite(shuffle_acceleration):
        return False, None
    if primary_acceleration <= _EPS:
        return shuffle_acceleration >= floor, None
    ratio = shuffle_acceleration / primary_acceleration
    return (
        shuffle_acceleration >= floor
        and ratio >= float(gate["phase_shuffle_to_primary_acceleration_ratio_min"]),
        ratio,
    )


def evaluate_crossfit_branch_v6(
    sketches: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    branch: str,
    height: int,
    width: int,
    prereg: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Evaluate one disjoint layer/time branch with all controls."""

    spec = dict(prereg or registry.load_preregistration())
    if branch not in BRANCHES:
        raise AnonymousObjectObserverV6Error("cross-fit branch differs")
    proposer_fold, evaluator_fold = _branch_direction(branch)
    proposer_blocks = LAYER_FOLDS[proposer_fold]
    evaluator_blocks = LAYER_FOLDS[evaluator_fold]
    phase_pairs = CROSS_FIT_PHASE_PAIRS[branch]
    proposal_phases = tuple(left for left, _right in phase_pairs)
    evaluation_phases = tuple(right for _left, right in phase_pairs)
    if set(proposer_blocks) & set(evaluator_blocks) or set(proposal_phases) & set(
        evaluation_phases
    ) or len(set(proposal_phases)) != 10 or len(set(evaluation_phases)) != 10:
        raise AnonymousObjectObserverV6Error("cross-fit folds overlap")

    proposer = {
        arm: _mean_block_feature(sketches, arm, proposer_blocks)
        for arm in registry.ARMS
    }
    evaluator_neutral = _mean_block_feature(
        sketches, "neutral", evaluator_blocks
    )
    proposer_neutral = proposer["neutral"]
    phase_permutation = tuple(int(item) for item in spec["controls"]["phase_shuffle"])
    if sorted(phase_permutation) != list(range(registry.PHASES)):
        raise AnonymousObjectObserverV6Error("phase shuffle is not a permutation")

    proposal_by_control: dict[str, dict[int, tuple[ProposalComponentV6, ...]]] = {}
    evaluation_by_control: dict[
        str, dict[int, tuple[EvaluatedComponentV6, ...]]
    ] = {}
    track_by_control: dict[str, tuple[TrackStateV6, ...]] = {}
    event_by_control: dict[str, tuple[Mapping[str, Any], ...]] = {}
    arms = (
        "action",
        "noop",
        "static",
        "reverse",
        "paraphrase",
        "lexical_placebo",
        "source_swap",
    )
    for arm in arms:
        delta = _control_delta(proposer, arm)
        proposal_rows: dict[int, tuple[ProposalComponentV6, ...]] = {}
        evaluated_rows: dict[int, tuple[EvaluatedComponentV6, ...]] = {}
        for phase in proposal_phases:
            components = discover_soft_components_v6(
                delta[phase],
                proposer_neutral[phase],
                phase=phase,
                height=height,
                width=width,
                prereg=spec,
            )
            proposal_rows[phase] = components
            target_phase = dict(phase_pairs)[phase]
            evaluated = []
            for component in components:
                row = evaluate_component_with_neutral_tokens_v6(
                    component,
                    evaluator_neutral[target_phase],
                    evaluation_phase=target_phase,
                    height=height,
                    width=width,
                    prereg=spec,
                )
                if row is not None:
                    evaluated.append(row)
            evaluated_rows[target_phase] = tuple(evaluated)
        tracks, events = track_hypotheses_v6(evaluated_rows, prereg=spec)
        proposal_by_control[arm] = proposal_rows
        evaluation_by_control[arm] = evaluated_rows
        track_by_control[arm] = tracks
        event_by_control[arm] = events

    # Execute the phase shuffle as its own proposer/evaluator path.
    primary_delta = _control_delta(proposer, "action")
    shuffled_delta = primary_delta[
        torch.tensor(phase_permutation, dtype=torch.long)
    ]
    shuffled_proposals: dict[int, tuple[ProposalComponentV6, ...]] = {}
    shuffled_evaluated: dict[int, tuple[EvaluatedComponentV6, ...]] = {}
    for phase in proposal_phases:
        components = discover_soft_components_v6(
            shuffled_delta[phase],
            proposer_neutral[phase],
            phase=phase,
            height=height,
            width=width,
            prereg=spec,
        )
        shuffled_proposals[phase] = components
        target_phase = dict(phase_pairs)[phase]
        rows = []
        for component in components:
            item = evaluate_component_with_neutral_tokens_v6(
                component,
                evaluator_neutral[target_phase],
                evaluation_phase=target_phase,
                height=height,
                width=width,
                prereg=spec,
            )
            if item is not None:
                rows.append(item)
        shuffled_evaluated[target_phase] = tuple(rows)
    shuffled_tracks, shuffled_events = track_hypotheses_v6(
        shuffled_evaluated, prereg=spec
    )
    proposal_by_control["phase_shuffle"] = shuffled_proposals
    evaluation_by_control["phase_shuffle"] = shuffled_evaluated
    track_by_control["phase_shuffle"] = shuffled_tracks
    event_by_control["phase_shuffle"] = shuffled_events

    qualified_track_by_control = {
        name: qualified_tracks_v6(rows, prereg=spec)
        for name, rows in track_by_control.items()
    }
    primary_tracks = qualified_track_by_control["action"]
    primary = _dominant_track(primary_tracks)
    primary_displacement = _track_displacement(primary) if primary is not None else None
    reverse = _dominant_track(qualified_track_by_control["reverse"])
    static = _dominant_track(qualified_track_by_control["static"])
    paraphrase = _dominant_track(qualified_track_by_control["paraphrase"])
    shuffled = _dominant_track(qualified_track_by_control["phase_shuffle"])
    primary_disp_norm = None if primary_displacement is None else math.hypot(*primary_displacement)
    static_displacement = _track_displacement(static) if static is not None else None
    static_disp_norm = 0.0 if static is None else (
        None if static_displacement is None else math.hypot(*static_displacement)
    )
    static_ratio = (
        None
        if primary_disp_norm is None or primary_disp_norm <= _EPS or static_disp_norm is None
        else static_disp_norm / primary_disp_norm
    )
    reverse_cosine = _direction_cosine(
        primary_displacement,
        _track_displacement(reverse) if reverse is not None else None,
    )
    paraphrase_cosine = _direction_cosine(
        primary_displacement,
        _track_displacement(paraphrase) if paraphrase is not None else None,
    )
    primary_acceleration = _path_acceleration(primary)
    shuffle_acceleration = _path_acceleration(shuffled)
    shuffle_pass, shuffle_ratio = phase_shuffle_gate_v6(
        primary_acceleration, shuffle_acceleration, prereg=spec
    )
    paraphrase_iou = _support_iou(
        proposal_by_control["action"], proposal_by_control["paraphrase"]
    )
    source_swap_iou = _support_iou(
        proposal_by_control["action"], proposal_by_control["source_swap"]
    )
    component_counts = {
        name: sum(len(rows) for rows in proposal_by_control[name].values())
        for name in proposal_by_control
    }
    primary_count = component_counts["action"]
    lexical_ratio = (
        None
        if primary_count <= 0
        else component_counts["lexical_placebo"] / float(primary_count)
    )
    coverage = (
        0.0
        if primary is None
        else len(primary.observations) / float(len(evaluation_phases))
    )
    edges = dynamic_edge_lifecycle_v6(primary_tracks, prereg=spec)
    lifecycle_count = sum(
        row["event"] in {"activate", "deactivate", "endpoint_death"}
        for row in edges
    )
    source_swap_track = _dominant_track(
        qualified_track_by_control["source_swap"]
    )
    source_swap_coverage = (
        0.0
        if source_swap_track is None
        else len(source_swap_track.observations) / float(len(evaluation_phases))
    )
    source_swap_edges = dynamic_edge_lifecycle_v6(
        qualified_track_by_control["source_swap"], prereg=spec
    )
    source_swap_lifecycle_count = sum(
        row["event"] in {"activate", "deactivate", "endpoint_death"}
        for row in source_swap_edges
    )
    gate_values = branch_gate_decision_v6(
        primary_track_count=len(primary_tracks),
        primary_coverage=coverage,
        primary_lifecycle_count=lifecycle_count,
        component_counts=component_counts,
        static_ratio=static_ratio,
        reverse_cosine=reverse_cosine,
        phase_shuffle_pass=shuffle_pass,
        paraphrase_iou=paraphrase_iou,
        paraphrase_cosine=paraphrase_cosine,
        lexical_ratio=lexical_ratio,
        source_swap_iou=source_swap_iou,
        source_swap_coverage=source_swap_coverage,
        source_swap_lifecycle_count=source_swap_lifecycle_count,
        prereg=spec,
    )
    branch_pass = all(gate_values.values())
    metrics = {
        "primary_track_coverage": coverage,
        "primary_dynamic_edge_lifecycle_count": lifecycle_count,
        "static_to_primary_displacement_ratio": static_ratio,
        "reverse_endpoint_direction_cosine": reverse_cosine,
        "phase_shuffle_to_primary_acceleration_ratio": shuffle_ratio,
        "paraphrase_support_iou": paraphrase_iou,
        "paraphrase_endpoint_direction_cosine": paraphrase_cosine,
        "lexical_placebo_to_primary_component_ratio": lexical_ratio,
        "source_swap_to_primary_support_iou": source_swap_iou,
        "source_swap_evaluated_track_coverage": source_swap_coverage,
        "source_swap_dynamic_edge_lifecycle_count": source_swap_lifecycle_count,
        "primary_path_acceleration": primary_acceleration,
        "phase_shuffle_path_acceleration": shuffle_acceleration,
    }
    value = {
        "branch": branch,
        "proposal_layer_fold": list(proposer_blocks),
        "evaluation_layer_fold": list(evaluator_blocks),
        "proposal_time_fold": list(proposal_phases),
        "evaluation_time_fold": list(evaluation_phases),
        "actual_phase_pairs": [list(pair) for pair in phase_pairs],
        "phase_pair_count": len(phase_pairs),
        "phase_pairs_one_to_one_without_overwrite": True,
        "layers_disjoint": True,
        "times_disjoint": True,
        "action_noop_residual_stop_gradient_proposal_only": True,
        "action_noop_residual_used_as_descriptor": False,
        "action_noop_residual_used_as_reward": False,
        "correspondence_descriptor": "prompt_neutral_visual_query_hidden_sketch",
        "component_counts": component_counts,
        "track_counts": {name: len(rows) for name, rows in track_by_control.items()},
        "qualified_track_counts": {
            name: len(rows) for name, rows in qualified_track_by_control.items()
        },
        "minimum_track_observed_phases": int(
            spec["tracking"]["minimum_track_observed_phases"]
        ),
        "control_executed": {name: True for name in CONTROL_ARMS},
        "metrics": metrics,
        "gates": gate_values,
        "graph_abstained": not gate_values["primary_graph_valid"],
        "R0_compensation_applied": False,
        "branch_pass": branch_pass,
        "primary_tracks": [
            {
                "track_id": track.track_id,
                "observed_phases": [
                    item.evaluation_phase for item in track.observations
                ],
                "event_kinds": [row["event"] for row in track.events],
                "semantic_role": None,
            }
            for track in primary_tracks
        ],
        "primary_dynamic_edge_lifecycle": list(edges),
        "proposal_receipt_digest": _digest(
            {
                name: {
                    str(phase): [row.public_row() for row in rows]
                    for phase, rows in sorted(phase_rows.items())
                }
                for name, phase_rows in sorted(proposal_by_control.items())
            }
        ),
    }
    return {**value, "digest": _digest(value)}


@dataclass(frozen=True)
class ReducedAnonymousCellV6:
    appearance_id: str
    sigma_band: str
    state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    branch_receipts: Mapping[str, Mapping[str, Any]]
    branchwise_diagnostic_admitted: bool
    raw_and_projected_inputs_zeroized: bool

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_id": self.appearance_id,
            "sigma_band": self.sigma_band,
            "state_sha256": self.state_sha256,
            "timestep_sha256": self.timestep_sha256,
            "rotary_sha256": self.rotary_sha256,
            "same_nontext_identity": True,
            "branch_receipts": {
                name: dict(self.branch_receipts[name]) for name in BRANCHES
            },
            "branchwise_formula": "A_to_B AND B_to_A",
            "branchwise_diagnostic_admitted": self.branchwise_diagnostic_admitted,
            "graph_abstention_cannot_be_compensated_by_R0": True,
            "raw_and_projected_inputs_zeroized": self.raw_and_projected_inputs_zeroized,
            "representation_admitted": False,
            "stable_transferable_action_representation_claimed": False,
        }
        return {**value, "digest": _digest(value)}


def reduce_anonymous_cell_v6(
    captures: Mapping[str, Mapping[int, AnonymousProjectedArmV6]],
    *,
    prereg: Optional[Mapping[str, Any]] = None,
) -> ReducedAnonymousCellV6:
    spec = dict(prereg or registry.load_preregistration())
    if set(captures) != set(registry.ARMS) or any(
        set(rows) != set(registry.BLOCKS) for rows in captures.values()
    ):
        raise AnonymousObjectObserverV6Error("V6 cell capture matrix differs")
    owned = [captures[arm][block] for arm in registry.ARMS for block in registry.BLOCKS]
    cpu: list[torch.Tensor] = []
    try:
        for row in owned:
            row.validate()
        for field_name in (
            "appearance_id",
            "sigma_band",
            "state_sha256",
            "timestep_sha256",
            "rotary_sha256",
            "projection_digest",
        ):
            if len({getattr(row, field_name) for row in owned}) != 1:
                raise AnonymousObjectObserverV6Error(
                    f"V6 cell {field_name} differs"
                )
        sketches: dict[str, dict[int, torch.Tensor]] = {}
        for arm in registry.ARMS:
            sketches[arm] = {}
            for block in registry.BLOCKS:
                row = captures[arm][block]
                query = row.query_sketch.detach().to(device="cpu", dtype=torch.float32)[0].contiguous()
                cpu.append(query)
                hidden = row.hidden_sketch.detach().to(device="cpu", dtype=torch.float32)[0].contiguous()
                cpu.append(hidden)
                combined = torch.cat((query, hidden), dim=-1).contiguous()
                cpu.append(combined)
                sketches[arm][block] = combined
        branch_receipts = {
            branch: evaluate_crossfit_branch_v6(
                sketches,
                branch=branch,
                height=registry.PATCH_HEIGHT,
                width=registry.PATCH_WIDTH,
                prereg=spec,
            )
            for branch in BRANCHES
        }
        admitted = all(branch_receipts[name]["branch_pass"] for name in BRANCHES)
        result = ReducedAnonymousCellV6(
            owned[0].appearance_id,
            owned[0].sigma_band,
            owned[0].state_sha256,
            owned[0].timestep_sha256,
            owned[0].rotary_sha256,
            branch_receipts,
            admitted,
            True,
        )
        return result
    finally:
        for row in owned:
            row.zeroize()
        _zeroize(cpu)


class AnonymousObjectObserverV6:
    """Streaming 3 appearances × 3 sigma-cell accumulator."""

    def __init__(self, prereg: Optional[Mapping[str, Any]] = None) -> None:
        self.prereg = dict(prereg or registry.load_preregistration())
        self._cells: dict[tuple[str, str], ReducedAnonymousCellV6] = {}
        self._closed = False

    def add(self, cell: ReducedAnonymousCellV6) -> None:
        if self._closed or not isinstance(cell, ReducedAnonymousCellV6):
            raise AnonymousObjectObserverV6Error("V6 stream is closed or cell differs")
        key = (cell.appearance_id, cell.sigma_band)
        if key in self._cells:
            raise AnonymousObjectObserverV6Error("duplicate V6 cell")
        self._cells[key] = cell

    def abort(self) -> None:
        self._cells.clear()
        self._closed = True

    def finalize(self) -> Mapping[str, Any]:
        if self._closed:
            raise AnonymousObjectObserverV6Error("V6 stream is closed")
        expected = {
            (appearance, sigma)
            for appearance in registry.APPEARANCE_IDS
            for sigma in registry.SIGMA_CELL_INDICES
        }
        if set(self._cells) != expected:
            raise AnonymousObjectObserverV6Error("V6 capture matrix is incomplete")
        self._closed = True
        rows = [self._cells[key].receipt() for key in sorted(self._cells)]
        diagnostic_count = sum(row["branchwise_diagnostic_admitted"] for row in rows)
        all_controls = all(
            all(
                all(branch["control_executed"].values())
                for branch in row["branch_receipts"].values()
            )
            for row in rows
        )
        value = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "cell_count": len(rows),
            "cells": rows,
            "branchwise_diagnostic_admitted_cell_count": diagnostic_count,
            "diagnostic_component_admitted": diagnostic_count == len(expected),
            "diagnostic_component_status": (
                "ADMITTED_9_OF_9" if diagnostic_count == len(expected) else "REJECTED"
            ),
            "overall_aggregation_formula": "all 9/9 cells pass A_to_B AND B_to_A",
            "cell_selection_or_compensation_permitted": False,
            "all_controls_executed": all_controls,
            "anonymous_variable_cardinality_slots": True,
            "slot_lifecycle_events": ["birth", "occlusion", "reentry", "death"],
            "dynamic_edge_lifecycle_events": [
                "activate",
                "persist",
                "deactivate",
                "endpoint_death",
            ],
            "caption_role_token_localization_used": False,
            "fixed_semantic_role_inventory_used": False,
            "action_noop_residual_proposal_only": True,
            "prompt_neutral_visual_correspondence_used": True,
            "crossfit_branches": list(BRANCHES),
            "crossfit_branchwise_AND": True,
            "graph_abstention_R0_compensation_permitted": False,
            "prompt_shuffle_control_executed": False,
            "heldout_transfer_control_executed": False,
            "representation_admitted": False,
            "stable_transferable_action_representation_claimed": False,
            "scientific_claim_authorized": False,
        }
        return {**value, "digest": _digest(value)}


__all__ = [
    "AnonymousObjectObserverV6",
    "AnonymousObjectObserverV6Error",
    "AnonymousProjectedArmV6",
    "BRANCHES",
    "CONTROL_ARMS",
    "EvaluatedComponentV6",
    "LAYER_FOLDS",
    "METHOD",
    "ProposalComponentV6",
    "ReducedAnonymousCellV6",
    "SCHEMA_VERSION",
    "TIME_FOLDS",
    "TrackStateV6",
    "discover_soft_components_v6",
    "branch_gate_decision_v6",
    "dynamic_edge_lifecycle_v6",
    "evaluate_component_with_neutral_tokens_v6",
    "evaluate_crossfit_branch_v6",
    "phase_shuffle_gate_v6",
    "qualified_tracks_v6",
    "reduce_anonymous_cell_v6",
    "track_hypotheses_v6",
    "unbalanced_ot_with_dustbin_v6",
]

#!/usr/bin/env python3
"""Streaming support-frame object graph for branch-specific T2V trajectories.

The observer turns the derived cross-attention role proxy at four frozen
transformer blocks into four object slots.  ``start_support -> end_support`` is
used only as a reference frame: translation, image-plane rotation and scale are
removed before any feature is retained.  Three dynamic edge hypotheses are
then represented by signed relative geometry and phase-varying soft edge
weights.  The reference edge itself never supplies action reward.

Unlike the v1 observer, action/noop/reverse/static are required to have their
own noisy states.  They need only share an authenticated initial Gaussian and
scheduler lineage, which the AUH runner seals separately.  Raw Q/K, dense role
proxies and image coordinates are zeroized in ``add``; only the compact
support-relative features survive until ``finalize``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:  # Package import in tests; direct module import in AUH create-only runners.
    from . import self_generated_relational_action_graph_observer_v1 as legacy
    from . import self_generated_relational_t2v_probe_registry_v3 as registry
except ImportError:  # pragma: no cover - exercised by the remote script entrypoint.
    import self_generated_relational_action_graph_observer_v1 as legacy
    import self_generated_relational_t2v_probe_registry_v3 as registry


METHOD = "self-generated-branch-interaction-graph-observer-v3"
SCHEMA_VERSION = "bernini-self-generated-branch-interaction-graph-v3"
PHASES = legacy.PHASES
ARMS = registry.ARMS
SIGMA_BANDS = legacy.SIGMA_BANDS
BLOCKS = legacy.BLOCKS
FEATURE_NAMES = registry.PUBLIC_FEATURES
CORE_ROLES = ("agent", "moving_object", "start_support", "end_support")
LIFECYCLE_BINS = ((0, 7), (7, 14), (14, 21))
SOFT_EDGE_TEMPERATURE = 0.25
_EPS = 1.0e-8


class BranchInteractionGraphError(ValueError):
    """A fail-closed branch trajectory or object graph invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise BranchInteractionGraphError(
            "value is not canonical finite JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    owned = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = canonical_json_bytes(
        {"shape": list(map(int, owned.shape)), "dtype": "float32"}
    )
    return hashlib.sha256(header + owned.numpy().tobytes(order="C")).hexdigest()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator <= _EPS:
        return 0.0
    return float(torch.dot(a, b).item() / denominator)


def _normalized_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = max(
        float(torch.linalg.vector_norm(a).item()),
        float(torch.linalg.vector_norm(b).item()),
        _EPS,
    )
    return float(torch.linalg.vector_norm(a - b).item()) / denominator


def _coordinates(height: int, width: int) -> torch.Tensor:
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)


@dataclass(frozen=True)
class _ReducedCell:
    key: tuple[str, str, str, int]
    state_sha256: str
    prompt_sha256: str
    values: torch.Tensor
    valid: torch.Tensor
    role_confidence: torch.Tensor
    support_scale: torch.Tensor
    dense_temporaries_zeroized: bool
    digest: str


def _reduce_cell(
    cell: legacy.CaptureCell,
    roles: Sequence[legacy.RoleSpec],
) -> _ReducedCell:
    cell.validate(roles)
    role_ids = tuple(item.role_id for item in roles)
    if role_ids != registry.ROLE_IDS:
        raise BranchInteractionGraphError("canonical role order differs")
    raw = cell.responsibilities.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    mass = raw.sum(dim=-1)
    normalized = raw / mass.clamp_min(_EPS).unsqueeze(-1)
    coords = _coordinates(cell.patch_height, cell.patch_width)
    centroids = torch.einsum("trp,pd->trd", normalized, coords)
    entropy = -(
        normalized.clamp_min(_EPS) * normalized.clamp_min(_EPS).log()
    ).sum(dim=-1) / max(math.log(float(coords.shape[0])), 1.0)
    confidence = (1.0 - entropy).clamp(0.0, 1.0)

    index = {name: role_ids.index(name) for name in CORE_ROLES}
    agent = centroids[:, index["agent"]]
    moving = centroids[:, index["moving_object"]]
    start = centroids[:, index["start_support"]]
    end = centroids[:, index["end_support"]]
    axis = end - start
    scale = torch.linalg.vector_norm(axis, dim=-1)
    safe_scale = scale.clamp_min(_EPS)
    unit_x = axis / safe_scale.unsqueeze(-1)
    unit_y = torch.stack((-unit_x[:, 1], unit_x[:, 0]), dim=-1)

    def canonical_position(point: torch.Tensor) -> torch.Tensor:
        relative = point - start
        return torch.stack(
            (
                (relative * unit_x).sum(dim=-1) / safe_scale,
                (relative * unit_y).sum(dim=-1) / safe_scale,
            ),
            dim=-1,
        )

    object_xy = canonical_position(moving)
    agent_xy = canonical_position(agent)
    agent_object = torch.linalg.vector_norm(agent_xy - object_xy, dim=-1)
    object_start = torch.linalg.vector_norm(object_xy, dim=-1)
    object_end = torch.linalg.vector_norm(
        object_xy - torch.tensor((1.0, 0.0), dtype=torch.float32), dim=-1
    )
    distances = torch.stack((agent_object, object_start, object_end), dim=-1)
    soft_edges = torch.softmax(-distances / SOFT_EDGE_TEMPERATURE, dim=-1)
    values = torch.cat(
        (
            object_xy,
            distances,
            soft_edges,
        ),
        dim=-1,
    ).contiguous()
    if tuple(values.shape) != (PHASES, len(FEATURE_NAMES)):
        raise BranchInteractionGraphError("canonical feature geometry differs")
    core_indices = torch.tensor([index[name] for name in CORE_ROLES], dtype=torch.long)
    core_mass_valid = (
        mass[:, core_indices]
        >= float(registry.ADMISSION_THRESHOLDS["role_mass_min"])
    ).all(dim=-1)
    valid = (core_mass_valid & (scale >= float(registry.ADMISSION_THRESHOLDS[
        "support_frame_scale_min"
    ]))).contiguous()
    core_confidence = confidence[:, core_indices].contiguous()
    if not bool(torch.isfinite(values).all().item()):
        raise BranchInteractionGraphError("canonical features are non-finite")
    retained_scale = scale.clone().contiguous()
    payload = {
        "key": [cell.appearance_id, cell.arm, cell.sigma_band, cell.block_index],
        "state_sha256": cell.state_sha256,
        "prompt_sha256": cell.prompt_sha256,
        "values_sha256": _tensor_digest(values),
        "valid_sha256": hashlib.sha256(
            valid.to(torch.uint8).numpy().tobytes(order="C")
        ).hexdigest(),
        "role_confidence_sha256": _tensor_digest(core_confidence),
        "support_scale_sha256": _tensor_digest(scale),
        "support_frame": ["start_support", "end_support"],
        "feature_names": list(FEATURE_NAMES),
        "dense_reduction_temporaries_zeroized_before_return": True,
    }
    reduced = _ReducedCell(
        key=(cell.appearance_id, cell.arm, cell.sigma_band, cell.block_index),
        state_sha256=cell.state_sha256,
        prompt_sha256=cell.prompt_sha256,
        values=values,
        valid=valid,
        role_confidence=core_confidence,
        support_scale=retained_scale,
        dense_temporaries_zeroized=True,
        digest=object_sha256(payload),
    )
    # ``raw`` is a CPU clone of the dense derived role proxy.  Scrubbing only
    # the caller-owned GPU tensor would leave this reduction copy resident.
    # Compact support-relative values/confidence/validity are distinct owned
    # tensors; every dense or absolute temporary is cleared before returning.
    dense_temporaries = (
        raw,
        normalized,
        coords,
        centroids,
        mass,
        entropy,
        confidence,
        axis,
        scale,
        safe_scale,
        unit_x,
        unit_y,
        object_xy,
        agent_xy,
        distances,
        soft_edges,
    )
    with torch.inference_mode():
        for tensor in dense_temporaries:
            tensor.zero_()
    if any(int(torch.count_nonzero(tensor).item()) != 0 for tensor in dense_temporaries):
        raise BranchInteractionGraphError("dense reduction temporary did not zeroize")
    return reduced


def _lifecycle_signature(value: torch.Tensor, *, reverse: bool = False) -> torch.Tensor:
    if tuple(value.shape) != (PHASES, len(FEATURE_NAMES)):
        raise BranchInteractionGraphError("lifecycle feature geometry differs")
    aligned = torch.flip(value, dims=(0,)) if reverse else value
    signed = aligned - aligned[0:1]
    pooled = torch.stack(
        [signed[first:last].mean(dim=0) for first, last in LIFECYCLE_BINS],
        dim=0,
    )
    return pooled.contiguous()


def _transition_energy(value: torch.Tensor) -> float:
    # All retained channels are support-frame relational quantities.  Keeping
    # their signs is essential; no abs/radial shortcut may erase direction.
    return float(torch.linalg.vector_norm(torch.diff(value.double(), dim=0)).item())


def _phase_rows(value: torch.Tensor) -> list[Mapping[str, Any]]:
    return [
        {
            "phase_index": phase,
            "support_relative_features": {
                name: float(value[phase, feature].item())
                for feature, name in enumerate(FEATURE_NAMES)
            },
        }
        for phase in range(PHASES)
    ]


class StreamingBranchInteractionGraphObserver:
    """One-use reducer for exact3 x branch4 x sigma3 x block4 captures."""

    def __init__(self, *, roles: Sequence[legacy.RoleSpec]) -> None:
        self.roles = tuple(roles)
        if tuple(item.role_id for item in self.roles) != registry.ROLE_IDS:
            raise BranchInteractionGraphError("canonical v3 role registry differs")
        if any(
            item.ownership != "self_generated_anchor_owned" for item in self.roles
        ):
            raise BranchInteractionGraphError("v3 roles must be anchor-owned")
        self._cells: dict[tuple[str, str, str, int], _ReducedCell] = {}
        self._raw_zeroized = 0
        self._finalized = False

    def add(self, cell: legacy.CaptureCell, *, zeroize: bool = True) -> None:
        if self._finalized:
            raise BranchInteractionGraphError("observer is already finalized")
        if not isinstance(cell, legacy.CaptureCell):
            raise BranchInteractionGraphError("observer received a non-CaptureCell")
        if zeroize is not True:
            raise BranchInteractionGraphError("raw capture zeroization is mandatory")
        tensors = (cell.queries, cell.keys, cell.responsibilities)
        try:
            reduced = _reduce_cell(cell, self.roles)
            if reduced.key in self._cells:
                raise BranchInteractionGraphError("duplicate branch capture cell")
            self._cells[reduced.key] = reduced
        finally:
            with torch.inference_mode():
                for tensor in tensors:
                    if isinstance(tensor, torch.Tensor) and tensor.device.type != "meta":
                        tensor.zero_()
            if all(
                isinstance(tensor, torch.Tensor)
                and int(torch.count_nonzero(tensor).item()) == 0
                for tensor in tensors
            ):
                self._raw_zeroized += 1

    def finalize(self) -> Mapping[str, Any]:
        if self._finalized:
            raise BranchInteractionGraphError("observer finalize is one-use")
        self._finalized = True
        expected = (
            len(registry.APPEARANCE_IDS) * len(ARMS) * len(SIGMA_BANDS) * len(BLOCKS)
        )
        if len(self._cells) != expected or self._raw_zeroized != expected:
            raise BranchInteractionGraphError(
                f"observer requires exactly {expected} reduced/zeroized captures"
            )
        expected_keys = {
            (appearance, arm, sigma, block)
            for appearance in registry.APPEARANCE_IDS
            for arm in ARMS
            for sigma in SIGMA_BANDS
            for block in BLOCKS
        }
        if set(self._cells) != expected_keys:
            raise BranchInteractionGraphError("branch capture matrix is incomplete")

        state_lineage_rows = []
        for appearance in registry.APPEARANCE_IDS:
            for sigma in SIGMA_BANDS:
                arm_states = {}
                for arm in ARMS:
                    states = {
                        self._cells[(appearance, arm, sigma, block)].state_sha256
                        for block in BLOCKS
                    }
                    prompts = {
                        self._cells[(appearance, arm, sigma, block)].prompt_sha256
                        for block in BLOCKS
                    }
                    if len(states) != 1 or len(prompts) != 1:
                        raise BranchInteractionGraphError(
                            "one branch/cell differs across observed blocks"
                        )
                    arm_states[arm] = next(iter(states))
                if len(set(arm_states.values())) != len(ARMS):
                    raise BranchInteractionGraphError(
                        "branch-specific states did not diverge across four arms"
                    )
                state_lineage_rows.append(
                    {
                        "appearance_id": appearance,
                        "sigma_band": sigma,
                        "arm_state_sha256": arm_states,
                        "four_branch_states_distinct": True,
                    }
                )

        aggregate: dict[str, dict[str, torch.Tensor]] = {}
        aggregation_rows = []
        thresholds = dict(registry.ADMISSION_THRESHOLDS)
        all_cells_observed = True
        all_localization_pass = True
        for appearance in registry.APPEARANCE_IDS:
            aggregate[appearance] = {}
            for arm in ARMS:
                cells = [
                    self._cells[(appearance, arm, sigma, block)]
                    for sigma in SIGMA_BANDS
                    for block in BLOCKS
                ]
                values = torch.stack([item.values for item in cells], dim=0)
                valid = torch.stack([item.valid for item in cells], dim=0)
                confidence_by_role = torch.stack(
                    [item.role_confidence for item in cells], dim=0
                )
                confidence = confidence_by_role.min(dim=-1).values
                weights = torch.where(valid, confidence.clamp_min(_EPS), torch.zeros_like(confidence))
                denominator = weights.sum(dim=0)
                safe_denominator = denominator.clamp_min(_EPS)
                combined = (
                    (values * weights.unsqueeze(-1)).sum(dim=0)
                    / safe_denominator.unsqueeze(-1)
                ).contiguous()
                observed = bool(valid.all().item()) and bool((denominator > _EPS).all().item())
                median_localization = float(confidence.median().item())
                localization_pass = (
                    median_localization
                    >= float(thresholds["role_localization_confidence_min"])
                )
                all_cells_observed = all_cells_observed and observed
                all_localization_pass = all_localization_pass and localization_pass
                aggregate[appearance][arm] = combined
                aggregation_rows.append(
                    {
                        "appearance_id": appearance,
                        "arm": arm,
                        "cell_count": len(cells),
                        "all_cells_observed": observed,
                        "median_min_role_localization_confidence": median_localization,
                        "role_localization_gate_passed": localization_pass,
                        "confidence_weighted_multi_block_multi_sigma": True,
                        "phase_rows": _phase_rows(combined),
                        "aggregate_sha256": _tensor_digest(combined),
                    }
                )

        appearance_rows = []
        all_control_gates = True
        action_signatures: dict[str, torch.Tensor] = {}
        for appearance in registry.APPEARANCE_IDS:
            action = aggregate[appearance]["action"]
            noop = aggregate[appearance]["noop"]
            reverse = aggregate[appearance]["reverse"]
            static = aggregate[appearance]["static"]
            action_signature = _lifecycle_signature(action)
            reverse_signature = _lifecycle_signature(reverse, reverse=True)
            action_signatures[appearance] = action_signature

            action_progress = float((action[-1, 0] - action[0, 0]).item())
            reverse_progress = float((reverse[-1, 0] - reverse[0, 0]).item())
            action_energy = _transition_energy(action)
            noop_energy = _transition_energy(noop)
            static_energy = _transition_energy(static)
            null_energy = max(noop_energy, static_energy, _EPS)
            reverse_cosine = _cosine(action_signature, reverse_signature)
            reverse_distance = _normalized_distance(action_signature, reverse_signature)
            action_start_recedes = float((action[-1, 3] - action[0, 3]).item())
            action_end_approaches = float((action[0, 4] - action[-1, 4]).item())
            action_start_edge_drops = float((action[0, 6] - action[-1, 6]).item())
            action_end_edge_rises = float((action[-1, 7] - action[0, 7]).item())
            reverse_end_recedes = float((reverse[-1, 4] - reverse[0, 4]).item())
            reverse_start_approaches = float((reverse[0, 3] - reverse[-1, 3]).item())
            reverse_end_edge_drops = float((reverse[0, 7] - reverse[-1, 7]).item())
            reverse_start_edge_rises = float((reverse[-1, 6] - reverse[0, 6]).item())
            reverse_start_endpoint_rms = float(
                torch.sqrt(torch.mean((reverse[0].double() - action[-1].double()).square())).item()
            )
            reverse_end_endpoint_rms = float(
                torch.sqrt(torch.mean((reverse[-1].double() - action[0].double()).square())).item()
            )
            reverse_endpoint_topology_rms = max(
                reverse_start_endpoint_rms,
                reverse_end_endpoint_rms,
            )
            reverse_endpoint_topology_max_abs = max(
                float(torch.max(torch.abs(reverse[0].double() - action[-1].double())).item()),
                float(torch.max(torch.abs(reverse[-1].double() - action[0].double())).item()),
            )

            cell_progress = []
            for sigma in SIGMA_BANDS:
                for block in BLOCKS:
                    row = self._cells[(appearance, "action", sigma, block)].values
                    cell_progress.append(float((row[-1, 0] - row[0, 0]).item()))
            positive_fraction = sum(
                item
                > float(
                    thresholds["per_cell_forward_progress_strictly_greater_than"]
                )
                for item in cell_progress
            ) / len(cell_progress)
            gates = {
                "role_localization": all(
                    item["role_localization_gate_passed"]
                    for item in aggregation_rows
                    if item["appearance_id"] == appearance
                ),
                "all_cells_observed": all(
                    item["all_cells_observed"]
                    for item in aggregation_rows
                    if item["appearance_id"] == appearance
                ),
                "forward_progress_nonzero": action_progress
                >= float(thresholds["forward_progress_min"]),
                "reverse_progress_nonzero": reverse_progress
                <= float(thresholds["reverse_progress_max"]),
                "dynamic_exceeds_null": action_energy / null_energy
                >= float(thresholds["dynamic_over_null_ratio_min"]),
                "noop_lacks_transition": noop_energy / max(action_energy, _EPS)
                <= float(thresholds["null_transition_ratio_max"]),
                "static_lacks_transition": static_energy / max(action_energy, _EPS)
                <= float(thresholds["null_transition_ratio_max"]),
                "reverse_cycle_cosine": reverse_cosine
                >= float(thresholds["reverse_cycle_cosine_min"]),
                "reverse_cycle_distance": reverse_distance
                <= float(thresholds["reverse_cycle_distance_max"]),
                "start_support_signed_recession": action_start_recedes
                >= float(thresholds["support_signed_distance_change_min"]),
                "end_support_signed_approach": action_end_approaches
                >= float(thresholds["support_signed_distance_change_min"]),
                "soft_graph_start_edge_deactivates": action_start_edge_drops
                >= float(thresholds["soft_edge_switch_min"]),
                "soft_graph_end_edge_activates": action_end_edge_rises
                >= float(thresholds["soft_edge_switch_min"]),
                "reverse_end_support_signed_recession": reverse_end_recedes
                >= float(thresholds["support_signed_distance_change_min"]),
                "reverse_start_support_signed_approach": reverse_start_approaches
                >= float(thresholds["support_signed_distance_change_min"]),
                "reverse_soft_end_edge_deactivates": reverse_end_edge_drops
                >= float(thresholds["soft_edge_switch_min"]),
                "reverse_soft_start_edge_activates": reverse_start_edge_rises
                >= float(thresholds["soft_edge_switch_min"]),
                "reverse_endpoint_topology_closes": reverse_endpoint_topology_rms
                <= float(thresholds["reverse_endpoint_topology_rms_max"]),
                "reverse_endpoint_each_feature_closes": reverse_endpoint_topology_max_abs
                <= float(thresholds["reverse_endpoint_topology_max_abs_max"]),
                "per_cell_forward_direction_consensus": positive_fraction
                >= float(thresholds["positive_progress_cell_fraction_min"]),
            }
            passed = all(gates.values())
            all_control_gates = all_control_gates and passed
            appearance_rows.append(
                {
                    "appearance_id": appearance,
                    "metrics": {
                        "forward_signed_progress": action_progress,
                        "reverse_signed_progress": reverse_progress,
                        "action_transition_energy": action_energy,
                        "noop_transition_energy": noop_energy,
                        "static_transition_energy": static_energy,
                        "dynamic_over_max_null_ratio": action_energy / null_energy,
                        "noop_transition_ratio": noop_energy / max(action_energy, _EPS),
                        "static_transition_ratio": static_energy / max(action_energy, _EPS),
                        "reverse_cycle_cosine": reverse_cosine,
                        "reverse_cycle_normalized_distance": reverse_distance,
                        "start_support_signed_recession": action_start_recedes,
                        "end_support_signed_approach": action_end_approaches,
                        "soft_graph_start_edge_drop": action_start_edge_drops,
                        "soft_graph_end_edge_rise": action_end_edge_rises,
                        "reverse_end_support_signed_recession": reverse_end_recedes,
                        "reverse_start_support_signed_approach": reverse_start_approaches,
                        "reverse_soft_end_edge_drop": reverse_end_edge_drops,
                        "reverse_soft_start_edge_rise": reverse_start_edge_rises,
                        "reverse_start_vs_action_end_topology_rms": reverse_start_endpoint_rms,
                        "reverse_end_vs_action_start_topology_rms": reverse_end_endpoint_rms,
                        "reverse_endpoint_topology_rms": reverse_endpoint_topology_rms,
                        "reverse_endpoint_topology_max_abs": reverse_endpoint_topology_max_abs,
                        "positive_progress_cell_fraction": positive_fraction,
                    },
                    "control_gates": gates,
                    "controls_passed": passed,
                    "action_lifecycle_signature": action_signature.tolist(),
                    "reverse_retimed_lifecycle_signature": reverse_signature.tolist(),
                }
            )

        consensus_rows = []
        consensus_pass = True
        for left_index, left in enumerate(registry.APPEARANCE_IDS):
            for right in registry.APPEARANCE_IDS[left_index + 1 :]:
                cosine = _cosine(action_signatures[left], action_signatures[right])
                distance = _normalized_distance(
                    action_signatures[left], action_signatures[right]
                )
                passed = (
                    cosine >= float(thresholds["appearance_cosine_min"])
                    and distance <= float(thresholds["appearance_distance_max"])
                )
                consensus_pass = consensus_pass and passed
                consensus_rows.append(
                    {
                        "left": left,
                        "right": right,
                        "cosine": cosine,
                        "normalized_distance": distance,
                        "passed": passed,
                    }
                )

        admitted = (
            all_control_gates
            and consensus_pass
            and all_cells_observed
            and all_localization_pass
        )
        cell_digest_rows = [
            {
                "appearance_id": key[0],
                "arm": key[1],
                "sigma_band": key[2],
                "block_index": key[3],
                "reduced_digest": cell.digest,
            }
            for key, cell in sorted(self._cells.items())
        ]
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "status": "MECHANICALLY_ADMITTED" if admitted else "REJECTED",
            "capture_matrix": {
                "appearance_ids": list(registry.APPEARANCE_IDS),
                "arms": list(ARMS),
                "sigma_bands": list(SIGMA_BANDS),
                "blocks": list(BLOCKS),
                "phase_count": PHASES,
                "capture_count": len(self._cells),
                "raw_zeroized_capture_count": self._raw_zeroized,
                "same_state_within_appearance_sigma": False,
                "branch_specific_state_within_appearance_arm_sigma": True,
            },
            "state_lineage": state_lineage_rows,
            "node_registry": [item.receipt() for item in self.roles],
            "reference_frame": {
                "source_role": "start_support",
                "target_role": "end_support",
                "translation_removed": True,
                "normalized_unit_square_rotation_removed_above_cutoff": True,
                "physical_pixel_rotation_invariance_claimed": False,
                "representation_scale_removed_above_degeneracy_cutoff": True,
                "whole_admission_receipt_scale_invariance_claimed": False,
                "reference_edge_contributes_to_action_reward": False,
                "absolute_frame_persisted": False,
            },
            "interaction_graph": {
                "candidate_edges": [
                    ["agent", "moving_object"],
                    ["moving_object", "start_support"],
                    ["moving_object", "end_support"],
                ],
                "phase_varying_soft_edges": True,
                "soft_edge_temperature": SOFT_EDGE_TEMPERATURE,
                "default_cartesian_product_used": False,
                "typed_relation_truth_claimed": False,
                "physical_contact_truth_claimed": False,
            },
            "feature_names": list(FEATURE_NAMES),
            "lifecycle_bins": [list(item) for item in LIFECYCLE_BINS],
            "admission_thresholds": thresholds,
            "aggregated_branch_features": aggregation_rows,
            "appearance_control_results": appearance_rows,
            "multiappearance_consensus": consensus_rows,
            "cell_reduction_receipts": cell_digest_rows,
            "dense_reduction_temporaries_zeroized_count": sum(
                int(item.dense_temporaries_zeroized) for item in self._cells.values()
            ),
            "summary": {
                "all_control_gates_passed": all_control_gates,
                "all_appearance_consensus_gates_passed": consensus_pass,
                "all_cells_observed": all_cells_observed,
                "all_role_localization_gates_passed": all_localization_pass,
                "mechanical_admission_passed": admitted,
            },
            "published_representation": {
                "support_relative_object_features_only": True,
                "signed_direction_preserved": True,
                "appearance_descriptors": False,
                "raw_q": False,
                "raw_k": False,
                "dense_role_responsibilities": False,
                "absolute_coordinates": False,
                "absolute_support_frame": False,
            },
            "base_frozen_required": True,
            "frozen_base_can_supply_graph_success": False,
            "optimizer_updates": 0,
            "renderer_output_modified": False,
            "target_inputs_consumed": False,
            "teacher_video_decoded": False,
            "generator_injection_authorized": False,
            "scientific_claim_authorized": False,
            "stable_transferable_action_representation_claimed": False,
        }

        internal_zeroized = 0
        with torch.inference_mode():
            for cell in self._cells.values():
                cell.values.zero_()
                cell.valid.zero_()
                cell.role_confidence.zero_()
                cell.support_scale.zero_()
                if (
                    int(torch.count_nonzero(cell.values).item()) == 0
                    and int(torch.count_nonzero(cell.valid).item()) == 0
                    and int(torch.count_nonzero(cell.role_confidence).item()) == 0
                    and int(torch.count_nonzero(cell.support_scale).item()) == 0
                ):
                    internal_zeroized += 1
        if internal_zeroized != expected:
            raise BranchInteractionGraphError("reduced internal tensors did not zeroize")
        result["reduced_internal_tensor_zeroized_count"] = internal_zeroized
        result["persistent_raw_tensor_artifact_created"] = False
        result["representation_digest"] = object_sha256(result)
        return result


__all__ = [
    "ARMS",
    "BLOCKS",
    "BranchInteractionGraphError",
    "FEATURE_NAMES",
    "LIFECYCLE_BINS",
    "METHOD",
    "PHASES",
    "SCHEMA_VERSION",
    "SIGMA_BANDS",
    "SOFT_EDGE_TEMPERATURE",
    "StreamingBranchInteractionGraphObserver",
    "canonical_json_bytes",
    "object_sha256",
]

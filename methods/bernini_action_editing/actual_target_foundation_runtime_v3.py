#!/usr/bin/env python3
"""Frozen one-device V3 runtime for the seen actual-target development canary.

The default is fail-closed.  Importing and CPU tests do not import torch,
SAM2, CoTracker or transformers.  A real run persists only complete scalar /
digest mechanical evidence.  The recursive inventory owns and scrubs every
external foundation-returned raw storage and named retained representation
leaf.  Non-persisted deterministic pure-CPU reduction workspaces are explicitly
outside that zeroization claim and never enter a receipt or persisted payload.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, NoReturn, Optional, Protocol, Sequence

import actual_target_foundation_canary_v3 as authority
import actual_target_foundation_graph_v3 as graph_v3
import actual_target_foundation_snapshot_v3 as snapshot_v3


SCHEMA = "actual-target-foundation-runtime-v3"
REAL_GPU_LAUNCH_AUTHORIZED = True
PHASES = 8
SHUFFLE = (0, 2, 4, 6, 7, 5, 3, 1)
VIEWS = (
    "target_forward_reference",
    "target_forward_eval",
    "target_reverse",
    "target_deterministic_shuffle",
    "source_noop",
)
EXPECTED_LOGICAL_COUNTS = {
    "media_decode": 8,
    "sam2": 96,
    "dinov2": 96,
    "cotracker": 20,
    "vjepa2": 20,
}
EXPECTED_HOOK_COUNTS = {
    "sam2_image_encoder": 96,
    "dinov2": 96,
    "cotracker": 20,
    "vjepa2": 20,
}
RAW_OBSERVED_COUNT_KEYS = (
    "compressed_video_hash_requests",
    "decoded_bgr_frames",
    "decoded_rgb_frames",
    "sam_ann_records_before_filter",
    "sam_mask_coordinate_calls",
    "dino_processor_tensor_items",
    "dino_model_output_unique_storages",
    "dino_filtered_ann_records",
    "dino_positive_support_records",
    "cotracker_membership_rows",
    "vjepa_processor_tensor_items",
    "vjepa_model_output_unique_storages",
    "model_tensor_hash_requests",
)
MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS = {"dinov2": 1, "vjepa2": 4}
FORBIDDEN_RECEIPT_KEYS = {
    "masks",
    "mask_payload",
    "embeddings",
    "trajectories",
    "track_coordinates",
    "teacher_payload",
    "descriptors",
    "feature_vectors",
}


class RuntimeV3Error(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise RuntimeV3Error(message)


def _cosine(
    left: Sequence[float], right: Sequence[float]
) -> Optional[float]:
    try:
        left_length = len(left)
        right_length = len(right)
    except (TypeError, ValueError, OverflowError):
        return None
    if left_length != right_length or left_length == 0:
        return None
    try:
        lhs_values = tuple(float(item) for item in left)
        rhs_values = tuple(float(item) for item in right)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in (*lhs_values, *rhs_values)):
        return None
    try:
        dot = sum(lhs * rhs for lhs, rhs in zip(lhs_values, rhs_values))
        lhs_norm = math.sqrt(sum(item**2 for item in lhs_values))
        rhs_norm = math.sqrt(sum(item**2 for item in rhs_values))
    except (ValueError, OverflowError):
        return None
    if lhs_norm <= 1e-12 and rhs_norm <= 1e-12:
        return None
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        return 0.0
    result = dot / (lhs_norm * rhs_norm)
    return result if math.isfinite(result) else None


def _margin(
    reference: Sequence[float], positive: Sequence[float], control: Sequence[float]
) -> Optional[float]:
    positive_similarity = _cosine(reference, positive)
    control_similarity = _cosine(reference, control)
    if positive_similarity is None or control_similarity is None:
        return None
    result = positive_similarity - control_similarity
    return result if math.isfinite(result) else None


def _l2(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    try:
        left_length = len(left)
        right_length = len(right)
    except (TypeError, ValueError, OverflowError):
        return None
    if left_length != right_length:
        return None
    try:
        differences = tuple(float(lhs) - float(rhs) for lhs, rhs in zip(left, right))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in differences):
        return None
    try:
        result = math.sqrt(sum(item**2 for item in differences))
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _norm(value: Sequence[float]) -> Optional[float]:
    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in items):
        return None
    try:
        result = math.sqrt(sum(item**2 for item in items))
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _permute_blocks(vector: Sequence[float], order: Sequence[int]) -> tuple[float, ...]:
    if len(vector) % PHASES:
        fail("feature vector is not divisible into eight phase blocks")
    width = len(vector) // PHASES
    return tuple(float(vector[phase * width + index]) for phase in order for index in range(width))


def _mask_descriptor_negative(nodes: "NodeSketch") -> tuple[float, ...]:
    if nodes.private_payload is None or any(len(phase) < 2 for phase in nodes.private_payload):
        return tuple()
    return graph_v3.canonical_node_signature(
        graph_v3.break_mask_descriptor_binding(nodes.private_payload)
    )


def _track_identity_negative(motion: "MotionSketch") -> tuple[float, ...]:
    width = motion.track_block_width
    if width <= 0 or motion.assigned_track_count < 2 or len(motion.track_signature) != width * motion.track_count:
        return tuple()
    blocks = [
        list(motion.track_signature[index * width : (index + 1) * width])
        for index in range(motion.track_count)
    ]
    descriptors = [block[:8] for block in blocks[: motion.assigned_track_count]]
    for index, block in enumerate(blocks[: motion.assigned_track_count]):
        block[:8] = descriptors[(index + 1) % len(descriptors)]
    return tuple(float(value) for block in blocks for value in block)


@dataclass(frozen=True)
class NodeSketch:
    signature: Any
    cardinalities: tuple[int, ...]
    mechanically_valid_phases: int
    dustbin_used: bool
    private_payload: Any = None
    unbalanced_phase_pair_count: int = 0
    dustbin_unmatched_count: int = 0
    dustbin_transport_mass: float = 0.0


@dataclass(frozen=True)
class MotionSketch:
    track_signature: Any
    edge_signature: Any
    drop_edge_signature: Any
    assigned_track_count: int
    assigned_point_count: int
    minimum_same_track_member_phases_observed: int
    visible_and_member_fraction: Optional[float]
    per_phase_visible_member_counts: tuple[int, ...]
    assignment_diagnostics: Mapping[str, int]
    state_counts: Mapping[str, int]
    lifecycle_counts: Mapping[str, int]
    valid_adjacent_velocity_count: int
    per_phase_active_counts: tuple[int, ...]
    per_phase_birth_counts: tuple[int, ...]
    per_phase_persist_counts: tuple[int, ...]
    per_phase_death_counts: tuple[int, ...]
    per_phase_valid_velocity_counts: tuple[int, ...]
    per_phase_qualified_lifecycle_counts: tuple[int, ...]
    evaluated_pairwise_edge_count: int
    drop_edge_removed_count: int
    track_block_width: int = 12
    track_count: int = 96


@dataclass(frozen=True)
class PhaseSketch:
    signature: Any


class FrozenBackend(Protocol):
    model_names: Sequence[str]

    def decode(self, path: str, expected_sha256: str) -> Sequence[Any]: ...
    def node(self, frames: Sequence[Any], view: str) -> NodeSketch: ...
    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch: ...
    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch: ...
    def frozen_receipt(self) -> Mapping[str, Any]: ...
    def begin_case(self) -> None: ...
    def scrub_case(self) -> None: ...


class RawInventoryV3:
    """Single-owner inventory for the explicitly in-scope mutable storages."""

    def __init__(self, required_categories: Sequence[str]):
        self.required = tuple(required_categories)
        self._owned: list[tuple[Any, str, frozenset[tuple[Any, ...]]]] = []
        self._owned_ids: set[int] = set()
        self._owned_storage_keys: set[tuple[Any, ...]] = set()
        self.opportunities: dict[str, int] = {name: 0 for name in self.required}
        self.produced: dict[str, int] = {name: 0 for name in self.required}
        self.registered: dict[str, int] = {name: 0 for name in self.required}
        self.zeroized: dict[str, int] = {name: 0 for name in self.required}
        self.failure_attempts: dict[str, int] = {name: 0 for name in self.required}
        self.observed: dict[str, int] = {
            name: 0 for name in RAW_OBSERVED_COUNT_KEYS
        }

    def observe(self, name: str, count: int = 1) -> None:
        if name not in self.observed:
            fail(f"raw inventory observed-count key is not preregistered: {name}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            fail("raw inventory observed count is invalid")
        self.observed[name] += count

    def mark_opportunity(self, category: str, count: int = 1) -> None:
        if category not in self.opportunities:
            fail(f"raw inventory category is not preregistered: {category}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            fail("raw inventory opportunity count is invalid")
        self.opportunities[category] += count

    def mark_produced(self, category: str, count: int = 1) -> None:
        if category not in self.produced:
            fail(f"raw inventory category is not preregistered: {category}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            fail("raw inventory produced count is invalid")
        if self.produced[category] + count > self.opportunities[category]:
            fail("raw inventory production exceeds independently marked opportunity")
        self.produced[category] += count

    def mark_unregistered_zeroized(self, category: str, count: int = 1) -> None:
        """Account for produced storage scrubbed after registration failed."""

        if category not in self.zeroized:
            fail(f"raw inventory category is not preregistered: {category}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            fail("raw inventory zeroized count is invalid")
        if self.zeroized[category] + count > self.produced[category]:
            fail("raw inventory zeroization exceeds produced storage count")
        self.zeroized[category] += count

    @staticmethod
    def _storage_key(value: Any) -> tuple[Any, ...]:
        try:
            storage = value.untyped_storage()
            pointer = int(storage.data_ptr())
            size = int(storage.nbytes())
            if pointer and size:
                return ("torch", str(value.device), pointer, size)
        except (AttributeError, RuntimeError, TypeError):
            pass
        if hasattr(value, "__array_interface__"):
            root = value
            seen = set()
            while getattr(root, "base", None) is not None and id(root) not in seen:
                seen.add(id(root))
                root = root.base
            try:
                size = int(value.nbytes)
            except (AttributeError, TypeError, ValueError):
                size = 0
            if size:
                return ("array", id(root))
        return ("identity", id(value))

    @classmethod
    def _storage_keys(
        cls, value: Any, seen: Optional[set[int]] = None
    ) -> frozenset[tuple[Any, ...]]:
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return frozenset()
        seen.add(identity)
        if isinstance(value, Mapping):
            keys = {
                key
                for child in value.values()
                for key in cls._storage_keys(child, seen)
            }
            return frozenset(keys or {("identity", identity)})
        if isinstance(value, (list, tuple)):
            keys = {
                key
                for child in value
                for key in cls._storage_keys(child, seen)
            }
            return frozenset(keys or {("identity", identity)})
        return frozenset({cls._storage_key(value)})

    def own(self, value: Any, category: str) -> Any:
        if category not in self.registered:
            fail(f"raw inventory category is not preregistered: {category}")
        if self.registered[category] >= self.produced[category]:
            fail("raw inventory ownership lacks an independent production mark")
        if id(value) in self._owned_ids:
            fail("raw inventory double ownership is forbidden")
        storage_keys = self._storage_keys(value)
        if storage_keys & self._owned_storage_keys:
            fail("raw inventory storage alias ownership is forbidden")
        self._owned.append((value, category, storage_keys))
        self._owned_ids.add(id(value))
        self._owned_storage_keys.update(storage_keys)
        self.registered[category] += 1
        return value

    def _scrub_recursive(self, value: Any, seen: set[int]) -> bool:
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        try:
            if isinstance(value, bytearray):
                if value:
                    ctypes.memset(
                        ctypes.addressof(ctypes.c_char.from_buffer(value)),
                        0,
                        len(value),
                    )
                return not any(value)
            if hasattr(value, "zero_") and callable(value.zero_):
                value.zero_()
                return not bool(value.any())
            if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "__setitem__"):
                value[...] = 0
                return not bool(value.any())
            if isinstance(value, Mapping):
                results = [self._scrub_recursive(child, seen) for child in list(value.values())]
                clean = all(results)
                if clean and hasattr(value, "clear"):
                    value.clear()
                return clean and (not hasattr(value, "clear") or len(value) == 0)
            if isinstance(value, list):
                results = [self._scrub_recursive(child, seen) for child in list(value)]
                clean = all(results)
                if clean:
                    value.clear()
                return clean and not value
            if isinstance(value, tuple):
                results = [self._scrub_recursive(child, seen) for child in value]
                return all(results)
            return False
        except BaseException:
            return False

    def release(self, value: Any) -> None:
        index = next(
            (i for i, (item, _, _) in enumerate(self._owned) if item is value),
            None,
        )
        if index is None:
            fail("raw inventory release is not single-owner")
        item, category, storage_keys = self._owned[index]
        clean = self._scrub_recursive(item, set())
        if not clean:
            self.failure_attempts[category] += 1
            fail(f"raw inventory immediate scrub failed: {category}")
        self._owned.pop(index)
        self._owned_ids.remove(id(item))
        self._owned_storage_keys.difference_update(storage_keys)
        self.zeroized[category] += 1

    def scrub_all(self) -> None:
        pending = list(self._owned)
        for _attempt in range(2):
            remaining = []
            for item, category, storage_keys in pending:
                if self._scrub_recursive(item, set()):
                    self.zeroized[category] += 1
                else:
                    self.failure_attempts[category] += 1
                    remaining.append((item, category, storage_keys))
            pending = remaining
            if not pending:
                break
        self._owned = list(pending)
        self._owned_ids = {id(item) for item, _, _ in pending}
        self._owned_storage_keys = {
            key for _, _, storage_keys in pending for key in storage_keys
        }
        if pending:
            fail(
                "raw inventory best-effort scrub failures: "
                f"{sorted(category for _, category, _ in pending)}"
            )

    def receipt(self, *, require_all_categories: bool) -> Mapping[str, Any]:
        raw_scope = authority.load_authority()["raw_ownership_contract"]
        missing = sorted(
            name
            for name in self.required
            if name not in self.opportunities
            or name not in self.produced
            or name not in self.registered
            or name not in self.zeroized
        )
        zero_produced = sorted(
            name for name in self.required if self.produced[name] == 0
        )
        value = {
            "schema_version": "actual-target-raw-inventory-v3",
            "required_categories": list(self.required),
            "opportunity_by_category": dict(self.opportunities),
            "produced_by_category": dict(self.produced),
            "registered_by_category": dict(self.registered),
            "zeroized_by_category": dict(self.zeroized),
            "failure_attempts_by_category": dict(self.failure_attempts),
            "observed_counts": dict(sorted(self.observed.items())),
            "opportunity_total": sum(self.opportunities.values()),
            "produced_total": sum(self.produced.values()),
            "registered_total": sum(self.registered.values()),
            "zeroized_total": sum(self.zeroized.values()),
            "outstanding_count": len(self._owned),
            "missing_required_categories": missing,
            "zero_produced_categories": zero_produced,
            "zero_produced_categories_are_valid_abstention": True,
            "observed_count_keys": list(RAW_OBSERVED_COUNT_KEYS),
            "model_output_unique_storage_multipliers": dict(
                MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS
            ),
            "model_output_unique_storage_evidence_digest": authority.object_sha256(
                raw_scope["model_output_unique_storage_evidence"]
            ),
            "production_binding_rule": (
                "the external controller independently rebuilds every category "
                "count from fixed logical calls, decoded-frame authority, model "
                "tensor_count, upstream SAM/DINO/CoTracker/V-JEPA observations, "
                "and preregistered fixed multipliers"
            ),
            "in_scope_storage_boundary": (
                "external foundation-returned raw storages and named retained "
                "representation leaves only"
            ),
            "excluded_ephemeral_workspace_boundary": (
                "non-persisted deterministic pure-CPU reduction workspaces"
            ),
            "recursive_best_effort_scrub": True,
            "verified": (
                not self._owned
                and self.opportunities == self.produced
                and self.produced == self.registered
                and self.registered == self.zeroized
                and not any(self.failure_attempts.values())
                and (not require_all_categories or not missing)
                and (
                    "sam_ann_mask_pre_filter" not in self.registered
                    or self.observed.get("sam_ann_records_before_filter", 0)
                    == self.registered["sam_ann_mask_pre_filter"]
                )
            ),
        }
        return {**value, "digest": authority.object_sha256(value)}


class CountedBackend:
    def __init__(self, backend: FrozenBackend):
        self.backend = backend
        self.counts = {name: 0 for name in EXPECTED_LOGICAL_COUNTS}

    def decode(self, path: str, digest: str) -> Sequence[Any]:
        self.counts["media_decode"] += 1
        return self.backend.decode(path, digest)

    def node(self, frames: Sequence[Any], view: str) -> NodeSketch:
        value = self.backend.node(frames, view)
        self.counts["sam2"] += PHASES
        self.counts["dinov2"] += PHASES
        return value

    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch:
        self.counts["cotracker"] += 1
        return self.backend.motion(frames, view, nodes)

    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch:
        self.counts["vjepa2"] += 1
        return self.backend.phase(frames, view)


def _sample(frames: Sequence[Any], count: int) -> tuple[Any, ...]:
    if len(frames) < count:
        fail(f"video has fewer than {count} decoded frames")
    indices = tuple(round(index * (len(frames) - 1) / (count - 1)) for index in range(count))
    if len(set(indices)) != count:
        fail("fixed sampling produced duplicate indices")
    return tuple(frames[index] for index in indices)


def _views(source: Sequence[Any], target: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    source8 = _sample(source, PHASES)
    target16 = _sample(target, 16)
    reference, evaluation = target16[0::2], target16[1::2]
    return {
        "target_forward_reference": reference,
        "target_forward_eval": evaluation,
        "target_reverse": tuple(reversed(evaluation)),
        "target_deterministic_shuffle": tuple(evaluation[index] for index in SHUFFLE),
        "source_noop": source8,
    }


def _phase_views(source: Sequence[Any], target: Sequence[Any]) -> Mapping[str, tuple[Any, ...]]:
    source16 = _sample(source, 16)
    target32 = _sample(target, 32)
    reference, evaluation = target32[0::2], target32[1::2]
    blocks = tuple(evaluation[index : index + 2] for index in range(0, 16, 2))
    return {
        "target_forward_reference": reference,
        "target_forward_eval": evaluation,
        "target_reverse": tuple(reversed(evaluation)),
        "target_deterministic_shuffle": tuple(frame for index in SHUFFLE for frame in blocks[index]),
        "source_noop": source16,
    }


def _hook_counts(backend: Any) -> Mapping[str, int]:
    value = getattr(backend, "actual_forward_counts", lambda: dict(EXPECTED_HOOK_COUNTS))()
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_HOOK_COUNTS):
        fail("actual foundation forward hook counter schema differs")
    return {name: int(value[name]) for name in EXPECTED_HOOK_COUNTS}


def _case_evidence(
    pair: Mapping[str, Any], backend: CountedBackend
) -> authority.CaseEvidenceV3:
    hook_before = _hook_counts(backend.backend)
    source = backend.decode(pair["source_video_path"], pair["source_video_sha256"])
    target = backend.decode(pair["target_video_path"], pair["target_video_sha256"])
    views = _views(source, target)
    phase_views = _phase_views(source, target)
    nodes = {
        name: backend.node(views[name], name)
        for name in ("target_forward_reference", "target_forward_eval", "source_noop")
    }
    positive_node = nodes["target_forward_eval"]
    own_derived = getattr(
        backend.backend,
        "own_derived_signature",
        lambda values, _category: tuple(float(value) for value in values),
    )
    nodes["target_reverse"] = NodeSketch(
        own_derived(
            _permute_blocks(positive_node.signature, tuple(reversed(range(PHASES)))),
            "node_signature",
        ),
        tuple(reversed(positive_node.cardinalities)),
        positive_node.mechanically_valid_phases,
        positive_node.dustbin_used,
        tuple(reversed(positive_node.private_payload)),
    )
    nodes["target_deterministic_shuffle"] = NodeSketch(
        own_derived(_permute_blocks(positive_node.signature, SHUFFLE), "node_signature"),
        tuple(positive_node.cardinalities[index] for index in SHUFFLE),
        positive_node.mechanically_valid_phases,
        positive_node.dustbin_used,
        tuple(positive_node.private_payload[index] for index in SHUFFLE),
    )
    motions = {name: backend.motion(views[name], name, nodes[name]) for name in VIEWS}
    phases = {name: backend.phase(phase_views[name], name) for name in VIEWS}
    hook_after = _hook_counts(backend.backend)
    hook_delta = {name: hook_after[name] - hook_before[name] for name in EXPECTED_HOOK_COUNTS}
    ref, pos = "target_forward_reference", "target_forward_eval"
    controls = {
        "target_reverse": "target_reverse",
        "target_deterministic_shuffle": "target_deterministic_shuffle",
        "source_noop": "source_noop",
    }

    def margins(
        reference: Sequence[float], positive: Sequence[float], branch: str
    ) -> Mapping[str, Optional[float]]:
        values = {
            "node": nodes,
            "track": {name: motions[name].track_signature for name in VIEWS},
            "edge": {name: motions[name].edge_signature for name in VIEWS},
            "ordered_phase": {name: phases[name].signature for name in VIEWS},
        }[branch]
        return {
            control: _margin(reference, positive, values[name] if branch != "node" else values[name].signature)
            for control, name in controls.items()
        }

    frozen = backend.backend.frozen_receipt()
    positive_motion = motions[pos]
    state_counts = {name: 0 for name in ("ABSENT", "VISIBLE_MEMBER", "OCCLUDED", "VISIBLE_OUTSIDE_MASK")}
    state_counts.update(positive_motion.state_counts)
    lifecycle_counts = {name: 0 for name in ("entry", "occlusion", "membership_loss", "reentry", "death")}
    lifecycle_counts.update(positive_motion.lifecycle_counts)
    assignment = positive_motion.assignment_diagnostics
    mask_negative = own_derived(_mask_descriptor_negative(positive_node), "node_signature")
    track_negative = own_derived(_track_identity_negative(positive_motion), "track_signature")
    drop_similarity = _cosine(motions[ref].edge_signature, positive_motion.drop_edge_signature)
    edge_positive_similarity = _cosine(
        motions[ref].edge_signature, positive_motion.edge_signature
    )
    drop_margin = (
        edge_positive_similarity - drop_similarity
        if edge_positive_similarity is not None and drop_similarity is not None
        else None
    )
    evidence = authority.CaseEvidenceV3(
        family=pair["family"],
        pair_id=pair["pair_id"],
        branches={
            "frozen_base": {
                "all_models_eval_frozen": frozen.get("all_models_eval_frozen") is True,
                "source_and_weight_closure_unchanged": frozen.get("source_and_weight_closure_unchanged") is True,
                "parameter_updates": frozen.get("parameter_updates"),
                "generator_forward_calls": frozen.get("generator_forward_calls"),
                "actual_forward_hook_delta": hook_delta,
                "full_model_closure_deferred_to_run_receipt": True,
            },
            "node": {
                "dustbin_used": positive_node.dustbin_used,
                "unbalanced_phase_pair_count": positive_node.unbalanced_phase_pair_count,
                "dustbin_unmatched_count": positive_node.dustbin_unmatched_count,
                "dustbin_transport_mass": positive_node.dustbin_transport_mass,
                "forced_nonempty_slot_used": False,
                "anonymous_slot_relabel_invariant": graph_v3.canonical_node_signature(graph_v3.relabel_slots(positive_node.private_payload)) == tuple(float(value) for value in positive_node.signature),
                "phase_cardinalities": list(positive_node.cardinalities),
                "mechanically_valid_phases": positive_node.mechanically_valid_phases,
                "positive_similarity": _cosine(nodes[ref].signature, positive_node.signature),
                "input_margins": margins(nodes[ref].signature, positive_node.signature, "node"),
                "mask_descriptor_binding_break_margin": _margin(nodes[ref].signature, positive_node.signature, mask_negative),
            },
            "track": {
                "assigned_track_count": positive_motion.assigned_track_count,
                "assigned_point_count": positive_motion.assigned_point_count,
                "minimum_same_track_member_phases_observed": positive_motion.minimum_same_track_member_phases_observed,
                "visible_and_member_fraction": positive_motion.visible_and_member_fraction,
                "per_phase_visible_member_counts": list(positive_motion.per_phase_visible_member_counts),
                "ambiguous_overlap_observation_count": assignment["ambiguous_overlap_observation_count"],
                "out_of_bounds_observation_count": assignment["out_of_bounds_observation_count"],
                "nonfinite_observation_count": assignment["nonfinite_observation_count"],
                "vote_tie_abstain_count": assignment["vote_tie_abstain_count"],
                "insufficient_membership_abstain_count": assignment["insufficient_membership_abstain_count"],
                "state_counts": state_counts,
                "lifecycle_counts": lifecycle_counts,
                "dynamic_nonentry_lifecycle_observed": sum(lifecycle_counts[name] for name in ("occlusion", "membership_loss", "reentry", "death")) >= 1,
                "valid_adjacent_velocity_count": positive_motion.valid_adjacent_velocity_count,
                "positive_similarity": _cosine(motions[ref].track_signature, positive_motion.track_signature),
                "input_margins": margins(motions[ref].track_signature, positive_motion.track_signature, "track"),
                "cross_phase_track_identity_break_margin": _margin(motions[ref].track_signature, positive_motion.track_signature, track_negative),
            },
            "edge": {
                "per_phase_active_counts": list(positive_motion.per_phase_active_counts),
                "per_phase_birth_counts": list(positive_motion.per_phase_birth_counts),
                "per_phase_persist_counts": list(positive_motion.per_phase_persist_counts),
                "per_phase_death_counts": list(positive_motion.per_phase_death_counts),
                "per_phase_valid_velocity_counts": list(positive_motion.per_phase_valid_velocity_counts),
                "per_phase_qualified_lifecycle_counts": list(positive_motion.per_phase_qualified_lifecycle_counts),
                "evaluated_pairwise_edge_count": positive_motion.evaluated_pairwise_edge_count,
                "real_per_phase_lifecycle_channels": True,
                "positive_similarity": edge_positive_similarity,
                "input_margins": margins(motions[ref].edge_signature, positive_motion.edge_signature, "edge"),
                "drop_edge_margin": drop_margin,
                "drop_edge_removed_count": positive_motion.drop_edge_removed_count,
                "drop_edge_control_norm": _norm(positive_motion.drop_edge_signature),
                "drop_edge_control_similarity": drop_similarity,
                "drop_edge_positive_l2_distance": _l2(positive_motion.drop_edge_signature, positive_motion.edge_signature),
            },
            "ordered_phase": {
                "input_margins": margins(phases[ref].signature, phases[pos].signature, "ordered_phase")
            },
        },
    )
    nodes.clear()
    motions.clear()
    phases.clear()
    return evidence


def _safe_receipt(value: Mapping[str, Any]) -> None:
    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if FORBIDDEN_RECEIPT_KEYS.intersection(node):
                fail("receipt attempts to persist raw teacher payload")
            for child in node.values():
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    authority.canonical_json_bytes(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_only_bytes(path: Path, payload: bytes, mode: int = 0o444) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        fail("create-only output must be an absolute absent path")
    try:
        snapshot_v3._plain_directory(path.parent)
    except Exception as error:
        raise RuntimeV3Error("create-only output parent is not a lexical plain directory") from error
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)
    _fsync_directory(path.parent)


def create_only_json(path: Path, value: Mapping[str, Any]) -> None:
    _safe_receipt(value)
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False
    ).encode("ascii") + b"\n"
    create_only_bytes(path, payload, 0o444)


def _case_receipt(evidence: authority.CaseEvidenceV3, evaluated: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence_mapping = evidence.to_mapping()
    evidence_row = {**evidence_mapping, "digest": authority.object_sha256(evidence_mapping)}
    value = {
        "schema_version": "actual-target-foundation-mechanical-case-v3",
        "pair_id": evidence.pair_id,
        "family": evidence.family,
        "case_evidence": evidence_row,
        "evaluated_case": evaluated,
        "raw_teacher_payload_persisted": False,
    }
    return {**value, "digest": authority.object_sha256(value)}


def run_canary(
    backend: FrozenBackend,
    *,
    output: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Mapping[str, Any]:
    prereg = authority.load_preregistration()
    counted = CountedBackend(backend)
    evidences: list[authority.CaseEvidenceV3] = []
    evaluated_rows: list[Mapping[str, Any]] = []
    for pair in prereg["pairs"]:
        backend.begin_case()
        try:
            evidence = _case_evidence(pair, counted)
            evaluated = authority.evaluate_case(evidence, prereg)
            case_receipt = _case_receipt(evidence, evaluated)
        finally:
            backend.scrub_case()
        evidences.append(evidence)
        evaluated_rows.append(evaluated)
        if cache_dir is not None:
            if not cache_dir.is_absolute() or cache_dir.is_symlink() or not cache_dir.is_dir():
                fail("cache directory must be absolute existing non-symlink directory")
            create_only_json(cache_dir / f'{pair["pair_id"]}.json', case_receipt)
    if counted.counts != EXPECTED_LOGICAL_COUNTS:
        fail(f"logical foundation counts differ: {counted.counts}")
    actual_hooks = _hook_counts(backend)
    aggregate = authority.aggregate_canary(evaluated_rows, evidences, prereg)
    model_closure = dict(
        getattr(backend, "finalize_model_closure", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    raw_receipt = dict(
        getattr(backend, "raw_ownership_receipt", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    asset_closure = dict(
        getattr(backend, "asset_closure_receipt", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    media_closure = dict(
        getattr(backend, "decoded_media_receipt", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    hydra_closure = dict(
        getattr(backend, "hydra_config_receipt", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    device_closure = dict(
        getattr(backend, "device_receipt", lambda: {"mode": "fake_cpu_contract", "verified": True})()
    )
    mechanical_rows = []
    for evidence in evidences:
        mapping = evidence.to_mapping()
        mechanical_rows.append({**mapping, "digest": authority.object_sha256(mapping)})
    forward_value = {
        "logical_counts": counted.counts,
        "actual_forward_hook_counts": actual_hooks,
        "expected_logical_counts": EXPECTED_LOGICAL_COUNTS,
        "expected_actual_forward_hook_counts": EXPECTED_HOOK_COUNTS,
        "verified": counted.counts == EXPECTED_LOGICAL_COUNTS and actual_hooks == EXPECTED_HOOK_COUNTS,
    }
    forward_closure = {**forward_value, "digest": authority.object_sha256(forward_value)}
    value = {
        "schema_version": SCHEMA,
        "experiment_id": authority.EXPERIMENT_ID,
        "scope": "seen_development_only_not_locked_validation_not_scientific_evidence",
        "mechanical_case_evidence": mechanical_rows,
        "cases": evaluated_rows,
        "aggregate": aggregate,
        "forward_closure": forward_closure,
        "raw_ownership": raw_receipt,
        "model_closure": model_closure,
        "device_closure": device_closure,
        "hydra_config_closure": hydra_closure,
        "asset_closure": asset_closure,
        "decoded_media_closure": media_closure,
        "runtime_source_closure": source_closure(),
        "training_performed": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "generator_loaded": False,
        "generator_forward_calls": 0,
        "raw_teacher_payload_persisted": False,
        "representation_admission_hard_false": True,
        "scientific_evidence_claimed": False,
        "completion_authority": {
            "candidate_file_presence_is_completion_authority": False,
            "external_controller_required": True,
            "external_controller_valid_outcomes": ["PASS", "REJECTED"],
            "external_completion_seal_written_by_probe": False,
        },
        "launch_contract_digest": launch_contract()["digest"],
    }
    receipt = {**value, "digest": authority.object_sha256(value)}
    _safe_receipt(receipt)
    if output is not None:
        create_only_json(output, receipt)
    return receipt


class RealFrozenBackend:
    model_names = ("sam2", "cotracker", "dinov2", "vjepa2")

    def __init__(self, device: str = "cuda:0"):
        if device != "cuda:0":
            fail("real V3 canary is sealed to exactly cuda:0")
        self._asset_receipt = authority.verify_remote_assets()
        import torch
        import numpy as np
        from transformers import AutoImageProcessor, AutoModel, AutoVideoProcessor
        from hydra import compose
        from omegaconf import OmegaConf
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        availability = authority.load_availability()["foundations"]
        v3_authority = authority.load_authority()
        hydra_spec = v3_authority["sam_hydra_authority"]
        cfg = compose(
            config_name=hydra_spec["config_name"],
            overrides=list(hydra_spec["exact_overrides"]),
        )
        OmegaConf.resolve(cfg)
        resolved = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        resolved_bytes = authority.canonical_json_bytes(resolved)
        resolved_digest = hashlib.sha256(resolved_bytes).hexdigest()
        if (
            len(resolved_bytes) != hydra_spec["resolved_canonical_json_bytes"]
            or resolved_digest != hydra_spec["resolved_canonical_sha256"]
            or resolved["model"]["_target_"] != hydra_spec["resolved_model_target"]
        ):
            fail("actual Hydra-resolved SAM config differs")
        self._hydra_receipt = {
            "verified": True,
            "config_name": hydra_spec["config_name"],
            "runtime_config_path": hydra_spec["runtime_config_path"],
            "runtime_config_sha256": hydra_spec["runtime_config_sha256"],
            "exact_overrides": list(hydra_spec["exact_overrides"]),
            "apply_postprocessing": False,
            "resolved_canonical_json_bytes": len(resolved_bytes),
            "resolved_canonical_sha256": resolved_digest,
            "resolved_model_target": resolved["model"]["_target_"],
        }
        self._hydra_receipt = {
            **self._hydra_receipt,
            "digest": authority.object_sha256(self._hydra_receipt),
        }

        cot_root = availability["cotracker"]["repository_root"]
        if cot_root not in sys.path:
            sys.path.insert(0, cot_root)
        from cotracker.predictor import CoTrackerPredictor

        self.torch, self.np, self.device = torch, np, device
        self._build_sam2 = build_sam2
        self.raw = RawInventoryV3(authority.load_authority()["raw_inventory_required_categories"])
        sam_model = build_sam2(
            hydra_spec["config_name"],
            availability["sam2"]["checkpoint_path"],
            device=device,
            mode="eval",
            hydra_overrides_extra=list(hydra_spec["exact_overrides"]),
            apply_postprocessing=False,
        )
        self.sam = SAM2AutomaticMaskGenerator(
            sam_model,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.90,
            output_mode="binary_mask",
        )
        self.cotracker = CoTrackerPredictor(
            checkpoint=availability["cotracker"]["checkpoint_path"],
            offline=True,
            v2=False,
            window_len=60,
        ).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(
            availability["dinov2"]["model_root"], local_files_only=True
        )
        self.dino = AutoModel.from_pretrained(
            availability["dinov2"]["model_root"], local_files_only=True
        ).to(device).eval()
        self.vjepa_processor = AutoVideoProcessor.from_pretrained(
            availability["vjepa2"]["model_root"], local_files_only=True
        )
        self.vjepa = AutoModel.from_pretrained(
            availability["vjepa2"]["model_root"], local_files_only=True
        ).to(device).eval()
        self.models = {
            "sam2": sam_model,
            "cotracker": self.cotracker,
            "dinov2": self.dino,
            "vjepa2": self.vjepa,
        }
        for model in self.models.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self._hook_counts = {name: 0 for name in EXPECTED_HOOK_COUNTS}
        self._hooks = [
            sam_model.image_encoder.register_forward_hook(self._hook("sam2_image_encoder")),
            self.dino.register_forward_hook(self._hook("dinov2")),
            self.cotracker.register_forward_hook(self._hook("cotracker")),
            self.vjepa.register_forward_hook(self._hook("vjepa2")),
        ]
        self._model_binding = self._binding_receipt()
        self._versions = self._version_pointer_state()
        self._model_before = self._full_model_state()
        self._decoded_rows: list[Mapping[str, Any]] = []

    def _hook(self, name: str) -> Any:
        def count(_module: Any, _inputs: Any, _output: Any) -> None:
            self._hook_counts[name] += 1

        return count

    def own_derived_signature(self, values: Sequence[float], category: str) -> Any:
        return self._own_created(
            self.np.asarray(tuple(float(value) for value in values), dtype=self.np.float64),
            category,
        )

    def actual_forward_counts(self) -> Mapping[str, int]:
        return dict(self._hook_counts)

    def _binding_receipt(self) -> Mapping[str, Any]:
        expected = {
            (row["module"], row["class"]): row["source_sha256"]
            for row in authority.load_availability()["runtime_class_authority"]
        }
        classes = (
            self.models["sam2"].__class__,
            self.sam.__class__,
            self.cotracker.__class__,
            self.dino.__class__,
            self.vjepa.__class__,
        )
        rows = []
        for cls in classes:
            key = (cls.__module__, cls.__name__)
            source = Path(os.path.abspath(inspect.getsourcefile(cls) or ""))
            if key not in expected:
                fail(f"foundation class binding differs: {key}")
            payload = authority.stable_file_bytes(source)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != expected[key]:
                fail(f"foundation class source SHA differs: {key}")
            rows.append(
                {"module": key[0], "class": key[1], "source_path": str(source), "source_sha256": digest}
            )
        non_tensor = authority.load_authority()["preprocessor_and_nontensor_config_authority"]
        extra_objects = (
            ("sam_build_function", self._build_sam2),
            ("dinov2_processor", self.dino_processor.__class__),
            ("vjepa2_processor", self.vjepa_processor.__class__),
        )
        extra_rows = []
        for name, obj in extra_objects:
            spec = non_tensor[name]
            module = obj.__module__
            object_name = obj.__name__
            source = Path(os.path.abspath(inspect.getsourcefile(obj) or ""))
            payload = authority.stable_file_bytes(source)
            digest = hashlib.sha256(payload).hexdigest()
            expected_name = spec.get("name", spec.get("class"))
            if module != spec["module"] or object_name != expected_name or str(source) != spec["source_path"] or digest != spec["source_sha256"]:
                fail(f"preprocessor/build source binding differs: {name}")
            extra_rows.append({"role": name, "module": module, "name": object_name, "source_path": str(source), "source_sha256": digest})
        config_rows = []
        for name, obj, spec_name in (
            ("dinov2_processor", self.dino_processor, "dinov2_processor"),
            ("vjepa2_processor", self.vjepa_processor, "vjepa2_processor"),
            ("dinov2_model_config", self.dino.config, "dinov2_model_config"),
            ("vjepa2_model_config", self.vjepa.config, "vjepa2_model_config"),
        ):
            payload = authority.canonical_json_bytes(obj.to_dict())
            spec = non_tensor[spec_name]
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != spec["canonical_config_bytes"] or digest != spec["canonical_config_sha256"]:
                fail(f"non-tensor config binding differs: {name}")
            config_rows.append({"role": name, "canonical_config_bytes": len(payload), "canonical_config_sha256": digest})
        cot_spec = non_tensor["cotracker_runtime_config"]
        cot_observed = {
            "offline": True,
            "v2": bool(self.cotracker.v2),
            "window_len": 60,
            "support_grid_size": int(self.cotracker.support_grid_size),
            "interp_shape": [int(value) for value in self.cotracker.interp_shape],
            "grid_size": 12,
            "grid_query_frame": 0,
            "backward_tracking": True,
        }
        if cot_observed != cot_spec:
            fail("CoTracker non-tensor runtime config differs")
        if self.dino.config.model_type != "dinov2" or self.vjepa.config.model_type != "vjepa2" or int(self.vjepa.config.tubelet_size) != 2:
            fail("foundation model config binding differs")
        value = {
            "verified": True,
            "classes": rows,
            "preprocessor_and_build_sources": extra_rows,
            "non_tensor_configs": config_rows,
            "cotracker_runtime_config": cot_observed,
            "hydra_config_digest": self._hydra_receipt["digest"],
        }
        return {**value, "digest": authority.object_sha256(value)}

    def _version_pointer_state(self) -> tuple[tuple[Any, ...], ...]:
        rows = []
        for model_name, model in self.models.items():
            for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
                for name, tensor in iterator:
                    rows.append((model_name, kind, name, tensor._version, tensor.data_ptr(), str(tensor.device)))
        return tuple(rows)

    def _own_single_cpu_tensor_copy(self, tensor: Any) -> Any:
        self.raw.observe("model_tensor_hash_requests", 1)
        detached = tensor.detach()
        if detached.device.type == "cpu":
            created = detached.clone(memory_format=self.torch.contiguous_format)
        else:
            created = detached.to(
                device="cpu",
                copy=True,
                memory_format=self.torch.contiguous_format,
            )
        return self._own_created(created, "model_hash_copy")

    def _own_created(self, created: Any, category: str) -> Any:
        """Immediately register one newly allocated mutable storage or scrub it."""

        try:
            self.raw.mark_opportunity(category)
            self.raw.mark_produced(category)
            return self.raw.own(created, category)
        except BaseException:
            clean = self.raw._scrub_recursive(created, set())
            if clean and self.raw.produced.get(category, 0) > self.raw.registered.get(
                category, 0
            ):
                self.raw.mark_unregistered_zeroized(category)
            if not clean:
                fail(f"unregistered mutable storage could not be scrubbed: {category}")
            raise

    def _own_external_batch(
        self, entries: Sequence[tuple[Any, str]]
    ) -> tuple[Any, ...]:
        """Take every external leaf before use; a partial claim scrubs the batch."""

        pending = tuple(entries)
        try:
            for _, category in pending:
                self.raw.mark_opportunity(category)
            for _, category in pending:
                self.raw.mark_produced(category)
        except BaseException:
            for value, _ in pending:
                self.raw._scrub_recursive(value, set())
            raise
        owned: list[Any] = []
        for index, (value, category) in enumerate(pending):
            try:
                owned.append(self.raw.own(value, category))
            except BaseException as error:
                scrub_failures = []
                for unowned, unowned_category in pending[index:]:
                    if self.raw._scrub_recursive(unowned, set()):
                        self.raw.mark_unregistered_zeroized(unowned_category)
                    else:
                        scrub_failures.append(unowned_category)
                release_failures = []
                for previous in reversed(owned):
                    try:
                        self.raw.release(previous)
                    except BaseException:
                        release_failures.append("previously_owned")
                if scrub_failures or release_failures:
                    raise RuntimeV3Error(
                        "external batch ownership failed and best-effort scrub was incomplete: "
                        f"unowned={scrub_failures}, owned={release_failures}"
                    ) from error
                raise
        return tuple(owned)

    def _release_owned_batch(self, values: Sequence[Any]) -> None:
        failures = 0
        for value in reversed(tuple(values)):
            try:
                self.raw.release(value)
            except BaseException:
                failures += 1
        if failures:
            fail(f"best-effort owned batch release failed for {failures} leaves")

    def _sam_mask_backing_owner(self, mask: Any) -> Any:
        """Return the ndarray that owns a SAM mask's complete byte storage."""

        if not isinstance(mask, self.np.ndarray):
            return mask
        owner = mask
        seen: set[int] = set()
        while isinstance(getattr(owner, "base", None), self.np.ndarray):
            if id(owner) in seen:
                fail("SAM mask ndarray base chain is cyclic")
            seen.add(id(owner))
            owner = owner.base
        return owner

    def _validate_sam_mask_storage(self, mask: Any, owner: Any) -> None:
        """Admit only a writable C/F view spanning one complete ndarray owner."""

        if (
            not isinstance(mask, self.np.ndarray)
            or mask.ndim != 2
            or mask.dtype != self.np.bool_
        ):
            fail("SAM mask must be a two-dimensional bool ndarray")
        if not (mask.flags.c_contiguous or mask.flags.f_contiguous):
            fail("SAM mask must be C- or F-contiguous before normalization")
        if (
            not isinstance(owner, self.np.ndarray)
            or getattr(owner, "base", None) is not None
            or not mask.flags.writeable
            or not owner.flags.writeable
            or int(mask.nbytes) <= 0
            or int(mask.nbytes) != int(owner.nbytes)
            or int(mask.__array_interface__["data"][0])
            != int(owner.__array_interface__["data"][0])
        ):
            fail("SAM mask must span its complete writable ndarray backing storage")

    def _scrub_pending_sam_masks(self, masks: Sequence[Any]) -> None:
        """Best-effort own and zero every not-yet-processed SAM storage."""

        failures = 0
        for mask in tuple(masks):
            owner = self._sam_mask_backing_owner(mask)
            try:
                (owned_owner,) = self._own_external_batch(
                    ((owner, "sam_ann_mask_pre_filter"),)
                )
                self.raw.release(owned_owner)
            except BaseException:
                failures += 1
        if failures:
            fail(
                "best-effort pending SAM storage scrub failed for "
                f"{failures} masks"
            )

    def _normalize_sam_annotations(
        self, anns: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """Claim, C-copy, and immediately scrub each external SAM mask in order."""

        self.raw.observe("sam_ann_records_before_filter", len(anns))
        if any(
            not isinstance(ann, Mapping) or "segmentation" not in ann
            for ann in anns
        ):
            available = tuple(
                ann["segmentation"]
                for ann in anns
                if isinstance(ann, Mapping) and "segmentation" in ann
            )
            self._scrub_pending_sam_masks(available)
            fail("SAM annotation schema differs before filtering")

        masks = tuple(ann["segmentation"] for ann in anns)
        normalized_masks: list[Any] = []
        normalized_anns: list[Mapping[str, Any]] = []
        seen_external_storages: set[tuple[Any, ...]] = set()
        try:
            for index, (ann, mask) in enumerate(zip(anns, masks)):
                owner = self._sam_mask_backing_owner(mask)
                try:
                    (owned_owner,) = self._own_external_batch(
                        ((owner, "sam_ann_mask_pre_filter"),)
                    )
                    try:
                        storage_key = RawInventoryV3._storage_key(owned_owner)
                        if storage_key in seen_external_storages:
                            fail("SAM annotations reuse one mutable backing storage")
                        self._validate_sam_mask_storage(mask, owned_owner)

                        # The pinned SAM2 source constructs a full-storage
                        # transpose in its uncompressed-RLE path. That source
                        # fact is not inferred from V3R2's compound failure
                        # message: runtime independently admits complete C/F
                        # layouts, then always creates a distinct C-layout
                        # owner. ``ascontiguousarray`` is forbidden because a
                        # C-layout input could otherwise alias SAM storage.
                        normalized = self._own_created(
                            self.np.array(
                                mask,
                                dtype=self.np.bool_,
                                order="C",
                                copy=True,
                            ),
                            "sam_mask_c_contiguous_copy",
                        )
                        normalized_masks.append(normalized)
                        if (
                            normalized.shape != mask.shape
                            or normalized.dtype != self.np.bool_
                            or not normalized.flags.c_contiguous
                            or self.np.shares_memory(normalized, mask)
                            or not self.np.array_equal(normalized, mask)
                        ):
                            fail("SAM C-layout ownership normalization differs")
                        normalized_anns.append(
                            {**dict(ann), "segmentation": normalized}
                        )
                        seen_external_storages.add(storage_key)
                    finally:
                        # This release precedes ownership of the next mask, so
                        # no earlier external SAM storage remains live while a
                        # later annotation is copied.
                        self.raw.release(owned_owner)
                except BaseException:
                    self._scrub_pending_sam_masks(masks[index + 1 :])
                    raise
        except BaseException:
            self._release_owned_batch(normalized_masks)
            raise
        return normalized_anns

    def _own_external_tensor_tree(
        self,
        value: Any,
        category: str,
        *,
        observed_count_name: Optional[str] = None,
    ) -> tuple[Any, ...]:
        """Own each unique non-None tensor storage in a model-output tree."""

        leaves: list[Any] = []
        storage_keys: set[tuple[Any, ...]] = set()
        seen_containers: set[int] = set()

        def walk(node: Any) -> None:
            if isinstance(node, self.torch.Tensor):
                key = RawInventoryV3._storage_key(node)
                if key not in storage_keys:
                    storage_keys.add(key)
                    leaves.append(node)
                return
            if node is None:
                return
            identity = id(node)
            if identity in seen_containers:
                return
            if isinstance(node, Mapping):
                seen_containers.add(identity)
                for child in node.values():
                    walk(child)
                return
            if isinstance(node, (list, tuple)):
                seen_containers.add(identity)
                for child in node:
                    walk(child)

        walk(value)
        if not leaves:
            fail(f"external model output has no tensor leaves: {category}")
        if observed_count_name is not None:
            self.raw.observe(observed_count_name, len(leaves))
        return self._own_external_batch(
            tuple((leaf, category) for leaf in leaves)
        )

    def _owned_tensor_copy(
        self,
        tensor: Any,
        category: str,
        *,
        device: Optional[str] = None,
        dtype: Any = None,
    ) -> Any:
        """Create exactly one contiguous tensor storage and immediately own it."""

        kwargs: dict[str, Any] = {
            "copy": True,
            "memory_format": self.torch.contiguous_format,
        }
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return self._own_created(tensor.to(**kwargs), category)

    def _tensor_value_sha256(self, tensor: Any) -> str:
        with self.torch.no_grad():
            copy = self._own_single_cpu_tensor_copy(tensor)
            try:
                byte_view = copy.view(-1).view(self.torch.uint8).view(-1)
                hasher = hashlib.sha256()
                byte_count = int(byte_view.numel())
                if byte_count:
                    raw_buffer = (ctypes.c_ubyte * byte_count).from_address(
                        int(byte_view.data_ptr())
                    )
                    raw_view = memoryview(raw_buffer).cast("B")
                    for start in range(0, byte_count, 8 * 1024 * 1024):
                        hasher.update(raw_view[start : start + 8 * 1024 * 1024])
                digest = hasher.hexdigest()
            finally:
                self.raw.release(copy)
        return digest

    def _full_model_state(self) -> Mapping[str, Any]:
        rows = []
        for model_name, model in self.models.items():
            for kind, iterator in (
                ("parameter", model.named_parameters(remove_duplicate=False)),
                ("buffer", model.named_buffers(remove_duplicate=False)),
            ):
                for name, tensor in iterator:
                    rows.append(
                        {
                            "model": model_name,
                            "kind": kind,
                            "name": name,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                            "device": str(tensor.device),
                            "data_ptr": int(tensor.data_ptr()),
                            "value_sha256": self._tensor_value_sha256(tensor),
                            "requires_grad": bool(tensor.requires_grad) if kind == "parameter" else False,
                        }
                    )
        value = {"tensor_count": len(rows), "tensors": rows}
        return {**value, "digest": authority.object_sha256(value)}

    def _compressed_video_sha256(self, candidate: Path) -> str:
        """Hash compressed media through one owned mutable buffer."""

        try:
            snapshot_v3._plain_regular_file(candidate)
        except Exception as error:
            raise RuntimeV3Error("compressed video path is not lexical/plain") from error
        before = candidate.stat()
        self.raw.observe("compressed_video_hash_requests", 1)
        buffer = self._own_created(
            bytearray(8 * 1024 * 1024), "compressed_video_hash_buffer"
        )
        descriptor = None
        try:
            descriptor = os.open(
                candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            inside_before = os.fstat(descriptor)
            digest = hashlib.sha256()
            view = memoryview(buffer)
            while True:
                count = os.readv(descriptor, [buffer])
                if count == 0:
                    break
                digest.update(view[:count])
            inside_after = os.fstat(descriptor)
        finally:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                self.raw.release(buffer)
        after = candidate.stat()
        identity = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if not (
            identity(before)
            == identity(inside_before)
            == identity(inside_after)
            == identity(after)
        ):
            fail("compressed video changed during mutable streaming hash")
        return digest.hexdigest()

    def _own_decoded_rgb(self, frame: Any, cv2: Any) -> Any:
        self.raw.observe("decoded_bgr_frames", 1)
        (bgr,) = self._own_external_batch(
            ((frame, "decoded_bgr_frame"),)
        )
        try:
            converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.raw.observe("decoded_rgb_frames", 1)
            rgb = self._own_created(converted, "decoded_rgb_frame")
            if (
                rgb.dtype != self.np.uint8
                or rgb.shape != (720, 1280, 3)
                or not rgb.flags.c_contiguous
            ):
                fail("decoded RGB dtype/shape/contiguity differs")
            return rgb
        finally:
            self.raw.release(bgr)

    def begin_case(self) -> None:
        if self.raw._owned:
            fail("raw inventory from prior case was not scrubbed")

    def decode(self, path: str, expected_sha256: str) -> Sequence[Any]:
        import cv2

        candidate = Path(path)
        if self._compressed_video_sha256(candidate) != expected_sha256:
            fail("media compressed SHA differs")
        decode_row = next(
            (row for row in authority.load_decode_receipt()["rows"] if row["compressed_sha256"] == expected_sha256),
            None,
        )
        if decode_row is None:
            fail("media absent from decoded RGB authority")
        digest = hashlib.sha256(
            authority.canonical_json_bytes(
                {"dtype": "uint8", "shape": [decode_row["frame_count"], 720, 1280, 3]}
            )
        )
        capture = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = self._own_decoded_rgb(frame, cv2)
                digest.update(memoryview(rgb).cast("B"))
                frames.append(rgb)
        finally:
            capture.release()
        if len(frames) != decode_row["frame_count"] or digest.hexdigest() != decode_row["decoded_rgb_sha256"]:
            fail("decoded RGB closure differs")
        self._decoded_rows.append(
            {
                "r1b_ordinal": decode_row["r1b_ordinal"],
                "role": decode_row["role"],
                "compressed_sha256": expected_sha256,
                "frame_count": len(frames),
                "shape_hwc": [720, 1280, 3],
                "dtype": "uint8",
                "decoded_rgb_sha256": digest.hexdigest(),
            }
        )
        return frames

    def node(self, frames: Sequence[Any], view: str) -> NodeSketch:
        torch = self.torch
        import torch.nn.functional as functional

        cardinalities = []
        valid_phases = 0
        payload = []
        with torch.inference_mode():
            for frame in frames:
                anns = self.sam.generate(frame)
                if not isinstance(anns, list):
                    fail("SAM automatic generator did not return an annotation list")
                anns = self._normalize_sam_annotations(anns)
                filtered = [
                    ann
                    for ann in anns
                    if 0.001 <= ann["area"] / (frame.shape[0] * frame.shape[1]) <= 0.45
                ]
                filtered = sorted(
                    filtered,
                    key=lambda ann: (-ann["predicted_iou"], -ann["stability_score"], -ann["area"], ann["bbox"]),
                )[:12]
                self.raw.observe("dino_filtered_ann_records", len(filtered))

                processor = self.dino_processor(images=frame, return_tensors="pt")
                processor_items = tuple(processor.items())
                self.raw.observe(
                    "dino_processor_tensor_items", len(processor_items)
                )
                cpu_inputs = self._own_external_batch(
                    tuple(
                        (value, "dino_processor_input")
                        for _, value in processor_items
                    )
                )
                inputs = {}
                try:
                    for (key, _), cpu_value in zip(processor_items, cpu_inputs):
                        inputs[key] = self._owned_tensor_copy(
                            cpu_value,
                            "dino_processor_input",
                            device=self.device,
                        )
                except BaseException:
                    self._release_owned_batch(tuple(inputs.values()))
                    raise
                finally:
                    self._release_owned_batch(cpu_inputs)
                try:
                    output = self.dino(**inputs)
                    output_tensors = self._own_external_tensor_tree(
                        output,
                        "dino_tokens",
                        observed_count_name="dino_model_output_unique_storages",
                    )
                except BaseException:
                    self._release_owned_batch(tuple(inputs.values()))
                    raise
                try:
                    self._release_owned_batch(tuple(inputs.values()))
                except BaseException:
                    self._release_owned_batch(output_tensors)
                    raise
                try:
                    hidden = output.last_hidden_state
                    hidden_storage = RawInventoryV3._storage_key(hidden)
                    if not any(
                        RawInventoryV3._storage_key(item) == hidden_storage
                        for item in output_tensors
                    ):
                        fail("DINO last_hidden_state is outside the owned output tree")
                    tokens = self._own_created(
                        hidden[:, 1:, :].clone(
                            memory_format=torch.contiguous_format
                        ),
                        "dino_tokens",
                    )
                except BaseException:
                    self._release_owned_batch(output_tensors)
                    raise
                side = int(math.isqrt(tokens.shape[1]))
                if side * side != tokens.shape[1]:
                    self._release_owned_batch((tokens, *output_tensors))
                    fail("DINO patch geometry is not square")
                valid_anns = []
                descriptors = []
                try:
                    for ann in filtered:
                        mask_storage = self._owned_tensor_copy(
                            torch.as_tensor(ann["segmentation"]),
                            "dino_mask_input",
                            device=self.device,
                            dtype=torch.float32,
                        )
                        mask_input = mask_storage[None, None]
                        height, width = mask_input.shape[-2:]
                        resized_shape = (256, round(width * 256 / height)) if height <= width else (round(height * 256 / width), 256)
                        resized = self._own_created(
                            functional.interpolate(
                                mask_input, size=resized_shape, mode="nearest"
                            ),
                            "dino_mask_resized",
                        )
                        top = (resized_shape[0] - 224) // 2
                        left = (resized_shape[1] - 224) // 2
                        cropped = self._own_created(
                            resized[
                                :, :, top : top + 224, left : left + 224
                            ].clone(memory_format=torch.contiguous_format),
                            "dino_mask_cropped",
                        )
                        weight_grid = self._own_created(
                            functional.interpolate(
                                cropped, size=(side, side), mode="area"
                            ),
                            "dino_patch_weights",
                        )
                        weights = weight_grid.view(-1)
                        support = self._own_created(
                            weights.sum(), "dino_patch_support"
                        )
                        try:
                            support_value = float(support)
                            if not math.isfinite(support_value) or support_value <= 1e-6:
                                continue
                            self.raw.observe("dino_positive_support_records", 1)
                            weighted_tokens = self._own_created(
                                tokens[0] * weights[:, None],
                                "dino_pooled_descriptor",
                            )
                            try:
                                pooled_sum = self._own_created(
                                    weighted_tokens.sum(0),
                                    "dino_pooled_descriptor",
                                )
                                try:
                                    pooled = self._own_created(
                                        pooled_sum / support,
                                        "dino_pooled_descriptor",
                                    )
                                    try:
                                        cpu_tensor = self._owned_tensor_copy(
                                            pooled[:8],
                                            "dino_pooled_descriptor_cpu",
                                            device="cpu",
                                            dtype=torch.float32,
                                        )
                                        try:
                                            descriptor = self._own_created(
                                                cpu_tensor.numpy().copy(),
                                                "dino_pooled_descriptor_cpu",
                                            )
                                        finally:
                                            self.raw.release(cpu_tensor)
                                    finally:
                                        self.raw.release(pooled)
                                finally:
                                    self.raw.release(pooled_sum)
                            finally:
                                self.raw.release(weighted_tokens)
                            valid_anns.append(ann)
                            descriptors.append(descriptor)
                        finally:
                            self.raw.release(support)
                            self.raw.release(weight_grid)
                            self.raw.release(cropped)
                            self.raw.release(resized)
                            self.raw.release(mask_storage)
                finally:
                    self._release_owned_batch((tokens, *output_tensors))
                retained_ids = {id(ann["segmentation"]) for ann in valid_anns}
                for ann in anns:
                    if id(ann["segmentation"]) not in retained_ids:
                        self.raw.release(ann["segmentation"])
                phase_nodes = []
                for ann, descriptor in zip(valid_anns, descriptors):
                    mask = ann["segmentation"]
                    self.raw.observe("sam_mask_coordinate_calls", 1)
                    # NumPy returns the row/column arrays from ``nonzero`` as
                    # disjoint views of one shared mutable backing array on
                    # the pinned AUH build.  Claim that backing storage once
                    # through the aggregate; claiming both views separately
                    # would violate the inventory's single-owner rule.
                    coordinate_pair = tuple(mask.nonzero())
                    (owned_coordinate_pair,) = self._own_external_batch(
                        ((coordinate_pair, "sam_mask_coordinate_indices"),)
                    )
                    ys, xs = owned_coordinate_pair
                    try:
                        if not len(xs):
                            self.raw.release(mask)
                            self.raw.release(descriptor)
                            continue
                        area_fraction = float(mask.mean())
                        centroid_xy = (
                            float(xs.mean() / max(mask.shape[1] - 1, 1)),
                            float(ys.mean() / max(mask.shape[0] - 1, 1)),
                        )
                    finally:
                        self.raw.release(owned_coordinate_pair)
                    if not bool(self.np.isfinite(descriptor).all()):
                        self.raw.release(mask)
                        self.raw.release(descriptor)
                        continue
                    phase_nodes.append(
                        graph_v3.AnonymousNodeV3(
                            mask=mask,
                            descriptor=descriptor,
                            area_fraction=area_fraction,
                            centroid_xy=centroid_xy,
                        )
                    )
                cardinalities.append(len(phase_nodes))
                valid_phases += bool(phase_nodes)
                payload.append(tuple(phase_nodes))
        tracked = graph_v3.assign_anonymous_tracks(
            payload, max_absent_gap_phases=2, gap_penalty=0.08
        )
        signature = self._own_created(
            self.np.asarray(graph_v3.canonical_node_signature(tracked), dtype=self.np.float64),
            "node_signature",
        )
        diagnostics = graph_v3.unbalanced_matching_diagnostics(payload)
        return NodeSketch(
            signature=signature,
            cardinalities=tuple(cardinalities),
            mechanically_valid_phases=int(valid_phases),
            dustbin_used=True,
            private_payload=tracked,
            unbalanced_phase_pair_count=int(diagnostics["phase_pair_count"]),
            dustbin_unmatched_count=int(diagnostics["unmatched_count"]),
            dustbin_transport_mass=float(diagnostics["dustbin_transport_mass"]),
        )

    def motion(self, frames: Sequence[Any], view: str, nodes: NodeSketch) -> MotionSketch:
        torch = self.torch
        if nodes.private_payload is None or len(nodes.private_payload) != PHASES:
            fail("CoTracker requires eight in-memory anonymous mask phases")
        stacked_video = self._own_created(
            torch.stack(
                [torch.as_tensor(frame).permute(2, 0, 1) for frame in frames]
            ),
            "cotracker_video",
        )
        try:
            cpu_video = self._owned_tensor_copy(
                stacked_video,
                "cotracker_video",
                device="cpu",
                dtype=torch.float32,
            )
        finally:
            self.raw.release(stacked_video)
        try:
            video = self._owned_tensor_copy(
                cpu_video[None], "cotracker_video", device=self.device
            )
        finally:
            self.raw.release(cpu_video)
        try:
            with torch.inference_mode():
                raw_output = self.cotracker(
                    video,
                    grid_size=12,
                    grid_query_frame=0,
                    backward_tracking=True,
                )
                if (
                    not isinstance(raw_output, (tuple, list))
                    or len(raw_output) != 2
                ):
                    if not self.raw._scrub_recursive(raw_output, set()):
                        fail(
                            "malformed CoTracker output could not be fully scrubbed"
                        )
                    fail("CoTracker output must be an exact two-leaf sequence")
                raw_tracks, raw_visible = raw_output
                # Inference tensors reject in-place zeroization after leaving
                # inference mode.  Keep the entire external-output claim,
                # derived-copy, and release lifecycle in the same context.
                tracks, visible = self._own_external_batch(
                    (
                        (raw_tracks, "cotracker_tracks"),
                        (raw_visible, "cotracker_visibility"),
                    )
                )
                try:
                    xy_tensor = self._owned_tensor_copy(
                        tracks[0],
                        "cotracker_coordinates_cpu",
                        device="cpu",
                        dtype=torch.float64,
                    )
                    try:
                        vis_tensor = self._owned_tensor_copy(
                            visible[0],
                            "cotracker_visibility_cpu",
                            device="cpu",
                            dtype=torch.bool,
                        )
                        try:
                            xy = self._own_created(
                                xy_tensor.numpy().copy(),
                                "cotracker_coordinates_cpu",
                            )
                            try:
                                vis = self._own_created(
                                    vis_tensor.numpy().copy(),
                                    "cotracker_visibility_cpu",
                                )
                            except BaseException:
                                self.raw.release(xy)
                                raise
                        finally:
                            self.raw.release(vis_tensor)
                    finally:
                        self.raw.release(xy_tensor)
                finally:
                    self._release_owned_batch((tracks, visible))
        finally:
            self.raw.release(video)

        assignment = graph_v3.assign_points_with_same_track_membership(
            nodes.private_payload, xy, vis, minimum_member_phases=3
        )
        memberships = assignment.memberships
        self.raw.observe("cotracker_membership_rows", len(memberships))
        membership_arrays = self._own_external_batch(
            tuple(
                (value, category)
                for row in memberships
                for value, category in (
                    (row.centers_xy, "cotracker_group_coordinates"),
                    (row.center_valid, "cotracker_group_visibility"),
                    (row.velocities_xy, "cotracker_group_coordinates"),
                    (row.velocity_valid, "cotracker_group_visibility"),
                )
            )
        )
        if len(memberships) > 96:
            fail("assigned worldlines exceed the preregistered mechanical 8x12 maximum")
        descriptor_by_track = {}
        for phase in nodes.private_payload:
            for node in phase:
                descriptor_by_track.setdefault(node.track_id, node.descriptor)
        track_values = graph_v3.canonical_track_signature(
            memberships, descriptor_by_track, maximum_worldlines=96
        )
        track_signature = self._own_created(
            self.np.asarray(track_values, dtype=self.np.float64), "track_signature"
        )
        edge_spec = authority.load_authority()["edge_contract"]
        edge = graph_v3.per_phase_edge_signatures(
            nodes.private_payload,
            memberships,
            overlap_iou_threshold=float(edge_spec["overlap_iou_threshold"]),
            boundary_gap_threshold=float(edge_spec["boundary_gap_threshold"]),
            predictive_gap_threshold=float(edge_spec["predictive_gap_threshold"]),
            converging_speed_threshold=float(edge_spec["converging_speed_threshold"]),
        )
        edge_signature = self._own_created(
            self.np.asarray(edge.signature, dtype=self.np.float64), "edge_signature"
        )
        drop_signature = self._own_created(
            self.np.asarray(edge.dropped_signature, dtype=self.np.float64), "drop_edge_signature"
        )
        assigned_points = sum(len(row.point_indices) for row in memberships)
        visible_members = sum(sum(row.phase_member_counts) for row in memberships)
        denominator = PHASES * assigned_points
        per_phase_members = tuple(
            sum(row.phase_member_counts[phase] for row in memberships)
            for phase in range(PHASES)
        )
        member_floor = min(
            (count for row in memberships for count in row.member_phase_counts), default=0
        )
        state_counts = {name: 0 for name in ("ABSENT", "VISIBLE_MEMBER", "OCCLUDED", "VISIBLE_OUTSIDE_MASK")}
        lifecycle_counts = {name: 0 for name in ("entry", "occlusion", "membership_loss", "reentry", "death")}
        for row in memberships:
            for state in row.phase_states:
                state_counts[state] += 1
            for name, count in row.lifecycle.items():
                lifecycle_counts[name] += int(count)
        diagnostics = {
            "ambiguous_overlap_observation_count": assignment.ambiguous_overlap_observation_count,
            "out_of_bounds_observation_count": assignment.out_of_bounds_observation_count,
            "nonfinite_observation_count": assignment.nonfinite_observation_count,
            "vote_tie_abstain_count": assignment.vote_tie_abstain_count,
            "insufficient_membership_abstain_count": assignment.insufficient_membership_abstain_count,
        }
        result = MotionSketch(
            track_signature=track_signature,
            edge_signature=edge_signature,
            drop_edge_signature=drop_signature,
            assigned_track_count=len(memberships),
            assigned_point_count=assigned_points,
            minimum_same_track_member_phases_observed=member_floor,
            visible_and_member_fraction=(
                visible_members / denominator if denominator else None
            ),
            per_phase_visible_member_counts=per_phase_members,
            assignment_diagnostics=diagnostics,
            state_counts=state_counts,
            lifecycle_counts=lifecycle_counts,
            valid_adjacent_velocity_count=sum(int(row.velocity_valid.sum()) for row in memberships),
            per_phase_active_counts=edge.per_phase_active_counts,
            per_phase_birth_counts=edge.per_phase_birth_counts,
            per_phase_persist_counts=edge.per_phase_persist_counts,
            per_phase_death_counts=edge.per_phase_death_counts,
            per_phase_valid_velocity_counts=edge.per_phase_valid_velocity_counts,
            per_phase_qualified_lifecycle_counts=edge.per_phase_qualified_lifecycle_counts,
            evaluated_pairwise_edge_count=sum(edge.per_phase_active_counts),
            drop_edge_removed_count=edge.removed_edge_count,
        )
        self._release_owned_batch((xy, vis, *membership_arrays))
        return result

    def phase(self, frames: Sequence[Any], view: str) -> PhaseSketch:
        torch = self.torch
        if len(frames) != 16 or int(self.vjepa.config.tubelet_size) != 2:
            fail("V-JEPA requires sixteen frames and tubelet_size=2")
        processor = self.vjepa_processor(videos=[list(frames)], return_tensors="pt")
        processor_items = tuple(processor.items())
        self.raw.observe("vjepa_processor_tensor_items", len(processor_items))
        cpu_inputs = self._own_external_batch(
            tuple(
                (value, "vjepa_processor_input")
                for _, value in processor_items
            )
        )
        inputs = {}
        try:
            for (key, _), cpu_value in zip(processor_items, cpu_inputs):
                inputs[key] = self._owned_tensor_copy(
                    cpu_value,
                    "vjepa_processor_input",
                    device=self.device,
                )
        except BaseException:
            self._release_owned_batch(tuple(inputs.values()))
            raise
        finally:
            self._release_owned_batch(cpu_inputs)
        with torch.inference_mode():
            try:
                output = self.vjepa(**inputs)
                # Claim every returned storage before any other cleanup can
                # raise and orphan an unregistered inference tensor.
                output_tensors = self._own_external_tensor_tree(
                    output,
                    "vjepa_hidden",
                    observed_count_name="vjepa_model_output_unique_storages",
                )
            except BaseException:
                self._release_owned_batch(tuple(inputs.values()))
                raise
            try:
                self._release_owned_batch(tuple(inputs.values()))
            except BaseException:
                self._release_owned_batch(output_tensors)
                raise
            # As for CoTracker, every V-JEPA model output is an inference
            # tensor.  Own, copy, and scrub all four unique output storages
            # before leaving the inference context.
            try:
                full_hidden = output.last_hidden_state
                hidden_storage = RawInventoryV3._storage_key(full_hidden)
                if not any(
                    RawInventoryV3._storage_key(item) == hidden_storage
                    for item in output_tensors
                ):
                    fail("V-JEPA last_hidden_state is outside the owned output tree")
                hidden = self._owned_tensor_copy(
                    full_hidden[0], "vjepa_hidden", dtype=torch.float32
                )
            except BaseException:
                self._release_owned_batch(output_tensors)
                raise
            try:
                spatial = (
                    int(self.vjepa.config.image_size)
                    // int(self.vjepa.config.patch_size)
                ) ** 2
                if hidden.shape[0] != PHASES * spatial:
                    fail(
                        "V-JEPA output does not contain exactly eight real tubelet2 blocks"
                    )
                block_means = self._own_created(
                    hidden.view(PHASES, spatial, hidden.shape[-1]).mean(1),
                    "vjepa_hidden",
                )
                try:
                    cpu_signature = self._owned_tensor_copy(
                        block_means[:, :16],
                        "vjepa_phase_signature",
                        device="cpu",
                        dtype=torch.float32,
                    )
                    try:
                        signature = self._own_created(
                            cpu_signature.view(-1).numpy().copy(),
                            "vjepa_phase_signature",
                        )
                    finally:
                        self.raw.release(cpu_signature)
                finally:
                    self.raw.release(block_means)
            finally:
                self._release_owned_batch((hidden, *output_tensors))
        return PhaseSketch(signature=signature)

    def frozen_receipt(self) -> Mapping[str, Any]:
        current = self._version_pointer_state()
        return {
            "all_models_eval_frozen": all(
                not model.training and all(not parameter.requires_grad for parameter in model.parameters())
                for model in self.models.values()
            ),
            "source_and_weight_closure_unchanged": current == self._versions,
            "parameter_updates": 0,
            "generator_forward_calls": 0,
        }

    def scrub_case(self) -> None:
        # Exception cleanup may receive registered inference tensors whose
        # local success-path finally was interrupted.  Re-enter inference
        # mode so best-effort zeroization remains legal and exhaustive.
        with self.torch.inference_mode():
            self.raw.scrub_all()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def finalize_model_closure(self) -> Mapping[str, Any]:
        after = self._full_model_state()
        value = {
            "mode": "real_frozen_full_tensor_closure",
            "verified": self._model_before == after and self._version_pointer_state() == self._versions,
            "before": self._model_before,
            "after": after,
            "exact_before_after_equality": self._model_before == after,
            "binding": self._model_binding,
        }
        return {**value, "digest": authority.object_sha256(value)}

    def raw_ownership_receipt(self) -> Mapping[str, Any]:
        return self.raw.receipt(require_all_categories=True)

    def hydra_config_receipt(self) -> Mapping[str, Any]:
        return self._hydra_receipt

    def asset_closure_receipt(self) -> Mapping[str, Any]:
        return self._asset_receipt

    def device_receipt(self) -> Mapping[str, Any]:
        if self.torch.cuda.device_count() != 1 or self.torch.device(self.device).index != 0:
            fail("exact one visible CUDA/ROCm device binding differs")
        visible = os.environ.get("ROCR_VISIBLE_DEVICES")
        if not isinstance(visible, str) or not visible or "," in visible:
            fail("ROCR_VISIBLE_DEVICES must bind exactly one external GPU token")
        fixed = authority.load_authority()["fixed_paths"]
        run_root = Path(fixed["fresh_formal_run_root"])
        expected_scratch = {
            "MIOPEN_USER_DB_PATH": run_root / fixed["miopen_user_dirname"],
            "MIOPEN_CUSTOM_CACHE_DIR": run_root
            / fixed["miopen_custom_cache_dirname"],
        }
        if any(
            os.environ.get(name) != str(path)
            for name, path in expected_scratch.items()
        ) or "MIOPEN_DISABLE_CACHE" in os.environ:
            fail("MIOpen scratch environment differs or cache was disabled")
        scratch_records = {}
        for name, path in expected_scratch.items():
            try:
                snapshot_v3._plain_directory(path)
            except Exception as error:
                raise RuntimeV3Error(
                    f"MIOpen scratch is not lexical absolute plain: {path}"
                ) from error
            row = path.stat()
            if stat.S_IMODE(row.st_mode) != 0o700:
                fail("MIOpen scratch root must remain mode 0700 during compute")
            scratch_records[name] = {
                "path": str(path),
                "device": row.st_dev,
                "inode": row.st_ino,
                "mode": stat.S_IMODE(row.st_mode),
                "no_symlink_components": True,
            }
        value = {
            "mode": "real_one_device",
            "verified": True,
            "type": "cuda",
            "index": 0,
            "visible_device_count": 1,
            "name": self.torch.cuda.get_device_name(0),
            "rocr_visible_devices": visible,
            "miopen_scratch_binding": {
                "environment": {
                    name: str(path) for name, path in expected_scratch.items()
                },
                "miopen_disable_cache_present": False,
                "directories": scratch_records,
            },
        }
        return {**value, "digest": authority.object_sha256(value)}

    def decoded_media_receipt(self) -> Mapping[str, Any]:
        value = {
            "verified": len(self._decoded_rows) == 8,
            "decode_receipt_file_sha256": authority.file_sha256(authority.base.DECODE_RECEIPT_PATH),
            "decode_receipt_self_sha256": authority.load_decode_receipt()["decode_receipt_self_sha256"],
            "rows": list(self._decoded_rows),
        }
        return {**value, "digest": authority.object_sha256(value)}


def source_closure() -> Mapping[str, Any]:
    method_root = Path(os.path.abspath(__file__)).parent
    manifest = method_root / snapshot_v3.MANIFEST_NAME
    if manifest.exists() or manifest.is_symlink():
        snapshot = snapshot_v3.verify_snapshot(method_root, verify_original=False)
        value = {"mode": "immutable_snapshot_v3", "snapshot": snapshot}
    else:
        development = snapshot_v3.original_source_closure(method_root)
        value = {"mode": "development_tree_preflip", "development_source": development}
    return {**value, "digest": authority.object_sha256(value)}


def launch_contract() -> Mapping[str, Any]:
    fixed = authority.load_authority()["fixed_paths"]
    value = {
        "schema_version": "actual-target-foundation-launch-contract-v3",
        "implementation_status": "V3R4_NUMPY_METRIC_REPAIR_SOURCE_INDEPENDENT_AUDIT_PASS_LAUNCH_AUTHORIZED",
        "real_gpu_launch_authorized": REAL_GPU_LAUNCH_AUTHORIZED,
        "independent_preflip_audit_required": True,
        "source_closure": source_closure(),
        "fixed_paths": fixed,
        "device": "exactly cuda:0 / one externally isolated MI210",
        "candidate": "absolute absent create-only finite scalar/digest mechanical JSON",
        "cache": "fresh fixed absolute directory; complete non-raw CaseEvidenceV3 per case",
        "completion": "external controller seals PASS/REJECTED only after authority-valid real srun; engineering failures produce a non-completion attempt ledger",
        "training_performed": False,
        "generator_loaded": False,
        "representation_admission_hard_false": True,
    }
    return {**value, "digest": authority.object_sha256(value)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-contract", action="store_true")
    mode.add_argument("--run-real", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.print_contract:
        print(json.dumps(launch_contract(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if not REAL_GPU_LAUNCH_AUTHORIZED:
        fail("real V3 GPU launch is not authorized by this immutable source")
    fixed = authority.load_authority()["fixed_paths"]
    if args.output is None or str(args.output) != str(Path(fixed["fresh_formal_run_root"]) / fixed["candidate_filename"]):
        fail("real V3 run requires the exact preregistered fresh candidate path")
    if args.cache_dir is None or str(args.cache_dir) != str(Path(fixed["fresh_formal_run_root"]) / fixed["cache_dirname"]):
        fail("real V3 run requires the exact preregistered fresh cache path")
    run_canary(RealFrozenBackend(), output=args.output, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_HOOK_COUNTS",
    "EXPECTED_LOGICAL_COUNTS",
    "MODEL_OUTPUT_UNIQUE_STORAGE_MULTIPLIERS",
    "MotionSketch",
    "RAW_OBSERVED_COUNT_KEYS",
    "NodeSketch",
    "PhaseSketch",
    "REAL_GPU_LAUNCH_AUTHORIZED",
    "RawInventoryV3",
    "RuntimeV3Error",
    "create_only_bytes",
    "create_only_json",
    "launch_contract",
    "run_canary",
    "source_closure",
]

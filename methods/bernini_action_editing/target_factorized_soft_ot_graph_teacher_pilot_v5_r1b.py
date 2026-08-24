#!/usr/bin/env python3
"""R1b factorized ordered-action + soft object-interaction graph diagnostic.

This is a frozen, representation-only target-video teacher probe.  The phase
trunk is the exact admitted R0 V-JEPA2 ordered-residual descriptor.  The object
trunk independently constructs appearance-free soft motion components,
marginalizes three unbalanced-OT tracking hypotheses, and compares soft node,
interaction-edge, and tracking sets.  The trunks are admitted with branchwise
AND gates; no scalar combination can let the R0 phase trunk compensate for an
invalid or non-discriminative object graph.

No text enters either descriptor, no generator/optimizer is loaded, and no RGB,
hidden state, component coordinate, or descriptor value is serialized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np


SCHEMA_MANIFEST = "bernini-target-factorized-soft-ot-graph-manifest-v5-r1b"
SCHEMA_PREREG = "bernini-target-factorized-soft-ot-graph-prereg-v5-r1b"
SCHEMA_RECEIPT = "bernini-target-factorized-soft-ot-graph-receipt-v5-r1b"
EXPERIMENT_ID = "target_factorized_soft_ot_graph_teacher_pilot_v5_r1b"
MANIFEST_FILE_SHA256 = "d43ff7f7c14b2c25bf949798fb71839f6f0e6325d8829784f2d9eef5a1516929"
MANIFEST_SELF_SHA256 = "231da71f38bdd982a9276b02fb3351200b372563cfdded39fb7f7f6f7de93446"
PREREG_FILE_SHA256 = "27c4e6ba4a3d1817e46cbf36ecd6fd5a2d37577b8431601443afde65db4ac3d4"
PREREG_SELF_SHA256 = "6ba384346befae4cac5eaceef7e22ebec4eb53f99419a2a1f2cf07ffc56f5525"
WRAPPER_SHA256 = "6c3d9d148e96fd6b5d14b703ded5c807baf46bd43f65d9cb09c94a9c10ae50b0"

CATALOG_SHA256 = "0c96e808114154e2d069da6ca698debfb6c9f824e0e780f6f39ec70612207ca8"
R1_BACKBONE_SOURCE_SHA256 = "92e1bb4f80a804935c1d7948dffed40d9a929b671b9274dae67f5b31205c22cf"
R0_PHASE_SOURCE_SHA256 = "fb001d6865f93d4e4f19fb03574f5eb089c21d9b393f632d7c26d85519018aca"
R0_AUDIT_SOURCE_SHA256 = "6cd89a6a7e870b5b433b7c4c4e5ad47f56f77babfa9a559581b18a339ce95aff"
R0_CORRECTED_SOURCE_SHA256 = "c31ac281da10fabda148b9f9614cf98847fd83271a91c922af0822230a9cc3fb"
V4C_SOURCE_SHA256 = "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc"
TEST_SOURCE_SHA256 = "334507e7355225b5fe399f9beac1f16c9ea62e61ddbc60a23b58976e40016fa4"

SEED = 20260823
FRAMES = 64
PHASES = 8
GRID = 16
PATCHES = GRID * GRID
PROJECTED = 144
MOTION_DIM = 16
MIN_COMPONENTS = 0
MAX_COMPONENTS = 6
VIEW_ORDER = (
    "target_forward_reference", "target_forward_eval", "target_reverse",
    "target_deterministic_shuffle", "source_noop",
)
FAMILIES = (
    "contact_transfer", "lifecycle_entry_exit",
    "multi_entity_interaction", "articulated_ordered_motion",
)

# Frozen before any of the fresh16 hidden features are extracted.  The 0.005
# phase epsilon is the exact R0 policy.  Object cutoffs are fixed by the
# independent synthetic/null calibration bank encoded in prereg and tests.
THRESHOLDS = {
    "phase_margin_each_negative_min": 0.005,
    "object_node_similarity_min": 0.46,
    "object_edge_similarity_min": 0.38,
    "object_tracking_similarity_min": 0.42,
    "object_margin_each_input_negative_min": 0.020,
    "slot_permutation_tracking_margin_min": 0.015,
    "slot_permutation_edge_margin_min": 0.015,
    "drop_edge_edge_margin_min": 0.020,
    "mechanical_effective_components_min": 1.20,
    "mechanical_valid_phases_min": 4,
    "development_branch_pass_min": 9,
    "locked_validation_branch_pass_min": 3,
    "all16_forward_above_each_control_min": 12,
    "family_development_pass_min": 2,
    "family_validation_pass_min": 1,
}

SOFT_CONSTANTS = {
    "motion_background_center": "coordinatewise_spatial_median_per_token_time",
    "phase_partition": [4, 4, 4, 4, 4, 4, 4, 3],
    "saliency_mad_multiplier": 1.4826,
    "saliency_epsilon": 1e-6,
    "component_participation_divisor": 40.0,
    "component_abstain_dynamic_mass_max": 1e-6,
    "component_single_dynamic_mass_max": 0.01,
    "component_single_participation_max": 8.0,
    "component_iterations": 12,
    "component_temperature": 0.18,
    "component_spatial_cost": 0.15,
    "component_motion_cost": 0.85,
    "tracking_spatial_hypotheses": [0.0, 0.15, 0.30],
    "tracking_motion_cost": 0.70,
    "tracking_mass_cost": 0.10,
    "tracking_sinkhorn_epsilon": 0.12,
    "tracking_sinkhorn_tau": 0.75,
    "tracking_sinkhorn_iterations": 48,
    "tracking_hypothesis_temperature": 0.08,
    "drop_edge_fraction": 0.50,
}


class PilotError(RuntimeError):
    pass


def fail(message: str) -> NoReturn:
    raise PilotError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PilotError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"dtype": str(array.dtype), "shape": list(array.shape)}))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_authority(path: Path, expected_file_sha256: str) -> Mapping[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        fail("authority path must be absolute canonical non-symlink")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
    if identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        fail("authority file identity/SHA differs")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_json_pairs,
                           parse_constant=lambda token: fail(f"nonfinite JSON: {token}"))
    except PilotError:
        raise
    except Exception as error:
        raise PilotError("authority JSON cannot be parsed") from error
    if type(value) is not dict:
        fail("authority root must be object")
    return value


def verify_self_hash(value: Mapping[str, Any], key: str, expected: str) -> None:
    if value.get(key) != expected:
        fail(f"{key} constant differs")
    payload = dict(value)
    payload.pop(key, None)
    if object_sha256(payload) != expected:
        fail(f"{key} self hash differs")


def bind_source(module: Any, expected_sha: str, label: str) -> Mapping[str, Any]:
    path = Path(module.__file__).resolve(strict=True)
    if path.is_symlink() or file_sha256(path) != expected_sha:
        fail(f"{label} source SHA differs")
    return {"path": str(path), "sha256": expected_sha, "size_bytes": path.stat().st_size}


def bind_file(path: Path, expected_sha: str, label: str) -> Mapping[str, Any]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha:
        fail(f"{label} file SHA differs")
    return {"path": str(path), "sha256": expected_sha, "size_bytes": path.stat().st_size}


def imported_sources() -> tuple[Any, Any, Mapping[str, Any]]:
    r1 = importlib.import_module(
        "methods.bernini_action_editing.target_middle_object_graph_teacher_pilot_v5"
    )
    r0 = importlib.import_module("methods.action_anchor_target_gap_audit.representation_eval")
    audit = importlib.import_module("methods.action_anchor_target_gap_audit.audit")
    corrected = importlib.import_module("methods.action_anchor_target_gap_audit.corrected_eval")
    v4c = importlib.import_module(
        "methods.bernini_action_editing.extract_vjepa2_ordered_contextual_features_v4c"
    )
    closure = {
        "r1_frozen_backbone_and_patch_projection": bind_source(
            r1, R1_BACKBONE_SOURCE_SHA256, "R1 backbone"
        ),
        "r0_exact_ordered_residual": bind_source(r0, R0_PHASE_SOURCE_SHA256, "R0 phase"),
        "r0_import_audit": bind_source(audit, R0_AUDIT_SOURCE_SHA256, "R0 audit dependency"),
        "r0_import_corrected_eval": bind_source(
            corrected, R0_CORRECTED_SOURCE_SHA256, "R0 corrected dependency"
        ),
        "v4c_frozen_model_authority": bind_source(v4c, V4C_SOURCE_SHA256, "v4c extractor"),
    }
    return r1, r0, closure


@dataclass(frozen=True)
class PairRow:
    ordinal: int
    pair_id: str
    uuid: str
    family: str
    split: str
    instruction: str
    source_path: Path
    target_path: Path
    source_media: Mapping[str, Any]
    target_media: Mapping[str, Any]
    sampling_geometry: Mapping[str, Any]


def validate_manifest(value: Mapping[str, Any]) -> list[PairRow]:
    if value.get("schema_version") != SCHEMA_MANIFEST or value.get("experiment_id") != EXPERIMENT_ID:
        fail("manifest schema/experiment differs")
    verify_self_hash(value, "manifest_sha256", MANIFEST_SELF_SHA256)
    if value.get("catalog_source") != {
        "path": "/vast/users/guangyi.chen/dataset/MEV/VideoEditing/action_data_construction/runs/full_v5_20260817T193449Z/final_metadata_annotation_v2/paired_training_candidates.jsonl",
        "sha256": CATALOG_SHA256, "row_count": 3749,
    }:
        fail("catalog source differs")
    authority = value.get("authority")
    expected_authority = {
        "formal_sft_authorized": False, "exploratory_representation_only": True,
        "generator_training_authorized": False, "generator_connection_authorized": False,
        "dataset_materialization_authorized": False, "target_pixels_or_hidden_export_authorized": False,
        "development_parameter_fitting_authorized": False,
        "locked_validation_parameter_fitting_authorized": False,
        "post_feature_algorithm_or_holdout_change_invalidates_all16": True,
    }
    if authority != expected_authority:
        fail("representation-only authority differs")
    selection = value.get("selection")
    if type(selection) is not dict or selection.get("selected_before_any_hidden_or_metric") is not True:
        fail("selection timing differs")
    if selection.get("family_quota") != {family: 4 for family in FAMILIES}:
        fail("family quota differs")
    excluded = selection.get("excluded_prior_uuid")
    if type(excluded) is not list or len(excluded) != 28 or len(set(excluded)) != 28:
        fail("prior UUID exclusion closure differs")
    rows = value.get("pairs")
    if type(rows) is not list or len(rows) != 16:
        fail("manifest requires exact fresh16")
    result: list[PairRow] = []
    seen_uuid: set[str] = set()
    seen_pair: set[str] = set()
    family_split = {family: {"development_report": 0, "locked_validation": 0} for family in FAMILIES}
    for ordinal, row in enumerate(rows):
        if type(row) is not dict or row.get("ordinal") != ordinal:
            fail("pair ordinal differs")
        pair_id, uuid = row.get("pair_id"), row.get("uuid")
        family, split = row.get("interaction_family"), row.get("report_split")
        if (
            type(pair_id) is not str or len(pair_id) != 64 or pair_id in seen_pair
            or type(uuid) is not str or uuid in seen_uuid or uuid in excluded
            or family not in FAMILIES or split not in ("development_report", "locked_validation")
            or row.get("formal_sft_authorized") is not False
            or row.get("qualification_status") != "qwen-visual-accepted-annotation-instruction-pending-human"
            or row.get("catalog_gates") != {
                "appearance_change": False, "source_camera_motion": False,
                "target_camera_motion": False, "verdict": "accept", "confidence": "high",
                "source_enables_target": "no", "target_action_quality": "clear_action",
            }
        ):
            fail(f"pair authority differs at ordinal {ordinal}")
        seen_pair.add(pair_id); seen_uuid.add(uuid); family_split[family][split] += 1
        paths = (row.get("source_video_path"), row.get("target_video_path"))
        if any(type(path) is not str or not path.startswith("/vast/") for path in paths):
            fail("media paths differ")
        sm, tm, geometry = row.get("source_media"), row.get("target_media"), row.get("sampling_geometry")
        for media in (sm, tm):
            if (
                type(media) is not dict or type(media.get("sha256")) is not str
                or len(media["sha256"]) != 64 or type(media.get("size_bytes")) is not int
                or type(media.get("decoded_frames")) is not int
                or type(media.get("decoded_rgb_sha256")) is not str
            ):
                fail("media closure differs")
        if type(geometry) is not dict:
            fail("sampling geometry differs")
        result.append(PairRow(
            ordinal, pair_id, uuid, family, split, row["instruction"], Path(paths[0]),
            Path(paths[1]), sm, tm, geometry,
        ))
    if any(counts != {"development_report": 3, "locked_validation": 1}
           for counts in family_split.values()):
        fail("each family must be exact 3 development + 1 locked validation")
    if sum(row.split == "development_report" for row in result) != 12:
        fail("fresh16 split must be exact12/4")
    return result


def validate_prereg(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_PREREG or value.get("experiment_id") != EXPERIMENT_ID:
        fail("prereg schema/experiment differs")
    verify_self_hash(value, "prereg_sha256", PREREG_SELF_SHA256)
    exact = {
        "fixed_thresholds": THRESHOLDS,
        "soft_object_constants": SOFT_CONSTANTS,
        "view_order": list(VIEW_ORDER),
        "phase_trunk": {
            "implementation_sha256": R0_PHASE_SOURCE_SHA256,
            "definition": "exact_R0_time_centered_spatial_means_plus_directed_residuals_strides_1_2_4",
            "target_margin": "cos(candidate,target_reference)-cos(candidate,source_noop)",
            "branchwise_gate": True,
        },
        "object_trunk": {
            "text_used": False, "absolute_appearance_in_final_representation": False,
            "absolute_layout_in_final_representation": False, "hard_nms_used": False,
            "variable_cardinality_soft_components": True,
            "unbalanced_ot_hypothesis_tracking": True,
            "similarity_components": ["node", "edge", "tracking"],
        },
        "admission": {
            "aggregation": "branchwise_AND_then_family_AND",
            "phase_may_compensate_object": False,
            "development_or_validation_tuning": False,
            "all_five_input_graphs_and_two_counterfactuals_must_be_valid": True,
        },
        "counterfactual_controls": {
            "spatial_slot_permutation": "phasewise_cyclic_center_permutation_breaks_motion_spatial_binding",
            "drop_edge": "remove_top_half_interaction_mass_globally_keep_at_least_one",
        },
    }
    if value.get("runtime_contract") != exact:
        fail("preregistered runtime contract differs")
    bank = value.get("independent_null_calibration_bank")
    if type(bank) is not dict or bank.get("real_fresh16_features_used") is not False:
        fail("null calibration authority differs")


def reference_indices(frame_count: int) -> np.ndarray:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame count differs")
    return np.asarray([(i * (frame_count - 1)) // 63 for i in range(64)], dtype=np.int64)


def eval_indices(frame_count: int) -> np.ndarray:
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame count differs")
    return np.asarray([i * (126 + i) * (frame_count - 1) // (63 * 189)
                       for i in range(64)], dtype=np.int64)


def shuffle_permutation(pair_id: str) -> tuple[int, ...]:
    blocks = sorted(range(8), key=lambda block: (
        hashlib.sha256(f"r1b-target-shuffle:{SEED}:{pair_id}:{block}".encode()).digest(), block
    ))
    if blocks in (list(range(8)), list(reversed(range(8)))):
        blocks = blocks[3:] + blocks[:3]
    if blocks in (list(range(8)), list(reversed(range(8)))):
        raise RuntimeError("forbidden shuffle")
    return tuple(blocks)


def shuffle64_indices(pair_id: str) -> np.ndarray:
    return np.asarray([8 * block + offset for block in shuffle_permutation(pair_id)
                       for offset in range(8)], dtype=np.int64)


def motion_projection() -> np.ndarray:
    matrix = np.empty((PROJECTED, MOTION_DIM), dtype=np.float32)
    for source in range(PROJECTED):
        for target in range(MOTION_DIM):
            bit = hashlib.sha256(
                f"r1b-motion-projection:{SEED}:{source}:{target}".encode("ascii")
            ).digest()[0] & 1
            matrix[source, target] = (1.0 if bit else -1.0) / math.sqrt(PROJECTED)
    return matrix


def grid_coordinates() -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, GRID, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)


def _unit_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    result = np.exp(np.clip(shifted, -80.0, 0.0))
    return result / np.maximum(result.sum(axis=axis, keepdims=True), 1e-30)


@dataclass(frozen=True)
class SoftComponent:
    mass: float
    center: np.ndarray
    motion: np.ndarray
    energy: float
    spread: np.ndarray
    entropy: float


def extract_soft_components(phase_motion: np.ndarray) -> tuple[SoftComponent, ...]:
    value = np.asarray(phase_motion, dtype=np.float64)
    if value.shape != (PATCHES, MOTION_DIM) or not np.isfinite(value).all():
        raise ValueError("phase motion geometry differs")
    energy = np.linalg.norm(value, axis=1)
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median)))
    scale = SOFT_CONSTANTS["saliency_mad_multiplier"] * mad + SOFT_CONSTANTS["saliency_epsilon"]
    z = np.clip((energy - median) / scale, -12.0, 12.0)
    saliency = (1.0 / (1.0 + np.exp(-z))) * energy / np.maximum(energy + median + scale, 1e-12)
    weights = saliency / PATCHES
    participation = float(weights.sum() ** 2 / max(float(np.dot(weights, weights)), 1e-12))
    dynamic_mass = float(weights.sum())
    if dynamic_mass <= SOFT_CONSTANTS["component_abstain_dynamic_mass_max"]:
        count = 0
    elif (dynamic_mass <= SOFT_CONSTANTS["component_single_dynamic_mass_max"]
          or participation <= SOFT_CONSTANTS["component_single_participation_max"]):
        count = 1
    else:
        count = int(np.clip(math.ceil(participation /
                                      SOFT_CONSTANTS["component_participation_divisor"]),
                            2, MAX_COMPONENTS))
    if count == 0:
        return ()
    coords = grid_coordinates().astype(np.float64)
    motion = _unit_rows(value)
    prototypes = np.empty((count, MOTION_DIM), dtype=np.float64)
    for k in range(count):
        for d in range(MOTION_DIM):
            prototypes[k, d] = 1.0 if hashlib.sha256(
                f"r1b-component-prototype:{SEED}:{count}:{k}:{d}".encode()
            ).digest()[0] & 1 else -1.0
    prototypes = _unit_rows(prototypes)
    angles = 2.0 * np.pi * (np.arange(count) + 0.5) / count
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    resp = _softmax(1.25 * coords @ directions.T + 0.75 * motion @ prototypes.T, axis=1)
    for _ in range(SOFT_CONSTANTS["component_iterations"]):
        weighted = weights[:, None] * resp
        masses = np.maximum(weighted.sum(axis=0), 1e-12)
        centers = weighted.T @ coords / masses[:, None]
        features = _unit_rows(weighted.T @ motion / masses[:, None])
        motion_cost = 1.0 - np.clip(motion @ features.T, -1.0, 1.0)
        spatial_cost = ((coords[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2) / 4.0
        cost = (SOFT_CONSTANTS["component_motion_cost"] * motion_cost
                + SOFT_CONSTANTS["component_spatial_cost"] * spatial_cost)
        resp = _softmax(-cost / SOFT_CONSTANTS["component_temperature"]
                        + np.log(masses[None, :]), axis=1)
    weighted = weights[:, None] * resp
    masses = np.maximum(weighted.sum(axis=0), 1e-12)
    components = []
    for k in range(count):
        wk = weighted[:, k]
        center = wk @ coords / masses[k]
        centered = coords - center
        covariance = (centered * wk[:, None]).T @ centered / masses[k]
        spread = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        feature = _unit_rows((wk @ motion).reshape(1, -1))[0]
        component_energy = float(wk @ energy / masses[k])
        probability = wk / masses[k]
        entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-30)))
                        / math.log(PATCHES))
        components.append(SoftComponent(float(masses[k]), center, feature,
                                        component_energy, spread, entropy))
    return tuple(components)


def phase_components(projected: np.ndarray) -> tuple[tuple[SoftComponent, ...], ...]:
    value = np.asarray(projected)
    if value.shape != (32, PATCHES, PROJECTED) or value.dtype != np.float32:
        raise ValueError("projected patch tokens differ")
    delta = value[1:].astype(np.float64) - value[:-1].astype(np.float64)
    delta -= np.median(delta, axis=1, keepdims=True)
    reduced = delta @ motion_projection().astype(np.float64)
    result = []
    cursor = 0
    for width in SOFT_CONSTANTS["phase_partition"]:
        phase = np.median(reduced[cursor:cursor + width], axis=0)
        result.append(extract_soft_components(phase))
        cursor += width
    if cursor != 31:
        raise RuntimeError("phase partition differs")
    return tuple(result)


def unbalanced_sinkhorn(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if cost.shape != (a.size, b.size) or min(a.size, b.size) < 1:
        raise ValueError("OT geometry differs")
    epsilon = SOFT_CONSTANTS["tracking_sinkhorn_epsilon"]
    tau = SOFT_CONSTANTS["tracking_sinkhorn_tau"]
    kernel = np.exp(np.clip(-cost / epsilon, -80.0, 0.0)) + 1e-30
    power = tau / (tau + epsilon)
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(SOFT_CONSTANTS["tracking_sinkhorn_iterations"]):
        u = (a / np.maximum(kernel @ v, 1e-30)) ** power
        v = (b / np.maximum(kernel.T @ u, 1e-30)) ** power
    plan = u[:, None] * kernel * v[None, :]
    transported = float(plan.sum())
    objective = float(np.sum(plan * cost) / max(transported, 1e-30))
    return plan, objective


@dataclass(frozen=True)
class SoftTransition:
    plan: np.ndarray
    objective: float
    hypothesis_entropy: float
    matched_ratio: float


def track_transition(left: Sequence[SoftComponent], right: Sequence[SoftComponent]) -> SoftTransition:
    a = np.asarray([row.mass for row in left], dtype=np.float64)
    b = np.asarray([row.mass for row in right], dtype=np.float64)
    if a.size == 0 or b.size == 0:
        # No phantom slot is created.  The zero transport mechanically means
        # full birth (0->K), full death (K->0), or abstention (0->0).
        plan = np.zeros((a.size, b.size), dtype=np.float64)
        objective = 0.0 if a.size == 0 and b.size == 0 else 1.0
        return SoftTransition(plan, objective, 0.0, 0.0)
    lm = np.stack([row.motion for row in left]); rm = np.stack([row.motion for row in right])
    lc = np.stack([row.center for row in left]); rc = np.stack([row.center for row in right])
    lc -= np.average(lc, axis=0, weights=a); rc -= np.average(rc, axis=0, weights=b)
    motion_cost = 1.0 - np.clip(lm @ rm.T, -1.0, 1.0)
    mass_cost = np.abs(np.log(np.maximum(a[:, None], 1e-12))
                       - np.log(np.maximum(b[None, :], 1e-12)))
    spatial = ((lc[:, None, :] - rc[None, :, :]) ** 2).sum(axis=2) / 4.0
    plans, objectives = [], []
    for spatial_weight in SOFT_CONSTANTS["tracking_spatial_hypotheses"]:
        cost = (SOFT_CONSTANTS["tracking_motion_cost"] * motion_cost
                + SOFT_CONSTANTS["tracking_mass_cost"] * mass_cost
                + spatial_weight * spatial)
        plan, objective = unbalanced_sinkhorn(a, b, cost)
        plans.append(plan); objectives.append(objective)
    posterior = _softmax(-np.asarray(objectives) /
                         SOFT_CONSTANTS["tracking_hypothesis_temperature"])
    plan = sum(float(weight) * candidate for weight, candidate in zip(posterior, plans))
    entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1e-30)))
                    / math.log(len(posterior)))
    matched = float(plan.sum() / max(min(a.sum(), b.sum()), 1e-12))
    return SoftTransition(plan, float(np.dot(posterior, objectives)), entropy,
                          float(np.clip(matched, 0.0, 1.0)))


@dataclass(frozen=True)
class SoftGraph:
    node_features: np.ndarray
    node_masses: np.ndarray
    edge_features: np.ndarray
    edge_masses: np.ndarray
    tracking_features: np.ndarray
    tracking_masses: np.ndarray
    mechanically_valid: bool
    diagnostics: Mapping[str, Any]
    phases: tuple[tuple[SoftComponent, ...], ...]


def _squash(value: float) -> float:
    return float(value / (1.0 + abs(value)))


def assemble_soft_graph(phases: tuple[tuple[SoftComponent, ...], ...]) -> SoftGraph:
    if len(phases) != PHASES or any(not (0 <= len(rows) <= MAX_COMPONENTS)
                                    for rows in phases):
        raise ValueError("phase component cardinality differs")
    effective = []
    for rows in phases:
        masses = np.asarray([row.mass for row in rows], dtype=np.float64)
        effective.append(0.0 if masses.size == 0 else float(
            masses.sum() ** 2 / max(float(np.dot(masses, masses)), 1e-12)))
    valid_phase_count = sum(len(rows) >= 2 and effective[index] >=
                            THRESHOLDS["mechanical_effective_components_min"]
                            for index, rows in enumerate(phases))
    transitions = tuple(track_transition(phases[p], phases[p + 1]) for p in range(PHASES - 1))
    nodes, node_masses, node_lookup = [], [], {}
    for phase, rows in enumerate(phases):
        for index, row in enumerate(rows):
            incoming = row.mass if phase == 0 else float(transitions[phase - 1].plan[:, index].sum())
            outgoing = row.mass if phase == PHASES - 1 else float(transitions[phase].plan[index].sum())
            birth = max(row.mass - incoming, 0.0) / max(row.mass, 1e-12)
            death = max(row.mass - outgoing, 0.0) / max(row.mass, 1e-12)
            anisotropy = float((row.spread[0] - row.spread[1])
                               / max(row.spread.sum(), 1e-12))
            feature = np.concatenate((
                np.asarray([phase / (PHASES - 1), row.mass, _squash(row.energy),
                            _squash(float(row.spread.sum())), anisotropy,
                            birth, death, row.entropy], dtype=np.float64),
                row.motion.astype(np.float64),
            ))
            node_lookup[(phase, index)] = len(nodes)
            nodes.append(feature); node_masses.append(row.mass)
    # Soft per-component displacement from the marginalized OT plan.
    displacement: dict[tuple[int, int], np.ndarray] = {}
    for phase, transition in enumerate(transitions):
        right_centers = (np.stack([row.center for row in phases[phase + 1]])
                         if phases[phase + 1] else np.empty((0, 2), dtype=np.float64))
        for index, row in enumerate(phases[phase]):
            weights = transition.plan[index]
            if weights.size == 0 or float(weights.sum()) <= 0.0:
                displacement[(phase, index)] = np.zeros(2, dtype=np.float64)
            else:
                predicted = weights @ right_centers / float(weights.sum())
                displacement[(phase, index)] = predicted - row.center
    for index, row in enumerate(phases[-1]):
        displacement[(PHASES - 1, index)] = np.zeros(2, dtype=np.float64)
    edge_rows = []
    for phase, rows in enumerate(phases):
        provisional = []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                a, b = rows[left], rows[right]
                relative = b.center - a.center
                distance = float(np.linalg.norm(relative))
                radius = math.sqrt(max(float(a.spread.sum()), 0.0)) + math.sqrt(
                    max(float(b.spread.sum()), 0.0))
                overlap = 1.0 / (1.0 + math.exp(np.clip((distance - radius) / 0.20, -40, 40)))
                motion_cos = float(np.clip(np.dot(a.motion, b.motion), -1.0, 1.0))
                relative_velocity = displacement[(phase, right)] - displacement[(phase, left)]
                convergence = -float(np.dot(relative_velocity, relative)
                                     / max(np.linalg.norm(relative_velocity) * distance, 1e-12))
                mass = math.sqrt(a.mass * b.mass) * (0.25 + 0.75 * overlap) * (
                    0.50 + 0.50 * (1.0 - motion_cos) / 2.0)
                provisional.append((left, right, mass, distance, overlap, motion_cos, convergence))
        degree = np.zeros(len(rows), dtype=np.float64)
        for left, right, mass, *_ in provisional:
            degree[left] += mass; degree[right] += mass
        for left, right, mass, distance, overlap, motion_cos, convergence in provisional:
            a, b = rows[left], rows[right]
            endpoint_a = (a.mass, _squash(a.energy), degree[left])
            endpoint_b = (b.mass, _squash(b.energy), degree[right])
            if endpoint_b < endpoint_a:
                endpoint_a, endpoint_b = endpoint_b, endpoint_a
            edge_rows.append((phase, np.asarray([
                phase / (PHASES - 1), distance / (1.0 + distance), overlap,
                (motion_cos + 1.0) / 2.0, (np.clip(convergence, -1, 1) + 1.0) / 2.0,
                *endpoint_a, *endpoint_b,
            ], dtype=np.float64), mass))
    tracking_features, tracking_masses = [], []
    for phase, transition in enumerate(transitions):
        a = sum(row.mass for row in phases[phase]); b = sum(row.mass for row in phases[phase + 1])
        transported = float(transition.plan.sum())
        support_mass = max(transported, 0.5 * (a + b))
        if support_mass <= 0.0:
            continue
        tracking_features.append(np.asarray([
            phase / (PHASES - 2), transition.matched_ratio,
            _squash(transition.objective), transition.hypothesis_entropy,
            max(b - transported, 0.0) / max(b, 1e-12),
            max(a - transported, 0.0) / max(a, 1e-12),
        ], dtype=np.float64))
        tracking_masses.append(support_mass)
    nodes_array = (np.stack(nodes) if nodes else np.empty((0, 8 + MOTION_DIM), dtype=np.float64))
    node_mass_array = np.asarray(node_masses, dtype=np.float64)
    edge_array = (np.stack([row[1] for row in edge_rows]) if edge_rows
                  else np.empty((0, 11), dtype=np.float64))
    edge_mass_array = np.asarray([row[2] for row in edge_rows])
    tracking_array = (np.stack(tracking_features) if tracking_features
                      else np.empty((0, 6), dtype=np.float64))
    tracking_mass_array = np.asarray(tracking_masses, dtype=np.float64)
    valid = (
        all(np.isfinite(row).all() for row in (nodes_array, node_mass_array, edge_array,
                                                edge_mass_array, tracking_array, tracking_mass_array))
        and valid_phase_count >= THRESHOLDS["mechanical_valid_phases_min"]
        and nodes_array.shape[0] > 0 and edge_array.shape[0] > 0 and tracking_array.shape[0] > 0
        and np.all(node_mass_array > 0) and np.all(edge_mass_array > 0)
        and np.all(tracking_mass_array > 0)
    )
    diagnostics = {
        "phase_component_counts": [len(rows) for rows in phases],
        "phase_effective_component_count": effective,
        "valid_phase_count": valid_phase_count,
        "node_count": len(nodes), "edge_count": len(edge_rows),
        "tracking_transition_count": len(transitions),
        "node_feature_sha256": array_sha256(nodes_array),
        "edge_feature_sha256": array_sha256(edge_array),
        "tracking_feature_sha256": array_sha256(tracking_array),
        "absolute_component_coordinates_exported": False,
    }
    return SoftGraph(nodes_array, node_mass_array, edge_array, edge_mass_array,
                     tracking_array, tracking_mass_array, bool(valid), diagnostics, phases)


def build_soft_graph(projected: np.ndarray) -> SoftGraph:
    return assemble_soft_graph(phase_components(projected))


def spatial_slot_permutation_control(graph: SoftGraph) -> SoftGraph:
    if not graph.mechanically_valid:
        return replace(graph, diagnostics={**graph.diagnostics,
            "counterfactual": "spatial_slot_permutation", "control_applicable": False})
    phases = []
    changed = False
    for phase, rows in enumerate(graph.phases):
        if phase == 0:
            phases.append(rows); continue
        count = len(rows); shift = 1 + (phase % (count - 1)) if count > 2 else 1
        centers = [rows[(index + shift) % count].center.copy() for index in range(count)]
        phases.append(tuple(replace(row, center=centers[index]) for index, row in enumerate(rows)))
        changed = changed or any(not np.array_equal(rows[index].center, centers[index])
                                 for index in range(count))
    if not changed:
        raise RuntimeError("spatial slot permutation must change motion-spatial binding")
    return assemble_soft_graph(tuple(phases))


def drop_edge_control(graph: SoftGraph) -> SoftGraph:
    if not graph.mechanically_valid:
        return replace(graph, diagnostics={**graph.diagnostics,
            "counterfactual": "drop_edge", "control_applicable": False})
    count = graph.edge_masses.size
    keep_count = max(1, count - int(math.ceil(count * SOFT_CONSTANTS["drop_edge_fraction"])))
    order = np.argsort(graph.edge_masses, kind="stable")
    keep = np.sort(order[:keep_count])
    result = replace(graph, edge_features=graph.edge_features[keep].copy(),
                     edge_masses=graph.edge_masses[keep].copy(),
                     diagnostics={**graph.diagnostics, "drop_edge_removed": int(count - keep_count),
                                  "edge_count": int(keep_count)})
    return replace(result, mechanically_valid=bool(
        graph.mechanically_valid and keep_count >= 1 and np.all(result.edge_masses > 0)
    ))


NODE_SCALES = np.asarray([1, .25, .5, .7, 1, 1, 1, 1] + [1] * MOTION_DIM, dtype=np.float64)
EDGE_SCALES = np.asarray([1, .5, 1, 1, 1, .25, .5, .5, .25, .5, .5], dtype=np.float64)
TRACK_SCALES = np.asarray([1, 1, .5, 1, 1, 1], dtype=np.float64)
NULL_FIXED_POINT_SCALE = 1_000_000_000


def set_ot_similarity(left: np.ndarray, left_mass: np.ndarray, right: np.ndarray,
                      right_mass: np.ndarray, scales: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64); right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or left.shape[1] != scales.size:
        raise ValueError("set feature geometry differs")
    cost = np.mean(np.minimum(((left[:, None, :] - right[None, :, :])
                               / scales[None, None, :]) ** 2, 4.0), axis=2)
    plan, objective = unbalanced_sinkhorn(left_mass, right_mass, cost)
    ratio = float(plan.sum() / max(min(float(left_mass.sum()), float(right_mass.sum())), 1e-12))
    return float(np.clip(ratio, 0.0, 1.0) * math.exp(-objective))


def graph_similarity(left: SoftGraph, right: SoftGraph) -> Mapping[str, float]:
    if not left.mechanically_valid or not right.mechanically_valid:
        return {"node": 0.0, "edge": 0.0, "tracking": 0.0, "lexicographic_min": 0.0}
    result = {
        "node": set_ot_similarity(left.node_features, left.node_masses,
                                  right.node_features, right.node_masses, NODE_SCALES),
        "edge": set_ot_similarity(left.edge_features, left.edge_masses,
                                  right.edge_features, right.edge_masses, EDGE_SCALES),
        "tracking": set_ot_similarity(left.tracking_features, left.tracking_masses,
                                      right.tracking_features, right.tracking_masses, TRACK_SCALES),
    }
    result["lexicographic_min"] = min(result.values())
    return result


def _hash_uniform(label: str, shape: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(shape)); values = []
    counter = 0
    while len(values) < size:
        block = hashlib.sha256(f"r1b-null-bank:{SEED}:{label}:{counter}".encode()).digest()
        values.extend((byte + 0.5) / 256.0 for byte in block)
        counter += 1
    return np.asarray(values[:size], dtype=np.float64).reshape(shape)


def independent_null_calibration_bank(count: int = 64) -> Mapping[str, Any]:
    """Pure synthetic calibration; it never reads a video or model feature."""
    if count != 64:
        raise ValueError("calibration bank count is frozen at64")
    positive = {name: [] for name in ("node", "edge", "tracking")}
    null = {name: [] for name in positive}
    margin = {name: [] for name in positive}
    drop_edge_margins = []
    slot_edge_margins = []
    slot_tracking_margins = []
    for index in range(count):
        node = _hash_uniform(f"node-base-{index}", (5, NODE_SCALES.size))
        edge = _hash_uniform(f"edge-base-{index}", (8, EDGE_SCALES.size))
        track = _hash_uniform(f"track-base-{index}", (7, TRACK_SCALES.size))
        node_mass = 0.05 + _hash_uniform(f"node-mass-{index}", (5,))
        edge_mass = 0.05 + _hash_uniform(f"edge-mass-{index}", (8,))
        track_mass = 0.05 + _hash_uniform(f"track-mass-{index}", (7,))
        perturb = lambda label, shape: 0.035 * (_hash_uniform(label, shape) - 0.5)
        candidate = (
            node + perturb(f"node-pos-{index}", node.shape),
            edge + perturb(f"edge-pos-{index}", edge.shape),
            track + perturb(f"track-pos-{index}", track.shape),
        )
        negative = (
            _hash_uniform(f"node-null-{index}", node.shape),
            _hash_uniform(f"edge-null-{index}", edge.shape),
            _hash_uniform(f"track-null-{index}", track.shape),
        )
        for name, base, cand, neg, masses, scales in zip(
            ("node", "edge", "tracking"), (node, edge, track), candidate, negative,
            (node_mass, edge_mass, track_mass), (NODE_SCALES, EDGE_SCALES, TRACK_SCALES),
        ):
            pos = set_ot_similarity(base, masses, cand, masses, scales)
            nul = set_ot_similarity(base, masses, neg, masses, scales)
            positive[name].append(pos); null[name].append(nul); margin[name].append(pos - nul)
        keep = np.argsort(edge_mass, kind="stable")[:4]
        dropped = set_ot_similarity(edge, edge_mass, edge[keep], edge_mass[keep], EDGE_SCALES)
        drop_edge_margins.append(positive["edge"][-1] - dropped)
        slot_edge = edge.copy(); slot_edge[:, 1:5] = np.roll(slot_edge[:, 1:5], 1, axis=0)
        slot_track = track.copy(); slot_track[:, 1:] = np.roll(slot_track[:, 1:], 2, axis=0)
        slot_edge_margins.append(positive["edge"][-1] - set_ot_similarity(
            edge, edge_mass, slot_edge, edge_mass, EDGE_SCALES))
        slot_tracking_margins.append(positive["tracking"][-1] - set_ot_similarity(
            track, track_mass, slot_track, track_mass, TRACK_SCALES))
    # Serialize only fixed-point integers.  The OT calculation is float64, but
    # different supported NumPy/libm builds may disagree in the last bit.  At
    # n=64 the frozen linear p05/p95 interpolation has the exact ranks below;
    # quantizing at 1e-9 makes the calibration authority byte-identical without
    # moving any threshold at a scientifically meaningful precision.
    def quantile_q1e9(rows: Sequence[float], percentile: int) -> int:
        ordered = sorted(float(value) for value in rows)
        if len(ordered) != 64 or percentile not in (5, 95):
            raise ValueError("null quantile geometry differs")
        if percentile == 5:       # (64 - 1) * .05 = 3.15
            value = (85.0 * ordered[3] + 15.0 * ordered[4]) / 100.0
        else:                     # (64 - 1) * .95 = 59.85
            value = (15.0 * ordered[59] + 85.0 * ordered[60]) / 100.0
        return int(math.floor(value * NULL_FIXED_POINT_SCALE + 0.5))

    summary = {
        "count": count, "seed": SEED, "real_video_or_model_features_used": False,
        "fixed_point_scale": NULL_FIXED_POINT_SCALE,
        "quantile_rule": "n64_linear_p05_ranks3_4_weights85_15_p95_ranks59_60_weights15_85",
        "positive_similarity_p05_q1e9": {
            name: quantile_q1e9(rows, 5) for name, rows in positive.items()},
        "null_similarity_p95_q1e9": {
            name: quantile_q1e9(rows, 95) for name, rows in null.items()},
        "positive_minus_null_margin_p05_q1e9": {
            name: quantile_q1e9(rows, 5) for name, rows in margin.items()},
        "drop_edge_margin_p05_q1e9": quantile_q1e9(drop_edge_margins, 5),
        "slot_edge_margin_p05_q1e9": quantile_q1e9(slot_edge_margins, 5),
        "slot_tracking_margin_p05_q1e9": quantile_q1e9(slot_tracking_margins, 5),
    }
    summary["summary_sha256"] = object_sha256(summary)
    return summary


def phase_descriptor(hidden: np.ndarray, r0: Any) -> np.ndarray:
    value = np.asarray(hidden)
    if value.shape != (3, 32, 256, 1024) or value.dtype != np.float32:
        raise ValueError("hidden geometry differs")
    spatial_means = value[2].mean(axis=1, dtype=np.float32)
    return r0.ordered_residual_descriptor(spatial_means)


def phase_scores(descriptors: Mapping[str, np.ndarray], r0: Any) -> Mapping[str, float]:
    target = descriptors["target_forward_reference"]
    source = descriptors["source_noop"]
    return {name: r0.cosine(descriptor, target) - r0.cosine(descriptor, source)
            for name, descriptor in descriptors.items()}


def evaluate_pair(phase: Mapping[str, float], graphs: Mapping[str, SoftGraph],
                  slot_control: SoftGraph, edge_control: SoftGraph) -> Mapping[str, Any]:
    valid = all(graph.mechanically_valid for graph in (*graphs.values(), slot_control, edge_control))
    reference = graphs["target_forward_reference"]
    object_scores = {name: graph_similarity(reference, graph) for name, graph in graphs.items()}
    slot_score = graph_similarity(reference, slot_control)
    edge_score = graph_similarity(reference, edge_control)
    negatives = ("target_reverse", "target_deterministic_shuffle", "source_noop")
    phase_margins = {name: phase["target_forward_eval"] - phase[name] for name in negatives}
    phase_pass = all(value >= THRESHOLDS["phase_margin_each_negative_min"]
                     for value in phase_margins.values())
    positive = object_scores["target_forward_eval"]
    object_floor = (
        positive["node"] >= THRESHOLDS["object_node_similarity_min"]
        and positive["edge"] >= THRESHOLDS["object_edge_similarity_min"]
        and positive["tracking"] >= THRESHOLDS["object_tracking_similarity_min"]
    )
    input_margins = {name: positive["lexicographic_min"] - object_scores[name]["lexicographic_min"]
                     for name in negatives}
    input_pass = all(value >= THRESHOLDS["object_margin_each_input_negative_min"]
                     for value in input_margins.values())
    counterfactual_margins = {
        "slot_permutation_tracking": positive["tracking"] - slot_score["tracking"],
        "slot_permutation_edge": positive["edge"] - slot_score["edge"],
        "drop_edge_edge": positive["edge"] - edge_score["edge"],
    }
    counterfactual_pass = (
        counterfactual_margins["slot_permutation_tracking"] >= THRESHOLDS["slot_permutation_tracking_margin_min"]
        and counterfactual_margins["slot_permutation_edge"] >= THRESHOLDS["slot_permutation_edge_margin_min"]
        and counterfactual_margins["drop_edge_edge"] >= THRESHOLDS["drop_edge_edge_margin_min"]
    )
    object_pass = bool(valid and object_floor and input_pass and counterfactual_pass)
    return {
        "all_seven_graphs_mechanically_valid": bool(valid),
        "phase_trunk": {"scores": dict(phase), "margins": phase_margins, "pass": bool(phase_pass)},
        "object_trunk": {
            "input_scores": object_scores, "input_margins": input_margins,
            "spatial_slot_permutation_score": slot_score,
            "drop_edge_score": edge_score,
            "counterfactual_margins": counterfactual_margins,
            "positive_floor_pass": bool(object_floor), "input_control_pass": bool(input_pass),
            "counterfactual_control_pass": bool(counterfactual_pass), "pass": object_pass,
        },
        "pair_pass": bool(phase_pass and object_pass),
        "aggregation": "phase_pass AND object_pass; no compensation",
    }


def validate_media(path: Path, media: Mapping[str, Any]) -> None:
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or path.stat().st_size != media["size_bytes"] or file_sha256(path) != media["sha256"]):
        fail("media authority differs")


def pair_rgb_views(pair: PairRow, r1: Any) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    source, source_receipt = r1.decode_rgb(pair.source_path)
    target, target_receipt = r1.decode_rgb(pair.target_path)
    for observed, expected, name in ((source, pair.source_media, "source"),
                                     (target, pair.target_media, "target")):
        if (observed.shape[0] != expected["decoded_frames"]
                or array_sha256(observed) != expected["decoded_rgb_sha256"]):
            fail(f"{name} decoded RGB closure differs")
    si = reference_indices(int(source.shape[0])); ri = reference_indices(int(target.shape[0])); ei = eval_indices(int(target.shape[0]))
    geometry = {
        "source_indices_sha256": array_sha256(si),
        "source_unique_indices": len(set(si.tolist())),
        "target_reference_indices_sha256": array_sha256(ri),
        "target_reference_unique_indices": len(set(ri.tolist())),
        "target_eval_indices_sha256": array_sha256(ei),
        "target_eval_unique_indices": len(set(ei.tolist())),
    }
    if geometry != pair.sampling_geometry or geometry["source_unique_indices"] < 16 or min(
        geometry["target_reference_unique_indices"], geometry["target_eval_unique_indices"]
    ) < 48 or np.array_equal(ri, ei):
        fail("pre-registered sampling geometry differs")
    shuffle = shuffle64_indices(pair.pair_id)
    views = {
        "target_forward_reference": np.ascontiguousarray(target[ri]),
        "target_forward_eval": np.ascontiguousarray(target[ei]),
        "target_reverse": np.ascontiguousarray(target[ri][::-1]),
        "target_deterministic_shuffle": np.ascontiguousarray(target[ri][shuffle]),
        "source_noop": np.ascontiguousarray(source[si]),
    }
    digests = {name: array_sha256(value) for name, value in views.items()}
    if tuple(views) != VIEW_ORDER or len(set(digests.values())) != 5:
        fail("exact five distinct RGB views required")
    return views, {
        "source_decode": source_receipt, "target_decode": target_receipt,
        "sampling_geometry": geometry, "view_rgb_sha256": digests,
        "shuffle_permutation": list(shuffle_permutation(pair.pair_id)),
        "shuffle_indices_sha256": array_sha256(shuffle),
    }


def aggregate(rows: Sequence[Mapping[str, Any]], pairs: Sequence[PairRow]) -> Mapping[str, Any]:
    by_id = {row["pair_id"]: row for row in rows}
    development = [by_id[p.pair_id] for p in pairs if p.split == "development_report"]
    validation = [by_id[p.pair_id] for p in pairs if p.split == "locked_validation"]
    summary = {
        "development_pair_pass": sum(row["metrics"]["pair_pass"] for row in development),
        "locked_validation_pair_pass": sum(row["metrics"]["pair_pass"] for row in validation),
        "phase_pair_pass": sum(row["metrics"]["phase_trunk"]["pass"] for row in rows),
        "object_pair_pass": sum(row["metrics"]["object_trunk"]["pass"] for row in rows),
        "all_seven_graphs_valid": sum(row["metrics"]["all_seven_graphs_mechanically_valid"] for row in rows),
    }
    negative_names = ("target_reverse", "target_deterministic_shuffle", "source_noop")
    phase_control_counts = {name: sum(
        row["metrics"]["phase_trunk"]["margins"][name]
        >= THRESHOLDS["phase_margin_each_negative_min"] for row in rows
    ) for name in negative_names}
    object_control_counts = {name: sum(
        row["metrics"]["object_trunk"]["input_margins"][name]
        >= THRESHOLDS["object_margin_each_input_negative_min"] for row in rows
    ) for name in negative_names}
    family = {}
    for name in FAMILIES:
        dev = [by_id[p.pair_id] for p in pairs if p.family == name and p.split == "development_report"]
        val = [by_id[p.pair_id] for p in pairs if p.family == name and p.split == "locked_validation"]
        family[name] = {
            "development_pass": sum(row["metrics"]["pair_pass"] for row in dev),
            "development_total": 3,
            "locked_validation_pass": sum(row["metrics"]["pair_pass"] for row in val),
            "locked_validation_total": 1,
        }
    admitted = (
        summary["development_pair_pass"] >= THRESHOLDS["development_branch_pass_min"]
        and summary["locked_validation_pair_pass"] >= THRESHOLDS["locked_validation_branch_pass_min"]
        and all(row["development_pass"] >= THRESHOLDS["family_development_pass_min"]
                and row["locked_validation_pass"] >= THRESHOLDS["family_validation_pass_min"]
                for row in family.values())
        and all(count >= THRESHOLDS["all16_forward_above_each_control_min"]
                for count in phase_control_counts.values())
        and all(count >= THRESHOLDS["all16_forward_above_each_control_min"]
                for count in object_control_counts.values())
    )
    summary["phase_forward_above_each_control"] = phase_control_counts
    summary["object_forward_above_each_control"] = object_control_counts
    summary["effective_locked_validation_required_due_to_each_family_gate"] = 4
    summary["nominal_total_locked_validation_threshold"] = THRESHOLDS[
        "locked_validation_branch_pass_min"]
    return {"summary": summary, "family": family, "admitted": bool(admitted),
            "aggregation": "phase branch AND object branch AND all four family gates; effective locked validation requirement is 4/4"}


def run(args: argparse.Namespace) -> int:
    started = time.time()
    manifest = load_authority(Path(args.manifest), MANIFEST_FILE_SHA256)
    prereg = load_authority(Path(args.prereg), PREREG_FILE_SHA256)
    pairs = validate_manifest(manifest); validate_prereg(prereg)
    r1, r0, source_closure_before = imported_sources()
    test_binding = bind_file(
        Path(__file__).resolve(strict=True).parent / "tests/test_target_factorized_soft_ot_graph_teacher_pilot_v5_r1b.py",
        TEST_SOURCE_SHA256, "R1b tests",
    )
    wrapper = Path(args.wrapper).resolve(strict=True)
    if file_sha256(wrapper) != WRAPPER_SHA256:
        fail("launch wrapper SHA differs")
    for pair in pairs:
        validate_media(pair.source_path, pair.source_media)
        validate_media(pair.target_path, pair.target_media)
    backbone = r1.FrozenPatchBackbone(Path(args.model_root), args.device)
    rows = []
    for pair in pairs:
        views, inputs = pair_rgb_views(pair, r1)
        graphs, phases, view_receipts, processor_digests = {}, {}, {}, {}
        before_forward, before_processor = backbone.forward_calls, backbone.processor_calls
        for name, frames in views.items():
            pixels, processor_sha = backbone.process(frames)
            hidden = backbone.forward(pixels)
            projected = r1.project_hidden_layers(hidden)
            graph = build_soft_graph(projected)
            phases[name] = phase_descriptor(hidden, r0)
            graphs[name] = graph; processor_digests[name] = processor_sha
            view_receipts[name] = {
                "processor_input_sha256": processor_sha,
                "selected_hidden_shape": list(hidden.shape),
                "selected_hidden_sha256": array_sha256(hidden),
                "phase_descriptor_sha256": array_sha256(phases[name]),
                "object_graph_mechanically_valid": graph.mechanically_valid,
                "object_graph_diagnostics": graph.diagnostics,
                "rgb_hidden_component_coordinates_or_descriptors_exported": False,
            }
            hidden.fill(0); projected.fill(0); del hidden, projected, pixels
        if len(set(processor_digests.values())) != 5:
            fail("exact five processor inputs must differ")
        if backbone.forward_calls - before_forward != 5 or backbone.processor_calls - before_processor != 5:
            fail("pair exact-five forward closure differs")
        slot_control = spatial_slot_permutation_control(graphs["target_forward_reference"])
        edge_control = drop_edge_control(graphs["target_forward_reference"])
        metrics = evaluate_pair(phase_scores(phases, r0), graphs, slot_control, edge_control)
        rows.append({
            "ordinal": pair.ordinal, "pair_id": pair.pair_id, "uuid": pair.uuid,
            "interaction_family": pair.family, "report_split": pair.split,
            "inputs": {**inputs, "processor_input_sha256": processor_digests,
                       "processor_calls": 5, "model_forward_calls": 5},
            "views": view_receipts,
            "counterfactuals": {
                "spatial_slot_permutation": slot_control.diagnostics,
                "drop_edge": edge_control.diagnostics,
            },
            "metrics": metrics,
        })
    if backbone.forward_calls != 80 or backbone.processor_calls != 80:
        fail("fresh16 requires exact80 model/processor forwards")
    aggregate_result = aggregate(rows, pairs)
    _, _, source_closure_after = imported_sources()
    if source_closure_after != source_closure_before:
        fail("source closure changed")
    receipt = {
        "schema_version": SCHEMA_RECEIPT, "experiment_id": EXPERIMENT_ID,
        "status": "R1B_FACTOR_BRANCHWISE_ADMITTED" if aggregate_result["admitted"] else "R1B_FACTOR_BRANCHWISE_REJECTED",
        "factorized_representation_admitted": aggregate_result["admitted"],
        "stable_transferable_action_representation_established": False,
        "self_generated_action_signal_established": False,
        "generator_quality_preserved_established": False,
        "generator_connection_authorized": False,
        "manifest": {"path": args.manifest, "file_sha256": MANIFEST_FILE_SHA256,
                     "self_sha256": MANIFEST_SELF_SHA256},
        "prereg": {"path": args.prereg, "file_sha256": PREREG_FILE_SHA256,
                  "self_sha256": PREREG_SELF_SHA256},
        "implementation": {
            "runner": {"path": str(Path(__file__).resolve(strict=True)),
                       "sha256": file_sha256(Path(__file__).resolve(strict=True))},
            "wrapper": {"path": str(wrapper), "sha256": WRAPPER_SHA256},
            "source_closure": source_closure_before,
            "tests": test_binding,
            "v4c_source_sha256": V4C_SOURCE_SHA256,
        },
        "fixed_thresholds": THRESHOLDS, "aggregate": aggregate_result,
        "optimization_boundary": {
            "optimizer_created": False, "loss_backward_calls": 0, "parameter_updates": 0,
            "development_parameter_fitting": False, "locked_validation_parameter_fitting": False,
            "threshold_selection_from_fresh16": False, "generator_loaded": False,
            "generator_forward_calls": 0, "dataset_written": False,
        },
        "backbone_final_closure": backbone.final_closure(),
        "processor_calls": backbone.processor_calls, "model_forward_calls": backbone.forward_calls,
        "pairs": rows, "elapsed_seconds": time.time() - started,
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    output = Path(args.output)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be fresh absolute non-symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.chmod(output, 0o444)
    print(json.dumps({"status": receipt["status"], "receipt": str(output),
                      "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


def contract(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = load_authority(Path(args.manifest), MANIFEST_FILE_SHA256)
    prereg = load_authority(Path(args.prereg), PREREG_FILE_SHA256)
    pairs = validate_manifest(manifest); validate_prereg(prereg)
    r1, _, closure = imported_sources()
    test_binding = bind_file(
        Path(__file__).resolve(strict=True).parent / "tests/test_target_factorized_soft_ot_graph_teacher_pilot_v5_r1b.py",
        TEST_SOURCE_SHA256, "R1b tests",
    )
    result = {
        "experiment_id": EXPERIMENT_ID, "manifest_file_sha256": MANIFEST_FILE_SHA256,
        "prereg_file_sha256": PREREG_FILE_SHA256,
        "runner_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "wrapper_expected_sha256": WRAPPER_SHA256,
        "source_closure": closure, "pair_count": len(pairs),
        "tests": test_binding,
        "development_count": sum(p.split == "development_report" for p in pairs),
        "locked_validation_count": sum(p.split == "locked_validation" for p in pairs),
        "family_split": {family: {
            "development_report": sum(p.family == family and p.split == "development_report" for p in pairs),
            "locked_validation": sum(p.family == family and p.split == "locked_validation" for p in pairs),
        } for family in FAMILIES},
        "motion_projection_sha256": array_sha256(motion_projection()),
        "fixed_thresholds": THRESHOLDS, "soft_object_constants": SOFT_CONSTANTS,
        "exact_model_forwards": 80, "gpu_launch_performed": False,
    }
    if args.wrapper:
        wrapper = Path(args.wrapper).resolve(strict=True)
        if file_sha256(wrapper) != WRAPPER_SHA256:
            fail("contract wrapper SHA differs")
        result["wrapper"] = {"path": str(wrapper), "sha256": WRAPPER_SHA256}
    if args.model_root:
        v4c = importlib.import_module(
            "methods.bernini_action_editing.extract_vjepa2_ordered_contextual_features_v4c"
        )
        result["frozen_model_closure"] = r1.required_model_closure(Path(args.model_root), v4c)
        result["transformers_module_closure"] = v4c.transformers_module_closure()
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prereg", required=True)
    parser.add_argument("--wrapper")
    parser.add_argument("--model-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    parser.add_argument("--print-contract", action="store_true")
    args = parser.parse_args(argv)
    if not args.print_contract and not all((args.wrapper, args.model_root, args.output)):
        parser.error("run requires --wrapper --model-root --output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_contract:
        print(json.dumps(contract(args), indent=2, sort_keys=True)); return 0
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as error:
        print(f"R1B_HARD_FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)

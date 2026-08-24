#!/usr/bin/env python3
"""Run the preregistered STARC core4 hidden-event critic pilot.

This executable consumes only sealed fixed-sketch hidden residual artifacts.
It performs exactly 200 AdamW steps over both fit cells, saves only the final
critic checkpoint, reloads it into a fresh model, freezes it, and only then
opens the two confirmation cells.  Confirmation is scored once in manifest
order.  It never enters an optimizer, checkpoint selection, threshold choice,
layer choice, or early-stopping decision.

The protocol is deliberately two-stage. ``fit-evaluate`` writes an immutable
provisional gate that is always negative because a current-RV2V live input-VJP
cannot be bound before the final step-200 critic exists.  ``finalize`` later
replays only sealed JSON/file/hash bindings, authenticates a composite VJP tied
to that exact checkpoint and runtime, and writes a separate final receipt in a
fresh directory.  It neither imports Torch nor reopens fit/confirmation tensor
artifacts.  The 24 held-out role margins and live VJP jointly decide only
whether fixed top-up generation is worth its cost.  This program has no editor
model, editor parameters, editor optimizer, generated-video target, or
source-video conditioning surface.

Materializer schema adaptation is deliberately concentrated in
``StarcMaterializerAdapter``.  The training/evaluation path operates only on
the normalized ``PilotManifestGraph`` and ``ArmArtifactBinding`` records.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import struct
import sys
import tarfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import latent_temporal_event_critic_dataset as data_contract  # noqa: E402


SCHEMA_VERSION = "bernini-starc-core4-critic-pilot-run-v1"
CONFIG_SCHEMA = "bernini-starc-core4-critic-pilot-config-v1"
TRACE_SCHEMA = "bernini-starc-core4-critic-fit-trace-v1"
CHECKPOINT_SCHEMA = "bernini-starc-core4-critic-checkpoint-v1"
PROVISIONAL_GATE_SCHEMA = "bernini-starc-core4-heldout-provisional-gate-v1"
FINAL_GATE_SCHEMA = "bernini-starc-core4-heldout-final-gate-v1"
# Backward-readable constant name; this is deliberately only provisional.
GATE_SCHEMA = PROVISIONAL_GATE_SCHEMA
HELDOUT_MARGIN_SCHEMA = "bernini-starc-core4-heldout-margin-gate-v1"
LIVE_VJP_BINDING_SCHEMA = (
    "bernini-starc-current-rv2v-live-vjp-composite-binding-v2"
)
LIVE_VJP_CANDIDATE_SCHEMA = "bernini-starc-current-candidate-vjp-binding-v1"
LIVE_VJP_BRIDGE_ARCHIVE_MEMBER = (
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py"
)
LIVE_VJP_BACKEND_ID = (
    "frozen_text_conditioned_temporal_event_critic_raw_score_vjp_v1"
)
LIVE_VJP_SP4_IMPLEMENTATION = (
    "torch.distributed.nn.functional.all_reduce_autograd"
)
BERNINI_OFFICIAL_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
VEOMNI_TESTED_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
BERNINI_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
BERNINI_CHECKPOINT_CONTENT_FILE_COUNT = 23

MASTER_SCHEMA = "bernini-starc-core4-same-state-hidden-master-v1"
GROUP_SCHEMA = "bernini-starc-core4-same-state-hidden-group-v1"
ARM_SCHEMA = "bernini-starc-core4-same-state-hidden-arm-v1"

MASTER_FILENAME = "starc-core4-hidden-master-v1.json"
GROUP_FILENAME_TEMPLATE = "starc-core4-hidden-group-{group_id}-v1.json"
ARM_RECEIPT_FILENAME = "starc-block15-hidden-arm-receipt-v1.json"
ARM_ARTIFACT_FILENAME = "starc-block15-hidden-residual.safetensors"
CHECKPOINT_FILENAME = "starc-core4-critic-final-step-0200.safetensors"
CONFIG_FILENAME = "starc-core4-critic-config-v1.json"
TRACE_FILENAME = "starc-core4-critic-fit-trace-v1.json"
CHECKPOINT_RECEIPT_FILENAME = "starc-core4-critic-checkpoint-receipt-v1.json"
PROVISIONAL_GATE_RECEIPT_FILENAME = (
    "starc-core4-heldout-provisional-gate-receipt-v1.json"
)
FINAL_GATE_RECEIPT_FILENAME = "starc-core4-heldout-final-gate-receipt-v1.json"
GATE_RECEIPT_FILENAME = PROVISIONAL_GATE_RECEIPT_FILENAME
PILOT_OUTPUT_FILENAMES = frozenset(
    {
        CHECKPOINT_FILENAME,
        CONFIG_FILENAME,
        TRACE_FILENAME,
        CHECKPOINT_RECEIPT_FILENAME,
        PROVISIONAL_GATE_RECEIPT_FILENAME,
    }
)

RESIDUAL_TENSOR_KEY = "sketched_action_minus_noop_hidden_residual"
RESIDUAL_SHAPE = (1, 21, 16, 1536)
RESIDUAL_DTYPE = "torch.float32"
GROUP_ORDER = ("sp4-a", "sp4-b")
NON_HEAD_STATE_KEYS = ("spatial_sketch", "nuisance_basis")
FIXED_MILESTONE_NAMES = (
    "actor_object_binding",
    "transition",
    "chronology",
    "terminal_hold",
)

FIXED_SEED = 20260808031
FIXED_OPTIMIZER_STEPS = 200
FIXED_LEARNING_RATE = 2.0e-4
FIXED_WEIGHT_DECAY = 1.0e-2
FIXED_MAXIMUM_GRADIENT_NORM = 1.0
FIXED_GLOBAL_MARGIN = 0.50
FIXED_MILESTONE_MARGIN = 0.25
FIXED_RANKING_TEMPERATURE = 0.50
FIXED_HELDOUT_MINIMUM_MARGIN = 0.20
FIXED_SPATIAL_SKETCH_SEED = 20260808017
SPATIAL_SKETCH_FAMILY_ID = "starc-counter-rademacher-s20260808017-v1"
SPATIAL_SKETCH_CONSTRUCTION_ID = "sha256-counter-rademacher-f32le-v1"
SPATIAL_SKETCH_VALUE_DIGEST_SCHEME = "fitq-canonical-fp32-little-endian-v1"
SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME = "bernini-ltec-f32le-v1"
REGISTERED_PATCH_GRIDS = ((30, 31), (32, 29), (34, 27))
REGISTERED_LATENT_TO_PATCH_GRID = {
    (60, 62): (30, 31),
    (64, 58): (32, 29),
    (68, 54): (34, 27),
}

FIXED_HYPERPARAMETERS = {
    "seed": FIXED_SEED,
    "optimizer": "AdamW_critic_head_only",
    "optimizer_steps": FIXED_OPTIMIZER_STEPS,
    "learning_rate": FIXED_LEARNING_RATE,
    "weight_decay": FIXED_WEIGHT_DECAY,
    "maximum_gradient_norm": FIXED_MAXIMUM_GRADIENT_NORM,
    "global_margin": FIXED_GLOBAL_MARGIN,
    "milestone_margin": FIXED_MILESTONE_MARGIN,
    "ranking_temperature": FIXED_RANKING_TEMPERATURE,
    "heldout_minimum_margin": FIXED_HELDOUT_MINIMUM_MARGIN,
    "fit_cells_per_step": 2,
    "confirmation_cells": 2,
    "checkpoint_selection": "final_step_200_only",
    "early_stopping": False,
    "confirmation_tuning": False,
}
GEOMETRY_NEUTRAL_CRITIC_CONFIG = {
    "hidden_size": 1536,
    "patch_positions": 16,
    "spatial_coordinates": 16,
    "spatial_sketch_seed": FIXED_SPATIAL_SKETCH_SEED,
    "projected_size": 48,
    "model_size": 96,
    "attention_heads": 4,
    "transformer_layers": 1,
    "softmin_temperature": 0.25,
    "dropout": 0.0,
    "require_nuisance_basis": False,
    "production_geometry": False,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_GROUP_BINDING_FIELDS = frozenset(
    {
        "group_id",
        "manifest_path",
        "manifest_file_sha256",
        "receipt_digest",
        "episode_order",
        "episode_splits",
        "arm_count",
        "model_forward_count",
    }
)
_ARM_BINDING_FIELDS = frozenset(
    {
        "episode_id",
        "split",
        "role",
        "label",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_digest",
        "artifact_path",
        "artifact_file_sha256",
        "artifact_tensor_sha256",
    }
)
_ARM_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "episode_id",
        "split",
        "role",
        "label",
        "action_family_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "seed",
        "source_candidate_binding",
        "event_label_binding",
        "critic_use_binding",
        "latent_binding",
        "official_gaussian_binding",
        "prompt_binding",
        "same_state_query_binding",
        "hidden_binding",
        "spatial_sketch_binding",
        "artifact",
        "model_binding",
        "runtime_binding",
        "model_forward_count",
        "labels_entered_model_condition",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)
_LATENT_BINDING_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "tensor_key",
        "stored_dtype",
        "shape",
        "raw_value_sha256",
        "content_sha256",
        "tensor_sha256",
        "clean_latent_authentication",
        "source_shape",
        "temporal_transform",
        "transform_applied_before_noising",
        "transformed_shape",
        "transformed_tensor_sha256",
        "generated_clean_latent_used_only_as_frozen_hidden_query",
    }
)
_CLEAN_LATENT_AUTHENTICATION_FIELDS = frozenset(
    {
        "shape",
        "dtype",
        "numel",
        "byte_count",
        "raw_value_sha256",
        "content_sha256",
        "authenticated_container_path",
        "authenticated_container_sha256",
        "single_tensor_container_reopened_byte_exact",
        "safetensors_metadata",
        "historical_native_coordinate_role_roundtrip_verified",
        "recorded_value_hashes_present",
        "historical_native_receipt_value_hashes_absent",
        "strict_recorded_value_identity_verified",
        "native_receipt_value_hashes_synthesized",
        "producer_time_value_digest_claimed_by_materializer",
        "observed_value_hashes_recomputed_after_authenticated_reopen",
        "value_identity_observation_time",
        "identity_authority",
        "binding_digest",
    }
)
_GROUP_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_binding",
        "bank_binding",
        "detached_event_label_binding",
        "critic_use_binding",
        "model_binding",
        "runtime_binding",
        "spatial_sketch_bindings_by_episode",
        "episode_order",
        "episode_splits",
        "arm_order",
        "arm_bindings",
        "candidate_count",
        "episode_count",
        "arm_count",
        "tensor_artifact_count",
        "model_forward_count",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)
_MASTER_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "root_spec_binding",
        "bank_binding",
        "detached_event_label_binding",
        "critic_use_binding",
        "model_binding",
        "runtime_binding",
        "spatial_sketch_bindings_by_episode",
        "group_order",
        "group_bindings",
        "episode_order",
        "episode_splits",
        "arm_order",
        "candidate_count",
        "episode_count",
        "arm_count",
        "tensor_artifact_count",
        "model_forward_count",
        "fit_episode_count",
        "confirmation_episode_count",
        "confirmation_consumed_by_optimizer",
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
        "receipt_digest",
    }
)
_CONFIG_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_schema_version",
        "fixed_hyperparameters",
        "fixed_hyperparameter_digest",
        "critic_config",
        "critic_config_content_digest",
        "pre_sketched_head_contract",
        "materializer_master_path",
        "materializer_master_file_sha256",
        "materializer_master_receipt_digest",
        "materialized_population_content_digest",
        "spatial_sketch_bindings_by_episode",
        "fit_episode_order",
        "confirmation_episode_order",
        "confirmation_tensor_load_phase",
        "nuisance_basis_used",
        "core4_scientific_claim_authorized",
        "editor_optimizer_present_or_authorized",
        "receipt_digest",
    }
)
_TRACE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "fixed_hyperparameters",
        "fit_episode_order",
        "fit_artifact_count",
        "optimizer_step_count",
        "both_fit_cells_consumed_every_step",
        "confirmation_manifest_metadata_authenticated_before_fit",
        "confirmation_tensor_artifacts_opened_before_fit_complete",
        "confirmation_samples_consumed_by_optimizer",
        "checkpoint_selection",
        "best_checkpoint_saved",
        "early_stopping_performed",
        "editor_parameter_present",
        "steps",
        "receipt_digest",
    }
)
_CHECKPOINT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_state_content_digest",
        "checkpoint_tensor_count",
        "checkpoint_scope",
        "excluded_constructor_buffer_keys",
        "config_receipt_digest",
        "optimizer_step",
        "only_final_checkpoint_saved",
        "best_checkpoint_saved",
        "confirmation_sample_seen_before_checkpoint_save",
        "state_tensor_byte_parity_after_fresh_load",
        "fit_score_parity_after_fresh_load",
        "critic_frozen_after_reload",
        "editor_checkpoint_or_parameter_present",
        "editor_optimizer_authorized",
        "receipt_digest",
    }
)
_PROVISIONAL_GATE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_schema_version",
        "gate_stage",
        "finalization_required",
        "materializer_binding",
        "config_binding",
        "fit_trace_binding",
        "checkpoint_binding",
        "fit_protocol",
        "confirmation_protocol",
        "heldout_margin_gate",
        "live_current_rv2v_input_vjp_gate",
        "worth_fixed_topup_generation",
        "scientific_critic_claim_authorized",
        "action_editing_success_claim_authorized",
        "editor_optimizer_present",
        "editor_optimizer_authorized",
        "generated_rgb_or_latent_used_as_editor_target_condition_donor_or_noise",
        "failure_reasons",
        "receipt_digest",
    }
)


class StarcCriticPilotError(RuntimeError):
    """A manifest, tensor, checkpoint, split, or gate failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise StarcCriticPilotError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_exact_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def reconstruct_geometry_spatial_sketch_binding(
    patch_height: int, patch_width: int
) -> dict[str, Any]:
    """Rebuild one geometry-specific 16-row sketch without Torch.

    The counter signs are shared across the family, while FP32 scale and every
    digest depend on ``P = patch_height * patch_width``.  This prevents a P930
    matrix from silently authenticating P928 or P918 residual coordinates.
    """

    if (patch_height, patch_width) not in REGISTERED_PATCH_GRIDS:
        raise StarcCriticPilotError("spatial sketch patch geometry is not registered")
    positions = patch_height * patch_width
    scale = struct.unpack("<f", struct.pack("<f", 1.0 / math.sqrt(positions)))[0]
    raw = bytearray()
    for row in range(16):
        for column in range(positions):
            token = f"{FIXED_SPATIAL_SKETCH_SEED}:{row}:{column}".encode("ascii")
            sign = 1.0 if hashlib.sha256(token).digest()[0] & 1 else -1.0
            raw.extend(struct.pack("<f", sign * scale))
    owned = bytes(raw)
    return {
        "sketch_family_id": SPATIAL_SKETCH_FAMILY_ID,
        "sketch_id": (
            f"starc-patch{patch_height}x{patch_width}-counter-rademacher-"
            f"s{FIXED_SPATIAL_SKETCH_SEED}-v1"
        ),
        "construction_id": SPATIAL_SKETCH_CONSTRUCTION_ID,
        "seed": FIXED_SPATIAL_SKETCH_SEED,
        "patch_positions": positions,
        "matrix_shape": [16, positions],
        "patch_grid_height_width": [patch_height, patch_width],
        "flatten_order": "patch-y-x",
        "normalization": f"per-row-rademacher-1-over-sqrt-{positions}",
        "matrix_dtype": "torch.float32",
        "matrix_raw_bytes_sha256": hashlib.sha256(owned).hexdigest(),
        "matrix_value_digest_scheme": SPATIAL_SKETCH_VALUE_DIGEST_SCHEME,
        "matrix_value_sha256": hashlib.sha256(
            f"{SPATIAL_SKETCH_VALUE_DIGEST_SCHEME}|shape=16,{positions}|".encode(
                "ascii"
            )
            + owned
        ).hexdigest(),
        "critic_tensor_digest_scheme": SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME,
        "critic_tensor_sha256": hashlib.sha256(
            f"{SPATIAL_SKETCH_CRITIC_DIGEST_SCHEME}|shape=16,{positions}|".encode(
                "ascii"
            )
            + owned
        ).hexdigest(),
        "full_support_no_mask_or_localizer": True,
        "data_dependent": False,
    }


REGISTERED_SPATIAL_SKETCH_BINDINGS = {
    f"{height}x{width}": reconstruct_geometry_spatial_sketch_binding(height, width)
    for height, width in REGISTERED_PATCH_GRIDS
}


def file_sha256(value: str | Path) -> str:
    path = Path(value)
    try:
        before = path.stat()
    except OSError as error:
        raise StarcCriticPilotError(f"could not stat file while hashing: {path}") from error
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as error:
        raise StarcCriticPilotError(f"could not hash file: {path}") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise StarcCriticPilotError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(unsigned)
    return {**row, "receipt_digest": object_sha256(row)}


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise StarcCriticPilotError(f"{label} must be lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise StarcCriticPilotError(f"{label} must be a path-safe identifier")
    return value


def _plain_file(
    value: str | Path, *, label: str, beneath: Optional[Path] = None
) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested == Path("/"):
        raise StarcCriticPilotError(f"{label} must be absolute and non-root")
    if requested.is_symlink():
        raise StarcCriticPilotError(f"{label} must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise StarcCriticPilotError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISREG(mode):
        raise StarcCriticPilotError(f"{label} must be a canonical regular file")
    if beneath is not None:
        try:
            resolved.relative_to(beneath)
        except ValueError as error:
            raise StarcCriticPilotError(
                f"{label} escapes the materializer output root"
            ) from error
    return resolved


def _plain_directory(value: str | Path, *, label: str) -> Path:
    requested = Path(value)
    if not requested.is_absolute() or requested == Path("/") or requested.is_symlink():
        raise StarcCriticPilotError(f"{label} must be absolute, non-root, and plain")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise StarcCriticPilotError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISDIR(mode):
        raise StarcCriticPilotError(f"{label} must be a canonical directory")
    return resolved


def _read_json_file(
    value: str | Path,
    *,
    label: str,
    expected_sha256: Optional[str] = None,
    beneath: Optional[Path] = None,
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label, beneath=beneath)
    observed_sha = file_sha256(path)
    if expected_sha256 is not None and observed_sha != _sha256(
        expected_sha256, label=f"{label} expected file SHA-256"
    ):
        raise StarcCriticPilotError(f"{label} file SHA-256 differs")
    def reject_constant(token: str) -> None:
        raise StarcCriticPilotError(f"{label} contains non-finite JSON: {token}")

    def reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise StarcCriticPilotError(
                    f"{label} contains duplicate JSON key: {key}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except StarcCriticPilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StarcCriticPilotError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise StarcCriticPilotError(f"{label} must be a JSON object")
    return parsed, path, observed_sha


def _validate_sealed_manifest(
    value: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    row = dict(value)
    declared = _sha256(row.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(row) != declared:
        raise StarcCriticPilotError(f"{label} receipt digest differs")
    if row.get("schema_version") != schema:
        raise StarcCriticPilotError(f"{label} schema differs")
    return {**row, "receipt_digest": declared}


def _closed_binding(
    value: Any, fields: frozenset[str], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise StarcCriticPilotError(f"{label} field closure differs")
    return dict(value)


def _normalize_episode_splits(
    value: Any, episode_order: Sequence[str], *, label: str
) -> dict[str, str]:
    order = tuple(episode_order)
    if isinstance(value, Mapping):
        if set(value) != set(order):
            raise StarcCriticPilotError(f"{label} key closure differs")
        result = {episode_id: value[episode_id] for episode_id in order}
    elif isinstance(value, list) and len(value) == len(order):
        result = dict(zip(order, value))
    else:
        raise StarcCriticPilotError(f"{label} must bind every episode in order")
    if any(split not in data_contract.PILOT_SPLITS for split in result.values()):
        raise StarcCriticPilotError(f"{label} contains an unregistered split")
    return result


def _denies_excess_authority(row: Mapping[str, Any], *, label: str) -> None:
    """Reject any materializer receipt that claims training/editor authority."""

    required_false_aliases = {
        "training": ("training_performed",),
        "critic_optimizer": (
            "optimizer_authorized",
            "critic_optimizer_authorized",
            "optimizer_step_performed",
        ),
        "editor_optimizer": (
            "editor_optimizer_authorized",
            "core4_can_authorize_editor_optimizer",
        ),
        "scientific_claim": (
            "scientific_critic_claim_authorized",
            "scientific_action_editing_claim_authorized",
            "core4_can_authorize_scientific_claim",
        ),
        "generated_media_editor_use": (
            "generated_media_editor_use_authorized",
        ),
    }
    for semantic, aliases in required_false_aliases.items():
        present = [name for name in aliases if name in row]
        if not present or any(row[name] is not False for name in present):
            raise StarcCriticPilotError(
                f"{label} does not explicitly deny {semantic} authority"
            )


def _validate_historical_clean_latent_authentication(
    value: Any,
    *,
    latent: Mapping[str, Any],
    source_shape: Sequence[int],
    label: str,
) -> dict[str, Any]:
    """Validate the historical no-value-digest compatibility path exactly."""

    authentication = _closed_binding(
        value,
        _CLEAN_LATENT_AUTHENTICATION_FIELDS,
        label=f"{label} clean latent authentication",
    )
    unsigned = dict(authentication)
    binding_digest = _sha256(
        unsigned.pop("binding_digest"),
        label=f"{label} clean latent authentication binding digest",
    )
    if object_sha256(unsigned) != binding_digest:
        raise StarcCriticPilotError(
            f"{label} clean latent authentication binding digest differs"
        )
    for name in (
        "raw_value_sha256",
        "content_sha256",
        "authenticated_container_sha256",
    ):
        _sha256(
            authentication[name],
            label=f"{label} clean latent authentication {name}",
        )
    container = _plain_file(
        authentication["authenticated_container_path"],
        label=f"{label} authenticated clean latent container",
    )
    shape = list(source_shape)
    numel = math.prod(shape)
    expected_metadata = {
        "coordinate": "bernini_normalized_clean_vae_latent",
        "frame_contract": "exact81_latent21",
        "artifact_role": "native_sampler_proposal",
        "source": "native_sampler_before_vae_decode",
    }
    if (
        authentication["shape"] != shape
        or authentication["dtype"] != "torch.float32"
        or authentication["numel"] != numel
        or authentication["byte_count"] != numel * 4
        or str(container) != latent.get("path")
        or str(container) != authentication["authenticated_container_path"]
        or file_sha256(container) != authentication["authenticated_container_sha256"]
        or authentication["authenticated_container_sha256"]
        != latent.get("file_sha256")
        or authentication["raw_value_sha256"] != latent.get("raw_value_sha256")
        or authentication["content_sha256"] != latent.get("content_sha256")
        or authentication["safetensors_metadata"] != expected_metadata
        or authentication["single_tensor_container_reopened_byte_exact"] is not True
        or authentication[
            "historical_native_coordinate_role_roundtrip_verified"
        ]
        is not True
        or authentication["recorded_value_hashes_present"] is not False
        or authentication["historical_native_receipt_value_hashes_absent"]
        is not True
        or authentication["strict_recorded_value_identity_verified"] is not False
        or authentication["native_receipt_value_hashes_synthesized"] is not False
        or authentication[
            "producer_time_value_digest_claimed_by_materializer"
        ]
        is not False
        or authentication[
            "observed_value_hashes_recomputed_after_authenticated_reopen"
        ]
        is not True
        or authentication["value_identity_observation_time"]
        != "materializer_authenticated_reopen"
        or authentication["identity_authority"]
        != "authenticated_single_tensor_container_sha256_and_native_fp32_roundtrip"
    ):
        raise StarcCriticPilotError(
            f"{label} historical clean latent authentication differs"
        )
    return authentication


@dataclass(frozen=True)
class ArmArtifactBinding:
    group_id: str
    episode_id: str
    split: str
    role: str
    label: int
    receipt_path: Path
    receipt_file_sha256: str
    receipt_digest: str
    artifact_path: Path
    artifact_file_sha256: str
    artifact_tensor_sha256: str
    tensor_key: str
    tensor_shape: tuple[int, ...]
    tensor_dtype: str
    source_latent_shape: tuple[int, ...]
    patch_grid_height_width: tuple[int, int]
    spatial_sketch_binding: Mapping[str, Any]

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "episode_id": self.episode_id,
            "split": self.split,
            "role": self.role,
            "label": self.label,
            "receipt_path": str(self.receipt_path),
            "receipt_file_sha256": self.receipt_file_sha256,
            "receipt_digest": self.receipt_digest,
            "artifact_path": str(self.artifact_path),
            "artifact_file_sha256": self.artifact_file_sha256,
            "artifact_tensor_sha256": self.artifact_tensor_sha256,
            "tensor_key": self.tensor_key,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "source_latent_shape": list(self.source_latent_shape),
            "patch_grid_height_width": list(self.patch_grid_height_width),
            "spatial_sketch_binding": dict(self.spatial_sketch_binding),
        }


@dataclass(frozen=True)
class PilotManifestGraph:
    master_path: Path
    master_file_sha256: str
    master_receipt_digest: str
    materializer_root: Path
    group_order: tuple[str, ...]
    group_manifest_file_sha256s: tuple[str, ...]
    group_receipt_digests: tuple[str, ...]
    episode_order: tuple[str, ...]
    episode_splits: Mapping[str, str]
    spatial_sketch_bindings_by_episode: Mapping[str, Mapping[str, Any]]
    arms: tuple[ArmArtifactBinding, ...]
    content_digest: str

    def episode_ids(self, split: str) -> tuple[str, ...]:
        if split not in data_contract.PILOT_SPLITS:
            raise StarcCriticPilotError("requested split is not registered")
        return tuple(
            episode_id
            for episode_id in self.episode_order
            if self.episode_splits[episode_id] == split
        )

    def arms_for_split(self, split: str) -> tuple[ArmArtifactBinding, ...]:
        ids = set(self.episode_ids(split))
        return tuple(row for row in self.arms if row.episode_id in ids)


class StarcMaterializerAdapter:
    """Strictly normalize the STARC materializer's master/group/arm graph."""

    @staticmethod
    def _validate_arm_receipt(
        value: Mapping[str, Any],
        *,
        group_id: str,
        binding: Mapping[str, Any],
        materializer_root: Path,
    ) -> ArmArtifactBinding:
        label = (
            f"arm receipt {binding.get('episode_id', '?')}/"
            f"{binding.get('role', '?')}"
        )
        row = _validate_sealed_manifest(value, schema=ARM_SCHEMA, label=label)
        _closed_binding(row, _ARM_MANIFEST_FIELDS, label=label)
        _denies_excess_authority(row, label=label)

        episode_id = _safe_id(binding["episode_id"], label=f"{label} episode ID")
        role = binding["role"]
        split = binding["split"]
        expected_label = 1 if role == "positive" else 0
        if (
            role not in data_contract.ARM_ROLES
            or split not in data_contract.PILOT_SPLITS
            or type(binding["label"]) is not int
            or type(row.get("label")) is not int
            or binding["label"] != expected_label
            or row.get("group_id") != group_id
            or row.get("episode_id") != episode_id
            or row.get("split") != split
            or row.get("role") != role
            or row.get("label") != expected_label
            or row["receipt_digest"] != binding["receipt_digest"]
            or row.get("model_forward_count") != 2
            or row.get("labels_entered_model_condition") is not False
        ):
            raise StarcCriticPilotError(f"{label} identity/role closure differs")
        for name in (
            "action_family_id",
            "actor_group_id",
            "scene_group_id",
            "action_group_id",
        ):
            _safe_id(row[name], label=f"{label} {name}")
        if type(row.get("seed")) is not int or not 0 <= row["seed"] < 2**63:
            raise StarcCriticPilotError(f"{label} seed differs")

        artifact = row.get("artifact")
        if not isinstance(artifact, Mapping):
            raise StarcCriticPilotError(f"{label} artifact binding is absent")
        required_artifact_fields = {
            "path",
            "file_sha256",
            "tensor_key",
            "tensor_shape",
            "tensor_dtype",
            "tensor_sha256",
            "detached_finite_fp32",
        }
        if set(artifact) != required_artifact_fields:
            raise StarcCriticPilotError(f"{label} artifact field closure differs")
        artifact_path = _plain_file(
            artifact["path"], label=f"{label} tensor artifact", beneath=materializer_root
        )
        tensor_shape = artifact["tensor_shape"]
        if (
            str(artifact_path) != binding["artifact_path"]
            or Path(binding["receipt_path"]).name != ARM_RECEIPT_FILENAME
            or artifact_path.name != ARM_ARTIFACT_FILENAME
            or artifact["file_sha256"] != binding["artifact_file_sha256"]
            or artifact["tensor_sha256"] != binding["artifact_tensor_sha256"]
            or artifact["tensor_key"] != RESIDUAL_TENSOR_KEY
            or tensor_shape != list(RESIDUAL_SHAPE)
            or any(type(item) is not int for item in tensor_shape)
            or artifact["tensor_dtype"] != RESIDUAL_DTYPE
            or artifact["detached_finite_fp32"] is not True
        ):
            raise StarcCriticPilotError(f"{label} tensor contract differs")
        latent = _closed_binding(
            row.get("latent_binding"),
            _LATENT_BINDING_FIELDS,
            label=f"{label} latent binding",
        )
        source_shape = latent.get("source_shape")
        expected_transform = data_contract.TEMPORAL_TRANSFORM_BY_ROLE[role]
        if (
            not isinstance(source_shape, list)
            or len(source_shape) != 5
            or source_shape[:3] != [1, 16, 21]
            or any(type(item) is not int or item <= 0 for item in source_shape)
            or source_shape[3] % 2 != 0
            or source_shape[4] % 2 != 0
            or latent.get("transformed_shape") != source_shape
            or latent.get("temporal_transform") != expected_transform
            or latent.get("transform_applied_before_noising") is not True
            or latent.get("tensor_key") != "normalized_clean_latent"
            or latent.get("stored_dtype") != "torch.float32"
            or latent.get("shape") != source_shape
            or latent.get(
                "generated_clean_latent_used_only_as_frozen_hidden_query"
            )
            is not True
        ):
            raise StarcCriticPilotError(f"{label} source latent geometry differs")
        for name in (
            "file_sha256",
            "raw_value_sha256",
            "content_sha256",
            "tensor_sha256",
            "transformed_tensor_sha256",
        ):
            _sha256(latent.get(name), label=f"{label} latent {name}")
        _validate_historical_clean_latent_authentication(
            latent.get("clean_latent_authentication"),
            latent=latent,
            source_shape=source_shape,
            label=label,
        )
        patch_grid = (source_shape[3] // 2, source_shape[4] // 2)
        expected_sketch = reconstruct_geometry_spatial_sketch_binding(*patch_grid)
        spatial_sketch = row.get("spatial_sketch_binding")
        if not _json_exact_equal(spatial_sketch, expected_sketch):
            raise StarcCriticPilotError(
                f"{label} geometry-specific spatial sketch binding differs"
            )
        query = row.get("same_state_query_binding")
        if (
            not isinstance(query, Mapping)
            or query.get("native_schedule_index")
            != data_contract.PILOT_HIDDEN_QUERY["native_schedule_index"]
            or not isinstance(query.get("sigma"), (int, float))
            or isinstance(query.get("sigma"), bool)
            or float(query["sigma"]).hex()
            != float(data_contract.PILOT_HIDDEN_QUERY["sigma"]).hex()
            or query.get("native_timestep")
            != data_contract.PILOT_HIDDEN_QUERY["native_timestep"]
            or query.get("action_and_noop_share_exact_x_sigma_object") is not True
            or query.get("action_and_noop_share_exact_rotary_object") is not True
            or query.get("action_and_noop_share_exact_timestep_object") is not True
            or query.get("shared_tensor_bytes_unchanged") is not True
            or query.get("block0_input_and_attn1_exact_parity") is not True
        ):
            raise StarcCriticPilotError(f"{label} same-state query proof differs")
        hidden = row.get("hidden_binding")
        if (
            not isinstance(hidden, Mapping)
            or hidden.get("hook_coordinate")
            != data_contract.PILOT_HIDDEN_QUERY["hook_coordinate"]
            or hidden.get("action_global_sketch_shape") != list(RESIDUAL_SHAPE)
            or hidden.get("noop_global_sketch_shape") != list(RESIDUAL_SHAPE)
            or hidden.get("residual_shape") != list(RESIDUAL_SHAPE)
            or hidden.get("patch_positions") != patch_grid[0] * patch_grid[1]
            or hidden.get("patch_grid_height_width") != list(patch_grid)
            or hidden.get("full_hidden_persisted") is not False
        ):
            raise StarcCriticPilotError(f"{label} hidden/sketch closure differs")
        event = row.get("event_label_binding")
        if (
            not isinstance(event, Mapping)
            or event.get("labels_are_external_and_detached") is not True
            or event.get("labels_may_enter_model_condition") is not False
        ):
            raise StarcCriticPilotError(f"{label} detached-label binding differs")
        critic_use = row.get("critic_use_binding")
        if (
            not isinstance(critic_use, Mapping)
            or critic_use.get("authorized_use") != data_contract.CRITIC_ONLY_USE
        ):
            raise StarcCriticPilotError(f"{label} critic-only authority differs")
        for name in (
            "receipt_file_sha256",
            "receipt_digest",
            "artifact_file_sha256",
            "artifact_tensor_sha256",
        ):
            _sha256(binding[name], label=f"{label} {name}")
        return ArmArtifactBinding(
            group_id=group_id,
            episode_id=episode_id,
            split=split,
            role=role,
            label=expected_label,
            receipt_path=Path(binding["receipt_path"]),
            receipt_file_sha256=binding["receipt_file_sha256"],
            receipt_digest=binding["receipt_digest"],
            artifact_path=artifact_path,
            artifact_file_sha256=binding["artifact_file_sha256"],
            artifact_tensor_sha256=binding["artifact_tensor_sha256"],
            tensor_key=artifact["tensor_key"],
            tensor_shape=tuple(tensor_shape),
            tensor_dtype=artifact["tensor_dtype"],
            source_latent_shape=tuple(source_shape),
            patch_grid_height_width=patch_grid,
            spatial_sketch_binding=expected_sketch,
        )

    @classmethod
    def load(
        cls, master_manifest: str | Path, *, expected_master_sha256: str
    ) -> PilotManifestGraph:
        master_raw, master_path, master_file_sha = _read_json_file(
            master_manifest,
            label="STARC hidden master manifest",
            expected_sha256=expected_master_sha256,
        )
        master = _validate_sealed_manifest(
            master_raw, schema=MASTER_SCHEMA, label="STARC hidden master manifest"
        )
        _closed_binding(
            master,
            _MASTER_MANIFEST_FIELDS,
            label="STARC hidden master manifest",
        )
        _denies_excess_authority(master, label="STARC hidden master manifest")
        materializer_root = master_path.parent.resolve(strict=True)

        episode_order_raw = master.get("episode_order")
        if (
            master_path.name != MASTER_FILENAME
            or master.get("group_order") != list(GROUP_ORDER)
            or not isinstance(episode_order_raw, list)
            or len(episode_order_raw) != 4
            or len(set(episode_order_raw)) != 4
            or any(
                _SAFE_ID_RE.fullmatch(str(episode_id)) is None
                for episode_id in episode_order_raw
            )
            or master.get("arm_order") != list(data_contract.ARM_ROLES)
            or master.get("candidate_count") != 40
            or master.get("episode_count") != 4
            or master.get("arm_count") != 52
            or master.get("tensor_artifact_count") != 52
            or master.get("model_forward_count") != 104
            or master.get("fit_episode_count") != 2
            or master.get("confirmation_episode_count") != 2
            or master.get("confirmation_consumed_by_optimizer") is not False
        ):
            raise StarcCriticPilotError("STARC hidden master topology differs")
        episode_order = tuple(str(item) for item in episode_order_raw)
        episode_splits = _normalize_episode_splits(
            master.get("episode_splits"), episode_order, label="master episode splits"
        )
        if sorted(episode_splits.values()) != [
            "confirmation",
            "confirmation",
            "fit",
            "fit",
        ]:
            raise StarcCriticPilotError("master must contain exact fit2/confirmation2")
        master_sketches = master.get("spatial_sketch_bindings_by_episode")
        if not isinstance(master_sketches, Mapping) or set(master_sketches) != set(
            episode_order
        ):
            raise StarcCriticPilotError(
                "master geometry-specific spatial sketch closure differs"
            )

        bindings = master.get("group_bindings")
        if not isinstance(bindings, list) or len(bindings) != len(GROUP_ORDER):
            raise StarcCriticPilotError("master group bindings differ")

        all_arms: list[ArmArtifactBinding] = []
        group_file_shas: list[str] = []
        group_receipt_digests: list[str] = []
        observed_group_episodes: list[str] = []
        for group_index, group_id in enumerate(GROUP_ORDER):
            binding = _closed_binding(
                bindings[group_index],
                _GROUP_BINDING_FIELDS,
                label=f"master group binding {group_id}",
            )
            if binding["group_id"] != group_id:
                raise StarcCriticPilotError("master group binding order differs")
            _sha256(
                binding["manifest_file_sha256"],
                label=f"{group_id} manifest file SHA-256",
            )
            _sha256(binding["receipt_digest"], label=f"{group_id} receipt digest")
            group_raw, group_path, group_file_sha = _read_json_file(
                binding["manifest_path"],
                label=f"{group_id} group manifest",
                expected_sha256=binding["manifest_file_sha256"],
                beneath=materializer_root,
            )
            group = _validate_sealed_manifest(
                group_raw, schema=GROUP_SCHEMA, label=f"{group_id} group manifest"
            )
            _closed_binding(
                group,
                _GROUP_MANIFEST_FIELDS,
                label=f"{group_id} group manifest",
            )
            _denies_excess_authority(group, label=f"{group_id} group manifest")
            if (
                str(group_path) != binding["manifest_path"]
                or group_path.name
                != GROUP_FILENAME_TEMPLATE.format(group_id=group_id)
                or group["receipt_digest"] != binding["receipt_digest"]
                or group.get("group_id") != group_id
                or group.get("episode_order") != binding["episode_order"]
                or group.get("episode_splits") != binding["episode_splits"]
                or group.get("arm_count") != binding["arm_count"]
                or group.get("model_forward_count")
                != binding["model_forward_count"]
                or binding["arm_count"] != 26
                or binding["model_forward_count"] != 52
                or group.get("candidate_count") != 20
                or group.get("episode_count") != 2
                or group.get("tensor_artifact_count") != 26
                or any(
                    group.get(name) != master.get(name)
                    for name in (
                        "root_spec_binding",
                        "bank_binding",
                        "detached_event_label_binding",
                        "critic_use_binding",
                        "model_binding",
                        "runtime_binding",
                    )
                )
            ):
                raise StarcCriticPilotError(f"{group_id} master/group binding differs")
            group_episodes_raw = group.get("episode_order")
            if (
                not isinstance(group_episodes_raw, list)
                or len(group_episodes_raw) != 2
                or len(set(group_episodes_raw)) != 2
            ):
                raise StarcCriticPilotError(f"{group_id} episode topology differs")
            group_episodes = tuple(str(item) for item in group_episodes_raw)
            group_splits = _normalize_episode_splits(
                group.get("episode_splits"),
                group_episodes,
                label=f"{group_id} episode splits",
            )
            if sorted(group_splits.values()) != ["confirmation", "fit"] or any(
                episode_splits.get(episode_id) != split
                for episode_id, split in group_splits.items()
            ):
                raise StarcCriticPilotError(f"{group_id} split binding differs")
            group_sketches = group.get("spatial_sketch_bindings_by_episode")
            if (
                not isinstance(group_sketches, Mapping)
                or set(group_sketches) != set(group_episodes)
                or any(
                    not _json_exact_equal(
                        group_sketches[episode_id], master_sketches[episode_id]
                    )
                    for episode_id in group_episodes
                )
            ):
                raise StarcCriticPilotError(
                    f"{group_id} spatial sketch family binding differs"
                )
            observed_group_episodes.extend(group_episodes)
            if group.get("arm_order") != list(data_contract.ARM_ROLES):
                raise StarcCriticPilotError(f"{group_id} arm order differs")
            arm_bindings = group.get("arm_bindings")
            if not isinstance(arm_bindings, list) or len(arm_bindings) != 26:
                raise StarcCriticPilotError(f"{group_id} arm bindings differ")
            for arm_index, raw_arm_binding in enumerate(arm_bindings):
                arm_binding = _closed_binding(
                    raw_arm_binding,
                    _ARM_BINDING_FIELDS,
                    label=f"{group_id} arm binding {arm_index}",
                )
                expected_episode = group_episodes[arm_index // len(data_contract.ARM_ROLES)]
                expected_role = data_contract.ARM_ROLES[
                    arm_index % len(data_contract.ARM_ROLES)
                ]
                expected_split = group_splits[expected_episode]
                if (
                    arm_binding["episode_id"] != expected_episode
                    or arm_binding["split"] != expected_split
                    or arm_binding["role"] != expected_role
                    or arm_binding["label"] != (1 if expected_role == "positive" else 0)
                ):
                    raise StarcCriticPilotError(
                        f"{group_id} arm binding order/label differs"
                    )
                arm_raw, receipt_path, _receipt_file_sha = _read_json_file(
                    arm_binding["receipt_path"],
                    label=f"{expected_episode}/{expected_role} arm receipt",
                    expected_sha256=arm_binding["receipt_file_sha256"],
                    beneath=materializer_root,
                )
                if str(receipt_path) != arm_binding["receipt_path"]:
                    raise StarcCriticPilotError("arm receipt path binding differs")
                checked_arm = cls._validate_arm_receipt(
                    arm_raw,
                    group_id=group_id,
                    binding=arm_binding,
                    materializer_root=materializer_root,
                )
                if (
                    not _json_exact_equal(
                        checked_arm.spatial_sketch_binding,
                        group_sketches[expected_episode],
                    )
                ):
                    raise StarcCriticPilotError(
                        f"{expected_episode} arm/group spatial sketch differs"
                    )
                all_arms.append(checked_arm)
            group_file_shas.append(group_file_sha)
            group_receipt_digests.append(group["receipt_digest"])

        if tuple(observed_group_episodes) != episode_order:
            raise StarcCriticPilotError("group episode order differs from master")
        expected_arm_pairs = [
            (episode_id, role)
            for episode_id in episode_order
            for role in data_contract.ARM_ROLES
        ]
        if [(row.episode_id, row.role) for row in all_arms] != expected_arm_pairs:
            raise StarcCriticPilotError("normalized arm population closure differs")
        if (
            len({row.receipt_path for row in all_arms}) != 52
            or len({row.artifact_path for row in all_arms}) != 52
        ):
            raise StarcCriticPilotError("arm receipt/artifact paths are not unique")
        normalized_sketches = {}
        for episode_id in episode_order:
            episode_arms = [row for row in all_arms if row.episode_id == episode_id]
            if len(episode_arms) != len(data_contract.ARM_ROLES) or any(
                not _json_exact_equal(
                    row.spatial_sketch_binding,
                    episode_arms[0].spatial_sketch_binding,
                )
                for row in episode_arms
            ):
                raise StarcCriticPilotError(
                    f"{episode_id} arm spatial sketches are not identical"
                )
            normalized_sketches[episode_id] = dict(
                episode_arms[0].spatial_sketch_binding
            )
        if not _json_exact_equal(dict(master_sketches), normalized_sketches):
            raise StarcCriticPilotError(
                "master spatial sketch mapping differs from authenticated arms"
            )
        content_identity = {
            "master_file_sha256": master_file_sha,
            "master_receipt_digest": master["receipt_digest"],
            "group_manifest_file_sha256s": group_file_shas,
            "group_receipt_digests": group_receipt_digests,
            "episode_order": list(episode_order),
            "episode_splits": dict(episode_splits),
            "spatial_sketch_bindings_by_episode": normalized_sketches,
            "arms": [row.canonical_identity() for row in all_arms],
        }
        return PilotManifestGraph(
            master_path=master_path,
            master_file_sha256=master_file_sha,
            master_receipt_digest=master["receipt_digest"],
            materializer_root=materializer_root,
            group_order=GROUP_ORDER,
            group_manifest_file_sha256s=tuple(group_file_shas),
            group_receipt_digests=tuple(group_receipt_digests),
            episode_order=episode_order,
            episode_splits=episode_splits,
            spatial_sketch_bindings_by_episode=normalized_sketches,
            arms=tuple(all_arms),
            content_digest=object_sha256(content_identity),
        )


def _runtime_modules() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        import latent_temporal_event_critic as critic_core
        import train_latent_temporal_event_critic as critic_trainer
    except ImportError as error:
        raise StarcCriticPilotError(
            "torch, safetensors, and STARC critic runtime are required"
        ) from error
    return torch, safe_open, load_file, save_file, (critic_core, critic_trainer)


def _tensor_state_digest(value: Any, *, label: str) -> str:
    torch, _safe_open, _load_file, _save_file, _modules = _runtime_modules()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise StarcCriticPilotError(f"{label} must be detached finite tensor")
    owned = value.detach().cpu().contiguous().clone()
    header = canonical_json_bytes(
        {"dtype": str(owned.dtype), "shape": list(map(int, owned.shape))}
    )
    return hashlib.sha256(header + b"|" + bytes(owned.untyped_storage())).hexdigest()


def checkpoint_state_content_digest(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise StarcCriticPilotError("checkpoint state must be a nonempty mapping")
    rows = []
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise StarcCriticPilotError("checkpoint parameter name differs")
        tensor = state[name]
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(map(int, tensor.shape)),
                "tensor_digest": _tensor_state_digest(
                    tensor.detach(), label=f"checkpoint tensor {name}"
                ),
            }
        )
    return object_sha256(rows)


def materializer_tensor_sha256(value: Any) -> str:
    """Reproduce the materializer's exact tensor-value digest scheme."""

    torch, _safe_open, _load_file, _save_file, _modules = _runtime_modules()
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise StarcCriticPilotError(
            "materializer tensor digest requires a detached finite tensor"
        )
    owned = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in owned.shape],
        "dtype": str(owned.dtype),
        "layout": str(owned.layout),
    }
    # ``bytes(UntypedStorage)`` avoids the PyTorch/NumPy ABI entirely while
    # yielding the same contiguous CPU byte sequence used by the materializer.
    raw = bytes(owned.untyped_storage())
    expected_nbytes = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected_nbytes:
        raise StarcCriticPilotError("materializer tensor storage size differs")
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + raw).hexdigest()


def _load_residual_tensor(binding: ArmArtifactBinding, *, device: str) -> Any:
    torch, _safe_open, _load_file, _save_file, _modules = _runtime_modules()
    path = _plain_file(
        binding.artifact_path,
        label=f"{binding.episode_id}/{binding.role} residual artifact",
    )
    try:
        # Read each residual artifact exactly once.  In particular, a held-out
        # file is not first hashed and then reopened by the tensor loader.
        payload = path.read_bytes()
    except OSError as error:
        raise StarcCriticPilotError(
            f"{binding.episode_id}/{binding.role} residual read failed"
        ) from error
    if hashlib.sha256(payload).hexdigest() != binding.artifact_file_sha256:
        raise StarcCriticPilotError(
            f"{binding.episode_id}/{binding.role} residual file SHA-256 differs"
        )
    try:
        from safetensors.torch import load as load_safetensors_bytes

        tensors = load_safetensors_bytes(payload)
        if list(tensors) != [binding.tensor_key]:
            raise StarcCriticPilotError(
                f"{binding.episode_id}/{binding.role} tensor key closure differs"
            )
        tensor = tensors[binding.tensor_key]
    except StarcCriticPilotError:
        raise
    except Exception as error:
        raise StarcCriticPilotError(
            f"{binding.episode_id}/{binding.role} safetensors load failed"
        ) from error
    if (
        tensor.dtype != torch.float32
        or tuple(map(int, tensor.shape)) != RESIDUAL_SHAPE
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise StarcCriticPilotError(
            f"{binding.episode_id}/{binding.role} residual tensor differs"
        )
    observed_digest = materializer_tensor_sha256(tensor)
    if observed_digest != binding.artifact_tensor_sha256:
        raise StarcCriticPilotError(
            f"{binding.episode_id}/{binding.role} tensor value digest differs"
        )
    return tensor.detach().contiguous().to(device=device)


@dataclass(frozen=True)
class LoadedEpisode:
    episode_id: str
    split: str
    tensors_by_role: Mapping[str, Any]
    loaded_artifact_count: int


def load_split_tensors(
    graph: PilotManifestGraph, *, split: str, device: str
) -> tuple[LoadedEpisode, ...]:
    """Open exactly one split's tensors; callers control the isolation time."""

    episode_ids = graph.episode_ids(split)
    if len(episode_ids) != 2:
        raise StarcCriticPilotError(f"{split} must contain exactly two episodes")
    rows = graph.arms_for_split(split)
    result = []
    for episode_id in episode_ids:
        episode_rows = [row for row in rows if row.episode_id == episode_id]
        if [row.role for row in episode_rows] != list(data_contract.ARM_ROLES):
            raise StarcCriticPilotError(f"{episode_id} role order differs")
        tensors = {
            row.role: _load_residual_tensor(row, device=device) for row in episode_rows
        }
        result.append(
            LoadedEpisode(
                episode_id=episode_id,
                split=split,
                tensors_by_role=tensors,
                loaded_artifact_count=len(tensors),
            )
        )
    return tuple(result)


def _make_group_batches(episodes: Sequence[LoadedEpisode]) -> tuple[Any, ...]:
    _torch, _safe_open, _load_file, _save_file, modules = _runtime_modules()
    _critic_core, critic_trainer = modules
    return tuple(
        critic_trainer.CriticGroupBatch(
            episode_id=episode.episode_id,
            positive_sketched_residual=episode.tensors_by_role["positive"],
            negative_sketched_residuals={
                role: episode.tensors_by_role[role]
                for role in data_contract.NEGATIVE_ROLES
            },
        )
        for episode in episodes
    )


def _configure_determinism(device: str) -> Any:
    # cuBLAS requires this workspace contract for deterministic GEMMs.  Set it
    # before the first CUDA allocation/model construction in this process.
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace not in (None, ":4096:8"):
        raise StarcCriticPilotError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the fixed deterministic run"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch, _safe_open, _load_file, _save_file, _modules = _runtime_modules()
    if not isinstance(device, str) or not device:
        raise StarcCriticPilotError("device must be nonempty")
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise StarcCriticPilotError("requested CUDA device is unavailable")
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FIXED_SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    return parsed


def _build_fresh_critic(*, device: str) -> Any:
    torch, _safe_open, _load_file, _save_file, modules = _runtime_modules()
    critic_core, _critic_trainer = modules
    # The fit/evaluation artifact is already spatially sketched.  The small
    # head must therefore be geometry-neutral; a P930 preprocessing buffer
    # would falsely claim to describe P928/P918 episodes.  This 16x16 identity
    # is an inert constructor sentinel and is excluded from the checkpoint.
    config = critic_core.CriticConfig(**GEOMETRY_NEUTRAL_CRITIC_CONFIG)
    sketch = torch.eye(16, dtype=torch.float32)
    model = critic_core.FrozenHiddenTemporalEventCritic(
        sketch,
        config=config,
    )
    if model.trainable_parameter_count >= 1_000_000:
        raise StarcCriticPilotError("critic head exceeds the preregistered budget")
    return model.to(device=device)


def _trainer_config() -> Any:
    _torch, _safe_open, _load_file, _save_file, modules = _runtime_modules()
    _critic_core, critic_trainer = modules
    return critic_trainer.TrainerConfig(
        learning_rate=FIXED_LEARNING_RATE,
        weight_decay=FIXED_WEIGHT_DECAY,
        maximum_gradient_norm=FIXED_MAXIMUM_GRADIENT_NORM,
        global_margin=FIXED_GLOBAL_MARGIN,
        milestone_margin=FIXED_MILESTONE_MARGIN,
        ranking_temperature=FIXED_RANKING_TEMPERATURE,
    )


def train_fixed_fit_cells(
    critic: Any, fit_episodes: Sequence[LoadedEpisode]
) -> dict[str, Any]:
    """Execute exact step 1..200; both fit cells enter every aggregate loss."""

    _torch, _safe_open, _load_file, _save_file, modules = _runtime_modules()
    _critic_core, critic_trainer = modules
    if len(fit_episodes) != 2 or any(row.split != "fit" for row in fit_episodes):
        raise StarcCriticPilotError("training requires exactly the two fit cells")
    batches = _make_group_batches(fit_episodes)
    expected_ids = tuple(row.episode_id for row in fit_episodes)
    config = _trainer_config()
    optimizer = critic_trainer.build_critic_optimizer(critic, config=config)
    trace = []
    for step_index in range(1, FIXED_OPTIMIZER_STEPS + 1):
        row = critic_trainer.train_critic_groups_one_step(
            critic, optimizer, batches, config=config
        )
        if (
            row.episode_ids != expected_ids
            or row.optimizer_step_performed is not True
            or row.editor_parameter_present is not False
        ):
            raise StarcCriticPilotError("critic training step exceeded its scope")
        trace.append(
            {
                "step": step_index,
                "loss": row.loss,
                "gradient_norm_before_clip": row.gradient_norm,
                "minimum_fit_group_margin": row.minimum_group_margin,
                "episode_ids": list(row.episode_ids),
            }
        )
    if len(trace) != FIXED_OPTIMIZER_STEPS or trace[-1]["step"] != 200:
        raise StarcCriticPilotError("critic did not complete exact 200 steps")
    return _seal(
        {
            "schema_version": TRACE_SCHEMA,
            "fixed_hyperparameters": dict(FIXED_HYPERPARAMETERS),
            "fit_episode_order": list(expected_ids),
            "fit_artifact_count": sum(row.loaded_artifact_count for row in fit_episodes),
            "optimizer_step_count": len(trace),
            "both_fit_cells_consumed_every_step": True,
            "confirmation_manifest_metadata_authenticated_before_fit": True,
            "confirmation_tensor_artifacts_opened_before_fit_complete": False,
            "confirmation_samples_consumed_by_optimizer": False,
            "checkpoint_selection": "final_step_200_only",
            "best_checkpoint_saved": False,
            "early_stopping_performed": False,
            "editor_parameter_present": False,
            "steps": trace,
        }
    )


def _score_loaded_episodes(
    critic: Any, episodes: Sequence[LoadedEpisode]
) -> dict[str, dict[str, dict[str, Any]]]:
    torch, _safe_open, _load_file, _save_file, modules = _runtime_modules()
    critic_core, _critic_trainer = modules
    if tuple(critic_core.MILESTONE_NAMES) != FIXED_MILESTONE_NAMES:
        raise StarcCriticPilotError("critic milestone registry differs")
    critic.eval()
    result: dict[str, dict[str, dict[str, Any]]] = {}
    with torch.inference_mode():
        for episode in episodes:
            role_scores: dict[str, dict[str, Any]] = {}
            for role in data_contract.ARM_ROLES:
                output = critic.forward_sketched_residual(
                    episode.tensors_by_role[role]
                )
                if tuple(output.score.shape) != (1,) or tuple(
                    output.milestone_scores.shape
                ) != (1, len(critic_core.MILESTONE_NAMES)):
                    raise StarcCriticPilotError("critic scalar/milestone shape differs")
                scalar = float(output.score[0].detach().cpu().item())
                milestones = [
                    float(value)
                    for value in output.milestone_scores[0].detach().cpu().tolist()
                ]
                if not math.isfinite(scalar) or any(
                    not math.isfinite(value) for value in milestones
                ):
                    raise StarcCriticPilotError("critic produced a non-finite heldout score")
                role_scores[role] = {
                    "score": scalar,
                    "milestone_scores": dict(
                        zip(FIXED_MILESTONE_NAMES, milestones)
                    ),
                }
            result[episode.episode_id] = role_scores
    return result


def make_heldout_margin_gate(
    confirmation_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    expected_episode_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the non-compensating 2x12 margin gate without averaging roles."""

    episode_ids = tuple(expected_episode_ids)
    if len(episode_ids) != 2 or len(set(episode_ids)) != 2:
        raise StarcCriticPilotError("heldout gate requires exactly two episodes")
    if set(confirmation_outputs) != set(episode_ids):
        raise StarcCriticPilotError("heldout output episode closure differs")
    margins = []
    failures = []
    scalar_scores: dict[str, dict[str, float]] = {}
    for episode_id in episode_ids:
        outputs = confirmation_outputs[episode_id]
        # JSON objects are emitted with sorted keys, so serialized replay must
        # authenticate role closure and then consume the preregistered order;
        # insertion order is not a scientific property of the receipt.
        if set(outputs) != set(data_contract.ARM_ROLES):
            raise StarcCriticPilotError(f"{episode_id} heldout role closure differs")
        scores = {}
        for role in data_contract.ARM_ROLES:
            role_output = outputs[role]
            if not isinstance(role_output, Mapping) or set(role_output) != {
                "score",
                "milestone_scores",
            }:
                raise StarcCriticPilotError("heldout role output closure differs")
            milestones = role_output["milestone_scores"]
            if (
                not isinstance(milestones, Mapping)
                or set(milestones) != set(FIXED_MILESTONE_NAMES)
                or any(
                    isinstance(milestones[name], bool)
                    or not isinstance(milestones[name], (int, float))
                    or not math.isfinite(float(milestones[name]))
                    for name in FIXED_MILESTONE_NAMES
                )
            ):
                raise StarcCriticPilotError("heldout milestone output closure differs")
            score = role_output["score"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise StarcCriticPilotError("heldout score must be finite")
            scores[role] = float(score)
        scalar_scores[episode_id] = scores
        for role in data_contract.NEGATIVE_ROLES:
            margin = scores["positive"] - scores[role]
            passed = margin >= FIXED_HELDOUT_MINIMUM_MARGIN
            margins.append(
                {
                    "episode_id": episode_id,
                    "negative_role": role,
                    "positive_score": scores["positive"],
                    "negative_score": scores[role],
                    "margin": margin,
                    "minimum_required_margin": FIXED_HELDOUT_MINIMUM_MARGIN,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"margin:{episode_id}:{role}")
    if len(margins) != 24:
        raise StarcCriticPilotError("heldout gate did not produce exact 24 margins")
    return {
        "schema_version": HELDOUT_MARGIN_SCHEMA,
        "confirmation_episode_order": list(episode_ids),
        "confirmation_role_order": list(data_contract.ARM_ROLES),
        "minimum_margin": FIXED_HELDOUT_MINIMUM_MARGIN,
        "scalar_scores": scalar_scores,
        "margins": margins,
        "margin_count": 24,
        "passed_margin_count": sum(row["passed"] for row in margins),
        "all_24_role_margins_passed": not failures,
        "role_margins_averaged_or_compensated": False,
        "confirmation_evaluated_once_after_checkpoint_reload_and_freeze": True,
        "each_confirmation_tensor_artifact_read_once": True,
        "confirmation_used_for_checkpoint_threshold_layer_or_hyperparameter_selection": False,
        "worth_fixed_topup_generation_recommended_by_this_gate": not failures,
        "scientific_critic_claim_authorized": False,
        "editor_optimizer_authorized": False,
        "failure_reasons": failures,
    }


def _state_dict_cpu(critic: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in critic.state_dict().items()
        if name not in NON_HEAD_STATE_KEYS
    }


def _load_head_state(critic: Any, state: Mapping[str, Any]) -> None:
    incompatible = critic.load_state_dict(state, strict=False)
    if (
        tuple(incompatible.missing_keys) != NON_HEAD_STATE_KEYS
        or incompatible.unexpected_keys
    ):
        raise StarcCriticPilotError(
            "geometry-neutral head checkpoint key closure differs"
        )


def _assert_exact_state_parity(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> None:
    torch, _safe_open, _load_file, _save_file, _modules = _runtime_modules()
    # ``safetensors`` owns the iteration order returned by ``load_file`` and may
    # canonicalize it independently of the module's ``state_dict`` insertion
    # order.  Order is not checkpoint state; the exact key closure and every
    # tensor value are.  The content digest below is likewise name-sorted.
    if set(expected) != set(observed):
        raise StarcCriticPilotError(
            "checkpoint state key closure differs after reload"
        )
    for name in sorted(expected):
        if (
            expected[name].dtype != observed[name].dtype
            or expected[name].shape != observed[name].shape
            or not torch.equal(expected[name].cpu(), observed[name].cpu())
        ):
            raise StarcCriticPilotError(
                f"checkpoint tensor {name} differs after reload"
            )


def save_reload_final_checkpoint(
    critic: Any,
    *,
    output_dir: Path,
    device: str,
    config_receipt: Mapping[str, Any],
    fit_episodes: Sequence[LoadedEpisode],
) -> tuple[Any, dict[str, Any]]:
    """Save only step 200, reload fresh, prove state and fit-score parity."""

    torch, safe_open, load_file, save_file, modules = _runtime_modules()
    _critic_core, critic_trainer = modules
    critic.eval()
    expected_state = _state_dict_cpu(critic)
    state_digest = checkpoint_state_content_digest(expected_state)
    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    if checkpoint_path.exists() or checkpoint_path.is_symlink():
        raise StarcCriticPilotError("final checkpoint path already exists")
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_receipt_digest": config_receipt["receipt_digest"],
        "checkpoint_state_content_digest": state_digest,
        "optimizer_step": str(FIXED_OPTIMIZER_STEPS),
        "selection": "final_step_200_only",
    }
    save_file(expected_state, str(checkpoint_path), metadata=metadata)
    checkpoint_file_sha = file_sha256(checkpoint_path)
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as opened:
        if opened.metadata() != metadata:
            raise StarcCriticPilotError("checkpoint safetensors metadata differs")
        if set(opened.keys()) != set(expected_state):
            raise StarcCriticPilotError("checkpoint safetensors key closure differs")
    loaded_state = load_file(str(checkpoint_path), device="cpu")
    _assert_exact_state_parity(expected_state, loaded_state)
    if checkpoint_state_content_digest(loaded_state) != state_digest:
        raise StarcCriticPilotError("checkpoint content digest differs after reload")

    reference_scores = _score_loaded_episodes(critic, fit_episodes)
    reloaded = _build_fresh_critic(device=device)
    _load_head_state(reloaded, loaded_state)
    _assert_exact_state_parity(expected_state, _state_dict_cpu(reloaded))
    reloaded_scores = _score_loaded_episodes(reloaded, fit_episodes)
    if canonical_json_bytes(reference_scores) != canonical_json_bytes(reloaded_scores):
        raise StarcCriticPilotError("checkpoint reload fit-score parity differs")
    critic_trainer.freeze_fitted_critic_for_reward(reloaded)
    if reloaded.training or any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise StarcCriticPilotError("reloaded critic did not freeze")
    receipt = _seal(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": checkpoint_file_sha,
            "checkpoint_state_content_digest": state_digest,
            "checkpoint_tensor_count": len(expected_state),
            "checkpoint_scope": "geometry_neutral_pre_sketched_critic_head_only",
            "excluded_constructor_buffer_keys": list(NON_HEAD_STATE_KEYS),
            "config_receipt_digest": config_receipt["receipt_digest"],
            "optimizer_step": FIXED_OPTIMIZER_STEPS,
            "only_final_checkpoint_saved": True,
            "best_checkpoint_saved": False,
            "confirmation_sample_seen_before_checkpoint_save": False,
            "state_tensor_byte_parity_after_fresh_load": True,
            "fit_score_parity_after_fresh_load": True,
            "critic_frozen_after_reload": True,
            "editor_checkpoint_or_parameter_present": False,
            "editor_optimizer_authorized": False,
        }
    )
    return reloaded, receipt


def _validate_live_vjp_source_archive(
    *,
    source_path_value: Any,
    source_file_sha256: Any,
    archive_path_value: Any,
    archive_file_sha256: Any,
    archive_member_path: Any,
    archive_member_sha256: Any,
    source_git_revision: Any,
) -> dict[str, Any]:
    """Re-open the git archive and bind its bridge member to executing code."""

    expected_source_path = METHOD_ROOT / "starch_live_vjp_bridge_v1.py"
    source_path = _plain_file(source_path_value, label="live VJP bridge source")
    source_sha = _sha256(
        source_file_sha256, label="live VJP bridge source SHA-256"
    )
    if source_path != expected_source_path or file_sha256(source_path) != source_sha:
        raise StarcCriticPilotError("live VJP executing bridge source differs")
    if archive_member_path != LIVE_VJP_BRIDGE_ARCHIVE_MEMBER:
        raise StarcCriticPilotError("live VJP bridge archive member path differs")
    member_sha = _sha256(
        archive_member_sha256, label="live VJP archive bridge member SHA-256"
    )
    if member_sha != source_sha:
        raise StarcCriticPilotError("live VJP archive member/source digest differs")
    if (
        not isinstance(source_git_revision, str)
        or _SHA1_RE.fullmatch(source_git_revision) is None
    ):
        raise StarcCriticPilotError("live VJP source git revision must be 40-hex")
    archive = _plain_file(
        archive_path_value, label="live VJP bridge source archive"
    )
    archive_sha = _sha256(
        archive_file_sha256, label="live VJP bridge source archive SHA-256"
    )
    if file_sha256(archive) != archive_sha:
        raise StarcCriticPilotError("live VJP bridge source archive SHA-256 differs")
    matches = []
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            revision = handle.pax_headers.get("comment")
            for member in handle.getmembers():
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or member.isfifo()
                ):
                    raise StarcCriticPilotError(
                        "live VJP bridge source archive contains an unsafe member"
                    )
                if pure.as_posix() == LIVE_VJP_BRIDGE_ARCHIVE_MEMBER:
                    matches.append(member)
            if revision != source_git_revision:
                raise StarcCriticPilotError(
                    "live VJP bridge source archive revision differs"
                )
            if len(matches) != 1 or not matches[0].isfile():
                raise StarcCriticPilotError(
                    "live VJP bridge source archive lacks one plain bridge member"
                )
            stream = handle.extractfile(matches[0])
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != member_sha:
                raise StarcCriticPilotError(
                    "live VJP bridge source archive member bytes differ"
                )
    except StarcCriticPilotError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise StarcCriticPilotError(
            "live VJP bridge source archive cannot be authenticated"
        ) from error
    return {
        "source_path": str(source_path),
        "source_file_sha256": source_sha,
        "source_archive_path": str(archive),
        "source_archive_file_sha256": archive_sha,
        "source_archive_bridge_member_path": LIVE_VJP_BRIDGE_ARCHIVE_MEMBER,
        "source_archive_bridge_member_sha256": member_sha,
        "source_git_revision": source_git_revision,
    }


def _validate_bernini_checkpoint_content(
    *,
    checkpoint_root_value: Any,
    manifest_path_value: Any,
    manifest_file_sha256: Any,
) -> dict[str, Any]:
    """Hash every non-cache Bernini file against the pinned content manifest."""

    checkpoint_root = _plain_directory(
        checkpoint_root_value, label="live VJP Bernini checkpoint root"
    )
    manifest_path = _plain_file(
        manifest_path_value, label="live VJP Bernini checkpoint content manifest"
    )
    manifest_sha = _sha256(
        manifest_file_sha256,
        label="live VJP Bernini checkpoint content manifest SHA-256",
    )
    if (
        manifest_sha != BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or file_sha256(manifest_path) != manifest_sha
    ):
        raise StarcCriticPilotError(
            "live VJP Bernini checkpoint content manifest differs from pinned identity"
        )
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise StarcCriticPilotError(
            "live VJP Bernini checkpoint content manifest cannot be read"
        ) from error
    if len(lines) != BERNINI_CHECKPOINT_CONTENT_FILE_COUNT:
        raise StarcCriticPilotError(
            "live VJP Bernini checkpoint content manifest file count differs"
        )
    expected: dict[str, str] = {}
    pattern = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")
    for line in lines:
        match = pattern.fullmatch(line)
        if match is None:
            raise StarcCriticPilotError(
                "live VJP checkpoint manifest line is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise StarcCriticPilotError(
                "live VJP checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in expected:
            raise StarcCriticPilotError(
                "live VJP checkpoint manifest contains an empty or duplicate path"
            )
        expected[normalized] = digest

    actual_paths: set[str] = set()
    try:
        descendants = tuple(checkpoint_root.rglob("*"))
    except OSError as error:
        raise StarcCriticPilotError(
            "live VJP Bernini checkpoint content cannot be enumerated"
        ) from error
    for path in descendants:
        relative = path.relative_to(checkpoint_root)
        if ".cache" in relative.parts:
            continue
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise StarcCriticPilotError(
                "live VJP Bernini checkpoint content cannot be inspected"
            ) from error
        if stat.S_ISLNK(mode):
            raise StarcCriticPilotError(
                "live VJP Bernini checkpoint contains a non-cache symlink"
            )
        if stat.S_ISREG(mode):
            actual_paths.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise StarcCriticPilotError(
                "live VJP Bernini checkpoint contains a non-regular entry"
            )
    if actual_paths != set(expected):
        raise StarcCriticPilotError(
            "live VJP Bernini checkpoint file closure differs from content manifest"
        )
    verified_entries = []
    for relative in sorted(expected):
        path = _plain_file(
            checkpoint_root / relative,
            label=f"live VJP Bernini checkpoint file {relative}",
            beneath=checkpoint_root,
        )
        observed = file_sha256(path)
        if observed != expected[relative]:
            raise StarcCriticPilotError(
                f"live VJP Bernini checkpoint content hash differs: {relative}"
            )
        verified_entries.append({"path": relative, "sha256": observed})
    return {
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_content_manifest_path": str(manifest_path),
        "checkpoint_content_manifest_file_sha256": manifest_sha,
        "checkpoint_content_verified_file_count": len(verified_entries),
        "checkpoint_content_verified_entries_digest": object_sha256(verified_entries),
    }


def _validate_live_vjp_candidate_manifest(
    clean: Mapping[str, Any], *, clean_shape: Sequence[int]
) -> dict[str, Any]:
    raw, path, observed_sha = _read_json_file(
        clean.get("candidate_manifest_path"),
        label="live VJP current candidate manifest",
        expected_sha256=clean.get("candidate_manifest_file_sha256"),
    )
    manifest = _validate_sealed_manifest(
        raw,
        schema=LIVE_VJP_CANDIDATE_SCHEMA,
        label="live VJP current candidate manifest",
    )
    expected_fields = {
        "schema_version",
        "candidate_id",
        "source_video_sha256",
        "instruction_sha256",
        "current_clean_latent_tensor_sha256",
        "latent_shape",
        "patch_order",
        "external_inference_inputs",
        "auxiliary_spatial_inputs",
        "receipt_digest",
    }
    if set(manifest) != expected_fields:
        raise StarcCriticPilotError("live VJP candidate manifest field closure differs")
    _safe_id(manifest.get("candidate_id"), label="live VJP candidate ID")
    for name in (
        "source_video_sha256",
        "instruction_sha256",
        "current_clean_latent_tensor_sha256",
    ):
        _sha256(manifest.get(name), label=f"live VJP candidate {name}")
    if (
        manifest.get("candidate_id") != clean.get("candidate_id")
        or manifest.get("source_video_sha256") != clean.get("source_video_sha256")
        or manifest.get("instruction_sha256") != clean.get("instruction_sha256")
        or manifest.get("current_clean_latent_tensor_sha256")
        != clean.get("tensor_sha256")
        or manifest.get("latent_shape") != list(clean_shape)
        or manifest.get("patch_order") != "phase_major_then_patch_row_major"
        or manifest.get("external_inference_inputs")
        != ["source_video", "instruction"]
        or manifest.get("auxiliary_spatial_inputs") != []
        or manifest.get("receipt_digest")
        != clean.get("candidate_manifest_receipt_digest")
        or str(path) != clean.get("candidate_manifest_path")
        or observed_sha != clean.get("candidate_manifest_file_sha256")
    ):
        raise StarcCriticPilotError("live VJP candidate manifest binding differs")
    return manifest


def validate_live_vjp_receipt(
    path_value: Optional[str | Path],
    expected_sha256: Optional[str],
    *,
    graph: Optional[PilotManifestGraph] = None,
    config_receipt: Optional[Mapping[str, Any]] = None,
    checkpoint_receipt: Optional[Mapping[str, Any]] = None,
    config_receipt_path: Optional[str | Path] = None,
    config_receipt_file_sha256: Optional[str] = None,
    checkpoint_receipt_path: Optional[str | Path] = None,
    checkpoint_receipt_file_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Validate a checkpoint-bound live VJP, or fail closed when absent.

    The old ``bernini-ltec-current-clean-latent-gradient-audit-v1`` surface is
    intentionally rejected: shape/norm alone can be replayed from a different
    critic or frozen model.  This composite receipt binds the exact critic,
    materializer, bridge source, frozen Bernini runtime, current clean latent,
    same-object action/no-op query, differentiable SP4 collective, and gradient.
    """

    if path_value is None and expected_sha256 is None:
        return {
            "provided": False,
            "passed": False,
            "checkpoint_and_runtime_bound": False,
            "reason": "live_current_rv2v_input_vjp_composite_receipt_missing",
            "receipt_path": None,
            "receipt_file_sha256": None,
        }
    if path_value is None or expected_sha256 is None:
        raise StarcCriticPilotError(
            "live VJP path and expected SHA-256 must be supplied together"
        )
    if (
        not isinstance(graph, PilotManifestGraph)
        or not isinstance(config_receipt, Mapping)
        or not isinstance(checkpoint_receipt, Mapping)
        or config_receipt_path is None
        or config_receipt_file_sha256 is None
        or checkpoint_receipt_path is None
        or checkpoint_receipt_file_sha256 is None
    ):
        raise StarcCriticPilotError(
            "live VJP validation requires current graph/config/checkpoint file bindings"
        )
    raw, path, observed_sha = _read_json_file(
        path_value,
        label="live current-RV2V input-VJP composite receipt",
        expected_sha256=expected_sha256,
    )
    receipt = _validate_sealed_manifest(
        raw,
        schema=LIVE_VJP_BINDING_SCHEMA,
        label="live current-RV2V input-VJP composite receipt",
    )
    expected_top_fields = {
        "schema_version",
        "critic_binding",
        "materializer_binding",
        "live_bridge_binding",
        "current_rv2v_clean_latent",
        "same_state_hidden_query",
        "sp4_differentiable_collective_proof",
        "gradient_audit",
        "generated_t2v_target_consumed",
        "editor_parameter_or_optimizer_present",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "receipt_digest",
    }
    if set(receipt) != expected_top_fields:
        raise StarcCriticPilotError("live VJP composite field closure differs")

    critic_binding = receipt["critic_binding"]
    if not isinstance(critic_binding, Mapping) or set(critic_binding) != {
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_state_content_digest",
        "checkpoint_receipt_path",
        "checkpoint_receipt_file_sha256",
        "checkpoint_receipt_digest",
        "config_receipt_path",
        "config_receipt_file_sha256",
        "config_receipt_digest",
    }:
        raise StarcCriticPilotError("live VJP critic binding closure differs")
    for name in (
        "checkpoint_file_sha256",
        "checkpoint_state_content_digest",
        "checkpoint_receipt_file_sha256",
        "checkpoint_receipt_digest",
        "config_receipt_file_sha256",
        "config_receipt_digest",
    ):
        _sha256(critic_binding[name], label=f"live VJP critic {name}")
    critic_checkpoint_path = _plain_file(
        critic_binding["checkpoint_path"], label="live VJP critic checkpoint"
    )
    critic_checkpoint_receipt_path = _plain_file(
        critic_binding["checkpoint_receipt_path"],
        label="live VJP critic checkpoint receipt",
    )
    critic_config_receipt_path = _plain_file(
        critic_binding["config_receipt_path"],
        label="live VJP critic config receipt",
    )
    if (
        str(critic_checkpoint_path) != checkpoint_receipt.get("checkpoint_path")
        or file_sha256(critic_checkpoint_path)
        != critic_binding["checkpoint_file_sha256"]
        or str(critic_checkpoint_receipt_path) != str(checkpoint_receipt_path)
        or file_sha256(critic_checkpoint_receipt_path)
        != critic_binding["checkpoint_receipt_file_sha256"]
        or critic_binding["checkpoint_receipt_file_sha256"]
        != checkpoint_receipt_file_sha256
        or critic_binding["checkpoint_receipt_digest"]
        != checkpoint_receipt.get("receipt_digest")
        or str(critic_config_receipt_path) != str(config_receipt_path)
        or file_sha256(critic_config_receipt_path)
        != critic_binding["config_receipt_file_sha256"]
        or critic_binding["config_receipt_file_sha256"]
        != config_receipt_file_sha256
        or critic_binding["config_receipt_digest"]
        != config_receipt.get("receipt_digest")
        or critic_binding["checkpoint_file_sha256"]
        != checkpoint_receipt.get("checkpoint_file_sha256")
        or critic_binding["checkpoint_state_content_digest"]
        != checkpoint_receipt.get("checkpoint_state_content_digest")
    ):
        raise StarcCriticPilotError("live VJP belongs to another critic checkpoint")

    materializer_binding = receipt["materializer_binding"]
    if not isinstance(materializer_binding, Mapping) or set(materializer_binding) != {
        "master_path",
        "master_file_sha256",
        "master_receipt_digest",
        "population_content_digest",
    }:
        raise StarcCriticPilotError("live VJP materializer binding closure differs")
    for name in (
        "master_file_sha256",
        "master_receipt_digest",
        "population_content_digest",
    ):
        _sha256(materializer_binding[name], label=f"live VJP materializer {name}")
    if materializer_binding != {
        "master_path": str(graph.master_path),
        "master_file_sha256": graph.master_file_sha256,
        "master_receipt_digest": graph.master_receipt_digest,
        "population_content_digest": graph.content_digest,
    }:
        raise StarcCriticPilotError("live VJP belongs to another materialized population")

    bridge = receipt["live_bridge_binding"]
    expected_bridge_fields = {
        "source_path",
        "source_file_sha256",
        "source_archive_path",
        "source_archive_file_sha256",
        "source_archive_bridge_member_path",
        "source_archive_bridge_member_sha256",
        "source_git_revision",
        "backend_id",
        "bernini_commit",
        "veomni_commit",
        "checkpoint_root",
        "checkpoint_tree_sha256",
        "checkpoint_content_manifest_path",
        "checkpoint_content_manifest_file_sha256",
        "checkpoint_content_verified_file_count",
        "checkpoint_content_verified_entries_digest",
        "adapter_enabled",
        "frozen_bernini_and_critic",
    }
    if not isinstance(bridge, Mapping) or set(bridge) != expected_bridge_fields:
        raise StarcCriticPilotError("live VJP bridge binding closure differs")
    source_authentication = _validate_live_vjp_source_archive(
        source_path_value=bridge["source_path"],
        source_file_sha256=bridge["source_file_sha256"],
        archive_path_value=bridge["source_archive_path"],
        archive_file_sha256=bridge["source_archive_file_sha256"],
        archive_member_path=bridge["source_archive_bridge_member_path"],
        archive_member_sha256=bridge["source_archive_bridge_member_sha256"],
        source_git_revision=bridge["source_git_revision"],
    )
    if any(bridge[name] != value for name, value in source_authentication.items()):
        raise StarcCriticPilotError("live VJP bridge source authentication differs")
    checkpoint_content = _validate_bernini_checkpoint_content(
        checkpoint_root_value=bridge["checkpoint_root"],
        manifest_path_value=bridge["checkpoint_content_manifest_path"],
        manifest_file_sha256=bridge["checkpoint_content_manifest_file_sha256"],
    )
    if any(bridge[name] != value for name, value in checkpoint_content.items()):
        raise StarcCriticPilotError("live VJP checkpoint content evidence differs")
    checkpoint_tree_sha = _sha256(
        bridge["checkpoint_tree_sha256"], label="frozen checkpoint tree SHA-256"
    )
    _sha256(
        bridge["checkpoint_content_verified_entries_digest"],
        label="frozen checkpoint verified entries digest",
    )
    if (
        bridge["backend_id"] != LIVE_VJP_BACKEND_ID
        or bridge["bernini_commit"] != BERNINI_OFFICIAL_COMMIT
        or bridge["veomni_commit"] != VEOMNI_TESTED_COMMIT
        or checkpoint_tree_sha != BERNINI_CHECKPOINT_TREE_SHA256
        or bridge["checkpoint_content_manifest_file_sha256"]
        != BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or bridge["checkpoint_content_verified_file_count"]
        != BERNINI_CHECKPOINT_CONTENT_FILE_COUNT
        or bridge["adapter_enabled"] is not False
        or bridge["frozen_bernini_and_critic"] is not True
    ):
        raise StarcCriticPilotError("live VJP bridge runtime differs")

    clean = receipt["current_rv2v_clean_latent"]
    if not isinstance(clean, Mapping) or set(clean) != {
        "candidate_id",
        "candidate_manifest_path",
        "candidate_manifest_file_sha256",
        "candidate_manifest_receipt_digest",
        "source_video_sha256",
        "instruction_sha256",
        "tensor_sha256",
        "tensor_shape",
        "tensor_dtype",
        "requires_grad",
        "generated_t2v_owner_or_target",
        "patch_grid_height_width",
        "patch_positions",
        "spatial_sketch_binding",
    }:
        raise StarcCriticPilotError("live VJP clean-latent binding closure differs")
    _safe_id(clean["candidate_id"], label="live VJP current candidate ID")
    for name in (
        "candidate_manifest_file_sha256",
        "candidate_manifest_receipt_digest",
        "source_video_sha256",
        "instruction_sha256",
        "tensor_sha256",
    ):
        _sha256(clean[name], label=f"live VJP current candidate {name}")
    clean_shape = clean["tensor_shape"]
    if (
        not isinstance(clean_shape, list)
        or len(clean_shape) != 5
        or clean_shape[:3] != [1, 16, 21]
        or tuple(clean_shape[3:]) not in REGISTERED_LATENT_TO_PATCH_GRID
    ):
        raise StarcCriticPilotError("live VJP current clean latent geometry differs")
    candidate_grid = REGISTERED_LATENT_TO_PATCH_GRID[tuple(clean_shape[3:])]
    candidate_positions = candidate_grid[0] * candidate_grid[1]
    candidate_sketch = reconstruct_geometry_spatial_sketch_binding(*candidate_grid)
    _validate_live_vjp_candidate_manifest(clean, clean_shape=clean_shape)
    if (
        clean["tensor_dtype"] != "torch.float32"
        or clean["requires_grad"] is not True
        or clean["generated_t2v_owner_or_target"] is not False
        or clean["patch_grid_height_width"] != list(candidate_grid)
        or clean["patch_positions"] != candidate_positions
        or not _json_exact_equal(
            clean["spatial_sketch_binding"], candidate_sketch
        )
    ):
        raise StarcCriticPilotError("live VJP current clean latent differs")

    query = receipt["same_state_hidden_query"]
    expected_query_fields = {
        "native_schedule_index",
        "physical_sigma",
        "native_timestep",
        "hook_coordinate",
        "action_text_tensor_sha256",
        "noop_text_tensor_sha256",
        "action_x_sigma_tensor_sha256",
        "noop_x_sigma_tensor_sha256",
        "action_and_noop_received_same_python_x_sigma_object",
        "action_and_noop_x_sigma_value_equal",
        "source_condition_consumed",
    }
    if not isinstance(query, Mapping) or set(query) != expected_query_fields:
        raise StarcCriticPilotError("live VJP same-state query closure differs")
    for name in (
        "action_text_tensor_sha256",
        "noop_text_tensor_sha256",
        "action_x_sigma_tensor_sha256",
        "noop_x_sigma_tensor_sha256",
    ):
        _sha256(query[name], label=f"live VJP {name}")
    if (
        query["native_schedule_index"]
        != data_contract.PILOT_HIDDEN_QUERY["native_schedule_index"]
        or not isinstance(query["physical_sigma"], (int, float))
        or isinstance(query["physical_sigma"], bool)
        or float(query["physical_sigma"]).hex()
        != float(data_contract.PILOT_HIDDEN_QUERY["sigma"]).hex()
        or query["native_timestep"]
        != data_contract.PILOT_HIDDEN_QUERY["native_timestep"]
        or query["hook_coordinate"]
        != data_contract.PILOT_HIDDEN_QUERY["hook_coordinate"]
        or query["action_text_tensor_sha256"]
        == query["noop_text_tensor_sha256"]
        or query["action_x_sigma_tensor_sha256"]
        != query["noop_x_sigma_tensor_sha256"]
        or query["action_and_noop_received_same_python_x_sigma_object"] is not True
        or query["action_and_noop_x_sigma_value_equal"] is not True
        or query["source_condition_consumed"] is not False
    ):
        raise StarcCriticPilotError("live VJP same-state hidden query differs")

    collective = receipt["sp4_differentiable_collective_proof"]
    expected_collective_fields = {
        "world_size",
        "implementation",
        "rank_local_hidden_global_shape",
        "autograd_collective_tensor_shape",
        "dynamic_spatial_sketch_critic_tensor_sha256",
        "preflight_replica_contract_digest",
        "replica_graph_input_consensus_observed",
        "replicated_score_consensus_digest",
        "all_rank_hidden_backward_evidence_digest",
        "forward_autograd_connected",
        "backward_reached_all_rank_local_hidden_shards",
        "detached_or_object_collective_used",
        "ordered_rank_hidden_backward_evidence",
        "rank_gradient_tensor_digests",
        "proof_digest",
    }
    if not isinstance(collective, Mapping) or set(collective) != expected_collective_fields:
        raise StarcCriticPilotError("live VJP SP4 proof closure differs")
    rank_digests = collective["rank_gradient_tensor_digests"]
    if not isinstance(rank_digests, list) or len(rank_digests) != 4:
        raise StarcCriticPilotError("live VJP SP4 rank-gradient closure differs")
    for index, digest in enumerate(rank_digests):
        _sha256(digest, label=f"live VJP SP4 rank {index} gradient digest")
    ordered_evidence = collective["ordered_rank_hidden_backward_evidence"]
    evidence_fields = {
        "rank",
        "shape",
        "action_digest",
        "noop_digest",
        "norm",
        "finite_nonzero",
        "action_is_exact_negative_noop",
    }
    if (
        not isinstance(ordered_evidence, list)
        or len(ordered_evidence) != 4
        or any(not isinstance(row, Mapping) for row in ordered_evidence)
        or any(set(row) != evidence_fields for row in ordered_evidence)
    ):
        raise StarcCriticPilotError("live VJP SP4 hidden-backward evidence differs")
    for index, row in enumerate(ordered_evidence):
        _sha256(
            row["action_digest"],
            label=f"live VJP SP4 rank {index} action hidden gradient digest",
        )
        _sha256(
            row["noop_digest"],
            label=f"live VJP SP4 rank {index} no-op hidden gradient digest",
        )
        norm = row["norm"]
        if (
            row["rank"] != index
            or row["shape"] != [1, 21, 16, 1536]
            or isinstance(norm, bool)
            or not isinstance(norm, (int, float))
            or not math.isfinite(float(norm))
            or float(norm) <= 0.0
            or row["finite_nonzero"] is not True
            or row["action_is_exact_negative_noop"] is not True
            or row["action_digest"] != rank_digests[index]
        ):
            raise StarcCriticPilotError(
                "live VJP SP4 per-rank hidden backward evidence failed"
            )
    sketch_digest = _sha256(
        collective["dynamic_spatial_sketch_critic_tensor_sha256"],
        label="live VJP SP4 dynamic sketch critic tensor SHA-256",
    )
    for name in (
        "preflight_replica_contract_digest",
        "replicated_score_consensus_digest",
        "all_rank_hidden_backward_evidence_digest",
    ):
        _sha256(collective[name], label=f"live VJP SP4 {name}")
    expected_hidden_backward_digest = object_sha256(
        {
            "schema_version": "bernini-starc-all-rank-hidden-vjp-v2",
            "ordered_rank_evidence": ordered_evidence,
        }
    )
    proof_digest = _sha256(
        collective["proof_digest"], label="live VJP SP4 proof digest"
    )
    proof_unsigned = dict(collective)
    proof_unsigned.pop("proof_digest")
    if (
        object_sha256(proof_unsigned) != proof_digest
        or collective["world_size"] != 4
        or collective["implementation"] != LIVE_VJP_SP4_IMPLEMENTATION
        or collective["rank_local_hidden_global_shape"]
        != [1, 21, candidate_positions, 1536]
        or collective["autograd_collective_tensor_shape"] != [1, 21, 16, 1536]
        or sketch_digest != candidate_sketch["critic_tensor_sha256"]
        or collective["replica_graph_input_consensus_observed"] is not True
        or collective["all_rank_hidden_backward_evidence_digest"]
        != expected_hidden_backward_digest
        or collective["forward_autograd_connected"] is not True
        or collective["backward_reached_all_rank_local_hidden_shards"] is not True
        or collective["detached_or_object_collective_used"] is not False
    ):
        raise StarcCriticPilotError("live VJP SP4 differentiable collective failed")

    gradient = receipt["gradient_audit"]
    expected_gradient_fields = {
        "tensor_sha256",
        "tensor_shape",
        "tensor_dtype",
        "gradient_norm",
        "minimum_norm",
        "finite",
        "nonzero",
        "reached_current_rv2v_clean_latent",
    }
    if not isinstance(gradient, Mapping) or set(gradient) != expected_gradient_fields:
        raise StarcCriticPilotError("live VJP gradient audit closure differs")
    _sha256(gradient["tensor_sha256"], label="live VJP gradient tensor SHA-256")
    gradient_norm = gradient["gradient_norm"]
    minimum_norm = gradient["minimum_norm"]
    if (
        gradient["tensor_shape"] != clean_shape
        or gradient["tensor_dtype"] != "torch.float32"
        or isinstance(gradient_norm, bool)
        or not isinstance(gradient_norm, (int, float))
        or not math.isfinite(float(gradient_norm))
        or float(gradient_norm) <= 0.0
        or isinstance(minimum_norm, bool)
        or not isinstance(minimum_norm, (int, float))
        or not math.isfinite(float(minimum_norm))
        or float(minimum_norm) <= 0.0
        or float(gradient_norm) < float(minimum_norm)
        or gradient["finite"] is not True
        or gradient["nonzero"] is not True
        or gradient["reached_current_rv2v_clean_latent"] is not True
        or receipt["generated_t2v_target_consumed"] is not False
        or receipt["editor_parameter_or_optimizer_present"] is not False
        or receipt["editor_optimizer_authorized"] is not False
        or receipt["scientific_critic_claim_authorized"] is not False
    ):
        raise StarcCriticPilotError("live current-RV2V input-VJP composite failed")
    return {
        "provided": True,
        "passed": True,
        "checkpoint_and_runtime_bound": True,
        "reason": None,
        "receipt_path": str(path),
        "receipt_file_sha256": observed_sha,
        "receipt_digest": receipt["receipt_digest"],
        "live_bridge_source_file_sha256": bridge["source_file_sha256"],
        "live_bridge_source_archive_file_sha256": bridge[
            "source_archive_file_sha256"
        ],
        "frozen_bernini_commit": bridge["bernini_commit"],
        "frozen_veomni_commit": bridge["veomni_commit"],
        "frozen_checkpoint_tree_sha256": bridge["checkpoint_tree_sha256"],
        "frozen_checkpoint_content_manifest_file_sha256": bridge[
            "checkpoint_content_manifest_file_sha256"
        ],
        "frozen_checkpoint_content_verified_entries_digest": bridge[
            "checkpoint_content_verified_entries_digest"
        ],
        "current_candidate_manifest_file_sha256": clean[
            "candidate_manifest_file_sha256"
        ],
        "current_rv2v_clean_latent_tensor_sha256": clean["tensor_sha256"],
        "current_rv2v_patch_grid_height_width": list(candidate_grid),
        "current_rv2v_spatial_sketch_critic_tensor_sha256": candidate_sketch[
            "critic_tensor_sha256"
        ],
        "x_sigma_tensor_sha256": query["action_x_sigma_tensor_sha256"],
        "sp4_proof_digest": collective["proof_digest"],
        "gradient_tensor_sha256": gradient["tensor_sha256"],
        "gradient_shape": gradient["tensor_shape"],
        "gradient_norm": float(gradient_norm),
        "minimum_norm": float(minimum_norm),
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise StarcCriticPilotError(f"output already exists: {path}")
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
    except OSError as error:
        raise StarcCriticPilotError(f"could not create output: {path}") from error


def _validate_pilot_output_closure(pilot_root: Path) -> None:
    try:
        children = tuple(pilot_root.iterdir())
    except OSError as error:
        raise StarcCriticPilotError("could not inspect pilot output closure") from error
    if (
        {child.name for child in children} != set(PILOT_OUTPUT_FILENAMES)
        or len(children) != len(PILOT_OUTPUT_FILENAMES)
        or any(child.is_symlink() or not child.is_file() for child in children)
    ):
        raise StarcCriticPilotError(
            "pilot output must contain only the final checkpoint and four sealed receipts"
        )


def run_fit_evaluate(
    *,
    master_manifest: str | Path,
    expected_master_sha256: str,
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    graph = StarcMaterializerAdapter.load(
        master_manifest, expected_master_sha256=expected_master_sha256
    )
    parsed_device = _configure_determinism(device)
    requested_output = Path(output_dir)
    if not requested_output.is_absolute() or requested_output == Path("/"):
        raise StarcCriticPilotError("output directory must be absolute and non-root")
    if requested_output.exists() or requested_output.is_symlink():
        raise StarcCriticPilotError("output directory must be fresh")
    parent = requested_output.parent.resolve(strict=True)
    if not parent.is_dir() or requested_output != parent / requested_output.name:
        raise StarcCriticPilotError("output path/parent must be canonical and plain")
    requested_output.mkdir(mode=0o700)

    critic = _build_fresh_critic(device=str(parsed_device))
    critic_config_value = asdict(critic.config)
    config_receipt = _seal(
        {
            "schema_version": CONFIG_SCHEMA,
            "run_schema_version": SCHEMA_VERSION,
            "fixed_hyperparameters": dict(FIXED_HYPERPARAMETERS),
            "fixed_hyperparameter_digest": object_sha256(FIXED_HYPERPARAMETERS),
            "critic_config": critic_config_value,
            "critic_config_content_digest": object_sha256(critic_config_value),
            "pre_sketched_head_contract": {
                "entrypoint": "forward_sketched_residual_only",
                "input_shape": list(RESIDUAL_SHAPE),
                "geometry_neutral_after_fixed_sketch": True,
                "constructor_spatial_buffer": "inert_16x16_identity_never_consumed",
                "constructor_spatial_buffer_checkpointed": False,
                "full_hidden_forward_authorized": False,
                "geometry_specific_sketches_authenticated_by_materializer": True,
                "trainable_parameter_count": critic.trainable_parameter_count,
            },
            "materializer_master_path": str(graph.master_path),
            "materializer_master_file_sha256": graph.master_file_sha256,
            "materializer_master_receipt_digest": graph.master_receipt_digest,
            "materialized_population_content_digest": graph.content_digest,
            "spatial_sketch_bindings_by_episode": {
                episode_id: dict(binding)
                for episode_id, binding in graph.spatial_sketch_bindings_by_episode.items()
            },
            "fit_episode_order": list(graph.episode_ids("fit")),
            "confirmation_episode_order": list(graph.episode_ids("confirmation")),
            "confirmation_tensor_load_phase": "after_step200_checkpoint_reload_and_freeze",
            "nuisance_basis_used": False,
            "core4_scientific_claim_authorized": False,
            "editor_optimizer_present_or_authorized": False,
        }
    )
    _write_json_create_only(requested_output / CONFIG_FILENAME, config_receipt)

    # Isolation boundary: no confirmation safetensors artifact is opened above.
    fit_episodes = load_split_tensors(
        graph, split="fit", device=str(parsed_device)
    )
    fit_trace = train_fixed_fit_cells(critic, fit_episodes)
    _write_json_create_only(requested_output / TRACE_FILENAME, fit_trace)

    frozen_critic, checkpoint_receipt = save_reload_final_checkpoint(
        critic,
        output_dir=requested_output,
        device=str(parsed_device),
        config_receipt=config_receipt,
        fit_episodes=fit_episodes,
    )
    _write_json_create_only(
        requested_output / CHECKPOINT_RECEIPT_FILENAME, checkpoint_receipt
    )

    # Held-out isolation boundary: this is the first confirmation tensor read.
    confirmation_episodes = load_split_tensors(
        graph, split="confirmation", device=str(parsed_device)
    )
    confirmation_outputs = _score_loaded_episodes(
        frozen_critic, confirmation_episodes
    )
    heldout_margin_gate = make_heldout_margin_gate(
        confirmation_outputs,
        expected_episode_ids=graph.episode_ids("confirmation"),
    )
    # A live VJP can only be produced after this checkpoint exists.  Therefore
    # fit-evaluate is permanently provisional, even when all 24 margins pass.
    worth_topup = False
    failures = list(heldout_margin_gate["failure_reasons"])
    failures.append("live_current_rv2v_input_vjp_composite_receipt_pending")
    deferred_live_vjp = validate_live_vjp_receipt(None, None)
    gate_receipt = _seal(
        {
            "schema_version": PROVISIONAL_GATE_SCHEMA,
            "run_schema_version": SCHEMA_VERSION,
            "gate_stage": "fit-evaluate-provisional",
            "finalization_required": True,
            "materializer_binding": {
                "master_path": str(graph.master_path),
                "master_file_sha256": graph.master_file_sha256,
                "master_receipt_digest": graph.master_receipt_digest,
                "group_manifest_file_sha256s": list(
                    graph.group_manifest_file_sha256s
                ),
                "group_receipt_digests": list(graph.group_receipt_digests),
                "population_content_digest": graph.content_digest,
                "spatial_sketch_bindings_by_episode": {
                    episode_id: dict(binding)
                    for episode_id, binding in graph.spatial_sketch_bindings_by_episode.items()
                },
            },
            "config_binding": {
                "path": str(requested_output / CONFIG_FILENAME),
                "file_sha256": file_sha256(requested_output / CONFIG_FILENAME),
                "receipt_digest": config_receipt["receipt_digest"],
            },
            "fit_trace_binding": {
                "path": str(requested_output / TRACE_FILENAME),
                "file_sha256": file_sha256(requested_output / TRACE_FILENAME),
                "receipt_digest": fit_trace["receipt_digest"],
                "optimizer_step_count": FIXED_OPTIMIZER_STEPS,
            },
            "checkpoint_binding": {
                "receipt_path": str(
                    requested_output / CHECKPOINT_RECEIPT_FILENAME
                ),
                "receipt_file_sha256": file_sha256(
                    requested_output / CHECKPOINT_RECEIPT_FILENAME
                ),
                "receipt_digest": checkpoint_receipt["receipt_digest"],
                "checkpoint_path": checkpoint_receipt["checkpoint_path"],
                "checkpoint_file_sha256": checkpoint_receipt[
                    "checkpoint_file_sha256"
                ],
                "checkpoint_state_content_digest": checkpoint_receipt[
                    "checkpoint_state_content_digest"
                ],
            },
            "fit_protocol": {
                "fit_episode_order": list(graph.episode_ids("fit")),
                "both_fit_cells_consumed_every_step": True,
                "optimizer_steps": FIXED_OPTIMIZER_STEPS,
                "only_final_checkpoint_saved": True,
                "early_stopping_performed": False,
                "confirmation_samples_consumed_by_optimizer": False,
            },
            "confirmation_protocol": {
                "confirmation_episode_order": list(
                    graph.episode_ids("confirmation")
                ),
                "tensor_artifact_count": sum(
                    row.loaded_artifact_count for row in confirmation_episodes
                ),
                "critic_forward_count": 26,
                "each_tensor_artifact_read_once": True,
                "evaluated_once_after_checkpoint_reload_and_freeze": True,
                "used_for_checkpoint_threshold_layer_or_hyperparameter_selection": False,
                "outputs": confirmation_outputs,
            },
            "heldout_margin_gate": heldout_margin_gate,
            "live_current_rv2v_input_vjp_gate": deferred_live_vjp,
            "worth_fixed_topup_generation": worth_topup,
            "scientific_critic_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "editor_optimizer_present": False,
            "editor_optimizer_authorized": False,
            "generated_rgb_or_latent_used_as_editor_target_condition_donor_or_noise": False,
            "failure_reasons": failures,
        }
    )
    _write_json_create_only(
        requested_output / PROVISIONAL_GATE_RECEIPT_FILENAME, gate_receipt
    )
    _validate_pilot_output_closure(requested_output)
    os.chmod(requested_output / CHECKPOINT_FILENAME, 0o400)
    for filename in (
        CONFIG_FILENAME,
        TRACE_FILENAME,
        CHECKPOINT_RECEIPT_FILENAME,
        PROVISIONAL_GATE_RECEIPT_FILENAME,
    ):
        os.chmod(requested_output / filename, 0o400)
    os.chmod(requested_output, 0o500)
    return gate_receipt


def _load_bound_sealed_receipt(
    binding: Mapping[str, Any],
    *,
    path_field: str,
    file_sha_field: str,
    receipt_digest_field: str,
    schema: str,
    label: str,
    beneath: Path,
) -> tuple[dict[str, Any], Path, str]:
    if not isinstance(binding, Mapping):
        raise StarcCriticPilotError(f"{label} binding must be an object")
    raw, path, observed_sha = _read_json_file(
        binding.get(path_field),
        label=label,
        expected_sha256=binding.get(file_sha_field),
        beneath=beneath,
    )
    receipt = _validate_sealed_manifest(raw, schema=schema, label=label)
    if receipt["receipt_digest"] != _sha256(
        binding.get(receipt_digest_field), label=f"{label} bound receipt digest"
    ):
        raise StarcCriticPilotError(f"{label} receipt binding differs")
    return receipt, path, observed_sha


def run_finalize(
    *,
    pilot_output_dir: str | Path,
    expected_provisional_gate_sha256: str,
    live_vjp_receipt: str | Path,
    expected_live_vjp_receipt_sha256: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Combine a later live VJP with an immutable fit/evaluate result.

    This path is standard-library only.  It does not import Torch, instantiate
    a critic or optimizer, load a residual safetensors file, reopen a
    confirmation tensor, or alter the fixed threshold.  It replays the sealed
    JSON/file graph and recomputes the 24 margins from stored scalar outputs.
    """

    pilot_root = _plain_directory(pilot_output_dir, label="STARC pilot output")
    _validate_pilot_output_closure(pilot_root)
    provisional_raw, provisional_path, provisional_file_sha = _read_json_file(
        pilot_root / PROVISIONAL_GATE_RECEIPT_FILENAME,
        label="STARC provisional gate receipt",
        expected_sha256=expected_provisional_gate_sha256,
        beneath=pilot_root,
    )
    provisional = _validate_sealed_manifest(
        provisional_raw,
        schema=PROVISIONAL_GATE_SCHEMA,
        label="STARC provisional gate receipt",
    )
    _closed_binding(
        provisional,
        _PROVISIONAL_GATE_RECEIPT_FIELDS,
        label="STARC provisional gate receipt",
    )
    if (
        provisional.get("gate_stage") != "fit-evaluate-provisional"
        or provisional.get("run_schema_version") != SCHEMA_VERSION
        or provisional.get("finalization_required") is not True
        or provisional.get("worth_fixed_topup_generation") is not False
        or provisional.get("scientific_critic_claim_authorized") is not False
        or provisional.get("action_editing_success_claim_authorized") is not False
        or provisional.get("editor_optimizer_present") is not False
        or provisional.get("editor_optimizer_authorized") is not False
        or provisional.get(
            "generated_rgb_or_latent_used_as_editor_target_condition_donor_or_noise"
        )
        is not False
    ):
        raise StarcCriticPilotError("provisional gate exceeded its authority")
    deferred = provisional.get("live_current_rv2v_input_vjp_gate")
    if (
        not isinstance(deferred, Mapping)
        or not _json_exact_equal(deferred, validate_live_vjp_receipt(None, None))
        or deferred.get("provided") is not False
        or deferred.get("passed") is not False
        or deferred.get("checkpoint_and_runtime_bound") is not False
    ):
        raise StarcCriticPilotError("provisional gate already consumed a live VJP")

    materializer = provisional.get("materializer_binding")
    if not isinstance(materializer, Mapping):
        raise StarcCriticPilotError("provisional materializer binding is absent")
    graph = StarcMaterializerAdapter.load(
        materializer.get("master_path"),
        expected_master_sha256=materializer.get("master_file_sha256"),
    )
    expected_materializer_binding = {
        "master_path": str(graph.master_path),
        "master_file_sha256": graph.master_file_sha256,
        "master_receipt_digest": graph.master_receipt_digest,
        "group_manifest_file_sha256s": list(graph.group_manifest_file_sha256s),
        "group_receipt_digests": list(graph.group_receipt_digests),
        "population_content_digest": graph.content_digest,
        "spatial_sketch_bindings_by_episode": {
            episode_id: dict(binding)
            for episode_id, binding in graph.spatial_sketch_bindings_by_episode.items()
        },
    }
    if not _json_exact_equal(materializer, expected_materializer_binding):
        raise StarcCriticPilotError("provisional materializer graph binding differs")

    config_receipt, config_path, config_file_sha = _load_bound_sealed_receipt(
        provisional.get("config_binding"),
        path_field="path",
        file_sha_field="file_sha256",
        receipt_digest_field="receipt_digest",
        schema=CONFIG_SCHEMA,
        label="STARC critic config receipt",
        beneath=pilot_root,
    )
    _closed_binding(
        provisional.get("config_binding"),
        frozenset({"path", "file_sha256", "receipt_digest"}),
        label="provisional config binding",
    )
    _closed_binding(
        config_receipt,
        _CONFIG_RECEIPT_FIELDS,
        label="STARC critic config receipt",
    )
    if (
        config_path != pilot_root / CONFIG_FILENAME
        or not _json_exact_equal(
            config_receipt.get("fixed_hyperparameters"), FIXED_HYPERPARAMETERS
        )
        or config_receipt.get("fixed_hyperparameter_digest")
        != object_sha256(FIXED_HYPERPARAMETERS)
        or config_receipt.get("run_schema_version") != SCHEMA_VERSION
        or config_receipt.get("critic_config_content_digest")
        != object_sha256(config_receipt.get("critic_config"))
        or config_receipt.get("materializer_master_path") != str(graph.master_path)
        or config_receipt.get("materializer_master_file_sha256")
        != graph.master_file_sha256
        or config_receipt.get("materializer_master_receipt_digest")
        != graph.master_receipt_digest
        or config_receipt.get("materialized_population_content_digest")
        != graph.content_digest
        or not _json_exact_equal(
            config_receipt.get("spatial_sketch_bindings_by_episode"),
            graph.spatial_sketch_bindings_by_episode,
        )
        or config_receipt.get("fit_episode_order") != list(graph.episode_ids("fit"))
        or config_receipt.get("confirmation_episode_order")
        != list(graph.episode_ids("confirmation"))
        or not _json_exact_equal(
            config_receipt.get("critic_config"), GEOMETRY_NEUTRAL_CRITIC_CONFIG
        )
        or config_receipt.get("pre_sketched_head_contract", {}).get("entrypoint")
        != "forward_sketched_residual_only"
        or config_receipt.get("pre_sketched_head_contract", {}).get(
            "constructor_spatial_buffer_checkpointed"
        )
        is not False
        or config_receipt.get("pre_sketched_head_contract", {}).get(
            "geometry_neutral_after_fixed_sketch"
        )
        is not True
        or config_receipt.get("pre_sketched_head_contract", {}).get(
            "full_hidden_forward_authorized"
        )
        is not False
        or config_receipt.get("confirmation_tensor_load_phase")
        != "after_step200_checkpoint_reload_and_freeze"
        or config_receipt.get("nuisance_basis_used") is not False
        or config_receipt.get("core4_scientific_claim_authorized") is not False
        or config_receipt.get("editor_optimizer_present_or_authorized") is not False
    ):
        raise StarcCriticPilotError("critic config replay differs")
    head_contract = config_receipt["pre_sketched_head_contract"]
    if (
        not isinstance(head_contract, Mapping)
        or set(head_contract)
        != {
            "entrypoint",
            "input_shape",
            "geometry_neutral_after_fixed_sketch",
            "constructor_spatial_buffer",
            "constructor_spatial_buffer_checkpointed",
            "full_hidden_forward_authorized",
            "geometry_specific_sketches_authenticated_by_materializer",
            "trainable_parameter_count",
        }
        or not _json_exact_equal(
            head_contract.get("input_shape"), list(RESIDUAL_SHAPE)
        )
        or head_contract.get("constructor_spatial_buffer")
        != "inert_16x16_identity_never_consumed"
        or head_contract.get(
            "geometry_specific_sketches_authenticated_by_materializer"
        )
        is not True
        or type(head_contract.get("trainable_parameter_count")) is not int
        or not 0 < head_contract["trainable_parameter_count"] < 1_000_000
    ):
        raise StarcCriticPilotError("pre-sketched head contract replay differs")

    trace_receipt, trace_path, trace_file_sha = _load_bound_sealed_receipt(
        provisional.get("fit_trace_binding"),
        path_field="path",
        file_sha_field="file_sha256",
        receipt_digest_field="receipt_digest",
        schema=TRACE_SCHEMA,
        label="STARC fit trace receipt",
        beneath=pilot_root,
    )
    _closed_binding(
        provisional.get("fit_trace_binding"),
        frozenset(
            {"path", "file_sha256", "receipt_digest", "optimizer_step_count"}
        ),
        label="provisional fit trace binding",
    )
    _closed_binding(
        trace_receipt,
        _TRACE_RECEIPT_FIELDS,
        label="STARC fit trace receipt",
    )
    steps = trace_receipt.get("steps")
    expected_fit_ids = list(graph.episode_ids("fit"))
    trace_step_fields = {
        "step",
        "loss",
        "gradient_norm_before_clip",
        "minimum_fit_group_margin",
        "episode_ids",
    }
    trace_rows_valid = isinstance(steps, list) and all(
        isinstance(row, Mapping)
        and set(row) == trace_step_fields
        and type(row["step"]) is int
        and all(
            not isinstance(row[name], bool)
            and isinstance(row[name], (int, float))
            and math.isfinite(float(row[name]))
            for name in (
                "loss",
                "gradient_norm_before_clip",
                "minimum_fit_group_margin",
            )
        )
        and float(row["gradient_norm_before_clip"]) > 0.0
        for row in steps
    )
    if (
        trace_path != pilot_root / TRACE_FILENAME
        or not _json_exact_equal(
            trace_receipt.get("fixed_hyperparameters"), FIXED_HYPERPARAMETERS
        )
        or trace_receipt.get("fit_episode_order") != expected_fit_ids
        or trace_receipt.get("optimizer_step_count") != FIXED_OPTIMIZER_STEPS
        or provisional["fit_trace_binding"].get("optimizer_step_count")
        != FIXED_OPTIMIZER_STEPS
        or trace_receipt.get("fit_artifact_count") != 26
        or not isinstance(steps, list)
        or len(steps) != FIXED_OPTIMIZER_STEPS
        or not trace_rows_valid
        or [row.get("step") for row in steps]
        != list(range(1, FIXED_OPTIMIZER_STEPS + 1))
        or any(row.get("episode_ids") != expected_fit_ids for row in steps)
        or trace_receipt.get("both_fit_cells_consumed_every_step") is not True
        or trace_receipt.get(
            "confirmation_manifest_metadata_authenticated_before_fit"
        )
        is not True
        or trace_receipt.get(
            "confirmation_tensor_artifacts_opened_before_fit_complete"
        )
        is not False
        or trace_receipt.get("confirmation_samples_consumed_by_optimizer") is not False
        or trace_receipt.get("best_checkpoint_saved") is not False
        or trace_receipt.get("checkpoint_selection") != "final_step_200_only"
        or trace_receipt.get("early_stopping_performed") is not False
        or trace_receipt.get("editor_parameter_present") is not False
    ):
        raise StarcCriticPilotError("fit trace replay differs")

    checkpoint_receipt, checkpoint_receipt_path, checkpoint_receipt_file_sha = (
        _load_bound_sealed_receipt(
            provisional.get("checkpoint_binding"),
            path_field="receipt_path",
            file_sha_field="receipt_file_sha256",
            receipt_digest_field="receipt_digest",
            schema=CHECKPOINT_SCHEMA,
            label="STARC checkpoint receipt",
            beneath=pilot_root,
        )
    )
    _closed_binding(
        provisional.get("checkpoint_binding"),
        frozenset(
            {
                "receipt_path",
                "receipt_file_sha256",
                "receipt_digest",
                "checkpoint_path",
                "checkpoint_file_sha256",
                "checkpoint_state_content_digest",
            }
        ),
        label="provisional checkpoint binding",
    )
    _closed_binding(
        checkpoint_receipt,
        _CHECKPOINT_RECEIPT_FIELDS,
        label="STARC checkpoint receipt",
    )
    checkpoint_path = _plain_file(
        checkpoint_receipt.get("checkpoint_path"),
        label="STARC final critic checkpoint",
        beneath=pilot_root,
    )
    if (
        checkpoint_receipt_path != pilot_root / CHECKPOINT_RECEIPT_FILENAME
        or checkpoint_path != pilot_root / CHECKPOINT_FILENAME
        or file_sha256(checkpoint_path)
        != _sha256(
            checkpoint_receipt.get("checkpoint_file_sha256"),
            label="critic checkpoint file SHA-256",
        )
        or checkpoint_receipt.get("config_receipt_digest")
        != config_receipt["receipt_digest"]
        or checkpoint_receipt.get("optimizer_step") != FIXED_OPTIMIZER_STEPS
        or type(checkpoint_receipt.get("checkpoint_tensor_count")) is not int
        or checkpoint_receipt.get("checkpoint_tensor_count") <= 0
        or checkpoint_receipt.get("checkpoint_scope")
        != "geometry_neutral_pre_sketched_critic_head_only"
        or checkpoint_receipt.get("excluded_constructor_buffer_keys")
        != list(NON_HEAD_STATE_KEYS)
        or checkpoint_receipt.get("only_final_checkpoint_saved") is not True
        or checkpoint_receipt.get("best_checkpoint_saved") is not False
        or checkpoint_receipt.get("confirmation_sample_seen_before_checkpoint_save")
        is not False
        or checkpoint_receipt.get("state_tensor_byte_parity_after_fresh_load")
        is not True
        or checkpoint_receipt.get("fit_score_parity_after_fresh_load") is not True
        or checkpoint_receipt.get("critic_frozen_after_reload") is not True
        or checkpoint_receipt.get("editor_checkpoint_or_parameter_present") is not False
        or checkpoint_receipt.get("editor_optimizer_authorized") is not False
    ):
        raise StarcCriticPilotError("final checkpoint replay differs")
    checkpoint_binding = provisional.get("checkpoint_binding")
    if (
        checkpoint_binding.get("checkpoint_path") != str(checkpoint_path)
        or checkpoint_binding.get("checkpoint_file_sha256")
        != checkpoint_receipt["checkpoint_file_sha256"]
        or checkpoint_binding.get("checkpoint_state_content_digest")
        != checkpoint_receipt["checkpoint_state_content_digest"]
    ):
        raise StarcCriticPilotError("provisional checkpoint binding differs")

    fit_protocol = provisional.get("fit_protocol")
    expected_fit_protocol = {
        "fit_episode_order": list(graph.episode_ids("fit")),
        "both_fit_cells_consumed_every_step": True,
        "optimizer_steps": FIXED_OPTIMIZER_STEPS,
        "only_final_checkpoint_saved": True,
        "early_stopping_performed": False,
        "confirmation_samples_consumed_by_optimizer": False,
    }
    if not _json_exact_equal(fit_protocol, expected_fit_protocol):
        raise StarcCriticPilotError("provisional fit protocol differs")

    confirmation = provisional.get("confirmation_protocol")
    if (
        not isinstance(confirmation, Mapping)
        or set(confirmation)
        != {
            "confirmation_episode_order",
            "tensor_artifact_count",
            "critic_forward_count",
            "each_tensor_artifact_read_once",
            "evaluated_once_after_checkpoint_reload_and_freeze",
            "used_for_checkpoint_threshold_layer_or_hyperparameter_selection",
            "outputs",
        }
        or confirmation.get("confirmation_episode_order")
        != list(graph.episode_ids("confirmation"))
        or confirmation.get("tensor_artifact_count") != 26
        or confirmation.get("critic_forward_count") != 26
        or confirmation.get("each_tensor_artifact_read_once") is not True
        or confirmation.get("evaluated_once_after_checkpoint_reload_and_freeze")
        is not True
        or confirmation.get(
            "used_for_checkpoint_threshold_layer_or_hyperparameter_selection"
        )
        is not False
        or not isinstance(confirmation.get("outputs"), Mapping)
    ):
        raise StarcCriticPilotError("confirmation replay contract differs")
    recomputed_margin_gate = make_heldout_margin_gate(
        confirmation["outputs"],
        expected_episode_ids=graph.episode_ids("confirmation"),
    )
    if canonical_json_bytes(recomputed_margin_gate) != canonical_json_bytes(
        provisional.get("heldout_margin_gate")
    ):
        raise StarcCriticPilotError("stored heldout margins do not recompute")
    expected_provisional_failures = list(
        recomputed_margin_gate["failure_reasons"]
    ) + ["live_current_rv2v_input_vjp_composite_receipt_pending"]
    if provisional.get("failure_reasons") != expected_provisional_failures:
        raise StarcCriticPilotError("provisional failure reasons differ")

    live_vjp = validate_live_vjp_receipt(
        live_vjp_receipt,
        expected_live_vjp_receipt_sha256,
        graph=graph,
        config_receipt=config_receipt,
        checkpoint_receipt=checkpoint_receipt,
        config_receipt_path=config_path,
        config_receipt_file_sha256=config_file_sha,
        checkpoint_receipt_path=checkpoint_receipt_path,
        checkpoint_receipt_file_sha256=checkpoint_receipt_file_sha,
    )
    worth_topup = (
        recomputed_margin_gate["all_24_role_margins_passed"] is True
        and live_vjp["passed"] is True
    )
    failures = list(recomputed_margin_gate["failure_reasons"])

    requested_output = Path(output_dir)
    if not requested_output.is_absolute() or requested_output == Path("/"):
        raise StarcCriticPilotError("final output directory must be absolute and non-root")
    if requested_output.exists() or requested_output.is_symlink():
        raise StarcCriticPilotError("final output directory must be fresh")
    parent = requested_output.parent.resolve(strict=True)
    if not parent.is_dir() or requested_output != parent / requested_output.name:
        raise StarcCriticPilotError(
            "final output path/parent must be canonical and plain"
        )
    canonical_requested_output = parent / requested_output.name
    try:
        canonical_requested_output.relative_to(pilot_root)
    except ValueError:
        pass
    else:
        raise StarcCriticPilotError(
            "final output directory must be separate from the pilot output"
        )
    requested_output.mkdir(mode=0o700)
    final_receipt = _seal(
        {
            "schema_version": FINAL_GATE_SCHEMA,
            "run_schema_version": SCHEMA_VERSION,
            "gate_stage": "post-live-vjp-finalize",
            "provisional_gate_binding": {
                "path": str(provisional_path),
                "file_sha256": provisional_file_sha,
                "receipt_digest": provisional["receipt_digest"],
            },
            "materializer_binding": dict(materializer),
            "config_binding": {
                "path": str(config_path),
                "file_sha256": config_file_sha,
                "receipt_digest": config_receipt["receipt_digest"],
            },
            "fit_trace_binding": {
                "path": str(trace_path),
                "file_sha256": trace_file_sha,
                "receipt_digest": trace_receipt["receipt_digest"],
                "optimizer_step_count": FIXED_OPTIMIZER_STEPS,
            },
            "checkpoint_binding": {
                "receipt_path": str(checkpoint_receipt_path),
                "receipt_file_sha256": checkpoint_receipt_file_sha,
                "receipt_digest": checkpoint_receipt["receipt_digest"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_file_sha256": checkpoint_receipt[
                    "checkpoint_file_sha256"
                ],
                "checkpoint_state_content_digest": checkpoint_receipt[
                    "checkpoint_state_content_digest"
                ],
            },
            "heldout_margin_gate": recomputed_margin_gate,
            "live_current_rv2v_input_vjp_gate": live_vjp,
            "finalization_retrained_critic": False,
            "finalization_loaded_fit_or_confirmation_tensor_artifact": False,
            "finalization_changed_threshold_layer_or_hyperparameter": False,
            "worth_fixed_topup_generation": worth_topup,
            "scientific_critic_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "editor_optimizer_present": False,
            "editor_optimizer_authorized": False,
            "generated_rgb_or_latent_used_as_editor_target_condition_donor_or_noise": False,
            "failure_reasons": failures,
        }
    )
    final_path = requested_output / FINAL_GATE_RECEIPT_FILENAME
    _write_json_create_only(final_path, final_receipt)
    os.chmod(final_path, 0o400)
    os.chmod(requested_output, 0o500)
    return final_receipt


def run_pilot(**kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for the explicitly provisional fit/evaluate stage."""

    return run_fit_evaluate(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or finalize the fixed STARC core4 critic-only pilot"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "fit-evaluate",
        help="fit step 1..200, save/reload/freeze, and evaluate confirmation once",
    )
    fit.add_argument("--master-manifest", required=True)
    fit.add_argument("--expected-master-sha256", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--device", default="cuda:0")
    finalize = commands.add_parser(
        "finalize",
        help="bind a later live VJP without retraining or reopening residual tensors",
    )
    finalize.add_argument("--pilot-output-dir", required=True)
    finalize.add_argument("--expected-provisional-gate-sha256", required=True)
    finalize.add_argument("--live-vjp-receipt", required=True)
    finalize.add_argument("--expected-live-vjp-receipt-sha256", required=True)
    finalize.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit-evaluate":
        receipt = run_fit_evaluate(
            master_manifest=args.master_manifest,
            expected_master_sha256=args.expected_master_sha256,
            output_dir=args.output_dir,
            device=args.device,
        )
        receipt_path = Path(args.output_dir) / PROVISIONAL_GATE_RECEIPT_FILENAME
    elif args.command == "finalize":
        receipt = run_finalize(
            pilot_output_dir=args.pilot_output_dir,
            expected_provisional_gate_sha256=args.expected_provisional_gate_sha256,
            live_vjp_receipt=args.live_vjp_receipt,
            expected_live_vjp_receipt_sha256=args.expected_live_vjp_receipt_sha256,
            output_dir=args.output_dir,
        )
        receipt_path = Path(args.output_dir) / FINAL_GATE_RECEIPT_FILENAME
    else:  # pragma: no cover - argparse enforces the command set.
        raise StarcCriticPilotError("unknown runner command")
    print(
        json.dumps(
            {
                "gate_receipt": str(receipt_path),
                "gate_stage": receipt["gate_stage"],
                "receipt_digest": receipt["receipt_digest"],
                "all_24_role_margins_passed": receipt["heldout_margin_gate"][
                    "all_24_role_margins_passed"
                ],
                "live_current_rv2v_input_vjp_passed": receipt[
                    "live_current_rv2v_input_vjp_gate"
                ]["passed"],
                "worth_fixed_topup_generation": receipt[
                    "worth_fixed_topup_generation"
                ],
                "editor_optimizer_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_SCHEMA",
    "ArmArtifactBinding",
    "BERNINI_CHECKPOINT_CONTENT_FILE_COUNT",
    "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "BERNINI_CHECKPOINT_TREE_SHA256",
    "BERNINI_OFFICIAL_COMMIT",
    "CHECKPOINT_FILENAME",
    "FINAL_GATE_RECEIPT_FILENAME",
    "FINAL_GATE_SCHEMA",
    "FIXED_HELDOUT_MINIMUM_MARGIN",
    "FIXED_HYPERPARAMETERS",
    "FIXED_MILESTONE_NAMES",
    "FIXED_OPTIMIZER_STEPS",
    "FIXED_SEED",
    "GATE_RECEIPT_FILENAME",
    "GEOMETRY_NEUTRAL_CRITIC_CONFIG",
    "GROUP_SCHEMA",
    "LIVE_VJP_BINDING_SCHEMA",
    "LIVE_VJP_BACKEND_ID",
    "LIVE_VJP_BRIDGE_ARCHIVE_MEMBER",
    "LIVE_VJP_CANDIDATE_SCHEMA",
    "LIVE_VJP_SP4_IMPLEMENTATION",
    "LoadedEpisode",
    "MASTER_SCHEMA",
    "PROVISIONAL_GATE_RECEIPT_FILENAME",
    "PROVISIONAL_GATE_SCHEMA",
    "PilotManifestGraph",
    "RESIDUAL_SHAPE",
    "RESIDUAL_TENSOR_KEY",
    "SCHEMA_VERSION",
    "StarcCriticPilotError",
    "StarcMaterializerAdapter",
    "build_parser",
    "canonical_json_bytes",
    "checkpoint_state_content_digest",
    "file_sha256",
    "load_split_tensors",
    "make_heldout_margin_gate",
    "materializer_tensor_sha256",
    "object_sha256",
    "reconstruct_geometry_spatial_sketch_binding",
    "run_finalize",
    "run_fit_evaluate",
    "run_pilot",
    "save_reload_final_checkpoint",
    "train_fixed_fit_cells",
    "validate_live_vjp_receipt",
    "VEOMNI_TESTED_COMMIT",
]

#!/usr/bin/env python3
"""Sealed contracts for same-video temporal-counterfactual action scoring.

The scorer never compares the absolute energy of two different videos.  For
one exact81 clean latent it deterministically constructs a chronological arm
and six temporal counterfactuals, reuses one official Gaussian, and evaluates
the target-action/no-op prompt pair at three pinned native Bernini coordinates.

This module is deliberately CPU importable.  Torch is imported only by
``apply_temporal_transform_tensor``; receipt construction and calibration
tests require only the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping, Sequence


TRANSFORM_PLAN_SCHEMA = "bernini-temporal-counterfactual-transform-plan-v1"
SIGMA_COORDINATE_SCHEMA = "bernini-temporal-counterfactual-native-sigmas-v1"
PROMPT_PAIR_RECEIPT_SCHEMA = "bernini-temporal-counterfactual-prompt-pair-v1"
CANDIDATE_SCORE_SCHEMA = "bernini-temporal-counterfactual-action-score-v1"
GROUP_RECEIPT_SCHEMA = "bernini-temporal-counterfactual-action-score-group-v1"

FRAME_COUNT = 81
LATENT_PHASES = 21
ACTION_BRANCH = "action"
NEGATIVE_BRANCHES = (
    "noop",
    "incomplete",
    "reverse",
    "shuffle",
    "wrong_actor",
    "wrong_object",
    "camera_only",
    "appearance_only",
    "generic_wrong_motion",
)
BRANCH_ORDER = (ACTION_BRANCH, *NEGATIVE_BRANCHES)
ANALYSIS_SPLITS = ("fit", "confirmation")

CHRONOLOGICAL = "chronological"
TRANSFORM_ORDER = (
    CHRONOLOGICAL,
    "reverse",
    "freeze_first",
    "truncate_hold",
    "terminal_only",
    "phase_shuffle",
    "transition_loop",
)
COUNTERFACTUAL_TRANSFORMS = TRANSFORM_ORDER[1:]
MULTISET_PRESERVING_TRANSFORMS = (
    CHRONOLOGICAL,
    "reverse",
    "phase_shuffle",
)
NON_MULTISET_CONTROL_TRANSFORMS = (
    "freeze_first",
    "truncate_hold",
    "terminal_only",
    "transition_loop",
)

# The index maps are part of the preregistration, not run-time choices.
_TRANSFORM_INDEX_MAPS: dict[str, tuple[int, ...]] = {
    CHRONOLOGICAL: tuple(range(LATENT_PHASES)),
    "reverse": tuple(range(LATENT_PHASES - 1, -1, -1)),
    "freeze_first": (0,) * LATENT_PHASES,
    # Preserve phases 0..12, then remove the late transition by holding phase 12.
    "truncate_hold": (*range(13), *((12,) * 8)),
    "terminal_only": (LATENT_PHASES - 1,) * LATENT_PHASES,
    # Multiplication by 8 is a fixed permutation modulo 21 (gcd(8,21)=1).
    "phase_shuffle": tuple((8 * index) % LATENT_PHASES for index in range(LATENT_PHASES)),
    # Traverse the early transition and loop back without ever using phases 11..20.
    "transition_loop": (*range(11), *range(9, -1, -1)),
}

_TRANSFORM_RATIONALES = {
    CHRONOLOGICAL: "unaltered candidate chronology",
    "reverse": "direction hard negative with an identical clean-latent value multiset",
    "freeze_first": "static initial-state control with duplicated phase zero",
    "truncate_hold": "late-phase ablation followed by a fixed hold at phase 12",
    "terminal_only": "last-phase repeat without an observed transition",
    "phase_shuffle": "fixed nonchronological permutation with an identical value multiset",
    "transition_loop": "fixed prefix palindrome using phases 0..10..0 and omitting 11..20",
}

# Three native UniPC-40 coordinates spanning middle-high, middle, and lower noise.
# Values are copied from the pinned Bernini schedule and checked again by the GPU
# entry point before any model call.
NATIVE_SCHEDULE_DIGEST = "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
NATIVE_SIGMA_COORDINATES = (
    (25, 0.7504994869232178, 750),
    (33, 0.5161304473876953, 516),
    (37, 0.2911904454231262, 291),
)

ENERGY_EPSILON = 1.0e-8
REQUIRED_D541801_SCORER_REVISION = "d541801a162796aacde34c2bfc2b1f0472d954d2"
REQUIRED_D541801_SCORER_SHA256 = (
    "3d7ce459ddb9a014873acd6384c7c4030b4e3aca9004c1b8486ebbc1f0f5d32e"
)
REQUIRED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
REQUIRED_CORE4_V2_SPEC_SHA256 = (
    "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
)
REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256 = (
    "8c4f77bdd24fa14786f3dff28a4044d819f444c0338484a2fa6df9588100cb59"
)
REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST = (
    "79276ad5f499fe37775a23bc5a789b7eb6dd83170517f750daecf07bb782cdb2"
)
REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST = (
    "068947cc4dd0bf4166422e8c6ea810998d0ce12d21e2787254a51b8603bcefe7"
)
REQUIRED_CORE4_V2_GROUP_ORDER_DIGESTS = {
    "sp4-a": "2a967a21be6685971c22bdc413d31a3b1b7d6932037e5ff36c8daa420fed59f3",
    "sp4-b": "38fbef547c9819414b7144bbde40a987fbc3c5f3d2b06b76d0ad39c967d01ec5",
}
REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS = {
    "sp4-a": "d0bccf3248754f906dac1b56e55a0248d1d83abfbc49d8f33c256dcbe994bc43",
    "sp4-b": "879d24bc8f42040d93720e028b4604d73797e4415c81debaa7b0478323108085",
}
REQUIRED_CORE4_V2_CANDIDATE_IDENTITY_DIGEST = (
    "e447d667c5d66221df18a65dc7b653702378625097d7fde48eeca50f494f3e08"
)
REQUIRED_BERNINI_REVISION = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
REQUIRED_VEOMNI_REVISION = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

T2V_INPUT_CLOSURE = {
    "candidate_own_predecode_clean_latent_only": True,
    "same_candidate_deterministic_temporal_index_maps_only": True,
    "same_cell_official_gaussian_reused_byte_identically_for_every_transform": True,
    "official_gaussian_temporal_transform_applied": False,
    "semantic_inputs": [
        "cell_fixed_target_action_caption",
        "cell_fixed_scene_matched_noop_caption",
    ],
    "source_video_or_source_latent_consumed": False,
    "rv2v_reference_target_donor_or_noise_consumed": False,
    "t2v_media_or_latent_may_enter_rv2v_training": False,
    "mask_flow_pose_track_or_trajectory_consumed": False,
    "event_audit_label_consumed_by_model": False,
    "training_performed": False,
    "optimizer_step_performed": False,
}

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class TemporalCounterfactualContractError(ValueError):
    """A transform, score, or provenance binding failed closed."""


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
        raise TemporalCounterfactualContractError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise TemporalCounterfactualContractError(f"{label} field closure differs")
    return dict(value)


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TemporalCounterfactualContractError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise TemporalCounterfactualContractError(f"{label} must be lowercase SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise TemporalCounterfactualContractError(f"{label} is not path-safe")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise TemporalCounterfactualContractError(f"{label} must be finite")
    result = float(value)
    if nonnegative and result < 0.0:
        raise TemporalCounterfactualContractError(f"{label} must be nonnegative")
    return result


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(unsigned)
    return {**value, "receipt_digest": object_sha256(value)}


def _verify_seal(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    row = dict(value)
    digest = _sha256(row.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(row) != digest:
        raise TemporalCounterfactualContractError(f"{label} digest differs")
    return dict(value)


def temporal_index_map(transform_name: str) -> tuple[int, ...]:
    if transform_name not in _TRANSFORM_INDEX_MAPS:
        raise TemporalCounterfactualContractError("unknown temporal transform")
    return _TRANSFORM_INDEX_MAPS[transform_name]


def gaussian_coupling_mode(transform_name: str) -> str:
    temporal_index_map(transform_name)
    return "fixed_original_official_gaussian_at_absolute_phase"


def fixed_official_gaussian_tensor(value: Any, transform_name: str) -> Any:
    """Return the exact same official Gaussian object for every temporal arm."""

    temporal_index_map(transform_name)
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in GPU runtime
        raise TemporalCounterfactualContractError("Torch is unavailable") from error
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, LATENT_PHASES)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise TemporalCounterfactualContractError(
            "official Gaussian must be detached FP32 exact81"
        )
    return value


def apply_temporal_transform_sequence(
    sequence: Sequence[Any], transform_name: str
) -> tuple[Any, ...]:
    if len(sequence) != LATENT_PHASES:
        raise TemporalCounterfactualContractError("sequence must contain 21 phases")
    return tuple(sequence[index] for index in temporal_index_map(transform_name))


def apply_temporal_transform_tensor(value: Any, transform_name: str) -> Any:
    """Apply the sealed phase map to a detached ``[1,C,21,H,W]`` tensor."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only in GPU runtime
        raise TemporalCounterfactualContractError("Torch is unavailable") from error
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 5
        or int(value.shape[0]) != 1
        or int(value.shape[2]) != LATENT_PHASES
        or not value.is_floating_point()
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise TemporalCounterfactualContractError(
            "temporal transform input must be detached finite [1,C,21,H,W]"
        )
    indices = torch.tensor(
        temporal_index_map(transform_name), dtype=torch.long, device=value.device
    )
    result = value.index_select(2, indices).contiguous().detach()
    if tuple(result.shape) != tuple(value.shape):
        raise TemporalCounterfactualContractError("temporal transform geometry changed")
    return result


def make_transform_plan() -> dict[str, Any]:
    specs = []
    for name in TRANSFORM_ORDER:
        index_map = list(temporal_index_map(name))
        is_permutation = sorted(index_map) == list(range(LATENT_PHASES))
        expected_multiset = name in MULTISET_PRESERVING_TRANSFORMS
        if is_permutation != expected_multiset:
            raise TemporalCounterfactualContractError(
                f"transform {name} multiset class is internally inconsistent"
            )
        specs.append(
            {
                "name": name,
                "index_map": index_map,
                "index_map_digest": object_sha256(index_map),
                "latent_value_multiset_preserved_by_construction": expected_multiset,
                "gaussian_coupling_mode": gaussian_coupling_mode(name),
                "official_gaussian_temporal_transform_applied": False,
                "noised_state_and_velocity_target_value_multisets_preserved": (
                    name == CHRONOLOGICAL
                ),
                "control_class": (
                    "exact_value_multiset_permutation"
                    if expected_multiset
                    else "duplicate_or_omit_phase_control"
                ),
                "rationale": _TRANSFORM_RATIONALES[name],
            }
        )
    unsigned = {
        "schema_version": TRANSFORM_PLAN_SCHEMA,
        "frame_count": FRAME_COUNT,
        "latent_phase_count": LATENT_PHASES,
        "transform_order": list(TRANSFORM_ORDER),
        "multiset_preserving_transforms": list(MULTISET_PRESERVING_TRANSFORMS),
        "non_multiset_control_transforms": list(NON_MULTISET_CONTROL_TRANSFORMS),
        "transform_specs": specs,
        "random_transform_sampling": False,
        "candidate_dependent_transform_choice": False,
    }
    return _seal(unsigned)


def validate_transform_plan(value: Any) -> dict[str, Any]:
    expected = make_transform_plan()
    if value != expected:
        raise TemporalCounterfactualContractError("temporal transform plan differs")
    return dict(expected)


def _fp32_bits(value: float) -> str:
    return struct.pack("!f", float(value)).hex()


def make_sigma_coordinate_receipt() -> dict[str, Any]:
    coordinates = [
        {
            "native_schedule_index": index,
            "physical_sigma_float64_hex": float(sigma).hex(),
            "physical_sigma_float32_be_hex": _fp32_bits(sigma),
            "native_scheduler_timestep": timestep,
            "timestep_mapping": "direct_native_unipc40_same_schedule_index",
        }
        for index, sigma, timestep in NATIVE_SIGMA_COORDINATES
    ]
    unsigned = {
        "schema_version": SIGMA_COORDINATE_SCHEMA,
        "native_schedule_digest": NATIVE_SCHEDULE_DIGEST,
        "coordinate_order": [row["native_schedule_index"] for row in coordinates],
        "coordinates": coordinates,
        "coordinate_count": len(coordinates),
        "legacy_1000_times_sigma_timestep_rejected": True,
        "coordinate_selection_was_preregistered": True,
    }
    return _seal(unsigned)


def validate_sigma_coordinate_receipt(value: Any) -> dict[str, Any]:
    expected = make_sigma_coordinate_receipt()
    if value != expected:
        raise TemporalCounterfactualContractError("native sigma coordinate receipt differs")
    return dict(expected)


_PROMPT_PAIR_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "transform_name",
        "native_schedule_index",
        "physical_sigma_float32_be_hex",
        "native_scheduler_timestep",
        "native_schedule_digest",
        "transformed_clean_tensor_sha256",
        "official_gaussian_tensor_sha256",
        "effective_gaussian_tensor_sha256",
        "gaussian_coupling_mode",
        "x_sigma_tensor_sha256",
        "velocity_target_tensor_sha256",
        "action_velocity_tensor_sha256",
        "noop_velocity_tensor_sha256",
        "action_full_prompt_sha256",
        "noop_full_prompt_sha256",
        "action_condition_tensor_sha256",
        "noop_condition_tensor_sha256",
        "frozen_model_receipt_digest",
        "same_state_execution_proof",
        "same_x_sigma_object_for_prompt_pair",
        "same_native_coordinate_and_timestep_object_for_prompt_pair",
        "patch_vae_latent_calls_per_prompt_pair",
        "prompt_order",
        "target_only_t2v_forward",
        "source_or_rv2v_condition_consumed",
        "training_performed",
        "receipt_digest",
    }
)

_SAME_STATE_STAGES = ("before_action", "after_action", "after_noop")
_SAME_STATE_PROOF_FIELDS = frozenset(
    {
        "noisy_latents_sha256_by_stage",
        "rotary_embs_sha256_by_stage",
        "native_timestep_sha256_by_stage",
        "same_noisy_latents_object_reused",
        "same_rotary_embs_object_reused",
        "same_native_timestep_object_reused",
        "post_call_tensor_bytes_unchanged",
    }
)


def _validated_same_state_execution_proof(value: Any) -> dict[str, Any]:
    row = _closed(value, _SAME_STATE_PROOF_FIELDS, label="same-state execution proof")
    for field in (
        "noisy_latents_sha256_by_stage",
        "rotary_embs_sha256_by_stage",
        "native_timestep_sha256_by_stage",
    ):
        hashes = _closed(
            row[field], set(_SAME_STATE_STAGES), label=f"same-state {field}"
        )
        for stage, digest in hashes.items():
            _sha256(digest, label=f"{field} {stage}")
        if len(set(hashes.values())) != 1:
            raise TemporalCounterfactualContractError(
                f"{field} changed across action/no-op execution"
            )
        row[field] = hashes
    for field in _SAME_STATE_PROOF_FIELDS - {
        "noisy_latents_sha256_by_stage",
        "rotary_embs_sha256_by_stage",
        "native_timestep_sha256_by_stage",
    }:
        if row[field] is not True:
            raise TemporalCounterfactualContractError(
                "same-state object/byte proof differs"
            )
    return row


def make_prompt_pair_receipt(
    *,
    candidate_id: str,
    transform_name: str,
    native_schedule_index: int,
    transformed_clean_tensor_sha256: str,
    official_gaussian_tensor_sha256: str,
    effective_gaussian_tensor_sha256: str,
    x_sigma_tensor_sha256: str,
    velocity_target_tensor_sha256: str,
    action_velocity_tensor_sha256: str,
    noop_velocity_tensor_sha256: str,
    action_full_prompt_sha256: str,
    noop_full_prompt_sha256: str,
    action_condition_tensor_sha256: str,
    noop_condition_tensor_sha256: str,
    frozen_model_receipt_digest: str,
    same_state_execution_proof: Mapping[str, Any],
) -> dict[str, Any]:
    _safe_id(candidate_id, label="candidate ID")
    if transform_name not in TRANSFORM_ORDER:
        raise TemporalCounterfactualContractError("prompt-pair transform differs")
    coordinate = next(
        (
            row
            for row in make_sigma_coordinate_receipt()["coordinates"]
            if row["native_schedule_index"] == native_schedule_index
        ),
        None,
    )
    if coordinate is None:
        raise TemporalCounterfactualContractError("prompt-pair sigma is not preregistered")
    for label, value in (
        ("transformed clean tensor", transformed_clean_tensor_sha256),
        ("official Gaussian tensor", official_gaussian_tensor_sha256),
        ("effective Gaussian tensor", effective_gaussian_tensor_sha256),
        ("x sigma tensor", x_sigma_tensor_sha256),
        ("velocity target tensor", velocity_target_tensor_sha256),
        ("action velocity tensor", action_velocity_tensor_sha256),
        ("noop velocity tensor", noop_velocity_tensor_sha256),
        ("action full prompt", action_full_prompt_sha256),
        ("noop full prompt", noop_full_prompt_sha256),
        ("action condition tensor", action_condition_tensor_sha256),
        ("noop condition tensor", noop_condition_tensor_sha256),
        ("frozen model receipt", frozen_model_receipt_digest),
    ):
        _sha256(value, label=label)
    if effective_gaussian_tensor_sha256 != official_gaussian_tensor_sha256:
        raise TemporalCounterfactualContractError(
            "effective Gaussian must byte-match the official Gaussian"
        )
    execution_proof = _validated_same_state_execution_proof(
        same_state_execution_proof
    )
    unsigned = {
        "schema_version": PROMPT_PAIR_RECEIPT_SCHEMA,
        "candidate_id": candidate_id,
        "transform_name": transform_name,
        "native_schedule_index": native_schedule_index,
        "physical_sigma_float32_be_hex": coordinate[
            "physical_sigma_float32_be_hex"
        ],
        "native_scheduler_timestep": coordinate["native_scheduler_timestep"],
        "native_schedule_digest": NATIVE_SCHEDULE_DIGEST,
        "transformed_clean_tensor_sha256": transformed_clean_tensor_sha256,
        "official_gaussian_tensor_sha256": official_gaussian_tensor_sha256,
        "effective_gaussian_tensor_sha256": effective_gaussian_tensor_sha256,
        "gaussian_coupling_mode": gaussian_coupling_mode(transform_name),
        "x_sigma_tensor_sha256": x_sigma_tensor_sha256,
        "velocity_target_tensor_sha256": velocity_target_tensor_sha256,
        "action_velocity_tensor_sha256": action_velocity_tensor_sha256,
        "noop_velocity_tensor_sha256": noop_velocity_tensor_sha256,
        "action_full_prompt_sha256": action_full_prompt_sha256,
        "noop_full_prompt_sha256": noop_full_prompt_sha256,
        "action_condition_tensor_sha256": action_condition_tensor_sha256,
        "noop_condition_tensor_sha256": noop_condition_tensor_sha256,
        "frozen_model_receipt_digest": frozen_model_receipt_digest,
        "same_state_execution_proof": execution_proof,
        "same_x_sigma_object_for_prompt_pair": True,
        "same_native_coordinate_and_timestep_object_for_prompt_pair": True,
        "patch_vae_latent_calls_per_prompt_pair": 1,
        "prompt_order": ["target_action", "noop"],
        "target_only_t2v_forward": True,
        "source_or_rv2v_condition_consumed": False,
        "training_performed": False,
    }
    return _seal(unsigned)


def validate_prompt_pair_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _PROMPT_PAIR_FIELDS, label="prompt-pair receipt")
    _verify_seal(row, label="prompt-pair receipt")
    expected = make_prompt_pair_receipt(
        candidate_id=row["candidate_id"],
        transform_name=row["transform_name"],
        native_schedule_index=row["native_schedule_index"],
        transformed_clean_tensor_sha256=row["transformed_clean_tensor_sha256"],
        official_gaussian_tensor_sha256=row["official_gaussian_tensor_sha256"],
        effective_gaussian_tensor_sha256=row[
            "effective_gaussian_tensor_sha256"
        ],
        x_sigma_tensor_sha256=row["x_sigma_tensor_sha256"],
        velocity_target_tensor_sha256=row["velocity_target_tensor_sha256"],
        action_velocity_tensor_sha256=row["action_velocity_tensor_sha256"],
        noop_velocity_tensor_sha256=row["noop_velocity_tensor_sha256"],
        action_full_prompt_sha256=row["action_full_prompt_sha256"],
        noop_full_prompt_sha256=row["noop_full_prompt_sha256"],
        action_condition_tensor_sha256=row["action_condition_tensor_sha256"],
        noop_condition_tensor_sha256=row["noop_condition_tensor_sha256"],
        frozen_model_receipt_digest=row["frozen_model_receipt_digest"],
        same_state_execution_proof=row["same_state_execution_proof"],
    )
    if row != expected:
        raise TemporalCounterfactualContractError("prompt-pair receipt semantics differ")
    return row


_IDENTITY_FIELDS = frozenset(
    {
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
    }
)
_GENERATION_BINDING_FIELDS = frozenset(
    {
        "candidate_envelope_sha256",
        "generation_receipt_digest",
        "generation_receipt_file_sha256",
        "native_rollout_receipt_digest",
        "native_rollout_receipt_file_sha256",
        "generated_mp4_sha256",
        "geometry_source_video_sha256",
        "candidate_own_caption_utf8_sha256",
        "clean_latent_artifact_sha256",
        "clean_latent_tensor_sha256",
        "official_gaussian_artifact_sha256",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "official_gaussian_tensor_sha256",
    }
)
_TARGET_ACTION_BINDING_FIELDS = frozenset(
    {
        "target_action_candidate_id",
        "target_noop_candidate_id",
        "calibration_group_id",
        "target_action_caption_utf8_sha256",
        "target_noop_caption_utf8_sha256",
    }
)
_PROMPT_BINDING_FIELDS = frozenset(
    {
        "action_raw_caption_utf8_sha256",
        "noop_raw_caption_utf8_sha256",
        "action_full_prompt_utf8_sha256",
        "noop_full_prompt_utf8_sha256",
        "action_condition_tensor_sha256",
        "noop_condition_tensor_sha256",
        "prompt_builder_contract_digest",
        "prompt_pair_digest",
    }
)
_MODEL_BINDING_FIELDS = frozenset(
    {
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_manifest_sha256",
        "checkpoint_content_binding_digest",
        "d541801_scorer_source_revision",
        "d541801_scorer_source_sha256",
        "bernini_revision",
        "veomni_revision",
        "native_schedule_digest",
    }
)
_ENERGY_ROW_FIELDS = frozenset(
    {
        "native_schedule_index",
        "action_energy",
        "noop_energy",
        "prompt_pair_receipt",
    }
)
_SCORE_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "candidate_identity",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "bank_receipt_file_sha256",
        "generation_binding",
        "target_action_binding",
        "prompt_binding",
        "model_binding",
        "transform_plan",
        "sigma_coordinate_receipt",
        "energy_epsilon",
        "energy_by_transform",
        "chronological_prompt_contrast_by_sigma",
        "chronological_action_energy_rank_by_sigma",
        "transform_contributions",
        "hard_gates",
        "diagnostic_composite_score",
        "input_closure",
        "single_scalar_authorizes_optimizer",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)


def _validate_identity(value: Any) -> dict[str, Any]:
    row = _closed(value, _IDENTITY_FIELDS, label="candidate identity")
    for name in _IDENTITY_FIELDS - {"analysis_split", "semantic_branch"}:
        _safe_id(row[name], label=name)
    if row["analysis_split"] not in ANALYSIS_SPLITS:
        raise TemporalCounterfactualContractError("analysis split differs")
    if row["semantic_branch"] not in BRANCH_ORDER:
        raise TemporalCounterfactualContractError("semantic branch differs")
    return row


def _validate_hash_mapping(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    row = _closed(value, fields, label=label)
    for name, digest in row.items():
        _sha256(digest, label=f"{label} {name}")
    return row


def _validated_target_action_binding(value: Any, *, identity: Mapping[str, Any]) -> dict[str, Any]:
    row = _closed(value, _TARGET_ACTION_BINDING_FIELDS, label="target-action binding")
    _safe_id(row["target_action_candidate_id"], label="target action candidate ID")
    _safe_id(row["target_noop_candidate_id"], label="target no-op candidate ID")
    _safe_id(row["calibration_group_id"], label="target action calibration group")
    _sha256(
        row["target_action_caption_utf8_sha256"], label="target action caption SHA-256"
    )
    _sha256(
        row["target_noop_caption_utf8_sha256"], label="target no-op caption SHA-256"
    )
    if (
        row["calibration_group_id"] != identity["calibration_group_id"]
        or row["target_action_candidate_id"] == row["target_noop_candidate_id"]
        or row["target_action_caption_utf8_sha256"]
        == row["target_noop_caption_utf8_sha256"]
    ):
        raise TemporalCounterfactualContractError("target action belongs to another cell")
    return row


def _validated_prompt_binding(
    value: Any, *, target_action_binding: Mapping[str, Any]
) -> dict[str, Any]:
    row = _validate_hash_mapping(value, _PROMPT_BINDING_FIELDS, label="prompt binding")
    if (
        row["action_raw_caption_utf8_sha256"]
        != target_action_binding["target_action_caption_utf8_sha256"]
        or row["noop_raw_caption_utf8_sha256"]
        != target_action_binding["target_noop_caption_utf8_sha256"]
        or row["action_raw_caption_utf8_sha256"]
        == row["noop_raw_caption_utf8_sha256"]
        or row["action_full_prompt_utf8_sha256"]
        == row["noop_full_prompt_utf8_sha256"]
        or row["action_condition_tensor_sha256"]
        == row["noop_condition_tensor_sha256"]
    ):
        raise TemporalCounterfactualContractError("action/no-op prompt contrast binding differs")
    expected_pair = object_sha256(
        {
            "action_full_prompt_utf8_sha256": row[
                "action_full_prompt_utf8_sha256"
            ],
            "noop_full_prompt_utf8_sha256": row["noop_full_prompt_utf8_sha256"],
            "action_condition_tensor_sha256": row[
                "action_condition_tensor_sha256"
            ],
            "noop_condition_tensor_sha256": row["noop_condition_tensor_sha256"],
        }
    )
    if row["prompt_pair_digest"] != expected_pair:
        raise TemporalCounterfactualContractError("prompt-pair digest differs")
    return row


def _validated_model_binding(value: Any) -> dict[str, Any]:
    row = _closed(value, _MODEL_BINDING_FIELDS, label="model binding")
    for name in (
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_manifest_sha256",
        "checkpoint_content_binding_digest",
    ):
        _sha256(row[name], label=name)
    for name in ("bernini_revision", "veomni_revision"):
        _sha1(row[name], label=name)
    if (
        row["d541801_scorer_source_revision"] != REQUIRED_D541801_SCORER_REVISION
        or row["d541801_scorer_source_sha256"] != REQUIRED_D541801_SCORER_SHA256
        or row["checkpoint_content_manifest_sha256"]
        != REQUIRED_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or row["bernini_revision"] != REQUIRED_BERNINI_REVISION
        or row["veomni_revision"] != REQUIRED_VEOMNI_REVISION
        or row["native_schedule_digest"] != NATIVE_SCHEDULE_DIGEST
    ):
        raise TemporalCounterfactualContractError("frozen d541801 model authority differs")
    return row


def _normalize_energy_grid(
    value: Any,
    *,
    identity: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt: Mapping[str, Any],
    model: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != set(TRANSFORM_ORDER):
        raise TemporalCounterfactualContractError("energy transform closure differs")
    coordinate_order = [row[0] for row in NATIVE_SIGMA_COORDINATES]
    normalized: dict[str, list[dict[str, Any]]] = {}
    for transform_name in TRANSFORM_ORDER:
        rows = value[transform_name]
        if not isinstance(rows, list) or len(rows) != len(coordinate_order):
            raise TemporalCounterfactualContractError("energy sigma coverage differs")
        output_rows = []
        for expected_index, raw in zip(coordinate_order, rows):
            row = _closed(raw, _ENERGY_ROW_FIELDS, label="energy row")
            action_energy = _finite(
                row["action_energy"], label="action energy", nonnegative=True
            )
            noop_energy = _finite(
                row["noop_energy"], label="noop energy", nonnegative=True
            )
            if row["native_schedule_index"] != expected_index:
                raise TemporalCounterfactualContractError("energy sigma order differs")
            pair = validate_prompt_pair_receipt(row["prompt_pair_receipt"])
            if (
                pair["candidate_id"] != identity["candidate_id"]
                or pair["transform_name"] != transform_name
                or pair["native_schedule_index"] != expected_index
                or pair["official_gaussian_tensor_sha256"]
                != generation["official_gaussian_tensor_sha256"]
                or pair["action_full_prompt_sha256"]
                != prompt["action_full_prompt_utf8_sha256"]
                or pair["noop_full_prompt_sha256"]
                != prompt["noop_full_prompt_utf8_sha256"]
                or pair["action_condition_tensor_sha256"]
                != prompt["action_condition_tensor_sha256"]
                or pair["noop_condition_tensor_sha256"]
                != prompt["noop_condition_tensor_sha256"]
                or pair["frozen_model_receipt_digest"]
                != model["frozen_checkpoint_receipt_digest"]
            ):
                raise TemporalCounterfactualContractError(
                    "energy row/prompt-pair binding differs"
                )
            output_rows.append(
                {
                    "native_schedule_index": expected_index,
                    "action_energy": action_energy,
                    "noop_energy": noop_energy,
                    "prompt_pair_receipt": pair,
                }
            )
        transformed_hashes = {
            row["prompt_pair_receipt"]["transformed_clean_tensor_sha256"]
            for row in output_rows
        }
        target_hashes = {
            row["prompt_pair_receipt"]["velocity_target_tensor_sha256"]
            for row in output_rows
        }
        effective_gaussian_hashes = {
            row["prompt_pair_receipt"]["effective_gaussian_tensor_sha256"]
            for row in output_rows
        }
        if (
            len(transformed_hashes) != 1
            or len(target_hashes) != 1
            or len(effective_gaussian_hashes) != 1
            or next(iter(effective_gaussian_hashes))
            != generation["official_gaussian_tensor_sha256"]
        ):
            raise TemporalCounterfactualContractError(
                "every transform/sigma must reuse the byte-identical official Gaussian"
            )
        if (
            transform_name == CHRONOLOGICAL
            and (
                next(iter(transformed_hashes))
                != generation["clean_latent_tensor_sha256"]
            )
        ):
            raise TemporalCounterfactualContractError(
                "chronological arm differs from candidate clean/Gaussian tensors"
            )
        normalized[transform_name] = output_rows
    return normalized


def _derive_temporal_statistics(
    energy_by_transform: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, bool],
    float,
]:
    chronological = energy_by_transform[CHRONOLOGICAL]
    chronological_contrast = []
    for row in chronological:
        contrast = math.log(row["noop_energy"] + ENERGY_EPSILON) - math.log(
            row["action_energy"] + ENERGY_EPSILON
        )
        chronological_contrast.append(
            {
                "native_schedule_index": row["native_schedule_index"],
                "noop_minus_action_log_energy": float(contrast),
            }
        )

    rank_rows = []
    for sigma_ordinal, base in enumerate(chronological):
        energies = {
            name: float(energy_by_transform[name][sigma_ordinal]["action_energy"])
            for name in TRANSFORM_ORDER
        }
        ordered = sorted(TRANSFORM_ORDER, key=lambda name: (energies[name], TRANSFORM_ORDER.index(name)))
        chronological_energy = energies[CHRONOLOGICAL]
        multiset_others = [
            name
            for name in MULTISET_PRESERVING_TRANSFORMS
            if name != CHRONOLOGICAL
        ]
        tied_exact_multiset = [
            name
            for name in MULTISET_PRESERVING_TRANSFORMS
            if energies[name] == chronological_energy
        ]
        rank_rows.append(
            {
                "native_schedule_index": base["native_schedule_index"],
                "action_energy_order_low_to_high": ordered,
                "chronological_rank_among_all_transforms": 1
                + sum(value < chronological_energy for value in energies.values()),
                "chronological_rank_among_exact_multiset_arms": 1
                + sum(energies[name] < chronological_energy for name in multiset_others),
                "chronological_tied_exact_multiset_transform_names": tied_exact_multiset,
                "chronological_strictly_beats_other_exact_multiset_arms": all(
                    chronological_energy < energies[name] for name in multiset_others
                ),
            }
        )

    contributions: dict[str, dict[str, Any]] = {}
    for transform_name in COUNTERFACTUAL_TRANSFORMS:
        per_sigma = []
        for base, transformed in zip(
            chronological, energy_by_transform[transform_name]
        ):
            action_margin = math.log(
                transformed["action_energy"] + ENERGY_EPSILON
            ) - math.log(base["action_energy"] + ENERGY_EPSILON)
            noop_margin = math.log(
                transformed["noop_energy"] + ENERGY_EPSILON
            ) - math.log(base["noop_energy"] + ENERGY_EPSILON)
            per_sigma.append(
                {
                    "native_schedule_index": base["native_schedule_index"],
                    "action_chronological_margin": float(action_margin),
                    "noop_chronological_margin": float(noop_margin),
                    "prompt_specific_chronological_margin": float(
                        action_margin - noop_margin
                    ),
                }
            )
        preserving = transform_name in MULTISET_PRESERVING_TRANSFORMS
        contributions[transform_name] = {
            "control_class": (
                "exact_value_multiset_permutation"
                if preserving
                else "duplicate_or_omit_phase_control"
            ),
            "latent_value_multiset_preserved_by_construction": preserving,
            "per_sigma": per_sigma,
            "minimum_action_chronological_margin": float(
                min(row["action_chronological_margin"] for row in per_sigma)
            ),
            "minimum_noop_chronological_margin": float(
                min(row["noop_chronological_margin"] for row in per_sigma)
            ),
            "minimum_prompt_specific_chronological_margin": float(
                min(row["prompt_specific_chronological_margin"] for row in per_sigma)
            ),
        }

    reverse_rows = contributions["reverse"]["per_sigma"]
    gates = {
        "chronological_action_beats_noop_at_every_sigma": all(
            row["noop_minus_action_log_energy"] > 0.0
            for row in chronological_contrast
        ),
        "reverse_action_direction_hard_gate_all_sigmas": all(
            row["action_chronological_margin"] > 0.0 for row in reverse_rows
        ),
        "reverse_prompt_specific_hard_gate_all_sigmas": all(
            row["prompt_specific_chronological_margin"] > 0.0
            for row in reverse_rows
        ),
        "chronological_rank1_among_multiset_controls_all_sigmas": all(
            row["chronological_rank_among_exact_multiset_arms"] == 1
            and row["chronological_tied_exact_multiset_transform_names"]
            == [CHRONOLOGICAL]
            and row["chronological_strictly_beats_other_exact_multiset_arms"]
            for row in rank_rows
        ),
    }
    gates["candidate_hard_gate_passed"] = all(gates.values())
    diagnostic = min(
        row["minimum_prompt_specific_chronological_margin"]
        for row in contributions.values()
    )
    return chronological_contrast, rank_rows, contributions, gates, float(diagnostic)


def make_candidate_score_receipt(
    *,
    group_id: str,
    candidate_identity: Mapping[str, Any],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    bank_receipt_file_sha256: str,
    generation_binding: Mapping[str, Any],
    target_action_binding: Mapping[str, Any],
    prompt_binding: Mapping[str, Any],
    model_binding: Mapping[str, Any],
    energy_by_transform: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if group_id not in ("sp4-a", "sp4-b"):
        raise TemporalCounterfactualContractError("SP4 group differs")
    identity = _validate_identity(candidate_identity)
    generation = _validate_hash_mapping(
        generation_binding, _GENERATION_BINDING_FIELDS, label="generation binding"
    )
    target = _validated_target_action_binding(target_action_binding, identity=identity)
    prompt = _validated_prompt_binding(prompt_binding, target_action_binding=target)
    model = _validated_model_binding(model_binding)
    normalized_energy = _normalize_energy_grid(
        energy_by_transform,
        identity=identity,
        generation=generation,
        prompt=prompt,
        model=model,
    )
    chronological, ranks, contributions, hard_gates, diagnostic = (
        _derive_temporal_statistics(normalized_energy)
    )
    unsigned = {
        "schema_version": CANDIDATE_SCORE_SCHEMA,
        "group_id": group_id,
        "candidate_identity": identity,
        "root_spec_raw_sha256": _sha256(
            root_spec_raw_sha256, label="root spec SHA-256"
        ),
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "bank_receipt_file_sha256": _sha256(
            bank_receipt_file_sha256, label="bank receipt file SHA-256"
        ),
        "generation_binding": generation,
        "target_action_binding": target,
        "prompt_binding": prompt,
        "model_binding": model,
        "transform_plan": make_transform_plan(),
        "sigma_coordinate_receipt": make_sigma_coordinate_receipt(),
        "energy_epsilon": ENERGY_EPSILON,
        "energy_by_transform": normalized_energy,
        "chronological_prompt_contrast_by_sigma": chronological,
        "chronological_action_energy_rank_by_sigma": ranks,
        "transform_contributions": contributions,
        "hard_gates": hard_gates,
        "diagnostic_composite_score": diagnostic,
        "input_closure": T2V_INPUT_CLOSURE,
        "single_scalar_authorizes_optimizer": False,
        "training_performed": False,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    if (
        unsigned["root_spec_raw_sha256"] != REQUIRED_CORE4_V2_SPEC_SHA256
        or unsigned["bank_receipt_digest"]
        != REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or unsigned["bank_receipt_file_sha256"]
        != REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
    ):
        raise TemporalCounterfactualContractError("formal core4-v2 bank authority differs")
    return _seal(unsigned)


def validate_candidate_score_receipt(value: Any) -> dict[str, Any]:
    row = _closed(value, _SCORE_FIELDS, label="candidate score receipt")
    _verify_seal(row, label="candidate score receipt")
    expected = make_candidate_score_receipt(
        group_id=row["group_id"],
        candidate_identity=row["candidate_identity"],
        root_spec_raw_sha256=row["root_spec_raw_sha256"],
        bank_receipt_digest=row["bank_receipt_digest"],
        bank_receipt_file_sha256=row["bank_receipt_file_sha256"],
        generation_binding=row["generation_binding"],
        target_action_binding=row["target_action_binding"],
        prompt_binding=row["prompt_binding"],
        model_binding=row["model_binding"],
        energy_by_transform=row["energy_by_transform"],
    )
    if row != expected:
        raise TemporalCounterfactualContractError("candidate score semantics differ")
    return row


_GROUP_FIELDS = frozenset(
    {
        "schema_version",
        "group_id",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "bank_receipt_file_sha256",
        "candidate_count",
        "candidate_order",
        "candidate_order_digest",
        "candidate_identity_digest",
        "candidate_receipt_digests",
        "target_action_binding_by_cell",
        "prompt_pair_digest_by_cell",
        "transform_plan_digest",
        "sigma_coordinate_digest",
        "method_source_revision",
        "method_source_archive_sha256",
        "scorer_source_sha256",
        "contract_source_sha256",
        "d541801_scorer_source_sha256",
        "input_closure",
        "single_scalar_authorizes_optimizer",
        "training_performed",
        "optimizer_authorized",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)


def make_group_receipt(
    *,
    group_id: str,
    candidate_receipts: Sequence[Mapping[str, Any]],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    method_source_revision: str,
    method_source_archive_sha256: str,
    scorer_source_sha256: str,
    contract_source_sha256: str,
) -> dict[str, Any]:
    if group_id not in ("sp4-a", "sp4-b"):
        raise TemporalCounterfactualContractError("group receipt SP4 ID differs")
    rows = [validate_candidate_score_receipt(row) for row in candidate_receipts]
    if len(rows) != 20 or any(row["group_id"] != group_id for row in rows):
        raise TemporalCounterfactualContractError("group receipt requires exact ordered 20")
    if any(
        row["root_spec_raw_sha256"] != root_spec_raw_sha256
        or row["bank_receipt_digest"] != bank_receipt_digest
        or row["bank_receipt_digest"] != REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or row["bank_receipt_file_sha256"]
        != REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        for row in rows
    ):
        raise TemporalCounterfactualContractError("group candidate bank binding differs")
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_cell.setdefault(
            row["candidate_identity"]["calibration_group_id"], []
        ).append(row)
    if len(rows_by_cell) != 2 or any(
        [row["candidate_identity"]["semantic_branch"] for row in cell_rows]
        != list(BRANCH_ORDER)
        for cell_rows in rows_by_cell.values()
    ):
        raise TemporalCounterfactualContractError(
            "group receipt requires exact two-cell branch closure"
        )

    target_by_cell: dict[str, dict[str, Any]] = {}
    prompt_pair_by_cell: dict[str, str] = {}
    for row in rows:
        cell = row["candidate_identity"]["calibration_group_id"]
        binding = row["target_action_binding"]
        prior = target_by_cell.setdefault(cell, binding)
        if prior != binding:
            raise TemporalCounterfactualContractError("cell target-action binding drifted")
        pair_digest = row["prompt_binding"]["prompt_pair_digest"]
        prior_pair = prompt_pair_by_cell.setdefault(cell, pair_digest)
        if prior_pair != pair_digest:
            raise TemporalCounterfactualContractError("cell action/no-op prompt pair drifted")
    for cell, cell_rows in rows_by_cell.items():
        action_row = cell_rows[0]
        noop_row = cell_rows[1]
        binding = target_by_cell[cell]
        if (
            action_row["candidate_identity"]["semantic_branch"] != ACTION_BRANCH
            or noop_row["candidate_identity"]["semantic_branch"] != "noop"
            or binding["target_action_candidate_id"]
            != action_row["candidate_identity"]["candidate_id"]
            or binding["target_noop_candidate_id"]
            != noop_row["candidate_identity"]["candidate_id"]
            or binding["target_action_caption_utf8_sha256"]
            != action_row["generation_binding"]["candidate_own_caption_utf8_sha256"]
            or binding["target_noop_caption_utf8_sha256"]
            != noop_row["generation_binding"]["candidate_own_caption_utf8_sha256"]
        ):
            raise TemporalCounterfactualContractError(
                "cell action/no-op target binding is not joined to generation receipts"
            )
    candidate_order = [row["candidate_identity"]["candidate_id"] for row in rows]
    candidate_order_digest = object_sha256(candidate_order)
    if candidate_order_digest != REQUIRED_CORE4_V2_GROUP_ORDER_DIGESTS[group_id]:
        raise TemporalCounterfactualContractError(
            "group candidate order differs from formal core4-v2"
        )
    candidate_identity_digest = object_sha256(
        [row["candidate_identity"] for row in rows]
    )
    if (
        candidate_identity_digest
        != REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS[group_id]
    ):
        raise TemporalCounterfactualContractError(
            "group candidate identity mapping differs from formal core4-v2"
        )
    unsigned = {
        "schema_version": GROUP_RECEIPT_SCHEMA,
        "group_id": group_id,
        "root_spec_raw_sha256": _sha256(
            root_spec_raw_sha256, label="root spec SHA-256"
        ),
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "bank_receipt_file_sha256": REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256,
        "candidate_count": len(rows),
        "candidate_order": candidate_order,
        "candidate_order_digest": candidate_order_digest,
        "candidate_identity_digest": candidate_identity_digest,
        "candidate_receipt_digests": [row["receipt_digest"] for row in rows],
        "target_action_binding_by_cell": target_by_cell,
        "prompt_pair_digest_by_cell": prompt_pair_by_cell,
        "transform_plan_digest": make_transform_plan()["receipt_digest"],
        "sigma_coordinate_digest": make_sigma_coordinate_receipt()["receipt_digest"],
        "method_source_revision": _sha1(
            method_source_revision, label="method source revision"
        ),
        "method_source_archive_sha256": _sha256(
            method_source_archive_sha256, label="method source archive SHA-256"
        ),
        "scorer_source_sha256": _sha256(
            scorer_source_sha256, label="temporal scorer source SHA-256"
        ),
        "contract_source_sha256": _sha256(
            contract_source_sha256, label="temporal contract source SHA-256"
        ),
        "d541801_scorer_source_sha256": REQUIRED_D541801_SCORER_SHA256,
        "input_closure": T2V_INPUT_CLOSURE,
        "single_scalar_authorizes_optimizer": False,
        "training_performed": False,
        "optimizer_authorized": False,
        "scientific_action_editing_claim": False,
    }
    return _seal(unsigned)


def validate_group_receipt(
    value: Any,
    *,
    candidate_receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    row = _closed(value, _GROUP_FIELDS, label="score group receipt")
    _verify_seal(row, label="score group receipt")
    if (
        row["schema_version"] != GROUP_RECEIPT_SCHEMA
        or row["group_id"] not in ("sp4-a", "sp4-b")
        or row["candidate_count"] != 20
        or len(row["candidate_order"]) != 20
        or len(set(row["candidate_order"])) != 20
        or len(row["candidate_receipt_digests"]) != 20
        or len(set(row["candidate_receipt_digests"])) != 20
        or row["candidate_order_digest"]
        != REQUIRED_CORE4_V2_GROUP_ORDER_DIGESTS.get(row["group_id"])
        or object_sha256(row["candidate_order"]) != row["candidate_order_digest"]
        or row["candidate_identity_digest"]
        != REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS.get(row["group_id"])
        or row["root_spec_raw_sha256"] != REQUIRED_CORE4_V2_SPEC_SHA256
        or row["bank_receipt_digest"] != REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST
        or row["bank_receipt_file_sha256"]
        != REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256
        or row["transform_plan_digest"] != make_transform_plan()["receipt_digest"]
        or row["sigma_coordinate_digest"]
        != make_sigma_coordinate_receipt()["receipt_digest"]
        or row["d541801_scorer_source_sha256"] != REQUIRED_D541801_SCORER_SHA256
        or row["input_closure"] != T2V_INPUT_CLOSURE
        or row["single_scalar_authorizes_optimizer"] is not False
        or row["training_performed"] is not False
        or row["optimizer_authorized"] is not False
        or row["scientific_action_editing_claim"] is not False
    ):
        raise TemporalCounterfactualContractError("score group semantics differ")
    for candidate_id in row["candidate_order"]:
        _safe_id(candidate_id, label="group candidate ID")
    for digest in row["candidate_receipt_digests"]:
        _sha256(digest, label="group candidate receipt digest")
    for name in (
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "bank_receipt_file_sha256",
        "method_source_archive_sha256",
        "scorer_source_sha256",
        "contract_source_sha256",
    ):
        _sha256(row[name], label=name)
    _sha1(row["method_source_revision"], label="method source revision")
    if (
        not isinstance(row["target_action_binding_by_cell"], Mapping)
        or not isinstance(row["prompt_pair_digest_by_cell"], Mapping)
        or set(row["target_action_binding_by_cell"])
        != set(row["prompt_pair_digest_by_cell"])
    ):
        raise TemporalCounterfactualContractError("group cell target/prompt registry differs")
    for digest in row["prompt_pair_digest_by_cell"].values():
        _sha256(digest, label="group cell prompt-pair digest")
    if candidate_receipts is not None:
        candidates = [
            validate_candidate_score_receipt(candidate)
            for candidate in candidate_receipts
        ]
        expected = make_group_receipt(
            group_id=row["group_id"],
            candidate_receipts=candidates,
            root_spec_raw_sha256=row["root_spec_raw_sha256"],
            bank_receipt_digest=row["bank_receipt_digest"],
            method_source_revision=row["method_source_revision"],
            method_source_archive_sha256=row["method_source_archive_sha256"],
            scorer_source_sha256=row["scorer_source_sha256"],
            contract_source_sha256=row["contract_source_sha256"],
        )
        if row != expected:
            raise TemporalCounterfactualContractError(
                "score group candidate order/digest join differs"
            )
    return row


__all__ = [
    "ACTION_BRANCH",
    "ANALYSIS_SPLITS",
    "BRANCH_ORDER",
    "CANDIDATE_SCORE_SCHEMA",
    "CHRONOLOGICAL",
    "COUNTERFACTUAL_TRANSFORMS",
    "ENERGY_EPSILON",
    "FRAME_COUNT",
    "GROUP_RECEIPT_SCHEMA",
    "LATENT_PHASES",
    "MULTISET_PRESERVING_TRANSFORMS",
    "NATIVE_SCHEDULE_DIGEST",
    "NATIVE_SIGMA_COORDINATES",
    "NEGATIVE_BRANCHES",
    "NON_MULTISET_CONTROL_TRANSFORMS",
    "PROMPT_PAIR_RECEIPT_SCHEMA",
    "REQUIRED_CORE4_V2_BANK_RECEIPT_DIGEST",
    "REQUIRED_CORE4_V2_BANK_RECEIPT_FILE_SHA256",
    "REQUIRED_CORE4_V2_CANDIDATE_ORDER_DIGEST",
    "REQUIRED_CORE4_V2_CANDIDATE_IDENTITY_DIGEST",
    "REQUIRED_CORE4_V2_GROUP_IDENTITY_DIGESTS",
    "REQUIRED_CORE4_V2_GROUP_ORDER_DIGESTS",
    "REQUIRED_CORE4_V2_SPEC_SHA256",
    "REQUIRED_BERNINI_REVISION",
    "REQUIRED_CHECKPOINT_CONTENT_MANIFEST_SHA256",
    "REQUIRED_D541801_SCORER_REVISION",
    "REQUIRED_D541801_SCORER_SHA256",
    "REQUIRED_VEOMNI_REVISION",
    "T2V_INPUT_CLOSURE",
    "TRANSFORM_ORDER",
    "TemporalCounterfactualContractError",
    "apply_temporal_transform_sequence",
    "apply_temporal_transform_tensor",
    "canonical_json_bytes",
    "file_sha256",
    "fixed_official_gaussian_tensor",
    "gaussian_coupling_mode",
    "make_candidate_score_receipt",
    "make_group_receipt",
    "make_prompt_pair_receipt",
    "make_sigma_coordinate_receipt",
    "make_transform_plan",
    "object_sha256",
    "temporal_index_map",
    "validate_candidate_score_receipt",
    "validate_group_receipt",
    "validate_prompt_pair_receipt",
    "validate_sigma_coordinate_receipt",
    "validate_transform_plan",
]

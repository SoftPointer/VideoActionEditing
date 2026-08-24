#!/usr/bin/env python3
"""Authenticated WORLD4 runtime for the SAIL relational-motion stop gate.

The live query is deliberately a T2V-style action/no-op query at an already
materialized, authenticated native-RV2V clean endpoint.  It does *not* replay
the RV2V source condition.  A matching event-confirmed pure-T2V positive arm
supplies only its detached block-15 sketched residual.  A parameter-free
``FrozenRelationalMotionScorer`` replaces the learned STARC critic completely.

Rank zero publishes base/plus/minus exact81 videos and latent artifacts only
after a real differentiable SP4 VJP, fixed source-safe projection, fixed
RMS=0.03 symmetric intervention, artifact reopen, and video postflight.
This is a mechanism stop gate; it does not certify identity, camera, action
editing success, or any trainable editor.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_starc_core4_hidden_v1 as materializer  # noqa: E402
import run_starc_live_vjp_sp4_v1 as starc_runtime  # noqa: E402
import self_imagined_relational_motion as relational  # noqa: E402
import starch_live_vjp_bridge_v1 as live_bridge  # noqa: E402
import temporal_counterfactual_action_scorer_v1 as temporal_scorer  # noqa: E402


SCHEMA_VERSION = "bernini-sail-relational-motion-sp4-v1"
METHOD = "self-imagined-relational-motion-symmetric-cotangent"
FIXED_DOSE_RMS = 0.03
EXPECTED_WORLD_SIZE = 4
EXPECTED_TEACHER_SHAPE = (1, 21, 16, 1536)
TEACHER_TENSOR_KEY = materializer.TENSOR_KEY
TENSOR_KEY_CLEAN = starc_runtime.TENSOR_KEY_CLEAN
TENSOR_KEY_NOISE = starc_runtime.TENSOR_KEY_NOISE
FPS = 25
FRAME_COUNT = 81
LOADER_SOURCE_ARCHIVE_MEMBER = (
    "methods/bernini_action_editing/"
    "run_self_imagined_relational_motion_sp4_v1.py"
)
STATIC_SOURCE_CLOSURE = (
    LOADER_SOURCE_ARCHIVE_MEMBER,
    "methods/bernini_action_editing/"
    "audit_self_imagined_relational_specificity_v1.py",
    "methods/bernini_action_editing/self_imagined_relational_motion.py",
    "methods/bernini_action_editing/starch_live_vjp_bridge_v1.py",
    "methods/bernini_action_editing/run_starc_live_vjp_sp4_v1.py",
    "methods/bernini_action_editing/materialize_starc_core4_hidden_v1.py",
    "methods/bernini_action_editing/temporal_counterfactual_action_scorer_v1.py",
    "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
    "methods/bernini_action_editing/infer_lora.py",
    "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
    "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
    "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
    "methods/bernini_action_editing/scripts/"
    "auh_run_self_imagined_relational_motion_dual4_v1.sbatch",
    "methods/bernini_action_editing/tests/"
    "test_run_self_imagined_relational_motion_sp4_v1.py",
    "methods/bernini_action_editing/tests/"
    "test_auh_run_self_imagined_relational_motion_dual4_v1_launcher.py",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FROZEN_D541801_RUNTIME_DEPENDENCIES = {
    "infer_lora.py": "ce0bb91aa1850fa4568b6441cc4be4f41db8b8dcfc2afe2d9fcc76a6fca2ebe4",
    "infer_native_identity_generation_canary.py": (
        "a60c37591c40206c6130185f1a2d2a7a8e473f5af4425205e268ae4a8b58f334"
    ),
    "infer_source_kv_carrier_oracle.py": (
        "fcf77576735c89e685415b94b2dc0f0c5b8d1dd8dc1c55832538ff0daafb4604"
    ),
    "infer_source_value_residual_oracle.py": (
        "40e581db7906f20103a16ad47fda76978cbad21c9277723f3e8e022d717ed2d8"
    ),
    "source_self_native_ref_contrastive_v3.py": (
        "d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0"
    ),
}


class SelfImaginedRelationalRuntimeError(RuntimeError):
    """Fail-closed input, WORLD4, intervention, or publication error."""


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SelfImaginedRelationalRuntimeError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise SelfImaginedRelationalRuntimeError(
            f"{label} must be lowercase 40-hex revision"
        )
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise SelfImaginedRelationalRuntimeError(f"{label} is not path-safe")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _producer_ascii_object_sha256(value: Any) -> str:
    """Replay the identity-orbit producer's canonical ASCII receipt seal."""

    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SelfImaginedRelationalRuntimeError(
            "current base provenance receipt is not canonical finite JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _frozen_d541801_runtime_facade() -> Any:
    """Expose only the unchanged d541801 runtime helpers SAIL actually uses.

    The historical monolithic scorer file was later upgraded in-place from
    receipt v3 to v4.  SAIL consumes no MACE scorer, so importing that mutable
    file merely to reach native generation/prompt helpers is both unnecessary
    and impossible to authenticate.  This facade instead pins every unchanged
    dependency byte-for-byte to d541801 and replays the original prompt-builder
    contract locally.
    """

    import infer_native_identity_generation_canary as native_generation
    import source_self_native_ref_contrastive_v3 as native_schedule
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    modules = {
        "infer_lora.py": native_generation.legacy,
        "infer_native_identity_generation_canary.py": native_generation,
        "infer_source_kv_carrier_oracle.py": native_generation.source_audit,
        "infer_source_value_residual_oracle.py": native_generation.value_audit,
        "source_self_native_ref_contrastive_v3.py": native_schedule,
    }
    for filename, expected_sha256 in _FROZEN_D541801_RUNTIME_DEPENDENCIES.items():
        module = modules[filename]
        source = Path(module.__file__).resolve(strict=True)
        if (
            source != METHOD_ROOT / filename
            or source.is_symlink()
            or starc_runtime.file_sha256(source) != expected_sha256
        ):
            raise SelfImaginedRelationalRuntimeError(
                f"d541801 runtime dependency differs: {filename}"
            )

    def prompt_builder_contract() -> Mapping[str, Any]:
        task_name = native_generation.ARM_TRAINING_TASK_NAMES["t2v"]
        system = native_generation.TASK_SYSTEM_PROMPTS[task_name]
        clause = native_generation.TASK_BINDING_CLAUSES["t2v"]
        try:
            builder_source = inspect.getsource(native_generation.build_task_prompt)
            cleaner_source = inspect.getsource(prompt_clean)
        except (OSError, TypeError) as error:
            raise SelfImaginedRelationalRuntimeError(
                "official prompt-builder source is unavailable"
            ) from error
        value = {
            "builder": "infer_native_identity_generation_canary.build_task_prompt",
            "arm": "t2v",
            "training_task_name": task_name,
            "prompt_cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "system_prompt_utf8_sha256": hashlib.sha256(
                system.encode("utf-8")
            ).hexdigest(),
            "task_binding_clause_utf8_sha256": hashlib.sha256(
                clause.encode("utf-8")
            ).hexdigest(),
            "builder_source_utf8_sha256": hashlib.sha256(
                builder_source.encode("utf-8")
            ).hexdigest(),
            "prompt_cleaner_source_utf8_sha256": hashlib.sha256(
                cleaner_source.encode("utf-8")
            ).hexdigest(),
        }
        return {**value, "contract_digest": _object_sha256(value)}

    def tensor_sha256(value: Any) -> str:
        import torch

        if not isinstance(value, torch.Tensor) or value.device.type == "meta":
            raise SelfImaginedRelationalRuntimeError(
                "d541801 tensor hash requires a materialized tensor"
            )
        cpu = value.detach().to(device="cpu").contiguous().clone()
        metadata = {
            "shape": [int(item) for item in cpu.shape],
            "dtype": str(cpu.dtype),
            "layout": str(cpu.layout),
        }
        canonical = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
        raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
        return hashlib.sha256(canonical + b"\x00" + raw).hexdigest()

    return SimpleNamespace(
        native_generation=native_generation,
        native_schedule=native_schedule,
        prompt_builder_contract=prompt_builder_contract,
        tensor_sha256=tensor_sha256,
    )


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelfImaginedRelationalRuntimeError(
                f"JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _strict_json(path: Path, *, expected_sha256: str, label: str) -> Mapping[str, Any]:
    authenticated = starc_runtime._authenticated_file(
        path, expected_sha256, label=label
    )
    try:
        value = json.loads(
            authenticated.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SelfImaginedRelationalRuntimeError(
                    f"{label} contains non-finite JSON number {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelfImaginedRelationalRuntimeError(
            f"{label} is not canonical strict JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise SelfImaginedRelationalRuntimeError(f"{label} must be an object")
    return value


def validate_current_base_provenance_receipt(
    receipt: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    action_caption_sha256: str,
    clean_path: Path,
    clean_file_sha256: str,
    noise_path: Path,
    noise_file_sha256: str,
    base_mp4_path: Path,
    base_mp4_sha256: str,
    latent_shape: Sequence[int],
) -> Mapping[str, Any]:
    """Bind clean/noise/MP4 to the same sealed native-RV2V base rollout."""

    row = dict(receipt)
    declared = row.pop("receipt_digest", None)
    if (
        not isinstance(declared, str)
        or _SHA256_RE.fullmatch(declared) is None
        or row.get("schema_version")
        != "bernini-identity-orbit-heldout-role-composition-receipt-v1"
        or _producer_ascii_object_sha256(row) != declared
    ):
        raise SelfImaginedRelationalRuntimeError(
            "current base provenance receipt seal differs"
        )
    cell_spec = row.get("cell_spec")
    cell = cell_spec.get("cell") if isinstance(cell_spec, Mapping) else None
    source = row.get("source")
    model = row.get("model")
    sampling = row.get("sampling")
    outputs = row.get("outputs")
    base = outputs.get("base") if isinstance(outputs, Mapping) else None
    latent = (
        base.get("normalized_clean_latent") if isinstance(base, Mapping) else None
    )
    noises = row.get("initial_noise_artifacts")
    noise = noises.get("base") if isinstance(noises, Mapping) else None
    checkpoint_content = (
        model.get("checkpoint_content") if isinstance(model, Mapping) else None
    )
    expected_shape = [int(item) for item in latent_shape]
    expected_hw = [expected_shape[-2] * 8, expected_shape[-1] * 8]
    if (
        not isinstance(cell, Mapping)
        or cell.get("source_video") != str(source_path)
        or cell.get("source_video_sha256") != source_sha256
        or cell.get("action_caption_utf8_sha256") != action_caption_sha256
        or not isinstance(source, Mapping)
        or source.get("path") != str(source_path)
        or source.get("sha256") != source_sha256
        or not isinstance(model, Mapping)
        or model.get("bernini_commit") != live_bridge.BERNINI_OFFICIAL_COMMIT
        or model.get("veomni_commit") != live_bridge.VEOMNI_TESTED_COMMIT
        or model.get("checkpoint_tree_sha256")
        != live_bridge.BERNINI_CHECKPOINT_TREE_SHA256
        or model.get("checkpoint_unchanged") is not True
        or not isinstance(checkpoint_content, Mapping)
        or checkpoint_content.get("manifest_sha256_computed")
        != live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or checkpoint_content.get("manifest_sha256_expected")
        != live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
        or checkpoint_content.get("every_file_sha256_verified") is not True
        or not isinstance(sampling, Mapping)
        or sampling.get("exact81") is not True
        or sampling.get("frame_count") != FRAME_COUNT
        or sampling.get("fps") != FPS
        or sampling.get("latent_phases") != 21
        or sampling.get("num_inference_steps") != 40
        or sampling.get("native_unipc_shift5") is not True
        or sampling.get("source_rich_noise") is not False
        or not isinstance(base, Mapping)
        or base.get("path") != str(base_mp4_path)
        or base.get("sha256") != base_mp4_sha256
        or base.get("frame_count") != FRAME_COUNT
        or base.get("fps") != FPS
        or [base.get("height"), base.get("width")] != expected_hw
        or not isinstance(latent, Mapping)
        or latent.get("path") != str(clean_path)
        or latent.get("sha256") != clean_file_sha256
        or latent.get("shape") != expected_shape
        or latent.get("tensor_key") != TENSOR_KEY_CLEAN
        or latent.get("native_sampler_before_vae_decode") is not True
        or latent.get("coordinate") != "bernini_normalized_clean_vae_latent"
        or latent.get("stored_dtype") != "torch.float32"
        or latent.get("roundtrip_byte_exact_fp32") is not True
        or not isinstance(noise, Mapping)
        or noise.get("path") != str(noise_path)
        or noise.get("sha256") != noise_file_sha256
        or noise.get("shape") != expected_shape
        or noise.get("tensor_key") != TENSOR_KEY_NOISE
        or noise.get("captured_from_native_sampler") is not True
        or noise.get("observer_only") is not True
        or noise.get("original_return_tensor_forwarded_by_identity") is not True
        or noise.get("roundtrip_raw_value_exact") is not True
        or noise.get("source_or_target_derived") is not False
        or noise.get("generator_initial_seed") != cell.get("target_seed")
        or noise.get("external_initial_noise_injection") is not False
        or noise.get("sampler_noise_replacement") is not False
    ):
        raise SelfImaginedRelationalRuntimeError(
            "clean/noise/base MP4 do not close one sealed native-RV2V base rollout"
        )
    return {
        "schema_version": receipt["schema_version"],
        "receipt_digest": declared,
        "source_video_sha256": source_sha256,
        "action_caption_sha256": action_caption_sha256,
        "clean_latent_file_sha256": clean_file_sha256,
        "official_gaussian_file_sha256": noise_file_sha256,
        "base_mp4_file_sha256": base_mp4_sha256,
        "latent_shape": expected_shape,
        "same_native_rv2v_base_rollout": True,
    }


def validate_positive_teacher_binding(
    receipt: Mapping[str, Any], *, expected_episode_id: str
) -> Mapping[str, Any]:
    """Validate the sealed positive arm before its tensor can become a teacher.

    File/path hashes and the tensor itself are authenticated separately by
    :func:`load_positive_teacher`; this function locks semantic/query/model
    authority.  It intentionally accepts no learned-critic checkpoint.
    """

    _safe_id(expected_episode_id, label="expected teacher episode ID")
    try:
        row = materializer.validate_arm_receipt(receipt, verify_artifact=True)
    except Exception as error:
        raise SelfImaginedRelationalRuntimeError(
            "teacher arm receipt failed sealed materializer validation"
        ) from error
    event = row.get("event_label_binding")
    query = row.get("same_state_query_binding")
    hidden = row.get("hidden_binding")
    artifact = row.get("artifact")
    model = row.get("model_binding")
    prompt = row.get("prompt_binding")
    source_candidate = row.get("source_candidate_binding")
    denials = (
        "training_performed",
        "optimizer_authorized",
        "editor_optimizer_authorized",
        "scientific_critic_claim_authorized",
        "generated_media_editor_use_authorized",
    )
    if (
        row.get("episode_id") != expected_episode_id
        or row.get("split") != "fit"
        or row.get("role") != "positive"
        or row.get("label") != 1
        or any(row.get(name) is not False for name in denials)
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher is not the matching fit positive with closed authority"
        )
    if (
        not isinstance(event, Mapping)
        or event.get("complete_target_transition_observed") is not True
        or event.get("terminal_hold_observed") is not True
        or event.get("full_target_action_observed") is not True
        or event.get("full_target_action_false_confirmed") is not False
        or event.get("labels_are_external_and_detached") is not True
        or event.get("labels_may_enter_model_condition") is not False
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher lacks the complete-action/terminal-hold event proof"
        )
    if (
        not isinstance(query, Mapping)
        or query.get("native_schedule_index") != materializer.SCHEDULE_INDEX
        or query.get("native_timestep") != materializer.NATIVE_TIMESTEP
        or float(query.get("sigma", -1.0)).hex()
        != float(materializer.SIGMA).hex()
        or query.get("action_and_noop_share_exact_x_sigma_object") is not True
        or query.get("action_and_noop_share_exact_rotary_object") is not True
        or query.get("action_and_noop_share_exact_timestep_object") is not True
        or query.get("shared_tensor_bytes_unchanged") is not True
        or query.get("block0_input_and_attn1_exact_parity") is not True
        or query.get("source_condition_consumed") is not False
        or query.get("mask_flow_pose_track_or_trajectory_consumed") is not False
        or query.get("event_labels_consumed") is not False
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher is not the fixed index33 same-state action/no-op query"
        )
    if (
        not isinstance(hidden, Mapping)
        or hidden.get("hook_coordinate") != materializer.HOOK_COORDINATE
        or hidden.get("residual_shape") != list(EXPECTED_TEACHER_SHAPE)
        or hidden.get("full_hidden_persisted") is not False
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher block15 residual geometry differs"
        )
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("tensor_key") != TEACHER_TENSOR_KEY
        or artifact.get("tensor_shape") != list(EXPECTED_TEACHER_SHAPE)
        or artifact.get("tensor_dtype") != "torch.float32"
        or artifact.get("detached_finite_fp32") is not True
    ):
        raise SelfImaginedRelationalRuntimeError("teacher artifact binding differs")
    if (
        not isinstance(model, Mapping)
        or model.get("bernini_revision") != live_bridge.BERNINI_OFFICIAL_COMMIT
        or model.get("veomni_revision") != live_bridge.VEOMNI_TESTED_COMMIT
        or model.get("native_schedule_digest")
        != temporal_scorer.contract.NATIVE_SCHEDULE_DIGEST
        or model.get("native_schedule_index") != materializer.SCHEDULE_INDEX
        or model.get("native_timestep") != materializer.NATIVE_TIMESTEP
        or float(model.get("sigma", -1.0)).hex()
        != float(materializer.SIGMA).hex()
        or model.get("hook_coordinate") != materializer.HOOK_COORDINATE
        or model.get("transformer_1_only") is not True
        or model.get("adapter_loaded") is not False
        or model.get("all_parameters_frozen") is not True
        or not isinstance(model.get("checkpoint_content_binding"), Mapping)
    ):
        raise SelfImaginedRelationalRuntimeError("teacher frozen-model binding differs")
    checkpoint_content = dict(model["checkpoint_content_binding"])
    freeze_certificate = {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }
    checkpoint_unsigned = {
        "manifest_sha256": live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        "verified_file_count": 23,
        "verified_entries_digest": (
            "676e6104eebee3ab1066c70f40af385346b013a3afcab8cafb06c5290994d9ba"
        ),
        "every_file_sha256_verified": True,
        "loaded_components": ["transformer_1", "umt5_text_encoder"],
        "all_loaded_parameters_frozen": True,
        "freeze_certificate": freeze_certificate,
    }
    if (
        set(checkpoint_content) != {*checkpoint_unsigned, "binding_digest"}
        or any(
            checkpoint_content.get(key) != value
            for key, value in checkpoint_unsigned.items()
        )
        or checkpoint_content.get("binding_digest")
        != _object_sha256(checkpoint_unsigned)
        or not isinstance(model.get("frozen_checkpoint_receipt_digest"), str)
        or _SHA256_RE.fullmatch(model["frozen_checkpoint_receipt_digest"]) is None
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher checkpoint-content/freeze binding differs"
        )
    if not isinstance(prompt, Mapping):
        raise SelfImaginedRelationalRuntimeError("teacher prompt binding is absent")
    if (
        not isinstance(source_candidate, Mapping)
        or source_candidate.get("semantic_branch")
        != materializer.dataset_contract.ACTION_BRANCH
        or prompt.get("target_action_candidate_id")
        != source_candidate.get("candidate_id")
        or prompt.get("target_noop_candidate_id")
        == source_candidate.get("candidate_id")
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher is not the bound pure-T2V action candidate"
        )
    for key in (
        "action_raw_caption_utf8_sha256",
        "noop_raw_caption_utf8_sha256",
        "action_full_prompt_utf8_sha256",
        "noop_full_prompt_utf8_sha256",
        "action_condition_tensor_sha256",
        "noop_condition_tensor_sha256",
        "prompt_builder_contract_digest",
        "prompt_pair_digest",
    ):
        _sha256(prompt.get(key), label=f"teacher {key}")
    expected_pair = _object_sha256(
        {
            "action_full_prompt_utf8_sha256": prompt[
                "action_full_prompt_utf8_sha256"
            ],
            "noop_full_prompt_utf8_sha256": prompt[
                "noop_full_prompt_utf8_sha256"
            ],
            "action_condition_tensor_sha256": prompt[
                "action_condition_tensor_sha256"
            ],
            "noop_condition_tensor_sha256": prompt[
                "noop_condition_tensor_sha256"
            ],
        }
    )
    if (
        prompt["prompt_pair_digest"] != expected_pair
        or prompt.get("all_13_arms_use_cell_fixed_prompt_pair") is not True
        or prompt.get("branch_caption_never_used_as_condition") is not True
        or prompt.get("detached_labels_never_used_as_condition") is not True
    ):
        raise SelfImaginedRelationalRuntimeError("teacher prompt-pair binding differs")
    return row


strict_positive_teacher_binding = validate_positive_teacher_binding


def load_positive_teacher(
    *,
    receipt_path: str | Path,
    expected_receipt_sha256: str,
    artifact_path: str | Path,
    expected_artifact_sha256: str,
    expected_tensor_sha256: str,
    expected_episode_id: str,
    action_caption_sha256: str,
    noop_caption_sha256: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    """Authenticate receipt bytes, artifact bytes, key, value, and prompt pair."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise SelfImaginedRelationalRuntimeError(
            "PyTorch and safetensors are required"
        ) from error
    receipt_file = starc_runtime._authenticated_file(
        receipt_path, expected_receipt_sha256, label="positive teacher receipt"
    )
    receipt = _strict_json(
        receipt_file,
        expected_sha256=expected_receipt_sha256,
        label="positive teacher receipt",
    )
    row = validate_positive_teacher_binding(
        receipt, expected_episode_id=expected_episode_id
    )
    artifact_file = starc_runtime._authenticated_file(
        artifact_path, expected_artifact_sha256,
        label="positive teacher residual artifact",
    )
    artifact = row["artifact"]
    if (
        str(artifact_file) != artifact.get("path")
        or expected_artifact_sha256 != artifact.get("file_sha256")
        or expected_tensor_sha256 != artifact.get("tensor_sha256")
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher receipt/CLI artifact binding differs"
        )
    with safe_open(str(artifact_file), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [TEACHER_TENSOR_KEY]:
            raise SelfImaginedRelationalRuntimeError("teacher tensor-key closure differs")
        tensor = opened.get_tensor(TEACHER_TENSOR_KEY).contiguous()
    if (
        tensor.dtype != torch.float32
        or tuple(int(v) for v in tensor.shape) != EXPECTED_TEACHER_SHAPE
        or tensor.requires_grad
        or tensor.grad_fn is not None
        or not bool(torch.isfinite(tensor).all().item())
        or materializer.tensor_sha256(tensor) != expected_tensor_sha256
        or starc_runtime.file_sha256(artifact_file) != expected_artifact_sha256
    ):
        raise SelfImaginedRelationalRuntimeError("teacher tensor value differs")
    prompt = row["prompt_binding"]
    if (
        prompt["action_raw_caption_utf8_sha256"] != action_caption_sha256
        or prompt["noop_raw_caption_utf8_sha256"] != noop_caption_sha256
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher and current raw action/no-op captions do not match"
        )
    binding = {
        "receipt_path": str(receipt_file),
        "receipt_file_sha256": expected_receipt_sha256,
        "receipt_digest": row["receipt_digest"],
        "episode_id": row["episode_id"],
        "role": row["role"],
        "event_confirmed_complete_action_and_hold": True,
        "artifact_path": str(artifact_file),
        "artifact_file_sha256": expected_artifact_sha256,
        "tensor_key": TEACHER_TENSOR_KEY,
        "tensor_sha256": expected_tensor_sha256,
        "tensor_shape": list(EXPECTED_TEACHER_SHAPE),
    }
    return tensor.detach().clone(), row, binding


load_positive_teacher_binding = load_positive_teacher


def build_fixed_dose_interventions(clean: Any, q: Any) -> Any:
    """The only production intervention dose; it is not a CLI hyperparameter."""

    return relational.symmetric_latent_interventions(
        clean.detach().float().contiguous(),
        q.detach().float().contiguous(),
        dose_rms=FIXED_DOSE_RMS,
    )


def _save_tensor_atomically(
    path: Path, *, key: str, tensor: Any, metadata: Mapping[str, str]
) -> Mapping[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if path.exists() or path.is_symlink() or path.suffix != ".safetensors":
        raise SelfImaginedRelationalRuntimeError("tensor output must be fresh")
    stored = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if stored.requires_grad or not bool(torch.isfinite(stored).all().item()):
        raise SelfImaginedRelationalRuntimeError("output tensor must be finite FP32")
    fd, raw_temp = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".safetensors"
    )
    os.close(fd)
    temporary = Path(raw_temp)
    try:
        save_file({key: stored}, str(temporary), metadata=dict(metadata))
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != [key]:
                raise SelfImaginedRelationalRuntimeError("saved tensor key differs")
            restored = opened.get_tensor(key).contiguous()
        if restored.dtype != torch.float32 or not torch.equal(restored, stored):
            raise SelfImaginedRelationalRuntimeError("saved tensor roundtrip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return {
        "path": str(path),
        "file_sha256": starc_runtime.file_sha256(path),
        "tensor_key": key,
        "tensor_sha256": live_bridge._tensor_value_digest(stored, label=key),
        "shape": [int(v) for v in stored.shape],
        "dtype": "torch.float32",
        "roundtrip_exact": True,
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise SelfImaginedRelationalRuntimeError("receipt output must be fresh")
    payload = _canonical_json_bytes(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists() or path.is_symlink():
            path.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


def _validate_video(path: Path) -> Mapping[str, Any]:
    from tools import materialize_vae

    frozen = _frozen_d541801_runtime_facade()
    frames, fps, hw = materialize_vae._decode_exact_video(path)
    frozen.native_generation.legacy.validate_exact_video_metadata(
        int(frames.shape[0]), fps
    )
    if int(frames.shape[0]) != FRAME_COUNT or int(round(float(fps))) != FPS:
        raise SelfImaginedRelationalRuntimeError("video is not exact81/fps25")
    return {
        "path": str(path),
        "file_sha256": starc_runtime.file_sha256(path),
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "height": int(hw[0]),
        "width": int(hw[1]),
    }


def _save_rank0_outputs(
    *,
    final_dir: Path,
    published_dir: Path,
    base: Any,
    pair: Any,
    raw_q: Any,
    source_video: Path,
    base_mp4: Path,
    checkpoint: Path,
    device: Any,
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create a sibling staging directory and publish the child atomically."""

    import torch
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode
    from diffusers.models import AutoencoderKLWan
    import infer_source_value_residual_oracle as video_io

    if final_dir.exists() or final_dir.is_symlink():
        raise SelfImaginedRelationalRuntimeError("child output must be fresh")
    parent = starc_runtime._plain_directory(final_dir.parent, label="output parent")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.stage-", dir=parent)
    )
    published = False
    try:
        raw_artifact = _save_tensor_atomically(
            staging / "raw-cotangent.safetensors", key="raw_clean_latent_cotangent",
            tensor=raw_q,
            metadata={"coordinate": "current_rv2v_clean_endpoint", "projected": "false"},
        )
        projected_artifact = _save_tensor_atomically(
            staging / "projected-cotangent.safetensors",
            key="nuisance_null_projected_cotangent",
            tensor=pair.projected_cotangent,
            metadata={
                "coordinate": "current_rv2v_clean_endpoint",
                "projected": "true",
                "projector": "fixed_phase0_temporal_dc_spatial_affine_null",
            },
        )
        delta_artifact = _save_tensor_atomically(
            staging / "fixed-dose-delta.safetensors",
            key="fixed_rms_nuisance_null_delta",
            tensor=pair.delta,
            metadata={
                "coordinate": "current_rv2v_clean_endpoint",
                "projected": "true",
                "dose_rms": float(FIXED_DOSE_RMS).hex(),
                "arm_selection": "none",
            },
        )
        latent_rows: dict[str, Mapping[str, Any]] = {}
        for arm, value in (("base", base), ("plus", pair.plus), ("minus", pair.minus)):
            latent_rows[arm] = _save_tensor_atomically(
                staging / f"{arm}.normalized-clean-latent.safetensors",
                key="normalized_clean_latent", tensor=value,
                metadata={
                    "coordinate": "bernini_normalized_clean_vae_latent",
                    "frame_contract": "exact81_latent21",
                    "artifact_role": (
                        "authenticated_current_rv2v_endpoint"
                        if arm == "base" else "fixed_rms_symmetric_intervention"
                    ),
                    "arm": arm,
                    "dose_rms": float(FIXED_DOSE_RMS).hex(),
                },
            )
        source_input_video = _validate_video(source_video)
        base_input_video = _validate_video(base_mp4)
        expected_hw = (base_input_video["height"], base_input_video["width"])
        if expected_hw != (int(base.shape[-2]) * 8, int(base.shape[-1]) * 8):
            raise SelfImaginedRelationalRuntimeError(
                "base MP4/latent spatial geometry differs"
            )
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint), subfolder="vae", torch_dtype=torch.float32,
            local_files_only=True,
        ).eval().requires_grad_(False).to(device)
        video_rows: dict[str, Mapping[str, Any]] = {}
        try:
            for arm, value in (("base", base), ("plus", pair.plus), ("minus", pair.minus)):
                with torch.no_grad():
                    decoded = _vae_decode(vae, value.to(device))
                if tuple(int(v) for v in decoded.shape) != (
                    FRAME_COUNT, expected_hw[0], expected_hw[1], 3
                ):
                    raise SelfImaginedRelationalRuntimeError(
                        f"{arm} VAE decoded geometry differs"
                    )
                output = staging / f"{arm}.mp4"
                video_io.save_video_atomically(
                    decoded, output, fps=FPS, save_output_fn=save_output
                )
                video_rows[arm] = _validate_video(output)
        finally:
            vae.to("cpu")
            del vae
            torch.cuda.empty_cache()
        unsigned = {
            **dict(receipt),
            # Receipts name the post-rename location, never the hidden staging
            # inode name.  The bytes are verified below at their current
            # staging paths before the directory-level atomic rename.
            "raw_cotangent_artifact": {
                **raw_artifact,
                "path": str(published_dir / Path(raw_artifact["path"]).name),
            },
            "projected_cotangent_artifact": {
                **projected_artifact,
                "path": str(published_dir / Path(projected_artifact["path"]).name),
            },
            "fixed_dose_delta_artifact": {
                **delta_artifact,
                "path": str(published_dir / Path(delta_artifact["path"]).name),
            },
            "latent_artifacts": {
                arm: {**row, "path": str(published_dir / Path(row["path"]).name)}
                for arm, row in latent_rows.items()
            },
            "video_artifacts": {
                arm: {**row, "path": str(published_dir / Path(row["path"]).name)}
                for arm, row in video_rows.items()
            },
            "authenticated_input_base_mp4": base_input_video,
            "authenticated_input_source_video": source_input_video,
        }
        sealed = {**unsigned, "receipt_digest": _object_sha256(unsigned)}
        _write_json_create_only(staging / "receipt.json", sealed)
        for row in (
            raw_artifact,
            projected_artifact,
            delta_artifact,
            *latent_rows.values(),
            *video_rows.values(),
        ):
            path = Path(row["path"])
            # Paths above still point into staging and must be byte-stable.
            if starc_runtime.file_sha256(path) != row["file_sha256"]:
                raise SelfImaginedRelationalRuntimeError(
                    "rank-zero artifact changed before publication"
                )
        os.replace(staging, final_dir)
        published = True
        return sealed
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def _validate_run_cli(args: argparse.Namespace) -> Mapping[str, Any]:
    if (
        args.expected_bernini_commit != live_bridge.BERNINI_OFFICIAL_COMMIT
        or args.expected_veomni_commit != live_bridge.VEOMNI_TESTED_COMMIT
        or args.expected_checkpoint_tree_sha256
        != live_bridge.BERNINI_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != live_bridge.BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        raise SelfImaginedRelationalRuntimeError("official model pin differs")
    _sha1(args.source_git_revision, label="source git revision")
    for name in (
        "expected_source_archive_sha256", "expected_loader_source_sha256",
        "expected_candidate_manifest_sha256", "expected_source_video_sha256",
        "expected_action_caption_file_sha256", "expected_noop_caption_file_sha256",
        "expected_current_clean_latent_sha256", "expected_native_noise_sha256",
        "expected_base_mp4_sha256", "expected_base_receipt_sha256",
        "expected_teacher_receipt_sha256",
        "expected_teacher_residual_sha256",
        "expected_teacher_residual_tensor_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "ack_mechanism_probe_only", "ack_no_editor_parameter_or_update",
        "ack_no_scientific_or_action_editing_claim",
        "ack_live_query_has_no_source_condition",
    ):
        if getattr(args, name) is not True:
            raise SelfImaginedRelationalRuntimeError(
                f"mandatory acknowledgement missing: {name}"
            )
    if not math.isfinite(args.minimum_norm) or args.minimum_norm <= 0.0:
        raise SelfImaginedRelationalRuntimeError("minimum norm must be positive")
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise SelfImaginedRelationalRuntimeError("output dir must be fresh absolute")
    starc_runtime._plain_directory(output.parent, label="output parent")
    published_output = Path(args.published_output_dir)
    if (
        not published_output.is_absolute()
        or published_output == Path("/")
        or published_output.exists()
        or published_output.is_symlink()
        or published_output == output
        or published_output != Path(os.path.normpath(str(published_output)))
    ):
        raise SelfImaginedRelationalRuntimeError(
            "published output dir must be a distinct fresh canonical absolute path"
        )
    loader = starc_runtime._plain_file(Path(__file__).resolve(), label="runtime loader")
    if starc_runtime.file_sha256(loader) != args.expected_loader_source_sha256:
        raise SelfImaginedRelationalRuntimeError("executing loader hash differs")
    source_archive = starc_runtime._authenticated_file(
        args.source_archive, args.expected_source_archive_sha256,
        label="method source archive",
    )
    source = starc_runtime._authenticated_file(
        args.source_video, args.expected_source_video_sha256,
        label="current source video",
    )
    base_mp4 = starc_runtime._authenticated_file(
        args.base_mp4, args.expected_base_mp4_sha256,
        label="current native RV2V base MP4",
    )
    action, action_path, action_sha = starc_runtime.load_canonical_text_file(
        args.action_caption_file, args.expected_action_caption_file_sha256,
        label="current action caption",
    )
    noop, noop_path, noop_sha = starc_runtime.load_canonical_text_file(
        args.noop_caption_file, args.expected_noop_caption_file_sha256,
        label="scene-matched no-op caption",
    )
    if action == noop or action_sha == noop_sha:
        raise SelfImaginedRelationalRuntimeError("action and no-op captions alias")
    candidate = live_bridge.authenticate_current_candidate_manifest(
        args.candidate_manifest,
        expected_manifest_sha256=args.expected_candidate_manifest_sha256,
        instruction=action,
    )
    if candidate.source_video_sha256 != args.expected_source_video_sha256:
        raise SelfImaginedRelationalRuntimeError("candidate/source binding differs")
    clean, clean_path, clean_file_sha, clean_sha = (
        starc_runtime.load_authenticated_exact81_tensor(
            args.current_clean_latent,
            args.expected_current_clean_latent_sha256,
            tensor_key=TENSOR_KEY_CLEAN,
            label="current native RV2V clean endpoint",
        )
    )
    noise, noise_path, noise_file_sha, noise_sha = (
        starc_runtime.load_authenticated_exact81_tensor(
            args.native_noise, args.expected_native_noise_sha256,
            tensor_key=TENSOR_KEY_NOISE,
            label="current candidate official Gaussian",
        )
    )
    if (
        tuple(clean.shape) != candidate.geometry.latent_shape
        or tuple(noise.shape) != candidate.geometry.latent_shape
        or clean_sha != candidate.clean_latent_tensor_sha256
        or clean_sha == noise_sha
    ):
        raise SelfImaginedRelationalRuntimeError("candidate clean/noise closure differs")
    base_receipt_file = starc_runtime._authenticated_file(
        args.base_receipt,
        args.expected_base_receipt_sha256,
        label="current native RV2V base provenance receipt",
    )
    base_receipt = _strict_json(
        base_receipt_file,
        expected_sha256=args.expected_base_receipt_sha256,
        label="current native RV2V base provenance receipt",
    )
    base_provenance = validate_current_base_provenance_receipt(
        base_receipt,
        source_path=source,
        source_sha256=args.expected_source_video_sha256,
        action_caption_sha256=action_sha,
        clean_path=clean_path,
        clean_file_sha256=clean_file_sha,
        noise_path=noise_path,
        noise_file_sha256=noise_file_sha,
        base_mp4_path=base_mp4,
        base_mp4_sha256=args.expected_base_mp4_sha256,
        latent_shape=candidate.geometry.latent_shape,
    )
    teacher, teacher_receipt, teacher_binding = load_positive_teacher(
        receipt_path=args.teacher_receipt,
        expected_receipt_sha256=args.expected_teacher_receipt_sha256,
        artifact_path=args.teacher_residual,
        expected_artifact_sha256=args.expected_teacher_residual_sha256,
        expected_tensor_sha256=args.expected_teacher_residual_tensor_sha256,
        expected_episode_id=args.teacher_episode_id,
        action_caption_sha256=action_sha,
        noop_caption_sha256=noop_sha,
    )
    expected_spatial_sketch = live_bridge.geometry_spatial_sketch_binding(
        candidate.geometry
    )
    teacher_hidden = teacher_receipt.get("hidden_binding")
    if (
        teacher_receipt.get("spatial_sketch_binding") != expected_spatial_sketch
        or not isinstance(teacher_hidden, Mapping)
        or teacher_hidden.get("latent_shape")
        != list(candidate.geometry.latent_shape)
        or teacher_hidden.get("patch_positions")
        != candidate.geometry.patch_positions
        or teacher_hidden.get("patch_grid_height_width")
        != [candidate.geometry.patch_rows, candidate.geometry.patch_columns]
        or teacher_hidden.get("patch_flatten_order") != "patch-y-x"
    ):
        raise SelfImaginedRelationalRuntimeError(
            "teacher/current latent geometry or fixed spatial sketch differs"
        )
    starc_runtime._plain_directory(args.bernini_root, label="official Bernini root")
    starc_runtime._plain_directory(args.veomni_root, label="official VeOmni root")
    starc_runtime._plain_directory(args.checkpoint, label="official checkpoint")
    starc_runtime._authenticated_file(
        args.checkpoint_content_manifest,
        args.expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest",
    )
    return {
        "loader": loader, "source_archive": source_archive, "source": source,
        "base_mp4": base_mp4, "action": action, "action_path": action_path,
        "action_sha": action_sha, "noop": noop, "noop_path": noop_path,
        "noop_sha": noop_sha, "candidate": candidate, "clean": clean,
        "clean_path": clean_path, "clean_file_sha": clean_file_sha,
        "noise": noise, "noise_path": noise_path,
        "noise_file_sha": noise_file_sha, "noise_tensor_sha": noise_sha,
        "teacher": teacher, "teacher_receipt": teacher_receipt,
        "teacher_binding": {
            **teacher_binding,
            "current_geometry_spatial_sketch_binding": expected_spatial_sketch,
            "teacher_current_geometry_and_sketch_exact_match": True,
        },
        "base_receipt_file": base_receipt_file,
        "base_receipt_file_sha256": args.expected_base_receipt_sha256,
        "base_provenance": base_provenance,
        "output": output,
        "published_output": published_output,
    }


def run_one_sp4(args: argparse.Namespace) -> int:
    pre = _validate_run_cli(args)
    frozen = _frozen_d541801_runtime_facade()
    temporal_scorer.validate_native_coordinate_runtime(frozen)
    legacy = frozen.native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root, args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise SelfImaginedRelationalRuntimeError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise SelfImaginedRelationalRuntimeError("official model config differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)
    try:
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer
        from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
        from bernini.parallel import init_parallel_state
    except ImportError as error:  # pragma: no cover - AUH dependency
        raise SelfImaginedRelationalRuntimeError("official WORLD4 runtime unavailable") from error
    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != EXPECTED_WORLD_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise SelfImaginedRelationalRuntimeError("requires AUH ROCm WORLD4")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(minutes=180),
        rank=distributed.rank, world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    try:
        checkpoint_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                identity = live_bridge.authenticate_frozen_bernini_checkpoint_content(
                    checkpoint, args.checkpoint_content_manifest,
                    expected_checkpoint_tree_sha256=args.expected_checkpoint_tree_sha256,
                    expected_checkpoint_content_manifest_sha256=(
                        args.expected_checkpoint_content_manifest_sha256
                    ),
                )
                checkpoint_rows[0] = {"ok": True, "receipt": identity.receipt()}
            except BaseException as error:
                checkpoint_rows[0] = {
                    "ok": False, "error_type": type(error).__name__, "error": str(error)
                }
        dist.broadcast_object_list(checkpoint_rows, src=0)
        if not isinstance(checkpoint_rows[0], Mapping) or checkpoint_rows[0].get("ok") is not True:
            raise SelfImaginedRelationalRuntimeError(
                f"checkpoint authentication failed: {checkpoint_rows[0]}"
            )
        checkpoint_binding = checkpoint_rows[0]["receipt"]
        teacher_model = pre["teacher_receipt"]["model_binding"]
        teacher_checkpoint = teacher_model["checkpoint_content_binding"]
        if (
            teacher_checkpoint.get("manifest_sha256")
            != checkpoint_binding["checkpoint_content_manifest_file_sha256"]
            or teacher_checkpoint.get("verified_file_count")
            != checkpoint_binding["checkpoint_content_verified_file_count"]
            or teacher_checkpoint.get("verified_entries_digest")
            != checkpoint_binding["checkpoint_content_verified_entries_digest"]
        ):
            raise SelfImaginedRelationalRuntimeError("teacher/current checkpoint differs")
        config = BerniniRendererConfig.from_pretrained(
            str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
            local_files_only=True,
            **legacy.inference_renderer_config_overrides(checkpoint),
        )
        config.dtype = torch.bfloat16
        legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
        renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
        diffusion = renderer.diff_dec
        transformer = diffusion.transformer
        if (
            transformer is None or diffusion.transformer_2 is not None
            or any(p.requires_grad for p in renderer.parameters())
        ):
            raise SelfImaginedRelationalRuntimeError("frozen transformer_1 differs")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        conditions, condition_hashes, prompt_text = materializer._encode_prompt_pair(
            renderer, tokenizer, action_caption=pre["action"],
            noop_caption=pre["noop"], device=device, frozen=frozen,
        )
        builder_digest = frozen.prompt_builder_contract()["contract_digest"]
        live_prompt = temporal_scorer._prompt_binding(
            target_action_caption_sha256=pre["action_sha"],
            target_noop_caption_sha256=pre["noop_sha"],
            action_prompt=prompt_text["action_prompt"],
            noop_prompt=prompt_text["noop_prompt"],
            condition_hashes=condition_hashes,
            prompt_builder_contract_digest=builder_digest,
        )
        teacher_prompt = pre["teacher_receipt"]["prompt_binding"]
        for key in (
            "action_raw_caption_utf8_sha256", "noop_raw_caption_utf8_sha256",
            "action_full_prompt_utf8_sha256", "noop_full_prompt_utf8_sha256",
            "action_condition_tensor_sha256", "noop_condition_tensor_sha256",
            "prompt_builder_contract_digest", "prompt_pair_digest",
        ):
            if live_prompt[key] != teacher_prompt[key]:
                raise SelfImaginedRelationalRuntimeError(
                    f"teacher/current prompt binding differs: {key}"
                )
        scorer = relational.FrozenRelationalMotionScorer(
            pre["teacher"].to(device=device, dtype=torch.float32)
        ).requires_grad_(False).eval()
        if scorer.training or any(p.requires_grad for p in scorer.parameters()):
            raise SelfImaginedRelationalRuntimeError("relational scorer not frozen")
        clean = pre["clean"].to(device=device).detach().requires_grad_(True)
        noise = pre["noise"].to(device=device).detach()
        bridge = live_bridge.STARCLiveVJPBridgeV1(
            diffusion=diffusion, transformer=transformer, critic=scorer,
            candidate=pre["candidate"], instruction=pre["action"],
            action_condition=conditions["target_action"],
            noop_condition=conditions["noop"], sp_rank=distributed.rank,
            critic_artifact=None,
        )
        proof = bridge.prove_current_clean_latent_vjp(
            clean, noise, minimum_norm=float(args.minimum_norm)
        )
        if proof.critic_artifact is not None:
            raise SelfImaginedRelationalRuntimeError("learned critic artifact entered SAIL")
        raw_q = proof.gradient.detach().float().contiguous()
        pair = build_fixed_dose_interventions(pre["clean"].to(device), raw_q)
        digests = {
            "raw": live_bridge._tensor_value_digest(raw_q, label="raw q"),
            "projected": live_bridge._tensor_value_digest(
                pair.projected_cotangent, label="projected q"
            ),
            "delta": live_bridge._tensor_value_digest(
                pair.delta, label="fixed dose delta"
            ),
            "plus": live_bridge._tensor_value_digest(pair.plus, label="plus"),
            "minus": live_bridge._tensor_value_digest(pair.minus, label="minus"),
        }
        gathered: list[Any] = [None] * EXPECTED_WORLD_SIZE
        dist.all_gather_object(gathered, digests)
        if len({_object_sha256(row) for row in gathered}) != 1:
            raise SelfImaginedRelationalRuntimeError("WORLD4 intervention values differ")
        score_components = scorer.last_score_components
        if (
            not isinstance(score_components, Mapping)
            or score_components.get("intervention_allowed") is not True
            or not math.isclose(
                float(score_components.get("score", float("nan"))),
                float(proof.critic_score),
                rel_tol=1.0e-6,
                abs_tol=1.0e-7,
            )
        ):
            raise SelfImaginedRelationalRuntimeError(
                "live signed-relational objective diagnostics differ"
            )
        scorer_contract = {
            "class": "FrozenRelationalMotionScorer",
            "learned_head_or_checkpoint_consumed": False,
            "teacher_stop_gradient": True,
            "signed_temporal_quotient": True,
            "squared_gram_loss": True,
            "teacher_scaled_squared_signed_and_magnitude_losses": True,
            "teacher_gamma_stabilized_squared_gram_loss": True,
            "minimum_meaningful_mismatch": relational.MIN_MEANINGFUL_MISMATCH,
            "objective_mismatch": float(score_components["objective_mismatch"]),
            "signed_feature_loss": float(score_components["signed_feature_loss"]),
            "gram_loss": float(score_components["relational_gram_loss"]),
            "objective_fields": dict(score_components),
        }
        projection = pair.projection_diagnostics
        proof_binding = {
            "critic_score": proof.critic_score,
            "gradient_norm": proof.gradient_norm,
            "minimum_norm": proof.minimum_norm,
            "raw_cotangent_tensor_sha256": digests["raw"],
            "projected_cotangent_tensor_sha256": digests["projected"],
            "fixed_dose_delta_tensor_sha256": digests["delta"],
            "plus_tensor_sha256": digests["plus"],
            "minus_tensor_sha256": digests["minus"],
            "hook_coordinate": live_bridge.HOOK_COORDINATE,
            "hook_call_order": list(proof.hook_call_order),
            "same_x_sigma_object_for_action_noop": True,
            "real_sp4_autograd_collective_observed": proof.real_sp4_autograd_collective,
            "replica_consensus_observed": proof.replica_consensus_observed,
            "all_rank_hidden_backward_digest": proof.all_rank_hidden_backward_digest,
            "production_runtime_dimensions": proof.production_runtime_dimensions,
        }
        # The live bridge is T2V-style at an RV2V-produced endpoint: no source prefix.
        receipt = {
            "schema_version": SCHEMA_VERSION, "method": METHOD,
            "candidate_id": pre["candidate"].candidate_id,
            "source_condition_in_live_query": False,
            "current_endpoint_origin": "authenticated_native_rv2v",
            "current_base_provenance": {
                **dict(pre["base_provenance"]),
                "receipt_path": str(pre["base_receipt_file"]),
                "receipt_file_sha256": pre["base_receipt_file_sha256"],
            },
            "source_preservation_mechanism": (
                "endpoint_base_plus_fixed_projection_only"
            ),
            "current_query_is_rv2v_forward": False,
            "source_native_composition_claimed": False,
            "source_video": {
                "path": str(pre["source"]),
                "file_sha256": args.expected_source_video_sha256,
            },
            "current_clean_latent": {
                "path": str(pre["clean_path"]),
                "file_sha256": pre["clean_file_sha"],
                "tensor_sha256": pre["candidate"].clean_latent_tensor_sha256,
            },
            "current_official_gaussian": {
                "path": str(pre["noise_path"]),
                "file_sha256": pre["noise_file_sha"],
                "tensor_sha256": pre["noise_tensor_sha"],
            },
            "teacher": dict(pre["teacher_binding"]),
            "prompt_binding": live_prompt,
            "checkpoint_binding": checkpoint_binding,
            "runtime_binding": {
                "loader_path": str(pre["loader"]),
                "loader_file_sha256": args.expected_loader_source_sha256,
                "source_archive_path": str(pre["source_archive"]),
                "source_archive_file_sha256": args.expected_source_archive_sha256,
                "source_git_revision": args.source_git_revision,
                "bernini_revision": bernini_revision,
                "veomni_revision": veomni_revision,
                "world_size": EXPECTED_WORLD_SIZE,
            },
            "scorer_contract": scorer_contract,
            "live_vjp_proof": proof_binding,
            "intervention": {
                "dose_rms": FIXED_DOSE_RMS,
                "dose_rms_hex": float(FIXED_DOSE_RMS).hex(),
                "dose_cli_or_search_exposed": False,
                "projector_survival_floor_fixed_inside_core": True,
                "minimum_projection_survival_ratio": (
                    relational.MIN_PROJECTION_SURVIVAL_RATIO
                ),
                "raw_rms": projection.raw_rms,
                "projected_rms": projection.projected_rms,
                "survival_ratio": projection.survival_ratio,
                "ascent_cosine": projection.ascent_cosine,
                "phase0_max_abs": projection.phase0_max_abs,
                "temporal_sum_max_abs": projection.temporal_sum_max_abs,
                "spatial_affine_max_abs_dot": (
                    projection.spatial_affine_max_abs_dot
                ),
                "phase0_preserved_by_projection": True,
                "temporal_dc_removed": True,
                "spatial_affine_basis_removed": True,
            },
            "training_performed": False,
            "editor_parameter_or_update_authorized": False,
            "identity_or_camera_preservation_proven": False,
            "scientific_claim_authorized": False,
            "action_editing_success_claim_authorized": False,
            "mechanism_probe_only": True,
        }
        del bridge, scorer, transformer, diffusion, renderer, tokenizer, conditions
        torch.cuda.empty_cache()
        write_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                _save_rank0_outputs(
                    final_dir=pre["output"], published_dir=pre["published_output"],
                    base=pre["clean"].to(device),
                    pair=pair, raw_q=raw_q, source_video=pre["source"],
                    base_mp4=pre["base_mp4"],
                    checkpoint=checkpoint, device=device, receipt=receipt,
                )
                write_rows[0] = {"ok": True}
            except BaseException as error:
                write_rows[0] = {
                    "ok": False, "error_type": type(error).__name__, "error": str(error)
                }
        dist.broadcast_object_list(write_rows, src=0)
        if not isinstance(write_rows[0], Mapping) or write_rows[0].get("ok") is not True:
            raise SelfImaginedRelationalRuntimeError(
                f"rank-zero output publication failed: {write_rows[0]}"
            )
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--expected-candidate-manifest-sha256", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-video-sha256", required=True)
    parser.add_argument("--action-caption-file", required=True)
    parser.add_argument("--expected-action-caption-file-sha256", required=True)
    parser.add_argument("--noop-caption-file", required=True)
    parser.add_argument("--expected-noop-caption-file-sha256", required=True)
    parser.add_argument("--current-clean-latent", required=True)
    parser.add_argument("--expected-current-clean-latent-sha256", required=True)
    parser.add_argument("--native-noise", required=True)
    parser.add_argument("--expected-native-noise-sha256", required=True)
    parser.add_argument("--base-mp4", required=True)
    parser.add_argument("--expected-base-mp4-sha256", required=True)
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--expected-base-receipt-sha256", required=True)
    parser.add_argument("--teacher-receipt", required=True)
    parser.add_argument("--expected-teacher-receipt-sha256", required=True)
    parser.add_argument("--teacher-residual", required=True)
    parser.add_argument("--expected-teacher-residual-sha256", required=True)
    parser.add_argument("--expected-teacher-residual-tensor-sha256", required=True)
    parser.add_argument("--teacher-episode-id", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-bernini-commit", required=True)
    parser.add_argument("--expected-veomni-commit", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--expected-loader-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--published-output-dir", required=True,
        help="final path after the enclosing all8 directory-level rename",
    )
    parser.add_argument("--minimum-norm", type=float, default=1.0e-12)
    parser.add_argument("--ack-mechanism-probe-only", action="store_true")
    parser.add_argument("--ack-no-editor-parameter-or-update", action="store_true")
    parser.add_argument(
        "--ack-no-scientific-or-action-editing-claim", action="store_true"
    )
    parser.add_argument(
        "--ack-live-query-has-no-source-condition", action="store_true"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_one_sp4(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_TEACHER_SHAPE", "FIXED_DOSE_RMS", "METHOD", "SCHEMA_VERSION",
    "STATIC_SOURCE_CLOSURE", "SelfImaginedRelationalRuntimeError",
    "build_fixed_dose_interventions", "build_parser", "load_positive_teacher",
    "load_positive_teacher_binding", "main", "run_one_sp4",
    "strict_positive_teacher_binding", "validate_positive_teacher_binding",
    "validate_current_base_provenance_receipt",
]

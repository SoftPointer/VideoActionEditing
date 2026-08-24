#!/usr/bin/env python3
"""Score a rendered PAIR-v5 T2V bank with frozen Bernini global MACE.

One invocation handles one Ulysses-SP4 group.  Every rendered candidate is
loaded from its exact native pre-decode FP32 latent and mixed with its cell's
tensor-value-identical official sampler Gaussian.  Each safetensors container
is independently hash-verified; container header bytes need not match.  At the preregistered exact40 pilot
coordinate (index 33, the first mid-sigma cell), a frozen Bernini target-only
scorer queries the complete action-plus-nine prompt registry on one shared
``x_sigma`` object.

The primary scalar is ``score.energy.reward`` (global MACE).  The
phase-conjunctive minimum is emitted under an explicitly diagnostic field and
never enters v3 calibration.  Score receipts contain scalar/text/hash
provenance but no MP4, media path, source, RV2V target, donor, mask, flow,
pose, track, or trajectory.  This executable performs no training and makes
no action-editing-success claim.
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
import struct
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_pair_v5_t2v_calibration_bank as bank_runner  # noqa: E402
import infer_native_identity_generation_canary as native_generation  # noqa: E402
import mace_candidate_action_energy as mace  # noqa: E402
import pair_v5_action_adapter as action_adapter  # noqa: E402
import pair_v5_native_bridge as native_bridge  # noqa: E402
import pair_v5_phase_conjunctive_energy as phase_energy  # noqa: E402
import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402
import source_self_native_ref_contrastive_v3 as native_schedule  # noqa: E402


SCORE_RECEIPT_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-score-v3"
GROUP_RECEIPT_SCHEMA = "bernini-pair-v5-frozen-t2v-global-energy-group-v3"
PILOT_SCHEDULE_INDEX = 33
PILOT_GATE_NAME = "mid"
PILOT_SIGMA = 0.5161304473876953
PILOT_NATIVE_SCHEDULER_TIMESTEP = 516
# Kept as a public compatibility name.  This is the *model timestep* paired
# with ``PILOT_SIGMA`` by the pinned native UniPC schedule, not 1000*sigma.
PILOT_SCORER_PHYSICAL_TIMESTEP = float(PILOT_NATIVE_SCHEDULER_TIMESTEP)
MACE_CROSS_DEVICE_REPLAY_RTOL = 1.0e-5
MACE_CROSS_DEVICE_REPLAY_ATOL = 1.0e-6

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")

SCORE_INPUT_CLOSURE = {
    "accepted_tensor_inputs": [
        "candidate_own_native_predecode_clean_latent",
        "same_cell_official_native_sampler_gaussian",
    ],
    "accepted_semantic_inputs": ["closed_action_plus_nine_t2v_prompt_bank"],
    "generated_mp4_consumed_by_scorer": False,
    "generated_mp4_bound_by_sha256_as_provenance": True,
    "source_video_or_source_latent": False,
    "geometry_source_video_hash_bound_as_bucket_provenance_only": True,
    "complete_caption_hashes_bound_as_semantic_provenance": True,
    "rv2v_video_reference_target_or_pseudo_target": False,
    "proposal_as_donor_input_or_noise": False,
    "mask_flow_pose_track_trajectory": False,
    "event_audit_label_consumed_by_model": False,
    "training_performed": False,
    "optimizer_step_performed": False,
}

_SCORE_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "analysis_split",
        "action_family_id",
        "calibration_group_id",
        "actor_group_id",
        "scene_group_id",
        "action_group_id",
        "semantic_branch",
        "candidate_envelope_sha256",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "generation_receipt_digest",
        "generation_receipt_file_sha256",
        "native_rollout_receipt_digest",
        "native_rollout_receipt_file_sha256",
        "generated_mp4_sha256",
        "clean_latent_artifact_sha256",
        "geometry_source_video_sha256",
        "full_t2v_caption_utf8_sha256",
        "checkpoint_content_identity",
        "frozen_checkpoint_receipt_digest",
        "checkpoint_content_binding",
        "frozen_scorer_receipt_digest",
        "frozen_t2v_packet_binding",
        "generation_runtime_binding_by_branch",
        "generation_runtime_registry_digest",
        "full_t2v_caption_by_branch",
        "prompt_by_branch",
        "prompt_builder_contract",
        "scorer_runtime_versions",
        "prompt_registry_digest",
        "prompt_utf8_sha256_by_branch",
        "full_t2v_caption_utf8_sha256_by_branch",
        "clean_latent_tensor_sha256",
        "official_gaussian_tensor_sha256",
        "official_gaussian_artifact_sha256",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "sigma_tensor_sha256",
        "schedule_coordinate",
        "phase_weight_commitment",
        "phase_weight_registration_digest",
        "energy_epsilon",
        "raw_global_action_energy_score",
        "raw_phase_conjunctive_score_diagnostic",
        "global_action_energy",
        "global_hard_negative_energy_by_branch",
        "global_negative_log_energy_ratio_by_branch",
        "global_hardest_negative_branch",
        "mace_live_tensor_formula_proof",
        "phase_diagnostic_receipt_digest",
        "phase_diagnostic_used_for_calibration",
        "input_closure",
        "scientific_action_editing_claim",
        "receipt_digest",
    }
)

_MACE_LIVE_TENSOR_FORMULA_PROOF_FIELDS = frozenset(
    {
        "branch_order",
        "hard_negative_order",
        "branch_energy_tensor_sha256",
        "negative_log_energy_ratio_tensor_sha256",
        "reward_tensor_sha256",
        "hardest_negative_index_tensor_sha256",
        "tensor_dtype",
        "formula_recomputed_on_origin_device_bit_exact",
        "reward_and_first_argmin_recomputed_on_origin_device_bit_exact",
        "digest",
    }
)

_CHECKPOINT_IDENTITY_FIELDS = frozenset(
    {
        "manifest_path",
        "manifest_sha256_computed",
        "manifest_sha256_expected",
        "verified_file_count",
        "every_file_sha256_verified",
        "verified_entries_digest",
    }
)
_FREEZE_CERTIFICATE_FIELDS = frozenset(
    {
        "base_frozen",
        "trainable_parameter_tensors",
        "trainable_parameter_elements",
        "lora_module_count",
    }
)
_CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "manifest_sha256",
        "verified_file_count",
        "verified_entries_digest",
        "every_file_sha256_verified",
        "loaded_components",
        "all_loaded_parameters_frozen",
        "freeze_certificate",
        "binding_digest",
    }
)
_FROZEN_T2V_PACKET_BINDING_FIELDS = frozenset(
    {
        "packet_receipt_digest",
        "prompt_registry_digest",
        "frozen_model_receipt_digest",
        "candidate_shape",
        "sigma_float32_bits_hex",
        "timestep_float32_bits_hex",
        "native_schedule_digest",
        "native_schedule_index",
        "native_scheduler_timestep",
        "timestep_mapping",
        "physical_sigma_and_model_timestep_share_native_exact40_index",
        "legacy_1000_sigma_timestep_rejected",
        "binding_digest",
    }
)
_GENERATION_CHECKPOINT_BINDING_FIELDS = frozenset(
    {
        "manifest_sha256",
        "verified_file_count",
        "verified_entries_digest",
        "every_file_sha256_verified",
    }
)
_GENERATION_PROMPT_CONTRACT_FIELDS = frozenset(
    {
        "training_task_name",
        "inference_arm",
        "guidance_mode",
        "system_prompt_sha256",
        "binding_clause_sha256",
        "full_prompt_sha256",
        "cleaner",
        "tokenizer_fix_mistral_regex",
    }
)
_GENERATION_RUNTIME_BINDING_FIELDS = frozenset(
    {
        "candidate_id",
        "semantic_branch",
        "generation_receipt_digest",
        "native_rollout_receipt_digest",
        "generation_method_source_revision",
        "generation_method_source_archive_sha256",
        "bernini_revision",
        "veomni_revision",
        "bernini_inference_files",
        "checkpoint_tree_sha256",
        "checkpoint_content_binding",
        "action_prompt_utf8_sha256",
        "full_prompt_utf8_sha256",
        "prompt_contract",
        "runtime_versions",
        "binding_digest",
    }
)
_RUNTIME_VERSION_FIELDS = frozenset(
    {"torch", "torch_hip", "transformers", "diffusers"}
)


class PairV5T2VEnergyScoringError(RuntimeError):
    """The bank, frozen runtime, or scalar receipt differs from its seal."""


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
        raise PairV5T2VEnergyScoringError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
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
        raise PairV5T2VEnergyScoringError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    """Hash exact tensor bytes without depending on a trainer module."""

    import torch

    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise PairV5T2VEnergyScoringError("tensor hash requires a real tensor")
    cpu = value.detach().to(device="cpu").contiguous().clone()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "layout": str(cpu.layout),
    }
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(metadata))
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def native_tensor_value_identity(value: Any) -> dict[str, Any]:
    """Recompute Bernini's native raw/content identity from tensor storage."""

    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.numel() <= 0
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PairV5T2VEnergyScoringError(
            "native tensor identity requires one finite real tensor"
        )
    cpu = value.detach().to(device="cpu").contiguous().clone()
    raw = cpu.view(torch.uint8).reshape(-1).numpy().tobytes()
    metadata = {
        "shape": [int(item) for item in cpu.shape],
        "dtype": str(cpu.dtype),
        "numel": int(cpu.numel()),
        "byte_count": len(raw),
    }
    return {
        **metadata,
        "raw_value_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": hashlib.sha256(
            canonical_json_bytes(metadata) + b"\x00" + raw
        ).hexdigest(),
    }


def verify_native_tensor_value_identity(
    value: Any, artifact: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    """Bind a loaded tensor back to the native artifact's declared identity."""

    identity = native_tensor_value_identity(value)
    expected = {
        "shape": artifact.get("shape"),
        "dtype": artifact.get("stored_dtype"),
        "raw_value_sha256": artifact.get("raw_value_sha256"),
        "content_sha256": artifact.get("content_sha256"),
    }
    observed = {
        "shape": identity["shape"],
        "dtype": identity["dtype"],
        "raw_value_sha256": identity["raw_value_sha256"],
        "content_sha256": identity["content_sha256"],
    }
    if observed != expected:
        raise PairV5T2VEnergyScoringError(
            f"{label} actual tensor value differs from native receipt"
        )
    return identity


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairV5T2VEnergyScoringError(f"{label} must be lowercase SHA-256")
    return value


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PairV5T2VEnergyScoringError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise PairV5T2VEnergyScoringError(f"{label} must be an absolute plain directory")
    return path.resolve(strict=True)


def _strict_json_file(
    value: str | Path, *, expected_sha256: Optional[str], label: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, label=f"{label} SHA-256"
    ):
        raise PairV5T2VEnergyScoringError(f"{label} SHA-256 differs")

    def reject_constant(token: str) -> None:
        raise PairV5T2VEnergyScoringError(f"{label} contains {token}")

    def reject_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PairV5T2VEnergyScoringError(f"{label} duplicate key {key!r}")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairV5T2VEnergyScoringError(f"{label} is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise PairV5T2VEnergyScoringError(f"{label} root must be an object")
    return decoded, path, digest


def _verify_embedded_receipt(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    digest = _sha256(unsigned.pop("receipt_digest", None), label=f"{label} digest")
    if object_sha256(unsigned) != digest:
        # Bank-generation receipts use UTF-8 canonicalization.  The payload is
        # ASCII in the core4 pilot, but retain the producer's exact helper.
        if bank_contract.sha256_bytes(bank_contract.canonical_json_bytes(unsigned)) != digest:
            raise PairV5T2VEnergyScoringError(f"{label} embedded digest differs")
    return digest


def _validated_checkpoint_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or set(identity) != set(
        _CHECKPOINT_IDENTITY_FIELDS
    ):
        raise PairV5T2VEnergyScoringError(
            "checkpoint content identity field closure differs"
        )
    manifest_path = identity["manifest_path"]
    if (
        type(manifest_path) is not str
        or not manifest_path.startswith("/")
        or "\x00" in manifest_path
    ):
        raise PairV5T2VEnergyScoringError(
            "checkpoint content manifest path differs"
        )
    manifest_computed = _sha256(
        identity["manifest_sha256_computed"],
        label="computed checkpoint manifest SHA-256",
    )
    manifest_expected = _sha256(
        identity["manifest_sha256_expected"],
        label="expected checkpoint manifest SHA-256",
    )
    entries_digest = _sha256(
        identity["verified_entries_digest"],
        label="checkpoint verified entries digest",
    )
    count = identity["verified_file_count"]
    if (
        manifest_computed != manifest_expected
        or manifest_computed
        != native_generation.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
        or type(count) is not int
        or count != native_generation.source_audit.CHECKPOINT_CONTENT_FILE_COUNT
        or identity["every_file_sha256_verified"] is not True
    ):
        raise PairV5T2VEnergyScoringError(
            "checkpoint content identity did not close"
        )
    return {
        "manifest_path": manifest_path,
        "manifest_sha256_computed": manifest_computed,
        "manifest_sha256_expected": manifest_expected,
        "verified_file_count": count,
        "every_file_sha256_verified": True,
        "verified_entries_digest": entries_digest,
    }


def _validated_freeze_certificate(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _FREEZE_CERTIFICATE_FIELDS
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen model certificate field closure differs"
        )
    if (
        value["base_frozen"] is not True
        or type(value["trainable_parameter_tensors"]) is not int
        or value["trainable_parameter_tensors"] != 0
        or type(value["trainable_parameter_elements"]) is not int
        or value["trainable_parameter_elements"] != 0
        or type(value["lora_module_count"]) is not int
        or value["lora_module_count"] != 0
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen model certificate did not close"
        )
    return {
        "base_frozen": True,
        "trainable_parameter_tensors": 0,
        "trainable_parameter_elements": 0,
        "lora_module_count": 0,
    }


def checkpoint_content_binding(
    identity: Mapping[str, Any], freeze_certificate: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce the full checkpoint audit to a path-free, replayable binding."""

    checked_identity = _validated_checkpoint_identity(identity)
    checked_freeze_certificate = _validated_freeze_certificate(freeze_certificate)
    unsigned = {
        "manifest_sha256": checked_identity["manifest_sha256_computed"],
        "verified_file_count": checked_identity["verified_file_count"],
        "verified_entries_digest": checked_identity["verified_entries_digest"],
        "every_file_sha256_verified": True,
        "loaded_components": ["transformer_1", "umt5_text_encoder"],
        "all_loaded_parameters_frozen": True,
        "freeze_certificate": checked_freeze_certificate,
    }
    return {**unsigned, "binding_digest": object_sha256(unsigned)}


def _generation_checkpoint_binding(identity: Any) -> dict[str, Any]:
    """Return the path-free checkpoint identity carried by native generation."""

    checked = _validated_checkpoint_identity(identity)
    return {
        "manifest_sha256": checked["manifest_sha256_computed"],
        "verified_file_count": checked["verified_file_count"],
        "verified_entries_digest": checked["verified_entries_digest"],
        "every_file_sha256_verified": True,
    }


def _validated_runtime_versions(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_RUNTIME_VERSION_FIELDS):
        raise PairV5T2VEnergyScoringError(f"{label} field closure differs")
    result: dict[str, str] = {}
    for name in sorted(_RUNTIME_VERSION_FIELDS):
        version = value[name]
        if (
            not isinstance(version, str)
            or not version
            or version != version.strip()
            or "\x00" in version
        ):
            raise PairV5T2VEnergyScoringError(f"{label} {name} differs")
        result[name] = str(version)
    return {name: result[name] for name in ("torch", "torch_hip", "transformers", "diffusers")}


def current_runtime_versions() -> dict[str, str]:
    """Bind the packages that implement text cleaning, encoding, and scoring."""

    import torch
    import diffusers
    import transformers

    return _validated_runtime_versions(
        {
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "transformers": transformers.__version__,
            "diffusers": diffusers.__version__,
        },
        label="scorer runtime versions",
    )


def _validated_generation_prompt_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _GENERATION_PROMPT_CONTRACT_FIELDS
    ):
        raise PairV5T2VEnergyScoringError(
            "generation T2V prompt contract field closure differs"
        )
    expected = {
        "training_task_name": native_generation.ARM_TRAINING_TASK_NAMES["t2v"],
        "inference_arm": "t2v",
        "guidance_mode": native_generation.ARM_GUIDANCE_MODES["t2v"],
        "system_prompt_sha256": hashlib.sha256(
            native_generation.TASK_SYSTEM_PROMPTS[
                native_generation.ARM_TRAINING_TASK_NAMES["t2v"]
            ].encode("utf-8")
        ).hexdigest(),
        "binding_clause_sha256": hashlib.sha256(
            native_generation.TASK_BINDING_CLAUSES["t2v"].encode("utf-8")
        ).hexdigest(),
        "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
        "tokenizer_fix_mistral_regex": True,
    }
    if any(value.get(name) != wanted for name, wanted in expected.items()):
        raise PairV5T2VEnergyScoringError(
            "generation T2V prompt contract differs from the pinned builder"
        )
    full_prompt_sha = _sha256(
        value["full_prompt_sha256"], label="generation full T2V prompt SHA-256"
    )
    return {**expected, "full_prompt_sha256": full_prompt_sha}


def generation_runtime_binding_from_native_receipt(
    native_receipt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    generation_receipt_digest: str,
    native_rollout_receipt_digest: str,
) -> dict[str, Any]:
    """Bind the exact generation source, checkpoint, and executed full prompt."""

    if not isinstance(native_receipt, Mapping) or not isinstance(candidate, Mapping):
        raise PairV5T2VEnergyScoringError("native generation binding inputs differ")
    revision = native_receipt.get("method_source_revision")
    if type(revision) is not str or _SHA1_RE.fullmatch(revision) is None:
        raise PairV5T2VEnergyScoringError(
            "generation method source revision differs"
        )
    archive_sha = _sha256(
        native_receipt.get("method_source_archive_sha256"),
        label="generation method source archive SHA-256",
    )
    bernini_revision = native_receipt.get("bernini_commit")
    veomni_revision = native_receipt.get("veomni_commit")
    if (
        bernini_revision != native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or veomni_revision != native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT
    ):
        raise PairV5T2VEnergyScoringError(
            "generation Bernini/VeOmni source revision differs"
        )
    inference_files = native_receipt.get("bernini_inference_files")
    if inference_files != native_generation.legacy.BERNINI_INFERENCE_FILE_HASHES:
        raise PairV5T2VEnergyScoringError(
            "generation Bernini inference source hashes differ"
        )
    checkpoint = native_receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "path",
        "tree_sha256",
        "content",
    }:
        raise PairV5T2VEnergyScoringError(
            "generation checkpoint receipt field closure differs"
        )
    checkpoint_path = checkpoint["path"]
    if (
        type(checkpoint_path) is not str
        or not checkpoint_path.startswith("/")
        or "\x00" in checkpoint_path
        or checkpoint["tree_sha256"]
        != native_generation.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise PairV5T2VEnergyScoringError("generation checkpoint identity differs")
    checkpoint_binding = _generation_checkpoint_binding(checkpoint["content"])
    prompt_registry = native_receipt.get("prompt_contract")
    if not isinstance(prompt_registry, Mapping) or set(prompt_registry) != {"t2v"}:
        raise PairV5T2VEnergyScoringError(
            "generation prompt-contract arm closure differs"
        )
    prompt_contract = _validated_generation_prompt_contract(
        prompt_registry["t2v"]
    )
    runtime_versions = _validated_runtime_versions(
        native_receipt.get("runtime_versions"),
        label="generation runtime versions",
    )
    input_value = native_receipt.get("input")
    if not isinstance(input_value, Mapping):
        raise PairV5T2VEnergyScoringError("generation input receipt differs")
    action_prompt_sha = _sha256(
        input_value.get("action_prompt_utf8_sha256"),
        label="generation raw action prompt SHA-256",
    )
    if action_prompt_sha != candidate.get("full_t2v_caption_utf8_sha256"):
        raise PairV5T2VEnergyScoringError(
            "generation raw action prompt differs from the root-spec caption"
        )
    candidate_id = candidate.get("candidate_id")
    branch = candidate.get("semantic_branch")
    if (
        type(candidate_id) is not str
        or _SAFE_ID_RE.fullmatch(candidate_id) is None
        or branch not in mace.BRANCH_ORDER
    ):
        raise PairV5T2VEnergyScoringError(
            "generation candidate/branch identity differs"
        )
    unsigned = {
        "candidate_id": candidate_id,
        "semantic_branch": branch,
        "generation_receipt_digest": _sha256(
            generation_receipt_digest, label="generation receipt digest"
        ),
        "native_rollout_receipt_digest": _sha256(
            native_rollout_receipt_digest,
            label="native rollout receipt digest",
        ),
        "generation_method_source_revision": revision,
        "generation_method_source_archive_sha256": archive_sha,
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "bernini_inference_files": dict(inference_files),
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
        "checkpoint_content_binding": checkpoint_binding,
        "action_prompt_utf8_sha256": action_prompt_sha,
        "full_prompt_utf8_sha256": prompt_contract["full_prompt_sha256"],
        "prompt_contract": prompt_contract,
        "runtime_versions": runtime_versions,
    }
    return {**unsigned, "binding_digest": object_sha256(unsigned)}


def _validated_generation_runtime_binding(
    value: Any, *, expected_branch: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        _GENERATION_RUNTIME_BINDING_FIELDS
    ):
        raise PairV5T2VEnergyScoringError(
            "generation runtime binding field closure differs"
        )
    row = dict(value)
    unsigned = dict(row)
    digest = _sha256(
        unsigned.pop("binding_digest", None),
        label="generation runtime binding digest",
    )
    if object_sha256(unsigned) != digest:
        raise PairV5T2VEnergyScoringError(
            "generation runtime binding digest differs"
        )
    if (
        row["semantic_branch"] != expected_branch
        or type(row["candidate_id"]) is not str
        or _SAFE_ID_RE.fullmatch(row["candidate_id"]) is None
        or type(row["generation_method_source_revision"]) is not str
        or _SHA1_RE.fullmatch(row["generation_method_source_revision"]) is None
        or row["bernini_revision"]
        != native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT
        or row["veomni_revision"]
        != native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT
        or row["bernini_inference_files"]
        != native_generation.legacy.BERNINI_INFERENCE_FILE_HASHES
        or row["checkpoint_tree_sha256"]
        != native_generation.legacy.trainer.CHECKPOINT_TREE_SHA256
    ):
        raise PairV5T2VEnergyScoringError(
            "generation runtime/source identity differs"
        )
    for name in (
        "generation_receipt_digest",
        "native_rollout_receipt_digest",
        "generation_method_source_archive_sha256",
        "action_prompt_utf8_sha256",
        "full_prompt_utf8_sha256",
    ):
        _sha256(row[name], label=name)
    checkpoint = row["checkpoint_content_binding"]
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != set(_GENERATION_CHECKPOINT_BINDING_FIELDS)
        or checkpoint.get("manifest_sha256")
        != native_generation.source_audit.CHECKPOINT_CONTENT_MANIFEST_SHA256
        or checkpoint.get("verified_file_count")
        != native_generation.source_audit.CHECKPOINT_CONTENT_FILE_COUNT
        or checkpoint.get("every_file_sha256_verified") is not True
    ):
        raise PairV5T2VEnergyScoringError(
            "generation checkpoint content binding differs"
        )
    _sha256(
        checkpoint.get("verified_entries_digest"),
        label="generation checkpoint entries digest",
    )
    prompt_contract = _validated_generation_prompt_contract(
        row["prompt_contract"]
    )
    if prompt_contract["full_prompt_sha256"] != row["full_prompt_utf8_sha256"]:
        raise PairV5T2VEnergyScoringError(
            "generation prompt contract/full-prompt hash differs"
        )
    row["runtime_versions"] = _validated_runtime_versions(
        row["runtime_versions"], label="generation runtime versions"
    )
    row["binding_digest"] = digest
    return row


def validate_generation_runtime_registry(
    value: Any,
    *,
    caption_by_branch: Mapping[str, str],
    prompt_by_branch: Mapping[str, str],
    checkpoint_identity: Mapping[str, Any],
    scorer_runtime_versions: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Close every scored branch back to its own native generation receipt."""

    if not isinstance(value, Mapping) or set(value) != set(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError(
            "generation runtime registry branch closure differs"
        )
    checked_checkpoint = _generation_checkpoint_binding(checkpoint_identity)
    checked_versions = _validated_runtime_versions(
        scorer_runtime_versions, label="scorer runtime versions"
    )
    result: dict[str, dict[str, Any]] = {}
    candidate_ids: list[str] = []
    for branch in mace.BRANCH_ORDER:
        row = _validated_generation_runtime_binding(
            value[branch], expected_branch=branch
        )
        caption = caption_by_branch.get(branch)
        prompt = prompt_by_branch.get(branch)
        if (
            type(caption) is not str
            or type(prompt) is not str
            or row["action_prompt_utf8_sha256"]
            != hashlib.sha256(caption.encode("utf-8")).hexdigest()
            or row["full_prompt_utf8_sha256"]
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or row["checkpoint_content_binding"] != checked_checkpoint
            or row["runtime_versions"] != checked_versions
        ):
            raise PairV5T2VEnergyScoringError(
                f"generation/scorer prompt-checkpoint chain differs for {branch}"
            )
        candidate_ids.append(row["candidate_id"])
        result[branch] = row
    if len(set(candidate_ids)) != len(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError(
            "generation runtime registry candidate IDs repeat"
        )
    shared_contexts = {
        object_sha256(
            {
                "generation_method_source_revision": row[
                    "generation_method_source_revision"
                ],
                "generation_method_source_archive_sha256": row[
                    "generation_method_source_archive_sha256"
                ],
                "bernini_revision": row["bernini_revision"],
                "veomni_revision": row["veomni_revision"],
                "bernini_inference_files": row["bernini_inference_files"],
                "checkpoint_tree_sha256": row["checkpoint_tree_sha256"],
                "checkpoint_content_binding": row["checkpoint_content_binding"],
                "runtime_versions": row["runtime_versions"],
            }
        )
        for row in result.values()
    }
    if len(shared_contexts) != 1:
        raise PairV5T2VEnergyScoringError(
            "generation cell did not share one source/checkpoint/runtime context"
        )
    return result


def schedule_coordinate_receipt() -> dict[str, Any]:
    """Return the single preregistered native exact40 pilot coordinate."""

    if (
        native_schedule.NATIVE_UNIPC40_SIGMAS[PILOT_SCHEDULE_INDEX] != PILOT_SIGMA
        or native_schedule.NATIVE_UNIPC40_TIMESTEPS[PILOT_SCHEDULE_INDEX]
        != PILOT_NATIVE_SCHEDULER_TIMESTEP
        or action_adapter.sigma_gate(PILOT_SCHEDULE_INDEX)
        != (PILOT_GATE_NAME, 0.5)
    ):
        raise PairV5T2VEnergyScoringError("pinned exact40 coordinate drifted")
    sigma_fp32 = struct.pack("!f", PILOT_SIGMA).hex()
    scorer_timestep_fp32 = struct.pack(
        "!f", float(PILOT_NATIVE_SCHEDULER_TIMESTEP)
    ).hex()
    value = {
        "schedule_name": "pinned_bernini_unipc40_flow_shift5",
        "schedule_digest": native_schedule.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST,
        "schedule_index": PILOT_SCHEDULE_INDEX,
        "action_adapter_gate": "mid_weight_0.5",
        "action_adapter_gate_weight": 0.5,
        "physical_sigma": PILOT_SIGMA,
        "physical_sigma_float32_be_hex": sigma_fp32,
        "native_scheduler_timestep": PILOT_NATIVE_SCHEDULER_TIMESTEP,
        "frozen_t2v_scorer_timestep_mapping": (
            "direct_native_unipc40_discrete_timestep_same_schedule_index"
        ),
        "frozen_t2v_scorer_timestep": PILOT_SCORER_PHYSICAL_TIMESTEP,
        "frozen_t2v_scorer_timestep_float32_be_hex": scorer_timestep_fp32,
        "physical_sigma_and_model_timestep_share_native_exact40_index": True,
        "legacy_1000_sigma_timestep_rejected": True,
        "pilot_rationale": "first_mid_cell_balances_candidate_signal_and_action_prior",
    }
    return {**value, "coordinate_digest": object_sha256(value)}


def diagnostic_phase_commitment() -> dict[str, Any]:
    """A fixed five-window exact81 diagnostic; never a calibration gate."""

    windows = {
        "actor": range(0, 5),
        "direction": range(5, 9),
        "contact": range(9, 13),
        "order": range(13, 17),
        "terminal": range(17, 21),
    }
    weights: dict[str, list[float]] = {}
    for milestone in phase_energy.MILESTONE_ORDER:
        indices = list(windows[milestone])
        row = [0.0] * phase_energy.LATENT_PHASES
        weight = 1.0 / len(indices)
        for index in indices:
            row[index] = weight
        weights[milestone] = row
    return phase_energy.make_phase_weight_commitment(weights)


def _bank_candidate_index(bank_receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = bank_receipt.get("candidate_receipts")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise PairV5T2VEnergyScoringError("bank candidate receipt index differs")
    by_id = {row.get("candidate_id"): row for row in rows}
    if len(by_id) != len(rows) or any(not isinstance(key, str) for key in by_id):
        raise PairV5T2VEnergyScoringError("bank candidate receipt IDs repeat")
    return by_id


def load_group_bank(
    *,
    root_spec: str | Path,
    root_spec_sha256: str,
    bank_output_dir: str | Path,
    bank_receipt: str | Path,
    bank_receipt_sha256: str,
    group_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Authenticate root, bank, every candidate receipt, and all artifacts."""

    try:
        spec, spec_digest = bank_contract.load_sealed_spec(
            root_spec, root_spec_sha256
        )
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error
    if spec_digest != root_spec_sha256:
        raise PairV5T2VEnergyScoringError("root spec digest differs")
    output = _plain_directory(bank_output_dir, label="rendered bank output")
    bank, bank_path, observed_bank_sha = _strict_json_file(
        bank_receipt,
        expected_sha256=bank_receipt_sha256,
        label="rendered bank receipt",
    )
    if bank_path.parent != output:
        raise PairV5T2VEnergyScoringError("bank receipt is outside rendered bank root")
    bank_digest = _verify_embedded_receipt(bank, label="rendered bank receipt")
    if (
        bank.get("schema_version") != bank_contract.BANK_RECEIPT_SCHEMA_VERSION
        or bank.get("root_spec_raw_sha256") != spec_digest
        or bank.get("candidate_count")
        != sum(len(group["candidates"]) for group in spec["groups"])
        or bank.get("cell_count")
        != len(
            {
                candidate["calibration_group_id"]
                for group in spec["groups"]
                for candidate in group["candidates"]
            }
        )
        or bank.get("mace_branch_order") != list(bank_contract.MACE_BRANCH_ORDER)
        or bank.get("sampling_contract") != bank_contract.SAMPLING_CONTRACT
        or bank.get("semantic_input_closure") != bank_contract.SEMANTIC_INPUT_CLOSURE
        or bank.get("artifact_use_contract") != bank_contract.ARTIFACT_USE_CONTRACT
        or bank.get("split_contract") != bank_contract.SPLIT_CONTRACT
        or bank.get("fit_confirmation_all_registered_axes_disjoint") is not True
    ):
        raise PairV5T2VEnergyScoringError("rendered bank contract differs")
    expected_bank_interpretation = {
        "calibration_evidence_only": True,
        "event_qualification_performed": False,
        "action_success_not_implied": True,
        "training_performed": False,
        "parameter_update_performed": False,
        "optimizer_authorized": False,
        "t2v_negative_media_are_rv2v_policy_candidates": False,
        "t2v_media_as_condition_target_donor_or_noise_forbidden": True,
    }
    if bank.get("interpretation") != expected_bank_interpretation:
        raise PairV5T2VEnergyScoringError("rendered bank exceeds calibration authority")
    expected_membership = {
        split: {
            axis: sorted(
                {
                    candidate[axis]
                    for item in spec["groups"]
                    for candidate in item["candidates"]
                    if candidate["analysis_split"] == split
                }
            )
            for axis in bank_contract.SPLIT_GROUP_AXES
        }
        for split in bank_contract.ANALYSIS_SPLITS
    }
    if bank.get("split_group_membership") != expected_membership:
        raise PairV5T2VEnergyScoringError("rendered bank split membership differs")
    indexed = _bank_candidate_index(bank)
    group = next((item for item in spec["groups"] if item["group_id"] == group_id), None)
    if group is None or group_id not in {"sp4-a", "sp4-b"}:
        raise PairV5T2VEnergyScoringError("group ID is absent from the sealed bank")

    bound: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(group["candidates"]):
        receipt_path = output / candidate["candidate_id"] / "pair-v5-t2v-calibration-receipt.json"
        try:
            receipt = bank_runner._load_pair_receipt(receipt_path)
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise PairV5T2VEnergyScoringError(str(error)) from error
        if (
            receipt["root_spec_raw_sha256"] != spec_digest
            or receipt["candidate"] != candidate
            or receipt["group_id"] != group_id
            or receipt["visible_gpus"] != group["visible_gpus"]
            or receipt["runtime_topology"]
            != {
                "world_size": 4,
                "ulysses_size": 4,
                "rocr_visible_devices": ",".join(
                    str(item) for item in group["visible_gpus"]
                ),
            }
            or receipt["ordinal"] != ordinal
            or receipt["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
            or receipt["semantic_input_closure"] != bank_contract.SEMANTIC_INPUT_CLOSURE
            or receipt["artifact_use_contract"] != bank_contract.ARTIFACT_USE_CONTRACT
            or receipt["split_contract"] != bank_contract.SPLIT_CONTRACT
        ):
            raise PairV5T2VEnergyScoringError("candidate generation receipt binding differs")
        expected_interpretation = {
            "calibration_evidence_only": True,
            "event_qualified_from_generation_receipt": False,
            "action_success_not_implied": True,
            "training_performed": False,
            "parameter_update_performed": False,
            "optimizer_authorized": False,
            "t2v_media_as_rv2v_policy_candidate_forbidden": True,
            "donor_or_pseudo_target_use_forbidden": True,
        }
        if receipt["interpretation"] != expected_interpretation:
            raise PairV5T2VEnergyScoringError(
                "candidate generation receipt exceeds calibration-only authority"
            )
        native_path = _plain_file(
            receipt["native_receipt_path"], label="bound native generation receipt"
        )
        if file_sha256(native_path) != receipt["native_receipt_sha256"]:
            raise PairV5T2VEnergyScoringError("native generation receipt SHA-256 differs")
        try:
            native_receipt = bank_runner._load_json(
                native_path, "bound native generation receipt"
            )
            native_artifacts = bank_runner._verify_native_receipt(
                native_receipt, candidate
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise PairV5T2VEnergyScoringError(str(error)) from error
        generation_runtime_binding = generation_runtime_binding_from_native_receipt(
            native_receipt,
            candidate,
            generation_receipt_digest=receipt["receipt_digest"],
            native_rollout_receipt_digest=receipt["native_receipt_digest"],
        )
        expected_artifacts = {
            "mp4": native_artifacts["mp4"],
            "predecode_clean_latent": native_artifacts[
                "predecode_clean_latent"
            ],
            "official_initial_gaussian": native_artifacts[
                "official_initial_gaussian"
            ],
        }
        geometry_certificate = receipt["geometry_use_certificate"]
        if (
            native_artifacts["native_receipt_digest"]
            != receipt["native_receipt_digest"]
            or receipt["artifacts"] != expected_artifacts
            or not isinstance(geometry_certificate, Mapping)
            or geometry_certificate.get("bucket_hw")
            != native_artifacts["bucket_hw"]
            or geometry_certificate.get("latent_shape")
            != native_artifacts["latent_shape"]
            or geometry_certificate.get("video_sha256")
            != candidate["geometry_source_video_sha256"]
            or geometry_certificate.get("used_to_derive_bucket_shape") is not True
            or geometry_certificate.get("vae_latent_created") is not False
            or geometry_certificate.get("pixels_entered_transformer") is not False
            or geometry_certificate.get("content_conditioning_count") != 0
        ):
            raise PairV5T2VEnergyScoringError(
                "native generation evidence/geometry binding differs"
            )
        row = indexed.get(candidate["candidate_id"])
        if not isinstance(row, Mapping):
            raise PairV5T2VEnergyScoringError("candidate is absent from bank index")
        receipt_sha = file_sha256(receipt_path)
        indexed_receipt_path = _plain_file(
            row.get("receipt_path"),
            label=f"{candidate['candidate_id']} indexed generation receipt",
        )
        if (
            indexed_receipt_path != receipt_path
            or row.get("receipt_sha256") != receipt_sha
            or row.get("receipt_digest") != receipt["receipt_digest"]
            or row.get("predecode_clean_latent_sha256")
            != receipt["artifacts"]["predecode_clean_latent"]["sha256"]
            or row.get("official_initial_gaussian_sha256")
            != receipt["artifacts"]["official_initial_gaussian"]["sha256"]
        ):
            raise PairV5T2VEnergyScoringError("candidate bank-index identity differs")
        native_path = _plain_file(
            receipt["native_receipt_path"],
            label=f"{candidate['candidate_id']} native rollout receipt",
        )
        native_file_sha = file_sha256(native_path)
        if native_file_sha != receipt["native_receipt_sha256"]:
            raise PairV5T2VEnergyScoringError("native rollout receipt file hash differs")
        try:
            native_receipt = bank_runner._load_json(
                native_path, "native T2V rollout receipt"
            )
            native_artifacts = bank_runner._verify_native_receipt(
                native_receipt, candidate
            )
        except bank_contract.PairT2VCalibrationSpecError as error:
            raise PairV5T2VEnergyScoringError(str(error)) from error
        if native_artifacts["native_receipt_digest"] != receipt["native_receipt_digest"]:
            raise PairV5T2VEnergyScoringError("native rollout receipt digest differs")

        artifacts: dict[str, dict[str, Any]] = {}
        for name in (
            "mp4",
            "predecode_clean_latent",
            "official_initial_gaussian",
        ):
            try:
                artifacts[name] = bank_runner._verify_file_artifact(
                    receipt["artifacts"][name], f"{candidate['candidate_id']} {name}"
                )
            except bank_contract.PairT2VCalibrationSpecError as error:
                raise PairV5T2VEnergyScoringError(str(error)) from error
            if artifacts[name] != native_artifacts[name]:
                raise PairV5T2VEnergyScoringError(
                    f"{candidate['candidate_id']} {name} differs from native rollout"
                )
        bound.append(
            {
                "candidate": candidate,
                "candidate_envelope_sha256": receipt["candidate_envelope_sha256"],
                "generation_receipt_digest": receipt["receipt_digest"],
                "generation_receipt_file_sha256": receipt_sha,
                "native_rollout_receipt_digest": receipt["native_receipt_digest"],
                "native_rollout_receipt_file_sha256": native_file_sha,
                "generation_runtime_binding": generation_runtime_binding,
                "artifacts": artifacts,
            }
        )

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in bound:
        by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
    bank_cell_proofs = bank.get("same_cell_gaussian_proofs")
    if not isinstance(bank_cell_proofs, list) or any(
        not isinstance(item, Mapping) for item in bank_cell_proofs
    ):
        raise PairV5T2VEnergyScoringError("bank Gaussian proof registry differs")
    for cell_id, rows in by_cell.items():
        if [row["candidate"]["semantic_branch"] for row in rows] != list(mace.BRANCH_ORDER):
            raise PairV5T2VEnergyScoringError(f"cell {cell_id} lost exact branch order")
        tensor_identity_fields = (
            "raw_value_sha256",
            "content_sha256",
            "shape",
            "dtype",
            "stored_dtype",
            "generator_initial_seed",
        )
        gaussian_tensor_identities = {
            object_sha256(
                {
                    field: row["artifacts"]["official_initial_gaussian"].get(field)
                    for field in tensor_identity_fields
                }
            )
            for row in rows
        }
        if len(gaussian_tensor_identities) != 1:
            raise PairV5T2VEnergyScoringError(
                f"cell {cell_id} official Gaussian tensor value differs"
            )
        proof = next(
            (
                item
                for item in bank_cell_proofs
                if item.get("calibration_group_id") == cell_id
            ),
            None,
        )
        first_gaussian = rows[0]["artifacts"]["official_initial_gaussian"]
        expected_containers = {
            row["candidate"]["semantic_branch"]: row["artifacts"][
                "official_initial_gaussian"
            ]["sha256"]
            for row in rows
        }
        if (
            not isinstance(proof, Mapping)
            or proof.get("analysis_split")
            != rows[0]["candidate"]["analysis_split"]
            or proof.get("action_family_id")
            != rows[0]["candidate"]["action_family_id"]
            or proof.get("semantic_branch_count") != len(rows)
            or proof.get("semantic_branch_order") != list(mace.BRANCH_ORDER)
            or proof.get("all_ten_official_gaussian_tensor_values_byte_equal")
            is not True
            or proof.get("all_container_files_individually_sha256_verified")
            is not True
            or proof.get("official_gaussian_file_sha256_by_branch")
            != expected_containers
            or proof.get("official_gaussian_raw_value_sha256")
            != first_gaussian.get("raw_value_sha256")
            or proof.get("official_gaussian_content_sha256")
            != first_gaussian.get("content_sha256")
            or proof.get("seed") != first_gaussian.get("generator_initial_seed")
        ):
            raise PairV5T2VEnergyScoringError(
                f"cell {cell_id} bank Gaussian proof differs"
            )
    return spec, {**bank, "receipt_digest": bank_digest, "file_sha256": observed_bank_sha}, bound


def prompt_bank_from_cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    task_prompt_builder: Callable[[str], str],
) -> dict[str, str]:
    if [row["candidate"]["semantic_branch"] for row in rows] != list(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError("prompt cell branch order differs")
    for row in rows:
        candidate = row["candidate"]
        caption = candidate.get("full_t2v_caption")
        if (
            type(caption) is not str
            or not caption
            or caption != caption.strip()
            or "\x00" in caption
            or hashlib.sha256(caption.encode("utf-8")).hexdigest()
            != candidate.get("full_t2v_caption_utf8_sha256")
        ):
            raise PairV5T2VEnergyScoringError(
                "sealed full T2V caption text/hash binding differs"
            )
    result = {
        row["candidate"]["semantic_branch"]: task_prompt_builder(
            row["candidate"]["full_t2v_caption"]
        )
        for row in rows
    }
    try:
        return mace.validate_prompt_closure(result)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error


def _official_prompt_cleaner() -> Callable[[str], str]:
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    except Exception as error:
        raise PairV5T2VEnergyScoringError(
            "official Wan prompt cleaner is unavailable"
        ) from error
    return prompt_clean


def prompt_builder_contract() -> dict[str, Any]:
    """Bind the exact Bernini/Wan function used to turn captions into prompts."""

    task_name = native_generation.ARM_TRAINING_TASK_NAMES["t2v"]
    system = native_generation.TASK_SYSTEM_PROMPTS[task_name]
    clause = native_generation.TASK_BINDING_CLAUSES["t2v"]
    cleaner = _official_prompt_cleaner()
    try:
        builder_source = inspect.getsource(native_generation.build_task_prompt)
        cleaner_source = inspect.getsource(cleaner)
    except (OSError, TypeError) as error:
        raise PairV5T2VEnergyScoringError(
            "official prompt builder source is unavailable"
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
    return {**value, "contract_digest": object_sha256(value)}


def official_prompt_bank_from_captions(
    caption_by_branch: Mapping[str, str],
    *,
    prompt_cleaner: Optional[Callable[[str], str]] = None,
) -> dict[str, str]:
    """Rebuild the prompt registry from sealed raw captions, never hashes alone."""

    if not isinstance(caption_by_branch, Mapping) or set(
        caption_by_branch
    ) != set(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError(
            "full T2V caption registry closure differs"
        )
    if prompt_cleaner is None:
        prompt_cleaner = _official_prompt_cleaner()
    result: dict[str, str] = {}
    for branch in mace.BRANCH_ORDER:
        caption = caption_by_branch[branch]
        if (
            type(caption) is not str
            or not caption
            or caption != caption.strip()
            or "\x00" in caption
        ):
            raise PairV5T2VEnergyScoringError(
                f"sealed full T2V caption for {branch} differs"
            )
        try:
            result[branch] = native_generation.build_task_prompt(
                "t2v", caption, prompt_cleaner=prompt_cleaner
            )
        except Exception as error:
            raise PairV5T2VEnergyScoringError(
                f"official T2V prompt builder rejected {branch} caption"
            ) from error
    try:
        return mace.validate_prompt_closure(result)
    except mace.MACECandidateActionEnergyError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error


class NativeExact40FrozenBerniniT2VScorer(
    native_bridge.FrozenBerniniT2VScorer
):
    """Use physical sigma 0.516130... but native schedule model time 516.

    The shared PAIR bridge historically mapped model time to ``1000*sigma``.
    Native Bernini UniPC instead calls the transformer at the discrete scheduler
    timestep paired with that sigma.  This scorer preserves MACE's physical
    noising coordinate and replaces only the transformer timestep before the
    first model call.  Its packet receipt records the exact replacement.
    """

    def _start_packet(self, x_sigma: Any, sigma: Any) -> None:
        import torch

        super()._start_packet(x_sigma, sigma)
        if (
            sigma.dtype != torch.float32
            or tuple(sigma.shape) != (1,)
            or struct.pack("!f", float(sigma.item())).hex()
            != struct.pack("!f", PILOT_SIGMA).hex()
            or self._packet_timestep is None
        ):
            self.abort_packet()
            raise PairV5T2VEnergyScoringError(
                "native exact40 scorer received a non-pilot physical sigma"
            )
        legacy_timestep = float(self._packet_timestep.item())
        if struct.pack("!f", legacy_timestep).hex() != struct.pack(
            "!f", 1000.0 * PILOT_SIGMA
        ).hex():
            self.abort_packet()
            raise PairV5T2VEnergyScoringError(
                "legacy scorer timestep mapping changed before native override"
            )
        self._packet_timestep = torch.tensor(
            [float(PILOT_NATIVE_SCHEDULER_TIMESTEP)],
            dtype=torch.float32,
            device=sigma.device,
        )

    def forward(self, x_sigma: Any, sigma: Any, prompt: str) -> Any:
        final_branch = self._packet_position == len(mace.BRANCH_ORDER) - 1
        result = super().forward(x_sigma, sigma, prompt)
        if final_branch:
            packet = self._last_packet_receipt
            if not isinstance(packet, Mapping):
                raise PairV5T2VEnergyScoringError(
                    "native exact40 scorer packet receipt is unavailable"
                )
            unsigned = dict(packet)
            unsigned.pop("digest", None)
            if (
                unsigned.get("timestep_float32_bits_hex")
                != struct.pack("!f", float(PILOT_NATIVE_SCHEDULER_TIMESTEP)).hex()
            ):
                raise PairV5T2VEnergyScoringError(
                    "native exact40 model timestep was not executed"
                )
            unsigned.update(
                {
                    "timestep_mapping": (
                        "direct_native_unipc40_discrete_timestep_same_schedule_index"
                    ),
                    "native_schedule_digest": (
                        native_schedule.PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST
                    ),
                    "native_schedule_index": PILOT_SCHEDULE_INDEX,
                    "native_scheduler_timestep": PILOT_NATIVE_SCHEDULER_TIMESTEP,
                    "physical_sigma_and_model_timestep_share_native_exact40_index": True,
                    "legacy_1000_sigma_timestep_rejected": True,
                }
            )
            self._last_packet_receipt = {
                **unsigned,
                "digest": native_bridge.object_sha256(unsigned),
            }
        return result


def frozen_t2v_packet_binding(
    packet: Any, scorer_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce the executed scorer packet to its native-coordinate proof."""

    if not isinstance(packet, Mapping) or not isinstance(scorer_receipt, Mapping):
        raise PairV5T2VEnergyScoringError("frozen T2V packet receipt differs")
    unsigned_packet = dict(packet)
    packet_digest = _sha256(
        unsigned_packet.pop("digest", None), label="frozen T2V packet digest"
    )
    if native_bridge.object_sha256(unsigned_packet) != packet_digest:
        raise PairV5T2VEnergyScoringError("frozen T2V packet digest differs")
    if scorer_receipt.get("packet_receipt_digest") != packet_digest:
        raise PairV5T2VEnergyScoringError(
            "frozen scorer/packet receipt binding differs"
        )
    expected_coordinate = schedule_coordinate_receipt()
    expected_shape = [1, 16, 21]
    candidate_shape = packet.get("candidate_shape")
    if (
        not isinstance(candidate_shape, list)
        or candidate_shape[:3] != expected_shape
        or len(candidate_shape) != 5
        or any(type(item) is not int or item <= 0 for item in candidate_shape)
        or packet.get("sigma_float32_bits_hex")
        != expected_coordinate["physical_sigma_float32_be_hex"]
        or packet.get("timestep_float32_bits_hex")
        != expected_coordinate["frozen_t2v_scorer_timestep_float32_be_hex"]
        or packet.get("timestep_mapping")
        != expected_coordinate["frozen_t2v_scorer_timestep_mapping"]
        or packet.get("native_schedule_digest")
        != expected_coordinate["schedule_digest"]
        or packet.get("native_schedule_index") != PILOT_SCHEDULE_INDEX
        or packet.get("native_scheduler_timestep")
        != PILOT_NATIVE_SCHEDULER_TIMESTEP
        or packet.get(
            "physical_sigma_and_model_timestep_share_native_exact40_index"
        )
        is not True
        or packet.get("legacy_1000_sigma_timestep_rejected") is not True
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen T2V packet native exact40 coordinate differs"
        )
    value = {
        "packet_receipt_digest": packet_digest,
        "prompt_registry_digest": _sha256(
            packet.get("prompt_registry_digest"),
            label="packet prompt registry digest",
        ),
        "frozen_model_receipt_digest": _sha256(
            packet.get("frozen_model_receipt_digest"),
            label="packet frozen model receipt digest",
        ),
        "candidate_shape": list(candidate_shape),
        "sigma_float32_bits_hex": packet["sigma_float32_bits_hex"],
        "timestep_float32_bits_hex": packet["timestep_float32_bits_hex"],
        "native_schedule_digest": packet["native_schedule_digest"],
        "native_schedule_index": packet["native_schedule_index"],
        "native_scheduler_timestep": packet["native_scheduler_timestep"],
        "timestep_mapping": packet["timestep_mapping"],
        "physical_sigma_and_model_timestep_share_native_exact40_index": True,
        "legacy_1000_sigma_timestep_rejected": True,
    }
    return {**value, "binding_digest": object_sha256(value)}


def _load_exact81_tensor(artifact: Mapping[str, Any], *, key: str, label: str) -> Any:
    import torch
    from safetensors import safe_open

    path = _plain_file(artifact.get("path"), label=label)
    if file_sha256(path) != artifact.get("sha256"):
        raise PairV5T2VEnergyScoringError(f"{label} file hash differs")
    with safe_open(str(path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != [key]:
            raise PairV5T2VEnergyScoringError(f"{label} tensor key closure differs")
        tensor = opened.get_tensor(key).float().contiguous()
    if (
        tensor.dtype != torch.float32
        or tensor.ndim != 5
        or tuple(int(item) for item in tensor.shape[:3]) != (1, 16, 21)
        or int(tensor.shape[3]) <= 0
        or int(tensor.shape[4]) <= 0
        or int(tensor.shape[3]) % 2
        or int(tensor.shape[4]) % 2
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise PairV5T2VEnergyScoringError(f"{label} is not detached FP32 exact81")
    return tensor


def _encode_prompt_bank(
    renderer: Any,
    tokenizer: Any,
    prompt_by_branch: Mapping[str, str],
    *,
    device: Any,
) -> dict[str, Any]:
    import torch

    legacy = native_generation.legacy
    result: dict[str, Any] = {}
    for branch in mace.BRANCH_ORDER:
        ids, mask = legacy._tokenize_training_prompt(tokenizer, prompt_by_branch[branch])
        with torch.inference_mode():
            condition = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        if (
            tuple(int(item) for item in condition.shape) != (1, 512, 4096)
            or condition.device != device
            or condition.requires_grad
            or not bool(torch.isfinite(condition).all().item())
        ):
            raise PairV5T2VEnergyScoringError(f"prompt condition {branch} differs")
        result[branch] = condition
    if len({tensor_sha256(value.float()) for value in result.values()}) != len(result):
        raise PairV5T2VEnergyScoringError("two prompt conditions alias exactly")
    return result


def _make_live_mace_tensor_formula_proof(energy: Any) -> tuple[dict[str, Any], dict[str, float]]:
    """Bind MACE's formula on its origin device before scalar serialization.

    ROCm and CPU ``log`` kernels need not return identical low bits.  The
    formula, reward, and first argmin are therefore recomputed bit-exactly on
    the device that produced MACE, while tensor hashes make the serialized
    scalar receipt replayable on any validating device.
    """

    import torch

    branch = getattr(energy, "branch_energies", None)
    ratios = getattr(energy, "negative_log_energy_ratios", None)
    reward = getattr(energy, "reward", None)
    hardest = getattr(energy, "hardest_negative_index", None)
    expected_shapes = (
        (branch, (len(mace.BRANCH_ORDER), 1), "branch energies", torch.float32),
        (
            ratios,
            (len(mace.HARD_NEGATIVE_BRANCHES), 1),
            "negative log-energy ratios",
            torch.float32,
        ),
        (reward, (1,), "MACE reward", torch.float32),
        (hardest, (1,), "hardest-negative index", torch.int64),
    )
    for tensor, shape, label, dtype in expected_shapes:
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(int(item) for item in tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device.type == "meta"
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise PairV5T2VEnergyScoringError(
                f"live {label} tensor closure differs"
            )
    if any(
        tensor.device != branch.device for tensor in (ratios, reward, hardest)
    ):
        raise PairV5T2VEnergyScoringError("live MACE tensors do not share one device")
    if bool((branch < 0.0).any().item()):
        raise PairV5T2VEnergyScoringError("live MACE branch energy is negative")

    epsilon = float(mace.DEFAULT_ENERGY_EPSILON)
    with torch.no_grad():
        recomputed_ratios = torch.log(branch[1:] + epsilon) - torch.log(
            branch[:1] + epsilon
        )
        recomputed_reward, recomputed_hardest = ratios.min(dim=0)
    if not torch.equal(recomputed_ratios, ratios):
        raise PairV5T2VEnergyScoringError(
            "live MACE formula is not bit-exact on its origin device"
        )
    if not torch.equal(recomputed_reward, reward) or not torch.equal(
        recomputed_hardest, hardest
    ):
        raise PairV5T2VEnergyScoringError(
            "live MACE reward/first argmin are not bit-exact"
        )

    unsigned = {
        "branch_order": list(mace.BRANCH_ORDER),
        "hard_negative_order": list(mace.HARD_NEGATIVE_BRANCHES),
        "branch_energy_tensor_sha256": tensor_sha256(branch),
        "negative_log_energy_ratio_tensor_sha256": tensor_sha256(ratios),
        "reward_tensor_sha256": tensor_sha256(reward),
        "hardest_negative_index_tensor_sha256": tensor_sha256(hardest),
        "tensor_dtype": "torch.float32",
        "formula_recomputed_on_origin_device_bit_exact": True,
        "reward_and_first_argmin_recomputed_on_origin_device_bit_exact": True,
    }
    proof = {**unsigned, "digest": object_sha256(unsigned)}
    ratio_by_branch = {
        name: float(ratios[index, 0].item())
        for index, name in enumerate(mace.HARD_NEGATIVE_BRANCHES)
    }
    return proof, ratio_by_branch


def _validated_energy_scalars(
    *,
    action_energy: Any,
    negative_energy_by_branch: Any,
    negative_log_ratio_by_branch: Any,
    raw_reward: Any,
    hardest_negative_branch: Any,
    energy_epsilon: Any,
    live_tensor_formula_proof: Any,
) -> tuple[float, dict[str, float], float, str]:
    """Replay the origin-device MACE binding without cross-device bit claims."""

    import torch

    def exact_fp32(value: Any, *, label: str, nonnegative: bool = False) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise PairV5T2VEnergyScoringError(f"{label} differs")
        if float(torch.tensor(value, dtype=torch.float32).item()) != value:
            raise PairV5T2VEnergyScoringError(f"{label} is not an exact FP32 scalar")
        if nonnegative and value < 0.0:
            raise PairV5T2VEnergyScoringError(f"{label} is negative")
        return value

    action_energy = exact_fp32(
        action_energy, label="global action energy", nonnegative=True
    )
    if not isinstance(negative_energy_by_branch, Mapping) or set(
        negative_energy_by_branch
    ) != set(mace.HARD_NEGATIVE_BRANCHES):
        raise PairV5T2VEnergyScoringError(
            "global hard-negative energy closure differs"
        )
    negatives: dict[str, float] = {}
    for branch in mace.HARD_NEGATIVE_BRANCHES:
        value = negative_energy_by_branch[branch]
        negatives[branch] = exact_fp32(
            value,
            label=f"global hard-negative energy for {branch}",
            nonnegative=True,
        )
    if not isinstance(negative_log_ratio_by_branch, Mapping) or set(
        negative_log_ratio_by_branch
    ) != set(mace.HARD_NEGATIVE_BRANCHES):
        raise PairV5T2VEnergyScoringError(
            "global negative log-energy ratio closure differs"
        )
    ratios_by_branch = {
        branch: exact_fp32(
            negative_log_ratio_by_branch[branch],
            label=f"global negative log-energy ratio for {branch}",
        )
        for branch in mace.HARD_NEGATIVE_BRANCHES
    }
    if (
        type(energy_epsilon) is not float
        or not math.isfinite(energy_epsilon)
        or energy_epsilon != float(mace.DEFAULT_ENERGY_EPSILON)
    ):
        raise PairV5T2VEnergyScoringError("MACE energy epsilon differs")
    raw_reward = exact_fp32(raw_reward, label="raw global action reward")
    if type(hardest_negative_branch) is not str:
        raise PairV5T2VEnergyScoringError("global hardest-negative branch differs")
    energy_tensor = torch.tensor(
        [action_energy, *(negatives[name] for name in mace.HARD_NEGATIVE_BRANCHES)],
        dtype=torch.float32,
    ).reshape(-1, 1)
    ratio_tensor = torch.tensor(
        [ratios_by_branch[name] for name in mace.HARD_NEGATIVE_BRANCHES],
        dtype=torch.float32,
    ).reshape(-1, 1)
    reward_tensor = torch.tensor([raw_reward], dtype=torch.float32)
    expected_reward_tensor, expected_index_tensor = ratio_tensor[:, 0].min(dim=0)
    expected_reward = float(expected_reward_tensor.item())
    expected_index = int(expected_index_tensor.item())
    expected_branch = mace.HARD_NEGATIVE_BRANCHES[expected_index]
    if raw_reward != expected_reward:
        raise PairV5T2VEnergyScoringError(
            "raw global action reward does not equal the direct MACE formula tensor"
        )
    if hardest_negative_branch != expected_branch:
        raise PairV5T2VEnergyScoringError(
            "global hardest-negative branch is not the first FP32 argmin"
        )

    if not isinstance(live_tensor_formula_proof, Mapping) or set(
        live_tensor_formula_proof
    ) != set(_MACE_LIVE_TENSOR_FORMULA_PROOF_FIELDS):
        raise PairV5T2VEnergyScoringError("live MACE formula proof closure differs")
    proof = dict(live_tensor_formula_proof)
    unsigned_proof = dict(proof)
    declared_proof_digest = _sha256(
        unsigned_proof.pop("digest", None), label="live MACE formula proof digest"
    )
    if (
        object_sha256(unsigned_proof) != declared_proof_digest
        or proof["branch_order"] != list(mace.BRANCH_ORDER)
        or proof["hard_negative_order"] != list(mace.HARD_NEGATIVE_BRANCHES)
        or proof["tensor_dtype"] != "torch.float32"
        or proof["formula_recomputed_on_origin_device_bit_exact"] is not True
        or proof["reward_and_first_argmin_recomputed_on_origin_device_bit_exact"]
        is not True
        or proof["branch_energy_tensor_sha256"] != tensor_sha256(energy_tensor)
        or proof["negative_log_energy_ratio_tensor_sha256"]
        != tensor_sha256(ratio_tensor)
        or proof["reward_tensor_sha256"] != tensor_sha256(reward_tensor)
        or proof["hardest_negative_index_tensor_sha256"]
        != tensor_sha256(
            torch.tensor([expected_index], dtype=torch.int64)
        )
    ):
        raise PairV5T2VEnergyScoringError("live MACE formula proof binding differs")

    # This replay is intentionally auxiliary: it catches gross corruption of
    # energies versus ratios, but the critical equality above is the exact
    # origin-device proof. CPU and ROCm log kernels may differ in low bits.
    cpu_formula = torch.log(energy_tensor[1:, 0] + energy_epsilon) - torch.log(
        energy_tensor[0] + energy_epsilon
    )
    if not torch.allclose(
        ratio_tensor[:, 0],
        cpu_formula,
        rtol=MACE_CROSS_DEVICE_REPLAY_RTOL,
        atol=MACE_CROSS_DEVICE_REPLAY_ATOL,
    ):
        raise PairV5T2VEnergyScoringError(
            "serialized MACE formula tensor differs from auxiliary CPU replay"
        )
    return action_energy, negatives, raw_reward, hardest_negative_branch


def make_score_receipt(
    *,
    row: Mapping[str, Any],
    root_spec_raw_sha256: str,
    bank_receipt_digest: str,
    checkpoint_identity: Mapping[str, Any],
    freeze_certificate: Mapping[str, Any],
    generation_runtime_binding_by_branch: Mapping[str, Mapping[str, Any]],
    scorer_runtime_versions: Mapping[str, str],
    prompt_by_branch: Mapping[str, str],
    caption_by_branch: Mapping[str, str],
    caption_sha256_by_branch: Mapping[str, str],
    phase_weight_commitment: Mapping[str, Any],
    clean: Any,
    epsilon: Any,
    sigma: Any,
    score: native_bridge.FrozenT2VActionScore,
    scorer_packet_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = row["candidate"]
    live_mace_formula_proof, negative_log_ratios = (
        _make_live_mace_tensor_formula_proof(score.energy)
    )
    reward = float(score.energy.reward.item())
    phase_reward = float(score.phase_energy.reward.item())
    if not math.isfinite(reward) or not math.isfinite(phase_reward):
        raise PairV5T2VEnergyScoringError("frozen action score is non-finite")
    branch_energies = score.energy.branch_energies[:, 0].tolist()
    if len(branch_energies) != len(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError("global branch-energy closure differs")
    hardest = int(score.energy.hardest_negative_index.item())
    if not 0 <= hardest < len(mace.HARD_NEGATIVE_BRANCHES):
        raise PairV5T2VEnergyScoringError("global hardest-negative index differs")
    try:
        commitment = phase_energy.validate_phase_weight_commitment(
            phase_weight_commitment
        )
    except phase_energy.PairV5PhaseEnergyError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error
    if (
        commitment["registration_digest"]
        != score.receipt["phase_weight_registration_digest"]
    ):
        raise PairV5T2VEnergyScoringError(
            "phase commitment differs from frozen scorer receipt"
        )
    if not isinstance(caption_sha256_by_branch, Mapping) or set(
        caption_sha256_by_branch
    ) != set(mace.BRANCH_ORDER):
        raise PairV5T2VEnergyScoringError("caption hash registry order differs")
    checked_captions = {
        branch: caption_by_branch[branch]
        for branch in mace.BRANCH_ORDER
    } if isinstance(caption_by_branch, Mapping) and set(
        caption_by_branch
    ) == set(mace.BRANCH_ORDER) else None
    if checked_captions is None:
        raise PairV5T2VEnergyScoringError("caption text registry order differs")
    checked_caption_hashes = {
        branch: _sha256(
            caption_sha256_by_branch[branch],
            label=f"{branch} full T2V caption SHA-256",
        )
        for branch in mace.BRANCH_ORDER
    }
    for branch in mace.BRANCH_ORDER:
        caption = checked_captions[branch]
        if (
            type(caption) is not str
            or hashlib.sha256(caption.encode("utf-8")).hexdigest()
            != checked_caption_hashes[branch]
        ):
            raise PairV5T2VEnergyScoringError(
                f"sealed caption text/hash binding differs for {branch}"
            )
    rebuilt_prompts = official_prompt_bank_from_captions(checked_captions)
    if dict(prompt_by_branch) != rebuilt_prompts:
        raise PairV5T2VEnergyScoringError(
            "prompt registry was not rebuilt from sealed captions"
        )
    prompt_hashes = {
        branch: hashlib.sha256(prompt_by_branch[branch].encode("utf-8")).hexdigest()
        for branch in mace.BRANCH_ORDER
    }
    checkpoint_binding = checkpoint_content_binding(
        checkpoint_identity, freeze_certificate
    )
    checked_checkpoint_identity = _validated_checkpoint_identity(
        checkpoint_identity
    )
    checked_scorer_runtime_versions = _validated_runtime_versions(
        scorer_runtime_versions, label="scorer runtime versions"
    )
    checked_generation_registry = validate_generation_runtime_registry(
        generation_runtime_binding_by_branch,
        caption_by_branch=checked_captions,
        prompt_by_branch=rebuilt_prompts,
        checkpoint_identity=checked_checkpoint_identity,
        scorer_runtime_versions=checked_scorer_runtime_versions,
    )
    candidate_generation_binding = checked_generation_registry[
        candidate["semantic_branch"]
    ]
    if (
        candidate_generation_binding["candidate_id"]
        != candidate["candidate_id"]
        or candidate_generation_binding["generation_receipt_digest"]
        != row["generation_receipt_digest"]
        or candidate_generation_binding["native_rollout_receipt_digest"]
        != row["native_rollout_receipt_digest"]
    ):
        raise PairV5T2VEnergyScoringError(
            "candidate generation runtime binding differs from its scored row"
        )
    checkpoint_receipt_digest = object_sha256(checked_checkpoint_identity)
    packet_binding = frozen_t2v_packet_binding(
        scorer_packet_receipt, score.receipt
    )
    action_energy = float(branch_energies[0])
    negative_energies = {
        branch: float(branch_energies[index + 1])
        for index, branch in enumerate(mace.HARD_NEGATIVE_BRANCHES)
    }
    hardest_branch = mace.HARD_NEGATIVE_BRANCHES[hardest]
    _validated_energy_scalars(
        action_energy=action_energy,
        negative_energy_by_branch=negative_energies,
        negative_log_ratio_by_branch=negative_log_ratios,
        raw_reward=reward,
        hardest_negative_branch=hardest_branch,
        energy_epsilon=float(mace.DEFAULT_ENERGY_EPSILON),
        live_tensor_formula_proof=live_mace_formula_proof,
    )
    value = {
        "schema_version": SCORE_RECEIPT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "analysis_split": candidate["analysis_split"],
        "action_family_id": candidate["action_family_id"],
        "calibration_group_id": candidate["calibration_group_id"],
        "actor_group_id": candidate["actor_group_id"],
        "scene_group_id": candidate["scene_group_id"],
        "action_group_id": candidate["action_group_id"],
        "semantic_branch": candidate["semantic_branch"],
        "candidate_envelope_sha256": _sha256(
            row["candidate_envelope_sha256"], label="candidate envelope SHA-256"
        ),
        "root_spec_raw_sha256": _sha256(
            root_spec_raw_sha256, label="root spec SHA-256"
        ),
        "bank_receipt_digest": _sha256(
            bank_receipt_digest, label="bank receipt digest"
        ),
        "generation_receipt_digest": _sha256(
            row["generation_receipt_digest"], label="generation receipt digest"
        ),
        "generation_receipt_file_sha256": _sha256(
            row["generation_receipt_file_sha256"],
            label="generation receipt file SHA-256",
        ),
        "native_rollout_receipt_digest": _sha256(
            row["native_rollout_receipt_digest"],
            label="native rollout receipt digest",
        ),
        "native_rollout_receipt_file_sha256": _sha256(
            row["native_rollout_receipt_file_sha256"],
            label="native rollout receipt file SHA-256",
        ),
        "generated_mp4_sha256": _sha256(
            row["artifacts"]["mp4"]["sha256"], label="generated MP4 SHA-256"
        ),
        "clean_latent_artifact_sha256": _sha256(
            row["artifacts"]["predecode_clean_latent"]["sha256"],
            label="clean latent artifact SHA-256",
        ),
        "geometry_source_video_sha256": _sha256(
            candidate["geometry_source_video_sha256"],
            label="geometry source video SHA-256",
        ),
        "full_t2v_caption_utf8_sha256": _sha256(
            candidate["full_t2v_caption_utf8_sha256"],
            label="candidate full T2V caption SHA-256",
        ),
        "checkpoint_content_identity": checked_checkpoint_identity,
        "frozen_checkpoint_receipt_digest": _sha256(
            checkpoint_receipt_digest,
            label="frozen checkpoint receipt digest",
        ),
        "checkpoint_content_binding": checkpoint_binding,
        "frozen_scorer_receipt_digest": _sha256(
            score.receipt["digest"], label="frozen scorer receipt digest"
        ),
        "frozen_t2v_packet_binding": packet_binding,
        "generation_runtime_binding_by_branch": checked_generation_registry,
        "generation_runtime_registry_digest": object_sha256(
            checked_generation_registry
        ),
        "full_t2v_caption_by_branch": checked_captions,
        "prompt_by_branch": rebuilt_prompts,
        "prompt_builder_contract": prompt_builder_contract(),
        "scorer_runtime_versions": checked_scorer_runtime_versions,
        "prompt_registry_digest": native_bridge.object_sha256(dict(prompt_by_branch)),
        "prompt_utf8_sha256_by_branch": prompt_hashes,
        "full_t2v_caption_utf8_sha256_by_branch": checked_caption_hashes,
        "clean_latent_tensor_sha256": tensor_sha256(clean),
        "official_gaussian_tensor_sha256": tensor_sha256(epsilon),
        "official_gaussian_artifact_sha256": row["artifacts"][
            "official_initial_gaussian"
        ]["sha256"],
        "official_gaussian_raw_value_sha256": _sha256(
            row["artifacts"]["official_initial_gaussian"]["raw_value_sha256"],
            label="official Gaussian raw-value SHA-256",
        ),
        "official_gaussian_content_sha256": _sha256(
            row["artifacts"]["official_initial_gaussian"]["content_sha256"],
            label="official Gaussian content SHA-256",
        ),
        "sigma_tensor_sha256": tensor_sha256(sigma),
        "schedule_coordinate": schedule_coordinate_receipt(),
        "phase_weight_commitment": commitment,
        "phase_weight_registration_digest": commitment["registration_digest"],
        "energy_epsilon": float(mace.DEFAULT_ENERGY_EPSILON),
        "raw_global_action_energy_score": reward,
        "raw_phase_conjunctive_score_diagnostic": phase_reward,
        "global_action_energy": action_energy,
        "global_hard_negative_energy_by_branch": negative_energies,
        "global_negative_log_energy_ratio_by_branch": negative_log_ratios,
        "global_hardest_negative_branch": hardest_branch,
        "mace_live_tensor_formula_proof": live_mace_formula_proof,
        "phase_diagnostic_receipt_digest": score.phase_energy.receipt[
            "receipt_digest"
        ],
        "phase_diagnostic_used_for_calibration": False,
        "input_closure": SCORE_INPUT_CLOSURE,
        "scientific_action_editing_claim": False,
    }
    return {**value, "receipt_digest": object_sha256(value)}


def validate_score_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_SCORE_RECEIPT_FIELDS):
        raise PairV5T2VEnergyScoringError("score receipt field closure differs")
    row = dict(value)
    digest = _verify_embedded_receipt(row, label="score receipt")
    for name in (
        "generation_runtime_binding_by_branch",
        "full_t2v_caption_by_branch",
        "prompt_by_branch",
        "prompt_utf8_sha256_by_branch",
        "full_t2v_caption_utf8_sha256_by_branch",
        "global_hard_negative_energy_by_branch",
        "global_negative_log_energy_ratio_by_branch",
        "scorer_runtime_versions",
    ):
        if not isinstance(row[name], Mapping):
            raise PairV5T2VEnergyScoringError(
                f"score receipt {name} mapping closure differs"
            )
    if (
        row["schema_version"] != SCORE_RECEIPT_SCHEMA
        or row["input_closure"] != SCORE_INPUT_CLOSURE
        or row["phase_diagnostic_used_for_calibration"] is not False
        or row["scientific_action_editing_claim"] is not False
        or row["semantic_branch"] not in mace.BRANCH_ORDER
        or row["schedule_coordinate"] != schedule_coordinate_receipt()
        or row["prompt_builder_contract"] != prompt_builder_contract()
        or set(row["full_t2v_caption_by_branch"]) != set(mace.BRANCH_ORDER)
        or set(row["generation_runtime_binding_by_branch"])
        != set(mace.BRANCH_ORDER)
        or set(row["prompt_by_branch"]) != set(mace.BRANCH_ORDER)
        or set(row["prompt_utf8_sha256_by_branch"]) != set(mace.BRANCH_ORDER)
        or set(row["full_t2v_caption_utf8_sha256_by_branch"])
        != set(mace.BRANCH_ORDER)
        or set(row["global_hard_negative_energy_by_branch"])
        != set(mace.HARD_NEGATIVE_BRANCHES)
        or set(row["global_negative_log_energy_ratio_by_branch"])
        != set(mace.HARD_NEGATIVE_BRANCHES)
    ):
        raise PairV5T2VEnergyScoringError("score receipt semantic closure differs")
    for name in (
        "raw_global_action_energy_score",
        "raw_phase_conjunctive_score_diagnostic",
        "global_action_energy",
        "energy_epsilon",
    ):
        if type(row[name]) is not float or not math.isfinite(row[name]):
            raise PairV5T2VEnergyScoringError(f"score receipt {name} differs")
    if native_bridge.object_sha256(row["prompt_by_branch"]) != row["prompt_registry_digest"]:
        raise PairV5T2VEnergyScoringError("score prompt registry digest differs")
    for branch in mace.BRANCH_ORDER:
        caption = row["full_t2v_caption_by_branch"][branch]
        if type(caption) is not str or hashlib.sha256(
            caption.encode("utf-8")
        ).hexdigest() != _sha256(
            row["full_t2v_caption_utf8_sha256_by_branch"][branch],
            label=f"{branch} full T2V caption UTF-8 SHA-256",
        ):
            raise PairV5T2VEnergyScoringError(
                "score sealed caption text/hash binding differs"
            )
        if hashlib.sha256(row["prompt_by_branch"][branch].encode("utf-8")).hexdigest() != _sha256(
            row["prompt_utf8_sha256_by_branch"][branch],
            label=f"{branch} prompt UTF-8 SHA-256",
        ):
            raise PairV5T2VEnergyScoringError("score prompt text hash differs")
    rebuilt_prompts = official_prompt_bank_from_captions(
        row["full_t2v_caption_by_branch"]
    )
    if row["prompt_by_branch"] != rebuilt_prompts:
        raise PairV5T2VEnergyScoringError(
            "score prompts were not rebuilt from sealed captions by the official builder"
        )
    if (
        row["full_t2v_caption_utf8_sha256"]
        != row["full_t2v_caption_utf8_sha256_by_branch"][row["semantic_branch"]]
    ):
        raise PairV5T2VEnergyScoringError(
            "candidate caption hash differs from its prompt-cell registry"
        )
    try:
        commitment = phase_energy.validate_phase_weight_commitment(
            row["phase_weight_commitment"]
        )
    except phase_energy.PairV5PhaseEnergyError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error
    if commitment["registration_digest"] != _sha256(
        row["phase_weight_registration_digest"],
        label="phase weight registration digest",
    ):
        raise PairV5T2VEnergyScoringError("score phase commitment digest differs")
    checked_identity = _validated_checkpoint_identity(
        row["checkpoint_content_identity"]
    )
    if object_sha256(checked_identity) != _sha256(
        row["frozen_checkpoint_receipt_digest"],
        label="frozen checkpoint receipt digest",
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen checkpoint receipt does not bind its content manifest"
        )
    binding = row["checkpoint_content_binding"]
    if not isinstance(binding, Mapping) or set(binding) != set(
        _CHECKPOINT_BINDING_FIELDS
    ):
        raise PairV5T2VEnergyScoringError("checkpoint content binding differs")
    rebuilt_binding = checkpoint_content_binding(
        checked_identity, binding.get("freeze_certificate")
    )
    if dict(binding) != rebuilt_binding:
        raise PairV5T2VEnergyScoringError(
            "checkpoint manifest/freeze binding differs"
        )
    checked_scorer_versions = _validated_runtime_versions(
        row["scorer_runtime_versions"], label="scorer runtime versions"
    )
    if checked_scorer_versions != current_runtime_versions():
        raise PairV5T2VEnergyScoringError(
            "score receipt runtime differs from the validating runtime"
        )
    generation_registry = validate_generation_runtime_registry(
        row["generation_runtime_binding_by_branch"],
        caption_by_branch=row["full_t2v_caption_by_branch"],
        prompt_by_branch=row["prompt_by_branch"],
        checkpoint_identity=checked_identity,
        scorer_runtime_versions=checked_scorer_versions,
    )
    if object_sha256(generation_registry) != _sha256(
        row["generation_runtime_registry_digest"],
        label="generation runtime registry digest",
    ):
        raise PairV5T2VEnergyScoringError(
            "generation runtime registry digest differs"
        )
    candidate_generation_binding = generation_registry[row["semantic_branch"]]
    if (
        candidate_generation_binding["candidate_id"] != row["candidate_id"]
        or candidate_generation_binding["generation_receipt_digest"]
        != row["generation_receipt_digest"]
        or candidate_generation_binding["native_rollout_receipt_digest"]
        != row["native_rollout_receipt_digest"]
    ):
        raise PairV5T2VEnergyScoringError(
            "score candidate does not join its generation runtime binding"
        )
    packet = row["frozen_t2v_packet_binding"]
    if not isinstance(packet, Mapping) or set(packet) != set(
        _FROZEN_T2V_PACKET_BINDING_FIELDS
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen T2V packet binding field closure differs"
        )
    unsigned_packet = dict(packet)
    packet_binding_digest = _sha256(
        unsigned_packet.pop("binding_digest", None),
        label="frozen T2V packet binding digest",
    )
    coordinate = schedule_coordinate_receipt()
    candidate_shape = packet["candidate_shape"]
    if not isinstance(candidate_shape, list):
        raise PairV5T2VEnergyScoringError(
            "frozen T2V packet candidate shape differs"
        )
    if (
        object_sha256(unsigned_packet) != packet_binding_digest
        or _sha256(
            packet["prompt_registry_digest"],
            label="packet prompt registry digest",
        )
        != row["prompt_registry_digest"]
        or _sha256(
            packet["frozen_model_receipt_digest"],
            label="packet frozen model receipt digest",
        )
        != row["frozen_checkpoint_receipt_digest"]
        or candidate_shape[:3] != [1, 16, 21]
        or len(candidate_shape) != 5
        or any(
            type(item) is not int or item <= 0
            for item in candidate_shape
        )
        or packet["sigma_float32_bits_hex"]
        != coordinate["physical_sigma_float32_be_hex"]
        or packet["timestep_float32_bits_hex"]
        != coordinate["frozen_t2v_scorer_timestep_float32_be_hex"]
        or packet["native_schedule_digest"] != coordinate["schedule_digest"]
        or packet["native_schedule_index"] != PILOT_SCHEDULE_INDEX
        or packet["native_scheduler_timestep"]
        != PILOT_NATIVE_SCHEDULER_TIMESTEP
        or packet["timestep_mapping"]
        != coordinate["frozen_t2v_scorer_timestep_mapping"]
        or packet[
            "physical_sigma_and_model_timestep_share_native_exact40_index"
        ]
        is not True
        or packet["legacy_1000_sigma_timestep_rejected"] is not True
    ):
        raise PairV5T2VEnergyScoringError(
            "frozen T2V packet native exact40 binding differs"
        )
    _validated_energy_scalars(
        action_energy=row["global_action_energy"],
        negative_energy_by_branch=row["global_hard_negative_energy_by_branch"],
        negative_log_ratio_by_branch=row[
            "global_negative_log_energy_ratio_by_branch"
        ],
        raw_reward=row["raw_global_action_energy_score"],
        hardest_negative_branch=row["global_hardest_negative_branch"],
        energy_epsilon=row["energy_epsilon"],
        live_tensor_formula_proof=row["mace_live_tensor_formula_proof"],
    )
    for name in (
        "candidate_envelope_sha256",
        "root_spec_raw_sha256",
        "bank_receipt_digest",
        "generation_receipt_digest",
        "generation_receipt_file_sha256",
        "native_rollout_receipt_digest",
        "native_rollout_receipt_file_sha256",
        "generated_mp4_sha256",
        "clean_latent_artifact_sha256",
        "geometry_source_video_sha256",
        "full_t2v_caption_utf8_sha256",
        "frozen_checkpoint_receipt_digest",
        "frozen_scorer_receipt_digest",
        "clean_latent_tensor_sha256",
        "official_gaussian_tensor_sha256",
        "official_gaussian_artifact_sha256",
        "official_gaussian_raw_value_sha256",
        "official_gaussian_content_sha256",
        "sigma_tensor_sha256",
        "phase_diagnostic_receipt_digest",
    ):
        _sha256(row[name], label=name)
    row["receipt_digest"] = digest
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-spec", required=True)
    parser.add_argument("--expected-root-spec-sha256", required=True)
    parser.add_argument("--bank-output-dir", required=True)
    parser.add_argument("--bank-receipt", required=True)
    parser.add_argument("--expected-bank-receipt-sha256", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-bernini-commit",
        default=native_generation.legacy.trainer.BERNINI_OFFICIAL_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit",
        default=native_generation.legacy.trainer.VEOMNI_TESTED_COMMIT,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--ack-no-action-success-claim", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "expected_root_spec_sha256",
        "expected_bank_receipt_sha256",
        "method_source_archive_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
            raise PairV5T2VEnergyScoringError(f"{name} must be lowercase SHA-1")
    if args.ack_no_action_success_claim is not True:
        raise PairV5T2VEnergyScoringError("no-action-success acknowledgement is mandatory")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_cli(args)
    spec, bank, bound_rows = load_group_bank(
        root_spec=args.root_spec,
        root_spec_sha256=args.expected_root_spec_sha256,
        bank_output_dir=args.bank_output_dir,
        bank_receipt=args.bank_receipt,
        bank_receipt_sha256=args.expected_bank_receipt_sha256,
        group_id=args.group_id,
    )
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists() or output.is_symlink():
        raise PairV5T2VEnergyScoringError("output must be a fresh absolute directory")

    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.trainer.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise PairV5T2VEnergyScoringError("pinned Bernini attention heads differ")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        raise PairV5T2VEnergyScoringError("frozen scorer requires four AUH ROCm GPUs")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=120),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)

    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        try:
            identity = native_generation.source_audit.validate_checkpoint_content(
                checkpoint, Path(args.checkpoint_content_manifest)
            )
            checkpoint_rows[0] = {"ok": True, "identity": identity}
        except Exception as error:  # broadcast the exact fail-closed reason
            checkpoint_rows[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_result = checkpoint_rows[0]
    if not isinstance(checkpoint_result, Mapping) or checkpoint_result.get("ok") is not True:
        raise PairV5T2VEnergyScoringError(
            f"rank-zero checkpoint audit failed: {checkpoint_result}"
        )
    checkpoint_identity = dict(checkpoint_result["identity"])
    checkpoint_receipt_digest = object_sha256(checkpoint_identity)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.trainer.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    try:
        freeze_certificate = native_generation.source_audit.model_freeze_certificate(
            renderer
        )
    except Exception as error:
        raise PairV5T2VEnergyScoringError(str(error)) from error
    checkpoint_content_binding(checkpoint_identity, freeze_certificate)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise PairV5T2VEnergyScoringError("global MACE requires transformer_1 only")
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise PairV5T2VEnergyScoringError("frozen renderer contains trainable parameters")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    scorer_runtime_versions = current_runtime_versions()

    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in bound_rows:
        by_cell.setdefault(row["candidate"]["calibration_group_id"], []).append(row)
    commitment = diagnostic_phase_commitment()
    sigma = torch.tensor([PILOT_SIGMA], dtype=torch.float32, device=device)
    if distributed.rank == 0:
        output.mkdir(parents=True)
    dist.barrier()
    receipts: list[dict[str, Any]] = []
    for cell_id, rows in by_cell.items():
        prompts = prompt_bank_from_cell(
            rows,
            task_prompt_builder=lambda text: native_generation.build_task_prompt(
                "t2v", text, prompt_cleaner=prompt_clean
            ),
        )
        caption_hashes = {
            row["candidate"]["semantic_branch"]: row["candidate"][
                "full_t2v_caption_utf8_sha256"
            ]
            for row in rows
        }
        captions = {
            row["candidate"]["semantic_branch"]: row["candidate"][
                "full_t2v_caption"
            ]
            for row in rows
        }
        generation_runtime_bindings = {
            row["candidate"]["semantic_branch"]: row[
                "generation_runtime_binding"
            ]
            for row in rows
        }
        if list(caption_hashes) != list(mace.BRANCH_ORDER):
            raise PairV5T2VEnergyScoringError("cell caption hash order differs")
        if list(captions) != list(mace.BRANCH_ORDER):
            raise PairV5T2VEnergyScoringError("cell caption text order differs")
        if list(generation_runtime_bindings) != list(mace.BRANCH_ORDER):
            raise PairV5T2VEnergyScoringError(
                "cell generation runtime binding order differs"
            )
        conditions = _encode_prompt_bank(
            renderer, tokenizer, prompts, device=device
        )
        scorer = NativeExact40FrozenBerniniT2VScorer(
            diffusion,
            transformer,
            prompts,
            conditions,
            frozen_model_receipt_digest=checkpoint_receipt_digest,
            model_id="transformer_1",
        )
        first_gaussian = _load_exact81_tensor(
            rows[0]["artifacts"]["official_initial_gaussian"],
            key="official_initial_gaussian",
            label=f"{cell_id} official Gaussian",
        )
        first_gaussian_identity = verify_native_tensor_value_identity(
            first_gaussian,
            rows[0]["artifacts"]["official_initial_gaussian"],
            label=f"{cell_id} first official Gaussian",
        )
        epsilon = first_gaussian.to(device=device).contiguous()
        for row_index, row in enumerate(rows):
            first_identity = rows[0]["artifacts"]["official_initial_gaussian"]
            candidate_identity = row["artifacts"]["official_initial_gaussian"]
            if any(
                candidate_identity.get(field) != first_identity.get(field)
                for field in (
                    "raw_value_sha256",
                    "content_sha256",
                    "shape",
                    "dtype",
                    "stored_dtype",
                    "generator_initial_seed",
                )
            ):
                raise PairV5T2VEnergyScoringError(
                    "same-cell official Gaussian tensor value drifted"
                )
            candidate_gaussian = (
                first_gaussian
                if row_index == 0
                else _load_exact81_tensor(
                    candidate_identity,
                    key="official_initial_gaussian",
                    label=(
                        f"{row['candidate']['candidate_id']} official Gaussian"
                    ),
                )
            )
            actual_candidate_identity = verify_native_tensor_value_identity(
                candidate_gaussian,
                candidate_identity,
                label=f"{row['candidate']['candidate_id']} official Gaussian",
            )
            if (
                actual_candidate_identity != first_gaussian_identity
                or not torch.equal(candidate_gaussian, first_gaussian)
            ):
                raise PairV5T2VEnergyScoringError(
                    "same-cell official Gaussian actual tensor values differ"
                )
            clean_cpu = _load_exact81_tensor(
                row["artifacts"]["predecode_clean_latent"],
                key="normalized_clean_latent",
                label=f"{row['candidate']['candidate_id']} clean latent",
            )
            clean = clean_cpu.to(device=device).contiguous()
            if clean.shape != epsilon.shape:
                raise PairV5T2VEnergyScoringError("candidate/Gaussian geometry differs")
            result = native_bridge.score_frozen_t2v_action_energy(
                clean,
                epsilon,
                sigma,
                prompts,
                scorer,
                commitment,
                registered_phase_weight_digest=commitment["registration_digest"],
            )
            try:
                freeze_after = native_generation.source_audit.model_freeze_certificate(
                    renderer
                )
            except Exception as error:
                raise PairV5T2VEnergyScoringError(str(error)) from error
            if freeze_after != freeze_certificate or any(
                parameter.requires_grad for parameter in renderer.parameters()
            ):
                raise PairV5T2VEnergyScoringError(
                    "frozen renderer changed during candidate scoring"
                )
            scorer_packet_receipt = scorer.last_packet_receipt
            if not isinstance(scorer_packet_receipt, Mapping):
                raise PairV5T2VEnergyScoringError(
                    "frozen scorer emitted no native timestep packet receipt"
                )
            receipt = make_score_receipt(
                row=row,
                root_spec_raw_sha256=args.expected_root_spec_sha256,
                bank_receipt_digest=bank["receipt_digest"],
                checkpoint_identity=checkpoint_identity,
                freeze_certificate=freeze_certificate,
                generation_runtime_binding_by_branch=generation_runtime_bindings,
                scorer_runtime_versions=scorer_runtime_versions,
                prompt_by_branch=prompts,
                caption_by_branch=captions,
                caption_sha256_by_branch=caption_hashes,
                phase_weight_commitment=commitment,
                clean=clean,
                epsilon=epsilon,
                sigma=sigma,
                score=result,
                scorer_packet_receipt=scorer_packet_receipt,
            )
            validate_score_receipt(receipt)
            digests: list[Any] = [None] * distributed.world_size
            dist.all_gather_object(digests, receipt["receipt_digest"])
            if len(set(digests)) != 1:
                raise PairV5T2VEnergyScoringError("SP4 scalar score receipts differ")
            if distributed.rank == 0:
                candidate_dir = output / row["candidate"]["candidate_id"]
                candidate_dir.mkdir()
                receipt_path = candidate_dir / "pair-v5-t2v-global-energy-score-v3.json"
                receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
                os.chmod(receipt_path, 0o400)
                receipts.append(receipt)
            if row_index != 0:
                del candidate_gaussian
            del clean, clean_cpu, result
        del scorer, conditions, epsilon, first_gaussian, first_gaussian_identity

    if distributed.rank == 0:
        group_value = {
            "schema_version": GROUP_RECEIPT_SCHEMA,
            "group_id": args.group_id,
            "root_spec_raw_sha256": args.expected_root_spec_sha256,
            "bank_receipt_digest": bank["receipt_digest"],
            "frozen_checkpoint_receipt_digest": checkpoint_receipt_digest,
            "checkpoint_content_binding": checkpoint_content_binding(
                checkpoint_identity, freeze_certificate
            ),
            "schedule_coordinate": schedule_coordinate_receipt(),
            "candidate_count": len(receipts),
            "candidate_receipt_digests": [row["receipt_digest"] for row in receipts],
            "primary_score_field": "raw_global_action_energy_score",
            "phase_conjunctive_role": "diagnostic_only_never_calibration_gate",
            "input_closure": SCORE_INPUT_CLOSURE,
            "training_performed": False,
            "optimizer_authorized": False,
            "scientific_action_editing_claim": False,
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
        }
        group_receipt = {
            **group_value,
            "receipt_digest": object_sha256(group_value),
        }
        path = output / f"pair-v5-t2v-global-energy-{args.group_id}-v3.json"
        path.write_bytes(canonical_json_bytes(group_receipt) + b"\n")
        os.chmod(path, 0o400)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GROUP_RECEIPT_SCHEMA",
    "PILOT_GATE_NAME",
    "PILOT_NATIVE_SCHEDULER_TIMESTEP",
    "PILOT_SCHEDULE_INDEX",
    "PILOT_SCORER_PHYSICAL_TIMESTEP",
    "PILOT_SIGMA",
    "PairV5T2VEnergyScoringError",
    "SCORE_INPUT_CLOSURE",
    "SCORE_RECEIPT_SCHEMA",
    "current_runtime_versions",
    "diagnostic_phase_commitment",
    "generation_runtime_binding_from_native_receipt",
    "load_group_bank",
    "make_score_receipt",
    "prompt_bank_from_cell",
    "schedule_coordinate_receipt",
    "validate_generation_runtime_registry",
    "validate_score_receipt",
]

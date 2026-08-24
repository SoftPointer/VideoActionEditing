#!/usr/bin/env python3
"""Fail-closed native execution runtime for the Round37 oracle canary.

This is an independent copy of the reviewed activation/runtime logic.  It does
not modify or rebind the frozen materializer authority core.  Diagnostic
execution authority is an exact model-reviewed packet and a distinct
model-review ledger receipt whose SHA-256 values are compiled into
this source *after* those files are finalized.  Caller, CLI, environment, and
JSON values cannot supply or override either trust anchor.

The exact independently issued e02 packet and ledger roots are compiled below.
The surrounding runner remains inert until its separate spec/component release
pins are finalized and validated before any Torch/model/distributed import.
e03 is an abstention-only policy row binding an existing frozen base; it has no
material receipt, active arm, or execution capability.
No FlowEdit, connected route, learned gate, optimizer, or training surface is
provided here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence

# Resolve import origins before executing any repository-owned dependency.
# The launcher must additionally put METHOD_ROOT first and clear PYTHONPATH;
# this guard ensures a shadow module cannot be selected even before the later
# full byte/closure validation runs.
METHOD_ROOT = Path(__file__).resolve().parent
_EXPECTED_IMPORT_ORIGINS = {
    "native_branch_homotopy_runtime_v1": "native_branch_homotopy_runtime_v1.py",
    "native_branch_homotopy_v1": "native_branch_homotopy_v1.py",
    "oracle_regeneration_canary_v1": "oracle_regeneration_canary_v1.py",
    "self_guided_action_field_v1": "self_guided_action_field_v1.py",
    "source_self_native_ref_contrastive_v3": (
        "source_self_native_ref_contrastive_v3.py"
    ),
    "tri_branch_unipc": "tri_branch_unipc.py",
}
for _module_name, _relative_name in _EXPECTED_IMPORT_ORIGINS.items():
    try:
        _spec = importlib.util.find_spec(_module_name)
    except Exception as _error:
        raise RuntimeError(
            f"activation dependency import origin differs: {_module_name}"
        ) from _error
    _origin = getattr(_spec, "origin", None)
    if (
        not isinstance(_origin, str)
        or Path(_origin).resolve(strict=True)
        != (METHOD_ROOT / _relative_name).resolve(strict=True)
    ):
        raise RuntimeError(f"activation dependency import origin differs: {_module_name}")

import native_branch_homotopy_runtime_v1 as native_runtime
import native_branch_homotopy_v1 as native_homotopy
import oracle_regeneration_canary_v1 as safe_core
import self_guided_action_field_v1 as sgaf
import tri_branch_unipc as sampler_contract


SCHEMA_VERSION = "bernini-oracle-regeneration-native-runtime-activation-v2-v1"
AUTHORITY_PACKET_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-authority-packet-v1"
)
LEDGER_RECEIPT_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-external-ledger-receipt-v1"
)
REFERENCE_RECEIPT_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-vae-reference-receipt-v1"
)
PROMPT_RECEIPT_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-prompt-receipt-v1"
)
LOCAL_RUNTIME_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-local-runtime-receipt-v1"
)
MANUAL_GATE_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-manual-dck-gate-v1"
)
MANUAL_GATE_REVIEW_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-manual-dck-review-v1"
)
MANUAL_GATE_LEAF_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-manual-dck-leaf-v1"
)
AUTHORING_TEMPLATE_SCHEMA_VERSION = (
    "bernini-oracle-regeneration-activation-v2-authoring-template-v1"
)

ALLOWED_CASES = ("e02", "e03")
REFERENCE_RGB_INDICES = (0, 27, 53, 80)
REFERENCE_LATENT_PHASES = (0, 7, 13, 20)
ARM_OFFICIAL = "official-v2v-base"
ARM_LOCAL = "local-source-reference-r2v4-in-manual-G"
EXPECTED_ARMS_E02 = (ARM_OFFICIAL, ARM_LOCAL)
EXPECTED_ARMS_E03: tuple[str, ...] = ()
EXPECTED_LATENT_GEOMETRY = {
    "e02": (1, 16, 21, 74, 50),
    "e03": (1, 16, 21, 70, 52),
}
EXPECTED_GATE_GEOMETRY = {
    case_id: (shape[0], 1, shape[2], shape[3], shape[4])
    for case_id, shape in EXPECTED_LATENT_GEOMETRY.items()
}
EXPECTED_REFERENCE_GEOMETRY = {
    case_id: (shape[0], shape[1], 1, shape[3], shape[4])
    for case_id, shape in EXPECTED_LATENT_GEOMETRY.items()
}
EXPECTED_SOURCE_INPUT_HW = {
    "e02": (1056, 704),
    "e03": (928, 704),
}
EXPECTED_CASE_BINDINGS = {
    "e02": {
        "source_iid": "10ed90644f81461d",
        "source_sha256": "63dd620627e18d0f6836058bf725ffba7ab1f9b0b455e784c6a71dc68a8de46c",
        "anchor_sha256": "a1076cbe83c9dae4a4fddc25f73077288f6a3240324fd2d5e1854aa842b07b63",
        "action_caption_sha256": "2d73663bc49f398d6e082cfca00497616c034a66843ad8e74651f5334a34e27f",
        "structured_action_program_sha256": "eec543086a443fda69ae26d5a6893648206d13fac22ba65120db5a4b16361a99",
    },
    "e03": {
        "source_iid": "7a33b36459c84289",
        "source_sha256": "c1455b9b89d1f352da69e7bb07e955ee4495df94f5ef6f3f09fe7fd9eac035bb",
        "anchor_sha256": "1d0a0e8895ec976d3cb1f9ee3070ac36c75ca29247bb52f81677135f7786f12f",
        "action_caption_sha256": "970625b4d83e8aaaa8133d8ca8c8964c6962ef221070f4cece4439e26442be2b",
        "structured_action_program_sha256": "1e4ddea7b5c304541f87986c0071f2eb089414c0dd06c47ddc5f59f110b08fb0",
    },
}
EXPECTED_CASE_SEEDS = {"e02": 0, "e03": 0}
EXPECTED_E02_MATERIAL_REVIEW_SHA256 = (
    "788cf7b83851b79a662c0040e7470fd32a8aea2456ce47e6b53baff0b2a73c6e"
)
EXPECTED_E03_FROZEN_BASE_PATH = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/action_flow_noise_stage0_job140846_v1/stage1/"
    "interaction_complex8_rv2v_candidates_v1/complex8-e03-rv2v-s0/rv2v.mp4"
)
EXPECTED_E03_FROZEN_BASE_SHA256 = (
    "d75bbafbbc225ea3935c2d149be8b3969fffd6d8b645c5ec9edb5968bf25f654"
)

# Exact diagnostic trust roots issued independently after source-only gate and
# material review.  The packet contains no runtime/runner/spec/launcher hash,
# so compiling these bytes does not create a self-pin cycle.  This is model-
# reviewed diagnostic authority only, never formal or training authority.
COMPILED_AUTHORITY_PACKET_SHA256: Optional[str] = (
    "6ae5602350d54696e0ddcd716a311f96a3569c6f062622840ad130fcbba0baeb"
)
COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256: Optional[str] = (
    "5a9efae443bc8d3cb0886dee7f950204377f653f7dbc474f820d7abbbe437e51"
)

FROZEN_DEPENDENCY_PINS = {
    "safe_blocked_oracle_core": (
        "oracle_regeneration_canary_v1.py",
        "0148b137c200e426ff18571f71d373a9e6ef595c620664925dae0ab9d1d91081",
    ),
    "native_five_forward_runtime": (
        "native_branch_homotopy_runtime_v1.py",
        "b81ee152e358e4d5a6638dfccf1232c4e221311ffb38937e61be3c6a799b84d5",
    ),
    "native_homotopy_schedule": (
        "native_branch_homotopy_v1.py",
        "2585416e61935db62cc7534daf19b4bb851f9fdcdeb92f78e6152f55e034f3d0",
    ),
    "self_guided_action_field": (
        "self_guided_action_field_v1.py",
        "2ad204c09f5eb60865017b1e596de25b777d8d6ed43774f4dcbc23a4ad58bc7e",
    ),
    "native_unipc40_schedule_contract": (
        "source_self_native_ref_contrastive_v3.py",
        "d8825bc167c64e497f8d29c807d9b0a69d9a9a59de09afee863b7fc9df2bdeb0",
    ),
    "tri_branch_unipc": (
        "tri_branch_unipc.py",
        "58d2e0e8d56a500eea07ec20f0fb101539ac846bbd039c0d50a22506b58fb3d2",
    ),
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID = re.compile(r"^[a-z0-9][a-z0-9._:@-]{2,63}$")
_PACKET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{7,95}$")
_AUTHORITY_TOKEN = object()
_BINDING_TOKEN = object()
_GATE_TOKEN = object()
_REFERENCE_TOKEN = object()
_PROMPT_TOKEN = object()
_CAPABILITY_TOKEN = object()
_PACKET_KEYS = {
    "schema_version",
    "status",
    "packet_id",
    "execution_contract",
    "safety_contract",
    "cases",
}
_EXECUTION_KEYS = {
    "native_only",
    "flowedit_enabled",
    "connected_runner_enabled",
    "learned_gate_enabled",
    "world_size",
    "sequence_parallel_size",
    "one_node",
    "same_seed_and_official_gaussian",
    "candidate_count_per_arm",
    "source_reference_r2v4_regeneration_expert",
    "self_generated_anchor_tensor_used_by_native_expert",
    "anchor_reference_or_quotient_arm_deferred",
    "global_source_reference_r2v4_upper_bound_arm_deferred",
}
_SAFETY_KEYS = {
    "training_authorized",
    "optimizer_authorized",
    "automatic_model_replacement_authorized",
    "background_cosine_selection_authorized",
    "target_video_or_latent_used",
}
_CASE_COMMON_KEYS = {
    "case_id",
    "source_iid",
    "decision",
    "source_video",
    "self_generated_anchor",
    "anchor_used_only_as_review_context",
    "self_generated_anchor_tensor_used_by_native_expert",
    "target_video_or_latent_used",
    "failed_active_used_to_author_gate",
    "anchor_source_difference_used_to_author_gate",
    "predicted_soft_gate_used_to_author_gate",
    "automatic_model_replacement_authorized",
    "action_caption",
    "action_caption_sha256",
    "structured_action_program",
    "structured_action_program_sha256",
    "seed",
    "full_source_latent_geometry",
    "hard_gate_geometry",
    "reference_latent_geometry",
    "reference_rgb_indices",
    "run_arms",
    "manual_gate_manifest",
    "independent_review_receipt",
    "annotation_authority_root_sha256",
    "vae_reference_receipt",
    "prompt_receipt",
}
_CASE_E02_KEYS = _CASE_COMMON_KEYS | {
    "anchor_terminal_disappearance_observed",
    "anchor_strict_target_pass",
    "vae_authoring_run_receipt",
    "prompt_authoring_run_receipt",
    "materialization_review_receipt",
}
_CASE_E03_KEYS = _CASE_COMMON_KEYS | {
    "local_regeneration_selection_authorized",
    "kept_frozen_base",
}
_LEDGER_KEYS = {
    "schema_version",
    "authority_packet_sha256",
    "packet_id",
    "annotator",
    "reviewer",
    "issuer",
    "annotator_kind",
    "reviewer_kind",
    "issuer_kind",
    "trust_root_kind",
    "accepted",
    "e02_exact_gate_reviewed",
    "e03_abstain_keep_base_reviewed",
    "authority_packet_contains_no_activation_code_hashes",
    "private_signing_material_present",
    "cryptographic_signature_claimed",
    "diagnostic_experimental_canary_only",
    "formal_authority",
    "training_authority",
}
_GATE_KEYS = {
    "schema_version",
    "case_id",
    "source_sha256",
    "anchor_sha256",
    "action_caption_sha256",
    "structured_action_program_sha256",
    "latent_geometry",
    "flattening",
    "dtype",
    "hard_support",
    "phase_zero_empty",
    "delete_rle",
    "create_rle",
    "contact_rle",
    "typed_semantics",
    "mask_sha256",
    "annotation_authority",
    "authority",
    "qualification",
}
_GATE_AUTHORITY_KEYS = {
    "role",
    "training_target_authorized",
    "action_representation_claimed",
    "forbidden_inputs_absent",
}
_GATE_FORBIDDEN_KEYS = {
    "failed_active_video_or_latent",
    "raw_anchor_source_pixel_or_latent_difference",
    "predicted_soft_gate",
    "target_video_or_latent",
    "self_generated_anchor_tensor",
}
_GATE_QUALIFICATION_KEYS = {
    "status",
    "annotator",
    "reviewer",
    "author_kind",
    "reviewer_kind",
    "review_receipt_path",
}
_GATE_ANNOTATION_KEYS = {
    "tree_shape",
    "ledger_root_sha256",
    "leaf_sha256",
    "leaf_index",
    "tree_size",
    "inclusion_proof",
}
_GATE_TYPED_SEMANTICS_KEYS = {
    "delete_D",
    "create_C",
    "contact_ownership_K",
    "execution_support_G",
    "coordinate_system",
    "expected_nonempty_phase_windows",
}
_GATE_REVIEW_KEYS = {
    "schema_version",
    "case_id",
    "source_sha256",
    "anchor_sha256",
    "action_caption_sha256",
    "structured_action_program_sha256",
    "gate_manifest_sha256",
    "mask_sha256",
    "annotation_authority_root_sha256",
    "annotation_authority_leaf_sha256",
    "annotator",
    "reviewer",
    "author_kind",
    "reviewer_kind",
    "source_only_model_proposal",
    "independent_model_review",
    "accepted",
    "phase_zero_source_authority_checked",
    "source_coordinate_authoring_checked",
    "delete_create_contact_semantics_checked",
    "D_C_disjoint_checked",
    "K_preserved_as_independent_channel_checked",
    "G_exact_union_D_C_K_checked",
    "channel_active_windows_checked",
    "no_large_rectangle_shortcut_checked",
    "single_actor_object_component_checked",
    "duplicate_actor_or_object_rejected",
    "terminal_hold_semantics_checked",
    "anchor_terminal_disappearance_observed",
    "anchor_strict_target_pass",
    "anchor_used_only_as_review_context",
    "failed_active_used_to_author_mask",
    "anchor_difference_used_to_author_mask",
    "predicted_soft_gate_used_to_author_mask",
    "target_video_or_latent_used_to_author_mask",
    "self_generated_anchor_tensor_used_to_author_mask",
}
_REFERENCE_RECEIPT_KEYS = {
    "schema_version",
    "case_id",
    "source_iid",
    "source_video_sha256",
    "source_frame_count",
    "source_fps_numerator",
    "source_fps_denominator",
    "source_input_frame_geometry",
    "source_bucket_hw",
    "reference_rgb_indices",
    "reference_raw_rgb_sha256",
    "full_preprocessed_source_identity",
    "reference_preprocessed_rgb_sha256",
    "preprocess_contract",
    "vae_contract",
    "full_source_latent_identity",
    "reference_latent_identities",
    "materializer_code_path",
    "materializer_code_sha256",
    "rank_world_receipt",
    "references_encoded_as_four_independent_rgb_frames",
    "references_not_sliced_from_full_source_latent",
    "source_reference_storage_alias_rejected",
    "reference_content_duplicates_rejected",
    "target_video_or_latent_used",
    "self_generated_anchor_tensor_used",
    "materialization_checks_passed",
}
_PREPROCESS_CONTRACT_KEYS = {
    "frame_decode_backend",
    "frame_decode_code_path",
    "frame_decode_code_sha256",
    "source_prepare_code_path",
    "source_prepare_code_sha256",
    "rgb_dtype",
    "rgb_channel_order",
    "resize_policy",
    "normalization",
}
_VAE_CONTRACT_KEYS = {
    "checkpoint_content_manifest_path",
    "checkpoint_content_manifest_sha256",
    "checkpoint_content_identity_sha256",
    "config_path",
    "config_sha256",
    "vae_code_path",
    "vae_code_sha256",
    "autoencoder_class_module_path",
    "autoencoder_class_module_sha256",
    "diffusers_version",
    "torch_version",
    "python_executable_path",
    "python_executable_sha256",
    "python_version",
    "rocm_version",
    "encode_function",
    "encode_dtype",
    "latent_coordinate",
}
_TENSOR_IDENTITY_KEYS = {"shape", "dtype", "content_sha256"}
_REFERENCE_LATENT_KEYS = {
    "frame_index",
    "raw_rgb_sha256",
    "preprocessed_rgb_sha256",
    "shape",
    "dtype",
    "content_sha256",
    "independently_vae_encoded",
}
_RANK_WORLD_KEYS = {
    "world_size",
    "sequence_parallel_size",
    "rank0_only_vae_encode",
    "all_rank_vae_load_roles",
    "broadcast_exact",
    "all_rank_full_source_latent_sha256",
    "all_rank_reference_latent_sha256",
}
_PROMPT_RECEIPT_KEYS = {
    "schema_version",
    "case_id",
    "source_iid",
    "action_caption",
    "action_caption_sha256",
    "prompt_contract",
    "low_action",
    "high_action",
    "negative",
    "materializer_code_path",
    "materializer_code_sha256",
    "rank_world_receipt",
    "rank0_only_text_encoder_load",
    "nonzero_ranks_never_deserialized_text_encoder",
    "self_generated_anchor_tensor_used",
    "target_video_or_latent_used",
    "materialization_checks_passed",
}
_PROMPT_CONTRACT_KEYS = {
    "tokenizer_config_path",
    "tokenizer_config_sha256",
    "tokenizer_code_path",
    "tokenizer_code_sha256",
    "checkpoint_content_manifest_path",
    "checkpoint_content_manifest_sha256",
    "checkpoint_content_identity_sha256",
    "text_encoder_config_path",
    "text_encoder_config_sha256",
    "renderer_code_path",
    "renderer_code_sha256",
    "prompt_builder_code_path",
    "prompt_builder_code_sha256",
    "native_prompt_code_path",
    "native_prompt_code_sha256",
    "prompt_cleaner_code_path",
    "prompt_cleaner_code_sha256",
    "auto_tokenizer_module_path",
    "auto_tokenizer_module_sha256",
    "resolved_tokenizer_class_module_path",
    "resolved_tokenizer_class_module_sha256",
    "text_encoder_class_module_path",
    "text_encoder_class_module_sha256",
    "transformers_version",
    "torch_version",
    "python_executable_path",
    "python_executable_sha256",
    "python_version",
    "rocm_version",
    "tokenizer_function",
    "text_encoder_function",
    "max_length",
    "embedding_dtype",
}
_PROMPT_ROLE_KEYS = {
    "mode",
    "rendered_text",
    "rendered_text_sha256",
    "token_ids_sha256",
    "attention_mask_sha256",
    "embedding_identity",
}
_PROMPT_RANK_WORLD_KEYS = {
    "world_size",
    "sequence_parallel_size",
    "rank0_only_text_encode",
    "broadcast_exact",
    "all_rank_low_action_sha256",
    "all_rank_high_action_sha256",
    "all_rank_negative_sha256",
    "all_rank_text_encoder_load_roles",
}


class OracleActivationV2Error(RuntimeError):
    """Raised before model/distributed initialization on any authority drift."""


@dataclass(frozen=True)
class OwnedFileSealV2:
    path: Path
    sha256: str
    device: int
    inode: int
    size: int
    mode: int
    nlink: int
    mtime_ns: int
    ctime_ns: int


def _seal_plain_file_v2(
    path: Path, *, label: str, retain_bytes: bool, require_frozen: bool = False
) -> tuple[OwnedFileSealV2, Optional[bytes]]:
    if not path.is_absolute() or path.is_symlink():
        raise OracleActivationV2Error(f"{label} must be a plain absolute file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise OracleActivationV2Error(f"{label} open failed") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (require_frozen and before.st_mode & 0o222)
        ):
            raise OracleActivationV2Error(
                f"{label} must be one-link regular-file authority"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if retain_bytes:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    try:
        named = path.lstat()
    except OSError as error:
        raise OracleActivationV2Error(f"{label} path disappeared") from error
    identity_named = (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mode,
        named.st_nlink,
        named.st_mtime_ns,
        named.st_ctime_ns,
    )
    if identity_before != identity_after or identity_after != identity_named:
        raise OracleActivationV2Error(f"{label} changed during owned read")
    seal = OwnedFileSealV2(
        path=path,
        sha256=digest.hexdigest(),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        size=int(after.st_size),
        mode=int(after.st_mode),
        nlink=int(after.st_nlink),
        mtime_ns=int(after.st_mtime_ns),
        ctime_ns=int(after.st_ctime_ns),
    )
    return seal, b"".join(chunks) if retain_bytes else None


def _strict_json_bytes_v2(value: bytes, *, label: str) -> Mapping[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise OracleActivationV2Error(f"{label} duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except OracleActivationV2Error:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise OracleActivationV2Error(f"{label} JSON differs") from error
    if not isinstance(parsed, Mapping):
        raise OracleActivationV2Error(f"{label} root must be an object")
    return parsed


def _seal_json_v2(path: Path, *, label: str) -> tuple[OwnedFileSealV2, Mapping[str, Any]]:
    seal, raw = _seal_plain_file_v2(
        path, label=label, retain_bytes=True, require_frozen=True
    )
    if raw is None:
        raise OracleActivationV2Error(f"{label} owned bytes are absent")
    return seal, _strict_json_bytes_v2(raw, label=label)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise OracleActivationV2Error(f"{label} must be lowercase SHA-256")
    return value


def _plain_absolute_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OracleActivationV2Error(f"{label} path is absent")
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise OracleActivationV2Error(f"{label} must be a plain absolute file")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise OracleActivationV2Error(f"{label} file is absent") from error
    if resolved != requested or not resolved.is_file() or resolved.is_symlink():
        raise OracleActivationV2Error(f"{label} path identity differs")
    return resolved


def _strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    return _seal_json_v2(path, label=label)[1]


def _canonical_object_sha256(value: Any) -> str:
    return hashlib.sha256(safe_core.canonical_json_bytes_v1(value)).hexdigest()


def verify_frozen_dependency_pins_v2() -> Mapping[str, str]:
    observed: dict[str, str] = {}
    modules = {
        "safe_blocked_oracle_core": safe_core,
        "native_five_forward_runtime": native_runtime,
        "native_homotopy_schedule": native_homotopy,
        "self_guided_action_field": sgaf,
        "tri_branch_unipc": sampler_contract,
    }
    for label, (relative, expected_sha256) in FROZEN_DEPENDENCY_PINS.items():
        path = (METHOD_ROOT / relative).resolve(strict=True)
        if label == "native_unipc40_schedule_contract":
            try:
                spec = importlib.util.find_spec(
                    "source_self_native_ref_contrastive_v3"
                )
            except Exception as error:
                raise OracleActivationV2Error(
                    "native UniPC40 schedule import origin is absent"
                ) from error
            origin = getattr(spec, "origin", None)
            if not isinstance(origin, str):
                raise OracleActivationV2Error(
                    "native UniPC40 schedule import origin is absent"
                )
            module_path = Path(origin).resolve(strict=True)
        else:
            module_path = Path(
                str(getattr(modules[label], "__file__", ""))
            ).resolve(strict=True)
        if module_path != path:
            raise OracleActivationV2Error(f"{label} imported module path differs")
        seal, _ = _seal_plain_file_v2(path, label=label, retain_bytes=False)
        digest = seal.sha256
        if digest != expected_sha256:
            raise OracleActivationV2Error(f"{label} dependency bytes differ")
        observed[label] = digest
    if (
        safe_core.native_runtime is not native_runtime
        or safe_core.homotopy is not native_homotopy
        or safe_core.sgaf is not sgaf
        or native_runtime.homotopy is not native_homotopy
        or native_runtime.sgaf is not sgaf
        or native_runtime.sampler_contract is not sampler_contract
    ):
        raise OracleActivationV2Error("frozen dependency import-object closure differs")
    if "source_self_native_ref_contrastive_v3" in sys.modules:
        _load_native_schedule_contract_v2()
    return observed


def _load_native_schedule_contract_v2() -> Any:
    """Load and authenticate the module imported dynamically by the B patch."""

    module_name = "source_self_native_ref_contrastive_v3"
    expected_path = (METHOD_ROOT / f"{module_name}.py").resolve(strict=True)
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as error:
        raise OracleActivationV2Error(
            "native UniPC40 schedule import origin differs"
        ) from error
    origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or Path(origin).resolve(strict=True) != expected_path:
        raise OracleActivationV2Error("native UniPC40 schedule import origin differs")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise OracleActivationV2Error(
            "cannot import pinned native UniPC40 schedule contract"
        ) from error
    module_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    receipt_fn = getattr(module, "native_unipc40_schedule_receipt", None)
    if (
        module_path != expected_path
        or sys.modules.get(module_name) is not module
        or not callable(receipt_fn)
        or Path(receipt_fn.__code__.co_filename).resolve(strict=True) != expected_path
    ):
        raise OracleActivationV2Error("native UniPC40 schedule module object differs")
    seal, _ = _seal_plain_file_v2(
        expected_path,
        label="native UniPC40 schedule contract",
        retain_bytes=False,
    )
    expected_file_sha = FROZEN_DEPENDENCY_PINS[
        "native_unipc40_schedule_contract"
    ][1]
    receipt = receipt_fn()
    expected_schedule_digest = (
        "46f3dcb6e2d65cb7921e5217e2a20dfe008b366cfacafe455fee4d3c45f63ae2"
    )
    if (
        seal.sha256 != expected_file_sha
        or getattr(module, "PINNED_NATIVE_UNIPC40_SCHEDULE_DIGEST", None)
        != expected_schedule_digest
        or not isinstance(receipt, Mapping)
        or receipt.get("digest") != expected_schedule_digest
        or len(tuple(getattr(module, "NATIVE_UNIPC40_TIMESTEPS", ()))) != 40
        or len(tuple(getattr(module, "NATIVE_UNIPC40_SIGMAS", ()))) != 40
    ):
        raise OracleActivationV2Error("native UniPC40 schedule contract differs")
    return module


def compiled_activation_available_v2() -> bool:
    return (
        isinstance(COMPILED_AUTHORITY_PACKET_SHA256, str)
        and _SHA256.fullmatch(COMPILED_AUTHORITY_PACKET_SHA256) is not None
        and isinstance(COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256, str)
        and _SHA256.fullmatch(COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256) is not None
    )


def _artifact_binding(
    value: Any, *, label: str, require_frozen: bool
) -> tuple[Path, str, OwnedFileSealV2]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise OracleActivationV2Error(f"{label} binding differs")
    path = _plain_absolute_file(value.get("path"), label=label)
    expected = _require_sha256(value.get("sha256"), label=label)
    seal, _ = _seal_plain_file_v2(
        path,
        label=label,
        retain_bytes=False,
        require_frozen=require_frozen,
    )
    if seal.sha256 != expected:
        raise OracleActivationV2Error(f"{label} bytes differ")
    return path, expected, seal


def _json_artifact_binding_v2(
    value: Any, *, label: str
) -> tuple[Path, str, OwnedFileSealV2, Mapping[str, Any]]:
    """Seal and parse one frozen packet-bound JSON from the same owned bytes."""

    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise OracleActivationV2Error(f"{label} binding differs")
    path = _plain_absolute_file(value.get("path"), label=label)
    expected = _require_sha256(value.get("sha256"), label=label)
    seal, payload = _seal_json_v2(path, label=label)
    if seal.sha256 != expected:
        raise OracleActivationV2Error(f"{label} bytes differ")
    return path, expected, seal, payload


def _validate_e02_materialization_graph_v2(
    *,
    case_row: Mapping[str, Any],
    source_iid: str,
    source_sha256: str,
    action_caption_sha256: str,
) -> tuple[
    tuple[Path, str, OwnedFileSealV2],
    tuple[Path, str, OwnedFileSealV2],
    tuple[Path, str, OwnedFileSealV2],
    tuple[Path, str, OwnedFileSealV2],
    tuple[Path, str, OwnedFileSealV2],
]:
    """Bind both primary receipts, both run receipts, and independent review."""

    reference_path, reference_sha, reference_seal, reference = (
        _json_artifact_binding_v2(
            case_row.get("vae_reference_receipt"),
            label="e02 VAE reference receipt",
        )
    )
    vae_run_path, vae_run_sha, vae_run_seal, vae_run = (
        _json_artifact_binding_v2(
            case_row.get("vae_authoring_run_receipt"),
            label="e02 VAE authoring run receipt",
        )
    )
    prompt_path, prompt_sha, prompt_seal, prompt = _json_artifact_binding_v2(
        case_row.get("prompt_receipt"), label="e02 prompt receipt"
    )
    prompt_run_path, prompt_run_sha, prompt_run_seal, prompt_run = (
        _json_artifact_binding_v2(
            case_row.get("prompt_authoring_run_receipt"),
            label="e02 prompt authoring run receipt",
        )
    )
    review_path, review_sha, review_seal, review = _json_artifact_binding_v2(
        case_row.get("materialization_review_receipt"),
        label="e02 materialization review receipt",
    )
    seals = (
        reference_seal,
        vae_run_seal,
        prompt_seal,
        prompt_run_seal,
        review_seal,
    )
    if len({(seal.device, seal.inode) for seal in seals}) != len(seals):
        raise OracleActivationV2Error("e02 material receipt files alias")
    if (
        review_sha != EXPECTED_E02_MATERIAL_REVIEW_SHA256
        or reference.get("schema_version") != REFERENCE_RECEIPT_SCHEMA_VERSION
        or reference.get("case_id") != "e02"
        or reference.get("source_iid") != source_iid
        or reference.get("source_video_sha256") != source_sha256
        or reference.get("materialization_checks_passed") is not True
        or prompt.get("schema_version") != PROMPT_RECEIPT_SCHEMA_VERSION
        or prompt.get("case_id") != "e02"
        or prompt.get("source_iid") != source_iid
        or prompt.get("action_caption_sha256") != action_caption_sha256
        or prompt.get("materialization_checks_passed") is not True
        or vae_run.get("schema_version")
        != "bernini-oracle-regeneration-activation-v2-vae-authoring-run-r2"
        or vae_run.get("receipt_sha256") != reference_sha
        or vae_run.get("diagnostic_authoring_material_only") is not True
        or vae_run.get("full_model_or_sampler_loaded") is not False
        or vae_run.get("scheduler_loaded") is not False
        or vae_run.get("transformer_loaded") is not False
        or vae_run.get("training") is not False
        or vae_run.get("optimizer") is not False
        or prompt_run.get("schema_version")
        != "bernini-oracle-regeneration-activation-v2-prompt-authoring-run-r2"
        or prompt_run.get("prompt_receipt_sha256") != prompt_sha
        or prompt_run.get("diagnostic_authoring_material_only") is not True
        or prompt_run.get("sampler_or_scheduler_called") is not False
        or prompt_run.get("denoising_transformer_moved_to_gpu") is not False
        or prompt_run.get("training") is not False
        or prompt_run.get("optimizer") is not False
        or review.get("schema_version")
        != "bernini-oracle-regeneration-e02-ai-agent-diagnostic-material-review-v1"
        or review.get("case_id") != "e02"
        or review.get("decision") != "ACCEPT_E02_DIAGNOSTIC_ONLY"
        or review.get("authority") is not False
        or review.get("accepted_case_ids") != ["e02"]
        or review.get("marker")
        != "AI_AGENT_REVIEW_NONHUMAN_NONFORMAL_NOT_TRAINING_AUTHORITY_DIAGNOSTIC_ONLY"
    ):
        raise OracleActivationV2Error("e02 materialization receipt graph differs")
    authority_flags = review.get("authority_flags")
    audit_results = review.get("audit_results")
    source_binding = review.get("source_binding")
    checkpoint_binding = review.get("checkpoint_binding")
    reviewed_receipts = review.get("materializer_receipts")
    vae_contract = reference.get("vae_contract")
    prompt_contract = prompt.get("prompt_contract")
    if (
        not isinstance(authority_flags, Mapping)
        or authority_flags.get("diagnostic_material_accepted") is not True
        or any(
            authority_flags.get(key) is not False
            for key in (
                "executable_authority",
                "formal_authority",
                "human_annotation",
                "human_review",
                "optimizer_authorized",
                "runnable",
                "training_authorized",
                "automatic_model_replacement_authorized",
            )
        )
        or not isinstance(audit_results, Mapping)
        or audit_results.get("materializer_exact_receipt_closure_passed") is not True
        or audit_results.get("checkpoint_23_file_sha_member_identity_closure_passed")
        is not True
        or audit_results.get("world4_cross_rank_identity_passed") is not True
        or audit_results.get("source_only_materialization") is not True
        or any(
            audit_results.get(key) is not False
            for key in (
                "optimizer_used",
                "sampler_or_scheduler_called",
                "self_generated_anchor_tensor_used",
                "target_video_or_latent_used",
                "training_used",
            )
        )
        or not isinstance(source_binding, Mapping)
        or source_binding.get("source_iid") != source_iid
        or source_binding.get("source_video_sha256") != source_sha256
        or source_binding.get("action_caption_sha256") != action_caption_sha256
        or not isinstance(vae_contract, Mapping)
        or not isinstance(prompt_contract, Mapping)
        or not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("checkpoint_content_identity_sha256")
        != vae_contract.get("checkpoint_content_identity_sha256")
        or checkpoint_binding.get("checkpoint_content_identity_sha256")
        != prompt_contract.get("checkpoint_content_identity_sha256")
        or checkpoint_binding.get("checkpoint_content_manifest_sha256")
        != vae_contract.get("checkpoint_content_manifest_sha256")
        or checkpoint_binding.get("checkpoint_content_manifest_sha256")
        != prompt_contract.get("checkpoint_content_manifest_sha256")
        or checkpoint_binding.get("verified_file_count") != 23
        or checkpoint_binding.get("every_file_sha256_independently_reverified")
        is not True
        or not isinstance(reviewed_receipts, Mapping)
        or set(reviewed_receipts)
        != {
            "e02_vae_reference_receipt",
            "e02_vae_run_receipt",
            "e02_prompt_receipt",
            "e02_prompt_run_receipt",
        }
    ):
        raise OracleActivationV2Error("e02 independent material review differs")
    expected_reviewed = {
        "e02_vae_reference_receipt": (
            reference_path,
            reference_sha,
            reference_seal,
            REFERENCE_RECEIPT_SCHEMA_VERSION,
        ),
        "e02_vae_run_receipt": (
            vae_run_path,
            vae_run_sha,
            vae_run_seal,
            "bernini-oracle-regeneration-activation-v2-vae-authoring-run-r2",
        ),
        "e02_prompt_receipt": (
            prompt_path,
            prompt_sha,
            prompt_seal,
            PROMPT_RECEIPT_SCHEMA_VERSION,
        ),
        "e02_prompt_run_receipt": (
            prompt_run_path,
            prompt_run_sha,
            prompt_run_seal,
            "bernini-oracle-regeneration-activation-v2-prompt-authoring-run-r2",
        ),
    }
    for key, (path, digest, seal, schema) in expected_reviewed.items():
        reviewed = reviewed_receipts.get(key)
        if (
            not isinstance(reviewed, Mapping)
            or reviewed.get("remote_path") != str(path)
            or reviewed.get("sha256") != digest
            or reviewed.get("bytes") != seal.size
            or reviewed.get("mode") != "0444"
            or reviewed.get("nlink") != 1
            or reviewed.get("schema_version") != schema
        ):
            raise OracleActivationV2Error(
                f"e02 independent material review binding differs: {key}"
            )
    return (
        (reference_path, reference_sha, reference_seal),
        (prompt_path, prompt_sha, prompt_seal),
        (vae_run_path, vae_run_sha, vae_run_seal),
        (prompt_run_path, prompt_run_sha, prompt_run_seal),
        (review_path, review_sha, review_seal),
    )


def _validate_instruction(row: Mapping[str, Any], *, case_id: str) -> None:
    caption = row.get("action_caption")
    program = row.get("structured_action_program")
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or hashlib.sha256(caption.encode("utf-8")).hexdigest()
        != row.get("action_caption_sha256")
        or not isinstance(program, Mapping)
        or _canonical_object_sha256(program)
        != row.get("structured_action_program_sha256")
    ):
        raise OracleActivationV2Error(f"{case_id} action instruction differs")


def _validate_case_geometry(row: Mapping[str, Any], *, case_id: str) -> None:
    expected_source = EXPECTED_LATENT_GEOMETRY[case_id]
    expected_gate = EXPECTED_GATE_GEOMETRY[case_id]
    expected_reference = EXPECTED_REFERENCE_GEOMETRY[case_id]
    def exact_tuple(value: Any, expected: tuple[int, ...]) -> bool:
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(type(item) is int for item in value)
            and tuple(value) == expected
        )

    if (
        not exact_tuple(row.get("full_source_latent_geometry"), expected_source)
        or not exact_tuple(row.get("hard_gate_geometry"), expected_gate)
        or not exact_tuple(row.get("reference_latent_geometry"), expected_reference)
        or not exact_tuple(row.get("reference_rgb_indices"), REFERENCE_RGB_INDICES)
    ):
        raise OracleActivationV2Error(f"{case_id} source/gate/reference geometry differs")


@dataclass(frozen=True)
class ActivationCaseAuthorityV2:
    case_id: str
    source_iid: str
    decision: str
    source_video_path: Path
    source_sha256: str
    anchor_video_path: Path
    anchor_sha256: str
    action_caption: str
    action_caption_sha256: str
    structured_action_program: Mapping[str, Any]
    structured_action_program_sha256: str
    seed: int
    full_source_latent_geometry: tuple[int, int, int, int, int]
    hard_gate_geometry: tuple[int, int, int, int, int]
    reference_latent_geometry: tuple[int, int, int, int, int]
    run_arms: tuple[str, ...]
    gate_manifest_path: Optional[Path]
    gate_manifest_sha256: Optional[str]
    review_receipt_path: Optional[Path]
    review_receipt_sha256: Optional[str]
    annotation_authority_root_sha256: Optional[str]
    reference_receipt_path: Optional[Path]
    reference_receipt_sha256: Optional[str]
    prompt_receipt_path: Optional[Path]
    prompt_receipt_sha256: Optional[str]
    vae_run_receipt_path: Optional[Path]
    vae_run_receipt_sha256: Optional[str]
    prompt_run_receipt_path: Optional[Path]
    prompt_run_receipt_sha256: Optional[str]
    material_review_receipt_path: Optional[Path]
    material_review_receipt_sha256: Optional[str]
    kept_frozen_base_path: Optional[Path]
    kept_frozen_base_sha256: Optional[str]
    artifact_seals: Mapping[str, OwnedFileSealV2]


@dataclass(frozen=True)
class ValidatedActivationAuthorityV2:
    packet_path: Path
    packet_sha256: str
    ledger_path: Path
    ledger_sha256: str
    packet_id: str
    cases: Mapping[str, ActivationCaseAuthorityV2]
    dependency_pins: Mapping[str, str]
    annotator: str
    reviewer: str
    issuer: str
    packet_seal: OwnedFileSealV2
    ledger_seal: OwnedFileSealV2
    _validation_token: Any = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class ValidatedManualGateManifestV2:
    path: Path
    file_sha256: str
    case_id: str
    source_sha256: str
    anchor_sha256: str
    action_caption_sha256: str
    structured_action_program_sha256: str
    latent_geometry: tuple[int, int, int, int, int]
    delete_rle: tuple[tuple[tuple[int, int], ...], ...]
    create_rle: tuple[tuple[tuple[int, int], ...], ...]
    contact_rle: tuple[tuple[tuple[int, int], ...], ...]
    mask_sha256: str
    review_receipt_path: Path
    review_receipt_sha256: str
    annotator: str
    reviewer: str
    annotation_authority_root_sha256: str
    annotation_authority_leaf_sha256: str
    annotation_authority_leaf_index: int
    annotation_authority_tree_size: int
    _validation_token: Any = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class _OwnedHardStateChangeGateV2:
    delete: Any
    create: Any
    contact: Any
    support: Any
    preserve: Any
    source_mask_sha256: str
    realized_gate_sha256: str
    delete_count: int
    create_count: int
    contact_count: int
    support_count: int


@dataclass(frozen=True)
class ValidatedManualGateV2:
    case_id: str
    manifest: ValidatedManualGateManifestV2
    owned_gate: Any = field(repr=False, compare=False)
    gate_seal: OwnedFileSealV2
    review_seal: OwnedFileSealV2
    delete_count: int
    create_count: int
    contact_count: int
    support_count: int
    support_fraction_by_phase: tuple[float, ...]
    _validation_token: Any = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class ValidatedReferenceReceiptV2:
    case_id: str
    receipt_seal: OwnedFileSealV2
    source_latent_sha256: str
    reference_latent_sha256: tuple[str, str, str, str]
    source_preprocessed_sha256: str
    reference_raw_rgb_sha256: tuple[str, str, str, str]
    reference_preprocessed_sha256: tuple[str, str, str, str]
    _validation_token: Any = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class ValidatedPromptReceiptV2:
    case_id: str
    receipt_seal: OwnedFileSealV2
    low_action_sha256: str
    high_action_sha256: str
    negative_sha256: str
    rendered_text_sha256: tuple[str, str, str]
    token_ids_sha256: tuple[str, str, str]
    attention_mask_sha256: tuple[str, str, str]
    _validation_token: Any = field(repr=False, compare=False, default=None)


@dataclass(frozen=True)
class NativeLocalExecutionCapabilityV2:
    authority: ValidatedActivationAuthorityV2
    case_id: str
    sample_id: str
    manifest: ValidatedManualGateManifestV2
    owned_gate: Any = field(repr=False, compare=False)
    realized_gate_sha256: str
    source_latent_sha256: str
    source_reference_latent_sha256: tuple[str, str, str, str]
    source_reference_rgb_indices: tuple[int, int, int, int]
    low_action_prompt_sha256: str
    r2v_action_prompt_sha256: str
    negative_prompt_sha256: str
    r2v_action_prompt_embeds: Any = field(repr=False, compare=False)
    authority_packet_path: Path
    authority_packet_sha256: str
    _validation_token: Any = field(repr=False, compare=False, default=None)


def _validate_packet_case_v2(
    row: Any, *, expected_case_id: str
) -> ActivationCaseAuthorityV2:
    expected_keys = _CASE_E02_KEYS if expected_case_id == "e02" else _CASE_E03_KEYS
    if (
        not isinstance(row, Mapping)
        or set(row) != expected_keys
        or row.get("case_id") != expected_case_id
    ):
        raise OracleActivationV2Error("authority case order/identity differs")
    _validate_instruction(row, case_id=expected_case_id)
    _validate_case_geometry(row, case_id=expected_case_id)
    source_path, source_sha, source_seal = _artifact_binding(
        row.get("source_video"),
        label=f"{expected_case_id} source video",
        require_frozen=False,
    )
    anchor_path, anchor_sha, anchor_seal = _artifact_binding(
        row.get("self_generated_anchor"),
        label=f"{expected_case_id} anchor",
        require_frozen=False,
    )
    expected_binding = EXPECTED_CASE_BINDINGS[expected_case_id]
    if (
        row.get("source_iid") != expected_binding["source_iid"]
        or source_sha != expected_binding["source_sha256"]
        or anchor_sha != expected_binding["anchor_sha256"]
        or row.get("action_caption_sha256")
        != expected_binding["action_caption_sha256"]
        or row.get("structured_action_program_sha256")
        != expected_binding["structured_action_program_sha256"]
        or source_sha == anchor_sha
        or (source_seal.device, source_seal.inode)
        == (anchor_seal.device, anchor_seal.inode)
    ):
        raise OracleActivationV2Error(f"{expected_case_id} compiled identity differs")
    if (
        row.get("anchor_used_only_as_review_context") is not True
        or row.get("self_generated_anchor_tensor_used_by_native_expert") is not False
        or row.get("target_video_or_latent_used") is not False
        or row.get("failed_active_used_to_author_gate") is not False
        or row.get("anchor_source_difference_used_to_author_gate") is not False
        or row.get("predicted_soft_gate_used_to_author_gate") is not False
    ):
        raise OracleActivationV2Error(f"{expected_case_id} forbidden authority input differs")
    if expected_case_id == "e02" and (
        row.get("decision") != "ACTIVE_DIAGNOSTIC"
        or tuple(row.get("run_arms", ())) != EXPECTED_ARMS_E02
        or row.get("anchor_terminal_disappearance_observed") is not True
        or row.get("anchor_strict_target_pass") is not False
        or row.get("automatic_model_replacement_authorized") is not False
    ):
        raise OracleActivationV2Error("e02 active/review contract differs")
    if expected_case_id == "e03" and (
        row.get("decision") != "ABSTAIN_KEEP_BASE"
        or tuple(row.get("run_arms", ())) != EXPECTED_ARMS_E03
        or row.get("local_regeneration_selection_authorized") is not False
        or row.get("automatic_model_replacement_authorized") is not False
    ):
        raise OracleActivationV2Error("e03 abstain/keep-base contract differs")
    seed = row.get("seed")
    if type(seed) is not int or seed != EXPECTED_CASE_SEEDS[expected_case_id]:
        raise OracleActivationV2Error(f"{expected_case_id} seed differs")
    gate_path: Optional[Path] = None
    gate_sha: Optional[str] = None
    review_path: Optional[Path] = None
    review_sha: Optional[str] = None
    annotation_root: Optional[str] = None
    if expected_case_id == "e02":
        gate_path, gate_sha, gate_seal = _artifact_binding(
            row.get("manual_gate_manifest"),
            label="e02 manual gate",
            require_frozen=True,
        )
        review_path, review_sha, review_seal = _artifact_binding(
            row.get("independent_review_receipt"),
            label="e02 gate review",
            require_frozen=True,
        )
        if gate_path == review_path or (
            gate_seal.device,
            gate_seal.inode,
        ) == (review_seal.device, review_seal.inode):
            raise OracleActivationV2Error("e02 gate and review receipt alias")
        annotation_root = _require_sha256(
            row.get("annotation_authority_root_sha256"),
            label="e02 annotation authority root",
        )
    elif row.get("manual_gate_manifest") is not None or row.get(
        "independent_review_receipt"
    ) is not None or row.get("annotation_authority_root_sha256") is not None:
        raise OracleActivationV2Error("e03 abstain packet must not carry an active gate")
    reference_path: Optional[Path] = None
    reference_sha: Optional[str] = None
    reference_seal: Optional[OwnedFileSealV2] = None
    prompt_path: Optional[Path] = None
    prompt_sha: Optional[str] = None
    prompt_seal: Optional[OwnedFileSealV2] = None
    vae_run_path: Optional[Path] = None
    vae_run_sha: Optional[str] = None
    vae_run_seal: Optional[OwnedFileSealV2] = None
    prompt_run_path: Optional[Path] = None
    prompt_run_sha: Optional[str] = None
    prompt_run_seal: Optional[OwnedFileSealV2] = None
    material_review_path: Optional[Path] = None
    material_review_sha: Optional[str] = None
    material_review_seal: Optional[OwnedFileSealV2] = None
    kept_base_path: Optional[Path] = None
    kept_base_sha: Optional[str] = None
    kept_base_seal: Optional[OwnedFileSealV2] = None
    if expected_case_id == "e02":
        (
            (reference_path, reference_sha, reference_seal),
            (prompt_path, prompt_sha, prompt_seal),
            (vae_run_path, vae_run_sha, vae_run_seal),
            (prompt_run_path, prompt_run_sha, prompt_run_seal),
            (material_review_path, material_review_sha, material_review_seal),
        ) = _validate_e02_materialization_graph_v2(
            case_row=row,
            source_iid=str(row["source_iid"]),
            source_sha256=source_sha,
            action_caption_sha256=str(row["action_caption_sha256"]),
        )
        material_seals = (
            reference_seal,
            prompt_seal,
            vae_run_seal,
            prompt_run_seal,
            material_review_seal,
        )
        gate_seals = (gate_seal, review_seal)
        if len(
            {
                (seal.device, seal.inode)
                for seal in (*material_seals, *gate_seals)
                if seal is not None
            }
        ) != 7:
            raise OracleActivationV2Error("e02 reviewed authority artifacts alias")
    else:
        if row.get("vae_reference_receipt") is not None or row.get(
            "prompt_receipt"
        ) is not None:
            raise OracleActivationV2Error(
                "e03 policy-only packet must not carry material receipts"
            )
        kept_base_path, kept_base_sha, kept_base_seal = _artifact_binding(
            row.get("kept_frozen_base"),
            label="e03 kept frozen base",
            require_frozen=False,
        )
        if (
            kept_base_path != EXPECTED_E03_FROZEN_BASE_PATH
            or kept_base_sha != EXPECTED_E03_FROZEN_BASE_SHA256
        ):
            raise OracleActivationV2Error("e03 kept frozen base differs")
    return ActivationCaseAuthorityV2(
        case_id=expected_case_id,
        source_iid=str(row["source_iid"]),
        decision=str(row["decision"]),
        source_video_path=source_path,
        source_sha256=source_sha,
        anchor_video_path=anchor_path,
        anchor_sha256=anchor_sha,
        action_caption=str(row["action_caption"]),
        action_caption_sha256=str(row["action_caption_sha256"]),
        structured_action_program=dict(row["structured_action_program"]),
        structured_action_program_sha256=str(
            row["structured_action_program_sha256"]
        ),
        seed=seed,
        full_source_latent_geometry=EXPECTED_LATENT_GEOMETRY[expected_case_id],
        hard_gate_geometry=EXPECTED_GATE_GEOMETRY[expected_case_id],
        reference_latent_geometry=EXPECTED_REFERENCE_GEOMETRY[expected_case_id],
        run_arms=tuple(row["run_arms"]),
        gate_manifest_path=gate_path,
        gate_manifest_sha256=gate_sha,
        review_receipt_path=review_path,
        review_receipt_sha256=review_sha,
        annotation_authority_root_sha256=annotation_root,
        reference_receipt_path=reference_path,
        reference_receipt_sha256=reference_sha,
        prompt_receipt_path=prompt_path,
        prompt_receipt_sha256=prompt_sha,
        vae_run_receipt_path=vae_run_path,
        vae_run_receipt_sha256=vae_run_sha,
        prompt_run_receipt_path=prompt_run_path,
        prompt_run_receipt_sha256=prompt_run_sha,
        material_review_receipt_path=material_review_path,
        material_review_receipt_sha256=material_review_sha,
        kept_frozen_base_path=kept_base_path,
        kept_frozen_base_sha256=kept_base_sha,
        artifact_seals={
            "source_video": source_seal,
            "self_generated_anchor": anchor_seal,
            **(
                {
                    "manual_gate_manifest": gate_seal,
                    "independent_review_receipt": review_seal,
                }
                if expected_case_id == "e02"
                else {}
            ),
            **(
                {
                    "vae_reference_receipt": reference_seal,
                    "prompt_receipt": prompt_seal,
                    "vae_authoring_run_receipt": vae_run_seal,
                    "prompt_authoring_run_receipt": prompt_run_seal,
                    "materialization_review_receipt": material_review_seal,
                }
                if expected_case_id == "e02"
                else {"kept_frozen_base": kept_base_seal}
            ),
        },
    )


def load_compiled_activation_authority_v2(
    packet_path: Path, ledger_receipt_path: Path
) -> ValidatedActivationAuthorityV2:
    """Load only the two exact files pinned inside this source revision."""

    if not compiled_activation_available_v2():
        raise OracleActivationV2Error(
            "activation trust anchors are not compiled in this moving candidate"
        )
    dependency_pins = verify_frozen_dependency_pins_v2()
    packet = _plain_absolute_file(str(packet_path), label="authority packet")
    ledger = _plain_absolute_file(str(ledger_receipt_path), label="external ledger")
    packet_seal, payload = _seal_json_v2(packet, label="activation authority packet")
    ledger_seal, ledger_payload = _seal_json_v2(
        ledger, label="external ledger receipt"
    )
    packet_sha = packet_seal.sha256
    ledger_sha = ledger_seal.sha256
    if (
        packet_sha != COMPILED_AUTHORITY_PACKET_SHA256
        or ledger_sha != COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256
    ):
        raise OracleActivationV2Error("compiled authority or ledger bytes differ")
    if payload.get("schema_version") == AUTHORING_TEMPLATE_SCHEMA_VERSION:
        raise OracleActivationV2Error("authoring template can never become authority")
    execution = payload.get("execution_contract")
    safety = payload.get("safety_contract")
    cases = payload.get("cases")
    if (
        set(payload) != _PACKET_KEYS
        or payload.get("schema_version") != AUTHORITY_PACKET_SCHEMA_VERSION
        or payload.get("status")
        != "INDEPENDENT_MODEL_REVIEWED_DIAGNOSTIC_EXPERIMENTAL_PACKET"
        or not isinstance(payload.get("packet_id"), str)
        or _PACKET_ID.fullmatch(str(payload.get("packet_id"))) is None
        or not isinstance(cases, list)
        or len(cases) != 2
        or not isinstance(execution, Mapping)
        or set(execution) != _EXECUTION_KEYS
        or execution.get("native_only") is not True
        or execution.get("flowedit_enabled") is not False
        or execution.get("connected_runner_enabled") is not False
        or execution.get("learned_gate_enabled") is not False
        or execution.get("world_size") != 4
        or execution.get("sequence_parallel_size") != 4
        or execution.get("one_node") is not True
        or execution.get("same_seed_and_official_gaussian") is not True
        or execution.get("candidate_count_per_arm") != 1
        or execution.get("source_reference_r2v4_regeneration_expert") is not True
        or execution.get("self_generated_anchor_tensor_used_by_native_expert")
        is not False
        or execution.get("anchor_reference_or_quotient_arm_deferred") is not True
        or execution.get("global_source_reference_r2v4_upper_bound_arm_deferred")
        is not True
        or not isinstance(safety, Mapping)
        or set(safety) != _SAFETY_KEYS
        or safety.get("training_authorized") is not False
        or safety.get("optimizer_authorized") is not False
        or safety.get("automatic_model_replacement_authorized") is not False
        or safety.get("background_cosine_selection_authorized") is not False
        or safety.get("target_video_or_latent_used") is not False
    ):
        raise OracleActivationV2Error("activation authority packet contract differs")
    validated_cases = {
        case_id: _validate_packet_case_v2(cases[index], expected_case_id=case_id)
        for index, case_id in enumerate(ALLOWED_CASES)
    }
    annotator = ledger_payload.get("annotator")
    reviewer = ledger_payload.get("reviewer")
    issuer = ledger_payload.get("issuer")
    if (
        set(ledger_payload) != _LEDGER_KEYS
        or ledger_payload.get("schema_version") != LEDGER_RECEIPT_SCHEMA_VERSION
        or ledger_payload.get("authority_packet_sha256") != packet_sha
        or ledger_payload.get("packet_id") != payload.get("packet_id")
        or not isinstance(annotator, str)
        or _STABLE_ID.fullmatch(annotator) is None
        or annotator != annotator.strip().lower()
        or not isinstance(reviewer, str)
        or _STABLE_ID.fullmatch(reviewer) is None
        or reviewer != reviewer.strip().lower()
        or reviewer == annotator
        or not isinstance(issuer, str)
        or _STABLE_ID.fullmatch(issuer) is None
        or issuer != issuer.strip().lower()
        or issuer in (annotator, reviewer)
        or ledger_payload.get("annotator_kind") != "AI_AGENT"
        or ledger_payload.get("reviewer_kind") != "AI_AGENT"
        or ledger_payload.get("issuer_kind") != "AI_AGENT"
        or ledger_payload.get("trust_root_kind")
        != "COMPILED_EXACT_PACKET_AND_LEDGER_SHA256_CODE_REVIEW"
        or ledger_payload.get("accepted") is not True
        or ledger_payload.get("e02_exact_gate_reviewed") is not True
        or ledger_payload.get("e03_abstain_keep_base_reviewed") is not True
        or ledger_payload.get("authority_packet_contains_no_activation_code_hashes")
        is not True
        or ledger_payload.get("private_signing_material_present") is not False
        or ledger_payload.get("cryptographic_signature_claimed") is not False
        or ledger_payload.get("diagnostic_experimental_canary_only") is not True
        or ledger_payload.get("formal_authority") is not False
        or ledger_payload.get("training_authority") is not False
    ):
        raise OracleActivationV2Error("external ledger authority differs")
    return ValidatedActivationAuthorityV2(
        packet_path=packet,
        packet_sha256=packet_sha,
        ledger_path=ledger,
        ledger_sha256=ledger_sha,
        packet_id=str(payload["packet_id"]),
        cases=validated_cases,
        dependency_pins=dependency_pins,
        annotator=annotator,
        reviewer=reviewer,
        issuer=issuer,
        packet_seal=packet_seal,
        ledger_seal=ledger_seal,
        _validation_token=_AUTHORITY_TOKEN,
    )


def revalidate_compiled_activation_authority_v2(
    authority: ValidatedActivationAuthorityV2,
) -> None:
    """Reparse and reseal the complete authority graph from current bytes.

    ``load_compiled_activation_authority_v2`` uses a single owned descriptor for
    each JSON/hash decision and recursively seals every case artifact.  Loading
    a fresh capability here therefore checks packet, ledger, dependencies,
    source/anchor media, gate/review, VAE receipt, and prompt receipt rather than
    trusting the dataclass snapshot supplied by a caller.
    """

    if (
        not isinstance(authority, ValidatedActivationAuthorityV2)
        or authority._validation_token is not _AUTHORITY_TOKEN
        or not compiled_activation_available_v2()
    ):
        raise OracleActivationV2Error("activation authority capability changed")
    fresh = load_compiled_activation_authority_v2(
        authority.packet_path, authority.ledger_path
    )
    if fresh != authority:
        raise OracleActivationV2Error("activation authority graph changed")


def _bound_authority_json_v2(
    case: ActivationCaseAuthorityV2,
    *,
    artifact_key: str,
    path: Path,
    expected_sha256: str,
    label: str,
) -> Mapping[str, Any]:
    seal, payload = _seal_json_v2(path, label=label)
    expected_seal = case.artifact_seals.get(artifact_key)
    if seal.sha256 != expected_sha256 or seal != expected_seal:
        raise OracleActivationV2Error(f"{label} authority seal changed")
    return payload


def _one_spatiotemporal_component_v2(value: Any) -> bool:
    """Check one 26-neighbour component without NumPy or floating masks."""

    coordinates = {
        tuple(int(item) for item in row)
        for row in value[0, 0].nonzero(as_tuple=False).tolist()
    }
    if not coordinates:
        return False
    pending = [next(iter(coordinates))]
    visited = {pending[0]}
    phases, height, width = (int(item) for item in value.shape[-3:])
    while pending:
        phase, y_value, x_value = pending.pop()
        for dt in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dt == dy == dx == 0:
                        continue
                    candidate = (phase + dt, y_value + dy, x_value + dx)
                    if (
                        0 <= candidate[0] < phases
                        and 0 <= candidate[1] < height
                        and 0 <= candidate[2] < width
                        and candidate in coordinates
                        and candidate not in visited
                    ):
                        visited.add(candidate)
                        pending.append(candidate)
    return len(visited) == len(coordinates)


def _one_spatial_component_4_v2(value: Any) -> bool:
    coordinates = {
        tuple(int(item) for item in row)
        for row in value.nonzero(as_tuple=False).tolist()
    }
    if not coordinates:
        return True
    pending = [next(iter(coordinates))]
    visited = {pending[0]}
    height, width = (int(item) for item in value.shape)
    while pending:
        y_value, x_value = pending.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (y_value + dy, x_value + dx)
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and candidate in coordinates
                and candidate not in visited
            ):
                visited.add(candidate)
                pending.append(candidate)
    return len(visited) == len(coordinates)


def _has_large_rectangle_shortcut_v2(value: Any) -> bool:
    for phase in range(1, int(value.shape[2])):
        coordinates = value[0, 0, phase].nonzero(as_tuple=False)
        count = int(coordinates.shape[0])
        if count < 12:
            continue
        y_min = int(coordinates[:, 0].min().item())
        y_max = int(coordinates[:, 0].max().item())
        x_min = int(coordinates[:, 1].min().item())
        x_max = int(coordinates[:, 1].max().item())
        bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        if bbox_area >= 16 and float(count) / float(bbox_area) >= 0.80:
            return True
    return False


def _manual_gate_payload_v2(
    *,
    geometry: Sequence[int],
    delete_rle: Any,
    create_rle: Any,
    contact_rle: Any,
) -> Mapping[str, Any]:
    return {
        "latent_geometry": [int(item) for item in geometry],
        "flattening": "per_phase_row_major_yx",
        "delete_rle": delete_rle,
        "create_rle": create_rle,
        "contact_rle": contact_rle,
        "dtype": "bool",
        "support_definition": "G=D_or_C_or_K",
    }


def _manual_gate_leaf_sha256_v2(
    *,
    case: ActivationCaseAuthorityV2,
    mask_sha256: str,
    annotator: str,
    reviewer: str,
) -> str:
    payload = {
        "schema_version": MANUAL_GATE_LEAF_SCHEMA_VERSION,
        "case_id": case.case_id,
        "source_sha256": case.source_sha256,
        "anchor_sha256": case.anchor_sha256,
        "action_caption_sha256": case.action_caption_sha256,
        "structured_action_program_sha256": case.structured_action_program_sha256,
        "mask_sha256": mask_sha256,
        "annotator": annotator,
        "reviewer": reviewer,
    }
    return hashlib.sha256(
        b"\x00" + safe_core.canonical_json_bytes_v1(payload)
    ).hexdigest()


def _realized_gate_sha256_v2(gate: _OwnedHardStateChangeGateV2) -> str:
    digest = hashlib.sha256()
    digest.update(
        safe_core.canonical_json_bytes_v1(
            {
                "schema_version": "bernini-realized-hard-dck-gate-tensor-v2",
                "source_mask_sha256": gate.source_mask_sha256,
                "geometry": [int(item) for item in gate.support.shape],
                "dtype": "bool",
                "tensor_order": ["delete_D", "create_C", "contact_K", "support_G", "preserve"],
                "support_definition": "G=D_or_C_or_K",
            }
        )
    )
    for label, value in (
        ("delete_D", gate.delete),
        ("create_C", gate.create),
        ("contact_K", gate.contact),
        ("support_G", gate.support),
        ("preserve", gate.preserve),
    ):
        raw = safe_core._tensor_raw_bytes_v1(value)
        digest.update(label.encode("ascii") + b"\x00")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_owned_gate_v2(
    gate: _OwnedHardStateChangeGateV2,
    *,
    expected_geometry: tuple[int, int, int, int, int],
) -> None:
    import torch

    if not isinstance(gate, _OwnedHardStateChangeGateV2):
        raise OracleActivationV2Error("owned D/C/K gate type differs")
    tensors = (gate.delete, gate.create, gate.contact, gate.support, gate.preserve)
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(int(item) for item in value.shape) != expected_geometry
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        for value in tensors
    ) or any(value.device != gate.support.device for value in tensors):
        raise OracleActivationV2Error("owned D/C/K gate tensor contract differs")
    union = torch.logical_or(torch.logical_or(gate.delete, gate.create), gate.contact)
    if (
        not torch.equal(gate.support, union)
        or not torch.equal(gate.preserve, torch.logical_not(gate.support))
        or bool(gate.support[:, :, 0].any().item())
        or bool(torch.logical_and(gate.delete, gate.create).any().item())
        or (gate.delete_count, gate.create_count, gate.contact_count, gate.support_count)
        != (
            int(gate.delete.sum().item()),
            int(gate.create.sum().item()),
            int(gate.contact.sum().item()),
            int(gate.support.sum().item()),
        )
        or _realized_gate_sha256_v2(gate) != gate.realized_gate_sha256
    ):
        raise OracleActivationV2Error("owned D/C/K gate algebra or digest differs")


def _materialize_owned_gate_v2(
    manifest: ValidatedManualGateManifestV2,
    *,
    device: Any = "cpu",
) -> _OwnedHardStateChangeGateV2:
    import torch

    if (
        not isinstance(manifest, ValidatedManualGateManifestV2)
        or manifest._validation_token is not _GATE_TOKEN
    ):
        raise OracleActivationV2Error("manual gate manifest capability differs")
    _, _, _, height, width = manifest.latent_geometry
    delete = safe_core._decode_rle_tensor(
        manifest.delete_rle, height=height, width=width
    ).to(device=device).clone().detach().contiguous()
    create = safe_core._decode_rle_tensor(
        manifest.create_rle, height=height, width=width
    ).to(device=device).clone().detach().contiguous()
    contact = safe_core._decode_rle_tensor(
        manifest.contact_rle, height=height, width=width
    ).to(device=device).clone().detach().contiguous()
    support = torch.logical_or(torch.logical_or(delete, create), contact).contiguous()
    preserve = torch.logical_not(support).contiguous()
    provisional = _OwnedHardStateChangeGateV2(
        delete=delete,
        create=create,
        contact=contact,
        support=support,
        preserve=preserve,
        source_mask_sha256=manifest.mask_sha256,
        realized_gate_sha256="0" * 64,
        delete_count=int(delete.sum().item()),
        create_count=int(create.sum().item()),
        contact_count=int(contact.sum().item()),
        support_count=int(support.sum().item()),
    )
    gate = _OwnedHardStateChangeGateV2(
        **{
            **provisional.__dict__,
            "realized_gate_sha256": _realized_gate_sha256_v2(provisional),
        }
    )
    _validate_owned_gate_v2(gate, expected_geometry=manifest.latent_geometry)
    return gate


def validate_manual_gate_v2(
    authority: ValidatedActivationAuthorityV2,
    *,
    case_id: str = "e02",
) -> ValidatedManualGateV2:
    """Validate the independently model-reviewed exact-bool D/C/K gate."""

    import torch

    revalidate_compiled_activation_authority_v2(authority)
    if case_id != "e02" or case_id not in authority.cases:
        raise OracleActivationV2Error("only e02 has an active manual gate")
    case = authority.cases[case_id]
    if (
        case.gate_manifest_path is None
        or case.gate_manifest_sha256 is None
        or case.review_receipt_path is None
        or case.review_receipt_sha256 is None
        or case.annotation_authority_root_sha256 is None
    ):
        raise OracleActivationV2Error("e02 active gate authority is incomplete")
    gate_raw = _bound_authority_json_v2(
        case,
        artifact_key="manual_gate_manifest",
        path=case.gate_manifest_path,
        expected_sha256=case.gate_manifest_sha256,
        label="e02 manual gate",
    )
    review_raw = _bound_authority_json_v2(
        case,
        artifact_key="independent_review_receipt",
        path=case.review_receipt_path,
        expected_sha256=case.review_receipt_sha256,
        label="e02 independent gate review",
    )
    authority_row = gate_raw.get("authority")
    qualification = gate_raw.get("qualification")
    annotation = gate_raw.get("annotation_authority")
    forbidden = (
        authority_row.get("forbidden_inputs_absent")
        if isinstance(authority_row, Mapping)
        else None
    )
    if (
        set(gate_raw) != _GATE_KEYS
        or gate_raw.get("schema_version") != MANUAL_GATE_SCHEMA_VERSION
        or gate_raw.get("case_id") != case_id
        or gate_raw.get("source_sha256") != case.source_sha256
        or gate_raw.get("anchor_sha256") != case.anchor_sha256
        or gate_raw.get("action_caption_sha256") != case.action_caption_sha256
        or gate_raw.get("structured_action_program_sha256")
        != case.structured_action_program_sha256
        or gate_raw.get("latent_geometry") != list(case.hard_gate_geometry)
        or gate_raw.get("flattening") != "per_phase_row_major_yx"
        or gate_raw.get("dtype") != "bool"
        or gate_raw.get("hard_support") is not True
        or gate_raw.get("phase_zero_empty") is not True
        or not isinstance(gate_raw.get("typed_semantics"), Mapping)
        or set(gate_raw["typed_semantics"]) != _GATE_TYPED_SEMANTICS_KEYS
        or gate_raw["typed_semantics"].get("delete_D")
        != "obsolete_source_state_occupancy_to_delete"
        or gate_raw["typed_semantics"].get("create_C")
        != "new_actor_object_state_occupancy_to_create"
        or gate_raw["typed_semantics"].get("contact_ownership_K")
        != "contact_and_ownership_transition_permission_corridor"
        or gate_raw["typed_semantics"].get("execution_support_G")
        != "exact_boolean_union_D_or_C_or_K"
        or gate_raw["typed_semantics"].get("coordinate_system")
        != "source_latent_phase_y_x"
        or gate_raw["typed_semantics"].get("expected_nonempty_phase_windows")
        != {
            "delete_D": [1, 20],
            "create_C": [5, 20],
            "contact_ownership_K": [4, 20],
            "execution_support_G": [1, 20],
        }
        or not isinstance(authority_row, Mapping)
        or set(authority_row) != _GATE_AUTHORITY_KEYS
        or authority_row.get("role")
        != "source_only_model_proposal_diagnostic_intervention_only"
        or authority_row.get("training_target_authorized") is not False
        or authority_row.get("action_representation_claimed") is not False
        or not isinstance(forbidden, Mapping)
        or set(forbidden) != _GATE_FORBIDDEN_KEYS
        or any(forbidden.get(key) is not True for key in _GATE_FORBIDDEN_KEYS)
        or not isinstance(qualification, Mapping)
        or set(qualification) != _GATE_QUALIFICATION_KEYS
        or qualification.get("status")
        != "independent_model_reviewed_diagnostic_exact_gate"
        or qualification.get("annotator") != authority.annotator
        or qualification.get("reviewer") != authority.reviewer
        or qualification.get("author_kind") != "AI_AGENT"
        or qualification.get("reviewer_kind") != "AI_AGENT"
        or qualification.get("review_receipt_path")
        != str(case.review_receipt_path)
        or not isinstance(annotation, Mapping)
        or set(annotation) != _GATE_ANNOTATION_KEYS
    ):
        raise OracleActivationV2Error("e02 manual gate contract differs")
    try:
        geometry = safe_core._validate_geometry(gate_raw.get("latent_geometry"))
        delete_rle = safe_core._validate_rle(
            gate_raw.get("delete_rle"),
            label="delete_rle",
            height=geometry[-2],
            width=geometry[-1],
        )
        create_rle = safe_core._validate_rle(
            gate_raw.get("create_rle"),
            label="create_rle",
            height=geometry[-2],
            width=geometry[-1],
        )
        contact_rle = safe_core._validate_rle(
            gate_raw.get("contact_rle"),
            label="contact_rle",
            height=geometry[-2],
            width=geometry[-1],
        )
    except Exception as error:
        raise OracleActivationV2Error(str(error)) from error
    mask_sha256 = _require_sha256(
        gate_raw.get("mask_sha256"), label="e02 manual mask payload"
    )
    observed_mask_sha256 = hashlib.sha256(
        safe_core.canonical_json_bytes_v1(
            _manual_gate_payload_v2(
                geometry=geometry,
                delete_rle=gate_raw["delete_rle"],
                create_rle=gate_raw["create_rle"],
                contact_rle=gate_raw["contact_rle"],
            )
        )
    ).hexdigest()
    if mask_sha256 != observed_mask_sha256:
        raise OracleActivationV2Error("e02 manual mask payload differs")
    leaf_sha256 = _manual_gate_leaf_sha256_v2(
        case=case,
        mask_sha256=mask_sha256,
        annotator=authority.annotator,
        reviewer=authority.reviewer,
    )
    try:
        leaf_index, tree_size = safe_core._verify_annotation_inclusion_v1(
            annotation,
            expected_root_sha256=case.annotation_authority_root_sha256,
            expected_leaf_sha256=leaf_sha256,
        )
    except Exception as error:
        raise OracleActivationV2Error(str(error)) from error
    required_review_true = {
        "accepted",
        "phase_zero_source_authority_checked",
        "source_coordinate_authoring_checked",
        "delete_create_contact_semantics_checked",
        "D_C_disjoint_checked",
        "K_preserved_as_independent_channel_checked",
        "G_exact_union_D_C_K_checked",
        "channel_active_windows_checked",
        "no_large_rectangle_shortcut_checked",
        "single_actor_object_component_checked",
        "duplicate_actor_or_object_rejected",
        "terminal_hold_semantics_checked",
        "anchor_terminal_disappearance_observed",
        "anchor_used_only_as_review_context",
        "source_only_model_proposal",
        "independent_model_review",
    }
    required_review_false = {
        "anchor_strict_target_pass",
        "failed_active_used_to_author_mask",
        "anchor_difference_used_to_author_mask",
        "predicted_soft_gate_used_to_author_mask",
        "target_video_or_latent_used_to_author_mask",
        "self_generated_anchor_tensor_used_to_author_mask",
    }
    if (
        set(review_raw) != _GATE_REVIEW_KEYS
        or review_raw.get("schema_version") != MANUAL_GATE_REVIEW_SCHEMA_VERSION
        or review_raw.get("case_id") != case_id
        or review_raw.get("source_sha256") != case.source_sha256
        or review_raw.get("anchor_sha256") != case.anchor_sha256
        or review_raw.get("action_caption_sha256") != case.action_caption_sha256
        or review_raw.get("structured_action_program_sha256")
        != case.structured_action_program_sha256
        or review_raw.get("gate_manifest_sha256") != case.gate_manifest_sha256
        or review_raw.get("mask_sha256") != mask_sha256
        or review_raw.get("annotation_authority_root_sha256")
        != case.annotation_authority_root_sha256
        or review_raw.get("annotation_authority_leaf_sha256") != leaf_sha256
        or review_raw.get("annotator") != authority.annotator
        or review_raw.get("reviewer") != authority.reviewer
        or review_raw.get("author_kind") != "AI_AGENT"
        or review_raw.get("reviewer_kind") != "AI_AGENT"
        or any(review_raw.get(key) is not True for key in required_review_true)
        or any(review_raw.get(key) is not False for key in required_review_false)
    ):
        raise OracleActivationV2Error("e02 independent manual-gate review differs")
    manifest = ValidatedManualGateManifestV2(
        path=case.gate_manifest_path,
        file_sha256=case.gate_manifest_sha256,
        case_id=case_id,
        source_sha256=case.source_sha256,
        anchor_sha256=case.anchor_sha256,
        action_caption_sha256=case.action_caption_sha256,
        structured_action_program_sha256=case.structured_action_program_sha256,
        latent_geometry=geometry,
        delete_rle=delete_rle,
        create_rle=create_rle,
        contact_rle=contact_rle,
        mask_sha256=mask_sha256,
        review_receipt_path=case.review_receipt_path,
        review_receipt_sha256=case.review_receipt_sha256,
        annotator=authority.annotator,
        reviewer=authority.reviewer,
        annotation_authority_root_sha256=case.annotation_authority_root_sha256,
        annotation_authority_leaf_sha256=leaf_sha256,
        annotation_authority_leaf_index=leaf_index,
        annotation_authority_tree_size=tree_size,
        _validation_token=_GATE_TOKEN,
    )
    owned_gate = _materialize_owned_gate_v2(manifest, device="cpu")
    k_unique = torch.logical_and(
        owned_gate.contact,
        torch.logical_not(torch.logical_or(owned_gate.delete, owned_gate.create)),
    ).contiguous()
    channels = (
        owned_gate.delete,
        owned_gate.create,
        owned_gate.contact,
        owned_gate.support,
        k_unique,
    )
    expected_active_phases = (
        set(range(1, 21)),
        set(range(5, 21)),
        set(range(4, 21)),
        set(range(1, 21)),
    )
    observed_active_phases = tuple(
        {
            phase
            for phase in range(21)
            if bool(value[:, :, phase].any().item())
        }
        for value in channels[:4]
    )
    k_active_phases = expected_active_phases[2]
    if (
        observed_active_phases != expected_active_phases
        or bool(owned_gate.contact[:, :, 0:4].any().item())
        or any(not _one_spatiotemporal_component_v2(value) for value in channels)
        or any(
            not _one_spatial_component_4_v2(value[0, 0, phase])
            for value, active_phases in zip(channels[:4], expected_active_phases)
            for phase in active_phases
        )
        or any(
            not _one_spatial_component_4_v2(k_unique[0, 0, phase])
            for phase in k_active_phases
        )
        or any(
            int(k_unique[:, :, phase].sum().item()) < 4
            or float(k_unique[:, :, phase].sum().item())
            / float(max(1, int(owned_gate.contact[:, :, phase].sum().item())))
            < 0.10
            for phase in k_active_phases
        )
        or any(_has_large_rectangle_shortcut_v2(value) for value in channels[:-1])
    ):
        raise OracleActivationV2Error("e02 D/C/K topology or typed phase semantics differ")
    area = geometry[-2] * geometry[-1]
    fractions = tuple(
        float(owned_gate.support[:, :, phase].sum().item()) / float(area)
        for phase in range(21)
    )
    if fractions[0] != 0.0 or any(value <= 0.0 or value > 0.30 for value in fractions[1:]):
        raise OracleActivationV2Error("e02 per-phase hard-support mass differs")
    return ValidatedManualGateV2(
        case_id=case_id,
        manifest=manifest,
        owned_gate=owned_gate,
        gate_seal=case.artifact_seals["manual_gate_manifest"],
        review_seal=case.artifact_seals["independent_review_receipt"],
        delete_count=int(owned_gate.delete.sum().item()),
        create_count=int(owned_gate.create.sum().item()),
        contact_count=int(owned_gate.contact.sum().item()),
        support_count=int(owned_gate.support.sum().item()),
        support_fraction_by_phase=fractions,
        _validation_token=_GATE_TOKEN,
    )


def _require_exact_int_v2(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OracleActivationV2Error(f"{label} must be an exact integer")
    return value


def _require_sha_list_v2(value: Any, *, label: str, count: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise OracleActivationV2Error(f"{label} count differs")
    return tuple(
        _require_sha256(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_json_tensor_identity_v2(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != _TENSOR_IDENTITY_KEYS
        or not isinstance(value.get("shape"), list)
        or any(type(item) is not int for item in value["shape"])
        or tuple(value["shape"]) != expected_shape
        or value.get("dtype") != expected_dtype
    ):
        raise OracleActivationV2Error(f"{label} tensor identity differs")
    return _require_sha256(value.get("content_sha256"), label=label)


def _validate_contract_file_v2(
    value: Any,
    expected_sha256: Any,
    *,
    label: str,
) -> OwnedFileSealV2:
    path = _plain_absolute_file(value, label=label)
    wanted = _require_sha256(expected_sha256, label=label)
    seal, _ = _seal_plain_file_v2(
        path, label=label, retain_bytes=False, require_frozen=False
    )
    if seal.sha256 != wanted:
        raise OracleActivationV2Error(f"{label} bytes differ")
    return seal


def _validate_rank_digest_rows_v2(
    value: Any,
    *,
    expected: Sequence[str],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise OracleActivationV2Error(f"{label} WORLD4 rows differ")
    expected_row = list(expected)
    for rank, row in enumerate(value):
        if row != expected_row:
            raise OracleActivationV2Error(f"{label} rank{rank} digest differs")


def validate_reference_receipt_v2(
    authority: ValidatedActivationAuthorityV2,
    *,
    case_id: str,
    source_video_latent: Any,
    source_reference_latents: Sequence[Any],
) -> ValidatedReferenceReceiptV2:
    """Bind live source/ref tensors to the VAE-only WORLD4 receipt."""

    import torch

    revalidate_compiled_activation_authority_v2(authority)
    if case_id != "e02" or case_id not in authority.cases:
        raise OracleActivationV2Error(
            "reference receipts are authorized only for active e02"
        )
    case = authority.cases[case_id]
    if case.reference_receipt_path is None or case.reference_receipt_sha256 is None:
        raise OracleActivationV2Error("e02 reference receipt is absent")
    raw = _bound_authority_json_v2(
        case,
        artifact_key="vae_reference_receipt",
        path=case.reference_receipt_path,
        expected_sha256=case.reference_receipt_sha256,
        label=f"{case_id} VAE reference receipt",
    )
    preprocess = raw.get("preprocess_contract")
    vae_contract = raw.get("vae_contract")
    rank_receipt = raw.get("rank_world_receipt")
    if (
        set(raw) != _REFERENCE_RECEIPT_KEYS
        or raw.get("schema_version") != REFERENCE_RECEIPT_SCHEMA_VERSION
        or raw.get("case_id") != case_id
        or raw.get("source_iid") != case.source_iid
        or raw.get("source_video_sha256") != case.source_sha256
        or raw.get("source_frame_count") != 81
        or raw.get("source_fps_numerator") != 25
        or raw.get("source_fps_denominator") != 1
        or raw.get("reference_rgb_indices") != list(REFERENCE_RGB_INDICES)
        or raw.get("references_encoded_as_four_independent_rgb_frames") is not True
        or raw.get("references_not_sliced_from_full_source_latent") is not True
        or raw.get("source_reference_storage_alias_rejected") is not True
        or raw.get("reference_content_duplicates_rejected") is not True
        or raw.get("target_video_or_latent_used") is not False
        or raw.get("self_generated_anchor_tensor_used") is not False
        or raw.get("materialization_checks_passed") is not True
        or not isinstance(preprocess, Mapping)
        or set(preprocess) != _PREPROCESS_CONTRACT_KEYS
        or preprocess.get("frame_decode_backend")
        != "decord_cpu0_num_threads1_private_source_snapshot"
        or preprocess.get("rgb_dtype") != "uint8"
        or preprocess.get("rgb_channel_order") != "RGB"
        or preprocess.get("resize_policy")
        != "torchvision_bicubic_antialias_true_source_aspect_bucket"
        or preprocess.get("normalization") != "uint8_div255_mul2_sub1_float32"
        or not isinstance(vae_contract, Mapping)
        or set(vae_contract) != _VAE_CONTRACT_KEYS
        or vae_contract.get("encode_function") != "bernini.pipeline._vae_encode"
        or vae_contract.get("encode_dtype") != "torch.float32"
        or vae_contract.get("latent_coordinate")
        != "official_bernini_vae_encode_output"
        or any(
            not isinstance(vae_contract.get(key), str)
            or not vae_contract.get(key)
            for key in (
                "diffusers_version",
                "torch_version",
                "python_version",
                "rocm_version",
            )
        )
        or not isinstance(rank_receipt, Mapping)
        or set(rank_receipt) != _RANK_WORLD_KEYS
        or rank_receipt.get("world_size") != 4
        or rank_receipt.get("sequence_parallel_size") != 4
        or rank_receipt.get("rank0_only_vae_encode") is not True
        or rank_receipt.get("all_rank_vae_load_roles")
        != [
            {"rank": rank, "vae_loaded": rank == 0}
            for rank in range(4)
        ]
        or rank_receipt.get("broadcast_exact") is not True
    ):
        raise OracleActivationV2Error(f"{case_id} VAE reference receipt contract differs")
    _validate_contract_file_v2(
        raw.get("materializer_code_path"),
        raw.get("materializer_code_sha256"),
        label=f"{case_id} VAE materializer code",
    )
    _validate_contract_file_v2(
        preprocess.get("frame_decode_code_path"),
        preprocess.get("frame_decode_code_sha256"),
        label=f"{case_id} frame decode code",
    )
    _validate_contract_file_v2(
        preprocess.get("source_prepare_code_path"),
        preprocess.get("source_prepare_code_sha256"),
        label=f"{case_id} source prepare code",
    )
    for prefix in (
        "checkpoint_content_manifest",
        "config",
        "vae_code",
        "autoencoder_class_module",
        "python_executable",
    ):
        _validate_contract_file_v2(
            vae_contract.get(f"{prefix}_path"),
            vae_contract.get(f"{prefix}_sha256"),
            label=f"{case_id} VAE {prefix}",
        )
    _require_sha256(
        vae_contract.get("checkpoint_content_identity_sha256"),
        label=f"{case_id} checkpoint content identity",
    )
    input_geometry = raw.get("source_input_frame_geometry")
    bucket_hw = raw.get("source_bucket_hw")
    expected_bucket = (case.full_source_latent_geometry[-2] * 8, case.full_source_latent_geometry[-1] * 8)
    if (
        not isinstance(input_geometry, list)
        or len(input_geometry) != 3
        or any(type(item) is not int or item <= 0 for item in input_geometry)
        or tuple(input_geometry) != (*EXPECTED_SOURCE_INPUT_HW[case_id], 3)
        or not isinstance(bucket_hw, list)
        or any(type(item) is not int for item in bucket_hw)
        or tuple(bucket_hw) != expected_bucket
    ):
        raise OracleActivationV2Error(f"{case_id} decoded/bucket geometry differs")
    raw_rgb_sha = _require_sha_list_v2(
        raw.get("reference_raw_rgb_sha256"),
        label=f"{case_id} raw RGB references",
        count=4,
    )
    preprocessed_sha = _require_sha_list_v2(
        raw.get("reference_preprocessed_rgb_sha256"),
        label=f"{case_id} preprocessed references",
        count=4,
    )
    if len(set(raw_rgb_sha)) != 4 or len(set(preprocessed_sha)) != 4:
        raise OracleActivationV2Error(f"{case_id} source reference content duplicates")
    source_preprocessed_sha = _validate_json_tensor_identity_v2(
        raw.get("full_preprocessed_source_identity"),
        label=f"{case_id} full preprocessed source",
        expected_shape=(1, 3, 81, *expected_bucket),
        expected_dtype="torch.float32",
    )
    source_latent_sha = _validate_json_tensor_identity_v2(
        raw.get("full_source_latent_identity"),
        label=f"{case_id} full source latent",
        expected_shape=case.full_source_latent_geometry,
        expected_dtype="torch.float32",
    )
    reference_rows = raw.get("reference_latent_identities")
    if not isinstance(reference_rows, list) or len(reference_rows) != 4:
        raise OracleActivationV2Error(f"{case_id} VAE reference rows differ")
    reference_latent_sha: list[str] = []
    for position, (row, frame_index) in enumerate(
        zip(reference_rows, REFERENCE_RGB_INDICES)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != _REFERENCE_LATENT_KEYS
            or row.get("frame_index") != frame_index
            or row.get("raw_rgb_sha256") != raw_rgb_sha[position]
            or row.get("preprocessed_rgb_sha256") != preprocessed_sha[position]
            or row.get("shape") != list(case.reference_latent_geometry)
            or row.get("dtype") != "torch.float32"
            or row.get("independently_vae_encoded") is not True
        ):
            raise OracleActivationV2Error(
                f"{case_id} reference latent row {position} differs"
            )
        reference_latent_sha.append(
            _require_sha256(
                row.get("content_sha256"),
                label=f"{case_id} reference latent {position}",
            )
        )
    reference_sha_tuple = tuple(reference_latent_sha)
    if len(set(reference_sha_tuple)) != 4:
        raise OracleActivationV2Error(f"{case_id} reference latent duplicates")
    if rank_receipt.get("all_rank_full_source_latent_sha256") != [source_latent_sha] * 4:
        raise OracleActivationV2Error(f"{case_id} all-rank source latent differs")
    _validate_rank_digest_rows_v2(
        rank_receipt.get("all_rank_reference_latent_sha256"),
        expected=reference_sha_tuple,
        label=f"{case_id} all-rank reference latents",
    )
    refs = tuple(source_reference_latents)
    tensors = (source_video_latent, *refs)
    if len(refs) != 4 or any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
        for value in tensors
    ):
        raise OracleActivationV2Error(f"{case_id} live source/reference tensor contract differs")
    if tuple(source_video_latent.shape) != case.full_source_latent_geometry or any(
        tuple(value.shape) != case.reference_latent_geometry
        or value.device != source_video_latent.device
        for value in refs
    ):
        raise OracleActivationV2Error(f"{case_id} live source/reference geometry differs")
    try:
        safe_core._require_pairwise_storage_disjoint_v1(tensors)
    except Exception as error:
        raise OracleActivationV2Error(str(error)) from error
    observed_source_sha = safe_core.tensor_content_sha256_v1(source_video_latent)
    observed_reference_sha = tuple(
        safe_core.tensor_content_sha256_v1(value) for value in refs
    )
    source_slice_sha = tuple(
        safe_core.tensor_content_sha256_v1(
            source_video_latent[:, :, phase : phase + 1].contiguous()
        )
        for phase in REFERENCE_LATENT_PHASES
    )
    provenance_diagnostic = _reference_provenance_diagnostic_v2(
        expected_source_sha256=source_latent_sha,
        observed_source_sha256=observed_source_sha,
        expected_reference_sha256=reference_sha_tuple,
        observed_reference_sha256=observed_reference_sha,
        observed_source_slice_sha256=source_slice_sha,
    )
    if provenance_diagnostic["contract_matches"] is not True:
        raise OracleActivationV2Error(
            f"{case_id} live VAE source/reference provenance differs: "
            + json.dumps(
                provenance_diagnostic,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return ValidatedReferenceReceiptV2(
        case_id=case_id,
        receipt_seal=case.artifact_seals["vae_reference_receipt"],
        source_latent_sha256=source_latent_sha,
        reference_latent_sha256=reference_sha_tuple,  # type: ignore[arg-type]
        source_preprocessed_sha256=source_preprocessed_sha,
        reference_raw_rgb_sha256=raw_rgb_sha,  # type: ignore[arg-type]
        reference_preprocessed_sha256=preprocessed_sha,  # type: ignore[arg-type]
        _validation_token=_REFERENCE_TOKEN,
    )


def _reference_provenance_diagnostic_v2(
    *,
    expected_source_sha256: str,
    observed_source_sha256: str,
    expected_reference_sha256: Sequence[str],
    observed_reference_sha256: Sequence[str],
    observed_source_slice_sha256: Sequence[str],
) -> Mapping[str, Any]:
    """Describe an exact VAE provenance failure without changing acceptance.

    The validator historically combined source drift, one-or-more reference
    drifts, and reference/full-source-slice collisions into one opaque error.
    This helper reports those exact predicates and their content digests.  It
    deliberately provides no tolerance, fallback, or alternative acceptance
    path: ``contract_matches`` is precisely the original conjunction.
    """

    expected_refs = tuple(expected_reference_sha256)
    observed_refs = tuple(observed_reference_sha256)
    source_slices = tuple(observed_source_slice_sha256)
    if len(expected_refs) != 4 or len(observed_refs) != 4 or len(source_slices) != 4:
        raise OracleActivationV2Error(
            "VAE provenance diagnostic requires one source and four references/slices"
        )
    reference_mismatch_positions = [
        position
        for position, (expected, observed) in enumerate(
            zip(expected_refs, observed_refs)
        )
        if expected != observed
    ]
    slice_collisions = [
        {
            "reference_position": reference_position,
            "source_phase": REFERENCE_LATENT_PHASES[slice_position],
        }
        for reference_position, reference_sha in enumerate(expected_refs)
        for slice_position, slice_sha in enumerate(source_slices)
        if reference_sha == slice_sha
    ]
    source_matches = expected_source_sha256 == observed_source_sha256
    contract_matches = (
        source_matches
        and not reference_mismatch_positions
        and not slice_collisions
    )
    return {
        "contract_matches": contract_matches,
        "source_matches": source_matches,
        "reference_mismatch_positions": reference_mismatch_positions,
        "source_slice_collision_pairs": slice_collisions,
        "expected_source_sha256": expected_source_sha256,
        "observed_source_sha256": observed_source_sha256,
        "expected_reference_sha256": list(expected_refs),
        "observed_reference_sha256": list(observed_refs),
        "observed_source_slice_sha256": list(source_slices),
    }


def preflight_case_material_receipts_v2(
    authority: ValidatedActivationAuthorityV2, *, case_id: str
) -> Mapping[str, Any]:
    """Seal every receipt-owned code/config input before model or dist startup.

    This is deliberately a metadata-only gate.  The two live validators below
    remain authoritative for tensor bytes and are rerun by the local runtime on
    every scheduler step.  This earlier pass closes the load-order gap: no VAE,
    tokenizer, text encoder, renderer, distributed process group, or sampler is
    allowed to initialize before all receipt-selected executable/configuration
    files have passed their exact hash bindings.
    """

    revalidate_compiled_activation_authority_v2(authority)
    if case_id not in authority.cases:
        raise OracleActivationV2Error("material preflight case differs")
    case = authority.cases[case_id]
    if case_id == "e03":
        if (
            case.decision != "ABSTAIN_KEEP_BASE"
            or case.run_arms
            or case.reference_receipt_path is not None
            or case.reference_receipt_sha256 is not None
            or case.prompt_receipt_path is not None
            or case.prompt_receipt_sha256 is not None
            or case.kept_frozen_base_path != EXPECTED_E03_FROZEN_BASE_PATH
            or case.kept_frozen_base_sha256 != EXPECTED_E03_FROZEN_BASE_SHA256
        ):
            raise OracleActivationV2Error("e03 policy-only material contract differs")
        return {
            "case_id": "e03",
            "decision": "ABSTAIN_KEEP_BASE",
            "executed": False,
            "material_receipts_present": False,
            "kept_frozen_base_path": str(case.kept_frozen_base_path),
            "kept_frozen_base_sha256": case.kept_frozen_base_sha256,
        }
    if (
        case.reference_receipt_path is None
        or case.reference_receipt_sha256 is None
        or case.prompt_receipt_path is None
        or case.prompt_receipt_sha256 is None
    ):
        raise OracleActivationV2Error("e02 material receipts are absent")
    reference = _bound_authority_json_v2(
        case,
        artifact_key="vae_reference_receipt",
        path=case.reference_receipt_path,
        expected_sha256=case.reference_receipt_sha256,
        label=f"{case_id} VAE reference receipt preflight",
    )
    prompt = _bound_authority_json_v2(
        case,
        artifact_key="prompt_receipt",
        path=case.prompt_receipt_path,
        expected_sha256=case.prompt_receipt_sha256,
        label=f"{case_id} prompt receipt preflight",
    )
    preprocess = reference.get("preprocess_contract")
    vae_contract = reference.get("vae_contract")
    prompt_contract = prompt.get("prompt_contract")
    if (
        set(reference) != _REFERENCE_RECEIPT_KEYS
        or reference.get("schema_version") != REFERENCE_RECEIPT_SCHEMA_VERSION
        or reference.get("case_id") != case_id
        or reference.get("source_iid") != case.source_iid
        or reference.get("source_video_sha256") != case.source_sha256
        or reference.get("materialization_checks_passed") is not True
        or not isinstance(preprocess, Mapping)
        or set(preprocess) != _PREPROCESS_CONTRACT_KEYS
        or not isinstance(vae_contract, Mapping)
        or set(vae_contract) != _VAE_CONTRACT_KEYS
        or set(prompt) != _PROMPT_RECEIPT_KEYS
        or prompt.get("schema_version") != PROMPT_RECEIPT_SCHEMA_VERSION
        or prompt.get("case_id") != case_id
        or prompt.get("source_iid") != case.source_iid
        or prompt.get("action_caption") != case.action_caption
        or prompt.get("action_caption_sha256") != case.action_caption_sha256
        or prompt.get("materialization_checks_passed") is not True
        or not isinstance(prompt_contract, Mapping)
        or set(prompt_contract) != _PROMPT_CONTRACT_KEYS
    ):
        raise OracleActivationV2Error(f"{case_id} material preflight receipt differs")
    reference_files = {
        "materializer_code": (
            reference.get("materializer_code_path"),
            reference.get("materializer_code_sha256"),
        ),
        "frame_decode_code": (
            preprocess.get("frame_decode_code_path"),
            preprocess.get("frame_decode_code_sha256"),
        ),
        "source_prepare_code": (
            preprocess.get("source_prepare_code_path"),
            preprocess.get("source_prepare_code_sha256"),
        ),
        **{
            f"vae_{prefix}": (
                vae_contract.get(f"{prefix}_path"),
                vae_contract.get(f"{prefix}_sha256"),
            )
            for prefix in (
                "checkpoint_content_manifest",
                "config",
                "vae_code",
                "autoencoder_class_module",
                "python_executable",
            )
        },
    }
    prompt_files = {
        "materializer_code": (
            prompt.get("materializer_code_path"),
            prompt.get("materializer_code_sha256"),
        ),
        **{
            prefix: (
                prompt_contract.get(f"{prefix}_path"),
                prompt_contract.get(f"{prefix}_sha256"),
            )
            for prefix in (
                "tokenizer_config",
                "tokenizer_code",
                "checkpoint_content_manifest",
                "text_encoder_config",
                "renderer_code",
                "prompt_builder_code",
                "native_prompt_code",
                "prompt_cleaner_code",
                "auto_tokenizer_module",
                "resolved_tokenizer_class_module",
                "text_encoder_class_module",
                "python_executable",
            )
        },
    }
    for group, rows in (("reference", reference_files), ("prompt", prompt_files)):
        for name, (path, digest) in rows.items():
            _validate_contract_file_v2(
                path, digest, label=f"{case_id} {group} preflight {name}"
            )
    if (
        vae_contract.get("checkpoint_content_manifest_path")
        != prompt_contract.get("checkpoint_content_manifest_path")
        or vae_contract.get("checkpoint_content_manifest_sha256")
        != prompt_contract.get("checkpoint_content_manifest_sha256")
        or vae_contract.get("checkpoint_content_identity_sha256")
        != prompt_contract.get("checkpoint_content_identity_sha256")
    ):
        raise OracleActivationV2Error(
            f"{case_id} VAE/prompt checkpoint provenance differs"
        )
    return {
        "case_id": case_id,
        "reference_receipt_sha256": case.reference_receipt_sha256,
        "prompt_receipt_sha256": case.prompt_receipt_sha256,
        "checkpoint_content_manifest_path": vae_contract[
            "checkpoint_content_manifest_path"
        ],
        "checkpoint_content_manifest_sha256": vae_contract[
            "checkpoint_content_manifest_sha256"
        ],
        "checkpoint_content_identity_sha256": vae_contract[
            "checkpoint_content_identity_sha256"
        ],
        "all_executable_and_config_files_rehashed_before_model_or_dist": True,
    }


def validate_prompt_receipt_v2(
    authority: ValidatedActivationAuthorityV2,
    *,
    case_id: str,
    low_action_prompt_embeds: Any,
    high_action_prompt_embeds: Any,
    negative_prompt_embeds: Any,
) -> ValidatedPromptReceiptV2:
    """Bind low/high/unconditional live embeddings to rank0-only T5 receipt."""

    import torch

    revalidate_compiled_activation_authority_v2(authority)
    if case_id != "e02" or case_id not in authority.cases:
        raise OracleActivationV2Error(
            "prompt receipts are authorized only for active e02"
        )
    case = authority.cases[case_id]
    if case.prompt_receipt_path is None or case.prompt_receipt_sha256 is None:
        raise OracleActivationV2Error("e02 prompt receipt is absent")
    raw = _bound_authority_json_v2(
        case,
        artifact_key="prompt_receipt",
        path=case.prompt_receipt_path,
        expected_sha256=case.prompt_receipt_sha256,
        label=f"{case_id} prompt receipt",
    )
    prompt_contract = raw.get("prompt_contract")
    rank_receipt = raw.get("rank_world_receipt")
    if (
        set(raw) != _PROMPT_RECEIPT_KEYS
        or raw.get("schema_version") != PROMPT_RECEIPT_SCHEMA_VERSION
        or raw.get("case_id") != case_id
        or raw.get("source_iid") != case.source_iid
        or raw.get("action_caption") != case.action_caption
        or raw.get("action_caption_sha256") != case.action_caption_sha256
        or raw.get("rank0_only_text_encoder_load") is not True
        or raw.get("nonzero_ranks_never_deserialized_text_encoder") is not True
        or raw.get("self_generated_anchor_tensor_used") is not False
        or raw.get("target_video_or_latent_used") is not False
        or raw.get("materialization_checks_passed") is not True
        or not isinstance(prompt_contract, Mapping)
        or set(prompt_contract) != _PROMPT_CONTRACT_KEYS
        or prompt_contract.get("tokenizer_function")
        != "infer_lora._tokenize_training_prompt+_tokenize_renderer_negative"
        or prompt_contract.get("text_encoder_function")
        != "bernini.models.renderer.BerniniRendererModel.encode_prompt"
        or prompt_contract.get("max_length") != 512
        or prompt_contract.get("embedding_dtype") != "torch.bfloat16"
        or any(
            not isinstance(prompt_contract.get(key), str)
            or not prompt_contract.get(key)
            for key in (
                "transformers_version",
                "torch_version",
                "python_version",
                "rocm_version",
            )
        )
        or not isinstance(rank_receipt, Mapping)
        or set(rank_receipt) != _PROMPT_RANK_WORLD_KEYS
        or rank_receipt.get("world_size") != 4
        or rank_receipt.get("sequence_parallel_size") != 4
        or rank_receipt.get("rank0_only_text_encode") is not True
        or rank_receipt.get("all_rank_text_encoder_load_roles")
        != [
            {
                "rank": rank,
                "real_t5_loaded": rank == 0,
                "bypassed_t5_load": rank != 0,
                "bypass_call_count": 0 if rank == 0 else 1,
                "placeholder_retained": rank != 0,
            }
            for rank in range(4)
        ]
        or rank_receipt.get("broadcast_exact") is not True
    ):
        raise OracleActivationV2Error(f"{case_id} prompt receipt contract differs")
    _validate_contract_file_v2(
        raw.get("materializer_code_path"),
        raw.get("materializer_code_sha256"),
        label=f"{case_id} prompt materializer code",
    )
    for prefix in (
        "tokenizer_config",
        "tokenizer_code",
        "checkpoint_content_manifest",
        "text_encoder_config",
        "renderer_code",
        "prompt_builder_code",
        "native_prompt_code",
        "prompt_cleaner_code",
        "auto_tokenizer_module",
        "resolved_tokenizer_class_module",
        "text_encoder_class_module",
        "python_executable",
    ):
        _validate_contract_file_v2(
            prompt_contract.get(f"{prefix}_path"),
            prompt_contract.get(f"{prefix}_sha256"),
            label=f"{case_id} prompt {prefix}",
        )
    _require_sha256(
        prompt_contract.get("checkpoint_content_identity_sha256"),
        label=f"{case_id} prompt checkpoint content identity",
    )
    roles = (
        ("low_action", "low-vr2v", low_action_prompt_embeds),
        ("high_action", "high-r2v4", high_action_prompt_embeds),
        ("negative", "renderer-negative", negative_prompt_embeds),
    )
    role_sha: list[str] = []
    rendered_sha: list[str] = []
    token_sha: list[str] = []
    mask_sha: list[str] = []
    for key, expected_mode, tensor in roles:
        row = raw.get(key)
        if (
            not isinstance(row, Mapping)
            or set(row) != _PROMPT_ROLE_KEYS
            or row.get("mode") != expected_mode
            or not isinstance(row.get("rendered_text"), str)
            or not row["rendered_text"]
            or hashlib.sha256(row["rendered_text"].encode("utf-8")).hexdigest()
            != row.get("rendered_text_sha256")
        ):
            raise OracleActivationV2Error(f"{case_id} {key} prompt role differs")
        rendered_sha.append(str(row["rendered_text_sha256"]))
        token_sha.append(
            _require_sha256(
                row.get("token_ids_sha256"), label=f"{case_id} {key} token IDs"
            )
        )
        mask_sha.append(
            _require_sha256(
                row.get("attention_mask_sha256"),
                label=f"{case_id} {key} attention mask",
            )
        )
        expected_sha = _validate_json_tensor_identity_v2(
            row.get("embedding_identity"),
            label=f"{case_id} {key} embedding",
            expected_shape=(1, 512, 4096),
            expected_dtype="torch.bfloat16",
        )
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != (1, 512, 4096)
            or tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or tensor.grad_fn is not None
            or not bool(torch.isfinite(tensor).all().item())
            or safe_core.tensor_content_sha256_v1(tensor) != expected_sha
        ):
            raise OracleActivationV2Error(f"{case_id} live {key} embedding differs")
        role_sha.append(expected_sha)
    if len(set(role_sha)) != 3:
        raise OracleActivationV2Error(f"{case_id} low/high/negative prompt content aliases")
    try:
        safe_core._require_pairwise_storage_disjoint_v1(
            (low_action_prompt_embeds, high_action_prompt_embeds, negative_prompt_embeds)
        )
    except Exception as error:
        raise OracleActivationV2Error(str(error)) from error
    for label, expected_sha in zip(
        ("low_action", "high_action", "negative"), role_sha
    ):
        if rank_receipt.get(f"all_rank_{label}_sha256") != [expected_sha] * 4:
            raise OracleActivationV2Error(f"{case_id} all-rank {label} prompt differs")
    return ValidatedPromptReceiptV2(
        case_id=case_id,
        receipt_seal=case.artifact_seals["prompt_receipt"],
        low_action_sha256=role_sha[0],
        high_action_sha256=role_sha[1],
        negative_sha256=role_sha[2],
        rendered_text_sha256=tuple(rendered_sha),  # type: ignore[arg-type]
        token_ids_sha256=tuple(token_sha),  # type: ignore[arg-type]
        attention_mask_sha256=tuple(mask_sha),  # type: ignore[arg-type]
        _validation_token=_PROMPT_TOKEN,
    )


def _clone_owned_gate_v2(gate: _OwnedHardStateChangeGateV2) -> _OwnedHardStateChangeGateV2:
    cloned = _OwnedHardStateChangeGateV2(
        delete=gate.delete.clone().detach().contiguous(),
        create=gate.create.clone().detach().contiguous(),
        contact=gate.contact.clone().detach().contiguous(),
        support=gate.support.clone().detach().contiguous(),
        preserve=gate.preserve.clone().detach().contiguous(),
        source_mask_sha256=gate.source_mask_sha256,
        realized_gate_sha256=gate.realized_gate_sha256,
        delete_count=gate.delete_count,
        create_count=gate.create_count,
        contact_count=gate.contact_count,
        support_count=gate.support_count,
    )
    _validate_owned_gate_v2(
        cloned, expected_geometry=tuple(int(item) for item in gate.support.shape)
    )
    return cloned


def mint_native_local_execution_capability_v2(
    authority: ValidatedActivationAuthorityV2,
    *,
    case_id: str,
    source_video_latent: Any,
    source_reference_latents: Sequence[Any],
    low_action_prompt_embeds: Any,
    high_action_prompt_embeds: Any,
    negative_prompt_embeds: Any,
) -> NativeLocalExecutionCapabilityV2:
    """Mint the sole local-runtime capability from compiled authority + live values."""

    revalidate_compiled_activation_authority_v2(authority)
    _load_native_schedule_contract_v2()
    if case_id != "e02":
        raise OracleActivationV2Error(
            "e03 is ABSTAIN_KEEP_BASE and cannot mint a local execution capability"
        )
    gate = validate_manual_gate_v2(authority, case_id=case_id)
    references = validate_reference_receipt_v2(
        authority,
        case_id=case_id,
        source_video_latent=source_video_latent,
        source_reference_latents=source_reference_latents,
    )
    prompts = validate_prompt_receipt_v2(
        authority,
        case_id=case_id,
        low_action_prompt_embeds=low_action_prompt_embeds,
        high_action_prompt_embeds=high_action_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )
    if (
        gate._validation_token is not _GATE_TOKEN
        or references._validation_token is not _REFERENCE_TOKEN
        or prompts._validation_token is not _PROMPT_TOKEN
    ):
        raise OracleActivationV2Error("execution material capability differs")
    case = authority.cases[case_id]
    sample_id = (
        f"round37:{case_id}:seed-{case.seed}:packet-{authority.packet_sha256[:16]}"
    )
    owned_gate = _clone_owned_gate_v2(gate.owned_gate)
    return NativeLocalExecutionCapabilityV2(
        authority=authority,
        case_id=case_id,
        sample_id=sample_id,
        manifest=gate.manifest,
        owned_gate=owned_gate,
        realized_gate_sha256=owned_gate.realized_gate_sha256,
        source_latent_sha256=references.source_latent_sha256,
        source_reference_latent_sha256=references.reference_latent_sha256,
        source_reference_rgb_indices=REFERENCE_RGB_INDICES,
        low_action_prompt_sha256=prompts.low_action_sha256,
        r2v_action_prompt_sha256=prompts.high_action_sha256,
        negative_prompt_sha256=prompts.negative_sha256,
        r2v_action_prompt_embeds=(
            high_action_prompt_embeds.clone().detach().contiguous()
        ),
        authority_packet_path=authority.packet_path,
        authority_packet_sha256=authority.packet_sha256,
        _validation_token=_CAPABILITY_TOKEN,
    )


def _packed_hard_support_v2(
    gate: _OwnedHardStateChangeGateV2,
    *,
    target_latent_shape: tuple[int, int, int, int, int],
    device: Any,
) -> Any:
    expected_gate_shape = (
        target_latent_shape[0],
        1,
        target_latent_shape[2],
        target_latent_shape[3],
        target_latent_shape[4],
    )
    _validate_owned_gate_v2(gate, expected_geometry=expected_gate_shape)
    spatial = gate.support.to(device=device).expand(target_latent_shape)
    return sgaf._spatial_to_packed(spatial, target_latent_shape)


def _scheduled_local_velocity_v2(
    *,
    sample: Any,
    high_r2v4_velocity: Any,
    official_v2v_velocity: Any,
    sigma: Any,
    gate: _OwnedHardStateChangeGateV2,
    target_latent_shape: tuple[int, int, int, int, int],
) -> tuple[Any, Mapping[str, Any]]:
    """Scheduled source-reference R2V-4 inside exact G=D|C|K only."""

    import torch

    if not all(
        isinstance(value, torch.Tensor) for value in (sample, official_v2v_velocity)
    ) or (
        sample.ndim != 3
        or sample.numel() <= 0
        or tuple(sample.shape) != tuple(official_v2v_velocity.shape)
        or sample.dtype != official_v2v_velocity.dtype
        or sample.device != official_v2v_velocity.device
        or not sample.dtype.is_floating_point
        or sample.requires_grad
        or sample.grad_fn is not None
        or official_v2v_velocity.requires_grad
        or official_v2v_velocity.grad_fn is not None
        or not bool(torch.isfinite(sample).all().item())
        or not bool(torch.isfinite(official_v2v_velocity).all().item())
    ):
        raise OracleActivationV2Error("sample/official local velocity contract differs")
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma).item())
        or not bool((sigma > 0).item())
    ):
        raise OracleActivationV2Error("local velocity sigma must be CPU fp32 positive")
    packed_support = _packed_hard_support_v2(
        gate,
        target_latent_shape=target_latent_shape,
        device=official_v2v_velocity.device,
    )
    if tuple(packed_support.shape) != tuple(official_v2v_velocity.shape):
        raise OracleActivationV2Error("packed D/C/K support geometry differs")
    support_nonzero = bool(packed_support.any().item())
    sigma_float = float(sigma.item())
    if sigma_float <= native_homotopy.SIGMA_LOW:
        endpoint, high_weight = "low_official_v2v_apg", 0.0
    elif sigma_float >= native_homotopy.SIGMA_HIGH:
        endpoint, high_weight = "high_r2v4_apg", 1.0
    else:
        weight = native_homotopy.smoothstep_high_branch_weight(sigma)
        endpoint, high_weight = "transition", float(weight.item())
    low_weight = 1.0 - high_weight
    common = {
        "schema_version": LOCAL_RUNTIME_SCHEMA_VERSION,
        "sigma": sigma_float,
        "high_r2v4_weight": high_weight,
        "low_official_v2v_apg_weight": low_weight,
        "endpoint": endpoint,
        "scheduled_endpoint_prelocal": endpoint,
        "outside_hard_support_byte_exact": True,
        "hard_support_definition": "G=D_or_C_or_K",
        "hard_support_fraction": float(packed_support.float().mean().item()),
        "realized_gate_sha256": gate.realized_gate_sha256,
    }
    if not support_nonzero:
        return official_v2v_velocity, {
            **common,
            "scheduled_expert_evaluated": False,
            "high_velocity_aggregated": False,
            "scheduled_endpoint_prelocal_direct_return_verified": None,
            "executed_local_where": False,
            "null_gate": True,
            "scheduler_received_original_official_object": True,
        }
    if (
        not isinstance(high_r2v4_velocity, torch.Tensor)
        or tuple(high_r2v4_velocity.shape) != tuple(sample.shape)
        or high_r2v4_velocity.dtype != sample.dtype
        or high_r2v4_velocity.device != sample.device
        or high_r2v4_velocity.requires_grad
        or high_r2v4_velocity.grad_fn is not None
        or not bool(torch.isfinite(high_r2v4_velocity).all().item())
    ):
        raise OracleActivationV2Error("active high R2V-4 velocity differs")
    try:
        scheduled = native_homotopy.native_branch_homotopy_step(
            sample,
            high_r2v4_velocity,
            official_v2v_velocity,
            sigma,
            high_r2v4_momentum=0.0,
            low_official_v2v_apg_momentum=0.0,
        )
    except Exception as error:
        raise OracleActivationV2Error(str(error)) from error
    expert = scheduled.velocity
    direct_verified = (
        scheduled.endpoint == "high_r2v4_apg" and expert is high_r2v4_velocity
    ) or (
        scheduled.endpoint == "low_official_v2v_apg"
        and expert is official_v2v_velocity
    )
    if scheduled.endpoint != "transition" and not direct_verified:
        raise OracleActivationV2Error("scheduled endpoint object identity differs")
    if expert is official_v2v_velocity:
        return official_v2v_velocity, {
            **scheduled.trace_dict(),
            **common,
            "scheduled_expert_evaluated": True,
            "high_velocity_aggregated": False,
            "scheduled_endpoint_prelocal_direct_return_verified": True,
            "executed_local_where": False,
            "null_gate": False,
            "scheduler_received_original_official_object": True,
        }
    executed = torch.where(packed_support, expert, official_v2v_velocity)
    outside = torch.logical_not(packed_support)
    if not torch.equal(
        executed[outside].contiguous().view(torch.uint8),
        official_v2v_velocity[outside].contiguous().view(torch.uint8),
    ):
        raise OracleActivationV2Error("local execution changed bytes outside G")
    return executed, {
        **scheduled.trace_dict(),
        **common,
        "scheduled_expert_evaluated": True,
        "high_velocity_aggregated": True,
        "scheduled_endpoint_prelocal_direct_return_verified": (
            direct_verified if scheduled.endpoint != "transition" else False
        ),
        "executed_local_where": True,
        "null_gate": False,
        "scheduler_received_original_official_object": False,
    }


class LocalOracleNativeBranchRuntimePatchV2(
    safe_core.LocalOracleNativeBranchRuntimePatchV1
):
    """Five-forward native patch authorized by the compiled v2 D/C/K packet."""

    def __init__(
        self,
        diffusion: Any,
        *,
        config: native_runtime.NativeBranchHomotopyRuntimeConfig,
        capability: NativeLocalExecutionCapabilityV2,
        expected_bernini_commit: str = native_runtime.PINNED_BERNINI_COMMIT,
        observed_wan_diffusion_sha256: str = native_runtime.PINNED_WAN_DIFFUSION_SHA256,
    ) -> None:
        if (
            not isinstance(capability, NativeLocalExecutionCapabilityV2)
            or capability._validation_token is not _CAPABILITY_TOKEN
            or capability.case_id != "e02"
        ):
            raise OracleActivationV2Error(
                "local runtime requires an authenticated e02 v2 capability"
            )
        revalidate_compiled_activation_authority_v2(capability.authority)
        native_schedule_contract = _load_native_schedule_contract_v2()
        owned_gate = _clone_owned_gate_v2(capability.owned_gate)
        expected_gate_shape = (
            config.target_latent_shape[0],
            1,
            config.target_latent_shape[2],
            config.target_latent_shape[3],
            config.target_latent_shape[4],
        )
        _validate_owned_gate_v2(owned_gate, expected_geometry=expected_gate_shape)
        if (
            not bool(owned_gate.support.any().item())
            or owned_gate.realized_gate_sha256 != capability.realized_gate_sha256
            or capability.authority_packet_sha256
            != capability.authority.packet_sha256
            or capability.authority_packet_path != capability.authority.packet_path
            or safe_core.tensor_content_sha256_v1(
                capability.r2v_action_prompt_embeds
            )
            != capability.r2v_action_prompt_sha256
        ):
            raise OracleActivationV2Error("local runtime capability content differs")
        self._validated_gate_manifest = capability.manifest
        self._native_execution_binding = capability
        self._expected_annotation_authority_root_sha256 = (
            capability.manifest.annotation_authority_root_sha256
        )
        self._owned_hard_gate = owned_gate
        self._expected_realized_gate_sha256 = owned_gate.realized_gate_sha256
        self._live_native_binding_tensors: Optional[tuple[Any, ...]] = None
        self._native_schedule_contract = native_schedule_contract
        self._binding_revalidation_count = 0
        native_runtime.NativeBranchHomotopyRuntimePatch.__init__(
            self,
            diffusion,
            r2v_action_prompt_embeds=(
                capability.r2v_action_prompt_embeds.clone().detach().contiguous()
            ),
            config=config,
            expected_bernini_commit=expected_bernini_commit,
            observed_wan_diffusion_sha256=observed_wan_diffusion_sha256,
        )

    def _validate_sample_contract(self, values: Mapping[str, Any]) -> Any:
        state = native_runtime.NativeBranchHomotopyRuntimePatch._validate_sample_contract(
            self, values
        )
        if self._live_native_binding_tensors is not None:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live activation-v2 tensors were already captured"
            )
        self._live_native_binding_tensors = (
            state.source_video,
            *state.references,
            state.low_action_prompt,
            state.high_action_prompt,
            state.low_negative_prompt,
        )
        self._certify_activation_v2_snapshot(revalidate_files=True)
        return state

    def _certify_activation_v2_snapshot(self, *, revalidate_files: bool) -> None:
        capability = self._native_execution_binding
        revalidate_compiled_activation_authority_v2(capability.authority)
        if _load_native_schedule_contract_v2() is not self._native_schedule_contract:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live native UniPC40 schedule module changed"
            )
        _validate_owned_gate_v2(
            self._owned_hard_gate,
            expected_geometry=(
                self.config.target_latent_shape[0],
                1,
                self.config.target_latent_shape[2],
                self.config.target_latent_shape[3],
                self.config.target_latent_shape[4],
            ),
        )
        if (
            self._owned_hard_gate.realized_gate_sha256
            != self._expected_realized_gate_sha256
            or capability.realized_gate_sha256
            != self._expected_realized_gate_sha256
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live activation-v2 D/C/K gate changed"
            )
        values = self._live_native_binding_tensors
        if values is None or len(values) != 8:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live activation-v2 source/reference/prompt snapshot is absent"
            )
        source, *tail = values
        references = tuple(tail[:4])
        low_action, high_action, negative = tail[4:]
        observed = (
            safe_core.tensor_content_sha256_v1(source),
            tuple(safe_core.tensor_content_sha256_v1(value) for value in references),
            safe_core.tensor_content_sha256_v1(low_action),
            safe_core.tensor_content_sha256_v1(high_action),
            safe_core.tensor_content_sha256_v1(negative),
        )
        expected = (
            capability.source_latent_sha256,
            capability.source_reference_latent_sha256,
            capability.low_action_prompt_sha256,
            capability.r2v_action_prompt_sha256,
            capability.negative_prompt_sha256,
        )
        if observed != expected:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "live activation-v2 source/reference/prompt bytes changed"
            )
        if revalidate_files:
            validate_manual_gate_v2(capability.authority, case_id="e02")
            validate_reference_receipt_v2(
                capability.authority,
                case_id="e02",
                source_video_latent=source,
                source_reference_latents=references,
            )
            validate_prompt_receipt_v2(
                capability.authority,
                case_id="e02",
                low_action_prompt_embeds=low_action,
                high_action_prompt_embeds=high_action,
                negative_prompt_embeds=negative,
            )
        self._binding_revalidation_count += 1

    def _certify_owned_gate_snapshot(self, *, revalidate_files: bool) -> None:
        self._certify_activation_v2_snapshot(revalidate_files=revalidate_files)

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        state = self._active
        if state is None:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "scheduler.step ran outside authenticated activation-v2 sample"
            )
        self._certify_activation_v2_snapshot(revalidate_files=False)
        if (
            len(state.patch_results) != 10
            or tuple(item.source_id for item in state.patch_results)
            != native_runtime.EXPECTED_PATCH_SOURCE_IDS
            or len(state.low_forwards) != 2
            or len(state.high_forwards) != 3
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "scheduler.step arrived before activation-v2 five-forward closure"
            )
        official = sgaf._extract_argument(args, kwargs, index=0, name="model_output")
        timestep = sgaf._extract_argument(args, kwargs, index=1, name="timestep")
        sample = sgaf._extract_argument(args, kwargs, index=2, name="sample")
        try:
            safe_core._certify_expanded_timestep_compat_v1(
                state.low_forwards[1].values["timesteps"], timestep
            )
        except Exception as error:
            raise native_runtime._raise_from_sgaf(error) from error
        expected_shape = (
            1,
            self.config.target_patch_tokens,
            self.config.target_latent_shape[1] * 4,
        )
        for label, value in (
            ("official model_output", official),
            ("scheduler sample", sample),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or native_runtime._shape(value, label=label) != expected_shape
                or not value.dtype.is_floating_point
                or value.requires_grad
                or value.grad_fn is not None
                or not bool(torch.isfinite(value).all().item())
            ):
                raise native_runtime.NativeBranchHomotopyRuntimeError(
                    f"{label} geometry differs"
                )
        if official.device != sample.device or official.dtype != sample.dtype:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "official output/sample dtype or device differs"
            )
        expected_target_patch_input = sgaf._packed_to_spatial(
            sample, self.config.target_latent_shape
        ).to(dtype=self.transformer.dtype)
        observed_target_patch_input = state.patch_results[9].input_value
        if (
            not isinstance(observed_target_patch_input, torch.Tensor)
            or observed_target_patch_input.shape != expected_target_patch_input.shape
            or observed_target_patch_input.device != expected_target_patch_input.device
            or observed_target_patch_input.dtype != expected_target_patch_input.dtype
            or not safe_core._tensor_bytes_equal_v1(
                observed_target_patch_input, expected_target_patch_input
            )
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "captured target patch input differs from scheduler sample"
            )
        try:
            step_index, sigma, sigma_float = sgaf._resolve_sigma(
                self.scheduler, timestep
            )
        except Exception as error:
            raise native_runtime._raise_from_sgaf(error) from error
        if step_index != state.completed_steps:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "activation-v2 scheduler step index differs"
            )
        if (
            not isinstance(sigma, torch.Tensor)
            or sigma.ndim != 0
            or sigma.device.type != "cpu"
            or sigma.dtype != torch.float32
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "active UniPC sigma must remain CPU fp32 scalar"
            )
        low_parameters = sgaf._APGParameters(
            guidance_scale=self.config.omega_text,
            eta=self.config.eta,
            norm_threshold=self.config.image_norm_threshold,
            momentum=0.0,
        )
        rebuilt_low = sgaf._guided_velocity(
            sample,
            state.low_forwards[0].target_tail,
            state.low_forwards[1].target_tail,
            sigma,
            shape=self.config.target_latent_shape,
            parameters=low_parameters,
            momentum_buffer=state.low_momentum,
            output_like=official,
        )
        parity_delta = rebuilt_low.float() - official.float()
        parity_rms = native_runtime._tensor_rms(parity_delta)
        parity_max = float(parity_delta.abs().max().item())
        if not safe_core._tensor_bytes_equal_v1(rebuilt_low, official):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "activation-v2 rebuilt low V2V APG bytes differ from official"
            )
        high = self._high_r2v4_velocity(
            state, sample=sample, sigma=sigma, official=official
        )
        try:
            executed, local_trace = _scheduled_local_velocity_v2(
                sample=sample,
                high_r2v4_velocity=high,
                official_v2v_velocity=official,
                sigma=sigma,
                gate=self._owned_hard_gate,
                target_latent_shape=self.config.target_latent_shape,
            )
        except OracleActivationV2Error as error:
            raise native_runtime.NativeBranchHomotopyRuntimeError(str(error)) from error
        if executed is official:
            call_args, call_kwargs = tuple(args), dict(kwargs)
        else:
            call_args, call_kwargs = native_runtime._replace(
                self.original_scheduler_step,
                args,
                kwargs,
                name="model_output",
                value=executed,
            )
        result = self.original_scheduler_step(*call_args, **call_kwargs)
        self.original_scheduler_call_count += 1
        state.completed_steps += 1
        self.trace.append(
            {
                "step_index": step_index,
                "timestep": native_runtime._scalar(timestep, label="timestep"),
                "sigma": sigma_float,
                "forward_order": list(native_runtime.PER_STEP_FORWARD_ORDER),
                "transformer_forwards": 5,
                "low_vi_forwards": 2,
                "high_r2v4_forwards": 3,
                "high_forwards_executed": True,
                "original_scheduler_calls": 1,
                "patch_call_count": 10,
                "patch_source_ids": list(native_runtime.EXPECTED_PATCH_SOURCE_IDS),
                "low_official_apg_exact_parity": True,
                "low_official_apg_parity_rms": parity_rms,
                "low_official_apg_parity_max_abs": parity_max,
                "high_low_velocity_delta_rms": native_runtime._tensor_rms(
                    high.float() - official.float()
                ),
                **local_trace,
                "schema_version": LOCAL_RUNTIME_SCHEMA_VERSION,
                "scheduler_received_original_model_output_object": executed is official,
                "hard_gate_dtype": "bool",
                "hard_gate_channels": ["D", "C", "K", "G"],
                "soft_gate_used": False,
                "freeze_safe_no_grad_outputs": all(
                    not value.requires_grad and value.grad_fn is None
                    for value in (official, high, executed)
                ),
            }
        )
        state.patch_results.clear()
        state.low_forwards.clear()
        state.high_forwards.clear()
        return result

    def finalize(self) -> Mapping[str, Any]:
        if not self.restored or self.finalized:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "activation-v2 local patch finalize differs"
            )
        self._certify_activation_v2_snapshot(revalidate_files=True)
        steps = self.config.expected_steps
        if (
            steps != 40
            or self.sample_call_count != 1
            or self.schedule_preflight is None
            or self.patch_call_count != 10 * steps
            or self.low_forward_count != 2 * steps
            or self.high_forward_count != 3 * steps
            or self.original_scheduler_call_count != steps
            or len(self.trace) != steps
        ):
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "activation-v2 local runtime call-count certificate differs"
            )
        expected_endpoints = (
            ["high_r2v4_apg"] * 15
            + ["transition"] * 16
            + ["low_official_v2v_apg"] * 9
        )
        if [row.get("scheduled_endpoint_prelocal") for row in self.trace] != expected_endpoints:
            raise native_runtime.NativeBranchHomotopyRuntimeError(
                "activation-v2 exact40 endpoint partition differs"
            )
        for index, row in enumerate(self.trace):
            local_expected = index < 31
            if (
                row.get("low_official_apg_exact_parity") is not True
                or row.get("transformer_forwards") != 5
                or row.get("original_scheduler_calls") != 1
                or row.get("high_forwards_executed") is not True
                or row.get("outside_hard_support_byte_exact") is not True
                or row.get("hard_support_definition") != "G=D_or_C_or_K"
                or row.get("realized_gate_sha256")
                != self._expected_realized_gate_sha256
                or row.get("soft_gate_used") is not False
                or row.get("freeze_safe_no_grad_outputs") is not True
                or row.get("executed_local_where") is not local_expected
                or row.get("high_velocity_aggregated") is not local_expected
                or row.get("scheduler_received_original_model_output_object")
                is not (not local_expected)
            ):
                raise native_runtime.NativeBranchHomotopyRuntimeError(
                    f"activation-v2 local trace step {index} differs"
                )
        self.finalized = True
        capability = self._native_execution_binding
        gate = self._owned_hard_gate
        trace_sha256 = _canonical_object_sha256(self.trace)
        return {
            "schema_version": LOCAL_RUNTIME_SCHEMA_VERSION,
            "execution": "scheduled_source_reference_r2v4_inside_exact_G_official_v2v_outside",
            "case_id": "e02",
            "sample_id": capability.sample_id,
            "compiled_authority_packet_sha256": capability.authority.packet_sha256,
            "compiled_external_ledger_receipt_sha256": capability.authority.ledger_sha256,
            "steps": steps,
            "transformer_forwards": self.low_forward_count + self.high_forward_count,
            "low_vi_forwards": self.low_forward_count,
            "high_r2v4_forwards": self.high_forward_count,
            "patch_vae_latent_calls": self.patch_call_count,
            "original_scheduler_calls": self.original_scheduler_call_count,
            "per_step_forward_order": list(native_runtime.PER_STEP_FORWARD_ORDER),
            "schedule_preflight": dict(self.schedule_preflight),
            "exact40_scheduled_endpoint_partition": {
                "high_r2v4_apg_indices": list(range(0, 15)),
                "transition_indices": list(range(15, 31)),
                "low_official_v2v_apg_indices": list(range(31, 40)),
            },
            "hard_gate_channels": ["D", "C", "K", "G"],
            "hard_support_definition": "G=D_or_C_or_K",
            "source_manual_mask_sha256": gate.source_mask_sha256,
            "realized_gate_sha256": gate.realized_gate_sha256,
            "gate_mass_receipt": {
                "delete_D_count": gate.delete_count,
                "create_C_count": gate.create_count,
                "contact_K_count": gate.contact_count,
                "union_G_count": gate.support_count,
            },
            "outside_G_official_bytes_exact_all_steps": True,
            "source_reference_prompt_live_rehash_count": self._binding_revalidation_count,
            "source_latent_sha256": capability.source_latent_sha256,
            "source_reference_latent_sha256": list(
                capability.source_reference_latent_sha256
            ),
            "source_reference_rgb_indices": list(
                capability.source_reference_rgb_indices
            ),
            "low_action_prompt_sha256": capability.low_action_prompt_sha256,
            "high_action_prompt_sha256": capability.r2v_action_prompt_sha256,
            "negative_prompt_sha256": capability.negative_prompt_sha256,
            "source_reference_r2v4_regeneration_expert": True,
            "self_generated_anchor_tensor_used": False,
            "anchor_reference_or_quotient_arm_deferred": True,
            "global_source_reference_r2v4_upper_bound_arm_deferred": True,
            "training": False,
            "optimizer": False,
            "flowedit": False,
            "connected_route": False,
            "automatic_replacement": False,
            "selection_authority": None,
            "trace_sha256": trace_sha256,
            "trace": [dict(row) for row in self.trace],
        }


def contract_v2() -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "compiled_authority_packet_present": isinstance(
            COMPILED_AUTHORITY_PACKET_SHA256, str
        ),
        "compiled_external_ledger_present": isinstance(
            COMPILED_EXTERNAL_LEDGER_RECEIPT_SHA256, str
        ),
        "activation_available": compiled_activation_available_v2(),
        "authority_override_from_cli_env_or_json": False,
        "authoring_template_is_authority": False,
        "native_only": True,
        "flowedit_enabled": False,
        "connected_runner_enabled": False,
        "learned_gate_enabled": False,
        "training_authorized": False,
        "optimizer_authorized": False,
        "automatic_model_replacement_authorized": False,
        "manual_gate_channels": ["D", "C", "K", "G"],
        "hard_support_definition": "G=D_or_C_or_K",
        "maximum_per_phase_hard_support_fraction": 0.30,
        "self_generated_anchor_tensor_used_by_native_expert": False,
        "anchor_reference_or_quotient_arm_deferred": True,
        "e03_default_decision": "ABSTAIN_KEEP_BASE",
    }


__all__ = [
    "ALLOWED_CASES",
    "ARM_LOCAL",
    "ARM_OFFICIAL",
    "AUTHORITY_PACKET_SCHEMA_VERSION",
    "ActivationCaseAuthorityV2",
    "LEDGER_RECEIPT_SCHEMA_VERSION",
    "LOCAL_RUNTIME_SCHEMA_VERSION",
    "LocalOracleNativeBranchRuntimePatchV2",
    "MANUAL_GATE_REVIEW_SCHEMA_VERSION",
    "MANUAL_GATE_SCHEMA_VERSION",
    "NativeLocalExecutionCapabilityV2",
    "OracleActivationV2Error",
    "PROMPT_RECEIPT_SCHEMA_VERSION",
    "REFERENCE_RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ValidatedActivationAuthorityV2",
    "ValidatedManualGateV2",
    "ValidatedPromptReceiptV2",
    "ValidatedReferenceReceiptV2",
    "compiled_activation_available_v2",
    "contract_v2",
    "load_compiled_activation_authority_v2",
    "mint_native_local_execution_capability_v2",
    "preflight_case_material_receipts_v2",
    "revalidate_compiled_activation_authority_v2",
    "validate_manual_gate_v2",
    "validate_prompt_receipt_v2",
    "validate_reference_receipt_v2",
    "verify_frozen_dependency_pins_v2",
]

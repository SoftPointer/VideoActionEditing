#!/usr/bin/env python3
"""Build a portable offline review site from one *real* case01 exact5 run.

This publisher is deliberately downstream of inference.  It accepts only a
staged exact-five final-summary bundle, validates its plan, report,
runner attestation, five source videos, five current-R64 outputs, and five
native receipts, then creates an all-relative HTML site.  It never fills a
missing arm, substitutes the historical case01 result, or creates synthetic
result media.

Expected local bundle layout (a minimal mirror of the exact5 package)::

    bundle/
      plan/case01_source_bone_exact5_r64_plan_v1.json
      final/case01_source_bone_exact5_r64_report_v1.json
      final/case01_source_bone_exact5_runner_attestation_v1.json
      sources/{exact_original,codec_only_present,bone_removed,
               bone_translated_up150,sham_control_up150}.mp4
      outputs/media/case01-<variant>-full644.mp4
      outputs/media/case01-<variant>-full644.mp4.receipt.json

The site is an exploratory causal-input visual review for one Heldout8 case.
It is not a Full644 training subset, a formal training evaluation, or a
scientific/generalization result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "case01-source-bone-exact5-r64-plan-v1"
REPORT_SCHEMA = "case01-source-bone-exact5-r64-report-v1"
ATTESTATION_SCHEMA = "case01-source-bone-exact5-runner-attestation-v1"
FAILURE_SCHEMA = "case01-source-bone-exact5-runner-failure-v1"
RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v5"
SITE_SCHEMA = "case01-source-bone-exact5-r64-offline-review-site-v1"
CAMPAIGN = "case01-source-bone-exact5-r64-canary"
EXPERIMENT_ID = "case01-288545b9c031491a-source-bone-exact5-r64-v1"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
INSTRUCTION_SHA256 = (
    "84df12ede824d239a4c7c3d21dccdf22663535d1e504e7b280544c8a9be0fd5d"
)
SEED = 2027
PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
REFERENCE_OUTPUT_SHA256 = (
    "e0d3c07d1d3e6ae4d45e59713d2af3f04786c305f8842c20d79172a9cae22403"
)
FROZEN_RUNNER_SHA256 = (
    "847b91a267fe55cfbfa793027548f82beb5ec9630efab329878576ae6c5a9223"
)
EXACT5_RUNNER_SHA256 = (
    "4a46e870b61ff42345c523cca97b8170c065033bc2af9a0353be9df4373ec3ea"
)
EXACT5_EVAL_SHA256 = (
    "891551d46b0ca11362fe8d446f202036b9957baa943de0aec6da1f3ad45d7d58"
)
FFPROBE_SHA256 = (
    "356754aa8e327b139dd54dda6846af6425673b73572ab6c1c182ed970f1107f5"
)
EXPECTED_CHECKPOINT = {
    "sha256": "7a4864a3ffa50c12af91f8d2b88610a6cd8f994aa68eef8d27b95bcc2d73d3b2",
    "manifest_digest": "7bae23da51a3c5a67adb41ee85dd026c374d2581bd3409e868e18b2f6f4dffc4",
    "global_step": 644,
    "receipt_digest": "aaf348a7daa6c5ca2fe721771857287125ee02eb2c9a499f45b11a2e113d15d7",
    "file_count": 5,
    "adapter_config_sha256": "94bfaf73d714d7e77095ff68ce57e24932e0c05bde324263f5fe321660b95f62",
    "adapter_model_sha256": "44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22",
    "training_receipt_sha256": "3402c8c93c092bfc4490bf86790ab6429b4cbaad38358956cb0beeb5df7d4c4c",
    "optimizer_sha256": "77b7b22db4da92f28f23b4ae91c7271f55ab6a92353bfc8b0bbeb30529a7af63",
}
EXPECTED_PRODUCER = {
    "inference_receipt_schema": RECEIPT_SCHEMA,
    "infer_lora_sha256": "acc46ff5b2106b7974bc8e1effd5e5c9b682b7ff16421c6d7d3d0d18d396a553",
    "method_source_revision": "ce4cffc1e8a144448c92252d9fb63087f03bbd8c",
    "method_source_archive_sha256": "12a28ddec99704963af42f1a82b09dff31828e3af8e53e5d0bbd0d43db272828",
    "ffprobe_sha256": FFPROBE_SHA256,
}
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_BERNINI_INFERENCE_FILES = {
    "bernini/cli.py": "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
    "bernini/io_utils.py": "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
    "bernini/pipeline.py": "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
}
CLAIM_LIMITS = {
    "exploratory_only": True,
    "scientific_claim_authorized": False,
    "formal_claim_authorized": False,
    "manual_blind_review_required": True,
}
EXPECTED_SOURCE_VIDEO = {
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "width": 704,
    "height": 736,
}
EXPECTED_MEDIA_FRAME_COUNT = 81
EXPECTED_MEDIA_FPS = Fraction(25, 1)
KEYFRAMES = (0, 20, 40, 60, 80)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

VARIANTS = (
    {
        "id": "exact_original",
        "title": "Exact original",
        "title_zh": "原始字节输入",
        "group": "controls",
        "treatment": "exact_original_bytes",
        "summary": "原视频字节不变；bone 位于原始位置。",
        "bone_present": True,
        "bone_position": "source_original",
        "source_sha256": "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
        "source_size": 10_887_043,
    },
    {
        "id": "codec_only_present",
        "title": "Codec-only present",
        "title_zh": "仅编解码控制",
        "group": "controls",
        "treatment": "matched_libx264_crf17_codec_only_transcode",
        "summary": "仅做匹配的 H.264 CRF17 转码；bone 仍在原始位置。",
        "bone_present": True,
        "bone_position": "source_original",
        "source_sha256": "7104ada43b9f8e0168f38dc9710e8dc76f47606c446e5b119155048929af403b",
        "source_size": 5_432_063,
    },
    {
        "id": "bone_removed",
        "title": "Bone removed",
        "title_zh": "移除 bone",
        "group": "interventions",
        "treatment": "per_frame_SAM2_bone_mask_dilate3_bidirectional_boundary_interpolation",
        "summary": "逐帧移除 bone，并在 dilate3 support 内做双向边界插值。",
        "bone_present": False,
        "bone_position": "absent",
        "source_sha256": "8c525385832586fa7b7fd7ae6e5701c599694d26ee27b502dbf0bb582e55e1c9",
        "source_size": 5_424_975,
    },
    {
        "id": "bone_translated_up150",
        "title": "Bone translated +150",
        "title_zh": "bone 上移 150 px",
        "group": "interventions",
        "treatment": "same_original_support_interpolation_plus_exact_source_bone_pixels_up150",
        "summary": "清除原 support，并将原始 bone 像素精确上移 150 px。",
        "bone_present": True,
        "bone_position": "translated_up150",
        "source_sha256": "5ba28a7efd6269872ba7924162eeba8df2fd29df06c6668557310984ef9ed5f6",
        "source_size": 5_441_472,
    },
    {
        "id": "sham_control_up150",
        "title": "Sham control +150",
        "title_zh": "空间匹配 sham 控制",
        "group": "controls",
        "treatment": "same_boundary_interpolation_on_spatially_matched_up150_support",
        "summary": "在上移 150 px 的空间匹配 support 做同样插值；bone 保留原位。",
        "bone_present": True,
        "bone_position": "source_original",
        "source_sha256": "ea380344b1d5c644beee9c886a67b413170d26517bab3315daed32a388ebfac3",
        "source_size": 5_420_413,
    },
)
VARIANT_ORDER = tuple(row["id"] for row in VARIANTS)
TASK_IDS = tuple(f"case01-{variant}-full644" for variant in VARIANT_ORDER)
VARIANT_BY_ID = {row["id"]: row for row in VARIANTS}

PLAN_REL = Path("plan/case01_source_bone_exact5_r64_plan_v1.json")
REPORT_REL = Path("final/case01_source_bone_exact5_r64_report_v1.json")
ATTESTATION_REL = Path(
    "final/case01_source_bone_exact5_runner_attestation_v1.json"
)

_ASSET_AUTHORITY_FIELDS = {
    "schema_version", "status", "launch_allowed",
    "independent_visual_audit_status", "manifest_path", "manifest_sha256",
    "manifest_size", "manifest_digest", "independent_audit_receipt_path",
    "independent_audit_receipt_sha256", "independent_audit_receipt_size",
    "independent_audit_receipt_digest", "iid", "sources",
    "source_rows_digest", "authority_digest",
}
_RECEIPT_TOP_FIELDS = {
    "schema_version", "infer_lora_source_sha256", "method_source_revision",
    "method_source_archive_sha256", "bernini_commit", "veomni_commit",
    "bernini_inference_files", "checkpoint_tree_sha256", "adapter", "input",
    "preprocessing", "prompt_contract", "sampling", "output", "runtime_versions",
    "experimental_inference", "production_claim_forbidden",
    "scientific_claim_authorized", "consumption_input_digest", "task_input_digest",
    "model_consumption", "receipt_digest",
}
_INPUT_FIELDS = {
    "source_video_path", "source_video_sha256", "instruction_utf8_sha256",
    "instruction_utf8_bytes", "accepted_model_conditions", "target_video_argument",
    "target_accessed_by_inference", "external_mask_or_swept_tube",
    "external_tracking_pose_or_trajectory", "reference_image_or_video",
    "external_shared_i0", "source_video_physical_authority",
    "source_video_physical_authority_digest", "retained_source_fd_consumed",
    "source_video_pre_and_post_decode_rehashed",
}
_PREPROCESSING_FIELDS = {
    "frame_count", "fps", "reported_fps", "source_input_hw",
    "source_derived_bucket_hw", "max_pixels", "stride", "temporal_policy",
    "spatial_policy", "resize", "external_shared_i0",
}
_PROMPT_FIELDS = {
    "task", "system_prompt_sha256", "cleaner", "tokenizer_fix_mistral_regex",
    "tokenizer_padding_side", "max_sequence_length", "prompt_enhancer",
}
_SAMPLING_FIELDS = {
    "num_frames", "num_inference_steps", "guidance_mode", "omega_vid",
    "omega_img", "omega_txt", "omega_scale", "flow_shift", "seed", "eta",
    "norm_threshold", "momentum", "single_expert", "ulysses_size",
    "rank0_decode_and_save_only", "source_onset_policy",
}
_OUTPUT_FIELDS = {
    "path", "sha256", "frame_count", "fps", "height", "width",
    "audio_preserved", "size", "publication_identity", "prepublication_identity",
    "anonymous_creation_method", "anonymous_seal_mask", "sealed_source_sha256",
    "sealed_source_size", "anonymous_inode_encoded_and_decoded_before_publication",
    "create_only_copy_publication_after_decode",
    "sealed_source_and_publication_bytes_equal", "retained_inode_encoded_and_replayed",
    "named_output_never_replaced",
}
_MODEL_CONSUMPTION_FIELDS = {
    "consumption_input_digest", "task_input_digest", "model_capture_digest",
    "model_view_root", "adapter_capture_digest", "adapter_view_root",
    "fd_view_files_authorized", "inherited_fd_binding_digest", "inherited_fd_count",
    "ptrace_authorization_used", "source_video_sha256",
    "source_video_physical_authority_digest", "all_ranks_use_retained_source_fd",
    "four_rank_attestation",
}
_ADAPTER_FIELDS = {
    "enabled", "mode", "strictly_reloaded", "safe_merged_for_inference",
    "training_global_step", "profile", "lora_rank", "lora_alpha", "tensor_count",
    "target_module_count", "target_modules_sha256", "checkpoint_root",
    "checkpoint_manifest", "adapter_model_path", "adapter_model_sha256",
    "training_receipt_path", "training_receipt_digest",
}
_TASK_RESULT_FIELDS = {
    "schema_version", "task_index", "task_id", "arm", "plan_digest",
    "task_input_digest", "argv_digest", "environment_digest",
    "ffmpeg_exec_authority_digest", "publication_handoff_authority_digest",
    "publication_handoff_payload_digest", "return_code", "attempt_count",
    "retry_allowed", "model_capture_digest", "adapter_capture_digest",
    "consumption_input_digest", "consumption_digest", "native_receipt_digest",
    "native_receipt_file_sha256", "native_output_sha256", "native_output_size",
    "native_receipt_identity", "native_output_identity", "output_path",
    "receipt_path", "log_basename", "authority_artifacts",
    "native_publication_completed_before_parent_post_use_replay",
    "parent_post_use_closed_before_native_publication", "post_use_replay_complete",
    "task_result_digest",
}
_ARTIFACT_REPLAY_FIELDS = {
    "task_id", "artifact_count", "artifact_rows_digest", "consumption_digest",
    "task_result_digest", "runner_task_file_sha256",
    "native_receipt_file_sha256", "native_receipt_mode", "native_receipt_nlink",
    "native_output_sha256", "publication_authority_digest",
    "publication_handoff_authority_digest", "publication_handoff_payload_digest",
    "retained_receipt_and_output_fds_replayed", "v2_verified_result_cross_linked",
    "all_post_use_artifacts_replayed",
}
_MODEL_FINAL_FIELDS = {
    "schema_version", "model_capture_digest", "task_count",
    "task_consumption_digests", "task_consumption_set_digest",
    "final_rehash_digest", "private_parent_current_identity",
    "all_model_bytes_rehashed_after_last_task",
    "all_model_file_and_directory_fds_retained_through_final_rehash",
    "model_final_digest",
}
_STAT_IDENTITY_FIELDS = {
    "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
    "blocks", "mtime_ns", "ctime_ns",
}
_DIRECTORY_IDENTITY_FIELDS = {"device", "inode", "uid", "gid", "mode", "rdev"}
_PINNED_FILE_IDENTITY_FIELDS = {
    "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
    "nlink",
}
_PHYSICAL_BINDINGS_FIELDS = {
    "schema_version", "plan_path", "plan_sha256", "plan_digest",
    "asset_authority_digest", "allocation", "identities",
    "captured_runner_entry", "captured_runner_entry_required",
    "exec_authority", "exec_authority_retained_source_and_python_fds",
    "ffprobe_authority", "ffprobe_retained_executable_fd",
    "isolated_child_interpreters", "child_environment_exact_allowlist",
    "model_root", "bernini_root", "veomni_root", "campaign_mode",
    "formal_full16_report", "task_count", "task_ids", "retry_allowed",
    "final_artifacts", "physical_bindings_digest",
}
_ALLOCATION_FIELDS = {
    "holder_job_id", "node", "slurm_step_id",
    "slurm_environment_source_names", "slurm_environment_raw_values",
    "slurm_observed_absent_fields", "normalized_slurm_authority",
    "world_size", "ulysses_size", "reserved_gpu_count", "visible_gpu_indices",
}
_PHYSICAL_IDENTITY_ROLES = {
    "runner", "frozen_runner", "exact5_eval", "bridge", "adapter", "eval_v1",
    "eval_v2", "model_authority", "python", "torchrun_source",
    "torchrun_handler_source", "torch_local_agent_source",
    "torch_dynamic_rendezvous_source", "torch_multiprocessing_api_source",
    "model_manifest", "ffmpeg", "ffprobe", "infer_lora",
}
_CAPTURED_ENTRY_FIELDS = {
    "schema_version", "runner_fd", "runner_path", "runner_sha256",
    "runner_identity", "python_fd", "python_path", "python_sha256",
    "python_identity", "release_digest", "bootstrap_sha256", "entry_method",
    "slurm_export_none_required", "bash_privileged_startup_required",
    "captured_source_entry", "authority_digest",
}
_CAPTURED_ENTRY_SUMMARY_FIELDS = {
    "authority_digest", "release_digest", "bootstrap_sha256",
    "captured_source_entry", "held_through_attestation_publication",
}
_EXEC_AUTHORITY_FIELDS = {
    "schema_version", "rows", "rows_digest", "binding_digest",
}
_EXEC_AUTHORITY_ROLES = (
    "python_executable", "bridge_source", "adapter_source", "ffmpeg_executable",
)
_FFPROBE_AUTHORITY_FIELDS = {
    "schema_version", "fd", "source_path", "sha256", "identity",
    "authority_digest",
}
_PINNED_PHYSICAL_SHA256 = {
    "runner": EXACT5_RUNNER_SHA256,
    "frozen_runner": FROZEN_RUNNER_SHA256,
    "exact5_eval": EXACT5_EVAL_SHA256,
    "bridge": "c91de7eb821a05c61f66349c02f9232ede27c49e54659f351f72930fb071d136",
    "adapter": "52dcde8797484504ab28a5c59c532c0877a145ea24762edaf3105a21b0719e19",
    "eval_v1": "d6ef0939a67598e66ccf2652d22520ae3a87a068789f70f921522ba86046138d",
    "eval_v2": "b675b84fd5f1b95a21f6454c9eb8c53b0965d7dbdd3fedf1ea92b6ad153ac982",
    "model_authority": "b9457e434b8000e5368056c925edd0227b4dd3d8a439090494af088817d51ecf",
    "torchrun_source": "1aed399471b08b12c536def56553a6dfe53be234a52e0df48df325c6477f7e8c",
    "torchrun_handler_source": "9871ee801f346c4952fcaf2cc87965f3c997d974b550df70e1fc7f4534c66e87",
    "torch_local_agent_source": "71f390071316417643aa91514ebb170b3adb7eca5c1fe8286d03fe2eef21e497",
    "torch_dynamic_rendezvous_source": "adc34f683614cdc6de5f5cc64e34ee7201b0671609a7ee574b9731f4266e5cec",
    "torch_multiprocessing_api_source": "f815c915fd857bbff12b4d00530c7c1ffb0badfcd48c41e7f378c65828192ef7",
    "model_manifest": "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
    "ffprobe": FFPROBE_SHA256,
    "infer_lora": EXPECTED_PRODUCER["infer_lora_sha256"],
}


class SiteBuildError(RuntimeError):
    """One fail-closed publication check did not pass."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise SiteBuildError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise SiteBuildError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise SiteBuildError(f"non-finite JSON number: {token}")


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def stable_file(path: Path, *, label: str) -> tuple[bytes, str, int]:
    """Read one unchanged, non-symlink regular file."""

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise SiteBuildError(f"{label} is not a plain regular file: {path}")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fd_before = os.fstat(descriptor)
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            size = 0
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                chunks.append(block)
                size += len(block)
            fd_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except SiteBuildError:
        raise
    except OSError as error:
        raise SiteBuildError(f"cannot stably read {label}: {path}: {error}") from error
    if not (
        _identity(before) == _identity(fd_before)
        == _identity(fd_after) == _identity(after)
    ) or size != before.st_size:
        raise SiteBuildError(f"{label} changed while being read: {path}")
    return b"".join(chunks), digest.hexdigest(), size


def load_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, int]:
    raw, sha256, size = stable_file(path, label=label)
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, SiteBuildError) as error:
        raise SiteBuildError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise SiteBuildError(f"{label} root is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise SiteBuildError(f"{label} is not canonical JSON plus one LF")
    return value, sha256, size


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SiteBuildError(f"{label} is not one lowercase SHA-256")
    return value


def require_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(value)
    claimed = unsigned.pop(field, None)
    if not isinstance(claimed, str) or claimed != object_sha256(unsigned):
        raise SiteBuildError(f"{label} canonical {field} differs")
    return claimed


def require_true(value: Any, *, label: str) -> None:
    if type(value) is not bool or value is not True:
        raise SiteBuildError(f"{label} must be true")


def require_false(value: Any, *, label: str) -> None:
    if type(value) is not bool or value is not False:
        raise SiteBuildError(f"{label} must be false")


def require_stat_identity(
    value: Any, *, label: str, permissions: int | None = None,
    nlink: int | None = None, size: int | None = None,
) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _STAT_IDENTITY_FIELDS
        or any(type(value.get(field)) is not int for field in _STAT_IDENTITY_FIELDS)
        or not stat.S_ISREG(value["mode"])
        or (permissions is not None and stat.S_IMODE(value["mode"]) != permissions)
        or (nlink is not None and value["nlink"] != nlink)
        or (size is not None and value["size"] != size)
    ):
        raise SiteBuildError(f"{label} stat identity differs")
    return dict(value)


def require_directory_identity(value: Any, *, label: str) -> dict[str, int]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _DIRECTORY_IDENTITY_FIELDS
        or any(type(value.get(field)) is not int for field in _DIRECTORY_IDENTITY_FIELDS)
        or not stat.S_ISDIR(value["mode"])
    ):
        raise SiteBuildError(f"{label} directory identity differs")
    return dict(value)


def require_absolute_path(value: Any, *, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or not Path(value).is_absolute()
        or os.path.normpath(value) != value
    ):
        raise SiteBuildError(f"{label} is not a canonical absolute path")
    return Path(value)


def _plain_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise SiteBuildError(f"missing {label}: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SiteBuildError(f"{label} is not a plain directory: {path}")


def validate_asset_authority(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != _ASSET_AUTHORITY_FIELDS:
        raise SiteBuildError("plan asset authority root schema differs")
    require_digest(value, "authority_digest", label="asset authority")
    sources = value.get("sources")
    if (
        value.get("schema_version") != "case01-source-bone-exact5-asset-authority-v1"
        or value.get("status")
        != "APPROVED_FOR_EXACT5_R64_RENDERER_CANARY"
        or value.get("launch_allowed") is not True
        or value.get("independent_visual_audit_status") != "PASS_P0_0_P1_0"
        or value.get("manifest_sha256")
        != "0a62b74056f4be1ab17ed632d31068964aed27c607212f58c2a7d17b74becf5e"
        or value.get("manifest_size") != 249_082
        or value.get("manifest_digest")
        != "879318860b7d96824ec2da4b10b657b320945285a1607faf8c89bb577a1cc538"
        or value.get("independent_audit_receipt_sha256")
        != "040c53a3647ae957212a1d2d6da3ffa75b4207ace07e1c7ba6ce128033dce969"
        or value.get("independent_audit_receipt_size") != 8_285
        or value.get("independent_audit_receipt_digest")
        != "13ea77d95e8529585f1bcda1ff5fc9b1f71a42062adfa2994c6dfbe51d22d7d1"
        or value.get("iid") != IID
        or not isinstance(value.get("manifest_path"), str)
        or not Path(value["manifest_path"]).is_absolute()
        or not isinstance(value.get("independent_audit_receipt_path"), str)
        or not Path(value["independent_audit_receipt_path"]).is_absolute()
        or not isinstance(sources, list)
        or len(sources) != 5
        or value.get("source_rows_digest") != object_sha256(sources)
    ):
        raise SiteBuildError("plan asset authority approval differs")
    checked: list[dict[str, Any]] = []
    for source, expected in zip(sources, VARIANTS):
        if (
            not isinstance(source, dict)
            or set(source) != {
                "variant", "path", "sha256", "size", "geometry", "treatment",
                "bone_present", "bone_position", "visual_audit_status",
            }
            or source.get("variant") != expected["id"]
            or not isinstance(source.get("path"), str)
            or not Path(source["path"]).is_absolute()
            or os.path.normpath(source["path"]) != source["path"]
            or Path(source["path"]).name != f"{expected['id']}.mp4"
            or source.get("sha256") != expected["source_sha256"]
            or source.get("size") != expected["source_size"]
            or source.get("geometry") != EXPECTED_SOURCE_VIDEO
            or source.get("treatment") != expected["treatment"]
            or source.get("bone_present") is not expected["bone_present"]
            or source.get("bone_position") != expected["bone_position"]
            or source.get("visual_audit_status") != "PASS"
        ):
            raise SiteBuildError(f"asset authority source differs: {expected['id']}")
        checked.append(dict(source))
    return checked


def validate_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = {
        "schema_version", "experiment_id", "production_ready", "launch_allowed",
        "asset_authority", "checkpoint_manifest", "producer", "condition_contract",
        "arms", "task_count", "tasks", "claim_limits", "plan_digest",
    }
    if not isinstance(plan, Mapping) or set(plan) != fields:
        raise SiteBuildError("plan root schema differs")
    require_digest(plan, "plan_digest", label="plan")
    sources = validate_asset_authority(plan.get("asset_authority", {}))
    checkpoint = plan.get("checkpoint_manifest")
    producer = plan.get("producer")
    condition = plan.get("condition_contract")
    tasks = plan.get("tasks")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("experiment_id") != EXPERIMENT_ID
        or plan.get("production_ready") is not True
        or plan.get("launch_allowed") is not True
        or plan.get("arms") != ["full644"]
        or plan.get("task_count") != 5
        or plan.get("claim_limits") != CLAIM_LIMITS
        or not isinstance(checkpoint, Mapping)
        or set(checkpoint) != set(EXPECTED_CHECKPOINT) | {"path"}
        or any(checkpoint.get(key) != item for key, item in EXPECTED_CHECKPOINT.items())
        or not isinstance(checkpoint.get("path"), str)
        or not Path(checkpoint["path"]).is_absolute()
        or not isinstance(producer, Mapping)
        or set(producer) != set(EXPECTED_PRODUCER) | {"infer_lora_path", "ffprobe_path"}
        or any(producer.get(key) != item for key, item in EXPECTED_PRODUCER.items())
        or not isinstance(producer.get("infer_lora_path"), str)
        or not Path(producer["infer_lora_path"]).is_absolute()
        or not isinstance(producer.get("ffprobe_path"), str)
        or not Path(producer["ffprobe_path"]).is_absolute()
        or condition != {
            "iid": IID, "instruction": INSTRUCTION,
            "instruction_sha256": INSTRUCTION_SHA256, "seed": SEED,
            "num_inference_steps": 40, "source_onset_policy": "none",
            "same_sampler_all_tasks": True,
            "same_model_capture_all_tasks_required": True,
            "codec_only_present_control_required": True,
        }
        or not isinstance(tasks, list)
        or len(tasks) != 5
        or [task.get("task_id") for task in tasks if isinstance(task, Mapping)]
        != list(TASK_IDS)
    ):
        raise SiteBuildError("plan identity/current-R64 closure differs")
    cases: list[dict[str, Any]] = []
    for index, (task, variant, source) in enumerate(
        zip(tasks, VARIANTS, sources)
    ):
        adapter = task.get("adapter") if isinstance(task, Mapping) else None
        output = task.get("output") if isinstance(task, Mapping) else None
        task_id = TASK_IDS[index]
        if (
            not isinstance(task, Mapping)
            or set(task) != {
                "task_id", "case_index", "iid", "intervention_variant",
                "source_video", "source_video_sha256", "instruction",
                "instruction_sha256", "seed", "num_inference_steps",
                "source_onset_policy", "arm", "adapter", "output",
            }
            or task.get("task_id") != task_id
            or task.get("case_index") != 1
            or task.get("iid") != IID
            or task.get("intervention_variant") != variant["id"]
            or task.get("source_video") != source["path"]
            or task.get("source_video_sha256") != variant["source_sha256"]
            or task.get("instruction") != INSTRUCTION
            or task.get("instruction_sha256") != INSTRUCTION_SHA256
            or task.get("seed") != SEED
            or task.get("num_inference_steps") != 40
            or task.get("source_onset_policy") != "none"
            or task.get("arm") != "full644"
            or not isinstance(adapter, Mapping)
            or set(adapter) != {
                "checkpoint_root", "checkpoint_manifest", "adapter_model_sha256",
                "profile",
            }
            or adapter.get("profile") != PROFILE
            or adapter.get("adapter_model_sha256")
            != EXPECTED_CHECKPOINT["adapter_model_sha256"]
            or adapter.get("checkpoint_manifest") != checkpoint
            or adapter.get("checkpoint_root") != str(Path(checkpoint["path"]).parent)
            or not isinstance(output, Mapping)
            or set(output) != {"video_path", "receipt_path", "create_only"}
            or output.get("create_only") is not True
            or not isinstance(output.get("video_path"), str)
            or not isinstance(output.get("receipt_path"), str)
            or not Path(output["video_path"]).is_absolute()
            or not Path(output["receipt_path"]).is_absolute()
            or os.path.normpath(output["video_path"]) != output["video_path"]
            or os.path.normpath(output["receipt_path"]) != output["receipt_path"]
            or Path(output["video_path"]).name != f"{task_id}.mp4"
            or Path(output["receipt_path"]).name != f"{task_id}.mp4.receipt.json"
            or Path(output["receipt_path"])
            != Path(output["video_path"]).with_name(f"{task_id}.mp4.receipt.json")
        ):
            raise SiteBuildError(f"plan task closure differs: {task_id}")
        cases.append({
            **variant, "task": dict(task), "task_id": task_id,
            "plan_digest": plan["plan_digest"],
        })
    source_paths = [case["task"]["source_video"] for case in cases]
    publication_paths = [
        case["task"]["output"][field]
        for case in cases for field in ("video_path", "receipt_path")
    ]
    if (
        len(set(source_paths)) != 5
        or len(set(publication_paths)) != 10
        or len({str(Path(path).parent) for path in publication_paths}) != 1
        or set(source_paths) & set(publication_paths)
    ):
        raise SiteBuildError("plan source/publication leaf closure differs")
    return cases


def validate_report(
    report: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    fields = {
        "schema_version", "status", "campaign_mode", "plan_schema_version",
        "plan_digest", "task_count", "task_ids", "variant_order",
        "all_exact5_tasks_verified_no_cherry_pick", "same_sampler_all_tasks",
        "same_prompt_contract_all_tasks", "same_model_capture_all_tasks",
        "deterministic_reference_parity", "codec_only_control_interpretation",
        "retained_publication_root_fd_replayed",
        "retained_ffprobe_executable_fd_replayed",
        "retained_publication_leaf_fds_replayed", "manual_blind_review_required",
        "formal_full16_report", "results", "claim_limits", "report_digest",
    }
    if not isinstance(report, Mapping) or set(report) != fields:
        raise SiteBuildError("exact5 report root schema differs")
    require_digest(report, "report_digest", label="exact5 report")
    for field in (
        "all_exact5_tasks_verified_no_cherry_pick", "same_sampler_all_tasks",
        "same_prompt_contract_all_tasks", "same_model_capture_all_tasks",
        "retained_publication_root_fd_replayed",
        "retained_ffprobe_executable_fd_replayed",
        "retained_publication_leaf_fds_replayed", "manual_blind_review_required",
    ):
        require_true(report.get(field), label=f"report {field}")
    require_false(report.get("formal_full16_report"), label="report formal_full16_report")
    parity = report.get("deterministic_reference_parity")
    rows = report.get("results")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW"
        or report.get("campaign_mode") != CAMPAIGN
        or report.get("plan_schema_version") != PLAN_SCHEMA
        or report.get("plan_digest") != plan.get("plan_digest")
        or report.get("task_count") != 5
        or report.get("task_ids") != list(TASK_IDS)
        or report.get("variant_order") != list(VARIANT_ORDER)
        or report.get("claim_limits") != CLAIM_LIMITS
        or parity != {
            "policy": "HARD_FAIL", "variant": "exact_original",
            "reference_output_sha256": REFERENCE_OUTPUT_SHA256,
            "observed_output_sha256": REFERENCE_OUTPUT_SHA256,
            "status": "PASS",
            "kept_separate_from_intervention_effect_interpretation": True,
        }
        or report.get("codec_only_control_interpretation")
        != "isolates matched transcode/container effects from source-object treatment"
        or not isinstance(rows, list)
        or len(rows) != 5
    ):
        raise SiteBuildError("exact5 report identity/parity closure differs")
    results: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        task_id = TASK_IDS[index]
        task_output = plan["tasks"][index]["output"]
        probe = row.get("media_probe") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "task_id", "arm", "receipt_path", "receipt_file_sha256",
                "receipt_digest", "output_path", "output_sha256", "output_size",
                "media_probe",
            }
            or row.get("task_id") != task_id
            or row.get("arm") != "full644"
            or row.get("output_path") != task_output["video_path"]
            or row.get("receipt_path") != task_output["receipt_path"]
            or Path(str(row.get("output_path", ""))).name != f"{task_id}.mp4"
            or Path(str(row.get("receipt_path", ""))).name
            != f"{task_id}.mp4.receipt.json"
            or type(row.get("output_size")) is not int
            or row["output_size"] <= 0
            or not isinstance(probe, Mapping)
            or set(probe) != {
                "ffprobe_path", "ffprobe_sha256", "ffprobe_size", "stream_count",
                "frame_count", "fps_num", "fps_den", "width", "height",
            }
            or probe.get("ffprobe_sha256") != FFPROBE_SHA256
            or probe.get("stream_count") != 1
            or probe.get("frame_count") != 81
            or probe.get("fps_num") != 25
            or probe.get("fps_den") != 1
            or type(probe.get("width")) is not int or probe["width"] <= 0
            or type(probe.get("height")) is not int or probe["height"] <= 0
        ):
            raise SiteBuildError(f"report result closure differs: {task_id}")
        require_sha256(row.get("output_sha256"), label=f"{task_id} output SHA")
        require_sha256(
            row.get("receipt_file_sha256"), label=f"{task_id} receipt file SHA"
        )
        require_sha256(row.get("receipt_digest"), label=f"{task_id} receipt digest")
        if index == 0 and row.get("output_sha256") != REFERENCE_OUTPUT_SHA256:
            raise SiteBuildError("exact_original output bytes fail frozen R64 parity")
        results[task_id] = dict(row)
    return results


def validate_physical_bindings(
    value: Any, *, plan: Mapping[str, Any], plan_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PHYSICAL_BINDINGS_FIELDS:
        raise SiteBuildError("physical bindings root schema differs")
    require_digest(value, "physical_bindings_digest", label="physical bindings")
    allocation = value.get("allocation")
    identities = value.get("identities")
    entry = value.get("captured_runner_entry")
    exec_authority = value.get("exec_authority")
    ffprobe = value.get("ffprobe_authority")
    final_artifacts = value.get("final_artifacts")
    if (
        value.get("schema_version")
        != "case01-source-bone-exact5-physical-bindings-v1"
        or value.get("plan_sha256") != plan_sha256
        or value.get("plan_digest") != plan.get("plan_digest")
        or value.get("asset_authority_digest")
        != plan["asset_authority"]["authority_digest"]
        or value.get("captured_runner_entry_required") is not True
        or value.get("exec_authority_retained_source_and_python_fds") is not True
        or value.get("ffprobe_retained_executable_fd") is not True
        or value.get("isolated_child_interpreters") != "-I -S -B"
        or value.get("child_environment_exact_allowlist") is not True
        or value.get("campaign_mode") != CAMPAIGN
        or value.get("formal_full16_report") is not False
        or value.get("task_count") != 5
        or value.get("task_ids") != list(TASK_IDS)
        or value.get("retry_allowed") is not False
    ):
        raise SiteBuildError("physical bindings exact5 closure differs")
    plan_path = require_absolute_path(value.get("plan_path"), label="physical plan path")
    for field in ("model_root", "bernini_root", "veomni_root"):
        require_absolute_path(value.get(field), label=f"physical {field}")
    if (
        not isinstance(final_artifacts, Mapping)
        or set(final_artifacts) != {"output_report", "runner_attestation"}
    ):
        raise SiteBuildError("physical final artifact closure differs")
    final_paths = {
        label: require_absolute_path(path, label=f"physical final {label}")
        for label, path in final_artifacts.items()
    }
    if (
        len(set(final_paths.values())) != 2
        or final_paths["output_report"].name != REPORT_REL.name
        or final_paths["runner_attestation"].name != ATTESTATION_REL.name
    ):
        raise SiteBuildError("physical final artifact identity differs")

    if not isinstance(allocation, Mapping) or set(allocation) != _ALLOCATION_FIELDS:
        raise SiteBuildError("physical allocation schema differs")
    source_names = allocation.get("slurm_environment_source_names")
    raw_values = allocation.get("slurm_environment_raw_values")
    normalized = allocation.get("normalized_slurm_authority")
    expected_source_names = {
        "job_id": "SLURM_JOB_ID", "step_id": "SLURM_STEP_ID",
        "gpu_count": "SLURM_GPUS_ON_NODE",
        "gpus_per_node": "SLURM_GPUS_PER_NODE",
        "step_gpu_indices": "SLURM_STEP_GPUS",
        "job_node_count": "SLURM_NNODES",
        "step_node_count": "SLURM_STEP_NUM_NODES",
        "job_nodelist": "SLURM_JOB_NODELIST",
        "step_nodelist": "SLURM_STEP_NODELIST",
    }
    expected_raw_keys = set(expected_source_names.values())
    holder = allocation.get("holder_job_id")
    node = allocation.get("node")
    step = allocation.get("slurm_step_id")
    if (
        source_names != expected_source_names
        or not isinstance(raw_values, Mapping)
        or set(raw_values) != expected_raw_keys
        or not isinstance(normalized, Mapping)
        or set(normalized) != {
            "job_node_count", "step_node_count", "gpu_count_on_node",
            "gpus_per_node", "step_gpu_indices", "job_node", "step_node",
        }
        or not isinstance(holder, str) or not holder.isdecimal()
        or str(int(holder)) != holder
        or not isinstance(step, str) or not step.isdecimal()
        or int(step) <= 0 or str(int(step)) != step
        or not isinstance(node, str) or not node or node.strip() != node
        or raw_values.get("SLURM_JOB_ID") != holder
        or raw_values.get("SLURM_STEP_ID") != step
        or raw_values.get("SLURM_GPUS_ON_NODE") != "8"
        or raw_values.get("SLURM_GPUS_PER_NODE") != "8"
        or raw_values.get("SLURM_STEP_GPUS") != "0,1,2,3,4,5,6,7"
        or raw_values.get("SLURM_NNODES") != "1"
        or raw_values.get("SLURM_STEP_NUM_NODES") != "1"
        or raw_values.get("SLURM_JOB_NODELIST") != node
        or raw_values.get("SLURM_STEP_NODELIST") != node
        or allocation.get("slurm_observed_absent_fields")
        != ["SLURM_JOB_GPUS", "SLURM_JOB_NUM_NODES"]
        or normalized != {
            "job_node_count": 1, "step_node_count": 1,
            "gpu_count_on_node": 8, "gpus_per_node": 8,
            "step_gpu_indices": list(range(8)), "job_node": node,
            "step_node": node,
        }
        or allocation.get("world_size") != 4
        or allocation.get("ulysses_size") != 4
        or allocation.get("reserved_gpu_count") != 8
        or allocation.get("visible_gpu_indices") != [0, 1, 2, 3]
    ):
        raise SiteBuildError("physical allocation value differs")

    if not isinstance(identities, Mapping) or set(identities) != _PHYSICAL_IDENTITY_ROLES:
        raise SiteBuildError("physical pinned identity roles differ")
    checked_identities: dict[str, dict[str, Any]] = {}
    for role, row in identities.items():
        if (
            not isinstance(row, Mapping)
            or set(row) != _PINNED_FILE_IDENTITY_FIELDS
            or type(row.get("size")) is not int or row["size"] <= 0
            or type(row.get("mode")) is not int
            or type(row.get("device")) is not int
            or type(row.get("inode")) is not int
            or type(row.get("uid")) is not int
            or type(row.get("gid")) is not int
            or row.get("nlink") != 1
        ):
            raise SiteBuildError(f"physical pinned identity differs: {role}")
        require_absolute_path(row.get("path"), label=f"physical {role} path")
        require_sha256(row.get("sha256"), label=f"physical {role} SHA")
        if role in _PINNED_PHYSICAL_SHA256 and row["sha256"] != _PINNED_PHYSICAL_SHA256[role]:
            raise SiteBuildError(f"physical source pin differs: {role}")
        checked_identities[role] = dict(row)
    if (
        checked_identities["ffprobe"]["path"] != plan["producer"]["ffprobe_path"]
        or checked_identities["infer_lora"]["path"]
        != plan["producer"]["infer_lora_path"]
    ):
        raise SiteBuildError("physical plan producer identity differs")

    if not isinstance(entry, Mapping) or set(entry) != _CAPTURED_ENTRY_FIELDS:
        raise SiteBuildError("captured runner entry schema differs")
    require_digest(entry, "authority_digest", label="captured runner entry")
    runner_identity = require_stat_identity(
        entry.get("runner_identity"), label="captured runner identity",
        permissions=0o444, nlink=1,
    )
    python_identity = require_stat_identity(
        entry.get("python_identity"), label="captured Python identity", nlink=1,
    )
    runner_fd = entry.get("runner_fd")
    python_fd = entry.get("python_fd")
    for field in ("release_digest", "bootstrap_sha256"):
        require_sha256(entry.get(field), label=f"captured entry {field}")
    if (
        entry.get("schema_version")
        != "full644-exploratory-matched-captured-runner-entry-authority-v1"
        or type(runner_fd) is not int or runner_fd < 3
        or type(python_fd) is not int or python_fd < 3 or runner_fd == python_fd
        or entry.get("runner_path") != checked_identities["runner"]["path"]
        or entry.get("runner_sha256") != EXACT5_RUNNER_SHA256
        or entry.get("python_path") != checked_identities["python"]["path"]
        or entry.get("python_sha256") != checked_identities["python"]["sha256"]
        or entry.get("entry_method")
        != "slurm-spooled-or-trusted-stdin-held-python-fd-v1"
        or entry.get("slurm_export_none_required") is not True
        or entry.get("bash_privileged_startup_required") is not True
        or entry.get("captured_source_entry") is not True
        or any(
            identity[field] != checked_identities[role][field]
            for identity, role in (
                (runner_identity, "runner"), (python_identity, "python")
            )
            for field in ("device", "inode", "uid", "gid", "nlink", "size")
        )
        or stat.S_IMODE(runner_identity["mode"])
        != checked_identities["runner"]["mode"]
        or stat.S_IMODE(python_identity["mode"])
        != checked_identities["python"]["mode"]
        or not python_identity["mode"] & 0o111
    ):
        raise SiteBuildError("captured runner entry value differs")

    if not isinstance(exec_authority, Mapping) or set(exec_authority) != _EXEC_AUTHORITY_FIELDS:
        raise SiteBuildError("retained exec authority schema differs")
    require_digest(exec_authority, "binding_digest", label="retained exec authority")
    exec_rows = exec_authority.get("rows")
    if (
        exec_authority.get("schema_version")
        != "full644-exploratory-matched-exec-authority-v2"
        or not isinstance(exec_rows, list)
        or [row.get("role") if isinstance(row, Mapping) else None for row in exec_rows]
        != list(_EXEC_AUTHORITY_ROLES)
        or exec_authority.get("rows_digest") != object_sha256(exec_rows)
    ):
        raise SiteBuildError("retained exec authority digest differs")
    physical_roles = ("python", "bridge", "adapter", "ffmpeg")
    exec_fds: list[int] = []
    for row, physical_role in zip(exec_rows, physical_roles):
        if not isinstance(row, Mapping) or set(row) != {"role", "fd", "source_path", "sha256", "identity"}:
            raise SiteBuildError("retained exec authority row differs")
        identity = require_stat_identity(
            row.get("identity"), label=f"retained exec {physical_role}", nlink=1,
        )
        fd = row.get("fd")
        physical = checked_identities[physical_role]
        if (
            type(fd) is not int or fd < 3
            or row.get("source_path") != physical["path"]
            or row.get("sha256") != physical["sha256"]
            or any(identity[field] != physical[field] for field in (
                "device", "inode", "uid", "gid", "nlink", "size"
            ))
            or stat.S_IMODE(identity["mode"]) != physical["mode"]
            or (physical_role in {"python", "ffmpeg"} and not identity["mode"] & 0o111)
        ):
            raise SiteBuildError(f"retained exec cross-link differs: {physical_role}")
        exec_fds.append(fd)
    if exec_fds != sorted(exec_fds) or len(set(exec_fds)) != 4:
        raise SiteBuildError("retained exec FD order differs")

    if not isinstance(ffprobe, Mapping) or set(ffprobe) != _FFPROBE_AUTHORITY_FIELDS:
        raise SiteBuildError("retained ffprobe authority schema differs")
    require_digest(ffprobe, "authority_digest", label="retained ffprobe authority")
    ffprobe_identity = require_stat_identity(
        ffprobe.get("identity"), label="retained ffprobe identity", nlink=1,
    )
    physical_ffprobe = checked_identities["ffprobe"]
    if (
        ffprobe.get("schema_version")
        != "bernini-full644-exploratory-matched-ffprobe-exec-authority-v1"
        or type(ffprobe.get("fd")) is not int or ffprobe["fd"] < 3
        or ffprobe.get("source_path") != physical_ffprobe["path"]
        or ffprobe.get("sha256") != physical_ffprobe["sha256"]
        or any(ffprobe_identity[field] != physical_ffprobe[field] for field in (
            "device", "inode", "uid", "gid", "nlink", "size"
        ))
        or stat.S_IMODE(ffprobe_identity["mode"]) != physical_ffprobe["mode"]
        or not ffprobe_identity["mode"] & 0o111
    ):
        raise SiteBuildError("retained ffprobe authority value differs")
    return {
        "plan_path": str(plan_path), "identities": checked_identities,
        "captured_runner_entry": dict(entry), "ffprobe_authority": dict(ffprobe),
        "ffmpeg_exec_authority_digest": object_sha256(exec_rows[3]),
        "final_artifacts": {key: str(path) for key, path in final_paths.items()},
    }


def validate_attestation(
    attestation: Mapping[str, Any], *, plan: Mapping[str, Any], plan_sha256: str,
    report: Mapping[str, Any], report_sha256: str,
) -> None:
    fields = {
        "schema_version", "status", "campaign_mode", "formal_full16_report",
        "manual_blind_review_required", "plan", "physical_bindings",
        "captured_runner_entry", "retained_publication_root",
        "retained_ffprobe_executable", "retained_task_publications",
        "retained_child_publication_handoffs", "retained_final_parents",
        "task_count", "task_ids", "unselected_task_ids", "unselected_task_count",
        "all_exact5_tasks_attempted_exactly_once", "all_exact5_tasks_succeeded",
        "retry_count", "task_result_digests", "task_environment_digests",
        "ffmpeg_exec_authority_digest",
        "all_rank0_encoders_used_retained_ffmpeg_executable", "task_results",
        "task_artifact_replays", "runner_task_json_replayed_for_all_tasks",
        "native_publication_before_parent_post_use_replay",
        "all_model_adapter_post_use_replays_complete",
        "native_receipts_replayed_0400_single_link", "model_capture_digest",
        "same_model_capture_all_exact5_tasks", "model_final", "verified_report",
        "reused_frozen_execution_contract", "exploratory_only",
        "scientific_claim_authorized", "formal_claim_authorized",
        "attestation_digest",
    }
    if not isinstance(attestation, Mapping) or set(attestation) != fields:
        raise SiteBuildError("runner attestation root schema differs")
    require_digest(attestation, "attestation_digest", label="runner attestation")
    for field in (
        "manual_blind_review_required", "all_exact5_tasks_attempted_exactly_once",
        "all_exact5_tasks_succeeded",
        "all_rank0_encoders_used_retained_ffmpeg_executable",
        "runner_task_json_replayed_for_all_tasks",
        "native_publication_before_parent_post_use_replay",
        "all_model_adapter_post_use_replays_complete",
        "native_receipts_replayed_0400_single_link",
        "same_model_capture_all_exact5_tasks", "exploratory_only",
    ):
        require_true(attestation.get(field), label=f"attestation {field}")
    for field in ("formal_full16_report", "scientific_claim_authorized", "formal_claim_authorized"):
        require_false(attestation.get(field), label=f"attestation {field}")
    plan_binding = attestation.get("plan")
    report_binding = attestation.get("verified_report")
    if (
        attestation.get("schema_version") != ATTESTATION_SCHEMA
        or attestation.get("schema_version") == FAILURE_SCHEMA
        or attestation.get("status") != "EXACT5_COMPLETE_AWAITING_BLIND_REVIEW"
        or attestation.get("campaign_mode") != CAMPAIGN
        or attestation.get("task_count") != 5
        or attestation.get("task_ids") != list(TASK_IDS)
        or attestation.get("unselected_task_ids") != []
        or attestation.get("unselected_task_count") != 0
        or attestation.get("retry_count") != 0
        or not isinstance(plan_binding, Mapping)
        or set(plan_binding) != {"path", "sha256", "plan_digest"}
        or plan_binding.get("sha256") != plan_sha256
        or plan_binding.get("plan_digest") != plan.get("plan_digest")
        or not isinstance(report_binding, Mapping)
        or set(report_binding) != {"path", "sha256", "report_digest", "verified_task_count"}
        or report_binding.get("sha256") != report_sha256
        or report_binding.get("report_digest") != report.get("report_digest")
        or report_binding.get("verified_task_count") != 5
    ):
        raise SiteBuildError("runner attestation plan/report binding differs")

    physical = validate_physical_bindings(
        attestation.get("physical_bindings"), plan=plan, plan_sha256=plan_sha256,
    )
    captured_summary = attestation.get("captured_runner_entry")
    captured_full = physical["captured_runner_entry"]
    if (
        not isinstance(captured_summary, Mapping)
        or set(captured_summary) != _CAPTURED_ENTRY_SUMMARY_FIELDS
        or captured_summary.get("authority_digest")
        != captured_full["authority_digest"]
        or captured_summary.get("release_digest") != captured_full["release_digest"]
        or captured_summary.get("bootstrap_sha256")
        != captured_full["bootstrap_sha256"]
        or captured_summary.get("captured_source_entry") is not True
        or captured_summary.get("held_through_attestation_publication") is not True
        or plan_binding.get("path") != physical["plan_path"]
        or report_binding.get("path")
        != physical["final_artifacts"]["output_report"]
        or attestation.get("ffmpeg_exec_authority_digest")
        != physical["ffmpeg_exec_authority_digest"]
    ):
        raise SiteBuildError("runner captured entry/final binding differs")
    physical_ffprobe = physical["ffprobe_authority"]
    for result in report["results"]:
        probe = result["media_probe"]
        if (
            probe.get("ffprobe_path") != physical_ffprobe["source_path"]
            or probe.get("ffprobe_sha256") != physical_ffprobe["sha256"]
            or probe.get("ffprobe_size") != physical_ffprobe["identity"]["size"]
        ):
            raise SiteBuildError("report/physical ffprobe binding differs")

    retained_root = attestation.get("retained_publication_root")
    retained_ffprobe = attestation.get("retained_ffprobe_executable")
    retained_tasks = attestation.get("retained_task_publications")
    handoffs = attestation.get("retained_child_publication_handoffs")
    retained_final = attestation.get("retained_final_parents")
    publication_parent = str(Path(plan["tasks"][0]["output"]["video_path"]).parent)
    if isinstance(retained_root, Mapping):
        require_directory_identity(
            retained_root.get("immutable_identity"),
            label="retained publication root",
        )
    if (
        not isinstance(retained_root, Mapping)
        or set(retained_root) != {
            "path", "fd", "immutable_identity",
            "held_through_attestation_publication",
        }
        or retained_root.get("path") != publication_parent
        or type(retained_root.get("fd")) is not int or retained_root["fd"] < 3
        or retained_root.get("held_through_attestation_publication") is not True
        or not isinstance(retained_ffprobe, Mapping)
        or set(retained_ffprobe) != {
            "authority_digest", "fd", "source_path", "sha256",
            "held_through_result_verification",
        }
        or retained_ffprobe.get("authority_digest")
        != physical_ffprobe["authority_digest"]
        or retained_ffprobe.get("fd") != physical_ffprobe["fd"]
        or retained_ffprobe.get("source_path") != physical_ffprobe["source_path"]
        or retained_ffprobe.get("sha256") != physical_ffprobe["sha256"]
        or retained_ffprobe.get("held_through_result_verification") is not True
        or not isinstance(retained_tasks, Mapping)
        or set(retained_tasks) != set(TASK_IDS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {
                "authority_digest", "receipt_fd", "output_fd",
                "held_through_result_verification",
            }
            or SHA256_RE.fullmatch(str(row.get("authority_digest", ""))) is None
            or type(row.get("receipt_fd")) is not int or row["receipt_fd"] < 3
            or type(row.get("output_fd")) is not int or row["output_fd"] < 3
            or row["receipt_fd"] == row["output_fd"]
            or row.get("held_through_result_verification") is not True
            for row in retained_tasks.values()
        )
        or not isinstance(handoffs, Mapping)
        or set(handoffs) != set(TASK_IDS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {
                "authority_digest", "fd", "payload_digest",
                "held_sealed_through_attestation",
            }
            or SHA256_RE.fullmatch(str(row.get("authority_digest", ""))) is None
            or SHA256_RE.fullmatch(str(row.get("payload_digest", ""))) is None
            or type(row.get("fd")) is not int or row["fd"] < 3
            or row.get("held_sealed_through_attestation") is not True
            for row in handoffs.values()
        )
        or not isinstance(retained_final, Mapping)
        or set(retained_final) != {"output_report", "runner_attestation"}
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"path", "fd", "immutable_identity"}
            or row.get("path")
            != str(Path(physical["final_artifacts"][label]).parent)
            or type(row.get("fd")) is not int or row["fd"] < 3
            or not isinstance(row.get("immutable_identity"), Mapping)
            or set(row["immutable_identity"]) != _DIRECTORY_IDENTITY_FIELDS
            or any(
                type(row["immutable_identity"].get(field)) is not int
                for field in _DIRECTORY_IDENTITY_FIELDS
            )
            or not stat.S_ISDIR(row["immutable_identity"].get("mode", 0))
            for label, row in retained_final.items()
        )
    ):
        raise SiteBuildError("runner retained-publication authority differs")

    task_results = attestation.get("task_results")
    artifact_rows = attestation.get("task_artifact_replays")
    task_digests = attestation.get("task_result_digests")
    environment_digests = attestation.get("task_environment_digests")
    if not all(isinstance(rows, list) and len(rows) == 5 for rows in (
        task_results, artifact_rows, task_digests, environment_digests,
    )):
        raise SiteBuildError("runner task evidence count differs")
    model_capture = require_sha256(
        attestation.get("model_capture_digest"), label="attestation model capture"
    )
    for index, task_id in enumerate(TASK_IDS):
        row = task_results[index]
        artifact = artifact_rows[index]
        result = report["results"][index]
        retained_task = retained_tasks[task_id]
        retained_handoff = handoffs[task_id]
        authority_artifacts = row.get("authority_artifacts") if isinstance(row, Mapping) else None
        receipt_identity = row.get("native_receipt_identity") if isinstance(row, Mapping) else None
        output_identity = row.get("native_output_identity") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != _TASK_RESULT_FIELDS
            or not isinstance(authority_artifacts, Mapping)
            or set(authority_artifacts) != {
                "model_capture", "model_pre_use", "consumption_input",
                "adapter_capture", "adapter_pre_use", "adapter_post_use",
                "adapter_final", "model_post_use", "eval_consumption_chain",
            }
            or not isinstance(artifact, Mapping)
            or set(artifact) != _ARTIFACT_REPLAY_FIELDS
        ):
            raise SiteBuildError(f"runner task evidence schema differs: {task_id}")
        expected_task_input = object_sha256({
            "schema_version": "full644-exploratory-matched-task-input-v2",
            "plan_digest": plan["plan_digest"], "task": plan["tasks"][index],
        })
        digest_fields = (
            "task_input_digest", "argv_digest", "environment_digest",
            "ffmpeg_exec_authority_digest",
            "publication_handoff_authority_digest",
            "publication_handoff_payload_digest", "model_capture_digest",
            "adapter_capture_digest", "consumption_input_digest",
            "consumption_digest", "native_receipt_digest",
            "native_receipt_file_sha256", "native_output_sha256",
        )
        for field in digest_fields:
            require_sha256(row.get(field), label=f"{task_id} {field}")
        require_sha256(
            artifact.get("artifact_rows_digest"),
            label=f"{task_id} artifact rows digest",
        )
        require_sha256(
            artifact.get("runner_task_file_sha256"),
            label=f"{task_id} runner-task file SHA",
        )
        receipt_identity_checked = require_stat_identity(
            receipt_identity, label=f"{task_id} native receipt identity",
            permissions=0o400, nlink=1,
        )
        output_identity_checked = require_stat_identity(
            output_identity, label=f"{task_id} native output identity",
            permissions=0o444, nlink=1, size=result["output_size"],
        )
        if (
            receipt_identity_checked["size"] <= 0
            or receipt_identity_checked["uid"] != output_identity_checked["uid"]
            or receipt_identity_checked["gid"] != output_identity_checked["gid"]
            or output_identity_checked["uid"]
            != retained_root["immutable_identity"]["uid"]
            or output_identity_checked["gid"]
            != retained_root["immutable_identity"]["gid"]
        ):
            raise SiteBuildError(f"{task_id} publication owner/size differs")
        prefix = f".matched-v2-{index:02d}-{task_id}"
        expected_artifact_basenames = {
            "model_capture": prefix + "-model-capture.json",
            "model_pre_use": prefix + "-model-pre-use.json",
            "consumption_input": prefix + "-consumption-input.json",
            "adapter_capture": prefix + "-adapter-capture.json",
            "adapter_pre_use": prefix + "-adapter-pre-use.json",
            "adapter_post_use": prefix + "-adapter-post-use.json",
            "adapter_final": prefix + "-adapter-final.json",
            "model_post_use": prefix + "-model-post-use.json",
            "eval_consumption_chain": prefix + "-eval-consumption-chain.json",
        }
        if any(
            not isinstance(reference, Mapping)
            or set(reference) != {"basename", "sha256"}
            or reference.get("basename") != expected_artifact_basenames[role]
            or SHA256_RE.fullmatch(str(reference.get("sha256", ""))) is None
            for role, reference in authority_artifacts.items()
        ):
            raise SiteBuildError(f"{task_id} authority artifact identity differs")
        replay_rows = [
            {
                "role": role,
                "basename": authority_artifacts[role]["basename"],
                "sha256": authority_artifacts[role]["sha256"],
            }
            for role in sorted(authority_artifacts)
        ]
        publication_authority = {
            "schema_version":
            "bernini-full644-exploratory-matched-publication-authority-v1",
            "task_id": task_id, "output_path": row["output_path"],
            "output_fd": retained_task["output_fd"],
            "output_identity": output_identity_checked,
            "output_sha256": row["native_output_sha256"],
            "output_size": row["native_output_size"],
            "receipt_path": row["receipt_path"],
            "receipt_fd": retained_task["receipt_fd"],
            "receipt_identity": receipt_identity_checked,
            "receipt_sha256": row["native_receipt_file_sha256"],
            "receipt_size": receipt_identity_checked["size"],
        }
        handoff_payload = {
            "schema_version":
            "full644-exploratory-matched-publication-handoff-payload-v1",
            "task_id": task_id, "output_path": row["output_path"],
            "output_identity": output_identity_checked,
            "output_sha256": row["native_output_sha256"],
            "output_size": row["native_output_size"],
            "receipt_path": row["receipt_path"],
            "receipt_identity": receipt_identity_checked,
            "receipt_sha256": row["native_receipt_file_sha256"],
            "receipt_size": receipt_identity_checked["size"],
            "receipt_digest": row["native_receipt_digest"],
        }
        if (
            row.get("schema_version")
            != "full644-exploratory-matched-runner-task-auh-r5"
            or row.get("task_id") != task_id
            or row.get("task_index") != index
            or row.get("arm") != "full644"
            or row.get("attempt_count") != 1
            or row.get("return_code") != 0
            or row.get("retry_allowed") is not False
            or row.get("plan_digest") != plan.get("plan_digest")
            or row.get("task_input_digest") != expected_task_input
            or row.get("native_output_sha256") != result["output_sha256"]
            or row.get("native_output_size") != result["output_size"]
            or row.get("native_receipt_file_sha256") != result["receipt_file_sha256"]
            or row.get("native_receipt_digest") != result["receipt_digest"]
            or row.get("output_path") != result["output_path"]
            or row.get("receipt_path") != result["receipt_path"]
            or row.get("native_publication_completed_before_parent_post_use_replay")
            is not True
            or row.get("parent_post_use_closed_before_native_publication") is not False
            or row.get("post_use_replay_complete") is not True
            or row.get("model_capture_digest") != model_capture
            or row.get("task_result_digest") != task_digests[index]
            or row.get("environment_digest") != environment_digests[index]
            or row.get("ffmpeg_exec_authority_digest")
            != attestation.get("ffmpeg_exec_authority_digest")
            or row.get("publication_handoff_authority_digest")
            != retained_handoff.get("authority_digest")
            or row.get("publication_handoff_payload_digest")
            != retained_handoff.get("payload_digest")
            or row.get("log_basename") != f".matched-v2-{index:02d}-{task_id}.log"
            or require_digest(row, "task_result_digest", label=f"{task_id} task result")
            != task_digests[index]
            or artifact.get("task_id") != task_id
            or artifact.get("task_result_digest") != task_digests[index]
            or artifact.get("artifact_count") != 9
            or artifact.get("artifact_rows_digest") != object_sha256(replay_rows)
            or artifact.get("runner_task_file_sha256")
            != hashlib.sha256(canonical_json_bytes(row) + b"\n").hexdigest()
            or artifact.get("consumption_digest") != row.get("consumption_digest")
            or artifact.get("native_output_sha256") != result["output_sha256"]
            or artifact.get("native_receipt_file_sha256")
            != result["receipt_file_sha256"]
            or artifact.get("native_receipt_mode") != 0o400
            or artifact.get("native_receipt_nlink") != 1
            or artifact.get("publication_authority_digest")
            != retained_task.get("authority_digest")
            or artifact.get("publication_authority_digest")
            != object_sha256(publication_authority)
            or artifact.get("publication_handoff_authority_digest")
            != retained_handoff.get("authority_digest")
            or artifact.get("publication_handoff_payload_digest")
            != retained_handoff.get("payload_digest")
            or artifact.get("publication_handoff_payload_digest")
            != object_sha256(handoff_payload)
            or artifact.get("retained_receipt_and_output_fds_replayed") is not True
            or artifact.get("v2_verified_result_cross_linked") is not True
            or artifact.get("all_post_use_artifacts_replayed") is not True
        ):
            raise SiteBuildError(f"runner task evidence differs: {task_id}")
    if len({
        row["authority_artifacts"]["model_capture"]["sha256"]
        for row in task_results
    }) != 1:
        raise SiteBuildError("runner model-capture artifact bytes differ across exact5")
    frozen = attestation.get("reused_frozen_execution_contract")
    model_final = attestation.get("model_final")
    if isinstance(model_final, Mapping):
        model_consumptions = model_final.get("task_consumption_digests")
        if not isinstance(model_consumptions, list) or len(model_consumptions) != 5:
            raise SiteBuildError("model final task consumption list differs")
        for index, digest in enumerate(model_consumptions):
            require_sha256(digest, label=f"model final task consumption {index}")
        require_sha256(
            model_final.get("task_consumption_set_digest"),
            label="model final task consumption set",
        )
        require_sha256(
            model_final.get("final_rehash_digest"),
            label="model final rehash digest",
        )
        private_identity = model_final.get("private_parent_current_identity")
        if (
            not isinstance(private_identity, Mapping)
            or set(private_identity) != _STAT_IDENTITY_FIELDS
            or any(
                type(private_identity.get(field)) is not int
                for field in _STAT_IDENTITY_FIELDS
            )
            or not stat.S_ISDIR(private_identity.get("mode", 0))
        ):
            raise SiteBuildError("model final private parent identity differs")
    if (
        frozen != {
            "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
            "retained_model_adapter_fd_closure": True,
            "sealed_publication_handoff": True,
            "four_rank_torchrun": True,
            "post_use_replay": True,
        }
        or not isinstance(model_final, Mapping)
        or set(model_final) != _MODEL_FINAL_FIELDS
        or model_final.get("schema_version")
        != "bernini-action-preservation-model-held-fd-final-v3"
        or require_digest(model_final, "model_final_digest", label="model final")
        != model_final.get("model_final_digest")
        or model_final.get("task_count") != 5
        or model_final.get("model_capture_digest") != model_capture
        or model_final.get("task_consumption_digests")
        != [row.get("consumption_digest") for row in task_results]
        or model_final.get("task_consumption_set_digest")
        != object_sha256(model_final.get("task_consumption_digests"))
        or model_final.get("all_model_bytes_rehashed_after_last_task") is not True
        or model_final.get("all_model_file_and_directory_fds_retained_through_final_rehash")
        is not True
    ):
        raise SiteBuildError("runner frozen execution/model final closure differs")


def validate_receipt(
    receipt: Mapping[str, Any], *, receipt_sha256: str,
    result: Mapping[str, Any], case: Mapping[str, Any],
    checkpoint: Mapping[str, Any], receipt_size: int,
    task_evidence: Mapping[str, Any] | None = None,
) -> dict[str, bytes | str]:
    task_id = str(case["task_id"])
    task = case["task"]
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_TOP_FIELDS:
        raise SiteBuildError(f"{task_id} native receipt root schema differs")
    receipt_digest = require_digest(receipt, "receipt_digest", label=f"{task_id} receipt")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or (
            task_evidence is not None
            and (
                not isinstance(task_evidence, Mapping)
                or set(task_evidence) != _TASK_RESULT_FIELDS
            )
        )
        or type(receipt_size) is not int or receipt_size <= 0
        or receipt_sha256 != result.get("receipt_file_sha256")
        or receipt_digest != result.get("receipt_digest")
        or (
            task_evidence is not None
            and (
                receipt_sha256 != task_evidence.get("native_receipt_file_sha256")
                or receipt_digest != task_evidence.get("native_receipt_digest")
            )
        )
        or receipt.get("infer_lora_source_sha256")
        != EXPECTED_PRODUCER["infer_lora_sha256"]
        or receipt.get("method_source_revision")
        != EXPECTED_PRODUCER["method_source_revision"]
        or receipt.get("method_source_archive_sha256")
        != EXPECTED_PRODUCER["method_source_archive_sha256"]
        or receipt.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or receipt.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or receipt.get("bernini_inference_files") != EXPECTED_BERNINI_INFERENCE_FILES
        or receipt.get("checkpoint_tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
    ):
        raise SiteBuildError(f"{task_id} receipt/report/producer binding differs")
    require_true(receipt.get("experimental_inference"), label=f"{task_id} experimental")
    require_true(
        receipt.get("production_claim_forbidden"),
        label=f"{task_id} production claim forbidden",
    )
    require_false(
        receipt.get("scientific_claim_authorized"),
        label=f"{task_id} scientific claim",
    )

    input_value = receipt.get("input")
    preprocessing = receipt.get("preprocessing")
    prompt = receipt.get("prompt_contract")
    sampling = receipt.get("sampling")
    output = receipt.get("output")
    adapter = receipt.get("adapter")
    consumption = receipt.get("model_consumption")
    runtime = receipt.get("runtime_versions")
    expected_task_input = object_sha256({
        "schema_version": "full644-exploratory-matched-task-input-v2",
        "plan_digest": case["plan_digest"], "task": task,
    })
    if not all(isinstance(value, Mapping) for value in (
        input_value, preprocessing, prompt, sampling, output, adapter,
        consumption, runtime,
    )):
        raise SiteBuildError(f"{task_id} receipt nested contract is absent")
    source_authority = input_value.get("source_video_physical_authority")
    if (
        set(input_value) != _INPUT_FIELDS
        or input_value.get("source_video_path") != task["source_video"]
        or input_value.get("source_video_sha256") != case["source_sha256"]
        or input_value.get("instruction_utf8_sha256") != INSTRUCTION_SHA256
        or input_value.get("instruction_utf8_bytes") != len(INSTRUCTION.encode("utf-8"))
        or input_value.get("accepted_model_conditions")
        != ["source_video", "edit_instruction"]
        or input_value.get("target_video_argument") is not False
        or input_value.get("target_accessed_by_inference") is not False
        or input_value.get("external_mask_or_swept_tube") is not False
        or input_value.get("external_tracking_pose_or_trajectory") is not False
        or input_value.get("reference_image_or_video") is not False
        or input_value.get("external_shared_i0") is not False
        or input_value.get("retained_source_fd_consumed") is not True
        or input_value.get("source_video_pre_and_post_decode_rehashed") is not True
        or not isinstance(source_authority, Mapping)
        or set(source_authority) != {
            "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
            "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
        }
        or source_authority.get("path") != task["source_video"]
        or source_authority.get("sha256") != case["source_sha256"]
        or source_authority.get("size") != case["source_size"]
        or any(
            type(source_authority.get(field)) is not int
            for field in {
                "size", "mode", "device", "inode", "uid", "gid", "nlink",
                "rdev", "blocks", "mtime_ns", "ctime_ns",
            }
        )
        or source_authority.get("mode") != 0o644
        or source_authority.get("nlink") != 1
        or input_value.get("source_video_physical_authority_digest")
        != object_sha256(source_authority)
    ):
        raise SiteBuildError(f"{task_id} receipt source-only input closure differs")

    if (
        set(preprocessing) != _PREPROCESSING_FIELDS
        or preprocessing.get("frame_count") != 81
        or preprocessing.get("fps") != 25.0
        or preprocessing.get("reported_fps") != 25.0
        or preprocessing.get("source_input_hw") != [736, 704]
        or not isinstance(preprocessing.get("source_derived_bucket_hw"), list)
        or len(preprocessing["source_derived_bucket_hw"]) != 2
        or not all(type(value) is int and value > 0 for value in preprocessing["source_derived_bucket_hw"])
        or preprocessing.get("max_pixels") != 245_760
        or preprocessing.get("stride") != 16
        or preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing.get("spatial_policy")
        != "sqrt_max_pixels_then_floor_each_dimension_to_stride"
        or preprocessing.get("resize") != "torchvision_bicubic_antialias_true"
        or preprocessing.get("external_shared_i0") is not False
    ):
        raise SiteBuildError(f"{task_id} receipt preprocessing differs")
    expected_prompt = {
        "task": "mv2v",
        "system_prompt_sha256": "12ce75b4360bf5f6d2fdb1e22619438fad6363fd5356634fa698fcb28a83e0ba",
        "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
        "tokenizer_fix_mistral_regex": True,
        "tokenizer_padding_side": "right",
        "max_sequence_length": 512,
        "prompt_enhancer": False,
    }
    expected_sampling = {
        "num_frames": 81, "num_inference_steps": 40,
        "guidance_mode": "v2v_apg", "omega_vid": 1.25, "omega_img": 0.0,
        "omega_txt": 4.0, "omega_scale": 0.8, "flow_shift": 5.0,
        "seed": SEED, "eta": 0.5, "norm_threshold": [50.0, 50.0],
        "momentum": 0.0, "single_expert": "transformer_1", "ulysses_size": 4,
        "rank0_decode_and_save_only": True, "source_onset_policy": "none",
    }
    if (
        set(prompt) != _PROMPT_FIELDS or dict(prompt) != expected_prompt
        or set(sampling) != _SAMPLING_FIELDS or dict(sampling) != expected_sampling
    ):
        raise SiteBuildError(f"{task_id} prompt/sampling contract differs")

    result_probe = result["media_probe"]
    expected_output_path = task["output"]["video_path"]
    publication_identity = output.get("publication_identity")
    prepublication_identity = output.get("prepublication_identity")
    publication_identity_checked = require_stat_identity(
        publication_identity, label=f"{task_id} receipt publication identity",
        permissions=0o444, nlink=1, size=result["output_size"],
    )
    prepublication_identity_checked = require_stat_identity(
        prepublication_identity,
        label=f"{task_id} receipt prepublication identity",
        permissions=0o600, nlink=0, size=result["output_size"],
    )
    if (
        set(output) != _OUTPUT_FIELDS
        or output.get("path") != expected_output_path
        or output.get("sha256") != result["output_sha256"]
        or output.get("size") != result["output_size"]
        or output.get("frame_count") != 81
        or output.get("fps") != 25.0
        or output.get("width") != result_probe["width"]
        or output.get("height") != result_probe["height"]
        or preprocessing["source_derived_bucket_hw"]
        != [result_probe["height"], result_probe["width"]]
        or output.get("audio_preserved") is not False
        or output.get("anonymous_creation_method") != "linux-sealed-memfd-v1"
        or output.get("anonymous_seal_mask") != 15
        or output.get("sealed_source_sha256") != result["output_sha256"]
        or output.get("sealed_source_size") != result["output_size"]
        or output.get("anonymous_inode_encoded_and_decoded_before_publication")
        is not True
        or output.get("create_only_copy_publication_after_decode") is not True
        or output.get("sealed_source_and_publication_bytes_equal") is not True
        or output.get("retained_inode_encoded_and_replayed") is not True
        or output.get("named_output_never_replaced") is not True
        or (
            task_evidence is not None
            and publication_identity_checked
            != task_evidence.get("native_output_identity")
        )
        or prepublication_identity_checked["uid"]
        != publication_identity_checked["uid"]
        or prepublication_identity_checked["gid"]
        != publication_identity_checked["gid"]
        or (
            task_evidence is not None
            and task_evidence.get("native_receipt_identity", {}).get("size")
            != receipt_size
        )
    ):
        raise SiteBuildError(f"{task_id} receipt output publication differs")

    if (
        set(adapter) != _ADAPTER_FIELDS
        or adapter.get("enabled") is not True
        or adapter.get("mode") != "lora_safe_merge"
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
        or adapter.get("training_global_step") != 644
        or adapter.get("profile") != PROFILE
        or adapter.get("lora_rank") != 64
        or adapter.get("lora_alpha") != 64
        or adapter.get("tensor_count") != 480
        or adapter.get("target_module_count") != 240
        or adapter.get("target_modules_sha256")
        != "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
        or adapter.get("checkpoint_manifest") != checkpoint
        or not isinstance(adapter.get("checkpoint_root"), str)
        or not adapter["checkpoint_root"]
        or not isinstance(adapter.get("adapter_model_path"), str)
        or not adapter["adapter_model_path"]
        or not isinstance(adapter.get("training_receipt_path"), str)
        or not adapter["training_receipt_path"]
        or adapter.get("adapter_model_sha256")
        != EXPECTED_CHECKPOINT["adapter_model_sha256"]
        or adapter.get("training_receipt_digest")
        != EXPECTED_CHECKPOINT["receipt_digest"]
    ):
        raise SiteBuildError(f"{task_id} receipt is not current Full644/R64")

    four_rank = consumption.get("four_rank_attestation")
    if (
        set(consumption) != _MODEL_CONSUMPTION_FIELDS
        or receipt.get("consumption_input_digest")
        != consumption.get("consumption_input_digest")
        or receipt.get("task_input_digest") != consumption.get("task_input_digest")
        or receipt.get("task_input_digest") != expected_task_input
        or SHA256_RE.fullmatch(
            str(receipt.get("consumption_input_digest", ""))
        ) is None
        or (
            task_evidence is not None
            and (
                receipt.get("task_input_digest")
                != task_evidence.get("task_input_digest")
                or receipt.get("consumption_input_digest")
                != task_evidence.get("consumption_input_digest")
                or consumption.get("model_capture_digest")
                != task_evidence.get("model_capture_digest")
                or consumption.get("adapter_capture_digest")
                != task_evidence.get("adapter_capture_digest")
            )
        )
        or consumption.get("source_video_sha256") != case["source_sha256"]
        or consumption.get("source_video_physical_authority_digest")
        != input_value.get("source_video_physical_authority_digest")
        or consumption.get("all_ranks_use_retained_source_fd") is not True
        or consumption.get("ptrace_authorization_used") is not False
        or not isinstance(consumption.get("adapter_capture_digest"), str)
        or SHA256_RE.fullmatch(consumption["adapter_capture_digest"]) is None
        or not isinstance(consumption.get("adapter_view_root"), str)
        or not consumption["adapter_view_root"]
        or not isinstance(consumption.get("model_view_root"), str)
        or not consumption["model_view_root"]
        or SHA256_RE.fullmatch(
            str(consumption.get("inherited_fd_binding_digest", ""))
        ) is None
        or type(consumption.get("fd_view_files_authorized")) is not int
        or consumption["fd_view_files_authorized"] <= 0
        or type(consumption.get("inherited_fd_count")) is not int
        or consumption["inherited_fd_count"] <= 0
        or not isinstance(four_rank, Mapping)
        or set(four_rank) != {
            "world_size", "all_ranks_replayed_exact_fd_views",
            "rank_evidence_digest", "ordered_rank_evidence_digests",
        }
        or four_rank.get("world_size") != 4
        or four_rank.get("all_ranks_replayed_exact_fd_views") is not True
        or require_sha256(
            four_rank.get("rank_evidence_digest"),
            label=f"{task_id} rank evidence",
        ) not in four_rank.get("ordered_rank_evidence_digests", [])
        or four_rank.get("ordered_rank_evidence_digests")
        != [four_rank.get("rank_evidence_digest")] * 4
        or set(runtime) != {"torch", "torch_hip", "transformers", "diffusers", "peft"}
        or runtime.get("peft") != "0.19.1"
        or not all(isinstance(value, str) and value for value in runtime.values())
    ):
        raise SiteBuildError(f"{task_id} model/four-rank consumption differs")
    model_capture = require_sha256(
        consumption.get("model_capture_digest"), label=f"{task_id} model capture"
    )
    return {
        "sampling": canonical_json_bytes(sampling),
        "prompt": canonical_json_bytes(prompt),
        "model_capture": model_capture,
    }


def resolve_tool(value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.parent != Path(".") or candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        located = shutil.which(value)
        if located is None:
            raise SiteBuildError(f"cannot find {label}: {value}")
        resolved = Path(located).resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise SiteBuildError(f"{label} is not an executable regular file: {resolved}")
    return resolved


def probe_video(path: Path, ffprobe: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C"})
    process = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=180, env=environment,
    )
    if process.returncode != 0:
        raise SiteBuildError(
            f"ffprobe failed for {path.name}: "
            + process.stderr.decode("utf-8", "replace")[:500]
        )
    try:
        payload = json.loads(process.stdout.decode("utf-8", "strict"))
        streams = payload["streams"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SiteBuildError(f"ffprobe output differs for {path.name}") from error
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise SiteBuildError(f"{path.name} does not contain exactly one video stream")
    stream = streams[0]
    count_raw = stream.get("nb_read_frames")
    if count_raw in (None, "N/A"):
        count_raw = stream.get("nb_frames")
    fps_raw = stream.get("avg_frame_rate")
    if fps_raw in (None, "0/0"):
        fps_raw = stream.get("r_frame_rate")
    try:
        frame_count = int(count_raw)
        fps = Fraction(str(fps_raw))
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise SiteBuildError(f"ffprobe media fields differ for {path.name}") from error
    if (
        stream.get("codec_type") != "video"
        or stream.get("codec_name") != "h264"
        or frame_count != EXPECTED_MEDIA_FRAME_COUNT
        or fps != EXPECTED_MEDIA_FPS
        or width <= 0 or height <= 0
    ):
        raise SiteBuildError(f"{path.name} is not H.264 81f@25fps")
    return {
        "codec": "h264", "frame_count": frame_count,
        "fps_num": fps.numerator, "fps_den": fps.denominator,
        "width": width, "height": height,
    }


def make_pair_sheet(
    source: Path, output_video: Path, destination: Path, ffmpeg: Path, *,
    all_frames: bool,
) -> None:
    if all_frames:
        frame_filter = "select=between(n\\,0\\,80),scale=112:-2,setsar=1,tile=9x9:padding=2:margin=2"
    else:
        expression = "+".join(f"eq(n\\,{frame})" for frame in KEYFRAMES)
        frame_filter = f"select={expression},scale=208:-2,setsar=1,tile=5x1:padding=2:margin=2"
    complex_filter = (
        f"[0:v]{frame_filter}[source];"
        f"[1:v]{frame_filter}[result];"
        "[source][result]vstack=inputs=2[pair]"
    )
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C"})
    process = subprocess.run(
        [
            str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-i", str(output_video), "-filter_complex",
            complex_filter, "-map", "[pair]", "-frames:v", "1", "-q:v", "2",
            str(destination),
        ],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=300, env=environment,
    )
    if process.returncode != 0:
        raise SiteBuildError(
            f"ffmpeg pair sheet failed for {source.name}/{output_video.name}: "
            + process.stderr.decode("utf-8", "replace")[:500]
        )
    raw, _, size = stable_file(destination, label=f"pair sheet {destination.name}")
    if size < 4 or not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise SiteBuildError(f"pair sheet is not a complete JPEG: {destination.name}")


def _write_new(path: Path, raw: bytes, *, mode: int = 0o444) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise SiteBuildError(f"write made no progress: {path}")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_verified(
    source: Path, destination: Path, *, expected_sha256: str, label: str,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        raise SiteBuildError(f"staged destination is not fresh: {destination}")
    shutil.copyfile(source, destination, follow_symlinks=False)
    os.chmod(destination, 0o444)
    _, observed_sha256, size = stable_file(destination, label=label)
    if observed_sha256 != expected_sha256:
        raise SiteBuildError(f"published copy SHA differs: {destination.name}")
    return {"sha256": observed_sha256, "size": size}


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render_html(
    cases: Sequence[Mapping[str, Any]], *, report_sha256: str,
    attestation_sha256: str, build_time: str,
) -> str:
    sections: list[str] = []
    for index, case in enumerate(cases):
        variant = str(case["id"])
        source = case["media"]["source"]
        result = case["media"]["result"]
        source_probe = source["probe"]
        result_probe = result["probe"]
        parity = (
            '<span class="pill parity">frozen parity PASS</span>'
            if variant == "exact_original" else ""
        )
        search = " ".join(
            str(value) for value in (
                variant, case["title"], case["title_zh"], case["group"],
                case["treatment"], case["bone_position"], IID, INSTRUCTION,
            )
        ).lower()
        sections.append(
            f'''<article class="case" id="variant-{_h(variant)}" data-group="{_h(case['group'])}" data-search="{_h(search)}">
  <header class="case-head">
    <div class="case-number">{index + 1:02d}</div>
    <div><div class="case-title"><h2>{_h(case['title_zh'])}</h2><span class="pill">{_h(variant)}</span>{parity}</div><p>{_h(case['summary'])}</p></div>
    <button class="sync-pair" type="button">两列同步从头播放</button>
  </header>
  <div class="intervention-meta"><span>bone present: {_h(str(case['bone_present']).lower())}</span><span>position: {_h(case['bone_position'])}</span><span>treatment: {_h(case['treatment'])}</span></div>
  <div class="video-grid">
    <article class="video-card">
      <div class="video-head"><div><h3>Source intervention</h3><p>{_h(case['title'])}</p></div><button class="play-one" type="button">播放 / 暂停</button></div>
      <video controls muted playsinline preload="metadata" aria-label="{_h(case['title_zh'])} source intervention" src="assets/media/{_h(source['basename'])}"></video>
      <div class="media-meta"><span>{source_probe['frame_count']}f · {source_probe['fps_num']}fps</span><span>{source_probe['width']}×{source_probe['height']}</span><span>H.264</span></div>
      <code title="{_h(source['sha256'])}">sha256 {_h(source['sha256'])}</code>
      <div class="asset-links"><a href="assets/media/{_h(source['basename'])}">源视频文件</a></div>
    </article>
    <article class="video-card r64">
      <div class="video-head"><div><h3>Current Full644 / R64 output</h3><p>checkpoint step 644 · rank 64</p></div><button class="play-one" type="button">播放 / 暂停</button></div>
      <video controls muted playsinline preload="metadata" aria-label="{_h(case['title_zh'])} current R64 output" src="assets/media/{_h(result['basename'])}"></video>
      <div class="media-meta"><span>{result_probe['frame_count']}f · {result_probe['fps_num']}fps</span><span>{result_probe['width']}×{result_probe['height']}</span><span>H.264</span></div>
      <code title="{_h(result['sha256'])}">sha256 {_h(result['sha256'])}</code>
      <div class="asset-links"><a href="assets/media/{_h(result['basename'])}">输出视频</a> · <a href="evidence/receipts/{_h(result['receipt_basename'])}">native receipt</a></div>
    </article>
  </div>
  <details class="sheets" open><summary>帧对照（每张图上半部 Source，下半部 R64 output）</summary><div class="sheet-grid">
    <button class="sheet" type="button" data-image="assets/sheets/{_h(variant)}-keyframes.jpg" data-title="{_h(case['title_zh'])} · frames 0/20/40/60/80"><span>关键帧 · 0 / 20 / 40 / 60 / 80</span><img loading="lazy" src="assets/sheets/{_h(variant)}-keyframes.jpg" alt="{_h(case['title_zh'])} Source 与 R64 五个关键帧对照"></button>
    <button class="sheet" type="button" data-image="assets/sheets/{_h(variant)}-all81.jpg" data-title="{_h(case['title_zh'])} · all 81 frames"><span>全部 81 帧 · 9 × 9</span><img loading="lazy" src="assets/sheets/{_h(variant)}-all81.jpg" alt="{_h(case['title_zh'])} Source 与 R64 全部 81 帧对照"></button>
  </div></details>
</article>'''
        )

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>Case01 Exact5 · Current R64 离线结果</title>
<style>
:root{{--bg:#080c11;--panel:#111821;--panel2:#172230;--line:#2a394a;--text:#eef4fa;--muted:#9dadbd;--blue:#78baff;--green:#62d79c;--amber:#ffc85c;--red:#ff8176;--max:1480px}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--text);background:radial-gradient(circle at 10% -10%,rgba(120,186,255,.14),transparent 34rem),radial-gradient(circle at 92% 0,rgba(98,215,156,.09),transparent 30rem),var(--bg);font:15px/1.58 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}a{{color:#9dceff}}button,input,select{{font:inherit}}code{{font:11px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}}.wrap{{width:min(calc(100% - 32px),var(--max));margin:auto}}
.topbar{{position:sticky;top:0;z-index:30;border-bottom:1px solid rgba(255,255,255,.08);background:rgba(8,12,17,.91);backdrop-filter:blur(16px)}}.topbar .wrap{{min-height:62px;display:flex;align-items:center;justify-content:space-between;gap:16px}}.brand{{font-weight:820}}.brand::before{{content:"";display:inline-block;width:9px;height:9px;margin-right:10px;border-radius:50%;background:var(--green);box-shadow:0 0 0 6px rgba(98,215,156,.12)}}.topmeta{{color:var(--muted);font-size:12px;text-align:right}}
.hero{{padding:62px 0 34px}}.eyebrow{{color:var(--green);font-size:12px;font-weight:850;letter-spacing:.15em;text-transform:uppercase}}h1{{margin:12px 0 18px;font-size:clamp(37px,6vw,75px);line-height:1;letter-spacing:-.05em}}.lede{{max-width:1030px;margin:0;color:#c6d0d9;font-size:clamp(17px,2vw,21px)}}.scope{{display:grid;grid-template-columns:auto 1fr;gap:15px;margin-top:27px;padding:20px;border:1px solid rgba(255,200,92,.5);border-radius:16px;background:linear-gradient(135deg,rgba(255,200,92,.14),rgba(255,200,92,.035))}}.scope .icon{{width:36px;height:36px;display:grid;place-items:center;border-radius:50%;color:#1c1404;background:var(--amber);font-weight:900}}.scope strong{{display:block;color:#ffe09a;font-size:17px}}.scope p{{margin:5px 0 0;color:#ddd2ba}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}.stat{{padding:17px;border:1px solid var(--line);border-radius:14px;background:rgba(17,24,33,.82)}}.stat b{{display:block;color:var(--green);font-size:26px}}.stat span{{color:var(--muted);font-size:12px}}.instruction{{margin-top:16px;padding:16px 18px;border-left:4px solid var(--blue);border-radius:0 12px 12px 0;background:var(--panel2);font-size:17px}}.instruction small{{display:block;color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}}.instruction code{{display:block;margin-top:7px;color:var(--muted)}}
.filters{{position:sticky;top:62px;z-index:20;padding:11px 0;border-block:1px solid rgba(255,255,255,.07);background:rgba(8,12,17,.89);backdrop-filter:blur(14px)}}.filter-row{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}}.filter,.global-button,select{{padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--muted);background:var(--panel);cursor:pointer}}.filter.active,.filter[aria-pressed="true"]{{color:#07120d;border-color:var(--green);background:var(--green);font-weight:760}}.global-button:hover,.filter:hover{{border-color:#56718c;color:var(--text)}}#search{{min-width:250px;flex:1;padding:9px 13px;border:1px solid var(--line);border-radius:10px;color:var(--text);background:var(--panel)}}#visible-count{{color:var(--muted);font-size:12px}}
.case-list{{display:grid;gap:27px;padding:30px 0}}.case{{scroll-margin-top:128px;overflow:hidden;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,rgba(23,34,48,.95),rgba(13,19,27,.96));box-shadow:0 24px 72px rgba(0,0,0,.24)}}.case[hidden]{{display:none}}.case-head{{display:grid;grid-template-columns:58px 1fr auto;gap:16px;align-items:center;padding:21px 22px 15px}}.case-number{{width:52px;height:52px;display:grid;place-items:center;border:1px solid var(--line);border-radius:14px;color:var(--green);background:#0a1017;font-weight:850;font-size:18px}}.case-title{{display:flex;flex-wrap:wrap;align-items:center;gap:8px}}.case-title h2{{margin:0;font-size:26px}}.case-head p{{margin:4px 0 0;color:var(--muted)}}.pill{{padding:3px 8px;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:11px}}.pill.parity{{color:#a3e9c1;border-color:rgba(98,215,156,.4);background:rgba(98,215,156,.08)}}.sync-pair,.play-one{{padding:7px 10px;border:1px solid #47647f;border-radius:8px;color:var(--text);background:#1a2a3b;cursor:pointer}}.sync-pair:hover,.play-one:hover{{background:#243a51}}.intervention-meta{{display:flex;flex-wrap:wrap;gap:7px;padding:0 22px 18px}}.intervention-meta span{{padding:4px 8px;border:1px solid var(--line);border-radius:7px;color:var(--muted);background:#0c131b;font:11px ui-monospace,SFMono-Regular,monospace;word-break:break-all}}
.video-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;padding:0 12px 12px}}.video-card{{overflow:hidden;border:1px solid var(--line);border-radius:13px;background:var(--panel)}}.video-card.r64{{border-color:#355f4a}}.video-head{{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:12px}}.video-head h3{{margin:0;font-size:17px}}.video-head p{{margin:2px 0 0;color:var(--muted);font-size:11px}}.play-one{{padding:5px 8px;font-size:11px}}video{{display:block;width:100%;aspect-ratio:704/736;max-height:72vh;background:#000;object-fit:contain}}.media-meta{{display:flex;flex-wrap:wrap;gap:6px;padding:9px 11px 4px}}.media-meta span{{padding:3px 6px;border-radius:6px;color:#b8c6d3;background:#0b1219;font-size:10px}}.video-card code{{display:block;padding:3px 11px;color:var(--muted)}}.asset-links{{padding:5px 11px 12px;color:var(--muted);font-size:11px}}
.sheets{{border-top:1px solid var(--line);background:#090e14}}.sheets summary{{padding:11px 15px;color:var(--muted);cursor:pointer}}.sheet-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:0 12px 12px}}.sheet{{padding:0;overflow:hidden;border:1px solid var(--line);border-radius:10px;color:var(--text);background:#0c1219;cursor:zoom-in;text-align:left}}.sheet span{{display:block;padding:7px 9px;font-size:11px}}.sheet img{{display:block;width:100%;height:auto}}.empty{{display:none;margin:30px 0;padding:30px;border:1px dashed var(--line);border-radius:14px;color:var(--muted);text-align:center}}footer{{padding:18px 0 42px;color:var(--muted);font-size:12px}}footer .evidence{{display:flex;flex-wrap:wrap;gap:8px 16px;margin-bottom:8px}}
dialog{{width:min(96vw,1800px);max-width:none;padding:0;border:1px solid #47596b;border-radius:14px;color:var(--text);background:#070b10;box-shadow:0 35px 120px rgba(0,0,0,.8)}}dialog::backdrop{{background:rgba(0,0,0,.86);backdrop-filter:blur(5px)}}dialog img{{display:block;width:100%;height:auto}}.dialog-bar{{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:10px 12px;border-bottom:1px solid var(--line)}}.dialog-bar span{{color:var(--muted)}}.dialog-bar button{{padding:6px 9px;border:1px solid var(--line);border-radius:7px;color:var(--text);background:var(--panel);cursor:pointer}}
@media(max-width:900px){{.stats{{grid-template-columns:repeat(2,1fr)}}.video-grid,.sheet-grid{{grid-template-columns:1fr}}.case-head{{grid-template-columns:50px 1fr}}.sync-pair{{grid-column:2;justify-self:start}}}}@media(max-width:560px){{.wrap{{width:min(calc(100% - 20px),var(--max))}}.topmeta{{display:none}}.hero{{padding-top:42px}}.scope{{grid-template-columns:1fr}}#search{{min-width:100%}}.case-head,.intervention-meta{{padding-inline:14px}}}}
</style>
</head>
<body>
<nav class="topbar"><div class="wrap"><div class="brand">Case01 Exact5 · Current R64</div><div class="topmeta">Heldout8 inference case · 5 source interventions · Source / R64</div></div></nav>
<header class="hero wrap">
  <div class="eyebrow">5 / 5 staged final results · offline visual review</div>
  <h1>一个 Heldout8 例子的<br>五种输入干预结果</h1>
  <p class="lede">同一 case01/IID 的五种 source-object intervention，分别送入同一个当前 Full644/R64 checkpoint。每组均展示真实 Source 与真实 R64 输出；这五组不是五个独立数据集样本。</p>
  <div class="scope"><div class="icon">!</div><div><strong>当前 R64 checkpoint；Heldout8 推理例；非 Full644 训练子集；非 formal training evaluation。</strong><p>本 exact5 staged bundle 未独立附带 Full644 membership / exposure audit。页面只支持 exploratory causal-input visual review，不能证明 IID 或 content-disjoint、预训练无 exposure、科学泛化或 intervention effect；必须进行 manual blind review。</p></div></div>
  <div class="stats"><div class="stat"><b>5 / 5</b><span>exact5 全部任务，无 cherry-pick</span></div><div class="stat"><b>10</b><span>五个 source + 五个 R64 output</span></div><div class="stat"><b>81f · 25fps</b><span>十个视频全部本地复验</span></div><div class="stat"><b>R64 · step644</b><span>current Full644 adapter</span></div></div>
  <div class="instruction"><small>Instruction · IID {IID}</small>{_h(INSTRUCTION)}<code>instruction sha256 {INSTRUCTION_SHA256}</code></div>
</header>
<section class="filters"><div class="wrap filter-row" aria-label="结果筛选与播放控制">
  <button class="filter active" type="button" data-group="all" aria-pressed="true">全部 5 组</button><button class="filter" type="button" data-group="controls" aria-pressed="false">controls · 3</button><button class="filter" type="button" data-group="interventions" aria-pressed="false">object interventions · 2</button>
  <button class="global-button" id="play-visible" type="button">可见视频同步从头播放</button><button class="global-button" id="pause-all" type="button">全部暂停</button>
  <label>速度 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
  <input id="search" type="search" placeholder="筛选 variant / treatment / bone state" aria-label="筛选干预条件"><span id="visible-count" aria-live="polite">5 / 5</span>
</div></section>
<main class="wrap"><div class="case-list">{''.join(sections)}</div><div class="empty" id="empty">没有匹配的 intervention。</div></main>
<footer class="wrap"><div class="evidence"><a href="evidence/plan.json">exact5 plan</a><a href="evidence/report.json">exact5 report</a><a href="evidence/runner-attestation.json">runner attestation</a><a href="site-manifest.json">site manifest</a></div><div>All-relative offline HTML · report {_h(report_sha256)} · attestation {_h(attestation_sha256)} · built {_h(build_time)}</div></footer>
<dialog id="lightbox"><div class="dialog-bar"><span id="dialog-title"></span><button type="button" id="dialog-close">关闭</button></div><img id="dialog-image" alt="放大的 Source 与 R64 帧对照图"></dialog>
<script>
const cards=[...document.querySelectorAll('.case')],filters=[...document.querySelectorAll('.filter')],search=document.querySelector('#search'),speed=document.querySelector('#speed');let group='all';
function visibleVideos(){{return cards.filter(card=>!card.hidden).flatMap(card=>[...card.querySelectorAll('video')]);}}
function applyFilter(){{const query=search.value.trim().toLowerCase();let visible=0;for(const card of cards){{const show=(group==='all'||card.dataset.group===group)&&(!query||card.dataset.search.includes(query));card.hidden=!show;if(show)visible++;else card.querySelectorAll('video').forEach(video=>video.pause());}}document.querySelector('#visible-count').textContent=`${{visible}} / 5`;document.querySelector('#empty').style.display=visible?'none':'block';}}
filters.forEach(button=>button.addEventListener('click',()=>{{group=button.dataset.group;filters.forEach(item=>{{const active=item===button;item.classList.toggle('active',active);item.setAttribute('aria-pressed',String(active));}});applyFilter();}}));search.addEventListener('input',applyFilter);
async function playTogether(videos){{videos.forEach(video=>{{video.pause();video.currentTime=0;video.playbackRate=Number(speed.value);}});await Promise.allSettled(videos.map(video=>video.play()));}}
document.querySelectorAll('.sync-pair').forEach(button=>button.addEventListener('click',()=>playTogether([...button.closest('.case').querySelectorAll('video')])));
document.querySelector('#play-visible').addEventListener('click',()=>playTogether(visibleVideos()));document.querySelector('#pause-all').addEventListener('click',()=>cards.flatMap(card=>[...card.querySelectorAll('video')]).forEach(video=>video.pause()));
speed.addEventListener('change',()=>cards.flatMap(card=>[...card.querySelectorAll('video')]).forEach(video=>{{video.playbackRate=Number(speed.value);}}));
document.querySelectorAll('.play-one').forEach(button=>button.addEventListener('click',async()=>{{const video=button.closest('.video-card').querySelector('video');video.playbackRate=Number(speed.value);if(video.paused)await video.play().catch(()=>{{}});else video.pause();}}));
const dialog=document.querySelector('#lightbox');document.querySelectorAll('.sheet').forEach(button=>button.addEventListener('click',()=>{{document.querySelector('#dialog-image').src=button.dataset.image;document.querySelector('#dialog-title').textContent=button.dataset.title;dialog.showModal();}}));document.querySelector('#dialog-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{{if(event.target===dialog)dialog.close();}});document.addEventListener('keydown',event=>{{if(event.key==='Escape'&&dialog.open)dialog.close();}});
</script>
</body>
</html>
'''


def _exact_names(path: Path, expected: set[str], *, label: str) -> None:
    _plain_directory(path, label=label)
    observed = {item.name for item in path.iterdir()}
    if observed != expected:
        raise SiteBuildError(
            f"{label} closure differs; missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _bundle_paths(bundle: Path) -> tuple[Path, Path, Path, Path, Path]:
    _exact_names(bundle, {"plan", "final", "sources", "outputs"}, label="bundle root")
    _exact_names(bundle / "plan", {PLAN_REL.name}, label="bundle plan")
    _exact_names(
        bundle / "final", {REPORT_REL.name, ATTESTATION_REL.name},
        label="bundle final",
    )
    _exact_names(
        bundle / "sources", {f"{variant}.mp4" for variant in VARIANT_ORDER},
        label="bundle sources",
    )
    _exact_names(bundle / "outputs", {"media"}, label="bundle outputs")
    media_names: set[str] = set()
    for task_id in TASK_IDS:
        media_names.update((f"{task_id}.mp4", f"{task_id}.mp4.receipt.json"))
    _exact_names(bundle / "outputs/media", media_names, label="bundle output media")
    return (
        bundle / PLAN_REL, bundle / REPORT_REL, bundle / ATTESTATION_REL,
        bundle / "sources", bundle / "outputs/media",
    )


def build_site(
    *, bundle: Path, output: Path, ffmpeg: Path, ffprobe: Path,
) -> dict[str, Any]:
    plan_path, report_path, attestation_path, source_root, output_root = _bundle_paths(bundle)
    plan, plan_sha256, _ = load_json(plan_path, label="exact5 plan")
    cases = validate_plan(plan)
    report, report_sha256, _ = load_json(report_path, label="exact5 report")
    results = validate_report(report, plan)
    attestation, attestation_sha256, _ = load_json(
        attestation_path, label="exact5 runner attestation"
    )
    validate_attestation(
        attestation, plan=plan, plan_sha256=plan_sha256,
        report=report, report_sha256=report_sha256,
    )
    task_evidence_by_id = {
        row["task_id"]: row for row in attestation["task_results"]
    }

    input_files: dict[str, dict[str, Any]] = {}
    coordinate_rows: list[dict[str, bytes | str]] = []
    for case in cases:
        variant = str(case["id"])
        task_id = str(case["task_id"])
        result = results[task_id]
        source_path = source_root / f"{variant}.mp4"
        result_path = output_root / f"{task_id}.mp4"
        receipt_path = output_root / f"{task_id}.mp4.receipt.json"

        _, source_sha256, source_size = stable_file(
            source_path, label=f"{variant} source"
        )
        source_probe = probe_video(source_path, ffprobe)
        _, source_replay_sha256, source_replay_size = stable_file(
            source_path, label=f"{variant} source replay"
        )
        if (
            source_sha256 != case["source_sha256"]
            or source_size != case["source_size"]
            or source_replay_sha256 != source_sha256
            or source_replay_size != source_size
            or source_probe != {"codec": "h264", **EXPECTED_SOURCE_VIDEO}
        ):
            raise SiteBuildError(f"{variant} local source authority differs")

        _, result_sha256, result_size = stable_file(
            result_path, label=f"{task_id} output"
        )
        result_probe = probe_video(result_path, ffprobe)
        _, result_replay_sha256, result_replay_size = stable_file(
            result_path, label=f"{task_id} output replay"
        )
        expected_probe = result["media_probe"]
        if (
            result_sha256 != result["output_sha256"]
            or result_size != result["output_size"]
            or result_replay_sha256 != result_sha256
            or result_replay_size != result_size
            or result_probe["frame_count"] != expected_probe["frame_count"]
            or result_probe["fps_num"] != expected_probe["fps_num"]
            or result_probe["fps_den"] != expected_probe["fps_den"]
            or result_probe["width"] != expected_probe["width"]
            or result_probe["height"] != expected_probe["height"]
        ):
            raise SiteBuildError(f"{task_id} local output/report binding differs")

        receipt, receipt_sha256, receipt_size = load_json(
            receipt_path, label=f"{task_id} native receipt"
        )
        coordinate_rows.append(
            validate_receipt(
                receipt, receipt_sha256=receipt_sha256, result=result, case=case,
                checkpoint=plan["checkpoint_manifest"],
                task_evidence=task_evidence_by_id[task_id],
                receipt_size=receipt_size,
            )
        )
        source_basename = f"{variant}-source.mp4"
        result_basename = f"{variant}-r64.mp4"
        receipt_basename = f"{task_id}.mp4.receipt.json"
        case["media"] = {
            "source": {
                "source": source_path, "basename": source_basename,
                "sha256": source_sha256, "size": source_size,
                "probe": source_probe,
            },
            "result": {
                "source": result_path, "basename": result_basename,
                "sha256": result_sha256, "size": result_size,
                "probe": result_probe, "receipt_source": receipt_path,
                "receipt_basename": receipt_basename,
                "receipt_sha256": receipt_sha256,
            },
        }
        input_files[source_basename] = {
            "source": source_path, "sha256": source_sha256,
        }
        input_files[result_basename] = {
            "source": result_path, "sha256": result_sha256,
        }
    if (
        len({row["sampling"] for row in coordinate_rows}) != 1
        or len({row["prompt"] for row in coordinate_rows}) != 1
        or len({row["model_capture"] for row in coordinate_rows}) != 1
    ):
        raise SiteBuildError("exact5 receipts do not share sampler/prompt/model capture")

    output = output.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise SiteBuildError(f"output must be fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent)
    )
    try:
        media_out = stage / "assets/media"
        sheets_out = stage / "assets/sheets"
        evidence_out = stage / "evidence"
        receipts_out = evidence_out / "receipts"
        for directory in (stage / "assets", media_out, sheets_out, evidence_out, receipts_out):
            directory.mkdir(mode=0o755)

        published: list[dict[str, Any]] = []
        for basename, row in sorted(input_files.items()):
            copied = _copy_verified(
                row["source"], media_out / basename,
                expected_sha256=row["sha256"], label=f"published {basename}",
            )
            published.append({"path": f"assets/media/{basename}", **copied})
        for case in cases:
            result_media = case["media"]["result"]
            receipt_basename = result_media["receipt_basename"]
            copied = _copy_verified(
                result_media["receipt_source"], receipts_out / receipt_basename,
                expected_sha256=result_media["receipt_sha256"],
                label=f"published {receipt_basename}",
            )
            published.append({"path": f"evidence/receipts/{receipt_basename}", **copied})
        for basename, source, sha256 in (
            ("plan.json", plan_path, plan_sha256),
            ("report.json", report_path, report_sha256),
            ("runner-attestation.json", attestation_path, attestation_sha256),
        ):
            copied = _copy_verified(
                source, evidence_out / basename, expected_sha256=sha256,
                label=f"published evidence {basename}",
            )
            published.append({"path": f"evidence/{basename}", **copied})

        for case in cases:
            variant = str(case["id"])
            source_media = media_out / case["media"]["source"]["basename"]
            result_media = media_out / case["media"]["result"]["basename"]
            for suffix, all_frames in (("keyframes", False), ("all81", True)):
                sheet = sheets_out / f"{variant}-{suffix}.jpg"
                make_pair_sheet(
                    source_media, result_media, sheet, ffmpeg,
                    all_frames=all_frames,
                )
                os.chmod(sheet, 0o444)
                _, sheet_sha256, sheet_size = stable_file(
                    sheet, label=f"published {variant} {suffix} sheet"
                )
                published.append({
                    "path": f"assets/sheets/{sheet.name}",
                    "sha256": sheet_sha256, "size": sheet_size,
                })

        build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        index_raw = render_html(
            cases, report_sha256=report_sha256,
            attestation_sha256=attestation_sha256, build_time=build_time,
        ).encode("utf-8")
        _write_new(stage / "index.html", index_raw)
        published.append({
            "path": "index.html", "sha256": hashlib.sha256(index_raw).hexdigest(),
            "size": len(index_raw),
        })
        manifest: dict[str, Any] = {
            "schema_version": SITE_SCHEMA,
            "status": "COMPLETE_EXPLORATORY_EXACT5_REVIEW_SITE",
            "built_at_utc": build_time,
            "iid": IID,
            "instruction": INSTRUCTION,
            "variant_count": 5,
            "source_video_count": 5,
            "generated_output_count": 5,
            "video_count": 10,
            "pair_sheet_count": 10,
            "keyframes": list(KEYFRAMES),
            "all81_sheet_included_for_each_variant": True,
            "media_contract": {"codec": "h264", "frame_count": 81, "fps_num": 25, "fps_den": 1},
            "dataset_scope": {
                "name": "Goku legacy Heldout8 inference case01 exact5 interventions",
                "independent_dataset_example_count": 1,
                "source_intervention_count": 5,
                "full644_training_subset": False,
                "full644_membership_audit_included": False,
                "iid_disjoint_proven_by_this_bundle": False,
                "content_disjoint_proven": False,
                "formal_training_evaluation": False,
                "scientific_generalization_claim_authorized": False,
            },
            "claim_limits": dict(CLAIM_LIMITS),
            "checkpoint": {
                "profile": PROFILE, "global_step": 644, "lora_rank": 64,
                "adapter_model_sha256": EXPECTED_CHECKPOINT["adapter_model_sha256"],
                "checkpoint_manifest_sha256": EXPECTED_CHECKPOINT["sha256"],
                "training_receipt_sha256": EXPECTED_CHECKPOINT["training_receipt_sha256"],
            },
            "deterministic_reference_parity": dict(
                report["deterministic_reference_parity"]
            ),
            "authorities": {
                "plan_sha256": plan_sha256, "plan_digest": plan["plan_digest"],
                "report_sha256": report_sha256,
                "report_digest": report["report_digest"],
                "attestation_sha256": attestation_sha256,
                "attestation_digest": attestation["attestation_digest"],
            },
            "human_visual_judgment_added_by_builder": False,
            "files_excluding_this_manifest": sorted(published, key=lambda row: row["path"]),
        }
        manifest["manifest_digest"] = object_sha256(manifest)
        manifest_raw = canonical_json_bytes(manifest) + b"\n"
        _write_new(stage / "site-manifest.json", manifest_raw)
        os.replace(stage, output)
        stage = None
        return {
            "output": str(output), "index": str(output / "index.html"),
            "manifest": str(output / "site-manifest.json"),
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "variant_count": 5, "video_count": 10, "pair_sheet_count": 10,
        }
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an all-relative case01 exact5 current-R64 HTML site from a "
            "staged real final-summary bundle."
        )
    )
    parser.add_argument(
        "--bundle", required=True,
        help="staged exact5 final-summary mirror; missing/extra files hard-fail",
    )
    parser.add_argument(
        "--output", required=True,
        help="fresh output directory; never overwritten",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_site(
            bundle=Path(args.bundle).expanduser().absolute(),
            output=Path(args.output).expanduser(),
            ffmpeg=resolve_tool(args.ffmpeg, label="ffmpeg"),
            ffprobe=resolve_tool(args.ffprobe, label="ffprobe"),
        )
    except (OSError, SiteBuildError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

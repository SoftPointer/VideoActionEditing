#!/usr/bin/env python3
"""Frozen-base versus terminal-full644 matched shared-8 evaluation contract.

This module only authors and audits local JSON artifacts.  It never launches
inference, training, SSH, Slurm, or a runner.  ``authority-check`` is useful on
a laptop without the eight source videos.  ``build-plan`` is intentionally
stricter: every source byte string and the terminal checkpoint manifest must
exist and match their externally pinned SHA-256 before a production plan can be
created.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence


INPUT_MANIFEST_SHA256 = (
    "c05c4e5b5bf85de882bde32c71a984d736247733e586ed91d40026b12aaaf701"
)
EXPOSURE_AUDIT_SHA256 = (
    "953933f1161b6d62826d388ba5ed42e42792fbf5f2bdeea199c1eb13cd251b4a"
)
EXPECTED_IIDS = (
    "1852ada01d7c43a4",
    "288545b9c031491a",
    "5ae88e1170c544b8",
    "81473c034c1b4839",
    "2766a3662fbf43d1",
    "219c4c5f56e74b86",
    "2206cde2643e470a",
    "7a2f54be92024a19",
)
EXPECTED_SOURCE_SHA256 = (
    "84d8361bb53d9a210b5c19ceba22ac31ba7a3b008760afd132f865065266bbf7",
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18",
    "cfaf78f330669eb5a303c1701cd8ac7b38f70e9d32f5b15afb2d30c4d3776adb",
    "a543d35d96c0744ff52734752dc30bbb20b8a25fd13f73d8e336148f06fc62f4",
    "a1f0da10376c0e80fc31f973eaf53a13d78271b17ace99886a23cec15619f436",
    "8d882b3070ef1db35a8b46698264ea89c3cc48fe0e00de52fac7ee46d14034a0",
    "9df40a0817e75fd6960b4289d3365edd626c906443b5fb12bb9c5e0e8676a4a3",
    "b8d2f6af9523a1f75f7a62d3ffa4e515e139a5e57ff18a843c2450893427f8fa",
)

INPUT_ROW_SCHEMA = "action-editing-shared8-input-v1"
EXPOSURE_SCHEMA = "action-editing-shared8-exposure-audit-v1"
CHECKPOINT_MANIFEST_SCHEMA = "bernini-r-action-lora-checkpoint-manifest-v1"
PLAN_SCHEMA = "bernini-full644-exploratory-matched-eval-plan-v1"
REPORT_SCHEMA = "bernini-full644-exploratory-matched-eval-report-v1"
FULL644_PROFILE = "full644-r64-reference-dpo-preservation-one-pass-v1"
FULL644_STEP = 644
INFERENCE_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-inference-receipt-v5"
EXPECTED_TARGET_MODULES_SHA256 = (
    "d253ba3f11ec5ac26710a829d543a18b939c6f111c64be785264fcd852f3f35a"
)
EXPECTED_TARGET_MODULE_COUNT = 240
EXPECTED_ADAPTER_TENSOR_COUNT = 480
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "12ce75b4360bf5f6d2fdb1e22619438fad6363fd5356634fa698fcb28a83e0ba"
)
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"
)
EXPECTED_BERNINI_INFERENCE_FILES = {
    "bernini/pipeline.py": "c6acf05c01a637d9bce69e8160eb6eb4260ff4ec798fd990de8e5aa73999ab40",
    "bernini/cli.py": "26949fbf246003403ed0cca1ec1bbb62c2099fc9740bb17ba5a1e7c86fbc0edf",
    "bernini/io_utils.py": "233541373746f5d97e1cb3680d3c2a41d5d212b797eefb97693afa6e3ab5f30a",
}
TRAINING_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-receipt-v2"
EXPECTED_BERNINI_TRAINING_FILES_INDEX_SHA256 = (
    "faeaa381cb076febd07ac0eb90d17396b61ff400eac2e02a6c7b3c70ff062764"
)
FULL644_SOURCE_AUTHORITY_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)
FULL644_DATASET_SUMMARY_SHA256 = (
    "5dc45b4a6d700b3cd0108e941242ae364396458f20f41249744e74e00acc02dd"
)
FULL644_DATASET_SUMMARY_DIGEST = (
    "29e2341f09d58289590ae48d17d02f2299bac3201df772584b6269bec0dbbe82"
)
FULL644_DATASET_INDEX_SHA256 = (
    "d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"
)
FULL644_TRAINABLE_PARAMETER_COUNT = 47_185_920
FULL644_SEED = 20_260_817
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SHA1_RE = re.compile(r"[0-9a-f]{40}")

CLAIM_LIMITS = {
    "historical_shared8_exposed": True,
    "iid_overlap_with_full644": 0,
    "iid_heldout_diagnostic": True,
    "content_disjoint_split": False,
    "human_reviewed_labels": False,
    "scientific_generalization_claim_authorized": False,
    "formal_claim_authorized": False,
    "evaluation_role": "engineering_diagnostic_only",
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
_SOURCE_AUTHORITY_FIELDS = {
    "path", "sha256", "size", "mode", "device", "inode", "uid", "gid",
    "nlink", "rdev", "blocks", "mtime_ns", "ctime_ns",
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
_PLAN_FIELDS = {
    "schema_version", "production_ready", "authority", "checkpoint_manifest",
    "producer", "pair_count", "task_count", "tasks", "claim_limits",
    "execution", "plan_digest",
}
_PLAN_AUTHORITY_FIELDS = {
    "input_manifest", "exposure_audit", "source_bytes_verified",
}
_FILE_IDENTITY_FIELDS = {"path", "sha256"}
_CHECKPOINT_IDENTITY_FIELDS = {
    "path", "sha256", "manifest_digest", "global_step", "receipt_digest",
    "file_count", "adapter_config_sha256", "adapter_model_sha256",
    "training_receipt_sha256", "optimizer_sha256",
}
_CHECKPOINT_MANIFEST_FIELDS = {
    "schema_version", "global_step", "receipt_digest", "file_count", "entries",
    "manifest_digest",
}
_TRAINING_RECEIPT_FIELDS = {
    "schema_version", "global_step", "max_steps", "last_loss",
    "last_preclip_gradient_norm", "bernini_commit",
    "bernini_training_files_index_sha256", "veomni_commit",
    "method_source_revision", "method_source_archive_sha256", "checkpoint",
    "checkpoint_tree_sha256", "dataset", "training_contract", "optimizer",
    "distributed", "seed", "target_module_count", "target_modules_sha256",
    "trainable_parameter_count", "resumed_from", "experimental_training",
    "production_claim_forbidden", "scientific_claim_authorized",
    "exploratory_full644", "receipt_digest",
}
_TRAINING_CONTRACT_FIELDS = {
    "model", "single_expert", "noise_tmin", "noise_tmax", "mv2v_flow_shift",
    "num_frames", "latent_frames", "task_source_name", "external_spatial_mask",
    "external_tracking_or_swept_tube", "conditioning", "supervision",
    "target_embedding_or_caption_conditioning", "lora_rank", "lora_alpha",
    "lora_scope", "tokenizer_fix_mistral_regex", "peft_version",
    "transformers_version", "gradient_checkpointing", "objective",
    "preference_weight", "preference_margin", "preference_temperature",
    "dpo_beta", "preservation_weight", "contrastive_negative_kinds",
    "contrastive_negative_schedule", "preservation_branch",
}
_EXPLORATORY_FULL644_FIELDS = {
    "profile", "historical_train_debug_rows", "optimizer_rows_consumed",
    "next_row_index", "row_sequence_prefix", "row_sequence_sha256",
    "no_replacement_within_pass", "complete_one_pass", "historical_dataset_exists",
    "historical_optimizer_contribution_rows",
    "historical_source_receipt_is_not_current_launch_authority",
    "runtime_data_integrity_validated", "dataset_quality_accepted_under_0817",
    "formal_training_dataset_authorized", "formal_heldout_contribution",
    "target_scientific_qualification_complete",
    "matched_frozen_evaluation_required_before_claim", "resume_policy",
    "intermediate_checkpoints_archival_only",
    "interrupted_run_requires_fresh_step0_restart", "dataset_summary_sha256",
    "dataset_summary_digest", "dataset_index_sha256", "dataset_content_signature",
    "source_authority", "indexed_source_and_target_vae_shards_verified_before_training",
    "indexed_source_and_target_vae_shards_reverified_after_training",
}
_TRAINING_CHECKPOINT_FIELDS = {"path", "configs"}
_TRAINING_CHECKPOINT_CONFIG_FIELDS = {
    "model_index.json", "transformer/config.json", "vae/config.json",
}
_TRAINING_DATASET_FIELDS = {
    "path", "rows", "signature", "content_signature", "summary",
}
_TRAINING_DATASET_SUMMARY_FIELDS = {
    "path", "sha256", "summary_digest", "complete", "allow_incomplete",
    "expected_rows", "materialized_rows", "index_path", "index_sha256",
    "indexed_shards_sha256", "dataset_content_signature",
    "reward_selected_synthetic_targets", "arm",
}
_TRAINING_OPTIMIZER_FIELDS = {
    "type", "learning_rate", "weight_decay", "max_gradient_norm",
}
_TRAINING_DISTRIBUTED_FIELDS = {
    "world_size", "ulysses_size", "backend", "same_sample_all_ranks",
    "same_seed_all_ranks", "lora_initialization_seeded_all_ranks",
    "lora_parameters_broadcast_from_rank", "lora_initialization_digest",
    "explicit_lora_gradient_all_reduce",
}
_FULL644_SOURCE_AUTHORITY_FIELDS = {
    "path", "sha256", "membership_rows", "action_family_count", "unique_group_id",
    "unique_source_video_sha256", "raw_parquet_sha256", "vae_index_sha256",
    "vae_summary_sha256", "role",
    "historical_receipt_user_authorization_is_not_current_launch_authority",
}


class MatchedEvalContractError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MatchedEvalContractError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _same_exact_json_value(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _same_exact_json_value(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _same_exact_json_value(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _is_exact_json_int(value: Any, expected: int | None = None) -> bool:
    """Reject JSON bool/float aliases for integer contract fields."""

    return type(value) is int and (expected is None or value == expected)


def _is_exact_json_float(value: Any, expected: float | None = None) -> bool:
    """Reject JSON bool/int aliases for floating-point contract fields."""

    return type(value) is float and (expected is None or value == expected)


def _pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        if key in value:
            raise MatchedEvalContractError("JSON contains a duplicate key")
        value[key] = item
    return value


def _json(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MatchedEvalContractError(f"{label} is not strict UTF-8 JSON") from error


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
        info.st_mtime_ns, info.st_ctime_ns,
    )


def _stable_file(path_value: str | Path, *, expected_sha256: str | None = None,
                 return_bytes: bool = True) -> tuple[bytes | None, str, int]:
    path = Path(path_value).expanduser()
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise MatchedEvalContractError(f"not a plain regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
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
                size += len(block)
                if return_bytes:
                    chunks.append(block)
            fd_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after = path.lstat()
    except MatchedEvalContractError:
        raise
    except OSError as error:
        raise MatchedEvalContractError(f"cannot read stable file {path}: {error}") from error
    if not (_identity(before) == _identity(fd_before) == _identity(fd_after)
            == _identity(after)) or size != before.st_size:
        raise MatchedEvalContractError(f"file identity changed while reading: {path}")
    observed = digest.hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise MatchedEvalContractError(f"file SHA-256 differs: {path}")
    return (b"".join(chunks) if return_bytes else None), observed, size


def _strict_digest(value: Mapping[str, Any], field: str, *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    if not isinstance(declared, str) or declared != object_sha256(unsigned):
        raise MatchedEvalContractError(f"{label} canonical digest differs")
    return declared


def validate_shared8_authority(
    input_manifest: str | Path,
    exposure_audit: str | Path,
    *,
    require_source_bytes: bool = False,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate exact shared-8 metadata, optionally including source bytes."""

    manifest_raw, manifest_sha, _ = _stable_file(
        input_manifest, expected_sha256=INPUT_MANIFEST_SHA256
    )
    exposure_raw, exposure_sha, _ = _stable_file(
        exposure_audit, expected_sha256=EXPOSURE_AUDIT_SHA256
    )
    assert manifest_raw is not None and exposure_raw is not None
    lines = manifest_raw.splitlines()
    if len(lines) != 8 or not manifest_raw.endswith(b"\n"):
        raise MatchedEvalContractError("shared8 manifest must contain exactly 8 rows")
    root = Path(source_root).expanduser() if source_root is not None else None
    if root is not None and (not root.is_absolute() or not root.is_dir() or root.is_symlink()):
        raise MatchedEvalContractError("source_root must be an absolute plain directory")

    rows: list[dict[str, Any]] = []
    expected_keys = {
        "schema_version", "index", "iid", "split", "source_video",
        "instruction", "seed",
    }
    for index, raw in enumerate(lines):
        row = _json(raw, label=f"shared8 row {index}")
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise MatchedEvalContractError(f"shared8 row {index} schema differs")
        if (
            row["schema_version"] != INPUT_ROW_SCHEMA
            or not _same_exact_json_value(row["index"], index)
            or row["iid"] != EXPECTED_IIDS[index]
            or not _same_exact_json_value(row["seed"], 2026 + index)
            or row["split"] != ("test" if index < 5 else "validation")
            or not isinstance(row["instruction"], str)
            or not row["instruction"].strip()
            or "\x00" in row["instruction"]
            or not isinstance(row["source_video"], str)
            or not Path(row["source_video"]).is_absolute()
        ):
            raise MatchedEvalContractError(f"shared8 row {index} authority differs")
        runtime_source = (
            root / row["iid"] / "source.mp4" if root is not None
            else Path(row["source_video"])
        )
        verified = False
        if require_source_bytes:
            if not runtime_source.is_absolute():
                raise MatchedEvalContractError("runtime source path must be absolute")
            _stable_file(
                runtime_source,
                expected_sha256=EXPECTED_SOURCE_SHA256[index],
                return_bytes=False,
            )
            verified = True
        rows.append(
            {
                "index": index,
                "iid": row["iid"],
                "split": row["split"],
                "manifest_source_path": row["source_video"],
                "runtime_source_path": str(runtime_source),
                "source_sha256": EXPECTED_SOURCE_SHA256[index],
                "source_bytes_verified": verified,
                "instruction": row["instruction"],
                "instruction_sha256": hashlib.sha256(
                    row["instruction"].encode("utf-8")
                ).hexdigest(),
                "seed": row["seed"],
            }
        )

    exposure = _json(exposure_raw, label="shared8 exposure audit")
    if not isinstance(exposure, dict) or exposure.get("schema_version") != EXPOSURE_SCHEMA:
        raise MatchedEvalContractError("shared8 exposure schema differs")
    bernini = exposure.get("bernini_full644")
    limits = exposure.get("claim_limits")
    exposure_rows = bernini.get("rows") if isinstance(bernini, dict) else None
    if (
        exposure.get("input_manifest_sha256") != INPUT_MANIFEST_SHA256
        or not isinstance(bernini, dict)
        or not _same_exact_json_value(bernini.get("membership_rows"), 644)
        or not _same_exact_json_value(bernini.get("iid_overlap_count"), 0)
        or not isinstance(exposure_rows, list)
        or len(exposure_rows) != 8
        or any(
            type(row.get("index")) is not int
            or type(row.get("train_seen")) is not bool
            for row in exposure_rows
        )
        or [(r.get("index"), r.get("iid"), r.get("train_seen")) for r in exposure_rows]
        != [(i, EXPECTED_IIDS[i], False) for i in range(8)]
        or not _same_exact_json_value(limits, {
            "iid_heldout_diagnostic": True,
            "content_disjoint_split": False,
            "human_reviewed_labels": False,
            "scientific_generalization_claim_authorized": False,
        })
    ):
        raise MatchedEvalContractError("shared8 exposure authority differs")
    return {
        "input_manifest": {"path": str(Path(input_manifest)), "sha256": manifest_sha},
        "exposure_audit": {"path": str(Path(exposure_audit)), "sha256": exposure_sha},
        "source_bytes_required": require_source_bytes,
        "source_bytes_verified": all(row["source_bytes_verified"] for row in rows),
        "rows": rows,
        "claim_limits": dict(CLAIM_LIMITS),
    }


def validate_terminal_checkpoint_manifest(
    path_value: str | Path, expected_sha256: str
) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or SHA256_RE.fullmatch(expected_sha256) is None:
        raise MatchedEvalContractError("checkpoint manifest path/SHA differs")
    raw, observed, _ = _stable_file(path, expected_sha256=expected_sha256)
    assert raw is not None
    manifest = _json(raw, label="checkpoint manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _CHECKPOINT_MANIFEST_FIELDS
        or raw != canonical_json_bytes(manifest) + b"\n"
    ):
        raise MatchedEvalContractError("checkpoint manifest root differs")
    digest = _strict_digest(manifest, "manifest_digest", label="checkpoint manifest")
    entries = manifest.get("entries")
    if (
        manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA
        or not _same_exact_json_value(manifest.get("global_step"), FULL644_STEP)
        or not isinstance(manifest.get("receipt_digest"), str)
        or SHA256_RE.fullmatch(manifest["receipt_digest"]) is None
        or not isinstance(entries, list)
        or not _same_exact_json_value(manifest.get("file_count"), len(entries))
    ):
        raise MatchedEvalContractError("checkpoint is not terminal full644 step 644")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "size"}:
            raise MatchedEvalContractError("checkpoint manifest row schema differs")
        relative, sha256, size = row["path"], row["sha256"], row["size"]
        if (
            not isinstance(relative, str) or not relative or relative.startswith("/")
            or ".." in Path(relative).parts or relative in seen
            or not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None
            or type(size) is not int or size <= 0
        ):
            raise MatchedEvalContractError("checkpoint manifest row value differs")
        seen.add(relative)
        normalized.append(dict(row))
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise MatchedEvalContractError("checkpoint manifest rows are not sorted")
    required = {
        "adapter/adapter_config.json", "adapter/adapter_model.safetensors",
        "optimizer.pt", "receipt.json",
    }
    if not required.issubset(seen):
        raise MatchedEvalContractError("checkpoint manifest payload closure is incomplete")
    by_path = {row["path"]: row for row in normalized}
    physical: list[str] = []
    for member in sorted(path.parent.rglob("*")):
        relative = member.relative_to(path.parent).as_posix()
        if member.is_symlink():
            raise MatchedEvalContractError(
                f"checkpoint physical closure contains a symlink: {relative}"
            )
        if member.is_file() and member != path:
            physical.append(relative)
    if physical != [row["path"] for row in normalized]:
        raise MatchedEvalContractError("checkpoint physical closure differs")
    for row in normalized:
        _, member_sha, member_size = _stable_file(
            path.parent / row["path"],
            expected_sha256=row["sha256"],
            return_bytes=False,
        )
        if member_sha != row["sha256"] or member_size != row["size"]:
            raise MatchedEvalContractError(
                f"checkpoint member bytes differ: {row['path']}"
            )
    receipt_raw, _, _ = _stable_file(
        path.parent / "receipt.json",
        expected_sha256=by_path["receipt.json"]["sha256"],
    )
    assert receipt_raw is not None
    receipt = _json(receipt_raw, label="full644 training receipt")
    if (
        not isinstance(receipt, dict)
        or set(receipt) != _TRAINING_RECEIPT_FIELDS
        or receipt_raw != canonical_json_bytes(receipt) + b"\n"
    ):
        raise MatchedEvalContractError("full644 training receipt root differs")
    _strict_digest(receipt, "receipt_digest", label="full644 training receipt")
    training = receipt.get("training_contract")
    exploratory = receipt.get("exploratory_full644")
    training_checkpoint = receipt.get("checkpoint")
    dataset = receipt.get("dataset")
    optimizer = receipt.get("optimizer")
    distributed = receipt.get("distributed")
    dataset_summary = dataset.get("summary") if isinstance(dataset, dict) else None
    source_authority = (
        exploratory.get("source_authority")
        if isinstance(exploratory, dict) else None
    )
    configs = (
        training_checkpoint.get("configs")
        if isinstance(training_checkpoint, dict) else None
    )
    expected_training = {
        "model": "Bernini-R-1.3B-Diffusers renderer-only",
        "single_expert": "transformer_1",
        "noise_tmin": 0.0,
        "noise_tmax": 1.0,
        "mv2v_flow_shift": 5.0,
        "num_frames": 81,
        "latent_frames": 21,
        "task_source_name": "mv2v$action_editing_81f",
        "external_spatial_mask": False,
        "external_tracking_or_swept_tube": False,
        "conditioning": ["clean_source_video_vae", "edit_instruction"],
        "supervision": ["noisy_target_video_vae", "target_velocity"],
        "target_embedding_or_caption_conditioning": False,
        "lora_rank": 64,
        "lora_alpha": 64,
        "lora_scope": "all Wan attn1/attn2 q,k,v,out projections",
        "tokenizer_fix_mistral_regex": True,
        "peft_version": "0.19.1",
        "transformers_version": (
            training.get("transformers_version") if isinstance(training, dict) else None
        ),
        "gradient_checkpointing": True,
        "objective": "reference_dpo_preservation",
        "preference_weight": 1.0,
        "preference_margin": 0.05,
        "preference_temperature": 20.0,
        "dpo_beta": 10.0,
        "preservation_weight": 0.25,
        "contrastive_negative_kinds": ["noop", "reverse", "incomplete"],
        "contrastive_negative_schedule": "rotate",
        "preservation_branch": "source_as_target_conditional_identity",
    }
    expected_optimizer = {
        "type": "AdamW", "learning_rate": 0.0001,
        "weight_decay": 0.0, "max_gradient_norm": 1.0,
    }
    expected_distributed = {
        "world_size": 4,
        "ulysses_size": 4,
        "backend": "nccl/rccl",
        "same_sample_all_ranks": True,
        "same_seed_all_ranks": True,
        "lora_initialization_seeded_all_ranks": True,
        "lora_parameters_broadcast_from_rank": 0,
        "lora_initialization_digest": (
            distributed.get("lora_initialization_digest")
            if isinstance(distributed, dict) else None
        ),
        "explicit_lora_gradient_all_reduce": True,
    }
    expected_source_authority = {
        "path": (
            source_authority.get("path") if isinstance(source_authority, dict) else None
        ),
        "sha256": FULL644_SOURCE_AUTHORITY_SHA256,
        "membership_rows": FULL644_STEP,
        "action_family_count": 28,
        "unique_group_id": FULL644_STEP,
        "unique_source_video_sha256": FULL644_STEP,
        "raw_parquet_sha256": (
            "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
        ),
        "vae_index_sha256": FULL644_DATASET_INDEX_SHA256,
        "vae_summary_sha256": FULL644_DATASET_SUMMARY_SHA256,
        "role": "historical_exposed_train_debug_not_heldout",
        "historical_receipt_user_authorization_is_not_current_launch_authority": True,
    }
    expected_dataset_summary = {
        "path": (
            dataset_summary.get("path") if isinstance(dataset_summary, dict) else None
        ),
        "sha256": FULL644_DATASET_SUMMARY_SHA256,
        "summary_digest": FULL644_DATASET_SUMMARY_DIGEST,
        "complete": True,
        "allow_incomplete": False,
        "expected_rows": FULL644_STEP,
        "materialized_rows": FULL644_STEP,
        "index_path": (
            dataset_summary.get("index_path")
            if isinstance(dataset_summary, dict) else None
        ),
        "index_sha256": FULL644_DATASET_INDEX_SHA256,
        "indexed_shards_sha256": (
            dataset_summary.get("indexed_shards_sha256")
            if isinstance(dataset_summary, dict) else None
        ),
        "dataset_content_signature": (
            dataset.get("content_signature") if isinstance(dataset, dict) else None
        ),
        "reward_selected_synthetic_targets": False,
        "arm": None,
    }
    expected_dataset = {
        "path": dataset.get("path") if isinstance(dataset, dict) else None,
        "rows": FULL644_STEP,
        "signature": dataset.get("signature") if isinstance(dataset, dict) else None,
        "content_signature": (
            dataset.get("signature") if isinstance(dataset, dict) else None
        ),
        "summary": expected_dataset_summary,
    }
    expected_exploratory = {
        "profile": FULL644_PROFILE,
        "historical_train_debug_rows": FULL644_STEP,
        "optimizer_rows_consumed": FULL644_STEP,
        "next_row_index": None,
        "row_sequence_prefix": "0..643",
        "row_sequence_sha256": object_sha256(list(range(FULL644_STEP))),
        "no_replacement_within_pass": True,
        "complete_one_pass": True,
        "historical_dataset_exists": True,
        "historical_optimizer_contribution_rows": FULL644_STEP,
        "historical_source_receipt_is_not_current_launch_authority": True,
        "runtime_data_integrity_validated": True,
        "dataset_quality_accepted_under_0817": False,
        "formal_training_dataset_authorized": False,
        "formal_heldout_contribution": 0,
        "target_scientific_qualification_complete": False,
        "matched_frozen_evaluation_required_before_claim": True,
        "resume_policy": "forbidden_for_this_profile",
        "intermediate_checkpoints_archival_only": True,
        "interrupted_run_requires_fresh_step0_restart": True,
        "dataset_summary_sha256": FULL644_DATASET_SUMMARY_SHA256,
        "dataset_summary_digest": FULL644_DATASET_SUMMARY_DIGEST,
        "dataset_index_sha256": FULL644_DATASET_INDEX_SHA256,
        "dataset_content_signature": (
            dataset.get("content_signature") if isinstance(dataset, dict) else None
        ),
        "source_authority": expected_source_authority,
        "indexed_source_and_target_vae_shards_verified_before_training": True,
        "indexed_source_and_target_vae_shards_reverified_after_training": True,
    }
    if (
        not isinstance(training, dict)
        or set(training) != _TRAINING_CONTRACT_FIELDS
        or not isinstance(exploratory, dict)
        or set(exploratory) != _EXPLORATORY_FULL644_FIELDS
        or not isinstance(training_checkpoint, dict)
        or set(training_checkpoint) != _TRAINING_CHECKPOINT_FIELDS
        or not isinstance(configs, dict)
        or set(configs) != _TRAINING_CHECKPOINT_CONFIG_FIELDS
        or not all(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
            for value in configs.values()
        )
        or not isinstance(dataset, dict)
        or set(dataset) != _TRAINING_DATASET_FIELDS
        or not isinstance(dataset_summary, dict)
        or set(dataset_summary) != _TRAINING_DATASET_SUMMARY_FIELDS
        or not isinstance(optimizer, dict)
        or set(optimizer) != _TRAINING_OPTIMIZER_FIELDS
        or not isinstance(distributed, dict)
        or set(distributed) != _TRAINING_DISTRIBUTED_FIELDS
        or not isinstance(source_authority, dict)
        or set(source_authority) != _FULL644_SOURCE_AUTHORITY_FIELDS
        or not _same_exact_json_value(training, expected_training)
        or not _same_exact_json_value(optimizer, expected_optimizer)
        or not _same_exact_json_value(distributed, expected_distributed)
        or not _same_exact_json_value(source_authority, expected_source_authority)
        or not _same_exact_json_value(dataset_summary, expected_dataset_summary)
        or not _same_exact_json_value(dataset, expected_dataset)
        or not _same_exact_json_value(exploratory, expected_exploratory)
        or receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("receipt_digest") != manifest["receipt_digest"]
        or not _same_exact_json_value(receipt.get("global_step"), FULL644_STEP)
        or not _same_exact_json_value(receipt.get("max_steps"), FULL644_STEP)
        or type(receipt.get("last_loss")) is not float
        or type(receipt.get("last_preclip_gradient_norm")) is not float
        or receipt.get("last_preclip_gradient_norm") <= 0
        or receipt.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or receipt.get("bernini_training_files_index_sha256")
        != EXPECTED_BERNINI_TRAINING_FILES_INDEX_SHA256
        or receipt.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or not isinstance(receipt.get("method_source_revision"), str)
        or SHA1_RE.fullmatch(receipt["method_source_revision"]) is None
        or not isinstance(receipt.get("method_source_archive_sha256"), str)
        or SHA256_RE.fullmatch(receipt["method_source_archive_sha256"]) is None
        or not isinstance(training_checkpoint.get("path"), str)
        or not Path(training_checkpoint["path"]).is_absolute()
        or receipt.get("checkpoint_tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
        or not isinstance(dataset.get("path"), str)
        or not Path(dataset["path"]).is_absolute()
        or not _same_exact_json_value(dataset.get("rows"), FULL644_STEP)
        or not isinstance(dataset.get("signature"), str)
        or SHA256_RE.fullmatch(dataset["signature"]) is None
        or dataset.get("content_signature") != dataset["signature"]
        or dataset_summary.get("sha256") != FULL644_DATASET_SUMMARY_SHA256
        or dataset_summary.get("summary_digest") != FULL644_DATASET_SUMMARY_DIGEST
        or dataset_summary.get("complete") is not True
        or dataset_summary.get("allow_incomplete") is not False
        or not _same_exact_json_value(dataset_summary.get("expected_rows"), FULL644_STEP)
        or not _same_exact_json_value(dataset_summary.get("materialized_rows"), FULL644_STEP)
        or not isinstance(dataset_summary.get("path"), str)
        or not Path(dataset_summary["path"]).is_absolute()
        or not isinstance(dataset_summary.get("index_path"), str)
        or not Path(dataset_summary["index_path"]).is_absolute()
        or dataset_summary.get("index_sha256") != FULL644_DATASET_INDEX_SHA256
        or not isinstance(dataset_summary.get("indexed_shards_sha256"), str)
        or SHA256_RE.fullmatch(dataset_summary["indexed_shards_sha256"]) is None
        or dataset_summary.get("dataset_content_signature")
        != dataset["content_signature"]
        or dataset_summary.get("reward_selected_synthetic_targets") is not False
        or dataset_summary.get("arm") is not None
        or not _same_exact_json_value(
            receipt.get("target_module_count"), EXPECTED_TARGET_MODULE_COUNT
        )
        or receipt.get("target_modules_sha256") != EXPECTED_TARGET_MODULES_SHA256
        or not _same_exact_json_value(
            receipt.get("trainable_parameter_count"), FULL644_TRAINABLE_PARAMETER_COUNT
        )
        or not _same_exact_json_value(receipt.get("seed"), FULL644_SEED)
        or receipt.get("resumed_from") is not None
        or receipt.get("experimental_training") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or training.get("model") != "Bernini-R-1.3B-Diffusers renderer-only"
        or training.get("single_expert") != "transformer_1"
        or training.get("noise_tmin") != 0.0
        or training.get("noise_tmax") != 1.0
        or training.get("mv2v_flow_shift") != 5.0
        or training.get("num_frames") != 81
        or training.get("latent_frames") != 21
        or training.get("task_source_name") != "mv2v$action_editing_81f"
        or training.get("external_spatial_mask") is not False
        or training.get("external_tracking_or_swept_tube") is not False
        or training.get("conditioning")
        != ["clean_source_video_vae", "edit_instruction"]
        or training.get("supervision")
        != ["noisy_target_video_vae", "target_velocity"]
        or training.get("target_embedding_or_caption_conditioning") is not False
        or training.get("lora_rank") != 64
        or training.get("lora_alpha") != 64
        or training.get("lora_scope")
        != "all Wan attn1/attn2 q,k,v,out projections"
        or training.get("tokenizer_fix_mistral_regex") is not True
        or training.get("peft_version") != "0.19.1"
        or not isinstance(training.get("transformers_version"), str)
        or not training["transformers_version"]
        or training.get("gradient_checkpointing") is not True
        or training.get("objective") != "reference_dpo_preservation"
        or training.get("preference_weight") != 1.0
        or training.get("preference_margin") != 0.05
        or training.get("preference_temperature") != 20.0
        or training.get("dpo_beta") != 10.0
        or training.get("preservation_weight") != 0.25
        or training.get("contrastive_negative_kinds")
        != ["noop", "reverse", "incomplete"]
        or training.get("contrastive_negative_schedule") != "rotate"
        or training.get("preservation_branch")
        != "source_as_target_conditional_identity"
        or optimizer != {
            "type": "AdamW", "learning_rate": 0.0001,
            "weight_decay": 0.0, "max_gradient_norm": 1.0,
        }
        or distributed.get("world_size") != 4
        or distributed.get("ulysses_size") != 4
        or distributed.get("backend") != "nccl/rccl"
        or distributed.get("same_sample_all_ranks") is not True
        or distributed.get("same_seed_all_ranks") is not True
        or distributed.get("lora_initialization_seeded_all_ranks") is not True
        or distributed.get("lora_parameters_broadcast_from_rank") != 0
        or not isinstance(distributed.get("lora_initialization_digest"), str)
        or SHA256_RE.fullmatch(distributed["lora_initialization_digest"]) is None
        or distributed.get("explicit_lora_gradient_all_reduce") is not True
        or exploratory.get("profile") != FULL644_PROFILE
        or not _same_exact_json_value(
            exploratory.get("historical_train_debug_rows"), FULL644_STEP
        )
        or exploratory.get("complete_one_pass") is not True
        or not _same_exact_json_value(
            exploratory.get("optimizer_rows_consumed"), FULL644_STEP
        )
        or exploratory.get("next_row_index") is not None
        or exploratory.get("row_sequence_prefix") != "0..643"
        or exploratory.get("row_sequence_sha256")
        != object_sha256(list(range(FULL644_STEP)))
        or exploratory.get("no_replacement_within_pass") is not True
        or exploratory.get("historical_dataset_exists") is not True
        or not _same_exact_json_value(
            exploratory.get("historical_optimizer_contribution_rows"), FULL644_STEP
        )
        or exploratory.get(
            "historical_source_receipt_is_not_current_launch_authority"
        ) is not True
        or exploratory.get("runtime_data_integrity_validated") is not True
        or exploratory.get("dataset_quality_accepted_under_0817") is not False
        or exploratory.get("formal_training_dataset_authorized") is not False
        or not _same_exact_json_value(
            exploratory.get("formal_heldout_contribution"), 0
        )
        or exploratory.get("target_scientific_qualification_complete") is not False
        or exploratory.get("matched_frozen_evaluation_required_before_claim") is not True
        or exploratory.get("resume_policy") != "forbidden_for_this_profile"
        or exploratory.get("intermediate_checkpoints_archival_only") is not True
        or exploratory.get("interrupted_run_requires_fresh_step0_restart") is not True
        or exploratory.get("dataset_summary_sha256")
        != FULL644_DATASET_SUMMARY_SHA256
        or exploratory.get("dataset_summary_digest")
        != FULL644_DATASET_SUMMARY_DIGEST
        or exploratory.get("dataset_index_sha256") != FULL644_DATASET_INDEX_SHA256
        or exploratory.get("dataset_content_signature")
        != dataset["content_signature"]
        or exploratory.get(
            "indexed_source_and_target_vae_shards_verified_before_training"
        ) is not True
        or exploratory.get(
            "indexed_source_and_target_vae_shards_reverified_after_training"
        ) is not True
        or source_authority.get("sha256") != FULL644_SOURCE_AUTHORITY_SHA256
        or not isinstance(source_authority.get("path"), str)
        or not Path(source_authority["path"]).is_absolute()
        or not _same_exact_json_value(
            source_authority.get("membership_rows"), FULL644_STEP
        )
        or not _same_exact_json_value(source_authority.get("action_family_count"), 28)
        or not _same_exact_json_value(
            source_authority.get("unique_group_id"), FULL644_STEP
        )
        or not _same_exact_json_value(
            source_authority.get("unique_source_video_sha256"), FULL644_STEP
        )
        or source_authority.get("raw_parquet_sha256")
        != "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
        or source_authority.get("vae_index_sha256") != FULL644_DATASET_INDEX_SHA256
        or source_authority.get("vae_summary_sha256")
        != FULL644_DATASET_SUMMARY_SHA256
        or source_authority.get("role")
        != "historical_exposed_train_debug_not_heldout"
        or source_authority.get(
            "historical_receipt_user_authorization_is_not_current_launch_authority"
        ) is not True
    ):
        raise MatchedEvalContractError("checkpoint training receipt is not terminal full644 R64")
    return {
        "path": str(path),
        "sha256": observed,
        "manifest_digest": digest,
        "global_step": FULL644_STEP,
        "receipt_digest": manifest["receipt_digest"],
        "file_count": len(normalized),
        "adapter_config_sha256": by_path["adapter/adapter_config.json"]["sha256"],
        "adapter_model_sha256": by_path["adapter/adapter_model.safetensors"]["sha256"],
        "training_receipt_sha256": by_path["receipt.json"]["sha256"],
        "optimizer_sha256": by_path["optimizer.pt"]["sha256"],
    }


def validate_producer_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "inference_receipt_schema", "infer_lora_path", "infer_lora_sha256",
        "method_source_revision", "method_source_archive_sha256",
        "ffprobe_path", "ffprobe_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise MatchedEvalContractError("inference producer authority schema differs")
    path = Path(value.get("infer_lora_path", "")).expanduser()
    ffprobe_path = Path(value.get("ffprobe_path", "")).expanduser()
    if (
        value.get("inference_receipt_schema") != INFERENCE_RECEIPT_SCHEMA
        or not path.is_absolute()
        or not isinstance(value.get("infer_lora_sha256"), str)
        or SHA256_RE.fullmatch(value["infer_lora_sha256"]) is None
        or not isinstance(value.get("method_source_revision"), str)
        or SHA1_RE.fullmatch(value["method_source_revision"]) is None
        or not isinstance(value.get("method_source_archive_sha256"), str)
        or SHA256_RE.fullmatch(value["method_source_archive_sha256"]) is None
        or not ffprobe_path.is_absolute()
        or not isinstance(value.get("ffprobe_sha256"), str)
        or SHA256_RE.fullmatch(value["ffprobe_sha256"]) is None
    ):
        raise MatchedEvalContractError("inference producer authority value differs")
    _stable_file(
        path, expected_sha256=value["infer_lora_sha256"], return_bytes=False
    )
    _stable_file(
        ffprobe_path, expected_sha256=value["ffprobe_sha256"], return_bytes=False
    )
    return dict(value)


def build_plan(
    authority: Mapping[str, Any], checkpoint_manifest: Mapping[str, Any],
    output_root_value: str | Path, *, production: bool,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    rows = authority.get("rows")
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute() or not output_root.is_dir() or output_root.is_symlink():
        raise MatchedEvalContractError("output_root must be an absolute plain directory")
    if not isinstance(rows, list) or len(rows) != 8:
        raise MatchedEvalContractError("shared8 authority row count differs")
    if production and authority.get("source_bytes_verified") is not True:
        raise MatchedEvalContractError("production plan requires all source bytes")
    if production:
        # Do not treat a caller-supplied boolean as source authority.  Replay
        # all eight byte checks immediately before committing the plan.
        for row in rows:
            _stable_file(
                row["runtime_source_path"],
                expected_sha256=row["source_sha256"],
                return_bytes=False,
            )
    if not _same_exact_json_value(
        checkpoint_manifest.get("global_step"), FULL644_STEP
    ):
        raise MatchedEvalContractError("adapted arm requires terminal step 644")
    producer_identity = validate_producer_authority(producer)
    tasks: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "case_index": row["index"], "iid": row["iid"],
            "source_video": row["runtime_source_path"],
            "source_video_sha256": row["source_sha256"],
            "instruction": row["instruction"],
            "instruction_sha256": row["instruction_sha256"],
            "seed": row["seed"], "num_inference_steps": 40,
            "source_onset_policy": "none",
        }
        for arm in ("base", "full644"):
            video = output_root / f"case{row['index']:02d}-{arm}.mp4"
            receipt = video.with_name(video.name + ".receipt.json")
            if video.exists() or video.is_symlink() or receipt.exists() or receipt.is_symlink():
                raise MatchedEvalContractError(f"planned output is not fresh: {video}")
            tasks.append(
                {
                    **common,
                    "task_id": f"shared8-{row['index']:02d}-{arm}",
                    "arm": arm,
                    "adapter": None if arm == "base" else {
                        "checkpoint_root": str(Path(checkpoint_manifest["path"]).parent),
                        "checkpoint_manifest": dict(checkpoint_manifest),
                        "adapter_model_sha256": checkpoint_manifest["adapter_model_sha256"],
                        "profile": FULL644_PROFILE,
                    },
                    "output": {
                        "video_path": str(video), "receipt_path": str(receipt),
                        "create_only": True,
                    },
                }
            )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "production_ready": bool(production),
        "authority": {
            "input_manifest": dict(authority["input_manifest"]),
            "exposure_audit": dict(authority["exposure_audit"]),
            "source_bytes_verified": authority.get("source_bytes_verified") is True,
        },
        "checkpoint_manifest": dict(checkpoint_manifest),
        "producer": producer_identity,
        "pair_count": 8,
        "task_count": 16,
        "tasks": tasks,
        "claim_limits": dict(CLAIM_LIMITS),
        "execution": {
            "local_contract_only": True,
            "runner_included": False,
            "training_or_inference_launched": False,
            "all_16_tasks_required_no_cherry_pick": True,
            "external_frozen_runner_attestation_required": True,
            "receipt_contract_alone_cannot_prove_process_execution": True,
        },
    }
    plan["plan_digest"] = object_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    if not isinstance(plan, dict) or set(plan) != _PLAN_FIELDS:
        raise MatchedEvalContractError("plan root differs")
    _strict_digest(plan, "plan_digest", label="plan")
    tasks = plan.get("tasks")
    authority = plan.get("authority")
    execution = plan.get("execution")
    checkpoint = plan.get("checkpoint_manifest")
    producer = plan.get("producer")
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or not _same_exact_json_value(plan.get("pair_count"), 8)
        or not _same_exact_json_value(plan.get("task_count"), 16)
        or not _same_exact_json_value(plan.get("claim_limits"), CLAIM_LIMITS)
        or not isinstance(tasks, list) or len(tasks) != 16
        or type(plan.get("production_ready")) is not bool
        or not isinstance(authority, dict)
        or set(authority) != _PLAN_AUTHORITY_FIELDS
        or not isinstance(authority.get("input_manifest"), dict)
        or set(authority["input_manifest"]) != _FILE_IDENTITY_FIELDS
        or not isinstance(authority.get("exposure_audit"), dict)
        or set(authority["exposure_audit"]) != _FILE_IDENTITY_FIELDS
        or authority.get("input_manifest", {}).get("sha256") != INPUT_MANIFEST_SHA256
        or authority.get("exposure_audit", {}).get("sha256") != EXPOSURE_AUDIT_SHA256
        or authority.get("source_bytes_verified") is not plan.get("production_ready")
        or not _same_exact_json_value(execution, {
            "local_contract_only": True,
            "runner_included": False,
            "training_or_inference_launched": False,
            "all_16_tasks_required_no_cherry_pick": True,
            "external_frozen_runner_attestation_required": True,
            "receipt_contract_alone_cannot_prove_process_execution": True,
        })
    ):
        raise MatchedEvalContractError("plan closure differs")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != _CHECKPOINT_IDENTITY_FIELDS
    ):
        raise MatchedEvalContractError("plan checkpoint manifest identity differs")
    observed_checkpoint = validate_terminal_checkpoint_manifest(
        checkpoint.get("path", ""), checkpoint.get("sha256", "")
    )
    if not _same_exact_json_value(observed_checkpoint, checkpoint):
        raise MatchedEvalContractError("plan checkpoint manifest changed")
    if not isinstance(producer, dict) or validate_producer_authority(producer) != producer:
        raise MatchedEvalContractError("plan producer authority changed")
    canonical_shared8 = validate_shared8_authority(
        authority["input_manifest"]["path"],
        authority["exposure_audit"]["path"],
        require_source_bytes=False,
    )
    canonical_rows = canonical_shared8["rows"]
    ids: set[str] = set()
    outputs: set[str] = set()
    for index in range(8):
        pair = [task for task in tasks if task.get("case_index") == index]
        if len(pair) != 2 or {task.get("arm") for task in pair} != {"base", "full644"}:
            raise MatchedEvalContractError(f"case {index} arm closure differs")
        base = next(task for task in pair if task["arm"] == "base")
        adapted = next(task for task in pair if task["arm"] == "full644")
        canonical_row = canonical_rows[index]
        expected_task_keys = {
            "case_index", "iid", "source_video", "source_video_sha256",
            "instruction", "instruction_sha256", "seed", "num_inference_steps",
            "source_onset_policy", "task_id", "arm", "adapter", "output",
        }
        if set(base) != expected_task_keys or set(adapted) != expected_task_keys:
            raise MatchedEvalContractError(f"case {index} task schema differs")
        for key in (
            "iid", "source_video", "source_video_sha256", "instruction",
            "instruction_sha256", "seed", "num_inference_steps", "source_onset_policy",
        ):
            if not _same_exact_json_value(base.get(key), adapted.get(key)):
                raise MatchedEvalContractError(f"case {index} is not matched on {key}")
        if (
            not _same_exact_json_value(base.get("case_index"), index)
            or not _same_exact_json_value(adapted.get("case_index"), index)
            or base.get("iid") != EXPECTED_IIDS[index]
            or base.get("source_video_sha256") != EXPECTED_SOURCE_SHA256[index]
            or not _same_exact_json_value(base.get("seed"), 2026 + index)
            or base.get("iid") != canonical_row["iid"]
            or base.get("source_video_sha256") != canonical_row["source_sha256"]
            or not _same_exact_json_value(base.get("seed"), canonical_row["seed"])
            or base.get("instruction") != canonical_row["instruction"]
            or base.get("instruction_sha256")
            != canonical_row["instruction_sha256"]
            or not _same_exact_json_value(base.get("num_inference_steps"), 40)
            or base.get("source_onset_policy") != "none"
            or not isinstance(base.get("instruction"), str)
            or base.get("instruction_sha256")
            != hashlib.sha256(base["instruction"].encode("utf-8")).hexdigest()
            or base.get("task_id") != f"shared8-{index:02d}-base"
            or adapted.get("task_id") != f"shared8-{index:02d}-full644"
            or base.get("adapter") is not None
            or not isinstance(adapted.get("adapter"), dict)
            or set(adapted["adapter"]) != {
                "checkpoint_root", "checkpoint_manifest", "adapter_model_sha256",
                "profile",
            }
            or adapted["adapter"].get("profile") != FULL644_PROFILE
            or adapted["adapter"].get("checkpoint_root")
            != str(Path(checkpoint["path"]).parent)
            or adapted["adapter"].get("adapter_model_sha256")
            != checkpoint["adapter_model_sha256"]
            or not _same_exact_json_value(
                adapted["adapter"].get("checkpoint_manifest"), checkpoint
            )
        ):
            raise MatchedEvalContractError(f"case {index} treatment closure differs")
        for task in pair:
            task_id = task.get("task_id")
            output = task.get("output")
            if (
                not isinstance(task_id, str) or task_id in ids
                or not isinstance(output, dict)
                or set(output) != {"video_path", "receipt_path", "create_only"}
            ):
                raise MatchedEvalContractError("task identity differs")
            paths = (output.get("video_path"), output.get("receipt_path"))
            if output.get("create_only") is not True or not all(
                isinstance(path, str) and Path(path).is_absolute() for path in paths
            ) or any(path in outputs for path in paths):
                raise MatchedEvalContractError("task output closure differs")
            ids.add(task_id)
            outputs.update(paths)


def _load_receipt(path_value: str | Path) -> tuple[dict[str, Any], str]:
    raw, sha256, _ = _stable_file(path_value)
    assert raw is not None
    value = _json(raw, label="inference receipt")
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise MatchedEvalContractError("inference receipt is not canonical JSON plus LF")
    _strict_digest(value, "receipt_digest", label="inference receipt")
    return value, sha256


_PUBLICATION_IDENTITY_FIELDS = {
    "device", "inode", "uid", "gid", "mode", "nlink", "rdev", "size",
    "blocks", "mtime_ns", "ctime_ns",
}


def _publication_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    return {
        "device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid,
        "gid": info.st_gid, "mode": info.st_mode, "nlink": info.st_nlink,
        "rdev": info.st_rdev, "size": info.st_size,
        "blocks": getattr(info, "st_blocks", 0), "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _probe_mp4(path: Path, producer: Mapping[str, Any]) -> dict[str, Any]:
    executable = Path(producer["ffprobe_path"])
    _, executable_sha, executable_size = _stable_file(
        executable,
        expected_sha256=producer["ffprobe_sha256"],
        return_bytes=False,
    )
    try:
        result = subprocess.run(
            [
                str(executable), "-v", "error", "-count_frames", "-show_entries",
                "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
                "-of", "json", str(path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            env={"LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise MatchedEvalContractError("ffprobe could not decode output") from error
    if result.returncode != 0:
        raise MatchedEvalContractError(
            "ffprobe rejected output: " + result.stderr.decode("utf-8", "replace")[:300]
        )
    value = _json(result.stdout, label="ffprobe output")
    streams = value.get("streams") if isinstance(value, dict) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise MatchedEvalContractError("output must contain exactly one media stream")
    stream = streams[0]
    try:
        rate = Fraction(stream.get("avg_frame_rate"))
        frames = int(stream.get("nb_read_frames"))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise MatchedEvalContractError("output decoded frame metadata differs") from error
    if (
        stream.get("codec_type") != "video" or frames != 81 or rate != Fraction(25, 1)
        or type(stream.get("width")) is not int or stream["width"] <= 0
        or type(stream.get("height")) is not int or stream["height"] <= 0
    ):
        raise MatchedEvalContractError("output is not an exact 81-frame 25-FPS video")
    _, replay_executable_sha, replay_executable_size = _stable_file(
        executable,
        expected_sha256=producer["ffprobe_sha256"],
        return_bytes=False,
    )
    if (
        replay_executable_sha != executable_sha
        or replay_executable_size != executable_size
    ):
        raise MatchedEvalContractError("externally pinned ffprobe changed during decode")
    return {
        "ffprobe_path": str(executable), "ffprobe_sha256": executable_sha,
        "ffprobe_size": executable_size,
        "frame_count": frames, "fps_num": rate.numerator,
        "fps_den": rate.denominator, "width": stream["width"],
        "height": stream["height"], "stream_count": 1,
    }


def _verify_arm(
    task: Mapping[str, Any], producer: Mapping[str, Any]
) -> dict[str, Any]:
    output = task["output"]
    receipt, receipt_sha = _load_receipt(output["receipt_path"])
    input_value = receipt.get("input")
    preprocessing = receipt.get("preprocessing")
    prompt = receipt.get("prompt_contract")
    sampling = receipt.get("sampling")
    adapter = receipt.get("adapter")
    output_value = receipt.get("output")
    consumption = receipt.get("model_consumption")
    runtime_versions = receipt.get("runtime_versions")
    if not all(
        isinstance(value, dict)
        for value in (
            input_value, preprocessing, prompt, sampling, adapter, output_value,
            consumption, runtime_versions,
        )
    ):
        raise MatchedEvalContractError("inference receipt core schema differs")
    source_authority = input_value.get("source_video_physical_authority")
    attestation = (
        consumption.get("four_rank_attestation")
        if isinstance(consumption, dict) else None
    )
    base_adapter_fields = {
        "enabled", "mode", "strictly_reloaded", "safe_merged_for_inference",
        "tensor_count",
    }
    full644_adapter_fields = {
        "enabled", "mode", "checkpoint_root", "adapter_model_path",
        "adapter_model_sha256", "training_receipt_path", "training_receipt_digest",
        "training_global_step", "strictly_reloaded", "safe_merged_for_inference",
        "tensor_count", "target_modules_sha256", "profile", "lora_rank",
        "lora_alpha", "target_module_count", "checkpoint_manifest",
    }
    expected_input = {
        "source_video_path": task["source_video"],
        "source_video_sha256": task["source_video_sha256"],
        "instruction_utf8_sha256": task["instruction_sha256"],
        "instruction_utf8_bytes": len(task["instruction"].encode("utf-8")),
        "accepted_model_conditions": ["source_video", "edit_instruction"],
        "target_video_argument": False,
        "target_accessed_by_inference": False,
        "external_mask_or_swept_tube": False,
        "external_tracking_pose_or_trajectory": False,
        "reference_image_or_video": False,
        "external_shared_i0": False,
        "source_video_physical_authority": source_authority,
        "source_video_physical_authority_digest": (
            object_sha256(source_authority) if isinstance(source_authority, dict) else None
        ),
        "retained_source_fd_consumed": True,
        "source_video_pre_and_post_decode_rehashed": True,
    }
    expected_preprocessing = {
        "frame_count": 81,
        "fps": 25.0,
        "reported_fps": 25.0,
        "source_input_hw": preprocessing.get("source_input_hw"),
        "source_derived_bucket_hw": preprocessing.get("source_derived_bucket_hw"),
        "max_pixels": 245_760,
        "stride": 16,
        "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
        "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
        "resize": "torchvision_bicubic_antialias_true",
        "external_shared_i0": False,
    }
    expected_sampling = {
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_mode": "v2v_apg",
        "omega_vid": 1.25,
        "omega_img": 0.0,
        "omega_txt": 4.0,
        "omega_scale": 0.8,
        "flow_shift": 5.0,
        "seed": task["seed"],
        "eta": 0.5,
        "norm_threshold": [50.0, 50.0],
        "momentum": 0.0,
        "single_expert": "transformer_1",
        "ulysses_size": 4,
        "rank0_decode_and_save_only": True,
        "source_onset_policy": "none",
    }
    if (
        set(receipt) != _RECEIPT_TOP_FIELDS
        or set(input_value) != _INPUT_FIELDS
        or set(preprocessing) != _PREPROCESSING_FIELDS
        or set(prompt) != _PROMPT_FIELDS
        or set(sampling) != _SAMPLING_FIELDS
        or set(output_value) != _OUTPUT_FIELDS
        or set(consumption) != _MODEL_CONSUMPTION_FIELDS
        or set(runtime_versions) != {
            "torch", "torch_hip", "transformers", "diffusers", "peft"
        }
        or not all(isinstance(value, str) and value for value in runtime_versions.values())
        or not _same_exact_json_value(input_value, expected_input)
        or not _same_exact_json_value(preprocessing, expected_preprocessing)
        or not _same_exact_json_value(sampling, expected_sampling)
        or runtime_versions.get("peft") != "0.19.1"
        or receipt.get("schema_version") != INFERENCE_RECEIPT_SCHEMA
        or receipt.get("infer_lora_source_sha256") != producer["infer_lora_sha256"]
        or receipt.get("method_source_revision") != producer["method_source_revision"]
        or receipt.get("method_source_archive_sha256")
        != producer["method_source_archive_sha256"]
        or receipt.get("bernini_commit") != EXPECTED_BERNINI_COMMIT
        or receipt.get("veomni_commit") != EXPECTED_VEOMNI_COMMIT
        or receipt.get("bernini_inference_files") != EXPECTED_BERNINI_INFERENCE_FILES
        or receipt.get("checkpoint_tree_sha256") != EXPECTED_CHECKPOINT_TREE_SHA256
        or input_value.get("source_video_path") != task["source_video"]
        or input_value.get("source_video_sha256") != task["source_video_sha256"]
        or input_value.get("instruction_utf8_sha256") != task["instruction_sha256"]
        or input_value.get("instruction_utf8_bytes") != len(task["instruction"].encode("utf-8"))
        or input_value.get("accepted_model_conditions")
        != ["source_video", "edit_instruction"]
        or input_value.get("target_video_argument") is not False
        or input_value.get("target_accessed_by_inference") is not False
        or input_value.get("external_mask_or_swept_tube") is not False
        or input_value.get("external_tracking_pose_or_trajectory") is not False
        or input_value.get("reference_image_or_video") is not False
        or input_value.get("external_shared_i0") is not False
        or not isinstance(source_authority, dict)
        or set(source_authority) != _SOURCE_AUTHORITY_FIELDS
        or source_authority.get("path") != task["source_video"]
        or source_authority.get("sha256") != task["source_video_sha256"]
        or any(
            not _is_exact_json_int(source_authority.get(field))
            for field in _SOURCE_AUTHORITY_FIELDS - {"path", "sha256"}
        )
        or source_authority.get("size") <= 0
        or not stat.S_ISREG(source_authority["mode"])
        or input_value.get("source_video_physical_authority_digest")
        != object_sha256(source_authority)
        or not _is_exact_json_int(preprocessing.get("frame_count"), 81)
        or not _is_exact_json_float(preprocessing.get("fps"), 25.0)
        or not _is_exact_json_float(preprocessing.get("reported_fps"), 25.0)
        or not isinstance(preprocessing.get("source_input_hw"), list)
        or len(preprocessing["source_input_hw"]) != 2
        or not all(type(value) is int and value > 0 for value in preprocessing["source_input_hw"])
        or not isinstance(preprocessing.get("source_derived_bucket_hw"), list)
        or len(preprocessing["source_derived_bucket_hw"]) != 2
        or not all(
            type(value) is int and value > 0
            for value in preprocessing["source_derived_bucket_hw"]
        )
        or not _is_exact_json_int(preprocessing.get("max_pixels"), 245_760)
        or not _is_exact_json_int(preprocessing.get("stride"), 16)
        or preprocessing.get("temporal_policy")
        != "all_integer_frames_0_through_80_no_subsampling"
        or preprocessing.get("spatial_policy")
        != "sqrt_max_pixels_then_floor_each_dimension_to_stride"
        or preprocessing.get("resize") != "torchvision_bicubic_antialias_true"
        or preprocessing.get("external_shared_i0") is not False
        or not _same_exact_json_value(prompt, {
            "task": "mv2v",
            "system_prompt_sha256": EXPECTED_SYSTEM_PROMPT_SHA256,
            "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
            "tokenizer_fix_mistral_regex": True,
            "tokenizer_padding_side": "right",
            "max_sequence_length": 512,
            "prompt_enhancer": False,
        })
        or not _same_exact_json_value(sampling.get("seed"), task["seed"])
        or not _is_exact_json_int(sampling.get("num_inference_steps"), 40)
        or sampling.get("source_onset_policy") != "none"
        or not _is_exact_json_int(sampling.get("num_frames"), 81)
        or sampling.get("guidance_mode") != "v2v_apg"
        or not _is_exact_json_float(sampling.get("omega_vid"), 1.25)
        or not _is_exact_json_float(sampling.get("omega_img"), 0.0)
        or not _is_exact_json_float(sampling.get("omega_txt"), 4.0)
        or not _is_exact_json_float(sampling.get("omega_scale"), 0.8)
        or not _is_exact_json_float(sampling.get("flow_shift"), 5.0)
        or not _is_exact_json_float(sampling.get("eta"), 0.5)
        or not _same_exact_json_value(
            sampling.get("norm_threshold"), [50.0, 50.0]
        )
        or not _is_exact_json_float(sampling.get("momentum"), 0.0)
        or sampling.get("single_expert") != "transformer_1"
        or not _is_exact_json_int(sampling.get("ulysses_size"), 4)
        or sampling.get("rank0_decode_and_save_only") is not True
        or output_value.get("path") != output["video_path"]
        or not _is_exact_json_int(output_value.get("frame_count"), 81)
        or not _is_exact_json_float(output_value.get("fps"), 25.0)
        or not _same_exact_json_value(
            output_value.get("height"), preprocessing["source_derived_bucket_hw"][0]
        )
        or not _same_exact_json_value(
            output_value.get("width"), preprocessing["source_derived_bucket_hw"][1]
        )
        or output_value.get("audio_preserved") is not False
        or receipt.get("experimental_inference") is not True
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
        or input_value.get("retained_source_fd_consumed") is not True
        or input_value.get("source_video_pre_and_post_decode_rehashed") is not True
        or not isinstance(input_value.get("source_video_physical_authority_digest"), str)
        or SHA256_RE.fullmatch(input_value["source_video_physical_authority_digest"])
        is None
        or not isinstance(consumption, dict)
        or receipt.get("consumption_input_digest")
        != consumption.get("consumption_input_digest")
        or receipt.get("task_input_digest") != consumption.get("task_input_digest")
        or consumption.get("source_video_sha256") != task["source_video_sha256"]
        or consumption.get("source_video_physical_authority_digest")
        != input_value.get("source_video_physical_authority_digest")
        or consumption.get("all_ranks_use_retained_source_fd") is not True
        or not isinstance(consumption.get("consumption_input_digest"), str)
        or SHA256_RE.fullmatch(consumption["consumption_input_digest"]) is None
        or not isinstance(consumption.get("task_input_digest"), str)
        or SHA256_RE.fullmatch(consumption["task_input_digest"]) is None
        or not isinstance(consumption.get("model_capture_digest"), str)
        or SHA256_RE.fullmatch(consumption["model_capture_digest"]) is None
        or not isinstance(consumption.get("model_view_root"), str)
        or not consumption["model_view_root"]
        or not isinstance(consumption.get("inherited_fd_binding_digest"), str)
        or SHA256_RE.fullmatch(consumption["inherited_fd_binding_digest"]) is None
        or consumption.get("ptrace_authorization_used") is not False
        or not _is_exact_json_int(consumption.get("fd_view_files_authorized"))
        or consumption.get("fd_view_files_authorized") <= 0
        or not _is_exact_json_int(consumption.get("inherited_fd_count"))
        or consumption.get("inherited_fd_count") <= 0
        or not isinstance(attestation, dict)
        or not _is_exact_json_int(attestation.get("world_size"), 4)
        or attestation.get("all_ranks_replayed_exact_fd_views") is not True
        or not isinstance(attestation.get("rank_evidence_digest"), str)
        or SHA256_RE.fullmatch(attestation["rank_evidence_digest"]) is None
        or not _same_exact_json_value(
            attestation.get("ordered_rank_evidence_digests"),
            [attestation["rank_evidence_digest"]] * 4,
        )
        or set(attestation) != {
            "world_size", "all_ranks_replayed_exact_fd_views",
            "rank_evidence_digest", "ordered_rank_evidence_digests",
        }
        or set(adapter) != (
            base_adapter_fields if task["arm"] == "base" else full644_adapter_fields
        )
    ):
        raise MatchedEvalContractError("inference receipt does not match its task")
    output_path = Path(output["video_path"])
    _, video_sha, video_size = _stable_file(output_path, return_bytes=False)
    published_identity = output_value.get("publication_identity")
    prepublication_identity = output_value.get("prepublication_identity")
    actual_identity = _publication_identity(output_path)
    if (
        output_value.get("sha256") != video_sha
        or not _same_exact_json_value(output_value.get("size"), video_size)
        or not isinstance(published_identity, dict)
        or set(published_identity) != _PUBLICATION_IDENTITY_FIELDS
        or not _same_exact_json_value(published_identity, actual_identity)
        or not stat.S_ISREG(actual_identity["mode"])
        or stat.S_IMODE(actual_identity["mode"]) != 0o444
        or not _is_exact_json_int(actual_identity["nlink"], 1)
        or not isinstance(prepublication_identity, dict)
        or set(prepublication_identity) != _PUBLICATION_IDENTITY_FIELDS
        or any(
            not _is_exact_json_int(prepublication_identity.get(field))
            for field in _PUBLICATION_IDENTITY_FIELDS
        )
        or not stat.S_ISREG(prepublication_identity.get("mode", 0))
        or stat.S_IMODE(prepublication_identity["mode"]) != 0o600
        or not _is_exact_json_int(prepublication_identity.get("nlink"), 0)
        or not _same_exact_json_value(
            prepublication_identity.get("size"), video_size
        )
        or output_value.get("anonymous_creation_method") != "linux-sealed-memfd-v1"
        or not _is_exact_json_int(output_value.get("anonymous_seal_mask"), 15)
        or output_value.get("sealed_source_sha256") != video_sha
        or not _same_exact_json_value(
            output_value.get("sealed_source_size"), video_size
        )
        or output_value.get("anonymous_inode_encoded_and_decoded_before_publication")
        is not True
        or output_value.get("create_only_copy_publication_after_decode") is not True
        or output_value.get("sealed_source_and_publication_bytes_equal") is not True
        or output_value.get("retained_inode_encoded_and_replayed") is not True
        or output_value.get("named_output_never_replaced") is not True
    ):
        raise MatchedEvalContractError(
            "output lacks strict anonymous create-only publication evidence"
        )
    media_probe = _probe_mp4(output_path, producer)
    _, replay_sha, replay_size = _stable_file(output_path, return_bytes=False)
    if (
        replay_sha != video_sha or replay_size != video_size
        or not _same_exact_json_value(
            _publication_identity(output_path), actual_identity
        )
        or not _same_exact_json_value(
            media_probe["width"], output_value.get("width")
        )
        or not _same_exact_json_value(
            media_probe["height"], output_value.get("height")
        )
    ):
        raise MatchedEvalContractError("output changed or decoded geometry differs")
    if task["arm"] == "base":
        if (
            not _same_exact_json_value(adapter, {
                "enabled": False, "mode": "frozen_base_no_adapter",
                "strictly_reloaded": False,
                "safe_merged_for_inference": False,
                "tensor_count": 0,
            })
            or consumption.get("adapter_capture_digest") is not None
            or consumption.get("adapter_view_root") is not None
        ):
            raise MatchedEvalContractError("base arm claims an adapter")
    else:
        expected = task["adapter"]
        if (
            adapter.get("enabled") is not True
            or adapter.get("mode") != "lora_safe_merge"
            or adapter.get("strictly_reloaded") is not True
            or adapter.get("safe_merged_for_inference") is not True
            or not _is_exact_json_int(
                adapter.get("training_global_step"), FULL644_STEP
            )
            or adapter.get("profile") != FULL644_PROFILE
            or not _is_exact_json_int(adapter.get("lora_rank"), 64)
            or not _is_exact_json_int(adapter.get("lora_alpha"), 64)
            or not _is_exact_json_int(
                adapter.get("tensor_count"), EXPECTED_ADAPTER_TENSOR_COUNT
            )
            or not _is_exact_json_int(
                adapter.get("target_module_count"), EXPECTED_TARGET_MODULE_COUNT
            )
            or adapter.get("target_modules_sha256")
            != EXPECTED_TARGET_MODULES_SHA256
            or not isinstance(adapter.get("checkpoint_root"), str)
            or not adapter["checkpoint_root"]
            or not isinstance(adapter.get("adapter_model_path"), str)
            or not adapter["adapter_model_path"]
            or not isinstance(adapter.get("training_receipt_path"), str)
            or not adapter["training_receipt_path"]
            or adapter.get("training_receipt_digest")
            != expected["checkpoint_manifest"]["receipt_digest"]
            or adapter.get("adapter_model_sha256") != expected["adapter_model_sha256"]
            or not _same_exact_json_value(
                adapter.get("checkpoint_manifest"), expected["checkpoint_manifest"]
            )
            or not isinstance(consumption.get("adapter_capture_digest"), str)
            or SHA256_RE.fullmatch(consumption["adapter_capture_digest"]) is None
            or not isinstance(consumption.get("adapter_view_root"), str)
            or not consumption["adapter_view_root"]
        ):
            raise MatchedEvalContractError("adapted arm identity differs")
    return {
        "task_id": task["task_id"], "arm": task["arm"],
        "receipt_path": output["receipt_path"], "receipt_file_sha256": receipt_sha,
        "receipt_digest": receipt["receipt_digest"],
        "output_path": output["video_path"], "output_sha256": video_sha,
        "output_size": video_size,
        "media_probe": media_probe,
        "receipt": receipt,
    }


def verify_results(plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_plan(plan)
    if plan.get("production_ready") is not True:
        raise MatchedEvalContractError(
            "result verification requires a production source-byte-authorized plan"
        )
    checkpoint = plan["checkpoint_manifest"]
    if validate_terminal_checkpoint_manifest(
        checkpoint["path"], checkpoint["sha256"]
    ) != checkpoint:
        raise MatchedEvalContractError("terminal checkpoint changed before result verification")
    verified: list[dict[str, Any]] = []
    for index in range(8):
        pair = [task for task in plan["tasks"] if task["case_index"] == index]
        base_task = next(task for task in pair if task["arm"] == "base")
        adapted_task = next(task for task in pair if task["arm"] == "full644")
        base = _verify_arm(base_task, plan["producer"])
        adapted = _verify_arm(adapted_task, plan["producer"])
        if (
            not _same_exact_json_value(
                base["receipt"]["model_consumption"]["model_capture_digest"],
                adapted["receipt"]["model_consumption"]["model_capture_digest"],
            )
        ):
            raise MatchedEvalContractError(
                f"case {index} frozen-base model capture differs between arms"
            )
        for key in ("input", "preprocessing", "prompt_contract", "sampling"):
            if not _same_exact_json_value(
                base["receipt"].get(key), adapted["receipt"].get(key)
            ):
                raise MatchedEvalContractError(f"case {index} pair differs on {key}")
        for key in (
            "method_source_revision", "method_source_archive_sha256", "bernini_commit",
            "infer_lora_source_sha256", "veomni_commit", "bernini_inference_files",
            "checkpoint_tree_sha256",
            "runtime_versions",
        ):
            if not _same_exact_json_value(
                base["receipt"].get(key), adapted["receipt"].get(key)
            ):
                raise MatchedEvalContractError(f"case {index} runtime differs on {key}")
        for item in (base, adapted):
            item.pop("receipt")
            verified.append(item)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "plan_digest": plan["plan_digest"],
        "pair_count": 8, "verified_task_count": 16,
        "all_16_tasks_verified_no_cherry_pick": True,
        "producer_execution_proven_by_receipt_contract": False,
        "external_frozen_runner_attestation_still_required": True,
        "results": verified,
        "claim_limits": dict(CLAIM_LIMITS),
    }
    report["report_digest"] = object_sha256(report)
    return report


def write_create_only(path_value: str | Path, value: Mapping[str, Any]) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute() or not path.parent.is_dir() or path.parent.is_symlink():
        raise MatchedEvalContractError("artifact output parent differs")
    payload = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise MatchedEvalContractError("create-only write made no progress")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as error:
        raise MatchedEvalContractError(f"refusing to overwrite artifact: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def _load_plan(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    raw, _, _ = _stable_file(path, expected_sha256=expected_sha256)
    assert raw is not None
    value = _json(raw, label="matched evaluation plan")
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise MatchedEvalContractError("plan file is not canonical JSON plus LF")
    validate_plan(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    authority = subparsers.add_parser("authority-check")
    authority.add_argument("--input-manifest", required=True)
    authority.add_argument("--exposure-audit", required=True)
    build = subparsers.add_parser("build-plan")
    build.add_argument("--input-manifest", required=True)
    build.add_argument("--exposure-audit", required=True)
    build.add_argument("--source-root")
    build.add_argument("--checkpoint-manifest", required=True)
    build.add_argument("--checkpoint-manifest-sha256", required=True)
    build.add_argument("--infer-lora-source", required=True)
    build.add_argument("--infer-lora-source-sha256", required=True)
    build.add_argument("--method-source-revision", required=True)
    build.add_argument("--method-source-archive-sha256", required=True)
    build.add_argument("--ffprobe", required=True)
    build.add_argument("--ffprobe-sha256", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--output-plan", required=True)
    verify = subparsers.add_parser("verify-results")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--plan-sha256", required=True)
    verify.add_argument("--output-report", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "authority-check":
        value = validate_shared8_authority(args.input_manifest, args.exposure_audit)
        print(canonical_json_bytes(value).decode("utf-8"))
        return 0
    if args.command == "build-plan":
        authority = validate_shared8_authority(
            args.input_manifest, args.exposure_audit,
            require_source_bytes=True, source_root=args.source_root,
        )
        checkpoint = validate_terminal_checkpoint_manifest(
            args.checkpoint_manifest, args.checkpoint_manifest_sha256
        )
        plan = build_plan(
            authority,
            checkpoint,
            args.output_root,
            production=True,
            producer={
                "inference_receipt_schema": INFERENCE_RECEIPT_SCHEMA,
                "infer_lora_path": str(Path(args.infer_lora_source).expanduser()),
                "infer_lora_sha256": args.infer_lora_source_sha256,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
                "ffprobe_path": str(Path(args.ffprobe).expanduser()),
                "ffprobe_sha256": args.ffprobe_sha256,
            },
        )
        print(write_create_only(args.output_plan, plan))
        return 0
    plan = _load_plan(args.plan, args.plan_sha256)
    report = verify_results(plan)
    print(write_create_only(args.output_report, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

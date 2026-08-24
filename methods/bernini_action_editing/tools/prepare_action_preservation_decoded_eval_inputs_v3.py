#!/usr/bin/env python3
"""Build the r7-specific exact15-r2 decoded-eval input authorities.

This tool is deliberately local-only.  It builds deterministic, create-only
authority JSON for the already completed r7 training tree and the four fixed
decoded-eval sources.  It never connects to AUH, launches a process remotely,
reads training loss for selection, or authorizes scientific promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve(strict=True).parents[1]
TOOLS_ROOT = Path(__file__).resolve(strict=True).parent
for import_root in (TOOLS_ROOT, METHOD_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as model_authority
import action_preservation_decoded_eval_verified_release_v1 as runtime
import build_action_preservation_decoded_eval_release_v3 as builder
import prepare_action_preservation_decoded_eval_inputs_v1 as source_authority


SCHEMA = "bernini-action-preservation-decoded-eval-r7-input-authority-v3"
SOURCE_PREPROCESSING_SCHEMA = (
    "bernini-action-preservation-decoded-eval-source-preprocessing-authority-v1"
)
SOURCE_PREPROCESSING_SERIALIZATION = "canonical-json-newline-v1"
R7_TRAINING_ROOT = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/"
    "action_preservation_v2_seed20260818_four_holder_r7"
)
R7_TRAINING_COMPLETE_SHA256 = (
    "2752c4aee78c833c55f7c66bf5bebf84d42f6babe00692dd6cc94b918842409b"
)
R7_TRAINING_COMPLETE_DIGEST = (
    "2134b0922c6c4fb284ebf0d6c3602faef806c58e5cd6066f8aaa47adec9228c4"
)
R7_TRAINING_AUDIT_SHA256 = (
    "70b743eb566ba80406473b3dbabfcacffacd028c811aef34069cbbd3aa5c59c5"
)
R7_TRAINING_AUDIT_DIGEST = (
    "da997ce9956202b233e7d13d9c79ad89f69503c0519007a3942d8a9d79fa5d35"
)
R7_TRAINING_AUDIT_SIZE = 17486
R7_TRAINING_AUDIT_IDENTITY = {
    "device": 48,
    "inode": 13154676782229940972,
    "uid": 2012,
    "gid": 2000,
    "mode": 0o444,
    "nlink": 1,
}
R7_EXACT32_ROWS_DIGEST = (
    "ce14c57623d93f68f49387ccb7042627b7592e255803db2f282371eff24380ff"
)
R7_SOURCE_REVISION = "54a2bafa2a09ddcd26add20c211ea9f055d339c3"
R7_SOURCE_ARCHIVE_SHA256 = (
    "71357c8a4212fd985ffc4f73e8422ae412502756e63c014bc1c260c10c53273f"
)
R7_ADAPTER_RELEASE_MANIFEST_SHA256 = (
    "ce97493465dc0d5b3733be25966f6d2ca909ac24931c4840daa4c73dc4c62198"
)
R7_CONTROLLER_SHA256 = (
    "d522fa711014a5ca5b671448ce24afab14e3dbf63fd9df45b0112745a01dd995"
)
R7_DEPLOYMENT_ENVELOPE_SHA256 = (
    "cea9bdce001b0b6f273f3e555c3a01a6715b92dd4b783f11dabff71043d8fd9b"
)
R7_REMOTE_SOURCE_PREPROCESSING_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/"
    "action_preservation_v2_decoded_eval_exact15_r2_r7_"
    "2752c4ae_207763b7_20260816/"
    "action_preservation_decoded_eval_r7_source_preprocessing_authority_v1.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Exact15R2InputPreparationError(RuntimeError):
    """An r7/exact15-r2 authority or create-only publication differs."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise Exact15R2InputPreparationError(
            "authority value is not finite canonical JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Exact15R2InputPreparationError(message)


def _sha256(value: Any, *, label: str) -> str:
    _require(type(value) is str and _SHA256.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA-256")
    return value


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"{label} contains a duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Exact15R2InputPreparationError(f"cannot decode {label}") from error
    _require(type(value) is dict, f"{label} root differs")
    _require(raw == canonical_json_bytes(value) + b"\n", f"{label} serialization differs")
    return value


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        value.st_mode,
        value.st_nlink,
        value.st_rdev,
        value.st_size,
        getattr(value, "st_blocks", 0),
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def write_create_only(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_absolute() and path.parent.is_dir(), "output path differs")
    _require(not os.path.lexists(path), "refusing to overwrite authority")
    raw = canonical_json_bytes(value) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    _require(hasattr(os, "O_NOFOLLOW"), "no-follow creation is unavailable")
    flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            _require(count > 0, "authority write made no progress")
            offset += count
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        first = _read_fd(descriptor)
        middle = os.fstat(descriptor)
        second = _read_fd(descriptor)
        after = os.fstat(descriptor)
        named = path.lstat()
        _require(
            stat.S_ISREG(before.st_mode)
            and stat.S_IMODE(before.st_mode) == 0o444
            and before.st_nlink == 1
            and _identity(before) == _identity(middle) == _identity(after)
            and _identity(before) == _identity(named)
            and first == raw == second,
            "authority same-FD replay differs",
        )
    finally:
        os.close(descriptor)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": 0o444,
    }


def build_source_preprocessing_authority() -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for source in source_authority.SOURCE_ROWS:
        iid = source["iid"]
        sources.append(
            {
                "iid": iid,
                "source_video_path": str(source_authority._source_video_path(iid)),
                "source_video_sha256": source["source_video_sha256"],
                "source_receipt_path": str(source_authority._source_receipt_path(iid)),
                "source_receipt_sha256": source["source_receipt_sha256"],
                "instruction": source["instruction"],
                "instruction_sha256": source["instruction_sha256"],
                "action_review_contract": source_authority.action_review_contract(source),
                "seed": source["seed"],
            }
        )
    value: dict[str, Any] = {
        "schema_version": SOURCE_PREPROCESSING_SCHEMA,
        "serialization": SOURCE_PREPROCESSING_SERIALIZATION,
        "source_manifest_sha256": source_authority.SOURCE_MANIFEST_SHA256,
        "source_manifest_digest": source_authority.SOURCE_MANIFEST_DIGEST,
        "source_order": [row["iid"] for row in sources],
        "sources": sources,
        "source_video_bytes_consumed_directly": True,
        "precomputed_transformed_source_artifact_used": False,
        "runtime_decode_bound_by_inference_release": True,
        "target_video_available_to_inference": False,
        "training_loss_read_or_used_for_selection": False,
        "remote_launch_performed": False,
        "scientific_promotion_authorized": False,
    }
    value["authority_digest"] = object_sha256(value)
    return value


def _load_source_preprocessing(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    value = _strict_json(raw, label="source preprocessing authority")
    unsigned = dict(value)
    claimed = unsigned.pop("authority_digest", None)
    _require(
        value.get("schema_version") == SOURCE_PREPROCESSING_SCHEMA
        and claimed == object_sha256(unsigned)
        and value == build_source_preprocessing_authority(),
        "source preprocessing authority differs",
    )
    info = resolved.lstat()
    _require(
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o444,
        "source preprocessing authority topology differs",
    )
    return value, {
        "local_path": str(resolved),
        "remote_path": R7_REMOTE_SOURCE_PREPROCESSING_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mode": 0o444,
    }


def build_authority_receipt(
    *, release_dir: Path, model_manifest_path: Path,
    source_preprocessing_path: Path,
) -> dict[str, Any]:
    try:
        release = builder.audit(release_dir.resolve(strict=True), against_workspace=True)
        _expected, model_manifest = model_authority.parse_exact23_manifest(
            model_manifest_path.resolve(strict=True),
            expected_manifest_sha256=model_authority.MODEL_MANIFEST_SHA256,
        )
    except (
        builder.Exact15ReleaseBuildError,
        model_authority.ModelConsumptionAuthorityError,
        FileNotFoundError,
    ) as error:
        raise Exact15R2InputPreparationError(str(error)) from error
    _source_value, source_binding = _load_source_preprocessing(
        source_preprocessing_path
    )
    expected_nonruntime = {
        relative: row
        for relative, row in builder.EXPECTED_COMPONENTS.items()
        if relative != builder.RUNTIME
    }
    _require(
        runtime.RELEASE_GENERATION == builder.RELEASE_GENERATION
        and tuple(runtime.EVAL_RELEASE_MEMBERS) == builder.MEMBER_ORDER
        and dict(runtime.TRUSTED_EXACT15) == expected_nonruntime
        and len(builder.MEMBER_ORDER) == 15,
        "runtime/builder exact15-r2 trust closure differs",
    )
    production_sources = b"\n".join(
        (METHOD_ROOT / relative).read_bytes()
        for relative in (
            "action_preservation_decoded_eval_model_authority_v2.py",
            "action_preservation_decoded_eval_executor_v2.py",
            "action_preservation_decoded_eval_decoder_adapter_v1.py",
            "action_preservation_decoded_eval_verified_release_v1.py",
        )
    )
    _require(
        not any(
            token in production_sources
            for token in (b"PR_SET_PTRACER", b"PR_SET_PTRACER_ANY", b"ptrace(")
        ),
        "forbidden ptrace authorization exists in exact15-r2",
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "release_generation": builder.RELEASE_GENERATION,
        "release": release,
        "release_member_count": 15,
        "release_member_order": list(builder.MEMBER_ORDER),
        "release_component_pin_set_digest": object_sha256(
            [
                {"path": relative, "sha256": row[0], "size": row[1], "mode": row[2]}
                for relative, row in builder.EXPECTED_COMPONENTS.items()
            ]
        ),
        "training_authority": {
            "experiment_root": R7_TRAINING_ROOT,
            "training_complete_sha256": R7_TRAINING_COMPLETE_SHA256,
            "training_complete_digest": R7_TRAINING_COMPLETE_DIGEST,
            "training_audit_sha256": R7_TRAINING_AUDIT_SHA256,
            "training_audit_digest": R7_TRAINING_AUDIT_DIGEST,
            "training_audit_size": R7_TRAINING_AUDIT_SIZE,
            "training_audit_identity": dict(R7_TRAINING_AUDIT_IDENTITY),
            "training_audit_serialization": (
                "python-json-sort-keys-indent2-ensure-ascii-false-finite-newline-v1"
            ),
            "exact_checkpoint_count": 32,
            "exact_checkpoint_rows_digest": R7_EXACT32_ROWS_DIGEST,
            "source_revision": R7_SOURCE_REVISION,
            "source_archive_sha256": R7_SOURCE_ARCHIVE_SHA256,
            "adapter_release_manifest_sha256": R7_ADAPTER_RELEASE_MANIFEST_SHA256,
            "training_controller_sha256": R7_CONTROLLER_SHA256,
            "training_deployment_envelope_sha256": R7_DEPLOYMENT_ENVELOPE_SHA256,
            "read_only_same_fd_double_read_full_identity_audit_completed": True,
            "remote_files_rewritten": False,
        },
        "source_preprocessing_authority": source_binding,
        "model_manifest": model_manifest,
        "model_manifest_sha256": model_authority.MODEL_MANIFEST_SHA256,
        "model_file_count": model_authority.MODEL_FILE_COUNT,
        "model_directory_count": model_authority.MODEL_DIRECTORY_COUNT,
        "model_expected_uid": executor.MODEL_FILE_UID,
        "model_expected_gid": executor.MODEL_FILE_GID,
        "model_expected_device": executor.MODEL_FILE_DEVICE,
        "model_expected_file_mode": executor.MODEL_FILE_MODE,
        "adapter_file_count": len(model_authority.ADAPTER_RELATIVE_FILES),
        "base_control_inherited_fd_count": model_authority.MODEL_FILE_COUNT,
        "candidate_inherited_fd_count": (
            model_authority.MODEL_FILE_COUNT
            + len(model_authority.ADAPTER_RELATIVE_FILES)
        ),
        "directory_fds_holder_private_cloexec": True,
        "leaf_file_fds_inherited_only_at_exact_spawn_boundaries": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
        "holder_model_capture_lifetime_required": True,
        "per_task_pre_post_replay_required": True,
        "final_full_rehash_required": True,
        "publication_after_post_use_close_only": True,
        "aggregate_offline_authority_replay_required": True,
        "training_loss_read_or_used_for_selection": False,
        "automatic_retry": False,
        "remote_upload_performed": False,
        "remote_launch_performed": False,
        "gpu_used": False,
        "scientific_promotion_authorized": False,
        "status": "R7_EXACT15_R2_INPUT_AUTHORITY_PREPARED_NOT_DEPLOYED",
    }
    value["authority_receipt_digest"] = object_sha256(value)
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source-preprocessing")
    source.add_argument("--output", required=True)
    authority = commands.add_parser("authority")
    authority.add_argument("--release-dir", required=True)
    authority.add_argument(
        "--model-manifest",
        default=str(METHOD_ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"),
    )
    authority.add_argument("--source-preprocessing", required=True)
    authority.add_argument("--output", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "source-preprocessing":
        value = build_source_preprocessing_authority()
        binding = write_create_only(Path(args.output).resolve(strict=False), value)
    else:
        value = build_authority_receipt(
            release_dir=Path(args.release_dir),
            model_manifest_path=Path(args.model_manifest),
            source_preprocessing_path=Path(args.source_preprocessing),
        )
        binding = write_create_only(Path(args.output).resolve(strict=False), value)
    print(canonical_json_bytes({"authority": value, "file": binding}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Exact15R2InputPreparationError",
    "R7_EXACT32_ROWS_DIGEST",
    "R7_TRAINING_COMPLETE_SHA256",
    "SCHEMA",
    "SOURCE_PREPROCESSING_SCHEMA",
    "build_authority_receipt",
    "build_source_preprocessing_authority",
    "canonical_json_bytes",
    "object_sha256",
    "write_create_only",
]

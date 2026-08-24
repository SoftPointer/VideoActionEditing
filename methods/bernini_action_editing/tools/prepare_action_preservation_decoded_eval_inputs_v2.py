#!/usr/bin/env python3
"""Prepare the audited exact15 retained-FD decoded-eval authority receipt.

This successor replaces the exact14 P0 status check.  It does not execute a
controller, torchrun, inference, Slurm, SSH, training, retry, or promotion.
It audits a fresh deterministic exact15 release and emits the literal model,
FD, and torchrun authority that a later detached deployment request must bind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

METHOD_ROOT = Path(__file__).resolve(strict=True).parents[1]
TOOLS_ROOT = Path(__file__).resolve(strict=True).parent
for import_root in (TOOLS_ROOT, METHOD_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import build_action_preservation_decoded_eval_release_v2 as builder
import action_preservation_decoded_eval_executor_v2 as executor
import action_preservation_decoded_eval_model_authority_v2 as authority
import action_preservation_decoded_eval_verified_release_v1 as runtime


SCHEMA = "bernini-action-preservation-decoded-eval-input-authority-v2"
MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION = True
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Exact15InputPreparationError(RuntimeError):
    pass


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
        raise Exact15InputPreparationError(
            "input authority is not canonical JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Exact15InputPreparationError(message)


def _write_create_only(path: Path, raw: bytes) -> None:
    _require(path.is_absolute() and path.parent.is_dir(), "output path differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError as error:
        raise Exact15InputPreparationError(
            "refusing to overwrite input authority receipt"
        ) from error
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            _require(written > 0, "input authority write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_authority_receipt(
    *, release_dir: Path, model_manifest_path: Path
) -> dict[str, Any]:
    try:
        release = builder.audit(
            Path(release_dir).resolve(strict=True), against_workspace=True
        )
        _expected, model_manifest = authority.parse_exact23_manifest(
            Path(model_manifest_path).resolve(strict=True),
            expected_manifest_sha256=authority.MODEL_MANIFEST_SHA256,
        )
    except (
        builder.Exact15ReleaseBuildError,
        authority.ModelConsumptionAuthorityError,
        FileNotFoundError,
    ) as error:
        raise Exact15InputPreparationError(str(error)) from error
    expected_nonruntime = {
        relative: row
        for relative, row in builder.EXPECTED_COMPONENTS.items()
        if relative != builder.RUNTIME
    }
    if (
        runtime.RELEASE_GENERATION != builder.RELEASE_GENERATION
        or tuple(runtime.EVAL_RELEASE_MEMBERS) != builder.MEMBER_ORDER
        or dict(runtime.TRUSTED_EXACT15) != expected_nonruntime
        or len(builder.MEMBER_ORDER) != 15
    ):
        raise Exact15InputPreparationError(
            "runtime/builder exact15 trust closure differs"
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
    if any(
        token in production_sources
        for token in (b"PR_SET_PTRACER", b"PR_SET_PTRACER_ANY", b"ptrace(")
    ):
        raise Exact15InputPreparationError(
            "forbidden ptrace authorization exists in exact15"
        )
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "release_generation": builder.RELEASE_GENERATION,
        "release": release,
        "release_member_count": 15,
        "release_member_order": list(builder.MEMBER_ORDER),
        "release_component_pin_set_digest": object_sha256(
            [
                {
                    "path": relative,
                    "sha256": row[0],
                    "size": row[1],
                    "mode": row[2],
                }
                for relative, row in builder.EXPECTED_COMPONENTS.items()
            ]
        ),
        "model_manifest": model_manifest,
        "model_manifest_sha256": authority.MODEL_MANIFEST_SHA256,
        "model_file_count": authority.MODEL_FILE_COUNT,
        "model_directory_count": authority.MODEL_DIRECTORY_COUNT,
        "model_expected_uid": executor.MODEL_FILE_UID,
        "model_expected_gid": executor.MODEL_FILE_GID,
        "model_expected_device": executor.MODEL_FILE_DEVICE,
        "model_expected_file_mode": executor.MODEL_FILE_MODE,
        "adapter_file_count": len(authority.ADAPTER_RELATIVE_FILES),
        "adapter_expected_file_mode": 0o444,
        "base_control_inherited_fd_count": authority.MODEL_FILE_COUNT,
        "candidate_inherited_fd_count": (
            authority.MODEL_FILE_COUNT + len(authority.ADAPTER_RELATIVE_FILES)
        ),
        "directory_fds_holder_private_cloexec": True,
        "leaf_file_fds_inherited_only_at_exact_spawn_boundaries": True,
        "proc_self_fd_consumption_required": True,
        "cross_process_proc_fd_access_forbidden": True,
        "ptrace_authorization_used": False,
        "torchrun_source_sha256": runtime.TORCHRUN_SOURCE_SHA256,
        "torchrun_source_size": runtime.TORCHRUN_SOURCE_SIZE,
        "torchrun_subprocess_handler_relative_path": (
            runtime.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        ),
        "torchrun_subprocess_handler_sha256": (
            runtime.TORCHRUN_SUBPROCESS_HANDLER_SHA256
        ),
        "torchrun_subprocess_handler_size": (
            runtime.TORCHRUN_SUBPROCESS_HANDLER_SIZE
        ),
        "holder_model_capture_lifetime_required": True,
        "per_task_pre_post_replay_required": True,
        "final_full_rehash_required": True,
        "publication_after_post_use_close_only": True,
        "aggregate_offline_authority_replay_required": True,
        "model_consumption_authority_enforced_by_production": (
            MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION
        ),
        "training_loss_read_or_used_for_selection": False,
        "automatic_retry": False,
        "remote_launch_performed": False,
        "gpu_used": False,
        "scientific_promotion_authorized": False,
        "status": "EXACT15_AUTHORITY_INPUTS_PREPARED_NOT_EXECUTED",
    }
    value["authority_receipt_digest"] = object_sha256(value)
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--release-dir", required=True)
    value.add_argument(
        "--model-manifest",
        default=str(METHOD_ROOT / "audits/bernini_r13_ff4c5d4_checkpoint.sha256"),
    )
    value.add_argument("--output")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = build_authority_receipt(
        release_dir=Path(args.release_dir),
        model_manifest_path=Path(args.model_manifest),
    )
    raw = canonical_json_bytes(receipt) + b"\n"
    if args.output is not None:
        _write_create_only(Path(args.output).resolve(strict=False), raw)
    print(raw.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Exact15InputPreparationError",
    "MODEL_CONSUMPTION_AUTHORITY_ENFORCED_BY_PRODUCTION",
    "SCHEMA",
    "build_authority_receipt",
    "canonical_json_bytes",
    "object_sha256",
]

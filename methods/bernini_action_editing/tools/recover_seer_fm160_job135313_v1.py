#!/usr/bin/env python3
"""Create/verify a read-only recovery audit for SEER FM160 Job 135313.

The Slurm job is and remains FAILED.  This audit only determines whether the
already-written checkpoint is complete enough for the preregistered heldout
decode.  It never edits the training output or claims method success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-seer-fm160-job135313-recovery-v1"
TRAINING_RECEIPT_SCHEMA = "bernini-seer-same-state-fm-training-receipt-v1"
JOB_ID = 135313
EXPECTED_LAUNCHER_SHA256 = "2ac5301b2aa70cf3e98502c9768a36a2f889ae5430865a92db136d2e759678be"
EXPECTED_STEPS = (40, 80, 120, 160)
EXPECTED_ADAPTER_TENSORS = 120
EXPECTED_ADAPTER_PARAMETERS = 1_474_560
EXPECTED_ARTIFACT_SHA256 = {
    "stdout": "6a0988052afe365d3c452f3d85148e62403a678ff7efb433828fb41d74aa0f2b",
    "stderr": "e3614b94fc0b3d74dcc93cde0eff15737dc1b5c9f4d8466a9d95c31ec56fe851",
    "latest": "53043bab5f64933d78866b15391b36e6b0d082fde06752084f52f83c101d61a9",
    "final_receipt": "c374395933fb1c4f10282c8ae41f3428f33501ffb54ff64234fbfd3923528eae",
    "final_adapter_config": "25c8f3a6cd501af60ddb6ad2712750d35df23c2622c3579771cacaa10776a369",
    "final_adapter_model": "3dadbd4a1f2551c34942c52bcae2694bb5a695e88b9a6d471f2720f4fc074c5d",
    "final_optimizer": "c8e0b38edb6c199fc25a9bfb07c76be68b7d69f5e710c8be4637d0f4be2e8b6f",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RecoveryError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RecoveryError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise RecoveryError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RecoveryError(f"{label} must be a plain file")
    return path.resolve(strict=True)


def _directory(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/"):
        raise RecoveryError(f"{label} must be absolute and non-root")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise RecoveryError(f"{label} must be a plain directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} root must be an object")
    return value


def _verified_digest_json(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json(_plain_file(path, label=label), label=label)
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if not isinstance(declared, str) or object_sha256(unsigned) != declared:
        raise RecoveryError(f"{label} digest differs")
    return value


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise RecoveryError("recovery receipt output must be absolute and fresh")
    path.parent.resolve(strict=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise RecoveryError("short recovery receipt write")
            view = view[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _json_objects(text: str) -> list[dict[str, Any]]:
    """Extract embedded JSON objects while tolerating NCCL text prefixes."""

    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        cursor = start + consumed
    return rows


def _validate_job(job: Mapping[str, Any]) -> None:
    if job != {
        "job_id": JOB_ID, "state": "FAILED", "exit_code": "1:0",
        "elapsed_seconds": 1921, "node": "auh7-1b-gpu-209",
    }:
        raise RecoveryError("Slurm accounting contract differs")


def _validate_step_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    by_step = {row.get("step"): row for row in rows if type(row.get("step")) is int}
    if set(by_step) != set(range(1, 161)) or len(rows) != 160:
        raise RecoveryError("stdout optimizer-step closure differs")
    for step, row in by_step.items():
        grad = row.get("preclip_gradient_norm")
        if row.get("same_state_exact") != 1.0 or isinstance(grad, bool) or not isinstance(grad, (int, float)) or not math.isfinite(float(grad)) or float(grad) <= 0:
            raise RecoveryError(f"optimizer step {step} finite/same-state evidence differs")


def _require_hardpin(path: Path, expected: str, *, label: str) -> None:
    if file_sha256(path) != expected:
        raise RecoveryError(f"{label} hard pin differs")


def _stable_recursive_digest(value: Any) -> str:
    import torch
    digest = hashlib.sha256()
    def frame(tag: str, payload: bytes = b"") -> None:
        encoded = tag.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big")); digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big")); digest.update(payload)
    def key_bytes(key: Any) -> bytes:
        if key is None: return b"none:"
        if type(key) is bool: return b"bool:" + (b"1" if key else b"0")
        if type(key) is int: return b"int:" + str(key).encode("ascii")
        if type(key) is float: return b"float:" + struct.pack(">d", key)
        if isinstance(key, str):
            encoded = key.encode("utf-8"); return b"str:" + len(encoded).to_bytes(8, "big") + encoded
        if isinstance(key, bytes): return b"bytes:" + len(key).to_bytes(8, "big") + key
        raise RecoveryError(f"unsupported optimizer mapping key: {type(key).__name__}")
    def visit(candidate: Any) -> None:
        if candidate is None: frame("none")
        elif type(candidate) is bool: frame("bool", b"1" if candidate else b"0")
        elif type(candidate) is int: frame("int", str(candidate).encode("ascii"))
        elif type(candidate) is float: frame("float64", struct.pack(">d", candidate))
        elif isinstance(candidate, str): frame("str", candidate.encode("utf-8"))
        elif isinstance(candidate, bytes): frame("bytes", candidate)
        elif isinstance(candidate, torch.Tensor):
            tensor = candidate.detach()
            if tensor.layout != torch.strided: raise RecoveryError("optimizer digest supports only strided tensors")
            tensor = tensor.contiguous()
            frame("tensor-metadata", canonical_json_bytes({"dtype": str(tensor.dtype), "shape": [int(x) for x in tensor.shape]}))
            frame("tensor-bytes", tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes(order="C"))
        elif isinstance(candidate, Mapping):
            ordered = sorted(((key_bytes(key), key) for key in candidate), key=lambda item: item[0])
            frame("mapping-begin", len(ordered).to_bytes(8, "big"))
            for encoded, key in ordered: frame("mapping-key", encoded); visit(candidate[key])
            frame("mapping-end")
        elif isinstance(candidate, list):
            frame("list-begin", len(candidate).to_bytes(8, "big")); [visit(x) for x in candidate]; frame("list-end")
        elif isinstance(candidate, tuple):
            frame("tuple-begin", len(candidate).to_bytes(8, "big")); [visit(x) for x in candidate]; frame("tuple-end")
        else: raise RecoveryError(f"unsupported optimizer checkpoint value: {type(candidate).__name__}")
    visit(value)
    return digest.hexdigest()


def _adapter_identity(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RecoveryError("safetensors is required for adapter audit") from error
    tensor_count = 0
    parameter_count = 0
    logical = hashlib.sha256()
    optimizer = receipt.get("optimizer")
    names = optimizer.get("parameter_names") if isinstance(optimizer, Mapping) else None
    if not isinstance(names, list) or len(names) != EXPECTED_ADAPTER_TENSORS:
        raise RecoveryError("adapter parameter-name closure differs")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
        tensor_count = len(names)
        saved = set(names)
        receipt_names = receipt["optimizer"]["parameter_names"]
        for name in receipt_names:
            saved_name = name.replace(".default.weight", ".weight")
            if saved_name not in saved:
                raise RecoveryError(f"adapter parameter is absent: {name}")
            tensor = handle.get_tensor(saved_name)
            if not bool(tensor.isfinite().all().item()):
                raise RecoveryError(f"adapter tensor is non-finite: {name}")
            if not bool(tensor.ne(0).any().item()):
                raise RecoveryError(f"adapter tensor is all-zero: {name}")
            parameter_count += int(tensor.numel())
            metadata = canonical_json_bytes({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
            logical.update(len(metadata).to_bytes(8, "big"))
            logical.update(metadata)
            import torch
            logical.update(tensor.contiguous().view(torch.uint8).numpy().tobytes(order="C"))
    if tensor_count != EXPECTED_ADAPTER_TENSORS:
        raise RecoveryError("adapter tensor count differs")
    if parameter_count != EXPECTED_ADAPTER_PARAMETERS:
        raise RecoveryError("adapter parameter count differs")
    parameter_digest = logical.hexdigest()
    declared = receipt.get("adapter", {}).get("checkpoint_parameter_digest")
    if parameter_digest != declared:
        raise RecoveryError("adapter logical parameter digest differs")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "checkpoint_parameter_digest": parameter_digest,
    }


def _checkpoint_row(root: Path, step: int) -> dict[str, Any]:
    checkpoint = _directory(root / f"checkpoint-{step:08d}", label=f"checkpoint {step}")
    receipt_path = _plain_file(checkpoint / "receipt.json", label=f"checkpoint {step} receipt")
    receipt = _verified_digest_json(receipt_path, label=f"checkpoint {step} receipt")
    adapter_config = _plain_file(checkpoint / "adapter" / "adapter_config.json", label=f"checkpoint {step} adapter config")
    adapter_model = _plain_file(checkpoint / "adapter" / "adapter_model.safetensors", label=f"checkpoint {step} adapter model")
    optimizer = _plain_file(checkpoint / "optimizer.pt", label=f"checkpoint {step} optimizer")
    evidence = receipt.get("parameter_update_evidence")
    if (
        receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA
        or receipt.get("global_step") != step
        or receipt.get("max_steps") != 160
        or not isinstance(evidence, Mapping)
        or evidence.get("engineering_execution_success") is not True
        or evidence.get("exact_parameter_bytes_changed") is not True
        or evidence.get("method_success_claimed") is not False
        or receipt.get("production_claim_forbidden") is not True
        or receipt.get("scientific_claim_authorized") is not False
    ):
        raise RecoveryError(f"checkpoint {step} training receipt contract differs")
    optimizer_receipt = receipt.get("optimizer")
    if not isinstance(optimizer_receipt, Mapping) or _SHA256.fullmatch(str(optimizer_receipt.get("checkpoint_state_digest"))) is None:
        raise RecoveryError(f"checkpoint {step} optimizer identity differs")
    return {
        "step": step,
        "path": str(checkpoint),
        "receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path), "digest": receipt["receipt_digest"]},
        "adapter_config": {"path": str(adapter_config), "sha256": file_sha256(adapter_config)},
        "adapter_model": _adapter_identity(adapter_model, receipt),
        "optimizer": {"path": str(optimizer), "sha256": file_sha256(optimizer), "checkpoint_state_digest": optimizer_receipt["checkpoint_state_digest"]},
    }


def audit(*, output_root: Path, launcher: Path, stdout: Path, stderr: Path,
          job: Mapping[str, Any]) -> dict[str, Any]:
    _require_hardpin(launcher, EXPECTED_LAUNCHER_SHA256, label="launcher")
    _validate_job(job)
    for path, key in ((stdout, "stdout"), (stderr, "stderr")):
        _require_hardpin(path, EXPECTED_ARTIFACT_SHA256[key], label=key)
    stderr_text = stderr.read_text(encoding="utf-8", errors="strict")
    failure_line = "SEER_FM160_POSTFLIGHT_FAILED: latest pointer differs"
    if stderr_text.count(failure_line) != 1:
        raise RecoveryError("terminal postflight failure evidence differs")
    forbidden = ("Traceback (most recent call last)", "CUDA out of memory", "ChildFailedError")
    if any(token in stderr_text for token in forbidden):
        raise RecoveryError("stderr contains a runtime/training failure")

    steps = [row for row in _json_objects(stdout.read_text(encoding="utf-8", errors="strict")) if type(row.get("step")) is int]
    _validate_step_rows(steps)

    checkpoints = [_checkpoint_row(output_root, step) for step in EXPECTED_STEPS]
    latest_path = _plain_file(output_root / "latest.json", label="latest pointer")
    latest = _read_json(latest_path, label="latest pointer")
    final = checkpoints[-1]
    for row, key in (
        (final["receipt"], "final_receipt"),
        (final["adapter_config"], "final_adapter_config"),
        (final["adapter_model"], "final_adapter_model"),
        (final["optimizer"], "final_optimizer"),
    ):
        if row["sha256"] != EXPECTED_ARTIFACT_SHA256[key]:
            raise RecoveryError(f"{key} hard pin differs")
    try:
        import torch
        optimizer_payload = torch.load(final["optimizer"]["path"], map_location="cpu", weights_only=False)
        optimizer_digest = _stable_recursive_digest(optimizer_payload)
    except Exception as error:
        raise RecoveryError(f"cannot audit optimizer payload: {error}") from error
    if optimizer_digest != final["optimizer"]["checkpoint_state_digest"]:
        raise RecoveryError("optimizer logical payload digest differs")
    expected_latest = {"checkpoint": final["path"], "global_step": 160}
    actual_latest = {**expected_latest, "receipt_digest": final["receipt"]["digest"]}
    if latest != actual_latest:
        raise RecoveryError("latest pointer is not the trainer-authored three-field value")
    if file_sha256(latest_path) != EXPECTED_ARTIFACT_SHA256["latest"]:
        raise RecoveryError("latest hard pin differs")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed_slurm_postflight_recovered_for_heldout_evaluation_only",
        "job": dict(job),
        "failure": {
            "stage": "launcher_terminal_postflight_after_training_and_checkpoint_publication",
            "reason": "launcher_compared_three_field_latest_pointer_to_obsolete_two_field_shape",
            "buggy_expected_latest": expected_latest,
            "actual_latest": actual_latest,
            "failure_line": failure_line,
        },
        "source": {
            "method_source_revision": "6bae2f9a70fb851fd1bf87f5a01a1064a8fe8a6a",
            "method_source_archive_sha256": "ec4064949b1cc8f4dba5d3c15fab375c9392a1407f9610ad01cee24c173bf822",
        },
        "artifacts": {
            "launcher": {"path": str(launcher), "sha256": file_sha256(launcher)},
            "stdout": {"path": str(stdout), "sha256": file_sha256(stdout)},
            "stderr": {"path": str(stderr), "sha256": file_sha256(stderr)},
            "latest": {"path": str(latest_path), "sha256": file_sha256(latest_path), "value": latest},
            "output_root": str(output_root),
        },
        "checkpoints": checkpoints,
        "final_checkpoint": final,
        "validation": {
            "optimizer_steps_exact_1_through_160": True,
            "all_steps_same_state_exact": True,
            "all_steps_positive_finite_gradient": True,
            "checkpoint_steps_exact": list(EXPECTED_STEPS),
            "latest_receipt_digest_matches_final_checkpoint": True,
            "failure_is_post_checkpoint_launcher_schema_bug_only": True,
        },
        "training_execution_complete": True,
        "engineering_execution_success": True,
        "slurm_job_success": False,
        "checkpoint_heldout_eligible": True,
        "method_success": False,
        "method_success_claimed": False,
        "heldout_evaluation_required": True,
        "heldout_decoded_review_required": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "original_artifacts_modified": False,
    }


def verify_receipt(path: Path) -> dict[str, Any]:
    value = _verified_digest_json(path, label="FM160 recovery receipt")
    required = {
        "training_execution_complete": True,
        "engineering_execution_success": True,
        "slurm_job_success": False,
        "checkpoint_heldout_eligible": True,
        "method_success": False,
        "method_success_claimed": False,
        "heldout_evaluation_required": True,
        "heldout_decoded_review_required": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "original_artifacts_modified": False,
    }
    if value.get("schema_version") != SCHEMA_VERSION or any(value.get(k) is not v for k, v in required.items()):
        raise RecoveryError("FM160 recovery authority differs")
    job = value.get("job")
    final = value.get("final_checkpoint")
    artifacts = value.get("artifacts")
    if not isinstance(job, Mapping) or job.get("job_id") != JOB_ID or job.get("state") != "FAILED" or job.get("exit_code") != "1:0":
        raise RecoveryError("FM160 recovery job identity differs")
    if not isinstance(final, Mapping) or final.get("step") != 160:
        raise RecoveryError("FM160 recovery final checkpoint differs")
    if not isinstance(artifacts, Mapping):
        raise RecoveryError("FM160 recovery artifacts differ")
    for row, label in (
        (artifacts.get("launcher"), "launcher"),
        (artifacts.get("stdout"), "stdout"),
        (artifacts.get("stderr"), "stderr"),
        (artifacts.get("latest"), "latest"),
        (final.get("receipt"), "final receipt"),
        (final.get("adapter_config"), "final adapter config"),
        (final.get("adapter_model"), "final adapter model"),
        (final.get("optimizer"), "final optimizer"),
    ):
        if not isinstance(row, Mapping):
            raise RecoveryError(f"FM160 recovery {label} binding differs")
        bound = _plain_file(row.get("path", ""), label=f"FM160 recovery {label}")
        if file_sha256(bound) != row.get("sha256"):
            raise RecoveryError(f"FM160 recovery {label} SHA differs")
    receipt = _verified_digest_json(Path(final["receipt"]["path"]), label="final training receipt")
    latest = _read_json(Path(artifacts["latest"]["path"]), label="latest pointer")
    if (
        receipt.get("receipt_digest") != final["receipt"].get("digest")
        or receipt.get("global_step") != 160
        or latest != artifacts["latest"].get("value")
        or latest.get("receipt_digest") != receipt.get("receipt_digest")
        or latest.get("checkpoint") != final.get("path")
    ):
        raise RecoveryError("FM160 recovery final/latest cross-bind differs")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output-root", required=True)
    create.add_argument("--launcher", required=True)
    create.add_argument("--stdout", required=True)
    create.add_argument("--stderr", required=True)
    create.add_argument("--sacct-row", required=True)
    create.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        result = verify_receipt(Path(args.receipt))
    else:
        try:
            job = json.loads(args.sacct_row)
        except json.JSONDecodeError as error:
            raise RecoveryError(f"sacct row is invalid JSON: {error}") from error
        if not isinstance(job, dict):
            raise RecoveryError("sacct row must be an object")
        unsigned = audit(
            output_root=_directory(args.output_root, label="training output root"),
            launcher=_plain_file(args.launcher, label="training launcher"),
            stdout=_plain_file(args.stdout, label="training stdout"),
            stderr=_plain_file(args.stderr, label="training stderr"),
            job=job,
        )
        result = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output = Path(args.output)
        _write_create_only(output, result)
        verify_receipt(output)
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryError as error:
        print(f"[seer-fm160-recovery] ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)

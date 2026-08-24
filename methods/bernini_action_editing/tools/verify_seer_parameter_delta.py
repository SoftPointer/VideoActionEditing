#!/usr/bin/env python3
"""Fail closed unless a SEER smoke checkpoint contains a real parameter update.

Training completion is deliberately not interpreted as method success here.  The
receipt proves only that the intended four optimizer steps produced finite,
non-zero gradients and changed the saved LoRA tensors from their synchronized
initialization.  Decoded held-out evaluation remains a separate gate.
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
import tempfile
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "bernini-seer-parameter-delta-verification-v1"
B0_TRAINING_RECEIPT_SCHEMA = "bernini-r-1p3b-action-lora-receipt-v2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class VerificationError(RuntimeError):
    """Raised when the saved checkpoint does not prove a parameter update."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain_file(path: Path, *, label: str) -> Path:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise VerificationError(f"missing {label}: {path}") from error
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise VerificationError(f"{label} is not a plain file: {path}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _plain_file(path, label="hashed file").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _plain_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must contain one object")
    return value


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    candidate = dict(receipt)
    declared = candidate.pop("receipt_digest", None)
    actual = object_sha256(candidate)
    if declared != actual:
        raise VerificationError("training receipt digest differs")
    return actual


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise VerificationError(f"{label} is not a lowercase SHA-256")
    return value


def verify_checkpoint(
    *,
    checkpoint: Path,
    expected_steps: int,
    expected_source_archive_sha256: str,
    expected_manifest_sha256: Optional[str] = None,
    expected_owner_spec_sha256: Optional[str] = None,
) -> dict[str, Any]:
    if type(expected_steps) is not int or expected_steps <= 0:
        raise VerificationError("expected steps must be positive")
    if (expected_manifest_sha256 is None) == (expected_owner_spec_sha256 is None):
        raise VerificationError(
            "provide exactly one SEER manifest or owner-spec hash binding"
        )
    expected_binding_kind: str
    expected_binding_sha256: str
    if expected_manifest_sha256 is not None:
        expected_binding_kind = "dataset_manifest"
        expected_binding_sha256 = _sha(
            expected_manifest_sha256, label="expected SEER manifest hash"
        )
    else:
        expected_binding_kind = "owner_spec"
        expected_binding_sha256 = _sha(
            expected_owner_spec_sha256, label="expected SEER owner-spec hash"
        )
    expected_source_archive_sha256 = _sha(
        expected_source_archive_sha256, label="expected source archive hash"
    )

    try:
        root = checkpoint.expanduser().resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"checkpoint is unavailable: {error}") from error
    if not root.is_dir() or root.is_symlink():
        raise VerificationError("checkpoint must be a plain directory")
    receipt_path = root / "receipt.json"
    adapter_path = root / "adapter" / "adapter_model.safetensors"
    receipt = _load_json(receipt_path, label="training receipt")
    receipt_digest = _receipt_digest(receipt)

    if receipt.get("global_step") != expected_steps:
        raise VerificationError("training receipt optimizer-step count differs")
    adapter = receipt.get("adapter")
    metrics = receipt.get("last_metrics")
    immutable = receipt.get("immutable_contract")
    update = receipt.get("parameter_update_evidence")
    seer = receipt.get("seer")
    # Accept exactly the two SEER trainer receipt layouts.  The B0 wrapper is a
    # specialization of train_lora and records its parameter delta under
    # parameter_update_evidence.  The same-state trainer records it under the
    # adapter/immutable contract.  Other mixtures fail closed.
    if (
        receipt.get("schema_version") == B0_TRAINING_RECEIPT_SCHEMA
        and isinstance(update, dict)
        and isinstance(seer, dict)
        and expected_binding_kind == "owner_spec"
    ):
        if (
            update.get("exact_parameter_bytes_changed") is not True
            or update.get("method_success_claimed") is not False
            or seer.get("owner_spec_sha256") != expected_binding_sha256
            or seer.get("training_completion_is_method_success") is not False
            or seer.get("heldout_decoded_review_required") is not True
        ):
            raise VerificationError("B0 SEER parameter-update contract differs")
        initial = _sha(
            update.get("initial_trainable_parameter_digest"),
            label="initial parameter digest",
        )
        final = _sha(
            update.get("final_trainable_parameter_digest"),
            label="final parameter digest",
        )
        gradient_norm = receipt.get("last_preclip_gradient_norm")
        source_hash = receipt.get("method_source_archive_sha256")
        trainer_receipt_layout = "b0_train_lora_specialization"
    elif (
        isinstance(adapter, dict)
        and isinstance(metrics, dict)
        and isinstance(immutable, dict)
        and expected_binding_kind == "dataset_manifest"
    ):
        initial = _sha(
            adapter.get("initialization_digest"), label="initial parameter digest"
        )
        final = _sha(
            adapter.get("checkpoint_parameter_digest"), label="final parameter digest"
        )
        gradient_norm = metrics.get("preclip_gradient_norm")
        manifest_hash = (
            immutable.get("expected_seer_manifest_sha256")
            or immutable.get("seer_manifest_sha256")
            or receipt.get("seer_manifest_sha256")
        )
        if manifest_hash != expected_binding_sha256:
            raise VerificationError("SEER manifest binding differs")
        source_hash = (
            immutable.get("method_source_archive_sha256")
            or receipt.get("method_source_archive_sha256")
        )
        trainer_receipt_layout = "same_state_seer"
    else:
        raise VerificationError("training receipt is not an admitted SEER layout")
    if initial == final:
        raise VerificationError("saved LoRA parameters equal initialization")
    if (
        isinstance(gradient_norm, bool)
        or not isinstance(gradient_norm, (int, float))
        or not math.isfinite(float(gradient_norm))
        or float(gradient_norm) <= 0.0
    ):
        raise VerificationError("final pre-clip gradient norm is not finite and positive")

    if source_hash != expected_source_archive_sha256:
        raise VerificationError("method source archive binding differs")

    adapter_sha256 = file_sha256(adapter_path)
    if adapter_path.stat().st_size <= 0:
        raise VerificationError("saved adapter is empty")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "parameter_delta_verified",
        "checkpoint": str(root),
        "expected_optimizer_steps": expected_steps,
        "completed_optimizer_steps": receipt["global_step"],
        "training_receipt": str(receipt_path),
        "training_receipt_sha256": file_sha256(receipt_path),
        "training_receipt_digest": receipt_digest,
        "seer_binding_kind": expected_binding_kind,
        "seer_binding_sha256": expected_binding_sha256,
        "trainer_receipt_layout": trainer_receipt_layout,
        "method_source_archive_sha256": expected_source_archive_sha256,
        "initial_parameter_digest": initial,
        "final_parameter_digest": final,
        "parameter_digest_changed": True,
        "final_preclip_gradient_norm": float(gradient_norm),
        "adapter_model": str(adapter_path),
        "adapter_model_sha256": adapter_sha256,
        "engineering_execution_success": True,
        "method_success": False,
        "method_success_requires_heldout_decoded_evaluation": True,
    }
    result["receipt_digest"] = object_sha256(result)
    return result


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise VerificationError(f"create-only verification output exists: {path}")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=4)
    binding = parser.add_mutually_exclusive_group(required=True)
    binding.add_argument("--expected-seer-manifest-sha256")
    binding.add_argument("--expected-seer-owner-spec-sha256")
    parser.add_argument("--expected-source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = verify_checkpoint(
        checkpoint=args.checkpoint,
        expected_steps=args.expected_steps,
        expected_manifest_sha256=args.expected_seer_manifest_sha256,
        expected_owner_spec_sha256=args.expected_seer_owner_spec_sha256,
        expected_source_archive_sha256=args.expected_source_archive_sha256,
    )
    output = args.output.expanduser().resolve()
    _atomic_create_json(output, result)
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

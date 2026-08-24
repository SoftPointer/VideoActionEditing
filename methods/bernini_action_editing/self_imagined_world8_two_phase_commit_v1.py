#!/usr/bin/env python3
"""Quarantined WORLD8 shadow preparation for one zero-B Jacobian-QP step.

The v1 safety boundary is deliberately narrower than an in-memory two-phase
parameter update: this module never mutates or restores authoritative LoRA-A
or LoRA-B tensors.  Rank 0 solves one global DP2 x SP4 QP, materializes an
out-of-place FP32 shadow state, and every rank audits the shadow before a
durable PREPARED record can exist.  This version is explicitly first-step-only:
all Action-LoRA-B tensors must be byte-exact FP32 zero at the evidence state.

Consequences of this design:

* no Python renderer callback is accepted, so fresh evaluation cannot mutate
  the live model or write undeclared files from inside the training process;
* the signed scientific contract covers every action/preservation bound, the
  exact checkpoint A/B manifest, topology, QP configuration, and trust radii;
* PREPARE, ABORT, and recovery are durable file protocols rather than claims
  inferred from process-local booleans;
* a process-group failure cannot leave a retained parameter add behind,
  because there is no authoritative add to roll back;
* local MOSAIC-QP receipts continue to say ``world8_apply_authorized=False``;
  this coordinator only produces a quarantined, non-publishable shadow.

Finalize, COMMIT, and publication are deliberately disabled until a separate
fresh-evaluation protocol has undergone its own adversarial review.  This file
does not load Bernini, render video, execute an optimizer, or write a model
checkpoint in place.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
from typing import Any, Mapping, Protocol, Sequence

import torch

import mosaic_starc_stateless_jacobian_qp as mosaic_qp


WORLD_SIZE = 8
SP_SIZE = 4
GLOBAL_RANKS = tuple(range(WORLD_SIZE))
ARM_ORDER = ("dog", "human")
ARM_RANKS = {"dog": (0, 1, 2, 3), "human": (4, 5, 6, 7)}
PRESERVATION_FAMILIES = (
    "identity",
    "camera",
    "background",
    "sharpness",
    "flicker",
    "noop",
)

METHOD_NAME = "bernini-self-imagined-world8-shadow-transaction"
TRANSACTION_SCHEMA = "bernini-world8-shadow-transaction-init-v1"
CHECKPOINT_MANIFEST_SCHEMA = "bernini-action-lora-ab-checkpoint-manifest-v1"
EXTERNAL_AUDIT_SCHEMA = "bernini-external-scientific-audit-v1"
EVIDENCE_PACKET_SCHEMA = "bernini-world8-rank-jacobian-evidence-packet-v1"
CANDIDATE_DESCRIPTOR_SCHEMA = "bernini-world8-shadow-candidate-v1"
SHADOW_MANIFEST_SCHEMA = "bernini-world8-action-lora-shadow-state-v1"
FRESH_REQUEST_SCHEMA = "bernini-world8-fresh-exact81-request-v1"
RANK_WAL_SCHEMA = "bernini-world8-rank-prepared-wal-v1"
GLOBAL_PREPARE_SCHEMA = "bernini-world8-global-prepared-v1"
FRESH_VERIFICATION_SCHEMA = "bernini-world8-fresh-verification-v1"
DECISION_SCHEMA = "bernini-world8-shadow-terminal-decision-v1"

MAX_SMALL_COLLECTIVE_BYTES = 128 * 1024
MAX_CANONICAL_RECEIPT_BYTES = 8 * 1024 * 1024
MAX_EXTERNAL_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_GAUSSIAN_BYTES = 2 * 1024 * 1024 * 1024
MAX_EVIDENCE_ROWS_PER_RANK = 64
MAX_EVIDENCE_ARTIFACT_BYTES = (
    MAX_EVIDENCE_ROWS_PER_RANK * 4 * 393_216
)
MAX_AB_ARTIFACT_BYTES = 256 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ED25519_PUBLIC_KEY_RE = re.compile(r"[0-9a-f]{64}")
_ED25519_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,255}")


class World8ShadowError(RuntimeError):
    """Invalid evidence, receipt, state, or durable transaction."""


class World8FailStopError(World8ShadowError):
    """The process group failed; local authoritative state remained unchanged."""


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
        raise World8ShadowError("value is not canonical finite ASCII JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise World8ShadowError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise World8ShadowError(f"{label} is unsafe")
    return value


def _finite_positive(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise World8ShadowError(f"{label} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise World8ShadowError(f"{label} must be positive and finite")
    return result


def _actor_for_rank(rank: int) -> str:
    if type(rank) is not int or rank not in GLOBAL_RANKS:
        raise World8ShadowError("global rank must be an integer in 0..7")
    return "dog" if rank < SP_SIZE else "human"


def _raw_tensor_bytes(value: torch.Tensor) -> bytes:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise World8ShadowError("tensor bytes require materialized data")
    tensor = value.detach().to(device="cpu").contiguous().clone()
    if tensor.dtype != torch.float32:
        raise World8ShadowError("tensor bytes require exact FP32")
    byte_view = tensor.view(torch.uint8)
    storage = (
        byte_view.untyped_storage()
        if hasattr(byte_view, "untyped_storage")
        else byte_view.storage()
    )
    if hasattr(storage, "_untyped"):
        storage = storage._untyped()
    buffer = io.BytesIO()
    try:
        storage._write_file(buffer, False, False, 1)
    except BaseException as error:
        raise World8ShadowError("cannot export exact tensor bytes") from error
    return buffer.getvalue()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(_raw_tensor_bytes(tensor))
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_stable_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    maximum_bytes: int | None = None,
    require_read_only: bool = False,
    require_single_link: bool = True,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise World8ShadowError(f"cannot open bound file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise World8ShadowError(f"bound path is not a regular file: {path}")
        if require_single_link and before.st_nlink != 1:
            raise World8ShadowError(f"bound path must have exactly one hard link: {path}")
        if require_read_only and before.st_mode & 0o222:
            raise World8ShadowError(f"bound receipt/artifact remains writable: {path}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise World8ShadowError(f"bound file exceeds size policy: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise World8ShadowError(f"bound file was truncated: {path}")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise World8ShadowError(f"bound file changed while reading: {path}")
    raw = b"".join(chunks)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != _sha256(
        expected_sha256, label=f"expected SHA for {path}"
    ):
        raise World8ShadowError(f"bound file SHA-256 differs: {path}")
    return raw


def _hash_stable_file(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    require_read_only: bool = True,
) -> tuple[int, str]:
    """Stream-hash one immutable single-link file without buffering its body."""

    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise World8ShadowError("maximum file size policy differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise World8ShadowError(f"cannot open bound file {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise World8ShadowError(f"bound path is not a regular file: {path}")
        if before.st_nlink != 1:
            raise World8ShadowError(f"bound path must have exactly one hard link: {path}")
        if require_read_only and before.st_mode & 0o222:
            raise World8ShadowError(f"bound receipt/artifact remains writable: {path}")
        if before.st_size > maximum_bytes:
            raise World8ShadowError(f"bound file exceeds size policy: {path}")
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise World8ShadowError(f"bound file was truncated: {path}")
            digest.update(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after:
        raise World8ShadowError(f"bound file changed while hashing: {path}")
    expected = _sha256(expected_sha256, label=f"expected SHA for {path}")
    actual = digest.hexdigest()
    if actual != expected:
        raise World8ShadowError(f"bound file SHA-256 differs: {path}")
    return before.st_size, actual


def _read_canonical_json(
    path: Path,
    *,
    expected_sha256: str,
    require_read_only: bool = True,
) -> tuple[Mapping[str, Any], bytes]:
    raw = _read_stable_file(
        path,
        expected_sha256=expected_sha256,
        maximum_bytes=MAX_CANONICAL_RECEIPT_BYTES,
        require_read_only=require_read_only,
    )
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise World8ShadowError(f"canonical JSON is malformed: {path}") from error
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise World8ShadowError(f"JSON bytes are not canonical: {path}")
    return value, raw


def _create_fsynced_file(
    path: Path,
    raw: bytes,
    *,
    final_mode: int = 0o444,
    allow_identical_existing: bool = True,
) -> str:
    path.parent.mkdir(parents=False, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except FileExistsError:
        if not allow_identical_existing:
            raise World8ShadowError(f"durable path already exists: {path}")
        existing = _read_stable_file(path, require_read_only=True)
        if existing != raw:
            raise World8ShadowError(f"durable path exists with different bytes: {path}")
        return hashlib.sha256(existing).hexdigest()
    except OSError as error:
        raise World8ShadowError(f"cannot create durable path {path}: {error}") from error
    try:
        view = memoryview(raw)
        cursor = 0
        while cursor < len(view):
            written = os.write(descriptor, view[cursor:])
            if written <= 0:
                raise World8ShadowError(f"short durable write: {path}")
            cursor += written
        os.fsync(descriptor)
        os.fchmod(descriptor, final_mode)
        os.fsync(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


def _atomic_publish_record(path: Path, payload: Mapping[str, Any]) -> str:
    """Expose a fully fsynced immutable record through one atomic hard link."""

    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()
    temporary = path.parent / f".{path.name}.{digest}.tmp"
    _create_fsynced_file(temporary, raw, final_mode=0o444)
    try:
        os.link(str(temporary), str(path), follow_symlinks=False)
        _fsync_directory(path.parent)
    except FileExistsError:
        existing = _read_stable_file(path, expected_sha256=digest, require_read_only=True)
        if existing != raw:
            raise World8ShadowError(f"atomic record collision: {path}")
    finally:
        try:
            temporary.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            pass
    return digest


@dataclass(frozen=True)
class CanonicalFileRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ExternalReceiptRef:
    receipt: CanonicalFileRef
    artifact: CanonicalFileRef


@dataclass(frozen=True)
class ExternalVerifierPolicy:
    verifier_id: str
    verifier_executable_sha256: str
    verifier_ed25519_public_key_hex: str


@dataclass(frozen=True)
class QuerySeedGateFiles:
    query_seed: int
    specificity: ExternalReceiptRef
    plus_q_minus_q: ExternalReceiptRef


@dataclass(frozen=True)
class ArmScientificGateFiles:
    actor_family: str
    owner: ExternalReceiptRef
    query_seed_gates: tuple[QuerySeedGateFiles, QuerySeedGateFiles]
    action_gradient_receipts: tuple[ExternalReceiptRef, ...]
    preservation_gradient_receipts: tuple[ExternalReceiptRef, ...]


@dataclass(frozen=True)
class ScientificGateFiles:
    verifier_policy: ExternalVerifierPolicy
    arms: tuple[ArmScientificGateFiles, ArmScientificGateFiles]
    qp_contract: ExternalReceiptRef
    fresh_plan: ExternalReceiptRef


@dataclass(frozen=True)
class FreshInputFiles:
    source_manifest: CanonicalFileRef
    official_gaussian: CanonicalFileRef


class CanonicalExternalVerifier:
    """Read immutable canonical receipts and rehash their declared artifacts."""

    _TOP_LEVEL_KEYS = {
        "schema_version",
        "verifier_id",
        "verifier_executable_sha256",
        "audit_type",
        "verdict",
        "artifact_sha256",
        "artifact_size",
        "bindings",
        "signature_ed25519",
    }

    def __init__(self, policy: ExternalVerifierPolicy) -> None:
        if not isinstance(policy, ExternalVerifierPolicy):
            raise World8ShadowError("external verifier policy type differs")
        self.policy = policy
        _safe_id(policy.verifier_id, label="external verifier ID")
        _sha256(policy.verifier_executable_sha256, label="verifier executable SHA")
        if (
            type(policy.verifier_ed25519_public_key_hex) is not str
            or _ED25519_PUBLIC_KEY_RE.fullmatch(
                policy.verifier_ed25519_public_key_hex
            )
            is None
        ):
            raise World8ShadowError("external verifier Ed25519 public key differs")

    def verify(
        self,
        ref: ExternalReceiptRef,
        *,
        audit_type: str,
        exact_bindings: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(ref, ExternalReceiptRef):
            raise World8ShadowError("external receipt reference type differs")
        expected_receipt_sha = _sha256(ref.receipt.sha256, label="receipt SHA")
        receipt, raw = _read_canonical_json(
            Path(ref.receipt.path), expected_sha256=expected_receipt_sha
        )
        if set(receipt) != self._TOP_LEVEL_KEYS:
            raise World8ShadowError("external audit receipt key closure differs")
        signature_hex = receipt["signature_ed25519"]
        if (
            type(signature_hex) is not str
            or _ED25519_SIGNATURE_RE.fullmatch(signature_hex) is None
        ):
            raise World8ShadowError("external audit Ed25519 signature differs")
        unsigned_receipt = dict(receipt)
        unsigned_receipt.pop("signature_ed25519")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.policy.verifier_ed25519_public_key_hex)
            )
            public_key.verify(
                bytes.fromhex(signature_hex),
                canonical_json_bytes(unsigned_receipt),
            )
        except BaseException as error:
            raise World8ShadowError(
                "external audit Ed25519 authentication failed"
            ) from error
        expected_type = _safe_id(audit_type, label="external audit type")
        if (
            receipt["schema_version"] != EXTERNAL_AUDIT_SCHEMA
            or receipt["verifier_id"] != self.policy.verifier_id
            or receipt["verifier_executable_sha256"]
            != self.policy.verifier_executable_sha256
            or receipt["audit_type"] != expected_type
            or receipt["verdict"] != "PASS"
            or receipt["bindings"] != dict(exact_bindings)
        ):
            raise World8ShadowError(
                f"external {expected_type} receipt identity/verdict/bindings differ"
            )
        artifact_sha = _sha256(receipt["artifact_sha256"], label="artifact SHA")
        if artifact_sha != _sha256(ref.artifact.sha256, label="bound artifact SHA"):
            raise World8ShadowError("receipt and caller disagree on artifact SHA")
        if (
            type(receipt["artifact_size"]) is not int
            or receipt["artifact_size"] < 0
            or receipt["artifact_size"] > MAX_EXTERNAL_ARTIFACT_BYTES
        ):
            raise World8ShadowError("external artifact size differs")
        artifact_size, _ = _hash_stable_file(
            Path(ref.artifact.path),
            expected_sha256=artifact_sha,
            maximum_bytes=MAX_EXTERNAL_ARTIFACT_BYTES,
            require_read_only=True,
        )
        if artifact_size != receipt["artifact_size"]:
            raise World8ShadowError("external artifact length differs")
        return {
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_sha256": artifact_sha,
            "artifact_size": artifact_size,
            "audit_type": expected_type,
            "bindings": dict(exact_bindings),
        }


def _canonical_ab_names() -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name.replace("action_lora_b.weight", "action_lora_a.weight"),
            name,
        )
        for name in mosaic_qp.CANONICAL_PARAMETER_NAMES
    )


CANONICAL_AB_NAMES = _canonical_ab_names()


def checkpoint_manifest_payload(
    *,
    checkpoint_content_receipt_digest: str,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
) -> Mapping[str, Any]:
    """Compute, but do not publish, the exact checkpoint-derived A/B manifest."""

    _validate_ab_sequences(
        ordered_parameters=ordered_parameters,
        ordered_fixed_lora_a=ordered_fixed_lora_a,
        require_manifest=None,
    )
    entries: list[dict[str, Any]] = []
    for (a_name, a_tensor), (b_name, b_tensor) in zip(
        ordered_fixed_lora_a, ordered_parameters
    ):
        entries.extend(
            (
                {
                    "name": a_name,
                    "role": "fixed_lora_a",
                    "shape": list(a_tensor.shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": tensor_sha256(a_tensor),
                },
                {
                    "name": b_name,
                    "role": "action_lora_b",
                    "shape": list(b_tensor.shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": tensor_sha256(b_tensor),
                },
            )
        )
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "checkpoint_content_receipt_digest": _sha256(
            checkpoint_content_receipt_digest,
            label="checkpoint content receipt",
        ),
        "entry_count": len(entries),
        "entries": entries,
        "ab_state_digest": object_sha256(entries),
    }


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        raise World8ShadowError("A/B tensor must own one contiguous offset-zero storage")
    view = tensor.detach()
    if hasattr(view, "untyped_storage"):
        storage = view.untyped_storage()
        nbytes = int(storage.nbytes())
    else:
        storage = view.storage()
        if hasattr(storage, "_untyped"):
            storage = storage._untyped()
        nbytes = (
            int(storage.nbytes())
            if hasattr(storage, "nbytes")
            else int(storage.size()) * int(tensor.element_size())
        )
    required = int(tensor.numel()) * int(tensor.element_size())
    if nbytes != required:
        raise World8ShadowError("A/B tensor storage contains aliases, padding, or extra bytes")
    index = tensor.device.index
    return (tensor.device.type, index, int(storage.data_ptr()), nbytes)


def _validate_ab_sequences(
    *,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
    require_manifest: Mapping[str, Any] | None,
) -> str:
    if isinstance(ordered_parameters, Mapping) or not isinstance(
        ordered_parameters, Sequence
    ) or isinstance(ordered_fixed_lora_a, Mapping) or not isinstance(
        ordered_fixed_lora_a, Sequence
    ):
        raise World8ShadowError("A/B states must be explicit ordered sequences")
    if len(ordered_parameters) != 32 or len(ordered_fixed_lora_a) != 32:
        raise World8ShadowError("A/B closure must contain exactly 32 A and 32 B tensors")
    identities: set[tuple[str, int | None, int, int]] = set()
    entries: list[dict[str, Any]] = []
    for ordinal, ((expected_a, expected_b), a_item, b_item) in enumerate(
        zip(CANONICAL_AB_NAMES, ordered_fixed_lora_a, ordered_parameters)
    ):
        if (
            not isinstance(a_item, tuple)
            or len(a_item) != 2
            or not isinstance(b_item, tuple)
            or len(b_item) != 2
        ):
            raise World8ShadowError(f"A/B entry {ordinal} differs")
        a_name, a_tensor = a_item
        b_name, b_tensor = b_item
        if a_name != expected_a or b_name != expected_b:
            raise World8ShadowError(f"A/B canonical name/order differs at {ordinal}")
        for role, name, tensor, shape in (
            ("fixed_lora_a", a_name, a_tensor, mosaic_qp.CANONICAL_A_SHAPE),
            ("action_lora_b", b_name, b_tensor, mosaic_qp.CANONICAL_B_SHAPE),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type == "meta"
                or tensor.dtype != torch.float32
                or tuple(tensor.shape) != tuple(shape)
                or not bool(torch.isfinite(tensor.detach()).all().item())
            ):
                raise World8ShadowError(f"{role} tensor {name} geometry/dtype differs")
            if role == "fixed_lora_a" and tensor.requires_grad:
                raise World8ShadowError(f"fixed LoRA-A tensor {name} requires grad")
            if role == "action_lora_b" and bool(
                torch.count_nonzero(tensor.detach()).item()
            ):
                raise World8ShadowError(
                    f"first-step-only Action-LoRA-B tensor {name} is not exact zero"
                )
            identity = _storage_identity(tensor)
            if identity in identities:
                raise World8ShadowError("A/B manifest contains aliased live storage")
            identities.add(identity)
            entries.append(
                {
                    "name": name,
                    "role": role,
                    "shape": list(shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": tensor_sha256(tensor),
                }
            )
    state_digest = object_sha256(entries)
    if require_manifest is not None:
        if set(require_manifest) != {
            "schema_version",
            "checkpoint_content_receipt_digest",
            "entry_count",
            "entries",
            "ab_state_digest",
        }:
            raise World8ShadowError("checkpoint A/B manifest key closure differs")
        if (
            require_manifest["schema_version"] != CHECKPOINT_MANIFEST_SCHEMA
            or require_manifest["entry_count"] != 64
            or require_manifest["entries"] != entries
            or require_manifest["ab_state_digest"] != state_digest
        ):
            raise World8ShadowError("live A/B tensors differ from checkpoint manifest")
        _sha256(
            require_manifest["checkpoint_content_receipt_digest"],
            label="manifest checkpoint content receipt",
        )
    return state_digest


@dataclass(frozen=True)
class _ABSnapshot:
    a_tensors: tuple[torch.Tensor, ...]
    b_tensors: tuple[torch.Tensor, ...]
    state_digest: str
    manifest_sha256: str
    checkpoint_content_receipt_digest: str


@dataclass(frozen=True)
class _ValidatedQPContract:
    layout: mosaic_qp.FixedParameterLayout
    layer_trust_radii: tuple[mosaic_qp.LayerTrustRadius, ...]
    global_trust_radius: float
    config: mosaic_qp.JacobianQPConfig
    bindings: Mapping[str, Any]


@dataclass(frozen=True)
class _ValidatedSnapshotTrust:
    layout: mosaic_qp.FixedParameterLayout
    layer_trust_radii: tuple[mosaic_qp.LayerTrustRadius, ...]
    rows: tuple[Mapping[str, Any], ...]


def _snapshot_from_checkpoint_manifest(
    *,
    checkpoint_manifest: CanonicalFileRef,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
) -> _ABSnapshot:
    manifest, _ = _read_canonical_json(
        Path(checkpoint_manifest.path),
        expected_sha256=_sha256(checkpoint_manifest.sha256, label="checkpoint manifest SHA"),
    )
    state_digest = _validate_ab_sequences(
        ordered_parameters=ordered_parameters,
        ordered_fixed_lora_a=ordered_fixed_lora_a,
        require_manifest=manifest,
    )
    return _ABSnapshot(
        a_tensors=tuple(
            tensor.detach().to(device="cpu").clone()
            for _, tensor in ordered_fixed_lora_a
        ),
        b_tensors=tuple(
            tensor.detach().to(device="cpu").clone()
            for _, tensor in ordered_parameters
        ),
        state_digest=state_digest,
        manifest_sha256=checkpoint_manifest.sha256,
        checkpoint_content_receipt_digest=manifest[
            "checkpoint_content_receipt_digest"
        ],
    )


def _qp_config_payload(config: mosaic_qp.JacobianQPConfig) -> Mapping[str, Any]:
    if not isinstance(config, mosaic_qp.JacobianQPConfig):
        raise World8ShadowError("Jacobian-QP config type differs")
    config.validate()
    payload = {
        name: getattr(config, name) for name in config.__dataclass_fields__
    }
    canonical_json_bytes(payload)
    return payload


def _validated_qp_contract(
    *,
    snapshot: _ABSnapshot,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    evidence: mosaic_qp.DP2SP4Evidence,
    topology_receipt_digest: str,
    global_trust_radius: float,
    layer_trust_radii: Sequence[mosaic_qp.LayerTrustRadius],
    config: mosaic_qp.JacobianQPConfig,
) -> _ValidatedQPContract:
    """Derive the exact signed QP contract from the checkpoint snapshot."""

    topology = _sha256(topology_receipt_digest, label="topology receipt")
    if evidence.topology_receipt_digest != topology:
        raise World8ShadowError("evidence/topology receipt binding differs")
    global_radius = _finite_positive(
        global_trust_radius, label="global trust radius"
    )
    config_payload = _qp_config_payload(config)
    snapshot_trust = _validated_snapshot_trust(
        snapshot=snapshot,
        ordered_parameters=ordered_parameters,
        layer_trust_radii=layer_trust_radii,
    )
    layout = snapshot_trust.layout
    validated_trust = snapshot_trust.layer_trust_radii
    try:
        union = mosaic_qp._validate_and_union_dp2_sp4_evidence(  # type: ignore[attr-defined]
            layout=layout,
            evidence=evidence,
            minimum_row_norm=config.minimum_row_norm,
        )
    except BaseException as error:
        raise World8ShadowError("QP contract evidence/trust validation failed") from error
    if union.checkpoint_content_receipt_digest != (
        snapshot.checkpoint_content_receipt_digest
    ):
        raise World8ShadowError(
            "evidence checkpoint digest differs from checkpoint manifest"
        )
    action_rows = [
        {
            "row_id": row.row_id,
            "actor_family": row.actor_family,
            "row_sha256": tensor_sha256(row.values),
            "minimum_dot": float(row.minimum_dot),
            "layout_digest": row.layout_digest,
            "checkpoint_content_receipt_digest": (
                row.checkpoint_content_receipt_digest
            ),
            "parameter_state_sha256": row.parameter_state_sha256,
            "gradient_computation_receipt_digest": (
                row.gradient_computation_receipt_digest
            ),
        }
        for row in union.action_rows
    ]
    preservation_rows = [
        {
            "row_id": row.row_id,
            "family": row.family,
            "row_sha256": tensor_sha256(row.values),
            "maximum_absolute_dot": float(row.maximum_absolute_dot),
            "layout_digest": row.layout_digest,
            "checkpoint_content_receipt_digest": (
                row.checkpoint_content_receipt_digest
            ),
            "parameter_state_sha256": row.parameter_state_sha256,
            "gradient_computation_receipt_digest": (
                row.gradient_computation_receipt_digest
            ),
        }
        for row in union.preservation_rows
    ]
    bindings = {
        "checkpoint_manifest_sha256": snapshot.manifest_sha256,
        "checkpoint_content_receipt_digest": (
            snapshot.checkpoint_content_receipt_digest
        ),
        "base_ab_state_digest": snapshot.state_digest,
        "parameter_layout_digest": layout.layout_digest,
        "zero_b_parameter_state_sha256": layout.parameter_state_sha256,
        "topology_receipt_digest": topology,
        "world_size": WORLD_SIZE,
        "dp_size": 2,
        "sp_size": SP_SIZE,
        "actor_families": list(ARM_ORDER),
        "first_step_exact_zero_b_required": True,
        "global_trust_radius": global_radius,
        "layer_trust_radii": list(snapshot_trust.rows),
        "qp_config": config_payload,
        "action_constraints": action_rows,
        "preservation_constraints": preservation_rows,
    }
    canonical_json_bytes(bindings)
    return _ValidatedQPContract(
        layout, validated_trust, global_radius, config, bindings
    )


def _validated_snapshot_trust(
    *,
    snapshot: _ABSnapshot,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    layer_trust_radii: Sequence[mosaic_qp.LayerTrustRadius],
) -> _ValidatedSnapshotTrust:
    layout = mosaic_qp.FixedParameterLayout.from_ordered_parameters(
        ordered_parameters, require_exact_zero_b=True
    )
    try:
        validated_trust = mosaic_qp._validate_layer_trust_radii(  # type: ignore[attr-defined]
            layout, layer_trust_radii
        )
    except BaseException as error:
        raise World8ShadowError("snapshot trust-radius validation failed") from error
    rows: list[Mapping[str, Any]] = []
    for ordinal, (bound, snapshot_a, (expected_a_name, _)) in enumerate(
        zip(validated_trust, snapshot.a_tensors, CANONICAL_AB_NAMES)
    ):
        if bound.fixed_lora_a_parameter_name != expected_a_name or not torch.equal(
            bound.fixed_lora_a.detach().cpu(), snapshot_a
        ):
            raise World8ShadowError(
                f"trust-radius fixed LoRA-A differs from checkpoint snapshot at {ordinal}"
            )
        rows.append(
            {
                "parameter_name": bound.parameter_name,
                "fixed_lora_a_parameter_name": bound.fixed_lora_a_parameter_name,
                "fixed_lora_a_sha256": tensor_sha256(snapshot_a),
                "maximum_relative_delta": float(bound.maximum_relative_delta),
                "reference_effective_weight_norm": float(
                    bound.reference_effective_weight_norm
                ),
                "maximum_absolute_delta_norm": float(
                    bound.maximum_absolute_delta_norm
                ),
                "fixed_gauge_receipt_digest": _sha256(
                    bound.fixed_gauge_receipt_digest,
                    label="fixed-gauge receipt",
                ),
                "reference_weight_receipt_digest": _sha256(
                    bound.reference_weight_receipt_digest,
                    label="reference-weight receipt",
                ),
            }
        )
    return _ValidatedSnapshotTrust(layout, validated_trust, tuple(rows))


def build_signed_qp_contract_bindings(
    *,
    checkpoint_manifest: CanonicalFileRef,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
    evidence: mosaic_qp.DP2SP4Evidence,
    topology_receipt_digest: str,
    global_trust_radius: float,
    layer_trust_radii: Sequence[mosaic_qp.LayerTrustRadius],
    config: mosaic_qp.JacobianQPConfig = mosaic_qp.JacobianQPConfig(),
) -> Mapping[str, Any]:
    """Build the canonical bindings that the external QP-contract signer sees."""

    snapshot = _snapshot_from_checkpoint_manifest(
        checkpoint_manifest=checkpoint_manifest,
        ordered_parameters=ordered_parameters,
        ordered_fixed_lora_a=ordered_fixed_lora_a,
    )
    return dict(
        _validated_qp_contract(
            snapshot=snapshot,
            ordered_parameters=ordered_parameters,
            evidence=evidence,
            topology_receipt_digest=topology_receipt_digest,
            global_trust_radius=global_trust_radius,
            layer_trust_radii=layer_trust_radii,
            config=config,
        ).bindings
    )
def _assert_authoritative_unchanged(
    snapshot: _ABSnapshot,
    *,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
) -> None:
    if len(ordered_parameters) != len(snapshot.b_tensors) or len(
        ordered_fixed_lora_a
    ) != len(snapshot.a_tensors):
        raise World8FailStopError("authoritative A/B closure changed")
    if not all(
        torch.equal(before, current.detach().to(device="cpu"))
        for before, (_, current) in zip(snapshot.b_tensors, ordered_parameters)
    ) or not all(
        torch.equal(before, current.detach().to(device="cpu"))
        for before, (_, current) in zip(snapshot.a_tensors, ordered_fixed_lora_a)
    ):
        raise World8FailStopError("authoritative A/B changed during shadow transaction")


class World8SmallCollective(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def all_gather_small(self, value: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]: ...

    def broadcast_small(self, value: Mapping[str, Any] | None, *, src: int) -> Mapping[str, Any]: ...


def _validate_small_payload(value: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise World8ShadowError(f"{label} payload type differs")
    if len(canonical_json_bytes(dict(value))) > MAX_SMALL_COLLECTIVE_BYTES:
        raise World8ShadowError(f"{label} exceeds the small-control payload limit")


class TorchWorld8SmallCollective:
    """WORLD-group-only control collective; subgroup construction is absent in v1."""

    def __init__(self) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise World8ShadowError("torch.distributed world group is not initialized")
        if dist.get_world_size() != WORLD_SIZE:
            raise World8ShadowError("shadow transaction requires exact WORLD8")
        rank = int(dist.get_rank())
        if rank not in GLOBAL_RANKS:
            raise World8ShadowError("WORLD8 rank is outside 0..7")
        raw_local_rank = os.environ.get("LOCAL_RANK")
        if raw_local_rank is None or not raw_local_rank.isdecimal():
            raise World8ShadowError("LOCAL_RANK must be a decimal integer")
        local_rank = int(raw_local_rank)
        if local_rank != rank:
            raise World8ShadowError("v1 requires single-node global/local rank identity")
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
            raise World8ShadowError("v1 production transport requires eight visible CUDA/HIP devices")
        if torch.cuda.current_device() != local_rank:
            raise World8ShadowError("CUDA current device is not bound to LOCAL_RANK")
        self._dist = dist
        self._rank = rank

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    def all_gather_small(self, value: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        _validate_small_payload(value, label="all-gather")
        rows: list[Any] = [None] * WORLD_SIZE
        self._dist.all_gather_object(rows, dict(value))
        for row in rows:
            _validate_small_payload(row, label="gathered")
        return tuple(rows)

    def broadcast_small(
        self, value: Mapping[str, Any] | None, *, src: int
    ) -> Mapping[str, Any]:
        if src != 0:
            raise World8ShadowError("v1 only permits global rank 0 broadcasts")
        if self.rank == src:
            assert value is not None
            _validate_small_payload(value, label="broadcast")
        box: list[Any] = [dict(value) if self.rank == src else None]
        self._dist.broadcast_object_list(box, src=src)
        _validate_small_payload(box[0], label="broadcast result")
        return box[0]


def _validate_collective(collective: World8SmallCollective) -> None:
    if collective.world_size != WORLD_SIZE or collective.rank not in GLOBAL_RANKS:
        raise World8ShadowError("collective must expose exact global ranks 0..7")


def _gather_envelopes(
    collective: World8SmallCollective,
    *,
    phase: str,
    ok: bool,
    payload: Mapping[str, Any],
    error: BaseException | None,
) -> tuple[Mapping[str, Any], ...]:
    envelope = {
        "phase": _safe_id(phase, label="phase"),
        "global_rank": collective.rank,
        "ok": bool(ok),
        "payload": dict(payload),
        "error_type": type(error).__name__ if error is not None else None,
    }
    rows = tuple(collective.all_gather_small(envelope))
    if len(rows) != WORLD_SIZE:
        raise World8ShadowError(f"{phase} did not gather eight envelopes")
    for expected_rank, row in enumerate(rows):
        if (
            set(row) != {"phase", "global_rank", "ok", "payload", "error_type"}
            or row["phase"] != phase
            or row["global_rank"] != expected_rank
            or type(row["ok"]) is not bool
            or not isinstance(row["payload"], Mapping)
        ):
            raise World8ShadowError(f"{phase} rank/order envelope differs")
    return rows


def _write_rank_evidence_packet(
    *,
    transaction_directory: Path,
    rank_evidence: mosaic_qp.SPRankEvidence,
    expected_rank: int,
) -> Mapping[str, Any]:
    if not isinstance(rank_evidence, mosaic_qp.SPRankEvidence):
        raise World8ShadowError("local rank evidence type differs")
    actor = _actor_for_rank(expected_rank)
    if rank_evidence.global_rank != expected_rank:
        raise World8ShadowError("rank evidence global rank differs")
    if type(rank_evidence.action_rows) is not tuple or type(
        rank_evidence.preservation_rows
    ) is not tuple:
        raise World8ShadowError("rank evidence row collections must be tuples")
    if not rank_evidence.action_rows:
        raise World8ShadowError("rank evidence has no action row")
    if (
        len(rank_evidence.action_rows) + len(rank_evidence.preservation_rows)
        > MAX_EVIDENCE_ROWS_PER_RANK
    ):
        raise World8ShadowError("rank evidence exceeds the closed row-count policy")
    binary = bytearray()
    rows: list[dict[str, Any]] = []

    def append_tensor(value: torch.Tensor) -> tuple[int, int, str]:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.ndim != 1
            or value.numel() != mosaic_qp.CANONICAL_PARAMETER_COUNT
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise World8ShadowError("Jacobian row tensor coordinate differs")
        raw = _raw_tensor_bytes(value)
        start = len(binary)
        binary.extend(raw)
        return start, len(raw), tensor_sha256(value)

    for row in rank_evidence.action_rows:
        if not isinstance(row, mosaic_qp.ActionConstraintRow) or row.actor_family != actor:
            raise World8ShadowError("action row belongs to the wrong DP arm")
        start, length, digest = append_tensor(row.values)
        rows.append(
            {
                "kind": "action",
                "row_id": _safe_id(row.row_id, label="action row ID"),
                "actor_family": actor,
                "minimum_dot": float(row.minimum_dot),
                "layout_digest": _sha256(row.layout_digest, label="row layout"),
                "checkpoint_content_receipt_digest": _sha256(
                    row.checkpoint_content_receipt_digest,
                    label="row checkpoint receipt",
                ),
                "parameter_state_sha256": _sha256(
                    row.parameter_state_sha256, label="row parameter state"
                ),
                "gradient_computation_receipt_digest": _sha256(
                    row.gradient_computation_receipt_digest,
                    label="action gradient receipt",
                ),
                "offset_bytes": start,
                "length_bytes": length,
                "tensor_sha256": digest,
            }
        )
    for row in rank_evidence.preservation_rows:
        if (
            not isinstance(row, mosaic_qp.PreservationConstraintRow)
            or row.family not in PRESERVATION_FAMILIES
        ):
            raise World8ShadowError("preservation row family/type differs")
        start, length, digest = append_tensor(row.values)
        rows.append(
            {
                "kind": "preservation",
                "row_id": _safe_id(row.row_id, label="preservation row ID"),
                "family": row.family,
                "maximum_absolute_dot": float(row.maximum_absolute_dot),
                "layout_digest": _sha256(row.layout_digest, label="row layout"),
                "checkpoint_content_receipt_digest": _sha256(
                    row.checkpoint_content_receipt_digest,
                    label="row checkpoint receipt",
                ),
                "parameter_state_sha256": _sha256(
                    row.parameter_state_sha256, label="row parameter state"
                ),
                "gradient_computation_receipt_digest": _sha256(
                    row.gradient_computation_receipt_digest,
                    label="preservation gradient receipt",
                ),
                "offset_bytes": start,
                "length_bytes": length,
                "tensor_sha256": digest,
            }
        )
    artifact_path = transaction_directory / f"rank-{expected_rank}.evidence.fp32.bin"
    if len(binary) > MAX_EVIDENCE_ARTIFACT_BYTES:
        raise World8ShadowError("rank evidence artifact exceeds size policy")
    artifact_sha = _create_fsynced_file(artifact_path, bytes(binary))
    manifest = {
        "schema_version": EVIDENCE_PACKET_SCHEMA,
        "global_rank": expected_rank,
        "actor_family": actor,
        "rank_evidence_receipt_digest": _sha256(
            rank_evidence.rank_evidence_receipt_digest,
            label="rank evidence receipt",
        ),
        "artifact_file": artifact_path.name,
        "artifact_sha256": artifact_sha,
        "artifact_size": len(binary),
        "row_count": len(rows),
        "rows": rows,
    }
    manifest_path = transaction_directory / f"rank-{expected_rank}.evidence.json"
    manifest_sha = _create_fsynced_file(manifest_path, canonical_json_bytes(manifest))
    return {
        "global_rank": expected_rank,
        "actor_family": actor,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "artifact_sha256": artifact_sha,
        "rank_evidence_receipt_digest": manifest[
            "rank_evidence_receipt_digest"
        ],
    }


def _tensor_from_raw_fp32(raw: bytes, *, expected_numel: int) -> torch.Tensor:
    if len(raw) != expected_numel * 4:
        raise World8ShadowError("raw FP32 tensor length differs")
    values = array("f")
    values.frombytes(raw)
    if values.itemsize != 4:
        raise World8ShadowError("host float representation is not 32-bit")
    if sys.byteorder != "little":
        values.byteswap()
    return torch.frombuffer(values, dtype=torch.float32).clone().contiguous()


def _load_rank_evidence_packet(
    transaction_directory: Path,
    descriptor: Mapping[str, Any],
) -> mosaic_qp.SPRankEvidence:
    if set(descriptor) != {
        "global_rank",
        "actor_family",
        "manifest_file",
        "manifest_sha256",
        "artifact_sha256",
        "rank_evidence_receipt_digest",
    }:
        raise World8ShadowError("rank evidence descriptor key closure differs")
    rank = descriptor["global_rank"]
    actor = _actor_for_rank(rank)
    if descriptor["actor_family"] != actor:
        raise World8ShadowError("rank evidence descriptor arm differs")
    manifest_name = descriptor["manifest_file"]
    if manifest_name != f"rank-{rank}.evidence.json":
        raise World8ShadowError("rank evidence manifest filename differs")
    manifest, _ = _read_canonical_json(
        transaction_directory / manifest_name,
        expected_sha256=_sha256(
            descriptor["manifest_sha256"], label="evidence manifest SHA"
        ),
    )
    expected_manifest_keys = {
        "schema_version",
        "global_rank",
        "actor_family",
        "rank_evidence_receipt_digest",
        "artifact_file",
        "artifact_sha256",
        "artifact_size",
        "row_count",
        "rows",
    }
    if set(manifest) != expected_manifest_keys or (
        manifest["schema_version"] != EVIDENCE_PACKET_SCHEMA
        or manifest["global_rank"] != rank
        or manifest["actor_family"] != actor
        or manifest["rank_evidence_receipt_digest"]
        != descriptor["rank_evidence_receipt_digest"]
        or manifest["artifact_sha256"] != descriptor["artifact_sha256"]
        or manifest["artifact_file"] != f"rank-{rank}.evidence.fp32.bin"
        or manifest["row_count"] != len(manifest["rows"])
    ):
        raise World8ShadowError("rank evidence manifest closure differs")
    if (
        type(manifest["row_count"]) is not int
        or not 1 <= manifest["row_count"] <= MAX_EVIDENCE_ROWS_PER_RANK
        or type(manifest["artifact_size"]) is not int
        or not 0 <= manifest["artifact_size"] <= MAX_EVIDENCE_ARTIFACT_BYTES
    ):
        raise World8ShadowError("rank evidence count/size policy differs")
    artifact = _read_stable_file(
        transaction_directory / manifest["artifact_file"],
        expected_sha256=_sha256(manifest["artifact_sha256"], label="evidence artifact SHA"),
        maximum_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
        require_read_only=True,
    )
    if len(artifact) != manifest["artifact_size"]:
        raise World8ShadowError("rank evidence artifact size differs")
    action_rows = []
    preservation_rows = []
    cursor = 0
    for row in manifest["rows"]:
        if not isinstance(row, Mapping):
            raise World8ShadowError("rank evidence row metadata differs")
        start = row.get("offset_bytes")
        length = row.get("length_bytes")
        if type(start) is not int or type(length) is not int or start != cursor:
            raise World8ShadowError("rank evidence row offsets are not contiguous")
        value = _tensor_from_raw_fp32(
            artifact[start : start + length],
            expected_numel=mosaic_qp.CANONICAL_PARAMETER_COUNT,
        )
        cursor += length
        if tensor_sha256(value) != row.get("tensor_sha256"):
            raise World8ShadowError("rank evidence tensor SHA differs")
        if row.get("kind") == "action":
            action_rows.append(
                mosaic_qp.ActionConstraintRow(
                    row_id=row["row_id"],
                    actor_family=row["actor_family"],
                    values=value,
                    minimum_dot=row["minimum_dot"],
                    layout_digest=row["layout_digest"],
                    checkpoint_content_receipt_digest=row[
                        "checkpoint_content_receipt_digest"
                    ],
                    parameter_state_sha256=row["parameter_state_sha256"],
                    gradient_computation_receipt_digest=row[
                        "gradient_computation_receipt_digest"
                    ],
                )
            )
        elif row.get("kind") == "preservation":
            preservation_rows.append(
                mosaic_qp.PreservationConstraintRow(
                    row_id=row["row_id"],
                    family=row["family"],
                    values=value,
                    maximum_absolute_dot=row["maximum_absolute_dot"],
                    layout_digest=row["layout_digest"],
                    checkpoint_content_receipt_digest=row[
                        "checkpoint_content_receipt_digest"
                    ],
                    parameter_state_sha256=row["parameter_state_sha256"],
                    gradient_computation_receipt_digest=row[
                        "gradient_computation_receipt_digest"
                    ],
                )
            )
        else:
            raise World8ShadowError("rank evidence row kind differs")
    if cursor != len(artifact):
        raise World8ShadowError("rank evidence artifact has trailing bytes")
    return mosaic_qp.SPRankEvidence(
        global_rank=rank,
        action_rows=tuple(action_rows),
        preservation_rows=tuple(preservation_rows),
        rank_evidence_receipt_digest=manifest[
            "rank_evidence_receipt_digest"
        ],
    )


def _build_union(
    transaction_directory: Path,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    topology_receipt_digest: str,
) -> mosaic_qp.DP2SP4Evidence:
    if len(descriptors) != WORLD_SIZE:
        raise World8ShadowError("WORLD8 evidence descriptor closure differs")
    ranks = tuple(
        _load_rank_evidence_packet(transaction_directory, row)
        for row in descriptors
    )
    if tuple(row.global_rank for row in ranks) != GLOBAL_RANKS:
        raise World8ShadowError("WORLD8 evidence global-rank ordering differs")
    return mosaic_qp.DP2SP4Evidence(
        dp_arms=(
            mosaic_qp.DPArmEvidence("dog", ranks[:4]),
            mosaic_qp.DPArmEvidence("human", ranks[4:]),
        ),
        topology_receipt_digest=_sha256(
            topology_receipt_digest, label="topology receipt"
        ),
    )


def _verify_scientific_gate_files(
    *,
    gate_files: ScientificGateFiles,
    evidence: mosaic_qp.DP2SP4Evidence,
    checkpoint_manifest_sha256: str,
    checkpoint_content_receipt_digest: str,
    fresh_inputs: FreshInputFiles,
    validated_contract: _ValidatedQPContract,
    pinned_verifier_policy: ExternalVerifierPolicy,
) -> Mapping[str, Any]:
    if not isinstance(gate_files, ScientificGateFiles):
        raise World8ShadowError("scientific gate-file bundle type differs")
    if tuple(arm.actor_family for arm in gate_files.arms) != ARM_ORDER:
        raise World8ShadowError("scientific gate arms must be dog then human")
    if gate_files.verifier_policy != pinned_verifier_policy:
        raise World8ShadowError(
            "scientific verifier policy differs from transaction-pinned policy"
        )
    _hash_stable_file(
        Path(fresh_inputs.source_manifest.path),
        expected_sha256=_sha256(
            fresh_inputs.source_manifest.sha256, label="source manifest SHA"
        ),
        maximum_bytes=MAX_SOURCE_MANIFEST_BYTES,
        require_read_only=True,
    )
    _hash_stable_file(
        Path(fresh_inputs.official_gaussian.path),
        expected_sha256=_sha256(
            fresh_inputs.official_gaussian.sha256, label="official Gaussian SHA"
        ),
        maximum_bytes=MAX_GAUSSIAN_BYTES,
        require_read_only=True,
    )
    verifier = CanonicalExternalVerifier(pinned_verifier_policy)
    qp_contract = verifier.verify(
        gate_files.qp_contract,
        audit_type="qp_contract",
        exact_bindings=validated_contract.bindings,
    )
    arm_reports = []
    for gate_arm, evidence_arm in zip(gate_files.arms, evidence.dp_arms):
        actor = gate_arm.actor_family
        canonical_rank = evidence_arm.sp_ranks[0]
        owner_bindings = {
            "actor_family": actor,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_content_receipt_digest": checkpoint_content_receipt_digest,
        }
        owner = verifier.verify(
            gate_arm.owner, audit_type="owner_audit", exact_bindings=owner_bindings
        )
        if type(gate_arm.query_seed_gates) is not tuple or len(
            gate_arm.query_seed_gates
        ) != 2:
            raise World8ShadowError(f"{actor} must provide exactly two query gates")
        seed_reports = []
        seeds = []
        for seed_gate in gate_arm.query_seed_gates:
            if type(seed_gate.query_seed) is not int or not (
                0 <= seed_gate.query_seed < 2**63
            ):
                raise World8ShadowError("query seed must be a nonnegative int64")
            seeds.append(seed_gate.query_seed)
            common = {
                "actor_family": actor,
                "query_seed": seed_gate.query_seed,
                "owner_receipt_sha256": owner["receipt_sha256"],
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            }
            specificity = verifier.verify(
                seed_gate.specificity,
                audit_type="two_seed_specificity",
                exact_bindings=common,
            )
            direction = verifier.verify(
                seed_gate.plus_q_minus_q,
                audit_type="plus_q_minus_q_direction",
                exact_bindings={
                    **common,
                    "specificity_receipt_sha256": specificity[
                        "receipt_sha256"
                    ],
                },
            )
            seed_reports.append(
                {
                    "query_seed": seed_gate.query_seed,
                    "specificity_receipt_sha256": specificity[
                        "receipt_sha256"
                    ],
                    "direction_receipt_sha256": direction["receipt_sha256"],
                }
            )
        if seeds != sorted(seeds) or len(set(seeds)) != 2:
            raise World8ShadowError("query seeds must be unique and sorted")
        query_gate_hashes = [
            [row["specificity_receipt_sha256"], row["direction_receipt_sha256"]]
            for row in seed_reports
        ]
        action_refs = gate_arm.action_gradient_receipts
        if len(action_refs) != len(canonical_rank.action_rows):
            raise World8ShadowError("action-gradient gate closure differs")
        action_reports = []
        for row, ref in zip(canonical_rank.action_rows, action_refs):
            report = verifier.verify(
                ref,
                audit_type="action_gradient",
                exact_bindings={
                    "actor_family": actor,
                    "row_id": row.row_id,
                    "row_sha256": tensor_sha256(row.values),
                    "gradient_computation_receipt_digest": (
                        row.gradient_computation_receipt_digest
                    ),
                    "minimum_dot": float(row.minimum_dot),
                    "layout_digest": row.layout_digest,
                    "checkpoint_content_receipt_digest": (
                        row.checkpoint_content_receipt_digest
                    ),
                    "parameter_state_sha256": row.parameter_state_sha256,
                    "owner_receipt_sha256": owner["receipt_sha256"],
                    "query_gate_receipt_sha256s": query_gate_hashes,
                    "qp_contract_receipt_sha256": qp_contract[
                        "receipt_sha256"
                    ],
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                },
            )
            action_reports.append(report["receipt_sha256"])
        preservation_rows = tuple(canonical_rank.preservation_rows)
        if len(gate_arm.preservation_gradient_receipts) != len(
            preservation_rows
        ):
            raise World8ShadowError("preservation-gradient gate closure differs")
        preservation_reports = []
        for row, ref in zip(
            preservation_rows, gate_arm.preservation_gradient_receipts
        ):
            report = verifier.verify(
                ref,
                audit_type="preservation_gradient",
                exact_bindings={
                    "actor_family": actor,
                    "family": row.family,
                    "row_id": row.row_id,
                    "row_sha256": tensor_sha256(row.values),
                    "gradient_computation_receipt_digest": (
                        row.gradient_computation_receipt_digest
                    ),
                    "maximum_absolute_dot": float(row.maximum_absolute_dot),
                    "layout_digest": row.layout_digest,
                    "checkpoint_content_receipt_digest": (
                        row.checkpoint_content_receipt_digest
                    ),
                    "parameter_state_sha256": row.parameter_state_sha256,
                    "qp_contract_receipt_sha256": qp_contract[
                        "receipt_sha256"
                    ],
                    "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
                },
            )
            preservation_reports.append(
                {
                    "family": row.family,
                    "row_id": row.row_id,
                    "receipt_sha256": report["receipt_sha256"],
                }
            )
        family_counts = {family: 0 for family in PRESERVATION_FAMILIES}
        for row in preservation_reports:
            family_counts[row["family"]] += 1
        if any(count == 0 for count in family_counts.values()):
            raise World8ShadowError("all six preservation families require external receipts")
        arm_reports.append(
            {
                "actor_family": actor,
                "owner_receipt_sha256": owner["receipt_sha256"],
                "query_seed_gates": seed_reports,
                "action_gradient_receipt_sha256s": action_reports,
                "preservation_gradient_receipts": preservation_reports,
            }
        )
    fresh_plan = verifier.verify(
        gate_files.fresh_plan,
        audit_type="fresh_exact81_plan",
        exact_bindings={
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_content_receipt_digest": (
                checkpoint_content_receipt_digest
            ),
            "qp_contract_receipt_sha256": qp_contract["receipt_sha256"],
            "source_manifest_sha256": fresh_inputs.source_manifest.sha256,
            "official_gaussian_sha256": fresh_inputs.official_gaussian.sha256,
            "exact_frame_count": 81,
            "actor_families": list(ARM_ORDER),
            "preservation_families": list(PRESERVATION_FAMILIES),
            "fresh_rollout_required": True,
            "source_disjoint_confirmation_required": True,
            "rollback_on_any_failure": True,
            "post_hoc_selection_allowed": False,
        },
    )
    report = {
        "external_verifier_id": gate_files.verifier_policy.verifier_id,
        "external_verifier_executable_sha256": (
            gate_files.verifier_policy.verifier_executable_sha256
        ),
        "external_verifier_ed25519_public_key_hex": (
            gate_files.verifier_policy.verifier_ed25519_public_key_hex
        ),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "qp_contract_receipt_sha256": qp_contract["receipt_sha256"],
        "qp_contract_artifact_sha256": qp_contract["artifact_sha256"],
        "qp_contract_bindings_digest": object_sha256(
            validated_contract.bindings
        ),
        "arms": arm_reports,
        "fresh_plan_receipt_sha256": fresh_plan["receipt_sha256"],
        "fresh_plan_artifact_sha256": fresh_plan["artifact_sha256"],
        "source_manifest_sha256": fresh_inputs.source_manifest.sha256,
        "official_gaussian_sha256": fresh_inputs.official_gaussian.sha256,
        "self_sealed_pass_booleans_consumed": False,
    }
    return {**report, "verification_digest": object_sha256(report)}


def compute_anticipated_realized(
    base: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact CPU/FP32 ``fl(base + delta)`` and its realized displacement."""

    if (
        not isinstance(base, torch.Tensor)
        or not isinstance(delta, torch.Tensor)
        or base.dtype != torch.float32
        or delta.dtype != torch.float32
        or tuple(base.shape) != tuple(delta.shape)
        or not bool(torch.isfinite(base).all().item())
        or not bool(torch.isfinite(delta).all().item())
    ):
        raise World8ShadowError("anticipated-realized inputs must be matching finite FP32")
    base_cpu = base.detach().to(device="cpu").contiguous()
    delta_cpu = delta.detach().to(device="cpu").contiguous()
    anticipated = torch.add(base_cpu, delta_cpu).to(dtype=torch.float32).contiguous()
    realized = torch.sub(anticipated, base_cpu).to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(anticipated).all().item()) or not bool(
        torch.isfinite(realized).all().item()
    ):
        raise World8ShadowError("anticipated FP32 add produced non-finite data")
    return anticipated, realized


def _anticipated_state(
    *,
    snapshot: _ABSnapshot,
    delta_by_parameter: Mapping[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], str, str]:
    if tuple(delta_by_parameter) != mosaic_qp.CANONICAL_PARAMETER_NAMES:
        raise World8ShadowError("candidate delta parameter closure differs")
    final_b = []
    realized = []
    for base, name in zip(snapshot.b_tensors, mosaic_qp.CANONICAL_PARAMETER_NAMES):
        after, displacement = compute_anticipated_realized(
            base, delta_by_parameter[name]
        )
        final_b.append(after)
        realized.append(displacement)
    entries = []
    for (a_name, b_name), a_tensor, b_tensor in zip(
        CANONICAL_AB_NAMES, snapshot.a_tensors, final_b
    ):
        entries.extend(
            (
                {
                    "name": a_name,
                    "role": "fixed_lora_a",
                    "shape": list(a_tensor.shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": tensor_sha256(a_tensor),
                },
                {
                    "name": b_name,
                    "role": "action_lora_b",
                    "shape": list(b_tensor.shape),
                    "dtype": "torch.float32",
                    "tensor_sha256": tensor_sha256(b_tensor),
                },
            )
        )
    realized_flat = torch.cat([row.reshape(-1) for row in realized]).contiguous()
    return (
        tuple(final_b),
        tuple(realized),
        object_sha256(entries),
        tensor_sha256(realized_flat),
    )


def _audit_anticipated_constraints(
    *,
    solution: mosaic_qp.JacobianQPSolution,
    realized: Sequence[torch.Tensor],
) -> Mapping[str, Any]:
    flat = torch.cat([row.reshape(-1) for row in realized]).contiguous()
    # The stateless core does not yet expose its audit-only primitive.  This
    # call is deliberately limited to that pure, non-mutating checker.
    union = mosaic_qp._validate_and_union_dp2_sp4_evidence(  # type: ignore[attr-defined]
        layout=solution.layout,
        evidence=solution.evidence,
        minimum_row_norm=solution.config.minimum_row_norm,
    )
    (
        action_rows,
        preservation_rows,
        layer_rows,
        global_row,
        failures,
    ) = mosaic_qp._constraint_audit(  # type: ignore[attr-defined]
        flat=flat,
        action_rows=union.action_rows,
        preservation_rows=union.preservation_rows,
        layout=solution.layout,
        layer_trust_radii=solution.layer_trust_radii,
        global_trust_radius=solution.global_trust_radius,
        active_tolerance=solution.config.active_constraint_tolerance,
    )
    report = {
        "anticipated_realized_sha256": tensor_sha256(flat),
        "action_rows": action_rows,
        "preservation_rows": preservation_rows,
        "per_layer_trust_radii": layer_rows,
        "global_trust_radius": global_row,
        "failure_codes": failures,
        "all_constraints_passed": not failures,
        "audit_uses_fl_b_plus_delta_minus_b": True,
    }
    if failures:
        raise World8ShadowError(
            "anticipated FP32 realized displacement violates QP constraints"
        )
    return {**report, "audit_digest": object_sha256(report)}


def _delta_artifact(
    delta_by_parameter: Mapping[str, torch.Tensor]
) -> tuple[bytes, list[Mapping[str, Any]], str]:
    raw = bytearray()
    entries = []
    for name in mosaic_qp.CANONICAL_PARAMETER_NAMES:
        tensor = delta_by_parameter[name].detach().to(device="cpu").contiguous()
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != tuple(
            mosaic_qp.CANONICAL_B_SHAPE
        ):
            raise World8ShadowError(f"candidate delta {name} geometry differs")
        chunk = _raw_tensor_bytes(tensor)
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "torch.float32",
                "offset_bytes": len(raw),
                "length_bytes": len(chunk),
                "tensor_sha256": tensor_sha256(tensor),
            }
        )
        raw.extend(chunk)
    flat = torch.cat(
        [delta_by_parameter[name].detach().cpu().reshape(-1) for name in mosaic_qp.CANONICAL_PARAMETER_NAMES]
    ).contiguous()
    return bytes(raw), entries, tensor_sha256(flat)


def _load_delta_artifact(
    transaction_directory: Path, descriptor: Mapping[str, Any]
) -> Mapping[str, torch.Tensor]:
    expected_keys = {
        "schema_version",
        "candidate_receipt_file",
        "candidate_receipt_sha256",
        "candidate_receipt_digest",
        "candidate_delta_file",
        "candidate_delta_artifact_sha256",
        "candidate_delta_sha256",
        "candidate_delta_entries",
        "checkpoint_manifest_sha256",
        "base_ab_state_digest",
        "scientific_gate_verification_digest",
    }
    if set(descriptor) != expected_keys or descriptor[
        "schema_version"
    ] != CANDIDATE_DESCRIPTOR_SCHEMA:
        raise World8ShadowError("candidate descriptor closure differs")
    if (
        descriptor["candidate_receipt_file"] != "qp-candidate.receipt.json"
        or descriptor["candidate_delta_file"] != "candidate-delta.fp32.bin"
    ):
        raise World8ShadowError("candidate descriptor filename closure differs")
    receipt, _ = _read_canonical_json(
        transaction_directory / descriptor["candidate_receipt_file"],
        expected_sha256=_sha256(
            descriptor["candidate_receipt_sha256"], label="candidate receipt SHA"
        ),
    )
    mosaic_qp.validate_candidate_receipt_schema(receipt)
    if (
        receipt["receipt_digest"] != descriptor["candidate_receipt_digest"]
        or receipt["actual_fp32_candidate_delta_sha256"]
        != descriptor["candidate_delta_sha256"]
        or receipt["world8_apply_authorized"] is not False
        or receipt["runtime_apply_authorized"] is not False
        or receipt["training_executed"] is not False
        or receipt["mathematical_candidate_authorized"] is not True
    ):
        raise World8ShadowError("candidate receipt policy/binding differs")
    raw = _read_stable_file(
        transaction_directory / descriptor["candidate_delta_file"],
        expected_sha256=_sha256(
            descriptor["candidate_delta_artifact_sha256"],
            label="candidate delta artifact SHA",
        ),
        maximum_bytes=4 * mosaic_qp.CANONICAL_PARAMETER_COUNT,
        require_read_only=True,
    )
    deltas: dict[str, torch.Tensor] = {}
    cursor = 0
    entries = descriptor["candidate_delta_entries"]
    if not isinstance(entries, list) or len(entries) != 32:
        raise World8ShadowError("candidate delta entry closure differs")
    for expected_name, entry in zip(mosaic_qp.CANONICAL_PARAMETER_NAMES, entries):
        if entry["name"] != expected_name or entry["offset_bytes"] != cursor:
            raise World8ShadowError("candidate delta ordering/offset differs")
        length = entry["length_bytes"]
        tensor = _tensor_from_raw_fp32(
            raw[cursor : cursor + length],
            expected_numel=mosaic_qp.HIDDEN_SIZE * mosaic_qp.LORA_RANK,
        ).reshape(mosaic_qp.CANONICAL_B_SHAPE)
        cursor += length
        if tensor_sha256(tensor) != entry["tensor_sha256"]:
            raise World8ShadowError("candidate delta tensor SHA differs")
        deltas[expected_name] = tensor
    if cursor != len(raw):
        raise World8ShadowError("candidate delta artifact has trailing bytes")
    flat = torch.cat([deltas[name].reshape(-1) for name in mosaic_qp.CANONICAL_PARAMETER_NAMES])
    if tensor_sha256(flat) != descriptor["candidate_delta_sha256"]:
        raise World8ShadowError("candidate aggregate delta SHA differs")
    return deltas


def _ab_state_artifact(
    *, snapshot: _ABSnapshot, final_b: Sequence[torch.Tensor]
) -> tuple[bytes, list[Mapping[str, Any]]]:
    raw = bytearray()
    entries = []
    for (a_name, b_name), a_tensor, b_tensor in zip(
        CANONICAL_AB_NAMES, snapshot.a_tensors, final_b
    ):
        for role, name, tensor in (
            ("fixed_lora_a", a_name, a_tensor),
            ("action_lora_b", b_name, b_tensor),
        ):
            chunk = _raw_tensor_bytes(tensor)
            entries.append(
                {
                    "name": name,
                    "role": role,
                    "shape": list(tensor.shape),
                    "dtype": "torch.float32",
                    "offset_bytes": len(raw),
                    "length_bytes": len(chunk),
                    "tensor_sha256": tensor_sha256(tensor),
                }
            )
            raw.extend(chunk)
    return bytes(raw), entries


def _verify_ab_artifact(
    *,
    transaction_directory: Path,
    artifact_file: str,
    artifact_sha256: str,
    entries: Sequence[Mapping[str, Any]],
    expected_state_digest: str,
) -> None:
    if artifact_file not in {
        "base-ab-state.fp32.bin",
        "shadow-ab-state.fp32.bin",
    }:
        raise World8ShadowError("A/B artifact filename escapes the closed protocol")
    raw = _read_stable_file(
        transaction_directory / artifact_file,
        expected_sha256=_sha256(artifact_sha256, label="A/B artifact SHA"),
        maximum_bytes=MAX_AB_ARTIFACT_BYTES,
        require_read_only=True,
    )
    if len(entries) != 64:
        raise World8ShadowError("A/B artifact entry closure differs")
    cursor = 0
    state_entries = []
    expected_rows = tuple(
        row
        for a_name, b_name in CANONICAL_AB_NAMES
        for row in (
            (a_name, "fixed_lora_a", mosaic_qp.CANONICAL_A_SHAPE),
            (b_name, "action_lora_b", mosaic_qp.CANONICAL_B_SHAPE),
        )
    )
    for entry, (expected_name, expected_role, expected_shape) in zip(
        entries, expected_rows
    ):
        if (
            entry.get("name") != expected_name
            or entry.get("role") != expected_role
            or entry.get("shape") != list(expected_shape)
            or entry.get("dtype") != "torch.float32"
            or entry.get("offset_bytes") != cursor
            or type(entry.get("length_bytes")) is not int
        ):
            raise World8ShadowError("A/B artifact entry geometry/order differs")
        length = entry["length_bytes"]
        tensor = _tensor_from_raw_fp32(
            raw[cursor : cursor + length],
            expected_numel=int(math.prod(expected_shape)),
        ).reshape(expected_shape)
        cursor += length
        digest = tensor_sha256(tensor)
        if digest != entry.get("tensor_sha256"):
            raise World8ShadowError("A/B artifact tensor digest differs")
        state_entries.append(
            {
                "name": expected_name,
                "role": expected_role,
                "shape": list(expected_shape),
                "dtype": "torch.float32",
                "tensor_sha256": digest,
            }
        )
    if cursor != len(raw) or object_sha256(state_entries) != _sha256(
        expected_state_digest, label="A/B state digest"
    ):
        raise World8ShadowError("A/B artifact final digest/trailing bytes differ")


def create_transaction_directory(
    path: Path,
    *,
    transaction_id: str,
    verifier_policy: ExternalVerifierPolicy,
) -> CanonicalFileRef:
    transaction_id = _safe_id(transaction_id, label="transaction ID")
    CanonicalExternalVerifier(verifier_policy)
    path = Path(path)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise World8ShadowError("transaction parent must be one real existing directory")
    try:
        os.mkdir(path, 0o700)
    except FileExistsError as error:
        raise World8ShadowError(
            "transaction directory must be newly and exclusively created"
        ) from error
    if path.is_symlink() or not path.is_dir():
        raise World8ShadowError("transaction path must be one real directory")
    directory_stat = os.stat(path, follow_symlinks=False)
    if directory_stat.st_uid != os.getuid() or directory_stat.st_mode & 0o077:
        raise World8ShadowError("transaction directory owner/mode policy differs")
    init = {
        "schema_version": TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "method_name": METHOD_NAME,
        "world_size": WORLD_SIZE,
        "global_ranks": list(GLOBAL_RANKS),
        "authoritative_parameter_mutation_allowed": False,
        "protocol_mode": "quarantined_shadow_prepare_only",
        "first_step_exact_zero_b_required": True,
        "external_verifier_id": verifier_policy.verifier_id,
        "external_verifier_executable_sha256": (
            verifier_policy.verifier_executable_sha256
        ),
        "external_verifier_ed25519_public_key_hex": (
            verifier_policy.verifier_ed25519_public_key_hex
        ),
        "transaction_directory_device": int(directory_stat.st_dev),
        "transaction_directory_inode": int(directory_stat.st_ino),
    }
    init_path = path / "TRANSACTION.json"
    digest = _create_fsynced_file(init_path, canonical_json_bytes(init))
    _fsync_directory(path.parent)
    return CanonicalFileRef(init_path, digest)


def _verify_transaction_directory(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise World8ShadowError("transaction directory identity differs")
    directory_stat = os.stat(path, follow_symlinks=False)
    if directory_stat.st_uid != os.getuid() or directory_stat.st_mode & 0o077:
        raise World8ShadowError("transaction directory owner/mode policy differs")
    raw = _read_stable_file(path / "TRANSACTION.json", require_read_only=True)
    digest = hashlib.sha256(raw).hexdigest()
    init, _ = _read_canonical_json(
        path / "TRANSACTION.json", expected_sha256=digest
    )
    if set(init) != {
        "schema_version",
        "transaction_id",
        "method_name",
        "world_size",
        "global_ranks",
        "authoritative_parameter_mutation_allowed",
        "protocol_mode",
        "first_step_exact_zero_b_required",
        "external_verifier_id",
        "external_verifier_executable_sha256",
        "external_verifier_ed25519_public_key_hex",
        "transaction_directory_device",
        "transaction_directory_inode",
    } or (
        init["schema_version"] != TRANSACTION_SCHEMA
        or init["method_name"] != METHOD_NAME
        or init["world_size"] != WORLD_SIZE
        or init["global_ranks"] != list(GLOBAL_RANKS)
        or init["authoritative_parameter_mutation_allowed"] is not False
        or init["protocol_mode"] != "quarantined_shadow_prepare_only"
        or init["first_step_exact_zero_b_required"] is not True
        or init["transaction_directory_device"] != int(directory_stat.st_dev)
        or init["transaction_directory_inode"] != int(directory_stat.st_ino)
    ):
        raise World8ShadowError("transaction init contract differs")
    _safe_id(init["transaction_id"], label="transaction ID")
    CanonicalExternalVerifier(
        ExternalVerifierPolicy(
            verifier_id=init["external_verifier_id"],
            verifier_executable_sha256=init[
                "external_verifier_executable_sha256"
            ],
            verifier_ed25519_public_key_hex=init[
                "external_verifier_ed25519_public_key_hex"
            ],
        )
    )
    return {**dict(init), "transaction_init_sha256": digest}


def _write_failure_envelope(
    *,
    transaction_directory: Path,
    rank: int,
    phase: str,
    error: BaseException,
    snapshot: _ABSnapshot | None,
    authoritative_unchanged: bool,
) -> None:
    payload = {
        "schema_version": "bernini-world8-local-fail-stop-v1",
        "global_rank": rank,
        "phase": phase,
        "exception_type": type(error).__name__,
        "base_ab_state_digest": snapshot.state_digest if snapshot else None,
        "authoritative_ab_byte_identical": authoritative_unchanged,
        "parameter_add_attempted": False,
        "fail_stop": True,
    }
    try:
        _create_fsynced_file(
            transaction_directory / f"rank-{rank}.FAIL.json",
            canonical_json_bytes(payload),
        )
    except BaseException:
        pass


@dataclass(frozen=True)
class ShadowPrepareResult:
    prepared: bool
    aborted: bool
    transaction_directory: Path
    candidate_descriptor: CanonicalFileRef | None
    shadow_manifest: CanonicalFileRef | None
    fresh_render_request: CanonicalFileRef | None
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ShadowFinalizeResult:
    durable_commit_recorded: bool
    aborted: bool
    authoritative_parameters_mutated: bool
    publishable_shadow_authorized: bool
    commit_record: CanonicalFileRef | None
    receipt: Mapping[str, Any]


def _abort_payload(
    *, transaction: Mapping[str, Any], phase: str, failure_types: Sequence[str]
) -> Mapping[str, Any]:
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": "ABORTED",
        "transaction_id": transaction["transaction_id"],
        "transaction_init_sha256": transaction["transaction_init_sha256"],
        "failure_phase": phase,
        "failure_types": sorted(set(str(row) for row in failure_types)),
        "authoritative_parameters_mutated": False,
        "shadow_published": False,
        "commit_and_publication_supported": False,
    }


def _distributed_abort(
    collective: World8SmallCollective,
    *,
    transaction_directory: Path,
    transaction: Mapping[str, Any],
    phase: str,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    failure_types = [
        row.get("error_type") or "LOCAL_PHASE_REJECTED"
        for row in rows
        if row.get("ok") is not True
    ]
    control = None
    if collective.rank == 0:
        try:
            recovery = recover_world8_transaction(transaction_directory)
            if recovery["state"] == "ABORTED":
                aborted, aborted_sha = _read_named_record(
                    transaction_directory, "DECISION.json"
                )
                control = {
                    "ok": False,
                    "abort_recorded": True,
                    "aborted_file": "DECISION.json",
                    "aborted_sha256": aborted_sha,
                    "failure_phase": aborted["failure_phase"],
                    "idempotent_existing_decision": True,
                }
            else:
                payload = _abort_payload(
                    transaction=transaction, phase=phase, failure_types=failure_types
                )
                digest = _atomic_publish_record(
                    transaction_directory / "DECISION.json", payload
                )
                control = {
                    "ok": False,
                    "abort_recorded": True,
                    "aborted_file": "DECISION.json",
                    "aborted_sha256": digest,
                    "failure_phase": phase,
                    "idempotent_existing_decision": False,
                }
        except BaseException as error:
            control = {
                "ok": False,
                "abort_recorded": False,
                "failure_phase": phase,
                "error_type": type(error).__name__,
            }
    decision = collective.broadcast_small(control, src=0)
    if decision.get("abort_recorded") is not True:
        raise World8ShadowError(
            "distributed ABORT could not become the durable terminal decision"
        )
    return decision


def _root_materialize_shadow(
    *,
    transaction_directory: Path,
    snapshot: _ABSnapshot,
    evidence: mosaic_qp.DP2SP4Evidence,
    validated_contract: _ValidatedQPContract,
    gate_report: Mapping[str, Any],
    fresh_inputs: FreshInputFiles,
) -> Mapping[str, Any]:
    layout = validated_contract.layout
    solution = mosaic_qp.solve_stateless_jacobian_qp(
        layout=layout,
        evidence=evidence,
        global_trust_radius=validated_contract.global_trust_radius,
        layer_trust_radii=validated_contract.layer_trust_radii,
        config=validated_contract.config,
    )
    mosaic_qp.validate_candidate_receipt_schema(solution.receipt)
    # Strict recomputation is independent of the first returned object.  It
    # catches mutable evidence/config or a nondeterministic candidate before
    # any PREPARED WAL can exist.
    recomputed = mosaic_qp.solve_stateless_jacobian_qp(
        layout=layout,
        evidence=evidence,
        global_trust_radius=validated_contract.global_trust_radius,
        layer_trust_radii=validated_contract.layer_trust_radii,
        config=validated_contract.config,
    )
    if (
        not solution.authorized
        or solution.receipt["world8_apply_authorized"] is not False
        or solution.receipt["training_executed"] is not False
        or recomputed.receipt["receipt_digest"]
        != solution.receipt["receipt_digest"]
        or any(
            not torch.equal(
                solution.delta_by_parameter[name],
                recomputed.delta_by_parameter[name],
            )
            for name in solution.layout.names
        )
    ):
        raise World8ShadowError("strict QP recomputation/authorization failed")

    gate_path = transaction_directory / "scientific-gate-verification.json"
    gate_sha = _create_fsynced_file(gate_path, canonical_json_bytes(gate_report))
    candidate_receipt_path = transaction_directory / "qp-candidate.receipt.json"
    candidate_receipt_raw = canonical_json_bytes(solution.receipt)
    candidate_receipt_sha = _create_fsynced_file(
        candidate_receipt_path, candidate_receipt_raw
    )
    delta_raw, delta_entries, candidate_delta_sha = _delta_artifact(
        solution.delta_by_parameter
    )
    delta_path = transaction_directory / "candidate-delta.fp32.bin"
    delta_artifact_sha = _create_fsynced_file(delta_path, delta_raw)
    candidate_descriptor = {
        "schema_version": CANDIDATE_DESCRIPTOR_SCHEMA,
        "candidate_receipt_file": candidate_receipt_path.name,
        "candidate_receipt_sha256": candidate_receipt_sha,
        "candidate_receipt_digest": solution.receipt["receipt_digest"],
        "candidate_delta_file": delta_path.name,
        "candidate_delta_artifact_sha256": delta_artifact_sha,
        "candidate_delta_sha256": candidate_delta_sha,
        "candidate_delta_entries": delta_entries,
        "checkpoint_manifest_sha256": snapshot.manifest_sha256,
        "base_ab_state_digest": snapshot.state_digest,
        "scientific_gate_verification_digest": gate_report[
            "verification_digest"
        ],
    }
    candidate_path = transaction_directory / "candidate.json"
    candidate_sha = _create_fsynced_file(
        candidate_path, canonical_json_bytes(candidate_descriptor)
    )

    final_b, realized, final_ab_digest, realized_sha = _anticipated_state(
        snapshot=snapshot, delta_by_parameter=solution.delta_by_parameter
    )
    anticipated_audit = _audit_anticipated_constraints(
        solution=solution, realized=realized
    )
    if anticipated_audit["anticipated_realized_sha256"] != realized_sha:
        raise World8ShadowError("anticipated audit displacement binding differs")
    anticipated_path = transaction_directory / "anticipated-realized-audit.json"
    anticipated_sha = _create_fsynced_file(
        anticipated_path, canonical_json_bytes(anticipated_audit)
    )

    base_raw, base_entries = _ab_state_artifact(
        snapshot=snapshot, final_b=snapshot.b_tensors
    )
    base_path = transaction_directory / "base-ab-state.fp32.bin"
    base_artifact_sha = _create_fsynced_file(base_path, base_raw)
    shadow_raw, shadow_entries = _ab_state_artifact(
        snapshot=snapshot, final_b=final_b
    )
    shadow_path = transaction_directory / "shadow-ab-state.fp32.bin"
    shadow_artifact_sha = _create_fsynced_file(shadow_path, shadow_raw)
    shadow_manifest = {
        "schema_version": SHADOW_MANIFEST_SCHEMA,
        "checkpoint_manifest_sha256": snapshot.manifest_sha256,
        "checkpoint_content_receipt_digest": (
            snapshot.checkpoint_content_receipt_digest
        ),
        "base_ab_state_digest": snapshot.state_digest,
        "base_ab_artifact_file": base_path.name,
        "base_ab_artifact_sha256": base_artifact_sha,
        "base_ab_entries": base_entries,
        "candidate_descriptor_sha256": candidate_sha,
        "candidate_delta_sha256": candidate_delta_sha,
        "anticipated_realized_sha256": realized_sha,
        "anticipated_audit_file": anticipated_path.name,
        "anticipated_audit_sha256": anticipated_sha,
        "shadow_ab_state_digest": final_ab_digest,
        "shadow_ab_artifact_file": shadow_path.name,
        "shadow_ab_artifact_sha256": shadow_artifact_sha,
        "shadow_ab_entries": shadow_entries,
        "authoritative_parameters_mutated": False,
        "first_step_exact_zero_b": True,
        "quarantined_non_publishable": True,
    }
    shadow_manifest_path = transaction_directory / "shadow-manifest.json"
    shadow_manifest_sha = _create_fsynced_file(
        shadow_manifest_path, canonical_json_bytes(shadow_manifest)
    )
    fresh_request = {
        "schema_version": FRESH_REQUEST_SCHEMA,
        "checkpoint_manifest_sha256": snapshot.manifest_sha256,
        "base_ab_artifact_file": base_path.name,
        "base_ab_artifact_sha256": base_artifact_sha,
        "candidate_descriptor_file": candidate_path.name,
        "candidate_descriptor_sha256": candidate_sha,
        "candidate_delta_sha256": candidate_delta_sha,
        "shadow_manifest_file": shadow_manifest_path.name,
        "shadow_manifest_sha256": shadow_manifest_sha,
        "shadow_ab_state_digest": final_ab_digest,
        "shadow_ab_artifact_file": shadow_path.name,
        "shadow_ab_artifact_sha256": shadow_artifact_sha,
        "fresh_plan_receipt_sha256": gate_report[
            "fresh_plan_receipt_sha256"
        ],
        "fresh_plan_artifact_sha256": gate_report[
            "fresh_plan_artifact_sha256"
        ],
        "qp_contract_receipt_sha256": gate_report[
            "qp_contract_receipt_sha256"
        ],
        "external_verifier_id": gate_report["external_verifier_id"],
        "external_verifier_executable_sha256": gate_report[
            "external_verifier_executable_sha256"
        ],
        "external_verifier_ed25519_public_key_hex": gate_report[
            "external_verifier_ed25519_public_key_hex"
        ],
        "source_manifest_path": str(Path(fresh_inputs.source_manifest.path).resolve()),
        "source_manifest_sha256": fresh_inputs.source_manifest.sha256,
        "official_gaussian_path": str(Path(fresh_inputs.official_gaussian.path).resolve()),
        "official_gaussian_sha256": fresh_inputs.official_gaussian.sha256,
        "exact_frame_count": 81,
        "input_access_mode": "read_only_hash_verified",
        "renderer_must_not_mutate_authoritative_parameters": True,
        "fresh_result_can_authorize_commit": False,
        "finalize_disabled": True,
        "publication_disabled": True,
    }
    request_path = transaction_directory / "fresh-render-request.json"
    request_sha = _create_fsynced_file(
        request_path, canonical_json_bytes(fresh_request)
    )
    return {
        "ok": True,
        "candidate_descriptor_file": candidate_path.name,
        "candidate_descriptor_sha256": candidate_sha,
        "candidate_delta_sha256": candidate_delta_sha,
        "shadow_manifest_file": shadow_manifest_path.name,
        "shadow_manifest_sha256": shadow_manifest_sha,
        "shadow_ab_state_digest": final_ab_digest,
        "anticipated_realized_sha256": realized_sha,
        "fresh_request_file": request_path.name,
        "fresh_request_sha256": request_sha,
        "gate_verification_file": gate_path.name,
        "gate_verification_sha256": gate_sha,
        "gate_verification_digest": gate_report["verification_digest"],
        "qp_contract_receipt_sha256": gate_report[
            "qp_contract_receipt_sha256"
        ],
    }


def prepare_world8_shadow(
    *,
    collective: World8SmallCollective,
    transaction_directory: Path,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
    checkpoint_manifest: CanonicalFileRef,
    local_rank_evidence: mosaic_qp.SPRankEvidence,
    scientific_gate_files: ScientificGateFiles,
    fresh_inputs: FreshInputFiles,
    topology_receipt_digest: str,
    global_trust_radius: float,
    layer_trust_radii: Sequence[mosaic_qp.LayerTrustRadius],
    config: mosaic_qp.JacobianQPConfig = mosaic_qp.JacobianQPConfig(),
) -> ShadowPrepareResult:
    """Durably PREPARE one shadow; never invoke a renderer or mutate live A/B."""

    transaction_directory = Path(transaction_directory)
    snapshot: _ABSnapshot | None = None
    transaction: Mapping[str, Any] | None = None
    validated_snapshot_trust: _ValidatedSnapshotTrust | None = None
    phase = "bootstrap_preflight"
    try:
        _validate_collective(collective)
        bootstrap_error: BaseException | None = None
        bootstrap_payload: Mapping[str, Any] = {}
        try:
            transaction = _verify_transaction_directory(transaction_directory)
            recovery_payload = dict(recover_world8_transaction(transaction_directory))
            bootstrap_payload = {
                "transaction_init_sha256": transaction[
                    "transaction_init_sha256"
                ],
                "transaction_id": transaction["transaction_id"],
                "external_verifier_id": transaction["external_verifier_id"],
                "external_verifier_executable_sha256": transaction[
                    "external_verifier_executable_sha256"
                ],
                "external_verifier_ed25519_public_key_hex": transaction[
                    "external_verifier_ed25519_public_key_hex"
                ],
                "recovery": recovery_payload,
            }
        except BaseException as error:
            bootstrap_error = error
        bootstrap_rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=bootstrap_error is None,
            payload=bootstrap_payload,
            error=bootstrap_error,
        )
        if not all(row["ok"] for row in bootstrap_rows):
            raise World8ShadowError("WORLD8 bootstrap preflight failed collectively")
        if len(
            {object_sha256(dict(row["payload"])) for row in bootstrap_rows}
        ) != 1:
            raise World8ShadowError("WORLD8 bootstrap/recovery state differs by rank")
        assert transaction is not None
        durable_state = bootstrap_rows[0]["payload"]["recovery"]["state"]
        if durable_state == "ABORTED":
            aborted, _ = _read_named_record(
                transaction_directory, "DECISION.json"
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, aborted
            )
        if durable_state not in {"INITIALIZED", "INCOMPLETE_RANK_PREPARE"}:
            if durable_state != "PREPARED_QUARANTINED":
                raise World8ShadowError(
                    f"unsupported durable recovery state during PREPARE: {durable_state}"
                )

        phase = "snapshot_preflight"
        local_error: BaseException | None = None
        local_payload: Mapping[str, Any] = {}
        try:
            snapshot = _snapshot_from_checkpoint_manifest(
                checkpoint_manifest=checkpoint_manifest,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            pinned_policy = ExternalVerifierPolicy(
                verifier_id=transaction["external_verifier_id"],
                verifier_executable_sha256=transaction[
                    "external_verifier_executable_sha256"
                ],
                verifier_ed25519_public_key_hex=transaction[
                    "external_verifier_ed25519_public_key_hex"
                ],
            )
            if scientific_gate_files.verifier_policy != pinned_policy:
                raise World8ShadowError(
                    "scientific verifier policy differs from transaction pin"
                )
            if (
                tuple(mosaic_qp.ACTION_FAMILIES) != ARM_ORDER
                or tuple(mosaic_qp.PRESERVATION_FAMILIES)
                != PRESERVATION_FAMILIES
                or dict(mosaic_qp.SP_GLOBAL_RANKS) != ARM_RANKS
            ):
                raise World8ShadowError("MOSAIC-QP WORLD8 contract differs")
            validated_snapshot_trust = _validated_snapshot_trust(
                snapshot=snapshot,
                ordered_parameters=ordered_parameters,
                layer_trust_radii=layer_trust_radii,
            )
            global_radius = _finite_positive(
                global_trust_radius, label="global trust radius"
            )
            config_payload = _qp_config_payload(config)
            if local_rank_evidence.global_rank != collective.rank:
                raise World8ShadowError("local rank evidence/collective rank differs")
            actor = _actor_for_rank(collective.rank)
            if any(
                row.actor_family != actor
                for row in local_rank_evidence.action_rows
            ):
                raise World8ShadowError("local action row belongs to wrong arm")
            local_payload = {
                "transaction_init_sha256": transaction[
                    "transaction_init_sha256"
                ],
                "checkpoint_manifest_sha256": snapshot.manifest_sha256,
                "base_ab_state_digest": snapshot.state_digest,
                "checkpoint_content_receipt_digest": (
                    snapshot.checkpoint_content_receipt_digest
                ),
                "parameter_layout_digest": (
                    validated_snapshot_trust.layout.layout_digest
                ),
                "zero_b_parameter_state_sha256": (
                    validated_snapshot_trust.layout.parameter_state_sha256
                ),
                "trust_contract_digest": object_sha256(
                    list(validated_snapshot_trust.rows)
                ),
                "topology_receipt_digest": _sha256(
                    topology_receipt_digest, label="topology receipt"
                ),
                "global_trust_radius": global_radius,
                "qp_config_digest": object_sha256(config_payload),
                "qp_contract_receipt_sha256": scientific_gate_files.qp_contract.receipt.sha256,
                "source_manifest_sha256": fresh_inputs.source_manifest.sha256,
                "official_gaussian_sha256": fresh_inputs.official_gaussian.sha256,
                "actor_family": actor,
            }
        except BaseException as error:
            local_error = error
        rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=local_error is None,
            payload=local_payload,
            error=local_error,
        )
        if not all(row["ok"] for row in rows):
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=rows,
            )
            if snapshot is not None:
                _assert_authoritative_unchanged(
                    snapshot,
                    ordered_parameters=ordered_parameters,
                    ordered_fixed_lora_a=ordered_fixed_lora_a,
                )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )
        assert snapshot is not None
        assert validated_snapshot_trust is not None
        consensus_keys = (
            "transaction_init_sha256",
            "checkpoint_manifest_sha256",
            "base_ab_state_digest",
            "checkpoint_content_receipt_digest",
            "parameter_layout_digest",
            "zero_b_parameter_state_sha256",
            "trust_contract_digest",
            "topology_receipt_digest",
            "global_trust_radius",
            "qp_config_digest",
            "qp_contract_receipt_sha256",
            "source_manifest_sha256",
            "official_gaussian_sha256",
        )
        if any(
            len({row["payload"][key] for row in rows}) != 1
            for key in consensus_keys
        ) or tuple(row["payload"]["actor_family"] for row in rows) != (
            "dog",
            "dog",
            "dog",
            "dog",
            "human",
            "human",
            "human",
            "human",
        ):
            synthetic = tuple(
                {
                    "ok": False,
                    "error_type": "WORLD8_PREFLIGHT_CONSENSUS_FAILED",
                }
                for _ in GLOBAL_RANKS
            )
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=synthetic,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )

        if durable_state == "PREPARED_QUARANTINED":
            return _resume_prepared_result(
                transaction_directory=transaction_directory,
                transaction=transaction,
                snapshot=snapshot,
            )

        phase = "rank_evidence_packet"
        local_error = None
        descriptor: Mapping[str, Any] = {}
        try:
            descriptor = _write_rank_evidence_packet(
                transaction_directory=transaction_directory,
                rank_evidence=local_rank_evidence,
                expected_rank=collective.rank,
            )
        except BaseException as error:
            local_error = error
        evidence_rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=local_error is None,
            payload=descriptor,
            error=local_error,
        )
        if not all(row["ok"] for row in evidence_rows):
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=evidence_rows,
            )
            _assert_authoritative_unchanged(
                snapshot,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )
        descriptors = tuple(row["payload"] for row in evidence_rows)

        phase = "rank0_qp_shadow_materialization"
        root_control = None
        if collective.rank == 0:
            try:
                evidence = _build_union(
                    transaction_directory,
                    descriptors,
                    topology_receipt_digest=topology_receipt_digest,
                )
                validated_contract = _validated_qp_contract(
                    snapshot=snapshot,
                    ordered_parameters=ordered_parameters,
                    evidence=evidence,
                    topology_receipt_digest=topology_receipt_digest,
                    global_trust_radius=global_trust_radius,
                    layer_trust_radii=layer_trust_radii,
                    config=config,
                )
                gate_report = _verify_scientific_gate_files(
                    gate_files=scientific_gate_files,
                    evidence=evidence,
                    checkpoint_manifest_sha256=snapshot.manifest_sha256,
                    checkpoint_content_receipt_digest=(
                        snapshot.checkpoint_content_receipt_digest
                    ),
                    fresh_inputs=fresh_inputs,
                    validated_contract=validated_contract,
                    pinned_verifier_policy=pinned_policy,
                )
                root_control = _root_materialize_shadow(
                    transaction_directory=transaction_directory,
                    snapshot=snapshot,
                    evidence=evidence,
                    validated_contract=validated_contract,
                    gate_report=gate_report,
                    fresh_inputs=fresh_inputs,
                )
            except BaseException as error:
                root_control = {
                    "ok": False,
                    "error_type": type(error).__name__,
                }
        control = collective.broadcast_small(root_control, src=0)
        if control.get("ok") is not True:
            failure = tuple(
                {
                    "ok": False,
                    "error_type": control.get("error_type") or "RANK0_SHADOW_FAILED",
                }
                for _ in GLOBAL_RANKS
            )
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=failure,
            )
            _assert_authoritative_unchanged(
                snapshot,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )

        phase = "anticipated_realized_consensus"
        local_error = None
        anticipated_payload: Mapping[str, Any] = {}
        try:
            if (
                control.get("candidate_descriptor_file") != "candidate.json"
                or control.get("shadow_manifest_file") != "shadow-manifest.json"
                or control.get("fresh_request_file") != "fresh-render-request.json"
                or control.get("gate_verification_file")
                != "scientific-gate-verification.json"
            ):
                raise World8ShadowError("rank-0 control filename closure differs")
            candidate_descriptor, _ = _read_canonical_json(
                transaction_directory / control["candidate_descriptor_file"],
                expected_sha256=control["candidate_descriptor_sha256"],
            )
            deltas = _load_delta_artifact(
                transaction_directory, candidate_descriptor
            )
            if (
                candidate_descriptor["checkpoint_manifest_sha256"]
                != snapshot.manifest_sha256
                or candidate_descriptor["base_ab_state_digest"]
                != snapshot.state_digest
                or candidate_descriptor["scientific_gate_verification_digest"]
                != control.get("gate_verification_digest")
            ):
                raise World8ShadowError("local candidate base/gate binding differs")
            _, _, local_final_digest, local_realized_sha = _anticipated_state(
                snapshot=snapshot, delta_by_parameter=deltas
            )
            shadow_manifest, _ = _read_canonical_json(
                transaction_directory / control["shadow_manifest_file"],
                expected_sha256=control["shadow_manifest_sha256"],
            )
            if (
                shadow_manifest.get("schema_version") != SHADOW_MANIFEST_SCHEMA
                or shadow_manifest.get("checkpoint_manifest_sha256")
                != snapshot.manifest_sha256
                or shadow_manifest.get("base_ab_state_digest")
                != snapshot.state_digest
                or shadow_manifest.get("candidate_descriptor_sha256")
                != control["candidate_descriptor_sha256"]
                or shadow_manifest.get("candidate_delta_sha256")
                != control["candidate_delta_sha256"]
                or shadow_manifest.get("anticipated_audit_file")
                != "anticipated-realized-audit.json"
                or shadow_manifest.get("first_step_exact_zero_b") is not True
                or shadow_manifest.get("quarantined_non_publishable") is not True
            ):
                raise World8ShadowError("local shadow manifest binding differs")
            _verify_ab_artifact(
                transaction_directory=transaction_directory,
                artifact_file=shadow_manifest["base_ab_artifact_file"],
                artifact_sha256=shadow_manifest["base_ab_artifact_sha256"],
                entries=shadow_manifest["base_ab_entries"],
                expected_state_digest=shadow_manifest["base_ab_state_digest"],
            )
            _verify_ab_artifact(
                transaction_directory=transaction_directory,
                artifact_file=shadow_manifest["shadow_ab_artifact_file"],
                artifact_sha256=shadow_manifest["shadow_ab_artifact_sha256"],
                entries=shadow_manifest["shadow_ab_entries"],
                expected_state_digest=shadow_manifest["shadow_ab_state_digest"],
            )
            anticipated_audit, _ = _read_canonical_json(
                transaction_directory / shadow_manifest[
                    "anticipated_audit_file"
                ],
                expected_sha256=shadow_manifest[
                    "anticipated_audit_sha256"
                ],
            )
            if (
                local_final_digest != control["shadow_ab_state_digest"]
                or local_realized_sha != control["anticipated_realized_sha256"]
                or anticipated_audit["all_constraints_passed"] is not True
                or anticipated_audit["anticipated_realized_sha256"]
                != local_realized_sha
            ):
                raise World8ShadowError("local anticipated shadow/audit differs")
            _assert_authoritative_unchanged(
                snapshot,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            anticipated_payload = {
                "candidate_descriptor_sha256": control[
                    "candidate_descriptor_sha256"
                ],
                "candidate_delta_sha256": control["candidate_delta_sha256"],
                "anticipated_realized_sha256": local_realized_sha,
                "shadow_ab_state_digest": local_final_digest,
                "authoritative_ab_unchanged": True,
            }
        except BaseException as error:
            local_error = error
        anticipated_rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=local_error is None,
            payload=anticipated_payload,
            error=local_error,
        )
        if not all(row["ok"] for row in anticipated_rows) or len(
            {object_sha256(dict(row["payload"])) for row in anticipated_rows}
        ) != 1:
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=anticipated_rows,
            )
            _assert_authoritative_unchanged(
                snapshot,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )

        phase = "durable_rank_prepare_wal"
        wal = {
            "schema_version": RANK_WAL_SCHEMA,
            "transaction_id": transaction["transaction_id"],
            "global_rank": collective.rank,
            "actor_family": _actor_for_rank(collective.rank),
            "checkpoint_manifest_sha256": snapshot.manifest_sha256,
            "base_ab_state_digest": snapshot.state_digest,
            "candidate_descriptor_sha256": control[
                "candidate_descriptor_sha256"
            ],
            "candidate_delta_sha256": control["candidate_delta_sha256"],
            "anticipated_realized_sha256": control[
                "anticipated_realized_sha256"
            ],
            "shadow_manifest_sha256": control["shadow_manifest_sha256"],
            "shadow_ab_state_digest": control["shadow_ab_state_digest"],
            "fresh_request_sha256": control["fresh_request_sha256"],
            "gate_verification_sha256": control[
                "gate_verification_sha256"
            ],
            "qp_contract_receipt_sha256": control[
                "qp_contract_receipt_sha256"
            ],
            "authoritative_parameters_mutated": False,
            "publisher_visible": False,
            "first_step_exact_zero_b": True,
        }
        wal_path = transaction_directory / f"rank-{collective.rank}.PREPARED.json"
        wal_error: BaseException | None = None
        wal_payload: Mapping[str, Any] = {}
        try:
            wal_sha = _create_fsynced_file(wal_path, canonical_json_bytes(wal))
            wal_payload = {
                "global_rank": collective.rank,
                "wal_file": wal_path.name,
                "wal_sha256": wal_sha,
            }
        except BaseException as error:
            wal_error = error
        wal_rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=wal_error is None,
            payload=wal_payload,
            error=wal_error,
        )
        if not all(row["ok"] for row in wal_rows):
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=wal_rows,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )
        final_control = None
        if collective.rank == 0:
            try:
                global_prepare = {
                    "schema_version": GLOBAL_PREPARE_SCHEMA,
                    "transaction_id": transaction["transaction_id"],
                    "transaction_init_sha256": transaction[
                        "transaction_init_sha256"
                    ],
                    "rank_wals": [dict(row["payload"]) for row in wal_rows],
                    "checkpoint_manifest_sha256": snapshot.manifest_sha256,
                    "base_ab_state_digest": snapshot.state_digest,
                    "candidate_descriptor_file": control[
                        "candidate_descriptor_file"
                    ],
                    "candidate_descriptor_sha256": control[
                        "candidate_descriptor_sha256"
                    ],
                    "candidate_delta_sha256": control["candidate_delta_sha256"],
                    "anticipated_realized_sha256": control[
                        "anticipated_realized_sha256"
                    ],
                    "shadow_manifest_file": control["shadow_manifest_file"],
                    "shadow_manifest_sha256": control["shadow_manifest_sha256"],
                    "shadow_ab_state_digest": control["shadow_ab_state_digest"],
                    "fresh_request_file": control["fresh_request_file"],
                    "fresh_request_sha256": control["fresh_request_sha256"],
                    "scientific_gate_verification_file": control[
                        "gate_verification_file"
                    ],
                    "scientific_gate_verification_sha256": control[
                        "gate_verification_sha256"
                    ],
                    "qp_contract_receipt_sha256": control[
                        "qp_contract_receipt_sha256"
                    ],
                    "authoritative_parameters_mutated": False,
                    "external_publication_authorized": False,
                    "first_step_exact_zero_b": True,
                    "quarantined_non_publishable": True,
                    "finalize_disabled": True,
                    "publication_disabled": True,
                }
                prepare_sha = _atomic_publish_record(
                    transaction_directory / "PREPARED.json", global_prepare
                )
                final_control = {
                    "prepared": True,
                    "prepared_sha256": prepare_sha,
                    **control,
                }
            except BaseException as error:
                final_control = {
                    "prepared": False,
                    "error_type": type(error).__name__,
                }
        result_control = collective.broadcast_small(final_control, src=0)
        if result_control.get("prepared") is not True:
            failure = tuple(
                {
                    "ok": False,
                    "error_type": result_control.get("error_type")
                    or "GLOBAL_PREPARE_RECORD_FAILED",
                }
                for _ in GLOBAL_RANKS
            )
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=failure,
            )
            return ShadowPrepareResult(
                False, True, transaction_directory, None, None, None, abort
            )
        _assert_authoritative_unchanged(
            snapshot,
            ordered_parameters=ordered_parameters,
            ordered_fixed_lora_a=ordered_fixed_lora_a,
        )
        receipt = {
            **dict(result_control),
            "transaction_id": transaction["transaction_id"],
            "authoritative_parameters_mutated": False,
            "fresh_renderer_invoked_by_this_module": False,
            "durable_commit_recorded": False,
            "checkpoint_published": False,
            "quarantined_non_publishable": True,
            "finalize_disabled": True,
            "publication_disabled": True,
        }
        return ShadowPrepareResult(
            prepared=True,
            aborted=False,
            transaction_directory=transaction_directory,
            candidate_descriptor=CanonicalFileRef(
                transaction_directory / result_control[
                    "candidate_descriptor_file"
                ],
                result_control["candidate_descriptor_sha256"],
            ),
            shadow_manifest=CanonicalFileRef(
                transaction_directory / result_control["shadow_manifest_file"],
                result_control["shadow_manifest_sha256"],
            ),
            fresh_render_request=CanonicalFileRef(
                transaction_directory / result_control["fresh_request_file"],
                result_control["fresh_request_sha256"],
            ),
            receipt=receipt,
        )
    except BaseException as error:
        unchanged = False
        if snapshot is not None:
            try:
                _assert_authoritative_unchanged(
                    snapshot,
                    ordered_parameters=ordered_parameters,
                    ordered_fixed_lora_a=ordered_fixed_lora_a,
                )
                unchanged = True
            except BaseException:
                unchanged = False
        _write_failure_envelope(
            transaction_directory=transaction_directory,
            rank=getattr(collective, "rank", -1),
            phase=phase,
            error=error,
            snapshot=snapshot,
            authoritative_unchanged=unchanged,
        )
        if not unchanged and snapshot is not None:
            raise World8FailStopError(
                "authoritative A/B changed during failed shadow transaction"
            ) from error
        raise World8FailStopError(
            "shadow transaction failed or lost its process group; authoritative A/B is unchanged"
        ) from error


def _read_named_record(
    directory: Path, name: str, *, expected_sha256: str | None = None
) -> tuple[Mapping[str, Any], str]:
    allowed = {
        "TRANSACTION.json",
        "DECISION.json",
        "PREPARED.json",
        "candidate.json",
        "shadow-manifest.json",
        "fresh-render-request.json",
        "scientific-gate-verification.json",
        "anticipated-realized-audit.json",
    }
    if name not in allowed and re.fullmatch(r"rank-[0-7]\.PREPARED\.json", name) is None:
        raise World8ShadowError("record filename escapes the closed transaction protocol")
    path = directory / name
    raw = _read_stable_file(
        path,
        expected_sha256=expected_sha256,
        require_read_only=True,
    )
    digest = hashlib.sha256(raw).hexdigest()
    value, _ = _read_canonical_json(path, expected_sha256=digest)
    return value, digest


def _read_validated_prepared_graph(
    *,
    transaction_directory: Path,
    transaction: Mapping[str, Any],
    snapshot: _ABSnapshot | None,
) -> tuple[Mapping[str, Any], str]:
    prepared, prepared_sha = _read_named_record(
        transaction_directory, "PREPARED.json"
    )
    expected_keys = {
        "schema_version",
        "transaction_id",
        "transaction_init_sha256",
        "rank_wals",
        "checkpoint_manifest_sha256",
        "base_ab_state_digest",
        "candidate_descriptor_file",
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "anticipated_realized_sha256",
        "shadow_manifest_file",
        "shadow_manifest_sha256",
        "shadow_ab_state_digest",
        "fresh_request_file",
        "fresh_request_sha256",
        "scientific_gate_verification_file",
        "scientific_gate_verification_sha256",
        "qp_contract_receipt_sha256",
        "authoritative_parameters_mutated",
        "external_publication_authorized",
        "first_step_exact_zero_b",
        "quarantined_non_publishable",
        "finalize_disabled",
        "publication_disabled",
    }
    if set(prepared) != expected_keys or (
        prepared["schema_version"] != GLOBAL_PREPARE_SCHEMA
        or prepared["transaction_id"] != transaction["transaction_id"]
        or prepared["transaction_init_sha256"]
        != transaction["transaction_init_sha256"]
        or prepared["candidate_descriptor_file"] != "candidate.json"
        or prepared["shadow_manifest_file"] != "shadow-manifest.json"
        or prepared["fresh_request_file"] != "fresh-render-request.json"
        or prepared["scientific_gate_verification_file"]
        != "scientific-gate-verification.json"
        or prepared["authoritative_parameters_mutated"] is not False
        or prepared["external_publication_authorized"] is not False
        or prepared["first_step_exact_zero_b"] is not True
        or prepared["quarantined_non_publishable"] is not True
        or prepared["finalize_disabled"] is not True
        or prepared["publication_disabled"] is not True
    ):
        raise World8ShadowError("PREPARED record closure differs")
    digest_keys = (
        "checkpoint_manifest_sha256",
        "base_ab_state_digest",
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "anticipated_realized_sha256",
        "shadow_manifest_sha256",
        "shadow_ab_state_digest",
        "fresh_request_sha256",
        "scientific_gate_verification_sha256",
        "qp_contract_receipt_sha256",
    )
    for key in digest_keys:
        _sha256(prepared[key], label=f"PREPARED {key}")
    if snapshot is not None and (
        prepared["checkpoint_manifest_sha256"] != snapshot.manifest_sha256
        or prepared["base_ab_state_digest"] != snapshot.state_digest
    ):
        raise World8ShadowError("PREPARED base differs from current checkpoint snapshot")
    rank_wals = prepared["rank_wals"]
    if not isinstance(rank_wals, list) or len(rank_wals) != WORLD_SIZE:
        raise World8ShadowError("PREPARED WAL closure differs")
    wal_keys = {
        "schema_version",
        "transaction_id",
        "global_rank",
        "actor_family",
        "checkpoint_manifest_sha256",
        "base_ab_state_digest",
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "anticipated_realized_sha256",
        "shadow_manifest_sha256",
        "shadow_ab_state_digest",
        "fresh_request_sha256",
        "gate_verification_sha256",
        "qp_contract_receipt_sha256",
        "authoritative_parameters_mutated",
        "publisher_visible",
        "first_step_exact_zero_b",
    }
    for expected_rank, wal_ref in enumerate(rank_wals):
        if set(wal_ref) != {"global_rank", "wal_file", "wal_sha256"} or (
            wal_ref["global_rank"] != expected_rank
            or wal_ref["wal_file"] != f"rank-{expected_rank}.PREPARED.json"
        ):
            raise World8ShadowError("PREPARED WAL reference ordering differs")
        wal, _ = _read_named_record(
            transaction_directory,
            wal_ref["wal_file"],
            expected_sha256=_sha256(wal_ref["wal_sha256"], label="rank WAL SHA"),
        )
        if set(wal) != wal_keys or (
            wal["schema_version"] != RANK_WAL_SCHEMA
            or wal["transaction_id"] != transaction["transaction_id"]
            or wal["global_rank"] != expected_rank
            or wal["actor_family"] != _actor_for_rank(expected_rank)
            or wal["checkpoint_manifest_sha256"]
            != prepared["checkpoint_manifest_sha256"]
            or wal["base_ab_state_digest"] != prepared["base_ab_state_digest"]
            or wal["candidate_descriptor_sha256"]
            != prepared["candidate_descriptor_sha256"]
            or wal["candidate_delta_sha256"] != prepared["candidate_delta_sha256"]
            or wal["anticipated_realized_sha256"]
            != prepared["anticipated_realized_sha256"]
            or wal["shadow_manifest_sha256"] != prepared["shadow_manifest_sha256"]
            or wal["shadow_ab_state_digest"] != prepared["shadow_ab_state_digest"]
            or wal["fresh_request_sha256"] != prepared["fresh_request_sha256"]
            or wal["gate_verification_sha256"]
            != prepared["scientific_gate_verification_sha256"]
            or wal["qp_contract_receipt_sha256"]
            != prepared["qp_contract_receipt_sha256"]
            or wal["authoritative_parameters_mutated"] is not False
            or wal["publisher_visible"] is not False
            or wal["first_step_exact_zero_b"] is not True
        ):
            raise World8ShadowError("rank PREPARED WAL binding differs")

    candidate, _ = _read_named_record(
        transaction_directory,
        "candidate.json",
        expected_sha256=prepared["candidate_descriptor_sha256"],
    )
    deltas = _load_delta_artifact(transaction_directory, candidate)
    if (
        candidate["checkpoint_manifest_sha256"]
        != prepared["checkpoint_manifest_sha256"]
        or candidate["base_ab_state_digest"] != prepared["base_ab_state_digest"]
        or candidate["candidate_delta_sha256"] != prepared["candidate_delta_sha256"]
    ):
        raise World8ShadowError("candidate/PREPARED base binding differs")
    del deltas

    shadow, _ = _read_named_record(
        transaction_directory,
        "shadow-manifest.json",
        expected_sha256=prepared["shadow_manifest_sha256"],
    )
    shadow_keys = {
        "schema_version",
        "checkpoint_manifest_sha256",
        "checkpoint_content_receipt_digest",
        "base_ab_state_digest",
        "base_ab_artifact_file",
        "base_ab_artifact_sha256",
        "base_ab_entries",
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "anticipated_realized_sha256",
        "anticipated_audit_file",
        "anticipated_audit_sha256",
        "shadow_ab_state_digest",
        "shadow_ab_artifact_file",
        "shadow_ab_artifact_sha256",
        "shadow_ab_entries",
        "authoritative_parameters_mutated",
        "first_step_exact_zero_b",
        "quarantined_non_publishable",
    }
    if set(shadow) != shadow_keys or (
        shadow["schema_version"] != SHADOW_MANIFEST_SCHEMA
        or shadow["checkpoint_manifest_sha256"]
        != prepared["checkpoint_manifest_sha256"]
        or shadow["base_ab_state_digest"] != prepared["base_ab_state_digest"]
        or shadow["base_ab_artifact_file"] != "base-ab-state.fp32.bin"
        or shadow["candidate_descriptor_sha256"]
        != prepared["candidate_descriptor_sha256"]
        or shadow["candidate_delta_sha256"] != prepared["candidate_delta_sha256"]
        or shadow["anticipated_realized_sha256"]
        != prepared["anticipated_realized_sha256"]
        or shadow["anticipated_audit_file"] != "anticipated-realized-audit.json"
        or shadow["shadow_ab_state_digest"] != prepared["shadow_ab_state_digest"]
        or shadow["shadow_ab_artifact_file"] != "shadow-ab-state.fp32.bin"
        or shadow["authoritative_parameters_mutated"] is not False
        or shadow["first_step_exact_zero_b"] is not True
        or shadow["quarantined_non_publishable"] is not True
    ):
        raise World8ShadowError("shadow manifest closure/binding differs")
    _verify_ab_artifact(
        transaction_directory=transaction_directory,
        artifact_file=shadow["base_ab_artifact_file"],
        artifact_sha256=shadow["base_ab_artifact_sha256"],
        entries=shadow["base_ab_entries"],
        expected_state_digest=shadow["base_ab_state_digest"],
    )
    _verify_ab_artifact(
        transaction_directory=transaction_directory,
        artifact_file=shadow["shadow_ab_artifact_file"],
        artifact_sha256=shadow["shadow_ab_artifact_sha256"],
        entries=shadow["shadow_ab_entries"],
        expected_state_digest=shadow["shadow_ab_state_digest"],
    )
    anticipated, _ = _read_named_record(
        transaction_directory,
        "anticipated-realized-audit.json",
        expected_sha256=shadow["anticipated_audit_sha256"],
    )
    if (
        anticipated.get("all_constraints_passed") is not True
        or anticipated.get("anticipated_realized_sha256")
        != prepared["anticipated_realized_sha256"]
        or anticipated.get("audit_uses_fl_b_plus_delta_minus_b") is not True
    ):
        raise World8ShadowError("anticipated realized audit differs")

    gate, _ = _read_named_record(
        transaction_directory,
        "scientific-gate-verification.json",
        expected_sha256=prepared["scientific_gate_verification_sha256"],
    )
    gate_unsigned = dict(gate)
    gate_digest = gate_unsigned.pop("verification_digest", None)
    if (
        gate_digest != object_sha256(gate_unsigned)
        or gate.get("qp_contract_receipt_sha256")
        != prepared["qp_contract_receipt_sha256"]
        or gate.get("checkpoint_manifest_sha256")
        != prepared["checkpoint_manifest_sha256"]
        or gate.get("self_sealed_pass_booleans_consumed") is not False
    ):
        raise World8ShadowError("scientific gate verification binding differs")

    request, _ = _read_named_record(
        transaction_directory,
        "fresh-render-request.json",
        expected_sha256=prepared["fresh_request_sha256"],
    )
    if (
        request.get("schema_version") != FRESH_REQUEST_SCHEMA
        or request.get("checkpoint_manifest_sha256")
        != prepared["checkpoint_manifest_sha256"]
        or request.get("candidate_descriptor_file") != "candidate.json"
        or request.get("candidate_descriptor_sha256")
        != prepared["candidate_descriptor_sha256"]
        or request.get("candidate_delta_sha256") != prepared["candidate_delta_sha256"]
        or request.get("shadow_manifest_file") != "shadow-manifest.json"
        or request.get("shadow_manifest_sha256") != prepared["shadow_manifest_sha256"]
        or request.get("shadow_ab_state_digest") != prepared["shadow_ab_state_digest"]
        or request.get("shadow_ab_artifact_file") != "shadow-ab-state.fp32.bin"
        or request.get("qp_contract_receipt_sha256")
        != prepared["qp_contract_receipt_sha256"]
        or request.get("exact_frame_count") != 81
        or request.get("fresh_result_can_authorize_commit") is not False
        or request.get("finalize_disabled") is not True
        or request.get("publication_disabled") is not True
    ):
        raise World8ShadowError("fresh request closure/binding differs")
    _hash_stable_file(
        Path(request["source_manifest_path"]),
        expected_sha256=request["source_manifest_sha256"],
        maximum_bytes=MAX_SOURCE_MANIFEST_BYTES,
        require_read_only=True,
    )
    _hash_stable_file(
        Path(request["official_gaussian_path"]),
        expected_sha256=request["official_gaussian_sha256"],
        maximum_bytes=MAX_GAUSSIAN_BYTES,
        require_read_only=True,
    )
    return prepared, prepared_sha


def _resume_prepared_result(
    *,
    transaction_directory: Path,
    transaction: Mapping[str, Any],
    snapshot: _ABSnapshot,
) -> ShadowPrepareResult:
    """Strictly revalidate and return one quarantined prior PREPARE."""

    prepared, prepared_sha = _read_validated_prepared_graph(
        transaction_directory=transaction_directory,
        transaction=transaction,
        snapshot=snapshot,
    )
    receipt = {
        "prepared": True,
        "resumed_from_durable_state": True,
        "prepared_sha256": prepared_sha,
        "transaction_id": transaction["transaction_id"],
        "candidate_descriptor_file": "candidate.json",
        "candidate_descriptor_sha256": prepared["candidate_descriptor_sha256"],
        "candidate_delta_sha256": prepared["candidate_delta_sha256"],
        "shadow_manifest_file": "shadow-manifest.json",
        "shadow_manifest_sha256": prepared["shadow_manifest_sha256"],
        "shadow_ab_state_digest": prepared["shadow_ab_state_digest"],
        "fresh_request_file": "fresh-render-request.json",
        "fresh_request_sha256": prepared["fresh_request_sha256"],
        "authoritative_parameters_mutated": False,
        "fresh_renderer_invoked_by_this_module": False,
        "durable_commit_recorded": False,
        "checkpoint_published": False,
        "quarantined_non_publishable": True,
        "finalize_disabled": True,
        "publication_disabled": True,
    }
    return ShadowPrepareResult(
        prepared=True,
        aborted=False,
        transaction_directory=transaction_directory,
        candidate_descriptor=CanonicalFileRef(
            transaction_directory / "candidate.json",
            prepared["candidate_descriptor_sha256"],
        ),
        shadow_manifest=CanonicalFileRef(
            transaction_directory / "shadow-manifest.json",
            prepared["shadow_manifest_sha256"],
        ),
        fresh_render_request=CanonicalFileRef(
            transaction_directory / "fresh-render-request.json",
            prepared["fresh_request_sha256"],
        ),
        receipt=receipt,
    )

    prepared, prepared_sha = _read_named_record(
        transaction_directory, "PREPARED.json"
    )
    expected_keys = {
        "schema_version",
        "transaction_id",
        "transaction_init_sha256",
        "rank_wals",
        "candidate_descriptor_file",
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "shadow_manifest_file",
        "shadow_manifest_sha256",
        "shadow_ab_state_digest",
        "fresh_request_file",
        "fresh_request_sha256",
        "authoritative_parameters_mutated",
        "external_publication_authorized",
    }
    if set(prepared) != expected_keys or (
        prepared["schema_version"] != GLOBAL_PREPARE_SCHEMA
        or prepared["transaction_id"] != transaction["transaction_id"]
        or prepared["transaction_init_sha256"]
        != transaction["transaction_init_sha256"]
        or prepared["candidate_descriptor_file"] != "candidate.json"
        or prepared["shadow_manifest_file"] != "shadow-manifest.json"
        or prepared["fresh_request_file"] != "fresh-render-request.json"
        or prepared["authoritative_parameters_mutated"] is not False
        or prepared["external_publication_authorized"] is not False
    ):
        raise World8ShadowError("resumed PREPARED record closure differs")
    for key in (
        "candidate_descriptor_sha256",
        "candidate_delta_sha256",
        "shadow_manifest_sha256",
        "shadow_ab_state_digest",
        "fresh_request_sha256",
    ):
        _sha256(prepared[key], label=f"resumed PREPARED {key}")
    if not isinstance(prepared["rank_wals"], list) or len(
        prepared["rank_wals"]
    ) != WORLD_SIZE:
        raise World8ShadowError("resumed PREPARED WAL closure differs")
    for expected_rank, wal_ref in enumerate(prepared["rank_wals"]):
        if set(wal_ref) != {"global_rank", "wal_file", "wal_sha256"} or (
            wal_ref["global_rank"] != expected_rank
            or wal_ref["wal_file"] != f"rank-{expected_rank}.PREPARED.json"
        ):
            raise World8ShadowError("resumed PREPARED WAL ordering differs")
        _read_named_record(
            transaction_directory,
            wal_ref["wal_file"],
            expected_sha256=_sha256(
                wal_ref["wal_sha256"], label="resumed rank WAL SHA"
            ),
        )
    candidate, _ = _read_named_record(
        transaction_directory,
        prepared["candidate_descriptor_file"],
        expected_sha256=prepared["candidate_descriptor_sha256"],
    )
    shadow, _ = _read_named_record(
        transaction_directory,
        prepared["shadow_manifest_file"],
        expected_sha256=prepared["shadow_manifest_sha256"],
    )
    request, _ = _read_named_record(
        transaction_directory,
        prepared["fresh_request_file"],
        expected_sha256=prepared["fresh_request_sha256"],
    )
    if (
        candidate.get("schema_version") != CANDIDATE_DESCRIPTOR_SCHEMA
        or shadow.get("schema_version") != SHADOW_MANIFEST_SCHEMA
        or request.get("schema_version") != FRESH_REQUEST_SCHEMA
        or candidate.get("candidate_delta_sha256")
        != prepared["candidate_delta_sha256"]
        or shadow.get("shadow_ab_state_digest")
        != prepared["shadow_ab_state_digest"]
        or request.get("shadow_ab_state_digest")
        != prepared["shadow_ab_state_digest"]
    ):
        raise World8ShadowError("resumed PREPARED artifact binding differs")
    if durable_commit_recorded:
        commit, _ = _read_named_record(transaction_directory, "COMMIT.json")
        if (
            commit.get("schema_version") != COMMIT_SCHEMA
            or commit.get("prepared_sha256") != prepared_sha
            or commit.get("shadow_manifest_sha256")
            != prepared["shadow_manifest_sha256"]
            or commit.get("authoritative_parameters_mutated") is not False
        ):
            raise World8ShadowError("resumed COMMIT/PREPARED binding differs")
    receipt = {
        "prepared": True,
        "resumed_from_durable_state": True,
        "prepared_sha256": prepared_sha,
        "transaction_id": transaction["transaction_id"],
        "candidate_descriptor_file": prepared["candidate_descriptor_file"],
        "candidate_descriptor_sha256": prepared[
            "candidate_descriptor_sha256"
        ],
        "candidate_delta_sha256": prepared["candidate_delta_sha256"],
        "shadow_manifest_file": prepared["shadow_manifest_file"],
        "shadow_manifest_sha256": prepared["shadow_manifest_sha256"],
        "shadow_ab_state_digest": prepared["shadow_ab_state_digest"],
        "fresh_request_file": prepared["fresh_request_file"],
        "fresh_request_sha256": prepared["fresh_request_sha256"],
        "authoritative_parameters_mutated": False,
        "fresh_renderer_invoked_by_this_module": False,
        "durable_commit_recorded": bool(durable_commit_recorded),
        "checkpoint_published": False,
    }
    return ShadowPrepareResult(
        prepared=True,
        aborted=False,
        transaction_directory=transaction_directory,
        candidate_descriptor=CanonicalFileRef(
            transaction_directory / prepared["candidate_descriptor_file"],
            prepared["candidate_descriptor_sha256"],
        ),
        shadow_manifest=CanonicalFileRef(
            transaction_directory / prepared["shadow_manifest_file"],
            prepared["shadow_manifest_sha256"],
        ),
        fresh_render_request=CanonicalFileRef(
            transaction_directory / prepared["fresh_request_file"],
            prepared["fresh_request_sha256"],
        ),
        receipt=receipt,
    )


def finalize_world8_shadow(
    *,
    collective: World8SmallCollective,
    transaction_directory: Path,
    ordered_parameters: Sequence[tuple[str, torch.Tensor]],
    ordered_fixed_lora_a: Sequence[tuple[str, torch.Tensor]],
    checkpoint_manifest: CanonicalFileRef,
    verifier_policy: ExternalVerifierPolicy,
    local_fresh_receipt: ExternalReceiptRef,
) -> ShadowFinalizeResult:
    """Fail closed: v1 cannot convert a quarantined shadow into a COMMIT."""

    raise World8ShadowError(
        "finalize/COMMIT is disabled in quarantined shadow-prepare-only v1"
    )

    transaction_directory = Path(transaction_directory)
    snapshot: _ABSnapshot | None = None
    phase = "finalize_preflight"
    try:
        _validate_collective(collective)
        transaction = _verify_transaction_directory(transaction_directory)
        # Durable decisions are idempotent and dominate a resumed process.
        if (transaction_directory / "COMMIT.json").exists() and (
            transaction_directory / "ABORTED.json"
        ).exists():
            raise World8ShadowError("transaction has conflicting durable decisions")
        if (transaction_directory / "COMMIT.json").exists():
            commit, commit_sha = _read_named_record(
                transaction_directory, "COMMIT.json"
            )
            if commit.get("schema_version") != COMMIT_SCHEMA:
                raise World8ShadowError("existing COMMIT record schema differs")
            return ShadowFinalizeResult(
                durable_commit_recorded=True,
                aborted=False,
                authoritative_parameters_mutated=False,
                publishable_shadow_authorized=bool(
                    commit.get("publishable_shadow_authorized")
                ),
                commit_record=CanonicalFileRef(
                    transaction_directory / "COMMIT.json", commit_sha
                ),
                receipt=commit,
            )
        if (transaction_directory / "ABORTED.json").exists():
            aborted, _ = _read_named_record(transaction_directory, "ABORTED.json")
            return ShadowFinalizeResult(
                False, True, False, False, None, aborted
            )
        prepared, prepared_sha = _read_named_record(
            transaction_directory, "PREPARED.json"
        )
        if prepared.get("schema_version") != GLOBAL_PREPARE_SCHEMA:
            raise World8ShadowError("global PREPARED record schema differs")
        if not isinstance(prepared.get("rank_wals"), list) or len(
            prepared["rank_wals"]
        ) != WORLD_SIZE:
            raise World8ShadowError("global PREPARED WAL closure differs")
        for expected_rank, wal_ref in enumerate(prepared["rank_wals"]):
            if wal_ref.get("global_rank") != expected_rank:
                raise World8ShadowError("global PREPARED WAL ordering differs")
            wal, _ = _read_named_record(
                transaction_directory,
                wal_ref["wal_file"],
                expected_sha256=wal_ref["wal_sha256"],
            )
            if (
                wal.get("schema_version") != RANK_WAL_SCHEMA
                or wal.get("global_rank") != expected_rank
                or wal.get("candidate_descriptor_sha256")
                != prepared["candidate_descriptor_sha256"]
                or wal.get("shadow_manifest_sha256")
                != prepared["shadow_manifest_sha256"]
                or wal.get("fresh_request_sha256")
                != prepared["fresh_request_sha256"]
                or wal.get("authoritative_parameters_mutated") is not False
            ):
                raise World8ShadowError("rank PREPARED WAL binding differs")

        local_error: BaseException | None = None
        payload: Mapping[str, Any] = {}
        try:
            snapshot = _snapshot_from_checkpoint_manifest(
                checkpoint_manifest=checkpoint_manifest,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            candidate_descriptor, _ = _read_canonical_json(
                transaction_directory / prepared["candidate_descriptor_file"],
                expected_sha256=prepared["candidate_descriptor_sha256"],
            )
            if snapshot.manifest_sha256 != candidate_descriptor[
                "checkpoint_manifest_sha256"
            ]:
                raise World8ShadowError("prepared candidate/checkpoint binding differs")
            if snapshot.state_digest != candidate_descriptor["base_ab_state_digest"]:
                raise World8ShadowError("authoritative A/B no longer matches PREPARED base")
            _load_delta_artifact(transaction_directory, candidate_descriptor)
            shadow_manifest, _ = _read_canonical_json(
                transaction_directory / prepared["shadow_manifest_file"],
                expected_sha256=prepared["shadow_manifest_sha256"],
            )
            _verify_ab_artifact(
                transaction_directory=transaction_directory,
                artifact_file=shadow_manifest["base_ab_artifact_file"],
                artifact_sha256=shadow_manifest["base_ab_artifact_sha256"],
                entries=shadow_manifest["base_ab_entries"],
                expected_state_digest=shadow_manifest["base_ab_state_digest"],
            )
            _verify_ab_artifact(
                transaction_directory=transaction_directory,
                artifact_file=shadow_manifest["shadow_ab_artifact_file"],
                artifact_sha256=shadow_manifest["shadow_ab_artifact_sha256"],
                entries=shadow_manifest["shadow_ab_entries"],
                expected_state_digest=shadow_manifest["shadow_ab_state_digest"],
            )
            _assert_authoritative_unchanged(
                snapshot,
                ordered_parameters=ordered_parameters,
                ordered_fixed_lora_a=ordered_fixed_lora_a,
            )
            request, _ = _read_canonical_json(
                transaction_directory / prepared["fresh_request_file"],
                expected_sha256=prepared["fresh_request_sha256"],
            )
            if request.get("schema_version") != FRESH_REQUEST_SCHEMA:
                raise World8ShadowError("fresh request schema differs")
            _read_stable_file(
                Path(request["source_manifest_path"]),
                expected_sha256=request["source_manifest_sha256"],
                require_read_only=True,
            )
            _read_stable_file(
                Path(request["official_gaussian_path"]),
                expected_sha256=request["official_gaussian_sha256"],
                require_read_only=True,
            )
            if (
                verifier_policy.verifier_id != request["external_verifier_id"]
                or verifier_policy.verifier_executable_sha256
                != request["external_verifier_executable_sha256"]
                or verifier_policy.verifier_ed25519_public_key_hex
                != request["external_verifier_ed25519_public_key_hex"]
            ):
                raise World8ShadowError(
                    "fresh verifier identity differs from PREPARED scientific gate"
                )
            actor = _actor_for_rank(collective.rank)
            verifier = CanonicalExternalVerifier(verifier_policy)
            fresh = verifier.verify(
                local_fresh_receipt,
                audit_type="fresh_exact81_endpoint",
                exact_bindings={
                    "actor_family": actor,
                    "fresh_request_sha256": prepared["fresh_request_sha256"],
                    "candidate_delta_sha256": prepared["candidate_delta_sha256"],
                    "shadow_ab_state_digest": prepared["shadow_ab_state_digest"],
                    "shadow_manifest_sha256": prepared["shadow_manifest_sha256"],
                    "fresh_plan_receipt_sha256": request[
                        "fresh_plan_receipt_sha256"
                    ],
                    "source_manifest_sha256": request[
                        "source_manifest_sha256"
                    ],
                    "official_gaussian_sha256": request[
                        "official_gaussian_sha256"
                    ],
                    "exact_frame_count": 81,
                    "full_action_conjunction_passed": True,
                    "preservation_families_noninferior": list(
                        PRESERVATION_FAMILIES
                    ),
                    "source_disjoint_confirmation_passed": True,
                },
            )
            payload = {
                "global_rank": collective.rank,
                "actor_family": actor,
                "fresh_receipt_sha256": fresh["receipt_sha256"],
                "fresh_artifact_sha256": fresh["artifact_sha256"],
                "prepared_sha256": prepared_sha,
                "authoritative_ab_unchanged": True,
            }
        except BaseException as error:
            local_error = error
        rows = _gather_envelopes(
            collective,
            phase=phase,
            ok=local_error is None,
            payload=payload,
            error=local_error,
        )
        if not all(row["ok"] for row in rows):
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=rows,
            )
            if snapshot is not None:
                _assert_authoritative_unchanged(
                    snapshot,
                    ordered_parameters=ordered_parameters,
                    ordered_fixed_lora_a=ordered_fixed_lora_a,
                )
            return ShadowFinalizeResult(False, True, False, False, None, abort)
        expected_actors = (
            "dog",
            "dog",
            "dog",
            "dog",
            "human",
            "human",
            "human",
            "human",
        )
        if tuple(row["payload"]["actor_family"] for row in rows) != expected_actors:
            raise World8ShadowError("fresh WORLD8 arm mapping differs")
        for ranks in ARM_RANKS.values():
            if len(
                {
                    (
                        rows[rank]["payload"]["fresh_receipt_sha256"],
                        rows[rank]["payload"]["fresh_artifact_sha256"],
                    )
                    for rank in ranks
                }
            ) != 1:
                failure = tuple(
                    {
                        "ok": False,
                        "error_type": "FRESH_ARM_IMMUTABLE_RECEIPT_CONSENSUS_FAILED",
                    }
                    for _ in GLOBAL_RANKS
                )
                abort = _distributed_abort(
                    collective,
                    transaction_directory=transaction_directory,
                    transaction=transaction,
                    phase="fresh_receipt_consensus",
                    rows=failure,
                )
                return ShadowFinalizeResult(False, True, False, False, None, abort)

        phase = "atomic_commit_record"
        control = None
        if collective.rank == 0:
            try:
                recovery = recover_world8_transaction(transaction_directory)
                if recovery["state"] == "ABORTED":
                    raise World8ShadowError(
                        "cannot record COMMIT after the durable ABORT decision"
                    )
                if recovery["state"] == "COMMITTED_UNPUBLISHED":
                    commit, commit_sha = _read_named_record(
                        transaction_directory, "COMMIT.json"
                    )
                    if commit.get("prepared_sha256") != prepared_sha:
                        raise World8ShadowError(
                            "existing COMMIT belongs to another PREPARED state"
                        )
                    control = {
                        "ok": True,
                        "commit_file": "COMMIT.json",
                        "commit_sha256": commit_sha,
                        "publishable_shadow_authorized": bool(
                            commit.get("publishable_shadow_authorized")
                        ),
                        "idempotent_existing_decision": True,
                    }
                elif recovery["state"] != "PREPARED_AWAITING_FRESH":
                    raise World8ShadowError(
                        "COMMIT requires one fully durable PREPARED state"
                    )
                else:
                    fresh_verification = {
                        "schema_version": FRESH_VERIFICATION_SCHEMA,
                        "transaction_id": transaction["transaction_id"],
                        "prepared_sha256": prepared_sha,
                        "rank_fresh_receipts": [
                            dict(row["payload"]) for row in rows
                        ],
                        "dog_receipt_sha256": rows[0]["payload"][
                            "fresh_receipt_sha256"
                        ],
                        "dog_artifact_sha256": rows[0]["payload"][
                            "fresh_artifact_sha256"
                        ],
                        "human_receipt_sha256": rows[4]["payload"][
                            "fresh_receipt_sha256"
                        ],
                        "human_artifact_sha256": rows[4]["payload"][
                            "fresh_artifact_sha256"
                        ],
                        "all8_passed": True,
                    }
                    fresh_sha = _create_fsynced_file(
                        transaction_directory / "fresh-verification.json",
                        canonical_json_bytes(fresh_verification),
                    )
                    production_transport = (
                        type(collective) is TorchWorld8SmallCollective
                    )
                    commit = {
                        "schema_version": COMMIT_SCHEMA,
                        "transaction_id": transaction["transaction_id"],
                        "transaction_init_sha256": transaction[
                            "transaction_init_sha256"
                        ],
                        "prepared_file": "PREPARED.json",
                        "prepared_sha256": prepared_sha,
                        "fresh_verification_file": "fresh-verification.json",
                        "fresh_verification_sha256": fresh_sha,
                        "candidate_descriptor_file": prepared[
                            "candidate_descriptor_file"
                        ],
                        "candidate_descriptor_sha256": prepared[
                            "candidate_descriptor_sha256"
                        ],
                        "candidate_delta_sha256": prepared[
                            "candidate_delta_sha256"
                        ],
                        "shadow_manifest_file": prepared[
                            "shadow_manifest_file"
                        ],
                        "shadow_manifest_sha256": prepared[
                            "shadow_manifest_sha256"
                        ],
                        "shadow_ab_state_digest": prepared[
                            "shadow_ab_state_digest"
                        ],
                        "production_world8_transport": production_transport,
                        "publishable_shadow_authorized": production_transport,
                        "authoritative_parameters_mutated": False,
                        "checkpoint_published": False,
                    }
                    commit_sha = _atomic_publish_record(
                        transaction_directory / "COMMIT.json", commit
                    )
                    control = {
                        "ok": True,
                        "commit_file": "COMMIT.json",
                        "commit_sha256": commit_sha,
                        "publishable_shadow_authorized": production_transport,
                        "idempotent_existing_decision": False,
                    }
            except BaseException as error:
                control = {"ok": False, "error_type": type(error).__name__}
        decision = collective.broadcast_small(control, src=0)
        if decision.get("ok") is not True:
            failure = tuple(
                {
                    "ok": False,
                    "error_type": decision.get("error_type")
                    or "ATOMIC_COMMIT_RECORD_FAILED",
                }
                for _ in GLOBAL_RANKS
            )
            abort = _distributed_abort(
                collective,
                transaction_directory=transaction_directory,
                transaction=transaction,
                phase=phase,
                rows=failure,
            )
            return ShadowFinalizeResult(False, True, False, False, None, abort)
        assert snapshot is not None
        _assert_authoritative_unchanged(
            snapshot,
            ordered_parameters=ordered_parameters,
            ordered_fixed_lora_a=ordered_fixed_lora_a,
        )
        commit, _ = _read_named_record(
            transaction_directory,
            decision["commit_file"],
            expected_sha256=decision["commit_sha256"],
        )
        return ShadowFinalizeResult(
            durable_commit_recorded=True,
            aborted=False,
            authoritative_parameters_mutated=False,
            publishable_shadow_authorized=decision[
                "publishable_shadow_authorized"
            ],
            commit_record=CanonicalFileRef(
                transaction_directory / decision["commit_file"],
                decision["commit_sha256"],
            ),
            receipt=commit,
        )
    except BaseException as error:
        unchanged = False
        if snapshot is not None:
            try:
                _assert_authoritative_unchanged(
                    snapshot,
                    ordered_parameters=ordered_parameters,
                    ordered_fixed_lora_a=ordered_fixed_lora_a,
                )
                unchanged = True
            except BaseException:
                try:
                    _restore_authoritative_snapshot(
                        snapshot,
                        ordered_parameters=ordered_parameters,
                        ordered_fixed_lora_a=ordered_fixed_lora_a,
                    )
                    unchanged = True
                except BaseException:
                    unchanged = False
        _write_failure_envelope(
            transaction_directory=transaction_directory,
            rank=getattr(collective, "rank", -1),
            phase=phase,
            error=error,
            snapshot=snapshot,
            authoritative_unchanged=unchanged,
        )
        raise World8FailStopError(
            "finalize lost safe distributed progress; authoritative A/B is unchanged"
        ) from error


def recover_world8_transaction(
    transaction_directory: Path, *, published_directory: Path | None = None
) -> Mapping[str, Any]:
    transaction_directory = Path(transaction_directory)
    transaction = _verify_transaction_directory(transaction_directory)
    if published_directory is not None:
        raise World8ShadowError(
            "published-directory recovery is disabled in quarantined v1"
        )
    forbidden = [
        name
        for name in ("COMMIT.json", "ABORTED.json", "PUBLISHED.json")
        if (transaction_directory / name).exists()
    ]
    if forbidden:
        raise World8ShadowError(
            f"legacy/forbidden terminal records exist: {forbidden}"
        )
    if (transaction_directory / "DECISION.json").exists():
        decision, digest = _read_named_record(
            transaction_directory, "DECISION.json"
        )
        if set(decision) != {
            "schema_version",
            "decision",
            "transaction_id",
            "transaction_init_sha256",
            "failure_phase",
            "failure_types",
            "authoritative_parameters_mutated",
            "shadow_published",
            "commit_and_publication_supported",
        } or (
            decision["schema_version"] != DECISION_SCHEMA
            or decision["decision"] != "ABORTED"
            or decision["transaction_id"] != transaction["transaction_id"]
            or decision["transaction_init_sha256"]
            != transaction["transaction_init_sha256"]
            or type(decision["failure_phase"]) is not str
            or not isinstance(decision["failure_types"], list)
            or not decision["failure_types"]
            or any(type(row) is not str for row in decision["failure_types"])
            or decision["failure_types"]
            != sorted(set(decision["failure_types"]))
            or decision["authoritative_parameters_mutated"] is not False
            or decision["shadow_published"] is not False
            or decision["commit_and_publication_supported"] is not False
        ):
            raise World8ShadowError("terminal DECISION record closure differs")
        return {
            "transaction_id": transaction["transaction_id"],
            "state": "ABORTED",
            "decision_sha256": digest,
            "authoritative_parameters_mutated_by_protocol": False,
            "finalize_disabled": True,
            "publication_disabled": True,
        }
    if (transaction_directory / "PREPARED.json").exists():
        _, digest = _read_validated_prepared_graph(
            transaction_directory=transaction_directory,
            transaction=transaction,
            snapshot=None,
        )
        return {
            "transaction_id": transaction["transaction_id"],
            "state": "PREPARED_QUARANTINED",
            "prepared_sha256": digest,
            "authoritative_parameters_mutated_by_protocol": False,
            "finalize_disabled": True,
            "publication_disabled": True,
        }
    present = sorted(
        path.name for path in transaction_directory.glob("rank-*.PREPARED.json")
    )
    if any(re.fullmatch(r"rank-[0-7]\.PREPARED\.json", name) is None for name in present):
        raise World8ShadowError("unexpected rank PREPARED filename exists")
    return {
        "transaction_id": transaction["transaction_id"],
        "state": "INCOMPLETE_RANK_PREPARE" if present else "INITIALIZED",
        "rank_wals_present": present,
        "authoritative_parameters_mutated_by_protocol": False,
        "finalize_disabled": True,
        "publication_disabled": True,
    }

    state = "INITIALIZED"
    details: dict[str, Any] = {}
    has_abort = (transaction_directory / "ABORTED.json").exists()
    has_commit = (transaction_directory / "COMMIT.json").exists()
    if has_abort and has_commit:
        raise World8ShadowError("transaction contains conflicting COMMIT and ABORT decisions")
    if has_abort:
        aborted, digest = _read_named_record(transaction_directory, "ABORTED.json")
        if aborted.get("schema_version") != ABORT_SCHEMA:
            raise World8ShadowError("ABORTED record schema differs")
        state = "ABORTED"
        details["aborted_sha256"] = digest
    elif has_commit:
        commit, digest = _read_named_record(transaction_directory, "COMMIT.json")
        if commit.get("schema_version") != COMMIT_SCHEMA:
            raise World8ShadowError("COMMIT record schema differs")
        _read_named_record(
            transaction_directory,
            commit["prepared_file"],
            expected_sha256=commit["prepared_sha256"],
        )
        _read_named_record(
            transaction_directory,
            commit["fresh_verification_file"],
            expected_sha256=commit["fresh_verification_sha256"],
        )
        state = "COMMITTED_UNPUBLISHED"
        details["commit_sha256"] = digest
        if published_directory is not None and Path(published_directory).exists():
            published, _ = _read_named_record(Path(published_directory), "PUBLISHED.json")
            if (
                published.get("schema_version") != PUBLISHED_SCHEMA
                or published.get("commit_sha256") != digest
            ):
                raise World8ShadowError("published directory belongs to another commit")
            state = "PUBLISHED"
    elif (transaction_directory / "PREPARED.json").exists():
        prepared, digest = _read_named_record(transaction_directory, "PREPARED.json")
        if prepared.get("schema_version") != GLOBAL_PREPARE_SCHEMA:
            raise World8ShadowError("PREPARED record schema differs")
        for expected_rank, wal in enumerate(prepared["rank_wals"]):
            if wal["global_rank"] != expected_rank:
                raise World8ShadowError("PREPARED WAL rank ordering differs")
            _read_named_record(
                transaction_directory,
                wal["wal_file"],
                expected_sha256=wal["wal_sha256"],
            )
        state = "PREPARED_AWAITING_FRESH"
        details["prepared_sha256"] = digest
    else:
        present = sorted(
            path.name for path in transaction_directory.glob("rank-*.PREPARED.json")
        )
        if present:
            state = "INCOMPLETE_RANK_PREPARE"
            details["rank_wals_present"] = present
    return {
        "transaction_id": transaction["transaction_id"],
        "state": state,
        "authoritative_parameters_mutated_by_protocol": False,
        **details,
    }


def publish_committed_shadow(
    *, transaction_directory: Path, published_directory: Path
) -> Mapping[str, Any]:
    """Fail closed: v1 never publishes a quarantined shadow adapter."""

    raise World8ShadowError(
        "publication is disabled in quarantined shadow-prepare-only v1"
    )

    transaction_directory = Path(transaction_directory)
    published_directory = Path(published_directory)
    recovery = recover_world8_transaction(transaction_directory)
    if recovery["state"] != "COMMITTED_UNPUBLISHED":
        if recovery["state"] == "PUBLISHED":
            published, _ = _read_named_record(published_directory, "PUBLISHED.json")
            return published
        raise World8ShadowError("publisher requires a durable COMMIT record")
    commit, commit_sha = _read_named_record(transaction_directory, "COMMIT.json")
    if published_directory.exists():
        published, _ = _read_named_record(published_directory, "PUBLISHED.json")
        if published.get("commit_sha256") != commit_sha:
            raise World8ShadowError("publish target belongs to another transaction")
        return published
    parent = published_directory.parent
    if not parent.is_dir():
        raise World8ShadowError("publish parent directory does not exist")
    staging = parent / f".{published_directory.name}.{commit_sha}.staging"
    try:
        os.mkdir(staging, 0o700)
        _fsync_directory(parent)
    except FileExistsError:
        if staging.is_symlink() or not staging.is_dir():
            raise World8ShadowError("publisher staging identity differs")
    shadow_manifest, _ = _read_named_record(
        transaction_directory,
        commit["shadow_manifest_file"],
        expected_sha256=commit["shadow_manifest_sha256"],
    )
    _verify_ab_artifact(
        transaction_directory=transaction_directory,
        artifact_file=shadow_manifest["shadow_ab_artifact_file"],
        artifact_sha256=shadow_manifest["shadow_ab_artifact_sha256"],
        entries=shadow_manifest["shadow_ab_entries"],
        expected_state_digest=shadow_manifest["shadow_ab_state_digest"],
    )
    files = {
        "shadow-manifest.json": (
            transaction_directory / commit["shadow_manifest_file"],
            commit["shadow_manifest_sha256"],
        ),
        "shadow-ab-state.fp32.bin": (
            transaction_directory / shadow_manifest["shadow_ab_artifact_file"],
            shadow_manifest["shadow_ab_artifact_sha256"],
        ),
        "candidate.json": (
            transaction_directory / commit["candidate_descriptor_file"],
            commit["candidate_descriptor_sha256"],
        ),
        "COMMIT.json": (transaction_directory / "COMMIT.json", commit_sha),
    }
    published_files = []
    for name, (source, digest) in files.items():
        raw = _read_stable_file(
            source, expected_sha256=digest, require_read_only=True
        )
        target_sha = _create_fsynced_file(staging / name, raw)
        published_files.append(
            {"file": name, "sha256": target_sha, "size": len(raw)}
        )
    publication = {
        "schema_version": PUBLISHED_SCHEMA,
        "transaction_id": commit["transaction_id"],
        "commit_sha256": commit_sha,
        "shadow_ab_state_digest": commit["shadow_ab_state_digest"],
        "files": published_files,
        "production_world8_transport": commit["production_world8_transport"],
        "scientifically_authorized": commit["publishable_shadow_authorized"],
        "authoritative_checkpoint_overwritten": False,
    }
    _create_fsynced_file(staging / "PUBLISHED.json", canonical_json_bytes(publication))
    _fsync_directory(staging)
    try:
        os.rename(staging, published_directory)
        _fsync_directory(parent)
    except OSError as error:
        if not published_directory.exists():
            raise World8ShadowError(f"atomic publish rename failed: {error}") from error
    published, _ = _read_named_record(published_directory, "PUBLISHED.json")
    if published != publication:
        raise World8ShadowError("published marker differs after atomic exposure")
    return published


__all__ = [
    "ARM_ORDER",
    "ARM_RANKS",
    "ArmScientificGateFiles",
    "CanonicalExternalVerifier",
    "CanonicalFileRef",
    "ExternalReceiptRef",
    "ExternalVerifierPolicy",
    "FreshInputFiles",
    "QuerySeedGateFiles",
    "ScientificGateFiles",
    "ShadowFinalizeResult",
    "ShadowPrepareResult",
    "TorchWorld8SmallCollective",
    "World8FailStopError",
    "World8ShadowError",
    "checkpoint_manifest_payload",
    "compute_anticipated_realized",
    "create_transaction_directory",
    "finalize_world8_shadow",
    "prepare_world8_shadow",
    "publish_committed_shadow",
    "recover_world8_transaction",
]

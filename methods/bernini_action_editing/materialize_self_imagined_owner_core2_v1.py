#!/usr/bin/env python3
"""Materialize audited per-query owner motion quotients for core2.

This is a fail-closed boundary between the pure-T2V owner process and the
editor.  It accepts the immutable, still-pending receipts produced by
``generate_self_imagined_owner_core2_v1.py`` plus a separately signed external
full-81-frame audit.  A pending receipt is never upgraded in place.

For each audited owner and each of its two pre-registered query seeds, the
owner clean latent is mixed with an independently sampled *official* Bernini
Gaussian at native schedule index 33.  The exact same ``x_sigma`` Python
object is queried in the fixed order action, reverse/wrong-family and
scene-matched no-op.  Only the detached, normalized, spatial-orderless
``Phi(H_action-H_noop)`` and a specificity receipt may be persisted.  RGB,
clean/noisy latents, Gaussian values, text embeddings, hidden states and
velocity never cross this boundary.

The tensor-only core is callback driven and dependency light for unit tests.
The AUH path reuses the STARC block-15 observer/sketch/all-reduce machinery;
it does not copy or persist full hidden states and constructs no optimizer.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import generate_self_imagined_owner_core2_v1 as owner_generation  # noqa: E402
import materialize_starc_core4_hidden_v1 as starc  # noqa: E402
import self_imagined_motion_cotangent_v1 as cotangent  # noqa: E402


AUDIT_SCHEMA = "bernini-self-imagined-owner-full81-audit-sidecar-v1"
CELL_RECEIPT_SCHEMA = "bernini-self-imagined-owner-quotient-specificity-v1"
MASTER_RECEIPT_SCHEMA = "bernini-self-imagined-owner-core2-specificity-master-v1"
AUDIT_JOB_ID = "131524"
AUDIT_SIGNATURE_SCHEME = "ed25519-canonical-json-v1"

QUOTIENT_FILENAME = "owner-motion-quotients.safetensors"
CELL_RECEIPT_FILENAME = "owner-motion-specificity-receipt.json"
MASTER_RECEIPT_FILENAME = "owner-core2-specificity-receipt.json"
TENSOR_KEY_PREFIX = "normalized_motion_quotient_query_seed_"
PROMPT_ORDER = ("action", "reverse_wrong_family", "common_scene_noop")
EXPECTED_WORLD_SIZE = 4
EXPECTED_VISIBLE_BY_CELL = dict(owner_generation.VISIBLE_GPUS_BY_CELL)

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")

_MASTER_FIELDS = (
    "schema_version",
    "probe_id",
    "registry_path",
    "registry_file_sha256",
    "topology",
    "cell_order",
    "children",
    "exact81_owner_count",
    "all8_used",
    "semantic_action_audit_complete",
    "owner_template_materialization_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
)
_MASTER_CHILD_FIELDS = (
    "cell_id",
    "receipt_path",
    "receipt_file_sha256",
    "receipt_digest",
    "mp4_path",
    "mp4_sha256",
)
_OWNER_CHILD_FIELDS = (
    "schema_version",
    "probe_id",
    "cell_id",
    "registry_path",
    "registry_file_sha256",
    "source_iid",
    "geometry_source_video_sha256",
    "geometry_source_role",
    "action_family_id",
    "action_caption_utf8_sha256",
    "owner_generation_seed",
    "native_receipt_path",
    "native_receipt_file_sha256",
    "native_receipt_digest",
    "bucket_hw",
    "latent_shape",
    "artifacts",
    "method_source_revision",
    "method_source_archive_sha256",
    "runtime_topology",
    "owner_source_condition_used",
    "owner_exact81_action_audit_status",
    "owner_template_materialization_authorized",
    "editor_condition_or_target_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
)
_AUDIT_FIELDS = (
    "schema_version",
    "owner_generation_job_id",
    "owner_master_binding",
    "audit_evidence_binding",
    "audit_authority_public_key_sha256",
    "authority_signature_scheme",
    "approval_scope",
    "cells",
    "receipt_digest",
    "authority_signature_ed25519_base64",
)
_AUDIT_MASTER_FIELDS = ("path", "file_sha256", "receipt_digest")
_AUDIT_EVIDENCE_FIELDS = ("path", "file_sha256")
_AUDIT_APPROVAL_FIELDS = (
    "decision",
    "semantic_action_audit_complete",
    "owner_template_materialization_authorized",
    "allowed_persistent_tensor_channel",
    "forbidden_owner_to_editor_channels",
    "optimizer_or_parameter_update_authorized",
)
_AUDIT_CELL_FIELDS = (
    "cell_id",
    "owner_child_receipt_file_sha256",
    "owner_child_receipt_digest",
    "owner_mp4_sha256",
    "action_family_id",
    "exact81_frame_count",
    "review_scope",
    "owner_exact81_action_audit_passed",
    "owner_source_condition_used",
    "materialize_template",
)
_CELL_PACKET_FIELDS = (
    "schema_version",
    "cell_id",
    "ordered_query_seeds",
    "owner_generation_seed",
    "owner_master_receipt_digest",
    "owner_child_receipt_digest",
    "external_full81_audit_sidecar_receipt_digest",
    "external_full81_audit_sidecar_file_sha256",
    "owner_exact81_action_audit_passed",
    "query_rows",
    "two_seed_audit",
    "quotient_artifact",
    "model_binding",
    "persistent_tensor_channel",
    "forbidden_owner_to_editor_channels",
    "owner_media_or_primal_tensor_persisted",
    "full_hidden_persisted",
    "seed_selection",
    "seed_averaging",
    "optimizer_constructed",
    "parameter_update_performed",
    "receipt_digest",
)


class OwnerTemplateMaterializationError(RuntimeError):
    """Raised before any unauthenticated or nonspecific template is published."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise OwnerTemplateMaterializationError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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
        raise OwnerTemplateMaterializationError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise OwnerTemplateMaterializationError(f"{label} must be lowercase SHA-256")
    return value


def _sha1(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise OwnerTemplateMaterializationError(f"{label} must be lowercase full SHA-1")
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise OwnerTemplateMaterializationError(f"{label} must be path-safe")
    return value


def _exact_fields(value: Any, fields: Iterable[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise OwnerTemplateMaterializationError(f"{label} field closure differs")
    return dict(value)


def _plain_file(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise OwnerTemplateMaterializationError(f"{label} must be an absolute plain file")
    return path.resolve(strict=True)


def _plain_dir(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise OwnerTemplateMaterializationError(
            f"{label} must be an absolute plain directory"
        )
    return path.resolve(strict=True)


def _fresh_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise OwnerTemplateMaterializationError(f"{label} must be fresh and absolute")
    parent = _plain_dir(path.parent, label=f"{label} parent")
    if path != parent / path.name or _SAFE_ID_RE.fullmatch(path.name) is None:
        raise OwnerTemplateMaterializationError(f"{label} path differs")
    return path


def _load_strict_json(
    value: str | Path, *, expected_sha256: Optional[str], label: str
) -> tuple[dict[str, Any], Path, str]:
    path = _plain_file(value, label=label)
    observed = file_sha256(path)
    if expected_sha256 is not None and observed != _sha256(
        expected_sha256, label=f"{label} expected SHA-256"
    ):
        raise OwnerTemplateMaterializationError(f"{label} SHA-256 differs")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise OwnerTemplateMaterializationError(
                    f"{label} contains duplicate key {key}"
                )
            result[key] = item
        return result

    try:
        decoded = json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                OwnerTemplateMaterializationError(
                    f"{label} contains forbidden non-finite {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OwnerTemplateMaterializationError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(decoded, dict):
        raise OwnerTemplateMaterializationError(f"{label} root differs")
    return decoded, path, observed


def _verify_seal(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop("receipt_digest", None)
    if declared != object_sha256(unsigned):
        raise OwnerTemplateMaterializationError(f"{label} receipt seal differs")
    return _sha256(declared, label=f"{label} receipt digest")


def _tensor_bytes(value: Any) -> bytes:
    """Return exact contiguous bytes on both old and current PyTorch."""

    import torch

    owned = value.detach().to(device="cpu").contiguous().clone()
    if hasattr(owned, "untyped_storage"):
        raw = bytes(owned.untyped_storage())
    else:  # Torch 1.12 used by the dependency-light local contract env.
        raw = owned.view(torch.uint8).numpy().tobytes(order="C")
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise OwnerTemplateMaterializationError("tensor raw-byte closure differs")
    return raw


def tensor_sha256(value: Any, *, label: str) -> str:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise OwnerTemplateMaterializationError(f"{label} must be detached and finite")
    metadata = {
        "dtype": str(value.dtype),
        "shape": list(map(int, value.shape)),
        "layout": str(value.layout),
    }
    return hashlib.sha256(canonical_json_bytes(metadata) + b"\x00" + _tensor_bytes(value)).hexdigest()


@dataclass(frozen=True)
class PendingOwnerGenerationInputs:
    """Authenticated job131524 proposal artifacts with *no* use authority."""

    registry: cotangent.ProbeRegistry
    registry_path: Path
    registry_file_sha256: str
    owner_root: Path
    master_path: Path
    master_file_sha256: str
    master_receipt_digest: str
    child_receipts: Mapping[str, Mapping[str, Any]]
    child_paths: Mapping[str, Path]
    child_file_sha256: Mapping[str, str]
    semantic_action_audit_complete: bool = False
    template_materialization_authorized: bool = False
    clean_latent_editor_input_authorized: bool = False

    def cell(self, cell_id: str) -> Mapping[str, Any]:
        if cell_id not in self.child_receipts:
            raise OwnerTemplateMaterializationError("pending owner cell differs")
        return self.child_receipts[cell_id]


@dataclass(frozen=True)
class AuthorizedOwnerInputs:
    registry: cotangent.ProbeRegistry
    registry_path: Path
    registry_file_sha256: str
    owner_root: Path
    master_path: Path
    master_file_sha256: str
    master_receipt_digest: str
    child_receipts: Mapping[str, Mapping[str, Any]]
    child_paths: Mapping[str, Path]
    child_file_sha256: Mapping[str, str]
    audit_sidecar_path: Path
    audit_sidecar_file_sha256: str
    audit_sidecar_receipt_digest: str
    audit_evidence_file_sha256: str
    audit_public_key_file_sha256: str

    def cell(self, cell_id: str) -> Mapping[str, Any]:
        if cell_id not in self.child_receipts:
            raise OwnerTemplateMaterializationError("authorized owner cell differs")
        return self.child_receipts[cell_id]


def _validate_owner_artifact(
    artifact: Any, *, child_root: Path, label: str
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise OwnerTemplateMaterializationError(f"{label} binding differs")
    path = _plain_file(artifact.get("path", ""), label=label)
    try:
        path.relative_to(child_root)
    except ValueError as error:
        raise OwnerTemplateMaterializationError(f"{label} escaped owner child root") from error
    expected = _sha256(artifact.get("sha256"), label=f"{label} SHA-256")
    if file_sha256(path) != expected:
        raise OwnerTemplateMaterializationError(f"{label} bytes changed")
    return dict(artifact)


def _load_owner_master_and_children(
    *,
    owner_root: str | Path,
    owner_master_receipt: str | Path,
    expected_owner_master_receipt_sha256: str,
    registry_path: Path,
    registry_file_sha256: str,
    registry: cotangent.ProbeRegistry,
) -> tuple[dict[str, Any], Path, str, dict[str, dict[str, Any]], dict[str, Path], dict[str, str]]:
    root = _plain_dir(owner_root, label="owner generation root")
    master, master_path, master_sha = _load_strict_json(
        owner_master_receipt,
        expected_sha256=expected_owner_master_receipt_sha256,
        label="owner master receipt",
    )
    if master_path != root / owner_generation.MASTER_RECEIPT_BASENAME:
        raise OwnerTemplateMaterializationError("owner master path differs")
    row = _exact_fields(master, _MASTER_FIELDS, label="owner master receipt")
    master_digest = _verify_seal(row, label="owner master receipt")
    if (
        row["schema_version"] != owner_generation.MASTER_SCHEMA_VERSION
        or row["probe_id"] != registry.probe_id
        or row["registry_file_sha256"] != registry_file_sha256
        or row["topology"] != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or row["cell_order"] != list(owner_generation.CELL_IDS)
        or row["exact81_owner_count"] != 2
        or row["all8_used"] is not True
        or row["semantic_action_audit_complete"] is not False
        or row["owner_template_materialization_authorized"] is not False
        or row["optimizer_or_parameter_update_authorized"] is not False
    ):
        raise OwnerTemplateMaterializationError(
            "owner master must remain the original pending, non-authorizing receipt"
        )
    children = row["children"]
    if not isinstance(children, list) or len(children) != 2:
        raise OwnerTemplateMaterializationError("owner master child closure differs")
    child_values: dict[str, dict[str, Any]] = {}
    child_paths: dict[str, Path] = {}
    child_hashes: dict[str, str] = {}
    for expected_cell, raw_binding in zip(owner_generation.CELL_IDS, children):
        binding = _exact_fields(
            raw_binding, _MASTER_CHILD_FIELDS, label=f"{expected_cell} master child binding"
        )
        if binding["cell_id"] != expected_cell:
            raise OwnerTemplateMaterializationError("owner master child order differs")
        child_path_expected = root / expected_cell / owner_generation.OWNER_RECEIPT_BASENAME
        child, child_path, child_sha = _load_strict_json(
            binding["receipt_path"],
            expected_sha256=binding["receipt_file_sha256"],
            label=f"{expected_cell} owner child receipt",
        )
        if child_path != child_path_expected:
            raise OwnerTemplateMaterializationError(f"{expected_cell} child path differs")
        checked = _exact_fields(child, _OWNER_CHILD_FIELDS, label=f"{expected_cell} owner child")
        child_digest = _verify_seal(checked, label=f"{expected_cell} owner child")
        cell = registry.cell(expected_cell)
        if (
            child_sha != binding["receipt_file_sha256"]
            or child_digest != binding["receipt_digest"]
            or checked["schema_version"] != owner_generation.SCHEMA_VERSION
            or checked["probe_id"] != registry.probe_id
            or checked["cell_id"] != expected_cell
            or checked["registry_file_sha256"] != registry_file_sha256
            or checked["source_iid"] != cell.source_iid
            or checked["action_family_id"] != cell.action_family_id
            or checked["action_caption_utf8_sha256"] != cell.action_caption_utf8_sha256
            or checked["owner_generation_seed"] != cell.owner_generation_seed
            or tuple(checked["latent_shape"]) != tuple(cell.latent_shape)
            or checked["geometry_source_role"]
            != "bucket_shape_only_never_transformer_condition"
            or checked["owner_source_condition_used"] is not False
            or checked["owner_exact81_action_audit_status"]
            != "pending_detached_full_video_review"
            or checked["owner_template_materialization_authorized"] is not False
            or checked["editor_condition_or_target_authorized"] is not False
            or checked["optimizer_or_parameter_update_authorized"] is not False
            or checked["runtime_topology"]
            != {
                "world_size": 4,
                "ulysses_size": 4,
                "rocr_visible_devices": EXPECTED_VISIBLE_BY_CELL[expected_cell],
            }
        ):
            raise OwnerTemplateMaterializationError(
                f"{expected_cell} pending owner receipt binding differs"
            )
        artifacts = checked["artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "mp4",
            "predecode_clean_latent",
            "official_initial_gaussian",
        }:
            raise OwnerTemplateMaterializationError(f"{expected_cell} artifact closure differs")
        child_root = child_path.parent
        verified_artifacts = {
            key: _validate_owner_artifact(
                artifacts[key], child_root=child_root, label=f"{expected_cell} {key}"
            )
            for key in artifacts
        }
        checked["artifacts"] = verified_artifacts
        if (
            verified_artifacts["mp4"]["path"] != binding["mp4_path"]
            or verified_artifacts["mp4"]["sha256"] != binding["mp4_sha256"]
        ):
            raise OwnerTemplateMaterializationError(f"{expected_cell} MP4 binding differs")
        native_path = _plain_file(
            checked["native_receipt_path"], label=f"{expected_cell} native receipt"
        )
        if (
            file_sha256(native_path) != checked["native_receipt_file_sha256"]
            or native_path.parent != child_root
        ):
            raise OwnerTemplateMaterializationError(f"{expected_cell} native receipt changed")
        child_values[expected_cell] = checked
        child_paths[expected_cell] = child_path
        child_hashes[expected_cell] = child_sha
    return row, master_path, master_sha, child_values, child_paths, child_hashes


def load_pending_owner_generation_inputs(
    *,
    registry: str | Path,
    expected_registry_sha256: str,
    owner_root: str | Path,
    owner_master_receipt: str | Path,
    expected_owner_master_receipt_sha256: str,
) -> PendingOwnerGenerationInputs:
    """Authenticate immutable pending owner artifacts without authorizing use.

    This read-only loader is suitable for isolated frozen diagnostics.  Its
    return type deliberately carries three explicit false authority bits; it
    does not consume the signed full81 audit and cannot authorize template
    publication, an editor condition/target, or a parameter update.
    """

    registry_path = _plain_file(registry, label="core2 registry")
    registry_sha = file_sha256(registry_path)
    if registry_sha != _sha256(expected_registry_sha256, label="registry SHA-256"):
        raise OwnerTemplateMaterializationError("core2 registry bytes changed")
    try:
        registry_value = cotangent.load_probe_registry(
            registry_path, expected_file_sha256=registry_sha
        )
    except cotangent.SelfImaginedCotangentContractError as error:
        raise OwnerTemplateMaterializationError(str(error)) from error
    (
        master,
        master_path,
        master_sha,
        children,
        child_paths,
        child_hashes,
    ) = _load_owner_master_and_children(
        owner_root=owner_root,
        owner_master_receipt=owner_master_receipt,
        expected_owner_master_receipt_sha256=expected_owner_master_receipt_sha256,
        registry_path=registry_path,
        registry_file_sha256=registry_sha,
        registry=registry_value,
    )
    return PendingOwnerGenerationInputs(
        registry=registry_value,
        registry_path=registry_path,
        registry_file_sha256=registry_sha,
        owner_root=_plain_dir(owner_root, label="owner generation root"),
        master_path=master_path,
        master_file_sha256=master_sha,
        master_receipt_digest=master["receipt_digest"],
        child_receipts=MappingProxyType(children),
        child_paths=MappingProxyType(child_paths),
        child_file_sha256=MappingProxyType(child_hashes),
    )


def _verify_audit_signature(
    sidecar: Mapping[str, Any], *, public_key_path: Path
) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise OwnerTemplateMaterializationError(
            "cryptography with Ed25519 support is required for audit verification"
        ) from error
    signed = dict(sidecar)
    encoded = signed.pop("authority_signature_ed25519_base64", None)
    if not isinstance(encoded, str):
        raise OwnerTemplateMaterializationError("audit Ed25519 signature differs")
    try:
        signature = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise OwnerTemplateMaterializationError("audit Ed25519 signature encoding differs") from error
    if len(signature) != 64:
        raise OwnerTemplateMaterializationError("audit Ed25519 signature length differs")
    try:
        key = serialization.load_pem_public_key(public_key_path.read_bytes())
    except Exception as error:
        raise OwnerTemplateMaterializationError("audit public key PEM differs") from error
    if not isinstance(key, Ed25519PublicKey):
        raise OwnerTemplateMaterializationError("audit public key is not Ed25519")
    try:
        key.verify(signature, canonical_json_bytes(signed))
    except InvalidSignature as error:
        raise OwnerTemplateMaterializationError("audit sidecar signature verification failed") from error


def load_authorized_owner_inputs(
    *,
    registry: str | Path,
    expected_registry_sha256: str,
    owner_root: str | Path,
    owner_master_receipt: str | Path,
    expected_owner_master_receipt_sha256: str,
    audit_sidecar: Optional[str | Path],
    expected_audit_sidecar_sha256: Optional[str],
    audit_evidence: Optional[str | Path],
    audit_public_key: Optional[str | Path],
    expected_audit_public_key_sha256: Optional[str],
) -> AuthorizedOwnerInputs:
    """Authenticate pending receipts and the separate signed approval overlay.

    Every audit argument is mandatory.  Therefore the default state of the
    job-131524 owner receipts remains pending and materialization cannot start.
    """

    pending = load_pending_owner_generation_inputs(
        registry=registry,
        expected_registry_sha256=expected_registry_sha256,
        owner_root=owner_root,
        owner_master_receipt=owner_master_receipt,
        expected_owner_master_receipt_sha256=expected_owner_master_receipt_sha256,
    )
    registry_path = pending.registry_path
    registry_sha = pending.registry_file_sha256
    registry_value = pending.registry
    master_path = pending.master_path
    master_sha = pending.master_file_sha256
    children = dict(pending.child_receipts)
    child_paths = dict(pending.child_paths)
    child_hashes = dict(pending.child_file_sha256)
    if any(
        value is None
        for value in (
            audit_sidecar,
            expected_audit_sidecar_sha256,
            audit_evidence,
            audit_public_key,
            expected_audit_public_key_sha256,
        )
    ):
        raise OwnerTemplateMaterializationError(
            "pending owner receipts require a signed external full81 audit sidecar"
        )
    public_key_path = _plain_file(audit_public_key, label="audit public key")  # type: ignore[arg-type]
    public_key_sha = file_sha256(public_key_path)
    if public_key_sha != _sha256(
        expected_audit_public_key_sha256, label="audit public key SHA-256"  # type: ignore[arg-type]
    ):
        raise OwnerTemplateMaterializationError("audit public key bytes changed")
    evidence_path = _plain_file(audit_evidence, label="external full81 audit evidence")  # type: ignore[arg-type]
    evidence_sha = file_sha256(evidence_path)
    sidecar, sidecar_path, sidecar_sha = _load_strict_json(
        audit_sidecar,  # type: ignore[arg-type]
        expected_sha256=expected_audit_sidecar_sha256,
        label="external full81 audit sidecar",
    )
    row = _exact_fields(sidecar, _AUDIT_FIELDS, label="external full81 audit sidecar")
    _verify_audit_signature(row, public_key_path=public_key_path)
    signed = dict(row)
    signed.pop("authority_signature_ed25519_base64")
    declared_digest = signed.pop("receipt_digest", None)
    if declared_digest != object_sha256(signed):
        raise OwnerTemplateMaterializationError("external audit sidecar seal differs")
    audit_digest = _sha256(declared_digest, label="audit sidecar receipt digest")
    master_binding = _exact_fields(
        row["owner_master_binding"], _AUDIT_MASTER_FIELDS, label="audit master binding"
    )
    evidence_binding = _exact_fields(
        row["audit_evidence_binding"], _AUDIT_EVIDENCE_FIELDS, label="audit evidence binding"
    )
    approval = _exact_fields(
        row["approval_scope"], _AUDIT_APPROVAL_FIELDS, label="audit approval scope"
    )
    if (
        row["schema_version"] != AUDIT_SCHEMA
        or str(row["owner_generation_job_id"]) != AUDIT_JOB_ID
        or row["audit_authority_public_key_sha256"] != public_key_sha
        or row["authority_signature_scheme"] != AUDIT_SIGNATURE_SCHEME
        or master_binding
        != {
            "path": str(master_path),
            "file_sha256": master_sha,
            "receipt_digest": pending.master_receipt_digest,
        }
        or evidence_binding
        != {"path": str(evidence_path), "file_sha256": evidence_sha}
        or approval
        != {
            "decision": "approve_owner_template_materialization",
            "semantic_action_audit_complete": True,
            "owner_template_materialization_authorized": True,
            "allowed_persistent_tensor_channel": cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
            "forbidden_owner_to_editor_channels": list(
                cotangent.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS
            ),
            "optimizer_or_parameter_update_authorized": False,
        }
    ):
        raise OwnerTemplateMaterializationError("external audit authority differs")
    audit_cells = row["cells"]
    if not isinstance(audit_cells, list) or len(audit_cells) != 2:
        raise OwnerTemplateMaterializationError("external audit cell closure differs")
    for expected_cell, raw in zip(owner_generation.CELL_IDS, audit_cells):
        audited = _exact_fields(raw, _AUDIT_CELL_FIELDS, label=f"{expected_cell} audit")
        child = children[expected_cell]
        spec = registry_value.cell(expected_cell)
        if audited != {
            "cell_id": expected_cell,
            "owner_child_receipt_file_sha256": child_hashes[expected_cell],
            "owner_child_receipt_digest": child["receipt_digest"],
            "owner_mp4_sha256": child["artifacts"]["mp4"]["sha256"],
            "action_family_id": spec.action_family_id,
            "exact81_frame_count": 81,
            "review_scope": "all_81_frames_start_transition_terminal_hold",
            "owner_exact81_action_audit_passed": True,
            "owner_source_condition_used": False,
            "materialize_template": True,
        }:
            raise OwnerTemplateMaterializationError(
                f"{expected_cell} owner failed or escaped the full81 action audit"
            )
    return AuthorizedOwnerInputs(
        registry=registry_value,
        registry_path=registry_path,
        registry_file_sha256=registry_sha,
        owner_root=_plain_dir(owner_root, label="owner generation root"),
        master_path=master_path,
        master_file_sha256=master_sha,
        master_receipt_digest=pending.master_receipt_digest,
        child_receipts=MappingProxyType(children),
        child_paths=MappingProxyType(child_paths),
        child_file_sha256=MappingProxyType(child_hashes),
        audit_sidecar_path=sidecar_path,
        audit_sidecar_file_sha256=sidecar_sha,
        audit_sidecar_receipt_digest=audit_digest,
        audit_evidence_file_sha256=evidence_sha,
        audit_public_key_file_sha256=public_key_sha,
    )


@dataclass(frozen=True)
class SeedTemplateMaterialization:
    query_seed: int
    template: cotangent.FrozenOwnerTemplate
    specificity: cotangent.PromptSpecificityAudit
    official_gaussian_tensor_digest: str
    same_x_sigma_binding_digest: str
    owner_spatial_coordinates: int
    hidden_forward_proof: Mapping[str, Any]

    def receipt(self) -> dict[str, Any]:
        return {
            "query_seed": self.query_seed,
            "template": self.template.receipt(),
            "prompt_specificity": self.specificity.receipt(),
            "official_gaussian_scheme": "diffusers.randn_tensor_cpu_generator_fp32_v1",
            "official_gaussian_tensor_digest": self.official_gaussian_tensor_digest,
            "same_x_sigma_binding_digest": self.same_x_sigma_binding_digest,
            "owner_spatial_coordinates": self.owner_spatial_coordinates,
            "same_x_sigma_object_prompt_order": list(PROMPT_ORDER),
            "same_x_sigma_object_for_all_three_prompts": True,
            "hidden_forward_proof": dict(self.hidden_forward_proof),
        }


@dataclass(frozen=True)
class CellTemplateMaterialization:
    cell_id: str
    ordered_query_seeds: tuple[int, int]
    rows: tuple[SeedTemplateMaterialization, SeedTemplateMaterialization]
    two_seed_audit: cotangent.TwoSeedTemplateAudit
    production_input_binding: Optional[Mapping[str, Any]] = None

    def tensors(self) -> dict[str, Any]:
        return {
            f"{TENSOR_KEY_PREFIX}{row.query_seed}": row.template.unit_feature.detach()
            .to(device="cpu", dtype=__import__("torch").float32)
            .contiguous()
            for row in self.rows
        }


def official_gaussian_for_query_seed(
    query_seed: int, *, shape: Sequence[int], device: Any
) -> Any:
    """Reproduce Bernini's official CPU-generator ``randn_tensor`` path."""

    import torch
    from diffusers.utils.torch_utils import randn_tensor

    if type(query_seed) is not int or query_seed < 0:
        raise OwnerTemplateMaterializationError("query seed differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(query_seed)
    value = randn_tensor(
        tuple(map(int, shape)),
        generator=generator,
        device=torch.device(device),
        dtype=torch.float32,
    ).float().contiguous().detach()
    if value.requires_grad or value.grad_fn is not None:
        raise OwnerTemplateMaterializationError("official Gaussian did not detach")
    return value


def _materialize_cell_from_hidden_callback_unsafe(
    *,
    cell: cotangent.ProbeCellSpec,
    owner_clean_latent: Any,
    hidden_forward: Callable[..., Any],
    gaussian_factory: Callable[..., Any] = official_gaussian_for_query_seed,
    sigma: float = cotangent.NATIVE_SIGMA,
    specificity_margin: float,
    minimum_template_cosine: float,
    production_input_binding: Optional[Mapping[str, Any]] = None,
) -> CellTemplateMaterialization:
    """Low-level tensor test helper; its result has no publication authority.

    Only ``materialize_cell`` may add a production input binding, and the
    private persistent boundary independently reloads that clean artifact and
    requires the runtime-owned frozen triplet proof.  Callers must not treat a
    callback result as an authenticated template packet.
    """

    import torch

    if not isinstance(cell, cotangent.ProbeCellSpec):
        raise OwnerTemplateMaterializationError("cell specification differs")
    clean = owner_clean_latent
    if (
        not isinstance(clean, torch.Tensor)
        or clean.ndim != 5
        or tuple(int(item) for item in clean.shape[:3]) != (1, 16, 21)
        or clean.dtype != torch.float32
        or clean.device.type == "meta"
        or clean.requires_grad
        or clean.grad_fn is not None
        or not bool(torch.isfinite(clean).all().item())
    ):
        raise OwnerTemplateMaterializationError(
            "owner clean latent must be detached finite exact81 FP32"
        )
    if not math.isfinite(float(sigma)) or float(sigma).hex() != float(
        cotangent.NATIVE_SIGMA
    ).hex():
        raise OwnerTemplateMaterializationError("native schedule-33 sigma differs")
    if tuple(cell.query_seeds) != tuple(dict.fromkeys(cell.query_seeds)) or len(
        cell.query_seeds
    ) != 2:
        raise OwnerTemplateMaterializationError("two fixed query seeds differ")
    rows: list[SeedTemplateMaterialization] = []
    gaussian_digests: list[str] = []
    x_digests: list[str] = []
    for query_seed in cell.query_seeds:
        epsilon = gaussian_factory(
            query_seed, shape=tuple(int(item) for item in clean.shape), device=clean.device
        )
        if (
            not isinstance(epsilon, torch.Tensor)
            or epsilon.shape != clean.shape
            or epsilon.dtype != torch.float32
            or epsilon.device != clean.device
            or epsilon.requires_grad
            or epsilon.grad_fn is not None
            or not bool(torch.isfinite(epsilon).all().item())
        ):
            raise OwnerTemplateMaterializationError("official query Gaussian closure differs")
        gaussian_digest = tensor_sha256(epsilon, label="official query Gaussian")
        x_sigma = (
            clean + torch.tensor(float(sigma), dtype=torch.float32, device=clean.device)
            * (epsilon - clean)
        ).float().contiguous().detach()
        x_digest = tensor_sha256(x_sigma, label="owner x_sigma")
        x_object_id = id(x_sigma)
        hidden_by_role: dict[str, Any] = {}
        proofs: list[Mapping[str, Any]] = []
        for role, caption in (
            ("action", cell.action_caption),
            ("reverse_wrong_family", cell.reverse_wrong_family_caption),
            ("common_scene_noop", cell.noop_caption),
        ):
            before = tensor_sha256(x_sigma, label=f"{role} x_sigma before")
            result = hidden_forward(
                x_sigma=x_sigma,
                prompt_role=role,
                prompt_caption=caption,
                query_seed=query_seed,
            )
            if isinstance(result, tuple) and len(result) == 2:
                hidden, proof = result
            else:
                hidden, proof = result, {}
            if id(x_sigma) != x_object_id or before != tensor_sha256(
                x_sigma, label=f"{role} x_sigma after"
            ):
                raise OwnerTemplateMaterializationError(
                    "same x_sigma object/value changed across prompt triplet"
                )
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 4
                or int(hidden.shape[0]) != 1
                or int(hidden.shape[1]) != cotangent.LATENT_PHASES
                or int(hidden.shape[2]) <= 0
                or int(hidden.shape[3]) != cotangent.HIDDEN_SIZE
                or hidden.dtype != torch.float32
                or hidden.device.type == "meta"
                or hidden.requires_grad
                or hidden.grad_fn is not None
                or not bool(torch.isfinite(hidden).all().item())
            ):
                raise OwnerTemplateMaterializationError(f"{role} hidden sketch differs")
            hidden_by_role[role] = hidden.detach()
            if not isinstance(proof, Mapping):
                raise OwnerTemplateMaterializationError("hidden forward proof differs")
            proofs.append(dict(proof))
        shapes = {tuple(value.shape) for value in hidden_by_role.values()}
        if len(shapes) != 1:
            raise OwnerTemplateMaterializationError("same-seed prompt hidden geometry differs")
        action_residual = (
            hidden_by_role["action"] - hidden_by_role["common_scene_noop"]
        ).float().contiguous().detach()
        reverse_residual = (
            hidden_by_role["reverse_wrong_family"]
            - hidden_by_role["common_scene_noop"]
        ).float().contiguous().detach()
        common_null = (
            hidden_by_role["common_scene_noop"]
            - hidden_by_role["common_scene_noop"]
        ).float().contiguous().detach()
        provenance = {
            "cell_id": cell.cell_id,
            "owner_generation_seed": cell.owner_generation_seed,
            "query_seed": query_seed,
            "owner_mode": "frozen_bernini_pure_t2v",
            "owner_exact81_action_audit_passed": True,
            "owner_used_source_video_condition": False,
        }
        try:
            template = cotangent.build_frozen_owner_template(
                action_residual,
                query_seed=query_seed,
                owner_provenance=provenance,
            )
            same_x_binding = object_sha256(
                {
                    "cell_id": cell.cell_id,
                    "query_seed": query_seed,
                    "native_schedule_index": cotangent.SCHEDULE_INDEX,
                    "x_sigma_tensor_digest": x_digest,
                    "prompt_order": list(PROMPT_ORDER),
                    "same_object": True,
                }
            )
            specificity = cotangent.audit_prompt_specificity(
                template,
                action_residual=action_residual,
                reverse_wrong_family_residual=reverse_residual,
                common_scene_null_residual=common_null,
                same_x_sigma_binding_digest=same_x_binding,
                minimum_margin=specificity_margin,
            )
        except cotangent.SelfImaginedCotangentContractError as error:
            raise OwnerTemplateMaterializationError(str(error)) from error
        if not specificity.passed:
            raise OwnerTemplateMaterializationError(
                f"{cell.cell_id} query seed {query_seed} failed prompt specificity"
            )
        merged_proof = {
            "prompt_order": list(PROMPT_ORDER),
            "same_x_sigma_object": True,
            "x_sigma_unchanged": True,
            "forward_receipts": proofs,
            "full_hidden_persisted": False,
        }
        rows.append(
            SeedTemplateMaterialization(
                query_seed=query_seed,
                template=template,
                specificity=specificity,
                official_gaussian_tensor_digest=gaussian_digest,
                same_x_sigma_binding_digest=same_x_binding,
                owner_spatial_coordinates=int(action_residual.shape[2]),
                hidden_forward_proof=MappingProxyType(merged_proof),
            )
        )
        gaussian_digests.append(gaussian_digest)
        x_digests.append(x_digest)
        # No owner primal is retained in a result object.
        del epsilon, x_sigma, hidden_by_role, action_residual, reverse_residual, common_null
    if len(set(gaussian_digests)) != 2 or len(set(x_digests)) != 2:
        raise OwnerTemplateMaterializationError(
            "the two fixed query seeds must use independent Gaussian/x_sigma values"
        )
    try:
        two_seed = cotangent.audit_two_seed_templates(
            [row.template for row in rows], minimum_cosine=minimum_template_cosine
        )
    except cotangent.SelfImaginedCotangentContractError as error:
        raise OwnerTemplateMaterializationError(str(error)) from error
    if not two_seed.passed:
        raise OwnerTemplateMaterializationError(
            f"{cell.cell_id} two-seed owner-template audit failed"
        )
    checked_production_binding: Optional[Mapping[str, Any]] = None
    if production_input_binding is not None:
        checked = _exact_fields(
            production_input_binding,
            (
                "owner_child_receipt_digest",
                "external_full81_audit_sidecar_receipt_digest",
                "owner_clean_latent_file_sha256",
                "owner_clean_latent_tensor_digest",
            ),
            label="production owner input binding",
        )
        if (
            checked["owner_clean_latent_tensor_digest"]
            != tensor_sha256(clean, label="bound owner clean latent")
            or any(
                not isinstance(checked[name], str)
                or _SHA256_RE.fullmatch(checked[name]) is None
                for name in checked
            )
        ):
            raise OwnerTemplateMaterializationError(
                "production owner input binding differs"
            )
        checked_production_binding = MappingProxyType(checked)
    return CellTemplateMaterialization(
        cell_id=cell.cell_id,
        ordered_query_seeds=tuple(cell.query_seeds),
        rows=(rows[0], rows[1]),
        two_seed_audit=two_seed,
        production_input_binding=checked_production_binding,
    )


def _snapshot_tensors(values: Mapping[str, Any]) -> dict[str, str]:
    return {name: starc.tensor_sha256(value) for name, value in values.items()}


def forward_same_state_hidden_triplet(
    *,
    diffusion: Any,
    transformer: Any,
    observer: starc.Block15SpatialSketchObserver,
    x_sigma: Any,
    prompt_conditions: Mapping[str, Any],
    query_key: str,
    dist_module: Any = None,
    group: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Three-prompt sibling of STARC's pair primitive, patching state once."""

    import torch
    import dclr_runtime_contract as runtime_contract
    import pair_v5_native_bridge as native_bridge

    dist = torch.distributed if dist_module is None else dist_module
    try:
        native_bridge._validate_exact81_spatial(
            x_sigma, label="owner x_sigma", detached_fp32=True
        )
    except native_bridge.PairV5NativeBridgeError as error:
        raise OwnerTemplateMaterializationError(str(error)) from error
    x_shape, patch_height, patch_width, patch_positions = starc.latent_geometry(x_sigma)
    if (
        observer.layout.patch_height != patch_height
        or observer.layout.patch_width != patch_width
        or observer.layout.patch_positions != patch_positions
    ):
        raise OwnerTemplateMaterializationError("observer/query geometry differs")
    if (
        set(prompt_conditions) != set(PROMPT_ORDER)
        or not callable(getattr(diffusion, "shared_step", None))
        or not callable(getattr(transformer, "patch_vae_latent", None))
        or any(parameter.requires_grad for parameter in diffusion.parameters())
        or any(parameter.requires_grad for parameter in transformer.parameters())
    ):
        raise OwnerTemplateMaterializationError("frozen triplet runtime differs")
    dtype = getattr(transformer, "dtype", None)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise OwnerTemplateMaterializationError("transformer dtype differs")
    condition_hashes = []
    for role in PROMPT_ORDER:
        condition = prompt_conditions[role]
        if (
            not isinstance(condition, torch.Tensor)
            or tuple(int(item) for item in condition.shape) != (1, 512, 4096)
            or condition.device != x_sigma.device
            or condition.requires_grad
            or condition.grad_fn is not None
            or not bool(torch.isfinite(condition).all().item())
        ):
            raise OwnerTemplateMaterializationError(f"{role} condition closure differs")
        condition_hashes.append(starc.tensor_sha256(condition.float()))
    if len(set(condition_hashes)) != 3:
        raise OwnerTemplateMaterializationError("triplet prompt conditions alias")
    with torch.inference_mode():
        patched = transformer.patch_vae_latent(
            x_sigma.to(dtype=dtype), source_id=native_bridge.T2V_TARGET_SOURCE_ID
        )
    if not isinstance(patched, (tuple, list)) or len(patched) != 2:
        raise OwnerTemplateMaterializationError("patch_vae_latent output differs")
    try:
        branch = runtime_contract.build_t2v_target_branch(
            patched[0], patched[1], target_source_id=native_bridge.T2V_TARGET_SOURCE_ID
        )
    except runtime_contract.DCLRRuntimeContractError as error:
        raise OwnerTemplateMaterializationError(str(error)) from error
    if branch.target_token_count != cotangent.LATENT_PHASES * patch_positions:
        raise OwnerTemplateMaterializationError("target suffix token geometry differs")
    timestep = torch.tensor(
        [float(cotangent.NATIVE_TIMESTEP)], dtype=torch.float32, device=x_sigma.device
    )
    tracked = {
        "x_sigma": x_sigma,
        "noisy_latents": branch.noisy_latents,
        "rotary_embs": branch.rotary_embs,
        "native_timestep": timestep,
    }
    object_ids = {name: id(value) for name, value in tracked.items()}
    before = _snapshot_tensors(tracked)
    captures: dict[str, Any] = {}
    local_parity: dict[str, tuple[str, str]] = {}
    for role in PROMPT_ORDER:
        with observer.capture(f"{query_key}:{role}") as holder:
            with torch.inference_mode():
                prediction = diffusion.shared_step(
                    model_id="transformer_1",
                    noisy_latents=branch.noisy_latents,
                    timesteps=timestep,
                    cond_embeds=prompt_conditions[role],
                    rotary_embs=branch.rotary_embs,
                    batch_vae_seqlen=list(branch.batch_vae_seqlen),
                    batch_text_seqlen=[runtime_contract.PINNED_TEXT_TOKENS],
                )
        if len(holder) != 1:
            raise OwnerTemplateMaterializationError("block15 capture closure differs")
        total = branch.total_token_count
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(int(item) for item in prediction.shape)
            != (1, total, runtime_contract.PINNED_PATCH_DIM)
            or prediction.requires_grad
            or prediction.grad_fn is not None
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise OwnerTemplateMaterializationError("frozen prediction closure differs")
        local = holder[0]
        global_capture = starc.all_reduce_block15_sketch(
            local, dist_module=dist, group=group
        )
        captures[role] = global_capture.sketch.float().contiguous().detach()
        local_parity[role] = (
            local.block0_input_value_sha256,
            local.block0_attn1_value_sha256,
        )
        if any(id(tracked[name]) != object_ids[name] for name in tracked):
            raise OwnerTemplateMaterializationError("shared state object identity changed")
        if _snapshot_tensors(tracked) != before:
            raise OwnerTemplateMaterializationError("shared state bytes changed")
        del prediction
    if len(set(local_parity.values())) != 1:
        raise OwnerTemplateMaterializationError("block0 prompt parity failed locally")
    parity_rows: list[Any] = [None] * EXPECTED_WORLD_SIZE
    dist.all_gather_object(
        parity_rows,
        {"rank": observer.layout.sp_rank, "block0_pair": list(next(iter(local_parity.values())))},
        group=group,
    )
    if [row.get("rank") for row in parity_rows] != list(range(EXPECTED_WORLD_SIZE)):
        raise OwnerTemplateMaterializationError("SP4 parity rank closure differs")
    for role, value in captures.items():
        if tuple(int(item) for item in value.shape) != (
            1,
            cotangent.LATENT_PHASES,
            starc.SKETCH_COORDINATES,
            cotangent.HIDDEN_SIZE,
        ):
            raise OwnerTemplateMaterializationError(f"{role} global sketch shape differs")
        digests: list[Any] = [None] * EXPECTED_WORLD_SIZE
        dist.all_gather_object(digests, starc.tensor_sha256(value), group=group)
        if len(set(digests)) != 1:
            raise OwnerTemplateMaterializationError(f"{role} SP4 sketch differs by rank")
    proof = {
        "native_schedule_index": cotangent.SCHEDULE_INDEX,
        "sigma": cotangent.NATIVE_SIGMA,
        "native_timestep": cotangent.NATIVE_TIMESTEP,
        "prompt_order": list(PROMPT_ORDER),
        "same_x_sigma_noisy_rotary_timestep_objects": True,
        "shared_tensor_bytes_unchanged": True,
        "block0_input_and_attn1_exact_parity": True,
        "target_suffix_only": True,
        "source_condition_consumed": False,
        "mask_flow_pose_track_or_trajectory_consumed": False,
        "full_hidden_persisted": False,
        "x_sigma_binding_digest": object_sha256(before),
        "latent_shape": list(x_shape),
        "patch_grid_height_width": [patch_height, patch_width],
        "patch_positions": patch_positions,
    }
    return captures, proof


def _encode_prompt_triplet(
    renderer: Any,
    tokenizer: Any,
    *,
    cell: cotangent.ProbeCellSpec,
    device: Any,
    frozen: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    import torch
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    native = frozen.native_generation
    captions = {
        "action": cell.action_caption,
        "reverse_wrong_family": cell.reverse_wrong_family_caption,
        "common_scene_noop": cell.noop_caption,
    }
    conditions: dict[str, Any] = {}
    for role in PROMPT_ORDER:
        prompt = native.build_task_prompt("t2v", captions[role], prompt_cleaner=prompt_clean)
        ids, mask = native.legacy._tokenize_training_prompt(tokenizer, prompt)
        with torch.inference_mode():
            condition = renderer.encode_prompt(ids.to(device), mask.to(device)).detach()
        if tuple(int(item) for item in condition.shape) != (1, 512, 4096):
            raise OwnerTemplateMaterializationError(f"{role} encoded prompt shape differs")
        conditions[role] = condition
    hashes = {role: starc.tensor_sha256(value.float()) for role, value in conditions.items()}
    if len(set(hashes.values())) != 3:
        raise OwnerTemplateMaterializationError("encoded prompt triplet aliases")
    return conditions, hashes


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def persist_cell_materialization(
    *,
    output_dir: str | Path,
    bundle: CellTemplateMaterialization,
    authority: AuthorizedOwnerInputs,
    model_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist exactly two quotient tensors and one specificity receipt."""

    child = authority.cell(bundle.cell_id)
    binding = bundle.production_input_binding
    if not isinstance(binding, Mapping):
        raise OwnerTemplateMaterializationError(
            "test/raw callback bundles cannot cross the persistent owner/editor boundary"
        )
    clean_artifact = child["artifacts"]["predecode_clean_latent"]
    expected_binding = {
        "owner_child_receipt_digest": child["receipt_digest"],
        "external_full81_audit_sidecar_receipt_digest": authority.audit_sidecar_receipt_digest,
        "owner_clean_latent_file_sha256": clean_artifact["sha256"],
        "owner_clean_latent_tensor_digest": binding["owner_clean_latent_tensor_digest"],
    }
    if dict(binding) != expected_binding:
        raise OwnerTemplateMaterializationError("persistent owner input authority differs")
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    clean_key = clean_artifact.get("tensor_key", "normalized_clean_latent")
    with safe_open(str(clean_artifact["path"]), framework="pt", device="cpu") as opened:
        if clean_key not in opened.keys():
            raise OwnerTemplateMaterializationError("bound owner clean tensor key differs")
        rebound_clean = opened.get_tensor(clean_key).float().contiguous().detach()
    if tensor_sha256(rebound_clean, label="rebound owner clean latent") != binding[
        "owner_clean_latent_tensor_digest"
    ]:
        raise OwnerTemplateMaterializationError(
            "bundle clean tensor is not the authenticated owner artifact"
        )
    required_model = {
        "native_schedule_index": cotangent.SCHEDULE_INDEX,
        "native_timestep": cotangent.NATIVE_TIMESTEP,
        "sigma": cotangent.NATIVE_SIGMA,
        "hook_coordinate": cotangent.HOOK_COORDINATE,
        "transformer_1_only": True,
        "all_parameters_frozen": True,
        "adapter_loaded": False,
    }
    if any(model_binding.get(name) != expected for name, expected in required_model.items()):
        raise OwnerTemplateMaterializationError("persistent frozen model binding differs")
    for row in bundle.rows:
        forwards = row.hidden_forward_proof.get("forward_receipts")
        if (
            not isinstance(forwards, list)
            or len(forwards) != 3
            or not isinstance(forwards[0], Mapping)
            or forwards[0].get("prompt_order") != list(PROMPT_ORDER)
            or forwards[0].get("same_x_sigma_noisy_rotary_timestep_objects") is not True
            or forwards[0].get("shared_tensor_bytes_unchanged") is not True
            or forwards[0].get("target_suffix_only") is not True
            or forwards[0].get("source_condition_consumed") is not False
            or forwards[0].get("full_hidden_persisted") is not False
        ):
            raise OwnerTemplateMaterializationError(
                "persistent template lacks the internal frozen same-state triplet proof"
            )
    output = _fresh_path(output_dir, label="cell materialization output")
    output.mkdir(mode=0o700)
    tensor_path = output / QUOTIENT_FILENAME
    tensors = bundle.tensors()
    expected_keys = [f"{TENSOR_KEY_PREFIX}{seed}" for seed in bundle.ordered_query_seeds]
    if list(tensors) != expected_keys or any(
        value.requires_grad or value.grad_fn is not None or value.dtype != torch.float32
        for value in tensors.values()
    ):
        raise OwnerTemplateMaterializationError("persistent quotient tensor closure differs")
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=f".{QUOTIENT_FILENAME}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={
                "schema_version": CELL_RECEIPT_SCHEMA,
                "allowed_channel": cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
                "stop_gradient": "true",
                "seed_selection": "false",
                "seed_averaging": "false",
            },
        )
        with safe_open(str(temporary), framework="pt", device="cpu") as opened:
            if list(opened.keys()) != expected_keys:
                raise OwnerTemplateMaterializationError("quotient safetensors keys differ")
            for key in expected_keys:
                restored = opened.get_tensor(key).float().contiguous().detach()
                if not torch.equal(restored, tensors[key]):
                    raise OwnerTemplateMaterializationError("quotient round trip differs")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, tensor_path)
        os.chmod(tensor_path, 0o444)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    quotient_binding = {
        "filename": QUOTIENT_FILENAME,
        "file_sha256": file_sha256(tensor_path),
        "tensor_keys_in_fixed_seed_order": expected_keys,
        "tensor_value_digests": {
            key: tensor_sha256(tensors[key], label=key) for key in expected_keys
        },
        "only_detached_normalized_quotients": True,
    }
    unsigned = {
        "schema_version": CELL_RECEIPT_SCHEMA,
        "cell_id": bundle.cell_id,
        "ordered_query_seeds": list(bundle.ordered_query_seeds),
        "owner_generation_seed": child["owner_generation_seed"],
        "owner_master_receipt_digest": authority.master_receipt_digest,
        "owner_child_receipt_digest": child["receipt_digest"],
        "external_full81_audit_sidecar_receipt_digest": authority.audit_sidecar_receipt_digest,
        "external_full81_audit_sidecar_file_sha256": authority.audit_sidecar_file_sha256,
        "owner_exact81_action_audit_passed": True,
        "query_rows": [row.receipt() for row in bundle.rows],
        "two_seed_audit": bundle.two_seed_audit.receipt(),
        "quotient_artifact": quotient_binding,
        "model_binding": dict(model_binding),
        "persistent_tensor_channel": cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
        "forbidden_owner_to_editor_channels": list(
            cotangent.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS
        ),
        "owner_media_or_primal_tensor_persisted": False,
        "full_hidden_persisted": False,
        "seed_selection": False,
        "seed_averaging": False,
        "optimizer_constructed": False,
        "parameter_update_performed": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_json_create_only(output / CELL_RECEIPT_FILENAME, receipt)
    if set(path.name for path in output.iterdir()) != {
        QUOTIENT_FILENAME,
        CELL_RECEIPT_FILENAME,
    }:
        raise OwnerTemplateMaterializationError("cell output channel closure differs")
    os.chmod(output, 0o500)
    return receipt


def validate_published_cell_packet(
    value: Mapping[str, Any],
    *,
    cell_root: str | Path,
    authority: AuthorizedOwnerInputs,
) -> dict[str, Any]:
    """Revalidate one sealed quotient packet and its only tensor artifact."""

    from safetensors import safe_open
    import torch

    row = _exact_fields(value, _CELL_PACKET_FIELDS, label="published cell packet")
    receipt_digest = _verify_seal(row, label="published cell packet")
    cell_id = _safe_id(row["cell_id"], label="published cell ID")
    cell = authority.registry.cell(cell_id)
    child = authority.cell(cell_id)
    seeds = list(cell.query_seeds)
    if (
        row["schema_version"] != CELL_RECEIPT_SCHEMA
        or row["ordered_query_seeds"] != seeds
        or row["owner_generation_seed"] != cell.owner_generation_seed
        or row["owner_master_receipt_digest"] != authority.master_receipt_digest
        or row["owner_child_receipt_digest"] != child["receipt_digest"]
        or row["external_full81_audit_sidecar_receipt_digest"]
        != authority.audit_sidecar_receipt_digest
        or row["external_full81_audit_sidecar_file_sha256"]
        != authority.audit_sidecar_file_sha256
        or row["owner_exact81_action_audit_passed"] is not True
        or row["persistent_tensor_channel"]
        != cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL
        or row["forbidden_owner_to_editor_channels"]
        != list(cotangent.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS)
        or row["owner_media_or_primal_tensor_persisted"] is not False
        or row["full_hidden_persisted"] is not False
        or row["seed_selection"] is not False
        or row["seed_averaging"] is not False
        or row["optimizer_constructed"] is not False
        or row["parameter_update_performed"] is not False
    ):
        raise OwnerTemplateMaterializationError("published cell authority differs")
    query_rows = row["query_rows"]
    if not isinstance(query_rows, list) or len(query_rows) != 2:
        raise OwnerTemplateMaterializationError("published query-row closure differs")
    model = row["model_binding"]
    model_sigma = model.get("sigma") if isinstance(model, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or model.get("native_schedule_index") != cotangent.SCHEDULE_INDEX
        or model.get("native_timestep") != cotangent.NATIVE_TIMESTEP
        or isinstance(model_sigma, bool)
        or not isinstance(model_sigma, (int, float))
        or float(model_sigma).hex()
        != float(cotangent.NATIVE_SIGMA).hex()
        or model.get("hook_coordinate") != cotangent.HOOK_COORDINATE
        or model.get("transformer_1_only") is not True
        or model.get("all_parameters_frozen") is not True
        or model.get("adapter_loaded") is not False
    ):
        raise OwnerTemplateMaterializationError("published frozen model proof differs")
    template_digest_by_key: dict[str, str] = {}
    for seed, raw_query in zip(seeds, query_rows):
        query = _exact_fields(
            raw_query,
            (
                "query_seed",
                "template",
                "prompt_specificity",
                "official_gaussian_scheme",
                "official_gaussian_tensor_digest",
                "same_x_sigma_binding_digest",
                "owner_spatial_coordinates",
                "same_x_sigma_object_prompt_order",
                "same_x_sigma_object_for_all_three_prompts",
                "hidden_forward_proof",
            ),
            label=f"query seed {seed} packet",
        )
        template = query["template"]
        specificity = query["prompt_specificity"]
        proof = query["hidden_forward_proof"]
        if not all(isinstance(item, Mapping) for item in (template, specificity, proof)):
            raise OwnerTemplateMaterializationError("published nested packet differs")
        forwards = proof.get("forward_receipts")
        if (
            query["query_seed"] != seed
            or query["official_gaussian_scheme"]
            != "diffusers.randn_tensor_cpu_generator_fp32_v1"
            or _SHA256_RE.fullmatch(str(query["official_gaussian_tensor_digest"])) is None
            or _SHA256_RE.fullmatch(str(query["same_x_sigma_binding_digest"])) is None
            or type(query["owner_spatial_coordinates"]) is not int
            or query["owner_spatial_coordinates"] <= 0
            or query["same_x_sigma_object_prompt_order"] != list(PROMPT_ORDER)
            or query["same_x_sigma_object_for_all_three_prompts"] is not True
            or template.get("query_seed") != seed
            or template.get("allowed_owner_to_editor_channel")
            != cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL
            or template.get("forbidden_owner_to_editor_channels")
            != list(cotangent.FORBIDDEN_OWNER_TO_EDITOR_CHANNELS)
            or template.get("stop_gradient") is not True
            or template.get("owner_media_or_primal_tensor_retained") is not False
            or template.get("candidate_spatial_coordinates_may_differ") is not True
            or specificity.get("query_seed") != seed
            or specificity.get("query_order")
            != ["action", "reverse_wrong_family", "common_scene_null"]
            or specificity.get("all_margins_pass_without_compensation") is not True
            or specificity.get("same_x_sigma_binding_digest")
            != query["same_x_sigma_binding_digest"]
            or proof.get("same_x_sigma_object") is not True
            or proof.get("x_sigma_unchanged") is not True
            or proof.get("full_hidden_persisted") is not False
            or not isinstance(forwards, list)
            or len(forwards) != 3
            or not isinstance(forwards[0], Mapping)
            or forwards[0].get("same_x_sigma_noisy_rotary_timestep_objects") is not True
            or forwards[0].get("target_suffix_only") is not True
            or forwards[0].get("source_condition_consumed") is not False
        ):
            raise OwnerTemplateMaterializationError(
                f"published query seed {seed} proof differs"
            )
        unit_digest = _sha256(
            template.get("unit_feature_digest"), label=f"query seed {seed} unit digest"
        )
        template_digest_by_key[f"{TENSOR_KEY_PREFIX}{seed}"] = unit_digest
    two_seed = row["two_seed_audit"]
    if (
        not isinstance(two_seed, Mapping)
        or two_seed.get("ordered_query_seeds") != seeds
        or two_seed.get("passed") is not True
        or two_seed.get("seed_ranking_or_selection") is not False
        or two_seed.get("seed_averaging") is not False
    ):
        raise OwnerTemplateMaterializationError("published two-seed audit differs")
    artifact = _exact_fields(
        row["quotient_artifact"],
        (
            "filename",
            "file_sha256",
            "tensor_keys_in_fixed_seed_order",
            "tensor_value_digests",
            "only_detached_normalized_quotients",
        ),
        label="published quotient artifact",
    )
    keys = [f"{TENSOR_KEY_PREFIX}{seed}" for seed in seeds]
    root = _plain_dir(cell_root, label="published cell root")
    quotient_path = _plain_file(root / QUOTIENT_FILENAME, label="published quotient")
    if (
        artifact["filename"] != QUOTIENT_FILENAME
        or artifact["tensor_keys_in_fixed_seed_order"] != keys
        or artifact["tensor_value_digests"] != template_digest_by_key
        or artifact["only_detached_normalized_quotients"] is not True
        or file_sha256(quotient_path)
        != _sha256(artifact["file_sha256"], label="quotient file SHA-256")
    ):
        raise OwnerTemplateMaterializationError("published quotient binding differs")
    with safe_open(str(quotient_path), framework="pt", device="cpu") as opened:
        if list(opened.keys()) != keys:
            raise OwnerTemplateMaterializationError("published quotient key closure differs")
        for key in keys:
            tensor = opened.get_tensor(key).float().contiguous().detach()
            if (
                tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.grad_fn is not None
                or tensor_sha256(tensor, label=f"published {key}")
                != template_digest_by_key[key]
            ):
                raise OwnerTemplateMaterializationError(
                    f"published quotient tensor {key} differs"
                )
    checked = dict(row)
    checked["receipt_digest"] = receipt_digest
    return checked


def _common_authority_from_args(args: argparse.Namespace) -> AuthorizedOwnerInputs:
    return load_authorized_owner_inputs(
        registry=args.registry,
        expected_registry_sha256=args.expected_registry_sha256,
        owner_root=args.owner_root,
        owner_master_receipt=args.owner_master_receipt,
        expected_owner_master_receipt_sha256=args.expected_owner_master_receipt_sha256,
        audit_sidecar=args.audit_sidecar,
        expected_audit_sidecar_sha256=args.expected_audit_sidecar_sha256,
        audit_evidence=args.audit_evidence,
        audit_public_key=args.audit_public_key,
        expected_audit_public_key_sha256=args.expected_audit_public_key_sha256,
    )


def preflight(args: argparse.Namespace) -> int:
    _common_authority_from_args(args)
    return 0


def _rank0_action(*, dist: Any, rank: int, action: Callable[[], Any], label: str) -> Any:
    payload: list[Any] = [None]
    if rank == 0:
        try:
            payload[0] = {"ok": True, "value": action()}
        except BaseException as error:
            payload[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        raise OwnerTemplateMaterializationError(f"rank-zero {label} failed: {result}")
    return result.get("value")


def materialize_cell(args: argparse.Namespace) -> int:
    authority = _common_authority_from_args(args)
    cell = authority.registry.cell(args.cell_id)
    if (
        os.environ.get("WORLD_SIZE") != "4"
        or os.environ.get("ROCR_VISIBLE_DEVICES") != EXPECTED_VISIBLE_BY_CELL[cell.cell_id]
    ):
        raise OwnerTemplateMaterializationError(
            "owner template cell must run in its sealed WORLD4/SP4 GPU group"
        )
    _sha1(args.method_source_revision, label="method source revision")
    _sha256(args.method_source_archive_sha256, label="method archive SHA-256")
    materializer_sha = file_sha256(Path(__file__).resolve())
    if materializer_sha != _sha256(
        args.expected_materializer_source_sha256, label="materializer source SHA-256"
    ):
        raise OwnerTemplateMaterializationError("materializer source bytes changed")
    frozen = starc.temporal_scorer._frozen_d541801_runtime()
    starc.temporal_scorer.validate_native_coordinate_runtime(frozen)
    native = frozen.native_generation
    legacy = native.legacy
    temporal_contract = starc.temporal_contract
    if (
        args.expected_bernini_commit != temporal_contract.REQUIRED_BERNINI_REVISION
        or args.expected_veomni_commit != temporal_contract.REQUIRED_VEOMNI_REVISION
    ):
        raise OwnerTemplateMaterializationError("Bernini/VeOmni revision differs")
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
        raise OwnerTemplateMaterializationError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise OwnerTemplateMaterializationError("pinned Bernini head count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

    distributed = legacy.inference_distributed_contract()
    if (
        distributed.world_size != EXPECTED_WORLD_SIZE
        or not torch.cuda.is_available()
        or getattr(torch.version, "hip", None) is None
    ):
        raise OwnerTemplateMaterializationError("materializer requires AUH ROCm SP4")
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=EXPECTED_WORLD_SIZE)
    device = torch.device("cuda", distributed.local_rank)
    observer: Optional[starc.Block15SpatialSketchObserver] = None
    try:
        identity_rows: list[Any] = [None]
        if distributed.rank == 0:
            try:
                identity = native.source_audit.validate_checkpoint_content(
                    checkpoint, Path(args.checkpoint_content_manifest)
                )
                identity_rows[0] = {"ok": True, "value": identity}
            except Exception as error:
                identity_rows[0] = {"ok": False, "error": str(error)}
        dist.broadcast_object_list(identity_rows, src=0)
        if not isinstance(identity_rows[0], Mapping) or identity_rows[0].get("ok") is not True:
            raise OwnerTemplateMaterializationError(
                f"checkpoint content audit failed: {identity_rows[0]}"
            )
        checkpoint_identity = dict(identity_rows[0]["value"])
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
            transformer is None
            or diffusion.transformer_2 is not None
            or any(parameter.requires_grad for parameter in renderer.parameters())
        ):
            raise OwnerTemplateMaterializationError("frozen transformer_1 closure differs")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
        )
        conditions, condition_hashes = _encode_prompt_triplet(
            renderer, tokenizer, cell=cell, device=device, frozen=frozen
        )
        child = authority.cell(cell.cell_id)
        clean_artifact = child["artifacts"]["predecode_clean_latent"]
        clean = frozen._load_exact81_tensor(
            clean_artifact,
            key="normalized_clean_latent",
            label=f"{cell.cell_id} owner clean latent",
        )
        starc.verify_authenticated_native_clean_tensor_identity(
            clean,
            clean_artifact,
            label=f"{cell.cell_id} owner clean latent",
            frozen=frozen,
        )
        if tuple(int(item) for item in clean.shape) != tuple(cell.latent_shape):
            raise OwnerTemplateMaterializationError("owner clean latent geometry differs")
        clean_tensor_digest = tensor_sha256(
            clean.detach(), label=f"{cell.cell_id} authenticated owner clean latent"
        )
        clean = clean.to(device=device).float().contiguous().detach()
        _shape, patch_height, patch_width, _positions = starc.latent_geometry(clean)
        spatial = starc.fixed_spatial_sketch(
            patch_height=patch_height, patch_width=patch_width, device=device
        )
        observer = starc.Block15SpatialSketchObserver(
            transformer,
            sp_rank=distributed.rank,
            patch_height=patch_height,
            patch_width=patch_width,
            spatial_sketch=spatial,
        ).install()
        cache: dict[tuple[int, int], tuple[dict[str, Any], dict[str, Any]]] = {}

        def hidden_forward(
            *, x_sigma: Any, prompt_role: str, prompt_caption: str, query_seed: int
        ) -> tuple[Any, Mapping[str, Any]]:
            del prompt_caption
            key = (query_seed, id(x_sigma))
            if key not in cache:
                cache[key] = forward_same_state_hidden_triplet(
                    diffusion=diffusion,
                    transformer=transformer,
                    observer=observer,
                    x_sigma=x_sigma,
                    prompt_conditions=conditions,
                    query_key=f"{cell.cell_id}:q{query_seed}",
                    dist_module=dist,
                )
            captures, proof = cache[key]
            return captures[prompt_role], proof if prompt_role == "action" else {
                "triplet_cache_reused": True,
                "prompt_role": prompt_role,
            }

        bundle = _materialize_cell_from_hidden_callback_unsafe(
            cell=cell,
            owner_clean_latent=clean,
            hidden_forward=hidden_forward,
            specificity_margin=float(
                authority.registry.contract["minimum_same_topology_specificity_margin"]
            ),
            minimum_template_cosine=float(
                authority.registry.contract["minimum_owner_template_cosine"]
            ),
            production_input_binding={
                "owner_child_receipt_digest": child["receipt_digest"],
                "external_full81_audit_sidecar_receipt_digest": (
                    authority.audit_sidecar_receipt_digest
                ),
                "owner_clean_latent_file_sha256": clean_artifact["sha256"],
                "owner_clean_latent_tensor_digest": clean_tensor_digest,
            },
        )
        receipt_digest = object_sha256(
            {
                "cell_id": bundle.cell_id,
                "rows": [row.receipt() for row in bundle.rows],
                "two_seed": bundle.two_seed_audit.receipt(),
            }
        )
        gathered: list[Any] = [None] * EXPECTED_WORLD_SIZE
        dist.all_gather_object(gathered, receipt_digest)
        if len(set(gathered)) != 1:
            raise OwnerTemplateMaterializationError("SP4 quotient receipts differ by rank")
        model_binding = {
            "bernini_revision": bernini_revision,
            "veomni_revision": veomni_revision,
            "checkpoint_content_receipt_digest": frozen.object_sha256(checkpoint_identity),
            "method_source_revision": args.method_source_revision,
            "method_source_archive_sha256": args.method_source_archive_sha256,
            "materializer_source_sha256": materializer_sha,
            "native_schedule_index": cotangent.SCHEDULE_INDEX,
            "native_timestep": cotangent.NATIVE_TIMESTEP,
            "sigma": cotangent.NATIVE_SIGMA,
            "hook_coordinate": cotangent.HOOK_COORDINATE,
            "spatial_sketch_family_id": starc.SKETCH_FAMILY_ID,
            "transformer_1_only": True,
            "all_parameters_frozen": True,
            "adapter_loaded": False,
        }
        _rank0_action(
            dist=dist,
            rank=distributed.rank,
            label=f"persist {cell.cell_id} quotient",
            action=lambda: persist_cell_materialization(
                output_dir=args.output_dir,
                bundle=bundle,
                authority=authority,
                model_binding=model_binding,
            )["receipt_digest"],
        )
        dist.barrier()
        return 0
    finally:
        if observer is not None and observer.active:
            observer.abort()
        if observer is not None and observer.installed:
            observer.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


def aggregate_master(args: argparse.Namespace) -> int:
    authority = _common_authority_from_args(args)
    root = _plain_dir(args.output_root, label="materialization staging root")
    output = _fresh_path(root / MASTER_RECEIPT_FILENAME, label="specificity master receipt")
    children = []
    for cell_id in owner_generation.CELL_IDS:
        cell_root = _plain_dir(root / cell_id, label=f"{cell_id} quotient output")
        if set(path.name for path in cell_root.iterdir()) != {
            QUOTIENT_FILENAME,
            CELL_RECEIPT_FILENAME,
        }:
            raise OwnerTemplateMaterializationError(f"{cell_id} output channel closure differs")
        receipt, receipt_path, receipt_sha = _load_strict_json(
            cell_root / CELL_RECEIPT_FILENAME,
            expected_sha256=None,
            label=f"{cell_id} specificity receipt",
        )
        checked = validate_published_cell_packet(
            receipt, cell_root=cell_root, authority=authority
        )
        if checked["cell_id"] != cell_id:
            raise OwnerTemplateMaterializationError(f"{cell_id} packet order differs")
        receipt_digest = checked["receipt_digest"]
        quotient_path = _plain_file(
            cell_root / QUOTIENT_FILENAME, label=f"{cell_id} quotient artifact"
        )
        children.append(
            {
                "cell_id": cell_id,
                "receipt_relative_path": f"{cell_id}/{CELL_RECEIPT_FILENAME}",
                "receipt_file_sha256": receipt_sha,
                "receipt_digest": receipt_digest,
                "quotient_relative_path": f"{cell_id}/{QUOTIENT_FILENAME}",
                "quotient_file_sha256": file_sha256(quotient_path),
                "ordered_query_seeds": list(authority.registry.cell(cell_id).query_seeds),
            }
        )
    unsigned = {
        "schema_version": MASTER_RECEIPT_SCHEMA,
        "cell_order": list(owner_generation.CELL_IDS),
        "children": children,
        "owner_master_receipt_digest": authority.master_receipt_digest,
        "external_full81_audit_sidecar_receipt_digest": authority.audit_sidecar_receipt_digest,
        "topology": "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
        "template_count": 4,
        "two_templates_per_cell_retained_separately": True,
        "seed_selection": False,
        "seed_averaging": False,
        "persistent_tensor_channel": cotangent.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
        "owner_media_or_primal_tensor_persisted": False,
        "optimizer_constructed": False,
        "parameter_update_performed": False,
    }
    receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
    _write_json_create_only(output, receipt)
    return 0


def _add_authority_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", required=True)
    parser.add_argument("--expected-registry-sha256", required=True)
    parser.add_argument("--owner-root", required=True)
    parser.add_argument("--owner-master-receipt", required=True)
    parser.add_argument("--expected-owner-master-receipt-sha256", required=True)
    parser.add_argument("--audit-sidecar", required=True)
    parser.add_argument("--expected-audit-sidecar-sha256", required=True)
    parser.add_argument("--audit-evidence", required=True)
    parser.add_argument("--audit-public-key", required=True)
    parser.add_argument("--expected-audit-public-key-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser(
        "preflight", description="Reject pending/tampered owner authority before GPU launch."
    )
    _add_authority_args(preflight_parser)
    materialize = commands.add_parser(
        "materialize-cell", description="Materialize one audited owner on one SP4 group."
    )
    _add_authority_args(materialize)
    materialize.add_argument("--cell-id", choices=owner_generation.CELL_IDS, required=True)
    materialize.add_argument("--bernini-root", required=True)
    materialize.add_argument("--veomni-root", required=True)
    materialize.add_argument("--checkpoint", required=True)
    materialize.add_argument("--checkpoint-content-manifest", required=True)
    materialize.add_argument("--expected-bernini-commit", required=True)
    materialize.add_argument("--expected-veomni-commit", required=True)
    materialize.add_argument("--method-source-revision", required=True)
    materialize.add_argument("--method-source-archive-sha256", required=True)
    materialize.add_argument("--expected-materializer-source-sha256", required=True)
    materialize.add_argument("--output-dir", required=True)
    aggregate = commands.add_parser(
        "aggregate-master", description="Bind the two independently retained seed pairs."
    )
    _add_authority_args(aggregate)
    aggregate.add_argument("--output-root", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "materialize-cell":
        return materialize_cell(args)
    if args.command == "aggregate-master":
        return aggregate_master(args)
    raise OwnerTemplateMaterializationError("unknown materializer command")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_JOB_ID",
    "AUDIT_SCHEMA",
    "AUDIT_SIGNATURE_SCHEME",
    "AuthorizedOwnerInputs",
    "CELL_RECEIPT_FILENAME",
    "CELL_RECEIPT_SCHEMA",
    "MASTER_RECEIPT_FILENAME",
    "MASTER_RECEIPT_SCHEMA",
    "OwnerTemplateMaterializationError",
    "PendingOwnerGenerationInputs",
    "PROMPT_ORDER",
    "QUOTIENT_FILENAME",
    "TENSOR_KEY_PREFIX",
    "aggregate_master",
    "build_parser",
    "canonical_json_bytes",
    "file_sha256",
    "forward_same_state_hidden_triplet",
    "load_authorized_owner_inputs",
    "load_pending_owner_generation_inputs",
    "main",
    "object_sha256",
    "official_gaussian_for_query_seed",
    "preflight",
    "tensor_sha256",
    "validate_published_cell_packet",
]

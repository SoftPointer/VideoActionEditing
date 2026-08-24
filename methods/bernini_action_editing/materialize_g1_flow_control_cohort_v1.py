#!/usr/bin/env python3
"""Materialize the missing G1 dense-flow controls as a sealed cohort.

This is a representation-only utility.  It never imports a generator, creates
an optimizer, or reads target pixels.  ``correct``, ``temporal_shuffle`` and
``reverse`` must already be independently extracted RAFT bundles.  This tool
adds the three controls that cannot be obtained by merely choosing an anchor
video:

* ``zero_or_noop``: all flow and validity values are exact zero;
* ``incomplete``: a strict prefix is retained and the remaining transitions
  (including validity) are exact zero;
* ``wrong_action_energy_matched``: a different action-family bundle is scaled
  by one common scalar so its validity-weighted motion RMS matches ``correct``.

Target and self-generated cohorts are separate invocations.  Publication is
atomic at the directory level and is not valid without a replayable cohort
receipt.  Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Mapping, Sequence


COHORT_SCHEMA_VERSION = "bernini-g1-flow-control-cohort-v1"
BUNDLE_RECEIPT_SCHEMA_VERSION = "bernini-g1-flow-control-bundle-receipt-v1"
EXTRACTOR_SCHEMA_VERSION = "bernini-anchor-raft-flow-bundle-v1"
ENERGY_DEFINITION = (
    "sqrt(sum(validity*(raw_xy_squared+camera_residual_xy_squared))"
    "/(4*sum(validity)))"
)
REQUIRED_TENSORS = (
    "backward_raw",
    "backward_camera_residual",
    "validity",
)
EXTERNAL_ROLES = ("correct", "temporal_shuffle", "reverse", "wrong_action_donor")
GENERATED_CONTROLS = (
    "zero_or_noop",
    "incomplete",
    "wrong_action_energy_matched",
)
ANCHOR_KINDS = ("target", "selfgen")
EXPECTED_TRANSITIONS = 20
ENERGY_MATCH_RTOL = 2.0e-5
ENERGY_EPSILON = 1.0e-12
MAX_ENERGY_SCALE = 100.0

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class G1FlowControlError(RuntimeError):
    """Raised when a G1 control cohort cannot be proven closed."""


@dataclass(frozen=True)
class LoadedBundle:
    path: Path
    sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    sidecar: dict[str, Any]
    metadata: dict[str, str]
    tensors: dict[str, Any]


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        else:
            text = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
    except (TypeError, ValueError) as error:
        raise G1FlowControlError("receipt is not finite JSON") from error
    return (text + ("\n" if pretty else "")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise G1FlowControlError(f"{label} must be a sealed identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise G1FlowControlError(f"{label} must be lowercase SHA-256")
    return value


def _regular_path(value: Path | str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise G1FlowControlError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_path(path, label=label)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise G1FlowControlError(f"{label} must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise G1FlowControlError(f"{label} must be a JSON object")
    return value, _sha256_bytes(payload)


def _validate_tensor_bundle(tensors: Mapping[str, Any], *, label: str) -> None:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise G1FlowControlError("PyTorch is required for G1 flow controls") from error

    if set(tensors) != set(REQUIRED_TENSORS) or len(tensors) != len(REQUIRED_TENSORS):
        raise G1FlowControlError(f"{label} tensor-key closure differs")
    raw = tensors["backward_raw"]
    camera = tensors["backward_camera_residual"]
    validity = tensors["validity"]
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise G1FlowControlError(f"{label}.{name} must be rank-4")
        if not tensor.dtype.is_floating_point:
            raise G1FlowControlError(f"{label}.{name} must be floating point")
        if not bool(torch.isfinite(tensor).all().item()):
            raise G1FlowControlError(f"{label}.{name} is non-finite")
    if tuple(map(int, raw.shape[:2])) != (EXPECTED_TRANSITIONS, 2):
        raise G1FlowControlError(f"{label}.backward_raw must have shape [20,2,H,W]")
    if tuple(camera.shape) != tuple(raw.shape) or camera.dtype != raw.dtype:
        raise G1FlowControlError(f"{label} raw/camera geometry or dtype differs")
    expected_validity = (EXPECTED_TRANSITIONS, 1, *map(int, raw.shape[-2:]))
    if tuple(map(int, validity.shape)) != expected_validity:
        raise G1FlowControlError(f"{label}.validity geometry differs")
    if bool((validity < 0).any().item()) or bool((validity > 1).any().item()):
        raise G1FlowControlError(f"{label}.validity must lie in [0,1]")


def _load_tensor_file(path: Path | str, *, label: str) -> tuple[Path, str, dict[str, Any], dict[str, str]]:
    path = _regular_path(path, label=label)
    if path.suffix != ".safetensors":
        raise G1FlowControlError(f"{label} must end in .safetensors")
    before = _sha256_file(path)
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            metadata = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key).contiguous() for key in keys}
    except Exception as error:
        raise G1FlowControlError(f"cannot load {label}: {path}") from error
    if _sha256_file(path) != before:
        raise G1FlowControlError(f"{label} changed while being read")
    _validate_tensor_bundle(tensors, label=label)
    return path, before, tensors, metadata


def _load_extractor_bundle(path: Path | str, *, label: str) -> LoadedBundle:
    resolved, digest, tensors, metadata = _load_tensor_file(path, label=label)
    sidecar_path = resolved.with_suffix(".json")
    sidecar, sidecar_sha = _read_json_file(sidecar_path, label=f"{label} sidecar")
    required = {
        "schema_version",
        "source_sha256",
        "anchor_sha256",
        "sampled_frame_indices",
        "latent_hw",
    }
    if not required.issubset(sidecar):
        raise G1FlowControlError(f"{label} extractor sidecar is incomplete")
    if sidecar["schema_version"] != EXTRACTOR_SCHEMA_VERSION:
        raise G1FlowControlError(f"{label} extractor schema differs")
    _sha(sidecar["source_sha256"], label=f"{label}.source_sha256")
    _sha(sidecar["anchor_sha256"], label=f"{label}.anchor_sha256")
    if sidecar["sampled_frame_indices"] != list(range(0, 81, 4)):
        raise G1FlowControlError(f"{label} sampled phases differ")
    if sidecar["latent_hw"] != list(map(int, tensors["backward_raw"].shape[-2:])):
        raise G1FlowControlError(f"{label} latent geometry sidecar differs")
    return LoadedBundle(
        path=resolved,
        sha256=digest,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha,
        sidecar=sidecar,
        metadata=metadata,
        tensors=tensors,
    )


def _bundle_ref(bundle: LoadedBundle) -> dict[str, Any]:
    return {
        "path": str(bundle.path),
        "sha256": bundle.sha256,
        "sidecar_path": str(bundle.sidecar_path),
        "sidecar_sha256": bundle.sidecar_sha256,
        "source_sha256": bundle.sidecar["source_sha256"],
        "anchor_sha256": bundle.sidecar["anchor_sha256"],
        "tensor_shapes": {
            key: list(map(int, bundle.tensors[key].shape)) for key in REQUIRED_TENSORS
        },
    }


def _effective_energy(tensors: Mapping[str, Any]) -> float:
    import torch

    validity = tensors["validity"].double()
    denominator = float(validity.sum().item()) * 4.0
    if not math.isfinite(denominator) or denominator <= ENERGY_EPSILON:
        raise G1FlowControlError("motion energy is undefined because validity is zero")
    squared = (
        tensors["backward_raw"].double().square().sum(dim=1, keepdim=True)
        + tensors["backward_camera_residual"].double().square().sum(dim=1, keepdim=True)
    )
    value = float(torch.sqrt((squared * validity).sum() / denominator).item())
    if not math.isfinite(value) or value <= ENERGY_EPSILON:
        raise G1FlowControlError("motion energy must be finite and nonzero")
    return value


def _build_transforms(
    correct: LoadedBundle,
    wrong: LoadedBundle,
    *,
    incomplete_transitions: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    if not 1 <= incomplete_transitions < EXPECTED_TRANSITIONS:
        raise G1FlowControlError("incomplete transitions must lie in [1,19]")
    zero = {
        key: torch.zeros_like(correct.tensors[key]).contiguous()
        for key in REQUIRED_TENSORS
    }
    incomplete = {key: correct.tensors[key].clone().contiguous() for key in REQUIRED_TENSORS}
    for value in incomplete.values():
        value[incomplete_transitions:] = 0
    removed = {
        key: int(torch.count_nonzero(correct.tensors[key][incomplete_transitions:]).item())
        for key in REQUIRED_TENSORS
    }
    if removed["backward_raw"] + removed["backward_camera_residual"] == 0:
        raise G1FlowControlError("incomplete control removes no nonzero tail motion")

    correct_energy = _effective_energy(correct.tensors)
    donor_energy = _effective_energy(wrong.tensors)
    scale = correct_energy / donor_energy
    if not math.isfinite(scale) or not 1.0 / MAX_ENERGY_SCALE <= scale <= MAX_ENERGY_SCALE:
        raise G1FlowControlError("wrong-action energy scale is outside the sealed bound")
    wrong_matched = {
        "backward_raw": (wrong.tensors["backward_raw"] * scale).contiguous(),
        "backward_camera_residual": (
            wrong.tensors["backward_camera_residual"] * scale
        ).contiguous(),
        "validity": wrong.tensors["validity"].clone().contiguous(),
    }
    matched_energy = _effective_energy(wrong_matched)
    relative_error = abs(matched_energy - correct_energy) / correct_energy
    if relative_error > ENERGY_MATCH_RTOL:
        raise G1FlowControlError("wrong-action energy matching tolerance failed")
    return (
        {
            "zero_or_noop": zero,
            "incomplete": incomplete,
            "wrong_action_energy_matched": wrong_matched,
        },
        {
            "energy_definition": ENERGY_DEFINITION,
            "correct_energy": correct_energy,
            "wrong_action_donor_energy_before_scale": donor_energy,
            "wrong_action_scale": scale,
            "wrong_action_energy_after_scale": matched_energy,
            "wrong_action_relative_energy_error": relative_error,
            "energy_match_rtol": ENERGY_MATCH_RTOL,
            "maximum_energy_scale": MAX_ENERGY_SCALE,
            "incomplete_transitions_retained": incomplete_transitions,
            "incomplete_transitions_removed": EXPECTED_TRANSITIONS - incomplete_transitions,
            "incomplete_removed_nonzero_counts": removed,
        },
    )


def _save_tensor_bytes(
    tensors: Mapping[str, Any],
    *,
    input_metadata: Mapping[str, str],
    case_id: str,
    anchor_kind: str,
    control_kind: str,
    correct_sha256: str,
) -> bytes:
    try:
        from safetensors.torch import load as load_safetensors
        from safetensors.torch import save as save_safetensors
    except ModuleNotFoundError as error:
        raise G1FlowControlError("safetensors is required for G1 controls") from error
    reserved = {
        "bernini_g1_schema_version": COHORT_SCHEMA_VERSION,
        "bernini_g1_case_id": case_id,
        "bernini_g1_anchor_kind": anchor_kind,
        "bernini_g1_control_kind": control_kind,
        "bernini_g1_correct_sha256": correct_sha256,
    }
    if set(input_metadata).intersection(reserved):
        raise G1FlowControlError("input metadata collides with G1 reserved keys")
    metadata = dict(input_metadata)
    metadata.update(reserved)
    payload = save_safetensors(dict(tensors), metadata=metadata)
    roundtrip = load_safetensors(payload)
    if set(roundtrip) != set(REQUIRED_TENSORS):
        raise G1FlowControlError("generated control failed tensor-key round trip")
    import torch
    for key in REQUIRED_TENSORS:
        if not torch.equal(roundtrip[key], tensors[key]):
            raise G1FlowControlError(f"generated control round trip differs: {key}")
    return payload


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise G1FlowControlError(f"refusing to overwrite {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _generated_ref(path: Path, payload: bytes, sidecar_path: Path, sidecar_payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256_bytes(sidecar_payload),
    }


def materialize_cohort(
    *,
    correct_path: Path | str,
    temporal_shuffle_path: Path | str,
    reverse_path: Path | str,
    wrong_action_path: Path | str,
    output_dir: Path | str,
    case_id: str,
    anchor_kind: str,
    action_family: str,
    wrong_case_id: str,
    wrong_action_family: str,
    incomplete_transitions: int = 10,
) -> dict[str, Any]:
    """Materialize and atomically publish one target or selfgen G1 cohort."""

    case_id = _identifier(case_id, label="case_id")
    wrong_case_id = _identifier(wrong_case_id, label="wrong_case_id")
    action_family = _identifier(action_family, label="action_family")
    wrong_action_family = _identifier(wrong_action_family, label="wrong_action_family")
    if anchor_kind not in ANCHOR_KINDS:
        raise G1FlowControlError(f"anchor_kind must be one of {ANCHOR_KINDS}")
    if case_id == wrong_case_id:
        raise G1FlowControlError("wrong-action donor must use a different case")
    if action_family == wrong_action_family:
        raise G1FlowControlError("wrong-action donor must use a different action family")

    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise G1FlowControlError(f"refusing to overwrite output directory: {output}")
    correct = _load_extractor_bundle(correct_path, label="correct")
    shuffle = _load_extractor_bundle(temporal_shuffle_path, label="temporal_shuffle")
    reverse = _load_extractor_bundle(reverse_path, label="reverse")
    wrong = _load_extractor_bundle(wrong_action_path, label="wrong_action_donor")
    external = {
        "correct": correct,
        "temporal_shuffle": shuffle,
        "reverse": reverse,
        "wrong_action_donor": wrong,
    }
    hashes = [bundle.sha256 for bundle in external.values()]
    if len(set(hashes)) != len(hashes):
        raise G1FlowControlError("external control bundles alias by SHA-256")
    correct_shape = tuple(correct.tensors["backward_raw"].shape)
    if any(tuple(bundle.tensors["backward_raw"].shape) != correct_shape for bundle in external.values()):
        raise G1FlowControlError("external control bundle geometry differs")
    correct_source = correct.sidecar["source_sha256"]
    for role in ("temporal_shuffle", "reverse"):
        if external[role].sidecar["source_sha256"] != correct_source:
            raise G1FlowControlError(f"{role} is not bound to the correct source")

    generated, diagnostics = _build_transforms(
        correct, wrong, incomplete_transitions=incomplete_transitions
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    if staging.exists():
        raise G1FlowControlError("staging collision")
    staging.mkdir(mode=0o700)
    published = False
    try:
        generated_receipts: dict[str, Any] = {}
        for control_kind in GENERATED_CONTROLS:
            bundle_path = staging / f"{control_kind}.safetensors"
            sidecar_path = staging / f"{control_kind}.json"
            payload = _save_tensor_bytes(
                generated[control_kind],
                input_metadata=correct.metadata,
                case_id=case_id,
                anchor_kind=anchor_kind,
                control_kind=control_kind,
                correct_sha256=correct.sha256,
            )
            item_receipt = {
                "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
                "case_id": case_id,
                "anchor_kind": anchor_kind,
                "control_kind": control_kind,
                "correct_sha256": correct.sha256,
                "wrong_action_donor_sha256": (
                    wrong.sha256 if control_kind == "wrong_action_energy_matched" else None
                ),
                "tensor_shapes": {
                    key: list(map(int, generated[control_kind][key].shape))
                    for key in REQUIRED_TENSORS
                },
                "transform_diagnostics": diagnostics,
                "target_rgb_vae_or_clean_latent_accessed": False,
                "optimizer_created": False,
                "current_experiment_optimization_steps": 0,
            }
            sidecar_payload = _canonical_json_bytes(item_receipt, pretty=True)
            _write_create_only(bundle_path, payload)
            _write_create_only(sidecar_path, sidecar_payload)
            generated_receipts[control_kind] = _generated_ref(
                output / bundle_path.name,
                payload,
                output / sidecar_path.name,
                sidecar_payload,
            )

        receipt = {
            "schema_version": COHORT_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "case_id": case_id,
            "anchor_kind": anchor_kind,
            "correct_role": "real_forward" if anchor_kind == "target" else "self_generated",
            "action_family": action_family,
            "wrong_case_id": wrong_case_id,
            "wrong_action_family": wrong_action_family,
            "external_bundles": {role: _bundle_ref(external[role]) for role in EXTERNAL_ROLES},
            "generated_controls": generated_receipts,
            "diagnostics": diagnostics,
            "contracts": {
                "required_controls": [
                    "zero_or_noop",
                    "temporal_shuffle",
                    "reverse",
                    "incomplete",
                    "wrong_action_energy_matched",
                ],
                "target_and_selfgen_judged_separately": True,
                "weighted_compensation_forbidden": True,
                "representation_only": True,
                "target_rgb_vae_or_clean_latent_accessed": False,
                "optimizer_created": False,
                "current_experiment_optimization_steps": 0,
                "atomic_directory_publication": True,
            },
        }
        receipt_path = staging / "cohort_receipt.json"
        _write_create_only(receipt_path, _canonical_json_bytes(receipt, pretty=True))
        if output.exists() or output.is_symlink():
            raise G1FlowControlError(f"output appeared during publication: {output}")
        os.rename(staging, output)
        published = True
        final_receipt = output / "cohort_receipt.json"
        verify_cohort_receipt(final_receipt)
        return json.loads(final_receipt.read_text(encoding="utf-8"))
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _verify_ref(value: Any, *, label: str, extractor: bool) -> LoadedBundle:
    if not isinstance(value, Mapping):
        raise G1FlowControlError(f"{label} reference must be an object")
    required = {
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "source_sha256",
        "anchor_sha256",
        "tensor_shapes",
    }
    if set(value) != required:
        raise G1FlowControlError(f"{label} reference field closure differs")
    bundle = _load_extractor_bundle(value["path"], label=label)
    if bundle.sha256 != _sha(value["sha256"], label=f"{label}.sha256"):
        raise G1FlowControlError(f"{label} bundle SHA differs")
    if str(bundle.sidecar_path) != value["sidecar_path"]:
        raise G1FlowControlError(f"{label} sidecar path differs")
    if bundle.sidecar_sha256 != _sha(value["sidecar_sha256"], label=f"{label}.sidecar_sha256"):
        raise G1FlowControlError(f"{label} sidecar SHA differs")
    if value["source_sha256"] != bundle.sidecar["source_sha256"]:
        raise G1FlowControlError(f"{label} source SHA differs")
    if value["anchor_sha256"] != bundle.sidecar["anchor_sha256"]:
        raise G1FlowControlError(f"{label} anchor SHA differs")
    expected_shapes = {key: list(map(int, bundle.tensors[key].shape)) for key in REQUIRED_TENSORS}
    if value["tensor_shapes"] != expected_shapes:
        raise G1FlowControlError(f"{label} recorded shape differs")
    return bundle


def verify_cohort_receipt(path: Path | str) -> dict[str, Any]:
    """Replay every hash and tensor transform in a published cohort receipt."""

    receipt_path = _regular_path(path, label="cohort receipt")
    receipt, _ = _read_json_file(receipt_path, label="cohort receipt")
    required_top = {
        "schema_version",
        "created_at_utc",
        "case_id",
        "anchor_kind",
        "correct_role",
        "action_family",
        "wrong_case_id",
        "wrong_action_family",
        "external_bundles",
        "generated_controls",
        "diagnostics",
        "contracts",
    }
    if set(receipt) != required_top or receipt["schema_version"] != COHORT_SCHEMA_VERSION:
        raise G1FlowControlError("cohort receipt schema/field closure differs")
    case_id = _identifier(receipt["case_id"], label="case_id")
    wrong_case_id = _identifier(receipt["wrong_case_id"], label="wrong_case_id")
    action_family = _identifier(receipt["action_family"], label="action_family")
    wrong_family = _identifier(receipt["wrong_action_family"], label="wrong_action_family")
    anchor_kind = receipt["anchor_kind"]
    if anchor_kind not in ANCHOR_KINDS or case_id == wrong_case_id or action_family == wrong_family:
        raise G1FlowControlError("cohort identity/action separation differs")
    expected_role = "real_forward" if anchor_kind == "target" else "self_generated"
    if receipt["correct_role"] != expected_role:
        raise G1FlowControlError("correct role does not match anchor kind")
    contracts = receipt["contracts"]
    expected_contracts = {
        "required_controls": [
            "zero_or_noop",
            "temporal_shuffle",
            "reverse",
            "incomplete",
            "wrong_action_energy_matched",
        ],
        "target_and_selfgen_judged_separately": True,
        "weighted_compensation_forbidden": True,
        "representation_only": True,
        "target_rgb_vae_or_clean_latent_accessed": False,
        "optimizer_created": False,
        "current_experiment_optimization_steps": 0,
        "atomic_directory_publication": True,
    }
    if contracts != expected_contracts:
        raise G1FlowControlError("cohort contracts differ")
    external_rows = receipt["external_bundles"]
    if not isinstance(external_rows, Mapping) or set(external_rows) != set(EXTERNAL_ROLES):
        raise G1FlowControlError("external bundle closure differs")
    external = {
        role: _verify_ref(external_rows[role], label=role, extractor=True)
        for role in EXTERNAL_ROLES
    }
    if len({bundle.sha256 for bundle in external.values()}) != len(EXTERNAL_ROLES):
        raise G1FlowControlError("external bundles alias")
    if any(
        external[role].sidecar["source_sha256"] != external["correct"].sidecar["source_sha256"]
        for role in ("temporal_shuffle", "reverse")
    ):
        raise G1FlowControlError("temporal controls are not source matched")
    diagnostics = receipt["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise G1FlowControlError("transform diagnostics must be an object")
    keep = diagnostics.get("incomplete_transitions_retained")
    if isinstance(keep, bool) or not isinstance(keep, int):
        raise G1FlowControlError("incomplete transition count differs")
    expected, replay = _build_transforms(external["correct"], external["wrong_action_donor"], incomplete_transitions=keep)
    if _canonical_json_bytes(dict(diagnostics)) != _canonical_json_bytes(replay):
        raise G1FlowControlError("transform diagnostics do not replay")
    generated_rows = receipt["generated_controls"]
    if not isinstance(generated_rows, Mapping) or set(generated_rows) != set(GENERATED_CONTROLS):
        raise G1FlowControlError("generated control closure differs")
    import torch
    for control_kind in GENERATED_CONTROLS:
        row = generated_rows[control_kind]
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "sidecar_path", "sidecar_sha256"}:
            raise G1FlowControlError(f"{control_kind} reference closure differs")
        bundle_path, digest, tensors, _ = _load_tensor_file(row["path"], label=control_kind)
        if digest != _sha(row["sha256"], label=f"{control_kind}.sha256"):
            raise G1FlowControlError(f"{control_kind} SHA differs")
        sidecar_path = _regular_path(row["sidecar_path"], label=f"{control_kind} sidecar")
        if sidecar_path != bundle_path.with_suffix(".json"):
            raise G1FlowControlError(f"{control_kind} sidecar path differs")
        sidecar, sidecar_sha = _read_json_file(sidecar_path, label=f"{control_kind} sidecar")
        if sidecar_sha != _sha(row["sidecar_sha256"], label=f"{control_kind}.sidecar_sha256"):
            raise G1FlowControlError(f"{control_kind} sidecar SHA differs")
        if (
            sidecar.get("schema_version") != BUNDLE_RECEIPT_SCHEMA_VERSION
            or sidecar.get("case_id") != case_id
            or sidecar.get("anchor_kind") != anchor_kind
            or sidecar.get("control_kind") != control_kind
            or sidecar.get("target_rgb_vae_or_clean_latent_accessed") is not False
            or sidecar.get("optimizer_created") is not False
            or sidecar.get("current_experiment_optimization_steps") != 0
        ):
            raise G1FlowControlError(f"{control_kind} sidecar contract differs")
        for key in REQUIRED_TENSORS:
            if not torch.equal(tensors[key], expected[control_kind][key]):
                raise G1FlowControlError(f"{control_kind}.{key} transform does not replay")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize or verify a complete G1 flow-control cohort.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize", help="create a sealed cohort directory")
    materialize.add_argument("--correct", required=True)
    materialize.add_argument("--temporal-shuffle", required=True)
    materialize.add_argument("--reverse", required=True)
    materialize.add_argument("--wrong-action", required=True)
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--case-id", required=True)
    materialize.add_argument("--anchor-kind", required=True, choices=ANCHOR_KINDS)
    materialize.add_argument("--action-family", required=True)
    materialize.add_argument("--wrong-case-id", required=True)
    materialize.add_argument("--wrong-action-family", required=True)
    materialize.add_argument("--incomplete-transitions", type=int, default=10)
    verify = subparsers.add_parser("verify", help="replay a sealed cohort receipt")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "materialize":
        receipt = materialize_cohort(
            correct_path=args.correct,
            temporal_shuffle_path=args.temporal_shuffle,
            reverse_path=args.reverse,
            wrong_action_path=args.wrong_action,
            output_dir=args.output_dir,
            case_id=args.case_id,
            anchor_kind=args.anchor_kind,
            action_family=args.action_family,
            wrong_case_id=args.wrong_case_id,
            wrong_action_family=args.wrong_action_family,
            incomplete_transitions=args.incomplete_transitions,
        )
    else:
        receipt = verify_cohort_receipt(args.receipt)
    print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

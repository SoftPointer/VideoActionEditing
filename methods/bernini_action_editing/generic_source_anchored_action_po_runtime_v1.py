#!/usr/bin/env python3
"""Strict Stage-P/Stage-O runtime for generic source-anchored editing.

This standalone continuation runtime implements ``smoke-p``, ``smoke-o``,
the combined ``smoke-po25`` P24->O1 transition, and ``resume-po40``.  It
resumes the byte-pinned R64 composite, trains one shared WORLD4=DP1xSP4
planner/operator, and never exposes self-generated RGB, latent, Gaussian,
velocity, or family identifiers to an optimizer-facing row.

Stage P consumes only frozen UMT5 states and detached reviewed ``[21,32]``
quotients.  Stage O constructs its query by frozen-VAE encoding a manifest-
bound real q0 source, forward-noising that source locally, and comparing the
block-22 adapted/noop hidden delta in the registered Phi_v1 coordinate.  The
two nuisance directions used by Phi_v1 must be supplied by a separate sealed
operator-coordinate manifest; hashes without bytes are deliberately rejected.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, NoReturn, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import clean_source_visual_context_adapter_v1 as visual  # noqa: E402
import clean_source_visual_context_stage_b_contract_v1 as old_contract  # noqa: E402
import generic_source_anchored_action_v1 as core  # noqa: E402
import inference_sigma_strata as exact40  # noqa: E402
import train_clean_source_visual_context_stage_b_v1 as native_r  # noqa: E402
from tools import generic_action_manifest_v1 as action_manifest  # noqa: E402


METHOD = "bernini-generic-source-anchored-action-v1"
PO_RUNTIME_SCHEMA = "bernini-generic-source-anchored-action-po-runtime-v1"
PO_HISTORY_SCHEMA = "bernini-generic-source-anchored-action-po-history-v1"
PO_CHECKPOINT_SCHEMA = "bernini-generic-source-anchored-action-checkpoint-v2"
PO_COMPLETION_MANIFEST_SCHEMA = (
    "bernini-generic-source-anchored-action-completion-manifest-v1"
)
PHI_OPERATOR_COORDINATE_SCHEMA = (
    "bernini-phi-v1-operator-runtime-coordinate-manifest-v1"
)
R64_CHECKPOINT_SCHEMA = "bernini-generic-source-anchored-action-checkpoint-v1"
R64_RECEIPT_SCHEMA = core.TRAINING_RECEIPT_SCHEMA
EXPECTED_R64_CHECKPOINT_SHA256 = (
    "b037496df99ea01d5a7e3fa509aac4c451806a6e47ecb7a1070529abde249726"
)
EXPECTED_R64_CARRIER_PARAMETER_SHA256 = (
    "144a13deb91bba460419de419a6dd9ac5362422d2f3230947a6f96351fa0dee3"
)
EXPECTED_R64_PLANNER_PARAMETER_SHA256 = (
    "80d440e00069741b39219ec083d53fe3711ee123c1435dfd1cd68f8b7d9cae19"
)
EXPECTED_R64_OPERATOR_PARAMETER_SHA256 = (
    "85daa966ffe57392dfc09fed7c2c46420e45025f48671a56e3576acb222eb721"
)
EXPECTED_R64_RECEIPT_SHA256 = (
    "0bcf24ce8aafabb37cf38eafe9da6b13c70043bb0f4c3146f16dc0bafd35618f"
)
EXPECTED_Q0_INFER_LORA_SHA256 = (
    "babd6d63287723ccd14b2bbe43bd4550c30b4feaa794d17c66f5a5ddefe979fe"
)
EXPECTED_DIFFUSERS_VERSION = "0.38.0"
EXPECTED_Q0_VAE_MODULE_NAME = (
    "diffusers.models.autoencoders.autoencoder_kl_wan"
)
EXPECTED_Q0_VAE_MODULE_SHA256 = (
    "836820d112a9310ece586ba9fa51d51daef04cbe866e59a673843476a4d7e087"
)
EXPECTED_Q0_VAE_MODULE_RELATIVE_PATH = Path(
    "models/autoencoders/autoencoder_kl_wan.py"
)
EXPECTED_VEOMNI_COMMIT = native_r.EXPECTED_VEOMNI_COMMIT
EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    native_r.EXPECTED_CHECKPOINT_MANIFEST_SHA256
)
ACTION_PROFILES = ("smoke-p", "smoke-o", "smoke-po25", "resume-po40")
OPERATOR_PROFILES = ("smoke-o", "smoke-po25", "resume-po40")
P_CHECKPOINT_STEPS = (12, 24)
O_CHECKPOINT_STEPS = (4, 8, 12, 16)
RAW_CODE_BYTES = core.LATENT_PHASES * core.PHASE_CODE_WIDTH * 4
RAW_P32_BYTES = core.HIDDEN_SIZE_1P3B * core.PHASE_CODE_WIDTH * 4
_SHA256_ALPHABET = frozenset("0123456789abcdef")
SMOKE_P_CHECKPOINT_NAME = "disposable_smoke_p_composite_checkpoint.pt"
SMOKE_O_CHECKPOINT_NAME = "disposable_smoke_o_composite_checkpoint.pt"
SMOKE_PO25_CHECKPOINT_NAME = "disposable_smoke_p24_o1_composite_checkpoint.pt"
FINAL_PO_CHECKPOINT_NAME = "stage_o_u016_global_u104_composite_checkpoint.pt"


class GenericActionPORuntimeError(RuntimeError):
    """Raised before accepting an ambiguous P/O update or artifact."""


def fail(message: str) -> NoReturn:
    raise GenericActionPORuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise GenericActionPORuntimeError(
            "value is not canonical finite ASCII JSON"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        fail(f"{label} must be a lowercase SHA-256")
    return value


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
        fail(f"input changed while hashing: {path}")
    return digest.hexdigest()


def read_verified_file_bytes(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, Mapping[str, Any]]:
    """Read one authority through one no-follow FD and seal those exact bytes."""

    expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
    try:
        path_before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
    except OSError as error:
        raise GenericActionPORuntimeError(
            f"cannot open single-FD {label} authority"
        ) from error
    try:
        fd_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(fd_before.st_mode)
            or stat.S_ISLNK(path_before.st_mode)
            or (path_before.st_dev, path_before.st_ino)
            != (fd_before.st_dev, fd_before.st_ino)
        ):
            fail(f"single-FD {label} path/inode authority differs")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        fd_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stat_fields_before = (
        fd_before.st_dev,
        fd_before.st_ino,
        fd_before.st_size,
        fd_before.st_mtime_ns,
        fd_before.st_ctime_ns,
    )
    stat_fields_after = (
        fd_after.st_dev,
        fd_after.st_ino,
        fd_after.st_size,
        fd_after.st_mtime_ns,
        fd_after.st_ctime_ns,
    )
    raw = b"".join(chunks)
    actual = digest.hexdigest()
    if (
        stat_fields_before != stat_fields_after
        or len(raw) != fd_before.st_size
        or actual != expected
    ):
        fail(f"single-FD {label} bytes changed or SHA-256 differs")
    identity = {
        "path": str(path),
        "file_sha256": actual,
        "byte_count": len(raw),
        "device": fd_before.st_dev,
        "inode": fd_before.st_ino,
        "mtime_ns": fd_before.st_mtime_ns,
        "ctime_ns": fd_before.st_ctime_ns,
        "single_fd_no_follow": True,
    }
    return raw, {**identity, "digest": object_sha256(identity)}


def admit_q0_vae_implementation(
    *,
    autoencoder_class: Any,
    diffusers_package: Any,
    implementation_module: Any,
) -> Mapping[str, Any]:
    """Fail closed unless q0 uses the registered exact diffusers VAE bytes."""

    package_file = getattr(diffusers_package, "__file__", None)
    package_version = getattr(diffusers_package, "__version__", None)
    module_file = getattr(implementation_module, "__file__", None)
    module_name = getattr(implementation_module, "__name__", None)
    module_spec = getattr(implementation_module, "__spec__", None)
    spec_name = getattr(module_spec, "name", None)
    spec_origin = getattr(module_spec, "origin", None)
    if any(
        type(value) is not str or not value
        for value in (package_file, module_file, spec_origin)
    ):
        fail("q0 VAE package/module origin metadata is absent")
    try:
        package_origin = Path(package_file).resolve(strict=True)
        observed_module_origin = Path(module_file).resolve(strict=True)
        observed_spec_origin = Path(spec_origin).resolve(strict=True)
        package_root = package_origin.parent
        site_packages_root = package_root.parent
        expected_module_origin = (
            package_root / EXPECTED_Q0_VAE_MODULE_RELATIVE_PATH
        ).resolve(strict=True)
    except OSError as error:
        raise GenericActionPORuntimeError(
            "q0 VAE package/module origin is unavailable"
        ) from error
    if (
        package_version != EXPECTED_DIFFUSERS_VERSION
        or package_origin.name != "__init__.py"
        or package_root.name != "diffusers"
        or site_packages_root.name != "site-packages"
        or getattr(autoencoder_class, "__name__", None) != "AutoencoderKLWan"
        or getattr(autoencoder_class, "__module__", None)
        != EXPECTED_Q0_VAE_MODULE_NAME
        or module_name != EXPECTED_Q0_VAE_MODULE_NAME
        or spec_name != EXPECTED_Q0_VAE_MODULE_NAME
        or observed_module_origin != expected_module_origin
        or observed_spec_origin != expected_module_origin
    ):
        fail("q0 VAE module name/version/exact site-packages origin differs")
    _, file_identity = read_verified_file_bytes(
        expected_module_origin,
        expected_sha256=EXPECTED_Q0_VAE_MODULE_SHA256,
        label="registered q0 AutoencoderKLWan implementation",
    )
    expected = {
        "diffusers_version": EXPECTED_DIFFUSERS_VERSION,
        "module_name": EXPECTED_Q0_VAE_MODULE_NAME,
        "module_origin": str(expected_module_origin),
        "site_packages_root": str(site_packages_root),
        "module_sha256": EXPECTED_Q0_VAE_MODULE_SHA256,
    }
    observed = {
        "diffusers_version": package_version,
        "class_name": autoencoder_class.__name__,
        "class_module": autoencoder_class.__module__,
        "module_name": module_name,
        "module_spec_name": spec_name,
        "module_file_origin": str(observed_module_origin),
        "module_spec_origin": str(observed_spec_origin),
        "module_sha256": file_identity["file_sha256"],
        "single_fd_file_identity": dict(file_identity),
    }
    value = {
        "expected": expected,
        "observed": observed,
        "exact_registered_implementation": True,
    }
    return {**value, "digest": object_sha256(value)}


def _plain_file(value: Any, *, expected_sha256: Optional[str], label: str) -> Path:
    if type(value) is not str:
        fail(f"{label} path must be text")
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        fail(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as error:
        raise GenericActionPORuntimeError(f"{label} is unavailable") from error
    if resolved != requested or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be one canonical plain file")
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, label=f"expected {label} SHA-256")
        if file_sha256(resolved) != expected:
            fail(f"{label} SHA-256 differs")
    return resolved


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise GenericActionPORuntimeError(f"cannot decode {label}") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    return value


def _load_json_bytes(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise GenericActionPORuntimeError(f"cannot decode {label} bytes") from error
    if type(value) is not dict:
        fail(f"{label} root must be an object")
    return value


def _closed(value: Any, fields: Sequence[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        fail(f"{label} field closure differs")
    return value


def _verify_embedded_digest(
    value: Mapping[str, Any], *, field: str, label: str
) -> None:
    declared = _sha256(value.get(field), label=f"{label}.{field}")
    unsigned = dict(value)
    del unsigned[field]
    if object_sha256(unsigned) != declared:
        fail(f"{label} digest differs")


def _raw_tensor_binding(value: Any, *, label: str) -> Mapping[str, Any]:
    row = _closed(
        value,
        ("path", "raw_sha256", "dtype", "byte_order", "shape"),
        label=label,
    )
    if (
        row["dtype"] != "float32"
        or row["byte_order"] != "little"
        or row["shape"] != [core.LATENT_PHASES, core.PHASE_CODE_WIDTH]
    ):
        fail(f"{label} tensor geometry differs")
    path = _plain_file(
        row["path"], expected_sha256=row["raw_sha256"], label=label
    )
    if path.stat().st_size != RAW_CODE_BYTES:
        fail(f"{label} tensor byte count differs")
    return {**dict(row), "path": str(path)}


@dataclass(frozen=True)
class PlannerRow:
    row_id: str
    source_iid: str
    seed: int
    branch: str
    instruction: str
    instruction_sha256: str
    quotient_path: str
    quotient_sha256: str

    def optimizer_payload(self, quotient_tensor: Any) -> Mapping[str, Any]:
        value = {
            "row_id": self.row_id,
            "source_iid": self.source_iid,
            "seed": self.seed,
            "branch": self.branch,
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
            "quotient_sha256": self.quotient_sha256,
            "quotient_tensor": quotient_tensor,
        }
        core.assert_optimizer_payload_safe(value)
        return value


@dataclass(frozen=True)
class OperatorRow(PlannerRow):
    source_video_path: str
    source_video_sha256: str
    camera_nuisance_path: str
    camera_nuisance_sha256: str
    appearance_nuisance_path: str
    appearance_nuisance_sha256: str

    def optimizer_payload(
        self,
        quotient_tensor: Any,
        camera_nuisance: Any,
        appearance_nuisance: Any,
    ) -> Mapping[str, Any]:
        value = dict(super().optimizer_payload(quotient_tensor))
        # The two reviewed quotient-space nuisance coordinates are permitted;
        # they are neither generated media nor editor targets.  Keep their
        # tensor objects outside core.assert_optimizer_payload_safe's one-
        # teacher-tensor API and bind only their hashes in the row payload.
        value.update(
            {
                "real_source_video_path": self.source_video_path,
                "real_source_video_sha256": self.source_video_sha256,
                "camera_nuisance_sha256": self.camera_nuisance_sha256,
                "appearance_nuisance_sha256": self.appearance_nuisance_sha256,
            }
        )
        if camera_nuisance is None or appearance_nuisance is None:
            fail("operator nuisance tensor is absent")
        return value


@dataclass(frozen=True)
class InstructionViews:
    """One UMT5 encode with distinct Planner and official Operator views."""

    planner_tokens: Any
    operator_text_embs: Any
    operator_text_lens: tuple[int, ...]


@dataclass(frozen=True)
class OperatorCoordinateAdmission:
    teacher: Any
    camera: Any
    appearance: Any
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ActionAuthority:
    representation_path: str
    representation_file_sha256: str
    representation_digest: str
    pair_path: str
    pair_file_sha256: str
    pair_digest: str
    row_order_sha256: str
    p32_path: str
    p32_raw_sha256: str
    planner_rows: tuple[PlannerRow, ...]
    operator_rows: tuple[OperatorRow, ...]
    coordinate_manifest_path: Optional[str]
    coordinate_manifest_file_sha256: Optional[str]
    coordinate_manifest_digest: Optional[str]

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "representation_manifest": {
                "path": self.representation_path,
                "file_sha256": self.representation_file_sha256,
                "manifest_digest": self.representation_digest,
            },
            "source_pair_manifest": {
                "path": self.pair_path,
                "file_sha256": self.pair_file_sha256,
                "manifest_digest": self.pair_digest,
            },
            "phi_operator_coordinate_manifest": (
                None
                if self.coordinate_manifest_path is None
                else {
                    "path": self.coordinate_manifest_path,
                    "file_sha256": self.coordinate_manifest_file_sha256,
                    "manifest_digest": self.coordinate_manifest_digest,
                }
            ),
            "row_order_sha256": self.row_order_sha256,
            "p32_path": self.p32_path,
            "p32_raw_sha256": self.p32_raw_sha256,
            "planner_optimizer_rows": len(self.planner_rows),
            "operator_optimizer_rows": len(self.operator_rows),
            "generated_media_bytes_read": False,
            "generated_latent_noise_velocity_read": False,
            "action_family_identifier_consumed": False,
        }
        return {**value, "digest": object_sha256(value)}


def _read_sidecar_nuisance_hashes(rep_row: Mapping[str, Any]) -> tuple[str, str]:
    reference = _closed(
        rep_row.get("sidecar_receipt"),
        ("path", "file_sha256"),
        label="representation sidecar reference",
    )
    path = _plain_file(
        reference["path"],
        expected_sha256=reference["file_sha256"],
        label="representation sidecar receipt",
    )
    sidecar = _load_json(path, label="representation sidecar receipt")
    nuisance = sidecar.get("nuisance_projection")
    if not isinstance(nuisance, Mapping):
        fail("representation sidecar has no nuisance projection")
    return (
        _sha256(nuisance.get("camera_raw_sha256"), label="camera nuisance SHA"),
        _sha256(
            nuisance.get("appearance_raw_sha256"),
            label="appearance nuisance SHA",
        ),
    )


def _read_sidecar_p32_binding(rep_row: Mapping[str, Any]) -> tuple[str, str]:
    reference = _closed(
        rep_row.get("sidecar_receipt"),
        ("path", "file_sha256"),
        label="representation sidecar reference",
    )
    path = _plain_file(
        reference["path"],
        expected_sha256=reference["file_sha256"],
        label="representation sidecar receipt",
    )
    sidecar = _load_json(path, label="representation sidecar receipt")
    phi = sidecar.get("phi_v1")
    if not isinstance(phi, Mapping):
        fail("representation sidecar has no Phi_v1 coordinate")
    p32_path = _plain_file(
        phi.get("p32_raw_path"),
        expected_sha256=phi.get("p32_raw_sha256"),
        label="sealed Phi_v1 P32",
    )
    if p32_path.stat().st_size != RAW_P32_BYTES:
        fail("sealed Phi_v1 P32 byte count differs")
    return str(p32_path), _sha256(
        phi.get("p32_raw_sha256"), label="sealed Phi_v1 P32 SHA"
    )


def _validate_coordinate_manifest(
    *,
    path: Path,
    representation_sha256: str,
    pair_sha256: str,
    rep_rows_by_id: Mapping[str, Mapping[str, Any]],
    operator_pair_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]:
    value = _load_json(path, label="Phi operator coordinate manifest")
    _closed(
        value,
        (
            "schema_version",
            "representation_manifest_file_sha256",
            "source_pair_manifest_file_sha256",
            "phi_v1",
            "counts",
            "generated_media_is_optimizer_input_or_target",
            "optimizer_authorized",
            "rows",
            "manifest_digest",
        ),
        label="Phi operator coordinate manifest",
    )
    _verify_embedded_digest(
        value, field="manifest_digest", label="Phi operator coordinate manifest"
    )
    if (
        value["schema_version"] != PHI_OPERATOR_COORDINATE_SCHEMA
        or value["representation_manifest_file_sha256"] != representation_sha256
        or value["source_pair_manifest_file_sha256"] != pair_sha256
        or value["counts"] != {"operator_optimizer": core.STAGE_UPDATES["O"]}
        or value["generated_media_is_optimizer_input_or_target"] is not False
        or value["optimizer_authorized"] is not True
    ):
        fail("Phi operator coordinate authority differs")
    phi = _closed(
        value["phi_v1"],
        (
            "block_index",
            "teacher_exact40_index",
            "sp_world",
            "p32_seed",
            "p32_raw_sha256",
            "nuisance_order",
        ),
        label="Phi operator coordinate",
    )
    if phi != {
        "block_index": core.PHI_BLOCK_INDEX,
        "teacher_exact40_index": core.PHI_TEACHER_SCHEDULE_INDEX,
        "sp_world": core.SP_SIZE,
        "p32_seed": core.P32_SEED,
        "p32_raw_sha256": next(iter(rep_rows_by_id.values()))["_p32_raw_sha256"],
        "nuisance_order": [
            "camera_only",
            "appearance_only_gram_schmidt_off_camera",
        ],
    }:
        fail("Phi operator coordinate definition differs")
    rows = value["rows"]
    expected_ids = [row["row_id"] for row in operator_pair_rows]
    if (
        type(rows) is not list
        or len(rows) != core.STAGE_UPDATES["O"]
        or [row.get("row_id") for row in rows] != expected_ids
    ):
        fail("Phi operator coordinate row order/closure differs")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _closed(
            row,
            (
                "row_id",
                "representation_tensor_sha256",
                "camera_nuisance_tensor",
                "appearance_nuisance_tensor",
            ),
            label="Phi operator coordinate row",
        )
        row_id = row["row_id"]
        rep = rep_rows_by_id.get(row_id)
        if rep is None or row_id in result:
            fail("Phi operator coordinate row identity differs")
        if row["representation_tensor_sha256"] != rep["quotient_tensor"]["raw_sha256"]:
            fail("Phi operator coordinate quotient binding differs")
        camera = _raw_tensor_binding(
            row["camera_nuisance_tensor"], label=f"{row_id} camera nuisance"
        )
        appearance = _raw_tensor_binding(
            row["appearance_nuisance_tensor"],
            label=f"{row_id} appearance nuisance",
        )
        expected_camera, expected_appearance = _read_sidecar_nuisance_hashes(rep)
        if (
            camera["raw_sha256"] != expected_camera
            or appearance["raw_sha256"] != expected_appearance
        ):
            fail("Phi operator coordinate/sidecar nuisance hash binding differs")
        result[row_id] = {"camera": camera, "appearance": appearance}
    return value, result


def load_action_authority(args: argparse.Namespace) -> ActionAuthority:
    representation_path = _plain_file(
        args.representation_manifest,
        expected_sha256=args.expected_representation_manifest_sha256,
        label="representation manifest",
    )
    pair_path = _plain_file(
        args.source_pair_manifest,
        expected_sha256=args.expected_source_pair_manifest_sha256,
        label="source-pair manifest",
    )
    representation_sha = file_sha256(representation_path)
    pair_sha = file_sha256(pair_path)
    try:
        representation, pairs = action_manifest.validate_manifest_pair(
            representation_path, pair_path
        )
    except Exception as error:
        raise GenericActionPORuntimeError(
            f"action manifest authority rejected: {type(error).__name__}: {error}"
        ) from error
    rep_rows = representation["rows"]
    pair_rows = pairs["rows"]
    rep_rows_by_id = {row["row_id"]: dict(row) for row in rep_rows}
    if len(rep_rows_by_id) != len(rep_rows):
        fail("representation row identities are not unique")
    for row in rep_rows_by_id.values():
        row["_p32_raw_sha256"] = representation["phi_v1_p32_raw_sha256"]
    p32_path, sidecar_p32_sha = _read_sidecar_p32_binding(rep_rows[0])
    if sidecar_p32_sha != representation["phi_v1_p32_raw_sha256"]:
        fail("representation manifest/sidecar P32 binding differs")
    planner_source = [row for row in rep_rows if row["planner_optimizer_eligible"]]
    operator_pair = [row for row in pair_rows if row["operator_optimizer_eligible"]]
    if (
        len(planner_source) != core.STAGE_UPDATES["P"]
        or len(operator_pair) != core.STAGE_UPDATES["O"]
        or any(row["analysis_split"] != "fit" for row in planner_source)
        or {row["branch"] for row in planner_source}
        != {"action", "reverse", "incomplete"}
        or any(
            row["analysis_split"] != "fit"
            or row["branch"] not in {"action", "incomplete"}
            or row["start_state"] != "q0"
            or row["real_source_available"] is not True
            for row in operator_pair
        )
    ):
        fail("P24/O16 manifest eligibility differs")
    coordinate_value: Optional[Mapping[str, Any]] = None
    coordinates: Mapping[str, Mapping[str, Any]] = {}
    coordinate_path: Optional[Path] = None
    coordinate_sha: Optional[str] = None
    if args.execution_profile in OPERATOR_PROFILES:
        coordinate_path = _plain_file(
            args.phi_operator_coordinate_manifest,
            expected_sha256=args.expected_phi_operator_coordinate_manifest_sha256,
            label="Phi operator coordinate manifest",
        )
        coordinate_sha = file_sha256(coordinate_path)
        coordinate_value, coordinates = _validate_coordinate_manifest(
            path=coordinate_path,
            representation_sha256=representation_sha,
            pair_sha256=pair_sha,
            rep_rows_by_id=rep_rows_by_id,
            operator_pair_rows=operator_pair,
        )
    planner_rows = tuple(
        PlannerRow(
            row_id=row["row_id"],
            source_iid=row["source_iid"],
            seed=row["seed"],
            branch=row["branch"],
            instruction=row["instruction"]["text"],
            instruction_sha256=row["instruction"]["utf8_sha256"],
            quotient_path=row["quotient_tensor"]["path"],
            quotient_sha256=row["quotient_tensor"]["raw_sha256"],
        )
        for row in planner_source
    )
    operator_rows_list: list[OperatorRow] = []
    for pair in operator_pair:
        rep = rep_rows_by_id[pair["row_id"]]
        coordinate = coordinates.get(pair["row_id"])
        if args.execution_profile in OPERATOR_PROFILES and coordinate is None:
            fail("operator row has no runtime nuisance coordinate")
        if coordinate is None:
            continue
        operator_rows_list.append(
            OperatorRow(
                row_id=rep["row_id"],
                source_iid=rep["source_iid"],
                seed=rep["seed"],
                branch=rep["branch"],
                instruction=rep["instruction"]["text"],
                instruction_sha256=rep["instruction"]["utf8_sha256"],
                quotient_path=rep["quotient_tensor"]["path"],
                quotient_sha256=rep["quotient_tensor"]["raw_sha256"],
                source_video_path=pair["real_source_video_path"],
                source_video_sha256=pair["real_source_video_sha256"],
                camera_nuisance_path=coordinate["camera"]["path"],
                camera_nuisance_sha256=coordinate["camera"]["raw_sha256"],
                appearance_nuisance_path=coordinate["appearance"]["path"],
                appearance_nuisance_sha256=coordinate["appearance"]["raw_sha256"],
            )
        )
    if args.execution_profile in OPERATOR_PROFILES and len(operator_rows_list) != 16:
        fail("operator runtime row closure differs")
    return ActionAuthority(
        representation_path=str(representation_path),
        representation_file_sha256=representation_sha,
        representation_digest=representation["manifest_digest"],
        pair_path=str(pair_path),
        pair_file_sha256=pair_sha,
        pair_digest=pairs["manifest_digest"],
        row_order_sha256=representation["row_order_sha256"],
        p32_path=p32_path,
        p32_raw_sha256=representation["phi_v1_p32_raw_sha256"],
        planner_rows=planner_rows,
        operator_rows=tuple(operator_rows_list),
        coordinate_manifest_path=(
            None if coordinate_path is None else str(coordinate_path)
        ),
        coordinate_manifest_file_sha256=coordinate_sha,
        coordinate_manifest_digest=(
            None if coordinate_value is None else coordinate_value["manifest_digest"]
        ),
    )


def read_f32le_21x32(path_value: str, expected_sha256: str) -> Any:
    import torch

    if sys.byteorder != "little":
        fail("P/O runtime requires a little-endian host")
    path = _plain_file(
        path_value, expected_sha256=None, label="quotient coordinate"
    )
    raw, _ = read_verified_file_bytes(
        path,
        expected_sha256=expected_sha256,
        label="quotient coordinate",
    )
    if len(raw) != RAW_CODE_BYTES:
        fail("quotient coordinate byte count differs")
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
        1, core.LATENT_PHASES, core.PHASE_CODE_WIDTH
    )
    if not bool(torch.isfinite(tensor).all().item()):
        fail("quotient coordinate is non-finite")
    return tensor.contiguous()


def read_f32le_p32(path_value: str, expected_sha256: str) -> Any:
    import torch

    if sys.byteorder != "little":
        fail("P/O runtime requires a little-endian host")
    path = _plain_file(
        path_value, expected_sha256=None, label="sealed Phi_v1 P32"
    )
    raw, _ = read_verified_file_bytes(
        path,
        expected_sha256=expected_sha256,
        label="sealed Phi_v1 P32",
    )
    if len(raw) != RAW_P32_BYTES:
        fail("sealed Phi_v1 P32 byte count differs")
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone().reshape(
        core.HIDDEN_SIZE_1P3B, core.PHASE_CODE_WIDTH
    )
    gram = tensor.T @ tensor
    identity = torch.eye(core.PHASE_CODE_WIDTH, dtype=torch.float32)
    if (
        not bool(torch.isfinite(tensor).all().item())
        or not torch.allclose(gram, identity, atol=2.0e-4, rtol=0.0)
    ):
        fail("sealed Phi_v1 P32 is non-finite/non-orthonormal")
    return tensor.contiguous()


def p32_raw_sha256() -> str:
    import torch

    value = core.fixed_p32().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(value).hexdigest()


def admit_planner_teachers(
    rows: Sequence[PlannerRow],
) -> Mapping[str, Any]:
    if len(rows) != core.STAGE_UPDATES["P"]:
        fail("pre-optimizer Planner teacher closure must contain P24")
    admitted: dict[str, Any] = {}
    for row in rows:
        if row.row_id in admitted:
            fail("pre-optimizer Planner teacher row is duplicated")
        teacher = read_f32le_21x32(
            row.quotient_path, row.quotient_sha256
        ).detach()
        _validate_nonnoop_teacher(
            teacher, label=f"pre-optimizer Stage P row {row.row_id}"
        )
        admitted[row.row_id] = teacher
    return admitted


def admit_operator_coordinates(
    rows: Sequence[OperatorRow],
) -> tuple[Mapping[str, OperatorCoordinateAdmission], Mapping[str, Any]]:
    """Seal all O16 tensors and prove the registered nuisance quotient math."""

    import torch

    if len(rows) != core.STAGE_UPDATES["O"]:
        fail("pre-optimizer Operator coordinate closure must contain O16")
    admitted: dict[str, OperatorCoordinateAdmission] = {}
    row_receipts: list[Mapping[str, Any]] = []
    for row in rows:
        if row.row_id in admitted:
            fail("pre-optimizer Operator coordinate row is duplicated")
        teacher = read_f32le_21x32(
            row.quotient_path, row.quotient_sha256
        ).detach()
        camera = read_f32le_21x32(
            row.camera_nuisance_path, row.camera_nuisance_sha256
        ).detach()
        appearance = read_f32le_21x32(
            row.appearance_nuisance_path, row.appearance_nuisance_sha256
        ).detach()
        _validate_nonnoop_teacher(
            teacher, label=f"pre-optimizer Stage O row {row.row_id}"
        )
        zero = torch.zeros_like(camera[:, 0, :])
        camera_dc = camera[:, 1:, :].mean(dim=1)
        appearance_dc = appearance[:, 1:, :].mean(dim=1)
        if (
            not torch.equal(camera[:, 0, :], zero)
            or not torch.equal(appearance[:, 0, :], zero)
            or float(camera_dc.abs().max().item()) > 2.0e-5
            or float(appearance_dc.abs().max().item()) > 2.0e-5
        ):
            fail("pre-optimizer nuisance phase0/temporal-DC closure differs")
        try:
            camera_direction, appearance_direction = core._gram_schmidt_nuisances(
                teacher, camera, appearance
            )
        except core.GenericSourceAnchoredActionError as error:
            raise GenericActionPORuntimeError(
                "pre-optimizer nuisance Gram-Schmidt closure differs"
            ) from error
        if camera_direction is None or appearance_direction is None:
            fail("pre-optimizer nuisance Gram-Schmidt directions are absent")
        teacher_flat = teacher.float().reshape(-1)
        camera_flat = camera_direction.float().reshape(-1)
        appearance_flat = appearance_direction.float().reshape(-1)
        camera_row_residual = abs(float((teacher_flat @ camera_flat).item()))
        appearance_row_residual = abs(
            float((teacher_flat @ appearance_flat).item())
        )
        nuisance_cross_residual = abs(
            float((camera_flat @ appearance_flat).item())
        )
        projected = core._project_off(teacher, camera_direction)
        projected = core._project_off(projected, appearance_direction)
        projection_fixed_point_residual = float(
            (projected - teacher).float().norm().item()
        )
        residuals = (
            camera_row_residual,
            appearance_row_residual,
            nuisance_cross_residual,
            projection_fixed_point_residual,
        )
        if any(not math.isfinite(value) or value > 5.0e-5 for value in residuals):
            fail("pre-optimizer nuisance/teacher row residual closure differs")
        receipt = {
            "row_id": row.row_id,
            "quotient_tensor_sha256": row.quotient_sha256,
            "camera_nuisance_sha256": row.camera_nuisance_sha256,
            "appearance_nuisance_sha256": row.appearance_nuisance_sha256,
            "camera_norm": float(camera.float().norm().item()),
            "appearance_norm": float(appearance.float().norm().item()),
            "camera_direction_norm": float(camera_flat.norm().item()),
            "appearance_after_gram_schmidt_norm": float(
                appearance_flat.norm().item()
            ),
            "camera_teacher_row_residual": camera_row_residual,
            "appearance_teacher_row_residual": appearance_row_residual,
            "camera_appearance_orthogonality_residual": nuisance_cross_residual,
            "teacher_projection_fixed_point_l2_residual": (
                projection_fixed_point_residual
            ),
            "residual_limit": 5.0e-5,
            "phase0_exact_zero": True,
            "temporal_dc_limit": 2.0e-5,
            "finite_non_degenerate": True,
            "admitted_before_optimizer_construction": True,
        }
        sealed_receipt = {**receipt, "digest": object_sha256(receipt)}
        admitted[row.row_id] = OperatorCoordinateAdmission(
            teacher=teacher,
            camera=camera,
            appearance=appearance,
            receipt=sealed_receipt,
        )
        row_receipts.append(sealed_receipt)
    closure = {
        "row_count": len(row_receipts),
        "row_ids": [row["row_id"] for row in row_receipts],
        "row_receipt_digests": [row["digest"] for row in row_receipts],
        "all_o16_admitted_before_optimizer": True,
    }
    return admitted, {**closure, "digest": object_sha256(closure)}


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().contiguous().reshape(-1)
    return value.view(torch.uint8).cpu().numpy().tobytes(order="C")


def named_parameters_sha256(named: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named:
        tensor = parameter.detach().contiguous()
        metadata = canonical_json_bytes(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _parameter_consensus(
    named: Sequence[tuple[str, Any]], *, world_group: Any, label: str
) -> str:
    import torch.distributed as dist

    digest = named_parameters_sha256(named)
    gathered: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(gathered, digest, group=world_group)
    if gathered != [digest] * core.WORLD_SIZE:
        fail(f"{label} parameters differ across WORLD4")
    return digest


def _rank0_call(
    *, rank: int, world_group: Any, label: str, callback: Any
) -> Any:
    import torch.distributed as dist

    box: list[Any] = [None]
    if rank == 0:
        try:
            box[0] = {"ok": True, "result": callback()}
        except Exception as error:  # noqa: BLE001 - fail-closed rank boundary
            box[0] = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
    dist.broadcast_object_list(box, src=0, group=world_group)
    result = box[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        fail(f"rank0 {label} failed: {result!r}")
    return result.get("result")


def _load_r64_authority(
    args: argparse.Namespace,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    import torch

    checkpoint_path = _plain_file(
        args.resume_checkpoint,
        expected_sha256=None,
        label="R64 composite checkpoint",
    )
    if args.expected_resume_checkpoint_sha256 != EXPECTED_R64_CHECKPOINT_SHA256:
        fail("P/O must resume the one completed frozen R64 checkpoint")
    receipt_path = _plain_file(
        args.resume_receipt,
        expected_sha256=None,
        label="R64 run receipt",
    )
    checkpoint_bytes, checkpoint_file_identity = read_verified_file_bytes(
        checkpoint_path,
        expected_sha256=args.expected_resume_checkpoint_sha256,
        label="R64 composite checkpoint",
    )
    receipt_bytes, receipt_file_identity = read_verified_file_bytes(
        receipt_path,
        expected_sha256=args.expected_resume_receipt_sha256,
        label="R64 run receipt",
    )
    receipt = _load_json_bytes(receipt_bytes, label="R64 run receipt")
    unsigned_receipt = dict(receipt)
    declared_receipt_digest = unsigned_receipt.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != R64_RECEIPT_SCHEMA
        or receipt.get("method") != METHOD
        or receipt.get("complete") is not True
        or receipt.get("execution_profile") != "stage-r64"
        or receipt.get("stage_r_complete") is not True
        or receipt.get("stage_r_updates") != core.STAGE_UPDATES["R"]
        or receipt.get("planner_updates") != 0
        or receipt.get("operator_updates") != 0
        or receipt.get("resume_po40_authorized") is not True
        or receipt.get("complete_action_result") is not False
        or receipt.get("checkpoint", {}).get("file_sha256")
        != EXPECTED_R64_CHECKPOINT_SHA256
        or object_sha256(unsigned_receipt) != declared_receipt_digest
    ):
        fail("R64 run receipt authority differs")
    try:
        value = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        value = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    except Exception as error:
        raise GenericActionPORuntimeError("cannot load pinned R64 checkpoint") from error
    if not isinstance(value, Mapping):
        fail("R64 checkpoint root must be an object")
    completed = value.get("completed_stage_updates")
    final_sha = value.get("final_component_sha256")
    pair = value.get("pair_invariants")
    if (
        value.get("schema_version") != R64_CHECKPOINT_SCHEMA
        or value.get("method") != METHOD
        or value.get("experiment") != "joint_source_anchored_v1"
        or value.get("execution_profile") != "stage-r64"
        or value.get("completed_stages") != ["R"]
        or value.get("incomplete_stages") != []
        or completed != {"R": core.STAGE_UPDATES["R"]}
        or value.get("resume_po40_authorized") is not True
        or value.get("complete_action_result") is not False
        or not isinstance(final_sha, Mapping)
        or final_sha.get("R") != EXPECTED_R64_CARRIER_PARAMETER_SHA256
        or final_sha.get("P") != EXPECTED_R64_PLANNER_PARAMETER_SHA256
        or final_sha.get("O") != EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        or not isinstance(pair, Mapping)
        or pair.get("source_manifest_sha256")
        != core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256
        or pair.get("representation_manifest_sha256") is not None
        or pair.get("source_pair_manifest_sha256") is not None
        or pair.get("gaussian_base_seed") != args.seed
        or pair.get("checkpoint_tree_sha256")
        != args.expected_checkpoint_tree_sha256
        or pair.get("checkpoint_content_manifest_sha256")
        != args.expected_checkpoint_content_manifest_sha256
        or pair.get("planner_initial_sha256")
        != EXPECTED_R64_PLANNER_PARAMETER_SHA256
        or pair.get("operator_initial_sha256")
        != EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        or pair.get("r_sigma_mapping") != list(core.fixed_sigma_schedule("R"))
        or pair.get("o_sigma_mapping") != list(core.fixed_sigma_schedule("O"))
        or pair.get("optimizer")
        != {
            "name": "AdamW",
            "learning_rate": core.DEFAULT_LEARNING_RATE,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "max_grad_norm": core.DEFAULT_MAX_GRAD_NORM,
        }
    ):
        fail("R64 composite checkpoint contract differs")
    byte_authority = {
        "checkpoint": dict(checkpoint_file_identity),
        "receipt": dict(receipt_file_identity),
        "torch_load_input": "io.BytesIO_of_single_fd_verified_checkpoint_bytes",
    }
    return checkpoint_path, value, receipt, byte_authority


def _load_component_state_strict(
    composite: core.CompositeHandle,
    checkpoint: Mapping[str, Any],
    *,
    world_group: Any,
) -> Mapping[str, str]:
    import torch
    import torch.distributed as dist

    state = checkpoint.get("component_state")
    groups = composite.named_parameter_groups()
    if not isinstance(state, Mapping) or set(state) != {"R", "P", "O"}:
        fail("R64 component-state stage closure differs")
    expected_sha = {
        "R": EXPECTED_R64_CARRIER_PARAMETER_SHA256,
        "P": EXPECTED_R64_PLANNER_PARAMETER_SHA256,
        "O": EXPECTED_R64_OPERATOR_PARAMETER_SHA256,
    }
    loaded: dict[str, str] = {}
    for stage in ("R", "P", "O"):
        stage_state = state[stage]
        rows = groups[stage]
        if not isinstance(stage_state, Mapping) or set(stage_state) != {
            name for name, _ in rows
        }:
            fail(f"R64 {stage} component key closure differs")
        for name, parameter in rows:
            tensor = stage_state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tuple(tensor.shape) != tuple(parameter.shape)
                or tensor.requires_grad
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                fail(f"R64 {stage}:{name} tensor contract differs")
            parameter.data.copy_(tensor.to(parameter.device))
            dist.broadcast(parameter.data, src=0, group=world_group)
        loaded[stage] = _parameter_consensus(
            rows, world_group=world_group, label=f"loaded R64 {stage}"
        )
        if loaded[stage] != expected_sha[stage]:
            fail(f"loaded R64 {stage} parameter SHA differs")
    return loaded


def _load_optimizer_state_strict(
    controller: core.StageOptimizerController,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    state = checkpoint.get("optimizer_state")
    if not isinstance(state, Mapping):
        fail("R64 optimizer state is absent")
    groups = state.get("param_groups")
    if (
        not isinstance(groups, list)
        or [group.get("name") for group in groups] != ["R", "P", "O"]
        or [float(group.get("lr", -1.0)) for group in groups]
        != [core.DEFAULT_LEARNING_RATE, 0.0, 0.0]
        or any(
            tuple(group.get("betas", ())) != (0.9, 0.95)
            or float(group.get("eps", -1.0)) != 1.0e-8
            or float(group.get("weight_decay", -1.0)) != 0.0
            for group in groups
        )
    ):
        fail("R64 optimizer parameter-group contract differs")
    try:
        controller.optimizer.load_state_dict(copy.deepcopy(state))
    except Exception as error:
        raise GenericActionPORuntimeError(
            "R64 optimizer state does not load into the exact composite"
        ) from error
    loaded = controller.optimizer.state_dict()
    if [group.get("name") for group in loaded["param_groups"]] != ["R", "P", "O"]:
        fail("loaded optimizer group identity differs")
    return {
        "groups": [group["name"] for group in loaded["param_groups"]],
        "state_entry_count": len(loaded["state"]),
        "stage_r_state_present": bool(loaded["state"]),
        "planner_operator_state_initially_absent": all(
            not controller.optimizer.state.get(parameter)
            for stage in ("P", "O")
            for _, parameter in controller.handle.named_parameter_groups()[stage]
        ),
    }


def _component_cpu_state(
    handle: core.CompositeHandle,
) -> Mapping[str, Mapping[str, Any]]:
    return {
        stage: {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in rows
        }
        for stage, rows in handle.named_parameter_groups().items()
    }


def _optimizer_state_keys(controller: core.StageOptimizerController) -> Mapping[str, Any]:
    groups = controller.handle.named_parameter_groups()
    value: dict[str, Any] = {}
    for stage, rows in groups.items():
        names = []
        for name, parameter in rows:
            if controller.optimizer.state.get(parameter):
                names.append(name)
        value[stage] = {
            "state_parameter_names": names,
            "state_parameter_count": len(names),
        }
    return value


def _rollback_active_step(
    *,
    active: Sequence[tuple[str, Any]],
    optimizer: Any,
    parameter_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
) -> None:
    for name, parameter in active:
        if name not in parameter_state:
            fail("rollback active parameter key is absent")
        parameter.data.copy_(parameter_state[name].to(parameter.device))
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def _snapshot_active_step(
    active: Sequence[tuple[str, Any]], optimizer: Any
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    return (
        {
            name: parameter.detach().cpu().contiguous().clone()
            for name, parameter in active
        },
        copy.deepcopy(optimizer.state_dict()),
    )


def _noise_seed(base_seed: int, row_id: str, schedule_index: int) -> int:
    if type(base_seed) is not int or not 0 <= base_seed < 2**63:
        fail("Gaussian base seed differs")
    payload = (
        f"{base_seed}\0generic-source-anchored-action-v1\0stage-o\0"
        f"{row_id}\0{schedule_index}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _o_coordinate(position: int) -> Any:
    schedule_index = core.fixed_sigma_schedule("O")[position]
    return old_contract.Exact80Coordinate(
        optimizer_step=position + 1,
        checkpoint_interval=0,
        step_in_checkpoint_interval=position,
        microbatch_index=0,
        interval_micro_ordinal=position,
        interval_schedule_cycle=0,
        schedule_index=schedule_index,
        timestep=exact40.PINNED_TIMESTEPS[schedule_index],
        sigma=exact40.PINNED_POSITIVE_SIGMAS[schedule_index],
        sigma_float32_be_hex=exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
            schedule_index
        ],
    )


def _prepare_hashed_source_snapshot(
    path: Path, *, expected_sha256: str
) -> tuple[Any, Mapping[str, Any], str]:
    import infer_lora as inference_source

    source_bytes, file_identity = read_verified_file_bytes(
        path,
        expected_sha256=expected_sha256,
        label="real q0 source video",
    )
    with tempfile.TemporaryDirectory(prefix="generic-action-q0-snapshot-") as root:
        snapshot = Path(root) / "source.mp4"
        descriptor = os.open(
            snapshot,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        try:
            view = memoryview(source_bytes)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    fail("private q0 source snapshot write was incomplete")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if snapshot.stat().st_mode & 0o777 != 0o400:
            fail("private q0 source snapshot mode differs")
        if file_sha256(snapshot) != expected_sha256:
            fail("private q0 source snapshot digest differs")
        pixels, metadata = inference_source.prepare_exact_source(snapshot)
    receipt = {
        **dict(metadata),
        "decoded_from_private_byte_snapshot": True,
        "source_file_identity": dict(file_identity),
        "source_read_once_single_fd": True,
        "private_snapshot_created_exclusive_mode_0400": True,
        "decoder_never_received_authority_path": True,
    }
    return pixels, receipt, expected_sha256


def _encode_real_q0_sources(
    *,
    rows: Sequence[OperatorRow],
    checkpoint: Path,
    rank: int,
    device: Any,
    world_group: Any,
    expected_vae_implementation_admission: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Frozen-VAE encode each unique real q0 source once, then broadcast."""

    import torch
    import torch.distributed as dist
    import diffusers as diffusers_package
    from diffusers import __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    import bernini.pipeline as bernini_pipeline
    from bernini.pipeline import _vae_encode

    import infer_lora as inference_source

    infer_lora_path = Path(inference_source.__file__).resolve(strict=True)
    pipeline_path = Path(bernini_pipeline.__file__).resolve(strict=True)
    vae_module = sys.modules.get(AutoencoderKLWan.__module__)
    vae_module_path = Path(vae_module.__file__).resolve(strict=True) if vae_module else None
    encode_vae_implementation_admission = admit_q0_vae_implementation(
        autoencoder_class=AutoencoderKLWan,
        diffusers_package=diffusers_package,
        implementation_module=vae_module,
    )
    if (
        infer_lora_path != (METHOD_ROOT / "infer_lora.py").resolve(strict=True)
        or file_sha256(infer_lora_path) != EXPECTED_Q0_INFER_LORA_SHA256
        or pipeline_path.name != "pipeline.py"
        or file_sha256(pipeline_path)
        != inference_source.BERNINI_INFERENCE_FILE_HASHES["bernini/pipeline.py"]
        or _vae_encode.__module__ != "bernini.pipeline"
        or diffusers_version != EXPECTED_DIFFUSERS_VERSION
        or vae_module_path is None
        or encode_vae_implementation_admission
        != expected_vae_implementation_admission
        or not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
    ):
        fail("pinned deterministic q0 VAE encode implementation differs")
    q0_encoder_implementation = {
        "infer_lora_path": str(infer_lora_path),
        "infer_lora_file_sha256": EXPECTED_Q0_INFER_LORA_SHA256,
        "bernini_pipeline_path": str(pipeline_path),
        "bernini_pipeline_file_sha256": file_sha256(pipeline_path),
        "vae_class": f"{AutoencoderKLWan.__module__}.{AutoencoderKLWan.__name__}",
        "vae_module_path": str(vae_module_path),
        "vae_module_file_sha256": file_sha256(vae_module_path),
        "registered_vae_implementation_admission": dict(
            encode_vae_implementation_admission
        ),
        "diffusers_version": diffusers_version,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "encode_repetitions": 2,
        "repeat_must_be_bit_exact": True,
    }

    by_iid: dict[str, OperatorRow] = {}
    for row in rows:
        existing = by_iid.setdefault(row.source_iid, row)
        if (
            existing.source_video_path != row.source_video_path
            or existing.source_video_sha256 != row.source_video_sha256
        ):
            fail("one q0 IID maps to multiple real source videos")
    expected_unique = 1 if len(rows) == 1 else 4
    if len(by_iid) != expected_unique:
        fail("O-stage real q0 source population differs")
    vae: Any = None
    if rank == 0:
        vae = AutoencoderKLWan.from_pretrained(
            str(checkpoint),
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval().requires_grad_(False)
        if vae.training or any(parameter.requires_grad for parameter in vae.parameters()):
            fail("q0 VAE must be frozen in eval mode")
        vae.to(device)
    latents: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for iid in sorted(by_iid):
        row = by_iid[iid]
        envelope: list[Any] = [None]
        encoded: Any = None
        if rank == 0:
            try:
                path = _plain_file(
                    row.source_video_path,
                    expected_sha256=row.source_video_sha256,
                    label=f"{iid} real q0 source video",
                )
                pixels, metadata, source_sha = _prepare_hashed_source_snapshot(
                    path, expected_sha256=row.source_video_sha256
                )
                if (
                    source_sha != row.source_video_sha256
                    or metadata.get("frame_count") != core.FRAME_COUNT
                    or float(metadata.get("fps", -1.0)) != 25.0
                ):
                    fail("real q0 source decode authority differs")
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    encoded_first = _vae_encode(
                        vae, pixels.to(device=device, dtype=torch.float32)
                    ).float().detach().contiguous()
                    encoded_second = _vae_encode(
                        vae, pixels.to(device=device, dtype=torch.float32)
                    ).float().detach().contiguous()
                if not torch.equal(encoded_first, encoded_second):
                    fail("frozen q0 VAE repeated encode is not bit-exact")
                encoded = encoded_first
                del encoded_second
                del pixels
                if (
                    encoded.ndim != 5
                    or tuple(int(item) for item in encoded.shape[:3])
                    != (1, 16, core.LATENT_PHASES)
                    or encoded.requires_grad
                    or not bool(torch.isfinite(encoded).all().item())
                ):
                    fail("frozen-VAE q0 latent geometry differs")
                envelope[0] = {
                    "ok": True,
                    "shape": list(encoded.shape),
                    "metadata": dict(metadata),
                    "source_video_sha256": source_sha,
                }
            except Exception as error:  # noqa: BLE001 - broadcast failure
                envelope[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(envelope, src=0, group=world_group)
        info = envelope[0]
        if not isinstance(info, Mapping) or info.get("ok") is not True:
            fail(f"{iid} q0 VAE encoding failed: {info!r}")
        shape = tuple(int(item) for item in info["shape"])
        if rank != 0:
            encoded = torch.empty(shape, device=device, dtype=torch.float32)
        dist.broadcast(encoded, src=0, group=world_group)
        local_sha = hashlib.sha256(_tensor_bytes(encoded)).hexdigest()
        gathered: list[Any] = [None] * core.WORLD_SIZE
        dist.all_gather_object(gathered, local_sha, group=world_group)
        if gathered != [local_sha] * core.WORLD_SIZE:
            fail("frozen-VAE q0 latent bytes differ across WORLD4")
        latents[iid] = encoded.cpu().contiguous()
        receipt = {
            "iid": iid,
            "source_video_path": row.source_video_path,
            "source_video_sha256": row.source_video_sha256,
            "decoded_metadata": dict(info["metadata"]),
            "vae_encoder": "AutoencoderKLWan+bernini.pipeline._vae_encode",
            "vae_encoder_implementation": q0_encoder_implementation,
            "repeated_encode_bit_exact": True,
            "latent_shape": list(shape),
            "latent_dtype": "torch.float32",
            "latent_raw_sha256": local_sha,
            "real_source_only": True,
            "self_generated_media": False,
            "detached": True,
        }
        receipts[iid] = {**receipt, "digest": object_sha256(receipt)}
        del encoded
        torch.cuda.empty_cache()
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return latents, receipts


def _unpadded_text_tokens(text_lens: Any, text_embs: Any) -> Any:
    import torch

    if isinstance(text_embs, (list, tuple)):
        if len(text_embs) != 1:
            fail("instruction encoding must contain exactly one embedding")
        text_embs = text_embs[0]
    if not isinstance(text_embs, torch.Tensor):
        fail("instruction embedding is not a tensor")
    lengths = torch.as_tensor(text_lens).reshape(-1)
    if lengths.numel() != 1 or int(lengths[0]) <= 0:
        fail("instruction encoding has no positive text length")
    length = int(lengths[0])
    if text_embs.ndim == 3:
        if int(text_embs.shape[0]) != 1 or int(text_embs.shape[1]) < length:
            fail("batched instruction embedding geometry differs")
        selected = text_embs[0, :length]
    elif text_embs.ndim == 2:
        if int(text_embs.shape[0]) < length:
            fail("instruction embedding is shorter than its declared length")
        selected = text_embs[:length]
    else:
        fail("instruction embedding must be [L,D] or [1,L,D]")
    if int(selected.shape[1]) != core.TEXT_WIDTH:
        fail("frozen UMT5 width differs")
    nonzero = selected.detach().abs().sum(dim=1) > 0
    if not bool(nonzero.all().item()) or not bool(torch.isfinite(selected).all().item()):
        fail("full instruction tokens contain padding/internal-zero/non-finite rows")
    return selected.detach().cpu().contiguous().unsqueeze(0)


def _encode_instruction_bank(
    *,
    renderer: Any,
    tokenizer: Any,
    rows: Sequence[PlannerRow],
    include_noop: bool,
    device: Any,
    runtime: Any,
    world_group: Any,
) -> tuple[Mapping[str, InstructionViews], Mapping[str, Any]]:
    import torch

    instructions: dict[str, tuple[str, str]] = {}
    for row in rows:
        existing = instructions.setdefault(
            row.instruction_sha256, (row.instruction, row.instruction_sha256)
        )
        if existing[0] != row.instruction:
            fail("instruction SHA maps to different text")
    if include_noop:
        instructions[core.EXACT_NOOP_INSTRUCTION_SHA256] = (
            core.EXACT_NOOP_INSTRUCTION,
            core.EXACT_NOOP_INSTRUCTION_SHA256,
        )
    bank: dict[str, InstructionViews] = {}
    receipts: dict[str, Any] = {}
    for digest in sorted(instructions):
        instruction, expected_digest = instructions[digest]
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != expected_digest:
            fail("instruction UTF-8 SHA differs")
        tokenized = runtime.tokenize_generic_instruction(tokenizer, instruction, device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            text_lens, text_embs = renderer.get_t5_text_embeddings(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                tokenized["t5_input_lens"],
            )
        # Bernini consumes the real token length but returns [512] for Wan's
        # packed cross-attention metadata and a zero-padded [1,512,4096]
        # embedding.  The Planner authority is the full *unpadded* token
        # state, so select with the tokenizer's sealed actual length rather
        # than the renderer's packed length.
        renderer_max_sequence_length = int(
            getattr(renderer, "max_sequence_length", 0)
        )
        if (
            renderer_max_sequence_length != 512
            or [int(item) for item in text_lens]
            != [renderer_max_sequence_length]
            or list(tokenized["t5_input_lens"].shape) != [1, 1]
            or tokenized["input_ids"].ndim != 2
            or list(tokenized["input_ids"].shape)
            != list(tokenized["attention_mask"].shape)
            or int(tokenized["input_ids"].shape[0]) != 1
            or int(tokenized["t5_input_lens"].item())
            != int(tokenized["input_ids"].shape[1])
            or not 0 < int(tokenized["t5_input_lens"].item()) <= 512
        ):
            fail("frozen UMT5 renderer/tokenizer length contract differs")
        if isinstance(text_embs, (list, tuple)):
            if len(text_embs) != 1:
                fail("official UMT5 operator view must contain one embedding")
            official_text_embs = text_embs[0]
        else:
            official_text_embs = text_embs
        if (
            not isinstance(official_text_embs, torch.Tensor)
            or list(official_text_embs.shape) != [1, 512, core.TEXT_WIDTH]
            or official_text_embs.dtype != torch.bfloat16
            or official_text_embs.requires_grad
            or not bool(torch.isfinite(official_text_embs).all().item())
        ):
            fail("official padded UMT5 operator view differs")
        actual_token_length = int(tokenized["t5_input_lens"].item())
        if actual_token_length < renderer_max_sequence_length and bool(
            torch.count_nonzero(
                official_text_embs[:, actual_token_length:, :]
            ).item()
        ):
            fail("official UMT5 operator padding rows are not exact zero")
        selected = _unpadded_text_tokens(
            tokenized["t5_input_lens"], official_text_embs
        )
        operator_text_embs = official_text_embs.detach().cpu().contiguous()
        planner_token_sha = hashlib.sha256(_tensor_bytes(selected)).hexdigest()
        operator_token_sha = hashlib.sha256(
            _tensor_bytes(operator_text_embs)
        ).hexdigest()
        view_identity = {
            "instruction_sha256": digest,
            "planner_unpadded_raw_sha256": planner_token_sha,
            "operator_official_padded_raw_sha256": operator_token_sha,
            "operator_text_lens": [renderer_max_sequence_length],
        }
        view_digest = object_sha256(view_identity)
        import torch.distributed as dist

        gathered: list[Any] = [None] * core.WORLD_SIZE
        dist.all_gather_object(gathered, view_digest, group=world_group)
        if gathered != [view_digest] * core.WORLD_SIZE:
            fail("dual frozen UMT5 instruction views differ across WORLD4")
        bank[digest] = InstructionViews(
            planner_tokens=selected,
            operator_text_embs=operator_text_embs,
            operator_text_lens=(renderer_max_sequence_length,),
        )
        receipt = {
            "instruction_sha256": digest,
            "planner_unpadded": {
                "shape": list(selected.shape),
                "dtype": str(selected.dtype),
                "raw_sha256": planner_token_sha,
                "actual_token_length": actual_token_length,
            },
            "operator_official_padded": {
                "shape": list(operator_text_embs.shape),
                "dtype": str(operator_text_embs.dtype),
                "raw_sha256": operator_token_sha,
                "text_lens": [renderer_max_sequence_length],
                "padding_rows_exact_zero": True,
            },
            "frozen_detached": True,
            "full_unpadded_token_states": True,
            "operator_uses_official_padded_view": True,
        }
        receipts[digest] = {**receipt, "digest": object_sha256(receipt)}
        del (
            tokenized,
            text_lens,
            text_embs,
            official_text_embs,
            selected,
            operator_text_embs,
        )
    return bank, receipts


def _read_decimal_file(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return int(raw) if raw.isdecimal() else None


def _cgroup_memory_receipt() -> Mapping[str, float]:
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise GenericActionPORuntimeError("cannot read process cgroup") from error
    unified: Optional[Path] = None
    memory_v1: Optional[Path] = None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        if hierarchy == "0" and controllers == "":
            unified = Path("/sys/fs/cgroup") / relative.lstrip("/")
        elif "memory" in controllers.split(","):
            memory_v1 = Path("/sys/fs/cgroup/memory") / relative.lstrip("/")
    current: Optional[int] = None
    peak: Optional[int] = None
    if unified is not None:
        current = _read_decimal_file(unified / "memory.current")
        peak = _read_decimal_file(unified / "memory.peak")
    if current is None and memory_v1 is not None:
        current = _read_decimal_file(memory_v1 / "memory.usage_in_bytes")
        peak = _read_decimal_file(memory_v1 / "memory.max_usage_in_bytes")
    if current is None:
        fail("live cgroup memory.current is unavailable")
    if peak is None:
        peak = current
    gib = float(1024**3)
    return {"current_gib": current / gib, "peak_gib": peak / gib}


def collect_and_gate_resources(
    *,
    stage: str,
    rank: int,
    device: Any,
    world_group: Any,
    gpu_limit_gib: float,
    host_limit_gib: float,
) -> tuple[Mapping[str, Any], ...]:
    import torch
    import torch.distributed as dist

    torch.cuda.synchronize(device)
    gib = float(1024**3)
    rss = native_r.linux_host_memory_receipt()
    cgroup = _cgroup_memory_receipt()
    local = {
        "stage": stage,
        "rank": rank,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "gpu_allocated_gib": torch.cuda.memory_allocated(device) / gib,
        "gpu_reserved_gib": torch.cuda.memory_reserved(device) / gib,
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "host_current_rss_gib": rss["current_rss_gib"],
        "host_peak_rss_gib": rss["peak_rss_gib"],
        "host_cgroup_current_gib": cgroup["current_gib"],
        "host_cgroup_peak_gib": cgroup["peak_gib"],
    }
    gathered: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(gathered, local, group=world_group)
    if [row.get("rank") for row in gathered] != list(range(core.WORLD_SIZE)):
        fail(f"{stage} resource WORLD4 closure differs")
    if any(
        float(row["gpu_peak_reserved_gib"]) >= gpu_limit_gib
        or float(row["host_peak_rss_gib"]) >= host_limit_gib
        or float(row["host_cgroup_current_gib"]) >= host_limit_gib
        or float(row["host_cgroup_peak_gib"]) >= host_limit_gib
        for row in gathered
    ):
        fail(
            f"{stage} crossed strict GPU<{gpu_limit_gib}GiB or "
            f"host<{host_limit_gib}GiB memory gate"
        )
    return tuple(dict(row) for row in gathered)


def _gradient_group_norms(
    active: Sequence[tuple[str, Any]], *, stage: str
) -> Mapping[str, float]:
    import torch

    expected_prefix = "planner." if stage == "P" else "operator."
    if stage not in {"P", "O"} or not active or any(
        not name.startswith(expected_prefix) for name, _ in active
    ):
        fail(f"Stage {stage} active parameter names differ")
    groups: dict[str, Any] = {}
    for name, parameter in active:
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
            fail(f"Stage {stage} gradient is missing/non-finite")
        if stage == "P":
            key = name.split(".", 2)[1]
        else:
            key = name.rsplit(".", 2)[-2]
        squared = parameter.grad.detach().float().square().sum()
        groups[key] = groups.get(key, squared.new_zeros(())) + squared
    return {key: float(value.sqrt().item()) for key, value in sorted(groups.items())}


def _operator_wrapper_gradient_norms(
    active: Sequence[tuple[str, Any]], *, stage_update: int
) -> Mapping[str, Mapping[str, float]]:
    expected_wrappers = {
        f"operator.blocks.{block}.attn2.{projection}"
        for block in core.ACTION_BLOCK_INDICES
        for projection in ("to_q", "to_out.0")
    }
    values: dict[str, dict[str, float]] = {}
    for name, parameter in active:
        prefix, component, suffix = name.rsplit(".", 2)
        if (
            prefix not in expected_wrappers
            or component not in {"state_down", "phase_gate", "output_up"}
            or suffix != "weight"
            or parameter.grad is None
        ):
            fail("Stage O per-wrapper gradient key closure differs")
        norm = float(parameter.grad.detach().float().norm().item())
        if not math.isfinite(norm):
            fail("Stage O per-wrapper gradient is non-finite")
        values.setdefault(prefix, {})[component] = norm
    if set(values) != expected_wrappers or any(
        set(row) != {"state_down", "phase_gate", "output_up"}
        for row in values.values()
    ):
        fail("Stage O per-wrapper gradient component closure differs")
    # native_r._zero_dependency contributes exactly-zero derivatives.  Thus
    # these strictly-positive norms can only come through a real wrapper edge.
    if any(row["output_up"] <= 0.0 for row in values.values()):
        fail("Stage O has a wrapper without a real output_up gradient")
    if stage_update == 1:
        if any(
            row["state_down"] != 0.0 or row["phase_gate"] != 0.0
            for row in values.values()
        ):
            fail("Stage O first-step zero-init gradient closure differs")
    elif any(
        row["state_down"] <= 0.0 or row["phase_gate"] <= 0.0
        for row in values.values()
    ):
        fail("Stage O learned wrapper lacks a real state/phase gradient")
    return {key: dict(sorted(value.items())) for key, value in sorted(values.items())}


def _clip_and_step(
    *,
    active: Sequence[tuple[str, Any]],
    controller: core.StageOptimizerController,
    parallel: Any,
    max_grad_norm: float,
    runtime: Any,
    stage_update: int,
) -> tuple[float, float, Mapping[str, float], Mapping[str, Any]]:
    import torch

    preclip = runtime.synchronize_gradients(active, parallel)
    gradient_groups = _gradient_group_norms(active, stage=controller.active_stage)
    wrapper_gradient_norms: Mapping[str, Any] = {}
    if controller.active_stage == "O":
        wrapper_gradient_norms = _operator_wrapper_gradient_norms(
            active, stage_update=stage_update
        )
    clipped = torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in active], max_grad_norm
    )
    clipped_value = float(clipped)
    if not math.isfinite(clipped_value) or preclip <= 0.0:
        fail("P/O synchronized gradient norm is zero/non-finite")
    controller.optimizer.step()
    return preclip, clipped_value, gradient_groups, wrapper_gradient_norms


def _run_planner_updates(
    *,
    rows: Sequence[PlannerRow],
    composite: core.CompositeHandle,
    controller: core.StageOptimizerController,
    text_bank: Mapping[str, InstructionViews],
    planner_teacher_bank: Mapping[str, Any],
    device: Any,
    parallel: Any,
    runtime: Any,
    max_grad_norm: float,
    resource_callback: Any,
    checkpoint_callback: Any,
) -> list[Mapping[str, Any]]:
    import torch
    import torch.distributed as dist

    if len(rows) not in (1, core.STAGE_UPDATES["P"]):
        fail("Stage P update traversal differs")
    active = controller.activate("P")
    optimizer = controller.optimizer
    previous_sha = _parameter_consensus(
        active, world_group=parallel.world_group, label="pre-Stage-P planner"
    )
    history: list[Mapping[str, Any]] = []
    for position, row in enumerate(rows):
        admitted_teacher = planner_teacher_bank.get(row.row_id)
        if not isinstance(admitted_teacher, torch.Tensor):
            fail("Stage P row is absent from the pre-optimizer P24 admission")
        teacher = admitted_teacher.to(
            device=device, dtype=torch.float32
        ).detach()
        _validate_nonnoop_teacher(teacher, label=f"Stage P row {row.row_id}")
        payload = row.optimizer_payload(teacher)
        if row.branch == "noop" or bool(torch.count_nonzero(teacher).item()) is False:
            fail("Stage P received a no-op/zero optimizer row")
        tokens = text_bank[row.instruction_sha256].planner_tokens.to(
            device=device
        )
        inactive_before = core.frozen_inactive_snapshot(composite, "P")
        optimizer.zero_grad(set_to_none=True)
        prediction = composite.planner(
            tokens,
            instruction=row.instruction,
            instruction_sha256=row.instruction_sha256,
            is_noop=False,
        )
        loss = core.cosine_quotient_loss(prediction, teacher)
        if position == 0:
            resource_callback("p_first_forward")
        loss_value = float(loss.detach().item())
        loss.backward()
        if position == 0:
            resource_callback("p_first_backward")
        preclip, clipped, gradient_groups, wrapper_gradient_norms = _clip_and_step(
            active=active,
            controller=controller,
            parallel=parallel,
            max_grad_norm=max_grad_norm,
            runtime=runtime,
            stage_update=position + 1,
        )
        if wrapper_gradient_norms:
            fail("Stage P unexpectedly produced Operator wrapper gradients")
        controller.assert_inactive_unchanged(inactive_before)
        planner_sha = _parameter_consensus(
            active,
            world_group=parallel.world_group,
            label=f"Stage P update {position + 1}",
        )
        if planner_sha == previous_sha:
            fail("Stage P optimizer update did not change planner parameters")
        previous_sha = planner_sha
        if position == 0:
            resource_callback("p_first_optimizer_step")
        local = {
            "stage": "P",
            "stage_update": position + 1,
            "global_update": core.STAGE_UPDATES["R"] + position + 1,
            "row_id": row.row_id,
            "source_iid": row.source_iid,
            "seed": row.seed,
            "branch": row.branch,
            "instruction_sha256": row.instruction_sha256,
            "quotient_tensor_sha256": row.quotient_sha256,
            "loss": loss_value,
            "preclip_gradient_norm_sp4_mean": preclip,
            "clip_return_norm": clipped,
            "gradient_group_norms": gradient_groups,
            "planner_parameter_sha256_after": planner_sha,
            "one_shared_logical_row": True,
            "optimizer_payload_fields": sorted(payload),
            "generated_media_optimizer_input": False,
            "optimizer_step_executed": True,
        }
        gathered: list[Any] = [None] * core.WORLD_SIZE
        dist.all_gather_object(gathered, local, group=parallel.world_group)
        if gathered != [local] * core.WORLD_SIZE:
            fail("SP4 ranks did not execute the same logical Stage P row")
        history.append({"schema_version": PO_HISTORY_SCHEMA, **local})
        if len(rows) == core.STAGE_UPDATES["P"] and position + 1 in P_CHECKPOINT_STEPS:
            checkpoint_callback("P", position + 1, history)
        del (
            teacher,
            payload,
            tokens,
            inactive_before,
            prediction,
            loss,
            local,
            gathered,
        )
        gc.collect()
    return history


def _validate_nonnoop_teacher(tensor: Any, *, label: str) -> None:
    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != (1, core.LATENT_PHASES, core.PHASE_CODE_WIDTH)
        or tensor.dtype != torch.float32
        or tensor.requires_grad
        or not bool(torch.isfinite(tensor).all().item())
        or bool(torch.count_nonzero(tensor).item()) is False
        or not torch.equal(tensor[:, 0, :], torch.zeros_like(tensor[:, 0, :]))
        or not math.isclose(
            float(tensor.norm().item()), 1.0, rel_tol=0.0, abs_tol=5.0e-5
        )
    ):
        fail(f"{label} must be one detached unit-norm non-noop [21,32] teacher")


def _operator_forward_audit(
    *,
    composite: core.CompositeHandle,
    route: core.ActionRoute,
    expect_noop: bool,
    expect_enabled: bool,
    expect_nonzero_delta: Optional[bool],
) -> Mapping[str, Any]:
    audits = composite.operator.pop_runtime_audits()
    expected_wrappers = [
        f"blocks.{block}.attn2.{projection}"
        for block in core.ACTION_BLOCK_INDICES
        for projection in ("to_q", "to_out.0")
    ]
    if [row.get("wrapper") for row in audits] != expected_wrappers:
        fail("Stage O runtime audit wrapper order/scope differs")
    for row in audits:
        expected_projection = (
            "to_out.0" if str(row["wrapper"]).endswith("to_out.0") else "to_q"
        )
        if (
            row.get("projection") != expected_projection
            or row.get("route_present") is not True
            or row.get("route_enabled") is not expect_enabled
            or row.get("is_noop") is not expect_noop
            or row.get("schedule_index") != route.schedule_index
            or row.get("protected_rows_bit_exact") is not True
            or row.get("hard_bypass") is not (expect_noop or not expect_enabled)
        ):
            fail("Stage O direct operator row-write gate failed")
        if expect_nonzero_delta is not None:
            expected_row_delta = expect_nonzero_delta and int(
                row.get("positive_phase_target_rows", -1)
            ) > 0
            if (
                row.get("selected_delta_nonzero") is not expected_row_delta
                or (float(row.get("selected_delta_l2", -1.0)) > 0.0)
                is not expected_row_delta
            ):
                fail("Stage O per-wrapper real delta gate failed")
    geometry = {
        (
            int(row["local_rows"]),
            int(row["source_reference_padding_rows"]),
            int(row["phase0_rows"]),
            int(row["positive_phase_target_rows"]),
        )
        for row in audits
    }
    if len(geometry) != 1:
        fail("Stage O operator wrappers observed different local token geometry")
    local_rows, protected_rows, phase0_rows, action_rows = next(iter(geometry))
    value = {
        "wrapper_count": len(audits),
        "wrapper_order_sha256": object_sha256(expected_wrappers),
        "blocks": list(core.ACTION_BLOCK_INDICES),
        "projections": ["attn2.to_q", "attn2.to_out.0"],
        "schedule_index": route.schedule_index,
        "is_noop": expect_noop,
        "route_enabled": expect_enabled,
        "canonical_noop_hard_bypass": expect_noop,
        "operator_off_hard_bypass": not expect_enabled,
        "local_rows": local_rows,
        "source_reference_padding_rows": protected_rows,
        "phase0_rows": phase0_rows,
        "positive_phase_target_rows": action_rows,
        "source_reference_padding_and_phase0_direct_bytes_unchanged": True,
        "selected_delta_l2_by_wrapper": {
            str(row["wrapper"]): float(row["selected_delta_l2"])
            for row in audits
        },
        "all_local_active_wrapper_deltas_nonzero": all(
            int(row["positive_phase_target_rows"]) == 0
            or row["selected_delta_nonzero"] is True
            for row in audits
        ),
        "audit_rows_digest": object_sha256([dict(row) for row in audits]),
    }
    return {**value, "digest": object_sha256(value)}


def _native_operator_forward(
    *,
    renderer: Any,
    transformer: Any,
    composite: core.CompositeHandle,
    packed: Any,
    coordinate: Any,
    carrier_route: visual.VisualContextRoute,
    phase_code: Any,
    text_view: InstructionViews,
    sp_rank: int,
    is_noop: bool,
    require_grad: bool,
    route_enabled: bool = True,
    expect_nonzero_delta: Optional[bool] = None,
) -> tuple[Any, Any, Mapping[str, Any], core.ActionRoute]:
    import torch

    if (
        not isinstance(text_view, InstructionViews)
        or not isinstance(text_view.operator_text_embs, torch.Tensor)
        or text_view.operator_text_embs.device.type != "cpu"
        or text_view.operator_text_embs.requires_grad
        or list(text_view.operator_text_embs.shape)
        != [1, 512, core.TEXT_WIDTH]
        or text_view.operator_text_lens != (512,)
    ):
        fail("Stage O official padded text view contract differs")
    action_route = core.ActionRoute(
        total_tokens=packed.total_tokens,
        condition_tokens=packed.condition_tokens,
        sequence_parallel_rank=sp_rank,
        sequence_parallel_size=core.SP_SIZE,
        phase_code=phase_code,
        schedule_index=coordinate.schedule_index,
        is_noop=is_noop,
        enabled=route_enabled,
    )
    composite.operator.clear_runtime_audits()
    capture = core.BlockOutputCapture(transformer)
    context = nullcontext() if require_grad else torch.no_grad()
    try:
        with context, core.composite_route(
            composite,
            carrier_route=carrier_route,
            action_route=action_route,
        ), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = native_r._prediction(
                renderer=renderer,
                transformer=transformer,
                condition=packed,
                coordinate=coordinate,
                text_lens=list(text_view.operator_text_lens),
                text_embs=text_view.operator_text_embs.to(
                    device=phase_code.device, non_blocking=False
                ),
            )
        hidden = capture.pop()
    finally:
        capture.close()
    audit = _operator_forward_audit(
        composite=composite,
        route=action_route,
        expect_noop=is_noop,
        expect_enabled=route_enabled,
        expect_nonzero_delta=expect_nonzero_delta,
    )
    if (
        tuple(hidden.shape[:1]) != (1,)
        or int(hidden.shape[1]) != action_route.local_length
        or int(hidden.shape[2]) != core.HIDDEN_SIZE_1P3B
        or (not require_grad and hidden.requires_grad)
        or not bool(torch.isfinite(hidden.detach()).all().item())
        or not bool(torch.isfinite(prediction.detach()).all().item())
    ):
        fail("Stage O captured native block-22 hidden contract differs")
    return prediction, hidden, audit, action_route


def _validate_world4_operator_geometry(
    *,
    gathered_audits: Sequence[Mapping[str, Any]],
    route: core.ActionRoute,
    expect_noop: bool,
    expect_enabled: bool = True,
) -> None:
    if len(gathered_audits) != core.WORLD_SIZE:
        fail("Stage O operator audit WORLD4 closure differs")
    if any(
        row.get("wrapper_count") != len(core.ACTION_BLOCK_INDICES) * 2
        or row.get("is_noop") is not expect_noop
        or row.get("route_enabled") is not expect_enabled
        or row.get(
            "source_reference_padding_and_phase0_direct_bytes_unchanged"
        )
        is not True
        for row in gathered_audits
    ):
        fail("Stage O operator direct-write WORLD4 gate failed")
    if (
        sum(int(row["phase0_rows"]) for row in gathered_audits)
        != route.patch_positions
        or sum(
            int(row["positive_phase_target_rows"])
            for row in gathered_audits
        )
        != (core.LATENT_PHASES - 1) * route.patch_positions
        or sum(
            int(row["source_reference_padding_rows"])
            for row in gathered_audits
        )
        != route.local_length * core.SP_SIZE - route.target_tokens
    ):
        fail("Stage O source/phase/padding WORLD4 geometry differs")


def _sentinel_bytes(value: Any) -> Mapping[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "raw_sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
    }


def preoptimizer_operator_off_parity(
    *,
    row: OperatorRow,
    composite: core.CompositeHandle,
    text_bank: Mapping[str, InstructionViews],
    q0_latents: Mapping[str, Any],
    renderer: Any,
    transformer: Any,
    rope: Any,
    device: Any,
    parallel: Any,
    runtime: Any,
    seed: int,
) -> Mapping[str, Any]:
    """Prove installed wrappers-off equals the canonical hard-bypass bytes."""

    import torch
    import torch.distributed as dist

    coordinate = _o_coordinate(0)
    clean_source = q0_latents.get(row.source_iid)
    if not isinstance(clean_source, torch.Tensor):
        fail("operator-off parity has no real q0 latent")
    noop_view = text_bank[core.EXACT_NOOP_INSTRUCTION_SHA256]
    with torch.no_grad():
        noop_code = composite.planner(
            noop_view.planner_tokens.to(device=device),
            instruction=core.EXACT_NOOP_INSTRUCTION,
            instruction_sha256=core.EXACT_NOOP_INSTRUCTION_SHA256,
            is_noop=True,
        ).detach()
    noise_seed = _noise_seed(seed, row.row_id, coordinate.schedule_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    epsilon = torch.randn(
        tuple(clean_source.shape), generator=generator, dtype=torch.float32
    ).contiguous()
    packed = native_r.prepare_noop_condition(
        clean_source=clean_source,
        epsilon=epsilon,
        coordinate=coordinate,
        memory_input_kind="same_noise_forward_noised_source",
        rope=rope,
        device=device,
        runtime=runtime,
    )
    with torch.no_grad():
        memory = composite.carrier.build_memory(
            packed.memory_input.to(
                device=device, dtype=torch.float32
            ).detach().contiguous(),
            source_video_sha256=row.source_video_sha256,
            memory_input_latent_sha256=packed.tensor_identities[
                "visual_context_input"
            ],
            input_kind="same_noise_forward_noised_source",
        )
    carrier_route = visual.VisualContextRoute(
        packed.total_tokens,
        packed.condition_tokens,
        parallel.contract.sp_rank,
        core.SP_SIZE,
        memory,
    )
    noop_prediction, noop_hidden, noop_audit, noop_route = (
        _native_operator_forward(
            renderer=renderer,
            transformer=transformer,
            composite=composite,
            packed=packed,
            coordinate=coordinate,
            carrier_route=carrier_route,
            phase_code=noop_code,
            text_view=noop_view,
            sp_rank=parallel.contract.sp_rank,
            is_noop=True,
            require_grad=False,
            route_enabled=True,
            expect_nonzero_delta=False,
        )
    )
    off_prediction, off_hidden, off_audit, off_route = _native_operator_forward(
        renderer=renderer,
        transformer=transformer,
        composite=composite,
        packed=packed,
        coordinate=coordinate,
        carrier_route=carrier_route,
        phase_code=noop_code,
        text_view=noop_view,
        sp_rank=parallel.contract.sp_rank,
        is_noop=False,
        require_grad=False,
        route_enabled=False,
        expect_nonzero_delta=False,
    )
    local_equal = torch.equal(noop_prediction, off_prediction) and torch.equal(
        noop_hidden, off_hidden
    )
    if not runtime.world_all_true(local_equal, group=parallel.world_group):
        fail("pre-optimizer operator-off/native hard-bypass parity differs")
    noop_world: list[Any] = [None] * core.WORLD_SIZE
    off_world: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(noop_world, noop_audit, group=parallel.world_group)
    dist.all_gather_object(off_world, off_audit, group=parallel.world_group)
    _validate_world4_operator_geometry(
        gathered_audits=noop_world,
        route=noop_route,
        expect_noop=True,
        expect_enabled=True,
    )
    _validate_world4_operator_geometry(
        gathered_audits=off_world,
        route=off_route,
        expect_noop=False,
        expect_enabled=False,
    )
    local = {
        "rank": parallel.contract.rank,
        "row_id": row.row_id,
        "source_iid": row.source_iid,
        "noise_seed": noise_seed,
        "official_padded_text_lens": list(noop_view.operator_text_lens),
        "noop_prediction": _sentinel_bytes(noop_prediction),
        "operator_off_prediction": _sentinel_bytes(off_prediction),
        "noop_block22_hidden": _sentinel_bytes(noop_hidden),
        "operator_off_block22_hidden": _sentinel_bytes(off_hidden),
        "prediction_and_block22_bit_exact": True,
        "operator_wrappers_installed_but_disabled": True,
        "executed_before_optimizer_construction": True,
        "noop_audit": noop_audit,
        "operator_off_audit": off_audit,
    }
    parity_world: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(parity_world, local, group=parallel.world_group)
    if [item.get("rank") for item in parity_world] != list(
        range(core.WORLD_SIZE)
    ) or any(
        item.get("prediction_and_block22_bit_exact") is not True
        for item in parity_world
    ):
        fail("pre-optimizer operator-off parity receipt differs across WORLD4")
    value = {
        "row_id": row.row_id,
        "source_iid": row.source_iid,
        "noise_seed": noise_seed,
        "world4_by_rank": parity_world,
        "all_ranks_prediction_and_block22_bit_exact": True,
        "executed_before_optimizer_construction": True,
    }
    del (
        epsilon,
        packed,
        memory,
        carrier_route,
        noop_prediction,
        noop_hidden,
        off_prediction,
        off_hidden,
    )
    return {**value, "digest": object_sha256(value)}


def _run_operator_updates(
    *,
    rows: Sequence[OperatorRow],
    composite: core.CompositeHandle,
    controller: core.StageOptimizerController,
    text_bank: Mapping[str, InstructionViews],
    q0_latents: Mapping[str, Any],
    operator_coordinate_bank: Mapping[str, OperatorCoordinateAdmission],
    renderer: Any,
    transformer: Any,
    rope: Any,
    p32: Any,
    device: Any,
    parallel: Any,
    runtime: Any,
    seed: int,
    global_update_offset: int,
    max_grad_norm: float,
    resource_callback: Any,
    checkpoint_callback: Any,
) -> list[Mapping[str, Any]]:
    import torch
    import torch.distributed as dist

    if len(rows) not in (1, core.STAGE_UPDATES["O"]):
        fail("Stage O update traversal differs")
    if global_update_offset not in (
        core.STAGE_UPDATES["R"],
        core.STAGE_UPDATES["R"] + core.STAGE_UPDATES["P"],
    ):
        fail("Stage O global update offset differs")
    if composite.carrier is None:
        fail("Stage O requires the frozen R64 source carrier")
    if set(row.branch for row in rows) - {"action", "incomplete"}:
        fail("Stage O received noop/reverse/non-q0 branch")
    active = controller.activate("O")
    optimizer = controller.optimizer
    previous_sha = _parameter_consensus(
        active, world_group=parallel.world_group, label="pre-Stage-O operator"
    )
    history: list[Mapping[str, Any]] = []
    noop_view = text_bank[core.EXACT_NOOP_INSTRUCTION_SHA256]
    with torch.no_grad():
        noop_code = composite.planner(
            noop_view.planner_tokens.to(device=device),
            instruction=core.EXACT_NOOP_INSTRUCTION,
            instruction_sha256=core.EXACT_NOOP_INSTRUCTION_SHA256,
            is_noop=True,
        ).detach()
    if (
        noop_code.dtype != torch.float32
        or noop_code.requires_grad
        or bool(torch.count_nonzero(noop_code).item())
    ):
        fail("canonical no-op planner hard bypass is not exact zero")

    for position, row in enumerate(rows):
        stage_update = position + 1
        coordinate = _o_coordinate(position)
        admission = operator_coordinate_bank.get(row.row_id)
        if not isinstance(admission, OperatorCoordinateAdmission):
            fail("Stage O row is absent from the pre-optimizer O16 admission")
        teacher = admission.teacher.to(device=device, dtype=torch.float32).detach()
        camera = admission.camera.to(device=device, dtype=torch.float32).detach()
        appearance = admission.appearance.to(
            device=device, dtype=torch.float32
        ).detach()
        _validate_nonnoop_teacher(teacher, label=f"Stage O row {row.row_id}")
        payload = row.optimizer_payload(teacher, camera, appearance)
        clean_source = q0_latents.get(row.source_iid)
        if (
            not isinstance(clean_source, torch.Tensor)
            or clean_source.device.type != "cpu"
            or clean_source.dtype != torch.float32
            or clean_source.requires_grad
            or tuple(clean_source.shape[:3])
            != (1, 16, core.LATENT_PHASES)
        ):
            fail("Stage O row has no detached frozen-VAE real q0 latent")
        action_view = text_bank[row.instruction_sha256]
        with torch.no_grad():
            phase_code = composite.planner(
                action_view.planner_tokens.to(device=device),
                instruction=row.instruction,
                instruction_sha256=row.instruction_sha256,
                is_noop=False,
            ).detach()
        if (
            phase_code.dtype != torch.float32
            or phase_code.requires_grad
            or bool(torch.count_nonzero(phase_code).item()) is False
        ):
            fail("Stage O frozen planner returned a zero/non-detached action code")

        noise_seed = _noise_seed(seed, row.row_id, coordinate.schedule_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        epsilon = torch.randn(
            tuple(clean_source.shape), generator=generator, dtype=torch.float32
        ).contiguous()
        packed = native_r.prepare_noop_condition(
            clean_source=clean_source,
            epsilon=epsilon,
            coordinate=coordinate,
            memory_input_kind="same_noise_forward_noised_source",
            rope=rope,
            device=device,
            runtime=runtime,
        )
        with torch.no_grad():
            memory = composite.carrier.build_memory(
                packed.memory_input.to(
                    device=device, dtype=torch.float32
                ).detach().contiguous(),
                source_video_sha256=row.source_video_sha256,
                memory_input_latent_sha256=packed.tensor_identities[
                    "visual_context_input"
                ],
                input_kind="same_noise_forward_noised_source",
            )
        carrier_route = visual.VisualContextRoute(
            packed.total_tokens,
            packed.condition_tokens,
            parallel.contract.sp_rank,
            core.SP_SIZE,
            memory,
        )
        inactive_before = core.frozen_inactive_snapshot(composite, "O")
        parameter_snapshot, optimizer_snapshot = _snapshot_active_step(
            active, optimizer
        )
        optimizer.zero_grad(set_to_none=True)
        try:
            pre_prediction, pre_hidden, pre_audit, noop_route = (
                _native_operator_forward(
                    renderer=renderer,
                    transformer=transformer,
                    composite=composite,
                    packed=packed,
                    coordinate=coordinate,
                    carrier_route=carrier_route,
                    phase_code=noop_code,
                    text_view=noop_view,
                    sp_rank=parallel.contract.sp_rank,
                    is_noop=True,
                    require_grad=False,
                    expect_nonzero_delta=False,
                )
            )
            adapted_prediction, adapted_hidden, adapted_audit, action_route = (
                _native_operator_forward(
                    renderer=renderer,
                    transformer=transformer,
                    composite=composite,
                    packed=packed,
                    coordinate=coordinate,
                    carrier_route=carrier_route,
                    phase_code=phase_code,
                    text_view=action_view,
                    sp_rank=parallel.contract.sp_rank,
                    is_noop=False,
                    require_grad=True,
                    expect_nonzero_delta=(stage_update != 1),
                )
            )
            # The O objective is defined at the captured block-22 hidden.
            # Release the unused blocks-23..29 prediction branch before Phi
            # and backward so it cannot inflate the live activation set.
            del adapted_prediction
            # Condition-only SP shards may own no positive target phase at a
            # particular wrapper.  Keep an exact-zero edge to every replicated
            # O parameter so autograd collectives and later SP gradient
            # averaging have identical parameter closure on all four ranks.
            adapted_hidden = native_r._zero_dependency(adapted_hidden, active)
            predicted_code = core.phi_v1_from_sp_hidden_delta(
                adapted_hidden - pre_hidden,
                route=action_route,
                p32=p32,
                sp_group=parallel.sp_group,
                camera_nuisance=camera,
                appearance_nuisance=appearance,
            )
            predicted_code = native_r._zero_dependency(predicted_code, active)
            zero_init_seen = bool(
                torch.count_nonzero(predicted_code.detach()).item() == 0
            )
            predicted_code_sha = hashlib.sha256(
                _tensor_bytes(predicted_code)
            ).hexdigest()
            predicted_code_shas: list[Any] = [None] * core.WORLD_SIZE
            dist.all_gather_object(
                predicted_code_shas,
                predicted_code_sha,
                group=parallel.world_group,
            )
            if predicted_code_shas != [predicted_code_sha] * core.WORLD_SIZE:
                fail("Stage O Phi_v1 bytes differ across WORLD4")
            loss = core.zero_init_operator_cosine_quotient_loss(
                predicted_code,
                teacher,
                stage_update=stage_update,
            )
            if position == 0:
                resource_callback("o_first_forward")
            loss_value = float(loss.detach().item())
            loss.backward()
            # Non-reentrant checkpoint replay invokes the wrappers again;
            # those recomputation audits are not a second logical forward.
            composite.operator.clear_runtime_audits()
            if position == 0:
                resource_callback("o_first_backward")
            (
                preclip,
                clipped,
                gradient_groups,
                wrapper_gradient_norms,
            ) = _clip_and_step(
                active=active,
                controller=controller,
                parallel=parallel,
                max_grad_norm=max_grad_norm,
                runtime=runtime,
                stage_update=stage_update,
            )
            if stage_update == 1 and gradient_groups.get("output_up", 0.0) <= 0.0:
                fail("Stage O zero-init output_up path has no first gradient")
            controller.assert_inactive_unchanged(inactive_before)
            operator_sha = _parameter_consensus(
                active,
                world_group=parallel.world_group,
                label=f"Stage O update {stage_update}",
            )
            if operator_sha == previous_sha:
                fail("Stage O optimizer update did not change operator parameters")

            post_prediction, post_hidden, post_audit, post_noop_route = (
                _native_operator_forward(
                    renderer=renderer,
                    transformer=transformer,
                    composite=composite,
                    packed=packed,
                    coordinate=coordinate,
                    carrier_route=carrier_route,
                    phase_code=noop_code,
                    text_view=noop_view,
                    sp_rank=parallel.contract.sp_rank,
                    is_noop=True,
                    require_grad=False,
                    expect_nonzero_delta=False,
                )
            )
            local_sentinel_equal = torch.equal(
                pre_prediction, post_prediction
            ) and torch.equal(pre_hidden, post_hidden)
            if not runtime.world_all_true(
                local_sentinel_equal, group=parallel.world_group
            ):
                fail("Stage O post-update exact-noop sentinel bytes changed")

            (
                post_action_prediction,
                post_action_hidden,
                post_action_audit,
                post_action_route,
            ) = _native_operator_forward(
                renderer=renderer,
                transformer=transformer,
                composite=composite,
                packed=packed,
                coordinate=coordinate,
                carrier_route=carrier_route,
                phase_code=phase_code,
                text_view=action_view,
                sp_rank=parallel.contract.sp_rank,
                is_noop=False,
                require_grad=False,
                expect_nonzero_delta=True,
            )
            del post_action_prediction, post_action_hidden

            pre_audits: list[Any] = [None] * core.WORLD_SIZE
            adapted_audits: list[Any] = [None] * core.WORLD_SIZE
            post_audits: list[Any] = [None] * core.WORLD_SIZE
            post_action_audits: list[Any] = [None] * core.WORLD_SIZE
            dist.all_gather_object(
                pre_audits, pre_audit, group=parallel.world_group
            )
            dist.all_gather_object(
                adapted_audits, adapted_audit, group=parallel.world_group
            )
            dist.all_gather_object(
                post_audits, post_audit, group=parallel.world_group
            )
            dist.all_gather_object(
                post_action_audits,
                post_action_audit,
                group=parallel.world_group,
            )
            _validate_world4_operator_geometry(
                gathered_audits=pre_audits,
                route=noop_route,
                expect_noop=True,
            )
            _validate_world4_operator_geometry(
                gathered_audits=adapted_audits,
                route=action_route,
                expect_noop=False,
            )
            _validate_world4_operator_geometry(
                gathered_audits=post_audits,
                route=post_noop_route,
                expect_noop=True,
            )
            _validate_world4_operator_geometry(
                gathered_audits=post_action_audits,
                route=post_action_route,
                expect_noop=False,
            )
            wrapper_names = {
                f"blocks.{block}.attn2.{projection}"
                for block in core.ACTION_BLOCK_INDICES
                for projection in ("to_q", "to_out.0")
            }
            if any(
                item.get("all_local_active_wrapper_deltas_nonzero") is not True
                for item in post_action_audits
            ) or any(
                not any(
                    float(item["selected_delta_l2_by_wrapper"][wrapper]) > 0.0
                    for item in post_action_audits
                )
                for wrapper in wrapper_names
            ):
                fail("Stage O post-update real per-wrapper delta gate failed")
            local_sentinel = {
                "rank": parallel.contract.rank,
                "pre_velocity": _sentinel_bytes(pre_prediction),
                "post_velocity": _sentinel_bytes(post_prediction),
                "pre_block22_hidden": _sentinel_bytes(pre_hidden),
                "post_block22_hidden": _sentinel_bytes(post_hidden),
                "velocity_bit_exact": torch.equal(
                    pre_prediction, post_prediction
                ),
                "block22_hidden_bit_exact": torch.equal(
                    pre_hidden, post_hidden
                ),
            }
            sentinel_world: list[Any] = [None] * core.WORLD_SIZE
            dist.all_gather_object(
                sentinel_world, local_sentinel, group=parallel.world_group
            )
            if [item.get("rank") for item in sentinel_world] != list(
                range(core.WORLD_SIZE)
            ) or any(
                item["pre_velocity"] != item["post_velocity"]
                or item["pre_block22_hidden"] != item["post_block22_hidden"]
                or item["velocity_bit_exact"] is not True
                or item["block22_hidden_bit_exact"] is not True
                for item in sentinel_world
            ):
                fail("Stage O exact-noop sentinel WORLD4 closure differs")
            if position == 0:
                resource_callback("o_first_optimizer_step")
        except Exception:
            composite.operator.clear_runtime_audits()
            _rollback_active_step(
                active=active,
                optimizer=optimizer,
                parameter_state=parameter_snapshot,
                optimizer_state=optimizer_snapshot,
            )
            if named_parameters_sha256(active) != previous_sha:
                fail("Stage O rollback did not restore exact operator bytes")
            raise

        accepted_history_row: Optional[Mapping[str, Any]] = None
        try:
            local = {
                "stage": "O",
                "stage_update": stage_update,
                "global_update": global_update_offset + stage_update,
                "row_id": row.row_id,
                "source_iid": row.source_iid,
                "seed": row.seed,
                "branch": row.branch,
                "start_state": "q0",
                "real_source_video_sha256": row.source_video_sha256,
                "instruction_sha256": row.instruction_sha256,
                "quotient_tensor_sha256": row.quotient_sha256,
                "camera_nuisance_sha256": row.camera_nuisance_sha256,
                "appearance_nuisance_sha256": row.appearance_nuisance_sha256,
                "preoptimizer_coordinate_admission_digest": admission.receipt[
                    "digest"
                ],
                "noise_seed": noise_seed,
                "schedule_index": coordinate.schedule_index,
                "timestep_int64": coordinate.timestep,
                "sigma": coordinate.sigma,
                "sigma_float32_be_hex": coordinate.sigma_float32_be_hex,
                "memory_input_kind": "same_noise_forward_noised_source",
                "tensor_identities": dict(packed.tensor_identities),
                "memory_receipt": dict(memory.receipt()),
                "planner_phase_code_raw_sha256": hashlib.sha256(
                    _tensor_bytes(phase_code)
                ).hexdigest(),
                "predicted_phi_v1_raw_sha256": predicted_code_sha,
                "loss": loss_value,
                "objective_variant": "stage_o_zero_init_safe_cosine_only_update1",
                "cosine_denominator_eps": core.OPERATOR_ZERO_INIT_COSINE_EPS,
                "zero_init_seen": zero_init_seen,
                "first_step_output_up_gradient_positive": (
                    stage_update != 1 or gradient_groups["output_up"] > 0.0
                ),
                "preclip_gradient_norm_sp4_mean": preclip,
                "clip_return_norm": clipped,
                "gradient_group_norms": gradient_groups,
                "real_per_wrapper_gradient_norms": wrapper_gradient_norms,
                "zero_dependency_gradient_excluded_by_positive_norm": True,
                "operator_parameter_sha256_after": operator_sha,
                "operator_scope": "blocks_0_22_attn2_to_q_and_to_out_0_only",
                "carrier_blocks_8_12_16_20_frozen": True,
                "planner_frozen": True,
                "base_frozen": True,
                "reverse_excluded_without_q1": True,
                "canonical_noop_exact_zero": True,
                "phase0_hard_bypass": True,
                "source_reference_padding_direct_bytes_unchanged": True,
                "noop_sentinel_pre_post_bit_exact": True,
                "noop_sentinel_by_rank": sentinel_world,
                "pre_noop_operator_audits_by_rank": pre_audits,
                "adapted_operator_audits_by_rank": adapted_audits,
                "post_noop_operator_audits_by_rank": post_audits,
                "post_update_action_operator_audits_by_rank": (
                    post_action_audits
                ),
                "all_post_update_wrapper_deltas_nonzero": True,
                "optimizer_payload_fields": sorted(payload),
                "self_generated_rgb_latent_noise_velocity_read": False,
                "action_family_identifier_consumed": False,
                "optimizer_step_executed": True,
                "rollback_required_on_any_gate_failure": True,
            }
            logical = {
                key: value
                for key, value in local.items()
                if key
                not in {
                    "noop_sentinel_by_rank",
                    "pre_noop_operator_audits_by_rank",
                    "adapted_operator_audits_by_rank",
                    "post_noop_operator_audits_by_rank",
                    "post_update_action_operator_audits_by_rank",
                }
            }
            logical_world: list[Any] = [None] * core.WORLD_SIZE
            dist.all_gather_object(
                logical_world, logical, group=parallel.world_group
            )
            if logical_world != [logical] * core.WORLD_SIZE:
                fail("SP4 ranks did not execute the same logical Stage O row")
            accepted_history_row = {
                "schema_version": PO_HISTORY_SCHEMA,
                **local,
            }
            history.append(accepted_history_row)
            if (
                len(rows) == core.STAGE_UPDATES["O"]
                and stage_update in O_CHECKPOINT_STEPS
            ):
                checkpoint_callback("O", stage_update, history)
        except Exception:
            if accepted_history_row is not None and history:
                if history[-1] is not accepted_history_row:
                    fail("Stage O rollback history identity differs")
                history.pop()
            _rollback_active_step(
                active=active,
                optimizer=optimizer,
                parameter_state=parameter_snapshot,
                optimizer_state=optimizer_snapshot,
            )
            if named_parameters_sha256(active) != previous_sha:
                fail("Stage O post-step acceptance rollback was not exact")
            raise
        previous_sha = operator_sha
        del (
            teacher,
            camera,
            appearance,
            payload,
            clean_source,
            action_view,
            phase_code,
            epsilon,
            packed,
            memory,
            carrier_route,
            inactive_before,
            parameter_snapshot,
            optimizer_snapshot,
            pre_prediction,
            pre_hidden,
            adapted_hidden,
            predicted_code,
            loss,
            post_prediction,
            post_hidden,
            post_action_audit,
            post_action_route,
            post_action_audits,
            local,
            logical,
            logical_world,
        )
        gc.collect()
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", choices=("joint_source_anchored_v1",), required=True
    )
    parser.add_argument(
        "--execution-profile", choices=ACTION_PROFILES, required=True
    )
    parser.add_argument(
        "--parallel-topology", choices=(core.TOPOLOGY,), required=True
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument(
        "--source-manifest", "--source-only-manifest", dest="source_manifest", required=True
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        "--expected-source-only-manifest-sha256",
        dest="expected_source_manifest_sha256",
        default=core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256,
    )
    parser.add_argument("--representation-manifest", required=True)
    parser.add_argument(
        "--expected-representation-manifest-sha256", required=True
    )
    parser.add_argument("--source-pair-manifest", required=True)
    parser.add_argument(
        "--expected-source-pair-manifest-sha256", required=True
    )
    parser.add_argument("--phi-operator-coordinate-manifest")
    parser.add_argument(
        "--expected-phi-operator-coordinate-manifest-sha256"
    )
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument(
        "--expected-resume-checkpoint-sha256",
        default=EXPECTED_R64_CHECKPOINT_SHA256,
    )
    parser.add_argument("--resume-receipt", required=True)
    parser.add_argument(
        "--expected-resume-receipt-sha256",
        default=EXPECTED_R64_RECEIPT_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--learning-rate", type=float, default=core.DEFAULT_LEARNING_RATE
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=core.DEFAULT_MAX_GRAD_NORM
    )
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument(
        "--gpu-memory-limit-gib", type=float, default=core.GPU_MEMORY_LIMIT_GIB
    )
    parser.add_argument(
        "--host-memory-limit-gib", type=float, default=core.HOST_MEMORY_LIMIT_GIB
    )
    parser.add_argument(
        "--expected-bernini-commit",
        default=visual.PINNED_BERNINI_SOURCE_COMMIT,
    )
    parser.add_argument(
        "--expected-veomni-commit", default=EXPECTED_VEOMNI_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=old_contract.EXPECTED_CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument("--ack-upstream-training-use-forbidden", action="store_true")
    parser.add_argument(
        "--ack-user-authorized-exploratory-training", action="store_true"
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if (
        args.experiment != "joint_source_anchored_v1"
        or args.execution_profile not in ACTION_PROFILES
        or args.parallel_topology != core.TOPOLOGY
    ):
        fail("P/O runtime requires one joint WORLD4=DP1xSP4 experiment")
    if (
        args.learning_rate != core.DEFAULT_LEARNING_RATE
        or args.max_grad_norm != core.DEFAULT_MAX_GRAD_NORM
        or args.gpu_memory_limit_gib != core.GPU_MEMORY_LIMIT_GIB
        or args.host_memory_limit_gib != core.HOST_MEMORY_LIMIT_GIB
    ):
        fail("P/O optimizer/resource settings differ from the reviewed contract")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        fail("P/O Gaussian seed must lie in [0,2^63)")
    if (
        args.ack_upstream_training_use_forbidden is not True
        or args.ack_user_authorized_exploratory_training is not True
    ):
        fail("both exploratory source-data acknowledgements are required")
    for name in (
        "expected_source_manifest_sha256",
        "expected_representation_manifest_sha256",
        "expected_source_pair_manifest_sha256",
        "expected_resume_checkpoint_sha256",
        "expected_resume_receipt_sha256",
        "expected_checkpoint_tree_sha256",
        "expected_checkpoint_content_manifest_sha256",
    ):
        _sha256(getattr(args, name), label=name)
    if args.execution_profile in OPERATOR_PROFILES:
        if (
            type(args.phi_operator_coordinate_manifest) is not str
            or type(args.expected_phi_operator_coordinate_manifest_sha256)
            is not str
        ):
            fail("Stage O requires the sealed Phi operator coordinate manifest")
        _sha256(
            args.expected_phi_operator_coordinate_manifest_sha256,
            label="expected_phi_operator_coordinate_manifest_sha256",
        )
    elif (
        args.phi_operator_coordinate_manifest is not None
        or args.expected_phi_operator_coordinate_manifest_sha256 is not None
    ):
        fail("smoke-p must not consume Stage-O nuisance coordinate bytes")
    if (
        args.expected_source_manifest_sha256
        != core.EXPECTED_SOURCE_ONLY_MANIFEST_SHA256
        or args.expected_resume_checkpoint_sha256
        != EXPECTED_R64_CHECKPOINT_SHA256
        or args.expected_resume_receipt_sha256 != EXPECTED_R64_RECEIPT_SHA256
        or args.expected_bernini_commit
        != visual.PINNED_BERNINI_SOURCE_COMMIT
        or args.expected_veomni_commit != EXPECTED_VEOMNI_COMMIT
        or args.expected_checkpoint_tree_sha256
        != old_contract.EXPECTED_CHECKPOINT_TREE_SHA256
        or args.expected_checkpoint_content_manifest_sha256
        != EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
    ):
        fail("pinned R64/source/model/runtime identity differs")
    output = Path(args.output).expanduser()
    if (
        not output.is_absolute()
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        fail("P/O output must be one fresh absolute directory")


def _component_hashes(
    composite: core.CompositeHandle, *, world_group: Any, label: str
) -> Mapping[str, str]:
    return {
        stage: _parameter_consensus(
            rows,
            world_group=world_group,
            label=f"{label} component {stage}",
        )
        for stage, rows in composite.named_parameter_groups().items()
    }


def _checkpoint_filename(stage: str, stage_update: int) -> str:
    if stage == "P" and stage_update in P_CHECKPOINT_STEPS:
        return f"stage_p_u{stage_update:03d}_global_u{64 + stage_update:03d}_composite_checkpoint.pt"
    if stage == "O" and stage_update in O_CHECKPOINT_STEPS:
        if stage_update == core.STAGE_UPDATES["O"]:
            return FINAL_PO_CHECKPOINT_NAME
        return f"stage_o_u{stage_update:03d}_global_u{88 + stage_update:03d}_composite_checkpoint.pt"
    fail("P/O formal checkpoint cadence differs")


def _peak_resource_vectors(
    milestones: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Mapping[str, Any]:
    if not milestones:
        fail("P/O resource milestones are absent")
    return {
        "gpu_peak_reserved_gib_by_rank": [
            max(float(rows[rank]["gpu_peak_reserved_gib"]) for rows in milestones.values())
            for rank in range(core.WORLD_SIZE)
        ],
        "host_peak_rss_gib_by_rank": [
            max(float(rows[rank]["host_peak_rss_gib"]) for rows in milestones.values())
            for rank in range(core.WORLD_SIZE)
        ],
        "host_cgroup_peak_gib_by_rank": [
            max(float(rows[rank]["host_cgroup_peak_gib"]) for rows in milestones.values())
            for rank in range(core.WORLD_SIZE)
        ],
    }


def main_from_args(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    import clean_source_visual_context_training_v1 as source_data
    import source_self_runtime as runtime
    import train_lora as legacy

    validate_cli(args)
    source_manifest_path = _plain_file(
        args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
        label="source-only manifest",
    )
    checkpoint_manifest_path = _plain_file(
        args.checkpoint_content_manifest,
        expected_sha256=args.expected_checkpoint_content_manifest_sha256,
        label="checkpoint content manifest",
    )
    source_manifest = source_data.load_source_only_split_manifest(
        source_manifest_path, verify_files=True
    )
    source_authorization = source_data.authorize_exploratory_training(
        source_manifest,
        ack_upstream_training_use_forbidden=(
            args.ack_upstream_training_use_forbidden
        ),
        ack_user_authorized_exploratory_training=(
            args.ack_user_authorized_exploratory_training
        ),
    )
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise GenericActionPORuntimeError(str(error)) from error
    if (
        transformer_config.get("num_attention_heads") != 12
        or legacy.CHECKPOINT_TREE_SHA256
        != args.expected_checkpoint_tree_sha256
    ):
        fail("pinned Bernini transformer/checkpoint differs")
    packed_sp_audit = native_r.audit_packed_sp_sources(
        bernini_root, veomni_root
    )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import diffusers as diffusers_package
    from diffusers import UniPCMultistepScheduler, __version__ as diffusers_version
    from diffusers.models import AutoencoderKLWan
    from transformers import AutoTokenizer, __version__ as transformers_version
    import bernini.pipeline as bernini_pipeline
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    import infer_lora as q0_inference
    from tools import materialize_vae as q0_materialize_vae

    q0_inference_path = Path(q0_inference.__file__).resolve(strict=True)
    if (
        q0_inference_path != (METHOD_ROOT / "infer_lora.py").resolve(strict=True)
        or file_sha256(q0_inference_path) != EXPECTED_Q0_INFER_LORA_SHA256
        or Path(bernini_pipeline.__file__).resolve(strict=True)
        != (bernini_root / "bernini/pipeline.py").resolve(strict=True)
        or diffusers_version != EXPECTED_DIFFUSERS_VERSION
    ):
        fail("q0 infer_lora/diffusers byte-version pin differs")
    try:
        bernini_inference_source_files = q0_inference.validate_inference_source_files(
            bernini_root
        )
    except q0_inference.InferenceContractError as error:
        raise GenericActionPORuntimeError(str(error)) from error
    vae_implementation_module = sys.modules.get(AutoencoderKLWan.__module__)
    if vae_implementation_module is None:
        fail("AutoencoderKLWan implementation module is absent")
    q0_vae_implementation_admission = admit_q0_vae_implementation(
        autoencoder_class=AutoencoderKLWan,
        diffusers_package=diffusers_package,
        implementation_module=vae_implementation_module,
    )

    topology = runtime.parallel_topology(core.TOPOLOGY)
    distributed = runtime.distributed_contract(topology=topology)
    if (
        distributed.world_size != core.WORLD_SIZE
        or distributed.local_world_size != core.WORLD_SIZE
        or distributed.topology.dp_size != core.DP_SIZE
        or distributed.topology.sp_size != core.SP_SIZE
        or distributed.arm_index != 0
        or distributed.rank != distributed.local_rank
    ):
        fail("P/O trainer requires one-node WORLD4=DP1xSP4, one shared model")
    device = runtime.initialise_distributed(distributed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    parallel = runtime.validate_parallel_state(
        distributed, init_parallel_state(ulysses_size=core.SP_SIZE)
    )
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_content_identity = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="base checkpoint content verification",
        callback=lambda: native_r.validate_checkpoint_content(
            checkpoint,
            checkpoint_manifest_path,
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        ),
    )
    if not isinstance(checkpoint_content_identity, Mapping):
        fail("base checkpoint content identity is absent")
    checkpoint_content_identity = dict(checkpoint_content_identity)

    authority = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="P24/O16 action authority validation",
        callback=lambda: load_action_authority(args),
    )
    if not isinstance(authority, ActionAuthority):
        fail("broadcast action authority type differs")
    planner_teacher_bank = admit_planner_teachers(authority.planner_rows)
    planner_admission_identity = object_sha256(
        [
            {
                "row_id": row.row_id,
                "quotient_tensor_sha256": row.quotient_sha256,
                "tensor_raw_sha256": hashlib.sha256(
                    _tensor_bytes(planner_teacher_bank[row.row_id])
                ).hexdigest(),
            }
            for row in authority.planner_rows
        ]
    )
    planner_admission_world: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(
        planner_admission_world,
        planner_admission_identity,
        group=parallel.world_group,
    )
    if planner_admission_world != [planner_admission_identity] * core.WORLD_SIZE:
        fail("pre-optimizer P24 teacher bytes differ across WORLD4")
    operator_coordinate_bank: Mapping[str, OperatorCoordinateAdmission] = {}
    operator_coordinate_admission: Mapping[str, Any] = {}
    if args.execution_profile in OPERATOR_PROFILES:
        operator_coordinate_bank, operator_coordinate_admission = (
            admit_operator_coordinates(authority.operator_rows)
        )
        operator_admission_world: list[Any] = [None] * core.WORLD_SIZE
        dist.all_gather_object(
            operator_admission_world,
            operator_coordinate_admission["digest"],
            group=parallel.world_group,
        )
        if operator_admission_world != [
            operator_coordinate_admission["digest"]
        ] * core.WORLD_SIZE:
            fail("pre-optimizer O16 nuisance math differs across WORLD4")
    generated_p32_sha = p32_raw_sha256()
    sealed_p32 = read_f32le_p32(authority.p32_path, authority.p32_raw_sha256)
    local_p32_sha = hashlib.sha256(_tensor_bytes(sealed_p32)).hexdigest()
    p32_shas: list[Any] = [None] * core.WORLD_SIZE
    dist.all_gather_object(p32_shas, local_p32_sha, group=parallel.world_group)
    if (
        p32_shas != [local_p32_sha] * core.WORLD_SIZE
        or local_p32_sha != authority.p32_raw_sha256
        or generated_p32_sha != authority.p32_raw_sha256
    ):
        fail("fixed P32 bytes differ from the representation authority")

    (
        r64_checkpoint_path,
        r64_checkpoint,
        r64_receipt,
        r64_byte_authority,
    ) = _load_r64_authority(args)
    runtime_dependency_modules = {
        "action_manifest": action_manifest,
        "core": core,
        "exact40": exact40,
        "legacy": legacy,
        "native_r": native_r,
        "old_contract": old_contract,
        "source_data": source_data,
        "source_self_runtime": runtime,
        "visual": visual,
        "q0_infer_lora": q0_inference,
        "bernini_pipeline": bernini_pipeline,
        "diffusers_autoencoder_kl_wan": vae_implementation_module,
        "q0_materialize_vae": q0_materialize_vae,
    }
    runtime_dependency_files = {
        name: {
            "path": str(Path(module.__file__).resolve(strict=True)),
            "file_sha256": file_sha256(
                Path(module.__file__).resolve(strict=True)
            ),
        }
        for name, module in sorted(runtime_dependency_modules.items())
    }
    runtime_dependency_files["po_runtime"] = {
        "path": str(Path(__file__).resolve(strict=True)),
        "file_sha256": file_sha256(Path(__file__).resolve(strict=True)),
    }
    runtime_source_identity = {
        "core_path": str(Path(core.__file__).resolve(strict=True)),
        "core_file_sha256": file_sha256(Path(core.__file__).resolve(strict=True)),
        "po_runtime_path": str(Path(__file__).resolve(strict=True)),
        "po_runtime_file_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "dependency_files": runtime_dependency_files,
    }

    if args.execution_profile == "smoke-p":
        planner_rows = authority.planner_rows[:1]
        operator_rows: tuple[OperatorRow, ...] = ()
    elif args.execution_profile == "smoke-o":
        planner_rows = ()
        operator_rows = authority.operator_rows[:1]
    elif args.execution_profile == "smoke-po25":
        planner_rows = authority.planner_rows
        operator_rows = authority.operator_rows[:1]
    else:
        planner_rows = authority.planner_rows
        operator_rows = authority.operator_rows
    q0_latents: Mapping[str, Any] = {}
    q0_receipts: Mapping[str, Any] = {}
    if operator_rows:
        q0_latents, q0_receipts = _encode_real_q0_sources(
            rows=operator_rows,
            checkpoint=checkpoint,
            rank=distributed.rank,
            device=device,
            world_group=parallel.world_group,
            expected_vae_implementation_admission=(
                q0_vae_implementation_admission
            ),
        )

    legacy.seed_same_sample(args.seed)
    renderer_config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    renderer_config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(renderer_config.to_dict(), checkpoint)
    renderer = None
    for loading_rank in range(core.WORLD_SIZE):
        if distributed.rank == loading_rank:
            renderer = BerniniRendererModel(renderer_config)
            renderer.requires_grad_(False)
            renderer.eval()
            renderer.t5_text_encoder.eval()
            renderer.to(device)
        dist.barrier(group=parallel.world_group)
    if renderer is None:
        fail("rank-serialized P/O renderer load failed")
    transformer = renderer.diff_dec.transformer
    if transformer is None or renderer.diff_dec.transformer_2 is not None:
        fail("P/O runtime requires only frozen transformer_1")
    base_transformer_named = tuple(transformer.named_parameters())
    if not base_transformer_named or any(
        parameter.requires_grad for _, parameter in base_transformer_named
    ):
        fail("loaded Bernini base transformer is absent/non-frozen")
    base_transformer_sha256 = _parameter_consensus(
        base_transformer_named,
        world_group=parallel.world_group,
        label="initial frozen base transformer",
    )
    renderer.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": core.composite_checkpoint_route_context_fn,
        }
    )
    if not bool(getattr(transformer, "gradient_checkpointing", False)):
        fail("P/O runtime requires composite route checkpoint replay")
    composite = core.install_composite_v1(
        transformer,
        experiment=args.experiment,
        runtime_source_commit=bernini_revision,
        model_revision=visual.PINNED_BERNINI_MODEL_REVISION,
        checkpoint_manifest_sha256=(
            args.expected_checkpoint_content_manifest_sha256
        ),
        initialization_seed=args.seed,
    )
    if composite.carrier is None:
        fail("P/O joint runtime did not install the R64 source carrier")
    loaded_component_sha256 = _load_component_state_strict(
        composite, r64_checkpoint, world_group=parallel.world_group
    )
    parameter_counts = composite.parameter_count_receipt()
    composite.planner.eval()

    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=5.0,
    )
    schedule_audit = exact40.audit_runtime_unipc_schedule(scheduler)
    if schedule_audit.get("schedule_sha256") != old_contract.EXPECTED_SCHEDULE_SHA256:
        fail("runtime UniPC exact40 schedule differs")
    del scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    text_rows: Sequence[PlannerRow] = tuple(planner_rows) + tuple(operator_rows)
    text_bank, text_receipts = _encode_instruction_bank(
        renderer=renderer,
        tokenizer=tokenizer,
        rows=text_rows,
        include_noop=bool(operator_rows),
        device=device,
        runtime=runtime,
        world_group=parallel.world_group,
    )
    renderer.t5_text_encoder = None
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if renderer.t5_text_encoder is not None:
        fail("frozen UMT5 encoder was not released before optimizer updates")
    p32 = sealed_p32.to(device=device, dtype=torch.float32).detach()
    if hashlib.sha256(_tensor_bytes(p32)).hexdigest() != authority.p32_raw_sha256:
        fail("device P32 bytes differ from sealed CPU P32 bytes")
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    operator_off_parity_receipt: Mapping[str, Any] = {}
    if operator_rows:
        operator_off_parity_receipt = preoptimizer_operator_off_parity(
            row=operator_rows[0],
            composite=composite,
            text_bank=text_bank,
            q0_latents=q0_latents,
            renderer=renderer,
            transformer=transformer,
            rope=rope,
            device=device,
            parallel=parallel,
            runtime=runtime,
            seed=args.seed,
        )

    # AdamW is deliberately constructed only after P24/O16 tensor admission,
    # dual-UMT5 closure, deterministic q0 encoding, and operator-off parity.
    controller = core.StageOptimizerController(
        composite, learning_rate=args.learning_rate
    )
    optimizer_load_receipt = _load_optimizer_state_strict(
        controller, r64_checkpoint
    )
    if optimizer_load_receipt["planner_operator_state_initially_absent"] is not True:
        fail("R64 unexpectedly contains planner/operator optimizer state")
    del r64_checkpoint
    gc.collect()

    output = Path(args.output)
    staging = output / ".po_runtime_staging"
    published = output / "artifacts"

    def create_output() -> bool:
        os.mkdir(output, mode=0o700)
        os.chmod(output, 0o700)
        os.mkdir(staging, mode=0o700)
        os.chmod(staging, 0o700)
        if output.stat().st_mode & 0o777 != 0o700:
            fail("P/O output directory mode is not exact 0700")
        if staging.stat().st_mode & 0o777 != 0o700 or published.exists():
            fail("P/O private staging directory contract differs")
        return True

    resource_milestones: dict[str, tuple[Mapping[str, Any], ...]] = {}

    def resource_callback(label: str) -> None:
        if label in resource_milestones:
            fail("P/O resource milestone was recorded twice")
        resource_milestones[label] = collect_and_gate_resources(
            stage=label,
            rank=distributed.rank,
            device=device,
            world_group=parallel.world_group,
            gpu_limit_gib=args.gpu_memory_limit_gib,
            host_limit_gib=args.host_memory_limit_gib,
        )

    resource_callback("model_load")
    _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="private P/O output creation",
        callback=create_output,
    )

    planner_history: list[Mapping[str, Any]] = []
    operator_history: list[Mapping[str, Any]] = []
    checkpoint_records: list[Mapping[str, Any]] = []

    def write_checkpoint(
        *,
        stage: str,
        stage_update: int,
        history: Sequence[Mapping[str, Any]],
        filename: str,
        terminal_audit: Optional[Mapping[str, Any]],
        complete_action_training: bool,
    ) -> Mapping[str, Any]:
        component_sha = _component_hashes(
            composite,
            world_group=parallel.world_group,
            label=f"checkpoint {stage} u{stage_update:03d}",
        )
        p_updates = (
            stage_update
            if stage == "P"
            else (
                core.STAGE_UPDATES["P"]
                if args.execution_profile in {"smoke-po25", "resume-po40"}
                else 0
            )
        )
        o_updates = stage_update if stage == "O" else 0
        staging_path = staging / filename
        published_path = published / filename
        p_complete = p_updates == core.STAGE_UPDATES["P"]
        o_complete = o_updates == core.STAGE_UPDATES["O"]
        if stage == "P":
            in_progress_stage = None if p_complete else "P"
        else:
            in_progress_stage = None if o_complete else "O"
        pending_stages = ([] if p_complete else ["P"]) + (
            [] if o_complete else ["O"]
        )

        def write_rank0_checkpoint() -> Mapping[str, Any]:
            value = {
                "schema_version": PO_CHECKPOINT_SCHEMA,
                "method": METHOD,
                "experiment": args.experiment,
                "execution_profile": args.execution_profile,
                "resumed_from_r64": {
                    "path": str(r64_checkpoint_path),
                    "file_sha256": EXPECTED_R64_CHECKPOINT_SHA256,
                    "receipt_path": str(
                        Path(args.resume_receipt).expanduser().resolve(strict=True)
                    ),
                    "receipt_file_sha256": args.expected_resume_receipt_sha256,
                    "stage_r_updates_reexecuted": 0,
                },
                "completed_stages": (
                    ["R", "P", "O"]
                    if complete_action_training
                    else (
                        ["R", "P"]
                        if p_updates == core.STAGE_UPDATES["P"]
                        else ["R"]
                    )
                ),
                "in_progress_stage": in_progress_stage,
                "next_stage": "O" if p_complete and not o_updates else None,
                "pending_stages": pending_stages,
                "completed_stage_updates": {
                    "R": core.STAGE_UPDATES["R"],
                    "P": p_updates,
                    "O": o_updates,
                },
                "publication_state": "requires_sibling_completion_manifest",
                "completion_manifest_required": True,
                "complete_action_training": False,
                "expected_complete_action_training_after_manifest": (
                    complete_action_training
                ),
                "complete_action_result": False,
                "scientific_success_claimed": False,
                "action_authority": authority.receipt(),
                "source_manifest_file_sha256": (
                    args.expected_source_manifest_sha256
                ),
                "initial_r64_component_sha256": loaded_component_sha256,
                "current_component_sha256": component_sha,
                "component_state": _component_cpu_state(composite),
                "optimizer_state": controller.optimizer.state_dict(),
                "optimizer_state_keys": _optimizer_state_keys(controller),
                "active_stage": controller.active_stage,
                "history_sha256": object_sha256(list(history)),
                "history": list(history),
                "operator_scope": {
                    "blocks": list(core.ACTION_BLOCK_INDICES),
                    "projections": ["attn2.to_q", "attn2.to_out.0"],
                    "carrier_blocks_frozen_during_po": list(
                        core.CARRIER_BLOCK_INDICES
                    ),
                },
                "terminal_toctou_audit": terminal_audit,
            }
            runtime.atomic_torch_save(staging_path, value)
            return {
                "stage": stage,
                "stage_update": stage_update,
                "global_update": (
                    core.STAGE_UPDATES["R"] + p_updates + o_updates
                ),
                "path": str(published_path),
                "relative_path": filename,
                "file_sha256": file_sha256(staging_path),
                "schema_version": PO_CHECKPOINT_SCHEMA,
                "publication_state": "requires_sibling_completion_manifest",
                "complete_action_training": False,
                "expected_complete_action_training_after_manifest": (
                    complete_action_training
                ),
                "complete_action_result": False,
            }

        result = _rank0_call(
            rank=distributed.rank,
            world_group=parallel.world_group,
            label=f"{stage} u{stage_update:03d} checkpoint write",
            callback=write_rank0_checkpoint,
        )
        if not isinstance(result, Mapping):
            fail("P/O checkpoint write receipt differs")
        return dict(result)

    def checkpoint_callback(
        stage: str,
        stage_update: int,
        stage_history: Sequence[Mapping[str, Any]],
    ) -> None:
        if stage == "O" and stage_update == core.STAGE_UPDATES["O"]:
            # O16 is staged exactly once by the terminal bundle path.  It is
            # never published without the later resource/TOCTOU gates and
            # sibling completion manifest.
            return
        if stage == "P":
            combined = list(stage_history)
        elif stage == "O":
            combined = list(planner_history) + list(stage_history)
        else:
            fail("checkpoint callback stage differs")
        checkpoint_records.append(
            write_checkpoint(
                stage=stage,
                stage_update=stage_update,
                history=combined,
                filename=_checkpoint_filename(stage, stage_update),
                terminal_audit=None,
                complete_action_training=False,
            )
        )

    started = time.monotonic()
    if planner_rows:
        planner_history = _run_planner_updates(
            rows=planner_rows,
            composite=composite,
            controller=controller,
            text_bank=text_bank,
            planner_teacher_bank=planner_teacher_bank,
            device=device,
            parallel=parallel,
            runtime=runtime,
            max_grad_norm=args.max_grad_norm,
            resource_callback=resource_callback,
            checkpoint_callback=checkpoint_callback,
        )
    if operator_rows:
        operator_history = _run_operator_updates(
            rows=operator_rows,
            composite=composite,
            controller=controller,
            text_bank=text_bank,
            q0_latents=q0_latents,
            operator_coordinate_bank=operator_coordinate_bank,
            renderer=renderer,
            transformer=transformer,
            rope=rope,
            p32=p32,
            device=device,
            parallel=parallel,
            runtime=runtime,
            seed=args.seed,
            global_update_offset=(
                core.STAGE_UPDATES["R"]
                + (
                    core.STAGE_UPDATES["P"]
                    if args.execution_profile in {"smoke-po25", "resume-po40"}
                    else 0
                )
            ),
            max_grad_norm=args.max_grad_norm,
            resource_callback=resource_callback,
            checkpoint_callback=checkpoint_callback,
        )
    resource_callback("training_complete")
    elapsed = time.monotonic() - started
    full_history = planner_history + operator_history
    expected_history = {
        "smoke-p": 1,
        "smoke-o": 1,
        "smoke-po25": core.STAGE_UPDATES["P"] + 1,
        "resume-po40": core.STAGE_UPDATES["P"] + core.STAGE_UPDATES["O"],
    }[args.execution_profile]
    if len(full_history) != expected_history:
        fail("P/O terminal history length differs")
    final_component_sha256 = _component_hashes(
        composite, world_group=parallel.world_group, label="terminal"
    )
    if final_component_sha256["R"] != EXPECTED_R64_CARRIER_PARAMETER_SHA256:
        fail("frozen R64 carrier changed during P/O")
    if args.execution_profile == "smoke-p":
        scope_ok = (
            final_component_sha256["P"]
            != EXPECTED_R64_PLANNER_PARAMETER_SHA256
            and final_component_sha256["O"]
            == EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        )
    elif args.execution_profile == "smoke-o":
        scope_ok = (
            final_component_sha256["P"]
            == EXPECTED_R64_PLANNER_PARAMETER_SHA256
            and final_component_sha256["O"]
            != EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        )
    elif args.execution_profile == "smoke-po25":
        scope_ok = (
            final_component_sha256["P"]
            != EXPECTED_R64_PLANNER_PARAMETER_SHA256
            and final_component_sha256["O"]
            != EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        )
    else:
        scope_ok = (
            final_component_sha256["P"]
            != EXPECTED_R64_PLANNER_PARAMETER_SHA256
            and final_component_sha256["O"]
            != EXPECTED_R64_OPERATOR_PARAMETER_SHA256
        )
    if not scope_ok:
        fail("P/O terminal component parameter-scope closure differs")
    terminal_base_transformer_sha256 = _parameter_consensus(
        base_transformer_named,
        world_group=parallel.world_group,
        label="terminal frozen base transformer",
    )
    if terminal_base_transformer_sha256 != base_transformer_sha256:
        fail("frozen Bernini base changed during P/O")
    optimizer_state_keys = _optimizer_state_keys(controller)
    groups = composite.named_parameter_groups()
    expected_state_stages = {
        "smoke-p": {"R", "P"},
        "smoke-o": {"R", "O"},
        "smoke-po25": {"R", "P", "O"},
        "resume-po40": {"R", "P", "O"},
    }[args.execution_profile]
    for stage, rows in groups.items():
        observed_count = optimizer_state_keys[stage]["state_parameter_count"]
        expected_count = len(rows) if stage in expected_state_stages else 0
        if observed_count != expected_count:
            fail(f"terminal optimizer-state key closure differs for Stage {stage}")

    def terminal_toctou() -> Mapping[str, Any]:
        drifted_runtime_dependencies = [
            name
            for name, identity in runtime_source_identity[
                "dependency_files"
            ].items()
            if file_sha256(Path(identity["path"]).resolve(strict=True))
            != identity["file_sha256"]
        ]
        if drifted_runtime_dependencies:
            fail(
                "P/O runtime dependency changed during training: "
                + ",".join(drifted_runtime_dependencies)
            )
        terminal_q0_vae_implementation_admission = (
            admit_q0_vae_implementation(
                autoencoder_class=AutoencoderKLWan,
                diffusers_package=diffusers_package,
                implementation_module=vae_implementation_module,
            )
        )
        if (
            terminal_q0_vae_implementation_admission
            != q0_vae_implementation_admission
        ):
            fail("registered q0 VAE implementation changed during P/O")
        _, terminal_r64_checkpoint_identity = read_verified_file_bytes(
            r64_checkpoint_path,
            expected_sha256=EXPECTED_R64_CHECKPOINT_SHA256,
            label="terminal R64 composite checkpoint",
        )
        terminal_r64_receipt_path = Path(args.resume_receipt).expanduser().resolve(
            strict=True
        )
        _, terminal_r64_receipt_identity = read_verified_file_bytes(
            terminal_r64_receipt_path,
            expected_sha256=args.expected_resume_receipt_sha256,
            label="terminal R64 run receipt",
        )
        if (
            terminal_r64_checkpoint_identity
            != r64_byte_authority["checkpoint"]
            or terminal_r64_receipt_identity != r64_byte_authority["receipt"]
        ):
            fail("R64 single-FD byte/inode authority changed during P/O")
        terminal_q0_file_identities: dict[str, Any] = {}
        for iid, row in sorted(
            {
                row.source_iid: row for row in operator_rows
            }.items()
        ):
            q0_path = _plain_file(
                row.source_video_path,
                expected_sha256=None,
                label=f"terminal {iid} real q0 source",
            )
            _, q0_file_identity = read_verified_file_bytes(
                q0_path,
                expected_sha256=row.source_video_sha256,
                label=f"terminal {iid} real q0 source",
            )
            initial_identity = q0_receipts.get(iid, {}).get(
                "decoded_metadata", {}
            ).get("source_file_identity")
            # smoke profiles encode only the selected q0; terminal closure is
            # exactly over those optimizer-readable sources.
            if not isinstance(initial_identity, Mapping):
                fail("initial real q0 single-FD authority receipt is absent")
            if q0_file_identity != initial_identity:
                fail("real q0 single-FD byte/inode authority changed")
            terminal_q0_file_identities[iid] = q0_file_identity
        if (
            file_sha256(source_manifest_path)
            != args.expected_source_manifest_sha256
            or file_sha256(checkpoint_manifest_path)
            != args.expected_checkpoint_content_manifest_sha256
        ):
            fail("P/O manifest/R64/runtime source changed during training")
        terminal_action_authority = load_action_authority(args)
        if terminal_action_authority.receipt() != authority.receipt():
            fail("P24/O16 action authority changed during training")
        if operator_rows:
            _, terminal_operator_admission = admit_operator_coordinates(
                terminal_action_authority.operator_rows
            )
            if terminal_operator_admission != operator_coordinate_admission:
                fail("terminal O16 nuisance mathematical closure changed")
        terminal_source_manifest = source_data.load_source_only_split_manifest(
            source_manifest_path, verify_files=True
        )
        if terminal_source_manifest.receipt() != source_manifest.receipt():
            fail("source-only manifest/files changed during P/O")
        terminal_checkpoint_identity = native_r.validate_checkpoint_content(
            checkpoint,
            checkpoint_manifest_path,
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
        )
        if dict(terminal_checkpoint_identity) != checkpoint_content_identity:
            fail("base checkpoint content changed during P/O")
        try:
            end_bernini, end_veomni, end_bernini_revision, end_veomni_revision = (
                legacy.validate_source_trees(
                    args.bernini_root,
                    args.veomni_root,
                    expected_bernini_commit=args.expected_bernini_commit,
                    expected_veomni_commit=args.expected_veomni_commit,
                )
            )
        except legacy.TrainingContractError as error:
            raise GenericActionPORuntimeError(str(error)) from error
        end_sp_audit = native_r.audit_packed_sp_sources(end_bernini, end_veomni)
        if (
            end_bernini != bernini_root
            or end_veomni != veomni_root
            or end_bernini_revision != bernini_revision
            or end_veomni_revision != veomni_revision
            or end_sp_audit != packed_sp_audit
        ):
            fail("Bernini/VeOmni runtime trees changed during P/O")
        value = {
            "source_manifest_sha256_reverified": (
                args.expected_source_manifest_sha256
            ),
            "representation_manifest_sha256_reverified": (
                authority.representation_file_sha256
            ),
            "source_pair_manifest_sha256_reverified": authority.pair_file_sha256,
            "phi_operator_coordinate_manifest_sha256_reverified": (
                authority.coordinate_manifest_file_sha256
            ),
            "r64_checkpoint_sha256_reverified": EXPECTED_R64_CHECKPOINT_SHA256,
            "r64_receipt_sha256_reverified": args.expected_resume_receipt_sha256,
            "r64_single_fd_byte_authority_reverified": r64_byte_authority,
            "q0_single_fd_file_identities_reverified": (
                terminal_q0_file_identities
            ),
            "checkpoint_content_identity_reverified": object_sha256(
                terminal_checkpoint_identity
            ),
            "bernini_commit_reverified": end_bernini_revision,
            "veomni_commit_reverified": end_veomni_revision,
            "packed_sp_audit_digest_reverified": end_sp_audit["digest"],
            "runtime_source_identity_reverified": runtime_source_identity,
            "q0_vae_implementation_admission_reverified": dict(
                terminal_q0_vae_implementation_admission
            ),
            "unchanged": True,
        }
        return {**value, "digest": object_sha256(value)}

    formal_complete = args.execution_profile == "resume-po40"
    if formal_complete:
        final_checkpoint_record = write_checkpoint(
            stage="O",
            stage_update=core.STAGE_UPDATES["O"],
            history=full_history,
            filename=FINAL_PO_CHECKPOINT_NAME,
            terminal_audit=None,
            complete_action_training=True,
        )
    elif args.execution_profile == "smoke-p":
        final_checkpoint_record = write_checkpoint(
            stage="P",
            stage_update=1,
            history=full_history,
            filename=SMOKE_P_CHECKPOINT_NAME,
            terminal_audit=None,
            complete_action_training=False,
        )
    elif args.execution_profile == "smoke-po25":
        final_checkpoint_record = write_checkpoint(
            stage="O",
            stage_update=1,
            history=full_history,
            filename=SMOKE_PO25_CHECKPOINT_NAME,
            terminal_audit=None,
            complete_action_training=False,
        )
    else:
        final_checkpoint_record = write_checkpoint(
            stage="O",
            stage_update=1,
            history=full_history,
            filename=SMOKE_O_CHECKPOINT_NAME,
            terminal_audit=None,
            complete_action_training=False,
        )
    checkpoint_records.append(final_checkpoint_record)

    def stage_precommit_bundle() -> bool:
        runtime.atomic_json(
            staging / "history.json",
            {
                "schema_version": PO_HISTORY_SCHEMA,
                "publication_state": "requires_sibling_completion_manifest",
                "completion_manifest_required": True,
                "steps": full_history,
                "digest": object_sha256(full_history),
            },
        )
        runtime.atomic_json(
            staging / "run_receipt.json",
            {
                "schema_version": PO_RUNTIME_SCHEMA,
                "publication_state": "staged_uncommitted",
                "execution_profile": args.execution_profile,
                "checkpoint_records": checkpoint_records,
                "history_sha256": object_sha256(full_history),
                "terminal_gates_pending": ["resource", "toctou"],
                "complete": False,
            },
        )
        for path in staging.iterdir():
            if not path.is_file() or path.is_symlink():
                fail("precommit staging contains a non-plain artifact")
            os.chmod(path, 0o400)
        runtime.fsync_directory(staging)
        return True

    _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="P/O precommit artifact staging",
        callback=stage_precommit_bundle,
    )
    resource_callback("post_staging")
    terminal_audit = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="post-staging terminal P/O TOCTOU audit",
        callback=terminal_toctou,
    )
    resource_callback("post_toctou")

    resource_peaks = _peak_resource_vectors(resource_milestones)
    planner_updates = len(planner_history)
    operator_updates = len(operator_history)
    pair_invariants = {
        "status": "r64_resumed_with_hash_bound_action_authority",
        "r64_checkpoint_sha256": EXPECTED_R64_CHECKPOINT_SHA256,
        "r64_receipt_sha256": args.expected_resume_receipt_sha256,
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_content_manifest_sha256": (
            args.expected_checkpoint_content_manifest_sha256
        ),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "representation_manifest_sha256": authority.representation_file_sha256,
        "source_pair_manifest_sha256": authority.pair_file_sha256,
        "phi_operator_coordinate_manifest_sha256": (
            authority.coordinate_manifest_file_sha256
        ),
        "action_row_order_sha256": authority.row_order_sha256,
        "p32_raw_sha256": authority.p32_raw_sha256,
        "gaussian_base_seed": args.seed,
        "o_sigma_mapping": list(core.fixed_sigma_schedule("O")),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "max_grad_norm": args.max_grad_norm,
        },
    }
    unsigned_receipt = {
        "schema_version": PO_RUNTIME_SCHEMA,
        "method": METHOD,
        "complete": True,
        "publication_state": "complete_only_with_completion_manifest",
        "completion_manifest_required": True,
        "completion_manifest_schema": PO_COMPLETION_MANIFEST_SCHEMA,
        "completion_manifest_path": str(published / "completion_manifest.json"),
        "experiment": args.experiment,
        "execution_profile": args.execution_profile,
        "disposable_smoke": not formal_complete,
        "complete_action_training": formal_complete,
        "complete_action_result": False,
        "scientific_success_claimed": False,
        "decoded_evaluation_executed": False,
        "stage_r": {
            "resumed_checkpoint": str(r64_checkpoint_path),
            "checkpoint_sha256": EXPECTED_R64_CHECKPOINT_SHA256,
            "receipt_sha256": args.expected_resume_receipt_sha256,
            "receipt_digest": r64_receipt["receipt_digest"],
            "single_fd_byte_authority": r64_byte_authority,
            "updates_in_p_o_process": 0,
            "completed_updates_before_resume": core.STAGE_UPDATES["R"],
        },
        "planner_updates": planner_updates,
        "operator_updates": operator_updates,
        "optimizer_updates_this_process": planner_updates + operator_updates,
        "checkpoint": final_checkpoint_record,
        "checkpoints": checkpoint_records,
        "contract": core.training_contract_receipt(args.experiment),
        "pair_invariants": pair_invariants,
        "unipc_schedule_audit": dict(schedule_audit),
        "action_authority": authority.receipt(),
        "source_data": {
            **source_manifest.receipt(),
            "manifest_path": str(source_manifest_path),
            "manifest_file_sha256": args.expected_source_manifest_sha256,
            "authorization": dict(source_authorization),
            "optimizer_reads": "none_in_P; manifest-bound_real_q0_only_in_O",
        },
        "q0_frozen_vae_encodings": dict(q0_receipts),
        "text_embeddings": dict(text_receipts),
        "preoptimizer_admission": {
            "planner_p24_teacher_identity": planner_admission_identity,
            "operator_o16_nuisance_math": dict(operator_coordinate_admission),
            "operator_off_parity": dict(operator_off_parity_receipt),
            "adamw_constructed_after_all_applicable_gates": True,
        },
        "q0_encoder_pins": {
            "infer_lora_file_sha256": EXPECTED_Q0_INFER_LORA_SHA256,
            "bernini_inference_source_files": bernini_inference_source_files,
            "diffusers_version": diffusers_version,
            "vae_implementation_admission": dict(
                q0_vae_implementation_admission
            ),
            "vae_module_name_expected": EXPECTED_Q0_VAE_MODULE_NAME,
            "vae_module_name_observed": (
                q0_vae_implementation_admission["observed"]["module_name"]
            ),
            "vae_module_origin_expected": (
                q0_vae_implementation_admission["expected"]["module_origin"]
            ),
            "vae_module_origin_observed": (
                q0_vae_implementation_admission["observed"][
                    "module_file_origin"
                ]
            ),
            "vae_module_sha256_expected": EXPECTED_Q0_VAE_MODULE_SHA256,
            "vae_module_sha256_observed": (
                q0_vae_implementation_admission["observed"]["module_sha256"]
            ),
            "deterministic_algorithms": True,
            "vae_repeated_encode_bit_exact": bool(operator_rows),
        },
        "objectives": {
            "P": {
                "name": "strict_cosine_frozen_umt5_to_detached_phi_v1_q",
                "teacher_shape": [core.LATENT_PHASES, core.PHASE_CODE_WIDTH],
                "noop_optimizer_updates": 0,
            },
            "O": {
                "name": "cosine_phi_v1_current_real_q0_adapted_minus_noop",
                "objective_variant": (
                    "zero_init_safe_cosine_denominator_only_at_stage_update_1"
                ),
                "cosine_denominator_eps": core.OPERATOR_ZERO_INIT_COSINE_EPS,
                "teacher_zero_allowed": False,
                "prediction_zero_allowed_after_update1": False,
                "camera_appearance_scalar_weights": [0.0, 0.0],
                "reverse_updates": 0,
                "q0_real_source_bound": True,
            },
            "self_generated_rgb_latent_noise_velocity_optimizer_input_or_target": False,
            "action_family_identifier_consumed": False,
        },
        "scope": {
            "carrier_blocks": list(core.CARRIER_BLOCK_INDICES),
            "carrier_frozen_during_po": True,
            "operator_blocks": list(core.ACTION_BLOCK_INDICES),
            "operator_projections": ["attn2.to_q", "attn2.to_out.0"],
            "planner_only_active_in_P": True,
            "operator_only_active_in_O": True,
            "base_vae_umt5_native_kv_ffn_frozen": True,
            "canonical_noop_hard_bypass": True,
            "phase0_hard_bypass": True,
        },
        "parameter_counts": parameter_counts,
        "initial_r64_component_sha256": loaded_component_sha256,
        "final_component_sha256": final_component_sha256,
        "frozen_base_transformer_sha256_initial": base_transformer_sha256,
        "frozen_base_transformer_sha256_terminal": (
            terminal_base_transformer_sha256
        ),
        "optimizer_load": optimizer_load_receipt,
        "optimizer_state_keys_terminal": optimizer_state_keys,
        "history": full_history,
        "history_sha256": object_sha256(full_history),
        "resources": {
            "strict_gpu_peak_reserved_limit_gib": args.gpu_memory_limit_gib,
            "strict_host_limit_gib": args.host_memory_limit_gib,
            **resource_peaks,
            "all_strictly_below_limits": True,
            "milestones": {
                key: list(value) for key, value in resource_milestones.items()
            },
        },
        "distributed": {
            "topology": core.TOPOLOGY,
            "world_size": core.WORLD_SIZE,
            "dp_size": core.DP_SIZE,
            "sp_size": core.SP_SIZE,
            "one_shared_model": True,
            "rank_or_gpu_action_family_partition": False,
            "gradient_sync": "SP4_mean",
            "packed_sp_source_audit": packed_sp_audit,
        },
        "model": {
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
            "checkpoint_content_identity": checkpoint_content_identity,
        },
        "runtime_source": runtime_source_identity,
        "runtime": {
            "torch": torch.__version__,
            "torch_hip": str(torch.version.hip),
            "transformers": transformers_version,
            "diffusers": diffusers_version,
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "elapsed_seconds": elapsed,
        },
        "terminal_toctou_audit": terminal_audit,
        "parent_allocation_released": False,
    }
    receipt = {
        **unsigned_receipt,
        "receipt_digest": object_sha256(unsigned_receipt),
    }

    def finalize_and_publish_bundle() -> Mapping[str, Any]:
        runtime.atomic_json(staging / "run_receipt.json", receipt)
        os.chmod(staging / "run_receipt.json", 0o400)
        artifact_rows: list[Mapping[str, Any]] = []
        for path in sorted(staging.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.is_symlink():
                fail("final staging contains a non-plain artifact")
            os.chmod(path, 0o400)
            artifact_rows.append(
                {
                    "relative_path": path.name,
                    "file_sha256": file_sha256(path),
                    "byte_count": path.stat().st_size,
                    "mode": "0400",
                }
            )
        completion_unsigned = {
            "schema_version": PO_COMPLETION_MANIFEST_SCHEMA,
            "method": METHOD,
            "execution_profile": args.execution_profile,
            "publication_state": "complete",
            "complete": True,
            "complete_action_training": formal_complete,
            "complete_action_result": False,
            "run_receipt_digest": receipt["receipt_digest"],
            "terminal_toctou_audit_digest": terminal_audit["digest"],
            "resource_receipt_digest": object_sha256(
                unsigned_receipt["resources"]
            ),
            "artifacts": artifact_rows,
            "atomic_publish": {
                "staging_directory": ".po_runtime_staging",
                "published_directory": "artifacts",
                "same_parent_atomic_rename": True,
                "completion_manifest_written_last_before_rename": True,
                "valid_only_at_published_directory": str(published),
            },
        }
        completion_manifest = {
            **completion_unsigned,
            "manifest_digest": object_sha256(completion_unsigned),
        }
        completion_path = staging / "completion_manifest.json"
        runtime.atomic_json(completion_path, completion_manifest)
        os.chmod(completion_path, 0o400)
        runtime.fsync_directory(staging)
        if published.exists() or staging.parent != published.parent:
            fail("atomic completion publication target differs")
        os.rename(staging, published)
        runtime.fsync_directory(output)
        return completion_manifest

    completion_manifest = _rank0_call(
        rank=distributed.rank,
        world_group=parallel.world_group,
        label="atomic P/O completion-manifest publication",
        callback=finalize_and_publish_bundle,
    )
    if not isinstance(completion_manifest, Mapping):
        fail("published P/O completion manifest receipt differs")
    if distributed.rank == 0:
        print(
            json.dumps(
                {
                    "output": str(output),
                    "execution_profile": args.execution_profile,
                    "planner_updates": planner_updates,
                    "operator_updates": operator_updates,
                    "complete_action_training": formal_complete,
                    "complete_action_result": False,
                    "checkpoint_sha256": final_checkpoint_record["file_sha256"],
                    "receipt_digest": receipt["receipt_digest"],
                    "completion_manifest": str(
                        published / "completion_manifest.json"
                    ),
                    "completion_manifest_digest": completion_manifest[
                        "manifest_digest"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return main_from_args(build_parser().parse_args(argv))


__all__ = [
    "ACTION_PROFILES",
    "ActionAuthority",
    "EXPECTED_R64_CHECKPOINT_SHA256",
    "EXPECTED_R64_RECEIPT_SHA256",
    "GenericActionPORuntimeError",
    "OPERATOR_PROFILES",
    "OperatorRow",
    "PHI_OPERATOR_COORDINATE_SCHEMA",
    "PlannerRow",
    "build_parser",
    "load_action_authority",
    "main",
    "main_from_args",
    "p32_raw_sha256",
    "read_f32le_21x32",
    "read_f32le_p32",
    "validate_cli",
]


if __name__ == "__main__":
    raise SystemExit(main())

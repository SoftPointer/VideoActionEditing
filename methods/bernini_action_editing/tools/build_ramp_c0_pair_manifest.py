#!/usr/bin/env python3
"""Build the two sealed inputs required by the exact-81 RAMP C0 canary.

``manifest`` creates exactly two materializer JSONL rows from one distinct
source/donor identity pair ``A != B`` and two distinct registered temporal
programs.  Both videos are byte-bound and fully decoded before publication;
each must contain exactly 81 RGB frames at 25 fps.  The result is accepted by
``materialize_ramp_motion_analogy_vae.py`` without hand-written shell JSON.

``pair-config`` runs after materialization.  It accepts two hash-bound sealed
sample receipts, verifies their referenced manifests, media, parquet shards,
VAE identity and posterior-role hashes, and publishes the exact closed config
schema consumed by ``train_ramp_c0.py``.

The builder accepts no external regression target or spatial/motion side
channel.  It does not materialize VAE tensors and does not authorize training
or a semantic action-editing claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import materialize_ramp_motion_analogy_vae as materializer  # noqa: E402


METHOD_NAME = "bernini-ramp-c0-pair-manifest-builder-v1"
BUILD_RECEIPT_SCHEMA = "bernini-ramp-c0-pair-manifest-build-receipt-v1"
PAIR_CONFIG_SCHEMA = "bernini-ramp-c0-paired-program-config-v1"
MANIFEST_ROW_SCHEMA = "bernini-ramp-motion-analogy-manifest-row-v1"
MATERIALIZER_RECEIPT_SCHEMA = "bernini-ramp-motion-analogy-vae-receipt-v1"
FRAME_COUNT = 81
FPS = 25.0
LATENT_FRAME_COUNT = 21
DEFAULT_ROW_A_ID = "ramp-c0-arm-a"
DEFAULT_ROW_B_ID = "ramp-c0-arm-b"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class RAMPC0PairBuilderError(RuntimeError):
    """Raised before an ambiguous pair artifact can be published."""


@dataclass(frozen=True)
class ProgramSpec:
    kind: str
    parameter: float
    parameter_hex: str
    spec_digest: str


@dataclass(frozen=True)
class MaterializedReceiptBinding:
    receipt_path: Path
    receipt_sha256: str
    receipt_digest: str
    row_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest_row_digest: str
    source_path: Path
    source_sha256: str
    donor_path: Path
    donor_sha256: str
    program_kind: str
    program_parameter_hex: str
    program_digest: str
    bucket_hw: tuple[int, int]
    instruction_sha256: str
    vae_identity_digest: str
    posterior_blob_sha256: Mapping[str, str]
    parquet_path: Path
    parquet_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return materializer.canonical_json_bytes(value)
    except Exception as error:
        raise RAMPC0PairBuilderError(f"value is not canonical finite JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise RAMPC0PairBuilderError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_plain_file(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute() or requested.is_symlink():
        raise RAMPC0PairBuilderError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except OSError as error:
        raise RAMPC0PairBuilderError(f"cannot resolve {label}: {requested}: {error}") from error
    if resolved != requested or resolved.is_symlink() or not stat.S_ISREG(mode):
        raise RAMPC0PairBuilderError(f"{label} must be a canonical plain file")
    return resolved


def _fresh_output(value: str | Path, *, label: str) -> Path:
    requested = Path(value).expanduser()
    if (
        not requested.is_absolute()
        or requested == Path("/")
        or _SAFE_NAME.fullmatch(requested.name) is None
    ):
        raise RAMPC0PairBuilderError(f"{label} must be an absolute safe non-root path")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as error:
        raise RAMPC0PairBuilderError(f"{label} parent is unavailable") from error
    if parent.is_symlink() or not parent.is_dir() or parent / requested.name != requested:
        raise RAMPC0PairBuilderError(f"{label} path must be canonical")
    if requested.exists() or requested.is_symlink():
        raise RAMPC0PairBuilderError(f"refusing to overwrite {label}")
    return requested


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RAMPC0PairBuilderError(f"{label} contains non-finite constant {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RAMPC0PairBuilderError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RAMPC0PairBuilderError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise RAMPC0PairBuilderError(f"{label} root must be one object")
    return value


def _atomic_create(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RAMPC0PairBuilderError(f"refusing to overwrite output: {path}")
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and (temporary.exists() or temporary.is_symlink()):
            temporary.unlink()


def _remove_own_partial(path: Path) -> None:
    """Remove only a regular output just created by this invocation."""

    if not path.exists() or path.is_symlink():
        return
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise RAMPC0PairBuilderError("refusing to remove a non-regular partial output")
    os.chmod(path, 0o600)
    path.unlink()


def _program_spec(kind: Any, parameter: Any) -> ProgramSpec:
    try:
        checked_kind, checked_parameter, parameter_hex = materializer._validate_program(
            kind, parameter
        )
    except Exception as error:
        raise RAMPC0PairBuilderError(str(error)) from error
    value = {
        "kind": checked_kind,
        "parameter_hex": parameter_hex,
        "frame_count": FRAME_COUNT,
        "construction": "registered_RGB_output_to_input_temporal_program",
    }
    return ProgramSpec(
        checked_kind,
        checked_parameter,
        parameter_hex,
        object_sha256(value),
    )


def _probe_exact_video(path: Path) -> dict[str, Any]:
    """Decode all frames through the same backend used by the materializer."""

    try:
        frames, fps, hw = materializer.base._decode_exact_video(path)
    except Exception as error:
        raise RAMPC0PairBuilderError(str(error)) from error
    shape = tuple(int(item) for item in getattr(frames, "shape", ()))
    if (
        shape != (FRAME_COUNT, int(hw[0]), int(hw[1]), 3)
        or str(getattr(frames, "dtype", "")) != "uint8"
        or not math.isclose(float(fps), FPS, rel_tol=0.0, abs_tol=1.0e-3)
    ):
        raise RAMPC0PairBuilderError("decoded exact81/25fps RGB contract differs")
    return {
        "probe_backend": "decord_decode_all_integer_frames",
        "frame_count": FRAME_COUNT,
        "fps": float(fps),
        "height": int(hw[0]),
        "width": int(hw[1]),
        "decoded_dtype": "uint8",
        "decoded_channels": 3,
        "all_frames_0_through_80_decoded": True,
    }


def _audit_video(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, label=f"expected {label} SHA-256")
    before = path.stat()
    snapshot = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    actual_before = file_sha256(path)
    if actual_before != expected:
        raise RAMPC0PairBuilderError(
            f"{label} SHA-256 differs: expected={expected} actual={actual_before}"
        )
    media = _probe_exact_video(path)
    after = path.stat()
    if snapshot != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RAMPC0PairBuilderError(f"{label} changed while probing")
    if file_sha256(path) != expected:
        raise RAMPC0PairBuilderError(f"{label} changed after probing")
    return {
        "path": str(path),
        "sha256": expected,
        "byte_count": before.st_size,
        "media": media,
        "snapshot_stable_during_probe": True,
    }


def _manifest_row(
    *,
    row_id: str,
    source: Path,
    source_sha256: str,
    donor: Path,
    donor_sha256: str,
    program: ProgramSpec,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": MANIFEST_ROW_SCHEMA,
        "row_id": row_id,
        "source_video_path": str(source),
        "source_video_sha256": source_sha256,
        "donor_video_path": str(donor),
        "donor_video_sha256": donor_sha256,
        "program_kind": program.kind,
        "program_parameter": program.parameter,
        "program_parameter_hex": program.parameter_hex,
    }
    row["manifest_row_digest"] = object_sha256(row)
    try:
        checked = materializer.validate_manifest_row(row)
    except Exception as error:
        raise RAMPC0PairBuilderError(str(error)) from error
    if checked.manifest_row_digest != row["manifest_row_digest"]:
        raise RAMPC0PairBuilderError("materializer recomputed a different row digest")
    return row


def build_manifest_bundle(
    *,
    source_a_video: str | Path,
    expected_source_a_sha256: str,
    donor_b_video: str | Path,
    expected_donor_b_sha256: str,
    program_a_kind: str,
    program_a_parameter: float,
    program_b_kind: str,
    program_b_parameter: float,
    output_manifest: str | Path,
    output_build_receipt: Optional[str | Path] = None,
    row_a_id: str = DEFAULT_ROW_A_ID,
    row_b_id: str = DEFAULT_ROW_B_ID,
) -> dict[str, Any]:
    """Publish exactly two materializer rows and a sealed builder receipt."""

    manifest_path = _fresh_output(output_manifest, label="output manifest")
    receipt_value = (
        Path(str(manifest_path) + ".receipt.json")
        if output_build_receipt is None
        else Path(output_build_receipt)
    )
    receipt_path = _fresh_output(receipt_value, label="output build receipt")
    if manifest_path == receipt_path:
        raise RAMPC0PairBuilderError("manifest and build receipt paths must differ")

    source = _canonical_plain_file(source_a_video, label="source A video")
    donor = _canonical_plain_file(donor_b_video, label="donor B video")
    if source == donor:
        raise RAMPC0PairBuilderError("source A and donor B paths must differ")
    source_sha = _require_sha256(
        expected_source_a_sha256, label="expected source A SHA-256"
    )
    donor_sha = _require_sha256(
        expected_donor_b_sha256, label="expected donor B SHA-256"
    )
    if source_sha == donor_sha:
        raise RAMPC0PairBuilderError("source A and donor B SHA-256 identities must differ")
    source_audit = _audit_video(source, source_sha, label="source A video")
    donor_audit = _audit_video(donor, donor_sha, label="donor B video")

    program_a = _program_spec(program_a_kind, program_a_parameter)
    program_b = _program_spec(program_b_kind, program_b_parameter)
    if (
        program_a.kind == program_b.kind
        and program_a.parameter_hex == program_b.parameter_hex
    ):
        raise RAMPC0PairBuilderError("RAMP C0 arms require two distinct temporal programs")
    if row_a_id == row_b_id:
        raise RAMPC0PairBuilderError("RAMP C0 arm row IDs must differ")
    rows = [
        _manifest_row(
            row_id=row_a_id,
            source=source,
            source_sha256=source_sha,
            donor=donor,
            donor_sha256=donor_sha,
            program=program_a,
        ),
        _manifest_row(
            row_id=row_b_id,
            source=source,
            source_sha256=source_sha,
            donor=donor,
            donor_sha256=donor_sha,
            program=program_b,
        ),
    ]
    manifest_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    build_receipt: dict[str, Any] = {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "complete": True,
        "builder_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "row_count": 2,
            "row_order": [row_a_id, row_b_id],
            "row_digests": [row["manifest_row_digest"] for row in rows],
            "materializer_row_schema": MANIFEST_ROW_SCHEMA,
        },
        "input": {
            "source_A": source_audit,
            "donor_B": donor_audit,
            "source_and_donor_paths_distinct": True,
            "source_and_donor_sha256_distinct": True,
        },
        "programs": {
            "arm_a": {
                "row_id": row_a_id,
                "kind": program_a.kind,
                "parameter_hex": program_a.parameter_hex,
                "spec_digest": program_a.spec_digest,
            },
            "arm_b": {
                "row_id": row_b_id,
                "kind": program_b.kind,
                "parameter_hex": program_b.parameter_hex,
                "spec_digest": program_b.spec_digest,
            },
            "distinct": True,
        },
        "media_contract": {
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "full_decode_before_publication": True,
            "same_A_and_B_for_both_program_arms": True,
        },
        "accepted_external_inputs": [
            "source_A_video",
            "donor_B_video",
            "two_registered_temporal_programs",
        ],
        "external_regression_target_accepted": False,
        "spatial_or_motion_side_channel_accepted": False,
        "vae_materialization_performed": False,
        "training_authorized": False,
        "semantic_action_claim_authorized": False,
    }
    build_receipt["receipt_digest"] = object_sha256(build_receipt)
    try:
        _atomic_create(manifest_path, manifest_payload)
        _atomic_create(
            receipt_path, canonical_json_bytes(build_receipt) + b"\n"
        )
    except Exception:
        _remove_own_partial(manifest_path)
        raise
    try:
        if file_sha256(manifest_path) != manifest_sha:
            raise RAMPC0PairBuilderError("published manifest SHA-256 differs")
        loaded = materializer.load_manifest(
            manifest_path, expected_sha256=manifest_sha
        )
        if [row.row_id for row in loaded.rows] != [row_a_id, row_b_id]:
            raise RAMPC0PairBuilderError("published manifest row order differs")
    except Exception as error:
        _remove_own_partial(receipt_path)
        _remove_own_partial(manifest_path)
        if isinstance(error, RAMPC0PairBuilderError):
            raise
        raise RAMPC0PairBuilderError(str(error)) from error
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "build_receipt_path": str(receipt_path),
        "build_receipt_sha256": file_sha256(receipt_path),
        "build_receipt_digest": build_receipt["receipt_digest"],
        "row_ids": [row_a_id, row_b_id],
    }


_RECEIPT_FALSE_FIELDS = (
    "external_target_accepted",
    "paired_action_dataset_used",
    "mask_flow_pose_track_box_trajectory_used",
    "direct_21_phase_permutation_authorized",
    "posterior_sample_materialized",
    "downstream_independent_posterior_sampling_authorized",
    "training_authorized",
    "action_training_authorized",
    "natural_semantic_action_learned",
    "scientific_claim_authorized",
)


def _load_materializer_receipt(
    path_value: str | Path,
    expected_file_sha256: str,
    *,
    label: str,
) -> MaterializedReceiptBinding:
    path = _canonical_plain_file(path_value, label=f"{label} receipt")
    expected_file_sha = _require_sha256(
        expected_file_sha256, label=f"expected {label} receipt file SHA-256"
    )
    if file_sha256(path) != expected_file_sha:
        raise RAMPC0PairBuilderError(f"{label} receipt file SHA-256 differs")
    receipt = _strict_json(path, label=f"{label} materializer receipt")
    declared = _require_sha256(
        receipt.get("receipt_digest"), label=f"{label} embedded receipt digest"
    )
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest", None)
    if object_sha256(unsigned) != declared:
        raise RAMPC0PairBuilderError(f"{label} embedded receipt digest differs")
    if (
        receipt.get("schema_version") != MATERIALIZER_RECEIPT_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("construction")
        != "source=A,donor_packet=(B,T(B)),target=T(A)"
        or receipt.get("frame_count") != FRAME_COUNT
        or receipt.get("fps") != FPS
        or receipt.get("latent_frame_count") != LATENT_FRAME_COUNT
        or receipt.get("create_only") is not True
        or receipt.get("four_independent_VAE_encode_calls") is not True
        or receipt.get("training_use_forbidden") is not True
        or receipt.get("target_origin")
        != "deterministic_RGB_transform_of_source_inside_committed_builder"
        or receipt.get("shared_i0_used") is not False
        or any(receipt.get(name) is not False for name in _RECEIPT_FALSE_FIELDS)
    ):
        raise RAMPC0PairBuilderError(f"{label} materializer receipt closure differs")

    manifest_info = receipt.get("manifest")
    input_info = receipt.get("input")
    program = receipt.get("program")
    motion_receipt = receipt.get("motion_analogy_builder_receipt")
    vae_identity = receipt.get("vae_identity")
    posterior_hashes = receipt.get("vae_posterior_blob_sha256")
    if not all(
        isinstance(value, Mapping)
        for value in (
            manifest_info,
            input_info,
            program,
            motion_receipt,
            vae_identity,
            posterior_hashes,
        )
    ):
        raise RAMPC0PairBuilderError(f"{label} receipt subcontracts are absent")

    manifest_path = _canonical_plain_file(
        manifest_info.get("path", ""), label=f"{label} source manifest"
    )
    manifest_sha = _require_sha256(
        manifest_info.get("sha256"), label=f"{label} source manifest SHA-256"
    )
    try:
        loaded_manifest = materializer.load_manifest(
            manifest_path, expected_sha256=manifest_sha
        )
    except Exception as error:
        raise RAMPC0PairBuilderError(str(error)) from error
    row_id = receipt.get("row_id")
    row_digest = _require_sha256(
        manifest_info.get("row_digest"), label=f"{label} manifest row digest"
    )
    matched_rows = [
        row for row in loaded_manifest.rows
        if row.row_id == row_id and row.manifest_row_digest == row_digest
    ]
    if len(matched_rows) != 1:
        raise RAMPC0PairBuilderError(f"{label} receipt is not bound to one manifest row")
    manifest_row = matched_rows[0]

    source_info = input_info.get("source_A")
    donor_info = input_info.get("donor_B")
    if not isinstance(source_info, Mapping) or not isinstance(donor_info, Mapping):
        raise RAMPC0PairBuilderError(f"{label} source/donor receipt inputs are absent")
    source_path = _canonical_plain_file(
        source_info.get("path", ""), label=f"{label} source A"
    )
    donor_path = _canonical_plain_file(
        donor_info.get("path", ""), label=f"{label} donor B"
    )
    source_sha = _require_sha256(
        source_info.get("sha256"), label=f"{label} source A SHA-256"
    )
    donor_sha = _require_sha256(
        donor_info.get("sha256"), label=f"{label} donor B SHA-256"
    )
    if (
        source_path == donor_path
        or source_sha == donor_sha
        or file_sha256(source_path) != source_sha
        or file_sha256(donor_path) != donor_sha
        or input_info.get("source_and_donor_paths_distinct") is not True
        or input_info.get("source_and_donor_sha256_distinct") is not True
        or input_info.get("external_target") is not None
        or manifest_row.source_path != source_path
        or manifest_row.donor_path != donor_path
        or manifest_row.source_sha256 != source_sha
        or manifest_row.donor_sha256 != donor_sha
    ):
        raise RAMPC0PairBuilderError(f"{label} source/donor binding differs")

    program_kind = program.get("kind")
    program_parameter_hex = program.get("parameter_hex")
    if type(program_parameter_hex) is not str:
        raise RAMPC0PairBuilderError(f"{label} program parameter hex is absent")
    try:
        parameter = float.fromhex(program_parameter_hex)
    except ValueError as error:
        raise RAMPC0PairBuilderError(f"{label} program parameter hex is invalid") from error
    checked_program = _program_spec(program_kind, parameter)
    program_digest = _require_sha256(
        program.get("digest"), label=f"{label} program digest"
    )
    if (
        checked_program.parameter_hex != program_parameter_hex
        or manifest_row.program_kind != checked_program.kind
        or manifest_row.program_parameter_hex != program_parameter_hex
    ):
        raise RAMPC0PairBuilderError(f"{label} program differs from manifest row")

    motion_declared = _require_sha256(
        motion_receipt.get("receipt_digest"),
        label=f"{label} motion-builder receipt digest",
    )
    motion_unsigned = dict(motion_receipt)
    motion_unsigned.pop("receipt_digest", None)
    motion_program = motion_receipt.get("program")
    if not isinstance(motion_program, Mapping):
        raise RAMPC0PairBuilderError(f"{label} motion-builder program is absent")
    instruction_sha = _require_sha256(
        motion_receipt.get("instruction_sha256"),
        label=f"{label} generic instruction SHA-256",
    )
    if (
        object_sha256(motion_unsigned) != motion_declared
        or motion_receipt.get("schema_version")
        != "bernini-mdr-exact-motion-analogy-v1"
        or motion_receipt.get("construction")
        != "source=A,donor_packet=(B,T(B)),target=T(A)"
        or motion_receipt.get("source_identity_sha256") != source_sha
        or motion_receipt.get("donor_identity_sha256") != donor_sha
        or motion_receipt.get("source_and_donor_identity_distinct") is not True
        or motion_receipt.get("frame_count") != FRAME_COUNT
        or motion_receipt.get("latent_frame_count_after_pinned_Wan_VAE")
        != LATENT_FRAME_COUNT
        or motion_receipt.get("program_digest") != program_digest
        or object_sha256(motion_program) != program_digest
        or motion_program.get("kind") != checked_program.kind
        or motion_program.get("parameter_hex") != program_parameter_hex
        or motion_program.get("frame_count") != FRAME_COUNT
        or motion_program.get("vae_phase_permutation_authorized") is not False
        or motion_receipt.get("instruction_is_generic_donor_follow") is not True
        or motion_receipt.get("external_target_accepted") is not False
        or motion_receipt.get("mask_flow_pose_track_trajectory_used") is not False
        or motion_receipt.get("direct_21_phase_permutation_authorized") is not False
    ):
        raise RAMPC0PairBuilderError(f"{label} motion-builder receipt differs")

    vae_digest = _require_sha256(
        vae_identity.get("vae_identity_digest"), label=f"{label} VAE identity digest"
    )
    vae_unsigned = dict(vae_identity)
    vae_unsigned.pop("vae_identity_digest", None)
    if (
        object_sha256(vae_unsigned) != vae_digest
        or vae_identity.get("every_vae_file_sha256_verified") is not True
        or vae_identity.get("posterior_sample_materialized") is not False
    ):
        raise RAMPC0PairBuilderError(f"{label} VAE identity differs")

    required_roles = tuple(materializer.ROLE_TO_BLOB_FIELD)
    hashes: dict[str, str] = {}
    for role in required_roles:
        hashes[role] = _require_sha256(
            posterior_hashes.get(role), label=f"{label} {role} posterior SHA-256"
        )
    if set(posterior_hashes) != set(required_roles):
        raise RAMPC0PairBuilderError(f"{label} posterior role closure differs")

    bucket = receipt.get("source_derived_bucket_hw")
    if (
        not isinstance(bucket, list)
        or len(bucket) != 2
        or any(type(item) is not int or item <= 0 or item % 16 for item in bucket)
    ):
        raise RAMPC0PairBuilderError(f"{label} source-derived bucket differs")
    parquet_path = _canonical_plain_file(
        receipt.get("parquet_path", ""), label=f"{label} parquet"
    )
    parquet_sha = _require_sha256(
        receipt.get("parquet_sha256"), label=f"{label} parquet SHA-256"
    )
    if file_sha256(parquet_path) != parquet_sha:
        raise RAMPC0PairBuilderError(f"{label} parquet SHA-256 differs")

    return MaterializedReceiptBinding(
        receipt_path=path,
        receipt_sha256=expected_file_sha,
        receipt_digest=declared,
        row_id=str(row_id),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest_row_digest=row_digest,
        source_path=source_path,
        source_sha256=source_sha,
        donor_path=donor_path,
        donor_sha256=donor_sha,
        program_kind=checked_program.kind,
        program_parameter_hex=program_parameter_hex,
        program_digest=program_digest,
        bucket_hw=(int(bucket[0]), int(bucket[1])),
        instruction_sha256=instruction_sha,
        vae_identity_digest=vae_digest,
        posterior_blob_sha256=hashes,
        parquet_path=parquet_path,
        parquet_sha256=parquet_sha,
    )


def _validate_materialized_pair(
    arm_a: MaterializedReceiptBinding,
    arm_b: MaterializedReceiptBinding,
) -> dict[str, Any]:
    if arm_a.receipt_path == arm_b.receipt_path or arm_a.parquet_path == arm_b.parquet_path:
        raise RAMPC0PairBuilderError("materialized arms require distinct receipt/parquet paths")
    held_equal = {
        "manifest_path": arm_a.manifest_path == arm_b.manifest_path,
        "manifest_sha256": arm_a.manifest_sha256 == arm_b.manifest_sha256,
        "source_path": arm_a.source_path == arm_b.source_path,
        "source_sha256": arm_a.source_sha256 == arm_b.source_sha256,
        "donor_path": arm_a.donor_path == arm_b.donor_path,
        "donor_sha256": arm_a.donor_sha256 == arm_b.donor_sha256,
        "bucket_hw": arm_a.bucket_hw == arm_b.bucket_hw,
        "instruction_sha256": arm_a.instruction_sha256 == arm_b.instruction_sha256,
        "vae_identity_digest": arm_a.vae_identity_digest == arm_b.vae_identity_digest,
        "source_posterior": (
            arm_a.posterior_blob_sha256["source_A"]
            == arm_b.posterior_blob_sha256["source_A"]
        ),
        "donor_before_posterior": (
            arm_a.posterior_blob_sha256["donor_before_B"]
            == arm_b.posterior_blob_sha256["donor_before_B"]
        ),
    }
    if not all(held_equal.values()):
        raise RAMPC0PairBuilderError(
            f"materialized arms differ outside temporal program: {held_equal}"
        )
    if (
        arm_a.row_id == arm_b.row_id
        or arm_a.manifest_row_digest == arm_b.manifest_row_digest
        or arm_a.program_digest == arm_b.program_digest
        or (
            arm_a.program_kind == arm_b.program_kind
            and arm_a.program_parameter_hex == arm_b.program_parameter_hex
        )
    ):
        raise RAMPC0PairBuilderError("materialized arms require distinct rows/programs")
    if (
        arm_a.posterior_blob_sha256["donor_after_TB"]
        == arm_b.posterior_blob_sha256["donor_after_TB"]
        or arm_a.posterior_blob_sha256["target_TA"]
        == arm_b.posterior_blob_sha256["target_TA"]
    ):
        raise RAMPC0PairBuilderError(
            "distinct programs produced byte-equal donor-after or target posterior"
        )
    return {
        "held_equal": held_equal,
        "changed_only": "registered_temporal_program",
        "program_a_digest": arm_a.program_digest,
        "program_b_digest": arm_b.program_digest,
        "target_posteriors_distinct": True,
    }


def build_pair_config(
    *,
    arm_a_receipt: str | Path,
    expected_arm_a_receipt_sha256: str,
    arm_b_receipt: str | Path,
    expected_arm_b_receipt_sha256: str,
    output_pair_config: str | Path,
) -> dict[str, Any]:
    """Publish the exact closed trainer config from two sealed receipts."""

    output = _fresh_output(output_pair_config, label="output pair config")
    arm_a = _load_materializer_receipt(
        arm_a_receipt,
        expected_arm_a_receipt_sha256,
        label="arm_a",
    )
    arm_b = _load_materializer_receipt(
        arm_b_receipt,
        expected_arm_b_receipt_sha256,
        label="arm_b",
    )
    pairing = _validate_materialized_pair(arm_a, arm_b)
    config = {
        "schema_version": PAIR_CONFIG_SCHEMA,
        "arm_a": {
            "parquet_path": str(arm_a.parquet_path),
            "parquet_sha256": arm_a.parquet_sha256,
            "receipt_path": str(arm_a.receipt_path),
            "receipt_sha256": arm_a.receipt_sha256,
        },
        "arm_b": {
            "parquet_path": str(arm_b.parquet_path),
            "parquet_sha256": arm_b.parquet_sha256,
            "receipt_path": str(arm_b.receipt_path),
            "receipt_sha256": arm_b.receipt_sha256,
        },
    }
    payload = canonical_json_bytes(config) + b"\n"
    _atomic_create(output, payload)
    config_sha = file_sha256(output)
    return {
        "pair_config_path": str(output),
        "pair_config_sha256": config_sha,
        "pair_config_schema": PAIR_CONFIG_SCHEMA,
        "arm_a_row_id": arm_a.row_id,
        "arm_b_row_id": arm_b.row_id,
        "pairing_audit": pairing,
        "training_authorized_by_builder": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="build the exact two-row materializer JSONL"
    )
    manifest.add_argument("--source-a-video", required=True)
    manifest.add_argument("--expected-source-a-sha256", required=True)
    manifest.add_argument("--donor-b-video", required=True)
    manifest.add_argument("--expected-donor-b-sha256", required=True)
    manifest.add_argument("--program-a-kind", required=True, choices=materializer.PROGRAM_KINDS)
    manifest.add_argument("--program-a-parameter", required=True, type=float)
    manifest.add_argument("--program-b-kind", required=True, choices=materializer.PROGRAM_KINDS)
    manifest.add_argument("--program-b-parameter", required=True, type=float)
    manifest.add_argument("--row-a-id", default=DEFAULT_ROW_A_ID)
    manifest.add_argument("--row-b-id", default=DEFAULT_ROW_B_ID)
    manifest.add_argument("--output-manifest", required=True)
    manifest.add_argument("--output-build-receipt")

    pair = subparsers.add_parser(
        "pair-config", help="build trainer config from sealed materializer receipts"
    )
    pair.add_argument("--arm-a-receipt", required=True)
    pair.add_argument("--expected-arm-a-receipt-sha256", required=True)
    pair.add_argument("--arm-b-receipt", required=True)
    pair.add_argument("--expected-arm-b-receipt-sha256", required=True)
    pair.add_argument("--output-pair-config", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            result = build_manifest_bundle(
                source_a_video=args.source_a_video,
                expected_source_a_sha256=args.expected_source_a_sha256,
                donor_b_video=args.donor_b_video,
                expected_donor_b_sha256=args.expected_donor_b_sha256,
                program_a_kind=args.program_a_kind,
                program_a_parameter=args.program_a_parameter,
                program_b_kind=args.program_b_kind,
                program_b_parameter=args.program_b_parameter,
                row_a_id=args.row_a_id,
                row_b_id=args.row_b_id,
                output_manifest=args.output_manifest,
                output_build_receipt=args.output_build_receipt,
            )
        elif args.command == "pair-config":
            result = build_pair_config(
                arm_a_receipt=args.arm_a_receipt,
                expected_arm_a_receipt_sha256=(
                    args.expected_arm_a_receipt_sha256
                ),
                arm_b_receipt=args.arm_b_receipt,
                expected_arm_b_receipt_sha256=(
                    args.expected_arm_b_receipt_sha256
                ),
                output_pair_config=args.output_pair_config,
            )
        else:  # pragma: no cover - argparse closes this branch
            raise RAMPC0PairBuilderError("unknown builder command")
    except RAMPC0PairBuilderError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    print(canonical_json_bytes(result).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUILD_RECEIPT_SCHEMA",
    "DEFAULT_ROW_A_ID",
    "DEFAULT_ROW_B_ID",
    "FRAME_COUNT",
    "FPS",
    "MANIFEST_ROW_SCHEMA",
    "MATERIALIZER_RECEIPT_SCHEMA",
    "METHOD_NAME",
    "PAIR_CONFIG_SCHEMA",
    "RAMPC0PairBuilderError",
    "build_manifest_bundle",
    "build_pair_config",
    "build_parser",
    "file_sha256",
    "main",
    "object_sha256",
]

#!/usr/bin/env python3
"""Materialize exact-81 RAMP motion-analogy VAE posterior rows.

The only media inputs are two distinct, hash-bound videos ``A`` and ``B``.
For every manifest row this program decodes all 81 RGB frames, derives one
spatial bucket from A, center-crops B to A's aspect ratio, and calls the
committed :mod:`mdr_exact_motion_analogy` tensor core to construct::

    A, (B, T(B)), T(A), with A != B.

All four RGB tensors are encoded independently by the pinned Bernini/Wan VAE.
The stored blobs are the deterministic FP32 posterior *parameters*, not a
posterior sample and not a permutation of the 21 causal VAE phases.

This materializer accepts no target, mask, flow, pose, track, box, trajectory,
edited keyframe, or generated action video.  Its create-only outputs remain
pretext artifacts and do not authorize action training or a scientific claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import materialize_vae as base  # noqa: E402


METHOD_NAME = "bernini-ramp-exact81-motion-analogy-vae-materializer-v1"
MANIFEST_ROW_FORMAT = "bernini-ramp-motion-analogy-manifest-row-v1"
MATERIALIZED_ROW_FORMAT = "bernini-ramp-motion-analogy-vae-row-v1"
SAMPLE_RECEIPT_FORMAT = "bernini-ramp-motion-analogy-vae-receipt-v1"
RANK_SUMMARY_FORMAT = "bernini-ramp-motion-analogy-vae-rank-summary-v1"
FRAME_COUNT = 81
FPS = 25.0
LATENT_FRAME_COUNT = 21
DEFAULT_MAX_PIXELS = base.DEFAULT_MAX_PIXELS
DEFAULT_STRIDE = base.DEFAULT_STRIDE
EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256 = (
    "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
)
EXPECTED_VAE_CONFIG_SHA256 = (
    "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527"
)
PROGRAM_KINDS = (
    "identity",
    "reverse",
    "speed_up",
    "slow_down",
    "pause_then_catch_up",
    "cyclic_phase",
)
ROLE_TO_BLOB_FIELD = {
    "source_A": "source_a_vae_posterior_blob",
    "donor_before_B": "donor_b_before_vae_posterior_blob",
    "donor_after_TB": "donor_b_after_vae_posterior_blob",
    "target_TA": "target_ta_vae_posterior_blob",
}
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "row_id",
        "source_video_path",
        "source_video_sha256",
        "donor_video_path",
        "donor_video_sha256",
        "program_kind",
        "program_parameter",
    }
)
_OPTIONAL_MANIFEST_FIELDS = frozenset(
    {"program_parameter_hex", "manifest_row_digest"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROW_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MANIFEST_LINE_RE = re.compile(r"([0-9a-f]{64})  (\./[^\n]+)")


class RampVaeMaterializationError(RuntimeError):
    """Raised before ambiguous or mutable RAMP evidence can be emitted."""


@dataclass(frozen=True)
class ManifestRow:
    row_id: str
    source_path: Path
    source_sha256: str
    donor_path: Path
    donor_sha256: str
    program_kind: str
    program_parameter: float
    program_parameter_hex: str
    manifest_row_digest: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedManifest:
    path: Path
    sha256: str
    rows: tuple[ManifestRow, ...]


def canonical_json_bytes(value: Any) -> bytes:
    return base.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return base.object_sha256(value)


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RampVaeMaterializationError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise RampVaeMaterializationError(
            f"{context} must be a regular non-symlink file: {path}"
        )
    return path


def _absolute_plain_file(value: Any, *, context: str) -> Path:
    if type(value) is not str or not value:
        raise RampVaeMaterializationError(f"{context} path must be non-empty text")
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise RampVaeMaterializationError(f"{context} path must be absolute")
    try:
        if requested.is_symlink():
            raise RampVaeMaterializationError(f"{context} path must not be a symlink")
    except OSError as error:
        raise RampVaeMaterializationError(
            f"cannot inspect {context}: {requested}: {error}"
        ) from error
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise RampVaeMaterializationError(
            f"cannot resolve {context}: {requested}: {error}"
        ) from error
    return _plain_file(resolved, context=context)


def _require_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RampVaeMaterializationError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _reject_constant(value: str) -> None:
    raise RampVaeMaterializationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RampVaeMaterializationError(
                f"duplicate JSON object key is forbidden: {key!r}"
            )
        result[key] = value
    return result


def _strict_json_loads(text: str, *, context: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise RampVaeMaterializationError(f"invalid {context}: {error}") from error


def _finite_parameter(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RampVaeMaterializationError(
            "program_parameter must be one finite JSON number"
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise RampVaeMaterializationError(
            "program_parameter must be one finite JSON number"
        ) from error
    if not math.isfinite(result):
        raise RampVaeMaterializationError("program_parameter must be finite")
    return result


def _validate_program(kind: Any, parameter_value: Any) -> tuple[str, float, str]:
    if type(kind) is not str or kind not in PROGRAM_KINDS:
        raise RampVaeMaterializationError(
            f"program_kind must be one of {PROGRAM_KINDS}"
        )
    parameter = _finite_parameter(parameter_value)
    if kind in {"identity", "reverse"} and parameter != 0.0:
        raise RampVaeMaterializationError(f"{kind} requires program_parameter=0")
    if kind == "speed_up" and not 0.2 <= parameter <= 0.8:
        raise RampVaeMaterializationError(
            "speed_up program_parameter must be in [0.2,0.8]"
        )
    if kind == "slow_down" and not 1.25 <= parameter <= 4.0:
        raise RampVaeMaterializationError(
            "slow_down program_parameter must be in [1.25,4]"
        )
    if kind == "pause_then_catch_up" and not 0.1 <= parameter <= 0.5:
        raise RampVaeMaterializationError(
            "pause_then_catch_up program_parameter must be in [0.1,0.5]"
        )
    if kind == "cyclic_phase":
        phase = int(parameter)
        if float(phase) != parameter or not 1 <= phase < FRAME_COUNT:
            raise RampVaeMaterializationError(
                "cyclic_phase program_parameter must be an integer in [1,80]"
            )
    return kind, parameter, parameter.hex()


def _verify_bound_file(path: Path, expected_sha256: str, *, context: str) -> None:
    _plain_file(path, context=context)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise RampVaeMaterializationError(
            f"{context} SHA-256 differs: expected={expected_sha256} actual={actual}"
        )


def validate_manifest_row(row: Any, *, line_number: int = 0) -> ManifestRow:
    """Validate one closed-surface manifest row and all media bindings."""

    context = f"manifest line {line_number}" if line_number else "manifest row"
    if not isinstance(row, dict):
        raise RampVaeMaterializationError(f"{context} must be one JSON object")
    fields = frozenset(row)
    missing = _REQUIRED_MANIFEST_FIELDS - fields
    extra = fields - _REQUIRED_MANIFEST_FIELDS - _OPTIONAL_MANIFEST_FIELDS
    if missing:
        raise RampVaeMaterializationError(
            f"{context} is missing fields: {sorted(missing)}"
        )
    if extra:
        raise RampVaeMaterializationError(
            f"{context} has forbidden/unrecognized fields: {sorted(extra)}"
        )
    if row.get("schema_version") != MANIFEST_ROW_FORMAT:
        raise RampVaeMaterializationError(f"{context} schema_version differs")
    row_id = row.get("row_id")
    if type(row_id) is not str or _ROW_ID_RE.fullmatch(row_id) is None:
        raise RampVaeMaterializationError(f"{context} has an unsafe row_id")
    source = _absolute_plain_file(row.get("source_video_path"), context="source A")
    donor = _absolute_plain_file(row.get("donor_video_path"), context="donor B")
    if source == donor:
        raise RampVaeMaterializationError("source A and donor B paths must differ")
    source_sha = _require_sha256(
        row.get("source_video_sha256"), context="source A SHA-256"
    )
    donor_sha = _require_sha256(
        row.get("donor_video_sha256"), context="donor B SHA-256"
    )
    if source_sha == donor_sha:
        raise RampVaeMaterializationError(
            "source A and donor B SHA-256 identities must differ"
        )
    _verify_bound_file(source, source_sha, context="source A")
    _verify_bound_file(donor, donor_sha, context="donor B")
    kind, parameter, parameter_hex = _validate_program(
        row.get("program_kind"), row.get("program_parameter")
    )
    declared_hex = row.get("program_parameter_hex")
    if declared_hex is not None and declared_hex != parameter_hex:
        raise RampVaeMaterializationError(
            "program_parameter_hex differs from the exact JSON numeric value"
        )
    unsigned = dict(row)
    declared_digest = unsigned.pop("manifest_row_digest", None)
    computed_digest = object_sha256(unsigned)
    if declared_digest is not None:
        _require_sha256(declared_digest, context="manifest row digest")
        if declared_digest != computed_digest:
            raise RampVaeMaterializationError("manifest row digest differs")
    return ManifestRow(
        row_id=row_id,
        source_path=source,
        source_sha256=source_sha,
        donor_path=donor,
        donor_sha256=donor_sha,
        program_kind=kind,
        program_parameter=parameter,
        program_parameter_hex=parameter_hex,
        manifest_row_digest=computed_digest,
        raw=dict(row),
    )


def load_manifest(
    path: Path, *, expected_sha256: Optional[str] = None
) -> LoadedManifest:
    """Load a plain JSONL manifest without accepting duplicate rows or keys."""

    requested = path.expanduser()
    if not requested.is_absolute():
        raise RampVaeMaterializationError("manifest path must be absolute")
    manifest_path = _plain_file(
        requested.resolve(strict=True), context="motion-analogy manifest"
    )
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, context="expected manifest SHA-256")
        if manifest_sha != expected:
            raise RampVaeMaterializationError(
                f"manifest SHA-256 differs: expected={expected} actual={manifest_sha}"
            )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RampVaeMaterializationError("manifest is not valid UTF-8") from error
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        raw_row = _strict_json_loads(line, context=f"manifest line {line_number}")
        row = validate_manifest_row(raw_row, line_number=line_number)
        if row.row_id in seen:
            raise RampVaeMaterializationError(
                f"duplicate manifest row_id: {row.row_id}"
            )
        seen.add(row.row_id)
        rows.append(row)
    if not rows:
        raise RampVaeMaterializationError("motion-analogy manifest is empty")
    return LoadedManifest(path=manifest_path, sha256=manifest_sha, rows=tuple(rows))


def validate_pinned_vae_checkpoint(
    checkpoint: Path,
    content_manifest: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    expected_vae_config_sha256: str = EXPECTED_VAE_CONFIG_SHA256,
) -> dict[str, Any]:
    """Verify every VAE file against a sealed checkpoint content manifest."""

    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256, context="checkpoint content manifest SHA-256"
    )
    expected_vae_config_sha256 = _require_sha256(
        expected_vae_config_sha256, context="VAE config SHA-256"
    )
    root = checkpoint.expanduser()
    if not root.is_absolute():
        raise RampVaeMaterializationError("checkpoint path must be absolute")
    if root.is_symlink():
        raise RampVaeMaterializationError("checkpoint path must not be a symlink")
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise RampVaeMaterializationError(f"checkpoint is unavailable: {error}") from error
    if not root.is_dir() or root.is_symlink():
        raise RampVaeMaterializationError(
            "checkpoint must be a non-symlink directory"
        )
    manifest = _absolute_plain_file(
        str(content_manifest.expanduser()), context="checkpoint content manifest"
    )
    if file_sha256(manifest) != expected_manifest_sha256:
        raise RampVaeMaterializationError(
            "checkpoint content manifest SHA-256 differs"
        )
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RampVaeMaterializationError(
            "cannot read checkpoint content manifest"
        ) from error
    for line in lines:
        match = _MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise RampVaeMaterializationError(
                "checkpoint manifest is not canonical sha256sum syntax"
            )
        digest, raw_path = match.groups()
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RampVaeMaterializationError(
                "checkpoint manifest contains an unsafe path"
            )
        normalized = PurePosixPath(
            *(part for part in relative.parts if part not in ("", "."))
        ).as_posix()
        if not normalized or normalized in entries:
            raise RampVaeMaterializationError(
                "checkpoint manifest contains an empty/duplicate path"
            )
        entries[normalized] = digest
    vae_entries = {
        relative: digest
        for relative, digest in entries.items()
        if relative.startswith("vae/")
    }
    if vae_entries.get("vae/config.json") != expected_vae_config_sha256:
        raise RampVaeMaterializationError(
            "checkpoint manifest does not bind the pinned Wan VAE config"
        )
    if len(vae_entries) < 2 or not any(
        name.endswith((".safetensors", ".bin")) for name in vae_entries
    ):
        raise RampVaeMaterializationError(
            "checkpoint manifest does not bind Wan VAE weights"
        )
    vae_root = root / "vae"
    if not vae_root.is_dir() or vae_root.is_symlink():
        raise RampVaeMaterializationError("checkpoint VAE directory differs")
    actual: set[str] = set()
    for candidate in vae_root.rglob("*"):
        relative = candidate.relative_to(root)
        if ".cache" in relative.parts:
            continue
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RampVaeMaterializationError("checkpoint VAE contains a symlink")
        if stat.S_ISREG(mode):
            actual.add(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise RampVaeMaterializationError(
                "checkpoint VAE contains a non-regular entry"
            )
    if actual != set(vae_entries):
        raise RampVaeMaterializationError(
            "checkpoint VAE file set differs from the pinned manifest"
        )
    for relative, digest in sorted(vae_entries.items()):
        path = _plain_file(root / relative, context=f"checkpoint {relative}")
        if file_sha256(path) != digest:
            raise RampVaeMaterializationError(
                f"checkpoint VAE file hash differs: {relative}"
            )
    identity = {
        "checkpoint_root": str(root),
        "checkpoint_content_manifest_path": str(manifest),
        "checkpoint_content_manifest_sha256": expected_manifest_sha256,
        "vae_config_sha256": expected_vae_config_sha256,
        "vae_files": dict(sorted(vae_entries.items())),
        "every_vae_file_sha256_verified": True,
        "posterior_representation": "latent_dist.parameters_fp32",
        "posterior_sample_materialized": False,
    }
    identity["vae_identity_digest"] = object_sha256(identity)
    return identity


class PinnedBerniniWanPosteriorEncoder(base.BerniniVaeEncoder):
    """Bernini encoder with a fully hash-verified VAE subtree."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        content_manifest: Path,
        device: str,
        expected_manifest_sha256: str = EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
        expected_vae_config_sha256: str = EXPECTED_VAE_CONFIG_SHA256,
    ) -> None:
        identity = validate_pinned_vae_checkpoint(
            checkpoint,
            content_manifest,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_vae_config_sha256=expected_vae_config_sha256,
        )
        super().__init__(checkpoint, device=device)
        if self.identity.get("vae_config_sha256") != expected_vae_config_sha256:
            raise RampVaeMaterializationError("loaded VAE config identity differs")
        self.identity = identity


def _load_motion_analogy_core() -> Any:
    try:
        import mdr_exact_motion_analogy as core
    except ImportError as error:
        raise RampVaeMaterializationError(
            "RGB motion-analogy construction requires torch and the committed core"
        ) from error
    if tuple(core.PROGRAM_KINDS) != PROGRAM_KINDS or core.FRAME_COUNT != FRAME_COUNT:
        raise RampVaeMaterializationError(
            "committed motion-analogy core contract differs"
        )
    return core


def prepare_motion_analogy_rgb(
    row: ManifestRow,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    stride: int = DEFAULT_STRIDE,
) -> tuple[Any, dict[str, Any]]:
    """Decode A/B, align B to A, and construct A,(B,T(B)),T(A) in RGB."""

    if not isinstance(row, ManifestRow):
        raise RampVaeMaterializationError("row must be a validated ManifestRow")
    _verify_bound_file(row.source_path, row.source_sha256, context="source A")
    _verify_bound_file(row.donor_path, row.donor_sha256, context="donor B")
    source_frames, source_fps, source_hw = base._decode_exact_video(row.source_path)
    donor_frames, donor_fps, donor_hw = base._decode_exact_video(row.donor_path)
    try:
        bucket = base.source_aspect_bucket(
            *source_hw, max_pixels=max_pixels, stride=stride
        )
        donor_crop, donor_retention = base.target_crop_to_source_aspect(
            *donor_hw, *source_hw
        )
        # Bicubic kernels may overshoot the input range by a few ULPs.  The
        # committed RGB builder requires [-1,1], so clamp both roles under one
        # recorded deterministic policy before T is applied.
        source = (
            base._resize_video(source_frames, bucket, None)
            .clamp_(-1.0, 1.0)
            .unsqueeze(0)
        )
        donor = (
            base._resize_video(donor_frames, bucket, donor_crop)
            .clamp_(-1.0, 1.0)
            .unsqueeze(0)
        )
    except base.VaeMaterializationError as error:
        raise RampVaeMaterializationError(str(error)) from error
    _verify_bound_file(row.source_path, row.source_sha256, context="source A after decode")
    _verify_bound_file(row.donor_path, row.donor_sha256, context="donor B after decode")
    if source.equal(donor):
        raise RampVaeMaterializationError(
            "source A and donor B become identical after deterministic alignment"
        )
    core = _load_motion_analogy_core()
    program = core.TemporalProgram(row.program_kind, row.program_parameter)
    example = core.build_motion_analogy_example(
        source,
        donor,
        program,
        source_identity_sha256=row.source_sha256,
        donor_identity_sha256=row.donor_sha256,
    )
    tensors = {
        "source_A": example.source_identity_video,
        "donor_before_B": example.motion_donor_before_video,
        "donor_after_TB": example.motion_donor_after_video,
        "target_TA": example.regression_target_video,
    }
    expected_shape = (1, 3, FRAME_COUNT, bucket[0], bucket[1])
    if any(
        tuple(int(item) for item in value.shape) != expected_shape
        for value in tensors.values()
    ):
        raise RampVaeMaterializationError("constructed RGB tensor geometry differs")
    source_equals_target = bool(
        tensors["source_A"].equal(tensors["target_TA"])
    )
    donor_before_equals_after = bool(
        tensors["donor_before_B"].equal(tensors["donor_after_TB"])
    )
    if row.program_kind == "identity":
        if not source_equals_target or not donor_before_equals_after:
            raise RampVaeMaterializationError(
                "identity program must preserve both RGB videos exactly"
            )
    elif source_equals_target or donor_before_equals_after:
        raise RampVaeMaterializationError(
            "non-identity program is uninformative on source or donor RGB"
        )
    receipt = dict(example.receipt)
    receipt_digest = receipt.pop("receipt_digest", None)
    if object_sha256(receipt) != receipt_digest:
        raise RampVaeMaterializationError(
            "committed motion-analogy builder receipt digest differs"
        )
    media = {
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "temporal_policy": "decode_all_integer_RGB_frames_0_through_80",
        "source_reported_fps": source_fps,
        "donor_reported_fps": donor_fps,
        "source_input_hw": list(source_hw),
        "donor_input_hw": list(donor_hw),
        "source_derived_bucket_hw": list(bucket),
        "bucket_rule": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
        "max_pixels": max_pixels,
        "stride": stride,
        "source_crop": None,
        "donor_center_crop_tlbr": list(donor_crop),
        "donor_crop_retention": donor_retention,
        "resize": "torchvision_bicubic_antialias_true",
        "post_resize_clamp": "clip_to_closed_interval_minus1_plus1",
        "normalization": "uint8_div_255_mul_2_minus_1",
        "shared_i0_used": False,
        "source_first_frame_overwritten": False,
        "donor_first_frame_overwritten": False,
        "source_equals_target": source_equals_target,
        "donor_before_equals_after": donor_before_equals_after,
        "rgb_program_applied_before_each_VAE_encode": True,
        "four_independent_VAE_encodes_required": True,
        "direct_21_phase_permutation_authorized": False,
        "rgb_tensor_sha256": {
            role: base._tensor_sha256(value) for role, value in tensors.items()
        },
        "motion_analogy_builder_receipt": dict(example.receipt),
    }
    media["media_contract_digest"] = object_sha256(media)
    return example, media


def encode_four_posteriors(
    example: Any, *, encoder: Any
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    """Encode the four RGB roles in four distinct VAE calls."""

    ordered = (
        ("source_A", example.source_identity_video),
        ("donor_before_B", example.motion_donor_before_video),
        ("donor_after_TB", example.motion_donor_after_video),
        ("target_TA", example.regression_target_video),
    )
    blobs: dict[str, bytes] = {}
    metadata: dict[str, dict[str, Any]] = {}
    shapes: set[tuple[int, ...]] = set()
    for call_index, (role, video) in enumerate(ordered):
        shape = tuple(int(item) for item in video.shape)
        if len(shape) != 5 or shape[:3] != (1, 3, FRAME_COUNT):
            raise RampVaeMaterializationError(
                f"{role} must be exact [1,3,81,H,W] before VAE encoding"
            )
        blob, details = encoder.encode(video.squeeze(0))
        if not isinstance(blob, bytes) or not blob:
            raise RampVaeMaterializationError(f"{role} VAE blob is absent")
        if not isinstance(details, Mapping):
            raise RampVaeMaterializationError(f"{role} VAE metadata is absent")
        details = dict(details)
        posterior_shape = details.get("posterior_parameters_shape")
        if (
            not isinstance(posterior_shape, list)
            or len(posterior_shape) != 5
            or posterior_shape[0] != 1
            or posterior_shape[2] != LATENT_FRAME_COUNT
        ):
            raise RampVaeMaterializationError(
                f"{role} posterior parameters must have [1,C,21,H,W] geometry"
            )
        shapes.add(tuple(int(item) for item in posterior_shape))
        details.update(
            {
                "encode_call_index": call_index,
                "encoded_independently": True,
                "artifact_role": role,
                "posterior_parameters_blob_sha256": hashlib.sha256(blob).hexdigest(),
                "posterior_sample_materialized": False,
            }
        )
        blobs[role] = blob
        metadata[role] = details
    if len(blobs) != 4 or len(shapes) != 1:
        raise RampVaeMaterializationError(
            "four posterior parameter blobs must share one exact geometry"
        )
    return blobs, metadata


def _materialized_row_digest(row: Mapping[str, Any]) -> str:
    candidate = {
        key: value
        for key, value in row.items()
        if key not in set(ROLE_TO_BLOB_FIELD.values()) | {"materialized_row_digest"}
    }
    candidate["vae_posterior_blob_sha256"] = {
        role: hashlib.sha256(row[field]).hexdigest()
        for role, field in ROLE_TO_BLOB_FIELD.items()
    }
    return object_sha256(candidate)


def materialize_one(
    row: ManifestRow,
    *,
    manifest: LoadedManifest,
    encoder: Any,
    output_root: Path,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    stride: int = DEFAULT_STRIDE,
) -> dict[str, Any]:
    """Create one immutable parquet shard and its sealed receipt."""

    if row not in manifest.rows:
        raise RampVaeMaterializationError("row is not bound to this manifest")
    shard = output_root / "shards" / f"{row.row_id}.parquet"
    receipt_path = output_root / "receipts" / f"{row.row_id}.json"
    if shard.exists() or shard.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise RampVaeMaterializationError(
            f"create-only sample output already exists: {row.row_id}"
        )
    example, media = prepare_motion_analogy_rgb(
        row, max_pixels=max_pixels, stride=stride
    )
    blobs, vae_metadata = encode_four_posteriors(example, encoder=encoder)
    encoder_identity = getattr(encoder, "identity", None)
    if not isinstance(encoder_identity, Mapping):
        raise RampVaeMaterializationError("encoder identity is absent")
    output_row: dict[str, Any] = {
        "schema_version": MATERIALIZED_ROW_FORMAT,
        "row_id": row.row_id,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "manifest_row_digest": row.manifest_row_digest,
        "source_video_path": str(row.source_path),
        "source_video_sha256": row.source_sha256,
        "donor_video_path": str(row.donor_path),
        "donor_video_sha256": row.donor_sha256,
        "program_kind": row.program_kind,
        "program_parameter_hex": row.program_parameter_hex,
        "program_digest": example.program.digest,
        "generic_instruction": example.instruction,
        "motion_analogy_receipt_json": canonical_json_bytes(
            dict(example.receipt)
        ).decode("utf-8").rstrip("\n"),
        "media_contract_json": canonical_json_bytes(media).decode("utf-8").rstrip("\n"),
        "vae_identity_json": canonical_json_bytes(dict(encoder_identity)).decode(
            "utf-8"
        ).rstrip("\n"),
        "vae_metadata_json": canonical_json_bytes(vae_metadata).decode("utf-8").rstrip("\n"),
        "posterior_role_order_json": canonical_json_bytes(
            list(ROLE_TO_BLOB_FIELD)
        ).decode("utf-8").rstrip("\n"),
        "target_origin": "deterministic_RGB_transform_of_source_inside_committed_builder",
        "shared_i0_used": False,
        "external_target_accepted": False,
        "mask_flow_pose_track_box_trajectory_used": False,
        "direct_21_phase_permutation_authorized": False,
        "training_authorized": False,
        "training_use_forbidden": True,
        "action_training_authorized": False,
        "scientific_claim_authorized": False,
    }
    for role, field in ROLE_TO_BLOB_FIELD.items():
        output_row[field] = blobs[role]
    output_row["materialized_row_digest"] = _materialized_row_digest(output_row)
    base._write_sample_parquet(shard, output_row)
    receipt = {
        "schema_version": SAMPLE_RECEIPT_FORMAT,
        "complete": True,
        "row_id": row.row_id,
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "row_digest": row.manifest_row_digest,
        },
        "input": {
            "source_A": {"path": str(row.source_path), "sha256": row.source_sha256},
            "donor_B": {"path": str(row.donor_path), "sha256": row.donor_sha256},
            "source_and_donor_paths_distinct": row.source_path != row.donor_path,
            "source_and_donor_sha256_distinct": row.source_sha256 != row.donor_sha256,
            "external_target": None,
        },
        "program": {
            "kind": row.program_kind,
            "parameter_hex": row.program_parameter_hex,
            "digest": example.program.digest,
        },
        "construction": "source=A,donor_packet=(B,T(B)),target=T(A)",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "latent_frame_count": LATENT_FRAME_COUNT,
        "source_derived_bucket_hw": media["source_derived_bucket_hw"],
        "donor_center_crop_tlbr": media["donor_center_crop_tlbr"],
        "rgb_tensor_sha256": media["rgb_tensor_sha256"],
        "motion_analogy_builder_receipt": dict(example.receipt),
        "four_independent_VAE_encode_calls": True,
        "vae_identity": dict(encoder_identity),
        "vae_posterior_metadata": vae_metadata,
        "vae_posterior_blob_sha256": {
            role: hashlib.sha256(blob).hexdigest() for role, blob in blobs.items()
        },
        "materialized_row_digest": output_row["materialized_row_digest"],
        "parquet_path": str(shard),
        "parquet_sha256": file_sha256(shard),
        "create_only": True,
        "target_origin": "deterministic_RGB_transform_of_source_inside_committed_builder",
        "shared_i0_used": False,
        "external_target_accepted": False,
        "paired_action_dataset_used": False,
        "mask_flow_pose_track_box_trajectory_used": False,
        "direct_21_phase_permutation_authorized": False,
        "posterior_sample_materialized": False,
        "downstream_independent_posterior_sampling_authorized": False,
        "training_authorized": False,
        "training_use_forbidden": True,
        "action_training_authorized": False,
        "natural_semantic_action_learned": False,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    try:
        base._atomic_json(receipt_path, receipt)
    except Exception:
        # The shard was created by this invocation and has no valid receipt;
        # remove only that just-created partial bundle.
        shard.unlink(missing_ok=True)
        raise
    return {
        "row_id": row.row_id,
        "status": "written",
        "parquet_path": str(shard),
        "parquet_sha256": receipt["parquet_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-content-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-content-manifest-sha256",
        default=EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--expected-vae-config-sha256", default=EXPECTED_VAE_CONFIG_SHA256
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _absolute_output_root(path: Path) -> Path:
    requested = path.expanduser()
    if not requested.is_absolute():
        raise RampVaeMaterializationError("output-root must be absolute")
    if requested.exists() and (requested.is_symlink() or not requested.is_dir()):
        raise RampVaeMaterializationError(
            "output-root must be a non-symlink directory"
        )
    requested.mkdir(parents=True, exist_ok=True)
    return requested.resolve(strict=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rank = int(os.environ.get("RANK", "0")) if args.rank is None else args.rank
        world_size = (
            int(os.environ.get("WORLD_SIZE", "1"))
            if args.world_size is None
            else args.world_size
        )
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise RampVaeMaterializationError("invalid rank/world-size")
        if args.max_rows is not None and args.max_rows <= 0:
            raise RampVaeMaterializationError("max-rows must be positive")
        if args.max_pixels < args.stride * args.stride or args.stride <= 0:
            raise RampVaeMaterializationError("invalid max-pixels/stride")
        if (
            args.expected_checkpoint_content_manifest_sha256
            != EXPECTED_CHECKPOINT_CONTENT_MANIFEST_SHA256
            or args.expected_vae_config_sha256 != EXPECTED_VAE_CONFIG_SHA256
        ):
            raise RampVaeMaterializationError(
                "only the pinned Bernini/Wan VAE identity is supported"
            )
        manifest = load_manifest(
            args.manifest, expected_sha256=args.expected_manifest_sha256
        )
        rows = manifest.rows
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        selected = rows[rank::world_size]
        output_root = _absolute_output_root(args.output_root)
        summary_path = output_root / "rank_summaries" / f"rank_{rank:04d}.json"
        if summary_path.exists() or summary_path.is_symlink():
            raise RampVaeMaterializationError(
                f"create-only rank summary already exists: {summary_path}"
            )
        device = args.device or f"cuda:{local_rank}"
        encoder = PinnedBerniniWanPosteriorEncoder(
            args.checkpoint,
            content_manifest=args.checkpoint_content_manifest,
            device=device,
            expected_manifest_sha256=(
                args.expected_checkpoint_content_manifest_sha256
            ),
            expected_vae_config_sha256=args.expected_vae_config_sha256,
        )
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for row in selected:
            try:
                results.append(
                    materialize_one(
                        row,
                        manifest=manifest,
                        encoder=encoder,
                        output_root=output_root,
                        max_pixels=args.max_pixels,
                        stride=args.stride,
                    )
                )
            except Exception as error:
                failure = {
                    "row_id": row.row_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "training_authorized": False,
                    "action_training_authorized": False,
                }
                errors.append(failure)
                rejection = (
                    output_root / "rejections" / f"{row.row_id}.rank{rank}.json"
                )
                base._atomic_json(rejection, failure)
                if not args.continue_on_error:
                    raise
        summary = {
            "schema_version": RANK_SUMMARY_FORMAT,
            "complete": not errors,
            "rank": rank,
            "world_size": world_size,
            "manifest_path": str(manifest.path),
            "manifest_sha256": manifest.sha256,
            "selected_rows": len(selected),
            "completed_rows": len(results),
            "error_rows": len(errors),
            "results": results,
            "errors": errors,
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "latent_frame_count": LATENT_FRAME_COUNT,
            "direct_21_phase_permutation_authorized": False,
            "training_authorized": False,
            "training_use_forbidden": True,
            "action_training_authorized": False,
            "scientific_claim_authorized": False,
        }
        summary["summary_digest"] = object_sha256(summary)
        base._atomic_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 3 if errors else 0
    except RampVaeMaterializationError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

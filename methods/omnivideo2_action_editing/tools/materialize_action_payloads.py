#!/usr/bin/env python3
"""Materialize hash-bound OmniVideo2 action payloads from v17 previews.

This is an exploratory-data tool, not a data-release tool.  Its input rows are
permanently preview-only and the outputs preserve that status.  The expensive
media and encoder stages are dependency-injected so the contract and
publication path can be tested without loading Qwen, UMT5, or the Wan VAE.

The target video has one privileged role: a frozen Qwen teacher describes its
motion in canonical text.  That text is then encoded *without visual input* and
pooled to fixed-length planner labels.  Those labels are stored in the
dedicated ``target_motion_tokens`` payload field; they are never merged into
the source renderer condition by this tool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Optional, Protocol, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = METHOD_ROOT.parents[1]
DEFAULT_OMNI_ROOT = REPOSITORY_ROOT / "methods" / "Omni-Video"

PREVIEW_ROW_FORMAT = "omnivideo2-action-preview-row-v1"
WAN_GENERATED_FORMAT = "motive-wan22-i2v-generated-target-v1"
ACTION_PAYLOAD_FORMAT = "omnivideo2-action-latents-v1"
ACTION_MANIFEST_FORMAT = "omnivideo2-action-manifest-v1"
PROVENANCE_FORMAT = "omnivideo2-action-materialization-provenance-v1"
RECEIPT_FORMAT = "omnivideo2-action-materialization-receipt-v2"

SOURCE_FRAME_COUNT = 81
SOURCE_FPS = 25.0
MIN_CROP_RETENTION = 0.8
TEMPORAL_MODE_FULL_81 = "full_81_25fps"
TEMPORAL_MODE_SMOKE_41 = "smoke_41_12p5fps"
TEMPORAL_MODES = (TEMPORAL_MODE_FULL_81, TEMPORAL_MODE_SMOKE_41)
DEFAULT_TEMPORAL_MODE = TEMPORAL_MODE_FULL_81
FULL_FRAME_INDICES = tuple(range(SOURCE_FRAME_COUNT))
SMOKE_FRAME_INDICES = tuple(range(0, SOURCE_FRAME_COUNT, 2))
# Backward-compatible module constants now describe the full-frame default.
FRAME_INDICES = FULL_FRAME_INDICES
MATERIALIZED_FRAME_COUNT = SOURCE_FRAME_COUNT
MATERIALIZED_FPS = SOURCE_FPS
SPATIAL_PROFILE_FULL_480P = "full_480p"
SPATIAL_PROFILE_MOTION_384P = "motion_384p"
SPATIAL_PROFILES = (SPATIAL_PROFILE_FULL_480P, SPATIAL_PROFILE_MOTION_384P)
DEFAULT_SPATIAL_PROFILE = SPATIAL_PROFILE_FULL_480P
FULL_LANDSCAPE_BUCKET_HW = (480, 832)
FULL_PORTRAIT_BUCKET_HW = (832, 480)
MOTION_LANDSCAPE_BUCKET_HW = (384, 640)
MOTION_PORTRAIT_BUCKET_HW = (640, 384)
VLM_DIM = 2048

PREVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "group_id",
        "family",
        "source_video_path",
        "source_video_sha256",
        "target_video_path",
        "target_video_sha256",
        "edit_instruction",
        "edit_instruction_sha256",
        "instruction_source",
        "generation_instruction",
        "generation_instruction_sha256",
        "source_census",
        "target_plan",
        "selection_gates",
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
        "post_video_acceptance",
        "provenance",
        "row_digest",
    }
)
MATERIALIZED_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "sample_id",
        "payload_path",
        "payload_sha256",
        "provenance_path",
        "provenance_sha256",
        "task_type",
        "preview_only",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


MOTION_VIDEO_SYSTEM_PROMPT = """You are a deterministic video-motion analyst.
Describe only observable temporal changes in the supplied video. Do not
describe identity, clothing, color, texture, lighting, scenery, or visual
style. Do not infer intent. Use exactly three non-empty lines:
SUBJECT_MOTION: <ordered body/object motion from start through end>
CAMERA_MOTION: <locked off, or the observed camera motion>
TIMING: <onset, transitions, and endpoint in temporal order>"""

MOTION_VIDEO_USER_PROMPT = (
    "Return the three-line canonical motion record for this video."
)

MOTION_TEXT_FEATURE_SYSTEM_PROMPT = """You encode canonical motion records for
a video generator. Preserve the subject-motion trajectory, camera motion, and
temporal ordering exactly. The user supplies text only. Do not add appearance
or scene attributes."""


class MaterializationError(RuntimeError):
    """A fail-closed materialization or provenance error."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MaterializationError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, context: str) -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, MaterializationError) as error:
        raise MaterializationError(f"invalid JSON in {context}: {error}") from error


def _closed_mapping(
    value: Any, *, fields: frozenset[str], context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializationError(f"{context} must be an object")
    result = dict(value)
    actual = set(result)
    if actual != fields:
        raise MaterializationError(
            f"{context} fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return result


def _sha(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MaterializationError(f"{context} must be a lowercase SHA-256")
    return value


def _text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise MaterializationError(f"{context} must be non-empty text")
    return value


def _iid(value: Any, *, context: str) -> str:
    if type(value) is not str or _IID_RE.fullmatch(value) is None:
        raise MaterializationError(f"unsafe {context}: {value!r}")
    return value


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise MaterializationError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise MaterializationError(f"{context} is not a plain file: {path}")
    return path


def _verified_file(path_value: Any, digest_value: Any, *, context: str) -> Path:
    path = Path(_text(path_value, context=f"{context} path")).expanduser()
    _plain_file(path, context=context)
    expected = _sha(digest_value, context=f"{context} hash")
    actual = file_sha256(path)
    if actual != expected:
        raise MaterializationError(
            f"{context} hash mismatch: expected={expected} actual={actual}"
        )
    return path.resolve(strict=True)


def _one_jsonl(path: Path, *, context: str) -> dict[str, Any]:
    _plain_file(path, context=context)
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise MaterializationError(f"{context} must end with a newline")
    lines = payload.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise MaterializationError(f"{context} must contain exactly one row")
    value = _parse_json(lines[0], context=context)
    if not isinstance(value, dict):
        raise MaterializationError(f"{context} row must be an object")
    return value


@dataclass(frozen=True)
class TemporalSamplingContract:
    """Auditable temporal sampling selected before media decoding."""

    mode: str
    frame_indices: tuple[int, ...]
    source_frame_count: int
    source_fps: float
    materialized_frame_count: int
    materialized_fps: float
    sampling_policy: str
    temporal_subsampled: bool

    def audit_dict(self) -> dict[str, Any]:
        return {
            "temporal_mode": self.mode,
            "frame_indices": list(self.frame_indices),
            "source_frame_count": self.source_frame_count,
            "source_fps": self.source_fps,
            "materialized_frame_count": self.materialized_frame_count,
            "materialized_fps": self.materialized_fps,
            "sampling_policy": self.sampling_policy,
            "temporal_subsampled": self.temporal_subsampled,
        }


def temporal_sampling_contract(
    mode: str = DEFAULT_TEMPORAL_MODE,
) -> TemporalSamplingContract:
    """Resolve an explicit temporal mode; full 81-frame decoding is the default."""

    if mode == TEMPORAL_MODE_FULL_81:
        contract = TemporalSamplingContract(
            mode=mode,
            frame_indices=FULL_FRAME_INDICES,
            source_frame_count=SOURCE_FRAME_COUNT,
            source_fps=SOURCE_FPS,
            materialized_frame_count=SOURCE_FRAME_COUNT,
            materialized_fps=SOURCE_FPS,
            sampling_policy="all_frames_in_order_no_temporal_subsampling",
            temporal_subsampled=False,
        )
    elif mode == TEMPORAL_MODE_SMOKE_41:
        contract = TemporalSamplingContract(
            mode=mode,
            frame_indices=SMOKE_FRAME_INDICES,
            source_frame_count=SOURCE_FRAME_COUNT,
            source_fps=SOURCE_FPS,
            materialized_frame_count=len(SMOKE_FRAME_INDICES),
            materialized_fps=SOURCE_FPS / 2.0,
            sampling_policy="explicit_stride_2_smoke_ablation_only",
            temporal_subsampled=True,
        )
    else:
        raise MaterializationError(
            f"unknown temporal mode {mode!r}; expected one of {list(TEMPORAL_MODES)}"
        )
    if (
        len(contract.frame_indices) != contract.materialized_frame_count
        or contract.frame_indices[0] != 0
        or contract.frame_indices[-1] != SOURCE_FRAME_COUNT - 1
    ):
        raise AssertionError("internal temporal sampling contract is inconsistent")
    return contract


def temporal_indices_81_to_41() -> tuple[int, ...]:
    """Return indices for the explicit 41-frame smoke/ablation profile."""

    return temporal_sampling_contract(TEMPORAL_MODE_SMOKE_41).frame_indices


@dataclass(frozen=True)
class SpatialProfileContract:
    """Auditable output buckets selected independently from temporal sampling."""

    profile: str
    landscape_bucket_hw: tuple[int, int]
    portrait_bucket_hw: tuple[int, int]

    def audit_dict(self) -> dict[str, Any]:
        return {
            "spatial_profile": self.profile,
            "landscape_bucket_hw": list(self.landscape_bucket_hw),
            "portrait_bucket_hw": list(self.portrait_bucket_hw),
        }


def spatial_profile_contract(
    profile: str = DEFAULT_SPATIAL_PROFILE,
) -> SpatialProfileContract:
    if profile == SPATIAL_PROFILE_FULL_480P:
        return SpatialProfileContract(
            profile=profile,
            landscape_bucket_hw=FULL_LANDSCAPE_BUCKET_HW,
            portrait_bucket_hw=FULL_PORTRAIT_BUCKET_HW,
        )
    if profile == SPATIAL_PROFILE_MOTION_384P:
        return SpatialProfileContract(
            profile=profile,
            landscape_bucket_hw=MOTION_LANDSCAPE_BUCKET_HW,
            portrait_bucket_hw=MOTION_PORTRAIT_BUCKET_HW,
        )
    raise MaterializationError(
        f"unknown spatial profile {profile!r}; expected one of {list(SPATIAL_PROFILES)}"
    )


def choose_bucket(
    height: int,
    width: int,
    *,
    spatial_profile: str = DEFAULT_SPATIAL_PROFILE,
) -> tuple[int, int]:
    if type(height) is not int or type(width) is not int or min(height, width) <= 0:
        raise MaterializationError("video dimensions must be positive integers")
    if height == width:
        raise MaterializationError("square video has no deterministic orientation")
    profile = spatial_profile_contract(spatial_profile)
    return (
        profile.landscape_bucket_hw
        if width > height
        else profile.portrait_bucket_hw
    )


def center_crop_box(
    height: int, width: int, target_hw: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Return the official Omni center-crop box as (top,left,bottom,right)."""

    target_h, target_w = target_hw
    if min(height, width, target_h, target_w) <= 0:
        raise MaterializationError("crop dimensions must be positive")
    ratio = float(target_w) / float(target_h)
    if width < height * ratio:
        crop_h, crop_w = int(float(width) / ratio), width
    else:
        crop_h, crop_w = height, int(float(height) * ratio)
    if min(crop_h, crop_w) <= 0:
        raise MaterializationError("computed center crop is empty")
    top = int(round((height - crop_h) / 2.0))
    left = int(round((width - crop_w) / 2.0))
    return top, left, top + crop_h, left + crop_w


def crop_retention(
    height: int, width: int, target_hw: tuple[int, int]
) -> float:
    """Return the fraction of input pixels retained by the center crop."""

    top, left, bottom, right = center_crop_box(height, width, target_hw)
    return float((bottom - top) * (right - left)) / float(height * width)


@dataclass(frozen=True)
class Frame0Binding:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ValidatedPreviewRow:
    row: dict[str, Any]
    source_video: Path
    target_video: Path
    shared_i0: Frame0Binding
    preview_join_row_digest: str
    preview_join_row_file_sha256: str


@dataclass
class PreparedMedia:
    """Output of the injected media stage.

    ``source_video`` and ``target_video`` are normally CPU tensors with shape
    [3,T,H,W], where T is 81 by default (or 41 only in the explicit smoke
    mode), but are intentionally typed as ``Any`` for lightweight tests.
    """

    source_video: Any
    target_video: Any
    source_qwen_path: str
    target_qwen_path: str
    metadata: dict[str, Any]


@dataclass
class EncodedSample:
    source_latent: Any
    target_latent: Any
    text_context: Any
    source_vlm_context: Any
    target_motion_tokens: Any
    target_caption: str
    motion_text: str
    metadata: dict[str, Any]


class MediaStage(Protocol):
    def prepare(self, item: ValidatedPreviewRow) -> PreparedMedia: ...


class EncoderStage(Protocol):
    contract: Mapping[str, Any]
    checkpoint_identities: Mapping[str, Any]

    def encode(
        self, item: ValidatedPreviewRow, media: PreparedMedia
    ) -> EncodedSample: ...


def _validate_preview_row(
    raw: Any,
    *,
    line_number: int,
    allow_failed_selection_gates: bool = False,
) -> ValidatedPreviewRow:
    row = _closed_mapping(
        raw,
        fields=PREVIEW_FIELDS,
        context=f"preview manifest line {line_number}",
    )
    if row["schema_version"] != PREVIEW_ROW_FORMAT:
        raise MaterializationError(
            f"preview line {line_number} format must be {PREVIEW_ROW_FORMAT}"
        )
    iid = _iid(row["iid"], context=f"IID on line {line_number}")
    if (
        row["preview_only"] is not True
        or row["training_authorized"] is not False
        or row["training_use_forbidden"] is not True
        or row["production_eligible"] is not False
        or row["post_video_acceptance"] != "pending"
    ):
        raise MaterializationError(f"preview authorization state differs for {iid}")
    digest = _sha(row["row_digest"], context=f"row digest for {iid}")
    candidate = dict(row)
    candidate.pop("row_digest")
    if object_sha256(candidate) != digest:
        raise MaterializationError(f"preview row digest mismatch for {iid}")
    gates = row["selection_gates"]
    if not isinstance(gates, Mapping) or not gates or any(
        type(value) is not bool for value in gates.values()
    ):
        raise MaterializationError(f"preview selection gates are invalid for {iid}")
    if not allow_failed_selection_gates and any(
        value is not True for value in gates.values()
    ):
        raise MaterializationError(f"preview selection gates are not all true for {iid}")

    instruction = _text(row["edit_instruction"], context=f"instruction for {iid}")
    instruction_sha = _sha(
        row["edit_instruction_sha256"], context=f"instruction hash for {iid}"
    )
    if text_sha256(instruction) != instruction_sha:
        raise MaterializationError(f"instruction hash mismatch for {iid}")
    generation_instruction = _text(
        row["generation_instruction"], context=f"generation instruction for {iid}"
    )
    if text_sha256(generation_instruction) != _sha(
        row["generation_instruction_sha256"],
        context=f"generation instruction hash for {iid}",
    ):
        raise MaterializationError(f"generation instruction hash mismatch for {iid}")
    if row["instruction_source"] not in {"structured", "natural"}:
        raise MaterializationError(f"unknown instruction source for {iid}")

    source = _verified_file(
        row["source_video_path"], row["source_video_sha256"], context=f"source {iid}"
    )
    target = _verified_file(
        row["target_video_path"], row["target_video_sha256"], context=f"target {iid}"
    )
    provenance = row["provenance"]
    if not isinstance(provenance, Mapping):
        raise MaterializationError(f"preview provenance must be an object for {iid}")
    generated_path = _verified_file(
        provenance.get("wan_generated_manifest_path"),
        provenance.get("wan_generated_manifest_sha256"),
        context=f"Wan generated manifest {iid}",
    )
    generated = _one_jsonl(generated_path, context=f"Wan generated manifest {iid}")
    if generated.get("schema_version") != WAN_GENERATED_FORMAT or generated.get("iid") != iid:
        raise MaterializationError(f"Wan generated identity differs for {iid}")
    if (
        generated.get("source_video_sha256") != row["source_video_sha256"]
        or generated.get("target_preview_mp4_sha256") != row["target_video_sha256"]
    ):
        raise MaterializationError(f"Wan media binding differs for {iid}")
    for generated_field, expected in (
        ("source_video", source),
        ("target_preview_mp4", target),
    ):
        declared = Path(
            _text(generated.get(generated_field), context=f"Wan {generated_field} {iid}")
        ).expanduser()
        _plain_file(declared, context=f"Wan {generated_field} {iid}")
        if declared.resolve(strict=True) != expected:
            raise MaterializationError(f"Wan {generated_field} path differs for {iid}")
    shared_i0_path = _verified_file(
        generated.get("conditioning_frame0_float32"),
        generated.get("conditioning_frame0_float32_sha256"),
        context=f"shared lossless I0 {iid}",
    )
    return ValidatedPreviewRow(
        row=row,
        source_video=source,
        target_video=target,
        shared_i0=Frame0Binding(
            path=shared_i0_path,
            sha256=str(generated["conditioning_frame0_float32_sha256"]),
        ),
        preview_join_row_digest=digest,
        preview_join_row_file_sha256=hashlib.sha256(
            canonical_json_bytes(row) + b"\n"
        ).hexdigest(),
    )


def load_preview_manifest(
    path: Path,
    *,
    max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    allow_failed_selection_gates: bool = False,
) -> list[ValidatedPreviewRow]:
    path = path.expanduser().resolve(strict=True)
    _plain_file(path, context="preview manifest")
    payload = path.read_bytes()
    if not payload.endswith(b"\n"):
        raise MaterializationError("preview manifest must end with a newline")
    lines = payload.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise MaterializationError("preview manifest must contain non-blank JSONL rows")
    items: list[ValidatedPreviewRow] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        raw = _parse_json(line, context=f"preview manifest line {line_number}")
        item = _validate_preview_row(
            raw,
            line_number=line_number,
            allow_failed_selection_gates=allow_failed_selection_gates,
        )
        iid = str(item.row["iid"])
        if iid in seen:
            raise MaterializationError(f"duplicate preview IID: {iid}")
        seen.add(iid)
        items.append(item)
    items.sort(key=lambda item: str(item.row["iid"]))
    if sample_ids is not None:
        requested = tuple(
            _iid(value, context="requested sample ID") for value in sample_ids
        )
        if not requested or len(set(requested)) != len(requested):
            raise MaterializationError(
                "sample_ids must be a non-empty sequence without duplicates"
            )
        by_iid = {str(item.row["iid"]): item for item in items}
        missing = sorted(set(requested) - set(by_iid))
        if missing:
            raise MaterializationError(
                f"requested sample IDs are absent from preview manifest: {missing}"
            )
        items = [by_iid[iid] for iid in requested]
    if max_rows is not None:
        if type(max_rows) is not int or max_rows <= 0:
            raise MaterializationError("max_rows must be a positive integer")
        items = items[:max_rows]
    if not items:
        raise MaterializationError("no preview rows selected")
    return items


def _tensor_sha256(value: Any) -> str:
    """Hash shape, dtype, and contiguous tensor bytes without using pickle."""

    try:
        import torch
    except ImportError as error:
        raise MaterializationError("tensor hashing requires PyTorch") from error
    if not isinstance(value, torch.Tensor):
        raise MaterializationError("encoder output must be a torch.Tensor")
    tensor = value.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    try:
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    except (RuntimeError, TypeError):
        # Some lightweight test environments have Torch built against another
        # NumPy ABI.  Preserve the original tensor bytes through a uint8 view
        # instead of silently changing the hashed dtype.
        raw = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


class DecordOmniMediaStage:
    """Official Omni preprocessing with a shared lossless frame zero."""

    def __init__(
        self,
        *,
        temporal_mode: str = DEFAULT_TEMPORAL_MODE,
        spatial_profile: str = DEFAULT_SPATIAL_PROFILE,
        fps_tolerance: float = 1e-3,
        min_crop_retention: float = MIN_CROP_RETENTION,
    ) -> None:
        self.temporal_sampling = temporal_sampling_contract(temporal_mode)
        self.spatial_profile = spatial_profile_contract(spatial_profile)
        if not math.isfinite(fps_tolerance) or fps_tolerance < 0:
            raise MaterializationError("fps_tolerance must be finite and non-negative")
        self.fps_tolerance = float(fps_tolerance)
        if (
            not math.isfinite(min_crop_retention)
            or min_crop_retention <= 0.0
            or min_crop_retention > 1.0
        ):
            raise MaterializationError(
                "min_crop_retention must be finite and in (0,1]"
            )
        self.min_crop_retention = float(min_crop_retention)

    @staticmethod
    def _decode(path: Path, indices: Sequence[int]) -> tuple[Any, float, tuple[int, int]]:
        try:
            import decord
        except ImportError as error:
            raise MaterializationError("media decoding requires decord") from error
        try:
            reader = decord.VideoReader(str(path))
            total = len(reader)
            fps = float(reader.get_avg_fps())
            if total != SOURCE_FRAME_COUNT:
                raise MaterializationError(
                    f"video must contain exactly {SOURCE_FRAME_COUNT} frames: {path} has {total}"
                )
            frames = reader.get_batch(list(indices)).asnumpy()
        except MaterializationError:
            raise
        except Exception as error:
            raise MaterializationError(f"cannot decode video {path}: {error}") from error
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] != len(indices):
            raise MaterializationError(f"decoded RGB frame shape differs for {path}")
        return frames, fps, (int(frames.shape[1]), int(frames.shape[2]))

    @staticmethod
    def _transform_rgb(frames: Any, target_hw: tuple[int, int]) -> Any:
        try:
            import numpy as np
            import torch
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as tvf
        except ImportError as error:
            raise MaterializationError(
                "Omni preprocessing requires NumPy, PyTorch, and torchvision"
            ) from error
        height, width = int(frames.shape[1]), int(frames.shape[2])
        top, left, bottom, right = center_crop_box(height, width, target_hw)
        tensors = []
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise MaterializationError("decoded frames must be uint8 NumPy arrays")
            tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255.0)
            tensor = tvf.crop(tensor, top, left, bottom - top, right - left)
            tensor = tvf.resize(
                tensor,
                list(target_hw),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensors.append(tensor.mul(2.0).sub(1.0))
        return torch.stack(tensors, dim=1).contiguous()

    @staticmethod
    def _transform_shared_i0(
        path: Path, target_hw: tuple[int, int]
    ) -> tuple[Any, tuple[int, int, int, int], tuple[int, int]]:
        try:
            import numpy as np
            import torch
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms import functional as tvf
        except ImportError as error:
            raise MaterializationError(
                "lossless I0 loading requires NumPy, PyTorch, and torchvision"
            ) from error
        try:
            array = np.load(path, allow_pickle=False)
        except Exception as error:
            raise MaterializationError(f"cannot load lossless I0 {path}: {error}") from error
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype("<f4")
            or array.ndim != 3
            or array.shape[0] != 3
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise MaterializationError(
                "lossless I0 must be finite C-contiguous little-endian float32 [3,H,W]"
            )
        height, width = int(array.shape[1]), int(array.shape[2])
        crop = center_crop_box(height, width, target_hw)
        top, left, bottom, right = crop
        shared = torch.from_numpy(array.copy())
        shared = tvf.crop(shared, top, left, bottom - top, right - left)
        shared = tvf.resize(
            shared,
            list(target_hw),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ).float().clamp_(-1.0, 1.0).contiguous()
        if not bool(torch.isfinite(shared).all()):
            raise MaterializationError("transformed lossless I0 is non-finite")
        return shared, crop, (height, width)

    def prepare(self, item: ValidatedPreviewRow) -> PreparedMedia:
        indices = self.temporal_sampling.frame_indices
        source_frames, source_fps, source_hw = self._decode(item.source_video, indices)
        target_frames, target_fps, target_hw_input = self._decode(item.target_video, indices)
        for name, fps in (("source", source_fps), ("target", target_fps)):
            if not math.isfinite(fps) or abs(fps - SOURCE_FPS) > self.fps_tolerance:
                raise MaterializationError(
                    f"{name} FPS must be {SOURCE_FPS}, got {fps} for {item.row['iid']}"
                )
        bucket = choose_bucket(
            *source_hw, spatial_profile=self.spatial_profile.profile
        )
        if (
            choose_bucket(
                *target_hw_input, spatial_profile=self.spatial_profile.profile
            )
            != bucket
        ):
            raise MaterializationError(
                f"source/target orientations differ for {item.row['iid']}"
            )
        source = self._transform_rgb(source_frames, bucket)
        target = self._transform_rgb(target_frames, bucket)
        shared, shared_crop, shared_hw = self._transform_shared_i0(
            item.shared_i0.path, bucket
        )
        if (
            choose_bucket(*shared_hw, spatial_profile=self.spatial_profile.profile)
            != bucket
        ):
            raise MaterializationError(
                f"lossless I0 orientation differs for {item.row['iid']}"
            )
        retentions = {
            "source": crop_retention(*source_hw, bucket),
            "target": crop_retention(*target_hw_input, bucket),
            "shared_i0": crop_retention(*shared_hw, bucket),
        }
        if min(retentions.values()) < self.min_crop_retention:
            raise MaterializationError(
                "center-crop retention is below "
                f"{self.min_crop_retention:.3f} for {item.row['iid']}: "
                f"{retentions}"
            )
        # Both endpoints receive the exact same tensor.  This is stronger than
        # trusting either H.264 decode at frame zero.
        source[:, 0].copy_(shared)
        target[:, 0].copy_(shared)
        try:
            import torch
        except ImportError as error:
            raise MaterializationError("media validation requires PyTorch") from error
        if not torch.equal(source[:, 0], target[:, 0]):
            raise MaterializationError(f"shared frame zero differs for {item.row['iid']}")
        metadata = {
            "temporal_mode": self.temporal_sampling.mode,
            "spatial_profile": self.spatial_profile.profile,
            "frame_indices": list(indices),
            "source_frame_count": SOURCE_FRAME_COUNT,
            "target_frame_count": SOURCE_FRAME_COUNT,
            "materialized_frame_count": self.temporal_sampling.materialized_frame_count,
            "source_fps": source_fps,
            "target_fps": target_fps,
            "materialized_fps": self.temporal_sampling.materialized_fps,
            "sampling_policy": self.temporal_sampling.sampling_policy,
            "temporal_subsampled": self.temporal_sampling.temporal_subsampled,
            "bucket_hw": list(bucket),
            "resize_interpolation": "torchvision_bilinear_antialias_true",
            "normalization": "uint8_to_float32_then_x_div_127.5_minus_1",
            "shared_i0_resize_policy": (
                "torchvision_bilinear_antialias_true_then_clamp_-1_1"
            ),
            "source_input_hw": list(source_hw),
            "target_input_hw": list(target_hw_input),
            "shared_i0_input_hw": list(shared_hw),
            "source_crop_tlbr": list(center_crop_box(*source_hw, bucket)),
            "target_crop_tlbr": list(center_crop_box(*target_hw_input, bucket)),
            "shared_i0_crop_tlbr": list(shared_crop),
            "crop_retention": retentions,
            "min_crop_retention_gate": self.min_crop_retention,
            "source_tensor_sha256": _tensor_sha256(source),
            "target_tensor_sha256": _tensor_sha256(target),
            "shared_frame0_tensor_sha256": _tensor_sha256(shared),
            "shared_frame0_exact": True,
        }
        return PreparedMedia(
            source_video=source,
            target_video=target,
            source_qwen_path=str(item.source_video),
            target_qwen_path=str(item.target_video),
            metadata=metadata,
        )


def deterministic_motion_pool(tokens: Any, num_tokens: int) -> Any:
    """Deterministically mean-pool [L,2048] text features into [K,2048]."""

    try:
        import torch
    except ImportError as error:
        raise MaterializationError("motion pooling requires PyTorch") from error
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
        raise MaterializationError("motion text features must be a rank-2 tensor")
    if tokens.shape[1] != VLM_DIM or tokens.shape[0] <= 0:
        raise MaterializationError(f"motion text features must have shape [L,{VLM_DIM}]")
    if type(num_tokens) is not int or num_tokens <= 0:
        raise MaterializationError("motion token count must be a positive integer")
    length = int(tokens.shape[0])
    pooled = []
    for index in range(num_tokens):
        start = (index * length) // num_tokens
        end = ((index + 1) * length) // num_tokens
        if end <= start:
            position = min(start, length - 1)
            pooled.append(tokens[position])
        else:
            pooled.append(tokens[start:end].float().mean(dim=0).to(tokens.dtype))
    result = torch.stack(pooled, dim=0).float().cpu().contiguous()
    if result.shape != (num_tokens, VLM_DIM) or not bool(torch.isfinite(result).all()):
        raise MaterializationError("pooled motion tokens violate the fixed contract")
    return result


def canonical_motion_record(raw_text: Any) -> str:
    """Normalize only an optional Markdown fence, then enforce three lines."""

    if type(raw_text) is not str or not raw_text.strip():
        raise MaterializationError("Qwen canonical motion record is empty")
    text = raw_text.strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() != "```":
            raise MaterializationError("Qwen motion record has an unclosed fence")
        lines = lines[1:-1]
    lines = [line.strip() for line in lines if line.strip()]
    prefixes = ("SUBJECT_MOTION:", "CAMERA_MOTION:", "TIMING:")
    if len(lines) != len(prefixes) or any(
        not line.startswith(prefix) or not line[len(prefix):].strip()
        for line, prefix in zip(lines, prefixes)
    ):
        raise MaterializationError(
            "Qwen canonical motion record must contain exactly the required three lines"
        )
    return "\n".join(lines)


def _tree_checkpoint_identity(label: str, roots: Sequence[Path]) -> dict[str, Any]:
    """Hash a deterministic, content-addressed checkpoint file index."""

    index: list[dict[str, Any]] = []
    total_bytes = 0
    for root_index, raw_root in enumerate(roots):
        root = raw_root.expanduser().resolve(strict=True)
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            relative_path = Path(path.name) if root.is_file() else path.relative_to(root)
            if any(part in {".cache", ".git"} for part in relative_path.parts):
                continue
            if not path.is_file():
                continue
            if path.is_symlink():
                raise MaterializationError(
                    f"checkpoint {label} contains a symlink: {path}"
                )
            resolved = path.resolve(strict=True)
            relative = str(relative_path)
            size = resolved.stat().st_size
            total_bytes += size
            index.append(
                {
                    "root_index": root_index,
                    "path": relative,
                    "bytes": size,
                    "sha256": file_sha256(resolved),
                }
            )
    if not index:
        raise MaterializationError(f"checkpoint {label} has no files")
    return {
        "label": label,
        "manifest_sha256": object_sha256(index),
        "file_count": len(index),
        "total_bytes": total_bytes,
    }


class OfficialOmniEncoderStage:
    """Frozen official OmniVideo2-1.3B Qwen/UMT5/Wan encoder contract."""

    def __init__(
        self,
        *,
        omni_root: Path,
        qwen_checkpoint: Path,
        vae_checkpoint: Path,
        umt5_checkpoint: Path,
        umt5_tokenizer: Path,
        motion_tokens: int,
        device: str,
        t5_device: str,
        qwen_device_map: str,
        qwen_flash_attention: bool,
        qwen_video_frames: int,
        temporal_mode: str = DEFAULT_TEMPORAL_MODE,
        spatial_profile: str = DEFAULT_SPATIAL_PROFILE,
    ) -> None:
        if type(motion_tokens) is not int or motion_tokens <= 0:
            raise MaterializationError("motion_tokens must be positive")
        temporal_sampling = temporal_sampling_contract(temporal_mode)
        spatial_sampling = spatial_profile_contract(spatial_profile)
        omni_root = omni_root.expanduser().resolve(strict=True)
        if str(omni_root) not in sys.path:
            sys.path.insert(0, str(omni_root))
        try:
            import torch
            from omnivideo.configs import WAN_CONFIGS
            from omnivideo.modules.t5 import T5EncoderModel
            from omnivideo.modules.vae2_1 import Wan2_1_VAE
            from omnivideo.vllm_model import load_qwen3vl_model_and_processor
        except ImportError as error:
            raise MaterializationError(
                f"cannot import official OmniVideo encoder code: {error}"
            ) from error

        self.torch = torch
        self.motion_tokens = motion_tokens
        self.device = torch.device(device)
        self.t5_device = torch.device(t5_device)
        self.qwen_checkpoint = qwen_checkpoint.expanduser().resolve(strict=True)
        self.vae_checkpoint = vae_checkpoint.expanduser().resolve(strict=True)
        self.umt5_checkpoint = umt5_checkpoint.expanduser().resolve(strict=True)
        self.umt5_tokenizer = umt5_tokenizer.expanduser().resolve(strict=True)
        config = WAN_CONFIGS["t2v-1.3B"]

        qwen_identity = _tree_checkpoint_identity("Qwen3-VL", [self.qwen_checkpoint])
        umt5_identity = _tree_checkpoint_identity(
            "UMT5-XXL", [self.umt5_checkpoint, self.umt5_tokenizer]
        )
        vae_identity = _tree_checkpoint_identity("Wan2.1-VAE", [self.vae_checkpoint])
        vae_identity["checkpoint_sha256"] = file_sha256(self.vae_checkpoint)
        omni_source_identity = _tree_checkpoint_identity(
            "OmniVideo encoder source",
            [
                omni_root / "omnivideo" / "vllm_model.py",
                omni_root / "omnivideo" / "modules" / "t5.py",
                omni_root / "omnivideo" / "modules" / "vae2_1.py",
                omni_root / "omnivideo" / "configs" / "__init__.py",
                omni_root / "omnivideo" / "configs" / "shared_config.py",
                omni_root / "omnivideo" / "configs" / "wan_t2v_1_3B.py",
            ],
        )
        self.checkpoint_identities = {
            "qwen": qwen_identity,
            "umt5": umt5_identity,
            "vae": vae_identity,
            "omnivideo_source": omni_source_identity,
        }
        vae_preprocess = {
            "implementation": (
                "project Decord uint8 to float tensor center-crop and "
                "torchvision bilinear-antialias resize"
            ),
            "official_vae_encoder": "omnivideo.modules.vae2_1.Wan2_1_VAE",
            **temporal_sampling.audit_dict(),
            **spatial_sampling.audit_dict(),
            "buckets_hw": [
                list(spatial_sampling.landscape_bucket_hw),
                list(spatial_sampling.portrait_bucket_hw),
            ],
            "crop": "center_crop_per_endpoint_to_shared_bucket_aspect",
            "resize": "torchvision_bilinear_antialias_true",
            "normalization": [-1.0, 1.0],
            "frame0": (
                "shared_verified_float32_npy resized then clamped to [-1,1] "
                "and copied exactly into both endpoints"
            ),
        }
        umt5_preprocess = {
            "official_class": "omnivideo.modules.t5.T5EncoderModel",
            "max_length": 512,
            "segment_order": ["target_caption", "edit_instruction"],
            "padding": "slice_to_attention_mask_length",
        }
        vlm_preprocess = {
            "official_source_function": "generate_caption_and_extract_features",
            "source_input": "source_video_plus_edit_instruction",
            "target_caption": "predicted_from_source_caption_plus_instruction",
            "teacher_input": "target_video_to_canonical_motion_text",
            "teacher_encoding": "canonical_motion_text_only",
            "teacher_pool": "deterministic_integer_bins_mean",
            "teacher_tokens": motion_tokens,
            "target_tokens_usage": "planner_loss_only",
            "source_caption_system_prompt_sha256": text_sha256(
                config.source_caption_system_prompt
            ),
            "target_caption_system_prompt_sha256": text_sha256(
                config.target_caption_system_prompt
            ),
            "feature_extraction_system_prompt_sha256": text_sha256(
                config.feature_extraction_system_prompt
            ),
            "motion_video_system_prompt_sha256": text_sha256(
                MOTION_VIDEO_SYSTEM_PROMPT
            ),
            "motion_video_user_prompt_sha256": text_sha256(
                MOTION_VIDEO_USER_PROMPT
            ),
            "motion_text_feature_system_prompt_sha256": text_sha256(
                MOTION_TEXT_FEATURE_SYSTEM_PROMPT
            ),
            "qwen_video_frames": qwen_video_frames,
            "qwen_flash_attention": qwen_flash_attention,
        }
        self.contract = {
            "format": "pact-omnivideo2-offline-encoder-contract-v1",
            "vae": {
                "checkpoint_sha256": vae_identity["checkpoint_sha256"],
                "preprocessing_contract_sha256": object_sha256(vae_preprocess),
                "input_pixel_range": [-1.0, 1.0],
                "posterior_mode": "mean",
                "channel_mean": [
                    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653,
                    -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632,
                    -0.1922, -0.9497, 0.2503, -0.2921,
                ],
                "channel_std": [
                    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708,
                    2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
                    1.6382, 1.1253, 2.8251, 1.9160,
                ],
                "stride": [4, 8, 8],
            },
            "umt5": {
                "checkpoint_manifest_sha256": umt5_identity["manifest_sha256"],
                "preprocessing_contract_sha256": object_sha256(umt5_preprocess),
                "embedding_dim": 4096,
                "max_sequence_length_per_segment": 512,
                "segment_order": ["target_caption", "edit_instruction"],
                "padding_policy": "slice_to_attention_mask_length",
            },
            "vlm": {
                "checkpoint_manifest_sha256": qwen_identity["manifest_sha256"],
                "feature_extraction_contract_sha256": object_sha256(vlm_preprocess),
                "embedding_dim": VLM_DIM,
                "feature_tensor": "vlm_last_hidden_states",
                "token_selection": "attention_mask_then_drop_system_prefix",
            },
        }

        self.qwen_model, self.qwen_processor = load_qwen3vl_model_and_processor(
            str(self.qwen_checkpoint),
            device="cuda" if self.device.type == "cuda" else str(self.device),
            dtype="bf16" if self.device.type == "cuda" else "fp32",
            flash_attn=qwen_flash_attention,
            video_nframes=qwen_video_frames,
            device_map=qwen_device_map,
        )
        self.text_encoder = T5EncoderModel(
            text_len=int(config.text_len),
            dtype=config.t5_dtype,
            device=self.t5_device,
            checkpoint_path=str(self.umt5_checkpoint),
            tokenizer_path=str(self.umt5_tokenizer),
        )
        self.vae = Wan2_1_VAE(
            vae_pth=str(self.vae_checkpoint),
            dtype=torch.float32,
            device=str(self.device),
        )
        self.config = config

    def _motion_only_text(self, target_video_path: str) -> str:
        torch = self.torch
        processor = self.qwen_processor
        model = self.qwen_model
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": MOTION_VIDEO_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": target_video_path},
                    {"type": "text", "text": MOTION_VIDEO_USER_PROMPT},
                ],
            },
        ]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if hasattr(processor, "video_processor"):
            count = getattr(processor.video_processor, "num_frames", 6) or 6
            kwargs.update({"num_frames": count, "do_sample_frames": True})
            if hasattr(processor.video_processor, "size"):
                kwargs["size"] = processor.video_processor.size
        inputs = processor.apply_chat_template(messages, **kwargs)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        converted = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(device)
                if "pixel_values" in key:
                    value = value.to(dtype)
            converted[key] = value
        with torch.no_grad():
            output = model.generate(**converted, max_new_tokens=192, do_sample=False)
        input_length = converted["input_ids"].shape[1]
        text = processor.batch_decode(
            output[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return canonical_motion_record(text)

    def encode(self, item: ValidatedPreviewRow, media: PreparedMedia) -> EncodedSample:
        torch = self.torch
        try:
            from omnivideo.vllm_model import (
                extract_qwen3vl_features,
                generate_caption_and_extract_features,
            )
        except ImportError as error:
            raise MaterializationError(f"cannot import official Qwen helpers: {error}") from error
        instruction = str(item.row["edit_instruction"])
        with torch.no_grad():
            target_caption, source_features = generate_caption_and_extract_features(
                self.qwen_model,
                self.qwen_processor,
                media.source_qwen_path,
                instruction,
                source_caption_system_prompt=self.config.source_caption_system_prompt,
                target_caption_system_prompt=self.config.target_caption_system_prompt,
                feature_extraction_system_prompt=self.config.feature_extraction_system_prompt,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
            )
            target_caption = target_caption.strip()
            if not target_caption:
                raise MaterializationError("official target-caption prediction is empty")
            source_vlm = source_features["vlm_last_hidden_states"].float().cpu().contiguous()
            if source_vlm.ndim != 2 or source_vlm.shape[1] != VLM_DIM:
                raise MaterializationError("official source Qwen features must be [L,2048]")

            motion_text = self._motion_only_text(media.target_qwen_path)
            motion_features = extract_qwen3vl_features(
                self.qwen_model,
                self.qwen_processor,
                "",
                motion_text,
                system_prompt=MOTION_TEXT_FEATURE_SYSTEM_PROMPT,
            )["vlm_last_hidden_states"]
            target_motion_tokens = deterministic_motion_pool(
                motion_features, self.motion_tokens
            )

            target_text = self.text_encoder([target_caption], self.t5_device)[0]
            instruction_text = self.text_encoder([instruction], self.t5_device)[0]
            text_context = (
                torch.cat([target_text, instruction_text], dim=0)
                .float()
                .cpu()
                .contiguous()
            )
            source_latent = self.vae.encode(
                [media.source_video.to(self.device, dtype=torch.float32)]
            )[0].float().cpu().contiguous()
            target_latent = self.vae.encode(
                [media.target_video.to(self.device, dtype=torch.float32)]
            )[0].float().cpu().contiguous()
        if source_latent.shape != target_latent.shape:
            raise MaterializationError("source and target VAE latent shapes differ")
        return EncodedSample(
            source_latent=source_latent,
            target_latent=target_latent,
            text_context=text_context,
            source_vlm_context=source_vlm,
            target_motion_tokens=target_motion_tokens,
            target_caption=target_caption,
            motion_text=motion_text,
            metadata={
                "target_caption_origin": (
                    "official_qwen_source_caption_plus_instruction_prediction"
                ),
                "source_vlm_origin": "official_qwen_source_video_plus_instruction",
                "motion_teacher_visual_input": "target_video_only",
                "motion_teacher_feature_input": "canonical_motion_text_only",
                "target_motion_tokens_usage": "planner_loss_only",
                "motion_pool": "deterministic_integer_bins_mean",
            },
        )


def _encoder_contract_sha256(contract: Mapping[str, Any]) -> str:
    if str(METHOD_ROOT) not in sys.path:
        sys.path.insert(0, str(METHOD_ROOT))
    try:
        from pact.dataset import encoder_contract_sha256
    except ImportError as error:
        raise MaterializationError(f"cannot validate encoder contract: {error}") from error
    try:
        return encoder_contract_sha256(contract)
    except ValueError as error:
        raise MaterializationError(f"invalid encoder contract: {error}") from error


def _validated_payload(
    *,
    sample_id: str,
    encoder_contract: Mapping[str, Any],
    encoded: EncodedSample,
) -> dict[str, Any]:
    if str(METHOD_ROOT) not in sys.path:
        sys.path.insert(0, str(METHOD_ROOT))
    try:
        from action.config import ACTION_TASK_TYPES
        from action.dataset import validate_action_payload
    except ImportError as error:
        raise MaterializationError(f"cannot import action payload contract: {error}") from error
    payload = {
        "format": ACTION_PAYLOAD_FORMAT,
        "sample_id": sample_id,
        "encoder_contract": dict(encoder_contract),
        "source_latent": encoded.source_latent,
        "target_latent": encoded.target_latent,
        "text_context": encoded.text_context,
        "source_vlm_context": encoded.source_vlm_context,
        "target_motion_tokens": encoded.target_motion_tokens,
        "task_type": "action_edit",
        "preview_only": True,
    }
    try:
        return validate_action_payload(
            payload,
            expected_motion_tokens=int(encoded.target_motion_tokens.shape[0]),
            allowed_task_types=ACTION_TASK_TYPES,
        )
    except (ValueError, AttributeError) as error:
        raise MaterializationError(f"invalid action payload for {sample_id}: {error}") from error


def _torch_save_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        import torch
    except ImportError as error:
        raise MaterializationError("payload serialization requires PyTorch") from error
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    return buffer.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise MaterializationError(f"create-only path already exists: {path}") from error


def _publish_tree_create_only(staging: Path, output: Path) -> None:
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o755)
    except FileExistsError as error:
        raise MaterializationError(f"create-only output already exists: {output}") from error
    try:
        for source in sorted(staging.rglob("*")):
            relative = source.relative_to(staging)
            destination = output / relative
            if source.is_dir():
                destination.mkdir(mode=0o755)
            elif source.is_file():
                os.link(source, destination)
            else:
                raise MaterializationError(f"unexpected staging entry: {source}")
        directory_fd = os.open(str(output), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # Leave an incomplete create-only directory.  Never silently retry into
        # it: an operator must audit and choose a new output path.
        raise


def _reverify_inputs(item: ValidatedPreviewRow) -> None:
    for name, path, digest in (
        ("source", item.source_video, item.row["source_video_sha256"]),
        ("target", item.target_video, item.row["target_video_sha256"]),
        ("shared I0", item.shared_i0.path, item.shared_i0.sha256),
    ):
        if file_sha256(path) != digest:
            raise MaterializationError(
                f"{name} changed while materializing {item.row['iid']}"
            )


def _validate_media_sampling_metadata(
    metadata: Mapping[str, Any],
    *,
    temporal_sampling: TemporalSamplingContract,
    spatial_sampling: SpatialProfileContract,
    sample_id: str,
) -> None:
    """Prevent a custom media stage from silently changing sampling contracts."""

    expected = {
        **temporal_sampling.audit_dict(),
        "spatial_profile": spatial_sampling.profile,
        "target_frame_count": temporal_sampling.source_frame_count,
        "target_fps": temporal_sampling.source_fps,
    }
    for key, expected_value in expected.items():
        actual_value = metadata.get(key)
        if type(expected_value) is float:
            matches = (
                type(actual_value) is float
                and math.isfinite(actual_value)
                and math.isclose(
                    actual_value,
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1e-3 if key in {"source_fps", "target_fps"} else 1e-9,
                )
            )
        else:
            matches = (
                type(actual_value) is type(expected_value)
                and actual_value == expected_value
            )
        if not matches:
            raise MaterializationError(
                f"media sampling metadata mismatch for {sample_id}: "
                f"{key} expected={expected_value!r} actual={actual_value!r}"
            )
    allowed_buckets = {
        spatial_sampling.landscape_bucket_hw,
        spatial_sampling.portrait_bucket_hw,
    }
    bucket = metadata.get("bucket_hw")
    if (
        type(bucket) is not list
        or len(bucket) != 2
        or any(type(value) is not int for value in bucket)
        or tuple(bucket) not in allowed_buckets
    ):
        raise MaterializationError(
            f"media spatial metadata mismatch for {sample_id}: "
            f"bucket_hw expected one of {sorted(allowed_buckets)!r} actual={bucket!r}"
        )


def _validate_prepared_media_geometry(
    media: PreparedMedia,
    *,
    temporal_sampling: TemporalSamplingContract,
    spatial_sampling: SpatialProfileContract,
    sample_id: str,
) -> None:
    """Validate the actual decoded tensors, not only stage-reported metadata."""

    try:
        import torch
    except ImportError as error:
        raise MaterializationError("media geometry validation requires PyTorch") from error
    bucket = media.metadata.get("bucket_hw")
    allowed_buckets = {
        spatial_sampling.landscape_bucket_hw,
        spatial_sampling.portrait_bucket_hw,
    }
    if not isinstance(bucket, list) or tuple(bucket) not in allowed_buckets:
        raise MaterializationError(f"invalid prepared media bucket for {sample_id}")
    expected_shape = (
        3,
        temporal_sampling.materialized_frame_count,
        int(bucket[0]),
        int(bucket[1]),
    )
    for name, tensor in (
        ("source", media.source_video),
        ("target", media.target_video),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_shape
            or not tensor.is_floating_point()
            or tensor.device.type != "cpu"
        ):
            raise MaterializationError(
                f"prepared {name} tensor geometry differs for {sample_id}: "
                f"expected={expected_shape}, actual={getattr(tensor, 'shape', None)}"
            )
    if not torch.equal(media.source_video[:, 0], media.target_video[:, 0]):
        raise MaterializationError(
            f"prepared source/target frame zero differs for {sample_id}"
        )


def materialize_action_payloads(
    *,
    preview_manifest: Path,
    output_dir: Path,
    allow_preview_exploration: bool = False,
    max_rows: Optional[int] = None,
    media_stage: Optional[MediaStage] = None,
    encoder_stage: Optional[EncoderStage] = None,
    omni_root: Optional[Path] = None,
    qwen_checkpoint: Optional[Path] = None,
    vae_checkpoint: Optional[Path] = None,
    umt5_checkpoint: Optional[Path] = None,
    umt5_tokenizer: Optional[Path] = None,
    motion_tokens: int = 16,
    device: str = "cuda:0",
    t5_device: str = "cuda:0",
    qwen_device_map: str = "balanced_low_0",
    qwen_flash_attention: bool = False,
    qwen_video_frames: int = 6,
    temporal_mode: str = DEFAULT_TEMPORAL_MODE,
    spatial_profile: str = DEFAULT_SPATIAL_PROFILE,
    min_crop_retention: float = MIN_CROP_RETENTION,
    sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    if allow_preview_exploration is not True:
        raise MaterializationError(
            "preview rows require explicit allow_preview_exploration=True"
        )
    if (
        not math.isfinite(min_crop_retention)
        or min_crop_retention <= 0.0
        or min_crop_retention > 1.0
    ):
        raise MaterializationError(
            "min_crop_retention must be finite and in (0,1]"
        )
    temporal_sampling = temporal_sampling_contract(temporal_mode)
    spatial_sampling = spatial_profile_contract(spatial_profile)
    preview_manifest = preview_manifest.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise MaterializationError(f"create-only output already exists: {output_dir}")
    items = load_preview_manifest(
        preview_manifest, max_rows=max_rows, sample_ids=sample_ids
    )
    preview_manifest_sha = file_sha256(preview_manifest)
    if media_stage is None:
        media_stage = DecordOmniMediaStage(
            temporal_mode=temporal_mode,
            spatial_profile=spatial_profile,
            min_crop_retention=min_crop_retention,
        )
    if encoder_stage is None:
        required = {
            "qwen_checkpoint": qwen_checkpoint,
            "vae_checkpoint": vae_checkpoint,
            "umt5_checkpoint": umt5_checkpoint,
            "umt5_tokenizer": umt5_tokenizer,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise MaterializationError(
                f"official encoder paths are required: missing={missing}"
            )
        encoder_stage = OfficialOmniEncoderStage(
            omni_root=DEFAULT_OMNI_ROOT if omni_root is None else omni_root,
            qwen_checkpoint=qwen_checkpoint,  # type: ignore[arg-type]
            vae_checkpoint=vae_checkpoint,  # type: ignore[arg-type]
            umt5_checkpoint=umt5_checkpoint,  # type: ignore[arg-type]
            umt5_tokenizer=umt5_tokenizer,  # type: ignore[arg-type]
            motion_tokens=motion_tokens,
            device=device,
            t5_device=t5_device,
            qwen_device_map=qwen_device_map,
            qwen_flash_attention=qwen_flash_attention,
            qwen_video_frames=qwen_video_frames,
            temporal_mode=temporal_mode,
            spatial_profile=spatial_profile,
        )
    contract = dict(encoder_stage.contract)
    contract_sha = _encoder_contract_sha256(contract)
    checkpoint_identities = dict(encoder_stage.checkpoint_identities)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.staging.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "payloads").mkdir()
        (staging / "provenance").mkdir()
        manifest_rows: list[dict[str, Any]] = []
        provenance_index: list[dict[str, str]] = []
        for item in items:
            iid = str(item.row["iid"])
            media = media_stage.prepare(item)
            _validate_media_sampling_metadata(
                media.metadata,
                temporal_sampling=temporal_sampling,
                spatial_sampling=spatial_sampling,
                sample_id=iid,
            )
            _validate_prepared_media_geometry(
                media,
                temporal_sampling=temporal_sampling,
                spatial_sampling=spatial_sampling,
                sample_id=iid,
            )
            encoded = encoder_stage.encode(item, media)
            if not isinstance(encoded.target_caption, str) or not encoded.target_caption.strip():
                raise MaterializationError(f"empty target caption for {iid}")
            if not isinstance(encoded.motion_text, str) or not encoded.motion_text.strip():
                raise MaterializationError(f"empty canonical motion text for {iid}")
            if encoded.metadata.get("target_motion_tokens_usage") != "planner_loss_only":
                raise MaterializationError(
                    f"encoder must mark target motion tokens planner-loss-only for {iid}"
                )
            payload = _validated_payload(
                sample_id=iid, encoder_contract=contract, encoded=encoded
            )
            payload_bytes = _torch_save_bytes(payload)
            payload_relative = Path("payloads") / f"{iid}.pt"
            payload_path = staging / payload_relative
            _write_exclusive(payload_path, payload_bytes)
            payload_sha = hashlib.sha256(payload_bytes).hexdigest()

            group_id = item.row.get("group_id")
            split_group = group_id if isinstance(group_id, str) and group_id else iid
            provenance = {
                "schema_version": PROVENANCE_FORMAT,
                "sample_id": iid,
                "parent_id": iid,
                "split_group": split_group,
                "direction": "forward",
                "task_type": "action_edit",
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "production_eligible": False,
                "post_video_acceptance": "pending",
                "preview_join": {
                    "manifest_path": str(preview_manifest),
                    "manifest_sha256": preview_manifest_sha,
                    "row_digest": item.preview_join_row_digest,
                    "row_file_sha256": item.preview_join_row_file_sha256,
                    "upstream_provenance_sha256": object_sha256(
                        item.row["provenance"]
                    ),
                },
                "media": {
                    "source_video_path": str(item.source_video),
                    "source_video_sha256": item.row["source_video_sha256"],
                    "target_video_path": str(item.target_video),
                    "target_video_sha256": item.row["target_video_sha256"],
                    "shared_i0_path": str(item.shared_i0.path),
                    "shared_i0_sha256": item.shared_i0.sha256,
                    "preprocessing": dict(media.metadata),
                },
                "conditioning": {
                    "instruction": item.row["edit_instruction"],
                    "instruction_sha256": item.row["edit_instruction_sha256"],
                    "instruction_source": item.row["instruction_source"],
                    "generation_instruction_sha256": item.row[
                        "generation_instruction_sha256"
                    ],
                    "target_caption": encoded.target_caption,
                    "target_caption_sha256": text_sha256(encoded.target_caption),
                    "target_caption_origin": encoded.metadata.get(
                        "target_caption_origin"
                    ),
                    "motion_text": encoded.motion_text,
                    "motion_text_sha256": text_sha256(encoded.motion_text),
                    "motion_teacher_visual_input": encoded.metadata.get(
                        "motion_teacher_visual_input"
                    ),
                    "motion_teacher_feature_input": encoded.metadata.get(
                        "motion_teacher_feature_input"
                    ),
                    "motion_pool": encoded.metadata.get("motion_pool"),
                    "target_motion_tokens_usage": "planner_loss_only",
                },
                "encoder": {
                    "contract": contract,
                    "contract_sha256": contract_sha,
                    "checkpoint_identities": checkpoint_identities,
                },
                "tensor_sha256": {
                    "source_latent": _tensor_sha256(encoded.source_latent),
                    "target_latent": _tensor_sha256(encoded.target_latent),
                    "text_context": _tensor_sha256(encoded.text_context),
                    "source_vlm_context": _tensor_sha256(
                        encoded.source_vlm_context
                    ),
                    "target_motion_tokens": _tensor_sha256(
                        encoded.target_motion_tokens
                    ),
                },
                "payload": {
                    "path": str(payload_relative),
                    "sha256": payload_sha,
                },
            }
            provenance_relative = Path("provenance") / f"{iid}.json"
            provenance_bytes = (
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            provenance_path = staging / provenance_relative
            _write_exclusive(provenance_path, provenance_bytes)
            provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
            manifest_row = {
                "format": ACTION_MANIFEST_FORMAT,
                "sample_id": iid,
                # Payload root is output_dir/payloads, matching ActionLatentDataset.
                "payload_path": payload_relative.name,
                "payload_sha256": payload_sha,
                "provenance_path": str(provenance_relative),
                "provenance_sha256": provenance_sha,
                "task_type": "action_edit",
                "preview_only": True,
            }
            _closed_mapping(
                manifest_row,
                fields=MATERIALIZED_MANIFEST_FIELDS,
                context=f"materialized manifest row {iid}",
            )
            manifest_rows.append(manifest_row)
            provenance_index.append(
                {
                    "sample_id": iid,
                    "path": str(provenance_relative),
                    "sha256": provenance_sha,
                }
            )
            _reverify_inputs(item)

        if file_sha256(preview_manifest) != preview_manifest_sha:
            raise MaterializationError("preview manifest changed during materialization")

        manifest_bytes = b"".join(
            canonical_json_bytes(row) + b"\n" for row in manifest_rows
        )
        manifest_path = staging / "manifest.jsonl"
        _write_exclusive(manifest_path, manifest_bytes)
        receipt = {
            "schema_version": RECEIPT_FORMAT,
            "complete": True,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "scientific_claim_authorized": False,
            "source_preview_manifest": str(preview_manifest),
            "source_preview_manifest_sha256": preview_manifest_sha,
            "sample_count": len(manifest_rows),
            "manifest": "manifest.jsonl",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "payload_root": "payloads",
            "provenance_index_sha256": object_sha256(provenance_index),
            "encoder_contract_sha256": contract_sha,
            "checkpoint_identities": checkpoint_identities,
            "temporal_mode": temporal_sampling.mode,
            "temporal_indices": list(temporal_sampling.frame_indices),
            "temporal_sampling_policy": temporal_sampling.sampling_policy,
            "temporal_subsampled": temporal_sampling.temporal_subsampled,
            "source_frame_count": temporal_sampling.source_frame_count,
            "materialized_frame_count": temporal_sampling.materialized_frame_count,
            "source_fps": temporal_sampling.source_fps,
            "materialized_fps": temporal_sampling.materialized_fps,
            "spatial_profile": spatial_sampling.profile,
            "landscape_bucket_hw": list(spatial_sampling.landscape_bucket_hw),
            "portrait_bucket_hw": list(spatial_sampling.portrait_bucket_hw),
            "min_crop_retention": min_crop_retention,
            "target_motion_tokens_usage": "planner_loss_only",
        }
        receipt["receipt_digest"] = object_sha256(receipt)
        _write_exclusive(
            staging / "materialization.json",
            (
                json.dumps(
                    receipt,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _publish_tree_create_only(staging, output_dir)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--omni-root", type=Path, default=DEFAULT_OMNI_ROOT)
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--umt5-checkpoint", type=Path, required=True)
    parser.add_argument("--umt5-tokenizer", type=Path, required=True)
    parser.add_argument("--motion-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--t5-device", default="cuda:0")
    parser.add_argument("--qwen-device-map", default="balanced_low_0")
    parser.add_argument("--qwen-video-frames", type=int, default=6)
    parser.add_argument(
        "--temporal-mode",
        choices=TEMPORAL_MODES,
        default=DEFAULT_TEMPORAL_MODE,
        help=(
            "Temporal preprocessing contract. Defaults to all 81 frames at 25 FPS; "
            "the 41-frame stride-2 option is explicit smoke/ablation only."
        ),
    )
    parser.add_argument(
        "--spatial-profile",
        choices=SPATIAL_PROFILES,
        default=DEFAULT_SPATIAL_PROFILE,
        help=(
            "Output bucket profile. motion_384p lowers only spatial resolution; "
            "it does not enable temporal subsampling."
        ),
    )
    parser.add_argument(
        "--min-crop-retention", type=float, default=MIN_CROP_RETENTION
    )
    flash_group = parser.add_mutually_exclusive_group()
    flash_group.add_argument(
        "--qwen-flash-attention",
        dest="qwen_flash_attention",
        action="store_true",
        help="Opt into flash-attention after a compatible-runtime probe.",
    )
    flash_group.add_argument(
        "--no-qwen-flash-attention",
        dest="qwen_flash_attention",
        action="store_false",
    )
    parser.set_defaults(qwen_flash_attention=False)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Materialize only this IID; repeat to preserve an explicit order.",
    )
    parser.add_argument(
        "--allow-preview-exploration",
        action="store_true",
        help="Acknowledge that outputs remain non-production preview artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        materialize_action_payloads(
            preview_manifest=args.preview_manifest,
            output_dir=args.output_dir,
            allow_preview_exploration=args.allow_preview_exploration,
            max_rows=args.max_rows,
            omni_root=args.omni_root,
            qwen_checkpoint=args.qwen_checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            umt5_checkpoint=args.umt5_checkpoint,
            umt5_tokenizer=args.umt5_tokenizer,
            motion_tokens=args.motion_tokens,
            device=args.device,
            t5_device=args.t5_device,
            qwen_device_map=args.qwen_device_map,
            qwen_flash_attention=args.qwen_flash_attention,
            qwen_video_frames=args.qwen_video_frames,
            temporal_mode=args.temporal_mode,
            spatial_profile=args.spatial_profile,
            min_crop_retention=args.min_crop_retention,
            sample_ids=args.sample_ids,
        )
    except MaterializationError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

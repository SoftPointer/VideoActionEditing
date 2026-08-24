#!/usr/bin/env python3
"""Materialize exact-81-frame Bernini-R VAE training rows.

This is a VAE-only replacement for Bernini's generic preprocessing utility.
It deliberately does not extract Qwen/VIT features: the renderer training
forward never consumes them.  Source and target are mapped to a common bucket
derived from the source aspect ratio, and the same lossless I0 tensor is copied
into both endpoints before either is encoded.

The input remains preview-only experimental data.  This program preserves that
state and never emits a production or scientific-use authorization.
"""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from tools import build_renderer_dataset as raw_builder  # noqa: E402


RAW_ROW_FORMAT = raw_builder.ROW_FORMAT
RAW_RECEIPT_FORMAT = raw_builder.RECEIPT_FORMAT
MATERIALIZED_ROW_FORMAT = "bernini-r-action-vae-row-v2"
SAMPLE_RECEIPT_FORMAT = "bernini-r-action-vae-sample-receipt-v2"
RANK_SUMMARY_FORMAT = "bernini-r-action-vae-rank-summary-v2"
FRAME_COUNT = 81
FPS = 25.0
LATENT_FRAME_COUNT = 21
DEFAULT_MAX_PIXELS = 245_760
DEFAULT_STRIDE = 16
DEFAULT_MIN_TARGET_RETENTION = 0.98
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class VaeMaterializationError(RuntimeError):
    """Fail-closed VAE materialization error."""


def canonical_json_bytes(value: Any) -> bytes:
    return raw_builder.canonical_json_bytes(value)


def object_sha256(value: Any) -> str:
    return raw_builder.object_sha256(value)


def file_sha256(path: Path) -> str:
    return raw_builder.file_sha256(path)


def _plain_file(path: Path, *, context: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise VaeMaterializationError(f"missing {context}: {path}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise VaeMaterializationError(f"{context} is not a plain file: {path}")
    return path


def _sha(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise VaeMaterializationError(f"{context} is not a lowercase SHA-256")
    return value


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    _plain_file(path, context=context)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VaeMaterializationError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise VaeMaterializationError(f"{context} must contain one object")
    return value


def _verify_file(path_value: Any, sha_value: Any, *, context: str) -> Path:
    if type(path_value) is not str or not path_value:
        raise VaeMaterializationError(f"invalid path for {context}")
    path = _plain_file(Path(path_value).expanduser().resolve(strict=True), context=context)
    expected = _sha(sha_value, context=f"{context} hash")
    actual = file_sha256(path)
    if actual != expected:
        raise VaeMaterializationError(
            f"{context} hash mismatch: expected={expected} actual={actual}"
        )
    return path


def source_aspect_bucket(
    height: int,
    width: int,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    stride: int = DEFAULT_STRIDE,
) -> tuple[int, int]:
    """Return the frozen floor-to-stride source-aspect bucket."""

    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise VaeMaterializationError("source dimensions must be positive integers")
    if type(stride) is not int or stride <= 0:
        raise VaeMaterializationError("stride must be positive")
    if type(max_pixels) is not int or max_pixels < stride * stride:
        raise VaeMaterializationError("max_pixels is too small")
    scale = math.sqrt(max_pixels / float(height * width))
    bucket_h = max(stride, math.floor(height * scale / stride) * stride)
    bucket_w = max(stride, math.floor(width * scale / stride) * stride)
    while bucket_h * bucket_w > max_pixels:
        if bucket_h >= bucket_w and bucket_h > stride:
            bucket_h -= stride
        elif bucket_w > stride:
            bucket_w -= stride
        else:
            raise VaeMaterializationError("cannot satisfy max_pixels with this stride")
    return bucket_h, bucket_w


def target_crop_to_source_aspect(
    target_height: int,
    target_width: int,
    source_height: int,
    source_width: int,
) -> tuple[tuple[int, int, int, int], float]:
    """Center-crop only the small source/target aspect mismatch."""

    dims = (target_height, target_width, source_height, source_width)
    if any(type(value) is not int or value <= 0 for value in dims):
        raise VaeMaterializationError("crop dimensions must be positive integers")
    source_ratio = source_width / source_height
    target_ratio = target_width / target_height
    if target_ratio > source_ratio:
        crop_h = target_height
        crop_w = min(target_width, max(1, round(target_height * source_ratio)))
    else:
        crop_w = target_width
        crop_h = min(target_height, max(1, round(target_width / source_ratio)))
    top = (target_height - crop_h) // 2
    left = (target_width - crop_w) // 2
    retention = (crop_h * crop_w) / float(target_height * target_width)
    return (top, left, top + crop_h, left + crop_w), retention


def _tensor_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise VaeMaterializationError("tensor hash input is not a tensor")
    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype), "shape": list(value.shape)}
    )
    raw = value.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _decode_exact_video(path: Path) -> tuple[Any, float, tuple[int, int]]:
    try:
        import decord
    except ImportError as error:
        raise VaeMaterializationError("video decoding requires decord") from error
    try:
        reader = decord.VideoReader(str(path), num_threads=1, ctx=decord.cpu(0))
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count != FRAME_COUNT:
            raise VaeMaterializationError(
                f"video must have exactly {FRAME_COUNT} frames: {path} has {frame_count}"
            )
        if not math.isfinite(fps) or abs(fps - FPS) > 1e-3:
            raise VaeMaterializationError(
                f"video must be {FPS} fps: {path} reports {fps}"
            )
        frames = reader.get_batch(list(range(FRAME_COUNT))).asnumpy()
    except VaeMaterializationError:
        raise
    except Exception as error:
        raise VaeMaterializationError(f"cannot decode {path}: {error}") from error
    if frames.shape[0] != FRAME_COUNT or frames.ndim != 4 or frames.shape[-1] != 3:
        raise VaeMaterializationError(f"decoded frame shape differs for {path}")
    return frames, fps, (int(frames.shape[1]), int(frames.shape[2]))


def _resize_video(frames: Any, bucket_hw: tuple[int, int], crop: Optional[tuple[int, int, int, int]]) -> Any:
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise VaeMaterializationError("decoded video must be uint8 NumPy RGB")
    video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    if crop is not None:
        top, left, bottom, right = crop
        video = tvf.crop(video, top, left, bottom - top, right - left)
    video = tvf.resize(
        video,
        list(bucket_hw),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return video.mul(2.0).sub(1.0).permute(1, 0, 2, 3).contiguous()


def _resize_shared_i0(
    path: Path,
    bucket_hw: tuple[int, int],
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    target_crop: tuple[int, int, int, int],
) -> tuple[Any, tuple[int, int], str]:
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    try:
        array = np.load(path, allow_pickle=False)
    except Exception as error:
        raise VaeMaterializationError(f"cannot load shared I0 {path}: {error}") from error
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype("<f4")
        or array.ndim != 3
        or array.shape[0] != 3
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
    ):
        raise VaeMaterializationError(
            "shared I0 must be finite C-contiguous little-endian float32 [3,H,W]"
        )
    i0_hw = (int(array.shape[1]), int(array.shape[2]))
    value = torch.from_numpy(array.copy())
    if i0_hw == target_hw:
        top, left, bottom, right = target_crop
        value = tvf.crop(value, top, left, bottom - top, right - left)
        alignment = "target_geometry_then_target_aspect_crop"
    elif i0_hw == source_hw:
        alignment = "source_geometry_no_crop"
    else:
        raise VaeMaterializationError(
            f"shared I0 geometry {i0_hw} matches neither source {source_hw} "
            f"nor target {target_hw}"
        )
    value = tvf.resize(
        value,
        list(bucket_hw),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return value.float().clamp_(-1.0, 1.0).contiguous(), i0_hw, alignment


def prepare_pair(
    row: Mapping[str, Any],
    *,
    max_pixels: int,
    stride: int,
    min_target_retention: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """Decode, align, and bind an exact shared first frame."""

    source_path = _verify_file(
        row.get("source_video_path"), row.get("source_video_sha256"), context="source video"
    )
    target_path = _verify_file(
        row.get("target_video_path"), row.get("target_video_sha256"), context="target video"
    )
    shared_path = _verify_file(
        row.get("shared_i0_path"), row.get("shared_i0_sha256"), context="shared I0"
    )
    source_frames, source_fps, source_hw = _decode_exact_video(source_path)
    target_frames, target_fps, target_hw = _decode_exact_video(target_path)
    bucket = source_aspect_bucket(
        *source_hw, max_pixels=max_pixels, stride=stride
    )
    crop, target_retention = target_crop_to_source_aspect(
        *target_hw, *source_hw
    )
    if target_retention < min_target_retention:
        raise VaeMaterializationError(
            f"target aspect-alignment retention {target_retention:.6f} is below "
            f"{min_target_retention:.6f}"
        )
    source = _resize_video(source_frames, bucket, None)
    target = _resize_video(target_frames, bucket, crop)
    shared, shared_i0_hw, shared_i0_alignment = _resize_shared_i0(
        shared_path, bucket, source_hw, target_hw, crop
    )
    source[:, 0].copy_(shared)
    target[:, 0].copy_(shared)
    import torch

    if not torch.equal(source[:, 0], target[:, 0]):
        raise VaeMaterializationError("source and target shared I0 differ")
    if source.shape != target.shape or tuple(source.shape[:2]) != (3, FRAME_COUNT):
        raise VaeMaterializationError("aligned source/target video shapes differ")
    metadata = {
        "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "source_reported_fps": source_fps,
        "target_reported_fps": target_fps,
        "source_input_hw": list(source_hw),
        "target_input_hw": list(target_hw),
        "shared_i0_input_hw": list(shared_i0_hw),
        "source_derived_bucket_hw": list(bucket),
        "bucket_rule": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
        "max_pixels": max_pixels,
        "stride": stride,
        "target_crop_tlbr": list(crop),
        "target_crop_retention": target_retention,
        "min_target_crop_retention": min_target_retention,
        "resize": "torchvision_bicubic_antialias_true",
        "normalization": "uint8_div_255_mul_2_minus_1",
        "shared_i0_exact": True,
        "shared_i0_source": "conditioning_frame0_float32.npy",
        "shared_i0_alignment": shared_i0_alignment,
        "source_tensor_sha256": _tensor_sha256(source),
        "target_tensor_sha256": _tensor_sha256(target),
        "shared_i0_tensor_sha256": _tensor_sha256(shared),
    }
    return source, target, metadata


def tensor_to_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    import torch

    torch.save(value, buffer)
    return buffer.getvalue()


class BerniniVaeEncoder:
    """The exact diffusers Wan VAE used by the Bernini-R checkpoint."""

    def __init__(self, checkpoint: Path, *, device: str) -> None:
        import torch
        from diffusers.models import AutoencoderKLWan

        self.torch = torch
        self.checkpoint = checkpoint.expanduser().resolve(strict=True)
        self.device = torch.device(device)
        vae_dir = self.checkpoint / "vae"
        _plain_file(vae_dir / "config.json", context="VAE config")
        self.identity = {
            "checkpoint_root": str(self.checkpoint),
            "vae_config_sha256": file_sha256(vae_dir / "config.json"),
        }
        revision = self.checkpoint / ".hf_revision"
        if revision.is_file() and not revision.is_symlink():
            self.identity["hf_revision"] = revision.read_text(encoding="utf-8").strip()
        self.model = AutoencoderKLWan.from_pretrained(
            str(self.checkpoint), subfolder="vae", torch_dtype=torch.float32
        ).eval().to(self.device)
        self.model.requires_grad_(False)

    def encode(self, video: Any) -> tuple[bytes, dict[str, Any]]:
        torch = self.torch
        value = video.unsqueeze(0).to(self.device, dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            parameters = self.model.encode(value).latent_dist.parameters
        parameters = parameters.detach().float().cpu().contiguous()
        if parameters.ndim != 5 or parameters.shape[2] != LATENT_FRAME_COUNT:
            raise VaeMaterializationError(
                f"VAE posterior must have latent T={LATENT_FRAME_COUNT}, got "
                f"{list(parameters.shape)}"
            )
        metadata = {
            "posterior_parameters_shape": list(parameters.shape),
            "posterior_parameters_dtype": str(parameters.dtype),
            "posterior_parameters_tensor_sha256": _tensor_sha256(parameters),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }
        del value
        return tensor_to_bytes(parameters), metadata


def _validate_raw_receipt(raw_parquet: Path, receipt_path: Path) -> dict[str, Any]:
    receipt = _load_json(receipt_path, context="raw dataset receipt")
    if receipt.get("schema_version") != RAW_RECEIPT_FORMAT:
        raise VaeMaterializationError("raw receipt schema differs")
    candidate = dict(receipt)
    declared_digest = candidate.pop("receipt_digest", None)
    if object_sha256(candidate) != declared_digest:
        raise VaeMaterializationError("raw receipt digest mismatch")
    if (
        receipt.get("preview_only") is not True
        or receipt.get("training_authorized") is not False
        or receipt.get("training_use_forbidden") is not True
        or receipt.get("production_claim_forbidden") is not True
    ):
        raise VaeMaterializationError("raw receipt authorization state differs")
    sample_count = receipt.get("sample_count")
    strict_rows = receipt.get("strict_selection_rows")
    non_strict_rows = receipt.get("non_strict_selection_rows")
    if (
        type(sample_count) is not int
        or sample_count <= 0
        or type(strict_rows) is not int
        or type(non_strict_rows) is not int
        or strict_rows < 0
        or non_strict_rows < 0
        or strict_rows + non_strict_rows != sample_count
    ):
        raise VaeMaterializationError("raw receipt cohort counts differ")
    policy = receipt.get("experimental_inclusion_policy")
    if policy not in {
        raw_builder.STRICT_INCLUSION_POLICY,
        raw_builder.NATURAL_RELEASE_INCLUSION_POLICY,
    }:
        raise VaeMaterializationError("raw receipt inclusion policy differs")
    if policy == raw_builder.NATURAL_RELEASE_INCLUSION_POLICY:
        release = receipt.get("natural_release")
        if (
            receipt.get("broader_natural_release_inclusion_acknowledged") is not True
            or not isinstance(release, Mapping)
            or release.get("release_rows") != sample_count
        ):
            raise VaeMaterializationError("raw receipt natural release binding differs")
        _sha(release.get("manifest_sha256"), context="natural release manifest")
        _sha(release.get("summary_sha256"), context="natural release summary")
    resolved = raw_parquet.expanduser().resolve(strict=True)
    if Path(str(receipt.get("parquet_path"))).expanduser().resolve(strict=True) != resolved:
        raise VaeMaterializationError("raw receipt parquet path differs")
    if file_sha256(resolved) != receipt.get("parquet_sha256"):
        raise VaeMaterializationError("raw parquet hash differs")
    return receipt


def _validate_raw_job_done(
    raw_parquet: Path, receipt_path: Path, job_done_path: Path
) -> dict[str, Any]:
    receipt = _validate_raw_receipt(raw_parquet, receipt_path)
    done = _load_json(job_done_path, context="raw job-done receipt")
    candidate = dict(done)
    declared_digest = candidate.pop("job_done_digest", None)
    if (
        done.get("schema_version") != raw_builder.JOB_DONE_FORMAT
        or done.get("complete") is not True
        or object_sha256(candidate) != declared_digest
        or done.get("sample_count") != receipt.get("sample_count")
        or done.get("strict_selection_rows")
        != receipt.get("strict_selection_rows")
        or done.get("non_strict_selection_rows")
        != receipt.get("non_strict_selection_rows")
        or done.get("experimental_inclusion_policy")
        != receipt.get("experimental_inclusion_policy")
    ):
        raise VaeMaterializationError("raw job-done contract differs")
    resolved_parquet = raw_parquet.expanduser().resolve(strict=True)
    resolved_receipt = receipt_path.expanduser().resolve(strict=True)
    preview_manifest = _plain_file(
        Path(str(receipt.get("source_preview_manifest")))
        .expanduser()
        .resolve(strict=True),
        context="source preview manifest",
    )
    if (
        done.get("raw_parquet_sha256") != file_sha256(resolved_parquet)
        or done.get("raw_receipt_sha256") != file_sha256(resolved_receipt)
        or done.get("preview_manifest_sha256")
        != receipt.get("source_preview_manifest_sha256")
        or done.get("preview_manifest_sha256") != file_sha256(preview_manifest)
    ):
        raise VaeMaterializationError("raw job-done artifact binding differs")
    if receipt.get("experimental_inclusion_policy") == raw_builder.NATURAL_RELEASE_INCLUSION_POLICY:
        release = receipt["natural_release"]
        if (
            done.get("natural_release_manifest_sha256")
            != release.get("manifest_sha256")
            or done.get("natural_release_summary_sha256")
            != release.get("summary_sha256")
            or done.get("natural_release_iid_set_sha256")
            != release.get("iid_set_sha256")
        ):
            raise VaeMaterializationError("raw job-done release binding differs")
    return receipt


def load_raw_rows(
    raw_parquet: Path, receipt_path: Path, job_done_path: Path
) -> list[dict[str, Any]]:
    receipt = _validate_raw_job_done(raw_parquet, receipt_path, job_done_path)
    try:
        import pyarrow.parquet as pq

        rows = pq.read_table(raw_parquet).to_pylist()
    except Exception as error:
        raise VaeMaterializationError(f"cannot read raw parquet: {error}") from error
    if len(rows) != receipt.get("sample_count") or not rows:
        raise VaeMaterializationError("raw parquet row count differs")
    seen: set[str] = set()
    for row in rows:
        if row.get("schema_version") != RAW_ROW_FORMAT:
            raise VaeMaterializationError("raw row schema differs")
        iid = row.get("iid")
        if type(iid) is not str or not iid or iid in seen:
            raise VaeMaterializationError(f"invalid or duplicate IID: {iid!r}")
        seen.add(iid)
        candidate = dict(row)
        digest = candidate.pop("renderer_row_digest", None)
        if object_sha256(candidate) != digest:
            raise VaeMaterializationError(f"raw row digest mismatch: {iid}")
        if (
            row.get("preview_only") is not True
            or row.get("training_authorized") is not False
            or row.get("training_use_forbidden") is not True
            or row.get("production_claim_forbidden") is not True
        ):
            raise VaeMaterializationError(f"raw row authorization differs: {iid}")
    rows.sort(key=lambda value: str(value["iid"]))
    if [row["iid"] for row in rows] != sorted(receipt["sample_ids"]):
        raise VaeMaterializationError("raw receipt sample IDs differ")
    return rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise VaeMaterializationError(f"create-only output exists: {path}")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_sample_parquet(path: Path, row: Mapping[str, Any]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise VaeMaterializationError("writing VAE rows requires pyarrow") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise VaeMaterializationError(f"create-only shard exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist([dict(row)])
        pq.write_table(table, temporary, compression="zstd", use_dictionary=False)
        if pq.read_metadata(temporary).num_rows != 1:
            raise VaeMaterializationError("persisted VAE shard row count differs")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_sample_is_valid(shard: Path, receipt_path: Path) -> bool:
    if not shard.exists() and not receipt_path.exists():
        return False
    if not shard.is_file() or shard.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        raise VaeMaterializationError("partial or unsafe existing sample output")
    receipt = _load_json(receipt_path, context="sample receipt")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    if (
        receipt.get("schema_version") != SAMPLE_RECEIPT_FORMAT
        or object_sha256(candidate) != digest
        or file_sha256(shard) != receipt.get("parquet_sha256")
    ):
        raise VaeMaterializationError("existing sample receipt differs")
    return True


def materialize_one(
    row: Mapping[str, Any],
    *,
    encoder: BerniniVaeEncoder,
    output_root: Path,
    max_pixels: int,
    stride: int,
    min_target_retention: float,
) -> dict[str, Any]:
    iid = str(row["iid"])
    shard = output_root / "shards" / f"{iid}.parquet"
    receipt_path = output_root / "receipts" / f"{iid}.json"
    if _existing_sample_is_valid(shard, receipt_path):
        return {"iid": iid, "status": "reused", "parquet": str(shard)}
    source, target, media = prepare_pair(
        row,
        max_pixels=max_pixels,
        stride=stride,
        min_target_retention=min_target_retention,
    )
    source_blob, source_vae = encoder.encode(source)
    target_blob, target_vae = encoder.encode(target)
    if source_vae["posterior_parameters_shape"] != target_vae["posterior_parameters_shape"]:
        raise VaeMaterializationError(f"source/target latent shapes differ for {iid}")
    output_row = dict(row)
    output_row["schema_version"] = MATERIALIZED_ROW_FORMAT
    output_row["video_vae_latents"] = [source_blob, target_blob]
    output_row["bernini_media_contract_json"] = canonical_json_bytes(media).decode("utf-8")
    output_row["bernini_vae_identity_json"] = canonical_json_bytes(encoder.identity).decode("utf-8")
    output_row["source_vae_metadata_json"] = canonical_json_bytes(source_vae).decode("utf-8")
    output_row["target_vae_metadata_json"] = canonical_json_bytes(target_vae).decode("utf-8")
    output_row["experimental_training_acknowledged"] = True
    output_row["production_claim_forbidden"] = True
    prior_digest = output_row.pop("renderer_row_digest")
    output_row["raw_renderer_row_digest"] = prior_digest
    output_row["materialized_row_digest"] = object_sha256(
        {
            key: value
            for key, value in output_row.items()
            if key != "video_vae_latents"
        }
        | {
            "video_vae_latents_sha256": [
                hashlib.sha256(source_blob).hexdigest(),
                hashlib.sha256(target_blob).hexdigest(),
            ]
        }
    )
    _write_sample_parquet(shard, output_row)
    receipt = {
        "schema_version": SAMPLE_RECEIPT_FORMAT,
        "complete": True,
        "iid": iid,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "experimental_inclusion_policy": row["experimental_inclusion_policy"],
        "strict_selection_gates_all_true": row[
            "strict_selection_gates_all_true"
        ],
        "selection_gates_json": row["selection_gates_json"],
        "frame_count": FRAME_COUNT,
        "fps": FPS,
        "latent_frame_count": LATENT_FRAME_COUNT,
        "bucket_hw": media["source_derived_bucket_hw"],
        "target_crop_retention": media["target_crop_retention"],
        "shared_i0_exact": True,
        "raw_renderer_row_digest": prior_digest,
        "materialized_row_digest": output_row["materialized_row_digest"],
        "source_latent_blob_sha256": hashlib.sha256(source_blob).hexdigest(),
        "target_latent_blob_sha256": hashlib.sha256(target_blob).hexdigest(),
        "posterior_parameters_shape": source_vae["posterior_parameters_shape"],
        "parquet_path": str(shard),
        "parquet_sha256": file_sha256(shard),
        "vae_identity": encoder.identity,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    _atomic_json(receipt_path, receipt)
    return {"iid": iid, "status": "written", "parquet": str(shard)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--raw-job-done", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--min-target-retention", type=float, default=DEFAULT_MIN_TARGET_RETENTION
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


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
            raise VaeMaterializationError("invalid rank/world-size")
        if not math.isfinite(args.min_target_retention) or not (
            0.0 < args.min_target_retention <= 1.0
        ):
            raise VaeMaterializationError("min-target-retention must be in (0,1]")
        rows = load_raw_rows(
            args.raw_parquet, args.raw_receipt, args.raw_job_done
        )
        if args.max_rows is not None:
            if args.max_rows <= 0:
                raise VaeMaterializationError("max-rows must be positive")
            rows = rows[: args.max_rows]
        selected = rows[rank::world_size]
        output_root = args.output_root.expanduser().absolute()
        output_root.mkdir(parents=True, exist_ok=True)
        device = args.device or f"cuda:{local_rank}"
        encoder = BerniniVaeEncoder(args.checkpoint, device=device)
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for row in selected:
            try:
                results.append(
                    materialize_one(
                        row,
                        encoder=encoder,
                        output_root=output_root,
                        max_pixels=args.max_pixels,
                        stride=args.stride,
                        min_target_retention=args.min_target_retention,
                    )
                )
            except Exception as error:
                failure = {
                    "iid": str(row.get("iid")),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                errors.append(failure)
                rejection = output_root / "rejections" / f"{row.get('iid')}.rank{rank}.json"
                _atomic_json(rejection, failure)
                if not args.continue_on_error:
                    raise
        summary = {
            "schema_version": RANK_SUMMARY_FORMAT,
            "complete": not errors,
            "rank": rank,
            "world_size": world_size,
            "selected_rows": len(selected),
            "completed_rows": len(results),
            "error_rows": len(errors),
            "results": results,
            "errors": errors,
            "preview_only": True,
            "production_claim_forbidden": True,
        }
        summary["summary_digest"] = object_sha256(summary)
        summary_path = output_root / "rank_summaries" / f"rank_{rank:04d}.json"
        if not summary_path.exists():
            _atomic_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        return 3 if errors else 0
    except VaeMaterializationError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

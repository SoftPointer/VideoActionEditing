"""Committed R7-P0 CoTracker/DINO preflight extraction.

The module has two deliberately separate responsibilities:

``extract``
    A ``torchrun`` rank reads the complete R5 pilot manifest, owns rows whose
    *input* index satisfies ``index % WORLD_SIZE == RANK``, and extracts
    source/target temporal-teacher and frozen-DINO features.  A rank writes a
    private atomic commit below ``OUTPUT/shards/rank-RRR-of-WWW``.

``finalize``
    Validate and merge exactly eight committed ranks in original manifest
    order, then compute the diagnostic-only R7-P0 quality gate.

Importing this module does not import OpenCV, PyTorch, Transformers, or
CoTracker.  Missing/broken model runtimes are process-global failures and
abort a rank.  A corrupt individual video or a conservative temporal-teacher
rejection is recorded on that source/target side and extraction continues.

R7-P0 uses the repeatedly inspected R5 pilot.  Consequently the formal
decision is always ``INSUFFICIENT`` and generation is never authorized, even
when the diagnostic preflight gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r7_temporal_teacher import (
    LazyCoTrackerAdapter,
    StableTemporalTeacher,
    TemporalTeacher,
    TemporalTeacherConfig,
    TemporalTeacherError,
    build_temporal_teacher_with_stability,
)


R7_PREFLIGHT_SCHEMA = "motive-r7-preflight-extract-v2"
R7_ROW_SCHEMA = "motive-r7-preflight-row-v2"
R7_SHARD_SUMMARY_SCHEMA = "motive-r7-preflight-shard-summary-v2"
R7_SHARD_DONE_SCHEMA = "motive-r7-preflight-shard-done-v2"
R7_FINAL_SUMMARY_SCHEMA = "motive-r7-preflight-final-summary-v2"
R7_FINAL_DONE_SCHEMA = "motive-r7-preflight-final-done-v2"
R7_P0_GATE_SCHEMA = "motive-r7-p0-diagnostic-gate-v2"
R7_EXTRACTION_PARTITION = "input-index-modulo-world-size-v1"
R7_VIDEO_SAMPLING = "uniform-32-decoded-frames-v1"
R7_DINO_SAMPLING = "uniform-6-from-uniform-32-v1"
R7_DINO_PREPROCESSING = "transformers-auto-image-processor-local-v1"
R7_DINO_POOLING = "last-hidden-state-cls-token-v1"

VIDEO_FRAMES = 32
MAX_VIDEO_SIDE = 384
DINO_FRAMES = 6
DINO_DIM = 768
PHASE_STEPS = 32
ACTOR_TRACKS = 8
TEACHER_EMBEDDING_DIM = 224
FINAL_WORLD_SIZE = 8
DEFAULT_SEED = 260108828

ARCHIVE_NAME = "features.npz"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_SIDES = ("source", "target")


class GlobalExtractionError(RuntimeError):
    """A process-global decoder/model/configuration failure."""


class PerVideoError(ValueError):
    """A failure confined to one video asset."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = str(reason)
        super().__init__(f"{self.reason}: {message}")


@dataclass(frozen=True)
class DecodedVideo:
    """Exactly 32 resized RGB frames and their source-frame provenance."""

    frames_rgb: np.ndarray
    frame_times: np.ndarray
    source_frame_indices: np.ndarray
    source_fps: float
    source_frame_count: int
    source_size: tuple[int, int]
    resized_size: tuple[int, int]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _object_digest(
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} is not a JSON object"
                )
            if line != _canonical_json(value) + "\n":
                raise ValueError(f"{path}:{line_number} is not canonical JSONL")
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no rows")
    return rows


def uniform_sample_indices(total: int, count: int) -> np.ndarray:
    """Return deterministic, endpoint-inclusive indices.

    Duplicate indices are allowed when a short video has fewer frames than
    requested.  R7 still emits a fixed-size tensor and uses strictly
    increasing normalized times for the sampled sequence.
    """

    if (
        isinstance(total, bool)
        or not isinstance(total, (int, np.integer))
        or int(total) < 1
    ):
        raise ValueError("total must be a positive integer")
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or int(count) < 1
    ):
        raise ValueError("count must be a positive integer")
    if int(count) == 1:
        return np.asarray([0], dtype=np.int64)
    values = np.linspace(0.0, float(int(total) - 1), int(count))
    return np.rint(values).astype(np.int64)


def resized_dimensions(
    height: int,
    width: int,
    maximum_side: int = MAX_VIDEO_SIDE,
) -> tuple[int, int]:
    """Aspect-preserving dimensions with neither side larger than the cap."""

    values = (height, width, maximum_side)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
        for value in values
    ):
        raise ValueError("height, width, and maximum_side must be positive")
    height_i, width_i, maximum_i = map(int, values)
    scale = min(1.0, maximum_i / float(max(height_i, width_i)))
    return (
        max(1, int(round(height_i * scale))),
        max(1, int(round(width_i * scale))),
    )


def dino_frame_offsets(
    decoded_frame_count: int = VIDEO_FRAMES,
    count: int = DINO_FRAMES,
) -> np.ndarray:
    indices = uniform_sample_indices(decoded_frame_count, count)
    if len(np.unique(indices)) != count:
        raise ValueError("DINO frame offsets must be unique")
    return indices


def difference_hash(frame_rgb: Any, *, hash_size: int = 8) -> str:
    """Small deterministic dHash used only for later visual-DSU preflight."""

    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("frame_rgb must have shape [H,W,3]")
    if hash_size < 2:
        raise ValueError("hash_size must be >=2")
    # Dependency-free nearest-neighbour resize is sufficient for a dHash.
    y = uniform_sample_indices(frame.shape[0], hash_size)
    x = uniform_sample_indices(frame.shape[1], hash_size + 1)
    sampled = frame[np.ix_(y, x)].astype(np.float64)
    gray = (
        sampled[..., 0] * 0.299
        + sampled[..., 1] * 0.587
        + sampled[..., 2] * 0.114
    )
    bits = (gray[:, 1:] > gray[:, :-1]).reshape(-1)
    encoded = 0
    for bit in bits:
        encoded = (encoded << 1) | int(bool(bit))
    width = (len(bits) + 3) // 4
    return f"{encoded:0{width}x}"


def sampled_frame_times(
    source_frame_indices: Any,
    source_fps: float,
) -> np.ndarray:
    """Return physical source timestamps or reject repeated samples."""

    indices = np.asarray(source_frame_indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(indices) < 2
    ):
        raise ValueError("source_frame_indices must be a 1D integer array")
    fps = float(source_fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    if bool((np.diff(indices.astype(np.int64)) <= 0).any()):
        raise PerVideoError(
            "duplicate_sampled_frames",
            "sampled source frame indices must be strictly increasing",
        )
    return indices.astype(np.float64) / fps


def decode_video_fixed_frames(
    path: Path,
    *,
    frame_count: int = VIDEO_FRAMES,
    maximum_side: int = MAX_VIDEO_SIDE,
) -> DecodedVideo:
    """Decode one video with OpenCV into the fixed R7 RGB contract."""

    if frame_count != VIDEO_FRAMES or maximum_side != MAX_VIDEO_SIDE:
        raise ValueError("R7 decode shape is schema-fixed at 32 frames / 384px")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PerVideoError("video_missing", str(path))
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError as error:
        raise GlobalExtractionError("OpenCV is not importable") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise PerVideoError("video_open_failed", str(path))
    try:
        total = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if total < 2:
            raise PerVideoError(
                "insufficient_video_frames",
                f"{path} reports {total} frames",
            )
        if not math.isfinite(fps) or fps <= 0.0:
            raise PerVideoError("invalid_video_fps", f"{path} reports fps={fps}")
        if height < 2 or width < 2:
            raise PerVideoError(
                "invalid_video_size",
                f"{path} reports {height}x{width}",
            )
        indices = uniform_sample_indices(total, VIDEO_FRAMES)
        try:
            times = sampled_frame_times(indices, fps)
        except PerVideoError as error:
            raise PerVideoError(
                error.reason,
                (
                    f"{path} has only {total} source frames for the "
                    f"{VIDEO_FRAMES}-frame temporal contract"
                ),
            ) from error
        output_height, output_width = resized_dimensions(
            height,
            width,
            MAX_VIDEO_SIDE,
        )
        frames: list[np.ndarray] = []
        for source_index in indices:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index)):
                raise PerVideoError(
                    "video_seek_failed",
                    f"{path} frame={int(source_index)}",
                )
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise PerVideoError(
                    "video_decode_failed",
                    f"{path} frame={int(source_index)}",
                )
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise PerVideoError(
                    "invalid_decoded_frame",
                    f"{path} frame={int(source_index)}",
                )
            if bgr.shape[:2] != (output_height, output_width):
                interpolation = (
                    cv2.INTER_AREA
                    if output_height <= bgr.shape[0]
                    and output_width <= bgr.shape[1]
                    else cv2.INTER_LINEAR
                )
                bgr = cv2.resize(
                    bgr,
                    (output_width, output_height),
                    interpolation=interpolation,
                )
            frames.append(np.ascontiguousarray(bgr[..., ::-1]))
    finally:
        capture.release()
    array = np.stack(frames).astype(np.uint8, copy=False)
    return DecodedVideo(
        frames_rgb=np.ascontiguousarray(array),
        frame_times=times,
        source_frame_indices=indices,
        source_fps=fps,
        source_frame_count=total,
        source_size=(height, width),
        resized_size=(output_height, output_width),
    )


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_digest(path),
                # Hugging Face snapshot trees commonly contain content-
                # addressed symlinks into ``blobs``.  The resolved bytes are
                # hashed above; recording the link bit avoids pretending the
                # filesystem layout was a regular file.
                "symlink": path.is_symlink(),
            }
        )
    if not rows:
        raise ValueError(f"model tree is empty: {root}")
    return rows, _object_digest(rows)


def _infer_model_revision(root: Path, inventory: Sequence[Mapping[str, Any]]) -> str:
    if _REVISION_RE.fullmatch(root.name.lower()):
        return root.name.lower()
    config_path = root / "config.json"
    if config_path.is_file():
        config = _load_json(config_path)
        for field in ("_commit_hash", "commit_hash", "revision"):
            value = str(config.get(field, "")).strip().lower()
            if _REVISION_RE.fullmatch(value):
                return value
    # A content-tree digest is immutable even for a non-HF local directory.
    return _object_digest(list(inventory))


class LazyDinoV2BaseEncoder:
    """Lazy, local-files-only DINOv2-base CLS encoder."""

    def __init__(
        self,
        *,
        model_root: Path,
        device: str,
        revision: str | None = None,
    ) -> None:
        self.model_root = model_root.expanduser().resolve(strict=True)
        if not self.model_root.is_dir():
            raise NotADirectoryError(self.model_root)
        self.device = str(device).strip()
        if not self.device:
            raise ValueError("DINO device is empty")
        inventory, tree_digest = _tree_inventory(self.model_root)
        inferred = _infer_model_revision(self.model_root, inventory)
        self.revision = str(revision or inferred).strip().lower()
        if _REVISION_RE.fullmatch(self.revision) is None:
            raise ValueError("DINO revision must be immutable hexadecimal")
        weight_rows = [
            row
            for row in inventory
            if Path(str(row["path"])).suffix
            in {".safetensors", ".bin", ".pt", ".pth"}
        ]
        if not weight_rows:
            raise ValueError("DINO model directory contains no weight file")
        self.provenance = {
            "encoder_id": "facebook/dinov2-base",
            "encoder_revision": self.revision,
            "resolved_path": str(self.model_root),
            "model_tree_sha256": tree_digest,
            "weights_sha256": _object_digest(weight_rows),
            "model_file_count": len(inventory),
            "frame_sampling_version": R7_DINO_SAMPLING,
            "preprocessing_version": R7_DINO_PREPROCESSING,
            "pooling": R7_DINO_POOLING,
            "embedding_dim": DINO_DIM,
            "dtype": "float32",
            "normalization": "l2-per-frame",
            "frozen_encoder": True,
            "local_files_only": True,
        }
        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is not None:
            return self._torch, self._processor, self._model
        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            processor_class = getattr(transformers, "AutoImageProcessor")
            model_class = getattr(transformers, "AutoModel")
        except (ImportError, AttributeError) as error:
            raise GlobalExtractionError(
                "torch/transformers with AutoImageProcessor and AutoModel "
                "are required for DINOv2"
            ) from error
        try:
            processor = processor_class.from_pretrained(
                str(self.model_root),
                local_files_only=True,
            )
            model = model_class.from_pretrained(
                str(self.model_root),
                local_files_only=True,
            )
            model.eval()
            model.requires_grad_(False)
            model.to(self.device)
        except Exception as error:
            raise GlobalExtractionError(
                f"failed to load local DINOv2 model at {self.model_root}"
            ) from error
        hidden_size = int(getattr(model.config, "hidden_size", -1))
        if hidden_size != DINO_DIM:
            raise GlobalExtractionError(
                f"DINOv2-base hidden size must be {DINO_DIM}, got {hidden_size}"
            )
        self._torch = torch
        self._processor = processor
        self._model = model
        return torch, processor, model

    def encode(self, frames_rgb: Any) -> np.ndarray:
        frames = np.asarray(frames_rgb)
        if (
            frames.ndim != 4
            or frames.shape[0] != DINO_FRAMES
            or frames.shape[-1] != 3
            or min(frames.shape[1:3]) < 2
        ):
            raise ValueError(
                f"DINO input must have shape [{DINO_FRAMES},H,W,3]"
            )
        if frames.dtype != np.uint8:
            raise ValueError("DINO RGB frames must be uint8")
        torch, processor, model = self._load()
        try:
            inputs = processor(
                images=[frame for frame in frames],
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device, non_blocking=False)
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                output = model(**inputs, return_dict=True)
                cls = output.last_hidden_state[:, 0, :].to(torch.float32)
                cls = torch.nn.functional.normalize(
                    cls,
                    p=2.0,
                    dim=1,
                    eps=1e-12,
                )
                matrix = cls.detach().cpu().numpy().astype(np.float32)
        except Exception as error:
            raise GlobalExtractionError("DINOv2 inference failed") from error
        if matrix.shape != (DINO_FRAMES, DINO_DIM):
            raise GlobalExtractionError(
                f"DINO output shape is {matrix.shape}, expected "
                f"{(DINO_FRAMES, DINO_DIM)}"
            )
        if not np.isfinite(matrix).all():
            raise GlobalExtractionError("DINO output contains non-finite values")
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
            raise GlobalExtractionError("DINO CLS vectors are not L2 normalized")
        return np.ascontiguousarray(matrix)


def _validate_rank_world(rank: int, world_size: int) -> tuple[int, int]:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("require integer 0 <= rank < world_size")
    return rank, world_size


def resolve_torchrun_coordinates(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> tuple[int, int, int]:
    """Resolve explicit values or standard ``torchrun`` environment values."""

    def resolved(explicit: int | None, variable: str, default: int) -> int:
        raw = explicit if explicit is not None else os.environ.get(variable)
        try:
            return default if raw is None else int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{variable} must be an integer") from error

    rank_value = resolved(rank, "RANK", 0)
    world_value = resolved(world_size, "WORLD_SIZE", 1)
    local_value = resolved(local_rank, "LOCAL_RANK", rank_value)
    _validate_rank_world(rank_value, world_value)
    if local_value < 0:
        raise ValueError("LOCAL_RANK must be nonnegative")
    return rank_value, world_value, local_value


def rank_directory(root: Path, rank: int, world_size: int) -> Path:
    _validate_rank_world(rank, world_size)
    return (
        root.expanduser()
        / "shards"
        / f"rank-{rank:03d}-of-{world_size:03d}"
    )


def _read_r5_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for input_index, line in enumerate(handle):
            if not line.strip():
                raise ValueError(
                    f"{path}:{input_index + 1} is unexpectedly blank"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{input_index + 1} is not an object")
            iid = row.get("iid")
            if not isinstance(iid, str) or not iid or iid.strip() != iid:
                raise ValueError(f"row {input_index} has an invalid iid")
            if iid in seen:
                raise ValueError(f"row {input_index} duplicates iid={iid}")
            seen.add(iid)
            for field in ("src_video", "tgt_video"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"row {input_index} has invalid {field}")
            pilot = row.get("r5_pilot_label")
            if not isinstance(pilot, Mapping):
                raise ValueError(f"row {input_index} lacks r5_pilot_label")
            if pilot.get("class") not in {"positive", "negative"}:
                raise ValueError(f"row {input_index} has invalid pilot class")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _safe_video_path(data_root: Path, value: str) -> Path:
    root = data_root.expanduser().resolve(strict=True)
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"video path escapes data root: {value!r}") from error
    return path


def _side_seed(iid: str, side: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}\0{iid}\0{side}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _side_audit_seed(iid: str, side: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}\0{iid}\0{side}\0independent-audit-v1".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:4], "little", signed=False)
    screening = _side_seed(iid, side, base_seed)
    # The collision probability is tiny, but independence is an artifact
    # contract rather than a probabilistic hope.
    return value if value != screening else value ^ 0x80000000


def _empty_arrays(row_count: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "input_indices": np.zeros(row_count, dtype=np.int64),
        "positive": np.zeros(row_count, dtype=np.bool_),
    }
    for side in _SIDES:
        arrays.update(
            {
                f"{side}_usable": np.zeros(row_count, dtype=np.bool_),
                f"{side}_base_valid": np.zeros(row_count, dtype=np.bool_),
                f"{side}_dino_valid": np.zeros(row_count, dtype=np.bool_),
                f"{side}_audit_available": np.zeros(
                    row_count, dtype=np.bool_
                ),
                f"{side}_audit_pass": np.zeros(row_count, dtype=np.bool_),
                f"{side}_camera_crossfit_valid": np.zeros(
                    row_count, dtype=np.bool_
                ),
                f"{side}_teacher_embedding": np.zeros(
                    (row_count, TEACHER_EMBEDDING_DIM), dtype=np.float32
                ),
                f"{side}_actor_trajectory": np.zeros(
                    (row_count, PHASE_STEPS, 2), dtype=np.float32
                ),
                f"{side}_actor_velocity": np.zeros(
                    (row_count, PHASE_STEPS, 2), dtype=np.float32
                ),
                f"{side}_actor_acceleration": np.zeros(
                    (row_count, PHASE_STEPS, 2), dtype=np.float32
                ),
                f"{side}_actor_track_trajectories": np.zeros(
                    (row_count, ACTOR_TRACKS, PHASE_STEPS, 2),
                    dtype=np.float32,
                ),
                f"{side}_actor_track_mask": np.zeros(
                    (row_count, ACTOR_TRACKS), dtype=np.bool_
                ),
                f"{side}_camera_trajectory": np.zeros(
                    (row_count, PHASE_STEPS, 4), dtype=np.float32
                ),
                f"{side}_phase_visibility": np.zeros(
                    (row_count, PHASE_STEPS), dtype=np.float32
                ),
                f"{side}_phase_uncertainty": np.zeros(
                    (row_count, PHASE_STEPS), dtype=np.float32
                ),
                f"{side}_phase_energy": np.zeros(
                    (row_count, PHASE_STEPS), dtype=np.float32
                ),
                f"{side}_dino_cls": np.zeros(
                    (row_count, DINO_FRAMES, DINO_DIM), dtype=np.float32
                ),
                f"{side}_perceptual_hashes": np.full(
                    (row_count, DINO_FRAMES), "", dtype="<U16"
                ),
            }
        )
        for metric in (
            "event_normalized_start",
            "event_normalized_end",
            "event_duration",
            "mean_visibility",
            "background_residual_reduction",
            "camera_explained_ratio",
            "camera_inlier_fraction",
            "camera_crossfit_raw_median",
            "camera_crossfit_residual_median",
            "camera_crossfit_residual_reduction",
            "stability_event_iou",
            "stability_embedding_cosine",
            "stability_duration_relative_error",
            "stability_embedding_norm_relative_error",
            "stability_trajectory_rmse",
            "audit_event_iou",
            "audit_embedding_cosine",
            "audit_duration_relative_error",
            "audit_embedding_norm_relative_error",
            "audit_trajectory_rmse",
        ):
            arrays[f"{side}_{metric}"] = np.zeros(
                row_count, dtype=np.float32
            )
        # Missing audit comparisons are failures.  Similarities stay at zero;
        # error metrics use one so that a missing audit can never improve a
        # lower-is-better median.
        for metric in (
            "audit_duration_relative_error",
            "audit_embedding_norm_relative_error",
            "audit_trajectory_rmse",
        ):
            arrays[f"{side}_{metric}"].fill(1.0)
    return arrays


def _fill_base(
    arrays: dict[str, np.ndarray],
    *,
    side: str,
    index: int,
    teacher: TemporalTeacher,
) -> None:
    embedding = teacher.embedding()
    if embedding.shape != (TEACHER_EMBEDDING_DIM,):
        raise GlobalExtractionError(
            f"teacher embedding schema changed to {embedding.shape}"
        )
    arrays[f"{side}_base_valid"][index] = True
    arrays[f"{side}_camera_crossfit_valid"][index] = (
        teacher.camera_crossfit_valid
    )
    arrays[f"{side}_teacher_embedding"][index] = embedding
    for field in (
        "actor_trajectory",
        "actor_velocity",
        "actor_acceleration",
        "actor_track_trajectories",
        "actor_track_mask",
        "camera_trajectory",
        "phase_visibility",
        "phase_uncertainty",
        "phase_energy",
    ):
        arrays[f"{side}_{field}"][index] = getattr(teacher, field)
    event = teacher.event_window
    scalar_values = {
        "event_normalized_start": event.normalized_start,
        "event_normalized_end": event.normalized_end,
        "event_duration": teacher.event_duration,
        "mean_visibility": teacher.mean_visibility,
        "background_residual_reduction":
            teacher.background_residual_reduction,
        "camera_explained_ratio": teacher.camera_explained_ratio,
        "camera_inlier_fraction": teacher.camera_inlier_fraction,
        "camera_crossfit_raw_median":
            teacher.camera_crossfit_raw_median,
        "camera_crossfit_residual_median":
            teacher.camera_crossfit_residual_median,
        "camera_crossfit_residual_reduction":
            teacher.camera_crossfit_residual_reduction,
    }
    for name, value in scalar_values.items():
        arrays[f"{side}_{name}"][index] = float(value)


def _extract_side(
    *,
    path: Path,
    side: str,
    iid: str,
    array_index: int,
    arrays: dict[str, np.ndarray],
    tracker: LazyCoTrackerAdapter,
    dino: LazyDinoV2BaseEncoder,
    teacher_config: TemporalTeacherConfig,
    base_seed: int,
) -> dict[str, Any]:
    video_digest = _file_digest(path) if path.is_file() else None
    try:
        decoded = decode_video_fixed_frames(path)
    except PerVideoError as error:
        return {
            "status": "failed",
            "usable": False,
            "failure_stage": "decode",
            "failure_reason": error.reason,
            "failure_message": str(error),
            "resolved_path": str(path),
            "video_sha256": video_digest,
        }
    offsets = dino_frame_offsets()
    selected = decoded.frames_rgb[offsets]
    # DINO/CoTracker runtime failures are global by design; do not convert
    # them into hundreds of apparently independent bad-video records.
    dino_matrix = dino.encode(selected)
    arrays[f"{side}_dino_cls"][array_index] = dino_matrix
    arrays[f"{side}_dino_valid"][array_index] = True
    hashes = [difference_hash(frame) for frame in selected]
    arrays[f"{side}_perceptual_hashes"][array_index] = hashes
    decode_record = {
        "sampling_version": R7_VIDEO_SAMPLING,
        "decoded_frames": VIDEO_FRAMES,
        "source_frame_indices": decoded.source_frame_indices.tolist(),
        "source_fps": decoded.source_fps,
        "source_frame_count": decoded.source_frame_count,
        "source_size": list(decoded.source_size),
        "resized_size": list(decoded.resized_size),
        "dino_frame_offsets": offsets.tolist(),
        "dino_source_frame_indices":
            decoded.source_frame_indices[offsets].tolist(),
        "perceptual_hashes": hashes,
    }
    screening_seed = _side_seed(iid, side, base_seed)
    audit_seed = _side_audit_seed(iid, side, base_seed)
    try:
        observations = tracker.track(
            decoded.frames_rgb,
            frame_times=decoded.frame_times,
        )
        stable = build_temporal_teacher_with_stability(
            observations.tracks,
            observations.visibility,
            observations.frame_times,
            observations.frame_size,
            seed=screening_seed,
            audit_seed=audit_seed,
            config=teacher_config,
        )
    except TemporalTeacherError as error:
        return {
            "status": "failed",
            "usable": False,
            "failure_stage": "temporal_teacher",
            "failure_reason": error.reason,
            "failure_message": str(error),
            "resolved_path": str(path),
            "video_sha256": video_digest,
            "decode": decode_record,
            "dino_valid": True,
        }
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        raise GlobalExtractionError(
            f"CoTracker runtime failed on {path}"
        ) from error
    _fill_base(
        arrays,
        side=side,
        index=array_index,
        teacher=stable.base,
    )
    if stable.stability is not None:
        stability = stable.stability
        arrays[f"{side}_stability_event_iou"][array_index] = (
            stability.event_window_iou
        )
        arrays[f"{side}_stability_embedding_cosine"][array_index] = (
            stability.embedding_cosine
        )
        arrays[
            f"{side}_stability_duration_relative_error"
        ][array_index] = stability.event_duration_relative_error
        arrays[
            f"{side}_stability_embedding_norm_relative_error"
        ][array_index] = stability.embedding_norm_relative_error
        arrays[f"{side}_stability_trajectory_rmse"][array_index] = (
            stability.trajectory_rmse
        )
        stability_record: dict[str, Any] | None = stability.to_dict()
    else:
        stability_record = None
    arrays[f"{side}_audit_available"][array_index] = stable.audit_available
    arrays[f"{side}_audit_pass"][array_index] = stable.audit_passed
    if stable.audit_stability is not None:
        audit = stable.audit_stability
        arrays[f"{side}_audit_event_iou"][array_index] = (
            audit.event_window_iou
        )
        arrays[f"{side}_audit_embedding_cosine"][array_index] = (
            audit.embedding_cosine
        )
        arrays[f"{side}_audit_duration_relative_error"][array_index] = (
            audit.event_duration_relative_error
        )
        arrays[
            f"{side}_audit_embedding_norm_relative_error"
        ][array_index] = audit.embedding_norm_relative_error
        arrays[f"{side}_audit_trajectory_rmse"][array_index] = (
            audit.trajectory_rmse
        )
        audit_record: dict[str, Any] | None = audit.to_dict()
    else:
        audit_record = None
    arrays[f"{side}_usable"][array_index] = stable.diagnostic_ready
    event = stable.base.event_window
    record = {
        "status": "usable" if stable.diagnostic_ready else "failed",
        "usable": stable.diagnostic_ready,
        "failure_stage": (
            None if stable.diagnostic_ready else "stability_gate"
        ),
        "failure_reason": stable.failure_reason,
        "failure_message": None,
        "resolved_path": str(path),
        "video_sha256": video_digest,
        "decode": decode_record,
        "dino_valid": True,
        "tracker": {
            "backend": observations.backend,
            "provenance": dict(observations.provenance),
            "tracks": int(observations.tracks.shape[1]),
        },
        "temporal_teacher": {
            "event_window": event.to_dict(),
            "active_tracks": len(stable.base.active_track_indices),
            "mean_visibility": stable.base.mean_visibility,
            "background_residual_reduction":
                stable.base.background_residual_reduction,
            "camera_explained_ratio":
                stable.base.camera_explained_ratio,
            "camera_inlier_fraction":
                stable.base.camera_inlier_fraction,
            "camera_crossfit": {
                "valid": stable.base.camera_crossfit_valid,
                "raw_normalized_motion_median":
                    stable.base.camera_crossfit_raw_median,
                "residual_normalized_motion_median":
                    stable.base.camera_crossfit_residual_median,
                "residual_reduction":
                    stable.base.camera_crossfit_residual_reduction,
                "fit_eval_track_disjoint": True,
                "folds": 2,
            },
            "screening_stability": {
                "seed": screening_seed,
                "available": stability_record is not None,
                "passed": stable.diagnostic_ready,
                "failure_reason": stable.failure_reason,
                "metrics": stability_record,
            },
            "independent_audit_stability": {
                "seed": audit_seed,
                "available": stable.audit_available,
                "passed": stable.audit_passed,
                "failure_reason": stable.audit_failure_reason,
                "metrics": audit_record,
            },
            "stability_limitation": (
                "Both perturbations reuse one CoTracker output and measure "
                "downstream teacher robustness; they do not measure visual "
                "re-tracking stability."
            ),
        },
    }
    return record


def _implementation_provenance() -> dict[str, str]:
    module = Path(__file__).resolve()
    teacher = module.with_name("r7_temporal_teacher.py")
    return {
        module.name: _file_digest(module),
        teacher.name: _file_digest(teacher),
    }


def build_extraction_contract(
    *,
    input_manifest: Path,
    data_root: Path,
    rank: int,
    world_size: int,
    device: str,
    tracker_checkpoint: Path,
    tracker_checkpoint_sha256: str,
    tracker_grid_size: int,
    dino_provenance: Mapping[str, Any],
    seed: int,
    teacher_config: TemporalTeacherConfig,
) -> dict[str, Any]:
    _validate_rank_world(rank, world_size)
    teacher_config.validate()
    return {
        "schema_version": R7_PREFLIGHT_SCHEMA,
        "input_manifest": str(input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _file_digest(input_manifest),
        "data_root": str(data_root.resolve(strict=True)),
        "rank": rank,
        "world_size": world_size,
        "partition": R7_EXTRACTION_PARTITION,
        "device": str(device),
        "video_sampling": {
            "version": R7_VIDEO_SAMPLING,
            "frames": VIDEO_FRAMES,
            "maximum_side": MAX_VIDEO_SIDE,
        },
        "tracker": {
            "checkpoint": str(tracker_checkpoint.resolve(strict=True)),
            "checkpoint_sha256": tracker_checkpoint_sha256,
            "grid_size": tracker_grid_size,
            "backward_tracking": False,
            "query_frame": 0,
        },
        "temporal_teacher_config": asdict(teacher_config),
        "dino": dict(dino_provenance),
        "seed": seed,
        "perturbation_seed_policy": {
            "screening": "sha256(base_seed,iid,side)-u32-v1",
            "independent_audit":
                "sha256(base_seed,iid,side,independent-audit-v1)-u32-v1",
            "audit_changes_usable": False,
        },
        "stability_limitation": (
            "Screening and audit reuse one CoTracker output; neither "
            "measures visual re-tracking stability."
        ),
        "implementation": _implementation_provenance(),
    }


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "archive": directory / ARCHIVE_NAME,
        "manifest": directory / MANIFEST_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
) -> None:
    expected = _empty_arrays(rows)
    if set(arrays) != set(expected):
        missing = sorted(set(expected) - set(arrays))
        extra = sorted(set(arrays) - set(expected))
        raise ValueError(f"archive array names differ; missing={missing}, extra={extra}")
    for name, template in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"{name} shape/dtype differs: got {value.shape}/{value.dtype}, "
                f"expected {template.shape}/{template.dtype}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    indices = np.asarray(arrays["input_indices"], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("archive input_indices contain duplicates")
    for side in _SIDES:
        usable = np.asarray(arrays[f"{side}_usable"], dtype=bool)
        base = np.asarray(arrays[f"{side}_base_valid"], dtype=bool)
        dino = np.asarray(arrays[f"{side}_dino_valid"], dtype=bool)
        audit_available = np.asarray(
            arrays[f"{side}_audit_available"], dtype=bool
        )
        audit_pass = np.asarray(arrays[f"{side}_audit_pass"], dtype=bool)
        crossfit = np.asarray(
            arrays[f"{side}_camera_crossfit_valid"], dtype=bool
        )
        if bool((usable & ~(base & dino)).any()):
            raise ValueError(f"{side} usable does not imply base+DINO validity")
        if bool((audit_available & ~base).any()):
            raise ValueError(f"{side} audit availability does not imply base")
        if bool((audit_pass & ~audit_available).any()):
            raise ValueError(f"{side} audit pass does not imply availability")
        if bool((crossfit & ~base).any()):
            raise ValueError(f"{side} camera cross-fit does not imply base")
        valid_dino = np.asarray(arrays[f"{side}_dino_cls"])[dino]
        if len(valid_dino):
            norms = np.linalg.norm(valid_dino.astype(np.float64), axis=2)
            if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
                raise ValueError(f"{side} DINO features are not L2 normalized")
        valid_teacher = np.asarray(
            arrays[f"{side}_teacher_embedding"]
        )[base]
        if len(valid_teacher) and bool(
            (np.linalg.norm(valid_teacher.astype(np.float64), axis=1) <= 1e-12).any()
        ):
            raise ValueError(f"{side} valid teacher contains a zero embedding")


def _commit_shard(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    input_rows: int,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(directory)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing/partial R7 shard: "
            + ", ".join(existing)
        )
    _validate_array_contract(arrays, rows=len(rows))
    canonical_rows = [dict(row) for row in rows]
    indices = [int(row["input_index"]) for row in canonical_rows]
    if indices != np.asarray(arrays["input_indices"]).tolist():
        raise ValueError("manifest/archive input index order differs")
    rank = int(contract["rank"])
    world_size = int(contract["world_size"])
    expected = [
        index
        for index in range(input_rows)
        if index % world_size == rank
    ]
    if indices != expected:
        raise ValueError("shard rows do not exactly cover their modulo partition")
    _atomic_npz(
        paths["archive"],
        {name: np.asarray(value) for name, value in arrays.items()},
    )
    _atomic_jsonl(paths["manifest"], canonical_rows)
    failures: dict[str, int] = {}
    for row in canonical_rows:
        for side in _SIDES:
            result = row[side]
            if not result["usable"]:
                reason = str(result.get("failure_reason") or "unknown")
                key = f"{side}:{reason}"
                failures[key] = failures.get(key, 0) + 1
    summary = {
        "schema_version": R7_SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "input_rows": input_rows,
        "rank": rank,
        "world_size": world_size,
        "partition": R7_EXTRACTION_PARTITION,
        "positive_rows": int(np.asarray(arrays["positive"]).sum()),
        "source_usable": int(
            np.asarray(arrays["source_usable"], dtype=bool).sum()
        ),
        "target_usable": int(
            np.asarray(arrays["target_usable"], dtype=bool).sum()
        ),
        "paired_usable": int(
            (
                np.asarray(arrays["source_usable"], dtype=bool)
                & np.asarray(arrays["target_usable"], dtype=bool)
            ).sum()
        ),
        "target_audit_available": int(
            np.asarray(arrays["target_audit_available"], dtype=bool).sum()
        ),
        "target_camera_crossfit_valid": int(
            np.asarray(
                arrays["target_camera_crossfit_valid"], dtype=bool
            ).sum()
        ),
        "stability_limitation": (
            "Screening and audit perturb the same tracked coordinates "
            "downstream; visual re-tracking stability is not measured."
        ),
        "failures": dict(sorted(failures.items())),
        "contract": dict(contract),
        "contract_sha256": _object_digest(dict(contract)),
        "array_contract": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
                "sha256": _array_digest(np.asarray(value)),
            }
            for name, value in sorted(arrays.items())
        },
        "archive_sha256": _file_digest(paths["archive"]),
        "manifest_sha256": _file_digest(paths["manifest"]),
    }
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R7_SHARD_DONE_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "rank": rank,
        "world_size": world_size,
        "contract_sha256": summary["contract_sha256"],
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
            for name, path in paths.items()
            if name != "done"
        },
    }
    _atomic_json(paths["done"], done)
    return done


def validate_shard(
    directory: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    input_manifest: Path | None = None,
    rehash_videos: bool = False,
) -> dict[str, Any]:
    """Strictly validate a committed rank and return rows/arrays/metadata."""

    paths = _artifact_paths(directory)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != R7_SHARD_DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("invalid R7 shard done marker")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "archive",
        "manifest",
        "summary",
    }:
        raise ValueError("R7 shard done artifact registry differs")
    for name in artifacts:
        record = artifacts[name]
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != paths[name].name
            or record.get("sha256") != _file_digest(paths[name])
        ):
            raise ValueError(f"R7 shard {name} digest mismatch")
    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != R7_SHARD_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("invalid R7 shard summary")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("R7 shard summary has no contract")
    contract_digest = _object_digest(dict(contract))
    if (
        summary.get("contract_sha256") != contract_digest
        or done.get("contract_sha256") != contract_digest
    ):
        raise ValueError("R7 shard contract digest mismatch")
    if expected_contract is not None and dict(contract) != dict(expected_contract):
        raise ValueError("resume arguments differ from committed R7 shard")
    rows = _load_jsonl(paths["manifest"], allow_empty=True)
    with np.load(paths["archive"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_array_contract(arrays, rows=len(rows))
    indices = [int(row.get("input_index", -1)) for row in rows]
    if indices != np.asarray(arrays["input_indices"]).tolist():
        raise ValueError("R7 shard manifest/archive input order differs")
    if summary.get("rows") != len(rows) or done.get("rows") != len(rows):
        raise ValueError("R7 shard row counts differ")
    for name, value in arrays.items():
        record = summary.get("array_contract", {}).get(name)
        if not isinstance(record, Mapping) or record != {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_digest(value),
        }:
            raise ValueError(f"R7 shard array contract differs for {name}")
    rank = int(contract["rank"])
    world_size = int(contract["world_size"])
    _validate_rank_world(rank, world_size)
    if done.get("rank") != rank or done.get("world_size") != world_size:
        raise ValueError("R7 shard rank/world metadata differs")
    if any(index % world_size != rank for index in indices):
        raise ValueError("R7 shard contains an index from another rank")
    if input_manifest is not None:
        manifest_path = input_manifest.expanduser().resolve(strict=True)
        if contract.get("input_manifest_sha256") != _file_digest(manifest_path):
            raise ValueError("R7 shard input manifest SHA-256 differs")
        input_rows = _read_r5_manifest(manifest_path)
        expected_indices = [
            index
            for index in range(len(input_rows))
            if index % world_size == rank
        ]
        if indices != expected_indices:
            raise ValueError("R7 shard modulo coverage is incomplete")
        for output, index in zip(rows, indices):
            if output.get("input_row_sha256") != _object_digest(
                input_rows[index]
            ):
                raise ValueError(f"R7 row {index} input digest differs")
    if rehash_videos:
        for row in rows:
            for side in _SIDES:
                result = row[side]
                digest = result.get("video_sha256")
                if digest is not None:
                    path = Path(str(result["resolved_path"]))
                    if not path.is_file() or _file_digest(path) != digest:
                        raise ValueError(
                            f"R7 {side} video bytes changed for {row['iid']}"
                        )
    return {
        "done": done,
        "summary": summary,
        "contract": dict(contract),
        "rows": rows,
        "arrays": arrays,
    }


def extract_rank(
    *,
    input_manifest: Path,
    data_root: Path,
    output_root: Path,
    tracker_checkpoint: Path,
    dino_model_root: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    tracker_grid_size: int = 10,
    seed: int = DEFAULT_SEED,
    dino_revision: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one rank of the real R7-P0 extraction."""

    _validate_rank_world(rank, world_size)
    if tracker_grid_size < 2:
        raise ValueError("tracker_grid_size must be >=2")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    root = data_root.expanduser().resolve(strict=True)
    checkpoint = tracker_checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = f"cuda:{local_rank}"
    dino = LazyDinoV2BaseEncoder(
        model_root=dino_model_root,
        device=device,
        revision=dino_revision,
    )
    teacher_config = TemporalTeacherConfig()
    contract = build_extraction_contract(
        input_manifest=manifest_path,
        data_root=root,
        rank=rank,
        world_size=world_size,
        device=device,
        tracker_checkpoint=checkpoint,
        tracker_checkpoint_sha256=_file_digest(checkpoint),
        tracker_grid_size=tracker_grid_size,
        dino_provenance=dino.provenance,
        seed=seed,
        teacher_config=teacher_config,
    )
    directory = rank_directory(output_root, rank, world_size)
    if (directory / DONE_NAME).exists():
        if not resume:
            raise FileExistsError(directory / DONE_NAME)
        return validate_shard(
            directory,
            expected_contract=contract,
            input_manifest=manifest_path,
            rehash_videos=True,
        )["done"]
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"partial R7 shard cannot be resumed: {directory}"
        )
    input_rows = _read_r5_manifest(manifest_path)
    selected = [
        (index, row)
        for index, row in enumerate(input_rows)
        if index % world_size == rank
    ]
    arrays = _empty_arrays(len(selected))
    tracker = LazyCoTrackerAdapter(
        checkpoint=checkpoint,
        device=device,
        grid_size=tracker_grid_size,
        backward_tracking=False,
    )
    output_rows: list[dict[str, Any]] = []
    for array_index, (input_index, row) in enumerate(selected):
        iid = str(row["iid"])
        positive = row["r5_pilot_label"]["class"] == "positive"
        arrays["input_indices"][array_index] = input_index
        arrays["positive"][array_index] = positive
        results: dict[str, dict[str, Any]] = {}
        for side, field in (("source", "src_video"), ("target", "tgt_video")):
            path = _safe_video_path(root, str(row[field]))
            results[side] = _extract_side(
                path=path,
                side=side,
                iid=iid,
                array_index=array_index,
                arrays=arrays,
                tracker=tracker,
                dino=dino,
                teacher_config=teacher_config,
                base_seed=seed,
            )
        output_rows.append(
            {
                "schema_version": R7_ROW_SCHEMA,
                "input_index": input_index,
                "shard_array_index": array_index,
                "shard_rank": rank,
                "world_size": world_size,
                "iid": iid,
                "input_row_sha256": _object_digest(row),
                "input_digest": row.get("input_digest"),
                "prompt": row.get("prompt"),
                "label_type": row["r5_pilot_label"]["class"],
                "negative_type": row["r5_pilot_label"].get(
                    "negative_type"
                ),
                "positive": positive,
                "action_signature": row["r5_pilot_label"].get(
                    "action_signature"
                ),
                "source": results["source"],
                "target": results["target"],
                "paired_usable": bool(
                    results["source"]["usable"]
                    and results["target"]["usable"]
                ),
            }
        )
    return _commit_shard(
        directory=directory,
        rows=output_rows,
        arrays=arrays,
        contract=contract,
        input_rows=len(input_rows),
    )


def _safe_median(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if len(finite) == 0 else float(np.median(finite))


def _negative_type(row: Mapping[str, Any]) -> str | None:
    direct = row.get("negative_type")
    if isinstance(direct, str) and direct:
        return direct
    pilot = row.get("r5_pilot_label")
    if isinstance(pilot, Mapping):
        nested = pilot.get("negative_type")
        if isinstance(nested, str) and nested:
            return nested
    signature = row.get("action_signature")
    if isinstance(signature, str) and signature.startswith("negative:"):
        return signature.split(":", 1)[1]
    label = row.get("label_type")
    if label in {"static", "endpoint_only", "instruction_mismatch"}:
        return str(label)
    return None


def compute_p0_gate(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compute the fixed R7-P0 gate without granting formal readiness."""

    _validate_array_contract(arrays, rows=len(rows))
    positive = np.asarray(arrays["positive"], dtype=bool)
    target_base = np.asarray(arrays["target_base_valid"], dtype=bool)
    target_usable = np.asarray(arrays["target_usable"], dtype=bool)
    source_usable = np.asarray(arrays["source_usable"], dtype=bool)
    audit_available = np.asarray(
        arrays["target_audit_available"], dtype=bool
    )
    audit_pass = np.asarray(arrays["target_audit_pass"], dtype=bool)
    crossfit_valid = np.asarray(
        arrays["target_camera_crossfit_valid"], dtype=bool
    )
    crossfit_raw = np.asarray(
        arrays["target_camera_crossfit_raw_median"], dtype=np.float64
    )
    positive_count = int(positive.sum())
    positive_base = positive & target_base
    positive_target = positive & target_usable
    positive_paired = positive & target_usable & source_usable
    audit_mask = positive_base
    target_usable_count = int(positive_target.sum())
    paired_count = int(positive_paired.sum())
    fraction = (
        target_usable_count / positive_count if positive_count else 0.0
    )
    # Audit stability is intentionally computed over every positive target
    # with a valid base teacher, never over the screening-selected usable
    # subset.  Failed independent audits contribute zero similarity and a
    # unit error, so selection cannot make the medians look better.
    audit_iou = np.where(
        audit_available,
        np.asarray(arrays["target_audit_event_iou"], dtype=np.float64),
        0.0,
    )
    audit_cosine = np.where(
        audit_available,
        np.asarray(
            arrays["target_audit_embedding_cosine"], dtype=np.float64
        ),
        0.0,
    )
    audit_duration_error = np.where(
        audit_available,
        np.asarray(
            arrays["target_audit_duration_relative_error"],
            dtype=np.float64,
        ),
        1.0,
    )
    audit_norm_error = np.where(
        audit_available,
        np.asarray(
            arrays["target_audit_embedding_norm_relative_error"],
            dtype=np.float64,
        ),
        1.0,
    )
    audit_trajectory_rmse = np.where(
        audit_available,
        np.asarray(
            arrays["target_audit_trajectory_rmse"], dtype=np.float64
        ),
        1.0,
    )
    camera_mask = (
        positive_base
        & crossfit_valid
        & (crossfit_raw >= 0.002)
    )
    negative_types = np.asarray(
        [_negative_type(row) for row in rows],
        dtype=object,
    )
    no_action_negative = (
        ~positive
        & np.isin(negative_types, ("static", "endpoint_only"))
    )
    instruction_mismatch = (
        ~positive & (negative_types == "instruction_mismatch")
    )
    no_action_count = int(no_action_negative.sum())
    false_event_count = int((no_action_negative & target_usable).sum())
    false_event_fraction = (
        false_event_count / no_action_count if no_action_count else None
    )
    medians = {
        "audit_event_window_iou": _safe_median(
            audit_iou[audit_mask]
        ),
        "audit_embedding_cosine": _safe_median(
            audit_cosine[audit_mask]
        ),
        "audit_duration_relative_error": _safe_median(
            audit_duration_error[audit_mask]
        ),
        "audit_embedding_norm_relative_error": _safe_median(
            audit_norm_error[audit_mask]
        ),
        "audit_trajectory_rmse": _safe_median(
            audit_trajectory_rmse[audit_mask]
        ),
        "camera_crossfit_residual_reduction": _safe_median(
            np.asarray(
                arrays["target_camera_crossfit_residual_reduction"]
            )[camera_mask]
        ),
    }
    criteria = {
        "positive_target_usable_fraction": {
            "value": fraction,
            "threshold": 0.85,
            "operator": ">=",
            "passed": fraction >= 0.85,
        },
        "paired_usable_positive_events": {
            "value": paired_count,
            "threshold": 80,
            "operator": ">=",
            "passed": paired_count >= 80,
        },
        "median_independent_audit_event_window_iou": {
            "value": medians["audit_event_window_iou"],
            "threshold": 0.70,
            "operator": ">=",
            "passed": (
                medians["audit_event_window_iou"] is not None
                and medians["audit_event_window_iou"] >= 0.70
            ),
        },
        "median_independent_audit_embedding_cosine": {
            "value": medians["audit_embedding_cosine"],
            "threshold": 0.85,
            "operator": ">=",
            "passed": (
                medians["audit_embedding_cosine"] is not None
                and medians["audit_embedding_cosine"] >= 0.85
            ),
        },
        "median_independent_audit_duration_relative_error": {
            "value": medians["audit_duration_relative_error"],
            "threshold": 0.10,
            "operator": "<=",
            "passed": (
                medians["audit_duration_relative_error"] is not None
                and medians["audit_duration_relative_error"] <= 0.10
            ),
        },
        "median_independent_audit_embedding_norm_relative_error": {
            "value": medians["audit_embedding_norm_relative_error"],
            "threshold": 0.10,
            "operator": "<=",
            "passed": (
                medians["audit_embedding_norm_relative_error"] is not None
                and medians["audit_embedding_norm_relative_error"] <= 0.10
            ),
        },
        "median_independent_audit_trajectory_rmse": {
            "value": medians["audit_trajectory_rmse"],
            "threshold": 0.01,
            "operator": "<=",
            "units": "normalized_frame_coordinates",
            "passed": (
                medians["audit_trajectory_rmse"] is not None
                and medians["audit_trajectory_rmse"] <= 0.01
            ),
        },
        "camera_crossfit_motion_eligible_samples": {
            "value": int(camera_mask.sum()),
            "threshold": 10,
            "operator": ">=",
            "raw_normalized_motion_minimum": 0.002,
            "passed": int(camera_mask.sum()) >= 10,
        },
        "median_camera_crossfit_residual_reduction": {
            "value": medians["camera_crossfit_residual_reduction"],
            "threshold": 0.30,
            "operator": ">=",
            "passed": (
                medians["camera_crossfit_residual_reduction"] is not None
                and medians["camera_crossfit_residual_reduction"] >= 0.30
            ),
        },
        "no_action_negative_samples": {
            "value": no_action_count,
            "threshold": 10,
            "operator": ">=",
            "included_types": ["static", "endpoint_only"],
            "excluded_types": ["instruction_mismatch"],
            "passed": no_action_count >= 10,
        },
        "no_action_negative_false_event_fraction": {
            "value": false_event_fraction,
            "threshold": 0.20,
            "operator": "<=",
            "numerator": false_event_count,
            "denominator": no_action_count,
            "passed": (
                false_event_fraction is not None
                and false_event_fraction <= 0.20
            ),
        },
    }
    passed = all(bool(value["passed"]) for value in criteria.values())
    return {
        "schema_version": R7_P0_GATE_SCHEMA,
        "diagnostic_status": (
            "DIAGNOSTIC_FEATURE_READY"
            if passed
            else "DIAGNOSTIC_FEATURE_NOT_READY"
        ),
        "diagnostic_gate_passed": passed,
        "counts": {
            "rows": len(rows),
            "positive_rows": positive_count,
            "positive_target_base_valid": int(positive_base.sum()),
            "positive_target_usable": target_usable_count,
            "positive_paired_usable": paired_count,
            "positive_target_audit_eligible": int(audit_mask.sum()),
            "positive_target_audit_available": int(
                (audit_mask & audit_available).sum()
            ),
            "positive_target_audit_failed": int(
                (audit_mask & ~audit_available).sum()
            ),
            "positive_target_audit_passed": int(
                (audit_mask & audit_pass).sum()
            ),
            "positive_target_camera_crossfit_valid": int(
                (positive_base & crossfit_valid).sum()
            ),
            "positive_target_camera_crossfit_motion_eligible": int(
                camera_mask.sum()
            ),
            "negative_no_action_rows": no_action_count,
            "negative_no_action_false_events": false_event_count,
            "negative_instruction_mismatch_excluded": int(
                instruction_mismatch.sum()
            ),
        },
        "criteria": criteria,
        "stability_audit_scope": (
            "all positive target_base_valid rows; screening usability is "
            "not part of the mask; unavailable audits contribute zero "
            "similarity and unit error"
        ),
        "stability_limitation": (
            "Screening and independent audit are downstream perturbations "
            "of one CoTracker result, not visual re-tracking tests."
        ),
        "formal_status": "INSUFFICIENT",
        "formal_reason": (
            "R7-P0 reuses the inspected R5 pseudo pilot and has no human "
            "event boundaries, actor masks, or fresh locked split"
        ),
        "production_decision": False,
        "generation_authorized": False,
    }


def _merged_directory(output_root: Path) -> Path:
    return output_root.expanduser() / "final"


def _commit_final(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    input_manifest: Path,
    shard_done_sha256: Sequence[str],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(directory)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing/partial R7 final artifact: "
            + ", ".join(existing)
        )
    _validate_array_contract(arrays, rows=len(rows))
    _atomic_npz(paths["archive"], arrays)
    _atomic_jsonl(paths["manifest"], rows)
    summary = {
        "schema_version": R7_FINAL_SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "world_size": FINAL_WORLD_SIZE,
        "input_manifest": str(input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _file_digest(input_manifest),
        "partition": R7_EXTRACTION_PARTITION,
        "shard_done_sha256": list(shard_done_sha256),
        "gate": dict(gate),
        "stability_limitation": (
            "Screening and independent audit perturb one stored track set; "
            "visual re-tracking stability remains unmeasured."
        ),
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
        "archive_sha256": _file_digest(paths["archive"]),
        "manifest_sha256": _file_digest(paths["manifest"]),
    }
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R7_FINAL_DONE_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "world_size": FINAL_WORLD_SIZE,
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": _file_digest(path),
            }
            for name, path in paths.items()
            if name != "done"
        },
    }
    _atomic_json(paths["done"], done)
    return done


def validate_final(
    directory: Path,
    *,
    input_manifest: Path | None = None,
) -> dict[str, Any]:
    paths = _artifact_paths(directory)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != R7_FINAL_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("formal_status") != "INSUFFICIENT"
        or done.get("production_decision") is not False
        or done.get("generation_authorized") is not False
    ):
        raise ValueError("invalid R7 final done marker")
    artifacts = done.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "archive",
        "manifest",
        "summary",
    }:
        raise ValueError("R7 final artifact registry differs")
    for name, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != paths[name].name
            or record.get("sha256") != _file_digest(paths[name])
        ):
            raise ValueError(f"R7 final {name} digest mismatch")
    rows = _load_jsonl(paths["manifest"])
    with np.load(paths["archive"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_array_contract(arrays, rows=len(rows))
    indices = np.asarray(arrays["input_indices"]).tolist()
    if indices != list(range(len(rows))):
        raise ValueError("R7 final archive is not in complete input order")
    if [int(row["input_index"]) for row in rows] != indices:
        raise ValueError("R7 final manifest/archive order differs")
    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != R7_FINAL_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("rows") != len(rows)
        or summary.get("formal_status") != "INSUFFICIENT"
        or summary.get("production_decision") is not False
        or summary.get("generation_authorized") is not False
    ):
        raise ValueError("invalid R7 final summary")
    rebuilt_gate = compute_p0_gate(rows, arrays)
    if summary.get("gate") != rebuilt_gate:
        raise ValueError("R7 final gate is not reproducible")
    if input_manifest is not None:
        manifest = input_manifest.expanduser().resolve(strict=True)
        input_rows = _read_r5_manifest(manifest)
        if (
            len(input_rows) != len(rows)
            or summary.get("input_manifest_sha256") != _file_digest(manifest)
        ):
            raise ValueError("R7 final input manifest differs")
        for index, row in enumerate(rows):
            if row.get("input_row_sha256") != _object_digest(
                input_rows[index]
            ):
                raise ValueError(f"R7 final row {index} digest differs")
    return {
        "done": done,
        "summary": summary,
        "rows": rows,
        "arrays": arrays,
    }


def finalize_shards(
    *,
    input_manifest: Path,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate and merge exactly eight modulo shards."""

    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = _read_r5_manifest(manifest_path)
    final_dir = _merged_directory(output_root)
    if (final_dir / DONE_NAME).exists():
        if not resume:
            raise FileExistsError(final_dir / DONE_NAME)
        return validate_final(
            final_dir,
            input_manifest=manifest_path,
        )["done"]
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(
            f"partial R7 final artifact cannot be resumed: {final_dir}"
        )
    shards: list[dict[str, Any]] = []
    for rank in range(FINAL_WORLD_SIZE):
        shard = validate_shard(
            rank_directory(output_root, rank, FINAL_WORLD_SIZE),
            input_manifest=manifest_path,
            rehash_videos=False,
        )
        if (
            shard["contract"]["rank"] != rank
            or shard["contract"]["world_size"] != FINAL_WORLD_SIZE
        ):
            raise ValueError(f"R7 shard rank contract differs for rank {rank}")
        shards.append(shard)
    records: list[tuple[int, dict[str, Any], int, int]] = []
    for rank, shard in enumerate(shards):
        for local_index, row in enumerate(shard["rows"]):
            records.append(
                (int(row["input_index"]), dict(row), rank, local_index)
            )
    records.sort(key=lambda item: item[0])
    indices = [item[0] for item in records]
    if indices != list(range(len(input_rows))):
        raise ValueError("eight R7 shards do not exactly cover the input")
    merged_rows: list[dict[str, Any]] = []
    merged_arrays = _empty_arrays(len(records))
    for merged_index, (input_index, row, rank, local_index) in enumerate(records):
        row["merged_array_index"] = merged_index
        merged_rows.append(row)
        source_arrays = shards[rank]["arrays"]
        for name in merged_arrays:
            merged_arrays[name][merged_index] = source_arrays[name][local_index]
        if int(merged_arrays["input_indices"][merged_index]) != input_index:
            raise ValueError("R7 shard array index changed while merging")
    gate = compute_p0_gate(merged_rows, merged_arrays)
    return _commit_final(
        directory=final_dir,
        rows=merged_rows,
        arrays=merged_arrays,
        input_manifest=manifest_path,
        shard_done_sha256=[
            _file_digest(
                rank_directory(output_root, rank, FINAL_WORLD_SIZE) / DONE_NAME
            )
            for rank in range(FINAL_WORLD_SIZE)
        ],
        gate=gate,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R7-P0 committed CoTracker/DINO extraction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract one torchrun rank")
    extract.add_argument("--input-manifest", type=Path, required=True)
    extract.add_argument("--data-root", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--cotracker-checkpoint", type=Path, required=True)
    extract.add_argument("--dinov2-model", type=Path, required=True)
    extract.add_argument("--dinov2-revision")
    extract.add_argument("--rank", type=int)
    extract.add_argument("--world-size", type=int)
    extract.add_argument("--local-rank", type=int)
    extract.add_argument("--tracker-grid-size", type=int, default=10)
    extract.add_argument("--seed", type=int, default=DEFAULT_SEED)
    extract.add_argument("--resume", action="store_true")

    finalize = subparsers.add_parser("finalize", help="merge exactly 8 ranks")
    finalize.add_argument("--input-manifest", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "extract":
        rank, world_size, local_rank = resolve_torchrun_coordinates(
            rank=args.rank,
            world_size=args.world_size,
            local_rank=args.local_rank,
        )
        result = extract_rank(
            input_manifest=args.input_manifest,
            data_root=args.data_root,
            output_root=args.output_dir,
            tracker_checkpoint=args.cotracker_checkpoint,
            dino_model_root=args.dinov2_model,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            tracker_grid_size=args.tracker_grid_size,
            seed=args.seed,
            dino_revision=args.dinov2_revision,
            resume=args.resume,
        )
    else:
        result = finalize_shards(
            input_manifest=args.input_manifest,
            output_root=args.output_dir,
            resume=args.resume,
        )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ARCHIVE_NAME",
    "DecodedVideo",
    "DINO_DIM",
    "DINO_FRAMES",
    "DONE_NAME",
    "FINAL_WORLD_SIZE",
    "GlobalExtractionError",
    "LazyDinoV2BaseEncoder",
    "MANIFEST_NAME",
    "MAX_VIDEO_SIDE",
    "PerVideoError",
    "R7_PREFLIGHT_SCHEMA",
    "SUMMARY_NAME",
    "VIDEO_FRAMES",
    "compute_p0_gate",
    "decode_video_fixed_frames",
    "difference_hash",
    "dino_frame_offsets",
    "extract_rank",
    "finalize_shards",
    "rank_directory",
    "resized_dimensions",
    "resolve_torchrun_coordinates",
    "sampled_frame_times",
    "uniform_sample_indices",
    "validate_final",
    "validate_shard",
]

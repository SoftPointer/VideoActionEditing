"""Hash-bound visual features for the unsplit R7 expansion candidates.

This stage is deliberately smaller than :mod:`r7_preflight_extract`.  It
decodes the fixed 32-frame video sample and stores only the six frozen
DINOv2 CLS vectors, six 64-bit difference hashes, and the source-video
SHA-256 for each side.  CoTracker and temporal-teacher code are never run.

``extract`` atomically publishes one of exactly eight input-index-modulo
shards.  ``finalize`` requires the exact eight-shard directory set, validates
all hashes and input bindings, and atomically publishes an input-ordered
commit.  ``--resume`` is verification-only: it never decodes a video or runs
the encoder.

A corrupt individual video is represented as a failed side with zero feature
arrays.  A final artifact can therefore be complete while
``summary.json["split_ready"]`` is false.  DINO runtime or output-contract
failures are process-global and abort without publishing a shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .r7_preflight_extract import (
    DINO_DIM,
    DINO_FRAMES,
    MAX_VIDEO_SIDE,
    R7_DINO_POOLING,
    R7_DINO_PREPROCESSING,
    R7_DINO_SAMPLING,
    R7_VIDEO_SAMPLING,
    VIDEO_FRAMES,
    GlobalExtractionError,
    LazyDinoV2BaseEncoder,
    PerVideoError,
    decode_video_fixed_frames,
    difference_hash,
    dino_frame_offsets,
    resolve_torchrun_coordinates,
)
from .r7_visual_candidate_manifest import (
    CANDIDATE_ROW_FIELDS,
    ROW_SCHEMA as CANDIDATE_ROW_SCHEMA,
)


SCHEMA_VERSION = "motive-r7-expansion-visual-features-v1"
ROW_SCHEMA = "motive-r7-expansion-visual-feature-row-v1"
SHARD_SUMMARY_SCHEMA = (
    "motive-r7-expansion-visual-feature-shard-summary-v1"
)
SHARD_DONE_SCHEMA = "motive-r7-expansion-visual-feature-shard-done-v1"
FINAL_SUMMARY_SCHEMA = (
    "motive-r7-expansion-visual-feature-final-summary-v1"
)
FINAL_DONE_SCHEMA = "motive-r7-expansion-visual-feature-final-done-v1"
PARTITION_VERSION = "input-index-modulo-exactly-8-v1"

FINAL_WORLD_SIZE = 8
SIDES = ("source", "target")
SIDE_FIELDS = {"source": "src_video", "target": "tgt_video"}

ARCHIVE_NAME = "features.npz"
MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ARTIFACT_NAMES = frozenset(
    {ARCHIVE_NAME, MANIFEST_NAME, SUMMARY_NAME, DONE_NAME}
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DHASH_RE = re.compile(r"^[0-9a-f]{16}$")
_FEATURE_ROW_KEYS = frozenset(
    {
        "schema_version",
        "input_index",
        "shard_array_index",
        "shard_rank",
        "world_size",
        "iid",
        "input_row_sha256",
        "source",
        "target",
        "paired_valid",
    }
)
_SIDE_RECORD_KEYS = frozenset(
    {
        "status",
        "valid",
        "failure_stage",
        "failure_reason",
        "failure_message",
        "resolved_path",
        "video_sha256",
        "decode",
    }
)
_DECODE_RECORD_KEYS = frozenset(
    {
        "sampling_version",
        "decoded_frames",
        "source_frame_indices",
        "source_fps",
        "source_frame_count",
        "source_size",
        "resized_size",
        "dino_frame_offsets",
        "dino_source_frame_indices",
        "difference_hashes",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "input_manifest",
        "input_manifest_sha256",
        "input_rows",
        "data_root",
        "rank",
        "world_size",
        "device",
        "partition",
        "video_sampling",
        "dino_sampling",
        "dino",
        "cotracker_executed",
        "implementation",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    ).encode("utf-8")


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _object_digest(
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_canonical_jsonl(
    path: Path,
    *,
    allow_empty: bool,
) -> list[dict[str, Any]]:
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
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    if not rows and not allow_empty:
        raise ValueError(f"{path} contains no rows")
    return rows


def load_candidate_manifest(path: Path) -> list[dict[str, Any]]:
    """Load the canonical, unsplit candidate manifest and enforce unique IIDs."""

    manifest = path.expanduser().resolve(strict=True)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = _load_canonical_jsonl(manifest, allow_empty=False)
    seen: set[str] = set()
    source_artifact_digest: str | None = None
    for index, row in enumerate(rows):
        if set(row) != set(CANDIDATE_ROW_FIELDS):
            raise ValueError(
                f"candidate row {index} field set differs: "
                f"missing={sorted(set(CANDIDATE_ROW_FIELDS) - set(row))}, "
                f"extra={sorted(set(row) - set(CANDIDATE_ROW_FIELDS))}"
            )
        if row.get("schema_version") != CANDIDATE_ROW_SCHEMA:
            raise ValueError(
                f"candidate row {index} schema version differs"
            )
        iid = row.get("iid")
        if (
            type(iid) is not str
            or not iid
            or iid.strip() != iid
            or "\x00" in iid
        ):
            raise ValueError(f"candidate row {index} has an invalid iid")
        if iid in seen:
            raise ValueError(f"candidate row {index} duplicates iid={iid}")
        seen.add(iid)
        for field in ("prompt", "primary_family", *SIDE_FIELDS.values()):
            value = row.get(field)
            if (
                type(value) is not str
                or not value
                or value.strip() != value
                or "\x00" in value
            ):
                raise ValueError(
                    f"candidate row {index} has invalid {field}"
                )
        for field in (
            "input_digest",
            "source_row_sha256",
            "source_artifact_digest",
        ):
            value = row.get(field)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(
                    f"candidate row {index} has invalid {field}"
                )
        row_source_digest = str(row["source_artifact_digest"])
        if source_artifact_digest is None:
            source_artifact_digest = row_source_digest
        elif row_source_digest != source_artifact_digest:
            raise ValueError(
                "candidate rows disagree on source_artifact_digest"
            )
        if row.get("cohort") not in {
            "pseudo_positive",
            "pseudo_negative",
        }:
            raise ValueError(
                f"candidate row {index} has invalid cohort"
            )
        if "split" in row or "split_provenance" in row:
            raise ValueError(
                f"candidate row {index} carries a pre-existing split"
            )
        for field in (
            "split_assigned",
            "human_label",
            "training_eligible",
        ):
            if row.get(field) is not False:
                raise ValueError(
                    f"candidate row {index} asserts {field}"
                )
    return rows


def _safe_video_path(data_root: Path, value: str) -> Path:
    root = data_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"video path escapes data_root: {value!r}"
        ) from error
    return resolved


def _validate_rank(rank: int, world_size: int) -> None:
    if (
        type(rank) is not int
        or type(world_size) is not int
        or world_size != FINAL_WORLD_SIZE
        or not 0 <= rank < world_size
    ):
        raise ValueError(
            f"visual extraction requires integer 0 <= rank < "
            f"{FINAL_WORLD_SIZE} and world_size={FINAL_WORLD_SIZE}"
        )


def rank_directory(root: Path, rank: int, world_size: int) -> Path:
    _validate_rank(rank, world_size)
    return (
        root.expanduser()
        / "shards"
        / f"rank-{rank:03d}-of-{world_size:03d}"
    )


def _final_directory(root: Path) -> Path:
    return root.expanduser() / "final"


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "archive": directory / ARCHIVE_NAME,
        "manifest": directory / MANIFEST_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _empty_arrays(rows: int) -> dict[str, np.ndarray]:
    if type(rows) is not int or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    arrays: dict[str, np.ndarray] = {
        "input_indices": np.zeros(rows, dtype=np.int64),
    }
    for side in SIDES:
        arrays[f"{side}_valid"] = np.zeros(rows, dtype=np.bool_)
        arrays[f"{side}_dino_cls"] = np.zeros(
            (rows, DINO_FRAMES, DINO_DIM),
            dtype=np.float32,
        )
        arrays[f"{side}_difference_hashes"] = np.full(
            (rows, DINO_FRAMES),
            "",
            dtype="<U16",
        )
        arrays[f"{side}_video_sha256"] = np.full(
            rows,
            "",
            dtype="<U64",
        )
    return arrays


def _validate_array_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
) -> None:
    expected = _empty_arrays(rows)
    if set(arrays) != set(expected):
        raise ValueError(
            "visual feature array names differ: "
            f"missing={sorted(set(expected) - set(arrays))}, "
            f"extra={sorted(set(arrays) - set(expected))}"
        )
    for name, template in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError(
                f"{name} shape/dtype differs: got "
                f"{value.shape}/{value.dtype}, expected "
                f"{template.shape}/{template.dtype}"
            )
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    indices = np.asarray(arrays["input_indices"], dtype=np.int64)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("visual feature input_indices contain duplicates")
    for side in SIDES:
        valid = np.asarray(arrays[f"{side}_valid"], dtype=bool)
        features = np.asarray(arrays[f"{side}_dino_cls"])
        hashes = np.asarray(arrays[f"{side}_difference_hashes"])
        video_hashes = np.asarray(arrays[f"{side}_video_sha256"])
        if bool((features[~valid] != 0).any()):
            raise ValueError(f"{side} invalid rows contain DINO features")
        if bool((hashes[~valid] != "").any()):
            raise ValueError(
                f"{side} invalid rows contain difference hashes"
            )
        if len(features[valid]):
            norms = np.linalg.norm(
                features[valid].astype(np.float64),
                axis=2,
            )
            if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
                raise ValueError(
                    f"{side} valid DINO CLS vectors are not L2-normalized"
                )
            if any(
                _DHASH_RE.fullmatch(str(value)) is None
                for value in hashes[valid].reshape(-1)
            ):
                raise ValueError(
                    f"{side} valid rows contain invalid 64-bit dHashes"
                )
        if any(
            value and _SHA256_RE.fullmatch(str(value)) is None
            for value in video_hashes.tolist()
        ):
            raise ValueError(f"{side} video SHA array is invalid")


def _validate_dino_provenance(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("DINO provenance is not a mapping")
    provenance = dict(raw)
    # Canonical serialization also rejects NaN and non-JSON values.
    _canonical_json(provenance)
    expected_keys = {
        "encoder_id",
        "encoder_revision",
        "resolved_path",
        "model_tree_sha256",
        "weights_sha256",
        "model_file_count",
        "frame_sampling_version",
        "preprocessing_version",
        "pooling",
        "embedding_dim",
        "dtype",
        "normalization",
        "frozen_encoder",
        "local_files_only",
    }
    if set(provenance) != expected_keys:
        raise ValueError(
            "DINO provenance field set differs: "
            f"missing={sorted(expected_keys - set(provenance))}, "
            f"extra={sorted(set(provenance) - expected_keys)}"
        )
    if (
        provenance.get("encoder_id") != "facebook/dinov2-base"
        or type(provenance.get("encoder_revision")) is not str
        or re.fullmatch(
            r"[0-9a-f]{7,64}",
            str(provenance["encoder_revision"]),
        )
        is None
        or type(provenance.get("resolved_path")) is not str
        or not Path(str(provenance["resolved_path"])).is_absolute()
        or type(provenance.get("model_file_count")) is not int
        or int(provenance["model_file_count"]) < 1
        or any(
            type(provenance.get(field)) is not str
            or _SHA256_RE.fullmatch(str(provenance[field])) is None
            for field in ("model_tree_sha256", "weights_sha256")
        )
        or provenance.get("frame_sampling_version") != R7_DINO_SAMPLING
        or provenance.get("preprocessing_version")
        != R7_DINO_PREPROCESSING
        or provenance.get("pooling") != R7_DINO_POOLING
        or provenance.get("embedding_dim") != DINO_DIM
        or provenance.get("dtype") != "float32"
        or provenance.get("normalization") != "l2-per-frame"
        or provenance.get("frozen_encoder") is not True
        or provenance.get("local_files_only") is not True
    ):
        raise ValueError(
            "DINO provenance must attest frozen float32 DINOv2-base "
            f"features with embedding_dim={DINO_DIM}"
        )
    return provenance


def _encoder_provenance(encoder: Any) -> dict[str, Any]:
    raw = getattr(encoder, "provenance", None)
    if raw is None:
        raise ValueError("DINO encoder has no provenance mapping")
    return _validate_dino_provenance(raw)


def _implementation_provenance() -> dict[str, Any]:
    module = Path(__file__).resolve(strict=True)
    return {
        "module": module.name,
        "module_sha256": _file_digest(module),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def _build_contract(
    *,
    input_manifest: Path,
    input_rows: int,
    data_root: Path,
    rank: int,
    world_size: int,
    device: str,
    dino_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_rank(rank, world_size)
    return {
        "schema_version": SCHEMA_VERSION,
        "input_manifest": str(input_manifest.resolve(strict=True)),
        "input_manifest_sha256": _file_digest(input_manifest),
        "input_rows": input_rows,
        "data_root": str(data_root.resolve(strict=True)),
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "partition": PARTITION_VERSION,
        "video_sampling": {
            "version": R7_VIDEO_SAMPLING,
            "decoded_frames": VIDEO_FRAMES,
            "maximum_side": MAX_VIDEO_SIDE,
        },
        "dino_sampling": {
            "version": R7_DINO_SAMPLING,
            "frames": DINO_FRAMES,
            "offsets": dino_frame_offsets().tolist(),
            "preprocessing": R7_DINO_PREPROCESSING,
            "pooling": R7_DINO_POOLING,
        },
        "dino": dict(dino_provenance),
        "cotracker_executed": False,
        "implementation": _implementation_provenance(),
    }


def _validate_contract_semantics(
    contract: Mapping[str, Any],
    *,
    common: bool = False,
) -> None:
    expected_keys = set(_CONTRACT_KEYS)
    if common:
        expected_keys -= {"rank", "device"}
    if set(contract) != expected_keys:
        raise ValueError(
            "visual feature contract key set differs: "
            f"missing={sorted(expected_keys - set(contract))}, "
            f"extra={sorted(set(contract) - expected_keys)}"
        )
    digest = contract.get("input_manifest_sha256")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("partition") != PARTITION_VERSION
        or contract.get("world_size") != FINAL_WORLD_SIZE
        or type(contract.get("input_rows")) is not int
        or int(contract["input_rows"]) < 1
        or type(contract.get("input_manifest")) is not str
        or type(contract.get("data_root")) is not str
        or type(digest) is not str
        or _SHA256_RE.fullmatch(digest) is None
        or contract.get("cotracker_executed") is not False
    ):
        raise ValueError("visual feature contract scalar fields differ")
    input_path = Path(str(contract["input_manifest"])).expanduser()
    data_root = Path(str(contract["data_root"])).expanduser()
    if (
        not input_path.is_absolute()
        or str(input_path.resolve(strict=False))
        != str(contract["input_manifest"])
        or not data_root.is_absolute()
        or str(data_root.resolve(strict=False)) != str(contract["data_root"])
    ):
        raise ValueError("visual feature contract paths are not canonical")
    if not common:
        _validate_rank(contract.get("rank"), contract.get("world_size"))
        device = contract.get("device")
        if (
            type(device) is not str
            or re.fullmatch(r"cuda:[0-9]+", device) is None
        ):
            raise ValueError("visual feature contract device differs")
    if contract.get("video_sampling") != {
        "version": R7_VIDEO_SAMPLING,
        "decoded_frames": VIDEO_FRAMES,
        "maximum_side": MAX_VIDEO_SIDE,
    }:
        raise ValueError("visual feature video-sampling contract differs")
    if contract.get("dino_sampling") != {
        "version": R7_DINO_SAMPLING,
        "frames": DINO_FRAMES,
        "offsets": dino_frame_offsets().tolist(),
        "preprocessing": R7_DINO_PREPROCESSING,
        "pooling": R7_DINO_POOLING,
    }:
        raise ValueError("visual feature DINO-sampling contract differs")
    _validate_dino_provenance(contract.get("dino"))
    implementation = contract.get("implementation")
    if (
        not isinstance(implementation, Mapping)
        or set(implementation)
        != {"module", "module_sha256", "python", "numpy"}
        or implementation.get("module")
        != Path(__file__).resolve(strict=True).name
        or type(implementation.get("module_sha256")) is not str
        or _SHA256_RE.fullmatch(str(implementation["module_sha256"]))
        is None
        or type(implementation.get("python")) is not str
        or not implementation["python"]
        or type(implementation.get("numpy")) is not str
        or not implementation["numpy"]
    ):
        raise ValueError("visual feature implementation contract differs")


def _validate_dino_matrix(value: Any) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float32)
    except Exception as error:
        raise GlobalExtractionError(
            "DINO output cannot be represented as float32"
        ) from error
    if matrix.shape != (DINO_FRAMES, DINO_DIM):
        raise GlobalExtractionError(
            f"DINO output shape is {matrix.shape}, expected "
            f"{(DINO_FRAMES, DINO_DIM)}"
        )
    if not np.isfinite(matrix).all():
        raise GlobalExtractionError("DINO output contains non-finite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
        raise GlobalExtractionError(
            "DINO CLS vectors are not L2-normalized"
        )
    return np.ascontiguousarray(matrix)


def _failed_side(
    *,
    path: Path,
    video_sha256: str | None,
    stage: str,
    reason: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "valid": False,
        "failure_stage": stage,
        "failure_reason": reason,
        "failure_message": message,
        "resolved_path": str(path),
        "video_sha256": video_sha256,
        "decode": None,
    }


def _extract_side(
    *,
    path: Path,
    side: str,
    array_index: int,
    arrays: dict[str, np.ndarray],
    encoder: Any,
) -> dict[str, Any]:
    video_sha256: str | None = None
    if path.is_file():
        video_sha256 = _file_digest(path)
    if video_sha256 is not None:
        arrays[f"{side}_video_sha256"][array_index] = video_sha256
    try:
        decoded = decode_video_fixed_frames(path)
    except PerVideoError as error:
        return _failed_side(
            path=path,
            video_sha256=video_sha256,
            stage="decode",
            reason=error.reason,
            message=str(error),
        )
    frames = np.asarray(decoded.frames_rgb)
    if (
        frames.ndim != 4
        or frames.shape[0] != VIDEO_FRAMES
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
    ):
        raise GlobalExtractionError(
            "fixed video decoder violated its 32-frame uint8 RGB contract"
        )
    source_indices = np.asarray(decoded.source_frame_indices)
    if (
        source_indices.shape != (VIDEO_FRAMES,)
        or not np.issubdtype(source_indices.dtype, np.integer)
    ):
        raise GlobalExtractionError(
            "fixed video decoder returned invalid source-frame indices"
        )
    offsets = dino_frame_offsets()
    selected = np.ascontiguousarray(frames[offsets])
    try:
        encoded = encoder.encode(selected)
    except GlobalExtractionError:
        raise
    except Exception as error:
        raise GlobalExtractionError(
            f"DINO inference failed for {path}"
        ) from error
    matrix = _validate_dino_matrix(encoded)
    try:
        hashes = [difference_hash(frame) for frame in selected]
    except Exception as error:
        raise GlobalExtractionError(
            "difference-hash computation violated the fixed frame contract"
        ) from error
    if any(_DHASH_RE.fullmatch(value) is None for value in hashes):
        raise GlobalExtractionError(
            "difference_hash did not return a 64-bit lowercase hex value"
        )
    arrays[f"{side}_valid"][array_index] = True
    arrays[f"{side}_dino_cls"][array_index] = matrix
    arrays[f"{side}_difference_hashes"][array_index] = hashes
    return {
        "status": "usable",
        "valid": True,
        "failure_stage": None,
        "failure_reason": None,
        "failure_message": None,
        "resolved_path": str(path),
        "video_sha256": video_sha256,
        "decode": {
            "sampling_version": R7_VIDEO_SAMPLING,
            "decoded_frames": VIDEO_FRAMES,
            "source_frame_indices": source_indices.astype(
                np.int64
            ).tolist(),
            "source_fps": float(decoded.source_fps),
            "source_frame_count": int(decoded.source_frame_count),
            "source_size": [int(value) for value in decoded.source_size],
            "resized_size": [
                int(value) for value in decoded.resized_size
            ],
            "dino_frame_offsets": offsets.tolist(),
            "dino_source_frame_indices": source_indices[
                offsets
            ].astype(np.int64).tolist(),
            "difference_hashes": hashes,
        },
    }


def _failure_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    failed_rows = 0
    side_valid: dict[str, int] = {}
    for side in SIDES:
        side_valid[side] = sum(
            bool(row[side]["valid"]) for row in rows
        )
    for row in rows:
        row_failed = False
        for side in SIDES:
            record = row[side]
            if not bool(record["valid"]):
                row_failed = True
                failures[
                    f"{side}:{record.get('failure_reason') or 'unknown'}"
                ] += 1
        failed_rows += int(row_failed)
    failed_sides = 2 * len(rows) - sum(side_valid.values())
    return {
        "extraction_status": "passed" if failed_sides == 0 else "failed",
        "split_ready": failed_sides == 0,
        "source_valid": side_valid["source"],
        "target_valid": side_valid["target"],
        "paired_valid": sum(
            bool(row["source"]["valid"])
            and bool(row["target"]["valid"])
            for row in rows
        ),
        "failed_rows": failed_rows,
        "failed_sides": failed_sides,
        "failures": dict(sorted(failures.items())),
    }


def _array_contract(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_digest(np.asarray(value)),
        }
        for name, value in sorted(arrays.items())
    }


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_commit(
    *,
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    summary: dict[str, Any],
    done_base: Mapping[str, Any],
) -> dict[str, Any]:
    target = directory.expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
    )
    try:
        archive = staging / ARCHIVE_NAME
        with archive.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    name: np.asarray(value)
                    for name, value in arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        manifest = staging / MANIFEST_NAME
        _write_file(manifest, _jsonl_bytes(rows))
        summary.update(
            {
                "archive_sha256": _file_digest(archive),
                "manifest_sha256": _file_digest(manifest),
            }
        )
        summary_path = staging / SUMMARY_NAME
        _write_file(summary_path, _pretty_json_bytes(summary))
        done = dict(done_base)
        done["artifacts"] = {
            "archive": {
                "filename": ARCHIVE_NAME,
                "sha256": _file_digest(archive),
            },
            "manifest": {
                "filename": MANIFEST_NAME,
                "sha256": _file_digest(manifest),
            },
            "summary": {
                "filename": SUMMARY_NAME,
                "sha256": _file_digest(summary_path),
            },
        }
        _write_file(staging / DONE_NAME, _pretty_json_bytes(done))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"commit target appeared during publication: {target}"
            )
        os.rename(staging, target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return done
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_artifact_set(
    directory: Path,
    *,
    done_schema: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(directory)
    actual = {entry.name for entry in directory.iterdir()}
    if actual != ARTIFACT_NAMES:
        raise ValueError(
            "visual feature artifact set differs: "
            f"missing={sorted(ARTIFACT_NAMES - actual)}, "
            f"extra={sorted(actual - ARTIFACT_NAMES)}"
        )
    paths = _artifact_paths(directory)
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
    done = _load_json(paths["done"])
    if (
        done.get("schema_version") != done_schema
        or done.get("status") != "complete"
    ):
        raise ValueError("invalid visual feature done marker")
    artifacts = done.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"archive", "manifest", "summary"}
    ):
        raise ValueError("visual feature artifact registry differs")
    for name, record in artifacts.items():
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != paths[name].name
            or record.get("sha256") != _file_digest(paths[name])
        ):
            raise ValueError(
                f"visual feature {name} artifact digest mismatch"
            )
    return paths, done


def _validate_side_record(
    *,
    row: Mapping[str, Any],
    row_position: int,
    side: str,
    arrays: Mapping[str, np.ndarray],
    data_root: Path,
    rehash_videos: bool,
) -> None:
    raw = row.get(side)
    if not isinstance(raw, Mapping) or set(raw) != _SIDE_RECORD_KEYS:
        raise ValueError(
            f"feature row {row_position} has invalid {side} record"
        )
    record = dict(raw)
    valid = record.get("valid")
    if type(valid) is not bool:
        raise ValueError(
            f"feature row {row_position} {side}.valid is not bool"
        )
    if record.get("status") != ("usable" if valid else "failed"):
        raise ValueError(
            f"feature row {row_position} {side} status differs"
        )
    path_value = record.get("resolved_path")
    if type(path_value) is not str or not path_value:
        raise ValueError(
            f"feature row {row_position} {side} path is invalid"
        )
    path = _safe_video_path(data_root, path_value)
    if str(path) != path_value:
        raise ValueError(
            f"feature row {row_position} {side} path is not resolved"
        )
    digest = record.get("video_sha256")
    if digest is not None and (
        type(digest) is not str or _SHA256_RE.fullmatch(digest) is None
    ):
        raise ValueError(
            f"feature row {row_position} {side} video SHA is invalid"
        )
    array_digest = str(
        np.asarray(arrays[f"{side}_video_sha256"])[row_position]
    )
    if array_digest != (digest or ""):
        raise ValueError(
            f"feature row {row_position} {side} video SHA differs "
            "from archive"
        )
    if rehash_videos:
        if digest is None:
            if path.is_file():
                raise ValueError(
                    f"{side} video appeared after extraction: {path}"
                )
        elif not path.is_file() or _file_digest(path) != digest:
            raise ValueError(
                f"{side} video bytes changed after extraction: {path}"
            )
    array_valid = bool(
        np.asarray(arrays[f"{side}_valid"])[row_position]
    )
    if array_valid != valid:
        raise ValueError(
            f"feature row {row_position} {side} validity differs "
            "from archive"
        )
    if not valid:
        if (
            type(record.get("failure_stage")) is not str
            or not record["failure_stage"]
            or type(record.get("failure_reason")) is not str
            or not record["failure_reason"]
            or type(record.get("failure_message")) is not str
            or not record["failure_message"]
            or record.get("decode") is not None
        ):
            raise ValueError(
                f"feature row {row_position} {side} failure record differs"
            )
        return
    if digest is None:
        raise ValueError(
            f"feature row {row_position} valid {side} lacks video SHA"
        )
    if any(
        record.get(field) is not None
        for field in (
            "failure_stage",
            "failure_reason",
            "failure_message",
        )
    ):
        raise ValueError(
            f"feature row {row_position} valid {side} has failure fields"
        )
    decode = record.get("decode")
    if not isinstance(decode, Mapping) or set(decode) != _DECODE_RECORD_KEYS:
        raise ValueError(
            f"feature row {row_position} {side} decode record differs"
        )
    offsets = dino_frame_offsets().tolist()
    source_indices = decode.get("source_frame_indices")
    hashes = decode.get("difference_hashes")
    if (
        decode.get("sampling_version") != R7_VIDEO_SAMPLING
        or decode.get("decoded_frames") != VIDEO_FRAMES
        or not isinstance(source_indices, list)
        or len(source_indices) != VIDEO_FRAMES
        or any(type(value) is not int for value in source_indices)
        or decode.get("dino_frame_offsets") != offsets
        or decode.get("dino_source_frame_indices")
        != [source_indices[index] for index in offsets]
        or not isinstance(hashes, list)
        or len(hashes) != DINO_FRAMES
        or any(
            type(value) is not str
            or _DHASH_RE.fullmatch(value) is None
            for value in hashes
        )
    ):
        raise ValueError(
            f"feature row {row_position} {side} sampling record differs"
        )
    fps = decode.get("source_fps")
    frame_count = decode.get("source_frame_count")
    source_size = decode.get("source_size")
    resized_size = decode.get("resized_size")
    if (
        type(fps) not in {int, float}
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or type(frame_count) is not int
        or frame_count < 2
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int or item < 1 for item in value)
            for value in (source_size, resized_size)
        )
    ):
        raise ValueError(
            f"feature row {row_position} {side} decode metadata differs"
        )
    archive_hashes = np.asarray(
        arrays[f"{side}_difference_hashes"]
    )[row_position].tolist()
    if archive_hashes != hashes:
        raise ValueError(
            f"feature row {row_position} {side} dHashes differ "
            "from archive"
        )


def _validate_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    *,
    data_root: Path,
    rehash_videos: bool,
) -> None:
    for position, raw in enumerate(rows):
        if set(raw) != _FEATURE_ROW_KEYS:
            raise ValueError(f"feature row {position} key set differs")
        if raw.get("schema_version") != ROW_SCHEMA:
            raise ValueError(f"feature row {position} schema differs")
        for name in (
            "input_index",
            "shard_array_index",
            "shard_rank",
            "world_size",
        ):
            if type(raw.get(name)) is not int:
                raise ValueError(
                    f"feature row {position} {name} is not an integer"
                )
        _validate_rank(
            int(raw["shard_rank"]),
            int(raw["world_size"]),
        )
        iid = raw.get("iid")
        digest = raw.get("input_row_sha256")
        if (
            type(iid) is not str
            or not iid
            or type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError(
                f"feature row {position} identity binding differs"
            )
        for side in SIDES:
            _validate_side_record(
                row=raw,
                row_position=position,
                side=side,
                arrays=arrays,
                data_root=data_root,
                rehash_videos=rehash_videos,
            )
        paired = bool(raw["source"]["valid"]) and bool(
            raw["target"]["valid"]
        )
        if type(raw.get("paired_valid")) is not bool or (
            raw["paired_valid"] != paired
        ):
            raise ValueError(
                f"feature row {position} paired validity differs"
            )


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except Exception as error:
        raise ValueError(f"invalid visual feature archive: {path}") from error


def validate_shard(
    directory: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    input_manifest: Path | None = None,
    rehash_videos: bool = False,
) -> dict[str, Any]:
    """Strictly validate one immutable expansion visual-feature shard."""

    paths, done = _validate_artifact_set(
        directory,
        done_schema=SHARD_DONE_SCHEMA,
    )
    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != SHARD_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise ValueError("invalid visual feature shard summary")
    contract_raw = summary.get("contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError("visual feature shard lacks a contract")
    contract = dict(contract_raw)
    _validate_contract_semantics(contract)
    contract_digest = _object_digest(contract)
    if (
        summary.get("contract_sha256") != contract_digest
        or done.get("contract_sha256") != contract_digest
    ):
        raise ValueError("visual feature shard contract digest mismatch")
    if expected_contract is not None and contract != dict(
        expected_contract
    ):
        raise ValueError("resume arguments differ from committed shard")
    rows = _load_canonical_jsonl(paths["manifest"], allow_empty=True)
    arrays = _load_archive(paths["archive"])
    _validate_array_contract(arrays, rows=len(rows))
    if summary.get("array_contract") != _array_contract(arrays):
        raise ValueError("visual feature shard array contract differs")
    if (
        summary.get("archive_sha256") != _file_digest(paths["archive"])
        or summary.get("manifest_sha256") != _file_digest(
            paths["manifest"]
        )
    ):
        raise ValueError("visual feature shard summary hashes differ")
    rank = contract.get("rank")
    world_size = contract.get("world_size")
    _validate_rank(rank, world_size)
    if (
        done.get("rank") != rank
        or done.get("world_size") != world_size
        or done.get("rows") != len(rows)
        or summary.get("rank") != rank
        or summary.get("world_size") != world_size
        or summary.get("rows") != len(rows)
    ):
        raise ValueError("visual feature shard rank/count metadata differs")
    if (
        summary.get("input_rows") != contract["input_rows"]
        or summary.get("input_manifest_sha256")
        != contract["input_manifest_sha256"]
        or summary.get("partition") != PARTITION_VERSION
        or summary.get("cotracker_executed") is not False
    ):
        raise ValueError("visual feature shard contract summary differs")
    data_root_value = contract.get("data_root")
    if type(data_root_value) is not str:
        raise ValueError("visual feature shard data_root differs")
    data_root = Path(data_root_value).resolve(strict=True)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)
    indices = np.asarray(arrays["input_indices"], dtype=np.int64).tolist()
    if [int(row["input_index"]) for row in rows] != indices:
        raise ValueError("visual feature shard row/archive order differs")
    if indices != sorted(indices) or any(
        index % world_size != rank for index in indices
    ):
        raise ValueError("visual feature shard modulo ownership differs")
    expected_contract_indices = [
        index
        for index in range(int(contract["input_rows"]))
        if index % world_size == rank
    ]
    if indices != expected_contract_indices:
        raise ValueError(
            "visual feature shard contract coverage is incomplete"
        )
    for local_index, row in enumerate(rows):
        if (
            row.get("shard_array_index") != local_index
            or row.get("shard_rank") != rank
            or row.get("world_size") != world_size
        ):
            raise ValueError(
                f"visual feature shard row {local_index} binding differs"
            )
    _validate_feature_rows(
        rows,
        arrays,
        data_root=data_root,
        rehash_videos=rehash_videos,
    )
    statistics = _failure_statistics(rows)
    if summary.get("statistics") != statistics:
        raise ValueError("visual feature shard failure statistics differ")
    if (
        summary.get("split_ready") is not statistics["split_ready"]
        or summary.get("extraction_status")
        != statistics["extraction_status"]
        or done.get("split_ready") is not statistics["split_ready"]
    ):
        raise ValueError("visual feature shard status differs")
    if input_manifest is not None:
        manifest_path = input_manifest.expanduser().resolve(strict=True)
        input_rows = load_candidate_manifest(manifest_path)
        if (
            contract.get("input_manifest_sha256")
            != _file_digest(manifest_path)
            or contract.get("input_rows") != len(input_rows)
            or summary.get("input_manifest_sha256")
            != _file_digest(manifest_path)
            or summary.get("input_rows") != len(input_rows)
        ):
            raise ValueError("visual feature shard input manifest differs")
        expected_indices = [
            index
            for index in range(len(input_rows))
            if index % world_size == rank
        ]
        if indices != expected_indices:
            raise ValueError(
                "visual feature shard modulo coverage is incomplete"
            )
        for row, input_index in zip(rows, indices):
            source = input_rows[input_index]
            if (
                row.get("iid") != source["iid"]
                or row.get("input_row_sha256")
                != _object_digest(source)
            ):
                raise ValueError(
                    f"visual feature row {input_index} input binding differs"
                )
            for side, field in SIDE_FIELDS.items():
                expected_path = _safe_video_path(
                    data_root,
                    str(source[field]),
                )
                if row[side]["resolved_path"] != str(expected_path):
                    raise ValueError(
                        f"visual feature row {input_index} {side} "
                        "path binding differs"
                    )
    return {
        "done": done,
        "summary": summary,
        "contract": contract,
        "rows": rows,
        "arrays": arrays,
    }


def extract_rank(
    *,
    input_manifest: Path,
    data_root: Path,
    output_root: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    dino_model_root: Path | None = None,
    dino_revision: str | None = None,
    encoder: Any | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Extract and atomically commit one of the exact eight modulo shards."""

    _validate_rank(rank, world_size)
    if type(local_rank) is not int or local_rank < 0:
        raise ValueError("local_rank must be a non-negative integer")
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = load_candidate_manifest(manifest_path)
    root = data_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    device = f"cuda:{local_rank}"
    if encoder is None:
        if dino_model_root is None:
            raise ValueError("dino_model_root is required without encoder")
        encoder = LazyDinoV2BaseEncoder(
            model_root=dino_model_root,
            device=device,
            revision=dino_revision,
        )
    provenance = _encoder_provenance(encoder)
    contract = _build_contract(
        input_manifest=manifest_path,
        input_rows=len(input_rows),
        data_root=root,
        rank=rank,
        world_size=world_size,
        device=device,
        dino_provenance=provenance,
    )
    output = output_root.expanduser().resolve(strict=False)
    directory = rank_directory(output, rank, world_size)
    if resume:
        if not directory.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires an existing "
                f"shard: {directory}"
            )
        return validate_shard(
            directory,
            expected_contract=contract,
            input_manifest=manifest_path,
            rehash_videos=True,
        )["done"]
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(
            f"{directory} exists; use a fresh output or strict --resume"
        )
    selected = [
        (index, row)
        for index, row in enumerate(input_rows)
        if index % world_size == rank
    ]
    arrays = _empty_arrays(len(selected))
    output_rows: list[dict[str, Any]] = []
    for array_index, (input_index, row) in enumerate(selected):
        arrays["input_indices"][array_index] = input_index
        side_records: dict[str, dict[str, Any]] = {}
        for side, field in SIDE_FIELDS.items():
            path = _safe_video_path(root, str(row[field]))
            side_records[side] = _extract_side(
                path=path,
                side=side,
                array_index=array_index,
                arrays=arrays,
                encoder=encoder,
            )
        output_rows.append(
            {
                "schema_version": ROW_SCHEMA,
                "input_index": input_index,
                "shard_array_index": array_index,
                "shard_rank": rank,
                "world_size": world_size,
                "iid": str(row["iid"]),
                "input_row_sha256": _object_digest(row),
                "source": side_records["source"],
                "target": side_records["target"],
                "paired_valid": bool(
                    side_records["source"]["valid"]
                    and side_records["target"]["valid"]
                ),
            }
        )
    _validate_array_contract(arrays, rows=len(output_rows))
    _validate_feature_rows(
        output_rows,
        arrays,
        data_root=root,
        rehash_videos=True,
    )
    statistics = _failure_statistics(output_rows)
    contract_sha256 = _object_digest(contract)
    summary = {
        "schema_version": SHARD_SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(output_rows),
        "input_rows": len(input_rows),
        "input_manifest_sha256": _file_digest(manifest_path),
        "rank": rank,
        "world_size": world_size,
        "partition": PARTITION_VERSION,
        "extraction_status": statistics["extraction_status"],
        "split_ready": statistics["split_ready"],
        "statistics": statistics,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "array_contract": _array_contract(arrays),
        "cotracker_executed": False,
    }
    done_base = {
        "schema_version": SHARD_DONE_SCHEMA,
        "status": "complete",
        "rows": len(output_rows),
        "rank": rank,
        "world_size": world_size,
        "contract_sha256": contract_sha256,
        "split_ready": statistics["split_ready"],
    }
    return _atomic_commit(
        directory=directory,
        rows=output_rows,
        arrays=arrays,
        summary=summary,
        done_base=done_base,
    )


def _common_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    common = dict(contract)
    common.pop("rank", None)
    common.pop("device", None)
    return common


def _validate_exact_shard_set(output_root: Path) -> list[Path]:
    shards_root = output_root.expanduser() / "shards"
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise FileNotFoundError(shards_root)
    expected_names = {
        f"rank-{rank:03d}-of-{FINAL_WORLD_SIZE:03d}"
        for rank in range(FINAL_WORLD_SIZE)
    }
    actual_names = {entry.name for entry in shards_root.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "visual feature shard directory set differs from exact eight: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    paths: list[Path] = []
    for rank in range(FINAL_WORLD_SIZE):
        path = rank_directory(output_root, rank, FINAL_WORLD_SIZE)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"visual feature shard is not a directory: {path}")
        paths.append(path)
    return paths


def _crosscheck_final_against_shards(
    *,
    final_rows: Sequence[Mapping[str, Any]],
    final_arrays: Mapping[str, np.ndarray],
    shards: Sequence[Mapping[str, Any]],
) -> None:
    for rank, shard in enumerate(shards):
        indices = np.asarray(
            shard["arrays"]["input_indices"],
            dtype=np.int64,
        ).tolist()
        for local_index, input_index in enumerate(indices):
            if dict(final_rows[input_index]) != dict(
                shard["rows"][local_index]
            ):
                raise ValueError(
                    f"final row {input_index} differs from shard {rank}"
                )
            for name in final_arrays:
                if not np.array_equal(
                    np.asarray(final_arrays[name])[input_index],
                    np.asarray(shard["arrays"][name])[local_index],
                ):
                    raise ValueError(
                        f"final array {name} row {input_index} differs "
                        f"from shard {rank}"
                    )


def validate_final(
    directory: Path,
    *,
    input_manifest: Path,
    output_root: Path | None = None,
    verify_source_shards: bool = True,
    rehash_videos: bool = True,
) -> dict[str, Any]:
    """Validate the final commit and, by default, all exact source shards."""

    paths, done = _validate_artifact_set(
        directory,
        done_schema=FINAL_DONE_SCHEMA,
    )
    summary = _load_json(paths["summary"])
    if (
        summary.get("schema_version") != FINAL_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("world_size") != FINAL_WORLD_SIZE
    ):
        raise ValueError("invalid visual feature final summary")
    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = load_candidate_manifest(manifest_path)
    rows = _load_canonical_jsonl(paths["manifest"], allow_empty=False)
    arrays = _load_archive(paths["archive"])
    _validate_array_contract(arrays, rows=len(rows))
    if len(rows) != len(input_rows):
        raise ValueError("visual feature final row count differs")
    indices = np.asarray(arrays["input_indices"], dtype=np.int64).tolist()
    if indices != list(range(len(input_rows))):
        raise ValueError("visual feature final archive is not in input order")
    if [int(row["input_index"]) for row in rows] != indices:
        raise ValueError("visual feature final row/archive order differs")
    for index, (row, source) in enumerate(zip(rows, input_rows)):
        if (
            row.get("iid") != source["iid"]
            or row.get("input_row_sha256") != _object_digest(source)
        ):
            raise ValueError(
                f"visual feature final row {index} input binding differs"
            )
    common_raw = summary.get("common_contract")
    if not isinstance(common_raw, Mapping):
        raise ValueError("visual feature final common contract is missing")
    common = dict(common_raw)
    _validate_contract_semantics(common, common=True)
    if (
        common.get("input_manifest_sha256") != _file_digest(manifest_path)
        or common.get("input_rows") != len(input_rows)
        or common.get("world_size") != FINAL_WORLD_SIZE
        or common.get("partition") != PARTITION_VERSION
        or summary.get("input_manifest_sha256")
        != _file_digest(manifest_path)
        or summary.get("input_rows") != len(input_rows)
        or summary.get("common_contract_sha256")
        != _object_digest(common)
    ):
        raise ValueError("visual feature final input/common contract differs")
    data_root_value = common.get("data_root")
    if type(data_root_value) is not str:
        raise ValueError("visual feature final data_root differs")
    data_root = Path(data_root_value).resolve(strict=True)
    for index, (row, source) in enumerate(zip(rows, input_rows)):
        for side, field in SIDE_FIELDS.items():
            expected_path = _safe_video_path(
                data_root,
                str(source[field]),
            )
            if row[side]["resolved_path"] != str(expected_path):
                raise ValueError(
                    f"visual feature final row {index} {side} "
                    "path binding differs"
                )
    _validate_feature_rows(
        rows,
        arrays,
        data_root=data_root,
        rehash_videos=rehash_videos,
    )
    statistics = _failure_statistics(rows)
    expected_reason = (
        None
        if statistics["split_ready"]
        else "one_or_more_video_sides_failed_feature_extraction"
    )
    if (
        summary.get("statistics") != statistics
        or summary.get("split_ready") is not statistics["split_ready"]
        or summary.get("extraction_status")
        != statistics["extraction_status"]
        or done.get("split_ready") is not statistics["split_ready"]
        or summary.get("array_contract") != _array_contract(arrays)
        or summary.get("archive_sha256") != _file_digest(paths["archive"])
        or summary.get("manifest_sha256")
        != _file_digest(paths["manifest"])
        or summary.get("split_fail_closed_reason") != expected_reason
        or summary.get("cotracker_executed") is not False
    ):
        raise ValueError("visual feature final derived metadata differs")
    if (
        done.get("rows") != len(rows)
        or done.get("world_size") != FINAL_WORLD_SIZE
        or done.get("common_contract_sha256") != _object_digest(common)
    ):
        raise ValueError("visual feature final done metadata differs")
    source_shards: list[dict[str, Any]] = []
    if verify_source_shards:
        root = (
            directory.parent
            if output_root is None
            else output_root.expanduser()
        )
        shard_paths = _validate_exact_shard_set(root)
        for rank, shard_path in enumerate(shard_paths):
            shard = validate_shard(
                shard_path,
                input_manifest=manifest_path,
                rehash_videos=False,
            )
            if (
                shard["contract"]["rank"] != rank
                or _common_contract(shard["contract"]) != common
            ):
                raise ValueError(
                    f"visual feature source shard {rank} contract differs"
                )
            source_shards.append(shard)
        expected_done_hashes = [
            _file_digest(path / DONE_NAME) for path in shard_paths
        ]
        if summary.get("shard_done_sha256") != expected_done_hashes:
            raise ValueError(
                "visual feature final source-shard hash registry differs"
            )
        _crosscheck_final_against_shards(
            final_rows=rows,
            final_arrays=arrays,
            shards=source_shards,
        )
    return {
        "done": done,
        "summary": summary,
        "rows": rows,
        "arrays": arrays,
        "source_shards": source_shards,
    }


def finalize_shards(
    *,
    input_manifest: Path,
    output_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate exactly eight shards and merge them in original input order."""

    manifest_path = input_manifest.expanduser().resolve(strict=True)
    input_rows = load_candidate_manifest(manifest_path)
    root = output_root.expanduser().resolve(strict=False)
    final_directory = _final_directory(root)
    if resume:
        if not final_directory.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires an existing "
                f"final commit: {final_directory}"
            )
        return validate_final(
            final_directory,
            input_manifest=manifest_path,
            output_root=root,
            verify_source_shards=True,
            rehash_videos=True,
        )["done"]
    if final_directory.exists() or final_directory.is_symlink():
        raise FileExistsError(
            f"{final_directory} exists; use strict --resume"
        )
    shard_paths = _validate_exact_shard_set(root)
    shards: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    for rank, path in enumerate(shard_paths):
        shard = validate_shard(
            path,
            input_manifest=manifest_path,
            rehash_videos=True,
        )
        if shard["contract"]["rank"] != rank:
            raise ValueError(f"visual feature shard {rank} rank differs")
        candidate_common = _common_contract(shard["contract"])
        if common is None:
            common = candidate_common
        elif candidate_common != common:
            raise ValueError("eight visual feature shard contracts differ")
        shards.append(shard)
    if common is None:  # pragma: no cover - exact eight paths prevent this.
        raise ValueError("no visual feature shards")
    arrays = _empty_arrays(len(input_rows))
    merged_rows: list[dict[str, Any] | None] = [None] * len(input_rows)
    seen: set[int] = set()
    for rank, shard in enumerate(shards):
        shard_indices = np.asarray(
            shard["arrays"]["input_indices"],
            dtype=np.int64,
        ).tolist()
        expected = [
            index
            for index in range(len(input_rows))
            if index % FINAL_WORLD_SIZE == rank
        ]
        if shard_indices != expected:
            raise ValueError(
                f"visual feature shard {rank} coverage differs"
            )
        for local_index, input_index in enumerate(shard_indices):
            if input_index in seen:
                raise ValueError(
                    f"visual feature input index {input_index} is duplicated"
                )
            seen.add(input_index)
            merged_rows[input_index] = dict(
                shard["rows"][local_index]
            )
            for name in arrays:
                arrays[name][input_index] = shard["arrays"][name][local_index]
    if seen != set(range(len(input_rows))) or any(
        row is None for row in merged_rows
    ):
        raise ValueError(
            "eight visual feature shards do not exactly cover the input"
        )
    rows = [dict(row) for row in merged_rows if row is not None]
    _validate_array_contract(arrays, rows=len(rows))
    data_root = Path(str(common["data_root"])).resolve(strict=True)
    _validate_feature_rows(
        rows,
        arrays,
        data_root=data_root,
        rehash_videos=False,
    )
    _crosscheck_final_against_shards(
        final_rows=rows,
        final_arrays=arrays,
        shards=shards,
    )
    statistics = _failure_statistics(rows)
    common_digest = _object_digest(common)
    summary = {
        "schema_version": FINAL_SUMMARY_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "input_rows": len(input_rows),
        "input_manifest_sha256": _file_digest(manifest_path),
        "world_size": FINAL_WORLD_SIZE,
        "partition": PARTITION_VERSION,
        "extraction_status": statistics["extraction_status"],
        "split_ready": statistics["split_ready"],
        "split_fail_closed_reason": (
            None
            if statistics["split_ready"]
            else "one_or_more_video_sides_failed_feature_extraction"
        ),
        "statistics": statistics,
        "common_contract": common,
        "common_contract_sha256": common_digest,
        "shard_done_sha256": [
            _file_digest(path / DONE_NAME) for path in shard_paths
        ],
        "array_contract": _array_contract(arrays),
        "cotracker_executed": False,
    }
    done_base = {
        "schema_version": FINAL_DONE_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "world_size": FINAL_WORLD_SIZE,
        "common_contract_sha256": common_digest,
        "split_ready": statistics["split_ready"],
    }
    return _atomic_commit(
        directory=final_directory,
        rows=rows,
        arrays=arrays,
        summary=summary,
        done_base=done_base,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hash-bound DINO/dHash features for unsplit R7 "
            "expansion candidates."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--input-manifest", required=True, type=Path)
    extract.add_argument("--data-root", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--dinov2-model", required=True, type=Path)
    extract.add_argument("--dinov2-revision")
    extract.add_argument("--rank", type=int)
    extract.add_argument("--world-size", type=int)
    extract.add_argument("--local-rank", type=int)
    extract.add_argument("--resume", action="store_true")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--input-manifest", required=True, type=Path)
    finalize.add_argument("--output-dir", required=True, type=Path)
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
        done = extract_rank(
            input_manifest=args.input_manifest,
            data_root=args.data_root,
            output_root=args.output_dir,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            dino_model_root=args.dinov2_model,
            dino_revision=args.dinov2_revision,
            resume=bool(args.resume),
        )
    else:
        done = finalize_shards(
            input_manifest=args.input_manifest,
            output_root=args.output_dir,
            resume=bool(args.resume),
        )
    print(_canonical_json(done), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_NAMES",
    "FINAL_DONE_SCHEMA",
    "FINAL_SUMMARY_SCHEMA",
    "FINAL_WORLD_SIZE",
    "ROW_SCHEMA",
    "SCHEMA_VERSION",
    "SHARD_DONE_SCHEMA",
    "SHARD_SUMMARY_SCHEMA",
    "extract_rank",
    "finalize_shards",
    "load_candidate_manifest",
    "main",
    "rank_directory",
    "validate_final",
    "validate_shard",
]

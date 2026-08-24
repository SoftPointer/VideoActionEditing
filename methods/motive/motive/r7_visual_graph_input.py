"""Consolidate the frozen R7 visual features into one graph input.

The graph builder needs identity features from two independently committed
extractors:

* the unsplit expansion candidates produced by
  :mod:`r7_expansion_visual_features`; and
* the historical R7-P0 pilot produced by :mod:`r7_preflight_extract`.

This module does not assign a split, assert a human label, or authorize
training.  It validates both exact-eight-shard commits, requires a valid
source/target DINO pair for every IID, proves that the frozen DINO contracts
are compatible, and emits a small canonical asset table plus one feature
matrix.  Publication is an atomic, no-overwrite directory rename.
``--resume`` is verification-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import r7_expansion_visual_features as expansion_features
from . import r7_preflight_extract as preflight


SCHEMA_VERSION = "motive-r7-visual-graph-input-v1"
ROW_SCHEMA = "motive-r7-visual-graph-input-row-v1"
SUMMARY_SCHEMA = "motive-r7-visual-graph-input-summary-v1"
DONE_SCHEMA = "motive-r7-visual-graph-input-done-v1"

MANIFEST_NAME = "manifest.jsonl"
ARCHIVE_NAME = "features.npz"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ARTIFACT_NAMES = frozenset(
    {MANIFEST_NAME, ARCHIVE_NAME, SUMMARY_NAME, DONE_NAME}
)
SOURCE_ARTIFACT_NAMES = frozenset(
    {
        expansion_features.MANIFEST_NAME,
        expansion_features.ARCHIVE_NAME,
        expansion_features.SUMMARY_NAME,
        expansion_features.DONE_NAME,
    }
)
ROLES = ("source", "target")
ROW_FIELDS = frozenset(
    {
        "schema_version",
        "asset_index",
        "iid",
        "role",
        "anchor",
        "cohort",
        "video_sha256",
        "dhashes",
        "source_artifact_digest",
        "source_input_index",
        "source_index_digest",
    }
)
DINO_COMPARISON_FIELDS = (
    "encoder_id",
    "encoder_revision",
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
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DHASH_RE = re.compile(r"^[0-9a-f]{16}$")


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


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON: {error}") from error
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = _parse_json(path.read_bytes(), context=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path} must be non-empty and newline-terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number} is blank")
        value = _parse_json(
            line,
            context=f"{path}:{line_number}",
        )
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        if line.decode("utf-8") != _canonical_json(value):
            raise ValueError(f"{path}:{line_number} is not canonical JSON")
        rows.append(value)
    return rows


def _artifact_paths(directory: Path) -> dict[str, Path]:
    return {
        "manifest": directory / MANIFEST_NAME,
        "archive": directory / ARCHIVE_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _strict_directory(
    directory: Path,
    *,
    expected_names: set[str] | frozenset[str],
) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(directory)
    actual = {entry.name for entry in directory.iterdir()}
    if actual != set(expected_names):
        raise ValueError(
            f"{directory} artifact set differs: "
            f"missing={sorted(set(expected_names) - actual)}, "
            f"extra={sorted(actual - set(expected_names))}"
        )
    for name in expected_names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source artifact is not a regular file: {path}")


def _source_paths(
    *,
    final_directory: Path,
    input_manifest: Path,
) -> dict[str, Path]:
    final = final_directory.expanduser().resolve(strict=True)
    manifest = input_manifest.expanduser().resolve(strict=True)
    _strict_directory(final, expected_names=SOURCE_ARTIFACT_NAMES)
    root = final.parent
    shards = root / "shards"
    if shards.is_symlink() or not shards.is_dir():
        raise FileNotFoundError(shards)
    expected_shards = {
        f"rank-{rank:03d}-of-{preflight.FINAL_WORLD_SIZE:03d}"
        for rank in range(preflight.FINAL_WORLD_SIZE)
    }
    actual_shards = {entry.name for entry in shards.iterdir()}
    if actual_shards != expected_shards:
        raise ValueError(
            f"{shards} is not an exact-eight shard set: "
            f"missing={sorted(expected_shards - actual_shards)}, "
            f"extra={sorted(actual_shards - expected_shards)}"
        )
    paths = {"input_manifest": manifest}
    for name in sorted(SOURCE_ARTIFACT_NAMES):
        paths[f"final/{name}"] = final / name
    for rank in range(preflight.FINAL_WORLD_SIZE):
        shard = shards / (
            f"rank-{rank:03d}-of-{preflight.FINAL_WORLD_SIZE:03d}"
        )
        _strict_directory(shard, expected_names=SOURCE_ARTIFACT_NAMES)
        for name in sorted(SOURCE_ARTIFACT_NAMES):
            paths[f"shards/{rank:03d}/{name}"] = shard / name
    return paths


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source artifact changed type: {path}")
        stat = path.stat()
        snapshot[name] = {
            "sha256": _file_digest(path),
            "bytes": int(stat.st_size),
        }
    return snapshot


def _assert_snapshot(
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    current = _snapshot(paths)
    if current != {
        key: dict(value) for key, value in sorted(expected.items())
    }:
        raise RuntimeError("source artifacts changed during consolidation")


def _source_artifact_digest(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> str:
    return _object_digest(
        {
            key: {
                "sha256": value["sha256"],
                "bytes": value["bytes"],
            }
            for key, value in sorted(snapshot.items())
        }
    )


def _validate_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _validate_dhashes(value: Any, *, context: str) -> list[str]:
    if (
        not isinstance(value, (list, tuple, np.ndarray))
        or len(value) != preflight.DINO_FRAMES
    ):
        raise ValueError(f"{context} must contain six dHashes")
    result = [str(item) for item in value]
    if any(_DHASH_RE.fullmatch(item) is None for item in result):
        raise ValueError(f"{context} contains an invalid 64-bit dHash")
    return result


def _validate_dino_contract(raw: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} DINO provenance is not an object")
    provenance = dict(raw)
    expected_fields = set(DINO_COMPARISON_FIELDS) | {"resolved_path"}
    if set(provenance) != expected_fields:
        raise ValueError(
            f"{context} DINO provenance field set differs: "
            f"missing={sorted(expected_fields - set(provenance))}, "
            f"extra={sorted(set(provenance) - expected_fields)}"
        )
    selected = {
        field: provenance[field] for field in DINO_COMPARISON_FIELDS
    }
    _validate_committed_dino_contract(selected, context=context)
    if (
        type(provenance["resolved_path"]) is not str
        or not Path(provenance["resolved_path"]).is_absolute()
    ):
        raise ValueError(f"{context} DINO provenance is incompatible")
    return selected


def _validate_committed_dino_contract(
    raw: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate the path-independent 13-field DINO graph contract."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} DINO contract is not an object")
    selected = dict(raw)
    if set(selected) != set(DINO_COMPARISON_FIELDS):
        raise ValueError(
            f"{context} DINO contract field set differs: "
            f"missing={sorted(set(DINO_COMPARISON_FIELDS) - set(selected))}, "
            f"extra={sorted(set(selected) - set(DINO_COMPARISON_FIELDS))}"
        )
    if (
        selected["encoder_id"] != "facebook/dinov2-base"
        or type(selected["encoder_revision"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{7,64}",
            selected["encoder_revision"],
        )
        is None
        or any(
            type(selected[field]) is not str
            or _SHA256_RE.fullmatch(selected[field]) is None
            for field in ("model_tree_sha256", "weights_sha256")
        )
        or type(selected["model_file_count"]) is not int
        or selected["model_file_count"] < 1
        or selected["frame_sampling_version"]
        != preflight.R7_DINO_SAMPLING
        or selected["preprocessing_version"]
        != preflight.R7_DINO_PREPROCESSING
        or selected["pooling"] != preflight.R7_DINO_POOLING
        or selected["embedding_dim"] != preflight.DINO_DIM
        or selected["dtype"] != "float32"
        or selected["normalization"] != "l2-per-frame"
        or selected["frozen_encoder"] is not True
        or selected["local_files_only"] is not True
    ):
        raise ValueError(f"{context} DINO contract is incompatible")
    _canonical_json(selected)
    return selected


def _without_rank_device(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(contract)
    result.pop("rank", None)
    result.pop("device", None)
    return result


def _validate_anchor_exact_eight(
    *,
    final_directory: Path,
    input_manifest: Path,
    final: Mapping[str, Any],
    paths: Mapping[str, Path],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Close validation gaps in the historical final validator."""

    summary = final.get("summary")
    rows = final.get("rows")
    arrays = final.get("arrays")
    if (
        not isinstance(summary, Mapping)
        or not isinstance(rows, list)
        or not isinstance(arrays, Mapping)
    ):
        raise ValueError("historical validate_final contract differs")
    if (
        summary.get("archive_sha256")
        != snapshot[f"final/{ARCHIVE_NAME}"]["sha256"]
        or summary.get("manifest_sha256")
        != snapshot[f"final/{MANIFEST_NAME}"]["sha256"]
    ):
        raise ValueError("historical final summary hash chain differs")
    expected_done_hashes = [
        snapshot[f"shards/{rank:03d}/{DONE_NAME}"]["sha256"]
        for rank in range(preflight.FINAL_WORLD_SIZE)
    ]
    if summary.get("shard_done_sha256") != expected_done_hashes:
        raise ValueError(
            "historical final is not bound to the exact source shards"
        )
    input_rows = preflight._read_r5_manifest(input_manifest)
    if len(input_rows) != len(rows):
        raise ValueError("historical final/input row count differs")
    shards: list[dict[str, Any]] = []
    common_contract: dict[str, Any] | None = None
    for rank in range(preflight.FINAL_WORLD_SIZE):
        shard_dir = final_directory.parent / "shards" / (
            f"rank-{rank:03d}-of-{preflight.FINAL_WORLD_SIZE:03d}"
        )
        shard = preflight.validate_shard(
            shard_dir,
            input_manifest=input_manifest,
            rehash_videos=True,
        )
        contract = shard.get("contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"historical shard {rank} lacks a contract")
        if (
            contract.get("rank") != rank
            or contract.get("world_size") != preflight.FINAL_WORLD_SIZE
        ):
            raise ValueError(f"historical shard {rank} rank differs")
        candidate_common = _without_rank_device(contract)
        if common_contract is None:
            common_contract = candidate_common
        elif candidate_common != common_contract:
            raise ValueError("historical exact-eight contracts differ")
        shards.append(shard)
    if common_contract is None:  # pragma: no cover
        raise ValueError("historical exact-eight set is empty")
    if common_contract.get("input_manifest_sha256") != _file_digest(
        input_manifest
    ):
        raise ValueError("historical contract/input manifest differs")
    if common_contract.get("video_sampling") != {
        "version": preflight.R7_VIDEO_SAMPLING,
        "frames": preflight.VIDEO_FRAMES,
        "maximum_side": preflight.MAX_VIDEO_SIDE,
    }:
        raise ValueError("historical video-sampling contract differs")

    final_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    seen: set[int] = set()
    for rank, shard in enumerate(shards):
        shard_rows = shard["rows"]
        shard_arrays = shard["arrays"]
        indices = np.asarray(
            shard_arrays["input_indices"],
            dtype=np.int64,
        ).tolist()
        expected = [
            index
            for index in range(len(input_rows))
            if index % preflight.FINAL_WORLD_SIZE == rank
        ]
        if indices != expected:
            raise ValueError(
                f"historical shard {rank} coverage is incomplete"
            )
        if set(shard_arrays) != set(final_arrays):
            raise ValueError(
                f"historical shard {rank} archive schema differs"
            )
        for local_index, input_index in enumerate(indices):
            if input_index in seen:
                raise ValueError("historical shard coverage overlaps")
            seen.add(input_index)
            final_row = dict(rows[input_index])
            merged_index = final_row.pop("merged_array_index", None)
            if merged_index != input_index:
                raise ValueError(
                    f"historical final row {input_index} merge index differs"
                )
            if final_row != dict(shard_rows[local_index]):
                raise ValueError(
                    f"historical final row {input_index} differs from shard"
                )
            for name in final_arrays:
                if not np.array_equal(
                    final_arrays[name][input_index],
                    np.asarray(shard_arrays[name])[local_index],
                ):
                    raise ValueError(
                        f"historical final array {name} row "
                        f"{input_index} differs from shard"
                    )
    if seen != set(range(len(input_rows))):
        raise ValueError("historical exact-eight coverage is incomplete")
    _assert_snapshot(paths, snapshot)
    return input_rows, common_contract


@dataclass(frozen=True)
class _SourceBundle:
    kind: str
    anchor: bool
    final_directory: Path
    input_manifest: Path
    input_rows: tuple[Mapping[str, Any], ...]
    feature_rows: tuple[Mapping[str, Any], ...]
    arrays: Mapping[str, np.ndarray]
    dino_contract: Mapping[str, Any]
    paths: Mapping[str, Path]
    snapshot: Mapping[str, Mapping[str, Any]]
    artifact_digest: str


def _load_candidate_bundle(
    *,
    final_directory: Path,
    input_manifest: Path,
) -> _SourceBundle:
    paths = _source_paths(
        final_directory=final_directory,
        input_manifest=input_manifest,
    )
    before = _snapshot(paths)
    result = expansion_features.validate_final(
        final_directory,
        input_manifest=input_manifest,
        output_root=final_directory.parent,
        verify_source_shards=True,
        rehash_videos=True,
    )
    after = _snapshot(paths)
    if after != before:
        raise RuntimeError(
            "candidate artifacts changed while validate_final was reading"
        )
    source_shards = result.get("source_shards")
    if not isinstance(source_shards, list) or len(source_shards) != 8:
        raise ValueError("candidate validate_final did not return exact8 shards")
    input_rows = expansion_features.load_candidate_manifest(input_manifest)
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("candidate validate_final summary differs")
    common = summary.get("common_contract")
    if not isinstance(common, Mapping):
        raise ValueError("candidate common contract is missing")
    dino = _validate_dino_contract(
        common.get("dino"),
        context="candidate",
    )
    rows = result.get("rows")
    arrays = result.get("arrays")
    if not isinstance(rows, list) or not isinstance(arrays, Mapping):
        raise ValueError("candidate validate_final return contract differs")
    return _SourceBundle(
        kind="candidate",
        anchor=False,
        final_directory=final_directory,
        input_manifest=input_manifest,
        input_rows=tuple(input_rows),
        feature_rows=tuple(rows),
        arrays={name: np.asarray(value) for name, value in arrays.items()},
        dino_contract=dino,
        paths=paths,
        snapshot=after,
        artifact_digest=_source_artifact_digest(after),
    )


def _load_anchor_bundle(
    *,
    final_directory: Path,
    input_manifest: Path,
) -> _SourceBundle:
    paths = _source_paths(
        final_directory=final_directory,
        input_manifest=input_manifest,
    )
    before = _snapshot(paths)
    result = preflight.validate_final(
        final_directory,
        input_manifest=input_manifest,
    )
    input_rows, common = _validate_anchor_exact_eight(
        final_directory=final_directory,
        input_manifest=input_manifest,
        final=result,
        paths=paths,
        snapshot=before,
    )
    after = _snapshot(paths)
    if after != before:
        raise RuntimeError(
            "historical artifacts changed while validators were reading"
        )
    dino = _validate_dino_contract(
        common.get("dino"),
        context="historical anchor",
    )
    rows = result.get("rows")
    arrays = result.get("arrays")
    if not isinstance(rows, list) or not isinstance(arrays, Mapping):
        raise ValueError("historical validate_final return contract differs")
    return _SourceBundle(
        kind="anchor",
        anchor=True,
        final_directory=final_directory,
        input_manifest=input_manifest,
        input_rows=tuple(input_rows),
        feature_rows=tuple(rows),
        arrays={name: np.asarray(value) for name, value in arrays.items()},
        dino_contract=dino,
        paths=paths,
        snapshot=after,
        artifact_digest=_source_artifact_digest(after),
    )


def _cohort(bundle: _SourceBundle, index: int) -> str:
    row = bundle.input_rows[index]
    if bundle.anchor:
        label = row.get("r5_pilot_label")
        if not isinstance(label, Mapping) or label.get("class") not in {
            "positive",
            "negative",
        }:
            raise ValueError(
                f"historical input row {index} has no valid pilot class"
            )
        return f"anchor_{label['class']}"
    cohort = row.get("cohort")
    if cohort not in {"pseudo_positive", "pseudo_negative"}:
        raise ValueError(f"candidate input row {index} cohort differs")
    return str(cohort)


def _project_bundle(
    bundle: _SourceBundle,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    if len(bundle.feature_rows) != len(bundle.input_rows):
        raise ValueError(f"{bundle.kind} feature/input row count differs")
    arrays = bundle.arrays
    required = {
        "input_indices",
        "source_dino_cls",
        "target_dino_cls",
    }
    if bundle.anchor:
        required |= {
            "source_dino_valid",
            "target_dino_valid",
            "source_perceptual_hashes",
            "target_perceptual_hashes",
        }
    else:
        required |= {
            "source_valid",
            "target_valid",
            "source_difference_hashes",
            "target_difference_hashes",
            "source_video_sha256",
            "target_video_sha256",
        }
    missing = required - set(arrays)
    if missing:
        raise ValueError(
            f"{bundle.kind} archive misses {sorted(missing)}"
        )
    indices = np.asarray(arrays["input_indices"])
    if indices.dtype != np.int64 or indices.tolist() != list(
        range(len(bundle.input_rows))
    ):
        raise ValueError(f"{bundle.kind} input indices differ")
    projected: list[tuple[dict[str, Any], np.ndarray]] = []
    seen: set[str] = set()
    for index, (input_row, feature_row) in enumerate(
        zip(bundle.input_rows, bundle.feature_rows)
    ):
        iid = input_row.get("iid")
        if (
            type(iid) is not str
            or not iid
            or iid.strip() != iid
            or "\x00" in iid
            or iid in seen
            or feature_row.get("iid") != iid
            or feature_row.get("input_index") != index
        ):
            raise ValueError(
                f"{bundle.kind} row {index} identity binding differs"
            )
        seen.add(iid)
        input_row_digest = _object_digest(input_row)
        if feature_row.get("input_row_sha256") != input_row_digest:
            raise ValueError(
                f"{bundle.kind} row {index} input digest differs"
            )
        cohort = _cohort(bundle, index)
        for role in ROLES:
            side = feature_row.get(role)
            if not isinstance(side, Mapping):
                raise ValueError(
                    f"{bundle.kind} {iid}/{role} record differs"
                )
            if bundle.anchor:
                valid = arrays[f"{role}_dino_valid"][index]
                hashes = arrays[f"{role}_perceptual_hashes"][index]
                row_valid = side.get("dino_valid")
                decode = side.get("decode")
                row_hashes = (
                    decode.get("perceptual_hashes")
                    if isinstance(decode, Mapping)
                    else None
                )
                if (
                    not isinstance(decode, Mapping)
                    or decode.get("sampling_version")
                    != preflight.R7_VIDEO_SAMPLING
                    or decode.get("decoded_frames")
                    != preflight.VIDEO_FRAMES
                    or decode.get("dino_frame_offsets")
                    != preflight.dino_frame_offsets().tolist()
                ):
                    raise ValueError(
                        f"{bundle.kind} {iid}/{role} sampling differs"
                    )
            else:
                valid = arrays[f"{role}_valid"][index]
                hashes = arrays[f"{role}_difference_hashes"][index]
                row_valid = side.get("valid")
                row_hashes = (
                    side.get("decode", {}).get("difference_hashes")
                    if isinstance(side.get("decode"), Mapping)
                    else None
                )
            if (
                not isinstance(valid, (bool, np.bool_))
                or not bool(valid)
                or row_valid is not True
            ):
                raise ValueError(
                    f"{bundle.kind} {iid}/{role} is not DINO-valid"
                )
            dhashes = _validate_dhashes(
                hashes,
                context=f"{bundle.kind} {iid}/{role}",
            )
            if row_hashes != dhashes:
                raise ValueError(
                    f"{bundle.kind} {iid}/{role} row/archive dHash differs"
                )
            video_sha256 = _validate_sha256(
                side.get("video_sha256"),
                context=f"{bundle.kind} {iid}/{role} video SHA-256",
            )
            if not bundle.anchor:
                archive_video_sha = str(
                    arrays[f"{role}_video_sha256"][index]
                )
                if archive_video_sha != video_sha256:
                    raise ValueError(
                        f"{bundle.kind} {iid}/{role} video hash differs"
                    )
            source_index_digest = _object_digest(
                {
                    "source_artifact_digest": bundle.artifact_digest,
                    "source_input_index": index,
                    "source_input_row_sha256": input_row_digest,
                    "iid": iid,
                    "role": role,
                    "video_sha256": video_sha256,
                    "dhashes": dhashes,
                }
            )
            feature = np.asarray(arrays[f"{role}_dino_cls"][index])
            if (
                feature.shape
                != (preflight.DINO_FRAMES, preflight.DINO_DIM)
                or feature.dtype != np.float32
                or not np.isfinite(feature).all()
            ):
                raise ValueError(
                    f"{bundle.kind} {iid}/{role} DINO tensor differs"
                )
            norms = np.linalg.norm(feature.astype(np.float64), axis=1)
            if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
                raise ValueError(
                    f"{bundle.kind} {iid}/{role} DINO is not L2-normalized"
                )
            projected.append(
                (
                    {
                        "schema_version": ROW_SCHEMA,
                        "asset_index": -1,
                        "iid": iid,
                        "role": role,
                        "anchor": bundle.anchor,
                        "cohort": cohort,
                        "video_sha256": video_sha256,
                        "dhashes": dhashes,
                        "source_artifact_digest": bundle.artifact_digest,
                        "source_input_index": index,
                        "source_index_digest": source_index_digest,
                    },
                    np.ascontiguousarray(feature),
                )
            )
    return projected


@dataclass(frozen=True)
class _Derived:
    rows: tuple[Mapping[str, Any], ...]
    arrays: Mapping[str, np.ndarray]
    summary_base: Mapping[str, Any]
    paths: Mapping[str, Path]
    snapshot: Mapping[str, Mapping[str, Any]]


def _derive(
    *,
    candidate_features_dir: Path,
    candidate_manifest: Path,
    anchor_features_dir: Path,
    anchor_input_manifest: Path,
) -> _Derived:
    candidate_final = candidate_features_dir.expanduser().resolve(
        strict=True
    )
    candidate_input = candidate_manifest.expanduser().resolve(strict=True)
    anchor_final = anchor_features_dir.expanduser().resolve(strict=True)
    anchor_input = anchor_input_manifest.expanduser().resolve(strict=True)
    candidate = _load_candidate_bundle(
        final_directory=candidate_final,
        input_manifest=candidate_input,
    )
    anchor = _load_anchor_bundle(
        final_directory=anchor_final,
        input_manifest=anchor_input,
    )
    if dict(candidate.dino_contract) != dict(anchor.dino_contract):
        differences = [
            field
            for field in DINO_COMPARISON_FIELDS
            if candidate.dino_contract[field]
            != anchor.dino_contract[field]
        ]
        raise ValueError(
            "candidate/anchor DINO contracts differ: "
            + ", ".join(differences)
        )
    candidate_iids = sorted(
        str(row["iid"]) for row in candidate.input_rows
    )
    anchor_iids = sorted(str(row["iid"]) for row in anchor.input_rows)
    overlap = sorted(set(candidate_iids) & set(anchor_iids))
    if overlap:
        raise ValueError(
            f"candidate/anchor IID sets overlap: {overlap[:5]}"
        )
    projected = _project_bundle(candidate) + _project_bundle(anchor)
    role_order = {"source": 0, "target": 1}
    projected.sort(
        key=lambda item: (
            str(item[0]["iid"]),
            role_order[str(item[0]["role"])],
        )
    )
    rows: list[dict[str, Any]] = []
    features = np.zeros(
        (len(projected), preflight.DINO_FRAMES, preflight.DINO_DIM),
        dtype=np.float32,
    )
    for asset_index, (raw_row, feature) in enumerate(projected):
        row = dict(raw_row)
        row["asset_index"] = asset_index
        if set(row) != set(ROW_FIELDS):
            raise RuntimeError("internal graph row field set differs")
        rows.append(row)
        features[asset_index] = feature
    expected_assets = 2 * (len(candidate_iids) + len(anchor_iids))
    if len(rows) != expected_assets:
        raise RuntimeError("graph input does not contain paired assets")
    arrays = {
        "asset_indices": np.arange(len(rows), dtype=np.int64),
        "dino_cls": np.ascontiguousarray(features),
    }
    paths = {
        **{
            f"candidate/{key}": value
            for key, value in candidate.paths.items()
        },
        **{
            f"anchor/{key}": value
            for key, value in anchor.paths.items()
        },
    }
    snapshot = {
        **{
            f"candidate/{key}": value
            for key, value in candidate.snapshot.items()
        },
        **{
            f"anchor/{key}": value
            for key, value in anchor.snapshot.items()
        },
    }
    _assert_snapshot(paths, snapshot)
    summary_base = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "assets": len(rows),
        "iids": len(candidate_iids) + len(anchor_iids),
        "candidate_iids": {
            "count": len(candidate_iids),
            "sha256": _object_digest(candidate_iids),
        },
        "anchor_iids": {
            "count": len(anchor_iids),
            "sha256": _object_digest(anchor_iids),
        },
        "asset_order": "lexicographic-iid-source-before-target-v1",
        "source_artifacts": {
            "candidate": {
                "artifact_digest": candidate.artifact_digest,
                "input_manifest_sha256":
                    candidate.snapshot["input_manifest"]["sha256"],
                "final_done_sha256":
                    candidate.snapshot[f"final/{DONE_NAME}"]["sha256"],
            },
            "anchor": {
                "artifact_digest": anchor.artifact_digest,
                "input_manifest_sha256":
                    anchor.snapshot["input_manifest"]["sha256"],
                "final_done_sha256":
                    anchor.snapshot[f"final/{DONE_NAME}"]["sha256"],
            },
        },
        "dino_contract": dict(candidate.dino_contract),
        "split_assigned": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    return _Derived(
        rows=tuple(rows),
        arrays=arrays,
        summary_base=summary_base,
        paths=paths,
        snapshot=snapshot,
    )


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


def _validate_output_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: int,
) -> None:
    if set(arrays) != {"asset_indices", "dino_cls"}:
        raise ValueError("graph input archive array set differs")
    indices = np.asarray(arrays["asset_indices"])
    features = np.asarray(arrays["dino_cls"])
    if (
        indices.dtype != np.int64
        or indices.shape != (rows,)
        or indices.tolist() != list(range(rows))
    ):
        raise ValueError("graph input asset_indices differs")
    if (
        features.dtype != np.float32
        or features.shape
        != (rows, preflight.DINO_FRAMES, preflight.DINO_DIM)
        or not np.isfinite(features).all()
    ):
        raise ValueError("graph input DINO tensor contract differs")
    norms = np.linalg.norm(features.astype(np.float64), axis=2)
    if not np.allclose(norms, 1.0, atol=2e-4, rtol=2e-4):
        raise ValueError("graph input DINO tensor is not L2-normalized")


def _load_output(directory: Path) -> tuple[
    dict[str, Path],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
]:
    requested = directory.expanduser()
    if requested.is_symlink():
        raise ValueError(f"graph input directory is a symlink: {requested}")
    target = requested.resolve(strict=True)
    _strict_directory(target, expected_names=ARTIFACT_NAMES)
    paths = _artifact_paths(target)
    done = _load_json(paths["done"])
    if (
        set(done)
        != {
            "schema_version",
            "status",
            "assets",
            "iids",
            "source_artifact_digests",
            "split_assigned",
            "human_labels_asserted",
            "training_authorized",
            "artifacts",
            "artifact_digest",
        }
        or done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("split_assigned") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_authorized") is not False
    ):
        raise ValueError("invalid graph input done marker")
    registry = done.get("artifacts")
    if not isinstance(registry, Mapping) or set(registry) != {
        "manifest",
        "archive",
        "summary",
    }:
        raise ValueError("graph input done artifact registry differs")
    output_sha: dict[str, str] = {}
    for name in ("manifest", "archive", "summary"):
        record = registry[name]
        path = paths[name]
        digest = _file_digest(path)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"filename", "sha256"}
            or record.get("filename") != path.name
            or record.get("sha256") != digest
        ):
            raise ValueError(f"graph input {name} digest differs")
        output_sha[name] = digest
    if done.get("artifact_digest") != _object_digest(output_sha):
        raise ValueError("graph input done artifact digest differs")
    rows = _load_canonical_jsonl(paths["manifest"])
    with np.load(paths["archive"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    _validate_output_arrays(arrays, rows=len(rows))
    summary = _load_json(paths["summary"])
    if paths["summary"].read_bytes() != _pretty_json_bytes(summary):
        raise ValueError("graph input summary is not canonical")
    if paths["done"].read_bytes() != _pretty_json_bytes(done):
        raise ValueError("graph input done marker is not canonical")
    return paths, rows, arrays, summary, done


def _validate_commit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> None:
    if not rows or len(rows) % len(ROLES) != 0:
        raise ValueError(
            "graph input manifest must contain non-empty source/target pairs"
        )
    expected_keys: list[tuple[str, int]] = []
    seen_assets: set[tuple[str, str]] = set()
    paired_iids: set[str] = set()
    seen_source_indices: set[tuple[bool, int]] = set()
    candidate_iids: list[str] = []
    anchor_iids: list[str] = []
    source_artifacts = summary.get("source_artifacts")
    if (
        not isinstance(source_artifacts, Mapping)
        or set(source_artifacts) != {"candidate", "anchor"}
    ):
        raise ValueError("graph input source artifact summary differs")
    source_digests: dict[bool, str] = {}
    for name, anchor in (("candidate", False), ("anchor", True)):
        record = source_artifacts.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "artifact_digest",
                "input_manifest_sha256",
                "final_done_sha256",
            }
        ):
            raise ValueError(
                f"graph input {name} source artifact record differs"
            )
        source_digests[anchor] = _validate_sha256(
            record.get("artifact_digest"),
            context=f"graph input {name} artifact digest",
        )
        _validate_sha256(
            record.get("input_manifest_sha256"),
            context=f"graph input {name} input manifest digest",
        )
        _validate_sha256(
            record.get("final_done_sha256"),
            context=f"graph input {name} final done digest",
        )

    for pair_offset in range(0, len(rows), len(ROLES)):
        pair = rows[pair_offset : pair_offset + len(ROLES)]
        first = pair[0]
        iid = first.get("iid")
        if (
            type(iid) is not str
            or not iid
            or iid.strip() != iid
            or "\x00" in iid
            or iid in paired_iids
        ):
            raise ValueError(
                f"graph input pair {pair_offset // 2} IID differs"
            )
        paired_iids.add(iid)
        anchor = first.get("anchor")
        cohort = first.get("cohort")
        source_index = first.get("source_input_index")
        if (
            type(anchor) is not bool
            or type(cohort) is not str
            or cohort
            not in {
                "pseudo_positive",
                "pseudo_negative",
                "anchor_positive",
                "anchor_negative",
            }
            or anchor != cohort.startswith("anchor_")
            or type(source_index) is not int
            or source_index < 0
            or (anchor, source_index) in seen_source_indices
        ):
            raise ValueError(f"graph input {iid} pair metadata differs")
        seen_source_indices.add((anchor, source_index))
        if anchor:
            anchor_iids.append(iid)
        else:
            candidate_iids.append(iid)
        for role_offset, (row, expected_role) in enumerate(
            zip(pair, ROLES)
        ):
            index = pair_offset + role_offset
            if (
                set(row) != set(ROW_FIELDS)
                or row.get("schema_version") != ROW_SCHEMA
                or type(row.get("asset_index")) is not int
                or row["asset_index"] != index
                or row.get("iid") != iid
                or row.get("role") != expected_role
                or row.get("anchor") is not anchor
                or row.get("cohort") != cohort
                or row.get("source_input_index") != source_index
                or row.get("source_artifact_digest")
                != source_digests[anchor]
            ):
                raise ValueError(
                    f"graph input row {index} pair/schema binding differs"
                )
            _validate_sha256(
                row.get("video_sha256"),
                context=f"graph input row {index} video digest",
            )
            _validate_dhashes(
                row.get("dhashes"),
                context=f"graph input row {index}",
            )
            _validate_sha256(
                row.get("source_index_digest"),
                context=f"graph input row {index} source index digest",
            )
            identity = (iid, expected_role)
            if identity in seen_assets:
                raise ValueError(f"duplicate graph asset identity: {identity}")
            seen_assets.add(identity)
            expected_keys.append((iid, role_offset))
    if expected_keys != sorted(expected_keys):
        raise ValueError(
            "graph input rows are not lexicographic IID/source-target ordered"
        )
    if candidate_iids != sorted(candidate_iids) or anchor_iids != sorted(
        anchor_iids
    ):
        raise ValueError("graph input cohort IID order differs")
    for name, values in (
        ("candidate_iids", candidate_iids),
        ("anchor_iids", anchor_iids),
    ):
        record = summary.get(name)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"count", "sha256"}
            or type(record.get("count")) is not int
            or record.get("count") != len(values)
            or record.get("sha256") != _object_digest(values)
        ):
            raise ValueError(f"graph input {name} summary differs")


def validate_graph_input_commit(directory: Path) -> dict[str, Any]:
    """Validate a graph-input commit without consulting mutable source trees.

    This is the sole lightweight consumer contract for downstream graph
    construction.  The full :func:`validate_graph_input` additionally
    rederives these exact rows and arrays from both frozen extractor trees.
    """

    paths, rows, arrays, summary, done = _load_output(directory)
    expected_summary_fields = {
        "schema_version",
        "status",
        "assets",
        "iids",
        "candidate_iids",
        "anchor_iids",
        "asset_order",
        "source_artifacts",
        "dino_contract",
        "split_assigned",
        "human_labels_asserted",
        "training_authorized",
        "array_contract",
        "archive_sha256",
        "manifest_sha256",
    }
    if (
        set(summary) != expected_summary_fields
        or summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("asset_order")
        != "lexicographic-iid-source-before-target-v1"
        or summary.get("split_assigned") is not False
        or summary.get("human_labels_asserted") is not False
        or summary.get("training_authorized") is not False
    ):
        raise ValueError("graph input commit summary semantics differ")
    _validate_output_arrays(arrays, rows=len(rows))
    if (
        type(summary.get("assets")) is not int
        or summary.get("assets") != len(rows)
        or type(summary.get("iids")) is not int
        or summary["iids"] * len(ROLES) != len(rows)
        or summary.get("array_contract") != _array_contract(arrays)
        or summary.get("archive_sha256")
        != _file_digest(paths["archive"])
        or summary.get("manifest_sha256")
        != _file_digest(paths["manifest"])
    ):
        raise ValueError("graph input commit array/count/hash binding differs")
    dino_contract = _validate_committed_dino_contract(
        summary.get("dino_contract"),
        context="graph input commit",
    )
    _validate_commit_rows(rows, summary=summary)
    source_artifacts = summary["source_artifacts"]
    expected_source_digests = {
        name: record["artifact_digest"]
        for name, record in source_artifacts.items()
    }
    if (
        type(done.get("assets")) is not int
        or done.get("assets") != len(rows)
        or type(done.get("iids")) is not int
        or done.get("iids") != summary["iids"]
        or done.get("source_artifact_digests")
        != expected_source_digests
        or done.get("split_assigned") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_authorized") is not False
    ):
        raise ValueError("graph input commit done semantics differ")
    artifact_hashes = {
        "manifest": _file_digest(paths["manifest"]),
        "archive": _file_digest(paths["archive"]),
        "summary": _file_digest(paths["summary"]),
        "done": _file_digest(paths["done"]),
    }
    return {
        "directory": paths["done"].parent,
        "paths": paths,
        "rows": rows,
        "arrays": arrays,
        "summary": summary,
        "done": done,
        "dino_contract": dino_contract,
        "artifact_hashes": artifact_hashes,
        "artifact_digest": done["artifact_digest"],
    }


def _expected_summary(
    *,
    derived: _Derived,
    paths: Mapping[str, Path],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    summary = dict(derived.summary_base)
    summary.update(
        {
            "array_contract": _array_contract(arrays),
            "archive_sha256": _file_digest(paths["archive"]),
            "manifest_sha256": _file_digest(paths["manifest"]),
        }
    )
    return summary


def _validate_against_derived(
    directory: Path,
    derived: _Derived,
) -> dict[str, Any]:
    committed = validate_graph_input_commit(directory)
    paths = committed["paths"]
    rows = committed["rows"]
    arrays = committed["arrays"]
    summary = committed["summary"]
    done = committed["done"]
    expected_rows = [dict(row) for row in derived.rows]
    if rows != expected_rows:
        raise ValueError("graph input manifest differs from frozen sources")
    _validate_output_arrays(arrays, rows=len(rows))
    for name, expected in derived.arrays.items():
        if not np.array_equal(arrays[name], np.asarray(expected)):
            raise ValueError(f"graph input array {name} differs")
    expected_summary = _expected_summary(
        derived=derived,
        paths=paths,
        arrays=arrays,
    )
    if summary != expected_summary:
        raise ValueError("graph input summary differs")
    expected_done_scalars = {
        "assets": len(rows),
        "iids": int(expected_summary["iids"]),
        "source_artifact_digests": {
            name: record["artifact_digest"]
            for name, record in expected_summary[
                "source_artifacts"
            ].items()
        },
    }
    for key, value in expected_done_scalars.items():
        if done.get(key) != value:
            raise ValueError(f"graph input done {key} differs")
    for index, row in enumerate(rows):
        if (
            set(row) != set(ROW_FIELDS)
            or row.get("schema_version") != ROW_SCHEMA
            or row.get("asset_index") != index
        ):
            raise ValueError(f"graph input row {index} contract differs")
    _assert_snapshot(derived.paths, derived.snapshot)
    return {
        "done": done,
        "summary": summary,
        "rows": rows,
        "arrays": arrays,
        "dino_contract": committed["dino_contract"],
        "artifact_hashes": committed["artifact_hashes"],
        "artifact_digest": committed["artifact_digest"],
    }


def validate_graph_input(
    directory: Path,
    *,
    candidate_features_dir: Path,
    candidate_manifest: Path,
    anchor_features_dir: Path,
    anchor_input_manifest: Path,
) -> dict[str, Any]:
    """Validate the output and freshly rederive every source binding."""

    derived = _derive(
        candidate_features_dir=candidate_features_dir,
        candidate_manifest=candidate_manifest,
        anchor_features_dir=anchor_features_dir,
        anchor_input_manifest=anchor_input_manifest,
    )
    return _validate_against_derived(directory, derived)


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_publish(
    *,
    directory: Path,
    derived: _Derived,
    pre_publish_check: Callable[[], None],
) -> None:
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
        paths = _artifact_paths(staging)
        with paths["archive"].open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    name: np.asarray(value)
                    for name, value in derived.arrays.items()
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        _write_file(paths["manifest"], _jsonl_bytes(derived.rows))
        summary = _expected_summary(
            derived=derived,
            paths=paths,
            arrays=derived.arrays,
        )
        _write_file(paths["summary"], _pretty_json_bytes(summary))
        output_sha = {
            name: _file_digest(paths[name])
            for name in ("manifest", "archive", "summary")
        }
        done = {
            "schema_version": DONE_SCHEMA,
            "status": "complete",
            "assets": len(derived.rows),
            "iids": int(summary["iids"]),
            "source_artifact_digests": {
                name: record["artifact_digest"]
                for name, record in summary["source_artifacts"].items()
            },
            "split_assigned": False,
            "human_labels_asserted": False,
            "training_authorized": False,
            "artifacts": {
                name: {
                    "filename": paths[name].name,
                    "sha256": output_sha[name],
                }
                for name in ("manifest", "archive", "summary")
            },
            "artifact_digest": _object_digest(output_sha),
        }
        _write_file(paths["done"], _pretty_json_bytes(done))
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        pre_publish_check()
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
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_graph_input(
    *,
    candidate_features_dir: Path,
    candidate_manifest: Path,
    anchor_features_dir: Path,
    anchor_input_manifest: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Build one immutable graph input, or strictly verify it on resume."""

    target = output_dir.expanduser()
    if resume:
        if not target.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires an existing "
                f"commit: {target}"
            )
        return validate_graph_input(
            target,
            candidate_features_dir=candidate_features_dir,
            candidate_manifest=candidate_manifest,
            anchor_features_dir=anchor_features_dir,
            anchor_input_manifest=anchor_input_manifest,
        )["done"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    derived = _derive(
        candidate_features_dir=candidate_features_dir,
        candidate_manifest=candidate_manifest,
        anchor_features_dir=anchor_features_dir,
        anchor_input_manifest=anchor_input_manifest,
    )
    _atomic_publish(
        directory=target,
        derived=derived,
        pre_publish_check=lambda: _assert_snapshot(
            derived.paths,
            derived.snapshot,
        ),
    )
    return _validate_against_derived(target, derived)["done"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a strict R7 visual identity-graph input",
    )
    parser.add_argument("--candidate-features-dir", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--anchor-features-dir", required=True, type=Path)
    parser.add_argument("--anchor-input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_graph_input(
        candidate_features_dir=args.candidate_features_dir,
        candidate_manifest=args.candidate_manifest,
        anchor_features_dir=args.anchor_features_dir,
        anchor_input_manifest=args.anchor_input_manifest,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_NAME",
    "DONE_NAME",
    "DONE_SCHEMA",
    "DINO_COMPARISON_FIELDS",
    "MANIFEST_NAME",
    "ROW_FIELDS",
    "ROW_SCHEMA",
    "SCHEMA_VERSION",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA",
    "build_graph_input",
    "main",
    "validate_graph_input",
    "validate_graph_input_commit",
]

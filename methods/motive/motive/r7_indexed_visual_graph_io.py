"""Immutable I/O for the R7 indexed visual-component graph.

This stage is deliberately a *diagnostic split proposal*, not a training
gate.  It consumes the committed visual graph input and the exact-eight-shard
DINO matcher result, converts them to the pure structures in
``r7_indexed_visual_graph``, builds the graph, and asks the pure verifier to
recompute it before publication.

The five output files are deterministic functions of the fully revalidated
inputs.  Publication is an atomic no-overwrite directory rename.  Resume is
verification-only: both inputs and the graph are recomputed, then every
output byte is compared with the existing commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_expansion_dino_edges as dino_edges
from . import r7_dino_quotient_calibration as quotient_calibration
from . import r7_visual_graph_input as graph_input_module
from .r7_indexed_visual_graph import (
    DINO_HARD_THRESHOLD as INDEXED_CORE_DINO_HARD_THRESHOLD,
    MAXIMUM_DHASH_HAMMING,
    R7_INDEXED_VISUAL_SPLIT_VERSION,
    R7IndexedDinoEdge,
    R7IndexedVisualAsset,
    R7IndexedVisualGraphConfig,
    R7IndexedVisualPair,
    build_r7_indexed_visual_graph,
    verify_r7_indexed_visual_graph,
)


SCHEMA_VERSION = "motive-r7-indexed-visual-graph-result-v1"
ASSIGNMENT_ROW_SCHEMA = (
    "motive-r7-indexed-visual-graph-assignment-row-v1"
)
COMPONENT_ROW_SCHEMA = "motive-r7-indexed-visual-graph-component-row-v1"
EDGE_ROW_SCHEMA = "motive-r7-indexed-visual-graph-spanning-edge-row-v1"
SUMMARY_SCHEMA = "motive-r7-indexed-visual-graph-result-summary-v1"
DONE_SCHEMA = "motive-r7-indexed-visual-graph-result-done-v1"

ASSIGNMENTS_NAME = "assignments.jsonl"
COMPONENTS_NAME = "components.jsonl"
SPANNING_EDGES_NAME = "spanning_edges.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = frozenset(
    {
        ASSIGNMENTS_NAME,
        COMPONENTS_NAME,
        SPANNING_EDGES_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    }
)

DATA_SEED = 260108828
DHASH_MAXIMUM_HAMMING = 6
DINO_HARD_THRESHOLD = INDEXED_CORE_DINO_HARD_THRESHOLD
TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.1

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT32_HEX_RE = re.compile(r"^[0-9a-f]{8}$")
_HARD_EDGE_FIELDS = frozenset(
    {
        "schema_version",
        "edge_type",
        "hard_edge",
        "asset_a",
        "asset_b",
        "iid_a",
        "role_a",
        "video_sha256_a",
        "iid_b",
        "role_b",
        "video_sha256_b",
        "cosine",
        "cosine_float32_hex",
        "frame_a",
        "frame_b",
        "owner_rank",
        "world_size",
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


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} is not a lowercase SHA-256 digest")
    return value


def _strict_directory(
    directory: Path,
    *,
    expected_names: set[str] | frozenset[str],
) -> Path:
    unresolved = directory.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    actual = {entry.name for entry in unresolved.iterdir()}
    if actual != set(expected_names):
        raise ValueError(
            f"{unresolved} artifact set differs: "
            f"missing={sorted(set(expected_names) - actual)}, "
            f"extra={sorted(actual - set(expected_names))}"
        )
    for name in expected_names:
        path = unresolved / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
    return unresolved.resolve(strict=True)


def _graph_source_paths(directory: Path) -> dict[str, Path]:
    expected = {
        graph_input_module.MANIFEST_NAME,
        graph_input_module.ARCHIVE_NAME,
        graph_input_module.SUMMARY_NAME,
        graph_input_module.DONE_NAME,
    }
    root = _strict_directory(directory, expected_names=expected)
    return {name: root / name for name in sorted(expected)}


def _strict_dino_commit(
    directory: Path,
    *,
    nested_name: str,
) -> tuple[Path, dict[str, Path]]:
    file_names = {
        dino_edges.HARD_EDGES_NAME,
        dino_edges.AUDIT_EDGES_NAME,
        dino_edges.CALIBRATION_EDGES_NAME,
        dino_edges.SUMMARY_NAME,
        dino_edges.DONE_NAME,
    }
    nested_files = {
        quotient_calibration.ARTIFACT_METADATA_NAME,
        quotient_calibration.ARTIFACT_ARRAYS_NAME,
        quotient_calibration.ARTIFACT_DONE_NAME,
    }
    unresolved = directory.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    if {entry.name for entry in unresolved.iterdir()} != (
        file_names | {nested_name}
    ):
        raise ValueError(f"{unresolved} DINO artifact set differs")
    for name in file_names:
        path = unresolved / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact is not a regular file: {path}")
    nested = unresolved / nested_name
    if nested.is_symlink() or not nested.is_dir():
        raise ValueError(
            f"quotient artifact is not a real directory: {nested}"
        )
    if {entry.name for entry in nested.iterdir()} != nested_files:
        raise ValueError("quotient artifact file set differs")
    for name in nested_files:
        path = nested / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"quotient artifact is not a regular file: {path}"
            )
    root = unresolved.resolve(strict=True)
    paths = {name: root / name for name in sorted(file_names)}
    paths.update(
        {
            f"{nested_name}/{name}": root / nested_name / name
            for name in sorted(nested_files)
        }
    )
    return root, paths


def _dino_source_paths(directory: Path) -> dict[str, Path]:
    final, final_relative_paths = _strict_dino_commit(
        directory,
        nested_name=dino_edges.IID_PAIR_MAXIMA_NAME,
    )
    root = final.parent
    shards_root = root / "shards"
    if shards_root.is_symlink() or not shards_root.is_dir():
        raise FileNotFoundError(shards_root)
    expected_shards = {
        f"rank-{rank:05d}-of-{dino_edges.WORLD_SIZE:05d}"
        for rank in range(dino_edges.WORLD_SIZE)
    }
    actual_shards = {entry.name for entry in shards_root.iterdir()}
    if actual_shards != expected_shards:
        raise ValueError(
            "DINO edge shard directory set is not exact8: "
            f"missing={sorted(expected_shards - actual_shards)}, "
            f"extra={sorted(actual_shards - expected_shards)}"
        )
    paths = {
        f"final/{name}": path
        for name, path in sorted(final_relative_paths.items())
    }
    for rank in range(dino_edges.WORLD_SIZE):
        _shard, shard_paths = _strict_dino_commit(
            shards_root
            / f"rank-{rank:05d}-of-{dino_edges.WORLD_SIZE:05d}",
            nested_name=dino_edges.QUOTIENT_RANK_PARTIAL_NAME,
        )
        for name, path in sorted(shard_paths.items()):
            paths[f"shards/{rank:05d}/{name}"] = path
    return paths


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source artifact changed type: {path}")
        stat = path.stat()
        result[name] = {
            "sha256": _file_digest(path),
            "bytes": int(stat.st_size),
        }
    return result


def _assert_snapshot(
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if _snapshot(paths) != {
        key: dict(value) for key, value in sorted(expected.items())
    }:
        raise RuntimeError("indexed graph inputs changed during processing")


def _snapshot_digest(
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


def _fixed_config() -> R7IndexedVisualGraphConfig:
    config = R7IndexedVisualGraphConfig(
        data_seed=DATA_SEED,
        maximum_dhash_hamming=DHASH_MAXIMUM_HAMMING,
        minimum_dino_cosine=DINO_HARD_THRESHOLD,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        split_version=R7_INDEXED_VISUAL_SPLIT_VERSION,
    )
    config.validate()
    remainder = (
        1.0
        - float(config.train_fraction)
        - float(config.validation_fraction)
    )
    if (
        config.maximum_dhash_hamming != MAXIMUM_DHASH_HAMMING
        or config.maximum_dhash_hamming != 6
        or float(config.minimum_dino_cosine) != 0.96
        or DINO_HARD_THRESHOLD != dino_edges.HARD_THRESHOLD
        or float(config.train_fraction) != 0.8
        or float(config.validation_fraction) != 0.1
        or not np.isclose(
            remainder,
            TEST_FRACTION,
            atol=1e-15,
            rtol=0,
        )
    ):
        raise ValueError("indexed graph fixed config differs")
    return config


def _validate_graph_commit_binding(
    graph: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    required = {
        "directory",
        "paths",
        "rows",
        "arrays",
        "summary",
        "done",
        "dino_contract",
        "artifact_hashes",
        "artifact_digest",
    }
    if not required.issubset(graph):
        raise ValueError("visual graph commit validator return differs")
    if Path(graph["directory"]).resolve(strict=True) != next(
        iter(paths.values())
    ).parent:
        raise ValueError("visual graph commit directory binding differs")
    returned_paths = graph.get("paths")
    expected_paths = {
        "manifest": paths[graph_input_module.MANIFEST_NAME],
        "archive": paths[graph_input_module.ARCHIVE_NAME],
        "summary": paths[graph_input_module.SUMMARY_NAME],
        "done": paths[graph_input_module.DONE_NAME],
    }
    if (
        not isinstance(returned_paths, Mapping)
        or set(returned_paths) != set(expected_paths)
        or any(
            Path(returned_paths[name]).resolve(strict=True)
            != expected_paths[name]
            for name in expected_paths
        )
    ):
        raise ValueError("visual graph commit artifact paths differ")
    artifact_hashes = graph.get("artifact_hashes")
    expected_hashes = {
        "manifest": snapshot[graph_input_module.MANIFEST_NAME]["sha256"],
        "archive": snapshot[graph_input_module.ARCHIVE_NAME]["sha256"],
        "summary": snapshot[graph_input_module.SUMMARY_NAME]["sha256"],
        "done": snapshot[graph_input_module.DONE_NAME]["sha256"],
    }
    if artifact_hashes != expected_hashes:
        raise ValueError("visual graph commit artifact hashes differ")
    _validate_sha256(
        graph.get("artifact_digest"),
        context="visual graph artifact digest",
    )
    rows = graph.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("visual graph commit contains no rows")
    if not isinstance(graph.get("summary"), Mapping):
        raise ValueError("visual graph commit summary differs")


@dataclass(frozen=True)
class _PairProjection:
    pairs: tuple[R7IndexedVisualPair, ...]
    metadata: Mapping[str, Mapping[str, Any]]
    anchor_iids: tuple[str, ...]
    candidate_iids: tuple[str, ...]


def _project_pairs(graph: Mapping[str, Any]) -> _PairProjection:
    rows = graph["rows"]
    if len(rows) % 2:
        raise ValueError("visual graph assets cannot form complete pairs")
    pairs: list[R7IndexedVisualPair] = []
    metadata: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(rows), 2):
        source = rows[offset]
        target = rows[offset + 1]
        iid = source.get("iid")
        if (
            set(source) != set(graph_input_module.ROW_FIELDS)
            or set(target) != set(graph_input_module.ROW_FIELDS)
            or source.get("schema_version")
            != graph_input_module.ROW_SCHEMA
            or target.get("schema_version")
            != graph_input_module.ROW_SCHEMA
            or source.get("asset_index") != offset
            or target.get("asset_index") != offset + 1
            or source.get("role") != "source"
            or target.get("role") != "target"
            or type(iid) is not str
            or not iid
            or target.get("iid") != iid
            or type(source.get("anchor")) is not bool
            or target.get("anchor") is not source["anchor"]
            or type(source.get("cohort")) is not str
            or not source["cohort"]
            or target.get("cohort") != source["cohort"]
            or iid in metadata
        ):
            raise ValueError(
                f"visual graph rows {offset}/{offset + 1} are not one pair"
            )
        pair = R7IndexedVisualPair.create(
            iid=iid,
            source=R7IndexedVisualAsset.create(
                video_sha256=source["video_sha256"],
                dhashes=source["dhashes"],
            ),
            target=R7IndexedVisualAsset.create(
                video_sha256=target["video_sha256"],
                dhashes=target["dhashes"],
            ),
        )
        pairs.append(pair)
        metadata[iid] = {
            "anchor": source["anchor"],
            "cohort": source["cohort"],
        }
    if [pair.iid for pair in pairs] != sorted(
        pair.iid for pair in pairs
    ):
        raise ValueError("visual graph IID pairs are not canonical")
    anchor_iids = tuple(
        iid for iid in sorted(metadata) if metadata[iid]["anchor"]
    )
    candidate_iids = tuple(
        iid for iid in sorted(metadata) if not metadata[iid]["anchor"]
    )
    summary = graph["summary"]
    for name, values in (
        ("anchor_iids", anchor_iids),
        ("candidate_iids", candidate_iids),
    ):
        record = summary.get(name)
        if (
            not isinstance(record, Mapping)
            or record.get("count") != len(values)
            or record.get("sha256") != _object_digest(list(values))
        ):
            raise ValueError(f"visual graph {name} binding differs")
    return _PairProjection(
        pairs=tuple(pairs),
        metadata=metadata,
        anchor_iids=anchor_iids,
        candidate_iids=candidate_iids,
    )


def _validate_dino_binding(
    dino: Mapping[str, Any],
    *,
    graph: Mapping[str, Any],
    graph_directory: Path,
    final_paths: Mapping[str, Path],
) -> Mapping[str, Any]:
    required = {
        "paths",
        "done",
        "summary",
        "contract",
        "hard_edges",
        "audit_edges",
        "calibration_edges",
        "iid_pair_maxima",
    }
    if not required.issubset(dino):
        raise ValueError("DINO final validator return differs")
    returned_paths = dino.get("paths")
    expected_paths = {
        name: final_paths[f"final/{name}"]
        for name in (
            dino_edges.HARD_EDGES_NAME,
            dino_edges.AUDIT_EDGES_NAME,
            dino_edges.CALIBRATION_EDGES_NAME,
            dino_edges.SUMMARY_NAME,
            dino_edges.DONE_NAME,
        )
    }
    if (
        not isinstance(returned_paths, Mapping)
        or set(returned_paths) != set(expected_paths)
        or any(
            Path(returned_paths[name]).resolve(strict=True)
            != expected_paths[name]
            for name in expected_paths
        )
    ):
        raise ValueError("DINO final artifact paths differ")
    contract = dino.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("DINO final contract is missing")
    expected_fields = set(dino_edges._CONTRACT_FIELDS) - {
        "rank",
        "device",
    }
    if set(contract) != expected_fields:
        raise ValueError("DINO final common contract field set differs")
    algorithm = contract.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError("DINO final algorithm contract is missing")
    if (
        contract.get("schema_version") != dino_edges.MATCHER_SCHEMA
        or contract.get("world_size") != dino_edges.WORLD_SIZE
        or contract.get("input_directory")
        != str(graph_directory.resolve(strict=True))
        or contract.get("input_artifact_digest")
        != graph["artifact_digest"]
        or contract.get("input_artifacts")
        != graph["artifact_hashes"]
        or contract.get("input_rows") != len(graph["rows"])
        or contract.get("dino_contract") != graph["dino_contract"]
        or contract.get("algorithm_sha256")
        != _object_digest(dict(algorithm))
        or algorithm.get("hard_threshold") != DINO_HARD_THRESHOLD
        or algorithm.get("hard_operator") != ">="
        or algorithm.get("audit_lower_threshold")
        != dino_edges.AUDIT_THRESHOLD
        or algorithm.get("audit_upper_threshold")
        != DINO_HARD_THRESHOLD
        or algorithm.get("audit_is_hard") is not False
        or algorithm.get("frames_per_asset") != dino_edges.DINO_FRAMES
        or algorithm.get("embedding_dim") != dino_edges.DINO_DIM
        or algorithm.get("compute_dtype") != "float32"
    ):
        raise ValueError("DINO final/input/threshold binding differs")
    dino_contract_sha = _object_digest(graph["dino_contract"])
    if contract.get("dino_contract_sha256") != dino_contract_sha:
        raise ValueError("DINO final DINO-contract digest differs")
    contract_sha = _object_digest(dict(contract))
    summary = dino.get("summary")
    done = dino.get("done")
    if (
        not isinstance(summary, Mapping)
        or not isinstance(done, Mapping)
        or summary.get("contract_sha256") != contract_sha
        or done.get("contract_sha256") != contract_sha
    ):
        raise ValueError("DINO final contract hash chain differs")
    if not isinstance(dino.get("hard_edges"), list):
        raise ValueError("DINO final hard edge list differs")
    if not isinstance(dino.get("calibration_edges"), list):
        raise ValueError("DINO final calibration edge list differs")
    iid_pair_maxima = dino.get("iid_pair_maxima")
    iid_contract = (
        iid_pair_maxima.get("contract")
        if isinstance(iid_pair_maxima, Mapping)
        else None
    )
    expected_iid_pairs = (
        len(graph["rows"]) // 2
        * (len(graph["rows"]) // 2 - 1)
        // 2
    )
    if (
        not isinstance(iid_contract, Mapping)
        or iid_pair_maxima.get("schema_version")
        != quotient_calibration.IID_PAIR_MAXIMA_SCHEMA
        or iid_contract.get("expected_iid_pairs")
        != expected_iid_pairs
        or iid_contract.get("observed_iid_pairs")
        != expected_iid_pairs
        or iid_contract.get("partials_per_iid_pair") != 2
        or iid_contract.get("observed_partials")
        != 2 * expected_iid_pairs
        or iid_contract.get("coverage_complete") is not True
        or iid_contract.get("training_authorized") is not False
    ):
        raise ValueError("DINO IID-pair maxima coverage differs")
    if (
        done.get("calibration_intended_use")
        != "diagnostic_not_threshold_calibrating"
        or done.get("thresholds_human_calibrated") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_authorized") is not False
    ):
        raise ValueError("DINO calibration authorization state differs")
    return contract


def _float32_from_hex(value: Any) -> np.float32:
    if type(value) is not str or _FLOAT32_HEX_RE.fullmatch(value) is None:
        raise ValueError("DINO cosine_float32_hex differs")
    return np.float32(struct.unpack(">f", bytes.fromhex(value))[0])


def _project_dino_edges(
    raw_edges: Sequence[Mapping[str, Any]],
    *,
    graph_rows: Sequence[Mapping[str, Any]],
) -> tuple[R7IndexedDinoEdge, ...]:
    projected: list[R7IndexedDinoEdge] = []
    previous: tuple[int, int] | None = None
    threshold = np.float32(DINO_HARD_THRESHOLD)
    for position, row in enumerate(raw_edges):
        if (
            set(row) != set(_HARD_EDGE_FIELDS)
            or row.get("schema_version") != dino_edges.HARD_EDGE_SCHEMA
            or row.get("edge_type") != "hard_dino"
            or row.get("hard_edge") is not True
            or row.get("world_size") != dino_edges.WORLD_SIZE
        ):
            raise ValueError(f"DINO hard edge {position} schema differs")
        asset_a = row.get("asset_a")
        asset_b = row.get("asset_b")
        if (
            type(asset_a) is not int
            or type(asset_b) is not int
            or not 0 <= asset_a < asset_b < len(graph_rows)
            or previous is not None
            and (asset_a, asset_b) <= previous
            or row.get("owner_rank")
            != asset_a % dino_edges.WORLD_SIZE
        ):
            raise ValueError(
                f"DINO hard edge {position} endpoints/order differ"
            )
        previous = (asset_a, asset_b)
        left = graph_rows[asset_a]
        right = graph_rows[asset_b]
        expected_identity = {
            "iid_a": left["iid"],
            "role_a": left["role"],
            "video_sha256_a": left["video_sha256"],
            "iid_b": right["iid"],
            "role_b": right["role"],
            "video_sha256_b": right["video_sha256"],
        }
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(
                f"DINO hard edge {position} identity binding differs"
            )
        score = _float32_from_hex(row.get("cosine_float32_hex"))
        rounded = round(float(score), dino_edges.COSINE_ROUND_DECIMALS)
        if (
            not np.isfinite(score)
            or score < threshold
            or type(row.get("cosine")) not in {int, float}
            or float(row["cosine"]) != rounded
            or any(
                type(row.get(field)) is not int
                or not 0 <= row[field] < dino_edges.DINO_FRAMES
                for field in ("frame_a", "frame_b")
            )
        ):
            raise ValueError(
                f"DINO hard edge {position} score/frame differs"
            )
        # The matcher threshold is evaluated in float32.  np.float32(0.96)
        # is microscopically below Python's exact 0.96, while the pure graph
        # intentionally uses the policy decimal.  Flooring an already
        # validated hard edge to that decimal preserves the inclusive
        # boundary in both stages.
        graph_score = max(DINO_HARD_THRESHOLD, float(score))
        projected.append(
            R7IndexedDinoEdge.create(
                left_iid=left["iid"],
                left_role=left["role"],
                right_iid=right["iid"],
                right_role=right["role"],
                cosine=graph_score,
            )
        )
    return tuple(projected)


@dataclass(frozen=True)
class _Derived:
    assignment_rows: tuple[Mapping[str, Any], ...]
    component_rows: tuple[Mapping[str, Any], ...]
    edge_rows: tuple[Mapping[str, Any], ...]
    summary_base: Mapping[str, Any]
    input_paths: Mapping[str, Path]
    input_snapshot: Mapping[str, Mapping[str, Any]]


def _derive(
    *,
    graph_input_dir: Path,
    dino_edges_dir: Path,
) -> _Derived:
    graph_directory = _strict_directory(
        graph_input_dir,
        expected_names={
            graph_input_module.MANIFEST_NAME,
            graph_input_module.ARCHIVE_NAME,
            graph_input_module.SUMMARY_NAME,
            graph_input_module.DONE_NAME,
        },
    )
    dino_directory, _unused_final_paths = _strict_dino_commit(
        dino_edges_dir,
        nested_name=dino_edges.IID_PAIR_MAXIMA_NAME,
    )
    del _unused_final_paths
    graph_paths = _graph_source_paths(graph_directory)
    dino_paths = _dino_source_paths(dino_directory)
    input_paths = {
        **{f"graph/{key}": value for key, value in graph_paths.items()},
        **{f"dino/{key}": value for key, value in dino_paths.items()},
    }
    before = _snapshot(input_paths)
    graph = graph_input_module.validate_graph_input_commit(
        graph_directory
    )
    _validate_graph_commit_binding(
        graph,
        paths=graph_paths,
        snapshot={
            key.removeprefix("graph/"): value
            for key, value in before.items()
            if key.startswith("graph/")
        },
    )
    dino = dino_edges.validate_final(
        dino_directory,
        input_directory=graph_directory,
        output_root=dino_directory.parent,
    )
    contract = _validate_dino_binding(
        dino,
        graph=graph,
        graph_directory=graph_directory,
        final_paths=dino_paths,
    )
    after = _snapshot(input_paths)
    if after != before:
        raise RuntimeError(
            "visual graph or DINO artifacts changed during validation"
        )

    projection = _project_pairs(graph)
    projected_edges = _project_dino_edges(
        dino["hard_edges"],
        graph_rows=graph["rows"],
    )
    config = _fixed_config()
    result = build_r7_indexed_visual_graph(
        projection.pairs,
        dino_edges=projected_edges,
        anchor_iids=projection.anchor_iids,
        previously_seen_iids=(),
        config=config,
    )
    verify_r7_indexed_visual_graph(
        result,
        projection.pairs,
        dino_edges=projected_edges,
        anchor_iids=projection.anchor_iids,
        previously_seen_iids=(),
        config=config,
    )
    assignment_rows = tuple(
        {
            "schema_version": ASSIGNMENT_ROW_SCHEMA,
            **assignment.to_dict(),
            "anchor": bool(
                projection.metadata[assignment.iid]["anchor"]
            ),
            "cohort": projection.metadata[assignment.iid]["cohort"],
        }
        for assignment in result.assignments
    )
    expected_iids = {pair.iid for pair in projection.pairs}
    assigned_iids = {row["iid"] for row in assignment_rows}
    candidate_assigned = {
        row["iid"] for row in assignment_rows if not row["anchor"]
    }
    if (
        assigned_iids != expected_iids
        or candidate_assigned != set(projection.candidate_iids)
        or len(assignment_rows) != len(expected_iids)
    ):
        raise AssertionError("indexed graph assignment conservation failed")
    component_rows = tuple(
        {
            "schema_version": COMPONENT_ROW_SCHEMA,
            **component.to_dict(),
        }
        for component in result.components
    )
    edge_rows = tuple(
        {
            "schema_version": EDGE_ROW_SCHEMA,
            **edge.to_dict(),
        }
        for edge in result.spanning_edges
    )
    statistics = result.statistics.to_dict()
    split_counts = {
        split: sum(row["split"] == split for row in assignment_rows)
        for split in ("train", "validation", "test")
    }
    fresh_counts = {
        "fresh_iids": sum(bool(row["fresh"]) for row in assignment_rows),
        "nonfresh_iids": sum(
            not bool(row["fresh"]) for row in assignment_rows
        ),
        "forced_train_iids": sum(
            bool(row["forced_train"]) for row in assignment_rows
        ),
        "forced_by_anchor_iids": sum(
            bool(row["forced_by_anchor"]) for row in assignment_rows
        ),
        "forced_by_previously_seen_iids": sum(
            bool(row["forced_by_previously_seen"])
            for row in assignment_rows
        ),
    }
    dino_snapshot = {
        key.removeprefix("dino/"): value
        for key, value in after.items()
        if key.startswith("dino/")
    }
    expected_iid_pairs = (
        len(projection.pairs) * (len(projection.pairs) - 1) // 2
    )
    summary_base = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "assignment_semantics":
            "diagnostic-provisional-component-split-v1",
        "input_bindings": {
            "visual_graph": {
                "artifact_digest": graph["artifact_digest"],
                "artifact_hashes": dict(graph["artifact_hashes"]),
                "candidate_iids": len(projection.candidate_iids),
                "anchor_iids": len(projection.anchor_iids),
            },
            "dino_edges": {
                "artifact_digest": _snapshot_digest(dino_snapshot),
                "contract_sha256": _object_digest(dict(contract)),
                "hard_edges": len(dino["hard_edges"]),
                "audit_edges": len(dino["audit_edges"]),
                "calibration_edges": len(dino["calibration_edges"]),
                "calibration_intended_use":
                    "diagnostic_not_threshold_calibrating",
                "iid_pair_maxima": expected_iid_pairs,
                "iid_pair_maxima_artifact_digest":
                    dino["iid_pair_maxima"]["artifact_digest"],
                "exact_shards": dino_edges.WORLD_SIZE,
            },
        },
        "counts": {
            "candidate_iids": len(projection.candidate_iids),
            "anchor_iids": len(projection.anchor_iids),
            "total_iids": len(projection.pairs),
            "assets": 2 * len(projection.pairs),
            "components": len(result.components),
            "spanning_edges": len(result.spanning_edges),
            "hard_dino_input_edges": len(projected_edges),
        },
        "split_iid_counts": split_counts,
        "freshness_counts": fresh_counts,
        "config": {
            "data_seed": DATA_SEED,
            "maximum_dhash_hamming": DHASH_MAXIMUM_HAMMING,
            "minimum_dino_cosine": DINO_HARD_THRESHOLD,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "split_version": R7_INDEXED_VISUAL_SPLIT_VERSION,
            "dino_float32_boundary_policy":
                "validated-hard-edge-floor-to-decimal-threshold-v1",
        },
        "statistics": statistics,
        "provenance": result.provenance.to_dict(),
        "giant_component_warning": bool(
            statistics["giant_component_warning"]
        ),
        "giant_component_warning_reason": (
            "largest_component_exceeds_5_percent_of_iids"
            if statistics["giant_component_warning"]
            else None
        ),
        "thresholds_human_calibrated": False,
        "formal_split": False,
        "training_authorized": False,
    }
    return _Derived(
        assignment_rows=assignment_rows,
        component_rows=component_rows,
        edge_rows=edge_rows,
        summary_base=summary_base,
        input_paths=input_paths,
        input_snapshot=after,
    )


def _payloads(
    derived: _Derived,
    *,
    include_permission_contract: bool = True,
) -> dict[str, bytes]:
    assignments = _jsonl_bytes(derived.assignment_rows)
    components = _jsonl_bytes(derived.component_rows)
    edges = _jsonl_bytes(derived.edge_rows)
    output_records = {
        ASSIGNMENTS_NAME: {
            "rows": len(derived.assignment_rows),
            "sha256": _bytes_digest(assignments),
            "order": "iid",
        },
        COMPONENTS_NAME: {
            "rows": len(derived.component_rows),
            "sha256": _bytes_digest(components),
            "order": "component_id",
        },
        SPANNING_EDGES_NAME: {
            "rows": len(derived.edge_rows),
            "sha256": _bytes_digest(edges),
            "order": "canonical-endpoints-relation-value",
        },
    }
    summary = dict(derived.summary_base)
    summary["outputs"] = {
        name: output_records[name] for name in sorted(output_records)
    }
    summary_bytes = _pretty_json_bytes(summary)
    artifact_hashes = {
        ASSIGNMENTS_NAME: output_records[ASSIGNMENTS_NAME]["sha256"],
        COMPONENTS_NAME: output_records[COMPONENTS_NAME]["sha256"],
        SPANNING_EDGES_NAME: output_records[SPANNING_EDGES_NAME]["sha256"],
        SUMMARY_NAME: _bytes_digest(summary_bytes),
    }
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "iids": int(summary["counts"]["total_iids"]),
        "components": int(summary["counts"]["components"]),
        "input_artifact_digests": {
            name: record["artifact_digest"]
            for name, record in summary["input_bindings"].items()
        },
        "artifacts": {
            name: {
                "filename": name,
                "sha256": digest,
            }
            for name, digest in sorted(artifact_hashes.items())
        },
        "artifact_digest": _object_digest(artifact_hashes),
        "giant_component_warning": summary["giant_component_warning"],
        "thresholds_human_calibrated": False,
        "formal_split": False,
        "training_authorized": False,
    }
    if include_permission_contract:
        done["permission_contract"] = (
            artifact_permissions.permission_contract()
        )
    return {
        ASSIGNMENTS_NAME: assignments,
        COMPONENTS_NAME: components,
        SPANNING_EDGES_NAME: edges,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: _pretty_json_bytes(done),
    }


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(
    directory: Path,
    *,
    payloads: Mapping[str, bytes],
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
        for name in sorted(OUTPUT_NAMES):
            _write_file(staging / name, payloads[name])
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        pre_publish_check()
        artifact_permissions.seal_staging_tree(
            staging,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            staging,
            allow_writable_root=True,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"commit target appeared during publication: {target}"
            )
        os.rename(staging, target)
        artifact_permissions.seal_published_root(target)
        parent_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def _validate_payloads(
    directory: Path,
    *,
    expected: Mapping[str, bytes],
    verify_permissions: bool = False,
) -> dict[str, Any]:
    root = _strict_directory(directory, expected_names=OUTPUT_NAMES)
    for name in sorted(OUTPUT_NAMES):
        actual = (root / name).read_bytes()
        if actual != expected[name]:
            raise ValueError(
                f"indexed graph output differs byte-for-byte: {name}"
            )
    done = json.loads(expected[DONE_NAME])
    if "permission_contract" in done:
        artifact_permissions.validate_permission_contract(
            done["permission_contract"]
        )
        if verify_permissions:
            artifact_permissions.assert_sealed_tree(root)
    summary = json.loads(expected[SUMMARY_NAME])
    return {
        "done": done,
        "summary": summary,
        "directory": root,
    }


def validate_indexed_visual_graph_result(
    directory: Path,
    *,
    graph_input_dir: Path,
    dino_edges_dir: Path,
) -> dict[str, Any]:
    """Fully revalidate both inputs and byte-recompute the result commit."""

    derived = _derive(
        graph_input_dir=graph_input_dir,
        dino_edges_dir=dino_edges_dir,
    )
    root = _strict_directory(directory, expected_names=OUTPUT_NAMES)
    try:
        actual_done = json.loads((root / DONE_NAME).read_text("utf-8"))
    except Exception as error:
        raise ValueError("indexed graph done JSON is invalid") from error
    sealed = (
        isinstance(actual_done, Mapping)
        and "permission_contract" in actual_done
    )
    result = _validate_payloads(
        root,
        expected=_payloads(
            derived,
            include_permission_contract=sealed,
        ),
        verify_permissions=sealed,
    )
    _assert_snapshot(derived.input_paths, derived.input_snapshot)
    return result


def build_indexed_visual_graph_result(
    *,
    graph_input_dir: Path,
    dino_edges_dir: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Build the immutable diagnostic graph result or verify on resume."""

    target = output_dir.expanduser()
    if resume:
        if not target.exists():
            raise FileNotFoundError(
                "--resume is verification-only and requires an existing "
                f"commit: {target}"
            )
        return validate_indexed_visual_graph_result(
            target,
            graph_input_dir=graph_input_dir,
            dino_edges_dir=dino_edges_dir,
        )["done"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    derived = _derive(
        graph_input_dir=graph_input_dir,
        dino_edges_dir=dino_edges_dir,
    )
    payloads = _payloads(derived)
    _atomic_publish(
        target,
        payloads=payloads,
        pre_publish_check=lambda: _assert_snapshot(
            derived.input_paths,
            derived.input_snapshot,
        ),
    )
    result = _validate_payloads(
        target,
        expected=payloads,
        verify_permissions=True,
    )
    _assert_snapshot(derived.input_paths, derived.input_snapshot)
    return result["done"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build immutable R7 indexed visual graph results",
    )
    parser.add_argument("--graph-input-dir", required=True, type=Path)
    parser.add_argument("--dino-edges-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_indexed_visual_graph_result(
        graph_input_dir=args.graph_input_dir,
        dino_edges_dir=args.dino_edges_dir,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENTS_NAME",
    "ASSIGNMENT_ROW_SCHEMA",
    "COMPONENTS_NAME",
    "COMPONENT_ROW_SCHEMA",
    "DATA_SEED",
    "DINO_HARD_THRESHOLD",
    "DONE_NAME",
    "DONE_SCHEMA",
    "EDGE_ROW_SCHEMA",
    "OUTPUT_NAMES",
    "SCHEMA_VERSION",
    "SPANNING_EDGES_NAME",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA",
    "build_indexed_visual_graph_result",
    "main",
    "validate_indexed_visual_graph_result",
]

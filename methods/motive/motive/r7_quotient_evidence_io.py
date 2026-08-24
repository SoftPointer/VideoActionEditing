"""Immutable CPU postprocess for pre-DINO R7 quotient evidence.

The postprocess consumes exactly two committed inputs:

* a validated visual graph-input commit; and
* a validated DINO final/root exact-eight commit.

The fixed ``final/iid_pair_maxima`` child is accepted only after the DINO
validator has revalidated all eight rank commits and remerged their quotient
partials.  A caller cannot inject a standalone IID-maxima artifact.  DINO
scores remain evidence only: base components are recomputed with the public
:mod:`r7_indexed_visual_graph` core using ``dino_edges=()``.  Consequently the
only component-forming relations are paired-sample atomicity, exact video
SHA-256 equality, and any-frame dHash Hamming distance at most six.  The
complete IID-pair maxima are then reduced to exact base-component-pair maxima
and a deterministic quotient-pair calibration sample.

Publication is create-only and atomic.  Resume is verification-only: both
inputs, the base graph, both ndarray artifacts, and every output byte are
recomputed before the existing directory is accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from . import r7_artifact_permissions as artifact_permissions
from . import r7_dino_quotient_calibration as quotient
from . import r7_expansion_dino_edges as dino_edges
from . import r7_indexed_visual_graph as indexed_core
from . import r7_visual_graph_input as graph_input


SCHEMA_VERSION = "motive-r7-quotient-evidence-postprocess-v3"
BASE_COMPONENTS_SCHEMA = "motive-r7-pre-dino-base-components-v3"
SUMMARY_SCHEMA = "motive-r7-quotient-evidence-summary-v3"
DONE_SCHEMA = "motive-r7-quotient-evidence-done-v3"

BASE_COMPONENTS_NAME = "base_components.json"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
COMPONENT_MAXIMA_DIR = "component_pair_maxima"
CALIBRATION_SAMPLE_DIR = "calibration_sample"

OUTPUT_ENTRIES = frozenset(
    {
        BASE_COMPONENTS_NAME,
        SUMMARY_NAME,
        DONE_NAME,
        COMPONENT_MAXIMA_DIR,
        CALIBRATION_SAMPLE_DIR,
    }
)
_ARRAY_ARTIFACT_FILES = frozenset(
    {
        quotient.ARTIFACT_METADATA_NAME,
        quotient.ARTIFACT_ARRAYS_NAME,
        quotient.ARTIFACT_DONE_NAME,
    }
)

BASE_COMPONENT_ALGORITHM = (
    "public-indexed-core-pair-sha-dhash6-with-empty-dino-v3"
)
MAXIMUM_DHASH_HAMMING = 6


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


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


def _strict_directory(
    directory: Path,
    *,
    files: set[str] | frozenset[str],
    directories: set[str] | frozenset[str] = frozenset(),
) -> Path:
    path = directory.expanduser()
    try:
        root_status = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(path) from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(
        root_status.st_mode
    ):
        raise ValueError(f"not a strict directory: {path}")
    expected = set(files) | set(directories)
    observed = {entry.name for entry in os.scandir(path)}
    if observed != expected:
        raise ValueError(
            f"{path} artifact closure differs: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for name in files:
        status = (path / name).lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(
            status.st_mode
        ):
            raise ValueError(f"artifact is not a regular file: {name}")
    for name in directories:
        status = (path / name).lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(
            status.st_mode
        ):
            raise ValueError(f"artifact is not a real directory: {name}")
    return path.resolve(strict=True)


def _snapshot(
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(
            status.st_mode
        ):
            raise ValueError(f"input changed type: {path}")
        result[name] = {
            "sha256": _file_digest(path),
            "bytes": int(status.st_size),
        }
    return result


def _snapshot_digest(
    value: Mapping[str, Mapping[str, Any]],
) -> str:
    normalized = {
        name: {
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for name, record in sorted(value.items())
    }
    return _object_digest(normalized)


def _assert_snapshot(
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if _snapshot(paths) != {
        name: dict(record)
        for name, record in sorted(expected.items())
    }:
        raise RuntimeError("quotient evidence inputs changed during work")


def _graph_paths(directory: Path) -> dict[str, Path]:
    names = {
        graph_input.MANIFEST_NAME,
        graph_input.ARCHIVE_NAME,
        graph_input.SUMMARY_NAME,
        graph_input.DONE_NAME,
    }
    root = _strict_directory(directory, files=names)
    return {name: root / name for name in sorted(names)}


def _dino_artifact_paths(
    final_directory: Path,
) -> tuple[Path, dict[str, Path]]:
    dino_files = {
        dino_edges.HARD_EDGES_NAME,
        dino_edges.AUDIT_EDGES_NAME,
        dino_edges.CALIBRATION_EDGES_NAME,
        dino_edges.SUMMARY_NAME,
        dino_edges.DONE_NAME,
    }
    final = _strict_directory(
        final_directory,
        files=dino_files,
        directories={dino_edges.IID_PAIR_MAXIMA_NAME},
    )
    if final.name != "final":
        raise ValueError("DINO final directory must have the fixed name")
    root = final.parent
    quotient_root = _strict_directory(
        final / dino_edges.IID_PAIR_MAXIMA_NAME,
        files=_ARRAY_ARTIFACT_FILES,
    )
    paths: dict[str, Path] = {
        **{
            f"final/{name}": final / name
            for name in sorted(dino_files)
        },
        **{
            f"final/{dino_edges.IID_PAIR_MAXIMA_NAME}/{name}":
                quotient_root / name
            for name in sorted(_ARRAY_ARTIFACT_FILES)
        },
    }
    shards_root = root / "shards"
    status = shards_root.lstat()
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise ValueError("DINO shards root is not a real directory")
    expected_shards = {
        f"rank-{rank:05d}-of-{dino_edges.WORLD_SIZE:05d}"
        for rank in range(dino_edges.WORLD_SIZE)
    }
    observed_shards = {
        entry.name for entry in os.scandir(shards_root)
    }
    if observed_shards != expected_shards:
        raise ValueError("DINO shard directory set is not exact8")
    for shard_name in sorted(expected_shards):
        shard = _strict_directory(
            shards_root / shard_name,
            files=dino_files,
            directories={dino_edges.QUOTIENT_RANK_PARTIAL_NAME},
        )
        partial = _strict_directory(
            shard / dino_edges.QUOTIENT_RANK_PARTIAL_NAME,
            files=_ARRAY_ARTIFACT_FILES,
        )
        paths.update(
            {
                f"shards/{shard_name}/{name}": shard / name
                for name in sorted(dino_files)
            }
        )
        paths.update(
            {
                (
                    f"shards/{shard_name}/"
                    f"{dino_edges.QUOTIENT_RANK_PARTIAL_NAME}/{name}"
                ): partial / name
                for name in sorted(_ARRAY_ARTIFACT_FILES)
            }
        )
    return root, paths


@dataclass(frozen=True)
class _Projection:
    pairs: tuple[indexed_core.R7IndexedVisualPair, ...]
    iids: tuple[str, ...]
    anchor_iids: tuple[str, ...]
    anchor_flags: Mapping[str, bool]


def _project_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> _Projection:
    if len(rows) < 4 or len(rows) % 2:
        raise ValueError("graph input does not contain paired IIDs")
    pairs: list[indexed_core.R7IndexedVisualPair] = []
    iids: list[str] = []
    anchor_flags: dict[str, bool] = {}
    for offset in range(0, len(rows), 2):
        source = rows[offset]
        target = rows[offset + 1]
        iid = source.get("iid")
        anchor = source.get("anchor")
        if (
            set(source) != set(graph_input.ROW_FIELDS)
            or set(target) != set(graph_input.ROW_FIELDS)
            or source.get("schema_version") != graph_input.ROW_SCHEMA
            or target.get("schema_version") != graph_input.ROW_SCHEMA
            or source.get("asset_index") != offset
            or target.get("asset_index") != offset + 1
            or source.get("role") != "source"
            or target.get("role") != "target"
            or type(iid) is not str
            or not iid
            or target.get("iid") != iid
            or type(anchor) is not bool
            or target.get("anchor") is not anchor
            or iid in anchor_flags
        ):
            raise ValueError(
                f"graph rows {offset}/{offset + 1} are not atomic"
            )
        pair = indexed_core.R7IndexedVisualPair.create(
            iid=iid,
            source=indexed_core.R7IndexedVisualAsset.create(
                video_sha256=source["video_sha256"],
                dhashes=source["dhashes"],
            ),
            target=indexed_core.R7IndexedVisualAsset.create(
                video_sha256=target["video_sha256"],
                dhashes=target["dhashes"],
            ),
        )
        pairs.append(pair)
        iids.append(iid)
        anchor_flags[iid] = anchor
    if iids != sorted(iids) or len(set(iids)) != len(iids):
        raise ValueError("graph IID pair order differs")
    anchors = tuple(
        iid for iid in iids if anchor_flags[iid]
    )
    return _Projection(
        pairs=tuple(pairs),
        iids=tuple(iids),
        anchor_iids=anchors,
        anchor_flags=anchor_flags,
    )


def _fixed_config() -> indexed_core.R7IndexedVisualGraphConfig:
    config = indexed_core.R7IndexedVisualGraphConfig(
        maximum_dhash_hamming=MAXIMUM_DHASH_HAMMING,
    )
    config.validate()
    if (
        config.maximum_dhash_hamming
        != indexed_core.MAXIMUM_DHASH_HAMMING
        or config.maximum_dhash_hamming != 6
    ):
        raise RuntimeError("pre-DINO dHash threshold differs")
    return config


def _base_component_payload(
    *,
    projection: _Projection,
    graph: Mapping[str, Any],
    graph_snapshot: Mapping[str, Mapping[str, Any]],
    iid_artifact: Mapping[str, Any],
    dino_validation: Mapping[str, Any],
    dino_snapshot: Mapping[str, Mapping[str, Any]],
    iid_snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, str],
    indexed_core.R7IndexedVisualGraphResult,
]:
    graph_commit_binding = quotient.make_graph_commit_binding(
        artifact_digest=graph["artifact_digest"],
        artifact_hashes=graph["artifact_hashes"],
    )
    inherited_binding = iid_artifact["contract"].get(
        "graph_commit_binding"
    )
    if (
        inherited_binding != graph_commit_binding
        or iid_artifact["contract"].get(
            "graph_commit_binding_sha256"
        )
        != _object_digest(graph_commit_binding)
    ):
        raise ValueError(
            "IID maxima does not inherit the exact validated graph commit"
        )
    config = _fixed_config()
    # This literal empty tuple is a security boundary: no indexed-result and
    # no caller-supplied DINO relation can influence pre-DINO components.
    result = indexed_core.build_r7_indexed_visual_graph(
        projection.pairs,
        dino_edges=(),
        anchor_iids=projection.anchor_iids,
        previously_seen_iids=(),
        config=config,
    )
    indexed_core.verify_r7_indexed_visual_graph(
        result,
        projection.pairs,
        dino_edges=(),
        anchor_iids=projection.anchor_iids,
        previously_seen_iids=(),
        config=config,
    )
    statistics = result.statistics.to_dict()
    relation_counts = dict(statistics["relation_counts"])
    candidate_counts = dict(
        statistics["indexed_candidate_relation_counts"]
    )
    allowed_relations = {
        "paired_sample",
        "exact_sha256",
        "dhash_hamming",
        "dino_cosine",
    }
    if (
        set(relation_counts) != allowed_relations
        or set(candidate_counts) != allowed_relations
        or relation_counts["paired_sample"] != len(projection.iids)
        or relation_counts["dino_cosine"] != 0
        or candidate_counts["dino_cosine"] != 0
        or statistics["dino_input_edge_count"] != 0
        or statistics["dino_above_threshold_count"] != 0
        or statistics["dino_below_threshold_count"] != 0
        or any(
            edge.relation == "dino_cosine"
            for edge in result.spanning_edges
        )
    ):
        raise RuntimeError("DINO relation entered pre-DINO components")
    mapping = {
        assignment.iid: assignment.component_id
        for assignment in result.assignments
    }
    if (
        set(mapping) != set(projection.iids)
        or len(mapping) != len(projection.iids)
    ):
        raise RuntimeError("base component IID mapping is not exhaustive")
    component_rows: list[dict[str, Any]] = []
    seen_iids: set[str] = set()
    seen_assets: set[tuple[str, str]] = set()
    for component in result.components:
        member_iids = list(component.member_iids)
        member_assets = [
            [iid, role] for iid, role in component.member_assets
        ]
        expected_assets = {
            (iid, role)
            for iid in member_iids
            for role in indexed_core.ASSET_ROLES
        }
        if (
            set(component.member_assets) != expected_assets
            or seen_iids & set(member_iids)
            or seen_assets & expected_assets
        ):
            raise RuntimeError("base component pair atomicity differs")
        seen_iids.update(member_iids)
        seen_assets.update(expected_assets)
        anchors = [
            iid for iid in member_iids
            if projection.anchor_flags[iid]
        ]
        component_rows.append(
            {
                "component_id": component.component_id,
                "member_iids": member_iids,
                "member_assets": member_assets,
                "anchor_iids": anchors,
                "contains_anchor": bool(anchors),
            }
        )
    component_rows.sort(key=lambda row: row["component_id"])
    components = len(component_rows)
    if (
        seen_iids != set(projection.iids)
        or len(seen_assets) != 2 * len(projection.iids)
        or statistics["pair_count"] != len(projection.iids)
        or statistics["asset_count"] != 2 * len(projection.iids)
        or statistics["component_count"] != components
        or len(result.spanning_edges)
        != 2 * len(projection.iids) - components
    ):
        raise RuntimeError("base component conservation differs")
    mapping_rows = [
        {
            "iid": iid,
            "base_component": mapping[iid],
            "anchor": projection.anchor_flags[iid],
        }
        for iid in projection.iids
    ]
    graph_binding = {
        "artifact_digest": graph["artifact_digest"],
        "artifact_hashes": dict(graph["artifact_hashes"]),
        "validated_graph_commit_binding": graph_commit_binding,
        "file_snapshot": {
            name: dict(record)
            for name, record in sorted(graph_snapshot.items())
        },
        "file_snapshot_digest": _snapshot_digest(graph_snapshot),
    }
    iid_binding = {
        "schema_version": iid_artifact["schema_version"],
        "logical_artifact_digest": iid_artifact["artifact_digest"],
        "validated_graph_commit_binding": inherited_binding,
        "file_snapshot": {
            name: dict(record)
            for name, record in sorted(iid_snapshot.items())
        },
        "file_snapshot_digest": _snapshot_digest(iid_snapshot),
        "parent_dino_quotient_binding": dict(
            dino_validation["quotient_binding"]
        ),
    }
    dino_binding = {
        "final_done_sha256": dino_snapshot[
            f"final/{dino_edges.DONE_NAME}"
        ]["sha256"],
        "final_summary_sha256": dino_snapshot[
            f"final/{dino_edges.SUMMARY_NAME}"
        ]["sha256"],
        "contract_sha256":
            dino_validation["summary"]["contract_sha256"],
        "coverage_proof": dict(
            dino_validation["summary"]["coverage_proof"]
        ),
        "coverage_proof_sha256": _object_digest(
            dino_validation["summary"]["coverage_proof"]
        ),
        "quotient_artifact": dict(
            dino_validation["quotient_binding"]
        ),
        "recursive_file_snapshot": {
            name: dict(record)
            for name, record in sorted(dino_snapshot.items())
        },
        "recursive_file_snapshot_digest":
            _snapshot_digest(dino_snapshot),
    }
    payload = {
        "schema_version": BASE_COMPONENTS_SCHEMA,
        "status": "complete",
        "algorithm": {
            "version": BASE_COMPONENT_ALGORITHM,
            "indexed_core_graph_version":
                indexed_core.R7_INDEXED_VISUAL_GRAPH_VERSION,
            "indexed_core_implementation_version":
                indexed_core.R7_INDEXED_VISUAL_GRAPH_IMPLEMENTATION,
            "indexed_core_component_version":
                indexed_core.R7_INDEXED_VISUAL_COMPONENT_VERSION,
            "pair_atomicity": True,
            "exact_video_sha256": True,
            "dhash_relation": "any-of-6-frame-hashes",
            "maximum_dhash_hamming": MAXIMUM_DHASH_HAMMING,
            "dino_edges": "forbidden-empty-tuple",
            "dino_input_edge_count": 0,
            "indexed_core_thresholds": config.thresholds_dict(),
            "indexed_core_config_digest": config.digest(),
            "minimum_dino_cosine_is_inert": True,
        },
        "input_bindings": {
            "graph_input": graph_binding,
            "dino_exact8": dino_binding,
            "iid_pair_maxima": iid_binding,
        },
        "validated_graph_commit_binding": graph_commit_binding,
        "validated_graph_commit_binding_sha256": _object_digest(
            graph_commit_binding
        ),
        "cross_input_binding": {
            "iid_maxima_inherits_exact_graph_commit_binding": True,
            "iid_maxima_is_validated_dino_final_child": True,
            "dino_final_remerged_exact8_before_consumption": True,
            "binding_sha256": _object_digest(graph_commit_binding),
        },
        "iid_to_base_component": mapping_rows,
        "iid_to_base_component_sha256": _object_digest(mapping_rows),
        "anchor_flags": [
            {
                "iid": iid,
                "anchor": projection.anchor_flags[iid],
            }
            for iid in projection.iids
        ],
        "components": component_rows,
        "components_sha256": _object_digest(component_rows),
        "counts": {
            "iids": len(projection.iids),
            "assets": 2 * len(projection.iids),
            "anchors": len(projection.anchor_iids),
            "base_components": components,
            "iid_pairs": len(projection.iids)
            * (len(projection.iids) - 1)
            // 2,
            "component_pairs": components * (components - 1) // 2,
            "spanning_edges": len(result.spanning_edges),
        },
        "relation_counts": relation_counts,
        "indexed_candidate_relation_counts": candidate_counts,
        "core_statistics": statistics,
        "core_provenance": result.provenance.to_dict(),
        "conservation": {
            "each_iid_exactly_one_component": True,
            "source_target_pair_atomic": True,
            "assets_equal_two_times_iids": True,
            "spanning_forest_edges_equal_assets_minus_components": True,
            "iid_pair_population_complete": True,
            "component_pair_population_complete": True,
        },
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    return payload, mapping, result


@dataclass(frozen=True)
class _Derived:
    graph_rows: tuple[Mapping[str, Any], ...]
    graph_binding: Mapping[str, Any]
    iid_artifact: Mapping[str, Any]
    projection: _Projection
    mapping: Mapping[str, str]
    base_components: Mapping[str, Any]
    component_artifact: Mapping[str, Any]
    sample_artifact: Mapping[str, Any]
    summary_base: Mapping[str, Any]
    input_paths: Mapping[str, Path]
    input_snapshot: Mapping[str, Mapping[str, Any]]


def _derive(
    *,
    graph_input_dir: Path,
    dino_final_dir: Path,
) -> _Derived:
    graph_paths = _graph_paths(graph_input_dir)
    dino_root, dino_paths = _dino_artifact_paths(dino_final_dir)
    input_paths = {
        **{
            f"graph_input/{name}": path
            for name, path in graph_paths.items()
        },
        **{
            f"dino/{name}": path
            for name, path in dino_paths.items()
        },
    }
    before = _snapshot(input_paths)
    # Formal trust entry: this revalidates all eight ranks, remerges their
    # quotient partials, and byte-compares the fixed final child artifact.
    dino_validation = dino_edges.validate_final(
        dino_final_dir,
        input_directory=graph_paths[graph_input.DONE_NAME].parent,
        output_root=dino_root,
    )
    graph = graph_input.validate_graph_input_commit(
        graph_paths[graph_input.DONE_NAME].parent
    )
    expected_graph_hashes = {
        "manifest": before[
            f"graph_input/{graph_input.MANIFEST_NAME}"
        ]["sha256"],
        "archive": before[
            f"graph_input/{graph_input.ARCHIVE_NAME}"
        ]["sha256"],
        "summary": before[
            f"graph_input/{graph_input.SUMMARY_NAME}"
        ]["sha256"],
        "done": before[
            f"graph_input/{graph_input.DONE_NAME}"
        ]["sha256"],
    }
    if graph["artifact_hashes"] != expected_graph_hashes:
        raise ValueError("graph input returned file hashes differ")
    validated_graph_binding = quotient.make_graph_commit_binding(
        artifact_digest=graph["artifact_digest"],
        artifact_hashes=graph["artifact_hashes"],
    )
    rows = tuple(graph["rows"])
    projection = _project_pairs(rows)
    iid_artifact = dino_validation.get("iid_pair_maxima")
    if (
        not isinstance(iid_artifact, Mapping)
        or iid_artifact.get("schema_version")
        != quotient.IID_PAIR_MAXIMA_SCHEMA
    ):
        raise ValueError("DINO final did not return global IID maxima")
    dino_contract = dino_validation.get("contract")
    if (
        not isinstance(dino_contract, Mapping)
        or dino_contract.get("input_artifact_digest")
        != graph["artifact_digest"]
        or dino_contract.get("input_artifacts")
        != graph["artifact_hashes"]
        or iid_artifact["contract"].get("graph_commit_binding")
        != validated_graph_binding
    ):
        raise ValueError(
            "DINO final, graph, and IID maxima bindings differ"
        )
    expected_final_paths = {
        name: dino_paths[f"final/{name}"]
        for name in (
            dino_edges.HARD_EDGES_NAME,
            dino_edges.AUDIT_EDGES_NAME,
            dino_edges.CALIBRATION_EDGES_NAME,
            dino_edges.SUMMARY_NAME,
            dino_edges.DONE_NAME,
        )
    }
    returned_paths = dino_validation.get("paths")
    if (
        not isinstance(returned_paths, Mapping)
        or set(returned_paths) != set(expected_final_paths)
        or any(
            Path(returned_paths[name]).resolve(strict=True)
            != expected_final_paths[name]
            for name in expected_final_paths
        )
    ):
        raise ValueError("DINO final returned path binding differs")
    coverage = dino_validation["summary"].get("coverage_proof")
    expected_iid_pairs = (
        len(projection.iids) * (len(projection.iids) - 1) // 2
    )
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("iid_count") != len(projection.iids)
        or coverage.get("expected_iid_pairs") != expected_iid_pairs
        or coverage.get("observed_iid_pairs") != expected_iid_pairs
        or coverage.get("quotient_partials_per_iid_pair") != 2
        or coverage.get("quotient_observed_partials")
        != 2 * expected_iid_pairs
        or coverage.get("quotient_coverage_complete") is not True
        or dino_validation["summary"].get("quotient_artifact")
        != dino_validation.get("quotient_binding")
        or dino_validation["done"].get("quotient_artifact")
        != dino_validation.get("quotient_binding")
    ):
        raise ValueError("DINO final quotient coverage binding differs")
    after = _snapshot(input_paths)
    if after != before:
        raise RuntimeError("quotient evidence inputs changed during load")
    graph_snapshot = {
        name.removeprefix("graph_input/"): record
        for name, record in after.items()
        if name.startswith("graph_input/")
    }
    dino_snapshot = {
        name.removeprefix("dino/"): record
        for name, record in after.items()
        if name.startswith("dino/")
    }
    iid_prefix = f"final/{dino_edges.IID_PAIR_MAXIMA_NAME}/"
    iid_snapshot = {
        name.removeprefix(iid_prefix): record
        for name, record in dino_snapshot.items()
        if name.startswith(iid_prefix)
    }
    base, mapping, core_result = _base_component_payload(
        projection=projection,
        graph=graph,
        graph_snapshot=graph_snapshot,
        iid_artifact=iid_artifact,
        dino_validation=dino_validation,
        dino_snapshot=dino_snapshot,
        iid_snapshot=iid_snapshot,
    )
    component_artifact = quotient.aggregate_base_component_pairs(
        iid_artifact,
        iid_to_base_component=mapping,
        iid_anchor_flags=projection.anchor_flags,
    )
    sample_artifact = quotient.build_quotient_calibration_sample(
        component_artifact,
        seed=quotient.SAMPLE_SEED,
        samples_per_stratum=quotient.DEFAULT_SAMPLE_PER_STRATUM,
    )
    iid_pairs = len(projection.iids) * (len(projection.iids) - 1) // 2
    components = len(core_result.components)
    component_pairs = components * (components - 1) // 2
    if (
        iid_artifact["contract"]["observed_iid_pairs"] != iid_pairs
        or component_artifact["contract"][
            "observed_component_pairs"
        ]
        != component_pairs
        or component_artifact["contract"]["base_components"]
        != components
        or sample_artifact["contract"]["population_count"]
        != component_pairs
        or sum(
            row["N_h"]
            for row in sample_artifact["contract"]["strata"]
        )
        != component_pairs
    ):
        raise RuntimeError("IID/component quotient conservation differs")
    summary_base = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "pipeline_schema": SCHEMA_VERSION,
        "input_bindings": dict(base["input_bindings"]),
        "validated_graph_commit_binding":
            base["validated_graph_commit_binding"],
        "validated_graph_commit_binding_sha256":
            base["validated_graph_commit_binding_sha256"],
        "cross_input_binding": dict(base["cross_input_binding"]),
        "algorithm": dict(base["algorithm"]),
        "counts": {
            **dict(base["counts"]),
            "calibration_sample_pairs":
                sample_artifact["contract"]["sample_count"],
        },
        "relation_counts": dict(base["relation_counts"]),
        "indexed_candidate_relation_counts": dict(
            base["indexed_candidate_relation_counts"]
        ),
        "conservation": dict(base["conservation"]),
        "calibration": {
            "statistical_unit": "pre-dino-base-component-pair",
            "seed": quotient.SAMPLE_SEED,
            "samples_per_stratum":
                quotient.DEFAULT_SAMPLE_PER_STRATUM,
            "score_bins": sample_artifact["contract"]["score_bins"],
            "strata": sample_artifact["contract"]["strata"],
        },
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    return _Derived(
        graph_rows=rows,
        graph_binding=validated_graph_binding,
        iid_artifact=iid_artifact,
        projection=projection,
        mapping=mapping,
        base_components=base,
        component_artifact=component_artifact,
        sample_artifact=sample_artifact,
        summary_base=summary_base,
        input_paths=input_paths,
        input_snapshot=after,
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
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


def _child_snapshot(directory: Path) -> dict[str, dict[str, Any]]:
    root = _strict_directory(
        directory,
        files=_ARRAY_ARTIFACT_FILES,
    )
    return _snapshot(
        {
            name: root / name
            for name in sorted(_ARRAY_ARTIFACT_FILES)
        }
    )


def _materialize(
    directory: Path,
    derived: _Derived,
    *,
    include_permission_contract: bool = True,
) -> dict[str, Any]:
    if any(directory.iterdir()):
        raise ValueError("quotient evidence staging directory is not empty")
    component_dir = directory / COMPONENT_MAXIMA_DIR
    sample_dir = directory / CALIBRATION_SAMPLE_DIR
    quotient.publish_artifact_directory(
        component_dir,
        derived.component_artifact,
        graph_binding=derived.graph_binding,
        iid_pair_maxima=derived.iid_artifact,
        iid_to_base_component=derived.mapping,
        iid_anchor_flags=derived.projection.anchor_flags,
        _include_permission_contract=include_permission_contract,
    )
    quotient.publish_artifact_directory(
        sample_dir,
        derived.sample_artifact,
        graph_binding=derived.graph_binding,
        component_pair_maxima=derived.component_artifact,
        _include_permission_contract=include_permission_contract,
    )
    component_snapshot = _child_snapshot(component_dir)
    sample_snapshot = _child_snapshot(sample_dir)
    base_bytes = _canonical_json_bytes(derived.base_components)
    base_record = {
        "filename": BASE_COMPONENTS_NAME,
        "sha256": hashlib.sha256(base_bytes).hexdigest(),
        "bytes": len(base_bytes),
    }
    child_records = {
        COMPONENT_MAXIMA_DIR: {
            "logical_artifact_digest":
                derived.component_artifact["artifact_digest"],
            "file_snapshot": component_snapshot,
            "file_snapshot_digest":
                _snapshot_digest(component_snapshot),
        },
        CALIBRATION_SAMPLE_DIR: {
            "logical_artifact_digest":
                derived.sample_artifact["artifact_digest"],
            "file_snapshot": sample_snapshot,
            "file_snapshot_digest": _snapshot_digest(sample_snapshot),
        },
    }
    summary = dict(derived.summary_base)
    summary["outputs"] = {
        "base_components": base_record,
        "component_pair_maxima": child_records[
            COMPONENT_MAXIMA_DIR
        ],
        "calibration_sample": child_records[
            CALIBRATION_SAMPLE_DIR
        ],
    }
    summary_bytes = _canonical_json_bytes(summary)
    summary_record = {
        "filename": SUMMARY_NAME,
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
    }
    output_binding = {
        BASE_COMPONENTS_NAME: base_record,
        SUMMARY_NAME: summary_record,
        COMPONENT_MAXIMA_DIR: child_records[COMPONENT_MAXIMA_DIR],
        CALIBRATION_SAMPLE_DIR: child_records[
            CALIBRATION_SAMPLE_DIR
        ],
    }
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "pipeline_schema": SCHEMA_VERSION,
        "artifact_closure": sorted(OUTPUT_ENTRIES),
        "artifacts": output_binding,
        "artifact_digest": _object_digest(output_binding),
        "input_artifact_digests": {
            "graph_input":
                derived.base_components["input_bindings"][
                    "graph_input"
                ]["artifact_digest"],
            "iid_pair_maxima":
                derived.iid_artifact["artifact_digest"],
            "dino_exact8_recursive_files":
                derived.base_components["input_bindings"][
                    "dino_exact8"
                ]["recursive_file_snapshot_digest"],
        },
        "iids": derived.base_components["counts"]["iids"],
        "base_components":
            derived.base_components["counts"]["base_components"],
        "iid_pairs": derived.base_components["counts"]["iid_pairs"],
        "component_pairs":
            derived.base_components["counts"]["component_pairs"],
        "thresholds_human_calibrated": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    if include_permission_contract:
        done["permission_contract"] = (
            artifact_permissions.permission_contract()
        )
    _write_exclusive(directory / BASE_COMPONENTS_NAME, base_bytes)
    _write_exclusive(directory / SUMMARY_NAME, summary_bytes)
    _write_exclusive(
        directory / DONE_NAME,
        _canonical_json_bytes(done),
    )
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _validate_materialized(directory, derived)
    return done


def _load_canonical(path: Path, name: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as error:
        raise ValueError(f"{name} JSON is invalid") from error
    if (
        not isinstance(value, dict)
        or payload != _canonical_json_bytes(value)
    ):
        raise ValueError(f"{name} JSON is not canonical")
    return value


def _validate_materialized(
    directory: Path,
    derived: _Derived,
    *,
    verify_permissions: bool = False,
) -> dict[str, Any]:
    root = _strict_directory(
        directory,
        files={
            BASE_COMPONENTS_NAME,
            SUMMARY_NAME,
            DONE_NAME,
        },
        directories={
            COMPONENT_MAXIMA_DIR,
            CALIBRATION_SAMPLE_DIR,
        },
    )
    base = _load_canonical(
        root / BASE_COMPONENTS_NAME,
        BASE_COMPONENTS_NAME,
    )
    summary = _load_canonical(root / SUMMARY_NAME, SUMMARY_NAME)
    done = _load_canonical(root / DONE_NAME, DONE_NAME)
    sealed = "permission_contract" in done
    if sealed:
        artifact_permissions.validate_permission_contract(
            done["permission_contract"]
        )
        if verify_permissions:
            artifact_permissions.assert_sealed_tree(root)
    if base != dict(derived.base_components):
        raise ValueError("base component mapping differs from recomputation")
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("artifact_closure") != sorted(OUTPUT_ENTRIES)
        or done.get("thresholds_human_calibrated") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_authorized") is not False
        or summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("thresholds_human_calibrated") is not False
        or summary.get("training_authorized") is not False
    ):
        raise ValueError("quotient evidence safety/closure differs")
    loaded_component = quotient.load_artifact_directory(
        root / COMPONENT_MAXIMA_DIR,
        graph_binding=derived.graph_binding,
        iid_pair_maxima=derived.iid_artifact,
        iid_to_base_component=derived.mapping,
        iid_anchor_flags=derived.projection.anchor_flags,
    )
    if (
        loaded_component["artifact_digest"]
        != derived.component_artifact["artifact_digest"]
    ):
        raise ValueError("component maxima logical artifact differs")
    loaded_sample = quotient.load_artifact_directory(
        root / CALIBRATION_SAMPLE_DIR,
        graph_binding=derived.graph_binding,
        component_pair_maxima=derived.component_artifact,
    )
    if (
        loaded_sample["artifact_digest"]
        != derived.sample_artifact["artifact_digest"]
    ):
        raise ValueError("calibration sample logical artifact differs")
    component_snapshot = _child_snapshot(
        root / COMPONENT_MAXIMA_DIR
    )
    sample_snapshot = _child_snapshot(
        root / CALIBRATION_SAMPLE_DIR
    )
    output_binding = {
        BASE_COMPONENTS_NAME: {
            "filename": BASE_COMPONENTS_NAME,
            "sha256": _file_digest(root / BASE_COMPONENTS_NAME),
            "bytes": (root / BASE_COMPONENTS_NAME).stat().st_size,
        },
        SUMMARY_NAME: {
            "filename": SUMMARY_NAME,
            "sha256": _file_digest(root / SUMMARY_NAME),
            "bytes": (root / SUMMARY_NAME).stat().st_size,
        },
        COMPONENT_MAXIMA_DIR: {
            "logical_artifact_digest":
                loaded_component["artifact_digest"],
            "file_snapshot": component_snapshot,
            "file_snapshot_digest":
                _snapshot_digest(component_snapshot),
        },
        CALIBRATION_SAMPLE_DIR: {
            "logical_artifact_digest":
                loaded_sample["artifact_digest"],
            "file_snapshot": sample_snapshot,
            "file_snapshot_digest": _snapshot_digest(sample_snapshot),
        },
    }
    if (
        done.get("artifacts") != output_binding
        or done.get("artifact_digest") != _object_digest(output_binding)
        or summary.get("outputs", {}).get("base_components")
        != output_binding[BASE_COMPONENTS_NAME]
        or summary.get("outputs", {}).get("component_pair_maxima")
        != output_binding[COMPONENT_MAXIMA_DIR]
        or summary.get("outputs", {}).get("calibration_sample")
        != output_binding[CALIBRATION_SAMPLE_DIR]
    ):
        raise ValueError("quotient evidence output hash chain differs")
    return {
        "directory": root,
        "base_components": base,
        "summary": summary,
        "done": done,
        "component_pair_maxima": loaded_component,
        "calibration_sample": loaded_sample,
    }


def _tree_entries(root: Path) -> dict[str, tuple[str, bytes | None]]:
    result: dict[str, tuple[str, bytes | None]] = {}

    def visit(directory: Path, prefix: str) -> None:
        status = directory.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(
            status.st_mode
        ):
            raise ValueError("artifact tree contains a non-directory")
        for entry in sorted(os.scandir(directory), key=lambda item: item.name):
            relative = f"{prefix}{entry.name}"
            path = Path(entry.path)
            child_status = path.lstat()
            if stat.S_ISLNK(child_status.st_mode):
                raise ValueError("artifact tree contains a symlink")
            if stat.S_ISDIR(child_status.st_mode):
                result[relative] = ("directory", None)
                visit(path, relative + "/")
            elif stat.S_ISREG(child_status.st_mode):
                result[relative] = ("file", path.read_bytes())
            else:
                raise ValueError("artifact tree contains a nonregular entry")

    visit(root, "")
    return result


def _compare_trees(actual: Path, expected: Path) -> None:
    if _tree_entries(actual) != _tree_entries(expected):
        raise ValueError(
            "quotient evidence commit differs byte-for-byte "
            "from full recomputation"
        )


def validate_quotient_evidence(
    directory: Path,
    *,
    graph_input_dir: Path,
    dino_final_dir: Path,
) -> dict[str, Any]:
    """Fully recompute inputs and require a byte-identical output commit."""

    derived = _derive(
        graph_input_dir=graph_input_dir,
        dino_final_dir=dino_final_dir,
    )
    target = _strict_directory(
        directory,
        files={
            BASE_COMPONENTS_NAME,
            SUMMARY_NAME,
            DONE_NAME,
        },
        directories={
            COMPONENT_MAXIMA_DIR,
            CALIBRATION_SAMPLE_DIR,
        },
    )
    target_done = _load_canonical(target / DONE_NAME, DONE_NAME)
    sealed = "permission_contract" in target_done
    comparison = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.verify-",
            dir=target.parent,
        )
    )
    try:
        _materialize(
            comparison,
            derived,
            include_permission_contract=sealed,
        )
        _compare_trees(target, comparison)
        result = _validate_materialized(
            target,
            derived,
            verify_permissions=sealed,
        )
        _assert_snapshot(derived.input_paths, derived.input_snapshot)
        return result
    finally:
        artifact_permissions.remove_staging_tree(comparison)


def build_quotient_evidence(
    *,
    graph_input_dir: Path,
    dino_final_dir: Path,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Build one immutable quotient-evidence commit or verify on resume."""

    target = output_dir.expanduser()
    if resume:
        if not target.exists() or target.is_symlink():
            raise FileNotFoundError(
                "resume requires an existing quotient-evidence commit"
            )
        return validate_quotient_evidence(
            target,
            graph_input_dir=graph_input_dir,
            dino_final_dir=dino_final_dir,
        )["done"]
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    parent = target.parent.resolve(strict=True)
    target = parent / target.name
    derived = _derive(
        graph_input_dir=graph_input_dir,
        dino_final_dir=dino_final_dir,
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.staging-",
            dir=parent,
        )
    )
    published = False
    try:
        done = _materialize(staging, derived)
        _assert_snapshot(derived.input_paths, derived.input_snapshot)
        artifact_permissions.seal_staging_tree(
            staging,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            staging,
            allow_writable_root=True,
        )
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.rename(staging, target)
        artifact_permissions.seal_published_root(target)
        published = True
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        _assert_snapshot(derived.input_paths, derived.input_snapshot)
        return done
    finally:
        if not published and staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build strict pre-DINO R7 quotient evidence",
    )
    parser.add_argument("--graph-input-dir", required=True, type=Path)
    parser.add_argument(
        "--dino-final-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    done = build_quotient_evidence(
        graph_input_dir=args.graph_input_dir,
        dino_final_dir=args.dino_final_dir,
        output_dir=args.output_dir,
        resume=args.resume,
    )
    print(_canonical_json(done))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BASE_COMPONENTS_NAME",
    "BASE_COMPONENTS_SCHEMA",
    "CALIBRATION_SAMPLE_DIR",
    "COMPONENT_MAXIMA_DIR",
    "DONE_NAME",
    "DONE_SCHEMA",
    "OUTPUT_ENTRIES",
    "SCHEMA_VERSION",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA",
    "build_quotient_evidence",
    "main",
    "validate_quotient_evidence",
]

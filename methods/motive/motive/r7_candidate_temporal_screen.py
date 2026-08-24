"""No-gradient temporal representation screen for the R7 candidate cohort.

The screen consumes three independently committed observations:

* the 947-positive plus 240 sampled-negative candidate temporal manifest;
* its label-neutral, camera-compensated CoTracker cache; and
* the original expansion-candidate DINO feature commit.

Inputs are joined by IID, never by incidental array position.  Every upstream
commit, hash chain, media digest, and numerical array contract is validated
before metrics are derived.  Evaluation uses only positive train targets as
the reference bank and excludes the query IID and visual component.  Family
retrieval is restricted to families with at least five train references and
five train components.  Held-out labels never select families or prune train
distractors.

This is a development diagnostic over pseudo labels and a provisional split.
It performs no optimization and deliberately keeps every formal, training,
generation, editing, and production gate false.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_candidate_temporal_manifest as candidate_manifest
from . import r7_candidate_track_cache as candidate_cache
from . import r7_expansion_visual_features as visual_features
from . import r7_indexed_visual_graph as indexed_graph
from . import r7_indexed_visual_graph_io as indexed_graph_io
from . import r7_visual_candidate_manifest as visual_candidates


SCREEN_SCHEMA = "motive-r7-candidate-temporal-screen-result-v1"
ROW_SCHEMA = "motive-r7-candidate-temporal-screen-result-row-v1"
DONE_SCHEMA = "motive-r7-candidate-temporal-screen-result-done-v1"
REPRESENTATION_SCHEMA = "motive-r7-candidate-motion-statistics-v1"
RETRIEVAL_PROTOCOL = (
    "positive-train-target-only-cosine-nn-exclude-iid-component-v1"
)
BINARY_PROTOCOL = (
    "heldout-positive-vs-sampled-negative-max-train-positive-cosine-v1"
)
SHUFFLE_PROTOCOL = "sha256-transition-block-order-query-only-v2"

ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (ROWS_NAME, SUMMARY_NAME, DONE_NAME)
PAYLOAD_NAMES = (ROWS_NAME, SUMMARY_NAME)

SPLITS = ("train", "validation", "test")
EVAL_SPLITS = ("validation", "test")
MINIMUM_TRAIN_REFERENCES = 5
MINIMUM_TRAIN_COMPONENTS = 5
MINIMUM_EVAL_QUERIES = 2
MINIMUM_ELIGIBLE_FAMILIES = 2
DEFAULT_SEED = 260108835
VISIBILITY_THRESHOLD = 0.5
ACTIVE_TRACK_FRACTION = 0.10
MINIMUM_ACTIVE_TRACKS = 8

TARGET_TEMPORAL = "camera_compensated_target_temporal"
DELTA_TEMPORAL = "source_to_target_delta_temporal"
TARGET_ENDPOINT = "target_endpoint"
ORDERLESS_TEMPORAL = "orderless_target_temporal"
CAMERA_NUISANCE = "target_camera_nuisance"
POOLED_DINO = "pooled_target_dino"
SHUFFLED_QUERY = "target_temporal_transition_shuffle_query"
REVERSED_QUERY = "target_temporal_physical_time_reverse_query"
MODALITIES = (
    TARGET_TEMPORAL,
    DELTA_TEMPORAL,
    TARGET_ENDPOINT,
    ORDERLESS_TEMPORAL,
    CAMERA_NUISANCE,
    POOLED_DINO,
    SHUFFLED_QUERY,
    REVERSED_QUERY,
)
REFERENCE_FEATURE = {
    TARGET_TEMPORAL: TARGET_TEMPORAL,
    DELTA_TEMPORAL: DELTA_TEMPORAL,
    TARGET_ENDPOINT: TARGET_ENDPOINT,
    ORDERLESS_TEMPORAL: ORDERLESS_TEMPORAL,
    CAMERA_NUISANCE: CAMERA_NUISANCE,
    POOLED_DINO: POOLED_DINO,
    SHUFFLED_QUERY: TARGET_TEMPORAL,
    REVERSED_QUERY: TARGET_TEMPORAL,
}

SAFETY_FIELDS = (
    "formal_evidence",
    "formal_split",
    "formal_report",
    "human_labels_asserted",
    "thresholds_human_calibrated",
    "training_authorized",
    "direct_training_supervision_allowed",
    "generation_authorized",
    "editing_authorized",
    "production_decision",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-12


class CandidateTemporalScreenError(ValueError):
    """An input, evaluation, or immutable-output contract is invalid."""


@dataclass(frozen=True)
class _Example:
    iid: str
    label_class: str
    family: str
    split: str
    component_id: str
    fresh: bool
    sampling_weight: float
    features: Mapping[str, np.ndarray]
    motion_energy: float


@dataclass(frozen=True)
class _Inputs:
    candidate: Mapping[str, Any]
    cache: Mapping[str, Any]
    visual: Mapping[str, Any]
    original_rows: tuple[Mapping[str, Any], ...]
    binding: Mapping[str, Any]
    identities: Mapping[str, tuple[int, ...]]
    media_identities: Mapping[str, tuple[int, ...]]


def _safety_flags() -> dict[str, bool]:
    return {field: False for field in SAFETY_FIELDS}


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


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(dict(row)) + "\n").encode("utf-8")
        for row in rows
    )


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


def _require_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise CandidateTemporalScreenError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {raw}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CandidateTemporalScreenError(
            f"invalid strict JSON object: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CandidateTemporalScreenError(
            f"{path} does not contain a JSON object"
        )
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CandidateTemporalScreenError(
                    f"{path}:{line_number} is blank"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CandidateTemporalScreenError(
                    f"{path}:{line_number} is invalid JSON"
                ) from error
            if (
                not isinstance(value, dict)
                or line != _canonical_json(value) + "\n"
            ):
                raise CandidateTemporalScreenError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    return rows


def _stat_identity(path: Path) -> tuple[int, ...]:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise CandidateTemporalScreenError(
            f"input is not one regular unlinked file: {path}"
        )
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _capture_identities(
    paths: Iterable[Path],
) -> dict[str, tuple[int, ...]]:
    return {
        str(path.resolve(strict=True)): _stat_identity(path)
        for path in sorted(
            {item.resolve(strict=True) for item in paths},
            key=str,
        )
    }


def _assert_identities(
    identities: Mapping[str, tuple[int, ...]],
) -> None:
    for raw_path, expected in identities.items():
        path = Path(raw_path)
        if _stat_identity(path) != tuple(expected):
            raise RuntimeError(f"input changed during screen: {path}")


def _artifact_records(
    directory: Path,
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": _file_digest(directory / name),
            "bytes": int((directory / name).stat().st_size),
        }
        for name in names
    }


def _validate_original_candidate_commit(
    manifest_path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    expected = _require_sha256(
        expected_sha256,
        context="expected original candidate manifest SHA",
    )
    unresolved = manifest_path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise FileNotFoundError(unresolved)
    manifest = unresolved.resolve(strict=True)
    if manifest.name != visual_candidates.CANDIDATES_NAME:
        raise CandidateTemporalScreenError(
            "original candidate manifest filename differs"
        )
    root = manifest.parent
    actual = {entry.name for entry in root.iterdir()}
    expected_names = set(visual_candidates.OUTPUT_ARTIFACT_NAMES)
    if actual != expected_names:
        raise CandidateTemporalScreenError(
            "original candidate commit artifact set differs"
        )
    artifact_permissions.assert_sealed_tree(root)
    if _file_digest(manifest) != expected:
        raise CandidateTemporalScreenError(
            "original candidate manifest external SHA differs"
        )
    rows = visual_features.load_candidate_manifest(manifest)
    summary_path = root / visual_candidates.SUMMARY_NAME
    done_path = root / visual_candidates.DONE_NAME
    summary = _load_object(summary_path)
    done = _load_object(done_path)
    manifest_sha = _file_digest(manifest)
    summary_sha = _file_digest(summary_path)
    output_sha = {
        visual_candidates.CANDIDATES_NAME: manifest_sha,
        visual_candidates.SUMMARY_NAME: summary_sha,
    }
    if (
        summary.get("schema_version") != visual_candidates.SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("output", {}).get("rows") != len(rows)
        or summary.get("output", {}).get("sha256") != manifest_sha
        or done.get("schema_version") != visual_candidates.DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("output_rows") != len(rows)
        or done.get("output_sha256") != output_sha
        or done.get("artifact_digest") != _object_digest(output_sha)
        or any(
            done.get(field) is not False
            for field in (
                "split_assigned",
                "human_labels_asserted",
                "training_eligible",
            )
        )
    ):
        raise CandidateTemporalScreenError(
            "original candidate commit hash/semantic chain differs"
        )
    return rows, summary, done


def _implementation_provenance() -> dict[str, Any]:
    modules = (
        Path(__file__).resolve(strict=True),
        Path(artifact_permissions.__file__).resolve(strict=True),
        Path(candidate_manifest.__file__).resolve(strict=True),
        Path(candidate_cache.__file__).resolve(strict=True),
        Path(indexed_graph.__file__).resolve(strict=True),
        Path(indexed_graph_io.__file__).resolve(strict=True),
        Path(visual_features.__file__).resolve(strict=True),
        Path(visual_candidates.__file__).resolve(strict=True),
    )
    files = {
        path.name: {"sha256": _file_digest(path)}
        for path in sorted(modules, key=lambda item: item.name)
    }
    return {
        "files": files,
        "bundle_sha256": _object_digest(
            {name: record["sha256"] for name, record in files.items()}
        ),
    }


def _validate_component_assignment(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    component_splits: dict[str, set[str]] = {}
    assignments: list[dict[str, Any]] = []
    for row in rows:
        assignment = row["assignment"]
        split = str(assignment["split"])
        component = str(assignment["component_id"])
        if split not in SPLITS:
            raise CandidateTemporalScreenError(
                f"iid={row['iid']} has an invalid split"
            )
        component_splits.setdefault(component, set()).add(split)
        if split in EVAL_SPLITS and assignment["fresh"] is not True:
            raise CandidateTemporalScreenError(
                f"iid={row['iid']} held-out assignment is not fresh"
            )
        if assignment["fresh"] is False and split != "train":
            raise CandidateTemporalScreenError(
                f"iid={row['iid']} nonfresh assignment escaped train"
            )
        assignments.append(
            {
                "iid": row["iid"],
                "assignment": dict(assignment),
            }
        )
    leaking = {
        component: sorted(splits)
        for component, splits in component_splits.items()
        if len(splits) != 1
    }
    if leaking:
        raise CandidateTemporalScreenError(
            "visual component spans split assignments: "
            + _canonical_json(leaking)
        )
    registry = {
        component: next(iter(splits))
        for component, splits in sorted(component_splits.items())
    }
    return {
        "schema_version": candidate_manifest.ASSIGNMENT_SCHEMA,
        "status": "frozen_diagnostic_assignment",
        "split_is_formal": False,
        "component_disjoint": True,
        "rows": len(rows),
        "components": len(registry),
        "assignment_digest": _object_digest(assignments),
        "component_split_digest": _object_digest(registry),
        "split_counts": {
            split: sum(
                row["assignment"]["split"] == split for row in rows
            )
            for split in SPLITS
        },
    }


def _validate_split_topology_binding(
    candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Prove whether a retained DINO edge changed the realized partition."""

    summary = candidate.get("summary")
    bindings = (
        summary.get("input_bindings")
        if isinstance(summary, Mapping)
        else None
    )
    indexed = (
        bindings.get("indexed_graph")
        if isinstance(bindings, Mapping)
        else None
    )
    if not isinstance(indexed, Mapping):
        raise CandidateTemporalScreenError(
            "candidate commit lacks indexed-graph binding"
    )
    raw_root = indexed.get("path")
    files = indexed.get("files")
    name = indexed_graph_io.SPANNING_EDGES_NAME
    summary_name = indexed_graph_io.SUMMARY_NAME
    record = files.get(name) if isinstance(files, Mapping) else None
    summary_record = (
        files.get(summary_name) if isinstance(files, Mapping) else None
    )
    if (
        type(raw_root) is not str
        or not isinstance(record, Mapping)
        or set(record) != {"sha256", "bytes"}
        or not isinstance(summary_record, Mapping)
        or set(summary_record) != {"sha256", "bytes"}
    ):
        raise CandidateTemporalScreenError(
            "candidate indexed spanning-edge binding differs"
        )
    unresolved_root = Path(raw_root).expanduser()
    if unresolved_root.is_symlink() or not unresolved_root.is_dir():
        raise CandidateTemporalScreenError(
            "bound indexed graph root is not a real directory"
        )
    root = unresolved_root.resolve(strict=True)
    path = root / name
    summary_path = root / summary_name

    def read_bound_bytes(
        bound_path: Path,
        bound_record: Mapping[str, Any],
        *,
        context: str,
    ) -> tuple[bytes, str, int]:
        expected_sha = _require_sha256(
            bound_record.get("sha256"),
            context=f"{context} SHA",
        )
        expected_bytes = bound_record.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise CandidateTemporalScreenError(
                f"{context} byte binding differs"
            )
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:  # pragma: no cover - supported target platforms
            raise CandidateTemporalScreenError(
                "O_NOFOLLOW is required for indexed-graph validation"
            )
        flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(bound_path, flags)
        except OSError as error:
            raise CandidateTemporalScreenError(
                f"{context} cannot be opened without symlink traversal"
            ) from error
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_bytes
            ):
                raise CandidateTemporalScreenError(
                    f"{context} is not one bound regular file"
                )
            chunks: list[bytes] = []
            remaining = expected_bytes
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise CandidateTemporalScreenError(
                    f"{context} grew while being read"
                )
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in identity_fields
        ):
            raise CandidateTemporalScreenError(
                f"{context} changed while being read"
            )
        payload = b"".join(chunks)
        if (
            len(payload) != expected_bytes
            or hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise CandidateTemporalScreenError(
                f"live {context} binding differs"
            )
        return payload, expected_sha, expected_bytes

    raw, expected_sha, expected_bytes = read_bound_bytes(
        path,
        record,
        context="indexed spanning-edge file",
    )
    summary_raw, _summary_sha, _summary_bytes = read_bound_bytes(
        summary_path,
        summary_record,
        context="indexed summary file",
    )
    try:
        indexed_summary = json.loads(
            summary_raw.decode("utf-8"),
            parse_constant=lambda raw_value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {raw_value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        raise CandidateTemporalScreenError(
            "bound indexed summary is invalid strict JSON"
        ) from error
    if (
        not isinstance(indexed_summary, dict)
        or summary_raw != _pretty_json_bytes(indexed_summary)
        or indexed_summary.get("schema_version")
        != indexed_graph_io.SUMMARY_SCHEMA
        or indexed_summary.get("status") != "complete"
    ):
        raise CandidateTemporalScreenError(
            "bound indexed summary schema/canonical encoding differs"
        )
    outputs = indexed_summary.get("outputs")
    output_record = (
        outputs.get(name) if isinstance(outputs, Mapping) else None
    )
    if (
        not isinstance(output_record, Mapping)
        or set(output_record) != {"rows", "sha256", "order"}
        or output_record.get("sha256") != expected_sha
        or output_record.get("order")
        != "canonical-endpoints-relation-value"
    ):
        raise CandidateTemporalScreenError(
            "indexed summary spanning-edge receipt differs"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CandidateTemporalScreenError(
            "indexed spanning edges are not UTF-8"
        ) from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        text.splitlines(keepends=True),
        start=1,
    ):
        if not line.endswith("\n") or line == "\n":
            raise CandidateTemporalScreenError(
                f"{path}:{line_number} is not canonical JSONL"
            )
        try:
            row = json.loads(
                line[:-1],
                parse_constant=lambda raw_value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {raw_value}")
                ),
            )
            canonical = _canonical_json(row)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise CandidateTemporalScreenError(
                f"{path}:{line_number} is invalid strict JSON"
            ) from error
        if not isinstance(row, dict) or line != canonical + "\n":
            raise CandidateTemporalScreenError(
                f"{path}:{line_number} is not canonical JSONL"
            )
        rows.append(row)

    allowed_relations = (
        "paired_sample",
        "exact_sha256",
        "dhash_hamming",
        "dino_cosine",
    )
    edge_fields = {
        "schema_version",
        "left_iid",
        "left_role",
        "right_iid",
        "right_role",
        "relation",
        "value",
    }
    relation_counts = Counter()
    edge_iids: set[str] = set()
    paired_iids: set[str] = set()
    previous_key: tuple[Any, ...] | None = None
    for line_number, row in enumerate(rows, start=1):
        if (
            set(row) != edge_fields
            or row.get("schema_version") != indexed_graph_io.EDGE_ROW_SCHEMA
        ):
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} schema differs"
            )
        left_iid = row.get("left_iid")
        right_iid = row.get("right_iid")
        if any(
            type(iid) is not str
            or not iid
            or iid != iid.strip()
            or "\x00" in iid
            for iid in (left_iid, right_iid)
        ):
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} IID differs"
            )
        left_role = row.get("left_role")
        right_role = row.get("right_role")
        if (
            type(left_role) is not str
            or left_role not in indexed_graph.VALID_ASSET_ROLES
            or type(right_role) is not str
            or right_role not in indexed_graph.VALID_ASSET_ROLES
        ):
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} role differs"
            )
        left = (left_iid, left_role)
        right = (right_iid, right_role)
        if left >= right:
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} endpoints differ"
            )
        relation = row.get("relation")
        if relation not in allowed_relations:
            raise CandidateTemporalScreenError(
                "indexed spanning edge has an unknown relation"
            )
        value = row.get("value")
        if relation in {"paired_sample", "exact_sha256"}:
            valid_value = value is None
        elif relation == "dhash_hamming":
            valid_value = (
                type(value) is int
                and 0 <= value <= indexed_graph.MAXIMUM_DHASH_HAMMING
            )
        else:
            valid_value = (
                type(value) is float
                and math.isfinite(value)
                and indexed_graph.DINO_HARD_THRESHOLD <= value <= 1.0
                and value == float(round(value, 12))
            )
        if not valid_value:
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} value differs"
            )
        if relation == "paired_sample" and (
            left_iid != right_iid
            or left_role != "source"
            or right_role != "target"
        ):
            raise CandidateTemporalScreenError(
                f"indexed spanning edge {line_number} pair differs"
            )
        edge_iids.update((left_iid, right_iid))
        if relation == "paired_sample":
            paired_iids.add(left_iid)
        edge_key = (
            left,
            right,
            relation,
            -2.0 if value is None else float(value),
        )
        if previous_key is not None and edge_key <= previous_key:
            raise CandidateTemporalScreenError(
                "indexed spanning edges are not strictly canonical ordered"
            )
        previous_key = edge_key
        relation_counts[str(relation)] += 1

    counts = indexed_summary.get("counts")
    statistics = indexed_summary.get("statistics")
    summary_relation_counts = (
        statistics.get("relation_counts")
        if isinstance(statistics, Mapping)
        else None
    )
    expected_relation_counts = {
        relation: int(relation_counts[relation])
        for relation in allowed_relations
    }
    if (
        type(output_record.get("rows")) is not int
        or output_record["rows"] != len(rows)
        or not isinstance(counts, Mapping)
        or type(counts.get("total_iids")) is not int
        or type(counts.get("assets")) is not int
        or type(counts.get("components")) is not int
        or type(counts.get("spanning_edges")) is not int
        or counts["total_iids"] <= 0
        or counts["assets"] != 2 * counts["total_iids"]
        or not 0 < counts["components"] <= counts["assets"]
        or counts["spanning_edges"] != len(rows)
        or len(rows) != counts["assets"] - counts["components"]
        or counts["total_iids"] != len(edge_iids)
        or paired_iids != edge_iids
        or type(summary_relation_counts) is not dict
        or set(summary_relation_counts) != set(allowed_relations)
        or any(
            type(value) is not int or value < 0
            for value in summary_relation_counts.values()
        )
        or summary_relation_counts != expected_relation_counts
    ):
        raise CandidateTemporalScreenError(
            "indexed spanning-edge row census/binding differs"
        )
    indexed_artifact_digest = _require_sha256(
        indexed.get("artifact_digest"),
        context="indexed graph artifact digest",
    )
    retained_dino = int(relation_counts["dino_cosine"])
    return {
        "schema_version":
            "motive-r7-realized-split-topology-dino-audit-v1",
        "indexed_graph_artifact_digest": indexed_artifact_digest,
        "spanning_edges_path": str(path),
        "spanning_edges_sha256": expected_sha,
        "spanning_edges_bytes": expected_bytes,
        "spanning_edge_rows": len(rows),
        "relation_counts": {
            relation: int(relation_counts[relation])
            for relation in (
                "paired_sample",
                "exact_sha256",
                "dhash_hamming",
                "dino_cosine",
            )
        },
        "retained_dino_spanning_edges": retained_dino,
        "dino_changed_realized_component_topology":
            retained_dino > 0,
        "realized_partition_is_dino_edge_free":
            retained_dino == 0,
        "relative_motion_vs_dino_diagnostic_is_split_confounded":
            retained_dino > 0,
        "formal_superiority_claim_allowed": False,
    }, path


def _validate_inputs(
    *,
    candidate_manifest_dir: Path,
    expected_candidate_manifest_done_sha256: str,
    track_cache_final: Path,
    expected_track_cache_done_sha256: str,
    visual_features_final: Path,
    expected_visual_features_done_sha256: str,
    visual_candidates_manifest: Path,
    expected_visual_candidates_sha256: str,
    verify_source_shards: bool,
    rehash_videos: bool,
) -> _Inputs:
    candidate_root = candidate_manifest_dir.expanduser().resolve(strict=True)
    candidate_done_path = candidate_root / candidate_manifest.DONE_NAME
    expected_candidate_done = _require_sha256(
        expected_candidate_manifest_done_sha256,
        context="expected candidate temporal done SHA",
    )
    if _file_digest(candidate_done_path) != expected_candidate_done:
        raise CandidateTemporalScreenError(
            "candidate temporal external done SHA differs"
        )
    candidate = candidate_manifest.validate_candidate_temporal_manifest(
        candidate_root
    )
    split_topology, split_topology_path = (
        _validate_split_topology_binding(candidate)
    )

    cache_root = track_cache_final.expanduser().resolve(strict=True)
    cache_done_path = cache_root / candidate_cache.DONE_NAME
    expected_cache_done = _require_sha256(
        expected_track_cache_done_sha256,
        context="expected candidate track-cache done SHA",
    )
    if _file_digest(cache_done_path) != expected_cache_done:
        raise CandidateTemporalScreenError(
            "candidate track-cache external done SHA differs"
        )
    cache = candidate_cache.validate_commit(
        cache_root,
        input_dir=candidate_root,
        expected_input_done_sha256=expected_candidate_done,
        final=True,
        verify_source_shards=verify_source_shards,
    )

    original_rows, original_summary, original_done = (
        _validate_original_candidate_commit(
            visual_candidates_manifest,
            expected_sha256=expected_visual_candidates_sha256,
        )
    )
    original_path = visual_candidates_manifest.expanduser().resolve(
        strict=True
    )

    visual_root = visual_features_final.expanduser().resolve(strict=True)
    artifact_permissions.assert_sealed_tree(visual_root)
    visual_done_path = visual_root / visual_features.DONE_NAME
    expected_visual_done = _require_sha256(
        expected_visual_features_done_sha256,
        context="expected visual-feature done SHA",
    )
    if _file_digest(visual_done_path) != expected_visual_done:
        raise CandidateTemporalScreenError(
            "visual-feature external done SHA differs"
        )
    visual = visual_features.validate_final(
        visual_root,
        input_manifest=original_path,
        output_root=visual_root.parent,
        verify_source_shards=verify_source_shards,
        rehash_videos=rehash_videos,
    )

    candidate_rows = candidate["rows"]
    cache_rows = cache["rows"]
    if len(candidate_rows) != len(cache_rows):
        raise CandidateTemporalScreenError(
            "candidate manifest/cache row census differs"
        )
    cache_by_iid: dict[str, int] = {}
    for index, row in enumerate(cache_rows):
        iid = str(row["iid"])
        if iid in cache_by_iid:
            raise CandidateTemporalScreenError(
                f"duplicate track-cache IID: {iid}"
            )
        cache_by_iid[iid] = index

    original_by_iid: dict[str, int] = {}
    for index, row in enumerate(original_rows):
        iid = str(row["iid"])
        if iid in original_by_iid:
            raise CandidateTemporalScreenError(
                f"duplicate original candidate IID: {iid}"
            )
        original_by_iid[iid] = index
    visual_by_iid: dict[str, int] = {}
    for index, row in enumerate(visual["rows"]):
        iid = str(row["iid"])
        if iid in visual_by_iid:
            raise CandidateTemporalScreenError(
                f"duplicate visual-feature IID: {iid}"
            )
        visual_by_iid[iid] = index

    for row in candidate_rows:
        iid = str(row["iid"])
        if iid not in cache_by_iid or iid not in original_by_iid:
            raise CandidateTemporalScreenError(
                f"iid={iid} is absent from an upstream observation"
            )
        original_index = original_by_iid[iid]
        visual_index = visual_by_iid.get(iid)
        if visual_index is None:
            raise CandidateTemporalScreenError(
                f"iid={iid} lacks expansion visual features"
            )
        original = original_rows[original_index]
        feature_row = visual["rows"][visual_index]
        binding = row["source_bindings"]["candidate"]
        media = row["source_bindings"]["media"]
        if (
            binding["row_index"] != original_index
            or binding["candidate_row_sha256"]
            != _object_digest(original)
            or binding["artifact_digest"]
            != original_done["artifact_digest"]
            or feature_row["input_row_sha256"]
            != binding["candidate_row_sha256"]
            or original["src_video"] != row["src_video"]
            or original["tgt_video"] != row["tgt_video"]
            or feature_row["source"]["video_sha256"]
            != media["src_video"]["sha256"]
            or feature_row["target"]["video_sha256"]
            != media["tgt_video"]["sha256"]
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} candidate/track/DINO/media binding differs"
            )
    if {
        row["source_bindings"]["candidate"]["artifact_digest"]
        for row in candidate_rows
    } != {original_done["artifact_digest"]}:
        raise CandidateTemporalScreenError(
            "candidate temporal rows bind another original candidate commit"
        )

    assignment = _validate_component_assignment(candidate_rows)
    candidate_paths = [
        candidate_root / name
        for name in candidate_manifest.OUTPUT_NAMES
    ]
    cache_paths = [
        cache_root / name for name in candidate_cache.OUTPUT_NAMES
    ]
    original_paths = [
        original_path.parent / name
        for name in visual_candidates.OUTPUT_ARTIFACT_NAMES
    ]
    visual_paths = [
        visual_root / name
        for name in (
            visual_features.ARCHIVE_NAME,
            visual_features.MANIFEST_NAME,
            visual_features.SUMMARY_NAME,
            visual_features.DONE_NAME,
        )
    ]
    identities = _capture_identities(
        [
            *candidate_paths,
            *cache_paths,
            *original_paths,
            *visual_paths,
            split_topology_path,
        ]
    )

    data_roots = {
        row["source_bindings"]["media"]["data_root"]
        for row in candidate_rows
    }
    if len(data_roots) != 1:
        raise CandidateTemporalScreenError(
            "candidate temporal media data roots differ"
        )
    data_root = Path(next(iter(data_roots))).resolve(strict=True)
    media_paths: set[Path] = set()
    for row in candidate_rows:
        for role in ("src_video", "tgt_video"):
            record = row["source_bindings"]["media"][role]
            path = (data_root / record["relative_path"]).resolve(strict=True)
            if data_root not in path.parents:
                raise CandidateTemporalScreenError(
                    f"iid={row['iid']} media escapes data root"
                )
            if (
                _file_digest(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]
            ):
                raise CandidateTemporalScreenError(
                    f"iid={row['iid']} live media binding differs"
                )
            media_paths.add(path)
    media_identities = _capture_identities(media_paths)

    binding = {
        "candidate_temporal_manifest": {
            "directory": str(candidate_root),
            "expected_done_sha256": expected_candidate_done,
            "artifact_digest": candidate["done"]["artifact_digest"],
            "files": _artifact_records(
                candidate_root,
                candidate_manifest.OUTPUT_NAMES,
            ),
        },
        "candidate_track_cache": {
            "directory": str(cache_root),
            "expected_done_sha256": expected_cache_done,
            "artifact_digest": cache["done"]["artifact_digest"],
            "contract_sha256": cache["summary"]["contract_sha256"],
            "files": _artifact_records(
                cache_root,
                candidate_cache.OUTPUT_NAMES,
            ),
        },
        "original_visual_candidates": {
            "manifest": str(original_path),
            "expected_manifest_sha256":
                _require_sha256(
                    expected_visual_candidates_sha256,
                    context="expected original candidate manifest SHA",
                ),
            "artifact_digest": original_done["artifact_digest"],
            "rows": len(original_rows),
            "files": _artifact_records(
                original_path.parent,
                visual_candidates.OUTPUT_ARTIFACT_NAMES,
            ),
            "summary_schema": original_summary["schema_version"],
        },
        "expansion_visual_features": {
            "directory": str(visual_root),
            "expected_done_sha256": expected_visual_done,
            "input_manifest_sha256":
                visual["summary"]["input_manifest_sha256"],
            "common_contract_sha256":
                visual["summary"]["common_contract_sha256"],
            "files": _artifact_records(
                visual_root,
                (
                    visual_features.ARCHIVE_NAME,
                    visual_features.MANIFEST_NAME,
                    visual_features.SUMMARY_NAME,
                    visual_features.DONE_NAME,
                ),
            ),
        },
        "assignment": assignment,
        "realized_split_topology": split_topology,
        "validation": {
            "source_shards_revalidated": bool(verify_source_shards),
            "live_videos_rehashed": bool(rehash_videos),
            "candidate_subset_live_media_rehashed": True,
            "joined_by": "iid",
        },
    }
    return _Inputs(
        candidate=candidate,
        cache=cache,
        visual=visual,
        original_rows=tuple(original_rows),
        binding=binding,
        identities=identities,
        media_identities=media_identities,
    )


def _vector_distribution(
    vectors: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return fixed robust 2-D vector and magnitude statistics."""

    value = np.asarray(vectors, dtype=np.float64)
    keep = np.asarray(valid, dtype=bool)
    if value.ndim != 2 or value.shape[1] != 2 or keep.shape != value.shape[:1]:
        raise CandidateTemporalScreenError(
            "motion vectors/validity have incompatible shapes"
        )
    selected = value[keep]
    if not len(selected):
        return np.zeros(15, dtype=np.float64)
    if not np.isfinite(selected).all():
        raise CandidateTemporalScreenError(
            "motion statistics received non-finite vectors"
        )
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    x = np.quantile(selected[:, 0], quantiles)
    y = np.quantile(selected[:, 1], quantiles)
    speed = np.linalg.norm(selected, axis=1)
    magnitude = np.quantile(speed, (0.25, 0.50, 0.75, 0.90, 0.95))
    return np.concatenate((x, y, magnitude)).astype(
        np.float64,
        copy=False,
    )


def _active_track_indices(
    tracks: np.ndarray,
    visibility: np.ndarray,
) -> np.ndarray:
    frames, track_count, _ = tracks.shape
    visible = visibility >= VISIBILITY_THRESHOLD
    transition_visible = visible[:-1] & visible[1:]
    step = np.diff(tracks.astype(np.float64), axis=0) * (frames - 1)
    speed = np.linalg.norm(step, axis=2)
    scores = np.full(track_count, -np.inf, dtype=np.float64)
    minimum_transitions = max(2, int(math.ceil(0.50 * (frames - 1))))
    eligible = np.flatnonzero(
        np.sum(transition_visible, axis=0) >= minimum_transitions
    )
    if not len(eligible):
        return np.zeros(0, dtype=np.int64)
    masked = np.where(
        transition_visible[:, eligible],
        speed[:, eligible],
        np.nan,
    )
    scores[eligible] = np.nanquantile(masked, 0.90, axis=0)
    count = min(
        len(eligible),
        max(
            MINIMUM_ACTIVE_TRACKS,
            int(math.ceil(ACTIVE_TRACK_FRACTION * len(eligible))),
        ),
    )
    ordered = sorted(
        eligible.tolist(),
        key=lambda index: (-float(scores[index]), int(index)),
    )
    return np.asarray(ordered[:count], dtype=np.int64)


def _motion_descriptors(
    tracks: Any,
    visibility: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ordered, endpoint, and orderless target motion descriptors."""

    track_array = np.asarray(tracks, dtype=np.float64)
    visibility_array = np.asarray(visibility, dtype=np.float64)
    if (
        track_array.ndim != 3
        or track_array.shape[2] != 2
        or track_array.shape[:2] != visibility_array.shape
        or track_array.shape[0] < 3
        or not np.isfinite(track_array).all()
        or not np.isfinite(visibility_array).all()
        or bool(
            (
                (visibility_array < 0.0)
                | (visibility_array > 1.0)
            ).any()
        )
    ):
        raise CandidateTemporalScreenError(
            "invalid stabilized-track descriptor input"
        )
    frame_count = track_array.shape[0]
    active = _active_track_indices(track_array, visibility_array)
    sequence = np.zeros((frame_count - 1, 15), dtype=np.float64)
    if len(active):
        steps = np.diff(track_array[:, active], axis=0) * (
            frame_count - 1
        )
        visible = visibility_array[:, active] >= VISIBILITY_THRESHOLD
        transition_visible = visible[:-1] & visible[1:]
        for index in range(frame_count - 1):
            sequence[index] = _vector_distribution(
                steps[index],
                transition_visible[index],
            )
        endpoint_vectors = (
            track_array[-1, active] - track_array[0, active]
        )
        endpoint_valid = visible[-1] & visible[0]
        endpoint = _vector_distribution(
            endpoint_vectors,
            endpoint_valid,
        )
    else:
        endpoint = np.zeros(15, dtype=np.float64)
    orderless = np.concatenate(
        (
            np.mean(sequence, axis=0),
            np.std(sequence, axis=0),
            np.quantile(sequence, 0.25, axis=0),
            np.quantile(sequence, 0.50, axis=0),
            np.quantile(sequence, 0.75, axis=0),
            np.max(sequence, axis=0),
        )
    )
    # The upper-quartile speed statistic is robust to a handful of tracker
    # spikes while retaining short actions.
    motion_energy = float(np.mean(sequence[:, 13]))
    return sequence.reshape(-1), endpoint, orderless, motion_energy


def _camera_descriptor(cumulative_affines: Any) -> np.ndarray:
    matrices = np.asarray(cumulative_affines, dtype=np.float64)
    if (
        matrices.ndim != 3
        or matrices.shape[1:] != (2, 3)
        or len(matrices) < 3
        or not np.isfinite(matrices).all()
    ):
        raise CandidateTemporalScreenError(
            "invalid cumulative camera-affine input"
        )
    center = np.asarray([0.5, 0.5], dtype=np.float64)
    parameters = np.empty((len(matrices), 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        linear = matrix[:, :2]
        determinant = float(np.linalg.det(linear))
        if determinant <= 0.0:
            raise CandidateTemporalScreenError(
                "camera affine contains a reflection/nonpositive scale"
            )
        projected = linear @ center + matrix[:, 2]
        parameters[index] = (
            projected[0] - center[0],
            projected[1] - center[1],
            math.atan2(float(linear[1, 0]), float(linear[0, 0])),
            0.5 * math.log(determinant),
        )
    parameters[:, 2] = np.unwrap(parameters[:, 2])
    parameters -= parameters[0]
    return parameters.reshape(-1)


def _shuffle_indices(iid: str, frames: int, *, seed: int) -> np.ndarray:
    if frames < 3:
        raise CandidateTemporalScreenError(
            "transition-shuffle control requires at least three blocks"
        )
    prefix = (
        f"{SHUFFLE_PROTOCOL}\0{seed}\0{iid}\0".encode("utf-8")
    )
    ordered = sorted(
        range(frames),
        key=lambda index: (
            hashlib.sha256(
                prefix + str(index).encode("ascii")
            ).digest(),
            index,
        ),
    )
    identity = list(range(frames))
    reverse = list(reversed(identity))
    if ordered == identity or ordered == reverse:
        ordered = ordered[1:] + ordered[:1]
    return np.asarray(ordered, dtype=np.int64)


def _normalise_feature(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(vector) or not np.isfinite(vector).all():
        raise CandidateTemporalScreenError(
            "representation vector is empty or non-finite"
        )
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return np.zeros_like(vector)
    return vector / norm


def _example_features(
    *,
    iid: str,
    cache_arrays: Mapping[str, np.ndarray],
    cache_index: int,
    visual_arrays: Mapping[str, np.ndarray],
    visual_index: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    source_tracks = cache_arrays["source_stabilized_tracks"][cache_index]
    source_visibility = cache_arrays["source_visibility"][cache_index]
    target_tracks = cache_arrays["target_stabilized_tracks"][cache_index]
    target_visibility = cache_arrays["target_visibility"][cache_index]
    source_temporal, _source_endpoint, _source_orderless, _ = (
        _motion_descriptors(source_tracks, source_visibility)
    )
    target_temporal, endpoint, orderless, motion_energy = (
        _motion_descriptors(target_tracks, target_visibility)
    )
    target_sequence = target_temporal.reshape(
        len(target_tracks) - 1,
        -1,
    )
    shuffled = _shuffle_indices(iid, len(target_sequence), seed=seed)
    # This is a pure order ablation: it preserves the exact clean transition
    # blocks, active-track set, visibility decisions, and displacement scale.
    shuffled_temporal = target_sequence[shuffled].reshape(-1)
    reversed_temporal, _, _, _ = _motion_descriptors(
        target_tracks[::-1],
        target_visibility[::-1],
    )
    camera = _camera_descriptor(
        cache_arrays["target_cumulative_affines"][cache_index]
    )
    dino = np.asarray(
        visual_arrays["target_dino_cls"][visual_index],
        dtype=np.float64,
    )
    if dino.ndim != 2 or not np.isfinite(dino).all():
        raise CandidateTemporalScreenError(
            f"iid={iid} target DINO feature shape differs"
        )
    features = {
        TARGET_TEMPORAL: _normalise_feature(target_temporal),
        DELTA_TEMPORAL: _normalise_feature(
            target_temporal - source_temporal
        ),
        TARGET_ENDPOINT: _normalise_feature(endpoint),
        ORDERLESS_TEMPORAL: _normalise_feature(orderless),
        CAMERA_NUISANCE: _normalise_feature(camera),
        POOLED_DINO: _normalise_feature(np.mean(dino, axis=0)),
        SHUFFLED_QUERY: _normalise_feature(shuffled_temporal),
        REVERSED_QUERY: _normalise_feature(reversed_temporal),
    }
    if set(features) != set(MODALITIES):
        raise RuntimeError("representation registry is incomplete")
    return features, motion_energy


def _build_examples(
    inputs: _Inputs,
    *,
    seed: int,
) -> tuple[list[_Example], dict[str, Any]]:
    candidate_rows = inputs.candidate["rows"]
    cache_rows = inputs.cache["rows"]
    cache_arrays = inputs.cache["arrays"]
    visual_rows = inputs.visual["rows"]
    visual_arrays = inputs.visual["arrays"]
    cache_by_iid = {
        str(row["iid"]): index for index, row in enumerate(cache_rows)
    }
    visual_by_iid = {
        str(row["iid"]): index for index, row in enumerate(visual_rows)
    }

    examples: list[_Example] = []
    reasons: Counter[str] = Counter()
    counts_by_stage: Counter[str] = Counter()
    for row in candidate_rows:
        counts_by_stage["input_rows"] += 1
        iid = str(row["iid"])
        cache_index = cache_by_iid[iid]
        visual_index = visual_by_iid[iid]
        source_camera = bool(
            cache_arrays["source_camera_valid"][cache_index]
        )
        target_camera = bool(
            cache_arrays["target_camera_valid"][cache_index]
        )
        target_dino = bool(
            visual_arrays["target_valid"][visual_index]
        )
        if source_camera:
            counts_by_stage["source_camera_valid"] += 1
        else:
            reasons["source_camera_invalid"] += 1
        if target_camera:
            counts_by_stage["target_camera_valid"] += 1
        else:
            reasons["target_camera_invalid"] += 1
        if target_dino:
            counts_by_stage["target_dino_valid"] += 1
        else:
            reasons["target_dino_invalid"] += 1
        if not (source_camera and target_camera and target_dino):
            continue
        features, motion_energy = _example_features(
            iid=iid,
            cache_arrays=cache_arrays,
            cache_index=cache_index,
            visual_arrays=visual_arrays,
            visual_index=visual_index,
            seed=seed,
        )
        label = row["label"]
        assignment = row["assignment"]
        sampling_weight = float(
            row["sampling"]["inverse_probability_weight"]
        )
        if (
            not math.isfinite(sampling_weight)
            or sampling_weight <= 0.0
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} sampling weight is invalid"
            )
        examples.append(
            _Example(
                iid=iid,
                label_class=str(label["class"]),
                family=str(label["primary_family"]),
                split=str(assignment["split"]),
                component_id=str(assignment["component_id"]),
                fresh=bool(assignment["fresh"]),
                sampling_weight=sampling_weight,
                features=features,
                motion_energy=motion_energy,
            )
        )
    counts_by_stage["common_cohort"] = len(examples)
    if not examples:
        raise CandidateTemporalScreenError(
            "no row has paired camera compensation and target DINO"
        )
    return examples, {
        "input_rows": len(candidate_rows),
        "stage_counts": dict(sorted(counts_by_stage.items())),
        "exclusion_reason_counts_nonexclusive": dict(sorted(reasons.items())),
        "common_cohort_by_label": dict(
            sorted(Counter(item.label_class for item in examples).items())
        ),
        "common_cohort_by_split": {
            split: sum(item.split == split for item in examples)
            for split in SPLITS
        },
        "common_cohort_by_freshness": {
            "fresh": sum(item.fresh for item in examples),
            "nonfresh": sum(not item.fresh for item in examples),
        },
        "modalities_share_exact_common_cohort": True,
    }


def _family_support(
    examples: Sequence[_Example],
) -> tuple[dict[str, Any], set[str]]:
    train = [
        item
        for item in examples
        if item.label_class == "positive" and item.split == "train"
    ]
    evaluation = [
        item
        for item in examples
        if (
            item.label_class == "positive"
            and item.split in EVAL_SPLITS
        )
    ]
    train_counts = Counter(item.family for item in train)
    eval_counts = Counter(item.family for item in evaluation)
    train_components: dict[str, set[str]] = {}
    for item in train:
        train_components.setdefault(item.family, set()).add(
            item.component_id
        )
    families = sorted(set(train_counts) | set(eval_counts))
    records: dict[str, dict[str, Any]] = {}
    eligible: set[str] = set()
    for family in families:
        reference_count = int(train_counts[family])
        component_count = len(train_components.get(family, set()))
        query_count = int(eval_counts[family])
        reasons: list[str] = []
        if reference_count < MINIMUM_TRAIN_REFERENCES:
            reasons.append("train_references_below_5")
        if component_count < MINIMUM_TRAIN_COMPONENTS:
            reasons.append("train_components_below_5")
        accepted = not reasons
        if accepted:
            eligible.add(family)
        records[family] = {
            "train_references": reference_count,
            "train_components": component_count,
            "evaluation_queries": query_count,
            "evaluation_support_sufficient_for_precision_reporting":
                query_count >= MINIMUM_EVAL_QUERIES,
            "eligible": accepted,
            "exclusion_reasons": reasons,
        }
    eligible_train = [
        item for item in train if item.family in eligible
    ]
    eligible_queries = [
        item for item in evaluation if item.family in eligible
    ]
    return {
        "thresholds": {
            "minimum_train_references_per_family":
                MINIMUM_TRAIN_REFERENCES,
            "minimum_train_components_per_family":
                MINIMUM_TRAIN_COMPONENTS,
            "minimum_evaluation_queries_for_precision_reporting_only":
                MINIMUM_EVAL_QUERIES,
            "evaluation_splits": list(EVAL_SPLITS),
            "eligibility_uses_train_only": True,
            "heldout_labels_select_families": False,
        },
        "families": records,
        "eligible_families": sorted(eligible),
        "eligible_family_count": len(eligible),
        "eligible_train_references": len(eligible_train),
        "eligible_evaluation_queries": len(eligible_queries),
        "excluded_family_count": len(families) - len(eligible),
    }, eligible


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    modality: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (
            row["label_class"] == "positive"
            and row["eligible_positive_query"] is True
        )
    ]

    def one_group(
        subset: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        valid = [
            row
            for row in subset
            if row["modalities"][modality]["valid_for_retrieval"] is True
        ]
        if not subset:
            return {
                "queries": 0,
                "valid_queries": 0,
                "invalid_queries": 0,
                "valid_fraction": None,
                "r_at_1": None,
                "r_at_5": None,
                "valid_only_r_at_1": None,
                "valid_only_r_at_5": None,
            }
        return {
            "queries": len(subset),
            "valid_queries": len(valid),
            "invalid_queries": len(subset) - len(valid),
            "valid_fraction": float(len(valid) / len(subset)),
            # Invalid queries count as misses in the conservative headline.
            "r_at_1": float(
                np.mean(
                    [
                        row["modalities"][modality]["correct_at_1"] is True
                        for row in subset
                    ]
                )
            ),
            "r_at_5": float(
                np.mean(
                    [
                        row["modalities"][modality]["correct_at_5"] is True
                        for row in subset
                    ]
                )
            ),
            "valid_only_r_at_1": (
                float(
                    np.mean(
                        [
                            row["modalities"][modality]["correct_at_1"]
                            for row in valid
                        ]
                    )
                )
                if valid
                else None
            ),
            "valid_only_r_at_5": (
                float(
                    np.mean(
                        [
                            row["modalities"][modality]["correct_at_5"]
                            for row in valid
                        ]
                    )
                )
                if valid
                else None
            ),
        }

    def aggregate(
        subset: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        per_family = {
            family: one_group(
                [row for row in subset if row["family"] == family]
            )
            for family in sorted({str(row["family"]) for row in subset})
        }
        micro = one_group(subset)
        valid_family_records = [
            record
            for record in per_family.values()
            if record["valid_queries"] > 0
        ]
        macro = {
            "families": len(per_family),
            "families_with_valid_queries": len(valid_family_records),
            "r_at_1": (
                float(
                    np.mean(
                        [
                            record["r_at_1"]
                            for record in per_family.values()
                        ]
                    )
                )
                if per_family
                else None
            ),
            "r_at_5": (
                float(
                    np.mean(
                        [
                            record["r_at_5"]
                            for record in per_family.values()
                        ]
                    )
                )
                if per_family
                else None
            ),
            "valid_only_r_at_1": (
                float(
                    np.mean(
                        [
                            record["valid_only_r_at_1"]
                            for record in valid_family_records
                        ]
                    )
                )
                if valid_family_records
                else None
            ),
            "valid_only_r_at_5": (
                float(
                    np.mean(
                        [
                            record["valid_only_r_at_5"]
                            for record in valid_family_records
                        ]
                    )
                )
                if valid_family_records
                else None
            ),
        }
        return {
            "micro": micro,
            "macro_family": macro,
            "per_family": per_family,
        }

    return {
        "overall": aggregate(selected),
        "by_split": {
            split: aggregate(
                [row for row in selected if row["split"] == split]
            )
            for split in EVAL_SPLITS
        },
    }


def _weighted_auc(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    positive_weights: Sequence[float],
    negative_weights: Sequence[float],
) -> float | None:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    pos_weight = np.asarray(positive_weights, dtype=np.float64)
    neg_weight = np.asarray(negative_weights, dtype=np.float64)
    if not len(positive) or not len(negative):
        return None
    if (
        positive.shape != pos_weight.shape
        or negative.shape != neg_weight.shape
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
        or not np.isfinite(pos_weight).all()
        or not np.isfinite(neg_weight).all()
        or bool((pos_weight <= 0.0).any())
        or bool((neg_weight <= 0.0).any())
    ):
        raise CandidateTemporalScreenError("invalid weighted-AUROC inputs")
    comparison = (
        (positive[:, None] > negative[None, :]).astype(np.float64)
        + 0.5
        * (positive[:, None] == negative[None, :]).astype(np.float64)
    )
    weights = pos_weight[:, None] * neg_weight[None, :]
    return float(np.sum(comparison * weights) / np.sum(weights))


def _binary_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_name: str,
) -> dict[str, Any]:
    def summarize(
        subset: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        all_positive = [
            row
            for row in subset
            if row["label_class"] == "positive"
        ]
        all_negative = [
            row for row in subset if row["label_class"] == "negative"
        ]
        positive = [
            row
            for row in all_positive
            if row["binary_scores"][score_name] is not None
        ]
        negative = [
            row
            for row in all_negative
            if row["binary_scores"][score_name] is not None
        ]
        positive_scores = [
            float(row["binary_scores"][score_name]) for row in positive
        ]
        negative_scores = [
            float(row["binary_scores"][score_name]) for row in negative
        ]
        positive_weights = [
            float(row["sampling_weight"]) for row in positive
        ]
        negative_weights = [
            float(row["sampling_weight"]) for row in negative
        ]
        all_positive_weights = [
            float(row["sampling_weight"]) for row in all_positive
        ]
        all_negative_weights = [
            float(row["sampling_weight"]) for row in all_negative
        ]
        return {
            "positive_rows": len(all_positive),
            "valid_positive_rows": len(positive),
            "sampled_negative_rows": len(all_negative),
            "valid_sampled_negative_rows": len(negative),
            "positive_weight": float(sum(all_positive_weights)),
            "valid_positive_weight": float(sum(positive_weights)),
            "sampled_negative_weight": float(sum(all_negative_weights)),
            "valid_sampled_negative_weight":
                float(sum(negative_weights)),
            "valid_positive_fraction": (
                float(len(positive) / len(all_positive))
                if all_positive
                else None
            ),
            "valid_sampled_negative_fraction": (
                float(len(negative) / len(all_negative))
                if all_negative
                else None
            ),
            "sampled_auroc": _weighted_auc(
                positive_scores,
                negative_scores,
                np.ones(len(positive), dtype=np.float64),
                np.ones(len(negative), dtype=np.float64),
            ),
            "sampling_weighted_auroc": _weighted_auc(
                positive_scores,
                negative_scores,
                positive_weights,
                negative_weights,
            ),
        }

    heldout = [row for row in rows if row["split"] in EVAL_SPLITS]
    return {
        "overall": summarize(heldout),
        "by_split": {
            split: summarize(
                [row for row in heldout if row["split"] == split]
            )
            for split in EVAL_SPLITS
        },
    }


def _rank_bank(
    query: _Example,
    bank: Sequence[_Example],
    *,
    modality: str,
) -> tuple[
    list[_Example],
    list[float],
    dict[str, int],
    str | None,
]:
    reference_modality = REFERENCE_FEATURE[modality]
    candidates: list[tuple[_Example, float]] = []
    excluded = {
        "same_iid": 0,
        "same_component": 0,
        "zero_reference": 0,
        "duplicate_component": 0,
    }
    query_feature = np.asarray(query.features[modality], dtype=np.float64)
    if float(np.linalg.norm(query_feature)) <= _EPS:
        return [], [], excluded, "zero_query"
    for reference in bank:
        reasons: list[str] = []
        if reference.iid == query.iid:
            reasons.append("same_iid")
        if reference.component_id == query.component_id:
            reasons.append("same_component")
        if reasons:
            for reason in reasons:
                excluded[reason] += 1
            continue
        reference_feature = np.asarray(
            reference.features[reference_modality],
            dtype=np.float64,
        )
        if reference_feature.shape != query_feature.shape:
            raise CandidateTemporalScreenError(
                f"{modality} query/reference dimensions differ"
            )
        if float(np.linalg.norm(reference_feature)) <= _EPS:
            excluded["zero_reference"] += 1
            continue
        similarity = float(
            np.clip(
                np.dot(query_feature, reference_feature),
                -1.0,
                1.0,
            )
        )
        candidates.append((reference, similarity))
    candidates.sort(key=lambda item: (-item[1], item[0].iid))
    independent: list[tuple[_Example, float]] = []
    seen_components: set[str] = set()
    for item in candidates:
        component_id = item[0].component_id
        if component_id in seen_components:
            excluded["duplicate_component"] += 1
            continue
        seen_components.add(component_id)
        independent.append(item)
    if len(independent) < 5:
        return [], [], excluded, "fewer_than_five_valid_reference_components"
    top = independent[:5]
    return (
        [item[0] for item in top],
        [item[1] for item in top],
        excluded,
        None,
    )


def _evaluate(
    examples: Sequence[_Example],
    *,
    eligible_families: set[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    bank = sorted(
        [
            item
            for item in examples
            if (
                item.label_class == "positive"
                and item.split == "train"
            )
        ],
        key=lambda item: item.iid,
    )
    heldout = sorted(
        [item for item in examples if item.split in EVAL_SPLITS],
        key=lambda item: item.iid,
    )
    if eligible_families and len(
        {item.component_id for item in bank}
    ) < 5:
        raise CandidateTemporalScreenError(
            "complete train-positive bank has fewer than five components"
        )
    output_rows: list[dict[str, Any]] = []
    exclusions = {
        modality: {
            "same_iid": 0,
            "same_component": 0,
            "zero_reference": 0,
            "duplicate_component": 0,
        }
        for modality in MODALITIES
    }
    invalid_retrieval = {
        modality: Counter() for modality in MODALITIES
    }
    zero_query = {modality: 0 for modality in MODALITIES}
    zero_reference = {
        modality: sum(
            float(
                np.linalg.norm(
                    item.features[REFERENCE_FEATURE[modality]]
                )
            )
            <= _EPS
            for item in bank
        )
        for modality in MODALITIES
    }
    for query in heldout:
        eligible_positive = (
            query.label_class == "positive"
            and query.family in eligible_families
        )
        modality_rows: dict[str, Any] = {}
        binary_scores: dict[str, float | None] = {
            "target_motion_energy": float(query.motion_energy)
        }
        for modality in MODALITIES:
            if float(np.linalg.norm(query.features[modality])) <= _EPS:
                zero_query[modality] += 1
            references, similarities, excluded, invalid_reason = _rank_bank(
                query,
                bank,
                modality=modality,
            )
            for reason, count in excluded.items():
                exclusions[modality][reason] += count
            valid_for_retrieval = invalid_reason is None
            if invalid_reason is not None:
                invalid_retrieval[modality][invalid_reason] += 1
            correct1 = (
                references[0].family == query.family
                if eligible_positive and valid_for_retrieval
                else None
            )
            correct5 = (
                any(
                    reference.family == query.family
                    for reference in references
                )
                if eligible_positive and valid_for_retrieval
                else None
            )
            score = (
                float(similarities[0])
                if valid_for_retrieval
                else None
            )
            binary_scores[modality] = score
            modality_rows[modality] = {
                "reference_feature": REFERENCE_FEATURE[modality],
                "valid_for_retrieval": valid_for_retrieval,
                "invalid_reason": invalid_reason,
                "top_reference_iids": [
                    reference.iid for reference in references
                ],
                "top_reference_components": [
                    reference.component_id for reference in references
                ],
                "top_reference_families": [
                    reference.family for reference in references
                ],
                "cosine_similarities": similarities,
                "max_train_positive_cosine": score,
                "correct_at_1": correct1,
                "correct_at_5": correct5,
            }
        output_rows.append(
            {
                "schema_version": ROW_SCHEMA,
                "iid": query.iid,
                "label_class": query.label_class,
                "family": query.family,
                "split": query.split,
                "component_id": query.component_id,
                "fresh": query.fresh,
                "sampling_weight": query.sampling_weight,
                "eligible_positive_query": eligible_positive,
                "modalities": modality_rows,
                "binary_scores": binary_scores,
                **_safety_flags(),
            }
        )
    retrieval = {
        modality: _metric_summary(output_rows, modality=modality)
        for modality in MODALITIES
    }
    score_names = ("target_motion_energy", *MODALITIES)
    binary = {
        score: _binary_summary(output_rows, score_name=score)
        for score in score_names
    }
    leakage = {
        "protocol": RETRIEVAL_PROTOCOL,
        "reference_examples": "positive_train_rows_only",
        "train_distractor_families_pruned": False,
        "reference_component_policy":
            "maximum_one_reference_per_component",
        "zero_vectors_ranked": False,
        "source_video_as_independent_reference": False,
        "query_splits": list(EVAL_SPLITS),
        "reference_feature_by_modality": dict(REFERENCE_FEATURE),
        "query_only_controls": [SHUFFLED_QUERY, REVERSED_QUERY],
        "exclusions": exclusions,
    }
    diagnostic_coverage = {
        "train_reference_rows": len(bank),
        "train_reference_components": len(
            {item.component_id for item in bank}
        ),
        "heldout_rows": len(heldout),
        "heldout_positive_rows": sum(
            item.label_class == "positive" for item in heldout
        ),
        "heldout_sampled_negative_rows": sum(
            item.label_class == "negative" for item in heldout
        ),
        "eligible_positive_query_rows": sum(
            row["eligible_positive_query"] for row in output_rows
        ),
        "zero_query_vector_counts": zero_query,
        "zero_reference_vector_counts": zero_reference,
        "invalid_retrieval_reason_counts": {
            modality: dict(sorted(counts.items()))
            for modality, counts in invalid_retrieval.items()
        },
    }
    return output_rows, retrieval, binary, {
        "leakage_control": leakage,
        "coverage": diagnostic_coverage,
    }


def _contract(
    *,
    inputs: _Inputs,
    examples: Sequence[_Example],
    seed: int,
) -> dict[str, Any]:
    split_topology = inputs.binding["realized_split_topology"]
    retained_dino = int(
        split_topology["retained_dino_spanning_edges"]
    )
    first = examples[0]
    dimensions = {
        modality: int(first.features[modality].shape[0])
        for modality in MODALITIES
    }
    for example in examples:
        for modality in MODALITIES:
            value = np.asarray(example.features[modality])
            if value.shape != (dimensions[modality],):
                raise CandidateTemporalScreenError(
                    f"{modality} dimensions vary across rows"
                )
    return {
        "schema_version": SCREEN_SCHEMA,
        "input_bindings": dict(inputs.binding),
        "implementation": _implementation_provenance(),
        "seed": seed,
        "representation": {
            "schema_version": REPRESENTATION_SCHEMA,
            "frames": int(
                inputs.cache["arrays"][
                    "target_stabilized_tracks"
                ].shape[1]
            ),
            "visibility_threshold": VISIBILITY_THRESHOLD,
            "active_track_policy": {
                "whole_clip_score": "visible_transition_speed_q90",
                "fraction": ACTIVE_TRACK_FRACTION,
                "minimum_tracks": MINIMUM_ACTIVE_TRACKS,
                "selection_order":
                    "descending_score_then_ascending_track_index",
            },
            "ordered_statistics": {
                "per_transition": [
                    "velocity_x_q10",
                    "velocity_x_q25",
                    "velocity_x_q50",
                    "velocity_x_q75",
                    "velocity_x_q90",
                    "velocity_y_q10",
                    "velocity_y_q25",
                    "velocity_y_q50",
                    "velocity_y_q75",
                    "velocity_y_q90",
                    "speed_q25",
                    "speed_q50",
                    "speed_q75",
                    "speed_q90",
                    "speed_q95",
                ],
                "time_scale": "per_normalized_clip_transition",
            },
            "orderless_pooling": [
                "mean",
                "standard_deviation",
                "q25",
                "q50",
                "q75",
                "maximum",
            ],
            "endpoint": "target_last_minus_first_active_track_statistics",
            "camera_nuisance":
                "target_center_translation_angle_log_scale_trajectory",
            "dino_pooling":
                "arithmetic_mean_of_six_target_l2_normalized_cls_vectors",
            "feature_normalization": "per-row-l2-or-exact-zero",
            "feature_dimensions": dimensions,
            "learned_parameters": False,
            "gradient_steps": 0,
        },
        "retrieval": {
            "protocol": RETRIEVAL_PROTOCOL,
            "modalities": list(MODALITIES),
            "reference_feature_by_modality": dict(REFERENCE_FEATURE),
            "family_support_thresholds": {
                "train_references": MINIMUM_TRAIN_REFERENCES,
                "train_components": MINIMUM_TRAIN_COMPONENTS,
                "minimum_eligible_families":
                    MINIMUM_ELIGIBLE_FAMILIES,
                "evaluation_queries_for_precision_reporting_only":
                    MINIMUM_EVAL_QUERIES,
                "eligibility_uses_train_only": True,
            },
            "reference_bank_keeps_all_train_positive_families": True,
            "one_reference_per_component": True,
            "zero_vectors_ranked": False,
            "headline_invalid_query_policy": "count_as_miss",
            "valid_only_metrics_also_reported": True,
            "split_bias": {
                "retained_dino_spanning_edges": retained_dino,
                "dino_changed_realized_component_topology":
                    retained_dino > 0,
                "relative_motion_vs_dino_diagnostic_is_split_confounded":
                    retained_dino > 0,
                "formal_superiority_claim_allowed": False,
                "evidence_binding":
                    dict(split_topology),
            },
            "evaluation_splits": list(EVAL_SPLITS),
        },
        "binary": {
            "protocol": BINARY_PROTOCOL,
            "scores": ["target_motion_energy", *MODALITIES],
            "negative_sampling_weight":
                "manifest_inverse_probability_weight",
        },
        "controls": {
            "shuffle_protocol": SHUFFLE_PROTOCOL,
            "shuffle_seed": seed,
            "time_reversal":
                "physical_frame_axis_reversal_query_only",
            "clean_train_reference_bank_for_controls": True,
        },
        "semantics": {
            "labels_are_pseudo": True,
            "split_is_provisional_diagnostic_only": True,
            "no_gradient": True,
            "no_optimization": True,
            **_safety_flags(),
        },
    }


def _done_payload(
    *,
    rows: int,
    contract_sha256: str,
    payload_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "diagnostic_completed": True,
        "formal_status": "INSUFFICIENT",
        "rows": rows,
        "contract_sha256": contract_sha256,
        "payload_files": {
            name: dict(payload_files[name]) for name in PAYLOAD_NAMES
        },
        "artifact_closure": list(OUTPUT_NAMES),
        "permission_contract": artifact_permissions.permission_contract(),
        **_safety_flags(),
    }
    return {
        **core,
        "artifact_digest": _object_digest(core["payload_files"]),
    }


def _derive(
    *,
    candidate_manifest_dir: Path,
    expected_candidate_manifest_done_sha256: str,
    track_cache_final: Path,
    expected_track_cache_done_sha256: str,
    visual_features_final: Path,
    expected_visual_features_done_sha256: str,
    visual_candidates_manifest: Path,
    expected_visual_candidates_sha256: str,
    seed: int,
    verify_source_shards: bool,
    rehash_videos: bool,
) -> tuple[dict[str, bytes], _Inputs]:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise CandidateTemporalScreenError(
            "seed must be an integer in [0,2**32)"
        )
    inputs = _validate_inputs(
        candidate_manifest_dir=candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            expected_candidate_manifest_done_sha256
        ),
        track_cache_final=track_cache_final,
        expected_track_cache_done_sha256=(
            expected_track_cache_done_sha256
        ),
        visual_features_final=visual_features_final,
        expected_visual_features_done_sha256=(
            expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            expected_visual_candidates_sha256
        ),
        verify_source_shards=verify_source_shards,
        rehash_videos=rehash_videos,
    )
    examples, input_coverage = _build_examples(inputs, seed=seed)
    support, eligible = _family_support(examples)
    if len(eligible) < MINIMUM_ELIGIBLE_FAMILIES:
        raise CandidateTemporalScreenError(
            "fewer than two positive families meet the train-only "
            "retrieval support thresholds"
        )
    rows, retrieval, binary, diagnostics = _evaluate(
        examples,
        eligible_families=eligible,
    )
    if not rows:
        raise CandidateTemporalScreenError(
            "the common cohort contains no held-out rows"
        )
    contract = _contract(inputs=inputs, examples=examples, seed=seed)
    contract_sha = _object_digest(contract)
    summary = {
        "schema_version": SCREEN_SCHEMA,
        "status": "complete",
        "diagnostic_scope":
            "no-gradient-pseudo-label-candidate-temporal-screen",
        "contract": contract,
        "contract_sha256": contract_sha,
        "coverage": {
            "input": input_coverage,
            "evaluation": diagnostics["coverage"],
        },
        "realized_split_topology":
            dict(inputs.binding["realized_split_topology"]),
        "support": support,
        "retrieval": retrieval,
        "positive_vs_sampled_negative": {
            "protocol": BINARY_PROTOCOL,
            "metrics": binary,
        },
        "leakage_control": diagnostics["leakage_control"],
        "controls": {
            "transition_block_shuffle": {
                "modality": SHUFFLED_QUERY,
                "protocol": SHUFFLE_PROTOCOL,
                "query_only": True,
                "clean_reference_feature": TARGET_TEMPORAL,
                "transition_multiset_preserved": True,
            },
            "physical_time_reversal": {
                "modality": REVERSED_QUERY,
                "protocol": "exact-frame-axis-reversal-query-only-v1",
                "query_only": True,
                "clean_reference_feature": TARGET_TEMPORAL,
            },
        },
        "decision": {
            "formal_status": "INSUFFICIENT",
            "reason": (
                "pseudo labels and a provisional indexed split cannot "
                "authorize training, generation, editing, or production"
            ),
            "diagnostic_completed": True,
            **_safety_flags(),
        },
        "formal_status": "INSUFFICIENT",
        **_safety_flags(),
        "output": {
            "rows_name": ROWS_NAME,
            "rows": len(rows),
            "row_order": "ascending_iid",
            "row_encoding": "canonical_json_utf8_lf",
        },
    }
    row_bytes = _jsonl_bytes(rows)
    summary["output"]["rows_sha256"] = hashlib.sha256(
        row_bytes
    ).hexdigest()
    summary_bytes = _pretty_json_bytes(summary)
    payload_files = {
        ROWS_NAME: {
            "sha256": hashlib.sha256(row_bytes).hexdigest(),
            "bytes": len(row_bytes),
            "mode_octal": "0444",
        },
        SUMMARY_NAME: {
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "bytes": len(summary_bytes),
            "mode_octal": "0444",
        },
    }
    done = _done_payload(
        rows=len(rows),
        contract_sha256=contract_sha,
        payload_files=payload_files,
    )
    return {
        ROWS_NAME: row_bytes,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: _pretty_json_bytes(done),
    }, inputs


def _validate_metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    row_fields = {
        "schema_version",
        "iid",
        "label_class",
        "family",
        "split",
        "component_id",
        "fresh",
        "sampling_weight",
        "eligible_positive_query",
        "modalities",
        "binary_scores",
        *SAFETY_FIELDS,
    }
    modality_fields = {
        "reference_feature",
        "valid_for_retrieval",
        "invalid_reason",
        "top_reference_iids",
        "top_reference_components",
        "top_reference_families",
        "cosine_similarities",
        "max_train_positive_cosine",
        "correct_at_1",
        "correct_at_5",
    }
    seen: set[str] = set()
    if [row.get("iid") for row in rows] != sorted(
        row.get("iid") for row in rows
    ):
        raise CandidateTemporalScreenError(
            "screen rows are not in ascending IID order"
        )
    for row in rows:
        if set(row) != row_fields or row.get("schema_version") != ROW_SCHEMA:
            raise CandidateTemporalScreenError(
                "screen row field/schema set differs"
            )
        iid = row.get("iid")
        if type(iid) is not str or not iid or iid in seen:
            raise CandidateTemporalScreenError(
                "screen row IID is invalid or duplicated"
            )
        seen.add(iid)
        if (
            row.get("label_class") not in {"positive", "negative"}
            or row.get("split") not in EVAL_SPLITS
            or row.get("fresh") is not True
            or type(row.get("eligible_positive_query")) is not bool
            or type(row.get("family")) is not str
            or not row["family"]
            or type(row.get("component_id")) is not str
            or not row["component_id"]
            or any(row.get(field) is not False for field in SAFETY_FIELDS)
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} label/split/safety fields differ"
            )
        weight = row.get("sampling_weight")
        if (
            type(weight) not in {int, float}
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} sampling weight is invalid"
            )
        modalities = row.get("modalities")
        binary_scores = row.get("binary_scores")
        if (
            not isinstance(modalities, Mapping)
            or set(modalities) != set(MODALITIES)
            or not isinstance(binary_scores, Mapping)
            or set(binary_scores)
            != {"target_motion_energy", *MODALITIES}
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} modality/score registry differs"
            )
        motion_energy = binary_scores["target_motion_energy"]
        if (
            type(motion_energy) not in {int, float}
            or not math.isfinite(float(motion_energy))
            or float(motion_energy) < 0.0
        ):
            raise CandidateTemporalScreenError(
                f"iid={iid} target motion energy is invalid"
            )
        for modality in MODALITIES:
            result = modalities[modality]
            if not isinstance(result, Mapping) or set(result) != modality_fields:
                raise CandidateTemporalScreenError(
                    f"iid={iid}/{modality} result fields differ"
                )
            references = result["top_reference_iids"]
            components = result["top_reference_components"]
            families = result["top_reference_families"]
            similarities = result["cosine_similarities"]
            valid_for_retrieval = result.get("valid_for_retrieval")
            if (
                result.get("reference_feature")
                != REFERENCE_FEATURE[modality]
                or type(valid_for_retrieval) is not bool
                or any(
                    not isinstance(value, list)
                    for value in (
                        references,
                        components,
                        families,
                        similarities,
                    )
                )
            ):
                raise CandidateTemporalScreenError(
                    f"iid={iid}/{modality} retrieval binding differs"
                )
            if valid_for_retrieval:
                score = result.get("max_train_positive_cosine")
                if (
                    result.get("invalid_reason") is not None
                    or not (
                        len(references)
                        == len(components)
                        == len(families)
                        == len(similarities)
                        == 5
                    )
                    or len(set(references)) != 5
                    or len(set(components)) != 5
                    or iid in references
                    or row["component_id"] in components
                    or any(
                        type(value) not in {int, float}
                        or not math.isfinite(float(value))
                        or not -1.0 <= float(value) <= 1.0
                        for value in similarities
                    )
                    or similarities
                    != sorted(similarities, reverse=True)
                    or type(score) not in {int, float}
                    or not math.isfinite(float(score))
                    or score != similarities[0]
                    or binary_scores.get(modality) != score
                ):
                    raise CandidateTemporalScreenError(
                        f"iid={iid}/{modality} valid retrieval differs"
                    )
            elif (
                result.get("invalid_reason")
                not in {
                    "zero_query",
                    "fewer_than_five_valid_reference_components",
                }
                or any(
                    value
                    for value in (
                        references,
                        components,
                        families,
                        similarities,
                    )
                )
                or result.get("max_train_positive_cosine") is not None
                or binary_scores.get(modality) is not None
            ):
                raise CandidateTemporalScreenError(
                    f"iid={iid}/{modality} invalid retrieval differs"
                )
            if row["eligible_positive_query"] and valid_for_retrieval:
                expected1 = families[0] == row["family"]
                expected5 = row["family"] in families
                if (
                    result.get("correct_at_1") is not expected1
                    or result.get("correct_at_5") is not expected5
                ):
                    raise CandidateTemporalScreenError(
                        f"iid={iid}/{modality} correctness differs"
                    )
            elif (
                result.get("correct_at_1") is not None
                or result.get("correct_at_5") is not None
            ):
                raise CandidateTemporalScreenError(
                    f"iid={iid}/{modality} ineligible correctness asserted"
                )


def _validate_candidate_temporal_screen_envelope(
    output_dir: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the sealed envelope and replay metrics from committed rows."""

    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    actual = {entry.name for entry in root.iterdir()}
    if actual != set(OUTPUT_NAMES):
        raise CandidateTemporalScreenError(
            "screen output artifact closure differs"
        )
    artifact_permissions.assert_sealed_tree(root)
    rows_path = root / ROWS_NAME
    summary_path = root / SUMMARY_NAME
    done_path = root / DONE_NAME
    rows = _load_canonical_jsonl(rows_path)
    summary = _load_object(summary_path)
    done = _load_object(done_path)
    if summary_path.read_bytes() != _pretty_json_bytes(summary):
        raise CandidateTemporalScreenError(
            "screen summary is not canonical pretty JSON"
        )
    if done_path.read_bytes() != _pretty_json_bytes(done):
        raise CandidateTemporalScreenError(
            "screen done is not canonical pretty JSON"
        )
    _validate_metric_rows(rows)
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise CandidateTemporalScreenError("screen contract is missing")
    contract_sha = _object_digest(dict(contract))
    if (
        summary.get("schema_version") != SCREEN_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("formal_status") != "INSUFFICIENT"
        or summary.get("contract_sha256") != contract_sha
        or any(summary.get(field) is not False for field in SAFETY_FIELDS)
        or expected_contract is not None
        and dict(expected_contract) != dict(contract)
    ):
        raise CandidateTemporalScreenError(
            "screen summary contract/safety fields differ"
        )
    semantics = contract.get("semantics")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("labels_are_pseudo") is not True
        or semantics.get("split_is_provisional_diagnostic_only") is not True
        or semantics.get("no_gradient") is not True
        or semantics.get("no_optimization") is not True
        or any(semantics.get(field) is not False for field in SAFETY_FIELDS)
    ):
        raise CandidateTemporalScreenError(
            "screen contract semantics differ"
        )
    decision = summary.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("formal_status") != "INSUFFICIENT"
        or decision.get("diagnostic_completed") is not True
        or any(decision.get(field) is not False for field in SAFETY_FIELDS)
    ):
        raise CandidateTemporalScreenError(
            "screen decision exceeds diagnostic scope"
        )
    if (
        not isinstance(summary.get("retrieval"), Mapping)
        or set(summary["retrieval"]) != set(MODALITIES)
    ):
        raise CandidateTemporalScreenError(
            "screen retrieval registry differs"
        )
    for modality in MODALITIES:
        expected = _metric_summary(rows, modality=modality)
        if summary["retrieval"][modality] != expected:
            raise CandidateTemporalScreenError(
                f"{modality} retrieval metrics do not replay"
            )
    binary = summary.get("positive_vs_sampled_negative")
    if (
        not isinstance(binary, Mapping)
        or binary.get("protocol") != BINARY_PROTOCOL
        or not isinstance(binary.get("metrics"), Mapping)
        or set(binary["metrics"]) != {"target_motion_energy", *MODALITIES}
    ):
        raise CandidateTemporalScreenError(
            "screen binary metric registry differs"
        )
    for score in ("target_motion_energy", *MODALITIES):
        if binary["metrics"][score] != _binary_summary(
            rows,
            score_name=score,
        ):
            raise CandidateTemporalScreenError(
                f"{score} binary metrics do not replay"
            )
    rows_sha = _file_digest(rows_path)
    output = summary.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("rows_name") != ROWS_NAME
        or output.get("rows") != len(rows)
        or output.get("rows_sha256") != rows_sha
        or output.get("row_order") != "ascending_iid"
        or output.get("row_encoding") != "canonical_json_utf8_lf"
    ):
        raise CandidateTemporalScreenError(
            "screen output row commitment differs"
        )
    payload_files = {
        name: {
            "sha256": _file_digest(root / name),
            "bytes": int((root / name).stat().st_size),
            "mode_octal": "0444",
        }
        for name in PAYLOAD_NAMES
    }
    expected_done = _done_payload(
        rows=len(rows),
        contract_sha256=contract_sha,
        payload_files=payload_files,
    )
    if done != expected_done:
        raise CandidateTemporalScreenError(
            "screen done/hash/safety chain differs"
        )
    return {
        "directory": root,
        "rows": rows,
        "summary": summary,
        "done": done,
    }


def validate_candidate_temporal_screen(
    output_dir: Path,
    *,
    expected_done_sha256: str,
    candidate_manifest_dir: Path,
    expected_candidate_manifest_done_sha256: str,
    track_cache_final: Path,
    expected_track_cache_done_sha256: str,
    visual_features_final: Path,
    expected_visual_features_done_sha256: str,
    visual_candidates_manifest: Path,
    expected_visual_candidates_sha256: str,
    seed: int = DEFAULT_SEED,
    verify_source_shards: bool = True,
    rehash_videos: bool = True,
) -> dict[str, Any]:
    """Reopen every upstream commit and reproduce the result byte-for-byte."""

    root = output_dir.expanduser().resolve(strict=True)
    expected_done = _require_sha256(
        expected_done_sha256,
        context="expected temporal-screen done SHA",
    )
    if _file_digest(root / DONE_NAME) != expected_done:
        raise CandidateTemporalScreenError(
            "temporal-screen external done SHA differs"
        )
    result = _validate_candidate_temporal_screen_envelope(root)
    payloads, inputs = _derive(
        candidate_manifest_dir=candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            expected_candidate_manifest_done_sha256
        ),
        track_cache_final=track_cache_final,
        expected_track_cache_done_sha256=(
            expected_track_cache_done_sha256
        ),
        visual_features_final=visual_features_final,
        expected_visual_features_done_sha256=(
            expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            expected_visual_candidates_sha256
        ),
        seed=seed,
        verify_source_shards=verify_source_shards,
        rehash_videos=rehash_videos,
    )
    for name in OUTPUT_NAMES:
        if (root / name).read_bytes() != payloads[name]:
            raise CandidateTemporalScreenError(
                f"screen output does not replay from bound inputs: {name}"
            )
    _assert_inputs_stable(inputs)
    result["upstream_replay_verified"] = True
    result["expected_done_sha256"] = expected_done
    return result


def _write_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_inputs_stable(inputs: _Inputs) -> None:
    _assert_identities(inputs.identities)
    _assert_identities(inputs.media_identities)


def _publish(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
    inputs: _Inputs,
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.publish.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        os.close(lock_fd)
        raise
    stage: Path | None = None
    try:
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.",
                suffix=".tmp",
                dir=output_dir.parent,
            )
        )
        for name in OUTPUT_NAMES:
            _write_file(stage / name, payloads[name])
        descriptor = os.open(stage, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _assert_inputs_stable(inputs)
        artifact_permissions.seal_staging_tree(
            stage,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            stage,
            allow_writable_root=True,
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"screen output appeared during publication: {output_dir}"
            )
        os.rename(stage, output_dir)
        artifact_permissions.seal_published_root(output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        _assert_inputs_stable(inputs)
    finally:
        if stage is not None and stage.exists():
            artifact_permissions.remove_staging_tree(stage)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _strict_resume(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
) -> None:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise FileNotFoundError(
            "resume requires an existing regular output directory"
        )
    mode = stat.S_IMODE(output_dir.lstat().st_mode)
    if mode == 0o700:
        artifact_permissions.assert_sealed_tree(
            output_dir,
            allow_writable_root=True,
        )
        artifact_permissions.seal_published_root(output_dir)
    elif mode != artifact_permissions.DIRECTORY_MODE:
        raise CandidateTemporalScreenError(
            f"resume output root mode is not repairable: {mode:04o}"
        )
    actual = {entry.name for entry in output_dir.iterdir()}
    if actual != set(OUTPUT_NAMES):
        raise CandidateTemporalScreenError(
            "resume output artifact closure differs"
        )
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.is_symlink() or not path.is_file():
            raise CandidateTemporalScreenError(
                f"resume output is not a regular file: {path}"
            )
        if path.read_bytes() != payloads[name]:
            raise CandidateTemporalScreenError(
                f"resume differs from fresh derivation: {name}"
            )


def _overlaps(first: Path, second: Path) -> bool:
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def run_candidate_temporal_screen(
    *,
    candidate_manifest_dir: Path,
    expected_candidate_manifest_done_sha256: str,
    track_cache_final: Path,
    expected_track_cache_done_sha256: str,
    visual_features_final: Path,
    expected_visual_features_done_sha256: str,
    visual_candidates_manifest: Path,
    expected_visual_candidates_sha256: str,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    verify_source_shards: bool = True,
    rehash_videos: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Run, atomically publish, or strictly resume the diagnostic screen."""

    output = output_dir.expanduser().resolve(strict=False)
    if output_dir.expanduser().is_symlink():
        raise CandidateTemporalScreenError(
            "screen output must not be a symlink"
        )
    if resume:
        if not output.exists():
            raise FileNotFoundError(
                "resume is verification-only and requires a commit"
            )
    elif output.exists():
        raise FileExistsError(
            f"{output} exists; use a fresh path or resume=True"
        )
    protected = (
        candidate_manifest_dir.expanduser().resolve(strict=True),
        track_cache_final.expanduser().resolve(strict=True),
        visual_features_final.expanduser().resolve(strict=True),
        visual_candidates_manifest.expanduser().resolve(strict=True).parent,
    )
    if any(_overlaps(output, source) for source in protected):
        raise CandidateTemporalScreenError(
            "screen output overlaps an input artifact"
        )
    payloads, inputs = _derive(
        candidate_manifest_dir=candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            expected_candidate_manifest_done_sha256
        ),
        track_cache_final=track_cache_final,
        expected_track_cache_done_sha256=(
            expected_track_cache_done_sha256
        ),
        visual_features_final=visual_features_final,
        expected_visual_features_done_sha256=(
            expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            expected_visual_candidates_sha256
        ),
        seed=seed,
        verify_source_shards=verify_source_shards,
        rehash_videos=rehash_videos,
    )
    if resume:
        _strict_resume(output, payloads=payloads)
        _assert_inputs_stable(inputs)
    else:
        _publish(output, payloads=payloads, inputs=inputs)
    expected_done_sha256 = _file_digest(output / DONE_NAME)
    result = validate_candidate_temporal_screen(
        output,
        expected_done_sha256=expected_done_sha256,
        candidate_manifest_dir=candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            expected_candidate_manifest_done_sha256
        ),
        track_cache_final=track_cache_final,
        expected_track_cache_done_sha256=(
            expected_track_cache_done_sha256
        ),
        visual_features_final=visual_features_final,
        expected_visual_features_done_sha256=(
            expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            expected_visual_candidates_sha256
        ),
        seed=seed,
        verify_source_shards=verify_source_shards,
        rehash_videos=rehash_videos,
    )
    _assert_inputs_stable(inputs)
    result["resume_verified"] = bool(resume)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable no-gradient R7 candidate temporal screen"
        )
    )
    parser.add_argument("--candidate-manifest-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-candidate-manifest-done-sha256",
        required=True,
    )
    parser.add_argument("--track-cache-final", required=True, type=Path)
    parser.add_argument(
        "--expected-track-cache-done-sha256",
        required=True,
    )
    parser.add_argument("--visual-features-final", required=True, type=Path)
    parser.add_argument(
        "--expected-visual-features-done-sha256",
        required=True,
    )
    parser.add_argument(
        "--visual-candidates-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-visual-candidates-sha256",
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verification-only: compare with a complete fresh derivation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_candidate_temporal_screen(
        candidate_manifest_dir=args.candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            args.expected_candidate_manifest_done_sha256
        ),
        track_cache_final=args.track_cache_final,
        expected_track_cache_done_sha256=(
            args.expected_track_cache_done_sha256
        ),
        visual_features_final=args.visual_features_final,
        expected_visual_features_done_sha256=(
            args.expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=args.visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            args.expected_visual_candidates_sha256
        ),
        output_dir=args.output_dir,
        seed=args.seed,
        verify_source_shards=True,
        rehash_videos=True,
        resume=bool(args.resume),
    )
    summary = result["summary"]
    support = summary["support"]
    print(
        "[motive-r7-candidate-temporal-screen] "
        f"rows={len(result['rows'])} "
        f"eligible_families={support['eligible_family_count']} "
        f"formal_status={summary['formal_status']} "
        f"resume_verified={result['resume_verified']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

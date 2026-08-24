"""Build the immutable R7 candidate-only temporal-screen commit.

The temporal screen is an *inference/audit input*, not a training manifest.
It contains the complete strict pseudo-positive census and a deterministic
SHA-256 bottom-k sample of trusted pseudo-negatives.  Historical anchors are
used only by the upstream indexed visual graph and are never emitted.

Every emitted row binds:

* the freshly revalidated R7 expansion row and compact candidate row;
* the indexed component assignment and all indexed upstream digests;
* source/target media bytes below a non-symlink data root; and
* pseudo-label provenance without asserting a human or R5 pilot label.

Publication is create-only and atomic.  ``resume`` is verification-only: all
live inputs and selected media are re-read, a fresh commit is derived, and
every output byte must match.  The standalone validator checks canonical
encoding, row semantics, sampling keys/ranks, and the summary/done hash chain.
It can additionally require byte-exact externally retained summary/done
anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from . import r7_artifact_permissions as artifact_permissions
from . import r7_visual_candidate_manifest as candidate_module
from . import r7_indexed_visual_graph as indexed_core
from . import r7_indexed_visual_graph_io as indexed_io


ROW_SCHEMA = "motive-r7-candidate-temporal-screen-row-v1"
LABEL_SCHEMA = "motive-r7-candidate-temporal-pseudo-label-v1"
ASSIGNMENT_SCHEMA = "motive-r7-candidate-temporal-assignment-v1"
SAMPLING_SCHEMA = "motive-r7-candidate-temporal-sampling-v1"
SOURCE_BINDINGS_SCHEMA = "motive-r7-candidate-temporal-bindings-v1"
SUMMARY_SCHEMA = "motive-r7-candidate-temporal-screen-v1"
DONE_SCHEMA = "motive-r7-candidate-temporal-screen-done-v1"
POLICY_VERSION = "r7-positive-census-trusted-negative-bottom-k-v1"
SELECTION_KEY_SCHEMA = "motive-r7-candidate-temporal-negative-v1"

MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (MANIFEST_NAME, SUMMARY_NAME, DONE_NAME)

POSITIVE_CENSUS = 947
TRUSTED_NEGATIVE_POPULATION = 2220
TRUSTED_NEGATIVE_SAMPLE = 240
CANDIDATE_POPULATION = POSITIVE_CENSUS + TRUSTED_NEGATIVE_POPULATION
EXPECTED_ANCHOR_ROWS = 181
EXPECTED_INDEXED_ROWS = CANDIDATE_POPULATION + EXPECTED_ANCHOR_ROWS
SELECTION_SEED = 260108834

ROW_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "input_digest",
        "prompt",
        "src_video",
        "tgt_video",
        "label",
        "assignment",
        "sampling",
        "source_bindings",
        "formal_evidence",
        "formal_split",
        "human_labels_asserted",
        "training_authorized",
        "generation_authorized",
    }
)
LABEL_FIELDS = frozenset(
    {
        "schema_version",
        "class",
        "negative_type",
        "action_signature",
        "primary_family",
        "provenance_kind",
        "human_label",
    }
)
ASSIGNMENT_FIELDS = frozenset(
    {
        "schema_version",
        "component_id",
        "split",
        "fresh",
        "anchor",
        "forced_train",
        "forced_by_anchor",
        "forced_by_previously_seen",
    }
)
SAMPLING_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "seed",
        "population_rows",
        "selected_rows",
        "selection_key_sha256",
        "selection_rank",
        "inclusion_probability",
        "inverse_probability_weight",
    }
)
SOURCE_BINDINGS_FIELDS = frozenset(
    {
        "schema_version",
        "expansion",
        "candidate",
        "indexed_graph",
        "media",
    }
)
EXPANSION_BINDING_FIELDS = frozenset(
    {"artifact_digest", "source_row_sha256", "source_line_number"}
)
CANDIDATE_BINDING_FIELDS = frozenset(
    {"artifact_digest", "candidate_row_sha256", "row_index"}
)
INDEXED_BINDING_FIELDS = frozenset(
    {
        "artifact_digest",
        "assignment_row_sha256",
        "assignment_row_index",
        "upstream_artifact_digests",
    }
)
MEDIA_BINDING_FIELDS = frozenset({"data_root", "src_video", "tgt_video"})
MEDIA_FILE_FIELDS = frozenset({"relative_path", "sha256", "bytes"})

_CANDIDATE_NAMES = (
    candidate_module.CANDIDATES_NAME,
    candidate_module.SUMMARY_NAME,
    candidate_module.DONE_NAME,
)
_INDEXED_NAMES = (
    indexed_io.ASSIGNMENTS_NAME,
    indexed_io.COMPONENTS_NAME,
    indexed_io.SPANNING_EDGES_NAME,
    indexed_io.SUMMARY_NAME,
    indexed_io.DONE_NAME,
)
_SOURCE_NAMES = tuple(candidate_module.SOURCE_ARTIFACT_NAMES)
_SHA_CHARS = frozenset("0123456789abcdef")
_SPLITS = ("train", "validation", "test")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _object_digest(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


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
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON: {error}") from error


def _load_pretty_object(
    path: Path,
    *,
    context: str,
) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = _parse_json(raw, context=context)
    if type(value) is not dict:
        raise ValueError(f"{context} must contain one JSON object")
    if raw != _pretty_json_bytes(value):
        raise ValueError(f"{context} is not canonical pretty JSON")
    return value, raw


def _load_canonical_jsonl(
    path: Path,
    *,
    context: str,
) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{context} must end with LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"{context}:{line_number} is blank")
        row = _parse_json(line, context=f"{context}:{line_number}")
        if type(row) is not dict:
            raise ValueError(f"{context}:{line_number} is not an object")
        if line != _canonical_bytes(row):
            raise ValueError(
                f"{context}:{line_number} is not canonical JSON"
            )
        rows.append(row)
    return rows, raw


def _sha_field(value: Any, *, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _plain_string(value: Any, *, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{context} must be a canonical non-empty string")
    return value


def _exact_object(
    value: Any,
    fields: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields):
        actual = set(value) if type(value) is dict else set()
        raise ValueError(
            f"{context} fields differ: "
            f"missing={sorted(set(fields) - actual)} "
            f"extra={sorted(actual - set(fields))}"
        )
    return value


def _strict_directory(
    raw_path: Path,
    *,
    names: Sequence[str],
    context: str,
) -> tuple[Path, dict[str, Path]]:
    unresolved = raw_path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"{context} must be a real directory: {unresolved}"
        )
    root = unresolved.resolve(strict=True)
    expected = set(names)
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise ValueError(
            f"{context} artifact set differs: "
            f"missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    paths: dict[str, Path] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"{context} artifact is not a regular file: {path}"
            )
        paths[name] = path
    return root, paths


def _snapshot(
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"input artifact changed type: {path}")
        info = path.stat()
        snapshot[name] = {
            "sha256": _sha256_file(path),
            "bytes": int(info.st_size),
        }
    return snapshot


def _assert_snapshot(
    paths: Mapping[str, Path],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if _snapshot(paths) != {
        name: dict(value) for name, value in sorted(expected.items())
    }:
        raise RuntimeError("an upstream commit changed during derivation")


def _implementation_provenance() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    files = {
        "r7_candidate_temporal_manifest.py": {
            "sha256": _sha256_file(path)
        }
    }
    return {
        "files": files,
        "bundle_sha256": _object_digest(
            {name: value["sha256"] for name, value in files.items()}
        ),
    }


def _validate_candidate_commit(
    *,
    expansion_manifest_dir: Path,
    candidate_manifest_dir: Path,
    expected_expansion_artifact_digest: str,
    expected_candidate_artifact_digest: str,
) -> dict[str, Any]:
    expected_expansion_artifact_digest = _sha_field(
        expected_expansion_artifact_digest,
        context="expected expansion artifact digest",
    )
    expected_candidate_artifact_digest = _sha_field(
        expected_candidate_artifact_digest,
        context="expected candidate artifact digest",
    )
    (
        expansion_root,
        source_rows_by_name,
        _source_summary,
        source_done,
        expansion_digest,
        _source_file_sha256,
    ) = candidate_module._validate_source_commit(expansion_manifest_dir)
    if expansion_digest != expected_expansion_artifact_digest:
        raise ValueError("expansion artifact digest differs from expectation")

    candidate_root, paths = _strict_directory(
        candidate_manifest_dir,
        names=_CANDIDATE_NAMES,
        context="R7 visual candidate commit",
    )
    rows, rows_raw = _load_canonical_jsonl(
        paths[candidate_module.CANDIDATES_NAME],
        context="candidate candidates.jsonl",
    )
    summary, summary_raw = _load_pretty_object(
        paths[candidate_module.SUMMARY_NAME],
        context="candidate summary.json",
    )
    done, _done_raw = _load_pretty_object(
        paths[candidate_module.DONE_NAME],
        context="candidate done.json",
    )
    if (
        summary.get("schema_version") != candidate_module.SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("policy_version") != candidate_module.POLICY_VERSION
    ):
        raise ValueError("candidate summary schema/status/policy mismatch")
    if (
        done.get("schema_version") != candidate_module.DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("candidate done schema/status mismatch")

    expected_candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for source_row in source_rows_by_name[candidate_module.POSITIVES_NAME]:
        line_number, projected = candidate_module._candidate_from_source(
            source_row,
            cohort="pseudo_positive",
            source_artifact_digest=expansion_digest,
        )
        expected_candidates.append((line_number, projected, source_row))
    for source_row in source_rows_by_name[candidate_module.NEGATIVES_NAME]:
        label = source_row["r7_expansion_manifest"]
        if label["negative_role"] == "pseudo_negative":
            line_number, projected = candidate_module._candidate_from_source(
                source_row,
                cohort="pseudo_negative",
                source_artifact_digest=expansion_digest,
            )
            expected_candidates.append(
                (line_number, projected, source_row)
            )
        elif label["negative_role"] != "audit_only":
            raise ValueError("source has an unsupported negative role")
    expected_candidates.sort(key=lambda item: item[0])
    expected_rows = [item[1] for item in expected_candidates]
    if rows != expected_rows:
        raise ValueError(
            "candidate rows differ from the authoritative source projection"
        )

    cohort_counts = Counter(row["cohort"] for row in rows)
    if cohort_counts != Counter(
        {
            "pseudo_positive": POSITIVE_CENSUS,
            "pseudo_negative": TRUSTED_NEGATIVE_POPULATION,
        }
    ):
        raise ValueError(
            "candidate cohort census differs from frozen R7 contract"
        )
    if len(rows) != CANDIDATE_POPULATION:
        raise ValueError("candidate population size differs from contract")
    iids = [row["iid"] for row in rows]
    if len(set(iids)) != len(iids):
        raise ValueError("candidate commit contains duplicate IID")

    row_sha = _sha256_bytes(rows_raw)
    summary_sha = _sha256_bytes(summary_raw)
    expected_output_sha = {
        candidate_module.CANDIDATES_NAME: row_sha,
        candidate_module.SUMMARY_NAME: summary_sha,
    }
    candidate_output = summary.get("output")
    if (
        type(candidate_output) is not dict
        or candidate_output.get("rows") != len(rows)
        or candidate_output.get("sha256") != row_sha
        or summary.get("cohort_counts")
        != dict(sorted(cohort_counts.items()))
        or summary.get("semantics")
        != {
            "split_assigned": False,
            "human_labels_asserted": False,
            "training_eligible": False,
            "candidate_labels_are_human_truth": False,
        }
    ):
        raise ValueError("candidate summary does not bind rows/semantics")
    if (
        done.get("input_artifact_digest") != expansion_digest
        or done.get("output_rows") != len(rows)
        or done.get("output_sha256") != expected_output_sha
        or done.get("artifact_digest")
        != _object_digest(expected_output_sha)
        or done.get("split_assigned") is not False
        or done.get("human_labels_asserted") is not False
        or done.get("training_eligible") is not False
    ):
        raise ValueError("candidate done hash chain/semantics mismatch")
    candidate_digest = _sha_field(
        done.get("artifact_digest"),
        context="candidate artifact digest",
    )
    if candidate_digest != expected_candidate_artifact_digest:
        raise ValueError("candidate artifact digest differs from expectation")
    if any(
        row["source_artifact_digest"] != expansion_digest for row in rows
    ):
        raise ValueError("candidate row source digest is inconsistent")
    if source_done.get("artifact_digest") != expansion_digest:
        raise RuntimeError("validated source digest unexpectedly changed")

    source_by_iid = {
        projected["iid"]: {
            "row": source_row,
            "line_number": line_number,
        }
        for line_number, projected, source_row in expected_candidates
    }
    return {
        "expansion_root": expansion_root,
        "candidate_root": candidate_root,
        "rows": rows,
        "source_by_iid": source_by_iid,
        "expansion_artifact_digest": expansion_digest,
        "candidate_artifact_digest": candidate_digest,
        "candidate_summary": summary,
        "candidate_done": done,
    }


def _validate_indexed_commit(
    *,
    indexed_graph_dir: Path,
    candidate_rows: Sequence[Mapping[str, Any]],
    expected_indexed_artifact_digest: str,
) -> dict[str, Any]:
    expected_indexed_artifact_digest = _sha_field(
        expected_indexed_artifact_digest,
        context="expected indexed artifact digest",
    )
    root, paths = _strict_directory(
        indexed_graph_dir,
        names=_INDEXED_NAMES,
        context="R7 indexed visual graph commit",
    )
    assignments, assignments_raw = _load_canonical_jsonl(
        paths[indexed_io.ASSIGNMENTS_NAME],
        context="indexed assignments.jsonl",
    )
    components, components_raw = _load_canonical_jsonl(
        paths[indexed_io.COMPONENTS_NAME],
        context="indexed components.jsonl",
    )
    edges, edges_raw = _load_canonical_jsonl(
        paths[indexed_io.SPANNING_EDGES_NAME],
        context="indexed spanning_edges.jsonl",
    )
    summary, summary_raw = _load_pretty_object(
        paths[indexed_io.SUMMARY_NAME],
        context="indexed summary.json",
    )
    done, _done_raw = _load_pretty_object(
        paths[indexed_io.DONE_NAME],
        context="indexed done.json",
    )
    if (
        summary.get("schema_version") != indexed_io.SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("assignment_semantics")
        != "diagnostic-provisional-component-split-v1"
    ):
        raise ValueError("indexed summary schema/status/semantics mismatch")
    if (
        done.get("schema_version") != indexed_io.DONE_SCHEMA
        or done.get("status") != "complete"
    ):
        raise ValueError("indexed done schema/status mismatch")
    for name, value in {
        "summary thresholds_human_calibrated": summary.get(
            "thresholds_human_calibrated"
        ),
        "summary formal_split": summary.get("formal_split"),
        "summary training_authorized": summary.get(
            "training_authorized"
        ),
        "done thresholds_human_calibrated": done.get(
            "thresholds_human_calibrated"
        ),
        "done formal_split": done.get("formal_split"),
        "done training_authorized": done.get("training_authorized"),
    }.items():
        if value is not False:
            raise ValueError(f"indexed {name} must remain false")

    raw_by_name = {
        indexed_io.ASSIGNMENTS_NAME: assignments_raw,
        indexed_io.COMPONENTS_NAME: components_raw,
        indexed_io.SPANNING_EDGES_NAME: edges_raw,
    }
    rows_by_name = {
        indexed_io.ASSIGNMENTS_NAME: assignments,
        indexed_io.COMPONENTS_NAME: components,
        indexed_io.SPANNING_EDGES_NAME: edges,
    }
    orders = {
        indexed_io.ASSIGNMENTS_NAME: "iid",
        indexed_io.COMPONENTS_NAME: "component_id",
        indexed_io.SPANNING_EDGES_NAME:
            "canonical-endpoints-relation-value",
    }
    outputs = summary.get("outputs")
    if type(outputs) is not dict or set(outputs) != set(raw_by_name):
        raise ValueError("indexed summary output set differs")
    artifact_hashes: dict[str, str] = {}
    for name in raw_by_name:
        entry = outputs[name]
        digest = _sha256_bytes(raw_by_name[name])
        if (
            type(entry) is not dict
            or set(entry) != {"rows", "sha256", "order"}
            or entry.get("rows") != len(rows_by_name[name])
            or entry.get("sha256") != digest
            or entry.get("order") != orders[name]
        ):
            raise ValueError(f"indexed summary output mismatch: {name}")
        artifact_hashes[name] = digest
    artifact_hashes[indexed_io.SUMMARY_NAME] = _sha256_bytes(summary_raw)
    expected_artifacts = {
        name: {"filename": name, "sha256": digest}
        for name, digest in sorted(artifact_hashes.items())
    }
    input_bindings = summary.get("input_bindings")
    if type(input_bindings) is not dict or not input_bindings:
        raise ValueError("indexed input_bindings must be non-empty")
    upstream_digests: dict[str, str] = {}
    for name, record in sorted(input_bindings.items()):
        if type(record) is not dict:
            raise ValueError(f"indexed input binding {name} is invalid")
        upstream_digests[name] = _sha_field(
            record.get("artifact_digest"),
            context=f"indexed input binding {name} artifact digest",
        )
    if (
        done.get("artifacts") != expected_artifacts
        or done.get("artifact_digest") != _object_digest(artifact_hashes)
        or done.get("input_artifact_digests") != upstream_digests
    ):
        raise ValueError("indexed done hash/input chain mismatch")
    indexed_digest = _sha_field(
        done.get("artifact_digest"),
        context="indexed artifact digest",
    )
    if indexed_digest != expected_indexed_artifact_digest:
        raise ValueError("indexed artifact digest differs from expectation")

    assignment_fields = {
        "schema_version",
        "iid",
        "component_id",
        "split",
        "fresh",
        "forced_train",
        "forced_by_anchor",
        "forced_by_previously_seen",
        "anchor",
        "cohort",
    }
    previous_iid: str | None = None
    assignment_by_iid: dict[str, dict[str, Any]] = {}
    assignment_index: dict[str, int] = {}
    for index, row in enumerate(assignments):
        if set(row) != assignment_fields:
            raise ValueError("indexed assignment fields differ")
        iid = _plain_string(
            row.get("iid"),
            context="indexed assignment IID",
        )
        if row.get("schema_version") != indexed_io.ASSIGNMENT_ROW_SCHEMA:
            raise ValueError(f"iid={iid} assignment schema mismatch")
        if previous_iid is not None and iid <= previous_iid:
            raise ValueError(
                "indexed assignments must be strictly IID ordered"
            )
        previous_iid = iid
        if iid in assignment_by_iid:
            raise ValueError(f"duplicate indexed assignment IID: {iid}")
        _plain_string(
            row.get("component_id"),
            context=f"iid={iid} component_id",
        )
        if row.get("split") not in _SPLITS:
            raise ValueError(f"iid={iid} has invalid diagnostic split")
        for field in (
            "fresh",
            "forced_train",
            "forced_by_anchor",
            "forced_by_previously_seen",
            "anchor",
        ):
            if type(row.get(field)) is not bool:
                raise ValueError(f"iid={iid} {field} must be boolean")
        _plain_string(
            row.get("cohort"),
            context=f"iid={iid} cohort",
        )
        assignment_by_iid[iid] = row
        assignment_index[iid] = index
    if len(assignments) != EXPECTED_INDEXED_ROWS:
        raise ValueError("indexed assignment census differs from contract")

    candidate_by_iid = {str(row["iid"]): row for row in candidate_rows}
    if len(candidate_by_iid) != len(candidate_rows):
        raise ValueError("candidate rows contain duplicate IID")
    nonanchors = {
        iid for iid, row in assignment_by_iid.items() if not row["anchor"]
    }
    anchors = {
        iid for iid, row in assignment_by_iid.items() if row["anchor"]
    }
    if nonanchors != set(candidate_by_iid):
        raise ValueError(
            "indexed non-anchor assignments differ from candidate IIDs"
        )
    if len(anchors) != EXPECTED_ANCHOR_ROWS:
        raise ValueError("indexed anchor census differs from contract")
    for iid, candidate in candidate_by_iid.items():
        assignment = assignment_by_iid[iid]
        if assignment["anchor"] is not False:
            raise ValueError(f"candidate IID injected as anchor: {iid}")
        if assignment["cohort"] != candidate["cohort"]:
            raise ValueError(f"iid={iid} candidate/indexed cohort mismatch")
    for iid in anchors:
        if assignment_by_iid[iid]["cohort"] not in {
            "anchor_positive",
            "anchor_negative",
        }:
            raise ValueError(f"anchor IID has invalid cohort: {iid}")

    component_fields = {
        "schema_version",
        "component_id",
        "member_assets",
        "member_iids",
        "split",
        "fresh",
        "forced_train",
        "anchor_iids",
        "previously_seen_iids",
    }
    component_by_id: dict[str, dict[str, Any]] = {}
    previous_component: str | None = None
    all_component_iids: set[str] = set()
    for component in components:
        if set(component) != component_fields:
            raise ValueError("indexed component fields differ")
        component_id = _plain_string(
            component.get("component_id"),
            context="indexed component_id",
        )
        if (
            component.get("schema_version")
            != indexed_io.COMPONENT_ROW_SCHEMA
        ):
            raise ValueError("indexed component schema mismatch")
        if (
            previous_component is not None
            and component_id <= previous_component
        ):
            raise ValueError("indexed components are not strictly ordered")
        previous_component = component_id
        if component_id in component_by_id:
            raise ValueError("duplicate indexed component_id")
        member_iids = component.get("member_iids")
        if (
            type(member_iids) is not list
            or not member_iids
            or any(type(iid) is not str for iid in member_iids)
            or member_iids != sorted(set(member_iids))
        ):
            raise ValueError("component member_iids are not canonical")
        if all_component_iids & set(member_iids):
            raise ValueError("IID occurs in multiple indexed components")
        all_component_iids.update(member_iids)
        expected_assets = [
            [iid, role]
            for iid in member_iids
            for role in ("source", "target")
        ]
        if component.get("member_assets") != expected_assets:
            raise ValueError("component member_assets are not exact pairs")
        if (
            indexed_core._component_id(
                tuple((item[0], item[1]) for item in expected_assets)
            )
            != component_id
        ):
            raise ValueError("component_id differs from canonical members")
        split = component.get("split")
        if split not in _SPLITS:
            raise ValueError("component split is invalid")
        for field in ("fresh", "forced_train"):
            if type(component.get(field)) is not bool:
                raise ValueError(f"component {field} must be boolean")
        anchor_iids = component.get("anchor_iids")
        seen_iids = component.get("previously_seen_iids")
        if (
            type(anchor_iids) is not list
            or anchor_iids != sorted(set(anchor_iids))
            or type(seen_iids) is not list
            or seen_iids != sorted(set(seen_iids))
        ):
            raise ValueError("component anchor/seen lists are not canonical")
        expected_anchor_iids = sorted(set(member_iids) & anchors)
        if anchor_iids != expected_anchor_iids:
            raise ValueError("component anchor_iids are inconsistent")
        if seen_iids:
            raise ValueError(
                "frozen R7 indexed commit must have no previously-seen IIDs"
            )
        forced = bool(anchor_iids or seen_iids)
        if (
            component["fresh"] is not (not forced)
            or component["forced_train"] is not forced
            or (forced and split != "train")
        ):
            raise ValueError("component freshness/forced split inconsistent")
        component_by_id[component_id] = component
    if all_component_iids != set(assignment_by_iid):
        raise ValueError("component/assignment IID conservation failed")

    for iid, assignment in assignment_by_iid.items():
        component = component_by_id.get(assignment["component_id"])
        if component is None or iid not in component["member_iids"]:
            raise ValueError(f"iid={iid} assignment component mismatch")
        if (
            assignment["split"] != component["split"]
            or assignment["fresh"] is not component["fresh"]
            or assignment["forced_train"] is not component["forced_train"]
            or assignment["forced_by_anchor"]
            is not bool(component["anchor_iids"])
            or assignment["forced_by_previously_seen"]
            is not bool(component["previously_seen_iids"])
        ):
            raise ValueError(f"iid={iid} assignment flags mismatch")

    counts = summary.get("counts")
    split_counts = Counter(row["split"] for row in assignments)
    freshness_counts = {
        "fresh_iids": sum(row["fresh"] for row in assignments),
        "nonfresh_iids": sum(not row["fresh"] for row in assignments),
        "forced_train_iids": sum(
            row["forced_train"] for row in assignments
        ),
        "forced_by_anchor_iids": sum(
            row["forced_by_anchor"] for row in assignments
        ),
        "forced_by_previously_seen_iids": sum(
            row["forced_by_previously_seen"] for row in assignments
        ),
    }
    if (
        type(counts) is not dict
        or counts.get("candidate_iids") != len(candidate_rows)
        or counts.get("anchor_iids") != len(anchors)
        or counts.get("total_iids") != len(assignments)
        or counts.get("components") != len(components)
        or counts.get("spanning_edges") != len(edges)
        or summary.get("split_iid_counts")
        != {split: split_counts[split] for split in _SPLITS}
        or summary.get("freshness_counts") != freshness_counts
        or done.get("iids") != len(assignments)
        or done.get("components") != len(components)
    ):
        raise ValueError("indexed summary/done census mismatch")

    return {
        "root": root,
        "assignments": assignments,
        "assignment_by_iid": assignment_by_iid,
        "assignment_index": assignment_index,
        "summary": summary,
        "done": done,
        "artifact_digest": indexed_digest,
        "upstream_artifact_digests": upstream_digests,
        "anchor_iids": anchors,
    }


def _strict_data_root(raw_path: Path) -> Path:
    unresolved = raw_path.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(
            f"data root must be a real directory: {unresolved}"
        )
    return unresolved.resolve(strict=True)


def _media_parts(relative_path: str) -> tuple[str, ...]:
    value = _plain_string(relative_path, context="media relative path")
    if "\\" in value:
        raise ValueError("media paths must use POSIX separators")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts:
        raise ValueError("media path must be relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("media path contains traversal/non-canonical parts")
    if pure.as_posix() != value:
        raise ValueError("media path is not canonical POSIX relative form")
    return tuple(pure.parts)


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _open_media(
    data_root: Path,
    relative_path: str,
) -> tuple[int, tuple[int, ...]]:
    parts = _media_parts(relative_path)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(data_root, os.O_RDONLY | directory_flag)
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=directory_fd,
            )
            info = os.fstat(next_fd)
            if not stat.S_ISDIR(info.st_mode):
                os.close(next_fd)
                raise ValueError(
                    f"media path component is not a directory: {component}"
                )
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | nofollow,
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise ValueError(
            f"media path cannot be opened safely: {relative_path}"
        ) from error
    finally:
        os.close(directory_fd)
    info = os.fstat(file_fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(file_fd)
        raise ValueError(f"media is not a regular file: {relative_path}")
    return file_fd, _stat_identity(info)


def _hash_media(
    data_root: Path,
    relative_path: str,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    file_fd, before = _open_media(data_root, relative_path)
    hasher = hashlib.sha256()
    try:
        while True:
            block = os.read(file_fd, 1024 * 1024)
            if not block:
                break
            hasher.update(block)
        after = _stat_identity(os.fstat(file_fd))
    finally:
        os.close(file_fd)
    if after != before:
        raise RuntimeError(
            f"media changed while hashing: {relative_path}"
        )
    return (
        {
            "relative_path": relative_path,
            "sha256": hasher.hexdigest(),
            "bytes": before[3],
        },
        before,
    )


def _assert_media_identity(
    data_root: Path,
    records: Mapping[str, tuple[int, ...]],
    *,
    expected_root_identity: tuple[int, ...],
) -> None:
    if data_root.is_symlink() or not data_root.is_dir():
        raise RuntimeError("media data root changed type")
    if _stat_identity(data_root.stat()) != expected_root_identity:
        raise RuntimeError("media data root changed before publication")
    for relative_path, expected in sorted(records.items()):
        file_fd, actual = _open_media(data_root, relative_path)
        os.close(file_fd)
        if actual != expected:
            raise RuntimeError(
                f"selected media changed before publication: {relative_path}"
            )


def _selection_key(
    *,
    iid: str,
) -> str:
    payload = (
        f"{SELECTION_KEY_SCHEMA}\0{SELECTION_SEED}\0{iid}"
    ).encode("utf-8")
    return _sha256_bytes(payload)


@dataclass(frozen=True)
class _Derived:
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    media_identities: Mapping[str, tuple[int, ...]]
    input_paths: Mapping[str, Path]
    input_snapshot: Mapping[str, Mapping[str, Any]]
    data_root: Path
    data_root_identity: tuple[int, ...]


def _derive(
    *,
    expansion_manifest_dir: Path,
    candidate_manifest_dir: Path,
    indexed_graph_dir: Path,
    data_root: Path,
    expected_expansion_artifact_digest: str,
    expected_candidate_artifact_digest: str,
    expected_indexed_artifact_digest: str,
) -> _Derived:
    expansion_root, expansion_paths = _strict_directory(
        expansion_manifest_dir,
        names=_SOURCE_NAMES,
        context="R7 expansion manifest",
    )
    candidate_root, candidate_paths = _strict_directory(
        candidate_manifest_dir,
        names=_CANDIDATE_NAMES,
        context="R7 visual candidate commit",
    )
    indexed_root, indexed_paths = _strict_directory(
        indexed_graph_dir,
        names=_INDEXED_NAMES,
        context="R7 indexed visual graph commit",
    )
    input_paths = {
        **{
            f"expansion/{name}": path
            for name, path in expansion_paths.items()
        },
        **{
            f"candidate/{name}": path
            for name, path in candidate_paths.items()
        },
        **{
            f"indexed_graph/{name}": path
            for name, path in indexed_paths.items()
        },
    }
    before = _snapshot(input_paths)
    candidate = _validate_candidate_commit(
        expansion_manifest_dir=expansion_root,
        candidate_manifest_dir=candidate_root,
        expected_expansion_artifact_digest=(
            expected_expansion_artifact_digest
        ),
        expected_candidate_artifact_digest=(
            expected_candidate_artifact_digest
        ),
    )
    indexed = _validate_indexed_commit(
        indexed_graph_dir=indexed_root,
        candidate_rows=candidate["rows"],
        expected_indexed_artifact_digest=expected_indexed_artifact_digest,
    )
    after_validation = _snapshot(input_paths)
    if after_validation != before:
        raise RuntimeError("upstream commit changed during validation")

    data_root = _strict_data_root(data_root)
    data_root_identity = _stat_identity(data_root.stat())
    negative_records: list[dict[str, Any]] = []
    for row in candidate["rows"]:
        if row["cohort"] != "pseudo_negative":
            continue
        key = _selection_key(
            iid=row["iid"],
        )
        negative_records.append(
            {
                "iid": row["iid"],
                "source_row_sha256": row["source_row_sha256"],
                "selection_key_sha256": key,
            }
        )
    negative_records.sort(
        key=lambda item: (item["selection_key_sha256"], item["iid"])
    )
    if len(negative_records) != TRUSTED_NEGATIVE_POPULATION:
        raise RuntimeError("trusted-negative population changed")
    selected_negative = negative_records[:TRUSTED_NEGATIVE_SAMPLE]
    selected_negative_iids = {
        item["iid"] for item in selected_negative
    }
    negative_rank = {
        item["iid"]: rank
        for rank, item in enumerate(selected_negative, start=1)
    }
    negative_key = {
        item["iid"]: item["selection_key_sha256"]
        for item in selected_negative
    }

    media_cache: dict[str, dict[str, Any]] = {}
    media_identities: dict[str, tuple[int, ...]] = {}

    def media(relative_path: str) -> dict[str, Any]:
        if relative_path not in media_cache:
            binding, identity = _hash_media(data_root, relative_path)
            media_cache[relative_path] = binding
            media_identities[relative_path] = identity
        return dict(media_cache[relative_path])

    output_rows: list[dict[str, Any]] = []
    candidate_row_index = {
        row["iid"]: index for index, row in enumerate(candidate["rows"])
    }
    for candidate_row in candidate["rows"]:
        iid = candidate_row["iid"]
        cohort = candidate_row["cohort"]
        if (
            cohort == "pseudo_negative"
            and iid not in selected_negative_iids
        ):
            continue
        if cohort not in {"pseudo_positive", "pseudo_negative"}:
            raise RuntimeError(f"unsupported candidate cohort: {cohort}")
        source_record = candidate["source_by_iid"][iid]
        source_row = source_record["row"]
        source_label = source_row["r7_expansion_manifest"]
        assignment_row = indexed["assignment_by_iid"][iid]
        if assignment_row["anchor"]:
            raise ValueError(f"candidate IID is an anchor: {iid}")
        positive = cohort == "pseudo_positive"
        if positive:
            if (
                source_label.get("bucket") != "positive"
                or source_label.get("negative_role") is not None
            ):
                raise ValueError(f"iid={iid} positive label mismatch")
            label_class = "positive"
            negative_type: str | None = None
            action_signature: str | None = _plain_string(
                source_label.get("action_signature"),
                context=f"iid={iid} action_signature",
            )
            method = "complete_positive_census"
            population_rows = POSITIVE_CENSUS
            selected_rows = POSITIVE_CENSUS
            selection_key: str | None = None
            selection_rank: int | None = None
            inclusion_probability = 1.0
            inverse_weight = 1.0
        else:
            if (
                source_label.get("bucket") != "negative"
                or source_label.get("negative_role")
                != "pseudo_negative"
            ):
                raise ValueError(f"iid={iid} trusted-negative mismatch")
            label_class = "negative"
            negative_type = _plain_string(
                source_label.get("negative_type"),
                context=f"iid={iid} negative_type",
            )
            if source_label.get("action_signature") is not None:
                raise ValueError(
                    f"iid={iid} negative action_signature must be null"
                )
            action_signature = None
            method = "sha256_bottom_k_without_replacement"
            population_rows = TRUSTED_NEGATIVE_POPULATION
            selected_rows = TRUSTED_NEGATIVE_SAMPLE
            selection_key = negative_key[iid]
            selection_rank = negative_rank[iid]
            inclusion_probability = (
                TRUSTED_NEGATIVE_SAMPLE
                / TRUSTED_NEGATIVE_POPULATION
            )
            inverse_weight = (
                TRUSTED_NEGATIVE_POPULATION
                / TRUSTED_NEGATIVE_SAMPLE
            )
        primary_family = _plain_string(
            source_label.get("primary_family"),
            context=f"iid={iid} primary_family",
        )
        src_video = candidate_row["src_video"]
        tgt_video = candidate_row["tgt_video"]
        output_row: dict[str, Any] = {
            "schema_version": ROW_SCHEMA,
            "iid": iid,
            "input_digest": candidate_row["input_digest"],
            "prompt": candidate_row["prompt"],
            "src_video": src_video,
            "tgt_video": tgt_video,
            "label": {
                "schema_version": LABEL_SCHEMA,
                "class": label_class,
                "negative_type": negative_type,
                "action_signature": action_signature,
                "primary_family": primary_family,
                "provenance_kind":
                    "r7_expansion_qwen_pseudo_label",
                "human_label": False,
            },
            "assignment": {
                "schema_version": ASSIGNMENT_SCHEMA,
                "component_id": assignment_row["component_id"],
                "split": assignment_row["split"],
                "fresh": assignment_row["fresh"],
                "anchor": False,
                "forced_train": assignment_row["forced_train"],
                "forced_by_anchor":
                    assignment_row["forced_by_anchor"],
                "forced_by_previously_seen":
                    assignment_row["forced_by_previously_seen"],
            },
            "sampling": {
                "schema_version": SAMPLING_SCHEMA,
                "method": method,
                "seed": SELECTION_SEED,
                "population_rows": population_rows,
                "selected_rows": selected_rows,
                "selection_key_sha256": selection_key,
                "selection_rank": selection_rank,
                "inclusion_probability": inclusion_probability,
                "inverse_probability_weight": inverse_weight,
            },
            "source_bindings": {
                "schema_version": SOURCE_BINDINGS_SCHEMA,
                "expansion": {
                    "artifact_digest":
                        candidate["expansion_artifact_digest"],
                    "source_row_sha256":
                        candidate_row["source_row_sha256"],
                    "source_line_number":
                        source_record["line_number"],
                },
                "candidate": {
                    "artifact_digest":
                        candidate["candidate_artifact_digest"],
                    "candidate_row_sha256":
                        _object_digest(candidate_row),
                    "row_index": candidate_row_index[iid],
                },
                "indexed_graph": {
                    "artifact_digest": indexed["artifact_digest"],
                    "assignment_row_sha256":
                        _object_digest(assignment_row),
                    "assignment_row_index":
                        indexed["assignment_index"][iid],
                    "upstream_artifact_digests": dict(
                        sorted(
                            indexed[
                                "upstream_artifact_digests"
                            ].items()
                        )
                    ),
                },
                "media": {
                    "data_root": str(data_root),
                    "src_video": media(src_video),
                    "tgt_video": media(tgt_video),
                },
            },
            "formal_evidence": False,
            "formal_split": False,
            "human_labels_asserted": False,
            "training_authorized": False,
            "generation_authorized": False,
        }
        if set(output_row) != set(ROW_FIELDS):
            raise RuntimeError("internal temporal row projection mismatch")
        output_rows.append(output_row)

    expected_output_rows = POSITIVE_CENSUS + TRUSTED_NEGATIVE_SAMPLE
    if len(output_rows) != expected_output_rows:
        raise RuntimeError("temporal-screen row count differs from contract")
    manifest_raw = _jsonl_bytes(output_rows)
    selected_counts = Counter(
        row["label"]["class"] for row in output_rows
    )
    split_counts = Counter(
        row["assignment"]["split"] for row in output_rows
    )
    fresh_counts = Counter(
        "fresh" if row["assignment"]["fresh"] else "nonfresh"
        for row in output_rows
    )
    family_counts = Counter(
        row["label"]["primary_family"] for row in output_rows
    )
    negative_type_counts = Counter(
        row["label"]["negative_type"]
        for row in output_rows
        if row["label"]["class"] == "negative"
    )
    unique_bytes = sum(
        binding["bytes"] for binding in media_cache.values()
    )
    referenced_bytes = sum(
        row["source_bindings"]["media"][role]["bytes"]
        for row in output_rows
        for role in ("src_video", "tgt_video")
    )
    snapshot_groups = {
        "expansion": {
            key.removeprefix("expansion/"): value
            for key, value in before.items()
            if key.startswith("expansion/")
        },
        "candidate": {
            key.removeprefix("candidate/"): value
            for key, value in before.items()
            if key.startswith("candidate/")
        },
        "indexed_graph": {
            key.removeprefix("indexed_graph/"): value
            for key, value in before.items()
            if key.startswith("indexed_graph/")
        },
    }
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete",
        "policy_version": POLICY_VERSION,
        "implementation": _implementation_provenance(),
        "input_bindings": {
            "expansion": {
                "path": str(expansion_root),
                "artifact_digest":
                    candidate["expansion_artifact_digest"],
                "files": snapshot_groups["expansion"],
            },
            "candidate": {
                "path": str(candidate_root),
                "artifact_digest":
                    candidate["candidate_artifact_digest"],
                "input_artifact_digest":
                    candidate["candidate_done"][
                        "input_artifact_digest"
                    ],
                "files": snapshot_groups["candidate"],
            },
            "indexed_graph": {
                "path": str(indexed_root),
                "artifact_digest": indexed["artifact_digest"],
                "input_artifact_digests": dict(
                    sorted(
                        indexed["upstream_artifact_digests"].items()
                    )
                ),
                "files": snapshot_groups["indexed_graph"],
            },
            "media": {
                "data_root": str(data_root),
                "device": data_root_identity[0],
                "inode": data_root_identity[1],
            },
        },
        "selection": {
            "schema_version": POLICY_VERSION,
            "seed": SELECTION_SEED,
            "positive_method": "complete_positive_census",
            "positive_population_rows": POSITIVE_CENSUS,
            "positive_selected_rows": POSITIVE_CENSUS,
            "negative_method":
                "sha256_bottom_k_without_replacement",
            "negative_key_schema": SELECTION_KEY_SCHEMA,
            "negative_population_rows":
                TRUSTED_NEGATIVE_POPULATION,
            "negative_selected_rows": TRUSTED_NEGATIVE_SAMPLE,
            "negative_population_digest":
                _object_digest(negative_records),
            "selected_negative_iids_digest": _object_digest(
                [item["iid"] for item in selected_negative]
            ),
            "negative_cutoff": {
                "selection_key_sha256":
                    selected_negative[-1][
                        "selection_key_sha256"
                    ],
                "iid": selected_negative[-1]["iid"],
            },
        },
        "counts": {
            "candidate_population_rows": CANDIDATE_POPULATION,
            "indexed_anchor_rows_excluded": EXPECTED_ANCHOR_ROWS,
            "output_rows": len(output_rows),
            "positive_rows": selected_counts["positive"],
            "negative_rows": selected_counts["negative"],
            "by_split": {
                split: split_counts[split] for split in _SPLITS
            },
            "by_freshness": {
                name: fresh_counts[name]
                for name in ("fresh", "nonfresh")
            },
            "by_primary_family": dict(sorted(family_counts.items())),
            "by_negative_type": dict(
                sorted(negative_type_counts.items())
            ),
        },
        "media": {
            "references": 2 * len(output_rows),
            "unique_paths": len(media_cache),
            "referenced_bytes": referenced_bytes,
            "unique_bytes": unique_bytes,
            "binding_digest": _object_digest(
                {
                    name: media_cache[name]
                    for name in sorted(media_cache)
                }
            ),
        },
        "semantics": {
            "labels_are_pseudo": True,
            "human_labels_asserted": False,
            "formal_evidence": False,
            "formal_split": False,
            "indexed_split_is_diagnostic_only": True,
            "training_authorized": False,
            "generation_authorized": False,
        },
        "output": {
            "name": MANIFEST_NAME,
            "rows": len(output_rows),
            "sha256": _sha256_bytes(manifest_raw),
            "order": "ascending_candidate_row_index",
            "row_encoding": "canonical_json_utf8_lf",
        },
    }
    _assert_snapshot(input_paths, before)
    _assert_media_identity(
        data_root,
        media_identities,
        expected_root_identity=data_root_identity,
    )
    return _Derived(
        rows=tuple(output_rows),
        summary=summary,
        media_identities=media_identities,
        input_paths=input_paths,
        input_snapshot=before,
        data_root=data_root,
        data_root_identity=data_root_identity,
    )


def _payloads(derived: _Derived) -> dict[str, bytes]:
    manifest_raw = _jsonl_bytes(derived.rows)
    summary_raw = _pretty_json_bytes(derived.summary)
    output_sha = {
        MANIFEST_NAME: _sha256_bytes(manifest_raw),
        SUMMARY_NAME: _sha256_bytes(summary_raw),
    }
    bindings = derived.summary["input_bindings"]
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "input_artifact_digests": {
            "expansion": bindings["expansion"]["artifact_digest"],
            "candidate": bindings["candidate"]["artifact_digest"],
            "indexed_graph":
                bindings["indexed_graph"]["artifact_digest"],
            "indexed_graph_upstreams": dict(
                bindings["indexed_graph"][
                    "input_artifact_digests"
                ]
            ),
        },
        "input_binding_digest": _object_digest(bindings),
        "output_rows": len(derived.rows),
        "output_sha256": output_sha,
        "artifact_digest": _object_digest(output_sha),
        "human_labels_asserted": False,
        "formal_evidence": False,
        "formal_split": False,
        "training_authorized": False,
        "generation_authorized": False,
        "permission_contract": artifact_permissions.permission_contract(),
    }
    return {
        MANIFEST_NAME: manifest_raw,
        SUMMARY_NAME: summary_raw,
        DONE_NAME: _pretty_json_bytes(done),
    }


def _validate_media_file_binding(
    value: Any,
    *,
    expected_path: str,
    context: str,
) -> dict[str, Any]:
    binding = _exact_object(value, MEDIA_FILE_FIELDS, context=context)
    if binding.get("relative_path") != expected_path:
        raise ValueError(f"{context} relative path mismatch")
    _media_parts(expected_path)
    _sha_field(binding.get("sha256"), context=f"{context} sha256")
    if type(binding.get("bytes")) is not int or binding["bytes"] < 0:
        raise ValueError(f"{context} bytes must be non-negative integer")
    return binding


def _validate_output_row(
    row: Mapping[str, Any],
    *,
    row_number: int,
) -> dict[str, Any]:
    _exact_object(row, ROW_FIELDS, context=f"row {row_number}")
    iid = _plain_string(row.get("iid"), context=f"row {row_number} IID")
    if row.get("schema_version") != ROW_SCHEMA:
        raise ValueError(f"iid={iid} row schema mismatch")
    _sha_field(row.get("input_digest"), context=f"iid={iid} input digest")
    _plain_string(row.get("prompt"), context=f"iid={iid} prompt")
    src_video = _plain_string(
        row.get("src_video"), context=f"iid={iid} src_video"
    )
    tgt_video = _plain_string(
        row.get("tgt_video"), context=f"iid={iid} tgt_video"
    )

    label = _exact_object(
        row.get("label"), LABEL_FIELDS, context=f"iid={iid} label"
    )
    if (
        label.get("schema_version") != LABEL_SCHEMA
        or label.get("class") not in {"positive", "negative"}
        or label.get("provenance_kind")
        != "r7_expansion_qwen_pseudo_label"
        or label.get("human_label") is not False
    ):
        raise ValueError(f"iid={iid} label semantics mismatch")
    _plain_string(
        label.get("primary_family"),
        context=f"iid={iid} primary_family",
    )
    if label["class"] == "positive":
        _plain_string(
            label.get("action_signature"),
            context=f"iid={iid} action_signature",
        )
        if label.get("negative_type") is not None:
            raise ValueError(f"iid={iid} negative_type mismatch")
    else:
        if label.get("action_signature") is not None:
            raise ValueError(
                f"iid={iid} negative action_signature must be null"
            )
        _plain_string(
            label.get("negative_type"),
            context=f"iid={iid} negative_type",
        )

    assignment = _exact_object(
        row.get("assignment"),
        ASSIGNMENT_FIELDS,
        context=f"iid={iid} assignment",
    )
    if (
        assignment.get("schema_version") != ASSIGNMENT_SCHEMA
        or assignment.get("split") not in _SPLITS
        or assignment.get("anchor") is not False
    ):
        raise ValueError(f"iid={iid} assignment semantics mismatch")
    _plain_string(
        assignment.get("component_id"),
        context=f"iid={iid} component_id",
    )
    for field in (
        "fresh",
        "forced_train",
        "forced_by_anchor",
        "forced_by_previously_seen",
    ):
        if type(assignment.get(field)) is not bool:
            raise ValueError(f"iid={iid} assignment {field} invalid")
    if (
        assignment["forced_train"]
        is not (
            assignment["forced_by_anchor"]
            or assignment["forced_by_previously_seen"]
        )
        or assignment["fresh"] is assignment["forced_train"]
        or (
            assignment["forced_train"]
            and assignment["split"] != "train"
        )
    ):
        raise ValueError(f"iid={iid} assignment flags inconsistent")

    sampling = _exact_object(
        row.get("sampling"),
        SAMPLING_FIELDS,
        context=f"iid={iid} sampling",
    )
    if (
        sampling.get("schema_version") != SAMPLING_SCHEMA
        or sampling.get("seed") != SELECTION_SEED
    ):
        raise ValueError(f"iid={iid} sampling schema/seed mismatch")
    if label["class"] == "positive":
        expected_sampling = {
            "method": "complete_positive_census",
            "population_rows": POSITIVE_CENSUS,
            "selected_rows": POSITIVE_CENSUS,
            "selection_key_sha256": None,
            "selection_rank": None,
            "inclusion_probability": 1.0,
            "inverse_probability_weight": 1.0,
        }
    else:
        expected_sampling = {
            "method": "sha256_bottom_k_without_replacement",
            "population_rows": TRUSTED_NEGATIVE_POPULATION,
            "selected_rows": TRUSTED_NEGATIVE_SAMPLE,
            "inclusion_probability":
                TRUSTED_NEGATIVE_SAMPLE
                / TRUSTED_NEGATIVE_POPULATION,
            "inverse_probability_weight":
                TRUSTED_NEGATIVE_POPULATION
                / TRUSTED_NEGATIVE_SAMPLE,
        }
        _sha_field(
            sampling.get("selection_key_sha256"),
            context=f"iid={iid} selection key",
        )
        if (
            type(sampling.get("selection_rank")) is not int
            or not 1
            <= sampling["selection_rank"]
            <= TRUSTED_NEGATIVE_SAMPLE
        ):
            raise ValueError(f"iid={iid} selection rank invalid")
    for field, expected in expected_sampling.items():
        if sampling.get(field) != expected:
            raise ValueError(f"iid={iid} sampling {field} mismatch")

    bindings = _exact_object(
        row.get("source_bindings"),
        SOURCE_BINDINGS_FIELDS,
        context=f"iid={iid} source_bindings",
    )
    if bindings.get("schema_version") != SOURCE_BINDINGS_SCHEMA:
        raise ValueError(f"iid={iid} binding schema mismatch")
    expansion = _exact_object(
        bindings.get("expansion"),
        EXPANSION_BINDING_FIELDS,
        context=f"iid={iid} expansion binding",
    )
    candidate = _exact_object(
        bindings.get("candidate"),
        CANDIDATE_BINDING_FIELDS,
        context=f"iid={iid} candidate binding",
    )
    indexed = _exact_object(
        bindings.get("indexed_graph"),
        INDEXED_BINDING_FIELDS,
        context=f"iid={iid} indexed binding",
    )
    media = _exact_object(
        bindings.get("media"),
        MEDIA_BINDING_FIELDS,
        context=f"iid={iid} media binding",
    )
    for context, value in (
        ("expansion artifact", expansion.get("artifact_digest")),
        ("source row", expansion.get("source_row_sha256")),
        ("candidate artifact", candidate.get("artifact_digest")),
        ("candidate row", candidate.get("candidate_row_sha256")),
        ("indexed artifact", indexed.get("artifact_digest")),
        ("assignment row", indexed.get("assignment_row_sha256")),
    ):
        _sha_field(value, context=f"iid={iid} {context}")
    if (
        type(expansion.get("source_line_number")) is not int
        or expansion["source_line_number"] <= 0
        or type(candidate.get("row_index")) is not int
        or not 0 <= candidate["row_index"] < CANDIDATE_POPULATION
        or type(indexed.get("assignment_row_index")) is not int
        or not 0
        <= indexed["assignment_row_index"]
        < EXPECTED_INDEXED_ROWS
    ):
        raise ValueError(f"iid={iid} source indices invalid")
    upstream = indexed.get("upstream_artifact_digests")
    if type(upstream) is not dict or not upstream:
        raise ValueError(f"iid={iid} indexed upstream digests invalid")
    for name, digest in upstream.items():
        _plain_string(name, context=f"iid={iid} upstream name")
        _sha_field(digest, context=f"iid={iid} upstream {name}")
    _plain_string(media.get("data_root"), context=f"iid={iid} data root")
    _validate_media_file_binding(
        media.get("src_video"),
        expected_path=src_video,
        context=f"iid={iid} source media",
    )
    _validate_media_file_binding(
        media.get("tgt_video"),
        expected_path=tgt_video,
        context=f"iid={iid} target media",
    )
    for field in (
        "formal_evidence",
        "formal_split",
        "human_labels_asserted",
        "training_authorized",
        "generation_authorized",
    ):
        if row.get(field) is not False:
            raise ValueError(f"iid={iid} forbidden flag {field} asserted")
    return dict(row)


def _single(values: set[Any], *, context: str) -> Any:
    if len(values) != 1:
        raise ValueError(f"{context} is inconsistent across rows")
    return next(iter(values))


def validate_candidate_temporal_manifest(
    path: Path,
    *,
    expected_summary_path: Path | None = None,
    expected_done_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one committed temporal screen without trusting itself.

    Passing externally retained summary/done files upgrades validation from
    internal hash-chain consistency to byte-exact comparison with those
    anchors.
    """

    root, paths = _strict_directory(
        path,
        names=OUTPUT_NAMES,
        context="candidate temporal-screen commit",
    )
    rows, manifest_raw = _load_canonical_jsonl(
        paths[MANIFEST_NAME],
        context=MANIFEST_NAME,
    )
    summary, summary_raw = _load_pretty_object(
        paths[SUMMARY_NAME],
        context=SUMMARY_NAME,
    )
    done, _done_raw = _load_pretty_object(
        paths[DONE_NAME],
        context=DONE_NAME,
    )
    if expected_summary_path is not None:
        expected = expected_summary_path.expanduser()
        if expected.is_symlink() or not expected.is_file():
            raise FileNotFoundError(
                f"expected summary is not a regular file: {expected}"
            )
        if expected.read_bytes() != summary_raw:
            raise ValueError("summary differs from external expected anchor")
    if expected_done_path is not None:
        expected = expected_done_path.expanduser()
        if expected.is_symlink() or not expected.is_file():
            raise FileNotFoundError(
                f"expected done is not a regular file: {expected}"
            )
        if expected.read_bytes() != paths[DONE_NAME].read_bytes():
            raise ValueError("done differs from external expected anchor")

    validated_rows = [
        _validate_output_row(row, row_number=index)
        for index, row in enumerate(rows, start=1)
    ]
    expected_rows = POSITIVE_CENSUS + TRUSTED_NEGATIVE_SAMPLE
    if len(validated_rows) != expected_rows:
        raise ValueError("temporal-screen row census differs from contract")
    iids = [row["iid"] for row in validated_rows]
    if len(set(iids)) != len(iids):
        raise ValueError("temporal-screen contains duplicate IID")
    candidate_indices = [
        row["source_bindings"]["candidate"]["row_index"]
        for row in validated_rows
    ]
    if candidate_indices != sorted(candidate_indices) or len(
        set(candidate_indices)
    ) != len(candidate_indices):
        raise ValueError("rows are not in unique candidate source order")
    source_lines = [
        row["source_bindings"]["expansion"]["source_line_number"]
        for row in validated_rows
    ]
    if source_lines != sorted(source_lines) or len(
        set(source_lines)
    ) != len(source_lines):
        raise ValueError("rows are not in unique expansion source order")
    assignment_indices = [
        row["source_bindings"]["indexed_graph"][
            "assignment_row_index"
        ]
        for row in validated_rows
    ]
    if len(set(assignment_indices)) != len(assignment_indices):
        raise ValueError("indexed assignment row indices are not unique")
    for field_path, context in (
        (("expansion", "source_row_sha256"), "source row digests"),
        (("candidate", "candidate_row_sha256"), "candidate row digests"),
        (
            ("indexed_graph", "assignment_row_sha256"),
            "assignment row digests",
        ),
    ):
        values = {
            row["source_bindings"][field_path[0]][field_path[1]]
            for row in validated_rows
        }
        if len(values) != len(validated_rows):
            raise ValueError(f"{context} are not unique")

    class_counts = Counter(row["label"]["class"] for row in validated_rows)
    if class_counts != Counter(
        {
            "positive": POSITIVE_CENSUS,
            "negative": TRUSTED_NEGATIVE_SAMPLE,
        }
    ):
        raise ValueError("positive/negative selected census differs")
    expansion_digest = _single(
        {
            row["source_bindings"]["expansion"]["artifact_digest"]
            for row in validated_rows
        },
        context="expansion artifact digest",
    )
    candidate_digest = _single(
        {
            row["source_bindings"]["candidate"]["artifact_digest"]
            for row in validated_rows
        },
        context="candidate artifact digest",
    )
    indexed_digest = _single(
        {
            row["source_bindings"]["indexed_graph"]["artifact_digest"]
            for row in validated_rows
        },
        context="indexed artifact digest",
    )
    upstream_digest = _single(
        {
            _canonical_bytes(
                row["source_bindings"]["indexed_graph"][
                    "upstream_artifact_digests"
                ]
            )
            for row in validated_rows
        },
        context="indexed upstream artifact digests",
    )
    upstream = _parse_json(
        upstream_digest,
        context="indexed upstream artifact digests",
    )
    data_root = _single(
        {
            row["source_bindings"]["media"]["data_root"]
            for row in validated_rows
        },
        context="media data root",
    )

    negative_rows = [
        row for row in validated_rows
        if row["label"]["class"] == "negative"
    ]
    for row in negative_rows:
        expected_key = _selection_key(
            iid=row["iid"],
        )
        if row["sampling"]["selection_key_sha256"] != expected_key:
            raise ValueError(
                f"iid={row['iid']} negative selection key mismatch"
            )
    ordered_negatives = sorted(
        negative_rows,
        key=lambda row: (
            row["sampling"]["selection_key_sha256"],
            row["iid"],
        ),
    )
    if [
        row["sampling"]["selection_rank"] for row in ordered_negatives
    ] != list(range(1, TRUSTED_NEGATIVE_SAMPLE + 1)):
        raise ValueError("negative bottom-k ranks are not exact")

    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("policy_version") != POLICY_VERSION
    ):
        raise ValueError("temporal summary schema/status/policy mismatch")
    required_summary_fields = {
        "schema_version",
        "status",
        "policy_version",
        "implementation",
        "input_bindings",
        "selection",
        "counts",
        "media",
        "semantics",
        "output",
    }
    if set(summary) != required_summary_fields:
        raise ValueError("temporal summary fields differ")
    implementation = summary.get("implementation")
    if (
        type(implementation) is not dict
        or set(implementation) != {"files", "bundle_sha256"}
        or type(implementation.get("files")) is not dict
        or not implementation["files"]
    ):
        raise ValueError("summary implementation provenance invalid")
    implementation_digests: dict[str, str] = {}
    for name, record in sorted(implementation["files"].items()):
        if type(record) is not dict or set(record) != {"sha256"}:
            raise ValueError("summary implementation file record invalid")
        implementation_digests[name] = _sha_field(
            record["sha256"],
            context=f"implementation {name}",
        )
    if implementation.get("bundle_sha256") != _object_digest(
        implementation_digests
    ):
        raise ValueError("summary implementation bundle mismatch")

    bindings = summary.get("input_bindings")
    if type(bindings) is not dict or set(bindings) != {
        "expansion",
        "candidate",
        "indexed_graph",
        "media",
    }:
        raise ValueError("summary input bindings differ")
    if any(type(record) is not dict for record in bindings.values()):
        raise ValueError("summary input binding records must be objects")
    if (
        bindings["expansion"].get("artifact_digest")
        != expansion_digest
        or bindings["candidate"].get("artifact_digest")
        != candidate_digest
        or bindings["candidate"].get("input_artifact_digest")
        != expansion_digest
        or bindings["indexed_graph"].get("artifact_digest")
        != indexed_digest
        or bindings["indexed_graph"].get("input_artifact_digests")
        != upstream
        or bindings["media"].get("data_root") != data_root
    ):
        raise ValueError("summary/row input bindings differ")
    for commit_name in ("expansion", "candidate", "indexed_graph"):
        record = bindings[commit_name]
        if (
            type(record.get("path")) is not str
            or type(record.get("files")) is not dict
            or not record["files"]
        ):
            raise ValueError(
                f"summary {commit_name} file binding is invalid"
            )
        for name, file_record in record["files"].items():
            if (
                type(name) is not str
                or type(file_record) is not dict
                or set(file_record) != {"sha256", "bytes"}
            ):
                raise ValueError("summary input file record invalid")
            _sha_field(
                file_record["sha256"],
                context=f"summary {commit_name}/{name}",
            )
            if (
                type(file_record["bytes"]) is not int
                or file_record["bytes"] < 0
            ):
                raise ValueError("summary input file bytes invalid")
    if (
        type(bindings["media"].get("device")) is not int
        or type(bindings["media"].get("inode")) is not int
    ):
        raise ValueError("summary media root identity invalid")

    selection = summary.get("selection")
    selection_fields = {
        "schema_version",
        "seed",
        "positive_method",
        "positive_population_rows",
        "positive_selected_rows",
        "negative_method",
        "negative_key_schema",
        "negative_population_rows",
        "negative_selected_rows",
        "negative_population_digest",
        "selected_negative_iids_digest",
        "negative_cutoff",
    }
    if type(selection) is not dict or set(selection) != selection_fields:
        raise ValueError("summary selection fields differ")
    expected_selection_values = {
        "schema_version": POLICY_VERSION,
        "seed": SELECTION_SEED,
        "positive_method": "complete_positive_census",
        "positive_population_rows": POSITIVE_CENSUS,
        "positive_selected_rows": POSITIVE_CENSUS,
        "negative_method": "sha256_bottom_k_without_replacement",
        "negative_key_schema": SELECTION_KEY_SCHEMA,
        "negative_population_rows": TRUSTED_NEGATIVE_POPULATION,
        "negative_selected_rows": TRUSTED_NEGATIVE_SAMPLE,
    }
    for field, expected in expected_selection_values.items():
        if selection.get(field) != expected:
            raise ValueError(f"summary selection {field} mismatch")
    _sha_field(
        selection.get("negative_population_digest"),
        context="negative population digest",
    )
    expected_selected_digest = _object_digest(
        [row["iid"] for row in ordered_negatives]
    )
    if (
        selection.get("selected_negative_iids_digest")
        != expected_selected_digest
        or selection.get("negative_cutoff")
        != {
            "selection_key_sha256": ordered_negatives[-1][
                "sampling"
            ]["selection_key_sha256"],
            "iid": ordered_negatives[-1]["iid"],
        }
    ):
        raise ValueError("summary selected-negative commitment mismatch")

    split_counts = Counter(
        row["assignment"]["split"] for row in validated_rows
    )
    freshness_counts = Counter(
        "fresh" if row["assignment"]["fresh"] else "nonfresh"
        for row in validated_rows
    )
    family_counts = Counter(
        row["label"]["primary_family"] for row in validated_rows
    )
    negative_type_counts = Counter(
        row["label"]["negative_type"]
        for row in validated_rows
        if row["label"]["class"] == "negative"
    )
    expected_counts = {
        "candidate_population_rows": CANDIDATE_POPULATION,
        "indexed_anchor_rows_excluded": EXPECTED_ANCHOR_ROWS,
        "output_rows": len(validated_rows),
        "positive_rows": class_counts["positive"],
        "negative_rows": class_counts["negative"],
        "by_split": {split: split_counts[split] for split in _SPLITS},
        "by_freshness": {
            name: freshness_counts[name]
            for name in ("fresh", "nonfresh")
        },
        "by_primary_family": dict(sorted(family_counts.items())),
        "by_negative_type": dict(sorted(negative_type_counts.items())),
    }
    if summary.get("counts") != expected_counts:
        raise ValueError("summary selected-row counts mismatch")
    expected_semantics = {
        "labels_are_pseudo": True,
        "human_labels_asserted": False,
        "formal_evidence": False,
        "formal_split": False,
        "indexed_split_is_diagnostic_only": True,
        "training_authorized": False,
        "generation_authorized": False,
    }
    if summary.get("semantics") != expected_semantics:
        raise ValueError("summary forbidden semantics asserted")

    media_by_path: dict[str, dict[str, Any]] = {}
    referenced_bytes = 0
    for row in validated_rows:
        for role in ("src_video", "tgt_video"):
            record = row["source_bindings"]["media"][role]
            referenced_bytes += record["bytes"]
            previous = media_by_path.setdefault(
                record["relative_path"], record
            )
            if previous != record:
                raise ValueError("same media path has inconsistent binding")
    expected_media = {
        "references": 2 * len(validated_rows),
        "unique_paths": len(media_by_path),
        "referenced_bytes": referenced_bytes,
        "unique_bytes": sum(
            record["bytes"] for record in media_by_path.values()
        ),
        "binding_digest": _object_digest(
            {
                name: media_by_path[name]
                for name in sorted(media_by_path)
            }
        ),
    }
    if summary.get("media") != expected_media:
        raise ValueError("summary media commitment mismatch")
    expected_output = {
        "name": MANIFEST_NAME,
        "rows": len(validated_rows),
        "sha256": _sha256_bytes(manifest_raw),
        "order": "ascending_candidate_row_index",
        "row_encoding": "canonical_json_utf8_lf",
    }
    if summary.get("output") != expected_output:
        raise ValueError("summary output commitment mismatch")

    done_fields = {
        "schema_version",
        "status",
        "input_artifact_digests",
        "input_binding_digest",
        "output_rows",
        "output_sha256",
        "artifact_digest",
        "human_labels_asserted",
        "formal_evidence",
        "formal_split",
        "training_authorized",
        "generation_authorized",
        "permission_contract",
    }
    if type(done) is not dict or set(done) != done_fields:
        raise ValueError("temporal done fields differ")
    output_sha = {
        MANIFEST_NAME: _sha256_bytes(manifest_raw),
        SUMMARY_NAME: _sha256_bytes(summary_raw),
    }
    expected_input_digests = {
        "expansion": expansion_digest,
        "candidate": candidate_digest,
        "indexed_graph": indexed_digest,
        "indexed_graph_upstreams": upstream,
    }
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("input_artifact_digests")
        != expected_input_digests
        or done.get("input_binding_digest") != _object_digest(bindings)
        or done.get("output_rows") != len(validated_rows)
        or done.get("output_sha256") != output_sha
        or done.get("artifact_digest") != _object_digest(output_sha)
    ):
        raise ValueError("temporal done hash/input chain mismatch")
    artifact_permissions.validate_permission_contract(
        done.get("permission_contract")
    )
    for field in (
        "human_labels_asserted",
        "formal_evidence",
        "formal_split",
        "training_authorized",
        "generation_authorized",
    ):
        if done.get(field) is not False:
            raise ValueError(f"done forbidden flag {field} asserted")
    artifact_permissions.assert_sealed_tree(root)
    return {
        "directory": root,
        "rows": validated_rows,
        "summary": summary,
        "done": done,
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
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
    pre_publish_check: Callable[[], None],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        for name in OUTPUT_NAMES:
            _write_file(staging / name, payloads[name])
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        pre_publish_check()
        artifact_permissions.seal_staging_tree(staging)
        artifact_permissions.assert_sealed_tree(staging)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(
                f"output appeared during publication: {output_dir}"
            )
        # Darwin rejects renaming a source directory whose own mode is 0555,
        # even when both parents are writable.  Children stay sealed at 0444;
        # thaw only the private staging root for the rename, then reseal and
        # fsync the now-visible root before fsyncing its parent.  No payload
        # file is writable during this compatibility transition.
        output_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fchmod(output_fd, 0o700)
            os.rename(staging, output_dir)
            os.fchmod(
                output_fd, artifact_permissions.DIRECTORY_MODE
            )
            os.fsync(output_fd)
        finally:
            os.close(output_fd)
        artifact_permissions.assert_sealed_tree(output_dir)
        parent_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if staging.exists():
            artifact_permissions.remove_staging_tree(staging)


def _strict_resume(
    output_dir: Path,
    *,
    expected: Mapping[str, bytes],
) -> None:
    root, paths = _strict_directory(
        output_dir,
        names=OUTPUT_NAMES,
        context="resume temporal-screen output",
    )
    del root
    for name in OUTPUT_NAMES:
        if paths[name].read_bytes() != expected[name]:
            raise RuntimeError(
                f"resume output differs from fresh derivation: {name}"
            )


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def build_candidate_temporal_manifest(
    *,
    expansion_manifest_dir: Path,
    candidate_manifest_dir: Path,
    indexed_graph_dir: Path,
    data_root: Path,
    expected_expansion_artifact_digest: str,
    expected_candidate_artifact_digest: str,
    expected_indexed_artifact_digest: str,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Build a create-only temporal-screen commit or strictly verify it."""

    output_unresolved = output_dir.expanduser()
    if output_unresolved.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output = output_unresolved.resolve(strict=False)
    if resume:
        if not output.exists():
            raise FileNotFoundError(
                "resume is verification-only and requires an output commit"
            )
    elif output.exists():
        raise FileExistsError(
            f"{output} exists; use a fresh path or resume=True"
        )
    protected = [
        expansion_manifest_dir.expanduser().resolve(strict=True),
        candidate_manifest_dir.expanduser().resolve(strict=True),
        indexed_graph_dir.expanduser().resolve(strict=True),
        data_root.expanduser().resolve(strict=True),
    ]
    if any(
        _is_within(output, source) or _is_within(source, output)
        for source in protected
    ):
        raise ValueError("output and an input directory overlap")

    derived = _derive(
        expansion_manifest_dir=expansion_manifest_dir,
        candidate_manifest_dir=candidate_manifest_dir,
        indexed_graph_dir=indexed_graph_dir,
        data_root=data_root,
        expected_expansion_artifact_digest=(
            expected_expansion_artifact_digest
        ),
        expected_candidate_artifact_digest=(
            expected_candidate_artifact_digest
        ),
        expected_indexed_artifact_digest=(
            expected_indexed_artifact_digest
        ),
    )
    payloads = _payloads(derived)

    def final_check() -> None:
        _assert_snapshot(derived.input_paths, derived.input_snapshot)
        _assert_media_identity(
            derived.data_root,
            derived.media_identities,
            expected_root_identity=derived.data_root_identity,
        )

    if resume:
        _strict_resume(output, expected=payloads)
        final_check()
    else:
        _atomic_publish(
            output,
            payloads=payloads,
            pre_publish_check=final_check,
        )
    result = validate_candidate_temporal_manifest(output)
    final_check()
    result["resume_verified"] = bool(resume)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/validate the immutable R7 temporal-screen commit"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "resume"):
        child = commands.add_parser(command)
        child.add_argument("--expansion-manifest-dir", type=Path, required=True)
        child.add_argument("--candidate-manifest-dir", type=Path, required=True)
        child.add_argument("--indexed-graph-dir", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True)
        child.add_argument(
            "--expected-expansion-artifact-digest", required=True
        )
        child.add_argument(
            "--expected-candidate-artifact-digest", required=True
        )
        child.add_argument(
            "--expected-indexed-artifact-digest", required=True
        )
        child.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--input-dir", type=Path, required=True)
    validate.add_argument("--expected-summary-path", type=Path)
    validate.add_argument("--expected-done-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_candidate_temporal_manifest(
            args.input_dir,
            expected_summary_path=args.expected_summary_path,
            expected_done_path=args.expected_done_path,
        )
    else:
        result = build_candidate_temporal_manifest(
            expansion_manifest_dir=args.expansion_manifest_dir,
            candidate_manifest_dir=args.candidate_manifest_dir,
            indexed_graph_dir=args.indexed_graph_dir,
            data_root=args.data_root,
            expected_expansion_artifact_digest=(
                args.expected_expansion_artifact_digest
            ),
            expected_candidate_artifact_digest=(
                args.expected_candidate_artifact_digest
            ),
            expected_indexed_artifact_digest=(
                args.expected_indexed_artifact_digest
            ),
            output_dir=args.output_dir,
            resume=args.command == "resume",
        )
    print(_canonical_bytes(result["done"]).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ASSIGNMENT_SCHEMA",
    "CANDIDATE_POPULATION",
    "DONE_NAME",
    "DONE_SCHEMA",
    "EXPECTED_ANCHOR_ROWS",
    "EXPECTED_INDEXED_ROWS",
    "LABEL_SCHEMA",
    "MANIFEST_NAME",
    "POLICY_VERSION",
    "POSITIVE_CENSUS",
    "ROW_SCHEMA",
    "SAMPLING_SCHEMA",
    "SELECTION_KEY_SCHEMA",
    "SELECTION_SEED",
    "SOURCE_BINDINGS_SCHEMA",
    "SUMMARY_NAME",
    "SUMMARY_SCHEMA",
    "TRUSTED_NEGATIVE_POPULATION",
    "TRUSTED_NEGATIVE_SAMPLE",
    "build_candidate_temporal_manifest",
    "main",
    "validate_candidate_temporal_manifest",
]

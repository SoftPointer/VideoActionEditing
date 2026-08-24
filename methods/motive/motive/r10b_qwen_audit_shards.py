"""Immutable split/merge support for the R10B frame-indexed Qwen audit.

``split`` turns one frozen full queue commit into non-empty queue commits.
Rows are assigned deterministically and are not modified.  ``merge`` validates
every published shard audit with the original audit validator, proves exact
non-overlapping coverage of the full queue, and publishes a normal four-file
full audit commit in the original full-queue order.

This module only changes scheduling granularity.  It grants no representation,
renderer, generation, or training authorization.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .r10b_bernini_pilot_manifest import (
    QUEUE_DONE_NAME,
    QUEUE_NAME,
    QUEUE_SUMMARY_NAME,
    _jsonl_bytes,
    _load_json_object,
    _load_jsonl,
    _load_queue_commit,
    _pretty_bytes,
    write_qwen_audit_queue,
)
from .r10b_family_qwen_audit import (
    ADAPTERS_NAME,
    DONE_NAME,
    DONE_SCHEMA,
    RECORDS_NAME,
    SUMMARY_NAME,
    SUMMARY_SCHEMA,
    _FALSE_AUTHORIZATION,
    file_record,
    validate_published_audit,
)
from .r10b_tangent_core import canonical_json, object_digest


SPLIT_SUMMARY_SCHEMA = "motive-r10b-qwen-audit-shards-v1"
SPLIT_DONE_SCHEMA = "motive-r10b-qwen-audit-shards-done-v1"
SPLIT_SUMMARY_NAME = "shards_summary.json"
SPLIT_DONE_NAME = "shards_done.json"
DEFAULT_SHARD_COUNT = 8
STRATEGIES = ("round_robin", "balanced_contiguous")

_DYNAMIC_QUEUE_SUMMARY_FIELDS = frozenset(
    {
        "rows",
        "unique_components",
        "component_disjoint",
        "screen_cell_counts",
        "candidate_expansion",
        "queue_sha256",
        "full_queue",
        "shard",
    }
)
_DYNAMIC_AUDIT_SUMMARY_FIELDS = frozenset(
    {
        "status",
        "rows",
        "successful_rows",
        "schema_error_rows",
        "generation_error_rows",
        "queue",
        "hard_role_counts",
        "invalid_or_rejected_rows",
        "video_files_read",
        "outputs",
    }
)
_FALSE_GATES = {
    "formal_evidence": False,
    "representation_gate_passed": False,
    "renderer_probe_authorized": False,
    "generation_authorized": False,
    "training_authorized": False,
}


class R10BQwenAuditShardsError(ValueError):
    """A queue shard assignment or merged audit binding is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_queue_bytes(
    queue_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    raw = (Path(queue_dir).expanduser().resolve(strict=True) / QUEUE_NAME).read_bytes()
    expected = _jsonl_bytes(rows)
    if raw != expected:
        raise R10BQwenAuditShardsError(
            "full/shard queue is not canonical JSONL; byte-preserving split "
            "would be impossible"
        )
    return raw


def _queue_binding(
    queue_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    files: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(queue_dir).expanduser().resolve(strict=True)
    return {
        "path": str(root),
        "rows": len(rows),
        "files": dict(files),
        "queue_sha256": files[QUEUE_NAME],
        "summary_sha256": files[QUEUE_SUMMARY_NAME],
    }


def _assignment_indices(
    row_count: int,
    shard_count: int,
    strategy: str,
) -> list[list[int]]:
    if isinstance(shard_count, bool) or not 1 <= shard_count <= row_count:
        raise R10BQwenAuditShardsError(
            "shard_count must be between one and the full queue row count"
        )
    if strategy not in STRATEGIES:
        raise R10BQwenAuditShardsError(
            f"strategy must be one of {', '.join(STRATEGIES)}"
        )
    if strategy == "round_robin":
        assignments = [
            list(range(shard_index, row_count, shard_count))
            for shard_index in range(shard_count)
        ]
    else:
        base, extra = divmod(row_count, shard_count)
        assignments = []
        cursor = 0
        for shard_index in range(shard_count):
            size = base + (1 if shard_index < extra else 0)
            assignments.append(list(range(cursor, cursor + size)))
            cursor += size
    if any(not indices for indices in assignments):
        raise R10BQwenAuditShardsError("every shard must be non-empty")
    return assignments


def _shard_name(index: int, shard_count: int) -> str:
    width = max(3, len(str(shard_count - 1)))
    return f"shard_{index:0{width}d}"


def _write_regular(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_split_tree(
    *,
    output_root: str | Path,
    shard_payloads: Sequence[tuple[str, Mapping[str, Any]]],
    summary_bytes: bytes,
    done_bytes: bytes,
) -> None:
    output = Path(output_root).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        for name, payload in shard_payloads:
            write_qwen_audit_queue(payload, staging / name)
        _write_regular(staging / SPLIT_SUMMARY_NAME, summary_bytes)
        _write_regular(staging / SPLIT_DONE_NAME, done_bytes)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def split_queue(
    *,
    full_queue_dir: str | Path,
    output_root: str | Path,
    shard_count: int = DEFAULT_SHARD_COUNT,
    strategy: str = "round_robin",
) -> dict[str, Any]:
    full_rows, full_summary, full_files = _load_queue_commit(full_queue_dir)
    _canonical_queue_bytes(full_queue_dir, full_rows)
    assignments = _assignment_indices(len(full_rows), shard_count, strategy)
    full_binding = _queue_binding(full_queue_dir, full_rows, full_files)

    shard_payloads: list[tuple[str, Mapping[str, Any]]] = []
    shard_records: list[dict[str, Any]] = []
    for shard_index, indices in enumerate(assignments):
        name = _shard_name(shard_index, shard_count)
        rows = [copy.deepcopy(full_rows[index]) for index in indices]
        queue_bytes = _jsonl_bytes(rows)
        summary = copy.deepcopy(full_summary)
        summary["rows"] = len(rows)
        summary["queue_sha256"] = _sha256_bytes(queue_bytes)
        if "unique_components" in full_summary:
            summary["unique_components"] = len(
                {str(row["component_id"]) for row in rows}
            )
        if "component_disjoint" in full_summary:
            summary["component_disjoint"] = len(
                {str(row["component_id"]) for row in rows}
            ) == len(rows)
        if "screen_cell_counts" in full_summary:
            summary["screen_cell_counts"] = dict(
                sorted(Counter(str(row["screen_cell"]) for row in rows).items())
            )
        if isinstance(full_summary.get("candidate_expansion"), Mapping):
            expansion_rows = [
                row
                for row in rows
                if row.get("candidate_expansion_tier") is not None
            ]
            expansion_summary = copy.deepcopy(
                full_summary["candidate_expansion"]
            )
            expansion_summary["eligible_before_component_dedup"] = len(
                expansion_rows
            )
            expansion_summary["selected_rows"] = len(expansion_rows)
            expansion_summary["selected_cell_counts"] = dict(
                sorted(
                    Counter(
                        str(row["screen_cell"]) for row in expansion_rows
                    ).items()
                )
            )
            expansion_summary["scope"] = "shard_selected_rows"
            summary["candidate_expansion"] = expansion_summary
        summary["full_queue"] = copy.deepcopy(full_binding)
        summary["shard"] = {
            "index": shard_index,
            "count": shard_count,
            "name": name,
            "strategy": strategy,
            "assignment": "full_queue_zero_based_indices",
            "full_queue_indices": indices,
            "full_queue_indices_sha256": object_digest(indices),
            "rows": len(rows),
        }
        shard_payloads.append((name, {"rows": rows, "summary": summary}))
        # ``write_qwen_audit_queue`` deterministically derives these digests.
        summary_bytes = _pretty_bytes(summary)
        shard_files = {
            QUEUE_NAME: _sha256_bytes(queue_bytes),
            QUEUE_SUMMARY_NAME: _sha256_bytes(summary_bytes),
        }
        shard_records.append(
            {
                "index": shard_index,
                "name": name,
                "relative_dir": name,
                "rows": len(rows),
                "strategy": strategy,
                "full_queue_indices": indices,
                "full_queue_indices_sha256": object_digest(indices),
                "queue_files": shard_files,
            }
        )

    top_summary = {
        "schema_version": SPLIT_SUMMARY_SCHEMA,
        "status": "complete",
        "strategy": strategy,
        "assignment": "full_queue_zero_based_indices",
        "shard_count": shard_count,
        "rows": len(full_rows),
        "full_queue": full_binding,
        "shards": shard_records,
        "row_objects_modified": False,
        "formal_evidence": False,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "generation_authorized": False,
        "training_authorized": False,
        "authorization": copy.deepcopy(_FALSE_AUTHORIZATION),
    }
    top_summary_bytes = _pretty_bytes(top_summary)
    top_done = {
        "schema_version": SPLIT_DONE_SCHEMA,
        "status": "complete",
        "files": {
            SPLIT_SUMMARY_NAME: file_record(top_summary_bytes),
        },
        "shards": {
            record["name"]: record["queue_files"] for record in shard_records
        },
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "training_authorized": False,
        "authorization": copy.deepcopy(_FALSE_AUTHORIZATION),
    }
    _publish_split_tree(
        output_root=output_root,
        shard_payloads=shard_payloads,
        summary_bytes=top_summary_bytes,
        done_bytes=_pretty_bytes(top_done),
    )
    return validate_split(output_root, full_queue_dir=full_queue_dir)


def _load_split_commit(
    shard_queues_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], bytes]:
    root = Path(shard_queues_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise R10BQwenAuditShardsError(
            "shard queues root must be a non-symlink directory"
        )
    root = root.resolve(strict=True)
    summary, summary_raw = _load_json_object(
        root / SPLIT_SUMMARY_NAME,
        field="shard split summary",
    )
    done, _done_raw = _load_json_object(
        root / SPLIT_DONE_NAME,
        field="shard split done",
    )
    if (
        summary.get("schema_version") != SPLIT_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or done.get("schema_version") != SPLIT_DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("files")
        != {SPLIT_SUMMARY_NAME: file_record(summary_raw)}
    ):
        raise R10BQwenAuditShardsError("shard split top commit differs")
    return root, summary, done, summary_raw


def validate_split(
    shard_queues_root: str | Path,
    *,
    full_queue_dir: str | Path | None = None,
) -> dict[str, Any]:
    root, summary, done, _summary_raw = _load_split_commit(shard_queues_root)
    bound_path = summary.get("full_queue", {}).get("path")
    if not isinstance(bound_path, str) or not bound_path:
        raise R10BQwenAuditShardsError("full queue path binding differs")
    expected_full = (
        Path(full_queue_dir).expanduser().resolve(strict=True)
        if full_queue_dir is not None
        else Path(bound_path).expanduser().resolve(strict=True)
    )
    if str(expected_full) != bound_path:
        raise R10BQwenAuditShardsError("requested full queue path differs")
    full_rows, full_summary, full_files = _load_queue_commit(expected_full)
    _canonical_queue_bytes(expected_full, full_rows)
    full_binding = _queue_binding(expected_full, full_rows, full_files)
    if summary.get("full_queue") != full_binding:
        raise R10BQwenAuditShardsError("full queue commit binding differs")

    shard_count = summary.get("shard_count")
    strategy = summary.get("strategy")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or strategy not in STRATEGIES
        or summary.get("assignment") != "full_queue_zero_based_indices"
        or summary.get("rows") != len(full_rows)
        or summary.get("row_objects_modified") is not False
        or summary.get("authorization") != _FALSE_AUTHORIZATION
    ):
        raise R10BQwenAuditShardsError("shard split summary contract differs")
    for field, expected in _FALSE_GATES.items():
        if summary.get(field) is not expected:
            raise R10BQwenAuditShardsError(
                f"shard split false gate differs: {field}"
            )
    expected_assignments = _assignment_indices(
        len(full_rows), shard_count, str(strategy)
    )
    records = summary.get("shards")
    if not isinstance(records, list) or len(records) != shard_count:
        raise R10BQwenAuditShardsError("shard record count differs")

    covered: list[int] = []
    expected_done_shards: dict[str, Any] = {}
    expected_names: set[str] = set()
    for shard_index, (record, indices) in enumerate(
        zip(records, expected_assignments)
    ):
        if not isinstance(record, dict):
            raise R10BQwenAuditShardsError("shard record must be an object")
        name = _shard_name(shard_index, shard_count)
        expected_record_base = {
            "index": shard_index,
            "name": name,
            "relative_dir": name,
            "rows": len(indices),
            "strategy": strategy,
            "full_queue_indices": indices,
            "full_queue_indices_sha256": object_digest(indices),
        }
        for key, value in expected_record_base.items():
            if record.get(key) != value:
                raise R10BQwenAuditShardsError(
                    f"{name} top-level shard metadata differs"
                )
        if set(record) != {*expected_record_base, "queue_files"}:
            raise R10BQwenAuditShardsError(
                f"{name} top-level shard fields differ"
            )
        shard_dir = root / name
        shard_rows, shard_summary, shard_files = _load_queue_commit(shard_dir)
        _canonical_queue_bytes(shard_dir, shard_rows)
        expected_rows = [full_rows[index] for index in indices]
        if shard_rows != expected_rows:
            raise R10BQwenAuditShardsError(
                f"{name} rows are not exact full-queue objects"
            )
        if record["queue_files"] != shard_files:
            raise R10BQwenAuditShardsError(f"{name} queue files differ")
        expected_done_shards[name] = shard_files
        expected_names.add(name)

        expected_shard_binding = {
            "index": shard_index,
            "count": shard_count,
            "name": name,
            "strategy": strategy,
            "assignment": "full_queue_zero_based_indices",
            "full_queue_indices": indices,
            "full_queue_indices_sha256": object_digest(indices),
            "rows": len(indices),
        }
        if (
            shard_summary.get("full_queue") != full_binding
            or shard_summary.get("shard") != expected_shard_binding
        ):
            raise R10BQwenAuditShardsError(
                f"{name} full-queue/shard binding differs"
            )
        for key, value in full_summary.items():
            if key not in _DYNAMIC_QUEUE_SUMMARY_FIELDS:
                if shard_summary.get(key) != value:
                    raise R10BQwenAuditShardsError(
                        f"{name} inherited queue contract differs: {key}"
                    )
        if (
            set(shard_summary)
            != set(full_summary) | {"full_queue", "shard"}
        ):
            raise R10BQwenAuditShardsError(
                f"{name} queue summary field set differs"
            )
        covered.extend(indices)

    if sorted(covered) != list(range(len(full_rows))) or len(set(covered)) != len(
        covered
    ):
        raise R10BQwenAuditShardsError(
            "shards do not exactly and disjointly cover the full queue"
        )
    if done.get("shards") != expected_done_shards:
        raise R10BQwenAuditShardsError("shard split done bindings differ")
    for field in (
        "representation_gate_passed",
        "renderer_probe_authorized",
        "training_authorized",
    ):
        if done.get(field) is not False:
            raise R10BQwenAuditShardsError(
                f"shard split done false gate differs: {field}"
            )
    if done.get("authorization") != _FALSE_AUTHORIZATION:
        raise R10BQwenAuditShardsError(
            "shard split done authorization differs"
        )
    expected_root_files = {SPLIT_SUMMARY_NAME, SPLIT_DONE_NAME}
    entries = list(root.iterdir())
    if {path.name for path in entries} != expected_root_files | expected_names:
        raise R10BQwenAuditShardsError(
            "shard split root closure differs"
        )
    for path in entries:
        if path.name in expected_root_files:
            valid_type = path.is_file()
        else:
            valid_type = path.is_dir()
        if path.is_symlink() or not valid_type:
            raise R10BQwenAuditShardsError(
                "shard split root entries must be regular non-symlink "
                "files or directories of the committed type"
            )
    return {
        "status": "VALID",
        "shard_queues_root": str(root),
        "full_queue_dir": str(expected_full),
        "rows": len(full_rows),
        "shard_count": shard_count,
        "strategy": strategy,
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "training_authorized": False,
    }


def _load_audit_files(
    audit_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, _record_raw = _load_jsonl(
        audit_dir / RECORDS_NAME,
        field=f"{audit_dir.name} Qwen records",
    )
    adapters, _adapter_raw = _load_jsonl(
        audit_dir / ADAPTERS_NAME,
        field=f"{audit_dir.name} Qwen adapters",
    )
    summary, _summary_raw = _load_json_object(
        audit_dir / SUMMARY_NAME,
        field=f"{audit_dir.name} Qwen summary",
    )
    return records, adapters, summary


def _assert_same_static_audit_contract(
    first: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    shard_name: str,
) -> None:
    if set(first) != set(current):
        raise R10BQwenAuditShardsError(
            f"{shard_name} audit summary fields differ"
        )
    for key, value in first.items():
        if key not in _DYNAMIC_AUDIT_SUMMARY_FIELDS and current.get(key) != value:
            raise R10BQwenAuditShardsError(
                f"{shard_name} audit contract differs: {key}"
            )


def merge_audits(
    *,
    full_queue_dir: str | Path,
    shard_queues_root: str | Path,
    shard_audits_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    validate_split(
        shard_queues_root,
        full_queue_dir=full_queue_dir,
    )
    split_root, split_summary, _split_done, _raw = _load_split_commit(
        shard_queues_root
    )
    full_rows, _full_summary, full_files = _load_queue_commit(full_queue_dir)
    audit_root = Path(shard_audits_root).expanduser()
    if audit_root.is_symlink() or not audit_root.is_dir():
        raise R10BQwenAuditShardsError(
            "shard audits root must be a non-symlink directory"
        )
    audit_root = audit_root.resolve(strict=True)
    expected_audit_dirs = {
        str(record["name"]) for record in split_summary["shards"]
    }
    observed_audit_dirs = {
        path.name
        for path in audit_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    if (
        observed_audit_dirs != expected_audit_dirs
        or any(
            path.is_symlink() or not path.is_dir()
            for path in audit_root.iterdir()
        )
    ):
        raise R10BQwenAuditShardsError(
            "shard audit root closure differs"
        )

    by_iid: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    first_summary: dict[str, Any] | None = None
    for shard_record in split_summary["shards"]:
        name = str(shard_record["name"])
        queue_dir = split_root / name
        audit_dir = audit_root / name
        # This validates schemas, derivations, model inventory, implementation,
        # queue bindings, gates, outputs, and done digests before any merge.
        validate_published_audit(audit_dir)
        shard_rows, _shard_summary, shard_files = _load_queue_commit(queue_dir)
        records, adapters, audit_summary = _load_audit_files(audit_dir)
        queue_binding = audit_summary.get("queue", {})
        if (
            Path(str(queue_binding.get("path", ""))).expanduser().resolve(
                strict=True
            )
            != queue_dir.resolve(strict=True)
            or queue_binding.get("files") != shard_files
            or queue_binding.get("rows") != len(shard_rows)
        ):
            raise R10BQwenAuditShardsError(
                f"{name} audit does not bind the declared shard queue"
            )
        if first_summary is None:
            first_summary = copy.deepcopy(audit_summary)
        else:
            _assert_same_static_audit_contract(
                first_summary,
                audit_summary,
                shard_name=name,
            )
        if len(records) != len(shard_rows) or len(adapters) != len(shard_rows):
            raise R10BQwenAuditShardsError(f"{name} audit row count differs")
        for queue_row, record, adapter in zip(shard_rows, records, adapters):
            iid = str(queue_row["iid"])
            if iid in by_iid:
                raise R10BQwenAuditShardsError(
                    f"duplicate IID across shard audits: {iid}"
                )
            if record.get("iid") != iid or adapter.get("iid") != iid:
                raise R10BQwenAuditShardsError(
                    f"{name} audit order differs from its queue"
                )
            by_iid[iid] = (record, adapter)

    if first_summary is None:  # pragma: no cover - split forbids zero shards
        raise R10BQwenAuditShardsError("no shard audits were found")
    full_iids = [str(row["iid"]) for row in full_rows]
    if set(by_iid) != set(full_iids) or len(by_iid) != len(full_iids):
        raise R10BQwenAuditShardsError(
            "shard audits do not exactly cover full queue IIDs"
        )
    records = [by_iid[iid][0] for iid in full_iids]
    adapters = [by_iid[iid][1] for iid in full_iids]
    record_bytes = _jsonl_bytes(records)
    adapter_bytes = _jsonl_bytes(adapters)
    role_counts = Counter(
        str(record["hard_classification"]["role"]) for record in records
    )
    outcome_counts = Counter(
        str(record["audit_outcome"]) for record in records
    )
    merged_status = (
        "partial_generation_failure"
        if outcome_counts["generation_error"]
        else "complete"
    )

    merged_summary = copy.deepcopy(first_summary)
    merged_summary.update(
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": merged_status,
            "rows": len(records),
            "successful_rows": outcome_counts["success"],
            "schema_error_rows": outcome_counts["schema_error"],
            "generation_error_rows": outcome_counts["generation_error"],
            "queue": {
                "path": str(
                    Path(full_queue_dir).expanduser().resolve(strict=True)
                ),
                "files": full_files,
                "rows": len(full_rows),
            },
            "hard_role_counts": dict(sorted(role_counts.items())),
            "invalid_or_rejected_rows": role_counts.get("reject", 0),
            "video_files_read": 2 * len(records),
            "outputs": {
                RECORDS_NAME: {
                    "rows": len(records),
                    **file_record(record_bytes),
                },
                ADAPTERS_NAME: {
                    "rows": len(adapters),
                    **file_record(adapter_bytes),
                },
            },
            **_FALSE_GATES,
            "authorization": copy.deepcopy(_FALSE_AUTHORIZATION),
        }
    )
    summary_bytes = _pretty_bytes(merged_summary)
    done = {
        "schema_version": DONE_SCHEMA,
        "status": merged_status,
        "rows": len(records),
        "successful_rows": outcome_counts["success"],
        "schema_error_rows": outcome_counts["schema_error"],
        "generation_error_rows": outcome_counts["generation_error"],
        "files": {
            RECORDS_NAME: file_record(record_bytes),
            ADAPTERS_NAME: file_record(adapter_bytes),
            SUMMARY_NAME: file_record(summary_bytes),
        },
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "training_authorized": False,
    }
    from .r10b_bernini_pilot_manifest import _atomic_directory

    _atomic_directory(
        output_dir,
        {
            RECORDS_NAME: record_bytes,
            ADAPTERS_NAME: adapter_bytes,
            SUMMARY_NAME: summary_bytes,
            DONE_NAME: _pretty_bytes(done),
        },
    )
    return validate_published_audit(output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split or merge immutable R10B Qwen audit commits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--full-queue-dir", type=Path, required=True)
    split.add_argument("--output-root", type=Path, required=True)
    split.add_argument("--shard-count", type=int, default=DEFAULT_SHARD_COUNT)
    split.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="round_robin",
    )

    validate_split_parser = subparsers.add_parser("validate-split")
    validate_split_parser.add_argument("--shard-queues-root", type=Path, required=True)
    validate_split_parser.add_argument("--full-queue-dir", type=Path)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--full-queue-dir", type=Path, required=True)
    merge.add_argument("--shard-queues-root", type=Path, required=True)
    merge.add_argument("--shard-audits-root", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    validate_merge = subparsers.add_parser("validate-merge")
    validate_merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "split":
        result = split_queue(
            full_queue_dir=args.full_queue_dir,
            output_root=args.output_root,
            shard_count=args.shard_count,
            strategy=args.strategy,
        )
    elif args.command == "validate-split":
        result = validate_split(
            args.shard_queues_root,
            full_queue_dir=args.full_queue_dir,
        )
    elif args.command == "merge":
        result = merge_audits(
            full_queue_dir=args.full_queue_dir,
            shard_queues_root=args.shard_queues_root,
            shard_audits_root=args.shard_audits_root,
            output_dir=args.output_dir,
        )
        result.pop("adapters", None)
    else:
        result = validate_published_audit(args.output_dir)
        result.pop("adapters", None)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

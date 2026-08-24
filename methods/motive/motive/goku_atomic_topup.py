"""Create-only, resumable batching for the Goku atomic-action pipeline.

The Qwen planner and atomic-label auditor are deliberately fallible gates.  A
smoke test therefore must not mean "the first eight candidates all pass", and
a 1,000-row build must not assume that processing the first 1,000 candidates
produces 1,000 labels.  This module freezes contiguous batches from an ordered
parent manifest, binds cumulative planner/atomic progress, and selects the
first *N final passes* in parent order.

All published files use O_EXCL on first creation.  Resume only accepts exact
byte matches; it never rewrites a receipt or changes candidate order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence


SELECTION_SCHEMA = "motive-goku-atomic-topup-selection-v1"
PROGRESS_SCHEMA = "motive-goku-atomic-topup-progress-v1"
GATE_SCHEMA = "motive-goku-atomic-topup-gate-v1"
ATOMIC_SUMMARY_SCHEMA = "motive-goku-atomic-motion-verify-summary-v1"
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA_RE = re.compile(r"[0-9a-f]{64}")


class AtomicTopupError(RuntimeError):
    """A top-up artifact is missing, mutable, or inconsistent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_digest(value: Mapping[str, Any], *, omit: str) -> str:
    payload = dict(value)
    payload.pop(omit, None)
    return _sha_bytes(_canonical(payload))


def _plain_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _plain_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and not path.is_symlink()
    except FileNotFoundError:
        return False


def _publish(path: Path, raw: bytes, *, resume: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        if not resume or not _plain_file(path) or path.read_bytes() != raw:
            raise AtomicTopupError(f"existing artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    if not _plain_file(path):
        raise AtomicTopupError(f"not a plain JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AtomicTopupError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not _plain_file(path):
        raise AtomicTopupError(f"not a plain JSONL file: {path}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise AtomicTopupError(f"JSONL is not newline terminated: {path}")
    if not raw and not allow_empty:
        raise AtomicTopupError(f"JSONL is empty: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AtomicTopupError(f"JSONL row {number} is not an object: {path}")
        rows.append(value)
    return rows


def _candidate_closure(
    path: Path, *, expected_count: int, expected_sha256: str
) -> tuple[list[dict[str, Any]], list[bytes], list[str]]:
    if _SHA_RE.fullmatch(expected_sha256) is None:
        raise AtomicTopupError("candidate SHA-256 is malformed")
    if not _plain_file(path) or _sha_file(path) != expected_sha256:
        raise AtomicTopupError("candidate manifest digest differs")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise AtomicTopupError("candidate manifest is not closed JSONL")
    lines = raw.splitlines(keepends=True)
    if len(lines) != expected_count:
        raise AtomicTopupError(
            f"candidate count={len(lines)} expected={expected_count}"
        )
    rows: list[dict[str, Any]] = []
    iids: list[str] = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise AtomicTopupError(f"candidate row {index} is not an object")
        iid = row.get("iid")
        if not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None:
            raise AtomicTopupError(f"candidate row {index} has unsafe IID")
        rows.append(row)
        iids.append(iid)
    if len(iids) != len(set(iids)):
        raise AtomicTopupError("candidate IIDs are duplicated")
    return rows, lines, iids


def _validate_selection(value: Mapping[str, Any], *, path: Path | None = None) -> None:
    required = {
        "schema_version",
        "batch_index",
        "stage",
        "candidate_manifest",
        "candidate_manifest_sha256",
        "candidate_count",
        "start_index",
        "end_index_exclusive",
        "batch_rows",
        "batch_manifest",
        "batch_manifest_sha256",
        "batch_iids",
        "batch_iid_order_sha256",
        "tail_merged_to_preserve_worker_floor",
        "selection_digest",
    }
    if set(value) != required or value.get("schema_version") != SELECTION_SCHEMA:
        raise AtomicTopupError("selection receipt schema differs")
    if value.get("selection_digest") != _object_digest(value, omit="selection_digest"):
        raise AtomicTopupError("selection receipt digest differs")
    start = value.get("start_index")
    end = value.get("end_index_exclusive")
    rows = value.get("batch_rows")
    iids = value.get("batch_iids")
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not isinstance(rows, int)
        or not 0 <= start < end <= value.get("candidate_count", -1)
        or rows != end - start
        or not isinstance(iids, list)
        or len(iids) != rows
        or any(not isinstance(iid, str) or _IID_RE.fullmatch(iid) is None for iid in iids)
        or len(iids) != len(set(iids))
    ):
        raise AtomicTopupError("selection receipt range/IIDs differ")
    batch = Path(str(value.get("batch_manifest", "")))
    if not batch.is_absolute() or not _plain_file(batch):
        raise AtomicTopupError("selection batch manifest is unsafe")
    if _sha_file(batch) != value.get("batch_manifest_sha256"):
        raise AtomicTopupError("selection batch manifest digest differs")
    observed = [row.get("iid") for row in _read_jsonl(batch)]
    if observed != iids:
        raise AtomicTopupError("selection batch IID order differs")
    expected_order_sha = _sha_bytes(("\n".join(iids) + "\n").encode("utf-8"))
    if value.get("batch_iid_order_sha256") != expected_order_sha:
        raise AtomicTopupError("selection batch IID-order digest differs")
    if path is not None and not _plain_file(path):
        raise AtomicTopupError("selection receipt path is unsafe")


def select_batch(args: argparse.Namespace) -> int:
    candidates = args.candidates.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    rows, lines, iids = _candidate_closure(
        candidates,
        expected_count=args.expected_candidates,
        expected_sha256=args.expected_candidates_sha256,
    )
    del rows
    if args.batch_index < 0 or args.start_index < 0 or args.batch_size < 1:
        raise AtomicTopupError("batch index/start/size is invalid")
    if args.minimum_workers < 1 or args.batch_size < args.minimum_workers:
        raise AtomicTopupError("batch size is below worker floor")
    if args.start_index >= len(lines):
        raise AtomicTopupError("candidate pool is already exhausted")
    end = min(len(lines), args.start_index + args.batch_size)
    tail_merged = 0 < len(lines) - end < args.minimum_workers
    if tail_merged:
        end = len(lines)
    if output_dir.exists() or output_dir.is_symlink():
        if not args.resume or not _plain_dir(output_dir):
            raise AtomicTopupError(f"existing batch directory differs: {output_dir}")
    else:
        output_dir.mkdir(mode=0o700)
    batch_path = output_dir / "planner_input.jsonl"
    receipt_path = output_dir / "selection_receipt.json"
    payload = b"".join(lines[args.start_index:end])
    _publish(batch_path, payload, resume=args.resume)
    batch_iids = iids[args.start_index:end]
    receipt: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "batch_index": args.batch_index,
        "stage": args.stage,
        "candidate_manifest": str(candidates),
        "candidate_manifest_sha256": args.expected_candidates_sha256,
        "candidate_count": len(lines),
        "start_index": args.start_index,
        "end_index_exclusive": end,
        "batch_rows": end - args.start_index,
        "batch_manifest": str(batch_path),
        "batch_manifest_sha256": _sha_bytes(payload),
        "batch_iids": batch_iids,
        "batch_iid_order_sha256": _sha_bytes(
            ("\n".join(batch_iids) + "\n").encode("utf-8")
        ),
        "tail_merged_to_preserve_worker_floor": tail_merged,
        "selection_digest": None,
    }
    receipt["selection_digest"] = _object_digest(receipt, omit="selection_digest")
    _publish(receipt_path, _pretty(receipt), resume=args.resume)
    _validate_selection(receipt, path=receipt_path)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def empty_atomic(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    manifest = args.manifest_output.expanduser().resolve()
    summary = args.summary_output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve(strict=True)
    if not _plain_dir(output_root):
        raise AtomicTopupError("atomic output root is unsafe")
    _publish(input_path, b"", resume=args.resume)
    _publish(manifest, b"", resume=args.resume)
    value: dict[str, Any] = {
        "schema_version": ATOMIC_SUMMARY_SCHEMA,
        "input_path": str(input_path),
        "input_sha256": _sha_bytes(b""),
        "output_root": str(output_root),
        "expected_rows": 0,
        "terminal_rows": 0,
        "ok_rows": 0,
        "error_rows": 0,
        "dataset_manifest_path": str(manifest),
        "dataset_manifest_sha256": _sha_bytes(b""),
        "summary_digest": None,
    }
    value["summary_digest"] = _object_digest(value, omit="summary_digest")
    _publish(summary, _pretty(value), resume=args.resume)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


def _validate_progress(value: Mapping[str, Any], *, path: Path | None = None) -> None:
    required = {
        "schema_version",
        "batch_index",
        "stage",
        "selection_receipt",
        "selection_receipt_sha256",
        "candidate_manifest",
        "candidate_manifest_sha256",
        "candidate_count",
        "consumed_rows",
        "remaining_rows",
        "planner_ok_rows",
        "planner_error_rows",
        "atomic_terminal_rows",
        "atomic_ok_rows",
        "atomic_error_rows",
        "target_atomic_ok",
        "status",
        "planner_receipt",
        "planner_receipt_sha256",
        "atomic_manifest",
        "atomic_manifest_sha256",
        "atomic_summary",
        "atomic_summary_sha256",
        "previous_progress",
        "previous_progress_sha256",
        "progress_digest",
    }
    if set(value) != required or value.get("schema_version") != PROGRESS_SCHEMA:
        raise AtomicTopupError("progress receipt schema differs")
    if value.get("progress_digest") != _object_digest(value, omit="progress_digest"):
        raise AtomicTopupError("progress receipt digest differs")
    if value.get("status") not in {"continue", "target_reached", "pool_exhausted"}:
        raise AtomicTopupError("progress status differs")
    if path is not None and not _plain_file(path):
        raise AtomicTopupError("progress receipt path is unsafe")


def publish_progress(args: argparse.Namespace) -> int:
    candidates = args.candidates.expanduser().resolve(strict=True)
    _, parent_lines, parent_iids = _candidate_closure(
        candidates,
        expected_count=args.expected_candidates,
        expected_sha256=args.expected_candidates_sha256,
    )
    selection_path = args.selection.expanduser().resolve(strict=True)
    selection = _read_json(selection_path)
    _validate_selection(selection, path=selection_path)
    if (
        selection["candidate_manifest"] != str(candidates)
        or selection["candidate_manifest_sha256"] != args.expected_candidates_sha256
        or selection["candidate_count"] != args.expected_candidates
    ):
        raise AtomicTopupError("selection is bound to another candidate manifest")
    expected_batch_bytes = b"".join(
        parent_lines[selection["start_index"] : selection["end_index_exclusive"]]
    )
    if Path(selection["batch_manifest"]).read_bytes() != expected_batch_bytes:
        raise AtomicTopupError("selection batch bytes differ from the parent slice")

    previous: dict[str, Any] | None = None
    previous_path: Path | None = None
    if args.previous_progress is not None:
        previous_path = args.previous_progress.expanduser().resolve(strict=True)
        previous = _read_json(previous_path)
        _validate_progress(previous, path=previous_path)
        if (
            previous["candidate_manifest"] != str(candidates)
            or previous["candidate_manifest_sha256"]
            != args.expected_candidates_sha256
            or previous["candidate_count"] != args.expected_candidates
        ):
            raise AtomicTopupError("previous progress belongs to another candidate pool")
    expected_start = 0 if previous is None else previous["consumed_rows"]
    expected_batch = 0 if previous is None else previous["batch_index"] + 1
    if (
        selection["start_index"] != expected_start
        or selection["batch_index"] != expected_batch
    ):
        raise AtomicTopupError("selection does not continue the previous prefix")

    planner_path = args.planner_receipt.expanduser().resolve(strict=True)
    planner = _read_json(planner_path)
    planner_required = {
        "schema_version",
        "input",
        "input_sha256",
        "expected_rows",
        "ok_rows",
        "error_rows",
        "minimum_ok",
        "records",
    }
    if set(planner) != planner_required:
        raise AtomicTopupError("planner phase receipt schema differs")
    if (
        planner["input"] != str(Path(selection["batch_manifest"]).resolve())
        or planner["input_sha256"] != selection["batch_manifest_sha256"]
        or planner["expected_rows"] != selection["batch_rows"]
        or planner["ok_rows"] + planner["error_rows"] != selection["batch_rows"]
        or planner["minimum_ok"] != 0
    ):
        raise AtomicTopupError("planner phase receipt binding differs")
    records = planner.get("records")
    if (
        not isinstance(records, list)
        or [row.get("iid") for row in records] != selection["batch_iids"]
        or any(row.get("status") not in {"ok", "error"} for row in records)
        or sum(row.get("status") == "ok" for row in records) != planner["ok_rows"]
        or sum(row.get("status") == "error" for row in records)
        != planner["error_rows"]
    ):
        raise AtomicTopupError("planner phase record order/status differs")

    prior_planner_ok = 0 if previous is None else previous["planner_ok_rows"]
    prior_planner_error = 0 if previous is None else previous["planner_error_rows"]
    planner_ok = prior_planner_ok + planner["ok_rows"]
    planner_error = prior_planner_error + planner["error_rows"]

    atomic_manifest_path = args.atomic_manifest.expanduser().resolve(strict=True)
    atomic_summary_path = args.atomic_summary.expanduser().resolve(strict=True)
    atomic_rows = _read_jsonl(atomic_manifest_path, allow_empty=True)
    atomic_summary = _read_json(atomic_summary_path)
    if atomic_summary.get("schema_version") != ATOMIC_SUMMARY_SCHEMA:
        raise AtomicTopupError("atomic summary schema differs")
    if atomic_summary.get("summary_digest") != _object_digest(
        atomic_summary, omit="summary_digest"
    ):
        raise AtomicTopupError("atomic summary digest differs")
    if (
        atomic_summary.get("expected_rows") != planner_ok
        or atomic_summary.get("terminal_rows") != planner_ok
        or atomic_summary.get("ok_rows") + atomic_summary.get("error_rows") != planner_ok
        or atomic_summary.get("ok_rows") != len(atomic_rows)
        or atomic_summary.get("dataset_manifest_path") != str(atomic_manifest_path)
        or atomic_summary.get("dataset_manifest_sha256") != _sha_file(atomic_manifest_path)
    ):
        raise AtomicTopupError("atomic cumulative summary binding differs")
    atomic_iids = [row.get("iid") for row in atomic_rows]
    parent_rank = {iid: index for index, iid in enumerate(parent_iids)}
    if (
        any(iid not in parent_rank for iid in atomic_iids)
        or len(atomic_iids) != len(set(atomic_iids))
        or atomic_iids != sorted(atomic_iids, key=parent_rank.__getitem__)
        or any(parent_rank[iid] >= selection["end_index_exclusive"] for iid in atomic_iids)
        or any(
            row.get("original_candidate_index") != parent_rank[row["iid"]]
            for row in atomic_rows
        )
    ):
        raise AtomicTopupError("atomic manifest is not the consumed parent-order subset")
    if previous is not None:
        prior_manifest = Path(previous["atomic_manifest"])
        if not _plain_file(prior_manifest) or _sha_file(prior_manifest) != previous["atomic_manifest_sha256"]:
            raise AtomicTopupError("previous atomic manifest binding differs")
        prior_iids = [row.get("iid") for row in _read_jsonl(prior_manifest, allow_empty=True)]
        prior_lines = prior_manifest.read_bytes().splitlines(keepends=True)
        current_lines = atomic_manifest_path.read_bytes().splitlines(keepends=True)
        if (
            atomic_iids[: len(prior_iids)] != prior_iids
            or current_lines[: len(prior_lines)] != prior_lines
        ):
            raise AtomicTopupError("cumulative atomic pass prefix regressed or mutated")
        if atomic_summary["ok_rows"] < previous["atomic_ok_rows"]:
            raise AtomicTopupError("cumulative atomic pass count regressed")

    consumed = selection["end_index_exclusive"]
    if atomic_summary["ok_rows"] >= args.target_atomic_ok:
        status_value = "target_reached"
    elif consumed == args.expected_candidates:
        status_value = "pool_exhausted"
    else:
        status_value = "continue"
    value: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA,
        "batch_index": selection["batch_index"],
        "stage": selection["stage"],
        "selection_receipt": str(selection_path),
        "selection_receipt_sha256": _sha_file(selection_path),
        "candidate_manifest": str(candidates),
        "candidate_manifest_sha256": args.expected_candidates_sha256,
        "candidate_count": args.expected_candidates,
        "consumed_rows": consumed,
        "remaining_rows": args.expected_candidates - consumed,
        "planner_ok_rows": planner_ok,
        "planner_error_rows": planner_error,
        "atomic_terminal_rows": atomic_summary["terminal_rows"],
        "atomic_ok_rows": atomic_summary["ok_rows"],
        "atomic_error_rows": atomic_summary["error_rows"],
        "target_atomic_ok": args.target_atomic_ok,
        "status": status_value,
        "planner_receipt": str(planner_path),
        "planner_receipt_sha256": _sha_file(planner_path),
        "atomic_manifest": str(atomic_manifest_path),
        "atomic_manifest_sha256": _sha_file(atomic_manifest_path),
        "atomic_summary": str(atomic_summary_path),
        "atomic_summary_sha256": _sha_file(atomic_summary_path),
        "previous_progress": str(previous_path) if previous_path else None,
        "previous_progress_sha256": _sha_file(previous_path) if previous_path else None,
        "progress_digest": None,
    }
    value["progress_digest"] = _object_digest(value, omit="progress_digest")
    output = args.output.expanduser().resolve()
    _publish(output, _pretty(value), resume=args.resume)
    _validate_progress(value, path=output)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def publish_gate(args: argparse.Namespace) -> int:
    candidates = args.candidates.expanduser().resolve(strict=True)
    _, _, parent_iids = _candidate_closure(
        candidates,
        expected_count=args.expected_candidates,
        expected_sha256=args.expected_candidates_sha256,
    )
    progress_path = args.progress.expanduser().resolve(strict=True)
    progress = _read_json(progress_path)
    _validate_progress(progress, path=progress_path)
    source_manifest = args.atomic_manifest.expanduser().resolve(strict=True)
    if (
        progress["candidate_manifest"] != str(candidates)
        or progress["candidate_manifest_sha256"]
        != args.expected_candidates_sha256
        or progress["candidate_count"] != args.expected_candidates
        or progress["atomic_manifest"] != str(source_manifest)
        or progress["atomic_manifest_sha256"] != _sha_file(source_manifest)
        or progress["atomic_ok_rows"] < args.target_ok
    ):
        raise AtomicTopupError("progress does not prove enough atomic passes")
    rows = _read_jsonl(source_manifest)
    rank = {iid: index for index, iid in enumerate(parent_iids)}
    iids = [row.get("iid") for row in rows]
    if iids != sorted(iids, key=rank.__getitem__):
        raise AtomicTopupError("atomic source manifest is not parent ordered")
    selected_rows = rows[: args.target_ok]
    payload = b"".join(_canonical(row) + b"\n" for row in selected_rows)
    output_manifest = args.output_manifest.expanduser().resolve()
    _publish(output_manifest, payload, resume=args.resume)
    selected_iids = [row["iid"] for row in selected_rows]
    value: dict[str, Any] = {
        "schema_version": GATE_SCHEMA,
        "status": "pass",
        "selection_policy": "first_final_atomic_passes_in_parent_candidate_order",
        "target_ok": args.target_ok,
        "selected_iids": selected_iids,
        "selected_parent_indices": [rank[iid] for iid in selected_iids],
        "candidate_manifest": str(candidates),
        "candidate_manifest_sha256": args.expected_candidates_sha256,
        "source_atomic_manifest": str(source_manifest),
        "source_atomic_manifest_sha256": _sha_file(source_manifest),
        "progress_receipt": str(progress_path),
        "progress_receipt_sha256": _sha_file(progress_path),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": _sha_bytes(payload),
        "execution_provenance": {
            "snapshot": args.snapshot,
            "snapshot_tree_sha256": args.snapshot_tree_sha256,
            "model": args.model,
            "planner_module": args.planner_module,
            "atomic_module": args.atomic_module,
        },
        "gate_digest": None,
    }
    value["gate_digest"] = _object_digest(value, omit="gate_digest")
    output_receipt = args.output_receipt.expanduser().resolve()
    _publish(output_receipt, _pretty(value), resume=args.resume)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-batch")
    select.add_argument("--candidates", type=Path, required=True)
    select.add_argument("--expected-candidates", type=int, required=True)
    select.add_argument("--expected-candidates-sha256", required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--batch-index", type=int, required=True)
    select.add_argument("--start-index", type=int, required=True)
    select.add_argument("--batch-size", type=int, required=True)
    select.add_argument("--minimum-workers", type=int, default=1)
    select.add_argument("--stage", choices=("smoke", "full"), required=True)
    select.add_argument("--resume", action="store_true")
    select.set_defaults(func=select_batch)

    empty = commands.add_parser("empty-atomic")
    empty.add_argument("--input", type=Path, required=True)
    empty.add_argument("--output-root", type=Path, required=True)
    empty.add_argument("--manifest-output", type=Path, required=True)
    empty.add_argument("--summary-output", type=Path, required=True)
    empty.add_argument("--resume", action="store_true")
    empty.set_defaults(func=empty_atomic)

    progress = commands.add_parser("publish-progress")
    progress.add_argument("--candidates", type=Path, required=True)
    progress.add_argument("--expected-candidates", type=int, required=True)
    progress.add_argument("--expected-candidates-sha256", required=True)
    progress.add_argument("--selection", type=Path, required=True)
    progress.add_argument("--planner-receipt", type=Path, required=True)
    progress.add_argument("--atomic-manifest", type=Path, required=True)
    progress.add_argument("--atomic-summary", type=Path, required=True)
    progress.add_argument("--target-atomic-ok", type=int, required=True)
    progress.add_argument("--previous-progress", type=Path)
    progress.add_argument("--output", type=Path, required=True)
    progress.add_argument("--resume", action="store_true")
    progress.set_defaults(func=publish_progress)

    gate = commands.add_parser("publish-gate")
    gate.add_argument("--candidates", type=Path, required=True)
    gate.add_argument("--expected-candidates", type=int, required=True)
    gate.add_argument("--expected-candidates-sha256", required=True)
    gate.add_argument("--atomic-manifest", type=Path, required=True)
    gate.add_argument("--progress", type=Path, required=True)
    gate.add_argument("--target-ok", type=int, required=True)
    gate.add_argument("--output-manifest", type=Path, required=True)
    gate.add_argument("--output-receipt", type=Path, required=True)
    gate.add_argument("--snapshot")
    gate.add_argument("--snapshot-tree-sha256")
    gate.add_argument("--model")
    gate.add_argument("--planner-module")
    gate.add_argument("--atomic-module")
    gate.add_argument("--resume", action="store_true")
    gate.set_defaults(func=publish_gate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for field in ("expected_candidates",):
        if hasattr(args, field) and getattr(args, field) < 1:
            raise AtomicTopupError(f"--{field.replace('_', '-')} must be positive")
    if hasattr(args, "target_atomic_ok") and args.target_atomic_ok < 1:
        raise AtomicTopupError("--target-atomic-ok must be positive")
    if hasattr(args, "target_ok") and args.target_ok < 1:
        raise AtomicTopupError("--target-ok must be positive")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except AtomicTopupError as error:
        print(f"[atomic-topup] ERROR: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)

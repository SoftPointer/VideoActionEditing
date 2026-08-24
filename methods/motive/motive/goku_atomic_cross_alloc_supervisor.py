"""Fail-closed cross-allocation supervision for Goku atomic round 2.

The GPU pipeline deliberately binds every run to one Slurm allocation.  A
different job therefore cannot resume the same run root.  This supervisor
uses a safer recovery unit: an immutable *attempt*.  If an epoch allocation
ends without a closed epoch manifest, a retry attempt receives a new run root
and a byte-preserving candidate manifest with every already committed Wan IID
removed.  Completed epoch roots are never modified.

This module is deliberately audit/plan-only.  It never writes a retry
manifest and never invokes ``sbatch``, ``scontrol update`` or ``scancel``.
Execution is left to a separately reviewed deployer consuming the emitted
hash-bound plan.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "motive-goku-atomic-cross-allocation-config-v1"
STATE_SCHEMA = "motive-goku-atomic-cross-allocation-state-v1"
RETRY_SCHEMA = "motive-goku-atomic-cross-allocation-retry-v1"
PLAN_SCHEMA = "motive-goku-atomic-cross-allocation-plan-v1"
EPOCH_SUMMARY_SCHEMA = "motive-goku-atomic1000-dataset-summary-v1"
ROW_SCHEMA = "motive-goku-atomic1000-dataset-row-v1"
WAN_SAMPLE_SCHEMA = "motive-wan22-i2v-sample-v1"

ACTIVE_STATES = frozenset(
    {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
)
TERMINAL_FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
JOB_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")


class CrossAllocationError(RuntimeError):
    """A frozen input, recovery decision, or state transition is unsafe."""


@dataclass(frozen=True)
class Epoch:
    index: int
    target: int
    lane: int
    predecessor: int | None
    initial_job_id: int
    run_root: Path
    selected: Path
    selected_sha256: str


@dataclass(frozen=True)
class Attempt:
    attempt: int
    job_id: int
    run_root: Path
    selected: Path
    selected_sha256: str


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
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


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_digest(value: Mapping[str, Any], field: str) -> str:
    copy = dict(value)
    copy[field] = None
    return _sha_bytes(_canonical(copy))


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CrossAllocationError(f"{context} is not an absolute plain file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite constant: {item}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise CrossAllocationError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise CrossAllocationError(f"{context} is not one JSON object")
    return value


def _sha(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CrossAllocationError(f"{context} is not a lowercase SHA-256")
    return value


def _absolute(value: Any, *, context: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise CrossAllocationError(f"{context} is not an absolute path")
    return Path(value)


def _plain_directory(path: Path, *, context: str, required: bool = True) -> Path:
    exists = path.exists() or path.is_symlink()
    if required and (not exists or path.is_symlink() or not path.is_dir()):
        raise CrossAllocationError(f"{context} is not a plain directory: {path}")
    if exists and (path.is_symlink() or not path.is_dir()):
        raise CrossAllocationError(f"unsafe {context}: {path}")
    return path


def load_config(path: Path) -> tuple[dict[str, Any], list[Epoch], str]:
    path = path.expanduser().resolve(strict=True)
    value = _load_json(path, context="cross-allocation config")
    required = {
        "schema_version",
        "status",
        "global_target",
        "state_root",
        "recovery_root",
        "retry_holder",
        "retry_holder_sha256",
        "merge_job_id",
        "slurm",
        "epochs",
        "config_digest",
    }
    if set(value) != required:
        raise CrossAllocationError("cross-allocation config schema is open")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["status"] != "frozen"
        or value["global_target"] != 1000
        or value["config_digest"] != _object_digest(value, "config_digest")
    ):
        raise CrossAllocationError("cross-allocation config identity/digest differs")
    state_root = _absolute(value["state_root"], context="state_root")
    recovery_root = _absolute(value["recovery_root"], context="recovery_root")
    if state_root == recovery_root:
        raise CrossAllocationError("state and recovery roots must differ")
    holder = _absolute(value["retry_holder"], context="retry holder")
    holder_sha = _sha(value["retry_holder_sha256"], context="retry holder SHA")
    if holder.exists():
        if holder.is_symlink() or not holder.is_file() or _sha_file(holder) != holder_sha:
            raise CrossAllocationError("retry holder bytes differ")
    slurm = value.get("slurm")
    if not isinstance(slurm, dict) or set(slurm) != {"sacct", "sbatch", "scontrol", "squeue"}:
        raise CrossAllocationError("Slurm command config differs")
    for name, command in slurm.items():
        _absolute(command, context=f"Slurm {name}")
    if type(value.get("merge_job_id")) is not int or value["merge_job_id"] <= 0:
        raise CrossAllocationError("merge job ID differs")
    raw_epochs = value.get("epochs")
    if not isinstance(raw_epochs, list) or len(raw_epochs) != 8:
        raise CrossAllocationError("exactly eight epoch records are required")
    epochs: list[Epoch] = []
    for position, raw in enumerate(raw_epochs, 1):
        keys = {
            "index",
            "target",
            "lane",
            "predecessor_epoch",
            "initial_job_id",
            "run_root",
            "selected_manifest",
            "selected_manifest_sha256",
        }
        if not isinstance(raw, dict) or set(raw) != keys:
            raise CrossAllocationError(f"epoch {position} config schema differs")
        expected_target = 128 if position < 8 else 104
        expected_predecessor = position - 2 if position > 2 else None
        if (
            raw["index"] != position
            or raw["target"] != expected_target
            or raw["lane"] != (position - 1) % 2
            or raw["predecessor_epoch"] != expected_predecessor
            or type(raw["initial_job_id"]) is not int
            or raw["initial_job_id"] <= 0
        ):
            raise CrossAllocationError(f"epoch {position} lane/target/job differs")
        epochs.append(
            Epoch(
                index=position,
                target=expected_target,
                lane=raw["lane"],
                predecessor=expected_predecessor,
                initial_job_id=raw["initial_job_id"],
                run_root=_absolute(raw["run_root"], context=f"epoch {position} run root"),
                selected=_absolute(
                    raw["selected_manifest"], context=f"epoch {position} selected"
                ),
                selected_sha256=_sha(
                    raw["selected_manifest_sha256"],
                    context=f"epoch {position} selected SHA",
                ),
            )
        )
    if sum(item.target for item in epochs) != value["global_target"]:
        raise CrossAllocationError("epoch targets do not sum to exact1000")
    if len({item.initial_job_id for item in epochs}) != len(epochs):
        raise CrossAllocationError("initial job IDs overlap")
    return value, epochs, _sha_file(path)


def _manifest_lines(path: Path, expected_sha: str | None = None) -> list[bytes]:
    if path.is_symlink() or not path.is_file():
        raise CrossAllocationError(f"manifest is not a plain file: {path}")
    raw = path.read_bytes()
    if expected_sha is not None and _sha_bytes(raw) != expected_sha:
        raise CrossAllocationError(f"manifest SHA differs: {path}")
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") or not line[:-1] for line in lines):
        raise CrossAllocationError(f"manifest JSONL framing differs: {path}")
    return lines


def validate_epoch_complete(epoch: Epoch, run_root: Path) -> bool:
    """Return False only when final artifacts are wholly absent.

    Once either final artifact exists, any mismatch is an error rather than a
    reason to overwrite or retry that root.
    """

    manifest = run_root / "atomic1000_dataset_manifest.jsonl"
    summary_path = run_root / "atomic1000_dataset_summary.json"
    present = [path.exists() or path.is_symlink() for path in (manifest, summary_path)]
    if not any(present):
        return False
    if not all(present):
        raise CrossAllocationError(f"epoch {epoch.index} has partial final publication")
    lines = _manifest_lines(manifest)
    if len(lines) != epoch.target:
        raise CrossAllocationError(f"epoch {epoch.index} final row count differs")
    manifest_raw = b"".join(lines)
    summary = _load_json(summary_path, context=f"epoch {epoch.index} summary")
    expected_summary_keys = {
        "schema_version",
        "status",
        "minimum_success",
        "total_rows",
        "new_wan_rows",
        "legacy_reused_rows",
        "manifest_sha256",
        "primary_training_label_field",
        "wan_generation_prompt_is_separate",
    }
    if (
        set(summary) != expected_summary_keys
        or summary.get("schema_version") != EPOCH_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("minimum_success") != epoch.target
        or summary.get("total_rows") != epoch.target
        or summary.get("new_wan_rows") != epoch.target
        or summary.get("legacy_reused_rows") != 0
        or summary.get("manifest_sha256") != _sha_bytes(manifest_raw)
        or summary.get("primary_training_label_field")
        != "atomic_action_instruction"
        or summary.get("wan_generation_prompt_is_separate") is not True
    ):
        raise CrossAllocationError(f"epoch {epoch.index} final summary differs")
    iids: set[str] = set()
    groups: set[str] = set()
    artifact_fields = (
        ("source_video", "source_video_sha256"),
        ("target_video", "target_video_sha256"),
        ("strict_target_frame0_float32_npy", "strict_target_frame0_float32_npy_sha256"),
        ("strict_target_frame0_png", "strict_target_frame0_png_sha256"),
        ("strict_source_frame0_anchor_png", "strict_source_frame0_anchor_png_sha256"),
        ("atomic_result", "atomic_result_sha256"),
        ("planner_passed", "planner_passed_sha256"),
        ("wan_result", "wan_result_sha256"),
        ("sample_metadata", "sample_metadata_sha256"),
    )
    for number, line in enumerate(lines, 1):
        row = json.loads(line, object_pairs_hook=_strict_pairs)
        iid = row.get("iid") if isinstance(row, dict) else None
        if (
            not isinstance(iid, str)
            or IID_RE.fullmatch(iid) is None
            or iid in iids
            or row.get("schema_version") != ROW_SCHEMA
            or row.get("lineage") != "atomic_new_wan"
        ):
            raise CrossAllocationError(f"epoch {epoch.index} row {number} identity differs")
        planner_path: Path | None = None
        for path_field, sha_field in artifact_fields:
            path = _absolute(row.get(path_field), context=f"row {iid} {path_field}")
            expected = _sha(row.get(sha_field), context=f"row {iid} {sha_field}")
            if path.is_symlink() or not path.is_file() or _sha_file(path) != expected:
                raise CrossAllocationError(f"row {iid} artifact differs: {path_field}")
            if path_field == "planner_passed":
                planner_path = path
        assert planner_path is not None
        planner_lines = _manifest_lines(planner_path)
        if len(planner_lines) != 1:
            raise CrossAllocationError(f"row {iid} planner fragment differs")
        planner = json.loads(planner_lines[0], object_pairs_hook=_strict_pairs)
        group = planner.get("group_id") if isinstance(planner, dict) else None
        if not isinstance(group, str) or not group or group in groups:
            raise CrossAllocationError(f"row {iid} group lineage differs")
        iids.add(iid)
        groups.add(group)
    return True


def _validate_committed_sample(result_path: Path) -> str:
    result = _load_json(result_path, context="Wan committed sample result")
    iid = result.get("iid")
    if (
        not isinstance(iid, str)
        or IID_RE.fullmatch(iid) is None
        or result.get("schema_version") != WAN_SAMPLE_SCHEMA
    ):
        raise CrossAllocationError(f"Wan committed sample identity differs: {result_path}")
    if result_path.parent.name != iid:
        raise CrossAllocationError(f"Wan committed sample directory differs iid={iid}")
    digest = _sha(result.get("result_digest"), context=f"Wan result digest iid={iid}")
    unsigned = dict(result)
    unsigned.pop("result_digest")
    if digest != _sha_bytes(_canonical(unsigned)):
        raise CrossAllocationError(f"Wan committed sample digest differs iid={iid}")
    outputs = result.get("outputs")
    if not isinstance(outputs, dict):
        raise CrossAllocationError(f"Wan committed outputs missing iid={iid}")
    for name_field, sha_field in (
        ("source_video", "source_video_sha256"),
        ("preview_mp4", "preview_mp4_sha256"),
        ("edit_instruction_file", "edit_instruction_file_sha256"),
    ):
        name = outputs.get(name_field)
        if not isinstance(name, str) or Path(name).name != name:
            raise CrossAllocationError(f"Wan output name differs iid={iid} field={name_field}")
        path = result_path.parent / name
        expected = _sha(outputs.get(sha_field), context=f"Wan {sha_field} iid={iid}")
        if path.is_symlink() or not path.is_file() or _sha_file(path) != expected:
            raise CrossAllocationError(f"Wan output bytes differ iid={iid} field={name_field}")
    return iid


def committed_iids(run_roots: Sequence[Path]) -> list[str]:
    """Return every hash-validated sample commit across prior attempts."""

    found: dict[str, Path] = {}
    for root in run_roots:
        if not root.exists():
            continue
        _plain_directory(root, context="prior attempt run root")
        batch_root = root / "wan_atomic" / "batches"
        if not batch_root.exists():
            continue
        _plain_directory(batch_root, context="prior Wan batch root")
        pattern = "batch_*/samples/*/samples/*/result.json"
        for path in sorted(batch_root.glob(pattern)):
            iid = _validate_committed_sample(path)
            previous = found.get(iid)
            if previous is not None and previous != path:
                raise CrossAllocationError(
                    f"duplicate committed IID across attempts iid={iid}: {previous} {path}"
                )
            found[iid] = path
    return sorted(found)


def materialize_retry(
    *,
    epoch: Epoch,
    attempt: int,
    prior_run_roots: Sequence[Path],
    recovery_root: Path,
) -> dict[str, Any]:
    """Create one retry manifest excluding every prior committed video IID."""

    if attempt < 1:
        raise CrossAllocationError("retry attempt must be positive")
    lines = _manifest_lines(epoch.selected, epoch.selected_sha256)
    parsed = [json.loads(line, object_pairs_hook=_strict_pairs) for line in lines]
    parent_iids = [row.get("iid") for row in parsed]
    if (
        any(not isinstance(iid, str) or IID_RE.fullmatch(iid) is None for iid in parent_iids)
        or len(parent_iids) != len(set(parent_iids))
    ):
        raise CrossAllocationError(f"epoch {epoch.index} parent IID closure differs")
    excluded = committed_iids(prior_run_roots)
    unknown = sorted(set(excluded) - set(parent_iids))
    if unknown:
        raise CrossAllocationError(f"committed IIDs absent from epoch candidates: {unknown[:4]}")
    excluded_set = set(excluded)
    selected_raw = b"".join(
        line for line, iid in zip(lines, parent_iids) if iid not in excluded_set
    )
    remaining = len(selected_raw.splitlines())
    if remaining < max(epoch.target, 64):
        raise CrossAllocationError(
            f"epoch {epoch.index} retry has only {remaining} candidates for target {epoch.target}"
        )
    attempt_root = recovery_root / f"epoch_{epoch.index:04d}" / f"attempt_{attempt:03d}"
    if attempt_root.exists() or attempt_root.is_symlink():
        raise CrossAllocationError(f"create-only retry attempt exists: {attempt_root}")
    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{attempt_root.name}.stage-", dir=attempt_root.parent))
    try:
        selected = stage / "selected.jsonl"
        selected.write_bytes(selected_raw)
        os.chmod(selected, 0o400)
        excluded_payload = ("\n".join(excluded) + ("\n" if excluded else "")).encode()
        excluded_path = stage / "excluded_committed_iids.txt"
        excluded_path.write_bytes(excluded_payload)
        os.chmod(excluded_path, 0o400)
        final_root = attempt_root
        receipt: dict[str, Any] = {
            "schema_version": RETRY_SCHEMA,
            "status": "prepared",
            "epoch_index": epoch.index,
            "epoch_target": epoch.target,
            "attempt": attempt,
            "original_selected": str(epoch.selected),
            "original_selected_sha256": epoch.selected_sha256,
            "original_rows": len(lines),
            "prior_run_roots": [str(path) for path in prior_run_roots],
            "excluded_committed_iids": excluded,
            "excluded_committed_iids_sha256": _sha_bytes(excluded_payload),
            "excluded_committed_count": len(excluded),
            "selected_manifest": str(final_root / "selected.jsonl"),
            "selected_manifest_sha256": _sha_bytes(selected_raw),
            "selected_rows": remaining,
            "run_root": str(final_root / "run"),
            "receipt_digest": None,
        }
        receipt["receipt_digest"] = _object_digest(receipt, "receipt_digest")
        receipt_path = stage / "retry_receipt.json"
        receipt_path.write_bytes(_pretty(receipt))
        os.chmod(receipt_path, 0o400)
        os.replace(stage, attempt_root)
        directory = os.open(attempt_root.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
    return receipt


def _normalize_state(value: str) -> str:
    return value.strip().split("+", 1)[0].split()[0].upper() if value.strip() else "MISSING"


def query_job(job_id: int, slurm: Mapping[str, str]) -> dict[str, Any]:
    squeue = subprocess.run(
        [slurm["squeue"], "-h", "-j", str(job_id), "-o", "%T|%r"],
        check=False,
        capture_output=True,
        text=True,
    )
    if squeue.returncode != 0:
        raise CrossAllocationError(
            f"squeue failed for job {job_id}: {squeue.stderr.strip()}"
        )
    lines = [line.strip() for line in squeue.stdout.splitlines() if line.strip()]
    state = "MISSING"
    reason = ""
    if len(lines) > 1:
        raise CrossAllocationError(f"ambiguous squeue rows for job {job_id}")
    if lines:
        state, _, reason = lines[0].partition("|")
    else:
        sacct = subprocess.run(
            [
                slurm["sacct"],
                "-n",
                "-P",
                "-j",
                str(job_id),
                "--format=JobIDRaw,State,ExitCode",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if sacct.returncode != 0:
            raise CrossAllocationError(
                f"sacct failed for job {job_id}: {sacct.stderr.strip()}"
            )
        rows = [row.split("|") for row in sacct.stdout.splitlines() if row.strip()]
        exact = [row for row in rows if row and row[0] == str(job_id)]
        if len(exact) > 1:
            raise CrossAllocationError(f"ambiguous sacct rows for job {job_id}")
        if exact:
            state = exact[0][1]
            reason = exact[0][2] if len(exact[0]) > 2 else ""
    dependency = None
    record = subprocess.run(
        [slurm["scontrol"], "show", "job", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if record.returncode != 0:
        raise CrossAllocationError(
            f"scontrol show failed for job {job_id}: {record.stderr.strip()}"
        )
    match = re.search(r"(?:^|\s)Dependency=(\S+)", record.stdout)
    if match:
        dependency = match.group(1)
    return {
        "job_id": job_id,
        "state": _normalize_state(state),
        "reason": reason,
        "dependency": dependency,
    }


def _initial_attempt(epoch: Epoch) -> Attempt:
    return Attempt(0, epoch.initial_job_id, epoch.run_root, epoch.selected, epoch.selected_sha256)


def _load_latest_state(
    config: Mapping[str, Any], epochs: Sequence[Epoch], config_sha: str
) -> tuple[dict[str, Any], Path | None]:
    root = Path(config["state_root"])
    if not root.exists():
        attempts = {
            str(epoch.index): {
                "attempt": 0,
                "job_id": epoch.initial_job_id,
                "run_root": str(epoch.run_root),
                "selected_manifest": str(epoch.selected),
                "selected_manifest_sha256": epoch.selected_sha256,
            }
            for epoch in epochs
        }
        return {"sequence": 0, "attempts": attempts}, None
    _plain_directory(root, context="state root")
    receipts = sorted((root / "receipts").glob("state_[0-9][0-9][0-9][0-9][0-9][0-9].json")) if (root / "receipts").exists() else []
    if not receipts:
        raise CrossAllocationError("state root exists without a state receipt")
    prior_path: Path | None = None
    prior_sha: str | None = None
    latest: dict[str, Any] | None = None
    for sequence, path in enumerate(receipts, 1):
        value = _load_json(path, context="state receipt")
        if (
            value.get("schema_version") != STATE_SCHEMA
            or value.get("sequence") != sequence
            or value.get("config_sha256") != config_sha
            or value.get("previous_receipt") != (str(prior_path) if prior_path else None)
            or value.get("previous_receipt_sha256") != prior_sha
            or value.get("receipt_digest") != _object_digest(value, "receipt_digest")
        ):
            raise CrossAllocationError(f"state receipt chain differs: {path}")
        prior_path = path
        prior_sha = _sha_file(path)
        latest = value
    assert latest is not None
    return latest, prior_path


def _attempt_from_state(epoch: Epoch, state: Mapping[str, Any]) -> Attempt:
    attempts = state.get("attempts")
    raw = attempts.get(str(epoch.index)) if isinstance(attempts, dict) else None
    expected = {
        "attempt",
        "job_id",
        "run_root",
        "selected_manifest",
        "selected_manifest_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise CrossAllocationError(f"state lacks epoch {epoch.index}")
    if (
        type(raw["attempt"]) is not int
        or raw["attempt"] < 0
        or type(raw["job_id"]) is not int
        or raw["job_id"] <= 0
    ):
        raise CrossAllocationError(f"state epoch {epoch.index} attempt/job differs")
    return Attempt(
        raw["attempt"],
        raw["job_id"],
        _absolute(raw["run_root"], context="state run root"),
        _absolute(raw["selected_manifest"], context="state selected"),
        _sha(raw["selected_manifest_sha256"], context="state selected SHA"),
    )


def build_plan(
    config: Mapping[str, Any],
    epochs: Sequence[Epoch],
    state: Mapping[str, Any],
    observations: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []
    current: dict[int, Attempt] = {}
    complete: dict[int, bool] = {}
    for epoch in epochs:
        attempt = _attempt_from_state(epoch, state)
        current[epoch.index] = attempt
        is_complete = validate_epoch_complete(epoch, attempt.run_root)
        complete[epoch.index] = is_complete
        observation = observations.get(attempt.job_id)
        if observation is None:
            raise CrossAllocationError(f"missing observation for active job {attempt.job_id}")
        statuses.append(
            {
                "epoch_index": epoch.index,
                "lane": epoch.lane,
                "attempt": attempt.attempt,
                "job_id": attempt.job_id,
                "run_root": str(attempt.run_root),
                "complete": is_complete,
                "slurm_state": observation["state"],
                "slurm_reason": observation.get("reason", ""),
            }
        )
    action: dict[str, Any] = {"kind": "none", "reason": "all epochs active or complete"}
    for epoch in epochs:
        if complete[epoch.index]:
            continue
        attempt = current[epoch.index]
        observation = observations[attempt.job_id]
        state_name = _normalize_state(str(observation["state"]))
        predecessor = epoch.predecessor
        if predecessor is not None and state_name == "PENDING":
            predecessor_attempt = current[predecessor]
            initial_predecessor_job = epochs[predecessor - 1].initial_job_id
            desired_dependency = f"afterok:{predecessor_attempt.job_id}"
            observed_dependency = observation.get("dependency")
            if (
                predecessor_attempt.job_id != initial_predecessor_job
                and observed_dependency != desired_dependency
            ):
                action = {
                    "kind": "rebind_dependency",
                    "epoch_index": epoch.index,
                    "job_id": attempt.job_id,
                    "predecessor_epoch": predecessor,
                    "predecessor_job_id": predecessor_attempt.job_id,
                    "current_dependency": observed_dependency,
                    "desired_dependency": desired_dependency,
                    "execution_enabled": False,
                }
                break
        if predecessor is not None and not complete[predecessor]:
            predecessor_attempt = current[predecessor]
            predecessor_state = _normalize_state(
                str(observations[predecessor_attempt.job_id]["state"])
            )
            if state_name in ACTIVE_STATES or predecessor_state in ACTIVE_STATES:
                continue
        if state_name in ACTIVE_STATES:
            continue
        if state_name == "COMPLETED" or state_name in TERMINAL_FAILURE_STATES or state_name == "MISSING":
            action = {
                "kind": "prepare_submit_retry",
                "epoch_index": epoch.index,
                "failed_job_id": attempt.job_id,
                "failed_state": state_name,
                "next_attempt": attempt.attempt + 1,
                "new_attempt_root": str(
                    Path(config["recovery_root"])
                    / f"epoch_{epoch.index:04d}"
                    / f"attempt_{attempt.attempt + 1:03d}"
                ),
                "recovery_semantics": "fresh_job_fresh_run_root",
                "candidate_rule": "byte_preserving_parent_minus_hash_validated_committed_iids",
                "same_run_root_resume_forbidden": True,
                "downstream_dependency_rebind_required": epoch.index <= 6,
                "merge_redeployment_required": True,
                "execution_enabled": False,
            }
            break
        raise CrossAllocationError(
            f"unsupported Slurm state epoch={epoch.index} job={attempt.job_id}: {state_name}"
        )
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "dry_run_plan",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_target": config["global_target"],
        "epochs": statuses,
        "action": action,
        "merge_job_id": config["merge_job_id"],
        "merge_handled_separately": True,
        "mutations_performed": False,
    }


def _publish_state(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    prior: Mapping[str, Any],
    prior_path: Path | None,
    attempts: Mapping[str, Any],
    event: Mapping[str, Any],
) -> Path:
    root = Path(config["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CrossAllocationError("state root is unsafe")
    receipts = root / "receipts"
    receipts.mkdir(exist_ok=True)
    sequence = int(prior.get("sequence", 0)) + 1
    target = receipts / f"state_{sequence:06d}.json"
    value: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "sequence": sequence,
        "config_sha256": config_sha,
        "previous_receipt": str(prior_path) if prior_path else None,
        "previous_receipt_sha256": _sha_file(prior_path) if prior_path else None,
        "attempts": dict(attempts),
        "event": dict(event),
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "receipt_digest": None,
    }
    value["receipt_digest"] = _object_digest(value, "receipt_digest")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_pretty(value))
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _run_checked(argv: Sequence[str]) -> str:
    process = subprocess.run(argv, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        raise CrossAllocationError(
            f"command failed rc={process.returncode}: {argv!r}: {process.stderr.strip()}"
        )
    return process.stdout.strip()


def apply_action(
    *,
    config: Mapping[str, Any],
    config_sha: str,
    epochs: Sequence[Epoch],
    state: Mapping[str, Any],
    state_path: Path | None,
    plan: Mapping[str, Any],
) -> Path | None:
    action = plan["action"]
    if action["kind"] == "none":
        return None
    attempts = json.loads(json.dumps(state["attempts"]))
    if action["kind"] == "rebind_dependency":
        epoch = epochs[int(action["epoch_index"]) - 1]
        current = _attempt_from_state(epoch, state)
        record = query_job(current.job_id, config["slurm"])
        if record["state"] != "PENDING" or validate_epoch_complete(epoch, current.run_root):
            raise CrossAllocationError("downstream job changed before dependency rebind")
        dependency = f"afterok:{int(action['predecessor_job_id'])}"
        _run_checked(
            [
                config["slurm"]["scontrol"],
                "update",
                f"JobId={current.job_id}",
                f"Dependency={dependency}",
            ]
        )
        event = {**action, "dependency": dependency}
        return _publish_state(
            config=config,
            config_sha=config_sha,
            prior=state,
            prior_path=state_path,
            attempts=attempts,
            event=event,
        )
    if action["kind"] != "prepare_submit_retry":
        raise CrossAllocationError(f"unknown action: {action['kind']}")
    epoch = epochs[int(action["epoch_index"]) - 1]
    current = _attempt_from_state(epoch, state)
    fresh = query_job(current.job_id, config["slurm"])
    if fresh["state"] in ACTIVE_STATES or validate_epoch_complete(epoch, current.run_root):
        raise CrossAllocationError("epoch changed before retry submission")
    prior_roots = [epoch.run_root]
    # Every prior attempt uses deterministic recovery paths.
    for number in range(1, current.attempt + 1):
        prior_roots.append(
            Path(config["recovery_root"])
            / f"epoch_{epoch.index:04d}"
            / f"attempt_{number:03d}"
            / "run"
        )
    retry = materialize_retry(
        epoch=epoch,
        attempt=current.attempt + 1,
        prior_run_roots=prior_roots,
        recovery_root=Path(config["recovery_root"]),
    )
    attempt_root = Path(config["recovery_root"]) / f"epoch_{epoch.index:04d}" / f"attempt_{current.attempt + 1:03d}"
    receipt_path = attempt_root / "retry_receipt.json"
    holder = Path(config["retry_holder"])
    if holder.is_symlink() or not holder.is_file() or _sha_file(holder) != config["retry_holder_sha256"]:
        raise CrossAllocationError("retry holder bytes changed before submission")
    job_name = f"goku-r2-e{epoch.index:02d}-r{current.attempt + 1:02d}"
    if JOB_NAME_RE.fullmatch(job_name) is None:
        raise CrossAllocationError("retry job name is unsafe")
    log_root = attempt_root / "slurm_logs"
    log_root.mkdir()
    export_values = {
        "EPOCH_INDEX": epoch.index,
        "EPOCH_TARGET": epoch.target,
        "EPOCH_ATTEMPT": current.attempt + 1,
        "EPOCH_EXPECTED_ROWS": retry["selected_rows"],
        "EPOCH_SELECTED_MANIFEST": retry["selected_manifest"],
        "EPOCH_SELECTED_SHA256": retry["selected_manifest_sha256"],
        "EPOCH_RUN_ROOT": retry["run_root"],
        "EPOCH_RETRY_RECEIPT": str(receipt_path),
        "EPOCH_RETRY_RECEIPT_SHA256": _sha_file(receipt_path),
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
    response = _run_checked(
        [
            config["slurm"]["sbatch"],
            "--parsable",
            f"--job-name={job_name}",
            f"--output={log_root}/slurm-%x-%j.out",
            f"--error={log_root}/slurm-%x-%j.err",
            f"--export={export}",
            str(holder),
        ]
    )
    job_token = response.split(";", 1)[0]
    if not job_token.isdigit() or int(job_token) <= 0:
        raise CrossAllocationError(f"sbatch returned invalid job ID: {response!r}")
    new_job = int(job_token)
    attempts[str(epoch.index)] = {
        "attempt": current.attempt + 1,
        "job_id": new_job,
        "run_root": retry["run_root"],
        "selected_manifest": retry["selected_manifest"],
        "selected_manifest_sha256": retry["selected_manifest_sha256"],
    }
    event = {
        **action,
        "new_job_id": new_job,
        "retry_receipt": str(receipt_path),
        "retry_receipt_sha256": _sha_file(receipt_path),
        "merge_redeployment_required": True,
    }
    return _publish_state(
        config=config,
        config_sha=config_sha,
        prior=state,
        prior_path=state_path,
        attempts=attempts,
        event=event,
    )


def _load_observations(path: Path) -> dict[int, dict[str, Any]]:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    value = _load_json(absolute, context="job observations")
    rows = value.get("jobs")
    if not isinstance(rows, list):
        raise CrossAllocationError("observation jobs differ")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("job_id")) is not int:
            raise CrossAllocationError("observation row differs")
        result[row["job_id"]] = row
    return result


def run_once(args: argparse.Namespace) -> int:
    config, epochs, config_sha = load_config(args.config)
    state, _state_path = _load_latest_state(config, epochs, config_sha)
    attempts = [_attempt_from_state(epoch, state) for epoch in epochs]
    if args.observations:
        observations = _load_observations(args.observations)
    else:
        observations = {
            attempt.job_id: query_job(attempt.job_id, config["slurm"])
            for attempt in attempts
        }
    plan = build_plan(config, epochs, state, observations)
    plan["config_sha256"] = config_sha
    plan["apply_requested"] = False
    print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    return 0


def freeze_config(args: argparse.Namespace) -> int:
    epochs_root = args.epochs_root.expanduser().resolve(strict=True)
    generation_root = args.generation_root.expanduser().resolve(strict=False)
    jobs = [int(value) for value in args.job_ids.split(":")]
    if jobs != [135096, 135151, 135152, 135153, 135154, 135155, 135156, 135157]:
        raise CrossAllocationError("round2 initial job IDs differ")
    holder = args.retry_holder.expanduser().resolve(strict=True)
    epochs: list[dict[str, Any]] = []
    for index, job_id in enumerate(jobs, 1):
        selected = (epochs_root / f"epoch_{index:04d}" / "selected.jsonl").resolve(strict=True)
        epochs.append(
            {
                "index": index,
                "target": 128 if index < 8 else 104,
                "lane": (index - 1) % 2,
                "predecessor_epoch": index - 2 if index > 2 else None,
                "initial_job_id": job_id,
                "run_root": str(generation_root / f"epoch_{index:04d}"),
                "selected_manifest": str(selected),
                "selected_manifest_sha256": _sha_file(selected),
            }
        )
    value: dict[str, Any] = {
        "schema_version": CONFIG_SCHEMA,
        "status": "frozen",
        "global_target": 1000,
        "state_root": str(args.state_root.expanduser().resolve(strict=False)),
        "recovery_root": str(args.recovery_root.expanduser().resolve(strict=False)),
        "retry_holder": str(holder),
        "retry_holder_sha256": _sha_file(holder),
        "merge_job_id": args.merge_job_id,
        "slurm": {
            "sacct": str(args.sacct),
            "sbatch": str(args.sbatch),
            "scontrol": str(args.scontrol),
            "squeue": str(args.squeue),
        },
        "epochs": epochs,
        "config_digest": None,
    }
    value["config_digest"] = _object_digest(value, "config_digest")
    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_pretty(value))
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"config": str(output), "sha256": _sha_file(output)}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-once")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--observations", type=Path)
    run.set_defaults(func=run_once)
    freeze = commands.add_parser("freeze-config")
    freeze.add_argument("--epochs-root", type=Path, required=True)
    freeze.add_argument("--generation-root", type=Path, required=True)
    freeze.add_argument("--state-root", type=Path, required=True)
    freeze.add_argument("--recovery-root", type=Path, required=True)
    freeze.add_argument("--retry-holder", type=Path, required=True)
    freeze.add_argument("--merge-job-id", type=int, default=135161)
    freeze.add_argument(
        "--job-ids", default="135096:135151:135152:135153:135154:135155:135156:135157"
    )
    freeze.add_argument("--sacct", type=Path, default=Path("/usr/bin/sacct"))
    freeze.add_argument("--sbatch", type=Path, default=Path("/usr/bin/sbatch"))
    freeze.add_argument("--scontrol", type=Path, default=Path("/usr/bin/scontrol"))
    freeze.add_argument("--squeue", type=Path, default=Path("/usr/bin/squeue"))
    freeze.add_argument("--output", type=Path, required=True)
    freeze.set_defaults(func=freeze_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CrossAllocationError as error:
        raise SystemExit(f"goku cross-allocation supervisor failed: {error}") from error


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

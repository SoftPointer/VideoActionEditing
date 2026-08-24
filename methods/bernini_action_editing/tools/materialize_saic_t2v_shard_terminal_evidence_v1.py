#!/usr/bin/env python3
"""Bind a completed SAIC T2V shard to its deep audit and Slurm terminal row.

This is deliberately a technical shard receipt.  It does not admit a full event
bank, detached human review, training, optimization, or scientific selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, NoReturn


SCHEMA = "saic-t2v-shard-terminal-evidence-v1"
STATUS = "terminal_technical_shard_evidence_not_full_bank_admission"
SACCT_FIELDS = (
    "JobIDRaw,JobName,State,ExitCode,Elapsed,AllocNodes,AllocTRES,SubmitLine"
)


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_digest", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256_bytes(encoded)


def load_receipt(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        die(f"cannot read {label}: {error}")
    if not isinstance(value, dict):
        die(f"{label} is not an object")
    if value.get("receipt_digest") != canonical_digest(value):
        die(f"{label} canonical digest differs")
    return value, sha256_bytes(raw)


def exact_sacct(job_id: str, sacct: Path) -> dict[str, Any]:
    if not sacct.is_file() or sacct.is_symlink():
        die("sacct executable is not one regular non-symlink file")
    command = [
        str(sacct), "-j", job_id, "-X", "--noheader", "-n", "-P",
        "-o", SACCT_FIELDS,
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0 or completed.stderr:
        die("sacct observation failed or wrote stderr")
    try:
        stdout = completed.stdout.decode("ascii")
    except UnicodeDecodeError as error:
        die(f"sacct output is not ASCII: {error}")
    rows = [row for row in stdout.splitlines() if row]
    if len(rows) != 1:
        die("sacct did not return exactly one allocation row")
    fields = rows[0].split("|", 7)
    if len(fields) != 8:
        die("sacct allocation row field count differs")
    keys = SACCT_FIELDS.split(",")
    parsed = dict(zip(keys, fields))
    if (
        parsed["JobIDRaw"] != job_id
        or parsed["State"] != "COMPLETED"
        or parsed["ExitCode"] != "0:0"
        or parsed["AllocNodes"] != "1"
    ):
        die("Slurm allocation is not exact single-node terminal success")
    alloc_tres = set(parsed["AllocTRES"].split(","))
    if "gres/gpu:mi210=4" not in alloc_tres or "node=1" not in alloc_tres:
        die("Slurm allocation is not the registered 1x4 MI210 shard topology")
    return {
        "command": command,
        "executable": str(sacct),
        "executable_sha256": sha256_file(sacct),
        "stdout_sha256": sha256_bytes(completed.stdout),
        "stderr_sha256": sha256_bytes(completed.stderr),
        "parsed_row": parsed,
    }


def validate_deep_audit(
    audit: dict[str, Any], *, root: Path, group_id: str, expected_count: int
) -> list[dict[str, Any]]:
    indices = list(range(expected_count))
    rows = audit.get("rows")
    if (
        audit.get("root") != str(root)
        or audit.get("group_id") != group_id
        or audit.get("planned_candidate_count") != expected_count
        or audit.get("completed_prefix_count") != expected_count
        or audit.get("completed_candidate_indices") != indices
        or audit.get("deep_generation_receipt_validation") is not True
        or audit.get("deep_rendezvous_completion_validation") is not True
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        die("deep audit does not close the exact requested shard")
    authority = audit.get("authority")
    if not isinstance(authority, dict) or any(value is not False for value in authority.values()):
        die("deep audit authority boundary differs")
    if [row.get("candidate_index") for row in rows] != indices:
        die("deep audit row indices differ")
    mp4_hashes = [row.get("mp4_sha256") for row in rows]
    if (
        any(not isinstance(value, str) or len(value) != 64 for value in mp4_hashes)
        or len(set(mp4_hashes)) != expected_count
    ):
        die("deep audit does not bind one unique MP4 per candidate")
    return rows


def validate_partial_receipt(
    partial: dict[str, Any], audit: dict[str, Any], audit_rows: list[dict[str, Any]],
    *, group_id: str, expected_count: int,
) -> None:
    rows = partial.get("rows")
    if (
        partial.get("group_id") != group_id
        or partial.get("candidate_count") != expected_count
        or partial.get("root_spec_raw_sha256") != audit.get("root_spec_raw_sha256")
        or not isinstance(rows, list)
        or len(rows) != expected_count
    ):
        die("partial shard receipt differs from the deep audit")
    authority = partial.get("authority")
    if not isinstance(authority, dict) or any(value is not False for value in authority.values()):
        die("partial shard authority boundary differs")
    for index, (row, audit_row) in enumerate(zip(rows, audit_rows)):
        if (
            row.get("candidate_index") != index
            or row.get("candidate_id") != audit_row.get("candidate_id")
            or row.get("attempt_receipt_sha256")
            != audit_row.get("attempt_receipt_sha256")
            or row.get("completion_receipt_sha256")
            != audit_row.get("completion_receipt_sha256")
        ):
            die(f"partial receipt row {index} differs from the deep audit")


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != payload:
        die("terminal evidence reread differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deep-audit", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, default=Path("/usr/bin/sacct"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.job_id.isdigit() or args.expected_count <= 0:
        die("job id or expected count differs")
    root = args.output_root.resolve(strict=True)
    audit, audit_sha = load_receipt(args.deep_audit, "deep audit")
    partial_path = root / f"saic-pure-t2v-event-bank-topup-partial-{args.group_id}-receipt.json"
    partial, partial_sha = load_receipt(partial_path, "partial shard receipt")
    audit_rows = validate_deep_audit(
        audit, root=root, group_id=args.group_id, expected_count=args.expected_count
    )
    validate_partial_receipt(
        partial, audit, audit_rows,
        group_id=args.group_id, expected_count=args.expected_count,
    )
    sacct = exact_sacct(args.job_id, args.sacct)
    submit_line = sacct["parsed_row"]["SubmitLine"]
    required_submit_tokens = (
        f"SAIC_T2V_V3_OUTPUT_ROOT={root}",
        f"SAIC_T2V_V3_GROUP_SELECT={args.group_id}",
        "--gres=gpu:mi210:4",
    )
    if any(token not in submit_line for token in required_submit_tokens):
        die("terminal SubmitLine is not bound to the audited shard")
    for path, label in ((args.stdout, "stdout"), (args.stderr, "stderr")):
        if not path.is_file() or path.is_symlink():
            die(f"{label} is not one regular non-symlink file")
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "job_id": args.job_id,
        "group_id": args.group_id,
        "candidate_count": args.expected_count,
        "candidate_indices": list(range(args.expected_count)),
        "unique_mp4_sha256_count": len({row["mp4_sha256"] for row in audit_rows}),
        "root": str(root),
        "root_spec_raw_sha256": audit["root_spec_raw_sha256"],
        "deep_audit": {
            "path": str(args.deep_audit),
            "sha256": audit_sha,
            "receipt_digest": audit["receipt_digest"],
        },
        "partial_shard_receipt": {
            "path": str(partial_path),
            "sha256": partial_sha,
            "receipt_digest": partial["receipt_digest"],
        },
        "slurm_terminal_observation": sacct,
        "logs": {
            "stdout": str(args.stdout),
            "stdout_sha256": sha256_file(args.stdout),
            "stderr": str(args.stderr),
            "stderr_sha256": sha256_file(args.stderr),
        },
        "authority": {
            "full_event_bank_admission": False,
            "detached_decoded_event_review": False,
            "human_review": False,
            "merge_or_partial_reuse": False,
            "optimizer": False,
            "scientific_selection": False,
            "training": False,
        },
    }
    value["receipt_digest"] = canonical_digest(value)
    write_create_only(args.output, value)
    print(json.dumps({
        "output": str(args.output),
        "receipt_digest": value["receipt_digest"],
        "status": STATUS,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bind one completed SAIC single-root exact60 bank to terminal evidence.

The output authorizes detached decoded-video review only.  It never authorizes
training, optimization, row admission, or an action-editing success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, NoReturn


SCHEMA = "saic-t2v-full60-terminal-evidence-v1"
STATUS = "terminal_technical_full60_complete_pending_detached_semantic_review"
MASTER_SCHEMA = "bernini-saic-pure-t2v-event-bank-topup-receipt-v2"
DEEP_SCHEMA = "saic-t2v-live-shard-prefix-audit-v1"
SACCT_FIELDS = (
    "JobIDRaw,JobName,State,ExitCode,Elapsed,AllocNodes,AllocTRES,SubmitLine"
)
GROUPS = ("sp4-a", "sp4-b")


def die(message: str) -> NoReturn:
    raise SystemExit(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_receipt(path: Path, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        die(f"{label} is not one absolute regular non-symlink file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        die(f"cannot read {label}: {error}")
    if not isinstance(value, dict):
        die(f"{label} is not an object")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_digest", None)
    if claimed != object_sha256(unsigned):
        die(f"{label} canonical digest differs")
    if raw != canonical_bytes(value) + b"\n":
        die(f"{label} bytes are not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def validate_master(master: dict[str, Any], *, root: Path) -> list[dict[str, Any]]:
    attempts = master.get("attempts")
    proofs = master.get("same_seed_official_gaussian_proofs")
    if (
        master.get("schema_version") != MASTER_SCHEMA
        or master.get("topology")
        != "two_concurrent_world4_sp4_groups_on_one_8gpu_node"
        or master.get("attempt_count") != 60
        or master.get("seed_cell_count") != 20
        or master.get("six_branch_spec_merge_cell_count") != 20
        or not isinstance(attempts, list)
        or len(attempts) != 60
        or not isinstance(proofs, list)
        or len(proofs) != 20
        or master.get("detached_full81_event_review_complete") is not False
        or master.get("event_verified") is not False
        or master.get("identity_preservation_verified") is not False
        or master.get("seed_selection_authorized") is not False
        or master.get("training_target_authorized") is not False
        or master.get("optimizer_or_parameter_update_authorized") is not False
    ):
        die("master receipt does not close an unreviewed exact60 bank")
    ids = [row.get("candidate_id") for row in attempts]
    mp4_hashes = [row.get("mp4_sha256") for row in attempts]
    if (
        len(set(ids)) != 60
        or len(set(mp4_hashes)) != 60
        or any(not isinstance(value, str) or len(value) != 64 for value in mp4_hashes)
        or {row.get("branch") for row in attempts}
        != {"incomplete", "camera_only", "appearance_only"}
    ):
        die("master exact60 candidate or MP4 coverage differs")
    for row in attempts:
        receipt_path = Path(str(row.get("receipt_path", "")))
        mp4_path = Path(str(row.get("mp4_path", "")))
        try:
            receipt_path.relative_to(root)
            mp4_path.relative_to(root)
        except ValueError:
            die("master artifact path escaped the output root")
        if row.get("event_audit_status") != "pending_detached_full81_review":
            die("master row silently claims semantic review")
    return attempts


def validate_deep_audit(
    audit: dict[str, Any], *, root: Path, group_id: str, job_id: str,
) -> list[dict[str, Any]]:
    rows = audit.get("rows")
    authority = audit.get("authority")
    if (
        group_id not in GROUPS
        or audit.get("schema_version") != DEEP_SCHEMA
        or audit.get("root") != str(root)
        or audit.get("group_id") != group_id
        or audit.get("slurm_job_id") != job_id
        or audit.get("planned_candidate_count") != 30
        or audit.get("completed_prefix_count") != 30
        or audit.get("completed_candidate_indices") != list(range(30))
        or audit.get("deep_generation_receipt_validation") is not True
        or audit.get("deep_rendezvous_completion_validation") is not True
        or audit.get("same_cell_gaussian_prefix_validation") is not True
        or not isinstance(authority, dict)
        or any(value is not False for value in authority.values())
        or not isinstance(rows, list)
        or len(rows) != 30
    ):
        die(f"{group_id} deep audit does not close exact30")
    if [row.get("candidate_index") for row in rows] != list(range(30)):
        die(f"{group_id} deep-audit indices differ")
    if len({row.get("candidate_id") for row in rows}) != 30:
        die(f"{group_id} deep-audit candidate IDs differ")
    if len({row.get("mp4_sha256") for row in rows}) != 30:
        die(f"{group_id} deep-audit MP4 hashes differ")
    return rows


def bind_master_to_deep(
    master_rows: list[dict[str, Any]], deep_rows: list[dict[str, Any]]
) -> None:
    master = {row["candidate_id"]: row for row in master_rows}
    if len(deep_rows) != 60 or len({row.get("candidate_id") for row in deep_rows}) != 60:
        die("combined deep audits do not cover exact60")
    if len({row.get("mp4_sha256") for row in deep_rows}) != 60:
        die("combined deep audits do not bind 60 unique MP4s")
    if set(master) != {row.get("candidate_id") for row in deep_rows}:
        die("master/deep candidate coverage differs")
    for row in deep_rows:
        bound = master[row["candidate_id"]]
        if (
            bound.get("receipt_sha256") != row.get("attempt_receipt_sha256")
            or bound.get("receipt_digest") != row.get("attempt_receipt_digest")
            or bound.get("mp4_sha256") != row.get("mp4_sha256")
            or bound.get("branch") != row.get("branch")
        ):
            die(f"master/deep binding differs for {row['candidate_id']}")


def exact_sacct(job_id: str, sacct: Path, *, required_tokens: tuple[str, ...]) -> dict[str, Any]:
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
    parsed = dict(zip(SACCT_FIELDS.split(","), fields))
    tres = set(parsed["AllocTRES"].split(","))
    if (
        parsed["JobIDRaw"] != job_id
        or parsed["State"] != "COMPLETED"
        or parsed["ExitCode"] != "0:0"
        or parsed["AllocNodes"] != "1"
        or "gres/gpu:mi210=8" not in tres
        or "node=1" not in tres
        or any(token not in parsed["SubmitLine"] for token in required_tokens)
    ):
        die("Slurm terminal allocation or submit binding differs")
    return {
        "command": command,
        "executable": str(sacct),
        "executable_sha256": file_sha256(sacct),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "parsed_row": parsed,
    }


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
    parser.add_argument("--master-receipt", type=Path, required=True)
    parser.add_argument("--deep-audit-sp4-a", type=Path, required=True)
    parser.add_argument("--deep-audit-sp4-b", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, default=Path("/usr/bin/sacct"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        not args.job_id.isdigit()
        or len(args.source_revision) != 40
        or len(args.source_archive_sha256) != 64
    ):
        die("job or source identity differs")
    root = args.output_root.resolve(strict=True)
    master, master_sha = load_receipt(args.master_receipt, "master receipt")
    master_rows = validate_master(master, root=root)
    deep_values: list[tuple[str, dict[str, Any], str, Path]] = []
    for group_id, path in (
        ("sp4-a", args.deep_audit_sp4_a), ("sp4-b", args.deep_audit_sp4_b)
    ):
        value, digest = load_receipt(path, f"{group_id} deep audit")
        deep_values.append((group_id, value, digest, path))
    if {value["root_spec_raw_sha256"] for _, value, _, _ in deep_values} != {
        master.get("root_spec_raw_sha256")
    }:
        die("master/deep root-spec binding differs")
    deep_rows: list[dict[str, Any]] = []
    for group_id, value, _, _ in deep_values:
        deep_rows.extend(validate_deep_audit(
            value, root=root, group_id=group_id, job_id=args.job_id
        ))
    bind_master_to_deep(master_rows, deep_rows)
    required_tokens = (
        f"SAIC_T2V_V3_OUTPUT_ROOT={root}",
        f"SAIC_T2V_V3_SOURCE_REVISION={args.source_revision}",
        f"SAIC_T2V_V3_SOURCE_ARCHIVE_SHA256={args.source_archive_sha256}",
        f"SAIC_T2V_V3_EVENT_SPEC_SHA256={master['root_spec_raw_sha256']}",
    )
    sacct = exact_sacct(args.job_id, args.sacct, required_tokens=required_tokens)
    for path, label in ((args.stdout, "stdout"), (args.stderr, "stderr")):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            die(f"{label} is not one absolute regular non-symlink file")
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "job_id": args.job_id,
        "root": str(root),
        "candidate_count": 60,
        "seed_cell_count": 20,
        "unique_mp4_sha256_count": 60,
        "source_revision": args.source_revision,
        "source_archive_sha256": args.source_archive_sha256,
        "root_spec_raw_sha256": master["root_spec_raw_sha256"],
        "master_receipt": {
            "path": str(args.master_receipt), "sha256": master_sha,
            "receipt_digest": master["receipt_digest"],
        },
        "deep_audits": {
            group_id: {
                "path": str(path), "sha256": digest,
                "receipt_digest": audit["receipt_digest"],
            }
            for group_id, audit, digest, path in deep_values
        },
        "slurm_terminal_observation": sacct,
        "logs": {
            "stdout": str(args.stdout), "stdout_sha256": file_sha256(args.stdout),
            "stderr": str(args.stderr), "stderr_sha256": file_sha256(args.stderr),
        },
        "authority": {
            "detached_decoded_event_review_input": True,
            "data_selection": False,
            "human_review": False,
            "optimizer": False,
            "scientific_action_editing_success_claim": False,
            "training": False,
            "training_target_admission": False,
        },
    }
    value["receipt_digest"] = object_sha256(value)
    write_create_only(args.output, value)
    print(json.dumps({
        "output": str(args.output), "receipt_digest": value["receipt_digest"],
        "status": STATUS,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

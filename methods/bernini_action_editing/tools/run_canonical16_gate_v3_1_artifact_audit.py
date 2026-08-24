#!/usr/bin/env python3
"""Run the SHA-bound canonical16 v3.1 artifact audit on existing videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


SCHEMA = "bernini-canonical16-gate-v3.1-input-manifest-v1"
GATE_MANIFEST_SCHEMA = "bernini-canonical16-gate-v3.1-report-manifest-v1"
_IID_RE = re.compile(r"^s([0-9]{8})-case([0-9]{2})$")


class ArtifactAuditError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactAuditError(f"not a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ArtifactAuditError("Qwen records must be newline-terminated JSONL")
    rows = []
    for index, line in enumerate(payload.splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ArtifactAuditError(f"Qwen row {index} is not an object")
        rows.append(row)
    return rows


def stable_media(identity: dict[str, Any], *, context: str) -> dict[str, Any]:
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ArtifactAuditError(f"invalid {context} path: {path}")
    observed = file_sha(path)
    if observed != identity.get("sha256"):
        raise ArtifactAuditError(f"{context} SHA differs: {path}")
    size = path.stat().st_size
    if identity.get("size_bytes") != size:
        raise ArtifactAuditError(f"{context} size differs: {path}")
    return {"path": str(path.resolve()), "sha256": observed, "bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-records", type=Path, required=True)
    parser.add_argument("--primary-evidence", type=Path, required=True)
    parser.add_argument("--memretry-evidence", type=Path, required=True)
    parser.add_argument("--gate-tool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    qwen_records = args.qwen_records.resolve(strict=True)
    primary = args.primary_evidence.resolve(strict=True)
    memretry = args.memretry_evidence.resolve(strict=True)
    gate_tool = args.gate_tool.resolve(strict=True)
    output = args.output_dir.expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ArtifactAuditError("output must be one absent absolute path")
    output.parent.mkdir(parents=True, exist_ok=True)
    qwen_rows = read_jsonl(qwen_records)
    if len(qwen_rows) != 16:
        raise ArtifactAuditError(f"expected 16 Qwen rows, got {len(qwen_rows)}")
    by_iid = {str(row.get("iid")): row for row in qwen_rows}
    if len(by_iid) != 16 or any(_IID_RE.fullmatch(iid) is None for iid in by_iid):
        raise ArtifactAuditError("canonical16 Qwen IID set differs")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        reports_dir = staging / "reports"
        reports_dir.mkdir()
        input_rows = []
        bindings = []
        for iid in sorted(by_iid):
            qwen = by_iid[iid]
            retry_path = memretry / f"{iid}.gate-login-v2.json"
            primary_path = primary / f"{iid}.gate-login-v2.json"
            binding_path = retry_path if retry_path.is_file() else primary_path
            if not binding_path.is_file():
                raise ArtifactAuditError(f"missing legacy media binding for {iid}")
            legacy = read_json(binding_path)
            metadata = legacy.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("sample_id") != iid[-6:]:
                raise ArtifactAuditError(f"legacy sample binding differs for {iid}")
            inputs = metadata.get("inputs")
            if not isinstance(inputs, dict) or set(inputs) != {
                "source",
                "candidate",
                "frozen_base",
            }:
                raise ArtifactAuditError(f"legacy media identities differ for {iid}")
            media = {
                name: stable_media(inputs[name], context=f"{iid}.{name}")
                for name in ("source", "candidate", "frozen_base")
            }
            qwen_input = qwen.get("input")
            if not isinstance(qwen_input, dict):
                raise ArtifactAuditError(f"Qwen input missing for {iid}")
            if (
                qwen_input.get("source_video", {}).get("sha256")
                != media["source"]["sha256"]
                or qwen_input.get("target_video", {}).get("sha256")
                != media["candidate"]["sha256"]
            ):
                raise ArtifactAuditError(f"Qwen/legacy media SHA differs for {iid}")
            match = _IID_RE.fullmatch(iid)
            assert match is not None
            input_rows.append(
                {
                    "iid": iid,
                    "checkpoint_step": int(match.group(1)),
                    "case_index": int(match.group(2)),
                    "media": media,
                    "legacy_binding_path": str(binding_path),
                    "legacy_binding_sha256": file_sha(binding_path),
                    "qwen_record_digest": qwen.get("record_digest"),
                }
            )
            bindings.append(binding_path)

        manifest = {
            "schema_version": SCHEMA,
            "complete": True,
            "rows": input_rows,
            "row_count": len(input_rows),
            "rows_digest": object_sha(input_rows),
            "qwen_records": {
                "path": str(qwen_records),
                "sha256": file_sha(qwen_records),
                "rows": len(qwen_rows),
            },
            "gate_tool": {"path": str(gate_tool), "sha256": file_sha(gate_tool)},
            "runner": {
                "path": str(Path(__file__).resolve(strict=True)),
                "sha256": file_sha(Path(__file__).resolve(strict=True)),
            },
        }
        input_manifest = staging / "input-manifest.json"
        write_json(input_manifest, manifest)

        report_rows = []
        for ordinal, row in enumerate(input_rows):
            iid = row["iid"]
            report_path = reports_dir / f"{iid}.quality-v3_1.json"
            command = [
                sys.executable,
                str(gate_tool),
                "--source",
                row["media"]["source"]["path"],
                "--candidate",
                row["media"]["candidate"]["path"],
                "--frozen-base",
                row["media"]["frozen_base"]["path"],
                "--output",
                str(report_path),
                "--sample-id",
                iid,
                "--checkpoint-step",
                str(row["checkpoint_step"]),
                "--checkpoint-label",
                f"checkpoint-{row['checkpoint_step']:08d}",
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode not in {0, 2} or not report_path.is_file():
                raise ArtifactAuditError(
                    f"gate-v3.1 failed for {iid}: rc={completed.returncode}"
                )
            report = read_json(report_path)
            if (
                report.get("schema_version")
                != "bernini-checkpoint-visual-quality-gate-v3.1"
                or report.get("metadata", {}).get("sample_id") != iid
            ):
                raise ArtifactAuditError(f"gate-v3.1 report binding differs for {iid}")
            for name in ("source", "candidate", "frozen_base"):
                if (
                    report["metadata"]["inputs"][name]["sha256"]
                    != row["media"][name]["sha256"]
                ):
                    raise ArtifactAuditError(f"gate report media SHA differs: {iid}.{name}")
            report_rows.append(
                {
                    "iid": iid,
                    "ordinal": ordinal,
                    "path": str(output / "reports" / report_path.name),
                    "sha256": file_sha(report_path),
                    "status": report.get("status"),
                    "hard_artifact_failure": bool(
                        report.get("hard_artifact_failure", False)
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "event": "canonical16_gate_v3_1",
                        "iid": iid,
                        "ordinal": ordinal,
                        "status": report.get("status"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        gate_manifest = {
            "schema_version": GATE_MANIFEST_SCHEMA,
            "complete": True,
            "input_manifest_path": str(output / input_manifest.name),
            "input_manifest_sha256": file_sha(input_manifest),
            "rows": report_rows,
            "row_count": len(report_rows),
            "rows_digest": object_sha(report_rows),
        }
        write_json(staging / "gate-manifest.json", gate_manifest)
        os.replace(staging, output)
    except BaseException:
        # Preserve staging on failure as diagnostic evidence; it is never
        # mistaken for the requested create-only final output directory.
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

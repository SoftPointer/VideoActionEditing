#!/usr/bin/env python3
"""Build a deterministic one-row-per-action-family evaluation projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "bernini-full644-self-generated-anchor-exact28-eval-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: Path) -> dict[str, Any]:
    raw = source.read_bytes()
    value = json.loads(raw)
    rows = value.get("rows")
    if type(rows) is not list or len(rows) != 644:
        raise ValueError("source manifest must contain exact644 rows")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("family")) is not str:
            raise ValueError("invalid source row")
        by_family.setdefault(row["family"], []).append(row)
    if len(by_family) != 28:
        raise ValueError("source manifest must contain exact28 action families")
    selected = []
    for family in sorted(by_family):
        candidates = sorted(
            by_family[family],
            key=lambda row: (
                not (row.get("strict_selection_gates_all_true") is True),
                row["iid"],
            ),
        )
        row = candidates[0]
        media = row["media_binding"]
        selected.append(
            {
                "action_anchor_video_path": media["action_anchor_video_path"],
                "action_anchor_video_sha256": media["action_anchor_video_sha256"],
                "family": family,
                "iid": row["iid"],
                "instruction": row["instruction"],
                "instruction_sha256": row["instruction_sha256"],
                "source_video_path": media["source_video_path"],
                "source_video_sha256": media["source_video_sha256"],
                "strict_selection_gates_all_true": (
                    row.get("strict_selection_gates_all_true") is True
                ),
            }
        )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "training_dataset_row_count": 644,
        "evaluation_row_count": 28,
        "selection_rule": "one_per_family_strict_first_then_lexicographic_iid",
        "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_digest": value.get("manifest_digest"),
        "rows": selected,
        "scientific_claim_authorized": False,
    }
    result["selection_digest"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.source_manifest.is_absolute() or not args.output.is_absolute():
        raise ValueError("paths must be absolute")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.source_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output, flags, 0o444)
    try:
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        os.write(fd, (payload + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(args.output, 0o444)
    print(json.dumps({"path": str(args.output), "sha256": _sha256(args.output), **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

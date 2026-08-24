#!/usr/bin/env python3
"""Add same-appearance RAFT correspondence authority to a motion manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--matched-flow-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve(strict=True)
    flow_root = Path(args.matched_flow_root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("output must be fresh")
    manifest = json.loads(input_path.read_text(encoding="utf-8"))
    stored = manifest.pop("manifest_digest", None)
    rows = manifest.get("rows")
    if (
        manifest.get("schema_version") != "bernini-same-video-motion-pairs-v1"
        or object_sha256(manifest) != stored
        or not isinstance(rows, list)
        or len(rows) != 32
    ):
        raise RuntimeError("input motion manifest closure differs")
    seen = set()
    for row in rows:
        iid = row.get("iid")
        if not isinstance(iid, str) or iid in seen:
            raise RuntimeError("input IID closure differs")
        seen.add(iid)
        flow = (flow_root / f"{iid}.safetensors").resolve(strict=True)
        row["source_correspondence_flow_bundle"] = str(flow)
        row["source_correspondence_flow_bundle_sha256"] = file_sha256(flow)
    manifest.update(
        {
            "source_correspondence_enabled": True,
            "source_correspondence_teacher": (
                "same_appearance_target_raft_cumulative_backward_raw_flow"
            ),
            "source_correspondence_flow_root": str(flow_root),
            "source_correspondence_flow_file_count": len(rows),
            "source_correspondence_used_at_inference": False,
        }
    )
    manifest["manifest_digest"] = object_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "manifest_digest": manifest["manifest_digest"],
                "sha256": file_sha256(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

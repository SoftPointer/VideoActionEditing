#!/usr/bin/env python3
"""Replace one geometrically invalid flow bundle in an expanded action bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "bernini-same-video-motion-pairs-v1"


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--iid", required=True)
    parser.add_argument("--flow-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.input_manifest).expanduser().resolve(strict=True)
    flow = Path(args.flow_bundle).expanduser().resolve(strict=True)
    sidecar = flow.with_suffix(".json").resolve(strict=True)
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise SystemExit("output must be a fresh absolute path")

    manifest = json.loads(source.read_text(encoding="utf-8"))
    stored = manifest.pop("manifest_digest", None)
    if manifest.get("schema_version") != SCHEMA or object_sha256(manifest) != stored:
        raise SystemExit("input manifest semantic digest differs")
    matches = [row for row in manifest.get("rows", []) if row.get("iid") == args.iid]
    if len(matches) != 1:
        raise SystemExit("repair IID must select exactly one row")
    row = matches[0]
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_hw = list(map(int, row["latents"]["shape"][-2:]))
    if metadata.get("latent_hw") != expected_hw:
        raise SystemExit(
            f"replacement flow/latent geometry differs: {metadata.get('latent_hw')} != {expected_hw}"
        )
    if (
        metadata.get("source_sha256") != row.get("source_video_sha256")
        or metadata.get("anchor_sha256") != row.get("anchor_video_sha256")
    ):
        raise SystemExit("replacement flow video provenance differs")

    old_path = row["flow_bundle"]
    old_sha = row["flow_bundle_sha256"]
    row["flow_bundle"] = str(flow)
    row["flow_bundle_sha256"] = file_sha256(flow)
    repairs = list(manifest.get("flow_geometry_repairs", []))
    repairs.append(
        {
            "iid": args.iid,
            "old_flow_bundle": old_path,
            "old_flow_bundle_sha256": old_sha,
            "replacement_flow_bundle": str(flow),
            "replacement_flow_bundle_sha256": row["flow_bundle_sha256"],
            "reason": "raft_source_bucket_must_match_native_inference_upscaling",
        }
    )
    manifest["flow_geometry_repairs"] = repairs
    manifest["manifest_digest"] = object_sha256(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical(manifest) + "\n", encoding="utf-8")
    print(canonical({"output": str(output), "iid": args.iid, "latent_hw": expected_hw}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

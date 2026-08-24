#!/usr/bin/env python3
"""Freeze the source-only, equal-seed evaluation rows for reward training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


SCHEMA = "bernini-reward-training-eval-manifest-v1"
RESULT_SCHEMA = "action-editing-reward-ablation-result-v1"


class ManifestError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(result_path: Path, output_path: Path, *, seed: int) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise ManifestError(f"create-only output exists: {output_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != RESULT_SCHEMA:
        raise ManifestError("reward result schema differs")
    groups = result.get("groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ManifestError("exactly four evaluation groups are required")
    rows = []
    for group in groups:
        iid = group.get("iid")
        instruction = group.get("instruction")
        source = group.get("source")
        if (
            not isinstance(iid, str)
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(source, dict)
        ):
            raise ManifestError("evaluation group fields differ")
        source_path = Path(str(source.get("path"))).resolve(strict=True)
        if source_path.is_symlink() or file_sha256(source_path) != source.get("sha256"):
            raise ManifestError(f"source identity differs for {iid}")
        rows.append(
            {
                "iid": iid,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "source_video_path": str(source_path),
                "source_video_sha256": source["sha256"],
            }
        )
    manifest = {
        "schema_version": SCHEMA,
        "complete": True,
        "source_only_inference": True,
        "same_seed_all_arms": True,
        "candidate_pool_seeds_excluded": True,
        "inference_seed": seed,
        "num_frames": 81,
        "fps": 25.0,
        "num_inference_steps": 40,
        "reward_result_path": str(result_path.resolve()),
        "reward_result_sha256": file_sha256(result_path),
        "rows": rows,
    }
    manifest["manifest_digest"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081601)
    args = parser.parse_args(argv)
    if args.seed in {2026080901, 2026080902, 2026081501, 2026081502}:
        raise ManifestError("evaluation seed overlaps the candidate pool")
    manifest = build(
        args.reward_result.expanduser().resolve(strict=True),
        args.output.expanduser().absolute(),
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

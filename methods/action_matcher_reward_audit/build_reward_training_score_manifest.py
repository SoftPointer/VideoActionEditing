#!/usr/bin/env python3
"""Bind trained/frozen evaluation videos into the existing fast reward scorer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


ARMS = ("frozen_base", "baseline", "action_only", "preservation_only", "composite")
NODES = (
    "auh7-1b-gpu-245",
    "auh7-1b-gpu-246",
    "auh7-1b-gpu-247",
    "auh7-1b-gpu-248",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(original_path: Path, eval_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise ValueError(f"create-only output exists: {output}")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("schema_version") != "action-editing-reward-ablation-manifest-v1":
        raise ValueError("original reward manifest schema differs")
    groups = []
    for index, group in enumerate(original["groups"]):
        iid = group["iid"]
        candidates = []
        for arm in ARMS:
            media = (eval_root / iid / f"{arm}.mp4").resolve(strict=True)
            if media.is_symlink():
                raise ValueError(f"evaluation media is a symlink: {media}")
            candidates.append(
                {
                    "candidate_id": f"reward-training-eval-{iid}-{arm}",
                    "seed": 2026081601,
                    "origin": f"job135096_u40_{arm}",
                    "media": {"path": str(media), "sha256": file_sha256(media)},
                }
            )
        groups.append(
            {
                "iid": iid,
                "node": NODES[index],
                "instruction": group["instruction"],
                "source": group["source"],
                "action_anchor": group["action_anchor"],
                "baseline_candidate_id": f"reward-training-eval-{iid}-frozen_base",
                "candidates": candidates,
            }
        )
    original_generator = original["generator"]
    manifest = {
        "schema_version": "action-editing-reward-ablation-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "reward-training-pairv5-job135096-u40-eval-v1",
        "generator": {
            **original_generator,
            "parameter_update_performed": True,
            "candidate_models": {
                "frozen_base": {"optimizer_updates": 0},
                "baseline": {"optimizer_updates": 40, "target_policy": "fixed_seed_no_reward"},
                "action_only": {"optimizer_updates": 40, "target_policy": "action_reward_selected"},
                "preservation_only": {
                    "optimizer_updates": 40,
                    "target_policy": "preservation_reward_selected",
                },
                "composite": {"optimizer_updates": 40, "target_policy": "composite_reward_selected"},
            },
            "training_seed": 20260805,
            "evaluation_seed": 2026081601,
            "evaluation_seed_excluded_from_training_target_pool": True,
        },
        "policy": {
            "diagnostic_only": True,
            "qwen_or_vlm": False,
            "human_review_required_for_efficacy": True,
            "comparison": "five separately frozen/trained models under one matched evaluation seed",
            "claim_boundary": "same-IID held-out-noise; not cross-IID generalization",
        },
        "groups": groups,
    }
    manifest["manifest_digest"] = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(
        args.original_manifest.expanduser().resolve(strict=True),
        args.eval_root.expanduser().resolve(strict=True),
        args.output.expanduser().absolute(),
    )
    print(result["manifest_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

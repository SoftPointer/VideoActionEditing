#!/usr/bin/env python3
"""Build the four-node rollout specs and frozen reward-ablation manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "action-editing-reward-ablation-manifest-v1"
ROLLOUT_SCHEMA = "pair-v5-native-rv2v4-rollout-spec-v1"
SEEDS = (2026081501, 2026081502)
NODE_BY_IID = {
    "7b88a1ca1f804f41": "auh7-1b-gpu-245",
    "841b5e0080a1441d": "auh7-1b-gpu-246",
    "a35b590961d24694": "auh7-1b-gpu-247",
    "a66e6818e4144928": "auh7-1b-gpu-248",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any, *, compact: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        raw = canonical_bytes(value) + b"\n"
    else:
        raw = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    destination.write_bytes(raw)


def source_rows(base_spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for group in base_spec["groups"]:
        for candidate in group["candidates"]:
            iid = candidate["candidate_id"].split("-action-s", 1)[0].rsplit("-", 1)[-1]
            previous = rows.setdefault(iid, candidate)
            for key in (
                "source_video",
                "source_video_sha256",
                "complete_caption",
                "complete_caption_sha256",
                "caption_contract",
                "guidance",
            ):
                if previous[key] != candidate[key]:
                    raise ValueError(f"base candidates disagree for {iid}: {key}")
    if set(rows) != set(NODE_BY_IID):
        raise ValueError(f"base source population differs: {sorted(rows)}")
    return rows


def build_specs(args: argparse.Namespace) -> int:
    base_spec = json.loads(Path(args.base_spec).read_text(encoding="utf-8"))
    if base_spec.get("schema_version") != ROLLOUT_SCHEMA:
        raise ValueError("base rollout schema differs")
    rows = source_rows(base_spec)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    receipt_rows = []
    for iid, node in NODE_BY_IID.items():
        base = rows[iid]
        candidates = []
        for seed in SEEDS:
            candidates.append(
                {
                    **{
                        key: base[key]
                        for key in (
                            "source_video",
                            "source_video_sha256",
                            "complete_caption",
                            "complete_caption_sha256",
                            "caption_contract",
                            "guidance",
                        )
                    },
                    "candidate_id": f"reward-ablation-v1-{iid}-action-s{seed}",
                    "seed": seed,
                }
            )
        spec = {
            "schema_version": ROLLOUT_SCHEMA,
            "sampling_contract": base_spec["sampling_contract"],
            "semantic_input_closure": base_spec["semantic_input_closure"],
            "groups": [
                {
                    "group_id": "sp4-a",
                    "visible_gpus": [0, 1, 2, 3],
                    "candidates": [candidates[0]],
                },
                {
                    "group_id": "sp4-b",
                    "visible_gpus": [4, 5, 6, 7],
                    "candidates": [candidates[1]],
                },
            ],
        }
        path = output / f"rollout-{iid}.json"
        write_json(path, spec)
        receipt_rows.append(
            {
                "iid": iid,
                "node": node,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "candidate_ids": [row["candidate_id"] for row in candidates],
            }
        )
    receipt = {
        "schema_version": "action-editing-reward-ablation-rollout-spec-set-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_spec": str(Path(args.base_spec).resolve()),
        "base_spec_sha256": file_sha256(args.base_spec),
        "rows": receipt_rows,
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    write_json(output / "spec-set-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def media(path: Path) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"media is absent/not plain: {path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def build_manifest(args: argparse.Namespace) -> int:
    base_spec = json.loads(Path(args.base_spec).read_text(encoding="utf-8"))
    rows = source_rows(base_spec)
    old_root = Path(args.old_rollout_root)
    topup_root = Path(args.topup_root)
    anchor_root = Path(args.anchor_root)
    groups = []
    for iid, node in NODE_BY_IID.items():
        row = rows[iid]
        candidates = []
        for seed in (2026080901, 2026080902):
            candidate_id = f"pair5-native-core4-v1-{iid}-action-s{seed}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "origin": "existing_frozen_native_rollout",
                    "media": media(old_root / candidate_id / "rv2v.mp4"),
                }
            )
        for seed in SEEDS:
            candidate_id = f"reward-ablation-v1-{iid}-action-s{seed}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "origin": f"job135096_topup_{node}",
                    "media": media(topup_root / iid / candidate_id / "rv2v.mp4"),
                }
            )
        groups.append(
            {
                "iid": iid,
                "node": node,
                "instruction": row["complete_caption"],
                "source": media(Path(row["source_video"])),
                "action_anchor": media(
                    anchor_root
                    / f"pair5-t2v-core4-v2-{iid}-action"
                    / "t2v.mp4"
                ),
                "baseline_candidate_id": candidates[0]["candidate_id"],
                "candidates": candidates,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "reward-bestof4-pairv5-job135096-v1",
        "generator": {
            "name": "Bernini-R-1.3B-Diffusers-renderer-only",
            "checkpoint_tree_sha256": "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "native_mode": "rv2v4",
            "frames": 81,
            "fps": 25,
            "steps": 40,
            "same_source_instruction_guidance_across_candidates": True,
            "parameter_update_performed": False,
        },
        "policy": {
            "baseline": "fixed seed 2026080901; no reward",
            "action_only": "SemanticMoments M3 + ordered DINO + reverse contrast + activity; abstain",
            "preservation_only": "relative max-min of four weak source-bound proxies; diagnostic only",
            "composite": "relative preservation gate followed by action score; no weighted sum; abstain",
            "qwen_or_vlm": False,
            "same_candidate_pool_for_all_arms": True,
        },
        "groups": groups,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "manifest_digest": manifest["manifest_digest"],
                "groups": len(groups),
                "videos": sum(2 + len(row["candidates"]) for row in groups),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    specs = subparsers.add_parser("build-specs")
    specs.add_argument("--base-spec", required=True)
    specs.add_argument("--output-dir", required=True)
    specs.set_defaults(func=build_specs)
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--base-spec", required=True)
    manifest.add_argument("--old-rollout-root", required=True)
    manifest.add_argument("--topup-root", required=True)
    manifest.add_argument("--anchor-root", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=build_manifest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

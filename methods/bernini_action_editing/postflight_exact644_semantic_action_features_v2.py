#!/usr/bin/env python3
"""Validate and seal exact644 source/anchor ordered feature shards (v2)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Sequence


FEATURE_SCHEMA = "semantic-moments-action-reward-features-v1"
MODEL_SHA256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def postflight(root: Path, manifest_path: Path) -> dict[str, Any]:
    import torch

    root = root.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    shards = []
    manifest_digest = None
    step_ids: set[str] = set()
    for index in range(8):
        path = root / "features" / f"features-shard-{index}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != FEATURE_SCHEMA:
            raise ValueError("feature schema differs")
        if payload.get("shard_index") != index or payload.get("num_shards") != 8:
            raise ValueError("feature shard placement differs")
        if payload.get("record_count") != 161:
            raise ValueError("every exact8 shard must contain 161 records")
        if manifest_digest is None:
            manifest_digest = payload.get("manifest_digest")
        elif manifest_digest != payload.get("manifest_digest"):
            raise ValueError("feature shards disagree on manifest")
        runtime = payload.get("runtime")
        if type(runtime) is not dict:
            raise ValueError("runtime receipt is absent")
        model_files = {
            row["relative_path"]: row["sha256"] for row in runtime["model_files"]
        }
        if model_files.get("model.safetensors") != MODEL_SHA256:
            raise ValueError("frozen DINO weights differ")
        if runtime.get("visible_device_count") != 1:
            raise ValueError("each shard must see exactly one GPU")
        if runtime.get("visible_device_name") != "AMD Instinct MI210":
            raise ValueError("feature extractor did not run on MI210")
        step_ids.add(str(runtime.get("slurm_step_id")))
        records.extend(payload["records"])
        shards.append(
            {
                "index": index,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "record_count": payload["record_count"],
                "rocr_visible_devices": runtime.get("rocr_visible_devices"),
            }
        )
    if len(step_ids) != 1 or "None" in step_ids:
        raise ValueError("all shards must bind one Slurm step")
    if {row["rocr_visible_devices"] for row in shards} != {
        str(index) for index in range(8)
    }:
        raise ValueError("feature shards must bind exact8 distinct physical GPU ordinals")
    if manifest_digest != manifest.get("manifest_digest"):
        raise ValueError("feature payload does not bind the published manifest")
    if len(records) != 1288:
        raise ValueError("feature population is not exact1288")
    ids = [row["item_id"] for row in records]
    if len(set(ids)) != 1288:
        raise ValueError("feature item IDs are not unique")
    groups: dict[str, int] = {}
    iid_roles: dict[str, set[str]] = {}
    for row in records:
        if tuple(row["frame_sequence"].shape) != (32, 768):
            raise ValueError("ordered feature geometry differs")
        if tuple(row["components"].shape) != (3, 768):
            raise ValueError("moment component geometry differs")
        if row["frame_sequence"].dtype != torch.float32:
            raise ValueError("ordered features must be FP32")
        if not bool(torch.isfinite(row["frame_sequence"]).all()):
            raise ValueError("ordered features contain non-finite values")
        if not bool(torch.isfinite(row["components"]).all()):
            raise ValueError("moment components contain non-finite values")
        group = row["group"]
        groups[group] = groups.get(group, 0) + 1
        metadata = row["metadata"]
        iid_roles.setdefault(metadata["iid"], set()).add(metadata["role"])
        if metadata.get("paired_ground_truth_claimed") is not False:
            raise ValueError("feature metadata must not claim paired ground truth")
    if groups != {"exact644_action_anchor": 644, "exact644_source": 644}:
        raise ValueError(f"feature group coverage differs: {groups}")
    if len(iid_roles) != 644 or any(
        roles != {"source", "action_anchor"} for roles in iid_roles.values()
    ):
        raise ValueError("every exact644 IID must have one source and one anchor feature")

    receipt: dict[str, Any] = {
        "schema_version": "semantic-action-exact644-feature-extraction-receipt-v1",
        "status": "FEATURES_EXTRACTED_NOT_REPRESENTATION_QUALIFIED",
        "authority": "feature_mechanics_diagnostic_only",
        "formal_training_authorized": False,
        "paired_ground_truth_claimed": False,
        "holder_job_id": os.environ.get("SLURM_JOB_ID", "141620"),
        "feature_step_id": next(iter(step_ids)),
        "postflight_hostname": socket.gethostname(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "manifest_digest": manifest_digest,
        },
        "population": {
            "unique_base_clips": 644,
            "action_anchor_records": 644,
            "source_records_for_nuisance_probe": 644,
            "total_feature_records": 1288,
            "counterfactual_rows": 0,
        },
        "feature_geometry": {"frames": 32, "dimension": 768, "moments": 3},
        "frozen_teacher": {
            "kind": "DINOv2-base ordered per-frame descriptors",
            "weights_sha256": MODEL_SHA256,
            "semantic_moments_role": "unordered auxiliary only",
        },
        "shards": shards,
    }
    unsigned = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    receipt["receipt_digest"] = hashlib.sha256(unsigned).hexdigest()
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = postflight(args.root, args.manifest)
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    destination = args.root / "feature_extraction_receipt.json"
    with destination.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    print(
        json.dumps(
            {
                "receipt": str(destination.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "receipt_digest": receipt["receipt_digest"],
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

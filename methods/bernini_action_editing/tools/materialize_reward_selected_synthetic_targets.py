#!/usr/bin/env python3
"""Build four-row Bernini training shards from frozen reward selections.

The native rollout receipts already contain normalized clean source and output
latents.  This tool converts those exact latents into near-deterministic Wan VAE
posterior parameters, without decoding and re-encoding RGB.  It never computes
or changes a reward: candidate membership is read from a frozen result JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA = "bernini-reward-selected-synthetic-target-dataset-summary-v1"
INDEX_SCHEMA = "bernini-r-action-vae-index-row-v2"
ROW_SCHEMA = "bernini-reward-selected-synthetic-target-row-v1"
REWARD_RESULT_SCHEMA = "action-editing-reward-ablation-result-v1"
ARMS = ("baseline", "action_only", "preservation_only", "composite")
LATENT_KEY = "normalized_clean_latent"
POSTERIOR_LOGVAR = -30.0


class MaterializationError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MaterializationError(f"{label} must be a plain file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must contain one object")
    return value


def _checkpoint_statistics(checkpoint: Path):
    import torch

    config = _load_object(checkpoint / "vae" / "config.json", label="VAE config")
    means, stds = config.get("latents_mean"), config.get("latents_std")
    if config.get("z_dim") != 16 or not isinstance(means, list) or not isinstance(stds, list):
        raise MaterializationError("checkpoint VAE statistics differ")
    mean = torch.tensor(means, dtype=torch.float32).view(1, 16, 1, 1, 1)
    std = torch.tensor(stds, dtype=torch.float32).view(1, 16, 1, 1, 1)
    if tuple(mean.shape) != (1, 16, 1, 1, 1) or bool((std <= 0).any()):
        raise MaterializationError("checkpoint VAE statistics are invalid")
    return mean, std


def _load_normalized_latent(path: Path):
    from safetensors.torch import load_file

    if not path.is_file() or path.is_symlink():
        raise MaterializationError(f"latent must be a plain file: {path}")
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {LATENT_KEY}:
        raise MaterializationError(f"latent keys differ: {path}: {sorted(tensors)}")
    value = tensors[LATENT_KEY].detach().float().contiguous()
    if value.ndim != 5 or tuple(value.shape[:3]) != (1, 16, 21):
        raise MaterializationError(f"latent geometry differs: {path}: {tuple(value.shape)}")
    if not bool(value.isfinite().all()):
        raise MaterializationError(f"latent contains non-finite values: {path}")
    return value


def _posterior_blob(normalized, mean, std) -> bytes:
    import torch

    raw_mean = (normalized * std + mean).float().contiguous()
    logvar = torch.full_like(raw_mean, POSTERIOR_LOGVAR)
    parameters = torch.cat((raw_mean, logvar), dim=1).contiguous()
    buffer = io.BytesIO()
    torch.save(parameters, buffer)
    return buffer.getvalue()


def _messages(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip() or "\x00" in instruction:
        raise MaterializationError("instruction must be non-empty text")
    return canonical_json_bytes(
        [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": instruction, "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]
    ).decode("utf-8")


def materialize(*, result_path: Path, checkpoint: Path, arm: str, output_root: Path) -> dict[str, Any]:
    if arm not in ARMS:
        raise MaterializationError(f"arm must be one of {ARMS}")
    if output_root.exists():
        raise MaterializationError(f"create-only output exists: {output_root}")
    result = _load_object(result_path, label="reward result")
    if (
        result.get("schema_version") != REWARD_RESULT_SCHEMA
        or result.get("authority", {}).get("machine_selection_is_ground_truth") is not False
        or result.get("authority", {}).get("qwen_or_vlm_used") is not False
    ):
        raise MaterializationError("reward result authority differs")
    groups = result.get("groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise MaterializationError("reward result must contain exactly four groups")
    mean, std = _checkpoint_statistics(checkpoint)
    shards = output_root / "shards"
    shards.mkdir(parents=True)
    index_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    import pyarrow as pa
    import pyarrow.parquet as pq

    for group in groups:
        iid = group.get("iid")
        instruction = group.get("instruction")
        arm_value = group.get("arms", {}).get(arm)
        candidate_id = arm_value.get("candidate_id") if isinstance(arm_value, Mapping) else None
        candidates = group.get("candidates")
        if not isinstance(iid, str) or not re.fullmatch(r"[0-9a-f]{16}", iid):
            raise MaterializationError("group IID differs")
        if not isinstance(candidates, list):
            raise MaterializationError(f"candidate list differs for {iid}")
        matches = [value for value in candidates if value.get("candidate_id") == candidate_id]
        if len(matches) != 1:
            raise MaterializationError(f"selected candidate is not unique for {iid}: {candidate_id}")
        media = matches[0].get("media")
        target_video = Path(str(media.get("path"))).resolve(strict=True)
        if file_sha256(target_video) != media.get("sha256"):
            raise MaterializationError(f"candidate media hash differs for {iid}")
        candidate_root = target_video.parent
        source_latent_path = candidate_root / "source.normalized-clean-latent.safetensors"
        target_latent_path = candidate_root / "rv2v.normalized-clean-latent.safetensors"
        source_latent = _load_normalized_latent(source_latent_path)
        target_latent = _load_normalized_latent(target_latent_path)
        if tuple(source_latent.shape) != tuple(target_latent.shape):
            raise MaterializationError(f"source/target latent geometry differs for {iid}")
        source_blob = _posterior_blob(source_latent, mean, std)
        target_blob = _posterior_blob(target_latent, mean, std)
        row = {
            "schema_version": ROW_SCHEMA,
            "iid": iid,
            "inputs": _messages(instruction),
            "video_vae_latents": [source_blob, target_blob],
            "reward_arm": arm,
            "selected_candidate_id": candidate_id,
            "source_normalized_clean_latent_sha256": file_sha256(source_latent_path),
            "target_normalized_clean_latent_sha256": file_sha256(target_latent_path),
            "candidate_video_sha256": media["sha256"],
            "posterior_logvar": POSTERIOR_LOGVAR,
            "experimental_training_acknowledged": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        shard = shards / f"{iid}.parquet"
        pq.write_table(pa.Table.from_pylist([row]), shard, compression="zstd", use_dictionary=False)
        shard_sha = file_sha256(shard)
        index_rows.append(
            {
                "schema_version": INDEX_SCHEMA,
                "iid": iid,
                "parquet_path": str(shard.resolve()),
                "parquet_sha256": shard_sha,
            }
        )
        bucket = f"{int(source_latent.shape[-2])}x{int(source_latent.shape[-1])}-latent"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        selected.append(
            {
                "iid": iid,
                "candidate_id": candidate_id,
                "candidate_video_path": str(target_video),
                "candidate_video_sha256": media["sha256"],
                "source_latent_path": str(source_latent_path),
                "target_latent_path": str(target_latent_path),
            }
        )
    index_path = output_root / "dataset_index.jsonl"
    index_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in index_rows))
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "complete": True,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "experimental_training_acknowledged": True,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "reward_selected_synthetic_target": True,
        "same_source_instruction_rows_across_arms": True,
        "arm": arm,
        "reward_result_path": str(result_path.resolve()),
        "reward_result_sha256": file_sha256(result_path),
        "reward_result_digest": result.get("result_digest"),
        "expected_sample_count": 4,
        "materialized_sample_count": 4,
        "missing_sample_count": 0,
        "frame_count": 81,
        "fps": 25.0,
        "latent_frame_count": 21,
        "posterior_construction": "inverse_checkpoint_normalization_plus_logvar_-30",
        "bucket_counts": bucket_counts,
        "shards_directory": str(shards.resolve()),
        "index_path": str(index_path.resolve()),
        "index_sha256": file_sha256(index_path),
        "selected_candidates": selected,
    }
    summary["summary_digest"] = object_sha256(summary)
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = materialize(
        result_path=args.reward_result.expanduser().resolve(strict=True),
        checkpoint=args.checkpoint.expanduser().resolve(strict=True),
        arm=args.arm,
        output_root=args.output_root.expanduser().absolute(),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

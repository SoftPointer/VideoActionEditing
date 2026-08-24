#!/usr/bin/env python3
"""Rescore frozen and trained complex8 decodes with the frozen action critic."""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native_generation
import pair_v5_native_bridge as native_bridge
import score_interaction_complex8_reward_v1 as reward
import score_pair_v5_t2v_energy_bank_v3 as frozen_runtime


SCHEMA_VERSION = "bernini-interaction-complex8-trained-action-score-v1"
ARMS = (
    ("dpo_only_s4", 4),
    ("dpo_identity005_s4", 4),
    ("dpo_identity015_s4", 4),
    ("dpo_identity010_s8", 8),
)


class TrainedDecodeScoreError(RuntimeError):
    pass


def plain_json(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TrainedDecodeScoreError(f"missing absolute plain JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainedDecodeScoreError(f"JSON root differs: {path}")
    return value


def tensor_artifact(path: Path) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TrainedDecodeScoreError(f"missing tensor artifact: {path}")
    return {"path": str(path), "sha256": reward.file_sha256(path)}


def records_for_event(
    event: int, rollout_root: Path, decode_root: Path
) -> list[dict[str, Any]]:
    candidate_id = f"complex8-e{event:02d}-rv2v-s0"
    candidate_receipt = plain_json(
        rollout_root / candidate_id / "pair-v5-rollout-receipt.json"
    )
    artifacts = candidate_receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TrainedDecodeScoreError("frozen candidate artifacts are absent")
    rows = [
        {
            "label": "frozen_rv2v_s0",
            "arm": "frozen",
            "step": 0,
            "clean": dict(artifacts["predecode_clean_latent"]),
            "noise": dict(artifacts["official_initial_gaussian"]),
            "video": dict(artifacts["mp4"]),
        }
    ]
    for arm, final_step in ARMS:
        for step in (1, final_step):
            directory = decode_root / arm / f"event_{event:02d}" / f"step_{step:04d}"
            inference_receipt = plain_json(directory / "adapter-inference-receipt.json")
            if (
                inference_receipt.get("complete") is not True
                or inference_receipt.get("input_closure", {}).get(
                    "pure_t2v_anchor_loaded"
                )
                is not False
            ):
                raise TrainedDecodeScoreError(
                    f"trained decode receipt closure differs: {directory}"
                )
            video = directory / "rv2v.mp4"
            rows.append(
                {
                    "label": f"{arm}_step_{step:04d}",
                    "arm": arm,
                    "step": step,
                    "clean": tensor_artifact(
                        directory / "rv2v.normalized-clean-latent.safetensors"
                    ),
                    "noise": tensor_artifact(
                        directory / "rv2v.official-initial-gaussian.safetensors"
                    ),
                    "video": {
                        "path": str(video),
                        "sha256": reward.file_sha256(video),
                    },
                }
            )
    if len(rows) != 9 or len({row["label"] for row in rows}) != len(rows):
        raise TrainedDecodeScoreError("trained action score record closure differs")
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--event", type=int, choices=range(8), required=True)
    result.add_argument("--authoring-manifest", required=True)
    result.add_argument("--rollout-root", required=True)
    result.add_argument("--decode-root", required=True)
    result.add_argument("--bernini-root", required=True)
    result.add_argument("--veomni-root", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--checkpoint-content-manifest", required=True)
    result.add_argument("--output-dir", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output_dir)
    if not output.is_absolute() or output.exists() or output == Path("/"):
        raise TrainedDecodeScoreError("output-dir must be fresh, absolute and non-root")
    authoring = plain_json(Path(args.authoring_manifest))
    events = reward._event_map(authoring)
    event = events[args.event]
    rollout_root = Path(args.rollout_root).resolve(strict=True)
    decode_root = Path(args.decode_root).resolve(strict=True)
    records = records_for_event(args.event, rollout_root, decode_root)

    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, _, _ = legacy.trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.trainer.VEOMNI_TESTED_COMMIT,
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(
            args.checkpoint
        )
    except legacy.trainer.TrainingContractError as error:
        raise TrainedDecodeScoreError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise TrainedDecodeScoreError("Bernini attention-head count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer

    distributed = legacy.inference_distributed_contract()
    torch.cuda.set_device(distributed.local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=180),
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    init_parallel_state(ulysses_size=4)
    device = torch.device("cuda", distributed.local_rank)
    checkpoint_rows: list[Any] = [None]
    if distributed.rank == 0:
        checkpoint_rows[0] = native_generation.source_audit.validate_checkpoint_content(
            checkpoint, Path(args.checkpoint_content_manifest)
        )
    dist.broadcast_object_list(checkpoint_rows, src=0)
    checkpoint_identity = checkpoint_rows[0]
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    renderer = BerniniRendererModel(config).requires_grad_(False).eval().to(device)
    freeze_before = native_generation.source_audit.model_freeze_certificate(renderer)
    diffusion = renderer.diff_dec
    transformer = diffusion.transformer
    if transformer is None or diffusion.transformer_2 is not None:
        raise TrainedDecodeScoreError("action critic requires frozen transformer_1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    captions = reward.prompt_captions(event)
    prompts = frozen_runtime.official_prompt_bank_from_captions(
        captions, prompt_cleaner=prompt_clean
    )
    conditions = frozen_runtime._encode_prompt_bank(
        renderer, tokenizer, prompts, device=device
    )
    scorer = frozen_runtime.NativeExact40FrozenBerniniT2VScorer(
        diffusion,
        transformer,
        prompts,
        conditions,
        frozen_model_receipt_digest=reward.object_sha256(checkpoint_identity),
        model_id="transformer_1",
    )
    sigma = torch.tensor(
        [frozen_runtime.PILOT_SIGMA], dtype=torch.float32, device=device
    )
    phase_commitment = frozen_runtime.diagnostic_phase_commitment()
    scored: list[dict[str, Any]] = []
    for row in records:
        clean = frozen_runtime._load_exact81_tensor(
            row["clean"], key="normalized_clean_latent", label=f"{row['label']} clean"
        ).to(device=device).contiguous()
        epsilon = frozen_runtime._load_exact81_tensor(
            row["noise"], key="official_initial_gaussian", label=f"{row['label']} noise"
        ).to(device=device).contiguous()
        result = native_bridge.score_frozen_t2v_action_energy(
            clean,
            epsilon,
            sigma,
            prompts,
            scorer,
            phase_commitment,
            registered_phase_weight_digest=phase_commitment["registration_digest"],
        )
        score = reward.score_record(result)
        if not math.isfinite(score["phase_conjunctive_reward"]):
            raise TrainedDecodeScoreError("trained action reward is non-finite")
        scored.append(
            {
                "label": row["label"],
                "arm": row["arm"],
                "step": row["step"],
                "video": row["video"],
                "score": score,
            }
        )
        del clean, epsilon, result
    freeze_after = native_generation.source_audit.model_freeze_certificate(renderer)
    if freeze_after != freeze_before:
        raise TrainedDecodeScoreError("frozen action critic changed during scoring")
    score_digest = reward.object_sha256(scored)
    gathered: list[Any] = [None] * distributed.world_size
    dist.all_gather_object(gathered, score_digest)
    if len(set(gathered)) != 1:
        raise TrainedDecodeScoreError("SP4 trained action score digests differ")
    if distributed.rank == 0:
        baseline = scored[0]["score"]["phase_conjunctive_reward"]
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "event_ordinal": args.event,
            "event_id": event["event_id"],
            "category": event["category"],
            "rows": [
                {
                    **row,
                    "phase_reward_delta_from_frozen": row["score"][
                        "phase_conjunctive_reward"
                    ]
                    - baseline,
                }
                for row in scored
            ],
            "input_closure": {
                "pure_t2v_anchor_loaded": False,
                "pure_t2v_anchor_appearance_loaded": False,
                "decoded_candidate_own_clean_and_noise_only": True,
                "qwen_or_vlm_used": False,
                "frozen_action_critic_changed": False,
            },
        }
        receipt = {
            **unsigned,
            "receipt_digest": reward.object_sha256(unsigned),
        }
        output.mkdir(parents=True)
        (output / "trained-action-score.json").write_bytes(
            reward.canonical_bytes(receipt) + b"\n"
        )
        (output / "COMPLETE").touch()
    del scorer, conditions, renderer, diffusion, transformer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

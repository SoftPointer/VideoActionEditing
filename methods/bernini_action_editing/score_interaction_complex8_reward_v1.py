#!/usr/bin/env python3
"""Score complex8 anchors and native RV2V candidates without anchor leakage.

Action is measured in each candidate's own clean-latent/noise coordinate by a
frozen Bernini T2V energy critic.  Source preservation is evaluated separately
from decoded candidate/source videos by a frozen DINOv2 model.  Pure-T2V anchor
pixels, latents, identity and appearance never enter candidate generation,
candidate preservation, or a training target.  They are used only to test
whether the action critic rejects deterministic noop/reverse/incomplete
counterfactuals of the same generated appearance.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import infer_native_identity_generation_canary as native_generation
import mace_candidate_action_energy as mace
import pair_v5_native_bridge as native_bridge
import pair_v5_native_rollout_spec as rollout_contract
import score_pair_v5_source_bound_preservation_v1 as preservation
import score_pair_v5_t2v_energy_bank_v3 as frozen_mace_runtime


SCHEMA_VERSION = "bernini-interaction-complex8-reward-group-v1"
PILOT_SIGMA = frozen_mace_runtime.PILOT_SIGMA
_CANDIDATE_ID = re.compile(r"complex8-e(?P<event>[0-7][0-9]?)-rv2v-s(?P<variant>[0-3])\Z")


class Complex8RewardError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_json(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_absolute() or not value.is_file() or value.is_symlink():
        raise Complex8RewardError(f"JSON input is not an absolute plain file: {value}")
    result = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise Complex8RewardError("JSON root must be an object")
    return result


def prompt_captions(event: Mapping[str, Any]) -> dict[str, str]:
    action = str(event["action"]).strip()
    category = str(event["category"]).strip()
    rows = {
        "action": f"A coherent continuous video in which this event is completed in causal order: {action}",
        "noop": f"A continuous video in which all participants remain in the initial state and never perform the requested {category} event.",
        "incomplete": f"A continuous video in which the requested {category} event begins but stops before contact, transfer, consequence, or stable terminal completion.",
        "reverse": f"A continuous video in which the requested {category} event occurs in reverse temporal and causal order, ending at its initial state.",
        "shuffle": f"A continuous video in which the phases of the requested {category} event are shuffled into an impossible noncausal order.",
        "wrong_actor": f"A continuous video in which a different participant performs the requested {category} event while the designated participant remains unchanged.",
        "wrong_object": f"A continuous video in which the motion acts on a different object and the designated object never undergoes the requested {category} transition.",
        "camera_only": f"A continuous video with camera pan, zoom, or shake but no participant completes the requested {category} event.",
        "appearance_only": f"A continuous video with clothing, fur, color, texture, or background changes but no requested {category} action occurs.",
        "generic_wrong_motion": f"A continuous video showing unrelated generic motion instead of the requested {category} event.",
    }
    if set(rows) != set(mace.BRANCH_ORDER) or len(set(rows.values())) != len(rows):
        raise Complex8RewardError("action prompt registry closure differs")
    return rows


def parse_candidate_id(value: str) -> tuple[int, int]:
    match = _CANDIDATE_ID.fullmatch(value)
    if match is None:
        raise Complex8RewardError(f"candidate ID differs: {value}")
    event, variant = int(match.group("event")), int(match.group("variant"))
    if not 0 <= event < 8:
        raise Complex8RewardError("candidate event lies outside complex8")
    return event, variant


def load_group_candidates(
    spec_path: Path, expected_sha256: str, group_id: str
) -> list[dict[str, Any]]:
    spec, _ = rollout_contract.load_sealed_spec(spec_path, expected_sha256)
    groups = {row["group_id"]: row for row in spec["groups"]}
    if set(groups) != {"sp4-a", "sp4-b"} or group_id not in groups:
        raise Complex8RewardError("candidate group closure differs")
    rows = [dict(row) for row in groups[group_id]["candidates"]]
    parsed = [parse_candidate_id(row["candidate_id"]) for row in rows]
    events = sorted({event for event, _ in parsed})
    variants = sorted({variant for _, variant in parsed})
    if len(rows) != 4 or len(events) != 2 or len(variants) != 2:
        raise Complex8RewardError("group must contain two events by two variants")
    if any(sum(event == wanted for event, _ in parsed) != 2 for wanted in events):
        raise Complex8RewardError("event candidate multiplicity differs")
    return rows


def _event_map(authoring: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    if authoring.get("schema_version") != "bernini-interaction-complex8-multianchor-authoring-v2":
        raise Complex8RewardError("anchor authoring schema differs")
    events = {int(row["ordinal"]): dict(row) for row in authoring.get("events", ())}
    if set(events) != set(range(8)):
        raise Complex8RewardError("complex8 event closure differs")
    return events


def _candidate_receipt(root: Path, candidate: Mapping[str, Any]) -> dict[str, Any]:
    directory = root / str(candidate["candidate_id"])
    receipt = _plain_json(directory / "pair-v5-rollout-receipt.json")
    if (
        receipt.get("schema_version") != rollout_contract.RECEIPT_SCHEMA_VERSION
        or receipt.get("candidate") != candidate
        or receipt.get("semantic_input_closure") != rollout_contract.SEMANTIC_INPUT_CLOSURE
        or receipt.get("sampling_contract") != rollout_contract.SAMPLING_CONTRACT
    ):
        raise Complex8RewardError("native candidate receipt binding differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "mp4", "predecode_clean_latent", "official_initial_gaussian"
    }:
        raise Complex8RewardError("native candidate artifact closure differs")
    for artifact in artifacts.values():
        path = Path(str(artifact["path"]))
        if not path.is_file() or path.is_symlink() or file_sha256(path) != artifact["sha256"]:
            raise Complex8RewardError("native candidate artifact hash differs")
    return receipt


def _tensor_artifacts_from_anchor(
    anchor_root: Path, event: Mapping[str, Any], variant: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    directory = anchor_root / f"e{int(event['ordinal']):02d}_{event['event_id']}" / f"v{variant}"
    receipt = _plain_json(directory / "receipt.json")
    clean = receipt.get("outputs", {}).get("t2v", {}).get("normalized_clean_latent")
    noise = receipt.get("initial_noise_artifacts", {}).get("t2v")
    if not isinstance(clean, Mapping) or not isinstance(noise, Mapping):
        raise Complex8RewardError("anchor native tensor artifacts differ")
    return clean, noise


def temporal_role_latent(action: Any, role: str) -> Any:
    import torch

    if tuple(action.shape[:3]) != (1, 16, 21):
        raise Complex8RewardError("action latent is not exact81 normalized clean state")
    if role == "action":
        result = action
    elif role == "noop":
        result = action[:, :, :1].expand(-1, -1, 21, -1, -1).clone()
    elif role == "reverse":
        result = torch.flip(action, dims=(2,)).contiguous()
    elif role == "incomplete":
        result = torch.cat(
            (action[:, :, :11], action[:, :, 10:11].expand(-1, -1, 10, -1, -1)),
            dim=2,
        ).contiguous()
    else:
        raise Complex8RewardError(f"unsupported temporal role: {role}")
    if result.dtype != torch.float32 or result.requires_grad or not bool(torch.isfinite(result).all()):
        raise Complex8RewardError("derived temporal role latent differs")
    return result.detach()


def score_record(result: Any) -> dict[str, Any]:
    reward = float(result.phase_energy.reward[0].item())
    global_reward = float(result.energy.reward[0].item())
    milestone = [float(row[0].item()) for row in result.phase_energy.milestone_rewards]
    values = [reward, global_reward, *milestone]
    if any(not math.isfinite(value) for value in values):
        raise Complex8RewardError("frozen action reward is non-finite")
    return {
        "phase_conjunctive_reward": reward,
        "global_reward": global_reward,
        "milestone_rewards": milestone,
        "hardest_global_negative": mace.HARD_NEGATIVE_BRANCHES[
            int(result.energy.hardest_negative_index[0].item())
        ],
        "branch_energies": {
            branch: float(result.energy.branch_energies[index, 0].item())
            for index, branch in enumerate(mace.BRANCH_ORDER)
        },
        "scorer_receipt_digest": result.receipt["digest"],
    }


def _mapped_frame_similarity(
    left: Any, right: Any, *, dense: bool, quantile: float | None = None
) -> float:
    import torch

    similarity = ((left * right).sum(dim=-1) + 1.0) * 0.5
    similarity = torch.clamp(similarity, 0.0, 1.0)
    flattened = similarity.reshape(-1)
    if quantile is not None:
        if not dense or not 0.0 <= quantile <= 1.0:
            raise Complex8RewardError("dense similarity quantile differs")
        value = torch.quantile(flattened, quantile)
    else:
        value = flattened.median() if dense else flattened.mean()
    return float(value.item())


def _dino_features(
    path: Path,
    sha256: str,
    *,
    model: Any,
    processor: Any,
    device: Any,
    register_tokens: int,
) -> dict[str, Any]:
    frames, decode_receipt = preservation.decode_exact81_rgb(path, expected_sha256=sha256)
    raw, normalized = preservation.preprocess_selected_rgb(frames, processor)
    global_feature, dense_feature, feature_receipt = preservation.extract_features(
        model,
        normalized,
        device=device,
        num_register_tokens=register_tokens,
    )
    return {
        "raw": raw,
        "global": global_feature,
        "dense": dense_feature,
        "decode": decode_receipt,
        "feature": feature_receipt,
    }


def preservation_record(candidate: Mapping[str, Any], receipt: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    event, _ = parse_candidate_id(str(candidate["candidate_id"]))
    own = features[str(candidate["candidate_id"])]
    correct = features[f"source:{event}"]
    other_events = sorted(
        int(key.split(":", 1)[1]) for key in features if key.startswith("source:") and key != f"source:{event}"
    )
    if len(other_events) != 1:
        raise Complex8RewardError("wrong-source feature closure differs")
    wrong = features[f"source:{other_events[0]}"]
    metrics = preservation.compute_metrics(
        candidate_global=own["global"],
        candidate_dense=own["dense"],
        correct_global=correct["global"],
        correct_dense=correct["dense"],
        wrong_global=wrong["global"],
        wrong_dense=wrong["dense"],
        candidate_raw=own["raw"],
        correct_raw=correct["raw"],
    )
    first_global = _mapped_frame_similarity(own["global"][0], correct["global"][0], dense=False)
    first_global_wrong = _mapped_frame_similarity(own["global"][0], wrong["global"][0], dense=False)
    first_dense = _mapped_frame_similarity(own["dense"][0], correct["dense"][0], dense=True)
    first_dense_wrong = _mapped_frame_similarity(own["dense"][0], wrong["dense"][0], dense=True)
    first_dense_p10 = _mapped_frame_similarity(
        own["dense"][0], correct["dense"][0], dense=True, quantile=0.10
    )
    first_dense_p10_wrong = _mapped_frame_similarity(
        own["dense"][0], wrong["dense"][0], dense=True, quantile=0.10
    )
    first_rgb_l1_similarity = float(
        1.0 - (own["raw"][0] - correct["raw"][0]).abs().mean().item()
    )
    checks = {
        "identity_correct_above_wrong": metrics["source_identity_appearance_correct_minus_wrong_margin"] > 0.0,
        "background_correct_above_wrong": metrics["background_appearance_correct_minus_wrong_margin"] > 0.0,
        "layout_correct_above_wrong": metrics["source_bound_spatial_layout_correct_minus_wrong_margin"] > 0.0,
        "initial_global_correct_above_wrong": first_global > first_global_wrong,
        "initial_dense_correct_above_wrong": first_dense > first_dense_wrong,
        "initial_dense_p10_correct_above_wrong": first_dense_p10 > first_dense_p10_wrong,
        "absolute_identity_floor": metrics["source_identity_appearance_proxy"] >= 0.60,
        "absolute_initial_global_floor": first_global >= 0.60,
        "absolute_initial_dense_median_floor": first_dense >= 0.50,
        "absolute_initial_dense_p10_floor": first_dense_p10 >= 0.35,
        "absolute_initial_rgb_l1_floor": first_rgb_l1_similarity >= 0.60,
        "decoded_quality_floor": metrics["decode_video_quality_diagnostic"] >= 0.35,
    }
    mp4 = receipt["artifacts"]["mp4"]
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_video_path": mp4["path"],
        "candidate_video_sha256": mp4["sha256"],
        "source_video_path": candidate["source_video"],
        "source_video_sha256": candidate["source_video_sha256"],
        "metrics": metrics,
        "initial_state": {
            "global_correct": first_global,
            "global_wrong": first_global_wrong,
            "dense_correct": first_dense,
            "dense_wrong": first_dense_wrong,
            "dense_p10_correct": first_dense_p10,
            "dense_p10_wrong": first_dense_p10_wrong,
            "rgb_l1_similarity_correct": first_rgb_l1_similarity,
        },
        "hard_gate_checks": checks,
        "hard_gate_pass": all(checks.values()),
        "weighted_preservation_score_used": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoring-manifest", required=True)
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--expected-candidate-spec-sha256", required=True)
    parser.add_argument("--group-id", choices=("sp4-a", "sp4-b"), required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--visual-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir)
    if not output.is_absolute() or output == Path("/") or output.exists():
        raise Complex8RewardError("output-dir must be a fresh absolute non-root directory")
    authoring = _plain_json(args.authoring_manifest)
    events = _event_map(authoring)
    spec_path = Path(args.candidate_spec)
    candidates = load_group_candidates(spec_path, args.expected_candidate_spec_sha256, args.group_id)
    candidate_events = sorted({parse_candidate_id(row["candidate_id"])[0] for row in candidates})
    variants = sorted({parse_candidate_id(row["candidate_id"])[1] for row in candidates})
    anchor_root = Path(args.anchor_root).resolve(strict=True)
    rollout_root = Path(args.rollout_root).resolve(strict=True)
    receipts = {row["candidate_id"]: _candidate_receipt(rollout_root, row) for row in candidates}

    legacy = native_generation.legacy
    try:
        bernini_root, veomni_root, _, _ = legacy.trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=legacy.trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.trainer.VEOMNI_TESTED_COMMIT,
        )
        checkpoint, transformer_config = legacy.trainer.validate_checkpoint(args.checkpoint)
    except legacy.trainer.TrainingContractError as error:
        raise Complex8RewardError(str(error)) from error
    if transformer_config.get("num_attention_heads") != 12:
        raise Complex8RewardError("Bernini attention-head count differs")
    legacy.trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.parallel import init_parallel_state

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
        raise Complex8RewardError("action critic requires frozen transformer_1")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), subfolder="tokenizer", **legacy.tokenizer_load_kwargs()
    )
    sigma = torch.tensor([PILOT_SIGMA], dtype=torch.float32, device=device)
    phase_commitment = frozen_mace_runtime.diagnostic_phase_commitment()
    action_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    for event_index in candidate_events:
        captions = prompt_captions(events[event_index])
        prompts = frozen_mace_runtime.official_prompt_bank_from_captions(
            captions, prompt_cleaner=prompt_clean
        )
        conditions = frozen_mace_runtime._encode_prompt_bank(
            renderer, tokenizer, prompts, device=device
        )
        scorer = frozen_mace_runtime.NativeExact40FrozenBerniniT2VScorer(
            diffusion,
            transformer,
            prompts,
            conditions,
            frozen_model_receipt_digest=object_sha256(checkpoint_identity),
            model_id="transformer_1",
        )
        for variant in variants:
            clean_artifact, noise_artifact = _tensor_artifacts_from_anchor(
                anchor_root, events[event_index], variant
            )
            action_clean = frozen_mace_runtime._load_exact81_tensor(
                clean_artifact, key="normalized_clean_latent", label="complex8 action anchor"
            ).to(device=device).contiguous()
            epsilon = frozen_mace_runtime._load_exact81_tensor(
                noise_artifact, key="official_initial_gaussian", label="complex8 anchor noise"
            ).to(device=device).contiguous()
            for role in ("action", "noop", "reverse", "incomplete"):
                clean = temporal_role_latent(action_clean, role)
                result = native_bridge.score_frozen_t2v_action_energy(
                    clean,
                    epsilon,
                    sigma,
                    prompts,
                    scorer,
                    phase_commitment,
                    registered_phase_weight_digest=phase_commitment["registration_digest"],
                )
                anchor_rows.append(
                    {
                        "event_ordinal": event_index,
                        "variant": variant,
                        "role": role,
                        "score": score_record(result),
                    }
                )
        for candidate in candidates:
            candidate_event, _ = parse_candidate_id(candidate["candidate_id"])
            if candidate_event != event_index:
                continue
            receipt = receipts[candidate["candidate_id"]]
            clean = frozen_mace_runtime._load_exact81_tensor(
                receipt["artifacts"]["predecode_clean_latent"],
                key="normalized_clean_latent",
                label=f"{candidate['candidate_id']} clean latent",
            ).to(device=device).contiguous()
            epsilon = frozen_mace_runtime._load_exact81_tensor(
                receipt["artifacts"]["official_initial_gaussian"],
                key="official_initial_gaussian",
                label=f"{candidate['candidate_id']} official noise",
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
            action_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "event_ordinal": event_index,
                    "score": score_record(result),
                }
            )
    freeze_after = native_generation.source_audit.model_freeze_certificate(renderer)
    if freeze_after != freeze_before:
        raise Complex8RewardError("frozen action critic changed during scoring")
    action_digest = object_sha256({"anchors": anchor_rows, "candidates": action_rows})
    gathered: list[Any] = [None] * distributed.world_size
    dist.all_gather_object(gathered, action_digest)
    if len(set(gathered)) != 1:
        raise Complex8RewardError("SP4 action score digests differ")

    del scorer, conditions, result, clean, epsilon, action_clean, prompts, captions
    del renderer, diffusion, transformer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if distributed.rank == 0:
        visual_root = Path(args.visual_checkpoint).resolve(strict=True)
        processor = AutoImageProcessor.from_pretrained(
            str(visual_root), local_files_only=True, trust_remote_code=False, use_fast=False
        )
        model = AutoModel.from_pretrained(
            str(visual_root), local_files_only=True, trust_remote_code=False,
            attn_implementation="eager"
        ).eval().requires_grad_(False).to(device=device, dtype=torch.float32)
        register_tokens = int(getattr(model.config, "num_register_tokens", 0))
        features: dict[str, Any] = {}
        for event_index in candidate_events:
            event_candidates = [row for row in candidates if parse_candidate_id(row["candidate_id"])[0] == event_index]
            source = event_candidates[0]
            if any(row["source_video_sha256"] != source["source_video_sha256"] for row in event_candidates):
                raise Complex8RewardError("same-event source binding differs")
            features[f"source:{event_index}"] = _dino_features(
                Path(source["source_video"]), source["source_video_sha256"],
                model=model, processor=processor, device=device, register_tokens=register_tokens,
            )
        for candidate in candidates:
            receipt = receipts[candidate["candidate_id"]]
            mp4 = receipt["artifacts"]["mp4"]
            features[candidate["candidate_id"]] = _dino_features(
                Path(mp4["path"]), mp4["sha256"],
                model=model, processor=processor, device=device, register_tokens=register_tokens,
            )
        preservation_rows = [
            preservation_record(row, receipts[row["candidate_id"]], features)
            for row in candidates
        ]
        calibration_rows = []
        for event_index in candidate_events:
            for variant in variants:
                rows = {
                    row["role"]: row
                    for row in anchor_rows
                    if row["event_ordinal"] == event_index and row["variant"] == variant
                }
                action_reward = rows["action"]["score"]["phase_conjunctive_reward"]
                comparisons = {
                    role: action_reward > rows[role]["score"]["phase_conjunctive_reward"]
                    for role in ("noop", "reverse", "incomplete")
                }
                calibration_rows.append(
                    {
                        "event_ordinal": event_index,
                        "variant": variant,
                        "action_reward": action_reward,
                        "negative_reward": {
                            role: rows[role]["score"]["phase_conjunctive_reward"]
                            for role in comparisons
                        },
                        "pairwise_pass": comparisons,
                    }
                )
        comparison_count = len(calibration_rows) * 3
        comparison_pass = sum(
            int(value) for row in calibration_rows for value in row["pairwise_pass"].values()
        )
        unsigned = {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "group_id": args.group_id,
            "candidate_spec_path": str(spec_path),
            "candidate_spec_sha256": args.expected_candidate_spec_sha256,
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "event_ordinals": candidate_events,
            "anchor_variants": variants,
            "anchor_action_validation": {
                "rows": calibration_rows,
                "pairwise_pass_count": comparison_pass,
                "pairwise_comparison_count": comparison_count,
                "pairwise_accuracy": comparison_pass / comparison_count,
            },
            "candidate_action_scores": action_rows,
            "candidate_preservation": preservation_rows,
            "input_closure": {
                "t2v_anchor_used_as_candidate_generation_input": False,
                "t2v_anchor_used_as_candidate_training_target": False,
                "t2v_anchor_appearance_used_by_candidate_scorer": False,
                "action_scorer_candidate_own_clean_and_noise_only": True,
                "preservation_scorer_source_and_candidate_rgb_only": True,
                "qwen_or_vlm_used": False,
                "training_performed": False,
            },
        }
        receipt = {**unsigned, "receipt_digest": object_sha256(unsigned)}
        output.mkdir(parents=True)
        (output / "reward-group.json").write_bytes(canonical_bytes(receipt) + b"\n")
        (output / "COMPLETE").touch()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

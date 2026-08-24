#!/usr/bin/env python3
"""Train the Bernini Prior-Preserving Phase-Plan Teacher LoRA bridge.

This is the first falsifiable motion/identity prototype, not a claim that the
current synthetic targets are accepted ground truth.  Every optimizer step has
two cells with one source posterior mode, one sigma, and one diffusion noise:

* action: source + edit instruction -> target-mode noisy velocity;
* copy:   same source + exact-no-op instruction -> source-mode noisy velocity.

The action cell is supervised through a projected source-relative field and
multi-lag differences.  A temporally constant source-to-target appearance
replacement (the failure seen in the dog example) is not rewarded by a full
framewise target loss.  Frozen-base replay supplies the discarded complement.

The frozen upstream Bernini source is imported byte-for-byte.  We extract its
raw velocity prediction in a local wrapper so the pinned renderer is never
patched.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import p3t  # noqa: E402
import train_lora as legacy  # noqa: E402


RECEIPT_SCHEMA = "bernini-r-1p3b-p3t-lora-receipt-v1"
OPTIMIZER_SCHEMA = "bernini-r-1p3b-p3t-lora-optimizer-v1"
METHOD_NAME = "prior-preserving-phase-plan-teacher-lora-v1-diagnostic"
DEFAULT_TEMPORAL_LAGS = (1, 2, 4)


class DeltaTrainingError(RuntimeError):
    """Raised when a P3T-LoRA training invariant is violated."""


@dataclass(frozen=True)
class AdapterPaths:
    root: Path
    adapter: Path
    optimizer: Optional[Path]
    receipt: Optional[Path]


def _translate(error: Exception) -> DeltaTrainingError:
    return DeltaTrainingError(str(error))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeltaTrainingError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise DeltaTrainingError(f"{label} must contain one JSON object")
    return value


def _validate_digest_object(value: Mapping[str, Any], *, label: str) -> None:
    candidate = dict(value)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or legacy.object_sha256(candidate) != declared:
        raise DeltaTrainingError(f"{label} receipt digest differs")


def _resolve_adapter(value: str | Path, *, require_training_state: bool) -> AdapterPaths:
    requested = Path(value).expanduser()
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise DeltaTrainingError(f"adapter checkpoint is unavailable: {error}") from error
    if not root.is_dir():
        raise DeltaTrainingError("adapter checkpoint must be a directory")
    adapter = root / "adapter" if (root / "adapter").is_dir() else root
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise DeltaTrainingError(f"adapter files are incomplete: {adapter}")
    optimizer = root / "optimizer.pt"
    receipt = root / "receipt.json"
    if require_training_state and (not optimizer.is_file() or not receipt.is_file()):
        raise DeltaTrainingError(f"resume state is incomplete: {root}")
    return AdapterPaths(
        root=root,
        adapter=adapter,
        optimizer=optimizer if optimizer.is_file() else None,
        receipt=receipt if receipt.is_file() else None,
    )


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = legacy.canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 81f Bernini P3T LoRA"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--preview-manifest", required=True)
    parser.add_argument("--expected-preview-manifest-sha256", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--routing-jsonl", default=None)
    parser.add_argument(
        "--unreviewed-tier", choices=("motion_only", "reject"), default="motion_only"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-adapter-checkpoint", default=None)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--lora-scope",
        choices=tuple(sorted(motion.MODULE_SCOPES)),
        default="cross_q_out",
    )
    parser.add_argument("--motion-loss-weight", type=float, default=1.0)
    parser.add_argument("--copy-loss-weight", type=float, default=0.25)
    parser.add_argument("--base-replay-loss-weight", type=float, default=0.25)
    parser.add_argument("--complement-distill-loss-weight", type=float, default=0.5)
    parser.add_argument("--plan-teacher-loss-weight", type=float, default=0.25)
    parser.add_argument("--integration-steps", type=int, default=40)
    parser.add_argument("--integration-flow-shift", type=float, default=5.0)
    parser.add_argument("--source-restoration-loss-weight", type=float, default=0.0)
    parser.add_argument("--anchor-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--motion-objective",
        choices=tuple(sorted(motion.MOTION_OBJECTIVES)),
        default="quotient_multilag",
    )
    parser.add_argument("--quotient-weight", type=float, default=0.5)
    parser.add_argument(
        "--temporal-lags",
        type=int,
        nargs="+",
        default=list(DEFAULT_TEMPORAL_LAGS),
    )
    parser.add_argument("--noop-instruction", default=motion.DEFAULT_NOOP_INSTRUCTION)
    parser.add_argument(
        "--expected-bernini-commit", default=legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256", default=legacy.CHECKPOINT_TREE_SHA256
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.num_frames != legacy.NUM_FRAMES:
        raise DeltaTrainingError("only exact 81-frame training is supported")
    if args.max_steps <= 0 or args.save_every < 0:
        raise DeltaTrainingError("max_steps must be positive and save_every non-negative")
    if args.resume and args.init_adapter_checkpoint:
        raise DeltaTrainingError("resume and init-adapter-checkpoint are mutually exclusive")
    if args.motion_objective not in motion.MOTION_OBJECTIVES:
        raise DeltaTrainingError(f"unknown motion objective: {args.motion_objective!r}")
    for name in ("learning_rate", "max_grad_norm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise DeltaTrainingError(f"{name} must be finite and positive")
    # Zero is a legitimate causal ablation for each loss term.  In particular,
    # copy_loss_weight=0 still runs the no-op forward because that branch
    # defines V(action)-V(no-op); it disables only the explicit copy MSE.
    for name in (
        "weight_decay",
        "motion_loss_weight",
        "copy_loss_weight",
        "anchor_loss_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise DeltaTrainingError(f"{name} must be finite and non-negative")
    if not 0.0 <= args.quotient_weight <= 1.0:
        raise DeltaTrainingError("quotient_weight must lie in [0, 1]")
    lags = tuple(args.temporal_lags)
    if (
        not lags
        or len(set(lags)) != len(lags)
        or any(type(lag) is not int or lag <= 0 or lag >= legacy.LATENT_FRAMES for lag in lags)
    ):
        raise DeltaTrainingError("temporal lags must be unique integers in [1, 20]")
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip() or "\x00" in args.noop_instruction:
        raise DeltaTrainingError("noop instruction must be non-empty text")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", getattr(args, name)) is None:
            raise DeltaTrainingError(f"{name} must be a full SHA-1")
    for name in ("expected_checkpoint_tree_sha256", "method_source_archive_sha256", "expected_preview_manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)) is None:
            raise DeltaTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise DeltaTrainingError("checkpoint identity differs from the audited 1.3B tree")
    if args.lora_scope != "cross_q_out":
        raise DeltaTrainingError("P3T-v1 freezes self-attention/source retrieval and requires cross_q_out")
    if args.routing_jsonl is not None or args.unreviewed_tier != "motion_only":
        raise DeltaTrainingError(
            "P3T-v1 is a motion-only diagnostic and forbids full-pair routing"
        )
    if args.motion_objective != "quotient_multilag":
        raise DeltaTrainingError("P3T-v1 requires quotient_multilag")
    if float(args.anchor_loss_weight) != 0.0:
        raise DeltaTrainingError("P3T-v1 forbids first-frame anchor loss")
    for name in (
        "base_replay_loss_weight",
        "complement_distill_loss_weight",
        "plan_teacher_loss_weight",
        "source_restoration_loss_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise DeltaTrainingError(f"{name} must be finite and non-negative")
    if type(args.integration_steps) is not int or args.integration_steps != 40 or float(args.integration_flow_shift) != 5.0:
        raise DeltaTrainingError("P3T-v1 sigma weighting is fixed to the 40-step shift-5 inference grid")
    if float(args.source_restoration_loss_weight) != 0.0:
        raise DeltaTrainingError(
            "source restoration is isolated in p3t.deterministic_source_corruption; "
            "the audited renderer batch does not expose a safe one-forward restoration cell"
        )


def _sigma_for_batch(noise_scheduler: Any, batch: Mapping[str, Any]) -> Any:
    task = legacy.TASK_SOURCE_NAME.split("$")[0].lower()
    shift_name = task if task in noise_scheduler.shift_config else "default"
    shift = noise_scheduler.shift_config[shift_name]
    scheduler = noise_scheduler.flow_scheduler[shift]["scheduler"]
    timestep = batch["timesteps"].reshape(-1)
    if timestep.numel() != 1:
        raise DeltaTrainingError("one-sample batch must contain one timestep")
    sigma = scheduler.get_noise_sigma(timestep)
    if sigma.reshape(-1).numel() != 1:
        raise DeltaTrainingError("one-sample batch must resolve to one sigma")
    return sigma.reshape(-1)


def _iid(raw_row: Mapping[str, Any]) -> str:
    value = raw_row.get("iid", raw_row.get("id"))
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DeltaTrainingError("every routed dataset row must contain a non-empty IID")
    return value


def _build_eligible_routes(
    dataset: Any, router: motion.ReviewRouter
) -> list[tuple[int, motion.Route]]:
    """Freeze a deterministic eligible stream before optimization/resume."""

    eligible: list[tuple[int, motion.Route]] = []
    for row_index in range(len(dataset)):
        route = router.route(_iid(dataset[row_index]))
        if route.tier != "reject":
            eligible.append((row_index, route))
    if not eligible:
        raise DeltaTrainingError("routing rejected every dataset row")
    return eligible


def _next_routed_row(
    dataset: Any,
    eligible: Sequence[tuple[int, motion.Route]],
    *,
    ordinal: int,
) -> tuple[int, Mapping[str, Any], motion.Route]:
    if not eligible:
        raise DeltaTrainingError("eligible route stream is empty")
    row_index, route = eligible[ordinal % len(eligible)]
    row = dataset[row_index]
    if _iid(row) != route.iid:
        raise DeltaTrainingError("dataset/routing membership changed after indexing")
    return row_index, row, route


def _validate_active_supervision(
    args: argparse.Namespace,
    eligible_routes: Sequence[tuple[int, motion.Route]],
) -> None:
    """Reject recipes that can encounter a row with no active training loss."""

    auxiliary_active = any(
        float(value) > 0.0
        for value in (
            args.motion_loss_weight,
            args.copy_loss_weight,
            args.anchor_loss_weight,
            args.base_replay_loss_weight,
            args.complement_distill_loss_weight,
            args.plan_teacher_loss_weight,
        )
    )
    if auxiliary_active:
        return
    inactive = [
        route.iid
        for _, route in eligible_routes
        if route.full_target_weight <= 0.0
    ]
    if inactive:
        raise DeltaTrainingError(
            "training recipe has rows with no active loss; enable motion/copy/anchor "
            "supervision or route every eligible row to positive full-target weight"
        )


def _motion_representation_name(args: argparse.Namespace) -> str:
    if args.motion_objective == "raw_delta":
        return "source-relative-raw-delta-v1"
    if float(args.quotient_weight) == 0.0:
        return "source-relative-multilag-v1"
    if float(args.quotient_weight) == 1.0:
        return "source-relative-temporal-quotient-v1"
    return "source-relative-temporal-quotient-multilag-v1"


def _prepare_paired_batches(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    student_instruction: str,
    compiled_plan: str,
    process_renderer_sample: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = legacy.sanitize_preprocessed_row(raw_row)
    legacy.validate_81_frame_latents(sample, expected_parameter_channels=2 * z_dim)
    student_sample = motion.replace_edit_instruction(
        sample, p3t.compile_generic_phase_wrapper(student_instruction)
    )
    teacher_sample = motion.replace_edit_instruction(sample, compiled_plan)
    copy_sample = motion.replace_edit_instruction(sample, noop_instruction)
    kwargs = dict(
        tokenizer=tokenizer,
        vae_rope_func=rope,
        vae_latent_mean=vae_mean,
        vae_latent_std=vae_std,
        noise_scheduler=scheduler,
        text_dropout_rate=0.0,
        img_dropout_rate=0.0,
        video_dropout_rate=0.0,
        max_vae_frames=legacy.LATENT_FRAMES,
        source_name=legacy.TASK_SOURCE_NAME,
    )
    action_transformed = process_renderer_sample(student_sample, **kwargs)
    teacher_transformed = process_renderer_sample(teacher_sample, **kwargs)
    copy_transformed = process_renderer_sample(copy_sample, **kwargs)
    action_batch = legacy.collate_single_renderer_sample(action_transformed)
    teacher_batch = legacy.collate_single_renderer_sample(teacher_transformed)
    copy_batch = legacy.collate_single_renderer_sample(copy_transformed)
    legacy.validate_collated_supervision(action_batch)
    legacy.validate_collated_supervision(teacher_batch)
    legacy.validate_collated_supervision(copy_batch)
    sigma = _sigma_for_batch(scheduler, action_batch)
    blobs = student_sample["video_vae_latents"]
    source_mode = motion.unpack_clean_mode(
        blobs[0], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    target_mode = motion.unpack_clean_mode(
        blobs[1], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    action_batch, copy_batch, auxiliary = motion.rebuild_paired_batches_from_modes(
        action_batch,
        copy_batch,
        source_mode=source_mode,
        target_mode=target_mode,
        sigma=sigma,
    )
    legacy.validate_collated_supervision(action_batch)
    legacy.validate_collated_supervision(copy_batch)
    # Copy the student's exact latent/noise/geometry state and replace only
    # the frozen teacher's text inputs.  This keeps the distillation comparison
    # causal and removes privileged target-plan text from the trainable branch.
    teacher_same_state = dict(action_batch)
    for key in ("input_ids", "attention_mask", "t5_input_lens"):
        # Bernini intentionally keeps text sequences variable-length.  Student
        # and teacher therefore need not share a token dimension.
        teacher_same_state[key] = teacher_batch[key]
    legacy.validate_collated_supervision(teacher_same_state)
    return action_batch, copy_batch, teacher_same_state, auxiliary


def _flatten_target(value: Any) -> Any:
    return motion.flatten_velocity_patches(value.unsqueeze(0))


def _losses(
    *,
    renderer: Any,
    action_batch: Mapping[str, Any],
    copy_batch: Mapping[str, Any],
    teacher_batch: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    route: motion.Route,
    args: argparse.Namespace,
    adapter_controller: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    action_prediction = motion.renderer_velocity_prediction(renderer, action_batch).float()
    action_target = _flatten_target(action_batch["target_velocity"]).float()
    source_velocity = _flatten_target(auxiliary["source_velocity"]).float()
    if tuple(action_prediction.shape) != tuple(action_target.shape):
        raise DeltaTrainingError(
            f"action prediction/target shape differs: {tuple(action_prediction.shape)} "
            f"vs {tuple(action_target.shape)}"
        )
    copy_prediction = motion.renderer_velocity_prediction(renderer, copy_batch).float()
    copy_target = _flatten_target(copy_batch["target_velocity"]).float()
    if tuple(copy_prediction.shape) != tuple(copy_target.shape):
        raise DeltaTrainingError("copy prediction/target shape differs")
    copy_loss = torch.mean((copy_prediction - copy_target) ** 2)

    # Both teachers are exact frozen Bernini priors.  The generic-prompt branch
    # protects the student's non-motion appearance, while the richer training-
    # only plan branch distils motion already expressible by the base generator.
    with torch.no_grad(), adapter_controller.disable_adapter():
        base_action = motion.renderer_velocity_prediction(renderer, action_batch).float()
        base_plan = motion.renderer_velocity_prediction(renderer, teacher_batch).float()
        base_noop = motion.renderer_velocity_prediction(renderer, copy_batch).float()
    for teacher in (base_action, base_plan, base_noop):
        if tuple(teacher.shape) != tuple(action_prediction.shape) or not bool(torch.isfinite(teacher).all().item()):
            raise DeltaTrainingError("frozen base teacher field differs or is non-finite")

    predicted_field = action_prediction - copy_prediction
    target_field = action_target - source_velocity
    projected_prediction = p3t.temporal_project(predicted_field, latent_frames=legacy.LATENT_FRAMES)
    projected_target = p3t.temporal_project(target_field, latent_frames=legacy.LATENT_FRAMES)
    projected_loss = torch.mean((projected_prediction - projected_target) ** 2)
    multilag_loss = motion.multiscale_temporal_difference_loss(
        projected_prediction, projected_target,
        latent_frames=legacy.LATENT_FRAMES, lags=tuple(args.temporal_lags),
    )
    motion_loss = args.quotient_weight * projected_loss + (1.0 - args.quotient_weight) * multilag_loss
    plan_teacher_field = base_plan - base_noop
    projected_plan_teacher = p3t.temporal_project(
        plan_teacher_field, latent_frames=legacy.LATENT_FRAMES
    )
    plan_projected_loss = torch.mean(
        (projected_prediction - projected_plan_teacher) ** 2
    )
    plan_multilag_loss = motion.multiscale_temporal_difference_loss(
        projected_prediction,
        projected_plan_teacher,
        latent_frames=legacy.LATENT_FRAMES,
        lags=tuple(args.temporal_lags),
    )
    plan_teacher_loss = (
        args.quotient_weight * plan_projected_loss
        + (1.0 - args.quotient_weight) * plan_multilag_loss
    )
    # Preserve the base model's non-motion prediction, rather than imposing an
    # arbitrary zero DC. This is the prior-preserving part of P3T.
    student_complement = p3t.temporal_complement(action_prediction, latent_frames=legacy.LATENT_FRAMES)
    teacher_complement = p3t.temporal_complement(base_action, latent_frames=legacy.LATENT_FRAMES)
    complement_loss = torch.mean((student_complement - teacher_complement) ** 2)
    replay_loss = torch.mean((copy_prediction - base_noop) ** 2)
    interval_weight = p3t.interval_weight(
        auxiliary["sigma"], steps=args.integration_steps,
        flow_shift=args.integration_flow_shift,
    ).mean()
    total = (
        args.motion_loss_weight * interval_weight * motion_loss
        + args.plan_teacher_loss_weight * interval_weight * plan_teacher_loss
        + args.complement_distill_loss_weight * complement_loss
        + args.copy_loss_weight * copy_loss
        + args.base_replay_loss_weight * replay_loss
    )
    zero = torch.zeros((), device=total.device, dtype=torch.float32)
    return total, {
        "motion": motion_loss,
        "motion_projected": projected_loss,
        "motion_multiscale": multilag_loss,
        "plan_teacher": plan_teacher_loss,
        "plan_teacher_projected": plan_projected_loss,
        "plan_teacher_multiscale": plan_multilag_loss,
        "complement_distill": complement_loss,
        "base_noop_replay": replay_loss,
        "integration_interval_weight": interval_weight,
        "full_target": zero,
        "full_target_weight": zero,
        "copy": copy_loss,
        "anchor": zero,
        "sigma": auxiliary["sigma"].float().mean(),
    }


def _optimizer_parameter_names(named: Sequence[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in named]


def _immutable_contract(
    *,
    args: argparse.Namespace,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: motion.ReviewRouter,
    eligible_routes: Sequence[tuple[int, motion.Route]],
    target_modules: Sequence[str],
    checkpoint: Path,
) -> dict[str, Any]:
    value = {
        "method": METHOD_NAME,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": args.expected_bernini_commit.lower(),
        "veomni_commit": args.expected_veomni_commit.lower(),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "checkpoint_path": str(checkpoint),
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "routing_digest": router.digest,
        "eligible_route_stream_count": len(eligible_routes),
        "eligible_route_stream_sha256": legacy.object_sha256(
            [
                {
                    "row_index": row_index,
                    "iid": route.iid,
                    "tier": route.tier,
                    "full_target_weight": route.full_target_weight,
                }
                for row_index, route in eligible_routes
            ]
        ),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "lora_rank": legacy.LORA_RANK,
        "lora_alpha": legacy.LORA_ALPHA,
        "lora_scope": args.lora_scope,
        "target_modules": list(target_modules),
        "paired_cells": ["action", "exact_copy"],
        "posterior_statistic": "mode",
        "shared_source_sigma_noise": True,
        "motion_objective": args.motion_objective,
        "motion_representation": _motion_representation_name(args),
        "temporal_lags": list(args.temporal_lags),
        "quotient_weight": float(args.quotient_weight),
        "motion_loss_weight": float(args.motion_loss_weight),
        "copy_loss_weight": float(args.copy_loss_weight),
        "anchor_loss_weight": float(args.anchor_loss_weight),
        "noop_instruction_sha256": hashlib.sha256(
            args.noop_instruction.encode("utf-8")
        ).hexdigest(),
        "preview_manifest": dict(args._p3t_manifest_receipt),
        "phase_plan_compiler": "manifest-grounded-prepare-execute-settle-21phase-v1",
        "sigma_weighting": "nearest-40step-shift5-integration-interval-width",
        "complement_teacher": "frozen-base-action-disable_adapter-no_grad",
        "complement_distill_loss_weight": float(args.complement_distill_loss_weight),
        "plan_teacher_loss_weight": float(args.plan_teacher_loss_weight),
        "base_replay_loss_weight": float(args.base_replay_loss_weight),
        "source_restoration": "tensor-helper-isolated-not-active",
        "method_status": "diagnostic_bridge_not_dense_transport",
        "full_pair_routing": "forbidden",
    }
    return {"value": value, "digest": legacy.object_sha256(value)}


def _validate_resume_receipt(
    prior: Mapping[str, Any], *, immutable: Mapping[str, Any]
) -> int:
    if prior.get("schema_version") != RECEIPT_SCHEMA:
        raise DeltaTrainingError("resume receipt schema differs")
    _validate_digest_object(prior, label="resume")
    if prior.get("immutable_contract") != immutable:
        raise DeltaTrainingError("resume immutable training contract differs")
    step = prior.get("global_step")
    if type(step) is not int or step < 0:
        raise DeltaTrainingError("resume global_step is invalid")
    return step


def _supervision_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Describe enabled losses separately from the mandatory field branches."""

    return {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "external_mask_track_pose_trajectory": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "counterfactual_noop_forward": True,
        "copy_calibration_enabled": float(args.copy_loss_weight) > 0.0,
        "copy_calibration_weight": float(args.copy_loss_weight),
        "motion_loss_enabled": float(args.motion_loss_weight) > 0.0,
        "motion_objective": args.motion_objective,
        "raw_delta_enabled": (
            float(args.motion_loss_weight) > 0.0
            and args.motion_objective == "raw_delta"
        ),
        "shared_source_posterior_mode": True,
        "shared_sigma": True,
        "shared_diffusion_noise": True,
        "unreviewed_full_target_weight": 0.0,
        "motion_representation": "generic-phase projected action/noop Wan velocity with training-only plan teacher",
        "student_action_condition": "raw edit instruction plus deterministic generic 21-phase wrapper",
        "frozen_teacher_condition": "hash-bound manifest generation_instruction plus structured target plan",
        "privileged_plan_visible_to_student": False,
        "frozen_base_teacher": True,
        "adapter_teacher_forbidden": True,
        "nonmotion_complement_distilled_from": "frozen base action prediction",
        "base_noop_replay_weight": float(args.base_replay_loss_weight),
        "complement_distill_weight": float(args.complement_distill_loss_weight),
        "plan_teacher_motion_distill_weight": float(args.plan_teacher_loss_weight),
        "sigma_weighting": "40-step shift-5 integration interval aligned",
        "source_restoration_pretext": "implemented tensor contract but isolated from renderer forward in v1",
        "temporal_quotient_enabled": (
            float(args.motion_loss_weight) > 0.0
            and args.motion_objective == "quotient_multilag"
            and float(args.quotient_weight) > 0.0
        ),
        "temporal_quotient_weight": float(args.quotient_weight),
        "multiscale_enabled": (
            float(args.motion_loss_weight) > 0.0
            and args.motion_objective == "quotient_multilag"
            and float(args.quotient_weight) < 1.0
        ),
        "temporal_lags": list(args.temporal_lags),
        "optional_first_latent_anchor_weight": float(args.anchor_loss_weight),
    }


def _build_receipt(
    *,
    args: argparse.Namespace,
    global_step: int,
    metrics: Optional[Mapping[str, float]],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: motion.ReviewRouter,
    checkpoint: Path,
    bernini_revision: str,
    veomni_revision: str,
    distributed: Any,
    backend: str,
    target_modules: Sequence[str],
    named_trainable: Sequence[tuple[str, Any]],
    initialization_digest: str,
    transformers_version: str,
    immutable: Mapping[str, Any],
    resumed_from: Optional[str],
    initialized_from: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    names = _optimizer_parameter_names(named_trainable)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "method": METHOD_NAME,
        "global_step": int(global_step),
        "max_steps": int(args.max_steps),
        "last_metrics": dict(metrics) if metrics is not None else None,
        "immutable_contract": dict(immutable),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": args.expected_checkpoint_tree_sha256,
        },
        "dataset": {
            "path": str(dataset.root),
            "rows": len(dataset),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "routing": router.receipt(),
        },
        "supervision": _supervision_receipt(args),
        "adapter": {
            "rank": legacy.LORA_RANK,
            "alpha": legacy.LORA_ALPHA,
            "scope": args.lora_scope,
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "parameter_names_sha256": legacy.object_sha256(names),
            "initialization_digest": initialization_digest,
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
        },
        "distributed": {
            "world_size": distributed.world_size,
            "ulysses_size": distributed.ulysses_size,
            "backend": backend,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": distributed.world_size > 1,
        },
        "transformers_version": transformers_version,
        "resumed_from": resumed_from,
        "initialized_from": dict(initialized_from) if initialized_from else None,
        "experimental_training": True,
        "dataset_post_video_acceptance": "pending",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = legacy.object_sha256(receipt)
    return receipt


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
    rank: int,
) -> Path:
    import torch
    import torch.distributed as dist

    final = output / f"checkpoint-{global_step:08d}"
    if rank == 0:
        if final.exists():
            raise DeltaTrainingError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        if temporary.exists():
            raise DeltaTrainingError(f"stale temporary checkpoint exists: {temporary}")
        temporary.mkdir()
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(
            {
                "schema_version": OPTIMIZER_SCHEMA,
                "global_step": global_step,
                "optimizer": optimizer.state_dict(),
                "immutable_contract": dict(immutable),
                "parameter_names": list(parameter_names),
            },
            temporary / "optimizer.pt",
        )
        _atomic_write_json(temporary / "receipt.json", receipt)
        os.replace(temporary, final)
        _atomic_write_json(
            output / "latest.json",
            {
                "checkpoint": str(final),
                "global_step": global_step,
                "receipt_digest": receipt["receipt_digest"],
            },
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return final


def _load_peft_adapter(
    *,
    base_model: Any,
    adapter: Path,
    target_modules: Sequence[str],
    trainable: bool,
) -> Any:
    import torch
    from peft import LoraConfig, PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors

    config = LoraConfig.from_pretrained(str(adapter), local_files_only=True)
    if int(config.r) != legacy.LORA_RANK or int(config.lora_alpha) != legacy.LORA_ALPHA:
        raise DeltaTrainingError("adapter rank/alpha differs")
    # PEFT serializes exact-name sets compactly.  Rebind the reviewed runtime
    # set so attn1/attn2 scope cannot expand accidentally on reload.
    config.target_modules = set(target_modules)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter),
        is_trainable=trainable,
        config=config,
        local_files_only=True,
    )
    expected = {
        f"base_model.model.{target}.lora_{factor}.weight"
        for target in target_modules
        for factor in ("A", "B")
    }
    saved = load_safetensors(str(adapter / "adapter_model.safetensors"), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    if set(saved) != expected or set(loaded) != expected:
        raise DeltaTrainingError(
            "strict adapter reload scope differs: "
            f"saved_delta={len(set(saved) ^ expected)} "
            f"loaded_delta={len(set(loaded) ^ expected)}"
        )
    unequal = [
        key
        for key in sorted(expected)
        if not bool(torch.equal(saved[key].cpu(), loaded[key].cpu()))
    ]
    if unequal:
        raise DeltaTrainingError(f"strict adapter reload tensor differs: {unequal[:4]}")
    return model


def _initialization_target_modules(
    receipt_path: Optional[Path], *, available_modules: Sequence[str]
) -> list[str]:
    """Recover the exact saved scope; never silently project a larger adapter."""

    if receipt_path is None:
        raise DeltaTrainingError(
            "initialization adapter requires its hash-bound training receipt"
        )
    receipt = _read_json(receipt_path, label="initialization receipt")
    candidate = dict(receipt)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or legacy.object_sha256(candidate) != declared:
        raise DeltaTrainingError("initialization receipt digest differs")
    schema = receipt.get("schema_version")
    if schema == RECEIPT_SCHEMA:
        adapter = receipt.get("adapter")
        targets = adapter.get("target_modules") if isinstance(adapter, dict) else None
        if not isinstance(targets, list) or not all(isinstance(name, str) for name in targets):
            raise DeltaTrainingError("CDF initialization receipt lacks exact targets")
        return list(targets)
    if schema == legacy.RECEIPT_SCHEMA:
        if (
            receipt.get("target_module_count") != legacy.EXPECTED_LORA_TARGET_MODULES
            or receipt.get("target_modules_sha256")
            != legacy.object_sha256(list(available_modules))
        ):
            raise DeltaTrainingError("legacy initialization target scope differs")
        return list(available_modules)
    raise DeltaTrainingError("unsupported initialization receipt schema")


def _optimizer_to(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=args.expected_bernini_commit,
            expected_veomni_commit=args.expected_veomni_commit,
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    if transformer_config["num_attention_heads"] % 4:
        raise DeltaTrainingError("1.3B attention heads must be divisible by Ulysses=4")
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, process_renderer_sample

    distributed = legacy.distributed_contract()
    device, backend = legacy.initialise_distributed(distributed)
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=distributed.ulysses_size)
    legacy.seed_same_sample(args.seed)
    output = Path(args.output).expanduser().resolve()
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
    )
    try:
        preview_manifest = p3t.PreviewManifest(
            args.preview_manifest,
            expected_sha256=args.expected_preview_manifest_sha256,
        )
        # Bind the entire dataset before model construction. A partial IID join
        # must never turn into a silently different training stream.
        dataset_iids = [_iid(dataset[index]) for index in range(len(dataset))]
        if len(set(dataset_iids)) != len(dataset_iids):
            raise DeltaTrainingError("dataset contains duplicate IIDs")
        for iid in dataset_iids:
            preview_manifest.require(iid)
    except p3t.P3TContractError as error:
        raise _translate(error) from error
    args._p3t_manifest_receipt = preview_manifest.receipt()
    try:
        router = motion.ReviewRouter.load(
            args.routing_jsonl, default_tier=args.unreviewed_tier
        )
    except motion.MotionContractError as error:
        raise _translate(error) from error
    eligible_routes = _build_eligible_routes(dataset, router)
    _validate_active_supervision(args, eligible_routes)

    config_dir = bernini_root / "configs/bernini_renderer_wan21_1p3b"
    config = BerniniRendererConfig.from_pretrained(
        str(config_dir),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    base_model = BerniniRendererModel(config)
    base_model.requires_grad_(False)
    base_model.t5_text_encoder.eval()
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    available_modules = legacy.select_attention_projection_names(base_model)
    try:
        target_modules = motion.select_lora_scope(
            available_modules, args.lora_scope
        )
    except motion.MotionContractError as error:
        raise _translate(error) from error
    immutable = _immutable_contract(
        args=args,
        dataset=dataset,
        dataset_summary=dataset_summary,
        router=router,
        eligible_routes=eligible_routes,
        target_modules=target_modules,
        checkpoint=checkpoint,
    )

    resume_paths: Optional[AdapterPaths] = None
    prior_receipt: Optional[dict[str, Any]] = None
    initialized_from: Optional[dict[str, Any]] = None
    if args.resume:
        resume_paths = _resolve_adapter(args.resume, require_training_state=True)
        assert resume_paths.receipt is not None
        prior_receipt = _read_json(resume_paths.receipt, label="resume receipt")
        prior_step = _validate_resume_receipt(prior_receipt, immutable=immutable)
        model = _load_peft_adapter(
            base_model=base_model,
            adapter=resume_paths.adapter,
            target_modules=target_modules,
            trainable=True,
        )
    elif args.init_adapter_checkpoint:
        init_paths = _resolve_adapter(
            args.init_adapter_checkpoint, require_training_state=False
        )
        initialization_targets = _initialization_target_modules(
            init_paths.receipt, available_modules=available_modules
        )
        if initialization_targets != target_modules:
            raise DeltaTrainingError(
                "initialization adapter scope differs from requested LoRA scope; "
                "explicit adapter projection is not supported"
            )
        model = _load_peft_adapter(
            base_model=base_model,
            adapter=init_paths.adapter,
            target_modules=target_modules,
            trainable=True,
        )
        initialized_from = {
            "path": str(init_paths.root),
            "adapter_model_sha256": legacy.file_sha256(
                init_paths.adapter / "adapter_model.safetensors"
            ),
            "receipt_sha256": legacy.file_sha256(init_paths.receipt)
            if init_paths.receipt is not None
            else None,
        }
        prior_step = 0
    else:
        lora_config = LoraConfig(
            r=legacy.LORA_RANK,
            lora_alpha=legacy.LORA_ALPHA,
            lora_dropout=0.0,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_config)
        prior_step = 0

    model.to(device)
    model.train()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    named_trainable = legacy.trainable_lora_parameters(model)
    initialization_digest = legacy.synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    parameter_names = _optimizer_parameter_names(named_trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    global_step = prior_step
    if resume_paths is not None:
        assert resume_paths.optimizer is not None
        try:
            state = torch.load(
                resume_paths.optimizer, map_location="cpu", weights_only=False
            )
        except TypeError:
            state = torch.load(resume_paths.optimizer, map_location="cpu")
        if (
            state.get("schema_version") != OPTIMIZER_SCHEMA
            or state.get("immutable_contract") != immutable
            or state.get("parameter_names") != parameter_names
            or int(state.get("global_step", -1)) != global_step
        ):
            raise DeltaTrainingError("resume optimizer contract differs")
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to(optimizer, device)
        # Optimizer state owns the actual LR.  Exact immutable matching above
        # guarantees it equals the CLI/receipt value; assert it again here.
        if any(
            float(group["lr"]) != float(args.learning_rate)
            or float(group["weight_decay"]) != float(args.weight_decay)
            for group in optimizer.param_groups
        ):
            raise DeltaTrainingError("restored optimizer hyperparameters differ")
    if global_step > args.max_steps:
        raise DeltaTrainingError("resume step exceeds requested max_steps")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())

    last_metrics: Optional[dict[str, float]] = None
    last_saved = global_step if resume_paths is not None else -1
    while global_step < args.max_steps:
        row_index, raw_row, route = _next_routed_row(
            dataset, eligible_routes, ordinal=global_step
        )
        identity = legacy.dataset_identity(raw_row, row_index)
        legacy.assert_identical_row(identity)
        current_seed = legacy.step_seed(args.seed, global_step, row_index)
        legacy.seed_same_sample(current_seed)
        try:
            manifest_plan = preview_manifest.require(route.iid)
            action_batch, copy_batch, teacher_batch, auxiliary = _prepare_paired_batches(
                raw_row=raw_row,
                tokenizer=tokenizer,
                rope=rope,
                vae_mean=vae_mean,
                vae_std=vae_std,
                z_dim=z_dim,
                scheduler=scheduler,
                noop_instruction=args.noop_instruction,
                student_instruction=manifest_plan.edit_instruction,
                compiled_plan=manifest_plan.compiled_plan,
                process_renderer_sample=process_renderer_sample,
            )
        except (
            legacy.TrainingContractError,
            motion.MotionContractError,
            p3t.P3TContractError,
        ) as error:
            raise _translate(error) from error
        action_batch = legacy._move_batch(action_batch, device)
        copy_batch = legacy._move_batch(copy_batch, device)
        teacher_batch = legacy._move_batch(teacher_batch, device)
        auxiliary = legacy._move_batch(auxiliary, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, components = _losses(
                renderer=renderer,
                action_batch=action_batch,
                copy_batch=copy_batch,
                teacher_batch=teacher_batch,
                auxiliary=auxiliary,
                route=route,
                args=args,
                adapter_controller=model,
            )
        finite = bool(torch.isfinite(loss.detach()).item()) and all(
            bool(torch.isfinite(value.detach()).item()) for value in components.values()
        )
        if not legacy._distributed_boolean(finite, op="all"):
            raise DeltaTrainingError(
                f"non-finite CDF loss at optimizer step {global_step + 1}"
            )
        loss.backward()
        gradient_norm = legacy.all_reduce_lora_gradients(named_trainable)
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named_trainable], args.max_grad_norm
        )
        optimizer.step()
        global_step += 1
        last_metrics = {
            "total": float(loss.detach().item()),
            **{
                name: float(value.detach().item())
                for name, value in components.items()
            },
            "preclip_gradient_norm": float(gradient_norm),
            "route_full_target_weight": float(route.full_target_weight),
        }
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "row": row_index,
                        "iid": route.iid,
                        "tier": route.tier,
                        "phase_plan_sha256": hashlib.sha256(manifest_plan.compiled_plan.encode("utf-8")).hexdigest(),
                        "seed": current_seed,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if args.save_every > 0 and global_step % args.save_every == 0:
            receipt = _build_receipt(
                args=args,
                global_step=global_step,
                metrics=last_metrics,
                dataset=dataset,
                dataset_summary=dataset_summary,
                router=router,
                checkpoint=checkpoint,
                bernini_revision=bernini_revision,
                veomni_revision=veomni_revision,
                distributed=distributed,
                backend=backend,
                target_modules=target_modules,
                named_trainable=named_trainable,
                initialization_digest=initialization_digest,
                transformers_version=transformers_version,
                immutable=immutable,
                resumed_from=str(resume_paths.root) if resume_paths else None,
                initialized_from=initialized_from,
            )
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                output=output,
                global_step=global_step,
                receipt=receipt,
                immutable=immutable,
                parameter_names=parameter_names,
                rank=distributed.rank,
            )
            last_saved = global_step

    if last_saved != global_step:
        receipt = _build_receipt(
            args=args,
            global_step=global_step,
            metrics=last_metrics,
            dataset=dataset,
            dataset_summary=dataset_summary,
            router=router,
            checkpoint=checkpoint,
            bernini_revision=bernini_revision,
            veomni_revision=veomni_revision,
            distributed=distributed,
            backend=backend,
            target_modules=target_modules,
            named_trainable=named_trainable,
            initialization_digest=initialization_digest,
            transformers_version=transformers_version,
            immutable=immutable,
            resumed_from=str(resume_paths.root) if resume_paths else None,
            initialized_from=initialized_from,
        )
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            output=output,
            global_step=global_step,
            receipt=receipt,
            immutable=immutable,
            parameter_names=parameter_names,
            rank=distributed.rank,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train Bernini Counterfactual Delta-Field LoRA on exact 81-frame pairs.

This is a falsifiable motion/identity prototype, not a claim that the current
synthetic targets are accepted ground truth.  In the default same-state mode,
every optimizer step has two text branches evaluated on the *identical* source,
noisy query, sigma, timestep, and rotary geometry:

* action: source + edit instruction -> action clean field;
* no-op:  same source + fixed no-op instruction -> source clean field.

Their supervised quantity is the clean counterfactual difference recovered by
``x=y-sigma*v``.  This is the same comparison made by C2FR inference and fixes
the older two-clean-path train/test mismatch.  The legacy construction remains
available only as an explicit ablation.

For unreviewed data the action cell is supervised only through a source-
relative causal-boundary quotient and multi-lag differences.  The field is
measured relative to its first latent phase, so a constant appearance change is
removed while a new action that begins and remains is not weakened or leaked
backwards in time.  A boundary penalty fixes the remaining raw-field gauge.
Full flow matching is enabled only by an explicit per-IID ``full_pair`` review.

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
import struct
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import motion_residual as motion  # noqa: E402
import inference_sigma_strata as sigma_strata  # noqa: E402
import train_lora as legacy  # noqa: E402


RECEIPT_SCHEMA = "bernini-r-1p3b-c2fr-lora-receipt-v4"
OPTIMIZER_SCHEMA = "bernini-r-1p3b-c2fr-lora-optimizer-v4"
METHOD_NAME = (
    "projected-bridge-consistent-robust-counterfactual-clean-field-lora-v4"
)
DEFAULT_TEMPORAL_LAGS = (1, 2, 4)
BRIDGE_FRACTIONS = (0.0, 1.0)


class DeltaTrainingError(RuntimeError):
    """Raised when a CDF-LoRA training invariant is violated."""


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
        description="Train 81f Bernini Counterfactual Delta-Field LoRA"
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--routing-jsonl", default=None)
    parser.add_argument("--expected-routing-jsonl-sha256", default=None)
    parser.add_argument(
        "--unreviewed-tier", choices=("motion_only", "reject"), default="reject"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(81,), default=81)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-adapter-checkpoint", default=None)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--lora-scope",
        choices=tuple(sorted(motion.MODULE_SCOPES)),
        default="cross_q",
    )
    parser.add_argument(
        "--branch-state-mode",
        choices=tuple(sorted(motion.BRANCH_STATE_MODES)),
        default="source_target_bridge_clean_field",
        help=(
            "source_target_bridge_clean_field is the v4 C2FR method; "
            "shared_noisy_clean_field is the target-path v3 ablation; "
            "separate_clean_paths is retained only as a legacy ablation"
        ),
    )
    parser.add_argument(
        "--minimum-training-sigma",
        type=float,
        default=0.1,
        help=(
            "lower flow-sigma bound; 0.1 covers the final positive sigma of "
            "the 40-step inference schedule"
        ),
    )
    parser.add_argument(
        "--inverse-sigma-weight-floor",
        type=float,
        default=sigma_strata.PINNED_POSITIVE_SIGMAS[-1],
        help=(
            "clamp for inverse-sigma clean-field weighting; formal v4 uses "
            "the exact final positive 40-step UniPC sigma"
        ),
    )
    parser.add_argument("--motion-loss-weight", type=float, default=1.0)
    parser.add_argument("--copy-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--boundary-gauge-loss-weight",
        type=float,
        default=0.0,
        help=(
            "force the raw action-minus-noop clean field to zero at the first "
            "latent phase, fixing the causal quotient gauge"
        ),
    )
    parser.add_argument(
        "--anchor-loss-weight",
        type=float,
        default=0.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--motion-objective",
        choices=tuple(sorted(motion.MOTION_OBJECTIVES)),
        default="causal_boundary_charbonnier",
    )
    parser.add_argument("--bridge-consistency-weight", type=float, default=0.1)
    parser.add_argument("--causal-ema-decay", type=float, default=0.5)
    parser.add_argument("--charbonnier-scale", type=float, default=0.1)
    parser.add_argument("--quotient-weight", type=float, default=0.5)
    parser.add_argument("--high-noise-floor", type=float, default=1.0)
    parser.add_argument("--high-noise-power", type=float, default=2.0)
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
    if args.branch_state_mode not in motion.BRANCH_STATE_MODES:
        raise DeltaTrainingError(
            f"unknown branch state mode: {args.branch_state_mode!r}"
        )
    for name in (
        "learning_rate",
        "max_grad_norm",
        "high_noise_power",
        "charbonnier_scale",
    ):
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
        "boundary_gauge_loss_weight",
        "bridge_consistency_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise DeltaTrainingError(f"{name} must be finite and non-negative")
    if not 0.0 <= args.quotient_weight <= 1.0:
        raise DeltaTrainingError("quotient_weight must lie in [0, 1]")
    if not 0.0 <= args.high_noise_floor <= 1.0:
        raise DeltaTrainingError("high_noise_floor must lie in [0, 1]")
    if (
        not math.isfinite(float(args.causal_ema_decay))
        or not 0.0 <= float(args.causal_ema_decay) < 1.0
    ):
        raise DeltaTrainingError("causal_ema_decay must lie in [0, 1)")
    if (
        not math.isfinite(float(args.minimum_training_sigma))
        or not 0.0 < float(args.minimum_training_sigma) < 1.0
    ):
        raise DeltaTrainingError("minimum_training_sigma must lie in (0, 1)")
    if (
        not math.isfinite(float(args.inverse_sigma_weight_floor))
        or not 0.0 < float(args.inverse_sigma_weight_floor) <= 1.0
    ):
        raise DeltaTrainingError("inverse_sigma_weight_floor must lie in (0, 1]")
    if args.inverse_sigma_weight_floor < args.minimum_training_sigma:
        raise DeltaTrainingError(
            "inverse_sigma_weight_floor must be at least minimum_training_sigma"
        )
    clean_field_modes = {
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    }
    if float(args.boundary_gauge_loss_weight) > 0.0 and (
        args.branch_state_mode not in clean_field_modes
        or args.motion_objective
        not in (
            "causal_boundary_multilag",
            "causal_boundary_charbonnier",
            "causal_ema_charbonnier",
        )
    ):
        raise DeltaTrainingError(
            "boundary gauge is supported only by same-state causal-boundary training"
        )
    legacy_anchor_weight = float(getattr(args, "anchor_loss_weight", 0.0))
    if not math.isfinite(legacy_anchor_weight) or legacy_anchor_weight != 0.0:
        raise DeltaTrainingError(
            "legacy first-frame anchor loss is unsupported; use the causal boundary gauge"
        )
    lags = tuple(args.temporal_lags)
    if (
        not lags
        or len(set(lags)) != len(lags)
        or any(type(lag) is not int or lag <= 0 or lag >= legacy.LATENT_FRAMES for lag in lags)
    ):
        raise DeltaTrainingError("temporal lags must be unique integers in [1, 20]")
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip() or "\x00" in args.noop_instruction:
        raise DeltaTrainingError("noop instruction must be non-empty text")
    if args.branch_state_mode in clean_field_modes and (
        args.noop_instruction != motion.DEFAULT_NOOP_INSTRUCTION
    ):
        raise DeltaTrainingError(
            "same-state C2FR training requires the fixed inference no-op instruction"
        )
    if args.branch_state_mode == "source_target_bridge_clean_field":
        if args.motion_objective != "causal_boundary_charbonnier":
            raise DeltaTrainingError(
                "v4 bridge training requires causal_boundary_charbonnier supervision"
            )
        if args.lora_scope != "cross_q":
            raise DeltaTrainingError(
                "formal v4 bridge training requires the 30-module cross_q scope"
            )
        if args.unreviewed_tier != "reject":
            raise DeltaTrainingError(
                "formal v4 bridge training requires reject as the default route"
            )
        if float(args.copy_loss_weight) != 0.0:
            raise DeltaTrainingError(
                "v4 bridge training forbids the target-query source-copy loss"
            )
        if float(args.bridge_consistency_weight) <= 0.0:
            raise DeltaTrainingError(
                "v4 bridge training requires positive endpoint consistency"
            )
        if float(args.bridge_consistency_weight) != 0.1:
            raise DeltaTrainingError(
                "formal v4 bridge training fixes endpoint consistency to 0.1"
            )
        if float(args.inverse_sigma_weight_floor) != float(
            sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
        ):
            raise DeltaTrainingError(
                "formal v4 inverse-sigma floor must equal the final UniPC sigma"
            )
        if float(args.high_noise_floor) != 1.0:
            raise DeltaTrainingError(
                "formal v4 forbids attenuating low-sigma Q0 supervision"
            )
        if (
            not isinstance(args.expected_routing_jsonl_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", args.expected_routing_jsonl_sha256)
            is None
        ):
            raise DeltaTrainingError(
                "v4 bridge training requires a pinned strict routing SHA-256"
            )
    elif float(args.bridge_consistency_weight) != 0.0:
        raise DeltaTrainingError(
            "bridge_consistency_weight is supported only by v4 bridge training"
        )
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", getattr(args, name)) is None:
            raise DeltaTrainingError(f"{name} must be a full SHA-1")
    for name in ("expected_checkpoint_tree_sha256", "method_source_archive_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)) is None:
            raise DeltaTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise DeltaTrainingError("checkpoint identity differs from the audited 1.3B tree")


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


def _validate_v4_strict_router(
    args: argparse.Namespace,
    router: motion.ReviewRouter,
    eligible_routes: Sequence[tuple[int, motion.Route]],
) -> None:
    """Bind v4 to the audited strict-359 / reject-285 cohort."""

    if args.branch_state_mode != "source_target_bridge_clean_field":
        return
    receipt = router.receipt()
    if (
        receipt.get("path") is None
        or receipt.get("default_tier") != "reject"
        or receipt.get("file_sha256") != args.expected_routing_jsonl_sha256
        or receipt.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or len(eligible_routes) != 359
        or any(
            route.tier != "motion_only" or route.full_target_weight != 0.0
            for _, route in eligible_routes
        )
    ):
        raise DeltaTrainingError(
            "v4 requires the hash-bound strict-359 motion-only / reject-285 route"
        )


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
            args.boundary_gauge_loss_weight,
            args.bridge_consistency_weight,
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
            "training recipe has rows with no active loss; enable motion/copy/boundary "
            "supervision or route every eligible row to positive full-target weight"
        )


def _motion_representation_name(args: argparse.Namespace) -> str:
    if args.motion_objective == "raw_delta":
        return "source-relative-raw-delta-v1"
    if args.motion_objective == "causal_boundary_charbonnier":
        return "source-relative-causal-boundary-charbonnier-v1"
    if args.motion_objective == "causal_ema_charbonnier":
        return "source-relative-causal-ema-boundary-charbonnier-v1"
    prefix = (
        "source-relative-causal-boundary"
        if args.motion_objective == "causal_boundary_multilag"
        else "source-relative-temporal-mean-quotient"
    )
    if float(args.quotient_weight) == 0.0:
        return "source-relative-multilag-v1"
    if float(args.quotient_weight) == 1.0:
        return f"{prefix}-v1"
    return f"{prefix}-multilag-v1"


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
    branch_state_mode: str,
    minimum_training_sigma: float,
    process_renderer_sample: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample = legacy.sanitize_preprocessed_row(raw_row)
    legacy.validate_81_frame_latents(sample, expected_parameter_channels=2 * z_dim)
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
    action_transformed = process_renderer_sample(sample, **kwargs)
    copy_transformed = process_renderer_sample(copy_sample, **kwargs)
    action_batch = legacy.collate_single_renderer_sample(action_transformed)
    copy_batch = legacy.collate_single_renderer_sample(copy_transformed)
    legacy.validate_collated_supervision(action_batch)
    legacy.validate_collated_supervision(copy_batch)
    sigma = _sigma_for_batch(scheduler, action_batch)
    blobs = sample["video_vae_latents"]
    source_mode = motion.unpack_clean_mode(
        blobs[0], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    target_mode = motion.unpack_clean_mode(
        blobs[1], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    if branch_state_mode == "shared_noisy_clean_field":
        action_batch, copy_batch, auxiliary = (
            motion.rebuild_same_state_batches_from_modes(
                action_batch,
                copy_batch,
                source_mode=source_mode,
                target_mode=target_mode,
                sigma=sigma,
                minimum_sigma=minimum_training_sigma,
            )
        )
    elif branch_state_mode == "separate_clean_paths":
        action_batch, copy_batch, auxiliary = motion.rebuild_paired_batches_from_modes(
            action_batch,
            copy_batch,
            source_mode=source_mode,
            target_mode=target_mode,
            sigma=sigma,
        )
        auxiliary["branch_state_mode"] = "separate_clean_paths"
    else:
        raise DeltaTrainingError(
            f"unknown branch state mode: {branch_state_mode!r}"
        )
    legacy.validate_collated_supervision(action_batch)
    legacy.validate_collated_supervision(copy_batch)
    return action_batch, copy_batch, auxiliary


def _prepare_bridge_batches(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    minimum_training_sigma: float,
    process_renderer_sample: Any,
    selected_stratum: sigma_strata.SigmaStratum,
) -> dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Construct source/target bridge endpoints with one epsilon and sigma."""

    import torch

    sample = legacy.sanitize_preprocessed_row(raw_row)
    legacy.validate_81_frame_latents(sample, expected_parameter_channels=2 * z_dim)
    noop_sample = motion.replace_edit_instruction(sample, noop_instruction)
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
    action_batch = legacy.collate_single_renderer_sample(
        process_renderer_sample(sample, **kwargs)
    )
    noop_batch = legacy.collate_single_renderer_sample(
        process_renderer_sample(noop_sample, **kwargs)
    )
    legacy.validate_collated_supervision(action_batch)
    legacy.validate_collated_supervision(noop_batch)

    old_sigma = _sigma_for_batch(scheduler, action_batch)
    selector = action_batch["vae_latents_mask"].squeeze(0).bool()
    target_noisy_old = action_batch["input_vae_latents"][selector]
    old_velocity = action_batch["target_velocity"]
    old_sigma_shape = [1] * target_noisy_old.ndim
    old_sigma_shape[0] = 1
    epsilon = target_noisy_old.float() + (
        1.0 - old_sigma.float().reshape(old_sigma_shape)
    ) * old_velocity.float()
    blobs = sample["video_vae_latents"]
    source_mode = motion.unpack_clean_mode(
        blobs[0], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    target_mode = motion.unpack_clean_mode(
        blobs[1], vae_mean, vae_std, max_frames=legacy.LATENT_FRAMES
    )
    executable_target_mode = motion.project_executable_target_mode(
        source_mode,
        target_mode,
        latent_frames=legacy.LATENT_FRAMES,
    )
    raw_target_clean = motion.flatten_velocity_patches(
        target_mode.unsqueeze(0)
    ).float()
    forced_sigma = torch.tensor(
        selected_stratum.sigma, dtype=torch.float32, device="cpu"
    ).reshape(())
    forced_timestep = torch.tensor(
        selected_stratum.timestep, dtype=torch.int64, device="cpu"
    ).reshape(())
    sigma_strata.assert_selected_timestep_sigma(
        timestep=forced_timestep,
        sigma=forced_sigma,
        selected=selected_stratum,
    )
    endpoints: dict[
        str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = {}
    for name, beta in zip(("source", "target"), BRIDGE_FRACTIONS):
        try:
            action, noop, auxiliary = (
                motion.rebuild_bridge_state_batches_from_modes(
                    action_batch,
                    noop_batch,
                    source_mode=source_mode,
                    target_mode=executable_target_mode,
                    epsilon=epsilon,
                    sigma=forced_sigma,
                    timestep=forced_timestep,
                    bridge_fraction=beta,
                    minimum_sigma=minimum_training_sigma,
                )
            )
        except motion.MotionContractError as error:
            raise _translate(error) from error
        auxiliary["raw_target_clean"] = raw_target_clean
        auxiliary["target_projection"] = (
            "executable_target=source+Q0(raw_target-source)"
        )
        sigma_strata.assert_selected_timestep_sigma(
            timestep=action["timesteps"],
            sigma=auxiliary["sigma"],
            selected=selected_stratum,
        )
        legacy.validate_collated_supervision(action)
        legacy.validate_collated_supervision(noop)
        endpoints[name] = (action, noop, auxiliary)
    source_auxiliary = endpoints["source"][2]
    target_auxiliary = endpoints["target"][2]
    for field in (
        "source_clean",
        "target_clean",
        "raw_target_clean",
        "epsilon",
        "sigma",
        "timestep",
    ):
        if not torch.equal(source_auxiliary[field], target_auxiliary[field]):
            raise DeltaTrainingError(
                f"bridge endpoints do not share exact {field}"
            )
    return endpoints


def _flatten_target(value: Any) -> Any:
    return motion.flatten_velocity_patches(value.unsqueeze(0))


def _move_auxiliary_to_device(
    auxiliary: Mapping[str, Any],
    *,
    device: Any,
    branch_state_mode: str,
) -> dict[str, Any]:
    """Move training targets while preserving pinned UniPC scalar semantics."""

    if branch_state_mode not in (
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    ):
        return legacy._move_batch(auxiliary, device)
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise DeltaTrainingError(
            "PyTorch is required for same-state training"
        ) from error
    sigma = auxiliary.get("sigma")
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.numel() != 1
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma).all())
        or not bool((sigma > 0).all())
    ):
        raise DeltaTrainingError(
            "same-state training sigma must enter device transfer as CPU fp32"
        )
    movable = dict(auxiliary)
    movable.pop("sigma")
    moved = legacy._move_batch(movable, device)
    moved["sigma"] = sigma.detach().reshape(())
    return moved


def _bridge_losses(
    *,
    renderer: Any,
    endpoints: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ],
    route: motion.Route,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    """Train one field on source and target query endpoints and align them."""

    import torch

    if route.full_target_weight != 0.0:
        raise DeltaTrainingError(
            "v4 bridge training forbids framewise full-target supervision"
        )
    if set(endpoints) != {"source", "target"}:
        raise DeltaTrainingError("v4 bridge training requires both query endpoints")
    endpoint_parts: dict[str, dict[str, Any]] = {}
    for name in ("source", "target"):
        action_batch, noop_batch, auxiliary = endpoints[name]
        shared_fields = (
            "input_vae_latents",
            "input_vae_rope",
            "vae_latents_mask",
            "timesteps",
            "vae_seqlen",
            "target_lens",
        )
        unequal = [
            field
            for field in shared_fields
            if not torch.equal(action_batch[field], noop_batch[field])
        ]
        if unequal:
            raise DeltaTrainingError(
                f"{name} bridge action/no-op renderer states differ: {unequal}"
            )
        if torch.equal(action_batch["input_ids"], noop_batch["input_ids"]):
            raise DeltaTrainingError(
                f"{name} bridge action/no-op text branches are identical"
            )
        action_prediction = motion.renderer_velocity_prediction(
            renderer, action_batch
        )
        noop_prediction = motion.renderer_velocity_prediction(renderer, noop_batch)
        if (
            action_prediction.dtype != torch.bfloat16
            or noop_prediction.dtype != torch.bfloat16
            or auxiliary["shared_noisy"].dtype != torch.float32
            or auxiliary.get("branch_state_mode")
            != "source_target_bridge_clean_field"
            or auxiliary.get("target_projection")
            != "executable_target=source+Q0(raw_target-source)"
        ):
            raise DeltaTrainingError(
                "bridge clean reconstruction requires fp32 noisy, native bf16 "
                "branches, and the v4 endpoint contract"
            )
        endpoint_loss, parts = motion.differential_clean_motion_loss(
            action_prediction,
            noop_prediction,
            auxiliary["shared_noisy"],
            auxiliary["sigma"],
            auxiliary["target_clean"],
            auxiliary["source_clean"],
            latent_frames=legacy.LATENT_FRAMES,
            lags=tuple(args.temporal_lags),
            quotient_weight=args.quotient_weight,
            objective=args.motion_objective,
            causal_ema_decay=args.causal_ema_decay,
            charbonnier_scale=args.charbonnier_scale,
        )
        action_field = parts["predicted_clean_delta"]
        action_grid = action_field.reshape(
            int(action_field.shape[0]),
            legacy.LATENT_FRAMES,
            int(action_field.shape[1]) // legacy.LATENT_FRAMES,
            int(action_field.shape[2]),
        )
        endpoint_parts[name] = {
            "loss": endpoint_loss,
            "raw_delta": parts["raw_delta"],
            "causal_boundary": parts["causal_boundary"],
            "causal_boundary_robust": parts[
                "causal_boundary_charbonnier"
            ],
            "multiscale": parts["multiscale_difference"],
            "predicted_causal_boundary": parts[
                "predicted_causal_boundary"
            ],
            "target_causal_boundary": parts["target_causal_boundary"],
            "field": action_field,
            "boundary": torch.mean(action_grid[:, 0].square()),
            "noop_source_diagnostic": torch.mean(
                (parts["noop_clean"] - auxiliary["source_clean"].float()).square()
            ),
            "sigma": auxiliary["sigma"],
        }

    source_parts = endpoint_parts["source"]
    target_parts = endpoint_parts["target"]
    if not torch.equal(source_parts["sigma"], target_parts["sigma"]):
        raise DeltaTrainingError("bridge endpoint sigmas differ")
    if not torch.equal(
        source_parts["target_causal_boundary"],
        target_parts["target_causal_boundary"],
    ):
        raise DeltaTrainingError("bridge endpoints have different target representation")
    bridge_consistency = motion.charbonnier_distance(
        source_parts["predicted_causal_boundary"],
        target_parts["predicted_causal_boundary"],
        scale=args.charbonnier_scale,
    )
    motion_loss = 0.5 * (source_parts["loss"] + target_parts["loss"])
    boundary_gauge_loss = 0.5 * (
        source_parts["boundary"] + target_parts["boundary"]
    )
    sigma = source_parts["sigma"]
    noise_weight = motion.high_noise_weight(
        sigma,
        floor=args.high_noise_floor,
        power=args.high_noise_power,
    ).mean()
    clean_field_weight = motion.clean_field_inverse_sigma_weight(
        sigma, weight_floor=args.inverse_sigma_weight_floor
    ).mean()
    total = clean_field_weight * (
        args.motion_loss_weight * noise_weight * motion_loss
        + args.bridge_consistency_weight * bridge_consistency
        + args.boundary_gauge_loss_weight * boundary_gauge_loss
    )

    def average(key: str) -> Any:
        return 0.5 * (source_parts[key] + target_parts[key])

    def rms(value: Any) -> Any:
        return value.float().square().mean().sqrt()

    return total, {
        "motion": motion_loss,
        "motion_raw_delta": average("raw_delta"),
        "motion_causal_boundary": average("causal_boundary"),
        "motion_causal_boundary_robust": average(
            "causal_boundary_robust"
        ),
        "motion_multiscale": average("multiscale"),
        "bridge_consistency": bridge_consistency,
        "source_query_executed_field_rms": rms(
            source_parts["predicted_causal_boundary"]
        ),
        "target_query_executed_field_rms": rms(
            target_parts["predicted_causal_boundary"]
        ),
        "target_supervision_field_rms": rms(
            source_parts["target_causal_boundary"]
        ),
        "noop_source_diagnostic": average("noop_source_diagnostic"),
        "high_noise_weight": noise_weight,
        "clean_field_weight": clean_field_weight,
        "full_target": torch.zeros((), device=motion_loss.device),
        "full_target_weight": torch.zeros((), device=motion_loss.device),
        "copy": torch.zeros((), device=motion_loss.device),
        "boundary_gauge": boundary_gauge_loss,
        "sigma": sigma.float().mean(),
        "same_state_exact": torch.ones((), device=motion_loss.device),
    }


def _losses(
    *,
    renderer: Any,
    action_batch: Mapping[str, Any],
    copy_batch: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    route: motion.Route,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    import torch

    action_prediction = motion.renderer_velocity_prediction(renderer, action_batch)
    action_target = _flatten_target(action_batch["target_velocity"]).float()
    if tuple(action_prediction.shape) != tuple(action_target.shape):
        raise DeltaTrainingError(
            f"action prediction/target shape differs: {tuple(action_prediction.shape)} "
            f"vs {tuple(action_target.shape)}"
        )
    noise_weight = motion.high_noise_weight(
        auxiliary["sigma"],
        floor=args.high_noise_floor,
        power=args.high_noise_power,
    ).mean()
    copy_prediction = motion.renderer_velocity_prediction(renderer, copy_batch)
    copy_target = _flatten_target(copy_batch["target_velocity"]).float()
    if tuple(copy_prediction.shape) != tuple(copy_target.shape):
        raise DeltaTrainingError("copy prediction/target shape differs")
    if args.branch_state_mode == "shared_noisy_clean_field":
        if auxiliary.get("branch_state_mode") != "shared_noisy_clean_field":
            raise DeltaTrainingError("same-state auxiliary contract differs")
        shared_fields = (
            "input_vae_latents",
            "input_vae_rope",
            "vae_latents_mask",
            "timesteps",
            "vae_seqlen",
            "target_lens",
        )
        unequal = [
            name
            for name in shared_fields
            if not torch.equal(action_batch[name], copy_batch[name])
        ]
        if unequal:
            raise DeltaTrainingError(
                f"action/no-op renderer states differ: {unequal}"
            )
        if torch.equal(action_batch["input_ids"], copy_batch["input_ids"]):
            raise DeltaTrainingError("action/no-op text branches are identical")
        if (
            action_prediction.dtype != torch.bfloat16
            or copy_prediction.dtype != torch.bfloat16
            or auxiliary["shared_noisy"].dtype != torch.float32
        ):
            raise DeltaTrainingError(
                "same-state clean reconstruction requires fp32 noisy and native bf16 branches"
            )
        motion_loss, motion_parts = motion.differential_clean_motion_loss(
            action_prediction,
            copy_prediction,
            auxiliary["shared_noisy"],
            auxiliary["sigma"],
            auxiliary["target_clean"],
            auxiliary["source_clean"],
            latent_frames=legacy.LATENT_FRAMES,
            lags=tuple(args.temporal_lags),
            quotient_weight=args.quotient_weight,
            objective=args.motion_objective,
            causal_ema_decay=args.causal_ema_decay,
            charbonnier_scale=args.charbonnier_scale,
        )
        action_clean = motion_parts["action_clean"]
        noop_clean = motion_parts["noop_clean"]
        # Reviewed full-pair rows retain standard flow matching.  Using clean
        # MSE here would silently add a sigma**2 weight to that control arm.
        full_target_loss = torch.mean(
            (action_prediction.float() - action_target) ** 2
        )
        copy_loss = torch.mean(
            (noop_clean - auxiliary["source_clean"].float()) ** 2
        )
        action_field = motion_parts["predicted_clean_delta"]
        state_exact = torch.ones((), device=action_prediction.device)
        clean_field_weight = motion.clean_field_inverse_sigma_weight(
            auxiliary["sigma"], weight_floor=args.inverse_sigma_weight_floor
        ).mean()
    elif args.branch_state_mode == "separate_clean_paths":
        action_prediction = action_prediction.float()
        copy_prediction = copy_prediction.float()
        source_velocity = _flatten_target(auxiliary["source_velocity"]).float()
        full_target_loss = torch.mean((action_prediction - action_target) ** 2)
        copy_loss = torch.mean((copy_prediction - copy_target) ** 2)
        motion_loss, motion_parts = motion.differential_motion_loss(
            action_prediction,
            copy_prediction,
            action_target,
            source_velocity,
            latent_frames=legacy.LATENT_FRAMES,
            lags=tuple(args.temporal_lags),
            quotient_weight=args.quotient_weight,
            objective=args.motion_objective,
            causal_ema_decay=args.causal_ema_decay,
            charbonnier_scale=args.charbonnier_scale,
        )
        action_field = action_prediction - copy_prediction
        state_exact = torch.zeros((), device=action_prediction.device)
        clean_field_weight = torch.ones((), device=action_prediction.device)
    else:
        raise DeltaTrainingError(
            f"unknown branch state mode: {args.branch_state_mode!r}"
        )

    action_grid = action_field.reshape(
        int(action_field.shape[0]),
        legacy.LATENT_FRAMES,
        int(action_field.shape[1]) // legacy.LATENT_FRAMES,
        int(action_field.shape[2]),
    )
    boundary_gauge_loss = torch.mean(action_grid[:, 0] ** 2)
    if args.branch_state_mode == "shared_noisy_clean_field":
        # 1/sigma cancels the first-order sigma attenuation introduced by
        # x=y-sigma*v.  The high-noise factor applies only to the motion term;
        # no-op reconstruction remains active throughout the inference range.
        total = clean_field_weight * (
            args.motion_loss_weight * noise_weight * motion_loss
            + args.copy_loss_weight * copy_loss
            + args.boundary_gauge_loss_weight * boundary_gauge_loss
        ) + route.full_target_weight * full_target_loss
    else:
        # Exact v1 two-path ablation semantics.
        total = (
            args.motion_loss_weight * noise_weight * motion_loss
            + route.full_target_weight * full_target_loss
            + args.copy_loss_weight * copy_loss
            + args.boundary_gauge_loss_weight * boundary_gauge_loss
        )
    return total, {
        "motion": motion_loss,
        "motion_raw_delta": motion_parts["raw_delta"],
        "motion_quotient": motion_parts["temporal_quotient"],
        "motion_causal_boundary": motion_parts["causal_boundary"],
        "motion_causal_ema": motion_parts["causal_ema_charbonnier"],
        "motion_multiscale": motion_parts["multiscale_difference"],
        "high_noise_weight": noise_weight,
        "clean_field_weight": clean_field_weight,
        "full_target": full_target_loss,
        "full_target_weight": torch.tensor(
            route.full_target_weight, device=action_prediction.device
        ),
        "copy": copy_loss,
        "boundary_gauge": boundary_gauge_loss,
        "sigma": auxiliary["sigma"].float().mean(),
        "same_state_exact": state_exact,
    }


def _optimizer_parameter_names(named: Sequence[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in named]


def _checkpoint_parameter_digest(named: Sequence[tuple[str, Any]]) -> str:
    """Hash the exact current LoRA tensors bound to a saved receipt."""

    try:
        return legacy.trainable_parameters_digest(named)
    except legacy.TrainingContractError as error:
        raise _translate(error) from error


def _validate_loaded_parameter_digest(
    receipt: Mapping[str, Any], named: Sequence[tuple[str, Any]]
) -> str:
    adapter = receipt.get("adapter")
    expected = (
        adapter.get("checkpoint_parameter_digest")
        if isinstance(adapter, Mapping)
        else None
    )
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise DeltaTrainingError("resume receipt lacks checkpoint parameter digest")
    actual = _checkpoint_parameter_digest(named)
    if actual != expected:
        raise DeltaTrainingError("loaded LoRA tensors differ from resume receipt")
    return actual


def _stable_recursive_digest(value: Any) -> str:
    """Hash nested checkpoint state without depending on pickle serialization.

    Mapping order is canonicalized, list/tuple order and scalar types are bound,
    and tensors contribute their dtype, shape, and exact logical bytes.  Devices
    and strides are intentionally excluded so a GPU payload hashes identically
    after ``torch.load(..., map_location="cpu")``.
    """

    import torch

    digest = hashlib.sha256()

    def frame(tag: str, payload: bytes = b"") -> None:
        tag_bytes = tag.encode("ascii")
        digest.update(len(tag_bytes).to_bytes(4, "big"))
        digest.update(tag_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def mapping_key_bytes(key: Any) -> bytes:
        if key is None:
            return b"none:"
        if type(key) is bool:
            return b"bool:" + (b"1" if key else b"0")
        if type(key) is int:
            return b"int:" + str(key).encode("ascii")
        if type(key) is float:
            return b"float:" + struct.pack(">d", key)
        if isinstance(key, str):
            encoded = key.encode("utf-8")
            return b"str:" + len(encoded).to_bytes(8, "big") + encoded
        if isinstance(key, bytes):
            return b"bytes:" + len(key).to_bytes(8, "big") + key
        raise DeltaTrainingError(
            f"unsupported optimizer checkpoint mapping key: {type(key).__name__}"
        )

    def visit(candidate: Any) -> None:
        if candidate is None:
            frame("none")
        elif type(candidate) is bool:
            frame("bool", b"1" if candidate else b"0")
        elif type(candidate) is int:
            frame("int", str(candidate).encode("ascii"))
        elif type(candidate) is float:
            frame("float64", struct.pack(">d", candidate))
        elif isinstance(candidate, str):
            frame("str", candidate.encode("utf-8"))
        elif isinstance(candidate, bytes):
            frame("bytes", candidate)
        elif isinstance(candidate, torch.Tensor):
            tensor = candidate.detach()
            if tensor.layout != torch.strided:
                raise DeltaTrainingError(
                    "optimizer checkpoint digest supports only strided tensors"
                )
            tensor = tensor.contiguous()
            metadata = legacy.canonical_json_bytes(
                {
                    "dtype": str(tensor.dtype),
                    "shape": [int(dimension) for dimension in tensor.shape],
                }
            )
            byte_view = tensor.reshape(-1).view(torch.uint8).cpu()
            frame("tensor-metadata", metadata)
            frame("tensor-bytes", byte_view.numpy().tobytes(order="C"))
        elif isinstance(candidate, Mapping):
            ordered = sorted(
                ((mapping_key_bytes(key), key) for key in candidate),
                key=lambda item: item[0],
            )
            frame("mapping-begin", len(ordered).to_bytes(8, "big"))
            for encoded_key, key in ordered:
                frame("mapping-key", encoded_key)
                visit(candidate[key])
            frame("mapping-end")
        elif isinstance(candidate, list):
            frame("list-begin", len(candidate).to_bytes(8, "big"))
            for item in candidate:
                visit(item)
            frame("list-end")
        elif isinstance(candidate, tuple):
            frame("tuple-begin", len(candidate).to_bytes(8, "big"))
            for item in candidate:
                visit(item)
            frame("tuple-end")
        else:
            raise DeltaTrainingError(
                "unsupported optimizer checkpoint value: "
                f"{type(candidate).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def _optimizer_checkpoint_payload(
    *,
    optimizer: Any,
    global_step: int,
    immutable: Mapping[str, Any],
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    """Build the one object used for both digesting and ``torch.save``."""

    return {
        "schema_version": OPTIMIZER_SCHEMA,
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "immutable_contract": dict(immutable),
        "parameter_names": list(parameter_names),
    }


def _validate_loaded_optimizer_digest(
    receipt: Mapping[str, Any], payload: Mapping[str, Any]
) -> str:
    optimizer_receipt = receipt.get("optimizer")
    expected = (
        optimizer_receipt.get("checkpoint_state_digest")
        if isinstance(optimizer_receipt, Mapping)
        else None
    )
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise DeltaTrainingError("resume receipt lacks optimizer checkpoint state digest")
    actual = _stable_recursive_digest(payload)
    if actual != expected:
        raise DeltaTrainingError("loaded optimizer state differs from resume receipt")
    return actual


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
    clean_field_mode = args.branch_state_mode in (
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    )
    bridge_mode = args.branch_state_mode == "source_target_bridge_clean_field"
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
        "routing_file_sha256": router.file_sha256,
        "expected_routing_jsonl_sha256": args.expected_routing_jsonl_sha256,
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
        "paired_cells": (
            [
                "source_query_action",
                "source_query_noop",
                "executable_target_query_action",
                "executable_target_query_noop",
            ]
            if bridge_mode
            else ["action", "exact_copy"]
        ),
        "posterior_statistic": "mode",
        "branch_state_mode": args.branch_state_mode,
        "minimum_training_sigma": (
            float(args.minimum_training_sigma)
            if clean_field_mode
            else 0.0
        ),
        "inverse_sigma_weight_floor": (
            float(args.inverse_sigma_weight_floor)
            if clean_field_mode
            else None
        ),
        "shared_source_sigma_noise": True,
        "exact_same_noisy_query": (
            clean_field_mode
        ),
        "clean_reconstruction_formula": (
            "x_clean = y - sigma * velocity"
            if clean_field_mode
            else None
        ),
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
            if clean_field_mode
            else None
        ),
        "training_sigma_representation": (
            "cpu_fp32_0d"
            if clean_field_mode
            else None
        ),
        "branch_prediction_dtype_before_clean_reconstruction": (
            "bfloat16" if clean_field_mode else None
        ),
        "predicted_clean_delta_formula": (
            "-sigma * (v_action - v_noop)"
            if clean_field_mode
            else None
        ),
        "target_clean_delta_formula": (
            "executable_target_clean - source_clean"
            if clean_field_mode
            else None
        ),
        "target_projection": (
            "executable_target=source+Q0(raw_target-source)"
            if bridge_mode
            else None
        ),
        "target_projection_idempotent": True if bridge_mode else None,
        "motion_loss_multiplier": (
            "1 / sigma"
            if bridge_mode
            else (
                "high_noise(sigma) / max(sigma, inverse_sigma_weight_floor)"
                if clean_field_mode
                else "high_noise(sigma)"
            )
        ),
        "copy_boundary_loss_multiplier": (
            "not_enabled"
            if bridge_mode
            else (
                "1 / max(sigma, inverse_sigma_weight_floor)"
                if clean_field_mode
                else "1"
            )
        ),
        "clean_field_loss_weight_range": (
            [1.0, 1.0 / float(args.inverse_sigma_weight_floor)]
            if clean_field_mode
            else None
        ),
        "motion_objective": args.motion_objective,
        "motion_representation": _motion_representation_name(args),
        "temporal_lags": list(args.temporal_lags),
        "quotient_weight": float(args.quotient_weight),
        "motion_loss_weight": float(args.motion_loss_weight),
        "copy_loss_weight": float(args.copy_loss_weight),
        "boundary_gauge_loss_weight": float(args.boundary_gauge_loss_weight),
        "boundary_gauge": "zero_first_latent_phase_of_raw_predicted_clean_delta",
        "bridge_fractions": list(BRIDGE_FRACTIONS) if bridge_mode else None,
        "bridge_consistency_weight": float(args.bridge_consistency_weight),
        "causal_ema_decay": (
            None if bridge_mode else float(args.causal_ema_decay)
        ),
        "charbonnier_scale": float(args.charbonnier_scale),
        "inference_sigma_schedule_sha256": (
            sigma_strata.SCHEDULE_SHA256 if bridge_mode else None
        ),
        "inference_sigma_selector": (
            "absolute_global_step_mod_40" if bridge_mode else None
        ),
        "high_noise_floor": float(args.high_noise_floor),
        "high_noise_power": float(args.high_noise_power),
        "noop_instruction_sha256": hashlib.sha256(
            args.noop_instruction.encode("utf-8")
        ).hexdigest(),
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
    optimizer_receipt = prior.get("optimizer")
    optimizer_digest = (
        optimizer_receipt.get("checkpoint_state_digest")
        if isinstance(optimizer_receipt, Mapping)
        else None
    )
    if (
        not isinstance(optimizer_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", optimizer_digest) is None
    ):
        raise DeltaTrainingError("resume receipt lacks optimizer checkpoint state digest")
    step = prior.get("global_step")
    if type(step) is not int or step < 0:
        raise DeltaTrainingError("resume global_step is invalid")
    return step


def _supervision_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Describe enabled losses separately from the mandatory field branches."""

    clean_field_mode = args.branch_state_mode in (
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    )
    bridge_mode = args.branch_state_mode == "source_target_bridge_clean_field"
    return {
        "inference_conditions": ["source_video", "edit_instruction"],
        "target_used_as_condition": False,
        "target_video_used_as_external_condition": False,
        "projected_target_used_as_training_query": bridge_mode,
        "external_mask_track_pose_trajectory": False,
        "paired_action_noop_forward_every_optimizer_step": True,
        "action_noop_forwards_per_optimizer_step": 4 if bridge_mode else 2,
        "counterfactual_noop_forward": True,
        "branch_state_mode": args.branch_state_mode,
        "exact_same_noisy_query": (
            clean_field_mode
        ),
        "minimum_training_sigma": (
            float(args.minimum_training_sigma)
            if clean_field_mode
            else 0.0
        ),
        "inverse_sigma_weight_floor": (
            float(args.inverse_sigma_weight_floor)
            if clean_field_mode
            else None
        ),
        "clean_reconstruction_formula": (
            "x_clean = y - sigma * velocity"
            if clean_field_mode
            else None
        ),
        "clean_reconstruction_numeric_program": (
            "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
            if clean_field_mode
            else None
        ),
        "training_sigma_representation": (
            "cpu_fp32_0d"
            if clean_field_mode
            else None
        ),
        "branch_prediction_dtype_before_clean_reconstruction": (
            "bfloat16" if clean_field_mode else None
        ),
        "predicted_clean_delta_formula": (
            "-sigma * (v_action - v_noop)"
            if clean_field_mode
            else None
        ),
        "target_clean_delta_formula": (
            "executable_target_clean - source_clean"
            if clean_field_mode
            else None
        ),
        "target_projection": (
            "executable_target=source+Q0(raw_target-source)"
            if bridge_mode
            else None
        ),
        "target_projection_idempotent": True if bridge_mode else None,
        "motion_loss_multiplier": (
            "1 / sigma"
            if bridge_mode
            else (
                "high_noise(sigma) / max(sigma, inverse_sigma_weight_floor)"
                if clean_field_mode
                else "high_noise(sigma)"
            )
        ),
        "copy_boundary_loss_multiplier": (
            "not_enabled"
            if bridge_mode
            else (
                "1 / max(sigma, inverse_sigma_weight_floor)"
                if clean_field_mode
                else "1"
            )
        ),
        "only_text_condition_differs": (
            clean_field_mode
        ),
        "copy_calibration_enabled": float(args.copy_loss_weight) > 0.0,
        "copy_calibration_weight": float(args.copy_loss_weight),
        "boundary_gauge_enabled": float(args.boundary_gauge_loss_weight) > 0.0,
        "boundary_gauge_loss_weight": float(args.boundary_gauge_loss_weight),
        "boundary_gauge_field": "raw_predicted_action_minus_noop_clean_field",
        "boundary_gauge_target": "zero_first_latent_phase",
        "boundary_gauge_uses_target_appearance": False,
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
        "motion_representation": _motion_representation_name(args),
        "temporal_quotient_enabled": (
            float(args.motion_loss_weight) > 0.0
            and args.motion_objective == "quotient_multilag"
            and float(args.quotient_weight) > 0.0
        ),
        "causal_boundary_quotient_enabled": (
            float(args.motion_loss_weight) > 0.0
            and (
                (
                    args.motion_objective == "causal_boundary_multilag"
                    and float(args.quotient_weight) > 0.0
                )
                or args.motion_objective == "causal_boundary_charbonnier"
            )
        ),
        "causal_boundary_projection_enabled": (
            args.motion_objective == "causal_boundary_charbonnier"
        ),
        "temporal_quotient_weight": float(args.quotient_weight),
        "multiscale_enabled": (
            float(args.motion_loss_weight) > 0.0
            and args.motion_objective in (
                "quotient_multilag", "causal_boundary_multilag"
            )
            and float(args.quotient_weight) < 1.0
        ),
        "temporal_lags": list(args.temporal_lags),
        "causal_boundary_gauge_loss_weight": float(
            args.boundary_gauge_loss_weight
        ),
        "bridge_endpoints": list(BRIDGE_FRACTIONS) if bridge_mode else None,
        "bridge_consistency_enabled": bridge_mode,
        "bridge_consistency_weight": float(args.bridge_consistency_weight),
        "bridge_query_formula": (
            "y_beta=(1-sigma)*((1-beta)*source+beta*executable_target)"
            "+sigma*epsilon"
            if bridge_mode
            else None
        ),
        "causal_ema_enabled": args.motion_objective == "causal_ema_charbonnier",
        "causal_ema_decay": (
            None if bridge_mode else float(args.causal_ema_decay)
        ),
        "charbonnier_scale": float(args.charbonnier_scale),
        "inference_sigma_stratification": (
            "exact_40_step_flow_shift_5_cycle" if bridge_mode else None
        ),
        "inference_sigma_schedule_sha256": (
            sigma_strata.SCHEDULE_SHA256 if bridge_mode else None
        ),
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
    optimizer_payload: Mapping[str, Any],
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
        "inference_sigma_strata": (
            sigma_strata.build_sigma_strata_receipt(
                completed_optimizer_steps=global_step
            )
            if args.branch_state_mode == "source_target_bridge_clean_field"
            else None
        ),
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
            "checkpoint_parameter_digest": _checkpoint_parameter_digest(
                named_trainable
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
            "checkpoint_state_digest": _stable_recursive_digest(
                optimizer_payload
            ),
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
    optimizer_payload: Mapping[str, Any],
    output: Path,
    global_step: int,
    receipt: Mapping[str, Any],
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
        torch.save(optimizer_payload, temporary / "optimizer.pt")
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
    from diffusers import UniPCMultistepScheduler
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
        router = motion.ReviewRouter.load(
            args.routing_jsonl, default_tier=args.unreviewed_tier
        )
    except motion.MotionContractError as error:
        raise _translate(error) from error
    eligible_routes = _build_eligible_routes(dataset, router)
    _validate_v4_strict_router(args, router, eligible_routes)
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
    # Eval mode preserves gradients while preventing stochastic frozen-base
    # layers from masquerading as an action/no-op text effect.  LoRA dropout
    # is fixed to zero by the immutable adapter contract.
    model.eval()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    named_trainable = legacy.trainable_lora_parameters(model)
    initialization_digest = legacy.synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    if prior_receipt is not None:
        _validate_loaded_parameter_digest(prior_receipt, named_trainable)
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
        if not isinstance(state, Mapping):
            raise DeltaTrainingError("resume optimizer payload is not a mapping")
        _validate_loaded_optimizer_digest(prior_receipt, state)
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
    scheduler_kwargs = legacy.noise_scheduler_kwargs()
    if args.branch_state_mode in (
        "shared_noisy_clean_field",
        "source_target_bridge_clean_field",
    ):
        scheduler_kwargs["noise_tmin"] = float(args.minimum_training_sigma)
    scheduler = NoiseScheduler(**scheduler_kwargs)
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=sigma_strata.FLOW_SHIFT,
    )
    sigma_strata.audit_runtime_unipc_schedule(inference_scheduler)

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
        selected_stratum = sigma_strata.select_sigma_stratum(global_step)
        try:
            if args.branch_state_mode == "source_target_bridge_clean_field":
                endpoint_batches = _prepare_bridge_batches(
                    raw_row=raw_row,
                    tokenizer=tokenizer,
                    rope=rope,
                    vae_mean=vae_mean,
                    vae_std=vae_std,
                    z_dim=z_dim,
                    scheduler=scheduler,
                    noop_instruction=args.noop_instruction,
                    minimum_training_sigma=args.minimum_training_sigma,
                    process_renderer_sample=process_renderer_sample,
                    selected_stratum=selected_stratum,
                )
            else:
                action_batch, copy_batch, auxiliary = _prepare_paired_batches(
                    raw_row=raw_row,
                    tokenizer=tokenizer,
                    rope=rope,
                    vae_mean=vae_mean,
                    vae_std=vae_std,
                    z_dim=z_dim,
                    scheduler=scheduler,
                    noop_instruction=args.noop_instruction,
                    branch_state_mode=args.branch_state_mode,
                    minimum_training_sigma=args.minimum_training_sigma,
                    process_renderer_sample=process_renderer_sample,
                )
        except (
            legacy.TrainingContractError,
            motion.MotionContractError,
            sigma_strata.InferenceSigmaStrataError,
        ) as error:
            raise _translate(error) from error
        if args.branch_state_mode == "source_target_bridge_clean_field":
            moved_endpoints = {}
            for endpoint_name, (
                endpoint_action,
                endpoint_noop,
                endpoint_auxiliary,
            ) in endpoint_batches.items():
                moved_endpoints[endpoint_name] = (
                    legacy._move_batch(endpoint_action, device),
                    legacy._move_batch(endpoint_noop, device),
                    _move_auxiliary_to_device(
                        endpoint_auxiliary,
                        device=device,
                        branch_state_mode=args.branch_state_mode,
                    ),
                )
        else:
            action_batch = legacy._move_batch(action_batch, device)
            copy_batch = legacy._move_batch(copy_batch, device)
            auxiliary = _move_auxiliary_to_device(
                auxiliary,
                device=device,
                branch_state_mode=args.branch_state_mode,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.branch_state_mode == "source_target_bridge_clean_field":
                loss, components = _bridge_losses(
                    renderer=renderer,
                    endpoints=moved_endpoints,
                    route=route,
                    args=args,
                )
            else:
                loss, components = _losses(
                    renderer=renderer,
                    action_batch=action_batch,
                    copy_batch=copy_batch,
                    auxiliary=auxiliary,
                    route=route,
                    args=args,
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
        if args.branch_state_mode == "source_target_bridge_clean_field":
            last_metrics.update(
                {
                    "sigma_schedule_index": float(
                        selected_stratum.schedule_index
                    ),
                    "sigma_timestep": float(selected_stratum.timestep),
                }
            )
        if distributed.rank == 0:
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "row": row_index,
                        "iid": route.iid,
                        "tier": route.tier,
                        "seed": current_seed,
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if args.save_every > 0 and global_step % args.save_every == 0:
            optimizer_payload = _optimizer_checkpoint_payload(
                optimizer=optimizer,
                global_step=global_step,
                immutable=immutable,
                parameter_names=parameter_names,
            )
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
                optimizer_payload=optimizer_payload,
                resumed_from=str(resume_paths.root) if resume_paths else None,
                initialized_from=initialized_from,
            )
            _save_checkpoint(
                model=model,
                optimizer_payload=optimizer_payload,
                output=output,
                global_step=global_step,
                receipt=receipt,
                rank=distributed.rank,
            )
            last_saved = global_step

    if last_saved != global_step:
        optimizer_payload = _optimizer_checkpoint_payload(
            optimizer=optimizer,
            global_step=global_step,
            immutable=immutable,
            parameter_names=parameter_names,
        )
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
            optimizer_payload=optimizer_payload,
            resumed_from=str(resume_paths.root) if resume_paths else None,
            initialized_from=initialized_from,
        )
        _save_checkpoint(
            model=model,
            optimizer_payload=optimizer_payload,
            output=output,
            global_step=global_step,
            receipt=receipt,
            rank=distributed.rank,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train Bernini Prior-Guided Tangent Trust-Region LoRA (v5).

This entry point is deliberately separate from the v4 counterfactual-field
trainer.  At each of the two source/target bridge endpoints it evaluates four
branches on one identical noisy query:

* frozen negative, no-op, and action branches (adapter disabled, no grad);
* one adapted action branch (adapter enabled, with grad).

The three conditional branches are converted to clean fields with Bernini's
official momentum-zero APG numerical program.  The frozen action-minus-no-op
difference is projected through ``Q0`` before it becomes the executable
generator-native motion prior, so its phase-zero appearance offset cannot be
copied into every output phase.  The LoRA may only add the bounded causal
correction implemented by :mod:`prior_guided_tangent`; it is never asked to
synthesize the whole motion field from 359 synthetic pairs.

Targets, masks, flow, tracks, pose, and trajectories are not inference
conditions.  The target posterior mode is used only by the offline teacher
loss and by the second training-only bridge query.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
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

import inference_sigma_strata as sigma_strata  # noqa: E402
import motion_residual as motion  # noqa: E402
import prior_guided_tangent as pgt  # noqa: E402
import train_delta_lora as v4  # noqa: E402
import train_lora as legacy  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


RECEIPT_SCHEMA = "bernini-r-1p3b-prior-tangent-lora-receipt-v5"
OPTIMIZER_SCHEMA = "bernini-r-1p3b-prior-tangent-lora-optimizer-v5"
METHOD_NAME = pgt.METHOD_NAME
BRIDGE_FRACTIONS = (0.0, 1.0)
FIELD_LOSS_WEIGHT = 1.0
BRIDGE_LOSS_WEIGHT = 0.05
LATE_REPLAY_LOSS_WEIGHT = 0.10
LEARNING_RATE = 2.0e-5
APG_GUIDANCE_SCALE = 4.0
APG_ETA = 0.5
APG_NORM_THRESHOLD = 50.0
APG_MOMENTUM = 0.0
CHARBONNIER_SCALE = 0.1
SHARED_STATE_FIELDS = (
    "input_vae_latents",
    "input_vae_rope",
    "vae_latents_mask",
    "timesteps",
    "vae_seqlen",
    "target_lens",
)
TEXT_FIELDS = ("input_ids", "attention_mask", "t5_input_lens")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
NEGATIVE_PROMPT_SHA256 = (
    "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
)
STRICT_ROUTING_SHA256 = (
    "0da09787889687726d9161b0c74b8df5d58226f6e431632b317891d630ef49eb"
)


class PriorTangentTrainingError(RuntimeError):
    """Raised before an update when any v5 invariant differs."""


@dataclass(frozen=True)
class EndpointFields:
    """Auditable clean fields and losses from one bridge endpoint."""

    base_negative: Any
    base_noop: Any
    base_action: Any
    adapted_action: Any
    student_executed: Any
    teacher_executed: Any
    field_loss: Any
    replay_loss: Any


def _translate(error: Exception) -> PriorTangentTrainingError:
    return PriorTangentTrainingError(str(error))


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PriorTangentTrainingError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise PriorTangentTrainingError(f"{label} must contain one JSON object")
    return value


def _validate_digest_object(value: Mapping[str, Any], *, label: str) -> None:
    candidate = dict(value)
    declared = candidate.pop("receipt_digest", None)
    if not isinstance(declared, str) or legacy.object_sha256(candidate) != declared:
        raise PriorTangentTrainingError(f"{label} receipt digest differs")


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
    """Build the v5 CLI while retaining v4's audited infrastructure flags."""

    parser = v4.build_parser()
    parser.description = "Train 81f Bernini Prior-Guided Tangent LoRA v5"
    parser.set_defaults(
        learning_rate=LEARNING_RATE,
        lora_scope="cross_q",
        branch_state_mode="source_target_bridge_clean_field",
        motion_loss_weight=FIELD_LOSS_WEIGHT,
        copy_loss_weight=0.0,
        boundary_gauge_loss_weight=0.0,
        anchor_loss_weight=0.0,
        motion_objective="causal_boundary_charbonnier",
        bridge_consistency_weight=BRIDGE_LOSS_WEIGHT,
        charbonnier_scale=CHARBONNIER_SCALE,
        high_noise_floor=1.0,
        unreviewed_tier="reject",
    )
    parser.add_argument(
        "--late-replay-loss-weight",
        type=float,
        default=LATE_REPLAY_LOSS_WEIGHT,
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="fixed internal Bernini APG negative prompt",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    """Reject every configuration that would no longer be the v5 main arm."""

    if args.num_frames != 81 or legacy.NUM_FRAMES != 81 or legacy.LATENT_FRAMES != 21:
        raise PriorTangentTrainingError("v5 requires exact 81-frame / 21-phase data")
    if args.max_steps <= 0 or args.save_every < 0:
        raise PriorTangentTrainingError(
            "max_steps must be positive and save_every non-negative"
        )
    if args.resume and args.init_adapter_checkpoint:
        raise PriorTangentTrainingError(
            "resume and init-adapter-checkpoint are mutually exclusive"
        )
    if args.allow_incomplete_dataset:
        raise PriorTangentTrainingError("v5 requires the complete 644-row dataset")
    fixed = {
        "learning_rate": LEARNING_RATE,
        "motion_loss_weight": FIELD_LOSS_WEIGHT,
        "bridge_consistency_weight": BRIDGE_LOSS_WEIGHT,
        "late_replay_loss_weight": LATE_REPLAY_LOSS_WEIGHT,
        "copy_loss_weight": 0.0,
        "boundary_gauge_loss_weight": 0.0,
        "anchor_loss_weight": 0.0,
        "charbonnier_scale": CHARBONNIER_SCALE,
        "high_noise_floor": 1.0,
    }
    for name, expected in fixed.items():
        value = float(getattr(args, name))
        if not math.isfinite(value) or value != expected:
            raise PriorTangentTrainingError(
                f"v5 fixes {name} to {expected!r}, got {value!r}"
            )
    if not math.isfinite(float(args.weight_decay)) or float(args.weight_decay) < 0.0:
        raise PriorTangentTrainingError("weight_decay must be finite and non-negative")
    if not math.isfinite(float(args.max_grad_norm)) or float(args.max_grad_norm) <= 0.0:
        raise PriorTangentTrainingError("max_grad_norm must be finite and positive")
    if args.lora_scope != "cross_q":
        raise PriorTangentTrainingError("v5 requires the 30-module cross_q scope")
    if args.branch_state_mode != "source_target_bridge_clean_field":
        raise PriorTangentTrainingError("v5 requires both bridge endpoints")
    if args.motion_objective != "causal_boundary_charbonnier":
        raise PriorTangentTrainingError("v5 fixes robust clean-field supervision")
    if (
        float(args.quotient_weight) != 0.5
        or float(args.causal_ema_decay) != 0.5
        or float(args.high_noise_power) != 2.0
        or list(args.temporal_lags) != [1, 2, 4]
    ):
        raise PriorTangentTrainingError(
            "v5 rejects altered inherited v4 ablation knobs"
        )
    if args.unreviewed_tier != "reject" or args.routing_jsonl is None:
        raise PriorTangentTrainingError("v5 requires an explicit strict routing file")
    if (
        not isinstance(args.expected_routing_jsonl_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_routing_jsonl_sha256) is None
    ):
        raise PriorTangentTrainingError("v5 requires the pinned routing SHA-256")
    if args.expected_routing_jsonl_sha256 != STRICT_ROUTING_SHA256:
        raise PriorTangentTrainingError("v5 routing differs from strict-359")
    if (
        not math.isfinite(float(args.minimum_training_sigma))
        or float(args.minimum_training_sigma) != 0.1
    ):
        raise PriorTangentTrainingError("v5 fixes minimum_training_sigma to 0.1")
    if float(args.inverse_sigma_weight_floor) != float(
        sigma_strata.PINNED_POSITIVE_SIGMAS[-1]
    ):
        raise PriorTangentTrainingError(
            "v5 inverse-sigma floor must equal the final positive UniPC sigma"
        )
    if not isinstance(args.noop_instruction, str) or not args.noop_instruction.strip():
        raise PriorTangentTrainingError("no-op instruction must be non-empty")
    if args.noop_instruction != motion.DEFAULT_NOOP_INSTRUCTION:
        raise PriorTangentTrainingError("v5 pins the semantic no-op instruction")
    if not isinstance(args.negative_prompt, str) or not args.negative_prompt.strip():
        raise PriorTangentTrainingError("negative prompt must be non-empty")
    if args.negative_prompt != DEFAULT_NEGATIVE_PROMPT:
        raise PriorTangentTrainingError("v5 pins the official Bernini negative prompt")
    if (
        hashlib.sha256(args.negative_prompt.encode("utf-8")).hexdigest()
        != NEGATIVE_PROMPT_SHA256
    ):
        raise PriorTangentTrainingError("v5 negative prompt hash differs")
    if args.noop_instruction == args.negative_prompt:
        raise PriorTangentTrainingError("no-op and negative prompts must differ")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", getattr(args, name)) is None:
            raise PriorTangentTrainingError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(args, name)) is None:
            raise PriorTangentTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != legacy.CHECKPOINT_TREE_SHA256:
        raise PriorTangentTrainingError(
            "checkpoint identity differs from the audited Bernini-R 1.3B tree"
        )
    if args.expected_bernini_commit.lower() != legacy.BERNINI_OFFICIAL_COMMIT:
        raise PriorTangentTrainingError("v5 pins the audited Bernini commit")
    if args.expected_veomni_commit.lower() != legacy.VEOMNI_TESTED_COMMIT:
        raise PriorTangentTrainingError("v5 pins the audited VeOmni commit")
    if legacy.LORA_RANK != 8 or legacy.LORA_ALPHA != 8:
        raise PriorTangentTrainingError("v5 fixes LoRA rank/alpha to 8/8")
    pgt.TangentTrustRegionConfig().validate()
    gamma = pgt.correction_gamma_schedule()
    if (
        len(gamma) != 40
        or gamma[24] != 1.0
        or gamma[34] != 0.0
        or gamma[35:] != (0.0,) * 5
    ):
        raise PriorTangentTrainingError("v5 gamma schedule differs")


def _as_phase_grid(value: Any) -> Any:
    """Expose the temporal phase axis before applying Q0/trust-region logic."""

    if getattr(value, "ndim", None) != 3 or int(value.shape[0]) != 1:
        raise PriorTangentTrainingError("packed field must have shape [1,N,D]")
    tokens = int(value.shape[1])
    if tokens <= 0 or tokens % legacy.LATENT_FRAMES:
        raise PriorTangentTrainingError("packed tokens do not divide into 21 phases")
    return value.reshape(
        1,
        legacy.LATENT_FRAMES,
        tokens // legacy.LATENT_FRAMES,
        int(value.shape[2]),
    )


def _from_phase_grid(value: Any) -> Any:
    if getattr(value, "ndim", None) != 4 or int(value.shape[0]) != 1:
        raise PriorTangentTrainingError(
            "phase field must have shape [1,21,spatial,channels]"
        )
    if int(value.shape[1]) != legacy.LATENT_FRAMES:
        raise PriorTangentTrainingError("phase field does not have 21 phases")
    return value.reshape(1, int(value.shape[1]) * int(value.shape[2]), int(value.shape[3]))


def _official_momentum_zero_apg(
    conditional_clean: Any,
    negative_clean: Any,
) -> Any:
    """Bernini ``normalized_guidance`` on a packed-equivalent phase grid.

    Each phase's channel/spatial vector is represented as ``[S,D]`` rather
    than ``[C,H,W]``.  This is a permutation of the same Wan patch elements;
    the official APG reductions are consequently over ``(-1,-2)`` here.
    Momentum is exactly zero, so no cross-step state enters training.
    """

    import torch
    import torch.nn.functional as torch_f

    if (
        tuple(conditional_clean.shape) != tuple(negative_clean.shape)
        or conditional_clean.ndim != 4
        or conditional_clean.dtype != torch.float32
        or negative_clean.dtype != torch.float32
        or conditional_clean.device != negative_clean.device
    ):
        raise PriorTangentTrainingError(
            "APG clean fields must share fp32 [B,phase,spatial,channel] geometry"
        )
    if APG_MOMENTUM != 0.0:
        raise PriorTangentTrainingError("training APG momentum must remain zero")
    difference = conditional_clean - negative_clean
    ones = torch.ones_like(difference)
    difference_norm = difference.norm(p=2, dim=(-1, -2), keepdim=True)
    scale_factor = torch.minimum(
        ones, APG_NORM_THRESHOLD / difference_norm
    )
    difference = difference * scale_factor
    orthogonal_source = difference.double()
    condition_direction = torch_f.normalize(
        conditional_clean.double(), dim=(-1, -2)
    )
    parallel = (
        (orthogonal_source * condition_direction).sum(
            dim=(-1, -2), keepdim=True
        )
        * condition_direction
    )
    orthogonal = orthogonal_source - parallel
    normalized = orthogonal.to(difference.dtype) + APG_ETA * parallel.to(
        difference.dtype
    )
    guided = negative_clean + APG_GUIDANCE_SCALE * normalized
    if guided.dtype != torch.float32 or not bool(torch.isfinite(guided).all()):
        raise PriorTangentTrainingError("momentum-zero APG field is non-finite")
    return guided


def _assert_same_endpoint_state(
    negative_batch: Mapping[str, Any],
    noop_batch: Mapping[str, Any],
    action_batch: Mapping[str, Any],
) -> None:
    import torch

    for name in SHARED_STATE_FIELDS:
        values = [branch.get(name) for branch in (negative_batch, noop_batch, action_batch)]
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise PriorTangentTrainingError(f"missing tensor branch state {name}")
        if not torch.equal(values[0], values[1]) or not torch.equal(values[0], values[2]):
            raise PriorTangentTrainingError(
                f"negative/noop/action endpoint state differs at {name}"
            )
    ids = [branch["input_ids"] for branch in (negative_batch, noop_batch, action_batch)]
    if torch.equal(ids[0], ids[1]) or torch.equal(ids[0], ids[2]) or torch.equal(ids[1], ids[2]):
        raise PriorTangentTrainingError(
            "negative/noop/action text branches must be pairwise distinct"
        )


def _official_negative_text_fields(
    tokenizer: Any,
    negative_prompt: str,
) -> dict[str, Any]:
    """Tokenize Bernini's unconditional prompt verbatim for training.

    ``process_renderer_sample`` is intentionally not involved: that positive
    training path prepends the mv2v system message and applies Wan
    ``prompt_clean``.  Official renderer inference passes ``DEFAULT_NEG_PROMPT``
    directly to the tokenizer.  Keeping only its real length ``L`` is
    equivalent to inference's fixed 512-token tensor after
    ``get_t5_text_embeddings`` selects the valid tokens and zero-pads the
    resulting embeddings.
    """

    import torch

    if (
        not isinstance(negative_prompt, str)
        or hashlib.sha256(negative_prompt.encode("utf-8")).hexdigest()
        != NEGATIVE_PROMPT_SHA256
    ):
        raise PriorTangentTrainingError(
            "official renderer unconditional prompt hash differs"
        )
    encoded = tokenizer(
        negative_prompt,
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    def field(name: str) -> Any:
        value = getattr(encoded, name, None)
        if value is None and isinstance(encoded, Mapping):
            value = encoded.get(name)
        return value

    input_ids = field("input_ids")
    attention_mask = field("attention_mask")
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(attention_mask, torch.Tensor)
        or input_ids.ndim != 2
        or attention_mask.ndim != 2
        or int(input_ids.shape[0]) != 1
        or tuple(input_ids.shape) != tuple(attention_mask.shape)
    ):
        raise PriorTangentTrainingError(
            "negative tokenizer must return matching [1,L] ids and attention"
        )
    length = int(input_ids.shape[1])
    if length <= 0 or length > 512:
        raise PriorTangentTrainingError(
            "negative tokenizer valid length must lie in [1,512]"
        )
    if (
        not bool(torch.all((attention_mask == 0) | (attention_mask == 1)))
        or int(attention_mask.sum().item()) != length
    ):
        raise PriorTangentTrainingError(
            "verbatim negative tokenization must contain exactly L valid tokens"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "t5_input_lens": torch.tensor(
            [[length]], dtype=torch.long, device=input_ids.device
        ),
    }


def _guided_clean(
    *,
    shared_noisy: Any,
    sigma: Any,
    negative_velocity: Any,
    conditional_velocity: Any,
) -> tuple[Any, Any]:
    """Reconstruct raw clean fields and apply official momentum-zero APG."""

    try:
        negative_clean = tri.pinned_raw_condition_clean(
            shared_noisy, negative_velocity, sigma
        )
        condition_clean = tri.pinned_raw_condition_clean(
            shared_noisy, conditional_velocity, sigma
        )
    except tri.TriBranchHookError as error:
        raise _translate(error) from error
    negative_grid = _as_phase_grid(negative_clean)
    condition_grid = _as_phase_grid(condition_clean)
    return negative_grid, _official_momentum_zero_apg(
        condition_grid, negative_grid
    )


def _endpoint_fields(
    *,
    renderer: Any,
    adapter_controller: Any,
    negative_batch: Mapping[str, Any],
    noop_batch: Mapping[str, Any],
    action_batch: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    step_index: int,
    config: pgt.TangentTrustRegionConfig = pgt.TangentTrustRegionConfig(),
) -> EndpointFields:
    """Execute the exact four-forward v5 cell for one bridge endpoint."""

    import torch

    _assert_same_endpoint_state(negative_batch, noop_batch, action_batch)
    if auxiliary.get("branch_state_mode") != "source_target_bridge_clean_field":
        raise PriorTangentTrainingError("endpoint auxiliary is not the v4 bridge state")
    shared_noisy = auxiliary.get("shared_noisy")
    sigma = auxiliary.get("sigma")
    if not isinstance(shared_noisy, torch.Tensor) or shared_noisy.dtype != torch.float32:
        raise PriorTangentTrainingError("shared noisy query must be fp32")

    # Ordering is part of the receipt: all base branches are both adapter-off
    # and no-grad; the only differentiable transformer call is adapted action.
    with torch.no_grad():
        with adapter_controller.disable_adapter():
            base_negative_velocity = motion.renderer_velocity_prediction(
                renderer, negative_batch
            )
            base_noop_velocity = motion.renderer_velocity_prediction(
                renderer, noop_batch
            )
            base_action_velocity = motion.renderer_velocity_prediction(
                renderer, action_batch
            )
    adapted_action_velocity = motion.renderer_velocity_prediction(
        renderer, action_batch
    )
    velocities = (
        base_negative_velocity,
        base_noop_velocity,
        base_action_velocity,
        adapted_action_velocity,
    )
    if any(
        tuple(value.shape) != tuple(shared_noisy.shape)
        or value.dtype != torch.bfloat16
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise PriorTangentTrainingError(
            "all four branches must be finite native-bf16 fields on the shared query"
        )
    if any(value.requires_grad for value in velocities[:3]):
        raise PriorTangentTrainingError("frozen base branches retained a gradient graph")
    if not adapted_action_velocity.requires_grad:
        raise PriorTangentTrainingError("adapted action branch has no LoRA gradient graph")

    negative_clean, base_noop = _guided_clean(
        shared_noisy=shared_noisy,
        sigma=sigma,
        negative_velocity=base_negative_velocity,
        conditional_velocity=base_noop_velocity,
    )
    _, base_action = _guided_clean(
        shared_noisy=shared_noisy,
        sigma=sigma,
        negative_velocity=base_negative_velocity,
        conditional_velocity=base_action_velocity,
    )
    _, adapted_action = _guided_clean(
        shared_noisy=shared_noisy,
        sigma=sigma,
        negative_velocity=base_negative_velocity,
        conditional_velocity=adapted_action_velocity,
    )
    source = _as_phase_grid(auxiliary["source_clean"].float())
    target = _as_phase_grid(auxiliary["target_clean"].float())
    try:
        student = pgt.student_executed_field(
            base_action,
            base_noop,
            adapted_action,
            step_index=step_index,
            config=config,
        )
        with torch.no_grad():
            teacher = pgt.teacher_executed_field(
                source,
                target,
                base_action,
                base_noop,
                step_index=step_index,
                config=config,
            )
        field_loss = motion.charbonnier_distance(
            student, teacher, scale=CHARBONNIER_SCALE
        )
        replay_loss = motion.charbonnier_distance(
            adapted_action, base_action, scale=CHARBONNIER_SCALE
        )
    except (pgt.PriorGuidedTangentError, motion.MotionContractError) as error:
        raise _translate(error) from error
    return EndpointFields(
        base_negative=negative_clean,
        base_noop=base_noop,
        base_action=base_action,
        adapted_action=adapted_action,
        student_executed=student,
        teacher_executed=teacher,
        field_loss=field_loss,
        replay_loss=replay_loss,
    )


def _prior_tangent_bridge_losses(
    *,
    renderer: Any,
    adapter_controller: Any,
    endpoints: Mapping[
        str,
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
        ],
    ],
    route: motion.Route,
    step_index: int,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    """Field + 0.05 bridge + 0.10 late-prior replay objective."""

    import torch

    if route.tier != "motion_only" or route.full_target_weight != 0.0:
        raise PriorTangentTrainingError("v5 accepts only strict motion_only routes")
    if set(endpoints) != {"source", "target"}:
        raise PriorTangentTrainingError("v5 requires source and target bridge endpoints")
    config = pgt.TangentTrustRegionConfig()
    fields: dict[str, EndpointFields] = {}
    sigmas = []
    for name in ("source", "target"):
        negative, noop, action, auxiliary = endpoints[name]
        fields[name] = _endpoint_fields(
            renderer=renderer,
            adapter_controller=adapter_controller,
            negative_batch=negative,
            noop_batch=noop,
            action_batch=action,
            auxiliary=auxiliary,
            step_index=step_index,
            config=config,
        )
        sigmas.append(auxiliary["sigma"])
    if not torch.equal(sigmas[0], sigmas[1]):
        raise PriorTangentTrainingError("bridge endpoint sigmas differ")
    field_loss = 0.5 * (fields["source"].field_loss + fields["target"].field_loss)
    bridge_loss = motion.charbonnier_distance(
        fields["source"].student_executed,
        fields["target"].student_executed,
        scale=CHARBONNIER_SCALE,
    )
    replay_loss = 0.5 * (
        fields["source"].replay_loss + fields["target"].replay_loss
    )
    gamma = pgt.correction_gamma(step_index)
    late_replay_gate = 1.0 - gamma
    clean_field_weight = motion.clean_field_inverse_sigma_weight(
        sigmas[0], weight_floor=args.inverse_sigma_weight_floor
    ).mean()
    total = clean_field_weight * (
        FIELD_LOSS_WEIGHT * field_loss
        + BRIDGE_LOSS_WEIGHT * bridge_loss
        + LATE_REPLAY_LOSS_WEIGHT * late_replay_gate * replay_loss
    )

    def rms(value: Any) -> Any:
        return value.float().square().mean().sqrt()

    prior_source = pgt.frozen_prior(
        fields["source"].base_action, fields["source"].base_noop
    )
    correction_source = pgt.adapter_correction(
        fields["source"].adapted_action, fields["source"].base_action
    )
    return total, {
        "field": field_loss,
        "bridge": bridge_loss,
        "late_replay": replay_loss,
        "late_replay_gate": torch.tensor(
            late_replay_gate, dtype=torch.float32, device=field_loss.device
        ),
        "gamma": torch.tensor(gamma, dtype=torch.float32, device=field_loss.device),
        "clean_field_weight": clean_field_weight,
        "sigma": sigmas[0].float().mean(),
        "frozen_prior_rms": rms(prior_source),
        "raw_adapter_correction_rms": rms(correction_source),
        "source_student_field_rms": rms(fields["source"].student_executed),
        "target_student_field_rms": rms(fields["target"].student_executed),
        "same_state_exact": torch.ones((), device=field_loss.device),
        "base_forwards_per_endpoint": torch.tensor(
            3.0, dtype=torch.float32, device=field_loss.device
        ),
        "adapted_forwards_per_endpoint": torch.tensor(
            1.0, dtype=torch.float32, device=field_loss.device
        ),
    }


def _prepare_prior_bridge_batches(
    *,
    raw_row: Mapping[str, Any],
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    noop_instruction: str,
    negative_prompt: str,
    minimum_training_sigma: float,
    process_renderer_sample: Any,
    selected_stratum: sigma_strata.SigmaStratum,
) -> dict[
    str,
    tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
]:
    """Reuse v4 bridge construction and graft one negative text branch."""

    base_endpoints = v4._prepare_bridge_batches(
        raw_row=raw_row,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        scheduler=scheduler,
        noop_instruction=noop_instruction,
        minimum_training_sigma=minimum_training_sigma,
        process_renderer_sample=process_renderer_sample,
        selected_stratum=selected_stratum,
    )
    negative_text_fields = _official_negative_text_fields(
        tokenizer, negative_prompt
    )
    endpoints = {}
    for name in ("source", "target"):
        action_batch, noop_batch, auxiliary = base_endpoints[name]
        negative_batch = dict(action_batch)
        for key in TEXT_FIELDS:
            negative_batch[key] = negative_text_fields[key]
        legacy.validate_collated_supervision(negative_batch)
        _assert_same_endpoint_state(negative_batch, noop_batch, action_batch)
        endpoints[name] = (negative_batch, noop_batch, action_batch, auxiliary)
    return endpoints


def _strict_router(
    args: argparse.Namespace,
    router: motion.ReviewRouter,
    eligible_routes: Sequence[tuple[int, motion.Route]],
    dataset: Any,
) -> None:
    receipt = router.receipt()
    if (
        receipt.get("path") is None
        or receipt.get("default_tier") != "reject"
        or receipt.get("file_sha256") != args.expected_routing_jsonl_sha256
        or receipt.get("explicit_route_counts")
        != {"full_pair": 0, "motion_only": 359, "reject": 285}
        or len(eligible_routes) != 359
        or len(dataset) != 644
        or any(
            route.tier != "motion_only" or route.full_target_weight != 0.0
            for _, route in eligible_routes
        )
    ):
        raise PriorTangentTrainingError(
            "v5 requires the hash-bound strict-359 motion-only / reject-285 route"
        )


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
    gamma = list(pgt.correction_gamma_schedule())
    value = {
        "method": METHOD_NAME,
        "schema_version": RECEIPT_SCHEMA,
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
        "eligible_route_count": len(eligible_routes),
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
        "frames": 81,
        "latent_phases": 21,
        "learning_rate": LEARNING_RATE,
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_scope": "cross_q",
        "target_modules": list(target_modules),
        "target_modules_sha256": legacy.object_sha256(list(target_modules)),
        "bridge_fractions": list(BRIDGE_FRACTIONS),
        "bridge_query": "source_and_executable_target_same_epsilon_sigma_timestep",
        "branches_per_endpoint": [
            "base_negative_adapter_off_no_grad",
            "base_noop_adapter_off_no_grad",
            "base_action_adapter_off_no_grad",
            "adapted_action_adapter_on_grad",
        ],
        "forwards_per_endpoint": 4,
        "forwards_per_optimizer_step": 8,
        "base_apg": {
            "guidance_scale": APG_GUIDANCE_SCALE,
            "eta": APG_ETA,
            "norm_threshold": APG_NORM_THRESHOLD,
            "momentum": APG_MOMENTUM,
            "negative_prompt_sha256": hashlib.sha256(
                args.negative_prompt.encode("utf-8")
            ).hexdigest(),
            "negative_tokenization": (
                "official_renderer_unconditional_verbatim"
            ),
            "clean_reconstruction": (
                "fp32_noisy_minus_cpu_fp32_0d_sigma_times_native_bf16_velocity"
            ),
        },
        "noop_instruction_sha256": hashlib.sha256(
            args.noop_instruction.encode("utf-8")
        ).hexdigest(),
        "raw_prior": "raw_frozen_prior=base_guided_action-base_guided_noop",
        "prior": "causal_frozen_prior=Q0(raw_frozen_prior)",
        "adapter_correction": "Q0(adapted_guided_action-base_guided_action)",
        "teacher_correction": "Q0((target-source)-causal_frozen_prior)",
        "phase_zero_contract": "executed_motion_exact_zero_source_exactly_preserved",
        "trust_region": {
            "kappa_parallel": pgt.DEFAULT_KAPPA_PARALLEL,
            "kappa_perp": pgt.DEFAULT_KAPPA_PERP,
            "epsilon": pgt.DEFAULT_EPSILON,
            "phase_dim": 1,
        },
        "gamma_schedule": gamma,
        "gamma_schedule_sha256": legacy.object_sha256(gamma),
        "gamma_contract": (
            "0-23 full; 24-34 inclusive cosine taper (gamma24=1,gamma34=0); "
            "35-39 exact causal frozen prior"
        ),
        "sigma_schedule": "exact_40_step_flow_shift_5_cycle",
        "sigma_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        "sigma_selector": "absolute_global_step_mod_40",
        "minimum_training_sigma": 0.1,
        "inverse_sigma_weight_floor": float(args.inverse_sigma_weight_floor),
        "loss": {
            "field": FIELD_LOSS_WEIGHT,
            "bridge": BRIDGE_LOSS_WEIGHT,
            "late_replay": LATE_REPLAY_LOSS_WEIGHT,
            "late_replay_gate": "1-gamma",
            "robust_distance": f"charbonnier_scale_{CHARBONNIER_SCALE}",
            "outer_multiplier": "1/max(sigma,final_positive_unipc_sigma)",
        },
        "inference_conditions": ["source_video", "action_instruction"],
        "training_only_conditions": ["target_video"],
        "forbidden_inference_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
    }
    if len(target_modules) != 30:
        raise PriorTangentTrainingError(
            f"v5 cross_q must resolve exactly 30 modules, got {len(target_modules)}"
        )
    return {"value": value, "digest": legacy.object_sha256(value)}


def _supervision_receipt(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "source_target_bridge": True,
        "four_branch_endpoint": True,
        "base_branches_adapter_disabled": True,
        "base_branches_no_grad": True,
        "adapted_action_only_trainable_forward": True,
        "causal_frozen_prior": "Q0(base_action-base_noop)",
        "executed_motion_phase_zero": "exact_zero",
        "official_apg_momentum": APG_MOMENTUM,
        "official_apg_guidance_scale": APG_GUIDANCE_SCALE,
        "official_apg_eta": APG_ETA,
        "official_apg_norm_threshold": APG_NORM_THRESHOLD,
        "negative_prompt_sha256": NEGATIVE_PROMPT_SHA256,
        "negative_tokenization": "official_renderer_unconditional_verbatim",
        "field_loss_weight": FIELD_LOSS_WEIGHT,
        "bridge_loss_weight": BRIDGE_LOSS_WEIGHT,
        "late_replay_loss_weight": LATE_REPLAY_LOSS_WEIGHT,
        "late_replay_gate": "1-gamma",
        "target_used_as_model_condition": False,
        "target_used_as_offline_teacher": True,
        "inference_conditions": ["source_video", "action_instruction"],
        "external_mask_track_flow_pose_trajectory": False,
        "post_video_acceptance": "pending",
        "production_claim_forbidden": True,
    }


def _optimizer_payload(
    *, optimizer: Any, global_step: int, immutable: Mapping[str, Any], parameter_names: Sequence[str]
) -> dict[str, Any]:
    return {
        "schema_version": OPTIMIZER_SCHEMA,
        "global_step": int(global_step),
        "optimizer": optimizer.state_dict(),
        "immutable_contract": dict(immutable),
        "parameter_names": list(parameter_names),
    }


def _validate_resume_receipt(
    prior: Mapping[str, Any], *, immutable: Mapping[str, Any]
) -> int:
    if prior.get("schema_version") != RECEIPT_SCHEMA:
        raise PriorTangentTrainingError("resume receipt schema differs from v5")
    _validate_digest_object(prior, label="resume")
    if prior.get("method") != METHOD_NAME or prior.get("immutable_contract") != immutable:
        raise PriorTangentTrainingError("resume immutable v5 contract differs")
    step = prior.get("global_step")
    if type(step) is not int or step < 0:
        raise PriorTangentTrainingError("resume global_step is invalid")
    return step


def _initialization_target_modules(
    receipt_path: Optional[Path], *, available_modules: Sequence[str]
) -> list[str]:
    if receipt_path is None:
        raise PriorTangentTrainingError(
            "initialization adapter requires a hash-bound receipt"
        )
    receipt = _read_json(receipt_path, label="initialization receipt")
    _validate_digest_object(receipt, label="initialization")
    schema = receipt.get("schema_version")
    if schema in (RECEIPT_SCHEMA, v4.RECEIPT_SCHEMA):
        adapter = receipt.get("adapter")
        targets = adapter.get("target_modules") if isinstance(adapter, Mapping) else None
        if not isinstance(targets, list) or not all(isinstance(name, str) for name in targets):
            raise PriorTangentTrainingError(
                "initialization receipt lacks exact adapter targets"
            )
        return list(targets)
    if schema == legacy.RECEIPT_SCHEMA:
        if (
            receipt.get("target_module_count") != legacy.EXPECTED_LORA_TARGET_MODULES
            or receipt.get("target_modules_sha256")
            != legacy.object_sha256(list(available_modules))
        ):
            raise PriorTangentTrainingError("legacy initialization scope differs")
        return list(available_modules)
    raise PriorTangentTrainingError("unsupported initialization receipt schema")


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
    names = v4._optimizer_parameter_names(named_trainable)
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
        "inference_sigma_strata": sigma_strata.build_sigma_strata_receipt(
            completed_optimizer_steps=global_step
        ),
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": "cross_q",
            "target_module_count": len(target_modules),
            "target_modules": list(target_modules),
            "target_modules_sha256": legacy.object_sha256(list(target_modules)),
            "trainable_parameter_count": sum(
                int(parameter.numel()) for _, parameter in named_trainable
            ),
            "parameter_names_sha256": legacy.object_sha256(names),
            "initialization_digest": initialization_digest,
            "checkpoint_parameter_digest": v4._checkpoint_parameter_digest(
                named_trainable
            ),
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": float(args.weight_decay),
            "max_gradient_norm": float(args.max_grad_norm),
            "parameter_names": names,
            "checkpoint_state_digest": v4._stable_recursive_digest(
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
            raise PriorTangentTrainingError(f"refusing to overwrite checkpoint: {final}")
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / f".{final.name}.tmp-{os.getpid()}"
        if temporary.exists():
            raise PriorTangentTrainingError(
                f"stale temporary checkpoint exists: {temporary}"
            )
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = legacy.validate_checkpoint(args.checkpoint)
    except legacy.TrainingContractError as error:
        raise _translate(error) from error
    if transformer_config["num_attention_heads"] % 4:
        raise PriorTangentTrainingError(
            "1.3B attention heads must be divisible by Ulysses=4"
        )
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, process_renderer_sample

    if DEFAULT_NEG_PROMPT != DEFAULT_NEGATIVE_PROMPT:
        raise PriorTangentTrainingError(
            "runtime Bernini unconditional prompt differs from the v5 contract"
        )

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
    eligible_routes = v4._build_eligible_routes(dataset, router)
    _strict_router(args, router, eligible_routes, dataset)

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
        target_modules = motion.select_lora_scope(available_modules, "cross_q")
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

    resume_paths = None
    prior_receipt: Optional[dict[str, Any]] = None
    initialized_from: Optional[dict[str, Any]] = None
    if args.resume:
        try:
            resume_paths = v4._resolve_adapter(args.resume, require_training_state=True)
        except v4.DeltaTrainingError as error:
            raise _translate(error) from error
        assert resume_paths.receipt is not None
        prior_receipt = _read_json(resume_paths.receipt, label="resume receipt")
        prior_step = _validate_resume_receipt(prior_receipt, immutable=immutable)
        try:
            model = v4._load_peft_adapter(
                base_model=base_model,
                adapter=resume_paths.adapter,
                target_modules=target_modules,
                trainable=True,
            )
        except v4.DeltaTrainingError as error:
            raise _translate(error) from error
    elif args.init_adapter_checkpoint:
        try:
            init_paths = v4._resolve_adapter(
                args.init_adapter_checkpoint, require_training_state=False
            )
        except v4.DeltaTrainingError as error:
            raise _translate(error) from error
        initialization_targets = _initialization_target_modules(
            init_paths.receipt, available_modules=available_modules
        )
        if initialization_targets != target_modules:
            raise PriorTangentTrainingError(
                "initialization adapter scope differs from v5 cross_q"
            )
        try:
            model = v4._load_peft_adapter(
                base_model=base_model,
                adapter=init_paths.adapter,
                target_modules=target_modules,
                trainable=True,
            )
        except v4.DeltaTrainingError as error:
            raise _translate(error) from error
        initialized_from = {
            "path": str(init_paths.root),
            "adapter_model_sha256": legacy.file_sha256(
                init_paths.adapter / "adapter_model.safetensors"
            ),
            "receipt_sha256": (
                legacy.file_sha256(init_paths.receipt)
                if init_paths.receipt is not None
                else None
            ),
        }
        prior_step = 0
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=target_modules,
            ),
        )
        prior_step = 0

    model.to(device)
    model.eval()
    renderer = model.get_base_model()
    renderer.t5_text_encoder.eval()
    named_trainable = legacy.trainable_lora_parameters(model)
    initialization_digest = legacy.synchronize_trainable_parameters(
        named_trainable, source_rank=0
    )
    if prior_receipt is not None:
        try:
            v4._validate_loaded_parameter_digest(prior_receipt, named_trainable)
        except v4.DeltaTrainingError as error:
            raise _translate(error) from error
    parameter_names = v4._optimizer_parameter_names(named_trainable)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_trainable],
        lr=LEARNING_RATE,
        weight_decay=args.weight_decay,
    )
    global_step = prior_step
    if resume_paths is not None:
        assert resume_paths.optimizer is not None and prior_receipt is not None
        try:
            state = torch.load(
                resume_paths.optimizer, map_location="cpu", weights_only=False
            )
        except TypeError:
            state = torch.load(resume_paths.optimizer, map_location="cpu")
        if not isinstance(state, Mapping):
            raise PriorTangentTrainingError("resume optimizer payload is not a mapping")
        optimizer_receipt = prior_receipt.get("optimizer")
        expected_digest = (
            optimizer_receipt.get("checkpoint_state_digest")
            if isinstance(optimizer_receipt, Mapping)
            else None
        )
        if expected_digest != v4._stable_recursive_digest(state):
            raise PriorTangentTrainingError("resume optimizer digest differs")
        if (
            state.get("schema_version") != OPTIMIZER_SCHEMA
            or state.get("immutable_contract") != immutable
            or state.get("parameter_names") != parameter_names
            or int(state.get("global_step", -1)) != global_step
        ):
            raise PriorTangentTrainingError("resume optimizer contract differs")
        optimizer.load_state_dict(state["optimizer"])
        v4._optimizer_to(optimizer, device)
        if any(
            float(group["lr"]) != LEARNING_RATE
            or float(group["weight_decay"]) != float(args.weight_decay)
            for group in optimizer.param_groups
        ):
            raise PriorTangentTrainingError(
                "restored optimizer hyperparameters differ"
            )
    if global_step > args.max_steps:
        raise PriorTangentTrainingError("resume step exceeds requested max_steps")

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
        row_index, raw_row, route = v4._next_routed_row(
            dataset, eligible_routes, ordinal=global_step
        )
        identity = legacy.dataset_identity(raw_row, row_index)
        legacy.assert_identical_row(identity)
        current_seed = legacy.step_seed(args.seed, global_step, row_index)
        legacy.seed_same_sample(current_seed)
        selected_stratum = sigma_strata.select_sigma_stratum(global_step)
        try:
            endpoint_batches = _prepare_prior_bridge_batches(
                raw_row=raw_row,
                tokenizer=tokenizer,
                rope=rope,
                vae_mean=vae_mean,
                vae_std=vae_std,
                z_dim=z_dim,
                scheduler=scheduler,
                noop_instruction=args.noop_instruction,
                negative_prompt=args.negative_prompt,
                minimum_training_sigma=args.minimum_training_sigma,
                process_renderer_sample=process_renderer_sample,
                selected_stratum=selected_stratum,
            )
        except (
            legacy.TrainingContractError,
            motion.MotionContractError,
            sigma_strata.InferenceSigmaStrataError,
            v4.DeltaTrainingError,
        ) as error:
            raise _translate(error) from error
        moved_endpoints = {}
        for endpoint_name, (negative, noop, action, auxiliary) in endpoint_batches.items():
            moved_endpoints[endpoint_name] = (
                legacy._move_batch(negative, device),
                legacy._move_batch(noop, device),
                legacy._move_batch(action, device),
                v4._move_auxiliary_to_device(
                    auxiliary,
                    device=device,
                    branch_state_mode="source_target_bridge_clean_field",
                ),
            )

        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            loss, components = _prior_tangent_bridge_losses(
                renderer=renderer,
                adapter_controller=model,
                endpoints=moved_endpoints,
                route=route,
                step_index=selected_stratum.schedule_index,
                args=args,
            )
        finite = bool(torch.isfinite(loss.detach()).item()) and all(
            bool(torch.isfinite(value.detach()).item()) for value in components.values()
        )
        if not legacy._distributed_boolean(finite, op="all"):
            raise PriorTangentTrainingError(
                f"non-finite v5 loss at optimizer step {global_step + 1}"
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
            "sigma_schedule_index": float(selected_stratum.schedule_index),
            "sigma_timestep": float(selected_stratum.timestep),
        }
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
            optimizer_payload = _optimizer_payload(
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
        optimizer_payload = _optimizer_payload(
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

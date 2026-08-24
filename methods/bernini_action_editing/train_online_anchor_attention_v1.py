#!/usr/bin/env python3
"""Train all-attention LoRA with an online pure-T2V action anchor.

The supervised target is always a complete, source-owned action video.  A
cross-appearance self-generated video is queried online by the frozen base
model and contributes only a detached cross-attention contrast to the target
suffix.  Anchor RGB/latent values are never used as the FM target.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import anchor_cross_attention_transport as cross
import anchor_qk_transport as qk
import train_lora as legacy
import train_same_video_dense_flow_adapter_v1 as pairs
import train_self_generated_action_fullfield_v4 as v4
import train_self_generated_action_quotient_v1 as data


METHOD = "bernini-online-anchor-attention-training-v1"
RECEIPT_SCHEMA = "bernini-online-anchor-attention-training-receipt-v2"
QK_ONLY_RECEIPT_SCHEMA = "bernini-online-anchor-attention-training-receipt-v3"
PROFILES = ("action_noop", "dynamic_static", "hybrid", "no_anchor")
TRAINING_OBJECTIVES = (
    "target_fm",
    "paired_delta_fm",
    "real_source_teacher_delta",
    "real_source_routed_teacher_delta",
    "real_source_target_owned_routed_teacher_delta_v14r2",
)
REAL_SOURCE_OBJECTIVES = (
    "real_source_teacher_delta",
    "real_source_routed_teacher_delta",
    "real_source_target_owned_routed_teacher_delta_v14r2",
)
ROUTED_TEACHER_OBJECTIVES = (
    "real_source_routed_teacher_delta",
    "real_source_target_owned_routed_teacher_delta_v14r2",
)
TRAINING_INTERFACES = ("mv2v_full_source", "first_phase_caption_i2v")
TEACHER_DELTA_MODES = ("raw", "phase0_relative")
ROUTED_TEACHER_MODES = (
    "same_action_route_only",
    "cross_caption_two_sided",
)
REPLAY_COMBINE_MODES = (
    "fixed_0025",
    "first_order_safe",
    "action_only",
    "norm_balanced_005",
    "norm_balanced_025",
    "source_safe_cap025",
    "source_halfspace_001",
    "action_priority_pcgrad_010",
)
REAL_SOURCE_SCHEMA = "bernini-complex8-real-source-latents-v2"
SHA256 = re.compile(r"[0-9a-f]{64}")
T2V_TRAINING_SOURCE_NAME = "t2v$action_editing_81f"
ROUTE_OPERATORS = (
    "cross_sparse",
    "self_temporal_kernel",
    "self_target_gated_kernel25",
    "self_correspondence_kernel25",
    "self_target_owned_temporal_kernel_v14r2",
    "self_target_owned_activity_kernel10_v14r2",
    "self_target_owned_activity_kernel25_v14r2",
)
QK_ONLY_ROUTE_OPERATORS = (
    "self_target_owned_temporal_kernel_v14r2",
    "self_target_owned_activity_kernel10_v14r2",
    "self_target_owned_activity_kernel25_v14r2",
)
LORA_RANK = 256
LORA_ALPHA = 256
LORA_PARAMETERS = 188_743_680
LORA_TARGET_MODULE_COUNT = 240
LORA_TRAINABLE_TENSOR_COUNT = 480
LORA_SCOPE = "all_30_blocks_attn1_attn2_qkvo"
COMPONENT_GRADIENT_EPSILON = 1.0e-12
SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED = False
SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE = "route_on_only_legacy"
SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED = False
SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT = 0.0
SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE = "disabled"
BLOCK_COUNT = 30
CHECKPOINT_BLOCKS = tuple(range(0, BLOCK_COUNT, 4))
# Route through every block that is not activation-checkpointed.  This keeps
# one-use anchor cache semantics exact while all 30 blocks remain trainable.
ROUTE_BLOCKS = tuple(index for index in range(BLOCK_COUNT) if index not in CHECKPOINT_BLOCKS)
SAVE_STEPS = (1, 4, 8, 16, 32, 64)

# These are same-video temporal counterfactuals, not independently generated
# videos.  Phase zero stays source-authoritative while every later latent phase
# is either reversed or placed in a fixed non-chronological permutation.
REVERSE_PHASE_INDICES = (0, *tuple(range(20, 0, -1)))
SHUFFLE_PHASE_INDICES = (
    0, 17, 18, 1, 6, 16, 4, 12, 11, 7, 13, 19, 2, 15, 8, 3, 9, 20, 5, 10, 14
)
SOURCE_VARIANTS = ("noop", "incomplete", "reverse", "shuffle")
IDENTITY_REPLAY_PROMPT = (
    "Keep the input video exactly unchanged. Preserve every frame, subject, "
    "object instance, appearance, clothing or fur color, motion, timing, "
    "contact, scene, lighting, framing and camera exactly as shown."
)


NOOP_BY_EVENT = {
    "pour-liquid-into-cup": "The person and both vessels remain in their initial state. No vessel is lifted or tilted, no liquid is poured, and the cup is not filled.",
    "reach-grasp-lift-stone": "The child remains in the initial pose and does not crouch, reach, contact, grasp, lift, or hold a stone.",
    "twist-pull-mushroom": "The hand remains near the rooted mushroom and does not grasp, twist, detach, lift, or leave an empty hole.",
    "release-harvest-into-basket": "The hand keeps holding the harvested cluster and does not move it over the basket, release it, or let it settle inside.",
    "close-door-then-drawer": "The cabinet remains in its initial state. The door and drawer do not move or close and no ordered interaction occurs.",
    "jet-ski-turn-with-wake": "The rider and jet ski keep the initial heading and do not turn, lean, carve a curved path, or create a curved wake.",
    "tap-plant-and-rebound": "The hand and plant remain in their initial state. The stem is not released or tapped and no bending or rebound occurs.",
    "players-contact-then-separate": "The two players continue in their initial contact and do not push off, separate into a gap, or take distinct paths.",
}


class OnlineAnchorTrainingError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise OnlineAnchorTrainingError(message)


def select_lora_target_names(renderer: Any) -> tuple[str, ...]:
    """Return the audited LoRA target closure for this training variant.

    Wrappers may temporarily replace this function together with the four
    ``LORA_*`` closure constants above.  The default remains byte-for-byte the
    original all-attention target selection used by v16r5.
    """

    return tuple(legacy.select_attention_projection_names(renderer))


def requires_sequential_source_side_backward(
    *, paired_action_loss: Any, same_action_route_only: bool
) -> bool:
    """Return whether the detached delta denominator needs its own backward.

    The default preserves the historical v16r5 behavior.  A wrapper may
    enable the same-action route-off branch, in which case the second forward
    supplies the missing ``-J_off`` term without retaining both 30-block
    graphs simultaneously.
    """

    return paired_action_loss is not None and (
        not same_action_route_only or SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED
    )


def sequential_source_side_record(
    *,
    same_action_route_only: bool,
    action_record: Mapping[str, Any],
    paired_source: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select the exact route-off state for the sequential denominator graph."""

    if same_action_route_only:
        if not SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED:
            fail("same-action route-off gradient was not enabled")
        return action_record
    return paired_source


def requires_same_action_route_off_absolute_anchor(
    *, same_action_route_only: bool
) -> bool:
    """Return whether a separate route-off graph supplies an absolute anchor."""

    enabled = bool(SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED)
    if enabled and (
        not same_action_route_only
        or not 0.0 < float(SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT) <= 1.0
    ):
        fail("same-action route-off absolute-anchor contract differs")
    return enabled


def same_action_route_off_absolute_anchor_loss(
    *, student_route_off_prediction: Any, frozen_route_off_teacher: Any
) -> Any:
    """Absolute same-state FM spring used only by the independent D variant."""

    return finite_mse(
        student_route_off_prediction,
        frozen_route_off_teacher.detach(),
        name="same-action route-off absolute common-mode FM anchor",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument(
        "--authoring",
        default="",
        help="Complex8 authoring JSON required by first_phase_caption_i2v",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True, choices=PROFILES)
    parser.add_argument(
        "--route-operator",
        choices=ROUTE_OPERATORS,
        default="cross_sparse",
        help=(
            "cross_sparse replays the legacy post-attn2 residual; "
            "self_temporal_kernel transfers only an action/no-op T-by-T "
            "self-attention kernel onto the target's own value stream; "
            "self_correspondence_kernel25 transfers a matched anchor-local "
            "temporal graph onto selected target-owned value trajectories; "
            "self_target_owned_*_v14r2 routes cache donor Q/K only"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--micro-records", type=int, default=2)
    parser.add_argument(
        "--source-variant",
        choices=("noop", "mixed", "counterfactual4", "not_applicable"),
        default="mixed",
        help=(
            "mixed uses noop/incomplete; counterfactual4 cycles the same "
            "target through noop, incomplete, reverse and phase-shuffle sources"
        ),
    )
    parser.add_argument("--route-strength", type=float, default=0.25)
    parser.add_argument(
        "--teacher-route-strength",
        type=float,
        default=1.0,
        help=(
            "Frozen target-coordinate route strength used only by "
            "real_source_routed_teacher_delta; the student keeps "
            "--route-strength"
        ),
    )
    parser.add_argument(
        "--training-objective",
        choices=TRAINING_OBJECTIVES,
        default="target_fm",
        help=(
            "target_fm is the legacy full-target flow-matching loss; "
            "paired_delta_fm trains the complete spatiotemporal target-minus-"
            "source velocity field using matched noise and timestep; "
            "real_source_routed_teacher_delta distils a stronger frozen "
            "target-coordinate anchor route into the trainable weaker route; "
            "real_source_target_owned_routed_teacher_delta_v14r2 enforces the v14r2 "
            "content-free donor ABI and component-gradient gates"
        ),
    )
    parser.add_argument(
        "--real-source-manifest",
        default="",
        help="Eight-source latent manifest required by real-source objectives",
    )
    parser.add_argument(
        "--real-source-manifest-sha256",
        default="",
        help="Exact manifest digest required by real-source objectives",
    )
    parser.add_argument(
        "--teacher-delta-mode",
        choices=TEACHER_DELTA_MODES,
        default="phase0_relative",
        help=(
            "raw matches the complete action-minus-noop teacher velocity; "
            "phase0_relative subtracts each spatial site's phase-0 delta from "
            "all 21 phases without reducing tensor dimensionality"
        ),
    )
    parser.add_argument(
        "--routed-teacher-mode",
        choices=ROUTED_TEACHER_MODES,
        default="cross_caption_two_sided",
        help=(
            "same_action_route_only distils routed-minus-route-off under the "
            "same target caption and noisy source state, with gradients only "
            "through the routed student; cross_caption_two_sided is the v14 "
            "action-caption-minus-source-caption control"
        ),
    )
    parser.add_argument(
        "--training-interface",
        choices=TRAINING_INTERFACES,
        default="mv2v_full_source",
        help=(
            "mv2v_full_source retains the historical instruction/full-video "
            "condition; first_phase_caption_i2v matches the DynaEdit decoder "
            "with a repeated source-owned phase zero and full T2V captions"
        ),
    )
    parser.add_argument(
        "--paired-target-fm-weight",
        type=float,
        default=0.25,
        help=(
            "Auxiliary ordinary target-FM weight used only by paired_delta_fm; "
            "zero gives a pure two-sided velocity-difference objective"
        ),
    )
    parser.add_argument(
        "--source-reconstruction-weight",
        type=float,
        default=0.25,
        help=(
            "Weight of a source-caption source-trajectory FM replay for real-source "
            "objectives; this is not an identity target"
        ),
    )
    parser.add_argument(
        "--replay-combine-mode",
        choices=REPLAY_COMBINE_MODES,
        default="fixed_0025",
        help=(
            "v14r2 combines separately averaged action and raw source-FM "
            "gradients.  The legacy modes use either fixed 0.025 or a "
            "first-order-safe raw replay scale; the v14r3 modes use action "
            "only, a 0.05/0.25 norm-balanced replay, a source-safe replay "
            "capped at q=0.25, a source half-space correction with 0.01 "
            "margin, or a 0.10 action-priority PCGrad replay"
        ),
    )
    parser.add_argument(
        "--gradient-diagnostic-only",
        action="store_true",
        help=(
            "Compute and globally reduce the v14r2 action/replay component "
            "gradients, print their interaction, then fail before gradient "
            "mutation, clipping, optimizer.step, checkpointing, or completion"
        ),
    )
    parser.add_argument(
        "--source-reconstruction-prompt",
        choices=("action", "noop", "identity"),
        default="action",
        help=(
            "Prompt used for source-to-source preservation replay.  noop avoids "
            "teaching the model to ignore an action instruction; identity is "
            "the content-neutral no-edit prompt valid for every source motion"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=2026081921)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.micro_records != 2:
        fail("max-steps must be positive and micro-records must be exactly two")
    if not 0.0 < float(args.route_strength) <= 1.0:
        fail("route-strength must be in (0,1]")
    if not 0.0 < float(args.teacher_route_strength) <= 1.0:
        fail("teacher-route-strength must be in (0,1]")
    if not 0.0 <= float(args.paired_target_fm_weight) <= 1.0:
        fail("paired-target-fm-weight must be in [0,1]")
    if not 0.0 <= float(args.source_reconstruction_weight) <= 1.0:
        fail("source-reconstruction-weight must be in [0,1]")
    if not 0.0 < float(args.learning_rate) <= 1.0e-3:
        fail("learning-rate is outside the audited range")
    if not 0.0 < float(args.max_grad_norm) <= 100.0:
        fail("max-grad-norm is outside the audited range")
    if args.training_interface == "first_phase_caption_i2v":
        if args.training_objective not in (
            "paired_delta_fm",
            "real_source_teacher_delta",
            "real_source_routed_teacher_delta",
            "real_source_target_owned_routed_teacher_delta_v14r2",
        ):
            fail("first_phase_caption_i2v requires a velocity-delta objective")
        if args.source_variant == "counterfactual4":
            fail("caption-I2V training rejects uncaptionable reverse/shuffle sources")
        if not args.authoring:
            fail("first_phase_caption_i2v requires --authoring")
    if args.training_objective in REAL_SOURCE_OBJECTIVES:
        if args.training_interface != "first_phase_caption_i2v":
            fail("real-source teacher objectives require caption-I2V")
        if not args.real_source_manifest:
            fail("real-source teacher objectives require their source manifest")
        if SHA256.fullmatch(args.real_source_manifest_sha256) is None:
            fail("real-source manifest SHA-256 differs")
        if float(args.paired_target_fm_weight) != 0.0:
            fail("real-source teacher objectives forbid synthetic target FM")
        if args.profile == "no_anchor":
            fail("real-source teacher objectives require an online T2V teacher")
    if args.training_objective in ROUTED_TEACHER_OBJECTIVES and args.profile != "action_noop":
        fail("routed teacher delta requires an action/noop anchor route")
    if args.training_objective in ROUTED_TEACHER_OBJECTIVES:
        if args.route_operator == "cross_sparse":
            fail("routed teacher delta forbids donor-valued cross-attention replay")
        if args.teacher_delta_mode != "raw":
            fail("routed teacher delta requires the complete raw velocity field")
        if float(args.teacher_route_strength) <= float(args.route_strength):
            fail("teacher route strength must be strictly greater than student route strength")
    if (
        args.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2"
        and args.route_operator not in QK_ONLY_ROUTE_OPERATORS
    ):
        fail("v14r2 routed teacher requires an explicit QK-only route operator")
    if args.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2":
        if float(args.source_reconstruction_weight) != 0.025:
            fail("v14r2 requires the pre-registered base replay scale 0.025")
        if args.source_variant != "not_applicable":
            fail("v14r2 real-source variant argument must be not_applicable")
    if args.gradient_diagnostic_only and (
        args.training_objective
        != "real_source_target_owned_routed_teacher_delta_v14r2"
        or args.max_steps != 2
    ):
        fail("gradient-diagnostic-only requires the v14r2 objective and max-steps=2")
    if (
        args.training_objective == "real_source_routed_teacher_delta"
        and args.routed_teacher_mode != "cross_caption_two_sided"
    ):
        fail("legacy routed-teacher objective is the cross-caption control only")
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        fail("output must be absolute")
    if output.exists() or output.is_symlink():
        fail("training output must be fresh")


def row_registry(rows: Sequence[Mapping[str, Any]]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    registry: dict[tuple[str, str], Mapping[str, Any]] = {}
    by_event: dict[str, set[str]] = {}
    for row in rows:
        event_id = row.get("event_id")
        variant_id = row.get("variant_id")
        if not isinstance(event_id, str) or event_id not in NOOP_BY_EVENT:
            fail("pair row event is outside Complex8")
        if not isinstance(variant_id, str):
            fail("pair row variant is absent")
        key = (event_id, variant_id)
        if key in registry:
            fail("event/variant registry is duplicated")
        registry[key] = row
        by_event.setdefault(event_id, set()).add(variant_id)
    if set(by_event) != set(NOOP_BY_EVENT) or any(
        variants != {"v0", "v1", "v2", "v3"} for variants in by_event.values()
    ):
        fail("Complex8 must contain exactly four variants per event")
    return registry


def load_caption_registry(path: Path) -> Mapping[tuple[str, str], Mapping[str, str]]:
    """Bind every training variant to the same complete captions used by I2V decode."""

    value = json.loads(path.read_text(encoding="utf-8"))
    events = value.get("events")
    if (
        value.get("schema_version")
        != "bernini-interaction-complex8-multianchor-authoring-v2"
        or not isinstance(events, list)
        or len(events) != 8
    ):
        fail("caption-I2V authoring closure differs")
    registry: dict[tuple[str, str], Mapping[str, str]] = {}
    for event in events:
        event_id = event.get("event_id")
        action = event.get("action")
        constraints = event.get("constraints")
        variants = event.get("variants")
        if (
            not isinstance(event_id, str)
            or event_id not in NOOP_BY_EVENT
            or not isinstance(action, str)
            or not action.strip()
            or not isinstance(constraints, str)
            or not constraints.strip()
            or not isinstance(variants, list)
            or len(variants) != 4
        ):
            fail("caption-I2V event authoring differs")
        for variant in variants:
            variant_id = variant.get("variant_id")
            setup = variant.get("setup")
            if (
                not isinstance(variant_id, str)
                or not isinstance(setup, str)
                or not setup.strip()
            ):
                fail("caption-I2V variant authoring differs")
            key = (event_id, variant_id)
            if key in registry:
                fail("caption-I2V event/variant is duplicated")
            registry[key] = {
                "target": f"{setup.strip()} {action.strip()} {constraints.strip()}",
                "noop": f"{setup.strip()} {NOOP_BY_EVENT[event_id]}",
                "incomplete": (
                    f"{setup.strip()} The requested action begins naturally but stops "
                    "halfway; the subjects and objects hold that incomplete mid-action "
                    f"state without reaching the requested terminal state. {constraints.strip()}"
                ),
            }
    if len(registry) != 32:
        fail("caption-I2V authoring must contain exactly 32 variants")
    return registry


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").float().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def load_real_source_registry(
    path: Path, expected_sha256: str
) -> Mapping[str, Mapping[str, Any]]:
    from safetensors.torch import load_file

    if file_sha256(path) != expected_sha256:
        fail("real-source manifest bytes differ")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if (
        value.get("schema_version") != REAL_SOURCE_SCHEMA
        or value.get("row_count") != 8
        or value.get("posterior_statistic") != "latent_dist.mode"
        or value.get("normalization")
        != "checkpoint_latents_mean_std_exactly_once"
        or value.get("normalization_count") != 1
        or value.get("normalization_application_count") != 1
        or value.get("bernini_private_vae_encode_used") is not False
        or value.get("self_generated_anchor_read_during_materialization") is not False
        or value.get("optimization_steps") != 0
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        fail("real-source latent manifest closure differs")
    registry: dict[str, Mapping[str, Any]] = {}
    for ordinal, row in enumerate(rows):
        event_id = row.get("event_id")
        latent_value = row.get("latent")
        if (
            row.get("ordinal") != ordinal
            or not isinstance(event_id, str)
            or event_id not in NOOP_BY_EVENT
            or event_id in registry
            or not isinstance(latent_value, str)
            or SHA256.fullmatch(str(row.get("latent_file_sha256"))) is None
            or SHA256.fullmatch(str(row.get("latent_tensor_sha256"))) is None
            or not isinstance(row.get("source_caption"), str)
            or not str(row["source_caption"]).strip()
            or not isinstance(row.get("target_caption"), str)
            or not str(row["target_caption"]).strip()
            or row.get("posterior_statistic") != "latent_dist.mode"
            or row.get("normalization_count") != 1
            or row.get("normalization_application_count") != 1
        ):
            fail("one real-source latent row differs")
        requested = Path(latent_value).expanduser()
        if not requested.is_absolute() or requested.is_symlink():
            fail("real-source latent must be an absolute non-symlink")
        latent_path = requested.resolve(strict=True)
        if latent_path != requested or file_sha256(latent_path) != row["latent_file_sha256"]:
            fail("real-source latent file bytes differ")
        tensors = load_file(str(latent_path), device="cpu")
        if tuple(tensors) != ("clean",):
            fail("real-source latent tensor key differs")
        clean = tensors["clean"].float().contiguous()
        if (
            list(map(int, clean.shape)) != row.get("latent_shape")
            or tuple(map(int, clean.shape[:3])) != (1, 16, 21)
            or tensor_sha256(clean) != row["latent_tensor_sha256"]
        ):
            fail("real-source latent tensor identity differs")
        registry[event_id] = {**row, "clean": clean}
    if set(registry) != set(NOOP_BY_EVENT):
        fail("real-source registry does not close Complex8")
    return registry


def repeated_phase_zero(clean: Any) -> Any:
    if clean.ndim != 5 or int(clean.shape[2]) != 21:
        fail("caption-I2V clean latent geometry differs")
    return clean[:, :, :1].expand_as(clean).clone().contiguous()


def build_i2v_paired_records(
    *,
    row: Mapping[str, Any],
    variant: str,
    captions: Mapping[tuple[str, str], Mapping[str, str]],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the exact paired caption/I2V fields consumed by DynaEdit."""

    if variant not in ("noop", "incomplete"):
        fail("caption-I2V source must be noop or incomplete")
    target, noop, incomplete, _, _ = pairs.load_row_tensors(row)
    source = source_clean_for_variant(
        target=target, noop=noop, incomplete=incomplete, variant=variant
    )
    condition = repeated_phase_zero(target)
    key = (str(row["event_id"]), str(row["variant_id"]))
    try:
        caption = captions[key]
    except KeyError as error:
        raise OnlineAnchorTrainingError("caption-I2V row is absent") from error
    condition_blob = _blob(condition, mean, std)
    action = transform(
        data.make_sample(
            instruction=caption["target"],
            source_blob=condition_blob,
            target_blob=_blob(target, mean, std),
        ),
        seed,
    )
    source_batch = transform(
        data.make_sample(
            instruction=caption[variant],
            source_blob=condition_blob,
            target_blob=_blob(source, mean, std),
        ),
        seed,
    )
    action_t = float(action["timesteps"].float().reshape(-1)[0].item())
    source_t = float(source_batch["timesteps"].float().reshape(-1)[0].item())
    if action_t != source_t:
        fail("caption-I2V pair did not share a timestep")
    shape = tuple(map(int, target.shape))
    return (
        {
            "batch": action,
            "shape": shape,
            "iid": str(row["iid"]),
            "variant": variant,
            "timestep": action_t,
        },
        {"batch": source_batch, "shape": shape},
    )


def build_real_source_paired_records(
    *,
    anchor_row: Mapping[str, Any],
    real_sources: Mapping[str, Mapping[str, Any]],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Query target/source captions on the exact same complete real source."""

    event_id = str(anchor_row["event_id"])
    try:
        source_row = real_sources[event_id]
    except KeyError as error:
        raise OnlineAnchorTrainingError("real-source event is absent") from error
    clean = source_row["clean"]
    condition = repeated_phase_zero(clean)
    condition_blob = _blob(condition, mean, std)
    target_blob = _blob(clean, mean, std)
    action = transform(
        data.make_sample(
            instruction=str(source_row["target_caption"]),
            source_blob=condition_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    source_batch = transform(
        data.make_sample(
            instruction=str(source_row["source_caption"]),
            source_blob=condition_blob,
            target_blob=target_blob,
        ),
        seed,
    )
    shape = tuple(map(int, clean.shape))
    source_batch, prebind_diagnostic = bind_real_source_caption_to_action_state(
        action,
        source_batch,
        spatial_shape=shape,
    )
    prebind_diagnostic = dict(prebind_diagnostic)
    prebind_diagnostic["transform_seed"] = int(seed)
    action_t = float(action["timesteps"].float().reshape(-1)[0].item())
    source_t = float(source_batch["timesteps"].float().reshape(-1)[0].item())
    if action_t != source_t:
        fail("real-source action/source pair did not share exact noisy state")
    require_same_real_source_noisy_state(
        action,
        source_batch,
        spatial_shape=shape,
    )
    return (
        {
            "batch": action,
            "shape": shape,
            "iid": str(source_row["source_iid"]),
            "variant": "complete_real_source",
            "timestep": action_t,
            "real_source_prebind_state_diagnostic": prebind_diagnostic,
        },
        {
            "batch": source_batch,
            "shape": shape,
            "iid": str(source_row["source_iid"]),
            "timestep": source_t,
        },
    )


REAL_SOURCE_EXACT_STATE_KEYS = (
    "input_vae_latents",
    "input_vae_rope",
    "vae_latents_mask",
    "vae_seqlen",
    "timesteps",
    "target_velocity",
    "target_lens",
)


def validate_real_source_packed_state(
    batch: Mapping[str, Any],
    *,
    spatial_shape: Sequence[int],
    label: str,
) -> dict[str, Any]:
    """Validate Bernini's packed I2V state against one 21-phase latent.

    ``process_renderer_sample`` has already patch-packed the VAE tensors here.
    Its ``input_vae_latents`` geometry is ``[tokens, C, pt, ph, pw]``; temporal
    phases are folded into ``tokens`` and therefore must never be read from
    ``shape[2]``.  The unpacked clean latent is the authority for the exact
    21-phase token count.
    """

    import torch

    if not isinstance(label, str) or not label:
        fail("real-source packed-state label is empty")
    try:
        clean_shape = tuple(map(int, spatial_shape))
    except (TypeError, ValueError) as error:
        raise OnlineAnchorTrainingError(
            "real-source unpacked spatial geometry is invalid"
        ) from error
    if len(clean_shape) != 5:
        fail("real-source unpacked spatial geometry must be rank five")
    clean_batch, channels, phases, height, width = clean_shape
    if (
        (clean_batch, channels, phases) != (1, 16, 21)
        or height <= 0
        or width <= 0
        or height % 2
        or width % 2
    ):
        fail("real-source unpacked latent is not one patchable 21-phase video")

    tokens_per_video = phases * (height // 2) * (width // 2)
    total_tokens = 2 * tokens_per_video

    def tensor(key: str) -> Any:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            fail(f"{label} real-source packed state is missing tensor {key}")
        return value

    def dtype_label(value: Any) -> str:
        rendered = str(value.dtype)
        if not rendered.startswith("torch."):
            fail(f"{label} real-source packed state has an unknown dtype label")
        return rendered.split(".", 1)[1]

    latents = tensor("input_vae_latents")
    rope = tensor("input_vae_rope")
    selector = tensor("vae_latents_mask")
    vae_seqlen = tensor("vae_seqlen")
    timesteps = tensor("timesteps")
    target_velocity = tensor("target_velocity")
    target_lens = tensor("target_lens")

    packed_tail = (channels, 1, 2, 2)
    if latents.ndim != 5 or tuple(map(int, latents.shape)) != (
        total_tokens,
        *packed_tail,
    ):
        fail(f"{label} real-source input_vae_latents packed geometry differs")
    if latents.dtype != torch.float32 or not bool(
        torch.isfinite(latents).all().item()
    ):
        fail(f"{label} real-source input_vae_latents must be finite fp32")
    if (
        tuple(map(int, rope.shape)) != (total_tokens, 1, 64)
        or rope.dtype != torch.complex128
    ):
        fail(f"{label} real-source input_vae_rope token geometry differs")
    if not bool(torch.isfinite(rope).all().item()):
        fail(f"{label} real-source input_vae_rope is non-finite")
    if (
        tuple(map(int, selector.shape)) != (1, total_tokens)
        or selector.dtype != torch.bool
    ):
        fail(f"{label} real-source VAE selector geometry differs")
    flat_selector = selector.reshape(-1)
    selector_transition_count = int(
        (flat_selector[1:] != flat_selector[:-1]).sum().item()
    )
    if (
        bool(flat_selector[:tokens_per_video].any().item())
        or not bool(flat_selector[tokens_per_video:].all().item())
        or int(flat_selector.sum().item()) != tokens_per_video
        or selector_transition_count != 1
    ):
        fail(f"{label} real-source selector is not source-then-target contiguous")
    if (
        tuple(map(int, vae_seqlen.shape)) != (1, 1)
        or vae_seqlen.dtype != torch.int64
        or int(vae_seqlen.reshape(-1)[0].item()) != total_tokens
    ):
        fail(f"{label} real-source vae_seqlen differs")
    if (
        tuple(map(int, timesteps.shape)) != (1, 1)
        or timesteps.dtype != torch.bfloat16
    ):
        fail(f"{label} real-source timestep geometry differs")
    timestep = float(timesteps.float().reshape(-1)[0].item())
    if (
        not bool(torch.isfinite(timesteps).all().item())
        or not 0.0 <= timestep <= 1000.0
    ):
        fail(f"{label} real-source timestep is non-finite or outside [0,1000]")
    if tuple(map(int, target_velocity.shape)) != (
        tokens_per_video,
        *packed_tail,
    ):
        fail(f"{label} real-source target_velocity packed geometry differs")
    if target_velocity.dtype != torch.float32 or not bool(
        torch.isfinite(target_velocity).all().item()
    ):
        fail(f"{label} real-source target_velocity must be finite fp32")
    if (
        tuple(map(int, target_lens.shape)) != (1, 1)
        or target_lens.dtype != torch.int64
        or int(target_lens.reshape(-1)[0].item()) != tokens_per_video
    ):
        fail(f"{label} real-source target_lens differs")
    if any(
        value.device != latents.device
        for value in (
            rope,
            selector,
            vae_seqlen,
            timesteps,
            target_velocity,
            target_lens,
        )
    ):
        fail(f"{label} real-source packed state spans multiple devices")

    return {
        "unpacked_spatial_shape": list(clean_shape),
        "latent_phases": phases,
        "tokens_per_video": tokens_per_video,
        "packed_total_tokens": total_tokens,
        "source_token_count": tokens_per_video,
        "target_token_count": tokens_per_video,
        "selector_transition_count": selector_transition_count,
        "input_vae_latents_shape": list(map(int, latents.shape)),
        "input_vae_rope_shape": list(map(int, rope.shape)),
        "vae_latents_mask_shape": list(map(int, selector.shape)),
        "vae_seqlen": int(vae_seqlen.reshape(-1)[0].item()),
        "timestep_value": timestep,
        "timestep_range_inclusive": [0.0, 1000.0],
        "target_velocity_shape": list(map(int, target_velocity.shape)),
        "target_lens": int(target_lens.reshape(-1)[0].item()),
        "state_fields": {
            key: {
                "shape": list(map(int, value.shape)),
                "dtype": dtype_label(value),
            }
            for key, value in (
                ("input_vae_latents", latents),
                ("input_vae_rope", rope),
                ("vae_latents_mask", selector),
                ("vae_seqlen", vae_seqlen),
                ("timesteps", timesteps),
                ("target_velocity", target_velocity),
                ("target_lens", target_lens),
            )
        },
    }


def real_source_prebind_state_diagnostic(
    action_batch: Mapping[str, Any],
    source_caption_batch: Mapping[str, Any],
    *,
    spatial_shape: Sequence[int],
) -> dict[str, Any]:
    """Audit raw same-seed states before any explicit tensor aliasing."""

    import torch

    action_geometry = validate_real_source_packed_state(
        action_batch,
        spatial_shape=spatial_shape,
        label="action-caption",
    )
    source_geometry = validate_real_source_packed_state(
        source_caption_batch,
        spatial_shape=spatial_shape,
        label="source-caption",
    )
    if action_geometry != source_geometry:
        fail("real-source raw same-seed packed geometries differ before binding")

    exact_by_field: dict[str, bool] = {}
    for key in REAL_SOURCE_EXACT_STATE_KEYS:
        action = action_batch.get(key)
        source = source_caption_batch.get(key)
        if (
            not isinstance(action, torch.Tensor)
            or not isinstance(source, torch.Tensor)
            or tuple(action.shape) != tuple(source.shape)
            or action.dtype != source.dtype
            or action.device != source.device
        ):
            fail(
                "real-source raw same-seed state geometry differs before "
                f"binding: {key}"
            )
        exact_by_field[key] = bool(torch.equal(action, source))
    unequal_fields = [
        key for key in REAL_SOURCE_EXACT_STATE_KEYS if not exact_by_field[key]
    ]
    return {
        "schema_version": "bernini-real-source-prebind-packed-state-v1",
        "raw_same_seed_state_exact": not unequal_fields,
        "raw_same_seed_exact_by_field": exact_by_field,
        "raw_same_seed_unequal_fields": unequal_fields,
        "packed_geometry": action_geometry,
    }


def bind_real_source_caption_to_action_state(
    action_batch: Mapping[str, Any],
    source_caption_batch: Mapping[str, Any],
    *,
    spatial_shape: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep source-caption text while explicitly sharing one VAE/FM state.

    The two independently tokenized captions must first prove that Bernini's
    raw same-seed transform produced byte-identical VAE/FM state.  Only then is
    the action transform made the sole state authority by aliasing those exact
    tensors; the source branch contributes text fields only.  This ordering
    prevents aliasing from hiding future RNG or collation drift.
    """

    diagnostic = real_source_prebind_state_diagnostic(
        action_batch,
        source_caption_batch,
        spatial_shape=spatial_shape,
    )
    if diagnostic["raw_same_seed_state_exact"] is not True:
        fields = ",".join(diagnostic["raw_same_seed_unequal_fields"])
        fail(f"real-source raw same-seed state differs before binding: {fields}")

    bound = dict(source_caption_batch)
    for key in REAL_SOURCE_EXACT_STATE_KEYS:
        # Deliberately alias the exact tensor.  Neither branch mutates renderer
        # inputs, and aliasing makes accidental stochastic divergence visible.
        bound[key] = action_batch[key]
    require_same_real_source_noisy_state(
        action_batch,
        bound,
        spatial_shape=spatial_shape,
    )
    return bound, diagnostic


def require_same_real_source_noisy_state(
    action_batch: Mapping[str, Any],
    source_batch: Mapping[str, Any],
    *,
    spatial_shape: Sequence[int],
) -> tuple[int, ...]:
    """Fail closed unless two captions share one exact packed VAE/FM state."""

    diagnostic = real_source_prebind_state_diagnostic(
        action_batch,
        source_batch,
        spatial_shape=spatial_shape,
    )
    if diagnostic["raw_same_seed_state_exact"] is not True:
        fail("real-source action/source pair did not share exact noisy state")
    return tuple(map(int, action_batch["input_vae_latents"].shape))


def donor_row(
    row: Mapping[str, Any],
    registry: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    donor_index: int,
) -> Mapping[str, Any]:
    variants = ["v0", "v1", "v2", "v3"]
    target_variant = str(row["variant_id"])
    candidates = [variant for variant in variants if variant != target_variant]
    selected = candidates[int(donor_index) % len(candidates)]
    donor = registry[(str(row["event_id"]), selected)]
    if donor["iid"] == row["iid"]:
        fail("online anchor must be cross-appearance")
    return donor


def _blob(clean: Any, mean: Any, std: Any) -> bytes:
    return v4.normalized_clean_to_posterior_blob(clean, mean, std)


def source_clean_for_variant(
    *, target: Any, noop: Any, incomplete: Any, variant: str
) -> Any:
    """Return one source-owned trajectory without importing anchor content."""

    import torch

    if variant == "noop":
        source = noop
    elif variant == "incomplete":
        source = incomplete
    elif variant == "reverse":
        index = torch.tensor(
            REVERSE_PHASE_INDICES, dtype=torch.long, device=target.device
        )
        source = target.index_select(2, index)
    elif variant == "shuffle":
        index = torch.tensor(
            SHUFFLE_PHASE_INDICES, dtype=torch.long, device=target.device
        )
        source = target.index_select(2, index)
    else:
        fail("unknown source counterfactual variant")
    if tuple(map(int, source.shape)) != tuple(map(int, target.shape)):
        fail("source counterfactual geometry differs")
    if variant in ("reverse", "shuffle") and not torch.equal(
        source[:, :, 0], target[:, :, 0]
    ):
        fail("temporal counterfactual changed phase zero")
    return source.contiguous()


def source_variants_for_update(
    *, source_variant: str, global_step: int, row_count: int
) -> tuple[str, str]:
    if source_variant == "noop":
        return ("noop", "noop")
    if source_variant == "mixed":
        return ("noop", "incomplete")
    if source_variant != "counterfactual4" or row_count <= 0:
        fail("unknown source-variant schedule")
    cycle = (int(global_step) // int(row_count)) % 2
    return (
        ("noop", "incomplete")
        if cycle == 0
        else ("reverse", "shuffle")
    )


def build_action_record(
    *,
    row: Mapping[str, Any],
    variant: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> dict[str, Any]:
    target, noop, incomplete, _, _ = pairs.load_row_tensors(row)
    source = source_clean_for_variant(
        target=target, noop=noop, incomplete=incomplete, variant=variant
    )
    batch = transform(
        data.make_sample(
            instruction=str(row["instruction"]),
            source_blob=_blob(source, mean, std),
            target_blob=_blob(target, mean, std),
        ),
        seed,
    )
    return {
        "batch": batch,
        "shape": tuple(map(int, target.shape)),
        "iid": str(row["iid"]),
        "variant": variant,
        "timestep": float(batch["timesteps"].float().reshape(-1)[0].item()),
    }


def build_anchor_batches(
    *,
    target_row: Mapping[str, Any],
    donor: Mapping[str, Any],
    profile: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    captions: Optional[Mapping[tuple[str, str], Mapping[str, str]]] = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, ...]]:
    import torch

    dynamic, _, _, _, _ = pairs.load_row_tensors(donor)
    static = dynamic[:, :, :1].expand_as(dynamic).clone().contiguous()
    if captions is None:
        action_prompt = str(target_row["instruction"])
        noop_prompt = NOOP_BY_EVENT[str(target_row["event_id"])]
    else:
        key = (str(donor["event_id"]), str(donor["variant_id"]))
        try:
            donor_captions = captions[key]
        except KeyError as error:
            raise OnlineAnchorTrainingError("anchor caption-I2V row is absent") from error
        action_prompt = donor_captions["target"]
        noop_prompt = donor_captions["noop"]
    if profile == "action_noop":
        action_clean, contrast_clean = dynamic, dynamic
        contrast_prompt = noop_prompt
    elif profile == "dynamic_static":
        action_clean, contrast_clean = dynamic, static
        contrast_prompt = action_prompt
    elif profile == "hybrid":
        action_clean, contrast_clean = dynamic, static
        contrast_prompt = noop_prompt
    else:
        fail("no-anchor profile cannot construct anchor batches")
    action = transform(
        data.make_sample(
            instruction=action_prompt,
            source_blob=None,
            target_blob=_blob(action_clean, mean, std),
        ),
        seed,
    )
    contrast = transform(
        data.make_sample(
            instruction=contrast_prompt,
            source_blob=None,
            target_blob=_blob(contrast_clean, mean, std),
        ),
        seed,
    )
    action_t = float(action["timesteps"].float().reshape(-1)[0].item())
    contrast_t = float(contrast["timesteps"].float().reshape(-1)[0].item())
    if action_t != contrast_t:
        fail("online anchor contrast did not share a timestep")
    if profile == "action_noop" and not torch.equal(
        action["input_vae_latents"], contrast["input_vae_latents"]
    ):
        fail("action/noop anchor contrast did not share the exact noisy state")
    return action, contrast, tuple(map(int, dynamic.shape))


def build_source_reconstruction_record(
    *,
    action_row: Mapping[str, Any],
    variant: str,
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
    prompt_mode: str = "action",
) -> dict[str, Any]:
    target, noop, incomplete, _, _ = pairs.load_row_tensors(action_row)
    source_clean = source_clean_for_variant(
        target=target, noop=noop, incomplete=incomplete, variant=variant
    )
    source_blob = _blob(source_clean, mean, std)
    if prompt_mode == "action":
        instruction = str(action_row["instruction"])
    elif prompt_mode == "noop":
        instruction = NOOP_BY_EVENT[str(action_row["event_id"])]
    elif prompt_mode == "identity":
        instruction = IDENTITY_REPLAY_PROMPT
    else:
        fail("unknown source reconstruction prompt mode")
    batch = transform(
        data.make_sample(
            instruction=instruction,
            source_blob=source_blob,
            target_blob=source_blob,
        ),
        seed,
    )
    return {"batch": batch, "shape": tuple(map(int, source_clean.shape))}


def velocity_prediction_and_target(
    renderer: Any, record: Mapping[str, Any]
) -> tuple[Any, Any]:
    prediction = data.predicted_target_velocity(
        renderer, record["batch"], spatial_shape=record["shape"]
    )
    target = v4._velocity_target(record["batch"], record["shape"])
    if tuple(prediction.shape) != tuple(target.shape):
        fail("predicted and target velocity geometry differs")
    return prediction, target


def finite_mse(prediction: Any, target: Any, *, name: str) -> Any:
    import torch

    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    if not loss.requires_grad or not bool(torch.isfinite(loss).item()):
        fail(f"{name} lost its finite gradient graph")
    return loss


def velocity_loss(renderer: Any, record: Mapping[str, Any]) -> Any:
    prediction, target = velocity_prediction_and_target(renderer, record)
    return finite_mse(prediction, target, name="online-anchor FM loss")


def paired_delta_loss(
    *,
    action_prediction: Any,
    source_prediction: Any,
    action_target: Any,
    source_target: Any,
    name: str,
) -> Any:
    shapes = {
        tuple(tensor.shape)
        for tensor in (
            action_prediction,
            source_prediction,
            action_target,
            source_target,
        )
    }
    if len(shapes) != 1:
        fail("paired velocity-difference geometry differs")
    return finite_mse(
        action_prediction - source_prediction,
        action_target - source_target,
        name=name,
    )


def teacher_delta_tensor(value: Any, *, mode: str) -> Any:
    if value.ndim != 5 or int(value.shape[2]) != 21:
        fail("teacher velocity delta is not a complete 21-phase video tensor")
    if mode == "raw":
        return value
    if mode == "phase0_relative":
        return (value - value[:, :, :1]).contiguous()
    fail("unknown teacher delta mode")


def real_source_teacher_delta_loss(
    *,
    action_prediction: Any,
    source_prediction: Any,
    teacher_action: Any,
    teacher_noop: Any,
    mode: str,
    name: str,
) -> Any:
    shapes = {
        tuple(tensor.shape)
        for tensor in (
            action_prediction,
            source_prediction,
            teacher_action,
            teacher_noop,
        )
    }
    if len(shapes) != 1:
        fail("real-source student and T2V teacher geometries differ")
    student = teacher_delta_tensor(
        action_prediction - source_prediction, mode=mode
    )
    teacher = teacher_delta_tensor(
        teacher_action - teacher_noop, mode=mode
    ).detach()
    return finite_mse(student, teacher, name=name)


def routed_teacher_supervision_residual(
    *,
    action_prediction: Any,
    source_prediction: Any,
    teacher_action: Any,
    teacher_source: Any,
    mode: str,
) -> Any:
    """Return a full target-coordinate student-minus-teacher velocity delta."""

    tensors = (
        action_prediction,
        source_prediction,
        teacher_action,
        teacher_source,
    )
    if any(value.ndim != 5 or int(value.shape[2]) != 21 for value in tensors):
        fail("routed teacher supervision is not a complete 21-phase video tensor")
    if len({tuple(value.shape) for value in tensors}) != 1:
        fail("routed teacher/student target-coordinate geometries differ")
    student = teacher_delta_tensor(action_prediction - source_prediction, mode=mode)
    teacher = teacher_delta_tensor(teacher_action - teacher_source, mode=mode).detach()
    return (student - teacher).contiguous()


def real_source_routed_teacher_delta_loss(
    *,
    action_prediction: Any,
    source_prediction: Any,
    teacher_action: Any,
    teacher_source: Any,
    mode: str,
    name: str,
) -> Any:
    import torch

    residual = routed_teacher_supervision_residual(
        action_prediction=action_prediction,
        source_prediction=source_prediction,
        teacher_action=teacher_action,
        teacher_source=teacher_source,
        mode=mode,
    )
    return finite_mse(residual, torch.zeros_like(residual), name=name)


def anchor_route_replay_uses(training_objective: str) -> int:
    """The routed teacher and student consume one capture exactly twice."""

    if training_objective not in TRAINING_OBJECTIVES:
        fail("unknown training objective for route replay")
    return 2 if training_objective in ROUTED_TEACHER_OBJECTIVES else 1


def qk_transport_for_route_operator(route_operator: str) -> str:
    mapping = {
        "self_temporal_kernel": qk.TEMPORAL_KERNEL_CONTRAST_ATTN_OUTPUT,
        "self_target_gated_kernel25": (
            qk.TARGET_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT
        ),
        "self_correspondence_kernel25": (
            qk.CORRESPONDENCE_GATED_HARD_KERNEL_TOP25_ATTN_OUTPUT
        ),
        "self_target_owned_temporal_kernel_v14r2": (
            qk.TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2
        ),
        "self_target_owned_activity_kernel10_v14r2": (
            qk.TARGET_OWNED_ACTIVITY_KERNEL_TOP10_ATTN_OUTPUT_V14R2
        ),
        "self_target_owned_activity_kernel25_v14r2": (
            qk.TARGET_OWNED_ACTIVITY_KERNEL_TOP25_ATTN_OUTPUT_V14R2
        ),
    }
    try:
        return mapping[route_operator]
    except KeyError:
        fail("unknown self-attention route operator")


def install_activation_checkpointing(model: Any) -> tuple[int, ...]:
    import torch
    from torch.utils.checkpoint import checkpoint

    transformer = model.get_base_model().diff_dec.transformer
    blocks = transformer.blocks
    if len(blocks) != BLOCK_COUNT:
        fail("online-anchor training requires the exact 30-block transformer")
    for index in CHECKPOINT_BLOCKS:
        block = blocks[index]
        original = block.forward

        def checkpointed_forward(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if not torch.is_grad_enabled():
                return _original(*args, **kwargs)
            return checkpoint(_original, *args, use_reentrant=False, **kwargs)

        block.forward = checkpointed_forward
    return CHECKPOINT_BLOCKS


def capture_anchor_route(
    *,
    renderer: Any,
    cache: Any,
    action_batch: Mapping[str, Any],
    contrast_batch: Mapping[str, Any],
    shape: Sequence[int],
    invocation_index: int,
    candidate_index: int,
    rank: int,
    world_size: int,
    strength: float,
    route_operator: str,
    replay_uses: int = 1,
    capture_only: bool = False,
) -> Optional[tuple[Any, Any]]:
    import torch

    disabled = getattr(renderer, "disable_adapter", None)
    if not callable(disabled):
        fail("PEFT adapter-disable context is unavailable")
    common = {
        "cache_bank": cache,
        "step_index": int(invocation_index),
        "candidate_index": int(candidate_index),
        "rank": int(rank),
        "ulysses_size": int(world_size),
        "transport_strength": float(strength),
        "replay_uses": int(replay_uses),
        "replay_scope": cross.FULL_SEQUENCE,
    }
    # ``no_grad`` keeps the cached teacher tensors ordinary tensors.  Tensors
    # created by ``inference_mode`` cannot safely participate as detached
    # constants in the later autograd replay on every supported torch build.
    if route_operator == "cross_sparse":
        invocation_type = cross.AnchorCrossAttentionInvocation
        invocation_context = cross.anchor_cross_attention_invocation
        capture_mode = cross.CAPTURE
        action_slot = cross.ACTION_SLOT
        noop_slot = cross.NOOP_SLOT
        route_kwargs: dict[str, Any] = {}
    elif route_operator in tuple(
        item for item in ROUTE_OPERATORS if item != "cross_sparse"
    ):
        invocation_type = qk.AnchorQKInvocation
        invocation_context = qk.anchor_qk_invocation
        capture_mode = qk.CAPTURE
        action_slot = qk.ACTION_SLOT
        noop_slot = qk.NOOP_SLOT
        route_kwargs = {"transport": qk_transport_for_route_operator(route_operator)}
    else:
        fail("unknown online anchor route operator")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), disabled():
        with invocation_context(
            invocation_type(
                mode=capture_mode, slot=action_slot, **route_kwargs, **common
            )
        ):
            action_velocity = data.predicted_target_velocity(
                renderer, action_batch, spatial_shape=shape
            )
        with invocation_context(
            invocation_type(
                mode=capture_mode, slot=noop_slot, **route_kwargs, **common
            )
        ):
            contrast_velocity = data.predicted_target_velocity(
                renderer, contrast_batch, spatial_shape=shape
            )
    if tuple(action_velocity.shape) != tuple(contrast_velocity.shape):
        fail("online anchor action/noop velocity geometry differs")
    if capture_only:
        # The donor forward exists only to populate the attention-route cache.
        # In particular, its post-head velocity cannot become a target in the
        # target-coordinate routed-teacher objective.
        return None
    return action_velocity.detach(), contrast_velocity.detach()


def replay_invocation(
    *,
    cache: Any,
    invocation_index: int,
    candidate_index: int,
    rank: int,
    world_size: int,
    strength: float,
    route_operator: str,
    replay_uses: int = 1,
) -> tuple[Any, Any]:
    common = {
        "cache_bank": cache,
        "step_index": int(invocation_index),
        "candidate_index": int(candidate_index),
        "rank": int(rank),
        "ulysses_size": int(world_size),
        "transport_strength": float(strength),
        "replay_uses": int(replay_uses),
    }
    if route_operator == "cross_sparse":
        return (
            cross.AnchorCrossAttentionInvocation(
                mode=cross.REPLAY,
                replay_scope=cross.PAIRED_SUFFIX,
                slot=cross.ACTION_SLOT,
                **common,
            ),
            cross.anchor_cross_attention_invocation,
        )
    if route_operator in tuple(
        item for item in ROUTE_OPERATORS if item != "cross_sparse"
    ):
        return (
            qk.AnchorQKInvocation(
                mode=qk.REPLAY,
                replay_scope=qk.PAIRED_SUFFIX,
                slot=qk.ACTION_SLOT,
                transport=qk_transport_for_route_operator(route_operator),
                **common,
            ),
            qk.anchor_qk_invocation,
        )
    fail("unknown online anchor route operator")


def gradient_coverage(named: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    nonzero = []
    for name, parameter in named:
        if parameter.grad is None:
            fail(f"LoRA parameter has no gradient: {name}")
        if bool(parameter.grad.detach().ne(0).any().item()):
            nonzero.append(name)
    return {
        "tensor_count": len(named),
        "nonzero_tensor_count": len(nonzero),
        "nonzero_names_sha256": legacy.object_sha256(nonzero),
    }


def component_gradient_probe(
    named: Sequence[tuple[str, Any]], *, epsilon: float = 1.0e-12
) -> Mapping[str, Any]:
    """Describe one already-all-reduced gradient component without conflation."""

    import math
    import torch

    if not named or not 0.0 < float(epsilon) < 1.0:
        fail("component-gradient probe controls differ")
    squared = torch.zeros((), dtype=torch.float64, device=named[0][1].grad.device)
    nonzero: list[str] = []
    epsilon_active: list[str] = []
    side = {
        "lora_A": {"tensor_count": 0, "nonzero_tensor_count": 0, "epsilon_active_tensor_count": 0},
        "lora_B": {"tensor_count": 0, "nonzero_tensor_count": 0, "epsilon_active_tensor_count": 0},
    }
    for name, parameter in named:
        if parameter.grad is None:
            fail(f"component gradient is absent: {name}")
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all().item()):
            fail(f"component gradient is non-finite: {name}")
        squared += gradient.double().square().sum()
        exact = bool(gradient.ne(0).any().item())
        active = bool(gradient.abs().max().item() > float(epsilon))
        if exact:
            nonzero.append(name)
        if active:
            epsilon_active.append(name)
        label = "lora_A" if "lora_A" in name else "lora_B" if "lora_B" in name else None
        if label is None:
            fail(f"component gradient parameter is not an A/B LoRA tensor: {name}")
        side[label]["tensor_count"] += 1
        side[label]["nonzero_tensor_count"] += int(exact)
        side[label]["epsilon_active_tensor_count"] += int(active)
    norm = math.sqrt(float(squared.item()))
    if not math.isfinite(norm):
        fail("component gradient norm is non-finite")
    return {
        "scope": "global_average_preclip",
        "tensor_count": len(named),
        "nonzero_tensor_count": len(nonzero),
        "epsilon": float(epsilon),
        "epsilon_active_tensor_count": len(epsilon_active),
        "l2_norm_fp64": norm,
        "nonzero_names_sha256": legacy.object_sha256(nonzero),
        "epsilon_active_names_sha256": legacy.object_sha256(epsilon_active),
        "adapter_sides": side,
    }


def validate_v14r2_component_coverage(
    probe: Mapping[str, Any], *, step: int, component: str
) -> None:
    """Fail closed on the pre-registered two-step LoRA gradient topology."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        fail("v14r2 component-gradient step is invalid")
    if component not in ("action", "raw_replay"):
        fail("v14r2 component-gradient label is invalid")
    sides = probe.get("adapter_sides")
    if not isinstance(sides, Mapping):
        fail("v14r2 component-gradient A/B receipt is absent")
    side_a = sides.get("lora_A")
    side_b = sides.get("lora_B")
    if not isinstance(side_a, Mapping) or not isinstance(side_b, Mapping):
        fail("v14r2 component-gradient A/B receipt differs")
    if (
        probe.get("tensor_count") != LORA_TRAINABLE_TENSOR_COUNT
        or side_a.get("tensor_count") != LORA_TARGET_MODULE_COUNT
        or side_b.get("tensor_count") != LORA_TARGET_MODULE_COUNT
    ):
        fail("v14r2 component-gradient tensor registry differs")

    if component == "action" and step == 1:
        expected_total = LORA_TARGET_MODULE_COUNT
        expected_a = 0
        expected_b = LORA_TARGET_MODULE_COUNT
    elif step >= 2:
        expected_total = LORA_TRAINABLE_TENSOR_COUNT
        expected_a = LORA_TARGET_MODULE_COUNT
        expected_b = LORA_TARGET_MODULE_COUNT
    else:
        # Raw replay step 1 is diagnostic only; step 2 is the hard closure.
        return

    for key in ("nonzero_tensor_count", "epsilon_active_tensor_count"):
        if probe.get(key) != expected_total:
            fail(
                f"v14r2 {component} {key} differs from the step-{step} contract"
            )
        if side_a.get(key) != expected_a or side_b.get(key) != expected_b:
            fail(
                f"v14r2 {component} A/B {key} differs from the step-{step} contract"
            )


def clone_component_gradients(named: Sequence[tuple[str, Any]]) -> tuple[Any, ...]:
    return tuple(parameter.grad.detach().clone() for _, parameter in named)


def clone_trainable_parameter_values(
    named: Sequence[tuple[str, Any]],
) -> tuple[Any, ...]:
    """Snapshot the exact stored LoRA values without changing optimizer state.

    The snapshot intentionally keeps each parameter's native dtype and device.
    Converting the complete 188M-parameter adapter to fp32/fp64 here would add a
    much larger persistent allocation; the audit converts one tensor at a time
    to fp64 only while accumulating its post-step geometry.
    """

    return tuple(parameter.detach().clone() for _, parameter in named)


def actual_optimizer_update_probe(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    step: int,
) -> Mapping[str, Any]:
    """Audit the actual optimizer displacement against both raw objectives.

    ``action_gradients`` and ``raw_replay_gradients`` are the independently
    all-reduced, pre-merge global gradients.  The optimizer and clipping logic
    run before this function and are not reproduced or approximated here.  In
    particular, this observes AdamW's real adaptive parameter displacement
    rather than treating the merged first-order gradient as the update.
    """

    import math
    import torch

    count = len(named)
    if (
        count == 0
        or len(parameter_values_before_step) != count
        or len(action_gradients) != count
        or len(raw_replay_gradients) != count
    ):
        fail("actual optimizer-update probe closure differs")
    if replay_combine_mode not in REPLAY_COMBINE_MODES:
        fail("actual optimizer-update probe has an unknown combine mode")
    if int(step) <= 0:
        fail("actual optimizer-update probe step must be positive")

    first_parameter = named[0][1]
    device = first_parameter.device
    parameter_sq = torch.zeros((), dtype=torch.float64, device=device)
    delta_sq = torch.zeros_like(parameter_sq)
    action_sq = torch.zeros_like(parameter_sq)
    replay_sq = torch.zeros_like(parameter_sq)
    action_dot_delta = torch.zeros_like(parameter_sq)
    replay_dot_delta = torch.zeros_like(parameter_sq)
    changed_tensor_count = torch.zeros((), dtype=torch.int64, device=device)
    changed_element_count = torch.zeros_like(changed_tensor_count)
    total_element_count = 0
    snapshot_dtypes = set()

    for (
        (name, parameter),
        before,
        action,
        replay,
    ) in zip(
        named,
        parameter_values_before_step,
        action_gradients,
        raw_replay_gradients,
    ):
        if (
            tuple(before.shape) != tuple(parameter.shape)
            or tuple(action.shape) != tuple(parameter.shape)
            or tuple(replay.shape) != tuple(parameter.shape)
        ):
            fail(f"actual optimizer-update probe geometry differs: {name}")
        if (
            before.device != parameter.device
            or action.device != parameter.device
            or replay.device != parameter.device
        ):
            fail(f"actual optimizer-update probe device differs: {name}")
        if before.dtype != parameter.dtype:
            fail(f"actual optimizer-update snapshot dtype differs: {name}")
        if not bool(torch.isfinite(parameter).all().item()):
            fail(f"post-step LoRA parameter is non-finite: {name}")
        if not bool(torch.isfinite(before).all().item()):
            fail(f"pre-step LoRA parameter snapshot is non-finite: {name}")
        if not bool(torch.isfinite(action).all().item()):
            fail(f"actual-update action gradient is non-finite: {name}")
        if not bool(torch.isfinite(replay).all().item()):
            fail(f"actual-update raw replay gradient is non-finite: {name}")

        # Subtract after conversion so the audit measures the difference of the
        # two exactly stored values rather than rounding the subtraction in the
        # (usually bf16) parameter dtype.
        before64 = before.detach().double()
        after64 = parameter.detach().double()
        delta64 = after64.sub(before64)
        action64 = action.detach().double()
        replay64 = replay.detach().double()
        parameter_sq += before64.square().sum()
        delta_sq += delta64.square().sum()
        action_sq += action64.square().sum()
        replay_sq += replay64.square().sum()
        action_dot_delta += (action64 * delta64).sum()
        replay_dot_delta += (replay64 * delta64).sum()
        changed = parameter.detach().ne(before)
        changed_tensor_count += changed.any().to(dtype=torch.int64)
        changed_element_count += changed.count_nonzero().to(dtype=torch.int64)
        total_element_count += int(parameter.numel())
        dtype_name = str(before.dtype)
        snapshot_dtypes.add(
            dtype_name.split("torch.", 1)[1]
            if dtype_name.startswith("torch.")
            else dtype_name
        )

    parameter_norm = math.sqrt(float(parameter_sq.item()))
    delta_norm = math.sqrt(float(delta_sq.item()))
    action_norm = math.sqrt(float(action_sq.item()))
    replay_norm = math.sqrt(float(replay_sq.item()))
    action_dot_delta_value = float(action_dot_delta.item())
    replay_dot_delta_value = float(replay_dot_delta.item())
    action_descent = -action_dot_delta_value
    source_descent = -replay_dot_delta_value
    if delta_norm <= 0.0:
        fail("actual optimizer update changed no LoRA parameter")
    if action_norm <= 0.0 or replay_norm <= 0.0:
        fail("actual optimizer-update probe requires two nonzero raw gradients")

    action_scale = action_norm * delta_norm
    replay_scale = replay_norm * delta_norm
    # The dot products themselves are accumulated in fp64.  This tolerance is
    # only for deciding whether an optimizer displacement lies infinitesimally
    # outside the replay descent half-space.
    source_descent_tolerance = max(1.0e-30, 1.0e-10 * replay_scale)
    source_descent_required = replay_combine_mode in (
        "norm_balanced_025",
        "source_halfspace_001",
    )
    action_descent_passed = action_descent > 0.0
    source_descent_passed = source_descent >= -source_descent_tolerance
    if not action_descent_passed:
        fail(
            "actual optimizer update is not an action-descent step: "
            f"mode={replay_combine_mode}, action_descent={action_descent!r}, "
            f"action_dot_delta={action_dot_delta_value!r}, "
            f"delta_l2={delta_norm!r}"
        )
    if source_descent_required and not source_descent_passed:
        fail(
            "actual optimizer update left the required source-descent half-space: "
            f"mode={replay_combine_mode}, source_descent={source_descent!r}, "
            f"minimum_allowed={-source_descent_tolerance!r}, "
            f"raw_replay_dot_delta={replay_dot_delta_value!r}, "
            f"delta_l2={delta_norm!r}"
        )

    values = {
        "schema_version": "bernini-actual-optimizer-update-probe-v1",
        "step": int(step),
        "replay_combine_mode": replay_combine_mode,
        "gradient_scope": "separately_allreduced_global_action_and_raw_replay",
        "optimizer_semantics_observed_not_modified": True,
        "parameter_snapshot_native_dtype": True,
        "parameter_snapshot_dtypes": sorted(snapshot_dtypes),
        "tensor_count": count,
        "parameter_element_count": total_element_count,
        "changed_tensor_count": int(changed_tensor_count.item()),
        "changed_element_count": int(changed_element_count.item()),
        "parameter_l2_norm_before_step_fp64": parameter_norm,
        "delta_theta_l2_norm_fp64": delta_norm,
        "delta_theta_relative_parameter_l2_norm": (
            delta_norm / parameter_norm if parameter_norm > 0.0 else None
        ),
        "action_gradient_l2_norm_fp64": action_norm,
        "raw_replay_gradient_l2_norm_fp64": replay_norm,
        "delta_theta_to_action_gradient_l2_ratio": delta_norm / action_norm,
        "delta_theta_to_raw_replay_gradient_l2_ratio": delta_norm / replay_norm,
        "action_gradient_dot_delta_theta_fp64": action_dot_delta_value,
        "raw_replay_gradient_dot_delta_theta_fp64": replay_dot_delta_value,
        "action_descent_fp64": action_descent,
        "source_descent_fp64": source_descent,
        "action_descent_cosine": action_descent / action_scale,
        "source_descent_cosine": source_descent / replay_scale,
        "action_descent_required": True,
        "action_descent_passed": action_descent_passed,
        "source_descent_required": source_descent_required,
        "source_descent_tolerance_fp64": source_descent_tolerance,
        "minimum_allowed_source_descent_fp64": -source_descent_tolerance,
        "source_descent_passed": source_descent_passed,
    }
    if not all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        fail("actual optimizer-update probe is non-finite")
    return values


def merge_component_gradients(
    named: Sequence[tuple[str, Any]],
    action_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    base_replay_scale: float,
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    """Combine global action and raw replay gradients with audited geometry.

    ``q`` is the L2 norm of the replay correction actually added to the action
    gradient divided by the action-gradient L2 norm.  The two legacy modes
    retain their historical raw-replay scaling and fraction gate.  The v14r3
    modes are scale invariant and deliberately do not inherit that lower bound.
    """

    import math
    import torch

    if not named or len(named) != len(action_gradients):
        fail("component-gradient merge length differs")
    first_gradient = named[0][1].grad
    if first_gradient is None:
        fail("component-gradient merge has no replay gradient")
    action_sq = torch.zeros((), dtype=torch.float64, device=first_gradient.device)
    replay_sq = torch.zeros_like(action_sq)
    dot = torch.zeros_like(action_sq)
    for (name, parameter), action in zip(named, action_gradients):
        replay = parameter.grad
        if replay is None or tuple(replay.shape) != tuple(action.shape):
            fail(f"component-gradient merge geometry differs: {name}")
        if not bool(torch.isfinite(action).all().item()):
            fail(f"action component gradient is non-finite: {name}")
        if not bool(torch.isfinite(replay).all().item()):
            fail(f"replay component gradient is non-finite: {name}")
        action64 = action.detach().double()
        replay64 = replay.detach().double()
        action_sq += action64.square().sum()
        replay_sq += replay64.square().sum()
        dot += (action64 * replay64).sum()

    if replay_combine_mode not in REPLAY_COMBINE_MODES:
        fail("unknown replay-gradient combine mode")
    if not 0.0 < float(base_replay_scale) <= 1.0:
        fail("base replay-gradient scale is outside (0,1]")
    action_sq_value = float(action_sq.item())
    replay_sq_value = float(replay_sq.item())
    action_norm = math.sqrt(action_sq_value)
    raw_replay_norm = math.sqrt(replay_sq_value)
    raw_dot = float(dot.item())
    if action_norm <= 0.0 or raw_replay_norm <= 0.0:
        fail("action and replay component gradients must both be nonzero")
    action_replay_cosine = raw_dot / (action_norm * raw_replay_norm)
    # Distributed fp64 summation can exceed the closed interval by a few ulps.
    action_replay_cosine = max(-1.0, min(1.0, action_replay_cosine))
    lambda_min = max(0.0, -raw_dot / replay_sq_value)

    projection_coefficient = 0.0
    replay_projection_applied = False
    processed_replay_norm = raw_replay_norm
    processed_replay_action_dot = raw_dot
    mode_feasible = True
    infeasible_reason = None

    if replay_combine_mode == "fixed_0025":
        replay_scale = float(base_replay_scale)
    elif replay_combine_mode == "first_order_safe":
        replay_scale = max(float(base_replay_scale), 1.1 * lambda_min)
    elif replay_combine_mode == "action_only":
        replay_scale = 0.0
    elif replay_combine_mode in ("norm_balanced_005", "norm_balanced_025"):
        target_q = 0.05 if replay_combine_mode == "norm_balanced_005" else 0.25
        replay_scale = target_q * action_norm / raw_replay_norm
    elif replay_combine_mode in ("source_safe_cap025", "source_halfspace_001"):
        requested_q = max(0.01, -action_replay_cosine + 0.01)
        if replay_combine_mode == "source_safe_cap025":
            requested_q = max(0.05, requested_q)
        replay_scale = requested_q * action_norm / raw_replay_norm
        if replay_combine_mode == "source_safe_cap025" and requested_q > 0.25:
            mode_feasible = False
            infeasible_reason = "required_q_exceeds_0.25"
    else:
        # PCGrad removes only a replay component that opposes action.  The
        # remaining replay is normalized to 0.10*A, so action remains primary.
        if raw_dot < 0.0:
            replay_projection_applied = True
            projection_coefficient = raw_dot / action_sq_value
            projected_sq = replay_sq_value - raw_dot * raw_dot / action_sq_value
            processed_replay_norm = math.sqrt(max(0.0, projected_sq))
            processed_replay_action_dot = 0.0
        if processed_replay_norm <= 1.0e-24:
            replay_scale = 0.0
            mode_feasible = False
            infeasible_reason = "projected_replay_is_zero"
        else:
            replay_scale = 0.10 * action_norm / processed_replay_norm

    weighted_replay_norm = replay_scale * processed_replay_norm
    correction_ratio_q = weighted_replay_norm / action_norm
    weighted_fraction = weighted_replay_norm / (action_norm + weighted_replay_norm)
    replay_to_action_ratio = raw_replay_norm / action_norm

    # Exact fp64 geometry of g=a+scale*(r-projection_coefficient*a).
    planned_action_inner = action_sq_value + (
        replay_scale * processed_replay_action_dot
    )
    raw_replay_processed_dot = replay_sq_value - (
        projection_coefficient * raw_dot
    )
    planned_raw_replay_inner = raw_dot + replay_scale * raw_replay_processed_dot
    planned_processed_replay_inner = processed_replay_action_dot + (
        replay_scale * processed_replay_norm * processed_replay_norm
    )
    planned_combined_sq = (
        action_sq_value
        + 2.0 * replay_scale * processed_replay_action_dot
        + replay_scale * replay_scale * processed_replay_norm * processed_replay_norm
    )
    planned_combined_norm = math.sqrt(max(0.0, planned_combined_sq))
    action_alignment_ratio = planned_action_inner / action_sq_value

    if (
        replay_combine_mode == "source_halfspace_001"
        and action_alignment_ratio < 0.1
    ):
        mode_feasible = False
        infeasible_reason = "action_alignment_below_0.1"

    if diagnostic_only:
        fraction_side = (
            "below_min"
            if weighted_fraction < 0.001
            else "above_max"
            if weighted_fraction > 0.25
            else "inside_registered_interval"
        )
        fail(
            "GRADIENT_DIAGNOSTIC_COMPLETE|optimizer_steps=0|"
            f"fraction_side={fraction_side}|"
            f"weighted_fraction={weighted_fraction!r}|"
            f"action_l2={action_norm!r}|raw_replay_l2={raw_replay_norm!r}|"
            f"raw_replay_to_action_ratio={replay_to_action_ratio!r}|"
            f"q={correction_ratio_q!r}|mode_feasible={mode_feasible!r}|"
            f"infeasible_reason={infeasible_reason!r}|"
            f"base_scale={float(base_replay_scale)!r}|"
            f"effective_scale={replay_scale!r}|lambda_min={lambda_min!r}|"
            f"raw_dot={raw_dot!r}|cosine={action_replay_cosine!r}|"
            f"action_alignment={planned_action_inner!r}|"
            f"action_alignment_ratio={action_alignment_ratio!r}|"
            f"replay_inner_product={planned_raw_replay_inner!r}"
        )
    if not mode_feasible:
        fail(
            f"replay-gradient combine mode is infeasible: mode={replay_combine_mode}, "
            f"reason={infeasible_reason}, q={correction_ratio_q!r}, "
            f"action_alignment_ratio={action_alignment_ratio!r}, "
            f"action_l2={action_norm!r}, raw_replay_l2={raw_replay_norm!r}, "
            f"cosine={action_replay_cosine!r}"
        )
    if (
        replay_combine_mode in ("norm_balanced_005", "norm_balanced_025")
        and planned_raw_replay_inner < -1.0e-8
    ):
        fail(
            "norm-balanced replay left the source-descent half-space: "
            f"mode={replay_combine_mode}, "
            f"raw_replay_dot_combined={planned_raw_replay_inner!r}, "
            f"cosine={action_replay_cosine!r}, q={correction_ratio_q!r}"
        )
    if replay_combine_mode == "action_only" and action_replay_cosine < -0.5:
        fail(
            "action-only formal update has excessive action/replay conflict: "
            f"cosine={action_replay_cosine!r}, minimum=-0.5"
        )
    processed_replay_action_cosine = (
        processed_replay_action_dot / (processed_replay_norm * action_norm)
        if processed_replay_norm > 0.0
        else None
    )
    if replay_combine_mode == "action_priority_pcgrad_010":
        pcgrad_geometry_failed = (
            replay_projection_applied
            and (
                processed_replay_norm / raw_replay_norm < 0.2
                or processed_replay_action_cosine is None
                or abs(processed_replay_action_cosine) > 1.0e-5
            )
        ) or (
            not replay_projection_applied and action_replay_cosine < 0.0
        )
        if pcgrad_geometry_failed:
            fail(
                "action-priority PCGrad formal geometry gate failed: "
                f"projection_applied={replay_projection_applied!r}, "
                f"raw_action_replay_cosine={action_replay_cosine!r}, "
                f"retained_raw_norm_fraction="
                f"{processed_replay_norm / raw_replay_norm!r}, "
                f"processed_replay_action_cosine="
                f"{processed_replay_action_cosine!r}"
            )
    if replay_combine_mode == "first_order_safe" and replay_scale > 1.0:
        fail("first-order-safe replay scale exceeds one")
    if replay_combine_mode in ("fixed_0025", "first_order_safe") and (
        weighted_fraction < 0.001 or weighted_fraction > 0.25
    ):
        fail(
            "weighted replay gradient is outside the pre-registered "
            "[0.001,0.25]: "
            f"weighted_fraction={weighted_fraction!r}, "
            f"action_l2={action_norm!r}, raw_replay_l2={raw_replay_norm!r}, "
            f"base_scale={float(base_replay_scale)!r}, "
            f"effective_scale={replay_scale!r}, lambda_min={lambda_min!r}, "
            f"raw_dot={raw_dot!r}, cosine={action_replay_cosine!r}"
        )

    combined_sq = torch.zeros_like(action_sq)
    actual_action_inner = torch.zeros_like(action_sq)
    actual_raw_replay_inner = torch.zeros_like(action_sq)
    actual_processed_replay_inner = torch.zeros_like(action_sq)
    for (_name, parameter), action in zip(named, action_gradients):
        # Avoid materializing another 188M-parameter gradient copy.
        action64 = action.detach().double()
        replay64 = parameter.grad.detach().double()
        processed_replay64 = replay64 - projection_coefficient * action64
        parameter.grad.mul_(replay_scale).add_(
            action, alpha=1.0 - replay_scale * projection_coefficient
        )
        combined64 = parameter.grad.detach().double()
        combined_sq += combined64.square().sum()
        actual_action_inner += (action64 * combined64).sum()
        actual_raw_replay_inner += (replay64 * combined64).sum()
        actual_processed_replay_inner += (processed_replay64 * combined64).sum()
    combined_norm = math.sqrt(float(combined_sq.item()))
    action_combined_inner = float(actual_action_inner.item())
    source_fm_combined_inner = float(actual_raw_replay_inner.item())
    processed_replay_inner = float(actual_processed_replay_inner.item())
    if (
        replay_combine_mode in (
            "first_order_safe",
            "source_safe_cap025",
            "source_halfspace_001",
        )
        and source_fm_combined_inner < -1.0e-8
    ):
        fail("source-safe replay did not protect the source-FM descent direction")
    if (
        replay_combine_mode == "source_halfspace_001"
        and action_combined_inner / action_sq_value < 0.1
    ):
        fail("source half-space correction lost the action-alignment floor")
    if (
        replay_combine_mode == "action_priority_pcgrad_010"
        and action_combined_inner <= 0.0
    ):
        fail("action-priority PCGrad lost the action descent direction")

    values = {
        "action_l2_norm_fp64": action_norm,
        "raw_replay_l2_norm_fp64": raw_replay_norm,
        "processed_replay_l2_norm_fp64": processed_replay_norm,
        "weighted_replay_l2_norm_fp64": weighted_replay_norm,
        "combined_l2_norm_fp64": combined_norm,
        "planned_combined_l2_norm_fp64": planned_combined_norm,
        "action_raw_replay_dot_fp64": raw_dot,
        "action_replay_cosine": action_replay_cosine,
        "replay_combine_mode": replay_combine_mode,
        "base_replay_scale": float(base_replay_scale),
        "first_order_safe_lambda_min": lambda_min,
        "effective_replay_scale": replay_scale,
        "weighted_replay_gradient_fraction": weighted_fraction,
        "weighted_replay_to_action_grad_norm_ratio": correction_ratio_q,
        "replay_component_to_action_norm_ratio_q": correction_ratio_q,
        "correction_ratio_q": correction_ratio_q,
        "replay_projection_applied": replay_projection_applied,
        "replay_projection_coefficient": projection_coefficient,
        "processed_replay_retained_raw_norm_fraction": (
            processed_replay_norm / raw_replay_norm
        ),
        "processed_replay_action_cosine": (
            processed_replay_action_cosine
        ),
        "action_priority_conflict_control_not_source_preservation": (
            replay_combine_mode == "action_priority_pcgrad_010"
        ),
        "action_gradient_dot_combined_gradient_fp64": action_combined_inner,
        "planned_action_gradient_dot_combined_gradient_fp64": planned_action_inner,
        "action_alignment_ratio": action_combined_inner / action_sq_value,
        "action_combined_cosine": (
            action_combined_inner / (action_norm * combined_norm)
            if combined_norm > 0.0
            else None
        ),
        "raw_replay_gradient_dot_combined_gradient_fp64": (
            source_fm_combined_inner
        ),
        "raw_replay_combined_alignment_over_action_replay_norms": (
            source_fm_combined_inner / (action_norm * raw_replay_norm)
        ),
        "planned_raw_replay_gradient_dot_combined_gradient_fp64": (
            planned_raw_replay_inner
        ),
        "raw_replay_combined_cosine": (
            source_fm_combined_inner / (raw_replay_norm * combined_norm)
            if combined_norm > 0.0
            else None
        ),
        "processed_replay_gradient_dot_combined_gradient_fp64": (
            processed_replay_inner
        ),
        "planned_processed_replay_gradient_dot_combined_gradient_fp64": (
            planned_processed_replay_inner
        ),
        "raw_source_fm_gradient_dot_combined_gradient_fp64": (
            source_fm_combined_inner
        ),
        "first_order_source_fm_preserved": source_fm_combined_inner >= -1.0e-8,
    }
    if not all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        fail("component-gradient interaction is non-finite")
    return values


def source_absorption_diagnostic(
    *, action_prediction: Any, source_prediction: Any, source_velocity_target: Any
) -> Mapping[str, float]:
    """Measure which side of an action-source contrast absorbs the separation."""

    import torch

    if (
        tuple(action_prediction.shape) != tuple(source_prediction.shape)
        or tuple(action_prediction.shape) != tuple(source_velocity_target.shape)
    ):
        fail("source-absorption diagnostic geometry differs")
    action = action_prediction.detach().float()
    source = source_prediction.detach().float()
    target = source_velocity_target.detach().float()
    delta = action - source
    denominator = delta.double().square().sum()
    if float(denominator.item()) <= 1.0e-24:
        return {"q_action": 0.0, "q_source": 0.0, "q_sum": 0.0, "defined": False}
    q_action = ((action - target).double() * delta.double()).sum() / denominator
    q_source = (-(source - target).double() * delta.double()).sum() / denominator
    return {
        "q_action": float(q_action.item()),
        "q_source": float(q_source.item()),
        "q_sum": float((q_action + q_source).item()),
        "defined": True,
    }


def checkpoint_receipt(
    *,
    args: argparse.Namespace,
    step: int,
    loss: float,
    action_objective: float,
    source_reconstruction: float,
    effective_replay_scale: float,
    grad_norm: float,
    memory: Mapping[str, Any],
    targets: Sequence[str],
    initial_digest: str,
    cache: cross.AnchorCrossAttentionCache,
    bernini_revision: str,
    veomni_revision: str,
    pair_manifest: Path,
    gradient: Mapping[str, Any],
    action_gradient: Mapping[str, Any],
    replay_gradient: Mapping[str, Any],
    gradient_interaction: Mapping[str, Any],
    actual_optimizer_update: Mapping[str, Any],
    source_absorption: Mapping[str, Any],
    route_off_absolute_anchor: Optional[Mapping[str, Any]] = None,
    real_source_prebind_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    routed_teacher = args.training_objective in ROUTED_TEACHER_OBJECTIVES
    qk_only_routed_teacher = (
        args.training_objective == "real_source_target_owned_routed_teacher_delta_v14r2"
    )
    base_replay_scale = float(args.source_reconstruction_weight)
    base_scaled_reconstruction = base_replay_scale * float(source_reconstruction)
    weighted_reconstruction = float(effective_replay_scale) * float(
        source_reconstruction
    )
    absolute_anchor = (
        dict(route_off_absolute_anchor)
        if route_off_absolute_anchor is not None
        else {"applicable": False, "weighted_mean_fm": None}
    )
    weighted_absolute_anchor = (
        float(absolute_anchor["weighted_mean_fm"])
        if absolute_anchor.get("weighted_mean_fm") is not None
        else 0.0
    )
    component_total = (
        float(action_objective)
        + weighted_reconstruction
        + weighted_absolute_anchor
    )
    return {
        "schema_version": (
            QK_ONLY_RECEIPT_SCHEMA if qk_only_routed_teacher else RECEIPT_SCHEMA
        ),
        "complete": True,
        "global_step": int(step),
        "max_steps": int(args.max_steps),
        # v14r2 does two independent backwards and merges their gradients; it
        # therefore has no single scalar that was backpropagated as a joint
        # objective.  Keep legacy ``last_loss`` only for legacy receipts.
        "last_loss": None if qk_only_routed_teacher else float(loss),
        "last_reporting_scalar": (
            float(loss) if qk_only_routed_teacher else None
        ),
        "last_reporting_scalar_is_not_a_joint_backpropagated_objective": (
            True if qk_only_routed_teacher else None
        ),
        "last_objective_components": {
            "action_objective_unweighted": float(action_objective),
            "source_caption_trajectory_replay_unweighted": (
                float(source_reconstruction)
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else None
            ),
            "base_replay_scale": (
                base_replay_scale if qk_only_routed_teacher else None
            ),
            "effective_replay_scale": (
                float(effective_replay_scale) if qk_only_routed_teacher else None
            ),
            "base_source_replay_scalar_diagnostic": (
                base_scaled_reconstruction if qk_only_routed_teacher else None
            ),
            "effective_source_replay_scalar_for_reporting": (
                weighted_reconstruction if qk_only_routed_teacher else None
            ),
            "effective_source_replay_reporting_fraction": (
                weighted_reconstruction / component_total
                if qk_only_routed_teacher and component_total > 0.0
                else None
            ),
            "source_caption_trajectory_replay_weighted": (
                weighted_reconstruction
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                and not qk_only_routed_teacher
                else None
            ),
            "source_caption_replay_weighted_fraction": (
                weighted_reconstruction / component_total
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                and not qk_only_routed_teacher
                and component_total > 0.0
                else None
            ),
            "same_state_route_off_absolute_anchor_unweighted": (
                absolute_anchor.get("mean_fm")
            ),
            "same_state_route_off_absolute_anchor_weight": (
                absolute_anchor.get("weight")
            ),
            "same_state_route_off_absolute_anchor_weighted": (
                absolute_anchor.get("weighted_mean_fm")
            ),
        },
        "last_preclip_gradient_norm": float(grad_norm),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "method_source_revision": args.method_source_revision,
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": pairs.file_sha256(pair_manifest),
        "authoring": (
            str(Path(args.authoring).resolve()) if args.authoring else None
        ),
        "authoring_sha256": (
            pairs.file_sha256(Path(args.authoring).resolve()) if args.authoring else None
        ),
        "real_source_manifest": (
            str(Path(args.real_source_manifest).resolve())
            if args.real_source_manifest
            else None
        ),
        "real_source_manifest_sha256": (
            args.real_source_manifest_sha256 if args.real_source_manifest else None
        ),
        "memory_gate": dict(memory),
        "gradient_coverage": dict(gradient),
        "component_gradient_probes": {
            "action_objective": dict(action_gradient),
            "raw_source_caption_trajectory_replay": dict(replay_gradient),
            "interaction": dict(gradient_interaction),
        },
        "actual_optimizer_update_probe": dict(actual_optimizer_update),
        "source_absorption_diagnostic": dict(source_absorption),
        "route_off_absolute_anchor_diagnostic": absolute_anchor,
        "real_source_prebind_state": (
            dict(real_source_prebind_state)
            if real_source_prebind_state is not None
            else None
        ),
        "anchor_cache": cache.receipt(),
        "training_contract": {
            "method": METHOD,
            "profile": args.profile,
            "full_attention_lora_enabled": (
                LORA_SCOPE == "all_30_blocks_attn1_attn2_qkvo"
            ),
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_scope": LORA_SCOPE,
            "lora_target_module_count": len(targets),
            "lora_target_modules_sha256": legacy.object_sha256(list(targets)),
            "trainable_parameter_count": LORA_PARAMETERS,
            "component_gradient_epsilon": COMPONENT_GRADIENT_EPSILON,
            "same_action_route_off_gradient_enabled": (
                SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED
            ),
            "same_action_student_delta_gradient_mode": (
                SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE
            ),
            "same_action_route_off_absolute_anchor_enabled": (
                SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED
            ),
            "same_action_route_off_absolute_anchor_weight": (
                SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT
            ),
            "same_action_route_off_absolute_anchor_mode": (
                SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE
            ),
            "lora_initialization_digest": initial_digest,
            "online_pure_t2v_anchor_model_calls_in_training": args.profile != "no_anchor",
            "anchor_rgb_or_latent_used_as_target": False,
            "anchor_route_detached_from_frozen_base": args.profile != "no_anchor",
            "anchor_cross_appearance": args.profile != "no_anchor",
            "route_blocks": list(ROUTE_BLOCKS if args.profile != "no_anchor" else ()),
            "route_operator": args.route_operator,
            "route_transport": (
                qk_transport_for_route_operator(args.route_operator)
                if args.route_operator != "cross_sparse"
                else "cross_sparse"
            ),
            "route_strength": float(args.route_strength),
            "student_route_strength": float(args.route_strength),
            "teacher_route_strength": (
                float(args.teacher_route_strength) if routed_teacher else None
            ),
            "anchor_route_replay_uses_per_capture": anchor_route_replay_uses(
                args.training_objective
            ),
            "anchor_value_pixel_latent_or_spatial_coordinate_copied": False,
            "anchor_donor_cached_fields": (
                ["query", "key"] if qk_only_routed_teacher else None
            ),
            "anchor_donor_value_cached_or_used_by_route": (
                False if qk_only_routed_teacher else None
            ),
            "anchor_donor_hidden_or_attention_output_cached_or_used_by_route": (
                False if qk_only_routed_teacher else None
            ),
            "anchor_donor_rgb_latent_or_absolute_spatial_coordinate_used_by_route": (
                False if qk_only_routed_teacher else None
            ),
            "anchor_qk_spatial_axis_integrated_before_target_application": (
                True if qk_only_routed_teacher else None
            ),
            "anchor_qk_kernel_query_phase0_dc_removed": (
                True if qk_only_routed_teacher else None
            ),
            "anchor_qk_support_uses_phase0_relative_action_noop_contrast": (
                True if qk_only_routed_teacher else None
            ),
            "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel": (
                True if qk_only_routed_teacher else None
            ),
            "anchor_qk_phase0_only_difference_produces_zero_route": (
                True if qk_only_routed_teacher else None
            ),
            "anchor_to_target_appearance_correspondence_used": (
                False if qk_only_routed_teacher else None
            ),
            "target_value_stream_is_sole_routed_content": (
                args.profile != "no_anchor"
                and args.route_operator
                in (
                    "self_temporal_kernel",
                    "self_target_gated_kernel25",
                    "self_correspondence_kernel25",
                    *QK_ONLY_ROUTE_OPERATORS,
                )
            ),
            "target_activity_gated_hard_temporal_route": (
                args.profile != "no_anchor"
                and args.route_operator
                in (
                    "self_target_gated_kernel25",
                    "self_correspondence_kernel25",
                    "self_target_owned_activity_kernel10_v14r2",
                    "self_target_owned_activity_kernel25_v14r2",
                )
            ),
            "phase0_cross_appearance_correspondence_only": (
                args.profile != "no_anchor"
                and args.route_operator == "self_correspondence_kernel25"
            ),
            "source_reconstruction_weight": (
                None
                if qk_only_routed_teacher
                else float(args.source_reconstruction_weight)
            ),
            "source_reconstruction_weight_argument": float(
                args.source_reconstruction_weight
            ),
            "base_replay_scale": (
                base_replay_scale if qk_only_routed_teacher else None
            ),
            "replay_combine_mode": (
                args.replay_combine_mode if qk_only_routed_teacher else None
            ),
            "effective_replay_scale": (
                float(effective_replay_scale) if qk_only_routed_teacher else None
            ),
            "effective_source_replay_scalar_is_reporting_only": (
                True if qk_only_routed_teacher else None
            ),
            "source_reconstruction_prompt_argument": args.source_reconstruction_prompt,
            "source_reconstruction_prompt": (
                "source_caption"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else args.source_reconstruction_prompt
            ),
            "source_reconstruction_is_source_caption_trajectory_replay": (
                args.training_objective in REAL_SOURCE_OBJECTIVES
            ),
            "source_reconstruction_is_identity_prompt": (
                args.training_objective not in REAL_SOURCE_OBJECTIVES
                and args.source_reconstruction_prompt == "identity"
            ),
            "source_variant_argument": (
                "not_applicable"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else args.source_variant
            ),
            "source_variant_schedule": (
                "complete_real_source"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else args.source_variant
            ),
            "real_source_variant_schedule": (
                "complete_real_source"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else None
            ),
            "micro_semantics": (
                "different_seed_and_cross_appearance_donor"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else "same_target_counterfactual_source_variants"
            ),
            "source_counterfactuals_are_deterministic_same_video_transforms": (
                args.training_objective not in REAL_SOURCE_OBJECTIVES
                and args.source_variant == "counterfactual4"
            ),
            "training_objective": args.training_objective,
            "training_interface": args.training_interface,
            "renderer_training_source_name": (
                T2V_TRAINING_SOURCE_NAME
                if args.training_interface == "first_phase_caption_i2v"
                else legacy.TASK_SOURCE_NAME
            ),
            "action_and_source_use_full_scene_captions": (
                args.training_interface == "first_phase_caption_i2v"
            ),
            "source_condition_is_repeated_source_owned_phase_zero": (
                args.training_interface == "first_phase_caption_i2v"
            ),
            "online_anchor_uses_t2v_system_prompt": (
                args.training_interface == "first_phase_caption_i2v"
            ),
            "paired_target_fm_weight": float(args.paired_target_fm_weight),
            "teacher_delta_mode": args.teacher_delta_mode,
            "objective_tensor_scope": "complete_spatiotemporal_video_velocity_field",
            "paired_source_and_target_share_exact_noise_seed_and_timestep": (
                args.training_objective
                in (
                    "paired_delta_fm",
                    "real_source_teacher_delta",
                    *ROUTED_TEACHER_OBJECTIVES,
                )
            ),
            "real_source_action_and_source_share_exact_noisy_tensor": (
                args.training_objective in REAL_SOURCE_OBJECTIVES
            ),
            "real_source_exact_state_binding": (
                "action_transform_vae_state_alias_source_caption_text_only"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else None
            ),
            "real_source_raw_same_seed_exact_required_before_alias": (
                True if args.training_objective in REAL_SOURCE_OBJECTIVES else None
            ),
            "real_source_packed_state_contract": (
                "tokens_c_pt_ph_pw_with_21_phases_verified_from_unpacked_clean_shape"
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else None
            ),
            "anchor_action_and_noop_share_exact_noisy_tensor": (
                args.profile == "action_noop"
            ),
            "anchor_and_real_source_noise_deliberately_unbound": routed_teacher,
            "paired_delta_has_gradients_through_both_model_queries": (
                args.training_objective
                in (
                    "paired_delta_fm",
                    "real_source_teacher_delta",
                    *ROUTED_TEACHER_OBJECTIVES,
                )
                and not (
                    qk_only_routed_teacher
                    and args.routed_teacher_mode == "same_action_route_only"
                )
            ),
            "student_clean_target_is_complete_real_source": (
                args.training_objective in REAL_SOURCE_OBJECTIVES
            ),
            "synthetic_clean_target_flow_matching_weight": (
                0.0
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else float(args.paired_target_fm_weight)
            ),
            "t2v_teacher_is_same_noisy_state_action_minus_noop_full_tensor": (
                args.training_objective == "real_source_teacher_delta"
            ),
            "target_coordinate_routed_teacher": routed_teacher,
            "routed_teacher_mode": (
                args.routed_teacher_mode if routed_teacher else None
            ),
            "target_owned_qk_route_v14r2": qk_only_routed_teacher,
            "correspondence_route_enabled": False if qk_only_routed_teacher else None,
            "anchor_only_captures_route": routed_teacher,
            "anchor_model_velocity_used_as_supervision": (
                args.training_objective == "real_source_teacher_delta"
            ),
            "routed_teacher_action_and_route_off_share_real_source_noisy_state": (
                routed_teacher
            ),
            "routed_teacher_cross_caption_source_branch": (
                routed_teacher
                and args.routed_teacher_mode == "cross_caption_two_sided"
            ),
            "student_route_off_branch_stop_gradient": (
                qk_only_routed_teacher
                and args.routed_teacher_mode == "same_action_route_only"
            ),
            "action_objective_backpropagates_only_routed_student_query": (
                qk_only_routed_teacher
                and args.routed_teacher_mode == "same_action_route_only"
            ),
            "routed_teacher_action_adapter_disabled": routed_teacher,
            "routed_teacher_route_off_adapter_disabled": routed_teacher,
            "routed_teacher_action_route_enabled": routed_teacher,
            "routed_teacher_route_off_route_enabled": False,
            "routed_student_action_route_enabled": routed_teacher,
            "routed_student_route_off_route_enabled": False,
            "action_and_replay_gradients_measured_after_separate_global_average": (
                qk_only_routed_teacher
            ),
            "component_gradient_merge": (
                "global_avg_action_plus_global_avg_weighted_replay"
                if qk_only_routed_teacher
                else None
            ),
            "actual_optimizer_parameter_displacement_audited_every_step": (
                True if qk_only_routed_teacher else None
            ),
            "actual_optimizer_update_compared_to_unmerged_global_components": (
                True if qk_only_routed_teacher else None
            ),
            "teacher_tensor_reduced_to_low_dimensional_statistic": False,
            "standard_full_video_flow_matching_is_primary_objective": (
                args.training_objective == "target_fm"
            ),
            "dynaedit_sga_implemented_in_training": False,
            "dynaedit_anc_implemented_in_training": False,
            "dynaedit_sga_anc_reserved_for_decode_solver": True,
            "true_training_memory_fraction_strictly_above_half": bool(memory["passed"]),
            "training_memory_gate_capture_phase": memory.get("capture_phase"),
            "actual_update_audit_allocations_excluded_from_training_memory_gate": (
                memory.get("actual_update_audit_allocations_excluded")
            ),
            "dummy_or_padding_allocations": False,
        },
    }


def save_checkpoint(
    *,
    output: Path,
    step: int,
    renderer: Any,
    receipt: Mapping[str, Any],
    rank: int,
    dist: Any,
) -> None:
    dist.barrier()
    if rank == 0:
        root = output / f"checkpoint-{step:08d}"
        root.mkdir(parents=True, exist_ok=False)
        adapter = root / "adapter"
        renderer.save_pretrained(adapter, safe_serialization=True)
        adapter_model = adapter / "adapter_model.safetensors"
        adapter_config = adapter / "adapter_config.json"
        if not all(
            path.is_file() and not path.is_symlink()
            for path in (adapter_model, adapter_config)
        ):
            fail("saved adapter artifact closure differs")
        receipt_to_write = dict(receipt)
        receipt_to_write["adapter_model_sha256"] = pairs.file_sha256(adapter_model)
        receipt_to_write["adapter_config_sha256"] = pairs.file_sha256(adapter_config)
        (root / "receipt.json").write_text(
            json.dumps(receipt_to_write, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="ascii",
        )
    dist.barrier()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    bernini_root, veomni_root, bernini_revision, veomni_revision = legacy.validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler

    contract = legacy.distributed_contract()
    if contract.world_size != 4:
        fail("online-anchor training requires exact SP4")
    device, _ = legacy.initialise_distributed(contract)
    init_parallel_state(ulysses_size=4)
    legacy.seed_same_sample(args.seed)

    pair_manifest = Path(args.pair_manifest).resolve(strict=True)
    manifest, rows = pairs.load_manifest(pair_manifest)
    registry = row_registry(rows)
    caption_registry = None
    if args.training_interface == "first_phase_caption_i2v":
        authoring_path = Path(args.authoring).resolve(strict=True)
        caption_registry = load_caption_registry(authoring_path)
        if set(caption_registry) != set(registry):
            fail("caption-I2V authoring and pair manifest rows differ")
    real_source_registry = None
    real_source_manifest = None
    if args.training_objective in REAL_SOURCE_OBJECTIVES:
        real_source_manifest = Path(args.real_source_manifest).resolve(strict=True)
        real_source_registry = load_real_source_registry(
            real_source_manifest, args.real_source_manifest_sha256
        )
    output = Path(args.output).resolve()
    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    with data.serialized_model_load():
        renderer = BerniniRendererModel(config)
        renderer.requires_grad_(False)
        renderer.t5_text_encoder.eval()
        targets = select_lora_target_names(renderer)
        if (
            not isinstance(targets, tuple)
            or len(targets) != LORA_TARGET_MODULE_COUNT
            or tuple(sorted(targets)) != targets
            or len(set(targets)) != len(targets)
        ):
            fail("LoRA target-module closure differs")
        renderer = get_peft_model(
            renderer,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.0,
                bias="none",
                target_modules=targets,
            ),
        )
        activation_blocks = install_activation_checkpointing(renderer)
        renderer.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    named = tuple(
        (name, parameter)
        for name, parameter in renderer.named_parameters()
        if parameter.requires_grad and legacy.is_lora_parameter_name(name)
    )
    if (
        len(named) != LORA_TRAINABLE_TENSOR_COUNT
        or sum(int(parameter.numel()) for _, parameter in named) != LORA_PARAMETERS
        or tuple(activation_blocks) != CHECKPOINT_BLOCKS
        or {id(parameter) for _, parameter in named}
        != {id(parameter) for parameter in renderer.parameters() if parameter.requires_grad}
    ):
        fail("rank-256 LoRA closure differs")
    initial_digest = legacy.synchronize_trainable_parameters(named, source_rank=0)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**legacy.noise_scheduler_kwargs())
    transform = data.build_transform(
        tokenizer=tokenizer,
        rope=rope,
        mean=mean,
        std=std,
        scheduler=scheduler,
        device=device,
        source_name=(
            T2V_TRAINING_SOURCE_NAME
            if args.training_interface == "first_phase_caption_i2v"
            else legacy.TASK_SOURCE_NAME
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    renderer.eval()
    renderer.t5_text_encoder.eval()

    transformer = renderer.get_base_model().diff_dec.transformer
    if args.route_operator == "cross_sparse":
        cache: Any = cross.AnchorCrossAttentionCache(ROUTE_BLOCKS)
        patch: Any = cross.AnchorCrossAttentionPatchHandle(transformer, cache)
    else:
        cache = qk.AnchorQKCacheBank(ROUTE_BLOCKS)
        patch = qk.AnchorQKPatchHandle(transformer, cache)
    if args.profile != "no_anchor":
        patch.install()
    memory = None
    last_loss = 0.0
    last_grad_norm = 0.0
    last_gradient: Mapping[str, Any] = {}
    last_action_gradient: Mapping[str, Any] = {}
    last_replay_gradient: Mapping[str, Any] = {}
    last_gradient_interaction: Mapping[str, Any] = {}
    last_actual_optimizer_update: Mapping[str, Any] = {}
    last_source_absorption: Mapping[str, Any] = {}
    last_route_off_absolute_anchor: Mapping[str, Any] = {
        "applicable": False,
        "reason": "same_action_route_off_absolute_anchor_disabled",
    }
    torch.cuda.reset_peak_memory_stats(device)
    try:
        for global_step in range(args.max_steps):
            optimizer.zero_grad(set_to_none=True)
            action_losses = []
            target_fm_losses = []
            paired_delta_losses = []
            reconstruction_losses = []
            route_off_absolute_anchor_losses = []
            records = []
            deferred_replays: list[Any] = []
            source_absorption_values: list[Mapping[str, Any]] = []
            qk_only_objective = (
                args.training_objective
                == "real_source_target_owned_routed_teacher_delta_v14r2"
            )
            same_action_route_only = (
                qk_only_objective
                and args.routed_teacher_mode == "same_action_route_only"
            )
            update_variants = (
                ("complete_real_source", "complete_real_source")
                if args.training_objective in REAL_SOURCE_OBJECTIVES
                else source_variants_for_update(
                    source_variant=args.source_variant,
                    global_step=global_step,
                    row_count=len(rows),
                )
            )
            for micro in range(args.micro_records):
                record_index = global_step * args.micro_records + micro
                # Real-source micros are two distinct deterministic seeds and
                # cross-appearance donors over one complete source trajectory;
                # only legacy synthetic pairs use noop/incomplete variants.
                row_index = global_step % len(rows)
                row = rows[row_index]
                variant = update_variants[micro]
                seed = legacy.step_seed(args.seed, record_index, row_index)
                paired_source = None
                source_prediction_reference = None
                source_velocity_target = None
                teacher_action_velocity = None
                teacher_noop_velocity = None
                routed_teacher_action = None
                routed_teacher_source = None
                student_route_off_prediction = None
                if args.training_objective in REAL_SOURCE_OBJECTIVES:
                    action_record, paired_source = build_real_source_paired_records(
                        anchor_row=row,
                        real_sources=real_source_registry,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                    )
                    variant = "complete_real_source"
                elif args.training_interface == "first_phase_caption_i2v":
                    action_record, paired_source = build_i2v_paired_records(
                        row=row,
                        variant=variant,
                        captions=caption_registry,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                    )
                else:
                    action_record = build_action_record(
                        row=row,
                        variant=variant,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                    )
                if args.training_objective == "paired_delta_fm" and paired_source is None:
                    paired_source = build_source_reconstruction_record(
                        action_row=row,
                        variant=variant,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        prompt_mode="identity",
                    )
                if (
                    args.training_objective
                    in ("paired_delta_fm", "real_source_teacher_delta")
                    or (
                        args.training_objective in ROUTED_TEACHER_OBJECTIVES
                        and not same_action_route_only
                    )
                ):
                    action_t = float(
                        action_record["batch"]["timesteps"]
                        .float()
                        .reshape(-1)[0]
                        .item()
                    )
                    source_t = float(
                        paired_source["batch"]["timesteps"]
                        .float()
                        .reshape(-1)[0]
                        .item()
                    )
                    if action_t != source_t:
                        fail("paired source/target did not share a timestep")
                    with torch.no_grad(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        (
                            source_prediction_reference,
                            source_velocity_target,
                        ) = velocity_prediction_and_target(renderer, paired_source)
                selected_donor = None
                donor_index = record_index % 3
                if args.profile != "no_anchor":
                    selected_donor = donor_row(
                        row, registry, donor_index=donor_index
                    )
                    anchor_action, anchor_contrast, anchor_shape = build_anchor_batches(
                        target_row=row,
                        donor=selected_donor,
                        profile=args.profile,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed,
                        captions=caption_registry,
                    )
                    if anchor_shape != action_record["shape"]:
                        fail("cross-appearance anchor/target latent geometry differs")
                    replay_uses = anchor_route_replay_uses(
                        args.training_objective
                    )
                    capture_result = capture_anchor_route(
                        renderer=renderer,
                        cache=cache,
                        action_batch=anchor_action,
                        contrast_batch=anchor_contrast,
                        shape=anchor_shape,
                        invocation_index=record_index,
                        candidate_index=donor_index,
                        rank=contract.rank,
                        world_size=contract.world_size,
                        strength=float(args.teacher_route_strength),
                        route_operator=args.route_operator,
                        replay_uses=replay_uses,
                        capture_only=(args.training_objective in ROUTED_TEACHER_OBJECTIVES),
                    )
                    if args.training_objective in ROUTED_TEACHER_OBJECTIVES:
                        if capture_result is not None:
                            fail("routed teacher unexpectedly exposed donor velocity")
                        require_same_real_source_noisy_state(
                            action_record["batch"],
                            paired_source["batch"],
                            spatial_shape=action_record["shape"],
                        )
                        teacher_invocation, teacher_invocation_context = replay_invocation(
                            cache=cache,
                            invocation_index=record_index,
                            candidate_index=donor_index,
                            rank=contract.rank,
                            world_size=contract.world_size,
                            strength=float(args.teacher_route_strength),
                            route_operator=args.route_operator,
                            replay_uses=replay_uses,
                        )
                        disabled = getattr(renderer, "disable_adapter", None)
                        if not callable(disabled):
                            fail("PEFT adapter-disable context is unavailable")
                        with torch.no_grad(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ), disabled():
                            # same_action uses one target caption/state on both
                            # sides; the cross-caption control uses the paired
                            # source caption.  Route-off never consumes cache.
                            route_off_record = (
                                action_record
                                if same_action_route_only
                                else paired_source
                            )
                            routed_teacher_source, _ = velocity_prediction_and_target(
                                renderer, route_off_record
                            )
                            with teacher_invocation_context(teacher_invocation):
                                routed_teacher_action, _ = velocity_prediction_and_target(
                                    renderer, action_record
                                )
                    else:
                        if capture_result is None:
                            fail("legacy teacher capture omitted its velocity")
                        (
                            teacher_action_velocity,
                            teacher_noop_velocity,
                        ) = capture_result
                    invocation, invocation_context = replay_invocation(
                        cache=cache,
                        invocation_index=record_index,
                        candidate_index=donor_index,
                        rank=contract.rank,
                        world_size=contract.world_size,
                        strength=float(args.route_strength),
                        route_operator=args.route_operator,
                        replay_uses=replay_uses,
                    )
                    if same_action_route_only:
                        with torch.no_grad(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16
                        ):
                            student_route_off_prediction, _ = (
                                velocity_prediction_and_target(renderer, action_record)
                            )
                    with invocation_context(invocation), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        if args.training_objective == "target_fm":
                            action_loss = velocity_loss(renderer, action_record)
                            target_fm_loss = action_loss
                            paired_action_loss = None
                        elif args.training_objective == "real_source_teacher_delta":
                            (
                                action_prediction,
                                action_velocity_target,
                            ) = velocity_prediction_and_target(renderer, action_record)
                            paired_action_loss = real_source_teacher_delta_loss(
                                action_prediction=action_prediction,
                                source_prediction=source_prediction_reference,
                                teacher_action=teacher_action_velocity,
                                teacher_noop=teacher_noop_velocity,
                                mode=args.teacher_delta_mode,
                                name="real-source teacher delta action-side loss",
                            )
                            target_fm_loss = action_velocity_target.new_zeros(())
                            action_loss = paired_action_loss
                        elif args.training_objective in ROUTED_TEACHER_OBJECTIVES:
                            (
                                action_prediction,
                                action_velocity_target,
                            ) = velocity_prediction_and_target(renderer, action_record)
                            paired_action_loss = real_source_routed_teacher_delta_loss(
                                action_prediction=action_prediction,
                                source_prediction=(
                                    student_route_off_prediction
                                    if same_action_route_only
                                    else source_prediction_reference
                                ),
                                teacher_action=routed_teacher_action,
                                teacher_source=routed_teacher_source,
                                mode=args.teacher_delta_mode,
                                name="real-source routed teacher action-side loss",
                            )
                            target_fm_loss = action_velocity_target.new_zeros(())
                            action_loss = paired_action_loss
                        else:
                            (
                                action_prediction,
                                action_velocity_target,
                            ) = velocity_prediction_and_target(renderer, action_record)
                            paired_action_loss = paired_delta_loss(
                                action_prediction=action_prediction,
                                source_prediction=source_prediction_reference,
                                action_target=action_velocity_target,
                                source_target=source_velocity_target,
                                name="paired delta action-side loss",
                            )
                            target_fm_loss = finite_mse(
                                action_prediction,
                                action_velocity_target,
                                name="paired auxiliary target FM loss",
                            )
                            action_loss = paired_action_loss + float(
                                args.paired_target_fm_weight
                            ) * target_fm_loss
                    cache.assert_empty()
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if args.training_objective == "target_fm":
                            action_loss = velocity_loss(renderer, action_record)
                            target_fm_loss = action_loss
                            paired_action_loss = None
                        else:
                            (
                                action_prediction,
                                action_velocity_target,
                            ) = velocity_prediction_and_target(renderer, action_record)
                            paired_action_loss = paired_delta_loss(
                                action_prediction=action_prediction,
                                source_prediction=source_prediction_reference,
                                action_target=action_velocity_target,
                                source_target=source_velocity_target,
                                name="paired delta action-side loss",
                            )
                            target_fm_loss = finite_mse(
                                action_prediction,
                                action_velocity_target,
                                name="paired auxiliary target FM loss",
                            )
                            action_loss = paired_action_loss + float(
                                args.paired_target_fm_weight
                            ) * target_fm_loss
                scaled_action = action_loss / float(args.micro_records)
                scaled_action.backward()
                action_losses.append(action_loss.detach())
                target_fm_losses.append(target_fm_loss.detach())
                if paired_action_loss is not None:
                    paired_delta_losses.append(paired_action_loss.detach())
                route_off_absolute_anchor_loss = None
                if requires_same_action_route_off_absolute_anchor(
                    same_action_route_only=same_action_route_only
                ):
                    # D is deliberately distinct from C: the scalar delta above
                    # retains the legacy route-on-only Jacobian.  Recompute one
                    # route-off graph only for an absolute same-state FM spring
                    # toward the already-materialized frozen-base teacher.
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        route_off_absolute_prediction, _ = (
                            velocity_prediction_and_target(renderer, action_record)
                        )
                        route_off_absolute_anchor_loss = (
                            same_action_route_off_absolute_anchor_loss(
                                student_route_off_prediction=(
                                    route_off_absolute_prediction
                                ),
                                frozen_route_off_teacher=routed_teacher_source,
                            )
                        )
                    (
                        float(SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT)
                        * route_off_absolute_anchor_loss
                        / float(args.micro_records)
                    ).backward()
                    route_off_absolute_anchor_losses.append(
                        route_off_absolute_anchor_loss.detach()
                    )
                if requires_sequential_source_side_backward(
                    paired_action_loss=paired_action_loss,
                    same_action_route_only=same_action_route_only,
                ):
                    # Recover the source-side gradient of the same scalar loss
                    # without retaining two 30-block graphs simultaneously.
                    # Combined with the action-side backward above, this is
                    # exactly the two-sided gradient of the velocity delta.
                    source_side_record = sequential_source_side_record(
                        same_action_route_only=same_action_route_only,
                        action_record=action_record,
                        paired_source=paired_source,
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        (
                            source_prediction_graph,
                            source_velocity_target_graph,
                        ) = velocity_prediction_and_target(
                            renderer, source_side_record
                        )
                        if args.training_objective == "real_source_teacher_delta":
                            paired_source_loss = real_source_teacher_delta_loss(
                                action_prediction=action_prediction.detach(),
                                source_prediction=source_prediction_graph,
                                teacher_action=teacher_action_velocity,
                                teacher_noop=teacher_noop_velocity,
                                mode=args.teacher_delta_mode,
                                name="real-source teacher delta source-side loss",
                            )
                        elif args.training_objective in ROUTED_TEACHER_OBJECTIVES:
                            paired_source_loss = real_source_routed_teacher_delta_loss(
                                action_prediction=action_prediction.detach(),
                                source_prediction=source_prediction_graph,
                                teacher_action=routed_teacher_action,
                                teacher_source=routed_teacher_source,
                                mode=args.teacher_delta_mode,
                                name="real-source routed teacher source-side loss",
                            )
                        else:
                            paired_source_loss = paired_delta_loss(
                                action_prediction=action_prediction.detach(),
                                source_prediction=source_prediction_graph,
                                action_target=action_velocity_target,
                                source_target=source_velocity_target_graph,
                                name="paired delta source-side loss",
                            )
                    (paired_source_loss / float(args.micro_records)).backward()
                    if qk_only_objective:
                        source_absorption_values.append(
                            source_absorption_diagnostic(
                                action_prediction=action_prediction,
                                source_prediction=source_prediction_graph,
                                source_velocity_target=source_velocity_target_graph,
                            )
                        )

                reconstruction_prebind_diagnostic = None
                if args.training_objective in REAL_SOURCE_OBJECTIVES:
                    reconstruction_action, reconstruction = build_real_source_paired_records(
                        anchor_row=row,
                        real_sources=real_source_registry,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed + 1_000_003,
                    )
                    reconstruction_prebind_diagnostic = reconstruction_action[
                        "real_source_prebind_state_diagnostic"
                    ]
                elif args.training_interface == "first_phase_caption_i2v":
                    _, reconstruction = build_i2v_paired_records(
                        row=row,
                        variant=variant,
                        captions=caption_registry,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed + 1_000_003,
                    )
                else:
                    reconstruction = build_source_reconstruction_record(
                        action_row=row,
                        variant=variant,
                        transform=transform,
                        mean=mean,
                        std=std,
                        seed=seed + 1_000_003,
                        prompt_mode=args.source_reconstruction_prompt,
                    )
                reconstruction_loss = None
                if qk_only_objective:
                    deferred_replays.append((reconstruction, len(records)))
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        reconstruction_loss = velocity_loss(renderer, reconstruction)
                        scaled_reconstruction = (
                            float(args.source_reconstruction_weight)
                            * reconstruction_loss
                            / float(args.micro_records)
                        )
                    scaled_reconstruction.backward()
                    reconstruction_losses.append(reconstruction_loss.detach())
                records.append(
                    {
                        "iid": str(action_record["iid"]),
                        "source_variant": variant,
                        "donor_iid": (
                            str(selected_donor["iid"]) if selected_donor else None
                        ),
                        "timestep": action_record["timestep"],
                        "action_fm": float(action_loss.detach().item()),
                        "target_fm": float(target_fm_loss.detach().item()),
                        "paired_delta_fm": (
                            float(paired_action_loss.detach().item())
                            if paired_action_loss is not None
                            else None
                        ),
                        "real_source_teacher_delta": (
                            float(paired_action_loss.detach().item())
                            if args.training_objective
                            == "real_source_teacher_delta"
                            else None
                        ),
                        "real_source_routed_teacher_delta": (
                            float(paired_action_loss.detach().item())
                            if args.training_objective in ROUTED_TEACHER_OBJECTIVES
                            else None
                        ),
                        "real_source_target_owned_routed_teacher_delta_v14r2": (
                            float(paired_action_loss.detach().item())
                            if args.training_objective
                            == "real_source_target_owned_routed_teacher_delta_v14r2"
                            else None
                        ),
                        "same_state_route_off_absolute_anchor_fm": (
                            float(route_off_absolute_anchor_loss.detach().item())
                            if route_off_absolute_anchor_loss is not None
                            else None
                        ),
                        "same_state_route_off_absolute_anchor_weight": (
                            float(SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT)
                            if route_off_absolute_anchor_loss is not None
                            else None
                        ),
                        "source_reconstruction_fm": (
                            float(reconstruction_loss.detach().item())
                            if reconstruction_loss is not None
                            else None
                        ),
                        "source_caption_trajectory_replay_fm": (
                            float(reconstruction_loss.detach().item())
                            if args.training_objective in REAL_SOURCE_OBJECTIVES
                            and reconstruction_loss is not None
                            else None
                        ),
                        "weighted_source_caption_trajectory_replay_fm": (
                            float(args.source_reconstruction_weight)
                            * float(reconstruction_loss.detach().item())
                            if args.training_objective in REAL_SOURCE_OBJECTIVES
                            and reconstruction_loss is not None
                            and not qk_only_objective
                            else None
                        ),
                        "base_replay_scale": (
                            float(args.source_reconstruction_weight)
                            if qk_only_objective
                            else None
                        ),
                        "base_source_replay_scalar_diagnostic": None,
                        "effective_replay_scale": None,
                        "effective_source_replay_scalar_for_reporting": None,
                        "real_source_action_prebind_state_diagnostic": (
                            action_record.get(
                                "real_source_prebind_state_diagnostic"
                            )
                            if args.training_objective in REAL_SOURCE_OBJECTIVES
                            else None
                        ),
                        "real_source_replay_prebind_state_diagnostic": (
                            reconstruction_prebind_diagnostic
                            if args.training_objective in REAL_SOURCE_OBJECTIVES
                            else None
                        ),
                    }
                )

            effective_replay_scale = float(args.source_reconstruction_weight)
            if qk_only_objective:
                if len(deferred_replays) != args.micro_records:
                    fail("v14r2 did not defer exactly two source-caption replays")
                legacy.all_reduce_lora_gradients(named)
                last_action_gradient = component_gradient_probe(
                    named, epsilon=COMPONENT_GRADIENT_EPSILON
                )
                validate_v14r2_component_coverage(
                    last_action_gradient,
                    step=global_step + 1,
                    component="action",
                )
                action_gradients = clone_component_gradients(named)
                optimizer.zero_grad(set_to_none=True)

                for reconstruction, record_position in deferred_replays:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        reconstruction_loss = velocity_loss(renderer, reconstruction)
                    # This is raw source-caption FM.  Its audited scale is
                    # applied only after its independent global average.
                    (reconstruction_loss / float(args.micro_records)).backward()
                    reconstruction_losses.append(reconstruction_loss.detach())
                    raw_value = float(reconstruction_loss.detach().item())
                    records[record_position]["source_reconstruction_fm"] = raw_value
                    records[record_position][
                        "source_caption_trajectory_replay_fm"
                    ] = raw_value

                legacy.all_reduce_lora_gradients(named)
                last_replay_gradient = component_gradient_probe(
                    named, epsilon=COMPONENT_GRADIENT_EPSILON
                )
                validate_v14r2_component_coverage(
                    last_replay_gradient,
                    step=global_step + 1,
                    component="raw_replay",
                )
                if memory is None:
                    # Capture the genuine two-component training peak before
                    # allocating the raw-replay copy and full-adapter parameter
                    # snapshot that exist only for the post-Adam audit. CUDA
                    # peak counters cannot be corrected after those clones.
                    memory = dict(
                        pairs.memory_receipt(device, args.micro_records)
                    )
                    memory["capture_phase"] = (
                        "after_two_real_component_backwards_before_actual_update_audit_clones"
                    )
                    memory["actual_update_audit_allocations_excluded"] = True
                    if not bool(memory["passed"]):
                        fail(
                            "real online-anchor training peak reserved memory is not strictly above 50%"
                        )
                # merge_component_gradients updates parameter.grad in place.
                # Retain the independently all-reduced raw replay gradient so
                # the post-AdamW parameter displacement can be audited against
                # the two original objectives, not reconstructed from the
                # clipped/merged gradient.
                raw_replay_gradients = clone_component_gradients(named)
                last_gradient_interaction = merge_component_gradients(
                    named,
                    action_gradients,
                    replay_combine_mode=args.replay_combine_mode,
                    base_replay_scale=float(args.source_reconstruction_weight),
                    diagnostic_only=bool(args.gradient_diagnostic_only),
                )
                effective_replay_scale = float(
                    last_gradient_interaction["effective_replay_scale"]
                )
                for record in records:
                    raw_replay_scalar = float(
                        record["source_caption_trajectory_replay_fm"]
                    )
                    record["effective_replay_scale"] = effective_replay_scale
                    record[
                        "base_source_replay_scalar_diagnostic"
                    ] = float(args.source_reconstruction_weight) * raw_replay_scalar
                    record[
                        "effective_source_replay_scalar_for_reporting"
                    ] = effective_replay_scale * raw_replay_scalar
                    record[
                        "weighted_source_caption_trajectory_replay_fm"
                    ] = None
                last_gradient = gradient_coverage(named)
                if global_step >= 1 and (
                    last_gradient["tensor_count"] != LORA_TRAINABLE_TENSOR_COUNT
                    or last_gradient["nonzero_tensor_count"]
                    != LORA_TRAINABLE_TENSOR_COUNT
                ):
                    fail("v14r2 step-2+ combined gradient coverage is incomplete")
                last_grad_norm = float(
                    last_gradient_interaction["combined_l2_norm_fp64"]
                )
                if source_absorption_values:
                    defined = [
                        value for value in source_absorption_values if value["defined"]
                    ]
                    last_source_absorption = {
                        "applicable": True,
                        "micro_count": len(source_absorption_values),
                        "defined_micro_count": len(defined),
                        "mean_q_action": (
                            sum(float(value["q_action"]) for value in defined)
                            / len(defined)
                            if defined
                            else None
                        ),
                        "mean_q_source": (
                            sum(float(value["q_source"]) for value in defined)
                            / len(defined)
                            if defined
                            else None
                        ),
                        "mean_q_sum": (
                            sum(float(value["q_sum"]) for value in defined)
                            / len(defined)
                            if defined
                            else None
                        ),
                    }
                else:
                    last_source_absorption = {
                        "applicable": False,
                        "reason": "same_action_route_only_has_no_cross_caption_action_delta",
                    }
            else:
                last_grad_norm = legacy.all_reduce_lora_gradients(named)
                last_gradient = gradient_coverage(named)
                last_action_gradient = {}
                last_replay_gradient = {}
                last_gradient_interaction = {}
                last_actual_optimizer_update = {}
                last_source_absorption = {}
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named], float(args.max_grad_norm)
            )
            parameter_values_before_step = (
                clone_trainable_parameter_values(named)
                if qk_only_objective
                else ()
            )
            optimizer.step()
            step = global_step + 1
            if qk_only_objective:
                last_actual_optimizer_update = actual_optimizer_update_probe(
                    named,
                    parameter_values_before_step,
                    action_gradients,
                    raw_replay_gradients,
                    replay_combine_mode=args.replay_combine_mode,
                    step=step,
                )
                # These full-adapter clones are per-step audit state.  Release
                # them before decoding/logging and never accumulate history in
                # GPU memory; checkpoints retain only the scalar probe.
                del parameter_values_before_step
                del raw_replay_gradients
                del action_gradients
            mean_action_objective = torch.stack(action_losses).mean()
            mean_source_reconstruction = torch.stack(reconstruction_losses).mean()
            mean_route_off_absolute_anchor = (
                torch.stack(route_off_absolute_anchor_losses).mean()
                if route_off_absolute_anchor_losses
                else None
            )
            if mean_route_off_absolute_anchor is not None:
                last_route_off_absolute_anchor = {
                    "applicable": True,
                    "mode": SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE,
                    "micro_count": len(route_off_absolute_anchor_losses),
                    "record": "same_action_record_same_noisy_state_same_timestep",
                    "teacher": "adapter_disabled_route_off_routed_teacher_source",
                    "student_delta_gradient_mode": (
                        SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE
                    ),
                    "sequential_backward": True,
                    "simultaneous_two_30_block_graph_retention": False,
                    "weight": float(
                        SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT
                    ),
                    "mean_fm": float(mean_route_off_absolute_anchor.item()),
                    "weighted_mean_fm": float(
                        SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT
                        * mean_route_off_absolute_anchor.item()
                    ),
                }
            else:
                last_route_off_absolute_anchor = {
                    "applicable": False,
                    "reason": "same_action_route_off_absolute_anchor_disabled",
                }
            weighted_source_reconstruction = (
                effective_replay_scale * mean_source_reconstruction
            )
            weighted_route_off_absolute_anchor = (
                float(SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT)
                * mean_route_off_absolute_anchor
                if mean_route_off_absolute_anchor is not None
                else mean_action_objective.new_zeros(())
            )
            local_loss = (
                mean_action_objective
                + weighted_source_reconstruction
                + weighted_route_off_absolute_anchor
            )
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            last_loss = float(local_loss.div(contract.world_size).item())
            if memory is None:
                memory = pairs.memory_receipt(device, args.micro_records)
                if not bool(memory["passed"]):
                    fail(
                        "real online-anchor training peak reserved memory is not strictly above 50%"
                    )
            if contract.rank == 0 and global_step == 0:
                output.mkdir(parents=True, exist_ok=False)
            if contract.rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "profile": args.profile,
                            "training_objective": args.training_objective,
                            "loss": None if qk_only_objective else last_loss,
                            "reporting_scalar": (
                                last_loss if qk_only_objective else None
                            ),
                            "reporting_scalar_is_not_a_joint_backpropagated_objective": (
                                True if qk_only_objective else None
                            ),
                            "mean_action_objective": float(
                                mean_action_objective.item()
                            ),
                            "mean_source_caption_trajectory_replay_fm": (
                                float(mean_source_reconstruction.item())
                                if args.training_objective in REAL_SOURCE_OBJECTIVES
                                else None
                            ),
                            "mean_weighted_source_caption_trajectory_replay_fm": (
                                float(weighted_source_reconstruction.item())
                                if args.training_objective in REAL_SOURCE_OBJECTIVES
                                and not qk_only_objective
                                else None
                            ),
                            "source_caption_replay_weighted_fraction": (
                                float(
                                    weighted_source_reconstruction.div(
                                        mean_action_objective
                                        + weighted_source_reconstruction
                                    ).item()
                                )
                                if args.training_objective in REAL_SOURCE_OBJECTIVES
                                and not qk_only_objective
                                and float(local_loss.item()) > 0.0
                                else None
                            ),
                            "base_replay_scale": (
                                float(args.source_reconstruction_weight)
                                if qk_only_objective
                                else None
                            ),
                            "effective_replay_scale": (
                                effective_replay_scale
                                if qk_only_objective
                                else None
                            ),
                            "mean_base_source_replay_scalar_diagnostic": (
                                float(
                                    float(args.source_reconstruction_weight)
                                    * mean_source_reconstruction
                                )
                                if qk_only_objective
                                else None
                            ),
                            "mean_effective_source_replay_scalar_for_reporting": (
                                float(weighted_source_reconstruction.item())
                                if qk_only_objective
                                else None
                            ),
                            "effective_source_replay_reporting_fraction": (
                                float(
                                    weighted_source_reconstruction.div(
                                        mean_action_objective
                                        + weighted_source_reconstruction
                                    ).item()
                                )
                                if qk_only_objective
                                and float(local_loss.item()) > 0.0
                                else None
                            ),
                            "mean_target_fm": float(
                                torch.stack(target_fm_losses).mean().item()
                            ),
                            "mean_paired_delta_fm": (
                                float(torch.stack(paired_delta_losses).mean().item())
                                if paired_delta_losses
                                else None
                            ),
                            "route_off_absolute_anchor_diagnostic": (
                                last_route_off_absolute_anchor
                            ),
                            "preclip_grad_norm": last_grad_norm,
                            "gradient_coverage": last_gradient,
                            "action_only_gradient": last_action_gradient,
                            "raw_replay_only_gradient": last_replay_gradient,
                            "component_gradient_interaction": (
                                last_gradient_interaction
                            ),
                            "actual_optimizer_update_probe": (
                                last_actual_optimizer_update
                            ),
                            "source_absorption_diagnostic": last_source_absorption,
                            "memory_gate": memory,
                            "records": records,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step in SAVE_STEPS or step == args.max_steps:
                receipt = checkpoint_receipt(
                    args=args,
                    step=step,
                    loss=last_loss,
                    action_objective=float(mean_action_objective.item()),
                    source_reconstruction=float(mean_source_reconstruction.item()),
                    effective_replay_scale=effective_replay_scale,
                    grad_norm=last_grad_norm,
                    memory=memory,
                    targets=targets,
                    initial_digest=initial_digest,
                    cache=cache,
                    bernini_revision=bernini_revision,
                    veomni_revision=veomni_revision,
                    pair_manifest=pair_manifest,
                    gradient=last_gradient,
                    action_gradient=last_action_gradient,
                    replay_gradient=last_replay_gradient,
                    gradient_interaction=last_gradient_interaction,
                    actual_optimizer_update=last_actual_optimizer_update,
                    source_absorption=last_source_absorption,
                    route_off_absolute_anchor=(
                        last_route_off_absolute_anchor
                    ),
                    real_source_prebind_state=(
                        {
                            "schema_version": (
                                "bernini-real-source-prebind-packed-update-v1"
                            ),
                            "micro_count": len(records),
                            "all_raw_same_seed_state_exact": all(
                                record[
                                    "real_source_action_prebind_state_diagnostic"
                                ]["raw_same_seed_state_exact"]
                                and record[
                                    "real_source_replay_prebind_state_diagnostic"
                                ]["raw_same_seed_state_exact"]
                                for record in records
                            ),
                            "action_branches": [
                                record[
                                    "real_source_action_prebind_state_diagnostic"
                                ]
                                for record in records
                            ],
                            "replay_branches": [
                                record[
                                    "real_source_replay_prebind_state_diagnostic"
                                ]
                                for record in records
                            ],
                        }
                        if args.training_objective in REAL_SOURCE_OBJECTIVES
                        else None
                    ),
                )
                save_checkpoint(
                    output=output,
                    step=step,
                    renderer=renderer,
                    receipt=receipt,
                    rank=contract.rank,
                    dist=dist,
                )
        if contract.rank == 0:
            (output / "TRAINING_COMPLETE").write_text("complete\n", encoding="ascii")
        dist.barrier()
    finally:
        if args.profile != "no_anchor":
            patch.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

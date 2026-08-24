#!/usr/bin/env python3
"""Fit the K=2 Bernini EGNTC controller on two true 40-step rollouts.

This runner is deliberately narrower than a video-model trainer.  WORLD8 is
partitioned as two independent Ulysses-SP4 groups: ranks 0--3 process support
one while ranks 4--7 process support two.  In each group the frozen Bernini-R
1.3B generator executes the official 40-step shift-5 UniPC sampler with the
three same-state negative/action/semantic-noop forwards supplied by
``tri_branch_unipc`` (120 transformer forwards per support and round).
Each step reaches one original UniPC scheduler update; no custom integrator is
present.

Round one returns Bernini's official action clean field while capturing a
detached trajectory cache.  The 36D EGNTC parameters are fitted offline.
Formal runs then execute a *new* full self-rollout with that fitted controller,
capture the newly visited states, and refit.  A cache from round one can never
be relabelled as round two: rollout IDs, input-controller hashes, policies and
lineage are all checked before publication.  Engineering smoke still performs
one complete 40-step capture, but uses one optimizer step and stops after the
first round.

The inference side of every rollout receives exactly an 81-frame source video
and an action instruction.  Paired targets are materialized into a separate
training-only object and are accepted only by the offline source-relative
motion objective.  No target, support, mask, flow, pose, track, trajectory or
edited first frame enters an EGNTC callback or ``renderer.sample`` call.

The output is fail-closed and non-overwriting.  A formal K=2 prototype is the
midpoint of the two independently fitted 36D support controllers.  After the
two training rounds it receives a third, evaluation-only, full self-rollout.
Its gate is evaluated separately on both fresh prototype caches and uses the
worst support ratios; own-support controllers are diagnostics, never the gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import inspect
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import generator_native_trajectory_controller as egntc  # noqa: E402
import train_fewshot_motion_code as episode_train  # noqa: E402
import tri_branch_unipc as tri  # noqa: E402


METHOD_NAME = "k2-episodic-generator-native-trajectory-controller"
RUN_RECEIPT_SCHEMA = "bernini-egntc-k2-world8-training-receipt-v1"
DIAGNOSTIC_SCHEMA = "bernini-egntc-k2-diagnostics-v1"

NUM_FRAMES = 81
LATENT_PHASES = 21
NUM_INFERENCE_STEPS = 40
FORWARDS_PER_STEP = 3
FORWARDS_PER_SUPPORT_ROUND = NUM_INFERENCE_STEPS * FORWARDS_PER_STEP
K_SHOT = 2
WORLD_SIZE = 8
ULYSSES_SIZE = 4
DATA_PARALLEL_SIZE = 2
TRAINABLE_DIMENSION = 36
FORMAL_ROUNDS = 2
SMOKE_ROUNDS = 1
CACHE_SPATIAL_HW = (60, 62)
SUPERVISED_STEP_INDICES = tuple(egntc.SIGMA_KNOT_SCHEDULE_INDICES)
DEFAULT_OPTIMIZER_STEPS_PER_ROUND = 25
DEFAULT_LEARNING_RATE = 2.0e-2
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 20260809
DEFAULT_ROLLOUT_SEED = 2029

FEATURE_WEIGHT = 1.0
PHASE_RMS_WEIGHT = 0.25
TEMPORAL_DC_WEIGHT = 0.05
PHASE0_WEIGHT = 0.25
CROSS_SIGMA_WEIGHT = 0.10
PARAMETER_L2_WEIGHT = 1.0e-5
SCHEDULE_SMOOTHNESS_WEIGHT = 1.0e-4

MIN_PROTOTYPE_NOOP_IMPROVEMENT = 0.15
MAX_PROTOTYPE_ACTION_REGRESSION = 0.05
MIN_CONTROL_IMPROVEMENT = 0.05
MAX_SECOND_ROUND_REGRESSION = 0.05
MIN_SUPPORT_PARAMETER_COSINE = 0.60

INFERENCE_CONDITIONS = ("source_video", "action_instruction")
FORBIDDEN_INFERENCE_CONDITIONS = (
    "target_video",
    "paired_target",
    "support_video",
    "mask",
    "flow",
    "optical_flow",
    "pose",
    "track",
    "swept_tube",
    "trajectory",
    "first_frame_anchor",
    "edited_first_frame",
    "reference_image",
    "reference_video",
)

_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class EGNTCTrainingError(RuntimeError):
    """Raised before an ambiguous step or artifact publication."""


def _tensor_content_bytes(value: Any) -> Any:
    """Return exact bytes with a zero-copy fast path and ABI-safe fallback.

    The pinned AUH runtime has a working NumPy ABI, where a memoryview is a
    zero-copy hashing buffer.  Some orchestration hosts have PyTorch compiled
    against NumPy 1.x but NumPy 2.x installed; only there do we fall back to a
    cloned untyped storage.  The fallback is correctness-first because real
    rollouts execute on the pinned fast path.
    """

    import torch

    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    try:
        return memoryview(tensor.numpy()).cast("B")
    except RuntimeError:
        pass
    # Clone the byte view so fallback storage is guaranteed to contain this
    # tensor and no bytes inherited from a larger base allocation.
    byte_view = tensor.view(torch.uint8).reshape(-1).clone()
    storage = byte_view.untyped_storage()
    if storage.nbytes() != byte_view.numel():
        raise EGNTCTrainingError("tensor byte storage size differs")
    return bytes(storage)


@dataclass(frozen=True)
class InferenceSupport:
    """The complete input available to capture and inference."""

    iid: str
    instruction: str
    source_clean_cpu: Any
    source_video_sha256: str
    instruction_sha256: str


@dataclass(frozen=True)
class TrainingOnlyMotionTeacher:
    """Privileged label object accepted only by offline objective functions."""

    iid: str
    source_clean_cpu: Any
    target_clean_cpu: Any
    target_video_sha256: str
    target_projection: str
    parquet_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class CachedCleanStep:
    """Detached full-resolution action/noop fields from one real solver state."""

    step_index: int
    timestep: float
    sigma: float
    model_id: str
    action_clean_cpu: Any
    noop_clean_cpu: Any
    native_delta_rms: float


@dataclass
class DetachedTrajectoryCache:
    """A single-use cache whose identity is bound to one physical sample call."""

    rollout_id: str
    round_index: int
    execution_policy: str
    input_controller_sha256: Optional[str]
    steps: list[CachedCleanStep] = field(default_factory=list)
    sealed: bool = False
    _trace_receipt: Optional[dict[str, Any]] = None
    _state_fields_sha256: Optional[str] = None

    def capture(self, fields: tri.CleanFieldStep) -> None:
        import torch

        if self.sealed:
            raise EGNTCTrainingError("cannot append to a sealed trajectory cache")
        expected = len(self.steps)
        if fields.step_index != expected or not 0 <= expected < NUM_INFERENCE_STEPS:
            raise EGNTCTrainingError(
                f"trajectory cache expected step {expected}, got {fields.step_index}"
            )
        action = _cache_clean_cpu(fields.action_guided_clean)
        noop = _cache_clean_cpu(fields.noop_guided_clean)
        delta = fields.action_delta_clean.detach().float()
        if tuple(delta.shape[:3]) != (1, 16, LATENT_PHASES):
            raise EGNTCTrainingError("captured native delta has wrong 81f geometry")
        native_rms = float(delta.square().mean().sqrt().cpu().item())
        if not math.isfinite(native_rms):
            raise EGNTCTrainingError("captured native delta RMS is non-finite")
        if not torch.equal(
            fields.action_delta_clean,
            fields.action_guided_clean - fields.noop_guided_clean,
        ):
            raise EGNTCTrainingError("tri-branch action/noop delta identity changed")
        self.steps.append(
            CachedCleanStep(
                step_index=fields.step_index,
                timestep=float(fields.timestep),
                sigma=float(fields.sigma),
                model_id=str(fields.model_id),
                action_clean_cpu=action,
                noop_clean_cpu=noop,
                native_delta_rms=native_rms,
            )
        )

    def seal(self, trace: tri.TriBranchTrace) -> dict[str, Any]:
        if self.sealed:
            raise EGNTCTrainingError("trajectory cache was sealed twice")
        trace_receipt = validate_full_rollout_trace(trace)
        if len(self.steps) != NUM_INFERENCE_STEPS:
            raise EGNTCTrainingError("trajectory cache does not contain 40 steps")
        if [item.step_index for item in self.steps] != list(
            range(NUM_INFERENCE_STEPS)
        ):
            raise EGNTCTrainingError("trajectory cache step order differs")
        for cached, record in zip(self.steps, trace.records):
            if (
                cached.step_index != record.step_index
                or cached.timestep != record.timestep
                or cached.sigma != record.sigma
                or cached.model_id != record.model_id
            ):
                raise EGNTCTrainingError("cache/UniPC trace state identity differs")
        self._trace_receipt = trace_receipt
        self._state_fields_sha256 = self._compute_state_fields_sha256()
        self.sealed = True
        return self.receipt()

    def _compute_state_fields_sha256(self) -> str:
        digest = hashlib.sha256()
        for item in self.steps:
            digest.update(
                episode_train.canonical_json_bytes(
                    {
                        "step_index": item.step_index,
                        "timestep": item.timestep,
                        "sigma": item.sigma,
                        "model_id": item.model_id,
                    }
                )
            )
            for tensor in (item.action_clean_cpu, item.noop_clean_cpu):
                digest.update(_tensor_content_bytes(tensor))
        return digest.hexdigest()

    def digest(self) -> str:
        if not self.sealed or self._state_fields_sha256 is None:
            raise EGNTCTrainingError("unsealed trajectory cache has no digest")
        return episode_train.object_sha256(
            {
                "rollout_id": self.rollout_id,
                "state_fields_sha256": self._state_fields_sha256,
            }
        )

    def receipt(self) -> dict[str, Any]:
        if not self.sealed or self._trace_receipt is None:
            raise EGNTCTrainingError("cannot publish an unsealed trajectory cache")
        return {
            "rollout_id": self.rollout_id,
            "round_index": self.round_index,
            "execution_policy": self.execution_policy,
            "input_controller_sha256": self.input_controller_sha256,
            "detached": True,
            "stored_on_cpu": True,
            "cache_spatial_hw": list(CACHE_SPATIAL_HW),
            "cache_precision": "torch.float32",
            "controller_replay_resolution": "exact_full_60x62_no_spatial_pooling",
            "step_count": len(self.steps),
            "transformer_forwards": FORWARDS_PER_SUPPORT_ROUND,
            "target_seen_by_callback": False,
            "state_fields_sha256": self._state_fields_sha256,
            "cache_sha256": self.digest(),
            "trace": dict(self._trace_receipt),
        }


class _RolloutCaptureCallback:
    """Capture same-state fields, then execute action or the learned policy.

    The constructor and call path intentionally have no target argument.
    """

    def __init__(
        self,
        *,
        cache: DetachedTrajectoryCache,
        execution_policy: str,
        parameters: Optional[Any],
        source_clean: Any,
    ) -> None:
        if execution_policy not in ("official_action", "learned_controller"):
            raise EGNTCTrainingError("unknown rollout execution policy")
        if (execution_policy == "learned_controller") != (parameters is not None):
            raise EGNTCTrainingError("learned rollout must receive exactly one controller")
        self.cache = cache
        self.execution_policy = execution_policy
        self.controller = (
            egntc.EGNTCCallback(parameters, source_clean)
            if parameters is not None
            else None
        )

    def __call__(self, fields: tri.CleanFieldStep) -> Any:
        self.cache.capture(fields)
        if self.execution_policy == "official_action":
            # Preserve the exact object so tri_branch_unipc can certify the
            # official action APG fast path before its one scheduler call.
            return fields.action_guided_clean
        if self.controller is None:
            raise EGNTCTrainingError("learned rollout lost its controller")
        return self.controller(fields)

    def controller_receipt(self) -> Optional[dict[str, Any]]:
        if self.controller is None:
            return None
        return self.controller.receipt()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k2-config", required=True)
    parser.add_argument("--expected-k2-config-sha256", required=True)
    parser.add_argument("--preview-manifest", required=True)
    parser.add_argument("--vae-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-frames", type=int, choices=(NUM_FRAMES,), default=NUM_FRAMES)
    parser.add_argument("--k-shot", type=int, choices=(K_SHOT,), default=K_SHOT)
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        choices=(NUM_INFERENCE_STEPS,),
        default=NUM_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--optimizer-steps-per-round",
        type=int,
        default=DEFAULT_OPTIMIZER_STEPS_PER_ROUND,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rollout-seed", type=int, default=DEFAULT_ROLLOUT_SEED)
    parser.add_argument("--teacher-sigma-index", type=int, default=20)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--ack-preview-experimental-only", action="store_true")
    parser.add_argument(
        "--expected-bernini-commit", default=episode_train.legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=episode_train.legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=episode_train.legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if args.ack_preview_experimental_only is not True:
        raise EGNTCTrainingError("--ack-preview-experimental-only is mandatory")
    if (
        args.num_frames != NUM_FRAMES
        or args.k_shot != K_SHOT
        or args.num_inference_steps != NUM_INFERENCE_STEPS
    ):
        raise EGNTCTrainingError("EGNTC training is frozen to 81f, K=2 and 40 UniPC steps")
    if type(args.optimizer_steps_per_round) is not int or args.optimizer_steps_per_round <= 0:
        raise EGNTCTrainingError("optimizer-steps-per-round must be positive")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
            raise EGNTCTrainingError(f"{name} must be finite and positive")
    for name in ("seed", "rollout_seed"):
        value = getattr(args, name)
        if type(value) is not int or not 0 <= value < 2**63:
            raise EGNTCTrainingError(f"{name} must lie in [0,2^63)")
    if type(args.teacher_sigma_index) is not int or not 0 <= args.teacher_sigma_index < 40:
        raise EGNTCTrainingError("teacher-sigma-index must lie in [0,39]")
    for name in ("expected_bernini_commit", "expected_veomni_commit", "method_source_revision"):
        if _SHA1.fullmatch(str(getattr(args, name))) is None:
            raise EGNTCTrainingError(f"{name} must be a lowercase full SHA-1")
    for name in (
        "expected_k2_config_sha256",
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
    ):
        if _SHA256.fullmatch(str(getattr(args, name))) is None:
            raise EGNTCTrainingError(f"{name} must be a lowercase SHA-256")
    if args.expected_checkpoint_tree_sha256 != episode_train.legacy.CHECKPOINT_TREE_SHA256:
        raise EGNTCTrainingError("checkpoint tree differs from the audited Bernini 1.3B tree")
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.suffix:
        raise EGNTCTrainingError("output must be an absolute directory")


def round_plan(*, engineering_smoke: bool, optimizer_steps: int) -> tuple[dict[str, Any], ...]:
    if type(engineering_smoke) is not bool or type(optimizer_steps) is not int or optimizer_steps <= 0:
        raise EGNTCTrainingError("invalid EGNTC round plan request")
    planned = [
        {
            "round_index": 1,
            "execution_policy": "official_action",
            "optimizer_steps": 1 if engineering_smoke else optimizer_steps,
            "full_unipc_steps": NUM_INFERENCE_STEPS,
            "transformer_forwards": FORWARDS_PER_SUPPORT_ROUND,
        }
    ]
    if not engineering_smoke:
        planned.append(
            {
                "round_index": 2,
                "execution_policy": "learned_controller",
                "optimizer_steps": optimizer_steps,
                "full_unipc_steps": NUM_INFERENCE_STEPS,
                "transformer_forwards": FORWARDS_PER_SUPPORT_ROUND,
            }
        )
    return tuple(planned)


def _cache_clean_cpu(value: Any) -> Any:
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3]) != (1, 16, LATENT_PHASES)
        or tuple(int(item) for item in value.shape[3:]) != CACHE_SPATIAL_HW
        or not bool(torch.isfinite(value).all().item())
    ):
        raise EGNTCTrainingError("clean field must be finite [1,16,21,60,62]")
    # RMSClip is nonlinear in spatial RMS.  Keeping exact fp32 resolution is
    # therefore required: applying the controller on an 8x8 pooled cache would
    # change its saturation decisions relative to full-resolution inference.
    return value.detach().float().contiguous().cpu()


def _controller_sha256(parameters: Any) -> str:
    vector = parameters.flat_tensor(detach=True).float().cpu().contiguous()
    if tuple(vector.shape) != (TRAINABLE_DIMENSION,):
        raise EGNTCTrainingError("controller vector is not exact 36D")
    return hashlib.sha256(bytes(vector.view(__import__("torch").uint8).tolist())).hexdigest()


def _rollout_id(
    *, iid: str, round_index: int, policy: str, input_controller_sha256: Optional[str], seed: int
) -> str:
    payload = {
        "iid": iid,
        "round_index": round_index,
        "execution_policy": policy,
        "input_controller_sha256": input_controller_sha256,
        "seed": seed,
    }
    return episode_train.object_sha256(payload)


def validate_full_rollout_trace(trace: tri.TriBranchTrace) -> dict[str, Any]:
    if trace.sample_calls != 1 or len(trace.records) != NUM_INFERENCE_STEPS:
        raise EGNTCTrainingError("capture must contain one complete 40-step sample")
    if [record.step_index for record in trace.records] != list(range(NUM_INFERENCE_STEPS)):
        raise EGNTCTrainingError("UniPC trace step order differs from 0..39")
    checks = (
        all(record.transformer_forwards == FORWARDS_PER_STEP for record in trace.records),
        all(record.shared_negative_forwards == 1 for record in trace.records),
        all(record.action_forwards == 1 for record in trace.records),
        all(record.noop_forwards == 1 for record in trace.records),
        all(record.original_scheduler_calls == 1 for record in trace.records),
        all(record.official_action_exact_parity is True for record in trace.records),
    )
    if not all(checks):
        raise EGNTCTrainingError("tri-branch trace violates the 3-forward/1-UniPC contract")
    return {
        "sample_calls": 1,
        "step_count": NUM_INFERENCE_STEPS,
        "transformer_forwards": sum(record.transformer_forwards for record in trace.records),
        "negative_forwards": NUM_INFERENCE_STEPS,
        "action_forwards": NUM_INFERENCE_STEPS,
        "noop_forwards": NUM_INFERENCE_STEPS,
        "original_unipc_scheduler_calls": NUM_INFERENCE_STEPS,
        "official_action_apg_certified_each_step": True,
        "step_indices": list(range(NUM_INFERENCE_STEPS)),
    }


def validate_round_lineage(receipts: Sequence[Mapping[str, Any]]) -> None:
    if len(receipts) not in (SMOKE_ROUNDS, FORMAL_ROUNDS):
        raise EGNTCTrainingError("round lineage must contain one smoke or two formal rounds")
    first = receipts[0]
    if (
        first.get("round_index") != 1
        or first.get("execution_policy") != "official_action"
        or first.get("input_controller_sha256") is not None
    ):
        raise EGNTCTrainingError("round one must be a controller-free official action rollout")
    ids = [item.get("rollout_id") for item in receipts]
    caches = [item.get("cache_sha256") for item in receipts]
    fields = [item.get("state_fields_sha256") for item in receipts]
    if any(
        not isinstance(item, str) or _SHA256.fullmatch(item) is None
        for item in ids + caches + fields
    ):
        raise EGNTCTrainingError("round lineage lacks full SHA-256 identities")
    if len(set(ids)) != len(ids):
        raise EGNTCTrainingError("a physical rollout ID was reused across rounds")
    if len(receipts) == FORMAL_ROUNDS:
        second = receipts[1]
        if (
            second.get("round_index") != 2
            or second.get("execution_policy") != "learned_controller"
            or second.get("input_controller_sha256")
            != first.get("output_controller_sha256")
        ):
            raise EGNTCTrainingError("round two is not a fresh rollout of round-one controller")
        if second["state_fields_sha256"] == first["state_fields_sha256"]:
            raise EGNTCTrainingError(
                "round-two learned self-rollout reproduced the round-one state cache"
            )


def _charbonnier_zero(value: Any, epsilon: float = 1.0e-3) -> Any:
    import torch

    return (torch.sqrt(value.float().square() + epsilon**2) - epsilon).mean()


def _execute_cached_controller(
    cache: DetachedTrajectoryCache,
    *,
    parameters: Any,
    source_clean: Any,
    device: Any,
    controls: Optional[Any] = None,
) -> tuple[list[Any], dict[str, Any]]:
    if not cache.sealed:
        raise EGNTCTrainingError("offline fitting requires a sealed physical rollout cache")
    callback = egntc.EGNTCCallback(parameters, source_clean, controls=controls)
    selected: list[Any] = []
    for item in cache.steps:
        executed = callback.apply_fields(
            action_clean=item.action_clean_cpu.to(device, non_blocking=True),
            noop_clean=item.noop_clean_cpu.to(device, non_blocking=True),
            step_index=item.step_index,
            timestep=item.timestep,
            sigma=item.sigma,
        )
        if item.step_index in SUPERVISED_STEP_INDICES:
            selected.append(executed)
    if len(selected) != len(SUPERVISED_STEP_INDICES) or callback.expected_step != NUM_INFERENCE_STEPS:
        raise EGNTCTrainingError("offline controller did not replay all 40 cached states")
    receipt = callback.receipt()
    egntc.validate_controller_receipt(receipt, require_complete=True)
    return selected, receipt


def _source_relative_objective(
    predicted_clean_fields: Sequence[Any],
    *,
    source_clean: Any,
    target_clean: Any,
    parameters: Optional[Any],
) -> tuple[Any, dict[str, float]]:
    """Training-only target use: source-relative pooled motion, never pixels."""

    import torch
    import fewshot_teacher_objective as teacher_objective

    if len(predicted_clean_fields) != len(SUPERVISED_STEP_INDICES):
        raise EGNTCTrainingError("objective requires all six registered sigma knots")
    target_features = teacher_objective.source_relative_motion_features(
        target_clean.detach(), source_clean.detach()
    )
    feature_losses = []
    rms_losses = []
    dc_losses = []
    phase0_losses = []
    predicted_features = []
    for predicted in predicted_clean_fields:
        features = teacher_objective.source_relative_motion_features(predicted, source_clean)
        predicted_features.append(features)
        feature_losses.append(
            teacher_objective.motion_feature_match_loss(features, target_features)
        )
        rms_losses.append(
            teacher_objective.phase_rms_match_loss(features, target_features)
        )
        dc_losses.append(
            teacher_objective.temporal_dc_residual_penalty(predicted, source_clean)
        )
        phase0_losses.append(
            teacher_objective.target_phase0_base_parity_penalty(predicted, source_clean)
        )
    feature = torch.stack(feature_losses).mean()
    phase_rms = torch.stack(rms_losses).mean()
    temporal_dc = torch.stack(dc_losses).mean()
    phase0 = torch.stack(phase0_losses).mean()
    consistency = torch.stack(
        [
            _charbonnier_zero(right.pooled_q0 - left.pooled_q0)
            for left, right in zip(predicted_features[:-1], predicted_features[1:])
        ]
    ).mean()
    if parameters is None:
        parameter_l2 = feature * 0.0
        smoothness = feature * 0.0
    else:
        vector = parameters.flat_tensor(detach=False)
        parameter_l2 = vector.square().mean()
        alpha = vector[:24].reshape(6, 4)
        kappa = vector[24:30]
        rho = vector[30:36]
        smoothness = torch.stack(
            (
                (alpha[1:] - alpha[:-1]).square().mean(),
                (kappa[1:] - kappa[:-1]).square().mean(),
                (rho[1:] - rho[:-1]).square().mean(),
            )
        ).mean()
    total = (
        FEATURE_WEIGHT * feature
        + PHASE_RMS_WEIGHT * phase_rms
        + TEMPORAL_DC_WEIGHT * temporal_dc
        + PHASE0_WEIGHT * phase0
        + CROSS_SIGMA_WEIGHT * consistency
        + PARAMETER_L2_WEIGHT * parameter_l2
        + SCHEDULE_SMOOTHNESS_WEIGHT * smoothness
    )
    if total.ndim != 0 or not bool(torch.isfinite(total).item()):
        raise EGNTCTrainingError("source-relative controller objective is invalid")
    stats = {
        "total": float(total.detach().item()),
        "feature_match": float(feature.detach().item()),
        "phase_rms_match": float(phase_rms.detach().item()),
        "temporal_dc": float(temporal_dc.detach().item()),
        "phase0_parity": float(phase0.detach().item()),
        "cross_sigma_consistency": float(consistency.detach().item()),
        "parameter_l2": float(parameter_l2.detach().item()),
        "schedule_smoothness": float(smoothness.detach().item()),
    }
    return total, stats


def _offline_fit(
    cache: DetachedTrajectoryCache,
    *,
    parameters: Any,
    teacher: TrainingOnlyMotionTeacher,
    device: Any,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    ulysses_group: Any,
) -> tuple[list[dict[str, float]], str]:
    """The only function that accepts the paired target object."""

    import torch

    if teacher.iid == "" or steps <= 0:
        raise EGNTCTrainingError("offline fitter received an invalid support")
    source = _cache_clean_cpu(teacher.source_clean_cpu).to(device)
    target = _cache_clean_cpu(teacher.target_clean_cpu).to(device)
    parameters.train()
    optimizer = torch.optim.AdamW(
        parameters.parameters(), lr=learning_rate, weight_decay=0.0
    )
    history: list[dict[str, float]] = []
    for optimizer_step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = _execute_cached_controller(
            cache, parameters=parameters, source_clean=source, device=device
        )
        loss, statistics = _source_relative_objective(
            predicted,
            source_clean=source,
            target_clean=target,
            parameters=parameters,
        )
        loss.backward()
        try:
            synchronized_norm = episode_train._all_reduce_code_gradients(
                parameters, ulysses_group=ulysses_group
            )
        except episode_train.FewShotCodeTrainingError as error:
            raise EGNTCTrainingError(str(error)) from error
        clipped = torch.nn.utils.clip_grad_norm_(
            tuple(parameters.parameters()), max_grad_norm
        )
        clipped_value = float(clipped.detach().float().item())
        if not math.isfinite(clipped_value):
            raise EGNTCTrainingError("controller gradient norm is non-finite")
        optimizer.step()
        parameters.validate_parameters()
        history.append(
            {
                "optimizer_step": float(optimizer_step + 1),
                "synchronized_gradient_norm": synchronized_norm,
                "preclip_gradient_norm": clipped_value,
                **statistics,
            }
        )
    parameters.eval()
    return history, _assert_sp_controller_exact(parameters, ulysses_group=ulysses_group)


def _assert_sp_controller_exact(parameters: Any, *, ulysses_group: Any) -> str:
    import torch
    import torch.distributed as dist

    local = parameters.flat_tensor(detach=True).float().contiguous()
    if tuple(local.shape) != (TRAINABLE_DIMENSION,):
        raise EGNTCTrainingError("local controller is not 36D")
    if dist.get_world_size(ulysses_group) != ULYSSES_SIZE:
        raise EGNTCTrainingError("controller replication requires one SP4 group")
    gathered = [torch.empty_like(local) for _ in range(ULYSSES_SIZE)]
    dist.all_gather(gathered, local, group=ulysses_group)
    if any(not torch.equal(item, local) for item in gathered):
        raise EGNTCTrainingError("controller parameters differ within SP4")
    return _controller_sha256(parameters)


def _exchange_k2_controller_vectors(
    parameters: Any,
    *,
    support_index: int,
    support_iids: Sequence[str],
    dp_group: Any,
    world_group: Any,
) -> tuple[list[Any], str]:
    import torch
    import torch.distributed as dist

    if len(support_iids) != K_SHOT or support_index not in range(K_SHOT):
        raise EGNTCTrainingError("controller exchange requires ordered K=2")
    if (
        dist.get_world_size(dp_group) != DATA_PARALLEL_SIZE
        or dist.get_rank(dp_group) != support_index
    ):
        raise EGNTCTrainingError("controller exchange DP2 lane ordering differs")
    local = parameters.flat_tensor(detach=True).float().contiguous()
    gathered = [torch.empty_like(local) for _ in range(K_SHOT)]
    dist.all_gather(gathered, local, group=dp_group)
    if any(
        tuple(item.shape) != (TRAINABLE_DIMENSION,)
        or item.dtype != torch.float32
        or not bool(torch.isfinite(item).all().item())
        for item in gathered
    ):
        raise EGNTCTrainingError("K=2 controller exchange returned invalid vectors")
    try:
        digest = episode_train._world_digest_consensus(
            {"support_iids": list(support_iids), "ordered_controller_vectors": gathered},
            world_group=world_group,
            context="ordered K=2 EGNTC controller exchange",
        )
    except episode_train.FewShotCodeTrainingError as error:
        raise EGNTCTrainingError(str(error)) from error
    return gathered, digest


def _support_parameter_cosine(
    vectors: Sequence[Any], *, origin: Optional[Any] = None
) -> float:
    import torch

    if len(vectors) != K_SHOT:
        raise EGNTCTrainingError("support cosine requires two controller vectors")
    left, right = (item.detach().float().reshape(-1) for item in vectors)
    if origin is not None:
        base = origin.detach().float().reshape(-1)
        if tuple(base.shape) != (TRAINABLE_DIMENSION,):
            raise EGNTCTrainingError("support cosine origin is not 36D")
        left = left - base
        right = right - base
    denominator = left.norm() * right.norm()
    if float(denominator.item()) == 0.0:
        return 1.0 if torch.equal(left, right) else 0.0
    value = float(torch.dot(left, right).div(denominator).item())
    if not math.isfinite(value):
        raise EGNTCTrainingError("support controller cosine is non-finite")
    return value


def _evaluate_cache_arm(
    cache: DetachedTrajectoryCache,
    *,
    arm: str,
    parameters: Optional[Any],
    controls: Optional[Any],
    teacher: TrainingOnlyMotionTeacher,
    device: Any,
) -> dict[str, float]:
    import torch

    source = _cache_clean_cpu(teacher.source_clean_cpu).to(device)
    target = _cache_clean_cpu(teacher.target_clean_cpu).to(device)
    with torch.no_grad():
        if arm in ("noop", "raw_action"):
            attr = "noop_clean_cpu" if arm == "noop" else "action_clean_cpu"
            predicted = [
                getattr(cache.steps[index], attr).to(device)
                for index in SUPERVISED_STEP_INDICES
            ]
            callback_receipt = None
        else:
            if parameters is None:
                raise EGNTCTrainingError("controller evaluation arm lacks parameters")
            predicted, callback_receipt = _execute_cached_controller(
                cache,
                parameters=parameters,
                source_clean=source,
                device=device,
                controls=controls,
            )
        _, statistics = _source_relative_objective(
            predicted,
            source_clean=source,
            target_clean=target,
            parameters=None,
        )
    result = {"loss": statistics["total"], **statistics}
    if callback_receipt is not None:
        # ``EGNTCCallback.receipt`` keeps mutable rollout counters under the
        # authenticated ``state`` object.  Do not confuse this controller
        # receipt with ``validate_full_rollout_trace`` whose step_count is a
        # top-level field.
        try:
            egntc.validate_controller_receipt(
                callback_receipt, require_complete=True
            )
        except egntc.EGNTCContractError as error:
            raise EGNTCTrainingError(
                "offline controller returned an invalid complete receipt"
            ) from error
        state = callback_receipt.get("state")
        if (
            not isinstance(state, Mapping)
            or type(state.get("step_count")) is not int
            or state["step_count"] != NUM_INFERENCE_STEPS
        ):
            raise EGNTCTrainingError(
                "offline controller receipt lacks the nested 40-step state"
            )
        result["controller_trace_steps"] = float(state["step_count"])
    return result


def prototype_gate(
    support_evaluations: Sequence[Mapping[str, Any]],
    *,
    support_parameter_cosine: float,
    engineering_smoke: bool,
) -> dict[str, Any]:
    if engineering_smoke:
        return {
            "representability_gate": "NOT_EVALUATED_ENGINEERING_SMOKE",
            "deployable": False,
            "diagnostic_only": True,
            "checks": {},
            "failed_checks": [],
        }
    if len(support_evaluations) != K_SHOT:
        raise EGNTCTrainingError("formal prototype gate requires both supports")
    per_support = []
    for item in support_evaluations:
        losses = item.get("losses")
        if not isinstance(losses, Mapping):
            raise EGNTCTrainingError("support evaluation lacks losses")
        noop = float(losses["noop"])
        action = float(losses["raw_action"])
        prototype = float(losses["prototype"])
        phase_reverse = float(losses["prototype_phase_reverse"])
        sigma_shuffle = float(losses["prototype_sigma_shuffle"])
        round_objectives = item.get("round_postfit_objectives")
        if not isinstance(round_objectives, Sequence) or len(round_objectives) != FORMAL_ROUNDS:
            raise EGNTCTrainingError("formal gate lacks both postfit round objectives")
        round_one, round_two = map(float, round_objectives)
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                noop,
                action,
                prototype,
                phase_reverse,
                sigma_shuffle,
                round_one,
                round_two,
            )
        ):
            raise EGNTCTrainingError("prototype gate received invalid loss")
        noop_ratio = prototype / max(noop, 1.0e-12)
        action_ratio = prototype / max(action, 1.0e-12)
        phase_reverse_ratio = prototype / max(phase_reverse, 1.0e-12)
        sigma_shuffle_ratio = prototype / max(sigma_shuffle, 1.0e-12)
        second_round_ratio = round_two / max(round_one, 1.0e-12)
        per_support.append(
            {
                "iid": item.get("iid"),
                "prototype_to_noop_ratio": noop_ratio,
                "prototype_to_raw_action_ratio": action_ratio,
                "prototype_to_phase_reverse_ratio": phase_reverse_ratio,
                "prototype_to_sigma_shuffle_ratio": sigma_shuffle_ratio,
                "round2_to_round1_postfit_objective_ratio": second_round_ratio,
                "noop_improvement": 1.0 - noop_ratio,
                "passes_noop": noop_ratio <= 1.0 - MIN_PROTOTYPE_NOOP_IMPROVEMENT,
                "passes_raw_action": action_ratio <= 1.0 + MAX_PROTOTYPE_ACTION_REGRESSION,
                "passes_phase_reverse": phase_reverse_ratio <= 1.0 - MIN_CONTROL_IMPROVEMENT,
                "passes_sigma_shuffle": sigma_shuffle_ratio <= 1.0 - MIN_CONTROL_IMPROVEMENT,
                "passes_second_round_regression": second_round_ratio <= 1.0 + MAX_SECOND_ROUND_REGRESSION,
            }
        )
    worst_noop = max(item["prototype_to_noop_ratio"] for item in per_support)
    worst_action = max(item["prototype_to_raw_action_ratio"] for item in per_support)
    checks = {
        "every_support_improves_noop_by_15pct": all(item["passes_noop"] for item in per_support),
        "every_support_within_5pct_of_raw_action": all(item["passes_raw_action"] for item in per_support),
        "every_support_beats_phase_reverse_by_5pct": all(item["passes_phase_reverse"] for item in per_support),
        "every_support_beats_sigma_shuffle_by_5pct": all(item["passes_sigma_shuffle"] for item in per_support),
        "every_support_round2_regression_le_5pct": all(
            item["passes_second_round_regression"] for item in per_support
        ),
        "support_parameter_cosine_ge_0p60": support_parameter_cosine >= MIN_SUPPORT_PARAMETER_COSINE,
    }
    passed = all(checks.values())
    return {
        "representability_gate": "GO" if passed else "NO_GO",
        "tensor_representability_gate": "GO" if passed else "NO_GO",
        # The formal runner adds a post-refit full-resolution prototype
        # self-rollout, but it still does not decode or score video quality.
        # Tensor GO therefore remains a diagnostic artifact.
        "deployable": False,
        "diagnostic_only": True,
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "per_support": per_support,
        "worst_support_prototype_to_noop_ratio": worst_noop,
        "worst_support_prototype_to_raw_action_ratio": worst_action,
        "support_update_vector_cosine": support_parameter_cosine,
        "thresholds": {
            "minimum_noop_improvement": MIN_PROTOTYPE_NOOP_IMPROVEMENT,
            "maximum_raw_action_regression": MAX_PROTOTYPE_ACTION_REGRESSION,
            "minimum_phase_reverse_improvement": MIN_CONTROL_IMPROVEMENT,
            "minimum_sigma_shuffle_improvement": MIN_CONTROL_IMPROVEMENT,
            "maximum_second_round_regression": MAX_SECOND_ROUND_REGRESSION,
            "minimum_support_parameter_cosine": MIN_SUPPORT_PARAMETER_COSINE,
        },
        "aggregation": "per_support_conjunction_and_worst_support_ratios",
        "own_support_controller_role": "diagnostic_only",
        "missing_deployment_gate": (
            "decoded_video_motion_identity_and_source_consistency_quality_gate"
        ),
    }


def _materialize_support(
    *,
    row: Any,
    raw: Mapping[str, Any],
    parquet_receipt: Mapping[str, Any],
    tokenizer: Any,
    rope: Any,
    vae_mean: Any,
    vae_std: Any,
    z_dim: int,
    scheduler: Any,
    process_renderer_sample: Any,
    device: Any,
    seed: int,
    sigma_index: int,
) -> tuple[InferenceSupport, TrainingOnlyMotionTeacher]:
    teacher = episode_train._prepare_teacher_cell(
        episode_row=row,
        raw_row=raw,
        parquet_receipt=parquet_receipt,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        noise_scheduler=scheduler,
        process_renderer_sample=process_renderer_sample,
        device=device,
        noise_seed=episode_train._seed_for_iid(seed, row.iid, "egntc-teacher"),
        sigma_index=sigma_index,
    )
    source_video = episode_train._packed_clean_video(
        teacher.auxiliary["source_clean"]
    ).detach().float().cpu()
    target_video = episode_train._packed_clean_video(
        teacher.auxiliary["target_clean"]
    ).detach().float().cpu()
    source_inference = teacher.source_latent_cpu.detach().float().cpu()
    if not __import__("torch").equal(source_video, source_inference):
        raise EGNTCTrainingError("teacher and inference source latent layouts differ")
    inference_support = InferenceSupport(
        iid=row.iid,
        instruction=row.edit_instruction,
        source_clean_cpu=source_inference,
        source_video_sha256=row.source_video_sha256,
        instruction_sha256=row.edit_instruction_sha256,
    )
    training_teacher = TrainingOnlyMotionTeacher(
        iid=row.iid,
        source_clean_cpu=source_video,
        target_clean_cpu=target_video,
        target_video_sha256=str(getattr(row, "target_video_sha256")),
        target_projection=str(teacher.auxiliary["target_projection"]),
        parquet_receipt=dict(teacher.parquet_receipt),
    )
    # Do not retain action/noop/negative training batches across inference.
    del teacher
    return inference_support, training_teacher


def _capture_rollout(
    *,
    renderer: Any,
    tokenizer: Any,
    support: InferenceSupport,
    parameters: Optional[Any],
    execution_policy: str,
    round_index: int,
    rollout_seed: int,
    device: Any,
    prompt_cleaner: Any,
    bernini_revision: str,
    wan_diffusion_path: Path,
) -> tuple[DetachedTrajectoryCache, dict[str, Any]]:
    """Run source+instruction inference; no target object is accepted."""

    import torch
    import infer_lora as inference
    import inference_sigma_strata as sigma_strata
    from spt_v2 import infer_c2fr as frozen_inference

    source = support.source_clean_cpu.to(device=device, dtype=torch.float32)
    input_controller_sha = (
        _controller_sha256(parameters) if parameters is not None else None
    )
    rollout_id = _rollout_id(
        iid=support.iid,
        round_index=round_index,
        policy=execution_policy,
        input_controller_sha256=input_controller_sha,
        seed=rollout_seed,
    )
    cache = DetachedTrajectoryCache(
        rollout_id=rollout_id,
        round_index=round_index,
        execution_policy=execution_policy,
        input_controller_sha256=input_controller_sha,
    )
    callback = _RolloutCaptureCallback(
        cache=cache,
        execution_policy=execution_policy,
        parameters=parameters,
        source_clean=source,
    )
    action_prompt = inference.build_training_prompt(
        support.instruction, prompt_cleaner=prompt_cleaner
    )
    noop_prompt = inference.build_training_prompt(
        episode_train.EXACT_NOOP_INSTRUCTION, prompt_cleaner=prompt_cleaner
    )
    action_ids, action_mask = inference._tokenize_training_prompt(tokenizer, action_prompt)
    noop_ids, noop_mask = inference._tokenize_training_prompt(tokenizer, noop_prompt)
    negative_ids, negative_mask = inference._tokenize_renderer_negative(
        tokenizer, inference.DEFAULT_NEGATIVE_PROMPT
    )
    noop_embeddings, noop_identity = frozen_inference.encode_semantic_noop_prompt(
        renderer, noop_ids, noop_mask, device=device
    )
    sampling = frozen_inference.exact_sampler_contract(seed=rollout_seed)
    if (
        sampling.get("num_frames") != NUM_FRAMES
        or sampling.get("num_inference_steps") != NUM_INFERENCE_STEPS
        or sampling.get("guidance_mode") != "v2v_apg"
        or sampling.get("momentum") != 0.0
        or sampling.get("flow_shift") != 5.0
    ):
        raise EGNTCTrainingError("official Bernini 81f/40-step sampler contract changed")
    diffusion = tri.resolve_diffusion_core(renderer)
    pre_schedule = sigma_strata.audit_runtime_unipc_schedule(
        diffusion.scheduler, initialize=True
    )
    generated = None
    trace = None
    try:
        with tri.tri_branch_unipc_hook(
            renderer,
            noop_prompt_embeds=noop_embeddings,
            latent_shape=tuple(int(item) for item in source.shape),
            clean_field_callback=callback,
            bernini_commit=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
            expected_steps=NUM_INFERENCE_STEPS,
            expected_flow_shift=5.0,
        ) as runtime_trace:
            with torch.no_grad():
                generated = renderer.sample(
                    input_ids=action_ids.to(device),
                    attention_mask=action_mask.to(device),
                    uncond_input_ids=negative_ids.to(device),
                    uncond_attention_mask=negative_mask.to(device),
                    image_vae_latents=None,
                    multi_video_vae_latents=[source],
                    multi_image_vae_latents=None,
                    width=episode_train.episode_io.EXPECTED_BUCKET_HW[1],
                    height=episode_train.episode_io.EXPECTED_BUCKET_HW[0],
                    device=device,
                    **sampling,
                )
            trace = runtime_trace
    finally:
        episode_train._restore_frozen_text_encoder(renderer, device)
    if generated is None or trace is None:
        raise EGNTCTrainingError("Bernini rollout returned no generated latent/trace")
    if tuple(int(item) for item in generated.shape) != tuple(int(item) for item in source.shape):
        raise EGNTCTrainingError("generated latent differs from exact 81f geometry")
    post_schedule = sigma_strata.audit_runtime_unipc_schedule(
        diffusion.scheduler, initialize=False
    )
    if pre_schedule != post_schedule:
        raise EGNTCTrainingError("renderer.sample changed the pinned UniPC schedule")
    cache_receipt = cache.seal(trace)
    output_digest = hashlib.sha256(_tensor_content_bytes(generated)).hexdigest()
    return cache, {
        **cache_receipt,
        "generated_latent_sha256": output_digest,
        "generated_latent_shape": list(generated.shape),
        "noop_prompt": noop_identity,
        "controller_execution": callback.controller_receipt(),
        "inference_conditions": list(INFERENCE_CONDITIONS),
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
    }


def _atomic_save_controller(path: Path, vector: Any) -> dict[str, Any]:
    import tempfile
    from safetensors.torch import save_file

    value = vector.detach().float().cpu().contiguous()
    if tuple(value.shape) != (TRAINABLE_DIMENSION,):
        raise EGNTCTrainingError("prototype checkpoint must contain exact 36D vector")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_file({"controller_raw_36d": value}, str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return {
        "filename": path.name,
        "sha256": episode_train.file_sha256(path),
        "tensor_keys": ["controller_raw_36d"],
        "shape": [TRAINABLE_DIMENSION],
        "dtype": "torch.float32",
    }


def _source_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "controller": Path(inspect.getsourcefile(egntc) or "").resolve(strict=True),
        "episode_trainer": Path(inspect.getsourcefile(episode_train) or "").resolve(strict=True),
        "tri_branch_unipc": Path(inspect.getsourcefile(tri) or "").resolve(strict=True),
    }
    return {name: episode_train.file_sha256(path) for name, path in files.items()}


def _write_controller_companion_receipt(
    *,
    output_dir: Path,
    checkpoint: Mapping[str, Any],
    vector: Any,
    gate: Mapping[str, Any],
    run_receipt_sha256: str,
    support_iids: Sequence[str],
) -> dict[str, Any]:
    try:
        import infer_generator_native_trajectory_controller as inference_runner
    except ImportError as error:
        raise EGNTCTrainingError("EGNTC inference receipt builder is unavailable") from error
    receipt = inference_runner.build_controller_training_receipt(
        state_filename=str(checkpoint["filename"]),
        state_file_sha256=str(checkpoint["sha256"]),
        raw_36d=vector.detach().float().cpu().contiguous(),
        representability_gate=str(gate["representability_gate"]),
        deployable=bool(gate["deployable"]),
        training_run_receipt_sha256=run_receipt_sha256,
        support_iids=list(support_iids),
    )
    episode_train._atomic_write_json(output_dir / "controller.receipt.json", receipt)
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    plan = round_plan(
        engineering_smoke=args.engineering_smoke,
        optimizer_steps=args.optimizer_steps_per_round,
    )
    episode = episode_train.load_audited_episode(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            episode_train.legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = episode_train.legacy.validate_checkpoint(
            args.checkpoint
        )
    except episode_train.legacy.TrainingContractError as error:
        raise EGNTCTrainingError(str(error)) from error
    if transformer_config["num_attention_heads"] % ULYSSES_SIZE:
        raise EGNTCTrainingError("Bernini's 12 heads must divide over Ulysses=4")
    episode_train.legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import __version__ as diffusers_version
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.parallel import init_parallel_state
    from bernini.training.data import NoiseScheduler, process_renderer_sample
    import infer_lora as inference

    contract = episode_train.epmc_distributed_contract()
    device, backend = episode_train.initialise_epmc_distributed(contract)
    parallel_state = init_parallel_state(ulysses_size=ULYSSES_SIZE)
    parallel = episode_train.validate_epmc_parallel_state(contract, parallel_state)
    support_assignments = episode_train._support_assignments(episode.supports)
    support_iids = [row.iid for row in episode.supports]
    assigned_row = episode.supports[parallel.support_index]
    distributed_receipt = episode_train._distributed_receipt(
        support_assignments=support_assignments, backend=backend
    )
    output_dir = Path(args.output).expanduser()
    try:
        episode_train._create_output_directory(
            output_dir, rank=contract.global_rank, world_group=parallel.world_group
        )
    except episode_train.FewShotCodeTrainingError as error:
        raise EGNTCTrainingError(str(error)) from error
    episode_train.legacy.seed_same_sample(args.seed)

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **inference.inference_renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    episode_train.legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.eval()
    renderer.t5_text_encoder.eval()
    renderer.to(device)
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise EGNTCTrainingError("frozen Bernini unexpectedly has trainable parameters")
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=episode_train.legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = episode_train.legacy._vae_statistics(checkpoint)
    scheduler = NoiseScheduler(**episode_train.legacy.noise_scheduler_kwargs())
    raw, parquet_receipt = episode_train._read_episode_parquet(assigned_row)
    support, training_teacher = _materialize_support(
        row=assigned_row,
        raw=raw,
        parquet_receipt=parquet_receipt,
        tokenizer=tokenizer,
        rope=rope,
        vae_mean=vae_mean,
        vae_std=vae_std,
        z_dim=z_dim,
        scheduler=scheduler,
        process_renderer_sample=process_renderer_sample,
        device=device,
        seed=args.seed,
        sigma_index=args.teacher_sigma_index,
    )
    del raw

    parameters = egntc.EGNTCParameters().to(device)
    parameters.eval()
    if sum(item.numel() for item in parameters.parameters()) != TRAINABLE_DIMENSION:
        raise EGNTCTrainingError("EGNTC parameter scope is not exact 36D")
    initial_controller_vector = parameters.flat_tensor(detach=True).clone()
    round_receipts: list[dict[str, Any]] = []
    round_diagnostics: list[dict[str, Any]] = []
    final_cache: Optional[DetachedTrajectoryCache] = None
    wan_diffusion_path = (bernini_root / "bernini/models/wan_diffusion.py").resolve(strict=True)

    for round_spec in plan:
        round_index = int(round_spec["round_index"])
        execution_policy = str(round_spec["execution_policy"])
        input_parameters = parameters if execution_policy == "learned_controller" else None
        cache, rollout_receipt = _capture_rollout(
            renderer=renderer,
            tokenizer=tokenizer,
            support=support,
            parameters=input_parameters,
            execution_policy=execution_policy,
            round_index=round_index,
            rollout_seed=args.rollout_seed,
            device=device,
            prompt_cleaner=prompt_clean,
            bernini_revision=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
        before_sha = _controller_sha256(parameters)
        history, after_sha = _offline_fit(
            cache,
            parameters=parameters,
            teacher=training_teacher,
            device=device,
            steps=int(round_spec["optimizer_steps"]),
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            ulysses_group=parallel.ulysses_group,
        )
        if after_sha == before_sha:
            raise EGNTCTrainingError("offline optimizer did not change the 36D controller")
        postfit_objective = _evaluate_cache_arm(
            cache,
            arm="postfit_controller",
            parameters=parameters,
            controls=None,
            teacher=training_teacher,
            device=device,
        )["loss"]
        round_receipt = {
            **rollout_receipt,
            "optimizer_steps": int(round_spec["optimizer_steps"]),
            "input_parameter_state_sha256": before_sha,
            "output_controller_sha256": after_sha,
            "postfit_offline_objective": postfit_objective,
            "target_used_by_rollout": False,
            "target_used_by_offline_objective": True,
            "objective": "source_relative_Q0_lag_1_2_4_phase_rms_cross_sigma_no_absolute_reconstruction",
        }
        round_receipts.append(round_receipt)
        round_diagnostics.append(
            {
                "round_index": round_index,
                "history": history,
                "controller": parameters.receipt(),
            }
        )
        final_cache = cache

    validate_round_lineage(round_receipts)
    assert final_cache is not None
    local_support_receipt = {
        "iid": support.iid,
        "support_index": parallel.support_index + 1,
        "dp_rank": parallel.support_index,
        "sp_ranks": list(episode_train.SP_GROUP_RANKS[parallel.support_index]),
        "rounds": round_receipts,
        "final_controller_sha256": _controller_sha256(parameters),
        "source_video_sha256": support.source_video_sha256,
        "instruction_sha256": support.instruction_sha256,
        "target_role": "offline_training_only",
        "target_video_sha256": training_teacher.target_video_sha256,
    }
    support_vectors, vector_exchange_digest = _exchange_k2_controller_vectors(
        parameters,
        support_index=parallel.support_index,
        support_iids=support_iids,
        dp_group=parallel.dp_group,
        world_group=parallel.world_group,
    )
    prototype_vector = torch.stack(support_vectors, dim=0).mean(dim=0).contiguous()
    prototype = egntc.EGNTCParameters.from_flat_tensor(prototype_vector).to(device)
    prototype.eval()
    prototype_sha = _assert_sp_controller_exact(
        prototype, ulysses_group=parallel.ulysses_group
    )
    try:
        prototype_consensus = episode_train._world_digest_consensus(
            {"prototype_sha256": prototype_sha, "prototype": prototype_vector},
            world_group=parallel.world_group,
            context="K=2 EGNTC midpoint prototype",
        )
    except episode_train.FewShotCodeTrainingError as error:
        raise EGNTCTrainingError(str(error)) from error

    prototype_evaluation_rollout: Optional[dict[str, Any]] = None
    if not args.engineering_smoke:
        evaluation_cache, prototype_evaluation_rollout = _capture_rollout(
            renderer=renderer,
            tokenizer=tokenizer,
            support=support,
            parameters=prototype,
            execution_policy="learned_controller",
            round_index=3,
            rollout_seed=args.rollout_seed,
            device=device,
            prompt_cleaner=prompt_clean,
            bernini_revision=bernini_revision,
            wan_diffusion_path=wan_diffusion_path,
        )
        if (
            prototype_evaluation_rollout["input_controller_sha256"] != prototype_sha
            or prototype_evaluation_rollout["rollout_id"]
            in {item["rollout_id"] for item in round_receipts}
            or prototype_evaluation_rollout["step_count"] != NUM_INFERENCE_STEPS
            or prototype_evaluation_rollout["transformer_forwards"]
            != FORWARDS_PER_SUPPORT_ROUND
        ):
            raise EGNTCTrainingError(
                "post-refit prototype evaluation is not a fresh full self-rollout"
            )
        final_cache = evaluation_cache
        local_support_receipt["prototype_evaluation_rollout"] = (
            prototype_evaluation_rollout
        )
    else:
        local_support_receipt["prototype_evaluation_rollout"] = None

    local_evaluation: dict[str, Any] = {
        "iid": support.iid,
        "cache_round_index": final_cache.round_index,
        "cache_rollout_id": final_cache.rollout_id,
        "evaluation_is_post_refit_prototype_on_policy": not args.engineering_smoke,
        "round_postfit_objectives": [
            float(item["postfit_offline_objective"]) for item in round_receipts
        ],
        "losses": {},
    }
    evaluation_arms = (
        ("noop", None, None),
        ("raw_action", None, None),
        ("own_support", parameters, None),
        ("prototype", prototype, None),
        ("prototype_phase_reverse", prototype, egntc.EGNTCControls(phase_reverse=True)),
        ("prototype_sigma_shuffle", prototype, egntc.EGNTCControls(sigma_shuffle=True)),
        ("prototype_kappa_off", prototype, egntc.EGNTCControls(kappa_off=True)),
        ("prototype_rho_off", prototype, egntc.EGNTCControls(rho_off=True)),
    )
    for arm, arm_parameters, controls in evaluation_arms:
        local_evaluation["losses"][arm] = _evaluate_cache_arm(
            final_cache,
            arm=arm,
            parameters=arm_parameters,
            controls=controls,
            teacher=training_teacher,
            device=device,
        )["loss"]

    try:
        support_receipts, support_receipt_digest = episode_train._exchange_k2_objects(
            local_support_receipt,
            support_index=parallel.support_index,
            local_iid=support.iid,
            support_iids=support_iids,
            dp_group=parallel.dp_group,
            world_group=parallel.world_group,
            context="ordered K=2 EGNTC support receipts",
        )
        support_evaluations, evaluation_digest = episode_train._exchange_k2_objects(
            local_evaluation,
            support_index=parallel.support_index,
            local_iid=support.iid,
            support_iids=support_iids,
            dp_group=parallel.dp_group,
            world_group=parallel.world_group,
            context="ordered K=2 EGNTC prototype evaluations",
        )
    except episode_train.FewShotCodeTrainingError as error:
        raise EGNTCTrainingError(str(error)) from error
    raw_parameter_cosine = _support_parameter_cosine(support_vectors)
    parameter_cosine = _support_parameter_cosine(
        support_vectors, origin=initial_controller_vector
    )
    gate = prototype_gate(
        support_evaluations,
        support_parameter_cosine=parameter_cosine,
        engineering_smoke=args.engineering_smoke,
    )
    gate["raw_support_parameter_cosine_diagnostic"] = raw_parameter_cosine
    gate["support_parameter_cosine_basis"] = (
        "support_update_vectors_relative_to_shared_initial_36d_state"
    )
    try:
        gate_consensus = episode_train._world_digest_consensus(
            gate, world_group=parallel.world_group, context="EGNTC prototype gate"
        )
    except episode_train.FewShotCodeTrainingError as error:
        raise EGNTCTrainingError(str(error)) from error

    if contract.global_rank == 0:
        diagnostics_path = output_dir / "diagnostics.pt"
        episode_train._atomic_torch_save(
            diagnostics_path,
            {
                "schema_version": DIAGNOSTIC_SCHEMA,
                "engineering_smoke": args.engineering_smoke,
                "support_round_diagnostics": round_diagnostics,
                "support_evaluations": support_evaluations,
                "gate": gate,
            },
        )
        controller_path = output_dir / "controller.safetensors"
        checkpoint_receipt = _atomic_save_controller(controller_path, prototype_vector)
        run_receipt: dict[str, Any] = {
            "schema_version": RUN_RECEIPT_SCHEMA,
            "method": METHOD_NAME,
            "engineering_smoke": args.engineering_smoke,
            "representability_gate": gate["representability_gate"],
            "deployable": gate["deployable"],
            "diagnostic_only": gate["diagnostic_only"],
            "gate": gate,
            "exact_geometry": {
                "rgb_frames": NUM_FRAMES,
                "latent_phases": LATENT_PHASES,
                "unipc_steps_per_round": NUM_INFERENCE_STEPS,
                "transformer_forwards_per_support_round": FORWARDS_PER_SUPPORT_ROUND,
                "rounds": len(plan),
                "evaluation_only_full_rollouts": 0 if args.engineering_smoke else 1,
                "total_transformer_forwards_per_support": (
                    len(plan) + (0 if args.engineering_smoke else 1)
                )
                * FORWARDS_PER_SUPPORT_ROUND,
            },
            "round_plan": list(plan),
            "fresh_round_two_self_rollout": not args.engineering_smoke,
            "post_refit_prototype_evaluation_rollout": (
                prototype_evaluation_rollout is not None
            ),
            "external_inference_conditions": list(INFERENCE_CONDITIONS),
            "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
            "target_role": "offline_source_relative_motion_objective_only",
            "full_target_reconstruction_weight": 0.0,
            "support_iids": support_iids,
            "support": support_receipts,
            "support_evaluations": support_evaluations,
            "prototype": {
                **prototype.receipt(),
                **checkpoint_receipt,
                "construction": "arithmetic_midpoint_of_two_ordered_support_36d_vectors",
            },
            "episode_audit": episode.audit_receipt(),
            "distributed": distributed_receipt,
            "controller_contract": egntc.controller_contract(),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers_version,
                "diffusers": diffusers_version,
                "bernini_commit": bernini_revision,
                "veomni_commit": veomni_revision,
                "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": args.method_source_archive_sha256,
            },
            "source_file_sha256": _source_hashes(),
            "consensus": {
                "support_vectors": vector_exchange_digest,
                "support_receipts": support_receipt_digest,
                "prototype": prototype_consensus,
                "support_evaluations": evaluation_digest,
                "gate": gate_consensus,
            },
            "diagnostics_sha256": episode_train.file_sha256(diagnostics_path),
        }
        run_receipt["receipt_sha256"] = episode_train.object_sha256(run_receipt)
        run_path = output_dir / "run.receipt.json"
        episode_train._atomic_write_json(run_path, run_receipt)
        companion = _write_controller_companion_receipt(
            output_dir=output_dir,
            checkpoint=checkpoint_receipt,
            vector=prototype_vector,
            gate=gate,
            run_receipt_sha256=episode_train.file_sha256(run_path),
            support_iids=support_iids,
        )
        provenance = companion.get("training_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("deployable") != bool(gate["deployable"])
        ):
            raise EGNTCTrainingError("inference companion receipt changed deployability")

    dist.barrier(group=parallel.world_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""v16r5 exact644 training with a capped dual-descent half-space update.

This is a thin runtime wrapper around v16r4.  Data, self-generated same-IID
donors, routed-teacher objectives, the 1e-6 active-coordinate RMS learning
rate, and the stateless global-RMS SGD optimizer are unchanged.  Only the
two-gradient geometry and a deterministic near-antipodal step attenuation are
new:

* ``q = min(1, max(0.01, -cos(action, replay) + 0.01))``;
* the antipodal gap and both normalized formal margins must exceed 1e-6;
* actual action/source descent cosines must both exceed 1e-8;
* when the antipodal gap is below 0.01, the global optimizer step is multiplied
  by ``sqrt(gap / 0.01)`` (the gradient direction itself is not changed).

There is no retry, action-only fallback, optimizer reset, row skip, or manual
per-sample admission path.  The inherited automatic decoded-canary contract
remains the only checkpoint-promotion gate.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r4 as parent


METHOD = "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r5"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r5"
)
REPLAY_COMBINE_MODE = parent.REPLAY_COMBINE_MODE
LEARNING_RATE = parent.LEARNING_RATE
MAX_GRAD_NORM = parent.MAX_GRAD_NORM
SEED = parent.SEED
OPTIMIZER = "global_rms_normalized_capped_halfspace_sgd_v16r5"
OPTIMIZER_FAILURE_POLICY = (
    "fail_closed_no_retry_no_action_only_fallback_no_optimizer_state_reset_v16r5"
)
DIRECTION_POLICY = "capped_source_halfspace_q_min001_max1_v16r5"
Q_MIN = 0.01
Q_MAX = 1.0
SOURCE_MARGIN = 0.01
MIN_ANTIPODAL_GAP = 1.0e-6
MIN_FORMAL_NORMALIZED_MARGIN = 1.0e-6
MIN_ACTUAL_DESCENT_COSINE = 1.0e-8
ATTENUATION_TRIGGER_GAP = 0.01
ATTENUATION_POLICY = "sqrt_antipodal_gap_over_001_below_001_v16r5"
DECODED_CANARY_SCHEMA = "bernini-v16r5-heldout8-checkpoint-canary-contract-v1"


base = parent.base
v16 = parent.v16
v16r3 = parent.v16r3
r2 = parent.r2
v15 = parent.v15

_PARENT_BUILD_PARSER = parent.build_parser
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def _empty_v16r5_audit() -> dict[str, Any]:
    return {
        "formal_steps": [],
        "actual_steps": [],
        "attenuated_steps": [],
        "minimum_antipodal_gap": None,
        "minimum_action_normalized_margin": None,
        "minimum_source_normalized_margin": None,
        "minimum_actual_action_descent_cosine": None,
        "minimum_actual_source_descent_cosine": None,
        "last_formal": None,
        "last_actual": None,
    }


_V16R5_AUDIT = _empty_v16r5_audit()
_PENDING_STEP_GEOMETRY: Optional[dict[str, Any]] = None


def fail(message: str) -> None:
    base.fail(message)


def _minimum(current: Optional[float], value: float) -> float:
    return value if current is None else min(float(current), value)


def _direction_plan(
    *, action_sq: float, replay_sq: float, raw_dot: float
) -> dict[str, float | bool | str]:
    """Return the exact fp64 v16r5 two-gradient geometry before mutation."""

    values = (float(action_sq), float(replay_sq), float(raw_dot))
    if not all(math.isfinite(value) for value in values):
        fail("v16r5 direction geometry is non-finite")
    if action_sq <= 0.0 or replay_sq <= 0.0:
        fail("v16r5 requires two nonzero component gradients")

    action_norm = math.sqrt(action_sq)
    replay_norm = math.sqrt(replay_sq)
    cosine = raw_dot / (action_norm * replay_norm)
    cosine = max(-1.0, min(1.0, cosine))
    antipodal_gap = 1.0 + cosine
    requested_q = max(Q_MIN, -cosine + SOURCE_MARGIN)
    q = min(Q_MAX, requested_q)
    action_margin = 1.0 + q * cosine
    source_margin = cosine + q
    combined_ratio_sq = 1.0 + 2.0 * q * cosine + q * q
    combined_ratio = math.sqrt(max(0.0, combined_ratio_sq))

    if antipodal_gap <= MIN_ANTIPODAL_GAP:
        fail(
            "v16r5 antipodal gap is infeasible: "
            f"gap={antipodal_gap!r}, minimum={MIN_ANTIPODAL_GAP!r}, "
            f"cosine={cosine!r}, q={q!r}"
        )
    if (
        action_margin <= MIN_FORMAL_NORMALIZED_MARGIN
        or source_margin <= MIN_FORMAL_NORMALIZED_MARGIN
    ):
        fail(
            "v16r5 normalized formal dual-descent margin is infeasible: "
            f"action_margin={action_margin!r}, "
            f"source_margin={source_margin!r}, "
            f"minimum={MIN_FORMAL_NORMALIZED_MARGIN!r}, "
            f"cosine={cosine!r}, q={q!r}"
        )
    if combined_ratio <= 0.0 or not math.isfinite(combined_ratio):
        fail("v16r5 combined gradient norm is infeasible")

    action_cosine = action_margin / combined_ratio
    source_cosine = source_margin / combined_ratio
    if (
        action_cosine <= MIN_ACTUAL_DESCENT_COSINE
        or source_cosine <= MIN_ACTUAL_DESCENT_COSINE
    ):
        fail("v16r5 planned dual-descent cosine is infeasible")

    attenuation_gamma = (
        math.sqrt(antipodal_gap / ATTENUATION_TRIGGER_GAP)
        if antipodal_gap < ATTENUATION_TRIGGER_GAP
        else 1.0
    )
    if not 0.0 < attenuation_gamma <= 1.0:
        fail("v16r5 optimizer attenuation is invalid")

    return {
        "policy": DIRECTION_POLICY,
        "action_norm": action_norm,
        "replay_norm": replay_norm,
        "raw_dot": raw_dot,
        "cosine": cosine,
        "antipodal_gap": antipodal_gap,
        "requested_q": requested_q,
        "q": q,
        "q_cap_applied": requested_q > Q_MAX,
        "effective_replay_scale": q * action_norm / replay_norm,
        "action_normalized_margin": action_margin,
        "source_normalized_margin": source_margin,
        "combined_norm_ratio_to_action": combined_ratio,
        "planned_action_descent_cosine": action_cosine,
        "planned_source_descent_cosine": source_cosine,
        "step_attenuation_gamma": attenuation_gamma,
        "step_attenuation_applied": attenuation_gamma < 1.0,
    }


def build_parser() -> argparse.ArgumentParser:
    return _PARENT_BUILD_PARSER()


def validate_args(args: argparse.Namespace) -> None:
    """Retain every v16r4 argument contract under a fresh v16r5 namespace."""

    shadow = argparse.Namespace(**vars(args))
    shadow.replay_combine_mode = v15.REPLAY_COMBINE_MODE
    shadow.learning_rate = 1.0e-5
    parent._V16_VALIDATE_ARGS(shadow)

    if getattr(args, "replay_combine_mode", None) != REPLAY_COMBINE_MODE:
        fail(f"v16r5 requires --replay-combine-mode={REPLAY_COMBINE_MODE}")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r5 requires --learning-rate=1e-6")
    if float(getattr(args, "max_grad_norm", float("nan"))) != MAX_GRAD_NORM:
        fail("v16r5 requires the inherited max gradient norm 10")
    if int(getattr(args, "max_steps", -1)) != v16.FULL644_ROWS:
        fail("v16r5 requires one continuous exact644 optimizer run")
    if int(getattr(args, "seed", -1)) != SEED:
        fail(f"v16r5 requires the unchanged v16r4 seed {SEED}")
    if bool(getattr(args, "gradient_diagnostic_only", False)):
        fail("v16r5 is an optimizer run, not gradient-diagnostic-only")
    if "v16r5" not in str(Path(args.output)).lower():
        fail("v16r5 output path must carry an explicit v16r5 namespace")

    v16r3._validate_zero_rms_operator()
    parent._CANARY_BINDING = parent._load_decoded_canary_binding(
        args.decoded_canary_manifest,
        args.decoded_canary_manifest_sha256,
    )


def merge_component_gradients(
    named: Sequence[tuple[str, Any]],
    action_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    base_replay_scale: float,
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    """Apply the capped v16r5 source-halfspace direction in place."""

    global _PENDING_STEP_GEOMETRY

    import torch

    if replay_combine_mode != REPLAY_COMBINE_MODE:
        fail("v16r5 merge is restricted to source_halfspace_001")
    if not named or len(named) != len(action_gradients):
        fail("v16r5 component-gradient merge closure differs")
    if not 0.0 < float(base_replay_scale) <= 1.0:
        fail("v16r5 base replay scale is outside (0,1]")
    if _PENDING_STEP_GEOMETRY is not None:
        fail("v16r5 previous formal geometry was not consumed by the optimizer")

    first_gradient = named[0][1].grad
    if first_gradient is None:
        fail("v16r5 component-gradient merge has no replay gradient")
    action_sq = torch.zeros((), dtype=torch.float64, device=first_gradient.device)
    replay_sq = torch.zeros_like(action_sq)
    raw_dot = torch.zeros_like(action_sq)
    for (name, parameter), action in zip(named, action_gradients):
        replay = parameter.grad
        if replay is None or tuple(replay.shape) != tuple(action.shape):
            fail(f"v16r5 component-gradient geometry differs: {name}")
        if not bool(torch.isfinite(action).all().item()):
            fail(f"v16r5 action component gradient is non-finite: {name}")
        if not bool(torch.isfinite(replay).all().item()):
            fail(f"v16r5 replay component gradient is non-finite: {name}")
        action64 = action.detach().double()
        replay64 = replay.detach().double()
        action_sq += action64.square().sum()
        replay_sq += replay64.square().sum()
        raw_dot += (action64 * replay64).sum()

    local_error: Optional[base.OnlineAnchorTrainingError] = None
    plan: Optional[dict[str, Any]] = None
    try:
        plan = dict(
            _direction_plan(
                action_sq=float(action_sq.item()),
                replay_sq=float(replay_sq.item()),
                raw_dot=float(raw_dot.item()),
            )
        )
        if diagnostic_only:
            fail(
                "V16R5_GRADIENT_DIAGNOSTIC_COMPLETE|optimizer_steps=0|"
                f"cosine={plan['cosine']!r}|q={plan['q']!r}|"
                f"antipodal_gap={plan['antipodal_gap']!r}|"
                f"action_margin={plan['action_normalized_margin']!r}|"
                f"source_margin={plan['source_normalized_margin']!r}|"
                f"attenuation_gamma={plan['step_attenuation_gamma']!r}"
            )
    except base.OnlineAnchorTrainingError as error:
        local_error = error
    passed = parent._collective_pass_or_failure(
        local_error is None,
        device=first_gradient.device,
        phase="v16r5 formal capped source-halfspace merge",
    )
    if not passed:
        if local_error is None:
            fail("v16r5 formal merge failed without a local error")
        raise local_error
    if plan is None:
        fail("v16r5 formal merge passed without a direction plan")

    replay_scale = float(plan["effective_replay_scale"])
    combined_sq = torch.zeros_like(action_sq)
    action_inner = torch.zeros_like(action_sq)
    replay_inner = torch.zeros_like(action_sq)
    for (_name, parameter), action in zip(named, action_gradients):
        replay64 = parameter.grad.detach().double()
        action64 = action.detach().double()
        parameter.grad.mul_(replay_scale).add_(action)
        combined64 = parameter.grad.detach().double()
        combined_sq += combined64.square().sum()
        action_inner += (action64 * combined64).sum()
        replay_inner += (replay64 * combined64).sum()

    combined_norm = math.sqrt(float(combined_sq.item()))
    action_norm = float(plan["action_norm"])
    replay_norm = float(plan["replay_norm"])
    action_inner_value = float(action_inner.item())
    replay_inner_value = float(replay_inner.item())
    actual_action_margin = action_inner_value / float(action_sq.item())
    actual_source_margin = replay_inner_value / (action_norm * replay_norm)
    actual_action_cosine = action_inner_value / (action_norm * combined_norm)
    actual_source_cosine = replay_inner_value / (replay_norm * combined_norm)
    local_error = None
    try:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    combined_norm,
                    actual_action_margin,
                    actual_source_margin,
                    actual_action_cosine,
                    actual_source_cosine,
                )
            )
            or combined_norm <= 0.0
            or actual_action_margin <= MIN_FORMAL_NORMALIZED_MARGIN
            or actual_source_margin <= MIN_FORMAL_NORMALIZED_MARGIN
            or actual_action_cosine <= MIN_ACTUAL_DESCENT_COSINE
            or actual_source_cosine <= MIN_ACTUAL_DESCENT_COSINE
        ):
            fail(
                "v16r5 stored merged gradient lost dual descent: "
                f"action_margin={actual_action_margin!r}, "
                f"source_margin={actual_source_margin!r}, "
                f"action_cosine={actual_action_cosine!r}, "
                f"source_cosine={actual_source_cosine!r}"
            )
    except base.OnlineAnchorTrainingError as error:
        local_error = error
    passed = parent._collective_pass_or_failure(
        local_error is None,
        device=first_gradient.device,
        phase="v16r5 stored merged-gradient dual descent",
    )
    if not passed:
        if local_error is None:
            fail("v16r5 stored merge failed without a local error")
        raise local_error

    q = float(plan["q"])
    planned_combined_norm = action_norm * float(
        plan["combined_norm_ratio_to_action"]
    )
    values = {
        "action_l2_norm_fp64": action_norm,
        "raw_replay_l2_norm_fp64": replay_norm,
        "processed_replay_l2_norm_fp64": replay_norm,
        "weighted_replay_l2_norm_fp64": q * action_norm,
        "combined_l2_norm_fp64": combined_norm,
        "planned_combined_l2_norm_fp64": planned_combined_norm,
        "action_raw_replay_dot_fp64": float(plan["raw_dot"]),
        "action_replay_cosine": float(plan["cosine"]),
        "replay_combine_mode": REPLAY_COMBINE_MODE,
        "base_replay_scale": float(base_replay_scale),
        "first_order_safe_lambda_min": max(
            0.0, -float(plan["raw_dot"]) / float(replay_sq.item())
        ),
        "effective_replay_scale": replay_scale,
        "weighted_replay_gradient_fraction": q / (1.0 + q),
        "weighted_replay_to_action_grad_norm_ratio": q,
        "replay_component_to_action_norm_ratio_q": q,
        "correction_ratio_q": q,
        "replay_projection_applied": False,
        "replay_projection_coefficient": 0.0,
        "processed_replay_retained_raw_norm_fraction": 1.0,
        "processed_replay_action_cosine": float(plan["cosine"]),
        "action_priority_conflict_control_not_source_preservation": False,
        "action_gradient_dot_combined_gradient_fp64": action_inner_value,
        "planned_action_gradient_dot_combined_gradient_fp64": (
            float(action_sq.item()) * float(plan["action_normalized_margin"])
        ),
        "action_alignment_ratio": actual_action_margin,
        "action_combined_cosine": actual_action_cosine,
        "raw_replay_gradient_dot_combined_gradient_fp64": replay_inner_value,
        "raw_replay_combined_alignment_over_action_replay_norms": (
            actual_source_margin
        ),
        "planned_raw_replay_gradient_dot_combined_gradient_fp64": (
            action_norm
            * replay_norm
            * float(plan["source_normalized_margin"])
        ),
        "raw_replay_combined_cosine": actual_source_cosine,
        "processed_replay_gradient_dot_combined_gradient_fp64": replay_inner_value,
        "planned_processed_replay_gradient_dot_combined_gradient_fp64": (
            action_norm
            * replay_norm
            * float(plan["source_normalized_margin"])
        ),
        "raw_source_fm_gradient_dot_combined_gradient_fp64": replay_inner_value,
        "first_order_source_fm_preserved": True,
        "v16r4_source_descent_required": True,
        "v16r4_action_descent_required": True,
        "v16r4_action_only_fallback_allowed": False,
        "v16r4_optimizer_state_reset_allowed": False,
        "v16r5_direction_policy": DIRECTION_POLICY,
        "v16r5_requested_correction_ratio_q": float(plan["requested_q"]),
        "v16r5_q_min": Q_MIN,
        "v16r5_q_max": Q_MAX,
        "v16r5_q_cap_applied": bool(plan["q_cap_applied"]),
        "v16r5_antipodal_gap": float(plan["antipodal_gap"]),
        "v16r5_minimum_antipodal_gap_exclusive": MIN_ANTIPODAL_GAP,
        "v16r5_action_normalized_margin": actual_action_margin,
        "v16r5_source_normalized_margin": actual_source_margin,
        "v16r5_minimum_formal_normalized_margin_exclusive": (
            MIN_FORMAL_NORMALIZED_MARGIN
        ),
        "v16r5_planned_action_descent_cosine": float(
            plan["planned_action_descent_cosine"]
        ),
        "v16r5_planned_source_descent_cosine": float(
            plan["planned_source_descent_cosine"]
        ),
        "v16r5_actual_merged_action_descent_cosine": actual_action_cosine,
        "v16r5_actual_merged_source_descent_cosine": actual_source_cosine,
        "v16r5_minimum_actual_descent_cosine_exclusive": (
            MIN_ACTUAL_DESCENT_COSINE
        ),
        "v16r5_step_attenuation_policy": ATTENUATION_POLICY,
        "v16r5_step_attenuation_trigger_gap": ATTENUATION_TRIGGER_GAP,
        "v16r5_step_attenuation_gamma": float(
            plan["step_attenuation_gamma"]
        ),
        "v16r5_step_attenuation_applied": bool(
            plan["step_attenuation_applied"]
        ),
        "v16r5_source_descent_required": True,
        "v16r5_action_descent_required": True,
        "v16r5_action_only_fallback_allowed": False,
        "v16r5_optimizer_state_reset_allowed": False,
    }
    if not all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        fail("v16r5 component-gradient interaction is non-finite")

    step = int(parent._RUNTIME_AUDIT["formal_merge_count"]) + 1
    _PENDING_STEP_GEOMETRY = {
        "step": step,
        "antipodal_gap": values["v16r5_antipodal_gap"],
        "step_attenuation_gamma": values["v16r5_step_attenuation_gamma"],
        "step_attenuation_applied": values["v16r5_step_attenuation_applied"],
    }
    _V16R5_AUDIT["formal_steps"].append(step)
    if values["v16r5_step_attenuation_applied"]:
        _V16R5_AUDIT["attenuated_steps"].append(step)
    _V16R5_AUDIT["minimum_antipodal_gap"] = _minimum(
        _V16R5_AUDIT["minimum_antipodal_gap"],
        float(values["v16r5_antipodal_gap"]),
    )
    _V16R5_AUDIT["minimum_action_normalized_margin"] = _minimum(
        _V16R5_AUDIT["minimum_action_normalized_margin"], actual_action_margin
    )
    _V16R5_AUDIT["minimum_source_normalized_margin"] = _minimum(
        _V16R5_AUDIT["minimum_source_normalized_margin"], actual_source_margin
    )
    _V16R5_AUDIT["last_formal"] = dict(values)
    parent._RUNTIME_AUDIT["formal_merge_count"] += 1
    parent._RUNTIME_AUDIT["last_formal_merge"] = dict(values)
    return values


def _make_attenuated_global_rms_sgd(
    parameters: Sequence[Any], *, lr: float
) -> Any:
    """Build the v16r4 stateless optimizer with one audited scalar attenuation."""

    import torch

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        fail("v16r5 projected optimizer has no parameter")
    if float(lr) != LEARNING_RATE:
        fail("v16r5 projected optimizer learning rate differs")

    class AttenuatedGlobalRMSProjectedSGD(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(parameter_tuple, {"lr": float(lr)})
            self._v16r4_step_count = 0
            self._v16r4_last_step: Optional[dict[str, Any]] = None
            self._v16r5_step_count = 0
            self._v16r5_last_step: Optional[dict[str, Any]] = None

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:
            global _PENDING_STEP_GEOMETRY

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            if len(self.param_groups) != 1:
                fail("v16r5 projected optimizer parameter-group count differs")
            group = self.param_groups[0]
            if float(group.get("lr", float("nan"))) != LEARNING_RATE:
                fail("v16r5 projected optimizer live learning rate differs")
            pending = _PENDING_STEP_GEOMETRY
            expected_step = self._v16r5_step_count + 1
            if pending is None or int(pending.get("step", -1)) != expected_step:
                fail("v16r5 optimizer has no matching formal geometry")
            gamma = float(pending["step_attenuation_gamma"])
            if not 0.0 < gamma <= 1.0:
                fail("v16r5 optimizer attenuation gamma differs")

            gradients: list[tuple[Any, Any]] = []
            first_device = None
            gradient_sq = None
            active_elements = None
            total_elements = 0
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    fail("v16r5 projected optimizer gradient is absent")
                if gradient.is_sparse:
                    fail("v16r5 projected optimizer forbids sparse gradients")
                if not bool(torch.isfinite(gradient).all().item()):
                    fail("v16r5 projected optimizer gradient is non-finite")
                if first_device is None:
                    first_device = gradient.device
                    gradient_sq = torch.zeros(
                        (), dtype=torch.float64, device=first_device
                    )
                    active_elements = torch.zeros(
                        (), dtype=torch.int64, device=first_device
                    )
                if gradient.device != first_device:
                    fail("v16r5 projected optimizer gradient device differs")
                gradient_sq += gradient.detach().double().square().sum()
                active_elements += gradient.detach().count_nonzero().to(
                    dtype=torch.int64
                )
                total_elements += int(gradient.numel())
                gradients.append((parameter, gradient))

            if gradient_sq is None or active_elements is None:
                fail("v16r5 projected optimizer gradient closure is empty")
            gradient_norm = math.sqrt(float(gradient_sq.item()))
            active_count = int(active_elements.item())
            if gradient_norm <= 0.0 or active_count <= 0:
                fail("v16r5 projected optimizer requires a nonzero gradient")
            unattenuated_delta_norm = LEARNING_RATE * math.sqrt(active_count)
            planned_delta_norm = gamma * unattenuated_delta_norm
            global_scale = planned_delta_norm / gradient_norm
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (
                    gradient_norm,
                    unattenuated_delta_norm,
                    planned_delta_norm,
                    global_scale,
                )
            ):
                fail("v16r5 projected optimizer scale is invalid")

            for parameter, gradient in gradients:
                parameter.add_(gradient, alpha=-global_scale)
            self._v16r4_step_count += 1
            self._v16r5_step_count += 1
            optimizer_step = {
                "schema_version": "bernini-global-rms-projected-sgd-step-v1",
                "step": int(self._v16r5_step_count),
                "optimizer": OPTIMIZER,
                "optimizer_scalar_learning_rate": LEARNING_RATE,
                "learning_rate_active_coordinate_rms_before_attenuation": (
                    LEARNING_RATE
                ),
                "learning_rate_active_coordinate_rms": LEARNING_RATE * gamma,
                "gradient_l2_norm_fp64": gradient_norm,
                "active_gradient_element_count": active_count,
                "total_parameter_element_count": total_elements,
                "unattenuated_planned_delta_theta_l2_norm_fp64": (
                    unattenuated_delta_norm
                ),
                "planned_delta_theta_l2_norm_fp64": planned_delta_norm,
                "global_positive_direction_scale": global_scale,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "coordinatewise_preconditioner": False,
                "global_gradient_direction_preserved_before_storage_rounding": True,
                "v16r5_step_attenuation_policy": ATTENUATION_POLICY,
                "v16r5_antipodal_gap": float(pending["antipodal_gap"]),
                "v16r5_step_attenuation_trigger_gap": ATTENUATION_TRIGGER_GAP,
                "v16r5_step_attenuation_gamma": gamma,
                "v16r5_step_attenuation_applied": bool(
                    pending["step_attenuation_applied"]
                ),
            }
            self._v16r4_last_step = dict(optimizer_step)
            self._v16r5_last_step = dict(optimizer_step)
            _PENDING_STEP_GEOMETRY = None
            return loss

    return AttenuatedGlobalRMSProjectedSGD()


def _projected_optimizer_factory() -> Any:
    def factory(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        if parent._ACTIVE_OPTIMIZER is not None:
            fail("v16r5 expected exactly one optimizer construction")
        if args:
            fail("v16r5 projected optimizer forbids positional AdamW options")
        options = dict(kwargs)
        lr = float(options.pop("lr", float("nan")))
        weight_decay = float(options.pop("weight_decay", float("nan")))
        if options:
            fail("v16r5 projected optimizer received unsupported AdamW options")
        if lr != LEARNING_RATE or weight_decay != 0.0:
            fail("v16r5 projected optimizer construction differs")
        optimizer = _make_attenuated_global_rms_sgd(tuple(parameters), lr=lr)
        parent._ACTIVE_OPTIMIZER = optimizer
        return optimizer

    return factory


def actual_optimizer_update_probe(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    step: int,
) -> Mapping[str, Any]:
    """Require the stored attenuated displacement to descend both objectives."""

    if replay_combine_mode != REPLAY_COMBINE_MODE or not named:
        fail("v16r5 actual-update probe closure differs")
    local_error: Optional[base.OnlineAnchorTrainingError] = None
    values: Optional[dict[str, Any]] = None
    expected_step = len(parent._RUNTIME_AUDIT["actual_update_steps"]) + 1
    try:
        values = dict(
            parent._BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE(
                named,
                parameter_values_before_step,
                action_gradients,
                raw_replay_gradients,
                replay_combine_mode=replay_combine_mode,
                step=step,
            )
        )
        if int(step) != expected_step:
            fail("v16r5 actual-update probe step sequence differs")
        action_cosine = float(values.get("action_descent_cosine", float("nan")))
        source_cosine = float(values.get("source_descent_cosine", float("nan")))
        if (
            values.get("optimizer_semantics_observed_not_modified") is not True
            or values.get("action_descent_required") is not True
            or values.get("action_descent_passed") is not True
            or values.get("source_descent_required") is not True
            or values.get("source_descent_passed") is not True
            or not math.isfinite(action_cosine)
            or not math.isfinite(source_cosine)
            or action_cosine <= MIN_ACTUAL_DESCENT_COSINE
            or source_cosine <= MIN_ACTUAL_DESCENT_COSINE
        ):
            fail("v16r5 actual optimizer dual-descent cosine closure differs")

        optimizer = parent._ACTIVE_OPTIMIZER
        optimizer_step = getattr(optimizer, "_v16r5_last_step", None)
        if (
            optimizer is None
            or not isinstance(optimizer_step, Mapping)
            or int(getattr(optimizer, "_v16r5_step_count", -1)) != int(step)
            or int(getattr(optimizer, "_v16r4_step_count", -1)) != int(step)
            or int(optimizer_step.get("step", -1)) != int(step)
            or optimizer_step.get("optimizer") != OPTIMIZER
            or len(optimizer.state) != 0
        ):
            fail("v16r5 projected optimizer runtime closure differs")
        planned_delta = float(
            optimizer_step.get("planned_delta_theta_l2_norm_fp64", float("nan"))
        )
        actual_delta = float(values.get("delta_theta_l2_norm_fp64", float("nan")))
        relative_delta_error = abs(actual_delta - planned_delta) / planned_delta
        if (
            not math.isfinite(relative_delta_error)
            or relative_delta_error > 1.0e-3
        ):
            fail(
                "v16r5 stored displacement differs from its attenuated plan: "
                f"actual={actual_delta!r}, planned={planned_delta!r}, "
                f"relative_error={relative_delta_error!r}"
            )
        formal = _V16R5_AUDIT.get("last_formal")
        if (
            not isinstance(formal, Mapping)
            or float(optimizer_step["v16r5_step_attenuation_gamma"])
            != float(formal["v16r5_step_attenuation_gamma"])
            or float(optimizer_step["v16r5_antipodal_gap"])
            != float(formal["v16r5_antipodal_gap"])
        ):
            fail("v16r5 optimizer attenuation/formal-geometry binding differs")
        values.update(
            {
                "v16r4_optimizer": OPTIMIZER,
                "v16r4_optimizer_step": dict(optimizer_step),
                "v16r4_actual_to_planned_delta_l2_ratio": (
                    actual_delta / planned_delta
                ),
                "v16r4_actual_vs_planned_delta_l2_relative_error": (
                    relative_delta_error
                ),
                "v16r4_optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
                "v16r4_probe_retry_count": 0,
                "v16r4_action_only_fallback_applied": False,
                "v16r4_optimizer_state_reset": False,
                "v16r4_failed_candidate_checkpoint_publication_allowed": False,
                "v16r5_optimizer": OPTIMIZER,
                "v16r5_optimizer_step": dict(optimizer_step),
                "v16r5_actual_to_planned_delta_l2_ratio": (
                    actual_delta / planned_delta
                ),
                "v16r5_actual_vs_planned_delta_l2_relative_error": (
                    relative_delta_error
                ),
                "v16r5_optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
                "v16r5_probe_retry_count": 0,
                "v16r5_action_only_fallback_applied": False,
                "v16r5_optimizer_state_reset": False,
                "v16r5_failed_candidate_checkpoint_publication_allowed": False,
                "v16r5_minimum_actual_descent_cosine_exclusive": (
                    MIN_ACTUAL_DESCENT_COSINE
                ),
                "v16r5_step_attenuation_gamma": float(
                    optimizer_step["v16r5_step_attenuation_gamma"]
                ),
                "v16r5_step_attenuation_applied": bool(
                    optimizer_step["v16r5_step_attenuation_applied"]
                ),
            }
        )
    except base.OnlineAnchorTrainingError as error:
        local_error = error
    passed = parent._collective_pass_or_failure(
        local_error is None,
        device=named[0][1].device,
        phase="v16r5 actual attenuated dual-descent update",
    )
    if not passed:
        parent._RUNTIME_AUDIT["failed_actual_probe_steps"].append(int(step))
        if local_error is None:
            fail("v16r5 actual probe failed without a local error")
        raise local_error
    if values is None:
        fail("v16r5 actual probe passed without values")

    action_cosine = float(values["action_descent_cosine"])
    source_cosine = float(values["source_descent_cosine"])
    _V16R5_AUDIT["actual_steps"].append(int(step))
    _V16R5_AUDIT["minimum_actual_action_descent_cosine"] = _minimum(
        _V16R5_AUDIT["minimum_actual_action_descent_cosine"], action_cosine
    )
    _V16R5_AUDIT["minimum_actual_source_descent_cosine"] = _minimum(
        _V16R5_AUDIT["minimum_actual_source_descent_cosine"], source_cosine
    )
    _V16R5_AUDIT["last_actual"] = dict(values)
    parent._RUNTIME_AUDIT["actual_update_steps"].append(int(step))
    parent._RUNTIME_AUDIT["last_actual_update"] = dict(values)
    parent._RUNTIME_AUDIT["optimizer_step_count"] = int(
        getattr(parent._ACTIVE_OPTIMIZER, "_v16r4_step_count")
    )
    parent._RUNTIME_AUDIT["last_optimizer_step"] = dict(
        getattr(parent._ACTIVE_OPTIMIZER, "_v16r4_last_step")
    )
    return values


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    """Extend the inherited exact644 receipt with v16r5 geometry history."""

    receipt = _PARENT_CHECKPOINT_RECEIPT(**kwargs)
    step = int(receipt.get("global_step", 0))
    contract = receipt.get("training_contract")
    expected_steps = list(range(1, step + 1))
    if (
        not isinstance(contract, MutableMapping)
        or step <= 0
        or _V16R5_AUDIT["formal_steps"] != expected_steps
        or _V16R5_AUDIT["actual_steps"] != expected_steps
        or _PENDING_STEP_GEOMETRY is not None
    ):
        fail("v16r5 receipt update accounting differs")

    source = receipt.pop("v16r4_source_descent_summary", None)
    canary = receipt.pop("v16r4_decoded_canary_contract", None)
    if not isinstance(source, MutableMapping) or not isinstance(
        canary, MutableMapping
    ):
        fail("v16r5 inherited receipt sections differ")
    attenuated_steps = list(_V16R5_AUDIT["attenuated_steps"])
    minimum_gap = float(_V16R5_AUDIT["minimum_antipodal_gap"])
    minimum_gamma = (
        math.sqrt(minimum_gap / ATTENUATION_TRIGGER_GAP)
        if minimum_gap < ATTENUATION_TRIGGER_GAP
        else 1.0
    )
    current_gamma = float(
        _V16R5_AUDIT["last_formal"]["v16r5_step_attenuation_gamma"]
    )
    attenuation = {
        "policy": ATTENUATION_POLICY,
        "trigger_antipodal_gap_strictly_below": ATTENUATION_TRIGGER_GAP,
        "gamma_formula": "sqrt(antipodal_gap/0.01)",
        "gamma_when_not_triggered": 1.0,
        "applied_step_count": len(attenuated_steps),
        "applied_steps": attenuated_steps,
        "applied_steps_sha256": base.legacy.object_sha256(attenuated_steps),
        "minimum_observed_antipodal_gap": minimum_gap,
        "minimum_observed_gamma": minimum_gamma,
        "current_step_gamma": current_gamma,
        "unattenuated_base_active_coordinate_rms": LEARNING_RATE,
        "minimum_effective_active_coordinate_rms": (
            LEARNING_RATE * minimum_gamma
        ),
        "current_effective_active_coordinate_rms": LEARNING_RATE * current_gamma,
        "changes_gradient_direction": False,
        "changes_only_global_positive_step_scale": True,
    }
    source.update(
        {
            "replay_combine_mode": REPLAY_COMBINE_MODE,
            "direction_policy": DIRECTION_POLICY,
            "q_formula": "min(1,max(0.01,-action_replay_cosine+0.01))",
            "q_min": Q_MIN,
            "q_max": Q_MAX,
            "source_margin": SOURCE_MARGIN,
            "minimum_antipodal_gap_exclusive": MIN_ANTIPODAL_GAP,
            "minimum_formal_normalized_margin_exclusive": (
                MIN_FORMAL_NORMALIZED_MARGIN
            ),
            "minimum_actual_descent_cosine_exclusive": (
                MIN_ACTUAL_DESCENT_COSINE
            ),
            "minimum_observed_action_normalized_margin": _V16R5_AUDIT[
                "minimum_action_normalized_margin"
            ],
            "minimum_observed_source_normalized_margin": _V16R5_AUDIT[
                "minimum_source_normalized_margin"
            ],
            "minimum_observed_actual_action_descent_cosine": _V16R5_AUDIT[
                "minimum_actual_action_descent_cosine"
            ],
            "minimum_observed_actual_source_descent_cosine": _V16R5_AUDIT[
                "minimum_actual_source_descent_cosine"
            ],
            "near_antipodal_global_step_attenuation": attenuation,
            "optimizer": OPTIMIZER,
            "optimizer_scalar_learning_rate": LEARNING_RATE,
            "optimizer_scalar_learning_rate_semantics": (
                "unattenuated_base_active_coordinate_rms_before_v16r5_gap_attenuation"
            ),
            "optimizer_unattenuated_base_active_coordinate_rms": LEARNING_RATE,
            "optimizer_current_effective_active_coordinate_rms": (
                LEARNING_RATE * current_gamma
            ),
            "optimizer_effective_active_coordinate_rms_is_step_dependent": True,
            "learning_rate_semantics": (
                "effective_active_coordinate_rms_equals_1e-6_times_"
                "v16r5_step_attenuation_gamma"
            ),
            "optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
            "successful_update_count": step,
            "action_only_fallback_allowed": False,
            "optimizer_retry_allowed": False,
            "optimizer_state_reset_allowed": False,
            "training_data_changed_from_v16r4": False,
            "self_generated_donor_policy_changed_from_v16r4": False,
            "action_and_source_objectives_changed_from_v16r4": False,
        }
    )
    canary["schema_version"] = DECODED_CANARY_SCHEMA
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["v16r5_source_descent_summary"] = source
    receipt["v16r5_decoded_canary_contract"] = canary
    contract.update(
        {
            "method": METHOD,
            "optimizer": OPTIMIZER,
            "optimizer_scalar_learning_rate": LEARNING_RATE,
            "optimizer_scalar_learning_rate_semantics": (
                "unattenuated_base_active_coordinate_rms_before_v16r5_gap_attenuation"
            ),
            "optimizer_unattenuated_base_active_coordinate_rms": LEARNING_RATE,
            "optimizer_current_effective_active_coordinate_rms": (
                LEARNING_RATE * current_gamma
            ),
            "optimizer_effective_active_coordinate_rms_is_step_dependent": True,
            "learning_rate_semantics": (
                "effective_active_coordinate_rms_equals_1e-6_times_"
                "v16r5_step_attenuation_gamma"
            ),
            "optimizer_failure_policy": OPTIMIZER_FAILURE_POLICY,
            "v16r5_direction_policy": DIRECTION_POLICY,
            "v16r5_q_formula": (
                "min(1,max(0.01,-action_replay_cosine+0.01))"
            ),
            "v16r5_q_min": Q_MIN,
            "v16r5_q_max": Q_MAX,
            "v16r5_minimum_antipodal_gap_exclusive": MIN_ANTIPODAL_GAP,
            "v16r5_minimum_formal_normalized_margin_exclusive": (
                MIN_FORMAL_NORMALIZED_MARGIN
            ),
            "v16r5_minimum_actual_descent_cosine_exclusive": (
                MIN_ACTUAL_DESCENT_COSINE
            ),
            "v16r5_near_antipodal_step_attenuation_policy": (
                ATTENUATION_POLICY
            ),
            "v16r5_near_antipodal_step_attenuation_trigger_gap": (
                ATTENUATION_TRIGGER_GAP
            ),
            "training_data_changed_from_v16r4": False,
            "self_generated_donor_policy_changed_from_v16r4": False,
            "action_and_source_objectives_changed_from_v16r4": False,
            "action_only_fallback_allowed": False,
            "optimizer_retry_allowed": False,
            "optimizer_state_reset_allowed": False,
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Install v16r5 hooks only for this invocation, then restore v16r4."""

    global _V16R5_AUDIT, _PENDING_STEP_GEOMETRY

    _V16R5_AUDIT = _empty_v16r5_audit()
    _PENDING_STEP_GEOMETRY = None
    replacements = {
        "METHOD": METHOD,
        "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "OPTIMIZER": OPTIMIZER,
        "OPTIMIZER_FAILURE_POLICY": OPTIMIZER_FAILURE_POLICY,
        "DECODED_CANARY_SCHEMA": DECODED_CANARY_SCHEMA,
        "build_parser": build_parser,
        "validate_args": validate_args,
        "merge_component_gradients": merge_component_gradients,
        "actual_optimizer_update_probe": actual_optimizer_update_probe,
        "checkpoint_receipt": checkpoint_receipt,
        "_projected_optimizer_factory": _projected_optimizer_factory,
    }
    originals = {name: getattr(parent, name) for name in replacements}
    for name, value in replacements.items():
        setattr(parent, name, value)
    try:
        return parent.main(argv)
    finally:
        for name, value in originals.items():
            setattr(parent, name, value)
        _PENDING_STEP_GEOMETRY = None


if __name__ == "__main__":
    raise SystemExit(main())

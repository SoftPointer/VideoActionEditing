#!/usr/bin/env python3
"""v16 exact644 training with a fail-closed post-Adam action fallback.

v16 already rejects any committed optimizer displacement that is not a
descent direction for the primary action objective.  A fresh v16 run showed
that accumulated AdamW momentum can violate that gate even when the merged
PCGrad direction itself is valid.  This revision does not loosen the gate.
When every distributed rank observes that exact failure, it discards the
candidate parameter values, resets the now-polluted AdamW state, and retries
the update once with only the clipped primary-action gradient.  The frozen
post-update probe is then run again and remains the authority for acceptance.

Resetting AdamW state deliberately loses the previous moment history.  The
receipt therefore claims an exact parameter rollback and an optimizer-state
reset, never an optimizer-state rollback or uninterrupted AdamW semantics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16 as v16


base = v16.base
METHOD = "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r2"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r2"
)
FALLBACK_POLICY = (
    "unanimous_actual_action_ascent_parameter_rollback_"
    "adamw_state_reset_action_only_retry_once_v16r2"
)
FALLBACK_REASON = "actual_adamw_parameter_displacement_failed_action_descent_gate"
EXPECTED_REPLAY_MODE = v16.v15.REPLAY_COMBINE_MODE
EXPECTED_TRAINABLE_TENSOR_COUNT = 480
ACTION_ASCENT_PREFIX = (
    "actual optimizer update is not an action-descent step:"
)


_V16_VALIDATE_ARGS = v16.validate_args
_V16_CHECKPOINT_RECEIPT = v16.checkpoint_receipt
_BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE = base.actual_optimizer_update_probe


def _empty_runtime_audit() -> dict[str, Any]:
    return {
        "fallback_count": 0,
        "fallback_steps": [],
        "fallback_target_iids": [],
        "fallback_geometry": [],
        "optimizer_state_reset_count": 0,
    }


_RUNTIME_AUDIT = _empty_runtime_audit()
_ACTIVE_OPTIMIZER: Any = None
_ACTIVE_MAX_GRAD_NORM: Optional[float] = None


def fail(message: str) -> None:
    base.fail(message)


def validate_args(args: argparse.Namespace) -> None:
    """Retain every v16 argument gate and bind the fallback clip norm."""

    global _ACTIVE_MAX_GRAD_NORM

    _V16_VALIDATE_ARGS(args)
    if "v16r2" not in str(Path(args.output)).lower():
        fail("v16r2 output path must carry an explicit v16r2 namespace")
    if float(args.max_grad_norm) != 10.0:
        fail("v16r2 requires the audited max gradient norm 10")
    _ACTIVE_MAX_GRAD_NORM = float(args.max_grad_norm)


def _capturing_adamw_factory(original_adamw: Any) -> Any:
    def factory(*args: Any, **kwargs: Any) -> Any:
        global _ACTIVE_OPTIMIZER

        if _ACTIVE_OPTIMIZER is not None:
            fail("v16r2 expected exactly one AdamW optimizer construction")
        optimizer = original_adamw(*args, **kwargs)
        _ACTIVE_OPTIMIZER = optimizer
        return optimizer

    return factory


def _validate_optimizer_closure(
    named: Sequence[tuple[str, Any]], optimizer: Any
) -> tuple[Any, ...]:
    if optimizer is None:
        fail("v16r2 did not capture the active AdamW optimizer")
    if len(named) != EXPECTED_TRAINABLE_TENSOR_COUNT:
        fail("v16r2 trainable tensor closure is not exactly 480")
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list) or len(groups) != 1:
        fail("v16r2 requires exactly one AdamW parameter group")
    group = groups[0]
    parameters = tuple(group.get("params", ()))
    expected = tuple(parameter for _, parameter in named)
    if len(parameters) != len(expected) or any(
        actual is not wanted for actual, wanted in zip(parameters, expected)
    ):
        fail("v16r2 captured optimizer parameter identity/order differs")
    if float(group.get("lr", math.nan)) != 1.0e-5:
        fail("v16r2 captured AdamW learning rate differs")
    if float(group.get("weight_decay", math.nan)) != 0.0:
        fail("v16r2 fallback requires AdamW weight_decay=0")
    if bool(group.get("maximize", False)):
        fail("v16r2 fallback forbids AdamW maximize mode")
    if tuple(group.get("betas", ())) != (0.9, 0.999):
        fail("v16r2 captured AdamW betas differ")
    if float(group.get("eps", math.nan)) != 1.0e-8:
        fail("v16r2 captured AdamW epsilon differs")
    if bool(group.get("amsgrad", False)):
        fail("v16r2 fallback forbids AdamW AMSGrad")
    return parameters


def _collective_category(local_category: str, *, device: Any, phase: str) -> str:
    """Require all initialized ranks to report the same probe category."""

    import torch
    import torch.distributed as dist

    categories = ("pass", "expected_action_ascent", "unexpected_failure")
    if local_category not in categories:
        fail(f"v16r2 {phase} local probe category differs")
    counts = torch.tensor(
        [int(local_category == category) for category in categories],
        dtype=torch.int32,
        device=device,
    )
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    values = tuple(int(value) for value in counts.cpu().tolist())
    unanimous = [
        category
        for category, count in zip(categories, values)
        if count == world_size
    ]
    if sum(values) != world_size or len(unanimous) != 1:
        fail(
            f"v16r2 {phase} probe result differs across ranks: "
            f"pass={values[0]}, expected_action_ascent={values[1]}, "
            f"unexpected_failure={values[2]}, world_size={world_size}"
        )
    return unanimous[0]


def _local_frozen_probe(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    step: int,
) -> tuple[str, Optional[dict[str, Any]], Optional[BaseException]]:
    try:
        values = dict(
            _BASE_ACTUAL_OPTIMIZER_UPDATE_PROBE(
                named,
                parameter_values_before_step,
                action_gradients,
                raw_replay_gradients,
                replay_combine_mode=replay_combine_mode,
                step=step,
            )
        )
        return "pass", values, None
    except base.OnlineAnchorTrainingError as error:
        if str(error).startswith(ACTION_ASCENT_PREFIX):
            return "expected_action_ascent", None, error
        return "unexpected_failure", None, error


def _candidate_geometry(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
) -> dict[str, Any]:
    """Measure the rejected candidate before its parameters are restored."""

    import torch

    count = len(named)
    if (
        count == 0
        or len(parameter_values_before_step) != count
        or len(action_gradients) != count
        or len(raw_replay_gradients) != count
    ):
        fail("v16r2 rejected-candidate geometry closure differs")
    device = named[0][1].device
    delta_sq = torch.zeros((), dtype=torch.float64, device=device)
    action_sq = torch.zeros_like(delta_sq)
    replay_sq = torch.zeros_like(delta_sq)
    action_dot = torch.zeros_like(delta_sq)
    replay_dot = torch.zeros_like(delta_sq)
    changed_tensors = torch.zeros((), dtype=torch.int64, device=device)
    changed_elements = torch.zeros_like(changed_tensors)
    for (name, parameter), before, action, replay in zip(
        named,
        parameter_values_before_step,
        action_gradients,
        raw_replay_gradients,
    ):
        if (
            tuple(parameter.shape) != tuple(before.shape)
            or tuple(parameter.shape) != tuple(action.shape)
            or tuple(parameter.shape) != tuple(replay.shape)
        ):
            fail(f"v16r2 rejected-candidate geometry differs: {name}")
        before64 = before.detach().double()
        delta64 = parameter.detach().double().sub(before64)
        action64 = action.detach().double()
        replay64 = replay.detach().double()
        delta_sq += delta64.square().sum()
        action_sq += action64.square().sum()
        replay_sq += replay64.square().sum()
        action_dot += (action64 * delta64).sum()
        replay_dot += (replay64 * delta64).sum()
        changed = parameter.detach().ne(before)
        changed_tensors += changed.any().to(dtype=torch.int64)
        changed_elements += changed.count_nonzero().to(dtype=torch.int64)
    delta_norm = math.sqrt(float(delta_sq.item()))
    action_norm = math.sqrt(float(action_sq.item()))
    replay_norm = math.sqrt(float(replay_sq.item()))
    action_dot_value = float(action_dot.item())
    replay_dot_value = float(replay_dot.item())
    values = {
        "delta_theta_l2_norm_fp64": delta_norm,
        "action_gradient_l2_norm_fp64": action_norm,
        "raw_replay_gradient_l2_norm_fp64": replay_norm,
        "action_gradient_dot_delta_theta_fp64": action_dot_value,
        "raw_replay_gradient_dot_delta_theta_fp64": replay_dot_value,
        "action_descent_fp64": -action_dot_value,
        "source_descent_fp64": -replay_dot_value,
        "changed_tensor_count": int(changed_tensors.item()),
        "changed_element_count": int(changed_elements.item()),
    }
    if (
        delta_norm <= 0.0
        or action_norm <= 0.0
        or replay_norm <= 0.0
        or float(values["action_descent_fp64"]) > 0.0
    ):
        fail("v16r2 rejected candidate does not reproduce action ascent")
    if not all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        fail("v16r2 rejected-candidate geometry is non-finite")
    return values


def _restore_parameters_exactly(
    named: Sequence[tuple[str, Any]], parameter_values_before_step: Sequence[Any]
) -> None:
    import torch

    if len(named) != len(parameter_values_before_step):
        fail("v16r2 parameter rollback closure differs")
    with torch.no_grad():
        for (name, parameter), before in zip(
            named, parameter_values_before_step
        ):
            if (
                tuple(parameter.shape) != tuple(before.shape)
                or parameter.dtype != before.dtype
                or parameter.device != before.device
            ):
                fail(f"v16r2 parameter rollback geometry differs: {name}")
            parameter.copy_(before)
            if not bool(torch.equal(parameter.detach(), before)):
                fail(f"v16r2 parameter rollback is not exact: {name}")


def _install_action_only_gradients(
    named: Sequence[tuple[str, Any]],
    action_gradients: Sequence[Any],
    *,
    optimizer: Any,
    max_grad_norm: float,
) -> tuple[float, float]:
    """Copy, then clip, without aliasing or mutating raw action snapshots."""

    import torch

    if len(named) != len(action_gradients):
        fail("v16r2 action-only gradient closure differs")
    action_versions = tuple(int(action._version) for action in action_gradients)
    optimizer.zero_grad(set_to_none=False)
    for (name, parameter), action in zip(named, action_gradients):
        gradient = parameter.grad
        if gradient is None:
            fail(f"v16r2 action-only retry lost a gradient buffer: {name}")
        if tuple(gradient.shape) != tuple(action.shape):
            fail(f"v16r2 action-only retry gradient geometry differs: {name}")
        if gradient.data_ptr() == action.data_ptr():
            fail(f"v16r2 action-only retry gradient aliases raw action: {name}")
        gradient.copy_(action)
    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in named], float(max_grad_norm)
        ).item()
    )
    clipped_sq = torch.zeros(
        (), dtype=torch.float64, device=named[0][1].device
    )
    for (name, parameter), action, version in zip(
        named, action_gradients, action_versions
    ):
        if int(action._version) != version:
            fail(f"v16r2 retry mutated the raw action gradient: {name}")
        if parameter.grad is None or parameter.grad.data_ptr() == action.data_ptr():
            fail(f"v16r2 clipped retry gradient aliases raw action: {name}")
        clipped_sq += parameter.grad.detach().double().square().sum()
    clipped_norm = math.sqrt(float(clipped_sq.item()))
    if not math.isfinite(preclip_norm) or not math.isfinite(clipped_norm):
        fail("v16r2 action-only retry gradient norm is non-finite")
    if preclip_norm <= 0.0 or clipped_norm <= 0.0:
        fail("v16r2 action-only retry requires a nonzero gradient")
    if clipped_norm > float(max_grad_norm) * (1.0 + 1.0e-5):
        fail("v16r2 action-only retry gradient clipping failed")
    return preclip_norm, clipped_norm


def _optimizer_state_steps(
    optimizer: Any, parameters: Sequence[Any]
) -> tuple[int, ...]:
    steps = []
    if len(optimizer.state) != len(parameters):
        fail("v16r2 reset AdamW state did not cover every trainable tensor")
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or "step" not in state:
            fail("v16r2 reset AdamW state has no per-parameter step")
        step = state["step"]
        value = float(step.item()) if hasattr(step, "item") else float(step)
        if not math.isfinite(value) or value != 1.0:
            fail("v16r2 reset AdamW state did not restart at step one")
        steps.append(int(value))
    return tuple(steps)


def _target_iid_for_step(step: int) -> str:
    manifest_iids = tuple(v16._RUNTIME_AUDIT.get("manifest_iids", ()))
    if step <= 0 or len(manifest_iids) < step:
        fail("v16r2 fallback has no sealed target IID for this step")
    iid = manifest_iids[step - 1]
    if not isinstance(iid, str) or not iid:
        fail("v16r2 fallback target IID differs")
    return iid


def actual_optimizer_update_probe(
    named: Sequence[tuple[str, Any]],
    parameter_values_before_step: Sequence[Any],
    action_gradients: Sequence[Any],
    raw_replay_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    step: int,
) -> Mapping[str, Any]:
    """Run the frozen probe; retry only unanimous primary-action ascent once."""

    if replay_combine_mode != EXPECTED_REPLAY_MODE:
        fail("v16r2 fallback is restricted to action-priority PCGrad")
    if _ACTIVE_MAX_GRAD_NORM is None:
        fail("v16r2 fallback has no validated gradient clip norm")
    parameters = _validate_optimizer_closure(named, _ACTIVE_OPTIMIZER)
    device = named[0][1].device

    category, values, error = _local_frozen_probe(
        named,
        parameter_values_before_step,
        action_gradients,
        raw_replay_gradients,
        replay_combine_mode=replay_combine_mode,
        step=step,
    )
    consensus = _collective_category(
        category, device=device, phase="initial actual-update"
    )
    if consensus == "unexpected_failure":
        if error is not None:
            raise error
        fail("v16r2 peer rank reported an unexpected initial probe failure")
    if consensus == "pass":
        if values is None:
            fail("v16r2 initial probe passed without values")
        values.update(
            {
                "v16r2_actual_action_descent_fallback_applied": False,
                "v16r2_actual_action_descent_fallback_policy": FALLBACK_POLICY,
                "v16r2_failed_candidate_committed": False,
                "v16r2_action_descent_gate_relaxed": False,
                "v16r2_optimizer_history_matches_uninterrupted_adamw": (
                    int(_RUNTIME_AUDIT["fallback_count"]) == 0
                ),
                "v16r2_cumulative_fallback_count": int(
                    _RUNTIME_AUDIT["fallback_count"]
                ),
            }
        )
        return values
    if category != "expected_action_ascent":
        fail("v16r2 local/collective initial probe category differs")

    rejected = _candidate_geometry(
        named,
        parameter_values_before_step,
        action_gradients,
        raw_replay_gradients,
    )
    rejected_state_entries = len(_ACTIVE_OPTIMIZER.state)
    if rejected_state_entries != len(parameters):
        fail("v16r2 failed candidate AdamW state closure differs")
    _restore_parameters_exactly(named, parameter_values_before_step)
    _ACTIVE_OPTIMIZER.state.clear()
    if len(_ACTIVE_OPTIMIZER.state) != 0:
        fail("v16r2 AdamW state reset failed")
    preclip_norm, clipped_norm = _install_action_only_gradients(
        named,
        action_gradients,
        optimizer=_ACTIVE_OPTIMIZER,
        max_grad_norm=float(_ACTIVE_MAX_GRAD_NORM),
    )
    _ACTIVE_OPTIMIZER.step()

    retry_category, retry_values, retry_error = _local_frozen_probe(
        named,
        parameter_values_before_step,
        action_gradients,
        raw_replay_gradients,
        replay_combine_mode=replay_combine_mode,
        step=step,
    )
    retry_consensus = _collective_category(
        retry_category, device=device, phase="action-only retry"
    )
    if retry_consensus != "pass":
        _restore_parameters_exactly(named, parameter_values_before_step)
        _ACTIVE_OPTIMIZER.state.clear()
        if retry_error is not None:
            raise retry_error
        fail("v16r2 action-only retry did not pass the frozen probe on every rank")
    if retry_values is None:
        fail("v16r2 action-only retry passed without frozen-probe values")
    state_steps = _optimizer_state_steps(_ACTIVE_OPTIMIZER, parameters)
    target_iid = _target_iid_for_step(int(step))

    committed = {
        "action_descent_fp64": float(retry_values["action_descent_fp64"]),
        "action_gradient_dot_delta_theta_fp64": float(
            retry_values["action_gradient_dot_delta_theta_fp64"]
        ),
        "delta_theta_l2_norm_fp64": float(
            retry_values["delta_theta_l2_norm_fp64"]
        ),
        "source_descent_fp64": float(retry_values["source_descent_fp64"]),
        "action_descent_passed": bool(
            retry_values["action_descent_passed"]
        ),
    }
    if not committed["action_descent_passed"] or committed[
        "action_descent_fp64"
    ] <= 0.0:
        fail("v16r2 committed retry lacks measured action descent")
    event = {
        "step": int(step),
        "target_iid": target_iid,
        "reason": FALLBACK_REASON,
        "distributed_expected_failure_was_unanimous": True,
        "failed_candidate_committed": False,
        "failed_candidate": rejected,
        "parameter_values_exactly_restored_before_retry": True,
        "optimizer_state_restored": False,
        "optimizer_state_reset": True,
        "optimizer_state_entries_after_failed_candidate": rejected_state_entries,
        "optimizer_state_entries_after_reset": 0,
        "committed_retry_gradient": "primary_action_only_clipped",
        "retry_preclip_action_gradient_l2_norm": preclip_norm,
        "retry_clipped_action_gradient_l2_norm_fp64": clipped_norm,
        "retry_max_grad_norm": float(_ACTIVE_MAX_GRAD_NORM),
        "raw_action_gradient_snapshots_mutated": False,
        "retry_count": 1,
        "committed_retry_reprobed_by_frozen_authority": True,
        "retry_probe_distributed_pass_was_unanimous": True,
        "reset_adamw_state_step_min": min(state_steps),
        "reset_adamw_state_step_max": max(state_steps),
        "committed_retry": committed,
    }
    _RUNTIME_AUDIT["fallback_count"] += 1
    _RUNTIME_AUDIT["fallback_steps"].append(int(step))
    _RUNTIME_AUDIT["fallback_target_iids"].append(target_iid)
    _RUNTIME_AUDIT["fallback_geometry"].append(event)
    _RUNTIME_AUDIT["optimizer_state_reset_count"] += 1

    retry_values.update(
        {
            "optimizer_semantics_observed_not_modified": False,
            "v16r2_actual_action_descent_fallback_applied": True,
            "v16r2_actual_action_descent_fallback_policy": FALLBACK_POLICY,
            "v16r2_actual_action_descent_fallback_reason": FALLBACK_REASON,
            "v16r2_failed_candidate_committed": False,
            "v16r2_parameter_values_exactly_restored_before_retry": True,
            "v16r2_optimizer_state_restored": False,
            "v16r2_optimizer_state_reset": True,
            "v16r2_committed_retry_gradient": "primary_action_only_clipped",
            "v16r2_committed_retry_reprobed_by_frozen_authority": True,
            "v16r2_retry_probe_distributed_pass_was_unanimous": True,
            "v16r2_action_descent_gate_relaxed": False,
            "v16r2_optimizer_history_matches_uninterrupted_adamw": False,
            "v16r2_cumulative_fallback_count": int(
                _RUNTIME_AUDIT["fallback_count"]
            ),
            "v16r2_rejected_candidate": rejected,
        }
    )
    return retry_values


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _V16_CHECKPOINT_RECEIPT(**kwargs)
    contract = receipt.get("training_contract")
    step = int(receipt.get("global_step", 0))
    if not isinstance(contract, dict) or step <= 0:
        fail("v16r2 inherited receipt closure differs")
    fallback_count = int(_RUNTIME_AUDIT["fallback_count"])
    fallback_steps = list(_RUNTIME_AUDIT["fallback_steps"])
    target_iids = list(_RUNTIME_AUDIT["fallback_target_iids"])
    geometry = list(_RUNTIME_AUDIT["fallback_geometry"])
    reset_count = int(_RUNTIME_AUDIT["optimizer_state_reset_count"])
    if not (
        fallback_count
        == reset_count
        == len(fallback_steps)
        == len(target_iids)
        == len(geometry)
    ):
        fail("v16r2 cumulative fallback receipt accounting differs")
    if fallback_steps != sorted(set(fallback_steps)) or any(
        fallback_step <= 0 or fallback_step > step
        for fallback_step in fallback_steps
    ):
        fail("v16r2 cumulative fallback step sequence differs")
    if any(
        int(item.get("step", -1)) != fallback_step
        or item.get("target_iid") != target_iid
        or item.get("failed_candidate_committed") is not False
        or item.get("parameter_values_exactly_restored_before_retry") is not True
        or item.get("optimizer_state_restored") is not False
        or item.get("optimizer_state_reset") is not True
        or item.get("committed_retry_reprobed_by_frozen_authority") is not True
        or item.get("committed_retry", {}).get("action_descent_passed") is not True
        for item, fallback_step, target_iid in zip(
            geometry, fallback_steps, target_iids
        )
    ):
        fail("v16r2 cumulative fallback event closure differs")

    uninterrupted = fallback_count == 0
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_held_out_evaluation"
    )
    receipt["v16r2_actual_action_descent_fallback_summary"] = {
        "policy": FALLBACK_POLICY,
        "reason": FALLBACK_REASON,
        "fallback_count": fallback_count,
        "fallback_steps": fallback_steps,
        "fallback_target_iids": target_iids,
        "fallback_geometry": geometry,
        "optimizer_state_reset_count": reset_count,
        "failed_candidates_committed": False,
        "parameter_values_exactly_restored_before_each_retry": True,
        "optimizer_state_restored": False,
        "optimizer_state_reset_before_each_retry": True,
        "committed_retry_gradient": "primary_action_only_clipped",
        "retry_limit_per_failed_candidate": 1,
        "committed_retries_reprobed_by_frozen_authority": True,
        "action_descent_gate_relaxed": False,
        "optimizer_history_matches_uninterrupted_adamw": uninterrupted,
        "continuous_parameter_trajectory_from_frozen_base": True,
        "source_preservation_claimed": False,
        "scientific_claim_authorized": False,
    }
    contract.update(
        {
            "method": METHOD,
            "actual_action_descent_fallback_policy": FALLBACK_POLICY,
            "actual_action_descent_fallback_count": fallback_count,
            "actual_action_descent_fallback_steps": fallback_steps,
            "actual_action_descent_fallback_target_iids": target_iids,
            "actual_action_descent_failed_candidates_committed": False,
            "actual_action_descent_fallback_parameter_values_exactly_restored": True,
            "actual_action_descent_fallback_optimizer_state_restored": False,
            "actual_action_descent_fallback_optimizer_state_reset_count": reset_count,
            "actual_action_descent_fallback_uses_primary_action_only": True,
            "actual_action_descent_fallback_retry_limit": 1,
            "actual_action_descent_fallback_reprobes_frozen_authority": True,
            "actual_action_descent_gate_relaxed": False,
            "optimizer_history_matches_uninterrupted_adamw": uninterrupted,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
            "source_preservation_claimed": False,
            "scientific_claim_authorized": False,
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT, _ACTIVE_OPTIMIZER, _ACTIVE_MAX_GRAD_NORM

    _RUNTIME_AUDIT = _empty_runtime_audit()
    _ACTIVE_OPTIMIZER = None
    _ACTIVE_MAX_GRAD_NORM = None

    import torch

    original_adamw = torch.optim.AdamW
    original_probe = base.actual_optimizer_update_probe
    original_v16_validate = v16.validate_args
    original_v16_receipt = v16.checkpoint_receipt
    torch.optim.AdamW = _capturing_adamw_factory(original_adamw)
    base.actual_optimizer_update_probe = actual_optimizer_update_probe
    v16.validate_args = validate_args
    v16.checkpoint_receipt = checkpoint_receipt
    try:
        result = v16.main(argv)
        if _ACTIVE_OPTIMIZER is None:
            fail("v16r2 completed without capturing an AdamW optimizer")
        return result
    finally:
        torch.optim.AdamW = original_adamw
        base.actual_optimizer_update_probe = original_probe
        v16.validate_args = original_v16_validate
        v16.checkpoint_receipt = original_v16_receipt
        _ACTIVE_OPTIMIZER = None
        _ACTIVE_MAX_GRAD_NORM = None


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fresh v15r2 training with an audited near-collinear replay fallback.

The v15 dynamic/static routed-teacher contract is unchanged.  Its registered
PCGrad merge is also unchanged for every feasible update.  If, and only if,
the frozen auxiliary replay gradient is so nearly anti-parallel to the primary
action gradient that the existing PCGrad retained-norm gate rejects it, this
entry point drops replay for that one update and applies the primary action
gradient.  The fallback is recorded per update and in every later checkpoint.

This is an engineering training run.  In particular, a fallback update makes
no source-preservation claim, and no receipt produced here authorizes a
scientific result.
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

import train_online_anchor_attention_dynamic_static_v15 as v15


base = v15.base
METHOD = "bernini-online-anchor-dynamic-static-routed-teacher-v15r2"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-dynamic-static-routed-teacher-receipt-v15r2"
)
FALLBACK_POLICY = "near_collinear_pcgrad_gate_primary_action_only_v15r2"
FALLBACK_REASON = "projected_replay_retained_raw_norm_fraction_below_0.2"
PCGRAD_RETAINED_NORM_FLOOR = 0.2
ALLOWED_STEPS = (8, 32)


_BASE_MERGE_COMPONENT_GRADIENTS = base.merge_component_gradients
_V15_VALIDATE_ARGS = v15.validate_args
_V15_BUILD_ANCHOR_BATCHES = v15.build_anchor_batches
_V15_CHECKPOINT_RECEIPT = v15.checkpoint_receipt


def _empty_runtime_audit() -> dict[str, Any]:
    return {
        "merge_call_count": 0,
        "fallback_count": 0,
        "fallback_steps": [],
        "fallback_target_iids": set(),
        "fallback_target_events": set(),
        "fallback_geometry": [],
        "current_target_iid": None,
        "current_target_event": None,
    }


_RUNTIME_AUDIT = _empty_runtime_audit()


def fail(message: str) -> None:
    base.fail(message)


def build_parser() -> argparse.ArgumentParser:
    return v15.build_parser()


def validate_args(args: argparse.Namespace) -> None:
    _V15_VALIDATE_ARGS(args)
    if int(args.max_steps) not in ALLOWED_STEPS:
        fail("v15r2 staged training permits max-steps 8 or 32 only")
    if "v15r2" not in str(Path(args.output)).lower():
        fail("v15r2 output path must carry an explicit v15r2 namespace")


def build_anchor_batches(**kwargs: Any) -> Any:
    result = _V15_BUILD_ANCHOR_BATCHES(**kwargs)
    target_row = kwargs.get("target_row")
    if not isinstance(target_row, Mapping):
        fail("v15r2 target row is absent")
    iid = target_row.get("iid")
    event = target_row.get("event_id")
    if not isinstance(iid, str) or not iid or not isinstance(event, str) or not event:
        fail("v15r2 target row identity is incomplete")
    _RUNTIME_AUDIT["current_target_iid"] = iid
    _RUNTIME_AUDIT["current_target_event"] = event
    return result


def _fallback_geometry(
    named: Sequence[tuple[str, Any]], action_gradients: Sequence[Any]
) -> dict[str, Any]:
    import torch

    if not named or len(named) != len(action_gradients):
        fail("v15r2 fallback component-gradient length differs")
    first = named[0][1].grad
    if first is None:
        fail("v15r2 fallback has no replay gradient")
    action_sq = torch.zeros((), dtype=torch.float64, device=first.device)
    replay_sq = torch.zeros_like(action_sq)
    dot = torch.zeros_like(action_sq)
    for (name, parameter), action in zip(named, action_gradients):
        replay = parameter.grad
        if replay is None or tuple(replay.shape) != tuple(action.shape):
            fail(f"v15r2 fallback gradient geometry differs: {name}")
        if not bool(torch.isfinite(action).all().item()):
            fail(f"v15r2 action gradient is non-finite: {name}")
        if not bool(torch.isfinite(replay).all().item()):
            fail(f"v15r2 replay gradient is non-finite: {name}")
        action64 = action.detach().double()
        replay64 = replay.detach().double()
        action_sq += action64.square().sum()
        replay_sq += replay64.square().sum()
        dot += (action64 * replay64).sum()

    action_sq_value = float(action_sq.item())
    replay_sq_value = float(replay_sq.item())
    raw_dot = float(dot.item())
    if action_sq_value <= 0.0 or replay_sq_value <= 0.0:
        fail("v15r2 fallback requires two nonzero component gradients")
    action_norm = math.sqrt(action_sq_value)
    replay_norm = math.sqrt(replay_sq_value)
    cosine = max(-1.0, min(1.0, raw_dot / (action_norm * replay_norm)))
    projection_coefficient = raw_dot / action_sq_value
    projected_sq = replay_sq_value - raw_dot * raw_dot / action_sq_value
    processed_norm = math.sqrt(max(0.0, projected_sq))
    retained = processed_norm / replay_norm
    processed_action_dot = raw_dot - projection_coefficient * action_sq_value
    processed_cosine = (
        processed_action_dot / (processed_norm * action_norm)
        if processed_norm > 0.0
        else None
    )
    return {
        "action_sq": action_sq_value,
        "replay_sq": replay_sq_value,
        "raw_dot": raw_dot,
        "action_norm": action_norm,
        "replay_norm": replay_norm,
        "cosine": cosine,
        "projection_coefficient": projection_coefficient,
        "processed_norm": processed_norm,
        "retained": retained,
        "processed_action_dot": processed_action_dot,
        "processed_cosine": processed_cosine,
    }


def merge_component_gradients(
    named: Sequence[tuple[str, Any]],
    action_gradients: Sequence[Any],
    *,
    replay_combine_mode: str,
    base_replay_scale: float,
    diagnostic_only: bool = False,
) -> Mapping[str, Any]:
    """Use v15 PCGrad normally; drop replay only at its retained-norm gate."""

    _RUNTIME_AUDIT["merge_call_count"] += 1
    step = int(_RUNTIME_AUDIT["merge_call_count"])
    try:
        values = dict(
            _BASE_MERGE_COMPONENT_GRADIENTS(
                named,
                action_gradients,
                replay_combine_mode=replay_combine_mode,
                base_replay_scale=base_replay_scale,
                diagnostic_only=diagnostic_only,
            )
        )
    except base.OnlineAnchorTrainingError as error:
        if (
            replay_combine_mode != v15.REPLAY_COMBINE_MODE
            or bool(diagnostic_only)
            or not str(error).startswith(
                "action-priority PCGrad formal geometry gate failed:"
            )
        ):
            raise
        geometry = _fallback_geometry(named, action_gradients)
        if not (
            float(geometry["raw_dot"]) < 0.0
            and float(geometry["retained"]) < PCGRAD_RETAINED_NORM_FLOOR
        ):
            raise

        import torch

        combined_sq = torch.zeros(
            (), dtype=torch.float64, device=named[0][1].grad.device
        )
        action_inner = torch.zeros_like(combined_sq)
        replay_inner = torch.zeros_like(combined_sq)
        processed_inner = torch.zeros_like(combined_sq)
        for (_name, parameter), action in zip(named, action_gradients):
            replay64 = parameter.grad.detach().double()
            action64 = action.detach().double()
            processed64 = replay64 - float(
                geometry["projection_coefficient"]
            ) * action64
            parameter.grad.copy_(action)
            combined64 = parameter.grad.detach().double()
            combined_sq += combined64.square().sum()
            action_inner += (action64 * combined64).sum()
            replay_inner += (replay64 * combined64).sum()
            processed_inner += (processed64 * combined64).sum()

        combined_norm = math.sqrt(float(combined_sq.item()))
        action_inner_value = float(action_inner.item())
        replay_inner_value = float(replay_inner.item())
        processed_inner_value = float(processed_inner.item())
        action_sq_value = float(geometry["action_sq"])
        action_norm = float(geometry["action_norm"])
        replay_norm = float(geometry["replay_norm"])
        raw_dot = float(geometry["raw_dot"])
        processed_norm = float(geometry["processed_norm"])
        projection = float(geometry["projection_coefficient"])
        retained = float(geometry["retained"])
        cosine = float(geometry["cosine"])
        processed_cosine = geometry["processed_cosine"]
        lambda_min = max(0.0, -raw_dot / float(geometry["replay_sq"]))

        if action_inner_value <= 0.0:
            fail("v15r2 primary-action fallback lost the action descent direction")
        values = {
            "action_l2_norm_fp64": action_norm,
            "raw_replay_l2_norm_fp64": replay_norm,
            "processed_replay_l2_norm_fp64": processed_norm,
            "weighted_replay_l2_norm_fp64": 0.0,
            "combined_l2_norm_fp64": combined_norm,
            "planned_combined_l2_norm_fp64": action_norm,
            "action_raw_replay_dot_fp64": raw_dot,
            "action_replay_cosine": cosine,
            "replay_combine_mode": replay_combine_mode,
            "base_replay_scale": float(base_replay_scale),
            "first_order_safe_lambda_min": lambda_min,
            "effective_replay_scale": 0.0,
            "weighted_replay_gradient_fraction": 0.0,
            "weighted_replay_to_action_grad_norm_ratio": 0.0,
            "replay_component_to_action_norm_ratio_q": 0.0,
            "correction_ratio_q": 0.0,
            "replay_projection_applied": True,
            "replay_projection_coefficient": projection,
            "processed_replay_retained_raw_norm_fraction": retained,
            "processed_replay_action_cosine": processed_cosine,
            "action_priority_conflict_control_not_source_preservation": True,
            "action_gradient_dot_combined_gradient_fp64": action_inner_value,
            "planned_action_gradient_dot_combined_gradient_fp64": action_sq_value,
            "action_alignment_ratio": action_inner_value / action_sq_value,
            "action_combined_cosine": (
                action_inner_value / (action_norm * combined_norm)
                if combined_norm > 0.0
                else None
            ),
            "raw_replay_gradient_dot_combined_gradient_fp64": replay_inner_value,
            "raw_replay_combined_alignment_over_action_replay_norms": (
                replay_inner_value / (action_norm * replay_norm)
            ),
            "planned_raw_replay_gradient_dot_combined_gradient_fp64": raw_dot,
            "raw_replay_combined_cosine": (
                replay_inner_value / (replay_norm * combined_norm)
                if combined_norm > 0.0
                else None
            ),
            "processed_replay_gradient_dot_combined_gradient_fp64": (
                processed_inner_value
            ),
            "planned_processed_replay_gradient_dot_combined_gradient_fp64": (
                float(geometry["processed_action_dot"])
            ),
            "raw_source_fm_gradient_dot_combined_gradient_fp64": replay_inner_value,
            "first_order_source_fm_preserved": replay_inner_value >= -1.0e-8,
            "v15r2_collinear_fallback_applied": True,
            "v15r2_collinear_fallback_policy": FALLBACK_POLICY,
            "v15r2_collinear_fallback_reason": FALLBACK_REASON,
            "v15r2_auxiliary_replay_dropped_for_update": True,
            "v15r2_processed_replay_used_in_update": False,
            "v15r2_source_preservation_claimed_for_update": False,
            "v15r2_original_pcgrad_retained_norm_floor_unchanged": (
                PCGRAD_RETAINED_NORM_FLOOR
            ),
        }
        if not all(
            math.isfinite(float(value))
            for value in values.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            fail("v15r2 fallback interaction is non-finite")

        iid = _RUNTIME_AUDIT["current_target_iid"]
        event = _RUNTIME_AUDIT["current_target_event"]
        if not isinstance(iid, str) or not isinstance(event, str):
            fail("v15r2 fallback has no audited current target identity")
        _RUNTIME_AUDIT["fallback_count"] += 1
        _RUNTIME_AUDIT["fallback_steps"].append(step)
        _RUNTIME_AUDIT["fallback_target_iids"].add(iid)
        _RUNTIME_AUDIT["fallback_target_events"].add(event)
        _RUNTIME_AUDIT["fallback_geometry"].append(
            {
                "step": step,
                "target_iid": iid,
                "target_event": event,
                "action_replay_cosine": cosine,
                "processed_replay_retained_raw_norm_fraction": retained,
                "effective_replay_scale": 0.0,
            }
        )
        return values

    values.update(
        {
            "v15r2_collinear_fallback_applied": False,
            "v15r2_collinear_fallback_policy": FALLBACK_POLICY,
            "v15r2_collinear_fallback_reason": None,
            "v15r2_auxiliary_replay_dropped_for_update": False,
            "v15r2_processed_replay_used_in_update": True,
            "v15r2_source_preservation_claimed_for_update": False,
            "v15r2_original_pcgrad_retained_norm_floor_unchanged": (
                PCGRAD_RETAINED_NORM_FLOOR
            ),
        }
    )
    return values


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _V15_CHECKPOINT_RECEIPT(**kwargs)
    contract = receipt.get("training_contract")
    if not isinstance(contract, dict):
        fail("v15r2 base checkpoint receipt has no mutable training contract")
    fallback_count = int(_RUNTIME_AUDIT["fallback_count"])
    if fallback_count != len(_RUNTIME_AUDIT["fallback_steps"]):
        fail("v15r2 fallback audit count differs")
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_held_out_evaluation"
    )
    receipt["v15r2_collinear_fallback_summary"] = {
        "policy": FALLBACK_POLICY,
        "fallback_count": fallback_count,
        "fallback_steps": list(_RUNTIME_AUDIT["fallback_steps"]),
        "fallback_target_iids": sorted(_RUNTIME_AUDIT["fallback_target_iids"]),
        "fallback_target_events": sorted(
            _RUNTIME_AUDIT["fallback_target_events"]
        ),
        "fallback_geometry": list(_RUNTIME_AUDIT["fallback_geometry"]),
        "source_preservation_claimed": False,
        "scientific_claim_authorized": False,
    }
    contract.update(
        {
            "method": METHOD,
            "replay_conflict_policy": FALLBACK_POLICY,
            "normal_updates_use_unmodified_action_priority_pcgrad_010": True,
            "pcgrad_retained_raw_norm_floor": PCGRAD_RETAINED_NORM_FLOOR,
            "pcgrad_retained_raw_norm_floor_was_loosened": False,
            "near_collinear_fallback_drops_auxiliary_replay_for_that_update": True,
            "near_collinear_fallback_keeps_primary_action_gradient": True,
            "near_collinear_fallback_count": fallback_count,
            "near_collinear_fallback_steps": list(
                _RUNTIME_AUDIT["fallback_steps"]
            ),
            "near_collinear_fallback_target_iids": sorted(
                _RUNTIME_AUDIT["fallback_target_iids"]
            ),
            "near_collinear_fallback_target_events": sorted(
                _RUNTIME_AUDIT["fallback_target_events"]
            ),
            "source_preservation_claimed": False,
            "scientific_claim_authorized": False,
        }
    )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT

    _RUNTIME_AUDIT = _empty_runtime_audit()
    original_merge = base.merge_component_gradients
    original_validate = v15.validate_args
    original_builder = v15.build_anchor_batches
    original_receipt = v15.checkpoint_receipt
    base.merge_component_gradients = merge_component_gradients
    v15.validate_args = validate_args
    v15.build_anchor_batches = build_anchor_batches
    v15.checkpoint_receipt = checkpoint_receipt
    try:
        return v15.main(argv)
    finally:
        base.merge_component_gradients = original_merge
        v15.validate_args = original_validate
        v15.build_anchor_batches = original_builder
        v15.checkpoint_receipt = original_receipt


if __name__ == "__main__":
    raise SystemExit(main())

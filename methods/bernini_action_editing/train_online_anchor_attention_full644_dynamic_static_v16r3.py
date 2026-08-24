#!/usr/bin/env python3
"""v16r3 exact644 training with a finite zero-RMS route subgradient.

The frozen v16r2 run reached its first action micro at scheduler timestep 1000
on S279.  At that endpoint the dynamic and phase-zero-static donor inputs are
exactly equal, so their target-owned temporal-kernel route is exactly zero.
The v16r2 forward correctly selected the identity route, but raw
``sqrt(mean(route**2))`` autograd evaluated a singular derivative before the
hard fallback could discard it, producing ``0 * inf -> NaN`` in the action
gradient.

The route implementation used by this revision preserves every RMS forward
value exactly.  Only at exact RMS zero it selects the natural zero
subgradient.  Positive inputs retain the ordinary square-root derivative, and
negative/non-finite inputs remain visible to the unchanged finite gates.  The
policy is fixed from step one: no sample, seed, timestep, loss scale, finite
gate, or optimizer-admission rule is changed.
"""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r2 as v16r2


v16 = v16r2.v16
base = v16r2.base
qk = base.qk
METHOD = "bernini-online-anchor-full644-dynamic-static-routed-teacher-v16r3"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-full644-dynamic-static-routed-teacher-receipt-v16r3"
)
ZERO_RMS_POLICY = "exact_forward_zero_rms_zero_subgradient_v1"
ZERO_RMS_SCOPE = ("current_temporal_rms", "route_rms")
S279_STEP = 279
S279_TARGET_IID = "4aeb0557a94b4db3"
S279_TARGET_FAMILY = "fall"
S279_EXPECTED_CALLS = (
    {
        "role": "action_micro_0",
        "seed": 1656484053,
        "timestep": 1000.0,
    },
    {
        "role": "raw_replay_micro_0",
        "seed": 1657484056,
        "timestep": 580.0,
    },
    {
        "role": "action_micro_1",
        "seed": 718898016,
        "timestep": 764.0,
    },
    {
        "role": "raw_replay_micro_1",
        "seed": 719898019,
        "timestep": 880.0,
    },
)


_V16_VALIDATE_ARGS = v16r2._V16_VALIDATE_ARGS
_V16R2_CHECKPOINT_RECEIPT = v16r2.checkpoint_receipt
_V16_BUILD_REAL_SOURCE = v16.build_real_source_paired_records_full644_v16


def _empty_runtime_audit() -> dict[str, Any]:
    return {"s279_builder_calls": []}


_RUNTIME_AUDIT = _empty_runtime_audit()


def fail(message: str) -> None:
    base.fail(message)


def _validate_zero_rms_operator() -> None:
    import torch

    if getattr(qk, "QK_ONLY_ZERO_RMS_BACKWARD_POLICY", None) != ZERO_RMS_POLICY:
        fail("v16r3 QK zero-RMS policy label differs")
    helper = getattr(qk, "_exact_forward_zero_subgradient_rms", None)
    active = getattr(
        qk,
        "_qk_only_target_gated_hard_temporal_kernel_contrast_output",
        None,
    )
    if not callable(helper) or not callable(active):
        fail("v16r3 QK zero-RMS operator is absent")
    active_source = inspect.getsource(active)
    if active_source.count("_exact_forward_zero_subgradient_rms(") != 2:
        fail("v16r3 active QK route does not bind exactly two safe RMS sites")

    zero = torch.zeros((2, 3, 4), dtype=torch.float32, requires_grad=True)
    rms = helper(zero, dim=(1, 2), keepdim=True)
    if not bool(torch.equal(rms, torch.zeros_like(rms))):
        fail("v16r3 zero-RMS forward value differs")
    rms.sum().backward()
    if zero.grad is None or not bool(torch.equal(zero.grad, torch.zeros_like(zero))):
        fail("v16r3 zero-RMS backward is not the exact zero subgradient")

    positive = torch.tensor(
        [[[1.0e-12, 0.25, 1.0, 9.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference = positive.detach().clone().requires_grad_(True)
    safe_value = helper(positive, dim=(1, 2), keepdim=True)
    ordinary_value = reference.square().mean(dim=(1, 2), keepdim=True).sqrt()
    if not bool(torch.equal(safe_value, ordinary_value)):
        fail("v16r3 positive RMS forward differs from ordinary sqrt")
    safe_value.sum().backward()
    ordinary_value.sum().backward()
    if positive.grad is None or reference.grad is None or not bool(
        torch.equal(positive.grad, reference.grad)
    ):
        fail("v16r3 positive RMS backward differs from ordinary sqrt")


def validate_args(args: argparse.Namespace) -> None:
    """Retain v16/v16r2 gates and bind the fixed v16r3 route operator."""

    _V16_VALIDATE_ARGS(args)
    if "v16r3" not in str(Path(args.output)).lower():
        fail("v16r3 output path must carry an explicit v16r3 namespace")
    if float(args.max_grad_norm) != 10.0:
        fail("v16r3 requires the inherited audited max gradient norm 10")
    # v16r2's actual-displacement fallback reads this validated clip norm.
    v16r2._ACTIVE_MAX_GRAD_NORM = float(args.max_grad_norm)
    _validate_zero_rms_operator()


def build_real_source_paired_records_full644_v16r3(
    *,
    anchor_row: Mapping[str, Any],
    real_sources: Mapping[str, Mapping[str, Any]],
    transform: Any,
    mean: Any,
    std: Any,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _V16_BUILD_REAL_SOURCE(
        anchor_row=anchor_row,
        real_sources=real_sources,
        transform=transform,
        mean=mean,
        std=std,
        seed=seed,
    )
    iid = str(anchor_row.get("iid"))
    if iid != S279_TARGET_IID:
        return result
    if str(anchor_row.get("family")) != S279_TARGET_FAMILY:
        fail("v16r3 S279 endpoint-canary family differs")
    calls = _RUNTIME_AUDIT["s279_builder_calls"]
    if not isinstance(calls, list) or len(calls) >= len(S279_EXPECTED_CALLS):
        fail("v16r3 S279 endpoint-canary call count differs")
    expected = S279_EXPECTED_CALLS[len(calls)]
    timestep = float(result[0]["timestep"])
    observed = {
        "role": str(expected["role"]),
        "seed": int(seed),
        "timestep": timestep,
    }
    if observed != expected:
        fail(
            "v16r3 S279 endpoint-canary seed/timestep differs: "
            f"observed={observed}, expected={expected}"
        )
    calls.append(observed)
    return result


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _V16R2_CHECKPOINT_RECEIPT(**kwargs)
    contract = receipt.get("training_contract")
    step = int(receipt.get("global_step", 0))
    if not isinstance(contract, MutableMapping) or step <= 0:
        fail("v16r3 inherited receipt closure differs")
    calls = list(_RUNTIME_AUDIT["s279_builder_calls"])
    canary_covered = step >= S279_STEP
    if canary_covered:
        if calls != list(S279_EXPECTED_CALLS):
            fail("v16r3 S279 endpoint-canary runtime closure differs")
    elif calls:
        fail("v16r3 S279 endpoint canary appeared before its sealed step")

    summary = {
        "policy": ZERO_RMS_POLICY,
        "scope": list(ZERO_RMS_SCOPE),
        "active_qk_route": (
            "qk_only_target_gated_hard_temporal_kernel_contrast_output"
        ),
        "finite_nonnegative_forward_values_bit_exact": True,
        "zero_forward_value": 0.0,
        "zero_backward_subgradient": 0.0,
        "positive_backward_matches_standard_sqrt": True,
        "negative_or_nonfinite_values_masked": False,
        "loss_scale_changed": False,
        "seed_or_timestep_changed": False,
        "sample_retry_or_skip": False,
        "component_preallreduce_finite_gate_relaxed": False,
        "nonfinite_gradient_committed": False,
        "policy_fixed_from_step_one": True,
        "single_continuous_fresh_from_base_exact644": True,
        "s279_endpoint_canary": {
            "step": S279_STEP,
            "target_iid": S279_TARGET_IID,
            "target_family": S279_TARGET_FAMILY,
            "expected_calls": [dict(item) for item in S279_EXPECTED_CALLS],
            "observed_calls": calls,
            "covered_by_checkpoint": canary_covered,
        },
        "scientific_claim_authorized": False,
    }
    receipt["schema_version"] = RECEIPT_SCHEMA
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_training_run_only_non_scientific_until_held_out_evaluation"
    )
    receipt["v16r3_zero_rms_backward_summary"] = summary
    contract.update(
        {
            "method": METHOD,
            "qk_only_zero_rms_backward_policy": ZERO_RMS_POLICY,
            "qk_only_zero_rms_backward_scope": list(ZERO_RMS_SCOPE),
            "qk_only_zero_rms_forward_values_changed": False,
            "qk_only_zero_rms_zero_subgradient": 0.0,
            "loss_scale_changed_for_v16r3": False,
            "seed_or_timestep_changed_for_v16r3": False,
            "sample_retry_or_skip_for_v16r3": False,
            "component_preallreduce_finite_gate_relaxed": False,
            "nonfinite_gradient_committed": False,
            "s279_endpoint_canary_covered": canary_covered,
            "s279_endpoint_canary_target_iid": S279_TARGET_IID,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": True,
            "scientific_claim_authorized": False,
        }
    )
    anchor_cache = receipt.get("anchor_cache")
    if (
        not isinstance(anchor_cache, Mapping)
        or anchor_cache.get("qk_only_zero_rms_backward_policy")
        != ZERO_RMS_POLICY
    ):
        fail("v16r3 anchor-cache zero-RMS policy receipt differs")
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _RUNTIME_AUDIT

    _RUNTIME_AUDIT = _empty_runtime_audit()
    original_validate = v16r2.validate_args
    original_receipt = v16r2.checkpoint_receipt
    original_builder = v16.build_real_source_paired_records_full644_v16
    v16r2.validate_args = validate_args
    v16r2.checkpoint_receipt = checkpoint_receipt
    v16.build_real_source_paired_records_full644_v16 = (
        build_real_source_paired_records_full644_v16r3
    )
    try:
        return v16r2.main(argv)
    finally:
        v16.build_real_source_paired_records_full644_v16 = original_builder
        v16r2.checkpoint_receipt = original_receipt
        v16r2.validate_args = original_validate


if __name__ == "__main__":
    raise SystemExit(main())

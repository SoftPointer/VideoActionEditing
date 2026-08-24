#!/usr/bin/env python3
"""v16r6-C: exact32 two-sided same-action student-delta gradient fix.

The sole training change is the gradient estimator for the already-defined
same-action scalar delta loss.  The route-on branch is backpropagated first;
then route-off is recomputed on the identical action record and the same loss
is backpropagated with the route-on prediction detached.  Their sum is exactly
``J_on - J_off`` and never retains both 30-block graphs simultaneously.

Full rank-256 LoRA, the 1e-6 active-RMS optimizer, data, teacher targets,
loss values, seed, and v16r5 dual-descent geometry are unchanged.  This is a
sealed-manifest prefix32 diagnostic, not exact644 or a scientific checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r5 as parent
import train_online_anchor_attention_v16r6_debug_common as debug


METHOD = "bernini-online-anchor-v16r6c-two-sided-student-delta-prefix32"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-v16r6c-two-sided-student-delta-prefix32-receipt-v1"
)
DECODED_CANARY_SCHEMA = "bernini-v16r6c-two-sided-delta-prefix32-canary-v1"
VARIANT = "v16r6c_full_lora_two_sided_student_delta_lr1e6_prefix32"
CHANGED_VARIABLE = "same_action_student_delta_gradient_estimator_only"
LEARNING_RATE = 1.0e-6
GRADIENT_MODE = "two_sided_sequential_j_on_minus_j_off_v16r6c"


base = parent.base
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def validate_args(args: argparse.Namespace) -> None:
    """Validate the unchanged v16r5 contract through a 644-step shadow."""

    shadow = argparse.Namespace(**vars(args))
    shadow.max_steps = parent.v16.FULL644_ROWS
    shadow.learning_rate = LEARNING_RATE
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")
    _PARENT_VALIDATE_ARGS(shadow)

    if int(getattr(args, "max_steps", -1)) != debug.DEBUG_STEPS:
        fail("v16r6c requires --max-steps=32")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6c keeps the v16r5 active-coordinate RMS learning rate 1e-6")
    if "v16r6c" not in str(Path(args.output)).lower():
        fail("v16r6c output path must carry an explicit v16r6c namespace")


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    receipt = _PARENT_CHECKPOINT_RECEIPT(**kwargs)
    receipt = debug.decorate_debug_receipt(
        receipt,
        method=METHOD,
        schema=RECEIPT_SCHEMA,
        variant=VARIANT,
        changed_variable=CHANGED_VARIABLE,
    )
    contract = receipt.get("training_contract")
    absorption = receipt.get("source_absorption_diagnostic")
    if not isinstance(contract, MutableMapping) or not isinstance(
        absorption, Mapping
    ):
        fail("v16r6c inherited receipt sections differ")
    expected_lora = (
        "all_30_blocks_attn1_attn2_qkvo",
        240,
        188_743_680,
    )
    observed_lora = (
        contract.get("lora_scope"),
        contract.get("lora_target_module_count"),
        contract.get("trainable_parameter_count"),
    )
    if observed_lora != expected_lora:
        fail("v16r6c full-LoRA closure changed")
    if (
        contract.get("same_action_route_off_gradient_enabled") is not True
        or contract.get("same_action_student_delta_gradient_mode") != GRADIENT_MODE
        or absorption.get("applicable") is not True
        or absorption.get("micro_count") != 2
    ):
        fail("v16r6c two-sided route-off gradient execution differs")

    contract.update(
        {
            "same_action_route_off_recomputed_with_grad": True,
            "same_action_route_off_record_is_action_record": True,
            "same_action_route_on_prediction_detached_on_source_side": True,
            "same_action_student_delta_jacobian": "J_route_on_minus_J_route_off",
            "simultaneous_two_30_block_graph_retention": False,
            "learning_rate_changed_from_v16r5": False,
            "lora_scope_changed_from_v16r5": False,
            "training_data_changed_from_v16r5": False,
            "teacher_target_and_scalar_loss_value_changed_from_v16r5": False,
            "optimizer_and_dual_descent_geometry_changed_from_v16r5": False,
        }
    )
    receipt["v16r6c_two_sided_delta_contract"] = {
        "gradient_mode": GRADIENT_MODE,
        "student_delta_jacobian": "J_route_on_minus_J_route_off",
        "route_off_record": "same_action_record_same_state_same_timestep",
        "route_off_forward_has_grad": True,
        "route_on_prediction_detached_during_route_off_backward": True,
        "sequential_backward": True,
        "simultaneous_two_30_block_graph_retention": False,
        "learning_rate": LEARNING_RATE,
        "lora_scope": expected_lora[0],
        "lora_target_module_count": expected_lora[1],
        "trainable_parameter_count": expected_lora[2],
        "sole_changed_training_variable": CHANGED_VARIABLE,
    }
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parent_patches = {
        "METHOD": METHOD,
        "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "DECODED_CANARY_SCHEMA": DECODED_CANARY_SCHEMA,
        "validate_args": validate_args,
        "checkpoint_receipt": checkpoint_receipt,
    }
    base_patches = {
        "SAME_ACTION_ROUTE_OFF_GRADIENT_ENABLED": True,
        "SAME_ACTION_STUDENT_DELTA_GRADIENT_MODE": GRADIENT_MODE,
    }
    parent_originals = {name: getattr(parent, name) for name in parent_patches}
    base_originals = {name: getattr(base, name) for name in base_patches}
    for name, value in parent_patches.items():
        setattr(parent, name, value)
    for name, value in base_patches.items():
        setattr(base, name, value)
    try:
        return debug.run_v16r5_debug32(parent, argv)
    finally:
        for name, value in reversed(tuple(base_originals.items())):
            setattr(base, name, value)
        for name, value in reversed(tuple(parent_originals.items())):
            setattr(parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())

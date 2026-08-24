#!/usr/bin/env python3
"""v16r6-A: exact32 active-RMS 1e-7 ablation, otherwise v16r5 exact.

The sole training-variable change is the global active-coordinate RMS learning
rate, reduced from 1e-6 to 1e-7.  The full 240-module rank-256 LoRA closure,
data, objectives, seed, and capped dual-descent direction are unchanged.

This is a sealed-manifest prefix32 diagnostic, not exact644 and not a
promotable scientific checkpoint.
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


METHOD = "bernini-online-anchor-v16r6a-full-lora-lr1e7-prefix32"
RECEIPT_SCHEMA = "bernini-online-anchor-v16r6a-full-lora-lr1e7-prefix32-receipt-v1"
DECODED_CANARY_SCHEMA = "bernini-v16r6a-full-lora-lr1e7-prefix32-canary-v1"
VARIANT = "v16r6a_full_lora_active_rms_lr1e7_prefix32"
CHANGED_VARIABLE = "active_coordinate_rms_learning_rate_only"
LEARNING_RATE = 1.0e-7
V16R5_LEARNING_RATE = 1.0e-6
V16R5_COMPONENT_GRADIENT_EPSILON = 1.0e-12
COMPONENT_GRADIENT_EPSILON_SCALE = LEARNING_RATE / V16R5_LEARNING_RATE
COMPONENT_GRADIENT_EPSILON = (
    V16R5_COMPONENT_GRADIENT_EPSILON * COMPONENT_GRADIENT_EPSILON_SCALE
)
OPTIMIZER = "global_rms_normalized_capped_halfspace_sgd_lr1e7_v16r6a"
OPTIMIZER_FAILURE_POLICY = (
    "fail_closed_no_retry_no_action_only_fallback_no_optimizer_state_reset_v16r6a"
)


base = parent.base
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def validate_args(args: argparse.Namespace) -> None:
    """Validate every v16r5 input through a 644-step/1e-6 shadow."""

    shadow = argparse.Namespace(**vars(args))
    shadow.max_steps = parent.v16.FULL644_ROWS
    shadow.learning_rate = V16R5_LEARNING_RATE
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")

    # The saved function resolves LEARNING_RATE in the parent module at call
    # time.  Restore the v16r5 value only for its immutable-contract shadow.
    active_rate = parent.LEARNING_RATE
    parent.LEARNING_RATE = V16R5_LEARNING_RATE
    try:
        _PARENT_VALIDATE_ARGS(shadow)
    finally:
        parent.LEARNING_RATE = active_rate

    if int(getattr(args, "max_steps", -1)) != debug.DEBUG_STEPS:
        fail("v16r6a requires --max-steps=32")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6a requires active-coordinate RMS learning rate 1e-7")
    if "v16r6a" not in str(Path(args.output)).lower():
        fail("v16r6a output path must carry an explicit v16r6a namespace")


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
    source = receipt.get("v16r5_source_descent_summary")
    probes = receipt.get("component_gradient_probes")
    if not isinstance(contract, MutableMapping) or not isinstance(
        source, MutableMapping
    ) or not isinstance(probes, Mapping):
        fail("v16r6a inherited receipt sections differ")
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
        fail("v16r6a full-LoRA closure changed")
    action_probe = probes.get("action_objective")
    replay_probe = probes.get("raw_source_caption_trajectory_replay")
    if (
        contract.get("component_gradient_epsilon")
        != COMPONENT_GRADIENT_EPSILON
        or not isinstance(action_probe, Mapping)
        or not isinstance(replay_probe, Mapping)
        or action_probe.get("epsilon") != COMPONENT_GRADIENT_EPSILON
        or replay_probe.get("epsilon") != COMPONENT_GRADIENT_EPSILON
    ):
        fail("v16r6a scale-equivalent component support audit differs")

    for row in (contract, source):
        row.update(
            {
                "optimizer": OPTIMIZER,
                "optimizer_scalar_learning_rate": LEARNING_RATE,
                "optimizer_unattenuated_base_active_coordinate_rms": LEARNING_RATE,
                "numeric_learning_rate_ratio_to_v16r3_adamw": 0.01,
            }
        )
    contract.update(
        {
            "optimizer_current_effective_active_coordinate_rms": (
                source["optimizer_current_effective_active_coordinate_rms"]
            ),
            "learning_rate_changed_from_v16r5": True,
            "lora_scope_changed_from_v16r5": False,
            "training_data_changed_from_v16r5": False,
            "action_and_source_objectives_changed_from_v16r5": False,
            "component_gradient_epsilon_is_training_variable": False,
        }
    )
    receipt["v16r6a_learning_rate_contract"] = {
        "optimizer": OPTIMIZER,
        "active_coordinate_rms_learning_rate": LEARNING_RATE,
        "v16r5_active_coordinate_rms_learning_rate": V16R5_LEARNING_RATE,
        "ratio_to_v16r5": LEARNING_RATE / V16R5_LEARNING_RATE,
        "lora_scope": expected_lora[0],
        "lora_target_module_count": expected_lora[1],
        "trainable_parameter_count": expected_lora[2],
        "sole_changed_training_variable": CHANGED_VARIABLE,
    }
    receipt["v16r6a_scale_equivalent_gradient_audit"] = {
        "v16r5_absolute_epsilon": V16R5_COMPONENT_GRADIENT_EPSILON,
        "learning_rate_ratio_to_v16r5": COMPONENT_GRADIENT_EPSILON_SCALE,
        "effective_absolute_epsilon": COMPONENT_GRADIENT_EPSILON,
        "support_requirement_changed": False,
        "required_action_tensor_count_step2_plus": 480,
        "required_raw_replay_tensor_count_step2_plus": 480,
        "training_gradient_loss_optimizer_or_data_changed_by_audit": False,
        "purpose": "avoid_absolute_threshold_false_negative_under_pure_lr_scaling",
    }
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parent_patches = {
        "METHOD": METHOD,
        "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "DECODED_CANARY_SCHEMA": DECODED_CANARY_SCHEMA,
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER": OPTIMIZER,
        "OPTIMIZER_FAILURE_POLICY": OPTIMIZER_FAILURE_POLICY,
        "validate_args": validate_args,
        "checkpoint_receipt": checkpoint_receipt,
    }
    v16r4_patches = {
        "LEARNING_RATE": LEARNING_RATE,
        "OPTIMIZER": OPTIMIZER,
        "OPTIMIZER_FAILURE_POLICY": OPTIMIZER_FAILURE_POLICY,
    }
    base_patches = {
        "COMPONENT_GRADIENT_EPSILON": COMPONENT_GRADIENT_EPSILON,
    }
    parent_originals = {name: getattr(parent, name) for name in parent_patches}
    v16r4_originals = {
        name: getattr(parent.parent, name) for name in v16r4_patches
    }
    base_originals = {name: getattr(base, name) for name in base_patches}
    for name, value in parent_patches.items():
        setattr(parent, name, value)
    for name, value in v16r4_patches.items():
        setattr(parent.parent, name, value)
    for name, value in base_patches.items():
        setattr(base, name, value)
    try:
        return debug.run_v16r5_debug32(parent, argv)
    finally:
        for name, value in reversed(tuple(base_originals.items())):
            setattr(base, name, value)
        for name, value in reversed(tuple(v16r4_originals.items())):
            setattr(parent.parent, name, value)
        for name, value in reversed(tuple(parent_originals.items())):
            setattr(parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())

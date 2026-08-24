#!/usr/bin/env python3
"""v16r6-E: route-attn1-Q/K LoRA plus an absolute route-off FM anchor.

This exact32 diagnostic composes the winning v16r6-B trainable closure with
the v16r6-D same-state restoring objective.  Relative to B, the sole training
change is a fixed 0.025-weight FM spring from the student's route-off output to
the adapter-disabled route-off teacher at the identical noisy state/timestep.

The run remains a sealed-manifest, 32-distinct-IID diagnostic.  It is not an
exact644 run and does not authorize a scientific or visual-quality claim.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r5 as parent
import train_online_anchor_attention_v16r6_debug_common as debug


METHOD = (
    "bernini-online-anchor-v16r6e-route-attn1-qk-absolute-route-off-anchor-"
    "prefix32"
)
RECEIPT_SCHEMA = (
    "bernini-online-anchor-v16r6e-route-attn1-qk-absolute-route-off-anchor-"
    "prefix32-receipt-v1"
)
DECODED_CANARY_SCHEMA = (
    "bernini-v16r6e-route-attn1-qk-absolute-anchor-prefix32-canary-v1"
)
VARIANT = "v16r6e_route_qk_absolute_route_off_anchor_lr1e6_prefix32"
CHANGED_VARIABLE = (
    "same_state_route_off_absolute_common_mode_fm_anchor_added_to_v16r6b_scope"
)
LEARNING_RATE = 1.0e-6

LORA_SCOPE = "route_blocks_22_attn1_qk_only"
LORA_TARGET_MODULE_COUNT = 44
LORA_TRAINABLE_TENSOR_COUNT = 88
LORA_PARAMETERS = 34_603_008
TARGET_MODULES_SHA256 = (
    "55d23681c5ee165e6f6b94f97730d7fe7e93031a0b83fd6ede20ce316f905cb4"
)
TARGET_PATTERN = re.compile(
    r"^diff_dec\.transformer\.blocks\.(?P<block>\d+)\."
    r"(?P<attention>attn1)\.(?P<projection>to_q|to_k)$"
)

ABSOLUTE_ANCHOR_WEIGHT = 0.025
ABSOLUTE_ANCHOR_MODE = (
    "same_state_route_off_frozen_base_fm_weight0025_v16r6e"
)
LEGACY_DELTA_GRADIENT_MODE = "route_on_only_legacy"


base = parent.base
ROUTE_BLOCKS = tuple(base.ROUTE_BLOCKS)
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def select_route_attn1_qk_target_names(renderer: Any) -> tuple[str, ...]:
    """Return the exact 44 route-block attn1 Q/K projections."""

    full = tuple(base.legacy.select_attention_projection_names(renderer))
    selected: list[str] = []
    for name in full:
        match = TARGET_PATTERN.fullmatch(name)
        if match is not None and int(match.group("block")) in ROUTE_BLOCKS:
            selected.append(name)
    result = tuple(sorted(selected))
    observed_blocks = {
        int(TARGET_PATTERN.fullmatch(name).group("block"))  # type: ignore[union-attr]
        for name in result
    }
    if (
        len(full) != 240
        or len(result) != LORA_TARGET_MODULE_COUNT
        or len(set(result)) != LORA_TARGET_MODULE_COUNT
        or observed_blocks != set(ROUTE_BLOCKS)
        or base.legacy.object_sha256(list(result)) != TARGET_MODULES_SHA256
    ):
        fail("v16r6e route-attn1-QK target closure differs")
    return result


def validate_args(args: argparse.Namespace) -> None:
    """Validate unchanged v16r5 inputs through an exact644 shadow."""

    shadow = argparse.Namespace(**vars(args))
    shadow.max_steps = parent.v16.FULL644_ROWS
    shadow.learning_rate = LEARNING_RATE
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")
    _PARENT_VALIDATE_ARGS(shadow)

    if int(getattr(args, "max_steps", -1)) != debug.DEBUG_STEPS:
        fail("v16r6e requires --max-steps=32")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6e keeps the v16r5 active-coordinate RMS learning rate 1e-6")
    if "v16r6e" not in str(Path(args.output)).lower():
        fail("v16r6e output path must carry an explicit v16r6e namespace")


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
    anchor = receipt.get("route_off_absolute_anchor_diagnostic")
    memory = receipt.get("memory_gate")
    if (
        not isinstance(contract, MutableMapping)
        or not isinstance(anchor, Mapping)
        or not isinstance(memory, MutableMapping)
    ):
        fail("v16r6e inherited receipt sections differ")

    observed_scope = (
        contract.get("lora_scope"),
        contract.get("lora_target_module_count"),
        contract.get("lora_target_modules_sha256"),
        contract.get("trainable_parameter_count"),
    )
    expected_scope = (
        LORA_SCOPE,
        LORA_TARGET_MODULE_COUNT,
        TARGET_MODULES_SHA256,
        LORA_PARAMETERS,
    )
    if observed_scope != expected_scope:
        fail("v16r6e receipt LoRA closure differs")
    if (
        contract.get("same_action_route_off_gradient_enabled") is not False
        or contract.get("same_action_student_delta_gradient_mode")
        != LEGACY_DELTA_GRADIENT_MODE
        or contract.get("same_action_route_off_absolute_anchor_enabled")
        is not True
        or contract.get("same_action_route_off_absolute_anchor_weight")
        != ABSOLUTE_ANCHOR_WEIGHT
        or contract.get("same_action_route_off_absolute_anchor_mode")
        != ABSOLUTE_ANCHOR_MODE
        or anchor.get("applicable") is not True
        or anchor.get("micro_count") != 2
        or anchor.get("weight") != ABSOLUTE_ANCHOR_WEIGHT
        or anchor.get("student_delta_gradient_mode")
        != LEGACY_DELTA_GRADIENT_MODE
    ):
        fail("v16r6e absolute route-off anchor execution differs")

    contract.update(
        {
            "full_attention_lora_enabled": False,
            "lora_scope_changed_from_v16r5": True,
            "lora_target_attention": "attn1",
            "lora_target_projections": ["to_k", "to_q"],
            "lora_target_blocks": list(ROUTE_BLOCKS),
            "lora_nonroute_blocks_trainable": False,
            "lora_attn2_trainable": False,
            "lora_value_or_output_trainable": False,
            "same_action_student_delta_jacobian": "J_route_on_only_legacy",
            "absolute_common_mode_fm_preservation_objective_added": True,
            "decoded_source_preservation_claimed": False,
            "learning_rate_changed_from_v16r6b": False,
            "lora_scope_changed_from_v16r6b": False,
            "training_data_changed_from_v16r6b": False,
            "optimizer_and_dual_descent_geometry_changed_from_v16r6b": False,
            "action_component_gradient_contains": [
                "legacy_route_on_only_student_delta",
                "weighted_absolute_route_off_common_mode_anchor",
            ],
        }
    )
    memory.update(
        {
            "capture_phase": (
                "after_action_delta_absolute_route_off_anchor_and_raw_replay_"
                "backwards_before_actual_update_audit_clones"
            ),
            "absolute_route_off_anchor_training_allocations_included": True,
        }
    )
    receipt["v16r6e_lora_scope_contract"] = {
        "scope": LORA_SCOPE,
        "target_module_count": LORA_TARGET_MODULE_COUNT,
        "trainable_tensor_count": LORA_TRAINABLE_TENSOR_COUNT,
        "trainable_parameter_count": LORA_PARAMETERS,
        "target_modules_sha256": TARGET_MODULES_SHA256,
        "route_blocks": list(ROUTE_BLOCKS),
        "attention": "attn1",
        "projections": ["to_k", "to_q"],
        "forbidden": ["attn2", "to_v", "to_out.0", "nonroute_blocks"],
    }
    receipt["v16r6e_absolute_route_off_anchor_contract"] = {
        "mode": ABSOLUTE_ANCHOR_MODE,
        "weight": ABSOLUTE_ANCHOR_WEIGHT,
        "student_record": "same_action_record_same_noisy_state_same_timestep",
        "teacher": "adapter_disabled_route_off_routed_teacher_source",
        "teacher_detached": True,
        "student_route_off_forward_has_grad": True,
        "student_delta_gradient_mode": LEGACY_DELTA_GRADIENT_MODE,
        "student_delta_jacobian": "J_route_on_only_legacy",
        "sequential_backward": True,
        "simultaneous_two_30_block_graph_retention": False,
        "decoded_source_preservation_claimed": False,
        "learning_rate": LEARNING_RATE,
        "lora_scope": LORA_SCOPE,
        "lora_target_module_count": LORA_TARGET_MODULE_COUNT,
        "trainable_parameter_count": LORA_PARAMETERS,
        "baseline_variant": "v16r6b_route_blocks_attn1_qk_only_lr1e6_prefix32",
        "sole_changed_training_variable_from_v16r6b": CHANGED_VARIABLE,
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
        "LORA_SCOPE": LORA_SCOPE,
        "LORA_TARGET_MODULE_COUNT": LORA_TARGET_MODULE_COUNT,
        "LORA_TRAINABLE_TENSOR_COUNT": LORA_TRAINABLE_TENSOR_COUNT,
        "LORA_PARAMETERS": LORA_PARAMETERS,
        "select_lora_target_names": select_route_attn1_qk_target_names,
        "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_ENABLED": True,
        "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_WEIGHT": ABSOLUTE_ANCHOR_WEIGHT,
        "SAME_ACTION_ROUTE_OFF_ABSOLUTE_ANCHOR_MODE": ABSOLUTE_ANCHOR_MODE,
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

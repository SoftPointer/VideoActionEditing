#!/usr/bin/env python3
"""v16r6-D: exact32 same-state route-off absolute FM anchor diagnostic.

The sole training change is a fixed-weight restoring spring on the student's
route-off prediction at the exact action noisy state/timestep.  Its target is
the already-computed adapter-disabled route-off ``routed_teacher_source``.
The historical student-delta loss keeps its route-on-only Jacobian, so this is
strictly separate from v16r6-C's two-sided ``J_on-J_off`` estimator.

The anchor weight is 0.025: it uses the same FM units and conservative nominal
coefficient as the existing source-caption replay while targeting a different,
absolute same-state quantity.  Full LoRA, active-RMS lr=1e-6, data, teacher
delta, replay, optimizer, and seed remain v16r5-identical.  This prefix32 run
is diagnostic only, not exact644 and not a decoded-preservation guarantee.
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


METHOD = "bernini-online-anchor-v16r6d-absolute-route-off-anchor-prefix32"
RECEIPT_SCHEMA = (
    "bernini-online-anchor-v16r6d-absolute-route-off-anchor-prefix32-receipt-v1"
)
DECODED_CANARY_SCHEMA = "bernini-v16r6d-absolute-anchor-prefix32-canary-v1"
VARIANT = "v16r6d_full_lora_absolute_route_off_anchor_lr1e6_prefix32"
CHANGED_VARIABLE = "same_state_route_off_absolute_common_mode_fm_anchor_only"
LEARNING_RATE = 1.0e-6
ABSOLUTE_ANCHOR_WEIGHT = 0.025
ABSOLUTE_ANCHOR_MODE = (
    "same_state_route_off_frozen_base_fm_weight0025_v16r6d"
)
LEGACY_DELTA_GRADIENT_MODE = "route_on_only_legacy"


base = parent.base
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def validate_args(args: argparse.Namespace) -> None:
    """Validate all unchanged v16r5 inputs through an exact644 shadow."""

    shadow = argparse.Namespace(**vars(args))
    shadow.max_steps = parent.v16.FULL644_ROWS
    shadow.learning_rate = LEARNING_RATE
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")
    _PARENT_VALIDATE_ARGS(shadow)

    if int(getattr(args, "max_steps", -1)) != debug.DEBUG_STEPS:
        fail("v16r6d requires --max-steps=32")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6d keeps the v16r5 active-coordinate RMS learning rate 1e-6")
    if "v16r6d" not in str(Path(args.output)).lower():
        fail("v16r6d output path must carry an explicit v16r6d namespace")


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
    if not isinstance(contract, MutableMapping) or not isinstance(
        anchor, Mapping
    ) or not isinstance(memory, MutableMapping):
        fail("v16r6d inherited receipt sections differ")
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
        fail("v16r6d full-LoRA closure changed")
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
        fail("v16r6d absolute route-off anchor execution differs")

    contract.update(
        {
            "same_action_student_delta_jacobian": "J_route_on_only_legacy",
            "same_action_route_off_absolute_anchor_record": (
                "same_action_record_same_noisy_state_same_timestep"
            ),
            "same_action_route_off_absolute_anchor_teacher": (
                "adapter_disabled_route_off_routed_teacher_source"
            ),
            "same_action_route_off_absolute_anchor_sequential_backward": True,
            "simultaneous_two_30_block_graph_retention": False,
            "decoded_source_preservation_claimed": False,
            "absolute_common_mode_fm_preservation_objective_added": True,
            "learning_rate_changed_from_v16r5": False,
            "lora_scope_changed_from_v16r5": False,
            "training_data_changed_from_v16r5": False,
            "student_delta_scalar_and_gradient_estimator_changed_from_v16r5": False,
            "source_caption_replay_changed_from_v16r5": False,
            "optimizer_and_dual_descent_geometry_changed_from_v16r5": False,
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
    receipt["v16r6d_absolute_route_off_anchor_contract"] = {
        "mode": ABSOLUTE_ANCHOR_MODE,
        "weight": ABSOLUTE_ANCHOR_WEIGHT,
        "weight_basis": (
            "same_fm_units_and_nominal_0025_as_existing_source_caption_replay"
        ),
        "student_record": "same_action_record_same_noisy_state_same_timestep",
        "teacher": "adapter_disabled_route_off_routed_teacher_source",
        "teacher_detached": True,
        "student_route_off_forward_has_grad": True,
        "student_delta_gradient_mode": LEGACY_DELTA_GRADIENT_MODE,
        "student_delta_jacobian": "J_route_on_only_legacy",
        "action_component_gradient_contains": [
            "legacy_route_on_only_student_delta",
            "weighted_absolute_route_off_common_mode_anchor",
        ],
        "sequential_backward": True,
        "simultaneous_two_30_block_graph_retention": False,
        "decoded_source_preservation_claimed": False,
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

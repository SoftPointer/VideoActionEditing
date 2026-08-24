#!/usr/bin/env python3
"""Shared fail-closed mechanics for one-variable v16r6 exact-prefix probes.

This module changes no training variable.  It only permits an authenticated
32-row prefix of the sealed 644-row schedule without pretending the result is
an exact644 checkpoint, and replaces v16r4's terminal 644-step assertion with
the exact debug budget.  Individual wrappers own exactly one experimental
change (LoRA scope or active-coordinate RMS learning rate).
"""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any, Optional


DEBUG_STEPS = 32
DEBUG_SCHEDULE = "sealed_full644_family_round_robin_prefix32_exact_once_debug"


def fail(base: Any, message: str) -> None:
    base.fail(message)


def run_v16r4_parent_debug32(parent: Any, argv: Optional[Sequence[str]]) -> int:
    """Execute the v16r4 runtime hooks with an exact 32-step terminal check."""

    parent._RUNTIME_AUDIT = parent._empty_runtime_audit()
    parent._CANARY_BINDING = None
    parent._ACTIVE_OPTIMIZER = None
    parent.v16r3._RUNTIME_AUDIT = parent.v16r3._empty_runtime_audit()

    import torch

    original_parser = parent.v16.build_parser
    original_validate = parent.v16.validate_args
    original_receipt = parent.v16.checkpoint_receipt
    original_real_builder = parent.v16.build_real_source_paired_records_full644_v16
    original_r2_receipt_parent = parent.v16._R2_CHECKPOINT_RECEIPT
    original_r2_merge = parent.r2.merge_component_gradients
    original_actual_probe = parent.base.actual_optimizer_update_probe
    original_adamw = torch.optim.AdamW

    parent.v16.build_parser = parent.build_parser
    parent.v16.validate_args = parent.validate_args
    parent.v16.checkpoint_receipt = parent.checkpoint_receipt
    parent.v16.build_real_source_paired_records_full644_v16 = (
        parent.build_real_source_paired_records_full644_dynamic_static_v16r4
    )
    parent.v16._R2_CHECKPOINT_RECEIPT = parent._V15_CHECKPOINT_RECEIPT
    parent.r2.merge_component_gradients = parent.merge_component_gradients
    parent.base.actual_optimizer_update_probe = parent.actual_optimizer_update_probe
    torch.optim.AdamW = parent._projected_optimizer_factory()
    try:
        result = parent.v16.main(argv)
        optimizer = parent._ACTIVE_OPTIMIZER
        if (
            optimizer is None
            or int(getattr(optimizer, "_v16r4_step_count", -1)) != DEBUG_STEPS
            or int(getattr(optimizer, "_v16r5_step_count", -1)) != DEBUG_STEPS
        ):
            fail(
                parent.base,
                "v16r6 debug completed without the exact 32-step optimizer closure",
            )
        return int(result)
    finally:
        torch.optim.AdamW = original_adamw
        parent.base.actual_optimizer_update_probe = original_actual_probe
        parent.r2.merge_component_gradients = original_r2_merge
        parent.v16._R2_CHECKPOINT_RECEIPT = original_r2_receipt_parent
        parent.v16.build_real_source_paired_records_full644_v16 = original_real_builder
        parent.v16.checkpoint_receipt = original_receipt
        parent.v16.validate_args = original_validate
        parent.v16.build_parser = original_parser
        parent._CANARY_BINDING = None
        parent._ACTIVE_OPTIMIZER = None


def run_v16r5_debug32(parent: Any, argv: Optional[Sequence[str]]) -> int:
    """Use v16r5 geometry while swapping only its terminal parent runner."""

    original_main = parent.parent.main

    def debug_parent_main(values: Optional[Sequence[str]] = None) -> int:
        return run_v16r4_parent_debug32(parent.parent, values)

    parent.parent.main = debug_parent_main
    try:
        return int(parent.main(argv))
    finally:
        parent.parent.main = original_main


def decorate_debug_receipt(
    receipt: dict[str, Any],
    *,
    method: str,
    schema: str,
    variant: str,
    changed_variable: str,
) -> dict[str, Any]:
    """Make the non-terminal scientific status impossible to misread."""

    step = int(receipt.get("global_step", 0))
    if step <= 0 or step > DEBUG_STEPS:
        raise ValueError("v16r6 debug receipt step is outside 1..32")
    contract = receipt.get("training_contract")
    if not isinstance(contract, MutableMapping):
        raise ValueError("v16r6 debug receipt has no mutable training contract")

    receipt["schema_version"] = schema
    receipt["complete"] = False
    receipt["exact644_training_complete"] = False
    receipt["terminal_full644_checkpoint"] = False
    receipt["scientific_claim_authorized"] = False
    receipt["claim_scope"] = (
        "engineering_exact32_prefix_ablation_only_non_scientific"
    )
    receipt["v16r6_debug_contract"] = {
        "variant": variant,
        "changed_variable": changed_variable,
        "sealed_manifest_row_count": 644,
        "debug_optimizer_step_budget": DEBUG_STEPS,
        "current_step": step,
        "debug_run_complete": step == DEBUG_STEPS,
        "exact644_training_complete": False,
        "terminal_full644_checkpoint": False,
        "schedule": DEBUG_SCHEDULE,
        "all_other_training_variables_inherited_from_v16r5": True,
        "scientific_claim_authorized": False,
    }
    contract.update(
        {
            "method": method,
            "max_steps": DEBUG_STEPS,
            "debug_optimizer_step_budget": DEBUG_STEPS,
            "debug_run_complete": step == DEBUG_STEPS,
            "exact644_training_complete": False,
            "terminal_full644_checkpoint": False,
            "single_continuous_fresh_from_base_exact644_run": False,
            "single_continuous_fresh_from_base_exact644_parameter_trajectory": False,
            "full644_optimizer_schedule": DEBUG_SCHEDULE,
            "all_full644_rows_targeted_exactly_once": False,
            "changed_variable_from_v16r5": changed_variable,
            "scientific_claim_authorized": False,
        }
    )
    summary = receipt.get("v16_full644_summary")
    if isinstance(summary, MutableMapping):
        summary["all_full644_rows_targeted_exactly_once"] = False
        summary["exact644_training_complete"] = False
        summary["debug_optimizer_step_budget"] = DEBUG_STEPS
        summary["debug_run_complete"] = step == DEBUG_STEPS
        summary["schedule"] = DEBUG_SCHEDULE
    return receipt


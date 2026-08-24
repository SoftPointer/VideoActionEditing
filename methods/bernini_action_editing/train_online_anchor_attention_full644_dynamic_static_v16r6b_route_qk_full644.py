#!/usr/bin/env python3
"""Promote the winning v16r6-B Route-QK scope to one exact644 run.

This wrapper keeps the v16r5 data, objectives, seed, gradient geometry,
optimizer, rank/alpha, and 1e-6 active-coordinate RMS step unchanged. The sole
training-variable change remains the v16r6-B adapter closure: Q/K projections
of attn1 in the 22 online-route blocks (44 modules, 88 LoRA tensors).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
from typing import Any, MutableMapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_online_anchor_attention_full644_dynamic_static_v16r5 as parent


METHOD = "bernini-online-anchor-v16r6b-route-attn1-qk-exact644"
RECEIPT_SCHEMA = "bernini-online-anchor-v16r6b-route-attn1-qk-exact644-receipt-v1"
DECODED_CANARY_SCHEMA = "bernini-v16r6b-route-attn1-qk-exact644-canary-v1"
VARIANT = "v16r6b_route_blocks_attn1_qk_only_lr1e6_exact644"
CHANGED_VARIABLE = "lora_target_scope_only"
LEARNING_RATE = 1.0e-6
FULL_STEPS = 644
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


base = parent.base
ROUTE_BLOCKS = tuple(base.ROUTE_BLOCKS)
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def trainer_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def overlay_revision(base_archive_sha256: str) -> str:
    payload = (
        "v16r6b-exact644-overlay-v1\0"
        + str(base_archive_sha256)
        + "\0"
        + trainer_source_sha256()
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_route_attn1_qk_target_names(renderer: Any) -> tuple[str, ...]:
    """Select the exact v16r6-B 44-module closure from the frozen registry."""

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
        fail("v16r6b exact644 route-attn1-QK target closure differs")
    return result


def validate_args(args: argparse.Namespace) -> None:
    """Validate the complete v16r5 contract, then bind the promoted namespace."""

    shadow = argparse.Namespace(**vars(args))
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")
    _PARENT_VALIDATE_ARGS(shadow)
    if int(getattr(args, "max_steps", -1)) != FULL_STEPS:
        fail("v16r6b exact644 requires --max-steps=644")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6b exact644 keeps the v16r5 active-coordinate RMS learning rate 1e-6")
    output = str(Path(args.output)).lower()
    if "v16r6b" not in output or "exact644" not in output:
        fail("v16r6b exact644 output path must carry both namespace markers")
    archive_sha = str(getattr(args, "method_source_archive_sha256", ""))
    revision = str(getattr(args, "method_source_revision", ""))
    if revision != overlay_revision(archive_sha):
        fail("v16r6b exact644 base-plus-overlay source revision differs")


def checkpoint_receipt(**kwargs: Any) -> dict[str, Any]:
    """Bind every checkpoint to the promoted scope without changing semantics."""

    receipt = _PARENT_CHECKPOINT_RECEIPT(**kwargs)
    contract = receipt.get("training_contract")
    if not isinstance(contract, MutableMapping):
        fail("v16r6b exact644 inherited training contract differs")
    observed = (
        contract.get("lora_scope"),
        contract.get("lora_target_module_count"),
        contract.get("lora_target_modules_sha256"),
        contract.get("trainable_parameter_count"),
    )
    expected = (
        LORA_SCOPE,
        LORA_TARGET_MODULE_COUNT,
        TARGET_MODULES_SHA256,
        LORA_PARAMETERS,
    )
    if observed != expected:
        fail("v16r6b exact644 receipt LoRA closure differs")

    receipt["schema_version"] = RECEIPT_SCHEMA
    contract.update(
        {
            "method": METHOD,
            "full_attention_lora_enabled": False,
            "lora_scope_changed_from_v16r5": True,
            "lora_target_attention": "attn1",
            "lora_target_projections": ["to_k", "to_q"],
            "lora_target_blocks": list(ROUTE_BLOCKS),
            "lora_nonroute_blocks_trainable": False,
            "lora_attn2_trainable": False,
            "lora_value_or_output_trainable": False,
            "learning_rate_changed_from_v16r5": False,
            "training_data_changed_from_v16r5": False,
            "action_and_source_objectives_changed_from_v16r5": False,
            "s32_selection_used_only_for_scope_selection": True,
            "s32_checkpoint_used_as_initialization": False,
            "fresh_from_frozen_base": True,
        }
    )
    receipt["v16r6b_full644_scope_contract"] = {
        "variant": VARIANT,
        "scope": LORA_SCOPE,
        "target_module_count": LORA_TARGET_MODULE_COUNT,
        "trainable_tensor_count": LORA_TRAINABLE_TENSOR_COUNT,
        "trainable_parameter_count": LORA_PARAMETERS,
        "target_modules_sha256": TARGET_MODULES_SHA256,
        "route_blocks": list(ROUTE_BLOCKS),
        "attention": "attn1",
        "projections": ["to_k", "to_q"],
        "forbidden": ["attn2", "to_v", "to_out.0", "nonroute_blocks"],
        "sole_changed_training_variable_from_v16r5": CHANGED_VARIABLE,
        "optimizer_step_budget": FULL_STEPS,
        "selected_by_s32_heldout8_automatic_four_axis_fusion": True,
        "selection_does_not_authorize_scientific_claim": True,
        "base_source_archive_sha256": str(
            getattr(kwargs.get("args"), "method_source_archive_sha256", "")
        ),
        "trainer_overlay_sha256": trainer_source_sha256(),
        "base_plus_overlay_revision_sha256": str(
            getattr(kwargs.get("args"), "method_source_revision", "")
        ),
    }
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Install the Route-QK scope for one v16r5 exact644 invocation."""

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
    }
    parent_originals = {name: getattr(parent, name) for name in parent_patches}
    base_originals = {name: getattr(base, name) for name in base_patches}
    for name, value in parent_patches.items():
        setattr(parent, name, value)
    for name, value in base_patches.items():
        setattr(base, name, value)
    try:
        return parent.main(argv)
    finally:
        for name, value in reversed(tuple(base_originals.items())):
            setattr(base, name, value)
        for name, value in reversed(tuple(parent_originals.items())):
            setattr(parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())

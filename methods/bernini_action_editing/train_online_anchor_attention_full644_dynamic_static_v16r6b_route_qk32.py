#!/usr/bin/env python3
"""v16r6-B: exact32 LoRA-scope ablation, otherwise identical to v16r5.

The sole training-variable change is the trainable adapter closure: only Q/K
projections of ``attn1`` in the 22 blocks where the online route is actually
installed are adapted.  Data, objectives, seed, rank/alpha, v16r5 gradient
geometry, and the 1e-6 active-coordinate RMS optimizer step are unchanged.

This is deliberately a sealed-manifest prefix32 diagnostic, not exact644 and
not a promotable scientific checkpoint.
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


METHOD = "bernini-online-anchor-v16r6b-route-attn1-qk-prefix32"
RECEIPT_SCHEMA = "bernini-online-anchor-v16r6b-route-attn1-qk-prefix32-receipt-v1"
DECODED_CANARY_SCHEMA = "bernini-v16r6b-route-attn1-qk-prefix32-canary-v1"
VARIANT = "v16r6b_route_blocks_attn1_qk_only_lr1e6_prefix32"
CHANGED_VARIABLE = "lora_target_scope_only"
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


base = parent.base
ROUTE_BLOCKS = tuple(base.ROUTE_BLOCKS)
_PARENT_VALIDATE_ARGS = parent.validate_args
_PARENT_CHECKPOINT_RECEIPT = parent.checkpoint_receipt


def fail(message: str) -> None:
    base.fail(message)


def select_route_attn1_qk_target_names(renderer: Any) -> tuple[str, ...]:
    """Filter the frozen v16r5 240-module registry to the exact 44 targets."""

    full = tuple(base.legacy.select_attention_projection_names(renderer))
    selected: list[str] = []
    for name in full:
        match = TARGET_PATTERN.fullmatch(name)
        if match is None:
            continue
        if int(match.group("block")) in ROUTE_BLOCKS:
            selected.append(name)
    result = tuple(sorted(selected))
    if (
        len(full) != 240
        or len(result) != LORA_TARGET_MODULE_COUNT
        or len(set(result)) != LORA_TARGET_MODULE_COUNT
        or {
            int(TARGET_PATTERN.fullmatch(name).group("block"))  # type: ignore[union-attr]
            for name in result
        }
        != set(ROUTE_BLOCKS)
        or base.legacy.object_sha256(list(result)) != TARGET_MODULES_SHA256
    ):
        fail("v16r6b route-attn1-QK target closure differs")
    return result


def validate_args(args: argparse.Namespace) -> None:
    """Validate every v16r5 input through a 644-step shadow, then bind 32."""

    shadow = argparse.Namespace(**vars(args))
    shadow.max_steps = parent.v16.FULL644_ROWS
    shadow.learning_rate = parent.LEARNING_RATE
    shadow.output = Path(str(args.output) + ".v16r5-contract-shadow")
    _PARENT_VALIDATE_ARGS(shadow)

    if int(getattr(args, "max_steps", -1)) != debug.DEBUG_STEPS:
        fail("v16r6b requires --max-steps=32")
    if float(getattr(args, "learning_rate", float("nan"))) != LEARNING_RATE:
        fail("v16r6b keeps the v16r5 active-coordinate RMS learning rate 1e-6")
    if "v16r6b" not in str(Path(args.output)).lower():
        fail("v16r6b output path must carry an explicit v16r6b namespace")


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
    if not isinstance(contract, MutableMapping):
        fail("v16r6b inherited training contract differs")
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
        fail("v16r6b receipt LoRA closure differs")
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
            "learning_rate_changed_from_v16r5": False,
            "training_data_changed_from_v16r5": False,
            "action_and_source_objectives_changed_from_v16r5": False,
        }
    )
    receipt["v16r6b_lora_scope_contract"] = {
        "scope": LORA_SCOPE,
        "target_module_count": LORA_TARGET_MODULE_COUNT,
        "trainable_tensor_count": LORA_TRAINABLE_TENSOR_COUNT,
        "trainable_parameter_count": LORA_PARAMETERS,
        "target_modules_sha256": TARGET_MODULES_SHA256,
        "route_blocks": list(ROUTE_BLOCKS),
        "attention": "attn1",
        "projections": ["to_k", "to_q"],
        "forbidden": ["attn2", "to_v", "to_out.0", "nonroute_blocks"],
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
        return debug.run_v16r5_debug32(parent, argv)
    finally:
        for name, value in reversed(tuple(base_originals.items())):
            setattr(base, name, value)
        for name, value in reversed(tuple(parent_originals.items())):
            setattr(parent, name, value)


if __name__ == "__main__":
    raise SystemExit(main())


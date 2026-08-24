#!/usr/bin/env python3
"""All-30-block rank-256 action adapter for native Bernini RV2V preference.

Unlike the old PAIR-v5 adapter, this surface covers q/k/v/out in both self-
and cross-attention in every block (240 affine projections, 188,743,680
trainable parameters).  It intentionally has no 32-D statistic, band loss,
sparse block list, or target-only low-rank route.  The reference policy is the
same model with PEFT disabled exactly for the corresponding native branch.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import inference_sigma_strata as sigma_strata
import packed_preservation_lora_v2 as large_core


SCHEMA_VERSION = "bernini-interaction-large-action-adapter-v1"
TOTAL_BLOCKS_1P3B = 30
ACTION_BLOCK_INDICES = tuple(range(TOTAL_BLOCKS_1P3B))
ACTION_LORA_RANK = large_core.LORA_RANK
ACTION_LORA_ALPHA = float(large_core.LORA_ALPHA)
ACTION_LORA_DROPOUT = 0.0
HIGH_SIGMA_INDICES = tuple(range(33))
MID_SIGMA_INDICES = tuple(range(33, 38))
LOW_SIGMA_INDICES = tuple(range(38, 40))
NATIVE_BRANCHES = ("none", "V", "I", "VI")
EXPECTED_TRAINABLE_PARAMETERS = large_core.EXPECTED_LORA_PARAMETER_COUNTS[
    "all-attention"
]


class LargeActionAdapterError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sigma_gate(schedule_index: int) -> tuple[str, float]:
    if type(schedule_index) is not int or not 0 <= schedule_index < 40:
        raise LargeActionAdapterError("sigma schedule index must be in [0,39]")
    if schedule_index in HIGH_SIGMA_INDICES:
        return "high", 1.0
    if schedule_index in MID_SIGMA_INDICES:
        return "mid", 1.0
    if schedule_index in LOW_SIGMA_INDICES:
        return "low_base_only", 0.0
    raise LargeActionAdapterError("sigma schedule partition differs")


if HIGH_SIGMA_INDICES + MID_SIGMA_INDICES + LOW_SIGMA_INDICES != tuple(range(40)):
    raise RuntimeError("large action sigma partition is not exact40")
if len(sigma_strata.PINNED_POSITIVE_SIGMAS) != 40:
    raise RuntimeError("pinned native schedule is not exact40")


@dataclass(frozen=True)
class PairV5ActionRoute:
    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    branch_name: str
    sigma_schedule_index: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.total_tokens) is not int
            or self.total_tokens <= 0
            or type(self.condition_tokens) is not int
            or not 0 <= self.condition_tokens < self.total_tokens
            or type(self.sequence_parallel_rank) is not int
            or type(self.sequence_parallel_size) is not int
            or self.sequence_parallel_size not in (1, 4)
            or not 0 <= self.sequence_parallel_rank < self.sequence_parallel_size
            or self.branch_name not in NATIVE_BRANCHES
            or type(self.enabled) is not bool
        ):
            raise LargeActionAdapterError("native large-action route differs")
        if (self.branch_name == "none") != (self.condition_tokens == 0):
            raise LargeActionAdapterError("native branch condition-token contract differs")
        sigma_gate(self.sigma_schedule_index)

    @property
    def adapter_active(self) -> bool:
        return self.enabled and sigma_gate(self.sigma_schedule_index)[1] > 0.0


_ACTIVE_ROUTE: ContextVar[Optional[PairV5ActionRoute]] = ContextVar(
    "interaction_large_action_route_v1", default=None
)


@dataclass
class LargeActionAdapterHandle:
    model: Any
    transformer: Any
    specs: tuple[large_core.ProjectionSpec, ...]
    architecture: Mapping[str, Any]

    def trainable_named_parameters(self) -> tuple[tuple[str, Any], ...]:
        result = tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        if not result or len({name for name, _ in result}) != len(result):
            raise LargeActionAdapterError("large adapter trainable inventory differs")
        if any(".lora_A." not in name and ".lora_B." not in name for name, _ in result):
            raise LargeActionAdapterError("non-LoRA parameter entered large adapter")
        count = sum(int(parameter.numel()) for _, parameter in result)
        if count != EXPECTED_TRAINABLE_PARAMETERS:
            raise LargeActionAdapterError(
                f"large adapter trainable count differs: {count} != {EXPECTED_TRAINABLE_PARAMETERS}"
            )
        return result

    def base_parameters_frozen(self) -> bool:
        try:
            return sum(
                int(parameter.numel())
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ) == EXPECTED_TRAINABLE_PARAMETERS
        except Exception:
            return False

    @contextmanager
    def route(self, route: PairV5ActionRoute) -> Iterator[None]:
        if not isinstance(route, PairV5ActionRoute) or _ACTIVE_ROUTE.get() is not None:
            raise LargeActionAdapterError("large adapter route is invalid or nested")
        token = _ACTIVE_ROUTE.set(route)
        try:
            if route.adapter_active:
                yield
            else:
                disable = getattr(self.model, "disable_adapter", None)
                if not callable(disable):
                    raise LargeActionAdapterError("PEFT disable_adapter context is unavailable")
                with disable():
                    yield
        finally:
            _ACTIVE_ROUTE.reset(token)

    def state_dict_for_save(self) -> Mapping[str, Any]:
        return {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in self.trainable_named_parameters()
        }

    def receipt(self) -> Mapping[str, Any]:
        named = self.trainable_named_parameters()
        rows = [
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "numel": int(parameter.numel()),
            }
            for name, parameter in named
        ]
        value = {
            "schema_version": SCHEMA_VERSION,
            "blocks": TOTAL_BLOCKS_1P3B,
            "attention_modules": ["attn1", "attn2"],
            "projections": ["to_q", "to_k", "to_v", "to_out.0"],
            "selected_affines": len(self.specs),
            "rank": ACTION_LORA_RANK,
            "alpha": ACTION_LORA_ALPHA,
            "trainable_parameters": sum(row["numel"] for row in rows),
            "trainable_inventory_sha256": object_sha256(rows),
            "reference_policy": "same_native_model_peft_disabled",
            "target_row_selector": False,
            "tiny_internal_representation": False,
            "band_or_32d_objective": False,
            "architecture": dict(self.architecture),
        }
        return {**value, "digest": object_sha256(value)}


def install_pair_v5_action_adapter(renderer: Any) -> LargeActionAdapterHandle:
    """Install exact all-attention rank-256 PEFT and return old-trainer API."""

    from peft import LoraConfig, get_peft_model

    renderer.requires_grad_(False)
    specs = large_core.select_projection_specs(renderer, "all-attention")
    model = get_peft_model(
        renderer,
        LoraConfig(
            r=ACTION_LORA_RANK,
            lora_alpha=int(ACTION_LORA_ALPHA),
            lora_dropout=ACTION_LORA_DROPOUT,
            bias="none",
            target_modules=[item.name for item in specs],
        ),
    )
    transformer = model.get_base_model().diff_dec.transformer
    if transformer is None:
        raise LargeActionAdapterError("PEFT base transformer is absent")
    architecture = large_core.architecture_receipt("all-attention", specs)
    installation = large_core.validate_lora_installation(model, specs)
    handle = LargeActionAdapterHandle(
        model=model,
        transformer=transformer,
        specs=tuple(specs),
        architecture={**dict(architecture), "installation": dict(installation)},
    )
    if not handle.base_parameters_frozen():
        raise LargeActionAdapterError("large adapter base freeze/count closure differs")
    return handle


__all__ = [
    "ACTION_BLOCK_INDICES",
    "ACTION_LORA_ALPHA",
    "ACTION_LORA_DROPOUT",
    "ACTION_LORA_RANK",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "HIGH_SIGMA_INDICES",
    "LOW_SIGMA_INDICES",
    "MID_SIGMA_INDICES",
    "LargeActionAdapterError",
    "LargeActionAdapterHandle",
    "PairV5ActionRoute",
    "SCHEMA_VERSION",
    "install_pair_v5_action_adapter",
    "sigma_gate",
]

#!/usr/bin/env python3
"""Fail-closed schedule × block policy for Bernini causal localization.

This module deliberately separates denoising time from transformer depth.
It does not claim that action belongs to a high-noise branch or that any block
band is a learned motion module.  The default policy is diagnostic-only and
contains no optimizer authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, NoReturn, Sequence

try:
    from . import inference_sigma_strata as exact40
except ImportError:
    import inference_sigma_strata as exact40


SCHEMA_VERSION = "bernini-schedule-block-causal-policy-v1"
NUM_SCHEDULE_STEPS = 40
NUM_TRANSFORMER_BLOCKS = 30

# Existing measurements include s16/s35, plus action-conditioned trajectory
# proxies at s29/s38 (not decoded action evidence).  These four strata are a
# diagnostic registry, not a trainable schedule gate.
REGISTERED_SCHEDULE_INDICES = (16, 29, 35, 38)
REGISTERED_BLOCK_BANDS = (
    ("early", tuple(range(0, 8))),
    ("early_middle", tuple(range(8, 16))),
    ("late_middle", tuple(range(16, 23))),
    ("late", tuple(range(23, 30))),
)
REGISTERED_BRANCHES = (
    "noop",
    "forward",
    "reverse",
    "incomplete",
    "camera_only",
    "appearance_only",
    "wrong_owner",
)
DEFAULT_SCHEDULE_INDICES = REGISTERED_SCHEDULE_INDICES
DEFAULT_BLOCK_BANDS = dict(REGISTERED_BLOCK_BANDS)
DEFAULT_BRANCHES = REGISTERED_BRANCHES


class ScheduleBlockPolicyError(RuntimeError):
    """Raised before an ambiguous causal policy can be used."""


def fail(message: str) -> NoReturn:
    raise ScheduleBlockPolicyError(message)


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ScheduleBlockPolicyError("policy is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


_REGISTERED_GRID = {
    "schema_version": SCHEMA_VERSION,
    "schedule_indices": list(REGISTERED_SCHEDULE_INDICES),
    "block_bands": {
        name: list(indices) for name, indices in REGISTERED_BLOCK_BANDS
    },
    "branches": list(REGISTERED_BRANCHES),
    "exact40_schedule_sha256": exact40.SCHEDULE_SHA256,
}
REGISTERED_GRID_SHA256 = "992dc6e59399216f7556c8a0db7faa7e8bb98d81e6b6a37d8340284232267de8"
if object_sha256(_REGISTERED_GRID) != REGISTERED_GRID_SHA256:
    raise RuntimeError("registered Bernini schedule/block grid differs from its hash")


def _exact_indices(
    values: Sequence[int], *, upper: int, label: str
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        fail(f"{label} must be an integer sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or not 0 <= value < upper for value in result)
        or tuple(sorted(set(result))) != result
    ):
        fail(f"{label} must be a sorted unique non-empty in-range sequence")
    return result


@dataclass(frozen=True)
class ScheduleBlockCausalPolicy:
    schedule_indices: tuple[int, ...] = DEFAULT_SCHEDULE_INDICES
    block_bands: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
        DEFAULT_BLOCK_BANDS.items()
    )
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    optimizer_authorized: bool = False
    parameter_update_authorized: bool = False

    def __post_init__(self) -> None:
        schedules = _exact_indices(
            self.schedule_indices,
            upper=NUM_SCHEDULE_STEPS,
            label="schedule indices",
        )
        if not isinstance(self.block_bands, tuple) or not self.block_bands:
            fail("block bands must be a non-empty tuple")
        normalized_bands: list[tuple[str, tuple[int, ...]]] = []
        all_blocks: list[int] = []
        for item in self.block_bands:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or type(item[0]) is not str
                or not item[0]
            ):
                fail("block-band row differs")
            band = _exact_indices(
                item[1], upper=NUM_TRANSFORMER_BLOCKS, label=f"{item[0]} blocks"
            )
            normalized_bands.append((item[0], band))
            all_blocks.extend(band)
        if (
            len({name for name, _ in normalized_bands}) != len(normalized_bands)
            or sorted(all_blocks) != list(range(NUM_TRANSFORMER_BLOCKS))
        ):
            fail("block bands must be a disjoint exact cover of blocks 0..29")
        if (
            schedules != REGISTERED_SCHEDULE_INDICES
            or tuple(normalized_bands) != REGISTERED_BLOCK_BANDS
            or not isinstance(self.branches, tuple)
            or self.branches != REGISTERED_BRANCHES
            or self.optimizer_authorized is not False
            or self.parameter_update_authorized is not False
        ):
            fail("v1 policy is diagnostic-only with the registered branch closure")
        object.__setattr__(self, "schedule_indices", schedules)
        object.__setattr__(self, "block_bands", tuple(normalized_bands))

    @property
    def cell_count(self) -> int:
        return len(self.schedule_indices) * len(self.block_bands)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "renderer": "Bernini-R-1.3B-single-transformer_1",
            "num_schedule_steps": NUM_SCHEDULE_STEPS,
            "num_transformer_blocks": NUM_TRANSFORMER_BLOCKS,
            "schedule_indices": list(self.schedule_indices),
            "registered_grid_sha256": REGISTERED_GRID_SHA256,
            "exact40_schedule_schema": exact40.SCHEDULE_SCHEMA,
            "exact40_schedule_sha256": exact40.SCHEDULE_SHA256,
            "schedule_cells": [
                {
                    "schedule_index": index,
                    "timestep_int64": exact40.PINNED_TIMESTEPS[index],
                    "sigma_float32_be_hex": exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index],
                }
                for index in self.schedule_indices
            ],
            "block_bands": {
                name: list(indices) for name, indices in self.block_bands
            },
            "branches": list(self.branches),
            "cell_count": self.cell_count,
            "video_time_is_not_diffusion_time": True,
            "diffusion_time_is_not_block_depth": True,
            "same_transformer_executes_every_schedule_step": True,
            "high_noise_motion_only_claimed": False,
            "low_noise_texture_only_claimed": False,
            "low_sigma_action_gate_forced_zero": False,
            "block_motion_specialization_claimed": False,
            "decoded_intervention_required": True,
            "runtime_integration_verified": False,
            "decoded_intervention_executed": False,
            "matched_query_provenance_bound": False,
            "correct_vs_wrong_owner_required": True,
            "optimizer_authorized": self.optimizer_authorized,
            "parameter_update_authorized": self.parameter_update_authorized,
            "method_success_claimed": False,
        }
        return {**value, "receipt_digest": object_sha256(value)}


def default_policy() -> ScheduleBlockCausalPolicy:
    return ScheduleBlockCausalPolicy()


__all__ = [
    "DEFAULT_BLOCK_BANDS",
    "DEFAULT_BRANCHES",
    "DEFAULT_SCHEDULE_INDICES",
    "NUM_SCHEDULE_STEPS",
    "NUM_TRANSFORMER_BLOCKS",
    "REGISTERED_BLOCK_BANDS",
    "REGISTERED_BRANCHES",
    "REGISTERED_GRID_SHA256",
    "REGISTERED_SCHEDULE_INDICES",
    "SCHEMA_VERSION",
    "ScheduleBlockCausalPolicy",
    "ScheduleBlockPolicyError",
    "default_policy",
    "object_sha256",
]

#!/usr/bin/env python3
"""Observer-only native capture scaffold for relational action graphs.

This file owns the *runtime boundary*, not the relational representation.  It
seals four prompt arms (action/noop/reverse/static) onto one original Bernini
``shared_step`` and requires the exact same noisy-latent, timestep, rotary and
all other non-text objects for the four calls.  A read-only hook bank may then
collect detached post-RoPE Q/K and a clearly labelled
``derived_qk_role_responsibility_proxy`` at blocks ``{6,12,18,24}`` for one
pre-registered high/mid/mid-low sigma cell.  Bernini's fused attention ABI
does not expose backend attention weights, so the proxy must never be reported
as an observed official softmax tensor.

The only consumer of those raw tensors is
``self_generated_relational_action_graph_observer_v1``.  Raw captures are
zeroized immediately after that pure-tensor reduction.  This runner has no
decoder, optimizer, renderer, adapter, injection, target-teacher or candidate
selection API.  Its production contract remains ``gpu_launch_authorized =
False`` and ``scientific_claim_authorized = False`` until a real Bernini
WORLD4 capture closes the native hook and P0 replay gates.

``--dry-run`` is a CPU/toy integration check.  It is not model evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch
from torch import nn

import infer_native_self_generated_intermediate_anchor_canary_v1 as paired
import self_generated_intermediate_action_anchor_v1 as anchor_core


METHOD = "bernini-native-self-generated-relational-graph-observer-v1"
SCHEMA_VERSION = "bernini-native-relational-graph-observer-receipt-v1"
CAPTURE_SCHEMA = "bernini-native-relational-middle-capture-v1"
GPU_CONTRACT_SCHEMA = "bernini-native-relational-observer-gpu-contract-v1"

APPEARANCE_IDS = ("appearance_0", "appearance_1", "appearance_2")
ARMS = ("action", "noop", "reverse", "static")
BLOCKS = (6, 12, 18, 24)
SIGMA_BAND_ORDER = ("high", "mid", "mid_low")
SIGMA_INTERVALS: Mapping[str, tuple[float, float]] = {
    # Intervals are (lower exclusive, upper inclusive).  The real scheduler
    # must materialize and seal one exact step/sigma inside every interval.
    "high": (0.70, 1.00),
    "mid": (0.35, 0.70),
    "mid_low": (0.10, 0.35),
}
PHASES = 21
MAX_ROLES = 8
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARM_AUTHORITY_SEAL = object()
_ROLE_MATCHED_T2V_AUTHORITY_SEAL = object()

PUBLISHED_FIELDS = (
    "derived_qk_role_responsibility_proxy_centroid_covariance",
    "signed_role_pair_relation_delta",
    "normalized_relative_kinematics",
    "edge_change_points",
    "multi_block_multi_sigma_confidence",
)
FORBIDDEN_PERSISTENT_FIELDS = (
    "query",
    "key",
    "value",
    "hidden",
    "absolute_coordinate",
    "dense_mask",
    "rgb",
    "latent",
    "absolute_velocity",
    "camera_velocity",
    "camera_trajectory",
    "appearance_descriptor",
    "target_teacher",
)


class NativeRelationalObserverError(RuntimeError):
    """Fail-closed native capture, provenance or observer error."""


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NativeRelationalObserverError(f"{label} must be lowercase SHA-256")
    return value


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise NativeRelationalObserverError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeRelationalObserverError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise NativeRelationalObserverError(f"{label} must be finite")
    return result


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise NativeRelationalObserverError(f"{label} is outside its integer domain")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return anchor_core.object_sha256(value)


def _text_digest(value: str, *, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value or "\x00" in value:
        raise NativeRelationalObserverError(f"{label} must be nonempty stripped text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SigmaCell:
    """One exact40 scheduler cell registered inside a semantic sigma band."""

    band: str
    step_index: int
    sigma: float

    def __post_init__(self) -> None:
        if self.band not in SIGMA_BAND_ORDER:
            raise NativeRelationalObserverError("sigma band differs")
        step = _exact_int(self.step_index, label="step index")
        if step >= 40:
            raise NativeRelationalObserverError("step index lies outside exact40")
        sigma = _finite(self.sigma, label="sigma")
        lower, upper = SIGMA_INTERVALS[self.band]
        if not lower < sigma <= upper:
            raise NativeRelationalObserverError(
                f"sigma {sigma} does not lie in registered {self.band} interval"
            )

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "band": self.band,
            "step_index": self.step_index,
            "sigma": float(self.sigma),
            "interval_lower_exclusive": SIGMA_INTERVALS[self.band][0],
            "interval_upper_inclusive": SIGMA_INTERVALS[self.band][1],
        }
        return {**value, "digest": _canonical_digest(value)}


@dataclass(frozen=True)
class CapturePlan:
    """Closed 3-appearance x 4-arm x 3-sigma x 4-block matrix."""

    sigma_cells: tuple[SigmaCell, ...]
    appearance_ids: tuple[str, ...] = APPEARANCE_IDS
    arms: tuple[str, ...] = ARMS
    blocks: tuple[int, ...] = BLOCKS

    def __post_init__(self) -> None:
        if self.appearance_ids != APPEARANCE_IDS:
            raise NativeRelationalObserverError("appearance registry must be exact3")
        if self.arms != ARMS:
            raise NativeRelationalObserverError("arm registry must be exact action/noop/reverse/static")
        if self.blocks != BLOCKS:
            raise NativeRelationalObserverError("block registry must be exact 6/12/18/24")
        if tuple(cell.band for cell in self.sigma_cells) != SIGMA_BAND_ORDER:
            raise NativeRelationalObserverError("one ordered cell per sigma band is required")
        if len({cell.step_index for cell in self.sigma_cells}) != len(self.sigma_cells):
            raise NativeRelationalObserverError("sigma cells must use distinct exact40 steps")

    @property
    def forward_count(self) -> int:
        return len(self.appearance_ids) * len(self.arms) * len(self.sigma_cells)

    @property
    def block_capture_count(self) -> int:
        return self.forward_count * len(self.blocks)

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "appearance_ids": list(self.appearance_ids),
            "arms": list(self.arms),
            "blocks": list(self.blocks),
            "sigma_cells": [dict(cell.receipt()) for cell in self.sigma_cells],
            "native_forward_count": self.forward_count,
            "block_capture_count": self.block_capture_count,
        }
        return {**value, "digest": _canonical_digest(value)}


def toy_capture_plan() -> CapturePlan:
    return CapturePlan(
        (
            SigmaCell("high", 5, 0.85),
            SigmaCell("mid", 18, 0.55),
            SigmaCell("mid_low", 29, 0.20),
        )
    )


def _bound_arguments(
    shared_step: Callable[..., Any], args: Sequence[Any], kwargs: Mapping[str, Any]
) -> inspect.BoundArguments:
    try:
        result = inspect.signature(shared_step).bind(*tuple(args), **dict(kwargs))
        result.apply_defaults()
    except (TypeError, ValueError) as error:
        raise NativeRelationalObserverError("shared_step call does not bind") from error
    return result


@dataclass(frozen=True)
class FourArmForwardAuthority:
    """Four calls derived from one authenticated action call.

    The canonical action/noop pair is delegated to the already-audited
    :class:`PairedStateForwardAuthority`.  Reverse/static are then derived from
    that same bound action call and may replace only the two text fields.
    """

    appearance_id: str
    sigma_cell: SigmaCell
    action_noop: paired.PairedStateForwardAuthority
    reverse_args: tuple[Any, ...]
    reverse_kwargs: Mapping[str, Any]
    static_args: tuple[Any, ...]
    static_kwargs: Mapping[str, Any]
    prompt_embedding_sha256: Mapping[str, str]
    instruction_sha256: Mapping[str, str]
    state_tensor_sha256: Mapping[str, str]
    _seal: Any = None

    def validate(self) -> None:
        if self._seal is not _ARM_AUTHORITY_SEAL:
            raise NativeRelationalObserverError("four-arm authority is not authentic")
        if self.appearance_id not in APPEARANCE_IDS:
            raise NativeRelationalObserverError("four-arm appearance differs")
        if not isinstance(self.sigma_cell, SigmaCell):
            raise NativeRelationalObserverError("four-arm sigma cell differs")
        self.action_noop.validate()
        shared_step = self.action_noop.shared_step
        action = _bound_arguments(
            shared_step, self.action_noop.action_args, self.action_noop.action_kwargs
        )
        noop = _bound_arguments(
            shared_step, self.action_noop.noop_args, self.action_noop.noop_kwargs
        )
        reverse = _bound_arguments(shared_step, self.reverse_args, self.reverse_kwargs)
        static = _bound_arguments(shared_step, self.static_args, self.static_kwargs)
        required = {
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        }
        for bound in (action, noop, reverse, static):
            if not required.issubset(bound.arguments):
                raise NativeRelationalObserverError("four-arm shared_step ABI differs")
            if bound.arguments["model_id"] != "transformer_1":
                raise NativeRelationalObserverError("four-arm model route differs")
        for label, bound in (("noop", noop), ("reverse", reverse), ("static", static)):
            for name, action_value in action.arguments.items():
                if name in {"cond_embeds", "batch_text_seqlen"}:
                    continue
                if bound.arguments.get(name) is not action_value:
                    raise NativeRelationalObserverError(
                        f"{label} differs outside sealed text fields: {name}"
                    )
        if set(self.state_tensor_sha256) != {
            "noisy_latents",
            "timesteps",
            "rotary_embs",
        }:
            raise NativeRelationalObserverError("same-state tensor digest registry differs")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            value = action.arguments[name]
            if not isinstance(value, torch.Tensor):
                raise NativeRelationalObserverError(f"{name} must be a tensor")
            if anchor_core.tensor_sha256(value) != _sha256(
                self.state_tensor_sha256[name], label=f"{name} state tensor"
            ):
                raise NativeRelationalObserverError(f"sealed {name} bytes changed")
        prompts = {
            "action": action.arguments["cond_embeds"],
            "noop": noop.arguments["cond_embeds"],
            "reverse": reverse.arguments["cond_embeds"],
            "static": static.arguments["cond_embeds"],
        }
        if len({id(value) for value in prompts.values()}) != len(ARMS):
            raise NativeRelationalObserverError("four prompt embeddings must be distinct objects")
        if set(self.prompt_embedding_sha256) != set(ARMS) or set(
            self.instruction_sha256
        ) != set(ARMS):
            raise NativeRelationalObserverError("four-arm prompt digest registry differs")
        for arm, prompt in prompts.items():
            if (
                not isinstance(prompt, torch.Tensor)
                or prompt.ndim != 3
                or int(prompt.shape[0]) != 1
                or prompt.requires_grad
                or prompt.grad_fn is not None
                or not bool(torch.isfinite(prompt).all().item())
            ):
                raise NativeRelationalObserverError(f"{arm} prompt tensor differs")
            if anchor_core.tensor_sha256(prompt) != _sha256(
                self.prompt_embedding_sha256[arm], label=f"{arm} prompt embedding"
            ):
                raise NativeRelationalObserverError(f"{arm} prompt embedding bytes changed")
            _sha256(self.instruction_sha256[arm], label=f"{arm} instruction")
            lengths = bound_for_arm(
                arm, action=action, noop=noop, reverse=reverse, static=static
            ).arguments["batch_text_seqlen"]
            try:
                actual_lengths = tuple(int(item) for item in lengths)
            except Exception as error:
                raise NativeRelationalObserverError(f"{arm} text length differs") from error
            if actual_lengths != (int(prompt.shape[1]),):
                raise NativeRelationalObserverError(f"{arm} text length differs")

    def call(self, arm: str) -> Any:
        self.validate()
        if arm == "action":
            return self.action_noop.action_call()
        if arm == "noop":
            return self.action_noop.noop_call()
        if arm == "reverse":
            return self.action_noop.shared_step(*self.reverse_args, **dict(self.reverse_kwargs))
        if arm == "static":
            return self.action_noop.shared_step(*self.static_args, **dict(self.static_kwargs))
        raise NativeRelationalObserverError("requested arm differs")

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        value = {
            "appearance_id": self.appearance_id,
            "sigma_cell": dict(self.sigma_cell.receipt()),
            "paired_action_noop": dict(self.action_noop.receipt()),
            "four_arms": list(ARMS),
            "same_original_shared_step": True,
            "same_noisy_timestep_rotary_and_nontext_objects": True,
            "only_replaced_fields": ["cond_embeds", "batch_text_seqlen"],
            "prompt_embedding_sha256": dict(self.prompt_embedding_sha256),
            "instruction_sha256": dict(self.instruction_sha256),
            "state_tensor_sha256": dict(self.state_tensor_sha256),
        }
        return {**value, "digest": _canonical_digest(value)}


@dataclass(frozen=True)
class RoleMatchedT2VFourArmForwardAuthority:
    """Four pure-T2V controls sharing one self-generated trajectory state.

    Unlike :class:`FourArmForwardAuthority`, this authority does not reuse the
    edit-specific sentence "keep the source video unchanged".  A pure T2V
    anchor has no source video.  Instead, every arm must retain the same exact
    semantic role phrases, while action/noop/reverse/static differ only in the
    registered interaction.  This closes the otherwise hidden mismatch between
    an editing no-op and a role-matched T2V counterfactual.
    """

    appearance_id: str
    sigma_cell: SigmaCell
    shared_step: Callable[..., Any]
    arm_args: Mapping[str, tuple[Any, ...]]
    arm_kwargs: Mapping[str, Mapping[str, Any]]
    prompt_embedding_sha256: Mapping[str, str]
    instruction_sha256: Mapping[str, str]
    role_phrases: Mapping[str, str]
    state_tensor_sha256: Mapping[str, str]
    _seal: Any = None

    def validate(self) -> None:
        if self._seal is not _ROLE_MATCHED_T2V_AUTHORITY_SEAL:
            raise NativeRelationalObserverError("role-matched T2V authority is not authentic")
        if self.appearance_id not in APPEARANCE_IDS or not isinstance(
            self.sigma_cell, SigmaCell
        ):
            raise NativeRelationalObserverError("role-matched T2V cell differs")
        if set(self.arm_args) != set(ARMS) or set(self.arm_kwargs) != set(ARMS):
            raise NativeRelationalObserverError("role-matched T2V arm calls differ")
        if set(self.prompt_embedding_sha256) != set(ARMS) or set(
            self.instruction_sha256
        ) != set(ARMS):
            raise NativeRelationalObserverError("role-matched T2V digest registry differs")
        if (
            not isinstance(self.role_phrases, Mapping)
            or len(self.role_phrases) < 2
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(phrase, str)
                or not phrase
                for name, phrase in self.role_phrases.items()
            )
        ):
            raise NativeRelationalObserverError("role-matched T2V phrases differ")
        bounds = {
            arm: _bound_arguments(
                self.shared_step, self.arm_args[arm], self.arm_kwargs[arm]
            )
            for arm in ARMS
        }
        required = {
            "model_id",
            "noisy_latents",
            "timesteps",
            "cond_embeds",
            "rotary_embs",
            "batch_vae_seqlen",
            "batch_text_seqlen",
        }
        action = bounds["action"]
        variadic_keyword_names = {
            name
            for name, parameter in inspect.signature(self.shared_step).parameters.items()
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        }
        for arm, bound in bounds.items():
            if not required.issubset(bound.arguments):
                raise NativeRelationalObserverError("role-matched T2V shared_step ABI differs")
            if bound.arguments["model_id"] != "transformer_1":
                raise NativeRelationalObserverError("role-matched T2V model route differs")
            anchor_core.assert_target_isolation_payload(
                dict(bound.arguments), path=f"role_matched_t2v.{arm}"
            )
            for name in variadic_keyword_names:
                value = bound.arguments.get(name)
                if not isinstance(value, Mapping) or value:
                    raise NativeRelationalObserverError(
                        f"role-matched T2V variadic kwargs must be exactly empty: {name}"
                    )
            if arm != "action":
                for name, action_value in action.arguments.items():
                    if name in {"cond_embeds", "batch_text_seqlen"}:
                        continue
                    # inspect.apply_defaults() materializes a fresh empty dict
                    # for **kwargs on every bind.  Empty content is the actual
                    # official-call invariant; Python object identity is not.
                    if name in variadic_keyword_names:
                        continue
                    if bound.arguments.get(name) is not action_value:
                        raise NativeRelationalObserverError(
                            f"{arm} differs outside sealed text fields: {name}"
                        )
        prompts = {arm: bounds[arm].arguments["cond_embeds"] for arm in ARMS}
        if len({id(value) for value in prompts.values()}) != len(ARMS):
            raise NativeRelationalObserverError("role-matched prompt objects must be distinct")
        for arm, prompt in prompts.items():
            if (
                not isinstance(prompt, torch.Tensor)
                or prompt.ndim != 3
                or int(prompt.shape[0]) != 1
                or prompt.requires_grad
                or prompt.grad_fn is not None
                or not bool(torch.isfinite(prompt).all().item())
                or anchor_core.tensor_sha256(prompt)
                != _sha256(
                    self.prompt_embedding_sha256[arm],
                    label=f"role-matched {arm} prompt embedding",
                )
            ):
                raise NativeRelationalObserverError(
                    f"role-matched {arm} prompt tensor differs"
                )
            lengths = tuple(int(item) for item in bounds[arm].arguments["batch_text_seqlen"])
            if lengths != (int(prompt.shape[1]),):
                raise NativeRelationalObserverError(
                    f"role-matched {arm} text length differs"
                )
        if set(self.state_tensor_sha256) != {
            "noisy_latents",
            "timesteps",
            "rotary_embs",
        }:
            raise NativeRelationalObserverError("role-matched state registry differs")
        for name, digest in self.state_tensor_sha256.items():
            value = action.arguments[name]
            if (
                not isinstance(value, torch.Tensor)
                or anchor_core.tensor_sha256(value)
                != _sha256(digest, label=f"role-matched {name}")
            ):
                raise NativeRelationalObserverError(f"role-matched {name} bytes changed")

    def call(self, arm: str) -> Any:
        self.validate()
        if arm not in ARMS:
            raise NativeRelationalObserverError("requested role-matched arm differs")
        return self.shared_step(*self.arm_args[arm], **dict(self.arm_kwargs[arm]))

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        value = {
            "appearance_id": self.appearance_id,
            "sigma_cell": dict(self.sigma_cell.receipt()),
            "four_arms": list(ARMS),
            "authority_kind": "role_matched_pure_t2v_same_trajectory_state_v1",
            "same_noisy_timestep_rotary_and_nontext_objects": True,
            "only_replaced_fields": ["cond_embeds", "batch_text_seqlen"],
            "generic_source_video_noop_used": False,
            "all_role_phrases_retained_in_every_arm": True,
            "role_phrase_sha256": {
                name: hashlib.sha256(phrase.encode("utf-8")).hexdigest()
                for name, phrase in sorted(self.role_phrases.items())
            },
            "prompt_embedding_sha256": dict(self.prompt_embedding_sha256),
            "instruction_sha256": dict(self.instruction_sha256),
            "state_tensor_sha256": dict(self.state_tensor_sha256),
        }
        return {**value, "digest": _canonical_digest(value)}


def seal_role_matched_t2v_four_arm_forward(
    *,
    appearance_id: str,
    sigma_cell: SigmaCell,
    shared_step: Callable[..., Any],
    action_args: Sequence[Any] = (),
    action_kwargs: Optional[Mapping[str, Any]] = None,
    prompt_embeds: Mapping[str, torch.Tensor],
    instructions: Mapping[str, str],
    role_phrases: Mapping[str, str],
) -> RoleMatchedT2VFourArmForwardAuthority:
    """Seal role-matched T2V controls without an edit/source no-op sentence."""

    if appearance_id not in APPEARANCE_IDS or not isinstance(sigma_cell, SigmaCell):
        raise NativeRelationalObserverError("role-matched T2V registry differs")
    if set(prompt_embeds) != set(ARMS) or set(instructions) != set(ARMS):
        raise NativeRelationalObserverError("role-matched exact four arms are required")
    if not isinstance(role_phrases, Mapping) or len(role_phrases) < 2:
        raise NativeRelationalObserverError("role-matched semantic phrases are absent")
    for arm, instruction in instructions.items():
        _text_digest(instruction, label=f"{arm} instruction")
        for role, phrase in role_phrases.items():
            if (
                not isinstance(phrase, str)
                or not phrase
                or phrase.casefold() not in instruction.casefold()
            ):
                raise NativeRelationalObserverError(
                    f"{arm} instruction lacks role phrase {role}"
                )
    if len(set(instructions.values())) != len(ARMS):
        raise NativeRelationalObserverError("role-matched instructions must be distinct")
    kwargs = dict(action_kwargs or {})
    anchor_core.assert_target_isolation_payload(kwargs, path="role_matched_t2v_action")
    action = _bound_arguments(shared_step, action_args, kwargs)
    if action.arguments.get("cond_embeds") is not prompt_embeds["action"]:
        raise NativeRelationalObserverError("role-matched action prompt ownership differs")
    signature = inspect.signature(shared_step)
    arm_args: dict[str, tuple[Any, ...]] = {}
    arm_kwargs: dict[str, Mapping[str, Any]] = {}
    for arm in ARMS:
        bound = signature.bind(*tuple(action_args), **kwargs)
        bound.apply_defaults()
        bound.arguments["cond_embeds"] = prompt_embeds[arm]
        bound.arguments["batch_text_seqlen"] = [int(prompt_embeds[arm].shape[1])]
        arm_args[arm] = tuple(bound.args)
        arm_kwargs[arm] = dict(bound.kwargs)
    authority = RoleMatchedT2VFourArmForwardAuthority(
        appearance_id=appearance_id,
        sigma_cell=sigma_cell,
        shared_step=shared_step,
        arm_args=arm_args,
        arm_kwargs=arm_kwargs,
        prompt_embedding_sha256={
            arm: anchor_core.tensor_sha256(prompt_embeds[arm]) for arm in ARMS
        },
        instruction_sha256={
            arm: _text_digest(instructions[arm], label=f"{arm} instruction")
            for arm in ARMS
        },
        role_phrases=dict(role_phrases),
        state_tensor_sha256={
            name: anchor_core.tensor_sha256(action.arguments[name])
            for name in ("noisy_latents", "timesteps", "rotary_embs")
        },
        _seal=_ROLE_MATCHED_T2V_AUTHORITY_SEAL,
    )
    authority.validate()
    return authority


def bound_for_arm(
    arm: str,
    *,
    action: inspect.BoundArguments,
    noop: inspect.BoundArguments,
    reverse: inspect.BoundArguments,
    static: inspect.BoundArguments,
) -> inspect.BoundArguments:
    return {"action": action, "noop": noop, "reverse": reverse, "static": static}[arm]


def seal_four_arm_forward(
    *,
    appearance_id: str,
    sigma_cell: SigmaCell,
    shared_step: Callable[..., Any],
    action_args: Sequence[Any] = (),
    action_kwargs: Optional[Mapping[str, Any]] = None,
    prompt_embeds: Mapping[str, torch.Tensor],
    instruction_sha256: Mapping[str, str],
) -> FourArmForwardAuthority:
    """Derive noop/reverse/static calls from one caller-supplied action call."""

    if appearance_id not in APPEARANCE_IDS or not isinstance(sigma_cell, SigmaCell):
        raise NativeRelationalObserverError("four-arm registry differs")
    if set(prompt_embeds) != set(ARMS) or set(instruction_sha256) != set(ARMS):
        raise NativeRelationalObserverError("exact four prompt arms are required")
    if instruction_sha256["noop"] != anchor_core.CANONICAL_NOOP_SHA256:
        raise NativeRelationalObserverError("noop instruction is not canonical")
    kwargs = dict(action_kwargs or {})
    anchor_core.assert_target_isolation_payload(kwargs, path="four_arm_action_kwargs")
    action_bound = _bound_arguments(shared_step, action_args, kwargs)
    if action_bound.arguments.get("cond_embeds") is not prompt_embeds["action"]:
        raise NativeRelationalObserverError("action call does not own action prompt object")
    action_noop = paired.seal_paired_state_forward(
        shared_step=shared_step,
        action_args=action_args,
        action_kwargs=kwargs,
        canonical_noop_embeds=prompt_embeds["noop"],
        canonical_noop_instruction=anchor_core.CANONICAL_NOOP_INSTRUCTION,
        canonical_noop_instruction_sha256=anchor_core.CANONICAL_NOOP_SHA256,
    )
    derived: dict[str, inspect.BoundArguments] = {}
    signature = inspect.signature(shared_step)
    for arm in ("reverse", "static"):
        bound = signature.bind(*tuple(action_args), **kwargs)
        bound.apply_defaults()
        bound.arguments["cond_embeds"] = prompt_embeds[arm]
        bound.arguments["batch_text_seqlen"] = [int(prompt_embeds[arm].shape[1])]
        derived[arm] = bound
    authority = FourArmForwardAuthority(
        appearance_id=appearance_id,
        sigma_cell=sigma_cell,
        action_noop=action_noop,
        reverse_args=tuple(derived["reverse"].args),
        reverse_kwargs=dict(derived["reverse"].kwargs),
        static_args=tuple(derived["static"].args),
        static_kwargs=dict(derived["static"].kwargs),
        prompt_embedding_sha256={
            arm: anchor_core.tensor_sha256(prompt_embeds[arm]) for arm in ARMS
        },
        instruction_sha256={arm: _sha256(instruction_sha256[arm], label=arm) for arm in ARMS},
        state_tensor_sha256={
            name: anchor_core.tensor_sha256(action_bound.arguments[name])
            for name in ("noisy_latents", "timesteps", "rotary_embs")
        },
        _seal=_ARM_AUTHORITY_SEAL,
    )
    authority.validate()
    return authority


@dataclass(frozen=True)
class CaptureInvocation:
    appearance_id: str
    arm: str
    sigma_cell: SigmaCell
    noisy_state_sha256: str
    timestep_sha256: str
    rotary_sha256: str
    patch_height: int
    patch_width: int
    blocks: tuple[int, ...] = BLOCKS

    def __post_init__(self) -> None:
        if self.appearance_id not in APPEARANCE_IDS or self.arm not in ARMS:
            raise NativeRelationalObserverError("capture invocation registry differs")
        if not isinstance(self.sigma_cell, SigmaCell) or self.blocks != BLOCKS:
            raise NativeRelationalObserverError("capture invocation cell/blocks differ")
        _sha256(self.noisy_state_sha256, label="capture noisy state")
        _sha256(self.timestep_sha256, label="capture timestep")
        _sha256(self.rotary_sha256, label="capture rotary")
        height = _exact_int(self.patch_height, label="patch height", minimum=1)
        width = _exact_int(self.patch_width, label="patch width", minimum=1)
        if height * width < 2:
            raise NativeRelationalObserverError("capture patch grid is degenerate")

    @property
    def state_sha256(self) -> str:
        return _canonical_digest(
            {
                "noisy_state_sha256": self.noisy_state_sha256,
                "timestep_sha256": self.timestep_sha256,
                "rotary_sha256": self.rotary_sha256,
            }
        )

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.appearance_id, self.arm, self.sigma_cell.band, self.sigma_cell.step_index)


@dataclass(frozen=True)
class NativeBlockCapture:
    """Ephemeral, anchor-local raw tensors for one block/call.

    Shapes are normalized before this boundary so the pure observer does not
    know Ulysses rank layout: Q/K ``[1,21,P,H,D]`` and the explicitly derived
    QK role-responsibility proxy ``[1,21,K,P]``.  No spatial coordinates or
    masks may accompany them, and no backend attention weights are claimed.
    """

    schema_version: str
    invocation: CaptureInvocation
    block_index: int
    query: torch.Tensor
    key: torch.Tensor
    derived_qk_role_responsibility_proxy: torch.Tensor

    def __post_init__(self) -> None:
        if self.schema_version != CAPTURE_SCHEMA:
            raise NativeRelationalObserverError("capture schema differs")
        if self.block_index not in BLOCKS:
            raise NativeRelationalObserverError("capture block differs")
        query = self.query
        key = self.key
        responsibility = self.derived_qk_role_responsibility_proxy
        if (
            not isinstance(query, torch.Tensor)
            or not isinstance(key, torch.Tensor)
            or query.ndim != 5
            or tuple(query.shape) != tuple(key.shape)
            or tuple(query.shape[:2]) != (1, PHASES)
            or int(query.shape[2]) < 2
            or int(query.shape[3]) < 1
            or int(query.shape[4]) < 1
            or query.dtype != key.dtype
            or query.device != key.device
        ):
            raise NativeRelationalObserverError("post-RoPE Q/K geometry differs")
        if (
            not isinstance(responsibility, torch.Tensor)
            or responsibility.ndim != 4
            or tuple(responsibility.shape[:2]) != (1, PHASES)
            or not 2 <= int(responsibility.shape[2]) <= MAX_ROLES
            or int(responsibility.shape[3]) != int(query.shape[2])
            or int(responsibility.shape[3])
            != self.invocation.patch_height * self.invocation.patch_width
            or responsibility.device != query.device
        ):
            raise NativeRelationalObserverError("derived QK role-proxy geometry differs")
        for label, tensor in (("query", query), ("key", key), ("role", responsibility)):
            if (
                tensor.requires_grad
                or tensor.grad_fn is not None
                or not tensor.is_contiguous()
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise NativeRelationalObserverError(f"{label} capture is not detached finite contiguous")
        if bool((responsibility < 0).any().item()):
            raise NativeRelationalObserverError("derived QK role proxy must be nonnegative")
        mass = responsibility.float().sum(dim=2)
        if not bool(torch.allclose(mass, torch.ones_like(mass), atol=2.0e-4, rtol=2.0e-4)):
            raise NativeRelationalObserverError("derived QK role proxy must sum to one per token")

    def zeroize(self) -> None:
        with torch.no_grad():
            self.query.zero_()
            self.key.zero_()
            self.derived_qk_role_responsibility_proxy.zero_()


_ACTIVE_CAPTURE: ContextVar[Optional[CaptureInvocation]] = ContextVar(
    "bernini_native_relational_capture", default=None
)


class InMemoryNativeCaptureBank:
    """In-memory hook sink with no tensor serialization method."""

    def __init__(self) -> None:
        self._captures: dict[tuple[str, str, str, int], dict[int, NativeBlockCapture]] = {}
        self.capture_count = 0
        self.consumed_count = 0
        self.zeroized_count = 0

    @contextmanager
    def observe(self, invocation: CaptureInvocation) -> Iterator[None]:
        if not isinstance(invocation, CaptureInvocation):
            raise NativeRelationalObserverError("capture context differs")
        if _ACTIVE_CAPTURE.get() is not None:
            raise NativeRelationalObserverError("nested capture contexts are forbidden")
        if invocation.key in self._captures:
            raise NativeRelationalObserverError("duplicate capture invocation")
        self._captures[invocation.key] = {}
        token: Token[Optional[CaptureInvocation]] = _ACTIVE_CAPTURE.set(invocation)
        try:
            yield
        finally:
            _ACTIVE_CAPTURE.reset(token)

    @staticmethod
    def current_invocation() -> Optional[CaptureInvocation]:
        return _ACTIVE_CAPTURE.get()

    def capture(self, value: NativeBlockCapture) -> None:
        invocation = self.current_invocation()
        if invocation is None or not isinstance(value, NativeBlockCapture):
            raise NativeRelationalObserverError("capture arrived outside its observer context")
        if value.invocation != invocation:
            raise NativeRelationalObserverError("capture invocation ownership differs")
        rows = self._captures.get(invocation.key)
        if rows is None or value.block_index in rows:
            raise NativeRelationalObserverError("duplicate/absent capture cell")
        rows[value.block_index] = value
        self.capture_count += 1

    def consume(self, invocation: CaptureInvocation) -> tuple[NativeBlockCapture, ...]:
        rows = self._captures.pop(invocation.key, None)
        succeeded = False
        try:
            if rows is None or tuple(sorted(rows)) != BLOCKS:
                raise NativeRelationalObserverError(
                    "capture did not close exact four blocks"
                )
            result = tuple(rows[index] for index in BLOCKS)
            self.consumed_count += len(result)
            succeeded = True
            return result
        finally:
            if not succeeded and isinstance(rows, dict):
                self.zeroize(tuple(rows.values()))

    def zeroize(self, captures: Sequence[NativeBlockCapture]) -> None:
        for value in captures:
            value.zeroize()
            self.zeroized_count += 1

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": CAPTURE_SCHEMA,
            "capture_count": self.capture_count,
            "consumed_count": self.consumed_count,
            "zeroized_count": self.zeroized_count,
            "resident_invocation_count": len(self._captures),
            "persistent_tensor_artifact_created": False,
            "captured_fields": [
                "post_rope_query",
                "post_rope_key",
                "derived_qk_role_responsibility_proxy",
            ],
            "backend_attention_weights_observed": False,
            "forbidden_fields": list(FORBIDDEN_PERSISTENT_FIELDS),
        }
        return {**value, "digest": _canonical_digest(value)}


def native_gpu_launch_contract(plan: Optional[CapturePlan] = None) -> Mapping[str, Any]:
    """Fail-closed WORLD4 contract; printing it never authorizes a launch."""

    selected = plan or toy_capture_plan()
    plan_is_toy = selected == toy_capture_plan()
    value = {
        "schema_version": GPU_CONTRACT_SCHEMA,
        "method": METHOD,
        "status": "cpu_fake_official_hook_closed_waiting_real_world4_capture",
        "model": "Bernini-R 1.3B transformer_1",
        "guidance_mode": "v2v_apg",
        "num_inference_steps": 40,
        "world_size": 4,
        "base_frozen": True,
        "model_eval": True,
        "optimizer": None,
        "trainable_parameter_count": 0,
        "parameter_updates": 0,
        "adapter_or_lora_loaded": False,
        "injection_or_route_strength": None,
        "decoder_available_to_runner": False,
        "renderer_available_to_runner": False,
        "candidate_output_modified": False,
        "target_teacher_available_to_runner": False,
        "target_inputs_consumed": False,
        "capture_plan": dict(selected.receipt()),
        "capture_plan_is_toy_placeholder": plan_is_toy,
        "real_scheduler_cell_authority_sealed": False,
        "same_state_authority": {
            "one_original_shared_step_per_appearance_sigma_cell": True,
            "identical_object_fields": ["noisy_latents", "timesteps", "rotary_embs"],
            "all_other_nontext_arguments_identical_by_object": True,
            "only_replaceable_fields": ["cond_embeds", "batch_text_seqlen"],
            "canonical_noop_bytes_sealed": True,
        },
        "native_hook_seams": {
            "implementation": "native_relational_attention_hook_v1",
            "action_noop_forward_authority": (
                "infer_native_self_generated_intermediate_anchor_canary_v1."
                "PairedStateForwardAuthority"
            ),
            "derived_qk_role_responsibility_proxy": (
                "same-call exact attn2 Q/K, float32 scaled-QK softmax, "
                "exhaustive text-token role partition"
            ),
            "backend_attention_weights_observed": False,
            "post_rope_qk": "official_project_qkv_read_only_observer",
            "official_attention_output_forwarded_same_object": True,
            "SP_collective_calls_added_inside_attention": 0,
            "WORLD4_assembly_location": "after_transformer_forward",
        },
        "cpu_fake_official_hook_validation": {
            "same_call_attn1_post_rope_qk": True,
            "same_call_attn2_projected_qk": True,
            "official_output_forwarded_same_object": True,
            "official_output_bit_exact": True,
            "WORLD4_head_and_sequence_assembly_fail_closed": True,
            "missing_rank_rejected": True,
            "backend_attention_weights_observed": False,
            "real_checkpoint_evidence": False,
        },
        "ephemeral_raw_capture": {
            "fields": [
                "post_rope_query",
                "post_rope_key",
                "derived_qk_role_responsibility_proxy",
            ],
            "backend_attention_weights_observed": False,
            "zeroize_after_pure_tensor_reduction": True,
            "serialization_authorized": False,
            "whole_144_block_raw_bundle_residency_authorized": False,
            "real_runtime_required_mode": (
                "stream_one_appearance_sigma_cell_reduce_four_arms_then_zeroize"
            ),
        },
        "published_field_allowlist": list(PUBLISHED_FIELDS),
        "forbidden_persistent_fields": list(FORBIDDEN_PERSISTENT_FIELDS),
        "p0_contract": {
            "hooks_not_installed": True,
            "before_after_output_bit_exact_required": True,
            "before_after_state_digest_equal_required": True,
        },
        "required_before_real_gpu_capture": [
            "real_exact40_scheduler_cells_sealed_from_checkpoint_runtime",
            "three_appearance_prompt_and_initial_noise_receipts_sealed",
            "native_attn1_post_rope_qk_hook_output_identity_tested",
            "native_attn2_qk_derived_role_proxy_provenance_tested",
            "WORLD4_global_tensor_assembly_tested",
            "observer_public_ABI_static_hash_sealed",
            "streaming_cell_reduction_ABI_closes_without_144_block_raw_residency",
            "P0_exact_replay_closes_on_real_checkpoint",
        ],
        "real_native_capture_completed": False,
        "gpu_launch_authorized": False,
        "scientific_claim_authorized": False,
        "stable_transferable_action_representation_claimed": False,
        "renderer_effectiveness_claimed": False,
    }
    return {**value, "digest": _canonical_digest(value)}


def _instruction_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _ToyBlock(nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((width,), index * 1.0e-4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias.reshape(1, 1, -1).to(hidden.dtype)


class _ToyTransformer(nn.Module):
    def __init__(self, bank: InMemoryNativeCaptureBank, *, patches: int = 6, width: int = 8) -> None:
        super().__init__()
        self.bank = bank
        self.patches = patches
        self.width = width
        self.blocks = nn.ModuleList([_ToyBlock(width, index) for index in range(30)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        invocation = self.bank.current_invocation()
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if invocation is not None and index in BLOCKS:
                phase = hidden.reshape(1, PHASES, self.patches, self.width)
                query = phase.reshape(1, PHASES, self.patches, 2, self.width // 2)
                key = torch.roll(query, shifts=1, dims=2)
                logits = torch.stack(
                    (phase[..., 0], phase[..., 1], phase[..., 2]), dim=2
                )
                responsibility = torch.softmax(logits, dim=2)
                self.bank.capture(
                    NativeBlockCapture(
                        CAPTURE_SCHEMA,
                        invocation,
                        index,
                        query.detach().contiguous(),
                        key.detach().contiguous(),
                        responsibility.detach().contiguous(),
                    )
                )
        return hidden


def _observer_module() -> Any:
    try:
        import self_generated_relational_action_graph_observer_v1 as observer
    except ImportError as error:
        raise NativeRelationalObserverError(
            "relational observer module is not available"
        ) from error
    return observer


def _toy_role_specs(observer: Any) -> tuple[Any, ...]:
    constructor = getattr(observer, "RoleSpec", None)
    if constructor is None:
        raise NativeRelationalObserverError("observer lacks RoleSpec")
    return (
        constructor(
            "human_agent",
            "source_owned",
            semantic_role="human_agent",
            evidence_mode="observed_internal",
            first_reliable_phase=0,
            source_node_id="toy_agent",
            critical=True,
        ),
        constructor(
            "moving_object",
            "source_owned",
            semantic_role="moving_object",
            evidence_mode="observed_internal",
            first_reliable_phase=0,
            source_node_id="toy_object",
            critical=True,
        ),
        constructor(
            "support_surface",
            "source_owned",
            semantic_role="support_surface",
            evidence_mode="observed_internal",
            first_reliable_phase=0,
            source_node_id="toy_support",
            critical=True,
        ),
    )


def _new_streaming_observer(observer: Any, roles: Sequence[Any]) -> Any:
    config_type = getattr(observer, "ObserverConfig", None)
    edge_type = getattr(observer, "EdgeSpec", None)
    stream_type = getattr(observer, "StreamingRelationalObserver", None)
    if config_type is None or edge_type is None or stream_type is None:
        raise NativeRelationalObserverError(
            "observer lacks EdgeSpec/ObserverConfig/StreamingRelationalObserver"
        )
    edge_specs = (
        edge_type(
            source_role="human_agent",
            target_role="moving_object",
            relation_type="relative_motion",
        ),
        edge_type(
            source_role="moving_object",
            target_role="support_surface",
            relation_type="approaching_or_receding",
        ),
    )
    return stream_type(
        roles=tuple(roles),
        config=config_type(edge_specs=edge_specs),
    )


def _stream_native_capture_group(
    *,
    observer: Any,
    stream: Any,
    captures: Sequence[NativeBlockCapture],
    roles: Sequence[Any],
    prompt_sha256: str,
) -> None:
    """Compress one four-block arm immediately; retain no raw Q/K/responsibility."""

    cell_type = getattr(observer, "CaptureCell", None)
    if cell_type is None or not callable(getattr(stream, "add", None)):
        raise NativeRelationalObserverError("observer streaming capture ABI differs")
    role_ids = tuple(str(getattr(item, "role_id")) for item in roles)
    for capture in captures:
        invocation = capture.invocation
        queries = capture.query[0].float().mean(dim=2).detach().contiguous()
        keys = capture.key[0].float().mean(dim=2).detach().contiguous()
        responsibilities = (
            capture.derived_qk_role_responsibility_proxy[0].float().detach().contiguous()
        )
        if int(responsibilities.shape[1]) != len(role_ids):
            raise NativeRelationalObserverError(
                "capture responsibility count differs from role registry"
            )
        cell = cell_type(
            appearance_id=invocation.appearance_id,
            arm=invocation.arm,
            sigma_band=invocation.sigma_cell.band,
            block_index=capture.block_index,
            state_sha256=invocation.state_sha256,
            prompt_sha256=_sha256(prompt_sha256, label="capture prompt"),
            patch_height=invocation.patch_height,
            patch_width=invocation.patch_width,
            roles=role_ids,
            queries=queries,
            keys=keys,
            responsibilities=responsibilities,
        )
        stream.add(cell, zeroize=True)
        # The streaming ABI must scrub its own temporary tensors before
        # returning.  The runner separately clears the upstream head tensors.
        for label, tensor in (
            ("queries", queries),
            ("keys", keys),
            ("responsibilities", responsibilities),
        ):
            if int(torch.count_nonzero(tensor).item()) != 0:
                raise NativeRelationalObserverError(
                    f"observer did not zeroize streamed {label}"
                )


def _finalize_streaming_observer(stream: Any) -> Mapping[str, Any]:
    if not callable(getattr(stream, "finalize", None)):
        raise NativeRelationalObserverError("observer stream lacks finalize()")
    result = stream.finalize()
    if hasattr(result, "receipt") and callable(result.receipt):
        result = result.receipt()
    if not isinstance(result, Mapping):
        raise NativeRelationalObserverError("observer finalize result is not JSON-safe mapping")
    try:
        json.dumps(result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise NativeRelationalObserverError("observer result is not finite JSON") from error
    if result.get("scientific_claim_authorized") is not False:
        raise NativeRelationalObserverError(
            "toy/native scaffold observer must keep scientific claim unauthorized"
        )
    if result.get("target_inputs_consumed", False) is not False:
        raise NativeRelationalObserverError("observer result reports target consumption")
    return dict(result)


def _dry_run() -> Mapping[str, Any]:
    """Run frozen toy capture -> real pure observer -> raw zeroization -> P0."""

    plan = toy_capture_plan()
    bank = InMemoryNativeCaptureBank()
    transformer = _ToyTransformer(bank).eval().requires_grad_(False)
    freeze_before = anchor_core.frozen_module_certificate(transformer)
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    neutral = torch.randn((1, PHASES * transformer.patches, transformer.width), generator=generator)
    p0a = transformer(neutral.clone()).detach()

    def shared_step(
        model_id: str,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_embeds: torch.Tensor,
        rotary_embs: torch.Tensor,
        batch_vae_seqlen: Sequence[int],
        batch_text_seqlen: Sequence[int],
    ) -> torch.Tensor:
        del noisy_latents, timesteps, rotary_embs, batch_vae_seqlen, batch_text_seqlen
        if model_id != "transformer_1":
            raise NativeRelationalObserverError("toy model route differs")
        return transformer(cond_embeds)

    observer = _observer_module()
    roles = _toy_role_specs(observer)
    stream = _new_streaming_observer(observer, roles)
    authorities: list[Mapping[str, Any]] = []
    outputs_exact = True
    for appearance_index, appearance_id in enumerate(APPEARANCE_IDS):
        appearance_offset = (appearance_index - 1) * 0.07
        for cell in plan.sigma_cells:
            shared_noisy = torch.randn((1, 2, 3), generator=generator)
            shared_timestep = torch.tensor([float(cell.step_index)])
            shared_rotary = torch.randn((1, 1, 2, 4), generator=generator)
            base = (
                neutral.reshape(1, PHASES, transformer.patches, transformer.width)[:, :1]
                .repeat(1, PHASES, 1, 1)
                .reshape_as(neutral)
                + appearance_offset
            )
            event_grid = torch.zeros(
                (1, PHASES, transformer.patches, transformer.width),
                dtype=base.dtype,
            )
            for phase_index in range(PHASES):
                moving_patch = min(
                    transformer.patches - 1,
                    int(round((transformer.patches - 1) * phase_index / (PHASES - 1))),
                )
                event_grid[0, phase_index, 0, 0] = 1.4
                event_grid[0, phase_index, moving_patch, 1] = 2.2
                event_grid[0, phase_index, transformer.patches - 1, 2] = 1.6
                event_grid[0, phase_index, moving_patch, 4:6] = (
                    0.2 + phase_index / PHASES
                )
            event = event_grid.reshape_as(base)
            prompt_embeds = {
                "action": (base + event).detach().contiguous(),
                "noop": base.clone().detach().contiguous(),
                "reverse": (
                    base
                    + torch.flip(event_grid, dims=(1,)).reshape_as(event)
                ).detach().contiguous(),
                "static": (
                    base + event_grid[:, :1].repeat(1, PHASES, 1, 1).reshape_as(event)
                ).detach().contiguous(),
            }
            authority = seal_four_arm_forward(
                appearance_id=appearance_id,
                sigma_cell=cell,
                shared_step=shared_step,
                action_kwargs={
                    "model_id": "transformer_1",
                    "noisy_latents": shared_noisy,
                    "timesteps": shared_timestep,
                    "cond_embeds": prompt_embeds["action"],
                    "rotary_embs": shared_rotary,
                    "batch_vae_seqlen": [int(prompt_embeds["action"].shape[1])],
                    "batch_text_seqlen": [int(prompt_embeds["action"].shape[1])],
                },
                prompt_embeds=prompt_embeds,
                instruction_sha256={
                    "action": _instruction_digest("toy action"),
                    "noop": anchor_core.CANONICAL_NOOP_SHA256,
                    "reverse": _instruction_digest("toy reverse"),
                    "static": _instruction_digest("toy static"),
                },
            )
            authorities.append(authority.receipt())
            for arm in ARMS:
                invocation = CaptureInvocation(
                    appearance_id,
                    arm,
                    cell,
                    authority.state_tensor_sha256["noisy_latents"],
                    authority.state_tensor_sha256["timesteps"],
                    authority.state_tensor_sha256["rotary_embs"],
                    2,
                    3,
                )
                unobserved = authority.call(arm).detach()
                with bank.observe(invocation):
                    observed = authority.call(arm)
                outputs_exact = outputs_exact and anchor_core.bits_equal(unobserved, observed)
                capture_group = bank.consume(invocation)
                _stream_native_capture_group(
                    observer=observer,
                    stream=stream,
                    captures=capture_group,
                    roles=roles,
                    prompt_sha256=authority.prompt_embedding_sha256[arm],
                )
                bank.zeroize(capture_group)
                if any(
                    int(torch.count_nonzero(tensor).item()) != 0
                    for block_capture in capture_group
                    for tensor in (
                        block_capture.query,
                        block_capture.key,
                        block_capture.derived_qk_role_responsibility_proxy,
                    )
                ):
                    raise NativeRelationalObserverError(
                        "upstream raw capture zeroization failed"
                    )

    if not outputs_exact:
        raise NativeRelationalObserverError("toy observer changed official forward output")
    observation_receipt = _finalize_streaming_observer(stream)
    p0b = transformer(neutral.clone()).detach()
    p0 = anchor_core.assert_p0_exact_replay(p0a, p0b)
    freeze_after = anchor_core.frozen_module_certificate(transformer)
    if freeze_before["digest"] != freeze_after["digest"]:
        raise NativeRelationalObserverError("frozen toy base changed")
    bank_receipt = bank.receipt()
    if (
        bank_receipt["capture_count"] != plan.block_capture_count
        or bank_receipt["zeroized_count"] != plan.block_capture_count
        or bank_receipt["resident_invocation_count"] != 0
    ):
        raise NativeRelationalObserverError("toy capture matrix did not close")
    value = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "toy_only": True,
        "native_contract": native_gpu_launch_contract(plan),
        "capture_plan": plan.receipt(),
        "four_arm_authorities": authorities,
        "capture_bank": bank_receipt,
        "relational_observer": observation_receipt,
        "frozen_base_unchanged": True,
        "freeze_before_digest": freeze_before["digest"],
        "freeze_after_digest": freeze_after["digest"],
        "observer_output_bit_exact": True,
        "p0_exact_replay": p0,
        "raw_qk_and_derived_role_proxies_zeroized": True,
        "whole_bundle_raw_residency": False,
        "streamed_one_arm_four_blocks_then_zeroized": True,
        "persistent_output_field_allowlist": list(PUBLISHED_FIELDS),
        "decoder_called": False,
        "renderer_called": False,
        "optimizer_created": False,
        "parameter_updates": 0,
        "injection_called": False,
        "candidate_output_modified": False,
        "target_teacher_consumed": False,
        "target_inputs_consumed": False,
        "real_native_capture_completed": False,
        "gpu_launch_authorized": False,
        "scientific_claim_authorized": False,
        "stable_transferable_action_representation_claimed": False,
    }
    return {**value, "digest": _canonical_digest(value)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--print-gpu-contract", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    value = _dry_run() if args.dry_run else native_gpu_launch_contract()
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
        except FileExistsError as error:
            raise NativeRelationalObserverError("refusing to overwrite output receipt") from error
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

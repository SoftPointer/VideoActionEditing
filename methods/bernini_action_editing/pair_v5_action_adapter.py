#!/usr/bin/env python3
"""Independent native-row Action-LoRA adapter for Bernini PAIR-v5.

PAIR-v5 keeps the frozen Bernini base and any frozen CIO self-attention
adapter unchanged.  This module installs a second, independent LoRA only on
``attn2.to_q`` and ``attn2.to_out[0]`` in blocks 0 through 22.  The residual
is visible only on the noisy-target suffix of Bernini's native
``none/V/I/VI`` visual pack after append padding and contiguous Ulysses SP4
slicing.  Source/reference condition rows and padding rows remain byte-exact
base outputs.

The action adapter is also tied to the released exact40 UniPC schedule.  Its
pre-registered high/mid steps are indices 0..32 and 33..37 respectively;
indices 38..39 are low-sigma base-only steps.  A low-sigma route returns the
base projection directly, rather than evaluating a LoRA and multiplying its
result by zero.

This file is an adapter/routing primitive, not evidence that PAIR-v5 performs
semantic action editing.  It consumes no proposal video, paired target, mask,
flow, pose, track, or trajectory.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional

import torch
from torch import nn

import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-pair-v5-native-target-action-lora-v1"
TOTAL_BLOCKS_1P3B = 30
ACTION_BLOCK_INDICES = tuple(range(23))
ACTION_LORA_RANK = 8
ACTION_LORA_ALPHA = 8.0
ACTION_LORA_DROPOUT = 0.0
ALLOWED_SP_SIZES = frozenset({1, 4})
NATIVE_BRANCHES = ("none", "V", "I", "VI")

HIGH_SIGMA_INDICES = tuple(range(33))
MID_SIGMA_INDICES = tuple(range(33, 38))
LOW_SIGMA_INDICES = tuple(range(38, 40))
HIGH_SIGMA_WEIGHT = 1.0
MID_SIGMA_WEIGHT = 0.5
LOW_SIGMA_WEIGHT = 0.0


class PairV5ActionAdapterError(RuntimeError):
    """Raised before an ambiguous PAIR-v5 action route or state is used."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PairV5ActionAdapterError(
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairV5ActionAdapterError(f"{label} must be a positive integer")
    return value


def _validate_registered_schedule() -> None:
    complete = HIGH_SIGMA_INDICES + MID_SIGMA_INDICES + LOW_SIGMA_INDICES
    if complete != tuple(range(sigma_strata.NUM_INFERENCE_STEPS)):
        raise RuntimeError("PAIR-v5 sigma index partition is not exact40")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in HIGH_SIGMA_INDICES
    ):
        raise RuntimeError("PAIR-v5 high-sigma index set differs from its threshold")
    if any(
        not 0.25 <= sigma_strata.PINNED_POSITIVE_SIGMAS[index] < 0.55
        for index in MID_SIGMA_INDICES
    ):
        raise RuntimeError("PAIR-v5 mid-sigma index set differs from its thresholds")
    if any(
        sigma_strata.PINNED_POSITIVE_SIGMAS[index] >= 0.25
        for index in LOW_SIGMA_INDICES
    ):
        raise RuntimeError("PAIR-v5 low-sigma index set differs from its threshold")


_validate_registered_schedule()


def sigma_gate(schedule_index: Any) -> tuple[str, float]:
    """Return the pre-registered exact40 action gate for one schedule index."""

    if (
        isinstance(schedule_index, bool)
        or not isinstance(schedule_index, int)
        or not 0 <= schedule_index < sigma_strata.NUM_INFERENCE_STEPS
    ):
        raise PairV5ActionAdapterError(
            "sigma_schedule_index must be an exact integer in [0,39]"
        )
    if schedule_index in HIGH_SIGMA_INDICES:
        return "high", HIGH_SIGMA_WEIGHT
    if schedule_index in MID_SIGMA_INDICES:
        return "mid", MID_SIGMA_WEIGHT
    if schedule_index in LOW_SIGMA_INDICES:
        return "low_base_only", LOW_SIGMA_WEIGHT
    raise PairV5ActionAdapterError("sigma_schedule_index is not preregistered")


@dataclass(frozen=True)
class PairV5ActionRoute:
    """One native Bernini visual branch before append-pad/SP4 slicing."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    branch_name: str
    sigma_schedule_index: int
    enabled: bool = True

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise PairV5ActionAdapterError(
                "condition_tokens must identify a strict noisy-target suffix"
            )
        if self.branch_name not in NATIVE_BRANCHES:
            raise PairV5ActionAdapterError("branch_name is not a native visual branch")
        if self.branch_name == "none" and self.condition_tokens != 0:
            raise PairV5ActionAdapterError("native none branch cannot contain condition rows")
        if self.branch_name != "none" and self.condition_tokens == 0:
            raise PairV5ActionAdapterError(
                "native conditioned branches must contain condition rows"
            )
        size = _positive_int(
            self.sequence_parallel_size, label="sequence_parallel_size"
        )
        rank = self.sequence_parallel_rank
        if size not in ALLOWED_SP_SIZES:
            raise PairV5ActionAdapterError("only SP1 tests and native SP4 are supported")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise PairV5ActionAdapterError("SP rank lies outside its group")
        sigma_gate(self.sigma_schedule_index)
        if not isinstance(self.enabled, bool):
            raise PairV5ActionAdapterError("enabled must be boolean")

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    @property
    def gate_name(self) -> str:
        return sigma_gate(self.sigma_schedule_index)[0]

    @property
    def gate_weight(self) -> float:
        if not self.enabled:
            return 0.0
        return sigma_gate(self.sigma_schedule_index)[1]

    @property
    def adapter_active(self) -> bool:
        return self.enabled and self.gate_weight > 0.0

    def global_target_selector(self, *, device: torch.device) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(self.condition_tokens, dtype=torch.bool, device=device),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
            )
        )

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
        selector = self.global_target_selector(device=device)
        padded_length = self.local_length * self.sequence_parallel_size
        if padded_length > self.total_tokens:
            selector = torch.cat(
                (
                    selector,
                    torch.zeros(
                        padded_length - self.total_tokens,
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
        start = self.sequence_parallel_rank * self.local_length
        return selector[start : start + self.local_length].contiguous()

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "branch_name": self.branch_name,
            "total_tokens": self.total_tokens,
            "condition_tokens": self.condition_tokens,
            "target_tokens": self.target_tokens,
            "sequence_parallel_rank": self.sequence_parallel_rank,
            "sequence_parallel_size": self.sequence_parallel_size,
            "padding_policy": "append_false_then_contiguous_rank_chunk",
            "sigma_schedule_index": self.sigma_schedule_index,
            "sigma_float32_be_hex": sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                self.sigma_schedule_index
            ],
            "sigma_gate": self.gate_name,
            "sigma_gate_weight": self.gate_weight,
            "enabled": self.enabled,
            "adapter_active": self.adapter_active,
        }
        return {**value, "digest": _object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[PairV5ActionRoute]] = ContextVar(
    "bernini_pair_v5_action_route", default=None
)


def active_route() -> Optional[PairV5ActionRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: PairV5ActionRoute) -> Iterator[None]:
    if not isinstance(route, PairV5ActionRoute):
        raise PairV5ActionAdapterError("route must be PairV5ActionRoute")
    if active_route() is not None:
        raise PairV5ActionAdapterError("nested PAIR-v5 action routes are forbidden")
    token: Token[Optional[PairV5ActionRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class PairV5TargetRowActionLoRA(nn.Module):
    """Cross-attention Q/O LoRA evaluated only for active local target rows."""

    def __init__(self, base: nn.Module, *, projection: str):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise PairV5ActionAdapterError(f"{projection} base must be nn.Linear")
        if projection not in {"to_q", "to_out.0"}:
            raise PairV5ActionAdapterError("only cross-attention Q/O may be wrapped")
        self.base = base
        self.projection = projection
        self.rank = ACTION_LORA_RANK
        self.alpha = ACTION_LORA_ALPHA
        self.dropout = ACTION_LORA_DROPOUT
        self.action_lora_a = nn.Linear(
            base.in_features, self.rank, bias=False, dtype=torch.float32
        )
        self.action_lora_b = nn.Linear(
            self.rank, base.out_features, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.action_lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.action_lora_b.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    @staticmethod
    def _selector(
        hidden_states: torch.Tensor, route: PairV5ActionRoute
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise PairV5ActionAdapterError(
                "native PAIR-v5 action LoRA expects hidden states [1,N,D]"
            )
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise PairV5ActionAdapterError(
                "local hidden sequence differs from native append-pad/SP slice"
            )
        return selector

    def _selected_delta(
        self, hidden_states: torch.Tensor, selector: torch.Tensor, gate_weight: float
    ) -> torch.Tensor:
        # An active V-only route may own zero target rows on the source-only
        # Ulysses shards.  Keep the legal [1,0,D] LoRA path in the graph: its
        # value and local B gradient are exact zero, while every SP rank still
        # participates in the same distributed backward.  Only absent,
        # disabled, and low-sigma routes bypass both factors in forward().
        selected = hidden_states[:, selector, :]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.action_lora_b(self.action_lora_a(selected.float()))
            delta = delta * (self.scale * gate_weight)
        return delta.to(hidden_states.dtype)

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        route = active_route()
        base_shape = (*hidden_states.shape[:-1], self.base.out_features)
        result = torch.zeros(
            base_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        if route is None:
            return result
        selector = self._selector(hidden_states, route)
        if not route.adapter_active:
            return result
        result[:, selector, :] = self._selected_delta(
            hidden_states, selector, route.gate_weight
        )
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = active_route()
        # This direct return gives absent, disabled, and low-sigma routes exact
        # base parity without evaluating either Action-LoRA factor.
        if route is None:
            return base
        selector = self._selector(hidden_states, route)
        if not route.adapter_active:
            return base
        result = base.clone()
        result[:, selector, :] = base[:, selector, :] + self._selected_delta(
            hidden_states, selector, route.gate_weight
        ).to(base.dtype)
        return result


def _trainable_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(list(value.shape)))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class PairV5ActionAdapterHandle:
    transformer: nn.Module
    q_wrappers: tuple[tuple[int, PairV5TargetRowActionLoRA], ...]
    o_wrappers: tuple[tuple[int, PairV5TargetRowActionLoRA], ...]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    original_patch_embedding_id: int
    original_self_attention_ids: tuple[tuple[int, int, int], ...]
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise PairV5ActionAdapterError("action adapter has been restored")
        result: list[tuple[str, nn.Parameter]] = []
        for index, wrapper in self.q_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn2.to_q.action_lora_a.weight",
                        wrapper.action_lora_a.weight,
                    ),
                    (
                        f"blocks.{index}.attn2.to_q.action_lora_b.weight",
                        wrapper.action_lora_b.weight,
                    ),
                )
            )
        for index, wrapper in self.o_wrappers:
            result.extend(
                (
                    (
                        f"blocks.{index}.attn2.to_out.0.action_lora_a.weight",
                        wrapper.action_lora_a.weight,
                    ),
                    (
                        f"blocks.{index}.attn2.to_out.0.action_lora_b.weight",
                        wrapper.action_lora_b.weight,
                    ),
                )
            )
        if len({id(parameter) for _, parameter in result}) != len(result):
            raise PairV5ActionAdapterError("Action-LoRA parameter aliases another")
        if any(not parameter.requires_grad for _, parameter in result):
            raise PairV5ActionAdapterError("Action-LoRA parameter is unexpectedly frozen")
        return tuple(result)

    def base_parameters_frozen(self) -> bool:
        trainable_ids = {
            id(parameter) for _, parameter in self.trainable_named_parameters()
        }
        observed_trainable_ids = {
            id(parameter)
            for parameter in self.transformer.parameters()
            if parameter.requires_grad
        }
        return observed_trainable_ids == trainable_ids

    def self_attention_untouched(self) -> bool:
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            return False
        current: list[tuple[int, int, int]] = []
        for block in blocks:
            attention = getattr(block, "attn1", None)
            output = getattr(attention, "to_out", None)
            if output is None or len(output) < 1:
                return False
            current.append(
                (id(attention), id(getattr(attention, "to_q", None)), id(output[0]))
            )
        return tuple(current) == self.original_self_attention_ids

    @contextmanager
    def route(self, route: PairV5ActionRoute) -> Iterator[None]:
        if self.restored:
            raise PairV5ActionAdapterError("cannot route a restored action adapter")
        with activate_route(route):
            yield

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        state = {
            name: parameter.detach().float().cpu().contiguous().clone()
            for name, parameter in self.trainable_named_parameters()
        }
        if set(state) != {name for name, _ in self.trainable_named_parameters()}:
            raise PairV5ActionAdapterError("Action-LoRA save-state closure failed")
        return state

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Any]:
        if self.restored:
            raise PairV5ActionAdapterError("cannot load a restored action adapter")
        if not isinstance(state, Mapping):
            raise PairV5ActionAdapterError("Action-LoRA state must be a mapping")
        expected = dict(self.trainable_named_parameters())
        actual_keys = set(state)
        if actual_keys != set(expected):
            missing = sorted(set(expected) - actual_keys)
            unexpected = sorted(actual_keys - set(expected))
            raise PairV5ActionAdapterError(
                "Action-LoRA state key closure differs: "
                f"missing={missing[:2]} unexpected={unexpected[:2]}"
            )
        normalized: dict[str, torch.Tensor] = {}
        for name in sorted(expected):
            value = state[name]
            parameter = expected[name]
            if not isinstance(value, torch.Tensor):
                raise PairV5ActionAdapterError(f"Action-LoRA state {name} is not a tensor")
            if (
                value.dtype != torch.float32
                or value.device.type != "cpu"
                or value.layout != torch.strided
                or value.requires_grad
                or value.grad_fn is not None
                or not value.is_contiguous()
                or tuple(value.shape) != tuple(parameter.shape)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise PairV5ActionAdapterError(
                    f"Action-LoRA state {name} must be detached finite contiguous CPU FP32 with exact shape"
                )
            normalized[name] = value
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(normalized[name].to(device=parameter.device))
        digest = _trainable_state_digest(normalized)
        value = {
            "schema_version": SCHEMA_VERSION,
            "state_key_count": len(normalized),
            "state_key_sha256": _object_sha256(sorted(normalized)),
            "state_tensor_sha256": digest,
            "closed_exact_key_set": True,
        }
        return {**value, "digest": _object_sha256(value)}

    def receipt(self) -> Mapping[str, Any]:
        patch = getattr(self.transformer, "patch_embedding", None)
        trainable = self.trainable_named_parameters()
        value = {
            "schema_version": SCHEMA_VERSION,
            "block_indices": list(ACTION_BLOCK_INDICES),
            "projections": ["attn2.to_q", "attn2.to_out.0"],
            "rank": ACTION_LORA_RANK,
            "alpha": ACTION_LORA_ALPHA,
            "dropout": ACTION_LORA_DROPOUT,
            "target_suffix_only": True,
            "condition_rows_exact_base": True,
            "padding_rows_exact_base": True,
            "patch_embedding_untouched": id(patch) == self.original_patch_embedding_id,
            "patch_vae_latent_untouched": True,
            "self_attention_and_frozen_cio_untouched": self.self_attention_untouched(),
            "key_value_trainable": False,
            "late_blocks_trainable": False,
            "native_branches": list(NATIVE_BRANCHES),
            "sp_selector": "append_false_then_contiguous_rank_chunk",
            "exact40_schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "sigma_gate_indices": {
                "high_weight_1": list(HIGH_SIGMA_INDICES),
                "mid_weight_0.5": list(MID_SIGMA_INDICES),
                "low_base_only_weight_0": list(LOW_SIGMA_INDICES),
            },
            "low_sigma_direct_base_return": True,
            "gradient_checkpointing_must_be_disabled": True,
            "route_context_scope": "native_shared_step_forward_and_backward",
            "base_and_frozen_cio_parameters_frozen": self.base_parameters_frozen(),
            "trainable_state_closed": True,
            "trainable_state_key_sha256": _object_sha256(
                sorted(name for name, _ in trainable)
            ),
            "trainable": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in trainable
            ],
            "proposal_visual_data_consumed": False,
            "paired_target_consumed": False,
            "mask_flow_pose_track_trajectory_consumed": False,
            "semantic_action_claim": False,
        }
        return {**value, "digest": _object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise PairV5ActionAdapterError("action adapter cannot be restored now")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            raise PairV5ActionAdapterError("transformer block count changed")
        if id(getattr(self.transformer, "patch_embedding", None)) != self.original_patch_embedding_id:
            raise PairV5ActionAdapterError(
                "native patch embedding changed while action adapter was active"
            )
        if not self.self_attention_untouched():
            raise PairV5ActionAdapterError(
                "self-attention/CIO modules changed while action adapter was active"
            )
        for index, original in self.original_q:
            blocks[index].attn2.to_q = original
        for index, original in self.original_o:
            blocks[index].attn2.to_out[0] = original
        self.restored = True


def install_pair_v5_action_adapter(
    transformer: nn.Module,
) -> PairV5ActionAdapterHandle:
    """Install the fixed rank-8/alpha-8 block-0..22 PAIR-v5 Action-LoRA."""

    if not isinstance(transformer, nn.Module):
        raise PairV5ActionAdapterError("transformer must be nn.Module")
    if any(parameter.requires_grad for parameter in transformer.parameters()):
        raise PairV5ActionAdapterError(
            "freeze the complete Bernini base and any CIO adapter before installation"
        )
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise PairV5ActionAdapterError(
            "Bernini-R 1.3B native transformer structure differs"
        )

    hidden = int(patch.out_channels)
    original_self_attention_ids: list[tuple[int, int, int]] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "attn1", None)
        output = getattr(attention, "to_out", None)
        if output is None or len(output) < 1:
            raise PairV5ActionAdapterError(f"block {index} self-attention structure differs")
        original_self_attention_ids.append(
            (id(attention), id(getattr(attention, "to_q", None)), id(output[0]))
        )

    original_q: list[tuple[int, nn.Module]] = []
    original_o: list[tuple[int, nn.Module]] = []
    for index in ACTION_BLOCK_INDICES:
        attention = getattr(blocks[index], "attn2", None)
        query = getattr(attention, "to_q", None)
        output = getattr(attention, "to_out", None)
        if (
            not isinstance(query, nn.Linear)
            or not isinstance(output, nn.ModuleList)
            or len(output) != 2
            or not isinstance(output[0], nn.Linear)
            or query.in_features != hidden
            or query.out_features != hidden
            or output[0].in_features != hidden
            or output[0].out_features != hidden
        ):
            raise PairV5ActionAdapterError(f"block {index} cross-attention Q/O differs")
        original_q.append((index, query))
        original_o.append((index, output[0]))

    device = patch.weight.device
    q_wrappers: list[tuple[int, PairV5TargetRowActionLoRA]] = []
    o_wrappers: list[tuple[int, PairV5TargetRowActionLoRA]] = []
    try:
        for (index, query), (_, output) in zip(original_q, original_o):
            q_wrapper = PairV5TargetRowActionLoRA(
                query, projection="to_q"
            ).to(device=device)
            o_wrapper = PairV5TargetRowActionLoRA(
                output, projection="to_out.0"
            ).to(device=device)
            blocks[index].attn2.to_q = q_wrapper
            blocks[index].attn2.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
    except Exception:
        for index, original in original_q:
            blocks[index].attn2.to_q = original
        for index, original in original_o:
            blocks[index].attn2.to_out[0] = original
        raise

    handle = PairV5ActionAdapterHandle(
        transformer=transformer,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(original_q),
        original_o=tuple(original_o),
        original_patch_embedding_id=id(patch),
        original_self_attention_ids=tuple(original_self_attention_ids),
    )
    receipt = handle.receipt()
    if (
        not handle.base_parameters_frozen()
        or receipt["patch_embedding_untouched"] is not True
        or receipt["self_attention_and_frozen_cio_untouched"] is not True
    ):
        handle.restore()
        raise PairV5ActionAdapterError("PAIR-v5 Action-LoRA scope closure failed")
    return handle


__all__ = [
    "ACTION_BLOCK_INDICES",
    "ACTION_LORA_ALPHA",
    "ACTION_LORA_DROPOUT",
    "ACTION_LORA_RANK",
    "HIGH_SIGMA_INDICES",
    "LOW_SIGMA_INDICES",
    "MID_SIGMA_INDICES",
    "PairV5ActionAdapterError",
    "PairV5ActionAdapterHandle",
    "PairV5ActionRoute",
    "PairV5TargetRowActionLoRA",
    "SCHEMA_VERSION",
    "activate_route",
    "active_route",
    "install_pair_v5_action_adapter",
    "sigma_gate",
]

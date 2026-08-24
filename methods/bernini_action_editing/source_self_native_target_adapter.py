#!/usr/bin/env python3
"""Native-pack-compatible target-row Q/O LoRA for Bernini-R 1.3B.

The adapter intentionally leaves ``patch_embedding`` and
``patch_vae_latent`` untouched.  It is activated only around Bernini's native
``shared_step`` calls and applies a low-rank residual to the noisy-target
suffix of each none/V/VI branch after Ulysses append-padding and contiguous
SP4 slicing.  Condition K/V rows and every non-target Q/O row are bitwise base
model outputs.

This is a routing primitive, not a training result.  A caller must still use
the native RV2V guidance formula and a train/inference-matched noise schedule.
Gradient checkpointing must be disabled: recomputation happens after a
per-branch route context has exited and could otherwise silently use the wrong
target selector.  Bernini-R 1.3B on SP4 is small enough for this explicit
scientific canary contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn


SCHEMA_VERSION = "bernini-native-target-row-qo-lora-v1"
TOTAL_BLOCKS_1P3B = 30
DEFAULT_BLOCK_INDICES = tuple(range(23))
REGISTERED_ALL_BLOCKS_ABLATION = tuple(range(TOTAL_BLOCKS_1P3B))
ALLOWED_BLOCK_SCOPES = {DEFAULT_BLOCK_INDICES, REGISTERED_ALL_BLOCKS_ABLATION}
ALLOWED_SP_SIZES = {1, 4}


class NativeTargetAdapterError(RuntimeError):
    """Raised before a non-native route or ambiguous adapter is used."""


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
        raise NativeTargetAdapterError(f"receipt is not canonical JSON: {error}") from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NativeTargetAdapterError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class NativeTargetRoute:
    """One native visual branch before Ulysses SP slicing."""

    total_tokens: int
    condition_tokens: int
    sequence_parallel_rank: int
    sequence_parallel_size: int
    branch_name: str
    enabled: bool = True

    def __post_init__(self) -> None:
        total = _positive_int(self.total_tokens, label="total_tokens")
        if (
            isinstance(self.condition_tokens, bool)
            or not isinstance(self.condition_tokens, int)
            or not 0 <= self.condition_tokens < total
        ):
            raise NativeTargetAdapterError(
                "condition_tokens must identify a strict target suffix"
            )
        size = _positive_int(
            self.sequence_parallel_size, label="sequence_parallel_size"
        )
        rank = self.sequence_parallel_rank
        if size not in ALLOWED_SP_SIZES:
            raise NativeTargetAdapterError("only SP1 tests and native SP4 are supported")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
            raise NativeTargetAdapterError("SP rank lies outside its group")
        if self.branch_name not in {"none", "V", "I", "VI"}:
            raise NativeTargetAdapterError("branch_name is not a native visual branch")
        if not isinstance(self.enabled, bool):
            raise NativeTargetAdapterError("enabled must be boolean")

    @property
    def target_tokens(self) -> int:
        return self.total_tokens - self.condition_tokens

    @property
    def local_length(self) -> int:
        return math.ceil(self.total_tokens / self.sequence_parallel_size)

    def global_target_selector(self, *, device: torch.device) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(
                    self.condition_tokens, dtype=torch.bool, device=device
                ),
                torch.ones(self.target_tokens, dtype=torch.bool, device=device),
            )
        )

    def local_target_selector(self, *, device: torch.device) -> torch.Tensor:
        selector = self.global_target_selector(device=device)
        padded = self.local_length * self.sequence_parallel_size
        if padded > self.total_tokens:
            selector = torch.cat(
                (
                    selector,
                    torch.zeros(
                        padded - self.total_tokens,
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
            "enabled": self.enabled,
        }
        return {**value, "digest": _object_sha256(value)}


_ACTIVE_ROUTE: ContextVar[Optional[NativeTargetRoute]] = ContextVar(
    "bernini_native_target_row_route", default=None
)


def active_route() -> Optional[NativeTargetRoute]:
    return _ACTIVE_ROUTE.get()


@contextmanager
def activate_route(route: NativeTargetRoute) -> Iterator[None]:
    if not isinstance(route, NativeTargetRoute):
        raise NativeTargetAdapterError("route must be NativeTargetRoute")
    if active_route() is not None:
        raise NativeTargetAdapterError("nested native target routes are forbidden")
    token: Token[Optional[NativeTargetRoute]] = _ACTIVE_ROUTE.set(route)
    try:
        yield
    finally:
        _ACTIVE_ROUTE.reset(token)


class NativeTargetRowLoRA(nn.Module):
    """Q/O residual that is exactly zero outside local target rows."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float, projection: str):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise NativeTargetAdapterError(f"{projection} base must be nn.Linear")
        self.base = base
        self.rank = _positive_int(rank, label="rank")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0.0
        ):
            raise NativeTargetAdapterError("alpha must be finite and positive")
        if projection not in {"to_q", "to_out.0"}:
            raise NativeTargetAdapterError("only self-attention Q/O may be wrapped")
        self.alpha = float(alpha)
        self.projection = projection
        self.lora_a = nn.Linear(
            base.in_features, self.rank, bias=False, dtype=torch.float32
        )
        self.lora_b = nn.Linear(
            self.rank, base.out_features, bias=False, dtype=torch.float32
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    @property
    def scale(self) -> float:
        return self.alpha / float(self.rank)

    @property
    def weight(self) -> Any:
        return self.base.weight

    @property
    def bias(self) -> Any:
        return self.base.bias

    def adapter_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        route = active_route()
        base_shape = self.base(hidden_states).shape
        if route is None or not route.enabled:
            return torch.zeros(base_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise NativeTargetAdapterError("native target LoRA expects [1,N,D]")
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise NativeTargetAdapterError(
                "local hidden sequence differs from native append-pad/SP slice"
            )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.lora_b(self.lora_a(hidden_states.float())) * self.scale
        return delta.to(hidden_states.dtype) * selector.view(1, -1, 1).to(
            hidden_states.dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base(hidden_states)
        route = active_route()
        if route is None or not route.enabled:
            return base
        if hidden_states.ndim != 3 or int(hidden_states.shape[0]) != 1:
            raise NativeTargetAdapterError("native target LoRA expects [1,N,D]")
        selector = route.local_target_selector(device=hidden_states.device)
        if int(hidden_states.shape[1]) != int(selector.numel()):
            raise NativeTargetAdapterError(
                "local hidden sequence differs from native append-pad/SP slice"
            )
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            delta = self.lora_b(self.lora_a(hidden_states.float())) * self.scale
        return base + delta.to(base.dtype) * selector.view(1, -1, 1).to(base.dtype)


@dataclass
class NativeTargetAdapterHandle:
    transformer: nn.Module
    q_wrappers: tuple[tuple[int, NativeTargetRowLoRA], ...]
    o_wrappers: tuple[tuple[int, NativeTargetRowLoRA], ...]
    original_q: tuple[tuple[int, nn.Module], ...]
    original_o: tuple[tuple[int, nn.Module], ...]
    block_indices: tuple[int, ...]
    original_patch_embedding_id: int
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise NativeTargetAdapterError("adapter has been restored")
        result: list[tuple[str, nn.Parameter]] = []
        for index, wrapper in self.q_wrappers:
            result.extend(
                (
                    (f"blocks.{index}.attn1.to_q.lora_a.weight", wrapper.lora_a.weight),
                    (f"blocks.{index}.attn1.to_q.lora_b.weight", wrapper.lora_b.weight),
                )
            )
        for index, wrapper in self.o_wrappers:
            result.extend(
                (
                    (f"blocks.{index}.attn1.to_out.0.lora_a.weight", wrapper.lora_a.weight),
                    (f"blocks.{index}.attn1.to_out.0.lora_b.weight", wrapper.lora_b.weight),
                )
            )
        if len({id(value) for _, value in result}) != len(result):
            raise NativeTargetAdapterError("trainable adapter parameter aliases another")
        if any(not value.requires_grad for _, value in result):
            raise NativeTargetAdapterError("adapter parameter is unexpectedly frozen")
        return tuple(result)

    def base_parameters_frozen(self) -> bool:
        trainable = {id(value) for _, value in self.trainable_named_parameters()}
        return all(
            id(value) in trainable or not value.requires_grad
            for value in self.transformer.parameters()
        )

    @contextmanager
    def route(self, route: NativeTargetRoute) -> Iterator[None]:
        if self.restored:
            raise NativeTargetAdapterError("cannot route a restored adapter")
        with activate_route(route):
            yield

    def state_dict_for_save(self) -> Mapping[str, torch.Tensor]:
        return {
            name: value.detach().float().cpu().contiguous()
            for name, value in self.trainable_named_parameters()
        }

    def receipt(self) -> Mapping[str, Any]:
        patch = getattr(self.transformer, "patch_embedding", None)
        value = {
            "schema_version": SCHEMA_VERSION,
            "block_indices": list(self.block_indices),
            "default_early_mid_scope": self.block_indices == DEFAULT_BLOCK_INDICES,
            "registered_all30_ablation": self.block_indices
            == REGISTERED_ALL_BLOCKS_ABLATION,
            "projections": ["attn1.to_q", "attn1.to_out.0"],
            "target_row_only": True,
            "patch_embedding_untouched": id(patch) == self.original_patch_embedding_id,
            "patch_vae_latent_untouched": True,
            "key_value_trainable": False,
            "cross_attention_trainable": False,
            "late_blocks_trainable": any(index >= 23 for index in self.block_indices),
            "native_branches": ["none", "V", "I", "VI"],
            "sp_selector": "append_false_then_contiguous_rank_chunk",
            "gradient_checkpointing_must_be_disabled": True,
            "route_context_scope": "native_shared_step_forward_only",
            "base_parameters_frozen": self.base_parameters_frozen(),
            "trainable": [
                {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in self.trainable_named_parameters()
            ],
            "semantic_action_claim": False,
        }
        return {**value, "digest": _object_sha256(value)}

    def restore(self) -> None:
        if self.restored or active_route() is not None:
            raise NativeTargetAdapterError("adapter cannot be restored now")
        blocks = tuple(getattr(self.transformer, "blocks", ()))
        if len(blocks) != TOTAL_BLOCKS_1P3B:
            raise NativeTargetAdapterError("transformer block count changed")
        if id(getattr(self.transformer, "patch_embedding", None)) != self.original_patch_embedding_id:
            raise NativeTargetAdapterError("native patch embedding changed while adapter active")
        for index, original in self.original_q:
            blocks[index].attn1.to_q = original
        for index, original in self.original_o:
            blocks[index].attn1.to_out[0] = original
        self.restored = True


def install_native_target_adapter(
    transformer: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    block_indices: Sequence[int] = DEFAULT_BLOCK_INDICES,
) -> NativeTargetAdapterHandle:
    """Install Q/O-only LoRA while preserving Bernini's native patch path."""

    if not isinstance(transformer, nn.Module):
        raise NativeTargetAdapterError("transformer must be nn.Module")
    if any(value.requires_grad for value in transformer.parameters()):
        raise NativeTargetAdapterError("freeze the complete transformer before installing")
    blocks = tuple(getattr(transformer, "blocks", ()))
    patch = getattr(transformer, "patch_embedding", None)
    if (
        len(blocks) != TOTAL_BLOCKS_1P3B
        or not isinstance(patch, nn.Conv3d)
        or not callable(getattr(transformer, "patch_vae_latent", None))
    ):
        raise NativeTargetAdapterError("Bernini 1.3B native transformer structure differs")
    indices = tuple(block_indices)
    if indices not in ALLOWED_BLOCK_SCOPES:
        raise NativeTargetAdapterError("block scope is not preregistered")
    hidden = int(patch.out_channels)
    originals_q: list[tuple[int, nn.Module]] = []
    originals_o: list[tuple[int, nn.Module]] = []
    for index in indices:
        attention = getattr(blocks[index], "attn1", None)
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
            raise NativeTargetAdapterError(f"block {index} self-attention Q/O differs")
        originals_q.append((index, query))
        originals_o.append((index, output[0]))

    device = patch.weight.device
    q_wrappers: list[tuple[int, NativeTargetRowLoRA]] = []
    o_wrappers: list[tuple[int, NativeTargetRowLoRA]] = []
    try:
        for (index, query), (_, output) in zip(originals_q, originals_o):
            q_wrapper = NativeTargetRowLoRA(
                query, rank=rank, alpha=alpha, projection="to_q"
            ).to(device=device)
            o_wrapper = NativeTargetRowLoRA(
                output, rank=rank, alpha=alpha, projection="to_out.0"
            ).to(device=device)
            blocks[index].attn1.to_q = q_wrapper
            blocks[index].attn1.to_out[0] = o_wrapper
            q_wrappers.append((index, q_wrapper))
            o_wrappers.append((index, o_wrapper))
    except Exception:
        for index, original in originals_q:
            blocks[index].attn1.to_q = original
        for index, original in originals_o:
            blocks[index].attn1.to_out[0] = original
        raise
    handle = NativeTargetAdapterHandle(
        transformer=transformer,
        q_wrappers=tuple(q_wrappers),
        o_wrappers=tuple(o_wrappers),
        original_q=tuple(originals_q),
        original_o=tuple(originals_o),
        block_indices=indices,
        original_patch_embedding_id=id(patch),
    )
    if not handle.base_parameters_frozen() or not handle.receipt()[
        "patch_embedding_untouched"
    ]:
        handle.restore()
        raise NativeTargetAdapterError("native adapter scope closure failed")
    return handle


__all__ = [
    "DEFAULT_BLOCK_INDICES",
    "NativeTargetAdapterError",
    "NativeTargetAdapterHandle",
    "NativeTargetRoute",
    "NativeTargetRowLoRA",
    "SCHEMA_VERSION",
    "activate_route",
    "active_route",
    "install_native_target_adapter",
]

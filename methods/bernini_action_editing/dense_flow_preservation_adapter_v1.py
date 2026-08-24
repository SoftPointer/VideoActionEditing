#!/usr/bin/env python3
"""Independent hidden-state preservation residual for dense-flow action editing.

The action adapter and this adapter are installed as two sequential residual
branches.  This branch deliberately receives exact-zero motion features and a
target-token activity mask: its checkpoint is trained only on
original-source -> original-source flow matching under the real action
instruction.  At inference its strength is independent from the frozen motion
adapter, so preservation no longer has to cancel motion in one shared weight
vector.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
import types
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn

import dense_flow_token_adapter_v1 as motion_core


SCHEMA_VERSION = "bernini-dense-flow-preservation-adapter-v1"
MODULE_NAME = "dense_flow_preservation_adapter"


class PreservationAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreservationInvocation:
    activity: torch.Tensor = field(repr=False, compare=False)

    def validate(self) -> None:
        if (
            not isinstance(self.activity, torch.Tensor)
            or self.activity.dtype != torch.bool
            or self.activity.ndim != 3
            or int(self.activity.shape[0]) != 1
            or int(self.activity.shape[2]) != 1
        ):
            raise PreservationAdapterError("activity must be bool [1,N,1]")


_CURRENT: contextvars.ContextVar[Optional[PreservationInvocation]] = (
    contextvars.ContextVar("bernini_preservation_invocation", default=None)
)


@contextlib.contextmanager
def preservation_invocation(
    invocation: PreservationInvocation,
) -> Iterator[PreservationInvocation]:
    if not isinstance(invocation, PreservationInvocation):
        raise PreservationAdapterError("preservation context received the wrong type")
    invocation.validate()
    if _CURRENT.get() is not None:
        raise PreservationAdapterError("nested preservation invocations are forbidden")
    token = _CURRENT.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT.reset(token)


def current_preservation_invocation() -> Optional[PreservationInvocation]:
    return _CURRENT.get()


def _local_zero_motion(
    invocation: PreservationInvocation, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    activity = invocation.activity.to(device=hidden.device)
    if int(activity.shape[1]) != int(hidden.shape[1]):
        try:
            from bernini.parallel import (
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )
        except ImportError as error:
            raise PreservationAdapterError(
                "global preservation activity requires Bernini SP helpers"
            ) from error
        activity = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(activity, dim=1), dim=1
        )
    if tuple(activity.shape[:2]) != tuple(hidden.shape[:2]):
        raise PreservationAdapterError(
            "rank-local preservation activity differs from hidden states"
        )
    features = torch.zeros(
        (*hidden.shape[:2], motion_core.FEATURE_WIDTH),
        dtype=torch.float32,
        device=hidden.device,
    )
    return features, activity.bool()


@dataclass
class PreservationPatchHandle:
    transformer: Any
    block_indices: tuple[int, ...]
    adapters: tuple[motion_core.DenseFlowResidualBlock, ...]
    original_forwards: tuple[Any, ...] = field(repr=False)
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise PreservationAdapterError("preservation patch is already restored")
        rows: list[tuple[str, nn.Parameter]] = []
        for index, adapter in zip(self.block_indices, self.adapters):
            for name, parameter in adapter.named_parameters():
                rows.append((f"blocks.{index}.{MODULE_NAME}.{name}", parameter))
        if not rows or len({id(parameter) for _, parameter in rows}) != len(rows):
            raise PreservationAdapterError(
                "preservation parameter closure differs"
            )
        return tuple(rows)

    def zero_effect(self) -> bool:
        return all(adapter.is_zero_effect() for adapter in self.adapters)

    def load_dense_flow_state_strict(
        self, state: Mapping[str, torch.Tensor], *, output_scale: float = 1.0
    ) -> None:
        expected = dict(self.trainable_named_parameters())
        remapped = {
            name.replace(".dense_flow_adapter.", f".{MODULE_NAME}."): value
            for name, value in state.items()
        }
        if set(remapped) != set(expected):
            raise PreservationAdapterError("preservation state-key closure differs")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = remapped[name]
                if value.shape != parameter.shape or not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise PreservationAdapterError(
                        f"preservation state tensor differs: {name}"
                    )
                if name.endswith("output.weight"):
                    value = value.float().mul(float(output_scale))
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def restore(self) -> None:
        if self.restored:
            return
        for index, adapter, original in zip(
            self.block_indices, self.adapters, self.original_forwards
        ):
            block = self.transformer.blocks[index]
            if getattr(block, MODULE_NAME, None) is not adapter:
                raise PreservationAdapterError(
                    "preservation module changed behind patch handle"
                )
            block.forward = original
            delattr(block, MODULE_NAME)
        self.restored = True


def install_preservation_adapter(
    model: Any,
    *,
    block_indices: Sequence[int] = motion_core.BLOCK_INDICES,
    hidden_width: int = motion_core.HIDDEN_WIDTH,
    bottleneck_width: int = motion_core.BOTTLENECK_WIDTH,
) -> PreservationPatchHandle:
    transformer = motion_core._resolve_transformer(model)
    transformer.requires_grad_(False)
    indices = tuple(int(item) for item in block_indices)
    if indices != tuple(sorted(set(indices))) or any(
        item < 0 or item >= motion_core.EXPECTED_BLOCK_COUNT for item in indices
    ):
        raise PreservationAdapterError(
            "preservation block indices must be sorted unique in [0,29]"
        )
    adapters: list[motion_core.DenseFlowResidualBlock] = []
    originals: list[Any] = []
    installed: list[int] = []
    try:
        for index in indices:
            block = transformer.blocks[index]
            if hasattr(block, MODULE_NAME):
                raise PreservationAdapterError(
                    f"block {index} already has a preservation adapter"
                )
            adapter = motion_core.DenseFlowResidualBlock(
                hidden_width=hidden_width,
                bottleneck_width=bottleneck_width,
            )
            block.add_module(MODULE_NAME, adapter)
            original = block.forward

            def wrapped_forward(
                self: Any,
                *args: Any,
                _original: Any = original,
                _adapter: motion_core.DenseFlowResidualBlock = adapter,
                **kwargs: Any,
            ) -> torch.Tensor:
                hidden = _original(*args, **kwargs)
                invocation = current_preservation_invocation()
                if invocation is None:
                    return hidden
                features, activity = _local_zero_motion(invocation, hidden)
                return _adapter(hidden, features, activity)

            block.forward = types.MethodType(wrapped_forward, block)
            adapters.append(adapter)
            originals.append(original)
            installed.append(index)
    except Exception:
        for index, adapter, original in zip(
            reversed(installed), reversed(adapters), reversed(originals)
        ):
            block = transformer.blocks[index]
            block.forward = original
            if getattr(block, MODULE_NAME, None) is adapter:
                delattr(block, MODULE_NAME)
        raise
    handle = PreservationPatchHandle(
        transformer=transformer,
        block_indices=indices,
        adapters=tuple(adapters),
        original_forwards=tuple(originals),
    )
    if not handle.zero_effect():
        handle.restore()
        raise PreservationAdapterError("preservation adapter lost zero-init fallback")
    return handle


__all__ = [
    "MODULE_NAME",
    "PreservationAdapterError",
    "PreservationInvocation",
    "PreservationPatchHandle",
    "SCHEMA_VERSION",
    "current_preservation_invocation",
    "install_preservation_adapter",
    "preservation_invocation",
]

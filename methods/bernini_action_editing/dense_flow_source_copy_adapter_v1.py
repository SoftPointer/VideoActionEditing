#!/usr/bin/env python3
"""Source-token preservation branch composed with a frozen motion adapter.

Unlike a generic source-reconstruction corrector, this branch is trained on
same-identity action pairs while the frozen dense-flow motion branch is active.
At every selected transformer block it receives both the current target hidden
state and an explicit source-token carrier.  The carrier is gathered across
sequence-parallel ranks, is detached from the frozen trunk, and never contains
self-generated target RGB or VAE latents at deployment.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
import math
import types
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

import dense_flow_token_adapter_v1 as motion_core


SCHEMA_VERSION = "bernini-dense-flow-source-copy-adapter-v1"
STATE_SCHEMA_VERSION = "bernini-dense-flow-source-copy-adapter-state-v1"
MODULE_NAME = "dense_flow_source_copy_adapter"
HARD_MODULE_NAME = "dense_flow_hard_source_transport"
MODES = (
    "phase_aligned",
    "phase0_broadcast",
    "phase0_attention_8x12",
    "phase0_attention_12x20",
    "phase0_flowwarp_raw",
    "phase0_flowwarp_camera_residual",
)
EXPECTED_TRAINABLE_PARAMETERS = 9_437_184
ATTENTION_TRAINABLE_PARAMETERS = 12_582_912
ATTENTION_MEMORY_SHAPES = {
    "phase0_attention_8x12": (8, 12),
    "phase0_attention_12x20": (12, 20),
}
FLOW_WARP_FEATURE_OFFSETS = {
    "phase0_flowwarp_raw": 2,
    "phase0_flowwarp_camera_residual": 6,
}
SPATIAL_MODES = frozenset(ATTENTION_MEMORY_SHAPES) | frozenset(
    FLOW_WARP_FEATURE_OFFSETS
)


def expected_trainable_parameters(mode: str, *, block_count: Optional[int] = None) -> int:
    if mode not in MODES:
        raise SourceCopyAdapterError("source-copy parameter-count mode differs")
    full_count = len(motion_core.BLOCK_INDICES)
    count = full_count if block_count is None else int(block_count)
    if count <= 0 or count > motion_core.EXPECTED_BLOCK_COUNT:
        raise SourceCopyAdapterError("source-copy block count differs")
    full_parameters = (
        ATTENTION_TRAINABLE_PARAMETERS
        if mode in ATTENTION_MEMORY_SHAPES
        else EXPECTED_TRAINABLE_PARAMETERS
    )
    if full_parameters % full_count:
        raise SourceCopyAdapterError("source-copy per-block parameter count differs")
    return full_parameters // full_count * count


class SourceCopyAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceCopyInvocation:
    activity: torch.Tensor = field(repr=False, compare=False)
    mode: str = "phase_aligned"
    spatial_shape: Optional[tuple[int, int]] = None
    motion_features: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    correspondence_labels: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False
    )
    correspondence_losses: Optional[list[torch.Tensor]] = field(
        default=None, repr=False, compare=False
    )
    max_correspondence_queries: int = 128

    def validate(self) -> None:
        if (
            not isinstance(self.activity, torch.Tensor)
            or self.activity.dtype != torch.bool
            or self.activity.ndim != 3
            or int(self.activity.shape[0]) != 1
            or int(self.activity.shape[2]) != 1
            or int(self.activity.shape[1]) % 2
        ):
            raise SourceCopyAdapterError("activity must be bool [1,2N,1]")
        if self.mode not in MODES:
            raise SourceCopyAdapterError("source-copy mode differs")
        target_tokens = int(self.activity.shape[1]) // 2
        if target_tokens % motion_core.LATENT_PHASES:
            raise SourceCopyAdapterError(
                "source-copy target tokens do not contain 21 phases"
            )
        if self.mode in SPATIAL_MODES:
            if (
                self.spatial_shape is None
                or len(self.spatial_shape) != 2
                or min(map(int, self.spatial_shape)) <= 0
                or math.prod(map(int, self.spatial_shape))
                != target_tokens // motion_core.LATENT_PHASES
            ):
                raise SourceCopyAdapterError(
                    "spatial source-copy requires the exact patch spatial shape"
                )
        if self.mode in FLOW_WARP_FEATURE_OFFSETS:
            if (
                not isinstance(self.motion_features, torch.Tensor)
                or not self.motion_features.is_floating_point()
                or self.motion_features.ndim != 3
                or tuple(self.motion_features.shape[:2])
                != tuple(self.activity.shape[:2])
                or int(self.motion_features.shape[2]) != motion_core.FEATURE_WIDTH
                or not bool(torch.isfinite(self.motion_features).all().item())
            ):
                raise SourceCopyAdapterError(
                    "flow-warp source-copy requires finite global motion features"
                )
        if self.correspondence_labels is not None:
            memory_shape = ATTENTION_MEMORY_SHAPES.get(self.mode)
            memory_tokens = math.prod(memory_shape) if memory_shape else 0
            labels = self.correspondence_labels
            if (
                memory_shape is None
                or not isinstance(labels, torch.Tensor)
                or labels.dtype != torch.long
                or tuple(labels.shape) != tuple(self.activity.shape[:2])
                or bool((labels < -1).any().item())
                or bool((labels >= memory_tokens).any().item())
                or self.correspondence_losses is None
                or not isinstance(self.correspondence_losses, list)
                or int(self.max_correspondence_queries) <= 0
            ):
                raise SourceCopyAdapterError(
                    "source-attention correspondence contract differs"
                )
            target_tokens = int(labels.shape[1]) // 2
            if bool((labels[:, :target_tokens] != -1).any().item()):
                raise SourceCopyAdapterError(
                    "source tokens cannot carry correspondence labels"
                )
        elif self.correspondence_losses is not None:
            raise SourceCopyAdapterError(
                "correspondence collector requires correspondence labels"
            )


def source_attention_correspondence_labels(
    features: torch.Tensor,
    activity: torch.Tensor,
    *,
    spatial_shape: tuple[int, int],
    memory_shape: tuple[int, int],
) -> torch.Tensor:
    """Map motion-active target tokens to phase-zero source memory cells.

    ``features`` must come from the *same-appearance target* RAFT bundle, not
    the cross-appearance action donor.  Channels 2:4 encode cumulative raw
    backward flow, so a target token can be traced to its source phase-zero
    coordinate.  Only the top motion quartile per phase is labeled; this keeps
    the explicit correspondence objective focused on actors/objects instead
    of letting static background dominate it.
    """

    if (
        not isinstance(features, torch.Tensor)
        or not features.is_floating_point()
        or features.ndim != 3
        or int(features.shape[0]) != 1
        or int(features.shape[2]) != motion_core.FEATURE_WIDTH
        or not isinstance(activity, torch.Tensor)
        or activity.dtype != torch.bool
        or tuple(activity.shape) != (*features.shape[:2], 1)
    ):
        raise SourceCopyAdapterError(
            "correspondence features/activity geometry differs"
        )
    height, width = map(int, spatial_shape)
    memory_height, memory_width = map(int, memory_shape)
    if min(height, width, memory_height, memory_width) <= 0:
        raise SourceCopyAdapterError("correspondence spatial geometry differs")
    target_tokens = motion_core.LATENT_PHASES * height * width
    if int(features.shape[1]) != 2 * target_tokens:
        raise SourceCopyAdapterError("correspondence token count differs")
    target = features[:, target_tokens:].float().reshape(
        1, motion_core.LATENT_PHASES, height, width, motion_core.FEATURE_WIDTH
    )
    target_activity = activity[:, target_tokens:, 0].reshape(
        1, motion_core.LATENT_PHASES, height, width
    )
    offsets = torch.atanh(target[..., 2:4].clamp(-0.999, 0.999)).mul(4.0)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=features.device, dtype=torch.float32),
        torch.arange(width, device=features.device, dtype=torch.float32),
        indexing="ij",
    )
    source_x = xx.view(1, 1, height, width) + offsets[..., 0]
    source_y = yy.view(1, 1, height, width) + offsets[..., 1]
    in_bounds = (
        source_x.ge(0)
        & source_x.le(width - 1)
        & source_y.ge(0)
        & source_y.le(height - 1)
    )
    motion_score = target[..., 8:10].amax(dim=-1)
    selected = torch.zeros_like(target_activity)
    for phase in range(1, motion_core.LATENT_PHASES):
        valid = target_activity[:, phase] & in_bounds[:, phase]
        values = motion_score[:, phase][valid]
        if int(values.numel()) == 0:
            continue
        threshold = torch.quantile(values.float(), 0.75)
        selected[:, phase] = valid & motion_score[:, phase].ge(threshold)
    memory_x = torch.floor(
        (source_x + 0.5) * float(memory_width) / float(width)
    ).long().clamp(0, memory_width - 1)
    memory_y = torch.floor(
        (source_y + 0.5) * float(memory_height) / float(height)
    ).long().clamp(0, memory_height - 1)
    target_labels = memory_y * memory_width + memory_x
    target_labels = torch.where(
        selected, target_labels, torch.full_like(target_labels, -1)
    ).reshape(1, target_tokens)
    source_labels = torch.full_like(target_labels, -1)
    labels = torch.cat((source_labels, target_labels), dim=1).contiguous()
    if not bool((labels >= 0).any().item()):
        raise SourceCopyAdapterError("correspondence supervision selected no tokens")
    return labels


_CURRENT: contextvars.ContextVar[Optional[SourceCopyInvocation]] = (
    contextvars.ContextVar("bernini_source_copy_invocation", default=None)
)
_DENOISE_WEIGHT: contextvars.ContextVar[float] = contextvars.ContextVar(
    "bernini_source_copy_denoise_weight", default=1.0
)


@contextlib.contextmanager
def source_copy_invocation(
    invocation: SourceCopyInvocation,
) -> Iterator[SourceCopyInvocation]:
    if not isinstance(invocation, SourceCopyInvocation):
        raise SourceCopyAdapterError("source-copy context received the wrong type")
    invocation.validate()
    if _CURRENT.get() is not None:
        raise SourceCopyAdapterError("nested source-copy invocations are forbidden")
    token = _CURRENT.set(invocation)
    try:
        yield invocation
    finally:
        _CURRENT.reset(token)


def current_source_copy_invocation() -> Optional[SourceCopyInvocation]:
    return _CURRENT.get()


@contextlib.contextmanager
def source_copy_denoise_weight(weight: float) -> Iterator[float]:
    """Scale source-copy only for the current native denoising forward.

    The outer :func:`source_copy_invocation` still authenticates the carrier.
    This inner route lets inference delay source-state transport until pose and
    global motion have been established by the frozen motion branch.  A zero
    weight is an exact source-copy no-op; the default remains one for backward
    compatibility and for ordinary training.
    """

    value = float(weight)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise SourceCopyAdapterError(
            "source-copy denoise weight must be finite in [0,1]"
        )
    token = _DENOISE_WEIGHT.set(value)
    try:
        yield value
    finally:
        _DENOISE_WEIGHT.reset(token)


def current_source_copy_denoise_weight() -> float:
    return float(_DENOISE_WEIGHT.get())


def _global_hidden(invocation: SourceCopyInvocation, hidden: torch.Tensor) -> torch.Tensor:
    global_tokens = int(invocation.activity.shape[1])
    if int(hidden.shape[1]) == global_tokens:
        return hidden.detach()
    try:
        import torch.distributed as dist
    except ImportError as error:  # pragma: no cover
        raise SourceCopyAdapterError(
            "source-copy SP gather requires torch.distributed"
        ) from error
    if not dist.is_initialized() or dist.get_world_size() != 4:
        raise SourceCopyAdapterError(
            "rank-local source-copy hidden requires an SP4 process group"
        )
    gathered = [torch.empty_like(hidden) for _ in range(4)]
    dist.all_gather(gathered, hidden.detach().contiguous())
    value = torch.cat(gathered, dim=1)
    if int(value.shape[1]) < global_tokens:
        raise SourceCopyAdapterError("gathered hidden is shorter than global tokens")
    return value[:, :global_tokens].contiguous()


def _global_carrier(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> torch.Tensor:
    value = _global_hidden(invocation, hidden)
    target_tokens = int(invocation.activity.shape[1]) // 2
    source = value[:, :target_tokens]
    if invocation.mode == "phase_aligned":
        target_carrier = source
    else:
        spatial_tokens = target_tokens // motion_core.LATENT_PHASES
        phase0 = source[:, :spatial_tokens]
        target_carrier = phase0.repeat(1, motion_core.LATENT_PHASES, 1)
    return torch.cat((torch.zeros_like(source), target_carrier), dim=1).contiguous()


def _global_phase0_source(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> torch.Tensor:
    value = _global_hidden(invocation, hidden)
    target_tokens = int(invocation.activity.shape[1]) // 2
    spatial_tokens = target_tokens // motion_core.LATENT_PHASES
    return value[:, :spatial_tokens].contiguous()


def _global_flow_warped_carrier(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> torch.Tensor:
    """Retrieve phase-zero source hidden states at flow-mapped coordinates.

    Dense-flow feature channels 2:4 and 6:8 contain bounded cumulative raw
    and camera-residual backward flow respectively.  A target patch at
    ``(x,y,t)`` therefore samples source phase zero at
    ``(x,y) + cumulative_backward(t)``.  The deterministic spatial transport
    supplies correspondence; the trainable residual only learns how to use
    the retrieved full-width transformer state.
    """

    if invocation.mode not in FLOW_WARP_FEATURE_OFFSETS:
        raise SourceCopyAdapterError("flow-warp source-copy mode differs")
    assert invocation.spatial_shape is not None
    assert invocation.motion_features is not None
    height, width = map(int, invocation.spatial_shape)
    target_tokens = int(invocation.activity.shape[1]) // 2
    spatial_tokens = target_tokens // motion_core.LATENT_PHASES
    source_phase0 = _global_phase0_source(invocation, hidden).float()
    if int(source_phase0.shape[1]) != spatial_tokens:
        raise SourceCopyAdapterError("flow-warp source memory geometry differs")

    feature_start = FLOW_WARP_FEATURE_OFFSETS[invocation.mode]
    target_features = invocation.motion_features[:, target_tokens:].to(
        device=hidden.device, dtype=torch.float32
    )
    bounded = target_features[..., feature_start : feature_start + 2]
    # Features were encoded as tanh(cumulative_flow / 8) before 2x2 patch
    # pooling.  Inverting the bounded representation gives an approximate
    # latent-pixel displacement; divide by two for transformer patch units.
    offsets = torch.atanh(bounded.clamp(-0.999, 0.999)).mul_(4.0)
    offsets = offsets.reshape(
        int(hidden.shape[0]), motion_core.LATENT_PHASES, height, width, 2
    )
    yy, xx = torch.meshgrid(
        torch.arange(height, device=hidden.device, dtype=torch.float32),
        torch.arange(width, device=hidden.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xx.view(1, 1, height, width) + offsets[..., 0]
    sample_y = yy.view(1, 1, height, width) + offsets[..., 1]
    grid = torch.stack(
        (
            2.0 * sample_x / max(width - 1, 1) - 1.0,
            2.0 * sample_y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    ).reshape(-1, height, width, 2)
    source_image = source_phase0.reshape(
        int(hidden.shape[0]), height, width, int(hidden.shape[2])
    ).permute(0, 3, 1, 2)
    source_image = source_image[:, None].expand(
        -1, motion_core.LATENT_PHASES, -1, -1, -1
    ).reshape(-1, int(hidden.shape[2]), height, width)
    transported = F.grid_sample(
        source_image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    target_carrier = transported.reshape(
        int(hidden.shape[0]),
        motion_core.LATENT_PHASES,
        int(hidden.shape[2]),
        height,
        width,
    ).permute(0, 1, 3, 4, 2).reshape(
        int(hidden.shape[0]), target_tokens, int(hidden.shape[2])
    )
    source = torch.zeros_like(target_carrier)
    return torch.cat((source, target_carrier), dim=1).contiguous()


def _local_flow_warped_source_copy(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    carrier = _global_flow_warped_carrier(invocation, hidden).to(
        device=hidden.device, dtype=hidden.dtype
    )
    activity = invocation.activity.to(device=hidden.device)
    if int(carrier.shape[1]) != int(hidden.shape[1]):
        try:
            from bernini.parallel import (
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )
        except ImportError as error:
            raise SourceCopyAdapterError(
                "global flow-warp carrier requires Bernini SP helpers"
            ) from error
        carrier = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(carrier, dim=1), dim=1
        )
        activity = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(activity, dim=1), dim=1
        )
    if tuple(carrier.shape[:2]) != tuple(hidden.shape[:2]):
        raise SourceCopyAdapterError(
            "rank-local flow-warp carrier differs from hidden states"
        )
    return carrier, activity.bool()


def _local_activity(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> torch.Tensor:
    activity = invocation.activity.to(device=hidden.device)
    if int(activity.shape[1]) != int(hidden.shape[1]):
        try:
            from bernini.parallel import (
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )
        except ImportError as error:
            raise SourceCopyAdapterError(
                "global source-copy activity requires Bernini SP helpers"
            ) from error
        activity = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(activity, dim=1), dim=1
        )
    if tuple(activity.shape[:2]) != tuple(hidden.shape[:2]):
        raise SourceCopyAdapterError(
            "rank-local source-copy activity differs from hidden states"
        )
    return activity.bool()


def _local_correspondence_labels(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> Optional[torch.Tensor]:
    labels = invocation.correspondence_labels
    if labels is None:
        return None
    labels = labels.to(device=hidden.device)
    if int(labels.shape[1]) != int(hidden.shape[1]):
        try:
            from bernini.parallel import (
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )
        except ImportError as error:
            raise SourceCopyAdapterError(
                "global correspondence labels require Bernini SP helpers"
            ) from error
        # Bernini pads with zero.  Shift by one before padding/slicing so
        # padded entries return to the required ignore label -1.
        labels = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(labels + 1, dim=1), dim=1
        ) - 1
    if tuple(labels.shape) != tuple(hidden.shape[:2]):
        raise SourceCopyAdapterError(
            "rank-local correspondence labels differ from hidden states"
        )
    return labels.long()


def _local_source_copy(
    invocation: SourceCopyInvocation, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    carrier = _global_carrier(invocation, hidden).to(device=hidden.device)
    activity = invocation.activity.to(device=hidden.device)
    if int(carrier.shape[1]) != int(hidden.shape[1]):
        try:
            from bernini.parallel import (
                padding_tensor_for_seqeunce_parallel,
                slice_input_tensor,
            )
        except ImportError as error:
            raise SourceCopyAdapterError(
                "global source-copy carrier requires Bernini SP helpers"
            ) from error
        carrier = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(carrier, dim=1), dim=1
        )
        activity = slice_input_tensor(
            padding_tensor_for_seqeunce_parallel(activity, dim=1), dim=1
        )
    if tuple(carrier.shape[:2]) != tuple(hidden.shape[:2]):
        raise SourceCopyAdapterError(
            "rank-local source-copy carrier differs from hidden states"
        )
    return carrier, activity.bool()


class SourceCopyResidualBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_width: int = motion_core.HIDDEN_WIDTH,
        bottleneck_width: int = motion_core.BOTTLENECK_WIDTH,
    ) -> None:
        super().__init__()
        self.hidden_width = int(hidden_width)
        self.bottleneck_width = int(bottleneck_width)
        self.norm_target = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.norm_source = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.target_down = nn.Linear(
            self.hidden_width, self.bottleneck_width, bias=False
        )
        self.source_down = nn.Linear(
            self.hidden_width, self.bottleneck_width, bias=False
        )
        self.output = nn.Linear(
            self.bottleneck_width, self.hidden_width, bias=False
        )
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_carrier: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or source_carrier.shape != hidden_states.shape
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden_states.shape[:2], 1)
        ):
            raise SourceCopyAdapterError("source-copy tensor geometry differs")
        target = self.norm_target(hidden_states.float())
        source = self.norm_source(source_carrier.float())
        delta = self.output(
            F.silu(self.target_down(target) + self.source_down(source))
        )
        delta = delta.mul(current_source_copy_denoise_weight())
        delta = torch.where(activity, delta, torch.zeros_like(delta))
        return (hidden_states.float() + delta).to(hidden_states.dtype)

    def is_zero_effect(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight.detach()).item() == 0)


class SourceAttentionResidualBlock(nn.Module):
    """Query-dependent retrieval from a spatial source phase-zero memory.

    The source is retained as 96 or 240 ordered memory tokens rather than a
    pooled identity statistic.  Every live target token forms its own query;
    this lets an edited pose retrieve subject state from a different source
    location.  PyTorch SDPA keeps the query-by-memory matrix off the persistent
    activation path on supported CUDA kernels.
    """

    def __init__(
        self,
        *,
        mode: str,
        hidden_width: int = motion_core.HIDDEN_WIDTH,
        bottleneck_width: int = motion_core.BOTTLENECK_WIDTH,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if mode not in ATTENTION_MEMORY_SHAPES:
            raise SourceCopyAdapterError("source-attention mode differs")
        if bottleneck_width % heads:
            raise SourceCopyAdapterError("source-attention head width differs")
        self.mode = mode
        self.memory_shape = ATTENTION_MEMORY_SHAPES[mode]
        self.hidden_width = int(hidden_width)
        self.bottleneck_width = int(bottleneck_width)
        self.heads = int(heads)
        self.norm_target = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.norm_source = nn.LayerNorm(self.hidden_width, elementwise_affine=False)
        self.query = nn.Linear(self.hidden_width, self.bottleneck_width, bias=False)
        self.key = nn.Linear(self.hidden_width, self.bottleneck_width, bias=False)
        self.value = nn.Linear(self.hidden_width, self.bottleneck_width, bias=False)
        self.output = nn.Linear(self.bottleneck_width, self.hidden_width, bias=False)
        nn.init.zeros_(self.output.weight)

    def _memory(
        self, source_phase0: torch.Tensor, spatial_shape: tuple[int, int]
    ) -> torch.Tensor:
        height, width = map(int, spatial_shape)
        if (
            source_phase0.ndim != 3
            or int(source_phase0.shape[1]) != height * width
            or int(source_phase0.shape[2]) != self.hidden_width
        ):
            raise SourceCopyAdapterError("source-attention memory geometry differs")
        source = source_phase0.float().reshape(
            int(source_phase0.shape[0]), height, width, self.hidden_width
        ).permute(0, 3, 1, 2)
        memory = F.adaptive_avg_pool2d(source, self.memory_shape)
        return memory.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_phase0: torch.Tensor,
        activity: torch.Tensor,
        spatial_shape: tuple[int, int],
        correspondence_labels: Optional[torch.Tensor] = None,
        correspondence_losses: Optional[list[torch.Tensor]] = None,
        max_correspondence_queries: int = 128,
    ) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden_states.shape[:2], 1)
        ):
            raise SourceCopyAdapterError("source-attention tensor geometry differs")
        target = self.norm_target(hidden_states.float())
        memory = self.norm_source(self._memory(source_phase0, spatial_shape))
        batch, tokens, _ = target.shape
        memory_tokens = int(memory.shape[1])
        head_width = self.bottleneck_width // self.heads
        query = self.query(target).reshape(
            batch, tokens, self.heads, head_width
        ).transpose(1, 2)
        key = self.key(memory).reshape(
            batch, memory_tokens, self.heads, head_width
        ).transpose(1, 2)
        value = self.value(memory).reshape(
            batch, memory_tokens, self.heads, head_width
        ).transpose(1, 2)
        if correspondence_labels is not None:
            if (
                correspondence_losses is None
                or batch != 1
                or tuple(correspondence_labels.shape) != (batch, tokens)
                or correspondence_labels.dtype != torch.long
                or int(max_correspondence_queries) <= 0
            ):
                raise SourceCopyAdapterError(
                    "source-attention correspondence runtime differs"
                )
            selected = torch.nonzero(
                correspondence_labels[0].ge(0) & activity[0, :, 0],
                as_tuple=False,
            ).flatten()
            has_real_correspondence = int(selected.numel()) > 0
            if has_real_correspondence:
                # Execute the exact same query/key matmul geometry on all SP
                # ranks.  Source-half ranks have no real labels, while the two
                # target-half ranks do; letting the former skip this work can
                # make the ranks enter distinct NCCL process groups in a
                # different order and deadlock RCCL.  Uniform resampling also
                # pads sparse target selections to the fixed query count.
                offsets = torch.linspace(
                    0,
                    int(selected.numel()) - 1,
                    steps=int(max_correspondence_queries),
                    device=selected.device,
                ).round().long()
                selected = selected.index_select(0, offsets)
                labels = correspondence_labels[0].index_select(0, selected)
            else:
                if tokens < int(max_correspondence_queries):
                    raise SourceCopyAdapterError(
                        "source-attention dummy correspondence lacks local tokens"
                    )
                selected = torch.arange(
                    int(max_correspondence_queries), device=query.device
                )
                labels = torch.zeros(
                    int(max_correspondence_queries),
                    dtype=torch.long,
                    device=query.device,
                )
            selected_query = query.index_select(2, selected)
            logits = torch.matmul(
                selected_query.float(), key.float().transpose(-1, -2)
            ).mul(1.0 / math.sqrt(float(head_width)))
            labels = labels.unsqueeze(0).expand(self.heads, -1).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, memory_tokens), labels)
            if not has_real_correspondence:
                loss = loss.mul(0.0)
            correspondence_losses.append(loss)
        retrieved = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        ).transpose(1, 2).reshape(batch, tokens, self.bottleneck_width)
        delta = self.output(retrieved).mul(current_source_copy_denoise_weight())
        delta = torch.where(activity, delta, torch.zeros_like(delta))
        return (hidden_states.float() + delta).to(hidden_states.dtype)

    def is_zero_effect(self) -> bool:
        return bool(torch.count_nonzero(self.output.weight.detach()).item() == 0)


class HardSourceTransportBlock(nn.Module):
    """Parameter-free source-state injection after a transformer block.

    This is deliberately not a small learned preservation statistic.  It
    transports the full-width source hidden state to target coordinates and
    performs an exact convex interpolation on active target tokens.  The
    denoising schedule is controlled by :func:`source_copy_denoise_weight`, so
    motion can first establish a new pose before late source appearance is
    reintroduced.
    """

    def __init__(self, *, scale: float) -> None:
        super().__init__()
        value = float(scale)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise SourceCopyAdapterError(
                "hard source-transport scale must be finite in (0,1]"
            )
        self.scale = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_carrier: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden_states.ndim != 3
            or source_carrier.shape != hidden_states.shape
            or activity.dtype != torch.bool
            or tuple(activity.shape) != (*hidden_states.shape[:2], 1)
        ):
            raise SourceCopyAdapterError(
                "hard source-transport tensor geometry differs"
            )
        weight = self.scale * current_source_copy_denoise_weight()
        if weight == 0.0:
            return hidden_states
        transported = hidden_states.float().lerp(source_carrier.float(), weight)
        return torch.where(
            activity, transported.to(hidden_states.dtype), hidden_states
        )


@dataclass
class HardSourceTransportPatchHandle:
    transformer: Any
    mode: str
    scale: float
    block_indices: tuple[int, ...]
    adapters: tuple[HardSourceTransportBlock, ...]
    original_forwards: tuple[Any, ...] = field(repr=False)
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        for index, adapter, original in zip(
            self.block_indices, self.adapters, self.original_forwards
        ):
            block = self.transformer.blocks[index]
            block.forward = original
            if getattr(block, HARD_MODULE_NAME, None) is adapter:
                delattr(block, HARD_MODULE_NAME)
        self.restored = True


@dataclass
class SourceCopyPatchHandle:
    transformer: Any
    block_indices: tuple[int, ...]
    adapters: tuple[nn.Module, ...]
    original_forwards: tuple[Any, ...] = field(repr=False)
    restored: bool = False

    def trainable_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        if self.restored:
            raise SourceCopyAdapterError("source-copy patch is already restored")
        rows: list[tuple[str, nn.Parameter]] = []
        for index, adapter in zip(self.block_indices, self.adapters):
            for name, parameter in adapter.named_parameters():
                rows.append((f"blocks.{index}.{MODULE_NAME}.{name}", parameter))
        if not rows or len({id(parameter) for _, parameter in rows}) != len(rows):
            raise SourceCopyAdapterError("source-copy parameter closure differs")
        return tuple(rows)

    def base_is_frozen(self) -> bool:
        trainable = {id(parameter) for _, parameter in self.trainable_named_parameters()}
        return all(
            id(parameter) in trainable or not parameter.requires_grad
            for parameter in self.transformer.parameters()
        )

    def zero_effect(self) -> bool:
        return all(adapter.is_zero_effect() for adapter in self.adapters)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().float().cpu().contiguous()
            for name, parameter in self.trainable_named_parameters()
        }

    def load_state_dict_strict(
        self, state: Mapping[str, torch.Tensor], *, output_scale: float = 1.0
    ) -> None:
        expected = dict(self.trainable_named_parameters())
        if set(state) != set(expected):
            raise SourceCopyAdapterError("source-copy state-key closure differs")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if value.shape != parameter.shape or not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise SourceCopyAdapterError(
                        f"source-copy state tensor differs: {name}"
                    )
                if name.endswith("output.weight"):
                    value = value.float().mul(float(output_scale))
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def install_source_copy_adapter(
    model: Any,
    *,
    mode: str = "phase_aligned",
    block_indices: Sequence[int] = motion_core.BLOCK_INDICES,
    hidden_width: int = motion_core.HIDDEN_WIDTH,
    bottleneck_width: int = motion_core.BOTTLENECK_WIDTH,
) -> SourceCopyPatchHandle:
    if mode not in MODES:
        raise SourceCopyAdapterError("source-copy install mode differs")
    transformer = motion_core._resolve_transformer(model)
    transformer.requires_grad_(False)
    indices = tuple(int(item) for item in block_indices)
    if indices != tuple(sorted(set(indices))) or any(
        item < 0 or item >= motion_core.EXPECTED_BLOCK_COUNT for item in indices
    ):
        raise SourceCopyAdapterError(
            "source-copy block indices must be sorted unique in [0,29]"
        )
    adapters: list[nn.Module] = []
    originals: list[Any] = []
    installed: list[int] = []
    try:
        for index in indices:
            block = transformer.blocks[index]
            if hasattr(block, MODULE_NAME):
                raise SourceCopyAdapterError(
                    f"block {index} already has a source-copy adapter"
                )
            if mode in ATTENTION_MEMORY_SHAPES:
                adapter = SourceAttentionResidualBlock(
                    mode=mode,
                    hidden_width=hidden_width,
                    bottleneck_width=bottleneck_width,
                )
            else:
                adapter = SourceCopyResidualBlock(
                    hidden_width=hidden_width, bottleneck_width=bottleneck_width
                )
            block.add_module(MODULE_NAME, adapter)
            original = block.forward

            def wrapped_forward(
                self: Any,
                *args: Any,
                _original: Any = original,
                _adapter: nn.Module = adapter,
                **kwargs: Any,
            ) -> torch.Tensor:
                hidden = _original(*args, **kwargs)
                invocation = current_source_copy_invocation()
                if invocation is None:
                    return hidden
                if invocation.mode != mode:
                    raise SourceCopyAdapterError(
                        "source-copy invocation/install mode differs"
                    )
                if mode in ATTENTION_MEMORY_SHAPES:
                    source = _global_phase0_source(invocation, hidden).to(
                        device=hidden.device
                    )
                    activity = _local_activity(invocation, hidden)
                    correspondence_labels = _local_correspondence_labels(
                        invocation, hidden
                    )
                    assert invocation.spatial_shape is not None
                    return _adapter(
                        hidden,
                        source,
                        activity,
                        invocation.spatial_shape,
                        correspondence_labels,
                        invocation.correspondence_losses,
                        invocation.max_correspondence_queries,
                    )
                if mode in FLOW_WARP_FEATURE_OFFSETS:
                    carrier, activity = _local_flow_warped_source_copy(
                        invocation, hidden
                    )
                    return _adapter(hidden, carrier, activity)
                carrier, activity = _local_source_copy(invocation, hidden)
                return _adapter(hidden, carrier, activity)

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
    handle = SourceCopyPatchHandle(
        transformer=transformer,
        block_indices=indices,
        adapters=tuple(adapters),
        original_forwards=tuple(originals),
    )
    if not handle.base_is_frozen() or not handle.zero_effect():
        raise SourceCopyAdapterError("source-copy freeze/zero-init closure differs")
    return handle


def install_hard_source_transport(
    model: Any,
    *,
    mode: str,
    scale: float,
    block_indices: Sequence[int],
) -> HardSourceTransportPatchHandle:
    """Install deterministic full-width source transport for inference.

    ``phase0_flowwarp_*`` is the primary action-editing route: the target
    action donor supplies only correspondence offsets, while appearance comes
    from the source phase-zero hidden field.  ``phase0_broadcast`` is retained
    as a no-correspondence control.
    """

    allowed = ("phase0_broadcast", *FLOW_WARP_FEATURE_OFFSETS)
    if mode not in allowed:
        raise SourceCopyAdapterError("hard source-transport mode differs")
    transformer = motion_core._resolve_transformer(model)
    indices = tuple(int(item) for item in block_indices)
    if (
        not indices
        or indices != tuple(sorted(set(indices)))
        or any(item < 0 or item >= motion_core.EXPECTED_BLOCK_COUNT for item in indices)
    ):
        raise SourceCopyAdapterError(
            "hard source-transport block indices must be sorted unique in [0,29]"
        )
    adapters: list[HardSourceTransportBlock] = []
    originals: list[Any] = []
    installed: list[int] = []
    try:
        for index in indices:
            block = transformer.blocks[index]
            if hasattr(block, HARD_MODULE_NAME):
                raise SourceCopyAdapterError(
                    f"block {index} already has hard source transport"
                )
            adapter = HardSourceTransportBlock(scale=scale)
            block.add_module(HARD_MODULE_NAME, adapter)
            original = block.forward

            def wrapped_forward(
                self: Any,
                *args: Any,
                _original: Any = original,
                _adapter: HardSourceTransportBlock = adapter,
                **kwargs: Any,
            ) -> torch.Tensor:
                hidden = _original(*args, **kwargs)
                invocation = current_source_copy_invocation()
                if invocation is None:
                    return hidden
                if invocation.mode != mode:
                    raise SourceCopyAdapterError(
                        "hard source-transport invocation/install mode differs"
                    )
                if mode in FLOW_WARP_FEATURE_OFFSETS:
                    carrier, local_activity = _local_flow_warped_source_copy(
                        invocation, hidden
                    )
                else:
                    carrier, local_activity = _local_source_copy(invocation, hidden)
                return _adapter(hidden, carrier, local_activity)

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
            if getattr(block, HARD_MODULE_NAME, None) is adapter:
                delattr(block, HARD_MODULE_NAME)
        raise
    return HardSourceTransportPatchHandle(
        transformer=transformer,
        mode=mode,
        scale=float(scale),
        block_indices=indices,
        adapters=tuple(adapters),
        original_forwards=tuple(originals),
    )


__all__ = [
    "ATTENTION_MEMORY_SHAPES",
    "ATTENTION_TRAINABLE_PARAMETERS",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "FLOW_WARP_FEATURE_OFFSETS",
    "HARD_MODULE_NAME",
    "MODES",
    "SPATIAL_MODES",
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "SourceCopyAdapterError",
    "SourceCopyInvocation",
    "SourceCopyPatchHandle",
    "SourceCopyResidualBlock",
    "HardSourceTransportBlock",
    "HardSourceTransportPatchHandle",
    "SourceAttentionResidualBlock",
    "current_source_copy_invocation",
    "expected_trainable_parameters",
    "install_source_copy_adapter",
    "install_hard_source_transport",
    "source_attention_correspondence_labels",
    "source_copy_invocation",
]

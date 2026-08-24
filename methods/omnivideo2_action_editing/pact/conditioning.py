"""Source-conditioning controls for selective motion editing."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .masks import validate_video_mask


def _validate_source_latent(source_latent: Tensor) -> Tensor:
    if not isinstance(source_latent, Tensor):
        raise TypeError("source_latent must be a torch.Tensor")
    if source_latent.ndim != 5:
        raise ValueError(
            "source_latent must have shape [B, C, T, H, W], "
            f"got {tuple(source_latent.shape)}"
        )
    if not source_latent.is_floating_point():
        raise TypeError("source_latent must have a floating dtype")
    if min(source_latent.shape) <= 0:
        raise ValueError("source_latent dimensions must all be non-zero")
    return source_latent


def erase_source_motion(
    source_latent: Tensor,
    source_actor_mask: Tensor,
    *,
    mode: str = "zero",
    keep_first_frame: bool = True,
) -> Tensor:
    """Erase only the selected actor's source motion conditioning.

    ``zero`` removes selected latent values. ``temporal_mean`` replaces them
    with the time-mean latent at the same spatial location, suppressing motion
    while retaining a low-frequency appearance cue. Soft masks blend smoothly.

    When ``keep_first_frame`` is true, frame zero is concatenated directly from
    the input and is therefore bitwise unchanged (not merely numerically close).
    """

    _validate_source_latent(source_latent)
    validate_video_mask(
        source_actor_mask,
        name="source_actor_mask",
        batch_size=source_latent.shape[0],
        frames=source_latent.shape[2],
        height=source_latent.shape[3],
        width=source_latent.shape[4],
    )
    if source_actor_mask.device != source_latent.device:
        raise ValueError("source_actor_mask and source_latent must share a device")
    if mode not in {"zero", "temporal_mean"}:
        raise ValueError("mode must be 'zero' or 'temporal_mean'")
    if not isinstance(keep_first_frame, bool):
        raise TypeError("keep_first_frame must be bool")

    mask = source_actor_mask.to(dtype=source_latent.dtype)
    if mode == "zero":
        replacement = torch.zeros_like(source_latent)
    else:
        replacement = source_latent.mean(dim=2, keepdim=True).expand_as(source_latent)

    if not keep_first_frame:
        return source_latent * (1.0 - mask) + replacement * mask
    if source_latent.shape[2] == 1:
        return source_latent
    tail = source_latent[:, :, 1:]
    tail_mask = mask[:, :, 1:]
    tail_replacement = replacement[:, :, 1:]
    erased_tail = tail * (1.0 - tail_mask) + tail_replacement * tail_mask
    return torch.cat((source_latent[:, :, :1], erased_tail), dim=2)


@dataclass(frozen=True)
class SourceLatentBudgetMetadata:
    """Accounting record returned by :func:`budget_source_latent`."""

    original_shape: tuple[int, int, int, int, int]
    output_shape: tuple[int, int, int, int, int]
    visual_patch_size: tuple[int, int, int]
    max_context_len: int
    nonvisual_tokens: int
    available_visual_tokens: int
    original_visual_tokens: int
    output_visual_tokens: int
    compressed: bool

    @property
    def original_total_tokens(self) -> int:
        return self.nonvisual_tokens + self.original_visual_tokens

    @property
    def output_total_tokens(self) -> int:
        return self.nonvisual_tokens + self.output_visual_tokens


def _positive_int(value: int, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _patch_size(value: Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or len(value) != 3:
        raise TypeError("visual_patch_size must be a length-3 integer sequence")
    return tuple(
        _positive_int(item, name=f"visual_patch_size[{index}]")
        for index, item in enumerate(value)
    )


def budget_source_latent(
    source_latent: Tensor,
    *,
    max_context_len: int,
    nonvisual_tokens: int,
    visual_patch_size: Sequence[int] = (1, 4, 4),
    min_temporal_tokens: int = 1,
) -> tuple[Tensor, SourceLatentBudgetMetadata]:
    """Fit source-adapter visual tokens by compressing latent time only.

    Token accounting exactly follows OmniVideo2's no-padding ``Conv3d`` with
    ``kernel_size == stride == visual_patch_size``. Spatial dimensions are
    never resized. Callers that intentionally pad time must do so before this
    function, making that policy explicit. If one temporal patch cannot fit,
    the function fails closed instead of silently dropping source conditioning.
    Under budget, the exact input tensor object is returned.
    """

    _validate_source_latent(source_latent)
    max_context_len = _positive_int(max_context_len, name="max_context_len")
    if not isinstance(nonvisual_tokens, Integral) or isinstance(nonvisual_tokens, bool):
        raise TypeError("nonvisual_tokens must be an integer")
    nonvisual_tokens = int(nonvisual_tokens)
    if nonvisual_tokens < 0:
        raise ValueError("nonvisual_tokens must be non-negative")
    min_temporal_tokens = _positive_int(
        min_temporal_tokens, name="min_temporal_tokens"
    )
    pt, ph, pw = _patch_size(visual_patch_size)

    _, _, frames, height, width = source_latent.shape
    if frames < pt or height < ph or width < pw:
        raise ValueError(
            "source latent dimensions must each be at least the corresponding "
            "visual adapter patch size"
        )
    spatial_tokens = (height // ph) * (width // pw)
    original_temporal_tokens = frames // pt
    original_visual_tokens = original_temporal_tokens * spatial_tokens
    available_visual_tokens = max_context_len - nonvisual_tokens
    max_temporal_tokens = available_visual_tokens // spatial_tokens

    if max_temporal_tokens < min_temporal_tokens:
        raise ValueError(
            "source latent cannot fit safely: the context budget cannot hold "
            f"{min_temporal_tokens} temporal visual token(s)"
        )

    if original_visual_tokens <= available_visual_tokens:
        output = source_latent
        output_visual_tokens = original_visual_tokens
        compressed = False
    else:
        output_frames = min(frames, max_temporal_tokens * pt)
        if output_frames <= 0:
            raise ValueError("source latent cannot fit even one latent frame")
        output = F.adaptive_avg_pool3d(
            source_latent, output_size=(output_frames, height, width)
        )
        output_visual_tokens = (output_frames // pt) * spatial_tokens
        compressed = True

    metadata = SourceLatentBudgetMetadata(
        original_shape=tuple(source_latent.shape),
        output_shape=tuple(output.shape),
        visual_patch_size=(pt, ph, pw),
        max_context_len=max_context_len,
        nonvisual_tokens=nonvisual_tokens,
        available_visual_tokens=available_visual_tokens,
        original_visual_tokens=original_visual_tokens,
        output_visual_tokens=output_visual_tokens,
        compressed=compressed,
    )
    if metadata.output_total_tokens > max_context_len:
        raise RuntimeError("internal error: source token budgeting exceeded context")
    return output, metadata

"""Mask utilities for preservation-aware action editing.

All public functions use the latent/video convention ``[B, C, T, H, W]``.
Masks have a singleton channel dimension so that broadcasting over latent
channels is explicit and accidental per-channel masks fail early.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


Radius3D = int | Sequence[int]


def validate_video_mask(
    mask: Tensor,
    *,
    name: str = "mask",
    batch_size: int | None = None,
    frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    require_unit_interval: bool = True,
) -> Tensor:
    """Validate and return a ``[B, 1, T, H, W]`` bool/float mask.

    Returning the input unchanged is intentional: callers can validate a mask
    without losing its dtype, device, autograd history, or storage identity.
    """

    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.ndim != 5:
        raise ValueError(
            f"{name} must have shape [B, 1, T, H, W], got {tuple(mask.shape)}"
        )
    if mask.shape[1] != 1:
        raise ValueError(f"{name} channel dimension must be 1, got {mask.shape[1]}")
    if mask.dtype is not torch.bool and not mask.is_floating_point():
        raise TypeError(f"{name} must have bool or floating dtype, got {mask.dtype}")
    expected = (batch_size, frames, height, width)
    actual = (mask.shape[0], mask.shape[2], mask.shape[3], mask.shape[4])
    labels = ("batch", "time", "height", "width")
    for label, wanted, got in zip(labels, expected, actual):
        if wanted is not None and got != wanted:
            raise ValueError(f"{name} {label} must be {wanted}, got {got}")
    if mask.is_floating_point():
        detached = mask.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError(f"{name} must contain only finite values")
        if require_unit_interval and not bool(
            ((detached >= 0) & (detached <= 1)).all()
        ):
            raise ValueError(f"{name} values must lie in [0, 1]")
    return mask


def _radius3(radius: Radius3D, *, name: str) -> tuple[int, int, int]:
    if isinstance(radius, Integral) and not isinstance(radius, bool):
        values = (int(radius),) * 3
    elif isinstance(radius, Sequence) and len(radius) == 3:
        values = tuple(radius)
        if any(not isinstance(item, Integral) or isinstance(item, bool) for item in values):
            raise TypeError(f"{name} entries must be integers")
        values = tuple(int(item) for item in values)
    else:
        raise TypeError(f"{name} must be an int or a length-3 integer sequence")
    if any(item < 0 for item in values):
        raise ValueError(f"{name} entries must be non-negative, got {values}")
    return values


def _float_mask(mask: Tensor) -> Tensor:
    validate_video_mask(mask)
    return mask if mask.is_floating_point() else mask.to(dtype=torch.get_default_dtype())


def source_target_tube_union(source_mask: Tensor, target_mask: Tensor) -> Tensor:
    """Return the soft union of source- and target-actor tubes.

    For soft masks, ``maximum`` is preferable to addition: overlapping support
    remains in ``[0, 1]`` and gradients flow to the stronger observation.
    """

    validate_video_mask(source_mask, name="source_mask")
    validate_video_mask(
        target_mask,
        name="target_mask",
        batch_size=source_mask.shape[0],
        frames=source_mask.shape[2],
        height=source_mask.shape[3],
        width=source_mask.shape[4],
    )
    if source_mask.device != target_mask.device:
        raise ValueError("source_mask and target_mask must be on the same device")
    if source_mask.dtype != target_mask.dtype:
        dtype = torch.promote_types(source_mask.dtype, target_mask.dtype)
        source_mask = source_mask.to(dtype=dtype)
        target_mask = target_mask.to(dtype=dtype)
    return torch.maximum(source_mask, target_mask)


tube_union = source_target_tube_union


def dilate_mask(mask: Tensor, radius: Radius3D = (0, 1, 1)) -> Tensor:
    """Morphologically dilate a mask with a rectangular 3-D kernel."""

    value = _float_mask(mask)
    rt, rh, rw = _radius3(radius, name="radius")
    if rt == rh == rw == 0:
        return value
    return F.max_pool3d(
        value,
        kernel_size=(2 * rt + 1, 2 * rh + 1, 2 * rw + 1),
        stride=1,
        padding=(rt, rh, rw),
    )


def erode_mask(mask: Tensor, radius: Radius3D = (0, 1, 1)) -> Tensor:
    """Morphologically erode a mask with a rectangular 3-D kernel."""

    value = _float_mask(mask)
    rt, rh, rw = _radius3(radius, name="radius")
    if rt == rh == rw == 0:
        return value
    return 1.0 - F.max_pool3d(
        1.0 - value,
        kernel_size=(2 * rt + 1, 2 * rh + 1, 2 * rw + 1),
        stride=1,
        padding=(rt, rh, rw),
    )


def dilate_and_feather(
    mask: Tensor,
    *,
    dilation_radius: Radius3D = (0, 1, 1),
    feather_radius: Radius3D = (0, 1, 1),
) -> Tensor:
    """Dilate support and add a soft, box-filtered outer feather.

    Existing dilated values are never reduced. Binary-mask cores therefore
    remain exactly one, while genuinely soft input masks remain soft.
    """

    hard = dilate_mask(mask, dilation_radius)
    rt, rh, rw = _radius3(feather_radius, name="feather_radius")
    if rt == rh == rw == 0:
        return hard.clamp(0.0, 1.0)
    blurred = F.avg_pool3d(
        hard,
        kernel_size=(2 * rt + 1, 2 * rh + 1, 2 * rw + 1),
        stride=1,
        padding=(rt, rh, rw),
    )
    return torch.maximum(hard, blurred).clamp(0.0, 1.0)


def boundary_ring(
    mask: Tensor,
    radius: Radius3D = (0, 1, 1),
    *,
    threshold: float = 0.5,
    mode: str = "both",
) -> Tensor:
    """Return an inner, outer, or two-sided binary boundary ring."""

    value = _float_mask(mask)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if mode not in {"inner", "outer", "both"}:
        raise ValueError("mode must be 'inner', 'outer', or 'both'")
    hard = (value >= threshold).to(dtype=value.dtype)
    dilated = dilate_mask(hard, radius)
    eroded = erode_mask(hard, radius)
    if mode == "inner":
        return (hard - eroded).clamp(0.0, 1.0)
    if mode == "outer":
        return (dilated - hard).clamp(0.0, 1.0)
    return (dilated - eroded).clamp(0.0, 1.0)

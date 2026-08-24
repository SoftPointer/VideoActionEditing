"""Area-balanced reconstruction losses for PACT training."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from .masks import boundary_ring, validate_video_mask


def _validate_video(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    return value


def _matching_videos(reference: Tensor, *values: tuple[str, Tensor]) -> None:
    for name, value in values:
        _validate_video(value, name=name)
        if value.shape != reference.shape:
            raise ValueError(f"{name} shape must match prediction")
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"{name} must share prediction dtype and device")


def _elementwise_error(prediction: Tensor, target: Tensor, loss_type: str) -> Tensor:
    if loss_type == "l1":
        return (prediction - target).abs()
    if loss_type in {"l2", "mse"}:
        return (prediction - target).square()
    raise ValueError("loss_type must be 'l1', 'l2', or 'mse'")


def area_normalized_masked_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    loss_type: str = "l1",
    eps: float = 1e-8,
) -> Tensor:
    """Average error per selected element, then average non-empty samples.

    This prevents large actor masks from dominating small actor masks. Empty
    samples contribute neither loss nor denominator; an all-empty batch returns
    a differentiable zero.
    """

    _validate_video(prediction, name="prediction")
    _matching_videos(prediction, ("target", target))
    validate_video_mask(
        mask,
        batch_size=prediction.shape[0],
        frames=prediction.shape[2],
        height=prediction.shape[3],
        width=prediction.shape[4],
    )
    if mask.device != prediction.device:
        raise ValueError("mask and prediction must share a device")
    if eps <= 0:
        raise ValueError("eps must be positive")

    weights = mask.to(dtype=prediction.dtype).expand_as(prediction)
    error = _elementwise_error(prediction, target, loss_type)
    reduce_dims = tuple(range(1, prediction.ndim))
    numerator = (error * weights).sum(dim=reduce_dims)
    area = weights.sum(dim=reduce_dims)
    valid = area > eps
    per_sample = numerator / area.clamp_min(eps)
    valid_weight = valid.to(dtype=prediction.dtype)
    return (per_sample * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)


def edit_preserve_losses(
    prediction: Tensor,
    target: Tensor,
    source: Tensor,
    edit_mask: Tensor,
    *,
    exclude_from_preserve: Tensor | None = None,
    loss_type: str = "l1",
) -> dict[str, Tensor]:
    """Return independently area-normalized edit and preservation losses."""

    _validate_video(prediction, name="prediction")
    _matching_videos(prediction, ("target", target), ("source", source))
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=prediction.shape[0],
        frames=prediction.shape[2],
        height=prediction.shape[3],
        width=prediction.shape[4],
    )
    mask = edit_mask.to(dtype=prediction.dtype)
    preserve_mask = 1.0 - mask
    if exclude_from_preserve is not None:
        validate_video_mask(
            exclude_from_preserve,
            name="exclude_from_preserve",
            batch_size=prediction.shape[0],
            frames=prediction.shape[2],
            height=prediction.shape[3],
            width=prediction.shape[4],
        )
        if exclude_from_preserve.device != prediction.device:
            raise ValueError(
                "exclude_from_preserve and prediction must share a device"
            )
        preserve_mask = preserve_mask * (
            1.0 - exclude_from_preserve.to(dtype=prediction.dtype)
        )
    return {
        "edit": area_normalized_masked_loss(
            prediction, target, mask, loss_type=loss_type
        ),
        "preserve": area_normalized_masked_loss(
            prediction, source, preserve_mask, loss_type=loss_type
        ),
    }


def boundary_consistency_loss(
    prediction: Tensor,
    target: Tensor,
    source: Tensor,
    edit_mask: Tensor,
    *,
    ring_mask: Tensor | None = None,
    ring_radius: int | tuple[int, int, int] = (0, 1, 1),
    loss_type: str = "l1",
) -> Tensor:
    """Match the source/target composite in a ring around the edit tube."""

    _validate_video(prediction, name="prediction")
    _matching_videos(prediction, ("target", target), ("source", source))
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=prediction.shape[0],
        frames=prediction.shape[2],
        height=prediction.shape[3],
        width=prediction.shape[4],
    )
    mask = edit_mask.to(device=prediction.device, dtype=prediction.dtype)
    if ring_mask is None:
        ring_mask = boundary_ring(mask, radius=ring_radius)
    else:
        validate_video_mask(
            ring_mask,
            name="ring_mask",
            batch_size=prediction.shape[0],
            frames=prediction.shape[2],
            height=prediction.shape[3],
            width=prediction.shape[4],
        )
        if ring_mask.device != prediction.device:
            raise ValueError("ring_mask and prediction must share a device")
    composite = source * (1.0 - mask) + target * mask
    return area_normalized_masked_loss(
        prediction, composite, ring_mask, loss_type=loss_type
    )


def outside_temporal_difference_loss(
    prediction: Tensor,
    source: Tensor,
    edit_mask: Tensor,
    *,
    loss_type: str = "l1",
) -> Tensor:
    """Preserve source frame-to-frame changes where both frames are outside.

    Requiring both endpoints to be outside avoids penalizing motion crossing the
    edit boundary. Soft masks produce a soft pairwise confidence product.
    """

    _validate_video(prediction, name="prediction")
    _matching_videos(prediction, ("source", source))
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=prediction.shape[0],
        frames=prediction.shape[2],
        height=prediction.shape[3],
        width=prediction.shape[4],
    )
    if prediction.shape[2] < 2:
        return prediction.sum() * 0.0
    mask = edit_mask.to(device=prediction.device, dtype=prediction.dtype)
    outside_pair = (1.0 - mask[:, :, 1:]) * (1.0 - mask[:, :, :-1])
    prediction_delta = prediction[:, :, 1:] - prediction[:, :, :-1]
    source_delta = source[:, :, 1:] - source[:, :, :-1]
    return area_normalized_masked_loss(
        prediction_delta, source_delta, outside_pair, loss_type=loss_type
    )


def pact_reconstruction_losses(
    prediction: Tensor,
    target: Tensor,
    source: Tensor,
    edit_mask: Tensor,
    *,
    weights: Mapping[str, float] | None = None,
    ring_mask: Tensor | None = None,
    ring_radius: int | tuple[int, int, int] = (0, 1, 1),
    loss_type: str = "l1",
) -> dict[str, Tensor]:
    """Return total PACT loss plus scalar components suitable for logging."""

    configured = {
        "edit": 1.0,
        "preserve": 1.0,
        "boundary": 1.0,
        "temporal_outside": 1.0,
    }
    if weights is not None:
        unknown = set(weights) - set(configured)
        if unknown:
            raise ValueError(f"unknown loss weights: {sorted(unknown)}")
        for name, value in weights.items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"loss weight {name!r} must be finite and non-negative"
                )
            configured[name] = value

    if ring_mask is None:
        ring_mask = boundary_ring(edit_mask, radius=ring_radius)
    components = edit_preserve_losses(
        prediction,
        target,
        source,
        edit_mask,
        exclude_from_preserve=ring_mask,
        loss_type=loss_type,
    )
    components["boundary"] = boundary_consistency_loss(
        prediction,
        target,
        source,
        edit_mask,
        ring_mask=ring_mask,
        ring_radius=ring_radius,
        loss_type=loss_type,
    )
    components["temporal_outside"] = outside_temporal_difference_loss(
        prediction, source, edit_mask, loss_type=loss_type
    )
    total = sum(configured[name] * components[name] for name in configured)
    return {"total": total, **components}


pact_losses = pact_reconstruction_losses

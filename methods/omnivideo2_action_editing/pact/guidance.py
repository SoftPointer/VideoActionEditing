"""Spatially gated guidance and source-noisy preservation anchoring."""

from __future__ import annotations

import torch
from torch import Tensor

from .flow import flow_noisy_latent
from .masks import validate_video_mask


def _validate_prediction(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    return value


def _matching(reference: Tensor, other: Tensor, *, name: str) -> None:
    _validate_prediction(other, name=name)
    if (
        other.shape != reference.shape
        or other.dtype != reference.dtype
        or other.device != reference.device
    ):
        raise ValueError(f"{name} must match shape, dtype, and device")


def _mask_for(reference: Tensor, mask: Tensor) -> Tensor:
    validate_video_mask(
        mask,
        name="edit_mask",
        batch_size=reference.shape[0],
        frames=reference.shape[2],
        height=reference.shape[3],
        width=reference.shape[4],
    )
    if mask.device != reference.device:
        raise ValueError("edit_mask and prediction must share a device")
    return mask.to(dtype=reference.dtype)


def _control_for(
    reference: Tensor,
    value: float | Tensor,
    *,
    name: str,
    unit_interval: bool = False,
) -> Tensor:
    if isinstance(value, Tensor):
        control = value.to(device=reference.device, dtype=reference.dtype)
    else:
        control = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if control.ndim == 1:
        if control.numel() == 1:
            control = control.reshape(())
        elif control.shape[0] == reference.shape[0]:
            control = control.reshape(reference.shape[0], 1, 1, 1, 1)
    try:
        torch.broadcast_shapes(reference.shape, control.shape)
    except RuntimeError as error:
        raise ValueError(
            f"{name} shape {tuple(control.shape)} cannot broadcast to "
            f"{tuple(reference.shape)}"
        ) from error
    detached = control.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError(f"{name} must be finite")
    if unit_interval and not bool(((detached >= 0) & (detached <= 1)).all()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return control


def gate_keep_edit_deltas(
    base_prediction: Tensor,
    keep_delta: Tensor,
    edit_delta: Tensor,
    edit_mask: Tensor,
    *,
    keep_scale: float | Tensor = 1.0,
    edit_scale: float | Tensor = 1.0,
) -> Tensor:
    """Apply keep guidance outside and edit guidance inside a soft tube."""

    _validate_prediction(base_prediction, name="base_prediction")
    _matching(base_prediction, keep_delta, name="keep_delta")
    _matching(base_prediction, edit_delta, name="edit_delta")
    mask = _mask_for(base_prediction, edit_mask)
    keep_scale = _control_for(base_prediction, keep_scale, name="keep_scale")
    edit_scale = _control_for(base_prediction, edit_scale, name="edit_scale")
    return (
        base_prediction
        + (1.0 - mask) * keep_delta * keep_scale
        + mask * edit_delta * edit_scale
    )


def spatially_gated_guidance(
    unconditional_prediction: Tensor,
    keep_prediction: Tensor,
    edit_prediction: Tensor,
    edit_mask: Tensor,
    *,
    keep_scale: float | Tensor = 1.0,
    edit_scale: float | Tensor = 1.0,
) -> Tensor:
    """Compute condition deltas from an unconditional branch and gate them."""

    _validate_prediction(unconditional_prediction, name="unconditional_prediction")
    _matching(unconditional_prediction, keep_prediction, name="keep_prediction")
    _matching(unconditional_prediction, edit_prediction, name="edit_prediction")
    return gate_keep_edit_deltas(
        unconditional_prediction,
        keep_prediction - unconditional_prediction,
        edit_prediction - unconditional_prediction,
        edit_mask,
        keep_scale=keep_scale,
        edit_scale=edit_scale,
    )


def anchor_to_source_noisy(
    current_x_t: Tensor,
    source_x_t: Tensor,
    edit_mask: Tensor,
    *,
    strength: float | Tensor = 1.0,
) -> Tensor:
    """Blend the current latent toward the noisy source outside the edit tube."""

    _validate_prediction(current_x_t, name="current_x_t")
    _matching(current_x_t, source_x_t, name="source_x_t")
    mask = _mask_for(current_x_t, edit_mask)
    strength = _control_for(
        current_x_t, strength, name="strength", unit_interval=True
    )
    anchor_weight = (1.0 - mask) * strength
    return current_x_t + anchor_weight * (source_x_t - current_x_t)


def source_noisy_anchor(
    current_x_t: Tensor,
    source_x0: Tensor,
    noise: Tensor,
    sigma: float | Tensor,
    edit_mask: Tensor,
    *,
    strength: float | Tensor = 1.0,
) -> Tensor:
    """Construct the source flow state and anchor the unchanged region to it."""

    source_x_t = flow_noisy_latent(source_x0, noise, sigma)
    return anchor_to_source_noisy(
        current_x_t, source_x_t, edit_mask, strength=strength
    )

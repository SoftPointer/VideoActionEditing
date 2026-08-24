"""Rectified-flow tensor primitives used by local action editing."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from .masks import validate_video_mask


def _validate_latent(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    return value


def _same_tensor_spec(first: Tensor, second: Tensor, *, names: str) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{names} must have identical shapes")
    if first.device != second.device:
        raise ValueError(f"{names} must be on the same device")
    if first.dtype != second.dtype:
        raise ValueError(f"{names} must have the same dtype")


def _sigma_like(sigma: float | Tensor, reference: Tensor) -> Tensor:
    if isinstance(sigma, Tensor):
        value = sigma.to(device=reference.device, dtype=reference.dtype)
    else:
        value = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
    if value.ndim == 1:
        if value.numel() == 1:
            value = value.reshape(())
        elif value.shape[0] == reference.shape[0]:
            value = value.reshape(reference.shape[0], 1, 1, 1, 1)
    try:
        torch.broadcast_shapes(reference.shape, value.shape)
    except RuntimeError as error:
        raise ValueError(
            f"sigma shape {tuple(value.shape)} is not broadcastable to {tuple(reference.shape)}"
        ) from error
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("sigma must be finite")
    if not bool(((detached >= 0) & (detached <= 1)).all()):
        raise ValueError("sigma must lie in [0, 1]")
    return value


def flow_noisy_latent(x0: Tensor, noise: Tensor, sigma: float | Tensor) -> Tensor:
    """Compute ``x_t = (1 - sigma) * x0 + sigma * noise``."""

    _validate_latent(x0, name="x0")
    _validate_latent(noise, name="noise")
    _same_tensor_spec(x0, noise, names="x0 and noise")
    sigma_value = _sigma_like(sigma, x0)
    return (1.0 - sigma_value) * x0 + sigma_value * noise


def velocity_target(x0: Tensor, noise: Tensor) -> Tensor:
    """Return the constant rectified-flow velocity ``noise - x0``."""

    _validate_latent(x0, name="x0")
    _validate_latent(noise, name="noise")
    _same_tensor_spec(x0, noise, names="x0 and noise")
    return noise - x0


def reconstruct_x0(
    x_t: Tensor, velocity: Tensor, sigma: float | Tensor
) -> Tensor:
    """Reconstruct clean data as ``x0 = x_t - sigma * velocity``."""

    _validate_latent(x_t, name="x_t")
    _validate_latent(velocity, name="velocity")
    _same_tensor_spec(x_t, velocity, names="x_t and velocity")
    return x_t - _sigma_like(sigma, x_t) * velocity


class SharedNoiseSplice(NamedTuple):
    """Outputs of :func:`shared_noise_local_latent_splice`."""

    x_t: Tensor
    source_x_t: Tensor
    target_x_t: Tensor
    noise: Tensor
    local_x0: Tensor
    target_velocity: Tensor


def shared_noise_local_latent_splice(
    source_x0: Tensor,
    target_x0: Tensor,
    edit_mask: Tensor,
    sigma: float | Tensor,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SharedNoiseSplice:
    """Noise source/target with one sample, then splice target inside the tube.

    Sharing noise removes a stochastic discontinuity at the edit boundary. The
    returned branches make the exact training target auditable.
    """

    _validate_latent(source_x0, name="source_x0")
    _validate_latent(target_x0, name="target_x0")
    _same_tensor_spec(source_x0, target_x0, names="source_x0 and target_x0")
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=source_x0.shape[0],
        frames=source_x0.shape[2],
        height=source_x0.shape[3],
        width=source_x0.shape[4],
    )
    if edit_mask.device != source_x0.device:
        raise ValueError("edit_mask and latents must be on the same device")
    if noise is None:
        noise = torch.randn(
            source_x0.shape,
            dtype=source_x0.dtype,
            device=source_x0.device,
            generator=generator,
        )
    else:
        _validate_latent(noise, name="noise")
        _same_tensor_spec(source_x0, noise, names="source_x0 and noise")

    source_x_t = flow_noisy_latent(source_x0, noise, sigma)
    target_x_t = flow_noisy_latent(target_x0, noise, sigma)
    mask = edit_mask.to(dtype=source_x0.dtype)
    local_x0 = source_x0 * (1.0 - mask) + target_x0 * mask
    x_t = source_x_t * (1.0 - mask) + target_x_t * mask
    target_velocity = velocity_target(local_x0, noise)
    return SharedNoiseSplice(
        x_t,
        source_x_t,
        target_x_t,
        noise,
        local_x0,
        target_velocity,
    )


shared_noise_local_splice = shared_noise_local_latent_splice

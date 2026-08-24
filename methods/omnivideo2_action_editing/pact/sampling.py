"""Pure-PyTorch rectified-flow sampling for preservation-aware editing.

The sampler follows Wan's rationally shifted, descending sigma schedule.  A
step first integrates the predicted velocity from ``sigma_current`` to
``sigma_next`` and only then anchors the keep region to the source flow state
at ``sigma_next``.  The latter detail is important: anchoring to the source at
the old sigma places the keep region on the wrong flow trajectory.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Callable, NamedTuple

import torch
from torch import Tensor

from .flow import flow_noisy_latent
from .guidance import anchor_to_source_noisy
from .masks import validate_video_mask


VelocityFunction = Callable[[Tensor, Tensor], Tensor]


def _validate_latent(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    return value


def _require_matching(reference: Tensor, other: Tensor, *, name: str) -> None:
    _validate_latent(other, name=name)
    if other.shape != reference.shape:
        raise ValueError(f"{name} must have the same shape as the latent")
    if other.dtype != reference.dtype:
        raise ValueError(f"{name} must have the same dtype as the latent")
    if other.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the latent")


def _scalar_sigma(value: float | Tensor, reference: Tensor, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        sigma = value.detach().to(device=reference.device, dtype=reference.dtype).reshape(())
    elif isinstance(value, Real) and not isinstance(value, bool):
        sigma = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    else:
        raise TypeError(f"{name} must be a real scalar or scalar tensor")
    if not bool(torch.isfinite(sigma)):
        raise ValueError(f"{name} must be finite")
    if not bool((sigma >= 0) & (sigma <= 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return sigma


def _anchor_weight(
    reference: Tensor, edit_mask: Tensor, strength: float | Tensor
) -> Tensor:
    if isinstance(strength, Tensor):
        value = strength.to(device=reference.device, dtype=reference.dtype)
    elif isinstance(strength, Real) and not isinstance(strength, bool):
        value = torch.as_tensor(
            strength, device=reference.device, dtype=reference.dtype
        )
    else:
        raise TypeError("anchor_strength must be a real scalar or tensor")
    if value.ndim == 1:
        if value.numel() == 1:
            value = value.reshape(())
        elif value.shape[0] == reference.shape[0]:
            value = value.reshape(reference.shape[0], 1, 1, 1, 1)
    try:
        torch.broadcast_shapes(reference.shape, value.shape)
    except RuntimeError as error:
        raise ValueError(
            "anchor_strength shape cannot broadcast to the latent shape"
        ) from error
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("anchor_strength must be finite")
    if not bool(((detached >= 0) & (detached <= 1)).all()):
        raise ValueError("anchor_strength must lie in [0, 1]")
    return (1.0 - edit_mask.to(dtype=reference.dtype)) * value


def validate_inference_sigmas(sigmas: Tensor) -> Tensor:
    """Validate a complete noise-to-data inference schedule.

    A valid schedule is a one-dimensional floating tensor with at least one
    integration step, starts exactly at one, ends exactly at zero, and is
    strictly descending.  It is returned unchanged so its dtype and device
    remain under the caller's control.
    """

    if not isinstance(sigmas, Tensor):
        raise TypeError("sigmas must be a torch.Tensor")
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be one-dimensional")
    if not sigmas.is_floating_point():
        raise TypeError("sigmas must have a floating dtype")
    if sigmas.numel() < 2:
        raise ValueError("sigmas must contain at least two entries")
    detached = sigmas.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("sigmas must contain only finite values")
    if not bool(((detached >= 0) & (detached <= 1)).all()):
        raise ValueError("sigmas must lie in [0, 1]")
    if not bool(detached[0] == 1):
        raise ValueError("sigmas must start at exactly 1")
    if not bool(detached[-1] == 0):
        raise ValueError("sigmas must include a final value of exactly 0")
    if not bool((detached[1:] < detached[:-1]).all()):
        raise ValueError("sigmas must be strictly descending")
    return sigmas


def wan_rational_shifted_sigmas(
    num_inference_steps: int,
    *,
    shift: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build Wan's rationally shifted schedule, including terminal zero.

    Starting with ``u = linspace(1, 0, steps + 1)``, Wan applies
    ``sigma = shift * u / (1 + (shift - 1) * u)``.  Thus the returned tensor
    contains ``num_inference_steps + 1`` states for exactly
    ``num_inference_steps`` Euler updates.
    """

    if (
        not isinstance(num_inference_steps, Integral)
        or isinstance(num_inference_steps, bool)
        or int(num_inference_steps) <= 0
    ):
        raise ValueError("num_inference_steps must be a positive integer")
    if not isinstance(shift, Real) or isinstance(shift, bool):
        raise TypeError("shift must be a real number")
    shift_value = float(shift)
    if not torch.isfinite(torch.tensor(shift_value)) or shift_value <= 0:
        raise ValueError("shift must be finite and positive")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")

    base = torch.linspace(
        1.0,
        0.0,
        int(num_inference_steps) + 1,
        dtype=dtype,
        device=device,
    )
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    # Preserve exact endpoints even for low-precision arithmetic.
    shifted[0] = 1.0
    shifted[-1] = 0.0
    return validate_inference_sigmas(shifted)


def euler_flow_step(
    current_x: Tensor,
    velocity: Tensor,
    *,
    sigma_current: float | Tensor,
    sigma_next: float | Tensor,
) -> Tensor:
    """Take one descending rectified-flow Euler step."""

    _validate_latent(current_x, name="current_x")
    _require_matching(current_x, velocity, name="velocity")
    current = _scalar_sigma(sigma_current, current_x, name="sigma_current")
    next_value = _scalar_sigma(sigma_next, current_x, name="sigma_next")
    if not bool(current > next_value):
        raise ValueError("sigma_current must be strictly greater than sigma_next")
    return current_x + (next_value - current) * velocity


class AnchoredEulerStep(NamedTuple):
    """Auditable outputs from :func:`anchored_euler_flow_step`."""

    x_next: Tensor
    euler_x_next: Tensor
    source_x_next: Tensor


def anchored_euler_flow_step(
    current_x: Tensor,
    velocity: Tensor,
    source_x0: Tensor,
    initial_noise: Tensor,
    edit_mask: Tensor,
    *,
    sigma_current: float | Tensor,
    sigma_next: float | Tensor,
    anchor_strength: float | Tensor = 1.0,
) -> AnchoredEulerStep:
    """Euler-step, then anchor outside ``edit_mask`` at ``sigma_next``.

    ``initial_noise`` must be the same realization used to initialize the
    complete trajectory.  The source anchor is deliberately constructed at
    ``sigma_next`` after integration, never at ``sigma_current``.
    """

    _validate_latent(current_x, name="current_x")
    _require_matching(current_x, velocity, name="velocity")
    _require_matching(current_x, source_x0, name="source_x0")
    _require_matching(current_x, initial_noise, name="initial_noise")
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=current_x.shape[0],
        frames=current_x.shape[2],
        height=current_x.shape[3],
        width=current_x.shape[4],
    )
    if edit_mask.device != current_x.device:
        raise ValueError("edit_mask and latents must be on the same device")

    euler_x_next = euler_flow_step(
        current_x,
        velocity,
        sigma_current=sigma_current,
        sigma_next=sigma_next,
    )
    next_value = _scalar_sigma(sigma_next, current_x, name="sigma_next")
    source_x_next = flow_noisy_latent(source_x0, initial_noise, next_value)
    x_next = anchor_to_source_noisy(
        euler_x_next,
        source_x_next,
        edit_mask,
        strength=anchor_strength,
    )
    # Preserve the exact source bits wherever a hard keep mask and full
    # strength request complete anchoring; the algebraically equivalent blend
    # can otherwise accumulate a rounding error through subtraction/addition.
    anchor_weight = _anchor_weight(current_x, edit_mask, anchor_strength)
    x_next = torch.where(anchor_weight == 1, source_x_next, x_next)
    return AnchoredEulerStep(x_next, euler_x_next, source_x_next)


def sample_anchored_flow(
    initial_noise: Tensor,
    source_x0: Tensor,
    edit_mask: Tensor,
    sigmas: Tensor,
    velocity_fn: VelocityFunction,
    *,
    anchor_strength: float | Tensor = 1.0,
) -> Tensor:
    """Sample a local edit while re-anchoring the keep region every step.

    ``velocity_fn`` receives ``(current_x, sigma_current)``.  The schedule must
    describe the complete trajectory from one to zero; use
    :func:`wan_rational_shifted_sigmas` for the canonical Wan schedule.
    """

    _validate_latent(initial_noise, name="initial_noise")
    _require_matching(initial_noise, source_x0, name="source_x0")
    validate_video_mask(
        edit_mask,
        name="edit_mask",
        batch_size=initial_noise.shape[0],
        frames=initial_noise.shape[2],
        height=initial_noise.shape[3],
        width=initial_noise.shape[4],
    )
    if edit_mask.device != initial_noise.device:
        raise ValueError("edit_mask and latents must be on the same device")
    validate_inference_sigmas(sigmas)
    if not callable(velocity_fn):
        raise TypeError("velocity_fn must be callable")

    current_x = initial_noise
    for index in range(sigmas.numel() - 1):
        sigma_current = _scalar_sigma(
            sigmas[index], current_x, name="sigma_current"
        )
        sigma_next = _scalar_sigma(sigmas[index + 1], current_x, name="sigma_next")
        velocity = velocity_fn(current_x, sigma_current)
        step = anchored_euler_flow_step(
            current_x,
            velocity,
            source_x0,
            initial_noise,
            edit_mask,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            anchor_strength=anchor_strength,
        )
        current_x = step.x_next
    return current_x


__all__ = [
    "AnchoredEulerStep",
    "VelocityFunction",
    "anchored_euler_flow_step",
    "euler_flow_step",
    "sample_anchored_flow",
    "validate_inference_sigmas",
    "wan_rational_shifted_sigmas",
]

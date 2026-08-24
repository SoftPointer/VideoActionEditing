"""Full-target rectified-flow primitives for action editing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def _validate_video_latent(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B, C, T, H, W]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if min(value.shape) <= 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _same_spec(first: Tensor, second: Tensor, *, names: str) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{names} must have identical shapes")
    if first.dtype != second.dtype:
        raise ValueError(f"{names} must have the same dtype")
    if first.device != second.device:
        raise ValueError(f"{names} must be on the same device")


def _sigma_like(sigma: float | Tensor, reference: Tensor) -> Tensor:
    if isinstance(sigma, Tensor):
        value = sigma.to(device=reference.device, dtype=reference.dtype)
    elif isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
        value = torch.as_tensor(
            float(sigma), device=reference.device, dtype=reference.dtype
        )
    else:
        raise TypeError("sigma must be a number or torch.Tensor")
    if value.ndim == 1:
        if value.numel() == 1:
            value = value.reshape(())
        elif value.shape[0] == reference.shape[0]:
            value = value.reshape(reference.shape[0], 1, 1, 1, 1)
    try:
        torch.broadcast_shapes(reference.shape, value.shape)
    except RuntimeError as error:
        raise ValueError(
            f"sigma shape {tuple(value.shape)} is not broadcastable to "
            f"{tuple(reference.shape)}"
        ) from error
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("sigma must be finite")
    if not bool(((detached >= 0.0) & (detached <= 1.0)).all()):
        raise ValueError("sigma must lie in [0, 1]")
    return value


def shifted_rectified_flow_sigma(uniform: Tensor, shift: float = 5.0) -> Tensor:
    """Apply Wan's rational shift to values in the unit interval."""

    if not isinstance(uniform, Tensor) or not uniform.is_floating_point():
        raise TypeError("uniform must be a floating torch.Tensor")
    detached = uniform.detach()
    if not bool(torch.isfinite(detached).all()):
        raise ValueError("uniform must be finite")
    if not bool(((detached >= 0.0) & (detached <= 1.0)).all()):
        raise ValueError("uniform must lie in [0, 1]")
    if not isinstance(shift, (int, float)) or isinstance(shift, bool):
        raise TypeError("shift must be a number")
    shift = float(shift)
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    return shift * uniform / (1.0 + (shift - 1.0) * uniform)


@dataclass(frozen=True)
class DiffSynthWanTrainingSample:
    """One batch-shared draw from the shifted discrete Wan SFT schedule."""

    timestep_id: int
    sigma: Tensor
    timestep: Tensor
    flow_weight: Tensor


class DiffSynthWanTrainingScheduler:
    """Standalone FP32 DiffSynth-compatible Wan training schedule.

    The schedule is built from ``linspace(1, 0, N + 1)[:-1]``, followed by
    Wan's rational shift. A single discrete entry is shared by the whole batch,
    matching the SFT sampling convention. No legacy editing module is imported.
    """

    def __init__(self, shift: float = 5.0, num_train_timesteps: int = 1000) -> None:
        if not isinstance(shift, (int, float)) or isinstance(shift, bool):
            raise TypeError("shift must be a number")
        self.shift = float(shift)
        if not math.isfinite(self.shift) or self.shift <= 0.0:
            raise ValueError("shift must be finite and positive")
        if type(num_train_timesteps) is not int or num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be an integer of at least 2")
        self.num_train_timesteps = num_train_timesteps
        unshifted = torch.linspace(
            1.0,
            0.0,
            num_train_timesteps + 1,
            dtype=torch.float32,
            device="cpu",
        )[:-1]
        self.sigmas = shifted_rectified_flow_sigma(unshifted, self.shift)
        self.timesteps = self.sigmas * float(num_train_timesteps)
        profile = torch.exp(
            -2.0
            * (
                (self.timesteps - float(num_train_timesteps) / 2.0)
                / float(num_train_timesteps)
            ).square()
        )
        shifted_profile = profile - profile.min()
        normalizer = shifted_profile.sum()
        if not bool(torch.isfinite(normalizer)) or not bool(normalizer > 0.0):
            raise RuntimeError("invalid flow-weight normalization")
        self.flow_weights = shifted_profile * (
            float(num_train_timesteps) / normalizer
        )

    @staticmethod
    def _batch_size(value: int) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("batch_size must be a positive integer")
        return value

    @staticmethod
    def _dtype(value: torch.dtype) -> torch.dtype:
        if not isinstance(value, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=value).is_floating_point():
            raise TypeError("dtype must be floating point")
        return value

    def at(
        self,
        timestep_id: int,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> DiffSynthWanTrainingSample:
        if type(timestep_id) is not int or not (
            0 <= timestep_id < self.num_train_timesteps
        ):
            raise ValueError(
                f"timestep_id must lie in [0, {self.num_train_timesteps})"
            )
        batch_size = self._batch_size(batch_size)
        dtype = self._dtype(dtype)
        target_device = torch.device(device)
        sigma = self.sigmas[timestep_id].to(
            device=target_device, dtype=dtype
        ).repeat(batch_size)
        timestep = self.timesteps[timestep_id].to(
            device=target_device, dtype=dtype
        ).repeat(batch_size)
        weight = self.flow_weights[timestep_id].to(
            device=target_device, dtype=torch.float32
        )
        return DiffSynthWanTrainingSample(
            timestep_id=timestep_id,
            sigma=sigma,
            timestep=timestep,
            flow_weight=weight,
        )

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> DiffSynthWanTrainingSample:
        batch_size = self._batch_size(batch_size)
        self._dtype(dtype)
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator or None")
            if torch.device(generator.device).type != "cpu":
                raise ValueError("schedule sampling requires a CPU generator")
        timestep_id = int(
            torch.randint(
                0,
                self.num_train_timesteps,
                (1,),
                device="cpu",
                generator=generator,
            ).item()
        )
        return self.at(
            timestep_id,
            batch_size,
            device=device,
            dtype=dtype,
        )


def flow_noisy_latent(target_x0: Tensor, noise: Tensor, sigma: float | Tensor) -> Tensor:
    """Interpolate the complete target with noise: ``x_t=(1-s)x_0+s*eps``."""

    _validate_video_latent(target_x0, name="target_x0")
    _validate_video_latent(noise, name="noise")
    _same_spec(target_x0, noise, names="target_x0 and noise")
    sigma_value = _sigma_like(sigma, target_x0)
    return (1.0 - sigma_value) * target_x0 + sigma_value * noise


def velocity_target(target_x0: Tensor, noise: Tensor) -> Tensor:
    """Return the complete-target rectified-flow velocity ``eps - x_0``."""

    _validate_video_latent(target_x0, name="target_x0")
    _validate_video_latent(noise, name="noise")
    _same_spec(target_x0, noise, names="target_x0 and noise")
    return noise - target_x0


def reconstruct_x0(x_t: Tensor, velocity: Tensor, sigma: float | Tensor) -> Tensor:
    _validate_video_latent(x_t, name="x_t")
    _validate_video_latent(velocity, name="velocity")
    _same_spec(x_t, velocity, names="x_t and velocity")
    return x_t - _sigma_like(sigma, x_t) * velocity


@dataclass(frozen=True)
class FullTargetFlowBatch:
    """A rectified-flow sample whose endpoint is always the complete target."""

    target_x0: Tensor
    x_t: Tensor
    noise: Tensor
    target_velocity: Tensor
    sigma: Tensor


def prepare_full_target_flow(
    target_x0: Tensor,
    sigma: float | Tensor,
    *,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FullTargetFlowBatch:
    """Noise and supervise the full target latent without spatial splicing."""

    _validate_video_latent(target_x0, name="target_x0")
    sigma_value = _sigma_like(sigma, target_x0)
    if noise is None:
        noise = torch.randn(
            target_x0.shape,
            device=target_x0.device,
            dtype=target_x0.dtype,
            generator=generator,
        )
    else:
        _validate_video_latent(noise, name="noise")
        _same_spec(target_x0, noise, names="target_x0 and noise")
    return FullTargetFlowBatch(
        target_x0=target_x0,
        x_t=(1.0 - sigma_value) * target_x0 + sigma_value * noise,
        noise=noise,
        target_velocity=noise - target_x0,
        sigma=sigma_value,
    )


def full_target_flow_loss(
    predicted_velocity: Tensor,
    target_velocity: Tensor,
    *,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Mean squared velocity error over every target latent element."""

    _validate_video_latent(predicted_velocity, name="predicted_velocity")
    _validate_video_latent(target_velocity, name="target_velocity")
    _same_spec(
        predicted_velocity,
        target_velocity,
        names="predicted_velocity and target_velocity",
    )
    per_sample = (predicted_velocity.float() - target_velocity.float()).square().mean(
        dim=(1, 2, 3, 4)
    )
    if sample_weight is None:
        return per_sample.mean()
    if not isinstance(sample_weight, Tensor):
        raise TypeError("sample_weight must be a torch.Tensor")
    weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype)
    if weight.ndim == 0:
        weight = weight.expand_as(per_sample)
    if weight.shape != per_sample.shape:
        raise ValueError("sample_weight must be scalar or have shape [B]")
    if not bool(torch.isfinite(weight.detach()).all()) or not bool(
        (weight.detach() >= 0.0).all()
    ):
        raise ValueError("sample_weight must be finite and non-negative")
    if not bool(weight.detach().sum() > 0.0):
        raise ValueError("sample_weight must have a positive sum")
    # DiffSynth's BSMNTW table is an objective multiplier, not an importance-
    # sampling probability. Do not renormalize it within a batch.
    return (per_sample * weight).mean()


__all__ = [
    "DiffSynthWanTrainingSample",
    "DiffSynthWanTrainingScheduler",
    "FullTargetFlowBatch",
    "flow_noisy_latent",
    "full_target_flow_loss",
    "prepare_full_target_flow",
    "reconstruct_x0",
    "shifted_rectified_flow_sigma",
    "velocity_target",
]

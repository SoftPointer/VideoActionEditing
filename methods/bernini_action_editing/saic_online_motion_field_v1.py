#!/usr/bin/env python3
"""Inference-available online motion field for Bernini SAIC Stage-B.

The representation is produced at the *current* native diffusion state, not
from an offline generated proposal.  A frozen T2V teacher is queried twice on
the exact same noisy target tensor and scheduler coordinate: once with the
natural-language action caption and once with its natural-language no-op
caption.  Their velocity difference is temporal-DC rejected and compressed
without learned parameters into one 32-D code for each of the 21 exact81
latent phases::

    d_t       = (v_action - v_noop)_t - mean_t(v_action - v_noop)
    code_t    = [ spatial_mean(d_t),
                  spatial_mean(d_t * normalized_current_state_t) ]

The first 16 channels retain signed motion-field transport.  The second 16
retain where that transport agrees with the current state, allowing a later
state-conditioned operator to act differently on different source content.
Temporal-DC rejection is deliberately fixed rather than learned so a static
appearance offset cannot become an action code merely through few-shot
training.

No T2V RGB, decoded video, clean/generated latent, sampled noise, proposal,
mask, pose, flow, track, or trajectory crosses this API.  The only visual
tensor visible to the frozen T2V callback is the current noisy target state
that the native editor is already evaluating.  A no-op (identical canonical
captions or identical frozen fields) produces an exact all-zero code.  The
same function is used under training and inference; the returned code and
teacher fields are always detached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping

import torch

if __package__:
    from . import inference_sigma_strata as sigma_strata
else:  # Direct import from methods/bernini_action_editing.
    import inference_sigma_strata as sigma_strata


SCHEMA_VERSION = "bernini-saic-online-motion-field-v1"
LATENT_CHANNELS = 16
LATENT_PHASES_EXACT81 = 21
PHASE_CODE_DIM = 32
FIELD_BRANCHES = ("action", "noop")

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class SAICOnlineMotionFieldError(RuntimeError):
    """Raised before an ambiguous or non-deployment motion field is used."""


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
        raise SAICOnlineMotionFieldError(
            f"value is not canonical finite ASCII JSON: {error}"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # pragma: no cover - older torch compatibility
        return int(value.storage().data_ptr())


def _tensor_binding(value: torch.Tensor) -> tuple[Any, ...]:
    return (
        id(value),
        _storage_pointer(value),
        int(value.storage_offset()),
        tuple(map(int, value.shape)),
        tuple(map(int, value.stride())),
        value.dtype,
        value.device,
        value.layout,
        int(getattr(value, "_version", 0)),
    )


def _assert_tensor_binding(
    value: Any, expected: tuple[Any, ...], *, label: str
) -> torch.Tensor:
    if type(value) is not torch.Tensor or _tensor_binding(value) != expected:
        raise SAICOnlineMotionFieldError(f"bound runtime {label} changed")
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(_canonical_json(list(map(int, cpu.shape))))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _validate_current_state(value: Any, *, label: str) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.ndim != 5
        or tuple(map(int, value.shape[:3]))
        != (1, LATENT_CHANNELS, LATENT_PHASES_EXACT81)
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or value.dtype not in (torch.float32, torch.bfloat16)
        or value.device.type == "meta"
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICOnlineMotionFieldError(
            f"{label} must be detached finite FP32/BF16 exact81 "
            "[1,16,21,H,W] current state"
        )
    return value


def _validate_natural_caption(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not 8 <= len(value) <= 2048
        or value != value.strip()
        or "\x00" in value
        or not any(character.isalpha() for character in value)
    ):
        raise SAICOnlineMotionFieldError(
            f"{label} must be one canonical natural-language caption"
        )
    # A bare registry-like token is an action ID, not a deployable caption.
    if _SAFE_IDENTIFIER.fullmatch(value) is not None:
        raise SAICOnlineMotionFieldError(
            f"{label} must be natural language rather than an action ID"
        )
    return value


def _timestep_schedule_index(timestep: Any) -> int:
    if (
        type(timestep) is not torch.Tensor
        or timestep.dtype != torch.float32
        or timestep.numel() != 1
        or timestep.requires_grad
        or timestep.grad_fn is not None
        or not bool(torch.isfinite(timestep).all().item())
    ):
        raise SAICOnlineMotionFieldError(
            "runtime timestep must be one detached finite device-local FP32 tensor"
        )
    numeric = float(timestep.detach().item())
    matches = [
        index
        for index, expected in enumerate(sigma_strata.PINNED_TIMESTEPS)
        if numeric == float(expected)
    ]
    if len(matches) != 1:
        raise SAICOnlineMotionFieldError(
            "runtime timestep is not one unique pinned UniPC40 coordinate"
        )
    return matches[0]


def _audit_scheduler(scheduler: Any, timestep: torch.Tensor) -> tuple[int, str]:
    try:
        receipt = sigma_strata.audit_runtime_unipc_schedule(
            scheduler, initialize=False
        )
    except Exception as error:
        raise SAICOnlineMotionFieldError(
            f"runtime UniPC schedule is not pinned exact40: {error}"
        ) from error
    if receipt.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256:
        raise SAICOnlineMotionFieldError("runtime UniPC schedule digest differs")
    index = _timestep_schedule_index(timestep)
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if (
        type(timesteps) is not torch.Tensor
        or type(sigmas) is not torch.Tensor
        or int(timesteps.numel()) != len(sigma_strata.PINNED_TIMESTEPS)
        or int(sigmas.numel()) != len(sigma_strata.PINNED_POSITIVE_SIGMAS) + 1
        or int(timesteps[index].item()) != sigma_strata.PINNED_TIMESTEPS[index]
    ):
        raise SAICOnlineMotionFieldError("actual scheduler coordinate differs")
    sigma_hex = sigma_strata._float32_hex(  # noqa: SLF001 - pinned-bit guard
        sigmas[index].item(), label="actual scheduler sigma"
    )
    if sigma_hex != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]:
        raise SAICOnlineMotionFieldError("actual scheduler sigma bits differ")
    return index, sigma_hex


@dataclass(frozen=True)
class FrozenT2VVelocityRequest:
    """The complete and deliberately narrow frozen-teacher call surface."""

    branch: str
    natural_language_prompt: str
    current_noisy_target: torch.Tensor = field(repr=False, compare=False)
    timestep: torch.Tensor = field(repr=False, compare=False)
    actual_sigma: torch.Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.branch not in FIELD_BRANCHES:
            raise SAICOnlineMotionFieldError("frozen T2V branch must be action/noop")
        _validate_natural_caption(
            self.natural_language_prompt,
            label=f"{self.branch} natural_language_prompt",
        )
        _validate_current_state(
            self.current_noisy_target, label="current_noisy_target"
        )
        _timestep_schedule_index(self.timestep)
        if (
            type(self.actual_sigma) is not torch.Tensor
            or self.actual_sigma.dtype != torch.float32
            or self.actual_sigma.numel() != 1
            or self.actual_sigma.requires_grad
            or self.actual_sigma.grad_fn is not None
            or not bool(torch.isfinite(self.actual_sigma).all().item())
        ):
            raise SAICOnlineMotionFieldError(
                "actual_sigma must be one detached finite scheduler FP32 view"
            )


@dataclass(frozen=True)
class SAICOnlineMotionField:
    """Detached 21x32 online code bound to one live current-state query."""

    phase_code: torch.Tensor = field(repr=False, compare=False)
    action_prompt_sha256: str
    noop_prompt_sha256: str
    schedule_index: int
    sigma_float32_be_hex: str
    is_noop: bool
    _current_state_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _scheduler: Any = field(repr=False, compare=False)
    _scheduler_timesteps_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _scheduler_sigmas_binding: tuple[Any, ...] = field(repr=False, compare=False)
    _timestep_binding: tuple[Any, ...] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.phase_code) is not torch.Tensor
            or self.phase_code.dtype != torch.float32
            or tuple(map(int, self.phase_code.shape))
            != (LATENT_PHASES_EXACT81, PHASE_CODE_DIM)
            or self.phase_code.requires_grad
            or self.phase_code.grad_fn is not None
            or not self.phase_code.is_contiguous()
            or not bool(torch.isfinite(self.phase_code).all().item())
        ):
            raise SAICOnlineMotionFieldError(
                "phase code must be detached contiguous finite FP32 [21,32]"
            )
        exact_zero = int(torch.count_nonzero(self.phase_code).item()) == 0
        if exact_zero != self.is_noop:
            raise SAICOnlineMotionFieldError("no-op iff phase code is exact zero")
        if (
            type(self.schedule_index) is not int
            or not 0 <= self.schedule_index < len(sigma_strata.PINNED_TIMESTEPS)
            or self.sigma_float32_be_hex
            != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[self.schedule_index]
        ):
            raise SAICOnlineMotionFieldError("motion-field scheduler binding differs")

    def assert_live(
        self,
        *,
        current_noisy_target: torch.Tensor,
        scheduler: Any,
        timestep: torch.Tensor,
    ) -> None:
        _validate_current_state(current_noisy_target, label="current_noisy_target")
        _assert_tensor_binding(
            current_noisy_target,
            self._current_state_binding,
            label="current noisy target",
        )
        if scheduler is not self._scheduler:
            raise SAICOnlineMotionFieldError("bound runtime scheduler object changed")
        _assert_tensor_binding(
            getattr(scheduler, "timesteps", None),
            self._scheduler_timesteps_binding,
            label="scheduler timesteps",
        )
        _assert_tensor_binding(
            getattr(scheduler, "sigmas", None),
            self._scheduler_sigmas_binding,
            label="scheduler sigmas",
        )
        _assert_tensor_binding(
            timestep, self._timestep_binding, label="forward timestep"
        )
        index, sigma_hex = _audit_scheduler(scheduler, timestep)
        if index != self.schedule_index or sigma_hex != self.sigma_float32_be_hex:
            raise SAICOnlineMotionFieldError("bound runtime scheduler coordinate changed")

    def receipt(self) -> Mapping[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "phase_code_shape": [LATENT_PHASES_EXACT81, PHASE_CODE_DIM],
            "phase_code_sha256": _tensor_digest(self.phase_code),
            "action_prompt_sha256": self.action_prompt_sha256,
            "noop_prompt_sha256": self.noop_prompt_sha256,
            "schedule_index": self.schedule_index,
            "sigma_float32_be_hex": self.sigma_float32_be_hex,
            "is_noop": self.is_noop,
            "same_current_state_for_action_and_noop": True,
            "teacher_frozen_and_outputs_detached": True,
            "temporal_dc_rejected": True,
            "learned_motion_encoder": False,
            "training_and_inference_function": "build_online_motion_field",
            "t2v_media_or_proposal_consumed": False,
            "mask_pose_flow_track_or_trajectory_consumed": False,
        }
        return {**value, "digest": _object_sha256(value)}


def _validate_teacher_velocity(
    value: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
    label: str,
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or tuple(map(int, value.shape)) != expected_shape
        or value.device != expected_device
        or value.dtype not in (torch.float32, torch.bfloat16)
        or value.layout != torch.strided
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SAICOnlineMotionFieldError(
            f"{label} must be a detached finite FP32/BF16 velocity with exact "
            "current-state geometry"
        )
    return value


def _encode_phase_code(
    current_noisy_target: torch.Tensor,
    action_velocity: torch.Tensor,
    noop_velocity: torch.Tensor,
) -> torch.Tensor:
    """Fixed state-aware compression; exact equal fields map to exact zero."""

    with torch.autocast(device_type=current_noisy_target.device.type, enabled=False):
        delta = action_velocity.float() - noop_velocity.float()
        # Reject the time-constant field before all statistics.  This is the
        # representation-level appearance nuisance quotient.
        delta = delta - delta.mean(dim=2, keepdim=True)
        current = current_noisy_target.float()
        centered = current - current.mean(dim=(-2, -1), keepdim=True)
        spatial_rms = centered.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        normalized_current = centered / spatial_rms.clamp_min(1.0e-6)
        signed_transport = delta.mean(dim=(-2, -1))
        state_alignment = (delta * normalized_current).mean(dim=(-2, -1))
        code = torch.cat((signed_transport, state_alignment), dim=1)
        code = code.squeeze(0).transpose(0, 1).contiguous()
    if tuple(code.shape) != (LATENT_PHASES_EXACT81, PHASE_CODE_DIM):
        raise SAICOnlineMotionFieldError("internal phase-code geometry differs")
    if not bool(torch.isfinite(code).all().item()):
        raise SAICOnlineMotionFieldError("phase-code compression became non-finite")
    return code.detach()


def build_online_motion_field(
    *,
    current_noisy_target: torch.Tensor,
    action_prompt: str,
    noop_prompt: str,
    scheduler: Any,
    timestep: torch.Tensor,
    frozen_t2v_velocity: Callable[[FrozenT2VVelocityRequest], torch.Tensor],
) -> SAICOnlineMotionField:
    """Query frozen action/no-op fields and mint one runtime-bound phase code.

    This is the sole representation path for both optimization and sampling.
    Callers cannot supply a precomputed phase code or any discrete action key.
    """

    state = _validate_current_state(
        current_noisy_target, label="current_noisy_target"
    )
    action = _validate_natural_caption(action_prompt, label="action_prompt")
    noop = _validate_natural_caption(noop_prompt, label="noop_prompt")
    if not callable(frozen_t2v_velocity):
        raise SAICOnlineMotionFieldError("frozen_t2v_velocity must be callable")
    if timestep.device != state.device:
        raise SAICOnlineMotionFieldError(
            "runtime timestep and current noisy target must share a device"
        )
    schedule_index, sigma_hex = _audit_scheduler(scheduler, timestep)
    timesteps = getattr(scheduler, "timesteps")
    sigmas = getattr(scheduler, "sigmas")
    state_binding = _tensor_binding(state)
    timestep_binding = _tensor_binding(timestep)
    timesteps_binding = _tensor_binding(timesteps)
    sigmas_binding = _tensor_binding(sigmas)

    # Canonical identical captions are the deployment no-op.  Bypass the
    # teacher altogether so even a stochastic/misconfigured callback cannot
    # create nonzero work for a no-op command.
    if action == noop:
        phase_code = torch.zeros(
            (LATENT_PHASES_EXACT81, PHASE_CODE_DIM),
            dtype=torch.float32,
            device=state.device,
        )
    else:
        actual_sigma = sigmas[schedule_index].detach()
        requests = (
            FrozenT2VVelocityRequest(
                branch="action",
                natural_language_prompt=action,
                current_noisy_target=state,
                timestep=timestep,
                actual_sigma=actual_sigma,
            ),
            FrozenT2VVelocityRequest(
                branch="noop",
                natural_language_prompt=noop,
                current_noisy_target=state,
                timestep=timestep,
                actual_sigma=actual_sigma,
            ),
        )
        outputs: list[torch.Tensor] = []
        for request in requests:
            try:
                with torch.inference_mode():
                    raw = frozen_t2v_velocity(request)
            except Exception as error:
                raise SAICOnlineMotionFieldError(
                    f"frozen T2V {request.branch} velocity query failed: {error}"
                ) from error
            _assert_tensor_binding(
                state, state_binding, label="current noisy target during teacher query"
            )
            _assert_tensor_binding(
                timestep, timestep_binding, label="timestep during teacher query"
            )
            _assert_tensor_binding(
                timesteps, timesteps_binding, label="scheduler timesteps during teacher query"
            )
            _assert_tensor_binding(
                sigmas, sigmas_binding, label="scheduler sigmas during teacher query"
            )
            outputs.append(
                _validate_teacher_velocity(
                    raw,
                    expected_shape=tuple(map(int, state.shape)),
                    expected_device=state.device,
                    label=f"frozen T2V {request.branch} velocity",
                )
            )
        phase_code = _encode_phase_code(state, outputs[0], outputs[1])

    result = SAICOnlineMotionField(
        phase_code=phase_code.contiguous(),
        action_prompt_sha256=hashlib.sha256(action.encode("utf-8")).hexdigest(),
        noop_prompt_sha256=hashlib.sha256(noop.encode("utf-8")).hexdigest(),
        schedule_index=schedule_index,
        sigma_float32_be_hex=sigma_hex,
        is_noop=int(torch.count_nonzero(phase_code).item()) == 0,
        _current_state_binding=state_binding,
        _scheduler=scheduler,
        _scheduler_timesteps_binding=timesteps_binding,
        _scheduler_sigmas_binding=sigmas_binding,
        _timestep_binding=timestep_binding,
    )
    result.assert_live(
        current_noisy_target=state, scheduler=scheduler, timestep=timestep
    )
    return result


__all__ = [
    "FIELD_BRANCHES",
    "FrozenT2VVelocityRequest",
    "LATENT_CHANNELS",
    "LATENT_PHASES_EXACT81",
    "PHASE_CODE_DIM",
    "SAICOnlineMotionField",
    "SAICOnlineMotionFieldError",
    "SCHEMA_VERSION",
    "build_online_motion_field",
]

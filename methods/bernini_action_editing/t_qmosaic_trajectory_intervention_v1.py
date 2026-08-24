#!/usr/bin/env python3
"""Strict actual-trajectory intervention primitive for T-Q-MOSAIC v1.

This module is an engineering-only adapter for an official 40-step sampler.
It deliberately does not implement a sampler and does not accept a callback,
mask, track, pose, optical flow, box, seed choice, dose choice, or candidate
selection rule.

The contract has two passes:

* a no-op/base pass captures the *actual pre-step states* at schedule indices
  20, 28, and 33; and
* a replay pass injects fixed state-space deltas immediately after scheduler
  outputs 19, 27, and 32, respectively.

The three deltas are derived from detached FP32 state VJPs.  Their individual
relative-L2 doses follow fixed weights ``(1, 1, .5)`` and their root-sum-square
relative dose is exactly the configured engineering budget ``0.01`` (up to
audited FP32 realization error).  ``sign=0`` is a true no-op: every scheduler
output tensor is returned by identity, without a clone or arithmetic.

Receipts bind the base states, VJPs, deltas, schedule, and replay observations.
They authorize neither semantic success nor training/parameter updates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
import inference_sigma_strata as _pinned_schedule


CAPTURE_SCHEMA_VERSION = "bernini-t-qmosaic-actual-trajectory-capture-v1"
INTERVENTION_SCHEMA_VERSION = "bernini-t-qmosaic-state-vjp-intervention-v1"
REPLAY_SCHEMA_VERSION = "bernini-t-qmosaic-exact40-replay-v1"

EXACT_SCHEDULER_CALLS = 40
PINNED_TIMESTEPS = _pinned_schedule.PINNED_TIMESTEPS
PINNED_SIGMAS = _pinned_schedule.PINNED_POSITIVE_SIGMAS
PINNED_SIGMA_FLOAT32_HEX = _pinned_schedule.PINNED_POSITIVE_SIGMA_FLOAT32_HEX
PINNED_SCHEDULE_SHA256 = _pinned_schedule.SCHEDULE_SHA256
CAPTURE_PRE_STEP_INDICES = (20, 28, 33)
INJECT_AFTER_STEP_INDICES = (19, 27, 32)
CAPTURE_TIMESTEPS = tuple(PINNED_TIMESTEPS[index] for index in CAPTURE_PRE_STEP_INDICES)
CAPTURE_SIGMAS = tuple(PINNED_SIGMAS[index] for index in CAPTURE_PRE_STEP_INDICES)
if (
    len(PINNED_TIMESTEPS) != EXACT_SCHEDULER_CALLS
    or len(PINNED_SIGMAS) != EXACT_SCHEDULER_CALLS
    or len(PINNED_SIGMA_FLOAT32_HEX) != EXACT_SCHEDULER_CALLS
    or CAPTURE_TIMESTEPS != (833, 682, 516)
    or CAPTURE_SIGMAS
    != (
        0.8336109519004822,
        0.6825404167175293,
        0.5161304473876953,
    )
):  # pragma: no cover - import-time cross-module schedule guard
    raise RuntimeError("pinned T-Q-MOSAIC exact40 schedule constants differ")
TRAJECTORY_WEIGHTS = (1.0, 1.0, 0.5)
TOTAL_RELATIVE_L2_DOSE = 0.01

_ANCHOR_BY_PRE_STEP = MappingProxyType(
    {
        index: (timestep, sigma)
        for index, timestep, sigma in zip(
            CAPTURE_PRE_STEP_INDICES,
            CAPTURE_TIMESTEPS,
            CAPTURE_SIGMAS,
            strict=True,
        )
    }
)
_ANCHOR_POSITION_BY_PRE_STEP = MappingProxyType(
    {index: position for position, index in enumerate(CAPTURE_PRE_STEP_INDICES)}
)
_ANCHOR_POSITION_BY_POST_STEP = MappingProxyType(
    {index: position for position, index in enumerate(INJECT_AFTER_STEP_INDICES)}
)
_MINIMUM_NORM = 1.0e-20
_DOSE_RELATIVE_TOLERANCE = 5.0e-5
_CONSTRUCTION_TOKEN = object()


class TQMosaicTrajectoryError(RuntimeError):
    """The closed T-Q-MOSAIC v1 trajectory contract was violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise TQMosaicTrajectoryError(
            "trajectory receipt is not finite canonical ASCII JSON"
        ) from error


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise TQMosaicTrajectoryError("trajectory receipt is already sealed")
    plain = dict(unsigned)
    return {
        **plain,
        "receipt_digest": hashlib.sha256(_canonical_json_bytes(plain)).hexdigest(),
    }


def _untyped_storage(value: torch.Tensor) -> Any:
    getter = getattr(value, "untyped_storage", None)
    if callable(getter):
        return getter()
    typed = value.storage()
    getter = getattr(typed, "_untyped", None)
    if callable(getter):
        return getter()
    raise TQMosaicTrajectoryError("tensor storage identity is unavailable")


def _tensor_sha256(value: torch.Tensor, *, label: str) -> str:
    """Hash exact logical bytes plus dtype/shape; signed zero stays distinct."""

    _validate_state_tensor(value, label=label)
    owned = value.detach().to(device="cpu").contiguous().clone()
    payload = io.BytesIO()
    storage = _untyped_storage(owned)
    storage._write_file(payload, False, False, 1)
    raw = payload.getvalue()
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise TQMosaicTrajectoryError(f"{label} byte closure differs")
    header = _canonical_json_bytes(
        {
            "dtype": str(owned.dtype),
            "shape": list(map(int, owned.shape)),
            "numel": int(owned.numel()),
        }
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def _validate_state_tensor(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...] | None = None,
    expected_device: str | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.ndim < 1
        or value.numel() < 1
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
    ):
        raise TQMosaicTrajectoryError(
            f"{label} must be a finite detached dense materialized FP32 tensor"
        )
    shape = tuple(map(int, value.shape))
    if expected_shape is not None and shape != expected_shape:
        raise TQMosaicTrajectoryError(f"{label} shape differs from base trajectory")
    if expected_device is not None and str(value.device) != expected_device:
        raise TQMosaicTrajectoryError(f"{label} device differs from base trajectory")
    return value


@dataclass(frozen=True)
class _TensorBinding:
    object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int
    storage_version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    tensor_sha256: str

    def portable(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "tensor_sha256": self.tensor_sha256,
            "construction_object_identity_live_checked": True,
            "construction_storage_identity_live_checked": True,
            "construction_storage_version_live_checked": True,
        }


def _bind_tensor(value: torch.Tensor, *, label: str) -> _TensorBinding:
    checked = _validate_state_tensor(value, label=label)
    try:
        storage = _untyped_storage(checked)
        pointer = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
        version = int(checked._version)  # noqa: SLF001 - mutation seal
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise TQMosaicTrajectoryError(
            f"{label} runtime storage identity cannot be sealed"
        ) from error
    if pointer <= 0 or nbytes <= 0:
        raise TQMosaicTrajectoryError(f"{label} storage is degenerate")
    return _TensorBinding(
        object_id=id(checked),
        storage_data_ptr=pointer,
        storage_nbytes=nbytes,
        storage_offset=int(checked.storage_offset()),
        storage_version=version,
        shape=tuple(map(int, checked.shape)),
        stride=tuple(map(int, checked.stride())),
        dtype=str(checked.dtype),
        device=str(checked.device),
        tensor_sha256=_tensor_sha256(checked, label=label),
    )


def _quick_assert_live(
    value: torch.Tensor, binding: _TensorBinding, *, label: str
) -> None:
    """Cheap per-step mutation check; terminal receipts also rehash bytes."""

    try:
        storage = _untyped_storage(value)
        live = (
            id(value),
            int(storage.data_ptr()),
            int(storage.nbytes()),
            int(value.storage_offset()),
            int(value._version),  # noqa: SLF001 - mutation seal
            tuple(map(int, value.shape)),
            tuple(map(int, value.stride())),
            str(value.dtype),
            str(value.device),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise TQMosaicTrajectoryError(f"{label} live seal cannot be read") from error
    expected = (
        binding.object_id,
        binding.storage_data_ptr,
        binding.storage_nbytes,
        binding.storage_offset,
        binding.storage_version,
        binding.shape,
        binding.stride,
        binding.dtype,
        binding.device,
    )
    if live != expected:
        raise TQMosaicTrajectoryError(f"{label} changed after construction")


def _full_assert_live(
    value: torch.Tensor, binding: _TensorBinding, *, label: str
) -> dict[str, Any]:
    _quick_assert_live(value, binding, label=label)
    if _tensor_sha256(value, label=f"live {label}") != binding.tensor_sha256:
        raise TQMosaicTrajectoryError(f"{label} bytes changed after construction")
    return binding.portable()


def _owned_snapshot(value: torch.Tensor, *, label: str) -> torch.Tensor:
    checked = _validate_state_tensor(value, label=label)
    result = checked.detach().clone(memory_format=torch.contiguous_format)
    _validate_state_tensor(result, label=f"owned {label}")
    return result


def _plain_timestep(value: Any, *, label: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.device.type == "meta":
            raise TQMosaicTrajectoryError(f"{label} must be one scalar timestep")
        value = value.detach().to(device="cpu").item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TQMosaicTrajectoryError(f"{label} must be an integer timestep")
    integer = int(value)
    if float(value) != float(integer) or integer < 0:
        raise TQMosaicTrajectoryError(f"{label} must be a nonnegative integer")
    return integer


def _plain_sigma(value: Any, *, label: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.device.type == "meta":
            raise TQMosaicTrajectoryError(f"{label} must be one scalar sigma")
        value = value.detach().to(device="cpu").item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TQMosaicTrajectoryError(f"{label} must be a scalar sigma")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TQMosaicTrajectoryError(f"{label} must be finite and nonnegative")
    try:
        # The official scheduler exposes float32 sigmas.  Canonicalizing here
        # accepts a faithful decimal spelling while preserving bit-exact
        # comparison against the pinned float32 schedule below.
        return float(struct.unpack(">f", struct.pack(">f", result))[0])
    except (OverflowError, struct.error) as error:
        raise TQMosaicTrajectoryError(f"{label} is not finite float32") from error


def _validate_schedule_metadata(step_index: int, timestep: int, sigma: float) -> None:
    expected_timestep = PINNED_TIMESTEPS[step_index]
    expected_sigma_hex = PINNED_SIGMA_FLOAT32_HEX[step_index]
    if timestep != expected_timestep:
        raise TQMosaicTrajectoryError(
            f"pre-step {step_index} timestep differs: {timestep} != {expected_timestep}"
        )
    observed_sigma_hex = struct.pack(">f", sigma).hex()
    if observed_sigma_hex != expected_sigma_hex:
        raise TQMosaicTrajectoryError(
            f"pre-step {step_index} sigma differs at float32 bits: "
            f"{observed_sigma_hex} != {expected_sigma_hex}"
        )


def _schedule_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes([dict(row) for row in rows])
    ).hexdigest()


class ActualTrajectoryCaptureV1:
    """State machine placed immediately around each official scheduler call."""

    def __init__(self) -> None:
        self._next_step = 0
        self._pending: tuple[int, int, float] | None = None
        self._previous_return: torch.Tensor | None = None
        self._previous_return_binding: _TensorBinding | None = None
        self._shape: tuple[int, ...] | None = None
        self._device: str | None = None
        self._schedule: list[dict[str, Any]] = []
        self._anchors: list[torch.Tensor] = []
        self._initial_sha256: str | None = None
        self._terminal_sha256: str | None = None
        self._finalized = False

    def before_scheduler_step(
        self,
        *,
        step_index: int,
        timestep: Any,
        sigma: Any,
        state: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise TQMosaicTrajectoryError("base trajectory capture is finalized")
        if self._pending is not None:
            raise TQMosaicTrajectoryError("repeated pre-step before scheduler output")
        if type(step_index) is not int or step_index != self._next_step:
            raise TQMosaicTrajectoryError(
                f"out-of-order pre-step: expected {self._next_step}, got {step_index!r}"
            )
        if step_index >= EXACT_SCHEDULER_CALLS:
            raise TQMosaicTrajectoryError("more than 40 scheduler calls were attempted")
        checked = _validate_state_tensor(
            state,
            label=f"base pre-step {step_index} state",
            expected_shape=self._shape,
            expected_device=self._device,
        )
        if self._shape is None:
            self._shape = tuple(map(int, checked.shape))
            self._device = str(checked.device)
            self._initial_sha256 = _tensor_sha256(
                checked, label="base initial pre-step state"
            )
        if self._previous_return is not None:
            assert self._previous_return_binding is not None
            _full_assert_live(
                self._previous_return,
                self._previous_return_binding,
                label=f"base output {step_index - 1}",
            )
            if checked is not self._previous_return:
                raise TQMosaicTrajectoryError(
                    "sampler did not feed the exact intercepted output object "
                    f"from step {step_index - 1} into pre-step {step_index}"
                )
        plain_timestep = _plain_timestep(
            timestep, label=f"base pre-step {step_index} timestep"
        )
        plain_sigma = _plain_sigma(sigma, label=f"base pre-step {step_index} sigma")
        if self._schedule:
            prior = self._schedule[-1]
            if plain_timestep >= int(prior["timestep"]):
                raise TQMosaicTrajectoryError("base timesteps are not strictly descending")
            if plain_sigma >= float(prior["sigma"]):
                raise TQMosaicTrajectoryError("base sigmas are not strictly descending")
        _validate_schedule_metadata(step_index, plain_timestep, plain_sigma)
        self._schedule.append(
            {
                "step_index": step_index,
                "timestep": plain_timestep,
                "sigma": plain_sigma,
            }
        )
        if step_index in _ANCHOR_POSITION_BY_PRE_STEP:
            self._anchors.append(
                _owned_snapshot(checked, label=f"base pre-step {step_index} state")
            )
        self._pending = (step_index, plain_timestep, plain_sigma)

    def after_scheduler_step(
        self, *, step_index: int, next_state: torch.Tensor
    ) -> torch.Tensor:
        if self._finalized:
            raise TQMosaicTrajectoryError("base trajectory capture is finalized")
        if self._pending is None:
            raise TQMosaicTrajectoryError("scheduler output arrived without a pre-step")
        pending_index, _timestep, _sigma = self._pending
        if type(step_index) is not int or step_index != pending_index:
            raise TQMosaicTrajectoryError(
                f"out-of-order post-step: expected {pending_index}, got {step_index!r}"
            )
        checked = _validate_state_tensor(
            next_state,
            label=f"base scheduler output {step_index}",
            expected_shape=self._shape,
            expected_device=self._device,
        )
        self._previous_return = checked
        self._previous_return_binding = _bind_tensor(
            checked, label=f"base scheduler output {step_index}"
        )
        if step_index == EXACT_SCHEDULER_CALLS - 1:
            self._terminal_sha256 = _tensor_sha256(
                checked, label="base terminal scheduler output"
            )
        self._pending = None
        self._next_step += 1
        # Capture is observational: never clone or modify the official output.
        return checked

    def finalize(self) -> "CapturedActualTrajectoryV1":
        if self._finalized:
            raise TQMosaicTrajectoryError("base trajectory capture is already finalized")
        if self._pending is not None:
            raise TQMosaicTrajectoryError("base trajectory ended before pending post-step")
        if self._next_step != EXACT_SCHEDULER_CALLS:
            raise TQMosaicTrajectoryError(
                f"base trajectory has {self._next_step} scheduler calls, expected 40"
            )
        if (
            len(self._schedule) != EXACT_SCHEDULER_CALLS
            or len(self._anchors) != len(CAPTURE_PRE_STEP_INDICES)
            or self._shape is None
            or self._device is None
            or self._initial_sha256 is None
            or self._terminal_sha256 is None
        ):
            raise TQMosaicTrajectoryError("base trajectory closure is incomplete")
        assert self._previous_return is not None
        assert self._previous_return_binding is not None
        terminal = _full_assert_live(
            self._previous_return,
            self._previous_return_binding,
            label="base terminal scheduler output",
        )
        if terminal["tensor_sha256"] != self._terminal_sha256:
            raise TQMosaicTrajectoryError("base terminal state bytes changed")
        self._finalized = True
        return CapturedActualTrajectoryV1(
            token=_CONSTRUCTION_TOKEN,
            schedule=tuple(dict(row) for row in self._schedule),
            states=tuple(self._anchors),
            initial_state_sha256=self._initial_sha256,
            terminal_state_sha256=self._terminal_sha256,
        )


class CapturedActualTrajectoryV1:
    """Mutation-sealed base trajectory with three actual pre-step states."""

    def __init__(
        self,
        *,
        token: object,
        schedule: tuple[dict[str, Any], ...],
        states: tuple[torch.Tensor, ...],
        initial_state_sha256: str,
        terminal_state_sha256: str,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TQMosaicTrajectoryError("captured trajectory constructor is private")
        if len(schedule) != EXACT_SCHEDULER_CALLS or len(states) != 3:
            raise TQMosaicTrajectoryError("captured trajectory geometry differs")
        self._schedule = tuple(MappingProxyType(dict(row)) for row in schedule)
        self._states = states
        self._bindings = tuple(
            _bind_tensor(value, label=f"captured state {index}")
            for index, value in zip(CAPTURE_PRE_STEP_INDICES, states, strict=True)
        )
        self._initial_state_sha256 = initial_state_sha256
        self._terminal_state_sha256 = terminal_state_sha256
        self._control_binding = (
            id(self._schedule),
            _schedule_digest(self._schedule),
            self._initial_state_sha256,
            self._terminal_state_sha256,
        )
        self._construction_digest = self.receipt()["receipt_digest"]

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        return self._states

    @property
    def schedule(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._schedule)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._bindings[0].shape

    @property
    def device(self) -> str:
        return self._bindings[0].device

    @property
    def initial_state_sha256(self) -> str:
        return self._initial_state_sha256

    @property
    def terminal_state_sha256(self) -> str:
        return self._terminal_state_sha256

    def _assert_live(self, *, full: bool) -> None:
        live_control = (
            id(self._schedule),
            _schedule_digest(self._schedule),
            self._initial_state_sha256,
            self._terminal_state_sha256,
        )
        if live_control != self._control_binding:
            raise TQMosaicTrajectoryError(
                "captured trajectory control changed after construction"
            )
        for index, value, binding in zip(
            CAPTURE_PRE_STEP_INDICES, self._states, self._bindings, strict=True
        ):
            if full:
                _full_assert_live(value, binding, label=f"captured state {index}")
            else:
                _quick_assert_live(value, binding, label=f"captured state {index}")

    def receipt(self) -> dict[str, Any]:
        if hasattr(self, "_control_binding"):
            self._assert_live(full=False)
        state_rows = []
        for position, (index, value, binding) in enumerate(
            zip(CAPTURE_PRE_STEP_INDICES, self._states, self._bindings, strict=True)
        ):
            timestep, sigma = _ANCHOR_BY_PRE_STEP[index]
            state_rows.append(
                {
                    "position": position,
                    "pre_step_index": index,
                    "timestep": timestep,
                    "sigma": sigma,
                    "state": _full_assert_live(
                        value, binding, label=f"captured state {index}"
                    ),
                }
            )
        unsigned = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "sampler_owned_by_this_module": False,
            "scheduler_callback_authority": False,
            "exact_scheduler_calls": EXACT_SCHEDULER_CALLS,
            "capture_coordinate": "actual_pre_scheduler_step_state",
            "schedule": [dict(row) for row in self._schedule],
            "schedule_sha256": PINNED_SCHEDULE_SHA256,
            "observed_schedule_rows_sha256": _schedule_digest(self._schedule),
            "initial_state_sha256": self._initial_state_sha256,
            "terminal_state_sha256": self._terminal_state_sha256,
            "captured_states": state_rows,
            "semantic_success_assessed": False,
            "scientific_claim_authorized": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        sealed = _seal(unsigned)
        construction_digest = getattr(self, "_construction_digest", None)
        if (
            construction_digest is not None
            and sealed["receipt_digest"] != construction_digest
        ):
            raise TQMosaicTrajectoryError(
                "captured trajectory receipt changed after construction"
            )
        return sealed


def _l2_norm_fp64(value: torch.Tensor, *, label: str) -> float:
    norm = torch.linalg.vector_norm(value.detach().to(dtype=torch.float64)).item()
    result = float(norm)
    if not math.isfinite(result) or result <= _MINIMUM_NORM:
        raise TQMosaicTrajectoryError(f"{label} L2 norm is degenerate")
    return result


def _relative_doses() -> tuple[float, ...]:
    denominator = math.sqrt(sum(weight * weight for weight in TRAJECTORY_WEIGHTS))
    doses = tuple(
        TOTAL_RELATIVE_L2_DOSE * weight / denominator
        for weight in TRAJECTORY_WEIGHTS
    )
    if not math.isclose(
        math.sqrt(sum(value * value for value in doses)),
        TOTAL_RELATIVE_L2_DOSE,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise TQMosaicTrajectoryError("fixed trajectory dose constants differ")
    return doses


FIXED_RELATIVE_L2_DOSES = _relative_doses()


class TQMosaicTrajectoryInterventionV1:
    """Immutable three-state direction and fixed-dose replay factory."""

    def __init__(
        self,
        *,
        token: object,
        capture: CapturedActualTrajectoryV1,
        state_vjps: tuple[torch.Tensor, ...],
        deltas: tuple[torch.Tensor, ...],
        measured_doses: tuple[float, ...],
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TQMosaicTrajectoryError("trajectory intervention constructor is private")
        self._capture = capture
        self._state_vjps = state_vjps
        self._deltas = deltas
        self._measured_doses = measured_doses
        self._vjp_bindings = tuple(
            _bind_tensor(value, label=f"state VJP {index}")
            for index, value in zip(CAPTURE_PRE_STEP_INDICES, state_vjps, strict=True)
        )
        self._delta_bindings = tuple(
            _bind_tensor(value, label=f"state delta {index}")
            for index, value in zip(CAPTURE_PRE_STEP_INDICES, deltas, strict=True)
        )
        self._construction_digest = self.receipt()["receipt_digest"]

    @property
    def capture(self) -> CapturedActualTrajectoryV1:
        return self._capture

    @property
    def state_vjps(self) -> tuple[torch.Tensor, ...]:
        return self._state_vjps

    @property
    def deltas(self) -> tuple[torch.Tensor, ...]:
        return self._deltas

    @property
    def relative_l2_doses(self) -> tuple[float, ...]:
        return FIXED_RELATIVE_L2_DOSES

    def _assert_live(self, *, full: bool) -> None:
        self._capture._assert_live(full=full)
        for role, values, bindings in (
            ("state VJP", self._state_vjps, self._vjp_bindings),
            ("state delta", self._deltas, self._delta_bindings),
        ):
            for index, value, binding in zip(
                CAPTURE_PRE_STEP_INDICES, values, bindings, strict=True
            ):
                if full:
                    _full_assert_live(value, binding, label=f"{role} {index}")
                else:
                    _quick_assert_live(value, binding, label=f"{role} {index}")

    def receipt(self) -> dict[str, Any]:
        capture_receipt = self._capture.receipt()
        rows = []
        for position, index in enumerate(CAPTURE_PRE_STEP_INDICES):
            rows.append(
                {
                    "position": position,
                    "pre_step_index": index,
                    "inject_after_step_index": INJECT_AFTER_STEP_INDICES[position],
                    "timestep": CAPTURE_TIMESTEPS[position],
                    "sigma": CAPTURE_SIGMAS[position],
                    "weight": TRAJECTORY_WEIGHTS[position],
                    "target_relative_l2_dose": FIXED_RELATIVE_L2_DOSES[position],
                    "measured_relative_l2_dose": self._measured_doses[position],
                    "base_state": _full_assert_live(
                        self._capture.states[position],
                        self._capture._bindings[position],
                        label=f"captured state {index}",
                    ),
                    "state_vjp": _full_assert_live(
                        self._state_vjps[position],
                        self._vjp_bindings[position],
                        label=f"state VJP {index}",
                    ),
                    "delta": _full_assert_live(
                        self._deltas[position],
                        self._delta_bindings[position],
                        label=f"state delta {index}",
                    ),
                }
            )
        unsigned = {
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "base_capture_receipt_digest": capture_receipt["receipt_digest"],
            "direction_coordinate": "actual_pre_step_state_vjp",
            "injection_coordinate": "scheduler_post_step_output_for_next_state",
            "total_relative_l2_dose_definition": (
                "sqrt(sum_i((||delta_i||_2/||base_state_i||_2)^2))"
            ),
            "total_relative_l2_dose": TOTAL_RELATIVE_L2_DOSE,
            "fixed_weights": list(TRAJECTORY_WEIGHTS),
            "fixed_relative_l2_doses": list(FIXED_RELATIVE_L2_DOSES),
            "rows": rows,
            "allowed_signs": [-1, 0, 1],
            "seed_input_authorized": False,
            "dose_input_authorized": False,
            "arm_selection_authorized": False,
            "mask_input_authorized": False,
            "track_input_authorized": False,
            "pose_input_authorized": False,
            "optical_flow_input_authorized": False,
            "callback_authority": False,
            "semantic_success_assessed": False,
            "scientific_claim_authorized": False,
            "optimizer_authorized": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        sealed = _seal(unsigned)
        construction_digest = getattr(self, "_construction_digest", None)
        if (
            construction_digest is not None
            and sealed["receipt_digest"] != construction_digest
        ):
            raise TQMosaicTrajectoryError(
                "trajectory intervention receipt changed after construction"
            )
        return sealed

    def new_replay(self, *, sign: int) -> "TQMosaicTrajectoryReplayV1":
        if type(sign) is not int or sign not in (-1, 0, 1):
            raise TQMosaicTrajectoryError("replay sign must be exactly -1, 0, or +1")
        self._assert_live(full=True)
        return TQMosaicTrajectoryReplayV1(
            token=_CONSTRUCTION_TOKEN,
            plan=self,
            sign=sign,
        )


def build_trajectory_intervention_v1(
    *,
    capture: CapturedActualTrajectoryV1,
    state_vjps: Sequence[torch.Tensor],
) -> TQMosaicTrajectoryInterventionV1:
    """Build the sole v1 intervention; there is no seed or dose argument."""

    if type(capture) is not CapturedActualTrajectoryV1:
        raise TQMosaicTrajectoryError("capture must be a finalized v1 base trajectory")
    capture._assert_live(full=True)
    if isinstance(state_vjps, torch.Tensor) or len(state_vjps) != 3:
        raise TQMosaicTrajectoryError("exactly three state VJPs are required")
    owned_vjps: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    measured: list[float] = []
    for position, (index, base_state, raw_vjp, target_dose) in enumerate(
        zip(
            CAPTURE_PRE_STEP_INDICES,
            capture.states,
            state_vjps,
            FIXED_RELATIVE_L2_DOSES,
            strict=True,
        )
    ):
        checked_vjp = _validate_state_tensor(
            raw_vjp,
            label=f"state VJP {index}",
            expected_shape=capture.shape,
            expected_device=capture.device,
        )
        owned_vjp = _owned_snapshot(checked_vjp, label=f"state VJP {index}")
        base_norm = _l2_norm_fp64(base_state, label=f"base state {index}")
        vjp_norm = _l2_norm_fp64(owned_vjp, label=f"state VJP {index}")
        scale = target_dose * base_norm / vjp_norm
        delta = (owned_vjp * scale).detach().contiguous()
        _validate_state_tensor(
            delta,
            label=f"state delta {index}",
            expected_shape=capture.shape,
            expected_device=capture.device,
        )
        delta_norm = _l2_norm_fp64(delta, label=f"state delta {index}")
        measured_dose = delta_norm / base_norm
        if not math.isclose(
            measured_dose,
            target_dose,
            rel_tol=_DOSE_RELATIVE_TOLERANCE,
            abs_tol=0.0,
        ):
            raise TQMosaicTrajectoryError(
                f"FP32 state delta {index} failed fixed-dose realization"
            )
        owned_vjps.append(owned_vjp)
        deltas.append(delta)
        measured.append(measured_dose)
    measured_total = math.sqrt(sum(value * value for value in measured))
    if not math.isclose(
        measured_total,
        TOTAL_RELATIVE_L2_DOSE,
        rel_tol=_DOSE_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    ):
        raise TQMosaicTrajectoryError("joint FP32 trajectory dose differs")
    return TQMosaicTrajectoryInterventionV1(
        token=_CONSTRUCTION_TOKEN,
        capture=capture,
        state_vjps=tuple(owned_vjps),
        deltas=tuple(deltas),
        measured_doses=tuple(measured),
    )


class TQMosaicTrajectoryReplayV1:
    """One strict sign arm around exactly 40 official scheduler calls."""

    def __init__(
        self,
        *,
        token: object,
        plan: TQMosaicTrajectoryInterventionV1,
        sign: int,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise TQMosaicTrajectoryError("trajectory replay constructor is private")
        self._plan = plan
        self._sign = sign
        self._construction_plan_object_id = id(plan)
        self._construction_sign = sign
        self._plan_digest = plan.receipt()["receipt_digest"]
        self._next_step = 0
        self._pending: tuple[int, int, float] | None = None
        self._previous_return: torch.Tensor | None = None
        self._previous_binding: _TensorBinding | None = None
        self._initial_sha256: str | None = None
        self._terminal_sha256: str | None = None
        self._anchors: list[dict[str, Any]] = []
        self._injections: list[dict[str, Any]] = []
        self._outputs_returned_by_identity = 0
        self._finalized = False

    @property
    def sign(self) -> int:
        self._assert_control_integrity()
        return self._sign

    def _assert_control_integrity(self) -> None:
        if (
            id(self._plan) != self._construction_plan_object_id
            or self._sign != self._construction_sign
            or self._sign not in (-1, 0, 1)
        ):
            raise TQMosaicTrajectoryError(
                "trajectory replay control changed after construction"
            )

    def before_scheduler_step(
        self,
        *,
        step_index: int,
        timestep: Any,
        sigma: Any,
        state: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise TQMosaicTrajectoryError("trajectory replay is finalized")
        self._assert_control_integrity()
        self._plan._assert_live(full=False)
        if self._pending is not None:
            raise TQMosaicTrajectoryError("repeated replay pre-step")
        if type(step_index) is not int or step_index != self._next_step:
            raise TQMosaicTrajectoryError(
                f"out-of-order replay pre-step: expected {self._next_step}, "
                f"got {step_index!r}"
            )
        if step_index >= EXACT_SCHEDULER_CALLS:
            raise TQMosaicTrajectoryError("replay attempted more than 40 calls")
        checked = _validate_state_tensor(
            state,
            label=f"replay pre-step {step_index} state",
            expected_shape=self._plan.capture.shape,
            expected_device=self._plan.capture.device,
        )
        if self._previous_return is not None:
            assert self._previous_binding is not None
            _full_assert_live(
                self._previous_return,
                self._previous_binding,
                label=f"replay output {step_index - 1}",
            )
            if checked is not self._previous_return:
                raise TQMosaicTrajectoryError(
                    "sampler did not feed the exact intercepted replay output object "
                    f"from step {step_index - 1} into pre-step {step_index}"
                )
        plain_timestep = _plain_timestep(
            timestep, label=f"replay pre-step {step_index} timestep"
        )
        plain_sigma = _plain_sigma(
            sigma, label=f"replay pre-step {step_index} sigma"
        )
        expected_schedule = self._plan.capture.schedule[step_index]
        if (
            plain_timestep != expected_schedule["timestep"]
            or plain_sigma != expected_schedule["sigma"]
        ):
            raise TQMosaicTrajectoryError(
                f"replay schedule differs from captured base at step {step_index}"
            )
        if step_index == 0:
            self._initial_sha256 = _tensor_sha256(
                checked, label="replay initial pre-step state"
            )
            if self._initial_sha256 != self._plan.capture.initial_state_sha256:
                raise TQMosaicTrajectoryError(
                    "replay initial state differs from captured base noise/state"
                )
        if step_index in _ANCHOR_POSITION_BY_PRE_STEP:
            position = _ANCHOR_POSITION_BY_PRE_STEP[step_index]
            observed_sha = _tensor_sha256(
                checked, label=f"replay pre-step {step_index} state"
            )
            base_sha = self._plan.capture._bindings[position].tensor_sha256
            if self._sign == 0 and observed_sha != base_sha:
                raise TQMosaicTrajectoryError(
                    f"zero-sign replay state {step_index} differs bytewise from base"
                )
            self._anchors.append(
                {
                    "pre_step_index": step_index,
                    "state_sha256": observed_sha,
                    "base_state_sha256": base_sha,
                    "zero_sign_base_bytes_equal": (
                        observed_sha == base_sha if self._sign == 0 else None
                    ),
                }
            )
        self._pending = (step_index, plain_timestep, plain_sigma)

    def after_scheduler_step(
        self, *, step_index: int, next_state: torch.Tensor
    ) -> torch.Tensor:
        if self._finalized:
            raise TQMosaicTrajectoryError("trajectory replay is finalized")
        self._assert_control_integrity()
        self._plan._assert_live(full=False)
        if self._pending is None:
            raise TQMosaicTrajectoryError("replay output arrived without a pre-step")
        pending_index, _timestep, _sigma = self._pending
        if type(step_index) is not int or step_index != pending_index:
            raise TQMosaicTrajectoryError(
                f"out-of-order replay post-step: expected {pending_index}, "
                f"got {step_index!r}"
            )
        raw = _validate_state_tensor(
            next_state,
            label=f"replay scheduler output {step_index}",
            expected_shape=self._plan.capture.shape,
            expected_device=self._plan.capture.device,
        )
        position = _ANCHOR_POSITION_BY_POST_STEP.get(step_index)
        if position is None or self._sign == 0:
            # This branch is intentionally free of clone/add/mul.  In
            # particular, all 40 sign=0 outputs preserve object identity and
            # byte identity with the official scheduler output.
            result = raw
        else:
            delta = self._plan.deltas[position]
            _full_assert_live(
                delta,
                self._plan._delta_bindings[position],
                label=f"state delta {CAPTURE_PRE_STEP_INDICES[position]}",
            )
            result = (raw + float(self._sign) * delta).detach().contiguous()
            _validate_state_tensor(
                result,
                label=f"injected replay output {step_index}",
                expected_shape=self._plan.capture.shape,
                expected_device=self._plan.capture.device,
            )
        if position is not None:
            raw_sha = _tensor_sha256(
                raw, label=f"raw scheduler output {step_index}"
            )
            result_sha = _tensor_sha256(
                result, label=f"intercepted scheduler output {step_index}"
            )
            base_sha = self._plan.capture._bindings[position].tensor_sha256
            if position == 0 and raw_sha != base_sha:
                raise TQMosaicTrajectoryError(
                    "first pre-intervention scheduler state differs from base"
                )
            if self._sign == 0 and (result is not raw or result_sha != raw_sha):
                raise TQMosaicTrajectoryError(
                    "zero-sign replay did not preserve scheduler output exactly"
                )
            self._injections.append(
                {
                    "inject_after_step_index": step_index,
                    "next_pre_step_index": CAPTURE_PRE_STEP_INDICES[position],
                    "sign": self._sign,
                    "raw_state_sha256": raw_sha,
                    "delta_sha256": self._plan._delta_bindings[position].tensor_sha256,
                    "returned_state_sha256": result_sha,
                    "returned_original_object": result is raw,
                    "zero_sign_original_bytes": (
                        result_sha == raw_sha if self._sign == 0 else None
                    ),
                }
            )
        self._previous_return = result
        if result is raw:
            self._outputs_returned_by_identity += 1
        self._previous_binding = _bind_tensor(
            result, label=f"replay returned output {step_index}"
        )
        if step_index == EXACT_SCHEDULER_CALLS - 1:
            self._terminal_sha256 = self._previous_binding.tensor_sha256
        self._pending = None
        self._next_step += 1
        return result

    def finalize(self) -> dict[str, Any]:
        self._assert_control_integrity()
        if self._finalized:
            raise TQMosaicTrajectoryError("trajectory replay is already finalized")
        if self._pending is not None:
            raise TQMosaicTrajectoryError("trajectory replay ended before post-step")
        if self._next_step != EXACT_SCHEDULER_CALLS:
            raise TQMosaicTrajectoryError(
                f"trajectory replay has {self._next_step} scheduler calls, expected 40"
            )
        if (
            len(self._anchors) != len(CAPTURE_PRE_STEP_INDICES)
            or len(self._injections) != len(INJECT_AFTER_STEP_INDICES)
            or self._initial_sha256 is None
            or self._terminal_sha256 is None
        ):
            raise TQMosaicTrajectoryError("trajectory replay closure is incomplete")
        assert self._previous_return is not None
        assert self._previous_binding is not None
        terminal = _full_assert_live(
            self._previous_return,
            self._previous_binding,
            label="replay terminal scheduler output",
        )
        if terminal["tensor_sha256"] != self._terminal_sha256:
            raise TQMosaicTrajectoryError("replay terminal state bytes changed")
        self._plan._assert_live(full=True)
        if self._sign == 0 and (
            self._terminal_sha256 != self._plan.capture.terminal_state_sha256
        ):
            raise TQMosaicTrajectoryError(
                "zero-sign replay terminal bytes differ from captured base"
            )
        self._finalized = True
        sealed = self.receipt()
        self._final_receipt_digest = sealed["receipt_digest"]
        return sealed

    def receipt(self) -> dict[str, Any]:
        self._assert_control_integrity()
        if not self._finalized:
            raise TQMosaicTrajectoryError("unfinished replay has no receipt")
        assert self._previous_return is not None
        assert self._previous_binding is not None
        terminal = _full_assert_live(
            self._previous_return,
            self._previous_binding,
            label="closed replay terminal scheduler output",
        )
        if terminal["tensor_sha256"] != self._terminal_sha256:
            raise TQMosaicTrajectoryError(
                "closed replay terminal state bytes changed"
            )
        plan_receipt = self._plan.receipt()
        if plan_receipt["receipt_digest"] != self._plan_digest:
            raise TQMosaicTrajectoryError("trajectory plan receipt changed during replay")
        unsigned = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "intervention_receipt_digest": self._plan_digest,
            "sign": self._sign,
            "scheduler_calls_observed": self._next_step,
            "exact_scheduler_calls_required": EXACT_SCHEDULER_CALLS,
            "schedule_sha256": PINNED_SCHEDULE_SHA256,
            "observed_schedule_rows_sha256": _schedule_digest(
                self._plan.capture.schedule
            ),
            "initial_state_sha256": self._initial_sha256,
            "terminal_state_sha256": self._terminal_sha256,
            "base_terminal_state_sha256": (
                self._plan.capture.terminal_state_sha256
            ),
            "anchor_observations": [dict(row) for row in self._anchors],
            "injection_observations": [dict(row) for row in self._injections],
            "zero_sign_all_scheduler_outputs_returned_by_identity": (
                self._outputs_returned_by_identity == EXACT_SCHEDULER_CALLS
                if self._sign == 0
                else None
            ),
            "scheduler_outputs_returned_by_identity_count": (
                self._outputs_returned_by_identity
            ),
            "semantic_success_assessed": False,
            "scientific_claim_authorized": False,
            "optimizer_authorized": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        sealed = _seal(unsigned)
        final_digest = getattr(self, "_final_receipt_digest", None)
        if final_digest is not None and sealed["receipt_digest"] != final_digest:
            raise TQMosaicTrajectoryError("trajectory replay receipt changed after closure")
        return sealed


__all__ = [
    "CAPTURE_PRE_STEP_INDICES",
    "CAPTURE_SIGMAS",
    "CAPTURE_TIMESTEPS",
    "EXACT_SCHEDULER_CALLS",
    "FIXED_RELATIVE_L2_DOSES",
    "INJECT_AFTER_STEP_INDICES",
    "PINNED_SCHEDULE_SHA256",
    "PINNED_SIGMAS",
    "PINNED_SIGMA_FLOAT32_HEX",
    "PINNED_TIMESTEPS",
    "TOTAL_RELATIVE_L2_DOSE",
    "TRAJECTORY_WEIGHTS",
    "ActualTrajectoryCaptureV1",
    "CapturedActualTrajectoryV1",
    "TQMosaicTrajectoryError",
    "TQMosaicTrajectoryInterventionV1",
    "TQMosaicTrajectoryReplayV1",
    "build_trajectory_intervention_v1",
]

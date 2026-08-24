#!/usr/bin/env python3
"""Zero-training oracle projection for packed object trajectories.

This module is deliberately only a tensor/scheduler core.  It is not wired to
``infer_lora.py`` or to a production Bernini runner.  A caller supplies one or
more *oracle* rows.  Each row contains a clean packed target authority and a
strict binary element mask.  At every active UniPC step the selected flow
velocity is replaced by ``initial_noise - clean``; after the original UniPC
step the same elements are projected to
``(1 - next_sigma) * clean + next_sigma * initial_noise``.

The original UniPC step remains the sole numerical integrator.  Inactive steps
are exact native delegates: the original positional/keyword objects are passed
without cloning or replacement.  An all-zero intervention is not installed at
all.  These two bypasses are important controls, not optimizations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = "bernini-object-trajectory-unipc-projection-v1"
PACKED_CHANNELS = 64
FLOW_SHIFT = 5.0
_WRAPPER_MARKER = "_bernini_object_trajectory_projection_v1"


class ObjectTrajectoryProjectionError(RuntimeError):
    """Raised when the oracle projection contract cannot be proved."""


def tensor_core_contract() -> dict[str, Any]:
    """Return the immutable scope and numerical contract of this core."""

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "zero_training_oracle_tensor_core",
        "production_runner_integration": False,
        "renderer_abi_integration": False,
        "integrator": "original_unipc_scheduler_step",
        "prediction_type": "flow_prediction",
        "packed_layout": "B,N,64",
        "supported_packed_dtypes": [
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        ],
        "weight_policy": "strict_binary_0_or_1_v1",
        "fractional_weights_supported": False,
        "step_gate_coordinate": "next_sigma_after_native_step",
        "velocity_on_selected_elements": "initial_noise_minus_clean_authority",
        "post_step_on_selected_elements": (
            "(1-next_sigma)*clean_authority+next_sigma*initial_noise"
        ),
        "inactive_step_policy": (
            "exact_native_delegate_no_argument_clone_or_replacement"
        ),
        "all_zero_policy": "do_not_install_wrapper",
        "initial_noise_policy": (
            "explicit_exact_first_sample_or_lazy_clone_of_first_native_sample_no_rng"
        ),
        "row_overlap_policy": "allow_only_if_clean_values_are_exactly_equal",
        "terminal_sigma_policy": "positive_zero_required",
        "error_policy": "fail_closed_and_restore_step_wrapper",
    }


@dataclass(frozen=True)
class ProjectionRow:
    """One named clean authority, binary support, and deterministic step gate.

    ``projection_weights`` must have shape ``[B,N,1]`` (whole-token) or
    ``[B,N,64]`` (per-channel).  Bounds are inclusive and are evaluated against
    the *next* sigma, because that is the coordinate of the post-step sample.
    ``step_gates``, when supplied, must contain exactly one strict bool/0/1 per
    expected scheduler step and is intersected with the sigma interval.
    """

    name: str
    clean_packed: Any
    projection_weights: Any
    active_next_sigma_min: float | None = None
    active_next_sigma_max: float | None = None
    step_gates: tuple[bool | int, ...] | None = None


@dataclass(frozen=True)
class ProjectionStepRecord:
    """Tensor-free evidence for one successful original scheduler step."""

    step_index: int
    timestep: float
    sigma: float
    next_sigma: float
    cursor_before: int | None
    cursor_after: int
    projection_applied: bool
    active_rows: tuple[str, ...]
    inactive_rows: tuple[str, ...]
    selected_token_count: int
    selected_element_count: int
    total_element_count: int
    original_scheduler_step_calls: int
    exact_native_delegate_no_argument_clone: bool
    initial_noise_snapshot_created_this_step: bool
    initial_sample_matches_registered_noise: bool
    selected_velocity_exact: bool | None
    unselected_velocity_exact: bool | None
    selected_post_step_exact: bool | None
    unselected_post_step_exact: bool | None


@dataclass
class ProjectionTrace:
    """Serializable, tensor-free audit of one installed projection context."""

    expected_steps: int
    source_token_count: int
    target_token_count: int
    batch_size: int
    packed_channels: int
    clean_dtype: str
    clean_device: str
    initial_noise_dtype: str
    initial_noise_device: str
    initial_noise_registration: str
    row_specs: tuple[dict[str, Any], ...]
    globally_selected_token_count: int
    globally_selected_element_count: int
    globally_enabled: bool
    records: list[ProjectionStepRecord] = field(default_factory=list)
    wrapper_installed: bool = False
    wrapper_restored: bool = False
    initial_noise_verified: bool = False
    initial_noise_captured_from_first_native_sample: bool = False
    finalized: bool = False

    def as_dict(self) -> dict[str, Any]:
        if not self.finalized:
            raise ObjectTrajectoryProjectionError(
                "object trajectory projection trace is not finalized"
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "contract": tensor_core_contract(),
            "zero_training_oracle": True,
            "production_runner_integration": False,
            "dimensions": {
                "source_reference": [
                    self.batch_size,
                    self.source_token_count,
                    self.packed_channels,
                ],
                "target_sampler": [
                    self.batch_size,
                    self.target_token_count,
                    self.packed_channels,
                ],
            },
            "clean_dtype": self.clean_dtype,
            "clean_device": self.clean_device,
            "initial_noise_dtype": self.initial_noise_dtype,
            "initial_noise_device": self.initial_noise_device,
            "initial_noise_registration": self.initial_noise_registration,
            "rows": [dict(item) for item in self.row_specs],
            "globally_selected_token_count": self.globally_selected_token_count,
            "globally_selected_element_count": self.globally_selected_element_count,
            "globally_enabled": self.globally_enabled,
            "wrapper_installed": self.wrapper_installed,
            "wrapper_restored": self.wrapper_restored,
            "initial_noise_verified": self.initial_noise_verified,
            "initial_noise_captured_from_first_native_sample": (
                self.initial_noise_captured_from_first_native_sample
            ),
            "step_count": len(self.records),
            "expected_steps": self.expected_steps,
            "steps": [asdict(record) for record in self.records],
            "finalized": self.finalized,
        }


@dataclass(frozen=True)
class _RegisteredRow:
    name: str
    clean: Any
    mask: Any
    active_next_sigma_min: float | None
    active_next_sigma_max: float | None
    step_gates: tuple[bool, ...] | None
    selected_token_count: int
    selected_element_count: int
    original_weight_shape: tuple[int, ...]

    def active(self, *, step_index: int, next_sigma: float) -> bool:
        if (
            self.active_next_sigma_min is not None
            and next_sigma < self.active_next_sigma_min
        ):
            return False
        if (
            self.active_next_sigma_max is not None
            and next_sigma > self.active_next_sigma_max
        ):
            return False
        if self.step_gates is not None and not self.step_gates[step_index]:
            return False
        return True


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ObjectTrajectoryProjectionError(f"{label} must be a positive int")
    return value


def _scalar_float(value: Any, *, label: str) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) != 1:
            raise ObjectTrajectoryProjectionError(f"{label} must be scalar")
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except ObjectTrajectoryProjectionError:
        raise
    except Exception as error:
        raise ObjectTrajectoryProjectionError(
            f"{label} must be a numeric scalar"
        ) from error
    if not math.isfinite(result):
        raise ObjectTrajectoryProjectionError(f"{label} must be finite")
    return result


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise ObjectTrajectoryProjectionError(
                f"UniPC scheduler config lacks {name}"
            )
        return config[name]
    if not hasattr(config, name):
        raise ObjectTrajectoryProjectionError(
            f"UniPC scheduler config lacks {name}"
        )
    return getattr(config, name)


def _audit_scheduler_contract(
    scheduler: Any, *, expected_steps: int, permit_installed_wrapper: bool = False
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if scheduler.__class__.__name__ != "UniPCMultistepScheduler":
        raise ObjectTrajectoryProjectionError(
            "object trajectory projection requires UniPCMultistepScheduler"
        )
    config = getattr(scheduler, "config", None)
    if config is None:
        raise ObjectTrajectoryProjectionError("UniPC scheduler lacks config")
    required = {
        "_class_name": "UniPCMultistepScheduler",
        "prediction_type": "flow_prediction",
        "use_flow_sigmas": True,
        "predict_x0": True,
        "final_sigmas_type": "zero",
    }
    for name, expected in required.items():
        actual = _config_value(config, name)
        matches = actual is expected if type(expected) is bool else actual == expected
        if not matches:
            raise ObjectTrajectoryProjectionError(
                f"UniPC scheduler {name} differs: expected {expected!r}, got {actual!r}"
            )
    flow_shift = _scalar_float(
        _config_value(config, "flow_shift"), label="UniPC scheduler flow_shift"
    )
    if flow_shift != FLOW_SHIFT:
        raise ObjectTrajectoryProjectionError(
            f"UniPC scheduler flow_shift differs: {flow_shift} != {FLOW_SHIFT}"
        )
    original_step = getattr(scheduler, "step", None)
    if not callable(original_step):
        raise ObjectTrajectoryProjectionError("UniPC scheduler.step must be callable")
    if getattr(original_step, _WRAPPER_MARKER, False) and not permit_installed_wrapper:
        raise ObjectTrajectoryProjectionError(
            "UniPC scheduler already has an object trajectory projection"
        )

    sigmas = getattr(scheduler, "sigmas", None)
    timesteps = getattr(scheduler, "timesteps", None)
    if sigmas is None or timesteps is None:
        raise ObjectTrajectoryProjectionError(
            "UniPC must expose runtime sigmas and timesteps before installation"
        )
    try:
        sigma_values = tuple(
            _scalar_float(value, label=f"UniPC sigma {index}")
            for index, value in enumerate(sigmas)
        )
        timestep_values = tuple(
            _scalar_float(value, label=f"UniPC timestep {index}")
            for index, value in enumerate(timesteps)
        )
    except TypeError as error:
        raise ObjectTrajectoryProjectionError(
            "UniPC sigmas and timesteps must be iterable"
        ) from error
    if len(sigma_values) != expected_steps + 1:
        raise ObjectTrajectoryProjectionError(
            "UniPC sigma count differs from expected_steps + 1"
        )
    if len(timestep_values) != expected_steps:
        raise ObjectTrajectoryProjectionError(
            "UniPC timestep count differs from expected_steps"
        )
    if (
        sigma_values[0] <= 0.0
        or sigma_values[-1] != 0.0
        or math.copysign(1.0, sigma_values[-1]) != 1.0
        or any(right >= left for left, right in zip(sigma_values, sigma_values[1:]))
    ):
        raise ObjectTrajectoryProjectionError(
            "UniPC sigmas must be strictly descending and end in positive zero"
        )
    return sigma_values, timestep_values


def _storage_key(value: Any) -> tuple[str, int | None, int]:
    try:
        pointer = int(value.untyped_storage().data_ptr())
    except Exception as error:
        raise ObjectTrajectoryProjectionError(
            "projection tensors must expose non-aliased dense storage"
        ) from error
    return value.device.type, value.device.index, pointer


def _validate_no_storage_aliases(values: Sequence[tuple[str, Any]]) -> None:
    owners: dict[tuple[str, int | None, int], str] = {}
    for label, value in values:
        key = _storage_key(value)
        if key in owners:
            raise ObjectTrajectoryProjectionError(
                f"tensor storage alias is forbidden: {owners[key]} and {label}"
            )
        owners[key] = label


def _validate_float_packed(
    value: Any,
    *,
    label: str,
    expected_shape: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production requires torch
        raise ObjectTrajectoryProjectionError("projection requires torch") from error
    if not isinstance(value, torch.Tensor):
        raise ObjectTrajectoryProjectionError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided or value.ndim != 3:
        raise ObjectTrajectoryProjectionError(f"{label} must be dense packed [B,N,64]")
    shape = tuple(int(item) for item in value.shape)
    if shape[0] <= 0 or shape[1] <= 0 or shape[2] != PACKED_CHANNELS:
        raise ObjectTrajectoryProjectionError(f"{label} must have shape [B,N,64]")
    if expected_shape is not None and shape != expected_shape:
        raise ObjectTrajectoryProjectionError(
            f"{label} shape differs: {shape} != {expected_shape}"
        )
    if not torch.is_floating_point(value) or value.is_complex():
        raise ObjectTrajectoryProjectionError(f"{label} must have a real floating dtype")
    if value.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise ObjectTrajectoryProjectionError(f"{label} floating dtype is unsupported")
    if value.requires_grad:
        raise ObjectTrajectoryProjectionError(f"{label} must be detached")
    if not value.is_contiguous():
        raise ObjectTrajectoryProjectionError(f"{label} must be contiguous")
    if not bool(torch.isfinite(value).all().item()):
        raise ObjectTrajectoryProjectionError(f"{label} must be finite")
    return shape


def _validate_binary_weights(
    weights: Any,
    *,
    batch_size: int,
    target_token_count: int,
    device: Any,
    label: str,
) -> tuple[Any, int, int, tuple[int, ...]]:
    import torch

    if not isinstance(weights, torch.Tensor):
        raise ObjectTrajectoryProjectionError(f"{label} must be a torch.Tensor")
    if weights.layout != torch.strided or weights.ndim != 3:
        raise ObjectTrajectoryProjectionError(
            f"{label} must have shape [B,N,1] or [B,N,64]"
        )
    shape = tuple(int(item) for item in weights.shape)
    if shape not in {
        (batch_size, target_token_count, 1),
        (batch_size, target_token_count, PACKED_CHANNELS),
    }:
        raise ObjectTrajectoryProjectionError(
            f"{label} must have shape [B,N,1] or [B,N,64]"
        )
    if weights.device != device:
        raise ObjectTrajectoryProjectionError(
            f"{label} device differs from clean authority"
        )
    if weights.requires_grad:
        raise ObjectTrajectoryProjectionError(f"{label} must be detached")
    if not weights.is_contiguous():
        raise ObjectTrajectoryProjectionError(f"{label} must be contiguous")
    if weights.dtype != torch.bool:
        if not torch.is_floating_point(weights) or weights.is_complex():
            raise ObjectTrajectoryProjectionError(
                f"{label} must be bool or real floating strict binary weights"
            )
        if not bool(torch.isfinite(weights).all().item()):
            raise ObjectTrajectoryProjectionError(f"{label} must be finite")
        binary = (weights == 0) | (weights == 1)
        if not bool(binary.all().item()):
            raise ObjectTrajectoryProjectionError(
                f"{label} contains fractional weights; v1 supports strict 0/1 only"
            )
    mask = weights.to(dtype=torch.bool)
    if shape[2] == 1:
        mask = mask.expand(batch_size, target_token_count, PACKED_CHANNELS)
    selected_elements = int(mask.count_nonzero().item())
    selected_tokens = int(mask.any(dim=2).count_nonzero().item())
    return mask, selected_tokens, selected_elements, shape


def _bound(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    result = _scalar_float(value, label=label)
    if result < 0.0:
        raise ObjectTrajectoryProjectionError(f"{label} must be non-negative")
    return result


def _step_gates(
    values: tuple[bool | int, ...] | None, *, expected_steps: int, label: str
) -> tuple[bool, ...] | None:
    if values is None:
        return None
    if type(values) is not tuple or len(values) != expected_steps:
        raise ObjectTrajectoryProjectionError(
            f"{label} must be a tuple with exactly expected_steps entries"
        )
    result: list[bool] = []
    for index, value in enumerate(values):
        if type(value) is bool:
            result.append(value)
        elif type(value) is int and value in (0, 1):
            result.append(bool(value))
        else:
            raise ObjectTrajectoryProjectionError(
                f"{label}[{index}] must be strict bool/0/1"
            )
    return tuple(result)


def _extract_step_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional and keyword:
        raise ObjectTrajectoryProjectionError(
            f"scheduler.step received duplicate {name}"
        )
    if positional:
        return args[index]
    if keyword:
        return kwargs[name]
    raise ObjectTrajectoryProjectionError(f"scheduler.step is missing {name}")


def _replace_model_output(
    args: Sequence[Any], kwargs: Mapping[str, Any], replacement: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if new_args:
        if "model_output" in new_kwargs:
            raise ObjectTrajectoryProjectionError(
                "scheduler.step received duplicate model_output"
            )
        new_args[0] = replacement
    elif "model_output" in new_kwargs:
        new_kwargs["model_output"] = replacement
    else:
        raise ObjectTrajectoryProjectionError(
            "scheduler.step is missing model_output"
        )
    return tuple(new_args), new_kwargs


def _same_storage(left: Any, right: Any) -> bool:
    return _storage_key(left) == _storage_key(right)


class _InstalledObjectTrajectoryProjection:
    def __init__(
        self,
        scheduler: Any,
        *,
        rows: Sequence[ProjectionRow],
        initial_noise: Any,
        source_token_count: int,
        target_token_count: int,
        expected_steps: int,
    ) -> None:
        import torch

        self.expected_steps = _positive_int(expected_steps, label="expected_steps")
        self.source_token_count = _positive_int(
            source_token_count, label="source_token_count"
        )
        self.target_token_count = _positive_int(
            target_token_count, label="target_token_count"
        )
        if self.source_token_count != self.target_token_count:
            raise ObjectTrajectoryProjectionError(
                "v1 requires equal source and target token counts"
            )
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise ObjectTrajectoryProjectionError("rows must be a non-empty sequence")
        if any(not isinstance(row, ProjectionRow) for row in rows):
            raise ObjectTrajectoryProjectionError(
                "every rows entry must be ProjectionRow"
            )

        self.scheduler = scheduler
        self.sigmas, self.timesteps = _audit_scheduler_contract(
            scheduler, expected_steps=self.expected_steps
        )

        first_shape = _validate_float_packed(
            rows[0].clean_packed, label=f"row {rows[0].name!r} clean_packed"
        )
        if first_shape[1] != self.target_token_count:
            raise ObjectTrajectoryProjectionError(
                "clean authority token count differs from target_token_count"
            )
        if initial_noise is None:
            noise_registration = "lazy_capture_first_native_sample"
            noise_device = rows[0].clean_packed.device
            noise_dtype = rows[0].clean_packed.dtype
        else:
            noise_shape = _validate_float_packed(
                initial_noise, label="initial_noise", expected_shape=first_shape
            )
            if noise_shape != first_shape:  # Defensive; expected_shape already proves it.
                raise ObjectTrajectoryProjectionError("initial_noise shape differs")
            if initial_noise.device != rows[0].clean_packed.device:
                raise ObjectTrajectoryProjectionError(
                    "initial_noise and clean authority devices differ"
                )
            if initial_noise.dtype != rows[0].clean_packed.dtype:
                raise ObjectTrajectoryProjectionError(
                    "initial_noise and clean authority dtypes differ"
                )
            noise_registration = "supplied_exact_match_first_native_sample"
            noise_device = initial_noise.device
            noise_dtype = initial_noise.dtype

        names: set[str] = set()
        aliases: list[tuple[str, Any]] = []
        if initial_noise is not None:
            aliases.append(("initial_noise", initial_noise))
        prepared: list[
            tuple[
                ProjectionRow,
                Any,
                int,
                int,
                tuple[int, ...],
                float | None,
                float | None,
                tuple[bool, ...] | None,
            ]
        ] = []
        for row_index, row in enumerate(rows):
            if (
                type(row.name) is not str
                or not row.name
                or row.name.strip() != row.name
                or row.name in names
            ):
                raise ObjectTrajectoryProjectionError(
                    "projection row names must be unique non-empty stripped strings"
                )
            names.add(row.name)
            shape = _validate_float_packed(
                row.clean_packed,
                label=f"row {row.name!r} clean_packed",
                expected_shape=first_shape,
            )
            if shape != first_shape:
                raise ObjectTrajectoryProjectionError(
                    f"row {row.name!r} clean shape differs"
                )
            if row.clean_packed.device != noise_device:
                raise ObjectTrajectoryProjectionError(
                    f"row {row.name!r} clean device differs"
                )
            if row.clean_packed.dtype != noise_dtype:
                raise ObjectTrajectoryProjectionError(
                    f"row {row.name!r} clean dtype differs"
                )
            mask, selected_tokens, selected_elements, weight_shape = (
                _validate_binary_weights(
                    row.projection_weights,
                    batch_size=first_shape[0],
                    target_token_count=self.target_token_count,
                    device=noise_device,
                    label=f"row {row.name!r} projection_weights",
                )
            )
            minimum = _bound(
                row.active_next_sigma_min,
                label=f"row {row.name!r} active_next_sigma_min",
            )
            maximum = _bound(
                row.active_next_sigma_max,
                label=f"row {row.name!r} active_next_sigma_max",
            )
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ObjectTrajectoryProjectionError(
                    f"row {row.name!r} sigma interval is empty"
                )
            gates = _step_gates(
                row.step_gates,
                expected_steps=self.expected_steps,
                label=f"row {row.name!r} step_gates",
            )
            aliases.extend(
                (
                    (f"row[{row_index}].clean_packed", row.clean_packed),
                    (
                        f"row[{row_index}].projection_weights",
                        row.projection_weights,
                    ),
                )
            )
            prepared.append(
                (
                    row,
                    mask,
                    selected_tokens,
                    selected_elements,
                    weight_shape,
                    minimum,
                    maximum,
                    gates,
                )
            )
        _validate_no_storage_aliases(aliases)

        # Validate all potential overlaps before taking private snapshots.
        for left_index, left in enumerate(prepared):
            for right in prepared[left_index + 1 :]:
                overlap = left[1] & right[1]
                if bool(overlap.any().item()) and not torch.equal(
                    left[0].clean_packed[overlap], right[0].clean_packed[overlap]
                ):
                    raise ObjectTrajectoryProjectionError(
                        f"rows {left[0].name!r} and {right[0].name!r} overlap "
                        "with different clean authority values"
                    )

        global_mask = torch.zeros(first_shape, device=noise_device, dtype=torch.bool)
        for item in prepared:
            global_mask |= item[1]
        global_elements = int(global_mask.count_nonzero().item())
        global_tokens = int(global_mask.any(dim=2).count_nonzero().item())
        self.globally_enabled = global_elements > 0 and any(
            any(
                (item[5] is None or self.sigmas[index + 1] >= item[5])
                and (item[6] is None or self.sigmas[index + 1] <= item[6])
                and (item[7] is None or item[7][index])
                for index in range(self.expected_steps)
            )
            for item in prepared
            if item[3] > 0
        )

        row_specs = tuple(
            {
                "name": item[0].name,
                "clean_shape": list(first_shape),
                "weight_shape": list(item[4]),
                "selected_token_count": item[2],
                "selected_element_count": item[3],
                "active_next_sigma_min": item[5],
                "active_next_sigma_max": item[6],
                "step_gates": list(item[7]) if item[7] is not None else None,
            }
            for item in prepared
        )
        self.trace = ProjectionTrace(
            expected_steps=self.expected_steps,
            source_token_count=self.source_token_count,
            target_token_count=self.target_token_count,
            batch_size=first_shape[0],
            packed_channels=PACKED_CHANNELS,
            clean_dtype=str(rows[0].clean_packed.dtype),
            clean_device=str(rows[0].clean_packed.device),
            initial_noise_dtype=str(noise_dtype),
            initial_noise_device=str(noise_device),
            initial_noise_registration=noise_registration,
            row_specs=row_specs,
            globally_selected_token_count=global_tokens,
            globally_selected_element_count=global_elements,
            globally_enabled=self.globally_enabled,
        )
        self.shape = first_shape
        self.dtype = noise_dtype
        self.device = noise_device
        self._installed = False
        self._registered_rows: tuple[_RegisteredRow, ...] = ()
        self._initial_noise: Any = None
        self._had_instance_step = False
        self._old_instance_step: Any = None
        self._original_step: Any = None

        if not self.globally_enabled:
            # Preserve the strongest null: no retained tensor snapshots and no
            # instance-level scheduler.step mutation.
            return

        self._initial_noise = (
            None
            if initial_noise is None
            else initial_noise.detach().clone().contiguous()
        )
        self._registered_rows = tuple(
            _RegisteredRow(
                name=item[0].name,
                clean=item[0].clean_packed.detach().clone().contiguous(),
                mask=item[1].detach().clone().contiguous(),
                active_next_sigma_min=item[5],
                active_next_sigma_max=item[6],
                step_gates=item[7],
                selected_token_count=item[2],
                selected_element_count=item[3],
                original_weight_shape=item[4],
            )
            for item in prepared
        )
        try:
            instance_dict = vars(scheduler)
        except TypeError as error:
            raise ObjectTrajectoryProjectionError(
                "scheduler must permit a reversible instance step wrapper"
            ) from error
        self._had_instance_step = "step" in instance_dict
        self._old_instance_step = instance_dict.get("step")
        self._original_step = getattr(scheduler, "step", None)
        if not callable(self._original_step):
            raise ObjectTrajectoryProjectionError("scheduler.step must be callable")

    def _audit_runtime_unchanged(self) -> None:
        sigmas, timesteps = _audit_scheduler_contract(
            self.scheduler,
            expected_steps=self.expected_steps,
            permit_installed_wrapper=True,
        )
        if sigmas != self.sigmas or timesteps != self.timesteps:
            raise ObjectTrajectoryProjectionError(
                "UniPC runtime schedule changed after projection installation"
            )

    def _validate_step_tensors(self, model_output: Any, sample: Any) -> None:
        _validate_float_packed(
            model_output, label="model_output", expected_shape=self.shape
        )
        _validate_float_packed(sample, label="sample", expected_shape=self.shape)
        if model_output.device != self.device or sample.device != self.device:
            raise ObjectTrajectoryProjectionError(
                "model_output, sample, authority, and noise devices differ"
            )
        if model_output.dtype != self.dtype or sample.dtype != self.dtype:
            raise ObjectTrajectoryProjectionError(
                "model_output, sample, authority, and noise dtypes differ"
            )
        if _same_storage(model_output, sample):
            raise ObjectTrajectoryProjectionError(
                "model_output and sample storage alias is forbidden"
            )
        retained: list[tuple[str, Any]] = []
        if self._initial_noise is not None:
            retained.append(("initial_noise_snapshot", self._initial_noise))
        for row in self._registered_rows:
            retained.extend(
                (
                    (f"row {row.name!r} clean snapshot", row.clean),
                    (f"row {row.name!r} mask snapshot", row.mask),
                )
            )
        for label, value in retained:
            if _same_storage(model_output, value) or _same_storage(sample, value):
                raise ObjectTrajectoryProjectionError(
                    f"scheduler tensor aliases retained {label}"
                )

    def _active_rows(self, index: int, next_sigma: float) -> tuple[_RegisteredRow, ...]:
        return tuple(
            row
            for row in self._registered_rows
            if row.selected_element_count > 0
            and row.active(step_index=index, next_sigma=next_sigma)
        )

    def _merged_authority(self, active: Sequence[_RegisteredRow]) -> tuple[Any, Any]:
        import torch

        mask = torch.zeros(self.shape, device=self.device, dtype=torch.bool)
        clean = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
        for row in active:
            clean[row.mask] = row.clean[row.mask]
            mask |= row.mask
        return clean, mask

    def _validate_native_result(
        self,
        result: Any,
        *,
        sample: Any,
        model_output: Any,
        index: int,
    ) -> tuple[Any, int]:
        import torch

        if type(result) is not tuple or len(result) != 1:
            raise ObjectTrajectoryProjectionError(
                "return_dict=False UniPC result must be one built-in tuple"
            )
        previous = result[0]
        _validate_float_packed(previous, label="UniPC previous sample", expected_shape=self.shape)
        if previous.device != self.device or previous.dtype != self.dtype:
            raise ObjectTrajectoryProjectionError(
                "UniPC previous sample dtype/device differs from sampler state"
            )
        if _same_storage(previous, sample) or _same_storage(previous, model_output):
            raise ObjectTrajectoryProjectionError(
                "UniPC previous sample must not alias step inputs"
            )
        cursor_after = getattr(self.scheduler, "step_index", None)
        if cursor_after is None:
            cursor_after = getattr(self.scheduler, "_step_index", None)
        if cursor_after is None or int(cursor_after) != index + 1:
            raise ObjectTrajectoryProjectionError(
                "UniPC scheduler cursor did not advance exactly once"
            )
        return previous, int(cursor_after)

    def _wrapped_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        if len(args) > 4:
            raise ObjectTrajectoryProjectionError(
                "scheduler.step received unsupported positional arguments"
            )
        if any(
            name not in {"model_output", "timestep", "sample", "return_dict"}
            for name in kwargs
        ):
            raise ObjectTrajectoryProjectionError(
                "scheduler.step received unsupported keyword arguments"
            )
        model_output = _extract_step_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = _extract_step_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_step_argument(args, kwargs, index=2, name="sample")
        return_dict = _extract_step_argument(
            args, kwargs, index=3, name="return_dict"
        )
        if return_dict is not False:
            raise ObjectTrajectoryProjectionError(
                "object trajectory projection requires explicit return_dict=False"
            )
        self._validate_step_tensors(model_output, sample)
        self._audit_runtime_unchanged()
        index = len(self.trace.records)
        if index >= self.expected_steps:
            raise ObjectTrajectoryProjectionError(
                "object trajectory projection observed too many scheduler steps"
            )
        timestep_float = _scalar_float(timestep, label="scheduler.step timestep")
        if timestep_float != self.timesteps[index]:
            raise ObjectTrajectoryProjectionError(
                f"scheduler.step timestep differs at {index}: "
                f"{timestep_float} != {self.timesteps[index]}"
            )
        cursor_before = getattr(self.scheduler, "step_index", None)
        if cursor_before is None:
            cursor_before = getattr(self.scheduler, "_step_index", None)
        lazy_noise_snapshot_created = False
        if index == 0:
            if cursor_before is not None:
                raise ObjectTrajectoryProjectionError(
                    "object trajectory projection requires a fresh UniPC cursor"
                )
            if self._initial_noise is None:
                # Bernini's legacy model.sample creates its noise after the
                # caller enters this context.  Capture those exact bytes; no
                # new generator or RNG operation is introduced.
                self._initial_noise = sample.detach().clone().contiguous()
                self.trace.initial_noise_captured_from_first_native_sample = True
                lazy_noise_snapshot_created = True
            elif not torch.equal(sample, self._initial_noise):
                raise ObjectTrajectoryProjectionError(
                    "first packed sample differs from registered initial_noise"
                )
            self.trace.initial_noise_verified = True
        elif cursor_before is None or int(cursor_before) != index:
            raise ObjectTrajectoryProjectionError(
                "UniPC scheduler cursor differs before step"
            )

        next_sigma = self.sigmas[index + 1]
        active = self._active_rows(index, next_sigma)
        active_names = tuple(row.name for row in active)
        inactive_names = tuple(
            row.name for row in self._registered_rows if row.name not in active_names
        )

        if not active:
            # Exact native delegate.  Do not create new args/kwargs dictionaries,
            # do not clone either packed input, and return the original tuple.
            result = self._original_step(*args, **kwargs)
            _previous, cursor_after = self._validate_native_result(
                result,
                sample=sample,
                model_output=model_output,
                index=index,
            )
            self.trace.records.append(
                ProjectionStepRecord(
                    step_index=index,
                    timestep=timestep_float,
                    sigma=self.sigmas[index],
                    next_sigma=next_sigma,
                    cursor_before=None if cursor_before is None else int(cursor_before),
                    cursor_after=cursor_after,
                    projection_applied=False,
                    active_rows=(),
                    inactive_rows=inactive_names,
                    selected_token_count=0,
                    selected_element_count=0,
                    total_element_count=int(sample.numel()),
                    original_scheduler_step_calls=1,
                    exact_native_delegate_no_argument_clone=True,
                    initial_noise_snapshot_created_this_step=(
                        lazy_noise_snapshot_created
                    ),
                    initial_sample_matches_registered_noise=(index == 0),
                    selected_velocity_exact=None,
                    unselected_velocity_exact=True,
                    selected_post_step_exact=None,
                    unselected_post_step_exact=True,
                )
            )
            return result

        clean, mask = self._merged_authority(active)
        compute_dtype = (
            torch.float64 if self.dtype == torch.float64 else torch.float32
        )
        forced_exact = self._initial_noise.to(dtype=compute_dtype) - clean.to(
            dtype=compute_dtype
        )
        forced_velocity = model_output.clone()
        forced_velocity[mask] = forced_exact[mask].to(dtype=model_output.dtype)
        if not bool(torch.isfinite(forced_velocity).all().item()):
            raise ObjectTrajectoryProjectionError(
                "projected flow velocity contains non-finite values"
            )
        selected_velocity_exact = torch.equal(
            forced_velocity[mask], forced_exact[mask].to(dtype=model_output.dtype)
        )
        unselected_velocity_exact = torch.equal(
            forced_velocity[~mask], model_output[~mask]
        )
        if not selected_velocity_exact or not unselected_velocity_exact:
            raise ObjectTrajectoryProjectionError(
                "flow velocity projection failed exact selected/unselected checks"
            )
        call_args, call_kwargs = _replace_model_output(
            args, kwargs, forced_velocity
        )
        result = self._original_step(*call_args, **call_kwargs)
        previous, cursor_after = self._validate_native_result(
            result,
            sample=sample,
            model_output=forced_velocity,
            index=index,
        )
        if next_sigma == 0.0:
            # Preserve the registered terminal authority exactly, including
            # float64 detail that a float32 round trip would otherwise lose.
            trajectory = clean
        else:
            trajectory = (
                (1.0 - next_sigma) * clean.to(dtype=compute_dtype)
                + next_sigma * self._initial_noise.to(dtype=compute_dtype)
            ).to(dtype=previous.dtype)
        projected_previous = previous.clone()
        projected_previous[mask] = trajectory[mask]
        selected_post_exact = torch.equal(
            projected_previous[mask], trajectory[mask]
        )
        unselected_post_exact = torch.equal(
            projected_previous[~mask], previous[~mask]
        )
        if not selected_post_exact or not unselected_post_exact:
            raise ObjectTrajectoryProjectionError(
                "post-step projection failed exact selected/unselected checks"
            )
        selected_elements = int(mask.count_nonzero().item())
        selected_tokens = int(mask.any(dim=2).count_nonzero().item())
        self.trace.records.append(
            ProjectionStepRecord(
                step_index=index,
                timestep=timestep_float,
                sigma=self.sigmas[index],
                next_sigma=next_sigma,
                cursor_before=None if cursor_before is None else int(cursor_before),
                cursor_after=cursor_after,
                projection_applied=True,
                active_rows=active_names,
                inactive_rows=inactive_names,
                selected_token_count=selected_tokens,
                selected_element_count=selected_elements,
                total_element_count=int(sample.numel()),
                original_scheduler_step_calls=1,
                exact_native_delegate_no_argument_clone=False,
                initial_noise_snapshot_created_this_step=(
                    lazy_noise_snapshot_created
                ),
                initial_sample_matches_registered_noise=(index == 0),
                selected_velocity_exact=True,
                unselected_velocity_exact=True,
                selected_post_step_exact=True,
                unselected_post_step_exact=True,
            )
        )
        return (projected_previous,)

    def install(self) -> None:
        if not self.globally_enabled:
            return
        if self._installed:
            raise ObjectTrajectoryProjectionError(
                "object trajectory projection is already installed"
            )

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_step(*args, **kwargs)

        setattr(wrapped, _WRAPPER_MARKER, True)
        try:
            setattr(self.scheduler, "step", wrapped)
        except Exception as error:
            raise ObjectTrajectoryProjectionError(
                "cannot install scheduler.step projection"
            ) from error
        self._installed = True
        self.trace.wrapper_installed = True

    def restore(self) -> None:
        if not self._installed:
            self.trace.wrapper_restored = True
            return
        try:
            if self._had_instance_step:
                setattr(self.scheduler, "step", self._old_instance_step)
            else:
                delattr(self.scheduler, "step")
        finally:
            self._installed = False
            self.trace.wrapper_restored = True

    def finalize(self) -> None:
        if self.globally_enabled:
            if (
                len(self.trace.records) != self.expected_steps
                or not self.trace.initial_noise_verified
                or self.sigmas[-1] != 0.0
                or self.trace.records[-1].next_sigma != 0.0
            ):
                raise ObjectTrajectoryProjectionError(
                    "projection did not complete the full terminal-zero schedule"
                )
        elif self.trace.records:
            raise ObjectTrajectoryProjectionError(
                "all-zero bypass unexpectedly recorded scheduler steps"
            )
        self.trace.finalized = True


@contextmanager
def project_object_trajectory_unipc_steps(
    scheduler: Any,
    *,
    rows: Sequence[ProjectionRow],
    initial_noise: Any | None = None,
    source_token_count: int,
    target_token_count: int,
    expected_steps: int,
) -> Iterator[ProjectionTrace]:
    """Install a reversible zero-training oracle projection around UniPC.

    No target video is consumed here.  ``clean_packed`` authorities and masks
    must already have been materialized by an explicitly oracle-only upstream
    process.  When ``initial_noise`` is omitted, the first native packed sample
    is cloned before the first solver call; this introduces no RNG operation.
    This function has intentionally not been connected to the production
    Bernini inference ABI.
    """

    installed = _InstalledObjectTrajectoryProjection(
        scheduler,
        rows=rows,
        initial_noise=initial_noise,
        source_token_count=source_token_count,
        target_token_count=target_token_count,
        expected_steps=expected_steps,
    )
    installed.install()
    try:
        yield installed.trace
        installed.finalize()
    finally:
        installed.restore()


def project_single_object_trajectory_unipc_steps(
    scheduler: Any,
    *,
    clean_packed: Any,
    initial_noise: Any | None = None,
    projection_weights: Any,
    source_token_count: int,
    target_token_count: int,
    expected_steps: int,
    name: str = "object_authority",
    active_next_sigma_min: float | None = None,
    active_next_sigma_max: float | None = None,
    step_gates: tuple[bool | int, ...] | None = None,
) -> Any:
    """Convenience context manager for the original single-authority ABI."""

    row = ProjectionRow(
        name=name,
        clean_packed=clean_packed,
        projection_weights=projection_weights,
        active_next_sigma_min=active_next_sigma_min,
        active_next_sigma_max=active_next_sigma_max,
        step_gates=step_gates,
    )
    return project_object_trajectory_unipc_steps(
        scheduler,
        rows=(row,),
        initial_noise=initial_noise,
        source_token_count=source_token_count,
        target_token_count=target_token_count,
        expected_steps=expected_steps,
    )


__all__ = [
    "FLOW_SHIFT",
    "ObjectTrajectoryProjectionError",
    "PACKED_CHANNELS",
    "ProjectionRow",
    "ProjectionStepRecord",
    "ProjectionTrace",
    "SCHEMA_VERSION",
    "project_object_trajectory_unipc_steps",
    "project_single_object_trajectory_unipc_steps",
    "tensor_core_contract",
]

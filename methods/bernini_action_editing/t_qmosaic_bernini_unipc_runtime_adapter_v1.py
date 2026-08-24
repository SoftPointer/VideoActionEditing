#!/usr/bin/env python3
"""One-shot adapter from Bernini's real UniPC loop to T-Q-MOSAIC v1.

This module owns neither a sampler nor a scheduler.  It temporarily replaces
the *instance* ``step`` attribute of one pinned Bernini ``GEN_Wanx22``
``UniPCMultistepScheduler`` while the original ``GEN_Wanx22.sample`` method is
running.  The vendor call remains exactly::

    scheduler.step(model_output, timestep, sample, return_dict=False)[0]

The adapter observes the input state immediately before that call, invokes the
original bound scheduler method exactly once, and passes the tuple's sole
tensor through the frozen ``t_qmosaic_trajectory_intervention_v1`` primitive.
For capture and sign-zero replay it returns the original scheduler tuple
object, so Bernini's ``[0]`` observes the original tensor object without a
clone or arithmetic.  A non-zero replay replaces only tuple element zero at
the three primitive-owned intervention coordinates.

The patch is one-shot and is restored in a ``finally`` block after normal
completion, any sampler/scheduler exception, or a trajectory-contract error.
Successful receipts are engineering evidence only.  They authorize no
semantic claim, arm selection, optimizer, training, or parameter update.

Pinned API audit basis (AUH ``vace``, 2026-08-09):

* Bernini commit ``2d2b4591ac053ec25c6371b01a5a6746679e5793``;
* ``bernini/models/wan_diffusion.py`` SHA256 ``59e860ba...2512``;
* Bernini-R scheduler config SHA256 ``3fed2abd...a6e``;
* Diffusers 0.38.0 returned a built-in one-element tuple and advanced
  ``step_index`` from ``None`` to ``1`` on the first real CPU UniPC step.

The source hashes are audit provenance, not a filesystem authority granted to
this module.  Runtime authority comes from the exact Python class identities,
the closed scheduler configuration, and the full bit-exact 40-step schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

import inference_sigma_strata as _pinned_schedule
import t_qmosaic_trajectory_intervention_v1 as _trajectory


ADAPTER_SCHEMA_VERSION = "bernini-t-qmosaic-unipc-runtime-adapter-v1"
PINNED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_WAN_DIFFUSION_SHA256 = (
    "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
)
PINNED_SCHEDULER_CONFIG_FILE_SHA256 = (
    "3fed2abbd9bbc301a74db01947198057ec5049808910dccab320925bf27bea6e"
)
PINNED_DIFFUSION_CLASS = ("bernini.models.wan_diffusion", "GEN_Wanx22")
PINNED_SCHEDULER_CLASS = (
    "diffusers.schedulers.scheduling_unipc_multistep",
    "UniPCMultistepScheduler",
)
PINNED_STEP_PARAMETER_NAMES = (
    "model_output",
    "timestep",
    "sample",
    "return_dict",
)

# This is the behavior-bearing runtime configuration observed after loading
# the released checkpoint config with ``flow_shift=5.0``.  Diffusers'
# bookkeeping-only ``_use_default_values`` list is deliberately excluded.
PINNED_SCHEDULER_CONFIG = MappingProxyType(
    {
        "_class_name": "UniPCMultistepScheduler",
        "_diffusers_version": "0.33.0.dev0",
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "disable_corrector": (),
        "dynamic_thresholding_ratio": 0.995,
        "final_sigmas_type": "zero",
        "flow_shift": 5.0,
        "lower_order_final": True,
        "num_train_timesteps": 1000,
        "predict_x0": True,
        "prediction_type": "flow_prediction",
        "rescale_betas_zero_snr": False,
        "sample_max_value": 1.0,
        "shift_terminal": None,
        "sigma_max": None,
        "sigma_min": None,
        "solver_order": 2,
        "solver_p": None,
        "solver_type": "bh2",
        "steps_offset": 0,
        "thresholding": False,
        "time_shift_type": "exponential",
        "timestep_spacing": "linspace",
        "trained_betas": None,
        "use_beta_sigmas": False,
        "use_dynamic_shifting": False,
        "use_exponential_sigmas": False,
        "use_flow_sigmas": True,
        "use_karras_sigmas": False,
    }
)


class TQMosaicBerniniRuntimeError(RuntimeError):
    """The pinned Bernini-to-trajectory runtime contract was violated."""


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
        raise TQMosaicBerniniRuntimeError(
            "runtime receipt is not finite canonical ASCII JSON"
        ) from error


def _seal(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise TQMosaicBerniniRuntimeError("runtime receipt is already sealed")
    plain = dict(unsigned)
    return {
        **plain,
        "receipt_digest": hashlib.sha256(_canonical_json_bytes(plain)).hexdigest(),
    }


def _normalized_config_value(value: Any, *, label: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TQMosaicBerniniRuntimeError(f"scheduler config {label} is non-finite")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalized_config_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TQMosaicBerniniRuntimeError(
        f"scheduler config {label} has unsupported type {type(value).__name__}"
    )


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        if name not in config:
            raise TQMosaicBerniniRuntimeError(
                f"scheduler config is missing required field {name}"
            )
        return config[name]
    if not hasattr(config, name):
        raise TQMosaicBerniniRuntimeError(
            f"scheduler config is missing required field {name}"
        )
    return getattr(config, name)


def _expected_config_snapshot() -> dict[str, Any]:
    return {
        name: _normalized_config_value(value, label=name)
        for name, value in PINNED_SCHEDULER_CONFIG.items()
    }


PINNED_SCHEDULER_CONFIG_DIGEST = (
    "376b2bc18f8801411e1a7bf7005c4734a9cbf52565b1a1d84fd3a86e34c6e595"
)
if hashlib.sha256(
    _canonical_json_bytes(_expected_config_snapshot())
).hexdigest() != PINNED_SCHEDULER_CONFIG_DIGEST:  # pragma: no cover
    raise RuntimeError("pinned Bernini UniPC config constants differ from their hash")


def _audit_scheduler_config(scheduler: Any) -> dict[str, Any]:
    config = getattr(scheduler, "config", None)
    if config is None:
        raise TQMosaicBerniniRuntimeError("UniPC scheduler must expose config")
    expected = _expected_config_snapshot()
    observed: dict[str, Any] = {}
    for name, expected_value in expected.items():
        raw = _config_value(config, name)
        value = _normalized_config_value(raw, label=name)
        if type(expected_value) is bool:
            matches = value is expected_value
        elif type(expected_value) is int:
            matches = type(value) is int and value == expected_value
        elif type(expected_value) is float:
            matches = type(value) in (int, float) and float(value) == expected_value
        else:
            matches = value == expected_value
        if not matches:
            raise TQMosaicBerniniRuntimeError(
                f"scheduler config {name} differs: expected {expected_value!r}, "
                f"got {value!r}"
            )
        observed[name] = value
    digest = hashlib.sha256(_canonical_json_bytes(observed)).hexdigest()
    if digest != PINNED_SCHEDULER_CONFIG_DIGEST:
        raise TQMosaicBerniniRuntimeError("scheduler config digest differs")
    return observed


def _audit_step_signature(step: Any) -> dict[str, Any]:
    try:
        parameters = tuple(inspect.signature(step).parameters.values())
    except (TypeError, ValueError) as error:
        raise TQMosaicBerniniRuntimeError(
            "cannot inspect original UniPC step signature"
        ) from error
    if tuple(parameter.name for parameter in parameters) != PINNED_STEP_PARAMETER_NAMES:
        raise TQMosaicBerniniRuntimeError("original UniPC step parameter names differ")
    positional_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if any(parameter.kind not in positional_kinds for parameter in parameters[:3]):
        raise TQMosaicBerniniRuntimeError(
            "original UniPC tensor arguments are not positional"
        )
    final = parameters[3]
    if (
        final.kind not in positional_kinds
        or final.default is not True
        or any(
            parameter.default is not inspect.Parameter.empty
            for parameter in parameters[:3]
        )
    ):
        raise TQMosaicBerniniRuntimeError(
            "original UniPC return_dict/default signature differs"
        )
    return {
        "parameter_names": list(PINNED_STEP_PARAMETER_NAMES),
        "three_required_tensor_arguments": True,
        "return_dict_default": True,
    }


def _callable_identity(value: Any) -> tuple[int | None, int]:
    owner = getattr(value, "__self__", None)
    function = getattr(value, "__func__", value)
    return (id(owner) if owner is not None else None, id(function))


def _plain_nonnegative_index(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise TQMosaicBerniniRuntimeError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TQMosaicBerniniRuntimeError(f"{label} must be an integer") from error
    if result < 0 or result != value:
        raise TQMosaicBerniniRuntimeError(f"{label} must be a nonnegative integer")
    return result


@dataclass(frozen=True)
class BerniniUniPCTrajectoryRunResultV1:
    """Output of one restored, receipt-bearing adapter run."""

    sample_output: Any
    trajectory_artifact: Any
    receipt: dict[str, Any]


class BerniniUniPCTrajectoryRuntimeAdapterV1:
    """Reversibly bind one capture or replay to one original Bernini sample."""

    def __init__(self, diffusion: Any, *, trajectory: Any) -> None:
        observed_diffusion_class = (
            type(diffusion).__module__,
            type(diffusion).__name__,
        )
        if observed_diffusion_class != PINNED_DIFFUSION_CLASS:
            raise TQMosaicBerniniRuntimeError(
                "diffusion class is not pinned bernini.models.wan_diffusion.GEN_Wanx22"
            )
        if getattr(diffusion, "use_unipc", None) is not True:
            raise TQMosaicBerniniRuntimeError("Bernini diffusion.use_unipc must be True")
        if not hasattr(diffusion, "transformer_2") or diffusion.transformer_2 is not None:
            raise TQMosaicBerniniRuntimeError(
                "adapter is pinned to single-expert Bernini-R 1.3B"
            )
        scheduler = getattr(diffusion, "scheduler", None)
        observed_scheduler_class = (
            type(scheduler).__module__,
            type(scheduler).__name__,
        )
        if observed_scheduler_class != PINNED_SCHEDULER_CLASS:
            raise TQMosaicBerniniRuntimeError(
                "scheduler class is not the pinned Diffusers UniPC class"
            )
        try:
            diffusion_instance = vars(diffusion)
            scheduler_instance = vars(scheduler)
        except TypeError as error:
            raise TQMosaicBerniniRuntimeError(
                "cannot inspect Bernini or scheduler instance patches"
            ) from error
        if "sample" in diffusion_instance:
            raise TQMosaicBerniniRuntimeError(
                "refusing an instance-level Bernini sample override"
            )
        if "step" in scheduler_instance:
            raise TQMosaicBerniniRuntimeError(
                "refusing to stack over an instance-level scheduler.step"
            )
        original_sample = getattr(diffusion, "sample", None)
        original_step = getattr(scheduler, "step", None)
        if not callable(original_sample) or not callable(original_step):
            raise TQMosaicBerniniRuntimeError(
                "Bernini sample and UniPC step must be callable"
            )
        if type(trajectory) is _trajectory.ActualTrajectoryCaptureV1:
            mode = "capture"
            sign: int | None = None
        elif type(trajectory) is _trajectory.TQMosaicTrajectoryReplayV1:
            mode = "replay"
            sign = trajectory.sign
        else:
            raise TQMosaicBerniniRuntimeError(
                "trajectory must be an exact T-Q-MOSAIC v1 capture or replay"
            )

        self.diffusion = diffusion
        self.scheduler = scheduler
        self.trajectory = trajectory
        self.mode = mode
        self.sign = sign
        self.original_sample = original_sample
        self.original_scheduler_step = original_step
        self._original_step_identity = _callable_identity(original_step)
        self._config_snapshot = _audit_scheduler_config(scheduler)
        self._step_signature = _audit_step_signature(original_step)
        self._wrapper: Any | None = None
        self._started = False
        self._installed = False
        self._restored = False
        self._completed = False
        self._step_calls = 0
        self._original_step_calls = 0
        self._tuple_identity_returns = 0
        self._tensor_identity_returns = 0
        self._schedule_audit: dict[str, Any] | None = None
        self._trajectory_artifact: Any | None = None
        self._receipt_bytes: bytes | None = None

    def _install(self) -> None:
        if self._started or self._installed or self._restored or self._completed:
            raise TQMosaicBerniniRuntimeError("adapter is one-shot")
        if "step" in vars(self.scheduler):
            raise TQMosaicBerniniRuntimeError(
                "scheduler.step gained an instance override before installation"
            )

        def scheduler_step_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        setattr(scheduler_step_wrapper, "_bernini_t_qmosaic_unipc_runtime_v1", self)
        self._wrapper = scheduler_step_wrapper
        self._started = True
        try:
            setattr(self.scheduler, "step", scheduler_step_wrapper)
        except Exception:
            self._wrapper = None
            raise
        self._installed = True

    def _restore(self) -> None:
        if not self._installed:
            return
        current = getattr(self.scheduler, "step", None)
        wrapper_changed = getattr(
            current, "_bernini_t_qmosaic_unipc_runtime_v1", None
        ) is not self
        restoration_error: Exception | None = None
        try:
            if "step" in vars(self.scheduler):
                delattr(self.scheduler, "step")
            else:
                restoration_error = TQMosaicBerniniRuntimeError(
                    "scheduler.step wrapper disappeared before restoration"
                )
        except Exception as error:
            restoration_error = error
        self._installed = False
        self._restored = True
        if _callable_identity(getattr(self.scheduler, "step", None)) != (
            self._original_step_identity
        ):
            restoration_error = TQMosaicBerniniRuntimeError(
                "original scheduler.step was not restored"
            )
        if wrapper_changed:
            restoration_error = TQMosaicBerniniRuntimeError(
                "scheduler.step changed while the adapter was active"
            )
        if restoration_error is not None:
            raise TQMosaicBerniniRuntimeError(
                "failed to restore original scheduler.step"
            ) from restoration_error

    def _audit_live_static_contract(self) -> None:
        if self.diffusion.scheduler is not self.scheduler:
            raise TQMosaicBerniniRuntimeError(
                "Bernini scheduler object changed during sampling"
            )
        observed_class = (
            type(self.scheduler).__module__,
            type(self.scheduler).__name__,
        )
        if observed_class != PINNED_SCHEDULER_CLASS:
            raise TQMosaicBerniniRuntimeError("live scheduler class changed")
        if _audit_scheduler_config(self.scheduler) != self._config_snapshot:
            raise TQMosaicBerniniRuntimeError("live scheduler config changed")

    def _audit_full_schedule(self) -> dict[str, Any]:
        self._audit_live_static_contract()
        try:
            audited = _pinned_schedule.audit_runtime_unipc_schedule(
                self.scheduler,
                initialize=False,
            )
        except Exception as error:
            raise TQMosaicBerniniRuntimeError(
                "runtime UniPC full exact40 schedule differs"
            ) from error
        if audited.get("schedule_sha256") != _trajectory.PINNED_SCHEDULE_SHA256:
            raise TQMosaicBerniniRuntimeError(
                "runtime schedule hash differs from trajectory primitive"
            )
        return audited

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        if not self._installed or self._wrapper is None or self._completed:
            raise TQMosaicBerniniRuntimeError(
                "scheduler.step ran outside the active one-shot adapter"
            )
        if len(args) != 3 or set(kwargs) != {"return_dict"}:
            raise TQMosaicBerniniRuntimeError(
                "Bernini must call scheduler.step with three positional tensors "
                "and explicit return_dict=False"
            )
        if kwargs["return_dict"] is not False:
            raise TQMosaicBerniniRuntimeError(
                "Bernini scheduler.step must explicitly use return_dict=False"
            )
        _model_output, timestep, sample = args
        step_index = self._step_calls
        if step_index >= _trajectory.EXACT_SCHEDULER_CALLS:
            raise TQMosaicBerniniRuntimeError(
                "Bernini attempted more than 40 scheduler calls"
            )
        if step_index == 0:
            self._schedule_audit = self._audit_full_schedule()
        else:
            self._audit_live_static_contract()
        cursor = getattr(self.scheduler, "step_index", None)
        if step_index == 0:
            if cursor is not None:
                raise TQMosaicBerniniRuntimeError(
                    "first UniPC step_index must be None after set_timesteps(40)"
                )
        elif _plain_nonnegative_index(
            cursor, label="scheduler.step_index"
        ) != step_index:
            raise TQMosaicBerniniRuntimeError(
                "UniPC pre-call step_index differs from adapter call count"
            )
        sigmas = getattr(self.scheduler, "sigmas", None)
        timesteps = getattr(self.scheduler, "timesteps", None)
        try:
            sigma = sigmas[step_index]
            official_timestep = timesteps[step_index]
        except Exception as error:
            raise TQMosaicBerniniRuntimeError(
                "cannot resolve live UniPC schedule coordinate"
            ) from error
        if (
            not isinstance(timestep, _trajectory.torch.Tensor)
            or timestep.ndim != 0
            or timestep.device.type == "meta"
            or timestep.dtype != _trajectory.torch.int64
        ):
            raise TQMosaicBerniniRuntimeError(
                "Bernini loop timestep must be one materialized int64 scalar"
            )
        observed_timestep = _trajectory._plain_timestep(
            timestep, label=f"runtime scheduler timestep {step_index}"
        )
        expected_timestep = _trajectory._plain_timestep(
            official_timestep,
            label=f"runtime scheduler timeline {step_index}",
        )
        if observed_timestep != expected_timestep:
            raise TQMosaicBerniniRuntimeError(
                "Bernini loop timestep differs from live UniPC timeline"
            )
        self.trajectory.before_scheduler_step(
            step_index=step_index,
            timestep=timestep,
            sigma=sigma,
            state=sample,
        )
        raw_result = self.original_scheduler_step(*args, **kwargs)
        self._original_step_calls += 1
        if type(raw_result) is not tuple or len(raw_result) != 1:
            raise TQMosaicBerniniRuntimeError(
                "real return_dict=False UniPC result must be one built-in tuple"
            )
        raw_state = raw_result[0]
        cursor_after = _plain_nonnegative_index(
            getattr(self.scheduler, "step_index", None),
            label="scheduler post-call step_index",
        )
        if cursor_after != step_index + 1:
            raise TQMosaicBerniniRuntimeError(
                "UniPC post-call step_index did not advance by exactly one"
            )
        returned_state = self.trajectory.after_scheduler_step(
            step_index=step_index,
            next_state=raw_state,
        )
        self._step_calls += 1
        if returned_state is raw_state:
            self._tensor_identity_returns += 1
            self._tuple_identity_returns += 1
            return raw_result
        return (returned_state,)

    def _bind_sampling_contract(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            signature = inspect.signature(self.original_sample)
            bound = signature.bind(*args, **dict(kwargs))
            bound.apply_defaults()
        except (TypeError, ValueError) as error:
            raise TQMosaicBerniniRuntimeError(
                "cannot bind original Bernini sample arguments"
            ) from error
        if "num_inference_steps" not in bound.arguments:
            raise TQMosaicBerniniRuntimeError(
                "Bernini sample lacks num_inference_steps"
            )
        if "flow_shift" not in bound.arguments:
            raise TQMosaicBerniniRuntimeError("Bernini sample lacks flow_shift")
        steps = bound.arguments["num_inference_steps"]
        if type(steps) is not int or steps != _trajectory.EXACT_SCHEDULER_CALLS:
            raise TQMosaicBerniniRuntimeError(
                "T-Q-MOSAIC adapter requires num_inference_steps=40"
            )
        shift = bound.arguments["flow_shift"]
        if (
            isinstance(shift, bool)
            or type(shift) not in (int, float)
            or not math.isfinite(float(shift))
            or float(shift) != _pinned_schedule.FLOW_SHIFT
        ):
            raise TQMosaicBerniniRuntimeError(
                "T-Q-MOSAIC adapter requires flow_shift=5.0"
            )
        return {
            "num_inference_steps": steps,
            "flow_shift": float(shift),
        }

    def _finalize_trajectory(self) -> tuple[Any, dict[str, Any]]:
        if self.mode == "capture":
            artifact = self.trajectory.finalize()
            receipt = artifact.receipt()
        else:
            receipt = self.trajectory.finalize()
            artifact = receipt
        if receipt.get("evidence_tier") != "ENGINEERING_ONLY":
            raise TQMosaicBerniniRuntimeError(
                "trajectory primitive receipt is not engineering-only"
            )
        for key in (
            "semantic_success_assessed",
            "scientific_claim_authorized",
            "training_update_authorized",
            "parameter_update_performed",
        ):
            if receipt.get(key) is not False:
                raise TQMosaicBerniniRuntimeError(
                    f"trajectory primitive unexpectedly authorized {key}"
                )
        return artifact, receipt

    def _build_receipt(
        self,
        *,
        sampling_contract: Mapping[str, Any],
        primitive_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            not self._restored
            or self._installed
            or self._schedule_audit is None
            or self._step_calls != _trajectory.EXACT_SCHEDULER_CALLS
            or self._original_step_calls != _trajectory.EXACT_SCHEDULER_CALLS
        ):
            raise TQMosaicBerniniRuntimeError(
                "cannot issue receipt for an unclosed Bernini adapter"
            )
        if self.sign == 0 and (
            self._tuple_identity_returns != _trajectory.EXACT_SCHEDULER_CALLS
            or self._tensor_identity_returns != _trajectory.EXACT_SCHEDULER_CALLS
        ):
            raise TQMosaicBerniniRuntimeError(
                "zero replay did not return every original tuple/tensor object"
            )
        unsigned = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "evidence_tier": "ENGINEERING_ONLY",
            "mode": self.mode,
            "sign": self.sign,
            "pinned_bernini_api_audit_basis": {
                "bernini_commit": PINNED_BERNINI_COMMIT,
                "wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
                "checkpoint_scheduler_config_file_sha256": (
                    PINNED_SCHEDULER_CONFIG_FILE_SHA256
                ),
                "runtime_source_file_hash_verified_by_adapter": False,
            },
            "runtime_class": {
                "diffusion_module": PINNED_DIFFUSION_CLASS[0],
                "diffusion_name": PINNED_DIFFUSION_CLASS[1],
                "scheduler_module": PINNED_SCHEDULER_CLASS[0],
                "scheduler_name": PINNED_SCHEDULER_CLASS[1],
                "single_expert_bernini_r_1_3b": True,
            },
            "scheduler_config": dict(self._config_snapshot),
            "scheduler_config_digest": PINNED_SCHEDULER_CONFIG_DIGEST,
            "scheduler_step_signature": dict(self._step_signature),
            "sampling_contract": dict(sampling_contract),
            "schedule": dict(self._schedule_audit),
            "schedule_sha256": _trajectory.PINNED_SCHEDULE_SHA256,
            "scheduler_calls_observed": self._step_calls,
            "original_scheduler_calls_observed": self._original_step_calls,
            "original_scheduler_called_once_per_step": True,
            "scheduler_return_contract": {
                "return_dict_explicit_false_every_step": True,
                "container_type": "builtins.tuple",
                "container_length": 1,
                "bernini_consumption": "scheduler.step(..., return_dict=False)[0]",
                "tuple_objects_returned_by_identity_count": (
                    self._tuple_identity_returns
                ),
                "tensor_objects_returned_by_identity_count": (
                    self._tensor_identity_returns
                ),
                "zero_sign_all_original_tuple_objects_returned_by_identity": (
                    self._tuple_identity_returns == _trajectory.EXACT_SCHEDULER_CALLS
                    if self.sign == 0
                    else None
                ),
                "zero_sign_all_original_tensor_objects_returned_by_identity": (
                    self._tensor_identity_returns == _trajectory.EXACT_SCHEDULER_CALLS
                    if self.sign == 0
                    else None
                ),
            },
            "patch_lifecycle": {
                "instance_scheduler_step_patch_only": True,
                "adapter_one_shot": True,
                "restored_after_sample_or_failure": True,
                "restored_before_receipt": True,
                "original_step_identity_restored": True,
            },
            "trajectory_primitive_schema_version": primitive_receipt.get(
                "schema_version"
            ),
            "trajectory_primitive_receipt_digest": primitive_receipt.get(
                "receipt_digest"
            ),
            "independent_audit_status": "NOT_PERFORMED_BY_THIS_MODULE",
            "gpu_experiment_authorized": False,
            "deployment_authorized": False,
            "sampler_replaced": False,
            "scheduler_replaced": False,
            "external_callback_authority": False,
            "mask_input_authorized": False,
            "track_input_authorized": False,
            "pose_input_authorized": False,
            "optical_flow_input_authorized": False,
            "seed_selection_authorized": False,
            "arm_selection_authorized": False,
            "semantic_success_assessed": False,
            "scientific_claim_authorized": False,
            "optimizer_authorized": False,
            "training_update_authorized": False,
            "parameter_update_performed": False,
        }
        return _seal(unsigned)

    def run_sample(self, *args: Any, **kwargs: Any) -> BerniniUniPCTrajectoryRunResultV1:
        """Run one original sample and restore the temporary step patch."""

        sampling_contract = self._bind_sampling_contract(args, kwargs)
        self._install()
        sample_output: Any
        artifact: Any
        primitive_receipt: dict[str, Any]
        try:
            sample_output = self.original_sample(*args, **kwargs)
            if self._step_calls != _trajectory.EXACT_SCHEDULER_CALLS:
                raise TQMosaicBerniniRuntimeError(
                    f"Bernini sample made {self._step_calls} scheduler calls, expected 40"
                )
            cursor = _plain_nonnegative_index(
                getattr(self.scheduler, "step_index", None),
                label="terminal scheduler.step_index",
            )
            if cursor != _trajectory.EXACT_SCHEDULER_CALLS:
                raise TQMosaicBerniniRuntimeError(
                    "terminal UniPC step_index differs from 40"
                )
            terminal_schedule = self._audit_full_schedule()
            if terminal_schedule != self._schedule_audit:
                raise TQMosaicBerniniRuntimeError(
                    "UniPC schedule changed during the sample"
                )
            artifact, primitive_receipt = self._finalize_trajectory()
        finally:
            self._restore()
        self._trajectory_artifact = artifact
        self._completed = True
        sealed = self._build_receipt(
            sampling_contract=sampling_contract,
            primitive_receipt=primitive_receipt,
        )
        self._receipt_bytes = _canonical_json_bytes(sealed)
        return BerniniUniPCTrajectoryRunResultV1(
            sample_output=sample_output,
            trajectory_artifact=artifact,
            receipt=json.loads(self._receipt_bytes.decode("ascii")),
        )

    def receipt(self) -> dict[str, Any]:
        """Return a mutation-independent copy of the successful receipt."""

        if not self._completed or self._receipt_bytes is None:
            raise TQMosaicBerniniRuntimeError(
                "unfinished or failed adapter run has no receipt"
            )
        value = json.loads(self._receipt_bytes.decode("ascii"))
        digest = value.pop("receipt_digest", None)
        if hashlib.sha256(_canonical_json_bytes(value)).hexdigest() != digest:
            raise TQMosaicBerniniRuntimeError("stored adapter receipt bytes differ")
        return {**value, "receipt_digest": digest}

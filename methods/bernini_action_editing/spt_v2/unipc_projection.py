#!/usr/bin/env python3
"""Inject SPT clean-space execution into Bernini's official UniPC step.

Bernini ``v2v_apg`` finishes CFG/APG by producing a packed flow velocity and
then calls ``scheduler.step(model_output, timestep, sample,
return_dict=False)``.  This module wraps exactly that boundary.  It converts
the already-guided velocity to a clean estimate, executes the student's dense
preserve/transport/generate plan through :func:`execute_packed_velocity`, and
passes the resulting velocity to the *original* scheduler step.  UniPC remains
the sole numerical integrator; this module deliberately contains no Euler
update.

The inference contract is source video + edit instruction only.  The caller
must first obtain a ``provenance='student'`` plan from a planner whose API is
``planner(source, instruction_embedding)``.  Target video, paired oracle plan,
mask, track, pose, flow, and first-frame anchors are rejected or absent from
this boundary API.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterator, Mapping, Sequence

from .phase_transport import (
    GATE_GENERATE,
    GATE_PRESERVE,
    GATE_TRANSPORT,
    PhasePlan,
    PhaseTransportError,
    execute_packed_velocity,
    packed_to_video,
)


class UniPCProjectionError(RuntimeError):
    """Raised before scheduler integration when the SPT boundary is invalid."""


INFERENCE_CONDITIONS = ("source_video", "edit_instruction")
FORBIDDEN_INFERENCE_CONDITIONS = (
    "target_video",
    "paired_oracle_plan",
    "mask",
    "track",
    "pose",
    "optical_flow",
    "trajectory",
    "first_frame_anchor",
)


def sampler_contract() -> dict[str, Any]:
    """Return the auditable inference and integration contract."""

    return {
        "inference_conditions": list(INFERENCE_CONDITIONS),
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
        "required_plan_provenance": "student",
        "interception": "after_cfg_apg_before_scheduler_step",
        "clean_projection_executor": "phase_transport.execute_packed_velocity",
        "integrator": "original_unipc_scheduler_step",
        "custom_euler_integrator": False,
        "packed_latent_phases": 21,
        "default_max_generate_fraction": 0.12,
        "generate_budget_scope": "each_sample_each_latent_phase_spatial_mean",
        "generate_budget_policy": "fail_before_scheduler_step",
        "unbounded_generate_budget_mode": "explicit_offline_oracle_ablation_only",
        "zero_sigma_policy": "exact_velocity_bypass_no_division",
        "projection_error_policy": "raise_without_calling_scheduler_step",
    }


@dataclass(frozen=True)
class ProjectionStepRecord:
    """One successfully integrated official scheduler step."""

    step_index: int
    timestep: float
    sigma: float
    projection_applied: bool
    correction_rms: float
    preserve_fraction: float
    transport_fraction: float
    generate_fraction: float
    max_sample_generate_fraction: float
    max_phase_generate_fraction: float
    generate_budget: float | None


@dataclass
class ProjectionTrace:
    """Tensor-free trace safe to serialize after distributed inference."""

    records: list[ProjectionStepRecord] = field(default_factory=list)
    max_generate_fraction: float | None = 0.12
    oracle_ablation: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": sampler_contract(),
            "max_generate_fraction": self.max_generate_fraction,
            "oracle_ablation": self.oracle_ablation,
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


def _coerce_scalar_float(value: Any, *, label: str) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and int(value.numel()) != 1:
            raise UniPCProjectionError(f"{label} must be scalar")
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        result = float(value)
    except UniPCProjectionError:
        raise
    except Exception as error:
        raise UniPCProjectionError(f"{label} must be a numeric scalar") from error
    if not math.isfinite(result):
        raise UniPCProjectionError(f"{label} must be finite")
    return result


def _coerce_index(value: Any, *, label: str) -> int:
    numeric = _coerce_scalar_float(value, label=label)
    integer = int(numeric)
    if numeric != float(integer) or integer < 0:
        raise UniPCProjectionError(f"{label} must be a non-negative integer")
    return integer


def _fallback_timestep_index(timesteps: Any, timestep: Any) -> int:
    """Mirror diffusers' duplicate-timestep choice without mutating state."""

    try:
        import torch

        schedule = torch.as_tensor(timesteps).reshape(-1)
        query = torch.as_tensor(timestep, device=schedule.device, dtype=schedule.dtype)
        if query.numel() != 1:
            raise UniPCProjectionError("timestep must be scalar")
        indices = (schedule == query.reshape(())).nonzero(as_tuple=False).reshape(-1)
        count = int(indices.numel())
        if count == 0:
            raise UniPCProjectionError("timestep is absent from scheduler.timesteps")
        # Diffusers deliberately selects the second of duplicated initial
        # timesteps so a mid-schedule start cannot skip a sigma accidentally.
        position = 1 if count > 1 else 0
        return int(indices[position].item())
    except UniPCProjectionError:
        raise
    except Exception as error:
        raise UniPCProjectionError("cannot resolve timestep in scheduler.timesteps") from error


def resolve_current_sigma(scheduler: Any, timestep: Any) -> tuple[int, Any, float]:
    """Resolve UniPC's current sigma before ``step`` changes ``step_index``.

    On the first call diffusers has ``step_index is None``.  We honor an
    explicit begin index, then its public ``index_for_timestep`` helper, and
    finally use an exact, duplicate-aware lookup.  No scheduler field is
    mutated, so a failed projection cannot advance solver state.
    """

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise UniPCProjectionError("scheduler must expose sigmas")

    current = getattr(scheduler, "step_index", None)
    if current is not None:
        index = _coerce_index(current, label="scheduler.step_index")
    else:
        begin = getattr(scheduler, "begin_index", None)
        if begin is None:
            begin = getattr(scheduler, "_begin_index", None)
        if begin is not None:
            index = _coerce_index(begin, label="scheduler.begin_index")
        else:
            resolver = getattr(scheduler, "index_for_timestep", None)
            if callable(resolver):
                try:
                    index = _coerce_index(
                        resolver(timestep), label="scheduler timestep index"
                    )
                except UniPCProjectionError:
                    raise
                except Exception as error:
                    raise UniPCProjectionError(
                        "scheduler.index_for_timestep failed"
                    ) from error
            else:
                timesteps = getattr(scheduler, "timesteps", None)
                if timesteps is None:
                    raise UniPCProjectionError(
                        "scheduler must expose timesteps before its first step"
                    )
                index = _fallback_timestep_index(timesteps, timestep)

    try:
        sigma_count = len(sigmas)
    except Exception as error:
        raise UniPCProjectionError("scheduler.sigmas must be indexable") from error
    if index >= sigma_count:
        raise UniPCProjectionError(
            f"scheduler sigma index {index} is outside [0,{sigma_count})"
        )
    sigma = sigmas[index]
    sigma_float = _coerce_scalar_float(sigma, label="scheduler sigma")
    if sigma_float < 0.0:
        raise UniPCProjectionError("scheduler sigma must be non-negative")
    return index, sigma, sigma_float


def _extract_step_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional and keyword:
        raise UniPCProjectionError(f"scheduler.step received duplicate {name}")
    if positional:
        return args[index]
    if keyword:
        return kwargs[name]
    raise UniPCProjectionError(f"scheduler.step is missing {name}")


def _replace_model_output(
    args: Sequence[Any], kwargs: Mapping[str, Any], projected: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    new_args = list(args)
    new_kwargs = dict(kwargs)
    if new_args:
        if "model_output" in new_kwargs:
            raise UniPCProjectionError(
                "scheduler.step received duplicate model_output"
            )
        new_args[0] = projected
    elif "model_output" in new_kwargs:
        new_kwargs["model_output"] = projected
    else:
        raise UniPCProjectionError("scheduler.step is missing model_output")
    return tuple(new_args), new_kwargs


def _tensor_rms(value: Any) -> float:
    import torch

    result = torch.sqrt(torch.mean(value.float().square()))
    scalar = _coerce_scalar_float(result, label="correction RMS")
    if scalar < 0.0:  # Defensive; sqrt should make this impossible.
        raise UniPCProjectionError("correction RMS must be non-negative")
    return scalar


def _gate_fractions(plan: PhasePlan) -> tuple[float, float, float]:
    values = tuple(
        _coerce_scalar_float(
            plan.gate_probs[:, gate].float().mean(), label=f"gate fraction {gate}"
        )
        for gate in (GATE_PRESERVE, GATE_TRANSPORT, GATE_GENERATE)
    )
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise UniPCProjectionError("mean P/T/G fractions must sum to one")
    return values


def _validate_generate_budget(
    plan: PhasePlan,
    max_generate_fraction: float | None,
    *,
    allow_unbounded_generate_oracle_ablation: bool,
) -> tuple[float | None, float, float]:
    """Validate a fixed plan without hiding leakage by renormalizing gates."""

    if max_generate_fraction is None:
        if not allow_unbounded_generate_oracle_ablation:
            raise UniPCProjectionError(
                "max_generate_fraction=None is restricted to an explicit oracle ablation"
            )
        budget = None
    else:
        try:
            budget = float(max_generate_fraction)
        except Exception as error:
            raise UniPCProjectionError(
                "max_generate_fraction must be numeric or None"
            ) from error
        if not math.isfinite(budget) or not 0.0 <= budget <= 1.0:
            raise UniPCProjectionError(
                "max_generate_fraction must be finite and lie in [0,1]"
            )
        if allow_unbounded_generate_oracle_ablation:
            raise UniPCProjectionError(
                "unbounded oracle ablation requires max_generate_fraction=None"
            )

    # Enforce per sample rather than allowing one batch member's redraw to be
    # hidden by another's preserve-heavy plan.  Bernini inference is normally
    # B=1, but this retains the same safety invariant for a future batched API.
    per_sample = plan.gate_probs[:, GATE_GENERATE].float().mean(dim=(1, 2, 3))
    per_phase = plan.gate_probs[:, GATE_GENERATE].float().mean(dim=(2, 3))
    torch = __import__("torch")
    if not bool(torch.isfinite(per_sample).all()) or not bool(torch.isfinite(per_phase).all()):
        raise UniPCProjectionError("generate gate fraction is non-finite")
    actual_max_sample = _coerce_scalar_float(
        per_sample.max(), label="maximum per-sample generate fraction"
    )
    actual_max_phase = _coerce_scalar_float(
        per_phase.max(), label="maximum per-phase generate fraction"
    )
    if budget is not None and actual_max_phase > budget + 2e-5:
        raise UniPCProjectionError(
            "student plan exceeds per-phase generate budget: "
            f"actual={actual_max_phase:.6f} budget={budget:.6f}"
        )
    return budget, actual_max_sample, actual_max_phase


class _InstalledProjection:
    """Stateful instance-level wrapper with exact restoration semantics."""

    def __init__(
        self,
        scheduler: Any,
        *,
        source_packed: Any,
        plan: PhasePlan,
        height: int,
        width: int,
        detach_source_bank: bool,
        max_generate_fraction: float | None,
        allow_unbounded_generate_oracle_ablation: bool,
    ) -> None:
        if allow_unbounded_generate_oracle_ablation:
            if plan.provenance != "oracle_pair_proxy":
                raise UniPCProjectionError(
                    "explicit unbounded oracle ablation requires an oracle_pair_proxy plan"
                )
        elif plan.provenance != "student":
            raise UniPCProjectionError(
                "inference requires a source+instruction student plan; paired oracle plans are forbidden"
            )
        try:
            source_video = packed_to_video(
                source_packed, height=height, width=width
            )
            plan.validate(source_video)
        except PhaseTransportError as error:
            raise UniPCProjectionError(str(error)) from error
        self.scheduler = scheduler
        self.source_packed = source_packed
        self.plan = plan
        self.height = height
        self.width = width
        self.detach_source_bank = detach_source_bank
        self._fractions = _gate_fractions(plan)
        (
            self.max_generate_fraction,
            self._max_sample_generate_fraction,
            self._max_phase_generate_fraction,
        ) = _validate_generate_budget(
            plan,
            max_generate_fraction,
            allow_unbounded_generate_oracle_ablation=(
                allow_unbounded_generate_oracle_ablation
            ),
        )
        self.trace = ProjectionTrace(
            max_generate_fraction=self.max_generate_fraction,
            oracle_ablation=allow_unbounded_generate_oracle_ablation,
        )
        try:
            instance_dict = vars(scheduler)
        except TypeError as error:
            raise UniPCProjectionError(
                "scheduler must permit a reversible instance-level step wrapper"
            ) from error
        self._had_instance_step = "step" in instance_dict
        self._old_instance_step = instance_dict.get("step")
        self._original_step = getattr(scheduler, "step", None)
        if not callable(self._original_step):
            raise UniPCProjectionError("scheduler.step must be callable")
        if getattr(self._original_step, "_spt_unipc_projection", None) is not None:
            raise UniPCProjectionError("scheduler.step already has an SPT projection wrapper")
        self._installed = False

    def _wrapped_step(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        model_output = _extract_step_argument(
            args, kwargs, index=0, name="model_output"
        )
        timestep = _extract_step_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_step_argument(args, kwargs, index=2, name="sample")

        try:
            if getattr(model_output, "ndim", None) != 3:
                raise UniPCProjectionError("model_output must be packed [B,N,D]")
            if getattr(sample, "ndim", None) != 3:
                raise UniPCProjectionError("sample must be packed [B,N,D]")
            expected_shape = tuple(int(value) for value in self.source_packed.shape)
            if tuple(int(value) for value in model_output.shape) != expected_shape:
                raise UniPCProjectionError("model_output/source packed shapes differ")
            if tuple(int(value) for value in sample.shape) != expected_shape:
                raise UniPCProjectionError("sample/source packed shapes differ")
            if model_output.device != sample.device or sample.device != self.source_packed.device:
                raise UniPCProjectionError(
                    "model_output, sample, and source must share one device"
                )
            if self.plan.offsets.device != sample.device or self.plan.gate_probs.device != sample.device:
                raise UniPCProjectionError("student plan and sampler tensors must share one device")

            step_index, sigma, sigma_float = resolve_current_sigma(
                self.scheduler, timestep
            )
            timestep_float = _coerce_scalar_float(timestep, label="timestep")
            if sigma_float == 0.0:
                # A terminal/synthetic zero-sigma call cannot form
                # (noisy-clean)/sigma.  Preserve the official model output
                # object exactly and let UniPC own whatever terminal behavior
                # its implementation defines.
                projected = model_output
                correction_rms = 0.0
                projection_applied = False
            else:
                with torch.no_grad():
                    projected = execute_packed_velocity(
                        source_packed=self.source_packed,
                        noisy_packed=sample,
                        base_velocity_packed=model_output,
                        sigma=sigma,
                        height=self.height,
                        width=self.width,
                        plan=self.plan,
                        detach_source_bank=self.detach_source_bank,
                    ).to(device=model_output.device, dtype=model_output.dtype)
                    if not bool(torch.isfinite(projected).all()):
                        raise UniPCProjectionError(
                            "SPT projected velocity contains non-finite values"
                        )
                    correction_rms = _tensor_rms(
                        projected.float() - model_output.float()
                    )
                projection_applied = True
        except UniPCProjectionError:
            # Fail closed: never send an unprojected fallback velocity into
            # UniPC after a projection-contract failure.
            raise
        except Exception as error:
            raise UniPCProjectionError(
                "SPT projection failed before scheduler integration"
            ) from error

        call_args, call_kwargs = _replace_model_output(args, kwargs, projected)
        result = self._original_step(*call_args, **call_kwargs)
        preserve, transport, generate = self._fractions
        self.trace.records.append(
            ProjectionStepRecord(
                step_index=step_index,
                timestep=timestep_float,
                sigma=sigma_float,
                projection_applied=projection_applied,
                correction_rms=correction_rms,
                preserve_fraction=preserve,
                transport_fraction=transport,
                generate_fraction=generate,
                max_sample_generate_fraction=self._max_sample_generate_fraction,
                max_phase_generate_fraction=self._max_phase_generate_fraction,
                generate_budget=self.max_generate_fraction,
            )
        )
        return result

    def install(self) -> None:
        if self._installed:
            raise UniPCProjectionError("SPT projection wrapper is already installed")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_step(*args, **kwargs)

        # Marker makes accidental nested wrappers fail closed.
        setattr(wrapper, "_spt_unipc_projection", self)
        try:
            setattr(self.scheduler, "step", wrapper)
        except Exception as error:
            raise UniPCProjectionError("cannot install scheduler.step wrapper") from error
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        try:
            if self._had_instance_step:
                setattr(self.scheduler, "step", self._old_instance_step)
            else:
                delattr(self.scheduler, "step")
        finally:
            self._installed = False


@contextmanager
def project_unipc_steps(
    scheduler: Any,
    *,
    source_packed: Any,
    plan: PhasePlan,
    height: int,
    width: int,
    detach_source_bank: bool = True,
    max_generate_fraction: float | None = 0.12,
    allow_unbounded_generate_oracle_ablation: bool = False,
) -> Iterator[ProjectionTrace]:
    """Temporarily project packed velocities before official UniPC steps.

    Usage at the Bernini renderer boundary is intentionally small::

        plan = planner(source_video, instruction_embedding)
        with project_unipc_steps(
            model.diff_dec.scheduler,
            source_packed=source_packed,
            plan=plan,
            height=latent_height // 2,
            width=latent_width // 2,
        ) as trace:
            edited = model.sample(..., guidance_mode="v2v_apg")

    The yielded trace stores only Python scalars.  The original ``step`` state
    is restored on normal exit and on every exception.

    ``max_generate_fraction=None`` is not a deployable inference setting.  It
    requires ``allow_unbounded_generate_oracle_ablation=True`` together with
    an ``oracle_pair_proxy`` plan and is marked as such in the trace.
    """

    bridge = _InstalledProjection(
        scheduler,
        source_packed=source_packed,
        plan=plan,
        height=height,
        width=width,
        detach_source_bank=detach_source_bank,
        max_generate_fraction=max_generate_fraction,
        allow_unbounded_generate_oracle_ablation=(
            allow_unbounded_generate_oracle_ablation
        ),
    )
    bridge.install()
    try:
        yield bridge.trace
    finally:
        bridge.restore()


__all__ = [
    "FORBIDDEN_INFERENCE_CONDITIONS",
    "INFERENCE_CONDITIONS",
    "ProjectionStepRecord",
    "ProjectionTrace",
    "UniPCProjectionError",
    "project_unipc_steps",
    "resolve_current_sigma",
    "sampler_contract",
]

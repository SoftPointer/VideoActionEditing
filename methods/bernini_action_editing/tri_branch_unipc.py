#!/usr/bin/env python3
"""Method-owned tri-branch clean-field hook for pinned Bernini ``v2v_apg``.

The pinned Bernini sampler performs two transformer forwards on one assembled
``vi_inp`` at every solver step: negative text followed by action text.  This
module installs reversible *instance-level* wrappers around ``GEN_Wanx22``'s
``sample``/``shared_step`` and its scheduler's ``step``.  The wrappers:

1. retain the single negative forward;
2. run action and semantic-noop forwards on the exact same latent, rotary and
   timestep objects;
3. form action/noop APG clean estimates with distinct momentum buffers;
4. pass both clean fields to a method-owned callback; and
5. convert the callback result back to a velocity immediately before calling
   the original UniPC ``scheduler.step`` exactly once.

No Bernini vendor file is modified and no alternative Euler integrator is
introduced.  The callback receives only sampler-internal predictions and may
close over a source latent or a source+instruction router.  It does not accept
target video, masks, tracks, pose, flow, or trajectories.

The implementation is deliberately fail-closed.  It is pinned to the audited
Bernini commit and checks the observed forward order/state at runtime; a
different sampler call graph raises before UniPC can consume a guessed field.
PyTorch is imported lazily so orchestration/contract tests remain lightweight.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


PINNED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
PINNED_WAN_DIFFUSION_SHA256 = (
    "59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512"
)
PACK_PATCH_HEIGHT = 2
PACK_PATCH_WIDTH = 2


class TriBranchHookError(RuntimeError):
    """Raised before integration when the pinned tri-branch contract fails."""


@dataclass(frozen=True)
class PackedLatentLayout:
    """Wan 1x2x2 latent packing geometry."""

    batch: int
    channels: int
    frames: int
    height: int
    width: int
    tokens: int
    packed_channels: int

    @classmethod
    def from_spatial_shape(cls, shape: Sequence[int]) -> "PackedLatentLayout":
        values = tuple(shape)
        if len(values) != 5 or any(type(value) is not int for value in values):
            raise TriBranchHookError(
                f"latent_shape must be five integers [B,C,T,H,W], got {values!r}"
            )
        batch, channels, frames, height, width = values
        if min(values) <= 0:
            raise TriBranchHookError("latent_shape dimensions must be positive")
        if height % PACK_PATCH_HEIGHT or width % PACK_PATCH_WIDTH:
            raise TriBranchHookError(
                "latent height/width must be divisible by Bernini's 2x2 patch"
            )
        return cls(
            batch=batch,
            channels=channels,
            frames=frames,
            height=height,
            width=width,
            tokens=frames * (height // 2) * (width // 2),
            packed_channels=channels * 4,
        )

    @property
    def packed_shape(self) -> tuple[int, int, int]:
        return (self.batch, self.tokens, self.packed_channels)


@dataclass(frozen=True)
class APGParameters:
    """The parameters captured from one official ``GEN_Wanx22.sample`` call."""

    guidance_scale: float
    omega_scale: float
    scale_transformer_2: bool
    eta: float
    norm_threshold: float
    momentum: float

    def guidance_scale_for(self, model_id: str) -> float:
        """Mirror the pinned sampler's one-time low-noise expert scaling."""

        if model_id == "transformer_1":
            return self.guidance_scale
        if model_id == "transformer_2":
            if self.scale_transformer_2:
                return self.guidance_scale * self.omega_scale
            return self.guidance_scale
        raise TriBranchHookError(f"unexpected Bernini model_id {model_id!r}")


@dataclass(frozen=True)
class CleanFieldStep:
    """Spatial clean predictions presented to the method callback.

    All tensor-valued fields use ``[B,C,T,H,W]``.  ``action_delta_clean`` is
    the independently guided action field minus the independently guided noop
    field; it is the generator-native motion proposal a sparse router can gate.
    """

    step_index: int
    timestep: float
    sigma: float
    model_id: str
    noisy: Any
    negative_velocity: Any
    action_velocity: Any
    noop_velocity: Any
    negative_clean: Any
    action_condition_clean: Any
    noop_condition_clean: Any
    action_guided_clean: Any
    noop_guided_clean: Any
    action_delta_clean: Any


@dataclass(frozen=True)
class RawTriBranchStep:
    """One captured pre-UniPC step supplied to an injectable projector."""

    step_index: int
    timestep: Any
    timestep_float: float
    sigma: Any
    sigma_float: float
    model_id: str
    sample_packed: Any
    official_model_output: Any
    negative_velocity_packed: Any
    action_velocity_packed: Any
    noop_velocity_packed: Any
    apg: APGParameters
    layout: PackedLatentLayout


@dataclass(frozen=True)
class ProjectedVelocity:
    """A projector result and an optional scalar diagnostic."""

    model_output: Any
    correction_rms: Optional[float] = None
    raw_action_noop_delta_rms: Optional[float] = None
    guided_action_noop_delta_rms: Optional[float] = None
    guided_action_noop_delta_l2: Optional[float] = None
    action_noop_exact_parity: Optional[bool] = None
    effective_guidance_scale: Optional[float] = None
    official_action_parity_rms_error: Optional[float] = None
    official_action_parity_max_abs_error: Optional[float] = None
    official_action_exact_parity: Optional[bool] = None
    sample_dtype: Optional[str] = None
    branch_velocity_dtype: Optional[str] = None
    official_model_output_dtype: Optional[str] = None


@dataclass(frozen=True)
class TriBranchStepRecord:
    """Tensor-free receipt for one successful original UniPC call."""

    step_index: int
    timestep: float
    sigma: float
    model_id: str
    transformer_forwards: int
    shared_negative_forwards: int
    action_forwards: int
    noop_forwards: int
    original_scheduler_calls: int
    callback_correction_rms: Optional[float]
    raw_action_noop_delta_rms: Optional[float]
    guided_action_noop_delta_rms: Optional[float]
    guided_action_noop_delta_l2: Optional[float]
    action_noop_exact_parity: Optional[bool]
    effective_guidance_scale: Optional[float]
    official_action_parity_rms_error: Optional[float]
    official_action_parity_max_abs_error: Optional[float]
    official_action_exact_parity: Optional[bool]
    sample_dtype: Optional[str]
    branch_velocity_dtype: Optional[str]
    official_model_output_dtype: Optional[str]


@dataclass
class TriBranchTrace:
    """Runtime trace populated only after successful scheduler integration."""

    records: list[TriBranchStepRecord] = field(default_factory=list)
    sample_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": sampler_contract(),
            "sample_calls": self.sample_calls,
            "step_count": len(self.records),
            "steps": [asdict(record) for record in self.records],
        }


class _MomentumBuffer:
    """API-compatible, branch-local copy of Bernini's APG momentum state."""

    def __init__(self, momentum: float, *, branch: str) -> None:
        self.momentum = float(momentum)
        self.branch = branch
        self.running_average: Any = 0
        self.update_count = 0

    def update(self, update_value: Any) -> None:
        self.running_average = update_value + self.momentum * self.running_average
        self.update_count += 1


CleanFieldCallback = Callable[[CleanFieldStep], Any]
StepProjector = Callable[..., ProjectedVelocity]


def sampler_contract() -> dict[str, Any]:
    """Return the auditable method/sampler boundary."""

    return {
        "method": "bernini_generator_native_tri_branch_clean_field_v1",
        "pinned_bernini_commit": PINNED_BERNINI_COMMIT,
        "pinned_wan_diffusion_sha256": PINNED_WAN_DIFFUSION_SHA256,
        "guidance_mode": "v2v_apg",
        "interception": "raw_shared_step_branches_then_before_original_unipc_step",
        "per_step_transformer_forwards": ["shared_negative", "action", "noop"],
        "branch_state_identity": [
            "model_id",
            "vi_inp_object",
            "rotary_object",
            "timestep_object",
            "batch_vae_seqlen",
        ],
        "apg_momentum": "independent_action_and_noop_buffers",
        "official_action_apg_certificate": (
            "exact_tensor_equality_before_every_original_unipc_step"
        ),
        "apg_dtype_order": "pinned_native_dtype_sigma_times_velocity_then_fp32_noisy_subtract",
        "expert_guidance_scale": "omega_txt_then_omega_txt_times_omega_scale_after_real_t2_switch",
        "clean_field": "guided_action_clean_minus_guided_noop_clean",
        "integrator": "one_original_unipc_scheduler_step_per_diffusion_step",
        "custom_euler_integrator": False,
        "vendor_source_modified": False,
        "expected_transformer_cost_vs_official_v2v_apg": 1.5,
        "external_inference_conditions": ["source_video", "action_instruction"],
        "internal_fixed_controls": ["noop_instruction", "negative_prompt"],
        "forbidden_inference_conditions": [
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ],
        "failure_policy": "raise_before_original_scheduler_step",
        "scheduler_contract": {
            "class": "UniPCMultistepScheduler",
            "prediction_type": "flow_prediction",
            "predict_x0": True,
            "use_flow_sigmas": True,
            "thresholding": False,
            "solver_order": 2,
            "solver_type": "bh2",
        },
    }


def resolve_diffusion_core(renderer_or_diffusion: Any) -> Any:
    """Resolve the method-ownable ``GEN_Wanx22`` instance through wrappers."""

    queue = [renderer_or_diffusion]
    seen: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if all(
            callable(getattr(candidate, name, None))
            for name in ("sample", "shared_step")
        ) and callable(getattr(getattr(candidate, "scheduler", None), "step", None)):
            return candidate
        nested = getattr(candidate, "diff_dec", None)
        if nested is not None:
            queue.append(nested)
        get_base_model = getattr(candidate, "get_base_model", None)
        if callable(get_base_model):
            try:
                queue.append(get_base_model())
            except Exception:
                pass
        for name in ("base_model", "model", "module"):
            nested = getattr(candidate, name, None)
            if nested is not None:
                queue.append(nested)
    raise TriBranchHookError("could not resolve a GEN_Wanx22-compatible diffusion core")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TriBranchHookError(f"cannot hash pinned Bernini source file: {path}") from error
    return digest.hexdigest()


def validate_runtime_source_identity(
    *, bernini_commit: str, wan_diffusion_path: str | Path
) -> str:
    """Bind installation to the actual pinned sampler bytes, not a declaration."""

    if bernini_commit != PINNED_BERNINI_COMMIT:
        raise TriBranchHookError(
            "Bernini revision differs from the tri-branch audited revision"
        )
    path = Path(wan_diffusion_path).expanduser()
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise TriBranchHookError(
            "wan_diffusion_path must be an absolute, plain, existing file"
        )
    actual = _file_sha256(path)
    if actual != PINNED_WAN_DIFFUSION_SHA256:
        raise TriBranchHookError(
            "bernini/models/wan_diffusion.py differs from the audited bytes"
        )
    return actual


def _config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None and isinstance(config, Mapping):
        value = config.get(name)
    return value


def _validate_scheduler_contract(scheduler: Any, *, expected_flow_shift: float) -> None:
    config = getattr(scheduler, "config", None)
    if config is None:
        raise TriBranchHookError("UniPC scheduler must expose config")
    configured_class = _config_value(config, "_class_name")
    if (
        type(scheduler).__name__ != "UniPCMultistepScheduler"
        and configured_class != "UniPCMultistepScheduler"
    ):
        raise TriBranchHookError("scheduler must be UniPCMultistepScheduler")
    configured_shift = _config_value(config, "flow_shift")
    if configured_shift is None or not math.isclose(
        _coerce_scalar(configured_shift, label="scheduler flow_shift"),
        expected_flow_shift,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise TriBranchHookError(
            "the constructed UniPC scheduler does not have the expected flow_shift"
        )
    expected = {
        "prediction_type": "flow_prediction",
        "predict_x0": True,
        "use_flow_sigmas": True,
        "thresholding": False,
        "solver_order": 2,
        "solver_type": "bh2",
    }
    for name, wanted in expected.items():
        observed = _config_value(config, name)
        if type(wanted) is bool:
            matches = observed is wanted
        elif type(wanted) is int:
            matches = type(observed) is int and observed == wanted
        else:
            matches = observed == wanted
        if not matches:
            raise TriBranchHookError(
                f"scheduler {name} must equal {wanted!r}, got {observed!r}"
            )


def _coerce_scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise TriBranchHookError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except TriBranchHookError:
        raise
    except Exception as error:
        raise TriBranchHookError(f"{label} must be a numeric scalar") from error
    if not math.isfinite(result):
        raise TriBranchHookError(f"{label} must be finite")
    return result


def _coerce_index(value: Any, *, label: str) -> int:
    numeric = _coerce_scalar(value, label=label)
    integer = int(numeric)
    if numeric != float(integer) or integer < 0:
        raise TriBranchHookError(f"{label} must be a non-negative integer")
    return integer


def _resolve_sigma(scheduler: Any, timestep: Any) -> tuple[int, Any, float]:
    """Resolve the current sigma without advancing or mutating UniPC."""

    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None:
        raise TriBranchHookError("UniPC scheduler must expose sigmas")
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
                    # Pinned Bernini passes the loop timestep on the active
                    # GPU while Diffusers keeps ``scheduler.timesteps`` on
                    # CPU.  ``UniPC.index_for_timestep`` compares those
                    # tensors directly, so normalize only the lookup value to
                    # the scheduler timeline device.  This is a read-only
                    # index query; the original timestep still reaches the
                    # untouched scheduler.step below.
                    lookup_timestep = timestep
                    schedule = getattr(scheduler, "timesteps", None)
                    schedule_device = getattr(schedule, "device", None)
                    if schedule_device is not None and hasattr(
                        lookup_timestep, "to"
                    ):
                        lookup_timestep = lookup_timestep.to(
                            device=schedule_device
                        )
                    index = _coerce_index(
                        resolver(lookup_timestep),
                        label="scheduler timestep index",
                    )
                except TriBranchHookError:
                    raise
                except Exception as error:
                    raise TriBranchHookError(
                        "scheduler.index_for_timestep failed"
                    ) from error
            else:
                schedule = getattr(scheduler, "timesteps", None)
                if schedule is None:
                    raise TriBranchHookError(
                        "scheduler must expose timesteps before its first step"
                    )
                query = _coerce_scalar(timestep, label="timestep")
                matches = [
                    position
                    for position, value in enumerate(schedule)
                    if _coerce_scalar(value, label="scheduler timestep") == query
                ]
                if not matches:
                    raise TriBranchHookError("timestep is absent from scheduler.timesteps")
                index = matches[1] if len(matches) > 1 else matches[0]
    try:
        sigma = sigmas[index]
    except Exception as error:
        raise TriBranchHookError("scheduler sigma index is invalid") from error
    sigma_float = _coerce_scalar(sigma, label="scheduler sigma")
    if sigma_float <= 0.0:
        raise TriBranchHookError(
            "tri-branch clean projection requires strictly positive current sigma"
        )
    return index, sigma, sigma_float


def _extract_argument(
    args: Sequence[Any], kwargs: Mapping[str, Any], *, index: int, name: str
) -> Any:
    positional = len(args) > index
    keyword = name in kwargs
    if positional and keyword:
        raise TriBranchHookError(f"call received duplicate {name}")
    if positional:
        return args[index]
    if keyword:
        return kwargs[name]
    raise TriBranchHookError(f"call is missing {name}")


def _replace_argument(
    callable_object: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Replace one named argument without changing the rest of a pinned call."""

    new_args = list(args)
    new_kwargs = dict(kwargs)
    if name in new_kwargs:
        new_kwargs[name] = value
        return tuple(new_args), new_kwargs
    try:
        parameters = list(inspect.signature(callable_object).parameters.values())
    except (TypeError, ValueError) as error:
        raise TriBranchHookError("cannot inspect pinned callable signature") from error
    positional_names = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if name in positional_names:
        position = positional_names.index(name)
        if position < len(new_args):
            new_args[position] = value
            return tuple(new_args), new_kwargs
    new_kwargs[name] = value
    return tuple(new_args), new_kwargs


def _bind_call(callable_object: Callable[..., Any], args: Sequence[Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(callable_object).bind(*args, **kwargs)
        bound.apply_defaults()
    except (TypeError, ValueError) as error:
        raise TriBranchHookError("call does not match pinned Bernini signature") from error
    return dict(bound.arguments)


def _same_object(left: Any, right: Any, *, label: str) -> None:
    if left is not right:
        raise TriBranchHookError(
            f"negative/action {label} must be the exact same object"
        )


def _equal_metadata(left: Any, right: Any, *, label: str) -> None:
    try:
        equal = left == right
        if hasattr(equal, "all"):
            equal = equal.all()
        if hasattr(equal, "item"):
            equal = equal.item()
        equal = bool(equal)
    except Exception as error:
        raise TriBranchHookError(f"cannot compare negative/action {label}") from error
    if not equal:
        raise TriBranchHookError(f"negative/action {label} differ")


def _packed_to_spatial(packed: Any, layout: PackedLatentLayout) -> Any:
    actual = tuple(int(value) for value in packed.shape)
    if actual != layout.packed_shape:
        raise TriBranchHookError(
            f"packed tensor shape {actual} != expected {layout.packed_shape}"
        )
    return (
        packed.reshape(
            layout.batch,
            layout.frames,
            layout.height // 2,
            layout.width // 2,
            2,
            2,
            layout.channels,
        )
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(
            layout.batch,
            layout.channels,
            layout.frames,
            layout.height,
            layout.width,
        )
    )


def _spatial_to_packed(spatial: Any, layout: PackedLatentLayout) -> Any:
    expected = (
        layout.batch,
        layout.channels,
        layout.frames,
        layout.height,
        layout.width,
    )
    actual = tuple(int(value) for value in spatial.shape)
    if actual != expected:
        raise TriBranchHookError(f"spatial tensor shape {actual} != expected {expected}")
    return (
        spatial.reshape(
            layout.batch,
            layout.channels,
            layout.frames,
            layout.height // 2,
            2,
            layout.width // 2,
            2,
        )
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(layout.packed_shape)
    )


def _normalized_guidance(
    pred_cond: Any,
    pred_uncond: Any,
    guidance_scale: float,
    momentum_buffer: _MomentumBuffer,
    eta: float,
    norm_threshold: float,
) -> Any:
    """Exact local equivalent of pinned Bernini ``normalized_guidance``."""

    try:
        import torch
        import torch.nn.functional as torch_f
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise TriBranchHookError("PyTorch is required for clean-field projection") from error

    diff = pred_cond - pred_uncond
    momentum_buffer.update(diff)
    diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), pred_cond.double()
    v1 = torch_f.normalize(v1, dim=[-1, -2, -4])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    normalized = v0_orthogonal.to(diff.dtype) + eta * v0_parallel.to(diff.dtype)
    return pred_uncond + guidance_scale * normalized


def _tensor_rms(value: Any) -> float:
    try:
        result = value.float().square().mean().sqrt()
    except Exception as error:
        raise TriBranchHookError("cannot compute projected velocity RMS") from error
    return _coerce_scalar(result, label="projected velocity RMS")


def _tensor_l2(value: Any) -> float:
    try:
        result = value.float().norm(p=2)
    except Exception as error:
        raise TriBranchHookError("cannot compute clean-field L2 norm") from error
    return _coerce_scalar(result, label="clean-field L2 norm")


def pinned_raw_condition_clean(
    noisy_fp32: Any,
    native_velocity_bf16: Any,
    sigma_cpu_fp32: Any,
) -> Any:
    """Execute Bernini's raw v-prediction-to-clean numerical program exactly.

    The CPU 0-d sigma is intentional.  PyTorch treats it as a wrapped scalar
    when multiplying the accelerator BF16 prediction; moving it to the GPU or
    casting the velocity to FP32 first is a different TensorIterator program.
    This helper is shared by inference and C2FR training so algebraic alignment
    cannot conceal a dtype/device train-test gap.
    """

    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise TriBranchHookError("PyTorch is required for clean projection") from error
    if not isinstance(noisy_fp32, torch.Tensor) or not isinstance(
        native_velocity_bf16, torch.Tensor
    ):
        raise TriBranchHookError("noisy state and velocity must be tensors")
    if tuple(noisy_fp32.shape) != tuple(native_velocity_bf16.shape):
        raise TriBranchHookError("noisy state and velocity shapes differ")
    if noisy_fp32.dtype != torch.float32:
        raise TriBranchHookError("pinned Bernini noisy state must be fp32")
    if native_velocity_bf16.dtype != torch.bfloat16:
        raise TriBranchHookError("pinned Bernini raw branch velocity must be bf16")
    if noisy_fp32.device != native_velocity_bf16.device:
        raise TriBranchHookError("noisy state and branch velocity devices differ")
    if (
        not isinstance(sigma_cpu_fp32, torch.Tensor)
        or sigma_cpu_fp32.ndim != 0
        or sigma_cpu_fp32.device.type != "cpu"
        or sigma_cpu_fp32.dtype != torch.float32
        or not bool(torch.isfinite(sigma_cpu_fp32))
        or not bool(sigma_cpu_fp32 > 0)
    ):
        raise TriBranchHookError(
            "pinned UniPC APG sigma must be one finite positive CPU fp32 scalar"
        )
    return noisy_fp32 - sigma_cpu_fp32 * native_velocity_bf16


def project_clean_fields(
    raw: RawTriBranchStep,
    *,
    action_momentum: _MomentumBuffer,
    noop_momentum: _MomentumBuffer,
    clean_field_callback: CleanFieldCallback,
) -> ProjectedVelocity:
    """Default tensor projector used immediately before original UniPC.

    This is public so an AUH smoke can exercise the numerical boundary without
    loading Bernini.  Normal adapter use reaches it through
    :func:`tri_branch_unipc_hook`.
    """

    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised on AUH
        raise TriBranchHookError("PyTorch is required for clean-field projection") from error

    tensors = (
        raw.sample_packed,
        raw.official_model_output,
        raw.negative_velocity_packed,
        raw.action_velocity_packed,
        raw.noop_velocity_packed,
    )
    if any(getattr(tensor, "ndim", None) != 3 for tensor in tensors):
        raise TriBranchHookError("all pre-UniPC tensors must be packed [B,N,D]")
    labels = ("sample", "official", "negative", "action", "noop")
    observed_shapes = {
        label: tuple(int(value) for value in tensor.shape)
        for label, tensor in zip(labels, tensors)
    }
    if any(shape != raw.layout.packed_shape for shape in observed_shapes.values()):
        raise TriBranchHookError(
            "pre-UniPC packed tensor shapes differ from "
            f"{raw.layout.packed_shape}: {observed_shapes}"
        )
    # Preserve the exact pinned dtype order.  In particular, official Bernini
    # multiplies its fp32 scalar sigma by the bf16 transformer velocity *before*
    # subtracting from the fp32 noisy sample.  Casting the velocity first is a
    # different numerical program and breaks APG parity on ROCm.
    sample = _packed_to_spatial(raw.sample_packed, raw.layout)
    official_v = _packed_to_spatial(raw.official_model_output, raw.layout)
    negative_v = _packed_to_spatial(raw.negative_velocity_packed, raw.layout)
    action_v = _packed_to_spatial(raw.action_velocity_packed, raw.layout)
    noop_v = _packed_to_spatial(raw.noop_velocity_packed, raw.layout)
    # Preserve the *object* and device used by pinned Bernini.  Diffusers
    # deliberately leaves UniPC ``scheduler.sigmas`` on CPU, and Bernini uses
    # that CPU fp32 0-d tensor directly with the GPU bf16 branch prediction.
    # Moving it to the latent device changes PyTorch's wrapped-scalar/type
    # promotion path and is measurably different on ROCm.  Reject any other
    # representation instead of silently materializing a different program.
    sigma = raw.sigma
    if (
        not isinstance(sigma, torch.Tensor)
        or sigma.ndim != 0
        or sigma.device.type != "cpu"
        or sigma.dtype != torch.float32
        or not bool(torch.isfinite(sigma))
        or not bool(sigma > 0)
    ):
        raise TriBranchHookError(
            "pinned UniPC APG sigma must be one finite positive CPU fp32 scalar"
        )

    negative_clean = pinned_raw_condition_clean(sample, negative_v, sigma)
    action_condition_clean = pinned_raw_condition_clean(sample, action_v, sigma)
    noop_condition_clean = pinned_raw_condition_clean(sample, noop_v, sigma)
    effective_guidance_scale = raw.apg.guidance_scale_for(raw.model_id)
    locally_rebuilt_action_clean = _normalized_guidance(
        action_condition_clean,
        negative_clean,
        effective_guidance_scale,
        action_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )
    noop_guided_clean = _normalized_guidance(
        noop_condition_clean,
        negative_clean,
        effective_guidance_scale,
        noop_momentum,
        raw.apg.eta,
        raw.apg.norm_threshold,
    )

    # This is a hard runtime certificate, not a soft diagnostic.  Rebuilding
    # the action branch locally must recover the exact tensor the pinned
    # sampler already supplied to scheduler.step.  It jointly certifies sigma,
    # target slicing, native dtype order, APG momentum, norm threshold and the
    # expert-specific guidance scale before the callback is allowed to run.
    locally_rebuilt_action_velocity = _spatial_to_packed(
        (sample - locally_rebuilt_action_clean) / sigma, raw.layout
    ).to(
        device=raw.official_model_output.device,
        dtype=raw.official_model_output.dtype,
    )
    official_parity_error = (
        locally_rebuilt_action_velocity.float()
        - raw.official_model_output.float()
    )
    official_parity_rms = _tensor_rms(official_parity_error)
    official_parity_max = _coerce_scalar(
        official_parity_error.abs().max(), label="official action parity max error"
    )
    official_parity_exact = bool(
        torch.equal(locally_rebuilt_action_velocity, raw.official_model_output)
    )
    if not official_parity_exact:
        raise TriBranchHookError(
            "local action APG does not exactly match pinned official model_output: "
            f"max_abs={official_parity_max:.9g} rms={official_parity_rms:.9g}"
        )

    # Use the official action field as the authoritative branch after the
    # certificate.  The local rebuild exists only to prove the contract.
    action_guided_clean = sample - sigma * official_v
    raw_delta = action_v - noop_v
    guided_delta = action_guided_clean - noop_guided_clean
    fields = CleanFieldStep(
        step_index=raw.step_index,
        timestep=raw.timestep_float,
        sigma=raw.sigma_float,
        model_id=raw.model_id,
        noisy=sample,
        negative_velocity=negative_v,
        action_velocity=action_v,
        noop_velocity=noop_v,
        negative_clean=negative_clean,
        action_condition_clean=action_condition_clean,
        noop_condition_clean=noop_condition_clean,
        action_guided_clean=action_guided_clean,
        noop_guided_clean=noop_guided_clean,
        action_delta_clean=guided_delta,
    )
    try:
        executed_clean = clean_field_callback(fields)
    except TriBranchHookError:
        raise
    except Exception as error:
        raise TriBranchHookError("clean-field callback failed") from error
    expected_spatial = (
        raw.layout.batch,
        raw.layout.channels,
        raw.layout.frames,
        raw.layout.height,
        raw.layout.width,
    )
    if tuple(int(value) for value in getattr(executed_clean, "shape", ())) != expected_spatial:
        raise TriBranchHookError("clean-field callback returned the wrong spatial shape")
    if not bool(torch.isfinite(executed_clean).all()):
        raise TriBranchHookError("clean-field callback returned non-finite values")
    if executed_clean is action_guided_clean:
        # The diagnostic control is an exact no-op at the scheduler boundary;
        # do not introduce an avoidable clean->velocity round-trip.
        executed_velocity = raw.official_model_output
    else:
        executed_velocity = _spatial_to_packed(
            (sample - executed_clean) / sigma, raw.layout
        ).to(
            device=raw.official_model_output.device,
            dtype=raw.official_model_output.dtype,
        )
    if not bool(torch.isfinite(executed_velocity).all()):
        raise TriBranchHookError("executed velocity contains non-finite values")
    return ProjectedVelocity(
        model_output=executed_velocity,
        correction_rms=_tensor_rms(
            executed_velocity.float() - raw.official_model_output.float()
        ),
        raw_action_noop_delta_rms=_tensor_rms(raw_delta),
        guided_action_noop_delta_rms=_tensor_rms(guided_delta),
        guided_action_noop_delta_l2=_tensor_l2(guided_delta),
        action_noop_exact_parity=bool(
            torch.equal(action_v, noop_v)
            and torch.equal(action_guided_clean, noop_guided_clean)
        ),
        effective_guidance_scale=effective_guidance_scale,
        official_action_parity_rms_error=official_parity_rms,
        official_action_parity_max_abs_error=official_parity_max,
        official_action_exact_parity=official_parity_exact,
        sample_dtype=str(sample.dtype),
        branch_velocity_dtype=str(action_v.dtype),
        official_model_output_dtype=str(raw.official_model_output.dtype),
    )


def action_clean_passthrough(fields: CleanFieldStep) -> Any:
    """Diagnostic callback: reconstruct official action APG in our hook."""

    return fields.action_guided_clean


def noop_clean_passthrough(fields: CleanFieldStep) -> Any:
    """Diagnostic callback: execute the independently guided noop field."""

    return fields.noop_guided_clean


def scaled_action_delta(scale: float) -> CleanFieldCallback:
    """Return ``noop + scale * (action - noop)`` without any external mask."""

    numeric = _coerce_scalar(scale, label="action delta scale")
    if numeric < 0.0:
        raise TriBranchHookError("action delta scale must be non-negative")

    def callback(fields: CleanFieldStep) -> Any:
        return fields.noop_guided_clean + numeric * fields.action_delta_clean

    return callback


@dataclass
class _CapturedForward:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    bound: dict[str, Any]
    prediction: Any


@dataclass
class _ActiveSample:
    expected_steps: int
    action_t1: Any
    action_t2: Any
    uncond_t1: Any
    uncond_t2: Any
    noop_t1: Any
    noop_t2: Any
    apg: APGParameters
    action_momentum: _MomentumBuffer
    noop_momentum: _MomentumBuffer
    pending_negative: Optional[_CapturedForward] = None
    pending_action: Optional[_CapturedForward] = None
    pending_noop: Optional[Any] = None
    integrated_steps: int = 0


def _branch_prompt(state: _ActiveSample, model_id: str, branch: str) -> Any:
    suffix = "t1" if model_id == "transformer_1" else "t2"
    if model_id not in ("transformer_1", "transformer_2"):
        raise TriBranchHookError(f"unexpected Bernini model_id {model_id!r}")
    return getattr(state, f"{branch}_{suffix}")


class _InstalledTriBranch:
    """Reversible wrapper set for one diffusion core and scheduler instance."""

    def __init__(
        self,
        diffusion: Any,
        *,
        noop_prompt_embeds: Any,
        noop_prompt_embeds_t2: Any,
        latent_shape: Sequence[int],
        clean_field_callback: CleanFieldCallback,
        expected_steps: int,
        expected_flow_shift: float,
        projector: StepProjector,
        bernini_commit: str,
        wan_diffusion_path: str | Path,
    ) -> None:
        validate_runtime_source_identity(
            bernini_commit=bernini_commit,
            wan_diffusion_path=wan_diffusion_path,
        )
        if not callable(clean_field_callback):
            raise TriBranchHookError("clean_field_callback must be callable")
        if not callable(projector):
            raise TriBranchHookError("projector must be callable")
        if type(expected_steps) is not int or expected_steps <= 0:
            raise TriBranchHookError("expected_steps must be a positive integer")
        if not math.isfinite(float(expected_flow_shift)) or float(expected_flow_shift) <= 0:
            raise TriBranchHookError("expected_flow_shift must be finite and positive")
        if noop_prompt_embeds is None:
            raise TriBranchHookError("noop_prompt_embeds is required")
        self.diffusion = diffusion
        self.scheduler = diffusion.scheduler
        self.noop_t1 = noop_prompt_embeds
        self.noop_t2 = (
            noop_prompt_embeds
            if noop_prompt_embeds_t2 is None
            else noop_prompt_embeds_t2
        )
        self.layout = PackedLatentLayout.from_spatial_shape(latent_shape)
        self.clean_field_callback = clean_field_callback
        self.expected_steps = expected_steps
        self.expected_flow_shift = float(expected_flow_shift)
        self.projector = projector
        self.trace = TriBranchTrace()
        self._active: Optional[_ActiveSample] = None
        self._installed = False
        self._patches: list[tuple[Any, str, bool, Any]] = []

        self._original_sample = getattr(diffusion, "sample", None)
        self._original_shared_step = getattr(diffusion, "shared_step", None)
        self._original_scheduler_step = getattr(self.scheduler, "step", None)
        for label, function in (
            ("diffusion.sample", self._original_sample),
            ("diffusion.shared_step", self._original_shared_step),
            ("scheduler.step", self._original_scheduler_step),
        ):
            if not callable(function):
                raise TriBranchHookError(f"{label} must be callable")
        if getattr(diffusion, "use_unipc", None) is not True:
            raise TriBranchHookError("tri-branch hook requires diffusion.use_unipc is True")
        _validate_scheduler_contract(
            self.scheduler, expected_flow_shift=self.expected_flow_shift
        )
        # A normal GEN_Wanx22/UniPC instance resolves all three methods from
        # its classes.  Unknown instance overrides make the audited call graph
        # ambiguous and are rejected rather than wrapped speculatively.
        for owner, name in (
            (diffusion, "sample"),
            (diffusion, "shared_step"),
            (self.scheduler, "step"),
        ):
            try:
                already_overridden = name in vars(owner)
            except TypeError as error:
                raise TriBranchHookError(
                    f"cannot inspect {name} owner for an existing wrapper"
                ) from error
            if already_overridden:
                raise TriBranchHookError(
                    f"refusing to stack on an existing instance-level {name} override"
                )
        for function in (self._original_sample, self._original_shared_step, self._original_scheduler_step):
            if getattr(function, "_bernini_tri_branch_unipc", None) is not None:
                raise TriBranchHookError("a tri-branch UniPC hook is already installed")
        if getattr(self._original_scheduler_step, "_spt_unipc_projection", None) is not None:
            raise TriBranchHookError(
                "install tri-branch and SPT execution as one clean callback; nested scheduler wrappers are forbidden"
            )

    @staticmethod
    def _validate_prompt_identity(actual: Any, expected: Any, *, branch: str) -> None:
        if actual is not expected:
            raise TriBranchHookError(
                f"observed {branch} prompt is not the exact sample prompt object"
            )

    def _query_prediction(self, prediction: Any, *, branch: str) -> Any:
        """Mirror pinned ``_fwd``'s target mask without intercepting vendor locals.

        Pinned ``_assemble`` always appends the noisy target after all source
        tokens, and its mask is false for every source token and true for all
        ``layout.tokens`` target tokens.  Slicing the final target-token block
        is therefore exactly ``pred[:, msk, :]`` for ``v2v_apg``.  Runtime
        shape checks fail closed if that audited invariant changes.
        """

        shape = tuple(int(value) for value in getattr(prediction, "shape", ()))
        if len(shape) != 3:
            raise TriBranchHookError(f"{branch} shared_step prediction must be [B,N,D]")
        if shape[0] != self.layout.batch or shape[1] < self.layout.tokens:
            raise TriBranchHookError(
                f"{branch} shared_step prediction cannot contain the target token block"
            )
        if shape[2] != self.layout.packed_channels:
            raise TriBranchHookError(
                f"{branch} shared_step channel width differs from packed latent"
            )
        try:
            selected = prediction[:, -self.layout.tokens :, :]
        except Exception as error:
            raise TriBranchHookError(
                f"cannot select {branch} target-token prediction"
            ) from error
        if tuple(int(value) for value in selected.shape) != self.layout.packed_shape:
            raise TriBranchHookError(f"{branch} target-token selection has wrong shape")
        return selected

    def _wrapped_shared_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise TriBranchHookError("shared_step ran outside one validated sample call")
        bound = _bind_call(self._original_shared_step, args, kwargs)
        try:
            model_id = str(bound["model_id"])
            prompt = bound["cond_embeds"]
        except KeyError as error:
            raise TriBranchHookError("shared_step lacks pinned branch arguments") from error

        if state.pending_negative is None:
            expected = _branch_prompt(state, model_id, "uncond")
            self._validate_prompt_identity(prompt, expected, branch="negative")
            prediction = self._original_shared_step(*args, **kwargs)
            state.pending_negative = _CapturedForward(
                tuple(args),
                dict(kwargs),
                bound,
                self._query_prediction(prediction, branch="negative"),
            )
            return prediction

        if state.pending_action is not None or state.pending_noop is not None:
            raise TriBranchHookError(
                "more than two official shared_step calls occurred before scheduler.step"
            )
        negative = state.pending_negative
        expected = _branch_prompt(state, model_id, "action")
        self._validate_prompt_identity(prompt, expected, branch="action")
        if str(negative.bound.get("model_id")) != model_id:
            raise TriBranchHookError("negative/action model_id differ")
        for name in ("noisy_latents", "timesteps", "rotary_embs"):
            _same_object(negative.bound.get(name), bound.get(name), label=name)
        _equal_metadata(
            negative.bound.get("batch_vae_seqlen"),
            bound.get("batch_vae_seqlen"),
            label="batch_vae_seqlen",
        )
        action_prediction = self._original_shared_step(*args, **kwargs)
        state.pending_action = _CapturedForward(
            tuple(args),
            dict(kwargs),
            bound,
            self._query_prediction(action_prediction, branch="action"),
        )

        noop_prompt = _branch_prompt(state, model_id, "noop")
        noop_args, noop_kwargs = _replace_argument(
            self._original_shared_step,
            args,
            kwargs,
            name="cond_embeds",
            value=noop_prompt,
        )
        text_length = getattr(noop_prompt, "shape", None)
        if text_length is None or len(text_length) < 2:
            raise TriBranchHookError("noop prompt embedding must expose [B,L,D] shape")
        noop_args, noop_kwargs = _replace_argument(
            self._original_shared_step,
            noop_args,
            noop_kwargs,
            name="batch_text_seqlen",
            value=[int(text_length[1])],
        )
        noop_bound = _bind_call(self._original_shared_step, noop_args, noop_kwargs)
        for name in ("model_id", "noisy_latents", "timesteps", "rotary_embs"):
            left, right = bound.get(name), noop_bound.get(name)
            if name == "model_id":
                _equal_metadata(left, right, label="noop model_id")
            else:
                _same_object(left, right, label=f"action/noop {name}")
        _equal_metadata(
            bound.get("batch_vae_seqlen"),
            noop_bound.get("batch_vae_seqlen"),
            label="action/noop batch_vae_seqlen",
        )
        noop_prediction = self._original_shared_step(*noop_args, **noop_kwargs)
        state.pending_noop = self._query_prediction(noop_prediction, branch="noop")
        return action_prediction

    def _wrapped_scheduler_step(self, *args: Any, **kwargs: Any) -> Any:
        state = self._active
        if state is None:
            raise TriBranchHookError("scheduler.step ran outside one validated sample call")
        if (
            state.pending_negative is None
            or state.pending_action is None
            or state.pending_noop is None
        ):
            raise TriBranchHookError(
                "scheduler.step arrived before negative/action/noop were all evaluated"
            )
        official = _extract_argument(args, kwargs, index=0, name="model_output")
        timestep = _extract_argument(args, kwargs, index=1, name="timestep")
        sample = _extract_argument(args, kwargs, index=2, name="sample")
        step_index, sigma, sigma_float = _resolve_sigma(self.scheduler, timestep)
        negative = state.pending_negative
        action = state.pending_action
        model_id = str(action.bound["model_id"])
        raw = RawTriBranchStep(
            step_index=step_index,
            timestep=timestep,
            timestep_float=_coerce_scalar(timestep, label="timestep"),
            sigma=sigma,
            sigma_float=sigma_float,
            model_id=model_id,
            sample_packed=sample,
            official_model_output=official,
            negative_velocity_packed=negative.prediction,
            action_velocity_packed=action.prediction,
            noop_velocity_packed=state.pending_noop,
            apg=state.apg,
            layout=self.layout,
        )
        try:
            projected = self.projector(
                raw,
                action_momentum=state.action_momentum,
                noop_momentum=state.noop_momentum,
                clean_field_callback=self.clean_field_callback,
            )
        except TriBranchHookError:
            raise
        except Exception as error:
            raise TriBranchHookError(
                "tri-branch projector failed before scheduler integration"
            ) from error
        if not isinstance(projected, ProjectedVelocity):
            raise TriBranchHookError("projector must return ProjectedVelocity")
        call_args, call_kwargs = _replace_argument(
            self._original_scheduler_step,
            args,
            kwargs,
            name="model_output",
            value=projected.model_output,
        )
        # The only invocation of the original numerical integrator in this
        # wrapper.  State is cleared only after it returns successfully.
        result = self._original_scheduler_step(*call_args, **call_kwargs)
        state.integrated_steps += 1
        self.trace.records.append(
            TriBranchStepRecord(
                step_index=step_index,
                timestep=raw.timestep_float,
                sigma=sigma_float,
                model_id=model_id,
                transformer_forwards=3,
                shared_negative_forwards=1,
                action_forwards=1,
                noop_forwards=1,
                original_scheduler_calls=1,
                callback_correction_rms=projected.correction_rms,
                raw_action_noop_delta_rms=projected.raw_action_noop_delta_rms,
                guided_action_noop_delta_rms=projected.guided_action_noop_delta_rms,
                guided_action_noop_delta_l2=projected.guided_action_noop_delta_l2,
                action_noop_exact_parity=projected.action_noop_exact_parity,
                effective_guidance_scale=projected.effective_guidance_scale,
                official_action_parity_rms_error=(
                    projected.official_action_parity_rms_error
                ),
                official_action_parity_max_abs_error=(
                    projected.official_action_parity_max_abs_error
                ),
                official_action_exact_parity=projected.official_action_exact_parity,
                sample_dtype=projected.sample_dtype,
                branch_velocity_dtype=projected.branch_velocity_dtype,
                official_model_output_dtype=projected.official_model_output_dtype,
            )
        )
        state.pending_negative = None
        state.pending_action = None
        state.pending_noop = None
        return result

    def _wrapped_sample(self, *args: Any, **kwargs: Any) -> Any:
        if self._active is not None:
            raise TriBranchHookError("nested/concurrent diffusion.sample calls are forbidden")
        if self.diffusion.scheduler is not self.scheduler:
            raise TriBranchHookError("diffusion.scheduler changed after hook installation")
        values = _bind_call(self._original_sample, args, kwargs)
        if values.get("guidance_mode") != "v2v_apg":
            raise TriBranchHookError("tri-branch hook only supports guidance_mode='v2v_apg'")
        if int(values.get("num_inference_steps")) != self.expected_steps:
            raise TriBranchHookError(
                f"sample must use exactly {self.expected_steps} inference steps"
            )
        if not math.isclose(
            float(values.get("flow_shift")),
            self.expected_flow_shift,
            rel_tol=0.0,
            abs_tol=1.0e-8,
        ):
            raise TriBranchHookError(
                f"sample flow_shift differs from {self.expected_flow_shift}"
            )
        action_t1 = values.get("prompt_embeds")
        uncond_t1 = values.get("uncond_prompt_embeds")
        if action_t1 is None or uncond_t1 is None:
            raise TriBranchHookError("action and negative prompt embeddings are required")
        action_t2 = values.get("prompt_embeds_t2")
        if action_t2 is None:
            action_t2 = action_t1
        uncond_t2 = values.get("uncond_embeds_t2")
        if uncond_t2 is None:
            uncond_t2 = uncond_t1
        parameters = APGParameters(
            guidance_scale=_coerce_scalar(values.get("omega_txt"), label="omega_txt"),
            omega_scale=_coerce_scalar(values.get("omega_scale"), label="omega_scale"),
            scale_transformer_2=getattr(self.diffusion, "transformer_2", None)
            is not None,
            eta=_coerce_scalar(values.get("eta"), label="eta"),
            norm_threshold=_coerce_scalar(
                values.get("norm_threshold")[0]
                if isinstance(values.get("norm_threshold"), (list, tuple))
                else values.get("norm_threshold"),
                label="norm_threshold",
            ),
            momentum=_coerce_scalar(values.get("momentum"), label="momentum"),
        )
        if parameters.norm_threshold < 0.0:
            raise TriBranchHookError("norm_threshold must be non-negative")
        if parameters.omega_scale < 0.0:
            raise TriBranchHookError("omega_scale must be non-negative")
        state = _ActiveSample(
            expected_steps=self.expected_steps,
            action_t1=action_t1,
            action_t2=action_t2,
            uncond_t1=uncond_t1,
            uncond_t2=uncond_t2,
            noop_t1=self.noop_t1,
            noop_t2=self.noop_t2,
            apg=parameters,
            action_momentum=_MomentumBuffer(parameters.momentum, branch="action"),
            noop_momentum=_MomentumBuffer(parameters.momentum, branch="noop"),
        )
        self._active = state
        records_before = len(self.trace.records)
        try:
            result = self._original_sample(*args, **kwargs)
            if any(
                value is not None
                for value in (
                    state.pending_negative,
                    state.pending_action,
                    state.pending_noop,
                )
            ):
                raise TriBranchHookError("sample returned with an incomplete branch triplet")
            if state.integrated_steps != self.expected_steps:
                raise TriBranchHookError(
                    "sample did not execute the expected number of original UniPC steps: "
                    f"{state.integrated_steps} != {self.expected_steps}"
                )
            if len(self.trace.records) - records_before != self.expected_steps:
                raise TriBranchHookError("trace/integrator step counts differ")
            if state.action_momentum.update_count != self.expected_steps:
                raise TriBranchHookError("action APG momentum did not update once per step")
            if state.noop_momentum.update_count != self.expected_steps:
                raise TriBranchHookError("noop APG momentum did not update once per step")
            if self.diffusion.scheduler is not self.scheduler:
                raise TriBranchHookError("diffusion.sample replaced the pinned scheduler")
            self.trace.sample_calls += 1
            return result
        finally:
            self._active = None

    def _set_patch(self, owner: Any, name: str, value: Any) -> None:
        try:
            instance_dict = vars(owner)
        except TypeError as error:
            raise TriBranchHookError(f"{name} owner cannot be patched reversibly") from error
        had_instance = name in instance_dict
        previous = instance_dict.get(name)
        setattr(owner, name, value)
        self._patches.append((owner, name, had_instance, previous))

    def install(self) -> None:
        if self._installed:
            raise TriBranchHookError("tri-branch hook is already installed")

        def shared_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_shared_step(*args, **kwargs)

        def scheduler_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_scheduler_step(*args, **kwargs)

        def sample_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._wrapped_sample(*args, **kwargs)

        for wrapper in (shared_wrapper, scheduler_wrapper, sample_wrapper):
            setattr(wrapper, "_bernini_tri_branch_unipc", self)
        try:
            self._set_patch(self.diffusion, "shared_step", shared_wrapper)
            self._set_patch(self.scheduler, "step", scheduler_wrapper)
            self._set_patch(self.diffusion, "sample", sample_wrapper)
        except Exception:
            self.restore()
            raise
        self._installed = True

    def restore(self) -> None:
        errors: list[Exception] = []
        while self._patches:
            owner, name, had_instance, previous = self._patches.pop()
            try:
                if had_instance:
                    setattr(owner, name, previous)
                else:
                    delattr(owner, name)
            except Exception as error:
                # Attempt all three restorations before surfacing the failure.
                errors.append(error)
        self._installed = False
        self._active = None
        if errors:
            raise TriBranchHookError(
                f"failed to restore {len(errors)} tri-branch instance wrapper(s)"
            ) from errors[0]


@contextmanager
def tri_branch_unipc_hook(
    renderer_or_diffusion: Any,
    *,
    noop_prompt_embeds: Any,
    latent_shape: Sequence[int],
    clean_field_callback: CleanFieldCallback,
    bernini_commit: str,
    wan_diffusion_path: str | Path,
    noop_prompt_embeds_t2: Any = None,
    expected_steps: int = 40,
    expected_flow_shift: float = 5.0,
    projector: StepProjector = project_clean_fields,
) -> Iterator[TriBranchTrace]:
    """Install a reversible tri-branch hook around pinned Bernini sampling.

    The caller still invokes the official renderer/diffusion ``sample`` with
    ``guidance_mode='v2v_apg'``.  ``noop_prompt_embeds`` must have been encoded
    by the same Bernini T5 path as the action/negative embeddings.  The two
    Bernini revision must come from the caller's git audit.  The hook itself
    hashes the actual ``wan_diffusion.py`` path before installing anything.
    It restores all three patched instance attributes on normal exit and on
    every exception.
    """

    diffusion = resolve_diffusion_core(renderer_or_diffusion)
    bridge = _InstalledTriBranch(
        diffusion,
        noop_prompt_embeds=noop_prompt_embeds,
        noop_prompt_embeds_t2=noop_prompt_embeds_t2,
        latent_shape=latent_shape,
        clean_field_callback=clean_field_callback,
        expected_steps=expected_steps,
        expected_flow_shift=expected_flow_shift,
        projector=projector,
        bernini_commit=bernini_commit,
        wan_diffusion_path=wan_diffusion_path,
    )
    bridge.install()
    try:
        yield bridge.trace
    finally:
        bridge.restore()


__all__ = [
    "APGParameters",
    "CleanFieldStep",
    "PACK_PATCH_HEIGHT",
    "PACK_PATCH_WIDTH",
    "PINNED_BERNINI_COMMIT",
    "PINNED_WAN_DIFFUSION_SHA256",
    "PackedLatentLayout",
    "ProjectedVelocity",
    "RawTriBranchStep",
    "TriBranchHookError",
    "TriBranchStepRecord",
    "TriBranchTrace",
    "action_clean_passthrough",
    "noop_clean_passthrough",
    "pinned_raw_condition_clean",
    "project_clean_fields",
    "resolve_diffusion_core",
    "scaled_action_delta",
    "sampler_contract",
    "tri_branch_unipc_hook",
    "validate_runtime_source_identity",
]

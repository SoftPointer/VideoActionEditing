#!/usr/bin/env python3
"""Episodic generator-native trajectory controller (EGNTC) for Bernini-R.

EGNTC is a small, frozen-generator controller intended for few-shot motion
editing experiments.  At every *official* Bernini UniPC step it consumes the
action and semantic-noop guided clean predictions produced on the same noisy
state.  It does not query an alternative noise trajectory or introduce a
second integrator.

For ``D_i = X_action_i - X_noop_i`` the controller executes

``M_i = (1-kappa_i) D_i + kappa_i M_(i-1)``

``C_i = alpha_(i,t) M_i + rho_i (S - X_noop_i)``

``X_exec_i = X_noop_i + RMSClip_(B,T)(C_i; RMS_(B,T)(D_i))``.

The clip reduces over channel and spatial axes independently for every
``(batch, latent-phase)`` cell.  It therefore prevents the controller from
having more per-phase RMS authority than Bernini's native action/noop field.
If the action and noop tensors are exactly equal, the method returns the noop
tensor exactly and clears trajectory memory before advancing the step.

There are exactly 36 trainable scalars: six sigma knots times four temporal
DCT coefficients, six monotone memory parameters, and six monotone source
tether parameters.  Sigma interpolation uses the actual captured float32
values of Bernini's pinned 40-step UniPC schedule, never normalized step
indices.  Target videos may supervise these parameters during few-shot
training, but are deliberately absent from every runtime API in this module.

PyTorch is optional at import time so the inference contract remains
inspectable on orchestration hosts.  Constructing a parameter module or
executing tensor code requires PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping, Optional, Sequence

try:  # Keep contract inspection available on non-training hosts.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by lightweight hosts
    torch = None
    nn = None

# Keep the experiment archive self-contained.  These values are the exact
# float32 bytes captured from Bernini's released UniPC scheduler; they match
# ``inference_sigma_strata.py`` when that wider training module is installed.
_PINNED_TIMESTEPS: tuple[int, ...] = (
    999, 994, 989, 984, 978, 972, 965, 959, 952, 945,
    937, 929, 921, 912, 902, 893, 882, 871, 859, 847,
    833, 819, 803, 787, 769, 750, 729, 707, 682, 655,
    625, 593, 556, 516, 470, 418, 359, 291, 211, 117,
)
_PINNED_SIGMA_HEX: tuple[str, ...] = (
    "3f7fffef", "3f7eb1f9", "3f7d560b", "3f7beb53", "3f7a70da",
    "3f78e594", "3f77485b", "3f7597f0", "3f73d2f4", "3f71f7e6",
    "3f70051e", "3f6df8cb", "3f6bd0e9", "3f698b3c", "3f67254a",
    "3f649c50", "3f61ed37", "3f5f148a", "3f5c0e64", "3f58d661",
    "3f556787", "3f51bc2a", "3f4dcdd4", "3f499515", "3f45095d",
    "3f4020bc", "3f3acf9b", "3f35085f", "3f2ebaf8", "3f27d446",
    "3f203d59", "3f17da71", "3f0e89a7", "3f042120", "3ef0d923",
    "3ed6539a", "3eb80796", "3e9516ea", "3e58b351", "3df0f309",
)
_PINNED_SIGMAS: tuple[float, ...] = tuple(
    float(struct.unpack(">f", bytes.fromhex(value))[0])
    for value in _PINNED_SIGMA_HEX
)
_SCHEDULE_SHA256 = "3e5ad4473d133318026cc9e8f32399782bf06313691b58870c89d9c4c87c3d03"


class _SelfContainedSigmaStrata:
    NUM_INFERENCE_STEPS = 40
    PINNED_TIMESTEPS = _PINNED_TIMESTEPS
    PINNED_POSITIVE_SIGMA_FLOAT32_HEX = _PINNED_SIGMA_HEX
    PINNED_POSITIVE_SIGMAS = _PINNED_SIGMAS
    SCHEDULE_SHA256 = _SCHEDULE_SHA256


sigma_strata: Any = _SelfContainedSigmaStrata
try:
    try:
        from . import inference_sigma_strata as _shared_sigma_strata
    except ImportError:
        import inference_sigma_strata as _shared_sigma_strata
except ImportError:
    _shared_sigma_strata = None
if _shared_sigma_strata is not None:
    if (
        tuple(_shared_sigma_strata.PINNED_TIMESTEPS) != _PINNED_TIMESTEPS
        or tuple(_shared_sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX)
        != _PINNED_SIGMA_HEX
        or _shared_sigma_strata.SCHEDULE_SHA256 != _SCHEDULE_SHA256
    ):
        raise RuntimeError("shared Bernini sigma schedule differs from EGNTC pins")
    sigma_strata = _shared_sigma_strata


METHOD_NAME = "episodic-generator-native-trajectory-controller"
SCHEMA_VERSION = "bernini-egntc-controller-receipt-v1"
PARAMETER_SCHEMA_VERSION = "bernini-egntc-parameters-v1"

EXPECTED_RGB_FRAMES = 81
EXPECTED_LATENT_PHASES = 21
NUM_SIGMA_KNOTS = 6
NUM_PHASE_DCT_MODES = 4
TRAINABLE_DIMENSION = NUM_SIGMA_KNOTS * NUM_PHASE_DCT_MODES + 2 * NUM_SIGMA_KNOTS

SIGMA_KNOT_SCHEDULE_INDICES: tuple[int, ...] = (0, 8, 16, 24, 32, 39)
PINNED_SIGMA_KNOTS: tuple[float, ...] = tuple(
    sigma_strata.PINNED_POSITIVE_SIGMAS[index]
    for index in SIGMA_KNOT_SCHEDULE_INDICES
)
# A fixed non-reversal permutation avoids conflating the sigma-shuffle and
# denoising-reversal controls.  It is applied to raw knot parameters before
# the monotone kappa/rho transforms, so their bounds and monotonicity survive.
SIGMA_SHUFFLE_PERMUTATION: tuple[int, ...] = (2, 5, 0, 4, 1, 3)

MAX_KAPPA = 0.90
MAX_RHO = 0.50
INITIAL_ALPHA = 0.10

FORBIDDEN_INFERENCE_CONDITIONS: tuple[str, ...] = (
    "target_video",
    "paired_target",
    "support_video",
    "mask",
    "track",
    "swept_tube",
    "pose",
    "trajectory",
    "optical_flow",
    "flow",
    "first_frame_anchor",
    "edited_first_frame",
)


class EGNTCContractError(RuntimeError):
    """Raised before execution when an EGNTC invariant is violated."""


def _require_torch() -> Any:
    if torch is None:  # pragma: no cover - depends on host installation
        raise EGNTCContractError("PyTorch is required for EGNTC tensor execution")
    return torch


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _object_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _float32_hex(value: float) -> str:
    try:
        return struct.pack(">f", float(value)).hex()
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise EGNTCContractError("sigma must be representable as float32") from error


def _scalar(value: Any, *, label: str) -> float:
    try:
        candidate = value.detach() if hasattr(value, "detach") else value
        if hasattr(candidate, "numel") and int(candidate.numel()) != 1:
            raise EGNTCContractError(f"{label} must be scalar")
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        result = float(candidate)
    except EGNTCContractError:
        raise
    except Exception as error:
        raise EGNTCContractError(f"{label} must be a numeric scalar") from error
    if not math.isfinite(result):
        raise EGNTCContractError(f"{label} must be finite")
    return result


def controller_contract() -> dict[str, Any]:
    """Return the source+instruction-only, auditable runtime contract."""

    contract = {
        "method": METHOD_NAME,
        "status": "few-shot-trainable-frozen-generator-controller",
        "external_inference_conditions": ["source_video", "action_instruction"],
        "internal_fixed_controls": [
            "semantic_noop_instruction",
            "negative_prompt",
            "causal_controller_memory",
        ],
        "forbidden_inference_conditions": list(FORBIDDEN_INFERENCE_CONDITIONS),
        "target_role": "training_only_parameter_supervision_never_runtime_input",
        "generator_boundary": (
            "same_state_action_and_noop_guided_clean_fields_before_original_unipc_step"
        ),
        "integrator": "one_original_unipc_step_per_denoising_step",
        "tensor_layout": "B,C,T,H,W",
        "clip_geometry": {
            "rgb_frames": EXPECTED_RGB_FRAMES,
            "latent_phases": EXPECTED_LATENT_PHASES,
        },
        "parameterization": {
            "alpha_logits": [NUM_SIGMA_KNOTS, NUM_PHASE_DCT_MODES],
            "kappa_raw": [NUM_SIGMA_KNOTS],
            "rho_raw": [NUM_SIGMA_KNOTS],
            "trainable_dimension": TRAINABLE_DIMENSION,
            "phase_basis": "orthonormal_dct_ii_first_4_modes",
            "alpha_decode": "sigmoid_dct_synthesis",
            "alpha_bounds": [0.0, 1.0],
            "sigma_interpolation": "piecewise_linear_in_actual_pinned_float32_sigma",
            "kappa_bound": MAX_KAPPA,
            "rho_bound": MAX_RHO,
        },
        "trajectory_recurrence": "M_i=(1-kappa_i)*D_i+kappa_i*M_(i-1)",
        "clean_execution": (
            "X_noop+per_(B,T)_RMSClip(alpha*M+rho*(source-X_noop),RMS(D))"
        ),
        "native_authority_axes": "RMS_over_C_H_W_independently_for_each_B_T",
        "exact_action_noop_parity": "hard_noop_bypass_and_memory_clear",
        "field_gradient_policy": "source_action_noop_detached_controller_only",
        "schedule": {
            "num_inference_steps": sigma_strata.NUM_INFERENCE_STEPS,
            "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
            "sigma_knot_schedule_indices": list(SIGMA_KNOT_SCHEDULE_INDICES),
            "sigma_knot_float32_be_hex": [
                sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
                for index in SIGMA_KNOT_SCHEDULE_INDICES
            ],
        },
        "diagnostic_controls": [
            "phase_reverse",
            "sigma_shuffle",
            "kappa_off",
            "rho_off",
        ],
    }
    validate_runtime_contract(contract)
    return contract


def validate_runtime_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed if a serialized runtime contract broadens its inputs."""

    if not isinstance(contract, Mapping):
        raise EGNTCContractError("runtime contract must be a mapping")
    if contract.get("method") != METHOD_NAME or contract.get("status") != (
        "few-shot-trainable-frozen-generator-controller"
    ):
        raise EGNTCContractError("EGNTC runtime contract identity differs")
    if contract.get("external_inference_conditions") != [
        "source_video",
        "action_instruction",
    ]:
        raise EGNTCContractError("EGNTC inference accepts only source and instruction")
    forbidden = set(contract.get("forbidden_inference_conditions", ()))
    if not set(FORBIDDEN_INFERENCE_CONDITIONS).issubset(forbidden):
        raise EGNTCContractError("runtime contract omits a forbidden condition")
    parameterization = contract.get("parameterization")
    if not isinstance(parameterization, Mapping):
        raise EGNTCContractError("runtime contract lacks parameterization")
    if parameterization.get("trainable_dimension") != TRAINABLE_DIMENSION:
        raise EGNTCContractError("runtime contract changes the 36D parameterization")
    if (
        parameterization.get("alpha_logits")
        != [NUM_SIGMA_KNOTS, NUM_PHASE_DCT_MODES]
        or parameterization.get("kappa_raw") != [NUM_SIGMA_KNOTS]
        or parameterization.get("rho_raw") != [NUM_SIGMA_KNOTS]
        or parameterization.get("alpha_bounds") != [0.0, 1.0]
        or parameterization.get("kappa_bound") != MAX_KAPPA
        or parameterization.get("rho_bound") != MAX_RHO
    ):
        raise EGNTCContractError("runtime contract changes controller parameterization")
    schedule = contract.get("schedule")
    expected_sigma_hex = [
        sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
        for index in SIGMA_KNOT_SCHEDULE_INDICES
    ]
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("num_inference_steps") != sigma_strata.NUM_INFERENCE_STEPS
        or schedule.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256
        or schedule.get("sigma_knot_schedule_indices")
        != list(SIGMA_KNOT_SCHEDULE_INDICES)
        or schedule.get("sigma_knot_float32_be_hex") != expected_sigma_hex
    ):
        raise EGNTCContractError("runtime contract changes the pinned UniPC schedule")


@dataclass(frozen=True)
class EGNTCControls:
    """Diagnostic ablations; all flags default to the proposed method."""

    phase_reverse: bool = False
    sigma_shuffle: bool = False
    kappa_off: bool = False
    rho_off: bool = False

    def validate(self) -> "EGNTCControls":
        for name, value in asdict(self).items():
            if type(value) is not bool:
                raise EGNTCContractError(f"control {name} must be bool")
        return self

    def active(self) -> list[str]:
        self.validate()
        return [name for name, enabled in asdict(self).items() if enabled]


def normalize_controls(
    controls: Optional[EGNTCControls | str | Sequence[str]],
) -> EGNTCControls:
    """Normalize a control object or an explicit list of ablation names."""

    if controls is None:
        return EGNTCControls()
    if isinstance(controls, EGNTCControls):
        return controls.validate()
    names: Sequence[str]
    if isinstance(controls, str):
        names = (controls,)
    elif isinstance(controls, Sequence):
        names = controls
    else:
        raise EGNTCContractError("controls must be EGNTCControls or control names")
    allowed = set(EGNTCControls.__dataclass_fields__)
    if any(not isinstance(name, str) for name in names):
        raise EGNTCContractError("control names must be strings")
    unknown = set(names) - allowed
    if unknown:
        raise EGNTCContractError(f"unknown EGNTC controls: {sorted(unknown)!r}")
    if len(set(names)) != len(tuple(names)):
        raise EGNTCContractError("control names must be unique")
    return EGNTCControls(**{name: name in names for name in allowed}).validate()


@dataclass(frozen=True)
class PinnedStep:
    step_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str


def validate_pinned_step(*, step_index: int, timestep: Any, sigma: Any) -> PinnedStep:
    """Validate one query against the exact captured 40-step schedule."""

    if type(step_index) is not int or not 0 <= step_index < sigma_strata.NUM_INFERENCE_STEPS:
        raise EGNTCContractError("step_index must be an integer in [0,39]")
    observed_timestep = _scalar(timestep, label="timestep")
    expected_timestep = sigma_strata.PINNED_TIMESTEPS[step_index]
    if observed_timestep != float(expected_timestep):
        raise EGNTCContractError(
            f"timestep differs from pinned schedule at step {step_index}"
        )
    observed_sigma = _scalar(sigma, label="sigma")
    observed_hex = _float32_hex(observed_sigma)
    expected_hex = sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[step_index]
    if observed_hex != expected_hex:
        raise EGNTCContractError(
            f"sigma differs from pinned schedule at step {step_index}"
        )
    return PinnedStep(
        step_index=step_index,
        timestep=expected_timestep,
        sigma=sigma_strata.PINNED_POSITIVE_SIGMAS[step_index],
        sigma_float32_be_hex=expected_hex,
    )


@dataclass(frozen=True)
class SigmaInterpolation:
    upper_knot: int
    lower_knot: int
    lower_weight: float


def sigma_interpolation(
    *, step_index: int, timestep: Any, sigma: Any
) -> SigmaInterpolation:
    """Return interpolation coordinates computed in real sigma space."""

    step = validate_pinned_step(
        step_index=step_index, timestep=timestep, sigma=sigma
    )
    for position, schedule_index in enumerate(SIGMA_KNOT_SCHEDULE_INDICES):
        if step_index == schedule_index:
            return SigmaInterpolation(position, position, 0.0)
        if position + 1 == NUM_SIGMA_KNOTS:
            break
        next_index = SIGMA_KNOT_SCHEDULE_INDICES[position + 1]
        if schedule_index < step_index < next_index:
            upper_sigma = PINNED_SIGMA_KNOTS[position]
            lower_sigma = PINNED_SIGMA_KNOTS[position + 1]
            weight = (upper_sigma - step.sigma) / (upper_sigma - lower_sigma)
            if not 0.0 < weight < 1.0:
                raise EGNTCContractError("invalid real-sigma interpolation weight")
            return SigmaInterpolation(position, position + 1, weight)
    raise EGNTCContractError("step is not bracketed by pinned sigma knots")


def _validate_knot_tensor(value: Any, *, label: str) -> None:
    library = _require_torch()
    if not isinstance(value, library.Tensor) or not value.is_floating_point():
        raise EGNTCContractError(f"{label} must be a floating torch tensor")
    if value.ndim < 1 or int(value.shape[0]) != NUM_SIGMA_KNOTS:
        raise EGNTCContractError(f"{label} first dimension must be six sigma knots")
    if not bool(library.isfinite(value).all().item()):
        raise EGNTCContractError(f"{label} must be finite")


def interpolate_sigma_knots(
    knot_values: Any,
    *,
    step_index: int,
    timestep: Any,
    sigma: Any,
) -> Any:
    """Differentiably interpolate knot values using actual pinned sigmas."""

    _validate_knot_tensor(knot_values, label="knot_values")
    interpolation = sigma_interpolation(
        step_index=step_index, timestep=timestep, sigma=sigma
    )
    upper = knot_values[interpolation.upper_knot]
    if interpolation.upper_knot == interpolation.lower_knot:
        return upper
    lower = knot_values[interpolation.lower_knot]
    weight = interpolation.lower_weight
    return upper * (1.0 - weight) + lower * weight


def phase_dct_basis(
    phase_count: int,
    *,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build the first four orthonormal DCT-II phase modes."""

    library = _require_torch()
    if type(phase_count) is not int or phase_count != EXPECTED_LATENT_PHASES:
        raise EGNTCContractError(
            f"phase_count must equal the 81-frame latent count {EXPECTED_LATENT_PHASES}"
        )
    if dtype is None:
        dtype = library.float32
    phases = library.arange(phase_count, device=device, dtype=dtype)[:, None]
    modes = library.arange(NUM_PHASE_DCT_MODES, device=device, dtype=dtype)[None, :]
    basis = library.cos(math.pi * (phases + 0.5) * modes / float(phase_count))
    basis[:, 0] *= 1.0 / math.sqrt(2.0)
    return basis * math.sqrt(2.0 / float(phase_count))


@dataclass(frozen=True)
class StepCoefficients:
    alpha: Any
    kappa: Any
    rho: Any
    interpolation: SigmaInterpolation

    def audit(self) -> dict[str, Any]:
        return {
            "alpha_min": float(self.alpha.detach().float().min().cpu().item()),
            "alpha_max": float(self.alpha.detach().float().max().cpu().item()),
            "kappa": float(self.kappa.detach().float().cpu().item()),
            "rho": float(self.rho.detach().float().cpu().item()),
            "interpolation": asdict(self.interpolation),
        }


_ModuleBase = nn.Module if nn is not None else object


class EGNTCParameters(_ModuleBase):
    """Exactly 36 trainable scalars with bounded trajectory schedules."""

    def __init__(self) -> None:
        library = _require_torch()
        super().__init__()
        alpha = library.zeros(
            (NUM_SIGMA_KNOTS, NUM_PHASE_DCT_MODES), dtype=library.float32
        )
        # In an orthonormal DCT-II basis the DC column is 1/sqrt(T).  This
        # initialization therefore decodes to exactly INITIAL_ALPHA at every
        # phase (up to float32 rounding), while higher modes remain neutral.
        alpha[:, 0] = (
            math.log(INITIAL_ALPHA / (1.0 - INITIAL_ALPHA))
            * math.sqrt(EXPECTED_LATENT_PHASES)
        )
        self.alpha_logits = nn.Parameter(alpha)
        self.kappa_raw = nn.Parameter(
            library.full((NUM_SIGMA_KNOTS,), -3.0, dtype=library.float32)
        )
        self.rho_raw = nn.Parameter(
            library.full((NUM_SIGMA_KNOTS,), -5.0, dtype=library.float32)
        )
        self.validate_parameters()

    def validate_parameters(self) -> None:
        library = _require_torch()
        expected = {
            "alpha_logits": (NUM_SIGMA_KNOTS, NUM_PHASE_DCT_MODES),
            "kappa_raw": (NUM_SIGMA_KNOTS,),
            "rho_raw": (NUM_SIGMA_KNOTS,),
        }
        for name, shape in expected.items():
            value = getattr(self, name, None)
            if not isinstance(value, library.Tensor) or not value.is_floating_point():
                raise EGNTCContractError(f"{name} must be a floating torch tensor")
            if tuple(int(item) for item in value.shape) != shape:
                raise EGNTCContractError(f"{name} must have shape {shape!r}")
            if not bool(library.isfinite(value).all().item()):
                raise EGNTCContractError(f"{name} must be finite")
        dimension = sum(int(parameter.numel()) for parameter in self.parameters())
        if dimension != TRAINABLE_DIMENSION:
            raise EGNTCContractError("EGNTC must contain exactly 36 trainable scalars")

    @staticmethod
    def _monotone_knots(raw: Any, *, upper_bound: float) -> Any:
        """Decode finite raw values into nondecreasing values below a bound."""

        library = _require_torch()
        increments = library.sigmoid(raw)
        cumulative = library.cumsum(increments, dim=0)
        return float(upper_bound) * (1.0 - library.exp(-cumulative))

    def decoded_knots(
        self, controls: Optional[EGNTCControls | str | Sequence[str]] = None
    ) -> tuple[Any, Any, Any]:
        self.validate_parameters()
        cfg = normalize_controls(controls)
        alpha_raw = self.alpha_logits
        kappa_raw = self.kappa_raw
        rho_raw = self.rho_raw
        if cfg.sigma_shuffle:
            permutation = torch.tensor(
                SIGMA_SHUFFLE_PERMUTATION,
                dtype=torch.int64,
                device=alpha_raw.device,
            )
            alpha_raw = alpha_raw.index_select(0, permutation)
            kappa_raw = kappa_raw.index_select(0, permutation)
            rho_raw = rho_raw.index_select(0, permutation)
        kappa = self._monotone_knots(kappa_raw, upper_bound=MAX_KAPPA)
        rho = self._monotone_knots(rho_raw, upper_bound=MAX_RHO)
        if cfg.kappa_off:
            kappa = kappa * 0.0
        if cfg.rho_off:
            rho = rho * 0.0
        if not bool(torch.isfinite(kappa).all().item()) or not bool(
            torch.isfinite(rho).all().item()
        ):
            raise EGNTCContractError("decoded kappa/rho schedules must be finite")
        return alpha_raw, kappa, rho

    def coefficients(
        self,
        *,
        step_index: int,
        timestep: Any,
        sigma: Any,
        phase_count: int = EXPECTED_LATENT_PHASES,
        controls: Optional[EGNTCControls | str | Sequence[str]] = None,
    ) -> StepCoefficients:
        """Decode one real-sigma and four-mode temporal controller query."""

        cfg = normalize_controls(controls)
        alpha_knots, kappa_knots, rho_knots = self.decoded_knots(cfg)
        interpolation = sigma_interpolation(
            step_index=step_index, timestep=timestep, sigma=sigma
        )
        alpha_coefficients = interpolate_sigma_knots(
            alpha_knots, step_index=step_index, timestep=timestep, sigma=sigma
        )
        kappa = interpolate_sigma_knots(
            kappa_knots, step_index=step_index, timestep=timestep, sigma=sigma
        )
        rho = interpolate_sigma_knots(
            rho_knots, step_index=step_index, timestep=timestep, sigma=sigma
        )
        basis = phase_dct_basis(
            phase_count,
            device=alpha_coefficients.device,
            dtype=alpha_coefficients.dtype,
        )
        alpha = torch.sigmoid(basis @ alpha_coefficients)
        if cfg.phase_reverse:
            alpha = alpha.flip(0)
        for label, value in (("alpha", alpha), ("kappa", kappa), ("rho", rho)):
            if not bool(torch.isfinite(value).all().item()):
                raise EGNTCContractError(f"decoded {label} must be finite")
        return StepCoefficients(
            alpha=alpha,
            kappa=kappa,
            rho=rho,
            interpolation=interpolation,
        )

    def parameter_vector(self, *, detach: bool = False) -> Any:
        """Return the canonical alpha/kappa/rho flattened ordering."""

        self.validate_parameters()
        vector = torch.cat(
            (
                self.alpha_logits.reshape(-1),
                self.kappa_raw.reshape(-1),
                self.rho_raw.reshape(-1),
            )
        )
        return vector.detach().clone() if detach else vector

    # Runner/inference compatibility names.  Both preserve the one canonical
    # alpha-then-kappa-then-rho ordering documented above.
    def flat_tensor(self, *, detach: bool = False) -> Any:
        return self.parameter_vector(detach=detach)

    @classmethod
    def from_flat_tensor(cls, vector: Any) -> "EGNTCParameters":
        result = cls()
        result.load_parameter_vector_(vector)
        return result

    def load_parameter_vector_(self, vector: Any) -> "EGNTCParameters":
        """Load an exact 36-vector, useful for support prototypes."""

        library = _require_torch()
        if not isinstance(vector, library.Tensor) or not vector.is_floating_point():
            raise EGNTCContractError("parameter vector must be a floating torch tensor")
        if tuple(int(item) for item in vector.shape) != (TRAINABLE_DIMENSION,):
            raise EGNTCContractError("parameter vector must contain exactly 36 scalars")
        if not bool(library.isfinite(vector).all().item()):
            raise EGNTCContractError("parameter vector must be finite")
        candidate = vector.to(device=self.alpha_logits.device, dtype=self.alpha_logits.dtype)
        with library.no_grad():
            self.alpha_logits.copy_(candidate[:24].reshape(6, 4))
            self.kappa_raw.copy_(candidate[24:30])
            self.rho_raw.copy_(candidate[30:36])
        self.validate_parameters()
        return self

    def receipt(self) -> dict[str, Any]:
        """Return tensor-free parameter and decoded-bound diagnostics."""

        self.validate_parameters()
        _, kappa, rho = self.decoded_knots()
        payload = {
            "schema_version": PARAMETER_SCHEMA_VERSION,
            "method": METHOD_NAME,
            "trainable_dimension": TRAINABLE_DIMENSION,
            "parameter_shapes": {
                "alpha_logits": [6, 4],
                "kappa_raw": [6],
                "rho_raw": [6],
            },
            "parameter_vector_sha256": hashlib.sha256(
                bytes(
                    self.parameter_vector(detach=True)
                    .float()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .tolist()
                )
            ).hexdigest(),
            "decoded_kappa_knots": [
                float(value) for value in kappa.detach().float().cpu().tolist()
            ],
            "decoded_rho_knots": [
                float(value) for value in rho.detach().float().cpu().tolist()
            ],
            "kappa_monotone_nondecreasing": bool(
                torch.all(kappa[1:] >= kappa[:-1]).item()
            ),
            "rho_monotone_nondecreasing": bool(
                torch.all(rho[1:] >= rho[:-1]).item()
            ),
            "kappa_strict_upper_bound": MAX_KAPPA,
            "rho_strict_upper_bound": MAX_RHO,
            "schedule_sha256": sigma_strata.SCHEDULE_SHA256,
        }
        payload["receipt_digest"] = _object_sha256(payload)
        return payload


@dataclass(frozen=True)
class ControllerStepRecord:
    step_index: int
    timestep: int
    sigma: float
    sigma_float32_be_hex: str
    upper_knot: int
    lower_knot: int
    lower_weight: float
    kappa: float
    rho: float
    action_noop_exact_parity: bool
    native_delta_rms_max: float
    proposal_correction_rms_max: float
    executed_correction_rms_max: float
    trust_region_satisfied: bool


def _validate_floating_tensor(value: Any, *, label: str) -> None:
    library = _require_torch()
    if not isinstance(value, library.Tensor) or not value.is_floating_point():
        raise EGNTCContractError(f"{label} must be a floating torch tensor")
    if value.ndim != 5:
        raise EGNTCContractError(f"{label} must have layout [B,C,T,H,W]")
    if int(value.shape[2]) != EXPECTED_LATENT_PHASES:
        raise EGNTCContractError(
            f"{label} must contain exactly {EXPECTED_LATENT_PHASES} latent phases"
        )
    if value.device.type == "meta":
        raise EGNTCContractError(f"{label} cannot be a meta tensor")
    if not bool(library.isfinite(value).all().item()):
        raise EGNTCContractError(f"{label} must be finite")


def _per_batch_phase_rms(value: Any) -> Any:
    """Return shape ``[B,1,T,1,1]`` RMS over C/H/W in fp32 or fp64."""

    work = value if value.dtype == torch.float64 else value.float()
    return work.square().mean(dim=(1, 3, 4), keepdim=True).sqrt()


def per_batch_phase_rms_clip(proposal: Any, native_delta: Any) -> tuple[Any, Any, Any]:
    """Clip proposal RMS to native-delta RMS independently per ``(B,T)``."""

    _validate_floating_tensor(proposal, label="proposal")
    _validate_floating_tensor(native_delta, label="native_delta")
    if proposal.shape != native_delta.shape or proposal.device != native_delta.device:
        raise EGNTCContractError("proposal and native_delta must share shape and device")
    proposal_rms = _per_batch_phase_rms(proposal)
    native_rms = _per_batch_phase_rms(native_delta)
    one = torch.ones_like(proposal_rms)
    # Division occurs only where proposal_rms is strictly positive and larger
    # than native_rms, including native_rms==0, so no epsilon weakens the cap.
    active = proposal_rms > native_rms
    safe_denominator = proposal_rms.clamp_min(torch.finfo(proposal_rms.dtype).tiny)
    scale = torch.where(active, native_rms / safe_denominator, one)
    clipped = proposal * scale.to(dtype=proposal.dtype)
    clipped_rms = _per_batch_phase_rms(clipped)
    tolerance = 8.0 * torch.finfo(clipped_rms.dtype).eps * torch.maximum(
        native_rms, torch.ones_like(native_rms)
    )
    if not bool(torch.all(clipped_rms <= native_rms + tolerance).item()):
        raise EGNTCContractError("per-(B,T) native-field trust region failed")
    return clipped, native_rms, proposal_rms


class EGNTCCallback:
    """Stateful callback for :func:`tri_branch_unipc.tri_branch_unipc_hook`."""

    def __init__(
        self,
        parameters: EGNTCParameters,
        source_clean: Any,
        *,
        controls: Optional[EGNTCControls | str | Sequence[str]] = None,
    ) -> None:
        if not isinstance(parameters, EGNTCParameters):
            raise EGNTCContractError("parameters must be EGNTCParameters")
        parameters.validate_parameters()
        _validate_floating_tensor(source_clean, label="source_clean")
        self.parameters = parameters
        # The source is a condition, never an optimization path.  Clone it so
        # external in-place mutation cannot silently change a rollout.
        self.source_clean = source_clean.detach().clone()
        self.controls = normalize_controls(controls)
        self._memory: Any = None
        self._expected_step = 0
        self._records: list[ControllerStepRecord] = []
        self._reset_count = 0

    @property
    def expected_step(self) -> int:
        return self._expected_step

    @property
    def memory(self) -> Any:
        return self._memory

    @property
    def records(self) -> tuple[ControllerStepRecord, ...]:
        """Expose immutable tensor-free step diagnostics to experiment runners."""

        return tuple(self._records)

    @property
    def trace(self) -> tuple[ControllerStepRecord, ...]:
        return self.records

    def reset(self) -> None:
        """Release recurrence/graph state and require step zero next."""

        self._memory = None
        self._expected_step = 0
        self._records.clear()
        self._reset_count += 1

    def _validate_fields(self, action_clean: Any, noop_clean: Any) -> tuple[Any, Any]:
        _validate_floating_tensor(action_clean, label="action_clean")
        _validate_floating_tensor(noop_clean, label="noop_clean")
        if action_clean.shape != noop_clean.shape:
            raise EGNTCContractError("action_clean and noop_clean shapes differ")
        if action_clean.device != noop_clean.device or action_clean.dtype != noop_clean.dtype:
            raise EGNTCContractError("action_clean and noop_clean dtype/device differ")
        if self.source_clean.shape != noop_clean.shape:
            raise EGNTCContractError("source_clean and clean-field shapes differ")
        if self.source_clean.device != noop_clean.device:
            raise EGNTCContractError("source_clean and clean fields must share device")
        # Frozen generator/source contract: gradients terminate at the fields.
        return action_clean.detach(), noop_clean.detach()

    def apply_fields(
        self,
        *,
        action_clean: Any,
        noop_clean: Any,
        step_index: int,
        timestep: Any,
        sigma: Any,
    ) -> Any:
        """Execute one ordered controller step on same-state clean fields."""

        if self._expected_step >= sigma_strata.NUM_INFERENCE_STEPS:
            raise EGNTCContractError("40-step rollout is complete; call reset()")
        if step_index != self._expected_step:
            raise EGNTCContractError(
                f"expected denoising step {self._expected_step}, got {step_index}"
            )
        pinned = validate_pinned_step(
            step_index=step_index, timestep=timestep, sigma=sigma
        )
        action, noop = self._validate_fields(action_clean, noop_clean)
        coefficient = self.parameters.coefficients(
            step_index=step_index,
            timestep=timestep,
            sigma=sigma,
            phase_count=int(noop.shape[2]),
            controls=self.controls,
        )
        delta = action - noop

        exact_parity = bool(torch.equal(action, noop))
        if exact_parity:
            # This bypass precedes both recurrence and source tether.  A prior
            # nonzero memory therefore cannot leak through a semantic no-op.
            self._memory = torch.zeros_like(delta)
            executed = noop
            native_rms = _per_batch_phase_rms(delta)
            proposal_rms = torch.zeros_like(native_rms)
            executed_rms = torch.zeros_like(native_rms)
        else:
            if self._memory is None:
                previous = torch.zeros_like(delta)
            else:
                if self._memory.shape != delta.shape or self._memory.device != delta.device:
                    raise EGNTCContractError("trajectory memory shape/device drifted")
                if not bool(torch.isfinite(self._memory).all().item()):
                    raise EGNTCContractError("trajectory memory is non-finite")
                previous = self._memory
            work_dtype = torch.float64 if delta.dtype == torch.float64 else torch.float32
            delta_work = delta.to(dtype=work_dtype)
            previous_work = previous.to(dtype=work_dtype)
            kappa = coefficient.kappa.to(device=delta.device, dtype=work_dtype)
            rho = coefficient.rho.to(device=delta.device, dtype=work_dtype)
            alpha = coefficient.alpha.to(device=delta.device, dtype=work_dtype)
            memory = (1.0 - kappa) * delta_work + kappa * previous_work
            source = self.source_clean.to(dtype=work_dtype)
            noop_work = noop.to(dtype=work_dtype)
            proposal = (
                alpha.reshape(1, 1, EXPECTED_LATENT_PHASES, 1, 1) * memory
                + rho * (source - noop_work)
            )
            clipped, native_rms, proposal_rms = per_batch_phase_rms_clip(
                proposal, delta_work
            )
            executed = noop_work + clipped
            executed_rms = _per_batch_phase_rms(executed - noop_work)
            if not bool(torch.isfinite(executed).all().item()):
                raise EGNTCContractError("executed clean field is non-finite")
            self._memory = memory

        native_max = float(native_rms.detach().max().cpu().item())
        proposal_max = float(proposal_rms.detach().max().cpu().item())
        executed_max = float(executed_rms.detach().max().cpu().item())
        tolerance = 8.0 * torch.finfo(executed_rms.dtype).eps * max(native_max, 1.0)
        trust = executed_max <= native_max + tolerance
        if not all(math.isfinite(value) for value in (native_max, proposal_max, executed_max)):
            raise EGNTCContractError("controller RMS diagnostics are non-finite")
        if not trust:
            raise EGNTCContractError("controller exceeded native action/noop authority")

        audit = coefficient.audit()
        interpolation = coefficient.interpolation
        self._records.append(
            ControllerStepRecord(
                step_index=step_index,
                timestep=pinned.timestep,
                sigma=pinned.sigma,
                sigma_float32_be_hex=pinned.sigma_float32_be_hex,
                upper_knot=interpolation.upper_knot,
                lower_knot=interpolation.lower_knot,
                lower_weight=interpolation.lower_weight,
                kappa=audit["kappa"],
                rho=audit["rho"],
                action_noop_exact_parity=exact_parity,
                native_delta_rms_max=native_max,
                proposal_correction_rms_max=proposal_max,
                executed_correction_rms_max=executed_max,
                trust_region_satisfied=trust,
            )
        )
        self._expected_step += 1
        return executed

    def __call__(self, clean_step: Any) -> Any:
        """Consume a ``tri_branch_unipc.CleanFieldStep`` by structural API."""

        required = (
            "step_index",
            "timestep",
            "sigma",
            "action_guided_clean",
            "noop_guided_clean",
        )
        if clean_step is None or any(not hasattr(clean_step, name) for name in required):
            raise EGNTCContractError("clean_step lacks tri-branch guided clean fields")
        executed = self.apply_fields(
            action_clean=clean_step.action_guided_clean,
            noop_clean=clean_step.noop_guided_clean,
            step_index=clean_step.step_index,
            timestep=clean_step.timestep,
            sigma=clean_step.sigma,
        )
        if torch.equal(
            clean_step.action_guided_clean, clean_step.noop_guided_clean
        ):
            # tri_branch_unipc recognizes this exact object identity and then
            # reuses Bernini's official model_output without a lossy
            # clean->velocity round trip.  apply_fields above has already
            # recorded the bypass, cleared memory and advanced schedule state.
            return clean_step.action_guided_clean
        return executed

    def receipt(self) -> dict[str, Any]:
        """Return and self-validate a deterministic tensor-free rollout receipt."""

        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD_NAME,
            "runtime_contract": controller_contract(),
            "parameters": self.parameters.receipt(),
            "controls": asdict(self.controls),
            "active_controls": self.controls.active(),
            "state": {
                "expected_next_step": self._expected_step,
                "completed": self._expected_step == sigma_strata.NUM_INFERENCE_STEPS,
                "step_count": len(self._records),
                "memory_present": self._memory is not None,
                "reset_count": self._reset_count,
            },
            "steps": [asdict(record) for record in self._records],
        }
        payload["receipt_digest"] = _object_sha256(payload)
        validate_controller_receipt(payload, require_complete=False)
        return payload

    def audit_receipt(self) -> dict[str, Any]:
        return self.receipt()


# Descriptive public name plus the compact runner-facing spelling.
GeneratorNativeTrajectoryController = EGNTCCallback


def _receipt_number(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EGNTCContractError(f"controller receipt {label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EGNTCContractError(f"controller receipt {label} is non-finite")
    if nonnegative and numeric < 0.0:
        raise EGNTCContractError(f"controller receipt {label} must be non-negative")
    return numeric


def _validate_parameter_receipt(parameters: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    """Validate the nested 36D state receipt without requiring PyTorch."""

    if not isinstance(parameters, Mapping):
        raise EGNTCContractError("controller receipt lacks parameter receipt")
    candidate = dict(parameters)
    digest = candidate.pop("receipt_digest", None)
    if not isinstance(digest, str) or digest != _object_sha256(candidate):
        raise EGNTCContractError("controller parameter receipt digest differs")
    if (
        candidate.get("schema_version") != PARAMETER_SCHEMA_VERSION
        or candidate.get("method") != METHOD_NAME
        or candidate.get("trainable_dimension") != TRAINABLE_DIMENSION
        or candidate.get("parameter_shapes")
        != {
            "alpha_logits": [NUM_SIGMA_KNOTS, NUM_PHASE_DCT_MODES],
            "kappa_raw": [NUM_SIGMA_KNOTS],
            "rho_raw": [NUM_SIGMA_KNOTS],
        }
        or candidate.get("schedule_sha256") != sigma_strata.SCHEDULE_SHA256
    ):
        raise EGNTCContractError("controller parameter receipt contract differs")
    vector_digest = candidate.get("parameter_vector_sha256")
    if (
        not isinstance(vector_digest, str)
        or len(vector_digest) != 64
        or any(character not in "0123456789abcdef" for character in vector_digest)
    ):
        raise EGNTCContractError("controller parameter vector digest is invalid")

    decoded: list[list[float]] = []
    for name, bound in (
        ("decoded_kappa_knots", MAX_KAPPA),
        ("decoded_rho_knots", MAX_RHO),
    ):
        values = candidate.get(name)
        if not isinstance(values, list) or len(values) != NUM_SIGMA_KNOTS:
            raise EGNTCContractError(f"controller parameter receipt {name} differs")
        numeric = [
            _receipt_number(value, label=f"parameters.{name}", nonnegative=True)
            for value in values
        ]
        if any(value >= bound for value in numeric):
            raise EGNTCContractError(f"controller parameter receipt {name} is out of bounds")
        if any(right < left for left, right in zip(numeric, numeric[1:])):
            raise EGNTCContractError(f"controller parameter receipt {name} is not monotone")
        decoded.append(numeric)
    kappa, rho = decoded
    if (
        candidate.get("kappa_monotone_nondecreasing") is not True
        or candidate.get("rho_monotone_nondecreasing") is not True
        or candidate.get("kappa_strict_upper_bound") != MAX_KAPPA
        or candidate.get("rho_strict_upper_bound") != MAX_RHO
    ):
        raise EGNTCContractError("controller parameter decoded-bound receipt differs")
    return kappa, rho


def _interpolate_receipt_knots(values: Sequence[float], interpolation: SigmaInterpolation) -> float:
    upper = float(values[interpolation.upper_knot])
    if interpolation.upper_knot == interpolation.lower_knot:
        return upper
    lower = float(values[interpolation.lower_knot])
    return upper * (1.0 - interpolation.lower_weight) + lower * interpolation.lower_weight


def validate_controller_receipt(
    receipt: Mapping[str, Any], *, require_complete: bool = False
) -> None:
    """Validate digest, schedule order, bounds, finite values, and trust flags."""

    if not isinstance(receipt, Mapping):
        raise EGNTCContractError("controller receipt must be a mapping")
    candidate = dict(receipt)
    digest = candidate.pop("receipt_digest", None)
    if not isinstance(digest, str) or digest != _object_sha256(candidate):
        raise EGNTCContractError("controller receipt digest differs")
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise EGNTCContractError("controller receipt schema differs")
    if candidate.get("method") != METHOD_NAME:
        raise EGNTCContractError("controller receipt method differs")
    validate_runtime_contract(candidate.get("runtime_contract", {}))
    parameters = candidate.get("parameters")
    decoded_kappa, decoded_rho = _validate_parameter_receipt(parameters)
    controls = candidate.get("controls")
    if not isinstance(controls, Mapping):
        raise EGNTCContractError("controller receipt lacks controls")
    try:
        control_object = EGNTCControls(**dict(controls)).validate()
    except TypeError as error:
        raise EGNTCContractError("controller receipt controls differ") from error
    active_controls = candidate.get("active_controls")
    if active_controls != control_object.active():
        raise EGNTCContractError("controller receipt active controls differ")
    state = candidate.get("state")
    steps = candidate.get("steps")
    if not isinstance(state, Mapping) or not isinstance(steps, list):
        raise EGNTCContractError("controller receipt lacks state/steps")
    if len(steps) > sigma_strata.NUM_INFERENCE_STEPS:
        raise EGNTCContractError("controller receipt contains more than 40 steps")
    if (
        type(state.get("step_count")) is not int
        or type(state.get("expected_next_step")) is not int
        or state.get("step_count") != len(steps)
        or state.get("expected_next_step") != len(steps)
    ):
        raise EGNTCContractError("controller receipt step counters differ")
    complete = len(steps) == sigma_strata.NUM_INFERENCE_STEPS
    if type(state.get("completed")) is not bool or state.get("completed") is not complete:
        raise EGNTCContractError("controller receipt completion flag differs")
    expected_memory_present = len(steps) > 0
    if (
        type(state.get("memory_present")) is not bool
        or state.get("memory_present") is not expected_memory_present
    ):
        raise EGNTCContractError("controller receipt memory flag differs")
    if type(state.get("reset_count")) is not int or state.get("reset_count") < 0:
        raise EGNTCContractError("controller receipt reset count differs")
    if require_complete and not complete:
        raise EGNTCContractError("controller receipt is not a complete 40-step rollout")
    observed_kappa: list[float] = []
    observed_rho: list[float] = []
    for index, record in enumerate(steps):
        if not isinstance(record, Mapping) or record.get("step_index") != index:
            raise EGNTCContractError("controller receipt steps are out of order")
        if (
            type(record.get("timestep")) is not int
            or record.get("timestep") != sigma_strata.PINNED_TIMESTEPS[index]
        ):
            raise EGNTCContractError("controller receipt timestep differs")
        sigma = _receipt_number(record.get("sigma"), label="sigma", nonnegative=True)
        if (
            sigma != sigma_strata.PINNED_POSITIVE_SIGMAS[index]
            or _float32_hex(sigma)
            != sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
        ):
            raise EGNTCContractError("controller receipt numeric sigma differs")
        if record.get("sigma_float32_be_hex") != (
            sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[index]
        ):
            raise EGNTCContractError("controller receipt sigma differs")
        interpolation = sigma_interpolation(
            step_index=index,
            timestep=record["timestep"],
            sigma=sigma,
        )
        if (
            type(record.get("upper_knot")) is not int
            or type(record.get("lower_knot")) is not int
            or record.get("upper_knot") != interpolation.upper_knot
            or record.get("lower_knot") != interpolation.lower_knot
        ):
            raise EGNTCContractError("controller receipt interpolation knots differ")
        lower_weight = _receipt_number(record.get("lower_weight"), label="lower_weight")
        if lower_weight != interpolation.lower_weight:
            raise EGNTCContractError("controller receipt interpolation weight differs")
        kappa = _receipt_number(record.get("kappa"), label="kappa", nonnegative=True)
        rho = _receipt_number(record.get("rho"), label="rho", nonnegative=True)
        native = _receipt_number(
            record.get("native_delta_rms_max"),
            label="native_delta_rms_max",
            nonnegative=True,
        )
        proposal = _receipt_number(
            record.get("proposal_correction_rms_max"),
            label="proposal_correction_rms_max",
            nonnegative=True,
        )
        executed = _receipt_number(
            record.get("executed_correction_rms_max"),
            label="executed_correction_rms_max",
            nonnegative=True,
        )
        if not 0.0 <= kappa < MAX_KAPPA:
            raise EGNTCContractError("controller receipt kappa is out of bounds")
        if not 0.0 <= rho < MAX_RHO:
            raise EGNTCContractError("controller receipt rho is out of bounds")
        if type(record.get("action_noop_exact_parity")) is not bool:
            raise EGNTCContractError("controller receipt parity flag must be bool")
        if record.get("trust_region_satisfied") is not True:
            raise EGNTCContractError("controller receipt contains a trust-region failure")
        tolerance = 8.0 * (2.0**-23) * max(native, 1.0)
        if executed > native + tolerance:
            raise EGNTCContractError("controller receipt RMS values violate trust region")
        if record["action_noop_exact_parity"] and any(
            value != 0.0 for value in (native, proposal, executed)
        ):
            raise EGNTCContractError("controller receipt parity RMS values differ")
        observed_kappa.append(kappa)
        observed_rho.append(rho)

        if not control_object.sigma_shuffle:
            expected_kappa = 0.0 if control_object.kappa_off else _interpolate_receipt_knots(
                decoded_kappa, interpolation
            )
            expected_rho = 0.0 if control_object.rho_off else _interpolate_receipt_knots(
                decoded_rho, interpolation
            )
            if not math.isclose(kappa, expected_kappa, rel_tol=0.0, abs_tol=1.0e-7):
                raise EGNTCContractError("controller receipt kappa schedule differs")
            if not math.isclose(rho, expected_rho, rel_tol=0.0, abs_tol=1.0e-7):
                raise EGNTCContractError("controller receipt rho schedule differs")

    if any(right < left for left, right in zip(observed_kappa, observed_kappa[1:])):
        raise EGNTCContractError("controller receipt kappa trace is not monotone")
    if any(right < left for left, right in zip(observed_rho, observed_rho[1:])):
        raise EGNTCContractError("controller receipt rho trace is not monotone")


if TRAINABLE_DIMENSION != 36:
    raise RuntimeError("EGNTC trainable dimension changed")
if len(PINNED_SIGMA_KNOTS) != NUM_SIGMA_KNOTS or not all(
    left > right for left, right in zip(PINNED_SIGMA_KNOTS, PINNED_SIGMA_KNOTS[1:])
):
    raise RuntimeError("EGNTC sigma knots must follow the pinned descending schedule")

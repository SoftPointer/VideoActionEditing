"""Factorized source/action initial noise for exact-81 Bernini experiments.

The action carrier is a frozen Bernini T2V proposal generated from the target
caption.  Following the narrow mechanism tested by Phi-Noise, this module
copies only low *spatial*-frequency phase from that proposal into an otherwise
native Gaussian realization and explicitly balances spectral energy.  It then
optionally applies the existing order-invariant source spectral bridge, which
rotates only the temporal-DC subspace toward all-frame source set statistics.

The intended factorization is therefore::

    T2V action proposal -> low spatial phase / temporal change
    source video        -> order-invariant temporal-DC carrier
    native Gaussian     -> all remaining stochastic degrees of freedom

No target edit, mask, flow, pose, track, or trajectory is accepted.  Energy
preservation is audited, but it is *not* described as proof that the modified
noise remains Gaussian.  At ``spatial_radius == 0`` and ``source_rho == 0``
the exact input Gaussian object is returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import source_spectral_bridge as source_bridge


EXACT_VIDEO_FRAMES = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
SCHEMA_VERSION = "bernini-self-imagined-source-action-phase-noise-v1"
DEFAULT_SPATIAL_RADIUS = 3
DEFAULT_GAMMA = 30.0
DEFAULT_SOURCE_RHO = 0.2


class SelfImaginedPhaseNoiseError(RuntimeError):
    """Raised when the factorized noise contract is ambiguous or invalid."""


@dataclass(frozen=True)
class PhaseNoiseDiagnostics:
    spatial_radius: int
    gamma: float
    latent_shape: tuple[int, ...]
    dtype: str
    device: str
    low_frequency_bin_count: int
    total_spatial_bin_count: int
    low_frequency_bin_fraction: float
    low_frequency_energy_fraction_min: float
    low_frequency_energy_fraction_max: float
    compensation_beta_min: float
    compensation_beta_max: float
    reference_zero_magnitude_fraction: float
    input_l2_norm: tuple[float, ...]
    output_l2_norm: tuple[float, ...]
    l2_norm_max_relative_error: float
    ifft_imaginary_max_abs: float
    low_phase_cosine_min: float
    high_spectrum_max_relative_error_before_beta: float
    radius_zero_exact_gaussian_alias: bool
    energy_audit_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FactorizedPhaseNoiseResult:
    initial_noise: Any
    action_phase_noise: Any
    phase_diagnostics: PhaseNoiseDiagnostics
    source_bridge_receipt: Mapping[str, Any]
    receipt: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise SelfImaginedPhaseNoiseError("PyTorch is required") from error
    return torch


def _validate_latent(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise SelfImaginedPhaseNoiseError(f"{label} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise SelfImaginedPhaseNoiseError(f"{label} must be FP32 or FP64")
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or tuple(int(item) for item in value.shape[1:3])
        != (LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) < 4
        or int(value.shape[4]) < 4
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SelfImaginedPhaseNoiseError(
            f"{label} must be detached contiguous finite "
            f"[B,{LATENT_CHANNELS},{LATENT_PHASES},H,W]"
        )
    return value


def _validate_inputs(gaussian: Any, action_reference: Any, source: Any) -> tuple[Any, Any, Any]:
    gaussian = _validate_latent(gaussian, label="gaussian")
    action_reference = _validate_latent(action_reference, label="action_reference")
    source = _validate_latent(source, label="source_normalized_latent")
    for label, value in (
        ("action_reference", action_reference),
        ("source_normalized_latent", source),
    ):
        if (
            tuple(value.shape) != tuple(gaussian.shape)
            or value.dtype != gaussian.dtype
            or value.device != gaussian.device
        ):
            raise SelfImaginedPhaseNoiseError(
                f"{label} must share Gaussian shape, dtype, and device"
            )
    return gaussian, action_reference, source


def _validate_radius(value: int, *, height: int, width: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SelfImaginedPhaseNoiseError("spatial_radius must be an integer")
    radius = int(value)
    maximum = min(height, width) // 2 - 1
    if not 0 <= radius <= maximum:
        raise SelfImaginedPhaseNoiseError(
            f"spatial_radius must lie in [0,{maximum}] for this latent"
        )
    return radius


def _validate_gamma(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelfImaginedPhaseNoiseError("gamma must be a real scalar")
    gamma = float(value)
    if not math.isfinite(gamma) or gamma < 1.0:
        raise SelfImaginedPhaseNoiseError("gamma must be finite and >= 1")
    return gamma


def _validate_rho(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SelfImaginedPhaseNoiseError("source_rho must be a real scalar")
    rho = float(value)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise SelfImaginedPhaseNoiseError("source_rho must lie in [0,1]")
    return rho


def spatial_low_frequency_mask(
    height: int,
    width: int,
    *,
    radius: int,
    device: Any = "cpu",
) -> Any:
    """Return one conjugate-symmetric unshifted radial FFT mask."""

    torch = _torch()
    if type(height) is not int or type(width) is not int or height < 4 or width < 4:
        raise SelfImaginedPhaseNoiseError("height/width must be integers >= 4")
    radius = _validate_radius(radius, height=height, width=width)
    fy = torch.fft.fftfreq(height, d=1.0 / height, device=device)
    fx = torch.fft.fftfreq(width, d=1.0 / width, device=device)
    squared_radius = fy[:, None].square() + fx[None, :].square()
    mask = squared_radius <= float(radius * radius)
    if int(mask.sum().item()) <= 0 or not bool(mask[0, 0].item()):
        raise SelfImaginedPhaseNoiseError("low-frequency mask lost the DC bin")
    # A real-valued inverse requires selection of conjugate pairs.  The radial
    # mask is symmetric by construction; verify it instead of relying on prose.
    conjugate_y = (-torch.arange(height, device=device)) % height
    conjugate_x = (-torch.arange(width, device=device)) % width
    if not torch.equal(mask, mask.index_select(0, conjugate_y).index_select(1, conjugate_x)):
        raise SelfImaginedPhaseNoiseError("frequency mask is not conjugate symmetric")
    return mask.contiguous()


def _sample_l2(value: Any) -> Any:
    return value.flatten(1).square().sum(dim=1).sqrt()


def _float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().tolist())


def _phase_cosine(left: Any, right: Any, valid: Any) -> float:
    torch = _torch()
    product = left * right.conj()
    cosine = product.real / (left.abs() * right.abs()).clamp_min(
        torch.finfo(left.real.dtype).tiny
    )
    selected = cosine.masked_select(valid)
    if selected.numel() == 0:
        return 1.0
    return float(selected.min().detach().cpu().item())


def spatial_action_phase_noise(
    gaussian: Any,
    action_reference: Any,
    *,
    spatial_radius: int = DEFAULT_SPATIAL_RADIUS,
    gamma: Real = DEFAULT_GAMMA,
) -> tuple[Any, PhaseNoiseDiagnostics]:
    """Inject proposal phase while preserving each ``[B,C,T]`` plane energy."""

    torch = _torch()
    gaussian = _validate_latent(gaussian, label="gaussian")
    action_reference = _validate_latent(action_reference, label="action_reference")
    if (
        tuple(action_reference.shape) != tuple(gaussian.shape)
        or action_reference.dtype != gaussian.dtype
        or action_reference.device != gaussian.device
    ):
        raise SelfImaginedPhaseNoiseError(
            "action_reference must share Gaussian shape, dtype, and device"
        )
    height, width = (int(gaussian.shape[-2]), int(gaussian.shape[-1]))
    radius = _validate_radius(spatial_radius, height=height, width=width)
    gamma_value = _validate_gamma(gamma)
    if radius == 0:
        norms = _sample_l2(gaussian.to(dtype=torch.float64))
        diagnostics = PhaseNoiseDiagnostics(
            spatial_radius=0,
            gamma=gamma_value,
            latent_shape=tuple(int(item) for item in gaussian.shape),
            dtype=str(gaussian.dtype),
            device=str(gaussian.device),
            low_frequency_bin_count=1,
            total_spatial_bin_count=height * width,
            low_frequency_bin_fraction=1.0 / float(height * width),
            low_frequency_energy_fraction_min=0.0,
            low_frequency_energy_fraction_max=0.0,
            compensation_beta_min=1.0,
            compensation_beta_max=1.0,
            reference_zero_magnitude_fraction=0.0,
            input_l2_norm=_float_tuple(norms),
            output_l2_norm=_float_tuple(norms),
            l2_norm_max_relative_error=0.0,
            ifft_imaginary_max_abs=0.0,
            low_phase_cosine_min=1.0,
            high_spectrum_max_relative_error_before_beta=0.0,
            radius_zero_exact_gaussian_alias=True,
            energy_audit_passed=True,
        )
        return gaussian, diagnostics

    # CPU FP64 is the reproducibility reference used by the runtime injector.
    # The function remains device-generic for tests, but callers should
    # establish cross-rank identity after moving the result to the sampler.
    work = gaussian.to(dtype=torch.float64)
    reference = action_reference.to(dtype=torch.float64)
    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    reference_spectrum = torch.fft.fft2(reference, dim=(-2, -1))
    mask_2d = spatial_low_frequency_mask(
        height, width, radius=radius, device=gaussian.device
    )
    mask = mask_2d.reshape(1, 1, 1, height, width)
    magnitude = spectrum.abs()
    reference_magnitude = reference_spectrum.abs()
    tiny = torch.finfo(torch.float64).eps * math.sqrt(float(height * width))
    usable_reference = reference_magnitude > tiny
    gaussian_phase = spectrum / magnitude.clamp_min(tiny)
    reference_phase = reference_spectrum / reference_magnitude.clamp_min(tiny)
    selected_phase = torch.where(usable_reference, reference_phase, gaussian_phase)

    magnitude_sq = magnitude.square()
    low_energy = (magnitude_sq * mask).sum(dim=(-2, -1), keepdim=True)
    high_energy = (magnitude_sq * (~mask)).sum(dim=(-2, -1), keepdim=True)
    total_energy = low_energy + high_energy
    if bool((high_energy <= torch.finfo(torch.float64).tiny).any().item()):
        raise SelfImaginedPhaseNoiseError("high-frequency Gaussian energy is degenerate")
    compensated = (total_energy - low_energy / (gamma_value * gamma_value)) / high_energy
    if bool((compensated <= 0.0).any().item()) or not bool(torch.isfinite(compensated).all().item()):
        raise SelfImaginedPhaseNoiseError("energy compensation factor is invalid")
    beta = compensated.sqrt()
    modified = torch.where(
        mask,
        magnitude * selected_phase / gamma_value,
        spectrum * beta,
    )
    inverse = torch.fft.ifft2(modified, dim=(-2, -1))
    imaginary_max = float(inverse.imag.abs().max().detach().cpu().item())
    real_scale = max(1.0, float(inverse.real.abs().max().detach().cpu().item()))
    if imaginary_max > 2.0e-10 * real_scale:
        raise SelfImaginedPhaseNoiseError(
            "phase substitution broke real-valued conjugate symmetry"
        )
    output = inverse.real.to(dtype=gaussian.dtype).contiguous()
    if not bool(torch.isfinite(output).all().item()):
        raise SelfImaginedPhaseNoiseError("phase noise is non-finite")

    input_norm = _sample_l2(work)
    output_norm = _sample_l2(output.to(dtype=torch.float64))
    relative_error = ((output_norm - input_norm).abs() / input_norm.clamp_min(1.0))
    tolerance = 5.0e-5 if gaussian.dtype == torch.float32 else 2.0e-10
    maximum_error = float(relative_error.max().detach().cpu().item())
    if maximum_error > tolerance:
        raise SelfImaginedPhaseNoiseError("realized spatial energy was not preserved")

    low_valid = mask & usable_reference & (magnitude > tiny)
    low_phase_cosine = _phase_cosine(modified, reference_spectrum, low_valid)
    high_expected = spectrum * beta
    high_relative = (
        (modified - high_expected).abs()
        / high_expected.abs().clamp_min(tiny)
    ).masked_select(~mask)
    high_error = (
        float(high_relative.max().detach().cpu().item())
        if high_relative.numel()
        else 0.0
    )
    low_fraction = low_energy / total_energy.clamp_min(torch.finfo(torch.float64).tiny)
    zero_reference = mask & (~usable_reference)
    diagnostics = PhaseNoiseDiagnostics(
        spatial_radius=radius,
        gamma=gamma_value,
        latent_shape=tuple(int(item) for item in gaussian.shape),
        dtype=str(gaussian.dtype),
        device=str(gaussian.device),
        low_frequency_bin_count=int(mask_2d.sum().item()),
        total_spatial_bin_count=height * width,
        low_frequency_bin_fraction=float(mask_2d.float().mean().item()),
        low_frequency_energy_fraction_min=float(low_fraction.min().cpu().item()),
        low_frequency_energy_fraction_max=float(low_fraction.max().cpu().item()),
        compensation_beta_min=float(beta.min().cpu().item()),
        compensation_beta_max=float(beta.max().cpu().item()),
        reference_zero_magnitude_fraction=float(zero_reference.float().mean().item()),
        input_l2_norm=_float_tuple(input_norm),
        output_l2_norm=_float_tuple(output_norm),
        l2_norm_max_relative_error=maximum_error,
        ifft_imaginary_max_abs=imaginary_max,
        low_phase_cosine_min=low_phase_cosine,
        high_spectrum_max_relative_error_before_beta=high_error,
        radius_zero_exact_gaussian_alias=False,
        energy_audit_passed=True,
    )
    return output, diagnostics


def build_factorized_phase_noise(
    gaussian: Any,
    action_reference: Any,
    source_normalized_latent: Any,
    *,
    spatial_radius: int = DEFAULT_SPATIAL_RADIUS,
    gamma: Real = DEFAULT_GAMMA,
    source_rho: Real = DEFAULT_SOURCE_RHO,
) -> FactorizedPhaseNoiseResult:
    """Compose self-imagined action phase with source temporal-DC identity."""

    gaussian, action_reference, source = _validate_inputs(
        gaussian, action_reference, source_normalized_latent
    )
    rho = _validate_rho(source_rho)
    action_noise, phase_diagnostics = spatial_action_phase_noise(
        gaussian,
        action_reference,
        spatial_radius=spatial_radius,
        gamma=gamma,
    )
    try:
        bridged = source_bridge.source_spectral_bridge(
            action_noise,
            source,
            rho=rho,
            arm=source_bridge.PRIMARY_ARM,
        )
    except source_bridge.SourceSpectralBridgeError as error:
        raise SelfImaginedPhaseNoiseError(str(error)) from error
    output = bridged.initial_noise
    if phase_diagnostics.spatial_radius == 0 and rho == 0.0 and output is not gaussian:
        raise SelfImaginedPhaseNoiseError("inactive operator lost exact Gaussian alias")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "self_imagined_action_phase_plus_source_dc_noise",
        "pure_tensor_operator": True,
        "exact_video_frames": EXACT_VIDEO_FRAMES,
        "latent_phases": LATENT_PHASES,
        "accepted_inputs": [
            "native_gaussian",
            "self_generated_action_proposal_latent",
            "source_normalized_latent",
        ],
        "forbidden_inputs": [
            "paired_target_edit",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "action_carrier": {
            "operator": "low_spatial_frequency_phase_substitution",
            "spatial_radius": phase_diagnostics.spatial_radius,
            "gamma": phase_diagnostics.gamma,
            "proposal_is_not_an_rgb_or_latent_regression_target": True,
            "caller_must_verify_frozen_t2v_provenance": True,
        },
        "source_carrier": {
            "operator": source_bridge.PRIMARY_ARM,
            "rho": rho,
            "all_frame_temporal_set_statistics": True,
            "temporal_order_invariant": True,
            "ordered_source_trajectory_injected": False,
            "intermediate_noise_temporal_residual_preserved_by_source_bridge": True,
            "semantic_action_preservation_not_claimed": True,
        },
        "distribution_claim": {
            "spectral_energy_preserved": True,
            "modified_noise_proven_gaussian": False,
            "likelihood_or_density_claim": False,
        },
        "train_test_contract": {
            "source_available_at_inference": True,
            "action_proposal_requires_internal_t2v_generation_or_distillation": True,
            "no_hidden_target_condition": True,
        },
        "phase_diagnostics": phase_diagnostics.as_dict(),
        "source_bridge": dict(bridged.receipt),
        "scientific_claim_authorized": False,
    }
    return FactorizedPhaseNoiseResult(
        initial_noise=output,
        action_phase_noise=action_noise,
        phase_diagnostics=phase_diagnostics,
        source_bridge_receipt=bridged.receipt,
        receipt=receipt,
    )


__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_SOURCE_RHO",
    "DEFAULT_SPATIAL_RADIUS",
    "EXACT_VIDEO_FRAMES",
    "FactorizedPhaseNoiseResult",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "PhaseNoiseDiagnostics",
    "SCHEMA_VERSION",
    "SelfImaginedPhaseNoiseError",
    "build_factorized_phase_noise",
    "spatial_action_phase_noise",
    "spatial_low_frequency_mask",
]

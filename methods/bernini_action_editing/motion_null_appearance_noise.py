"""Isolated motion-null appearance-noise ablation for exact-81 Bernini.

This module deliberately does not self-register with a trainer, sampler,
launcher, or model registry.  A dedicated runner may bind the pure operator
explicitly and must record that runtime binding separately.  The operator's
complete public input surface is:

* the canonical Gaussian target latent;
* a collection of independently materialized ``T=1`` source-image latents;
* a bounded rotation coefficient ``rho``; and
* a domain-separated carrier seed.

The source descriptor retains channel means, high-pass channel Gram statistics,
and radially pooled mid/high spatial-frequency power.  It never retains a
source frame index, temporal phase, spatial Fourier phase, low-frequency
layout, full-video latent, target, or action proposal.  A source-independent
random spatial phase is coloured by those appearance statistics and repeated
unchanged over all 21 latent phases.  Only the centered temporal-DC component
of the realized Gaussian is rotated toward that carrier; the scalar Gaussian
mean and complete Gaussian temporal residual are preserved.

``rho > 0`` is a source-conditioned non-Gaussian endpoint.  Norm preservation
does not make it Gaussian and motion-null construction does not prove semantic
old-action independence: static pose/appearance statistics can still correlate
with action labels.  This file therefore authorizes only an isolated ablation
and requires external identity/action probes before any integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence


SCHEMA_VERSION = "bernini-motion-null-appearance-noise-v2"
EXACT_VIDEO_FRAMES = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
MIN_INDEPENDENT_FRAMES = 2
MAX_INDEPENDENT_FRAMES = EXACT_VIDEO_FRAMES
MIN_SPATIAL_EXTENT = 8
HIGH_PASS_CUTOFF = 0.125
NUM_RADIAL_BANDS = 4


class MotionNullAppearanceNoiseError(RuntimeError):
    """Raised before an ambiguous or motion-bearing carrier is accepted."""


@dataclass(frozen=True)
class MotionNullAppearanceNoiseDiagnostics:
    """Serializable numerical and isolation audit for one builder call."""

    rho: float
    carrier_seed: int
    gaussian_shape: tuple[int, ...]
    independent_frame_count: int
    dtype: str
    device: str
    rho_zero_exact_object_alias: bool
    source_conditioned_non_gaussian: bool
    carrier_constructed: bool
    independent_t1_storage_verified: bool
    source_temporal_indices_consumed: bool
    source_temporal_phase_consumed: bool
    source_spatial_phase_consumed: bool
    source_low_frequency_layout_consumed: bool
    carrier_strict_temporal_dc: bool
    descriptor_sha256: Optional[str]
    carrier_sha256: Optional[str]
    high_pass_cutoff: float
    radial_band_count: int
    gaussian_scalar_mean_max_abs_error: float
    gaussian_total_norm_max_relative_error: float
    gaussian_centered_dc_norm_max_relative_error: float
    gaussian_temporal_residual_max_abs_error: float
    carrier_temporal_dc_max_abs_error: float
    carrier_gaussian_dc_normalized_dot_max: float
    carrier_gaussian_dc_norm_max_relative_error: float
    synthesis_ifft_imaginary_max_abs: float
    numerical_audit_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionNullAppearanceNoiseResult:
    """The initial noise, optional active carrier, diagnostics, and receipt."""

    initial_noise: Any
    temporal_dc_carrier: Optional[Any]
    diagnostics: MotionNullAppearanceNoiseDiagnostics
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class _AppearanceDescriptor:
    channel_mean: Any
    high_pass_channel_gram: Any
    radial_power: Any
    band_masks: tuple[Any, ...]
    high_pass_mask: Any
    digest: str


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - AUH runtime supplies torch
        raise MotionNullAppearanceNoiseError(
            "motion-null appearance noise requires PyTorch"
        ) from error
    return torch


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
        raise MotionNullAppearanceNoiseError(
            f"receipt is not canonical finite ASCII JSON: {error}"
        ) from error


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        raise MotionNullAppearanceNoiseError("tensor digest requires materialized tensor")
    tensor = value.detach().contiguous().cpu()
    metadata = _canonical_json_bytes(
        {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _validate_rho(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MotionNullAppearanceNoiseError("rho must be a real scalar")
    rho = float(value)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise MotionNullAppearanceNoiseError("rho must be finite and lie in [0,1]")
    return rho


def _validate_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MotionNullAppearanceNoiseError("carrier_seed must be an integer")
    seed = int(value)
    if not 0 <= seed < 2**63:
        raise MotionNullAppearanceNoiseError(
            "carrier_seed must lie in [0,2^63)"
        )
    return seed


def _is_standalone_storage(value: Any) -> bool:
    """Reject slices/views that could be cut from one full-video latent."""

    expected = int(value.numel()) * int(value.element_size())
    try:
        storage_bytes = int(value.untyped_storage().nbytes())
    except (AttributeError, RuntimeError):  # pragma: no cover - old torch fallback
        storage_bytes = int(value.storage().nbytes())
    return (
        value._base is None
        and int(value.storage_offset()) == 0
        and storage_bytes == expected
    )


def _validate_gaussian(value: Any) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise MotionNullAppearanceNoiseError(
            "canonical_gaussian must be a torch.Tensor"
        )
    if value.dtype not in (torch.float32, torch.float64):
        raise MotionNullAppearanceNoiseError(
            "canonical_gaussian must be FP32 or FP64"
        )
    if (
        value.layout != torch.strided
        or value.device.type == "meta"
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or value.ndim != 5
        or tuple(int(item) for item in value.shape[:3])
        != (1, LATENT_CHANNELS, LATENT_PHASES)
        or int(value.shape[3]) < MIN_SPATIAL_EXTENT
        or int(value.shape[4]) < MIN_SPATIAL_EXTENT
        or int(value.shape[3]) % 2
        or int(value.shape[4]) % 2
        or not bool(torch.isfinite(value).all().item())
    ):
        raise MotionNullAppearanceNoiseError(
            "canonical_gaussian must be detached contiguous finite "
            f"[1,{LATENT_CHANNELS},{LATENT_PHASES},evenH,evenW]"
        )
    return value


def _validate_independent_frames(
    values: Sequence[Any], canonical_gaussian: Any
) -> tuple[Any, ...]:
    torch = _torch()
    if isinstance(values, torch.Tensor) or not isinstance(values, (tuple, list)):
        raise MotionNullAppearanceNoiseError(
            "independent_frame_latents must be a tuple/list of standalone T=1 tensors; "
            "a full-video tensor is forbidden"
        )
    frames = tuple(values)
    if not MIN_INDEPENDENT_FRAMES <= len(frames) <= MAX_INDEPENDENT_FRAMES:
        raise MotionNullAppearanceNoiseError(
            f"independent_frame_latents must contain {MIN_INDEPENDENT_FRAMES}.."
            f"{MAX_INDEPENDENT_FRAMES} standalone frames"
        )
    expected_shape = (
        1,
        LATENT_CHANNELS,
        1,
        int(canonical_gaussian.shape[3]),
        int(canonical_gaussian.shape[4]),
    )
    storage_pointers: set[tuple[str, int]] = set()
    gaussian_pointer = (
        str(canonical_gaussian.device),
        int(canonical_gaussian.untyped_storage().data_ptr()),
    )
    for index, frame in enumerate(frames):
        if not isinstance(frame, torch.Tensor):
            raise MotionNullAppearanceNoiseError(
                f"independent frame {index} is not a tensor"
            )
        if (
            frame.dtype != canonical_gaussian.dtype
            or frame.device != canonical_gaussian.device
            or frame.layout != torch.strided
            or frame.requires_grad
            or frame.grad_fn is not None
            or not frame.is_contiguous()
            or tuple(int(item) for item in frame.shape) != expected_shape
            or not bool(torch.isfinite(frame).all().item())
            or not _is_standalone_storage(frame)
        ):
            raise MotionNullAppearanceNoiseError(
                f"independent frame {index} must be a standalone detached finite "
                f"T=1 image latent with shape {expected_shape}"
            )
        pointer = (str(frame.device), int(frame.untyped_storage().data_ptr()))
        if pointer == gaussian_pointer or pointer in storage_pointers:
            raise MotionNullAppearanceNoiseError(
                "independent frame tensors must not alias Gaussian or each other"
            )
        storage_pointers.add(pointer)
    return frames


def _canonical_sorted_mean(value: Any) -> Any:
    """Reduce the frame axis without retaining input order or FP sum order."""

    return _torch().sort(value, dim=0).values.mean(dim=0)


def _canonicalize_cyclic_spatial_origin(value: Any) -> Any:
    """Choose one byte-exact representative of a frame's translation orbit.

    FFT magnitudes and cross-channel Grams are translation invariant over the
    reals, but two equivalent rolled tensors can take slightly different
    floating-point FFT paths.  Choosing the lexicographically smallest cyclic
    roll first makes that invariance auditable at the descriptor/digest level.
    The chosen origin is never stored, returned, or used as synthesis phase.
    """

    torch = _torch()
    if value.device.type != "cpu" or value.dtype != torch.float64 or value.ndim != 3:
        raise MotionNullAppearanceNoiseError(
            "spatial-origin canonicalization requires CPU float64 [C,H,W]"
        )
    height, width = int(value.shape[-2]), int(value.shape[-1])
    best_value = None
    best_key = None
    for shift_y in range(height):
        for shift_x in range(width):
            candidate = torch.roll(
                value, shifts=(shift_y, shift_x), dims=(-2, -1)
            ).contiguous()
            key = candidate.view(torch.uint8).numpy().tobytes(order="C")
            if best_key is None or key < best_key:
                best_key = key
                best_value = candidate
    if best_value is None:  # pragma: no cover - validated nonempty geometry
        raise MotionNullAppearanceNoiseError("empty spatial translation orbit")
    return best_value


def _frequency_masks(height: int, width: int) -> tuple[Any, tuple[Any, ...]]:
    torch = _torch()
    fy = torch.fft.fftfreq(height, d=1.0, device="cpu", dtype=torch.float64)
    fx = torch.fft.fftfreq(width, d=1.0, device="cpu", dtype=torch.float64)
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    maximum = float(radius.max().item())
    if not maximum > HIGH_PASS_CUTOFF:
        raise MotionNullAppearanceNoiseError(
            "spatial geometry has no registered mid/high-frequency support"
        )
    edges = torch.linspace(
        HIGH_PASS_CUTOFF,
        maximum,
        NUM_RADIAL_BANDS + 1,
        dtype=torch.float64,
        device="cpu",
    )
    bands: list[Any] = []
    for index in range(NUM_RADIAL_BANDS):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (
            (radius >= lower) & (radius <= upper)
            if index == NUM_RADIAL_BANDS - 1
            else (radius >= lower) & (radius < upper)
        )
        if int(mask.sum().item()) == 0:
            raise MotionNullAppearanceNoiseError(
                f"radial band {index} is empty for the supplied geometry"
            )
        bands.append(mask.contiguous())
    high_pass = torch.stack(bands, dim=0).any(dim=0).contiguous()
    if bool(high_pass[0, 0].item()):
        raise MotionNullAppearanceNoiseError("high-pass mask unexpectedly retains DC")
    return high_pass, tuple(bands)


def _descriptor_digest(channel_mean: Any, gram: Any, radial_power: Any) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "channel_mean_sha256": _tensor_sha256(channel_mean),
        "high_pass_channel_gram_sha256": _tensor_sha256(gram),
        "radial_power_sha256": _tensor_sha256(radial_power),
        "high_pass_cutoff_hex": float(HIGH_PASS_CUTOFF).hex(),
        "radial_band_count": NUM_RADIAL_BANDS,
        "source_spatial_phase_retained": False,
        "source_temporal_order_retained": False,
    }
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _build_appearance_descriptor(frames: tuple[Any, ...]) -> _AppearanceDescriptor:
    torch = _torch()
    height, width = int(frames[0].shape[-2]), int(frames[0].shape[-1])
    values = torch.stack(
        [
            _canonicalize_cyclic_spatial_origin(
                frame.detach().cpu().double()[0, :, 0]
            )
            for frame in frames
        ],
        dim=0,
    ).contiguous()
    # Every aggregation first sorts the independent frame contributions.  A
    # permutation/reversal therefore cannot alter the realized FP reduction.
    frame_channel_means = values.mean(dim=(-2, -1))
    channel_mean = _canonical_sorted_mean(frame_channel_means).contiguous()
    centered = values - channel_mean.reshape(1, LATENT_CHANNELS, 1, 1)
    spectrum = torch.fft.fft2(centered, dim=(-2, -1))
    high_pass_mask, band_masks = _frequency_masks(height, width)
    high_spectrum = spectrum * high_pass_mask.reshape(1, 1, height, width)
    # Parseval-equivalent Gram retains channel co-occurrence but no absolute
    # source spatial phase.  A common spatial translation cancels in the cross
    # spectrum before this reduction.
    high_spatial = torch.fft.ifft2(high_spectrum, dim=(-2, -1)).real
    flat = high_spatial.reshape(len(frames), LATENT_CHANNELS, height * width)
    frame_gram = torch.einsum("fcp,fdp->fcd", flat, flat) / float(height * width)
    gram = _canonical_sorted_mean(frame_gram).contiguous()
    power = spectrum.abs().square() / float(height * width)
    frame_radial: list[Any] = []
    for mask in band_masks:
        frame_radial.append(power[:, :, mask].mean(dim=-1))
    radial_by_frame = torch.stack(frame_radial, dim=-1)
    radial_power = _canonical_sorted_mean(radial_by_frame).contiguous()
    if (
        not bool(torch.isfinite(channel_mean).all().item())
        or not bool(torch.isfinite(gram).all().item())
        or not bool(torch.isfinite(radial_power).all().item())
        or bool((radial_power < 0.0).any().item())
    ):
        raise MotionNullAppearanceNoiseError("appearance descriptor is non-finite")
    gram = 0.5 * (gram + gram.transpose(0, 1))
    digest = _descriptor_digest(channel_mean, gram, radial_power)
    return _AppearanceDescriptor(
        channel_mean,
        gram,
        radial_power,
        band_masks,
        high_pass_mask,
        digest,
    )


def _symmetric_matrix_power(value: Any, exponent: float, *, label: str) -> Any:
    torch = _torch()
    symmetric = 0.5 * (value + value.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    maximum = float(eigenvalues.abs().max().item())
    floor = max(1.0e-12, maximum * 1.0e-10)
    if exponent < 0.0:
        powered = eigenvalues.clamp_min(floor).pow(exponent)
    else:
        powered = eigenvalues.clamp_min(0.0).pow(exponent)
    result = (eigenvectors * powered.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    if not bool(torch.isfinite(result).all().item()):
        raise MotionNullAppearanceNoiseError(f"{label} matrix power is non-finite")
    return result


def _synthesize_spatial_carrier(
    descriptor: _AppearanceDescriptor,
    *,
    height: int,
    width: int,
    carrier_seed: int,
) -> tuple[Any, float]:
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(carrier_seed)
    random_spatial = torch.randn(
        (LATENT_CHANNELS, height, width),
        dtype=torch.float64,
        device="cpu",
        generator=generator,
    )
    random_spectrum = torch.fft.fft2(random_spatial, dim=(-2, -1))
    shaped_spectrum = torch.zeros_like(random_spectrum)
    tiny = torch.finfo(torch.float64).eps
    for band_index, mask in enumerate(descriptor.band_masks):
        base_power = (
            random_spectrum[:, mask].abs().square().mean(dim=-1)
            / float(height * width)
        )
        source_power = descriptor.radial_power[:, band_index]
        scale = ((source_power + tiny) / (base_power + tiny)).sqrt()
        shaped_spectrum[:, mask] = (
            random_spectrum[:, mask] * scale.unsqueeze(1)
        )
    inverse = torch.fft.ifft2(shaped_spectrum, dim=(-2, -1))
    imaginary_max = float(inverse.imag.abs().max().item())
    real_scale = max(1.0, float(inverse.real.abs().max().item()))
    if imaginary_max > 2.0e-10 * real_scale:
        raise MotionNullAppearanceNoiseError(
            "source-independent spectral synthesis lost real conjugate symmetry"
        )
    texture = inverse.real.reshape(LATENT_CHANNELS, height * width)
    texture = texture - texture.mean(dim=1, keepdim=True)
    covariance = texture @ texture.transpose(0, 1) / float(height * width)
    whitened = _symmetric_matrix_power(
        covariance, -0.5, label="random texture covariance"
    ) @ texture
    coloured = _symmetric_matrix_power(
        descriptor.high_pass_channel_gram,
        0.5,
        label="source high-pass Gram",
    ) @ whitened
    relative_channel_mean = (
        descriptor.channel_mean - descriptor.channel_mean.mean()
    ).reshape(LATENT_CHANNELS, 1)
    carrier_phase = (
        coloured + relative_channel_mean
    ).reshape(LATENT_CHANNELS, height, width)
    carrier_phase = carrier_phase - carrier_phase.mean()
    if (
        not bool(torch.isfinite(carrier_phase).all().item())
        or float(carrier_phase.square().sum().sqrt().item()) <= 1.0e-12
    ):
        raise MotionNullAppearanceNoiseError(
            "appearance statistics produced a degenerate spatial carrier"
        )
    return carrier_phase.contiguous(), imaginary_max


def _flat_l2(value: Any) -> Any:
    return value.flatten(1).square().sum(dim=1).sqrt()


def _normalized_dot_max(left: Any, right: Any) -> float:
    torch = _torch()
    numerator = (left.flatten(1) * right.flatten(1)).sum(dim=1).abs()
    denominator = _flat_l2(left) * _flat_l2(right)
    ratio = torch.where(
        denominator > torch.finfo(left.dtype).tiny,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return float(ratio.max().item())


def _relative_error_max(actual: Any, expected: Any) -> float:
    torch = _torch()
    denominator = torch.maximum(expected.abs(), torch.ones_like(expected))
    return float(((actual - expected).abs() / denominator).max().item())


def _active_noise(
    gaussian: Any,
    descriptor: _AppearanceDescriptor,
    *,
    rho: float,
    carrier_seed: int,
) -> tuple[Any, Any, dict[str, float]]:
    torch = _torch()
    work = gaussian.detach().cpu().double()
    height, width = int(work.shape[-2]), int(work.shape[-1])
    carrier_phase, imaginary_max = _synthesize_spatial_carrier(
        descriptor,
        height=height,
        width=width,
        carrier_seed=carrier_seed,
    )
    carrier = carrier_phase.reshape(
        1, LATENT_CHANNELS, 1, height, width
    ).expand(1, LATENT_CHANNELS, LATENT_PHASES, height, width)
    carrier = carrier - carrier.mean()

    gaussian_dc_phase = work.mean(dim=2, keepdim=True)
    scalar_mean = gaussian_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    centered_dc = (gaussian_dc_phase - scalar_mean).expand_as(work)
    temporal_residual = work - gaussian_dc_phase.expand_as(work)
    dc_energy = centered_dc.flatten(1).square().sum(dim=1, keepdim=True)
    if bool((dc_energy <= 1.0e-20).any().item()):
        raise MotionNullAppearanceNoiseError(
            "canonical Gaussian centered temporal-DC subspace is degenerate"
        )
    projection = (
        (carrier.flatten(1) * centered_dc.flatten(1)).sum(dim=1, keepdim=True)
        / dc_energy
    )
    view = (int(work.shape[0]),) + (1,) * (work.ndim - 1)
    orthogonal = carrier - projection.reshape(view) * centered_dc
    orthogonal = orthogonal - orthogonal.mean()
    # Re-project after numerical centering so the registered orthogonality is
    # established by the actual tensor that will be norm matched.
    projection_2 = (
        (orthogonal.flatten(1) * centered_dc.flatten(1)).sum(dim=1, keepdim=True)
        / dc_energy
    )
    orthogonal = orthogonal - projection_2.reshape(view) * centered_dc
    orthogonal_norm = _flat_l2(orthogonal)
    dc_norm = _flat_l2(centered_dc)
    if bool((orthogonal_norm <= 1.0e-12).any().item()):
        raise MotionNullAppearanceNoiseError(
            "appearance carrier is degenerate or collinear with Gaussian DC"
        )
    matched_carrier = orthogonal * (dc_norm / orthogonal_norm).reshape(view)
    mixed_dc = math.sqrt(max(0.0, 1.0 - rho * rho)) * centered_dc + rho * matched_carrier
    output_work = scalar_mean.expand_as(work) + temporal_residual + mixed_dc
    output = output_work.to(dtype=gaussian.dtype).to(device=gaussian.device).contiguous()
    carrier_output = matched_carrier.to(dtype=gaussian.dtype).to(
        device=gaussian.device
    ).contiguous()
    if (
        not bool(torch.isfinite(output).all().item())
        or output.requires_grad
        or output.grad_fn is not None
    ):
        raise MotionNullAppearanceNoiseError(
            "motion-null appearance mix produced invalid initial noise"
        )

    output64 = output.detach().cpu().double()
    output_dc_phase = output64.mean(dim=2, keepdim=True)
    output_scalar = output_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    output_centered_dc = (output_dc_phase - output_scalar).expand_as(output64)
    output_residual = output64 - output_dc_phase.expand_as(output64)
    residual_error = float((output_residual - temporal_residual).abs().max().item())
    carrier_dc_error = float(
        (carrier_output.detach().cpu().double()
         - carrier_output.detach().cpu().double()[:, :, :1].expand_as(
             carrier_output.detach().cpu().double()
         )).abs().max().item()
    )
    metrics = {
        "scalar_mean_error": float((output_scalar - scalar_mean).abs().max().item()),
        "total_norm_error": _relative_error_max(_flat_l2(output64), _flat_l2(work)),
        "dc_norm_error": _relative_error_max(
            _flat_l2(output_centered_dc), _flat_l2(centered_dc)
        ),
        "residual_error": residual_error,
        "carrier_dc_error": carrier_dc_error,
        "carrier_dot": _normalized_dot_max(matched_carrier, centered_dc),
        "carrier_norm_error": _relative_error_max(
            _flat_l2(matched_carrier), _flat_l2(centered_dc)
        ),
        "imaginary_max": imaginary_max,
    }
    scale = max(1.0, float(work.abs().max().item()))
    if gaussian.dtype == torch.float64:
        absolute_tolerance, relative_tolerance = 2.0e-10, 2.0e-10
    else:
        absolute_tolerance, relative_tolerance = 8.0e-6, 6.0e-5
    if (
        metrics["scalar_mean_error"] > absolute_tolerance * scale
        or metrics["total_norm_error"] > relative_tolerance
        or metrics["dc_norm_error"] > relative_tolerance
        or metrics["residual_error"] > absolute_tolerance * scale
        or metrics["carrier_dc_error"] != 0.0
        or metrics["carrier_dot"] > 2.0e-10
        or metrics["carrier_norm_error"] > 2.0e-10
    ):
        raise MotionNullAppearanceNoiseError(
            "motion-null appearance noise failed its numerical invariants"
        )
    return output, carrier_output, metrics


def _receipt(
    diagnostics: MotionNullAppearanceNoiseDiagnostics,
) -> Mapping[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "motion_null_appearance_statistic_dc_noise",
        "ablation_only": True,
        "trainer_integration_executed": False,
        "operator_self_registers_sampler_hook": False,
        "operator_self_registers_launcher": False,
        "scientific_claim_authorized": False,
        "semantic_old_action_absence_claimed": False,
        "public_api_inputs": [
            "canonical_gaussian",
            "independent_frame_latents",
            "rho",
            "carrier_seed",
        ],
        "forbidden_api_inputs": [
            "full_video_latent",
            "target",
            "paired_target",
            "action_proposal",
            "motion_reference",
            "mask",
            "flow",
            "pose",
            "track",
            "trajectory",
        ],
        "source_contract": {
            "independently_materialized_T1_image_latents_required": True,
            "full_video_tensor_rejected": True,
            "shared_storage_slices_rejected": True,
            "frame_indices_consumed": False,
            "frame_order_consumed": False,
        },
        "descriptor": {
            "channel_mean": True,
            "high_pass_channel_gram": True,
            "radial_mid_high_frequency_power": True,
            "high_pass_cutoff": HIGH_PASS_CUTOFF,
            "radial_band_count": NUM_RADIAL_BANDS,
            "source_spatial_fourier_phase_retained": False,
            "source_low_frequency_layout_retained": False,
        },
        "carrier": {
            "source_independent_random_spatial_phase": True,
            "strict_temporal_dc": diagnostics.carrier_strict_temporal_dc,
            "source_temporal_phase_retained": False,
            "source_order_retained": False,
        },
        "mix": {
            "rotated_subspace": "centered_gaussian_temporal_dc_only",
            "scalar_gaussian_mean_preserved": True,
            "gaussian_temporal_residual_preserved": True,
            "centered_gaussian_dc_norm_preserved": True,
            "total_realized_l2_norm_preserved": True,
            "rho_parameterization": "sqrt(1-rho^2)*gaussian_dc+rho*orthogonal_carrier",
        },
        "distribution": {
            "rho_zero_exact_gaussian_object_alias": diagnostics.rho_zero_exact_object_alias,
            "rho_positive_source_conditioned_non_gaussian": diagnostics.source_conditioned_non_gaussian,
            "gaussianity_claimed_for_active_rho": False,
        },
        "required_external_gates": [
            "heldout_identity_retrieval",
            "heldout_old_action_direction_milestone_order_probe",
            "correct_vs_wrong_source_causal_render",
            "target_action_noninferiority",
            "blur_flicker_camera_quality_noninferiority",
        ],
        "diagnostics": diagnostics.as_dict(),
    }


def build_motion_null_appearance_noise(
    *,
    canonical_gaussian: Any,
    independent_frame_latents: Sequence[Any],
    rho: Real,
    carrier_seed: int,
) -> MotionNullAppearanceNoiseResult:
    """Build one isolated motion-null appearance endpoint.

    The keyword-only signature intentionally has no extensibility hook.  Python
    rejects target-, proposal-, and full-video-like keyword arguments before
    this function executes.
    """

    gaussian = _validate_gaussian(canonical_gaussian)
    frames = _validate_independent_frames(independent_frame_latents, gaussian)
    rho_value = _validate_rho(rho)
    seed = _validate_seed(carrier_seed)
    if rho_value == 0.0:
        diagnostics = MotionNullAppearanceNoiseDiagnostics(
            rho=0.0,
            carrier_seed=seed,
            gaussian_shape=tuple(int(item) for item in gaussian.shape),
            independent_frame_count=len(frames),
            dtype=str(gaussian.dtype),
            device=str(gaussian.device),
            rho_zero_exact_object_alias=True,
            source_conditioned_non_gaussian=False,
            carrier_constructed=False,
            independent_t1_storage_verified=True,
            source_temporal_indices_consumed=False,
            source_temporal_phase_consumed=False,
            source_spatial_phase_consumed=False,
            source_low_frequency_layout_consumed=False,
            carrier_strict_temporal_dc=True,
            descriptor_sha256=None,
            carrier_sha256=None,
            high_pass_cutoff=HIGH_PASS_CUTOFF,
            radial_band_count=NUM_RADIAL_BANDS,
            gaussian_scalar_mean_max_abs_error=0.0,
            gaussian_total_norm_max_relative_error=0.0,
            gaussian_centered_dc_norm_max_relative_error=0.0,
            gaussian_temporal_residual_max_abs_error=0.0,
            carrier_temporal_dc_max_abs_error=0.0,
            carrier_gaussian_dc_normalized_dot_max=0.0,
            carrier_gaussian_dc_norm_max_relative_error=0.0,
            synthesis_ifft_imaginary_max_abs=0.0,
            numerical_audit_passed=True,
        )
        return MotionNullAppearanceNoiseResult(
            initial_noise=gaussian,
            temporal_dc_carrier=None,
            diagnostics=diagnostics,
            receipt=_receipt(diagnostics),
        )

    descriptor = _build_appearance_descriptor(frames)
    output, carrier, metrics = _active_noise(
        gaussian,
        descriptor,
        rho=rho_value,
        carrier_seed=seed,
    )
    diagnostics = MotionNullAppearanceNoiseDiagnostics(
        rho=rho_value,
        carrier_seed=seed,
        gaussian_shape=tuple(int(item) for item in gaussian.shape),
        independent_frame_count=len(frames),
        dtype=str(gaussian.dtype),
        device=str(gaussian.device),
        rho_zero_exact_object_alias=False,
        source_conditioned_non_gaussian=True,
        carrier_constructed=True,
        independent_t1_storage_verified=True,
        source_temporal_indices_consumed=False,
        source_temporal_phase_consumed=False,
        source_spatial_phase_consumed=False,
        source_low_frequency_layout_consumed=False,
        carrier_strict_temporal_dc=metrics["carrier_dc_error"] == 0.0,
        descriptor_sha256=descriptor.digest,
        carrier_sha256=_tensor_sha256(carrier),
        high_pass_cutoff=HIGH_PASS_CUTOFF,
        radial_band_count=NUM_RADIAL_BANDS,
        gaussian_scalar_mean_max_abs_error=metrics["scalar_mean_error"],
        gaussian_total_norm_max_relative_error=metrics["total_norm_error"],
        gaussian_centered_dc_norm_max_relative_error=metrics["dc_norm_error"],
        gaussian_temporal_residual_max_abs_error=metrics["residual_error"],
        carrier_temporal_dc_max_abs_error=metrics["carrier_dc_error"],
        carrier_gaussian_dc_normalized_dot_max=metrics["carrier_dot"],
        carrier_gaussian_dc_norm_max_relative_error=metrics["carrier_norm_error"],
        synthesis_ifft_imaginary_max_abs=metrics["imaginary_max"],
        numerical_audit_passed=True,
    )
    return MotionNullAppearanceNoiseResult(
        initial_noise=output,
        temporal_dc_carrier=carrier,
        diagnostics=diagnostics,
        receipt=_receipt(diagnostics),
    )


__all__ = [
    "EXACT_VIDEO_FRAMES",
    "HIGH_PASS_CUTOFF",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "MAX_INDEPENDENT_FRAMES",
    "MIN_INDEPENDENT_FRAMES",
    "MotionNullAppearanceNoiseDiagnostics",
    "MotionNullAppearanceNoiseError",
    "MotionNullAppearanceNoiseResult",
    "NUM_RADIAL_BANDS",
    "SCHEMA_VERSION",
    "build_motion_null_appearance_noise",
]

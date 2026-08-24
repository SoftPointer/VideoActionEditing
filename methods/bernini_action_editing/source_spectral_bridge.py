"""Source-rich, norm-audited initial-noise ablations for exact-81 Bernini.

The primary arm in this module is intentionally much weaker than copying a
source trajectory into the sampler state.  It derives a time-constant carrier
from temporal *set* statistics of a normalized source latent, so permuting the
21 latent frames cannot change the carrier.  A supplied Gaussian realization
is decomposed into mutually orthogonal temporal-DC and temporal-residual
subspaces.  Only the centered temporal-DC subspace is rotated toward the
source carrier; the scalar mean and the complete Gaussian temporal residual
are preserved.

For a deliberately unsafe comparison, the module also exposes one deterministic
shuffled-frame arm.  That arm may rotate the Gaussian temporal residual toward
a permuted source residual.  It can never receive the ordered source trajectory,
but it is permanently labelled ``action_leakage_risk`` because shuffling does
not erase the source's motion content.

Both arms are pure tensor functions.  They accept no target, text, trainer, or
stage flag, which makes the train and inference operator contract identical.
Any active source-conditioned arm is explicitly non-Gaussian.  At ``rho == 0``
the exact input Gaussian tensor object is returned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


EXACT_VIDEO_FRAMES = 81
LATENT_FRAMES = 21
LATENT_CHANNELS = 16
PRIMARY_ARM = "temporal_set_dc"
SHUFFLED_FRAME_ABLATION_ARM = "deterministic_shuffled_frames_ablation"
SUPPORTED_ARMS = (PRIMARY_ARM, SHUFFLED_FRAME_ABLATION_ARM)
DEFAULT_SHUFFLE_SEED = 20260808
RECEIPT_SCHEMA = "bernini-source-spectral-bridge-v1"
LATENT_COORDINATE = "bernini_normalized_clean_vae_latent"
SET_STATISTIC_WEIGHTS = {
    "temporal_dc": 1.0,
    "temporal_rms": 0.5,
    "temporal_midrange": 0.25,
}


class SourceSpectralBridgeError(RuntimeError):
    """Raised when an ablation would violate the source/noise contract."""


@dataclass(frozen=True)
class BridgeDiagnostics:
    """Serializable numerical and leakage audit for one bridge call."""

    arm: str
    rho: float
    latent_shape: tuple[int, ...]
    dtype: str
    device: str
    source_only: bool
    paired_target_accessed: bool
    target_media_accessed: bool
    temporal_order_invariant_carrier: bool
    temporal_set_statistics: tuple[str, ...]
    ordered_source_trajectory_injected: bool
    source_temporal_residual_injected: bool
    action_leakage_risk: bool
    non_gaussian_initial_noise: bool
    rho_zero_exact_gaussian_alias: bool
    train_inference_same_contract: bool
    input_global_mean: tuple[float, ...]
    output_global_mean: tuple[float, ...]
    input_l2_norm: tuple[float, ...]
    output_l2_norm: tuple[float, ...]
    input_centered_dc_l2_norm: tuple[float, ...]
    output_centered_dc_l2_norm: tuple[float, ...]
    input_temporal_residual_l2_norm: tuple[float, ...]
    output_temporal_residual_l2_norm: tuple[float, ...]
    input_dc_residual_normalized_dot_max: float
    output_dc_residual_normalized_dot_max: float
    global_mean_max_abs_error: float
    total_norm_max_relative_error: float
    centered_dc_norm_max_relative_error: float
    temporal_residual_norm_max_relative_error: float
    primary_residual_max_abs_error: float
    carrier_base_normalized_dot_max: float
    shuffled_frame_permutation: Optional[tuple[int, ...]]
    moment_norm_audit_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpectralBridgeResult:
    """Initial noise plus its self-contained, non-claim receipt."""

    initial_noise: Any
    diagnostics: BridgeDiagnostics
    receipt: Mapping[str, Any]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - repository runtime has torch
        raise SourceSpectralBridgeError("source spectral bridge requires torch") from error
    return torch


def _validate_exact81_latent(value: Any, *, label: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise SourceSpectralBridgeError(f"{label} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise SourceSpectralBridgeError(f"{label} must be FP32 or FP64")
    if (
        value.ndim != 5
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != LATENT_CHANNELS
        or int(value.shape[2]) != LATENT_FRAMES
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
    ):
        raise SourceSpectralBridgeError(
            f"{label} must be [B,{LATENT_CHANNELS},{LATENT_FRAMES},H,W] "
            f"for exact-{EXACT_VIDEO_FRAMES} video"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise SourceSpectralBridgeError(f"{label} must be finite")
    return value


def _validate_pair(gaussian: Any, source: Any) -> tuple[Any, Any]:
    gaussian = _validate_exact81_latent(gaussian, label="gaussian")
    source = _validate_exact81_latent(source, label="source_normalized_latent")
    if (
        tuple(gaussian.shape) != tuple(source.shape)
        or gaussian.dtype != source.dtype
        or gaussian.device != source.device
    ):
        raise SourceSpectralBridgeError(
            "Gaussian and source normalized latent must share shape/dtype/device"
        )
    return gaussian, source


def _validate_rho(value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SourceSpectralBridgeError("rho must be a real scalar")
    rho = float(value)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise SourceSpectralBridgeError("rho must be finite and lie in [0,1]")
    return rho


def _validate_shuffle_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise SourceSpectralBridgeError("shuffle_seed must be an integer in [0,2^63)")
    return value


def temporal_dc_residual(value: Any) -> tuple[Any, Any]:
    """Return the orthogonal temporal-DC and zero-temporal-mean residual.

    The tensors retain the input shape.  This public helper uses the input
    dtype; the bridge itself repeats the decomposition in FP64 for its audit.
    """

    value = _validate_exact81_latent(value, label="latent")
    dc = value.mean(dim=2, keepdim=True).expand_as(value)
    residual = value - dc
    return dc, residual


def deterministic_frame_permutation(
    *, seed: int = DEFAULT_SHUFFLE_SEED,
) -> tuple[int, ...]:
    """Return one platform-stable, non-identity permutation of 21 phases."""

    torch = _torch()
    seed = _validate_shuffle_seed(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(LATENT_FRAMES, generator=generator).tolist()
    identity = list(range(LATENT_FRAMES))
    if permutation == identity:  # fail closed even for an adversarial seed
        permutation = identity[1:] + identity[:1]
    return tuple(int(index) for index in permutation)


def _center_and_unit_rms(feature: Any) -> Any:
    """Center one [B,C,1,H,W] statistic and normalize per nonzero sample."""

    torch = _torch()
    centered = feature - feature.mean(dim=(1, 2, 3, 4), keepdim=True)
    rms = centered.square().mean(dim=(1, 2, 3, 4), keepdim=True).sqrt()
    positive = rms > torch.finfo(centered.dtype).eps
    safe = torch.where(positive, rms, torch.ones_like(rms))
    return torch.where(positive, centered / safe, torch.zeros_like(centered))


def temporal_set_carrier(source_normalized_latent: Any) -> Any:
    """Build the primary, temporal-order-invariant DC carrier.

    Sorting along time before every reduction makes the implementation itself,
    not just its real-arithmetic formula, invariant to a frame permutation.
    The fixed carrier combines temporal mean, RMS, and midrange set statistics,
    then repeats one centered feature over all 21 phases.  It therefore cannot
    contain an ordered source trajectory.
    """

    source = _validate_exact81_latent(
        source_normalized_latent, label="source_normalized_latent"
    )
    ordered = _torch().sort(source.to(dtype=_torch().float64), dim=2).values
    temporal_dc = ordered.mean(dim=2, keepdim=True)
    temporal_rms = (
        (ordered - temporal_dc).square().mean(dim=2, keepdim=True).sqrt()
    )
    temporal_midrange = 0.5 * (
        ordered[:, :, :1, :, :] + ordered[:, :, -1:, :, :]
    )
    carrier_phase = (
        SET_STATISTIC_WEIGHTS["temporal_dc"]
        * _center_and_unit_rms(temporal_dc)
        + SET_STATISTIC_WEIGHTS["temporal_rms"]
        * _center_and_unit_rms(temporal_rms)
        + SET_STATISTIC_WEIGHTS["temporal_midrange"]
        * _center_and_unit_rms(temporal_midrange)
    )
    carrier_phase = carrier_phase - carrier_phase.mean(
        dim=(1, 2, 3, 4), keepdim=True
    )
    return carrier_phase.expand(-1, -1, LATENT_FRAMES, -1, -1).to(
        dtype=source.dtype
    )


def deterministic_shuffled_frame_carrier(
    source_normalized_latent: Any,
    *,
    seed: int = DEFAULT_SHUFFLE_SEED,
) -> Any:
    """Return a zero-DC permuted source residual for the unsafe ablation.

    This is deterministic and never returns the ordered source trajectory, but
    it is not an order-invariant statistic and may retain action information.
    Callers must keep the resulting arm labelled ``action_leakage_risk``.
    """

    source = _validate_exact81_latent(
        source_normalized_latent, label="source_normalized_latent"
    )
    permutation = deterministic_frame_permutation(seed=seed)
    index = _torch().tensor(permutation, device=source.device, dtype=_torch().long)
    shuffled = source.index_select(2, index)
    residual = shuffled.to(dtype=_torch().float64)
    residual = residual - residual.mean(dim=2, keepdim=True)
    return residual.to(dtype=source.dtype)


def _flat_dot(left: Any, right: Any) -> Any:
    return (left.flatten(1) * right.flatten(1)).sum(dim=1)


def _l2(value: Any) -> Any:
    return value.flatten(1).square().sum(dim=1).sqrt()


def _normalized_dot_max(left: Any, right: Any) -> float:
    torch = _torch()
    denominator = _l2(left) * _l2(right)
    numerator = _flat_dot(left, right).abs()
    ratio = torch.where(
        denominator > torch.finfo(left.dtype).tiny,
        numerator / denominator,
        torch.zeros_like(numerator),
    )
    return float(ratio.max().detach().cpu().item())


def _relative_error(actual: Any, expected: Any) -> Any:
    torch = _torch()
    denominator = torch.maximum(expected.abs(), torch.ones_like(expected))
    return (actual - expected).abs() / denominator


def _orthogonal_norm_mix(
    base: Any,
    carrier: Any,
    *,
    rho: float,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    """Rotate ``base`` toward a source carrier without changing its L2 norm."""

    torch = _torch()
    base_norm_sq = _flat_dot(base, base)
    tiny = torch.finfo(base.dtype).eps * max(1, int(base[0].numel()))
    if bool((base_norm_sq <= tiny).any().item()):
        raise SourceSpectralBridgeError(
            f"{label} Gaussian subspace is degenerate for active rho"
        )
    projection = _flat_dot(carrier, base) / base_norm_sq
    view = (int(base.shape[0]),) + (1,) * (base.ndim - 1)
    orthogonal = carrier - projection.reshape(view) * base
    orthogonal_norm_sq = _flat_dot(orthogonal, orthogonal)
    if bool((orthogonal_norm_sq <= tiny).any().item()):
        raise SourceSpectralBridgeError(
            f"{label} source carrier is degenerate or collinear with Gaussian"
        )
    scaled = orthogonal * (base_norm_sq / orthogonal_norm_sq).sqrt().reshape(view)
    base_weight = math.sqrt(max(0.0, 1.0 - rho * rho))
    mixed = base_weight * base + rho * scaled
    mixed_norm_sq = _flat_dot(mixed, mixed)
    if bool((mixed_norm_sq <= tiny).any().item()):
        raise SourceSpectralBridgeError(f"{label} mixed norm collapsed")
    # Remove the last FP roundoff in the realized norm.  This is not a learned
    # scale and is identical at train and inference.
    mixed = mixed * (base_norm_sq / mixed_norm_sq).sqrt().reshape(view)
    return mixed, {
        "base_norm": base_norm_sq.sqrt(),
        "mixed_norm": _l2(mixed),
        "base_carrier_normalized_dot_max": _normalized_dot_max(base, scaled),
    }


def _as_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in value.detach().cpu().tolist())


def _tolerances(dtype: Any) -> tuple[float, float]:
    torch = _torch()
    if dtype == torch.float64:
        return 2.0e-10, 2.0e-12
    return 4.0e-5, 4.0e-6


def _make_diagnostics(
    *,
    gaussian: Any,
    output: Any,
    arm: str,
    rho: float,
    carrier_base_normalized_dot_max: float,
    permutation: Optional[tuple[int, ...]],
) -> BridgeDiagnostics:
    torch = _torch()
    work_gaussian = gaussian.detach().to(dtype=torch.float64)
    work_output = output.detach().to(dtype=torch.float64)
    input_dc_phase = work_gaussian.mean(dim=2, keepdim=True)
    output_dc_phase = work_output.mean(dim=2, keepdim=True)
    input_scalar = input_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    output_scalar = output_dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    input_centered_dc = (input_dc_phase - input_scalar).expand_as(work_gaussian)
    output_centered_dc = (output_dc_phase - output_scalar).expand_as(work_output)
    input_residual = work_gaussian - input_dc_phase.expand_as(work_gaussian)
    output_residual = work_output - output_dc_phase.expand_as(work_output)
    input_total_norm = _l2(work_gaussian)
    output_total_norm = _l2(work_output)
    input_dc_norm = _l2(input_centered_dc)
    output_dc_norm = _l2(output_centered_dc)
    input_residual_norm = _l2(input_residual)
    output_residual_norm = _l2(output_residual)
    global_mean_error = (output_scalar - input_scalar).abs().flatten(1).max(dim=1).values
    total_relative_error = _relative_error(output_total_norm, input_total_norm)
    dc_relative_error = _relative_error(output_dc_norm, input_dc_norm)
    residual_relative_error = _relative_error(
        output_residual_norm, input_residual_norm
    )
    primary_residual_error = (
        (output_residual - input_residual).abs().max()
        if arm == PRIMARY_ARM
        else torch.zeros((), dtype=torch.float64, device=gaussian.device)
    )
    rtol, atol = _tolerances(gaussian.dtype)
    mean_limit = atol + rtol * input_scalar.abs().max()
    moment_norm_passed = bool(
        (global_mean_error.max() <= mean_limit).item()
        and (total_relative_error.max() <= rtol).item()
        and (dc_relative_error.max() <= rtol).item()
        and (residual_relative_error.max() <= rtol).item()
        and (
            arm != PRIMARY_ARM
            or (primary_residual_error <= atol + rtol * input_residual.abs().max()).item()
        )
    )
    if not moment_norm_passed:
        raise SourceSpectralBridgeError(
            "source spectral bridge failed its realized moment/norm audit"
        )
    active = rho > 0.0
    return BridgeDiagnostics(
        arm=arm,
        rho=rho,
        latent_shape=tuple(int(item) for item in gaussian.shape),
        dtype=str(gaussian.dtype),
        device=str(gaussian.device),
        source_only=True,
        paired_target_accessed=False,
        target_media_accessed=False,
        temporal_order_invariant_carrier=arm == PRIMARY_ARM,
        temporal_set_statistics=(
            "temporal_dc",
            "temporal_rms",
            "temporal_midrange",
        ),
        ordered_source_trajectory_injected=False,
        source_temporal_residual_injected=(
            active and arm == SHUFFLED_FRAME_ABLATION_ARM
        ),
        action_leakage_risk=arm == SHUFFLED_FRAME_ABLATION_ARM,
        non_gaussian_initial_noise=active,
        rho_zero_exact_gaussian_alias=(rho == 0.0 and output is gaussian),
        train_inference_same_contract=True,
        input_global_mean=_as_float_tuple(input_scalar.flatten(1)[:, 0]),
        output_global_mean=_as_float_tuple(output_scalar.flatten(1)[:, 0]),
        input_l2_norm=_as_float_tuple(input_total_norm),
        output_l2_norm=_as_float_tuple(output_total_norm),
        input_centered_dc_l2_norm=_as_float_tuple(input_dc_norm),
        output_centered_dc_l2_norm=_as_float_tuple(output_dc_norm),
        input_temporal_residual_l2_norm=_as_float_tuple(input_residual_norm),
        output_temporal_residual_l2_norm=_as_float_tuple(output_residual_norm),
        input_dc_residual_normalized_dot_max=_normalized_dot_max(
            input_dc_phase.expand_as(work_gaussian), input_residual
        ),
        output_dc_residual_normalized_dot_max=_normalized_dot_max(
            output_dc_phase.expand_as(work_output), output_residual
        ),
        global_mean_max_abs_error=float(global_mean_error.max().cpu().item()),
        total_norm_max_relative_error=float(total_relative_error.max().cpu().item()),
        centered_dc_norm_max_relative_error=float(dc_relative_error.max().cpu().item()),
        temporal_residual_norm_max_relative_error=float(
            residual_relative_error.max().cpu().item()
        ),
        primary_residual_max_abs_error=float(primary_residual_error.cpu().item()),
        carrier_base_normalized_dot_max=float(carrier_base_normalized_dot_max),
        shuffled_frame_permutation=permutation,
        moment_norm_audit_passed=True,
    )


def _build_receipt(diagnostics: BridgeDiagnostics) -> dict[str, Any]:
    active = diagnostics.rho > 0.0
    primary = diagnostics.arm == PRIMARY_ARM
    return {
        "schema_version": RECEIPT_SCHEMA,
        "method": "source_rich_initial_noise_ablation",
        "ablation_only": True,
        "scientific_claim_authorized": False,
        "pure_tensor_operator": True,
        "trainer_integration": False,
        "source_only": True,
        "paired_target_accessed": False,
        "non_gaussian_initial_noise": diagnostics.non_gaussian_initial_noise,
        "ordered_source_trajectory_injected": False,
        "action_leakage_risk": diagnostics.action_leakage_risk,
        "train_inference_same_contract": True,
        "arm": diagnostics.arm,
        "rho": diagnostics.rho,
        "geometry": {
            "video_frames": EXACT_VIDEO_FRAMES,
            "latent_frames": LATENT_FRAMES,
            "latent_channels": LATENT_CHANNELS,
            "latent_coordinate": LATENT_COORDINATE,
        },
        "inputs": {
            "accepted": ["gaussian", "source_normalized_latent", "rho"],
            "source_only_conditioning": True,
            "target_columns_accessed": [],
            "target_media_accessed": False,
            "paired_target_accessed": False,
            "text_or_action_plan_accessed": False,
        },
        "distribution": {
            "exact_gaussian_baseline": not active,
            "rho_zero_exact_input_alias": diagnostics.rho_zero_exact_gaussian_alias,
            "non_gaussian_initial_noise": diagnostics.non_gaussian_initial_noise,
            "source_conditioned_non_gaussian_when_active": True,
            "likelihood_or_density_claim": False,
        },
        "carrier": {
            "type": (
                "temporal_order_invariant_dc_set_statistics"
                if primary
                else "deterministic_shuffled_source_temporal_residual"
            ),
            "temporal_order_invariant": diagnostics.temporal_order_invariant_carrier,
            "set_statistic_weights": dict(SET_STATISTIC_WEIGHTS),
            "ordered_source_trajectory_injected": False,
            "source_temporal_residual_injected": diagnostics.source_temporal_residual_injected,
            "shuffled_frame_permutation": (
                list(diagnostics.shuffled_frame_permutation)
                if diagnostics.shuffled_frame_permutation is not None
                else None
            ),
            "action_leakage_risk": diagnostics.action_leakage_risk,
            "primary_arm_eligible": primary,
        },
        "mix": {
            "gaussian_decomposition": "orthogonal_temporal_dc_plus_residual",
            "scalar_gaussian_mean_preserved": True,
            "centered_temporal_dc_norm_preserved": True,
            "temporal_residual_norm_preserved": True,
            "total_realized_l2_norm_preserved": True,
            "rho_parameterization": "sqrt(1-rho^2)*gaussian+rho*orthogonal_source",
            "moment_norm_audited": diagnostics.moment_norm_audit_passed,
        },
        "train_inference_contract": {
            "same_operator": "source_spectral_bridge",
            "stage_dependent_branch": False,
            "same_contract": True,
        },
        "diagnostics": diagnostics.as_dict(),
    }


def source_spectral_bridge(
    gaussian: Any,
    source_normalized_latent: Any,
    *,
    rho: Real,
    arm: str = PRIMARY_ARM,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> SourceSpectralBridgeResult:
    """Construct one exact-81 source-rich initial-noise ablation.

    ``PRIMARY_ARM`` never changes the Gaussian temporal residual.  The shuffled
    arm is provided only for leakage diagnosis and is labelled unsafe in every
    returned receipt.  No arm accepts an ordered source carrier.
    """

    torch = _torch()
    gaussian, source = _validate_pair(gaussian, source_normalized_latent)
    rho_value = _validate_rho(rho)
    if arm not in SUPPORTED_ARMS:
        raise SourceSpectralBridgeError(
            f"arm must be one of {SUPPORTED_ARMS}; ordered source carriers are forbidden"
        )
    shuffle_seed = _validate_shuffle_seed(shuffle_seed)
    permutation = (
        deterministic_frame_permutation(seed=shuffle_seed)
        if arm == SHUFFLED_FRAME_ABLATION_ARM
        else None
    )
    if rho_value == 0.0:
        diagnostics = _make_diagnostics(
            gaussian=gaussian,
            output=gaussian,
            arm=arm,
            rho=rho_value,
            carrier_base_normalized_dot_max=0.0,
            permutation=permutation,
        )
        return SourceSpectralBridgeResult(
            initial_noise=gaussian,
            diagnostics=diagnostics,
            receipt=_build_receipt(diagnostics),
        )

    gaussian_work = gaussian.to(dtype=torch.float64)
    source_work = source.to(dtype=torch.float64)
    dc_phase = gaussian_work.mean(dim=2, keepdim=True)
    scalar_mean = dc_phase.mean(dim=(1, 2, 3, 4), keepdim=True)
    centered_dc = (dc_phase - scalar_mean).expand_as(gaussian_work)
    temporal_residual = gaussian_work - dc_phase.expand_as(gaussian_work)

    dc_carrier = temporal_set_carrier(source_work).to(dtype=torch.float64)
    mixed_dc, dc_audit = _orthogonal_norm_mix(
        centered_dc,
        dc_carrier,
        rho=rho_value,
        label="temporal-DC",
    )
    carrier_dot = float(dc_audit["base_carrier_normalized_dot_max"])
    output_residual = temporal_residual
    if arm == SHUFFLED_FRAME_ABLATION_ARM:
        shuffled_carrier = deterministic_shuffled_frame_carrier(
            source_work, seed=shuffle_seed
        ).to(dtype=torch.float64)
        output_residual, residual_audit = _orthogonal_norm_mix(
            temporal_residual,
            shuffled_carrier,
            rho=rho_value,
            label="temporal-residual",
        )
        carrier_dot = max(
            carrier_dot,
            float(residual_audit["base_carrier_normalized_dot_max"]),
        )

    output_work = scalar_mean.expand_as(gaussian_work) + mixed_dc + output_residual
    output = output_work.to(dtype=gaussian.dtype)
    if not bool(torch.isfinite(output).all().item()):
        raise SourceSpectralBridgeError("source spectral bridge produced non-finite noise")
    diagnostics = _make_diagnostics(
        gaussian=gaussian,
        output=output,
        arm=arm,
        rho=rho_value,
        carrier_base_normalized_dot_max=carrier_dot,
        permutation=permutation,
    )
    return SourceSpectralBridgeResult(
        initial_noise=output,
        diagnostics=diagnostics,
        receipt=_build_receipt(diagnostics),
    )


def bridge_initial_noise(
    gaussian: Any,
    source_normalized_latent: Any,
    *,
    rho: Real,
    arm: str = PRIMARY_ARM,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> SourceSpectralBridgeResult:
    """Descriptive alias for :func:`source_spectral_bridge`."""

    return source_spectral_bridge(
        gaussian,
        source_normalized_latent,
        rho=rho,
        arm=arm,
        shuffle_seed=shuffle_seed,
    )


__all__ = [
    "BridgeDiagnostics",
    "DEFAULT_SHUFFLE_SEED",
    "EXACT_VIDEO_FRAMES",
    "LATENT_CHANNELS",
    "LATENT_COORDINATE",
    "LATENT_FRAMES",
    "PRIMARY_ARM",
    "RECEIPT_SCHEMA",
    "SET_STATISTIC_WEIGHTS",
    "SHUFFLED_FRAME_ABLATION_ARM",
    "SUPPORTED_ARMS",
    "SourceSpectralBridgeError",
    "SourceSpectralBridgeResult",
    "bridge_initial_noise",
    "deterministic_frame_permutation",
    "deterministic_shuffled_frame_carrier",
    "source_spectral_bridge",
    "temporal_dc_residual",
    "temporal_set_carrier",
]

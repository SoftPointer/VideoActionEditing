#!/usr/bin/env python3
"""Pure BRAID prior-space source-carrier mathematics.

This module implements only the temporal-band operation proposed in section
4.5 of the BRAID design.  Given a *caller supplied* source-prior tensor
``eta_source`` and an official-Gaussian tensor ``epsilon`` with the same
exact81 Bernini latent geometry, it computes

``P_<=k eta_source + rho P_>k eta_source + sqrt(1-rho^2) P_>k epsilon``.

The temporal projections use one orthonormal DCT-II basis.  Configuration is
selected from a closed, prospective registry; callers cannot choose arbitrary
``k`` or ``rho`` after observing results.  Training and inference are aliases
of the exact same tensor function and there is no stage-dependent branch.

This is deliberately only a local mathematical primitive.  It does not run or
authenticate source inversion, source round-trip, prior-statistics, or action
gates.  It accepts no media, owner, prompt, model, scheduler, mask, pose, flow,
or trajectory input.  It does not construct an optimizer, run backward,
update parameters, decode video, or save a checkpoint.  Its receipt therefore
has ``math_primitive_only/no_authority`` classification and cannot authorize
training or make a scientific claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import sys
from types import MappingProxyType
from typing import Any, Mapping

import torch


METHOD = "bernini-braid-prior-space-source-carrier-v1"
SCHEMA_VERSION = "bernini-braid-prior-space-source-carrier-math-v1"
CONFIG_SCHEMA_VERSION = "bernini-braid-prior-space-source-carrier-config-v1"
CLASSIFICATION = "math_primitive_only/no_authority"

NUM_FRAMES = 81
LATENT_CHANNELS = 16
LATENT_PHASES = 21
TEMPORAL_AXIS = 2
LOW_FREQUENCY_MAX_MODE = 3

PRIMARY_ARM_ID = "dct-low4-high-mix-rho0p5"
RHO_ZERO_CONTROL_ARM_ID = "dct-low4-high-official-rho0"
RHO_ONE_CONTROL_ARM_ID = "dct-low4-source-prior-rho1"

_PREREGISTERED_VALUES = MappingProxyType(
    {
        PRIMARY_ARM_ID: (LOW_FREQUENCY_MAX_MODE, 0.5, "prospective_primary"),
        RHO_ZERO_CONTROL_ARM_ID: (
            LOW_FREQUENCY_MAX_MODE,
            0.0,
            "full_high_frequency_resample_control",
        ),
        RHO_ONE_CONTROL_ARM_ID: (
            LOW_FREQUENCY_MAX_MODE,
            1.0,
            "source_prior_identity_control",
        ),
    }
)
PREREGISTERED_ARM_IDS = tuple(_PREREGISTERED_VALUES)


class BraidPriorSpaceCarrierError(RuntimeError):
    """The closed prior-carrier tensor or authority contract was violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    """Return a deterministic finite ASCII representation for receipts."""

    def validate(item: Any, *, path: str) -> None:
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise BraidPriorSpaceCarrierError(
                    f"{path} contains a non-finite float"
                )
            return
        if type(item) is list:
            for index, child in enumerate(item):
                validate(child, path=f"{path}[{index}]")
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise BraidPriorSpaceCarrierError(
                        f"{path} contains a non-string key"
                    )
                validate(child, path=f"{path}.{key}")
            return
        raise BraidPriorSpaceCarrierError(
            f"{path} contains unsupported type {type(item).__name__}"
        )

    validate(value, path="receipt")
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise BraidPriorSpaceCarrierError(
            "receipt is not canonical finite ASCII JSON"
        ) from error


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class PriorSpaceSourceCarrierConfig:
    """One exact entry in the prospective ``(k, rho)`` registry."""

    arm_id: str
    num_frames: int
    latent_phases: int
    low_frequency_max_mode: int
    rho: float
    role: str

    def validate(self) -> None:
        if type(self) is not PriorSpaceSourceCarrierConfig:
            raise BraidPriorSpaceCarrierError("config must have exact registered type")
        if type(self.arm_id) is not str:
            raise BraidPriorSpaceCarrierError("carrier arm is not preregistered")
        expected = _PREREGISTERED_VALUES.get(self.arm_id)
        if expected is None:
            raise BraidPriorSpaceCarrierError("carrier arm is not preregistered")
        expected_k, expected_rho, expected_role = expected
        if (
            type(self.arm_id) is not str
            or type(self.num_frames) is not int
            or type(self.latent_phases) is not int
            or type(self.low_frequency_max_mode) is not int
            or type(self.rho) is not float
            or type(self.role) is not str
            or self.num_frames != NUM_FRAMES
            or self.latent_phases != LATENT_PHASES
            or self.low_frequency_max_mode != expected_k
            or self.rho != expected_rho
            or self.role != expected_role
            or not 0 <= self.low_frequency_max_mode < LATENT_PHASES - 1
            or not 0.0 <= self.rho <= 1.0
        ):
            raise BraidPriorSpaceCarrierError(
                "carrier config differs from its preregistered coordinate"
            )

    def receipt(self) -> Mapping[str, Any]:
        self.validate()
        unsigned = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "role": self.role,
            "num_frames": self.num_frames,
            "latent_phases": self.latent_phases,
            "temporal_axis": TEMPORAL_AXIS,
            "dct_convention": "orthonormal_DCT_II",
            "low_frequency_max_mode_inclusive": self.low_frequency_max_mode,
            "low_frequency_modes": list(
                range(self.low_frequency_max_mode + 1)
            ),
            "high_frequency_modes": list(
                range(self.low_frequency_max_mode + 1, LATENT_PHASES)
            ),
            "rho": self.rho,
            "registry_coordinate_embedded_in_source": True,
            "decoded_result_input_accepted": False,
            "caller_selectable_k_or_rho": False,
        }
        return {**unsigned, "digest": _object_sha256(unsigned)}


def preregistered_config(arm_id: str = PRIMARY_ARM_ID) -> PriorSpaceSourceCarrierConfig:
    """Resolve one arm without accepting a caller-provided ``k`` or ``rho``."""

    if type(arm_id) is not str or arm_id not in _PREREGISTERED_VALUES:
        raise BraidPriorSpaceCarrierError("carrier arm is not preregistered")
    k, rho, role = _PREREGISTERED_VALUES[arm_id]
    config = PriorSpaceSourceCarrierConfig(
        arm_id=arm_id,
        num_frames=NUM_FRAMES,
        latent_phases=LATENT_PHASES,
        low_frequency_max_mode=k,
        rho=rho,
        role=role,
    )
    config.validate()
    return config


def build_temporal_orthonormal_dct_basis() -> torch.Tensor:
    """Return the fixed float64 ``[21,21]`` orthonormal DCT-II basis."""

    positions = torch.arange(LATENT_PHASES, dtype=torch.float64).add_(0.5)
    modes = torch.arange(LATENT_PHASES, dtype=torch.float64)
    basis = torch.cos(
        (math.pi / float(LATENT_PHASES)) * positions[:, None] * modes[None, :]
    )
    basis[:, 0].mul_(1.0 / math.sqrt(float(LATENT_PHASES)))
    basis[:, 1:].mul_(math.sqrt(2.0 / float(LATENT_PHASES)))
    basis = basis.contiguous()
    gram = basis.T @ basis
    if not torch.allclose(
        gram,
        torch.eye(LATENT_PHASES, dtype=torch.float64),
        rtol=0.0,
        atol=2.0e-14,
    ):
        raise RuntimeError("fixed temporal DCT-II basis is not orthonormal")
    return basis


def _storage_pointer(value: torch.Tensor) -> int:
    try:
        return int(value.untyped_storage().data_ptr())
    except AttributeError:  # PyTorch 1.12 compatibility
        return int(value.storage().data_ptr())


def _validate_latent(value: Any, *, label: str) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise BraidPriorSpaceCarrierError(
            f"{label} must be an exact-type torch.Tensor"
        )
    if (
        value.layout != torch.strided
        or value.dtype != torch.float32
        or value.ndim != 5
        or int(value.shape[0]) <= 0
        or int(value.shape[1]) != LATENT_CHANNELS
        or int(value.shape[2]) != LATENT_PHASES
        or int(value.shape[3]) <= 0
        or int(value.shape[4]) <= 0
        or not value.is_contiguous()
        or int(value.storage_offset()) != 0
        or getattr(value, "_base", None) is not None
    ):
        raise BraidPriorSpaceCarrierError(
            f"{label} must be an owned contiguous FP32 "
            f"[B,{LATENT_CHANNELS},{LATENT_PHASES},H,W] exact81 latent"
        )
    if value.requires_grad or value.grad_fn is not None or value.grad is not None:
        raise BraidPriorSpaceCarrierError(f"{label} must be detached and grad-free")
    if not bool(torch.isfinite(value).all().item()):
        raise BraidPriorSpaceCarrierError(f"{label} must be finite")
    return value


def _validate_pair(
    eta_source: Any, official_gaussian: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    source = _validate_latent(eta_source, label="eta_source")
    gaussian = _validate_latent(official_gaussian, label="official_gaussian")
    if (
        tuple(source.shape) != tuple(gaussian.shape)
        or source.dtype != gaussian.dtype
        or source.device != gaussian.device
    ):
        raise BraidPriorSpaceCarrierError(
            "eta_source and official_gaussian must share shape/dtype/device"
        )
    if source is gaussian or _storage_pointer(source) == _storage_pointer(gaussian):
        raise BraidPriorSpaceCarrierError(
            "eta_source and official_gaussian must not alias storage"
        )
    return source, gaussian


def _split_work(
    value_fp64: torch.Tensor,
    *,
    basis: torch.Tensor,
    low_frequency_max_mode: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = torch.einsum("tm,bcthw->bcmhw", basis, value_fp64)
    low_stop = low_frequency_max_mode + 1
    low = torch.einsum(
        "tq,bcqhw->bcthw",
        basis[:, :low_stop],
        coefficients[:, :, :low_stop],
    )
    high = torch.einsum(
        "tq,bcqhw->bcthw",
        basis[:, low_stop:],
        coefficients[:, :, low_stop:],
    )
    return low.contiguous(), high.contiguous()


@dataclass(frozen=True)
class TemporalDCTBands:
    """Caller-owned low/high tensors from one registered DCT split."""

    low: torch.Tensor
    high: torch.Tensor


def temporal_dct_split(
    value: Any,
    *,
    config: PriorSpaceSourceCarrierConfig,
) -> TemporalDCTBands:
    """Split one exact81 latent into registered orthogonal temporal bands."""

    if type(config) is not PriorSpaceSourceCarrierConfig:
        raise BraidPriorSpaceCarrierError("config must have exact registered type")
    config.validate()
    checked = _validate_latent(value, label="latent")
    owned = checked.detach().clone(memory_format=torch.contiguous_format)
    basis = build_temporal_orthonormal_dct_basis().to(device=owned.device)
    with torch.no_grad():
        low, high = _split_work(
            owned.to(dtype=torch.float64),
            basis=basis,
            low_frequency_max_mode=config.low_frequency_max_mode,
        )
        low = low.to(dtype=torch.float32).contiguous()
        high = high.to(dtype=torch.float32).contiguous()
    if not torch.equal(checked, owned):
        raise BraidPriorSpaceCarrierError("latent input changed during DCT split")
    return TemporalDCTBands(low=low, high=high)


@dataclass(frozen=True)
class PriorSpaceSourceCarrierResult:
    """Caller-owned prior noise and an explicitly non-authoritative receipt."""

    initial_noise: torch.Tensor
    receipt: Mapping[str, Any]


def _band_audit(
    original: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> dict[str, float]:
    reconstruction = low + high
    original_energy = original.square().sum()
    split_energy = low.square().sum() + high.square().sum()
    return {
        "reconstruction_max_abs": float(
            (reconstruction - original).abs().max().detach().cpu().item()
        ),
        "low_high_inner_product_abs": float(
            (low * high).sum().abs().detach().cpu().item()
        ),
        "energy_decomposition_abs_error": float(
            (split_energy - original_energy).abs().detach().cpu().item()
        ),
    }


def _basis_sha256() -> str:
    basis = build_temporal_orthonormal_dct_basis()
    header = _canonical_json_bytes(
        {
            "dtype": str(basis.dtype),
            "shape": list(map(int, basis.shape)),
            "element_endianness": sys.byteorder,
            "storage_order": "C_contiguous",
        }
    )
    # This basis is only 21 x 21 float64 values.  Materialize its canonical
    # native-endian storage bytes without crossing the NumPy ABI boundary.
    raw = bytes(basis.contiguous().view(torch.uint8).flatten().tolist())
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


def build_prior_space_source_carrier(
    eta_source: Any,
    official_gaussian: Any,
    *,
    config: PriorSpaceSourceCarrierConfig,
) -> PriorSpaceSourceCarrierResult:
    """Apply the sole train/inference prior-carrier tensor function.

    ``eta_source`` is not authenticated as an inversion product here.  Any
    runtime that needs that claim must establish inversion, round-trip, prior
    statistics, and action gates outside this primitive before using it.
    """

    if type(config) is not PriorSpaceSourceCarrierConfig:
        raise BraidPriorSpaceCarrierError("config must have exact registered type")
    config.validate()
    source, gaussian = _validate_pair(eta_source, official_gaussian)
    source_owned = source.detach().clone(memory_format=torch.contiguous_format)
    gaussian_owned = gaussian.detach().clone(memory_format=torch.contiguous_format)
    basis = build_temporal_orthonormal_dct_basis().to(device=source.device)
    with torch.no_grad():
        source_work = source_owned.to(dtype=torch.float64)
        gaussian_work = gaussian_owned.to(dtype=torch.float64)
        source_low, source_high = _split_work(
            source_work,
            basis=basis,
            low_frequency_max_mode=config.low_frequency_max_mode,
        )
        gaussian_low, gaussian_high = _split_work(
            gaussian_work,
            basis=basis,
            low_frequency_max_mode=config.low_frequency_max_mode,
        )
        if config.rho == 1.0:
            initial_noise = source_owned.clone()
        else:
            gaussian_weight = math.sqrt(max(0.0, 1.0 - config.rho**2))
            initial_noise = (
                source_low
                + config.rho * source_high
                + gaussian_weight * gaussian_high
            ).to(dtype=torch.float32).contiguous()

    if not torch.equal(source, source_owned) or not torch.equal(
        gaussian, gaussian_owned
    ):
        raise BraidPriorSpaceCarrierError(
            "caller input changed during prior-carrier construction"
        )
    if (
        type(initial_noise) is not torch.Tensor
        or tuple(initial_noise.shape) != tuple(source.shape)
        or initial_noise.dtype != torch.float32
        or initial_noise.device != source.device
        or initial_noise.requires_grad
        or initial_noise.grad_fn is not None
        or not initial_noise.is_contiguous()
        or not bool(torch.isfinite(initial_noise).all().item())
        or _storage_pointer(initial_noise)
        in {_storage_pointer(source), _storage_pointer(gaussian)}
    ):
        raise BraidPriorSpaceCarrierError(
            "constructed prior carrier is not a fresh finite detached tensor"
        )

    numerical_audit = {
        "eta_source": _band_audit(source_work, source_low, source_high),
        "official_gaussian": _band_audit(
            gaussian_work, gaussian_low, gaussian_high
        ),
    }
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "classification": CLASSIFICATION,
        "config": dict(config.receipt()),
        "latent_shape": list(map(int, source.shape)),
        "dtype": str(source.dtype),
        "device_type": source.device.type,
        "temporal_basis_sha256": _basis_sha256(),
        "operator": {
            "equation": (
                "P_<=k(eta_source)+rho*P_>k(eta_source)+"
                "sqrt(1-rho^2)*P_>k(official_gaussian)"
            ),
            "low_source_band_preserved_directly": True,
            "official_gaussian_low_band_consumed": False,
            "source_high_band_coefficient": config.rho,
            "official_gaussian_high_band_coefficient": math.sqrt(
                max(0.0, 1.0 - config.rho**2)
            ),
            "both_high_bands_active": 0.0 < config.rho < 1.0,
            "orthonormal_temporal_dct_split": True,
        },
        "numerical_audit": numerical_audit,
        "condition_contract": {
            "training_entrypoint": "build_prior_space_source_carrier",
            "inference_entrypoint": "build_prior_space_source_carrier",
            "training_inference_same_function_object": True,
            "stage_flag_accepted": False,
            "training_only_condition_present": False,
        },
        "upstream_claims": {
            "eta_source_is_caller_supplied_tensor_only": True,
            "eta_source_inversion_executed_by_this_primitive": False,
            "eta_source_inversion_authenticated": False,
            "official_gaussian_is_caller_supplied_tensor_only": True,
            "official_gaussian_provenance_authenticated": False,
            "source_roundtrip_executed": False,
            "source_roundtrip_passed": None,
            "prior_statistics_gate_executed": False,
            "prior_statistics_gate_passed": None,
            "action_gate_executed": False,
            "action_gate_passed": None,
        },
        "side_effects_and_authority": {
            "media_read": False,
            "owner_read": False,
            "model_loaded_or_forwarded": False,
            "scheduler_executed": False,
            "video_decoded": False,
            "optimizer_created": False,
            "backward_executed": False,
            "parameter_update_performed": False,
            "checkpoint_or_tensor_saved": False,
            "training_authorized": False,
            "scientific_authority": False,
        },
    }
    receipt = {**unsigned, "receipt_digest": _object_sha256(unsigned)}
    return PriorSpaceSourceCarrierResult(
        initial_noise=initial_noise,
        receipt=receipt,
    )


# These are aliases, not wrappers: neither usage can acquire a hidden branch or
# a training-only condition without changing the function object itself.
training_source_carrier = build_prior_space_source_carrier
inference_source_carrier = build_prior_space_source_carrier


__all__ = [
    "BraidPriorSpaceCarrierError",
    "CLASSIFICATION",
    "CONFIG_SCHEMA_VERSION",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "LOW_FREQUENCY_MAX_MODE",
    "METHOD",
    "NUM_FRAMES",
    "PREREGISTERED_ARM_IDS",
    "PRIMARY_ARM_ID",
    "PriorSpaceSourceCarrierConfig",
    "PriorSpaceSourceCarrierResult",
    "RHO_ONE_CONTROL_ARM_ID",
    "RHO_ZERO_CONTROL_ARM_ID",
    "SCHEMA_VERSION",
    "TEMPORAL_AXIS",
    "TemporalDCTBands",
    "build_prior_space_source_carrier",
    "build_temporal_orthonormal_dct_basis",
    "inference_source_carrier",
    "preregistered_config",
    "temporal_dct_split",
    "training_source_carrier",
]

#!/usr/bin/env python3
"""Scheduler-boundary wrapper for fixed-grid robust camera consensus.

For exact FP32 Bernini clean fields ``S`` and ``Xa`` with layout
``[B,C,21,H,W]``, execution is

``Xexec = Xa + beta * C_consensus(S - Xa)``.

``C_consensus`` is the per-phase fixed-grid robust estimator implemented in
``fixed_grid_camera_consensus``.  It is not an orthogonal projector and this
wrapper makes no orthogonal-complement invariant claim.  A phase without a
strict cross-tile consensus is selected directly from ``Xa`` and is therefore
bitwise unchanged.  An all-zero beta bypass returns the original action tensor
object before camera geometry is built or validated.

The runtime surface accepts only the source field, action field, beta,
numerical config, and an optional source-derived precomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:  # Support both namespace-package and method-root imports.
    from . import fixed_grid_camera_consensus as camera_core
except ImportError:  # pragma: no cover - selected by standalone method runners
    import fixed_grid_camera_consensus as camera_core


METHOD_NAME = "fixed-grid-camera-consensus-stabilizer"
SCHEMA_VERSION = "bernini-fixed-grid-camera-consensus-stabilizer-v1"
EXPECTED_LATENT_PHASES = camera_core.EXPECTED_LATENT_PHASES

CameraConsensusConfig = camera_core.CameraConsensusConfig
CameraConsensusError = camera_core.CameraConsensusError
CameraConsensusProjection = camera_core.CameraConsensusProjection
FixedGridCameraGeometry = camera_core.FixedGridCameraGeometry


@dataclass(frozen=True)
class CameraConsensusStabilizationTrace:
    """Auditable evidence for one scheduler-boundary stabilization call."""

    schema_version: str
    method: str
    bypassed: bool
    bypass_reason: str | None
    beta_mode: str
    beta_per_phase: Any
    geometry_built: bool
    geometry_reused: bool
    estimator: str
    consensus_scope: str
    consensus_valid: Any
    correction_rms: Any
    geometry_valid_tile_count: Any
    fit_valid_tile_count: Any
    inlier_tile_count: Any
    spatial_coverage_valid: Any
    consensus_coefficient_max_abs: Any
    tile_relative_fit_residual_max: Any
    invalid_phases_exact_action: bool

    def to_receipt(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": self.schema_version,
                "method": self.method,
                "bypassed": self.bypassed,
                "bypass_reason": self.bypass_reason,
                "beta_mode": self.beta_mode,
                "beta_per_phase": self.beta_per_phase,
                "geometry_built": self.geometry_built,
                "geometry_reused": self.geometry_reused,
                "estimator": self.estimator,
                "consensus_scope": self.consensus_scope,
                "consensus_valid": self.consensus_valid,
                "correction_rms": self.correction_rms,
                "geometry_valid_tile_count": self.geometry_valid_tile_count,
                "fit_valid_tile_count": self.fit_valid_tile_count,
                "inlier_tile_count": self.inlier_tile_count,
                "spatial_coverage_valid": self.spatial_coverage_valid,
                "consensus_coefficient_max_abs": (
                    self.consensus_coefficient_max_abs
                ),
                "tile_relative_fit_residual_max": (
                    self.tile_relative_fit_residual_max
                ),
                "invalid_phases_exact_action": self.invalid_phases_exact_action,
            }
        )


@dataclass(frozen=True)
class CameraConsensusStabilizationResult:
    """Executed clean field, optional estimator output, and trace."""

    executed_clean_field: Any
    projection: CameraConsensusProjection | None
    trace: CameraConsensusStabilizationTrace


def camera_consensus_stabilizer_contract_receipt() -> dict[str, Any]:
    """Describe the scheduler-boundary contract without adding side inputs."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "tensor_contract": {
            "layout": "B,C,T,H,W",
            "dtype": "float32",
            "latent_phases": EXPECTED_LATENT_PHASES,
            "branches": ["source_clean_field", "action_clean_field"],
        },
        "execution": "Xa+beta*C_consensus(S-Xa)",
        "estimator": "fixed_grid_median_MAD_trimmed_robust_consensus",
        "consensus_scope": "independent_per_batch_and_latent_phase",
        "runtime_inputs": [
            "source_clean_field",
            "action_clean_field",
            "beta",
            "config",
            "precomputed_geometry",
        ],
        "zero_beta": "original_action_object_passthrough_without_geometry",
        "invalid_phase": "exact_action_value_passthrough",
        "geometry": "build_from_exact_source_or_strictly_reuse_source_geometry",
        "diagnostics": [
            "geometry_valid_tile_count",
            "fit_valid_tile_count",
            "inlier_tile_count",
            "spatial_coverage_valid",
            "consensus_coefficient_max_abs",
            "tile_relative_fit_residual_max",
        ],
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    candidate = value.detach() if hasattr(value, "detach") else value
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "tolist"):
        return _json_safe(candidate.tolist())
    if hasattr(candidate, "item"):
        return _json_safe(candidate.item())
    raise CameraConsensusError("trace contains a non-serializable value")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise CameraConsensusError(
            "camera-consensus stabilization requires PyTorch"
        ) from error
    return torch


def _validate_field(field: Any, *, label: str) -> None:
    torch = _require_torch()
    if not isinstance(field, torch.Tensor):
        raise CameraConsensusError(f"{label} must be a torch tensor")
    if (
        field.ndim != 5
        or int(field.shape[0]) <= 0
        or int(field.shape[1]) <= 0
        or int(field.shape[2]) != EXPECTED_LATENT_PHASES
        or int(field.shape[3]) <= 0
        or int(field.shape[4]) <= 0
        or field.dtype != torch.float32
    ):
        raise CameraConsensusError(f"{label} must be exact float32 [B,C,21,H,W]")
    if not bool(torch.isfinite(field).all()):
        raise CameraConsensusError(f"{label} contains non-finite values")


def _validate_pair(source_clean_field: Any, action_clean_field: Any) -> None:
    _validate_field(source_clean_field, label="source_clean_field")
    _validate_field(action_clean_field, label="action_clean_field")
    if tuple(source_clean_field.shape) != tuple(action_clean_field.shape):
        raise CameraConsensusError("source and action clean-field shapes differ")
    if source_clean_field.device != action_clean_field.device:
        raise CameraConsensusError("source and action clean-field devices differ")


def _canonical_beta(beta: Any, reference: Any) -> tuple[Any, str]:
    torch = _require_torch()
    if isinstance(beta, bool):
        raise CameraConsensusError("beta must be finite scalar or per-phase values")
    try:
        value = torch.as_tensor(
            beta, dtype=torch.float32, device=reference.device
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise CameraConsensusError(
            "beta must be finite scalar or per-phase values"
        ) from error
    batch = int(reference.shape[0])
    if value.ndim == 0:
        mode = "scalar"
        value = value.reshape(1, 1).expand(batch, EXPECTED_LATENT_PHASES)
    elif tuple(value.shape) == (EXPECTED_LATENT_PHASES,):
        mode = "shared_per_phase"
        value = value.reshape(1, EXPECTED_LATENT_PHASES).expand(batch, -1)
    elif tuple(value.shape) == (1, EXPECTED_LATENT_PHASES):
        mode = "shared_per_phase"
        value = value.expand(batch, -1)
    elif tuple(value.shape) == (batch, EXPECTED_LATENT_PHASES):
        mode = "per_batch_phase"
    else:
        raise CameraConsensusError(
            "beta must be scalar, [21], [1,21], or [B,21]"
        )
    if not bool(torch.isfinite(value).all()):
        raise CameraConsensusError("beta contains non-finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise CameraConsensusError("beta must be in the closed interval [0,1]")
    return value.detach(), mode


def _phase_rms(value: Any) -> Any:
    return value.square().mean(dim=(1, 3, 4)).sqrt()


def stabilize_camera_consensus(
    source_clean_field: Any,
    action_clean_field: Any,
    *,
    beta: Any = 1.0,
    config: CameraConsensusConfig = CameraConsensusConfig(),
    precomputed_geometry: FixedGridCameraGeometry | None = None,
) -> CameraConsensusStabilizationResult:
    """Apply robust source-minus-action camera consensus at the scheduler edge."""

    torch = _require_torch()
    if type(config) is not CameraConsensusConfig:
        raise CameraConsensusError("config must be an exact CameraConsensusConfig")
    config.validate()
    _validate_pair(source_clean_field, action_clean_field)
    beta_per_phase, beta_mode = _canonical_beta(beta, action_clean_field)

    if bool(torch.equal(beta_per_phase, torch.zeros_like(beta_per_phase))):
        return CameraConsensusStabilizationResult(
            executed_clean_field=action_clean_field,
            projection=None,
            trace=CameraConsensusStabilizationTrace(
                schema_version=SCHEMA_VERSION,
                method=METHOD_NAME,
                bypassed=True,
                bypass_reason="zero_beta",
                beta_mode=beta_mode,
                beta_per_phase=beta_per_phase,
                geometry_built=False,
                geometry_reused=False,
                estimator="fixed_grid_median_MAD_trimmed_robust_consensus",
                consensus_scope="independent_per_batch_and_latent_phase",
                consensus_valid=None,
                correction_rms=None,
                geometry_valid_tile_count=None,
                fit_valid_tile_count=None,
                inlier_tile_count=None,
                spatial_coverage_valid=None,
                consensus_coefficient_max_abs=None,
                tile_relative_fit_residual_max=None,
                invalid_phases_exact_action=True,
            ),
        )

    residual = source_clean_field.detach() - action_clean_field
    projection = camera_core.project_camera_consensus(
        residual,
        source_clean_field,
        config=config,
        precomputed_geometry=precomputed_geometry,
    )
    beta_field = beta_per_phase[:, None, :, None, None]
    correction = beta_field * projection.camera_component
    active_phase = projection.consensus_valid & (beta_per_phase != 0.0)
    active_field = active_phase[:, None, :, None, None]
    candidate = action_clean_field + correction
    executed = torch.where(active_field, candidate, action_clean_field)

    invalid = ~active_phase
    invalid_executed = executed.permute(0, 2, 1, 3, 4)[invalid]
    invalid_action = action_clean_field.permute(0, 2, 1, 3, 4)[invalid]
    invalid_exact = bool(torch.equal(invalid_executed, invalid_action))
    if not invalid_exact:
        raise CameraConsensusError("non-consensus phase changed at scheduler boundary")

    return CameraConsensusStabilizationResult(
        executed_clean_field=executed,
        projection=projection,
        trace=CameraConsensusStabilizationTrace(
            schema_version=SCHEMA_VERSION,
            method=METHOD_NAME,
            bypassed=False,
            bypass_reason=None,
            beta_mode=beta_mode,
            beta_per_phase=beta_per_phase,
            geometry_built=precomputed_geometry is None,
            geometry_reused=precomputed_geometry is not None,
            estimator="fixed_grid_median_MAD_trimmed_robust_consensus",
            consensus_scope="independent_per_batch_and_latent_phase",
            consensus_valid=projection.consensus_valid.detach(),
            correction_rms=_phase_rms(correction).detach(),
            geometry_valid_tile_count=(
                projection.geometry_valid_tiles.sum(dim=-1).detach()
            ),
            fit_valid_tile_count=(
                projection.fit_valid_tiles.sum(dim=-1).detach()
            ),
            inlier_tile_count=projection.inlier_tiles.sum(dim=-1).detach(),
            spatial_coverage_valid=(
                projection.spatial_coverage_valid.detach()
            ),
            consensus_coefficient_max_abs=(
                projection.consensus_coefficients.abs().amax(dim=-1).detach()
            ),
            tile_relative_fit_residual_max=(
                projection.per_tile_relative_fit_residual.amax(dim=-1).detach()
            ),
            invalid_phases_exact_action=invalid_exact,
        ),
    )


__all__ = [
    "CameraConsensusConfig",
    "CameraConsensusError",
    "CameraConsensusStabilizationResult",
    "CameraConsensusStabilizationTrace",
    "FixedGridCameraGeometry",
    "camera_consensus_stabilizer_contract_receipt",
    "stabilize_camera_consensus",
]

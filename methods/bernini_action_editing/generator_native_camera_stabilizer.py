#!/usr/bin/env python3
"""Generator-native camera-tangent stabilization for Bernini clean fields.

The operator consumes three same-state FP32 clean fields with layout
``[B,C,21,H,W]``.  For every batch item and latent phase independently, it
linearizes an eight-parameter homography at the identity using spatial
gradients of the *complete detached source field*.  An FP32 thin SVD turns
those eight tangent images into a rank- and condition-truncated orthonormal
camera subspace.

Writing ``P`` for that phasewise projector, execution is exactly

``X_exec = X_action + beta * P(S-X_action)``.

The implementation retains the algebraically equivalent tri-branch form
``P(S-X_noop)-P(X_action-X_noop)`` so the official action/no-op/APG boundary
remains auditable.  Because ``P`` is linear, the no-op field cancels from the
actual correction direction.

The correction therefore lives entirely in the source-derived camera
subspace.  The complementary, generally non-rigid action component is
preserved; the implementation verifies this invariant numerically before it
returns.  No external spatial annotation is accepted by any runtime API.

PyTorch is imported only when a tensor operation is invoked, leaving the
configuration and contract inspectable on lightweight orchestration hosts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


METHOD_NAME = "generator-native-camera-tangent-stabilizer"
SCHEMA_VERSION = "bernini-camera-tangent-trace-v1"
EXPECTED_LATENT_PHASES = 21
HOMOGRAPHY_DOF = 8

DEFAULT_RELATIVE_RANK_CUTOFF = 1.0e-5
DEFAULT_ABSOLUTE_RANK_CUTOFF = 1.0e-7
DEFAULT_MAX_CONDITION_NUMBER = 1.0e5
DEFAULT_MINIMUM_RANK = 1
DEFAULT_INVARIANCE_ATOL = 4.0e-5
DEFAULT_INVARIANCE_RTOL = 4.0e-5


class CameraTangentError(RuntimeError):
    """Raised before an invalid camera-tangent operation reaches a scheduler."""


@dataclass(frozen=True)
class CameraTangentConfig:
    """Numerical policy for the phasewise source camera subspace."""

    relative_rank_cutoff: float = DEFAULT_RELATIVE_RANK_CUTOFF
    absolute_rank_cutoff: float = DEFAULT_ABSOLUTE_RANK_CUTOFF
    max_condition_number: float = DEFAULT_MAX_CONDITION_NUMBER
    minimum_rank: int = DEFAULT_MINIMUM_RANK
    invariance_atol: float = DEFAULT_INVARIANCE_ATOL
    invariance_rtol: float = DEFAULT_INVARIANCE_RTOL

    def validate(self) -> None:
        for name in (
            "relative_rank_cutoff",
            "absolute_rank_cutoff",
            "invariance_atol",
            "invariance_rtol",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise CameraTangentError(f"{name} must be finite and nonnegative")
        value = self.max_condition_number
        if (
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 1.0
        ):
            raise CameraTangentError(
                "max_condition_number must be finite and greater than one"
            )
        if (
            type(self.minimum_rank) is not int
            or not 1 <= self.minimum_rank <= HOMOGRAPHY_DOF
        ):
            raise CameraTangentError("minimum_rank must be an integer in [1,8]")


@dataclass(frozen=True)
class CameraTangentBasis:
    """An auditable orthonormal camera basis for every ``(B,T)`` cell."""

    # Shape [B,T,C*H*W,8].  Columns rejected by the cutoff are exact zero.
    orthonormal_vectors: Any
    singular_values: Any
    retained_modes: Any
    retained_rank: Any
    condition_number: Any
    valid_phase: Any
    cutoff: Any
    source_shape: tuple[int, int, int, int, int]
    source_device: Any
    source_dtype: Any
    source_data_ptr: int
    source_version: int
    config_signature: tuple[float | int, ...]
    source_was_detached: bool
    # Keep the precise source object alive: cache reuse is deliberately bound
    # to object identity, not merely equal values in a replacement tensor.
    source_identity: Any = field(repr=False, compare=False)


def _json_safe(value: Any) -> Any:
    """Convert trace values to finite, JSON-compatible Python objects."""

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
    raise CameraTangentError("trace contains a value that cannot be serialized")


@dataclass(frozen=True)
class CameraStabilizationTrace:
    """Explicit evidence for one stabilization call."""

    schema_version: str
    method: str
    bypassed: bool
    bypass_reason: str | None
    beta_mode: str
    beta_per_phase: Any
    basis_built: bool
    basis_reused: bool
    source_basis_detached: bool
    retained_rank: Any
    condition_number: Any
    valid_phase: Any
    camera_component_rms_before: Any
    camera_component_rms_after: Any
    source_camera_component_rms: Any
    applied_correction_rms: Any
    noncamera_invariance_max_abs: float
    noncamera_invariance_rms: float
    noncamera_invariance_tolerance: float
    invariant_satisfied: bool

    def to_receipt(self) -> dict[str, Any]:
        """Return a JSON-safe trace without adding any side-input schema."""

        return _json_safe(
            {
                "schema_version": self.schema_version,
                "method": self.method,
                "bypassed": self.bypassed,
                "bypass_reason": self.bypass_reason,
                "beta_mode": self.beta_mode,
                "beta_per_phase": self.beta_per_phase,
                "basis_built": self.basis_built,
                "basis_reused": self.basis_reused,
                "source_basis_detached": self.source_basis_detached,
                "retained_rank": self.retained_rank,
                "condition_number": self.condition_number,
                "valid_phase": self.valid_phase,
                "camera_component_rms_before": self.camera_component_rms_before,
                "camera_component_rms_after": self.camera_component_rms_after,
                "source_camera_component_rms": self.source_camera_component_rms,
                "applied_correction_rms": self.applied_correction_rms,
                "noncamera_invariance_max_abs": (
                    self.noncamera_invariance_max_abs
                ),
                "noncamera_invariance_rms": self.noncamera_invariance_rms,
                "noncamera_invariance_tolerance": (
                    self.noncamera_invariance_tolerance
                ),
                "invariant_satisfied": self.invariant_satisfied,
            }
        )


@dataclass(frozen=True)
class CameraStabilizationResult:
    """Scheduler-boundary clean field and its numerical trace."""

    executed_clean_field: Any
    trace: CameraStabilizationTrace


def camera_stabilizer_contract_receipt() -> dict[str, Any]:
    """Describe the immutable source/action/no-op tensor contract."""

    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "tensor_contract": {
            "layout": "B,C,T,H,W",
            "dtype": "float32",
            "latent_phases": EXPECTED_LATENT_PHASES,
            "branches": [
                "source_clean_field",
                "action_clean_field",
                "noop_clean_field",
            ],
        },
        "basis": {
            "source_policy": "complete_detached_source_each_phase",
            "coordinates": "normalized_minus_one_to_one",
            "degrees_of_freedom": HOMOGRAPHY_DOF,
            "parameters": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "factorization": "float32_thin_svd_orthonormal_projection",
            "rank_policy": "relative_absolute_and_condition_cutoff_per_phase",
            "reuse_policy": (
                "exact_source_identity_version_shape_device_dtype_and_config_signature"
            ),
        },
        "execution": "X_action+beta*Pcam(source-X_action)",
        "tri_branch_audit_equivalent": (
            "X_action+beta*(Pcam(source-X_noop)-Pcam(X_action-X_noop))"
        ),
        "complement_invariant": (
            "(I-Pcam)(X_exec-X_noop)=(I-Pcam)(X_action-X_noop)"
        ),
        "controls": ["beta", "enabled", "camera_edit_requested"],
        "exact_bypass": [
            "disabled",
            "zero_beta",
            "camera_edit_requested",
            "action_noop_exact_parity",
            "all_phases_degenerate",
        ],
    }


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - host dependent
        raise CameraTangentError(
            "camera-tangent tensor operations require PyTorch"
        ) from error
    return torch


def _validate_config(config: Any) -> None:
    if type(config) is not CameraTangentConfig:
        raise CameraTangentError("config must be an exact CameraTangentConfig")
    config.validate()


def _config_signature(config: CameraTangentConfig) -> tuple[float | int, ...]:
    """Return the exact immutable numerical identity of ``config``."""

    _validate_config(config)
    return (
        float(config.relative_rank_cutoff),
        float(config.absolute_rank_cutoff),
        float(config.max_condition_number),
        int(config.minimum_rank),
        float(config.invariance_atol),
        float(config.invariance_rtol),
    )


def _validate_field(field: Any, *, label: str) -> None:
    torch = _require_torch()
    if not isinstance(field, torch.Tensor):
        raise CameraTangentError(f"{label} must be a torch tensor")
    if (
        field.ndim != 5
        or int(field.shape[0]) <= 0
        or int(field.shape[1]) <= 0
        or int(field.shape[2]) != EXPECTED_LATENT_PHASES
        or int(field.shape[3]) <= 0
        or int(field.shape[4]) <= 0
        or field.dtype != torch.float32
    ):
        raise CameraTangentError(
            f"{label} must be exact float32 [B,C,21,H,W]"
        )
    if not bool(torch.isfinite(field).all()):
        raise CameraTangentError(f"{label} contains non-finite values")


def _validate_tri_branch(
    source_clean_field: Any,
    action_clean_field: Any,
    noop_clean_field: Any,
) -> None:
    _validate_field(source_clean_field, label="source_clean_field")
    _validate_field(action_clean_field, label="action_clean_field")
    _validate_field(noop_clean_field, label="noop_clean_field")
    expected = tuple(source_clean_field.shape)
    if (
        tuple(action_clean_field.shape) != expected
        or tuple(noop_clean_field.shape) != expected
    ):
        raise CameraTangentError("the three clean fields must have identical shapes")
    if (
        action_clean_field.device != source_clean_field.device
        or noop_clean_field.device != source_clean_field.device
    ):
        raise CameraTangentError("the three clean fields must share one device")


def _spatial_gradients(source: Any) -> tuple[Any, Any]:
    """Return derivatives in normalized ``[-1,1]`` image coordinates.

    The finite differences below are initially derivatives per pixel index.
    Homography vector fields are defined on normalized coordinates, so the
    chain rule requires ``(W-1)/2`` and ``(H-1)/2`` scale factors.  The
    distinction changes the perspective tangent span for non-square latents;
    it is not merely a harmless rescaling of all eight columns.
    """

    torch = _require_torch()
    gradient_x = torch.zeros_like(source)
    gradient_y = torch.zeros_like(source)
    width = int(source.shape[-1])
    height = int(source.shape[-2])
    if width > 1:
        gradient_x[..., 0] = source[..., 1] - source[..., 0]
        gradient_x[..., -1] = source[..., -1] - source[..., -2]
        if width > 2:
            gradient_x[..., 1:-1] = 0.5 * (
                source[..., 2:] - source[..., :-2]
            )
    if height > 1:
        gradient_y[..., 0, :] = source[..., 1, :] - source[..., 0, :]
        gradient_y[..., -1, :] = source[..., -1, :] - source[..., -2, :]
        if height > 2:
            gradient_y[..., 1:-1, :] = 0.5 * (
                source[..., 2:, :] - source[..., :-2, :]
            )
    gradient_x.mul_(0.5 * float(width - 1))
    gradient_y.mul_(0.5 * float(height - 1))
    return gradient_x, gradient_y


def build_camera_tangent_basis(
    source_clean_field: Any,
    config: CameraTangentConfig = CameraTangentConfig(),
) -> CameraTangentBasis:
    """Build a separate infinitesimal 8-DoF camera basis for every phase.

    At identity, homography coordinate derivatives for parameters
    ``(a,b,c,d,e,f,g,h)`` are respectively
    ``(x,0)``, ``(y,0)``, ``(1,0)``, ``(0,x)``, ``(0,y)``, ``(0,1)``,
    ``(-x^2,-xy)``, and ``(-xy,-y^2)``.  Contracting these vectors with
    source spatial gradients gives eight tangent images.
    """

    torch = _require_torch()
    _validate_config(config)
    _validate_field(source_clean_field, label="source_clean_field")
    source = source_clean_field.detach()
    gradient_x, gradient_y = _spatial_gradients(source)

    height = int(source.shape[-2])
    width = int(source.shape[-1])
    y_axis = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=source.device)
    x_axis = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=source.device)
    y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x_coord = x_coord.reshape(1, 1, 1, height, width)
    y_coord = y_coord.reshape(1, 1, 1, height, width)

    tangent_images = torch.stack(
        (
            gradient_x * x_coord,
            gradient_x * y_coord,
            gradient_x,
            gradient_y * x_coord,
            gradient_y * y_coord,
            gradient_y,
            gradient_x * (-x_coord.square())
            + gradient_y * (-x_coord * y_coord),
            gradient_x * (-x_coord * y_coord)
            + gradient_y * (-y_coord.square()),
        ),
        dim=-1,
    )
    batch, channels, phases, _, _, _ = tangent_images.shape
    matrix = tangent_images.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, phases, channels * height * width, HOMOGRAPHY_DOF
    )
    # Thin SVD is performed on exact FP32 input.  U is an orthonormal basis;
    # masking rejected columns is consequently an orthogonal projector, not a
    # normal-equation approximation.
    try:
        left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as error:
        raise CameraTangentError("FP32 camera-basis SVD failed") from error
    maximum = singular_values[..., :1]
    absolute = torch.full_like(maximum, float(config.absolute_rank_cutoff))
    relative = maximum * float(config.relative_rank_cutoff)
    conditioned = maximum / float(config.max_condition_number)
    cutoff = torch.maximum(torch.maximum(absolute, relative), conditioned)
    retained = singular_values > cutoff
    rank = retained.sum(dim=-1)
    valid = rank >= int(config.minimum_rank)
    retained = retained & valid.unsqueeze(-1)
    orthonormal = left * retained.unsqueeze(-2).to(dtype=left.dtype)

    infinity = torch.full_like(singular_values, float("inf"))
    smallest = torch.where(retained, singular_values, infinity).amin(dim=-1)
    condition = singular_values[..., 0] / smallest
    condition = torch.where(
        valid,
        condition,
        torch.full_like(condition, float("inf")),
    )
    if bool((valid & (condition > float(config.max_condition_number))).any()):
        raise CameraTangentError("retained camera basis violates condition cutoff")
    if bool(orthonormal.requires_grad):
        raise CameraTangentError("source camera basis was not detached")

    return CameraTangentBasis(
        orthonormal_vectors=orthonormal,
        singular_values=singular_values,
        retained_modes=retained,
        retained_rank=rank,
        condition_number=condition,
        valid_phase=valid,
        cutoff=cutoff.squeeze(-1),
        source_shape=tuple(int(value) for value in source.shape),
        source_device=source.device,
        source_dtype=source.dtype,
        source_data_ptr=int(source_clean_field.data_ptr()),
        source_version=int(source_clean_field._version),
        config_signature=_config_signature(config),
        source_was_detached=True,
        source_identity=source_clean_field,
    )


def validate_precomputed_camera_tangent_basis(
    basis: Any,
    source_clean_field: Any,
    config: CameraTangentConfig = CameraTangentConfig(),
) -> CameraTangentBasis:
    """Fail closed unless ``basis`` belongs to this exact source and config.

    Reuse is intentionally stricter than value equality.  A cloned source,
    an in-place source mutation, or any shape/device/dtype/config change must
    rebuild the basis rather than silently applying a stale camera subspace.
    """

    torch = _require_torch()
    _validate_config(config)
    _validate_field(source_clean_field, label="source_clean_field")
    if type(basis) is not CameraTangentBasis:
        raise CameraTangentError(
            "precomputed_basis must be an exact CameraTangentBasis"
        )
    source_shape = tuple(int(value) for value in source_clean_field.shape)
    if source_clean_field is not basis.source_identity:
        raise CameraTangentError("precomputed basis source identity differs")
    if tuple(basis.source_shape) != source_shape:
        raise CameraTangentError("precomputed basis source shape differs")
    if basis.source_device != source_clean_field.device:
        raise CameraTangentError("precomputed basis source device differs")
    if basis.source_dtype != source_clean_field.dtype:
        raise CameraTangentError("precomputed basis source dtype differs")
    if int(basis.source_data_ptr) != int(source_clean_field.data_ptr()):
        raise CameraTangentError("precomputed basis source storage differs")
    if int(basis.source_version) != int(source_clean_field._version):
        raise CameraTangentError("precomputed basis source was modified in place")
    if tuple(basis.config_signature) != _config_signature(config):
        raise CameraTangentError("precomputed basis config signature differs")
    if basis.source_was_detached is not True:
        raise CameraTangentError("precomputed basis lacks detached-source evidence")

    batch, channels, phases, height, width = source_shape
    flattened = channels * height * width
    tensor_contracts = (
        (
            basis.orthonormal_vectors,
            (batch, phases, flattened, HOMOGRAPHY_DOF),
            torch.float32,
            "orthonormal_vectors",
        ),
        (
            basis.singular_values,
            (batch, phases, HOMOGRAPHY_DOF),
            torch.float32,
            "singular_values",
        ),
        (
            basis.retained_modes,
            (batch, phases, HOMOGRAPHY_DOF),
            torch.bool,
            "retained_modes",
        ),
        (
            basis.retained_rank,
            (batch, phases),
            torch.int64,
            "retained_rank",
        ),
        (
            basis.condition_number,
            (batch, phases),
            torch.float32,
            "condition_number",
        ),
        (
            basis.valid_phase,
            (batch, phases),
            torch.bool,
            "valid_phase",
        ),
        (
            basis.cutoff,
            (batch, phases),
            torch.float32,
            "cutoff",
        ),
    )
    for tensor, expected_shape, expected_dtype, label in tensor_contracts:
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtype
            or tensor.device != source_clean_field.device
            or bool(tensor.requires_grad)
        ):
            raise CameraTangentError(
                f"precomputed basis {label} violates shape/device/dtype/detach contract"
            )
    for tensor, label in (
        (basis.orthonormal_vectors, "orthonormal_vectors"),
        (basis.singular_values, "singular_values"),
        (basis.cutoff, "cutoff"),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise CameraTangentError(f"precomputed basis {label} is non-finite")
    if not bool(
        torch.equal(
            basis.valid_phase,
            basis.retained_rank >= int(config.minimum_rank),
        )
    ):
        raise CameraTangentError("precomputed basis validity metadata differs")
    valid_condition = basis.condition_number[basis.valid_phase]
    if bool(
        valid_condition.numel()
        and (
            (not bool(torch.isfinite(valid_condition).all()))
            or bool(
                (
                    valid_condition > float(config.max_condition_number)
                ).any()
            )
        )
    ):
        raise CameraTangentError("precomputed basis condition metadata differs")
    return basis


def _flatten_by_phase(field: Any) -> Any:
    batch, channels, phases, height, width = field.shape
    return field.permute(0, 2, 1, 3, 4).reshape(
        batch, phases, channels * height * width
    )


def _restore_from_phase(flat: Any, shape: tuple[int, ...]) -> Any:
    batch, channels, phases, height, width = shape
    return flat.reshape(batch, phases, channels, height, width).permute(0, 2, 1, 3, 4)


def project_camera_tangent(field: Any, basis: CameraTangentBasis) -> Any:
    """Orthogonally project a clean-field difference into ``basis``."""

    torch = _require_torch()
    _validate_field(field, label="field")
    if tuple(field.shape) != tuple(basis.source_shape):
        raise CameraTangentError("field shape differs from camera basis source shape")
    vectors = basis.orthonormal_vectors
    if (
        not isinstance(vectors, torch.Tensor)
        or vectors.dtype != torch.float32
        or vectors.device != field.device
        or vectors.ndim != 4
        or tuple(vectors.shape[:2])
        != (int(field.shape[0]), EXPECTED_LATENT_PHASES)
        or int(vectors.shape[2])
        != int(field.shape[1]) * int(field.shape[3]) * int(field.shape[4])
        or int(vectors.shape[3]) != HOMOGRAPHY_DOF
    ):
        raise CameraTangentError("camera basis tensor differs from field contract")
    flat = _flatten_by_phase(field)
    coefficients = torch.matmul(
        vectors.transpose(-2, -1), flat.unsqueeze(-1)
    )
    projected = torch.matmul(vectors, coefficients).squeeze(-1)
    return _restore_from_phase(projected, tuple(int(value) for value in field.shape))


def _canonical_beta(beta: Any, reference: Any) -> tuple[Any, str]:
    torch = _require_torch()
    if isinstance(beta, bool):
        raise CameraTangentError("beta must be a finite scalar or per-phase value")
    try:
        value = torch.as_tensor(beta, dtype=torch.float32, device=reference.device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CameraTangentError(
            "beta must be a finite scalar or per-phase value"
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
        raise CameraTangentError(
            "beta must be scalar, [21], [1,21], or [B,21]"
        )
    if not bool(torch.isfinite(value).all()):
        raise CameraTangentError("beta contains non-finite values")
    return value.detach(), mode


def _phase_rms(value: Any) -> Any:
    return value.square().mean(dim=(1, 3, 4)).sqrt()


def _bypass_trace(
    *,
    reason: str,
    beta_mode: str,
    beta_per_phase: Any,
) -> CameraStabilizationTrace:
    return CameraStabilizationTrace(
        schema_version=SCHEMA_VERSION,
        method=METHOD_NAME,
        bypassed=True,
        bypass_reason=reason,
        beta_mode=beta_mode,
        beta_per_phase=beta_per_phase.detach(),
        basis_built=False,
        basis_reused=False,
        source_basis_detached=False,
        retained_rank=None,
        condition_number=None,
        valid_phase=None,
        camera_component_rms_before=None,
        camera_component_rms_after=None,
        source_camera_component_rms=None,
        applied_correction_rms=None,
        noncamera_invariance_max_abs=0.0,
        noncamera_invariance_rms=0.0,
        noncamera_invariance_tolerance=0.0,
        invariant_satisfied=True,
    )


def stabilize_camera_tangent(
    source_clean_field: Any,
    action_clean_field: Any,
    noop_clean_field: Any,
    *,
    beta: Any = 1.0,
    enabled: bool = True,
    camera_edit_requested: bool = False,
    config: CameraTangentConfig = CameraTangentConfig(),
    precomputed_basis: CameraTangentBasis | None = None,
) -> CameraStabilizationResult:
    """Replace only the action field's source-relative camera component.

    Every hard bypass returns the original ``action_clean_field`` Python
    object.  With a partially degenerate basis, invalid phases receive an
    exact zero correction while valid phases are stabilized independently.
    """

    torch = _require_torch()
    _validate_config(config)
    _validate_tri_branch(
        source_clean_field, action_clean_field, noop_clean_field
    )
    if type(enabled) is not bool:
        raise CameraTangentError("enabled must be bool")
    if type(camera_edit_requested) is not bool:
        raise CameraTangentError("camera_edit_requested must be bool")
    beta_per_phase, beta_mode = _canonical_beta(beta, action_clean_field)

    reason = None
    if not enabled:
        reason = "disabled"
    elif bool(torch.equal(beta_per_phase, torch.zeros_like(beta_per_phase))):
        reason = "zero_beta"
    elif camera_edit_requested:
        reason = "camera_edit_requested"
    elif bool(torch.equal(action_clean_field, noop_clean_field)):
        reason = "action_noop_exact_parity"
    if reason is not None:
        return CameraStabilizationResult(
            executed_clean_field=action_clean_field,
            trace=_bypass_trace(
                reason=reason,
                beta_mode=beta_mode,
                beta_per_phase=beta_per_phase,
            ),
        )

    if precomputed_basis is None:
        basis = build_camera_tangent_basis(source_clean_field, config)
        basis_reused = False
    else:
        basis = validate_precomputed_camera_tangent_basis(
            precomputed_basis, source_clean_field, config
        )
        basis_reused = True
    if not bool(basis.valid_phase.any()):
        trace = _bypass_trace(
            reason="all_phases_degenerate",
            beta_mode=beta_mode,
            beta_per_phase=beta_per_phase,
        )
        trace = CameraStabilizationTrace(
            **{
                **trace.__dict__,
                "basis_built": True,
                "basis_reused": basis_reused,
                "source_basis_detached": basis.source_was_detached,
                "retained_rank": basis.retained_rank.detach(),
                "condition_number": basis.condition_number.detach(),
                "valid_phase": basis.valid_phase.detach(),
            }
        )
        return CameraStabilizationResult(
            executed_clean_field=action_clean_field,
            trace=trace,
        )

    source_delta = source_clean_field.detach() - noop_clean_field
    action_delta = action_clean_field - noop_clean_field
    source_camera = project_camera_tangent(source_delta, basis)
    action_camera = project_camera_tangent(action_delta, basis)
    phase_scale = beta_per_phase[:, None, :, None, None]
    correction = phase_scale * (source_camera - action_camera)
    executed = action_clean_field + correction

    executed_delta = executed - noop_clean_field
    executed_camera = project_camera_tangent(executed_delta, basis)
    noncamera_before = action_delta - action_camera
    noncamera_after = executed_delta - executed_camera
    invariant_error = (noncamera_after - noncamera_before).abs()
    element_tolerance = float(config.invariance_atol) + float(
        config.invariance_rtol
    ) * noncamera_before.abs()
    invariant_satisfied = bool(torch.all(invariant_error <= element_tolerance))
    maximum_error = float(invariant_error.max().detach().cpu().item())
    rms_error = float(invariant_error.square().mean().sqrt().detach().cpu().item())
    maximum_tolerance = float(element_tolerance.max().detach().cpu().item())
    if not invariant_satisfied:
        raise CameraTangentError(
            "orthogonal-complement invariant failed: "
            f"max_abs={maximum_error:.8g}, tolerance={maximum_tolerance:.8g}"
        )
    if not bool(torch.isfinite(executed).all()):
        raise CameraTangentError("camera stabilization produced non-finite values")

    trace = CameraStabilizationTrace(
        schema_version=SCHEMA_VERSION,
        method=METHOD_NAME,
        bypassed=False,
        bypass_reason=None,
        beta_mode=beta_mode,
        beta_per_phase=beta_per_phase.detach(),
        basis_built=True,
        basis_reused=basis_reused,
        source_basis_detached=basis.source_was_detached,
        retained_rank=basis.retained_rank.detach(),
        condition_number=basis.condition_number.detach(),
        valid_phase=basis.valid_phase.detach(),
        camera_component_rms_before=_phase_rms(action_camera).detach(),
        camera_component_rms_after=_phase_rms(executed_camera).detach(),
        source_camera_component_rms=_phase_rms(source_camera).detach(),
        applied_correction_rms=_phase_rms(correction).detach(),
        noncamera_invariance_max_abs=maximum_error,
        noncamera_invariance_rms=rms_error,
        noncamera_invariance_tolerance=maximum_tolerance,
        invariant_satisfied=True,
    )
    return CameraStabilizationResult(executed_clean_field=executed, trace=trace)


__all__ = [
    "CameraStabilizationResult",
    "CameraStabilizationTrace",
    "CameraTangentBasis",
    "CameraTangentConfig",
    "CameraTangentError",
    "EXPECTED_LATENT_PHASES",
    "HOMOGRAPHY_DOF",
    "METHOD_NAME",
    "build_camera_tangent_basis",
    "camera_stabilizer_contract_receipt",
    "project_camera_tangent",
    "stabilize_camera_tangent",
    "validate_precomputed_camera_tangent_basis",
]

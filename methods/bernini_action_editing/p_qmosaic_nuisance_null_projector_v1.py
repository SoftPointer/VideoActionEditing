#!/usr/bin/env python3
"""Versioned nuisance-null cotangent projection for P-Q-MOSAIC.

This module is deliberately separate from the authenticated Q-MOSAIC runner.
It prepares a future projected direction canary, but it cannot authorize a
decode, a scientific conclusion, an optimizer, or a parameter update.

The only public intervention constructor performs these operations in order:

1. accept one detached FP32 exact81 clean-latent VJP ``[1,16,21,H,W]``;
2. reuse :func:`project_source_safe_cotangent` to remove the fixed phase-0,
   active-phase temporal-DC, and spatial ``{1,x,y}`` affine nuisance modes;
3. independently reconstruct and verify the same orthogonal projection in
   FP64, including its geometry, energy accounting, and null residuals;
4. normalize the *projected* VJP with the unchanged raw Q-MOSAIC rule
   ``delta = 0.01 * ||base||_2 * projected / ||projected||_2``; and
5. return both symmetric arms.  There is no seed or dose input and no arm
   selection path.

No mask, track, pose, optical flow, box, or content-derived spatial support is
accepted.  These algebraic nuisance constraints do not establish identity,
camera, or action preservation; only a later pre-registered exact81 outcome
gate may assess those properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from typing import Any

import torch

import self_imagined_relational_motion as _relational


PROJECTION_SCHEMA_VERSION = "bernini-p-qmosaic-nuisance-null-projection-v1"
INTERVENTION_SCHEMA_VERSION = "bernini-p-qmosaic-relative-l2-intervention-v1"
LATENT_CHANNELS = 16
LATENT_PHASES = 21
RELATIVE_L2_DOSE = 0.01
MINIMUM_PROJECTION_SURVIVAL_RATIO = 0.10

_MIN_L2_NORM = 1.0e-12
_NULL_ABSOLUTE_TOLERANCE = 3.0e-6
_FORMULA_RELATIVE_TOLERANCE = 1.0e-6
_ENERGY_RELATIVE_TOLERANCE = 2.0e-6
_FP32_NORM_RELATIVE_TOLERANCE = 5.0e-5


class PQMosaicProjectionError(RuntimeError):
    """A projected clean-latent direction violated the closed v1 contract."""


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
        raise PQMosaicProjectionError(
            "projection receipt is not finite canonical ASCII JSON"
        ) from error


def _seal(unsigned: dict[str, Any]) -> dict[str, Any]:
    if "receipt_digest" in unsigned:
        raise PQMosaicProjectionError("projection receipt is already sealed")
    digest = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    return {**unsigned, "receipt_digest": digest}


def _untyped_storage(value: torch.Tensor) -> Any:
    """Return the underlying storage across Bernini's pinned/newer Torch."""

    getter = getattr(value, "untyped_storage", None)
    if callable(getter):
        return getter()
    typed = value.storage()
    getter = getattr(typed, "_untyped", None)
    if callable(getter):
        return getter()
    else:
        raise PQMosaicProjectionError("tensor storage identity is unavailable")


def _tensor_value_sha256(value: torch.Tensor, *, label: str) -> str:
    """Hash exact logical tensor bytes plus dtype/shape without NumPy."""

    if (
        not isinstance(value, torch.Tensor)
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PQMosaicProjectionError(f"{label} must be a finite real tensor")
    # The clone is an owned byte snapshot used only for hashing.  It is never
    # returned and never replaces a mutated live tensor.
    owned = value.detach().to(device="cpu").contiguous().clone()
    payload = io.BytesIO()
    untyped = _untyped_storage(owned)
    untyped._write_file(payload, False, False, 1)
    raw = payload.getvalue()
    expected = int(owned.numel()) * int(owned.element_size())
    if len(raw) != expected:
        raise PQMosaicProjectionError(f"{label} owned byte closure differs")
    header = _canonical_json_bytes(
        {
            "dtype": str(owned.dtype),
            "shape": list(map(int, owned.shape)),
            "numel": int(owned.numel()),
        }
    )
    return hashlib.sha256(header + b"\x00" + raw).hexdigest()


@dataclass(frozen=True)
class _TensorRuntimeBinding:
    object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int
    storage_version: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    is_contiguous: bool
    numel: int
    tensor_sha256: str

    def portable_receipt(self) -> dict[str, Any]:
        """Expose the portable value identity; runtime pointers remain private."""

        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "layout": self.layout,
            "requires_grad": self.requires_grad,
            "is_contiguous": self.is_contiguous,
            "numel": self.numel,
            "tensor_sha256": self.tensor_sha256,
            "construction_object_identity_live_checked": True,
            "construction_storage_identity_live_checked": True,
            "construction_storage_version_live_checked": True,
        }


def _bind_tensor(value: torch.Tensor, *, label: str) -> _TensorRuntimeBinding:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not bool(torch.isfinite(value).all().item())
    ):
        raise PQMosaicProjectionError(f"{label} tensor identity cannot be sealed")
    try:
        version = int(value._version)  # noqa: SLF001 - mutation seal
        storage = _untyped_storage(value)
        storage_data_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise PQMosaicProjectionError(
            f"{label} runtime storage identity cannot be sealed"
        ) from error
    if storage_data_ptr <= 0 or storage_nbytes <= 0:
        raise PQMosaicProjectionError(f"{label} runtime storage is degenerate")
    return _TensorRuntimeBinding(
        object_id=id(value),
        storage_data_ptr=storage_data_ptr,
        storage_nbytes=storage_nbytes,
        storage_offset=int(value.storage_offset()),
        storage_version=version,
        shape=tuple(map(int, value.shape)),
        stride=tuple(map(int, value.stride())),
        dtype=str(value.dtype),
        device=str(value.device),
        layout=str(value.layout),
        requires_grad=bool(value.requires_grad),
        is_contiguous=bool(value.is_contiguous()),
        numel=int(value.numel()),
        tensor_sha256=_tensor_value_sha256(value, label=label),
    )


def _assert_live_tensor(
    value: torch.Tensor,
    binding: _TensorRuntimeBinding,
    *,
    label: str,
) -> dict[str, Any]:
    """Rehash and compare the live tensor to its construction-time seal."""

    if type(binding) is not _TensorRuntimeBinding:
        raise PQMosaicProjectionError(f"{label} construction seal is invalid")
    live = _bind_tensor(value, label=f"live {label}")
    if live != binding:
        raise PQMosaicProjectionError(
            f"{label} changed after construction; receipt denied"
        )
    return live.portable_receipt()


def _assert_no_storage_aliases(
    bindings: tuple[tuple[str, _TensorRuntimeBinding], ...],
) -> None:
    seen: dict[tuple[str, int], str] = {}
    for role, binding in bindings:
        key = (binding.device, binding.storage_data_ptr)
        previous = seen.get(key)
        if previous is not None:
            raise PQMosaicProjectionError(
                f"{previous}/{role} unexpectedly share mutable storage"
            )
        seen[key] = role


def _validate_clean_latent_tensor(value: Any, *, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.dtype != torch.float32
        or value.ndim != 5
    ):
        raise PQMosaicProjectionError(
            f"{label} must be a dense materialized FP32 [1,16,21,H,W] tensor"
        )
    shape = tuple(int(item) for item in value.shape)
    if (
        shape[0] != 1
        or shape[1] != LATENT_CHANNELS
        or shape[2] != LATENT_PHASES
        or shape[3] < 2
        or shape[4] < 2
    ):
        raise PQMosaicProjectionError(
            f"{label} geometry must be exactly [1,16,21,H>=2,W>=2]"
        )
    if value.requires_grad:
        raise PQMosaicProjectionError(f"{label} must be detached")
    if not bool(torch.isfinite(value).all().item()):
        raise PQMosaicProjectionError(f"{label} contains NaN or infinity")
    return value


def _orthonormal_affine_basis(
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Independently construct the fixed FP64 spatial ``{1,x,y}`` basis."""

    if height < 2 or width < 2:
        raise PQMosaicProjectionError("spatial affine geometry is degenerate")
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rows = torch.stack((torch.ones_like(xx), xx, yy), dim=0).reshape(3, -1)
    norms = torch.linalg.vector_norm(rows, ord=2, dim=1, keepdim=True)
    if (
        not bool(torch.isfinite(norms).all().item())
        or bool((norms <= _MIN_L2_NORM).any().item())
    ):
        raise PQMosaicProjectionError("spatial affine basis is degenerate")
    basis = rows / norms
    gram = basis @ basis.transpose(0, 1)
    if not bool(
        torch.allclose(
            gram,
            torch.eye(3, dtype=torch.float64, device=device),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    ):
        raise PQMosaicProjectionError("spatial {1,x,y} basis is not orthonormal")
    return basis


@dataclass(frozen=True)
class _IndependentDecomposition:
    projected: torch.Tensor
    affine_basis: torch.Tensor
    phase0_removed: torch.Tensor
    temporal_dc_removed: torch.Tensor
    spatial_affine_removed: torch.Tensor


def _independent_decomposition(raw: torch.Tensor) -> _IndependentDecomposition:
    """Compute a mutually orthogonal FP64 nuisance decomposition.

    The fixed accounting order resolves intersections between nuisance
    subspaces without double counting:

    ``phase0 -> active temporal DC -> spatial affine of the survivor``.

    The final projector is order-independent because the temporal and spatial
    operators act on different axes.  This order is only for energy receipts.
    """

    raw64 = raw.detach().to(dtype=torch.float64)
    zero_phase0 = torch.zeros_like(raw64[:, :, :1])
    zero_active = torch.zeros_like(raw64[:, :, 1:])

    phase0_removed = torch.cat((raw64[:, :, :1], zero_active), dim=2)
    after_phase0 = torch.cat((zero_phase0, raw64[:, :, 1:]), dim=2)

    active_mean = after_phase0[:, :, 1:].mean(dim=2, keepdim=True)
    temporal_dc_removed = torch.cat(
        (zero_phase0, active_mean.expand_as(after_phase0[:, :, 1:])), dim=2
    )
    after_temporal = after_phase0 - temporal_dc_removed

    height, width = int(raw64.shape[-2]), int(raw64.shape[-1])
    basis = _orthonormal_affine_basis(height, width, device=raw64.device)
    flat = after_temporal.reshape(-1, height * width)
    coefficients = flat @ basis.transpose(0, 1)
    spatial_affine_removed = (coefficients @ basis).reshape_as(raw64)
    projected = (after_temporal - spatial_affine_removed).contiguous()
    return _IndependentDecomposition(
        projected=projected,
        affine_basis=basis,
        phase0_removed=phase0_removed,
        temporal_dc_removed=temporal_dc_removed,
        spatial_affine_removed=spatial_affine_removed,
    )


def _l2_energy(value: torch.Tensor) -> float:
    result = value.to(dtype=torch.float64).square().sum()
    if not bool(torch.isfinite(result).item()):
        raise PQMosaicProjectionError("projection energy is non-finite")
    return float(result.detach().item())


def _l2_norm(value: torch.Tensor) -> float:
    result = torch.linalg.vector_norm(value.to(dtype=torch.float64))
    if not bool(torch.isfinite(result).item()):
        raise PQMosaicProjectionError("projection norm is non-finite")
    return float(result.detach().item())


@dataclass(frozen=True)
class PQMosaicNuisanceNullProjection:
    """Owned projected VJP plus numerical-only, non-authoritative evidence."""

    tensor: torch.Tensor
    raw_l2_norm: float
    projected_l2_norm: float
    projection_survival_ratio: float
    projection_ascent_cosine: float
    phase0_removed_l2_energy: float
    temporal_dc_removed_l2_energy: float
    spatial_affine_removed_l2_energy: float
    total_removed_l2_energy: float
    energy_closure_relative_error: float
    projection_formula_relative_l2_error: float
    projection_formula_relative_max_abs_error: float
    phase0_max_abs: float
    active_temporal_sum_max_abs: float
    spatial_affine_max_abs_coefficient: float
    removed_projected_dot_over_raw_energy: float
    nuisance_component_pairwise_dot_over_raw_energy: float
    _raw_clean_latent_vjp: torch.Tensor = field(repr=False, compare=False)
    _raw_binding: _TensorRuntimeBinding = field(repr=False, compare=False)
    _projected_binding: _TensorRuntimeBinding = field(repr=False, compare=False)

    def receipt(self) -> dict[str, Any]:
        raw_tensor_binding = _assert_live_tensor(
            self._raw_clean_latent_vjp,
            self._raw_binding,
            label="raw clean-latent VJP",
        )
        projected_tensor_binding = _assert_live_tensor(
            self.tensor,
            self._projected_binding,
            label="projected clean-latent VJP",
        )
        _assert_no_storage_aliases(
            (
                ("raw_clean_latent_vjp", self._raw_binding),
                ("projected_clean_latent_vjp", self._projected_binding),
            )
        )
        raw_energy = self.raw_l2_norm * self.raw_l2_norm
        unsigned: dict[str, Any] = {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "coordinate": "detached_fp32_clean_latent_vjp_exact81",
            "latent_geometry": {
                "batch": 1,
                "channels": LATENT_CHANNELS,
                "phases": LATENT_PHASES,
                "height": int(self.tensor.shape[-2]),
                "width": int(self.tensor.shape[-1]),
            },
            "tensor_bindings": {
                "raw_clean_latent_vjp": raw_tensor_binding,
                "projected_clean_latent_vjp": projected_tensor_binding,
            },
            "live_construction_seal": {
                "all_tensor_values_rehashed": True,
                "object_identity_revalidated": True,
                "storage_identity_revalidated": True,
                "storage_version_revalidated": True,
                "shape_dtype_device_stride_revalidated": True,
                "mutable_storage_aliases": False,
            },
            "projector": {
                "reused_function": (
                    "self_imagined_relational_motion."
                    "project_source_safe_cotangent"
                ),
                "reused_projector_called": True,
                "independent_fp64_formula_recomputed": True,
                "minimum_projection_survival_ratio": (
                    MINIMUM_PROJECTION_SURVIVAL_RATIO
                ),
                "fixed_nulls": [
                    "phase0",
                    "active_phases_1_to_20_temporal_dc",
                    "per_channel_phase_spatial_affine_1_x_y",
                ],
            },
            "norms": {
                "raw_l2_norm": self.raw_l2_norm,
                "projected_l2_norm": self.projected_l2_norm,
                "projection_survival_ratio": self.projection_survival_ratio,
                "projection_ascent_cosine": self.projection_ascent_cosine,
            },
            "removed_subspace_l2_energy": {
                "phase0": self.phase0_removed_l2_energy,
                "active_temporal_dc_after_phase0": (
                    self.temporal_dc_removed_l2_energy
                ),
                "spatial_affine_1_x_y_after_phase0_and_temporal_dc": (
                    self.spatial_affine_removed_l2_energy
                ),
                "total_without_double_counting": self.total_removed_l2_energy,
            },
            "removed_subspace_energy_fraction_of_raw": {
                "phase0": self.phase0_removed_l2_energy / raw_energy,
                "active_temporal_dc_after_phase0": (
                    self.temporal_dc_removed_l2_energy / raw_energy
                ),
                "spatial_affine_1_x_y_after_phase0_and_temporal_dc": (
                    self.spatial_affine_removed_l2_energy / raw_energy
                ),
                "total_without_double_counting": (
                    self.total_removed_l2_energy / raw_energy
                ),
            },
            "energy_accounting_order": [
                "phase0",
                "active_temporal_dc_after_phase0",
                "spatial_affine_1_x_y_after_phase0_and_temporal_dc",
                "projected_survivor",
            ],
            "orthogonality_residuals": {
                "phase0_max_abs": self.phase0_max_abs,
                "active_temporal_sum_max_abs": (
                    self.active_temporal_sum_max_abs
                ),
                "spatial_affine_max_abs_coefficient": (
                    self.spatial_affine_max_abs_coefficient
                ),
                "removed_projected_dot_over_raw_energy": (
                    self.removed_projected_dot_over_raw_energy
                ),
                "nuisance_component_pairwise_dot_over_raw_energy": (
                    self.nuisance_component_pairwise_dot_over_raw_energy
                ),
                "energy_closure_relative_error": (
                    self.energy_closure_relative_error
                ),
                "independent_formula_relative_l2_error": (
                    self.projection_formula_relative_l2_error
                ),
                "independent_formula_relative_max_abs_error": (
                    self.projection_formula_relative_max_abs_error
                ),
            },
            "content_inputs": {
                "mask": False,
                "track": False,
                "pose": False,
                "flow": False,
                "box": False,
                "content_derived_spatial_support": False,
            },
            "scientific_authority": False,
            "update": False,
            "optimizer_created": False,
            "parameter_update": False,
            "identity_or_camera_preservation_proven": False,
        }
        return _seal(unsigned)


def project_raw_clean_latent_vjp(
    raw_clean_latent_vjp: torch.Tensor,
) -> PQMosaicNuisanceNullProjection:
    """Project and independently audit one raw exact81 clean-latent VJP.

    The signature intentionally accepts no seed, dose, mask, or semantic
    selection input.  Any failure returns no partial direction.
    """

    raw = _validate_clean_latent_tensor(
        raw_clean_latent_vjp, label="raw clean-latent VJP"
    )
    raw_snapshot = raw.detach().clone(memory_format=torch.contiguous_format)
    raw64 = raw_snapshot.to(dtype=torch.float64)
    raw_norm = _l2_norm(raw64)
    raw_energy = _l2_energy(raw64)
    if raw_norm <= _MIN_L2_NORM or raw_energy <= _MIN_L2_NORM * _MIN_L2_NORM:
        raise PQMosaicProjectionError("raw clean-latent VJP is degenerate")

    reference = _independent_decomposition(raw_snapshot)
    try:
        reused = _relational.project_source_safe_cotangent(
            raw,
            minimum_survival_ratio=MINIMUM_PROJECTION_SURVIVAL_RATIO,
        )
    except _relational.RelationalMotionError as error:
        raise PQMosaicProjectionError(
            "reused nuisance-null projector failed closed"
        ) from error
    if not torch.equal(raw, raw_snapshot):
        raise PQMosaicProjectionError("reused projector mutated the raw VJP")
    projected = _validate_clean_latent_tensor(
        reused, label="reused projected clean-latent VJP"
    )
    if (
        tuple(projected.shape) != tuple(raw.shape)
        or projected.device != raw.device
    ):
        raise PQMosaicProjectionError("raw/projected VJP geometry or device differs")
    projected = projected.detach().contiguous().clone()
    projected64 = projected.to(dtype=torch.float64)
    projected_norm = _l2_norm(projected64)
    if projected_norm <= _MIN_L2_NORM:
        raise PQMosaicProjectionError("nuisance projection is degenerate")

    reference_norm = _l2_norm(reference.projected)
    if reference_norm <= _MIN_L2_NORM:
        raise PQMosaicProjectionError(
            "independent nuisance projection removed the entire VJP"
        )
    formula_difference = projected64 - reference.projected
    formula_relative_l2 = _l2_norm(formula_difference) / reference_norm
    formula_relative_max = float(formula_difference.abs().max().item()) / max(
        float(reference.projected.abs().max().item()), _MIN_L2_NORM
    )
    if (
        not math.isfinite(formula_relative_l2)
        or not math.isfinite(formula_relative_max)
        or formula_relative_l2 > _FORMULA_RELATIVE_TOLERANCE
        or formula_relative_max > _FORMULA_RELATIVE_TOLERANCE
    ):
        raise PQMosaicProjectionError(
            "reused projector differs from the independent fixed-null formula"
        )

    survival = projected_norm / raw_norm
    ascent = float(
        torch.dot(raw64.reshape(-1), projected64.reshape(-1)).item()
        / (raw_norm * projected_norm)
    )
    if (
        not math.isfinite(survival)
        or not math.isfinite(ascent)
        or survival < MINIMUM_PROJECTION_SURVIVAL_RATIO
        or ascent < MINIMUM_PROJECTION_SURVIVAL_RATIO
    ):
        raise PQMosaicProjectionError(
            "projected VJP did not survive the fixed nuisance quotient"
        )

    height, width = int(raw.shape[-2]), int(raw.shape[-1])
    phase0_max = float(projected64[:, :, 0].abs().max().item())
    temporal_sum_max = float(
        projected64[:, :, 1:].sum(dim=2).abs().max().item()
    )
    affine_coefficients = (
        projected64.reshape(-1, height * width)
        @ reference.affine_basis.transpose(0, 1)
    )
    affine_max = float(affine_coefficients.abs().max().item())
    if phase0_max != 0.0:
        raise PQMosaicProjectionError("phase0-null constraint is not exact")
    if temporal_sum_max > _NULL_ABSOLUTE_TOLERANCE:
        raise PQMosaicProjectionError("temporal-DC-null residual is too large")
    if affine_max > _NULL_ABSOLUTE_TOLERANCE:
        raise PQMosaicProjectionError(
            "spatial-{1,x,y}-affine-null residual is too large"
        )

    phase0_energy = _l2_energy(reference.phase0_removed)
    temporal_energy = _l2_energy(reference.temporal_dc_removed)
    affine_energy = _l2_energy(reference.spatial_affine_removed)
    total_removed_energy = phase0_energy + temporal_energy + affine_energy
    projected_energy = _l2_energy(projected64)
    closure_relative = abs(
        raw_energy - total_removed_energy - projected_energy
    ) / raw_energy

    removed = raw64 - projected64
    removed_projected_residual = abs(
        float(torch.dot(removed.reshape(-1), projected64.reshape(-1)).item())
    ) / raw_energy
    components = (
        reference.phase0_removed,
        reference.temporal_dc_removed,
        reference.spatial_affine_removed,
    )
    pairwise_residual = max(
        abs(float(torch.dot(left.reshape(-1), right.reshape(-1)).item()))
        / raw_energy
        for index, left in enumerate(components)
        for right in components[index + 1 :]
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                closure_relative,
                removed_projected_residual,
                pairwise_residual,
            )
        )
        or closure_relative > _ENERGY_RELATIVE_TOLERANCE
        or removed_projected_residual > _ENERGY_RELATIVE_TOLERANCE
        or pairwise_residual > _ENERGY_RELATIVE_TOLERANCE
    ):
        raise PQMosaicProjectionError(
            "nuisance projection energy/orthogonality accounting did not close"
        )

    raw_binding = _bind_tensor(raw, label="raw clean-latent VJP")
    projected_binding = _bind_tensor(
        projected, label="projected clean-latent VJP"
    )
    _assert_no_storage_aliases(
        (
            ("raw_clean_latent_vjp", raw_binding),
            ("projected_clean_latent_vjp", projected_binding),
        )
    )

    return PQMosaicNuisanceNullProjection(
        tensor=projected,
        raw_l2_norm=raw_norm,
        projected_l2_norm=projected_norm,
        projection_survival_ratio=survival,
        projection_ascent_cosine=ascent,
        phase0_removed_l2_energy=phase0_energy,
        temporal_dc_removed_l2_energy=temporal_energy,
        spatial_affine_removed_l2_energy=affine_energy,
        total_removed_l2_energy=total_removed_energy,
        energy_closure_relative_error=closure_relative,
        projection_formula_relative_l2_error=formula_relative_l2,
        projection_formula_relative_max_abs_error=formula_relative_max,
        phase0_max_abs=phase0_max,
        active_temporal_sum_max_abs=temporal_sum_max,
        spatial_affine_max_abs_coefficient=affine_max,
        removed_projected_dot_over_raw_energy=removed_projected_residual,
        nuisance_component_pairwise_dot_over_raw_energy=pairwise_residual,
        _raw_clean_latent_vjp=raw,
        _raw_binding=raw_binding,
        _projected_binding=projected_binding,
    )


@dataclass(frozen=True)
class PQMosaicSymmetricLatents:
    """Projected fixed-dose base/plus/minus coordinates for a future runner."""

    base: torch.Tensor
    plus: torch.Tensor
    minus: torch.Tensor
    delta: torch.Tensor
    unit_direction: torch.Tensor
    projection: PQMosaicNuisanceNullProjection
    base_l2_norm: float
    projected_vjp_l2_norm_fp32: float
    direction_l2_norm: float
    absolute_dose_l2_fp32_scale: float
    plus_delta_l2: float
    minus_delta_l2: float
    observed_relative_l2_dose: float
    midpoint_max_abs_error: float
    delta_antisymmetry_max_abs_error: float
    delta_norm_symmetry_absolute_error: float
    symmetry_tolerance: float
    _base_binding: _TensorRuntimeBinding = field(repr=False, compare=False)
    _unit_direction_binding: _TensorRuntimeBinding = field(
        repr=False, compare=False
    )
    _delta_binding: _TensorRuntimeBinding = field(repr=False, compare=False)
    _plus_binding: _TensorRuntimeBinding = field(repr=False, compare=False)
    _minus_binding: _TensorRuntimeBinding = field(repr=False, compare=False)

    def receipt(self) -> dict[str, Any]:
        # The nested call rehashes both raw and projected VJPs.  Rehash every
        # downstream tensor separately; do not repair or replace mutations.
        projection_receipt = self.projection.receipt()
        tensor_bindings = {
            "base_clean_latent": _assert_live_tensor(
                self.base, self._base_binding, label="base clean latent"
            ),
            "unit_projected_direction": _assert_live_tensor(
                self.unit_direction,
                self._unit_direction_binding,
                label="unit projected direction",
            ),
            "projected_delta": _assert_live_tensor(
                self.delta, self._delta_binding, label="projected delta"
            ),
            "plus_clean_latent": _assert_live_tensor(
                self.plus, self._plus_binding, label="plus clean latent"
            ),
            "minus_clean_latent": _assert_live_tensor(
                self.minus, self._minus_binding, label="minus clean latent"
            ),
        }
        _assert_no_storage_aliases(
            (
                ("projected_clean_latent_vjp", self.projection._projected_binding),
                ("base_clean_latent", self._base_binding),
                ("unit_projected_direction", self._unit_direction_binding),
                ("projected_delta", self._delta_binding),
                ("plus_clean_latent", self._plus_binding),
                ("minus_clean_latent", self._minus_binding),
            )
        )
        unsigned: dict[str, Any] = {
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "formula": (
                "q=P_null(raw_clean_vjp)/l2(P_null(raw_clean_vjp));"
                "scale=0.01*l2(base);plus=base+scale*q;minus=base-scale*q"
            ),
            "fp32_operation_order": (
                "project_then_torch.float32_vector_norm_then_mul_then_add_or_sub"
            ),
            "relative_l2_dose": RELATIVE_L2_DOSE,
            "tensor_bindings": tensor_bindings,
            "live_construction_seal": {
                "raw_and_projected_vjp_rehashed_by_nested_receipt": True,
                "all_downstream_tensor_values_rehashed": True,
                "object_identity_revalidated": True,
                "storage_identity_revalidated": True,
                "storage_version_revalidated": True,
                "shape_dtype_device_stride_revalidated": True,
                "mutable_storage_aliases": False,
            },
            "base_l2_norm": self.base_l2_norm,
            "projected_vjp_l2_norm_fp32": self.projected_vjp_l2_norm_fp32,
            "direction_l2_norm": self.direction_l2_norm,
            "absolute_dose_l2": self.absolute_dose_l2_fp32_scale,
            "plus_delta_l2": self.plus_delta_l2,
            "minus_delta_l2": self.minus_delta_l2,
            "observed_relative_l2_dose": self.observed_relative_l2_dose,
            "midpoint_max_abs_error": self.midpoint_max_abs_error,
            "delta_antisymmetry_max_abs_error": (
                self.delta_antisymmetry_max_abs_error
            ),
            "delta_norm_symmetry_absolute_error": (
                self.delta_norm_symmetry_absolute_error
            ),
            "symmetry_tolerance": self.symmetry_tolerance,
            "formula_recomputed_exact_fp32": True,
            "projection_precedes_normalization": True,
            "same_relative_l2_dose_as_raw_qmosaic": True,
            "latent_symmetry_passed": True,
            "seed_selection": False,
            "dose_selection": False,
            "arm_selection": False,
            "mask_track_pose_flow_used": False,
            "scientific_authority": False,
            "update": False,
            "optimizer_created": False,
            "parameter_update": False,
            "projection_receipt": projection_receipt,
        }
        return _seal(unsigned)


def construct_projected_symmetric_latents(
    *,
    base_clean_latent: torch.Tensor,
    raw_clean_latent_vjp: torch.Tensor,
) -> PQMosaicSymmetricLatents:
    """Apply the fixed projector, then the sole relative-L2 dose ``0.01``.

    There is intentionally no caller-configurable dose, seed, arm, or spatial
    support.  The FP32 normalization/addition order matches raw Q-MOSAIC; only
    the direction is replaced by the independently verified projected VJP.
    """

    base_input = _validate_clean_latent_tensor(
        base_clean_latent, label="base clean latent"
    )
    raw = _validate_clean_latent_tensor(
        raw_clean_latent_vjp, label="raw clean-latent VJP"
    )
    if tuple(base_input.shape) != tuple(raw.shape) or base_input.device != raw.device:
        raise PQMosaicProjectionError(
            "base clean latent and raw VJP geometry/device differ"
        )
    base = base_input.detach().contiguous().clone()
    base_snapshot = base.clone()
    projection = project_raw_clean_latent_vjp(raw)
    projected = projection.tensor

    # This is deliberately the same FP32 normalization and scale construction
    # used by the raw Q-MOSAIC exact81 direction runner.
    base_norm_fp32 = torch.linalg.vector_norm(base)
    projected_norm_fp32 = torch.linalg.vector_norm(projected)
    if (
        not bool(torch.isfinite(base_norm_fp32).item())
        or not bool(torch.isfinite(projected_norm_fp32).item())
        or float(base_norm_fp32.item()) <= _MIN_L2_NORM
        or float(projected_norm_fp32.item()) <= _MIN_L2_NORM
    ):
        raise PQMosaicProjectionError("base/projected FP32 norm is degenerate")
    direction = (projected / projected_norm_fp32).contiguous()
    scale = (
        torch.tensor(RELATIVE_L2_DOSE, dtype=torch.float32, device=base.device)
        * base_norm_fp32
    )
    delta = (scale * direction).contiguous()
    plus = (base + delta).contiguous()
    minus = (base - delta).contiguous()
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (direction, delta, plus, minus)
    ):
        raise PQMosaicProjectionError(
            "projected relative-L2 intervention is non-finite"
        )
    if not torch.equal(base, base_snapshot) or not torch.equal(base_input, base_snapshot):
        raise PQMosaicProjectionError("relative-L2 construction mutated the base")

    expected_direction = (projected / projected_norm_fp32).contiguous()
    expected_delta = (scale * expected_direction).contiguous()
    expected_plus = (base + expected_delta).contiguous()
    expected_minus = (base - expected_delta).contiguous()
    if not (
        torch.equal(direction, expected_direction)
        and torch.equal(delta, expected_delta)
        and torch.equal(plus, expected_plus)
        and torch.equal(minus, expected_minus)
    ):
        raise PQMosaicProjectionError(
            "project-then-normalize FP32 formula did not recompute exactly"
        )

    base_norm = _l2_norm(base)
    direction_norm = _l2_norm(direction)
    plus_delta = plus.to(torch.float64) - base.to(torch.float64)
    minus_delta = minus.to(torch.float64) - base.to(torch.float64)
    plus_delta_norm = _l2_norm(plus_delta)
    minus_delta_norm = _l2_norm(minus_delta)
    observed_relative_dose = _l2_norm(delta) / base_norm
    midpoint_error = float(
        (
            (plus.to(torch.float64) + minus.to(torch.float64)) * 0.5
            - base.to(torch.float64)
        )
        .abs()
        .max()
        .item()
    )
    antisymmetry_error = float((plus_delta + minus_delta).abs().max().item())
    norm_symmetry_error = abs(plus_delta_norm - minus_delta_norm)
    scale_value = float(scale.item())
    tolerance = max(2.0e-6 * max(scale_value, 1.0), 2.0e-7)
    if (
        not math.isclose(
            observed_relative_dose,
            RELATIVE_L2_DOSE,
            rel_tol=_FP32_NORM_RELATIVE_TOLERANCE,
            abs_tol=5.0e-7,
        )
        or not math.isclose(
            direction_norm,
            1.0,
            rel_tol=_FP32_NORM_RELATIVE_TOLERANCE,
            abs_tol=5.0e-6,
        )
        or midpoint_error > tolerance
        or antisymmetry_error > 2.0 * tolerance
        or norm_symmetry_error > 2.0e-6 * max(scale_value, 1.0)
    ):
        raise PQMosaicProjectionError(
            "projected symmetric relative-L2 construction failed closed"
        )

    base_binding = _bind_tensor(base, label="base clean latent")
    unit_direction_binding = _bind_tensor(
        direction, label="unit projected direction"
    )
    delta_binding = _bind_tensor(delta, label="projected delta")
    plus_binding = _bind_tensor(plus, label="plus clean latent")
    minus_binding = _bind_tensor(minus, label="minus clean latent")
    _assert_no_storage_aliases(
        (
            ("projected_clean_latent_vjp", projection._projected_binding),
            ("base_clean_latent", base_binding),
            ("unit_projected_direction", unit_direction_binding),
            ("projected_delta", delta_binding),
            ("plus_clean_latent", plus_binding),
            ("minus_clean_latent", minus_binding),
        )
    )

    return PQMosaicSymmetricLatents(
        base=base,
        plus=plus,
        minus=minus,
        delta=delta,
        unit_direction=direction,
        projection=projection,
        base_l2_norm=base_norm,
        projected_vjp_l2_norm_fp32=float(projected_norm_fp32.item()),
        direction_l2_norm=direction_norm,
        absolute_dose_l2_fp32_scale=scale_value,
        plus_delta_l2=plus_delta_norm,
        minus_delta_l2=minus_delta_norm,
        observed_relative_l2_dose=observed_relative_dose,
        midpoint_max_abs_error=midpoint_error,
        delta_antisymmetry_max_abs_error=antisymmetry_error,
        delta_norm_symmetry_absolute_error=norm_symmetry_error,
        symmetry_tolerance=tolerance,
        _base_binding=base_binding,
        _unit_direction_binding=unit_direction_binding,
        _delta_binding=delta_binding,
        _plus_binding=plus_binding,
        _minus_binding=minus_binding,
    )


__all__ = [
    "INTERVENTION_SCHEMA_VERSION",
    "LATENT_CHANNELS",
    "LATENT_PHASES",
    "MINIMUM_PROJECTION_SURVIVAL_RATIO",
    "PQMosaicNuisanceNullProjection",
    "PQMosaicProjectionError",
    "PQMosaicSymmetricLatents",
    "PROJECTION_SCHEMA_VERSION",
    "RELATIVE_L2_DOSE",
    "construct_projected_symmetric_latents",
    "project_raw_clean_latent_vjp",
]

#!/usr/bin/env python3
"""Robust fixed-grid camera consensus for Bernini clean fields.

The existing global camera-tangent projector solves one least-squares problem
over the complete frame.  A spatially local actor motion can consequently
receive a non-zero global camera coefficient merely because it correlates
with a homography tangent image.  This prototype changes the estimator, not
the camera model:

1. split the latent image into a deterministic rectangular grid;
2. solve the same eight *physical* infinitesimal-homography coefficients in
   every tile, always using global normalized ``[-1, 1]`` coordinates;
3. reject poorly fitted tiles, then form a robust global consensus with a
   coordinate median, MAD rejection, and coordinate-wise trimmed mean; and
4. render one global camera tangent only when the inliers are a strict
   majority with support across every row/column, all quadrants, and at least
   three corners.

No runtime API accepts a spatial side condition.  A texture/rank failure, an
insufficient or spatially concentrated support, excessive disagreement, or
non-finite input fails closed: invalid phases receive an exact zero camera
component.  Operations explicitly disable ambient autocast.

Tensor operations require exact FP32 ``[B,C,21,H,W]`` tensors.  PyTorch is
imported lazily so the numerical contract remains inspectable on orchestration
hosts without the training environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


METHOD_NAME = "fixed-grid-robust-camera-consensus"
SCHEMA_VERSION = "bernini-fixed-grid-camera-consensus-v1"
EXPECTED_LATENT_PHASES = 21
HOMOGRAPHY_DOF = 8


class CameraConsensusError(RuntimeError):
    """Raised before an invalid consensus operation reaches a scheduler."""


@dataclass(frozen=True)
class CameraConsensusConfig:
    """Numerical and evidence policy for fixed-grid camera consensus."""

    tile_rows: int = 4
    tile_columns: int = 4
    relative_rank_cutoff: float = 1.0e-6
    absolute_rank_cutoff: float = 1.0e-7
    max_condition_number: float = 1.0e6
    minimum_valid_tile_fraction: float = 0.625
    minimum_valid_tiles: int = 6
    mad_floor: float = 1.0e-4
    mad_z_threshold: float = 3.5
    trim_fraction: float = 0.10
    maximum_tile_coefficient: float = 1.0
    maximum_tile_relative_fit_residual: float = 0.50
    fit_energy_floor: float = 1.0e-12
    maximum_consensus_coefficient: float = 0.35
    maximum_consensus_mad: float = 0.01
    relative_consensus_mad: float = 0.25
    minimum_inliers_per_row: int = 1
    minimum_inliers_per_column: int = 1
    minimum_inliers_per_quadrant: int = 2
    minimum_corner_inliers: int = 3

    def validate(self) -> None:
        for name in (
            "tile_rows",
            "tile_columns",
            "minimum_valid_tiles",
            "minimum_inliers_per_row",
            "minimum_inliers_per_column",
            "minimum_inliers_per_quadrant",
            "minimum_corner_inliers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise CameraConsensusError(f"{name} must be a positive integer")

        if int(self.tile_rows) < 2 or int(self.tile_columns) < 2:
            raise CameraConsensusError(
                "fixed-grid spatial coverage requires at least two rows and columns"
            )

        nonnegative = (
            "relative_rank_cutoff",
            "absolute_rank_cutoff",
            "mad_floor",
            "trim_fraction",
            "maximum_consensus_mad",
            "relative_consensus_mad",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise CameraConsensusError(f"{name} must be finite and nonnegative")

        positive = (
            "max_condition_number",
            "mad_z_threshold",
            "maximum_tile_coefficient",
            "maximum_tile_relative_fit_residual",
            "fit_energy_floor",
            "maximum_consensus_coefficient",
        )
        for name in positive:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise CameraConsensusError(f"{name} must be finite and positive")

        if float(self.max_condition_number) <= 1.0:
            raise CameraConsensusError("max_condition_number must exceed one")
        if (
            isinstance(self.minimum_valid_tile_fraction, bool)
            or not math.isfinite(float(self.minimum_valid_tile_fraction))
            or not 0.5 < float(self.minimum_valid_tile_fraction) <= 1.0
        ):
            raise CameraConsensusError(
                "minimum_valid_tile_fraction must be in (0.5,1]"
            )
        if not 0.0 <= float(self.trim_fraction) < 0.5:
            raise CameraConsensusError("trim_fraction must be in [0,0.5)")
        tile_count = int(self.tile_rows) * int(self.tile_columns)
        if int(self.minimum_valid_tiles) > tile_count:
            raise CameraConsensusError(
                "minimum_valid_tiles cannot exceed the fixed-grid tile count"
            )
        if int(self.minimum_inliers_per_row) > int(self.tile_columns):
            raise CameraConsensusError(
                "minimum_inliers_per_row exceeds the number of columns"
            )
        if int(self.minimum_inliers_per_column) > int(self.tile_rows):
            raise CameraConsensusError(
                "minimum_inliers_per_column exceeds the number of rows"
            )
        smallest_quadrant = (int(self.tile_rows) // 2) * (
            int(self.tile_columns) // 2
        )
        if int(self.minimum_inliers_per_quadrant) > smallest_quadrant:
            raise CameraConsensusError(
                "minimum_inliers_per_quadrant exceeds the smallest quadrant"
            )
        if int(self.minimum_corner_inliers) > 4:
            raise CameraConsensusError("minimum_corner_inliers cannot exceed four")


def _config_signature(config: CameraConsensusConfig) -> tuple[float | int, ...]:
    _validate_config(config)
    return (
        int(config.tile_rows),
        int(config.tile_columns),
        float(config.relative_rank_cutoff),
        float(config.absolute_rank_cutoff),
        float(config.max_condition_number),
        float(config.minimum_valid_tile_fraction),
        int(config.minimum_valid_tiles),
        float(config.mad_floor),
        float(config.mad_z_threshold),
        float(config.trim_fraction),
        float(config.maximum_tile_coefficient),
        float(config.maximum_tile_relative_fit_residual),
        float(config.fit_energy_floor),
        float(config.maximum_consensus_coefficient),
        float(config.maximum_consensus_mad),
        float(config.relative_consensus_mad),
        int(config.minimum_inliers_per_row),
        int(config.minimum_inliers_per_column),
        int(config.minimum_inliers_per_quadrant),
        int(config.minimum_corner_inliers),
    )


@dataclass(frozen=True)
class FixedGridCameraGeometry:
    """Detached source geometry and per-tile physical inverse problems."""

    # [B,21,C*H*W,8], with columns ordered (a,b,c,d,e,f,g,h).
    tangent_matrix: Any
    # Each index tensor addresses the C*H*W dimension in tangent_matrix.
    tile_indices: tuple[Any, ...]
    # One [B,21,8,N_tile] physical-coefficient pseudoinverse per tile.
    tile_pseudoinverses: tuple[Any, ...]
    singular_values: Any
    valid_tiles: Any
    tile_bounds: tuple[tuple[int, int, int, int], ...]
    source_shape: tuple[int, int, int, int, int]
    source_device: Any
    source_dtype: Any
    source_data_ptr: int
    source_version: int
    config_signature: tuple[float | int, ...]
    source_was_detached: bool
    source_identity: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CameraConsensusProjection:
    """A camera component plus auditable robust-consensus evidence."""

    camera_component: Any
    consensus_coefficients: Any
    per_tile_coefficients: Any
    per_tile_relative_fit_residual: Any
    geometry_valid_tiles: Any
    bounded_valid_tiles: Any
    fit_valid_tiles: Any
    inlier_tiles: Any
    spatial_coverage_valid: Any
    consensus_valid: Any
    initial_median: Any
    initial_mad: Any
    final_mad: Any
    required_tile_count: int

    def to_receipt(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD_NAME,
                "consensus_valid": self.consensus_valid,
                "geometry_valid_tiles": self.geometry_valid_tiles,
                "bounded_valid_tiles": self.bounded_valid_tiles,
                "inlier_tiles": self.inlier_tiles,
                "consensus_coefficients": self.consensus_coefficients,
                "per_tile_relative_fit_residual": (
                    self.per_tile_relative_fit_residual
                ),
                "initial_median": self.initial_median,
                "initial_mad": self.initial_mad,
                "final_mad": self.final_mad,
                "fit_valid_tiles": self.fit_valid_tiles,
                "spatial_coverage_valid": self.spatial_coverage_valid,
                "required_tile_count": self.required_tile_count,
            }
        )


def camera_consensus_contract_receipt(
    config: CameraConsensusConfig = CameraConsensusConfig(),
) -> dict[str, Any]:
    """Return the static, side-condition-free numerical contract."""

    _validate_config(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "tensor_contract": {
            "layout": "B,C,T,H,W",
            "dtype": "float32",
            "latent_phases": EXPECTED_LATENT_PHASES,
        },
        "grid": {
            "rows": int(config.tile_rows),
            "columns": int(config.tile_columns),
            "coordinates": "fixed_global_normalized_minus_one_to_one",
            "partition": "deterministic_integer_edges",
        },
        "homography": {
            "degrees_of_freedom": HOMOGRAPHY_DOF,
            "physical_coefficients": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "tile_solve": "float32_thin_svd_full_rank_only",
        },
        "aggregation": ["coordinate_median", "scaled_MAD_rejection", "trimmed_mean"],
        "tile_evidence": {
            "maximum_relative_fit_residual": float(
                config.maximum_tile_relative_fit_residual
            ),
            "fit_energy_floor": float(config.fit_energy_floor),
        },
        "spatial_support": {
            "minimum_inliers_per_row": int(config.minimum_inliers_per_row),
            "minimum_inliers_per_column": int(
                config.minimum_inliers_per_column
            ),
            "minimum_inliers_per_quadrant": int(
                config.minimum_inliers_per_quadrant
            ),
            "minimum_corner_inliers": int(config.minimum_corner_inliers),
        },
        "failure_policy": "exact_zero_without_strict_cross_tile_consensus",
        "runtime_inputs": ["reference_clean_field", "field", "config_or_geometry"],
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
    raise CameraConsensusError("receipt contains a non-serializable value")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise CameraConsensusError(
            "fixed-grid camera consensus requires PyTorch"
        ) from error
    return torch


def _validate_config(config: Any) -> None:
    if type(config) is not CameraConsensusConfig:
        raise CameraConsensusError("config must be an exact CameraConsensusConfig")
    config.validate()


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


def _spatial_gradients(reference: Any) -> tuple[Any, Any]:
    """Finite differences expressed per normalized-coordinate unit."""

    torch = _require_torch()
    gradient_x = torch.zeros_like(reference)
    gradient_y = torch.zeros_like(reference)
    height = int(reference.shape[-2])
    width = int(reference.shape[-1])
    if width > 1:
        gradient_x[..., 0] = reference[..., 1] - reference[..., 0]
        gradient_x[..., -1] = reference[..., -1] - reference[..., -2]
        if width > 2:
            gradient_x[..., 1:-1] = 0.5 * (
                reference[..., 2:] - reference[..., :-2]
            )
    if height > 1:
        gradient_y[..., 0, :] = reference[..., 1, :] - reference[..., 0, :]
        gradient_y[..., -1, :] = reference[..., -1, :] - reference[..., -2, :]
        if height > 2:
            gradient_y[..., 1:-1, :] = 0.5 * (
                reference[..., 2:, :] - reference[..., :-2, :]
            )
    gradient_x.mul_(0.5 * float(width - 1))
    gradient_y.mul_(0.5 * float(height - 1))
    return gradient_x, gradient_y


def _global_tangent_matrix(reference: Any) -> Any:
    """Build global-coordinate physical homography tangents."""

    torch = _require_torch()
    gradient_x, gradient_y = _spatial_gradients(reference)
    height = int(reference.shape[-2])
    width = int(reference.shape[-1])
    y_axis = torch.linspace(
        -1.0, 1.0, height, dtype=torch.float32, device=reference.device
    )
    x_axis = torch.linspace(
        -1.0, 1.0, width, dtype=torch.float32, device=reference.device
    )
    y_coord, x_coord = torch.meshgrid(y_axis, x_axis, indexing="ij")
    x_coord = x_coord.reshape(1, 1, 1, height, width)
    y_coord = y_coord.reshape(1, 1, 1, height, width)
    tangent = torch.stack(
        (
            gradient_x * x_coord,
            gradient_x * y_coord,
            gradient_x,
            gradient_y * x_coord,
            gradient_y * y_coord,
            gradient_y,
            -gradient_x * x_coord.square() - gradient_y * x_coord * y_coord,
            -gradient_x * x_coord * y_coord - gradient_y * y_coord.square(),
        ),
        dim=-1,
    )
    batch, channels, phases, _, _, _ = tangent.shape
    return tangent.permute(0, 2, 1, 3, 4, 5).reshape(
        batch, phases, channels * height * width, HOMOGRAPHY_DOF
    )


def _integer_edges(length: int, sections: int) -> tuple[int, ...]:
    return tuple((index * length) // sections for index in range(sections + 1))


def _fixed_tile_indices(
    *,
    channels: int,
    height: int,
    width: int,
    rows: int,
    columns: int,
    device: Any,
) -> tuple[tuple[Any, ...], tuple[tuple[int, int, int, int], ...]]:
    torch = _require_torch()
    if rows > height or columns > width:
        raise CameraConsensusError(
            "fixed grid cannot have more rows/columns than the latent image"
        )
    y_edges = _integer_edges(height, rows)
    x_edges = _integer_edges(width, columns)
    plane = torch.arange(height * width, dtype=torch.int64, device=device).reshape(
        height, width
    )
    offsets = torch.arange(channels, dtype=torch.int64, device=device) * (
        height * width
    )
    indices = []
    bounds = []
    for row in range(rows):
        for column in range(columns):
            y0, y1 = y_edges[row], y_edges[row + 1]
            x0, x1 = x_edges[column], x_edges[column + 1]
            spatial = plane[y0:y1, x0:x1].reshape(-1)
            tile = (offsets[:, None] + spatial[None, :]).reshape(-1)
            indices.append(tile)
            bounds.append((y0, y1, x0, x1))
    return tuple(indices), tuple(bounds)


def _build_fixed_grid_camera_geometry_fp32(
    reference_clean_field: Any,
    config: CameraConsensusConfig = CameraConsensusConfig(),
) -> FixedGridCameraGeometry:
    """Precompute full-rank per-tile physical homography solvers."""

    torch = _require_torch()
    _validate_config(config)
    _validate_field(reference_clean_field, label="reference_clean_field")
    reference = reference_clean_field.detach()
    batch, channels, _, height, width = (
        int(value) for value in reference.shape
    )
    matrix = _global_tangent_matrix(reference)
    tile_indices, tile_bounds = _fixed_tile_indices(
        channels=channels,
        height=height,
        width=width,
        rows=int(config.tile_rows),
        columns=int(config.tile_columns),
        device=reference.device,
    )

    pseudoinverses = []
    singular_values = []
    valid_tiles = []
    for index in tile_indices:
        tile_matrix = matrix.index_select(-2, index)
        if int(tile_matrix.shape[-2]) < HOMOGRAPHY_DOF:
            singular = torch.zeros(
                batch,
                EXPECTED_LATENT_PHASES,
                HOMOGRAPHY_DOF,
                dtype=torch.float32,
                device=reference.device,
            )
            valid = torch.zeros(
                batch,
                EXPECTED_LATENT_PHASES,
                dtype=torch.bool,
                device=reference.device,
            )
            pseudoinverse = torch.zeros(
                batch,
                EXPECTED_LATENT_PHASES,
                HOMOGRAPHY_DOF,
                int(tile_matrix.shape[-2]),
                dtype=torch.float32,
                device=reference.device,
            )
        else:
            try:
                left, singular, right_h = torch.linalg.svd(
                    tile_matrix, full_matrices=False
                )
            except RuntimeError as error:
                raise CameraConsensusError("FP32 per-tile SVD failed") from error
            maximum = singular[..., :1]
            cutoff = torch.maximum(
                torch.full_like(maximum, float(config.absolute_rank_cutoff)),
                maximum * float(config.relative_rank_cutoff),
            )
            retained = singular > cutoff
            full_rank = retained.sum(dim=-1) == HOMOGRAPHY_DOF
            smallest = singular[..., -1].clamp_min(
                float(config.absolute_rank_cutoff)
            )
            condition = singular[..., 0] / smallest
            valid = full_rank & torch.isfinite(condition) & (
                condition <= float(config.max_condition_number)
            )
            reciprocal = torch.where(
                retained,
                singular.clamp_min(float(config.absolute_rank_cutoff)).reciprocal(),
                torch.zeros_like(singular),
            )
            pseudoinverse = torch.matmul(
                right_h.transpose(-2, -1) * reciprocal.unsqueeze(-2),
                left.transpose(-2, -1),
            )
            pseudoinverse = pseudoinverse * valid[..., None, None].to(
                dtype=torch.float32
            )
        pseudoinverses.append(pseudoinverse.detach())
        singular_values.append(singular.detach())
        valid_tiles.append(valid.detach())

    return FixedGridCameraGeometry(
        tangent_matrix=matrix.detach(),
        tile_indices=tile_indices,
        tile_pseudoinverses=tuple(pseudoinverses),
        singular_values=torch.stack(singular_values, dim=2),
        valid_tiles=torch.stack(valid_tiles, dim=2),
        tile_bounds=tile_bounds,
        source_shape=tuple(int(value) for value in reference.shape),
        source_device=reference.device,
        source_dtype=reference.dtype,
        source_data_ptr=int(reference_clean_field.data_ptr()),
        source_version=int(reference_clean_field._version),
        config_signature=_config_signature(config),
        source_was_detached=True,
        source_identity=reference_clean_field,
    )


def build_fixed_grid_camera_geometry(
    reference_clean_field: Any,
    config: CameraConsensusConfig = CameraConsensusConfig(),
) -> FixedGridCameraGeometry:
    """Build geometry with autocast disabled so every solver stays FP32."""

    torch = _require_torch()
    _validate_field(reference_clean_field, label="reference_clean_field")
    with torch.autocast(
        device_type=reference_clean_field.device.type,
        enabled=False,
    ):
        return _build_fixed_grid_camera_geometry_fp32(
            reference_clean_field,
            config,
        )


def validate_fixed_grid_camera_geometry(
    geometry: Any,
    reference_clean_field: Any,
    config: CameraConsensusConfig = CameraConsensusConfig(),
) -> FixedGridCameraGeometry:
    """Reject stale or structurally incompatible precomputed geometry."""

    torch = _require_torch()
    _validate_config(config)
    _validate_field(reference_clean_field, label="reference_clean_field")
    if type(geometry) is not FixedGridCameraGeometry:
        raise CameraConsensusError(
            "precomputed_geometry must be an exact FixedGridCameraGeometry"
        )
    if geometry.source_identity is not reference_clean_field:
        raise CameraConsensusError("precomputed geometry source identity differs")
    if tuple(geometry.source_shape) != tuple(reference_clean_field.shape):
        raise CameraConsensusError("precomputed geometry source shape differs")
    if geometry.source_device != reference_clean_field.device:
        raise CameraConsensusError("precomputed geometry source device differs")
    if geometry.source_dtype != reference_clean_field.dtype:
        raise CameraConsensusError("precomputed geometry source dtype differs")
    if int(geometry.source_data_ptr) != int(reference_clean_field.data_ptr()):
        raise CameraConsensusError("precomputed geometry source storage differs")
    if int(geometry.source_version) != int(reference_clean_field._version):
        raise CameraConsensusError("precomputed geometry source was modified in place")
    if tuple(geometry.config_signature) != _config_signature(config):
        raise CameraConsensusError("precomputed geometry config differs")
    if geometry.source_was_detached is not True:
        raise CameraConsensusError("precomputed geometry lacks detach evidence")

    batch, channels, phases, height, width = geometry.source_shape
    tile_count = int(config.tile_rows) * int(config.tile_columns)
    contracts = (
        (
            geometry.tangent_matrix,
            (batch, phases, channels * height * width, HOMOGRAPHY_DOF),
            torch.float32,
            "tangent_matrix",
        ),
        (
            geometry.singular_values,
            (batch, phases, tile_count, HOMOGRAPHY_DOF),
            torch.float32,
            "singular_values",
        ),
        (
            geometry.valid_tiles,
            (batch, phases, tile_count),
            torch.bool,
            "valid_tiles",
        ),
    )
    for tensor, shape, dtype, label in contracts:
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != reference_clean_field.device
            or bool(tensor.requires_grad)
        ):
            raise CameraConsensusError(
                f"precomputed geometry {label} violates its tensor contract"
            )
    if len(geometry.tile_indices) != tile_count or len(
        geometry.tile_pseudoinverses
    ) != tile_count:
        raise CameraConsensusError("precomputed geometry tile count differs")
    if not bool(torch.isfinite(geometry.tangent_matrix).all()) or not bool(
        torch.isfinite(geometry.singular_values).all()
    ):
        raise CameraConsensusError("precomputed geometry contains non-finite values")
    return geometry


def _required_tile_count(config: CameraConsensusConfig) -> int:
    tile_count = int(config.tile_rows) * int(config.tile_columns)
    fraction_count = math.ceil(
        float(config.minimum_valid_tile_fraction) * tile_count
    )
    return max(int(config.minimum_valid_tiles), fraction_count)


def _aggregate_consensus(
    coefficients: Any,
    valid_tiles: Any,
    config: CameraConsensusConfig,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Median/MAD reject, trimmed aggregate, and fail closed per phase."""

    torch = _require_torch()
    batch, phases, tile_count, _ = coefficients.shape
    consensus = torch.zeros(
        batch, phases, HOMOGRAPHY_DOF, dtype=torch.float32, device=coefficients.device
    )
    consensus_valid = torch.zeros(
        batch, phases, dtype=torch.bool, device=coefficients.device
    )
    inliers = torch.zeros_like(valid_tiles)
    spatial_coverage_valid = torch.zeros_like(consensus_valid)
    initial_median = torch.zeros_like(consensus)
    initial_mad = torch.zeros_like(consensus)
    final_mad = torch.zeros_like(consensus)
    required = _required_tile_count(config)

    for batch_index in range(batch):
        for phase_index in range(phases):
            valid_index = torch.nonzero(
                valid_tiles[batch_index, phase_index], as_tuple=False
            ).flatten()
            if int(valid_index.numel()) < required:
                continue
            values = coefficients[batch_index, phase_index].index_select(
                0, valid_index
            )
            median = values.median(dim=0).values
            absolute_deviation = (values - median).abs()
            mad = 1.4826 * absolute_deviation.median(dim=0).values
            scale = mad.clamp_min(float(config.mad_floor))
            # A tile is a coherent physical hypothesis: disagreement in any
            # one of its eight coefficients invalidates that tile rather than
            # being diluted by seven matching coordinates.
            robust_distance = (absolute_deviation / scale).amax(dim=-1)
            local_inlier = robust_distance <= float(config.mad_z_threshold)
            kept_index = valid_index[local_inlier]
            initial_median[batch_index, phase_index] = median
            initial_mad[batch_index, phase_index] = mad
            inliers[batch_index, phase_index, kept_index] = True
            if int(kept_index.numel()) < required:
                continue

            tile_rows = int(config.tile_rows)
            tile_columns = int(config.tile_columns)
            kept_rows = torch.div(
                kept_index,
                tile_columns,
                rounding_mode="floor",
            )
            kept_columns = kept_index.remainder(tile_columns)
            row_counts = torch.bincount(kept_rows, minlength=tile_rows)
            column_counts = torch.bincount(
                kept_columns,
                minlength=tile_columns,
            )
            row_support = bool(
                (
                    row_counts
                    >= int(config.minimum_inliers_per_row)
                ).all()
            )
            column_support = bool(
                (
                    column_counts
                    >= int(config.minimum_inliers_per_column)
                ).all()
            )
            row_half = tile_rows // 2
            column_half = tile_columns // 2
            quadrant_counts = []
            for lower_row, upper_row in ((0, row_half), (row_half, tile_rows)):
                for lower_column, upper_column in (
                    (0, column_half),
                    (column_half, tile_columns),
                ):
                    quadrant_counts.append(
                        int(
                            (
                                (kept_rows >= lower_row)
                                & (kept_rows < upper_row)
                                & (kept_columns >= lower_column)
                                & (kept_columns < upper_column)
                            ).sum()
                        )
                    )
            quadrant_support = min(quadrant_counts) >= int(
                config.minimum_inliers_per_quadrant
            )
            corner_indices = torch.tensor(
                (
                    0,
                    tile_columns - 1,
                    (tile_rows - 1) * tile_columns,
                    tile_rows * tile_columns - 1,
                ),
                dtype=kept_index.dtype,
                device=kept_index.device,
            )
            corner_count = int(
                (kept_index[:, None] == corner_indices[None, :]).any(dim=0).sum()
            )
            corner_support = corner_count >= int(config.minimum_corner_inliers)
            coverage = (
                row_support
                and column_support
                and quadrant_support
                and corner_support
            )
            spatial_coverage_valid[batch_index, phase_index] = coverage
            if not coverage:
                continue

            kept = coefficients[batch_index, phase_index].index_select(
                0, kept_index
            )
            ordered = kept.sort(dim=0).values
            trim = int(math.floor(float(config.trim_fraction) * len(kept)))
            if trim > 0 and 2 * trim < len(kept):
                ordered = ordered[trim:-trim]
            estimate = ordered.mean(dim=0)
            dispersion = 1.4826 * (kept - estimate).abs().median(dim=0).values
            allowed_dispersion = float(config.maximum_consensus_mad) + float(
                config.relative_consensus_mad
            ) * estimate.abs()
            stable = bool((dispersion <= allowed_dispersion).all())
            bounded = bool(
                (estimate.abs() <= float(config.maximum_consensus_coefficient)).all()
            )
            finite = bool(torch.isfinite(estimate).all()) and bool(
                torch.isfinite(dispersion).all()
            )
            final_mad[batch_index, phase_index] = dispersion
            if stable and bounded and finite:
                consensus[batch_index, phase_index] = estimate
                consensus_valid[batch_index, phase_index] = True

    return (
        consensus,
        consensus_valid,
        inliers,
        initial_median,
        initial_mad,
        final_mad,
        spatial_coverage_valid,
    )


def _project_camera_consensus_fp32(
    field: Any,
    reference_clean_field: Any,
    *,
    config: CameraConsensusConfig = CameraConsensusConfig(),
    precomputed_geometry: FixedGridCameraGeometry | None = None,
) -> CameraConsensusProjection:
    """Estimate and render only a strict cross-tile camera consensus.

    ``field`` is a clean-field difference expressed in the reference image's
    tangent coordinates.  The function accepts no external spatial support.
    """

    torch = _require_torch()
    _validate_config(config)
    _validate_field(field, label="field")
    _validate_field(reference_clean_field, label="reference_clean_field")
    if tuple(field.shape) != tuple(reference_clean_field.shape):
        raise CameraConsensusError("field and reference_clean_field shapes differ")
    if field.device != reference_clean_field.device:
        raise CameraConsensusError("field and reference_clean_field devices differ")

    if precomputed_geometry is None:
        geometry = build_fixed_grid_camera_geometry(reference_clean_field, config)
    else:
        geometry = validate_fixed_grid_camera_geometry(
            precomputed_geometry, reference_clean_field, config
        )

    batch, channels, phases, height, width = (
        int(value) for value in field.shape
    )
    flattened = field.permute(0, 2, 1, 3, 4).reshape(
        batch, phases, channels * height * width
    )
    tile_coefficients = []
    tile_relative_fit_residuals = []
    for index, pseudoinverse in zip(
        geometry.tile_indices, geometry.tile_pseudoinverses
    ):
        tile_field = flattened.index_select(-1, index)
        coefficient = torch.matmul(
            pseudoinverse, tile_field.unsqueeze(-1)
        ).squeeze(-1)
        tile_coefficients.append(coefficient)
        tile_matrix = geometry.tangent_matrix.index_select(-2, index)
        fitted = torch.matmul(
            tile_matrix,
            coefficient.unsqueeze(-1),
        ).squeeze(-1)
        residual_norm = (tile_field - fitted).square().sum(dim=-1).sqrt()
        field_norm = tile_field.square().sum(dim=-1).sqrt()
        fit_floor = float(config.fit_energy_floor)
        relative_fit = residual_norm / field_norm.clamp_min(fit_floor)
        relative_fit = torch.where(
            field_norm <= fit_floor,
            torch.zeros_like(relative_fit),
            relative_fit,
        )
        tile_relative_fit_residuals.append(relative_fit)
    coefficients = torch.stack(tile_coefficients, dim=2)
    relative_fit_residual = torch.stack(tile_relative_fit_residuals, dim=2)
    bounded = torch.isfinite(coefficients).all(dim=-1) & (
        coefficients.abs().amax(dim=-1)
        <= float(config.maximum_tile_coefficient)
    )
    bounded_valid_tiles = geometry.valid_tiles & bounded
    fit_valid_tiles = bounded_valid_tiles & torch.isfinite(
        relative_fit_residual
    ) & (
        relative_fit_residual
        <= float(config.maximum_tile_relative_fit_residual)
    )
    (
        consensus,
        consensus_valid,
        inliers,
        initial_median,
        initial_mad,
        final_mad,
        spatial_coverage_valid,
    ) = _aggregate_consensus(coefficients, fit_valid_tiles, config)

    camera_flat = torch.matmul(
        geometry.tangent_matrix, consensus.unsqueeze(-1)
    ).squeeze(-1)
    camera_component = camera_flat.reshape(
        batch, phases, channels, height, width
    ).permute(0, 2, 1, 3, 4)
    # This assertion protects the fail-closed promise from a future renderer
    # change that might add a bias independent of the zero coefficients.
    invalid_component = camera_component.permute(0, 2, 1, 3, 4)[
        ~consensus_valid
    ]
    if bool(invalid_component.numel()) and not bool(
        torch.equal(invalid_component, torch.zeros_like(invalid_component))
    ):
        raise CameraConsensusError("invalid consensus did not render exact zero")

    return CameraConsensusProjection(
        camera_component=camera_component,
        consensus_coefficients=consensus,
        per_tile_coefficients=coefficients,
        per_tile_relative_fit_residual=relative_fit_residual,
        geometry_valid_tiles=geometry.valid_tiles,
        bounded_valid_tiles=bounded_valid_tiles,
        fit_valid_tiles=fit_valid_tiles,
        inlier_tiles=inliers,
        spatial_coverage_valid=spatial_coverage_valid,
        consensus_valid=consensus_valid,
        initial_median=initial_median,
        initial_mad=initial_mad,
        final_mad=final_mad,
        required_tile_count=_required_tile_count(config),
    )


def project_camera_consensus(
    field: Any,
    reference_clean_field: Any,
    *,
    config: CameraConsensusConfig = CameraConsensusConfig(),
    precomputed_geometry: FixedGridCameraGeometry | None = None,
) -> CameraConsensusProjection:
    """Project with autocast disabled under CPU, CUDA, or ROCm execution."""

    torch = _require_torch()
    _validate_field(field, label="field")
    _validate_field(reference_clean_field, label="reference_clean_field")
    with torch.autocast(
        device_type=reference_clean_field.device.type,
        enabled=False,
    ):
        return _project_camera_consensus_fp32(
            field,
            reference_clean_field,
            config=config,
            precomputed_geometry=precomputed_geometry,
        )


__all__ = [
    "CameraConsensusConfig",
    "CameraConsensusError",
    "CameraConsensusProjection",
    "FixedGridCameraGeometry",
    "build_fixed_grid_camera_geometry",
    "camera_consensus_contract_receipt",
    "project_camera_consensus",
    "validate_fixed_grid_camera_geometry",
]

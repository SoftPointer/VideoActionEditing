"""Checkpoint-independent geometry descriptors for action retrieval.

These descriptors are intentionally not called Motive fingerprints.  They are
cheap pre-filtering, clustering, and distillation targets.  The paper's actual
fingerprint is a projected generator gradient and is implemented separately in
``attribution.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescriptorConfig:
    temporal_bins: int = 4
    grid_rows: int = 2
    grid_cols: int = 2
    orientation_bins: int = 8
    active_speed_threshold: float = 0.005
    minimum_active_fraction: float = 0.001
    magnitude_clip_percentile: float = 95.0
    eps: float = 1e-8

    def validate(self) -> None:
        for name in (
            "temporal_bins",
            "grid_rows",
            "grid_cols",
            "orientation_bins",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.minimum_active_fraction <= 1.0:
            raise ValueError("minimum_active_fraction must be in [0, 1]")


def _histogram_of_oriented_flow(
    flow: np.ndarray,
    weight_clip: float,
    bins: int,
    active_speed_threshold: float,
    eps: float,
) -> np.ndarray:
    x_component = flow[..., 0]
    y_component = flow[..., 1]
    magnitude = np.linalg.norm(flow, axis=-1)
    active = magnitude >= active_speed_threshold
    weights = np.where(active, np.minimum(magnitude, weight_clip), 0.0)
    angles = np.mod(np.arctan2(y_component, x_component), 2.0 * np.pi)
    histogram, _ = np.histogram(
        angles,
        bins=bins,
        range=(0.0, 2.0 * np.pi),
        weights=weights,
    )
    histogram = histogram.astype(np.float32)
    weight_sum = float(np.sum(histogram))
    if weight_sum <= eps:
        return np.zeros_like(histogram)
    # A separately normalized HOOF would turn arbitrarily small tracker noise,
    # or one isolated active pixel, into a unit-strength direction feature.
    # Reliability scaling retains direction while down-weighting tiny support.
    active_support = float(np.mean(active))
    return (histogram / weight_sum) * np.sqrt(active_support)


def _safe_quantiles(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return [0.0 for _ in quantiles]
    return [float(value) for value in np.quantile(values, quantiles)]


def _time_derivative_stats(values: np.ndarray, eps: float) -> list[float]:
    if len(values) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    first = np.diff(values)
    second = np.diff(first) if len(first) >= 2 else np.zeros(1, dtype=np.float32)
    return [
        float(np.mean(np.abs(first))),
        float(np.max(np.abs(first))),
        float(np.mean(np.abs(second))),
        float(np.std(first) / (float(np.mean(np.abs(first))) + eps)),
    ]


def encode_action_descriptor(
    residual_flows: np.ndarray,
    frame_times: np.ndarray,
    frame_width: int,
    *,
    global_flows: np.ndarray | None = None,
    config: DescriptorConfig | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Encode direction, spatial support, rhythm, and camera motion.

    Args:
        residual_flows: Camera-compensated flow with shape ``[T, H, W, 2]``.
        frame_times: Sampled frame timestamps with shape ``[T + 1]``.
        frame_width: Width at which flow was estimated.
        global_flows: Optional estimated camera flow with the same shape.

    Returns:
        A 1-D float32 descriptor. By default it is unit normalized for legacy
        retrieval. Pass ``normalize=False`` when actor and camera components
        will be factorized before source→target differencing.
    """

    config = config or DescriptorConfig()
    config.validate()
    flows = np.asarray(residual_flows, dtype=np.float32)
    times = np.asarray(frame_times, dtype=np.float32)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError("residual_flows must have shape [T, H, W, 2]")
    if len(times) != len(flows) + 1:
        raise ValueError("frame_times must contain one timestamp per sampled frame")
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if global_flows is not None and np.shape(global_flows) != flows.shape:
        raise ValueError("global_flows must have the same shape as residual_flows")

    dt = np.maximum(np.diff(times), config.eps)
    velocity = flows / (float(frame_width) * dt[:, None, None, None])
    magnitude = np.linalg.norm(velocity, axis=-1)
    active = magnitude >= config.active_speed_threshold
    # Remove sub-threshold tracker noise before *all* statistics. Otherwise,
    # a tiny temporal coefficient-of-variation can survive and become a unit
    # vector at the final descriptor normalization.
    actor_reliable = float(np.mean(active)) >= config.minimum_active_fraction
    if not actor_reliable:
        active = np.zeros_like(active)
    velocity = np.where(active[..., None], velocity, 0.0)
    magnitude = np.where(active, magnitude, 0.0)
    positive = magnitude[magnitude > 0]
    clip_value = (
        float(np.percentile(positive, config.magnitude_clip_percentile))
        if positive.size
        else 1.0
    )
    clip_value = max(clip_value, config.eps)

    features: list[float] = []
    time_groups = np.array_split(np.arange(len(velocity)), config.temporal_bins)
    row_groups = np.array_split(np.arange(velocity.shape[1]), config.grid_rows)
    column_groups = np.array_split(np.arange(velocity.shape[2]), config.grid_cols)

    # Time x space HOOF preserves direction and rough actor location.
    for time_indices in time_groups:
        for row_indices in row_groups:
            for column_indices in column_groups:
                if not len(time_indices) or not len(row_indices) or not len(column_indices):
                    histogram = np.zeros(config.orientation_bins, dtype=np.float32)
                else:
                    patch = velocity[np.ix_(time_indices, row_indices, column_indices)]
                    histogram = _histogram_of_oriented_flow(
                        patch,
                        clip_value,
                        config.orientation_bins,
                        config.active_speed_threshold,
                        config.eps,
                    )
                features.extend(float(value) for value in histogram)

    # Per-phase speed, vector direction, and active support.
    for time_indices in time_groups:
        if not len(time_indices):
            features.extend([0.0] * 7)
            continue
        phase_velocity = velocity[time_indices]
        phase_magnitude = magnitude[time_indices]
        features.extend(
            [
                float(np.mean(phase_velocity[..., 0])),
                float(np.mean(phase_velocity[..., 1])),
                float(np.mean(phase_magnitude)),
                float(np.percentile(phase_magnitude, 90.0)),
                float(
                    np.mean(phase_magnitude >= config.active_speed_threshold)
                ),
                float(np.std(phase_magnitude)),
                float(np.max(phase_magnitude)),
            ]
        )

    # Whole-clip amplitude and temporal rhythm.
    features.extend(_safe_quantiles(magnitude, (0.25, 0.5, 0.75, 0.9, 0.99)))
    frame_energy = np.mean(magnitude, axis=(1, 2))
    features.extend(
        [
            float(np.mean(magnitude)),
            float(np.std(magnitude)),
            float(np.mean(magnitude >= config.active_speed_threshold)),
            float(np.mean(frame_energy >= config.active_speed_threshold)),
        ]
    )
    features.extend(_time_derivative_stats(frame_energy, config.eps))

    # Keep camera motion as a separate factor instead of allowing it to dominate
    # the object/action HOOF.
    if global_flows is None:
        features.extend([0.0] * 8)
    else:
        camera = np.asarray(global_flows, dtype=np.float32)
        camera_velocity = camera / (
            float(frame_width) * dt[:, None, None, None]
        )
        camera_vector = np.median(camera_velocity, axis=(1, 2))
        camera_speed = np.linalg.norm(camera_vector, axis=-1)
        camera_active = camera_speed >= config.active_speed_threshold
        camera_vector = np.where(camera_active[:, None], camera_vector, 0.0)
        camera_speed = np.where(camera_active, camera_speed, 0.0)
        features.extend(
            [
                float(np.mean(camera_vector[:, 0])),
                float(np.mean(camera_vector[:, 1])),
                float(np.std(camera_vector[:, 0])),
                float(np.std(camera_vector[:, 1])),
                float(np.mean(camera_speed)),
                float(np.percentile(camera_speed, 90.0)),
                *_time_derivative_stats(camera_speed, config.eps)[:2],
            ]
        )

    descriptor = np.asarray(features, dtype=np.float32)
    if not normalize:
        return descriptor
    norm = float(np.linalg.norm(descriptor))
    if norm <= config.eps:
        return np.zeros_like(descriptor)
    return descriptor / norm


def encode_action_delta(
    source_descriptor: np.ndarray,
    target_descriptor: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Encode an edit as target motion minus source motion."""

    source = np.asarray(source_descriptor, dtype=np.float32)
    target = np.asarray(target_descriptor, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("source and target descriptors must share a 1-D shape")
    delta = target - source
    norm = float(np.linalg.norm(delta))
    return np.zeros_like(delta) if norm <= eps else delta / norm


def encode_factorized_action_delta(
    source_descriptor: np.ndarray,
    target_descriptor: np.ndarray,
    *,
    camera_dims: int = 8,
    eps: float = 1e-8,
) -> tuple[np.ndarray, float, float]:
    """Difference actor and camera factors without cross-normalization leakage.

    Inputs must be the *unnormalized* outputs of
    :func:`encode_action_descriptor`. Actor and camera blocks are normalized
    independently at each endpoint, differenced independently, and each delta
    is normalized independently. The concatenated representation therefore
    keeps camera as an inspectable final block without allowing camera
    magnitude to rescale the actor teacher.
    """

    source = np.asarray(source_descriptor, dtype=np.float32)
    target = np.asarray(target_descriptor, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 1:
        raise ValueError("source and target descriptors must share a 1-D shape")
    if camera_dims <= 0 or len(source) <= camera_dims:
        raise ValueError("invalid camera_dims")

    def unit(values: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(values))
        return np.zeros_like(values) if norm <= eps else values / norm

    source_actor = unit(source[:-camera_dims])
    target_actor = unit(target[:-camera_dims])
    source_camera = unit(source[-camera_dims:])
    target_camera = unit(target[-camera_dims:])
    raw_actor_delta = target_actor - source_actor
    raw_camera_delta = target_camera - source_camera
    actor_delta_norm = float(np.linalg.norm(raw_actor_delta))
    camera_delta_norm = float(np.linalg.norm(raw_camera_delta))
    return (
        np.concatenate(
            (unit(raw_actor_delta), unit(raw_camera_delta)),
        ).astype(np.float32),
        actor_delta_norm,
        camera_delta_norm,
    )

"""Scalar motion-quality features for actor-vs-background triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import MotionAnalysis


FEATURE_VERSION = "actor-motion-features-v1"


@dataclass(frozen=True)
class ActorMotionFeatures:
    active_fraction: float
    temporal_coverage: float
    largest_component_share: float
    support_bbox_fraction: float
    spatial_energy_entropy: float
    direction_consistency: float
    centroid_path_length: float
    centroid_acceleration: float
    adjacent_energy_coherence: float
    periodicity: float
    actor_likeness: float
    version: str = FEATURE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_entropy(values: np.ndarray, eps: float) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(np.sum(flat))
    if total <= eps or len(flat) <= 1:
        return 0.0
    probability = flat / total
    probability = probability[probability > eps]
    entropy = -float(np.sum(probability * np.log(probability)))
    return float(np.clip(entropy / np.log(len(flat)), 0.0, 1.0))


def _largest_component_share(mask: np.ndarray) -> tuple[float, float]:
    binary = np.asarray(mask, dtype=np.uint8)
    active = int(binary.sum())
    if active == 0:
        return 0.0, 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return 0.0, 0.0
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    width = int(stats[component, cv2.CC_STAT_WIDTH])
    height = int(stats[component, cv2.CC_STAT_HEIGHT])
    frame_area = binary.shape[0] * binary.shape[1]
    return area / max(active, 1), (width * height) / max(frame_area, 1)


def _adjacent_cosine(maps: np.ndarray, eps: float) -> float:
    if len(maps) < 2:
        return 0.0
    flat = maps.reshape(len(maps), -1).astype(np.float64)
    values = []
    for first, second in zip(flat[:-1], flat[1:]):
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator > eps:
            values.append(float(np.dot(first, second) / denominator))
    return float(np.median(values)) if values else 0.0


def _periodicity(energy: np.ndarray, eps: float) -> float:
    values = np.asarray(energy, dtype=np.float64)
    if len(values) < 5 or float(np.std(values)) <= eps:
        return 0.0
    centered = values - float(np.mean(values))
    correlation = np.correlate(centered, centered, mode="full")[len(values) - 1 :]
    if correlation[0] <= eps or len(correlation) < 3:
        return 0.0
    normalized = correlation[1:] / correlation[0]
    return float(np.clip(np.max(normalized), 0.0, 1.0))


def extract_actor_motion_features(
    analysis: MotionAnalysis,
    *,
    active_speed_threshold: float = 0.005,
    minimum_frame_support: float = 0.001,
    eps: float = 1e-8,
) -> ActorMotionFeatures:
    """Summarize whether residual motion resembles a coherent moving actor.

    This is a transparent full-frame proxy. It must eventually be replaced or
    augmented with an actor detector/segmenter and tracker confidence.
    """

    flows = np.asarray(analysis.residual_flows, dtype=np.float32)
    dt = np.maximum(np.diff(analysis.frame_times), eps)
    width = float(analysis.frames_gray.shape[2])
    velocity = flows / (width * dt[:, None, None, None])
    speed = np.linalg.norm(velocity, axis=-1)
    active = speed >= active_speed_threshold
    frame_support = np.mean(active, axis=(1, 2))
    temporal_coverage = float(np.mean(frame_support >= minimum_frame_support))

    kernel = np.ones((3, 3), dtype=np.uint8)
    component_shares: list[float] = []
    bbox_fractions: list[float] = []
    centroids: list[tuple[float, float]] = []
    for frame_mask, frame_speed in zip(active, speed):
        cleaned = cv2.morphologyEx(
            frame_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            kernel,
        )
        if int(cleaned.sum()) == 0 and int(frame_mask.sum()) > 0:
            cleaned = frame_mask.astype(np.uint8)
        share, bbox_fraction = _largest_component_share(cleaned)
        if share > 0:
            component_shares.append(share)
            bbox_fractions.append(bbox_fraction)
        weights = frame_speed * cleaned
        total = float(np.sum(weights))
        if total > eps:
            y_grid, x_grid = np.indices(frame_speed.shape, dtype=np.float32)
            centroids.append(
                (
                    float(np.sum(x_grid * weights) / total) / max(frame_speed.shape[1], 1),
                    float(np.sum(y_grid * weights) / total) / max(frame_speed.shape[0], 1),
                )
            )

    energy_map = np.mean(speed, axis=0)
    entropy = _normalized_entropy(energy_map, eps)
    vector_sum = np.linalg.norm(np.sum(velocity, axis=(0, 1, 2)))
    magnitude_sum = float(np.sum(speed))
    direction_consistency = float(
        np.clip(vector_sum / (magnitude_sum + eps), 0.0, 1.0)
    )
    centroid_path = 0.0
    centroid_acceleration = 0.0
    if len(centroids) >= 2:
        centroid_array = np.asarray(centroids, dtype=np.float32)
        steps = np.diff(centroid_array, axis=0)
        centroid_path = float(np.sum(np.linalg.norm(steps, axis=-1)))
        if len(steps) >= 2:
            centroid_acceleration = float(
                np.mean(np.linalg.norm(np.diff(steps, axis=0), axis=-1))
            )

    frame_energy = np.mean(speed, axis=(1, 2))
    coherence = _adjacent_cosine(speed, eps)
    periodicity = _periodicity(frame_energy, eps)
    component_share = float(np.median(component_shares)) if component_shares else 0.0
    bbox_fraction = float(np.median(bbox_fractions)) if bbox_fractions else 0.0
    localized = 1.0 - entropy
    smooth_centroid = float(np.exp(-8.0 * centroid_acceleration))
    actor_likeness = (
        0.28 * component_share
        + 0.20 * localized
        + 0.20 * temporal_coverage
        + 0.12 * direction_consistency
        + 0.12 * max(coherence, 0.0)
        + 0.08 * smooth_centroid
    )
    if float(np.mean(active)) < minimum_frame_support:
        actor_likeness = 0.0

    return ActorMotionFeatures(
        active_fraction=float(np.mean(active)),
        temporal_coverage=temporal_coverage,
        largest_component_share=component_share,
        support_bbox_fraction=bbox_fraction,
        spatial_energy_entropy=entropy,
        direction_consistency=direction_consistency,
        centroid_path_length=centroid_path,
        centroid_acceleration=centroid_acceleration,
        adjacent_energy_coherence=coherence,
        periodicity=periodicity,
        actor_likeness=float(np.clip(actor_likeness, 0.0, 1.0)),
    )


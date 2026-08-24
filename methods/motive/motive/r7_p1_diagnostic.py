"""Strict development-only R7-P1 coherent-actor diagnostic.

This module consumes one *finalized eight-shard* :mod:`r7_track_cache`
commit.  It never tracks videos, tunes thresholds, trains a model, or
authorizes generation.  Its purpose is to test a frozen coherent-motion
selector on the already inspected 181-row development pilot.  The design was
motivated by P0 failures, so this is not independent preregistration; only the
promise not to tune against P1 cache results is enforced.

The independent audit is deliberately selection-bias resistant: every
positive target with a camera-valid cache entry is perturbed and evaluated,
including rows rejected by the base selector.  Missing comparisons receive
zero credit in the aggregate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r7_coherent_actor import (
    R7_COHERENT_ACTOR_SCHEMA,
    CoherentActorConfig,
    CoherentActorSelection,
    select_coherent_actor,
)
from .r7_preflight_extract import (
    _array_digest,
    _atomic_json,
    _atomic_jsonl,
    _canonical_json,
    _file_digest,
    _object_digest,
)
from .r7_temporal_teacher import (
    EventWindow,
    TemporalTeacherConfig,
    TemporalTeacherError,
    event_window_iou,
    select_event_window,
)
from .r7_track_cache import (
    ARCHIVE_NAME,
    DONE_NAME,
    FINAL_DIR_NAME,
    FINAL_WORLD_SIZE,
    MANIFEST_NAME,
    R7_TRACK_CACHE_PARTITION,
    R7_TRACK_CACHE_SCHEMA,
    SUMMARY_NAME,
    _rank_directory,
    validate_commit,
)


R7_P1_DIAGNOSTIC_SCHEMA = "motive-r7-p1-coherent-diagnostic-v1"
R7_P1_DIAGNOSTIC_ROW_SCHEMA = "motive-r7-p1-coherent-diagnostic-row-v1"
R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA = (
    "motive-r7-p1-coherent-diagnostic-summary-v1"
)
R7_P1_DIAGNOSTIC_DONE_SCHEMA = "motive-r7-p1-coherent-diagnostic-done-v1"
R7_P1_GATE_SCHEMA = "motive-r7-p1-frozen-development-gate-v1"
ROWS_NAME = "rows.jsonl"
OUTPUT_SUMMARY_NAME = "summary.json"
OUTPUT_DONE_NAME = "done.json"
SIDES = ("source", "target")
AUDIT_DOMAIN = "motive-r7-p1-independent-downstream-audit-v1"
DEVELOPMENT_SCOPE = (
    "inspected R5 181-row pilot; development diagnostics only; "
    "not a locked evaluation split"
)


@dataclass(frozen=True)
class DownstreamAuditConfig:
    """Pre-registered perturbation and joint-comparison thresholds."""

    track_drop_fraction: float = 0.10
    visibility_drop_fraction: float = 0.03
    time_jitter_fraction: float = 0.08
    coordinate_jitter_std: float = 0.00025
    actor_mask_iou_threshold: float = 0.60
    event_window_iou_threshold: float = 0.70
    trajectory_rmse_threshold: float = 0.01
    per_track_trajectory_rmse_threshold: float = 0.01
    energy_cosine_threshold: float = 0.85
    shape_profile_cosine_threshold: float = 0.80
    event_duration_relative_error_threshold: float = 0.10
    eps: float = 1e-8

    def validate(self) -> None:
        fractions = (
            "track_drop_fraction",
            "visibility_drop_fraction",
            "time_jitter_fraction",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value < 0.5:
                raise ValueError(f"{name} must be finite and in [0,0.5)")
        if (
            not math.isfinite(self.coordinate_jitter_std)
            or self.coordinate_jitter_std < 0.0
        ):
            raise ValueError(
                "coordinate_jitter_std must be finite and nonnegative"
            )
        unit_thresholds = (
            "actor_mask_iou_threshold",
            "event_window_iou_threshold",
            "energy_cosine_threshold",
            "shape_profile_cosine_threshold",
        )
        for name in unit_thresholds:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        positive_thresholds = (
            "trajectory_rmse_threshold",
            "per_track_trajectory_rmse_threshold",
            "event_duration_relative_error_threshold",
            "eps",
        )
        for name in positive_thresholds:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DiagnosticGateConfig:
    """Gate frozen before P1 cache results, but driven by prior P0 failure."""

    target_camera_cache_coverage: float = 0.90
    positive_target_ready_fraction: float = 0.65
    paired_positive_ready_count: int = 50
    no_action_minimum_samples: int = 30
    no_action_false_event_fraction: float = 0.10
    positive_target_audit_joint_pass_fraction: float = 0.70
    positive_vs_no_action_score_auroc: float = 0.75

    def validate(self) -> None:
        for name in (
            "target_camera_cache_coverage",
            "positive_target_ready_fraction",
            "no_action_false_event_fraction",
            "positive_target_audit_joint_pass_fraction",
            "positive_vs_no_action_score_auroc",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in (
            "paired_positive_ready_count",
            "no_action_minimum_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class P1DiagnosticConfig:
    seed: int = 20260727
    selector: CoherentActorConfig = field(
        default_factory=CoherentActorConfig
    )
    audit: DownstreamAuditConfig = field(
        default_factory=DownstreamAuditConfig
    )
    gate: DiagnosticGateConfig = field(
        default_factory=DiagnosticGateConfig
    )

    def validate(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("seed must be an unsigned 64-bit integer")
        self.selector.validate()
        self.audit.validate()
        self.gate.validate()


def _output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        "rows": output_directory / ROWS_NAME,
        "summary": output_directory / OUTPUT_SUMMARY_NAME,
        "done": output_directory / OUTPUT_DONE_NAME,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            if line != _canonical_json(value) + "\n":
                raise ValueError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            output.append(value)
    return output


def _shared_cache_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "input_manifest",
        "input_manifest_sha256",
        "data_root",
        "world_size",
        "partition",
        "seed",
        "frames",
        "tracker",
        "camera_config",
        "cache_scope",
        "formal_status",
        "production_decision",
        "generation_authorized",
        "implementation_sha256",
    )
    return {key: contract.get(key) for key in keys}


def load_final_cache(
    *,
    input_manifest: Path,
    cache_root: Path,
) -> dict[str, Any]:
    """Validate the final commit and all eight exact source shards.

    This intentionally does more than trusting ``final/done.json``.  Every
    source shard is revalidated, its digest is matched to the final contract,
    modulo ownership is checked, and its arrays are compared to the merged
    archive.  A missing shard or a non-eight-way commit therefore fails
    closed.
    """

    manifest = input_manifest.expanduser().resolve(strict=True)
    root = cache_root.expanduser().resolve(strict=True)
    final_directory = root / FINAL_DIR_NAME
    final = validate_commit(
        final_directory,
        input_manifest=manifest,
        final=True,
    )
    contract = final["contract"]
    if contract.get("schema_version") != R7_TRACK_CACHE_SCHEMA:
        raise ValueError("final cache contract schema differs")
    if contract.get("partition") != R7_TRACK_CACHE_PARTITION:
        raise ValueError("final cache partition differs")
    if contract.get("rank") != "merged":
        raise ValueError("cache is not a merged final commit")
    if contract.get("world_size") != FINAL_WORLD_SIZE:
        raise ValueError("final cache world_size is not eight")
    if contract.get("merge_world_size") != FINAL_WORLD_SIZE:
        raise ValueError("final cache merge_world_size is not eight")
    source_digests = contract.get("source_shard_done_sha256")
    if (
        not isinstance(source_digests, list)
        or len(source_digests) != FINAL_WORLD_SIZE
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in source_digests
        )
    ):
        raise ValueError("final cache lacks eight source-shard digests")

    shards_directory = root / "shards"
    expected_names = {
        f"rank-{rank:03d}-of-{FINAL_WORLD_SIZE:03d}"
        for rank in range(FINAL_WORLD_SIZE)
    }
    if not shards_directory.is_dir():
        raise FileNotFoundError(shards_directory)
    actual_names = {
        path.name for path in shards_directory.iterdir() if path.is_dir()
    }
    if actual_names != expected_names:
        raise ValueError(
            "cache shard directory set differs from the exact eight-way set"
        )

    final_indices = np.asarray(
        final["arrays"]["input_indices"], dtype=np.int64
    )
    row_count = len(final["rows"])
    if not np.array_equal(final_indices, np.arange(row_count)):
        raise ValueError("final cache is not in complete input order")
    shared_final = _shared_cache_contract(contract)
    for rank in range(FINAL_WORLD_SIZE):
        shard_directory = _rank_directory(
            root, rank, FINAL_WORLD_SIZE
        )
        shard = validate_commit(
            shard_directory,
            input_manifest=manifest,
            final=False,
        )
        if _file_digest(shard_directory / DONE_NAME) != source_digests[rank]:
            raise ValueError(f"cache shard {rank} done digest differs")
        shard_contract = shard["contract"]
        if shard_contract.get("rank") != rank:
            raise ValueError(f"cache shard {rank} contract rank differs")
        if _shared_cache_contract(shard_contract) != shared_final:
            raise ValueError(f"cache shard {rank} contract differs")
        indices = np.asarray(
            shard["arrays"]["input_indices"], dtype=np.int64
        )
        expected = np.arange(rank, row_count, FINAL_WORLD_SIZE)
        if not np.array_equal(indices, expected):
            raise ValueError(f"cache shard {rank} modulo ownership differs")
        for local_index, input_index_value in enumerate(indices):
            input_index = int(input_index_value)
            shard_row = dict(shard["rows"][local_index])
            final_row = dict(final["rows"][input_index])
            final_array_index = final_row.pop("final_array_index", None)
            if final_array_index != input_index or final_row != shard_row:
                raise ValueError(
                    f"cache shard/final row differs at {input_index}"
                )
            for name, final_array in final["arrays"].items():
                if not np.array_equal(
                    np.asarray(final_array)[input_index],
                    np.asarray(shard["arrays"][name])[local_index],
                ):
                    raise ValueError(
                        f"cache shard/final array {name} differs at "
                        f"{input_index}"
                    )
    return final


def _audit_seed(iid: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}\0{iid}\0target\0{AUDIT_DOMAIN}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_downstream_perturbation(
    tracks: Any,
    visibility: Any,
    frame_times: Any,
    *,
    seed: int,
    config: DownstreamAuditConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply all four frozen perturbations with one audit-only seed."""

    config.validate()
    track_array = np.asarray(tracks, dtype=np.float64)
    visibility_array = np.asarray(visibility, dtype=np.float64)
    times = np.asarray(frame_times, dtype=np.float64)
    if (
        track_array.ndim != 3
        or track_array.shape[-1] != 2
        or visibility_array.shape != track_array.shape[:2]
        or times.shape != (track_array.shape[0],)
    ):
        raise ValueError("audit perturbation input shapes differ")
    if (
        not np.isfinite(track_array).all()
        or not np.isfinite(visibility_array).all()
        or not np.isfinite(times).all()
        or bool((np.diff(times) <= 0.0).any())
    ):
        raise ValueError("audit perturbation inputs are invalid")
    rng = np.random.default_rng(seed)
    perturbed_tracks = track_array.copy()
    perturbed_visibility = visibility_array.copy()
    perturbed_times = times.copy()
    track_count = track_array.shape[1]

    drop_count = (
        min(
            track_count,
            max(1, int(round(config.track_drop_fraction * track_count))),
        )
        if config.track_drop_fraction > 0.0
        else 0
    )
    dropped = (
        np.sort(rng.choice(track_count, size=drop_count, replace=False))
        if drop_count
        else np.zeros(0, dtype=np.int64)
    )
    if drop_count:
        perturbed_visibility[:, dropped] = 0.0

    candidates = np.argwhere(perturbed_visibility > 0.0)
    visibility_drop_count = (
        min(
            len(candidates),
            max(
                1,
                int(
                    round(
                        config.visibility_drop_fraction * len(candidates)
                    )
                ),
            ),
        )
        if config.visibility_drop_fraction > 0.0 and len(candidates)
        else 0
    )
    if visibility_drop_count:
        chosen = rng.choice(
            len(candidates),
            size=visibility_drop_count,
            replace=False,
        )
        selected = candidates[chosen]
        perturbed_visibility[selected[:, 0], selected[:, 1]] = 0.0

    if config.time_jitter_fraction > 0.0 and len(times) > 2:
        intervals = np.diff(times)
        local = np.minimum(intervals[:-1], intervals[1:])
        jitter = rng.uniform(
            -config.time_jitter_fraction,
            config.time_jitter_fraction,
            size=len(times) - 2,
        )
        perturbed_times[1:-1] += jitter * local
    if bool((np.diff(perturbed_times) <= 0.0).any()):
        raise ValueError("audit time jitter broke strict ordering")

    if config.coordinate_jitter_std > 0.0:
        perturbed_tracks += rng.normal(
            0.0,
            config.coordinate_jitter_std,
            size=perturbed_tracks.shape,
        )
    output_tracks = perturbed_tracks.astype(np.float32)
    output_visibility = perturbed_visibility.astype(np.float32)
    provenance = {
        "seed": int(seed),
        "domain": AUDIT_DOMAIN,
        "track_drop_count": int(drop_count),
        "track_drop_indices": dropped.tolist(),
        "visibility_drop_count": int(visibility_drop_count),
        "time_jitter_fraction": config.time_jitter_fraction,
        "coordinate_jitter_std": config.coordinate_jitter_std,
        "tracks_sha256": _array_digest(output_tracks),
        "visibility_sha256": _array_digest(output_visibility),
        "frame_times_sha256": _array_digest(perturbed_times),
    }
    return (
        output_tracks,
        output_visibility,
        perturbed_times,
        provenance,
    )


def _actor_phase_speed(
    trajectory: np.ndarray,
    phase_times: np.ndarray,
) -> np.ndarray:
    dt = np.diff(np.asarray(phase_times, dtype=np.float64))
    if len(dt) < 1 or bool((dt <= 0.0).any()):
        raise ValueError("actor phase times are not strictly increasing")
    speed = np.linalg.norm(
        np.diff(np.asarray(trajectory, dtype=np.float64), axis=0)
        / dt[:, None],
        axis=-1,
    )
    if not np.isfinite(speed).all():
        raise ValueError("actor phase speed is non-finite")
    return speed


def _selector_transition_energy(
    phase_energy: Any,
    phase_times: Any,
) -> np.ndarray:
    """Put selector energy on the transition grid expected by the locator.

    The coherent selector's energy is defined from the selected *track
    component*, so it remains nonzero for rotations or symmetric deformation
    whose actor centroid does not translate.  Adjacent phase samples are
    averaged without shifting the corresponding phase boundaries.
    """

    energy = np.asarray(phase_energy, dtype=np.float64)
    times = np.asarray(phase_times, dtype=np.float64)
    if (
        energy.ndim != 1
        or times.shape != energy.shape
        or len(energy) < 2
        or not np.isfinite(energy).all()
        or not np.isfinite(times).all()
        or bool((energy < 0.0).any())
        or bool((np.diff(times) <= 0.0).any())
    ):
        raise ValueError("selector phase energy/times are invalid")
    return 0.5 * (energy[:-1] + energy[1:])


def _selection_score(selection: CoherentActorSelection) -> float:
    if (
        not selection.diagnostic_ready
        or selection.selected_component is None
        or not 0 <= selection.selected_component < len(selection.components)
    ):
        return 0.0
    selected = selection.components[selection.selected_component]
    if not selected.accepted:
        return 0.0
    score = float(selected.selection_score)
    return score if math.isfinite(score) and score >= 0.0 else 0.0


def _evaluate_side(
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_times: np.ndarray,
    *,
    selector_config: CoherentActorConfig,
    event_config: TemporalTeacherConfig,
) -> dict[str, Any]:
    selection = select_coherent_actor(
        tracks,
        visibility,
        frame_times,
        config=selector_config,
    )
    selector_ready = bool(selection.diagnostic_ready)
    trajectory = np.asarray(selection.actor_trajectory, dtype=np.float32)
    phase_times = np.asarray(selection.phase_times, dtype=np.float64)
    phase_speed = np.zeros(0, dtype=np.float64)
    event_energy = np.zeros(0, dtype=np.float64)
    event: EventWindow | None = None
    event_failure_reason: str | None = None
    if selector_ready:
        try:
            phase_speed = _actor_phase_speed(trajectory, phase_times)
            event_energy = _selector_transition_energy(
                selection.phase_energy,
                phase_times,
            )
            event = select_event_window(
                event_energy,
                phase_times,
                config=event_config,
            )
        except (TemporalTeacherError, ValueError) as error:
            event_failure_reason = (
                error.reason
                if isinstance(error, TemporalTeacherError)
                else "invalid_actor_trajectory"
            )
    ready = selector_ready and event is not None
    if not selector_ready:
        failure_stage = "coherent_actor_selector"
        failure_reason = selection.failure_reason
        failure_detail = selection.failure_detail
    elif event is None:
        failure_stage = "continuous_event_locator"
        failure_reason = event_failure_reason
        failure_detail = "accepted component has no valid continuous event"
    else:
        failure_stage = None
        failure_reason = None
        failure_detail = None
    return {
        "diagnostic_ready": bool(ready),
        "selector_ready": selector_ready,
        "event_ready": event is not None,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "failure_detail": failure_detail,
        "selector": selection.to_summary(),
        "score": _selection_score(selection),
        "actor_track_mask": selection.actor_track_mask.tolist(),
        "actor_trajectory": trajectory.tolist(),
        "actor_track_trajectories": (
            np.asarray(
                selection.actor_track_trajectories, dtype=np.float32
            ).tolist()
        ),
        "actor_track_phase_mask": (
            np.asarray(selection.actor_track_phase_mask, dtype=bool).tolist()
        ),
        "phase_times": phase_times.tolist(),
        "selector_phase_energy": (
            np.asarray(selection.phase_energy, dtype=np.float32).tolist()
        ),
        "actor_phase_speed": phase_speed.tolist(),
        "event_transition_energy": event_energy.tolist(),
        "phase_visibility": (
            np.asarray(selection.phase_visibility, dtype=np.float32).tolist()
        ),
        "event_window": event.to_dict() if event is not None else None,
    }


def _camera_invalid_side() -> dict[str, Any]:
    return {
        "diagnostic_ready": False,
        "selector_ready": False,
        "event_ready": False,
        "failure_stage": "track_cache",
        "failure_reason": "camera_cache_invalid",
        "failure_detail": "selector was not run without camera compensation",
        "selector": None,
        "score": 0.0,
        "actor_track_mask": [],
        "actor_trajectory": [],
        "actor_track_trajectories": [],
        "actor_track_phase_mask": [],
        "phase_times": [],
        "selector_phase_energy": [],
        "actor_phase_speed": [],
        "event_transition_energy": [],
        "phase_visibility": [],
        "event_window": None,
    }


def _mask_iou(first: Sequence[bool], second: Sequence[bool]) -> float:
    first_array = np.asarray(first, dtype=bool)
    second_array = np.asarray(second, dtype=bool)
    if first_array.shape != second_array.shape or first_array.ndim != 1:
        return 0.0
    union = int(np.sum(first_array | second_array))
    return (
        float(np.sum(first_array & second_array) / union)
        if union
        else 0.0
    )


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    first_array = np.asarray(first, dtype=np.float64).reshape(-1)
    second_array = np.asarray(second, dtype=np.float64).reshape(-1)
    if (
        first_array.shape != second_array.shape
        or not np.isfinite(first_array).all()
        or not np.isfinite(second_array).all()
    ):
        return 0.0
    denominator = float(
        np.linalg.norm(first_array) * np.linalg.norm(second_array)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(
        np.clip(
            np.dot(first_array, second_array) / denominator,
            -1.0,
            1.0,
        )
    )


def _event_from_record(record: Mapping[str, Any]) -> EventWindow:
    value = record.get("event_window")
    if not isinstance(value, Mapping):
        raise ValueError("side record lacks event window")
    return EventWindow(**dict(value))


def _shared_track_metrics(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    eps: float,
) -> dict[str, float]:
    """Compare shared physical tracks and permutation-invariant shape."""

    first_selector = first.get("selector")
    second_selector = second.get("selector")
    if not isinstance(first_selector, Mapping) or not isinstance(
        second_selector, Mapping
    ):
        return {
            "shared_actor_track_fraction": 0.0,
            "per_track_trajectory_rmse": 1.0,
            "shape_profile_cosine": 0.0,
        }
    first_indices = [
        int(value)
        for value in first_selector.get("actor_track_indices", [])
    ]
    second_indices = [
        int(value)
        for value in second_selector.get("actor_track_indices", [])
    ]
    first_lookup = {value: index for index, value in enumerate(first_indices)}
    second_lookup = {
        value: index for index, value in enumerate(second_indices)
    }
    shared = sorted(set(first_lookup) & set(second_lookup))
    union = set(first_lookup) | set(second_lookup)
    if not shared or not union:
        return {
            "shared_actor_track_fraction": 0.0,
            "per_track_trajectory_rmse": 1.0,
            "shape_profile_cosine": 0.0,
        }
    first_tracks = np.asarray(
        first.get("actor_track_trajectories"), dtype=np.float64
    )
    second_tracks = np.asarray(
        second.get("actor_track_trajectories"), dtype=np.float64
    )
    first_masks = np.asarray(
        first.get("actor_track_phase_mask"), dtype=bool
    )
    second_masks = np.asarray(
        second.get("actor_track_phase_mask"), dtype=bool
    )
    if (
        first_tracks.ndim != 3
        or second_tracks.ndim != 3
        or first_tracks.shape[-1] != 2
        or second_tracks.shape[-1] != 2
        or first_masks.shape != first_tracks.shape[:2]
        or second_masks.shape != second_tracks.shape[:2]
        or first_tracks.shape[1:] != second_tracks.shape[1:]
    ):
        return {
            "shared_actor_track_fraction": float(len(shared) / len(union)),
            "per_track_trajectory_rmse": 1.0,
            "shape_profile_cosine": 0.0,
        }

    squared: list[np.ndarray] = []
    first_aligned: list[np.ndarray] = []
    second_aligned: list[np.ndarray] = []
    shared_masks: list[np.ndarray] = []
    for track_index in shared:
        first_row = first_lookup[track_index]
        second_row = second_lookup[track_index]
        keep = first_masks[first_row] & second_masks[second_row]
        phases = np.flatnonzero(keep)
        if not len(phases):
            continue
        first_value = first_tracks[first_row].copy()
        second_value = second_tracks[second_row].copy()
        # Re-anchor at the first commonly visible phase.  A visibility drop
        # can otherwise create a meaningless per-track translation offset.
        anchor = int(phases[0])
        first_value -= first_value[anchor]
        second_value -= second_value[anchor]
        squared.append((first_value[keep] - second_value[keep]) ** 2)
        first_aligned.append(first_value)
        second_aligned.append(second_value)
        shared_masks.append(keep)
    if not squared:
        return {
            "shared_actor_track_fraction": float(len(shared) / len(union)),
            "per_track_trajectory_rmse": 1.0,
            "shape_profile_cosine": 0.0,
        }
    per_track_rmse = float(
        np.sqrt(np.mean(np.concatenate(squared, axis=0)))
    )

    def shape_profile(
        trajectories: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        profile = np.zeros(trajectories.shape[1], dtype=np.float64)
        for phase in range(trajectories.shape[1]):
            keep = masks[:, phase]
            if int(np.sum(keep)) < 2:
                continue
            values = trajectories[keep, phase]
            center = np.median(values, axis=0)
            profile[phase] = float(
                np.sqrt(np.mean(np.sum((values - center) ** 2, axis=-1)))
            )
        return profile

    first_array = np.stack(first_aligned)
    second_array = np.stack(second_aligned)
    mask_array = np.stack(shared_masks)
    first_shape = shape_profile(first_array, mask_array)
    second_shape = shape_profile(second_array, mask_array)
    first_norm = float(np.linalg.norm(first_shape))
    second_norm = float(np.linalg.norm(second_shape))
    if first_norm <= eps and second_norm <= eps:
        # Rigid translations contain no differential shape signal.  They are
        # evaluated by mask, aggregate, energy, and per-track metrics.
        shape_cosine = 1.0
    elif first_norm <= eps or second_norm <= eps:
        shape_cosine = 0.0
    else:
        shape_cosine = _cosine(first_shape, second_shape)
    return {
        "shared_actor_track_fraction": float(len(shared) / len(union)),
        "per_track_trajectory_rmse": per_track_rmse,
        "shape_profile_cosine": shape_cosine,
    }


def _audit_defaults() -> dict[str, float]:
    return {
        "actor_mask_iou": 0.0,
        "event_window_iou": 0.0,
        "trajectory_rmse": 1.0,
        "shared_actor_track_fraction": 0.0,
        "per_track_trajectory_rmse": 1.0,
        "energy_cosine": 0.0,
        "shape_profile_cosine": 0.0,
        "event_duration_relative_error": 1.0,
    }


def _evaluate_audit(
    *,
    base: Mapping[str, Any],
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_times: np.ndarray,
    seed: int,
    config: P1DiagnosticConfig,
    event_config: TemporalTeacherConfig,
) -> dict[str, Any]:
    """Run an audit regardless of whether ``base`` was selected."""

    (
        perturbed_tracks,
        perturbed_visibility,
        perturbed_times,
        perturbation,
    ) = deterministic_downstream_perturbation(
        tracks,
        visibility,
        frame_times,
        seed=seed,
        config=config.audit,
    )
    perturbed = _evaluate_side(
        perturbed_tracks,
        perturbed_visibility,
        perturbed_times,
        selector_config=config.selector,
        event_config=event_config,
    )
    ready_consistent = bool(
        bool(base["diagnostic_ready"])
        == bool(perturbed["diagnostic_ready"])
    )
    available = bool(
        base["diagnostic_ready"] and perturbed["diagnostic_ready"]
    )
    metrics = _audit_defaults()
    if available:
        base_trajectory = np.asarray(
            base["actor_trajectory"], dtype=np.float64
        )
        perturbed_trajectory = np.asarray(
            perturbed["actor_trajectory"], dtype=np.float64
        )
        if base_trajectory.shape == perturbed_trajectory.shape:
            trajectory_rmse = float(
                np.sqrt(
                    np.mean(
                        (base_trajectory - perturbed_trajectory) ** 2
                    )
                )
            )
        else:
            trajectory_rmse = 1.0
        base_event = _event_from_record(base)
        perturbed_event = _event_from_record(perturbed)
        duration_error = abs(
            perturbed_event.duration - base_event.duration
        ) / max(base_event.duration, config.audit.eps)
        shared_track_metrics = _shared_track_metrics(
            base,
            perturbed,
            eps=config.audit.eps,
        )
        metrics = {
            "actor_mask_iou": _mask_iou(
                base["actor_track_mask"],
                perturbed["actor_track_mask"],
            ),
            "event_window_iou": event_window_iou(
                base_event, perturbed_event
            ),
            "trajectory_rmse": trajectory_rmse,
            **shared_track_metrics,
            "energy_cosine": _cosine(
                base["event_transition_energy"],
                perturbed["event_transition_energy"],
            ),
            "event_duration_relative_error": float(duration_error),
        }
    thresholds = config.audit
    joint_pass = bool(
        ready_consistent
        and available
        and metrics["actor_mask_iou"]
        >= thresholds.actor_mask_iou_threshold
        and metrics["event_window_iou"]
        >= thresholds.event_window_iou_threshold
        and metrics["trajectory_rmse"]
        <= thresholds.trajectory_rmse_threshold
        and metrics["per_track_trajectory_rmse"]
        <= thresholds.per_track_trajectory_rmse_threshold
        and metrics["energy_cosine"]
        >= thresholds.energy_cosine_threshold
        and metrics["shape_profile_cosine"]
        >= thresholds.shape_profile_cosine_threshold
        and metrics["event_duration_relative_error"]
        <= thresholds.event_duration_relative_error_threshold
    )
    if not base["diagnostic_ready"]:
        failure_reason = "base_not_ready"
    elif not perturbed["diagnostic_ready"]:
        failure_reason = "perturbed_not_ready"
    elif not joint_pass:
        failure_reason = "joint_threshold_failed"
    else:
        failure_reason = None
    return {
        "eligible": True,
        "performed": True,
        "seed": int(seed),
        "seed_derivation": (
            "sha256(base_seed,iid,target,"
            f"{AUDIT_DOMAIN})-u64-v1"
        ),
        "perturbation": perturbation,
        "ready_consistent": ready_consistent,
        "comparison_available": available,
        "metrics": metrics,
        "joint_pass": joint_pass,
        "failure_reason": failure_reason,
        "perturbed": perturbed,
    }


def _ineligible_audit(reason: str) -> dict[str, Any]:
    return {
        "eligible": False,
        "performed": False,
        "seed": None,
        "seed_derivation": None,
        "perturbation": None,
        "ready_consistent": False,
        "comparison_available": False,
        "metrics": _audit_defaults(),
        "joint_pass": False,
        "failure_reason": reason,
        "perturbed": None,
    }


def _negative_type(row: Mapping[str, Any]) -> str | None:
    direct = row.get("negative_type")
    if isinstance(direct, str) and direct:
        return direct
    label = row.get("label_type")
    if label in {"static", "endpoint_only", "instruction_mismatch"}:
        return str(label)
    return None


def _auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float | None:
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if not len(positive) or not len(negative):
        return None
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("AUROC scores are non-finite")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (
            np.sum(comparisons > 0.0)
            + 0.5 * np.sum(comparisons == 0.0)
        )
        / comparisons.size
    )


def compute_p1_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: DiagnosticGateConfig | None = None,
) -> dict[str, Any]:
    """Compute only the fixed development gate; never formal readiness."""

    cfg = config or DiagnosticGateConfig()
    cfg.validate()
    row_count = len(rows)
    positive = np.asarray(
        [bool(row.get("positive")) for row in rows], dtype=bool
    )
    target_camera = np.asarray(
        [bool(row["target_camera_valid"]) for row in rows], dtype=bool
    )
    source_ready = np.asarray(
        [bool(row["source"]["diagnostic_ready"]) for row in rows],
        dtype=bool,
    )
    target_ready = np.asarray(
        [bool(row["target"]["diagnostic_ready"]) for row in rows],
        dtype=bool,
    )
    audit_pass = np.asarray(
        [bool(row["target_audit"]["joint_pass"]) for row in rows],
        dtype=bool,
    )
    negative_types = np.asarray(
        [_negative_type(row) for row in rows], dtype=object
    )
    no_action = (
        ~positive
        & np.isin(negative_types, ("static", "endpoint_only"))
    )
    instruction_mismatch = (
        ~positive & (negative_types == "instruction_mismatch")
    )

    positive_count = int(np.sum(positive))
    positive_target_ready = int(np.sum(positive & target_ready))
    target_coverage = (
        float(np.sum(target_camera) / row_count) if row_count else 0.0
    )
    positive_ready_fraction = (
        float(positive_target_ready / positive_count)
        if positive_count
        else 0.0
    )
    paired_ready = int(
        np.sum(positive & source_ready & target_ready)
    )
    no_action_count = int(np.sum(no_action))
    false_events = int(np.sum(no_action & target_ready))
    false_event_fraction = (
        float(false_events / no_action_count)
        if no_action_count
        else None
    )
    audit_eligible = positive & target_camera
    audit_eligible_count = int(np.sum(audit_eligible))
    audit_pass_count = int(np.sum(audit_eligible & audit_pass))
    audit_fraction = (
        float(audit_pass_count / audit_eligible_count)
        if audit_eligible_count
        else 0.0
    )

    auroc_mask = target_camera & (positive | no_action)
    positive_scores = [
        float(rows[index]["target"]["score"])
        for index in np.flatnonzero(auroc_mask & positive)
    ]
    no_action_scores = [
        float(rows[index]["target"]["score"])
        for index in np.flatnonzero(auroc_mask & no_action)
    ]
    score_auroc = _auroc(positive_scores, no_action_scores)
    criteria = {
        "target_camera_cache_coverage": {
            "value": target_coverage,
            "threshold": cfg.target_camera_cache_coverage,
            "operator": ">=",
            "numerator": int(np.sum(target_camera)),
            "denominator": row_count,
            "passed": target_coverage
            >= cfg.target_camera_cache_coverage,
        },
        "positive_target_ready_fraction": {
            "value": positive_ready_fraction,
            "threshold": cfg.positive_target_ready_fraction,
            "operator": ">=",
            "numerator": positive_target_ready,
            "denominator": positive_count,
            "passed": positive_ready_fraction
            >= cfg.positive_target_ready_fraction,
        },
        "paired_positive_ready_count": {
            "value": paired_ready,
            "threshold": cfg.paired_positive_ready_count,
            "operator": ">=",
            "passed": paired_ready >= cfg.paired_positive_ready_count,
        },
        "no_action_negative_samples": {
            "value": no_action_count,
            "threshold": cfg.no_action_minimum_samples,
            "operator": ">=",
            "included_types": ["static", "endpoint_only"],
            "excluded_types": ["instruction_mismatch"],
            "passed": no_action_count >= cfg.no_action_minimum_samples,
        },
        "no_action_false_event_fraction": {
            "value": false_event_fraction,
            "threshold": cfg.no_action_false_event_fraction,
            "operator": "<=",
            "numerator": false_events,
            "denominator": no_action_count,
            "passed": (
                false_event_fraction is not None
                and false_event_fraction
                <= cfg.no_action_false_event_fraction
            ),
        },
        "positive_target_audit_joint_pass_fraction": {
            "value": audit_fraction,
            "threshold": (
                cfg.positive_target_audit_joint_pass_fraction
            ),
            "operator": ">=",
            "numerator": audit_pass_count,
            "denominator": audit_eligible_count,
            "missing_or_failed": audit_eligible_count - audit_pass_count,
            "passed": audit_fraction
            >= cfg.positive_target_audit_joint_pass_fraction,
        },
        "positive_vs_no_action_score_auroc": {
            "value": score_auroc,
            "threshold": cfg.positive_vs_no_action_score_auroc,
            "operator": ">=",
            "positive_camera_valid_samples": len(positive_scores),
            "no_action_camera_valid_samples": len(no_action_scores),
            "instruction_mismatch_excluded": True,
            "passed": (
                score_auroc is not None
                and score_auroc
                >= cfg.positive_vs_no_action_score_auroc
            ),
        },
    }
    passed = all(bool(value["passed"]) for value in criteria.values())
    return {
        "schema_version": R7_P1_GATE_SCHEMA,
        "diagnostic_status": (
            "DIAGNOSTIC_SELECTOR_READY"
            if passed
            else "DIAGNOSTIC_SELECTOR_NOT_READY"
        ),
        "diagnostic_gate_passed": passed,
        "criteria": criteria,
        "counts": {
            "rows": row_count,
            "positive_rows": positive_count,
            "target_camera_valid": int(np.sum(target_camera)),
            "positive_target_camera_valid": audit_eligible_count,
            "positive_target_ready": positive_target_ready,
            "positive_paired_ready": paired_ready,
            "positive_target_audit_joint_pass": audit_pass_count,
            "positive_target_audit_zero_credit": (
                audit_eligible_count - audit_pass_count
            ),
            "negative_no_action_rows": no_action_count,
            "negative_no_action_false_events": false_events,
            "negative_instruction_mismatch_excluded": int(
                np.sum(instruction_mismatch)
            ),
        },
        "audit_scope": (
            "all positive target camera-valid rows; base readiness is not "
            "an audit mask; unavailable comparisons receive zero credit"
        ),
        "auroc_scope": (
            "camera-valid positive versus static/endpoint_only target "
            "selector scores; instruction_mismatch is excluded"
        ),
        "formal_status": "INSUFFICIENT",
        "formal_reason": DEVELOPMENT_SCOPE,
        "production_decision": False,
        "generation_authorized": False,
    }


def _artifact_binding(
    *,
    cache_root: Path,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    final_directory = cache_root.resolve(strict=True) / FINAL_DIR_NAME
    contract = cache["contract"]
    return {
        "root": str(cache_root.resolve(strict=True)),
        "final_directory": str(final_directory),
        "done_sha256": _file_digest(final_directory / DONE_NAME),
        "archive_name": ARCHIVE_NAME,
        "archive_sha256": _file_digest(final_directory / ARCHIVE_NAME),
        "manifest_name": MANIFEST_NAME,
        "manifest_sha256": _file_digest(final_directory / MANIFEST_NAME),
        "summary_name": SUMMARY_NAME,
        "summary_sha256": _file_digest(final_directory / SUMMARY_NAME),
        "contract_sha256": _object_digest(contract),
        "source_shard_done_sha256": list(
            contract["source_shard_done_sha256"]
        ),
        "world_size": FINAL_WORLD_SIZE,
        "strict_source_shards_revalidated": True,
    }


def build_diagnostic_contract(
    *,
    input_manifest: Path,
    cache_root: Path,
    cache: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> dict[str, Any]:
    config.validate()
    module = Path(__file__).resolve()
    selector_module = module.with_name("r7_coherent_actor.py")
    cache_module = module.with_name("r7_track_cache.py")
    event_module = module.with_name("r7_temporal_teacher.py")
    selector_config = asdict(config.selector)
    audit_config = asdict(config.audit)
    gate_config = asdict(config.gate)
    event_config = asdict(TemporalTeacherConfig())
    return {
        "schema_version": R7_P1_DIAGNOSTIC_SCHEMA,
        "input_manifest": str(
            input_manifest.expanduser().resolve(strict=True)
        ),
        "input_manifest_sha256": _file_digest(input_manifest),
        "cache": _artifact_binding(
            cache_root=cache_root,
            cache=cache,
        ),
        "seed": config.seed,
        "selector": {
            "schema_version": R7_COHERENT_ACTOR_SCHEMA,
            "implementation_sha256": _file_digest(selector_module),
            "config": selector_config,
            "config_sha256": _object_digest(selector_config),
        },
        "continuous_event_locator": {
            "implementation": "r7_temporal_teacher.select_event_window",
            "implementation_sha256": _file_digest(event_module),
            "config": event_config,
            "config_sha256": _object_digest(event_config),
            "energy_definition": (
                "adjacent mean of coherent selector component phase_energy; "
                "actor centroid speed is recorded separately"
            ),
        },
        "independent_audit": {
            "domain": AUDIT_DOMAIN,
            "implementation_sha256": _file_digest(module),
            "config": audit_config,
            "config_sha256": _object_digest(audit_config),
            "base_selection_is_audit_mask": False,
            "missing_comparison_credit": 0,
        },
        "diagnostic_gate": {
            "schema_version": R7_P1_GATE_SCHEMA,
            "config": gate_config,
            "config_sha256": _object_digest(gate_config),
            "thresholds_frozen_before_p1_cache_results": True,
            "not_independent_preregistration": True,
            "design_was_driven_by_prior_p0_failure": True,
            "thresholds_may_not_be_adjusted_from_results": True,
        },
        "implementation_sha256": {
            module.name: _file_digest(module),
            selector_module.name: _file_digest(selector_module),
            cache_module.name: _file_digest(cache_module),
            event_module.name: _file_digest(event_module),
        },
        "development_scope": DEVELOPMENT_SCOPE,
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }


def evaluate_cache(
    *,
    cache: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> list[dict[str, Any]]:
    """Evaluate both sides and an unbiased positive-target audit."""

    config.validate()
    cache_rows = cache["rows"]
    arrays = cache["arrays"]
    event_config = TemporalTeacherConfig()
    output: list[dict[str, Any]] = []
    for array_index, cache_row in enumerate(cache_rows):
        input_index = int(cache_row["input_index"])
        iid = str(cache_row["iid"])
        positive = bool(arrays["positive"][array_index])
        side_records: dict[str, dict[str, Any]] = {}
        camera_flags: dict[str, bool] = {}
        for side in SIDES:
            camera_valid = bool(
                arrays[f"{side}_camera_valid"][array_index]
            )
            camera_flags[side] = camera_valid
            if camera_valid:
                side_records[side] = _evaluate_side(
                    arrays[f"{side}_stabilized_tracks"][array_index],
                    arrays[f"{side}_visibility"][array_index],
                    arrays[f"{side}_frame_times"][array_index],
                    selector_config=config.selector,
                    event_config=event_config,
                )
            else:
                side_records[side] = _camera_invalid_side()
        if positive and camera_flags["target"]:
            seed = _audit_seed(iid, config.seed)
            audit = _evaluate_audit(
                base=side_records["target"],
                tracks=arrays["target_stabilized_tracks"][array_index],
                visibility=arrays["target_visibility"][array_index],
                frame_times=arrays["target_frame_times"][array_index],
                seed=seed,
                config=config,
                event_config=event_config,
            )
        else:
            audit = _ineligible_audit(
                "not_positive"
                if not positive
                else "target_camera_cache_invalid"
            )
        output.append(
            {
                "schema_version": R7_P1_DIAGNOSTIC_ROW_SCHEMA,
                "input_index": input_index,
                "iid": iid,
                "cache_row_sha256": _object_digest(cache_row),
                "positive": positive,
                "label_type": cache_row.get("label_type"),
                "negative_type": cache_row.get("negative_type"),
                "action_signature": cache_row.get("action_signature"),
                "source_camera_valid": camera_flags["source"],
                "target_camera_valid": camera_flags["target"],
                "source": side_records["source"],
                "target": side_records["target"],
                "target_audit": audit,
                "formal_status": "INSUFFICIENT",
                "production_decision": False,
                "generation_authorized": False,
            }
        )
    return output


def _finite_json_value(value: Any, *, path: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} is non-finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_json_value(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json_value(child, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} has unsupported type {type(value).__name__}")


def validate_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> None:
    cache_rows = cache["rows"]
    arrays = cache["arrays"]
    track_count = int(cache["contract"]["tracker"]["track_count"])
    phase_steps = int(config.selector.phase_steps)
    if len(rows) != len(cache_rows):
        raise ValueError("diagnostic/cache row count differs")
    seen: set[str] = set()
    for index, (row, cache_row) in enumerate(zip(rows, cache_rows)):
        if row.get("schema_version") != R7_P1_DIAGNOSTIC_ROW_SCHEMA:
            raise ValueError(f"diagnostic row {index} schema differs")
        if row.get("input_index") != int(cache_row["input_index"]):
            raise ValueError(f"diagnostic row {index} input index differs")
        iid = str(cache_row["iid"])
        if row.get("iid") != iid or iid in seen:
            raise ValueError(f"diagnostic row {index} iid differs/duplicates")
        seen.add(iid)
        if row.get("cache_row_sha256") != _object_digest(cache_row):
            raise ValueError(f"diagnostic row {index} cache digest differs")
        expected_positive = bool(arrays["positive"][index])
        if row.get("positive") is not expected_positive:
            raise ValueError(f"diagnostic row {index} positive flag differs")
        for side in SIDES:
            expected_camera = bool(
                arrays[f"{side}_camera_valid"][index]
            )
            if row.get(f"{side}_camera_valid") is not expected_camera:
                raise ValueError(
                    f"diagnostic row {index} {side} camera flag differs"
                )
            record = row.get(side)
            if not isinstance(record, Mapping):
                raise ValueError(f"diagnostic row {index} lacks {side}")
            if bool(record.get("diagnostic_ready")) and not expected_camera:
                raise ValueError(
                    f"diagnostic row {index} ready without camera cache"
                )
            mask = record.get("actor_track_mask")
            if expected_camera:
                if not isinstance(mask, list) or len(mask) != track_count:
                    raise ValueError(
                        f"diagnostic row {index} {side} actor mask differs"
                    )
                selector = record.get("selector")
                if not isinstance(selector, Mapping):
                    raise ValueError(
                        f"diagnostic row {index} {side} selector missing"
                    )
                if selector.get("schema_version") != R7_COHERENT_ACTOR_SCHEMA:
                    raise ValueError(
                        f"diagnostic row {index} {side} selector schema differs"
                    )
            elif mask != [] or record.get("selector") is not None:
                raise ValueError(
                    f"diagnostic row {index} invalid-camera outputs differ"
                )
            if bool(record.get("selector_ready")):
                trajectory = np.asarray(
                    record.get("actor_trajectory"), dtype=np.float64
                )
                times = np.asarray(
                    record.get("phase_times"), dtype=np.float64
                )
                if trajectory.shape != (phase_steps, 2):
                    raise ValueError(
                        f"diagnostic row {index} trajectory shape differs"
                    )
                if times.shape != (phase_steps,) or bool(
                    (np.diff(times) <= 0.0).any()
                ):
                    raise ValueError(
                        f"diagnostic row {index} phase times differ"
                    )
                event_energy = np.asarray(
                    record.get("event_transition_energy"),
                    dtype=np.float64,
                )
                if (
                    event_energy.shape != (phase_steps - 1,)
                    or bool((event_energy < 0.0).any())
                ):
                    raise ValueError(
                        f"diagnostic row {index} event energy differs"
                    )
            if bool(record.get("diagnostic_ready")):
                if record.get("event_window") is None:
                    raise ValueError(
                        f"diagnostic row {index} ready without event"
                    )
        audit = row.get("target_audit")
        if not isinstance(audit, Mapping):
            raise ValueError(f"diagnostic row {index} lacks target audit")
        eligible = expected_positive and bool(
            arrays["target_camera_valid"][index]
        )
        if bool(audit.get("eligible")) is not eligible:
            raise ValueError(f"diagnostic row {index} audit eligibility differs")
        if bool(audit.get("performed")) is not eligible:
            raise ValueError(
                f"diagnostic row {index} audit was selection-masked"
            )
        if eligible and audit.get("seed") != _audit_seed(iid, config.seed):
            raise ValueError(f"diagnostic row {index} audit seed differs")
        if bool(audit.get("joint_pass")) and not bool(
            audit.get("comparison_available")
        ):
            raise ValueError(
                f"diagnostic row {index} unavailable audit passed"
            )
        if not bool(audit.get("comparison_available")):
            if audit.get("metrics") != _audit_defaults():
                raise ValueError(
                    f"diagnostic row {index} missing audit gained credit"
                )
        for flag in (
            "formal_status",
            "production_decision",
            "generation_authorized",
        ):
            expected = "INSUFFICIENT" if flag == "formal_status" else False
            if row.get(flag) != expected:
                raise ValueError(
                    f"diagnostic row {index} {flag} is unsafe"
                )
        _finite_json_value(row, path=f"row[{index}]")


def _failure_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for side in SIDES:
        counter = Counter(
            str(row[side].get("failure_reason") or "ready")
            for row in rows
        )
        output[side] = dict(sorted(counter.items()))
    audit = Counter(
        str(row["target_audit"].get("failure_reason") or "passed")
        for row in rows
        if row["target_audit"].get("eligible")
    )
    output["target_audit"] = dict(sorted(audit.items()))
    return output


def _make_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> dict[str, Any]:
    gate = compute_p1_gate(rows, config=config.gate)
    return {
        "schema_version": R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
        "rows": len(rows),
        "contract": dict(contract),
        "contract_sha256": _object_digest(contract),
        "rows_object_sha256": _object_digest(list(rows)),
        "failure_counts": _failure_counts(rows),
        "gate": gate,
        "formal_status": "INSUFFICIENT",
        "formal_reason": DEVELOPMENT_SCOPE,
        "production_decision": False,
        "generation_authorized": False,
    }


def _commit(
    *,
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    cache: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> dict[str, Any]:
    paths = _output_paths(output_directory)
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite P1 diagnostic artifacts: "
            + ", ".join(existing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    canonical_rows = [dict(row) for row in rows]
    validate_diagnostic_rows(
        canonical_rows,
        cache=cache,
        config=config,
    )
    summary = _make_summary(
        rows=canonical_rows,
        contract=contract,
        config=config,
    )
    _atomic_jsonl(paths["rows"], canonical_rows)
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R7_P1_DIAGNOSTIC_DONE_SCHEMA,
        "committed": True,
        "rows": len(canonical_rows),
        "rows_sha256": _file_digest(paths["rows"]),
        "summary_sha256": _file_digest(paths["summary"]),
        "contract_sha256": summary["contract_sha256"],
        "diagnostic_gate_passed": summary["gate"][
            "diagnostic_gate_passed"
        ],
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_json(paths["done"], done)
    return done


def validate_output_commit(
    *,
    output_directory: Path,
    expected_contract: Mapping[str, Any],
    cache: Mapping[str, Any],
    config: P1DiagnosticConfig,
) -> dict[str, Any]:
    """Byte-revalidate an existing complete result without recomputation."""

    paths = _output_paths(output_directory)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    done = _load_json(paths["done"])
    summary = _load_json(paths["summary"])
    if done.get("schema_version") != R7_P1_DIAGNOSTIC_DONE_SCHEMA:
        raise ValueError("P1 done schema differs")
    if done.get("committed") is not True:
        raise ValueError("P1 commit is incomplete")
    if done.get("rows_sha256") != _file_digest(paths["rows"]):
        raise ValueError("P1 rows byte digest differs")
    if done.get("summary_sha256") != _file_digest(paths["summary"]):
        raise ValueError("P1 summary byte digest differs")
    rows = _load_rows(paths["rows"])
    validate_diagnostic_rows(rows, cache=cache, config=config)
    expected_summary = _make_summary(
        rows=rows,
        contract=expected_contract,
        config=config,
    )
    if summary != expected_summary:
        raise ValueError("P1 summary/contract/recomputed gate differs")
    if done.get("rows") != len(rows):
        raise ValueError("P1 done row count differs")
    if done.get("contract_sha256") != _object_digest(expected_contract):
        raise ValueError("P1 done contract digest differs")
    if done.get("diagnostic_gate_passed") is not bool(
        expected_summary["gate"]["diagnostic_gate_passed"]
    ):
        raise ValueError("P1 done gate flag differs")
    for key, expected in (
        ("formal_status", "INSUFFICIENT"),
        ("production_decision", False),
        ("generation_authorized", False),
    ):
        if done.get(key) != expected:
            raise ValueError(f"P1 done {key} is unsafe")
    expected_done = {
        "schema_version": R7_P1_DIAGNOSTIC_DONE_SCHEMA,
        "committed": True,
        "rows": len(rows),
        "rows_sha256": _file_digest(paths["rows"]),
        "summary_sha256": _file_digest(paths["summary"]),
        "contract_sha256": _object_digest(expected_contract),
        "diagnostic_gate_passed": expected_summary["gate"][
            "diagnostic_gate_passed"
        ],
        "formal_status": "INSUFFICIENT",
        "production_decision": False,
        "generation_authorized": False,
    }
    if done != expected_done:
        raise ValueError("P1 done fields/values differ")
    return {"done": done, "summary": summary, "rows": rows}


def run_diagnostic(
    *,
    input_manifest: Path,
    cache_root: Path,
    output_directory: Path,
    config: P1DiagnosticConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cfg = config or P1DiagnosticConfig()
    cfg.validate()
    cache = load_final_cache(
        input_manifest=input_manifest,
        cache_root=cache_root,
    )
    contract = build_diagnostic_contract(
        input_manifest=input_manifest,
        cache_root=cache_root,
        cache=cache,
        config=cfg,
    )
    paths = _output_paths(output_directory)
    if paths["done"].exists():
        if not resume:
            raise FileExistsError(paths["done"])
        return validate_output_commit(
            output_directory=output_directory,
            expected_contract=contract,
            cache=cache,
            config=cfg,
        )["done"]
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"partial P1 diagnostic cannot be resumed: {output_directory}"
        )
    rows = evaluate_cache(cache=cache, config=cfg)
    return _commit(
        output_directory=output_directory,
        rows=rows,
        contract=contract,
        cache=cache,
        config=cfg,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=P1DiagnosticConfig().seed)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    done = run_diagnostic(
        input_manifest=args.input_manifest,
        cache_root=args.cache_dir,
        output_directory=args.output_dir,
        config=P1DiagnosticConfig(seed=args.seed),
        resume=args.resume,
    )
    print(json.dumps(done, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_DOMAIN",
    "DiagnosticGateConfig",
    "DownstreamAuditConfig",
    "P1DiagnosticConfig",
    "R7_P1_DIAGNOSTIC_SCHEMA",
    "R7_P1_DIAGNOSTIC_ROW_SCHEMA",
    "R7_P1_GATE_SCHEMA",
    "build_diagnostic_contract",
    "compute_p1_gate",
    "deterministic_downstream_perturbation",
    "evaluate_cache",
    "load_final_cache",
    "main",
    "run_diagnostic",
    "validate_diagnostic_rows",
    "validate_output_commit",
]

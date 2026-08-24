"""R10A cross-fitted group search for reusable dynamic representations.

R9 established that weighted camera-compensated track statistics do not
separate action from content on the current Goku cohort.  R10A therefore
tests a qualitatively different, still inexpensive representation family:

* signed temporal DCT descriptors of per-frame DINOv2 CLS dynamics;
* source-to-target dynamic quotients;
* motion-active event descriptors aligned by camera-compensated tracks;
* train-only content residualization and a small closed-form action head.

This is deliberately a *proxy* ceiling test, not an implementation of
Motive's generator-gradient representation.  Motive represents each clip by
a projected, normalized motion-weighted parameter gradient.  R10A uses the
already sealed six-frame DINO and track caches to decide whether a cheaper
dynamic representation contains enough cross-content information to justify
the substantially more expensive frozen-model tangent experiment.

The legacy R7 test split has already informed the decision to create R10 and
is therefore not a fresh promotion holdout.  Model selection uses repeated
appearance-grouped development folds.  The legacy test is evaluated once
after the spec is frozen, as diagnostic evidence only.  Consequently this
module can never authorize a renderer probe or editor training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_candidate_temporal_screen as r7


SEARCH_SCHEMA = "motive-r10a-dynamic-dino-representation-search-v1"
TRIAL_SCHEMA = "motive-r10a-dynamic-dino-representation-trial-v1"
FOLD_SCHEMA = "motive-r10a-appearance-group-fold-v1"
FAILURE_SCHEMA = "motive-r10a-representation-failure-v1"
DONE_SCHEMA = "motive-r10a-dynamic-dino-representation-done-v1"
TRANSFORM_SCHEMA = "motive-r10a-frozen-transform-v1"

TRIALS_NAME = "trials.jsonl"
FOLDS_NAME = "folds.jsonl"
FAILURES_NAME = "failure_memory.jsonl"
PREDICTIONS_NAME = "champion_predictions.jsonl"
SUMMARY_NAME = "summary.json"
TRANSFORM_NAME = "frozen_transform.npz"
DONE_NAME = "done.json"
OUTPUT_NAMES = (
    TRIALS_NAME,
    FOLDS_NAME,
    FAILURES_NAME,
    PREDICTIONS_NAME,
    SUMMARY_NAME,
    TRANSFORM_NAME,
    DONE_NAME,
)
PAYLOAD_NAMES = OUTPUT_NAMES[:-1]

DEFAULT_SEED = 260108837
DEFAULT_REPEATS = 2
DEFAULT_FOLDS = 3
DEFAULT_DINO_CHANNEL_DIM = 64
DEFAULT_APPEARANCE_CLUSTERS = 96

MIN_COHORT_COVERAGE = 0.90
MIN_QUERY_COVERAGE = 0.90
MIN_CONTROL_MARGIN = 0.02
MIN_TEMPORAL_MARGIN = 0.02
MIN_AUROC = 0.55
MIN_AUROC_OVER_ENERGY = 0.01
MAX_APPEARANCE_SIMILARITY_CORRELATION = 0.80
MIN_DEVELOPMENT_FOLD_PASS_FRACTION = 0.80
MIN_STABLE_FAMILIES = 8
_EPS = 1e-12

CONTROL_CLEAN = "clean"
CONTROL_SHUFFLE = "shuffle"
CONTROL_REVERSE = "reverse"
CONTROLS = (CONTROL_CLEAN, CONTROL_SHUFFLE, CONTROL_REVERSE)

TRACK_ACCELERATION = "track_target_acceleration"
TRACK_DELTA_ACCELERATION = "track_edit_acceleration"
TRACK_SIGNED_SUMMARY = "track_target_signed_direction_speed_duration"
TRACK_DELTA_SIGNED_SUMMARY = "track_edit_signed_direction_speed_duration"
TRACK_EVENT_DCT = "track_target_active_event_signed_dct"
TRACK_DELTA_EVENT_DCT = "track_edit_active_event_signed_dct"
DINO_TARGET_DCT = "dino_target_signed_dct"
DINO_DELTA_DCT = "dino_edit_signed_dct"
DINO_TARGET_EVENT_DCT = "dino_target_active_event_signed_dct"
DINO_DELTA_EVENT_DCT = "dino_edit_active_event_signed_dct"


class R10DynamicRepresentationError(ValueError):
    """An input, fold, fit, or immutable publication contract is invalid."""


@dataclass(frozen=True)
class _R10Example:
    iid: str
    label_class: str
    family: str
    original_split: str
    component_id: str
    fresh: bool
    sampling_weight: float
    motion_energy: float
    source_dino: np.ndarray
    target_dino: np.ndarray
    source_track: np.ndarray
    target_track: np.ndarray
    reverse_source_track: np.ndarray
    reverse_target_track: np.ndarray
    pooled_target_dino: np.ndarray
    target_endpoint: np.ndarray
    target_orderless: np.ndarray
    camera_nuisance: np.ndarray


@dataclass(frozen=True)
class _Fold:
    fold_id: str
    repeat: int
    fold: int
    train_indices: tuple[int, ...]
    query_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    query_groups: tuple[str, ...]


@dataclass(frozen=True)
class _DinoBasis:
    mean: np.ndarray
    basis: np.ndarray


@dataclass(frozen=True)
class _FittedTransform:
    spec_digest: str
    raw_mean: np.ndarray
    raw_scale: np.ndarray
    appearance_mean: np.ndarray
    appearance_scale: np.ndarray
    content_ridge: np.ndarray
    projection: np.ndarray
    action_head: np.ndarray
    action_families: tuple[str, ...]
    geometry_keep: int
    raw_dimension: int
    embedding_dimension: int


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(dict(row)) + "\n").encode("utf-8")
        for row in rows
    )


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_digest(value: Any) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"))


def _stable_u32(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _unit_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise R10DynamicRepresentationError(
            "row normalization requires one finite matrix"
        )
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > _EPS
    output = np.zeros_like(matrix)
    output[valid] = matrix[valid] / norms[valid, None]
    return output, valid


def _unit_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(vector) or not np.isfinite(vector).all():
        raise R10DynamicRepresentationError(
            "vector normalization requires a finite nonempty vector"
        )
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > _EPS else np.zeros_like(vector)


def _canonicalize_basis(basis: np.ndarray) -> np.ndarray:
    output = np.asarray(basis, dtype=np.float64).copy()
    if output.ndim != 2:
        raise R10DynamicRepresentationError("basis must be a matrix")
    for column in range(output.shape[1]):
        values = output[:, column]
        pivot = int(np.argmax(np.abs(values)))
        if values[pivot] < 0.0:
            output[:, column] *= -1.0
    return output


def _dct_basis(length: int) -> np.ndarray:
    """Return the orthonormal DCT-II matrix with rows ordered by frequency."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise R10DynamicRepresentationError("DCT length must be >= 2")
    time_index = np.arange(length, dtype=np.float64)
    frequency = np.arange(length, dtype=np.float64)[:, None]
    basis = np.cos(
        math.pi * (time_index[None, :] + 0.5) * frequency / length
    )
    basis[0] *= math.sqrt(1.0 / length)
    basis[1:] *= math.sqrt(2.0 / length)
    return basis


def _resample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if (
        values.ndim != 2
        or not len(values)
        or not np.isfinite(values).all()
        or length < 2
    ):
        raise R10DynamicRepresentationError(
            "invalid temporal resampling input"
        )
    if len(values) == length:
        return values.copy()
    old = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    new = np.linspace(0.0, 1.0, length, dtype=np.float64)
    result = np.empty((length, values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(new, old, values[:, column])
    return result


def _signed_dct(
    sequence: np.ndarray,
    *,
    coefficients: int,
    include_dc: bool,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise R10DynamicRepresentationError("invalid signed-DCT sequence")
    basis = _dct_basis(len(values))
    start = 0 if include_dc else 1
    stop = min(len(values), start + coefficients)
    if stop <= start:
        raise R10DynamicRepresentationError("DCT has no selected band")
    return (basis[start:stop] @ values).reshape(-1)


def _active_interval(track_sequence: np.ndarray) -> tuple[int, int]:
    """Select one deterministic contiguous motion event from track energy."""

    values = np.asarray(track_sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15 or len(values) < 3:
        raise R10DynamicRepresentationError(
            "track event sequence must be [T,15]"
        )
    energy = np.maximum(values[:, 13], 0.0)
    threshold = float(np.quantile(energy, 0.60))
    active = energy >= max(threshold, _EPS)
    best = (0, len(values))
    best_key: tuple[float, int, int] | None = None
    begin = 0
    while begin < len(active):
        if not bool(active[begin]):
            begin += 1
            continue
        end = begin + 1
        while end < len(active) and bool(active[end]):
            end += 1
        expanded_begin = max(0, begin - 1)
        expanded_end = min(len(active), end + 1)
        key = (
            float(np.sum(energy[expanded_begin:expanded_end])),
            expanded_end - expanded_begin,
            -expanded_begin,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (expanded_begin, expanded_end)
        begin = end
    return best


def _track_signed_summary(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 15 or len(values) < 2:
        raise R10DynamicRepresentationError(
            "signed track summary requires [T,15]"
        )
    begin, end = _active_interval(values)
    signed = values[:, :10]
    speed = np.maximum(values[:, 10:15], 0.0)
    return np.concatenate(
        (
            np.mean(signed, axis=0),
            np.sum(signed, axis=0),
            np.mean(speed, axis=0),
            np.asarray(
                [
                    (end - begin) / len(values),
                    float(np.mean(speed[:, 3])),
                ]
            ),
        )
    )


def _mapped_interval(
    begin: int,
    end: int,
    *,
    source_length: int,
    target_length: int,
) -> tuple[int, int]:
    left = int(math.floor(begin * target_length / source_length))
    right = int(math.ceil(end * target_length / source_length))
    left = min(max(left, 0), target_length - 1)
    right = min(max(right, left + 2), target_length)
    if right - left < 2:
        left = max(0, min(left, target_length - 2))
        right = min(target_length, left + 2)
    return left, right


def _controlled_sequences(
    example: _R10Example,
    *,
    control: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if control not in CONTROLS:
        raise R10DynamicRepresentationError(
            f"unknown temporal control: {control}"
        )
    source_track = example.source_track
    target_track = example.target_track
    source_dino = example.source_dino
    target_dino = example.target_dino
    if control == CONTROL_REVERSE:
        # Reverse both source and target for edit-delta representations.
        # This avoids the R9 bug where delta controls silently reused clean.
        source_track = example.reverse_source_track
        target_track = example.reverse_target_track
        source_dino = source_dino[::-1]
        target_dino = target_dino[::-1]
    elif control == CONTROL_SHUFFLE:
        track_order = r7._shuffle_indices(
            example.iid,
            len(target_track),
            seed=seed,
        )
        dino_order = r7._shuffle_indices(
            f"{example.iid}:dino",
            len(target_dino),
            seed=seed,
        )
        # Shuffle the transition/state blocks on both sides with the same
        # order.  This is a pure order ablation for edit quotients.
        source_track = source_track[track_order]
        target_track = target_track[track_order]
        source_dino = source_dino[dino_order]
        target_dino = target_dino[dino_order]
    return source_track, target_track, source_dino, target_dino


def _fit_dino_basis(
    examples: Sequence[_R10Example],
    train_indices: Sequence[int],
    *,
    maximum_dimension: int,
) -> _DinoBasis:
    tokens: list[np.ndarray] = []
    for index in train_indices:
        example = examples[index]
        if example.label_class == "positive":
            source = example.source_dino - np.mean(
                example.source_dino,
                axis=0,
                keepdims=True,
            )
            target = example.target_dino - np.mean(
                example.target_dino,
                axis=0,
                keepdims=True,
            )
            tokens.extend((source, target))
    if not tokens:
        raise R10DynamicRepresentationError(
            "DINO PCA has no positive train tokens"
        )
    matrix = np.concatenate(tokens, axis=0).astype(np.float64, copy=False)
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    dimension = min(
        maximum_dimension,
        eigenvectors.shape[0],
        len(order),
    )
    if dimension < 2:
        raise R10DynamicRepresentationError(
            "DINO train PCA has fewer than two dimensions"
        )
    basis = _canonicalize_basis(eigenvectors[:, order[:dimension]])
    return _DinoBasis(mean=mean, basis=basis)


def _fit_appearance_basis(
    examples: Sequence[_R10Example],
    train_indices: Sequence[int],
    *,
    maximum_dimension: int = 32,
) -> _DinoBasis:
    pooled: list[np.ndarray] = []
    for index in train_indices:
        example = examples[index]
        pooled.append(np.mean(example.source_dino, axis=0))
        pooled.append(np.mean(example.target_dino, axis=0))
    if not pooled:
        raise R10DynamicRepresentationError(
            "appearance PCA has no train rows"
        )
    matrix = np.stack(pooled).astype(np.float64, copy=False)
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    dimension = min(
        maximum_dimension,
        eigenvectors.shape[0],
        len(order),
    )
    if dimension < 2:
        raise R10DynamicRepresentationError(
            "appearance train PCA has fewer than two dimensions"
        )
    return _DinoBasis(
        mean=mean,
        basis=_canonicalize_basis(eigenvectors[:, order[:dimension]]),
    )


def _project_dino(
    sequence: np.ndarray,
    basis: _DinoBasis,
    *,
    dimension: int,
) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    selected = basis.basis[:, : min(dimension, basis.basis.shape[1])]
    return (values - basis.mean[None, :]) @ selected


def _raw_block(
    example: _R10Example,
    *,
    block: str,
    control: str,
    seed: int,
    dino_basis: _DinoBasis,
    dino_dimension: int,
) -> np.ndarray:
    source_track, target_track, source_dino, target_dino = (
        _controlled_sequences(example, control=control, seed=seed)
    )
    if block == TRACK_ACCELERATION:
        return np.diff(target_track, axis=0).reshape(-1)
    if block == TRACK_DELTA_ACCELERATION:
        return np.diff(target_track - source_track, axis=0).reshape(-1)
    if block == TRACK_SIGNED_SUMMARY:
        return _track_signed_summary(target_track)
    if block == TRACK_DELTA_SIGNED_SUMMARY:
        return _track_signed_summary(target_track - source_track)

    begin, end = _active_interval(target_track)
    if block in (TRACK_EVENT_DCT, TRACK_DELTA_EVENT_DCT):
        sequence = (
            target_track
            if block == TRACK_EVENT_DCT
            else target_track - source_track
        )
        event = _resample_sequence(sequence[begin:end], 8)
        event -= np.mean(event, axis=0, keepdims=True)
        return _signed_dct(
            event,
            coefficients=6,
            include_dc=False,
        )

    projected_source = _project_dino(
        source_dino,
        dino_basis,
        dimension=dino_dimension,
    )
    projected_target = _project_dino(
        target_dino,
        dino_basis,
        dimension=dino_dimension,
    )
    projected_source -= np.mean(
        projected_source,
        axis=0,
        keepdims=True,
    )
    projected_target -= np.mean(
        projected_target,
        axis=0,
        keepdims=True,
    )
    sequence = (
        projected_target
        if block in (DINO_TARGET_DCT, DINO_TARGET_EVENT_DCT)
        else projected_target - projected_source
    )
    if block in (DINO_TARGET_EVENT_DCT, DINO_DELTA_EVENT_DCT):
        dino_begin, dino_end = _mapped_interval(
            begin,
            end,
            source_length=len(target_track),
            target_length=len(sequence),
        )
        sequence = _resample_sequence(
            sequence[dino_begin:dino_end],
            6,
        )
    return _signed_dct(
        sequence,
        coefficients=min(5, len(sequence) - 1),
        include_dc=False,
    )


def _raw_matrix(
    examples: Sequence[_R10Example],
    spec: Mapping[str, Any],
    *,
    control: str,
    seed: int,
    dino_basis: _DinoBasis,
) -> np.ndarray:
    blocks = tuple(str(value) for value in spec["raw_blocks"])
    dimension = int(spec["dino_channel_dim"])
    rows: list[np.ndarray] = []
    for example in examples:
        parts = [
            _raw_block(
                example,
                block=block,
                control=control,
                seed=seed,
                dino_basis=dino_basis,
                dino_dimension=dimension,
            )
            for block in blocks
        ]
        rows.append(np.concatenate(parts))
    if not rows or any(row.shape != rows[0].shape for row in rows):
        raise R10DynamicRepresentationError(
            "raw representation dimensions differ"
        )
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all():
        raise R10DynamicRepresentationError(
            "raw representation contains non-finite values"
        )
    return matrix


def _precompute_raw_blocks(
    examples: Sequence[_R10Example],
    *,
    seed: int,
    dino_basis: _DinoBasis,
    dino_dimension: int,
    indices: Sequence[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute every structural block once for one fold-specific DINO basis."""

    names = (
        TRACK_ACCELERATION,
        TRACK_DELTA_ACCELERATION,
        TRACK_SIGNED_SUMMARY,
        TRACK_DELTA_SIGNED_SUMMARY,
        TRACK_EVENT_DCT,
        TRACK_DELTA_EVENT_DCT,
        DINO_TARGET_DCT,
        DINO_DELTA_DCT,
        DINO_TARGET_EVENT_DCT,
        DINO_DELTA_EVENT_DCT,
    )
    pending: dict[str, dict[str, list[tuple[int, np.ndarray]]]] = {
        control: {name: [] for name in names} for control in CONTROLS
    }
    selected_indices = (
        tuple(range(len(examples)))
        if indices is None
        else tuple(int(index) for index in indices)
    )
    for control in CONTROLS:
        for example_index in selected_indices:
            example = examples[example_index]
            source_track, target_track, source_dino, target_dino = (
                _controlled_sequences(
                    example,
                    control=control,
                    seed=seed,
                )
            )
            begin, end = _active_interval(target_track)
            track_target_event = _resample_sequence(
                target_track[begin:end],
                8,
            )
            track_target_event -= np.mean(
                track_target_event,
                axis=0,
                keepdims=True,
            )
            track_delta_event = _resample_sequence(
                (target_track - source_track)[begin:end],
                8,
            )
            track_delta_event -= np.mean(
                track_delta_event,
                axis=0,
                keepdims=True,
            )
            projected_source = _project_dino(
                source_dino,
                dino_basis,
                dimension=dino_dimension,
            )
            projected_target = _project_dino(
                target_dino,
                dino_basis,
                dimension=dino_dimension,
            )
            projected_source -= np.mean(
                projected_source,
                axis=0,
                keepdims=True,
            )
            projected_target -= np.mean(
                projected_target,
                axis=0,
                keepdims=True,
            )
            dino_delta = projected_target - projected_source
            dino_begin, dino_end = _mapped_interval(
                begin,
                end,
                source_length=len(target_track),
                target_length=len(projected_target),
            )
            dino_target_event = _resample_sequence(
                projected_target[dino_begin:dino_end],
                6,
            )
            dino_delta_event = _resample_sequence(
                dino_delta[dino_begin:dino_end],
                6,
            )
            values = {
                TRACK_ACCELERATION: np.diff(
                    target_track,
                    axis=0,
                ).reshape(-1),
                TRACK_DELTA_ACCELERATION: np.diff(
                    target_track - source_track,
                    axis=0,
                ).reshape(-1),
                TRACK_SIGNED_SUMMARY: _track_signed_summary(target_track),
                TRACK_DELTA_SIGNED_SUMMARY: _track_signed_summary(
                    target_track - source_track
                ),
                TRACK_EVENT_DCT: _signed_dct(
                    track_target_event,
                    coefficients=6,
                    include_dc=False,
                ),
                TRACK_DELTA_EVENT_DCT: _signed_dct(
                    track_delta_event,
                    coefficients=6,
                    include_dc=False,
                ),
                DINO_TARGET_DCT: _signed_dct(
                    projected_target,
                    coefficients=min(5, len(projected_target) - 1),
                    include_dc=False,
                ),
                DINO_DELTA_DCT: _signed_dct(
                    dino_delta,
                    coefficients=min(5, len(dino_delta) - 1),
                    include_dc=False,
                ),
                DINO_TARGET_EVENT_DCT: _signed_dct(
                    dino_target_event,
                    coefficients=min(5, len(dino_target_event) - 1),
                    include_dc=False,
                ),
                DINO_DELTA_EVENT_DCT: _signed_dct(
                    dino_delta_event,
                    coefficients=min(5, len(dino_delta_event) - 1),
                    include_dc=False,
                ),
            }
            for name, value in values.items():
                pending[control][name].append((example_index, value))
    output: dict[str, dict[str, np.ndarray]] = {}
    for control in CONTROLS:
        output[control] = {}
        for name in names:
            rows = pending[control][name]
            if not rows or any(
                row.shape != rows[0][1].shape for _index, row in rows
            ):
                raise R10DynamicRepresentationError(
                    f"precomputed block dimensions differ: {control}/{name}"
                )
            matrix = np.zeros(
                (len(examples), rows[0][1].shape[0]),
                dtype=np.float64,
            )
            for example_index, row in rows:
                matrix[example_index] = row
            if not np.isfinite(matrix).all():
                raise R10DynamicRepresentationError(
                    f"precomputed block is non-finite: {control}/{name}"
                )
            output[control][name] = matrix
    return output


def _combine_precomputed_raw(
    spec: Mapping[str, Any],
    block_cache: Mapping[str, Mapping[str, np.ndarray]],
    *,
    control: str,
) -> np.ndarray:
    blocks = tuple(str(value) for value in spec["raw_blocks"])
    try:
        values = [block_cache[control][block] for block in blocks]
    except KeyError as error:
        raise R10DynamicRepresentationError(
            f"precomputed block is missing: {error}"
        ) from error
    return np.concatenate(values, axis=1)


def _appearance_matrix(
    examples: Sequence[_R10Example],
    dino_basis: _DinoBasis,
    *,
    dimension: int = 32,
    indices: Sequence[int] | None = None,
) -> np.ndarray:
    selected = dino_basis.basis[
        :,
        : min(dimension, dino_basis.basis.shape[1]),
    ]
    output = np.zeros(
        (len(examples), 2 * selected.shape[1]),
        dtype=np.float64,
    )
    selected_indices = (
        tuple(range(len(examples)))
        if indices is None
        else tuple(int(index) for index in indices)
    )
    for index in selected_indices:
        example = examples[index]
        source = (
            np.mean(example.source_dino, axis=0) - dino_basis.mean
        ) @ selected
        target = (
            np.mean(example.target_dino, axis=0) - dino_basis.mean
        ) @ selected
        output[index] = np.concatenate((source, target))
    return output


def _spec(
    raw_blocks: Sequence[str],
    *,
    name: str,
    standardize: bool,
    content_residual: bool,
    projection: str,
    projection_dim: int,
    head: str,
    ridge: float,
    geometry_keep: int = 0,
    dino_channel_dim: int = 64,
) -> dict[str, Any]:
    core = {
        "schema_version": "motive-r10a-dynamic-spec-v1",
        "name": name,
        "raw_blocks": list(raw_blocks),
        "dino_channel_dim": int(dino_channel_dim),
        "standardize": bool(standardize),
        "content_residual": bool(content_residual),
        "content_covariate": (
            "source_and_target_pooled_dino_train_pca32"
            if content_residual
            else "none"
        ),
        "projection": projection,
        "projection_dim": int(projection_dim),
        "head": head,
        "ridge": float(ridge),
        "geometry_keep": int(geometry_keep),
        "similarity": "cosine",
        "selection_role": (
            "reusable_representation_candidate"
            if head == "identity"
            else "closed_set_supervised_upper_bound"
        ),
        "champion_eligible": head == "identity",
    }
    return {**core, "spec_digest": _object_digest(core)}


def _candidate_specs() -> list[dict[str, Any]]:
    """Finite structural grid; R9-style weight mutation is intentionally gone."""

    candidates = [
        _spec(
            [TRACK_ACCELERATION],
            name="r9_track_acceleration_exact_control",
            standardize=False,
            content_residual=False,
            projection="identity",
            projection_dim=0,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [TRACK_DELTA_ACCELERATION],
            name="track_edit_acceleration",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [TRACK_SIGNED_SUMMARY, TRACK_EVENT_DCT],
            name="track_active_event_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [TRACK_DELTA_SIGNED_SUMMARY, TRACK_DELTA_EVENT_DCT],
            name="track_edit_active_event_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [DINO_TARGET_DCT],
            name="dino_target_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [DINO_DELTA_DCT],
            name="dino_edit_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [DINO_DELTA_EVENT_DCT],
            name="dino_edit_active_event_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [
                TRACK_DELTA_SIGNED_SUMMARY,
                TRACK_DELTA_EVENT_DCT,
                DINO_DELTA_EVENT_DCT,
            ],
            name="hybrid_edit_active_event_dct",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.0,
        ),
        _spec(
            [DINO_DELTA_DCT],
            name="dino_edit_dct_content_residual",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.1,
        ),
        _spec(
            [
                TRACK_DELTA_SIGNED_SUMMARY,
                TRACK_DELTA_EVENT_DCT,
                DINO_DELTA_EVENT_DCT,
            ],
            name="hybrid_edit_event_content_residual",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="identity",
            ridge=0.1,
        ),
        _spec(
            [TRACK_SIGNED_SUMMARY, TRACK_EVENT_DCT],
            name="track_event_ridge_action",
            standardize=True,
            content_residual=False,
            projection="jl",
            projection_dim=128,
            head="ridge_action",
            ridge=1.0,
        ),
        _spec(
            [DINO_DELTA_DCT],
            name="dino_edit_residual_ridge_action",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="ridge_action",
            ridge=1.0,
        ),
        _spec(
            [DINO_DELTA_EVENT_DCT],
            name="dino_edit_event_residual_ridge_action",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="ridge_action",
            ridge=1.0,
        ),
        _spec(
            [
                TRACK_DELTA_SIGNED_SUMMARY,
                TRACK_DELTA_EVENT_DCT,
                DINO_DELTA_EVENT_DCT,
            ],
            name="hybrid_edit_event_residual_ridge_action",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="ridge_action",
            ridge=0.1,
        ),
        _spec(
            [
                TRACK_DELTA_SIGNED_SUMMARY,
                TRACK_DELTA_EVENT_DCT,
                DINO_DELTA_EVENT_DCT,
            ],
            name="hybrid_edit_event_residual_ridge_plus_geometry",
            standardize=True,
            content_residual=True,
            projection="jl",
            projection_dim=128,
            head="ridge_action_plus_geometry",
            ridge=1.0,
            geometry_keep=32,
        ),
    ]
    digests = [item["spec_digest"] for item in candidates]
    if len(set(digests)) != len(digests):
        raise RuntimeError("R10 candidate specs are not unique")
    return candidates


def _relative_ridge_gram(matrix: np.ndarray, ridge: float) -> np.ndarray:
    gram = matrix.T @ matrix
    scale = float(np.trace(gram) / max(len(gram), 1))
    regularization = max(float(ridge) * max(scale, _EPS), _EPS)
    return gram + regularization * np.eye(
        gram.shape[0],
        dtype=np.float64,
    )


def _jl_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int,
    family_key: str,
) -> np.ndarray:
    if input_dim <= output_dim:
        return np.eye(input_dim, dtype=np.float64)
    rng = np.random.default_rng(
        _stable_u32("r10-jl", seed, family_key, input_dim, output_dim)
    )
    signs = rng.integers(
        0,
        2,
        size=(input_dim, output_dim),
        dtype=np.int8,
    ).astype(np.float64)
    signs = 2.0 * signs - 1.0
    return signs / math.sqrt(output_dim)


def _eligible_families(
    examples: Sequence[_R10Example],
    train_indices: Sequence[int],
) -> tuple[set[str], dict[str, Any]]:
    rows = [
        examples[index]
        for index in train_indices
        if examples[index].label_class == "positive"
    ]
    counts = Counter(item.family for item in rows)
    components: dict[str, set[str]] = defaultdict(set)
    for item in rows:
        components[item.family].add(item.component_id)
    eligible = {
        family
        for family in counts
        if (
            counts[family] >= r7.MINIMUM_TRAIN_REFERENCES
            and len(components[family]) >= r7.MINIMUM_TRAIN_COMPONENTS
        )
    }
    return eligible, {
        "eligible_families": sorted(eligible),
        "families": {
            family: {
                "train_references": int(counts[family]),
                "train_components": len(components[family]),
                "eligible": family in eligible,
            }
            for family in sorted(counts)
        },
    }


def _fit_transform(
    examples: Sequence[_R10Example],
    spec: Mapping[str, Any],
    *,
    train_indices: Sequence[int],
    raw_clean: np.ndarray,
    appearance: np.ndarray,
    eligible_families: set[str],
    seed: int,
) -> _FittedTransform:
    train = np.asarray(tuple(train_indices), dtype=np.int64)
    if not len(train):
        raise R10DynamicRepresentationError("transform has no train rows")
    raw = np.asarray(raw_clean, dtype=np.float64)
    if bool(spec["standardize"]):
        raw_mean = np.mean(raw[train], axis=0)
        raw_scale = np.std(raw[train], axis=0)
        raw_scale[raw_scale <= 1e-8] = 1.0
    else:
        raw_mean = np.zeros(raw.shape[1], dtype=np.float64)
        raw_scale = np.ones(raw.shape[1], dtype=np.float64)
    standardized = (raw - raw_mean) / raw_scale

    appearance_mean = np.mean(appearance[train], axis=0)
    appearance_scale = np.std(appearance[train], axis=0)
    appearance_scale[appearance_scale <= 1e-8] = 1.0
    standardized_appearance = (
        appearance - appearance_mean
    ) / appearance_scale
    if bool(spec["content_residual"]):
        q_train = standardized_appearance[train]
        m_train = standardized[train]
        gram = _relative_ridge_gram(
            q_train,
            float(spec["ridge"]),
        )
        content_ridge = np.linalg.solve(
            gram,
            q_train.T @ m_train,
        )
        standardized = standardized - (
            standardized_appearance @ content_ridge
        )
    else:
        content_ridge = np.zeros(
            (appearance.shape[1], raw.shape[1]),
            dtype=np.float64,
        )

    if spec["projection"] == "identity":
        projection = np.eye(raw.shape[1], dtype=np.float64)
    elif spec["projection"] == "jl":
        projection = _jl_projection(
            raw.shape[1],
            int(spec["projection_dim"]),
            seed=seed,
            family_key="|".join(str(x) for x in spec["raw_blocks"]),
        )
    else:
        raise R10DynamicRepresentationError(
            f"unknown projection: {spec['projection']}"
        )
    projected = standardized @ projection

    head = str(spec["head"])
    families = tuple(sorted(eligible_families))
    if head == "identity":
        # Identity candidates expose geometry only.  Publishing closed-set
        # family names beside a zero-column head would falsely imply logits.
        families = ()
        action_head = np.zeros(
            (projected.shape[1], 0),
            dtype=np.float64,
        )
        geometry_keep = 0
        embedding_dimension = projected.shape[1]
    else:
        if len(families) < r7.MINIMUM_ELIGIBLE_FAMILIES:
            raise R10DynamicRepresentationError(
                "action head lacks two eligible train families"
            )
        family_index = {family: index for index, family in enumerate(families)}
        selected = [
            index
            for index in train_indices
            if (
                examples[index].label_class == "positive"
                and examples[index].family in family_index
            )
        ]
        z_train = projected[np.asarray(selected, dtype=np.int64)]
        y = np.zeros((len(selected), len(families)), dtype=np.float64)
        class_counts = Counter(examples[index].family for index in selected)
        weights = np.empty(len(selected), dtype=np.float64)
        for row, example_index in enumerate(selected):
            family = examples[example_index].family
            y[row, family_index[family]] = 1.0
            weights[row] = (
                float(examples[example_index].sampling_weight)
                / class_counts[family]
            )
        weights /= float(np.mean(weights))
        root = np.sqrt(weights)
        weighted_z = z_train * root[:, None]
        weighted_y = y * root[:, None]
        gram = _relative_ridge_gram(
            weighted_z,
            float(spec["ridge"]),
        )
        action_head = np.linalg.solve(
            gram,
            weighted_z.T @ weighted_y,
        )
        geometry_keep = (
            min(int(spec["geometry_keep"]), projected.shape[1])
            if head == "ridge_action_plus_geometry"
            else 0
        )
        embedding_dimension = len(families) + geometry_keep
    return _FittedTransform(
        spec_digest=str(spec["spec_digest"]),
        raw_mean=raw_mean,
        raw_scale=raw_scale,
        appearance_mean=appearance_mean,
        appearance_scale=appearance_scale,
        content_ridge=content_ridge,
        projection=projection,
        action_head=action_head,
        action_families=families,
        geometry_keep=geometry_keep,
        raw_dimension=raw.shape[1],
        embedding_dimension=embedding_dimension,
    )


def _encode_transform(
    raw: np.ndarray,
    appearance: np.ndarray,
    fitted: _FittedTransform,
) -> tuple[np.ndarray, np.ndarray]:
    standardized = (raw - fitted.raw_mean) / fitted.raw_scale
    standardized_appearance = (
        appearance - fitted.appearance_mean
    ) / fitted.appearance_scale
    standardized = standardized - (
        standardized_appearance @ fitted.content_ridge
    )
    projected = standardized @ fitted.projection
    if fitted.action_head.shape[1]:
        output = projected @ fitted.action_head
        if fitted.geometry_keep:
            output = np.concatenate(
                (
                    output,
                    0.25 * projected[:, : fitted.geometry_keep],
                ),
                axis=1,
            )
    else:
        output = projected
    return _unit_rows(output)


def _appearance_groups(
    examples: Sequence[_R10Example],
    *,
    maximum_groups: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Cluster development components by frozen source appearance.

    The clustering is label-free and deliberately coarser than the upstream
    near-duplicate component graph.  It is only a split nuisance control; its
    IDs are never supplied to the representation.
    """

    dev_indices = [
        index
        for index, example in enumerate(examples)
        if example.original_split != "test"
    ]
    by_component: dict[str, list[int]] = defaultdict(list)
    for index in dev_indices:
        by_component[examples[index].component_id].append(index)
    components = sorted(by_component)
    if len(components) < 6:
        raise R10DynamicRepresentationError(
            "appearance grouping needs at least six development components"
        )
    input_dim = examples[dev_indices[0]].source_dino.shape[1]
    output_dim = min(64, input_dim)
    projection = _jl_projection(
        input_dim,
        output_dim,
        seed=DEFAULT_SEED,
        family_key="appearance-content-group-v1",
    )
    vectors: list[np.ndarray] = []
    for component in components:
        pooled = np.stack(
            [
                np.mean(examples[index].source_dino, axis=0)
                for index in by_component[component]
            ]
        )
        vectors.append(np.mean(pooled, axis=0) @ projection)
    matrix, valid = _unit_rows(np.stack(vectors))
    if not bool(valid.all()):
        raise R10DynamicRepresentationError(
            "source-appearance component centroid is zero"
        )
    cluster_count = min(
        maximum_groups,
        max(6, int(math.ceil(math.sqrt(len(components)) * 2.5))),
        len(components),
    )

    # Deterministic farthest-first initialization in the projected sphere.
    center_indices = [0]
    nearest_similarity = matrix @ matrix[0]
    for _ in range(1, cluster_count):
        candidate = min(
            (
                index
                for index in range(len(components))
                if index not in set(center_indices)
            ),
            key=lambda index: (
                float(nearest_similarity[index]),
                components[index],
            ),
        )
        center_indices.append(candidate)
        nearest_similarity = np.maximum(
            nearest_similarity,
            matrix @ matrix[candidate],
        )
    centers = matrix[np.asarray(center_indices, dtype=np.int64)].copy()
    assignment = np.zeros(len(components), dtype=np.int64)
    for _ in range(8):
        scores = matrix @ centers.T
        updated = np.argmax(scores, axis=1)
        if np.array_equal(updated, assignment):
            assignment = updated
            break
        assignment = updated
        new_centers = centers.copy()
        for cluster in range(cluster_count):
            selected = matrix[assignment == cluster]
            if len(selected):
                center = np.mean(selected, axis=0)
                norm = float(np.linalg.norm(center))
                if norm > _EPS:
                    new_centers[cluster] = center / norm
        centers = new_centers

    members: dict[int, list[str]] = defaultdict(list)
    for component, cluster in zip(components, assignment.tolist()):
        members[int(cluster)].append(component)
    canonical_group = {
        cluster: (
            "appearance-group-"
            + hashlib.sha256(
                "\n".join(sorted(values)).encode("utf-8")
            ).hexdigest()[:16]
        )
        for cluster, values in members.items()
    }
    index_to_group: dict[int, str] = {}
    component_to_group: dict[str, str] = {}
    for component, cluster in zip(components, assignment.tolist()):
        group = canonical_group[int(cluster)]
        component_to_group[component] = group
        for index in by_component[component]:
            index_to_group[index] = group
    group_sizes = Counter(index_to_group.values())
    return index_to_group, {
        "schema_version": "motive-r10a-source-appearance-groups-v1",
        "development_rows": len(dev_indices),
        "development_components": len(components),
        "groups": len(group_sizes),
        "minimum_group_rows": min(group_sizes.values()),
        "maximum_group_rows": max(group_sizes.values()),
        "median_group_rows": float(np.median(list(group_sizes.values()))),
        "source_embedding": (
            "mean six frozen source DINO CLS; fixed JL64; spherical k-means"
        ),
        "label_free": True,
        "component_never_split": True,
        "test_used_for_grouping": False,
        "projection_seed": DEFAULT_SEED,
        "iterations": 8,
    }


def _fold_assignment_rows_digest(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Commit to fold membership without making the seed part of the digest."""

    commitments: list[dict[str, str]] = []
    fold_ids: set[str] = set()
    for position, row in enumerate(rows):
        fold_id = row.get("fold_id")
        if (
            not isinstance(fold_id, str)
            or not fold_id
            or fold_id in fold_ids
        ):
            raise R10DynamicRepresentationError(
                f"fold assignment row {position} has an invalid fold ID"
            )
        fold_ids.add(fold_id)
        commitment = row.get("assignment_commitment")
        if not isinstance(commitment, Mapping):
            raise R10DynamicRepresentationError(
                f"fold {fold_id} lacks an assignment commitment"
            )
        normalized: dict[str, str] = {"fold_id": fold_id}
        for name in (
            "query_group_ids_sha256",
            "query_iids_sha256",
            "query_component_ids_sha256",
        ):
            digest = commitment.get(name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise R10DynamicRepresentationError(
                    f"fold {fold_id} has an invalid {name}"
                )
            normalized[name] = digest
        commitments.append(normalized)
    if not commitments:
        raise R10DynamicRepresentationError(
            "fold assignment commitment is empty"
        )
    return _object_digest(
        sorted(commitments, key=lambda value: value["fold_id"])
    )


def _make_folds(
    examples: Sequence[_R10Example],
    *,
    index_to_group: Mapping[int, str],
    seed: int,
    repeats: int,
    folds: int,
) -> tuple[list[_Fold], list[dict[str, Any]]]:
    if repeats < 1 or folds < 2:
        raise R10DynamicRepresentationError("invalid repeated-fold budget")
    dev_indices = tuple(
        index
        for index, example in enumerate(examples)
        if example.original_split != "test"
    )
    groups: dict[str, list[int]] = defaultdict(list)
    for index in dev_indices:
        groups[str(index_to_group[index])].append(index)
    forced_groups = {
        group
        for group, indices in groups.items()
        if any(not examples[index].fresh for index in indices)
    }
    movable_groups = tuple(
        group for group in sorted(groups) if group not in forced_groups
    )
    if len(movable_groups) < folds:
        raise R10DynamicRepresentationError(
            "appearance folds have fewer movable groups than folds"
        )
    output: list[_Fold] = []
    records: list[dict[str, Any]] = []
    for repeat in range(repeats):
        assignment = _stratified_group_assignment(
            examples,
            groups=groups,
            movable_groups=movable_groups,
            folds=folds,
            seed=seed,
            namespace=f"r10-outer-repeat-{repeat}",
        )
        assignment.update({group: None for group in forced_groups})
        for fold_index in range(folds):
            query_groups = tuple(
                group
                for group in sorted(groups)
                if assignment[group] == fold_index
            )
            train_groups = tuple(
                group
                for group in sorted(groups)
                if group not in set(query_groups)
            )
            query = tuple(
                sorted(
                    index
                    for group in query_groups
                    for index in groups[group]
                )
            )
            train = tuple(
                sorted(
                    index
                    for group in train_groups
                    for index in groups[group]
                )
            )
            if not train or not query:
                raise R10DynamicRepresentationError(
                    "appearance fold has an empty arm"
                )
            train_components = {
                examples[index].component_id for index in train
            }
            query_components = {
                examples[index].component_id for index in query
            }
            if train_components & query_components:
                raise RuntimeError("component crossed an R10 fold")
            fold = _Fold(
                fold_id=f"repeat_{repeat}_fold_{fold_index}",
                repeat=repeat,
                fold=fold_index,
                train_indices=train,
                query_indices=query,
                train_groups=train_groups,
                query_groups=query_groups,
            )
            output.append(fold)
            label_counts = Counter(
                examples[index].label_class for index in query
            )
            family_counts = Counter(
                examples[index].family
                for index in query
                if examples[index].label_class == "positive"
            )
            assignment_commitment = {
                "query_group_ids_sha256": _object_digest(
                    list(query_groups)
                ),
                "query_iids_sha256": _object_digest(
                    sorted(examples[index].iid for index in query)
                ),
                "query_component_ids_sha256": _object_digest(
                    sorted(
                        {
                            examples[index].component_id
                            for index in query
                        }
                    )
                ),
            }
            records.append(
                {
                    "schema_version": FOLD_SCHEMA,
                    "fold_id": fold.fold_id,
                    "seed": seed,
                    "repeat": repeat,
                    "fold": fold_index,
                    "train_rows": len(train),
                    "query_rows": len(query),
                    "train_groups": len(train_groups),
                    "query_groups": len(query_groups),
                    "forced_train_groups": len(forced_groups),
                    "query_label_counts": dict(sorted(label_counts.items())),
                    "query_positive_family_counts": dict(
                        sorted(family_counts.items())
                    ),
                    "train_query_group_disjoint": True,
                    "train_query_component_disjoint": True,
                    "legacy_test_excluded": True,
                    "group_assignment":
                        "deterministic_label_stratified_greedy_v1",
                    "assignment_commitment": assignment_commitment,
                }
            )
    return output, records


def _stratified_group_assignment(
    examples: Sequence[_R10Example],
    *,
    groups: Mapping[str, Sequence[int]],
    movable_groups: Sequence[str],
    folds: int,
    seed: int,
    namespace: str,
) -> dict[str, int]:
    """Balance label/family support while keeping whole appearance groups.

    Stratification uses only pseudo-labels belonging to the pool being
    partitioned.  It never supplies those labels or assignments to the
    representation itself.
    """

    if folds < 2 or len(movable_groups) < folds:
        raise R10DynamicRepresentationError(
            "group stratification has insufficient support"
        )

    def category(index: int) -> str:
        example = examples[index]
        return (
            f"positive:{example.family}"
            if example.label_class == "positive"
            else "negative"
        )

    group_category_counts: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    total_rows = 0
    for group in movable_groups:
        counts = Counter(category(index) for index in groups[group])
        group_category_counts[group] = counts
        totals.update(counts)
        total_rows += len(groups[group])
    categories = tuple(sorted(totals))
    targets = {
        name: totals[name] / folds for name in categories
    }
    target_rows = total_rows / folds
    target_groups = len(movable_groups) / folds

    def priority(group: str) -> tuple[Any, ...]:
        counts = group_category_counts[group]
        rarity = max(
            (
                counts[name] / max(totals[name], 1)
                for name in categories
            ),
            default=0.0,
        )
        return (
            -rarity,
            -len(groups[group]),
            _stable_u32(namespace, seed, "group-order", group),
            group,
        )

    fold_counts = [Counter() for _ in range(folds)]
    fold_rows = [0 for _ in range(folds)]
    fold_groups = [0 for _ in range(folds)]
    assignment: dict[str, int] = {}
    for group in sorted(movable_groups, key=priority):
        counts = group_category_counts[group]

        def score(candidate: int) -> tuple[Any, ...]:
            category_error = 0.0
            for fold_index in range(folds):
                for name in categories:
                    value = fold_counts[fold_index][name]
                    if fold_index == candidate:
                        value += counts[name]
                    category_error += (
                        (value - targets[name]) ** 2
                        / max(targets[name], 1.0)
                    )
            row_error = sum(
                (
                    (
                        fold_rows[fold_index]
                        + (
                            len(groups[group])
                            if fold_index == candidate
                            else 0
                        )
                        - target_rows
                    )
                    ** 2
                )
                / max(target_rows, 1.0)
                for fold_index in range(folds)
            )
            group_error = sum(
                (
                    (
                        fold_groups[fold_index]
                        + (1 if fold_index == candidate else 0)
                        - target_groups
                    )
                    ** 2
                )
                / max(target_groups, 1.0)
                for fold_index in range(folds)
            )
            return (
                category_error,
                0.25 * row_error,
                0.05 * group_error,
                fold_groups[candidate],
                fold_rows[candidate],
                _stable_u32(
                    namespace,
                    seed,
                    "fold-tie",
                    group,
                    candidate,
                ),
                candidate,
            )

        selected = min(range(folds), key=score)
        assignment[group] = selected
        fold_counts[selected].update(counts)
        fold_rows[selected] += len(groups[group])
        fold_groups[selected] += 1
    if any(value == 0 for value in fold_groups):
        raise R10DynamicRepresentationError(
            "stratified assignment produced an empty fold"
        )
    return assignment


def _make_inner_folds(
    examples: Sequence[_R10Example],
    *,
    index_to_group: Mapping[int, str],
    outer_fold: _Fold,
    seed: int,
    folds: int = 2,
) -> tuple[list[_Fold], list[dict[str, Any]]]:
    """Create true inner folds using only one outer fold's training arm."""

    groups: dict[str, list[int]] = defaultdict(list)
    for index in outer_fold.train_indices:
        groups[str(index_to_group[index])].append(index)
    forced_groups = {
        group
        for group, indices in groups.items()
        if any(not examples[index].fresh for index in indices)
    }
    movable_groups = tuple(
        group for group in sorted(groups) if group not in forced_groups
    )
    if len(movable_groups) < folds:
        return [], []
    assignment = _stratified_group_assignment(
        examples,
        groups=groups,
        movable_groups=movable_groups,
        folds=folds,
        seed=seed,
        namespace=f"r10-inner-{outer_fold.fold_id}",
    )
    assignment.update({group: None for group in forced_groups})
    output: list[_Fold] = []
    records: list[dict[str, Any]] = []
    outer_query = set(outer_fold.query_indices)
    outer_train = set(outer_fold.train_indices)
    for fold_index in range(folds):
        query_groups = tuple(
            group
            for group in sorted(groups)
            if assignment[group] == fold_index
        )
        train_groups = tuple(
            group
            for group in sorted(groups)
            if group not in set(query_groups)
        )
        query = tuple(
            sorted(
                index
                for group in query_groups
                for index in groups[group]
            )
        )
        train = tuple(
            sorted(
                index
                for group in train_groups
                for index in groups[group]
            )
        )
        if (
            not query
            or not train
            or not set(query).isdisjoint(train)
            or not set(query).issubset(outer_train)
            or not set(train).issubset(outer_train)
            or outer_query & (set(query) | set(train))
        ):
            raise R10DynamicRepresentationError(
                f"{outer_fold.fold_id} inner split violated outer isolation"
            )
        fold = _Fold(
            fold_id=(
                f"{outer_fold.fold_id}_nested_inner_{fold_index}"
            ),
            repeat=outer_fold.repeat,
            fold=fold_index,
            train_indices=train,
            query_indices=query,
            train_groups=train_groups,
            query_groups=query_groups,
        )
        output.append(fold)
        records.append(
            {
                "inner_fold_id": fold.fold_id,
                "train_rows": len(train),
                "query_rows": len(query),
                "train_groups": len(train_groups),
                "query_groups": len(query_groups),
                "outer_query_rows_seen": 0,
                "outer_train_only": True,
            }
        )
    return output, records


def _balanced_bank(
    examples: Sequence[_R10Example],
    train_indices: Sequence[int],
    eligible_families: set[str],
    *,
    seed: int,
    fold_id: str,
) -> tuple[int, ...]:
    by_family_component: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in train_indices:
        example = examples[index]
        if (
            example.label_class == "positive"
            and example.family in eligible_families
        ):
            by_family_component[example.family][
                example.component_id
            ].append(index)
    if not by_family_component:
        return ()
    cap = min(
        12,
        min(len(components) for components in by_family_component.values()),
    )
    if cap < r7.MINIMUM_TRAIN_COMPONENTS:
        return ()
    bank: list[int] = []
    for family in sorted(by_family_component):
        representatives: list[tuple[str, int]] = []
        for component, indices in by_family_component[family].items():
            representative = min(
                indices,
                key=lambda index: (
                    _stable_u32(
                        "r10-bank-row",
                        seed,
                        fold_id,
                        family,
                        examples[index].iid,
                    ),
                    examples[index].iid,
                ),
            )
            representatives.append((component, representative))
        representatives.sort(
            key=lambda item: (
                _stable_u32(
                    "r10-bank-component",
                    seed,
                    fold_id,
                    family,
                    item[0],
                ),
                item[0],
            )
        )
        bank.extend(index for _component, index in representatives[:cap])
    return tuple(sorted(bank, key=lambda index: examples[index].iid))


def _retrieval_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["eligible_positive_query"] is True
    ]
    per_family: dict[str, Any] = {}
    for family in sorted({str(row["family"]) for row in selected}):
        family_rows = [row for row in selected if row["family"] == family]
        per_family[family] = {
            "queries": len(family_rows),
            "valid_queries": sum(
                row["arms"][arm]["valid"] is True for row in family_rows
            ),
            "r_at_1": float(
                np.mean(
                    [
                        row["arms"][arm]["correct_at_1"] is True
                        for row in family_rows
                    ]
                )
            ),
            "r_at_5": float(
                np.mean(
                    [
                        row["arms"][arm]["correct_at_5"] is True
                        for row in family_rows
                    ]
                )
            ),
        }
    queries = len(selected)
    valid = sum(row["arms"][arm]["valid"] is True for row in selected)
    return {
        "queries": queries,
        "valid_queries": valid,
        "valid_fraction": float(valid / queries) if queries else None,
        "micro_r_at_1": (
            float(
                np.mean(
                    [
                        row["arms"][arm]["correct_at_1"] is True
                        for row in selected
                    ]
                )
            )
            if selected
            else None
        ),
        "micro_r_at_5": (
            float(
                np.mean(
                    [
                        row["arms"][arm]["correct_at_5"] is True
                        for row in selected
                    ]
                )
            )
            if selected
            else None
        ),
        "macro_family_r_at_1": (
            float(np.mean([value["r_at_1"] for value in per_family.values()]))
            if per_family
            else None
        ),
        "macro_family_r_at_5": (
            float(np.mean([value["r_at_5"] for value in per_family.values()]))
            if per_family
            else None
        ),
        "per_family": per_family,
    }


def _weighted_auc_with_invalid_floor(
    examples: Sequence[_R10Example],
    query_indices: Sequence[int],
    scores: Mapping[int, float | None],
) -> dict[str, Any]:
    positive_indices = [
        index
        for index in query_indices
        if examples[index].label_class == "positive"
    ]
    negative_indices = [
        index
        for index in query_indices
        if examples[index].label_class == "negative"
    ]
    valid_positive = [
        index for index in positive_indices if scores.get(index) is not None
    ]
    valid_negative = [
        index for index in negative_indices if scores.get(index) is not None
    ]
    valid_auc = r7._weighted_auc(
        [float(scores[index]) for index in valid_positive],
        [float(scores[index]) for index in valid_negative],
        [examples[index].sampling_weight for index in valid_positive],
        [examples[index].sampling_weight for index in valid_negative],
    )
    conservative_auc = r7._weighted_auc(
        [
            float(scores[index])
            if scores.get(index) is not None
            else -1.0
            for index in positive_indices
        ],
        [
            float(scores[index])
            if scores.get(index) is not None
            else 1.0
            for index in negative_indices
        ],
        [examples[index].sampling_weight for index in positive_indices],
        [examples[index].sampling_weight for index in negative_indices],
    )
    return {
        "positive_rows": len(positive_indices),
        "negative_rows": len(negative_indices),
        "valid_positive_rows": len(valid_positive),
        "valid_negative_rows": len(valid_negative),
        "positive_valid_fraction": (
            float(len(valid_positive) / len(positive_indices))
            if positive_indices
            else None
        ),
        "negative_valid_fraction": (
            float(len(valid_negative) / len(negative_indices))
            if negative_indices
            else None
        ),
        "valid_only_sampling_weighted_auroc": valid_auc,
        "conservative_sampling_weighted_auroc": conservative_auc,
        "invalid_positive_score": -1.0,
        "invalid_negative_score": 1.0,
    }


def _similarity_correlation(
    candidate: np.ndarray,
    appearance: np.ndarray,
    query_indices: Sequence[int],
    bank_indices: Sequence[int],
) -> float | None:
    if not query_indices or not bank_indices:
        return None
    query = np.asarray(tuple(query_indices), dtype=np.int64)
    bank = np.asarray(tuple(bank_indices), dtype=np.int64)
    left = (candidate[query] @ candidate[bank].T).reshape(-1)
    right = (appearance[query] @ appearance[bank].T).reshape(-1)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return None
    return float(np.dot(left, right) / denominator)


def _evaluate_embeddings(
    examples: Sequence[_R10Example],
    embeddings: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    train_indices: Sequence[int],
    query_indices: Sequence[int],
    eligible_families: set[str],
    seed: int,
    fold_id: str,
) -> dict[str, Any]:
    bank_indices = _balanced_bank(
        examples,
        train_indices,
        eligible_families,
        seed=seed,
        fold_id=fold_id,
    )
    if not bank_indices:
        raise R10DynamicRepresentationError(
            f"{fold_id} has no balanced reference bank"
        )
    appearance_raw = np.zeros(
        (len(examples), examples[0].pooled_target_dino.shape[0]),
        dtype=np.float64,
    )
    for index in set(query_indices) | set(bank_indices):
        appearance_raw[index] = examples[index].pooled_target_dino
    appearance, _appearance_valid = _unit_rows(appearance_raw)
    rows: list[dict[str, Any]] = []
    clean_scores: dict[int, float | None] = {}
    pair_positive: list[float] = []
    pair_negative: list[float] = []
    for query_index in query_indices:
        query = examples[query_index]
        arms: dict[str, Any] = {}
        for arm in CONTROLS:
            query_matrix, query_valid = embeddings[arm]
            bank_matrix, bank_valid = embeddings[CONTROL_CLEAN]
            candidates: list[tuple[int, float]] = []
            if bool(query_valid[query_index]):
                for bank_index in bank_indices:
                    reference = examples[bank_index]
                    if (
                        not bool(bank_valid[bank_index])
                        or reference.component_id == query.component_id
                        or reference.iid == query.iid
                    ):
                        continue
                    candidates.append(
                        (
                            bank_index,
                            float(
                                np.dot(
                                    query_matrix[query_index],
                                    bank_matrix[bank_index],
                                )
                            ),
                        )
                    )
            candidates.sort(
                key=lambda item: (
                    -item[1],
                    examples[item[0]].iid,
                )
            )
            independent: list[tuple[int, float]] = []
            seen_components: set[str] = set()
            for bank_index, score in candidates:
                component = examples[bank_index].component_id
                if component in seen_components:
                    continue
                seen_components.add(component)
                independent.append((bank_index, score))
                if len(independent) == 5:
                    break
            is_valid = len(independent) == 5
            eligible = (
                query.label_class == "positive"
                and query.family in eligible_families
            )
            arms[arm] = {
                "valid": is_valid,
                "invalid_reason": (
                    None if is_valid else "fewer_than_five_valid_references"
                ),
                "correct_at_1": (
                    examples[independent[0][0]].family == query.family
                    if eligible and is_valid
                    else None
                ),
                "correct_at_5": (
                    any(
                        examples[index].family == query.family
                        for index, _score in independent
                    )
                    if eligible and is_valid
                    else None
                ),
                "top_reference_iids": [
                    examples[index].iid for index, _score in independent
                ],
                "top_reference_families": [
                    examples[index].family for index, _score in independent
                ],
                "similarities": [score for _index, score in independent],
            }
            if arm == CONTROL_CLEAN:
                clean_scores[query_index] = (
                    independent[0][1] if is_valid else None
                )
                if eligible and is_valid:
                    all_scores = [
                        (
                            examples[index].family == query.family,
                            score,
                        )
                        for index, score in candidates
                    ]
                    pair_positive.extend(
                        score for same, score in all_scores if same
                    )
                    pair_negative.extend(
                        score for same, score in all_scores if not same
                    )
        rows.append(
            {
                "iid": query.iid,
                "family": query.family,
                "label_class": query.label_class,
                "component_id": query.component_id,
                "eligible_positive_query": (
                    query.label_class == "positive"
                    and query.family in eligible_families
                ),
                "sampling_weight": float(query.sampling_weight),
                "arms": arms,
            }
        )
    clean = _retrieval_summary(rows, arm=CONTROL_CLEAN)
    shuffle = _retrieval_summary(rows, arm=CONTROL_SHUFFLE)
    reverse = _retrieval_summary(rows, arm=CONTROL_REVERSE)
    clean_r5 = clean["macro_family_r_at_5"]
    shuffle_r5 = shuffle["macro_family_r_at_5"]
    reverse_r5 = reverse["macro_family_r_at_5"]
    binary = _weighted_auc_with_invalid_floor(
        examples,
        query_indices,
        clean_scores,
    )
    pair_auc = r7._weighted_auc(
        pair_positive,
        pair_negative,
        np.ones(len(pair_positive), dtype=np.float64),
        np.ones(len(pair_negative), dtype=np.float64),
    )
    correlation = _similarity_correlation(
        embeddings[CONTROL_CLEAN][0],
        appearance,
        query_indices,
        bank_indices,
    )
    return {
        "fold_id": fold_id,
        "balanced_reference_bank": {
            "rows": len(bank_indices),
            "families": len(
                {examples[index].family for index in bank_indices}
            ),
            "components": len(
                {examples[index].component_id for index in bank_indices}
            ),
            "family_balanced": True,
            "maximum_references_per_family": 12,
        },
        "retrieval": clean,
        "shuffle_control": shuffle,
        "reverse_control": reverse,
        "shuffle_margin_macro_r_at_5": (
            float(clean_r5 - shuffle_r5)
            if clean_r5 is not None and shuffle_r5 is not None
            else None
        ),
        "reverse_margin_macro_r_at_5": (
            float(clean_r5 - reverse_r5)
            if clean_r5 is not None and reverse_r5 is not None
            else None
        ),
        "positive_vs_negative": binary,
        "same_action_vs_different_action_pair_auroc": pair_auc,
        "candidate_vs_pooled_target_dino_similarity_correlation":
            correlation,
        "query_rows": len(query_indices),
        "eligible_families": sorted(eligible_families),
        "rows": rows,
    }


def _pooled_embeddings(
    examples: Sequence[_R10Example],
    *,
    side: str,
    active_indices: Sequence[int] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    if side not in {"source", "target", "target_minus_source"}:
        raise R10DynamicRepresentationError("invalid pooled-DINO side")
    selected = (
        tuple(range(len(examples)))
        if active_indices is None
        else tuple(int(index) for index in active_indices)
    )
    dimension = examples[0].source_dino.shape[1]
    raw = np.zeros((len(examples), dimension), dtype=np.float64)
    for index in selected:
        if side == "source":
            value = np.mean(examples[index].source_dino, axis=0)
        elif side == "target":
            value = np.mean(examples[index].target_dino, axis=0)
        else:
            value = (
                np.mean(examples[index].target_dino, axis=0)
                - np.mean(examples[index].source_dino, axis=0)
            )
        raw[index] = value
    matrix, valid = _unit_rows(raw)
    embeddings = {arm: (matrix, valid) for arm in CONTROLS}
    raw_controls = {arm: raw.copy() for arm in CONTROLS}
    return embeddings, raw_controls


def _static_feature_embeddings(
    examples: Sequence[_R10Example],
    *,
    field: str,
    active_indices: Sequence[int] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    selected = (
        tuple(range(len(examples)))
        if active_indices is None
        else tuple(int(index) for index in active_indices)
    )
    first = np.asarray(getattr(examples[0], field), dtype=np.float64)
    raw = np.zeros((len(examples), first.size), dtype=np.float64)
    for index in selected:
        raw[index] = np.asarray(
            getattr(examples[index], field),
            dtype=np.float64,
        ).reshape(-1)
    matrix, valid = _unit_rows(raw)
    embeddings = {arm: (matrix, valid) for arm in CONTROLS}
    raw_controls = {arm: raw.copy() for arm in CONTROLS}
    return embeddings, raw_controls


def _orderless_dino_embeddings(
    examples: Sequence[_R10Example],
    *,
    active_indices: Sequence[int] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    selected = (
        tuple(range(len(examples)))
        if active_indices is None
        else tuple(int(index) for index in active_indices)
    )
    dimension = 2 * examples[0].source_dino.shape[1]
    raw = np.zeros((len(examples), dimension), dtype=np.float64)
    for index in selected:
        example = examples[index]
        velocity = np.diff(example.target_dino, axis=0)
        raw[index] = np.concatenate(
            (
                np.mean(velocity, axis=0),
                np.std(velocity, axis=0),
            )
        )
    matrix, valid = _unit_rows(raw)
    embeddings = {arm: (matrix, valid) for arm in CONTROLS}
    raw_controls = {arm: raw.copy() for arm in CONTROLS}
    return embeddings, raw_controls


def _track_acceleration_embeddings(
    examples: Sequence[_R10Example],
    *,
    seed: int,
    active_indices: Sequence[int] | None = None,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
]:
    selected = (
        tuple(range(len(examples)))
        if active_indices is None
        else tuple(int(index) for index in active_indices)
    )
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_controls: dict[str, np.ndarray] = {}
    for control in CONTROLS:
        dimension = (len(examples[0].target_track) - 1) * 15
        raw = np.zeros((len(examples), dimension), dtype=np.float64)
        for index in selected:
            example = examples[index]
            _source, target, _source_dino, _target_dino = (
                _controlled_sequences(example, control=control, seed=seed)
            )
            raw[index] = np.diff(target, axis=0).reshape(-1)
        raw_controls[control] = raw
        output[control] = _unit_rows(raw)
    return output, raw_controls


def _matched_ridge_action_embeddings(
    examples: Sequence[_R10Example],
    raw: Mapping[str, np.ndarray],
    *,
    train_indices: Sequence[int],
    eligible_families: set[str],
    seed: int,
    name: str,
    ridge: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply the same supervised head class used by learned R10 candidates."""

    clean = np.asarray(raw[CONTROL_CLEAN], dtype=np.float64)
    spec = {
        "spec_digest": _object_digest(
            {
                "matched_control": name,
                "ridge": float(ridge),
            }
        ),
        "standardize": True,
        "content_residual": False,
        "projection": "jl",
        "projection_dim": 128,
        "head": "ridge_action",
        "ridge": float(ridge),
        "geometry_keep": 0,
        "raw_blocks": [name],
    }
    appearance = np.zeros((len(examples), 1), dtype=np.float64)
    fitted = _fit_transform(
        examples,
        spec,
        train_indices=train_indices,
        raw_clean=clean,
        appearance=appearance,
        eligible_families=eligible_families,
        seed=seed,
    )
    return {
        control: _encode_transform(value, appearance, fitted)
        for control, value in raw.items()
    }


def _energy_auc(
    examples: Sequence[_R10Example],
    query_indices: Sequence[int],
) -> dict[str, float | None]:
    positive = [
        index
        for index in query_indices
        if examples[index].label_class == "positive"
    ]
    negative = [
        index
        for index in query_indices
        if examples[index].label_class == "negative"
    ]

    def auc(values: Sequence[float]) -> float | None:
        return r7._weighted_auc(
            [values[index] for index in positive],
            [values[index] for index in negative],
            [examples[index].sampling_weight for index in positive],
            [examples[index].sampling_weight for index in negative],
        )

    # Derive energies only for the active query arm.  During development
    # search this prevents the already-consumed legacy test from being
    # touched before the candidate spec is frozen.
    track = {
        index: examples[index].motion_energy for index in query_indices
    }
    dino = {
        index: float(
            np.mean(
                np.linalg.norm(
                    np.diff(examples[index].target_dino, axis=0),
                    axis=1,
                )
            )
        )
        for index in query_indices
    }
    return {
        "track_motion_energy_auroc": auc(track),
        "dino_path_energy_auroc": auc(dino),
    }


def _evaluate_baselines(
    examples: Sequence[_R10Example],
    *,
    train_indices: Sequence[int],
    query_indices: Sequence[int],
    eligible_families: set[str],
    seed: int,
    fold_id: str,
    active_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    source_pooled, source_pooled_raw = _pooled_embeddings(
        examples,
        side="source",
        active_indices=active_indices,
    )
    target_pooled, target_pooled_raw = _pooled_embeddings(
        examples,
        side="target",
        active_indices=active_indices,
    )
    appearance_delta, appearance_delta_raw = _pooled_embeddings(
        examples,
        side="target_minus_source",
        active_indices=active_indices,
    )
    orderless, orderless_raw = _orderless_dino_embeddings(
        examples,
        active_indices=active_indices,
    )
    acceleration, acceleration_raw = _track_acceleration_embeddings(
        examples,
        seed=seed,
        active_indices=active_indices,
    )
    endpoint, endpoint_raw = _static_feature_embeddings(
        examples,
        field="target_endpoint",
        active_indices=active_indices,
    )
    track_orderless, track_orderless_raw = _static_feature_embeddings(
        examples,
        field="target_orderless",
        active_indices=active_indices,
    )
    camera, camera_raw = _static_feature_embeddings(
        examples,
        field="camera_nuisance",
        active_indices=active_indices,
    )
    embeddings: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "source_pooled_dino": source_pooled,
        "target_pooled_dino": target_pooled,
        "target_minus_source_pooled_dino": appearance_delta,
        "target_orderless_dino_dynamics": orderless,
        "r9_track_acceleration": acceleration,
        "track_endpoint": endpoint,
        "track_orderless": track_orderless,
        "camera_nuisance": camera,
    }
    raw_controls = {
        "source_pooled_dino": source_pooled_raw,
        "target_pooled_dino": target_pooled_raw,
        "target_minus_source_pooled_dino": appearance_delta_raw,
        "target_orderless_dino_dynamics": orderless_raw,
        "r9_track_acceleration": acceleration_raw,
        "track_endpoint": endpoint_raw,
        "track_orderless": track_orderless_raw,
        "camera_nuisance": camera_raw,
    }
    for control_name, raw in raw_controls.items():
        for ridge in (0.1, 1.0):
            name = f"{control_name}_matched_ridge_{ridge:g}"
            embeddings[name] = _matched_ridge_action_embeddings(
                examples,
                raw,
                train_indices=train_indices,
                eligible_families=eligible_families,
                seed=seed,
                name=control_name,
                ridge=ridge,
            )
    metrics = {
        name: _evaluate_embeddings(
            examples,
            value,
            train_indices=train_indices,
            query_indices=query_indices,
            eligible_families=eligible_families,
            seed=seed,
            # All candidates and controls use the exact same balanced bank.
            fold_id=fold_id,
        )
        for name, value in embeddings.items()
    }
    return metrics, _energy_auc(examples, query_indices)


def _baseline_r5(baselines: Mapping[str, Mapping[str, Any]]) -> float | None:
    values = [
        value["retrieval"]["macro_family_r_at_5"]
        for value in baselines.values()
        if value["retrieval"]["macro_family_r_at_5"] is not None
    ]
    return max(values) if values else None


def _fold_gate_failures(
    metrics: Mapping[str, Any],
    *,
    baselines: Mapping[str, Mapping[str, Any]],
    energies: Mapping[str, float | None],
) -> list[str]:
    failures: list[str] = []
    retrieval = metrics["retrieval"]
    if (
        retrieval["valid_fraction"] is None
        or retrieval["valid_fraction"] < MIN_QUERY_COVERAGE
    ):
        failures.append("representation_zero_or_unstable")
    binary = metrics["positive_vs_negative"]
    for field in ("positive_valid_fraction", "negative_valid_fraction"):
        value = binary[field]
        if value is None or value < MIN_QUERY_COVERAGE:
            failures.append("binary_query_coverage_failed")
    if (
        len(metrics["eligible_families"]) < MIN_STABLE_FAMILIES
        or retrieval["queries"] < 2
    ):
        failures.append("family_support_insufficient")
    candidate_r5 = retrieval["macro_family_r_at_5"]
    strongest = _baseline_r5(baselines)
    if (
        candidate_r5 is None
        or strongest is None
        or candidate_r5 < strongest + MIN_CONTROL_MARGIN
    ):
        failures.append("content_generalization_failed")
    shuffle_margin = metrics["shuffle_margin_macro_r_at_5"]
    if shuffle_margin is None or shuffle_margin < MIN_TEMPORAL_MARGIN:
        failures.append("temporal_order_insensitive")
    auc = binary["conservative_sampling_weighted_auroc"]
    energy_values = [
        value for value in energies.values() if value is not None
    ]
    required_auc = MIN_AUROC
    if energy_values:
        required_auc = max(
            required_auc,
            max(energy_values) + MIN_AUROC_OVER_ENERGY,
        )
    if auc is None or auc < required_auc:
        failures.append("static_or_low_motion_coverage")
    pair_auc = metrics["same_action_vs_different_action_pair_auroc"]
    if pair_auc is None or pair_auc < 0.55:
        failures.append("cross_content_pair_separation_failed")
    correlation = metrics[
        "candidate_vs_pooled_target_dino_similarity_correlation"
    ]
    if (
        correlation is not None
        and abs(float(correlation))
        > MAX_APPEARANCE_SIMILARITY_CORRELATION
    ):
        failures.append("appearance_leakage")
    return sorted(set(failures))


def _fold_objective(
    metrics: Mapping[str, Any],
    *,
    baselines: Mapping[str, Mapping[str, Any]],
) -> float:
    retrieval = metrics["retrieval"]
    values = (
        retrieval["macro_family_r_at_5"],
        retrieval["macro_family_r_at_1"],
        metrics["shuffle_margin_macro_r_at_5"],
        metrics["same_action_vs_different_action_pair_auroc"],
        metrics["positive_vs_negative"][
            "conservative_sampling_weighted_auroc"
        ],
    )
    if any(value is None for value in values):
        return -1e9
    strongest = _baseline_r5(baselines)
    control_margin = (
        float(values[0] - strongest)
        if strongest is not None
        else -1.0
    )
    correlation = metrics[
        "candidate_vs_pooled_target_dino_similarity_correlation"
    ]
    leakage_penalty = (
        max(0.0, abs(float(correlation)) - 0.5)
        if correlation is not None
        else 0.0
    )
    return float(
        values[0]
        + 0.35 * values[1]
        + 0.25 * max(0.0, values[2])
        + 0.20 * values[3]
        + 0.15 * values[4]
        + 0.50 * control_margin
        - 0.20 * leakage_penalty
    )


def _number_summary(values: Sequence[float | None]) -> dict[str, Any]:
    selected = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if not len(selected):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "q10": None,
            "maximum": None,
        }
    return {
        "count": len(selected),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "q10": float(np.quantile(selected, 0.10)),
        "maximum": float(np.max(selected)),
    }


def _compact_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "rows"}


def _headline_evaluation(value: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = value["retrieval"]
    binary = value["positive_vs_negative"]
    return {
        "macro_family_r_at_1": retrieval["macro_family_r_at_1"],
        "macro_family_r_at_5": retrieval["macro_family_r_at_5"],
        "valid_fraction": retrieval["valid_fraction"],
        "conservative_positive_negative_auroc":
            binary["conservative_sampling_weighted_auroc"],
        "same_action_vs_different_action_pair_auroc":
            value["same_action_vs_different_action_pair_auroc"],
    }


def _aggregate_fold_evaluations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    passed = sum(not record["failure_codes"] for record in records)
    total = len(records)
    return {
        "folds": total,
        "gate_passed_folds": passed,
        "gate_pass_fraction": float(passed / total) if total else 0.0,
        "macro_family_r_at_1": _number_summary(
            [
                record["metrics"]["retrieval"]["macro_family_r_at_1"]
                for record in records
            ]
        ),
        "macro_family_r_at_5": _number_summary(
            [
                record["metrics"]["retrieval"]["macro_family_r_at_5"]
                for record in records
            ]
        ),
        "control_margin_macro_r_at_5": _number_summary(
            [
                (
                    record["metrics"]["retrieval"][
                        "macro_family_r_at_5"
                    ]
                    - record["strongest_control_macro_r_at_5"]
                )
                if (
                    record["metrics"]["retrieval"][
                        "macro_family_r_at_5"
                    ]
                    is not None
                    and record["strongest_control_macro_r_at_5"]
                    is not None
                )
                else None
                for record in records
            ]
        ),
        "shuffle_margin_macro_r_at_5": _number_summary(
            [
                record["metrics"]["shuffle_margin_macro_r_at_5"]
                for record in records
            ]
        ),
        "reverse_margin_macro_r_at_5_diagnostic": _number_summary(
            [
                record["metrics"]["reverse_margin_macro_r_at_5"]
                for record in records
            ]
        ),
        "conservative_positive_negative_auroc": _number_summary(
            [
                record["metrics"]["positive_vs_negative"][
                    "conservative_sampling_weighted_auroc"
                ]
                for record in records
            ]
        ),
        "same_action_different_action_pair_auroc": _number_summary(
            [
                record["metrics"][
                    "same_action_vs_different_action_pair_auroc"
                ]
                for record in records
            ]
        ),
        "appearance_similarity_correlation": _number_summary(
            [
                record["metrics"][
                    "candidate_vs_pooled_target_dino_similarity_correlation"
                ]
                for record in records
            ]
        ),
        "objective": _number_summary(
            [record["objective"] for record in records]
        ),
        "failure_code_counts": dict(
            sorted(
                Counter(
                    code
                    for record in records
                    for code in record["failure_codes"]
                ).items()
            )
        ),
    }


def _spec_complexity(spec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(bool(spec["content_residual"])),
        int(spec["head"] != "identity"),
        len(spec["raw_blocks"]),
        int(spec["projection_dim"]),
    )


def _fit_and_evaluate_spec(
    examples: Sequence[_R10Example],
    spec: Mapping[str, Any],
    *,
    fold: _Fold,
    raw_block_cache: Mapping[str, Mapping[str, np.ndarray]],
    appearance: np.ndarray,
    eligible_families: set[str],
    baselines: Mapping[str, Mapping[str, Any]],
    energies: Mapping[str, float | None],
    seed: int,
) -> tuple[dict[str, Any], _FittedTransform]:
    raw = {
        control: _combine_precomputed_raw(
            spec,
            control=control,
            block_cache=raw_block_cache,
        )
        for control in CONTROLS
    }
    fitted = _fit_transform(
        examples,
        spec,
        train_indices=fold.train_indices,
        raw_clean=raw[CONTROL_CLEAN],
        appearance=appearance,
        eligible_families=eligible_families,
        seed=seed,
    )
    embeddings = {
        control: _encode_transform(value, appearance, fitted)
        for control, value in raw.items()
    }
    metrics = _evaluate_embeddings(
        examples,
        embeddings,
        train_indices=fold.train_indices,
        query_indices=fold.query_indices,
        eligible_families=eligible_families,
        seed=seed,
        fold_id=fold.fold_id,
    )
    failures = _fold_gate_failures(
        metrics,
        baselines=baselines,
        energies=energies,
    )
    objective = _fold_objective(metrics, baselines=baselines)
    return (
        {
            "fold_id": fold.fold_id,
            "metrics": metrics,
            "baselines": baselines,
            "energy_controls": dict(energies),
            "strongest_control_macro_r_at_5": _baseline_r5(baselines),
            "failure_codes": failures,
            "objective": objective,
        },
        fitted,
    )


def search_examples(
    examples: Sequence[_R10Example],
    *,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    folds: int = DEFAULT_FOLDS,
    maximum_trials: int | None = None,
) -> dict[str, Any]:
    """Run cross-fitted development selection, then legacy-test diagnosis."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise R10DynamicRepresentationError("invalid seed")
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, int)
        or repeats < 1
        or isinstance(folds, bool)
        or not isinstance(folds, int)
        or folds < 2
    ):
        raise R10DynamicRepresentationError("invalid fold configuration")
    if maximum_trials is not None and (
        isinstance(maximum_trials, bool)
        or not isinstance(maximum_trials, int)
        or maximum_trials < 1
    ):
        raise R10DynamicRepresentationError("invalid maximum_trials")
    if not examples:
        raise R10DynamicRepresentationError("R10 cohort is empty")
    development_indices = tuple(
        index
        for index, example in enumerate(examples)
        if example.original_split != "test"
    )
    legacy_test_indices = tuple(
        index
        for index, example in enumerate(examples)
        if example.original_split == "test"
    )

    index_to_group, grouping = _appearance_groups(
        examples,
        maximum_groups=DEFAULT_APPEARANCE_CLUSTERS,
    )
    fold_objects, fold_rows = _make_folds(
        examples,
        index_to_group=index_to_group,
        seed=seed,
        repeats=repeats,
        folds=folds,
    )
    contexts: dict[str, dict[str, Any]] = {}
    usable_folds: list[_Fold] = []
    fold_eligible_sets: list[set[str]] = []
    for fold in fold_objects:
        eligible, support = _eligible_families(
            examples,
            fold.train_indices,
        )
        query_labels = Counter(
            examples[index].label_class for index in fold.query_indices
        )
        eligible_queries = sum(
            examples[index].label_class == "positive"
            and examples[index].family in eligible
            for index in fold.query_indices
        )
        if (
            len(eligible) < r7.MINIMUM_ELIGIBLE_FAMILIES
            or not query_labels["positive"]
            or not query_labels["negative"]
            or eligible_queries < 2
        ):
            for record in fold_rows:
                if record["fold_id"] == fold.fold_id:
                    record["usable_for_search"] = False
                    record["exclusion_reason"] = (
                        "family_or_binary_query_support_insufficient"
                    )
                    record["fold_train_eligible_families"] = sorted(
                        eligible
                    )
            continue
        basis = _fit_dino_basis(
            examples,
            fold.train_indices,
            maximum_dimension=DEFAULT_DINO_CHANNEL_DIM,
        )
        appearance_basis = _fit_appearance_basis(
            examples,
            fold.train_indices,
        )
        appearance = _appearance_matrix(
            examples,
            appearance_basis,
            indices=development_indices,
        )
        raw_blocks = _precompute_raw_blocks(
            examples,
            seed=seed,
            dino_basis=basis,
            dino_dimension=DEFAULT_DINO_CHANNEL_DIM,
            indices=development_indices,
        )
        contexts[fold.fold_id] = {
            "fold_eligible": eligible,
            "support": support,
            "basis": basis,
            "appearance_basis": appearance_basis,
            "appearance": appearance,
            "raw_blocks": raw_blocks,
        }
        usable_folds.append(fold)
        fold_eligible_sets.append(set(eligible))
        for record in fold_rows:
            if record["fold_id"] == fold.fold_id:
                record["usable_for_search"] = True
                record["exclusion_reason"] = None
                record["fold_train_eligible_families"] = sorted(eligible)
    minimum_usable = max(2, int(math.ceil(0.67 * len(fold_objects))))
    if len(usable_folds) < minimum_usable:
        raise R10DynamicRepresentationError(
            "fewer than two-thirds repeated appearance folds are usable"
        )
    stable_eligible = set.intersection(*fold_eligible_sets)
    if len(stable_eligible) < r7.MINIMUM_ELIGIBLE_FAMILIES:
        raise R10DynamicRepresentationError(
            "fewer than two families are supported in every usable fold"
        )
    all_requested_folds_usable = len(usable_folds) == len(fold_objects)
    for fold in usable_folds:
        context = contexts[fold.fold_id]
        baselines, energies = _evaluate_baselines(
            examples,
            train_indices=fold.train_indices,
            query_indices=fold.query_indices,
            eligible_families=stable_eligible,
            seed=seed,
            fold_id=fold.fold_id,
            active_indices=development_indices,
        )
        context["eligible"] = stable_eligible
        context["baselines"] = baselines
        context["energies"] = energies
        for record in fold_rows:
            if record["fold_id"] == fold.fold_id:
                record["stable_cross_fold_eligible_families"] = sorted(
                    stable_eligible
                )

    specs = _candidate_specs()
    if maximum_trials is not None:
        specs = specs[:maximum_trials]
    trials: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    evaluation_cache: dict[str, list[dict[str, Any]]] = {}
    for trial_index, spec in enumerate(specs):
        started = time.monotonic()
        fold_evaluations: list[dict[str, Any]] = []
        for fold in usable_folds:
            context = contexts[fold.fold_id]
            evaluation, _fitted = _fit_and_evaluate_spec(
                examples,
                spec,
                fold=fold,
                raw_block_cache=context["raw_blocks"],
                appearance=context["appearance"],
                eligible_families=context["eligible"],
                baselines=context["baselines"],
                energies=context["energies"],
                seed=seed,
            )
            fold_evaluations.append(evaluation)
            for code in evaluation["failure_codes"]:
                failures.append(
                    {
                        "schema_version": FAILURE_SCHEMA,
                        "trial_index": trial_index,
                        "spec_digest": spec["spec_digest"],
                        "fold_id": fold.fold_id,
                        "failure_code": code,
                        "selection_stage": "repeated_group_cv",
                        "retry_same_spec_allowed": False,
                    }
                )
        aggregate = _aggregate_fold_evaluations(fold_evaluations)
        raw_development_signal_passed = (
            all_requested_folds_usable
            and len(stable_eligible) >= MIN_STABLE_FAMILIES
            and
            aggregate["gate_pass_fraction"]
            >= MIN_DEVELOPMENT_FOLD_PASS_FRACTION
        )
        development_passed = bool(
            spec["champion_eligible"]
            and raw_development_signal_passed
        )
        trials.append(
            {
                "schema_version": TRIAL_SCHEMA,
                "trial_index": trial_index,
                "seed": seed,
                "spec": dict(spec),
                "selection_stage": "repeated_appearance_group_cv",
                "legacy_test_metrics_read": False,
                "fold_evaluations": [
                    {
                        **{
                            key: value
                            for key, value in evaluation.items()
                            if key not in {"metrics", "baselines"}
                        },
                        "metrics": _compact_evaluation(
                            evaluation["metrics"]
                        ),
                        "baselines": {
                            name: _headline_evaluation(value)
                            for name, value in evaluation[
                                "baselines"
                            ].items()
                        },
                    }
                    for evaluation in fold_evaluations
                ],
                "aggregate": aggregate,
                "all_requested_folds_usable":
                    all_requested_folds_usable,
                "stable_cross_fold_family_count": len(stable_eligible),
                "raw_development_signal_passed":
                    raw_development_signal_passed,
                "development_candidate_passed": development_passed,
                "elapsed_seconds": float(time.monotonic() - started),
                "fresh_holdout_passed": False,
                "renderer_probe_authorized": False,
                "editor_training_authorized": False,
            }
        )
        evaluation_cache[str(spec["spec_digest"])] = fold_evaluations
    if not trials:
        raise R10DynamicRepresentationError("R10 evaluated no specs")

    def selection_key(trial: Mapping[str, Any]) -> tuple[Any, ...]:
        aggregate = trial["aggregate"]
        control_q10 = aggregate["control_margin_macro_r_at_5"]["q10"]
        shuffle_q10 = aggregate["shuffle_margin_macro_r_at_5"]["q10"]
        r1_median = aggregate["macro_family_r_at_1"]["median"]
        objective = aggregate["objective"]["median"]
        return (
            -int(trial["spec"]["champion_eligible"]),
            -int(trial["development_candidate_passed"]),
            -float(aggregate["gate_pass_fraction"]),
            -float(control_q10 if control_q10 is not None else -1e9),
            -float(shuffle_q10 if shuffle_q10 is not None else -1e9),
            -float(r1_median if r1_median is not None else -1e9),
            -float(objective if objective is not None else -1e9),
            _spec_complexity(trial["spec"]),
            str(trial["spec"]["spec_digest"]),
        )

    # True nested estimate: each inner transform, including its DINO and
    # appearance PCA plus any pseudo-label head, is fitted solely within one
    # outer training arm.  Reusing the other top-level folds here would leak
    # the current outer query into their training arms.
    usable_by_id = {fold.fold_id: fold for fold in usable_folds}
    trial_index_by_digest = {
        str(trial["spec"]["spec_digest"]): int(trial["trial_index"])
        for trial in trials
    }
    nested_outer_records: list[dict[str, Any]] = []
    for outer_fold in fold_objects:
        if outer_fold.fold_id not in usable_by_id:
            nested_outer_records.append(
                {
                    "outer_fold_id": outer_fold.fold_id,
                    "repeat": outer_fold.repeat,
                    "inner_fold_count": 0,
                    "inner_folds": [],
                    "outer_query_seen_by_inner_fit": False,
                    "outer_evaluable": False,
                    "selected_spec_digest": None,
                    "selected_spec_name": None,
                    "outer_failure_codes": [
                        "outer_fold_support_insufficient"
                    ],
                    "outer_gate_passed": False,
                    "outer_metrics": None,
                }
            )
            continue
        inner_folds, inner_records = _make_inner_folds(
            examples,
            index_to_group=index_to_group,
            outer_fold=outer_fold,
            seed=seed,
            folds=2,
        )
        inner_contexts: dict[str, dict[str, Any]] = {}
        inner_eligible_sets: list[set[str]] = []
        for inner_fold, inner_record in zip(inner_folds, inner_records):
            eligible, support = _eligible_families(
                examples,
                inner_fold.train_indices,
            )
            query_labels = Counter(
                examples[index].label_class
                for index in inner_fold.query_indices
            )
            eligible_queries = sum(
                examples[index].label_class == "positive"
                and examples[index].family in eligible
                for index in inner_fold.query_indices
            )
            usable = bool(
                len(eligible) >= r7.MINIMUM_ELIGIBLE_FAMILIES
                and query_labels["positive"]
                and query_labels["negative"]
                and eligible_queries >= 2
            )
            inner_record["usable_for_nested_selection"] = usable
            inner_record["eligible_families"] = sorted(eligible)
            inner_record["support"] = support
            if not usable:
                inner_record["exclusion_reason"] = (
                    "family_or_binary_query_support_insufficient"
                )
                continue
            basis = _fit_dino_basis(
                examples,
                inner_fold.train_indices,
                maximum_dimension=DEFAULT_DINO_CHANNEL_DIM,
            )
            appearance_basis = _fit_appearance_basis(
                examples,
                inner_fold.train_indices,
            )
            appearance = _appearance_matrix(
                examples,
                appearance_basis,
                indices=outer_fold.train_indices,
            )
            raw_blocks = _precompute_raw_blocks(
                examples,
                seed=seed,
                dino_basis=basis,
                dino_dimension=DEFAULT_DINO_CHANNEL_DIM,
                indices=outer_fold.train_indices,
            )
            inner_contexts[inner_fold.fold_id] = {
                "fold": inner_fold,
                "fold_eligible": eligible,
                "appearance": appearance,
                "raw_blocks": raw_blocks,
            }
            inner_eligible_sets.append(set(eligible))
            inner_record["exclusion_reason"] = None

        all_inner_folds_usable = bool(
            len(inner_folds) == 2
            and len(inner_contexts) == len(inner_folds)
        )
        stable_inner = (
            set.intersection(*inner_eligible_sets)
            if inner_eligible_sets
            else set()
        )
        if (
            not all_inner_folds_usable
            or len(stable_inner) < r7.MINIMUM_ELIGIBLE_FAMILIES
        ):
            nested_outer_records.append(
                {
                    "outer_fold_id": outer_fold.fold_id,
                    "repeat": outer_fold.repeat,
                    "inner_fold_count": len(inner_folds),
                    "inner_folds": inner_records,
                    "stable_inner_eligible_families": sorted(
                        stable_inner
                    ),
                    "outer_query_seen_by_inner_fit": False,
                    "outer_evaluable": False,
                    "selected_spec_digest": None,
                    "selected_spec_name": None,
                    "outer_failure_codes": [
                        "nested_inner_support_insufficient"
                    ],
                    "outer_gate_passed": False,
                    "outer_metrics": None,
                }
            )
            continue

        for context in inner_contexts.values():
            inner_fold = context["fold"]
            baselines, energies = _evaluate_baselines(
                examples,
                train_indices=inner_fold.train_indices,
                query_indices=inner_fold.query_indices,
                eligible_families=stable_inner,
                seed=seed,
                fold_id=inner_fold.fold_id,
                active_indices=outer_fold.train_indices,
            )
            context["eligible"] = stable_inner
            context["baselines"] = baselines
            context["energies"] = energies

        inner_candidates: list[dict[str, Any]] = []
        for spec in specs:
            if not bool(spec["champion_eligible"]):
                continue
            inner_evaluations: list[dict[str, Any]] = []
            for inner_fold in inner_folds:
                context = inner_contexts[inner_fold.fold_id]
                evaluation, _fitted = _fit_and_evaluate_spec(
                    examples,
                    spec,
                    fold=inner_fold,
                    raw_block_cache=context["raw_blocks"],
                    appearance=context["appearance"],
                    eligible_families=context["eligible"],
                    baselines=context["baselines"],
                    energies=context["energies"],
                    seed=seed,
                )
                inner_evaluations.append(evaluation)
                for code in evaluation["failure_codes"]:
                    failures.append(
                        {
                            "schema_version": FAILURE_SCHEMA,
                            "trial_index": trial_index_by_digest[
                                str(spec["spec_digest"])
                            ],
                            "spec_digest": spec["spec_digest"],
                            "fold_id": inner_fold.fold_id,
                            "failure_code": code,
                            "selection_stage": "true_nested_inner_cv",
                            "retry_same_spec_allowed": False,
                        }
                    )
            aggregate = _aggregate_fold_evaluations(inner_evaluations)
            inner_candidates.append(
                {
                    "spec": spec,
                    "aggregate": aggregate,
                    "development_candidate_passed": bool(
                        len(stable_inner) >= MIN_STABLE_FAMILIES
                        and aggregate["gate_pass_fraction"]
                        >= MIN_DEVELOPMENT_FOLD_PASS_FRACTION
                    ),
                }
            )
        if not inner_candidates:
            raise R10DynamicRepresentationError(
                f"{outer_fold.fold_id} has no champion-eligible inner spec"
            )
        selected = sorted(inner_candidates, key=selection_key)[0]
        selected_digest = str(selected["spec"]["spec_digest"])
        outer_context = contexts[outer_fold.fold_id]
        outer_eligible = set(outer_context["fold_eligible"])
        outer_baselines, outer_energies = _evaluate_baselines(
            examples,
            train_indices=outer_fold.train_indices,
            query_indices=outer_fold.query_indices,
            eligible_families=outer_eligible,
            seed=seed,
            fold_id=outer_fold.fold_id,
            active_indices=development_indices,
        )
        outer_evaluation, _outer_fitted = _fit_and_evaluate_spec(
            examples,
            selected["spec"],
            fold=outer_fold,
            raw_block_cache=outer_context["raw_blocks"],
            appearance=outer_context["appearance"],
            eligible_families=outer_eligible,
            baselines=outer_baselines,
            energies=outer_energies,
            seed=seed,
        )
        outer_failure_codes = list(outer_evaluation["failure_codes"])
        if not selected["development_candidate_passed"]:
            outer_failure_codes.append(
                "nested_inner_model_selection_gate_failed"
            )
        nested_outer_records.append(
            {
                "outer_fold_id": outer_fold.fold_id,
                "repeat": outer_fold.repeat,
                "inner_fold_count": len(inner_folds),
                "inner_folds": inner_records,
                "stable_inner_eligible_families": sorted(stable_inner),
                "outer_train_eligible_families": sorted(outer_eligible),
                "outer_query_seen_by_inner_fit": False,
                "outer_evaluable": True,
                "selected_spec_digest": selected_digest,
                "selected_spec_name": selected["spec"]["name"],
                "selected_inner_aggregate": selected["aggregate"],
                "selected_inner_gate_passed":
                    selected["development_candidate_passed"],
                "outer_failure_codes": sorted(set(outer_failure_codes)),
                "outer_gate_passed": not outer_failure_codes,
                "outer_metrics": _compact_evaluation(
                    outer_evaluation["metrics"]
                ),
            }
        )
    nested_outer_pass_fraction = float(
        np.mean(
            [
                record["outer_gate_passed"] is True
                for record in nested_outer_records
            ]
        )
    )
    all_nested_outer_folds_evaluable = all(
        record["outer_evaluable"] is True
        for record in nested_outer_records
    )

    champion_trials = [
        trial
        for trial in trials
        if trial["spec"]["champion_eligible"] is True
    ]
    if not champion_trials:
        raise R10DynamicRepresentationError(
            "R10 has no reusable-representation candidate"
        )
    champion_trial = sorted(champion_trials, key=selection_key)[0]
    frozen_spec = dict(champion_trial["spec"])
    champion_fold_evaluations = evaluation_cache[
        str(frozen_spec["spec_digest"])
    ]

    # The legacy test is touched for scoring only after the spec digest and
    # every hyperparameter have been frozen.  It is diagnostic, not a fresh
    # promotion holdout, because R9 already used it to motivate this family.
    final_eligible, final_support = _eligible_families(
        examples,
        development_indices,
    )
    if (
        len(final_eligible) < r7.MINIMUM_ELIGIBLE_FAMILIES
        or not legacy_test_indices
    ):
        raise R10DynamicRepresentationError(
            "final development/test support is insufficient"
        )
    final_basis = _fit_dino_basis(
        examples,
        development_indices,
        maximum_dimension=DEFAULT_DINO_CHANNEL_DIM,
    )
    final_appearance_basis = _fit_appearance_basis(
        examples,
        development_indices,
    )
    final_appearance = _appearance_matrix(
        examples,
        final_appearance_basis,
    )
    final_raw_blocks = _precompute_raw_blocks(
        examples,
        seed=seed,
        dino_basis=final_basis,
        dino_dimension=DEFAULT_DINO_CHANNEL_DIM,
    )
    final_raw = {
        control: _combine_precomputed_raw(
            frozen_spec,
            control=control,
            block_cache=final_raw_blocks,
        )
        for control in CONTROLS
    }
    final_transform = _fit_transform(
        examples,
        frozen_spec,
        train_indices=development_indices,
        raw_clean=final_raw[CONTROL_CLEAN],
        appearance=final_appearance,
        eligible_families=final_eligible,
        seed=seed,
    )
    final_embeddings = {
        control: _encode_transform(
            value,
            final_appearance,
            final_transform,
        )
        for control, value in final_raw.items()
    }
    final_baselines, final_energies = _evaluate_baselines(
        examples,
        train_indices=development_indices,
        query_indices=legacy_test_indices,
        eligible_families=final_eligible,
        seed=seed,
        fold_id="legacy_test_after_freeze",
    )
    legacy_test_metrics = _evaluate_embeddings(
        examples,
        final_embeddings,
        train_indices=development_indices,
        query_indices=legacy_test_indices,
        eligible_families=final_eligible,
        seed=seed,
        fold_id="legacy_test_after_freeze",
    )
    legacy_test_failures = _fold_gate_failures(
        legacy_test_metrics,
        baselines=final_baselines,
        energies=final_energies,
    )
    for code in legacy_test_failures:
        failures.append(
            {
                "schema_version": FAILURE_SCHEMA,
                "trial_index": champion_trial["trial_index"],
                "spec_digest": frozen_spec["spec_digest"],
                "fold_id": "legacy_test_after_freeze",
                "failure_code": code,
                "selection_stage": "post_freeze_legacy_test_diagnostic",
                "retry_same_spec_allowed": False,
            }
        )
    failures.append(
        {
            "schema_version": FAILURE_SCHEMA,
            "trial_index": champion_trial["trial_index"],
            "spec_digest": frozen_spec["spec_digest"],
            "fold_id": "promotion_gate",
            "failure_code": "fresh_holdout_absent",
            "selection_stage": "promotion_gate",
            "retry_same_spec_allowed": False,
        }
    )

    prediction_rows: list[dict[str, Any]] = []
    for evaluation in champion_fold_evaluations:
        for row in evaluation["metrics"]["rows"]:
            prediction_rows.append(
                {
                    "schema_version":
                        "motive-r10a-champion-prediction-v1",
                    "stage": "repeated_group_cv",
                    "fold_id": evaluation["fold_id"],
                    "spec_digest": frozen_spec["spec_digest"],
                    **row,
                }
            )
    for row in legacy_test_metrics["rows"]:
        prediction_rows.append(
            {
                "schema_version":
                    "motive-r10a-champion-prediction-v1",
                "stage": "post_freeze_legacy_test_diagnostic",
                "fold_id": "legacy_test_after_freeze",
                "spec_digest": frozen_spec["spec_digest"],
                **row,
            }
        )

    single_seed_development_signal_passed = bool(
        champion_trial["development_candidate_passed"]
        and all_nested_outer_folds_evaluable
        and nested_outer_pass_fraction
        >= MIN_DEVELOPMENT_FOLD_PASS_FRACTION
    )
    legacy_diagnostic_passed = not legacy_test_failures
    if (
        single_seed_development_signal_passed
        and legacy_diagnostic_passed
    ):
        status = (
            "SINGLE_SEED_SIGNAL_REQUIRES_CROSS_SEED_AND_FRESH_HOLDOUT"
        )
        next_experiment = (
            "combine both fold-assignment seeds, then freeze a new "
            "content-disjoint holdout before any renderer probe"
        )
    else:
        status = "CONTINUE_TO_FROZEN_MODEL_DELTA_TANGENT"
        next_experiment = (
            "R10B frozen-generator motion-weighted source-target "
            "delta-gradient representation"
        )
    summary = {
        "schema_version": SEARCH_SCHEMA,
        "status": "complete",
        "scope": (
            "R10A proxy ceiling: dynamic DINO + signed tracks + train-only "
            "closed-form representation learning; no generation or editor "
            "training"
        ),
        "motive_alignment_boundary": {
            "is_motive_generator_gradient_representation": False,
            "what_is_reused": [
                "motion-focused similarity",
                "fixed low-dimensional projection",
                "normalization",
                "multi-example action aggregation",
                "content and motion-magnitude controls",
            ],
            "what_is_missing": (
                "frozen generator motion-weighted parameter-gradient "
                "compatibility"
            ),
        },
        "seed": seed,
        "budget": {
            "requested_repeats": repeats,
            "requested_folds_per_repeat": folds,
            "realized_fold_rows": len(fold_objects),
            "usable_search_folds": len(usable_folds),
            "candidate_specs": len(specs),
            "champion_eligible_specs": sum(
                spec["champion_eligible"] is True for spec in specs
            ),
            "closed_set_supervised_upper_bound_specs": sum(
                spec["champion_eligible"] is False for spec in specs
            ),
        },
        "appearance_grouping": grouping,
        "fold_protocol": {
            "development_pool": "legacy train plus validation",
            "legacy_test_excluded_from_selection": True,
            "fresh_false_groups_forced_train": True,
            "seed_changes_group_fold_assignment": True,
            "seed_is_stability_perturbation_not_independent_replication":
                True,
            "seed_also_changes": [
                "JL projection",
                "shuffle control",
                "balanced-bank tie breaking",
            ],
            "development_fold_assignment_sha256":
                _fold_assignment_rows_digest(fold_rows),
            "assignment_digest_excludes_seed": True,
            "all_specs_share_exact_folds_and_balanced_banks": True,
            "train_query_appearance_group_disjoint": True,
            "train_query_component_disjoint": True,
            "test_derived_scores_computed_after_spec_freeze": True,
            "legacy_test_input_commit_validated_and_loaded_upfront": True,
            "legacy_test_descriptors_used_by_selection": False,
            "legacy_test_is_fresh_promotion_holdout": False,
            "reason_legacy_test_not_fresh": (
                "R9 test results already motivated the R10 family"
            ),
        },
        "selection_protocol": {
            "feasibility_before_objective": True,
            "true_nested_outer_model_selection_estimate": True,
            "inner_splits_are_subsets_of_outer_train_only": True,
            "outer_query_rows_seen_by_inner_fit": 0,
            "outer_evaluation_family_set_fit_from_outer_train_only":
                True,
            "minimum_development_fold_pass_fraction":
                MIN_DEVELOPMENT_FOLD_PASS_FRACTION,
            "train_only_dino_pca": True,
            "train_only_content_residualizer": True,
            "train_only_action_head": True,
            "balanced_reference_bank": True,
            "all_delta_temporal_controls_recomputed": True,
            "reverse_is_diagnostic_not_a_hard_gate": True,
            "closed_set_supervised_heads_are_upper_bounds_only": True,
            "closed_set_supervised_heads_champion_eligible": False,
            "matched_controls_use_unnormalized_raw_descriptors": True,
            "labels_are_pseudo": True,
        },
        "final_support": final_support,
        "nested_outer_model_selection": {
            "records": nested_outer_records,
            "gate_pass_fraction": nested_outer_pass_fraction,
            "all_requested_folds_usable": all_requested_folds_usable,
            "all_nested_outer_folds_evaluable":
                all_nested_outer_folds_evaluable,
            "stable_cross_fold_eligible_families": sorted(
                stable_eligible
            ),
            "stable_family_count": len(stable_eligible),
            "minimum_stable_families": MIN_STABLE_FAMILIES,
        },
        "champion": {
            "trial_index": champion_trial["trial_index"],
            "frozen_spec": frozen_spec,
            "development_aggregate": champion_trial["aggregate"],
            "single_seed_development_signal_passed":
                single_seed_development_signal_passed,
            "legacy_test_metrics": _compact_evaluation(
                legacy_test_metrics
            ),
            "legacy_test_baselines": {
                name: _compact_evaluation(value)
                for name, value in final_baselines.items()
            },
            "legacy_test_energy_controls": final_energies,
            "legacy_test_failure_codes": legacy_test_failures,
            "legacy_test_diagnostic_passed":
                legacy_diagnostic_passed,
        },
        "decision": {
            "status": status,
            "next_experiment": next_experiment,
            "single_seed_development_signal_passed":
                single_seed_development_signal_passed,
            "cross_seed_aggregation_passed": False,
            "development_candidate_passed": False,
            "legacy_test_diagnostic_passed":
                legacy_diagnostic_passed,
            "fresh_holdout_available": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
        },
        "failure_code_counts": dict(
            sorted(Counter(row["failure_code"] for row in failures).items())
        ),
        "formal_evidence": False,
        "human_labels_asserted": False,
        "training_authorized": False,
    }
    return {
        "trials": trials,
        "folds": fold_rows,
        "failure_memory": failures,
        "predictions": prediction_rows,
        "summary": summary,
        "frozen_transform": final_transform,
        "frozen_dino_basis": final_basis,
        "frozen_appearance_basis": final_appearance_basis,
    }


def _build_r10_examples(
    inputs: r7._Inputs,
    *,
    seed: int,
) -> tuple[list[_R10Example], dict[str, Any]]:
    base_examples, base_coverage = r7._build_examples(inputs, seed=seed)
    cache_rows = inputs.cache["rows"]
    cache_arrays = inputs.cache["arrays"]
    visual_rows = inputs.visual["rows"]
    visual_arrays = inputs.visual["arrays"]
    cache_by_iid = {
        str(row["iid"]): index for index, row in enumerate(cache_rows)
    }
    visual_by_iid = {
        str(row["iid"]): index for index, row in enumerate(visual_rows)
    }
    output: list[_R10Example] = []
    exclusions: Counter[str] = Counter()
    for base in base_examples:
        visual_index = visual_by_iid[base.iid]
        cache_index = cache_by_iid[base.iid]
        if not bool(visual_arrays["source_valid"][visual_index]):
            exclusions["source_dino_invalid"] += 1
            continue
        source_dino = np.asarray(
            visual_arrays["source_dino_cls"][visual_index],
            dtype=np.float64,
        )
        target_dino = np.asarray(
            visual_arrays["target_dino_cls"][visual_index],
            dtype=np.float64,
        )
        if (
            source_dino.ndim != 2
            or target_dino.shape != source_dino.shape
            or len(source_dino) < 3
            or not np.isfinite(source_dino).all()
            or not np.isfinite(target_dino).all()
        ):
            raise R10DynamicRepresentationError(
                f"iid={base.iid} paired DINO shape differs"
            )
        source_track_flat, _endpoint, _orderless, _energy = (
            r7._motion_descriptors(
                cache_arrays["source_stabilized_tracks"][cache_index],
                cache_arrays["source_visibility"][cache_index],
            )
        )
        target_track_flat, _endpoint, _orderless, motion_energy = (
            r7._motion_descriptors(
                cache_arrays["target_stabilized_tracks"][cache_index],
                cache_arrays["target_visibility"][cache_index],
            )
        )
        reverse_source_flat, _endpoint, _orderless, _energy = (
            r7._motion_descriptors(
                cache_arrays["source_stabilized_tracks"][cache_index][::-1],
                cache_arrays["source_visibility"][cache_index][::-1],
            )
        )
        reverse_target_flat, _endpoint, _orderless, _energy = (
            r7._motion_descriptors(
                cache_arrays["target_stabilized_tracks"][cache_index][::-1],
                cache_arrays["target_visibility"][cache_index][::-1],
            )
        )
        frames = (
            cache_arrays["target_stabilized_tracks"][cache_index].shape[0]
            - 1
        )
        source_track = source_track_flat.reshape(frames, 15)
        target_track = target_track_flat.reshape(frames, 15)
        reverse_source_track = reverse_source_flat.reshape(frames, 15)
        reverse_target_track = reverse_target_flat.reshape(frames, 15)
        output.append(
            _R10Example(
                iid=base.iid,
                label_class=base.label_class,
                family=base.family,
                original_split=base.split,
                component_id=base.component_id,
                fresh=base.fresh,
                sampling_weight=base.sampling_weight,
                motion_energy=float(motion_energy),
                source_dino=np.ascontiguousarray(source_dino),
                target_dino=np.ascontiguousarray(target_dino),
                source_track=np.ascontiguousarray(source_track),
                target_track=np.ascontiguousarray(target_track),
                reverse_source_track=np.ascontiguousarray(
                    reverse_source_track
                ),
                reverse_target_track=np.ascontiguousarray(
                    reverse_target_track
                ),
                pooled_target_dino=_unit_vector(
                    np.mean(target_dino, axis=0)
                ),
                target_endpoint=np.asarray(
                    base.features[r7.TARGET_ENDPOINT],
                    dtype=np.float64,
                ),
                target_orderless=np.asarray(
                    base.features[r7.ORDERLESS_TEMPORAL],
                    dtype=np.float64,
                ),
                camera_nuisance=np.asarray(
                    base.features[r7.CAMERA_NUISANCE],
                    dtype=np.float64,
                ),
            )
        )
    coverage = float(len(output) / len(base_examples)) if base_examples else 0.0
    return output, {
        "r7_common_cohort": base_coverage,
        "r7_common_rows": len(base_examples),
        "r10_paired_source_target_dino_rows": len(output),
        "r10_common_cohort_fraction_of_r7": coverage,
        "r10_exclusion_reason_counts": dict(sorted(exclusions.items())),
        "all_specs_share_exact_r10_cohort": True,
        "minimum_required_fraction": MIN_COHORT_COVERAGE,
    }


def _transform_arrays(
    transform: _FittedTransform,
    dino_basis: _DinoBasis,
    appearance_basis: _DinoBasis,
) -> dict[str, np.ndarray]:
    arrays = {
        "schema_version": np.asarray([TRANSFORM_SCHEMA]),
        "spec_digest": np.asarray([transform.spec_digest]),
        "raw_mean": np.asarray(transform.raw_mean, dtype=np.float64),
        "raw_scale": np.asarray(transform.raw_scale, dtype=np.float64),
        "appearance_mean": np.asarray(
            transform.appearance_mean,
            dtype=np.float64,
        ),
        "appearance_scale": np.asarray(
            transform.appearance_scale,
            dtype=np.float64,
        ),
        "content_ridge": np.asarray(
            transform.content_ridge,
            dtype=np.float64,
        ),
        "projection": np.asarray(transform.projection, dtype=np.float64),
        "action_head": np.asarray(
            transform.action_head,
            dtype=np.float64,
        ),
        "action_families": np.asarray(transform.action_families),
        "geometry_keep": np.asarray(
            [transform.geometry_keep],
            dtype=np.int64,
        ),
        "raw_dimension": np.asarray(
            [transform.raw_dimension],
            dtype=np.int64,
        ),
        "embedding_dimension": np.asarray(
            [transform.embedding_dimension],
            dtype=np.int64,
        ),
        "dino_mean": np.asarray(dino_basis.mean, dtype=np.float64),
        "dino_basis": np.asarray(dino_basis.basis, dtype=np.float64),
        "appearance_dino_mean": np.asarray(
            appearance_basis.mean,
            dtype=np.float64,
        ),
        "appearance_dino_basis": np.asarray(
            appearance_basis.basis,
            dtype=np.float64,
        ),
    }
    if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "f"):
        raise R10DynamicRepresentationError(
            "frozen transform contains non-finite arrays"
        )
    return arrays


def _array_records(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        records[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes_sha256": _digest_bytes(array.tobytes()),
        }
    return records


def _transform_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    handle = io.BytesIO()
    np.savez_compressed(handle, **dict(arrays))
    return handle.getvalue()


def _implementation_provenance() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(strict=True),
        Path(r7.__file__).resolve(strict=True),
        Path(artifact_permissions.__file__).resolve(strict=True),
    )
    files = {
        path.name: _digest_file(path)
        for path in sorted(paths, key=lambda item: item.name)
    }
    return {
        "files": files,
        "bundle_sha256": _object_digest(files),
    }


def _publish(
    output_dir: Path,
    *,
    trials: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    transform_arrays: Mapping[str, np.ndarray],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    payloads = {
        TRIALS_NAME: _jsonl_bytes(trials),
        FOLDS_NAME: _jsonl_bytes(folds),
        FAILURES_NAME: _jsonl_bytes(failures),
        PREDICTIONS_NAME: _jsonl_bytes(predictions),
        SUMMARY_NAME: _pretty_json_bytes(summary),
        TRANSFORM_NAME: _transform_bytes(transform_arrays),
    }
    records = {
        name: {
            "sha256": _digest_bytes(payload),
            "bytes": len(payload),
        }
        for name, payload in payloads.items()
    }
    decision = summary["decision"]
    if any(
        decision.get(field) is not False
        for field in (
            "representation_gate_passed",
            "renderer_probe_authorized",
            "editor_training_authorized",
        )
    ):
        raise R10DynamicRepresentationError(
            "R10A publication attempted to open a promotion/training gate"
        )
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "payload_files": records,
        "artifact_digest": _object_digest(records),
        "frozen_spec_digest":
            summary["champion"]["frozen_spec"]["spec_digest"],
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
        "permission_contract": artifact_permissions.permission_contract(),
    }
    payloads[DONE_NAME] = _pretty_json_bytes(done)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    )
    try:
        for name in OUTPUT_NAMES:
            path = stage / name
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(payloads[name])
                    handle.flush()
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        artifact_permissions.seal_staging_tree(
            stage,
            leave_root_writable=True,
        )
        artifact_permissions.assert_sealed_tree(
            stage,
            allow_writable_root=True,
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        os.rename(stage, output_dir)
        artifact_permissions.seal_published_root(output_dir)
    finally:
        if stage.exists():
            for path in sorted(stage.iterdir()):
                path.chmod(0o600)
                path.unlink()
            stage.chmod(0o700)
            stage.rmdir()


def validate_published_search(output_dir: Path) -> dict[str, Any]:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    if {path.name for path in root.iterdir()} != set(OUTPUT_NAMES):
        raise R10DynamicRepresentationError(
            "R10 published artifact closure differs"
        )
    artifact_permissions.assert_sealed_tree(root)
    payload_bytes = {
        name: (root / name).read_bytes() for name in OUTPUT_NAMES
    }
    try:
        done = json.loads(payload_bytes[DONE_NAME])
        summary = json.loads(payload_bytes[SUMMARY_NAME])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise R10DynamicRepresentationError(
            "R10 summary/done is invalid JSON"
        ) from error
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or summary.get("schema_version") != SEARCH_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise R10DynamicRepresentationError(
            "R10 summary/done semantic contract differs"
        )
    records = done.get("payload_files")
    if (
        not isinstance(records, dict)
        or set(records) != set(PAYLOAD_NAMES)
    ):
        raise R10DynamicRepresentationError(
            "R10 payload registry differs"
        )
    for name in PAYLOAD_NAMES:
        payload = payload_bytes[name]
        record = records[name]
        if (
            not isinstance(record, dict)
            or record.get("sha256") != _digest_bytes(payload)
            or record.get("bytes") != len(payload)
        ):
            raise R10DynamicRepresentationError(
                f"R10 payload digest differs: {name}"
            )
    if done.get("artifact_digest") != _object_digest(records):
        raise R10DynamicRepresentationError(
            "R10 artifact digest differs"
        )
    decision = summary.get("decision")
    if (
        not isinstance(decision, dict)
        or any(
            decision.get(field) is not False
            for field in (
                "representation_gate_passed",
                "renderer_probe_authorized",
                "editor_training_authorized",
            )
        )
        or any(
            done.get(field) is not False
            for field in (
                "representation_gate_passed",
                "renderer_probe_authorized",
                "editor_training_authorized",
            )
        )
    ):
        raise R10DynamicRepresentationError(
            "R10 promotion/training closure differs"
        )
    source_snapshot = summary.get("source_snapshot")
    source_tree_sha256 = (
        source_snapshot.get("tree_sha256")
        if isinstance(source_snapshot, dict)
        else None
    )
    controller_verified = (
        source_snapshot.get(
            "exact_tree_verified_by_controller_before_search"
        )
        if isinstance(source_snapshot, dict)
        else None
    )
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_tree_sha256
        )
        or not isinstance(controller_verified, bool)
    ):
        raise R10DynamicRepresentationError(
            "R10 source snapshot contract differs"
        )
    jsonl_rows: dict[str, list[dict[str, Any]]] = {}
    for name, schema in (
        (TRIALS_NAME, TRIAL_SCHEMA),
        (FOLDS_NAME, FOLD_SCHEMA),
        (FAILURES_NAME, FAILURE_SCHEMA),
    ):
        rows = [
            json.loads(line)
            for line in payload_bytes[name].decode("utf-8").splitlines()
        ]
        if not rows or any(row.get("schema_version") != schema for row in rows):
            raise R10DynamicRepresentationError(
                f"R10 JSONL schema differs: {name}"
            )
        jsonl_rows[name] = rows
    fold_protocol = summary.get("fold_protocol")
    if (
        not isinstance(fold_protocol, dict)
        or fold_protocol.get("assignment_digest_excludes_seed") is not True
        or fold_protocol.get("development_fold_assignment_sha256")
        != _fold_assignment_rows_digest(jsonl_rows[FOLDS_NAME])
    ):
        raise R10DynamicRepresentationError(
            "R10 fold assignment commitment differs"
        )
    prediction_rows = [
        json.loads(line)
        for line in payload_bytes[PREDICTIONS_NAME]
        .decode("utf-8")
        .splitlines()
    ]
    if not prediction_rows or any(
        row.get("schema_version")
        != "motive-r10a-champion-prediction-v1"
        for row in prediction_rows
    ):
        raise R10DynamicRepresentationError(
            "R10 champion prediction schema differs"
        )
    try:
        archive = np.load(
            io.BytesIO(payload_bytes[TRANSFORM_NAME]),
            allow_pickle=False,
        )
        arrays = {name: archive[name] for name in archive.files}
    except (ValueError, OSError) as error:
        raise R10DynamicRepresentationError(
            "R10 frozen transform archive is invalid"
        ) from error
    required_arrays = {
        "schema_version",
        "spec_digest",
        "raw_mean",
        "raw_scale",
        "appearance_mean",
        "appearance_scale",
        "content_ridge",
        "projection",
        "action_head",
        "action_families",
        "geometry_keep",
        "raw_dimension",
        "embedding_dimension",
        "dino_mean",
        "dino_basis",
        "appearance_dino_mean",
        "appearance_dino_basis",
    }
    if set(arrays) != required_arrays:
        raise R10DynamicRepresentationError(
            "R10 frozen transform array closure differs"
        )
    if (
        arrays["schema_version"].tolist() != [TRANSFORM_SCHEMA]
        or arrays["spec_digest"].tolist()
        != [summary["champion"]["frozen_spec"]["spec_digest"]]
        or done.get("frozen_spec_digest")
        != summary["champion"]["frozen_spec"]["spec_digest"]
        or any(
            not np.isfinite(value).all()
            for value in arrays.values()
            if value.dtype.kind in "f"
        )
    ):
        raise R10DynamicRepresentationError(
            "R10 frozen transform semantic contract differs"
        )
    raw_dimension = int(arrays["raw_dimension"][0])
    embedding_dimension = int(arrays["embedding_dimension"][0])
    geometry_keep = int(arrays["geometry_keep"][0])
    projection = arrays["projection"]
    action_head = arrays["action_head"]
    action_families = arrays["action_families"]
    appearance_dimension = arrays["appearance_mean"].shape[0]
    expected_embedding_dimension = (
        action_head.shape[1] + geometry_keep
        if action_head.shape[1]
        else projection.shape[1]
    )
    frozen_spec = summary["champion"]["frozen_spec"]
    if (
        arrays["raw_mean"].shape != (raw_dimension,)
        or arrays["raw_scale"].shape != (raw_dimension,)
        or arrays["appearance_mean"].ndim != 1
        or arrays["appearance_scale"].shape != (appearance_dimension,)
        or arrays["content_ridge"].shape
        != (appearance_dimension, raw_dimension)
        or projection.ndim != 2
        or projection.shape[0] != raw_dimension
        or action_head.ndim != 2
        or action_head.shape[0] != projection.shape[1]
        or action_families.ndim != 1
        or len(action_families) != action_head.shape[1]
        or not 0 <= geometry_keep <= projection.shape[1]
        or embedding_dimension != expected_embedding_dimension
        or arrays["dino_basis"].ndim != 2
        or arrays["dino_basis"].shape[0] != arrays["dino_mean"].shape[0]
        or arrays["appearance_dino_basis"].ndim != 2
        or arrays["appearance_dino_basis"].shape[0]
        != arrays["appearance_dino_mean"].shape[0]
        or (
            frozen_spec["head"] == "identity"
            and (
                action_head.shape[1] != 0
                or len(action_families) != 0
                or geometry_keep != 0
            )
        )
    ):
        raise R10DynamicRepresentationError(
            "R10 frozen transform shape contract differs"
        )
    expected_array_records = summary["frozen_transform"][
        "array_records"
    ]
    if _array_records(arrays) != expected_array_records:
        raise R10DynamicRepresentationError(
            "R10 frozen transform array digest differs"
        )
    return {
        "root": str(root),
        "done": done,
        "summary": summary,
        "trials": len(
            payload_bytes[TRIALS_NAME].decode("utf-8").splitlines()
        ),
        "predictions": len(prediction_rows),
    }


def run_search(
    *,
    candidate_manifest_dir: Path,
    expected_candidate_manifest_done_sha256: str,
    track_cache_final: Path,
    expected_track_cache_done_sha256: str,
    visual_features_final: Path,
    expected_visual_features_done_sha256: str,
    visual_candidates_manifest: Path,
    expected_visual_candidates_sha256: str,
    source_tree_sha256: str,
    output_dir: Path,
    source_tree_verified_by_controller: bool = False,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    folds: int = DEFAULT_FOLDS,
    maximum_trials: int | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
        or any(character not in "0123456789abcdef"
               for character in source_tree_sha256)
    ):
        raise R10DynamicRepresentationError(
            "source tree SHA-256 is invalid"
        )
    if not isinstance(source_tree_verified_by_controller, bool):
        raise R10DynamicRepresentationError(
            "source tree verification provenance must be boolean"
        )
    inputs = r7._validate_inputs(
        candidate_manifest_dir=candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            expected_candidate_manifest_done_sha256
        ),
        track_cache_final=track_cache_final,
        expected_track_cache_done_sha256=(
            expected_track_cache_done_sha256
        ),
        visual_features_final=visual_features_final,
        expected_visual_features_done_sha256=(
            expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            expected_visual_candidates_sha256
        ),
        verify_source_shards=True,
        rehash_videos=True,
    )
    examples, coverage = _build_r10_examples(inputs, seed=seed)
    cohort_fraction = coverage[
        "r10_common_cohort_fraction_of_r7"
    ]
    if cohort_fraction < MIN_COHORT_COVERAGE:
        raise R10DynamicRepresentationError(
            "paired source/target DINO coverage is below the preregistered "
            "R10 threshold"
        )
    result = search_examples(
        examples,
        seed=seed,
        repeats=repeats,
        folds=folds,
        maximum_trials=maximum_trials,
    )
    summary = dict(result["summary"])
    summary["input_coverage"] = coverage
    summary["input_bindings"] = {
        "candidate_manifest_dir": str(
            candidate_manifest_dir.expanduser().resolve(strict=True)
        ),
        "candidate_manifest_done_sha256":
            expected_candidate_manifest_done_sha256,
        "track_cache_final": str(
            track_cache_final.expanduser().resolve(strict=True)
        ),
        "track_cache_done_sha256": expected_track_cache_done_sha256,
        "visual_features_final": str(
            visual_features_final.expanduser().resolve(strict=True)
        ),
        "visual_features_done_sha256":
            expected_visual_features_done_sha256,
        "visual_candidates_manifest": str(
            visual_candidates_manifest.expanduser().resolve(strict=True)
        ),
        "visual_candidates_sha256": expected_visual_candidates_sha256,
    }
    summary["implementation"] = _implementation_provenance()
    summary["source_snapshot"] = {
        "tree_sha256": source_tree_sha256,
        "exact_tree_verified_by_controller_before_search":
            source_tree_verified_by_controller,
    }
    transform_arrays = _transform_arrays(
        result["frozen_transform"],
        result["frozen_dino_basis"],
        result["frozen_appearance_basis"],
    )
    summary["frozen_transform"] = {
        "schema_version": TRANSFORM_SCHEMA,
        "spec_digest":
            summary["champion"]["frozen_spec"]["spec_digest"],
        "array_records": _array_records(transform_arrays),
        "contains_video_pixels": False,
        "contains_model_weights": False,
        "contains_only_representation_parameters": True,
    }
    r7._assert_inputs_stable(inputs)
    resolved_output = output_dir.expanduser().resolve(strict=False)
    _publish(
        resolved_output,
        trials=result["trials"],
        folds=result["folds"],
        failures=result["failure_memory"],
        predictions=result["predictions"],
        summary=summary,
        transform_arrays=transform_arrays,
    )
    r7._assert_inputs_stable(inputs)
    validated = validate_published_search(resolved_output)
    return {
        **result,
        "summary": summary,
        "output_dir": str(resolved_output.resolve(strict=True)),
        "validated": validated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-candidate-manifest-done-sha256",
        required=True,
    )
    parser.add_argument("--track-cache-final", required=True, type=Path)
    parser.add_argument(
        "--expected-track-cache-done-sha256",
        required=True,
    )
    parser.add_argument("--visual-features-final", required=True, type=Path)
    parser.add_argument(
        "--expected-visual-features-done-sha256",
        required=True,
    )
    parser.add_argument(
        "--visual-candidates-manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--expected-visual-candidates-sha256",
        required=True,
    )
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument(
        "--source-tree-verified-by-controller",
        action="store_true",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--maximum-trials", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_search(
        candidate_manifest_dir=args.candidate_manifest_dir,
        expected_candidate_manifest_done_sha256=(
            args.expected_candidate_manifest_done_sha256
        ),
        track_cache_final=args.track_cache_final,
        expected_track_cache_done_sha256=(
            args.expected_track_cache_done_sha256
        ),
        visual_features_final=args.visual_features_final,
        expected_visual_features_done_sha256=(
            args.expected_visual_features_done_sha256
        ),
        visual_candidates_manifest=args.visual_candidates_manifest,
        expected_visual_candidates_sha256=(
            args.expected_visual_candidates_sha256
        ),
        source_tree_sha256=args.source_tree_sha256,
        output_dir=args.output_dir,
        source_tree_verified_by_controller=(
            args.source_tree_verified_by_controller
        ),
        seed=args.seed,
        repeats=args.repeats,
        folds=args.folds,
        maximum_trials=args.maximum_trials,
    )
    decision = result["summary"]["decision"]
    print(
        "[motive-r10a-dynamic-representation] "
        f"trials={len(result['trials'])} "
        f"status={decision['status']} "
        f"renderer_probe_authorized="
        f"{decision['renderer_probe_authorized']} "
        f"output={result['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

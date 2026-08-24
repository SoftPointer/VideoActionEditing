"""Deterministic, validation-only search for reusable motion representations.

The search consumes the already committed R7 candidate cohort, track cache,
visual features, and content-component split.  It searches weighted,
factorized combinations of motion descriptors on validation only.  A winner
is frozen before the test split is evaluated.  Every attempted spec and every
failure reason is written to an immutable ledger.

This stage performs no renderer inference and no training.  Passing it only
authorizes a later frozen-renderer probe; it never authorizes editor training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_candidate_temporal_screen as r7


SEARCH_SCHEMA = "motive-r9-automated-representation-search-v1"
TRIAL_SCHEMA = "motive-r9-representation-trial-v1"
FAILURE_SCHEMA = "motive-r9-representation-failure-v1"
DONE_SCHEMA = "motive-r9-automated-representation-search-done-v1"

TRIALS_NAME = "trials.jsonl"
FAILURES_NAME = "failure_memory.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (TRIALS_NAME, FAILURES_NAME, SUMMARY_NAME, DONE_NAME)
PAYLOAD_NAMES = (TRIALS_NAME, FAILURES_NAME, SUMMARY_NAME)

TARGET = r7.TARGET_TEMPORAL
DELTA = r7.DELTA_TEMPORAL
ENDPOINT = r7.TARGET_ENDPOINT
ORDERLESS = r7.ORDERLESS_TEMPORAL
CAMERA = r7.CAMERA_NUISANCE
DINO = r7.POOLED_DINO
SHUFFLE = r7.SHUFFLED_QUERY
REVERSE = r7.REVERSED_QUERY

TARGET_PYRAMID = "camera_compensated_target_temporal_pyramid_2_4_8"
DELTA_PYRAMID = "source_to_target_delta_temporal_pyramid_2_4_8"
TARGET_ACCELERATION = "camera_compensated_target_acceleration"
DELTA_ACCELERATION = "source_to_target_delta_acceleration"
TARGET_PHASE = "camera_compensated_target_speed_phase"
DELTA_PHASE = "source_to_target_delta_speed_phase"
DERIVED_SOURCE = {
    TARGET_PYRAMID: TARGET,
    DELTA_PYRAMID: DELTA,
    TARGET_ACCELERATION: TARGET,
    DELTA_ACCELERATION: DELTA,
    TARGET_PHASE: TARGET,
    DELTA_PHASE: DELTA,
}
TARGET_DERIVED = {
    TARGET,
    TARGET_PYRAMID,
    TARGET_ACCELERATION,
    TARGET_PHASE,
}
SEARCH_COMPONENTS = (
    TARGET,
    DELTA,
    TARGET_PYRAMID,
    DELTA_PYRAMID,
    TARGET_ACCELERATION,
    DELTA_ACCELERATION,
    TARGET_PHASE,
    DELTA_PHASE,
    ENDPOINT,
    ORDERLESS,
)
CONTROL_COMPONENTS = (ENDPOINT, ORDERLESS, CAMERA, DINO)
ALL_COMPONENTS = (*SEARCH_COMPONENTS, CAMERA, DINO)
REFERENCE_COMPONENT = {
    TARGET: TARGET,
    DELTA: DELTA,
    ENDPOINT: ENDPOINT,
    ORDERLESS: ORDERLESS,
    CAMERA: CAMERA,
    DINO: DINO,
    TARGET_PYRAMID: TARGET_PYRAMID,
    DELTA_PYRAMID: DELTA_PYRAMID,
    TARGET_ACCELERATION: TARGET_ACCELERATION,
    DELTA_ACCELERATION: DELTA_ACCELERATION,
    TARGET_PHASE: TARGET_PHASE,
    DELTA_PHASE: DELTA_PHASE,
    SHUFFLE: TARGET,
    REVERSE: TARGET,
}

DEFAULT_SEED = 260108835
DEFAULT_GENERATIONS = 5
DEFAULT_BEAM_WIDTH = 8
DEFAULT_MAX_TRIALS = 64
MIN_COVERAGE = 0.90
MIN_CONTROL_MARGIN = 0.02
MIN_TEMPORAL_MARGIN = 0.02
MIN_AUROC = 0.55
MIN_AUROC_OVER_ENERGY = 0.01
_EPS = 1e-12


class AutomatedRepresentationSearchError(ValueError):
    """An input, search, or immutable-publication contract is invalid."""


@dataclasses.dataclass(frozen=True)
class _Matrices:
    queries: tuple[r7._Example, ...]
    bank: tuple[r7._Example, ...]
    dots: Mapping[str, np.ndarray]
    query_nonzero: Mapping[str, np.ndarray]
    bank_nonzero: Mapping[str, np.ndarray]


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


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (_canonical_json(dict(row)) + "\n").encode("utf-8")
        for row in rows
    )


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_digest(value: Any) -> str:
    return _digest_bytes(_canonical_json(value).encode("utf-8"))


def _normalize_weights(raw: Mapping[str, float]) -> dict[str, float]:
    if not raw:
        raise AutomatedRepresentationSearchError(
            "representation has no components"
        )
    unknown = set(raw) - set(ALL_COMPONENTS)
    if unknown:
        raise AutomatedRepresentationSearchError(
            f"unknown search components: {sorted(unknown)}"
        )
    values: dict[str, float] = {}
    for name in ALL_COMPONENTS:
        if name not in raw:
            continue
        value = float(raw[name])
        if not math.isfinite(value) or value <= 0.0:
            raise AutomatedRepresentationSearchError(
                f"invalid component weight: {name}"
            )
        values[name] = value
    total = sum(values.values())
    normalized = {
        name: round(values[name] / total, 12)
        for name in ALL_COMPONENTS
        if name in values
    }
    # Make the canonical sum exact enough for stable cosine bookkeeping.
    last = next(reversed(normalized))
    normalized[last] = round(
        normalized[last] + (1.0 - sum(normalized.values())),
        12,
    )
    return normalized


def _spec(
    weights: Mapping[str, float],
    *,
    parent: str | None,
    generation: int,
    role: str = "search",
) -> dict[str, Any]:
    normalized = _normalize_weights(weights)
    core = {
        "schema_version": "motive-r9-factorized-motion-spec-v1",
        "components": normalized,
        "combination": "weighted-concatenation-then-l2",
        "similarity": "cosine",
        "role": role,
    }
    return {
        **core,
        "spec_digest": _object_digest(core),
        "parent_spec_digest": parent,
        "generation": int(generation),
    }


def _initial_specs() -> list[dict[str, Any]]:
    raw = (
        {TARGET: 1.0},
        {DELTA: 1.0},
        {TARGET_PYRAMID: 1.0},
        {DELTA_PYRAMID: 1.0},
        {TARGET_ACCELERATION: 1.0},
        {DELTA_ACCELERATION: 1.0},
        {TARGET_PHASE: 1.0},
        {DELTA_PHASE: 1.0},
        {ENDPOINT: 1.0},
        {ORDERLESS: 1.0},
        {TARGET: 1.0, DELTA: 1.0},
        {TARGET_PYRAMID: 1.0, DELTA_PYRAMID: 1.0},
        {TARGET_ACCELERATION: 1.0, DELTA_ACCELERATION: 1.0},
        {TARGET_PHASE: 1.0, DELTA_PHASE: 1.0},
        {
            TARGET_PYRAMID: 2.0,
            TARGET_ACCELERATION: 1.0,
            TARGET_PHASE: 1.0,
        },
        {TARGET: 2.0, DELTA: 1.0},
        {TARGET: 1.0, DELTA: 2.0},
        {TARGET: 2.0, ENDPOINT: 1.0},
        {TARGET: 2.0, ORDERLESS: 1.0},
        {DELTA: 2.0, ENDPOINT: 1.0},
        {DELTA: 2.0, ORDERLESS: 1.0},
        {TARGET: 2.0, DELTA: 2.0, ENDPOINT: 1.0},
        {TARGET: 2.0, DELTA: 2.0, ORDERLESS: 1.0},
        {TARGET: 2.0, DELTA: 2.0, ENDPOINT: 1.0, ORDERLESS: 1.0},
    )
    return [_spec(item, parent=None, generation=0) for item in raw]


def _mutations(parent: Mapping[str, Any], generation: int) -> list[dict[str, Any]]:
    weights = dict(parent["components"])
    digest = str(parent["spec_digest"])
    proposals: list[dict[str, float]] = []
    for name in tuple(weights):
        for multiplier in (0.5, 2.0):
            mutated = dict(weights)
            mutated[name] *= multiplier
            proposals.append(mutated)
        if len(weights) > 1:
            mutated = dict(weights)
            del mutated[name]
            proposals.append(mutated)
    for name in SEARCH_COMPONENTS:
        if name not in weights:
            for value in (0.25, 0.5, 1.0):
                mutated = dict(weights)
                mutated[name] = value
                proposals.append(mutated)
    return [
        _spec(
            proposal,
            parent=digest,
            generation=generation,
        )
        for proposal in proposals
    ]


def _eligible_families(
    examples: Sequence[r7._Example],
) -> tuple[set[str], dict[str, Any]]:
    train = [
        item
        for item in examples
        if item.label_class == "positive" and item.split == "train"
    ]
    heldout = [
        item
        for item in examples
        if item.label_class == "positive"
        and item.split in r7.EVAL_SPLITS
    ]
    train_rows = Counter(item.family for item in train)
    train_components: dict[str, set[str]] = defaultdict(set)
    heldout_rows = Counter(item.family for item in heldout)
    for item in train:
        train_components[item.family].add(item.component_id)
    records: dict[str, Any] = {}
    eligible: set[str] = set()
    for family in sorted(set(train_rows) | set(heldout_rows)):
        accepted = (
            train_rows[family] >= r7.MINIMUM_TRAIN_REFERENCES
            and len(train_components[family])
            >= r7.MINIMUM_TRAIN_COMPONENTS
        )
        if accepted:
            eligible.add(family)
        records[family] = {
            "train_references": int(train_rows[family]),
            "train_components": len(train_components[family]),
            "heldout_queries": int(heldout_rows[family]),
            "eligible_from_train_only": accepted,
        }
    return eligible, {
        "thresholds": {
            "minimum_train_references": r7.MINIMUM_TRAIN_REFERENCES,
            "minimum_train_components": r7.MINIMUM_TRAIN_COMPONENTS,
        },
        "families": records,
        "eligible_families": sorted(eligible),
    }


def _temporal_transform(feature: str, value: np.ndarray) -> np.ndarray:
    if feature not in DERIVED_SOURCE:
        return value.reshape(-1)
    vector = value.reshape(-1)
    if not len(vector) or len(vector) % 15:
        raise AutomatedRepresentationSearchError(
            f"ordered temporal feature has invalid length for {feature}"
        )
    sequence = vector.reshape(-1, 15)
    if feature in (TARGET_ACCELERATION, DELTA_ACCELERATION):
        return np.diff(sequence, axis=0).reshape(-1)
    if feature in (TARGET_PHASE, DELTA_PHASE):
        # Speed quantiles retain the action envelope while discarding most
        # spatial appearance-like track layout.
        return sequence[:, 10:15].reshape(-1)
    pooled: list[np.ndarray] = []
    for bins in (2, 4, 8):
        for indices in np.array_split(np.arange(len(sequence)), bins):
            if len(indices):
                pooled.append(np.mean(sequence[indices], axis=0))
            else:
                pooled.append(np.zeros(sequence.shape[1], dtype=np.float64))
    return np.concatenate(pooled)


def _stack(
    examples: Sequence[r7._Example],
    feature: str,
    *,
    temporal_control: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = DERIVED_SOURCE.get(feature, feature)
    if feature in TARGET_DERIVED and temporal_control in (SHUFFLE, REVERSE):
        source = temporal_control
    values = [
        _temporal_transform(
            feature,
            np.asarray(item.features[source], dtype=np.float64),
        )
        for item in examples
    ]
    if not values or any(value.shape != values[0].shape for value in values):
        raise AutomatedRepresentationSearchError(
            f"feature dimensions differ: {feature}"
        )
    matrix = np.stack(values)
    if not np.isfinite(matrix).all():
        raise AutomatedRepresentationSearchError(
            f"feature is non-finite: {feature}"
        )
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > _EPS
    normalized = np.zeros_like(matrix)
    normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
    return normalized, nonzero.astype(np.float64)


def _build_matrices(
    examples: Sequence[r7._Example],
    *,
    split: str,
) -> _Matrices:
    queries = tuple(
        sorted(
            [item for item in examples if item.split == split],
            key=lambda item: item.iid,
        )
    )
    bank = tuple(
        sorted(
            [
                item
                for item in examples
                if item.split == "train"
                and item.label_class == "positive"
            ],
            key=lambda item: item.iid,
        )
    )
    if not queries or len({item.component_id for item in bank}) < 5:
        raise AutomatedRepresentationSearchError(
            f"split={split} lacks queries or five train components"
        )
    query_arrays: dict[str, np.ndarray] = {}
    query_nonzero: dict[str, np.ndarray] = {}
    bank_arrays: dict[str, np.ndarray] = {}
    bank_nonzero: dict[str, np.ndarray] = {}
    for feature in ALL_COMPONENTS:
        query_arrays[feature], query_nonzero[feature] = _stack(
            queries,
            feature,
        )
    for feature in ALL_COMPONENTS:
        bank_arrays[feature], bank_nonzero[feature] = _stack(
            bank,
            feature,
        )
    dots: dict[str, np.ndarray] = {}
    for feature in ALL_COMPONENTS:
        reference = REFERENCE_COMPONENT[feature]
        dots[feature] = (
            query_arrays[feature] @ bank_arrays[reference].T
        )
    for control in (SHUFFLE, REVERSE):
        for feature in ALL_COMPONENTS:
            controlled, controlled_nonzero = _stack(
                queries,
                feature,
                temporal_control=control,
            )
            key = f"{feature}@{control}"
            dots[key] = (
                controlled @ bank_arrays[REFERENCE_COMPONENT[feature]].T
            )
            query_nonzero[key] = controlled_nonzero
    return _Matrices(
        queries=queries,
        bank=bank,
        dots=dots,
        query_nonzero=query_nonzero,
        bank_nonzero=bank_nonzero,
    )


def _similarities(
    matrices: _Matrices,
    weights: Mapping[str, float],
    *,
    temporal_control: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dot = np.zeros(
        (len(matrices.queries), len(matrices.bank)),
        dtype=np.float64,
    )
    query_norm_sq = np.zeros(len(matrices.queries), dtype=np.float64)
    bank_norm_sq = np.zeros(len(matrices.bank), dtype=np.float64)
    for feature, weight in weights.items():
        query_feature = (
            f"{feature}@{temporal_control}"
            if temporal_control in (SHUFFLE, REVERSE)
            else feature
        )
        dot += float(weight) * matrices.dots[query_feature]
        query_norm_sq += (
            float(weight) * matrices.query_nonzero[query_feature]
        )
        bank_norm_sq += (
            float(weight)
            * matrices.bank_nonzero[REFERENCE_COMPONENT[feature]]
        )
    query_norm = np.sqrt(query_norm_sq)
    bank_norm = np.sqrt(bank_norm_sq)
    denominator = query_norm[:, None] * bank_norm[None, :]
    valid = denominator > _EPS
    result = np.full_like(dot, -np.inf)
    result[valid] = np.clip(dot[valid] / denominator[valid], -1.0, 1.0)
    return result, query_norm > _EPS, bank_norm > _EPS


def _rank(
    query: r7._Example,
    bank: Sequence[r7._Example],
    similarities: np.ndarray,
    *,
    query_valid: bool,
    bank_valid: np.ndarray,
) -> tuple[list[r7._Example], list[float], str | None]:
    if not query_valid:
        return [], [], "zero_query"
    candidates: list[tuple[r7._Example, float]] = []
    for index, reference in enumerate(bank):
        if (
            reference.iid == query.iid
            or reference.component_id == query.component_id
            or not bool(bank_valid[index])
            or not math.isfinite(float(similarities[index]))
        ):
            continue
        candidates.append((reference, float(similarities[index])))
    candidates.sort(key=lambda item: (-item[1], item[0].iid))
    independent: list[tuple[r7._Example, float]] = []
    seen: set[str] = set()
    for item in candidates:
        if item[0].component_id in seen:
            continue
        seen.add(item[0].component_id)
        independent.append(item)
        if len(independent) == 5:
            break
    if len(independent) < 5:
        return [], [], "fewer_than_five_valid_reference_components"
    return (
        [item[0] for item in independent],
        [item[1] for item in independent],
        None,
    )


def _retrieval_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["eligible_positive_query"] is True
    ]
    per_family: dict[str, dict[str, float | int]] = {}
    for family in sorted({str(row["family"]) for row in selected}):
        family_rows = [row for row in selected if row["family"] == family]
        per_family[family] = {
            "queries": len(family_rows),
            "r_at_1": float(
                np.mean([row[field]["correct_at_1"] is True for row in family_rows])
            ),
            "r_at_5": float(
                np.mean([row[field]["correct_at_5"] is True for row in family_rows])
            ),
        }
    queries = len(selected)
    valid = sum(row[field]["valid"] is True for row in selected)
    return {
        "queries": queries,
        "valid_queries": valid,
        "valid_fraction": float(valid / queries) if queries else None,
        "micro_r_at_1": (
            float(np.mean([row[field]["correct_at_1"] is True for row in selected]))
            if selected
            else None
        ),
        "micro_r_at_5": (
            float(np.mean([row[field]["correct_at_5"] is True for row in selected]))
            if selected
            else None
        ),
        "macro_family_r_at_1": (
            float(np.mean([record["r_at_1"] for record in per_family.values()]))
            if per_family
            else None
        ),
        "macro_family_r_at_5": (
            float(np.mean([record["r_at_5"] for record in per_family.values()]))
            if per_family
            else None
        ),
        "per_family": per_family,
    }


def _evaluate_spec(
    matrices: _Matrices,
    spec: Mapping[str, Any],
    *,
    eligible_families: set[str],
) -> dict[str, Any]:
    weights = dict(spec["components"])
    clean, clean_query, clean_bank = _similarities(matrices, weights)
    shuffled, shuffle_query, shuffle_bank = _similarities(
        matrices,
        weights,
        temporal_control=SHUFFLE,
    )
    reversed_matrix, reverse_query, reverse_bank = _similarities(
        matrices,
        weights,
        temporal_control=REVERSE,
    )
    rows: list[dict[str, Any]] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    positive_weights: list[float] = []
    negative_weights: list[float] = []
    for index, query in enumerate(matrices.queries):
        arms: dict[str, Any] = {}
        for name, matrix, qvalid, bvalid in (
            ("clean", clean, clean_query, clean_bank),
            ("shuffle", shuffled, shuffle_query, shuffle_bank),
            ("reverse", reversed_matrix, reverse_query, reverse_bank),
        ):
            references, similarities, invalid = _rank(
                query,
                matrices.bank,
                matrix[index],
                query_valid=bool(qvalid[index]),
                bank_valid=bvalid,
            )
            eligible = (
                query.label_class == "positive"
                and query.family in eligible_families
            )
            arms[name] = {
                "valid": invalid is None,
                "invalid_reason": invalid,
                "correct_at_1": (
                    references[0].family == query.family
                    if eligible and invalid is None
                    else None
                ),
                "correct_at_5": (
                    any(item.family == query.family for item in references)
                    if eligible and invalid is None
                    else None
                ),
                "top_reference_iids": [item.iid for item in references],
                "top_reference_components": [
                    item.component_id for item in references
                ],
                "top_reference_families": [item.family for item in references],
                "cosine_similarities": similarities,
            }
        score = (
            float(arms["clean"]["cosine_similarities"][0])
            if arms["clean"]["valid"]
            else None
        )
        if score is not None:
            if query.label_class == "positive":
                positive_scores.append(score)
                positive_weights.append(float(query.sampling_weight))
            elif query.label_class == "negative":
                negative_scores.append(score)
                negative_weights.append(float(query.sampling_weight))
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
                **arms,
            }
        )
    clean_summary = _retrieval_summary(rows, field="clean")
    shuffle_summary = _retrieval_summary(rows, field="shuffle")
    reverse_summary = _retrieval_summary(rows, field="reverse")
    auc = r7._weighted_auc(
        positive_scores,
        negative_scores,
        positive_weights,
        negative_weights,
    )
    clean_r5 = clean_summary["macro_family_r_at_5"]
    temporal_control_r5 = max(
        value
        for value in (
            shuffle_summary["macro_family_r_at_5"],
            reverse_summary["macro_family_r_at_5"],
        )
        if value is not None
    )
    return {
        "split": matrices.queries[0].split,
        "retrieval": clean_summary,
        "shuffle_control": shuffle_summary,
        "reverse_control": reverse_summary,
        "temporal_order_margin_macro_r_at_5": (
            float(clean_r5 - temporal_control_r5)
            if clean_r5 is not None
            else None
        ),
        "positive_vs_negative_sampling_weighted_auroc": auc,
        "valid_binary_positive_rows": len(positive_scores),
        "valid_binary_negative_rows": len(negative_scores),
        "component_exclusion_enforced": True,
        "one_reference_per_component": True,
    }


def _energy_auc(
    examples: Sequence[r7._Example],
    *,
    split: str,
) -> float | None:
    positive = [
        item for item in examples
        if item.split == split and item.label_class == "positive"
    ]
    negative = [
        item for item in examples
        if item.split == split and item.label_class == "negative"
    ]
    return r7._weighted_auc(
        [item.motion_energy for item in positive],
        [item.motion_energy for item in negative],
        [item.sampling_weight for item in positive],
        [item.sampling_weight for item in negative],
    )


def _objective(metrics: Mapping[str, Any]) -> float:
    retrieval = metrics["retrieval"]
    values = (
        retrieval["macro_family_r_at_5"],
        retrieval["macro_family_r_at_1"],
        metrics["positive_vs_negative_sampling_weighted_auroc"],
        metrics["temporal_order_margin_macro_r_at_5"],
        retrieval["valid_fraction"],
    )
    if any(value is None for value in values):
        return -1e9
    return float(
        values[0]
        + 0.25 * values[1]
        + 0.10 * values[2]
        + 0.20 * max(0.0, values[3])
        + 0.10 * values[4]
    )


def _gate_failures(
    metrics: Mapping[str, Any],
    *,
    baselines: Mapping[str, Mapping[str, Any]],
    energy_auc: float | None,
) -> list[str]:
    failures: list[str] = []
    retrieval = metrics["retrieval"]
    coverage = retrieval["valid_fraction"]
    if coverage is None or coverage < MIN_COVERAGE:
        failures.append("representation_zero_or_unstable")
    if not retrieval["queries"]:
        failures.append("family_support_insufficient")
    candidate_r5 = retrieval["macro_family_r_at_5"]
    control_values = [
        record["retrieval"]["macro_family_r_at_5"]
        for record in baselines.values()
        if record["retrieval"]["macro_family_r_at_5"] is not None
    ]
    strongest_control = max(control_values) if control_values else None
    if (
        candidate_r5 is None
        or strongest_control is None
        or candidate_r5 < strongest_control + MIN_CONTROL_MARGIN
    ):
        failures.append("content_generalization_failed")
    temporal_margin = metrics["temporal_order_margin_macro_r_at_5"]
    if temporal_margin is None or temporal_margin < MIN_TEMPORAL_MARGIN:
        failures.append("temporal_order_insensitive")
    auc = metrics["positive_vs_negative_sampling_weighted_auroc"]
    required_auc = MIN_AUROC
    if energy_auc is not None:
        required_auc = max(
            required_auc,
            energy_auc + MIN_AUROC_OVER_ENERGY,
        )
    if auc is None or auc < required_auc:
        failures.append("static_or_low_motion_coverage")
    return sorted(set(failures))


def search_examples(
    examples: Sequence[r7._Example],
    *,
    seed: int = DEFAULT_SEED,
    generations: int = DEFAULT_GENERATIONS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> dict[str, Any]:
    """Search validation, freeze one spec, and only then evaluate test."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise AutomatedRepresentationSearchError("invalid seed")
    for name, value in (
        ("generations", generations),
        ("beam_width", beam_width),
        ("max_trials", max_trials),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AutomatedRepresentationSearchError(f"invalid {name}")

    eligible, support = _eligible_families(examples)
    if len(eligible) < r7.MINIMUM_ELIGIBLE_FAMILIES:
        raise AutomatedRepresentationSearchError(
            "fewer than two train-supported positive families"
        )
    validation = _build_matrices(examples, split="validation")
    test = _build_matrices(examples, split="test")
    validation_energy_auc = _energy_auc(examples, split="validation")
    test_energy_auc = _energy_auc(examples, split="test")

    baseline_specs = {
        name: _spec(
            {name: 1.0},
            parent=None,
            generation=-1,
            role="shortcut_or_information_ablation_control",
        )
        for name in CONTROL_COMPONENTS
    }
    validation_baselines = {
        name: _evaluate_spec(
            validation,
            spec,
            eligible_families=eligible,
        )
        for name, spec in baseline_specs.items()
    }

    trials: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    frontier = _initial_specs()
    for generation in range(generations):
        evaluated_generation: list[dict[str, Any]] = []
        for spec in sorted(frontier, key=lambda item: item["spec_digest"]):
            digest = str(spec["spec_digest"])
            if digest in seen or len(trials) >= max_trials:
                continue
            seen.add(digest)
            started = time.monotonic()
            metrics = _evaluate_spec(
                validation,
                spec,
                eligible_families=eligible,
            )
            objective = _objective(metrics)
            failures = _gate_failures(
                metrics,
                baselines=validation_baselines,
                energy_auc=validation_energy_auc,
            )
            trial = {
                "schema_version": TRIAL_SCHEMA,
                "trial_index": len(trials),
                "seed": seed,
                "generation": generation,
                "spec": dict(spec),
                "selection_split": "validation",
                "test_metrics_read": False,
                "validation_metrics": metrics,
                "objective": objective,
                "development_gate_failures": failures,
                "development_gate_passed": not failures,
                "elapsed_seconds": float(time.monotonic() - started),
                "renderer_probe_authorized": False,
                "training_authorized": False,
            }
            trials.append(trial)
            evaluated_generation.append(trial)
            for code in failures:
                failure_rows.append(
                    {
                        "schema_version": FAILURE_SCHEMA,
                        "trial_index": trial["trial_index"],
                        "spec_digest": digest,
                        "generation": generation,
                        "failure_code": code,
                        "selection_split": "validation",
                        "retry_same_spec_allowed": False,
                    }
                )
        if len(trials) >= max_trials or not evaluated_generation:
            break
        ranked = sorted(
            trials,
            key=lambda item: (
                -float(item["objective"]),
                str(item["spec"]["spec_digest"]),
            ),
        )
        parents = ranked[:beam_width]
        proposals: list[dict[str, Any]] = []
        for parent in parents:
            proposals.extend(
                _mutations(parent["spec"], generation + 1)
            )
        frontier = proposals

    if not trials:
        raise AutomatedRepresentationSearchError("search evaluated no trials")
    ranked = sorted(
        trials,
        key=lambda item: (
            -float(item["objective"]),
            str(item["spec"]["spec_digest"]),
        ),
    )
    champion_trial = ranked[0]
    frozen_spec = dict(champion_trial["spec"])
    # This is the first point at which test-derived metrics are computed for
    # a searched spec.  They never affect the frozen spec digest.
    test_metrics = _evaluate_spec(
        test,
        frozen_spec,
        eligible_families=eligible,
    )
    test_baselines = {
        name: _evaluate_spec(
            test,
            spec,
            eligible_families=eligible,
        )
        for name, spec in baseline_specs.items()
    }
    validation_failures = list(
        champion_trial["development_gate_failures"]
    )
    test_failures = _gate_failures(
        test_metrics,
        baselines=test_baselines,
        energy_auc=test_energy_auc,
    )
    for code in test_failures:
        failure_rows.append(
            {
                "schema_version": FAILURE_SCHEMA,
                "trial_index": champion_trial["trial_index"],
                "spec_digest": frozen_spec["spec_digest"],
                "generation": champion_trial["generation"],
                "failure_code": code,
                "selection_split": "frozen_test",
                "retry_same_spec_allowed": False,
            }
        )
    passed = not validation_failures and not test_failures
    return {
        "trials": trials,
        "failure_memory": failure_rows,
        "summary": {
            "schema_version": SEARCH_SCHEMA,
            "status": "complete",
            "scope": (
                "no-gradient factorized motion representation search; "
                "source-video + instruction editor remains downstream"
            ),
            "seed": seed,
            "budget": {
                "generations": generations,
                "beam_width": beam_width,
                "max_trials": max_trials,
                "realized_trials": len(trials),
                "unique_specs": len(seen),
            },
            "support": support,
            "selection_protocol": {
                "search_split": "validation_only",
                "test_opened_after_spec_freeze": True,
                "test_used_for_search_or_threshold_selection": False,
                "train_reference_bank_positive_only": True,
                "query_reference_same_component_excluded": True,
                "one_reference_per_component": True,
                "labels_are_pseudo": True,
            },
            "controls": {
                "validation": validation_baselines,
                "test": test_baselines,
                "validation_motion_energy_auroc":
                    validation_energy_auc,
                "test_motion_energy_auroc": test_energy_auc,
            },
            "champion": {
                "trial_index": champion_trial["trial_index"],
                "frozen_spec": frozen_spec,
                "validation_objective": champion_trial["objective"],
                "validation_metrics":
                    champion_trial["validation_metrics"],
                "test_metrics": test_metrics,
                "validation_failure_codes": validation_failures,
                "test_failure_codes": test_failures,
            },
            "failure_code_counts": dict(
                sorted(
                    Counter(
                        row["failure_code"] for row in failure_rows
                    ).items()
                )
            ),
            "decision": {
                "representation_gate_passed": passed,
                "status": (
                    "DEVELOPMENT_PROMOTED_TO_FROZEN_RENDERER_PROBE"
                    if passed
                    else "CONTINUE_REPRESENTATION_SEARCH"
                ),
                "renderer_probe_authorized": passed,
                "editor_training_authorized": False,
                "reason": (
                    "validation and frozen test gates passed"
                    if passed
                    else "one or more validation/frozen-test gates failed"
                ),
            },
            "formal_evidence": False,
            "human_labels_asserted": False,
            "training_authorized": False,
        },
    }


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
    failures: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    trial_bytes = _jsonl_bytes(trials)
    failure_bytes = _jsonl_bytes(failures)
    summary_bytes = _pretty_json_bytes(summary)
    payloads = {
        TRIALS_NAME: trial_bytes,
        FAILURES_NAME: failure_bytes,
        SUMMARY_NAME: summary_bytes,
    }
    records = {
        name: {"sha256": _digest_bytes(payload), "bytes": len(payload)}
        for name, payload in payloads.items()
    }
    done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "payload_files": records,
        "artifact_digest": _object_digest(records),
        "representation_gate_passed":
            bool(summary["decision"]["representation_gate_passed"]),
        "renderer_probe_authorized":
            bool(summary["decision"]["renderer_probe_authorized"]),
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
            artifact_permissions.remove_staging_tree(stage)


def validate_published_search(output_dir: Path) -> dict[str, Any]:
    """Fail closed on a pre-existing immutable R9 search commit."""

    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise AutomatedRepresentationSearchError(
            "search commit is not a regular directory"
        )
    root = unresolved.resolve(strict=True)
    artifact_permissions.assert_sealed_tree(root)
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != set(OUTPUT_NAMES):
        raise AutomatedRepresentationSearchError(
            "search commit artifact closure differs"
        )

    payload_bytes: dict[str, bytes] = {}
    for name in OUTPUT_NAMES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise AutomatedRepresentationSearchError(
                f"search commit member is not a regular file: {name}"
            )
        payload_bytes[name] = path.read_bytes()

    try:
        done = json.loads(payload_bytes[DONE_NAME])
        summary = json.loads(payload_bytes[SUMMARY_NAME])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AutomatedRepresentationSearchError(
            "search commit JSON is invalid"
        ) from error
    if not isinstance(done, Mapping) or not isinstance(summary, Mapping):
        raise AutomatedRepresentationSearchError(
            "search commit JSON roots must be objects"
        )
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "complete"
        or done.get("editor_training_authorized") is not False
        or done.get("permission_contract")
        != artifact_permissions.permission_contract()
    ):
        raise AutomatedRepresentationSearchError(
            "search done contract differs"
        )

    expected_records = done.get("payload_files")
    if (
        not isinstance(expected_records, Mapping)
        or set(expected_records) != set(PAYLOAD_NAMES)
    ):
        raise AutomatedRepresentationSearchError(
            "search payload record closure differs"
        )
    actual_records: dict[str, dict[str, Any]] = {}
    for name in PAYLOAD_NAMES:
        expected = expected_records[name]
        if not isinstance(expected, Mapping):
            raise AutomatedRepresentationSearchError(
                f"search payload record is not an object: {name}"
            )
        payload = payload_bytes[name]
        record = {
            "sha256": _digest_bytes(payload),
            "bytes": len(payload),
        }
        if dict(expected) != record:
            raise AutomatedRepresentationSearchError(
                f"search payload digest differs: {name}"
            )
        actual_records[name] = record
    if done.get("artifact_digest") != _object_digest(actual_records):
        raise AutomatedRepresentationSearchError(
            "search artifact digest differs"
        )

    decision = summary.get("decision")
    if not isinstance(decision, Mapping):
        raise AutomatedRepresentationSearchError(
            "search summary decision is not an object"
        )
    gate_passed = decision.get("representation_gate_passed")
    renderer_authorized = decision.get("renderer_probe_authorized")
    if (
        not isinstance(gate_passed, bool)
        or not isinstance(renderer_authorized, bool)
        or renderer_authorized != gate_passed
        or decision.get("editor_training_authorized") is not False
        or summary.get("training_authorized") is not False
        or done.get("representation_gate_passed") != gate_passed
        or done.get("renderer_probe_authorized") != renderer_authorized
    ):
        raise AutomatedRepresentationSearchError(
            "search promotion/training gate differs"
        )

    for name, expected_schema in (
        (TRIALS_NAME, TRIAL_SCHEMA),
        (FAILURES_NAME, FAILURE_SCHEMA),
    ):
        try:
            rows = [
                json.loads(line)
                for line in payload_bytes[name].decode("utf-8").splitlines()
                if line
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AutomatedRepresentationSearchError(
                f"search ledger is invalid JSONL: {name}"
            ) from error
        if any(
            not isinstance(row, Mapping)
            or row.get("schema_version") != expected_schema
            for row in rows
        ):
            raise AutomatedRepresentationSearchError(
                f"search ledger schema differs: {name}"
            )
    return {
        "root": str(root),
        "done": dict(done),
        "summary": dict(summary),
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
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    generations: int = DEFAULT_GENERATIONS,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_trials: int = DEFAULT_MAX_TRIALS,
) -> dict[str, Any]:
    """Validate bound inputs, execute the search, and publish atomically."""

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
    examples, coverage = r7._build_examples(inputs, seed=seed)
    result = search_examples(
        examples,
        seed=seed,
        generations=generations,
        beam_width=beam_width,
        max_trials=max_trials,
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
    r7._assert_inputs_stable(inputs)
    _publish(
        output_dir.expanduser().resolve(strict=False),
        trials=result["trials"],
        failures=result["failure_memory"],
        summary=summary,
    )
    r7._assert_inputs_stable(inputs)
    return {
        **result,
        "summary": summary,
        "output_dir": str(output_dir.expanduser().resolve(strict=True)),
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
    )
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
        output_dir=args.output_dir,
        seed=args.seed,
        generations=args.generations,
        beam_width=args.beam_width,
        max_trials=args.max_trials,
    )
    print(
        "[motive-r9-representation-search] "
        f"trials={len(result['trials'])} "
        f"decision={result['summary']['decision']['status']} "
        f"output={result['output_dir']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

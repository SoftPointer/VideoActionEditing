"""Paired component-bootstrap analysis of the R7 candidate motion screen.

This module is deliberately downstream of one immutable
``r7_candidate_temporal_screen`` output.  It does not reopen feature caches,
fit a representation, select a threshold, or authorize any training or
generation.  The fixed train-positive retrieval bank is treated as
conditional; uncertainty is estimated only over held-out visual components.

Every comparison is paired on the same query rows.  Within an evaluation
scope, every modality and contrast also shares the exact same component
bootstrap draws.  Invalid retrieval queries retain the conservative policy
of the upstream screen and count as misses.  Coverage for each arm and for
the shared-valid subset is reported separately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from . import r7_artifact_permissions as artifact_permissions
from . import r7_candidate_temporal_screen as screen


ANALYSIS_SCHEMA = "motive-r7-candidate-temporal-analysis-v1"
COMPARISON_SCHEMA = "motive-r7-candidate-temporal-analysis-comparison-v1"
DONE_SCHEMA = "motive-r7-candidate-temporal-analysis-done-v1"
BOOTSTRAP_PROTOCOL = (
    "heldout-family-stratified-visual-component-cluster-paired-"
    "shared-draws-v2"
)

COMPARISONS_NAME = "comparisons.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
OUTPUT_NAMES = (COMPARISONS_NAME, SUMMARY_NAME, DONE_NAME)
PAYLOAD_NAMES = (COMPARISONS_NAME, SUMMARY_NAME)

SCOPES = ("overall", "validation", "test")
METRICS = (
    "micro_r_at_1",
    "micro_r_at_5",
    "macro_family_r_at_1",
    "macro_family_r_at_5",
)
DEFAULT_BOOTSTRAP_REPETITIONS = 20_000
DEFAULT_SEED = 260108836
CONFIDENCE = 0.95
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-15

EXTRA_SAFETY_FIELDS = (
    "statistical_significance_asserted",
    "representation_learning_established",
    "actor_disentanglement_established",
    "next_gpu_experiment_authorized",
)
SAFETY_FIELDS = tuple(screen.SAFETY_FIELDS) + EXTRA_SAFETY_FIELDS


@dataclass(frozen=True)
class _Contrast:
    name: str
    treatment: str
    control: str
    role: str
    interpretation: str


CONTRASTS = (
    _Contrast(
        "target_temporal_vs_endpoint",
        screen.TARGET_TEMPORAL,
        screen.TARGET_ENDPOINT,
        "primary_temporal_content",
        "ordered target motion versus last-minus-first endpoint",
    ),
    _Contrast(
        "target_temporal_vs_orderless",
        screen.TARGET_TEMPORAL,
        screen.ORDERLESS_TEMPORAL,
        "primary_temporal_content",
        "ordered target motion versus orderless temporal statistics",
    ),
    _Contrast(
        "target_temporal_vs_transition_shuffle",
        screen.TARGET_TEMPORAL,
        screen.SHUFFLED_QUERY,
        "primary_query_only_control",
        "clean query versus transition-block shuffle on one clean bank",
    ),
    _Contrast(
        "target_temporal_vs_physical_reverse",
        screen.TARGET_TEMPORAL,
        screen.REVERSED_QUERY,
        "primary_query_only_control",
        "clean query versus physical time reversal on one clean bank",
    ),
    _Contrast(
        "target_temporal_vs_pooled_dino",
        screen.TARGET_TEMPORAL,
        screen.POOLED_DINO,
        "shortcut_diagnostic",
        "motion statistics versus pooled target appearance",
    ),
    _Contrast(
        "target_temporal_vs_camera",
        screen.TARGET_TEMPORAL,
        screen.CAMERA_NUISANCE,
        "shortcut_diagnostic",
        "camera-compensated motion statistics versus camera trajectory",
    ),
    _Contrast(
        "delta_temporal_vs_target_temporal",
        screen.DELTA_TEMPORAL,
        screen.TARGET_TEMPORAL,
        "source_conditioning_diagnostic",
        "source-to-target temporal delta versus target-only temporal motion",
    ),
    _Contrast(
        "delta_temporal_vs_endpoint",
        screen.DELTA_TEMPORAL,
        screen.TARGET_ENDPOINT,
        "secondary_delta_baseline",
        "source-to-target temporal delta versus target endpoint",
    ),
    _Contrast(
        "delta_temporal_vs_orderless",
        screen.DELTA_TEMPORAL,
        screen.ORDERLESS_TEMPORAL,
        "secondary_delta_baseline",
        "source-to-target temporal delta versus orderless target motion",
    ),
    _Contrast(
        "delta_temporal_vs_pooled_dino",
        screen.DELTA_TEMPORAL,
        screen.POOLED_DINO,
        "secondary_shortcut_diagnostic",
        "source-to-target temporal delta versus target appearance",
    ),
    _Contrast(
        "delta_temporal_vs_camera",
        screen.DELTA_TEMPORAL,
        screen.CAMERA_NUISANCE,
        "secondary_shortcut_diagnostic",
        "source-to-target temporal delta versus camera trajectory",
    ),
)


class CandidateTemporalAnalysisError(ValueError):
    """The input, statistical contract, or immutable output is invalid."""


@dataclass(frozen=True)
class _Input:
    root: Path
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    done: Mapping[str, Any]
    binding: Mapping[str, Any]
    identities: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class _BootstrapPlan:
    scope: str
    repetitions: int
    seed: int
    components: tuple[str, ...]
    family_component_counts: tuple[tuple[str, int], ...]
    row_weights: np.ndarray
    draw_digest: str


def _safety_flags() -> dict[str, bool]:
    return {field: False for field in SAFETY_FIELDS}


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


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or not _SHA256_RE.fullmatch(value):
        raise CandidateTemporalAnalysisError(
            f"{context} is not a lowercase SHA-256"
        )
    return value


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateTemporalAnalysisError(
            f"invalid JSON object: {path}"
        ) from error
    if not isinstance(value, dict):
        raise CandidateTemporalAnalysisError(
            f"JSON root is not an object: {path}"
        )
    return value


def _load_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise CandidateTemporalAnalysisError(
                    f"{path}:{line_number} is blank or lacks LF"
                )
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CandidateTemporalAnalysisError(
                    f"{path}:{line_number} is invalid JSON"
                ) from error
            if (
                not isinstance(value, dict)
                or line
                != _canonical_json(value) + "\n"
            ):
                raise CandidateTemporalAnalysisError(
                    f"{path}:{line_number} is not canonical JSONL"
                )
            rows.append(value)
    return rows


def _status_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_nlink),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _stat_identity(path: Path) -> tuple[int, ...]:
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode):
        raise CandidateTemporalAnalysisError(
            f"input contains a symlink: {path}"
        )
    if not (stat.S_ISDIR(status.st_mode) or stat.S_ISREG(status.st_mode)):
        raise CandidateTemporalAnalysisError(
            f"input is not a directory/regular file: {path}"
        )
    if stat.S_ISREG(status.st_mode) and status.st_nlink != 1:
        raise CandidateTemporalAnalysisError(
            f"input is hard-linked: {path}"
        )
    return _status_identity(status)


def _capture_identities(root: Path) -> dict[str, tuple[int, ...]]:
    return {
        ".": _stat_identity(root),
        screen.ROWS_NAME: _stat_identity(root / screen.ROWS_NAME),
        screen.SUMMARY_NAME: _stat_identity(root / screen.SUMMARY_NAME),
        screen.DONE_NAME: _stat_identity(root / screen.DONE_NAME),
    }


def _assert_input_stable(value: _Input) -> None:
    actual = _capture_identities(value.root)
    if actual != dict(value.identities):
        raise CandidateTemporalAnalysisError(
            "screen input identities changed during analysis"
        )


def _implementation_provenance() -> dict[str, Any]:
    path = Path(__file__).resolve(strict=True)
    return {
        "module": "motive.r7_candidate_temporal_analysis",
        "module_sha256": _file_digest(path),
        "numpy_version": np.__version__,
    }


def _load_screen(
    screen_dir: Path,
    *,
    expected_screen_done_sha256: str,
) -> _Input:
    expected = _require_sha256(
        expected_screen_done_sha256,
        context="expected screen done SHA",
    )
    unresolved = screen_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    identities = _capture_identities(root)
    if _file_digest(root / screen.DONE_NAME) != expected:
        raise CandidateTemporalAnalysisError(
            "screen external done SHA differs"
        )
    verified = screen._validate_candidate_temporal_screen_envelope(root)
    rows = tuple(dict(row) for row in verified["rows"])
    summary = dict(verified["summary"])
    done = dict(verified["done"])
    binding = {
        "directory": str(root),
        "done_sha256": expected,
        "artifact_digest": done["artifact_digest"],
        "contract_sha256": summary["contract_sha256"],
        "rows": len(rows),
        "rows_sha256": summary["output"]["rows_sha256"],
        "screen_schema_version": summary["schema_version"],
        "formal_status": summary["formal_status"],
    }
    value = _Input(
        root=root,
        rows=rows,
        summary=summary,
        done=done,
        binding=binding,
        identities=identities,
    )
    _assert_input_stable(value)
    return value


def _eligible_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> list[Mapping[str, Any]]:
    if scope not in SCOPES:
        raise CandidateTemporalAnalysisError(
            f"unsupported analysis scope: {scope}"
        )
    selected = [
        row
        for row in rows
        if (
            row["label_class"] == "positive"
            and row["eligible_positive_query"] is True
            and (scope == "overall" or row["split"] == scope)
        )
    ]
    return sorted(selected, key=lambda row: str(row["iid"]))


def _validate_component_topology(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    component_splits: dict[str, set[str]] = {}
    eligible_component_families: dict[str, set[str]] = {}
    for row in rows:
        component = str(row["component_id"])
        component_splits.setdefault(component, set()).add(
            str(row["split"])
        )
        if (
            row["label_class"] == "positive"
            and row["eligible_positive_query"] is True
        ):
            eligible_component_families.setdefault(
                component,
                set(),
            ).add(str(row["family"]))
    crossing = sorted(
        component
        for component, splits in component_splits.items()
        if len(splits) != 1
    )
    if crossing:
        raise CandidateTemporalAnalysisError(
            "one or more held-out visual components cross validation/test: "
            + ", ".join(crossing[:5])
        )
    mixed_family = sorted(
        component
        for component, families in eligible_component_families.items()
        if len(families) != 1
    )
    if mixed_family:
        raise CandidateTemporalAnalysisError(
            "one or more eligible visual components cross action families: "
            + ", ".join(mixed_family[:5])
        )


def _stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(
        f"{ANALYSIS_SCHEMA}\0{base_seed}\0{label}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _bootstrap_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    repetitions: int,
    seed: int,
) -> _BootstrapPlan | None:
    if not rows:
        return None
    components = tuple(
        sorted({str(row["component_id"]) for row in rows})
    )
    component_index = {
        component: index for index, component in enumerate(components)
    }
    component_families: dict[str, set[str]] = {
        component: set() for component in components
    }
    for row in rows:
        component_families[str(row["component_id"])].add(
            str(row["family"])
        )
    mixed = sorted(
        component
        for component, families in component_families.items()
        if len(families) != 1
    )
    if mixed:
        raise CandidateTemporalAnalysisError(
            "component bootstrap requires one action family per component: "
            + ", ".join(mixed[:5])
        )
    family_components: dict[str, list[int]] = {}
    for component in components:
        family = next(iter(component_families[component]))
        family_components.setdefault(family, []).append(
            component_index[component]
        )
    row_component = np.asarray(
        [component_index[str(row["component_id"])] for row in rows],
        dtype=np.int64,
    )
    draw_seed = _stable_seed(seed, f"bootstrap:{scope}")
    rng = np.random.default_rng(draw_seed)
    multiplicities = np.zeros(
        (repetitions, len(components)),
        dtype=np.int64,
    )
    family_component_counts: list[tuple[str, int]] = []
    for family in sorted(family_components):
        indices = np.asarray(
            family_components[family],
            dtype=np.int64,
        )
        count = len(indices)
        family_component_counts.append((family, count))
        draws = rng.multinomial(
            count,
            np.full(count, 1.0 / count, dtype=np.float64),
            size=repetitions,
        )
        multiplicities[:, indices] = draws
    row_weights = np.asarray(
        multiplicities[:, row_component],
        dtype=np.float64,
        order="C",
    )
    digest = hashlib.sha256()
    digest.update(
        np.asarray(multiplicities.shape, dtype="<i8").tobytes()
    )
    digest.update(
        _canonical_json(family_component_counts).encode("utf-8")
    )
    digest.update(
        np.asarray(multiplicities, dtype="<i4", order="C").tobytes()
    )
    return _BootstrapPlan(
        scope=scope,
        repetitions=repetitions,
        seed=draw_seed,
        components=components,
        family_component_counts=tuple(family_component_counts),
        row_weights=row_weights,
        draw_digest=digest.hexdigest(),
    )


def _hit_arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if modality not in screen.MODALITIES:
        raise CandidateTemporalAnalysisError(
            f"unknown screen modality: {modality}"
        )
    valid = np.asarray(
        [
            row["modalities"][modality]["valid_for_retrieval"] is True
            for row in rows
        ],
        dtype=bool,
    )
    hit1 = np.asarray(
        [
            row["modalities"][modality]["correct_at_1"] is True
            for row in rows
        ],
        dtype=np.float64,
    )
    hit5 = np.asarray(
        [
            row["modalities"][modality]["correct_at_5"] is True
            for row in rows
        ],
        dtype=np.float64,
    )
    return valid, hit1, hit5


def _point_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    modality: str,
) -> dict[str, Any]:
    if not rows:
        return {
            "queries": 0,
            "components": 0,
            "families": 0,
            "valid_queries": 0,
            "invalid_queries": 0,
            "valid_fraction": None,
            "metrics": {metric: None for metric in METRICS},
        }
    valid, hit1, hit5 = _hit_arrays(rows, modality=modality)
    families = np.asarray(
        [str(row["family"]) for row in rows], dtype=object
    )
    per_family_1: list[float] = []
    per_family_5: list[float] = []
    for family in sorted(set(families.tolist())):
        mask = families == family
        per_family_1.append(float(np.mean(hit1[mask])))
        per_family_5.append(float(np.mean(hit5[mask])))
    return {
        "queries": len(rows),
        "components": len(
            {str(row["component_id"]) for row in rows}
        ),
        "families": len(per_family_1),
        "valid_queries": int(np.sum(valid)),
        "invalid_queries": int(len(valid) - np.sum(valid)),
        "valid_fraction": float(np.mean(valid)),
        "metrics": {
            "micro_r_at_1": float(np.mean(hit1)),
            "micro_r_at_5": float(np.mean(hit5)),
            "macro_family_r_at_1": float(np.mean(per_family_1)),
            "macro_family_r_at_5": float(np.mean(per_family_5)),
        },
    }


def _bootstrap_metric_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    modality: str,
    plan: _BootstrapPlan,
) -> dict[str, np.ndarray]:
    if len(rows) != plan.row_weights.shape[1]:
        raise CandidateTemporalAnalysisError(
            "bootstrap row weights differ from the analysis cohort"
        )
    _valid, hit1, hit5 = _hit_arrays(rows, modality=modality)
    weights = plan.row_weights
    denominator = np.sum(weights, axis=1)
    if bool((denominator <= 0.0).any()):
        raise CandidateTemporalAnalysisError(
            "component bootstrap produced an empty draw"
        )
    micro1 = weights @ hit1 / denominator
    micro5 = weights @ hit5 / denominator
    families = np.asarray(
        [str(row["family"]) for row in rows], dtype=object
    )
    family1: list[np.ndarray] = []
    family5: list[np.ndarray] = []
    for family in sorted(set(families.tolist())):
        mask = families == family
        family_denominator = np.sum(weights[:, mask], axis=1)
        if bool((family_denominator <= 0.0).any()):
            raise CandidateTemporalAnalysisError(
                "family-stratified bootstrap omitted an observed family"
            )
        numerator1 = weights[:, mask] @ hit1[mask]
        numerator5 = weights[:, mask] @ hit5[mask]
        rate1 = numerator1 / family_denominator
        rate5 = numerator5 / family_denominator
        family1.append(rate1)
        family5.append(rate5)
    macro1 = np.mean(np.stack(family1, axis=1), axis=1)
    macro5 = np.mean(np.stack(family5, axis=1), axis=1)
    result = {
        "micro_r_at_1": micro1,
        "micro_r_at_5": micro5,
        "macro_family_r_at_1": macro1,
        "macro_family_r_at_5": macro5,
    }
    if any(
        values.shape != (plan.repetitions,)
        or not np.isfinite(values).all()
        for values in result.values()
    ):
        raise CandidateTemporalAnalysisError(
            "bootstrap metric series is non-finite or malformed"
        )
    return result


def _difference_interval(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    point: float,
) -> dict[str, Any]:
    difference = np.asarray(treatment, dtype=np.float64) - np.asarray(
        control, dtype=np.float64
    )
    if not len(difference) or not np.isfinite(difference).all():
        raise CandidateTemporalAnalysisError(
            "paired bootstrap difference is empty or non-finite"
        )
    alpha = (1.0 - CONFIDENCE) / 2.0
    return {
        "point": float(point),
        "lower": float(np.quantile(difference, alpha)),
        "upper": float(np.quantile(difference, 1.0 - alpha)),
        "bootstrap_mean": float(np.mean(difference)),
        "tail_fraction_le_zero_diagnostic": float(
            (1 + np.count_nonzero(difference <= 0.0))
            / (len(difference) + 1)
        ),
    }


def _scope_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    treatment: str,
    control: str,
    plan: _BootstrapPlan | None,
    series: Mapping[str, Mapping[str, np.ndarray]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    treatment_point = _point_metrics(rows, modality=treatment)
    control_point = _point_metrics(rows, modality=control)
    paired_valid = sum(
        row["modalities"][treatment]["valid_for_retrieval"] is True
        and row["modalities"][control]["valid_for_retrieval"] is True
        for row in rows
    )
    if not rows:
        intervals: dict[str, Any] = {
            metric: None for metric in METRICS
        }
        bootstrap = {
            "protocol": BOOTSTRAP_PROTOCOL,
            "confidence": CONFIDENCE,
            "repetitions": repetitions,
            "seed": _stable_seed(seed, f"bootstrap:{scope}"),
            "components": 0,
            "family_component_counts": {},
            "draw_digest": None,
            "status": "UNAVAILABLE_NO_QUERIES",
            "intervals": intervals,
        }
    else:
        if plan is None:
            raise RuntimeError("non-empty scope lacks a bootstrap plan")
        intervals = {}
        for metric in METRICS:
            treatment_value = treatment_point["metrics"][metric]
            control_value = control_point["metrics"][metric]
            if treatment_value is None or control_value is None:
                raise RuntimeError("non-empty point metric is unavailable")
            intervals[metric] = _difference_interval(
                series[treatment][metric],
                series[control][metric],
                point=float(treatment_value) - float(control_value),
            )
        bootstrap = {
            "protocol": BOOTSTRAP_PROTOCOL,
            "confidence": CONFIDENCE,
            "repetitions": plan.repetitions,
            "seed": plan.seed,
            "components": len(plan.components),
            "family_component_counts": dict(
                plan.family_component_counts
            ),
            "draw_digest": plan.draw_digest,
            "status": "DESCRIPTIVE_CONDITIONAL_ON_FIXED_BANK",
            "intervals": intervals,
        }
    return {
        "scope": scope,
        "query_policy":
            "eligible_pseudo-positive; invalid retrieval counts as miss",
        "queries": len(rows),
        "components": len(
            {str(row["component_id"]) for row in rows}
        ),
        "families": len({str(row["family"]) for row in rows}),
        "treatment": {
            "modality": treatment,
            **treatment_point,
        },
        "control": {
            "modality": control,
            **control_point,
        },
        "shared_valid_queries": int(paired_valid),
        "shared_valid_fraction": (
            float(paired_valid / len(rows)) if rows else None
        ),
        "paired_component_bootstrap": bootstrap,
    }


def _direction(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    number = float(value)
    if not math.isfinite(number):
        raise CandidateTemporalAnalysisError(
            "direction received a non-finite value"
        )
    if number > _EPS:
        return "POSITIVE"
    if number < -_EPS:
        return "NEGATIVE"
    return "ZERO"


def _val_test_direction(
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    statuses: list[str] = []
    for metric in METRICS:
        validation_record = validation["paired_component_bootstrap"][
            "intervals"
        ][metric]
        test_record = test["paired_component_bootstrap"]["intervals"][
            metric
        ]
        validation_value = (
            None
            if validation_record is None
            else validation_record["point"]
        )
        test_value = (
            None if test_record is None else test_record["point"]
        )
        left = _direction(validation_value)
        right = _direction(test_value)
        if left == right == "POSITIVE":
            consistency = "SAME_POSITIVE"
        elif left == right == "NEGATIVE":
            consistency = "SAME_NEGATIVE"
        elif "UNAVAILABLE" in {left, right}:
            consistency = "UNAVAILABLE"
        elif left == "ZERO" or right == "ZERO":
            consistency = "INCLUDES_ZERO"
        else:
            consistency = "DISAGREE"
        statuses.append(consistency)
        output[metric] = {
            "validation_point": validation_value,
            "validation_direction": left,
            "test_point": test_value,
            "test_direction": right,
            "consistency": consistency,
        }
    if all(value == "SAME_POSITIVE" for value in statuses):
        overall = "ALL_METRICS_SAME_POSITIVE"
    elif all(value == "SAME_NEGATIVE" for value in statuses):
        overall = "ALL_METRICS_SAME_NEGATIVE"
    elif all(value == "UNAVAILABLE" for value in statuses):
        overall = "UNAVAILABLE"
    else:
        overall = "MIXED_DISAGREE_OR_ZERO"
    return {
        "metrics": output,
        "overall": overall,
        "formal_evidence": False,
    }


def _coverage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    eligible = _eligible_rows(rows, scope="overall")
    return {
        "screen_rows": len(rows),
        "by_split": {
            split: sum(row["split"] == split for row in rows)
            for split in screen.EVAL_SPLITS
        },
        "by_label": {
            label: sum(row["label_class"] == label for row in rows)
            for label in ("positive", "negative")
        },
        "heldout_components": len(
            {str(row["component_id"]) for row in rows}
        ),
        "eligible_positive_queries": len(eligible),
        "eligible_positive_components": len(
            {str(row["component_id"]) for row in eligible}
        ),
        "eligible_positive_families": len(
            {str(row["family"]) for row in eligible}
        ),
        "eligible_positive_by_split": {
            split: sum(row["split"] == split for row in eligible)
            for split in screen.EVAL_SPLITS
        },
        "eligible_positive_components_by_split": {
            split: len(
                {
                    str(row["component_id"])
                    for row in eligible
                    if row["split"] == split
                }
            )
            for split in screen.EVAL_SPLITS
        },
        "modality_validity_on_eligible_positive": {
            modality: {
                "valid_queries": sum(
                    row["modalities"][modality][
                        "valid_for_retrieval"
                    ]
                    is True
                    for row in eligible
                ),
                "valid_fraction": (
                    float(
                        np.mean(
                            [
                                row["modalities"][modality][
                                    "valid_for_retrieval"
                                ]
                                is True
                                for row in eligible
                            ]
                        )
                    )
                    if eligible
                    else None
                ),
            }
            for modality in screen.MODALITIES
        },
        "modalities_share_exact_query_cohort": True,
        "component_cross_split_count": 0,
    }


def _dino_split_confounded(summary: Mapping[str, Any]) -> bool:
    try:
        value = summary["contract"]["retrieval"]["split_bias"][
            "relative_motion_vs_dino_diagnostic_is_split_confounded"
        ]
    except (KeyError, TypeError) as error:
        raise CandidateTemporalAnalysisError(
            "screen contract lacks the DINO split-confound binding"
        ) from error
    if type(value) is not bool:
        raise CandidateTemporalAnalysisError(
            "DINO split-confound binding is not boolean"
        )
    return value


def _contract(
    *,
    value: _Input,
    bootstrap_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "input_screen": dict(value.binding),
        "implementation": _implementation_provenance(),
        "config": {
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "confidence": CONFIDENCE,
            "scopes": list(SCOPES),
            "metrics": list(METRICS),
        },
        "contrasts": [
            {
                "name": item.name,
                "treatment": item.treatment,
                "control": item.control,
                "role": item.role,
                "interpretation": item.interpretation,
            }
            for item in CONTRASTS
        ],
        "estimand": {
            "query_population":
                "eligible held-out pseudo-positive screen rows",
            "invalid_query_policy": "count_as_retrieval_miss",
            "point_weighting":
                "equal query micro and equal-family macro",
            "resampling_unit": "heldout_visual_component_id",
            "resampling": (
                "components_with_replacement_within_each_observed_"
                "action_family"
            ),
            "family_stratification": (
                "fixed observed family registry and observed component "
                "count per family in every draw"
            ),
            "component_family_requirement":
                "each eligible component belongs to exactly one family",
            "pairing":
                "same query rows and same component multiplicities",
            "shared_draws":
                "all modalities and contrasts within an evaluation scope",
            "train_reference_bank": "fixed_and_not_resampled",
            "confidence_interpretation":
                "conditional_on_current_train_bank_and_pseudo_labels",
            "split_comparison": "direction_only_not_a_formal_replication",
        },
        "semantics": {
            "labels_are_pseudo": True,
            "split_is_provisional_diagnostic_only": True,
            "learned_representation_evaluated": False,
            "human_review_substitute": False,
            "formal_gate": False,
            **_safety_flags(),
        },
    }


def _derive(
    *,
    screen_dir: Path,
    expected_screen_done_sha256: str,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[dict[str, bytes], _Input]:
    if (
        isinstance(bootstrap_repetitions, bool)
        or not isinstance(bootstrap_repetitions, int)
        or bootstrap_repetitions < 1
    ):
        raise CandidateTemporalAnalysisError(
            "bootstrap_repetitions must be a positive integer"
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise CandidateTemporalAnalysisError(
            "seed must be an integer in [0,2**32)"
        )
    value = _load_screen(
        screen_dir,
        expected_screen_done_sha256=expected_screen_done_sha256,
    )
    _validate_component_topology(value.rows)
    scope_rows = {
        scope: _eligible_rows(value.rows, scope=scope)
        for scope in SCOPES
    }
    plans = {
        scope: _bootstrap_plan(
            rows,
            scope=scope,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        for scope, rows in scope_rows.items()
    }
    series: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scope, rows in scope_rows.items():
        plan = plans[scope]
        series[scope] = {}
        if plan is not None:
            series[scope] = {
                modality: _bootstrap_metric_series(
                    rows,
                    modality=modality,
                    plan=plan,
                )
                for modality in screen.MODALITIES
            }

    dino_confounded = _dino_split_confounded(value.summary)
    comparisons: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        scopes = {
            scope: _scope_record(
                scope_rows[scope],
                scope=scope,
                treatment=contrast.treatment,
                control=contrast.control,
                plan=plans[scope],
                series=series[scope],
                repetitions=bootstrap_repetitions,
                seed=seed,
            )
            for scope in SCOPES
        }
        comparisons.append(
            {
                "schema_version": COMPARISON_SCHEMA,
                "name": contrast.name,
                "treatment": contrast.treatment,
                "control": contrast.control,
                "role": contrast.role,
                "interpretation": contrast.interpretation,
                "scopes": scopes,
                "validation_test_direction":
                    _val_test_direction(
                        scopes["validation"], scopes["test"]
                    ),
                "dino_split_confounded": (
                    dino_confounded
                    and screen.POOLED_DINO
                    in {contrast.treatment, contrast.control}
                ),
                "formal_status": "INSUFFICIENT",
                **_safety_flags(),
            }
        )

    contract = _contract(
        value=value,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    contract_sha = _object_digest(contract)
    comparison_bytes = _jsonl_bytes(comparisons)
    direction_counts: dict[str, int] = {}
    for row in comparisons:
        status = row["validation_test_direction"]["overall"]
        direction_counts[status] = direction_counts.get(status, 0) + 1
    headline = {
        row["name"]: {
            "overall_macro_family_r_at_1_gain": row["scopes"][
                "overall"
            ]["paired_component_bootstrap"]["intervals"][
                "macro_family_r_at_1"
            ],
            "validation_test_macro_family_r_at_1": row[
                "validation_test_direction"
            ]["metrics"]["macro_family_r_at_1"],
            "dino_split_confounded": row["dino_split_confounded"],
        }
        for row in comparisons
    }
    summary: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "diagnostic_scope":
            "paired-component-bootstrap-over-frozen-screen-rows",
        "contract": contract,
        "contract_sha256": contract_sha,
        "coverage": _coverage(value.rows),
        "headline_descriptive": headline,
        "validation_test_direction_counts": dict(
            sorted(direction_counts.items())
        ),
        "evidence_limitations": {
            "pseudo_labels": True,
            "provisional_split": True,
            "fixed_train_bank_not_resampled": True,
            "train_label_permutation_null_available": False,
            "human_labels_available": False,
            "learned_representation_evidence": False,
            "actor_localization_or_disentanglement_established": False,
            "dino_relative_comparisons_split_confounded":
                dino_confounded,
            "sampling_weighted_negative_auroc_not_reanalysed": True,
        },
        "decision": {
            "formal_status": "INSUFFICIENT",
            "diagnostic_completed": True,
            "reason": (
                "component-bootstrap intervals over pseudo-labelled rows "
                "and a fixed train bank cannot authorize learning, "
                "generation, editing, or production"
            ),
            "next_step_recommendation":
                "human_review_before_any_representation_learning",
            **_safety_flags(),
        },
        "formal_status": "INSUFFICIENT",
        **_safety_flags(),
        "output": {
            "comparisons_name": COMPARISONS_NAME,
            "comparisons": len(comparisons),
            "comparisons_sha256": hashlib.sha256(
                comparison_bytes
            ).hexdigest(),
            "comparison_order": [item.name for item in CONTRASTS],
            "comparison_encoding": "canonical_json_utf8_lf",
        },
    }
    summary_bytes = _pretty_json_bytes(summary)
    payload_files = {
        COMPARISONS_NAME: {
            "sha256": hashlib.sha256(comparison_bytes).hexdigest(),
            "bytes": len(comparison_bytes),
            "mode_octal": "0444",
        },
        SUMMARY_NAME: {
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "bytes": len(summary_bytes),
            "mode_octal": "0444",
        },
    }
    done_core = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "diagnostic_completed": True,
        "formal_status": "INSUFFICIENT",
        "contract_sha256": contract_sha,
        "comparisons": len(comparisons),
        "input_screen_done_sha256":
            value.binding["done_sha256"],
        "payload_files": payload_files,
        "artifact_closure": list(OUTPUT_NAMES),
        "permission_contract": artifact_permissions.permission_contract(),
        **_safety_flags(),
    }
    done = {
        **done_core,
        "artifact_digest": _object_digest(payload_files),
    }
    _assert_input_stable(value)
    return {
        COMPARISONS_NAME: comparison_bytes,
        SUMMARY_NAME: summary_bytes,
        DONE_NAME: _pretty_json_bytes(done),
    }, value


def _validate_safety(value: Mapping[str, Any], *, context: str) -> None:
    if any(value.get(field) is not False for field in SAFETY_FIELDS):
        raise CandidateTemporalAnalysisError(
            f"{context} asserts a forbidden safety field"
        )


def _validate_analysis_envelope(
    output_dir: Path,
    *,
    expected_contract: Mapping[str, Any] | None = None,
    expected_identities: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise FileNotFoundError(unresolved)
    root = unresolved.resolve(strict=True)
    if {entry.name for entry in root.iterdir()} != set(OUTPUT_NAMES):
        raise CandidateTemporalAnalysisError(
            "analysis output artifact closure differs"
        )
    identities = _capture_identities_for_names(root, OUTPUT_NAMES)
    if (
        expected_identities is not None
        and identities != dict(expected_identities)
    ):
        raise CandidateTemporalAnalysisError(
            "analysis output identities differ from the bound operation"
        )
    artifact_permissions.assert_sealed_tree(root)
    comparisons_path = root / COMPARISONS_NAME
    summary_path = root / SUMMARY_NAME
    done_path = root / DONE_NAME
    comparisons = _load_canonical_jsonl(comparisons_path)
    summary = _load_object(summary_path)
    done = _load_object(done_path)
    if summary_path.read_bytes() != _pretty_json_bytes(summary):
        raise CandidateTemporalAnalysisError(
            "analysis summary is not canonical pretty JSON"
        )
    if done_path.read_bytes() != _pretty_json_bytes(done):
        raise CandidateTemporalAnalysisError(
            "analysis done is not canonical pretty JSON"
        )
    expected_names = [item.name for item in CONTRASTS]
    if [row.get("name") for row in comparisons] != expected_names:
        raise CandidateTemporalAnalysisError(
            "analysis comparison registry/order differs"
        )
    for row, definition in zip(comparisons, CONTRASTS):
        if (
            row.get("schema_version") != COMPARISON_SCHEMA
            or row.get("treatment") != definition.treatment
            or row.get("control") != definition.control
            or row.get("role") != definition.role
            or row.get("formal_status") != "INSUFFICIENT"
            or not isinstance(row.get("scopes"), Mapping)
            or set(row["scopes"]) != set(SCOPES)
        ):
            raise CandidateTemporalAnalysisError(
                f"analysis comparison differs: {definition.name}"
            )
        _validate_safety(row, context=f"comparison {definition.name}")
        direction = row.get("validation_test_direction")
        if (
            not isinstance(direction, Mapping)
            or direction.get("formal_evidence") is not False
            or set(direction.get("metrics", {})) != set(METRICS)
        ):
            raise CandidateTemporalAnalysisError(
                f"comparison direction differs: {definition.name}"
            )
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise CandidateTemporalAnalysisError(
            "analysis summary contract is missing"
        )
    contract_sha = _object_digest(dict(contract))
    if (
        summary.get("schema_version") != ANALYSIS_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("formal_status") != "INSUFFICIENT"
        or summary.get("contract_sha256") != contract_sha
        or (
            expected_contract is not None
            and dict(expected_contract) != dict(contract)
        )
    ):
        raise CandidateTemporalAnalysisError(
            "analysis summary contract/status differs"
        )
    _validate_safety(summary, context="analysis summary")
    semantics = contract.get("semantics")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("labels_are_pseudo") is not True
        or semantics.get("split_is_provisional_diagnostic_only")
        is not True
        or semantics.get("formal_gate") is not False
    ):
        raise CandidateTemporalAnalysisError(
            "analysis contract semantics differ"
        )
    _validate_safety(semantics, context="analysis contract semantics")
    decision = summary.get("decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("formal_status") != "INSUFFICIENT"
        or decision.get("diagnostic_completed") is not True
    ):
        raise CandidateTemporalAnalysisError(
            "analysis decision differs"
        )
    _validate_safety(decision, context="analysis decision")
    comparison_sha = _file_digest(comparisons_path)
    output = summary.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("comparisons_name") != COMPARISONS_NAME
        or output.get("comparisons") != len(comparisons)
        or output.get("comparisons_sha256") != comparison_sha
        or output.get("comparison_order") != expected_names
        or output.get("comparison_encoding")
        != "canonical_json_utf8_lf"
    ):
        raise CandidateTemporalAnalysisError(
            "analysis summary output commitment differs"
        )
    payload_files = {
        name: {
            "sha256": _file_digest(root / name),
            "bytes": int((root / name).stat().st_size),
            "mode_octal": "0444",
        }
        for name in PAYLOAD_NAMES
    }
    expected_done_core = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "diagnostic_completed": True,
        "formal_status": "INSUFFICIENT",
        "contract_sha256": contract_sha,
        "comparisons": len(comparisons),
        "input_screen_done_sha256": contract["input_screen"][
            "done_sha256"
        ],
        "payload_files": payload_files,
        "artifact_closure": list(OUTPUT_NAMES),
        "permission_contract": artifact_permissions.permission_contract(),
        **_safety_flags(),
    }
    expected_done = {
        **expected_done_core,
        "artifact_digest": _object_digest(payload_files),
    }
    if done != expected_done:
        raise CandidateTemporalAnalysisError(
            "analysis done/hash/safety chain differs"
        )
    if _capture_identities_for_names(root, OUTPUT_NAMES) != identities:
        raise CandidateTemporalAnalysisError(
            "analysis output identities changed during validation"
        )
    return {
        "directory": root,
        "comparisons": comparisons,
        "summary": summary,
        "done": done,
    }


def _capture_identities_for_names(
    root: Path,
    names: Sequence[str],
) -> dict[str, tuple[int, ...]]:
    return {
        ".": _stat_identity(root),
        **{
            name: _stat_identity(root / name)
            for name in names
        },
    }


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise CandidateTemporalAnalysisError(
            "O_DIRECTORY and O_NOFOLLOW are required for publication"
        )
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _assert_bound_directory(
    *,
    parent_fd: int,
    name: str,
    directory_fd: int,
    expected_mode: int,
) -> None:
    bound = os.fstat(directory_fd)
    linked = os.stat(
        name,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(bound.st_mode)
        or not stat.S_ISDIR(linked.st_mode)
        or (bound.st_dev, bound.st_ino) != (linked.st_dev, linked.st_ino)
        or stat.S_IMODE(bound.st_mode) != expected_mode
        or stat.S_IMODE(linked.st_mode) != expected_mode
    ):
        raise CandidateTemporalAnalysisError(
            "claimed analysis output path identity changed"
        )


def _write_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> tuple[int, tuple[int, ...]]:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\0" in name
    ):
        raise CandidateTemporalAnalysisError(
            "invalid analysis payload name"
        )
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "pread"):
        raise CandidateTemporalAnalysisError(
            "O_NOFOLLOW and pread are required for payload publication"
        )
    flags |= os.O_NOFOLLOW
    descriptor = os.open(
        name,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise CandidateTemporalAnalysisError(
                f"new analysis payload inode differs: {name}"
            )
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
        os.fchmod(descriptor, artifact_permissions.FILE_MODE)
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode)
            != artifact_permissions.FILE_MODE
            or final.st_size != len(payload)
            or os.pread(descriptor, len(payload) + 1, 0) != payload
        ):
            raise CandidateTemporalAnalysisError(
                f"sealed analysis payload inode differs: {name}"
            )
        identity = _status_identity(final)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _assert_bound_payloads(
    *,
    directory_fd: int,
    descriptors: Mapping[str, int],
    sealed_identities: Mapping[str, tuple[int, ...]],
    payloads: Mapping[str, bytes],
) -> None:
    if (
        set(descriptors) != set(OUTPUT_NAMES)
        or set(sealed_identities) != set(OUTPUT_NAMES)
    ):
        raise CandidateTemporalAnalysisError(
            "bound analysis payload descriptor closure differs"
        )
    for name in OUTPUT_NAMES:
        bound = os.fstat(descriptors[name])
        linked = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        bound_identity = _status_identity(bound)
        linked_identity = _status_identity(linked)
        if (
            not stat.S_ISREG(bound.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or bound_identity != sealed_identities[name]
            or linked_identity != sealed_identities[name]
            or (bound.st_dev, bound.st_ino)
            != (linked.st_dev, linked.st_ino)
            or bound.st_nlink != 1
            or linked.st_nlink != 1
            or stat.S_IMODE(bound.st_mode)
            != artifact_permissions.FILE_MODE
            or stat.S_IMODE(linked.st_mode)
            != artifact_permissions.FILE_MODE
            or bound.st_size != len(payloads[name])
            or linked.st_size != len(payloads[name])
            or bound.st_mtime_ns != linked.st_mtime_ns
            or bound.st_ctime_ns != linked.st_ctime_ns
            or os.pread(
                descriptors[name],
                len(payloads[name]) + 1,
                0,
            ) != payloads[name]
        ):
            raise CandidateTemporalAnalysisError(
                f"claimed analysis payload path identity changed: {name}"
            )


def _publish(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
    input_value: _Input,
) -> dict[str, tuple[int, ...]]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    parent = output_dir.parent
    if parent.is_symlink() or not parent.is_dir():
        raise CandidateTemporalAnalysisError(
            "analysis output parent must be an existing real directory"
        )
    parent_fd = os.open(parent, _directory_open_flags())
    directory_fd: int | None = None
    payload_fds: dict[str, int] = {}
    payload_identities: dict[str, tuple[int, ...]] = {}
    committed = False
    try:
        try:
            os.mkdir(
                output_dir.name,
                mode=0o700,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            raise FileExistsError(output_dir) from None
        directory_fd = os.open(
            output_dir.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        _assert_bound_directory(
            parent_fd=parent_fd,
            name=output_dir.name,
            directory_fd=directory_fd,
            expected_mode=0o700,
        )
        for name in OUTPUT_NAMES:
            descriptor, identity = _write_file_at(
                directory_fd,
                name,
                payloads[name],
            )
            payload_fds[name] = descriptor
            payload_identities[name] = identity
        if set(os.listdir(directory_fd)) != set(OUTPUT_NAMES):
            raise CandidateTemporalAnalysisError(
                "claimed analysis output closure differs before sealing"
            )
        _assert_bound_payloads(
            directory_fd=directory_fd,
            descriptors=payload_fds,
            sealed_identities=payload_identities,
            payloads=payloads,
        )
        os.fsync(directory_fd)
        _assert_bound_directory(
            parent_fd=parent_fd,
            name=output_dir.name,
            directory_fd=directory_fd,
            expected_mode=0o700,
        )
        _assert_bound_payloads(
            directory_fd=directory_fd,
            descriptors=payload_fds,
            sealed_identities=payload_identities,
            payloads=payloads,
        )
        _assert_input_stable(input_value)
        os.fsync(parent_fd)
        os.fchmod(directory_fd, artifact_permissions.DIRECTORY_MODE)
        os.fsync(directory_fd)
        _assert_bound_directory(
            parent_fd=parent_fd,
            name=output_dir.name,
            directory_fd=directory_fd,
            expected_mode=artifact_permissions.DIRECTORY_MODE,
        )
        _assert_bound_payloads(
            directory_fd=directory_fd,
            descriptors=payload_fds,
            sealed_identities=payload_identities,
            payloads=payloads,
        )
        _assert_input_stable(input_value)
        _assert_bound_directory(
            parent_fd=parent_fd,
            name=output_dir.name,
            directory_fd=directory_fd,
            expected_mode=artifact_permissions.DIRECTORY_MODE,
        )
        _assert_bound_payloads(
            directory_fd=directory_fd,
            descriptors=payload_fds,
            sealed_identities=payload_identities,
            payloads=payloads,
        )
        root_status = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(root_status.st_mode)
            or stat.S_IMODE(root_status.st_mode)
            != artifact_permissions.DIRECTORY_MODE
        ):
            raise CandidateTemporalAnalysisError(
                "sealed analysis output root inode differs"
            )
        identities = {
            ".": _status_identity(root_status),
            **payload_identities,
        }
        committed = True
        return identities
    except BaseException:
        if directory_fd is not None and not committed:
            try:
                os.fchmod(directory_fd, 0o700)
                os.fsync(directory_fd)
            except OSError:
                pass
        raise
    finally:
        for descriptor in payload_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                if not committed:
                    raise
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                if not committed:
                    raise
        try:
            os.close(parent_fd)
        except OSError:
            if not committed:
                raise


def _strict_resume(
    output_dir: Path,
    *,
    payloads: Mapping[str, bytes],
) -> dict[str, tuple[int, ...]]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise CandidateTemporalAnalysisError(
            "resume requires an existing regular output directory"
        )
    mode = stat.S_IMODE(output_dir.lstat().st_mode)
    if mode != artifact_permissions.DIRECTORY_MODE:
        raise CandidateTemporalAnalysisError(
            f"resume requires a sealed output root: {mode:04o}"
        )
    actual = {entry.name for entry in output_dir.iterdir()}
    if actual != set(OUTPUT_NAMES):
        raise CandidateTemporalAnalysisError(
            "resume output artifact closure differs"
        )
    identities = _capture_identities_for_names(
        output_dir,
        OUTPUT_NAMES,
    )
    artifact_permissions.assert_sealed_tree(output_dir)
    for name in OUTPUT_NAMES:
        path = output_dir / name
        status = path.lstat()
        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or path.read_bytes() != payloads[name]
        ):
            raise CandidateTemporalAnalysisError(
                f"resume payload differs: {name}"
            )
    if _capture_identities_for_names(
        output_dir,
        OUTPUT_NAMES,
    ) != identities:
        raise CandidateTemporalAnalysisError(
            "analysis output identities changed during resume"
        )
    return identities


def _overlaps(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def run_candidate_temporal_analysis(
    *,
    screen_dir: Path,
    expected_screen_done_sha256: str,
    output_dir: Path,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
    resume: bool = False,
) -> dict[str, Any]:
    """Derive, atomically claim and seal, or strictly resume the analysis."""

    output = output_dir.expanduser().resolve(strict=False)
    if output_dir.expanduser().is_symlink():
        raise CandidateTemporalAnalysisError(
            "analysis output must not be a symlink"
        )
    if resume:
        if not output.exists():
            raise FileNotFoundError(output)
    elif output.exists():
        raise FileExistsError(
            f"{output} exists; use a fresh path or resume=True"
        )
    screen_root = screen_dir.expanduser().resolve(strict=True)
    if _overlaps(output, screen_root):
        raise CandidateTemporalAnalysisError(
            "analysis output overlaps the screen input"
        )
    payloads, input_value = _derive(
        screen_dir=screen_root,
        expected_screen_done_sha256=expected_screen_done_sha256,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    if resume:
        output_identities = _strict_resume(
            output,
            payloads=payloads,
        )
    else:
        output_identities = _publish(
            output,
            payloads=payloads,
            input_value=input_value,
        )
    _assert_input_stable(input_value)
    result = _validate_analysis_envelope(
        output,
        expected_identities=output_identities,
    )
    if _capture_identities_for_names(
        output,
        OUTPUT_NAMES,
    ) != output_identities:
        raise CandidateTemporalAnalysisError(
            "analysis output identities changed across the run operation"
        )
    result["input_screen_verified"] = True
    return result


def validate_candidate_temporal_analysis(
    output_dir: Path,
    *,
    expected_done_sha256: str,
    screen_dir: Path,
    expected_screen_done_sha256: str,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Re-derive every payload from the bound sealed screen output."""

    expected = _require_sha256(
        expected_done_sha256,
        context="expected analysis done SHA",
    )
    root = output_dir.expanduser().resolve(strict=True)
    output_identities = _capture_identities_for_names(
        root,
        OUTPUT_NAMES,
    )
    if _file_digest(root / DONE_NAME) != expected:
        raise CandidateTemporalAnalysisError(
            "analysis external done SHA differs"
        )
    result = _validate_analysis_envelope(
        root,
        expected_identities=output_identities,
    )
    payloads, input_value = _derive(
        screen_dir=screen_dir,
        expected_screen_done_sha256=expected_screen_done_sha256,
        bootstrap_repetitions=bootstrap_repetitions,
        seed=seed,
    )
    for name in OUTPUT_NAMES:
        if (root / name).read_bytes() != payloads[name]:
            raise CandidateTemporalAnalysisError(
                f"analysis output does not replay from screen: {name}"
            )
    if _capture_identities_for_names(
        root,
        OUTPUT_NAMES,
    ) != output_identities:
        raise CandidateTemporalAnalysisError(
            "analysis output identities changed during full replay"
        )
    _assert_input_stable(input_value)
    result["input_screen_verified"] = True
    result["expected_done_sha256"] = expected
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "paired held-out component-bootstrap analysis of one sealed "
            "R7 candidate temporal screen"
        )
    )
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-screen-done-sha256",
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_candidate_temporal_analysis(
        screen_dir=args.screen_dir,
        expected_screen_done_sha256=(
            args.expected_screen_done_sha256
        ),
        output_dir=args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        resume=args.resume,
    )
    summary = result["summary"]
    print(
        _canonical_json(
            {
                "status": summary["status"],
                "formal_status": summary["formal_status"],
                "comparisons": summary["output"]["comparisons"],
                "eligible_positive_queries": summary["coverage"][
                    "eligible_positive_queries"
                ],
                "output_dir": str(result["directory"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_SCHEMA",
    "BOOTSTRAP_PROTOCOL",
    "COMPARISON_SCHEMA",
    "COMPARISONS_NAME",
    "CONTRASTS",
    "DEFAULT_BOOTSTRAP_REPETITIONS",
    "DEFAULT_SEED",
    "DONE_NAME",
    "DONE_SCHEMA",
    "SAFETY_FIELDS",
    "SUMMARY_NAME",
    "CandidateTemporalAnalysisError",
    "main",
    "run_candidate_temporal_analysis",
    "validate_candidate_temporal_analysis",
]

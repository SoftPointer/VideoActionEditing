"""Post-hoc R8 mechanism diagnostic over one frozen R7-P1 commit.

This module is deliberately downstream of R7/P1.  It consumes only the three
files in a finalized P1 diagnostic directory (``rows.jsonl``,
``summary.json``, and ``done.json``), never reads the track cache, and never
changes or reinterprets the frozen P1 gate.

For every target base/perturbed record that P1 marked ``diagnostic_ready``,
the module calls :func:`r8_stable_motion.build_stable_motion_representation`.
It then reports the old aggregate-trajectory RMSE beside the R8 robust global
trajectory RMSE and descriptive energy/shape/support comparisons.  Missing
comparisons remain missing and the full 97-row positive target camera-valid
denominator is always explicit.

Each serialized stable representation also preserves the complete per-phase
center certificate and its solver delta, tolerance, and storage dtype.

The result is a post-hoc mechanism artifact, not a new gate, formal evidence,
or generation authorization.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r7_coherent_actor import R7_COHERENT_ACTOR_SCHEMA
from .r7_p1_diagnostic import (
    R7_P1_DIAGNOSTIC_DONE_SCHEMA,
    R7_P1_DIAGNOSTIC_ROW_SCHEMA,
    R7_P1_DIAGNOSTIC_SCHEMA,
    R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
)
from .r7_preflight_extract import (
    _atomic_json,
    _atomic_jsonl,
    _canonical_json,
    _file_digest,
    _object_digest,
)
from .r8_stable_motion import (
    R8_STABLE_MOTION_PHASE_STEPS,
    R8_STABLE_MOTION_SCHEMA,
    StableMotionConfig,
    StableMotionRepresentation,
    build_stable_motion_representation,
)


R8_P1_MECHANISM_SCHEMA = "motive-r8-p1-mechanism-diagnostic-v2"
R8_P1_MECHANISM_ROW_SCHEMA = "motive-r8-p1-mechanism-row-v2"
R8_P1_MECHANISM_SUMMARY_SCHEMA = (
    "motive-r8-p1-mechanism-summary-v2"
)
R8_P1_MECHANISM_DONE_SCHEMA = "motive-r8-p1-mechanism-done-v2"

ROWS_NAME = "rows.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
EXPECTED_ROWS = 181
EXPECTED_AUDIT_DENOMINATOR = 97
FORMAL_STATUS = "INSUFFICIENT"
POSTHOC_SCOPE = (
    "post-hoc mechanism diagnostic on the inspected 181-row development "
    "cohort; does not create or modify a gate"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "rows": directory / ROWS_NAME,
        "summary": directory / SUMMARY_NAME,
        "done": directory / DONE_NAME,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular non-symlink file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular non-symlink file")
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_safe(mapping: Mapping[str, Any], *, where: str) -> None:
    if mapping.get("formal_status") != FORMAL_STATUS:
        raise ValueError(f"{where} formal_status is unsafe")
    for key in ("production_decision", "generation_authorized"):
        if mapping.get(key) is not False:
            raise ValueError(f"{where} {key} is unsafe")


def _finite_json(value: Any, *, where: str) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} is non-finite")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_json(child, where=f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_json(child, where=f"{where}[{index}]")
        return
    raise ValueError(f"{where} has unsupported type {type(value).__name__}")


def _validate_ready_side(record: Mapping[str, Any], *, where: str) -> None:
    if record.get("diagnostic_ready") is not True:
        return
    if (
        record.get("selector_ready") is not True
        or record.get("event_ready") is not True
        or record.get("event_window") is None
    ):
        raise ValueError(f"{where} has inconsistent ready flags")
    selector = record.get("selector")
    if not isinstance(selector, Mapping):
        raise ValueError(f"{where} lacks selector metadata")
    if selector.get("schema_version") != R7_COHERENT_ACTOR_SCHEMA:
        raise ValueError(f"{where} selector schema differs")
    indices = np.asarray(selector.get("actor_track_indices"))
    trajectories = np.asarray(record.get("actor_track_trajectories"))
    mask = np.asarray(record.get("actor_track_phase_mask"))
    energy = np.asarray(record.get("selector_phase_energy"))
    times = np.asarray(record.get("phase_times"))
    aggregate = np.asarray(record.get("actor_trajectory"))
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or bool((indices < 0).any())
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(f"{where} component indices are invalid")
    expected_tracks = (
        len(indices),
        R8_STABLE_MOTION_PHASE_STEPS,
        2,
    )
    if trajectories.shape != expected_tracks:
        raise ValueError(f"{where} track trajectory shape differs")
    if mask.shape != expected_tracks[:2] or mask.dtype != np.bool_:
        raise ValueError(f"{where} phase mask differs")
    if energy.shape != (R8_STABLE_MOTION_PHASE_STEPS,):
        raise ValueError(f"{where} selector energy shape differs")
    if aggregate.shape != (R8_STABLE_MOTION_PHASE_STEPS, 2):
        raise ValueError(f"{where} aggregate trajectory shape differs")
    if times.shape != (R8_STABLE_MOTION_PHASE_STEPS,) or bool(
        (np.diff(times.astype(np.float64)) <= 0.0).any()
    ):
        raise ValueError(f"{where} phase times differ")
    numeric = (trajectories, energy, times, aggregate)
    if not all(
        np.issubdtype(value.dtype, np.number)
        and np.isfinite(value.astype(np.float64)).all()
        for value in numeric
    ):
        raise ValueError(f"{where} contains invalid numeric arrays")
    if bool((energy.astype(np.float64) < 0.0).any()):
        raise ValueError(f"{where} selector energy is negative")
    coordinate_space = selector.get("coordinate_space")
    if not isinstance(coordinate_space, str) or not coordinate_space:
        raise ValueError(f"{where} coordinate space is missing")


def _validate_p1_row(
    row: Mapping[str, Any],
    *,
    expected_index: int,
    seen_iids: set[str],
) -> None:
    if row.get("schema_version") != R7_P1_DIAGNOSTIC_ROW_SCHEMA:
        raise ValueError(f"P1 row {expected_index} schema differs")
    if row.get("input_index") != expected_index:
        raise ValueError(f"P1 row {expected_index} order/index differs")
    iid = row.get("iid")
    if not isinstance(iid, str) or not iid or iid in seen_iids:
        raise ValueError(f"P1 row {expected_index} iid differs/duplicates")
    seen_iids.add(iid)
    if not _is_sha256(row.get("cache_row_sha256")):
        raise ValueError(f"P1 row {expected_index} cache digest differs")
    for flag in ("positive", "source_camera_valid", "target_camera_valid"):
        if not isinstance(row.get(flag), bool):
            raise ValueError(f"P1 row {expected_index} {flag} is not boolean")
    for side in ("source", "target"):
        record = row.get(side)
        if not isinstance(record, Mapping):
            raise ValueError(f"P1 row {expected_index} lacks {side}")
        _validate_ready_side(record, where=f"P1 row {expected_index}.{side}")
    audit = row.get("target_audit")
    if not isinstance(audit, Mapping):
        raise ValueError(f"P1 row {expected_index} lacks target audit")
    eligible = bool(row["positive"] and row["target_camera_valid"])
    if (
        audit.get("eligible") is not eligible
        or audit.get("performed") is not eligible
    ):
        raise ValueError(f"P1 row {expected_index} audit scope differs")
    comparison_available = audit.get("comparison_available")
    if not isinstance(comparison_available, bool):
        raise ValueError(
            f"P1 row {expected_index} comparison flag is not boolean"
        )
    perturbed = audit.get("perturbed")
    base_ready = bool(row["target"].get("diagnostic_ready"))
    perturbed_ready = bool(
        isinstance(perturbed, Mapping)
        and perturbed.get("diagnostic_ready")
    )
    if comparison_available is not bool(base_ready and perturbed_ready):
        raise ValueError(
            f"P1 row {expected_index} comparison availability differs"
        )
    if isinstance(perturbed, Mapping):
        _validate_ready_side(
            perturbed,
            where=f"P1 row {expected_index}.target_audit.perturbed",
        )
    if comparison_available:
        metrics = audit.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"P1 row {expected_index} lacks audit metrics")
        for key in (
            "trajectory_rmse",
            "energy_cosine",
            "shape_profile_cosine",
        ):
            value = metrics.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"P1 row {expected_index} audit {key} differs"
                )
    _require_safe(row, where=f"P1 row {expected_index}")
    _finite_json(row, where=f"P1 row {expected_index}")


def _validate_p1_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != R7_P1_DIAGNOSTIC_SCHEMA:
        raise ValueError("P1 contract schema differs")
    _require_safe(contract, where="P1 contract")
    gate = contract.get("diagnostic_gate")
    audit = contract.get("independent_audit")
    if not isinstance(gate, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("P1 contract lacks audit/gate provenance")
    for key in (
        "thresholds_frozen_before_p1_cache_results",
        "not_independent_preregistration",
        "design_was_driven_by_prior_p0_failure",
        "thresholds_may_not_be_adjusted_from_results",
    ):
        if gate.get(key) is not True:
            raise ValueError(f"P1 gate provenance {key} differs")
    if (
        audit.get("base_selection_is_audit_mask") is not False
        or audit.get("missing_comparison_credit") != 0
    ):
        raise ValueError("P1 audit denominator policy differs")
    audit_config = audit.get("config")
    gate_config = gate.get("config")
    if (
        not isinstance(audit_config, Mapping)
        or not isinstance(gate_config, Mapping)
        or audit.get("config_sha256") != _object_digest(audit_config)
        or gate.get("config_sha256") != _object_digest(gate_config)
    ):
        raise ValueError("P1 audit/gate config digest differs")
    threshold = audit_config.get("trajectory_rmse_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) <= 0.0
    ):
        raise ValueError("P1 trajectory reference threshold differs")
    implementation = contract.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("P1 implementation provenance is missing")
    if not all(_is_sha256(value) for value in implementation.values()):
        raise ValueError("P1 implementation digest differs")


def load_p1_commit(input_directory: Path) -> dict[str, Any]:
    """Validate the self-contained frozen P1 three-file commit."""

    root = input_directory.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("P1 input must be a real directory")
    paths = _paths(root)
    rows = _load_rows(paths["rows"])
    summary = _load_json(paths["summary"])
    done = _load_json(paths["done"])
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"P1 input must contain exactly {EXPECTED_ROWS} rows")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _validate_p1_row(row, expected_index=index, seen_iids=seen)

    if summary.get("schema_version") != R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA:
        raise ValueError("P1 summary schema differs")
    if done.get("schema_version") != R7_P1_DIAGNOSTIC_DONE_SCHEMA:
        raise ValueError("P1 done schema differs")
    if done.get("committed") is not True:
        raise ValueError("P1 commit is incomplete")
    _require_safe(summary, where="P1 summary")
    _require_safe(done, where="P1 done")
    contract = summary.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("P1 summary lacks contract")
    _validate_p1_contract(contract)
    contract_digest = _object_digest(contract)
    rows_object_digest = _object_digest(rows)
    if (
        summary.get("rows") != EXPECTED_ROWS
        or summary.get("contract_sha256") != contract_digest
        or summary.get("rows_object_sha256") != rows_object_digest
    ):
        raise ValueError("P1 summary binding differs")
    rows_digest = _file_digest(paths["rows"])
    summary_digest = _file_digest(paths["summary"])
    if (
        done.get("rows") != EXPECTED_ROWS
        or done.get("rows_sha256") != rows_digest
        or done.get("summary_sha256") != summary_digest
        or done.get("contract_sha256") != contract_digest
    ):
        raise ValueError("P1 done byte binding differs")
    gate = summary.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("P1 summary lacks frozen gate")
    _require_safe(gate, where="P1 frozen gate")
    counts = gate.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("P1 gate lacks counts")
    denominator = sum(
        int(bool(row["positive"] and row["target_camera_valid"]))
        for row in rows
    )
    if (
        denominator != EXPECTED_AUDIT_DENOMINATOR
        or counts.get("rows") != EXPECTED_ROWS
        or counts.get("positive_target_camera_valid") != denominator
    ):
        raise ValueError("P1 audit denominator is not the frozen 97 rows")
    gate_decision = gate.get("diagnostic_gate_passed")
    done_decision = done.get("diagnostic_gate_passed")
    if (
        not isinstance(gate_decision, bool)
        or not isinstance(done_decision, bool)
        or done_decision is not gate_decision
    ):
        raise ValueError("P1 done gate binding differs")
    return {
        "directory": root,
        "paths": paths,
        "rows": rows,
        "summary": summary,
        "done": done,
        "contract": dict(contract),
        "rows_sha256": rows_digest,
        "summary_sha256": summary_digest,
        "done_sha256": _file_digest(paths["done"]),
        "contract_sha256": contract_digest,
        "rows_object_sha256": rows_object_digest,
        "audit_denominator": denominator,
    }


def _safe_output_root(*, input_directory: Path, output_directory: Path) -> Path:
    input_root = input_directory.expanduser().resolve(strict=True)
    output = output_directory.expanduser().resolve(strict=False)
    if output == Path(output.anchor):
        raise ValueError("refusing a filesystem-root output")
    if (
        output == input_root
        or output in input_root.parents
        or input_root in output.parents
    ):
        raise ValueError("R8 output must not overlap the frozen P1 input")
    return output


def _representation_to_dict(
    representation: StableMotionRepresentation,
) -> dict[str, Any]:
    return {
        "summary": representation.to_summary(),
        "trajectory": representation.trajectory.tolist(),
        "transition_displacement": (
            representation.transition_displacement.tolist()
        ),
        "center_certificate_kind": list(
            representation.center_certificate_kind
        ),
        "center_position_error_upper_bound": (
            representation.center_position_error_upper_bound.tolist()
        ),
        "center_global_curvature_lower_bound": (
            representation.center_global_curvature_lower_bound.tolist()
        ),
        "center_gradient_upper_bound": (
            representation.center_gradient_upper_bound.tolist()
        ),
        "phase_energy": representation.phase_energy.tolist(),
        "shape_tokens": representation.shape_tokens.tolist(),
        "phase_support": representation.phase_support.tolist(),
        "transition_support": representation.transition_support.tolist(),
        "transition_support_count": (
            representation.transition_support_count.tolist()
        ),
        "phase_times": representation.phase_times.tolist(),
        "component_track_indices": (
            representation.component_track_indices.tolist()
        ),
        "track_anchor_phase": representation.track_anchor_phase.tolist(),
    }


def _evaluate_ready_record(
    record: Mapping[str, Any],
    *,
    config: StableMotionConfig,
) -> dict[str, Any]:
    upstream_ready = bool(record.get("diagnostic_ready"))
    if not upstream_ready:
        return {
            "upstream_diagnostic_ready": False,
            "attempted": False,
            "r8_diagnostic_ready": False,
            "failure_reason": "upstream_not_ready",
            "representation": None,
        }
    selector = record["selector"]
    representation = build_stable_motion_representation(
        record["actor_track_trajectories"],
        record["actor_track_phase_mask"],
        record["selector_phase_energy"],
        component_track_indices=selector["actor_track_indices"],
        phase_times=record["phase_times"],
        coordinate_space=selector["coordinate_space"],
        config=config,
    )
    return {
        "upstream_diagnostic_ready": True,
        "attempted": True,
        "r8_diagnostic_ready": representation.diagnostic_ready,
        "failure_reason": representation.failure_reason,
        "representation": _representation_to_dict(representation),
    }


def _rmse(first: Any, second: Any) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or not first_array.size:
        raise ValueError("comparison arrays differ")
    if not np.isfinite(first_array).all() or not np.isfinite(second_array).all():
        raise ValueError("comparison arrays are non-finite")
    return float(np.sqrt(np.mean((first_array - second_array) ** 2)))


def _cosine(first: Any, second: Any) -> float:
    first_array = np.asarray(first, dtype=np.float64).reshape(-1)
    second_array = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_array.shape != second_array.shape or not first_array.size:
        raise ValueError("cosine arrays differ")
    first_norm = float(np.linalg.norm(first_array))
    second_norm = float(np.linalg.norm(second_array))
    if first_norm <= 1e-12 and second_norm <= 1e-12:
        return 1.0
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return 0.0
    return float(
        np.clip(
            np.dot(first_array, second_array) / (first_norm * second_norm),
            -1.0,
            1.0,
        )
    )


def _membership_iou(first: Sequence[int], second: Sequence[int]) -> float:
    first_set = {int(value) for value in first}
    second_set = {int(value) for value in second}
    union = first_set | second_set
    return float(len(first_set & second_set) / len(union)) if union else 0.0


def _max_center_certificate_bound(
    representation: Mapping[str, Any],
) -> float:
    summary = representation.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("R8 representation lacks center summary")
    if (
        summary.get("schema_version") != R8_STABLE_MOTION_SCHEMA
        or summary.get("diagnostic_ready") is not True
        or summary.get("phase_steps") != R8_STABLE_MOTION_PHASE_STEPS
        or summary.get("center_storage_dtype") != "float64"
    ):
        raise ValueError("R8 center summary contract differs")
    for name in (
        "smoothed_center_delta",
        "smoothed_center_absolute_position_tolerance",
    ):
        raw_value = summary.get(name)
        if isinstance(raw_value, bool):
            raise ValueError(f"R8 center summary {name} differs")
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"R8 center summary {name} differs") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"R8 center summary {name} differs")

    kinds = representation.get("center_certificate_kind")
    if (
        not isinstance(kinds, list)
        or len(kinds) != R8_STABLE_MOTION_PHASE_STEPS
        or any(not isinstance(kind, str) or not kind for kind in kinds)
    ):
        raise ValueError("R8 center certificate kinds differ")
    expected_kind_counts = {
        kind: kinds.count(kind) for kind in sorted(set(kinds))
    }
    if summary.get("center_certificate_kind_counts") != expected_kind_counts:
        raise ValueError("R8 center certificate kind counts differ")
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "center_position_error_upper_bound",
        "center_global_curvature_lower_bound",
        "center_gradient_upper_bound",
    ):
        array = np.asarray(representation.get(name), dtype=np.float64)
        if (
            array.shape != (R8_STABLE_MOTION_PHASE_STEPS,)
            or not np.isfinite(array).all()
            or bool((array < 0.0).any())
        ):
            raise ValueError(f"R8 {name} differs")
        arrays[name] = array
    position_bound = arrays["center_position_error_upper_bound"]
    maximum = float(np.max(position_bound))
    summary_maximum = summary.get(
        "max_center_position_error_upper_bound"
    )
    if (
        isinstance(summary_maximum, bool)
        or not isinstance(summary_maximum, (int, float))
        or not math.isclose(
            maximum,
            float(summary_maximum),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("R8 center certificate maximum differs")
    tolerance = float(
        summary["smoothed_center_absolute_position_tolerance"]
    )
    if maximum > tolerance:
        raise ValueError("R8 center certificate exceeds tolerance")
    return maximum


def _comparison(
    *,
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    perturbed: Mapping[str, Any],
    trajectory_threshold: float,
) -> dict[str, Any]:
    audit = row["target_audit"]
    upstream_available = bool(audit["comparison_available"])
    r8_available = bool(
        upstream_available
        and base["r8_diagnostic_ready"]
        and perturbed["r8_diagnostic_ready"]
    )
    output: dict[str, Any] = {
        "audit_eligible": bool(audit["eligible"]),
        "upstream_comparison_available": upstream_available,
        "r8_comparison_available": r8_available,
        "frozen_p1_trajectory_threshold": trajectory_threshold,
        "threshold_role": "descriptive frozen-P1 reference; not a new gate",
        "old_aggregate_trajectory_rmse": None,
        "r8_global_trajectory_rmse": None,
        "p1_event_energy_cosine": None,
        "p1_shape_profile_cosine": None,
        "r8_selector_energy_cosine": None,
        "r8_shape_token_cosine": None,
        "phase_support_rmse": None,
        "transition_support_rmse": None,
        "phase_support_cosine": None,
        "transition_support_cosine": None,
        "component_membership_iou": None,
        "base_max_center_position_error_upper_bound": None,
        "perturbed_max_center_position_error_upper_bound": None,
        "old_reference_pass": False,
        "r8_reference_pass": False,
        "reference_rescue": False,
        "reference_regression": False,
    }
    if not upstream_available:
        return output
    metrics = audit["metrics"]
    recomputed_old = _rmse(
        row["target"]["actor_trajectory"],
        audit["perturbed"]["actor_trajectory"],
    )
    old_rmse = float(metrics["trajectory_rmse"])
    if not math.isclose(old_rmse, recomputed_old, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"P1 row {row['input_index']} aggregate RMSE does not bind rows"
        )
    output.update(
        {
            "old_aggregate_trajectory_rmse": old_rmse,
            "p1_event_energy_cosine": float(metrics["energy_cosine"]),
            "p1_shape_profile_cosine": float(
                metrics["shape_profile_cosine"]
            ),
            "old_reference_pass": old_rmse <= trajectory_threshold,
        }
    )
    if not r8_available:
        return output
    base_rep = base["representation"]
    perturbed_rep = perturbed["representation"]
    r8_rmse = _rmse(
        base_rep["trajectory"],
        perturbed_rep["trajectory"],
    )
    phase_support_rmse = _rmse(
        base_rep["phase_support"],
        perturbed_rep["phase_support"],
    )
    transition_support_rmse = _rmse(
        base_rep["transition_support"],
        perturbed_rep["transition_support"],
    )
    r8_pass = r8_rmse <= trajectory_threshold
    old_pass = bool(output["old_reference_pass"])
    output.update(
        {
            "r8_global_trajectory_rmse": r8_rmse,
            "r8_selector_energy_cosine": _cosine(
                base_rep["phase_energy"],
                perturbed_rep["phase_energy"],
            ),
            "r8_shape_token_cosine": _cosine(
                base_rep["shape_tokens"],
                perturbed_rep["shape_tokens"],
            ),
            "phase_support_rmse": phase_support_rmse,
            "transition_support_rmse": transition_support_rmse,
            "phase_support_cosine": _cosine(
                base_rep["phase_support"],
                perturbed_rep["phase_support"],
            ),
            "transition_support_cosine": _cosine(
                base_rep["transition_support"],
                perturbed_rep["transition_support"],
            ),
            "component_membership_iou": _membership_iou(
                base_rep["component_track_indices"],
                perturbed_rep["component_track_indices"],
            ),
            "base_max_center_position_error_upper_bound": (
                _max_center_certificate_bound(base_rep)
            ),
            "perturbed_max_center_position_error_upper_bound": (
                _max_center_certificate_bound(perturbed_rep)
            ),
            "r8_reference_pass": r8_pass,
            "reference_rescue": bool(not old_pass and r8_pass),
            "reference_regression": bool(old_pass and not r8_pass),
        }
    )
    return output


def build_rows(
    p1: Mapping[str, Any],
    *,
    config: StableMotionConfig,
) -> list[dict[str, Any]]:
    """Recompute all R8 mechanism rows from the frozen P1 rows."""

    threshold = float(
        p1["contract"]["independent_audit"]["config"][
            "trajectory_rmse_threshold"
        ]
    )
    output: list[dict[str, Any]] = []
    for p1_row in p1["rows"]:
        audit = p1_row["target_audit"]
        perturbed_record = audit.get("perturbed")
        base = _evaluate_ready_record(p1_row["target"], config=config)
        perturbed = (
            _evaluate_ready_record(perturbed_record, config=config)
            if isinstance(perturbed_record, Mapping)
            else {
                "upstream_diagnostic_ready": False,
                "attempted": False,
                "r8_diagnostic_ready": False,
                "failure_reason": "upstream_not_ready",
                "representation": None,
            }
        )
        output.append(
            {
                "schema_version": R8_P1_MECHANISM_ROW_SCHEMA,
                "input_index": p1_row["input_index"],
                "iid": p1_row["iid"],
                "p1_row_sha256": _object_digest(p1_row),
                "positive": p1_row["positive"],
                "target_camera_valid": p1_row["target_camera_valid"],
                "base": base,
                "perturbed": perturbed,
                "comparison": _comparison(
                    row=p1_row,
                    base=base,
                    perturbed=perturbed,
                    trajectory_threshold=threshold,
                ),
                "posthoc": True,
                "creates_new_gate": False,
                "overrides_p1": False,
                "formal_status": FORMAL_STATUS,
                "production_decision": False,
                "generation_authorized": False,
            }
        )
    return output


def build_contract(
    p1: Mapping[str, Any],
    *,
    config: StableMotionConfig,
) -> dict[str, Any]:
    module = Path(__file__).resolve(strict=True)
    stable_module = module.with_name("r8_stable_motion.py")
    config_dict = asdict(config)
    threshold = float(
        p1["contract"]["independent_audit"]["config"][
            "trajectory_rmse_threshold"
        ]
    )
    return {
        "schema_version": R8_P1_MECHANISM_SCHEMA,
        "input_p1": {
            "directory": str(p1["directory"]),
            "rows_sha256": p1["rows_sha256"],
            "summary_sha256": p1["summary_sha256"],
            "done_sha256": p1["done_sha256"],
            "contract_sha256": p1["contract_sha256"],
            "rows_object_sha256": p1["rows_object_sha256"],
            "rows": EXPECTED_ROWS,
            "audit_denominator": EXPECTED_AUDIT_DENOMINATOR,
            "row_schema": R7_P1_DIAGNOSTIC_ROW_SCHEMA,
            "summary_schema": R7_P1_DIAGNOSTIC_SUMMARY_SCHEMA,
            "done_schema": R7_P1_DIAGNOSTIC_DONE_SCHEMA,
        },
        "stable_motion": {
            "schema_version": R8_STABLE_MOTION_SCHEMA,
            "config": config_dict,
            "config_sha256": _object_digest(config_dict),
            "serialized_center_evidence": {
                "per_phase_fields": [
                    "center_certificate_kind",
                    "center_position_error_upper_bound",
                    "center_global_curvature_lower_bound",
                    "center_gradient_upper_bound",
                ],
                "summary_fields": [
                    "smoothed_center_delta",
                    "smoothed_center_absolute_position_tolerance",
                    "center_storage_dtype",
                    "max_center_position_error_upper_bound",
                ],
                "phase_steps": R8_STABLE_MOTION_PHASE_STEPS,
                "center_storage_dtype": "float64",
            },
        },
        "implementation_sha256": {
            module.name: _file_digest(module),
            stable_module.name: _file_digest(stable_module),
        },
        "comparison": {
            "old_quantity": "P1 aggregate actor trajectory RMSE",
            "new_quantity": "R8 robust global trajectory RMSE",
            "frozen_p1_trajectory_threshold": threshold,
            "threshold_role": (
                "descriptive frozen-P1 reference only; not a new gate"
            ),
            "also_reports": [
                "selector energy cosine",
                "shape-token cosine",
                "phase/transition support",
                "component membership IoU",
                "base/perturbed maximum center-certificate bound",
            ],
        },
        "posthoc": True,
        "creates_new_gate": False,
        "overrides_p1": False,
        "development_scope": POSTHOC_SCOPE,
        "formal_status": FORMAL_STATUS,
        "production_decision": False,
        "generation_authorized": False,
    }


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("descriptive values are non-finite")
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [row for row in rows if row["comparison"]["audit_eligible"]]
    upstream = [
        row
        for row in eligible
        if row["comparison"]["upstream_comparison_available"]
    ]
    available = [
        row
        for row in eligible
        if row["comparison"]["r8_comparison_available"]
    ]
    failures = Counter()
    for row in rows:
        for side in ("base", "perturbed"):
            value = row[side]
            if value["attempted"] and not value["r8_diagnostic_ready"]:
                failures[str(value["failure_reason"])] += 1
    counts = {
        "rows": len(rows),
        "audit_denominator_all_positive_target_camera_valid": len(eligible),
        "upstream_comparison_available": len(upstream),
        "upstream_unavailable_zero_credit": len(eligible) - len(upstream),
        "r8_comparison_available": len(available),
        "r8_unavailable_zero_credit": len(eligible) - len(available),
        "old_reference_pass": sum(
            bool(row["comparison"]["old_reference_pass"])
            for row in eligible
        ),
        "r8_reference_pass": sum(
            bool(row["comparison"]["r8_reference_pass"])
            for row in eligible
        ),
        "reference_rescue": sum(
            bool(row["comparison"]["reference_rescue"])
            for row in eligible
        ),
        "reference_regression": sum(
            bool(row["comparison"]["reference_regression"])
            for row in eligible
        ),
    }
    if len(rows) != EXPECTED_ROWS or len(eligible) != EXPECTED_AUDIT_DENOMINATOR:
        raise ValueError("R8 summary denominator differs")
    return {
        "schema_version": R8_P1_MECHANISM_SUMMARY_SCHEMA,
        "rows": len(rows),
        "contract": dict(contract),
        "contract_sha256": _object_digest(contract),
        "rows_object_sha256": _object_digest(list(rows)),
        "counts": counts,
        "r8_failure_counts": dict(sorted(failures.items())),
        "descriptive_only": {
            "old_aggregate_trajectory_rmse": _descriptive(
                [
                    float(row["comparison"][
                        "old_aggregate_trajectory_rmse"
                    ])
                    for row in upstream
                ]
            ),
            "r8_global_trajectory_rmse": _descriptive(
                [
                    float(row["comparison"]["r8_global_trajectory_rmse"])
                    for row in available
                ]
            ),
            "r8_selector_energy_cosine": _descriptive(
                [
                    float(row["comparison"]["r8_selector_energy_cosine"])
                    for row in available
                ]
            ),
            "r8_shape_token_cosine": _descriptive(
                [
                    float(row["comparison"]["r8_shape_token_cosine"])
                    for row in available
                ]
            ),
            "phase_support_rmse": _descriptive(
                [
                    float(row["comparison"]["phase_support_rmse"])
                    for row in available
                ]
            ),
            "transition_support_rmse": _descriptive(
                [
                    float(row["comparison"]["transition_support_rmse"])
                    for row in available
                ]
            ),
            "base_max_center_position_error_upper_bound": _descriptive(
                [
                    float(
                        row["comparison"][
                            "base_max_center_position_error_upper_bound"
                        ]
                    )
                    for row in available
                ]
            ),
            "perturbed_max_center_position_error_upper_bound": (
                _descriptive(
                    [
                        float(
                            row["comparison"][
                                "perturbed_max_center_position_error_upper_bound"
                            ]
                        )
                        for row in available
                    ]
                )
            ),
        },
        "denominator_policy": (
            "all 97 positive target camera-valid P1 rows remain the explicit "
            "denominator; unavailable comparisons receive no success credit"
        ),
        "posthoc": True,
        "creates_new_gate": False,
        "overrides_p1": False,
        "formal_status": FORMAL_STATUS,
        "formal_reason": POSTHOC_SCOPE,
        "production_decision": False,
        "generation_authorized": False,
    }


def _validate_existing(
    *,
    output_directory: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    expected_summary: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _paths(output_directory)
    rows = _load_rows(paths["rows"])
    summary = _load_json(paths["summary"])
    done = _load_json(paths["done"])
    if _canonical_json(rows) != _canonical_json(list(expected_rows)):
        raise ValueError("R8 rows differ from deterministic recomputation")
    if _canonical_json(summary) != _canonical_json(
        dict(expected_summary)
    ):
        raise ValueError("R8 summary differs from deterministic recomputation")
    if done.get("schema_version") != R8_P1_MECHANISM_DONE_SCHEMA:
        raise ValueError("R8 done schema differs")
    expected_done = {
        "schema_version": R8_P1_MECHANISM_DONE_SCHEMA,
        "committed": True,
        "rows": EXPECTED_ROWS,
        "rows_sha256": _file_digest(paths["rows"]),
        "summary_sha256": _file_digest(paths["summary"]),
        "contract_sha256": expected_summary["contract_sha256"],
        "posthoc": True,
        "creates_new_gate": False,
        "overrides_p1": False,
        "formal_status": FORMAL_STATUS,
        "production_decision": False,
        "generation_authorized": False,
    }
    if _canonical_json(done) != _canonical_json(expected_done):
        raise ValueError("R8 done fields/hash binding differ")
    return {"rows": rows, "summary": summary, "done": done}


def run_mechanism_diagnostic(
    *,
    input_directory: Path,
    output_directory: Path,
    config: StableMotionConfig | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Build or byte/recompute-validate one immutable R8 result."""

    cfg = config or StableMotionConfig()
    cfg.validate()
    p1 = load_p1_commit(input_directory)
    output = _safe_output_root(
        input_directory=p1["directory"],
        output_directory=output_directory,
    )
    contract = build_contract(p1, config=cfg)
    rows = build_rows(p1, config=cfg)
    summary = build_summary(rows, contract=contract)
    paths = _paths(output)
    if paths["done"].exists():
        if not resume:
            raise FileExistsError(paths["done"])
        return _validate_existing(
            output_directory=output,
            expected_rows=rows,
            expected_summary=summary,
        )["done"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"partial/nonempty R8 output cannot be resumed: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(paths["rows"], rows)
    _atomic_json(paths["summary"], summary)
    done = {
        "schema_version": R8_P1_MECHANISM_DONE_SCHEMA,
        "committed": True,
        "rows": EXPECTED_ROWS,
        "rows_sha256": _file_digest(paths["rows"]),
        "summary_sha256": _file_digest(paths["summary"]),
        "contract_sha256": summary["contract_sha256"],
        "posthoc": True,
        "creates_new_gate": False,
        "overrides_p1": False,
        "formal_status": FORMAL_STATUS,
        "production_decision": False,
        "generation_authorized": False,
    }
    _atomic_json(paths["done"], done)
    return done


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    done = run_mechanism_diagnostic(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        resume=args.resume,
    )
    print(json.dumps(done, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_AUDIT_DENOMINATOR",
    "EXPECTED_ROWS",
    "R8_P1_MECHANISM_DONE_SCHEMA",
    "R8_P1_MECHANISM_ROW_SCHEMA",
    "R8_P1_MECHANISM_SCHEMA",
    "R8_P1_MECHANISM_SUMMARY_SCHEMA",
    "build_contract",
    "build_rows",
    "build_summary",
    "load_p1_commit",
    "main",
    "run_mechanism_diagnostic",
]

"""Read-only scalar audit for an R10B Lucy or Bernini tangent smoke.

The extractor artifacts contain CountSketch coordinates whose coordinate
systems differ across projection seeds.  This audit therefore computes vector
quantities only within one role, row, and projection seed.  The cross-seed
path receives scalar diagnostics and row rankings only; it never receives
projected vectors.

This is an engineering diagnostic, not a representation evaluation.  It does
not construct retrieval pairs, estimate cross-content generalization, promote
a representation, authorize rendering, or authorize training.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np


AUDIT_SCHEMA = "motive-r10b-tangent-smoke-audit-v1"
SUMMARY_NAME = "summary.json"
ROWS_NAME = "rows.jsonl"
FEATURES_NAME = "features.npz"
_EPSILON = 1e-12
_SCALAR_METRICS = (
    "paired_norm",
    "noop_norm",
    "did_norm",
    "paired_over_noop",
    "did_over_paired",
    "did_over_noop",
    "did_over_paired_plus_noop",
    "paired_vs_noop_cosine",
)
_GATE_FIELDS = (
    "representation_gate_passed",
    "renderer_probe_authorized",
    "editor_training_authorized",
)


class R10BTangentSmokeAuditError(ValueError):
    """The input cannot support a closed, read-only smoke audit."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _pretty_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R10BTangentSmokeAuditError(
            f"cannot read {context}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise R10BTangentSmokeAuditError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise R10BTangentSmokeAuditError(
                        f"row {line_number} in {path} is not an object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise R10BTangentSmokeAuditError(
            f"cannot read rows artifact: {path}"
        ) from error
    return rows


def _backend_contract(
    artifact_dir: Path,
) -> tuple[
    str,
    Callable[[str | Path], dict[str, Any]],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Read only the schema discriminator before invoking the validator."""

    summary = _read_json_object(
        artifact_dir / SUMMARY_NAME,
        context="summary schema discriminator",
    )
    schema = summary.get("schema_version")

    from . import r10b_bernini_tangent_extract as bernini
    from . import r10b_lucy_tangent_extract as lucy

    if schema == lucy.EXTRACT_SCHEMA:
        return (
            "lucy",
            lucy.validate_published_extract,
            tuple(lucy.ROLE_NAMES),
            tuple(lucy.CELL_NAMES),
        )
    if schema == bernini.EXTRACT_SCHEMA:
        return (
            "bernini_r_1_3b",
            bernini.validate_published_extract,
            tuple(bernini.ROLE_NAMES),
            tuple(bernini.CELL_NAMES),
        )
    raise R10BTangentSmokeAuditError(
        f"unsupported R10B extract schema: {schema!r}"
    )


def _require_false_gates(
    value: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for field in _GATE_FIELDS:
        if value.get(field) is not False:
            raise R10BTangentSmokeAuditError(
                f"{context}.{field} must remain false"
            )


def _projection_contract(
    summary: Mapping[str, Any],
) -> tuple[tuple[int, ...], int]:
    measurement = summary.get("measurement")
    if not isinstance(measurement, Mapping):
        raise R10BTangentSmokeAuditError("measurement metadata is missing")
    seeds = measurement.get("projection_seeds")
    dimension = measurement.get("projection_dimension_per_role")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise R10BTangentSmokeAuditError(
            "projection_seeds must contain at least two unique integers"
        )
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise R10BTangentSmokeAuditError(
            "projection_dimension_per_role must be positive"
        )
    return tuple(int(seed) for seed in seeds), int(dimension)


def _positive_finite(value: object, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise R10BTangentSmokeAuditError(
            f"{context} must be finite and positive"
        )
    return float(value)


def _instruction_target_loss_sensitivity(
    row: Mapping[str, Any],
) -> dict[str, float]:
    cells = row.get("cells")
    if not isinstance(cells, Mapping):
        raise R10BTangentSmokeAuditError("row cells metadata is missing")

    losses: dict[str, float] = {}
    for cell in ("tc", "t0"):
        cell_record = cells.get(cell)
        if not isinstance(cell_record, Mapping):
            raise R10BTangentSmokeAuditError(f"row cell {cell} is missing")
        loss_record = cell_record.get("loss")
        if not isinstance(loss_record, Mapping):
            raise R10BTangentSmokeAuditError(
                f"row cell {cell}.loss is missing"
            )
        losses[cell] = _positive_finite(
            loss_record.get("combined_loss"),
            context=f"row cell {cell}.loss.combined_loss",
        )

    difference = losses["tc"] - losses["t0"]
    symmetric_denominator = 0.5 * (abs(losses["tc"]) + abs(losses["t0"]))
    return {
        "edit_instruction_target_loss": losses["tc"],
        "noop_instruction_target_loss": losses["t0"],
        "edit_minus_noop": difference,
        "relative_to_noop": difference / losses["t0"],
        "percent_relative_to_noop": 100.0 * difference / losses["t0"],
        "symmetric_relative_difference": difference / symmetric_denominator,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= _EPSILON:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        return None
    return float(value)


def _finite_norm(value: np.ndarray, *, context: str) -> float:
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm):
        raise R10BTangentSmokeAuditError(
            f"{context} norm is non-finite"
        )
    return norm


def _same_seed_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    """Cosine for two quotients in the same CountSketch coordinate system."""

    left_norm = _finite_norm(left, context="left quotient")
    right_norm = _finite_norm(right, context="right quotient")
    if left_norm <= _EPSILON or right_norm <= _EPSILON:
        return None
    product = math.fsum(
        float(left_value) * float(right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )
    value = product / (left_norm * right_norm)
    if not math.isfinite(value):
        return None
    return float(min(1.0, max(-1.0, value)))


def _one_seed_diagnostics(
    cells: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, float | None]]:
    tc = cells["tc"].astype(np.float64, copy=False)
    sc = cells["sc"].astype(np.float64, copy=False)
    t0 = cells["t0"].astype(np.float64, copy=False)
    s0 = cells["s0"].astype(np.float64, copy=False)
    paired = tc - sc
    noop = t0 - s0
    did = paired - noop

    raw_norms = {
        name: _finite_norm(value, context=f"raw cell {name}")
        for name, value in (("tc", tc), ("sc", sc), ("t0", t0), ("s0", s0))
    }
    quotient_norms = {
        "paired": _finite_norm(paired, context="paired quotient"),
        "noop": _finite_norm(noop, context="noop quotient"),
        "did": _finite_norm(did, context="DID quotient"),
    }
    total = math.fsum(quotient_norms.values())
    ratios = {
        "paired_over_noop": _safe_ratio(
            quotient_norms["paired"],
            quotient_norms["noop"],
        ),
        "did_over_paired": _safe_ratio(
            quotient_norms["did"],
            quotient_norms["paired"],
        ),
        "did_over_noop": _safe_ratio(
            quotient_norms["did"],
            quotient_norms["noop"],
        ),
        "did_over_paired_plus_noop": _safe_ratio(
            quotient_norms["did"],
            quotient_norms["paired"] + quotient_norms["noop"],
        ),
        "paired_share_of_three_norm_sum": _safe_ratio(
            quotient_norms["paired"],
            total,
        ),
        "noop_share_of_three_norm_sum": _safe_ratio(
            quotient_norms["noop"],
            total,
        ),
        "did_share_of_three_norm_sum": _safe_ratio(
            quotient_norms["did"],
            total,
        ),
    }
    cosine = _same_seed_cosine(paired, noop)
    public = {
        "raw_cell_norms": raw_norms,
        "quotient_norms": quotient_norms,
        "quotient_norm_ratios": ratios,
        "paired_vs_noop_cosine": cosine,
        "zero_or_undefined_diagnostics": sorted(
            name
            for name, value in {
                **ratios,
                "paired_vs_noop_cosine": cosine,
            }.items()
            if value is None
        ),
    }
    scalars: dict[str, float | None] = {
        "paired_norm": quotient_norms["paired"],
        "noop_norm": quotient_norms["noop"],
        "did_norm": quotient_norms["did"],
        "paired_over_noop": ratios["paired_over_noop"],
        "did_over_paired": ratios["did_over_paired"],
        "did_over_noop": ratios["did_over_noop"],
        "did_over_paired_plus_noop": ratios[
            "did_over_paired_plus_noop"
        ],
        "paired_vs_noop_cosine": cosine,
    }
    return public, scalars


def _scalar_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise R10BTangentSmokeAuditError("cannot summarize no scalar values")
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    left_centered = [value - left_mean for value in left_rank]
    right_centered = [value - right_mean for value in right_rank]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator <= _EPSILON:
        return None
    numerator = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered,
            right_centered,
            strict=True,
        )
    )
    return float(min(1.0, max(-1.0, numerator / denominator)))


def _cross_seed_scalar_diagnostics(
    scalar_table: Mapping[
        str,
        Mapping[int, Mapping[str, Sequence[float | None]]],
    ],
    *,
    row_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Compare only scalar series and rankings across projection seeds."""

    output: dict[str, Any] = {}
    for role, seed_records in scalar_table.items():
        role_output: dict[str, Any] = {
            "per_seed_scalar_summaries": {},
            "pairwise_seed_row_rank_spearman": [],
            "per_row_scalar_spread": [],
        }
        for seed in seeds:
            metric_summaries: dict[str, Any] = {}
            for metric in _SCALAR_METRICS:
                values = [
                    float(value)
                    for value in seed_records[seed][metric]
                    if value is not None
                ]
                metric_summaries[metric] = (
                    _scalar_summary(values) if values else None
                )
            role_output["per_seed_scalar_summaries"][str(seed)] = (
                metric_summaries
            )

        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                metric_correlations = {}
                comparable_counts = {}
                for metric in _SCALAR_METRICS:
                    left_values = seed_records[left_seed][metric]
                    right_values = seed_records[right_seed][metric]
                    paired_values = [
                        (float(left), float(right))
                        for left, right in zip(
                            left_values,
                            right_values,
                            strict=True,
                        )
                        if left is not None and right is not None
                    ]
                    comparable_counts[metric] = len(paired_values)
                    metric_correlations[metric] = _rank_correlation(
                        [pair[0] for pair in paired_values],
                        [pair[1] for pair in paired_values],
                    )
                role_output["pairwise_seed_row_rank_spearman"].append(
                    {
                        "left_seed": int(left_seed),
                        "right_seed": int(right_seed),
                        "comparable_row_counts": comparable_counts,
                        "metric_correlations": metric_correlations,
                    }
                )

        for row_index, iid in enumerate(row_ids):
            metric_spreads = {}
            for metric in _SCALAR_METRICS:
                values_by_seed = {
                    str(seed): (
                        None
                        if seed_records[seed][metric][row_index] is None
                        else float(seed_records[seed][metric][row_index])
                    )
                    for seed in seeds
                }
                finite_values = [
                    value
                    for value in values_by_seed.values()
                    if value is not None
                ]
                if finite_values:
                    mean = float(statistics.fmean(finite_values))
                    absolute_range = float(max(finite_values) - min(finite_values))
                    relative_range = _safe_ratio(
                        absolute_range,
                        max(abs(mean), _EPSILON),
                    )
                else:
                    mean = None
                    absolute_range = None
                    relative_range = None
                metric_spreads[metric] = {
                    "values_by_seed": values_by_seed,
                    "mean": mean,
                    "absolute_range": absolute_range,
                    "relative_range_to_abs_mean": relative_range,
                }
            role_output["per_row_scalar_spread"].append(
                {
                    "iid": iid,
                    "metrics": metric_spreads,
                }
            )
        output[role] = role_output
    return output


def _load_raw_features(
    artifact_dir: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    roles: Sequence[str],
    cells: Sequence[str],
    seeds: Sequence[int],
    dimension: int,
) -> tuple[list[str], dict[tuple[str, int, str], np.ndarray]]:
    path = artifact_dir / FEATURES_NAME
    raw_features: dict[tuple[str, int, str], np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "ids" not in archive.files:
                raise R10BTangentSmokeAuditError(
                    "features archive has no ids"
                )
            ids = np.asarray(archive["ids"]).astype(str).tolist()
            expected_ids = [str(row.get("iid")) for row in rows]
            if ids != expected_ids or len(set(ids)) != len(ids):
                raise R10BTangentSmokeAuditError(
                    "feature ids and row ids differ"
                )
            for role in roles:
                for seed in seeds:
                    for cell in cells:
                        name = f"raw__{role}__{cell}__p{seed}"
                        if name not in archive.files:
                            raise R10BTangentSmokeAuditError(
                                f"raw feature is missing: {name}"
                            )
                        values = np.asarray(archive[name])
                        if (
                            values.shape != (len(rows), dimension)
                            or not np.issubdtype(values.dtype, np.number)
                            or not np.isfinite(values).all()
                        ):
                            raise R10BTangentSmokeAuditError(
                                f"raw feature is malformed: {name}"
                            )
                        raw_features[(role, seed, cell)] = values.astype(
                            np.float64,
                            copy=True,
                        )
    except (OSError, ValueError) as error:
        if isinstance(error, R10BTangentSmokeAuditError):
            raise
        raise R10BTangentSmokeAuditError(
            f"cannot read features archive: {path}"
        ) from error
    return ids, raw_features


def audit_published_smoke(artifact_dir: str | Path) -> dict[str, Any]:
    """Validate, read, and scalar-audit one immutable extractor artifact."""

    artifact = Path(artifact_dir)
    if not artifact.is_dir():
        raise R10BTangentSmokeAuditError(
            f"artifact directory is missing: {artifact}"
        )
    backend, validator, roles, cells = _backend_contract(artifact)

    # No rows or projected coordinates are consumed before the backend's
    # artifact-closure validator succeeds.
    validation_before = validator(artifact)
    if not isinstance(validation_before, dict):
        raise R10BTangentSmokeAuditError(
            "backend validator returned a non-object"
        )
    _require_false_gates(validation_before, context="validator")

    summary = _read_json_object(
        artifact / SUMMARY_NAME,
        context="validated summary",
    )
    rows = _read_jsonl(artifact / ROWS_NAME)
    if not rows:
        raise R10BTangentSmokeAuditError("smoke artifact contains no rows")
    seeds, dimension = _projection_contract(summary)
    ids, raw_features = _load_raw_features(
        artifact,
        rows=rows,
        roles=roles,
        cells=cells,
        seeds=seeds,
        dimension=dimension,
    )

    scalar_table: dict[
        str,
        dict[int, dict[str, list[float | None]]],
    ] = {
        role: {
            seed: {metric: [] for metric in _SCALAR_METRICS}
            for seed in seeds
        }
        for role in roles
    }
    audit_rows: list[dict[str, Any]] = []
    loss_sensitivities: list[float] = []
    for row_index, row in enumerate(rows):
        sensitivity = _instruction_target_loss_sensitivity(row)
        loss_sensitivities.append(sensitivity["relative_to_noop"])
        role_outputs: dict[str, Any] = {}
        for role in roles:
            seed_outputs: dict[str, Any] = {}
            for seed in seeds:
                cell_vectors = {
                    cell: raw_features[(role, seed, cell)][row_index]
                    for cell in cells
                }
                diagnostics, scalars = _one_seed_diagnostics(cell_vectors)
                seed_outputs[str(seed)] = diagnostics
                for metric in _SCALAR_METRICS:
                    scalar_table[role][seed][metric].append(scalars[metric])
            role_outputs[role] = {
                "by_projection_seed": seed_outputs,
            }
        audit_rows.append(
            {
                "iid": ids[row_index],
                "family": row.get("family"),
                "component_id": row.get("component_id"),
                "instruction_target_loss_sensitivity": sensitivity,
                "by_role": role_outputs,
                "formal_retrieval_evidence": False,
                "representation_gate_passed": False,
                "renderer_probe_authorized": False,
                "editor_training_authorized": False,
            }
        )

    cross_seed = _cross_seed_scalar_diagnostics(
        scalar_table,
        row_ids=ids,
        seeds=seeds,
    )

    # Close the time-of-check/time-of-use window and prove the same validated
    # artifact still exists after all reads.
    validation_after = validator(artifact)
    if validation_after != validation_before:
        raise R10BTangentSmokeAuditError(
            "artifact validation result changed during read-only audit"
        )

    result = {
        "schema_version": AUDIT_SCHEMA,
        "status": "TECHNICAL_SMOKE_AUDITED",
        "backend": backend,
        "input": {
            "artifact_dir": str(artifact.resolve()),
            "artifact_digest": validation_before.get("artifact_digest"),
            "rows": len(rows),
            "projection_seeds": list(seeds),
            "projection_dimension_per_role": dimension,
            "roles": list(roles),
            "backend_validation": validation_before,
            "backend_revalidated_after_reads": True,
        },
        "metric_definitions": {
            "paired": "raw(tc - sc)",
            "noop": "raw(t0 - s0)",
            "did": "raw(tc - sc - t0 + s0)",
            "instruction_target_loss_sensitivity": (
                "(motion_x0_loss(tc) - motion_x0_loss(t0)) / "
                "motion_x0_loss(t0)"
            ),
            "paired_vs_noop_cosine": (
                "computed only within the same row, role, and projection seed"
            ),
        },
        "instruction_target_loss_sensitivity_summary": _scalar_summary(
            loss_sensitivities
        ),
        "rows": audit_rows,
        "cross_projection_seed_diagnostics": {
            "policy": {
                "projected_coordinates_comparable_across_seeds": False,
                "cross_seed_vector_dot_products_computed": False,
                "cross_seed_vector_cosines_computed": False,
                "cross_seed_comparisons": [
                    "scalar summaries",
                    "per-row scalar spread",
                    "row-rank Spearman correlation of scalar diagnostics",
                ],
            },
            "by_role": cross_seed,
        },
        "limitations": {
            "engineering_smoke_only": True,
            "row_count": len(rows),
            "two_row_smoke": len(rows) == 2,
            "two_row_smoke_is_retrieval_evidence": False,
            "retrieval_metrics_computed": False,
            "positive_negative_pairs_evaluated": False,
            "cross_content_generalization_tested": False,
            "formal_retrieval_evidence": False,
        },
        "decision": {
            "artifact_closure_validated": True,
            "diagnostic_completed": True,
            "representation_evidence_present": False,
            "representation_gate_passed": False,
            "renderer_probe_authorized": False,
            "editor_training_authorized": False,
            "next_step_authorized_by_this_audit": False,
        },
        "representation_gate_passed": False,
        "renderer_probe_authorized": False,
        "editor_training_authorized": False,
    }
    # Fail locally if any future edit accidentally introduces NaN/Infinity.
    _canonical_json(result)
    return result


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    if not path.parent.is_dir():
        raise R10BTangentSmokeAuditError(
            f"output parent directory is missing: {path.parent}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_pretty_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and scalar-audit an R10B Lucy/Bernini tangent smoke "
            "without comparing CountSketch coordinates across seeds."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = audit_published_smoke(args.input_dir)
    if args.output_json is not None:
        _atomic_write_json(args.output_json, result)
    print(_pretty_json(result), end="")


if __name__ == "__main__":
    main()

"""Analyze a committed R7-P0 preflight without changing its frozen gate.

This report is deliberately post-hoc and diagnostic.  It validates the
hash-bound final artifact first, then measures coverage, false events,
independent-audit joint pass rate, and simple action-vs-no-action
separability.  It never upgrades the frozen decision or authorizes generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from motive.r7_preflight_extract import _negative_type, validate_final


REPORT_SCHEMA = "motive-r7-p0-diagnostic-report-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None
    result = np.percentile(finite, [0, 25, 50, 75, 100])
    return {
        key: float(value)
        for key, value in zip(
            ("minimum", "q25", "median", "q75", "maximum"),
            result,
        )
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    first = np.asarray(positive, dtype=np.float64)
    second = np.asarray(negative, dtype=np.float64)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if len(first) == 0 or len(second) == 0:
        return None
    differences = first[:, None] - second[None, :]
    return float(
        np.mean(differences > 0.0)
        + 0.5 * np.mean(differences == 0.0)
    )


def _group_summary(
    mask: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    side: str,
) -> dict[str, Any]:
    base = np.asarray(arrays[f"{side}_base_valid"], dtype=bool)
    usable = np.asarray(arrays[f"{side}_usable"], dtype=bool)
    audit_available = np.asarray(
        arrays[f"{side}_audit_available"], dtype=bool
    )
    audit_pass = np.asarray(arrays[f"{side}_audit_pass"], dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    base_selected = selected & base
    count = int(selected.sum())
    base_count = int(base_selected.sum())
    usable_count = int((selected & usable).sum())
    audit_available_count = int(
        (base_selected & audit_available).sum()
    )
    audit_pass_count = int((base_selected & audit_pass).sum())
    actor_tracks = np.asarray(
        arrays[f"{side}_actor_track_mask"], dtype=bool
    ).sum(axis=1)
    trajectory = np.asarray(
        arrays[f"{side}_actor_trajectory"], dtype=np.float64
    )
    path_length = np.linalg.norm(
        np.diff(trajectory, axis=1),
        axis=2,
    ).sum(axis=1)
    phase_energy = np.asarray(
        arrays[f"{side}_phase_energy"], dtype=np.float64
    ).sum(axis=1)
    scalar_names = (
        "event_duration",
        "mean_visibility",
        "audit_event_iou",
        "audit_embedding_cosine",
        "audit_duration_relative_error",
        "audit_embedding_norm_relative_error",
        "audit_trajectory_rmse",
        "camera_crossfit_raw_median",
        "camera_crossfit_residual_reduction",
    )
    distributions = {
        name: _percentiles(
            np.asarray(arrays[f"{side}_{name}"])[base_selected]
        )
        for name in scalar_names
    }
    distributions.update(
        {
            "active_actor_tracks": _percentiles(
                actor_tracks[base_selected]
            ),
            "actor_path_length": _percentiles(
                path_length[base_selected]
            ),
            "phase_energy_sum": _percentiles(
                phase_energy[base_selected]
            ),
        }
    )
    return {
        "rows": count,
        "base_valid": base_count,
        "base_valid_fraction": _fraction(base_count, count),
        "screening_usable": usable_count,
        "screening_usable_fraction": _fraction(usable_count, count),
        "audit_available": audit_available_count,
        "audit_available_given_base": _fraction(
            audit_available_count,
            base_count,
        ),
        "audit_joint_pass": audit_pass_count,
        "audit_joint_pass_given_base": _fraction(
            audit_pass_count,
            base_count,
        ),
        "distributions_on_base_valid": distributions,
    }


def analyze_preflight(final_dir: Path) -> dict[str, Any]:
    resolved = final_dir.expanduser().resolve(strict=True)
    validated = validate_final(resolved)
    rows = validated["rows"]
    arrays = validated["arrays"]
    positive = np.asarray(arrays["positive"], dtype=bool)
    negative_types = np.asarray(
        [_negative_type(row) for row in rows],
        dtype=object,
    )
    no_action = ~positive & np.isin(
        negative_types,
        ("static", "endpoint_only"),
    )
    mismatch = ~positive & (negative_types == "instruction_mismatch")
    masks = {
        "positive": positive,
        "no_action_negative": no_action,
        "instruction_mismatch_negative": mismatch,
    }
    groups = {
        name: {
            side: _group_summary(mask, arrays, side=side)
            for side in ("source", "target")
        }
        for name, mask in masks.items()
    }

    target_trajectory = np.asarray(
        arrays["target_actor_trajectory"], dtype=np.float64
    )
    target_path_length = np.linalg.norm(
        np.diff(target_trajectory, axis=1),
        axis=2,
    ).sum(axis=1)
    target_energy = np.asarray(
        arrays["target_phase_energy"], dtype=np.float64
    ).sum(axis=1)
    target_tracks = np.asarray(
        arrays["target_actor_track_mask"], dtype=bool
    ).sum(axis=1).astype(np.float64)
    target_usable = np.asarray(arrays["target_usable"], dtype=bool)
    target_base = np.asarray(arrays["target_base_valid"], dtype=bool)
    auc_masks = {
        "positive": positive & target_base,
        "no_action": no_action & target_base,
    }
    separability = {
        "screening_usable_auc": _auc(
            target_usable[positive].astype(np.float64),
            target_usable[no_action].astype(np.float64),
        ),
        "actor_path_length_auc_on_base_valid": _auc(
            target_path_length[auc_masks["positive"]],
            target_path_length[auc_masks["no_action"]],
        ),
        "phase_energy_sum_auc_on_base_valid": _auc(
            target_energy[auc_masks["positive"]],
            target_energy[auc_masks["no_action"]],
        ),
        "active_actor_tracks_auc_on_base_valid": _auc(
            target_tracks[auc_masks["positive"]],
            target_tracks[auc_masks["no_action"]],
        ),
    }

    failures: dict[str, Counter[str]] = {
        "source": Counter(),
        "target": Counter(),
    }
    for row in rows:
        for side in ("source", "target"):
            if not bool(row[side]["usable"]):
                failures[side][
                    str(row[side].get("failure_reason") or "unknown")
                ] += 1

    signatures: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "target_usable": 0, "paired_usable": 0}
    )
    for index, row in enumerate(rows):
        if not positive[index]:
            continue
        signature = str(row.get("action_signature") or "unknown")
        signatures[signature]["rows"] += 1
        signatures[signature]["target_usable"] += int(
            target_usable[index]
        )
        signatures[signature]["paired_usable"] += int(
            row.get("paired_usable") is True
        )

    def example(index: int) -> dict[str, Any]:
        return {
            "iid": rows[index]["iid"],
            "action_signature": rows[index].get("action_signature"),
            "negative_type": _negative_type(rows[index]),
            "target_usable": bool(target_usable[index]),
            "target_base_valid": bool(target_base[index]),
            "target_actor_path_length": float(target_path_length[index]),
            "target_phase_energy_sum": float(target_energy[index]),
            "target_active_actor_tracks": int(target_tracks[index]),
            "target_failure_reason": rows[index]["target"].get(
                "failure_reason"
            ),
        }

    false_event_indices = np.flatnonzero(no_action & target_usable)
    rejected_positive_indices = np.flatnonzero(positive & ~target_usable)
    false_event_indices = sorted(
        false_event_indices.tolist(),
        key=lambda index: (-target_path_length[index], rows[index]["iid"]),
    )
    rejected_positive_indices = sorted(
        rejected_positive_indices.tolist(),
        key=lambda index: (
            target_path_length[index],
            rows[index]["iid"],
        ),
    )

    gate = validated["summary"]["gate"]
    audit_eligible = int((positive & target_base).sum())
    audit_joint_passed = int(
        (
            positive
            & target_base
            & np.asarray(arrays["target_audit_pass"], dtype=bool)
        ).sum()
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "artifact": {
            "final_dir": str(resolved),
            "done_sha256": _sha256(resolved / "done.json"),
            "summary_sha256": _sha256(resolved / "summary.json"),
            "manifest_sha256": _sha256(resolved / "manifest.jsonl"),
            "archive_sha256": _sha256(resolved / "features.npz"),
        },
        "rows": len(rows),
        "label_counts": {
            "positive": int(positive.sum()),
            "no_action_negative": int(no_action.sum()),
            "instruction_mismatch_negative": int(mismatch.sum()),
        },
        "groups": groups,
        "target_action_vs_no_action_separability": separability,
        "failure_reasons": {
            side: dict(sorted(counts.items()))
            for side, counts in failures.items()
        },
        "positive_action_signature_coverage": {
            key: value
            for key, value in sorted(signatures.items())
        },
        "examples": {
            "no_action_false_events": [
                example(index) for index in false_event_indices[:20]
            ],
            "rejected_positives_lowest_path_length": [
                example(index) for index in rejected_positive_indices[:20]
            ],
        },
        "frozen_gate": gate,
        "posthoc_joint_audit": {
            "eligible_positive_base_valid": audit_eligible,
            "joint_passed": audit_joint_passed,
            "joint_pass_fraction": _fraction(
                audit_joint_passed,
                audit_eligible,
            ),
            "note": (
                "The frozen v2 gate reports marginal medians but does not "
                "gate this joint audit-pass fraction.  This post-hoc metric "
                "cannot change the immutable v2 artifact."
            ),
        },
        "decision": {
            "teacher_feature_ready": False,
            "reason": (
                "Use the immutable gate decision; post-hoc metrics are "
                "diagnostic only."
            ),
            "formal_status": "INSUFFICIENT",
            "production_decision": False,
            "generation_authorized": False,
        },
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_preflight(args.final_dir)
    _atomic_json(args.output_json, report)
    print(
        "[motive-r7-p0-report] "
        f"rows={report['rows']} "
        f"gate={report['frozen_gate']['diagnostic_status']} "
        f"output={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

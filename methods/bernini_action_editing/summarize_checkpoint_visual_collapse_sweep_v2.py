#!/usr/bin/env python3
"""Aggregate exact-81 collapse-gate reports across checkpoints and samples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from .checkpoint_visual_collapse_gate_v2 import (
        SCHEMA_VERSION as GATE_SCHEMA_VERSION,
        write_json_atomic,
    )
except ImportError:
    from checkpoint_visual_collapse_gate_v2 import (  # type: ignore
        SCHEMA_VERSION as GATE_SCHEMA_VERSION,
        write_json_atomic,
    )


SCHEMA_VERSION = "bernini-checkpoint-visual-collapse-sweep-summary-v2"


class SweepSummaryError(RuntimeError):
    """Raised when gate reports do not form an auditable sweep."""


def _report_key(report: Mapping[str, Any]) -> tuple[int, str]:
    metadata = report.get("metadata")
    if not isinstance(metadata, Mapping):
        raise SweepSummaryError("gate report metadata is missing")
    step = metadata.get("checkpoint_step")
    sample_id = metadata.get("sample_id")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise SweepSummaryError("checkpoint_step must be a non-negative integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise SweepSummaryError("sample_id must be a non-empty string")
    return step, sample_id


def summarize_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    expected_sample_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Build a fail-closed checkpoint matrix and collapse frontier."""

    values = [dict(report) for report in reports]
    if not values:
        raise SweepSummaryError("at least one gate report is required")
    expected = None
    if expected_sample_ids is not None:
        expected = list(expected_sample_ids)
        if not expected or len(set(expected)) != len(expected):
            raise SweepSummaryError("expected sample IDs must be non-empty and unique")

    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    for report in values:
        if report.get("schema_version") != GATE_SCHEMA_VERSION:
            raise SweepSummaryError("gate report schema differs")
        key = _report_key(report)
        if key in keyed:
            raise SweepSummaryError(
                f"duplicate checkpoint/sample report: step={key[0]} sample={key[1]}"
            )
        keyed[key] = report

    observed_sample_ids = sorted({sample_id for _, sample_id in keyed})
    if expected is None:
        expected = observed_sample_ids
    unexpected = sorted(set(observed_sample_ids) - set(expected))
    if unexpected:
        raise SweepSummaryError(
            "reports contain unexpected sample IDs: " + ", ".join(unexpected)
        )

    checkpoint_rows = []
    all_failure_codes: Counter[str] = Counter()
    for step in sorted({checkpoint_step for checkpoint_step, _ in keyed}):
        rows = [keyed[(step, sample_id)] for sample_id in expected if (step, sample_id) in keyed]
        present_ids = [str(row["metadata"]["sample_id"]) for row in rows]
        missing_ids = [sample_id for sample_id in expected if sample_id not in present_ids]
        pass_count = sum(int(row.get("passed") is True) for row in rows)
        fail_count = sum(int(row.get("passed") is not True) for row in rows)
        collapse_count = sum(int(row.get("collapsed") is True) for row in rows)
        error_count = sum(int(row.get("status") == "error") for row in rows)
        failure_codes: Counter[str] = Counter()
        sample_rows = []
        for row in rows:
            codes = row.get("failure_codes")
            if not isinstance(codes, list) or not all(
                isinstance(code, str) for code in codes
            ):
                raise SweepSummaryError("gate report failure_codes are malformed")
            failure_codes.update(codes)
            all_failure_codes.update(codes)
            sample_rows.append(
                {
                    "sample_id": row["metadata"]["sample_id"],
                    "status": row.get("status"),
                    "passed": row.get("passed") is True,
                    "collapsed": row.get("collapsed"),
                    "failure_codes": codes,
                    "calibration_fingerprint": row.get("calibration", {}).get(
                        "calibration_fingerprint"
                    ),
                }
            )
        complete = not missing_ids
        fully_passing = bool(complete and rows and pass_count == len(expected))
        any_collapse = collapse_count > 0
        all_collapsed = bool(
            complete and rows and collapse_count == len(expected)
        )
        checkpoint_rows.append(
            {
                "checkpoint_step": step,
                "checkpoint_label": f"checkpoint-{step:08d}",
                "expected_sample_count": len(expected),
                "observed_sample_count": len(rows),
                "missing_sample_ids": missing_ids,
                "complete": complete,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "collapse_count": collapse_count,
                "error_count": error_count,
                "fully_passing": fully_passing,
                "any_collapse": any_collapse,
                "all_collapsed": all_collapsed,
                "failure_code_counts": dict(sorted(failure_codes.items())),
                "samples": sample_rows,
            }
        )

    complete_steps = [row for row in checkpoint_rows if row["complete"]]
    fully_passing_steps = [
        row["checkpoint_step"] for row in checkpoint_rows if row["fully_passing"]
    ]
    any_collapse_steps = [
        row["checkpoint_step"] for row in checkpoint_rows if row["any_collapse"]
    ]
    all_collapse_steps = [
        row["checkpoint_step"] for row in checkpoint_rows if row["all_collapsed"]
    ]
    incomplete_steps = [
        row["checkpoint_step"] for row in checkpoint_rows if not row["complete"]
    ]

    fingerprint_values = sorted(
        {
            row.get("calibration", {}).get("calibration_fingerprint")
            for row in values
            if row.get("calibration", {}).get("calibration_fingerprint") is not None
        }
    )
    calibration_consistent = len(fingerprint_values) == 1 and all(
        row.get("calibration", {}).get("calibration_fingerprint")
        == fingerprint_values[0]
        for row in values
        if row.get("status") != "error"
    )
    complete = bool(complete_steps and not incomplete_steps)
    sweep_passed = bool(
        complete
        and calibration_consistent
        and len(fully_passing_steps) == len(checkpoint_rows)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if sweep_passed else "fail",
        "passed": sweep_passed,
        "fail_closed": True,
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "expected_sample_ids": expected,
        "expected_sample_count": len(expected),
        "checkpoint_count": len(checkpoint_rows),
        "report_count": len(values),
        "all_checkpoints_complete": complete,
        "incomplete_checkpoint_steps": incomplete_steps,
        "calibration_consistent": calibration_consistent,
        "calibration_fingerprints": fingerprint_values,
        "frontier": {
            "fully_passing_checkpoint_steps": fully_passing_steps,
            "last_fully_passing_checkpoint_step": (
                max(fully_passing_steps) if fully_passing_steps else None
            ),
            "first_any_collapse_checkpoint_step": (
                min(any_collapse_steps) if any_collapse_steps else None
            ),
            "first_all_collapsed_checkpoint_step": (
                min(all_collapse_steps) if all_collapse_steps else None
            ),
        },
        "failure_code_counts": dict(sorted(all_failure_codes.items())),
        "checkpoints": checkpoint_rows,
    }


def _load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve(strict=True).read_text("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SweepSummaryError(f"could not read gate report: {path}") from error
    if not isinstance(value, dict):
        raise SweepSummaryError(f"gate report root is not an object: {path}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--expected-sample-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = summarize_reports(
            [_load_report(path) for path in args.reports],
            expected_sample_ids=(
                args.expected_sample_id if args.expected_sample_id else None
            ),
        )
        write_json_atomic(args.output, report)
        return 0 if report["passed"] else 2
    except (OSError, ValueError, SweepSummaryError) as error:
        raise SystemExit(str(error))


__all__ = [
    "SCHEMA_VERSION",
    "SweepSummaryError",
    "summarize_reports",
]


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed validation for Lucy action-routing training artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .finalize_action_experiment import parse_training_log


SCHEMA = "motive-action-training-validation-v1"


def _iter_tensors(value: Any, prefix: str = "checkpoint") -> Iterable[tuple[str, Any]]:
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - remote runtime contract
        raise RuntimeError("PyTorch is required to validate a checkpoint") from error

    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{prefix}[{index}]")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_action_training(
    *,
    log_path: Path,
    checkpoint_path: Path,
    expected_step: int,
    max_total_grad: float,
    max_transformer_grad: float,
    max_router_grad: float,
    allow_legacy_group_gradients: bool = False,
) -> dict[str, object]:
    import torch

    records = [dict(record) for record in parse_training_log(log_path)]
    legacy_group_gradients_derived = False
    if allow_legacy_group_gradients:
        for record in records:
            if "grad" not in record:
                continue
            if "gradl" not in record:
                record["gradl"] = record["grad"]
                legacy_group_gradients_derived = True
            if "gradr" not in record:
                record["gradr"] = 0.0
                legacy_group_gradients_derived = True
    checks: dict[str, bool] = {
        "log_present": log_path.is_file() and log_path.stat().st_size > 0,
        "records_present": bool(records),
        "last_step_match": bool(records and records[-1].get("step") == expected_step),
        "all_logged_scalars_finite": all(
            math.isfinite(float(value))
            for record in records
            for value in record.values()
            if isinstance(value, (int, float))
        ),
        "gradient_fields_present": bool(records)
        and all(
            all(key in record for key in ("grad", "gradl", "gradr"))
            for record in records
        ),
    }

    maxima: dict[str, float | None] = {}
    for key, limit in (
        ("grad", max_total_grad),
        ("gradl", max_transformer_grad),
        ("gradr", max_router_grad),
    ):
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
        ]
        maximum = max(values) if values else None
        maxima[key] = maximum
        checks[f"{key}_within_limit"] = bool(
            maximum is not None
            and math.isfinite(maximum)
            and maximum <= float(limit)
        )

    checks["checkpoint_present"] = (
        checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0
    )
    nonfinite_tensors: list[str] = []
    floating_tensors = 0
    total_tensors = 0
    checkpoint_step = None
    if checks["checkpoint_present"]:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as error:
            raise RuntimeError(
                "checkpoint validation requires torch.load(weights_only=True)"
            ) from error
        checkpoint_step = payload.get("step") if isinstance(payload, dict) else None
        for name, tensor in _iter_tensors(payload):
            total_tensors += 1
            if tensor.is_floating_point():
                floating_tensors += 1
                if not bool(torch.isfinite(tensor).all()):
                    nonfinite_tensors.append(name)
    checks["checkpoint_step_match"] = checkpoint_step == expected_step
    checks["checkpoint_tensors_finite"] = not nonfinite_tensors

    return {
        "schema": SCHEMA,
        "complete": all(checks.values()),
        "log_path": str(log_path),
        "checkpoint_path": str(checkpoint_path),
        "expected_step": int(expected_step),
        "checks": checks,
        "gradient_maxima": maxima,
        "gradient_limits": {
            "grad": float(max_total_grad),
            "gradl": float(max_transformer_grad),
            "gradr": float(max_router_grad),
        },
        "checkpoint_step": checkpoint_step,
        "checkpoint_total_tensors": total_tensors,
        "checkpoint_floating_tensors": floating_tensors,
        "checkpoint_nonfinite_tensors": nonfinite_tensors[:32],
        "legacy_group_gradients_derived": legacy_group_gradients_derived,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--expected-step", required=True, type=int)
    parser.add_argument("--max-total-grad", default=100.0, type=float)
    parser.add_argument("--max-transformer-grad", default=100.0, type=float)
    parser.add_argument("--max-router-grad", default=100.0, type=float)
    parser.add_argument(
        "--allow-legacy-group-gradients",
        action="store_true",
        help=(
            "For a reused no-router baseline, derive gradl=grad and gradr=0 "
            "when the historical log predates grouped gradient logging."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_action_training(
        log_path=args.log_path,
        checkpoint_path=args.checkpoint_path,
        expected_step=args.expected_step,
        max_total_grad=args.max_total_grad,
        max_transformer_grad=args.max_transformer_grad,
        max_router_grad=args.max_router_grad,
        allow_legacy_group_gradients=args.allow_legacy_group_gradients,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["complete"]:
        failed = [
            name for name, passed in report["checks"].items() if not passed
        ]
        raise SystemExit("action training validation failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

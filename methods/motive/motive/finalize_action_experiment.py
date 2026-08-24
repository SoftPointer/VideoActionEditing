"""Collect representation and Lucy action-screen artifacts into one report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


SCHEMA = "motive-action-experiment-summary-v1"
ARMS = (
    "e1_plain_lora",
    "e2_fixed_random",
    "e2_random_router",
    "e3_motive_frozen",
)
PAIR_PATTERN = re.compile(
    r"(?<![\w/])([A-Za-z][A-Za-z0-9_]*)="
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
)
INTEGER_PATTERN = re.compile(r"^-?\d+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_training_log(path: Path) -> list[dict[str, object]]:
    """Parse the compact per-step scalar line emitted by lucy.train."""

    if not path.is_file():
        return []
    by_step: dict[int, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "step=" not in line or " loss=" not in line:
            continue
        pairs: dict[str, object] = {}
        for name, raw_value in PAIR_PATTERN.findall(line):
            value: object
            if INTEGER_PATTERN.fullmatch(raw_value):
                value = int(raw_value)
            else:
                value = float(raw_value)
            pairs[name] = value
        step = pairs.get("step")
        loss = pairs.get("loss")
        if not isinstance(step, int) or not isinstance(loss, (int, float)):
            continue
        pairs["step"] = step
        by_step[step] = pairs
    return [by_step[step] for step in sorted(by_step)]


def _finite_curve(records: Sequence[dict[str, object]]) -> bool:
    for record in records:
        for value in record.values():
            if isinstance(value, float) and not math.isfinite(value):
                return False
    return True


def _curve_summary(records: Sequence[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "points": len(records),
        "finite": _finite_curve(records),
        "first_step": records[0]["step"] if records else None,
        "last_step": records[-1]["step"] if records else None,
    }
    for key in (
        "loss",
        "diffusion",
        "ledit",
        "lpres",
        "tres",
        "gact",
        "gent",
        "gmean",
        "grs",
        "grad",
        "gradl",
        "gradr",
        "samples_per_s",
    ):
        values = [
            float(record[key])
            for record in records
            if isinstance(record.get(key), (int, float))
            and math.isfinite(float(record[key]))
        ]
        if values:
            summary[key] = {
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
            }
    return summary


def _representation_summary(run_root: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for seed in (2026, 2027, 2028):
        directory = run_root / "prep" / f"repr_seed_{seed}"
        metrics_path = directory / "metrics.json"
        checkpoint = directory / "prompt_action_encoder.pt"
        sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
        entry: dict[str, object] = {
            "complete": all(
                path.is_file() and path.stat().st_size > 0
                for path in (metrics_path, checkpoint, sidecar)
            ),
            "directory": str(directory),
        }
        if metrics_path.is_file():
            metrics = _load_json(metrics_path)
            entry["metrics_sha256"] = _sha256(metrics_path)
            entry["initial_loss"] = metrics.get("initial_loss")
            entry["final_loss"] = metrics.get("final_loss")
            entry["loss_history"] = metrics.get("loss_history")
            entry["train"] = metrics.get("train")
            entry["validation"] = metrics.get("validation")
            entry["test"] = metrics.get("test")
            entry["shortcut_baselines"] = metrics.get("shortcut_baselines")
        if checkpoint.is_file():
            entry["checkpoint_sha256"] = _sha256(checkpoint)
            entry["checkpoint_size"] = checkpoint.stat().st_size
        if sidecar.is_file():
            entry["sidecar_sha256"] = _sha256(sidecar)
        result[str(seed)] = entry
    return result


def _arm_summary(
    run_root: Path,
    arm: str,
    *,
    seed: int,
    expected_step: int,
    expected_processes: int,
    expected_global_batch: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    output_dir = run_root / "lucy" / arm / f"seed_{seed}"
    log_path = output_dir.with_suffix(".log")
    status_path = run_root / "status" / f"{arm}_seed_{seed}.json"
    config_path = output_dir / "run_config.json"
    validation_path = output_dir / "training_validation.json"
    checkpoint = output_dir / f"checkpoint_step_{expected_step:06d}.pt"
    records = parse_training_log(log_path)
    status = _load_json(status_path) if status_path.is_file() else None
    config = _load_json(config_path) if config_path.is_file() else None
    validation = (
        _load_json(validation_path) if validation_path.is_file() else None
    )
    rng_sidecars = sorted(
        output_dir.glob(
            f"checkpoint_step_{expected_step:06d}.rng_rank_*.pt"
        )
    )
    checks = {
        "status_succeeded": bool(status and status.get("state") == "succeeded"),
        "run_config_present": config is not None,
        "num_processes_match": bool(
            config and config.get("num_processes") == expected_processes
        ),
        "global_batch_match": bool(
            config
            and config.get("effective_global_batch_size")
            == expected_global_batch
        ),
        "loss_curve_present": bool(records),
        "loss_curve_finite": _finite_curve(records),
        "last_step_match": bool(
            records and records[-1].get("step") == expected_step
        ),
        "checkpoint_present": checkpoint.is_file()
        and checkpoint.stat().st_size > 0,
        "rng_sidecars_complete": len(rng_sidecars) == expected_processes
        and all(path.stat().st_size > 0 for path in rng_sidecars),
        "training_validation_complete": bool(
            validation and validation.get("complete") is True
        ),
    }
    entry: dict[str, object] = {
        "arm": arm,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "status": status,
        "training_validation": validation,
        "checks": checks,
        "complete": all(checks.values()),
        "loss": _curve_summary(records),
        "rng_sidecars": len(rng_sidecars),
    }
    if log_path.is_file():
        entry["log_sha256"] = _sha256(log_path)
    if config_path.is_file():
        entry["run_config_sha256"] = _sha256(config_path)
    if validation_path.is_file():
        entry["training_validation_sha256"] = _sha256(validation_path)
    if checkpoint.is_file():
        entry["checkpoint_sha256"] = _sha256(checkpoint)
        entry["checkpoint_size"] = checkpoint.stat().st_size
    return entry, records


def finalize(
    run_root: Path,
    *,
    seed: int,
    expected_step: int,
    expected_processes: int,
    expected_global_batch: int,
    allow_incomplete: bool,
    overwrite: bool,
) -> dict[str, object]:
    run_root = run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"missing run root: {run_root}")
    analysis_dir = run_root / "analysis"
    final_path = analysis_dir / "final_summary.json"
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite final report: {final_path}")

    representation = _representation_summary(run_root)
    prep_path = run_root / "prep" / "summary.json"
    arms: dict[str, object] = {}
    for arm in ARMS:
        arm_summary, records = _arm_summary(
            run_root,
            arm,
            seed=seed,
            expected_step=expected_step,
            expected_processes=expected_processes,
            expected_global_batch=expected_global_batch,
        )
        arms[arm] = arm_summary
        _atomic_jsonl(analysis_dir / f"{arm}_losses.jsonl", records)

    representation_complete = all(
        bool(entry.get("complete"))
        for entry in representation.values()
        if isinstance(entry, dict)
    )
    arms_complete = all(
        bool(entry.get("complete"))
        for entry in arms.values()
        if isinstance(entry, dict)
    )
    complete = bool(
        prep_path.is_file()
        and representation_complete
        and arms_complete
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "scientific_scope": (
            "pseudo-label wiring/training-stability pilot; not a held-out "
            "action-editing quality or production-label experiment"
        ),
        "expected": {
            "seed": seed,
            "optimizer_step": expected_step,
            "processes_per_arm": expected_processes,
            "effective_global_batch_size": expected_global_batch,
        },
        "prep": {
            "present": prep_path.is_file(),
            "sha256": _sha256(prep_path) if prep_path.is_file() else None,
            "summary": _load_json(prep_path) if prep_path.is_file() else None,
        },
        "representation": representation,
        "arms": arms,
        "complete": complete,
    }
    _atomic_json(final_path, payload)
    final_digest = _sha256(final_path)
    _atomic_json(
        analysis_dir / "FINALIZED.json",
        {
            "schema": "motive-action-experiment-finalized-v1",
            "complete": complete,
            "final_summary_sha256": final_digest,
        },
    )
    if not complete and not allow_incomplete:
        raise RuntimeError(
            f"action experiment is incomplete; report written to {final_path}"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--expected-step", default=100, type=int)
    parser.add_argument("--expected-processes", default=8, type=int)
    parser.add_argument("--expected-global-batch", default=8, type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = finalize(
        args.run_root,
        seed=args.seed,
        expected_step=args.expected_step,
        expected_processes=args.expected_processes,
        expected_global_batch=args.expected_global_batch,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "complete": payload["complete"],
                "run_root": payload["run_root"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

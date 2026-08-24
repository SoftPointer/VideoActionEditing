#!/usr/bin/env python3
"""Audit the exact first-N SPT oracle teachers without an optimizer step."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
for root in (HERE, METHOD_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import phase_transport as spt  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_student as student  # noqa: E402


SCHEMA = "bernini-spt-v2-oracle-prefix-audit-v1"


class OraclePrefixAuditError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a fixed SPT oracle prefix")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-rows", type=int, default=8)
    parser.add_argument("--teacher-feature-channels", type=int, default=64)
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--teacher-generate-threshold", type=float, default=0.35)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    return parser


def _mse(left: Any, right: Any) -> float:
    import torch

    return float(torch.mean((left.float() - right.float()) ** 2).item())


def _row_report(
    *,
    row_index: int,
    row: Mapping[str, Any],
    source: Any,
    target: Any,
    oracle: spt.PhasePlan,
) -> dict[str, Any]:
    proxy = spt.make_proxy_target(source, target, oracle)
    source_mse = _mse(source, target)
    proxy_mse = _mse(proxy, target)
    improvement = (source_mse - proxy_mse) / max(source_mse, 1e-12)
    diagnostics = dict(oracle.diagnostics or {})
    report = {
        "row_index": int(row_index),
        "iid": str(row.get("iid", row.get("id", ""))),
        "identity_sha256": legacy.dataset_identity(row, row_index),
        "gate_fraction": {
            "preserve": float(oracle.gate_probs[:, spt.GATE_PRESERVE].float().mean().item()),
            "transport": float(oracle.gate_probs[:, spt.GATE_TRANSPORT].float().mean().item()),
            "generate": float(oracle.gate_probs[:, spt.GATE_GENERATE].float().mean().item()),
        },
        "generate_budget": {
            name: diagnostics[name]
            for name in (
                "prebudget_generate_fraction",
                "postbudget_generate_fraction",
                "budget_reject_fraction",
                "budget_reject_fraction_of_prebudget_generate",
                "max_generate_fraction_per_phase",
                "observed_max_prebudget_generate_fraction_per_phase",
                "observed_max_postbudget_generate_fraction_per_phase",
                "generate_budget_score",
                "generate_budget_selection",
                "generate_budget_reject_fallback",
            )
        },
        "mse": {
            "source_to_target": source_mse,
            "proxy_to_target": proxy_mse,
            "proxy_to_source": _mse(proxy, source),
        },
        "proxy_relative_improvement_over_copy": float(improvement),
        "rejection_fraction": {
            name: float(diagnostics[name])
            for name in (
                "valid_reject_fraction",
                "margin_reject_fraction",
                "cycle_reject_fraction",
                "cycle_inconsistent_fraction",
            )
        },
        "cost": {
            name: float(diagnostics[name])
            for name in (
                "mean_zero_cost",
                "mean_best_nonzero_cost",
                "mean_absolute_zero_improvement",
                "mean_relative_zero_improvement",
            )
        },
        "mean_absolute_offset_cells": {
            "dt": float(oracle.offsets[:, 0].float().abs().mean().item()),
            "dy": float(oracle.offsets[:, 1].float().abs().mean().item()),
            "dx": float(oracle.offsets[:, 2].float().abs().mean().item()),
        },
        "hard_executor_candidate_mse": float(
            diagnostics["hard_executor_candidate_mse"]
        ),
    }
    numeric = [
        source_mse,
        proxy_mse,
        improvement,
        *report["gate_fraction"].values(),
        *report["rejection_fraction"].values(),
        *report["cost"].values(),
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise OraclePrefixAuditError(f"row {row_index} has a non-finite audit metric")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prefix_rows != 8:
        raise OraclePrefixAuditError("this diagnostic is bound to the first eight rows")
    if args.teacher_feature_channels != 64:
        raise OraclePrefixAuditError("oracle prefix audit requires all 64 packed channels")
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    distributed = legacy.distributed_contract()
    device, backend = legacy.initialise_distributed(distributed)
    import torch.distributed as dist

    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
    )
    membership = student._training_membership(dataset, args.prefix_rows)
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    config = spt.PhaseTransportConfig(
        latent_channels=64,
        teacher_temperature=args.teacher_temperature,
        teacher_generate_threshold=args.teacher_generate_threshold,
    )
    local_reports = []
    for row_index in range(distributed.rank, args.prefix_rows, distributed.world_size):
        row = dataset[row_index]
        source, target = student._clean_pair(row, vae_mean, vae_std, z_dim, device)
        oracle = spt.build_oracle_plan(
            source,
            target,
            config,
            feature_channels=args.teacher_feature_channels,
        )
        local_reports.append(
            _row_report(
                row_index=row_index,
                row=row,
                source=source,
                target=target,
                oracle=oracle,
            )
        )
    if dist.is_available() and dist.is_initialized():
        gathered: list[Optional[list[dict[str, Any]]]] = [None] * distributed.world_size
        dist.all_gather_object(gathered, local_reports)
        reports = [report for rank_reports in gathered if rank_reports for report in rank_reports]
    else:
        reports = local_reports
    reports.sort(key=lambda report: report["row_index"])
    if [report["row_index"] for report in reports] != list(range(args.prefix_rows)):
        raise OraclePrefixAuditError("distributed audit did not cover rows 0..7 exactly once")
    for report, member in zip(reports, membership["members"]):
        if (
            report["iid"] != member["iid"]
            or report["identity_sha256"] != member["identity_sha256"]
        ):
            raise OraclePrefixAuditError("audit report differs from bound membership")

    output = Path(args.output).expanduser().resolve()
    if distributed.rank == 0:
        if output.exists():
            raise OraclePrefixAuditError(f"refusing to overwrite {output}")
        gate_names = ("preserve", "transport", "generate")
        aggregate = {
            name: sum(report["gate_fraction"][name] for report in reports) / len(reports)
            for name in gate_names
        }
        payload: dict[str, Any] = {
            "schema_version": SCHEMA,
            "backend": backend,
            "world_size": distributed.world_size,
            "optimizer_steps": 0,
            "optimizer_exists": False,
            "target_used_by_training_oracle_only": True,
            "dataset": {
                "signature": dataset.signature,
                "summary": dataset_summary,
                "membership": membership,
            },
            "config": {
                "feature_channels": args.teacher_feature_channels,
                "generate_threshold": args.teacher_generate_threshold,
                "transport_margin": config.teacher_transport_margin,
                "cycle_gate": config.teacher_require_cycle,
                "max_generate_fraction_per_phase": config.max_generate_fraction_per_phase,
                "generate_budget_reject_fallback": "preserve",
            },
            "mean_gate_fraction": aggregate,
            "rows": reports,
            "method_files_sha256": {
                "phase_transport.py": legacy.file_sha256(HERE / "phase_transport.py"),
                "audit_oracle_prefix.py": legacy.file_sha256(Path(__file__).resolve()),
            },
        }
        payload["audit_digest"] = legacy.object_sha256(payload)
        student._atomic_json(output, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

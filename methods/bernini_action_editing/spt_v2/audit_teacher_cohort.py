#!/usr/bin/env python3
"""Read-only trust audit and cohort selector for the hardened SPT teacher.

The scanner never constructs Bernini, an optimizer, or a planner.  Four ROCm
ranks evaluate disjoint dataset rows with the exact paired-latent SPT oracle,
then rank zero writes an immutable full audit and a minimal membership file.
The membership format is intentionally strict-loadable by a later planner or
joint trainer, but this module does not change either trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
METHOD_ROOT = HERE.parent
for _root in (HERE, METHOD_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import phase_transport as spt  # noqa: E402
import train_lora as legacy  # noqa: E402
import train_student as student  # noqa: E402


AUDIT_SCHEMA = "bernini-spt-v2-teacher-trust-cohort-audit-v1"
MEMBERSHIP_SCHEMA = "bernini-spt-v2-teacher-trust-membership-v1"
SELECTION_ALGORITHM = "four-conjunctive-hard-thresholds-v1"
HARD_ORACLE_GENERATE_BUDGET = 0.12
TEACHER_FEATURE_CHANNELS = 64
DEFAULT_PREFIX_ROWS = 64
DEFAULT_MINIMUM_SELECTED = 8


class TeacherCohortAuditError(RuntimeError):
    """Raised before accepting an ambiguous audit or membership."""


@dataclass(frozen=True)
class ScanWindow:
    mode: str
    start: int
    stop: int

    @property
    def row_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))

    def receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "start": self.start,
            "stop_exclusive": self.stop,
            "row_count": self.stop - self.start,
        }


@dataclass(frozen=True)
class SelectorThresholds:
    max_prebudget_generate_fraction: float = 0.25
    min_postbudget_transport_fraction: float = 0.03
    min_proxy_relative_improvement_over_copy: float = 0.40
    max_postbudget_generate_fraction_per_phase: float = 0.12

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise TeacherCohortAuditError(f"{name} must lie in [0,1]")
        if (
            self.max_postbudget_generate_fraction_per_phase
            > HARD_ORACLE_GENERATE_BUDGET
        ):
            raise TeacherCohortAuditError(
                "postbudget per-phase Generate threshold cannot exceed 0.12"
            )

    def receipt(self) -> dict[str, float]:
        self.validate()
        return {name: float(value) for name, value in asdict(self).items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and select a trusted hardened-SPT teacher cohort"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    scan = parser.add_mutually_exclusive_group()
    scan.add_argument("--prefix-rows", type=int, default=None)
    scan.add_argument(
        "--row-range", type=int, nargs=2, metavar=("START", "STOP_EXCLUSIVE")
    )
    parser.add_argument(
        "--max-prebudget-generate-fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--min-postbudget-transport-fraction", type=float, default=0.03
    )
    parser.add_argument(
        "--min-proxy-relative-improvement-over-copy", type=float, default=0.40
    )
    parser.add_argument(
        "--max-postbudget-generate-fraction-per-phase",
        type=float,
        default=HARD_ORACLE_GENERATE_BUDGET,
    )
    parser.add_argument(
        "--minimum-selected", type=int, default=DEFAULT_MINIMUM_SELECTED
    )
    parser.add_argument("--allow-insufficient-selection", action="store_true")
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--teacher-generate-threshold", type=float, default=0.35)
    parser.add_argument("--teacher-transport-margin", type=float, default=0.05)
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def selector_thresholds(args: argparse.Namespace) -> SelectorThresholds:
    value = SelectorThresholds(
        max_prebudget_generate_fraction=float(
            args.max_prebudget_generate_fraction
        ),
        min_postbudget_transport_fraction=float(
            args.min_postbudget_transport_fraction
        ),
        min_proxy_relative_improvement_over_copy=float(
            args.min_proxy_relative_improvement_over_copy
        ),
        max_postbudget_generate_fraction_per_phase=float(
            args.max_postbudget_generate_fraction_per_phase
        ),
    )
    value.validate()
    return value


def validate_cli(args: argparse.Namespace) -> SelectorThresholds:
    thresholds = selector_thresholds(args)
    if type(args.minimum_selected) is not int or args.minimum_selected <= 0:
        raise TeacherCohortAuditError("minimum-selected must be positive")
    for name in (
        "teacher_temperature",
        "teacher_generate_threshold",
        "teacher_transport_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise TeacherCohortAuditError(f"{name} must be finite and positive")
    if re.fullmatch(r"[0-9a-fA-F]{40}", args.method_source_revision) is None:
        raise TeacherCohortAuditError("method-source-revision must be a full SHA-1")
    if re.fullmatch(r"[0-9a-f]{64}", args.method_source_archive_sha256) is None:
        raise TeacherCohortAuditError(
            "method-source-archive-sha256 must be a lowercase SHA-256"
        )
    output = Path(args.output_dir).expanduser()
    if not output.is_absolute():
        raise TeacherCohortAuditError("output-dir must be absolute")
    return thresholds


def resolve_scan_window(args: argparse.Namespace, dataset_rows: int) -> ScanWindow:
    if type(dataset_rows) is not int or dataset_rows <= 0:
        raise TeacherCohortAuditError("dataset row count must be positive")
    if args.row_range is not None:
        start, stop = args.row_range
        mode = "explicit_half_open_range"
    else:
        start = 0
        stop = DEFAULT_PREFIX_ROWS if args.prefix_rows is None else args.prefix_rows
        mode = "ordered_prefix"
    if (
        type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
        or stop > dataset_rows
    ):
        raise TeacherCohortAuditError(
            f"scan window [{start},{stop}) lies outside dataset rows [0,{dataset_rows})"
        )
    if args.minimum_selected > stop - start:
        raise TeacherCohortAuditError(
            "minimum-selected cannot exceed the number of scanned rows"
        )
    return ScanWindow(mode=mode, start=start, stop=stop)


def _iid(row: Mapping[str, Any]) -> str:
    value = row.get("iid", row.get("id"))
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise TeacherCohortAuditError("each audited row requires a stable IID")
    return value


def _mse(left: Any, right: Any) -> float:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        raise TeacherCohortAuditError("MSE tensors have different shapes")
    value = float(torch.mean((left.float() - right.float()) ** 2).item())
    if not math.isfinite(value) or value < 0.0:
        raise TeacherCohortAuditError("MSE is non-finite or negative")
    return value


def selection_decision(
    report: Mapping[str, Any], thresholds: SelectorThresholds
) -> dict[str, Any]:
    """Return four explicit booleans; selection never uses a hidden score."""

    thresholds.validate()
    try:
        checks = {
            "prebudget_generate_fraction": float(
                report["prebudget_generate_fraction"]
            )
            <= thresholds.max_prebudget_generate_fraction,
            "postbudget_transport_fraction": float(
                report["postbudget_gate_fraction"]["transport"]
            )
            >= thresholds.min_postbudget_transport_fraction,
            "proxy_relative_improvement_over_copy": float(
                report["proxy_relative_improvement_over_copy"]
            )
            >= thresholds.min_proxy_relative_improvement_over_copy,
            "postbudget_generate_fraction_per_phase": float(
                report["observed_max_postbudget_generate_fraction_per_phase"]
            )
            <= thresholds.max_postbudget_generate_fraction_per_phase + 1.0e-6,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise TeacherCohortAuditError(
            f"row report cannot be selected strictly: {error}"
        ) from error
    selected = all(checks.values())
    return {
        "selected": selected,
        "criteria": checks,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }


def row_report(
    *,
    row_index: int,
    row: Mapping[str, Any],
    source: Any,
    target: Any,
    oracle: spt.PhasePlan,
    thresholds: SelectorThresholds,
) -> dict[str, Any]:
    """Measure one oracle without exposing target data to any trainable object."""

    import torch

    if type(row_index) is not int or row_index < 0:
        raise TeacherCohortAuditError("row index must be non-negative")
    oracle.validate(source)
    if tuple(source.shape) != tuple(target.shape):
        raise TeacherCohortAuditError("source and target geometry differs")
    gates = oracle.gate_probs.float()
    if not bool(((gates == 0.0) | (gates == 1.0)).all().item()):
        raise TeacherCohortAuditError("hardened oracle gates must be exactly one-hot")
    diagnostics = dict(oracle.diagnostics or {})
    required = (
        "preserve_fraction",
        "transport_fraction",
        "generate_fraction",
        "prebudget_generate_fraction",
        "postbudget_generate_fraction",
        "observed_max_prebudget_generate_fraction_per_phase",
        "observed_max_postbudget_generate_fraction_per_phase",
        "max_generate_fraction_per_phase",
        "budget_reject_fraction",
        "hard_executor_candidate_mse",
        "grid_sampler_integer_numeric_mse",
    )
    missing = [name for name in required if name not in diagnostics]
    if missing:
        raise TeacherCohortAuditError(
            f"hardened oracle diagnostics are incomplete: {missing}"
        )
    if float(diagnostics["max_generate_fraction_per_phase"]) != (
        HARD_ORACLE_GENERATE_BUDGET
    ):
        raise TeacherCohortAuditError("oracle is not using the hardened 0.12 budget")
    max_post = float(
        diagnostics["observed_max_postbudget_generate_fraction_per_phase"]
    )
    if max_post > HARD_ORACLE_GENERATE_BUDGET + 1.0e-6:
        raise TeacherCohortAuditError("oracle violated its hardened per-phase budget")
    proxy = spt.make_proxy_target(source, target, oracle)
    source_target = _mse(source, target)
    proxy_target = _mse(proxy, target)
    source_proxy = _mse(source, proxy)
    improvement = (source_target - proxy_target) / max(source_target, 1.0e-12)
    gate_fraction = {
        "preserve": float(gates[:, spt.GATE_PRESERVE].mean().item()),
        "transport": float(gates[:, spt.GATE_TRANSPORT].mean().item()),
        "generate": float(gates[:, spt.GATE_GENERATE].mean().item()),
    }
    for name, value in gate_fraction.items():
        if abs(value - float(diagnostics[f"{name}_fraction"])) > 1.0e-6:
            raise TeacherCohortAuditError(
                f"oracle {name} diagnostic differs from its hard gates"
            )
    numeric = (
        source_target,
        proxy_target,
        source_proxy,
        improvement,
        *gate_fraction.values(),
        *(float(diagnostics[name]) for name in required),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise TeacherCohortAuditError(f"row {row_index} has a non-finite metric")
    report: dict[str, Any] = {
        "row_index": row_index,
        "iid": _iid(row),
        "identity_sha256": legacy.dataset_identity(row, row_index),
        "prebudget_generate_fraction": float(
            diagnostics["prebudget_generate_fraction"]
        ),
        "observed_max_prebudget_generate_fraction_per_phase": float(
            diagnostics["observed_max_prebudget_generate_fraction_per_phase"]
        ),
        "postbudget_gate_fraction": gate_fraction,
        "observed_max_postbudget_generate_fraction_per_phase": max_post,
        "proxy_relative_improvement_over_copy": float(improvement),
        "mse": {
            "source_to_target": source_target,
            "proxy_to_target": proxy_target,
            "source_to_proxy": source_proxy,
        },
        "oracle_diagnostics": diagnostics,
    }
    report["selection"] = selection_decision(report, thresholds)
    return report


def _selection_aggregate(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise TeacherCohortAuditError("cannot aggregate an empty scan")
    criterion_names = tuple(reports[0]["selection"]["criteria"])
    return {
        "scanned_count": len(reports),
        "selected_count": sum(bool(report["selection"]["selected"]) for report in reports),
        "criterion_pass_count": {
            name: sum(
                bool(report["selection"]["criteria"][name]) for report in reports
            )
            for name in criterion_names
        },
        "criterion_reject_count": {
            name: sum(
                not bool(report["selection"]["criteria"][name])
                for report in reports
            )
            for name in criterion_names
        },
        "mean_postbudget_gate_fraction": {
            name: sum(
                float(report["postbudget_gate_fraction"][name]) for report in reports
            )
            / len(reports)
            for name in ("preserve", "transport", "generate")
        },
    }


def build_selected_membership(
    *,
    reports: Sequence[Mapping[str, Any]],
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    scan: ScanWindow,
    thresholds: SelectorThresholds,
    minimum_selected: int,
) -> dict[str, Any]:
    selected = [report for report in reports if report["selection"]["selected"]]
    selected.sort(key=lambda report: int(report["row_index"]))
    members = [
        {
            "ordinal": ordinal,
            "row_index": int(report["row_index"]),
            "iid": str(report["iid"]),
            "identity_sha256": str(report["identity_sha256"]),
        }
        for ordinal, report in enumerate(selected)
    ]
    value: dict[str, Any] = {
        "schema_version": MEMBERSHIP_SCHEMA,
        "selection_algorithm": SELECTION_ALGORITHM,
        "dataset": {
            "signature": dataset.signature,
            "summary_sha256": dataset_summary["sha256"],
            "summary_digest": dataset_summary["summary_digest"],
            "index_sha256": dataset_summary["index_sha256"],
            "full_dataset_rows": len(dataset),
        },
        "scan": scan.receipt(),
        "selector_thresholds": thresholds.receipt(),
        "minimum_selected": int(minimum_selected),
        "selected_count": len(members),
        "sufficient": len(members) >= int(minimum_selected),
        "ordered_selected_row_indices": [member["row_index"] for member in members],
        "members": members,
        "trainer_load_contract": {
            "verify_membership_digest": True,
            "verify_exact_dataset_signature": True,
            "verify_exact_dataset_summary_sha256": True,
            "recompute_each_row_identity_sha256": True,
            "iteration_order": "members_ascending_row_index",
            "implicit_dataset_fallback_forbidden": True,
        },
    }
    value["membership_digest"] = legacy.object_sha256(value)
    return value


def validate_selected_membership(
    value: Mapping[str, Any],
    *,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    require_sufficient: bool = True,
) -> tuple[int, ...]:
    """Strict-load entry point intended for later planner/joint integration."""

    candidate = dict(value)
    declared = candidate.pop("membership_digest", None)
    if (
        value.get("schema_version") != MEMBERSHIP_SCHEMA
        or value.get("selection_algorithm") != SELECTION_ALGORITHM
        or not isinstance(declared, str)
        or legacy.object_sha256(candidate) != declared
    ):
        raise TeacherCohortAuditError("membership schema or digest differs")
    identity = value.get("dataset")
    if not isinstance(identity, Mapping) or identity != {
        "signature": dataset.signature,
        "summary_sha256": dataset_summary["sha256"],
        "summary_digest": dataset_summary["summary_digest"],
        "index_sha256": dataset_summary["index_sha256"],
        "full_dataset_rows": len(dataset),
    }:
        raise TeacherCohortAuditError("membership dataset identity differs")
    members = value.get("members")
    ordered = value.get("ordered_selected_row_indices")
    if not isinstance(members, list) or not isinstance(ordered, list):
        raise TeacherCohortAuditError("membership rows are unavailable")
    if any(type(row_index) is not int for row_index in ordered):
        raise TeacherCohortAuditError("membership row indices must be integers")
    if (
        value.get("selected_count") != len(members)
        or ordered != [member.get("row_index") for member in members]
        or ordered != sorted(set(ordered))
    ):
        raise TeacherCohortAuditError("membership order/count differs")
    raw_thresholds = value.get("selector_thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise TeacherCohortAuditError("membership selector thresholds are unavailable")
    try:
        recorded_thresholds = SelectorThresholds(**dict(raw_thresholds))
    except TypeError as error:
        raise TeacherCohortAuditError(
            f"membership selector threshold fields differ: {error}"
        ) from error
    recorded_thresholds.validate()
    raw_scan = value.get("scan")
    if not isinstance(raw_scan, Mapping):
        raise TeacherCohortAuditError("membership scan window is unavailable")
    start = raw_scan.get("start")
    stop = raw_scan.get("stop_exclusive")
    count = raw_scan.get("row_count")
    if (
        raw_scan.get("mode") not in ("ordered_prefix", "explicit_half_open_range")
        or type(start) is not int
        or type(stop) is not int
        or type(count) is not int
        or start < 0
        or stop <= start
        or stop > len(dataset)
        or count != stop - start
        or any(not start <= row_index < stop for row_index in ordered)
    ):
        raise TeacherCohortAuditError("membership scan window differs")
    expected_load_contract = {
        "verify_membership_digest": True,
        "verify_exact_dataset_signature": True,
        "verify_exact_dataset_summary_sha256": True,
        "recompute_each_row_identity_sha256": True,
        "iteration_order": "members_ascending_row_index",
        "implicit_dataset_fallback_forbidden": True,
    }
    if value.get("trainer_load_contract") != expected_load_contract:
        raise TeacherCohortAuditError("membership trainer load contract differs")
    minimum = value.get("minimum_selected")
    sufficient = value.get("sufficient")
    if (
        type(minimum) is not int
        or minimum <= 0
        or type(sufficient) is not bool
        or sufficient != (len(members) >= minimum)
        or (require_sufficient and not sufficient)
    ):
        raise TeacherCohortAuditError("membership is below its trusted minimum")
    for ordinal, member in enumerate(members):
        if not isinstance(member, Mapping) or member.get("ordinal") != ordinal:
            raise TeacherCohortAuditError("membership ordinal differs")
        row_index = member.get("row_index")
        if type(row_index) is not int or not 0 <= row_index < len(dataset):
            raise TeacherCohortAuditError("membership row index is invalid")
        row = dataset[row_index]
        if (
            member.get("iid") != _iid(row)
            or member.get("identity_sha256")
            != legacy.dataset_identity(row, row_index)
        ):
            raise TeacherCohortAuditError(
                f"membership row identity differs at index {row_index}"
            )
    return tuple(int(row_index) for row_index in ordered)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TeacherCohortAuditError(f"cannot read membership {path}: {error}") from error
    if not isinstance(value, dict):
        raise TeacherCohortAuditError("membership file must contain one JSON object")
    return value


def load_selected_membership(
    path: str | Path,
    *,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    require_sufficient: bool = True,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise TeacherCohortAuditError("membership file cannot be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise TeacherCohortAuditError(f"membership is unavailable: {error}") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise TeacherCohortAuditError("membership must be a plain file")
    value = _read_json(resolved)
    rows = validate_selected_membership(
        value,
        dataset=dataset,
        dataset_summary=dataset_summary,
        require_sufficient=require_sufficient,
    )
    return value, rows


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = legacy.canonical_json_bytes(value) + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_output_directory(
    output: Path,
    *,
    audit: Mapping[str, Any],
    membership: Mapping[str, Any],
) -> None:
    if output.exists() or output.is_symlink():
        raise TeacherCohortAuditError(f"refusing to overwrite output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.tmp-")
    )
    _atomic_json(temporary / "selected_membership.json", membership)
    _atomic_json(temporary / "audit.json", audit)
    os.replace(temporary, output)


def _assert_equal_across_ranks(value: Mapping[str, Any], *, label: str) -> None:
    import torch.distributed as dist

    gathered: list[Optional[dict[str, Any]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(value))
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise TeacherCohortAuditError(f"{label} differs across data-parallel ranks")


def _method_hashes() -> dict[str, str]:
    paths = (HERE / "audit_teacher_cohort.py", HERE / "phase_transport.py")
    return {
        str(path.relative_to(METHOD_ROOT)): legacy.file_sha256(path) for path in paths
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = validate_cli(args)
    try:
        checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
        distributed = legacy.distributed_contract()
    except legacy.TrainingContractError as error:
        raise TeacherCohortAuditError(str(error)) from error
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise TeacherCohortAuditError("teacher trust cohort audit requires exactly 4 ranks")
    try:
        device, backend = legacy.initialise_distributed(distributed)
    except legacy.TrainingContractError as error:
        raise TeacherCohortAuditError(str(error)) from error

    import torch
    import torch.distributed as dist

    output = Path(args.output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise TeacherCohortAuditError(f"refusing to overwrite output {output}")
    dataset = legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    dataset_summary = legacy.validate_preprocessed_dataset_summary(
        args.dataset_summary,
        dataset,
        allow_incomplete=args.allow_incomplete_dataset,
    )
    scan = resolve_scan_window(args, len(dataset))
    _assert_equal_across_ranks(
        {
            "dataset_signature": dataset.signature,
            "dataset_summary_sha256": dataset_summary["sha256"],
            "dataset_index_sha256": dataset_summary["index_sha256"],
            "scan": scan.receipt(),
            "thresholds": thresholds.receipt(),
        },
        label="audit input contract",
    )
    vae_mean, vae_std, z_dim = legacy._vae_statistics(checkpoint)
    oracle_config = spt.PhaseTransportConfig(
        latent_channels=TEACHER_FEATURE_CHANNELS,
        teacher_temperature=args.teacher_temperature,
        teacher_generate_threshold=args.teacher_generate_threshold,
        teacher_transport_margin=args.teacher_transport_margin,
        teacher_require_cycle=True,
        teacher_allow_lossy_projection=False,
        max_generate_fraction_per_phase=HARD_ORACLE_GENERATE_BUDGET,
        teacher_allow_unbounded_generate_ablation=False,
    )
    oracle_config.validate()
    assigned = scan.row_indices[distributed.rank :: distributed.world_size]
    local_reports: list[dict[str, Any]] = []
    with torch.inference_mode():
        for row_index in assigned:
            row = dataset[row_index]
            source, target = student._clean_pair(
                row, vae_mean, vae_std, z_dim, device
            )
            oracle = spt.build_oracle_plan(
                source,
                target,
                oracle_config,
                feature_channels=TEACHER_FEATURE_CHANNELS,
            )
            local_reports.append(
                row_report(
                    row_index=row_index,
                    row=row,
                    source=source,
                    target=target,
                    oracle=oracle,
                    thresholds=thresholds,
                )
            )
            del source, target, oracle

    gathered: list[Optional[list[dict[str, Any]]]] = [None] * distributed.world_size
    dist.all_gather_object(gathered, local_reports)
    reports = [
        report
        for rank_reports in gathered
        if rank_reports is not None
        for report in rank_reports
    ]
    reports.sort(key=lambda report: int(report["row_index"]))
    if [report["row_index"] for report in reports] != list(scan.row_indices):
        raise TeacherCohortAuditError(
            "four-rank scan did not cover each requested row exactly once"
        )
    membership = build_selected_membership(
        reports=reports,
        dataset=dataset,
        dataset_summary=dataset_summary,
        scan=scan,
        thresholds=thresholds,
        minimum_selected=args.minimum_selected,
    )
    validate_selected_membership(
        membership,
        dataset=dataset,
        dataset_summary=dataset_summary,
        require_sufficient=False,
    )
    aggregate = _selection_aggregate(reports)
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "method_files_sha256": _method_hashes(),
        "read_only_dataset_scan": True,
        "optimizer_exists": False,
        "optimizer_steps": 0,
        "model_or_planner_constructed": False,
        "paired_target_use": "hardened_oracle_and_proxy_metrics_only",
        "external_mask_track_pose_flow": False,
        "checkpoint": {
            "path": str(checkpoint),
            "tree_sha256": legacy.CHECKPOINT_TREE_SHA256,
        },
        "dataset": {
            "path": str(dataset.root),
            "signature": dataset.signature,
            "summary": dict(dataset_summary),
            "full_dataset_rows": len(dataset),
        },
        "scan": scan.receipt(),
        "rank_assignments": {
            str(rank): list(scan.row_indices[rank :: distributed.world_size])
            for rank in range(distributed.world_size)
        },
        "distributed": {
            "world_size": distributed.world_size,
            "mode": "four_rank_disjoint_data_parallel_read_only",
            "backend": backend,
        },
        "oracle": {
            "config": asdict(oracle_config),
            "feature_channels": TEACHER_FEATURE_CHANNELS,
            "all_packed_channels_used": True,
            "hard_per_phase_generate_budget": HARD_ORACLE_GENERATE_BUDGET,
        },
        "selector": {
            "algorithm": SELECTION_ALGORITHM,
            "thresholds": thresholds.receipt(),
            "minimum_selected": args.minimum_selected,
            "allow_insufficient_selection": bool(args.allow_insufficient_selection),
        },
        "aggregate": aggregate,
        "selection": {
            "selected_count": membership["selected_count"],
            "sufficient": membership["sufficient"],
            "ordered_selected_row_indices": membership[
                "ordered_selected_row_indices"
            ],
            "members": membership["members"],
            "membership_file": "selected_membership.json",
            "membership_schema": MEMBERSHIP_SCHEMA,
            "membership_digest": membership["membership_digest"],
        },
        "rows": reports,
    }
    audit["audit_digest"] = legacy.object_sha256(audit)
    if distributed.rank == 0:
        _write_output_directory(output, audit=audit, membership=membership)
        print(
            json.dumps(
                {
                    "output_dir": str(output),
                    "audit_digest": audit["audit_digest"],
                    "membership_digest": membership["membership_digest"],
                    "scanned": len(reports),
                    "selected": membership["selected_count"],
                    "minimum_selected": args.minimum_selected,
                    "sufficient": membership["sufficient"],
                    "ordered_selected_row_indices": membership[
                        "ordered_selected_row_indices"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    sufficient = bool(membership["sufficient"])
    dist.destroy_process_group()
    if not sufficient and not args.allow_insufficient_selection:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

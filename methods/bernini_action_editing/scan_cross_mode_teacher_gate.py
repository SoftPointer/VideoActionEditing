#!/usr/bin/env python3
"""Read-only, exhaustive frozen-teacher gate scan for Bernini CMMD v6.

The online v6 trainer advances its routed-row cursor until the frozen T2V
teacher passes a gate.  The primary purpose of this program is to reproduce
or falsify that raw-coordinate teacher assumption as a negative-result
diagnostic.  It also prevents a failure from being hidden by making the pair
distribution conditional on the sigma at which rejection took place:

* it freezes the exact strict-359 x 40-step UniPC pair/sigma grid up front;
* it derives one explicit seed for every grid cell;
* it loads the same pinned Bernini renderer and constructs candidates through
  ``train_cross_mode_cmsg_auh._prepare_candidate_cpu`` and
  ``_move_candidate_to_device``;
* it executes all five frozen branches from the six-forward training cell
  (the adapted editor action branch is intentionally absent);
* it calls the exact v6 ``compute_frozen_prior_gate`` implementation; and
* it emits hash-bound records, distributions, and a deterministic 40-entry
  selection table before any optimizer is allowed to exist.

The scan never imports PEFT, creates an optimizer, calls backward, or mutates
model/data inputs.  Paired targets remain offline gate labels.  A partial scan
is deliberately not a training authorization: only a complete, validated
grid receives a summary and selection-table digest.  ``--resume`` accepts only
an exact, valid JSONL prefix of the same immutable scan contract.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_cross_mode_cmsg_auh as trainer  # noqa: E402


SCAN_SCHEMA = "bernini-cross-mode-teacher-gate-scan-v1"
GRID_SCHEMA = "bernini-cross-mode-teacher-gate-grid-v1"
RECORD_SCHEMA = "bernini-cross-mode-teacher-gate-record-v1"
SELECTION_SCHEMA = "bernini-cross-mode-teacher-gate-selection-v1"
SUMMARY_SCHEMA = "bernini-cross-mode-teacher-gate-summary-v1"
STRICT_PAIR_COUNT = 359
SIGMA_COUNT = trainer.sigma_strata.NUM_INFERENCE_STEPS
GRID_SIZE = STRICT_PAIR_COUNT * SIGMA_COUNT
SELECTION_RULE = "minimum_sha256_per_sigma_over_gate_eligible_grid_cells-v1"

# This is the literal frozen subset of trainer.FORWARD_CELL_ORDER.  It is kept
# in training-cell order, with only the adapted (graph-bearing) action branch
# removed.
FROZEN_FORWARD_ORDER = (
    "frozen_editor_negative_full_source",
    "frozen_editor_noop_full_source",
    "frozen_editor_action_full_source",
    "frozen_generator_negative_target_only",
    "frozen_generator_action_target_only",
)

GATE_METRIC_NAMES = (
    "active_phase_count",
    "mean_direction_cosine",
    "log_amplitude_mae",
    "covered_phase_fraction",
    "normalized_rmse",
    "frozen_prior_rms",
    "target_motion_rms",
)


class OfflineGateScanError(RuntimeError):
    """Raised before an incomplete or mismatched scan can look authoritative."""


@dataclass(frozen=True)
class GateGridCandidate:
    candidate_ordinal: int
    pair_ordinal: int
    row_index: int
    iid: str
    sigma_schedule_index: int
    sigma_timestep: int
    sigma_float32_be_hex: str
    seed: int
    teacher_active: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenScanCellResult:
    gate: Any
    frozen_velocity_rms: Mapping[str, float]


def _translate(error: Exception) -> OfflineGateScanError:
    return OfflineGateScanError(str(error))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustively scan the fixed strict359 x 40 CMMD v6 frozen-T2V "
            "teacher gate before training"
        )
    )
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preprocessed-parquet-dir", required=True)
    parser.add_argument("--dataset-summary", required=True)
    parser.add_argument("--routing-jsonl", required=True)
    parser.add_argument(
        "--expected-routing-jsonl-sha256",
        default=trainer.v5.STRICT_ROUTING_SHA256,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--noop-instruction", default=trainer.motion.DEFAULT_NOOP_INSTRUCTION
    )
    parser.add_argument(
        "--negative-prompt", default=trainer.v5.DEFAULT_NEGATIVE_PROMPT
    )
    parser.add_argument(
        "--expected-bernini-commit", default=trainer.legacy.BERNINI_OFFICIAL_COMMIT
    )
    parser.add_argument(
        "--expected-veomni-commit", default=trainer.legacy.VEOMNI_TESTED_COMMIT
    )
    parser.add_argument(
        "--expected-checkpoint-tree-sha256",
        default=trainer.legacy.CHECKPOINT_TREE_SHA256,
    )
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    return parser


def validate_cli(args: argparse.Namespace) -> None:
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise OfflineGateScanError("seed must be an integer in [0,2^63)")
    if type(args.resume) is not bool:
        raise OfflineGateScanError("resume must be boolean")
    if args.noop_instruction != trainer.motion.DEFAULT_NOOP_INSTRUCTION:
        raise OfflineGateScanError("the scan pins the v6 semantic no-op instruction")
    if args.negative_prompt != trainer.v5.DEFAULT_NEGATIVE_PROMPT:
        raise OfflineGateScanError("the scan pins Bernini's verbatim negative prompt")
    if args.expected_routing_jsonl_sha256 != trainer.v5.STRICT_ROUTING_SHA256:
        raise OfflineGateScanError("the scan requires the hash-bound strict359 route")
    for name in (
        "expected_bernini_commit",
        "expected_veomni_commit",
        "method_source_revision",
    ):
        if re.fullmatch(r"[0-9a-fA-F]{40}", str(getattr(args, name))) is None:
            raise OfflineGateScanError(f"{name} must be a full SHA-1")
    for name in (
        "expected_checkpoint_tree_sha256",
        "method_source_archive_sha256",
        "expected_routing_jsonl_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))) is None:
            raise OfflineGateScanError(f"{name} must be a lowercase SHA-256")
    if args.expected_bernini_commit.lower() != trainer.legacy.BERNINI_OFFICIAL_COMMIT:
        raise OfflineGateScanError("Bernini revision differs from the pinned release")
    if args.expected_veomni_commit.lower() != trainer.legacy.VEOMNI_TESTED_COMMIT:
        raise OfflineGateScanError("VeOmni revision differs from the pinned release")
    if args.expected_checkpoint_tree_sha256 != trainer.legacy.CHECKPOINT_TREE_SHA256:
        raise OfflineGateScanError("Bernini 1.3B checkpoint tree differs")
    if SIGMA_COUNT != 40 or trainer.NUM_FRAMES != 81 or trainer.LATENT_PHASES != 21:
        raise OfflineGateScanError("the scanner requires exact 81-frame/21-phase/40-sigma v6")
    config = trainer.core.CMSGTrainingLossConfig(enforce_frozen_prior_gate=True)
    config.validate()


def _candidate_digest(candidate: GateGridCandidate) -> str:
    return trainer.legacy.object_sha256(candidate.as_dict())


def build_fixed_grid(
    eligible_routes: Sequence[tuple[int, Any]], *, base_seed: int
) -> tuple[GateGridCandidate, ...]:
    """Freeze the pair-major strict359 x official-40 candidate grid."""

    if type(base_seed) is not int or not 0 <= base_seed < 2**63:
        raise OfflineGateScanError("base_seed must be an integer in [0,2^63)")
    if len(eligible_routes) != STRICT_PAIR_COUNT:
        raise OfflineGateScanError("fixed scan grid requires exactly 359 routed pairs")
    seen_rows: set[int] = set()
    seen_iids: set[str] = set()
    grid: list[GateGridCandidate] = []
    for pair_ordinal, item in enumerate(eligible_routes):
        if not isinstance(item, tuple) or len(item) != 2:
            raise OfflineGateScanError("eligible route entries must be (row_index, route)")
        row_index, route = item
        iid = getattr(route, "iid", None)
        if (
            type(row_index) is not int
            or row_index < 0
            or row_index in seen_rows
            or not isinstance(iid, str)
            or not iid
            or "\x00" in iid
            or iid in seen_iids
            or getattr(route, "tier", None) != "motion_only"
            or float(getattr(route, "full_target_weight", -1.0)) != 0.0
        ):
            raise OfflineGateScanError("eligible routes violate strict359 identity")
        seen_rows.add(row_index)
        seen_iids.add(iid)
        for schedule_index in range(SIGMA_COUNT):
            selected = trainer.sigma_strata.select_sigma_stratum(schedule_index)
            candidate_ordinal = pair_ordinal * SIGMA_COUNT + schedule_index
            grid.append(
                GateGridCandidate(
                    candidate_ordinal=candidate_ordinal,
                    pair_ordinal=pair_ordinal,
                    row_index=row_index,
                    iid=iid,
                    sigma_schedule_index=schedule_index,
                    sigma_timestep=selected.timestep,
                    sigma_float32_be_hex=selected.sigma_float32_be_hex,
                    seed=trainer.legacy.step_seed(
                        base_seed, candidate_ordinal, row_index
                    ),
                    teacher_active=(
                        trainer.spectrum.release_rho(schedule_index) > 0.0
                    ),
                )
            )
    if len(grid) != GRID_SIZE or any(
        candidate.candidate_ordinal != ordinal
        for ordinal, candidate in enumerate(grid)
    ):
        raise OfflineGateScanError("fixed grid cardinality/order differs")
    return tuple(grid)


def fixed_grid_sha256(grid: Sequence[GateGridCandidate]) -> str:
    if len(grid) != GRID_SIZE:
        raise OfflineGateScanError("cannot hash an incomplete fixed grid")
    return trainer.legacy.object_sha256(
        {
            "schema_version": GRID_SCHEMA,
            "order": "pair_major_then_official_sigma_index",
            "pair_count": STRICT_PAIR_COUNT,
            "sigma_count": SIGMA_COUNT,
            "candidates": [candidate.as_dict() for candidate in grid],
        }
    )


def _run_five_frozen_forward_cell(
    *, renderer: Any, candidate: trainer.MovedCandidate
) -> FrozenScanCellResult:
    """Run the five graph-free branches that surround v6's adapted branch."""

    import torch

    batches = (
        candidate.editor_negative,
        candidate.editor_noop,
        candidate.editor_action,
        candidate.generator_negative,
        candidate.generator_action,
    )
    with torch.no_grad():
        velocities = tuple(
            trainer.motion.renderer_velocity_prediction(renderer, batch)
            for batch in batches
        )
    shared_noisy = candidate.auxiliary["shared_noisy"]
    if any(
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bfloat16
        or tuple(value.shape) != tuple(shared_noisy.shape)
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        for value in velocities
    ):
        raise OfflineGateScanError(
            "all five scan forwards must be finite, frozen native-BF16 fields"
        )
    (
        _editor_negative_v,
        _editor_noop_v,
        _editor_action_v,
        generator_negative_v,
        generator_action_v,
    ) = velocities
    try:
        generator_uncond, frozen_generator_action = (
            trainer._generator_plain_cfg_clean(
                shared_noisy=shared_noisy,
                sigma=candidate.auxiliary["sigma"],
                negative_velocity=generator_negative_v,
                action_velocity=generator_action_v,
            )
        )
        source = trainer.v5._as_phase_grid(
            candidate.auxiliary["source_clean"].float()
        )
        target = trainer.v5._as_phase_grid(
            candidate.auxiliary["target_clean"].float()
        )
        target_motion = trainer.spectrum.q0(target - source).detach()
        generator_teacher = trainer.spectrum.q0(
            frozen_generator_action - generator_uncond
        ).detach()
        gate = trainer.core.compute_frozen_prior_gate(
            generator_teacher,
            target_motion,
            config=trainer.core.CMSGTrainingLossConfig(
                enforce_frozen_prior_gate=True
            ),
        )
    except (
        trainer.CMSGauhTrainingError,
        trainer.core.CrossModeCMSGTrainingError,
        trainer.spectrum.CrossModeMotionSpectrumError,
        trainer.v5.PriorTangentTrainingError,
    ) as error:
        raise _translate(error) from error
    velocity_rms = {
        name: float(value.float().square().mean().sqrt().detach().cpu().item())
        for name, value in zip(FROZEN_FORWARD_ORDER, velocities)
    }
    if any(not math.isfinite(value) for value in velocity_rms.values()):
        raise OfflineGateScanError("a frozen branch RMS is non-finite")
    return FrozenScanCellResult(gate=gate, frozen_velocity_rms=velocity_rms)


def _make_record(
    *,
    candidate: GateGridCandidate,
    cell: FrozenScanCellResult,
    prepared: trainer.PreparedCandidate,
    scan_contract_sha256: str,
) -> dict[str, Any]:
    audit = trainer._gate_audit_record(
        cell.gate,
        global_step=candidate.sigma_schedule_index,
        attempt_ordinal=candidate.candidate_ordinal,
        attempt_in_step=candidate.pair_ordinal,
        row_index=candidate.row_index,
        iid=candidate.iid,
        schedule_index=candidate.sigma_schedule_index,
        timestep=candidate.sigma_timestep,
        accepted=False,
        teacher_active=candidate.teacher_active,
    )
    gate = {
        "passed": bool(audit["gate_passed"]),
        **{name: audit[name] for name in GATE_METRIC_NAMES},
    }
    for name in GATE_METRIC_NAMES:
        value = gate[name]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise OfflineGateScanError(f"gate metric {name} is non-finite")
    record: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA,
        "scan_contract_sha256": scan_contract_sha256,
        "candidate": candidate.as_dict(),
        "candidate_sha256": _candidate_digest(candidate),
        "instruction_sha256": prepared.instruction_sha256,
        "t2v_rope_parity": dict(prepared.t2v_rope_parity),
        "frozen_forward_order": list(FROZEN_FORWARD_ORDER),
        "frozen_velocity_rms": dict(cell.frozen_velocity_rms),
        "gate": gate,
    }
    record["record_sha256"] = trainer.legacy.object_sha256(record)
    return record


def _validate_record(
    record: Mapping[str, Any],
    *,
    expected: GateGridCandidate,
    scan_contract_sha256: str,
) -> None:
    if not isinstance(record, Mapping):
        raise OfflineGateScanError("gate record is not an object")
    candidate = record.get("candidate")
    if (
        record.get("schema_version") != RECORD_SCHEMA
        or record.get("scan_contract_sha256") != scan_contract_sha256
        or candidate != expected.as_dict()
        or record.get("candidate_sha256") != _candidate_digest(expected)
        or record.get("frozen_forward_order") != list(FROZEN_FORWARD_ORDER)
    ):
        raise OfflineGateScanError(
            f"gate record identity differs at candidate {expected.candidate_ordinal}"
        )
    digest_candidate = dict(record)
    digest = digest_candidate.pop("record_sha256", None)
    if digest != trainer.legacy.object_sha256(digest_candidate):
        raise OfflineGateScanError(
            f"gate record hash differs at candidate {expected.candidate_ordinal}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", str(record.get("instruction_sha256"))) is None:
        raise OfflineGateScanError("gate record instruction hash is invalid")
    parity = record.get("t2v_rope_parity")
    if not isinstance(parity, Mapping) or parity.get("verified") is not True:
        raise OfflineGateScanError("gate record lacks native T2V RoPE parity")
    velocity_rms = record.get("frozen_velocity_rms")
    if (
        not isinstance(velocity_rms, Mapping)
        or set(velocity_rms) != set(FROZEN_FORWARD_ORDER)
        or any(
            isinstance(value, bool) or not math.isfinite(float(value))
            for value in velocity_rms.values()
        )
    ):
        raise OfflineGateScanError("gate record frozen branch RMS differs")
    gate = record.get("gate")
    if (
        not isinstance(gate, Mapping)
        or type(gate.get("passed")) is not bool
        or set(gate) != {"passed", *GATE_METRIC_NAMES}
    ):
        raise OfflineGateScanError("gate record metrics schema differs")
    for name in GATE_METRIC_NAMES:
        value = gate[name]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise OfflineGateScanError(f"gate record metric {name} is invalid")
    active_count = gate["active_phase_count"]
    if type(active_count) is not int or not 0 <= active_count < trainer.LATENT_PHASES:
        raise OfflineGateScanError("gate active phase count is invalid")


def validate_complete_records(
    records: Sequence[Mapping[str, Any]],
    grid: Sequence[GateGridCandidate],
    *,
    scan_contract_sha256: str,
    allow_prefix: bool = False,
) -> None:
    if len(grid) != GRID_SIZE:
        raise OfflineGateScanError("record validation requires the complete fixed grid")
    if len(records) > len(grid) or (not allow_prefix and len(records) != len(grid)):
        raise OfflineGateScanError("gate record count differs from the fixed grid")
    for record, expected in zip(records, grid):
        _validate_record(
            record,
            expected=expected,
            scan_contract_sha256=scan_contract_sha256,
        )


def _selection_priority(
    *,
    candidate: GateGridCandidate,
    grid_sha256: str,
    scan_contract_sha256: str,
) -> str:
    return trainer.legacy.object_sha256(
        {
            "schema_version": SELECTION_SCHEMA,
            "selection_rule": SELECTION_RULE,
            "grid_sha256": grid_sha256,
            "scan_contract_sha256": scan_contract_sha256,
            "candidate_sha256": _candidate_digest(candidate),
        }
    )


def build_selection_table(
    records: Sequence[Mapping[str, Any]],
    grid: Sequence[GateGridCandidate],
    *,
    grid_sha256: str,
    scan_contract_sha256: str,
) -> dict[str, Any]:
    """Build an auditable selection or a hash-bound negative-result table.

    Missing gate support at an active sigma is evidence, not an I/O failure.
    Such a stratum receives an explicit null selection and makes
    ``training_authorized`` false.  This preserves the complete scan for
    diagnosis while still failing closed for any downstream trainer.
    """

    validate_complete_records(
        records, grid, scan_contract_sha256=scan_contract_sha256
    )
    entries: list[dict[str, Any]] = []
    missing_active_sigma_indices: list[int] = []
    for schedule_index in range(SIGMA_COUNT):
        candidates: list[tuple[str, GateGridCandidate, Mapping[str, Any]]] = []
        for candidate, record in zip(grid, records):
            if candidate.sigma_schedule_index != schedule_index:
                continue
            gate_required = candidate.teacher_active
            if gate_required and record["gate"]["passed"] is not True:
                continue
            priority = _selection_priority(
                candidate=candidate,
                grid_sha256=grid_sha256,
                scan_contract_sha256=scan_contract_sha256,
            )
            candidates.append((priority, candidate, record))
        if not candidates:
            # This can only happen for a rho-positive stratum: rho-zero strata
            # deliberately admit every record irrespective of the diagnostic
            # gate bit.  Keep the null in the signed table instead of hiding a
            # scientifically important negative result behind an exception.
            missing_active_sigma_indices.append(schedule_index)
            reference = grid[schedule_index]
            entries.append(
                {
                    "optimizer_step": schedule_index,
                    "sigma_schedule_index": schedule_index,
                    "sigma_timestep": reference.sigma_timestep,
                    "sigma_float32_be_hex": reference.sigma_float32_be_hex,
                    "teacher_active": True,
                    "gate_required": True,
                    "eligibility": "no_gate_passing_candidate",
                    "selection_priority_sha256": None,
                    "candidate": None,
                    "candidate_sha256": None,
                    "record_sha256": None,
                    "gate": None,
                }
            )
            continue
        priority, chosen, record = min(candidates, key=lambda item: item[0])
        entries.append(
            {
                "optimizer_step": schedule_index,
                "sigma_schedule_index": schedule_index,
                "sigma_timestep": chosen.sigma_timestep,
                "sigma_float32_be_hex": chosen.sigma_float32_be_hex,
                "teacher_active": chosen.teacher_active,
                "gate_required": chosen.teacher_active,
                "eligibility": (
                    "gate_passed" if chosen.teacher_active else "rho_zero_gate_inactive"
                ),
                "selection_priority_sha256": priority,
                "candidate": chosen.as_dict(),
                "candidate_sha256": record["candidate_sha256"],
                "record_sha256": record["record_sha256"],
                "gate": dict(record["gate"]),
            }
        )
    value: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "selection_rule": SELECTION_RULE,
        "scan_contract_sha256": scan_contract_sha256,
        "grid_sha256": grid_sha256,
        "entry_count": len(entries),
        "selected_count": sum(entry["candidate"] is not None for entry in entries),
        "missing_active_sigma_indices": missing_active_sigma_indices,
        "training_authorized": not missing_active_sigma_indices,
        "raw_coordinate_teacher_gate_supports_full_active_schedule": (
            not missing_active_sigma_indices
        ),
        "one_candidate_per_official_sigma": not missing_active_sigma_indices,
        "online_rejection_forbidden_when_consuming_this_table": True,
        "entries": entries,
    }
    value["selection_table_sha256"] = trainer.legacy.object_sha256(value)
    return value


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise OfflineGateScanError("cannot summarize an empty metric")
    if not 0.0 <= fraction <= 1.0:
        raise OfflineGateScanError("quantile fraction must lie in [0,1]")
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    )


def _distribution(values: Sequence[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if not numeric or any(not math.isfinite(value) for value in numeric):
        raise OfflineGateScanError("distribution values must be non-empty and finite")
    ordered = sorted(numeric)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _quantile(ordered, 0.10),
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / len(ordered),
    }


def build_summary(
    records: Sequence[Mapping[str, Any]],
    grid: Sequence[GateGridCandidate],
    *,
    grid_sha256: str,
    scan_contract_sha256: str,
    selection_table: Mapping[str, Any],
    records_file_sha256: str,
) -> dict[str, Any]:
    validate_complete_records(
        records, grid, scan_contract_sha256=scan_contract_sha256
    )
    passed = sum(record["gate"]["passed"] is True for record in records)
    per_sigma: list[dict[str, Any]] = []
    for schedule_index in range(SIGMA_COUNT):
        subset = [
            record
            for candidate, record in zip(grid, records)
            if candidate.sigma_schedule_index == schedule_index
        ]
        sigma_passed = sum(record["gate"]["passed"] is True for record in subset)
        selected = selection_table["entries"][schedule_index]
        per_sigma.append(
            {
                "sigma_schedule_index": schedule_index,
                "sigma_timestep": grid[schedule_index].sigma_timestep,
                "sigma_float32_be_hex": grid[schedule_index].sigma_float32_be_hex,
                "teacher_active": grid[schedule_index].teacher_active,
                "candidate_count": len(subset),
                "gate_pass_count": sigma_passed,
                "gate_pass_rate": sigma_passed / len(subset),
                "selected_candidate_sha256": selected["candidate_sha256"],
                "metrics": {
                    name: _distribution([record["gate"][name] for record in subset])
                    for name in GATE_METRIC_NAMES
                },
            }
        )
    per_pair = []
    for pair_ordinal in range(STRICT_PAIR_COUNT):
        subset = [
            record
            for candidate, record in zip(grid, records)
            if candidate.pair_ordinal == pair_ordinal
        ]
        pair_passed = sum(record["gate"]["passed"] is True for record in subset)
        first = grid[pair_ordinal * SIGMA_COUNT]
        per_pair.append(
            {
                "pair_ordinal": pair_ordinal,
                "row_index": first.row_index,
                "iid": first.iid,
                "candidate_count": len(subset),
                "gate_pass_count": pair_passed,
                "gate_pass_rate": pair_passed / len(subset),
            }
        )
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "complete": True,
        "read_only_frozen_scan": True,
        "model_parameter_updates": 0,
        "optimizer_created": False,
        "backward_calls": 0,
        "paired_target_offline_only": True,
        "inference_contract_changed": False,
        "production_claim_forbidden": True,
        "scan_contract_sha256": scan_contract_sha256,
        "grid_sha256": grid_sha256,
        "record_count": len(records),
        "records_semantic_sha256": trainer.legacy.object_sha256(list(records)),
        "records_file_sha256": records_file_sha256,
        "gate_pass_count": passed,
        "gate_pass_rate": passed / len(records),
        "metric_distributions": {
            name: _distribution([record["gate"][name] for record in records])
            for name in GATE_METRIC_NAMES
        },
        "per_sigma": per_sigma,
        "per_pair": per_pair,
        "selection_table_sha256": selection_table["selection_table_sha256"],
        "selection_entry_count": selection_table["entry_count"],
        "selection_selected_count": selection_table["selected_count"],
        "selection_missing_active_sigma_indices": list(
            selection_table["missing_active_sigma_indices"]
        ),
        "training_authorized": bool(selection_table["training_authorized"]),
        "raw_coordinate_teacher_gate_supports_full_active_schedule": bool(
            selection_table[
                "raw_coordinate_teacher_gate_supports_full_active_schedule"
            ]
        ),
        "selection_preserves_all_40_sigmas": bool(
            selection_table["one_candidate_per_official_sigma"]
        ),
        "selection_consumption_requirement": (
            "trainer may consume only when training_authorized=true, must bind "
            "selection_table_sha256, and must perform no online rejection"
        ),
    }
    summary["summary_sha256"] = trainer.legacy.object_sha256(summary)
    return summary


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="ascii") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OfflineGateScanError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise OfflineGateScanError(f"{path.name} is not a JSON object")
    return value


def _read_record_prefix(
    path: Path,
    grid: Sequence[GateGridCandidate],
    *,
    scan_contract_sha256: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise OfflineGateScanError("gate_records.jsonl must be a plain file")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="ascii") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith("\n") or not line.strip():
                    raise OfflineGateScanError(
                        f"gate record line {line_number} is blank or truncated"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OfflineGateScanError(
                        f"gate record line {line_number} is not an object"
                    )
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OfflineGateScanError(f"cannot read gate record prefix: {error}") from error
    validate_complete_records(
        records,
        grid,
        scan_contract_sha256=scan_contract_sha256,
        allow_prefix=True,
    )
    return records


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _json_bytes(record)
    with path.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_output_location(output: Path, protected: Sequence[Path]) -> None:
    for root in protected:
        if output == root or _is_within(output, root):
            raise OfflineGateScanError(
                f"output may not be inside read-only input tree: {root}"
            )
    if output.exists() and output.is_symlink():
        raise OfflineGateScanError("output directory may not be a symlink")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise OfflineGateScanError("output parent must be a plain existing directory")


def _scan_contract(
    *,
    args: argparse.Namespace,
    bernini_revision: str,
    veomni_revision: str,
    checkpoint: Path,
    dataset: Any,
    dataset_summary: Mapping[str, Any],
    router: Any,
    eligible_routes: Sequence[tuple[int, Any]],
    grid_sha256: str,
    backend: str,
    transformers_version: str,
) -> dict[str, Any]:
    config = trainer.core.CMSGTrainingLossConfig(enforce_frozen_prior_gate=True)
    value: dict[str, Any] = {
        "schema_version": SCAN_SCHEMA,
        "method": "bernini-cmmd-v6-offline-frozen-teacher-gate-scan",
        "diagnostic_role": (
            "reproduce_or_falsify_raw_coordinate_t2v_teacher_gate_support"
        ),
        "method_source_revision": args.method_source_revision.lower(),
        "method_source_archive_sha256": args.method_source_archive_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_path": str(checkpoint),
        "checkpoint_tree_sha256": args.expected_checkpoint_tree_sha256,
        "dataset_signature": dataset.signature,
        "dataset_summary_sha256": dataset_summary["sha256"],
        "dataset_index_sha256": dataset_summary["index_sha256"],
        "routing_digest": router.digest,
        "routing_file_sha256": router.file_sha256,
        "eligible_route_stream_sha256": trainer.legacy.object_sha256(
            [
                {
                    "row_index": row_index,
                    "iid": route.iid,
                    "tier": route.tier,
                    "full_target_weight": route.full_target_weight,
                }
                for row_index, route in eligible_routes
            ]
        ),
        "base_seed": int(args.seed),
        "candidate_seed_formula": "step_seed(base_seed,candidate_ordinal,row_index)",
        "grid_schema": GRID_SCHEMA,
        "grid_order": "pair_major_then_official_sigma_index",
        "pair_count": STRICT_PAIR_COUNT,
        "sigma_count": SIGMA_COUNT,
        "candidate_count": GRID_SIZE,
        "grid_sha256": grid_sha256,
        "sigma_schedule_sha256": trainer.sigma_strata.SCHEDULE_SHA256,
        "release_schedule_sha256": trainer.legacy.object_sha256(
            list(trainer.spectrum.release_rho_schedule())
        ),
        "gate_config": asdict(config),
        "gate_config_sha256": trainer.legacy.object_sha256(asdict(config)),
        "frozen_forward_order": list(FROZEN_FORWARD_ORDER),
        "forwards_per_candidate": len(FROZEN_FORWARD_ORDER),
        "adapted_editor_forward_present": False,
        "all_forwards_no_grad": True,
        "model_parameter_updates": 0,
        "optimizer_created": False,
        "backward_calls": 0,
        "target_used_as_model_condition": False,
        "paired_target_offline_gate_label_only": True,
        "training_bridge_endpoint": trainer.TRAINING_BRIDGE_ENDPOINT,
        "target_endpoint_teacher_leakage_forbidden": True,
        "generator_guidance": {
            "mode": "official_t2v_plain_cfg",
            "scale": trainer.T2V_GUIDANCE_SCALE,
            "native_bf16_combine_before_clean": True,
        },
        "selection_rule": SELECTION_RULE,
        "selection_requires_gate_when_rho_positive": True,
        "selection_does_not_require_gate_when_rho_zero": True,
        "online_rejection_forbidden_when_selection_is_consumed": True,
        "inference_conditions_unchanged": list(trainer.core.INFERENCE_CONDITIONS),
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": backend,
            "same_candidate_all_ranks": True,
            "exact_gate_record_all_ranks": True,
        },
        "transformers_version": transformers_version,
        "complete": False,
        "production_claim_forbidden": True,
    }
    return {
        "value": value,
        "scan_contract_sha256": trainer.legacy.object_sha256(value),
    }


def _prepare_output(
    output: Path,
    contract: Mapping[str, Any],
    *,
    resume: bool,
    rank: int,
) -> None:
    import torch.distributed as dist

    if rank == 0:
        contract_path = output / "scan_contract.json"
        if resume:
            if not output.is_dir() or output.is_symlink():
                raise OfflineGateScanError("resume output must be a plain directory")
            observed = _read_json(contract_path)
            if observed != dict(contract):
                raise OfflineGateScanError("resume scan contract differs")
        else:
            if output.exists():
                raise OfflineGateScanError(
                    "output already exists; use --resume only for an exact prefix"
                )
            output.mkdir(mode=0o700)
            _write_atomic(contract_path, contract)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _assert_equal_across_ranks(value: Any, *, label: str) -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise OfflineGateScanError(f"{label} differs across ranks")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_cli(args)
    try:
        bernini_root, veomni_root, bernini_revision, veomni_revision = (
            trainer.legacy.validate_source_trees(
                args.bernini_root,
                args.veomni_root,
                expected_bernini_commit=args.expected_bernini_commit,
                expected_veomni_commit=args.expected_veomni_commit,
            )
        )
        checkpoint, transformer_config = trainer.legacy.validate_checkpoint(
            args.checkpoint
        )
    except trainer.legacy.TrainingContractError as error:
        raise _translate(error) from error
    if transformer_config["num_attention_heads"] % 4:
        raise OfflineGateScanError("1.3B attention heads must divide Ulysses=4")
    trainer.legacy.activate_source_trees(bernini_root, veomni_root)

    import torch
    import torch.distributed as dist
    from diffusers import UniPCMultistepScheduler
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer, __version__ as transformers_version

    from bernini.cli import DEFAULT_NEG_PROMPT
    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from bernini.models.transformer_wan import WanRotaryPosEmbed
    from bernini.training.data import NoiseScheduler, SYSTEM_PROMPTS, process_renderer_sample

    if DEFAULT_NEG_PROMPT != trainer.v5.DEFAULT_NEGATIVE_PROMPT:
        raise OfflineGateScanError("runtime Bernini negative prompt differs")
    if SYSTEM_PROMPTS.get("t2v") != trainer.T2V_SYSTEM_PROMPT:
        raise OfflineGateScanError("runtime Bernini T2V system prompt differs")

    distributed = trainer.legacy.distributed_contract()
    if distributed.world_size != 4 or distributed.ulysses_size != 4:
        raise OfflineGateScanError("formal gate scan requires exactly four AUH ranks")
    try:
        device, backend = trainer.legacy.initialise_distributed(distributed)
    except trainer.legacy.TrainingContractError as error:
        raise _translate(error) from error
    from bernini.parallel import init_parallel_state

    init_parallel_state(ulysses_size=4)
    dataset = trainer.legacy.ParquetRowStore(args.preprocessed_parquet_dir)
    try:
        dataset_summary = trainer.legacy.validate_preprocessed_dataset_summary(
            args.dataset_summary, dataset, allow_incomplete=False
        )
        router = trainer.motion.ReviewRouter.load(
            args.routing_jsonl, default_tier="reject"
        )
        eligible_routes = trainer.v4._build_eligible_routes(dataset, router)
        trainer._strict_router(args, router, eligible_routes, dataset)
    except (
        trainer.legacy.TrainingContractError,
        trainer.motion.MotionContractError,
        trainer.v4.DeltaTrainingError,
        trainer.CMSGauhTrainingError,
    ) as error:
        raise _translate(error) from error
    grid = build_fixed_grid(eligible_routes, base_seed=args.seed)
    grid_sha256 = fixed_grid_sha256(grid)

    output = Path(args.output).expanduser().resolve()
    _validate_output_location(
        output,
        (
            bernini_root,
            veomni_root,
            checkpoint,
            dataset.root,
            Path(args.routing_jsonl).expanduser().resolve(strict=True).parent,
        ),
    )

    config = BerniniRendererConfig.from_pretrained(
        str(bernini_root / "configs/bernini_renderer_wan21_1p3b"),
        local_files_only=True,
        **trainer.legacy.renderer_config_overrides(checkpoint),
    )
    config.dtype = torch.bfloat16
    try:
        trainer.legacy.validate_renderer_config_mapping(config.to_dict(), checkpoint)
    except trainer.legacy.TrainingContractError as error:
        raise _translate(error) from error
    renderer = BerniniRendererModel(config)
    renderer.requires_grad_(False)
    renderer.to(device)
    renderer.eval()
    renderer.t5_text_encoder.eval()
    if any(parameter.requires_grad for parameter in renderer.parameters()):
        raise OfflineGateScanError("frozen scan renderer retained trainable parameters")

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint),
        subfolder="tokenizer",
        padding_side="right",
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=trainer.legacy.TOKENIZER_FIX_MISTRAL_REGEX,
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=True)
    vae_mean, vae_std, z_dim = trainer.legacy._vae_statistics(checkpoint)
    scheduler_kwargs = trainer.legacy.noise_scheduler_kwargs()
    scheduler_kwargs["noise_tmin"] = trainer.MINIMUM_TRAINING_SIGMA
    scheduler = NoiseScheduler(**scheduler_kwargs)
    inference_scheduler = UniPCMultistepScheduler.from_pretrained(
        str(checkpoint),
        subfolder="scheduler",
        local_files_only=True,
        flow_shift=trainer.sigma_strata.FLOW_SHIFT,
    )
    trainer.sigma_strata.audit_runtime_unipc_schedule(inference_scheduler)

    contract = _scan_contract(
        args=args,
        bernini_revision=bernini_revision,
        veomni_revision=veomni_revision,
        checkpoint=checkpoint,
        dataset=dataset,
        dataset_summary=dataset_summary,
        router=router,
        eligible_routes=eligible_routes,
        grid_sha256=grid_sha256,
        backend=backend,
        transformers_version=transformers_version,
    )
    contract_sha256 = contract["scan_contract_sha256"]
    _assert_equal_across_ranks(contract_sha256, label="scan contract")
    _prepare_output(
        output, contract, resume=args.resume, rank=distributed.rank
    )
    record_path = output / "gate_records.jsonl"
    records = _read_record_prefix(
        record_path, grid, scan_contract_sha256=contract_sha256
    )
    _assert_equal_across_ranks(len(records), label="resume prefix length")

    for candidate in grid[len(records) :]:
        row_index, raw_row, route = trainer.v4._next_routed_row(
            dataset, eligible_routes, ordinal=candidate.pair_ordinal
        )
        if row_index != candidate.row_index or route.iid != candidate.iid:
            raise OfflineGateScanError("fixed grid no longer matches routed dataset")
        identity = trainer.legacy.dataset_identity(raw_row, row_index)
        trainer.legacy.assert_identical_row(identity)
        trainer.legacy.seed_same_sample(candidate.seed)
        selected_stratum = trainer.sigma_strata.select_sigma_stratum(
            candidate.sigma_schedule_index
        )
        prepared = trainer._prepare_candidate_cpu(
            raw_row=raw_row,
            tokenizer=tokenizer,
            prompt_cleaner=prompt_clean,
            system_prompts=SYSTEM_PROMPTS,
            rope=rope,
            vae_mean=vae_mean,
            vae_std=vae_std,
            z_dim=z_dim,
            scheduler=scheduler,
            noop_instruction=args.noop_instruction,
            negative_prompt=args.negative_prompt,
            process_renderer_sample=process_renderer_sample,
            selected_stratum=selected_stratum,
        )
        moved = trainer._move_candidate_to_device(prepared, device=device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            cell = _run_five_frozen_forward_cell(
                renderer=renderer, candidate=moved
            )
        record = _make_record(
            candidate=candidate,
            cell=cell,
            prepared=prepared,
            scan_contract_sha256=contract_sha256,
        )
        _validate_record(
            record,
            expected=candidate,
            scan_contract_sha256=contract_sha256,
        )
        trainer._assert_gate_record_equal_across_ranks(record)
        records.append(record)
        if distributed.rank == 0:
            _append_record(record_path, record)
            if (candidate.candidate_ordinal + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "event": "offline_gate_scan_progress",
                            "completed": candidate.candidate_ordinal + 1,
                            "total": GRID_SIZE,
                            "last_gate_passed": record["gate"]["passed"],
                            "last_sigma_schedule_index": (
                                candidate.sigma_schedule_index
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    validate_complete_records(
        records, grid, scan_contract_sha256=contract_sha256
    )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if distributed.rank == 0:
        selection = build_selection_table(
            records,
            grid,
            grid_sha256=grid_sha256,
            scan_contract_sha256=contract_sha256,
        )
        _write_atomic(output / "selection_table.json", selection)
        records_file_sha256 = trainer.legacy.file_sha256(record_path)
        summary = build_summary(
            records,
            grid,
            grid_sha256=grid_sha256,
            scan_contract_sha256=contract_sha256,
            selection_table=selection,
            records_file_sha256=records_file_sha256,
        )
        _write_atomic(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    "event": "offline_gate_scan_complete",
                    "record_count": len(records),
                    "gate_pass_rate": summary["gate_pass_rate"],
                    "grid_sha256": grid_sha256,
                    "selection_table_sha256": selection[
                        "selection_table_sha256"
                    ],
                    "training_authorized": selection["training_authorized"],
                    "missing_active_sigma_indices": selection[
                        "missing_active_sigma_indices"
                    ],
                    "summary_sha256": summary["summary_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


__all__ = [
    "FROZEN_FORWARD_ORDER",
    "GRID_SIZE",
    "GateGridCandidate",
    "OfflineGateScanError",
    "STRICT_PAIR_COUNT",
    "build_fixed_grid",
    "build_parser",
    "build_selection_table",
    "build_summary",
    "fixed_grid_sha256",
    "main",
    "validate_cli",
    "validate_complete_records",
]


if __name__ == "__main__":
    raise SystemExit(main())

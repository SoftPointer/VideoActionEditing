#!/usr/bin/env python3
"""Independent receipt-only postflight for PAIR-v7 Phase-A2.

A scientific NO-GO is a valid completed measurement, not a launcher failure.
Neither GO nor NO-GO authorizes an update.  Because Phase-A2 persists no raw
gradient or safe direction, a GO may only hand off to a separately authorized
Phase-B job that remeasures the entire sealed bank and applies the newly solved
direction in memory; the receipt cannot reconstruct it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import audit_pair_v7_phase_a2_multicondition_geometry as runtime


POSTFLIGHT_SCHEMA = "bernini-pair-v7-phase-a2-postflight-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PhaseA2PostflightError(RuntimeError):
    """The published receipt does not close the read-only Phase-A2 audit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseA2PostflightError(message)


def _seal(receipt: Mapping[str, Any], *, label: str) -> str:
    try:
        return runtime._check_seal(receipt, label=label)
    except Exception as error:
        raise PhaseA2PostflightError(f"{label} seal differs") from error


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PhaseA2PostflightError(f"{label} must be lowercase SHA-256")
    return value


def _rehash_bound_file(path_value: Any, digest_value: Any, *, label: str) -> None:
    digest = _sha(digest_value, label=f"{label} file")
    raw = Path(str(path_value))
    _require(raw.is_absolute(), f"{label} path must be absolute")
    try:
        path = raw.resolve(strict=True)
    except OSError as error:
        raise PhaseA2PostflightError(f"{label} is absent") from error
    _require(
        path == raw and path.is_file() and not path.is_symlink(),
        f"{label} must be a canonical plain file",
    )
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(observed == digest, f"{label} bytes changed")


def _read_receipt(path_value: str | Path) -> tuple[Path, Mapping[str, Any], str]:
    raw_path = Path(path_value)
    _require(raw_path.is_absolute(), "receipt path must be absolute")
    try:
        path = raw_path.resolve(strict=True)
    except OSError as error:
        raise PhaseA2PostflightError("receipt is absent") from error
    _require(path == raw_path and path.is_file() and not path.is_symlink(), "receipt must be a canonical plain file")
    payload = path.read_bytes()
    try:
        receipt = json.loads(payload.decode("ascii", errors="strict"))
    except Exception as error:
        raise PhaseA2PostflightError("receipt must be strict ASCII JSON") from error
    _require(isinstance(receipt, Mapping), "receipt must be an object")
    return path, receipt, hashlib.sha256(payload).hexdigest()


def validate_phase_a2_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt_digest = _seal(receipt, label="Phase-A2 final receipt")
    go = receipt.get("primary_replication_go")
    _require(type(go) is bool, "primary_replication_go must be a JSON bool")
    _require(
        receipt.get("schema_version") == runtime.RUN_RECEIPT_SCHEMA
        and receipt.get("method_name") == runtime.METHOD_NAME
        and receipt.get("audit_complete") is True
        and receipt.get("geometry_audit_performed") is True
        and receipt.get("geometry_audit_passed") is go
        and receipt.get("topology") == "WORLD8-DP2xUlysses-SP4"
        and receipt.get("frame_count") == 81
        and receipt.get("fps") == 25.0
        and receipt.get("primary_pair_ids") == list(runtime.PAIR_IDS)
        and receipt.get("primary_schedule_indices")
        == list(runtime.SCHEDULE_INDICES)
        and receipt.get("action_condition_count") == 8
        and receipt.get("identity_probe_count") == 64,
        "Phase-A2 exact81/WORLD8/8x64 closure differs",
    )
    for field in (
        "optimizer_constructed",
        "optimizer_step_called",
        "candidate_delta_constructed",
        "parameter_add_called",
        "parameter_mutation_performed",
        "parameter_update_authorized",
        "scientific_action_editing_success_claim",
        "global_population_go",
        "optimizer_authorized",
        "action_success_claimed",
    ):
        _require(receipt.get(field) is False, f"{field} must remain false")

    prereg = receipt.get("preregistration")
    _require(
        isinstance(prereg, Mapping)
        and _SHA256_RE.fullmatch(str(prereg.get("file_sha256"))) is not None
        and _SHA256_RE.fullmatch(str(prereg.get("preregistration_digest")))
        is not None
        and prereg.get("preregistration_alone_geometry_measurement_authorized")
        is False,
        "preregistration boundary differs",
    )
    bank = receipt.get("live_cast_bank_binding")
    _require(isinstance(bank, Mapping), "live CAST bank binding is absent")
    bank_digest = _seal(bank, label="live CAST bank binding")
    selected = bank.get("selected_events")
    _require(
        bank.get("schema_version") == runtime.BANK_BINDING_SCHEMA
        and bank.get("preregistration_alone_geometry_measurement_authorized")
        is False
        and bank.get("combined_read_only_geometry_measurement_authorized")
        is True
        and bank.get("all_forty_cast_children_semantically_validated") is True
        and bank.get("cast_candidate_receipt_count") == 40
        and bank.get("selected_event_count") == 4
        and isinstance(selected, list)
        and len(selected) == 4
        and bank.get("primary_pair_ids") == list(runtime.PAIR_IDS)
        and bank.get("primary_schedule_indices") == list(runtime.SCHEDULE_INDICES)
        and bank.get("pilot_schedule_index_33_reused") is False
        and bank.get("optimizer_constructed") is False
        and bank.get("parameter_mutation_performed") is False
        and bank.get("mask_flow_pose_track_or_trajectory_used") is False,
        "live CAST authority closure differs",
    )
    event_keys = set()
    for event in selected:
        _require(isinstance(event, Mapping), "selected CAST event type differs")
        prompts = event.get("prompt_by_branch")
        captions = event.get("full_t2v_caption_by_branch")
        _require(
            isinstance(prompts, Mapping)
            and set(prompts) == set(runtime.BRANCH_ORDER)
            and isinstance(captions, Mapping)
            and set(captions) == set(runtime.BRANCH_ORDER)
            and runtime.object_sha256(prompts) == event.get("prompt_bank_sha256")
            and runtime.object_sha256(captions)
            == event.get("raw_caption_bank_sha256")
            and all(
                _SHA256_RE.fullmatch(str(event.get(field))) is not None
                for field in (
                    "event_digest",
                    "cast_score_receipt_file_sha256",
                    "cast_score_receipt_digest",
                    "geometry_source_video_sha256",
                    "clean_latent_tensor_sha256",
                    "official_gaussian_tensor_sha256",
                    "source_noise_key_sha256",
                )
            )
            and type(event.get("generation_seed")) is int
            and event.get("generation_seed") >= 0
            and event.get("candidate_shape", [])[:3] == [1, 16, 21],
            "selected CAST prompt/caption/source/tensor binding differs",
        )
        for path_field, digest_field, label in (
            (
                "cast_score_receipt_path",
                "cast_score_receipt_file_sha256",
                "selected CAST child",
            ),
            ("source_video_path", "source_video_file_sha256", "source video"),
            ("clean_latent_path", "clean_latent_file_sha256", "clean latent"),
            (
                "official_gaussian_path",
                "official_gaussian_file_sha256",
                "official Gaussian",
            ),
        ):
            _rehash_bound_file(
                event.get(path_field), event.get(digest_field), label=label
            )
        event_keys.add(
            (
                event.get("pair_id"),
                event.get("source_sample_id"),
                event.get("event_id"),
                event.get("action_family"),
            )
        )
    _require(len(event_keys) == 4, "selected CAST event identities repeat")

    solver = receipt.get("world_solver_authority")
    _require(isinstance(solver, Mapping), "WORLD solver authority is absent")
    solver_digest = _seal(solver, label="WORLD solver authority")
    runtime_source = receipt.get("runtime_source")
    _require(isinstance(runtime_source, Mapping), "runtime source binding is absent")
    runtime_archive_sha = _sha(
        runtime_source.get("archive_sha256"), label="runtime archive"
    )
    _require(
        solver.get("schema_version") == runtime.WORLD_SOLVER_AUTHORITY_SCHEMA
        and solver.get("world_size") == 8
        and solver.get("input_consensus") is True
        and solver.get("input_consensus_rank_count") == 8
        and solver.get("final_toctou_bank_binding_digest") == bank_digest
        and solver.get("manifest_digest") == bank_digest
        and solver.get("runtime_source_archive_sha256") == runtime_archive_sha
        and solver.get("runtime_source_revision")
        == runtime_source.get("revision")
        and solver.get("solver_execution_rank") == 0
        and solver.get("solver_execution_device") == "cpu"
        and solver.get("solver_execution_count") == 1
        and solver.get("single_global_direction_solve") is True
        and solver.get("local_project_then_average") is False
        and solver.get("raw_gradient_artifact_written") is False
        and solver.get("safe_direction_artifact_written") is False
        and solver.get("phase_b_requires_independent_remeasurement") is True
        and solver.get("phase_b_must_apply_remeasured_direction_in_memory")
        is True
        and solver.get("receipt_can_reconstruct_safe_direction") is False,
        "WORLD rank-zero CPU authority closure differs",
    )
    transport = receipt.get("multicondition_transport_receipt")
    try:
        checked_transport, transport_digest, checked_go = (
            runtime.validate_root_solver_result(
                {
                    "ok": True,
                    "transport_receipt": transport,
                    "transport_receipt_digest": (
                        transport.get("receipt_digest")
                        if isinstance(transport, Mapping)
                        else None
                    ),
                    "primary_replication_go": go,
                },
                bank_binding_digest=bank_digest,
                expected_world_input_digest=solver.get("input_digest"),
            )
        )
    except Exception as error:
        raise PhaseA2PostflightError("multicondition transport differs") from error
    _require(checked_go is go, "transport GO/NO-GO differs")
    _require(
        solver.get("transport_receipt_digest") == transport_digest
        and checked_transport.get("validated_world_input_digest")
        == solver.get("input_digest"),
        "solver/transport digest binding differs",
    )
    failures = checked_transport.get("failure_codes")
    _require(
        isinstance(failures, list)
        and ((go and failures == []) or (not go and len(failures) > 0)),
        "GO/NO-GO failure-code semantics differ",
    )

    actions = receipt.get("action_gradient_metadata")
    identities = receipt.get("identity_probe_metadata")
    _require(
        isinstance(actions, list)
        and len(actions) == 8
        and isinstance(identities, list)
        and len(identities) == 64,
        "persisted gradient metadata count differs",
    )
    expected_actions = {
        (pair, source, schedule)
        for pair, source, _event, _family in event_keys
        for schedule in runtime.SCHEDULE_INDICES
    }
    observed_actions = {
        (row.get("pair_id"), row.get("source_sample_id"), row.get("schedule_index"))
        for row in actions
        if isinstance(row, Mapping)
    }
    _require(observed_actions == expected_actions, "action metadata factorial differs")
    try:
        coordinate_cells = runtime.validate_cross_family_identity_coordinate_closure(
            identities
        )
    except Exception as error:
        raise PhaseA2PostflightError(
            "identity cross-family coordinate closure differs"
        ) from error
    _require(len(coordinate_cells) == 8, "identity coordinate cell count differs")

    flow = receipt.get("gradient_information_flow")
    handoff = receipt.get("phase_b_handoff")
    _require(
        isinstance(flow, Mapping)
        and flow.get("pure_t2v_action_gradient_count") == 8
        and flow.get("deployment_identity_probe_count") == 64
        and flow.get("unprojected_rows_preserved_until_root_solver") is True
        and flow.get("local_project_then_average") is False
        and flow.get("mask_flow_pose_track_or_trajectory_used") is False
        and flow.get("raw_gradient_artifact_written") is False
        and handoff
        == {
            "phase_a2_safe_direction_persisted": False,
            "receipt_can_reconstruct_safe_direction": False,
            "if_primary_replication_go_then_next_job_must_remeasure": True,
            "phase_b_must_apply_remeasured_direction_in_memory": True,
            "phase_b_is_separate_root_authorized_job": True,
        },
        "gradient-flow/Phase-B handoff boundary differs",
    )
    ranks = receipt.get("rank_runtime_provenance")
    _require(
        isinstance(ranks, list)
        and len(ranks) == 8
        and all(isinstance(rows, list) and len(rows) == 4 for rows in ranks),
        "WORLD runtime provenance closure differs",
    )
    return {
        "schema_version": POSTFLIGHT_SCHEMA,
        "postflight_passed": True,
        "phase_a2_receipt_digest": receipt_digest,
        "live_cast_bank_binding_digest": bank_digest,
        "world_solver_authority_digest": solver_digest,
        "multicondition_transport_receipt_digest": transport_digest,
        "primary_replication_go": go,
        "phase_b_authorized": False,
        "phase_b_next_action": (
            "separate_job_remeasure_full_bank_then_apply_in_memory"
            if go
            else "terminate_no_go_no_update"
        ),
        "receipt_can_reconstruct_safe_direction": False,
        "parameter_mutation_performed": False,
        "scientific_action_editing_success_claim": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-runtime-archive-sha256", required=True)
    parser.add_argument("--expected-bank-binding-digest")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path, receipt, file_sha = _read_receipt(args.receipt)
    result = dict(validate_phase_a2_receipt(receipt))
    expected_runtime = _sha(
        args.expected_runtime_archive_sha256, label="expected runtime archive"
    )
    _require(
        receipt.get("runtime_source", {}).get("archive_sha256")
        == expected_runtime,
        "postflight runtime archive pin differs",
    )
    if args.expected_bank_binding_digest is not None:
        expected_bank = _sha(
            args.expected_bank_binding_digest, label="expected bank binding"
        )
        _require(
            result["live_cast_bank_binding_digest"] == expected_bank,
            "postflight bank binding pin differs",
        )
    print(
        json.dumps(
            {**result, "receipt_path": str(path), "receipt_file_sha256": file_sha},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "POSTFLIGHT_SCHEMA",
    "PhaseA2PostflightError",
    "build_parser",
    "main",
    "validate_phase_a2_receipt",
]

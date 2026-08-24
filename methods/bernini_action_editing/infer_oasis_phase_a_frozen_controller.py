#!/usr/bin/env python3
"""Preflight the Bernini OASIS Phase-A DP2 x SP4 frozen candidate scorer.

This executable intentionally stops at a hash-bound WORLD8 preflight.  The
tensor core in :mod:`oasis_phase_a_core` is executable and tested, but the
Bernini-specific backend still needs reviewed independent exact40 candidate
rollouts, final-clean capture, registered re-noising, and post-native-CFG/APG
branch capture.  Arbitrary-state velocity-norm scoring is explicitly NO-GO.
Running ordinary native inference and labelling it OASIS would be false
evidence, so this file fails closed instead.

No model forward, scheduler step, optimizer, backward, checkpoint write, or
action-editing success claim occurs here.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import oasis_phase_a_manifest as oasis_manifest  # noqa: E402


SCHEMA_VERSION = "bernini-oasis-phase-a-world8-preflight-v2"
EXPECTED_BERNINI_COMMIT = "2d2b4591ac053ec25c6371b01a5a6746679e5793"
EXPECTED_VEOMNI_COMMIT = "f90b3dc6fbb0ce693745223cc7a94064123dbf4d"

_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class OASISPhaseARuntimeError(RuntimeError):
    """The frozen-oracle preflight is incomplete or differs across ranks."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--expected-checkpoint-content-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-tree-sha256", required=True)
    parser.add_argument("--method-source-revision", required=True)
    parser.add_argument("--method-source-archive-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--ack-frozen-no-training-no-success-claim", action="store_true")
    parser.add_argument("--ack-bernini-controller-backend-not-yet-executed", action="store_true")
    return parser


def _sha(value: Any, *, length: int, label: str) -> str:
    pattern = _SHA1_RE if length == 40 else _SHA256_RE
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise OASISPhaseARuntimeError(f"{label} must be lowercase SHA-{length}")
    return value


def _plain_absolute(value: Any, *, label: str, directory: bool) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise OASISPhaseARuntimeError(f"{label} must be absolute")
    path = Path(value)
    valid = path.is_dir() if directory else path.is_file()
    if not valid or path.is_symlink():
        raise OASISPhaseARuntimeError(f"{label} must be an existing plain {'directory' if directory else 'file'}")
    return path.resolve(strict=True)


def _file_sha256(path: Path) -> str:
    return oasis_manifest.file_sha256(path)


def runtime_boundary_contract() -> Mapping[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "phase": "preflight_only",
        "world_size": 8,
        "dp_size": 2,
        "sequence_parallel_size": 4,
        "exact81": True,
        "exact40": True,
        "low_sigma_exact_base_off_indices": [38, 39],
        "planned_family_by_dp_rank": {
            "0": "dog_sit_hold",
            "1": "human_stand_hold",
        },
        "planned_cells_per_family": 12,
        "planned_total_rollouts": 24,
        "completed_total_rollouts": 0,
        "controller_tensor_core_executed": False,
        "bernini_model_loaded": False,
        "model_forward_count": 0,
        "native_scheduler_step_count": 0,
        "optimizer_constructed": False,
        "backward_count": 0,
        "parameter_mutation_performed": False,
        "missing_reviewed_backend": [
            "independent_full_exact40_candidate_grid_from_same_registered_gaussian",
            "source_set_only_motion_null_appearance_noise_native_injector",
            "detached_fp32_final_clean_exact81_capture",
            "candidate_owned_registered_epsilon_sigma_renoise_probe",
            "post_native_cfg_apg_branch_capture_for_t2v_rv2v_refdrop",
            "same_object_candidate_xsigma_frozen_expert_query",
            "endpoint_exact81_decode_and_detached_action_source_camera_audit",
        ],
        "candidate_score_coordinate": "final_clean_candidate_registered_renoise",
        "known_probe_target_velocity": "epsilon-z0",
        "arbitrary_native_state_velocity_norm_oracle": "NO-GO",
        "shallow_unipc_history_fork": "NO-GO",
        "t2v_scalar_calibration_required": True,
        "old_training_authority_accepted": False,
        "rho_zero_exact_native_gaussian_control_required": True,
        "active_rho_external_initial_noise_injection": True,
        "active_rho_legacy_native_endpoint_schema_accepted": False,
        "wrongref_proxy_used_for_authorization": False,
        "target_or_paired_target_consumed": False,
        "t2v_media_or_latent_consumed_by_rv2v": False,
        "mask_flow_pose_track_consumed": False,
        "frozen_oracle_execution_authorized": False,
        "training_authorized": False,
        "scientific_action_editing_success_claim": False,
    }
    return {**value, "receipt_digest": oasis_manifest.object_sha256(value)}


def _validate_static_inputs(
    args: argparse.Namespace,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if args.preflight_only is not True:
        raise OASISPhaseARuntimeError(
            "only --preflight-only is implemented; Bernini controller execution is not yet wired"
        )
    if args.ack_frozen_no_training_no_success_claim is not True:
        raise OASISPhaseARuntimeError(
            "acknowledge frozen/no-training/no-success-claim semantics"
        )
    if args.ack_bernini_controller_backend_not_yet_executed is not True:
        raise OASISPhaseARuntimeError(
            "acknowledge that this run does not execute the Bernini controller backend"
        )
    manifest = oasis_manifest.load_phase_a_manifest(
        args.manifest, args.expected_manifest_sha256, verify_files=True
    )
    try:
        scalar_evidence = oasis_manifest.load_dedicated_scalar_calibration_evidence(
            manifest
        )
    except Exception as error:
        raise OASISPhaseARuntimeError(
            f"dedicated T2V scalar calibration prerequisite failed: {error}"
        ) from error
    expected_tree = _sha(
        args.expected_checkpoint_tree_sha256, length=64, label="checkpoint tree"
    )
    if manifest.checkpoint_tree_sha256 != expected_tree:
        raise OASISPhaseARuntimeError("manifest/runtime checkpoint tree differs")
    checkpoint = _plain_absolute(args.checkpoint, label="checkpoint", directory=True)
    checkpoint_manifest = _plain_absolute(
        args.checkpoint_content_manifest,
        label="checkpoint content manifest",
        directory=False,
    )
    expected_checkpoint_manifest_sha = _sha(
        args.expected_checkpoint_content_manifest_sha256,
        length=64,
        label="checkpoint manifest SHA",
    )
    if _file_sha256(checkpoint_manifest) != expected_checkpoint_manifest_sha:
        raise OASISPhaseARuntimeError("checkpoint content manifest SHA differs")
    bernini_root = _plain_absolute(
        args.bernini_root, label="Bernini source root", directory=True
    )
    veomni_root = _plain_absolute(
        args.veomni_root, label="VeOmni source root", directory=True
    )

    import infer_source_kv_carrier_oracle as checkpoint_audit
    import train_lora as trainer

    try:
        checked_bernini, checked_veomni, bernini_revision, veomni_revision = (
            trainer.validate_source_trees(
                str(bernini_root),
                str(veomni_root),
                expected_bernini_commit=EXPECTED_BERNINI_COMMIT,
                expected_veomni_commit=EXPECTED_VEOMNI_COMMIT,
            )
        )
        checked_checkpoint, transformer_config = trainer.validate_checkpoint(
            str(checkpoint)
        )
        checkpoint_identity = checkpoint_audit.validate_checkpoint_content(
            checked_checkpoint,
            checkpoint_manifest,
            expected_manifest_sha256=expected_checkpoint_manifest_sha,
        )
    except Exception as error:
        raise OASISPhaseARuntimeError(f"authoritative preflight failed: {error}") from error
    if (
        str(checked_bernini) != str(bernini_root)
        or str(checked_veomni) != str(veomni_root)
        or str(checked_checkpoint) != str(checkpoint)
        or bernini_revision != EXPECTED_BERNINI_COMMIT
        or veomni_revision != EXPECTED_VEOMNI_COMMIT
        or int(transformer_config.get("num_attention_heads", 0)) % 4
    ):
        raise OASISPhaseARuntimeError("Bernini/VeOmni/checkpoint SP4 binding differs")
    checkpoint_receipt = {
        "checkpoint": str(checkpoint),
        "checkpoint_tree_sha256": expected_tree,
        "checkpoint_content_manifest_sha256": expected_checkpoint_manifest_sha,
        "checkpoint_content_receipt_digest": oasis_manifest.object_sha256(
            checkpoint_identity
        ),
    }
    provenance = {
        "method_source_revision": _sha(
            args.method_source_revision, length=40, label="method source revision"
        ),
        "method_source_archive_sha256": _sha(
            args.method_source_archive_sha256,
            length=64,
            label="method source archive SHA",
        ),
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
    }
    scalar_receipt = {
        "status": manifest.scalar_calibration_status,
        "path": str(manifest.scalar_calibration_path),
        "file_sha256": manifest.scalar_calibration_file_sha256,
        "evidence_digest": manifest.scalar_calibration_evidence_digest,
        "formal_scalar_source": dict(scalar_evidence["formal_scalar_source"]),
        "optimizer_authorized": False,
        "calibration_media_latent_or_gaussian_consumed": False,
    }
    return manifest, checkpoint_receipt, provenance, scalar_receipt


def _rank_plan(manifest: Any, *, rank: int, world_size: int) -> Mapping[str, Any]:
    if world_size != 8 or not 0 <= rank < world_size:
        raise OASISPhaseARuntimeError("runtime topology must be WORLD8")
    dp_rank = rank // 4
    sp_rank = rank % 4
    family = oasis_manifest.FAMILY_ORDER[dp_rank]
    cells = manifest.cells_for_family(family)
    if tuple(cell.analysis_split for cell in cells) != oasis_manifest.SPLIT_ORDER:
        raise OASISPhaseARuntimeError("DP family lacks fit/confirmation source cells")
    schedule = [
        {
            "sample_id": cell.sample_id,
            "analysis_split": cell.analysis_split,
            "seed": seed,
            "arm": arm,
        }
        for cell in cells
        for seed in manifest.seed_order
        for arm in oasis_manifest.ARM_ORDER
    ]
    if len(schedule) != 12:
        raise OASISPhaseARuntimeError("per-family Phase-A schedule is not exact12")
    value = {
        "rank": rank,
        "world_size": world_size,
        "dp_rank": dp_rank,
        "sp_rank": sp_rank,
        "family": family,
        "planned_cells": schedule,
        "planned_cell_count": len(schedule),
        "model_forward_count": 0,
        "scheduler_step_count": 0,
    }
    return {**value, "plan_digest": oasis_manifest.object_sha256(value)}


def _write_rank_zero_receipt(
    output: Path,
    *,
    manifest: Any,
    checkpoint_receipt: Mapping[str, Any],
    provenance: Mapping[str, Any],
    scalar_calibration_receipt: Mapping[str, Any],
    rank_plans: Sequence[Mapping[str, Any]],
) -> Path:
    if output.exists() or output.is_symlink() or not output.is_absolute() or output == Path("/"):
        raise OASISPhaseARuntimeError("output must be a fresh absolute non-root path")
    expected_ranks = list(range(8))
    if [row.get("rank") for row in rank_plans] != expected_ranks:
        raise OASISPhaseARuntimeError("WORLD8 rank plan closure differs")
    for dp_rank, family in enumerate(oasis_manifest.FAMILY_ORDER):
        rows = [row for row in rank_plans if row.get("dp_rank") == dp_rank]
        if (
            len(rows) != 4
            or {row.get("sp_rank") for row in rows} != {0, 1, 2, 3}
            or {row.get("family") for row in rows} != {family}
            or len({row.get("plan_digest") for row in rows}) != 4
        ):
            # Digests include global rank/SP rank, hence four distinct values.
            raise OASISPhaseARuntimeError("DP2 x SP4 family/rank closure differs")
        schedules = [row.get("planned_cells") for row in rows]
        if any(schedule != schedules[0] for schedule in schedules[1:]):
            raise OASISPhaseARuntimeError("SP4 ranks disagree on family schedule")
    manifest.assert_unchanged()
    boundary = runtime_boundary_contract()
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest.path),
        "manifest_file_sha256": manifest.file_sha256,
        "manifest_digest": manifest.manifest_digest,
        "checkpoint": dict(checkpoint_receipt),
        "provenance": dict(provenance),
        "t2v_scalar_calibration": dict(scalar_calibration_receipt),
        "topology": oasis_manifest.TOPOLOGY,
        "rank_plans": list(rank_plans),
        "boundary_contract": boundary,
        "preflight_complete": True,
        "artifact_kind": "preflight_receipt_not_oracle_result",
        "controller_runtime_executed": False,
        "frozen_oracle_rollouts_executed": 0,
        "training_performed": False,
        "optimizer_constructed": False,
        "parameter_mutation_performed": False,
        "scientific_action_editing_success_claim": False,
    }
    receipt = {**unsigned, "receipt_digest": oasis_manifest.object_sha256(unsigned)}
    output.mkdir(parents=False, exist_ok=False)
    path = output / "preflight-receipt.json"
    payload = oasis_manifest.canonical_json_bytes(receipt) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o400)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if rank < 0 or world_size != 8:
        raise OASISPhaseARuntimeError("preflight must run under torchrun WORLD8")
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )
    try:
        # Hashing the full checkpoint and checking the dedicated scalar-only
        # evidence graph is rank-zero authority work.  Doing it independently
        # on all eight ranks would multiply filesystem traffic without adding
        # evidence.  The authenticated result is broadcast before any rank
        # constructs its deterministic family plan.
        authority: list[Any] = [None]
        if rank == 0:
            try:
                (
                    manifest,
                    checkpoint_receipt,
                    provenance,
                    scalar_calibration_receipt,
                ) = _validate_static_inputs(args)
                authority[0] = {
                    "ok": True,
                    "manifest": manifest,
                    "checkpoint_receipt": checkpoint_receipt,
                    "provenance": provenance,
                    "scalar_calibration_receipt": scalar_calibration_receipt,
                }
            except Exception as error:
                authority[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(authority, src=0)
        checked = authority[0]
        if not isinstance(checked, Mapping) or checked.get("ok") is not True:
            raise OASISPhaseARuntimeError(f"rank-zero authority failed: {checked}")
        manifest = checked["manifest"]
        checkpoint_receipt = checked["checkpoint_receipt"]
        provenance = checked["provenance"]
        scalar_calibration_receipt = checked["scalar_calibration_receipt"]
        local_plan = _rank_plan(manifest, rank=rank, world_size=world_size)
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local_plan)
        gathered.sort(key=lambda row: int(row["rank"]))
        result: list[Any] = [None]
        if rank == 0:
            try:
                output = Path(args.output)
                path = _write_rank_zero_receipt(
                    output,
                    manifest=manifest,
                    checkpoint_receipt=checkpoint_receipt,
                    provenance=provenance,
                    scalar_calibration_receipt=scalar_calibration_receipt,
                    rank_plans=gathered,
                )
                result[0] = {
                    "ok": True,
                    "receipt": str(path),
                    "receipt_file_sha256": _file_sha256(path),
                }
            except Exception as error:
                result[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        dist.broadcast_object_list(result, src=0)
        if not isinstance(result[0], Mapping) or result[0].get("ok") is not True:
            raise OASISPhaseARuntimeError(f"rank-zero receipt failed: {result[0]}")
        print(
            json.dumps(
                {
                    "rank": rank,
                    "family": local_plan["family"],
                    "preflight_complete": True,
                    "controller_runtime_executed": False,
                    "training_performed": False,
                    "scientific_action_editing_success_claim": False,
                    "receipt": result[0]["receipt"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

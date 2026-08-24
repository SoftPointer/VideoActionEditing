#!/usr/bin/env python3
"""Pure postflight replay for the v15c-r8 LOCAL-schema bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from . import materialize_source_sam2_proposal_tracks_v15c_r8 as materializer
    from . import run_source_object_proposal_role_probe_v15c_r8 as runner
    from . import source_object_proposal_role_probe_v15c as core
except ImportError:  # pragma: no cover
    import materialize_source_sam2_proposal_tracks_v15c_r8 as materializer
    import run_source_object_proposal_role_probe_v15c_r8 as runner
    import source_object_proposal_role_probe_v15c as core


SCHEMA = "bernini-source-sam2-proposal-role-postflight-v15c-r8-local"
GATE_KEYS = (
    "spec_raw_and_canonical_pins",
    "source_and_r6_pins",
    "track_receipt_exact_schema_and_self_hash",
    "track_output_and_artifact_manifests",
    "one_to_64_full_sha_sorted_proposals",
    "both_repeat_transcripts_rebuilt",
    "all_prompt_and_p_times_81_mask_bytes_reopened",
    "all_geometry_and_whole_object_gates_recomputed",
    "all_phase_coverage_recomputed",
    "raw_amg_masks_scores_area_bbox_and_admission_replayed",
    "all_prompt_and_propagation_safetensors_reopened",
    "all_propagation_masks_derived_from_logit_sign",
    "all_model_param_buffer_manifests_and_freeze_replayed",
    "two_complete_tensor_runs_semantically_equal",
    "local_schema_only_remote_observer_unverified",
    "source_family_overlap_nesting_fail_closed",
    "r6_core_result_replayed",
)
POSTFLIGHT_KEYS = (
    "schema_version",
    "status",
    "gates",
    "file_sha256",
    "mechanical_candidate_qualified",
    "assignments_for_reject_only_audit",
    "human_audit_action",
    "human_audit_may_authorize_route",
    "localization_semantically_certified",
    "action_success_certified",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "optimizer_updates",
    "renderer_forward_calls",
    "receipt_sha256",
)


class PostflightSourceProposalRoleV15CR2Error(RuntimeError):
    """Published bytes do not replay to the sealed observer result."""


def _read(path: Path) -> Mapping[str, Any]:
    value = runner.read_json(path)
    return value


def _verify_result_self_hash(result: Mapping[str, Any]) -> None:
    payload = dict(result)
    claimed = payload.pop("receipt_sha256", None)
    if (
        not isinstance(claimed, str)
        or materializer.SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != core.object_sha256(payload)
    ):
        raise PostflightSourceProposalRoleV15CR2Error("result self-hash differs")


def replay_postflight(
    *,
    spec_path: Path,
    source_video: Path,
    r6_receipt_path: Path,
    r6_tensors: Path,
    track_receipt_path: Path,
    track_tensors: Path,
    result_json: Path,
) -> Mapping[str, Any]:
    """Pure byte replay called by both the CLI and the sealed finalizer."""

    paths = {
        name: value.resolve(strict=True)
        for name, value in {
            "spec": spec_path,
            "source": source_video,
            "r6_receipt": r6_receipt_path,
            "r6_tensors": r6_tensors,
            "track_receipt": track_receipt_path,
            "track_tensors": track_tensors,
            "result": result_json,
        }.items()
    }
    spec = materializer.read_spec(paths["spec"])
    if (
        not os.path.samefile(paths["source"], Path(spec["source"]["path"]))
        or core.file_sha256(paths["source"]) != spec["source"]["sha256"]
        or core.file_sha256(paths["r6_receipt"])
        != spec["r6"]["probe_receipt_file_sha256"]
        or core.file_sha256(paths["r6_tensors"])
        != spec["r6"]["affinity_tensor_file_sha256"]
    ):
        raise PostflightSourceProposalRoleV15CR2Error("sealed input binding differs")

    r6_receipt = _read(paths["r6_receipt"])
    track_receipt = _read(paths["track_receipt"])
    runner.validate_r6_receipt(r6_receipt, spec, paths["r6_tensors"])
    # This independently reopens every prompt/mask PNG and recomputes all 81
    # geometries plus every 21x37x25 phase-coverage value.
    local_evidence_replay = runner.validate_track_bundle(
        track_receipt, spec, paths["track_receipt"], paths["track_tensors"]
    )
    if (
        local_evidence_replay["status"]
        != "LOCAL_SCHEMA_REPLAY_PASS_REMOTE_OBSERVER_UNVERIFIED"
        or local_evidence_replay["observer_execution_authorized"] is not False
    ):
        raise PostflightSourceProposalRoleV15CR2Error(
            "LOCAL evidence replay claim boundary differs"
        )
    tracks, replayed_track_receipt = core.load_tracks_for_v15c(
        paths["track_receipt"], paths["track_tensors"]
    )
    if replayed_track_receipt != track_receipt:
        raise PostflightSourceProposalRoleV15CR2Error("track reload differs")
    affinity = core.load_r6_affinity_for_v15c(paths["r6_tensors"])
    replay = dict(
        core.run_source_object_proposal_role_probe_v15c(
            tracks=tracks,
            affinity=affinity,
            thresholds=runner.thresholds_from_spec(spec),
        )
    )
    replay_core_receipt = replay.pop("receipt_sha256")

    result = _read(paths["result"])
    _verify_result_self_hash(result)
    provenance = result.get("provenance")
    observed_core = {
        key: value
        for key, value in result.items()
        if key not in {"provenance", "receipt_sha256"}
    }
    expected_provenance = {
        "spec_raw_sha256": core.file_sha256(paths["spec"]),
        "spec_canonical_sha256": core.object_sha256(spec),
        "source_video_sha256": spec["source"]["sha256"],
        "source_text_provenance_sha256": spec["r6"][
            "source_text_provenance_sha256"
        ],
        "r6_receipt_file_sha256": core.file_sha256(paths["r6_receipt"]),
        "r6_affinity_file_sha256": core.file_sha256(paths["r6_tensors"]),
        "r6_internal_receipt_sha256": r6_receipt["receipt_sha256"],
        "track_receipt_file_sha256": core.file_sha256(paths["track_receipt"]),
        "track_tensor_file_sha256": core.file_sha256(paths["track_tensors"]),
        "track_internal_receipt_sha256": track_receipt["receipt_sha256"],
        "track_output_manifest_file_sha256": core.file_sha256(
            paths["track_receipt"].parent / "output_manifest.json"
        ),
        "observer_evidence_file_sha256": core.file_sha256(
            paths["track_receipt"].parent
            / "observer_evidence/local_evidence.json"
        ),
        "observer_evidence_internal_sha256": track_receipt[
            "observer_evidence_internal_sha256"
        ],
        "observer_evidence_replay_sha256": local_evidence_replay[
            "receipt_sha256"
        ],
        "assignment_core_receipt_sha256": replay_core_receipt,
    }
    if (
        observed_core != replay
        or provenance != expected_provenance
        or result.get("route_authorized") is not False
        or result.get("training_authorized") is not False
        or result.get("decode_authorized") is not False
    ):
        raise PostflightSourceProposalRoleV15CR2Error("assignment replay differs")

    gates = {
        "spec_raw_and_canonical_pins": True,
        "source_and_r6_pins": True,
        "track_receipt_exact_schema_and_self_hash": True,
        "track_output_and_artifact_manifests": True,
        "one_to_64_full_sha_sorted_proposals": True,
        "both_repeat_transcripts_rebuilt": True,
        "all_prompt_and_p_times_81_mask_bytes_reopened": True,
        "all_geometry_and_whole_object_gates_recomputed": True,
        "all_phase_coverage_recomputed": True,
        "raw_amg_masks_scores_area_bbox_and_admission_replayed": True,
        "all_prompt_and_propagation_safetensors_reopened": True,
        "all_propagation_masks_derived_from_logit_sign": True,
        "all_model_param_buffer_manifests_and_freeze_replayed": True,
        "two_complete_tensor_runs_semantically_equal": True,
        "local_schema_only_remote_observer_unverified": True,
        "source_family_overlap_nesting_fail_closed": True,
        "r6_core_result_replayed": True,
    }
    runner.require_exact_keys(gates, GATE_KEYS, "postflight gates")
    if not gates or any(type(value) is not bool or value is not True for value in gates.values()):
        raise PostflightSourceProposalRoleV15CR2Error("postflight gates differ")
    receipt = {
        "schema_version": SCHEMA,
        "status": "LOCAL_SCHEMA_POSTFLIGHT_PASS_REMOTE_OBSERVER_UNVERIFIED",
        "gates": gates,
        "file_sha256": {
            name: core.file_sha256(path) for name, path in paths.items()
        },
        "mechanical_candidate_qualified": result.get(
            "mechanical_candidate_qualified"
        ),
        "assignments_for_reject_only_audit": result.get("assignments"),
        "human_audit_action": "reject_only",
        "human_audit_may_authorize_route": False,
        "localization_semantically_certified": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    runner.require_exact_keys(receipt, POSTFLIGHT_KEYS, "postflight receipt")
    runner.require_sha256(receipt["receipt_sha256"], "postflight receipt hash")
    if set(receipt["file_sha256"]) != set(paths):
        raise PostflightSourceProposalRoleV15CR2Error("postflight file registry differs")
    for name, digest in receipt["file_sha256"].items():
        runner.require_sha256(digest, f"postflight {name} hash")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--r6-receipt", required=True, type=Path)
    parser.add_argument("--r6-tensors", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--track-tensors", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_json.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise PostflightSourceProposalRoleV15CR2Error("postflight output is not fresh")
    receipt = replay_postflight(
        spec_path=args.spec,
        source_video=args.source_video,
        r6_receipt_path=args.r6_receipt,
        r6_tensors=args.r6_tensors,
        track_receipt_path=args.track_receipt,
        track_tensors=args.track_tensors,
        result_json=args.result_json,
    )
    output.write_bytes(core.canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

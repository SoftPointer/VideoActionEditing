#!/usr/bin/env python3
"""Replay r8 LOCAL bytes and run the r9 source-only four-role authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from . import materialize_source_sam2_proposal_tracks_v15c_r8 as r8_materializer
    from . import run_source_object_proposal_role_probe_v15c_r8 as r8_runner
    from . import source_role_authority_v15c_r9 as core
    from . import validate_source_role_authority_v15c_r9 as authority_contract
except ImportError:  # pragma: no cover - flat sealed snapshot
    import materialize_source_sam2_proposal_tracks_v15c_r8 as r8_materializer
    import run_source_object_proposal_role_probe_v15c_r8 as r8_runner
    import source_role_authority_v15c_r9 as core
    import validate_source_role_authority_v15c_r9 as authority_contract


class RunSourceFourRoleAuthorityV15CR9Error(RuntimeError):
    """One byte-derived r8/r6/r9 authority input differs."""


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        return authority_contract.read_json(path)
    except authority_contract.ValidateSourceRoleAuthorityV15CR9Error as error:
        raise RunSourceFourRoleAuthorityV15CR9Error(str(error)) from error


def verify_self_hash(value: Mapping[str, Any], field: str = "receipt_sha256") -> None:
    payload = dict(value)
    claimed = payload.pop(field, None)
    if (
        type(claimed) is not str
        or core.SHA256_PATTERN.fullmatch(claimed) is None
        or claimed != core.object_sha256(payload)
    ):
        raise RunSourceFourRoleAuthorityV15CR9Error(f"{field} differs")


def replay_assignment(
    *,
    repo_root: Path,
    authority_path: Path,
    r6_receipt_path: Path,
    r6_tensors_path: Path,
    track_receipt_path: Path,
    track_tensors_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Pure replay used by the CLI, ownership builder, and postflight."""

    try:
        authority, base_spec, role_asset = authority_contract.validate_authority(
            root=repo_root, authority_path=authority_path
        )
        r8_materializer.validate_spec(base_spec)
    except (
        authority_contract.ValidateSourceRoleAuthorityV15CR9Error,
        r8_materializer.SourceSAM2ProposalTracksV15CError,
    ) as error:
        raise RunSourceFourRoleAuthorityV15CR9Error(str(error)) from error
    resolved = {
        "authority": authority_path.resolve(strict=True),
        "r6_receipt": r6_receipt_path.resolve(strict=True),
        "r6_tensors": r6_tensors_path.resolve(strict=True),
        "track_receipt": track_receipt_path.resolve(strict=True),
        "track_tensors": track_tensors_path.resolve(strict=True),
    }
    if (
        core.file_sha256(resolved["r6_receipt"])
        != authority["r6_affinity_authority"]["probe_receipt_file_sha256"]
        or core.file_sha256(resolved["r6_tensors"])
        != authority["r6_affinity_authority"]["affinity_tensor_file_sha256"]
    ):
        raise RunSourceFourRoleAuthorityV15CR9Error("r6 byte authority differs")
    r6_receipt = read_json(resolved["r6_receipt"])
    track_receipt = read_json(resolved["track_receipt"])
    try:
        r8_runner.validate_r6_receipt(
            r6_receipt, base_spec, resolved["r6_tensors"]
        )
        local_evidence_replay = r8_runner.validate_track_bundle(
            track_receipt,
            base_spec,
            resolved["track_receipt"],
            resolved["track_tensors"],
        )
    except r8_runner.RunSourceProposalRoleProbeV15CError as error:
        raise RunSourceFourRoleAuthorityV15CR9Error(str(error)) from error
    if (
        local_evidence_replay.get("status")
        != "LOCAL_SCHEMA_REPLAY_PASS_REMOTE_OBSERVER_UNVERIFIED"
        or local_evidence_replay.get("remote_worker_execution_verified") is not False
        or local_evidence_replay.get("observer_execution_authorized") is not False
    ):
        raise RunSourceFourRoleAuthorityV15CR9Error(
            "r8 LOCAL replay claim boundary differs"
        )
    tracks, replayed_track = core.load_tracks_for_v15c_r9(
        resolved["track_receipt"], resolved["track_tensors"]
    )
    if replayed_track != track_receipt:
        raise RunSourceFourRoleAuthorityV15CR9Error("track reload differs")
    r6_authority = authority["r6_affinity_authority"]
    null_alignment = bool(
        r6_receipt.get("null_registry_sha256")
        == r6_authority["null_registry_sha256"]
        and r6_receipt.get("null_span_count") == core.NULL_COUNT
        and r6_receipt.get("selected_block_indices") == list(core.BLOCK_INDICES)
        and r6_receipt.get("role_names") == list(core.FULL_R6_ROLE_NAMES)
        and r6_authority["null_index_alignment"]
        == "same_preregistered_null_registry_index_j_across_blocks_phases_and_proposals_only"
        and r6_authority["four_role_joint_null_axis_available"] is False
        and r6_authority["common_null_broadcast_for_certification"] is False
        and r6_authority["fwer_status"]
        == "FOUR_ROLE_JOINT_FWER_UNCERTIFIED"
    )
    affinity = core.load_r6_affinity_for_v15c_r9(
        resolved["r6_tensors"],
        null_registry_sha256=r6_authority["null_registry_sha256"],
        null_index_alignment_verified=null_alignment,
    )
    thresholds = authority_contract.thresholds_from_authority(authority)
    result = dict(
        core.run_source_four_role_authority_v15c_r9(
            tracks=tracks, affinity=affinity, thresholds=thresholds
        )
    )
    if (
        affinity.four_role_joint_null_available is not False
        or result.get("multiple_comparison_control", {}).get(
            "global_four_role_fwer_certified"
        )
        is not False
        or result.get("multiple_comparison_control", {}).get("method")
        != "NO_GO_existing_r6_common_null_lacks_four_role_joint_axis"
        or result.get("role_assignment_mechanical_candidate_qualified") is not False
        or any(value is not None for value in result.get("assignments", {}).values())
    ):
        raise RunSourceFourRoleAuthorityV15CR9Error(
            "existing r6 common-null diagnostic was reinterpreted as joint FWER"
        )
    core_receipt_sha256 = result.pop("receipt_sha256")
    track_root = resolved["track_receipt"].parent
    result["provenance"] = {
        "authority_raw_sha256": core.file_sha256(resolved["authority"]),
        "authority_canonical_sha256": core.object_sha256(authority),
        "r8_base_spec_raw_sha256": authority["r8_local_replay_base"][
            "spec_raw_sha256"
        ],
        "role_asset_file_sha256": authority["token_source_authority"][
            "asset_file_sha256"
        ],
        "role_asset_internal_sha256": role_asset["asset_sha256"],
        "source_video_sha256": authority["source"]["video_sha256"],
        "source_text_provenance_sha256": authority["token_source_authority"][
            "source_text_provenance_sha256"
        ],
        "r6_receipt_file_sha256": core.file_sha256(resolved["r6_receipt"]),
        "r6_affinity_file_sha256": core.file_sha256(resolved["r6_tensors"]),
        "r6_internal_receipt_sha256": r6_receipt["receipt_sha256"],
        "null_registry_sha256": r6_authority["null_registry_sha256"],
        "r6_common_null_scope": "COMMON_NULL_DIAGNOSTIC_ONLY",
        "four_role_joint_fwer_certified": False,
        "track_receipt_file_sha256": core.file_sha256(resolved["track_receipt"]),
        "track_tensor_file_sha256": core.file_sha256(resolved["track_tensors"]),
        "track_internal_receipt_sha256": track_receipt["receipt_sha256"],
        "track_output_manifest_file_sha256": core.file_sha256(
            track_root / "output_manifest.json"
        ),
        "observer_evidence_file_sha256": core.file_sha256(
            track_root / "observer_evidence/local_evidence.json"
        ),
        "observer_evidence_internal_sha256": track_receipt[
            "observer_evidence_internal_sha256"
        ],
        "observer_evidence_replay_sha256": local_evidence_replay[
            "receipt_sha256"
        ],
        "assignment_core_receipt_sha256": core_receipt_sha256,
    }
    # These fields are repeated after composition so no caller can reinterpret
    # a local mechanical receipt as execution authority.
    result["mechanical_candidate_qualified"] = False
    result["remote_worker_execution_verified"] = False
    result["observer_execution_authorized"] = False
    result["localization_semantically_certified"] = False
    result["scientific_claim_authorized"] = False
    result["route_authorized"] = False
    result["decode_authorized"] = False
    result["training_authorized"] = False
    result["receipt_sha256"] = core.object_sha256(result)
    return result, local_evidence_replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--r6-receipt", required=True, type=Path)
    parser.add_argument("--r6-tensors", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--track-tensors", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_json.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise RunSourceFourRoleAuthorityV15CR9Error("output path is not fresh")
    result, _local_evidence = replay_assignment(
        repo_root=args.repo_root,
        authority_path=args.authority,
        r6_receipt_path=args.r6_receipt,
        r6_tensors_path=args.r6_tensors,
        track_receipt_path=args.track_receipt,
        track_tensors_path=args.track_tensors,
    )
    output.write_bytes(core.canonical_bytes(result))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

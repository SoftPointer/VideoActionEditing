#!/usr/bin/env python3
"""Future joint-null postflight replay for r9 assignment and ownership.

The sealed r6 common-null artifact is diagnostic-only and cannot reach this
ownership postflight.  This module preserves the exact future role-indexed-null
ABI without upgrading the current LOCAL result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from . import materialize_source_role_ownership_v15c_r9 as ownership_builder
    from . import run_source_four_role_authority_v15c_r9 as runner
    from . import source_role_authority_v15c_r9 as core
    from . import validate_source_role_authority_v15c_r9 as authority_contract
except ImportError:  # pragma: no cover
    import materialize_source_role_ownership_v15c_r9 as ownership_builder
    import run_source_four_role_authority_v15c_r9 as runner
    import source_role_authority_v15c_r9 as core
    import validate_source_role_authority_v15c_r9 as authority_contract


SCHEMA = "bernini-source-four-role-postflight-v15c-r9-local"
GATE_KEYS = (
    "authority_spec_token_and_source_bytes",
    "r8_local_tensor_replay_remote_unverified",
    "r6_human_and_three_vessel_channels",
    "exact_role_indexed_64_null_global_four_role_max_t",
    "vessel_three_role_bonferroni_extra_gate",
    "human_81_frame_area_continuity_topology_gates",
    "same_role_duplicate_nesting_fail_closed",
    "cross_role_conflict_and_runner_up_fail_closed",
    "limited_human_vessel_overlap_is_relation_evidence",
    "three_vessel_roles_strictly_mutually_exclusive",
    "both_selected_raw_logit_runs_reopened_and_equal",
    "published_masks_are_exact_logit_positive_sets",
    "ownership_strict_argmax_or_unassigned",
    "ownership_connectivity_hole_area_continuity_replayed",
    "four_role_final_ownership_pairwise_exclusive",
    "contact_relation_mask_independent",
    "v15b_adapter_never_receives_raw_overlap",
    "no_morphological_repair",
    "local_observer_claim_boundary",
)
POSTFLIGHT_KEYS = (
    "schema_version",
    "status",
    "gates",
    "file_sha256",
    "local_evidence_replay_sha256",
    "assignment_result_internal_sha256",
    "ownership_result_internal_sha256",
    "mechanical_candidate_qualified",
    "human_audit_action",
    "human_audit_may_authorize_route",
    "local_schema_replay_only",
    "remote_worker_execution_verified",
    "observer_execution_authorized",
    "localization_semantically_certified",
    "scientific_claim_authorized",
    "action_success_certified",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "optimizer_updates",
    "renderer_forward_calls",
    "receipt_sha256",
)


class PostflightSourceFourRoleV15CR9Error(RuntimeError):
    """The r9 published bytes do not replay exactly."""


def _read(path: Path) -> Mapping[str, Any]:
    try:
        value = runner.read_json(path.resolve(strict=True))
        runner.verify_self_hash(value)
    except runner.RunSourceFourRoleAuthorityV15CR9Error as error:
        raise PostflightSourceFourRoleV15CR9Error(str(error)) from error
    return value


def replay_postflight(
    *,
    repo_root: Path,
    authority_path: Path,
    r6_receipt_path: Path,
    r6_tensors_path: Path,
    track_receipt_path: Path,
    track_tensors_path: Path,
    assignment_result_path: Path,
    ownership_dir: Path,
) -> Mapping[str, Any]:
    import numpy as np
    replayed_assignment, local_evidence = runner.replay_assignment(
        repo_root=repo_root,
        authority_path=authority_path,
        r6_receipt_path=r6_receipt_path,
        r6_tensors_path=r6_tensors_path,
        track_receipt_path=track_receipt_path,
        track_tensors_path=track_tensors_path,
    )
    published_assignment = _read(assignment_result_path)
    if replayed_assignment != published_assignment:
        raise PostflightSourceFourRoleV15CR9Error("assignment replay differs")
    ownership_root = ownership_dir.resolve(strict=True)
    ownership_receipt_path = ownership_root / "ownership_receipt.json"
    ownership_tensor_path = ownership_root / "ownership.safetensors"
    ownership_receipt = _read(ownership_receipt_path)
    if (
        set(ownership_receipt) != set(ownership_builder.RECEIPT_KEYS)
        or
        ownership_receipt.get("schema_version") != ownership_builder.RECEIPT_SCHEMA
        or ownership_receipt.get("assignment_result_file_sha256")
        != core.file_sha256(assignment_result_path)
        or ownership_receipt.get("track_receipt_file_sha256")
        != core.file_sha256(track_receipt_path)
        or ownership_receipt.get("tensor_file") != "ownership.safetensors"
        or ownership_receipt.get("tensor_file_sha256")
        != core.file_sha256(ownership_tensor_path)
    ):
        raise PostflightSourceFourRoleV15CR9Error("ownership binding differs")
    authority, _base, _roles = authority_contract.validate_authority(
        root=repo_root, authority_path=authority_path
    )
    thresholds = authority_contract.thresholds_from_authority(authority)
    expected_shapes = {
        "raw_proposal_masks": (4, 81, 1056, 704),
        "final_ownership_masks": (4, 81, 1056, 704),
        "unassigned_occlusion_mask": (81, 1056, 704),
        "contact_human_old_actor": (81, 1056, 704),
        "contact_human_new_actor": (81, 1056, 704),
        "contact_human_recipient": (81, 1056, 704),
        "v15b_role_masks": (4, 21, core.GRID_HEIGHT, core.GRID_WIDTH),
        "v15b_contact_relation_mask": (21, core.GRID_HEIGHT, core.GRID_WIDTH),
        "v15b_unassigned_mask": (21, core.GRID_HEIGHT, core.GRID_WIDTH),
    }
    try:
        parsed = ownership_builder.observer_evidence.strict_safetensors(
            ownership_tensor_path,
            expected_order=ownership_builder.TENSOR_ORDER,
            expected_contract={
                name: ("U8", expected_shapes[name])
                for name in ownership_builder.TENSOR_ORDER
            },
            expected_file_sha256=ownership_receipt["tensor_file_sha256"],
            expected_array_sha256=ownership_receipt[
                "tensor_safetensors_array_sha256"
            ],
            expected_metadata={
                "schema_version": ownership_builder.TENSOR_SCHEMA,
                "role_names": ",".join(core.ROLE_NAMES),
                "source_video_sha256": authority["source"]["video_sha256"],
            },
        )
    except ownership_builder.observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
        raise PostflightSourceFourRoleV15CR9Error(str(error)) from error
    arrays = parsed["arrays"]
    for name, shape in expected_shapes.items():
        value = arrays[name]
        if (
            value.shape != shape
            or value.dtype != np.uint8
            or not bool(np.isin(value, [0, 1]).all())
            or ownership_receipt.get("tensor_array_sha256", {}).get(name)
            != core.array_sha256(value)
        ):
            raise PostflightSourceFourRoleV15CR9Error(
                "ownership tensor contract differs"
            )
    if bool((arrays["final_ownership_masks"].sum(axis=0) > 1).any()):
        raise PostflightSourceFourRoleV15CR9Error("full-resolution ownership overlaps")
    if bool((arrays["v15b_role_masks"].sum(axis=0) > 1).any()):
        raise PostflightSourceFourRoleV15CR9Error("v15b ownership overlaps")

    logits, loaded = ownership_builder.load_selected_replayed_raw_logits(
        track_receipt_path=track_receipt_path,
        assignment_result=published_assignment,
    )
    partition = core.partition_source_role_ownership_v15c_r9(
        proposal_masks=loaded["proposal_masks"],
        replayed_raw_signed_valued_logits=logits,
        thresholds=thresholds,
    )
    adapted = core.adapt_qualified_ownership_to_v15b_v15c_r9(
        final_ownership_masks=partition["final_ownership_masks"],
        human_vessel_contact_masks=partition["human_vessel_contact_masks"],
    )
    expected_arrays = {
        "raw_proposal_masks": partition["raw_proposal_masks"],
        "final_ownership_masks": partition["final_ownership_masks"],
        "unassigned_occlusion_mask": partition["unassigned_occlusion_mask"],
        "contact_human_old_actor": partition["human_vessel_contact_masks"]["old_actor"],
        "contact_human_new_actor": partition["human_vessel_contact_masks"]["new_actor"],
        "contact_human_recipient": partition["human_vessel_contact_masks"]["recipient"],
        "v15b_role_masks": adapted["role_masks"],
        "v15b_contact_relation_mask": adapted["contact_relation_mask"],
        "v15b_unassigned_mask": adapted["unassigned_mask"],
    }
    if any(
        not np.array_equal(arrays[name] != 0, expected_arrays[name])
        for name in ownership_builder.TENSOR_NAMES
    ):
        raise PostflightSourceFourRoleV15CR9Error("ownership array replay differs")
    if (
        ownership_receipt.get("selected_logit_replay") != loaded["receipt"]
        or ownership_receipt.get("partition_receipt") != partition["receipt"]
        or ownership_receipt.get("v15b_adapter_receipt") != adapted["receipt"]
        or ownership_receipt.get("raw_proposal_overlap_preserved_as_evidence")
        is not True
        or ownership_receipt.get("final_four_role_ownership_pairwise_exclusive")
        is not True
        or ownership_receipt.get("contact_relation_mask_independent") is not True
        or ownership_receipt.get("raw_overlapping_proposals_passed_to_v15b")
        is not False
        or ownership_receipt.get("morphological_repair_applied") is not False
        or ownership_receipt.get("remote_worker_execution_verified") is not False
        or ownership_receipt.get("observer_execution_authorized") is not False
        or ownership_receipt.get("scientific_claim_authorized") is not False
        or ownership_receipt.get("route_authorized") is not False
        or ownership_receipt.get("decode_authorized") is not False
        or ownership_receipt.get("training_authorized") is not False
    ):
        raise PostflightSourceFourRoleV15CR9Error(
            "ownership receipt replay/claim boundary differs"
        )
    selected_rows = {
        role: next(
            row
            for row in published_assignment["evidence"][role]
            if row["proposal_id"] == published_assignment["assignments"][role]
        )
        for role in core.ROLE_NAMES
    }
    gates = {
        "authority_spec_token_and_source_bytes": (
            core.file_sha256(authority_path)
            == authority_contract.EXPECTED_AUTHORITY_RAW_SHA256
        ),
        "r8_local_tensor_replay_remote_unverified": (
            local_evidence.get("status")
            == "LOCAL_SCHEMA_REPLAY_PASS_REMOTE_OBSERVER_UNVERIFIED"
            and local_evidence.get("remote_worker_execution_verified") is False
        ),
        "r6_human_and_three_vessel_channels": (
            published_assignment.get("r6_role_channel_index")
            == core.R6_ROLE_INDEX
        ),
        "exact_role_indexed_64_null_global_four_role_max_t": (
            published_assignment.get("multiple_comparison_control", {}).get(
                "method"
            )
            == "aligned_64_null_global_max_T_over_four_roles_and_all_geometry_valid_proposals"
            and published_assignment.get("multiple_comparison_control", {}).get(
                "four_role_joint_null_available"
            )
            is True
            and published_assignment.get("multiple_comparison_control", {}).get(
                "global_four_role_fwer_certified"
            )
            is True
            and published_assignment.get("multiple_comparison_control", {}).get(
                "common_null_broadcast_used_for_certification"
            )
            is False
            and all(
                row["gates"]["global_four_role_max_t_fwer"] is True
                for row in selected_rows.values()
            )
        ),
        "vessel_three_role_bonferroni_extra_gate": all(
            selected_rows[role]["gates"][
                "vessel_three_role_bonferroni_extra_gate"
            ]
            is True
            for role in core.VESSEL_ROLE_NAMES
        ),
        "human_81_frame_area_continuity_topology_gates": all(
            value is True
            for key, value in selected_rows["human_agent"]["gates"].items()
            if key.startswith("human_")
        ),
        "same_role_duplicate_nesting_fail_closed": (
            all(
                not published_assignment["same_role_duplicate_nesting_families"][
                    role
                ]
                for role in core.ROLE_NAMES
            )
            and all(
                row["gates"]["no_same_role_duplicate_or_nesting_candidate"]
                is True
                for row in selected_rows.values()
            )
        ),
        "cross_role_conflict_and_runner_up_fail_closed": (
            not published_assignment["cross_role_conflicts"]
            and all(
                published_assignment["competition"][role]["status"]
                in {
                    "unique_eligible_proposal",
                    "unique_top_winner_dominated_every_eligible_proposal",
                }
                for role in core.ROLE_NAMES
            )
        ),
        "limited_human_vessel_overlap_is_relation_evidence": all(
            row["limited_overlap_gate"] is True
            for row in published_assignment[
                "human_vessel_contact_or_occlusion_evidence"
            ]
        ),
        "three_vessel_roles_strictly_mutually_exclusive": len(
            {
                published_assignment["assignments"][role]
                for role in core.VESSEL_ROLE_NAMES
            }
        )
        == len(core.VESSEL_ROLE_NAMES),
        "both_selected_raw_logit_runs_reopened_and_equal": loaded["receipt"][
            "both_complete_selected_logit_runs_array_equal"
        ]
        is True,
        "published_masks_are_exact_logit_positive_sets": loaded["receipt"][
            "published_masks_equal_replayed_raw_logit_positive_sets"
        ]
        is True,
        "ownership_strict_argmax_or_unassigned": partition["receipt"][
            "arbitration"
        ]
        == "strict_argmax_of_replayed_raw_signed_valued_sam2_logits_else_unassigned",
        "ownership_connectivity_hole_area_continuity_replayed": all(
            all(value is True for value in role_gates.values())
            for role_gates in partition["receipt"]["role_gates"].values()
        ),
        "four_role_final_ownership_pairwise_exclusive": (
            partition["receipt"]["pairwise_exclusive_final_ownership"] is True
            and not bool(
                (arrays["final_ownership_masks"].sum(axis=0) > 1).any()
            )
        ),
        "contact_relation_mask_independent": (
            partition["receipt"]["contact_relation_is_not_ownership"] is True
            and ownership_receipt["contact_relation_mask_independent"] is True
        ),
        "v15b_adapter_never_receives_raw_overlap": (
            adapted["receipt"]["raw_overlapping_proposals_passed_to_v15b"]
            is False
            and not bool((arrays["v15b_role_masks"].sum(axis=0) > 1).any())
        ),
        "no_morphological_repair": (
            partition["receipt"]["morphological_repair_applied"] is False
            and adapted["receipt"]["morphological_repair_applied"] is False
            and ownership_receipt["morphological_repair_applied"] is False
        ),
        "local_observer_claim_boundary": all(
            ownership_receipt[key] is False
            for key in (
                "remote_worker_execution_verified",
                "observer_execution_authorized",
                "localization_semantically_certified",
                "scientific_claim_authorized",
                "route_authorized",
                "decode_authorized",
                "training_authorized",
            )
        ),
    }
    if set(gates) != set(GATE_KEYS) or any(type(value) is not bool or value is not True for value in gates.values()):
        raise PostflightSourceFourRoleV15CR9Error("postflight gate registry differs")
    mechanical = bool(
        published_assignment.get("role_assignment_mechanical_candidate_qualified")
        and ownership_receipt.get("mechanical_candidate_qualified")
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "LOCAL_SOURCE_OBSERVER_MECHANICAL_POSTFLIGHT_PASS_REMOTE_UNVERIFIED",
        "gates": gates,
        "file_sha256": {
            "authority": core.file_sha256(authority_path),
            "r6_receipt": core.file_sha256(r6_receipt_path),
            "r6_tensors": core.file_sha256(r6_tensors_path),
            "track_receipt": core.file_sha256(track_receipt_path),
            "track_tensors": core.file_sha256(track_tensors_path),
            "assignment_result": core.file_sha256(assignment_result_path),
            "ownership_receipt": core.file_sha256(ownership_receipt_path),
            "ownership_tensors": core.file_sha256(ownership_tensor_path),
        },
        "local_evidence_replay_sha256": local_evidence["receipt_sha256"],
        "assignment_result_internal_sha256": published_assignment["receipt_sha256"],
        "ownership_result_internal_sha256": ownership_receipt["receipt_sha256"],
        "mechanical_candidate_qualified": mechanical,
        "human_audit_action": "reject_only",
        "human_audit_may_authorize_route": False,
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "action_success_certified": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    if set(receipt) != set(POSTFLIGHT_KEYS):
        raise PostflightSourceFourRoleV15CR9Error(
            "postflight receipt registry differs"
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--r6-receipt", required=True, type=Path)
    parser.add_argument("--r6-tensors", required=True, type=Path)
    parser.add_argument("--track-receipt", required=True, type=Path)
    parser.add_argument("--track-tensors", required=True, type=Path)
    parser.add_argument("--assignment-result", required=True, type=Path)
    parser.add_argument("--ownership-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_json.absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise PostflightSourceFourRoleV15CR9Error("postflight output is not fresh")
    receipt = replay_postflight(
        repo_root=args.repo_root,
        authority_path=args.authority,
        r6_receipt_path=args.r6_receipt,
        r6_tensors_path=args.r6_tensors,
        track_receipt_path=args.track_receipt,
        track_tensors_path=args.track_tensors,
        assignment_result_path=args.assignment_result,
        ownership_dir=args.ownership_dir,
    )
    output.write_bytes(core.canonical_bytes(receipt))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize mutually-exclusive r9 ownership from replayed SAM2 logits.

This future-ABI local observer stage is reachable only after an exact
role-indexed 64-null tensor has certified joint four-role FWER.  The currently
sealed r6 artifact has one common null axis and the r9 runner therefore cannot
enter this stage.  If that prerequisite is eventually supplied, this stage
reopens both r8 SAM2 propagation runs, compares the
selected raw floating-point logits byte-for-byte, verifies that every published
proposal mask is exactly ``logit > 0``, and then calls the r9 fail-closed
ownership partition.  It performs no model call and has no GPU path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from . import run_source_four_role_authority_v15c_r9 as runner
    from . import sam2_observer_evidence_v15c_r8 as observer_evidence
    from . import source_role_authority_v15c_r9 as core
    from . import validate_source_role_authority_v15c_r9 as authority_contract
except ImportError:  # pragma: no cover
    import run_source_four_role_authority_v15c_r9 as runner
    import sam2_observer_evidence_v15c_r8 as observer_evidence
    import source_role_authority_v15c_r9 as core
    import validate_source_role_authority_v15c_r9 as authority_contract


RECEIPT_SCHEMA = "bernini-source-four-role-ownership-artifact-v15c-r9-local"
TENSOR_SCHEMA = "bernini-source-four-role-ownership-tensors-v15c-r9-local"
TENSOR_NAMES = (
    "raw_proposal_masks",
    "final_ownership_masks",
    "unassigned_occlusion_mask",
    "contact_human_old_actor",
    "contact_human_new_actor",
    "contact_human_recipient",
    "v15b_role_masks",
    "v15b_contact_relation_mask",
    "v15b_unassigned_mask",
)
TENSOR_ORDER = tuple(sorted(TENSOR_NAMES))
RECEIPT_KEYS = (
    "schema_version",
    "status",
    "authority_raw_sha256",
    "assignment_result_file_sha256",
    "assignment_result_internal_sha256",
    "track_receipt_file_sha256",
    "selected_logit_replay",
    "partition_receipt",
    "v15b_adapter_receipt",
    "tensor_file",
    "tensor_file_sha256",
    "tensor_array_sha256",
    "tensor_safetensors_array_sha256",
    "raw_proposal_overlap_preserved_as_evidence",
    "final_four_role_ownership_pairwise_exclusive",
    "contact_relation_mask_independent",
    "raw_overlapping_proposals_passed_to_v15b",
    "morphological_repair_applied",
    "mechanical_candidate_qualified",
    "local_schema_replay_only",
    "remote_worker_execution_verified",
    "observer_execution_authorized",
    "localization_semantically_certified",
    "scientific_claim_authorized",
    "route_authorized",
    "decode_authorized",
    "training_authorized",
    "optimizer_updates",
    "renderer_forward_calls",
    "receipt_sha256",
)


class MaterializeSourceRoleOwnershipV15CR9Error(RuntimeError):
    """The selected proposal/logit replay or ownership artifact differs."""


def _load_result(path: Path) -> Mapping[str, Any]:
    value = runner.read_json(path.resolve(strict=True))
    try:
        runner.verify_self_hash(value)
    except runner.RunSourceFourRoleAuthorityV15CR9Error as error:
        raise MaterializeSourceRoleOwnershipV15CR9Error(str(error)) from error
    return value


def _strict_logit_artifact(
    *,
    root: Path,
    descriptor: Mapping[str, Any],
    run_ordinal: int,
    batch_index: int,
    batch_start: int,
    batch_stop: int,
    frame_index: int,
) -> Any:
    relative = (
        f"observer_evidence/run_{run_ordinal}/batch_{batch_index:03d}/"
        f"propagation_frame_{frame_index:05d}.safetensors"
    )
    if type(descriptor) is not dict or descriptor.get("relative_path") != relative:
        raise MaterializeSourceRoleOwnershipV15CR9Error("logit descriptor path differs")
    path = root / relative
    try:
        parsed = observer_evidence.strict_safetensors(
            path,
            expected_order=("logits",),
            expected_contract={
                "logits": (
                    "F32",
                    (batch_stop - batch_start, 1, 1056, 704),
                )
            },
            expected_file_sha256=descriptor.get("file_sha256"),
            expected_array_sha256=descriptor.get("tensor_array_sha256"),
            expected_metadata={
                "schema_version": observer_evidence.LOGIT_FILE_SCHEMA,
                "kind": "propagation",
                "run_ordinal": str(run_ordinal),
                "batch_index": str(batch_index),
                "frame_index": str(frame_index),
                "out_ids": ",".join(
                    str(item) for item in range(batch_start, batch_stop)
                ),
            },
        )
    except observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
        raise MaterializeSourceRoleOwnershipV15CR9Error(str(error)) from error
    return parsed["arrays"]["logits"]


def load_selected_replayed_raw_logits(
    *,
    track_receipt_path: Path,
    assignment_result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Reopen both complete tensor runs for the four selected proposals."""

    import cv2
    import numpy as np

    track_path = track_receipt_path.resolve(strict=True)
    track_root = track_path.parent
    track = runner.read_json(track_path)
    evidence_path = track_root / "observer_evidence/local_evidence.json"
    evidence = runner.read_json(evidence_path)
    try:
        runner.verify_self_hash(track)
        runner.verify_self_hash(evidence)
    except runner.RunSourceFourRoleAuthorityV15CR9Error as error:
        raise MaterializeSourceRoleOwnershipV15CR9Error(str(error)) from error
    provenance = assignment_result.get("provenance")
    proposals = track.get("proposals")
    assignments = assignment_result.get("assignments")
    if (
        type(proposals) is not list
        or type(assignments) is not dict
        or set(assignments) != set(core.ROLE_NAMES)
        or any(type(assignments[role]) is not str for role in core.ROLE_NAMES)
        or assignment_result.get("role_assignment_mechanical_candidate_qualified")
        is not True
        or assignment_result.get("mechanical_candidate_qualified") is not False
        or type(provenance) is not dict
        or provenance.get("track_receipt_file_sha256")
        != core.file_sha256(track_path)
        or provenance.get("track_internal_receipt_sha256")
        != track.get("receipt_sha256")
        or provenance.get("observer_evidence_file_sha256")
        != core.file_sha256(evidence_path)
        or provenance.get("observer_evidence_internal_sha256")
        != evidence.get("receipt_sha256")
        or type(evidence.get("runs")) is not list
        or len(evidence["runs"]) != 2
    ):
        raise MaterializeSourceRoleOwnershipV15CR9Error(
            "complete assignment/evidence registry is unavailable"
        )
    proposal_ids = [row.get("proposal_id") for row in proposals]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise MaterializeSourceRoleOwnershipV15CR9Error("proposal IDs differ")
    indices = {}
    for role in core.ROLE_NAMES:
        try:
            indices[role] = proposal_ids.index(assignments[role])
        except ValueError as error:
            raise MaterializeSourceRoleOwnershipV15CR9Error(
                "assigned proposal is not published"
            ) from error
    first = {
        role: np.empty((81, 1056, 704), dtype=np.float32)
        for role in core.ROLE_NAMES
    }
    artifact_hashes: dict[str, list[str]] = {"run_1": [], "run_2": []}
    for run_index, run in enumerate(evidence["runs"]):
        run_ordinal = run_index + 1
        batches = run.get("tracking_batches")
        if type(batches) is not list or not batches:
            raise MaterializeSourceRoleOwnershipV15CR9Error("tracking batches differ")
        for frame_index in range(81):
            loaded_batches = {}
            for role in core.ROLE_NAMES:
                object_index = indices[role]
                matching = [
                    row
                    for row in batches
                    if type(row) is dict
                    and type(row.get("batch_start")) is int
                    and type(row.get("batch_stop")) is int
                    and row["batch_start"] <= object_index < row["batch_stop"]
                ]
                if len(matching) != 1:
                    raise MaterializeSourceRoleOwnershipV15CR9Error(
                        "selected object batch differs"
                    )
                batch = matching[0]
                batch_index = batch.get("batch_index")
                if type(batch_index) is not int or batch_index < 0:
                    raise MaterializeSourceRoleOwnershipV15CR9Error(
                        "batch index differs"
                    )
                if batch_index not in loaded_batches:
                    artifacts = batch.get("propagation_artifacts")
                    if type(artifacts) is not list or len(artifacts) != 81:
                        raise MaterializeSourceRoleOwnershipV15CR9Error(
                            "propagation artifact registry differs"
                        )
                    descriptor = artifacts[frame_index]
                    loaded_batches[batch_index] = _strict_logit_artifact(
                        root=track_root,
                        descriptor=descriptor,
                        run_ordinal=run_ordinal,
                        batch_index=batch_index,
                        batch_start=batch["batch_start"],
                        batch_stop=batch["batch_stop"],
                        frame_index=frame_index,
                    )
                    artifact_hashes[f"run_{run_ordinal}"].append(
                        descriptor["file_sha256"]
                    )
                values = np.ascontiguousarray(
                    loaded_batches[batch_index][
                        object_index - batch["batch_start"], 0
                    ],
                    dtype=np.float32,
                )
                if run_index == 0:
                    first[role][frame_index] = values
                elif not np.array_equal(values, first[role][frame_index]):
                    raise MaterializeSourceRoleOwnershipV15CR9Error(
                        "selected replayed raw logits differ across runs"
                    )
    proposal_masks = {}
    mask_hashes = {}
    for role in core.ROLE_NAMES:
        frames = np.empty((81, 1056, 704), dtype=np.bool_)
        digests = []
        for frame_index in range(81):
            path = (
                track_root
                / "masks"
                / assignments[role]
                / f"{frame_index:05d}.png"
            )
            mask_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if (
                not isinstance(mask_u8, np.ndarray)
                or mask_u8.shape != (1056, 704)
                or not set(np.unique(mask_u8)).issubset({0, 255})
            ):
                raise MaterializeSourceRoleOwnershipV15CR9Error(
                    "selected proposal mask differs"
                )
            mask = mask_u8 > 0
            if not np.array_equal(mask, first[role][frame_index] > 0.0):
                raise MaterializeSourceRoleOwnershipV15CR9Error(
                    "selected mask is not replayed raw logit positive set"
                )
            frames[frame_index] = mask
            digests.append(core.array_sha256(mask.astype(np.uint8)))
        proposal_masks[role] = frames
        mask_hashes[role] = digests
    receipt = {
        "evidence_file_sha256": core.file_sha256(evidence_path),
        "evidence_internal_sha256": evidence.get("receipt_sha256"),
        "selected_proposal_indices": indices,
        "selected_proposal_ids": dict(assignments),
        "run_propagation_artifact_file_sha256": artifact_hashes,
        "selected_mask_array_sha256_by_frame": mask_hashes,
        "both_complete_selected_logit_runs_array_equal": True,
        "published_masks_equal_replayed_raw_logit_positive_sets": True,
        "external_signature_or_tee_verified": False,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    return first, {"proposal_masks": proposal_masks, "receipt": receipt}


def materialize_ownership(
    *,
    repo_root: Path,
    authority_path: Path,
    r6_receipt_path: Path,
    r6_tensors_path: Path,
    track_receipt_path: Path,
    track_tensors_path: Path,
    assignment_result_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    import numpy as np
    from safetensors.numpy import save_file

    output = output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise MaterializeSourceRoleOwnershipV15CR9Error("output is not fresh")
    replayed_result, _local = runner.replay_assignment(
        repo_root=repo_root,
        authority_path=authority_path,
        r6_receipt_path=r6_receipt_path,
        r6_tensors_path=r6_tensors_path,
        track_receipt_path=track_receipt_path,
        track_tensors_path=track_tensors_path,
    )
    published_result = _load_result(assignment_result_path)
    if replayed_result != published_result:
        raise MaterializeSourceRoleOwnershipV15CR9Error(
            "assignment result does not replay"
        )
    authority, _base, _roles = authority_contract.validate_authority(
        root=repo_root, authority_path=authority_path
    )
    thresholds = authority_contract.thresholds_from_authority(authority)
    logits, loaded = load_selected_replayed_raw_logits(
        track_receipt_path=track_receipt_path,
        assignment_result=published_result,
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
    output.mkdir(mode=0o700, parents=True)
    tensor_path = output / "ownership.safetensors"
    arrays = {
        "raw_proposal_masks": partition["raw_proposal_masks"].astype(np.uint8),
        "final_ownership_masks": partition["final_ownership_masks"].astype(np.uint8),
        "unassigned_occlusion_mask": partition["unassigned_occlusion_mask"].astype(np.uint8),
        "contact_human_old_actor": partition["human_vessel_contact_masks"]["old_actor"].astype(np.uint8),
        "contact_human_new_actor": partition["human_vessel_contact_masks"]["new_actor"].astype(np.uint8),
        "contact_human_recipient": partition["human_vessel_contact_masks"]["recipient"].astype(np.uint8),
        "v15b_role_masks": adapted["role_masks"].astype(np.uint8),
        "v15b_contact_relation_mask": adapted["contact_relation_mask"].astype(np.uint8),
        "v15b_unassigned_mask": adapted["unassigned_mask"].astype(np.uint8),
    }
    ordered_arrays = {name: arrays[name] for name in TENSOR_ORDER}
    save_file(
        ordered_arrays,
        str(tensor_path),
        metadata={
            "schema_version": TENSOR_SCHEMA,
            "role_names": ",".join(core.ROLE_NAMES),
            "source_video_sha256": authority["source"]["video_sha256"],
        },
    )
    standard_hashes = {
        name: observer_evidence.array_sha256(arrays[name]) for name in TENSOR_ORDER
    }
    try:
        parsed = observer_evidence.strict_safetensors(
            tensor_path,
            expected_order=TENSOR_ORDER,
            expected_contract={
                name: ("U8", tuple(arrays[name].shape)) for name in TENSOR_ORDER
            },
            expected_file_sha256=core.file_sha256(tensor_path),
            expected_array_sha256=standard_hashes,
            expected_metadata={
                "schema_version": TENSOR_SCHEMA,
                "role_names": ",".join(core.ROLE_NAMES),
                "source_video_sha256": authority["source"]["video_sha256"],
            },
        )
    except observer_evidence.SAM2ObserverEvidenceV15CR8Error as error:
        raise MaterializeSourceRoleOwnershipV15CR9Error(str(error)) from error
    reopened = parsed["arrays"]
    for name in TENSOR_NAMES:
        if not np.array_equal(reopened[name], arrays[name]):
            raise MaterializeSourceRoleOwnershipV15CR9Error(
                "ownership tensor reopen differs"
            )
    complete = bool(
        partition["receipt"]["mechanical_candidate_qualified"]
        and adapted["receipt"]["mechanical_candidate_qualified"]
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "LOCAL_OWNERSHIP_ARTIFACT_PASS_REMOTE_UNVERIFIED" if complete else "NO_GO_LOCAL_OWNERSHIP_ARTIFACT",
        "authority_raw_sha256": core.file_sha256(authority_path),
        "assignment_result_file_sha256": core.file_sha256(assignment_result_path),
        "assignment_result_internal_sha256": published_result["receipt_sha256"],
        "track_receipt_file_sha256": core.file_sha256(track_receipt_path),
        "selected_logit_replay": loaded["receipt"],
        "partition_receipt": partition["receipt"],
        "v15b_adapter_receipt": adapted["receipt"],
        "tensor_file": "ownership.safetensors",
        "tensor_file_sha256": core.file_sha256(tensor_path),
        "tensor_array_sha256": {
            name: core.array_sha256(arrays[name]) for name in TENSOR_NAMES
        },
        "tensor_safetensors_array_sha256": standard_hashes,
        "raw_proposal_overlap_preserved_as_evidence": True,
        "final_four_role_ownership_pairwise_exclusive": True,
        "contact_relation_mask_independent": True,
        "raw_overlapping_proposals_passed_to_v15b": False,
        "morphological_repair_applied": False,
        "mechanical_candidate_qualified": complete,
        "local_schema_replay_only": True,
        "remote_worker_execution_verified": False,
        "observer_execution_authorized": False,
        "localization_semantically_certified": False,
        "scientific_claim_authorized": False,
        "route_authorized": False,
        "decode_authorized": False,
        "training_authorized": False,
        "optimizer_updates": 0,
        "renderer_forward_calls": 0,
    }
    receipt["receipt_sha256"] = core.object_sha256(receipt)
    if set(receipt) != set(RECEIPT_KEYS):
        raise MaterializeSourceRoleOwnershipV15CR9Error(
            "ownership receipt registry differs"
        )
    receipt_path = output / "ownership_receipt.json"
    receipt_path.write_bytes(core.canonical_bytes(receipt))
    os.chmod(tensor_path, 0o400)
    os.chmod(receipt_path, 0o400)
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
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = materialize_ownership(
        repo_root=args.repo_root,
        authority_path=args.authority,
        r6_receipt_path=args.r6_receipt,
        r6_tensors_path=args.r6_tensors,
        track_receipt_path=args.track_receipt,
        track_tensors_path=args.track_tensors,
        assignment_result_path=args.assignment_result,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

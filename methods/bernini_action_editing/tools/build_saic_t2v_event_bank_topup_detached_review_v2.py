#!/usr/bin/env python3
"""Build a detached zero-authority review packet for the SAIC v2 top-up bank.

This is a narrow adapter over ``build_saic_t2v_event_bank_detached_review_v1``.
It retains the exact81 media diagnostics, immutable packet construction, blind
human-review ordering, and fail-closed authority model from v1 while replacing
the input audit and observer criteria with the v2 top-up contract.  Before the
v1 packet builder sees rows, this adapter pre-registers a domain-separated
opaque alias order and replaces every public source/candidate basename; its
blind renderer omits registered identifiers, sampling values, and branch names.

The input bank remains immutable.  Machine diagnostics never assign semantic
labels, rank candidates, select seeds, create training targets, or authorize an
optimizer/parameter update.  The emitted observer artifacts are blank templates
for two independent humans and cannot establish that review occurred.
"""

from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import re
import secrets
import sys
from typing import Any, Mapping, Sequence


TOOL_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = TOOL_ROOT.parent
for search_root in (TOOL_ROOT, METHOD_ROOT):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import build_saic_t2v_event_bank_detached_review_v1 as base  # noqa: E402
import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import generate_saic_pure_t2v_event_bank_topup_v2 as generation  # noqa: E402
import saic_pure_t2v_event_bank_topup_v2 as topup  # noqa: E402


SCHEMA_VERSION = "bernini-saic-t2v-event-bank-topup-detached-review-manifest-v2"
RECEIPT_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-topup-detached-review-receipt-v2"
)
OBSERVER_TEMPLATE_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-topup-independent-observer-blank-template-v2"
)
OBSERVER_PROTOCOL_SCHEMA_VERSION = (
    "bernini-saic-t2v-event-bank-topup-independent-full81-observer-protocol-v2"
)
PACKET_ID = (
    "t2v-events-topup-r6-formal-v2-r2-retfd-20260812-r1-"
    "detached-review-v2-r1"
)
MASTER_SCHEMA_VERSION = generation.MASTER_SCHEMA_VERSION
MASTER_BASENAME = generation.MASTER_RECEIPT_BASENAME
SOURCE_MANIFEST_BASENAME = "sealed-saic-source-manifest.json"
EVENT_SPEC_BASENAME = "sealed-saic-t2v-event-topup-v2-spec.json"
BASE_V1_SPEC_BASENAME = "sealed-base-saic-t2v-event-v1-spec.json"
ATTEMPT_RECEIPT_BASENAME = generation.ATTEMPT_RECEIPT_BASENAME
BRANCH_ORDER = tuple(topup.BRANCH_ORDER)
_BLIND_ALIAS_DOMAIN = b"saic-topup-review-v2-r1-opaque-alias\0"
_BLIND_ALIAS_NONCE_BYTES = 32

_MASTER_FIELDS = {
    "schema_version",
    "bank_id",
    "top_up_only",
    "root_spec_raw_sha256",
    "base_v1_spec_raw_sha256",
    "base_v1_spec_content_sha256",
    "source_manifest_content_sha256",
    "topology",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "attempt_count",
    "row_count",
    "seed_cell_count",
    "branch_order",
    "merged_branch_order",
    "six_branch_spec_merge_cell_count",
    "same_seed_official_gaussian_proofs",
    "attempts",
    "detached_full81_event_review_complete",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_or_parameter_update_authorized",
    "receipt_digest",
}
_MASTER_ATTEMPT_FIELDS = {
    "candidate_id",
    "row_id",
    "iid",
    "analysis_split",
    "branch",
    "seed",
    "receipt_path",
    "receipt_sha256",
    "receipt_digest",
    "mp4_path",
    "mp4_sha256",
    "event_audit_status",
}
_FALSE_MASTER_FIELDS = (
    "detached_full81_event_review_complete",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_or_parameter_update_authorized",
)
_OFFICIAL_GAUSSIAN_IDENTITY_FIELDS = (
    "raw_value_sha256",
    "content_sha256",
    "shape",
    "dtype",
    "stored_dtype",
    "generator_initial_seed",
)

_ORIGINAL_COPY_VERIFIED = base._copy_verified_create_only
_ACTIVE_BASE_SPEC: tuple[Path, str] | None = None


def _require(condition: bool, message: str) -> None:
    base._require(condition, message)


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    return base._closed(value, fields, label=label)


def _flatten_candidates(
    spec: Mapping[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return base._flatten_candidates(spec)


def _opaque_review_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Randomly pre-register opaque aliases before the base packet builder.

    A fresh cryptographic nonce makes the permutation unpredictable from the
    public source/spec, and the nonce is discarded after registration.  The
    original-to-opaque mapping remains only in the sealed assessor-private
    manifest.  Neither the public media namespace nor the blind HTML receives
    an original candidate id, source row id, branch name, or seed.
    """

    _require(len(rows) == 60, "opaque alias candidate count differs")
    original_ids = [str(row["candidate_id"]) for row in rows]
    _require(len(set(original_ids)) == 60, "opaque alias source ids differ")
    nonce = secrets.token_bytes(_BLIND_ALIAS_NONCE_BYTES)
    ranked_pairs = [
        (
            hashlib.sha256(
                _BLIND_ALIAS_DOMAIN
                + nonce
                + str(row["candidate_id"]).encode("ascii")
            ).digest(),
            dict(row),
        )
        for row in rows
    ]
    _require(
        len({rank for rank, _ in ranked_pairs}) == 60,
        "opaque alias random rank collision",
    )
    ranked = [row for _, row in sorted(ranked_pairs, key=lambda pair: pair[0])]
    source_aliases: dict[str, str] = {}
    opaque: list[dict[str, Any]] = []
    for alias_index, row in enumerate(ranked, start=1):
        original_row_id = str(row["row_id"])
        original_candidate_id = str(row["candidate_id"])
        source_alias = source_aliases.setdefault(
            original_row_id, f"source-{len(source_aliases) + 1:04d}"
        )
        row["assessor_private_candidate_id"] = original_candidate_id
        row["assessor_private_source_row_id"] = original_row_id
        row["registered_candidate_index"] = alias_index
        row["candidate_id"] = f"candidate-{alias_index:04d}"
        row["row_id"] = source_alias
        opaque.append(row)
    _require(len(source_aliases) == 8, "opaque source alias count differs")
    return opaque


def _validate_master_header(
    master: Mapping[str, Any],
    *,
    master_digest: str,
    spec: Mapping[str, Any],
    spec_raw_sha256: str,
    source_manifest_summary: Mapping[str, Any],
    merged_cell_count: int,
) -> list[Mapping[str, Any]]:
    row = _closed(dict(master), _MASTER_FIELDS, label="v2 top-up master receipt")
    _require(row["receipt_digest"] == master_digest, "v2 master seal differs")
    _require(row["schema_version"] == MASTER_SCHEMA_VERSION, "v2 master schema differs")
    _require(row["bank_id"] == topup.BANK_ID, "v2 master bank id differs")
    _require(row["top_up_only"] is True, "v2 master is not top-up-only")
    _require(row["root_spec_raw_sha256"] == spec_raw_sha256, "v2 master spec binding differs")
    _require(
        row["base_v1_spec_raw_sha256"] == topup.BASE_V1_SPEC_RAW_SHA256
        and row["base_v1_spec_content_sha256"]
        == topup.BASE_V1_SPEC_CONTENT_SHA256,
        "v2 master base-v1 binding differs",
    )
    _require(
        row["source_manifest_content_sha256"]
        == source_manifest_summary["manifest_content_sha256"]
        == topup.SOURCE_MANIFEST_CONTENT_SHA256,
        "v2 master source-manifest binding differs",
    )
    for field in (
        "sampling_contract",
        "semantic_input_closure",
        "geometry_proxy_contract",
        "artifact_authority",
    ):
        _require(row[field] == spec[field], f"v2 master {field} differs")
    _require(
        row["topology"] == "two_concurrent_world4_sp4_groups_on_one_8gpu_node",
        "v2 master topology differs",
    )
    _require(row["branch_order"] == list(BRANCH_ORDER), "v2 branch order differs")
    _require(
        row["merged_branch_order"] == list(topup.MERGED_BRANCH_ORDER),
        "v2 merged branch order differs",
    )
    _require(
        row["six_branch_spec_merge_cell_count"] == merged_cell_count == 20,
        "v2 six-branch merge cardinality differs",
    )
    _require(row["attempt_count"] == 60, "v2 master attempt count differs")
    _require(row["row_count"] == 8, "v2 master row count differs")
    _require(row["seed_cell_count"] == 20, "v2 master seed-cell count differs")
    for field in _FALSE_MASTER_FIELDS:
        _require(row[field] is False, f"v2 master unexpectedly authorizes {field}")
    attempts = row["attempts"]
    _require(type(attempts) is list and len(attempts) == 60, "v2 master attempts differ")
    for index, attempt in enumerate(attempts):
        _closed(attempt, _MASTER_ATTEMPT_FIELDS, label=f"v2 master attempt {index}")
    return attempts


def _validate_attempt(
    path: Path,
    *,
    master_attempt: Mapping[str, Any],
    candidate: Mapping[str, Any],
    group: Mapping[str, Any],
    root_spec_sha256: str,
    real_source_paths: set[str],
    real_source_hashes: set[str],
) -> dict[str, Any]:
    expected_file_sha = base._sha(
        master_attempt["receipt_sha256"], label="v2 attempt receipt file hash"
    )
    _require(base.file_sha256(path) == expected_file_sha, "v2 attempt receipt hash differs")
    try:
        receipt = generation._load_attempt_receipt(
            path,
            candidate=candidate,
            group=group,
            root_spec_sha256=root_spec_sha256,
            real_source_paths=real_source_paths,
            real_source_hashes=real_source_hashes,
        )
    except generation.SAICPureT2VTopupGenerationError as error:
        raise base.DetachedReviewError("v2 attempt receipt failed full re-audit") from error
    _require(
        receipt["receipt_digest"]
        == base._sha(master_attempt["receipt_digest"], label="v2 attempt digest"),
        "v2 attempt receipt digest differs",
    )
    return receipt


def _official_gaussian_proofs(
    cells: Mapping[tuple[str, int], Sequence[tuple[str, Mapping[str, Any]]]]
) -> list[dict[str, Any]]:
    proofs: list[dict[str, Any]] = []
    _require(len(cells) == 20, "v2 official-Gaussian cell count differs")
    for (iid, seed), rows in cells.items():
        _require(
            [branch for branch, _ in rows] == list(BRANCH_ORDER),
            "v2 official-Gaussian branch order differs",
        )
        identities = {
            topup.object_sha256(
                {field: gaussian.get(field) for field in _OFFICIAL_GAUSSIAN_IDENTITY_FIELDS}
            )
            for _, gaussian in rows
        }
        _require(len(identities) == 1, "v2 same-cell official Gaussian differs")
        proofs.append(
            {
                "iid": iid,
                "seed": seed,
                "branch_order": list(BRANCH_ORDER),
                "official_gaussian_tensor_values_byte_equal": True,
                "official_gaussian_identity_digest": next(iter(identities)),
            }
        )
    return proofs


def _load_and_validate_inputs(input_root: Path) -> dict[str, Any]:
    global _ACTIVE_BASE_SPEC

    root = base._plain_dir(input_root, label="v2 top-up event-bank input root")
    master_path = base._plain_file(root / MASTER_BASENAME, label="v2 top-up master receipt")
    source_manifest_path = base._plain_file(
        root / SOURCE_MANIFEST_BASENAME, label="sealed source manifest"
    )
    event_spec_path = base._plain_file(
        root / EVENT_SPEC_BASENAME, label="sealed v2 top-up event spec"
    )
    base_v1_spec_path = base._plain_file(
        root / BASE_V1_SPEC_BASENAME, label="sealed base-v1 event spec"
    )

    try:
        source_manifest = source_set.load_manifest(source_manifest_path)
        source_manifest_summary = source_set.validate_manifest(source_manifest)
        master, master_digest = base._load_sealed(master_path, label="v2 top-up master receipt")
        spec, spec_raw_sha256 = topup.load_sealed_spec(
            event_spec_path,
            expected_raw_sha256=base._sha(
                master.get("root_spec_raw_sha256"), label="v2 master root spec hash"
            ),
            source_manifest_path=source_manifest_path,
            base_v1_spec_path=base_v1_spec_path,
        )
        base_v1_spec = topup.load_base_v1_spec(
            base_v1_spec_path, source_manifest_path=source_manifest_path
        )
        merged = topup.merge_six_branch_cells(base_v1_spec, spec)
    except (
        source_set.SAICReversibleSourceSetError,
        topup.SAICPureT2VEventBankTopupError,
    ) as error:
        raise base.DetachedReviewError("v2 root/source/spec closure failed") from error

    attempts = _validate_master_header(
        master,
        master_digest=master_digest,
        spec=spec,
        spec_raw_sha256=spec_raw_sha256,
        source_manifest_summary=source_manifest_summary,
        merged_cell_count=len(merged),
    )
    flattened = _flatten_candidates(spec)
    _require(len(flattened) == 60, "v2 spec candidate count differs")
    _require(
        [candidate["candidate_id"] for _, candidate in flattened]
        == [attempt["candidate_id"] for attempt in attempts],
        "v2 master reordered spec candidates",
    )

    source_rows = source_manifest.get("rows")
    _require(type(source_rows) is list and len(source_rows) == 8, "source rows differ")
    sources_by_iid = {str(row["iid"]): row for row in source_rows}
    _require(len(sources_by_iid) == 8, "duplicate source iid")
    real_source_paths = {row["source_video"] for row in source_rows}
    real_source_hashes = {row["source_video_sha256"] for row in source_rows}

    candidate_rows: list[dict[str, Any]] = []
    gaussian_cells: dict[
        tuple[str, int], list[tuple[str, Mapping[str, Any]]]
    ] = {}
    for ordinal, ((group, candidate), master_attempt) in enumerate(
        zip(flattened, attempts), start=1
    ):
        candidate_id = base._safe_id(candidate["candidate_id"], label="v2 candidate id")
        for field in ("row_id", "iid", "analysis_split", "branch", "seed"):
            _require(
                master_attempt[field] == candidate[field],
                f"v2 master candidate {field} differs",
            )
        _require(
            master_attempt["event_audit_status"]
            == "pending_detached_full81_review",
            "v2 master event status differs",
        )
        source = sources_by_iid.get(candidate["iid"])
        _require(source is not None, "v2 candidate source row absent")
        _require(source["row_id"] == candidate["row_id"], "v2 source row binding differs")
        _require(
            source["source_video_sha256"]
            == candidate["source_media_sha256_for_nonuse_audit"],
            "v2 real-source nonuse hash binding differs",
        )
        source_video = base._plain_file(source["source_video"], label="source video")
        source_sha = base._sha(source["source_video_sha256"], label="source video hash")
        _require(base.file_sha256(source_video) == source_sha, "source video hash differs")

        expected_attempt_path = (
            root / "attempts" / candidate_id / ATTEMPT_RECEIPT_BASENAME
        ).resolve()
        attempt_path = base._plain_file(
            master_attempt["receipt_path"], label="v2 attempt receipt"
        )
        _require(attempt_path == expected_attempt_path, "v2 attempt escaped canonical root")
        attempt_receipt = _validate_attempt(
            attempt_path,
            master_attempt=master_attempt,
            candidate=candidate,
            group=group,
            root_spec_sha256=spec_raw_sha256,
            real_source_paths=real_source_paths,
            real_source_hashes=real_source_hashes,
        )
        official_gaussian = attempt_receipt["artifacts"]["official_initial_gaussian"]
        gaussian_cells.setdefault((candidate["iid"], candidate["seed"]), []).append(
            (candidate["branch"], official_gaussian)
        )

        expected_mp4_path = (root / "attempts" / candidate_id / "t2v.mp4").resolve()
        mp4_path = base._plain_file(master_attempt["mp4_path"], label="v2 candidate MP4")
        mp4_sha = base._sha(master_attempt["mp4_sha256"], label="v2 candidate MP4 hash")
        _require(mp4_path == expected_mp4_path, "v2 candidate MP4 escaped canonical root")
        _require(base.file_sha256(mp4_path) == mp4_sha, "v2 candidate MP4 hash differs")
        _require(
            attempt_receipt["artifacts"]["mp4"].get("path") == str(mp4_path)
            and attempt_receipt["artifacts"]["mp4"].get("sha256") == mp4_sha,
            "v2 attempt/master MP4 binding differs",
        )
        candidate_rows.append(
            {
                "registered_candidate_index": ordinal,
                "candidate_id": candidate_id,
                "row_id": candidate["row_id"],
                "iid": candidate["iid"],
                "analysis_split": candidate["analysis_split"],
                "actor_family": candidate["actor_family"],
                "action_family_id": candidate["action_family_id"],
                "branch": candidate["branch"],
                "seed": candidate["seed"],
                "initial_state_type": candidate["initial_state_type"],
                "terminal_state_type": candidate["terminal_state_type"],
                "branch_start_state_caption": candidate["branch_start_state_caption"],
                "branch_instruction": candidate["branch_instruction"],
                "full_t2v_caption": candidate["full_t2v_caption"],
                "source_input_path": str(source_video),
                "source_sha256": source_sha,
                "candidate_input_path": str(mp4_path),
                "candidate_sha256": mp4_sha,
                "attempt_receipt_input_path": str(attempt_path),
                "attempt_receipt_sha256": master_attempt["receipt_sha256"],
                "attempt_receipt_digest": master_attempt["receipt_digest"],
                "semantic_status": base.SEMANTIC_STATUS,
                "event_verified": False,
                "identity_preservation_verified": False,
            }
        )

    _require(
        master["same_seed_official_gaussian_proofs"]
        == _official_gaussian_proofs(gaussian_cells),
        "v2 master official-Gaussian proof closure differs",
    )
    base_v1_sha = base.file_sha256(base_v1_spec_path)
    _require(base_v1_sha == topup.BASE_V1_SPEC_RAW_SHA256, "base-v1 spec hash differs")
    _ACTIVE_BASE_SPEC = (base_v1_spec_path, base_v1_sha)
    return {
        "input_root": root,
        "master": master,
        "master_digest": master_digest,
        "master_path": master_path,
        "source_manifest": source_manifest,
        "source_manifest_path": source_manifest_path,
        "source_manifest_summary": source_manifest_summary,
        "event_spec": spec,
        "event_spec_path": event_spec_path,
        "event_spec_raw_sha256": spec_raw_sha256,
        "base_v1_spec_path": base_v1_spec_path,
        "base_v1_spec_raw_sha256": base_v1_sha,
        "candidate_rows": _opaque_review_rows(candidate_rows),
    }


def _copy_verified_with_base_spec(
    source: Path, destination: Path, expected_sha256: str
) -> dict[str, Any]:
    binding = _ORIGINAL_COPY_VERIFIED(source, destination, expected_sha256)
    if destination.name == EVENT_SPEC_BASENAME and destination.parent.name == "evidence":
        _require(_ACTIVE_BASE_SPEC is not None, "base-v1 evidence source is unset")
        base_source, base_sha = _ACTIVE_BASE_SPEC
        base_destination = destination.parent / BASE_V1_SPEC_BASENAME
        base_binding = _ORIGINAL_COPY_VERIFIED(base_source, base_destination, base_sha)
        base_binding["portable_path"] = f"evidence/{BASE_V1_SPEC_BASENAME}"
        binding["base_v1_spec"] = base_binding
    return binding


def _observer_protocol(*, review_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    item_set_digest = base.object_sha256(list(review_items))
    body = {
        "schema_version": OBSERVER_PROTOCOL_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "protocol_timing": "SEALED_AFTER_GENERATION_BEFORE_ANY_DETACHED_HUMAN_LABELS",
        "post_generation_protocol_cannot_claim_pre_generation_preregistration": True,
        "review_item_set_digest": item_set_digest,
        "media_contract": {
            "source_and_candidate_shown_side_by_side": True,
            "candidate_frame_count": base.FRAME_COUNT,
            "source_frame_count": base.FRAME_COUNT,
            "fps": base.FPS,
            "all_frames_must_be_viewed_at_normal_speed_at_least_once": True,
            "frame_scrubbing_and_replay_after_first_view_allowed": True,
            "candidate_id_seed_and_machine_metrics_hidden_in_human_stage": True,
        },
        "stage_order": [
            {
                "stage": 1,
                "artifact": "blind-review.html",
                "rule": "Each observer independently seals all 60 responses before opening any machine diagnostic artifact.",
            },
            {
                "stage": 2,
                "artifact": "index.html and diagnostics/*.json",
                "rule": "Machine diagnostics may be inspected only after both human response artifacts are immutable and cannot revise labels.",
            },
        ],
        "branch_specific_full81_criteria": {
            "incomplete": {
                "start": "The registered q0 start state is visibly correct throughout frames 0-15.",
                "progress": "A directionally correct partial progression begins and is visibly sustained, but remains incomplete.",
                "terminal_absence": "The registered target terminal state is never reached at any frame 0-80.",
                "end_hold": "A visibly partial, nonterminal state remains held through every frame 73-80.",
                "camera_lock": "The registered locked camera and framing remain fixed throughout frames 0-80.",
                "event_pass": "start AND progress AND terminal_absence AND end_hold AND camera_lock must all be true.",
            },
            "camera_only": {
                "state_hold": "The registered q0 actor/action state remains held throughout frames 0-80 and the target event never occurs.",
                "camera_motion": "The registered conspicuous, smooth camera motion occurs without a cut or discontinuity.",
                "appearance_hold": "The subject appearance and geometry remain materially unchanged apart from viewpoint effects.",
                "event_pass": "state_hold AND camera_motion AND appearance_hold must all be true.",
            },
            "appearance_only": {
                "state_hold": "The registered q0 actor/action state remains held throughout frames 0-80 and the target event never occurs.",
                "appearance_change": "The registered appearance-only change is clearly visible without creating the target action event.",
                "camera_lock": "The camera and framing remain locked throughout frames 0-80.",
                "geometry_hold": "Subject geometry and scene layout remain materially fixed while only the registered appearance attributes change.",
                "event_pass": "state_hold AND appearance_change AND camera_lock AND geometry_hold must all be true.",
            },
        },
        "shared_full81_axes": {
            "identity_preserved_full81": "The same source subject remains recognizable with no swap, replacement, or material identity morph at every frame 0-80.",
            "technical_quality_acceptable_full81": "No ghosting, duplication, tearing, disappearance, or corruption obscures the branch judgment at any frame.",
        },
        "observer_contract": {
            "minimum_independent_observers": 2,
            "observer_kind": "independent_human_full81_review",
            "different_people_required": True,
            "communication_or_label_sharing_before_seal_forbidden": True,
            "distinct_observer_identity_and_authority_artifacts_required": True,
            "same_preparer_must_not_act_as_either_observer": True,
            "one_person_filling_both_templates_forbidden": True,
        },
        "aggregation_rule": {
            "majority_vote_allowed": False,
            "tie_break_or_adjudication_inside_v2_allowed": False,
            "missing_response_result": base.SEMANTIC_STATUS,
            "observer_disagreement_result": base.SEMANTIC_STATUS,
            "agreed_positive_result": "AGREED_POSITIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "agreed_negative_result": "AGREED_NEGATIVE_PENDING_SEPARATE_EXTERNAL_SEAL",
            "event_verified_may_be_set_by_this_packet": False,
            "identity_verified_may_be_set_by_this_packet": False,
            "separate_versioned_aggregator_required": True,
        },
        "machine_diagnostic_contract": {
            "human_labels_must_precede_machine_diagnostic_access": True,
            "machine_camera_or_technical_thresholds_calibrated": False,
            "machine_diagnostics_may_fill_or_change_human_labels": False,
            "machine_diagnostics_have_semantic_authority": False,
            "machine_diagnostics_may_select_seed_or_training_target": False,
        },
        "authority": base._false_authority(),
    }
    return {**body, "protocol_digest": base.object_sha256(body)}


def _observer_template(
    *,
    slot: int,
    review_items: Sequence[Mapping[str, Any]],
    protocol_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(slot in (1, 2), "observer slot differs")
    responses = []
    for item in review_items:
        responses.append(
            {
                "review_item_id": item["review_item_id"],
                "registered_q0_state_correct_frames_0_15": None,
                "registered_q0_state_held_full81": None,
                "registered_target_terminal_absent_full81": None,
                "incomplete_directional_progress_visible": None,
                "incomplete_partial_state_held_frames_73_80": None,
                "registered_smooth_camera_motion_visible": None,
                "camera_cut_or_discontinuity_absent": None,
                "registered_appearance_only_change_visible": None,
                "camera_locked_full81": None,
                "nonregistered_subject_appearance_or_geometry_change_absent": None,
                "event_branch_pass": None,
                "identity_preserved_full81": None,
                "technical_quality_acceptable_full81": None,
                "observer_notes": None,
            }
        )
    body = {
        "schema_version": OBSERVER_TEMPLATE_SCHEMA_VERSION,
        "packet_id": PACKET_ID,
        "observer_slot": slot,
        "template_only": True,
        "semantic_status": base.SEMANTIC_STATUS,
        "observer_id": None,
        "observer_kind": None,
        "observer_authority_artifact": None,
        "observer_protocol_artifact": dict(protocol_binding),
        "completed_at": None,
        "independent_observer_required": True,
        "same_person_must_not_fill_both_slots": True,
        "copy_outside_sealed_packet_before_completion": True,
        "blindness_or_independence_established_by_template": False,
        "review_item_set_digest": base.object_sha256(list(review_items)),
        "responses": responses,
        "authority": base._false_authority(),
    }
    return {**body, "template_digest": base.object_sha256(body)}


def _render_blind_html(
    items: Sequence[Mapping[str, Any]], *, job_id: str, protocol_digest: str
) -> str:
    """Render the only stage-1 UI with opaque paths and labels."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        _require(
            re.fullmatch(r"source-[0-9]{4}", str(item["row_id"])) is not None,
            "blind source alias differs",
        )
        _require(
            re.fullmatch(r"candidate-[0-9]{4}", str(item["candidate_id"]))
            is not None,
            "blind candidate alias differs",
        )
        _require(
            re.fullmatch(
                r"media/candidates/[0-9]{4}-candidate-[0-9]{4}\.mp4",
                str(item["portable_candidate"]),
            )
            is not None,
            "blind candidate path is not opaque",
        )
        _require(
            re.fullmatch(
                r"media/sources/source-[0-9]{4}\.mp4",
                str(item["portable_source"]),
            )
            is not None,
            "blind source path is not opaque",
        )
        grouped.setdefault(str(item["row_id"]), []).append(item)

    sections: list[str] = []
    for row_index, row_items in enumerate(grouped.values(), start=1):
        first = row_items[0]
        cards: list[str] = []
        for item in row_items:
            instruction = str(item["branch_instruction"])
            prefix = {
                "incomplete": "",
                "camera_only": "Counterfactual camera-only negative: ",
                "appearance_only": "Counterfactual appearance-only negative: ",
            }[str(item["branch"])]
            _require(
                instruction.startswith(prefix),
                "blind registered criterion prefix differs",
            )
            instruction = instruction[len(prefix):]
            if instruction:
                instruction = instruction[0].upper() + instruction[1:]
            cards.append(
                f'''<article class="card"><header><span class="eyebrow">{html.escape(str(item['review_item_id']))}</span><h3>Registered evaluation criterion</h3><p>{html.escape(str(item['branch_start_state_caption']))}</p><p>{html.escape(instruction)}</p></header><video controls muted playsinline preload="metadata" src="{base._url(str(item['portable_candidate']))}"></video></article>'''
            )
        sections.append(
            f'''<section class="sample"><h2>Blind source set {row_index:02d}</h2><p class="muted">Registered identifiers, sampling metadata, and machine measurements are absent from this page.</p><div class="source"><article class="card source-card"><header><span class="eyebrow">SOURCE REFERENCE</span><h3>Hash-bound exact81 source</h3></header><video controls muted playsinline preload="metadata" src="{base._url(str(first['portable_source']))}"></video></article></div><div class="grid">{''.join(cards)}</div></section>'''
        )
    rendered = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SAIC T2V blind human review · Job {html.escape(job_id)}</title><style>
:root{{--ink:#18211d;--paper:#f3efe7;--panel:#fffdf7;--line:#cdc6b8;--muted:#68716c;--accent:#166953;--warn:#873d1b;--warnbg:#ffefdc}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1880px;margin:auto;padding:24px}}.hero,.sample{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:17px}}h1{{font-size:clamp(29px,4vw,52px);line-height:1.03;margin:7px 0 13px}}h2,h3,p{{margin-top:0}}.eyebrow{{font-size:11px;font-weight:800;color:var(--accent);letter-spacing:.1em}}.warning{{background:var(--warnbg);border:1px solid #dda171;border-radius:10px;padding:13px;color:var(--warn)}}.muted,.card p{{color:var(--muted)}}.source{{max-width:330px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(195px,1fr));gap:9px}}.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#faf8f2}}.source-card{{border-color:#6b9f8b;background:#edf7f2}}.card header{{padding:9px;min-height:150px}}video{{display:block;width:100%;aspect-ratio:1/1;object-fit:contain;background:#0c0e0d}}@media(max-width:1200px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{main{{padding:8px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="hero"><span class="eyebrow">INDEPENDENT HUMAN STAGE 1 · AUH {html.escape(job_id)}</span><h1>Blind full81 review</h1><p class="warning"><strong>Use only this stage-1 page and its embedded media.</strong> Watch every source/proposal pair through all 81 frames at normal speed, complete one assigned observer template independently, and seal that response outside this packet. Do not access any technical or assessor-private artifact before both response seals exist. Machine diagnostics cannot revise a human label.</p><p>Protocol digest: <code>{html.escape(protocol_digest)}</code>. Registered identifiers, sampling metadata, and all machine metrics are absent from this page.</p></section>{''.join(sections)}</main></body></html>'''
    lower = rendered.lower()
    assessor_private_ids = {
        str(item[field])
        for item in items
        for field in (
            "assessor_private_candidate_id",
            "assessor_private_source_row_id",
            "iid",
        )
    }
    forbidden = assessor_private_ids | {
        "incomplete", "camera_only", "appearance_only",
        "camera-only", "appearance-only",
    } | {
        f"s{item['seed']}" for item in items
    }
    _require(
        all(token.lower() not in lower for token in forbidden),
        "blind HTML leaks a registered identifier, sampling value, or branch name",
    )
    _require("href=" not in lower and "seed" not in lower, "blind HTML link/text closure differs")
    _require(
        rendered.count("<video ") == 68 and rendered.count(' src="') == 68,
        "blind HTML opaque media closure differs",
    )
    return rendered


def _configure_base() -> None:
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.RECEIPT_SCHEMA_VERSION = RECEIPT_SCHEMA_VERSION
    base.OBSERVER_TEMPLATE_SCHEMA_VERSION = OBSERVER_TEMPLATE_SCHEMA_VERSION
    base.OBSERVER_PROTOCOL_SCHEMA_VERSION = OBSERVER_PROTOCOL_SCHEMA_VERSION
    base.MASTER_SCHEMA_VERSION = MASTER_SCHEMA_VERSION
    base.ATTEMPT_SCHEMA_VERSION = generation.SCHEMA_VERSION
    base.PACKET_ID = PACKET_ID
    base.MASTER_BASENAME = MASTER_BASENAME
    base.SOURCE_MANIFEST_BASENAME = SOURCE_MANIFEST_BASENAME
    base.EVENT_SPEC_BASENAME = EVENT_SPEC_BASENAME
    base.ATTEMPT_RECEIPT_BASENAME = ATTEMPT_RECEIPT_BASENAME
    base.BRANCH_ORDER = BRANCH_ORDER
    base._load_and_validate_inputs = _load_and_validate_inputs
    base._observer_protocol = _observer_protocol
    base._observer_template = _observer_template
    base._render_blind_html = _render_blind_html
    base._copy_verified_create_only = _copy_verified_with_base_spec


def build_review(
    *,
    input_root: str | Path,
    output_root: str | Path,
    job_id: str,
    workers: int = 16,
) -> dict[str, Any]:
    _configure_base()
    return base.build_review(
        input_root=input_root,
        output_root=output_root,
        job_id=job_id,
        workers=workers,
    )


def validate_packet(output_root: str | Path) -> dict[str, Any]:
    _configure_base()
    receipt = base.validate_packet(output_root)
    output = base._plain_dir(output_root, label="v2 detached review packet")
    manifest = base._load_canonical_json(
        output / "review-manifest.json", label="v2 review manifest"
    )
    protocol = base._load_canonical_json(
        output / "observer-protocol.json", label="v2 observer protocol"
    )
    items = manifest.get("items")
    _require(type(items) is list and len(items) == 60, "v2 opaque item closure differs")
    for alias_index, item in enumerate(items, start=1):
        _require(
            item.get("candidate_id") == f"candidate-{alias_index:04d}"
            and item.get("review_item_id") == f"review-{alias_index:04d}"
            and item.get("registered_candidate_index") == alias_index,
            "v2 opaque candidate registration differs",
        )
        _require(
            item.get("assessor_private_candidate_id")
            == Path(str(item.get("candidate_input_path"))).parent.name
            and item.get("assessor_private_candidate_id")
            != item.get("candidate_id")
            and str(item.get("assessor_private_source_row_id", "")).endswith(
                "-" + str(item.get("iid"))
            ),
            "v2 assessor-private alias mapping differs",
        )
    _require(
        {item.get("row_id") for item in items}
        == {f"source-{index:04d}" for index in range(1, 9)}
        and all(
            re.fullmatch(r"source-[0-9]{4}", str(item.get("row_id")))
            is not None
            for item in items
        ),
        "v2 opaque source registration differs",
    )
    source_mapping = {
        (
            str(item["assessor_private_source_row_id"]),
            str(item["row_id"]),
        )
        for item in items
    }
    _require(
        len(source_mapping) == 8
        and len({private for private, _ in source_mapping}) == 8
        and len({alias for _, alias in source_mapping}) == 8
        and len({item["assessor_private_candidate_id"] for item in items}) == 60,
        "v2 assessor-private mapping cardinality differs",
    )
    for alias_index, item in enumerate(items, start=1):
        candidate_alias = f"candidate-{alias_index:04d}"
        _require(
            item.get("portable_source") == f"media/sources/{item['row_id']}.mp4"
            and item.get("portable_candidate")
            == f"media/candidates/{alias_index:04d}-{candidate_alias}.mp4"
            and item.get("portable_attempt_receipt")
            == f"evidence/attempts/{candidate_alias}.json"
            and item.get("portable_diagnostic")
            == f"diagnostics/{candidate_alias}.json",
            "v2 opaque artifact namespace differs",
        )
    blind_path = output / "blind-review.html"
    try:
        blind_text = blind_path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise base.DetachedReviewError("v2 blind HTML encoding differs") from error
    _require(
        blind_text
        == _render_blind_html(
            items,
            job_id=str(manifest.get("job_id")),
            protocol_digest=str(protocol.get("protocol_digest")),
        ),
        "v2 blind HTML bytes differ from opaque renderer",
    )
    nested = (
        manifest.get("input_bindings", {})
        .get("event_spec", {})
        .get("base_v1_spec", {})
    )
    _require(type(nested) is dict, "v2 packet base-v1 evidence binding absent")
    expected_path = output / "evidence" / BASE_V1_SPEC_BASENAME
    observed_path = base._plain_file(expected_path, label="portable base-v1 spec")
    _require(
        nested.get("path") == str(observed_path)
        and nested.get("portable_path") == f"evidence/{BASE_V1_SPEC_BASENAME}"
        and nested.get("sha256") == topup.BASE_V1_SPEC_RAW_SHA256
        and nested.get("bytes") == observed_path.stat().st_size
        and base.file_sha256(observed_path) == topup.BASE_V1_SPEC_RAW_SHA256,
        "v2 packet base-v1 evidence closure differs",
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--input-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--job-id", required=True)
    build.add_argument("--workers", type=int, default=16)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        receipt = build_review(
            input_root=args.input_root,
            output_root=args.output_root,
            job_id=args.job_id,
            workers=args.workers,
        )
    else:
        receipt = validate_packet(args.output_root)
    print(
        base.canonical_json_bytes(
            {
                "candidate_count": receipt["candidate_count"],
                "machine_diagnostic_count": receipt["machine_diagnostic_count"],
                "receipt_digest": receipt["receipt_digest"],
                "semantic_status": receipt["semantic_status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_V1_SPEC_BASENAME",
    "BRANCH_ORDER",
    "OBSERVER_PROTOCOL_SCHEMA_VERSION",
    "OBSERVER_TEMPLATE_SCHEMA_VERSION",
    "PACKET_ID",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_review",
    "main",
    "validate_packet",
]

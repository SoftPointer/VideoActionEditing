#!/usr/bin/env python3
"""Read-only candidate census and future D0 construction plan for 0817.

This module is deliberately not a manifest builder and never grants target or
training authority.  It inventories five legacy target/anchor pools, one
separately typed GOKU source-candidate pool, and a mandatory pinned InsViE
metadata CSV; it keeps seed/copy counts separate from semantic counts and emits
a *future requirement* plan for 2,000 D0 rows.  Actual rows
must still pass :mod:`action_edit_sft_manifest_v2`, its external equivalence
and qualification authorities, split closure, decode checks, and human review.

The important invariant is simple::

    candidate artifact != qualified edited target != train-ready row

No network access, media download, remote write, optimizer, or launcher is
implemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import action_edit_sft_manifest_v2 as manifest_v2


CENSUS_SCHEMA = "bernini-action-edit-candidate-census-v1"
D0_PLAN_SCHEMA = "bernini-action-edit-d0-2k-build-plan-v1"
ROW_CANDIDATE_REPORT_SCHEMA = "bernini-action-edit-row-candidate-report-v1"

# These pins are populated only after the complete current-census and D0 plan
# bytes have been independently regenerated.  A semantic edit requires a new
# schema/version and new pins; recomputing an object self-digest is insufficient.
FROZEN_CENSUS_DIGEST = "5c8ed30d36478f286c83b659075038ac33a8fa2d2dc3ebc6e0870a63a98e834f"
FROZEN_D0_PLAN_DIGEST = "e660f44c77ce2275603318b0b565deed2a647d9681ea7bb6a520aaaa463580e8"

INSVIE_REVISION = "12efa8dee73ec310f9ad42aee502ca4fad73bc30"
INSVIE_METADATA_SHA256 = (
    "b20cc78dce637eb3a1d5fbd77d6ff6684273b8a740cd6c8d26c57468e41e3ffb"
)
INSVIE_METADATA_SIZE = 61_164_130
INSVIE_CARD_ROW_COUNT = 1_019_570
INSVIE_CARD_ARCHIVE_BYTES = 765_117_294_333
INSVIE_SOURCE_ZIP_COUNT = 18
INSVIE_EDITED_ZIP_COUNT = 19
INSVIE_NORMALIZED_INSTRUCTION_COUNT = 85_399
INSVIE_DERIVED_SOURCE_ROOT_COUNT = 371_451

INSVIE_PRELIMINARY_ACTION_PATTERN = (
    r"\b(?:sit|stand|walk|run|jump|wave|raise|lower|pick|grab|reach|throw|"
    r"catch|kick|punch|hug|shake|dance|kneel|crouch|climb|ride|drive|eat|"
    r"drink|push|pull|lift|drop|pet|smile|laugh|talk|speak|cook|pour|"
    r"approach|enter|point|stretch|bend|swing|crawl|swim)(?:s|ed|ing)?\b"
)
INSVIE_PRELIMINARY_ACTION_COUNT = 1_243
INSVIE_PRELIMINARY_PREFIX_COUNTS = {
    "instructp2p": 617,
    "magicbrush": 578,
    "openvid_static": 1,
    "pexel_dynamic": 41,
    "pexel_static": 6,
}

TARGET_ANCHOR_POOL_IDS = (
    "legacy_full644_preview_pairs",
    "historical_factorial_forward_target_comparison18",
    "native_core4_rv2v_proposals",
    "quotient_fitted_unseen_anchor8",
    "outcome5_confirmation40",
)
SOURCE_CANDIDATE_POOL_IDS = (
    "goku_fullmotion_source_census_high_recall_16000",
)

_TARGET_POOL_CRITICAL_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "legacy_full644_preview_pairs": {
        "role": "legacy_preview_triplet_candidates",
        "candidate_artifact_count": 644,
        "semantic_candidate_count": None,
        "known_source_group_count": 0,
        "source_group_closure": False,
        "source_exposure": "ALL_HISTORICAL_OPTIMIZER_EXPOSED",
        "source_conditioned": True,
        "target_candidate_count": 644,
        "anchor_candidate_count": 0,
        "preference_candidate_count": 0,
        "possible_manifest_truth_classes_after_qualification": ["teacher-pseudo"],
    },
    "historical_factorial_forward_target_comparison18": {
        "role": "historical_paired_target_scout_and_failure_analysis",
        "candidate_artifact_count": 18,
        "semantic_candidate_count": 18,
        "known_source_group_count": 18,
        "source_group_closure": False,
        "source_exposure": "OBSOLETE_FIT_CALIBRATION_CONFIRMATION_SPLIT_REQUIRES_GLOBAL_REBUILD",
        "source_conditioned": True,
        "target_candidate_count": 18,
        "anchor_candidate_count": 0,
        "preference_candidate_count": 0,
        "possible_manifest_truth_classes_after_qualification": ["teacher-pseudo"],
    },
    "native_core4_rv2v_proposals": {
        "role": "source_conditioned_native_sampler_proposal",
        "candidate_artifact_count": 8,
        "semantic_candidate_count": 4,
        "known_source_group_count": 4,
        "source_group_closure": True,
        "source_exposure": "HISTORICAL_PROJECT_SOURCE",
        "source_conditioned": True,
        "target_candidate_count": 8,
        "anchor_candidate_count": 0,
        "preference_candidate_count": 0,
        "possible_manifest_truth_classes_after_qualification": ["teacher-pseudo"],
    },
    "quotient_fitted_unseen_anchor8": {
        "role": "detached_action_teacher_reference",
        "candidate_artifact_count": 8,
        "semantic_candidate_count": 8,
        "known_source_group_count": 8,
        "source_group_closure": True,
        "source_exposure": "MIXED_FITTED_EXPOSED_AND_HISTORICAL_UNSEEN",
        "source_conditioned": False,
        "target_candidate_count": 0,
        "anchor_candidate_count": 8,
        "preference_candidate_count": 0,
        "possible_manifest_truth_classes_after_qualification": [],
    },
    "outcome5_confirmation40": {
        "role": "standalone_t2v_anchor_negative_review_bank",
        "candidate_artifact_count": 40,
        "semantic_candidate_count": 20,
        "known_source_group_count": 2,
        "source_group_closure": True,
        "source_exposure": "HISTORICAL_PROJECT_SOURCE",
        "source_conditioned": False,
        "target_candidate_count": 0,
        "anchor_candidate_count": 4,
        "preference_candidate_count": 36,
        "possible_manifest_truth_classes_after_qualification": [],
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_ROOT_SUFFIX = re.compile(r"_\d+_\d+_\d+\.mp4\Z")
_NATIVE_PROPOSAL_DIR = re.compile(
    r"pair5-native-core4-v1-([0-9a-f]{16})-action-s(\d+)\Z"
)
_OUTCOME_CANDIDATE = re.compile(
    r"pair5-t2v-reserve4-(?:v1|seed2)-([0-9a-f]{16})-(.+)\Z"
)


class CandidateCensusError(RuntimeError):
    """Raised when candidate evidence is ambiguous or its closure changes."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CandidateCensusError("value is not canonical JSON: {}".format(error))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fail(message: str) -> None:
    raise CandidateCensusError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CandidateCensusError("cannot read JSON {}: {}".format(path, error))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    absolute = path.resolve(strict=False)
    _require(absolute == path.absolute(), "{} path is not canonical".format(label))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current = current / part
        item_stat = os.lstat(str(current))
        _require(not stat.S_ISLNK(item_stat.st_mode), "{} traverses symlink".format(label))
    item_stat = os.lstat(str(absolute))
    _require(
        stat.S_ISREG(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode),
        "{} is not a plain regular file".format(label),
    )
    return absolute


def _relative_envelope(path: Path, repo_root: Path) -> Dict[str, Any]:
    absolute = _regular_file(path, label="evidence")
    relative = absolute.relative_to(repo_root).as_posix()
    return {
        "path": relative,
        "sha256": _hash_file(absolute),
        "size_bytes": absolute.stat().st_size,
    }


def _external_envelope(path: Path) -> Dict[str, Any]:
    absolute = _regular_file(path, label="external evidence")
    return {
        "path": absolute.as_posix(),
        "sha256": _hash_file(absolute),
        "size_bytes": absolute.stat().st_size,
    }


def _closure(paths: Iterable[Path], repo_root: Path, *, label: str) -> Dict[str, Any]:
    members = [_relative_envelope(path, repo_root) for path in sorted(set(paths))]
    _require(bool(members), "{} closure is empty".format(label))
    return {
        "label": label,
        "member_count": len(members),
        "total_bytes": sum(item["size_bytes"] for item in members),
        "members_digest": object_sha256(members),
        "exact_member_closure": True,
    }


def _pool(
    *,
    pool_id: str,
    role: str,
    candidate_artifact_count: int,
    semantic_candidate_count: Optional[int],
    known_source_group_ids: Sequence[str],
    source_group_closure: bool,
    source_exposure: str,
    source_conditioned: Optional[bool],
    target_candidate_count: int,
    anchor_candidate_count: int,
    preference_candidate_count: int,
    possible_manifest_truth_classes: Sequence[str],
    permitted_uses: Sequence[str],
    forbidden_uses: Sequence[str],
    evidence_closures: Sequence[Mapping[str, Any]],
    blocking_reasons: Sequence[str],
) -> Dict[str, Any]:
    _require(candidate_artifact_count >= 0, "candidate count is negative")
    _require(target_candidate_count >= 0, "target candidate count is negative")
    _require(anchor_candidate_count >= 0, "anchor count is negative")
    _require(preference_candidate_count >= 0, "preference count is negative")
    if semantic_candidate_count is not None:
        _require(semantic_candidate_count >= 0, "semantic count is negative")
    allowed_truth = set(manifest_v2.TARGET_SEMANTIC_TRUTH_CLASSES)
    _require(
        set(possible_manifest_truth_classes).issubset(allowed_truth),
        "pool names a truth class outside manifest v2",
    )
    return {
        "pool_id": pool_id,
        "role": role,
        "candidate_artifact_count": candidate_artifact_count,
        "semantic_candidate_count": semantic_candidate_count,
        "known_source_group_count": len(set(known_source_group_ids)),
        "known_source_group_ids": sorted(set(known_source_group_ids)),
        "source_group_closure": source_group_closure,
        "source_exposure": source_exposure,
        "source_conditioned": source_conditioned,
        "seed_is_semantic_identity": False,
        "target_candidate_count": target_candidate_count,
        "anchor_candidate_count": anchor_candidate_count,
        "preference_candidate_count": preference_candidate_count,
        "possible_manifest_truth_classes_after_qualification": sorted(
            possible_manifest_truth_classes
        ),
        "current_manifest_truth_class": None,
        "target_eligible_count": 0,
        "train_ready_contribution": 0,
        "qualification_state": "UNQUALIFIED_OR_NOT_APPLICABLE",
        "permitted_uses": sorted(set(permitted_uses)),
        "forbidden_uses": sorted(set(forbidden_uses)),
        "evidence_closures": list(evidence_closures),
        "blocking_reasons": list(blocking_reasons),
    }


def _full644_pool(repo_root: Path) -> Dict[str, Any]:
    audit_path = repo_root / (
        "methods/bernini_action_editing/audits/"
        "bernini_r13_action_81f_full644_auh_20260805.json"
    )
    audit = _read_json(audit_path)
    dataset = audit.get("dataset", {})
    _require(audit.get("schema_version") == "bernini-r13-action-editing-full-run-audit-v1", "full644 schema differs")
    _require(dataset.get("membership_rows") == 644, "full644 declared rows differ")
    _require(dataset.get("training_authorized") is False, "full644 training flag differs")
    _require(dataset.get("training_use_forbidden") is True, "full644 use flag differs")
    _require(dataset.get("preview_only") is True, "full644 preview flag differs")
    return _pool(
        pool_id="legacy_full644_preview_pairs",
        role="legacy_preview_triplet_candidates",
        candidate_artifact_count=644,
        semantic_candidate_count=None,
        known_source_group_ids=[],
        source_group_closure=False,
        source_exposure="ALL_HISTORICAL_OPTIMIZER_EXPOSED",
        source_conditioned=True,
        target_candidate_count=644,
        anchor_candidate_count=0,
        preference_candidate_count=0,
        possible_manifest_truth_classes=["teacher-pseudo"],
        permitted_uses=["train_or_debug_source_census_after_byte_recovery", "legacy_engineering_baseline"],
        forbidden_uses=["count_as_d0", "heldout_split", "reuse_preview_target_without_requalification"],
        evidence_closures=[_closure([audit_path], repo_root, label="full644-local-audit-only")],
        blocking_reasons=[
            "original row manifest and source/target bytes are not locally closed",
            "equivalence and actor-scene authorities are missing",
            "preview targets are training-use-forbidden and post-video acceptance is pending",
        ],
    )


def _history18_pool(repo_root: Path) -> Dict[str, Any]:
    audit_path = (
        repo_root
        / "methods/bernini_action_editing/assets/factorial_forward_target_audit_review_v1.json"
    )
    note_path = (
        repo_root
        / "md/action_editing/20260813_1538_html_review_followup/10_existing_forward_target_audit.md"
    )
    audit = _read_json(audit_path)
    _require(
        audit.get("schema_version") == "bernini-factorial-forward-target-audit-v1",
        "history18 schema differs",
    )
    rows = audit.get("rows")
    _require(isinstance(rows, list) and len(rows) == 18, "history comparison row count differs")
    summary = audit.get("summary", {})
    _require(
        summary
        == {
            "reviewed": 18,
            "strict_eligible": 6,
            "compound_instruction": 10,
            "action_failure": 1,
            "wrong_target_family": 1,
        },
        "history comparison verdict counts differ",
    )
    authority = audit.get("authority", {})
    _require(authority.get("source_target_pair_reviewed") is True, "history18 pair review flag differs")
    _require(authority.get("training_target_authorized") is False, "history18 target authority differs")
    _require(authority.get("optimizer_step_authorized") is False, "history18 optimizer authority differs")
    source_ids = [item["source_id"] for item in rows]
    _require(len(set(source_ids)) == 18, "history18 source IDs are not unique")
    return _pool(
        pool_id="historical_factorial_forward_target_comparison18",
        role="historical_paired_target_scout_and_failure_analysis",
        candidate_artifact_count=18,
        semantic_candidate_count=18,
        known_source_group_ids=source_ids,
        source_group_closure=False,
        source_exposure="OBSOLETE_FIT_CALIBRATION_CONFIRMATION_SPLIT_REQUIRES_GLOBAL_REBUILD",
        source_conditioned=True,
        target_candidate_count=18,
        anchor_candidate_count=0,
        preference_candidate_count=0,
        possible_manifest_truth_classes=["teacher-pseudo"],
        permitted_uses=["failure_analysis", "six_strict_rows_as_generation_reference_outside_training_authority"],
        forbidden_uses=["treat_scout_strict_eligible_as_manifest_accepted", "reuse_old_split", "count_as_d0"],
        evidence_closures=[_closure([audit_path, note_path], repo_root, label="history18-local-audit-only")],
        blocking_reasons=[
            "only six of eighteen were strict scout-eligible and scout status is not target qualification",
            "ten compound instructions, one action failure, and one wrong-family target are rejected",
            "independent source and edited endpoint bytes are not locally closed in this repository",
            "old fit/calibration/confirmation labels are invalid under the 0817 global split",
            "audit explicitly denies training-target and optimizer authority",
        ],
    )


def _native8_pool(repo_root: Path) -> Dict[str, Any]:
    root = repo_root / "tmp/pair_v5_native_core4_action_population_0bb0f20"
    proposals = sorted(root.glob("*/rv2v.mp4"))
    sources = sorted(root.glob("*-source.mp4"))
    receipts = sorted(root.glob("*/receipt.json"))
    rollout_receipts = sorted(root.glob("*/pair-v5-rollout-receipt.json"))
    _require(len(proposals) == 8 and len(sources) == 4, "native proposal count differs")
    _require(len(receipts) == 8 and len(rollout_receipts) == 8, "native receipt count differs")
    source_ids: Set[str] = set()
    semantic_keys: Set[str] = set()
    seeds: Set[int] = set()
    for proposal, receipt_path in zip(proposals, receipts):
        match = _NATIVE_PROPOSAL_DIR.fullmatch(proposal.parent.name)
        _require(match is not None, "native proposal directory differs")
        iid, seed_text = match.groups()
        receipt = _read_json(receipt_path)
        output = receipt.get("outputs", {}).get("rv2v", {})
        _require(output.get("normalized_clean_latent", {}).get("artifact_role") == "native_sampler_proposal", "native proposal role differs")
        _require(output.get("sha256") == _hash_file(proposal), "native proposal SHA differs")
        source_ids.add(iid)
        semantic_keys.add(iid)
        seeds.add(int(seed_text))
    _require(len(source_ids) == 4 and len(semantic_keys) == 4 and len(seeds) == 2, "native proposal grouping differs")
    return _pool(
        pool_id="native_core4_rv2v_proposals",
        role="source_conditioned_native_sampler_proposal",
        candidate_artifact_count=8,
        semantic_candidate_count=4,
        known_source_group_ids=sorted(source_ids),
        source_group_closure=True,
        source_exposure="HISTORICAL_PROJECT_SOURCE",
        source_conditioned=True,
        target_candidate_count=8,
        anchor_candidate_count=0,
        preference_candidate_count=0,
        possible_manifest_truth_classes=["teacher-pseudo"],
        permitted_uses=["teacher_candidate_queue", "source_conditioned_target_qualification_queue"],
        forbidden_uses=["seed_as_new_row", "automatic_target", "count_as_d0"],
        evidence_closures=[
            _closure(proposals, repo_root, label="native8-proposal-media"),
            _closure(sources, repo_root, label="native8-source-media"),
            _closure(receipts + rollout_receipts, repo_root, label="native8-receipts"),
        ],
        blocking_reasons=[
            "two seeds collapse to four semantic source/instruction rows",
            "native_sampler_proposal is not a qualification verdict",
            "no dual-axis human review or manifest-v2 qualification authority exists",
        ],
    )


def _quotient8_pool(repo_root: Path) -> Dict[str, Any]:
    base = repo_root / "md/action_editing/20260815_reward/action_quotient_140846"
    fitted = sorted((base / "review/media").glob("*/anchor.mp4"))
    unseen = sorted((base / "unseen_review/media").glob("*/anchor.mp4"))
    anchors = fitted + unseen
    _require(len(fitted) == 4 and len(unseen) == 4, "quotient anchor count differs")
    ids = [path.parent.name for path in anchors]
    _require(len(set(ids)) == 8, "quotient anchor source grouping differs")
    return _pool(
        pool_id="quotient_fitted_unseen_anchor8",
        role="detached_action_teacher_reference",
        candidate_artifact_count=8,
        semantic_candidate_count=8,
        known_source_group_ids=ids,
        source_group_closure=True,
        source_exposure="MIXED_FITTED_EXPOSED_AND_HISTORICAL_UNSEEN",
        source_conditioned=False,
        target_candidate_count=0,
        anchor_candidate_count=8,
        preference_candidate_count=0,
        possible_manifest_truth_classes=[],
        permitted_uses=["detached_action_reference", "representation_diagnostic"],
        forbidden_uses=["rgb_or_latent_target", "merge_with_native_proposal_count", "count_as_d0"],
        evidence_closures=[_closure(anchors, repo_root, label="quotient8-anchor-media")],
        blocking_reasons=[
            "anchor appearance/scene is not source-faithful target truth",
            "fitted and unseen labels are obsolete for the 0817 split",
        ],
    )


def _outcome40_pool(repo_root: Path) -> Dict[str, Any]:
    root = repo_root / "artifacts/confirmation40_outcome5_report/public"
    export_path = root / "export-validation.json"
    audit_path = root / "evidence/generation-audit.json"
    export = _read_json(export_path)
    audit = _read_json(audit_path)
    _require(export.get("media_count") == 40 and export.get("media_exact_member_closure") is True, "outcome40 export closure differs")
    _require(audit.get("candidate_count") == 40, "outcome40 audit count differs")
    _require(audit.get("generated_media_is_editor_input_or_target") is False, "outcome40 role differs")
    _require(audit.get("independent_full81_review_performed") is False, "outcome40 review state differs")
    media = sorted((root / "media").glob("*/*.mp4"))
    _require(len(media) == 40, "outcome40 physical media count differs")
    export_rows = {item["relative_path"]: item for item in export["media"]}
    for path in media:
        relative = path.relative_to(root).as_posix()
        item = export_rows.get(relative)
        _require(item is not None, "outcome40 member is absent from export")
        _require(item["sha256"] == _hash_file(path), "outcome40 media SHA differs")
        _require(item["size_bytes"] == path.stat().st_size, "outcome40 media size differs")
    source_ids: Set[str] = set()
    branches: Set[str] = set()
    for item in audit["candidate_receipts"]:
        match = _OUTCOME_CANDIDATE.fullmatch(item["candidate_id"])
        _require(match is not None, "outcome40 candidate ID differs")
        source_ids.add(match.group(1))
        branches.add(match.group(2))
    _require(len(source_ids) == 2 and len(branches) == 10, "outcome40 semantic axes differ")
    evidence = sorted((root / "evidence").glob("*"))
    _require(len(evidence) == 7, "outcome40 evidence member count differs")
    return _pool(
        pool_id="outcome5_confirmation40",
        role="standalone_t2v_anchor_negative_review_bank",
        candidate_artifact_count=40,
        semantic_candidate_count=20,
        known_source_group_ids=sorted(source_ids),
        source_group_closure=True,
        source_exposure="HISTORICAL_PROJECT_SOURCE",
        source_conditioned=False,
        target_candidate_count=0,
        anchor_candidate_count=4,
        preference_candidate_count=36,
        possible_manifest_truth_classes=[],
        permitted_uses=["detached_anchor_review", "hard_negative_review", "evaluator_calibration_candidate"],
        forbidden_uses=["standalone_t2v_as_target", "seed_as_new_row", "count_as_d0"],
        evidence_closures=[
            _closure(media, repo_root, label="outcome40-media"),
            _closure([export_path, audit_path] + evidence, repo_root, label="outcome40-local-evidence"),
        ],
        blocking_reasons=[
            "40 files are two IIDs by two seeds by ten branches",
            "seed collapse leaves twenty branch semantics, not forty edit rows",
            "only four raw action branches exist and none has independent full-video review",
            "generation audit explicitly denies editor-input/target and optimizer authority",
        ],
    )


def _goku_source_candidate_pool() -> Dict[str, Any]:
    base = (
        "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
        "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
        "fullmotion_atomic1000_round2_20260813T060000Z/source_census_v1"
    )
    authorities = [
        {
            "path": base + "/final/summary.json",
            "sha256": "d87e7b549ef50aa20bb9343e3980c73230562ee0e96eb3b79c1d3cf01877f300",
            "size_bytes": 24_685,
            "schema_version": "motive-goku-atomic-source-expand-summary-v1",
            "status": "complete",
        },
        {
            "path": base + "/final/selected.jsonl",
            "sha256": "7a76d6b2dec10203f1af00016c60f6bb654e2ebea19d5b08b74a49d31300cac2",
            "size_bytes": 7_196_076,
            "schema_version": None,
            "status": None,
        },
        {
            "path": base + "/high_recall_16000_v1/summary.json",
            "sha256": "acd873686bfff42fff517f60651c117dc0cf28e9c62ab6bc64748135166b93b8",
            "size_bytes": 387_953,
            "schema_version": "motive-goku-atomic-source-high-recall-summary-v1",
            "status": "complete",
        },
        {
            "path": base + "/high_recall_16000_v1/selected.jsonl",
            "sha256": "cb81325bf79996d1f26a8859df5f06a3334c96d91c18d6eec6c61f6abd6b2fde",
            "size_bytes": 75_770_321,
            "schema_version": None,
            "status": None,
        },
        {
            "path": base + "/high_recall_16000_epochs_v1/summary.json",
            "sha256": "ae63b0b752f15c3785f21d5af1f579ffcc906e92df6e669740addf5fb070bae8",
            "size_bytes": 9_955,
            "schema_version": "motive-goku-atomic-candidate-epochs-summary-v1",
            "status": "complete",
        },
        {
            "path": base + "/high_recall_16000_epochs_v1/done.json",
            "sha256": "87f50dd3bcfb93c6400d32573f6434fcbd4fe5e443dd5ed321e82105afe3b7aa",
            "size_bytes": 1_286,
            "schema_version": "motive-goku-atomic-candidate-epochs-done-v1",
            "status": "complete",
        },
    ]
    return {
        "pool_id": "goku_fullmotion_source_census_high_recall_16000",
        "role": "LICENSED_PAIRED_SOURCE_CANDIDATE_ENDPOINTS_UNQUALIFIED",
        "candidate_class_after_full_qualification": "licensed-paired",
        "possible_target_provenance_after_full_qualification": "licensed-dataset",
        "authority_scope": "USER_SUPPLIED_REMOTE_METADATA_PINS_PLUS_UNSEALED_READONLY_ENDPOINT_STAT",
        "authority_observation_origin": "PARENT_AGENT_REPORTED_READONLY_REMOTE_AUDIT",
        "locally_reverified_by_this_planner": False,
        "official_dataset_card_authority": {
            "path": "/vast/users/guangyi.chen/dataset/goku/README.md",
            "sha256": "c8fb7f1a024c0c83d72e46ac76dfca590b95da69862b14f7dca6a15c910a4e49",
            "size_bytes": 7_823,
            "repository": "Goku-2M/GOKU-2M",
            "license_label": "CC-BY-NC-4.0",
            "license_use_scope": "NON_COMMERCIAL_RESEARCH_WITH_ATTRIBUTION",
            "declared_configuration": "subject_movement",
            "declared_fields": ["source", "edited", "instruction"],
            "hf_cache_metadata_opaque_lines": [
                "f0ee242abea24e3a410d5e49b9d5821450d08acb",
                "d67f60931ccf54502c7d9531eca00617bb5c0e5b",
                "1784845015.9175262",
            ],
            "cache_timestamp_has_qualification_semantics": False,
        },
        "paper_method_context_not_row_qualification": {
            "paper_id": "arXiv:2606.30599",
            "section_3_2_subject_movement": (
                "Gemini-2.5-Pro produces two action descriptions for the same "
                "subject and Wan2.2 synthesizes both endpoints to target identity "
                "and background consistency with action variation"
            ),
            "section_3_3_provider_filtering": (
                "three-stage provider filtering with approximately 88 percent "
                "reported filtered"
            ),
            "provider_filtering_is_0817_per_row_human_review": False,
        },
        "remote_authority_members": authorities,
        "remote_member_identity_contract": {
            "regular_file": True,
            "symlink": False,
            "nlink": 1,
            "mode": "0644",
            "remote_bytes_read_by_this_planner": False,
        },
        "counts": {
            "upstream_corpus_approximate_N": "~222k",
            "mother_or_requested_candidate_N": 28_538,
            "strict_selected_N": 1_676,
            "upstream_summary_field_eligible_N_not_project_qualification": 1_676,
            "high_recall_evaluated_N": 28_538,
            "high_recall_upstream_filter_eligible_N_not_project_qualification": 20_861,
            "high_recall_selected_N": 16_000,
            "epoch_count": 8,
            "selected_iid_unique": True,
            "selected_group_unique": True,
        },
        "user_reported_readonly_endpoint_stat": {
            "sealed_authority": False,
            "source_regular_N": 16_000,
            "edited_regular_N": 16_000,
            "source_unique_N": 16_000,
            "edited_unique_N": 16_000,
            "symlink_N": 0,
            "bad_N": 0,
            "source_total_bytes": 107_717_235_357,
            "edited_total_bytes": 108_355_603_046,
        },
        "candidate_source_N": 16_000,
        "candidate_paired_endpoint_presence_N": 16_000,
        "source_catalog_v2_eligible_N": 0,
        "target_eligible_N": 0,
        "train_ready_contribution": 0,
        "production_eligible": False,
        "formal_training_authorized": False,
        "historical_optimizer_exposure_closure": False,
        "equivalence_authority_present": False,
        "pair_provenance_authority_present": False,
        "target_qualification_authority_present": False,
        "split_authority_present": False,
        "local_media_downloaded_by_this_planner": 0,
        "permitted_uses": [
            "source_catalog_candidate_queue",
            "licensed_paired_endpoint_provenance_and_target_qualification_queue",
            "planning_only_count_diagnostic",
        ],
        "forbidden_uses": [
            "count_as_d0",
            "count_endpoint_presence_as_paired_truth",
            "promotion_or_locked_final_before_exposure_audit",
            "production_release",
            "train_sampler",
        ],
        "blocking_reasons": [
            "the 16k endpoint stat is user-reported and has not been sealed as an exact pair-byte authority",
            "source equivalence and historical optimizer exposure closures are absent",
            "the official card supports licensed-paired candidacy but does not replace exact per-row byte and provenance closure",
            "provider three-stage filtering is not an 0817 per-row full-video human-review receipt",
            "edited endpoint presence does not prove clean-target truth",
            "target qualification, split authority, and action/preservation review are absent",
        ],
    }


def audit_insvie_metadata(path: Path) -> Dict[str, Any]:
    envelope = _external_envelope(path)
    _require(envelope["sha256"] == INSVIE_METADATA_SHA256, "InsViE metadata SHA differs")
    _require(envelope["size_bytes"] == INSVIE_METADATA_SIZE, "InsViE metadata size differs")
    action_pattern = re.compile(INSVIE_PRELIMINARY_ACTION_PATTERN, re.IGNORECASE)
    row_count = 0
    video_ids: Set[str] = set()
    source_roots: Set[str] = set()
    normalized_instructions: Set[str] = set()
    preliminary_counts: Dict[str, int] = {
        key: 0 for key in INSVIE_PRELIMINARY_PREFIX_COUNTS
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames == ["video", "instruct"], "InsViE columns differ")
        for item in reader:
            row_count += 1
            video = item["video"]
            instruction = item["instruct"]
            _require(bool(video) and bool(instruction), "InsViE metadata contains empty cell")
            video_ids.add(video)
            source_roots.add(_SOURCE_ROOT_SUFFIX.sub("", video))
            normalized_instructions.add(instruction.strip().lower())
            if action_pattern.search(instruction):
                if video.startswith("pexel-static-"):
                    prefix = "pexel_static"
                elif video.startswith("pexel-"):
                    prefix = "pexel_dynamic"
                elif video.startswith("openvid"):
                    prefix = "openvid_static"
                elif video.startswith("magicbrush"):
                    prefix = "magicbrush"
                elif video.startswith("instructp2p"):
                    prefix = "instructp2p"
                else:
                    _fail("unknown InsViE prefix in preliminary action hit")
                preliminary_counts[prefix] += 1
    _require(row_count == INSVIE_CARD_ROW_COUNT, "InsViE metadata row count differs")
    _require(len(video_ids) == INSVIE_CARD_ROW_COUNT, "InsViE video IDs are not unique")
    _require(len(source_roots) == INSVIE_DERIVED_SOURCE_ROOT_COUNT, "InsViE source-root count differs")
    _require(len(normalized_instructions) == INSVIE_NORMALIZED_INSTRUCTION_COUNT, "InsViE normalized instruction count differs")
    _require(preliminary_counts == INSVIE_PRELIMINARY_PREFIX_COUNTS, "InsViE preliminary prefix counts differ")
    _require(sum(preliminary_counts.values()) == INSVIE_PRELIMINARY_ACTION_COUNT, "InsViE preliminary action count differs")
    return {
        "evidence": envelope,
        "pinned_hf_repo_revision": INSVIE_REVISION,
        "metadata_row_count": row_count,
        "unique_video_id_count": len(video_ids),
        "derived_source_root_count": len(source_roots),
        "normalized_lower_strip_instruction_count": len(normalized_instructions),
        "preliminary_action_regex": INSVIE_PRELIMINARY_ACTION_PATTERN,
        "preliminary_action_regex_sha256": hashlib.sha256(
            INSVIE_PRELIMINARY_ACTION_PATTERN.encode("utf-8")
        ).hexdigest(),
        "preliminary_action_candidate_count": sum(preliminary_counts.values()),
        "preliminary_action_prefix_counts": preliminary_counts,
        "preliminary_action_screen_authoritative": False,
        "known_false_positive_examples": [
            "pointed/point used outside an action transition",
            "raise contrast or another image operation",
            "drink or ride used as a noun",
        ],
        "source_video_bytes_materialized": 0,
        "edited_video_bytes_materialized": 0,
        "target_eligible_count": 0,
        "train_ready_contribution": 0,
    }


def _known_group_overlap(pools: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    owners: Dict[str, List[str]] = {}
    for pool in pools:
        for group_id in pool["known_source_group_ids"]:
            owners.setdefault(group_id, []).append(pool["pool_id"])
    return [
        {"source_group_id": group_id, "pool_ids": sorted(pool_ids)}
        for group_id, pool_ids in sorted(owners.items())
        if len(pool_ids) > 1
    ]


def _assemble_current_census(
    repo_root: Path,
    *,
    insvie_metadata_csv: Optional[Path],
) -> Dict[str, Any]:
    """Assemble bytes for the one frozen 2026-08-17 census.

    This internal function exists so maintainers can independently calculate a
    proposed new pin.  Public construction always passes through
    :func:`validate_current_census` and therefore cannot accept a locally
    re-digested semantic mutation.
    """

    repo = repo_root.resolve(strict=True)
    manifest_path = repo / "methods/bernini_action_editing/action_edit_sft_manifest_v2.py"
    _require(
        insvie_metadata_csv is not None,
        "the pinned InsViE metadata CSV is mandatory for the frozen census",
    )
    target_anchor_pools = [
        _full644_pool(repo),
        _history18_pool(repo),
        _native8_pool(repo),
        _quotient8_pool(repo),
        _outcome40_pool(repo),
    ]
    source_candidate_pools = [_goku_source_candidate_pool()]
    insvie = audit_insvie_metadata(insvie_metadata_csv)
    unsigned = {
        "schema_version": CENSUS_SCHEMA,
        "authority_scope": "READ_ONLY_CANDIDATE_ACCOUNTING_NOT_DATA_QUALIFICATION",
        "manifest_v2_contract": {
            "path": manifest_path.relative_to(repo).as_posix(),
            "sha256": _hash_file(manifest_path),
            "d0_minimum_count": manifest_v2.D0_MINIMUM_COUNT,
            "source_semantic_edit_cap": manifest_v2.SOURCE_SEMANTIC_EDIT_CAP,
            "actor_scene_row_cap": manifest_v2.ACTOR_SCENE_ROW_CAP,
        },
        "candidate_pool_partition_contract": {
            "target_anchor_candidate_pool_count": 5,
            "source_candidate_pool_count": 1,
            "pool_classes_must_not_be_merged_for_counting": True,
            "paired_endpoint_presence_is_target_truth": False,
            "candidate_source_is_source_catalog_v2_eligible": False,
        },
        "target_anchor_candidate_pools": target_anchor_pools,
        "source_candidate_pools": source_candidate_pools,
        "known_cross_pool_source_group_overlaps": _known_group_overlap(
            target_anchor_pools
        ),
        "insvie_external_metadata_candidate": insvie,
        "train_ready_N": 0,
        "D0_train_eligible_effective_N": 0,
        "d0_gap": manifest_v2.D0_MINIMUM_COUNT,
        "status": "DATA_NOT_READY",
        "formal_training_authorized": False,
        "optimizer_launch_authorized": False,
        "remote_write_performed": False,
        "media_download_performed": False,
    }
    return {**unsigned, "census_digest": object_sha256(unsigned)}


def build_current_census(
    repo_root: Path,
    *,
    insvie_metadata_csv: Path,
) -> Dict[str, Any]:
    return validate_current_census(
        _assemble_current_census(
            repo_root,
            insvie_metadata_csv=insvie_metadata_csv,
        )
    )


def validate_current_census(value: Any) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "census must be an object")
    expected_keys = {
        "schema_version",
        "authority_scope",
        "manifest_v2_contract",
        "candidate_pool_partition_contract",
        "target_anchor_candidate_pools",
        "source_candidate_pools",
        "known_cross_pool_source_group_overlaps",
        "insvie_external_metadata_candidate",
        "train_ready_N",
        "D0_train_eligible_effective_N",
        "d0_gap",
        "status",
        "formal_training_authorized",
        "optimizer_launch_authorized",
        "remote_write_performed",
        "media_download_performed",
        "census_digest",
    }
    _require(set(value) == expected_keys, "census top-level field closure differs")
    row = dict(value)
    _require(row.get("schema_version") == CENSUS_SCHEMA, "census schema differs")
    digest = row.pop("census_digest", None)
    _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None, "census digest is invalid")
    _require(object_sha256(row) == digest, "census self-digest differs")
    _require(FROZEN_CENSUS_DIGEST != "PENDING_FREEZE", "census pin is not frozen")
    _require(digest == FROZEN_CENSUS_DIGEST, "census differs from frozen semantic template")
    _require(
        value.get("authority_scope")
        == "READ_ONLY_CANDIDATE_ACCOUNTING_NOT_DATA_QUALIFICATION",
        "census authority scope differs",
    )
    manifest_contract = value.get("manifest_v2_contract")
    _require(
        isinstance(manifest_contract, Mapping)
        and set(manifest_contract)
        == {
            "path",
            "sha256",
            "d0_minimum_count",
            "source_semantic_edit_cap",
            "actor_scene_row_cap",
        },
        "manifest-v2 contract field closure differs",
    )
    _require(
        manifest_contract["path"]
        == "methods/bernini_action_editing/action_edit_sft_manifest_v2.py",
        "manifest-v2 path differs",
    )
    _require(
        _SHA256.fullmatch(manifest_contract["sha256"]) is not None,
        "manifest-v2 SHA is invalid",
    )
    _require(
        manifest_contract["d0_minimum_count"] == manifest_v2.D0_MINIMUM_COUNT
        and manifest_contract["source_semantic_edit_cap"]
        == manifest_v2.SOURCE_SEMANTIC_EDIT_CAP
        and manifest_contract["actor_scene_row_cap"]
        == manifest_v2.ACTOR_SCENE_ROW_CAP,
        "manifest-v2 fixed counts differ",
    )
    _require(
        value.get("candidate_pool_partition_contract")
        == {
            "target_anchor_candidate_pool_count": 5,
            "source_candidate_pool_count": 1,
            "pool_classes_must_not_be_merged_for_counting": True,
            "paired_endpoint_presence_is_target_truth": False,
            "candidate_source_is_source_catalog_v2_eligible": False,
        },
        "candidate-pool partition contract differs",
    )
    _require(value.get("train_ready_N") == 0, "candidate census cannot grant train-ready rows")
    _require(value.get("D0_train_eligible_effective_N") == 0, "candidate census cannot claim D0 rows")
    _require(value.get("d0_gap") == manifest_v2.D0_MINIMUM_COUNT, "candidate census D0 gap differs")
    _require(value.get("status") == "DATA_NOT_READY", "candidate census status differs")
    _require(value.get("formal_training_authorized") is False, "candidate census grants formal training")
    _require(value.get("optimizer_launch_authorized") is False, "candidate census grants optimizer launch")
    _require(value.get("remote_write_performed") is False, "candidate census claims remote write")
    _require(value.get("media_download_performed") is False, "candidate census claims media download")
    pools = value.get("target_anchor_candidate_pools")
    _require(isinstance(pools, list) and len(pools) == 5, "target/anchor pool closure differs")
    _require(
        tuple(item.get("pool_id") for item in pools) == TARGET_ANCHOR_POOL_IDS,
        "target/anchor pool ID or order differs",
    )
    pool_keys = {
        "pool_id",
        "role",
        "candidate_artifact_count",
        "semantic_candidate_count",
        "known_source_group_count",
        "known_source_group_ids",
        "source_group_closure",
        "source_exposure",
        "source_conditioned",
        "seed_is_semantic_identity",
        "target_candidate_count",
        "anchor_candidate_count",
        "preference_candidate_count",
        "possible_manifest_truth_classes_after_qualification",
        "current_manifest_truth_class",
        "target_eligible_count",
        "train_ready_contribution",
        "qualification_state",
        "permitted_uses",
        "forbidden_uses",
        "evidence_closures",
        "blocking_reasons",
    }
    for pool in pools:
        _require(set(pool) == pool_keys, "target/anchor pool field closure differs")
        contract = _TARGET_POOL_CRITICAL_CONTRACTS[pool["pool_id"]]
        for field, expected in contract.items():
            _require(pool[field] == expected, "{} {} differs".format(pool["pool_id"], field))
        _require(pool["train_ready_contribution"] == 0, "pool grants train authority")
        _require(pool["target_eligible_count"] == 0, "pool grants target authority")
        _require(pool["seed_is_semantic_identity"] is False, "seed changes semantic identity")
        _require(pool["current_manifest_truth_class"] is None, "unqualified pool claims truth")
        _require(
            pool["qualification_state"] == "UNQUALIFIED_OR_NOT_APPLICABLE",
            "pool qualification state differs",
        )
        _require(
            pool["known_source_group_count"] == len(set(pool["known_source_group_ids"])),
            "pool source-group accounting differs",
        )
        _require(bool(pool["evidence_closures"]), "pool evidence closure is empty")
        for closure in pool["evidence_closures"]:
            _require(
                set(closure)
                == {
                    "label",
                    "member_count",
                    "total_bytes",
                    "members_digest",
                    "exact_member_closure",
                }
                and closure["exact_member_closure"] is True
                and type(closure["member_count"]) is int
                and closure["member_count"] > 0
                and type(closure["total_bytes"]) is int
                and closure["total_bytes"] > 0
                and _SHA256.fullmatch(closure["members_digest"]) is not None,
                "pool evidence closure differs",
            )
    overlaps = value.get("known_cross_pool_source_group_overlaps")
    expected_overlaps = [
        {
            "source_group_id": iid,
            "pool_ids": [
                "native_core4_rv2v_proposals",
                "quotient_fitted_unseen_anchor8",
            ],
        }
        for iid in (
            "7b88a1ca1f804f41",
            "841b5e0080a1441d",
            "a35b590961d24694",
            "a66e6818e4144928",
        )
    ]
    _require(overlaps == expected_overlaps, "known target/anchor overlap closure differs")
    source_pools = value.get("source_candidate_pools")
    _require(
        isinstance(source_pools, list)
        and tuple(item.get("pool_id") for item in source_pools)
        == SOURCE_CANDIDATE_POOL_IDS,
        "source-candidate pool closure differs",
    )
    _require(
        source_pools == [_goku_source_candidate_pool()],
        "Goku source-candidate authority or semantics differ",
    )
    insvie = value.get("insvie_external_metadata_candidate")
    _require(isinstance(insvie, Mapping), "pinned InsViE metadata authority is missing")
    _require(
        set(insvie)
        == {
            "evidence",
            "pinned_hf_repo_revision",
            "metadata_row_count",
            "unique_video_id_count",
            "derived_source_root_count",
            "normalized_lower_strip_instruction_count",
            "preliminary_action_regex",
            "preliminary_action_regex_sha256",
            "preliminary_action_candidate_count",
            "preliminary_action_prefix_counts",
            "preliminary_action_screen_authoritative",
            "known_false_positive_examples",
            "source_video_bytes_materialized",
            "edited_video_bytes_materialized",
            "target_eligible_count",
            "train_ready_contribution",
        },
        "InsViE metadata census field closure differs",
    )
    _require(
        insvie["evidence"]
        == {
            "path": "/private/tmp/insvie_12efa8d_train_insvie_align.csv",
            "sha256": INSVIE_METADATA_SHA256,
            "size_bytes": INSVIE_METADATA_SIZE,
        },
        "InsViE metadata evidence pin differs",
    )
    _require(insvie["pinned_hf_repo_revision"] == INSVIE_REVISION, "InsViE revision differs")
    _require(
        insvie["metadata_row_count"] == INSVIE_CARD_ROW_COUNT
        and insvie["unique_video_id_count"] == INSVIE_CARD_ROW_COUNT
        and insvie["derived_source_root_count"] == INSVIE_DERIVED_SOURCE_ROOT_COUNT
        and insvie["normalized_lower_strip_instruction_count"]
        == INSVIE_NORMALIZED_INSTRUCTION_COUNT,
        "InsViE fixed metadata counts differ",
    )
    _require(
        insvie["preliminary_action_regex"] == INSVIE_PRELIMINARY_ACTION_PATTERN
        and insvie["preliminary_action_regex_sha256"]
        == hashlib.sha256(INSVIE_PRELIMINARY_ACTION_PATTERN.encode("utf-8")).hexdigest()
        and insvie["preliminary_action_candidate_count"]
        == INSVIE_PRELIMINARY_ACTION_COUNT
        and insvie["preliminary_action_prefix_counts"]
        == INSVIE_PRELIMINARY_PREFIX_COUNTS
        and insvie["preliminary_action_screen_authoritative"] is False,
        "InsViE preliminary screen semantics differ",
    )
    _require(
        insvie["source_video_bytes_materialized"] == 0
        and insvie["edited_video_bytes_materialized"] == 0
        and insvie["target_eligible_count"] == 0
        and insvie["train_ready_contribution"] == 0,
        "InsViE candidate grants media or target authority",
    )
    return dict(value)


def classify_candidate_rows(rows: Sequence[Any], *, verify_files: bool = True) -> Dict[str, Any]:
    """Validate row-shaped candidates without ever granting train authority."""

    normalized = [manifest_v2.validate_train_row(item, verify_files=verify_files) for item in rows]
    statuses: Dict[str, int] = {key: 0 for key in manifest_v2.QUALIFICATION_STATUSES}
    for row in normalized:
        statuses[row["target"]["qualification_status"]] += 1
    unsigned = {
        "schema_version": ROW_CANDIDATE_REPORT_SCHEMA,
        "candidate_row_count": len(normalized),
        "qualification_shape_counts": statuses,
        "accepted_shape_only_N": statuses["accepted"],
        "train_ready_N": 0,
        "formal_manifest_v2_required": True,
        "equivalence_authority_required": True,
        "qualification_authority_required": True,
        "split_and_group_leakage_receipt_required": True,
        "decode_and_human_review_closure_required": True,
        "formal_training_authorized": False,
    }
    return {**unsigned, "report_digest": object_sha256(unsigned)}


def _assemble_d0_plan(sealed: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble a proposed plan; public construction validates frozen pins."""

    matrix = [
        {"training_subset": "general_edit", "semantic_truth_class": "licensed-paired", "target_provenance": "licensed-dataset", "future_required_rows": 300},
        {"training_subset": "general_edit", "semantic_truth_class": "real-counterfactual", "target_provenance": "real", "future_required_rows": 50},
        {"training_subset": "general_edit", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 50},
        {"training_subset": "general_edit", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 200},
        {"training_subset": "action_motion", "semantic_truth_class": "licensed-paired", "target_provenance": "licensed-dataset", "future_required_rows": 100},
        {"training_subset": "action_motion", "semantic_truth_class": "real-counterfactual", "target_provenance": "real", "future_required_rows": 50},
        {"training_subset": "action_motion", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 150},
        {"training_subset": "action_motion", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "action_motion", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "interaction_contact", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 150},
        {"training_subset": "interaction_contact", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 100},
        {"training_subset": "interaction_contact", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "noop_preservation", "semantic_truth_class": "noop", "target_provenance": "real", "future_required_rows": 200},
        {"training_subset": "noop_preservation", "semantic_truth_class": "noop", "target_provenance": "licensed-dataset", "future_required_rows": 100},
        {"training_subset": "long_horizon", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 100},
    ]
    subset_counts: Dict[str, int] = {key: 0 for key in manifest_v2.TRAINING_SUBSETS}
    truth_counts: Dict[str, int] = {key: 0 for key in manifest_v2.TARGET_SEMANTIC_TRUTH_CLASSES}
    provenance_counts: Dict[str, int] = {key: 0 for key in manifest_v2.TARGET_PROVENANCE}
    for item in matrix:
        count = item["future_required_rows"]
        subset_counts[item["training_subset"]] += count
        truth_counts[item["semantic_truth_class"]] += count
        provenance_counts[item["target_provenance"]] += count
    unsigned = {
        "schema_version": D0_PLAN_SCHEMA,
        "input_census_digest": sealed["census_digest"],
        "counts_are_future_requirements_not_existing_assets": True,
        "current_train_ready_N": 0,
        "target_D0_train_eligible_effective_N": manifest_v2.D0_MINIMUM_COUNT,
        "current_gap": manifest_v2.D0_MINIMUM_COUNT,
        "future_row_matrix": matrix,
        "future_training_subset_counts": subset_counts,
        "future_target_truth_class_counts": truth_counts,
        "future_target_provenance_counts": provenance_counts,
        "provenance_gates": {
            "teacher_pseudo_max_fraction": 0.50,
            "teacher_pseudo_planned_fraction": provenance_counts["teacher-pseudo"] / 2000.0,
            "single_teacher_max_rows": 300,
            "single_teacher_max_fraction": 0.15,
            "minimum_train_teacher_family_count_for_planned_teacher_rows": 3,
            "continuation_max_rows": 400,
            "continuation_max_fraction": 0.20,
            "noop_exact_rows": 300,
            "noop_exact_fraction": 0.15,
            "high_confidence_real_simulator_licensed_non_noop_min_rows": 700,
            "high_confidence_real_simulator_licensed_non_noop_planned_rows": 850,
        },
        "source_and_group_plan": {
            "future_unique_source_goal": 1000,
            "future_rows_per_source_goal": 2,
            "hard_source_semantic_edit_cap": manifest_v2.SOURCE_SEMANTIC_EDIT_CAP,
            "hard_actor_scene_row_cap": manifest_v2.ACTOR_SCENE_ROW_CAP,
            "full644_recoverable_source_upper_bound_before_dedup": 644,
            "minimum_new_unexposed_sources_if_all_644_are_verified_unique": 356,
            "all_full644_sources_must_be_marked_exposed": True,
            "seed_transcode_copy_paraphrase_increase_effective_N": False,
            "equivalence_authority_frozen_before_split": True,
        },
        "source_candidate_pool_inputs": [
            {
                "pool_id": "goku_fullmotion_source_census_high_recall_16000",
                "role": "licensed_paired_source_candidate_pending_exact_pair_closure",
                "current_candidate_source_N": 16_000,
                "current_endpoint_presence_pair_N": 16_000,
                "current_source_catalog_v2_eligible_N": 0,
                "current_target_eligible_N": 0,
                "current_train_ready_contribution": 0,
                "production_eligible": False,
                "download_authorized_by_this_plan": False,
                "required_before_use": [
                    "seal exact source/edited member bytes and per-row pair provenance",
                    "freeze source equivalence and historical optimizer exposure",
                    "verify CC-BY-NC-4.0 non-commercial research scope and attribution",
                    "run action-axis, preservation-axis, decode, and full-video human review",
                    "assign globally disjoint split and bind qualification receipts",
                ],
            }
        ],
        "licensed_goku_candidate_materialization": {
            "counts_are_future_requirements_not_existing_assets": True,
            "current_materialized_and_qualified_N": 0,
            "future_required_total_N": 500,
            "future_rows": [
                {
                    "route": "goku_subject_movement_general_endpoint_pair",
                    "source_queue": "upstream_combined_approximately_222k",
                    "training_subset": "general_edit",
                    "semantic_truth_class": "licensed-paired",
                    "target_provenance": "licensed-dataset",
                    "future_required_rows": 300,
                },
                {
                    "route": "goku_subject_movement_strict_continuous_action_pair",
                    "source_queue": "source_census_final_strict_1676",
                    "training_subset": "action_motion",
                    "semantic_truth_class": "licensed-paired",
                    "target_provenance": "licensed-dataset",
                    "future_required_rows": 100,
                },
                {
                    "route": "independent_source_exact_noop",
                    "source_queue": "independent_source_bytes_after_global_dedup",
                    "training_subset": "noop_preservation",
                    "semantic_truth_class": "noop",
                    "target_provenance": "licensed-dataset",
                    "future_required_rows": 100,
                },
            ],
            "required_per_row_closure": [
                "exact source and target hashes with provider pin",
                "decode and geometry receipt",
                "global source actor scene upstream and instruction-template group",
                "full-video action-axis and preservation-axis human-review receipt",
                "manifest-v2 target qualification receipt",
            ],
            "provider_or_census_selection_self_qualifies_row": False,
            "target_eligible_N": 0,
            "train_ready_contribution": 0,
        },
        "split_and_leakage_requirements": {
            "group_key_fields": ["actor_identity", "scene_or_source_hash", "upstream_group", "action_family", "instruction_template_family"],
            "assign_group_before_target_generation": True,
            "train_calibration_promotion_locked_final_physical_roots_disjoint": True,
            "source_actor_scene_upstream_target_bytes_cross_split_overlap_allowed": False,
            "known_current_pool_overlaps_are_not_new_sources": sealed["known_cross_pool_source_group_overlaps"],
            "historically_exposed_sources_allowed_splits": ["train", "debug"],
            "promotion_rows_outside_D0_count": 500,
            "locked_final_rows_outside_D0_count": 500,
        },
        "teacher_disjoint_requirements": {
            "train_teacher_outputs_byte_disjoint_across_splits": True,
            "reserve_at_least_one_teacher_family_absent_from_train_for_promotion_diagnostic": True,
            "reserve_separate_teacher_family_absent_from_train_calibration_promotion_for_locked_final_subset": True,
            "teacher_identity_checkpoint_and_inference_config_byte_pinned": True,
            "independent_multi_teacher_consensus_requires_human_review": True,
            "teacher_family_or_checkpoint_does_not_self_qualify_target": True,
        },
        "qualification_pipeline": [
            "materialize source/instruction/candidate with byte provenance; default pending",
            "verify upstream license/right per source separately from dataset-card license",
            "decode exact 81-frame geometry and reject cuts/watermarks/artifacts",
            "freeze equivalence groups and split before any reviewer sees model outputs",
            "run independent action-axis and source-preservation-axis gates with abstain",
            "perform full-video human review and bind canonical qualification receipt",
            "validate accepted row through action_edit_sft_manifest_v2 and external qualification authority",
            "deduplicate then apply source and actor-scene caps before counting D0 effective N",
        ],
        "external_candidate_priority": [
            {
                "candidate_id": "insvie-1m-metadata-pinned",
                "role": "licensed_paired_candidate_pending_per_source_rights_and_bytes",
                "dataset_card_url": "https://huggingface.co/datasets/wyh6666/InsViE",
                "paper_url": "https://huggingface.co/papers/2503.20287",
                "pinned_repo_revision": INSVIE_REVISION,
                "dataset_card_license_label": "CC-BY-4.0",
                "dataset_card_row_count": INSVIE_CARD_ROW_COUNT,
                "metadata_materialized": True,
                "video_bytes_materialized": 0,
                "target_eligible_count": 0,
                "download_authorized": False,
                "archive_byte_count": INSVIE_CARD_ARCHIVE_BYTES,
                "source_zip_count": INSVIE_SOURCE_ZIP_COUNT,
                "edited_zip_count": INSVIE_EDITED_ZIP_COUNT,
                "rights_layers": {
                    "dataset_card_license_recorded": True,
                    "per_source_upstream_license_and_redistribution_rights": "UNKNOWN_REQUIRES_ROW_LEVEL_CLOSURE",
                    "card_license_proves_each_upstream_asset_rights": False,
                    "unknown_rights_use": "RESEARCH_ONLY_CANDIDATE_NOT_PUBLISHABLE_OR_TRAIN_READY",
                },
                "action_screen": {
                    "preliminary_candidate_count": sealed["insvie_external_metadata_candidate"]["preliminary_action_candidate_count"],
                    "preliminary_prefix_counts": sealed["insvie_external_metadata_candidate"]["preliminary_action_prefix_counts"],
                    "authoritative": False,
                    "eligible_count": 0,
                    "warning": "lexical matches include image operations, nouns, and other false positives",
                },
                "retrieval_plan": [
                    "pin every source/edited ZIP LFS SHA and size",
                    "range-read remote ZIP central directories without downloading media",
                    "map CSV selected rows to exact source/edited archive members",
                    "estimate action density per archive part before transfer",
                    "range-fetch only selected paired members when explicitly authorized",
                    "hash/decode/license-audit selected source and edited bytes before qualification",
                ],
            },
            {
                "candidate_id": "easyv2v-human-action-route",
                "role": "pipeline_inspiration_only",
                "downloadable_paired_asset_asserted": False,
                "current_materialized_rows": 0,
                "target_eligible_count": 0,
            },
            {
                "candidate_id": "dynaedit-source-conditioned-teacher",
                "role": "teacher_candidate_only",
                "current_materialized_rows": 0,
                "target_eligible_count": 0,
                "teacher_output_requires_independent_qualification": True,
            },
            {
                "candidate_id": "controlled-simulator-and-manual-pairs",
                "role": "future_simulator_gt_or_real_counterfactual_candidate",
                "current_materialized_rows": 0,
                "target_eligible_count": 0,
            },
        ],
        "required_outputs_before_d0": [
            "source_catalog_v2 with exposed flags and equivalence authority",
            "pending candidate manifest with no optimizer authority",
            "qualification authority and canonical receipts",
            "train-only action_edit_sft_manifest_v2 exact closure",
            "split/group leakage receipt",
            "decode/full-video human-review receipt",
            "exact sampler archive with effective N at least 2000",
        ],
        "status": "FUTURE_BUILD_PLAN_DATA_NOT_READY",
        "formal_training_authorized": False,
        "optimizer_launch_authorized": False,
    }
    return {**unsigned, "plan_digest": object_sha256(unsigned)}


def build_d0_plan(census: Mapping[str, Any]) -> Dict[str, Any]:
    sealed = validate_current_census(census)
    return validate_d0_plan(
        _assemble_d0_plan(sealed),
        expected_census_digest=sealed["census_digest"],
    )


def validate_d0_plan(value: Any, *, expected_census_digest: str) -> Dict[str, Any]:
    _require(isinstance(value, Mapping), "D0 plan must be an object")
    expected_keys = {
        "schema_version",
        "input_census_digest",
        "counts_are_future_requirements_not_existing_assets",
        "current_train_ready_N",
        "target_D0_train_eligible_effective_N",
        "current_gap",
        "future_row_matrix",
        "future_training_subset_counts",
        "future_target_truth_class_counts",
        "future_target_provenance_counts",
        "provenance_gates",
        "source_and_group_plan",
        "source_candidate_pool_inputs",
        "licensed_goku_candidate_materialization",
        "split_and_leakage_requirements",
        "teacher_disjoint_requirements",
        "qualification_pipeline",
        "external_candidate_priority",
        "required_outputs_before_d0",
        "status",
        "formal_training_authorized",
        "optimizer_launch_authorized",
        "plan_digest",
    }
    _require(set(value) == expected_keys, "D0 plan top-level field closure differs")
    row = dict(value)
    _require(row.get("schema_version") == D0_PLAN_SCHEMA, "D0 plan schema differs")
    digest = row.pop("plan_digest", None)
    _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None, "D0 plan digest invalid")
    _require(object_sha256(row) == digest, "D0 plan self-digest differs")
    _require(FROZEN_D0_PLAN_DIGEST != "PENDING_FREEZE", "D0 plan pin is not frozen")
    _require(digest == FROZEN_D0_PLAN_DIGEST, "D0 plan differs from frozen semantic template")
    _require(expected_census_digest == FROZEN_CENSUS_DIGEST, "expected census digest is not frozen")
    _require(value.get("input_census_digest") == expected_census_digest, "D0 plan census pin differs")
    _require(value.get("counts_are_future_requirements_not_existing_assets") is True, "D0 plan mislabels future counts")
    _require(value.get("current_train_ready_N") == 0, "D0 plan invents current rows")
    _require(
        value.get("target_D0_train_eligible_effective_N") == 2000
        and value.get("current_gap") == 2000,
        "D0 target or current gap differs",
    )
    matrix = value.get("future_row_matrix")
    expected_matrix = [
        {"training_subset": "general_edit", "semantic_truth_class": "licensed-paired", "target_provenance": "licensed-dataset", "future_required_rows": 300},
        {"training_subset": "general_edit", "semantic_truth_class": "real-counterfactual", "target_provenance": "real", "future_required_rows": 50},
        {"training_subset": "general_edit", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 50},
        {"training_subset": "general_edit", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 200},
        {"training_subset": "action_motion", "semantic_truth_class": "licensed-paired", "target_provenance": "licensed-dataset", "future_required_rows": 100},
        {"training_subset": "action_motion", "semantic_truth_class": "real-counterfactual", "target_provenance": "real", "future_required_rows": 50},
        {"training_subset": "action_motion", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 150},
        {"training_subset": "action_motion", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "action_motion", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "interaction_contact", "semantic_truth_class": "simulator-gt", "target_provenance": "simulator", "future_required_rows": 150},
        {"training_subset": "interaction_contact", "semantic_truth_class": "teacher-pseudo", "target_provenance": "teacher-pseudo", "future_required_rows": 100},
        {"training_subset": "interaction_contact", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 150},
        {"training_subset": "noop_preservation", "semantic_truth_class": "noop", "target_provenance": "real", "future_required_rows": 200},
        {"training_subset": "noop_preservation", "semantic_truth_class": "noop", "target_provenance": "licensed-dataset", "future_required_rows": 100},
        {"training_subset": "long_horizon", "semantic_truth_class": "continuation", "target_provenance": "teacher-pseudo", "future_required_rows": 100},
    ]
    _require(matrix == expected_matrix, "D0 frozen row matrix differs")
    _require(
        sum(item["future_required_rows"] for item in matrix) == 2000,
        "D0 row matrix total differs",
    )
    rebuilt_subsets: Dict[str, int] = {key: 0 for key in manifest_v2.TRAINING_SUBSETS}
    rebuilt_truth: Dict[str, int] = {
        key: 0 for key in manifest_v2.TARGET_SEMANTIC_TRUTH_CLASSES
    }
    rebuilt_provenance: Dict[str, int] = {
        key: 0 for key in manifest_v2.TARGET_PROVENANCE
    }
    for index, item in enumerate(matrix):
        _require(
            isinstance(item, Mapping)
            and set(item)
            == {
                "training_subset",
                "semantic_truth_class",
                "target_provenance",
                "future_required_rows",
            },
            "D0 matrix row {} field closure differs".format(index),
        )
        subset = item["training_subset"]
        truth_class = item["semantic_truth_class"]
        provenance_class = item["target_provenance"]
        count = item["future_required_rows"]
        _require(subset in rebuilt_subsets, "D0 matrix subset differs")
        _require(truth_class in rebuilt_truth, "D0 matrix truth class differs")
        _require(provenance_class in rebuilt_provenance, "D0 matrix provenance differs")
        _require(type(count) is int and count > 0, "D0 matrix count is not positive integer")
        rebuilt_subsets[subset] += count
        rebuilt_truth[truth_class] += count
        rebuilt_provenance[provenance_class] += count
    expected_subsets = {"general_edit": 600, "action_motion": 600, "interaction_contact": 400, "noop_preservation": 300, "long_horizon": 100}
    expected_truth = {"real-counterfactual": 100, "simulator-gt": 350, "licensed-paired": 400, "teacher-pseudo": 450, "continuation": 400, "noop": 300}
    expected_provenance = {"real": 300, "simulator": 350, "licensed-dataset": 500, "teacher-pseudo": 850}
    _require(value["future_training_subset_counts"] == expected_subsets, "D0 subset plan differs")
    provenance = value["future_target_provenance_counts"]
    truth = value["future_target_truth_class_counts"]
    _require(truth == expected_truth, "D0 target-truth plan differs")
    _require(provenance == expected_provenance, "D0 target-provenance plan differs")
    _require(rebuilt_subsets == value["future_training_subset_counts"], "D0 matrix/subset accounting differs")
    _require(rebuilt_truth == truth, "D0 matrix/truth accounting differs")
    _require(rebuilt_provenance == provenance, "D0 matrix/provenance accounting differs")
    _require(sum(provenance.values()) == 2000 and sum(truth.values()) == 2000, "D0 provenance/truth total differs")
    _require(
        value["provenance_gates"]
        == {
            "teacher_pseudo_max_fraction": 0.50,
            "teacher_pseudo_planned_fraction": 0.425,
            "single_teacher_max_rows": 300,
            "single_teacher_max_fraction": 0.15,
            "minimum_train_teacher_family_count_for_planned_teacher_rows": 3,
            "continuation_max_rows": 400,
            "continuation_max_fraction": 0.20,
            "noop_exact_rows": 300,
            "noop_exact_fraction": 0.15,
            "high_confidence_real_simulator_licensed_non_noop_min_rows": 700,
            "high_confidence_real_simulator_licensed_non_noop_planned_rows": 850,
        },
        "D0 provenance gates differ",
    )
    _require(
        value["source_and_group_plan"]
        == {
            "future_unique_source_goal": 1000,
            "future_rows_per_source_goal": 2,
            "hard_source_semantic_edit_cap": manifest_v2.SOURCE_SEMANTIC_EDIT_CAP,
            "hard_actor_scene_row_cap": manifest_v2.ACTOR_SCENE_ROW_CAP,
            "full644_recoverable_source_upper_bound_before_dedup": 644,
            "minimum_new_unexposed_sources_if_all_644_are_verified_unique": 356,
            "all_full644_sources_must_be_marked_exposed": True,
            "seed_transcode_copy_paraphrase_increase_effective_N": False,
            "equivalence_authority_frozen_before_split": True,
        },
        "D0 source/group plan differs",
    )
    expected_source_pool_input = {
        "pool_id": "goku_fullmotion_source_census_high_recall_16000",
        "role": "licensed_paired_source_candidate_pending_exact_pair_closure",
        "current_candidate_source_N": 16_000,
        "current_endpoint_presence_pair_N": 16_000,
        "current_source_catalog_v2_eligible_N": 0,
        "current_target_eligible_N": 0,
        "current_train_ready_contribution": 0,
        "production_eligible": False,
        "download_authorized_by_this_plan": False,
        "required_before_use": [
            "seal exact source/edited member bytes and per-row pair provenance",
            "freeze source equivalence and historical optimizer exposure",
            "verify CC-BY-NC-4.0 non-commercial research scope and attribution",
            "run action-axis, preservation-axis, decode, and full-video human review",
            "assign globally disjoint split and bind qualification receipts",
        ],
    }
    _require(
        value["source_candidate_pool_inputs"] == [expected_source_pool_input],
        "D0 source-candidate pool input differs",
    )
    licensed_goku = value["licensed_goku_candidate_materialization"]
    _require(
        licensed_goku["counts_are_future_requirements_not_existing_assets"] is True
        and licensed_goku["current_materialized_and_qualified_N"] == 0
        and licensed_goku["future_required_total_N"] == 500
        and licensed_goku["provider_or_census_selection_self_qualifies_row"] is False
        and licensed_goku["target_eligible_N"] == 0
        and licensed_goku["train_ready_contribution"] == 0,
        "licensed Goku materialization authority differs",
    )
    _require(
        licensed_goku["future_rows"]
        == [
            {
                "route": "goku_subject_movement_general_endpoint_pair",
                "source_queue": "upstream_combined_approximately_222k",
                "training_subset": "general_edit",
                "semantic_truth_class": "licensed-paired",
                "target_provenance": "licensed-dataset",
                "future_required_rows": 300,
            },
            {
                "route": "goku_subject_movement_strict_continuous_action_pair",
                "source_queue": "source_census_final_strict_1676",
                "training_subset": "action_motion",
                "semantic_truth_class": "licensed-paired",
                "target_provenance": "licensed-dataset",
                "future_required_rows": 100,
            },
            {
                "route": "independent_source_exact_noop",
                "source_queue": "independent_source_bytes_after_global_dedup",
                "training_subset": "noop_preservation",
                "semantic_truth_class": "noop",
                "target_provenance": "licensed-dataset",
                "future_required_rows": 100,
            },
        ]
        and sum(item["future_required_rows"] for item in licensed_goku["future_rows"])
        == licensed_goku["future_required_total_N"],
        "licensed Goku future row allocation differs",
    )
    split = value["split_and_leakage_requirements"]
    _require(
        split["group_key_fields"]
        == ["actor_identity", "scene_or_source_hash", "upstream_group", "action_family", "instruction_template_family"]
        and split["assign_group_before_target_generation"] is True
        and split["train_calibration_promotion_locked_final_physical_roots_disjoint"] is True
        and split["source_actor_scene_upstream_target_bytes_cross_split_overlap_allowed"] is False
        and split["historically_exposed_sources_allowed_splits"] == ["train", "debug"]
        and split["promotion_rows_outside_D0_count"] == 500
        and split["locked_final_rows_outside_D0_count"] == 500,
        "D0 split/group leakage contract differs",
    )
    _require(
        split["known_current_pool_overlaps_are_not_new_sources"]
        == [
            {
                "source_group_id": iid,
                "pool_ids": ["native_core4_rv2v_proposals", "quotient_fitted_unseen_anchor8"],
            }
            for iid in (
                "7b88a1ca1f804f41",
                "841b5e0080a1441d",
                "a35b590961d24694",
                "a66e6818e4144928",
            )
        ],
        "D0 known overlap closure differs",
    )
    _require(
        value["teacher_disjoint_requirements"]
        == {
            "train_teacher_outputs_byte_disjoint_across_splits": True,
            "reserve_at_least_one_teacher_family_absent_from_train_for_promotion_diagnostic": True,
            "reserve_separate_teacher_family_absent_from_train_calibration_promotion_for_locked_final_subset": True,
            "teacher_identity_checkpoint_and_inference_config_byte_pinned": True,
            "independent_multi_teacher_consensus_requires_human_review": True,
            "teacher_family_or_checkpoint_does_not_self_qualify_target": True,
        },
        "D0 teacher-disjoint contract differs",
    )
    external = value["external_candidate_priority"]
    _require(
        isinstance(external, list)
        and [item.get("candidate_id") for item in external]
        == [
            "insvie-1m-metadata-pinned",
            "easyv2v-human-action-route",
            "dynaedit-source-conditioned-teacher",
            "controlled-simulator-and-manual-pairs",
        ],
        "D0 external-candidate role closure differs",
    )
    insvie = external[0]
    _require(
        insvie["role"] == "licensed_paired_candidate_pending_per_source_rights_and_bytes"
        and insvie["pinned_repo_revision"] == INSVIE_REVISION
        and insvie["dataset_card_license_label"] == "CC-BY-4.0"
        and insvie["dataset_card_row_count"] == INSVIE_CARD_ROW_COUNT
        and insvie["metadata_materialized"] is True
        and insvie["video_bytes_materialized"] == 0
        and insvie["target_eligible_count"] == 0
        and insvie["download_authorized"] is False
        and insvie["archive_byte_count"] == INSVIE_CARD_ARCHIVE_BYTES
        and insvie["source_zip_count"] == INSVIE_SOURCE_ZIP_COUNT
        and insvie["edited_zip_count"] == INSVIE_EDITED_ZIP_COUNT
        and insvie["rights_layers"]["card_license_proves_each_upstream_asset_rights"] is False
        and insvie["action_screen"]
        == {
            "preliminary_candidate_count": INSVIE_PRELIMINARY_ACTION_COUNT,
            "preliminary_prefix_counts": INSVIE_PRELIMINARY_PREFIX_COUNTS,
            "authoritative": False,
            "eligible_count": 0,
            "warning": "lexical matches include image operations, nouns, and other false positives",
        },
        "D0 InsViE candidate authority differs",
    )
    _require(
        external[1]["role"] == "pipeline_inspiration_only"
        and external[1]["downloadable_paired_asset_asserted"] is False
        and external[1]["target_eligible_count"] == 0
        and external[2]["role"] == "teacher_candidate_only"
        and external[2]["target_eligible_count"] == 0
        and external[2]["teacher_output_requires_independent_qualification"] is True
        and external[3]["role"] == "future_simulator_gt_or_real_counterfactual_candidate"
        and external[3]["target_eligible_count"] == 0,
        "D0 EasyV2V/DynaEdit/simulator candidate semantics differ",
    )
    _require(
        value["status"] == "FUTURE_BUILD_PLAN_DATA_NOT_READY"
        and value["formal_training_authorized"] is False
        and value["optimizer_launch_authorized"] is False,
        "D0 plan grants optimizer authority or changes status",
    )
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        _require(path.read_bytes() == payload, "refusing to overwrite differing output {}".format(path))
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, 0o444)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "short write while sealing {}".format(path))
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--insvie-metadata-csv", type=Path, required=True)
    parser.add_argument("--census-output", type=Path, required=True)
    parser.add_argument("--plan-output", type=Path, required=True)
    args = parser.parse_args(argv)
    census = build_current_census(args.repo_root, insvie_metadata_csv=args.insvie_metadata_csv)
    plan = build_d0_plan(census)
    _write_json(args.census_output, census)
    _write_json(args.plan_output, plan)
    print(json.dumps({
        "status": census["status"],
        "train_ready_N": census["train_ready_N"],
        "d0_gap": census["d0_gap"],
        "census_digest": census["census_digest"],
        "plan_digest": plan["plan_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CENSUS_SCHEMA",
    "CandidateCensusError",
    "D0_PLAN_SCHEMA",
    "INSVIE_PRELIMINARY_ACTION_PATTERN",
    "audit_insvie_metadata",
    "build_current_census",
    "build_d0_plan",
    "classify_candidate_rows",
    "object_sha256",
    "validate_current_census",
    "validate_d0_plan",
]

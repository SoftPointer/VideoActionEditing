"""Select exactly 512 pending rows from one scale-512 Goku finalizer.

This is an authorization-free, fail-closed bridge.  It verifies a completed
``scale512`` finalizer, copies generation rows byte-for-byte, and publishes an
exact-512 manifest plus a provenance receipt.  It never invokes Wan, signs a
release, or changes authorization semantics.

An optional prior exact-eight manifest can be retained.  When supplied, it
must contain exactly eight canonical, unique pending v9 rows.  Those eight
rows are kept byte-for-byte and in their original order; the remaining 504
rows are filled by ascending unique finalizer ``review_rank``.  A finalizer
row with the same IID and group is skipped as already retained.  Partial
IID/group collisions are rejected as conflicting identities.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from . import wan22_select_exact8 as _strict


FINALIZER_REVIEW_SCHEMA = (
    "motive-goku-action-anchor-final-row-scale512-v1"
)
FINALIZER_GENERATION_SCHEMA = (
    "motive-goku-action-anchor-generation-scale512-v1"
)
FINALIZER_SUMMARY_SCHEMA = (
    "motive-goku-action-anchor-finalize-scale512-v1"
)
FINALIZER_DONE_SCHEMA = (
    "motive-goku-action-anchor-finalize-done-scale512-v1"
)
FINALIZER_POLICY_VERSION = (
    "goku-action-anchor-strict-continuity-scale512-v1"
)
FINALIZER_PROFILE_SCHEMA = (
    "motive-goku-action-anchor-finalization-profile-v1"
)
FINALIZER_PROFILE_NAME = "scale512"
RETAINED_GENERATION_SCHEMA = "motive-goku-action-anchor-generation-v9"

RECEIPT_SCHEMA = "motive-wan22-exact512-selection-receipt-v1"
RANK_ONLY_POLICY = "lowest_finalizer_review_rank_exact512"
RETAIN_EXACT8_POLICY = (
    "retain_exact8_then_lowest_finalizer_review_rank_exact512"
)
SELECTED_ROW_COUNT = 512
RETAINED_ROW_COUNT = 8

REVIEW_NAME = "review_candidates.jsonl"
PROPOSED_NAME = "proposed_512.jsonl"
RESERVE_NAME = "reserve_128.jsonl"
PARENT_GENERATION_NAME = "generation_manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"

OUTPUT_MANIFEST_NAME = "generation_manifest.jsonl"
OUTPUT_RECEIPT_NAME = "selection_receipt.json"

_FINALIZER_HASHED_OUTPUTS = (
    REVIEW_NAME,
    PROPOSED_NAME,
    RESERVE_NAME,
    PARENT_GENERATION_NAME,
    SUMMARY_NAME,
)
_SUMMARY_HASHED_OUTPUTS = (
    REVIEW_NAME,
    PROPOSED_NAME,
    RESERVE_NAME,
    PARENT_GENERATION_NAME,
)

_PENDING_SEMANTICS = {
    "manifest_role": "review_proposal",
    "production_eligible": False,
    "human_review_status": "pending",
    "generation_authorized": False,
    "approval": None,
    "authorization_interface_available": False,
}
_INSTRUCTION_CONTRACT = {
    "sole_candidate_instruction_field": "edit_instruction",
    "candidate_instruction_source": "frozen_selected_prompt",
    "writer_proposal_payload_included": False,
    "writer_proposals_executable": False,
    "requires_future_signed_release_verifier": True,
}


class Wan22Exact512SelectionError(RuntimeError):
    """The scale-512 parent or requested exact-512 output is invalid."""


_CanonicalRow = _strict._CanonicalRow


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_read(path: Path, *, context: str) -> bytes:
    try:
        return _strict._stable_read(path, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _load_json_object(raw: bytes, *, context: str) -> dict[str, Any]:
    try:
        return _strict._load_json_object(raw, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _load_canonical_jsonl(
    raw: bytes,
    *,
    context: str,
) -> list[_CanonicalRow]:
    try:
        return _strict._load_canonical_jsonl(raw, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _canonical_bytes(value: Any) -> bytes:
    try:
        return _strict._canonical_bytes(value)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Wan22Exact512SelectionError(f"{context} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise Wan22Exact512SelectionError(
            f"{context} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _string(value: Any, *, context: str) -> str:
    try:
        return _strict._string(value, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _iid(value: Any, *, context: str) -> str:
    try:
        return _strict._iid(value, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _digest(value: Any, *, context: str) -> str:
    try:
        return _strict._digest(value, context=context)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def _source_digest(path: Path, *, context: str) -> str:
    unresolved = Path(os.path.abspath(path))
    if unresolved.is_symlink():
        raise Wan22Exact512SelectionError(
            f"{context} must not be a symlink: {unresolved}"
        )
    return _sha256(_stable_read(unresolved, context=context))


def _validate_profile(
    summary: Mapping[str, Any],
    done: Mapping[str, Any],
) -> None:
    """Validate the scale512 marker shared by summary and done.

    The finalizer owns the detailed profile configuration.  The selector
    requires a self-hashed canonical profile object, exact agreement between
    summary and done, and the immutable profile name.  This binds future
    additive configuration fields without silently accepting a different
    profile.
    """

    summary_profile = _mapping(
        summary.get("profile"), context="summary.json.profile"
    )
    done_profile = _mapping(done.get("profile"), context="done.json.profile")
    if dict(summary_profile) != dict(done_profile):
        raise Wan22Exact512SelectionError(
            "done.json profile differs from summary.json profile"
        )
    _exact_keys(
        summary_profile,
        {"schema_version", "name", "config", "config_sha256"},
        context="summary.json.profile",
    )
    if summary_profile.get("schema_version") != FINALIZER_PROFILE_SCHEMA:
        raise Wan22Exact512SelectionError("finalizer profile schema differs")
    if summary_profile.get("name") != FINALIZER_PROFILE_NAME:
        raise Wan22Exact512SelectionError("finalizer profile is not scale512")
    config = _mapping(
        summary_profile.get("config"),
        context="summary.json.profile.config",
    )
    config_sha = _digest(
        summary_profile.get("config_sha256"),
        context="summary.json.profile.config_sha256",
    )
    if config_sha != _sha256(_canonical_bytes(dict(config))):
        raise Wan22Exact512SelectionError("profile config SHA differs")
    expected_config: dict[str, Any] = {
        "required_qwen_shard_count": 8,
        "review_limit": 768,
        "proposed_size": 512,
        "reserve_size": 128,
        "max_per_target_verb": 48,
        "category_quotas": {
            "locomotion": 128,
            "posture": 128,
            "interaction": 192,
            "articulated": 64,
        },
        "artifacts": {
            "review": REVIEW_NAME,
            "proposed": PROPOSED_NAME,
            "reserve": RESERVE_NAME,
            "generation": PARENT_GENERATION_NAME,
            "summary": SUMMARY_NAME,
            "done": DONE_NAME,
        },
        "schemas": {
            "row": FINALIZER_REVIEW_SCHEMA,
            "generation": FINALIZER_GENERATION_SCHEMA,
            "summary": FINALIZER_SUMMARY_SCHEMA,
            "done": FINALIZER_DONE_SCHEMA,
        },
        "policy_version": FINALIZER_POLICY_VERSION,
    }
    if dict(config) != expected_config:
        raise Wan22Exact512SelectionError(
            "finalizer scale512 profile config differs"
        )
    profile_sha = _digest(
        done.get("profile_sha256"),
        context="done.json.profile_sha256",
    )
    if profile_sha != _sha256(_canonical_bytes(dict(summary_profile))):
        raise Wan22Exact512SelectionError("done.json profile SHA differs")


def _validate_parent_hashes(
    *,
    finalizer_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes], bytes, str]:
    done_raw = _stable_read(finalizer_dir / DONE_NAME, context=DONE_NAME)
    done = _load_json_object(done_raw, context=DONE_NAME)
    _exact_keys(
        done,
        {
            "schema_version",
            "status",
            "profile",
            "profile_sha256",
            "summary_sha256",
            "implementation_sha256",
            "output_sha256",
        },
        context=DONE_NAME,
    )
    if done.get("schema_version") != FINALIZER_DONE_SCHEMA:
        raise Wan22Exact512SelectionError("done.json schema differs")
    if done.get("status") != "complete":
        raise Wan22Exact512SelectionError("done.json status is not complete")

    done_outputs = _mapping(
        done.get("output_sha256"), context="done.json.output_sha256"
    )
    _exact_keys(
        done_outputs,
        set(_FINALIZER_HASHED_OUTPUTS),
        context="done.json.output_sha256",
    )
    raw_outputs = {
        name: _stable_read(
            finalizer_dir / name,
            context=f"finalizer output {name}",
        )
        for name in _FINALIZER_HASHED_OUTPUTS
    }
    for name, raw in raw_outputs.items():
        expected = _digest(
            done_outputs.get(name),
            context=f"done.json.output_sha256[{name!r}]",
        )
        if _sha256(raw) != expected:
            raise Wan22Exact512SelectionError(
                f"done.json hash differs for {name}"
            )

    summary_raw = raw_outputs[SUMMARY_NAME]
    if _digest(
        done.get("summary_sha256"), context="done.json.summary_sha256"
    ) != _sha256(summary_raw):
        raise Wan22Exact512SelectionError("done.json summary_sha256 differs")
    summary = _load_json_object(summary_raw, context=SUMMARY_NAME)
    _exact_keys(
        summary,
        {
            "schema_version",
            "policy_version",
            "profile",
            "seed",
            "input",
            "hard_gate",
            "diversity",
            "selection",
            "semantics",
            "implementation_sha256",
            "output_sha256",
        },
        context=SUMMARY_NAME,
    )
    if summary.get("schema_version") != FINALIZER_SUMMARY_SCHEMA:
        raise Wan22Exact512SelectionError("summary.json schema differs")
    if summary.get("policy_version") != FINALIZER_POLICY_VERSION:
        raise Wan22Exact512SelectionError("summary.json policy differs")
    _validate_profile(summary, done)

    implementation_path = Path(__file__).resolve(strict=True).with_name(
        "goku_action_anchor_finalize.py"
    )
    implementation_sha = _source_digest(
        implementation_path,
        context="finalizer implementation sibling",
    )
    done_implementation = _digest(
        done.get("implementation_sha256"),
        context="done.json.implementation_sha256",
    )
    summary_implementation = _digest(
        summary.get("implementation_sha256"),
        context="summary.json.implementation_sha256",
    )
    if (
        done_implementation != implementation_sha
        or summary_implementation != implementation_sha
    ):
        raise Wan22Exact512SelectionError(
            "recorded finalizer implementation does not match sibling source"
        )

    summary_outputs = _mapping(
        summary.get("output_sha256"),
        context="summary.json.output_sha256",
    )
    _exact_keys(
        summary_outputs,
        set(_SUMMARY_HASHED_OUTPUTS),
        context="summary.json.output_sha256",
    )
    for name in _SUMMARY_HASHED_OUTPUTS:
        expected = _digest(
            summary_outputs.get(name),
            context=f"summary.json.output_sha256[{name!r}]",
        )
        actual = _sha256(raw_outputs[name])
        if expected != actual or done_outputs.get(name) != expected:
            raise Wan22Exact512SelectionError(
                f"summary.json hash differs for {name}"
            )

    semantics = _mapping(
        summary.get("semantics"), context="summary.json.semantics"
    )
    if dict(semantics) != {
        **_PENDING_SEMANTICS,
        "human_labels_asserted": False,
    }:
        raise Wan22Exact512SelectionError(
            "summary.json does not assert exact pending semantics"
        )
    return done, summary, raw_outputs, done_raw, implementation_sha


def _validate_review_rows(
    rows: Sequence[_CanonicalRow],
    *,
    expected_profile: Mapping[str, Any],
) -> dict[str, tuple[_CanonicalRow, int]]:
    by_iid: dict[str, tuple[_CanonicalRow, int]] = {}
    groups: set[str] = set()
    ranks: set[int] = set()
    for record in rows:
        row = record.value
        context = f"{REVIEW_NAME}:{record.line_number}"
        iid = _iid(row.get("iid"), context=f"{context}.iid")
        group = _string(row.get("group_id"), context=f"{context}.group_id")
        _string(row.get("prompt"), context=f"{context}.prompt")
        if iid in by_iid:
            raise Wan22Exact512SelectionError(f"duplicate review iid: {iid}")
        if group in groups:
            raise Wan22Exact512SelectionError(
                f"duplicate review group_id: {group}"
            )
        finalization = _mapping(
            row.get("action_anchor_finalization"),
            context=f"{context}.action_anchor_finalization",
        )
        if finalization.get("schema_version") != FINALIZER_REVIEW_SCHEMA:
            raise Wan22Exact512SelectionError(
                f"{context} finalization schema differs"
            )
        if finalization.get("policy_version") != FINALIZER_POLICY_VERSION:
            raise Wan22Exact512SelectionError(
                f"{context} finalization policy differs"
            )
        profile = _mapping(
            finalization.get("profile"),
            context=f"{context}.action_anchor_finalization.profile",
        )
        if dict(profile) != dict(expected_profile):
            raise Wan22Exact512SelectionError(
                f"{context} finalization profile differs"
            )
        if finalization.get("hard_gate_passed") is not True:
            raise Wan22Exact512SelectionError(
                f"{context} did not pass the hard gate"
            )
        failures = finalization.get("hard_gate_failures")
        if not isinstance(failures, list) or failures:
            raise Wan22Exact512SelectionError(
                f"{context} has hard-gate failures"
            )
        rank = finalization.get("review_rank")
        if type(rank) is not int or rank <= 0:
            raise Wan22Exact512SelectionError(
                f"{context} review_rank must be a positive integer"
            )
        if rank in ranks:
            raise Wan22Exact512SelectionError(
                f"duplicate review_rank: {rank}"
            )
        if finalization.get("selection_bucket") not in {
            "proposed",
            "reserve",
            "review_only",
        }:
            raise Wan22Exact512SelectionError(
                f"{context} has an invalid selection bucket"
            )
        for field, expected in _PENDING_SEMANTICS.items():
            if finalization.get(field) != expected:
                raise Wan22Exact512SelectionError(
                    f"{context}.{field} is not the exact pending value"
                )
        if finalization.get("human_label") is not False:
            raise Wan22Exact512SelectionError(
                f"{context} unexpectedly asserts a human label"
            )
        by_iid[iid] = (record, rank)
        groups.add(group)
        ranks.add(rank)
    return by_iid


def _validate_generation_record(
    record: _CanonicalRow,
    *,
    context: str,
    expected_schema: str,
    expected_profile: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    row = record.value
    if row.get("schema_version") != expected_schema:
        raise Wan22Exact512SelectionError(f"{context} schema differs")
    if expected_profile is None:
        if "finalization_profile" in row or "policy_version" in row:
            raise Wan22Exact512SelectionError(
                f"{context} contains unexpected scale512 profile metadata"
            )
    else:
        if row.get("policy_version") != FINALIZER_POLICY_VERSION:
            raise Wan22Exact512SelectionError(f"{context} policy differs")
        profile = _mapping(
            row.get("finalization_profile"),
            context=f"{context}.finalization_profile",
        )
        if dict(profile) != dict(expected_profile):
            raise Wan22Exact512SelectionError(
                f"{context} finalization profile differs"
            )
    iid = _iid(row.get("iid"), context=f"{context}.iid")
    group = _string(row.get("group_id"), context=f"{context}.group_id")
    for field, expected in _PENDING_SEMANTICS.items():
        if row.get(field) != expected:
            raise Wan22Exact512SelectionError(
                f"{context}.{field} is not the exact pending v9 value"
            )
    if row.get("action_change_substantive") != "yes":
        raise Wan22Exact512SelectionError(
            f"{context} action change is not substantive"
        )
    if "_authorization_mode" in row:
        raise Wan22Exact512SelectionError(
            f"{context} contains a forbidden authorization marker"
        )
    if (
        "absolute_target_prompt" in row
        or "writer_absolute_target_prompt" in row
    ):
        raise Wan22Exact512SelectionError(
            f"{context} contains a forbidden writer prompt"
        )
    instruction = _string(
        row.get("edit_instruction"),
        context=f"{context}.edit_instruction",
    )
    instruction_sha = _digest(
        row.get("edit_instruction_sha256"),
        context=f"{context}.edit_instruction_sha256",
    )
    if instruction_sha != _sha256(instruction.encode("utf-8")):
        raise Wan22Exact512SelectionError(
            f"{context} edit_instruction SHA differs"
        )
    if row.get("source_instruction_provenance") != instruction:
        raise Wan22Exact512SelectionError(
            f"{context} instruction provenance differs"
        )
    if row.get("source_edited_caption_provenance_role") != (
        "non_executable_provenance"
    ):
        raise Wan22Exact512SelectionError(
            f"{context} edited-caption provenance is executable"
        )
    contract = _mapping(
        row.get("instruction_contract"),
        context=f"{context}.instruction_contract",
    )
    if dict(contract) != _INSTRUCTION_CONTRACT:
        raise Wan22Exact512SelectionError(
            f"{context} instruction contract differs"
        )
    return iid, group, instruction


def _validate_generation_rows(
    rows: Sequence[_CanonicalRow],
    *,
    review_by_iid: Mapping[str, tuple[_CanonicalRow, int]],
    expected_profile: Mapping[str, Any],
) -> list[tuple[int, _CanonicalRow, str, str]]:
    ranked: list[tuple[int, _CanonicalRow, str, str]] = []
    iids: set[str] = set()
    groups: set[str] = set()
    for record in rows:
        context = f"{PARENT_GENERATION_NAME}:{record.line_number}"
        iid, group, instruction = _validate_generation_record(
            record,
            context=context,
            expected_schema=FINALIZER_GENERATION_SCHEMA,
            expected_profile=expected_profile,
        )
        if iid in iids:
            raise Wan22Exact512SelectionError(
                f"duplicate generation iid: {iid}"
            )
        if group in groups:
            raise Wan22Exact512SelectionError(
                f"duplicate generation group_id: {group}"
            )
        match = review_by_iid.get(iid)
        if match is None:
            raise Wan22Exact512SelectionError(
                f"{context} has no matching review row"
            )
        review_record, rank = match
        review = review_record.value
        finalization = _mapping(
            review.get("action_anchor_finalization"),
            context=f"review iid={iid} finalization",
        )
        if finalization.get("selection_bucket") != "proposed":
            raise Wan22Exact512SelectionError(
                f"generation iid={iid} is not in the proposed bucket"
            )
        if review.get("group_id") != group:
            raise Wan22Exact512SelectionError(
                f"generation iid={iid} group differs from review"
            )
        if review.get("prompt") != instruction:
            raise Wan22Exact512SelectionError(
                f"generation iid={iid} instruction differs from frozen prompt"
            )
        ranked.append((rank, record, iid, group))
        iids.add(iid)
        groups.add(group)
    if len(ranked) != SELECTED_ROW_COUNT:
        raise Wan22Exact512SelectionError(
            "scale512 finalizer must contain exactly 512 valid proposed "
            f"generation rows, found {len(ranked)}"
        )
    return ranked


def _validate_summary_counts(
    summary: Mapping[str, Any],
    *,
    review_count: int,
    generation_count: int,
) -> None:
    if review_count < 640 or review_count > 768:
        raise Wan22Exact512SelectionError(
            "scale512 review row count must be between 640 and 768"
        )
    selection = _mapping(
        summary.get("selection"), context="summary.json.selection"
    )
    _exact_keys(
        selection,
        {
            "mode",
            "allow_partial",
            "requested_proposed_rows",
            "requested_reserve_rows",
            "effective_proposed_target",
            "effective_reserve_target",
            "review_rows",
            "proposed_rows",
            "reserve_rows",
            "generation_rows",
            "requested_category_quotas",
            "effective_category_quotas",
            "proposed_category_counts",
            "quota_shortfall_before_backfill",
            "review_category_counts",
            "reserve_category_counts",
            "proposed_target_verb_counts",
            "proposal_reserve_disjoint",
        },
        context="summary.json.selection",
    )
    required = {
        "mode": "strict_512_plus_128",
        "requested_proposed_rows": SELECTED_ROW_COUNT,
        "effective_proposed_target": SELECTED_ROW_COUNT,
        "proposed_rows": generation_count,
        "generation_rows": generation_count,
        "review_rows": review_count,
        "requested_reserve_rows": 128,
        "effective_reserve_target": 128,
        "reserve_rows": 128,
        "allow_partial": False,
        "proposal_reserve_disjoint": True,
    }
    for field, expected in required.items():
        if (
            selection.get(field) != expected
            or type(selection.get(field)) is not type(expected)
        ):
            raise Wan22Exact512SelectionError(
                f"summary.json.selection.{field} differs"
            )
    expected_quotas = {
        "locomotion": 128,
        "posture": 128,
        "interaction": 192,
        "articulated": 64,
    }
    for field in (
        "requested_category_quotas",
        "effective_category_quotas",
    ):
        quotas = _mapping(
            selection.get(field), context=f"summary.json.selection.{field}"
        )
        if dict(quotas) != expected_quotas:
            raise Wan22Exact512SelectionError(
                f"summary.json.selection.{field} differs"
            )


def _validate_bucket_artifacts(
    *,
    proposed_rows: Sequence[_CanonicalRow],
    reserve_rows: Sequence[_CanonicalRow],
    review_by_iid: Mapping[str, tuple[_CanonicalRow, int]],
    generation_rows: Sequence[_CanonicalRow],
) -> None:
    if len(proposed_rows) != 512:
        raise Wan22Exact512SelectionError(
            "proposed_512.jsonl must contain exactly 512 rows"
        )
    if len(reserve_rows) != 128:
        raise Wan22Exact512SelectionError(
            "reserve_128.jsonl must contain exactly 128 rows"
        )

    artifact_iids: dict[str, set[str]] = {
        PROPOSED_NAME: set(),
        RESERVE_NAME: set(),
    }
    artifact_groups: dict[str, set[str]] = {
        PROPOSED_NAME: set(),
        RESERVE_NAME: set(),
    }
    proposed_iids_in_order: list[str] = []
    for name, rows, expected_bucket in (
        (PROPOSED_NAME, proposed_rows, "proposed"),
        (RESERVE_NAME, reserve_rows, "reserve"),
    ):
        for record in rows:
            context = f"{name}:{record.line_number}"
            iid = _iid(record.value.get("iid"), context=f"{context}.iid")
            group = _string(
                record.value.get("group_id"), context=f"{context}.group_id"
            )
            if iid in artifact_iids[name]:
                raise Wan22Exact512SelectionError(
                    f"duplicate {name} iid: {iid}"
                )
            if group in artifact_groups[name]:
                raise Wan22Exact512SelectionError(
                    f"duplicate {name} group_id: {group}"
                )
            match = review_by_iid.get(iid)
            if match is None or match[0].value != record.value:
                raise Wan22Exact512SelectionError(
                    f"{context} differs from its bound review row"
                )
            finalization = _mapping(
                record.value.get("action_anchor_finalization"),
                context=f"{context}.action_anchor_finalization",
            )
            if finalization.get("selection_bucket") != expected_bucket:
                raise Wan22Exact512SelectionError(
                    f"{context} selection bucket differs"
                )
            artifact_iids[name].add(iid)
            artifact_groups[name].add(group)
            if name == PROPOSED_NAME:
                proposed_iids_in_order.append(iid)

    if artifact_iids[PROPOSED_NAME] & artifact_iids[RESERVE_NAME]:
        raise Wan22Exact512SelectionError(
            "proposed and reserve IID sets are not disjoint"
        )
    if artifact_groups[PROPOSED_NAME] & artifact_groups[RESERVE_NAME]:
        raise Wan22Exact512SelectionError(
            "proposed and reserve group sets are not disjoint"
        )
    generation_iids = [
        _iid(
            record.value.get("iid"),
            context=f"{PARENT_GENERATION_NAME}:{record.line_number}.iid",
        )
        for record in generation_rows
    ]
    if generation_iids != proposed_iids_in_order:
        raise Wan22Exact512SelectionError(
            "generation row order differs from proposed_512.jsonl"
        )


def _load_retained_exact8(
    path: Path,
) -> tuple[list[_CanonicalRow], bytes, Path]:
    unresolved = Path(os.path.abspath(path.expanduser()))
    if unresolved.is_symlink() or not unresolved.is_file():
        raise Wan22Exact512SelectionError(
            "retain_exact8_manifest must be a non-symlink regular file"
        )
    resolved = unresolved.resolve(strict=True)
    raw = _stable_read(resolved, context="retained exact8 manifest")
    rows = _load_canonical_jsonl(raw, context="retained exact8 manifest")
    if len(rows) != RETAINED_ROW_COUNT:
        raise Wan22Exact512SelectionError(
            "retained exact8 manifest must contain exactly eight rows"
        )
    iids: set[str] = set()
    groups: set[str] = set()
    for record in rows:
        context = f"retained exact8 manifest:{record.line_number}"
        iid, group, _instruction = _validate_generation_record(
            record,
            context=context,
            expected_schema=RETAINED_GENERATION_SCHEMA,
            expected_profile=None,
        )
        if iid in iids:
            raise Wan22Exact512SelectionError(
                f"duplicate retained exact8 iid: {iid}"
            )
        if group in groups:
            raise Wan22Exact512SelectionError(
                f"duplicate retained exact8 group_id: {group}"
            )
        iids.add(iid)
        groups.add(group)
    return rows, raw, resolved


def _select_rows(
    ranked: Sequence[tuple[int, _CanonicalRow, str, str]],
    retained: Sequence[_CanonicalRow],
) -> tuple[list[_CanonicalRow], list[int], list[str], list[str]]:
    retained_iids: dict[str, str] = {}
    retained_groups: dict[str, str] = {}
    for record in retained:
        iid, group, _instruction = _validate_generation_record(
            record,
            context="validated retained exact8 row",
            expected_schema=RETAINED_GENERATION_SCHEMA,
            expected_profile=None,
        )
        retained_iids[iid] = group
        retained_groups[group] = iid

    # Detect every cross-artifact identity conflict before rank truncation;
    # a high-rank conflicting row must not hide outside the selected prefix.
    for _rank, _record, iid, group in ranked:
        retained_group = retained_iids.get(iid)
        retained_iid = retained_groups.get(group)
        if retained_group is None and retained_iid is None:
            continue
        if retained_group != group or retained_iid != iid:
            raise Wan22Exact512SelectionError(
                "retained/finalizer IID-group identity conflict: "
                f"iid={iid!r} group_id={group!r}"
            )

    selected = list(retained)
    fill_ranks: list[int] = []
    fill_iids: list[str] = []
    fill_groups: list[str] = []
    needed = SELECTED_ROW_COUNT - len(selected)
    for rank, record, iid, group in sorted(ranked, key=lambda item: item[0]):
        retained_group = retained_iids.get(iid)
        retained_iid = retained_groups.get(group)
        if retained_group is not None or retained_iid is not None:
            if retained_group == group and retained_iid == iid:
                continue
            raise Wan22Exact512SelectionError(
                "retained/finalizer IID-group identity conflict: "
                f"iid={iid!r} group_id={group!r}"
            )
        selected.append(record)
        fill_ranks.append(rank)
        fill_iids.append(iid)
        fill_groups.append(group)
        if len(fill_ranks) == needed:
            break
    if len(selected) != SELECTED_ROW_COUNT:
        raise Wan22Exact512SelectionError(
            "fewer than 512 unique rows remain after exact8 retention"
        )
    return selected, fill_ranks, fill_iids, fill_groups


def _write_new(path: Path, payload: bytes) -> None:
    try:
        _strict._write_new(path, payload)
    except _strict.Wan22Exact8SelectionError as error:
        raise Wan22Exact512SelectionError(str(error)) from error


def select_exact512(
    *,
    finalizer_dir: str | Path,
    output_dir: str | Path,
    retain_exact8_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Verify inputs and atomically publish one exact-512 subset."""

    parent_unresolved = Path(os.path.abspath(Path(finalizer_dir).expanduser()))
    if parent_unresolved.is_symlink() or not parent_unresolved.is_dir():
        raise Wan22Exact512SelectionError(
            "finalizer_dir must be a non-symlink directory: "
            f"{parent_unresolved}"
        )
    parent = parent_unresolved.resolve(strict=True)
    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    (
        _done,
        summary,
        raw_outputs,
        done_raw,
        finalizer_implementation_sha,
    ) = _validate_parent_hashes(finalizer_dir=parent)
    review_rows = _load_canonical_jsonl(
        raw_outputs[REVIEW_NAME], context=REVIEW_NAME
    )
    generation_rows = _load_canonical_jsonl(
        raw_outputs[PARENT_GENERATION_NAME],
        context=PARENT_GENERATION_NAME,
    )
    proposed_rows = _load_canonical_jsonl(
        raw_outputs[PROPOSED_NAME], context=PROPOSED_NAME
    )
    reserve_rows = _load_canonical_jsonl(
        raw_outputs[RESERVE_NAME], context=RESERVE_NAME
    )
    expected_profile = _mapping(
        summary.get("profile"), context="summary.json.profile"
    )
    review_by_iid = _validate_review_rows(
        review_rows, expected_profile=expected_profile
    )
    ranked = _validate_generation_rows(
        generation_rows,
        review_by_iid=review_by_iid,
        expected_profile=expected_profile,
    )
    _validate_bucket_artifacts(
        proposed_rows=proposed_rows,
        reserve_rows=reserve_rows,
        review_by_iid=review_by_iid,
        generation_rows=generation_rows,
    )
    _validate_summary_counts(
        summary,
        review_count=len(review_rows),
        generation_count=len(generation_rows),
    )

    retained: list[_CanonicalRow] = []
    retained_raw: bytes | None = None
    retained_path: Path | None = None
    if retain_exact8_manifest is not None:
        retained, retained_raw, retained_path = _load_retained_exact8(
            Path(retain_exact8_manifest)
        )
    selected, fill_ranks, fill_iids, fill_groups = _select_rows(
        ranked, retained
    )
    selected_payload = b"".join(record.raw_line for record in selected)
    ordered_iids: list[str] = []
    ordered_groups: list[str] = []
    for record in selected:
        iid, group, _instruction = _validate_generation_record(
            record,
            context="selected generation row",
            expected_schema=(
                RETAINED_GENERATION_SCHEMA
                if len(ordered_iids) < len(retained)
                else FINALIZER_GENERATION_SCHEMA
            ),
            expected_profile=(
                None
                if len(ordered_iids) < len(retained)
                else expected_profile
            ),
        )
        ordered_iids.append(iid)
        ordered_groups.append(group)

    selector_path = Path(__file__).resolve(strict=True)
    shared_selector_path = selector_path.with_name("wan22_select_exact8.py")
    policy = RETAIN_EXACT8_POLICY if retained else RANK_ONLY_POLICY
    retention_source: dict[str, Any] | None = None
    if retained_raw is not None and retained_path is not None:
        retention_source = {
            "path": str(retained_path),
            "sha256": _sha256(retained_raw),
            "bytes": len(retained_raw),
            "row_count": len(retained),
        }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "policy": policy,
        "implementation": {
            "selector_sha256": _source_digest(
                selector_path, context="exact512 selector implementation"
            ),
            "shared_strict_io_sha256": _source_digest(
                shared_selector_path,
                context="shared strict selector implementation",
            ),
        },
        "parent": {
            "done_sha256": _sha256(done_raw),
            "summary_sha256": _sha256(raw_outputs[SUMMARY_NAME]),
            "review_candidates_sha256": _sha256(raw_outputs[REVIEW_NAME]),
            "proposed_512_sha256": _sha256(raw_outputs[PROPOSED_NAME]),
            "reserve_128_sha256": _sha256(raw_outputs[RESERVE_NAME]),
            "generation_manifest_sha256": _sha256(
                raw_outputs[PARENT_GENERATION_NAME]
            ),
            "finalizer_implementation_sha256": finalizer_implementation_sha,
        },
        "retention": {
            "retained_row_count": len(retained),
            "source": retention_source,
            "ordered_iids": ordered_iids[: len(retained)],
            "ordered_group_ids": ordered_groups[: len(retained)],
        },
        "selection": {
            "row_count": SELECTED_ROW_COUNT,
            "retained_row_count": len(retained),
            "ranked_fill_row_count": len(fill_ranks),
            "ordered_iids": ordered_iids,
            "ordered_group_ids": ordered_groups,
            "ordered_fill_iids": fill_iids,
            "ordered_fill_group_ids": fill_groups,
            "ordered_fill_review_ranks": fill_ranks,
            "output_file": OUTPUT_MANIFEST_NAME,
            "output_sha256": _sha256(selected_payload),
            "output_bytes": len(selected_payload),
        },
    }
    receipt_payload = _canonical_bytes(receipt) + b"\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
    )
    try:
        _write_new(staging / OUTPUT_MANIFEST_NAME, selected_payload)
        _write_new(staging / OUTPUT_RECEIPT_NAME, receipt_payload)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        _strict._publish_directory_noreplace(staging, output)
    except _strict.Wan22Exact8SelectionError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise Wan22Exact512SelectionError(str(error)) from error
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a completed scale512 Goku finalizer and atomically "
            "select exactly 512 still-pending generation rows."
        )
    )
    parser.add_argument("--finalizer-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--retain-exact8-manifest",
        type=Path,
        help=(
            "Optional canonical exact-eight v9 manifest to preserve as the "
            "first eight output rows."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = select_exact512(
        finalizer_dir=args.finalizer_dir,
        output_dir=args.output_dir,
        retain_exact8_manifest=args.retain_exact8_manifest,
    )
    print(
        "[wan22-select-exact512] "
        f"rows={receipt['selection']['row_count']} "
        f"retained={receipt['selection']['retained_row_count']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

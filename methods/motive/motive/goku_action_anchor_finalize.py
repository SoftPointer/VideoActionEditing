"""Strictly finalize Goku action-editing anchor candidates.

The input to this stage is one prefilter ``selected.jsonl`` and exactly eight
Qwen-VL shards produced by :mod:`motive.goku_action_anchor_qwen`.  The merge is
fail-closed:

* every prefilter IID must occur exactly once in its SHA-256 IID shard;
* no unknown IID or duplicate is accepted;
* every Qwen ``input_digest`` must bind the complete canonical input row; and
* all Qwen rows must share one immutable run-configuration digest, while
  each shard must have one shard-bound configuration digest.

Only rows with a directly validated blind observation, independent Judge-A
admissibility pass, and independent Judge-B continuity pass are eligible.
The writer draft may be original or the sole target-core-locked continuity
repair.  Generic schema repairs remain ineligible.  Eligible rows are quality
ranked, de-duplicated by
``group_id``, and capped at twelve examples per normalized target verb.  The
proposed set uses deterministic category quotas with deterministic global
backfill.

The default contract below remains the historical 192/128/32 policy.  The
explicit ``scale512`` profile scales it to a 768-row review limit, 512 proposed
rows, 128 reserve rows, a 48-row target-verb cap, and 128/128/192/64 category
quotas.  That profile writes ``proposed_512.jsonl`` and ``reserve_128.jsonl``
and embeds a digest-bound closed profile in its row, summary, and done
metadata.

The output is an atomic, no-overwrite directory:

``review_candidates.jsonl``
    At most 192 hard-pass rows in deterministic quality order.
``proposed_128.jsonl``
    Exactly 128 automatically proposed, human-review-pending rows under the
    default strict policy; an explicitly requested partial run publishes every
    available hard-pass row up to 128.
``reserve_32.jsonl``
    Exactly 32 disjoint reserve rows by default; a partial run publishes as
    many disjoint reserve rows as remain, including zero.
``generation_manifest.jsonl``
    Review-only prompt/input records.  They are unconditionally marked
    production-ineligible and bind the exact selected-media evidence, verified
    source/anchor file digests, and a same-geometry output contract.
``summary.json`` and ``done.json``
    Provenance, policy, counts, hashes, and a terminal commit marker.

This module has no approval interface: it creates proposals, not human labels.
No row emitted by this stage is production-authorized.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from . import goku_action_anchor_qwen as qwen_module
from .goku_action_anchor_qwen import (
    ANCHOR_COMPATIBILITY_SCHEMA,
    ANCHOR_OBSERVATION_SCHEMA,
    DRAFT_CONTINUITY_SCHEMA,
    SHARD_RECEIPT_SCHEMA,
    TARGET_ADMISSIBILITY_SCHEMA,
    aggregate_draft_continuity,
    aggregate_target_admissibility,
    assigned_iids_for_shard,
    compatibility_target_support_evidence,
    deterministic_risk_codes,
    draft_continuity_evidence_failures,
    judge_a_instruction_support_evidence,
    qwen_provenance_digest,
    qwen_result_payload,
    resolve_draft_continuity_evidence,
    resolve_target_admissibility_evidence,
    shard_receipt_path,
    target_admissibility_evidence_failures,
    target_core_agreement_evidence,
    validate_anchor_observation,
    validate_compatibility,
    validate_compatibility_structure,
    validate_draft_continuity,
    validate_generic_repair_provenance,
    validate_semantic_repair_provenance,
    validate_target_admissibility,
    writer_target_instruction_support_evidence,
)


ROW_SCHEMA = "motive-goku-action-anchor-final-row-v8"
GENERATION_SCHEMA = "motive-goku-action-anchor-generation-v9"
SUMMARY_SCHEMA = "motive-goku-action-anchor-finalize-v8"
DONE_SCHEMA = "motive-goku-action-anchor-finalize-done-v8"
POLICY_VERSION = "goku-action-anchor-strict-continuity-v8"
SCALE512_PROFILE = "scale512"
SCALE512_ROW_SCHEMA = "motive-goku-action-anchor-final-row-scale512-v1"
SCALE512_GENERATION_SCHEMA = (
    "motive-goku-action-anchor-generation-scale512-v1"
)
SCALE512_SUMMARY_SCHEMA = (
    "motive-goku-action-anchor-finalize-scale512-v1"
)
SCALE512_DONE_SCHEMA = (
    "motive-goku-action-anchor-finalize-done-scale512-v1"
)
SCALE512_POLICY_VERSION = (
    "goku-action-anchor-strict-continuity-scale512-v1"
)
FINALIZATION_PROFILE_SCHEMA = (
    "motive-goku-action-anchor-finalization-profile-v1"
)
TEMPORAL_GEOMETRY_SCHEMA = (
    "motive-goku-action-anchor-temporal-geometry-v1"
)
MEDIA_FILE_VERIFICATION_SCHEMA = (
    "motive-goku-action-anchor-media-file-verification-v1"
)

REQUIRED_SHARD_COUNT = 8
REVIEW_LIMIT = 192
PROPOSED_SIZE = 128
RESERVE_SIZE = 32
MAX_PER_TARGET_VERB = 12
DEFAULT_SEED = 260730

REVIEW_NAME = "review_candidates.jsonl"
PROPOSED_NAME = "proposed_128.jsonl"
RESERVE_NAME = "reserve_32.jsonl"
GENERATION_NAME = "generation_manifest.jsonl"
SUMMARY_NAME = "summary.json"
DONE_NAME = "done.json"
ARTIFACT_NAMES = (
    REVIEW_NAME,
    PROPOSED_NAME,
    RESERVE_NAME,
    GENERATION_NAME,
    SUMMARY_NAME,
    DONE_NAME,
)

ACTION_CATEGORIES = (
    "locomotion",
    "posture",
    "interaction",
    "articulated",
)
FAMILY_QUOTAS: dict[str, int] = {
    "locomotion": 32,
    "posture": 32,
    "interaction": 48,
    "articulated": 16,
}

SCALE512_REVIEW_LIMIT = 768
SCALE512_PROPOSED_SIZE = 512
SCALE512_RESERVE_SIZE = 128
SCALE512_MAX_PER_TARGET_VERB = 48
SCALE512_PROPOSED_NAME = "proposed_512.jsonl"
SCALE512_RESERVE_NAME = "reserve_128.jsonl"
SCALE512_FAMILY_QUOTAS: dict[str, int] = {
    "locomotion": 128,
    "posture": 128,
    "interaction": 192,
    "articulated": 64,
}


@dataclass(frozen=True)
class _FinalizationProfile:
    """Closed internal policy selected before any input is read."""

    name: str | None
    review_limit: int
    proposed_size: int
    reserve_size: int
    max_per_target_verb: int
    category_quotas: tuple[tuple[str, int], ...]
    proposed_name: str
    reserve_name: str
    row_schema: str
    generation_schema: str
    summary_schema: str
    done_schema: str
    policy_version: str


_DEFAULT_PROFILE = _FinalizationProfile(
    name=None,
    review_limit=REVIEW_LIMIT,
    proposed_size=PROPOSED_SIZE,
    reserve_size=RESERVE_SIZE,
    max_per_target_verb=MAX_PER_TARGET_VERB,
    category_quotas=tuple(FAMILY_QUOTAS.items()),
    proposed_name=PROPOSED_NAME,
    reserve_name=RESERVE_NAME,
    row_schema=ROW_SCHEMA,
    generation_schema=GENERATION_SCHEMA,
    summary_schema=SUMMARY_SCHEMA,
    done_schema=DONE_SCHEMA,
    policy_version=POLICY_VERSION,
)
_SCALE512_PROFILE = _FinalizationProfile(
    name=SCALE512_PROFILE,
    review_limit=SCALE512_REVIEW_LIMIT,
    proposed_size=SCALE512_PROPOSED_SIZE,
    reserve_size=SCALE512_RESERVE_SIZE,
    max_per_target_verb=SCALE512_MAX_PER_TARGET_VERB,
    category_quotas=tuple(SCALE512_FAMILY_QUOTAS.items()),
    proposed_name=SCALE512_PROPOSED_NAME,
    reserve_name=SCALE512_RESERVE_NAME,
    row_schema=SCALE512_ROW_SCHEMA,
    generation_schema=SCALE512_GENERATION_SCHEMA,
    summary_schema=SCALE512_SUMMARY_SCHEMA,
    done_schema=SCALE512_DONE_SCHEMA,
    policy_version=SCALE512_POLICY_VERSION,
)

_SELECTED_REQUIRED_FIELDS = (
    "iid",
    "group_id",
    "family",
    "src_video",
    "resolved_src_video",
    "source_caption",
    "edited_caption",
    "prompt",
    "anchor_image",
    "resolved_anchor_image",
    "anchor_sha256",
    "source_video_sha256",
    "prefilter_score",
    "media",
    "motion",
)


class GokuActionAnchorFinalizeError(ValueError):
    """The frozen inputs or the requested final selection are invalid."""


def _reject_constant(value: str) -> None:
    raise GokuActionAnchorFinalizeError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuActionAnchorFinalizeError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise GokuActionAnchorFinalizeError(
            f"value is not canonical JSON: {error}"
        ) from error
    return text.encode("utf-8")


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _resolve_profile(profile: str | None) -> _FinalizationProfile:
    if profile is None:
        resolved = _DEFAULT_PROFILE
    elif type(profile) is str and profile == SCALE512_PROFILE:
        resolved = _SCALE512_PROFILE
    else:
        raise GokuActionAnchorFinalizeError(
            f"unsupported finalization profile: {profile!r}"
        )

    integer_fields = {
        "review_limit": resolved.review_limit,
        "proposed_size": resolved.proposed_size,
        "reserve_size": resolved.reserve_size,
        "max_per_target_verb": resolved.max_per_target_verb,
    }
    if any(type(value) is not int or value <= 0 for value in integer_fields.values()):
        raise GokuActionAnchorFinalizeError(
            f"finalization profile has invalid integer fields: {integer_fields}"
        )
    if resolved.review_limit < resolved.proposed_size + resolved.reserve_size:
        raise GokuActionAnchorFinalizeError(
            "finalization profile review_limit is smaller than proposed plus "
            "reserve"
        )
    quotas = dict(resolved.category_quotas)
    if (
        tuple(quotas) != ACTION_CATEGORIES
        or len(quotas) != len(resolved.category_quotas)
        or any(type(value) is not int or value < 0 for value in quotas.values())
        or sum(quotas.values()) != resolved.proposed_size
    ):
        raise GokuActionAnchorFinalizeError(
            "finalization profile category quotas are not closed"
        )
    artifact_names = (
        REVIEW_NAME,
        resolved.proposed_name,
        resolved.reserve_name,
        GENERATION_NAME,
        SUMMARY_NAME,
        DONE_NAME,
    )
    if len(set(artifact_names)) != len(artifact_names):
        raise GokuActionAnchorFinalizeError(
            "finalization profile artifact names are not unique"
        )
    for name in artifact_names:
        if Path(name).name != name or not name:
            raise GokuActionAnchorFinalizeError(
                f"finalization profile artifact name is unsafe: {name!r}"
            )
    return resolved


def _profile_metadata(profile: _FinalizationProfile) -> dict[str, Any] | None:
    """Return immutable metadata only for an explicitly selected profile."""

    if profile.name is None:
        return None
    config: dict[str, Any] = {
        "required_qwen_shard_count": REQUIRED_SHARD_COUNT,
        "review_limit": profile.review_limit,
        "proposed_size": profile.proposed_size,
        "reserve_size": profile.reserve_size,
        "max_per_target_verb": profile.max_per_target_verb,
        "category_quotas": dict(profile.category_quotas),
        "artifacts": {
            "review": REVIEW_NAME,
            "proposed": profile.proposed_name,
            "reserve": profile.reserve_name,
            "generation": GENERATION_NAME,
            "summary": SUMMARY_NAME,
            "done": DONE_NAME,
        },
        "schemas": {
            "row": profile.row_schema,
            "generation": profile.generation_schema,
            "summary": profile.summary_schema,
            "done": profile.done_schema,
        },
        "policy_version": profile.policy_version,
    }
    return {
        "schema_version": FINALIZATION_PROFILE_SCHEMA,
        "name": profile.name,
        "config": config,
        "config_sha256": _object_digest(config),
    }


def _frozen_run_config(
    *,
    model_path: str,
    model_revision: str,
    transformers_version: str,
    implementation_digest: str,
) -> dict[str, Any]:
    """Rebuild the complete frozen v8 Qwen runtime contract locally."""

    text_digest = lambda value: hashlib.sha256(  # noqa: E731
        value.encode("utf-8")
    ).hexdigest()
    return {
        "model_path": model_path,
        "model_revision": model_revision,
        "transformers_version": transformers_version,
        "max_samples": None,
        "num_shards": REQUIRED_SHARD_COUNT,
        "max_new_tokens": 1536,
        "nframes": 12,
        "max_pixels": 589_824,
        "attn_implementation": "sdpa",
        "allow_download": False,
        "repair_attempts": 1,
        "anchor_observation_schema": ANCHOR_OBSERVATION_SCHEMA,
        "anchor_compatibility_schema": ANCHOR_COMPATIBILITY_SCHEMA,
        "target_admissibility_schema": TARGET_ADMISSIBILITY_SCHEMA,
        "draft_continuity_schema": DRAFT_CONTINUITY_SCHEMA,
        "blind_prompt_digest": text_digest(
            qwen_module.BLIND_SYSTEM + "\n" + qwen_module.BLIND_PROMPT
        ),
        "compatibility_prompt_digest": text_digest(
            qwen_module.COMPATIBILITY_SYSTEM
            + "\n"
            + qwen_module.COMPATIBILITY_PROMPT
        ),
        "judge_a_prompt_digest": text_digest(
            qwen_module.JUDGE_A_SYSTEM + "\n" + qwen_module.JUDGE_A_PROMPT
        ),
        "judge_b_prompt_digest": text_digest(
            qwen_module.JUDGE_B_SYSTEM + "\n" + qwen_module.JUDGE_B_PROMPT
        ),
        "draft_repair_prompt_digest": text_digest(
            qwen_module.DRAFT_REPAIR_SYSTEM
            + "\n"
            + qwen_module.DRAFT_REPAIR_PROMPT
        ),
        "repair_schema_digest": _object_digest(
            {
                "anchor_observation": (
                    qwen_module.ANCHOR_OBSERVATION_REPAIR_SCHEMA
                ),
                "compatibility": qwen_module.COMPATIBILITY_REPAIR_SCHEMA,
                "target_admissibility": (
                    qwen_module.TARGET_ADMISSIBILITY_REPAIR_SCHEMA
                ),
                "draft_continuity": (
                    qwen_module.DRAFT_CONTINUITY_REPAIR_SCHEMA
                ),
            }
        ),
        "implementation_digest": implementation_digest,
        "generation": {
            "do_sample": False,
            "visual_input": "exact_i0_plus_source_mosaic",
        },
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _parse_json(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuActionAnchorFinalizeError(
            f"{context} is not UTF-8"
        ) from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, GokuActionAnchorFinalizeError):
            raise
        raise GokuActionAnchorFinalizeError(
            f"{context} is not strict JSON: {error}"
        ) from error


def _regular_file(path: str | Path, *, context: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise FileNotFoundError(
            f"{context} must be a regular non-symlink file: {expanded}"
        )
    return expanded.resolve(strict=True)


def _load_jsonl(
    path: str | Path,
    *,
    context: str,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], bytes, Path]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise GokuActionAnchorFinalizeError(
            f"{context} must end with a newline: {resolved}"
        )
    if not raw and not allow_empty:
        raise GokuActionAnchorFinalizeError(
            f"{context} is empty: {resolved}"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise GokuActionAnchorFinalizeError(
                f"{context} contains a blank line: {resolved}:{line_number}"
            )
        value = _parse_json(
            line,
            context=f"{context} {resolved}:{line_number}",
        )
        if not isinstance(value, dict):
            raise GokuActionAnchorFinalizeError(
                f"{context} row is not an object: "
                f"{resolved}:{line_number}"
            )
        rows.append(value)
    return rows, raw, resolved


def _canonical_string(
    value: Any,
    *,
    context: str,
    lower: bool = False,
) -> str:
    if not isinstance(value, str) or not value:
        raise GokuActionAnchorFinalizeError(
            f"{context} must be a non-empty string"
        )
    if value != value.strip() or "\x00" in value:
        raise GokuActionAnchorFinalizeError(
            f"{context} is not a canonical string"
        )
    if lower and value != value.casefold():
        raise GokuActionAnchorFinalizeError(
            f"{context} must be case-folded"
        )
    return value


def _sha256_field(value: Any, *, context: str) -> str:
    digest = _canonical_string(value, context=context)
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GokuActionAnchorFinalizeError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return digest


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GokuActionAnchorFinalizeError(
            f"{context} must be an object"
        )
    return value


def _list(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise GokuActionAnchorFinalizeError(
            f"{context} must be an array"
        )
    return value


def _positive_finite_float(value: Any, *, context: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise GokuActionAnchorFinalizeError(
            f"{context} must be a positive finite number"
        )
    return float(value)


def _strict_temporal_geometry(
    media_value: Any,
    *,
    iid: str,
) -> dict[str, Any]:
    """Derive the immutable source/output geometry contract from ``media``."""

    media = _mapping(media_value, context=f"selected iid={iid} media")
    required = {"frame_count", "fps", "duration_seconds"}
    missing = sorted(required - set(media))
    if missing:
        raise GokuActionAnchorFinalizeError(
            f"selected iid={iid} media is missing temporal fields: {missing}"
        )
    frame_count = media["frame_count"]
    if type(frame_count) is not int or frame_count <= 0:
        raise GokuActionAnchorFinalizeError(
            f"selected iid={iid} media frame_count must be a positive integer"
        )
    if frame_count % 4 != 1:
        raise GokuActionAnchorFinalizeError(
            f"selected iid={iid} media frame_count must satisfy 4n+1"
        )
    fps = _positive_finite_float(
        media["fps"],
        context=f"selected iid={iid} media fps",
    )
    duration = _positive_finite_float(
        media["duration_seconds"],
        context=f"selected iid={iid} media duration_seconds",
    )
    one_frame_seconds = 1.0 / fps
    timeline_duration = (frame_count - 1) / fps
    timeline_error = abs(duration - timeline_duration)
    numeric_slack = max(1e-9, one_frame_seconds * 1e-7)
    if timeline_error > one_frame_seconds + numeric_slack:
        raise GokuActionAnchorFinalizeError(
            f"selected iid={iid} media duration differs from its frame "
            "timeline by more than one frame"
        )
    return {
        "schema_version": TEMPORAL_GEOMETRY_SCHEMA,
        "source_frame_count": frame_count,
        "required_output_frame_count": frame_count,
        "source_fps": fps,
        "required_output_fps": fps,
        "source_duration_seconds": duration,
        "required_output_duration_seconds": duration,
        "maximum_duration_delta_frames": 1,
        "maximum_duration_delta_seconds": one_frame_seconds,
        "frame_count_form": "4n+1",
        "frame_count_modulus": 4,
        "frame_count_remainder": 1,
        "source_timeline_duration_seconds": timeline_duration,
        "source_timeline_error_seconds": timeline_error,
        "requirements": {
            "same_frame_count": True,
            "same_fps": True,
            "duration_absolute_delta_at_most_one_frame": True,
        },
    }


def _verified_selected_media_files(
    row: Mapping[str, Any],
    *,
    iid: str,
) -> dict[str, Any]:
    """Open and independently re-hash the selected source and exact-I0 files."""

    verified: dict[str, Any] = {
        "schema_version": MEDIA_FILE_VERIFICATION_SCHEMA,
    }
    bindings = (
        (
            "source_video",
            "resolved_src_video",
            "source_video_sha256",
        ),
        (
            "anchor_image",
            "resolved_anchor_image",
            "anchor_sha256",
        ),
    )
    for output_name, path_field, digest_field in bindings:
        path_text = _canonical_string(
            row[path_field],
            context=f"selected iid={iid} {path_field}",
        )
        if not Path(path_text).is_absolute():
            raise GokuActionAnchorFinalizeError(
                f"selected iid={iid} {path_field} must be absolute"
            )
        try:
            resolved = _regular_file(
                path_text,
                context=f"selected iid={iid} {path_field}",
            )
        except FileNotFoundError as error:
            raise GokuActionAnchorFinalizeError(
                f"selected iid={iid} {path_field} is missing or not a "
                "regular non-symlink file"
            ) from error
        if str(resolved) != path_text:
            raise GokuActionAnchorFinalizeError(
                f"selected iid={iid} {path_field} is not a canonical "
                "resolved path"
            )
        expected_digest = _sha256_field(
            row[digest_field],
            context=f"selected iid={iid} {digest_field}",
        )
        actual_digest = _sha256_file(resolved)
        if actual_digest != expected_digest:
            raise GokuActionAnchorFinalizeError(
                f"selected iid={iid} {output_name} SHA-256 differs from "
                f"{digest_field}"
            )
        verified[output_name] = {
            "resolved_path": str(resolved),
            "sha256": actual_digest,
            "bytes": resolved.stat().st_size,
        }
    verified["verification_digest"] = _object_digest(verified)
    return verified


def _selected_iid(row: Mapping[str, Any], *, line_number: int) -> str:
    return _canonical_string(
        row.get("iid"),
        context=f"selected row {line_number} iid",
    )


def _validate_selected_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
) -> str:
    missing = [field for field in _SELECTED_REQUIRED_FIELDS if field not in row]
    if missing:
        raise GokuActionAnchorFinalizeError(
            f"selected row {line_number} is missing fields: {missing}"
        )
    iid = _selected_iid(row, line_number=line_number)
    for field in (
        "group_id",
        "family",
        "src_video",
        "resolved_src_video",
        "source_caption",
        "edited_caption",
        "prompt",
        "anchor_image",
        "resolved_anchor_image",
    ):
        # Upstream Goku can contain empty language fields; this curation
        # contract deliberately excludes those rows.
        _canonical_string(
            row[field],
            context=f"selected iid={iid} {field}",
        )
    _sha256_field(
        row["anchor_sha256"],
        context=f"selected iid={iid} anchor_sha256",
    )
    _sha256_field(
        row["source_video_sha256"],
        context=f"selected iid={iid} source_video_sha256",
    )
    score = row["prefilter_score"]
    if type(score) not in (int, float) or not math.isfinite(float(score)):
        raise GokuActionAnchorFinalizeError(
            f"selected iid={iid} prefilter_score must be finite"
        )
    _strict_temporal_geometry(row["media"], iid=iid)
    _mapping(row["motion"], context=f"selected iid={iid} motion")
    if "actor_motion" in row:
        _mapping(
            row["actor_motion"],
            context=f"selected iid={iid} actor_motion",
        )
    return iid


def _iid_shard(iid: str, shard_count: int = REQUIRED_SHARD_COUNT) -> int:
    # This is the exact partition contract used by
    # goku_action_anchor_qwen.run_audit.  Binding the first 64 digest bits is
    # intentional and must not be silently changed to the full 256-bit value.
    prefix = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:16]
    return int(prefix, 16) % shard_count


def _discover_shards(
    qwen_root: str | Path,
) -> tuple[tuple[Path, Path], ...]:
    root = Path(qwen_root).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(
            f"Qwen root must be a non-symlink directory: {root}"
        )
    resolved = root.resolve(strict=True)
    expected = tuple(
        resolved / f"qwen_shard_{index:03d}.jsonl"
        for index in range(REQUIRED_SHARD_COUNT)
    )
    expected_receipts = tuple(
        shard_receipt_path(path) for path in expected
    )
    actual = set(resolved.glob("qwen_shard_*.jsonl"))
    if actual != set(expected):
        raise GokuActionAnchorFinalizeError(
            "Qwen shard set mismatch: "
            f"missing={sorted(str(path) for path in set(expected) - actual)} "
            f"extra={sorted(str(path) for path in actual - set(expected))}"
        )
    for index, path in enumerate(expected):
        _regular_file(path, context=f"Qwen shard {index}")
    actual_receipts = set(resolved.glob("qwen_shard_*.receipt.json"))
    if actual_receipts != set(expected_receipts):
        raise GokuActionAnchorFinalizeError(
            "Qwen receipt set mismatch: "
            f"missing={sorted(str(path) for path in set(expected_receipts) - actual_receipts)} "
            f"extra={sorted(str(path) for path in actual_receipts - set(expected_receipts))}"
        )
    for index, path in enumerate(expected_receipts):
        _regular_file(path, context=f"Qwen shard receipt {index}")
    return tuple(zip(expected, expected_receipts))


def _load_and_validate_shard_receipt(
    receipt_path: Path,
    *,
    shard_path: Path,
    shard_raw: bytes,
    shard_rows: Sequence[Mapping[str, Any]],
    shard_index: int,
    selected_rows: Sequence[Mapping[str, Any]],
    selected_path: Path,
    selected_sha256: str,
    qwen_implementation_digest: str,
) -> dict[str, Any]:
    """Independently verify one terminal receipt, including an empty shard."""

    resolved = _regular_file(
        receipt_path,
        context=f"Qwen shard receipt {shard_index}",
    )
    raw = resolved.read_bytes()
    receipt_value = _parse_json(
        raw,
        context=f"Qwen shard receipt {shard_index} {resolved}",
    )
    receipt = _mapping(
        receipt_value,
        context=f"Qwen shard receipt {shard_index}",
    )
    required = {
        "schema_version",
        "status",
        "execution_manifest",
        "execution_manifest_sha256",
        "root",
        "shard_index",
        "num_shards",
        "assigned_iids",
        "implementation_digest",
        "config_digest",
        "run_config_digest",
        "run_config",
        "model_path",
        "model_revision",
        "transformers_version",
        "output",
        "receipt_digest",
    }
    if set(receipt) != required:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} is not closed: "
            f"{sorted(set(receipt) ^ required)}"
        )
    if receipt.get("schema_version") != SHARD_RECEIPT_SCHEMA:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} schema differs"
        )
    if receipt.get("status") != "complete":
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} is not terminal complete"
        )
    if receipt.get("shard_index") != shard_index:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} index differs"
        )
    if receipt.get("num_shards") != REQUIRED_SHARD_COUNT:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} num_shards differs"
        )
    if receipt.get("execution_manifest") != str(selected_path):
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} manifest path differs"
        )
    if receipt.get("execution_manifest_sha256") != selected_sha256:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} manifest SHA differs"
        )
    root_text = _canonical_string(
        receipt.get("root"),
        context=f"Qwen shard receipt {shard_index} root",
    )
    root_path = Path(root_text)
    if root_path.is_symlink() or not root_path.is_dir():
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} root is not a directory"
        )
    if str(root_path.resolve(strict=True)) != root_text:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} root is not canonical"
        )

    run_config = _mapping(
        receipt.get("run_config"),
        context=f"Qwen shard receipt {shard_index} run_config",
    )
    run_config_digest = _sha256_field(
        receipt.get("run_config_digest"),
        context=f"Qwen shard receipt {shard_index} run_config_digest",
    )
    if _object_digest(run_config) != run_config_digest:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} run_config digest differs"
        )
    implementation_digest = _sha256_field(
        receipt.get("implementation_digest"),
        context=f"Qwen shard receipt {shard_index} implementation_digest",
    )
    if implementation_digest != qwen_implementation_digest:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} implementation differs"
        )
    runtime_identity = {
        field: _canonical_string(
            receipt.get(field),
            context=f"Qwen shard receipt {shard_index} {field}",
        )
        for field in ("model_path", "model_revision", "transformers_version")
    }
    expected_run_config = _frozen_run_config(
        **runtime_identity,
        implementation_digest=qwen_implementation_digest,
    )
    if dict(run_config) != expected_run_config:
        differing = sorted(
            key
            for key in set(run_config) | set(expected_run_config)
            if run_config.get(key) != expected_run_config.get(key)
        )
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} full run_config differs: "
            f"{differing}"
        )

    config_digest = _sha256_field(
        receipt.get("config_digest"),
        context=f"Qwen shard receipt {shard_index} config_digest",
    )
    expected_config_digest = _object_digest(
        {
            "run_config_digest": run_config_digest,
            "execution_manifest": str(selected_path),
            "execution_manifest_sha256": selected_sha256,
            "root": root_text,
            "shard_index": shard_index,
            "num_shards": REQUIRED_SHARD_COUNT,
        }
    )
    if config_digest != expected_config_digest:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} config digest differs"
        )

    expected_iids = assigned_iids_for_shard(
        selected_rows,
        shard_index=shard_index,
        num_shards=REQUIRED_SHARD_COUNT,
        max_samples=None,
    )
    assigned = _list(
        receipt.get("assigned_iids"),
        context=f"Qwen shard receipt {shard_index} assigned_iids",
    )
    if assigned != expected_iids:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} assigned_iids differ"
        )
    actual_iids = [str(row.get("iid", "")) for row in shard_rows]
    if actual_iids != expected_iids:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard {shard_index} IID/order differs from receipt"
        )
    status_counts = Counter(
        str(row.get("status", "missing")) for row in shard_rows
    )
    expected_output = {
        "path": str(shard_path),
        "sha256": hashlib.sha256(shard_raw).hexdigest(),
        "bytes": len(shard_raw),
        "rows": len(shard_rows),
        "status_counts": dict(sorted(status_counts.items())),
    }
    output = _mapping(
        receipt.get("output"),
        context=f"Qwen shard receipt {shard_index} output",
    )
    if dict(output) != expected_output:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} output binding differs"
        )
    receipt_payload = dict(receipt)
    actual_receipt_digest = _sha256_field(
        receipt_payload.pop("receipt_digest"),
        context=f"Qwen shard receipt {shard_index} receipt_digest",
    )
    if actual_receipt_digest != _object_digest(receipt_payload):
        raise GokuActionAnchorFinalizeError(
            f"Qwen shard receipt {shard_index} digest differs"
        )
    return {
        **dict(receipt),
        "path": str(resolved),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_v8_evidence_shapes(
    qwen: Mapping[str, Any],
    *,
    iid: str,
    selected: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any] | None,
    Mapping[str, Any] | None,
]:
    """Validate the complete v8 result/provenance/media envelope."""

    anchor = _mapping(
        qwen.get("anchor_observation"),
        context=f"Qwen iid={iid} anchor_observation",
    )
    judge_a = _mapping(
        qwen.get("target_admissibility"),
        context=f"Qwen iid={iid} target_admissibility",
    )
    compatibility_value = qwen.get("compatibility")
    judge_b_value = qwen.get("draft_continuity")
    if (compatibility_value is None) != (judge_b_value is None):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} writer/Judge-B route is only partially present"
        )
    compatibility = (
        _mapping(
            compatibility_value,
            context=f"Qwen iid={iid} compatibility",
        )
        if compatibility_value is not None
        else None
    )
    judge_b = (
        _mapping(
            judge_b_value,
            context=f"Qwen iid={iid} draft_continuity",
        )
        if judge_b_value is not None
        else None
    )
    try:
        validate_anchor_observation(dict(anchor))
        validate_target_admissibility(dict(judge_a))
        if compatibility is not None and judge_b is not None:
            validate_compatibility_structure(
                dict(compatibility),
                observation=dict(anchor),
            )
            validate_draft_continuity(dict(judge_b))
    except (KeyError, TypeError, ValueError) as error:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} v8 closed-schema validation failed: {error}"
        ) from error
    if anchor.get("schema_version") != ANCHOR_OBSERVATION_SCHEMA:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} anchor schema differs"
        )
    if judge_a.get("schema_version") != TARGET_ADMISSIBILITY_SCHEMA:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} Judge-A schema differs"
        )
    if (
        compatibility is not None
        and compatibility.get("schema_version")
        != ANCHOR_COMPATIBILITY_SCHEMA
    ):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} compatibility schema differs"
        )
    if (
        judge_b is not None
        and judge_b.get("schema_version") != DRAFT_CONTINUITY_SCHEMA
    ):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} Judge-B schema differs"
        )
    try:
        expected_result_digest = _object_digest(qwen_result_payload(qwen))
    except (KeyError, TypeError, ValueError) as error:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} v8 result payload is incomplete"
        ) from error
    if _sha256_field(
        qwen.get("result_digest"),
        context=f"Qwen iid={iid} result_digest",
    ) != expected_result_digest:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} result_digest differs"
        )
    if _sha256_field(
        qwen.get("anchor_observation_digest"),
        context=f"Qwen iid={iid} anchor_observation_digest",
    ) != _object_digest(anchor):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} anchor digest differs"
        )
    visual_input_digest = _sha256_field(
        qwen.get("visual_input_digest"),
        context=f"Qwen iid={iid} visual_input_digest",
    )
    judge_a_visual_input_digest = _sha256_field(
        qwen.get("target_admissibility_visual_input_digest"),
        context=(
            f"Qwen iid={iid} "
            "target_admissibility_visual_input_digest"
        ),
    )
    if judge_a_visual_input_digest != visual_input_digest:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} Judge-A visual input digest differs"
        )
    try:
        expected_judge_a_evidence = resolve_target_admissibility_evidence(
            judge_a,
            row=selected,
            observation=anchor,
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} Judge-A evidence cannot be resolved"
        ) from error
    if (
        qwen.get("target_admissibility_resolved_evidence")
        != expected_judge_a_evidence
    ):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} Judge-A resolved evidence differs"
        )
    if compatibility is not None and judge_b is not None:
        try:
            expected_judge_b_evidence = resolve_draft_continuity_evidence(
                judge_b,
                compatibility=compatibility,
            )
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise GokuActionAnchorFinalizeError(
                f"Qwen iid={iid} Judge-B evidence cannot be resolved"
            ) from error
        if (
            qwen.get("draft_continuity_resolved_evidence")
            != expected_judge_b_evidence
        ):
            raise GokuActionAnchorFinalizeError(
                f"Qwen iid={iid} Judge-B resolved evidence differs"
            )

    media = _mapping(
        qwen.get("media_verification"),
        context=f"Qwen iid={iid} media_verification",
    )
    required_media = {
        "exact_i0",
        "lossless_png",
        "width",
        "height",
        "anchor_sha256",
        "source_video_sha256",
        "frame_zero_rgb_sha256",
    }
    if set(media) != required_media:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} media_verification is not closed"
        )
    if media.get("exact_i0") is not True or media.get("lossless_png") is not True:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} exact lossless I0 binding is false"
        )
    for dimension in ("width", "height"):
        if type(media.get(dimension)) is not int or media[dimension] <= 0:
            raise GokuActionAnchorFinalizeError(
                f"Qwen iid={iid} media {dimension} is invalid"
            )
    if _sha256_field(
        media.get("anchor_sha256"),
        context=f"Qwen iid={iid} media anchor_sha256",
    ) != selected["anchor_sha256"]:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} media anchor SHA differs"
        )
    if _sha256_field(
        media.get("source_video_sha256"),
        context=f"Qwen iid={iid} media source_video_sha256",
    ) != selected["source_video_sha256"]:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} media source SHA differs"
        )
    _sha256_field(
        media.get("frame_zero_rgb_sha256"),
        context=f"Qwen iid={iid} frame_zero_rgb_sha256",
    )
    selected_media = _mapping(
        selected["media"],
        context=f"selected iid={iid} media",
    )
    expected_width = selected_media.get(
        "anchor_width",
        selected_media.get("width"),
    )
    expected_height = selected_media.get(
        "anchor_height",
        selected_media.get("height"),
    )
    if (
        media["width"] != expected_width
        or media["height"] != expected_height
    ):
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} media dimensions differ"
        )
    if qwen.get("resolved_src_video") != selected["resolved_src_video"]:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} resolved source differs"
        )
    if qwen.get("resolved_anchor_image") != selected["resolved_anchor_image"]:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} resolved anchor differs"
        )

    try:
        expected_provenance = qwen_provenance_digest(qwen)
    except (KeyError, TypeError, ValueError) as error:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} v8 provenance payload is incomplete"
        ) from error
    if _sha256_field(
        qwen.get("provenance_digest"),
        context=f"Qwen iid={iid} provenance_digest",
    ) != expected_provenance:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} provenance digest differs"
        )
    generic_failures = validate_generic_repair_provenance(qwen)
    if generic_failures:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} generic repair provenance differs: "
            f"{generic_failures}"
        )
    semantic_failures = validate_semantic_repair_provenance(
        qwen,
        selected_row=selected,
        observation=anchor,
    )
    if semantic_failures:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} v8 semantic provenance differs: "
            f"{semantic_failures}"
        )
    return anchor, judge_a, compatibility, judge_b


def _hard_gate_failures(
    qwen: Mapping[str, Any],
    *,
    iid: str,
    selected: Mapping[str, Any],
) -> list[str]:
    """Recompute both v8 judges and accept only a final aggregate pass."""

    status = qwen.get("status")
    if status not in {"ok", "error"}:
        raise GokuActionAnchorFinalizeError(
            f"Qwen iid={iid} status must be ok or error"
        )
    if status != "ok":
        return [f"qwen_status:{status}"]
    anchor, judge_a, compatibility, judge_b = _validate_v8_evidence_shapes(
        qwen,
        iid=iid,
        selected=selected,
    )
    failures: list[str] = []
    aggregate_a = aggregate_target_admissibility(
        judge_a,
        row=selected,
        observation=anchor,
    )
    instruction_support = judge_a_instruction_support_evidence(
        judge_a,
        row=selected,
        observation=anchor,
    )
    required_instruction_support = (
        "target_evidence_ref_is_instruction",
        "target_action_normalized_supports_verb",
        "instruction_supports_target_action",
        "novel_trajectory_description_supports_target_action",
    )
    for field in required_instruction_support:
        if instruction_support.get(field) is not True:
            failures.append(f"judge_a:immutable_instruction_support:{field}")
    if qwen.get("target_admissibility_aggregate") != aggregate_a:
        failures.append("judge_a:aggregate_binding")
    if aggregate_a["decision"] != "pass":
        failures.append("judge_a:not_pass")
    if qwen.get("target_admissibility_validated_from") != "original":
        failures.append("judge_a:not_direct_original")
    if judge_a.get("confidence") not in {"medium", "high"}:
        failures.append("judge_a:confidence")
    failures.extend(
        target_admissibility_evidence_failures(
            judge_a,
            row=selected,
            observation=anchor,
        )
    )
    if aggregate_a["decision"] != "pass":
        return sorted(set(failures))
    if compatibility is None or judge_b is None:
        failures.append("judge_a:pass_without_writer_or_judge_b")
        return sorted(set(failures))
    writer_instruction_support = writer_target_instruction_support_evidence(
        compatibility,
        selected,
    )
    if not writer_instruction_support[
        "complete_instruction_target_contract"
    ]:
        failures.append("writer:incomplete_immutable_instruction_target")
    target_core_agreement = target_core_agreement_evidence(
        judge_a,
        compatibility,
        selected,
    )
    if not target_core_agreement["agreement_verified"]:
        failures.append("judge_a_writer_target_core:not_agreed")
        for field in (
            "judge_a_instruction_bound",
            "writer_instruction_bound",
            "normalized_action_bidirectional_agreement",
            "target_verb_overlap",
        ):
            if not target_core_agreement[field]:
                failures.append(
                    f"judge_a_writer_target_core:{field}"
                )

    aggregate_b = aggregate_draft_continuity(
        judge_b,
        compatibility=compatibility,
        observation=anchor,
    )
    if qwen.get("draft_continuity_aggregate") != aggregate_b:
        failures.append("judge_b:aggregate_binding")
    if aggregate_b["decision"] != "pass":
        failures.append("judge_b:not_pass")
    if qwen.get("draft_continuity_validated_from") != "original":
        failures.append("judge_b:not_direct_original")
    if judge_b.get("confidence") not in {"medium", "high"}:
        failures.append("judge_b:confidence")
    failures.extend(
        draft_continuity_evidence_failures(
            judge_b,
            compatibility=compatibility,
        )
    )
    try:
        validate_compatibility(
            dict(compatibility),
            observation=dict(anchor),
        )
    except (KeyError, TypeError, ValueError):
        failures.append("compatibility:final_strict")

    recomputed_risks = deterministic_risk_codes(
        judge_a,
        judge_b,
        row=selected,
        observation=anchor,
        compatibility=compatibility,
    )
    if qwen.get("deterministic_risk_codes") != recomputed_risks:
        failures.append("deterministic_risk_codes:binding")
    if recomputed_risks:
        failures.extend(
            f"deterministic_risk:{code}" for code in recomputed_risks
        )
    if (
        qwen.get("pipeline_stage") != "judge_b"
        or qwen.get("pipeline_decision") != "pass"
    ):
        failures.append("pipeline:not_clean_judge_b_pass")
    final_writer = qwen.get("compatibility_validated_from")
    if final_writer not in {"original", "semantic_repair_1"}:
        failures.append(f"compatibility_validated_from:{final_writer}")

    anchor_equal = {
        "initial_state_clarity": "clear",
        "subject_visibility": "clear",
        "actor_motion": "clear",
        "single_continuous_shot": "yes",
    }
    for field, expected in anchor_equal.items():
        if anchor.get(field) != expected:
            failures.append(f"anchor:{field}")
    if anchor.get("motion_dynamics") not in {"strong", "moderate"}:
        failures.append("anchor:motion_dynamics")
    if anchor.get("source_quality") not in {"high", "acceptable"}:
        failures.append("anchor:source_quality")
    if anchor.get("resolution_quality") not in {"high", "acceptable"}:
        failures.append("anchor:resolution_quality")
    if anchor.get("camera_motion") not in {"none", "weak"}:
        failures.append("anchor:camera_motion")
    if anchor.get("background_motion") not in {"none", "weak"}:
        failures.append("anchor:background_motion")
    if anchor.get("artifact_level") not in {"none", "low"}:
        failures.append("anchor:artifact_level")
    if anchor.get("uncertainty_codes") != []:
        failures.append("anchor:uncertainty_codes")
    if compatibility.get("action_category") not in ACTION_CATEGORIES:
        failures.append("compatibility:action_category")
    return sorted(set(failures))


def _quality_key(row: Mapping[str, Any], *, seed: int) -> tuple[Any, ...]:
    qwen = _mapping(
        row["qwen_action_anchor"],
        context="fused Qwen action anchor",
    )
    anchor = _mapping(qwen["anchor_observation"], context="anchor observation")
    compatibility = _mapping(
        qwen["compatibility"],
        context="anchor compatibility",
    )
    judge_a = _mapping(
        qwen["target_admissibility"],
        context="target admissibility",
    )
    judge_b = _mapping(
        qwen["draft_continuity"],
        context="draft continuity",
    )
    iid = str(row["iid"])
    tie = hashlib.sha256(f"{seed}\0{iid}".encode("utf-8")).hexdigest()
    return (
        0 if judge_a["confidence"] == "high" else 1,
        0 if judge_b["confidence"] == "high" else 1,
        0 if compatibility["confidence"] == "high" else 1,
        0 if compatibility["decision"] == "accept" else 1,
        0 if compatibility["anchor_compatibility"] == "compatible" else 1,
        0 if compatibility["caption_consistency"] == "consistent" else 1,
        0 if compatibility["causal_bridge"] == "direct" else 1,
        0 if anchor["source_quality"] == "high" else 1,
        0 if anchor["resolution_quality"] == "high" else 1,
        0 if anchor["camera_motion"] == "none" else 1,
        0 if anchor["background_motion"] == "none" else 1,
        0 if anchor["artifact_level"] == "none" else 1,
        -float(row["prefilter_score"]),
        tie,
        iid,
    )


def _diverse_review_pool(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    max_per_target_verb: int = MAX_PER_TARGET_VERB,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if type(limit) is not int or limit <= 0:
        raise GokuActionAnchorFinalizeError("review limit must be positive")
    if type(max_per_target_verb) is not int or max_per_target_verb <= 0:
        raise GokuActionAnchorFinalizeError(
            "target verb cap must be positive"
        )
    selected: list[dict[str, Any]] = []
    groups: set[str] = set()
    verb_counts: Counter[str] = Counter()
    diversity_rejections: Counter[str] = Counter()
    for row in rows:
        group_id = str(row["group_id"])
        compatibility = row["qwen_action_anchor"]["compatibility"]
        target_verb = str(compatibility["target_action_verb"])
        if group_id in groups:
            diversity_rejections["duplicate_group_id"] += 1
            continue
        if verb_counts[target_verb] >= max_per_target_verb:
            diversity_rejections["target_verb_cap"] += 1
            continue
        selected.append(row)
        groups.add(group_id)
        verb_counts[target_verb] += 1
        if len(selected) == limit:
            break
    return selected, diversity_rejections


def _scaled_category_quotas(
    proposed_size: int,
    *,
    maximum_proposed_size: int = PROPOSED_SIZE,
    full_category_quotas: Mapping[str, int] = FAMILY_QUOTAS,
) -> dict[str, int]:
    if (
        type(proposed_size) is not int
        or type(maximum_proposed_size) is not int
        or not 1 <= proposed_size <= maximum_proposed_size
    ):
        raise GokuActionAnchorFinalizeError(
            "proposed_size must be in [1, maximum_proposed_size]"
        )
    if (
        tuple(full_category_quotas) != ACTION_CATEGORIES
        or any(
            type(value) is not int or value < 0
            for value in full_category_quotas.values()
        )
        or sum(full_category_quotas.values()) != maximum_proposed_size
    ):
        raise GokuActionAnchorFinalizeError(
            "full_category_quotas must be a closed maximum-size quota"
        )
    exact = {
        category: (
            full_category_quotas[category]
            * proposed_size
            / maximum_proposed_size
        )
        for category in ACTION_CATEGORIES
    }
    quotas = {
        category: math.floor(exact[category])
        for category in ACTION_CATEGORIES
    }
    remainder = proposed_size - sum(quotas.values())
    category_order = {
        category: index
        for index, category in enumerate(ACTION_CATEGORIES)
    }
    ranked_remainders = sorted(
        ACTION_CATEGORIES,
        key=lambda category: (
            -(exact[category] - quotas[category]),
            category_order[category],
        ),
    )
    for category in ranked_remainders[:remainder]:
        quotas[category] += 1
    if sum(quotas.values()) != proposed_size:
        raise AssertionError("scaled category quotas do not sum to target")
    return quotas


def _select_proposed(
    review_rows: Sequence[dict[str, Any]],
    *,
    proposed_size: int = PROPOSED_SIZE,
    maximum_proposed_size: int = PROPOSED_SIZE,
    full_category_quotas: Mapping[str, int] = FAMILY_QUOTAS,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    effective_quotas = _scaled_category_quotas(
        proposed_size,
        maximum_proposed_size=maximum_proposed_size,
        full_category_quotas=full_category_quotas,
    )
    selected: list[dict[str, Any]] = []
    selected_iids: set[str] = set()
    realized: Counter[str] = Counter()

    for category in ACTION_CATEGORIES:
        quota = effective_quotas[category]
        for row in review_rows:
            if realized[category] >= quota:
                break
            iid = str(row["iid"])
            if iid in selected_iids:
                continue
            compatibility = row["qwen_action_anchor"]["compatibility"]
            if compatibility["action_category"] != category:
                continue
            selected.append(row)
            selected_iids.add(iid)
            realized[category] += 1

    shortfall = {
        category: max(0, effective_quotas[category] - realized[category])
        for category in ACTION_CATEGORIES
    }
    for row in review_rows:
        if len(selected) >= proposed_size:
            break
        iid = str(row["iid"])
        if iid in selected_iids:
            continue
        selected.append(row)
        selected_iids.add(iid)
        category = str(
            row["qwen_action_anchor"]["compatibility"]["action_category"]
        )
        realized[category] += 1

    if len(selected) != proposed_size:
        raise GokuActionAnchorFinalizeError(
            "insufficient hard-pass diversity candidates for proposed set: "
            f"required={proposed_size} available={len(selected)}"
        )
    return (
        selected,
        dict(sorted(realized.items())),
        dict(sorted(shortfall.items())),
    )


def _annotated_row(
    row: Mapping[str, Any],
    *,
    review_rank: int,
    bucket: str,
    row_schema: str = ROW_SCHEMA,
    policy_version: str = POLICY_VERSION,
    profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(row)
    qwen = row["qwen_action_anchor"]
    compatibility = qwen["compatibility"]
    judge_a = qwen["target_admissibility"]
    judge_b = qwen["draft_continuity"]
    target_support = compatibility_target_support_evidence(compatibility)
    writer_instruction_support = writer_target_instruction_support_evidence(
        compatibility,
        row,
    )
    target_core_agreement = target_core_agreement_evidence(
        judge_a,
        compatibility,
        row,
    )
    result["action_anchor_finalization"] = {
        "schema_version": row_schema,
        "policy_version": policy_version,
        "hard_gate_passed": True,
        "hard_gate_failures": [],
        "review_rank": review_rank,
        "selection_bucket": bucket,
        "action_category": compatibility["action_category"],
        "target_action_verb": compatibility["target_action_verb"],
        "action_change_substantive": compatibility[
            "action_change_substantive"
        ],
        "human_review_status": "pending",
        "human_label": False,
        "generation_authorized": False,
        "manifest_role": "review_proposal",
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
        "target_support_evidence": target_support,
        "writer_instruction_target_support_evidence": (
            writer_instruction_support
        ),
        "judge_a_writer_target_core_agreement_evidence": (
            target_core_agreement
        ),
        "target_admissibility": judge_a,
        "target_admissibility_aggregate": qwen[
            "target_admissibility_aggregate"
        ],
        "draft_continuity": judge_b,
        "draft_continuity_aggregate": qwen[
            "draft_continuity_aggregate"
        ],
        "deterministic_risk_codes": qwen[
            "deterministic_risk_codes"
        ],
    }
    if profile_metadata is not None:
        result["action_anchor_finalization"]["profile"] = dict(
            profile_metadata
        )
    return result


def _generation_row(
    row: Mapping[str, Any],
    *,
    file_verification: Mapping[str, Any],
    generation_schema: str = GENERATION_SCHEMA,
    policy_version: str = POLICY_VERSION,
    profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    qwen = row["qwen_action_anchor"]
    compatibility = qwen["compatibility"]
    judge_a = qwen["target_admissibility"]
    judge_b = qwen["draft_continuity"]
    media_evidence = dict(
        _mapping(
            row["media"],
            context=f"generation iid={row['iid']} selected media",
        )
    )
    temporal_geometry = _strict_temporal_geometry(
        media_evidence,
        iid=str(row["iid"]),
    )
    generation: dict[str, Any] = {
        "schema_version": generation_schema,
        "iid": row["iid"],
        "group_id": row["group_id"],
        "action_category": compatibility["action_category"],
        # Judge A is independently bound to the immutable instruction and is
        # the only target-semantic payload admitted here.  Writer prose stays
        # in the review/Qwen records and cannot silently become a generation
        # prompt when it omits an actor, object, relation, or direction.
        "target_action_verb": judge_a["target_action_verb"],
        "target_action_normalized": judge_a[
            "target_action_normalized"
        ],
        "target_semantics_source": "judge_a_instruction_bound",
        "action_change_substantive": compatibility[
            "action_change_substantive"
        ],
        "source_video": row["src_video"],
        "resolved_source_video": row["resolved_src_video"],
        "anchor_image": row["anchor_image"],
        "anchor_sha256": row["anchor_sha256"],
        "source_video_sha256": row["source_video_sha256"],
        "selected_media_evidence": media_evidence,
        "selected_media_evidence_sha256": _object_digest(media_evidence),
        "strict_temporal_geometry": temporal_geometry,
        "finalizer_media_file_verification": dict(file_verification),
        # This is the immutable requested edit.  The writer may propose a
        # clearer rendering for review, but it must never replace the frozen
        # instruction that defines the training example.
        "edit_instruction": row["prompt"],
        "edit_instruction_sha256": hashlib.sha256(
            str(row["prompt"]).encode("utf-8")
        ).hexdigest(),
        "instruction_contract": {
            "sole_candidate_instruction_field": "edit_instruction",
            "candidate_instruction_source": "frozen_selected_prompt",
            "writer_proposal_payload_included": False,
            "writer_proposals_executable": False,
            "requires_future_signed_release_verifier": True,
        },
        "source_caption": row["source_caption"],
        "source_edited_caption_provenance": row["edited_caption"],
        "source_edited_caption_provenance_role": (
            "non_executable_provenance"
        ),
        "source_instruction_provenance": row["prompt"],
        "qwen_input_digest": qwen["input_digest"],
        "qwen_config_digest": qwen["config_digest"],
        "manifest_role": "review_proposal",
        "production_eligible": False,
        "human_review_status": "pending",
        "generation_authorized": False,
        "approval": None,
        "authorization_interface_available": False,
        "target_admissibility": judge_a,
        "target_admissibility_aggregate": qwen[
            "target_admissibility_aggregate"
        ],
        "draft_continuity": judge_b,
        "draft_continuity_aggregate": qwen[
            "draft_continuity_aggregate"
        ],
        "deterministic_risk_codes": qwen[
            "deterministic_risk_codes"
        ],
    }
    generation["resolved_anchor_image"] = row["resolved_anchor_image"]
    if profile_metadata is not None:
        generation["policy_version"] = policy_version
        generation["finalization_profile"] = dict(profile_metadata)
    return generation


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _pretty_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _implementation_digest() -> str:
    return _sha256_file(Path(__file__).resolve(strict=True))


def finalize_action_anchors(
    *,
    selected_path: str | Path,
    qwen_root: str | Path,
    output_dir: str | Path,
    approval_path: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    allow_partial: bool = False,
    profile: str | None = None,
) -> dict[str, Any]:
    """Validate, select, and atomically publish an action-anchor proposal."""

    finalization_profile = _resolve_profile(profile)
    profile_metadata = _profile_metadata(finalization_profile)
    if approval_path is not None:
        raise GokuActionAnchorFinalizeError(
            "approval input is forbidden: this finalizer only publishes "
            "pending, production-ineligible review proposals"
        )
    if type(seed) is not int or seed < 0:
        raise GokuActionAnchorFinalizeError(
            "seed must be a non-negative integer"
        )
    if type(allow_partial) is not bool:
        raise GokuActionAnchorFinalizeError(
            "allow_partial must be a boolean"
        )
    selected_rows, selected_raw, selected_resolved = _load_jsonl(
        selected_path,
        context="prefilter selected input",
    )
    selected_by_iid: dict[str, dict[str, Any]] = {}
    verified_media_by_iid: dict[str, dict[str, Any]] = {}
    selected_order: list[str] = []
    selected_sha256 = hashlib.sha256(selected_raw).hexdigest()
    selected_canonical_path = str(selected_resolved)
    for line_number, row in enumerate(selected_rows, start=1):
        iid = _validate_selected_row(row, line_number=line_number)
        if iid in selected_by_iid:
            raise GokuActionAnchorFinalizeError(
                f"duplicate selected IID: {iid}"
            )
        verified_media_by_iid[iid] = _verified_selected_media_files(
            row,
            iid=iid,
        )
        selected_by_iid[iid] = row
        selected_order.append(iid)

    shard_pairs = _discover_shards(qwen_root)
    qwen_by_iid: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    config_digests_by_shard: dict[int, set[str]] = {}
    run_config_digests: set[str] = set()
    implementation_digests: set[str] = set()
    model_paths: set[str] = set()
    model_revisions: set[str] = set()
    transformers_versions: set[str] = set()
    shard_counts: set[int] = set()
    receipt_roots: set[str] = set()
    receipt_run_configs: set[str] = set()
    qwen_implementation_digest = _sha256_file(
        Path(qwen_module.__file__).resolve(strict=True)
    )
    for shard_index, (shard_path, receipt_path) in enumerate(shard_pairs):
        shard_rows, shard_raw, resolved = _load_jsonl(
            shard_path,
            context=f"Qwen shard {shard_index}",
            allow_empty=True,
        )
        receipt = _load_and_validate_shard_receipt(
            receipt_path,
            shard_path=resolved,
            shard_raw=shard_raw,
            shard_rows=shard_rows,
            shard_index=shard_index,
            selected_rows=selected_rows,
            selected_path=selected_resolved,
            selected_sha256=selected_sha256,
            qwen_implementation_digest=qwen_implementation_digest,
        )
        receipt_records.append(
            {
                "index": shard_index,
                "path": receipt["path"],
                "sha256": receipt["file_sha256"],
                "receipt_digest": receipt["receipt_digest"],
                "assigned_rows": len(receipt["assigned_iids"]),
            }
        )
        config_digests_by_shard.setdefault(shard_index, set()).add(
            str(receipt["config_digest"])
        )
        run_config_digests.add(str(receipt["run_config_digest"]))
        implementation_digests.add(str(receipt["implementation_digest"]))
        model_paths.add(str(receipt["model_path"]))
        model_revisions.add(str(receipt["model_revision"]))
        transformers_versions.add(str(receipt["transformers_version"]))
        shard_counts.add(int(receipt["num_shards"]))
        receipt_roots.add(str(receipt["root"]))
        receipt_run_configs.add(_object_digest(receipt["run_config"]))
        actual_shard_iids: list[str] = []
        shard_records.append(
            {
                "index": shard_index,
                "path": str(resolved),
                "rows": len(shard_rows),
                "sha256": hashlib.sha256(shard_raw).hexdigest(),
                "bytes": len(shard_raw),
                "receipt_path": receipt["path"],
                "receipt_sha256": receipt["file_sha256"],
            }
        )
        for line_number, qwen in enumerate(shard_rows, start=1):
            iid = _canonical_string(
                qwen.get("iid"),
                context=(
                    f"Qwen shard {shard_index} row {line_number} iid"
                ),
            )
            actual_shard_iids.append(iid)
            if iid not in selected_by_iid:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen shard {shard_index} has unknown IID: {iid}"
                )
            if iid in qwen_by_iid:
                raise GokuActionAnchorFinalizeError(
                    f"duplicate Qwen IID across shards: {iid}"
                )
            expected_shard = _iid_shard(iid)
            if expected_shard != shard_index:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} is in shard {shard_index}, "
                    f"expected {expected_shard}"
                )
            if qwen.get("shard_index") != shard_index:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} shard_index binding differs"
                )
            if qwen.get("num_shards") != REQUIRED_SHARD_COUNT:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} num_shards binding differs"
                )
            shard_counts.add(int(qwen["num_shards"]))
            if qwen.get("group_id") != selected_by_iid[iid]["group_id"]:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} group_id binding differs"
                )
            if qwen.get("family") != selected_by_iid[iid]["family"]:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} family binding differs"
                )
            expected_input_digest = _object_digest(selected_by_iid[iid])
            actual_input_digest = _sha256_field(
                qwen.get("input_digest"),
                context=f"Qwen iid={iid} input_digest",
            )
            if actual_input_digest != expected_input_digest:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} input_digest binding differs"
                )
            config_digest = _sha256_field(
                qwen.get("config_digest"),
                context=f"Qwen iid={iid} config_digest",
            )
            config_digests_by_shard.setdefault(shard_index, set()).add(
                config_digest
            )
            run_config_digest = _sha256_field(
                qwen.get("run_config_digest"),
                context=f"Qwen iid={iid} run_config_digest",
            )
            run_config_digests.add(run_config_digest)
            implementation_digest = _sha256_field(
                qwen.get("implementation_digest"),
                context=f"Qwen iid={iid} implementation_digest",
            )
            if implementation_digest != qwen_implementation_digest:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} implementation_digest differs "
                    "from the loaded audit implementation"
                )
            implementation_digests.add(implementation_digest)
            execution_manifest_sha256 = _sha256_field(
                qwen.get("execution_manifest_sha256"),
                context=f"Qwen iid={iid} execution_manifest_sha256",
            )
            if execution_manifest_sha256 != selected_sha256:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} execution_manifest_sha256 differs "
                    "from the actual selected input"
                )
            execution_manifest = _canonical_string(
                qwen.get("execution_manifest"),
                context=f"Qwen iid={iid} execution_manifest",
            )
            if execution_manifest != selected_canonical_path:
                raise GokuActionAnchorFinalizeError(
                    f"Qwen iid={iid} execution_manifest path differs "
                    "from the canonical selected input path"
                )
            model_path = _canonical_string(
                qwen.get("model_path"),
                context=f"Qwen iid={iid} model_path",
            )
            model_revision = _canonical_string(
                qwen.get("model_revision"),
                context=f"Qwen iid={iid} model_revision",
            )
            transformers_version = _canonical_string(
                qwen.get("transformers_version"),
                context=f"Qwen iid={iid} transformers_version",
            )
            model_paths.add(model_path)
            model_revisions.add(model_revision)
            transformers_versions.add(transformers_version)
            qwen_by_iid[iid] = qwen
        expected_shard_iids = {
            iid for iid in selected_order if _iid_shard(iid) == shard_index
        }
        actual_shard_iid_set = set(actual_shard_iids)
        if actual_shard_iid_set != expected_shard_iids:
            raise GokuActionAnchorFinalizeError(
                f"Qwen shard {shard_index} IID coverage differs: "
                f"missing={sorted(expected_shard_iids - actual_shard_iid_set)[:20]} "
                f"extra={sorted(actual_shard_iid_set - expected_shard_iids)[:20]}"
            )

    missing = set(selected_by_iid) - set(qwen_by_iid)
    if missing:
        raise GokuActionAnchorFinalizeError(
            f"Qwen shards lack selected IIDs: {sorted(missing)[:20]}"
        )
    nonuniform_shards = {
        index: sorted(values)
        for index, values in config_digests_by_shard.items()
        if len(values) != 1
    }
    if nonuniform_shards:
        raise GokuActionAnchorFinalizeError(
            "Qwen rows within a shard do not share one config_digest: "
            f"{nonuniform_shards}"
        )
    if len(run_config_digests) != 1:
        raise GokuActionAnchorFinalizeError(
            "Qwen rows do not share exactly one run_config_digest: "
            f"{sorted(run_config_digests)}"
        )
    consistency_sets: tuple[tuple[str, set[Any]], ...] = (
        ("implementation_digest", implementation_digests),
        ("model_path", model_paths),
        ("model_revision", model_revisions),
        ("transformers_version", transformers_versions),
        ("num_shards", shard_counts),
    )
    for name, values in consistency_sets:
        if len(values) != 1:
            raise GokuActionAnchorFinalizeError(
                f"Qwen rows do not share exactly one {name}: "
                f"{sorted(values, key=str)}"
            )
    if len(receipt_roots) != 1:
        raise GokuActionAnchorFinalizeError(
            "Qwen receipts do not share exactly one canonical root"
        )
    if len(receipt_run_configs) != 1:
        raise GokuActionAnchorFinalizeError(
            "Qwen receipts do not share exactly one full run_config"
        )
    hard_pass: list[dict[str, Any]] = []
    hard_rejections: Counter[str] = Counter()
    for iid in selected_order:
        selected = dict(selected_by_iid[iid])
        qwen = qwen_by_iid[iid]
        failures = _hard_gate_failures(
            qwen,
            iid=iid,
            selected=selected,
        )
        for failure in failures:
            hard_rejections[failure] += 1
        if failures:
            continue
        selected["qwen_action_anchor"] = qwen
        hard_pass.append(selected)
    ranked = sorted(
        hard_pass,
        key=lambda row: _quality_key(row, seed=seed),
    )
    review, diversity_rejections = _diverse_review_pool(
        ranked,
        limit=finalization_profile.review_limit,
        max_per_target_verb=finalization_profile.max_per_target_verb,
    )
    strict_minimum = (
        finalization_profile.proposed_size
        + finalization_profile.reserve_size
    )
    if not allow_partial and len(review) < strict_minimum:
        raise GokuActionAnchorFinalizeError(
            "insufficient hard-pass review candidates after diversity caps: "
            f"required={strict_minimum} available={len(review)}"
        )
    if allow_partial and not review:
        raise GokuActionAnchorFinalizeError(
            "partial finalization still requires at least one hard-pass "
            "review candidate"
        )
    proposed_target = (
        min(finalization_profile.proposed_size, len(review))
        if allow_partial
        else finalization_profile.proposed_size
    )
    profile_category_quotas = dict(finalization_profile.category_quotas)
    proposed, proposed_categories, quota_shortfall = _select_proposed(
        review,
        proposed_size=proposed_target,
        maximum_proposed_size=finalization_profile.proposed_size,
        full_category_quotas=profile_category_quotas,
    )
    proposed_iids = {str(row["iid"]) for row in proposed}
    reserve_target = (
        min(
            finalization_profile.reserve_size,
            len(review) - len(proposed),
        )
        if allow_partial
        else finalization_profile.reserve_size
    )
    reserve = [
        row for row in review if str(row["iid"]) not in proposed_iids
    ][:reserve_target]
    if len(reserve) != reserve_target:
        raise GokuActionAnchorFinalizeError(
            "insufficient disjoint reserve candidates"
        )
    reserve_iids = {str(row["iid"]) for row in reserve}

    review_index = {
        str(row["iid"]): index
        for index, row in enumerate(review, start=1)
    }
    review_output = [
        _annotated_row(
            row,
            review_rank=review_index[str(row["iid"])],
            bucket=(
                "proposed"
                if str(row["iid"]) in proposed_iids
                else (
                    "reserve"
                    if str(row["iid"]) in reserve_iids
                    else "review_only"
                )
            ),
            row_schema=finalization_profile.row_schema,
            policy_version=finalization_profile.policy_version,
            profile_metadata=profile_metadata,
        )
        for row in review
    ]
    proposed_output = [
        _annotated_row(
            row,
            review_rank=review_index[str(row["iid"])],
            bucket="proposed",
            row_schema=finalization_profile.row_schema,
            policy_version=finalization_profile.policy_version,
            profile_metadata=profile_metadata,
        )
        for row in proposed
    ]
    reserve_output = [
        _annotated_row(
            row,
            review_rank=review_index[str(row["iid"])],
            bucket="reserve",
            row_schema=finalization_profile.row_schema,
            policy_version=finalization_profile.policy_version,
            profile_metadata=profile_metadata,
        )
        for row in reserve
    ]
    proposed_payload = _jsonl_bytes(proposed_output)
    generation_output = [
        _generation_row(
            row,
            file_verification=verified_media_by_iid[str(row["iid"])],
            generation_schema=finalization_profile.generation_schema,
            policy_version=finalization_profile.policy_version,
            profile_metadata=profile_metadata,
        )
        for row in proposed
    ]

    payloads: dict[str, bytes] = {
        REVIEW_NAME: _jsonl_bytes(review_output),
        finalization_profile.proposed_name: proposed_payload,
        finalization_profile.reserve_name: _jsonl_bytes(reserve_output),
        GENERATION_NAME: _jsonl_bytes(generation_output),
    }
    verb_counts = Counter(
        row["qwen_action_anchor"]["compatibility"]["target_action_verb"]
        for row in proposed
    )
    review_category_counts = Counter(
        row["qwen_action_anchor"]["compatibility"]["action_category"]
        for row in review
    )
    reserve_category_counts = Counter(
        row["qwen_action_anchor"]["compatibility"]["action_category"]
        for row in reserve
    )
    summary: dict[str, Any] = {
        "schema_version": finalization_profile.summary_schema,
        "policy_version": finalization_profile.policy_version,
        "seed": seed,
        "input": {
            "selected_path": str(selected_resolved),
            "selected_rows": len(selected_rows),
            "selected_sha256": selected_sha256,
            "qwen_shards": shard_records,
            "qwen_shard_receipts": receipt_records,
            "qwen_config_digest_by_shard": {
                str(index): next(iter(values))
                for index, values in sorted(config_digests_by_shard.items())
            },
            "qwen_run_config_digest": next(iter(run_config_digests)),
            "qwen_implementation_digest": qwen_implementation_digest,
            "qwen_model_path": next(iter(model_paths)),
            "qwen_model_revision": next(iter(model_revisions)),
            "qwen_transformers_version": next(
                iter(transformers_versions)
            ),
            "qwen_num_shards": next(iter(shard_counts)),
            "selected_media_files_reverified": len(
                verified_media_by_iid
            ),
            "selected_media_file_verification_sha256": _object_digest(
                verified_media_by_iid
            ),
        },
        "hard_gate": {
            "passed_rows": len(hard_pass),
            "rejected_rows": len(selected_rows) - len(hard_pass),
            "rejection_counts": dict(sorted(hard_rejections.items())),
        },
        "diversity": {
            "group_id_max": 1,
            "target_verb_max": finalization_profile.max_per_target_verb,
            "review_rejection_counts": dict(
                sorted(diversity_rejections.items())
            ),
        },
        "selection": {
            "mode": (
                f"partial_up_to_{finalization_profile.proposed_size}"
                if allow_partial
                else (
                    f"strict_{finalization_profile.proposed_size}_plus_"
                    f"{finalization_profile.reserve_size}"
                )
            ),
            "allow_partial": allow_partial,
            "requested_proposed_rows": finalization_profile.proposed_size,
            "requested_reserve_rows": finalization_profile.reserve_size,
            "effective_proposed_target": proposed_target,
            "effective_reserve_target": reserve_target,
            "review_rows": len(review_output),
            "proposed_rows": len(proposed_output),
            "reserve_rows": len(reserve_output),
            "generation_rows": len(generation_output),
            "requested_category_quotas": profile_category_quotas,
            "effective_category_quotas": _scaled_category_quotas(
                proposed_target,
                maximum_proposed_size=finalization_profile.proposed_size,
                full_category_quotas=profile_category_quotas,
            ),
            "proposed_category_counts": proposed_categories,
            "quota_shortfall_before_backfill": quota_shortfall,
            "review_category_counts": dict(
                sorted(review_category_counts.items())
            ),
            "reserve_category_counts": dict(
                sorted(reserve_category_counts.items())
            ),
            "proposed_target_verb_counts": dict(sorted(verb_counts.items())),
            "proposal_reserve_disjoint": proposed_iids.isdisjoint(
                reserve_iids
            ),
        },
        "semantics": {
            "manifest_role": "review_proposal",
            "human_review_status": "pending",
            "human_labels_asserted": False,
            "generation_authorized": False,
            "production_eligible": False,
            "approval": None,
            "authorization_interface_available": False,
        },
        "implementation_sha256": _implementation_digest(),
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
    }
    if profile_metadata is not None:
        summary["profile"] = profile_metadata
    summary_payload = _pretty_bytes(summary)
    payloads[SUMMARY_NAME] = summary_payload
    done = {
        "schema_version": finalization_profile.done_schema,
        "status": "complete",
        "summary_sha256": hashlib.sha256(summary_payload).hexdigest(),
        "implementation_sha256": summary["implementation_sha256"],
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
    }
    if profile_metadata is not None:
        done["profile"] = profile_metadata
        done["profile_sha256"] = _object_digest(profile_metadata)
    done_payload = _pretty_bytes(done)

    output = Path(output_dir).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        for name in (
            REVIEW_NAME,
            finalization_profile.proposed_name,
            finalization_profile.reserve_name,
            GENERATION_NAME,
            SUMMARY_NAME,
        ):
            _write_new(staging / name, payloads[name])
        _write_new(staging / DONE_NAME, done_payload)
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly merge eight Goku anchor Qwen shards and publish "
            "action-editing candidate artifacts under the default or an "
            "explicit scale profile."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--qwen-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--profile",
        choices=(SCALE512_PROFILE,),
        default=None,
        help=(
            "Explicit scale profile. Omit this option to preserve the "
            "legacy 192/128/32 contract and artifact names."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Publish every strict hard-pass candidate up to the profile's "
            "proposed limit instead of "
            "requiring the selected profile's full proposed plus reserve "
            "counts."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = finalize_action_anchors(
        selected_path=args.input,
        qwen_root=args.qwen_root,
        output_dir=args.output_dir,
        seed=args.seed,
        allow_partial=args.allow_partial,
        profile=args.profile,
    )
    print(
        "[goku-action-anchor-finalize] "
        f"review={summary['selection']['review_rows']} "
        f"proposed={summary['selection']['proposed_rows']} "
        f"reserve={summary['selection']['reserve_rows']} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

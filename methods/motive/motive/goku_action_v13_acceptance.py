"""Independent, fail-closed acceptance gate for the frozen v16 Goku smoke.

This module intentionally does not import the Qwen producer or finalizer.  It
recomputes their byte, prompt, route, receipt, result, provenance, and pending
proposal contracts from standard-library primitives and source-snapshot bytes.
Passing this smoke may authorize the full 123-row Qwen audit.  It never
authorizes Wan generation.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata


ACCEPTANCE_CONTRACT_SCHEMA = (
    "motive-goku-action-v16-acceptance-contract-v1"
)
SUBMISSION_CONTRACT_SCHEMA = (
    "motive-goku-action-v16-submission-contract-v1"
)
COMPLETION_RECEIPT_SCHEMA = (
    "motive-goku-action-v16-completion-receipt-v1"
)
ACCEPTANCE_RESULT_SCHEMA = "motive-goku-action-v16-acceptance-result-v1"
SMOKE_GOLD_SCHEMA = "goku-action-v16-smoke-gold-v1"
SMOKE_GOLD_CANONICAL_RELPATH = Path(
    "methods/motive/audits/goku_action_v16_smoke_gold.json"
)
TARGET_CONTRACT_SCHEMA = "goku-action-v16-target-contract-v1"

ANCHOR_OBSERVATION_SCHEMA = "goku-action-anchor-observation-v1"
ANCHOR_COMPATIBILITY_SCHEMA = "goku-action-anchor-compatibility-v1"
TARGET_ADMISSIBILITY_SCHEMA = (
    "goku-action-anchor-target-admissibility-v8"
)
DRAFT_CONTINUITY_SCHEMA = "goku-action-anchor-draft-continuity-v8"
JUDGE_AGGREGATE_SCHEMA = "goku-action-anchor-judge-aggregate-v8"
QWEN_PROVENANCE_SCHEMA = "goku-action-anchor-qwen-provenance-v8"
SHARD_RECEIPT_SCHEMA = "goku-action-anchor-shard-receipt-v8"
FINAL_ROW_SCHEMA = "motive-goku-action-anchor-final-row-v8"
GENERATION_SCHEMA = "motive-goku-action-anchor-generation-v9"
SUMMARY_SCHEMA = "motive-goku-action-anchor-finalize-v8"
DONE_SCHEMA = "motive-goku-action-anchor-finalize-done-v8"
POLICY_VERSION = "goku-action-anchor-strict-continuity-v8"

FINAL_NAMES = (
    "review_candidates.jsonl",
    "proposed_128.jsonl",
    "reserve_32.jsonl",
    "generation_manifest.jsonl",
    "summary.json",
    "done.json",
)
JSONL_FINAL_NAMES = FINAL_NAMES[:4]
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
# Filled only after the separately reviewed gold file is frozen.  A caller,
# contract, or completion receipt cannot replace this source-level trust
# anchor.  Tests patch it to the digest of their closed fixture.
EXPECTED_SMOKE_GOLD_SHA256 = (
    "b99972b81139e7a3193e6589efdf8de38075102cf14f312e3e1e73dfc3d626df"
)
EXPECTED_QWEN_IMPLEMENTATION_SHA256 = (
    "f5535e0f68e515609a1b578b494197ae0c45a5ca79030ba9ceaa25ba0d7b772e"
)
EXPECTED_FINALIZER_IMPLEMENTATION_SHA256 = (
    "63d98952f400dd30a069fee72f169a2d512b8d3b0b9b7c4779475663e26758e3"
)
EXPECTED_SBATCH_SHA256 = (
    "fff73cd87643c1b069a9ae3f118a678b410cf00007b2da4e3c26ef968cb7871d"
)
EXPECTED_MODEL_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/"
    "MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct"
)
EXPECTED_MODEL_CONFIG_SHA256 = (
    "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
)
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
EXPECTED_MODEL_REVISION = "Qwen3-VL-32B-Instruct"
MODEL_CLOSURE_CANONICAL_RELPATH = Path(
    "methods/motive/audits/qwen3_vl_32b_instruct_model_closure.json"
)
MODEL_CLOSURE_SCHEMA = "motive-qwen-model-closure-v1"
EXPECTED_MODEL_CLOSURE_SHA256 = (
    "395236b156d85409ca40643683b47b1badb28602df0ef41e519e50f9a60f6c05"
)
EXPECTED_MODEL_CLOSURE_FILE_COUNT = 54
EXPECTED_MODEL_CLOSURE_TOTAL_BYTES = 66_726_522_473

SOURCE_SNAPSHOT_KEYS = {
    "path",
    "tree_sha256",
    "manifest_path",
    "manifest_sha256",
    "archive_path",
    "archive_sha256",
    "qwen_relpath",
    "qwen_implementation_sha256",
    "finalizer_relpath",
    "finalizer_implementation_sha256",
    "verifier_relpath",
    "verifier_implementation_sha256",
    "sbatch_relpath",
    "sbatch_sha256",
}
MODEL_KEYS = {"path", "config_path", "config_sha256"}
MODEL_CLOSURE_KEYS = {"path", "sha256", "file_count", "total_bytes"}
QWEN_EXECUTION_KEYS = {
    "num_shards",
    "max_samples",
    "max_new_tokens",
    "nframes",
    "max_pixels",
    "attn_implementation",
    "allow_download",
    "repair_attempts",
}
EXPECTED_QWEN_EXECUTION = {
    "num_shards": 8,
    "max_samples": None,
    "max_new_tokens": 1536,
    "nframes": 12,
    "max_pixels": 589_824,
    "attn_implementation": "sdpa",
    "allow_download": False,
    "repair_attempts": 1,
}
EXPECTED_FINAL = {
    "seed": 260730,
    "allow_partial": True,
    "manifest_role": "review_proposal",
    "human_review_status": "pending",
    "generation_authorized": False,
    "production_eligible": False,
    "wan_generation_authorized": False,
}

OBSERVATION_KEYS = {
    "schema_version",
    "source_quality",
    "resolution_quality",
    "initial_state_clarity",
    "subject_visibility",
    "initial_state",
    "visible_entities",
    "interaction_affordances",
    "source_action",
    "actor_motion",
    "motion_dynamics",
    "camera_motion",
    "background_motion",
    "single_continuous_shot",
    "artifact_level",
    "temporal_evidence",
    "uncertainty_codes",
}
JUDGE_A_KEYS = {
    "schema_version",
    "target_change_class",
    "source_target_relation",
    "target_action_normalized",
    "target_action_verb",
    "target_already_true",
    "target_start_state_visually_verifiable",
    "prerequisite_grounded",
    "novel_trajectory",
    "novel_trajectory_description",
    "scalar_or_endpoint_only",
    "source_evidence_ref",
    "target_evidence_ref",
    "uncertainty_codes",
    "confidence",
}
COMPATIBILITY_KEYS = {
    "schema_version",
    "decision",
    "anchor_compatibility",
    "caption_consistency",
    "source_action_normalized",
    "target_action_normalized",
    "target_action_verb",
    "action_change_substantive",
    "action_category",
    "required_entities",
    "prerequisites_visible_at_i0",
    "target_presupposes_prior_action",
    "causal_bridge",
    "causal_bridge_description",
    "causal_stages",
    "complete_within_clip",
    "rewritten_edit_instruction",
    "absolute_target_prompt",
    "preservation_constraints",
    "unrequested_changes",
    "reason_codes",
    "uncertainty_codes",
    "confidence",
}
JUDGE_B_KEYS = {
    "schema_version",
    "continuity_mode",
    "target_dominance",
    "actor_entity_consistency",
    "direction_state_consistency",
    "unrequested_action",
    "source_replay_ref",
    "target_support_ref",
    "uncertainty_codes",
    "confidence",
}
TARGET_CORE_FIELDS = (
    "source_action_normalized",
    "target_action_normalized",
    "target_action_verb",
    "action_change_substantive",
    "action_category",
    "required_entities",
    "prerequisites_visible_at_i0",
    "target_presupposes_prior_action",
    "complete_within_clip",
)
TARGET_ATOMIC_FIELDS = (
    "target_already_true",
    "target_start_state_visually_verifiable",
    "prerequisite_grounded",
    "novel_trajectory",
    "scalar_or_endpoint_only",
)
EXPECTED_SEMANTIC_CONTRACT_POLICY = {
    "instruction_hash_algorithm": "sha256_utf8_exact_string_no_newline",
    "target_semantic_text_fields": [
        "target_action_normalized",
        "target_action_verb",
        "novel_trajectory_description",
    ],
    "token_normalization": (
        "unicode_nfkc_casefold_punctuation_hyphen_underscore_to_space"
    ),
    "token_group_semantics": (
        "all_groups_required_any_of_contiguous_token_sequence"
    ),
    "atomic_tuple_comparison": "exact",
    "class_relation_comparison": "exact",
    "semantic_source": "signed_raw_judge_a_only",
    "free_form_sentence_equality_forbidden": True,
}

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ACTION_VERB_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_PLACEHOLDER_RE = re.compile(
    r"(?:<[^<>\n]+>|\b(?:placeholder|short string|describe here)\b)",
    flags=re.IGNORECASE,
)

QUALITY = {"high", "acceptable", "poor", "unclear"}
CLARITY = {"clear", "partial", "poor", "unclear"}
MOTION_LEVELS = {"clear", "weak", "none", "unclear"}
MOTION_DYNAMICS = {"strong", "moderate", "weak", "none", "unclear"}
SCENE_MOTION = {"none", "weak", "strong", "dominant", "unclear"}
YES_NO_UNCLEAR = {"yes", "no", "unclear"}
ARTIFACT_LEVELS = {"none", "low", "medium", "high", "unclear"}
DECISIONS = {"accept", "rewrite", "reject", "unclear"}
ANCHOR_COMPATIBILITY = {
    "compatible", "repairable", "incompatible", "unclear"
}
CAPTION_CONSISTENCY = {
    "consistent", "repairable", "contradictory", "unclear"
}
CAUSAL_BRIDGES = {"direct", "requires_transition", "impossible", "unclear"}
ACTION_CATEGORY_VALUES = {
    "locomotion", "posture", "interaction", "articulated", "unclear"
}
CONFIDENCE = {"low", "medium", "high"}
TARGET_CHANGE_TYPES = {
    "formation_trajectory",
    "relational_locomotion_trajectory",
    "new_articulated_action",
    "new_posture_transition",
    "new_interaction_action",
    "new_direction_trajectory",
    "other_new_trajectory",
    "same_action_intensity_only",
    "same_action_endpoint_or_phase_only",
    "appearance_content_state_only",
    "object_orientation_state_only",
    "source_action_restatement",
    "unclear",
}
SOURCE_TARGET_RELATIONS = {
    "novel_future",
    "shared_base_with_novel_action",
    "later_source_phase_or_endpoint",
    "repeats_source_future",
    "same_action_scalar_only",
    "state_or_appearance_only",
    "unclear",
}
CONTINUITY_MODES = {
    "clean_direct",
    "repairable_source_preface",
    "source_dominant_or_target_changed",
    "unclear",
}
TARGET_DOMINANCE = {
    "dominant", "present_but_diluted", "absent_or_changed", "unclear"
}
ACTOR_ENTITY_CONSISTENCY = {"consistent", "conflict", "unclear"}
DIRECTION_STATE_CONSISTENCY = {"consistent", "conflict", "unclear"}
UNREQUESTED_ACTION = {"none", "present", "unclear"}
SOURCE_EVIDENCE_REF_RE = re.compile(
    r"^(?:initial_state|source_action|"
    r"temporal_evidence:(?:0|[1-9][0-9]*))$"
)
TARGET_EVIDENCE_REFS = {"instruction"}
DRAFT_EVIDENCE_REF_RE = re.compile(
    r"^(?:rewritten_edit_instruction|causal_bridge_description|"
    r"absolute_target_prompt|causal_stages:(?:0|[1-9][0-9]*))$"
)


class AcceptanceError(ValueError):
    """A deterministic contract or artifact assertion failed."""


class UnauditableError(AcceptanceError):
    """Required bytes or a closed contract could not be audited."""


def _reject_constant(value: str) -> None:
    raise UnauditableError(f"non-finite JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UnauditableError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise UnauditableError(f"non-canonical JSON value: {error}") from error


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _semantic_contract_tokens(value: str) -> list[str]:
    """Apply the frozen v16 gold token normalization exactly.

    NFKC normalization and Unicode case folding happen before every Unicode
    punctuation character (including hyphens and underscore) is converted to
    a separator.  Whitespace is then collapsed by ``split``.  No stemming,
    synonym expansion, bag-of-words matching, or free-form sentence equality
    is permitted by the gold contract.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    separated = "".join(
        " "
        if character == "_" or unicodedata.category(character).startswith("P")
        else character
        for character in normalized
    )
    return separated.split()


def _contains_contiguous_tokens(
    haystack: Sequence[str],
    needle: Sequence[str],
) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        list(haystack[index : index + width]) == list(needle)
        for index in range(len(haystack) - width + 1)
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_loads(raw: bytes, *, context: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnauditableError(f"{context}: not UTF-8") from error
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, UnauditableError):
            raise
        raise UnauditableError(f"{context}: invalid strict JSON: {error}") from error


def _regular_file(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise UnauditableError(
            f"{context}: expected regular non-symlink file: {expanded}"
        )
    return expanded.resolve(strict=True)


def _directory(path: Path, *, context: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise UnauditableError(
            f"{context}: expected non-symlink directory: {expanded}"
        )
    return expanded.resolve(strict=True)


def _load_json(path: Path, *, context: str) -> tuple[dict[str, Any], bytes, Path]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    value = _strict_loads(raw, context=context)
    if not isinstance(value, dict):
        raise UnauditableError(f"{context}: top level is not an object")
    return value, raw, resolved


def _require_compact_json_bytes(
    value: Mapping[str, Any],
    raw: bytes,
    *,
    context: str,
) -> None:
    _assert_equal(raw, _canonical_bytes(value) + b"\n", f"{context} bytes")


def _load_jsonl(
    path: Path,
    *,
    context: str,
    allow_empty: bool = False,
) -> tuple[list[dict[str, Any]], bytes, Path]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise UnauditableError(f"{context}: missing terminal newline")
    if not raw and not allow_empty:
        raise UnauditableError(f"{context}: empty file")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise UnauditableError(f"{context}:{index}: blank line")
        value = _strict_loads(line, context=f"{context}:{index}")
        if not isinstance(value, dict):
            raise UnauditableError(f"{context}:{index}: row is not an object")
        rows.append(value)
    return rows, raw, resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise UnauditableError(
            f"{context}: closed keys differ: {sorted(set(value) ^ expected)}"
        )


def _sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise UnauditableError(f"{context}: invalid lowercase SHA-256")
    return value


def _nonempty(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise UnauditableError(f"{context}: expected canonical non-empty string")
    return value


def _literal_text(
    value: Any,
    context: str,
    *,
    allow_sentinel: bool = False,
) -> str:
    text = _nonempty(value, context)
    if _PLACEHOLDER_RE.search(text):
        raise AcceptanceError(f"{context}: unresolved placeholder")
    if not allow_sentinel and text.casefold() in {
        "unknown",
        "unclear",
        "unavailable",
        "none",
        "n/a",
    }:
        raise AcceptanceError(f"{context}: lacks literal evidence")
    return text


def _path_text(value: Any, context: str) -> str:
    text = _nonempty(value, context)
    if not Path(text).is_absolute():
        raise UnauditableError(f"{context}: path is not absolute")
    return text


def _assert_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise AcceptanceError(
            f"{context}: expected {expected!r}, observed {actual!r}"
        )


def _strict_object_from_raw(raw: Any, *, context: str) -> dict[str, Any]:
    """Independently reproduce the producer's raw-response object parser."""

    if not isinstance(raw, str):
        raise AcceptanceError(f"{context}: raw response is not text")
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    def parse(text: str) -> Any:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )

    try:
        value = parse(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise AcceptanceError(f"{context}: raw response has no JSON object")
        try:
            value = parse(cleaned[start : end + 1])
        except (json.JSONDecodeError, ValueError) as error:
            raise AcceptanceError(f"{context}: raw response is invalid") from error
    if not isinstance(value, dict):
        raise AcceptanceError(f"{context}: parsed raw response is not an object")
    return value


def _raw_equals(raw: Any, value: Any, *, context: str) -> None:
    parsed = _strict_object_from_raw(raw, context=context)
    if parsed != value:
        raise AcceptanceError(f"{context}: parsed raw object differs")


def _iid_shard(iid: str) -> int:
    prefix = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:16]
    return int(prefix, 16) % 8


def _ordered_iids_digest(iids: Sequence[str]) -> str:
    return _text_digest("".join(f"{iid}\n" for iid in iids))


def _iid_set_digest(iids: Sequence[str]) -> str:
    return _ordered_iids_digest(sorted(iids))


def _validate_target_contract(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UnauditableError(f"{context} is not an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "instruction_sha256",
            "expected_target_change_class",
            "expected_source_target_relation",
            "expected_atomic_tuple",
            "target_token_groups",
        },
        context,
    )
    _assert_equal(
        value["schema_version"],
        TARGET_CONTRACT_SCHEMA,
        f"{context}.schema_version",
    )
    _sha(value["instruction_sha256"], f"{context}.instruction_sha256")
    if value["expected_target_change_class"] not in TARGET_CHANGE_TYPES:
        raise UnauditableError(
            f"{context}.expected_target_change_class is outside enum"
        )
    if value["expected_source_target_relation"] not in SOURCE_TARGET_RELATIONS:
        raise UnauditableError(
            f"{context}.expected_source_target_relation is outside enum"
        )
    atomic = value["expected_atomic_tuple"]
    if not isinstance(atomic, dict):
        raise UnauditableError(f"{context}.expected_atomic_tuple is not object")
    _exact_keys(
        atomic,
        set(TARGET_ATOMIC_FIELDS),
        f"{context}.expected_atomic_tuple",
    )
    for field in TARGET_ATOMIC_FIELDS:
        if atomic[field] not in YES_NO_UNCLEAR:
            raise UnauditableError(
                f"{context}.expected_atomic_tuple.{field} is outside enum"
            )

    groups = value["target_token_groups"]
    if not isinstance(groups, list) or not groups:
        raise UnauditableError(f"{context}.target_token_groups is empty")
    seen_group_ids: set[str] = set()
    for group_index, group in enumerate(groups):
        group_context = f"{context}.target_token_groups[{group_index}]"
        if not isinstance(group, dict):
            raise UnauditableError(f"{group_context} is not an object")
        _exact_keys(group, {"group_id", "any_of"}, group_context)
        group_id = _nonempty(group["group_id"], f"{group_context}.group_id")
        if _ACTION_VERB_RE.fullmatch(group_id) is None:
            raise UnauditableError(
                f"{group_context}.group_id is not lowercase snake_case"
            )
        if group_id in seen_group_ids:
            raise UnauditableError(
                f"{context} has duplicate target token group {group_id!r}"
            )
        seen_group_ids.add(group_id)
        alternatives = group["any_of"]
        if not isinstance(alternatives, list) or not alternatives:
            raise UnauditableError(f"{group_context}.any_of is empty")
        seen_alternatives: set[tuple[str, ...]] = set()
        for alternative_index, alternative in enumerate(alternatives):
            alternative_context = (
                f"{group_context}.any_of[{alternative_index}]"
            )
            if not isinstance(alternative, list) or not alternative:
                raise UnauditableError(
                    f"{alternative_context} is not a nonempty token sequence"
                )
            tokens: list[str] = []
            for token_index, token in enumerate(alternative):
                token_context = f"{alternative_context}[{token_index}]"
                literal = _nonempty(token, token_context)
                normalized = _semantic_contract_tokens(literal)
                if normalized != [literal]:
                    raise UnauditableError(
                        f"{token_context} is not one canonical normalized token"
                    )
                tokens.append(literal)
            frozen = tuple(tokens)
            if frozen in seen_alternatives:
                raise UnauditableError(
                    f"{group_context} has duplicate token alternative"
                )
            seen_alternatives.add(frozen)
    return value


def _load_smoke_gold(
    path: Path,
) -> tuple[dict[str, Any], bytes, Path, dict[str, dict[str, Any]]]:
    """Load the externally reviewed oracle under a source-level SHA anchor."""

    value, raw, resolved = _load_json(path, context="smoke gold")
    canonical_parts = SMOKE_GOLD_CANONICAL_RELPATH.parts
    if resolved.parts[-len(canonical_parts) :] != canonical_parts:
        raise UnauditableError(
            "smoke gold path is not the canonical v16 relative path"
        )
    if _SHA_RE.fullmatch(EXPECTED_SMOKE_GOLD_SHA256) is None:
        raise UnauditableError(
            "smoke gold source trust anchor is not frozen"
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    _assert_equal(
        actual_sha,
        EXPECTED_SMOKE_GOLD_SHA256,
        "smoke gold non-resignable source trust anchor",
    )
    _exact_keys(
        value,
        {
            "schema_version",
            "gold_authority",
            "review_method",
            "reviewed_at_utc",
            "policy",
            "semantic_contract_policy",
            "parent_selected",
            "selected_smoke",
            "labels",
            "quarantine_stress_iids_not_in_gating_smoke",
        },
        "smoke gold",
    )
    _assert_equal(
        value["schema_version"],
        SMOKE_GOLD_SCHEMA,
        "smoke gold schema",
    )
    for field in ("gold_authority", "review_method", "reviewed_at_utc"):
        _literal_text(value[field], f"smoke gold.{field}")
    _assert_equal(
        value["gold_authority"],
        "codex_visual_audit_not_generation_approval",
        "smoke gold authority",
    )
    policy = value["policy"]
    if not isinstance(policy, Mapping):
        raise UnauditableError("smoke gold.policy is not an object")
    _exact_keys(
        policy,
        {
            "admissible",
            "inadmissible",
            "writer_route_is_not_a_gold_label",
            "positive_acceptance",
            "negative_acceptance",
            "wan_generation_authorized",
        },
        "smoke gold.policy",
    )
    for field in (
        "admissible",
        "inadmissible",
        "positive_acceptance",
        "negative_acceptance",
    ):
        _literal_text(policy[field], f"smoke gold.policy.{field}")
    _assert_equal(
        policy["writer_route_is_not_a_gold_label"],
        True,
        "smoke gold route policy",
    )
    _assert_equal(
        policy["wan_generation_authorized"],
        False,
        "smoke gold Wan policy",
    )
    semantic_policy = value["semantic_contract_policy"]
    if not isinstance(semantic_policy, Mapping):
        raise UnauditableError(
            "smoke gold.semantic_contract_policy is not an object"
        )
    _exact_keys(
        semantic_policy,
        set(EXPECTED_SEMANTIC_CONTRACT_POLICY),
        "smoke gold.semantic_contract_policy",
    )
    _assert_equal(
        dict(semantic_policy),
        EXPECTED_SEMANTIC_CONTRACT_POLICY,
        "smoke gold semantic contract policy",
    )
    parent = value["parent_selected"]
    if not isinstance(parent, Mapping):
        raise UnauditableError("smoke gold.parent_selected is not an object")
    _exact_keys(
        parent, {"path", "sha256", "rows", "bytes"}, "smoke gold.parent_selected"
    )
    _path_text(parent["path"], "smoke gold.parent_selected.path")
    _sha(parent["sha256"], "smoke gold.parent_selected.sha256")
    for field in ("rows", "bytes"):
        if type(parent[field]) is not int or parent[field] < 1:
            raise UnauditableError(
                f"smoke gold.parent_selected.{field} is invalid"
            )

    selected = value["selected_smoke"]
    if not isinstance(selected, Mapping):
        raise UnauditableError("smoke gold.selected_smoke is not an object")
    _exact_keys(
        selected,
        {
            "relative_path",
            "sha256",
            "rows",
            "bytes",
            "ordered_iids_sha256",
            "iid_set_sha256",
            "num_shards",
            "expected_shard_rows",
        },
        "smoke gold.selected_smoke",
    )
    relative = _nonempty(
        selected["relative_path"], "smoke gold.selected_smoke.relative_path"
    )
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise UnauditableError("smoke gold selected relative_path is unsafe")
    for field in ("sha256", "ordered_iids_sha256", "iid_set_sha256"):
        _sha(selected[field], f"smoke gold.selected_smoke.{field}")
    if type(selected["rows"]) is not int or selected["rows"] < 1:
        raise UnauditableError("smoke gold selected row count is invalid")
    if type(selected["bytes"]) is not int or selected["bytes"] < 1:
        raise UnauditableError("smoke gold selected byte count is invalid")
    _assert_equal(
        selected["num_shards"], 8, "smoke gold selected num_shards"
    )
    shard_rows = selected["expected_shard_rows"]
    if (
        not isinstance(shard_rows, list)
        or len(shard_rows) != 8
        or not all(type(item) is int and item >= 0 for item in shard_rows)
        or sum(shard_rows) != selected["rows"]
    ):
        raise UnauditableError(
            "smoke gold selected expected_shard_rows is invalid"
        )

    labels = value["labels"]
    if not isinstance(labels, list) or len(labels) != selected["rows"]:
        raise UnauditableError("smoke gold labels row count differs")
    by_iid: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(labels):
        context = f"smoke gold.labels[{index}]"
        if not isinstance(item, dict):
            raise UnauditableError(f"{context} is not an object")
        _exact_keys(
            item,
            {
                "iid",
                "label",
                "target_contract",
                "reason_code",
                "visual_evidence",
            },
            context,
        )
        iid = _nonempty(item["iid"], f"{context}.iid")
        if iid in by_iid:
            raise UnauditableError(f"smoke gold duplicate IID: {iid}")
        if item["label"] not in {"admissible", "inadmissible"}:
            raise UnauditableError(f"{context}.label is outside binary enum")
        reason = _nonempty(item["reason_code"], f"{context}.reason_code")
        if _ACTION_VERB_RE.fullmatch(reason) is None:
            raise UnauditableError(f"{context}.reason_code is not snake_case")
        _validate_target_contract(
            item["target_contract"],
            context=f"{context}.target_contract",
        )
        _literal_text(item["visual_evidence"], f"{context}.visual_evidence")
        by_iid[iid] = item

    quarantine = value["quarantine_stress_iids_not_in_gating_smoke"]
    if not isinstance(quarantine, list):
        raise UnauditableError("smoke gold quarantine set is not a list")
    quarantine_iids: set[str] = set()
    for index, item in enumerate(quarantine):
        context = f"smoke gold.quarantine[{index}]"
        if not isinstance(item, Mapping):
            raise UnauditableError(f"{context} is not an object")
        _exact_keys(item, {"iid", "reason"}, context)
        iid = _nonempty(item["iid"], f"{context}.iid")
        if iid in quarantine_iids:
            raise UnauditableError(f"smoke gold duplicate quarantine IID: {iid}")
        _literal_text(item["reason"], f"{context}.reason")
        quarantine_iids.add(iid)
    if quarantine_iids & set(by_iid):
        raise AcceptanceError(
            "smoke gold quarantine IIDs overlap the gating labels"
        )
    return value, raw, resolved, by_iid


def _bind_gold_to_selected(
    gold: Mapping[str, Any],
    labels_by_iid: Mapping[str, Mapping[str, Any]],
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    selected_raw: bytes,
) -> dict[str, Any]:
    binding = gold["selected_smoke"]
    selected_iids = [str(row["iid"]) for row in selected_rows]
    label_iids = [str(item["iid"]) for item in gold["labels"]]
    _assert_equal(label_iids, selected_iids, "smoke gold label/selected order")
    _assert_equal(
        set(labels_by_iid), set(selected_iids), "smoke gold IID closure"
    )
    for index, row in enumerate(selected_rows):
        iid = selected_iids[index]
        prompt = _literal_text(
            row.get("prompt"), f"selected iid={iid}.prompt"
        )
        target_contract = labels_by_iid[iid]["target_contract"]
        _assert_equal(
            target_contract["instruction_sha256"],
            _text_digest(prompt),
            f"smoke gold iid={iid} immutable instruction SHA",
        )
    selected_sha = hashlib.sha256(selected_raw).hexdigest()
    ordered_sha = _ordered_iids_digest(selected_iids)
    set_sha = _iid_set_digest(selected_iids)
    expected = {
        "sha256": selected_sha,
        "rows": len(selected_rows),
        "bytes": len(selected_raw),
        "ordered_iids_sha256": ordered_sha,
        "iid_set_sha256": set_sha,
        "num_shards": 8,
        "expected_shard_rows": [
            sum(_iid_shard(iid) == index for iid in selected_iids)
            for index in range(8)
        ],
    }
    for field, actual in expected.items():
        _assert_equal(
            binding[field], actual, f"smoke gold selected binding {field}"
        )
    return expected


def _verify_gold_target_semantics(
    *,
    record: Mapping[str, Any],
    selected: Mapping[str, Any],
    gold_label: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the immutable gold target contract against signed raw Judge A.

    Binary routing is deliberately insufficient: the raw Judge-A object must
    classify the exact target, relation, and five-field atomic tuple expected
    by the human visual audit.  Its target text must also cover every frozen
    token group.  Groups are ANDed, alternatives within a group are ORed, and
    each alternative is matched only as a contiguous normalized token
    sequence inside one of the three policy-named target-semantic fields.
    """

    iid = str(selected["iid"])
    contract = _validate_target_contract(
        gold_label["target_contract"],
        context=f"smoke gold iid={iid}.target_contract",
    )
    instruction = _literal_text(
        selected.get("prompt"), f"selected iid={iid}.prompt"
    )
    _assert_equal(
        contract["instruction_sha256"],
        _text_digest(instruction),
        f"smoke gold iid={iid} immutable instruction SHA",
    )

    raw_judge = _strict_object_from_raw(
        record.get("target_admissibility_raw"),
        context=f"Qwen iid={iid}.signed raw Judge-A",
    )
    _assert_equal(
        raw_judge,
        record.get("target_admissibility"),
        f"Qwen iid={iid}.signed raw Judge-A object binding",
    )
    _validate_judge_a(raw_judge, f"Qwen iid={iid}.signed raw Judge-A")

    _assert_equal(
        raw_judge["target_change_class"],
        contract["expected_target_change_class"],
        f"Qwen iid={iid}.gold target_change_class",
    )
    _assert_equal(
        raw_judge["source_target_relation"],
        contract["expected_source_target_relation"],
        f"Qwen iid={iid}.gold source_target_relation",
    )
    observed_atomic = {
        field: raw_judge[field] for field in TARGET_ATOMIC_FIELDS
    }
    _assert_equal(
        observed_atomic,
        contract["expected_atomic_tuple"],
        f"Qwen iid={iid}.gold atomic tuple",
    )

    text_fields = EXPECTED_SEMANTIC_CONTRACT_POLICY[
        "target_semantic_text_fields"
    ]
    token_fields = {
        field: _semantic_contract_tokens(str(raw_judge[field]))
        for field in text_fields
    }
    matched_groups: dict[str, dict[str, Any]] = {}
    missing_groups: list[str] = []
    for group in contract["target_token_groups"]:
        group_id = str(group["group_id"])
        match: dict[str, Any] | None = None
        for alternative in group["any_of"]:
            for field in text_fields:
                if _contains_contiguous_tokens(
                    token_fields[field], alternative
                ):
                    match = {
                        "field": field,
                        "tokens": list(alternative),
                    }
                    break
            if match is not None:
                break
        if match is None:
            missing_groups.append(group_id)
        else:
            matched_groups[group_id] = match
    if missing_groups:
        raise AcceptanceError(
            f"Qwen iid={iid}.gold target token groups missing contiguous "
            f"semantic support: {missing_groups}"
        )
    return {
        "instruction_sha256": contract["instruction_sha256"],
        "target_change_class": raw_judge["target_change_class"],
        "source_target_relation": raw_judge["source_target_relation"],
        "atomic_tuple": observed_atomic,
        "matched_target_token_groups": matched_groups,
    }


def _safe_ast_eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(node.id)
        return env[node.id]
    if isinstance(node, ast.Dict):
        return {
            _safe_ast_eval(key, env): _safe_ast_eval(value, env)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_safe_ast_eval(item, env) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_ast_eval(item, env) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_ast_eval(item, env) for item in node.elts}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_ast_eval(node.left, env) + _safe_ast_eval(node.right, env)
    if isinstance(node, ast.Call):
        args = [_safe_ast_eval(item, env) for item in node.args]
        if isinstance(node.func, ast.Name) and node.func.id == "sorted":
            return sorted(*args)
        if isinstance(node.func, ast.Name) and node.func.id == "frozenset":
            return frozenset(*args)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "join"
        ):
            receiver = _safe_ast_eval(node.func.value, env)
            return receiver.join(*args)
    raise ValueError(ast.dump(node, include_attributes=False))


def _source_constants(qwen_path: Path) -> dict[str, Any]:
    try:
        tree = ast.parse(qwen_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise UnauditableError(f"cannot parse frozen Qwen source: {error}") from error
    env: dict[str, Any] = {}
    for statement in tree.body:
        target: ast.Name | None = None
        value_node: ast.AST | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0]
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target = statement.target
            value_node = statement.value
        if target is None or value_node is None:
            continue
        try:
            env[target.id] = _safe_ast_eval(value_node, env)
        except (TypeError, ValueError):
            continue
    required = {
        "BLIND_SYSTEM",
        "BLIND_PROMPT",
        "COMPATIBILITY_SYSTEM",
        "COMPATIBILITY_PROMPT",
        "JUDGE_A_SYSTEM",
        "JUDGE_A_PROMPT",
        "JUDGE_B_SYSTEM",
        "JUDGE_B_PROMPT",
        "DRAFT_REPAIR_SYSTEM",
        "DRAFT_REPAIR_PROMPT",
        "ANCHOR_OBSERVATION_REPAIR_SCHEMA",
        "COMPATIBILITY_REPAIR_SCHEMA",
        "TARGET_ADMISSIBILITY_REPAIR_SCHEMA",
        "DRAFT_CONTINUITY_REPAIR_SCHEMA",
    }
    missing = sorted(required - set(env))
    if missing:
        raise UnauditableError(
            f"frozen Qwen source constants are incomplete: {missing}"
        )
    return env


def _render_context(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "source_caption": json.dumps(
            str(row["source_caption"]), ensure_ascii=False
        ),
        "edited_caption": json.dumps(
            str(row["edited_caption"]), ensure_ascii=False
        ),
        "instruction": json.dumps(str(row["prompt"]), ensure_ascii=False),
    }


def _render_prompt(
    kind: str,
    constants: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
    judge_a: Mapping[str, Any] | None = None,
    compatibility: Mapping[str, Any] | None = None,
    judge_b: Mapping[str, Any] | None = None,
    repair_codes: Sequence[str] = (),
) -> tuple[str, str]:
    context = _render_context(row)
    context["observation"] = _canonical_bytes(observation).decode("utf-8")
    if kind == "judge_a":
        temporal = observation.get("temporal_evidence")
        if not isinstance(temporal, list):
            raise AcceptanceError(
                "Judge-A prompt requires temporal_evidence as a JSON array"
            )
        context["source_evidence_refs"] = _canonical_bytes(
            [
                "initial_state",
                "source_action",
                *(
                    f"temporal_evidence:{index}"
                    for index in range(len(temporal))
                ),
            ]
        ).decode("utf-8")
        system = str(constants["JUDGE_A_SYSTEM"])
        template = str(constants["JUDGE_A_PROMPT"])
    elif kind == "writer":
        if judge_a is None:
            raise AcceptanceError("writer prompt context is incomplete")
        system = str(constants["COMPATIBILITY_SYSTEM"])
        template = str(constants["COMPATIBILITY_PROMPT"])
        context["frozen_target_core"] = _canonical_bytes(
            {
                "target_action_normalized": judge_a[
                    "target_action_normalized"
                ],
                "target_action_verb": judge_a["target_action_verb"],
            }
        ).decode("utf-8").rstrip("\n")
        context["target_classification"] = _canonical_bytes(
            {
                "source_target_relation": judge_a[
                    "source_target_relation"
                ],
                "target_change_class": judge_a["target_change_class"],
            }
        ).decode("utf-8").rstrip("\n")
    elif kind == "judge_b":
        if judge_a is None or compatibility is None:
            raise AcceptanceError("Judge-B prompt context is incomplete")
        system = str(constants["JUDGE_B_SYSTEM"])
        template = str(constants["JUDGE_B_PROMPT"])
        context["judge_a"] = _canonical_bytes(judge_a).decode("utf-8")
        context["target_support"] = _canonical_bytes(
            _exact_target_clause_support(compatibility)
        ).decode("utf-8")
        context["compatibility"] = _canonical_bytes(compatibility).decode(
            "utf-8"
        )
    elif kind == "repair":
        if judge_a is None or compatibility is None or judge_b is None:
            raise AcceptanceError("repair prompt context is incomplete")
        system = str(constants["DRAFT_REPAIR_SYSTEM"])
        template = str(constants["DRAFT_REPAIR_PROMPT"])
        context["schema"] = _canonical_bytes(
            constants["COMPATIBILITY_REPAIR_SCHEMA"]
        ).decode("utf-8")
        context["target_core"] = _canonical_bytes(
            _target_core(compatibility)
        ).decode("utf-8")
        context["judge_a"] = _canonical_bytes(judge_a).decode("utf-8")
        context["judge_b"] = _canonical_bytes(judge_b).decode("utf-8")
        context["repair_codes"] = _canonical_bytes(
            sorted(set(repair_codes))
        ).decode("utf-8")
        context["compatibility"] = _canonical_bytes(compatibility).decode(
            "utf-8"
        )
    else:
        raise AssertionError(kind)
    prompt = template.format(**context)
    return prompt, _text_digest(system + "\n" + prompt)


def _target_core(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in TARGET_CORE_FIELDS if field not in value]
    if missing:
        raise AcceptanceError(f"target core missing fields: {missing}")
    return {field: value[field] for field in TARGET_CORE_FIELDS}


def _aggregate(
    stage: str,
    decision: str,
    risk_codes: Sequence[str],
    repair_codes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": JUDGE_AGGREGATE_SCHEMA,
        "stage": stage,
        "decision": decision,
        "risk_codes": sorted(set(risk_codes)),
        "repair_codes": sorted(set(repair_codes)),
    }


def _validate_observation(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{context}: observation is not an object")
    _exact_keys(value, OBSERVATION_KEYS, f"{context}.observation")
    _assert_equal(
        value["schema_version"], ANCHOR_OBSERVATION_SCHEMA, context
    )
    for field in (
        "initial_state",
        "source_action",
    ):
        _literal_text(value[field], f"{context}.{field}")
    for field in (
        "visible_entities",
        "interaction_affordances",
        "temporal_evidence",
        "uncertainty_codes",
    ):
        if not isinstance(value[field], list) or not all(
            isinstance(item, str) and item.strip() for item in value[field]
        ):
            raise AcceptanceError(f"{context}.{field}: invalid string list")
        for index, item in enumerate(value[field]):
            _literal_text(item, f"{context}.{field}[{index}]")
    if not value["visible_entities"] or not value["temporal_evidence"]:
        raise AcceptanceError(f"{context}: observation evidence is empty")
    enums = {
        "source_quality": QUALITY,
        "resolution_quality": QUALITY,
        "initial_state_clarity": CLARITY,
        "subject_visibility": CLARITY,
        "actor_motion": MOTION_LEVELS,
        "motion_dynamics": MOTION_DYNAMICS,
        "camera_motion": SCENE_MOTION,
        "background_motion": SCENE_MOTION,
        "single_continuous_shot": YES_NO_UNCLEAR,
        "artifact_level": ARTIFACT_LEVELS,
    }
    for field, allowed in enums.items():
        if value[field] not in allowed:
            raise AcceptanceError(
                f"{context}.{field}: value outside closed enum"
            )
    definite = all(value[field] != "unclear" for field in enums)
    if definite and value["uncertainty_codes"]:
        raise AcceptanceError(
            f"{context}: definite observation has uncertainty codes"
        )
    if value["motion_dynamics"] in {"strong", "moderate"}:
        _assert_equal(value["actor_motion"], "clear", context)
    if value["motion_dynamics"] == "weak" and value["actor_motion"] not in {
        "clear",
        "weak",
    }:
        raise AcceptanceError(f"{context}: weak dynamics/motion conflict")
    if (
        value["motion_dynamics"] == "none"
        and value["actor_motion"] != "none"
    ):
        raise AcceptanceError(f"{context}: no dynamics/motion conflict")
    if value["actor_motion"] == "none" and value["motion_dynamics"] != "none":
        raise AcceptanceError(f"{context}: no motion/dynamics conflict")
    return value


def _validate_judge_a(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{context}: Judge A is not an object")
    _exact_keys(value, JUDGE_A_KEYS, f"{context}.judge_a")
    _assert_equal(value["schema_version"], TARGET_ADMISSIBILITY_SCHEMA, context)
    enums = {
        "target_change_class": TARGET_CHANGE_TYPES,
        "source_target_relation": SOURCE_TARGET_RELATIONS,
        "target_already_true": YES_NO_UNCLEAR,
        "target_start_state_visually_verifiable": YES_NO_UNCLEAR,
        "prerequisite_grounded": YES_NO_UNCLEAR,
        "novel_trajectory": YES_NO_UNCLEAR,
        "scalar_or_endpoint_only": YES_NO_UNCLEAR,
        "target_evidence_ref": TARGET_EVIDENCE_REFS,
        "confidence": CONFIDENCE,
    }
    for field, allowed in enums.items():
        if value[field] not in allowed:
            raise AcceptanceError(
                f"{context}.{field}: value outside closed enum"
            )
    target_action = _literal_text(
        value["target_action_normalized"],
        f"{context}.target_action_normalized",
        allow_sentinel=True,
    )
    target_verb = _literal_text(
        value["target_action_verb"],
        f"{context}.target_action_verb",
        allow_sentinel=True,
    )
    if _ACTION_VERB_RE.fullmatch(target_verb) is None or len(target_verb) > 64:
        raise AcceptanceError(
            f"{context}: target_action_verb is not canonical lowercase "
            "snake_case of at most 64 characters"
        )
    description = _literal_text(
        value["novel_trajectory_description"],
        f"{context}.novel_trajectory_description",
        allow_sentinel=True,
    )
    if value["novel_trajectory"] == "yes":
        _literal_text(
            description,
            f"{context}.novel_trajectory_description",
        )
    elif value["novel_trajectory"] == "no" and description != "none":
        raise AcceptanceError(
            f"{context}: novel_trajectory=no requires description=none"
        )
    elif (
        value["novel_trajectory"] == "unclear"
        and description != "unclear"
    ):
        raise AcceptanceError(
            f"{context}: novel_trajectory=unclear requires description=unclear"
        )
    source_ref = value["source_evidence_ref"]
    if (
        not isinstance(source_ref, str)
        or SOURCE_EVIDENCE_REF_RE.fullmatch(source_ref) is None
    ):
        raise AcceptanceError(f"{context}: invalid source_evidence_ref")
    uncertainties = value["uncertainty_codes"]
    if not isinstance(uncertainties, list) or not all(
        isinstance(item, str)
        and _ACTION_VERB_RE.fullmatch(item) is not None
        for item in uncertainties
    ):
        raise AcceptanceError(f"{context}: invalid uncertainty_codes")
    atomic_fields = (
        "target_already_true",
        "target_start_state_visually_verifiable",
        "prerequisite_grounded",
        "novel_trajectory",
        "scalar_or_endpoint_only",
    )
    unclear = (
        value["target_change_class"] == "unclear"
        or value["source_target_relation"] == "unclear"
        or target_action.casefold() == "unclear"
        or target_verb == "unclear"
        or any(value[field] == "unclear" for field in atomic_fields)
    )
    if unclear != bool(uncertainties):
        raise AcceptanceError(
            f"{context}: unclear fields and uncertainty_codes disagree"
        )
    if (
        value["target_already_true"] == "yes"
        and value["novel_trajectory"] != "no"
    ):
        raise AcceptanceError(
            f"{context}: already-true target cannot be novel"
        )
    if (
        value["scalar_or_endpoint_only"] == "yes"
        and value["novel_trajectory"] != "no"
    ):
        raise AcceptanceError(
            f"{context}: scalar/endpoint-only target cannot be novel"
        )
    return value


def _validate_compatibility(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{context}: compatibility is not an object")
    _exact_keys(value, COMPATIBILITY_KEYS, f"{context}.compatibility")
    _assert_equal(value["schema_version"], ANCHOR_COMPATIBILITY_SCHEMA, context)
    enums = {
        "decision": DECISIONS,
        "anchor_compatibility": ANCHOR_COMPATIBILITY,
        "caption_consistency": CAPTION_CONSISTENCY,
        "action_change_substantive": YES_NO_UNCLEAR,
        "action_category": ACTION_CATEGORY_VALUES,
        "prerequisites_visible_at_i0": YES_NO_UNCLEAR,
        "target_presupposes_prior_action": YES_NO_UNCLEAR,
        "causal_bridge": CAUSAL_BRIDGES,
        "complete_within_clip": YES_NO_UNCLEAR,
        "confidence": CONFIDENCE,
    }
    for field, allowed in enums.items():
        if value[field] not in allowed:
            raise AcceptanceError(
                f"{context}.{field}: value outside closed enum"
            )
    if value["decision"] not in {"accept", "rewrite"}:
        raise AcceptanceError(f"{context}: writer decision is not usable")
    expected = {
        "action_change_substantive": "yes",
        "prerequisites_visible_at_i0": "yes",
        "target_presupposes_prior_action": "no",
        "complete_within_clip": "yes",
    }
    for field, item in expected.items():
        _assert_equal(value[field], item, f"{context}.{field}")
    if value["causal_bridge"] not in {"direct", "requires_transition"}:
        raise AcceptanceError(f"{context}: infeasible causal bridge")
    target_verb = str(value["target_action_verb"])
    if (
        _ACTION_VERB_RE.fullmatch(target_verb) is None
        or len(target_verb) > 64
        or target_verb == "unclear"
    ):
        raise AcceptanceError(f"{context}: invalid target_action_verb")
    if value["action_category"] not in {
        "locomotion",
        "posture",
        "interaction",
        "articulated",
    }:
        raise AcceptanceError(f"{context}: invalid action_category")
    for field in (
        "source_action_normalized",
        "target_action_normalized",
        "target_action_verb",
        "causal_bridge_description",
        "rewritten_edit_instruction",
        "absolute_target_prompt",
    ):
        _literal_text(value[field], f"{context}.{field}")
    for field in (
        "required_entities",
        "causal_stages",
        "preservation_constraints",
        "unrequested_changes",
        "reason_codes",
        "uncertainty_codes",
    ):
        if not isinstance(value[field], list) or not all(
            isinstance(item, str) and item.strip() for item in value[field]
        ):
            raise AcceptanceError(f"{context}.{field}: invalid string list")
        for index, item in enumerate(value[field]):
            _literal_text(item, f"{context}.{field}[{index}]")
    if not value["causal_stages"] or not value["preservation_constraints"]:
        raise AcceptanceError(f"{context}: generation fields are empty")
    for field in ("unrequested_changes", "reason_codes", "uncertainty_codes"):
        _assert_equal(value[field], [], f"{context}.{field}")
    if (
        value["causal_bridge"] == "requires_transition"
        and len(value["causal_stages"]) < 2
    ):
        raise AcceptanceError(f"{context}: transition lacks stages")
    if value["decision"] == "accept":
        _assert_equal(
            value["anchor_compatibility"],
            "compatible",
            f"{context}.accept anchor",
        )
        _assert_equal(
            value["caption_consistency"],
            "consistent",
            f"{context}.accept caption",
        )
    if (
        value["decision"] == "rewrite"
        and value["anchor_compatibility"] not in {"compatible", "repairable"}
    ):
        raise AcceptanceError(f"{context}: invalid rewrite anchor")
    return value


def _validate_judge_b(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{context}: Judge B is not an object")
    _exact_keys(value, JUDGE_B_KEYS, f"{context}.judge_b")
    _assert_equal(value["schema_version"], DRAFT_CONTINUITY_SCHEMA, context)
    enums = {
        "continuity_mode": CONTINUITY_MODES,
        "target_dominance": TARGET_DOMINANCE,
        "actor_entity_consistency": ACTOR_ENTITY_CONSISTENCY,
        "direction_state_consistency": DIRECTION_STATE_CONSISTENCY,
        "unrequested_action": UNREQUESTED_ACTION,
        "confidence": CONFIDENCE,
    }
    for field, allowed in enums.items():
        if value[field] not in allowed:
            raise AcceptanceError(
                f"{context}.{field}: value outside closed enum"
            )
    replay_ref = value["source_replay_ref"]
    if replay_ref != "none" and (
        not isinstance(replay_ref, str)
        or DRAFT_EVIDENCE_REF_RE.fullmatch(replay_ref) is None
    ):
        raise AcceptanceError(f"{context}: invalid source_replay_ref")
    target_ref = value["target_support_ref"]
    if (
        not isinstance(target_ref, str)
        or DRAFT_EVIDENCE_REF_RE.fullmatch(target_ref) is None
    ):
        raise AcceptanceError(f"{context}: invalid target_support_ref")
    uncertainties = value["uncertainty_codes"]
    if not isinstance(uncertainties, list) or not all(
        isinstance(item, str)
        and _ACTION_VERB_RE.fullmatch(item) is not None
        for item in uncertainties
    ):
        raise AcceptanceError(f"{context}: invalid uncertainty_codes")
    mode = value["continuity_mode"]
    if mode == "clean_direct":
        expected = {
            "target_dominance": "dominant",
            "actor_entity_consistency": "consistent",
            "direction_state_consistency": "consistent",
            "unrequested_action": "none",
            "source_replay_ref": "none",
        }
        if any(value[field] != expected_item for field, expected_item in expected.items()):
            raise AcceptanceError(f"{context}: inconsistent clean_direct tuple")
        if uncertainties:
            raise AcceptanceError(
                f"{context}: clean_direct has uncertainty codes"
            )
    elif mode == "repairable_source_preface":
        if (
            value["target_dominance"]
            not in {"dominant", "present_but_diluted"}
            or value["actor_entity_consistency"] != "consistent"
            or value["direction_state_consistency"] != "consistent"
            or value["unrequested_action"] != "none"
            or replay_ref == "none"
            or uncertainties
        ):
            raise AcceptanceError(
                f"{context}: inconsistent repairable_source_preface tuple"
            )
    elif mode == "unclear":
        unclear_diagnostic = any(
            value[field] == "unclear"
            for field in (
                "target_dominance",
                "actor_entity_consistency",
                "direction_state_consistency",
                "unrequested_action",
            )
        )
        if not uncertainties or not unclear_diagnostic:
            raise AcceptanceError(f"{context}: invalid unclear tuple")
    elif uncertainties:
        raise AcceptanceError(
            f"{context}: definite reject has uncertainty codes"
        )
    return value


def _verify_hard_pass_anchor(
    observation: Mapping[str, Any],
    *,
    context: str,
) -> None:
    exact = {
        "initial_state_clarity": "clear",
        "subject_visibility": "clear",
        "actor_motion": "clear",
        "single_continuous_shot": "yes",
        "uncertainty_codes": [],
    }
    for field, expected in exact.items():
        _assert_equal(
            observation[field], expected, f"{context}.hard_gate.{field}"
        )
    allowed = {
        "motion_dynamics": {"strong", "moderate"},
        "source_quality": {"high", "acceptable"},
        "resolution_quality": {"high", "acceptable"},
        "camera_motion": {"none", "weak"},
        "background_motion": {"none", "weak"},
        "artifact_level": {"none", "low"},
    }
    for field, values in allowed.items():
        if observation[field] not in values:
            raise AcceptanceError(
                f"{context}.hard_gate.{field}: unacceptable value"
            )


def _judge_a_evidence(
    judge: Mapping[str, Any],
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
    context: str,
) -> dict[str, str]:
    source_ref = str(judge["source_evidence_ref"])
    if source_ref == "initial_state":
        source = observation["initial_state"]
    elif source_ref == "source_action":
        source = observation["source_action"]
    else:
        index = int(source_ref.split(":", 1)[1])
        temporal = observation["temporal_evidence"]
        if not isinstance(temporal, list) or index >= len(temporal):
            raise AcceptanceError(
                f"{context}: source evidence selector is out of range"
            )
        source = temporal[index]
    target_ref = str(judge["target_evidence_ref"])
    if target_ref != "instruction":
        raise AcceptanceError(
            f"{context}: target evidence must resolve to immutable instruction"
        )
    target = row["prompt"]
    return {
        "source_evidence_ref": source_ref,
        "source_evidence": _literal_text(
            source, f"{context}.resolved_source_evidence"
        ),
        "target_evidence_ref": target_ref,
        "target_evidence": _literal_text(
            target, f"{context}.resolved_target_evidence"
        ),
    }


def _judge_b_evidence(
    judge: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    context: str,
) -> dict[str, str]:
    def resolve(ref: str, *, allow_none: bool) -> str:
        if ref == "none":
            if allow_none:
                return "none"
            raise AcceptanceError(f"{context}: target selector cannot be none")
        if DRAFT_EVIDENCE_REF_RE.fullmatch(ref) is None:
            raise AcceptanceError(f"{context}: invalid draft selector")
        if ref.startswith("causal_stages:"):
            index = int(ref.split(":", 1)[1])
            stages = compatibility["causal_stages"]
            if not isinstance(stages, list) or index >= len(stages):
                raise AcceptanceError(
                    f"{context}: causal stage selector is out of range"
                )
            selected = stages[index]
        else:
            selected = compatibility[ref]
        return _literal_text(selected, f"{context}.resolved_{ref}")

    replay_ref = str(judge["source_replay_ref"])
    target_ref = str(judge["target_support_ref"])
    return {
        "source_replay_ref": replay_ref,
        "source_replay_evidence": resolve(replay_ref, allow_none=True),
        "target_support_ref": target_ref,
        "target_support_evidence": resolve(target_ref, allow_none=False),
    }


def _target_request_risks(
    judge: Mapping[str, Any],
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[str]:
    class_risks = {
        "same_action_intensity_only": "same_action_scalar_only",
        "same_action_endpoint_or_phase_only": "later_source_phase_or_endpoint",
        "appearance_content_state_only": "appearance_state_only",
        "object_orientation_state_only": "object_orientation_state_only",
        "source_action_restatement": "target_restates_source_action",
    }
    relation_risks = {
        "later_source_phase_or_endpoint": "later_source_phase_or_endpoint",
        "repeats_source_future": "target_restates_source_action",
        "same_action_scalar_only": "same_action_scalar_only",
        "state_or_appearance_only": "appearance_state_only",
    }
    risks: list[str] = []
    if judge["target_change_class"] in class_risks:
        risks.append(class_risks[judge["target_change_class"]])
    if judge["source_target_relation"] in relation_risks:
        risks.append(relation_risks[judge["source_target_relation"]])
    if judge["target_already_true"] == "yes":
        risks.append("target_restates_source_action")
    support = _judge_a_instruction_support(
        judge,
        row=row,
        observation=observation,
    )
    if not support["target_evidence_ref_is_instruction"]:
        risks.append("judge_a:target_evidence_not_immutable_instruction")
    if not support["target_action_normalized_supports_verb"]:
        risks.append(
            "judge_a:target_action_normalized_does_not_support_verb"
        )
    if not support["instruction_supports_target_action"]:
        risks.append("judge_a:instruction_does_not_support_target_action")
    if not support[
        "novel_trajectory_description_supports_target_action"
    ]:
        risks.append(
            "judge_a:novel_trajectory_description_target_mismatch"
        )
    if support["target_matches_observed_source_action"]:
        risks.append("target_restates_source_action")
    if support["instruction_explicitly_restates_source_action"]:
        risks.append("judge_a:instruction_explicit_source_restatement")
    request = str(row["prompt"]).casefold()
    patterns = (
        (
            "same_action_scalar_only",
            r"\b(?:jumping?|jump)\s+higher\b|"
            r"\b(?:larger|massive|powerful)\s+splash\b|"
            r"\bincrease\s+the\s+intensity\b",
        ),
        ("appearance_state_only", r"\bhairstyle\b|\bhair\s+falls?\s+behind\b"),
        (
            "object_orientation_state_only",
            r"\breorient\b.*\b(?:upright|non-pouring)\b",
        ),
        (
            "later_source_phase_or_endpoint",
            r"\bfully\s+extend(?:ed)?\s+(?:his|her|their)?\s*arms\b|"
            r"\bland\s+on\s+the\s+snow\s+and\s+stand\b|"
            r"\babout\s+to\s+plunge\b",
        ),
    )
    for code, pattern in patterns:
        if re.search(pattern, request):
            risks.append(code)
    return sorted(set(risks))


def _aggregate_a(
    judge: Mapping[str, Any],
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    _judge_a_evidence(judge, row, observation, str(row["iid"]))
    admissible_changes = {
        "formation_trajectory",
        "relational_locomotion_trajectory",
        "new_articulated_action",
        "new_posture_transition",
        "new_interaction_action",
        "new_direction_trajectory",
        "other_new_trajectory",
    }
    admissible_relations = {
        "novel_future",
        "shared_base_with_novel_action",
    }
    diagnostics: list[str] = []
    if judge["target_change_class"] not in admissible_changes:
        diagnostics.append(
            f"judge_a:target_change_class:{judge['target_change_class']}"
        )
    if judge["source_target_relation"] not in admissible_relations:
        diagnostics.append(
            "judge_a:source_target_relation:"
            + str(judge["source_target_relation"])
        )
    required_atomic = {
        "target_already_true": "no",
        "target_start_state_visually_verifiable": "yes",
        "prerequisite_grounded": "yes",
        "novel_trajectory": "yes",
        "scalar_or_endpoint_only": "no",
    }
    for field, expected in required_atomic.items():
        if judge[field] != expected:
            diagnostics.append(f"judge_a:{field}:{judge[field]}")
    diagnostics.extend(
        f"judge_a:uncertainty:{code}"
        for code in judge["uncertainty_codes"]
    )
    if judge["confidence"] not in {"medium", "high"}:
        diagnostics.append(f"judge_a:confidence:{judge['confidence']}")
    risks = _target_request_risks(judge, row, observation)
    decision = "pass" if not diagnostics and not risks else "reject"
    return _aggregate(
        "target_admissibility",
        decision,
        [*risks, *diagnostics],
    )


def _semantic_signature(value: str) -> str:
    return " ".join(_TOKEN_RE.findall(value.casefold()))


def _semantic_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing") and len(token[:-3]) >= 3:
        return token[:-3]
    if len(token) > 4 and token.endswith("ed") and len(token[:-2]) >= 3:
        return token[:-2]
    if len(token) > 4 and token.endswith("es") and len(token[:-1]) >= 3:
        return token[:-1]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(value: str) -> set[str]:
    return {_semantic_stem(token) for token in _TOKEN_RE.findall(value.casefold())}


_STOPWORDS = {
    "a", "an", "and", "at", "be", "begin", "begins", "by", "camera",
    "clip", "from", "have", "identity", "in", "into", "it", "its", "make",
    "motion", "of", "on", "perform", "performs", "same", "scene", "shown",
    "start", "starts", "subject", "target", "the", "then", "their", "them",
    "they", "this", "to", "unchanged", "while", "with",
}
_ACTION_LEXICAL_EQUIVALENTS = {
    "ahead": "overtake",
    "arrange": "form",
    "disperse": "spread",
    "fac": "look",
    "face": "look",
    "faced": "look",
    "faces": "look",
    "facing": "look",
    "gaze": "look",
    "gazed": "look",
    "gazes": "look",
    "gazing": "look",
    "greet": "wave",
    "jog": "run",
    "jogg": "run",
    "jogged": "run",
    "jogging": "run",
    "lift": "pick",
    "lifted": "pick",
    "lifting": "pick",
    "organise": "form",
    "organising": "form",
    "organize": "form",
    "organiz": "form",
    "organizing": "form",
    "pass": "overtake",
    "passed": "overtake",
    "passing": "overtake",
    "rais": "raise",
    "raised": "raise",
    "raising": "raise",
    "rearrange": "form",
    "rearranging": "form",
    "reorganise": "form",
    "reorganising": "form",
    "reorganize": "form",
    "reorganiz": "form",
    "reorganizing": "form",
    "rid": "ride",
    "riding": "ride",
    "ris": "stand",
    "rise": "stand",
    "rising": "stand",
    "rotate": "turn",
    "rotat": "turn",
    "rotating": "turn",
    "runn": "run",
    "running": "run",
    "seat": "sit",
    "seated": "sit",
    "sitt": "sit",
    "sitting": "sit",
    "sprint": "run",
    "take": "pick",
    "takes": "pick",
    "taking": "pick",
    "wav": "wave",
    "waved": "wave",
    "waves": "wave",
    "waving": "wave",
}
_ACTION_LEXICAL_STOPWORDS = _STOPWORDS | {"toward", "towards"}
_ACTION_CONCEPT_TOKENS = {
    "advance",
    "approach",
    "bend",
    "bow",
    "carry",
    "cartwheel",
    "catch",
    "climb",
    "close",
    "crawl",
    "crouch",
    "dance",
    "descend",
    "drink",
    "drive",
    "drop",
    "eat",
    "enter",
    "exit",
    "extend",
    "fall",
    "fly",
    "follow",
    "form",
    "grab",
    "hold",
    "jump",
    "kick",
    "kneel",
    "land",
    "lean",
    "leave",
    "look",
    "move",
    "open",
    "overtake",
    "pedal",
    "pick",
    "pull",
    "push",
    "raise",
    "reach",
    "ride",
    "roll",
    "run",
    "shake",
    "sit",
    "slide",
    "spin",
    "spread",
    "squat",
    "stand",
    "stop",
    "swap",
    "swim",
    "throw",
    "turn",
    "walk",
    "wave",
}
_EXPLICIT_SOURCE_RESTATEMENT_PATTERNS = (
    re.compile(
        r"\b(?:continue|keep|maintain|repeat|replay|preserve)\b"
        r".{0,96}\b(?:source|same|shown|existing|current|original)\b"
        r".{0,64}\b(?:action|motion|movement|trajectory|doing)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:continue|repeat|replay)\b.{0,96}"
        r"\b(?:action|motion|movement|trajectory)\b.{0,64}"
        r"\b(?:exactly|unchanged|shown|source|same)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do|perform)\b.{0,48}\b(?:exactly|same)\b.{0,64}"
        r"\b(?:shown|source|original)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+change\b.{0,48}\b(?:action|motion|movement|trajectory)\b",
        flags=re.IGNORECASE,
    ),
)


def _action_contract_tokens(value: str) -> set[str]:
    normalized: set[str] = set()
    for raw in _TOKEN_RE.findall(value.casefold()):
        if raw in _ACTION_LEXICAL_STOPWORDS or len(raw) <= 1:
            continue
        stem = _semantic_stem(raw)
        token = _ACTION_LEXICAL_EQUIVALENTS.get(
            raw, _ACTION_LEXICAL_EQUIVALENTS.get(stem, stem)
        )
        if token not in _ACTION_LEXICAL_STOPWORDS and len(token) > 1:
            normalized.add(token)
    if "pick" in normalized:
        normalized.discard("up")
    if "stand" in normalized:
        normalized.discard("up")
    if "sit" in normalized:
        normalized.discard("down")
    if "spread" in normalized:
        normalized.discard("out")
    if "overtake" in normalized:
        normalized.discard("run")
    return normalized


def _normalize_action_concepts(concepts: set[str]) -> set[str]:
    normalized = set(concepts)
    if "pick" in normalized:
        normalized.discard("hold")
    if "wave" in normalized:
        normalized.discard("raise")
    return normalized


def _action_concept_tokens(value: str) -> set[str]:
    return _normalize_action_concepts(
        _action_contract_tokens(value) & _ACTION_CONCEPT_TOKENS
    )


def _action_verb_tokens(value: str) -> set[str]:
    return _action_contract_tokens(value) - {
        "around",
        "down",
        "out",
        "over",
        "up",
    }


def _coverage(required: set[str], evidence: set[str]) -> float:
    return len(required & evidence) / len(required) if required else 0.0


def _instruction_target_contract(
    *,
    target: str,
    verb: str,
    instruction: str,
) -> dict[str, Any]:
    instruction_tokens = _action_contract_tokens(instruction)
    target_tokens = _action_contract_tokens(target)
    verb_tokens = _action_verb_tokens(verb)
    instruction_concepts = _action_concept_tokens(instruction)
    target_concepts = _normalize_action_concepts(
        _action_concept_tokens(target) | verb_tokens
    )
    instruction_concepts = _normalize_action_concepts(
        instruction_concepts | (verb_tokens & instruction_tokens)
    )
    target_supports_verb = bool(verb_tokens) and verb_tokens <= target_tokens
    return {
        "instruction_sha256": _text_digest(instruction),
        "target_action_normalized_sha256": _text_digest(target),
        "target_action_verb_sha256": _text_digest(verb),
        "instruction_contract_tokens": sorted(instruction_tokens),
        "target_action_contract_tokens": sorted(target_tokens),
        "target_action_verb_tokens": sorted(verb_tokens),
        "instruction_action_concept_tokens": sorted(instruction_concepts),
        "target_action_concept_tokens": sorted(target_concepts),
        "target_action_normalized_supports_verb": target_supports_verb,
        "target_tokens_supported_by_instruction_coverage": round(
            _coverage(target_tokens, instruction_tokens), 6
        ),
        "instruction_tokens_covered_by_target_coverage": round(
            _coverage(instruction_tokens, target_tokens), 6
        ),
        "instruction_action_concepts_covered_by_target": round(
            _coverage(instruction_concepts, target_concepts), 6
        ),
        "target_action_concepts_supported_by_instruction": round(
            _coverage(target_concepts, instruction_concepts), 6
        ),
        "complete_instruction_target_contract": bool(
            target_concepts
            and instruction_concepts
            and target_supports_verb
            and target_concepts == instruction_concepts
        ),
    }


def _writer_instruction_support(
    compatibility: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "policy_version": "writer-immutable-instruction-lexical-evidence-v1",
        **_instruction_target_contract(
            target=str(compatibility["target_action_normalized"]),
            verb=str(compatibility["target_action_verb"]),
            instruction=str(row["prompt"]),
        ),
    }


def _target_core_agreement(
    judge_a: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    instruction = str(row["prompt"])
    judge_contract = _instruction_target_contract(
        target=str(judge_a["target_action_normalized"]),
        verb=str(judge_a["target_action_verb"]),
        instruction=instruction,
    )
    writer_contract = _writer_instruction_support(compatibility, row)
    judge_tokens = set(judge_contract["target_action_concept_tokens"])
    writer_tokens = set(writer_contract["target_action_concept_tokens"])
    judge_verbs = set(judge_contract["target_action_verb_tokens"])
    writer_verbs = set(writer_contract["target_action_verb_tokens"])
    verb_overlap = judge_verbs & writer_verbs
    verb_union = judge_verbs | writer_verbs
    judge_bound = bool(
        judge_a["target_evidence_ref"] == "instruction"
        and judge_contract["complete_instruction_target_contract"]
    )
    writer_bound = bool(
        writer_contract["complete_instruction_target_contract"]
    )
    normalized_exact_match = bool(
        isinstance(judge_a.get("target_action_normalized"), str)
        and isinstance(
            compatibility.get("target_action_normalized"),
            str,
        )
        and judge_a["target_action_normalized"]
        == compatibility["target_action_normalized"]
    )
    verb_exact_match = bool(
        isinstance(judge_a.get("target_action_verb"), str)
        and isinstance(compatibility.get("target_action_verb"), str)
        and judge_a["target_action_verb"]
        == compatibility["target_action_verb"]
    )
    normalized_action_agreement = bool(
        judge_tokens and writer_tokens and judge_tokens == writer_tokens
    )
    return {
        "policy_version": "judge-a-writer-target-core-exact-copy-v2",
        "instruction_sha256": _text_digest(instruction),
        "judge_a_instruction_bound": judge_bound,
        "writer_instruction_bound": writer_bound,
        "normalized_exact_match": normalized_exact_match,
        "verb_exact_match": verb_exact_match,
        "normalized_action_bidirectional_agreement": (
            normalized_action_agreement
        ),
        "judge_a_action_tokens_covered_by_writer": round(
            _coverage(judge_tokens, writer_tokens), 6
        ),
        "writer_action_tokens_covered_by_judge_a": round(
            _coverage(writer_tokens, judge_tokens), 6
        ),
        "target_verb_overlap": bool(verb_overlap),
        "target_verb_overlap_tokens": sorted(verb_overlap),
        "target_verb_overlap_ratio": round(
            len(verb_overlap) / len(verb_union) if verb_union else 0.0,
            6,
        ),
        "agreement_verified": bool(
            judge_bound
            and writer_bound
            and normalized_exact_match
            and verb_exact_match
        ),
        "judge_a_contract": judge_contract,
        "writer_contract": writer_contract,
    }


def _judge_a_instruction_support(
    judge: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    instruction = str(row["prompt"])
    target = str(judge["target_action_normalized"])
    verb = str(judge["target_action_verb"])
    trajectory = str(judge["novel_trajectory_description"])
    contract = _instruction_target_contract(
        target=target,
        verb=verb,
        instruction=instruction,
    )
    target_tokens = set(contract["target_action_contract_tokens"])
    verb_tokens = set(contract["target_action_verb_tokens"])
    trajectory_tokens = (
        _action_contract_tokens(trajectory)
        if trajectory not in {"none", "unclear"}
        else set()
    )
    source_tokens = _action_contract_tokens(
        str(observation["source_action"])
    )
    target_signature = _semantic_signature(target)
    trajectory_signature = _semantic_signature(trajectory)
    target_literal_in_trajectory = bool(
        target_signature
        and trajectory_signature not in {"none", "unclear"}
        and target_signature in trajectory_signature
    )
    target_supports_verb = bool(
        contract["target_action_normalized_supports_verb"]
    )
    trajectory_supports_target = (
        judge["novel_trajectory"] != "yes"
        or target_literal_in_trajectory
        or (
            target_supports_verb
            and bool(verb_tokens)
            and verb_tokens <= trajectory_tokens
            and bool(target_tokens)
            and target_tokens <= trajectory_tokens
        )
    )
    source_target_coverage = _coverage(target_tokens, source_tokens)
    target_source_coverage = _coverage(source_tokens, target_tokens)
    return {
        **contract,
        "target_evidence_ref_is_instruction": (
            judge["target_evidence_ref"] == "instruction"
        ),
        "instruction_supports_target_action": bool(
            contract["complete_instruction_target_contract"]
        ),
        "novel_trajectory_description_supports_target_action": (
            trajectory_supports_target
        ),
        "target_matches_observed_source_action": bool(
            target_tokens
            and source_tokens
            and source_target_coverage >= 0.85
            and target_source_coverage >= 0.60
        ),
        "instruction_explicitly_restates_source_action": any(
            pattern.search(instruction) is not None
            for pattern in _EXPLICIT_SOURCE_RESTATEMENT_PATTERNS
        ),
    }


def _semantic_failures(
    compatibility: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[str]:
    source_signatures = {
        _semantic_signature(str(observation["source_action"])),
        _semantic_signature(str(compatibility["source_action_normalized"])),
    }
    source_signatures.discard("")
    target = _semantic_signature(
        str(compatibility["target_action_normalized"])
    )
    failures: list[str] = []
    if target in source_signatures:
        failures.append("target_action_restates_source_action")
    for field in ("rewritten_edit_instruction", "causal_bridge_description"):
        signature = _semantic_signature(str(compatibility[field]))
        if signature in source_signatures and target not in source_signatures:
            failures.append(f"{field}_restates_source_action")
    absolute = _semantic_signature(str(compatibility["absolute_target_prompt"]))
    if target not in source_signatures and any(
        len(source.split()) >= 5 and source in absolute
        for source in source_signatures
    ):
        failures.append("absolute_target_prompt_copies_source_trajectory")
    stages = " ".join(str(item) for item in compatibility["causal_stages"])
    stage_tokens = _tokens(stages)
    source_tokens = _tokens(str(observation["source_action"]))
    if (
        target not in source_signatures
        and len(stage_tokens) >= 8
        and stage_tokens
        and len(stage_tokens & source_tokens) / len(stage_tokens) >= 0.90
    ):
        failures.append("causal_stages_restate_source_trajectory")
    return failures


def _target_support(compatibility: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "rewritten_edit_instruction",
        "causal_bridge_description",
        "causal_stages",
        "absolute_target_prompt",
    )
    target = str(compatibility["target_action_normalized"])
    verb = str(compatibility["target_action_verb"])
    target_tokens = _tokens(target)
    verb_tokens = _tokens(verb)
    distinctive = {
        token for token in target_tokens if token not in _STOPWORDS and len(token) > 1
    }
    verified: list[str] = []
    unverified: list[str] = []
    target_signature = _semantic_signature(target)
    for field in fields:
        raw = compatibility[field]
        text = (
            " ".join(str(item) for item in raw)
            if field == "causal_stages"
            else str(raw)
        )
        field_tokens = _tokens(text)
        coverage = (
            len(distinctive & field_tokens) / len(distinctive)
            if distinctive
            else 0.0
        )
        supported = (
            bool(target_signature)
            and target_signature in _semantic_signature(text)
        ) or (bool(verb_tokens) and verb_tokens <= field_tokens and coverage >= 0.5)
        (verified if supported else unverified).append(field)
    target_supports_verb = bool(verb_tokens) and verb_tokens <= target_tokens
    return {
        "policy_version": "target-action-lexical-evidence-v2",
        "target_action_normalized_supports_verb": target_supports_verb,
        "lexically_verified_fields": verified,
        "lexically_unverified_fields": unverified,
        "requires_proposal_bound_human_review": (
            not target_supports_verb or bool(unverified)
        ),
    }


def _exact_target_clause_support(
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently mirror the producer's byte-exact Writer-field gate."""

    fields = (
        "rewritten_edit_instruction",
        "causal_bridge_description",
        "causal_stages",
        "absolute_target_prompt",
    )
    target = compatibility.get("target_action_normalized")
    if not isinstance(target, str) or not target:
        return {
            "policy_version": "writer-exact-target-clause-evidence-v1",
            "target_clause_sha256": None,
            "exact_verified_fields": [],
            "exact_unverified_fields": list(fields),
        }
    verified: list[str] = []
    unverified: list[str] = []
    for field in fields:
        raw = compatibility.get(field)
        if field == "causal_stages":
            present = bool(
                isinstance(raw, list)
                and any(
                    isinstance(item, str) and target in item
                    for item in raw
                )
            )
        else:
            present = isinstance(raw, str) and target in raw
        (verified if present else unverified).append(field)
    return {
        "policy_version": "writer-exact-target-clause-evidence-v1",
        "target_clause_sha256": hashlib.sha256(
            target.encode("utf-8")
        ).hexdigest(),
        "exact_verified_fields": verified,
        "exact_unverified_fields": unverified,
    }


def _policy_failures(
    compatibility: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    if compatibility["decision"] == "accept":
        if compatibility["anchor_compatibility"] != "compatible":
            failures.append("decision=accept requires anchor_compatibility=compatible")
        if compatibility["caption_consistency"] != "consistent":
            failures.append("decision=accept requires caption_consistency=consistent")
    if (
        compatibility["decision"] == "rewrite"
        and compatibility["anchor_compatibility"]
        not in {"compatible", "repairable"}
    ):
        failures.append("decision=rewrite requires compatible anchor")
    failures.extend(_semantic_failures(compatibility, observation))
    return failures


def _generation_risks(
    compatibility: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> list[str]:
    semantic = _semantic_failures(compatibility, observation)
    risks: list[str] = []
    if "absolute_target_prompt_copies_source_trajectory" in semantic:
        risks.append("absolute_prompt_copies_source_future")
    if "causal_stages_restate_source_trajectory" in semantic:
        risks.append("causal_stages_copy_source_future")
    if any("restates_source_action" in item for item in semantic):
        risks.append("target_restates_source_action")
    if not _target_support(compatibility)["lexically_verified_fields"]:
        risks.append("target_missing_from_generation_fields")
    if compatibility["unrequested_changes"]:
        risks.append("unrequested_actor_or_action")
    return sorted(set(risks))


def _aggregate_b(
    judge: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_evidence = _judge_b_evidence(judge, compatibility, "Judge B")
    risks = _generation_risks(compatibility, observation)
    policy = _policy_failures(compatibility, observation)
    semantic = set(_semantic_failures(compatibility, observation))
    nonsemantic = [failure for failure in policy if failure not in semantic]
    clean = (
        judge["continuity_mode"] == "clean_direct"
        and judge["target_dominance"] == "dominant"
        and judge["actor_entity_consistency"] == "consistent"
        and judge["direction_state_consistency"] == "consistent"
        and judge["unrequested_action"] == "none"
        and judge["source_replay_ref"] == "none"
        and not judge["uncertainty_codes"]
        and judge["confidence"] in {"medium", "high"}
        and not risks
        and not policy
    )
    if clean:
        return _aggregate("draft_continuity", "pass", ())
    allowed = {
        "absolute_prompt_copies_source_future",
        "causal_stages_copy_source_future",
    }
    repair_risks = sorted(code for code in risks if code in allowed)
    replay_ref = str(judge["source_replay_ref"])
    selected_copy_is_bound = (
        replay_ref == "absolute_target_prompt"
        and "absolute_prompt_copies_source_future" in risks
    ) or (
        replay_ref.startswith("causal_stages:")
        and "causal_stages_copy_source_future" in risks
    )
    repairable = (
        judge["continuity_mode"] == "repairable_source_preface"
        and judge["target_dominance"] in {"dominant", "present_but_diluted"}
        and judge["actor_entity_consistency"] == "consistent"
        and judge["direction_state_consistency"] == "consistent"
        and judge["unrequested_action"] == "none"
        and judge["source_replay_ref"] != "none"
        and not judge["uncertainty_codes"]
        and judge["confidence"] in {"medium", "high"}
        and selected_copy_is_bound
        and bool(repair_risks)
        and set(risks) <= allowed
        and not nonsemantic
    )
    if repairable:
        return _aggregate(
            "draft_continuity", "repair", repair_risks, repair_risks
        )
    reject = list(risks)
    if policy:
        reject.append("judge_b:compatibility_policy_failure")
    if judge["continuity_mode"] not in {
        "clean_direct",
        "repairable_source_preface",
    }:
        reject.append(
            "judge_b:continuity_mode:" + str(judge["continuity_mode"])
        )
    if judge["target_dominance"] not in {
        "dominant",
        "present_but_diluted",
    }:
        reject.append(
            "judge_b:target_dominance:" + str(judge["target_dominance"])
        )
    if judge["actor_entity_consistency"] != "consistent":
        reject.append(
            "judge_b:actor_entity_consistency:"
            + str(judge["actor_entity_consistency"])
        )
    if judge["direction_state_consistency"] != "consistent":
        reject.append(
            "judge_b:direction_state_consistency:"
            + str(judge["direction_state_consistency"])
        )
    if judge["unrequested_action"] == "present":
        reject.append("unrequested_actor_or_action")
    elif judge["unrequested_action"] != "none":
        reject.append(
            "judge_b:unrequested_action:"
            + str(judge["unrequested_action"])
        )
    reject.extend(
        f"judge_b:uncertainty:{code}"
        for code in judge["uncertainty_codes"]
    )
    if judge["confidence"] not in {"medium", "high"}:
        reject.append(f"judge_b:confidence:{judge['confidence']}")
    if (
        judge["continuity_mode"] == "clean_direct"
        and judge["source_replay_ref"] != "none"
    ):
        reject.append("judge_b:source_replay_ref:unexpected")
    if (
        judge["continuity_mode"] == "repairable_source_preface"
        and not repair_risks
    ):
        reject.append(
            "judge_b:source_preface_without_deterministic_copy_evidence"
        )
    if resolved_evidence["target_support_evidence"] == "":
        reject.append("judge_b:evidence_ref:invalid")
    if not reject:
        reject.append("judge_b:tuple_not_permitted")
    return _aggregate("draft_continuity", "reject", reject)


def _validate_source_snapshot_shape(
    value: Any,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnauditableError(f"{context}: source_snapshot is not an object")
    _exact_keys(value, SOURCE_SNAPSHOT_KEYS, f"{context}.source_snapshot")
    for field in (
        "path",
        "manifest_path",
        "archive_path",
    ):
        _path_text(value[field], f"{context}.{field}")
    for field in (
        "tree_sha256",
        "manifest_sha256",
        "archive_sha256",
        "qwen_implementation_sha256",
        "finalizer_implementation_sha256",
        "verifier_implementation_sha256",
        "sbatch_sha256",
    ):
        _sha(value[field], f"{context}.{field}")
    for field in (
        "qwen_relpath",
        "finalizer_relpath",
        "verifier_relpath",
        "sbatch_relpath",
    ):
        relative = _nonempty(value[field], f"{context}.{field}")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise UnauditableError(f"{context}.{field}: unsafe relative path")
    trust_anchors = {
        "qwen_implementation_sha256": EXPECTED_QWEN_IMPLEMENTATION_SHA256,
        "finalizer_implementation_sha256": (
            EXPECTED_FINALIZER_IMPLEMENTATION_SHA256
        ),
        "sbatch_sha256": EXPECTED_SBATCH_SHA256,
    }
    for field, expected in trust_anchors.items():
        _assert_equal(
            value[field], expected, f"{context}.source_snapshot.{field}"
        )
    return value


def _validate_model_shape(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnauditableError(f"{context}: model is not an object")
    _exact_keys(value, MODEL_KEYS, f"{context}.model")
    _path_text(value["path"], f"{context}.model.path")
    _path_text(value["config_path"], f"{context}.model.config_path")
    _sha(value["config_sha256"], f"{context}.model.config_sha256")
    _assert_equal(value["path"], EXPECTED_MODEL_PATH, f"{context}.model.path")
    _assert_equal(
        value["config_sha256"],
        EXPECTED_MODEL_CONFIG_SHA256,
        f"{context}.model.config_sha256",
    )
    return value


def _validate_model_closure_shape(
    value: Any,
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UnauditableError(f"{context}: model_closure is not an object")
    _exact_keys(
        value,
        MODEL_CLOSURE_KEYS,
        f"{context}.model_closure",
    )
    path_text = _path_text(
        value["path"], f"{context}.model_closure.path"
    )
    canonical_parts = MODEL_CLOSURE_CANONICAL_RELPATH.parts
    if Path(path_text).parts[-len(canonical_parts) :] != canonical_parts:
        raise UnauditableError(
            f"{context}.model_closure.path is not the canonical relative path"
        )
    _assert_equal(
        _sha(value["sha256"], f"{context}.model_closure.sha256"),
        EXPECTED_MODEL_CLOSURE_SHA256,
        f"{context}.model_closure source trust anchor",
    )
    _assert_equal(
        value["file_count"],
        EXPECTED_MODEL_CLOSURE_FILE_COUNT,
        f"{context}.model_closure.file_count",
    )
    _assert_equal(
        value["total_bytes"],
        EXPECTED_MODEL_CLOSURE_TOTAL_BYTES,
        f"{context}.model_closure.total_bytes",
    )
    return value


def _validate_acceptance_contract(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "selected",
            "smoke_gold",
            "expected_shard_counts",
            "source_snapshot",
            "model",
            "model_closure",
            "execution",
            "final",
            "bindings",
        },
        "acceptance contract",
    )
    _assert_equal(
        value["schema_version"],
        ACCEPTANCE_CONTRACT_SCHEMA,
        "acceptance contract schema",
    )
    selected = value["selected"]
    if not isinstance(selected, Mapping):
        raise UnauditableError("acceptance selected is not an object")
    _exact_keys(
        selected, {"rows", "sha256", "ordered_iids_sha256"}, "acceptance selected"
    )
    if type(selected["rows"]) is not int or selected["rows"] < 1:
        raise UnauditableError("acceptance selected rows is invalid")
    _sha(selected["sha256"], "acceptance selected sha256")
    _sha(
        selected["ordered_iids_sha256"],
        "acceptance selected ordered_iids_sha256",
    )
    smoke_gold = value["smoke_gold"]
    if not isinstance(smoke_gold, Mapping):
        raise UnauditableError("acceptance smoke_gold is not an object")
    _exact_keys(smoke_gold, {"path", "sha256"}, "acceptance smoke_gold")
    _path_text(smoke_gold["path"], "acceptance smoke_gold.path")
    _assert_equal(
        _sha(smoke_gold["sha256"], "acceptance smoke_gold.sha256"),
        EXPECTED_SMOKE_GOLD_SHA256,
        "acceptance smoke gold source trust anchor",
    )
    shard_counts = value["expected_shard_counts"]
    if (
        not isinstance(shard_counts, list)
        or len(shard_counts) != 8
        or not all(type(item) is int and item >= 0 for item in shard_counts)
        or sum(shard_counts) != selected["rows"]
    ):
        raise UnauditableError("acceptance expected_shard_counts is invalid")
    _validate_source_snapshot_shape(
        value["source_snapshot"], context="acceptance"
    )
    _validate_model_shape(value["model"], context="acceptance")
    _validate_model_closure_shape(
        value["model_closure"], context="acceptance"
    )
    execution = value["execution"]
    if not isinstance(execution, Mapping):
        raise UnauditableError("acceptance execution is not an object")
    _exact_keys(execution, QWEN_EXECUTION_KEYS, "acceptance execution")
    _assert_equal(
        dict(execution), EXPECTED_QWEN_EXECUTION, "acceptance execution"
    )
    final = value["final"]
    if not isinstance(final, Mapping):
        raise UnauditableError("acceptance final is not an object")
    _exact_keys(final, set(EXPECTED_FINAL), "acceptance final")
    _assert_equal(dict(final), EXPECTED_FINAL, "acceptance final")
    bindings = value["bindings"]
    if not isinstance(bindings, Mapping):
        raise UnauditableError("acceptance bindings is not an object")
    _exact_keys(
        bindings,
        {"submission_contract_sha256", "completion_receipt_sha256"},
        "acceptance bindings",
    )
    _sha(
        bindings["submission_contract_sha256"],
        "acceptance bindings submission SHA",
    )
    _sha(
        bindings["completion_receipt_sha256"],
        "acceptance bindings completion SHA",
    )


def _validate_submission_contract(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "selected",
            "smoke_gold",
            "source_snapshot",
            "model",
            "model_closure",
            "runtime",
            "outputs",
        },
        "submission contract",
    )
    _assert_equal(
        value["schema_version"],
        SUBMISSION_CONTRACT_SCHEMA,
        "submission contract schema",
    )
    selected = value["selected"]
    if not isinstance(selected, Mapping):
        raise UnauditableError("submission selected is not an object")
    _exact_keys(selected, {"path", "sha256", "rows"}, "submission selected")
    _path_text(selected["path"], "submission selected.path")
    _sha(selected["sha256"], "submission selected.sha256")
    if type(selected["rows"]) is not int or selected["rows"] < 1:
        raise UnauditableError("submission selected.rows is invalid")
    smoke_gold = value["smoke_gold"]
    if not isinstance(smoke_gold, Mapping):
        raise UnauditableError("submission smoke_gold is not an object")
    _exact_keys(smoke_gold, {"path", "sha256"}, "submission smoke_gold")
    _path_text(smoke_gold["path"], "submission smoke_gold.path")
    _assert_equal(
        _sha(smoke_gold["sha256"], "submission smoke_gold.sha256"),
        EXPECTED_SMOKE_GOLD_SHA256,
        "submission smoke gold source trust anchor",
    )
    _validate_source_snapshot_shape(
        value["source_snapshot"], context="submission"
    )
    _validate_model_shape(value["model"], context="submission")
    _validate_model_closure_shape(
        value["model_closure"], context="submission"
    )
    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise UnauditableError("submission runtime is not an object")
    expected_runtime = {
        **EXPECTED_QWEN_EXECUTION,
        "final_seed": 260730,
        "allow_partial": True,
    }
    _exact_keys(runtime, set(expected_runtime), "submission runtime")
    _assert_equal(dict(runtime), expected_runtime, "submission runtime")
    outputs = value["outputs"]
    if not isinstance(outputs, Mapping):
        raise UnauditableError("submission outputs is not an object")
    _exact_keys(outputs, {"qwen_root", "final_output"}, "submission outputs")
    _path_text(outputs["qwen_root"], "submission outputs.qwen_root")
    _path_text(outputs["final_output"], "submission outputs.final_output")


def _validate_completion_receipt(value: Mapping[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "job_id",
            "submission_contract_path",
            "submission_contract_sha256",
            "selected_sha256",
            "smoke_gold_sha256",
            "model_closure",
            "qwen_root",
            "final_output",
            "qwen_shards",
            "final_artifacts",
        },
        "completion receipt",
    )
    _assert_equal(
        value["schema_version"],
        COMPLETION_RECEIPT_SCHEMA,
        "completion receipt schema",
    )
    _assert_equal(value["status"], "complete", "completion receipt status")
    _nonempty(value["job_id"], "completion receipt job_id")
    _path_text(
        value["submission_contract_path"],
        "completion receipt submission_contract_path",
    )
    _sha(
        value["submission_contract_sha256"],
        "completion receipt submission SHA",
    )
    _sha(value["selected_sha256"], "completion receipt selected SHA")
    _assert_equal(
        _sha(
            value["smoke_gold_sha256"],
            "completion receipt smoke gold SHA",
        ),
        EXPECTED_SMOKE_GOLD_SHA256,
        "completion smoke gold source trust anchor",
    )
    _validate_model_closure_shape(
        value["model_closure"], context="completion receipt"
    )
    _path_text(value["qwen_root"], "completion receipt qwen_root")
    _path_text(value["final_output"], "completion receipt final_output")
    shards = value["qwen_shards"]
    if not isinstance(shards, list) or len(shards) != 8:
        raise UnauditableError("completion receipt requires eight shards")
    for index, item in enumerate(shards):
        if not isinstance(item, Mapping):
            raise UnauditableError(
                f"completion receipt shard {index} is not an object"
            )
        _exact_keys(
            item,
            {
                "index",
                "path",
                "sha256",
                "bytes",
                "receipt_path",
                "receipt_sha256",
            },
            f"completion receipt shard {index}",
        )
        _assert_equal(item["index"], index, f"completion shard {index} index")
        _path_text(item["path"], f"completion shard {index} path")
        _path_text(
            item["receipt_path"], f"completion shard {index} receipt_path"
        )
        _sha(item["sha256"], f"completion shard {index} sha256")
        _sha(
            item["receipt_sha256"],
            f"completion shard {index} receipt_sha256",
        )
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise UnauditableError(
                f"completion shard {index} bytes is invalid"
            )
    artifacts = value["final_artifacts"]
    if not isinstance(artifacts, Mapping):
        raise UnauditableError("completion final_artifacts is not an object")
    _exact_keys(artifacts, set(FINAL_NAMES), "completion final_artifacts")
    for name in FINAL_NAMES:
        item = artifacts[name]
        if not isinstance(item, Mapping):
            raise UnauditableError(f"completion artifact {name} is invalid")
        _exact_keys(item, {"path", "sha256", "bytes"}, f"artifact {name}")
        _path_text(item["path"], f"artifact {name}.path")
        _sha(item["sha256"], f"artifact {name}.sha256")
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise UnauditableError(f"artifact {name}.bytes is invalid")


def _verify_source_snapshot(
    snapshot_contract: Mapping[str, Any],
    source_snapshot_arg: Path,
) -> dict[str, Any]:
    root = _directory(source_snapshot_arg, context="source snapshot")
    _assert_equal(str(root), snapshot_contract["path"], "source snapshot path")
    manifest = _regular_file(
        Path(str(snapshot_contract["manifest_path"])),
        context="source snapshot manifest",
    )
    _assert_equal(
        str(manifest),
        str(root / "SOURCE_FILES.jsonl"),
        "source snapshot manifest path",
    )
    manifest_rows, manifest_raw, _ = _load_jsonl(
        manifest, context="source snapshot manifest"
    )
    canonical_manifest = b"".join(
        _canonical_bytes(row) + b"\n" for row in manifest_rows
    )
    _assert_equal(
        manifest_raw,
        canonical_manifest,
        "source snapshot canonical manifest bytes",
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    tree_sha = hashlib.sha256(canonical_manifest).hexdigest()
    _assert_equal(
        manifest_sha,
        snapshot_contract["manifest_sha256"],
        "source snapshot manifest SHA",
    )
    _assert_equal(
        tree_sha,
        snapshot_contract["tree_sha256"],
        "source snapshot tree SHA",
    )

    seen: set[str] = set()
    ordered_paths: list[str] = []
    expected_directories: set[str] = set()
    for index, row in enumerate(manifest_rows):
        _exact_keys(
            row,
            {"mode", "path", "sha256", "size", "type"},
            f"source manifest row {index}",
        )
        _assert_equal(row["type"], "file", f"source manifest row {index} type")
        relative = _nonempty(row["path"], f"source manifest row {index} path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise AcceptanceError(f"unsafe/duplicate source path: {relative}")
        seen.add(relative)
        ordered_paths.append(relative)
        parent = relative_path.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        path = _regular_file(root / relative_path, context=f"source {relative}")
        _assert_equal(
            path.stat().st_size, row["size"], f"source {relative} size"
        )
        _assert_equal(
            _file_digest(path), row["sha256"], f"source {relative} SHA"
        )
        _sha(row["sha256"], f"source {relative} manifest SHA")
        if not isinstance(row["mode"], str) or not re.fullmatch(
            r"[0-7]{4}", row["mode"]
        ):
            raise AcceptanceError(f"source {relative}: invalid mode")
        actual_mode = f"{stat.S_IMODE(path.stat().st_mode):04o}"
        _assert_equal(actual_mode, row["mode"], f"source {relative} mode")
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise AcceptanceError(f"source {relative}: file remains writable")
    _assert_equal(
        ordered_paths, sorted(ordered_paths), "source manifest path ordering"
    )

    expected_files = seen | {"SOURCE_FILES.jsonl", "SOURCE_PROVENANCE.json"}
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        if stat.S_IMODE(directory_path.stat().st_mode) & 0o222:
            raise AcceptanceError(
                "source snapshot directory remains writable: "
                + directory_path.relative_to(root).as_posix()
            )
        for name in [*directory_names, *file_names]:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            entry = path.lstat()
            if stat.S_ISLNK(entry.st_mode):
                raise AcceptanceError(
                    f"source snapshot contains symlink: {relative}"
                )
            if stat.S_ISDIR(entry.st_mode):
                actual_directories.add(relative)
            elif stat.S_ISREG(entry.st_mode):
                actual_files.add(relative)
            else:
                raise AcceptanceError(
                    f"source snapshot contains special file: {relative}"
                )
    _assert_equal(actual_files, expected_files, "source snapshot file closure")
    _assert_equal(
        actual_directories,
        expected_directories,
        "source snapshot directory closure",
    )
    for name in ("SOURCE_FILES.jsonl", "SOURCE_PROVENANCE.json"):
        if stat.S_IMODE((root / name).stat().st_mode) & 0o222:
            raise AcceptanceError(f"source snapshot {name} remains writable")
    provenance, _, provenance_path = _load_json(
        root / "SOURCE_PROVENANCE.json",
        context="source snapshot provenance",
    )
    _assert_equal(
        provenance.get("schema"),
        "motive-action-source-snapshot-v1",
        "source provenance schema",
    )
    _assert_equal(
        provenance.get("source_tree_sha256"),
        tree_sha,
        "source provenance tree SHA",
    )
    _assert_equal(
        provenance.get("source_manifest_sha256"),
        manifest_sha,
        "source provenance manifest SHA",
    )
    _assert_equal(
        provenance.get("source_file_count"),
        len(manifest_rows),
        "source provenance file count",
    )

    file_bindings = {
        "qwen": (
            "qwen_relpath",
            "qwen_implementation_sha256",
        ),
        "finalizer": (
            "finalizer_relpath",
            "finalizer_implementation_sha256",
        ),
        "verifier": (
            "verifier_relpath",
            "verifier_implementation_sha256",
        ),
        "sbatch": ("sbatch_relpath", "sbatch_sha256"),
    }
    bound: dict[str, dict[str, Any]] = {}
    for label, (path_field, digest_field) in file_bindings.items():
        path = _regular_file(
            root / str(snapshot_contract[path_field]),
            context=f"source snapshot {label}",
        )
        digest = _file_digest(path)
        _assert_equal(
            digest,
            snapshot_contract[digest_field],
            f"source snapshot {label} SHA",
        )
        bound[label] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    executed_verifier = _regular_file(
        Path(__file__).resolve(), context="executed verifier"
    )
    _assert_equal(
        _file_digest(executed_verifier),
        bound["verifier"]["sha256"],
        "executed verifier SHA",
    )
    archive = _regular_file(
        Path(str(snapshot_contract["archive_path"])),
        context="source snapshot archive",
    )
    archive_sha = _file_digest(archive)
    _assert_equal(
        archive_sha,
        snapshot_contract["archive_sha256"],
        "source archive SHA",
    )
    archive_files: dict[str, tuple[int, str, int]] = {}
    archive_directories: set[str] = set()
    archive_roots: set[str] = set()
    archive_names: set[str] = set()
    try:
        with tarfile.open(archive, mode="r:*") as handle:
            members = handle.getmembers()
            if not members:
                raise AcceptanceError("source archive is empty")
            for member in members:
                pure = Path(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or not pure.parts
                    or member.name in archive_names
                ):
                    raise AcceptanceError(
                        f"source archive unsafe/duplicate member: {member.name}"
                    )
                archive_names.add(member.name)
                root_name = pure.parts[0]
                archive_roots.add(root_name)
                relative_parts = pure.parts[1:]
                if member.issym() or member.islnk() or not (
                    member.isdir() or member.isfile()
                ):
                    raise AcceptanceError(
                        f"source archive special/link member: {member.name}"
                    )
                if member.mode & 0o222:
                    raise AcceptanceError(
                        f"source archive writable member: {member.name}"
                    )
                if not relative_parts:
                    if not member.isdir():
                        raise AcceptanceError(
                            "source archive root is not a directory"
                        )
                    continue
                relative = Path(*relative_parts).as_posix()
                if member.isdir():
                    archive_directories.add(relative.rstrip("/"))
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise AcceptanceError(
                        f"source archive cannot read member: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                archive_files[relative] = (
                    member.mode & 0o7777,
                    digest.hexdigest(),
                    size,
                )
    except (tarfile.TarError, OSError) as error:
        raise AcceptanceError(f"source archive is invalid: {error}") from error
    _assert_equal(len(archive_roots), 1, "source archive single root")
    _assert_equal(
        set(archive_files), actual_files, "source archive file closure"
    )
    _assert_equal(
        archive_directories,
        actual_directories,
        "source archive directory closure",
    )
    for relative, (mode, digest, size) in archive_files.items():
        extracted = root / relative
        _assert_equal(size, extracted.stat().st_size, f"archive {relative} size")
        _assert_equal(digest, _file_digest(extracted), f"archive {relative} SHA")
        _assert_equal(
            mode,
            stat.S_IMODE(extracted.stat().st_mode),
            f"archive {relative} mode",
        )
    return {
        "path": str(root),
        "tree_sha256": tree_sha,
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha,
            "bytes": len(manifest_raw),
            "rows": len(manifest_rows),
        },
        "archive": {
            "path": str(archive),
            "sha256": archive_sha,
            "bytes": archive.stat().st_size,
        },
        "provenance_path": str(provenance_path),
        "implementations": bound,
    }


def _verify_model(model_contract: Mapping[str, Any]) -> dict[str, Any]:
    model_text = str(model_contract["path"])
    model_path = _directory(Path(model_text), context="model path")
    _assert_equal(str(model_path), model_text, "canonical model path")
    expected_config_text = str(model_path / "config.json")
    _assert_equal(
        model_contract["config_path"],
        expected_config_text,
        "model config ownership",
    )
    config_path = Path(str(model_contract["config_path"])).expanduser()
    if not config_path.is_file():
        raise UnauditableError(f"model config is missing: {config_path}")
    digest = _file_digest(config_path)
    _assert_equal(digest, model_contract["config_sha256"], "model config SHA")
    return {
        "path": str(model_path),
        "config_path": str(config_path),
        "config_sha256": digest,
        "config_bytes": config_path.stat().st_size,
    }


def _verify_model_closure(
    closure_contract: Mapping[str, Any],
    *,
    source_snapshot: Path,
    model_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash the exact frozen model file closure against its anchored manifest."""

    snapshot_root = _directory(
        source_snapshot, context="model closure source snapshot"
    )
    expected_manifest = snapshot_root / MODEL_CLOSURE_CANONICAL_RELPATH
    manifest, manifest_raw, manifest_path = _load_json(
        expected_manifest, context="model closure manifest"
    )
    _assert_equal(
        closure_contract["path"],
        str(manifest_path),
        "model closure canonical snapshot path",
    )
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    _assert_equal(
        manifest_sha,
        closure_contract["sha256"],
        "model closure manifest SHA",
    )
    _exact_keys(
        manifest,
        {
            "schema_version",
            "model_id",
            "revision",
            "model_path",
            "hash_algorithm",
            "file_count",
            "total_bytes",
            "files",
        },
        "model closure manifest",
    )
    _assert_equal(
        manifest["schema_version"],
        MODEL_CLOSURE_SCHEMA,
        "model closure manifest schema",
    )
    _assert_equal(
        _literal_text(manifest["model_id"], "model closure model_id"),
        EXPECTED_MODEL_ID,
        "model closure model_id",
    )
    _assert_equal(
        _literal_text(manifest["revision"], "model closure revision"),
        EXPECTED_MODEL_REVISION,
        "model closure revision",
    )
    _assert_equal(
        manifest["model_path"],
        model_contract["path"],
        "model closure/model contract path",
    )
    _assert_equal(
        manifest["hash_algorithm"],
        "sha256",
        "model closure hash algorithm",
    )
    _assert_equal(
        manifest["file_count"],
        closure_contract["file_count"],
        "model closure manifest file_count",
    )
    _assert_equal(
        manifest["total_bytes"],
        closure_contract["total_bytes"],
        "model closure manifest total_bytes",
    )
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != manifest["file_count"]:
        raise UnauditableError("model closure files do not match file_count")
    expected_files: dict[str, dict[str, Any]] = {}
    ordered_relpaths: list[str] = []
    total_bytes = 0
    for index, item in enumerate(files):
        context = f"model closure files[{index}]"
        if not isinstance(item, dict):
            raise UnauditableError(f"{context} is not an object")
        _exact_keys(item, {"relative_path", "bytes", "sha256"}, context)
        relative = _nonempty(item["relative_path"], f"{context}.relative_path")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise UnauditableError(f"{context}.relative_path is unsafe")
        if relative in expected_files:
            raise UnauditableError(
                f"model closure duplicate relative path: {relative}"
            )
        if type(item["bytes"]) is not int or item["bytes"] < 0:
            raise UnauditableError(f"{context}.bytes is invalid")
        _sha(item["sha256"], f"{context}.sha256")
        expected_files[relative] = item
        ordered_relpaths.append(relative)
        total_bytes += item["bytes"]
    _assert_equal(
        ordered_relpaths,
        sorted(ordered_relpaths),
        "model closure manifest file order",
    )
    _assert_equal(
        total_bytes,
        manifest["total_bytes"],
        "model closure summed bytes",
    )

    model_root = _directory(
        Path(str(model_contract["path"])), context="model closure model path"
    )
    actual_paths: dict[str, Path] = {}
    for candidate in model_root.rglob("*"):
        relative = candidate.relative_to(model_root).as_posix()
        if candidate.is_symlink() and candidate.is_dir():
            raise AcceptanceError(
                f"model closure contains symlink directory: {relative}"
            )
        if candidate.is_file():
            actual_paths[relative] = candidate
        elif not candidate.is_dir():
            raise AcceptanceError(
                f"model closure contains special or broken entry: {relative}"
            )
    _assert_equal(
        set(actual_paths),
        set(expected_files),
        "model closure exact file set",
    )
    verified_files: list[dict[str, Any]] = []
    for relative in ordered_relpaths:
        candidate = actual_paths[relative]
        expected = expected_files[relative]
        actual_size = candidate.stat().st_size
        actual_sha = _file_digest(candidate)
        _assert_equal(
            actual_size,
            expected["bytes"],
            f"model closure {relative} bytes",
        )
        _assert_equal(
            actual_sha,
            expected["sha256"],
            f"model closure {relative} SHA",
        )
        verified_files.append(
            {
                "relative_path": relative,
                "bytes": actual_size,
                "sha256": actual_sha,
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "model_path": str(model_root),
        "file_count": len(verified_files),
        "total_bytes": sum(item["bytes"] for item in verified_files),
        "files_digest": _object_digest(verified_files),
    }


def _verify_selected_media(
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    finalizer_by_iid: dict[str, dict[str, Any]] = {}
    source_paths: set[str] = set()
    anchor_paths: set[str] = set()
    for row in selected_rows:
        iid = str(row["iid"])
        source_text = _path_text(
            row["resolved_src_video"],
            f"selected iid={iid}.resolved_src_video",
        )
        anchor_text = _path_text(
            row["resolved_anchor_image"],
            f"selected iid={iid}.resolved_anchor_image",
        )
        source = _regular_file(
            Path(source_text), context=f"selected iid={iid} source video"
        )
        anchor = _regular_file(
            Path(anchor_text), context=f"selected iid={iid} anchor image"
        )
        _assert_equal(str(source), source_text, f"selected iid={iid} source path")
        _assert_equal(str(anchor), anchor_text, f"selected iid={iid} anchor path")
        if source_text in source_paths:
            raise AcceptanceError(f"duplicate selected source path: {source_text}")
        if anchor_text in anchor_paths:
            raise AcceptanceError(f"duplicate selected anchor path: {anchor_text}")
        source_paths.add(source_text)
        anchor_paths.add(anchor_text)
        source_sha = _file_digest(source)
        anchor_sha = _file_digest(anchor)
        _assert_equal(
            source_sha,
            row["source_video_sha256"],
            f"selected iid={iid} source media SHA",
        )
        _assert_equal(
            anchor_sha,
            row["anchor_sha256"],
            f"selected iid={iid} anchor media SHA",
        )
        records.append(
            {
                "iid": iid,
                "source_path": source_text,
                "source_sha256": source_sha,
                "source_bytes": source.stat().st_size,
                "anchor_path": anchor_text,
                "anchor_sha256": anchor_sha,
                "anchor_bytes": anchor.stat().st_size,
            }
        )
        finalizer_verification: dict[str, Any] = {
            "schema_version": (
                "motive-goku-action-anchor-media-file-verification-v1"
            ),
            "source_video": {
                "resolved_path": source_text,
                "sha256": source_sha,
                "bytes": source.stat().st_size,
            },
            "anchor_image": {
                "resolved_path": anchor_text,
                "sha256": anchor_sha,
                "bytes": anchor.stat().st_size,
            },
        }
        finalizer_verification["verification_digest"] = _object_digest(
            finalizer_verification
        )
        finalizer_by_iid[iid] = finalizer_verification
    if source_paths & anchor_paths:
        raise AcceptanceError("selected source and anchor path sets overlap")
    return {
        "rows": len(records),
        "binding_digest": _object_digest(records),
        "finalizer_file_verification_by_iid": finalizer_by_iid,
        "finalizer_file_verification_sha256": _object_digest(
            finalizer_by_iid
        ),
    }


OK_ROW_KEYS = {
    "iid",
    "group_id",
    "family",
    "status",
    "input_digest",
    "config_digest",
    "run_config_digest",
    "implementation_digest",
    "model_path",
    "model_revision",
    "transformers_version",
    "shard_index",
    "num_shards",
    "execution_manifest",
    "execution_manifest_sha256",
    "failure_stage",
    "pipeline_stage",
    "pipeline_decision",
    "target_admissibility_raw",
    "target_admissibility_prompt_digest",
    "target_admissibility_visual_input_digest",
    "target_admissibility",
    "target_admissibility_resolved_evidence",
    "target_admissibility_validated_from",
    "target_admissibility_repairs",
    "target_admissibility_aggregate",
    "target_admissibility_failure_stage",
    "compatibility_raw",
    "compatibility_prompt_digest",
    "compatibility",
    "compatibility_initial_validated_from",
    "compatibility_validated_from",
    "compatibility_repairs",
    "compatibility_semantic_repairs",
    "compatibility_failure_stage",
    "draft_continuity_raw",
    "draft_continuity_prompt_digest",
    "draft_continuity",
    "draft_continuity_resolved_evidence",
    "draft_continuity_validated_from",
    "draft_continuity_repairs",
    "draft_continuity_aggregate",
    "draft_continuity_failure_stage",
    "deterministic_risk_codes",
    "anchor_observation_failure_stage",
    "media_verification",
    "resolved_src_video",
    "resolved_anchor_image",
    "visual_input_digest",
    "anchor_observation_raw",
    "anchor_observation_repairs",
    "anchor_observation",
    "anchor_observation_validated_from",
    "anchor_observation_digest",
    "result_digest",
    "provenance_digest",
}

SEMANTIC_REPAIR_KEYS = {
    "attempt",
    "status",
    "draft_compatibility",
    "draft_digest",
    "draft_target_core_digest",
    "judge_before_raw",
    "judge_before_prompt_digest",
    "judge_before_repairs",
    "judge_before",
    "judge_before_resolved_evidence",
    "judge_before_digest",
    "judge_before_validated_from",
    "judge_before_aggregate",
    "judge_before_failure_stage",
    "repair_codes",
    "repair_raw",
    "repair_prompt_digest",
    "repair_failure_stage",
    "repair_validated_from",
    "frozen_target_core_digest",
    "repaired_target_core_digest",
    "repaired_digest",
    "judge_after_raw",
    "judge_after_prompt_digest",
    "judge_after_repairs",
    "judge_after_failure_stage",
    "judge_after",
    "judge_after_resolved_evidence",
    "judge_after_digest",
    "judge_after_validated_from",
    "judge_after_aggregate",
    "error_type",
    "error",
}


def _result_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_observation": record["anchor_observation"],
        "target_admissibility": record["target_admissibility"],
        "target_admissibility_resolved_evidence": record[
            "target_admissibility_resolved_evidence"
        ],
        "target_admissibility_aggregate": record[
            "target_admissibility_aggregate"
        ],
        "compatibility": record["compatibility"],
        "draft_continuity": record["draft_continuity"],
        "draft_continuity_resolved_evidence": record[
            "draft_continuity_resolved_evidence"
        ],
        "draft_continuity_aggregate": record[
            "draft_continuity_aggregate"
        ],
        "deterministic_risk_codes": record["deterministic_risk_codes"],
        "pipeline_stage": record["pipeline_stage"],
        "pipeline_decision": record["pipeline_decision"],
    }


def _provenance_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": QWEN_PROVENANCE_SCHEMA,
        "iid": record["iid"],
        "input_digest": record["input_digest"],
        "config_digest": record["config_digest"],
        "run_config_digest": record["run_config_digest"],
        "implementation_digest": record["implementation_digest"],
        "execution_manifest": record["execution_manifest"],
        "execution_manifest_sha256": record["execution_manifest_sha256"],
        "shard_index": record["shard_index"],
        "num_shards": record["num_shards"],
        "model_path": record["model_path"],
        "model_revision": record["model_revision"],
        "transformers_version": record["transformers_version"],
        "media_verification": record["media_verification"],
        "visual_input_digest": record["visual_input_digest"],
        "anchor_observation_raw": record["anchor_observation_raw"],
        "anchor_observation_digest": record["anchor_observation_digest"],
        "anchor_observation_failure_stage": record.get(
            "anchor_observation_failure_stage"
        ),
        "anchor_observation_validated_from": record[
            "anchor_observation_validated_from"
        ],
        "anchor_observation_repairs": record["anchor_observation_repairs"],
        "target_admissibility_raw": record["target_admissibility_raw"],
        "target_admissibility_prompt_digest": record[
            "target_admissibility_prompt_digest"
        ],
        "target_admissibility_visual_input_digest": record[
            "target_admissibility_visual_input_digest"
        ],
        "target_admissibility_resolved_evidence": record[
            "target_admissibility_resolved_evidence"
        ],
        "target_admissibility_validated_from": record[
            "target_admissibility_validated_from"
        ],
        "target_admissibility_repairs": record[
            "target_admissibility_repairs"
        ],
        "target_admissibility_aggregate": record[
            "target_admissibility_aggregate"
        ],
        "target_admissibility_failure_stage": record.get(
            "target_admissibility_failure_stage"
        ),
        "compatibility_raw": record["compatibility_raw"],
        "compatibility_prompt_digest": record["compatibility_prompt_digest"],
        "compatibility_initial_validated_from": record[
            "compatibility_initial_validated_from"
        ],
        "compatibility_repairs": record["compatibility_repairs"],
        "compatibility_validated_from": record[
            "compatibility_validated_from"
        ],
        "compatibility_semantic_repairs": record[
            "compatibility_semantic_repairs"
        ],
        "compatibility_failure_stage": record.get(
            "compatibility_failure_stage"
        ),
        "draft_continuity_raw": record["draft_continuity_raw"],
        "draft_continuity_prompt_digest": record[
            "draft_continuity_prompt_digest"
        ],
        "draft_continuity_resolved_evidence": record[
            "draft_continuity_resolved_evidence"
        ],
        "draft_continuity_validated_from": record[
            "draft_continuity_validated_from"
        ],
        "draft_continuity_repairs": record["draft_continuity_repairs"],
        "draft_continuity_aggregate": record[
            "draft_continuity_aggregate"
        ],
        "draft_continuity_failure_stage": record.get(
            "draft_continuity_failure_stage"
        ),
        "deterministic_risk_codes": record["deterministic_risk_codes"],
        "pipeline_stage": record["pipeline_stage"],
        "pipeline_decision": record["pipeline_decision"],
        "failure_stage": record["failure_stage"],
        "result_digest": record["result_digest"],
    }


def _expected_run_config(
    *,
    constants: Mapping[str, Any],
    model_path: str,
    model_revision: str,
    transformers_version: str,
    implementation_digest: str,
) -> dict[str, Any]:
    return {
        "model_path": model_path,
        "model_revision": model_revision,
        "transformers_version": transformers_version,
        "max_samples": None,
        "num_shards": 8,
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
        "blind_prompt_digest": _text_digest(
            str(constants["BLIND_SYSTEM"])
            + "\n"
            + str(constants["BLIND_PROMPT"])
        ),
        "compatibility_prompt_digest": _text_digest(
            str(constants["COMPATIBILITY_SYSTEM"])
            + "\n"
            + str(constants["COMPATIBILITY_PROMPT"])
        ),
        "judge_a_prompt_digest": _text_digest(
            str(constants["JUDGE_A_SYSTEM"])
            + "\n"
            + str(constants["JUDGE_A_PROMPT"])
        ),
        "judge_b_prompt_digest": _text_digest(
            str(constants["JUDGE_B_SYSTEM"])
            + "\n"
            + str(constants["JUDGE_B_PROMPT"])
        ),
        "draft_repair_prompt_digest": _text_digest(
            str(constants["DRAFT_REPAIR_SYSTEM"])
            + "\n"
            + str(constants["DRAFT_REPAIR_PROMPT"])
        ),
        "repair_schema_digest": _object_digest(
            {
                "anchor_observation": constants[
                    "ANCHOR_OBSERVATION_REPAIR_SCHEMA"
                ],
                "compatibility": constants["COMPATIBILITY_REPAIR_SCHEMA"],
                "target_admissibility": constants[
                    "TARGET_ADMISSIBILITY_REPAIR_SCHEMA"
                ],
                "draft_continuity": constants[
                    "DRAFT_CONTINUITY_REPAIR_SCHEMA"
                ],
            }
        ),
        "implementation_digest": implementation_digest,
        "generation": {
            "do_sample": False,
            "visual_input": "exact_i0_plus_source_mosaic",
        },
    }


def _verify_receipts_and_load_rows(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    selected_path: Path,
    selected_sha256: str,
    qwen_root: Path,
    qwen_implementation_sha256: str,
    constants: Mapping[str, Any],
    completion: Mapping[str, Any],
    expected_shard_counts: Sequence[int],
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    root = _directory(qwen_root, context="Qwen root")
    expected_shards = [
        root / f"qwen_shard_{index:03d}.jsonl" for index in range(8)
    ]
    expected_receipts = [
        path.with_name(f"{path.stem}.receipt.json") for path in expected_shards
    ]
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise AcceptanceError(
                f"Qwen root contains non-regular entry: {entry.name}"
            )
    actual_shards = set(root.glob("qwen_shard_*.jsonl"))
    actual_receipts = set(root.glob("qwen_shard_*.receipt.json"))
    _assert_equal(actual_shards, set(expected_shards), "Qwen shard file set")
    _assert_equal(
        actual_receipts, set(expected_receipts), "Qwen receipt file set"
    )
    _assert_equal(
        {entry.resolve(strict=True) for entry in entries},
        set(expected_shards) | set(expected_receipts),
        "Qwen root exact file closure",
    )

    selected_iids = [str(row["iid"]) for row in selected_rows]
    assignments = [
        [iid for iid in selected_iids if _iid_shard(iid) == index]
        for index in range(8)
    ]
    _assert_equal(
        [len(items) for items in assignments],
        list(expected_shard_counts),
        "derived shard counts",
    )

    rows_by_iid: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    completion_shards = completion["qwen_shards"]
    for index, (shard_path, receipt_path) in enumerate(
        zip(expected_shards, expected_receipts)
    ):
        rows, raw, resolved_shard = _load_jsonl(
            shard_path,
            context=f"Qwen shard {index}",
            allow_empty=True,
        )
        receipt, receipt_raw, resolved_receipt = _load_json(
            receipt_path, context=f"Qwen receipt {index}"
        )
        _require_compact_json_bytes(
            receipt, receipt_raw, context=f"Qwen receipt {index}"
        )
        _exact_keys(
            receipt,
            {
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
            },
            f"Qwen receipt {index}",
        )
        expected_basic = {
            "schema_version": SHARD_RECEIPT_SCHEMA,
            "status": "complete",
            "execution_manifest": str(selected_path),
            "execution_manifest_sha256": selected_sha256,
            "shard_index": index,
            "num_shards": 8,
            "assigned_iids": assignments[index],
            "implementation_digest": qwen_implementation_sha256,
        }
        for field, expected in expected_basic.items():
            _assert_equal(
                receipt[field], expected, f"Qwen receipt {index}.{field}"
            )
        receipt_root = _directory(
            Path(str(receipt["root"])), context=f"receipt {index} root"
        )
        _assert_equal(
            str(receipt_root), receipt["root"], f"receipt {index} root canonical"
        )
        for field in (
            "model_path",
            "model_revision",
            "transformers_version",
        ):
            _nonempty(receipt[field], f"receipt {index}.{field}")
        expected_run_config = _expected_run_config(
            constants=constants,
            model_path=receipt["model_path"],
            model_revision=receipt["model_revision"],
            transformers_version=receipt["transformers_version"],
            implementation_digest=qwen_implementation_sha256,
        )
        _assert_equal(
            receipt["run_config"],
            expected_run_config,
            f"receipt {index} full run_config",
        )
        run_config_digest = _object_digest(expected_run_config)
        _assert_equal(
            receipt["run_config_digest"],
            run_config_digest,
            f"receipt {index} run_config_digest",
        )
        config_digest = _object_digest(
            {
                "run_config_digest": run_config_digest,
                "execution_manifest": str(selected_path),
                "execution_manifest_sha256": selected_sha256,
                "root": str(receipt_root),
                "shard_index": index,
                "num_shards": 8,
            }
        )
        _assert_equal(
            receipt["config_digest"],
            config_digest,
            f"receipt {index} config_digest",
        )
        actual_iids = [str(row.get("iid", "")) for row in rows]
        _assert_equal(actual_iids, assignments[index], f"shard {index} IID order")
        status_counts = dict(
            sorted(Counter(str(row.get("status", "missing")) for row in rows).items())
        )
        expected_output = {
            "path": str(resolved_shard),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": len(rows),
            "status_counts": status_counts,
        }
        _assert_equal(
            receipt["output"], expected_output, f"receipt {index} output"
        )
        receipt_payload = dict(receipt)
        stored_receipt_digest = receipt_payload.pop("receipt_digest")
        _assert_equal(
            stored_receipt_digest,
            _object_digest(receipt_payload),
            f"receipt {index} digest",
        )
        if not assignments[index]:
            _assert_equal(raw, b"", f"empty shard {index} bytes")
            _assert_equal(
                expected_output["sha256"],
                EMPTY_SHA256,
                f"empty shard {index} SHA",
            )

        completion_item = completion_shards[index]
        expected_completion = {
            "index": index,
            "path": str(resolved_shard),
            "sha256": expected_output["sha256"],
            "bytes": len(raw),
            "receipt_path": str(resolved_receipt),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
        _assert_equal(
            completion_item,
            expected_completion,
            f"completion shard {index}",
        )
        shard_records.append(
            {
                "index": index,
                "path": str(resolved_shard),
                "rows": len(rows),
                "sha256": expected_output["sha256"],
                "bytes": len(raw),
                "receipt_path": str(resolved_receipt),
                "receipt_sha256": expected_completion["receipt_sha256"],
            }
        )
        receipt_records.append(
            {
                "index": index,
                "path": str(resolved_receipt),
                "sha256": expected_completion["receipt_sha256"],
                "receipt_digest": stored_receipt_digest,
                "assigned_rows": len(assignments[index]),
            }
        )
        identity = {
            "run_config_digest": run_config_digest,
            "implementation_digest": qwen_implementation_sha256,
            "model_path": receipt["model_path"],
            "model_revision": receipt["model_revision"],
            "transformers_version": receipt["transformers_version"],
            "root": str(receipt_root),
        }
        if common is None:
            common = identity
        else:
            _assert_equal(identity, common, f"receipt {index} common identity")
        for row in rows:
            iid = str(row["iid"])
            if iid in rows_by_iid:
                raise AcceptanceError(f"duplicate Qwen IID: {iid}")
            rows_by_iid[iid] = row
    if common is None:
        raise AssertionError("eight receipts produced no common identity")
    _assert_equal(set(rows_by_iid), set(selected_iids), "Qwen IID coverage")
    return rows_by_iid, shard_records, receipt_records, common


def _verify_common_row(
    record: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    selected_path: Path,
    selected_sha256: str,
    constants: Mapping[str, Any],
    common: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    iid = str(selected["iid"])
    _exact_keys(record, OK_ROW_KEYS, f"Qwen iid={iid}")
    expected_common = {
        "iid": iid,
        "group_id": selected["group_id"],
        "family": selected["family"],
        "status": "ok",
        "input_digest": _object_digest(selected),
        "run_config_digest": common["run_config_digest"],
        "implementation_digest": common["implementation_digest"],
        "model_path": common["model_path"],
        "model_revision": common["model_revision"],
        "transformers_version": common["transformers_version"],
        "shard_index": _iid_shard(iid),
        "num_shards": 8,
        "execution_manifest": str(selected_path),
        "execution_manifest_sha256": selected_sha256,
        "failure_stage": None,
        "resolved_src_video": selected["resolved_src_video"],
        "resolved_anchor_image": selected["resolved_anchor_image"],
    }
    for field, expected in expected_common.items():
        _assert_equal(record[field], expected, f"Qwen iid={iid}.{field}")
    _sha(record["config_digest"], f"Qwen iid={iid}.config_digest")
    expected_config = _object_digest(
        {
            "run_config_digest": common["run_config_digest"],
            "execution_manifest": str(selected_path),
            "execution_manifest_sha256": selected_sha256,
            "root": common["root"],
            "shard_index": _iid_shard(iid),
            "num_shards": 8,
        }
    )
    _assert_equal(
        record["config_digest"], expected_config, f"Qwen iid={iid}.config_digest"
    )
    observation = _validate_observation(record["anchor_observation"], iid)
    _raw_equals(
        record["anchor_observation_raw"],
        observation,
        context=f"Qwen iid={iid}.anchor_observation_raw",
    )
    _assert_equal(
        record["anchor_observation_digest"],
        _object_digest(observation),
        f"Qwen iid={iid}.anchor_observation_digest",
    )
    _assert_equal(
        record["anchor_observation_validated_from"],
        "original",
        f"Qwen iid={iid}.anchor observation source",
    )
    _assert_equal(
        record["anchor_observation_repairs"],
        [],
        f"Qwen iid={iid}.anchor repairs",
    )
    _assert_equal(
        record["anchor_observation_failure_stage"],
        None,
        f"Qwen iid={iid}.anchor failure stage",
    )
    _sha(record["visual_input_digest"], f"Qwen iid={iid}.visual_input_digest")

    media = record["media_verification"]
    if not isinstance(media, Mapping):
        raise AcceptanceError(f"Qwen iid={iid}: media verification not object")
    _exact_keys(
        media,
        {
            "exact_i0",
            "lossless_png",
            "width",
            "height",
            "anchor_sha256",
            "source_video_sha256",
            "frame_zero_rgb_sha256",
        },
        f"Qwen iid={iid}.media",
    )
    _assert_equal(media["exact_i0"], True, f"Qwen iid={iid}.exact_i0")
    _assert_equal(media["lossless_png"], True, f"Qwen iid={iid}.lossless_png")
    selected_media = selected["media"]
    if not isinstance(selected_media, Mapping):
        raise AcceptanceError(f"selected iid={iid}.media is invalid")
    expected_width = selected_media.get(
        "anchor_width", selected_media.get("width")
    )
    expected_height = selected_media.get(
        "anchor_height", selected_media.get("height")
    )
    _assert_equal(media["width"], expected_width, f"Qwen iid={iid}.width")
    _assert_equal(media["height"], expected_height, f"Qwen iid={iid}.height")
    _assert_equal(
        media["anchor_sha256"],
        selected["anchor_sha256"],
        f"Qwen iid={iid}.anchor SHA",
    )
    _assert_equal(
        media["source_video_sha256"],
        selected["source_video_sha256"],
        f"Qwen iid={iid}.source SHA",
    )
    _sha(media["frame_zero_rgb_sha256"], f"Qwen iid={iid}.frame zero SHA")

    judge_a = _validate_judge_a(record["target_admissibility"], iid)
    if judge_a["confidence"] not in {"medium", "high"}:
        raise AcceptanceError(
            f"Qwen iid={iid}.Judge-A confidence is not medium/high"
        )
    _raw_equals(
        record["target_admissibility_raw"],
        judge_a,
        context=f"Qwen iid={iid}.target_admissibility_raw",
    )
    _assert_equal(
        record["target_admissibility_validated_from"],
        "original",
        f"Qwen iid={iid}.Judge-A source",
    )
    _assert_equal(
        record["target_admissibility_repairs"],
        [],
        f"Qwen iid={iid}.Judge-A repairs",
    )
    _assert_equal(
        record["target_admissibility_failure_stage"],
        None,
        f"Qwen iid={iid}.Judge-A failure stage",
    )
    _, expected_a_prompt = _render_prompt(
        "judge_a",
        constants,
        row=selected,
        observation=observation,
    )
    _assert_equal(
        record["target_admissibility_prompt_digest"],
        expected_a_prompt,
        f"Qwen iid={iid}.Judge-A prompt",
    )
    _assert_equal(
        record["target_admissibility_visual_input_digest"],
        record["visual_input_digest"],
        f"Qwen iid={iid}.Judge-A visual input",
    )
    _assert_equal(
        record["target_admissibility_resolved_evidence"],
        _judge_a_evidence(judge_a, selected, observation, iid),
        f"Qwen iid={iid}.Judge-A resolved evidence",
    )
    aggregate_a = _aggregate_a(judge_a, selected, observation)
    _assert_equal(
        record["target_admissibility_aggregate"],
        aggregate_a,
        f"Qwen iid={iid}.Judge-A aggregate",
    )
    return observation, judge_a, aggregate_a


def _verify_repair_route(
    record: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    observation: Mapping[str, Any],
    judge_a: Mapping[str, Any],
    constants: Mapping[str, Any],
) -> Mapping[str, Any]:
    iid = str(selected["iid"])
    repairs = record["compatibility_semantic_repairs"]
    if not isinstance(repairs, list) or len(repairs) != 1:
        raise AcceptanceError(f"Qwen iid={iid}: repair count is not one")
    entry = repairs[0]
    if not isinstance(entry, Mapping):
        raise AcceptanceError(f"Qwen iid={iid}: repair entry is not object")
    _exact_keys(entry, SEMANTIC_REPAIR_KEYS, f"Qwen iid={iid}.repair")
    expected_terminals = {
        "attempt": 1,
        "status": "ok",
        "error_type": None,
        "error": None,
        "judge_before_validated_from": "original",
        "judge_before_repairs": [],
        "judge_before_failure_stage": None,
        "repair_validated_from": "original",
        "repair_failure_stage": None,
        "judge_after_validated_from": "original",
        "judge_after_repairs": [],
        "judge_after_failure_stage": None,
    }
    for field, expected in expected_terminals.items():
        _assert_equal(entry[field], expected, f"Qwen iid={iid}.repair.{field}")
    draft = _validate_compatibility(
        entry["draft_compatibility"], f"{iid}.draft"
    )
    final = _validate_compatibility(record["compatibility"], f"{iid}.final")
    for label, candidate in (("draft", draft), ("final", final)):
        agreement = _target_core_agreement(judge_a, candidate, selected)
        _assert_equal(
            agreement["agreement_verified"],
            True,
            f"Qwen iid={iid}.{label} exact Judge-A target core",
        )
    judge_before = _validate_judge_b(
        entry["judge_before"], f"{iid}.judge_before"
    )
    judge_after = _validate_judge_b(
        entry["judge_after"], f"{iid}.judge_after"
    )
    _raw_equals(
        record["compatibility_raw"],
        draft,
        context=f"Qwen iid={iid}.compatibility_raw",
    )
    _raw_equals(
        entry["judge_before_raw"],
        judge_before,
        context=f"Qwen iid={iid}.judge_before_raw",
    )
    _raw_equals(
        entry["repair_raw"],
        final,
        context=f"Qwen iid={iid}.repair_raw",
    )
    _raw_equals(
        entry["judge_after_raw"],
        judge_after,
        context=f"Qwen iid={iid}.judge_after_raw",
    )
    _raw_equals(
        record["draft_continuity_raw"],
        judge_after,
        context=f"Qwen iid={iid}.final_judge_b_raw",
    )
    _assert_equal(
        record["draft_continuity"],
        judge_after,
        f"Qwen iid={iid}.final Judge B object",
    )
    draft_core = _target_core(draft)
    final_core = _target_core(final)
    _assert_equal(final_core, draft_core, f"Qwen iid={iid}.target core")
    core_digest = _object_digest(draft_core)
    expected_digests = {
        "draft_digest": _object_digest(draft),
        "draft_target_core_digest": core_digest,
        "frozen_target_core_digest": core_digest,
        "repaired_target_core_digest": core_digest,
        "repaired_digest": _object_digest(final),
        "judge_before_digest": _object_digest(judge_before),
        "judge_after_digest": _object_digest(judge_after),
    }
    for field, expected in expected_digests.items():
        _assert_equal(entry[field], expected, f"Qwen iid={iid}.repair.{field}")
    before = _aggregate_b(judge_before, draft, observation)
    after = _aggregate_b(judge_after, final, observation)
    _assert_equal(before["decision"], "repair", f"Qwen iid={iid}.before route")
    _assert_equal(after["decision"], "pass", f"Qwen iid={iid}.after route")
    _assert_equal(
        entry["judge_before_aggregate"],
        before,
        f"Qwen iid={iid}.before aggregate",
    )
    _assert_equal(
        entry["judge_after_aggregate"],
        after,
        f"Qwen iid={iid}.after aggregate",
    )
    _assert_equal(
        entry["judge_before_resolved_evidence"],
        _judge_b_evidence(judge_before, draft, f"Qwen iid={iid}.before"),
        f"Qwen iid={iid}.before resolved evidence",
    )
    _assert_equal(
        entry["judge_after_resolved_evidence"],
        _judge_b_evidence(judge_after, final, f"Qwen iid={iid}.after"),
        f"Qwen iid={iid}.after resolved evidence",
    )
    _assert_equal(
        entry["repair_codes"], before["repair_codes"], f"Qwen iid={iid}.repair codes"
    )
    _, before_prompt = _render_prompt(
        "judge_b",
        constants,
        row=selected,
        observation=observation,
        judge_a=judge_a,
        compatibility=draft,
    )
    _, repair_prompt = _render_prompt(
        "repair",
        constants,
        row=selected,
        observation=observation,
        judge_a=judge_a,
        compatibility=draft,
        judge_b=judge_before,
        repair_codes=before["repair_codes"],
    )
    _, after_prompt = _render_prompt(
        "judge_b",
        constants,
        row=selected,
        observation=observation,
        judge_a=judge_a,
        compatibility=final,
    )
    expected_prompts = {
        "judge_before_prompt_digest": before_prompt,
        "repair_prompt_digest": repair_prompt,
        "judge_after_prompt_digest": after_prompt,
    }
    for field, expected in expected_prompts.items():
        _assert_equal(entry[field], expected, f"Qwen iid={iid}.repair.{field}")
    _assert_equal(
        record["compatibility_validated_from"],
        "semantic_repair_1",
        f"Qwen iid={iid}.final writer source",
    )
    _assert_equal(
        record["draft_continuity_prompt_digest"],
        after_prompt,
        f"Qwen iid={iid}.final Judge-B prompt",
    )
    _assert_equal(
        record["draft_continuity_aggregate"],
        after,
        f"Qwen iid={iid}.final Judge-B aggregate",
    )
    return final


def _positive_route(
    record: Mapping[str, Any],
    *,
    context: str,
) -> str:
    """Classify a positive writer route without consulting IID identity."""

    source = record.get("compatibility_validated_from")
    repairs = record.get("compatibility_semantic_repairs")
    if source == "original" and repairs == []:
        return "direct"
    if (
        source == "semantic_repair_1"
        and isinstance(repairs, list)
        and len(repairs) == 1
    ):
        return "repair_once"
    raise AcceptanceError(
        f"{context}: positive route is neither direct nor exactly one "
        "semantic repair"
    )


def _verify_qwen_rows(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    selected_path: Path,
    selected_sha256: str,
    rows_by_iid: Mapping[str, Mapping[str, Any]],
    constants: Mapping[str, Any],
    common: Mapping[str, Any],
    gold_labels: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    observed_routes: dict[str, str] = {}
    observed_labels: dict[str, str] = {}
    semantic_contracts: dict[str, dict[str, Any]] = {}
    for selected in selected_rows:
        iid = str(selected["iid"])
        record = rows_by_iid[iid]
        observation, judge_a, aggregate_a = _verify_common_row(
            record,
            selected=selected,
            selected_path=selected_path,
            selected_sha256=selected_sha256,
            constants=constants,
            common=common,
        )
        expected_label = str(gold_labels[iid]["label"])
        semantic_contracts[iid] = _verify_gold_target_semantics(
            record=record,
            selected=selected,
            gold_label=gold_labels[iid],
        )
        if expected_label == "inadmissible":
            observed_routes[iid] = "judge_a_reject"
            observed_labels[iid] = "inadmissible"
            _assert_equal(
                aggregate_a["decision"], "reject", f"Qwen iid={iid}.route"
            )
            null_fields = (
                "compatibility_raw",
                "compatibility_prompt_digest",
                "compatibility",
                "compatibility_initial_validated_from",
                "compatibility_validated_from",
                "compatibility_failure_stage",
                "draft_continuity_raw",
                "draft_continuity_prompt_digest",
                "draft_continuity",
                "draft_continuity_resolved_evidence",
                "draft_continuity_validated_from",
                "draft_continuity_aggregate",
                "draft_continuity_failure_stage",
            )
            for field in null_fields:
                _assert_equal(
                    record[field], None, f"Qwen iid={iid}.short circuit {field}"
                )
            for field in (
                "compatibility_repairs",
                "compatibility_semantic_repairs",
                "draft_continuity_repairs",
            ):
                _assert_equal(
                    record[field], [], f"Qwen iid={iid}.short circuit {field}"
                )
            _assert_equal(
                record["pipeline_stage"], "judge_a", f"Qwen iid={iid}.stage"
            )
            _assert_equal(
                record["pipeline_decision"], "reject", f"Qwen iid={iid}.decision"
            )
            _assert_equal(
                record["deterministic_risk_codes"],
                aggregate_a["risk_codes"],
                f"Qwen iid={iid}.risks",
            )
        else:
            _assert_equal(
                expected_label, "admissible", f"Qwen iid={iid}.gold label"
            )
            observed_labels[iid] = "admissible"
            _assert_equal(
                aggregate_a["decision"], "pass", f"Qwen iid={iid}.route"
            )
            _verify_hard_pass_anchor(observation, context=f"Qwen iid={iid}")
            compatibility = _validate_compatibility(
                record["compatibility"], f"{iid}.compatibility"
            )
            target_core_agreement = _target_core_agreement(
                judge_a,
                compatibility,
                selected,
            )
            _assert_equal(
                target_core_agreement["agreement_verified"],
                True,
                f"Qwen iid={iid}.exact Judge-A target core",
            )
            _, writer_prompt = _render_prompt(
                "writer",
                constants,
                row=selected,
                observation=observation,
                judge_a=judge_a,
            )
            _assert_equal(
                record["compatibility_prompt_digest"],
                writer_prompt,
                f"Qwen iid={iid}.writer prompt",
            )
            _assert_equal(
                record["compatibility_initial_validated_from"],
                "original",
                f"Qwen iid={iid}.initial writer source",
            )
            _assert_equal(
                record["compatibility_repairs"],
                [],
                f"Qwen iid={iid}.writer generic repairs",
            )
            _assert_equal(
                record["compatibility_failure_stage"],
                None,
                f"Qwen iid={iid}.writer failure stage",
            )
            positive_route = _positive_route(
                record, context=f"Qwen iid={iid}"
            )
            if positive_route == "direct":
                observed_routes[iid] = "direct"
                _raw_equals(
                    record["compatibility_raw"],
                    compatibility,
                    context=f"Qwen iid={iid}.compatibility_raw",
                )
                judge_b = _validate_judge_b(
                    record["draft_continuity"], f"{iid}.judge_b"
                )
                _raw_equals(
                    record["draft_continuity_raw"],
                    judge_b,
                    context=f"Qwen iid={iid}.draft_continuity_raw",
                )
                _, judge_b_prompt = _render_prompt(
                    "judge_b",
                    constants,
                    row=selected,
                    observation=observation,
                    judge_a=judge_a,
                    compatibility=compatibility,
                )
                _assert_equal(
                    record["draft_continuity_prompt_digest"],
                    judge_b_prompt,
                    f"Qwen iid={iid}.Judge-B prompt",
                )
                aggregate_b = _aggregate_b(
                    judge_b, compatibility, observation
                )
                _assert_equal(
                    aggregate_b["decision"], "pass", f"Qwen iid={iid}.Judge-B"
                )
                _assert_equal(
                    record["draft_continuity_aggregate"],
                    aggregate_b,
                    f"Qwen iid={iid}.Judge-B aggregate",
                )
            else:
                observed_routes[iid] = "repair_once"
                compatibility = _verify_repair_route(
                    record,
                    selected=selected,
                    observation=observation,
                    judge_a=judge_a,
                    constants=constants,
                )
                judge_b = record["draft_continuity"]
                aggregate_b = record["draft_continuity_aggregate"]
            _assert_equal(
                record["draft_continuity_resolved_evidence"],
                _judge_b_evidence(judge_b, compatibility, f"Qwen iid={iid}"),
                f"Qwen iid={iid}.Judge-B resolved evidence",
            )
            _assert_equal(
                record["draft_continuity_validated_from"],
                "original",
                f"Qwen iid={iid}.Judge-B source",
            )
            _assert_equal(
                record["draft_continuity_repairs"],
                [],
                f"Qwen iid={iid}.Judge-B generic repairs",
            )
            _assert_equal(
                record["draft_continuity_failure_stage"],
                None,
                f"Qwen iid={iid}.Judge-B failure stage",
            )
            _assert_equal(
                record["pipeline_stage"], "judge_b", f"Qwen iid={iid}.stage"
            )
            _assert_equal(
                record["pipeline_decision"], "pass", f"Qwen iid={iid}.decision"
            )
            _assert_equal(
                record["deterministic_risk_codes"],
                [],
                f"Qwen iid={iid}.risks",
            )
            _assert_equal(
                aggregate_b["risk_codes"], [], f"Qwen iid={iid}.aggregate risks"
            )

        _assert_equal(
            record["result_digest"],
            _object_digest(_result_payload(record)),
            f"Qwen iid={iid}.result_digest",
        )
        _assert_equal(
            record["provenance_digest"],
            _object_digest(_provenance_payload(record)),
            f"Qwen iid={iid}.provenance_digest",
        )
    expected_labels = {
        iid: str(item["label"]) for iid, item in gold_labels.items()
    }
    _assert_equal(
        observed_labels, expected_labels, "smoke gold binary accuracy"
    )
    return (
        dict(sorted(observed_routes.items())),
        {
            "expected_labels": dict(sorted(expected_labels.items())),
            "observed_labels": dict(sorted(observed_labels.items())),
            "correct_rows": len(observed_labels),
            "total_rows": len(expected_labels),
            "route_counts": dict(
                sorted(Counter(observed_routes.values()).items())
            ),
            "semantic_contract_rows": len(semantic_contracts),
            "semantic_contracts": dict(sorted(semantic_contracts.items())),
        },
    )


ACTION_CATEGORIES = (
    "locomotion",
    "posture",
    "interaction",
    "articulated",
)
FAMILY_QUOTAS = {
    "locomotion": 32,
    "posture": 32,
    "interaction": 48,
    "articulated": 16,
}
REVIEW_LIMIT = 192
PROPOSED_SIZE = 128
RESERVE_SIZE = 32
MAX_PER_TARGET_VERB = 12


def _quality_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    qwen = row["qwen_action_anchor"]
    anchor = qwen["anchor_observation"]
    compatibility = qwen["compatibility"]
    judge_a = qwen["target_admissibility"]
    judge_b = qwen["draft_continuity"]
    iid = str(row["iid"])
    tie = hashlib.sha256(f"260730\0{iid}".encode("utf-8")).hexdigest()
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


def _scaled_quotas(size: int) -> dict[str, int]:
    if type(size) is not int or not 1 <= size <= PROPOSED_SIZE:
        raise AcceptanceError("proposed size is outside [1, 128]")
    exact = {
        category: FAMILY_QUOTAS[category] * size / PROPOSED_SIZE
        for category in ACTION_CATEGORIES
    }
    quotas = {
        category: math.floor(exact[category])
        for category in ACTION_CATEGORIES
    }
    remainder = size - sum(quotas.values())
    order = {category: index for index, category in enumerate(ACTION_CATEGORIES)}
    ranked = sorted(
        ACTION_CATEGORIES,
        key=lambda category: (
            -(exact[category] - quotas[category]),
            order[category],
        ),
    )
    for category in ranked[:remainder]:
        quotas[category] += 1
    return quotas


def _select_proposed(
    review: Sequence[Mapping[str, Any]],
    *,
    proposed_size: int,
) -> tuple[list[Mapping[str, Any]], dict[str, int], dict[str, int]]:
    quotas = _scaled_quotas(proposed_size)
    selected: list[Mapping[str, Any]] = []
    selected_iids: set[str] = set()
    realized: Counter[str] = Counter()
    for category in ACTION_CATEGORIES:
        for row in review:
            if realized[category] >= quotas[category]:
                break
            iid = str(row["iid"])
            if iid in selected_iids:
                continue
            if (
                row["qwen_action_anchor"]["compatibility"]["action_category"]
                != category
            ):
                continue
            selected.append(row)
            selected_iids.add(iid)
            realized[category] += 1
    shortfall = {
        category: max(0, quotas[category] - realized[category])
        for category in ACTION_CATEGORIES
    }
    for row in review:
        if len(selected) >= proposed_size:
            break
        iid = str(row["iid"])
        if iid not in selected_iids:
            selected.append(row)
            selected_iids.add(iid)
            category = row["qwen_action_anchor"]["compatibility"][
                "action_category"
            ]
            realized[category] += 1
    if len(selected) != proposed_size:
        raise AcceptanceError("insufficient proposed candidates")
    return (
        selected,
        {
            category: realized[category]
            for category in ACTION_CATEGORIES
            if realized[category]
        },
        shortfall,
    )


def _annotated(
    row: Mapping[str, Any],
    *,
    rank: int,
    bucket: str,
) -> dict[str, Any]:
    result = dict(row)
    qwen = row["qwen_action_anchor"]
    compatibility = qwen["compatibility"]
    result["action_anchor_finalization"] = {
        "schema_version": FINAL_ROW_SCHEMA,
        "policy_version": POLICY_VERSION,
        "hard_gate_passed": True,
        "hard_gate_failures": [],
        "review_rank": rank,
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
        "target_support_evidence": _target_support(compatibility),
        "writer_instruction_target_support_evidence": (
            _writer_instruction_support(compatibility, row)
        ),
        "judge_a_writer_target_core_agreement_evidence": (
            _target_core_agreement(
                qwen["target_admissibility"], compatibility, row
            )
        ),
        "target_admissibility": qwen["target_admissibility"],
        "target_admissibility_aggregate": qwen[
            "target_admissibility_aggregate"
        ],
        "draft_continuity": qwen["draft_continuity"],
        "draft_continuity_aggregate": qwen["draft_continuity_aggregate"],
        "deterministic_risk_codes": qwen["deterministic_risk_codes"],
    }
    return result


def _strict_temporal_geometry(
    media: Mapping[str, Any],
    *,
    iid: str,
) -> dict[str, Any]:
    required = {"frame_count", "fps", "duration_seconds"}
    missing = sorted(required - set(media))
    if missing:
        raise AcceptanceError(
            f"selected iid={iid} media is missing temporal fields: {missing}"
        )
    frame_count = media["frame_count"]
    if type(frame_count) is not int or frame_count <= 0:
        raise AcceptanceError(
            f"selected iid={iid} media frame_count is invalid"
        )
    if frame_count % 4 != 1:
        raise AcceptanceError(
            f"selected iid={iid} media frame_count does not satisfy 4n+1"
        )
    fps = media["fps"]
    duration = media["duration_seconds"]
    if (
        type(fps) not in (int, float)
        or not math.isfinite(float(fps))
        or float(fps) <= 0.0
        or type(duration) not in (int, float)
        or not math.isfinite(float(duration))
        or float(duration) <= 0.0
    ):
        raise AcceptanceError(f"selected iid={iid} media timing is invalid")
    fps_value = float(fps)
    duration_value = float(duration)
    one_frame = 1.0 / fps_value
    timeline_duration = (frame_count - 1) / fps_value
    timeline_error = abs(duration_value - timeline_duration)
    if timeline_error > one_frame + max(1e-9, one_frame * 1e-7):
        raise AcceptanceError(
            f"selected iid={iid} duration differs by more than one frame"
        )
    return {
        "schema_version": "motive-goku-action-anchor-temporal-geometry-v1",
        "source_frame_count": frame_count,
        "required_output_frame_count": frame_count,
        "source_fps": fps_value,
        "required_output_fps": fps_value,
        "source_duration_seconds": duration_value,
        "required_output_duration_seconds": duration_value,
        "maximum_duration_delta_frames": 1,
        "maximum_duration_delta_seconds": one_frame,
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


def _generation_row(
    row: Mapping[str, Any],
    *,
    file_verification: Mapping[str, Any],
) -> dict[str, Any]:
    qwen = row["qwen_action_anchor"]
    compatibility = qwen["compatibility"]
    judge_a = qwen["target_admissibility"]
    return {
        "schema_version": GENERATION_SCHEMA,
        "iid": row["iid"],
        "group_id": row["group_id"],
        "action_category": compatibility["action_category"],
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
        "selected_media_evidence": dict(row["media"]),
        "selected_media_evidence_sha256": _object_digest(row["media"]),
        "strict_temporal_geometry": _strict_temporal_geometry(
            row["media"], iid=str(row["iid"])
        ),
        "finalizer_media_file_verification": dict(file_verification),
        "edit_instruction": row["prompt"],
        "edit_instruction_sha256": _text_digest(str(row["prompt"])),
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
        "target_admissibility": qwen["target_admissibility"],
        "target_admissibility_aggregate": qwen[
            "target_admissibility_aggregate"
        ],
        "draft_continuity": qwen["draft_continuity"],
        "draft_continuity_aggregate": qwen["draft_continuity_aggregate"],
        "deterministic_risk_codes": qwen["deterministic_risk_codes"],
        "resolved_anchor_image": row["resolved_anchor_image"],
    }


def _diverse_review_pool(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Counter[str]]:
    selected: list[Mapping[str, Any]] = []
    groups: set[str] = set()
    verb_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    for row in rows:
        group_id = str(row["group_id"])
        target_verb = str(
            row["qwen_action_anchor"]["compatibility"]["target_action_verb"]
        )
        if group_id in groups:
            rejections["duplicate_group_id"] += 1
            continue
        if verb_counts[target_verb] >= MAX_PER_TARGET_VERB:
            rejections["target_verb_cap"] += 1
            continue
        selected.append(row)
        groups.add(group_id)
        verb_counts[target_verb] += 1
        if len(selected) == REVIEW_LIMIT:
            break
    return selected, rejections


def _verify_final_artifacts(
    *,
    final_dir: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    rows_by_iid: Mapping[str, Mapping[str, Any]],
    selected_path: Path,
    selected_sha256: str,
    shard_records: Sequence[Mapping[str, Any]],
    receipt_records: Sequence[Mapping[str, Any]],
    common: Mapping[str, Any],
    finalizer_sha256: str,
    completion: Mapping[str, Any],
    observed_routes: Mapping[str, str],
    media_result: Mapping[str, Any],
) -> dict[str, Any]:
    root = _directory(final_dir, context="final directory")
    entries = list(root.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise AcceptanceError(
                f"final directory contains non-regular entry: {entry.name}"
            )
    actual_files = {path.name for path in entries}
    _assert_equal(actual_files, set(FINAL_NAMES), "final artifact file set")

    loaded: dict[str, list[dict[str, Any]]] = {}
    raw_by_name: dict[str, bytes] = {}
    artifact_records: dict[str, dict[str, Any]] = {}
    for name in JSONL_FINAL_NAMES:
        rows, raw, path = _load_jsonl(
            root / name,
            context=f"final {name}",
            allow_empty=name == "reserve_32.jsonl",
        )
        canonical = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
        _assert_equal(raw, canonical, f"final {name} canonical bytes")
        loaded[name] = rows
        raw_by_name[name] = raw
        artifact_records[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    selected_by_iid = {
        str(row["iid"]): row for row in selected_rows
    }
    fused = []
    pass_iids = {
        iid
        for iid, route in observed_routes.items()
        if route in {"direct", "repair_once"}
    }
    for iid in pass_iids:
        row = dict(selected_by_iid[iid])
        row["qwen_action_anchor"] = rows_by_iid[iid]
        fused.append(row)
    ranked = sorted(fused, key=_quality_key)
    review, diversity_rejections = _diverse_review_pool(ranked)
    review_rank = {
        str(row["iid"]): index
        for index, row in enumerate(review, start=1)
    }
    proposed_target = min(PROPOSED_SIZE, len(review))
    if proposed_target < 1:
        raise AcceptanceError("finalizer has no hard-pass review candidate")
    proposed, proposed_counts, quota_shortfall = _select_proposed(
        review, proposed_size=proposed_target
    )
    proposed_iids = {str(row["iid"]) for row in proposed}
    reserve_target = min(RESERVE_SIZE, len(review) - len(proposed))
    reserve = [
        row for row in review if str(row["iid"]) not in proposed_iids
    ][:reserve_target]
    reserve_iids = {str(row["iid"]) for row in reserve}
    expected_review = [
        _annotated(
            row,
            rank=review_rank[str(row["iid"])],
            bucket=(
                "proposed"
                if str(row["iid"]) in proposed_iids
                else (
                    "reserve"
                    if str(row["iid"]) in reserve_iids
                    else "review_only"
                )
            ),
        )
        for row in review
    ]
    expected_proposed = [
        _annotated(
            row,
            rank=review_rank[str(row["iid"])],
            bucket="proposed",
        )
        for row in proposed
    ]
    expected_reserve = [
        _annotated(
            row,
            rank=review_rank[str(row["iid"])],
            bucket="reserve",
        )
        for row in reserve
    ]
    file_verification_by_iid = media_result[
        "finalizer_file_verification_by_iid"
    ]
    expected_generation = [
        _generation_row(
            row,
            file_verification=file_verification_by_iid[str(row["iid"])],
        )
        for row in proposed
    ]
    _assert_equal(
        loaded["review_candidates.jsonl"],
        expected_review,
        "review_candidates exact content/order",
    )
    _assert_equal(
        loaded["proposed_128.jsonl"],
        expected_proposed,
        "proposed_128 exact content/order",
    )
    _assert_equal(
        loaded["reserve_32.jsonl"],
        expected_reserve,
        "reserve_32 exact content/order",
    )
    _assert_equal(
        loaded["generation_manifest.jsonl"],
        expected_generation,
        "generation_manifest exact pending content/order",
    )

    summary, summary_raw, summary_path = _load_json(
        root / "summary.json", context="final summary"
    )
    pretty_summary = (
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _assert_equal(summary_raw, pretty_summary, "summary canonical pretty bytes")
    raw_by_name["summary.json"] = summary_raw
    artifact_records["summary.json"] = {
        "path": str(summary_path),
        "sha256": hashlib.sha256(summary_raw).hexdigest(),
        "bytes": len(summary_raw),
    }
    review_categories = Counter(
        row["qwen_action_anchor"]["compatibility"]["action_category"]
        for row in review
    )
    verb_counts = Counter(
        row["qwen_action_anchor"]["compatibility"]["target_action_verb"]
        for row in proposed
    )
    config_by_shard = {
        str(index): _object_digest(
            {
                "run_config_digest": common["run_config_digest"],
                "execution_manifest": str(selected_path),
                "execution_manifest_sha256": selected_sha256,
                "root": common["root"],
                "shard_index": index,
                "num_shards": 8,
            }
        )
        for index in range(8)
    }
    hard_rejected = len(selected_rows) - len(pass_iids)
    hard_rejection_counter: Counter[str] = Counter()
    for selected in selected_rows:
        iid = str(selected["iid"])
        if iid in pass_iids:
            continue
        qwen = rows_by_iid[iid]
        support = _judge_a_instruction_support(
            qwen["target_admissibility"],
            row=selected,
            observation=qwen["anchor_observation"],
        )
        for field in (
            "target_evidence_ref_is_instruction",
            "target_action_normalized_supports_verb",
            "instruction_supports_target_action",
            "novel_trajectory_description_supports_target_action",
        ):
            if support[field] is not True:
                hard_rejection_counter[
                    f"judge_a:immutable_instruction_support:{field}"
                ] += 1
        hard_rejection_counter["judge_a:not_pass"] += 1
    hard_rejection_counts = dict(sorted(hard_rejection_counter.items()))
    reserve_categories = Counter(
        row["qwen_action_anchor"]["compatibility"]["action_category"]
        for row in reserve
    )
    expected_summary = {
        "schema_version": SUMMARY_SCHEMA,
        "policy_version": POLICY_VERSION,
        "seed": 260730,
        "input": {
            "selected_path": str(selected_path),
            "selected_rows": len(selected_rows),
            "selected_sha256": selected_sha256,
            "qwen_shards": list(shard_records),
            "qwen_shard_receipts": list(receipt_records),
            "qwen_config_digest_by_shard": config_by_shard,
            "qwen_run_config_digest": common["run_config_digest"],
            "qwen_implementation_digest": common["implementation_digest"],
            "qwen_model_path": common["model_path"],
            "qwen_model_revision": common["model_revision"],
            "qwen_transformers_version": common[
                "transformers_version"
            ],
            "qwen_num_shards": 8,
            "selected_media_files_reverified": len(selected_rows),
            "selected_media_file_verification_sha256": media_result[
                "finalizer_file_verification_sha256"
            ],
        },
        "hard_gate": {
            "passed_rows": len(pass_iids),
            "rejected_rows": hard_rejected,
            "rejection_counts": hard_rejection_counts,
        },
        "diversity": {
            "group_id_max": 1,
            "target_verb_max": MAX_PER_TARGET_VERB,
            "review_rejection_counts": dict(
                sorted(diversity_rejections.items())
            ),
        },
        "selection": {
            "mode": "partial_up_to_128",
            "allow_partial": True,
            "requested_proposed_rows": PROPOSED_SIZE,
            "requested_reserve_rows": RESERVE_SIZE,
            "effective_proposed_target": proposed_target,
            "effective_reserve_target": reserve_target,
            "review_rows": len(expected_review),
            "proposed_rows": len(expected_proposed),
            "reserve_rows": len(expected_reserve),
            "generation_rows": len(expected_generation),
            "requested_category_quotas": dict(FAMILY_QUOTAS),
            "effective_category_quotas": _scaled_quotas(proposed_target),
            "proposed_category_counts": proposed_counts,
            "quota_shortfall_before_backfill": quota_shortfall,
            "review_category_counts": dict(
                sorted(review_categories.items())
            ),
            "reserve_category_counts": dict(
                sorted(reserve_categories.items())
            ),
            "proposed_target_verb_counts": dict(sorted(verb_counts.items())),
            "proposal_reserve_disjoint": True,
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
        "implementation_sha256": finalizer_sha256,
        "output_sha256": {
            name: hashlib.sha256(raw_by_name[name]).hexdigest()
            for name in sorted(JSONL_FINAL_NAMES)
        },
    }
    _assert_equal(summary, expected_summary, "summary exact reconstruction")

    done, done_raw, done_path = _load_json(
        root / "done.json", context="final done"
    )
    pretty_done = (
        json.dumps(
            done,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _assert_equal(done_raw, pretty_done, "done canonical pretty bytes")
    artifact_records["done.json"] = {
        "path": str(done_path),
        "sha256": hashlib.sha256(done_raw).hexdigest(),
        "bytes": len(done_raw),
    }
    expected_done = {
        "schema_version": DONE_SCHEMA,
        "status": "complete",
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "implementation_sha256": finalizer_sha256,
        "output_sha256": {
            name: hashlib.sha256(raw_by_name[name]).hexdigest()
            for name in sorted((*JSONL_FINAL_NAMES, "summary.json"))
        },
    }
    _assert_equal(done, expected_done, "done exact reconstruction")
    _assert_equal(
        completion["final_artifacts"],
        artifact_records,
        "completion final artifact bindings",
    )
    return {
        "path": str(root),
        "artifacts": artifact_records,
        "review_iids": [str(row["iid"]) for row in expected_review],
        "proposed_iids": [str(row["iid"]) for row in expected_proposed],
    }


def verify_acceptance(
    *,
    contract_path: Path,
    smoke_gold_path: Path,
    selected_path: Path,
    qwen_root: Path,
    final_dir: Path,
    source_snapshot: Path,
    submission_contract_path: Path,
    completion_receipt_path: Path,
) -> dict[str, Any]:
    """Verify the complete frozen smoke and return a deterministic report."""

    contract, contract_raw, contract_resolved = _load_json(
        contract_path, context="acceptance contract"
    )
    submission, submission_raw, submission_resolved = _load_json(
        submission_contract_path, context="submission contract"
    )
    completion, completion_raw, completion_resolved = _load_json(
        completion_receipt_path, context="completion receipt"
    )
    gold, gold_raw, gold_resolved, gold_labels = _load_smoke_gold(
        smoke_gold_path
    )
    _validate_acceptance_contract(contract)
    _validate_submission_contract(submission)
    _validate_completion_receipt(completion)
    _require_compact_json_bytes(
        contract, contract_raw, context="acceptance contract"
    )
    _require_compact_json_bytes(
        submission, submission_raw, context="submission contract"
    )
    _require_compact_json_bytes(
        completion, completion_raw, context="completion receipt"
    )
    contract_sha = hashlib.sha256(contract_raw).hexdigest()
    submission_sha = hashlib.sha256(submission_raw).hexdigest()
    completion_sha = hashlib.sha256(completion_raw).hexdigest()
    _assert_equal(
        contract["bindings"]["submission_contract_sha256"],
        submission_sha,
        "acceptance submission binding",
    )
    _assert_equal(
        contract["bindings"]["completion_receipt_sha256"],
        completion_sha,
        "acceptance completion binding",
    )
    _assert_equal(
        completion["submission_contract_path"],
        str(submission_resolved),
        "completion submission path",
    )
    _assert_equal(
        completion["submission_contract_sha256"],
        submission_sha,
        "completion submission SHA",
    )
    _assert_equal(
        contract["source_snapshot"],
        submission["source_snapshot"],
        "acceptance/submission source snapshot",
    )
    _assert_equal(
        contract["model"],
        submission["model"],
        "acceptance/submission model",
    )
    _assert_equal(
        contract["model_closure"],
        submission["model_closure"],
        "acceptance/submission model closure",
    )
    _assert_equal(
        contract["model_closure"],
        completion["model_closure"],
        "acceptance/completion model closure",
    )
    _assert_equal(
        contract["smoke_gold"],
        submission["smoke_gold"],
        "acceptance/submission smoke gold",
    )
    expected_gold_binding = {
        "path": str(gold_resolved),
        "sha256": hashlib.sha256(gold_raw).hexdigest(),
    }
    _assert_equal(
        contract["smoke_gold"],
        expected_gold_binding,
        "acceptance smoke gold binding",
    )
    _assert_equal(
        completion["smoke_gold_sha256"],
        expected_gold_binding["sha256"],
        "completion smoke gold binding",
    )
    _assert_equal(
        contract["execution"],
        {
            field: submission["runtime"][field]
            for field in QWEN_EXECUTION_KEYS
        },
        "acceptance/submission Qwen execution",
    )

    selected_rows, selected_raw, selected_resolved = _load_jsonl(
        selected_path, context="selected input"
    )
    selected_sha = hashlib.sha256(selected_raw).hexdigest()
    selected_iids = [str(row.get("iid", "")) for row in selected_rows]
    if len(selected_iids) != len(set(selected_iids)):
        raise AcceptanceError("selected input contains duplicate IIDs")
    ordered_iids_sha = _ordered_iids_digest(selected_iids)
    gold_selected_binding = _bind_gold_to_selected(
        gold,
        gold_labels,
        selected_rows=selected_rows,
        selected_raw=selected_raw,
    )
    for row in selected_rows:
        required = {
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
        }
        missing = sorted(required - set(row))
        if missing:
            raise AcceptanceError(
                f"selected iid={row.get('iid')} missing fields: {missing}"
            )
        _sha(row["anchor_sha256"], "selected anchor SHA")
        _sha(row["source_video_sha256"], "selected source SHA")
        score = row["prefilter_score"]
        if type(score) not in (int, float) or not math.isfinite(float(score)):
            raise AcceptanceError("selected prefilter_score is not finite")
    media_result = _verify_selected_media(selected_rows)
    expected_selected_binding = {
        "path": str(selected_resolved),
        "sha256": selected_sha,
        "rows": len(selected_rows),
    }
    _assert_equal(
        submission["selected"],
        expected_selected_binding,
        "submission selected binding",
    )
    _assert_equal(
        contract["selected"],
        {
            "rows": len(selected_rows),
            "sha256": selected_sha,
            "ordered_iids_sha256": ordered_iids_sha,
        },
        "acceptance selected binding",
    )
    _assert_equal(
        completion["selected_sha256"],
        selected_sha,
        "completion selected binding",
    )
    _assert_equal(
        contract["expected_shard_counts"],
        gold_selected_binding["expected_shard_rows"],
        "acceptance/gold shard counts",
    )

    snapshot_result = _verify_source_snapshot(
        contract["source_snapshot"], source_snapshot
    )
    model_result = _verify_model(contract["model"])
    model_closure_result = _verify_model_closure(
        contract["model_closure"],
        source_snapshot=source_snapshot,
        model_contract=contract["model"],
    )
    qwen_resolved = _directory(qwen_root, context="Qwen root argument")
    final_resolved = _directory(final_dir, context="final directory argument")
    _assert_equal(
        submission["outputs"]["qwen_root"],
        str(qwen_resolved),
        "submission qwen_root",
    )
    _assert_equal(
        submission["outputs"]["final_output"],
        str(final_resolved),
        "submission final_output",
    )
    _assert_equal(
        completion["qwen_root"], str(qwen_resolved), "completion qwen_root"
    )
    _assert_equal(
        completion["final_output"],
        str(final_resolved),
        "completion final_output",
    )

    qwen_source = Path(
        snapshot_result["implementations"]["qwen"]["path"]
    )
    constants = _source_constants(qwen_source)
    rows_by_iid, shard_records, receipt_records, common = (
        _verify_receipts_and_load_rows(
            selected_rows=selected_rows,
            selected_path=selected_resolved,
            selected_sha256=selected_sha,
            qwen_root=qwen_resolved,
            qwen_implementation_sha256=contract["source_snapshot"][
                "qwen_implementation_sha256"
            ],
            constants=constants,
            completion=completion,
            expected_shard_counts=contract["expected_shard_counts"],
        )
    )
    _assert_equal(
        common["model_path"],
        contract["model"]["path"],
        "Qwen receipt/model contract path",
    )
    observed_routes, gold_accuracy = _verify_qwen_rows(
        selected_rows=selected_rows,
        selected_path=selected_resolved,
        selected_sha256=selected_sha,
        rows_by_iid=rows_by_iid,
        constants=constants,
        common=common,
        gold_labels=gold_labels,
    )
    final_result = _verify_final_artifacts(
        final_dir=final_resolved,
        selected_rows=selected_rows,
        rows_by_iid=rows_by_iid,
        selected_path=selected_resolved,
        selected_sha256=selected_sha,
        shard_records=shard_records,
        receipt_records=receipt_records,
        common=common,
        finalizer_sha256=contract["source_snapshot"][
            "finalizer_implementation_sha256"
        ],
        completion=completion,
        observed_routes=observed_routes,
        media_result=media_result,
    )
    return {
        "schema_version": ACCEPTANCE_RESULT_SCHEMA,
        "contract": {
            "path": str(contract_resolved),
            "sha256": contract_sha,
            "bytes": len(contract_raw),
        },
        "submission_contract": {
            "path": str(submission_resolved),
            "sha256": submission_sha,
            "bytes": len(submission_raw),
        },
        "completion_receipt": {
            "path": str(completion_resolved),
            "sha256": completion_sha,
            "bytes": len(completion_raw),
            "job_id": completion["job_id"],
        },
        "smoke_gold": {
            "path": str(gold_resolved),
            "sha256": hashlib.sha256(gold_raw).hexdigest(),
            "bytes": len(gold_raw),
            "schema_version": gold["schema_version"],
            "gold_authority": gold["gold_authority"],
            "accuracy": gold_accuracy,
            "quarantine_stress_iids_not_in_gating_smoke": gold[
                "quarantine_stress_iids_not_in_gating_smoke"
            ],
        },
        "selected": {
            "path": str(selected_resolved),
            "sha256": selected_sha,
            "bytes": len(selected_raw),
            "rows": len(selected_rows),
            "ordered_iids_sha256": ordered_iids_sha,
            "media_binding": media_result,
        },
        "source_snapshot": snapshot_result,
        "model": model_result,
        "model_closure": model_closure_result,
        "qwen": {
            "root": str(qwen_resolved),
            "expected_shard_counts": list(
                contract["expected_shard_counts"]
            ),
            "shards": shard_records,
            "receipts": receipt_records,
            "run_identity": common,
            "routes": observed_routes,
            "route_counts": gold_accuracy["route_counts"],
        },
        "final": final_result,
        "failures": [],
        "passed": True,
        "full_123_authorized": True,
        "generation_authorized": False,
        "production_eligible": False,
        "wan_generation_authorized": False,
        "authorization_interface_available": False,
    }


def _atomic_write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            payload = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify the frozen Goku v16 smoke."
    )
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--smoke-gold", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--qwen-root", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    parser.add_argument("--source-snapshot", required=True, type=Path)
    parser.add_argument("--submission-contract", required=True, type=Path)
    parser.add_argument("--completion-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_acceptance(
            contract_path=args.contract,
            smoke_gold_path=args.smoke_gold,
            selected_path=args.selected,
            qwen_root=args.qwen_root,
            final_dir=args.final_dir,
            source_snapshot=args.source_snapshot,
            submission_contract_path=args.submission_contract,
            completion_receipt_path=args.completion_receipt,
        )
        status = 0
    except (UnauditableError, FileNotFoundError, OSError) as error:
        result = {
            "schema_version": ACCEPTANCE_RESULT_SCHEMA,
            "failures": [f"{type(error).__name__}: {error}"],
            "passed": False,
            "full_123_authorized": False,
            "generation_authorized": False,
            "production_eligible": False,
            "wan_generation_authorized": False,
            "authorization_interface_available": False,
        }
        status = 3
    except (AcceptanceError, KeyError, TypeError, ValueError) as error:
        result = {
            "schema_version": ACCEPTANCE_RESULT_SCHEMA,
            "failures": [f"{type(error).__name__}: {error}"],
            "passed": False,
            "full_123_authorized": False,
            "generation_authorized": False,
            "production_eligible": False,
            "wan_generation_authorized": False,
            "authorization_interface_available": False,
        }
        status = 2
    _atomic_write_new_json(args.output, result)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

"""Build fail-closed run-control artifacts for the frozen v16 smoke.

This module deliberately does *not* launch Slurm or run Qwen.  It creates the
immutable inputs and closed contracts around those actions:

``prepare``
    Atomically creates one fresh run root, validates the source-anchored visual
    audit gold file, byte-copies its selected smoke JSONL, validates the source
    snapshot/archive and Qwen model, and publishes ``subset_provenance.json``
    plus ``submission_contract.json``.

``complete``
    After a successful allocation, independently inventories exactly eight
    Qwen shards, eight terminal shard receipts, and the six finalizer
    artifacts.  It publishes a contract- and job-bound completion receipt.

``acceptance-contract``
    Builds the immutable input contract consumed by the independent v16
    acceptance verifier.  Selected identity, labels, and exact eight-shard
    geometry come only from the source-anchored visual-audit gold file.  Writer
    routes are deliberately not gold labels.

Every published JSON object uses a closed schema, canonical JSON encoding, and
no-overwrite atomic publication.  The auxiliary subset provenance and upstream
Qwen shard receipts carry self-digests; the three verifier-facing contracts
instead bind one another by exact file SHA-256, as required by their frozen
closed schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence


SUBMISSION_SCHEMA = "motive-goku-action-v16-submission-contract-v1"
SUBSET_SCHEMA = "goku-action-v16-smoke-subset-provenance-v1"
COMPLETION_SCHEMA = "motive-goku-action-v16-completion-receipt-v1"
ACCEPTANCE_CONTRACT_SCHEMA = "motive-goku-action-v16-acceptance-contract-v1"
SMOKE_GOLD_SCHEMA = "goku-action-v16-smoke-gold-v1"
TARGET_CONTRACT_SCHEMA = "goku-action-v16-target-contract-v1"
SOURCE_SNAPSHOT_SCHEMA = "motive-action-source-snapshot-v1"
SHARD_RECEIPT_SCHEMA = "goku-action-anchor-shard-receipt-v8"

SMOKE_GOLD_RELPATH = (
    "methods/motive/audits/goku_action_v16_smoke_gold.json"
)
MODEL_CLOSURE_SCHEMA = "motive-qwen-model-closure-v1"
MODEL_CLOSURE_RELPATH = (
    "methods/motive/audits/qwen3_vl_32b_instruct_model_closure.json"
)
FROZEN_SMOKE_GOLD_SHA256 = (
    "b99972b81139e7a3193e6589efdf8de38075102cf14f312e3e1e73dfc3d626df"
)
FROZEN_MODEL_CLOSURE_SHA256 = (
    "395236b156d85409ca40643683b47b1badb28602df0ef41e519e50f9a60f6c05"
)
FROZEN_PARENT_SELECTED_SHA256 = (
    "824e92112159d559691a039fd949b26e0ca9ff07efe98483814aba2386123a9d"
)
FROZEN_QWEN_IMPLEMENTATION_SHA256 = (
    "f5535e0f68e515609a1b578b494197ae0c45a5ca79030ba9ceaa25ba0d7b772e"
)
FROZEN_FINALIZER_IMPLEMENTATION_SHA256 = (
    "63d98952f400dd30a069fee72f169a2d512b8d3b0b9b7c4779475663e26758e3"
)
FROZEN_ACCEPTANCE_VERIFIER_SHA256 = (
    "ae5ff95e628e68b15d44704edb29fa9351118db23cbbc18872bb6006d8760070"
)
FROZEN_SBATCH_SHA256 = (
    "fff73cd87643c1b069a9ae3f118a678b410cf00007b2da4e3c26ef968cb7871d"
)
FROZEN_MODEL_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VLM/"
    "MEV-Annotation/checkpoints/Qwen3-VL-32B-Instruct"
)
FROZEN_MODEL_CONFIG_SHA256 = (
    "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427"
)
FROZEN_MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
FROZEN_MODEL_REVISION = "Qwen3-VL-32B-Instruct"

FINAL_ARTIFACT_NAMES = (
    "review_candidates.jsonl",
    "proposed_128.jsonl",
    "reserve_32.jsonl",
    "generation_manifest.jsonl",
    "summary.json",
    "done.json",
)
CONTRACT_SOURCE_RELATIVE_PATHS = {
    "qwen": "methods/motive/motive/goku_action_anchor_qwen.py",
    "finalizer": "methods/motive/motive/goku_action_anchor_finalize.py",
    "verifier": (
        "methods/motive/motive/goku_action_v13_acceptance.py"
    ),
    "sbatch": "methods/motive/scripts/auh_goku_action_anchor_qwen.sbatch",
}
RUN_ARTIFACT_BUILDER_RELPATH = (
    "methods/motive/motive/goku_action_v13_run_artifacts.py"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IID_RE = re.compile(r"[0-9a-f]{16}\Z")
_SNAKE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_SLURM_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")

_TARGET_ATOMIC_FIELDS = (
    "target_already_true",
    "target_start_state_visually_verifiable",
    "prerequisite_grounded",
    "novel_trajectory",
    "scalar_or_endpoint_only",
)
_YES_NO_UNCLEAR = {"yes", "no", "unclear"}
_TARGET_CHANGE_TYPES = {
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
_SOURCE_TARGET_RELATIONS = {
    "novel_future",
    "shared_base_with_novel_action",
    "later_source_phase_or_endpoint",
    "repeats_source_future",
    "same_action_scalar_only",
    "state_or_appearance_only",
    "unclear",
}
_SEMANTIC_CONTRACT_POLICY = {
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


class GokuActionV13RunArtifactError(ValueError):
    """A frozen input or terminal artifact violates the v16 contract."""


def _reject_constant(value: str) -> None:
    raise GokuActionV13RunArtifactError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GokuActionV13RunArtifactError(
                f"duplicate JSON object key: {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GokuActionV13RunArtifactError(
            f"value is not canonical JSON: {error}"
        ) from error


def _object_digest(value: Any) -> str:
    # Self-digests use canonical JSON without the file's terminal newline.
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GokuActionV13RunArtifactError(
            f"value is not digestible canonical JSON: {error}"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(
    raw: bytes,
    *,
    context: str,
    require_canonical: bool = False,
) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GokuActionV13RunArtifactError(
            f"{context} is not UTF-8"
        ) from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, GokuActionV13RunArtifactError):
            raise
        raise GokuActionV13RunArtifactError(
            f"{context} is not strict JSON: {error}"
        ) from error
    if require_canonical and raw != _canonical_bytes(value):
        raise GokuActionV13RunArtifactError(
            f"{context} is not canonical JSON"
        )
    return value


def _strict_json_file(
    path: Path,
    *,
    context: str,
    require_canonical: bool = False,
) -> tuple[Any, bytes, Path]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    return (
        _strict_json_bytes(
            raw,
            context=f"{context} {resolved}",
            require_canonical=require_canonical,
        ),
        raw,
        resolved,
    )


def _strict_jsonl_bytes(
    raw: bytes,
    *,
    context: str,
    allow_empty: bool = False,
    require_canonical_rows: bool = False,
) -> list[dict[str, Any]]:
    if raw and not raw.endswith(b"\n"):
        raise GokuActionV13RunArtifactError(
            f"{context} lacks a terminal newline"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise GokuActionV13RunArtifactError(
                f"{context} line {line_number} is blank"
            )
        value = _strict_json_bytes(
            line,
            context=f"{context} line {line_number}",
        )
        if not isinstance(value, dict):
            raise GokuActionV13RunArtifactError(
                f"{context} line {line_number} is not an object"
            )
        if (
            require_canonical_rows
            and line + b"\n" != _canonical_bytes(value)
        ):
            raise GokuActionV13RunArtifactError(
                f"{context} line {line_number} is not canonical"
            )
        rows.append(value)
    if not rows and not allow_empty:
        raise GokuActionV13RunArtifactError(f"{context} is empty")
    return rows


def _strict_jsonl_file(
    path: Path,
    *,
    context: str,
    allow_empty: bool = False,
    require_canonical_rows: bool = False,
) -> tuple[list[dict[str, Any]], bytes, Path]:
    resolved = _regular_file(path, context=context)
    raw = resolved.read_bytes()
    return (
        _strict_jsonl_bytes(
            raw,
            context=f"{context} {resolved}",
            allow_empty=allow_empty,
            require_canonical_rows=require_canonical_rows,
        ),
        raw,
        resolved,
    )


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GokuActionV13RunArtifactError(
            f"{context} must be an object"
        )
    return value


def _sequence(value: Any, *, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise GokuActionV13RunArtifactError(
            f"{context} must be an array"
        )
    return value


def _closed(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise GokuActionV13RunArtifactError(
            f"{context} is not closed: "
            f"missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise GokuActionV13RunArtifactError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _regular_file(path: Path, *, context: str) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise GokuActionV13RunArtifactError(
            f"{context} must not be a symlink: {unresolved}"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {context}: {unresolved}") from error
    if not resolved.is_file():
        raise GokuActionV13RunArtifactError(
            f"{context} is not a regular file: {resolved}"
        )
    return resolved


def _directory(
    path: Path,
    *,
    context: str,
    reject_symlink: bool = True,
) -> Path:
    unresolved = path.expanduser()
    if reject_symlink and unresolved.is_symlink():
        raise GokuActionV13RunArtifactError(
            f"{context} must not be a symlink: {unresolved}"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {context}: {unresolved}") from error
    if not resolved.is_dir():
        raise GokuActionV13RunArtifactError(
            f"{context} is not a directory: {resolved}"
        )
    return resolved


def _iid_shard(iid: str, *, num_shards: int = 8) -> int:
    if type(num_shards) is not int or num_shards < 1:
        raise GokuActionV13RunArtifactError(
            "num_shards must be a positive integer"
        )
    prefix = hashlib.sha256(iid.encode("utf-8")).hexdigest()[:16]
    return int(prefix, 16) % num_shards


def _iid_digest(iids: Iterable[str]) -> str:
    return hashlib.sha256(
        "".join(f"{iid}\n" for iid in iids).encode("utf-8")
    ).hexdigest()


def _self_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _object_digest(payload)


def _bind_self_digest(
    value: Mapping[str, Any],
    *,
    digest_field: str,
) -> dict[str, Any]:
    result = dict(value)
    if digest_field in result:
        raise GokuActionV13RunArtifactError(
            f"refusing pre-existing self-digest field {digest_field}"
        )
    result[digest_field] = _object_digest(result)
    return result


def _write_new_atomic(path: Path, payload: bytes) -> None:
    """Publish one new regular file without replacing any existing entry."""

    path = path.expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o400)
        # A hard link gives no-overwrite publication.  os.replace would be
        # vulnerable to a target appearing after the initial existence check.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_atomic(path, _canonical_bytes(value))


def _validate_target_contract(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    contract = _mapping(value, context=context)
    _closed(
        contract,
        {
            "schema_version",
            "instruction_sha256",
            "expected_target_change_class",
            "expected_source_target_relation",
            "expected_atomic_tuple",
            "target_token_groups",
        },
        context=context,
    )
    if contract["schema_version"] != TARGET_CONTRACT_SCHEMA:
        raise GokuActionV13RunArtifactError(
            f"{context} schema differs"
        )
    _sha256(
        contract["instruction_sha256"],
        context=f"{context} instruction SHA-256",
    )
    if contract["expected_target_change_class"] not in (
        _TARGET_CHANGE_TYPES
    ):
        raise GokuActionV13RunArtifactError(
            f"{context} target-change class is outside enum"
        )
    if contract["expected_source_target_relation"] not in (
        _SOURCE_TARGET_RELATIONS
    ):
        raise GokuActionV13RunArtifactError(
            f"{context} source-target relation is outside enum"
        )
    atomic = _mapping(
        contract["expected_atomic_tuple"],
        context=f"{context} expected atomic tuple",
    )
    _closed(
        atomic,
        set(_TARGET_ATOMIC_FIELDS),
        context=f"{context} expected atomic tuple",
    )
    if any(
        atomic[field] not in _YES_NO_UNCLEAR
        for field in _TARGET_ATOMIC_FIELDS
    ):
        raise GokuActionV13RunArtifactError(
            f"{context} expected atomic tuple is outside enum"
        )

    groups = _sequence(
        contract["target_token_groups"],
        context=f"{context} target token groups",
    )
    if not groups:
        raise GokuActionV13RunArtifactError(
            f"{context} target token groups are empty"
        )
    seen_group_ids: set[str] = set()
    for group_index, value_group in enumerate(groups):
        group_context = f"{context} target token group {group_index}"
        group = _mapping(value_group, context=group_context)
        _closed(
            group,
            {"group_id", "any_of"},
            context=group_context,
        )
        group_id = group["group_id"]
        if (
            not isinstance(group_id, str)
            or not _SNAKE_RE.fullmatch(group_id)
            or group_id in seen_group_ids
        ):
            raise GokuActionV13RunArtifactError(
                f"{group_context} group_id is invalid/duplicate"
            )
        seen_group_ids.add(group_id)
        alternatives = _sequence(
            group["any_of"],
            context=f"{group_context} alternatives",
        )
        if not alternatives:
            raise GokuActionV13RunArtifactError(
                f"{group_context} alternatives are empty"
            )
        seen_alternatives: set[tuple[str, ...]] = set()
        for alternative_index, value_alternative in enumerate(
            alternatives
        ):
            tokens_raw = _sequence(
                value_alternative,
                context=(
                    f"{group_context} alternative {alternative_index}"
                ),
            )
            if not tokens_raw:
                raise GokuActionV13RunArtifactError(
                    f"{group_context} has an empty alternative"
                )
            tokens: list[str] = []
            for token in tokens_raw:
                if (
                    not isinstance(token, str)
                    or not token
                    or token != token.casefold()
                    or any(character.isspace() for character in token)
                ):
                    raise GokuActionV13RunArtifactError(
                        f"{group_context} has a noncanonical token"
                    )
                tokens.append(token)
            frozen = tuple(tokens)
            if frozen in seen_alternatives:
                raise GokuActionV13RunArtifactError(
                    f"{group_context} has duplicate alternatives"
                )
            seen_alternatives.add(frozen)
    return dict(contract)


def _validate_smoke_gold(
    path: Path,
    *,
    source_snapshot: Path,
) -> dict[str, Any]:
    expected_path = source_snapshot / SMOKE_GOLD_RELPATH
    value, raw, resolved = _strict_json_file(
        path,
        context="v16 smoke gold",
    )
    if resolved != expected_path:
        raise GokuActionV13RunArtifactError(
            "smoke gold must use its canonical source-snapshot path"
        )
    if _sha256_bytes(raw) != FROZEN_SMOKE_GOLD_SHA256:
        raise GokuActionV13RunArtifactError(
            "smoke gold differs from its source-level trust anchor"
        )
    gold = _mapping(value, context="v16 smoke gold")
    _closed(
        gold,
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
        context="v16 smoke gold",
    )
    if gold["schema_version"] != SMOKE_GOLD_SCHEMA:
        raise GokuActionV13RunArtifactError(
            "smoke gold schema differs"
        )
    if gold["gold_authority"] != (
        "codex_visual_audit_not_generation_approval"
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold authority differs"
        )
    for field in ("review_method", "reviewed_at_utc"):
        if not isinstance(gold[field], str) or not gold[field]:
            raise GokuActionV13RunArtifactError(
                f"smoke gold {field} is invalid"
            )

    policy = _mapping(gold["policy"], context="smoke gold policy")
    _closed(
        policy,
        {
            "admissible",
            "inadmissible",
            "writer_route_is_not_a_gold_label",
            "positive_acceptance",
            "negative_acceptance",
            "wan_generation_authorized",
        },
        context="smoke gold policy",
    )
    if (
        policy["writer_route_is_not_a_gold_label"] is not True
        or policy["wan_generation_authorized"] is not False
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold policy is not route-neutral/review-only"
        )
    for field in (
        "admissible",
        "inadmissible",
        "positive_acceptance",
        "negative_acceptance",
    ):
        if not isinstance(policy[field], str) or not policy[field]:
            raise GokuActionV13RunArtifactError(
                f"smoke gold policy {field} is invalid"
            )

    semantic_policy = _mapping(
        gold["semantic_contract_policy"],
        context="smoke gold semantic contract policy",
    )
    _closed(
        semantic_policy,
        set(_SEMANTIC_CONTRACT_POLICY),
        context="smoke gold semantic contract policy",
    )
    if dict(semantic_policy) != _SEMANTIC_CONTRACT_POLICY:
        raise GokuActionV13RunArtifactError(
            "smoke gold semantic contract policy differs"
        )

    parent = _mapping(
        gold["parent_selected"],
        context="smoke gold parent_selected",
    )
    _closed(
        parent,
        {"path", "sha256", "rows", "bytes"},
        context="smoke gold parent_selected",
    )
    if (
        not isinstance(parent["path"], str)
        or not Path(parent["path"]).is_absolute()
        or parent["sha256"] != FROZEN_PARENT_SELECTED_SHA256
        or type(parent["rows"]) is not int
        or parent["rows"] < 1
        or type(parent["bytes"]) is not int
        or parent["bytes"] < 1
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold parent-selected binding differs"
        )

    selected = _mapping(
        gold["selected_smoke"],
        context="smoke gold selected_smoke",
    )
    _closed(
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
        context="smoke gold selected_smoke",
    )
    relative = selected["relative_path"]
    if not isinstance(relative, str) or not relative:
        raise GokuActionV13RunArtifactError(
            "smoke gold selected relative path is invalid"
        )
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold selected relative path is unsafe"
        )
    for field in (
        "sha256",
        "ordered_iids_sha256",
        "iid_set_sha256",
    ):
        _sha256(selected[field], context=f"smoke gold selected {field}")
    if (
        type(selected["rows"]) is not int
        or selected["rows"] < 1
        or type(selected["bytes"]) is not int
        or selected["bytes"] < 1
        or selected["num_shards"] != 8
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold selected geometry is invalid"
        )
    shard_rows = _sequence(
        selected["expected_shard_rows"],
        context="smoke gold expected_shard_rows",
    )
    if (
        len(shard_rows) != selected["num_shards"]
        or not all(type(item) is int and item >= 0 for item in shard_rows)
        or sum(shard_rows) != selected["rows"]
    ):
        raise GokuActionV13RunArtifactError(
            "smoke gold shard row geometry is invalid"
        )

    labels = _sequence(gold["labels"], context="smoke gold labels")
    if len(labels) != selected["rows"]:
        raise GokuActionV13RunArtifactError(
            "smoke gold label count differs from selected rows"
        )
    labels_by_iid: dict[str, dict[str, Any]] = {}
    for index, value_label in enumerate(labels):
        label = _mapping(
            value_label,
            context=f"smoke gold label {index}",
        )
        _closed(
            label,
            {
                "iid",
                "label",
                "target_contract",
                "reason_code",
                "visual_evidence",
            },
            context=f"smoke gold label {index}",
        )
        iid = label["iid"]
        if (
            not isinstance(iid, str)
            or not _IID_RE.fullmatch(iid)
            or iid in labels_by_iid
            or label["label"] not in {"admissible", "inadmissible"}
            or not isinstance(label["reason_code"], str)
            or not label["reason_code"]
            or not isinstance(label["visual_evidence"], str)
            or not label["visual_evidence"]
        ):
            raise GokuActionV13RunArtifactError(
                f"smoke gold label {index} is invalid"
            )
        validated_label = dict(label)
        validated_label["target_contract"] = _validate_target_contract(
            label["target_contract"],
            context=f"smoke gold label {index} target contract",
        )
        labels_by_iid[iid] = validated_label

    quarantine = _sequence(
        gold["quarantine_stress_iids_not_in_gating_smoke"],
        context="smoke gold quarantine",
    )
    quarantine_iids: set[str] = set()
    for index, value_item in enumerate(quarantine):
        item = _mapping(
            value_item,
            context=f"smoke gold quarantine {index}",
        )
        _closed(
            item,
            {"iid", "reason"},
            context=f"smoke gold quarantine {index}",
        )
        iid = item["iid"]
        if (
            not isinstance(iid, str)
            or not _IID_RE.fullmatch(iid)
            or iid in quarantine_iids
            or not isinstance(item["reason"], str)
            or not item["reason"]
        ):
            raise GokuActionV13RunArtifactError(
                f"smoke gold quarantine {index} is invalid"
            )
        quarantine_iids.add(iid)
    if quarantine_iids & set(labels_by_iid):
        raise GokuActionV13RunArtifactError(
            "smoke gold quarantine overlaps gating labels"
        )

    selected_path = _regular_file(
        source_snapshot / relative,
        context="source-anchored selected smoke",
    )
    return {
        "value": dict(gold),
        "path": str(resolved),
        "sha256": _sha256_bytes(raw),
        "selected": dict(selected),
        "selected_path": str(selected_path),
        "labels_by_iid": labels_by_iid,
    }


def _validated_selected(
    path: Path,
    *,
    smoke_gold: Mapping[str, Any],
    require_gold_path: bool = True,
) -> tuple[list[dict[str, Any]], bytes, Path]:
    rows, raw, resolved = _strict_jsonl_file(
        path,
        context="gold-bound selected smoke",
    )
    binding = _mapping(
        smoke_gold["selected"],
        context="validated smoke gold selected binding",
    )
    if (
        require_gold_path
        and resolved != Path(str(smoke_gold["selected_path"]))
    ):
        raise GokuActionV13RunArtifactError(
            "selected smoke path differs from smoke gold"
        )
    if len(raw) != binding["bytes"]:
        raise GokuActionV13RunArtifactError(
            "gold-bound selected byte count differs: "
            f"expected={binding['bytes']} actual={len(raw)}"
        )
    if _sha256_bytes(raw) != binding["sha256"]:
        raise GokuActionV13RunArtifactError(
            "gold-bound selected SHA-256 differs"
        )
    if len(rows) != binding["rows"]:
        raise GokuActionV13RunArtifactError(
            "gold-bound selected row count differs"
        )
    iids: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        iid = row.get("iid")
        if not isinstance(iid, str) or not _IID_RE.fullmatch(iid):
            raise GokuActionV13RunArtifactError(
                f"selected line {line_number} has invalid iid"
            )
        iids.append(iid)
    if len(set(iids)) != len(iids):
        raise GokuActionV13RunArtifactError(
            "gold-bound selected contains duplicate IIDs"
        )
    label_iids = [
        str(item["iid"])
        for item in _sequence(
            smoke_gold["value"]["labels"],
            context="validated smoke gold labels",
        )
    ]
    if iids != label_iids:
        raise GokuActionV13RunArtifactError(
            "gold labels do not match selected IID order"
        )
    labels_by_iid = _mapping(
        smoke_gold["labels_by_iid"],
        context="validated smoke gold labels",
    )
    for line_number, row in enumerate(rows, start=1):
        iid = str(row["iid"])
        instruction = row.get("prompt")
        if not isinstance(instruction, str) or not instruction:
            raise GokuActionV13RunArtifactError(
                f"selected line {line_number} has invalid edit instruction"
            )
        target_contract = _mapping(
            labels_by_iid[iid]["target_contract"],
            context=f"smoke gold iid={iid} target contract",
        )
        instruction_sha256 = _sha256_bytes(
            instruction.encode("utf-8")
        )
        if (
            target_contract["instruction_sha256"]
            != instruction_sha256
        ):
            raise GokuActionV13RunArtifactError(
                f"smoke gold iid={iid} instruction binding differs"
            )
    if _iid_digest(iids) != binding["ordered_iids_sha256"]:
        raise GokuActionV13RunArtifactError(
            "gold-bound selected ordered IID digest differs"
        )
    if _iid_digest(sorted(iids)) != binding["iid_set_sha256"]:
        raise GokuActionV13RunArtifactError(
            "gold-bound selected IID-set digest differs"
        )
    num_shards = int(binding["num_shards"])
    counts = tuple(
        sum(
            _iid_shard(iid, num_shards=num_shards) == index
            for iid in iids
        )
        for index in range(num_shards)
    )
    if counts != tuple(binding["expected_shard_rows"]):
        raise GokuActionV13RunArtifactError(
            "gold-bound selected shard geometry differs"
        )
    return rows, raw, resolved


def _validate_snapshot_manifest_row(
    value: Any,
    *,
    index: int,
) -> dict[str, Any]:
    row = _mapping(value, context=f"snapshot manifest row {index}")
    _closed(
        row,
        {"mode", "path", "sha256", "size", "type"},
        context=f"snapshot manifest row {index}",
    )
    relative = row["path"]
    if not isinstance(relative, str) or not relative:
        raise GokuActionV13RunArtifactError(
            f"snapshot manifest row {index} path is invalid"
        )
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != relative
    ):
        raise GokuActionV13RunArtifactError(
            f"snapshot manifest row {index} path is unsafe: {relative!r}"
        )
    if row["type"] != "file":
        raise GokuActionV13RunArtifactError(
            f"snapshot manifest row {index} type differs"
        )
    mode = row["mode"]
    if (
        not isinstance(mode, str)
        or re.fullmatch(r"0[0-7]{3}", mode) is None
        or int(mode, 8) & 0o222
    ):
        raise GokuActionV13RunArtifactError(
            f"snapshot manifest row {index} mode is not frozen"
        )
    _sha256(row["sha256"], context=f"snapshot manifest row {index} sha256")
    if type(row["size"]) is not int or row["size"] < 0:
        raise GokuActionV13RunArtifactError(
            f"snapshot manifest row {index} size is invalid"
        )
    return dict(row)


def _validate_source_snapshot(
    snapshot_path: Path,
    *,
    expected_tree_sha256: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_tree_sha256 = _sha256(
        expected_tree_sha256,
        context="expected source tree SHA-256",
    )
    expected_manifest_sha256 = _sha256(
        expected_manifest_sha256,
        context="expected source manifest SHA-256",
    )
    snapshot = _directory(snapshot_path, context="source snapshot")
    manifest = _regular_file(
        snapshot / "SOURCE_FILES.jsonl",
        context="source snapshot manifest",
    )
    manifest_raw = manifest.read_bytes()
    manifest_rows_raw = _strict_jsonl_bytes(
        manifest_raw,
        context=f"source snapshot manifest {manifest}",
        require_canonical_rows=True,
    )
    manifest_rows = [
        _validate_snapshot_manifest_row(value, index=index)
        for index, value in enumerate(manifest_rows_raw, start=1)
    ]
    paths = [str(row["path"]) for row in manifest_rows]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise GokuActionV13RunArtifactError(
            "source snapshot manifest paths are not unique sorted paths"
        )
    manifest_sha256 = _sha256_bytes(manifest_raw)
    tree_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise GokuActionV13RunArtifactError(
            "source snapshot manifest SHA-256 differs"
        )
    if tree_sha256 != expected_tree_sha256:
        raise GokuActionV13RunArtifactError(
            "source snapshot tree SHA-256 differs"
        )

    expected_files = {
        "SOURCE_FILES.jsonl",
        "SOURCE_PROVENANCE.json",
        *paths,
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        snapshot,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        directory_stat = directory_path.lstat()
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or stat.S_IMODE(directory_stat.st_mode) & 0o222
        ):
            raise GokuActionV13RunArtifactError(
                f"source snapshot directory is unsafe/writable: {directory_path}"
            )
        for name in sorted([*directory_names, *file_names]):
            entry = directory_path / name
            relative = entry.relative_to(snapshot).as_posix()
            entry_stat = entry.lstat()
            if stat.S_ISLNK(entry_stat.st_mode):
                raise GokuActionV13RunArtifactError(
                    f"source snapshot contains symlink: {relative}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                actual_directories.add(relative)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise GokuActionV13RunArtifactError(
                    f"source snapshot contains special file: {relative}"
                )
            if stat.S_IMODE(entry_stat.st_mode) & 0o222:
                raise GokuActionV13RunArtifactError(
                    f"source snapshot file is writable: {relative}"
                )
            actual_files.add(relative)
    if actual_files != expected_files:
        raise GokuActionV13RunArtifactError(
            "source snapshot file closure differs: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )

    expected_directories: set[str] = set()
    for relative in paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if actual_directories != expected_directories:
        raise GokuActionV13RunArtifactError(
            "source snapshot directory closure differs"
        )

    row_by_path = {str(row["path"]): row for row in manifest_rows}
    for relative, row in row_by_path.items():
        file_path = snapshot / relative
        file_stat = file_path.stat()
        if file_stat.st_size != row["size"]:
            raise GokuActionV13RunArtifactError(
                f"source snapshot size differs: {relative}"
            )
        if _sha256_file(file_path) != row["sha256"]:
            raise GokuActionV13RunArtifactError(
                f"source snapshot file SHA-256 differs: {relative}"
            )
        if f"{stat.S_IMODE(file_stat.st_mode):04o}" != row["mode"]:
            raise GokuActionV13RunArtifactError(
                f"source snapshot file mode differs: {relative}"
            )

    provenance_value, provenance_raw, provenance = _strict_json_file(
        snapshot / "SOURCE_PROVENANCE.json",
        context="source snapshot provenance",
    )
    provenance_object = _mapping(
        provenance_value,
        context="source snapshot provenance",
    )
    _closed(
        provenance_object,
        {
            "schema",
            "created_at_utc",
            "repo_root",
            "source_roots",
            "source_file_count",
            "source_tree_sha256",
            "source_manifest_sha256",
            "git_base_commit",
            "git_status_short",
        },
        context="source snapshot provenance",
    )
    if provenance_object["schema"] != SOURCE_SNAPSHOT_SCHEMA:
        raise GokuActionV13RunArtifactError(
            "source snapshot provenance schema differs"
        )
    provenance_bindings = {
        "source_file_count": len(manifest_rows),
        "source_tree_sha256": tree_sha256,
        "source_manifest_sha256": manifest_sha256,
    }
    for field, expected in provenance_bindings.items():
        if provenance_object.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"source snapshot provenance {field} differs"
            )

    implementations: dict[str, dict[str, Any]] = {}
    for role, relative in CONTRACT_SOURCE_RELATIVE_PATHS.items():
        if relative not in row_by_path:
            raise GokuActionV13RunArtifactError(
                f"source snapshot omits required {role}: {relative}"
            )
        row = row_by_path[relative]
        implementations[role] = {
            "relative_path": relative,
            "sha256": row["sha256"],
            "bytes": row["size"],
        }
    frozen_implementation_sha256 = {
        "qwen": FROZEN_QWEN_IMPLEMENTATION_SHA256,
        "finalizer": FROZEN_FINALIZER_IMPLEMENTATION_SHA256,
        "verifier": FROZEN_ACCEPTANCE_VERIFIER_SHA256,
        "sbatch": FROZEN_SBATCH_SHA256,
    }
    for role, expected in frozen_implementation_sha256.items():
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(
            expected
        ):
            raise GokuActionV13RunArtifactError(
                f"source-level {role} trust anchor is not frozen"
            )
        if implementations[role]["sha256"] != expected:
            raise GokuActionV13RunArtifactError(
                f"source snapshot {role} differs from frozen production"
            )
    if RUN_ARTIFACT_BUILDER_RELPATH not in row_by_path:
        raise GokuActionV13RunArtifactError(
            "source snapshot omits the executing run-artifact builder"
        )
    builder_row = row_by_path[RUN_ARTIFACT_BUILDER_RELPATH]
    return {
        "path": str(snapshot),
        "tree_sha256": tree_sha256,
        "manifest": {
            "path": str(manifest),
            "sha256": manifest_sha256,
            "bytes": len(manifest_raw),
            "rows": len(manifest_rows),
        },
        "provenance": {
            "path": str(provenance),
            "sha256": _sha256_bytes(provenance_raw),
            "bytes": len(provenance_raw),
        },
        "implementations": implementations,
        "_builder": {
            "relative_path": RUN_ARTIFACT_BUILDER_RELPATH,
            "sha256": builder_row["sha256"],
            "bytes": builder_row["size"],
        },
        "_manifest_rows": manifest_rows,
        "_provenance_raw": provenance_raw,
    }


def _safe_tar_relative(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise GokuActionV13RunArtifactError(
            f"source archive member name is invalid: {name!r}"
        )
    normalized = name[:-1] if name.endswith("/") else name
    pure = PurePosixPath(normalized)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or pure.as_posix() != normalized
        or not pure.parts
    ):
        raise GokuActionV13RunArtifactError(
            f"source archive member path is unsafe: {name!r}"
        )
    return pure


def _validate_source_archive(
    archive_path: Path,
    *,
    expected_archive_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    expected_archive_sha256 = _sha256(
        expected_archive_sha256,
        context="expected source archive SHA-256",
    )
    archive = _regular_file(archive_path, context="source snapshot archive")
    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != expected_archive_sha256:
        raise GokuActionV13RunArtifactError(
            "source snapshot archive SHA-256 differs"
        )
    manifest_rows = _sequence(
        snapshot["_manifest_rows"],
        context="validated source manifest rows",
    )
    expected_relative_files = {
        "SOURCE_FILES.jsonl",
        "SOURCE_PROVENANCE.json",
        *(str(row["path"]) for row in manifest_rows),
    }
    seen_files: set[str] = set()
    seen_members: set[str] = set()
    top_levels: set[str] = set()
    archive_directories: dict[str, int] = {}
    file_digests: dict[str, tuple[int, str, int]] = {}
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in handle:
                pure = _safe_tar_relative(member.name)
                normalized_name = pure.as_posix()
                if normalized_name in seen_members:
                    raise GokuActionV13RunArtifactError(
                        f"source archive duplicate member: {member.name}"
                    )
                seen_members.add(normalized_name)
                top_levels.add(pure.parts[0])
                if member.issym() or member.islnk():
                    raise GokuActionV13RunArtifactError(
                        f"source archive contains a link: {member.name}"
                    )
                if member.isdir():
                    if member.mode & 0o222:
                        raise GokuActionV13RunArtifactError(
                            f"source archive directory is writable: {member.name}"
                        )
                    if len(pure.parts) > 1:
                        relative_directory = PurePosixPath(
                            *pure.parts[1:]
                        ).as_posix()
                        archive_directories[relative_directory] = (
                            member.mode & 0o7777
                        )
                    continue
                if not member.isfile():
                    raise GokuActionV13RunArtifactError(
                        f"source archive contains special member: {member.name}"
                    )
                if member.mode & 0o222:
                    raise GokuActionV13RunArtifactError(
                        f"source archive file is writable: {member.name}"
                    )
                if len(pure.parts) < 2:
                    raise GokuActionV13RunArtifactError(
                        "source archive has a regular file outside its root"
                    )
                relative = PurePosixPath(*pure.parts[1:]).as_posix()
                if relative in seen_files:
                    raise GokuActionV13RunArtifactError(
                        f"source archive duplicate relative file: {relative}"
                    )
                seen_files.add(relative)
                extracted = handle.extractfile(member)
                if extracted is None:
                    raise GokuActionV13RunArtifactError(
                        f"cannot stream source archive file: {member.name}"
                    )
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                file_digests[relative] = (
                    size,
                    digest.hexdigest(),
                    member.mode & 0o7777,
                )
    except (tarfile.TarError, OSError) as error:
        if isinstance(error, GokuActionV13RunArtifactError):
            raise
        raise GokuActionV13RunArtifactError(
            f"cannot validate source snapshot archive: {error}"
        ) from error
    if len(top_levels) != 1:
        raise GokuActionV13RunArtifactError(
            f"source archive must have one top-level root: {sorted(top_levels)}"
        )
    if seen_files != expected_relative_files:
        raise GokuActionV13RunArtifactError(
            "source archive file closure differs from snapshot: "
            f"missing={sorted(expected_relative_files - seen_files)} "
            f"extra={sorted(seen_files - expected_relative_files)}"
        )
    snapshot_path = Path(str(snapshot["path"]))
    expected_directories = {
        path.relative_to(snapshot_path).as_posix(): stat.S_IMODE(
            path.stat().st_mode
        )
        for path in snapshot_path.rglob("*")
        if path.is_dir()
    }
    if archive_directories != expected_directories:
        raise GokuActionV13RunArtifactError(
            "source archive directory closure/modes differ from snapshot"
        )
    for relative, (size, digest, mode) in file_digests.items():
        actual = snapshot_path / relative
        if (
            actual.stat().st_size != size
            or _sha256_file(actual) != digest
            or stat.S_IMODE(actual.stat().st_mode) != mode
        ):
            raise GokuActionV13RunArtifactError(
                f"source archive differs from extracted snapshot: {relative}"
            )
    return {
        "path": str(archive),
        "sha256": archive_sha256,
        "bytes": archive.stat().st_size,
        "top_level_root": next(iter(top_levels)),
        "regular_files": len(seen_files),
    }


def _validate_model_closure(
    closure_path: Path,
    *,
    source_snapshot: Path,
    model_path: Path,
) -> dict[str, Any]:
    expected_path = source_snapshot / MODEL_CLOSURE_RELPATH
    value, raw, resolved = _strict_json_file(
        closure_path,
        context="v16 Qwen model closure",
    )
    if resolved != expected_path:
        raise GokuActionV13RunArtifactError(
            "model closure must use its canonical source-snapshot path"
        )
    digest = _sha256_bytes(raw)
    if digest != FROZEN_MODEL_CLOSURE_SHA256:
        raise GokuActionV13RunArtifactError(
            "model closure differs from its source-level trust anchor"
        )
    closure = _mapping(value, context="v16 Qwen model closure")
    _closed(
        closure,
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
        context="v16 Qwen model closure",
    )
    model = _directory(model_path, context="Qwen model closure root")
    if (
        closure["schema_version"] != MODEL_CLOSURE_SCHEMA
        or closure["model_id"] != FROZEN_MODEL_ID
        or closure["revision"] != FROZEN_MODEL_REVISION
        or model.name != FROZEN_MODEL_REVISION
        or closure["model_path"] != str(model)
        or closure["hash_algorithm"] != "sha256"
        or type(closure["file_count"]) is not int
        or closure["file_count"] < 1
        or type(closure["total_bytes"]) is not int
        or closure["total_bytes"] < 1
    ):
        raise GokuActionV13RunArtifactError(
            "model closure identity/header differs"
        )

    values = _sequence(
        closure["files"],
        context="v16 Qwen model closure files",
    )
    if len(values) != closure["file_count"]:
        raise GokuActionV13RunArtifactError(
            "model closure file count differs"
        )
    manifest_files: list[dict[str, Any]] = []
    for index, value_file in enumerate(values):
        item = _mapping(
            value_file,
            context=f"model closure file {index}",
        )
        _closed(
            item,
            {"relative_path", "bytes", "sha256"},
            context=f"model closure file {index}",
        )
        relative = item["relative_path"]
        if not isinstance(relative, str) or not relative:
            raise GokuActionV13RunArtifactError(
                f"model closure file {index} path is invalid"
            )
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or pure.as_posix() != relative
        ):
            raise GokuActionV13RunArtifactError(
                f"model closure file {index} path is unsafe"
            )
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise GokuActionV13RunArtifactError(
                f"model closure file {index} byte count is invalid"
            )
        _sha256(
            item["sha256"],
            context=f"model closure file {index} SHA-256",
        )
        manifest_files.append(dict(item))
    relative_paths = [
        str(item["relative_path"]) for item in manifest_files
    ]
    if (
        relative_paths != sorted(relative_paths)
        or len(relative_paths) != len(set(relative_paths))
        or sum(int(item["bytes"]) for item in manifest_files)
        != closure["total_bytes"]
    ):
        raise GokuActionV13RunArtifactError(
            "model closure paths/order/total bytes differ"
        )

    expected_names = set(relative_paths)
    actual_names: set[str] = set()
    actual_directories: set[str] = set()
    for directory, directory_names, file_names in os.walk(
        model,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in sorted(directory_names):
            entry = directory_path / name
            relative = entry.relative_to(model).as_posix()
            if entry.is_symlink():
                raise GokuActionV13RunArtifactError(
                    "model closure contains a directory symlink: "
                    f"{relative}"
                )
            try:
                mode = entry.lstat().st_mode
            except OSError as error:
                raise GokuActionV13RunArtifactError(
                    f"cannot inspect model directory {relative}: {error}"
                ) from error
            if not stat.S_ISDIR(mode):
                raise GokuActionV13RunArtifactError(
                    f"model closure contains special entry: {relative}"
                )
            actual_directories.add(relative)
        for name in sorted(file_names):
            entry = directory_path / name
            relative = entry.relative_to(model).as_posix()
            try:
                entry_mode = entry.lstat().st_mode
                resolved_entry = entry.resolve(strict=True)
            except (FileNotFoundError, OSError) as error:
                raise GokuActionV13RunArtifactError(
                    f"cannot resolve model file {relative}: {error}"
                ) from error
            if not (
                stat.S_ISREG(entry_mode)
                or stat.S_ISLNK(entry_mode)
            ) or not resolved_entry.is_file():
                raise GokuActionV13RunArtifactError(
                    f"model closure contains non-regular file: {relative}"
                )
            actual_names.add(relative)

    expected_directories: set[str] = set()
    for relative in relative_paths:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if (
        actual_names != expected_names
        or actual_directories != expected_directories
    ):
        raise GokuActionV13RunArtifactError(
            "model file closure differs: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)} "
            f"directory_diff={sorted(actual_directories ^ expected_directories)}"
        )

    files_by_relative: dict[str, dict[str, Any]] = {}
    for item in manifest_files:
        relative = str(item["relative_path"])
        logical = model / relative
        resolved_file = logical.resolve(strict=True)
        actual_bytes = resolved_file.stat().st_size
        actual_sha256 = _sha256_file(resolved_file)
        if (
            actual_bytes != item["bytes"]
            or actual_sha256 != item["sha256"]
        ):
            raise GokuActionV13RunArtifactError(
                f"model closure file binding differs: {relative}"
            )
        files_by_relative[relative] = {
            **item,
            "logical_path": str(logical),
            "resolved_path": str(resolved_file),
        }
    return {
        "path": str(resolved),
        "sha256": digest,
        "file_count": closure["file_count"],
        "total_bytes": closure["total_bytes"],
        "_model_path": str(model),
        "_files_by_relative": files_by_relative,
    }


def _public_model_closure_binding(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": validated["path"],
        "sha256": validated["sha256"],
        "file_count": validated["file_count"],
        "total_bytes": validated["total_bytes"],
    }


def _validate_model_closure_binding(
    value: Any,
    *,
    source_snapshot: Path,
    model_path: Path,
) -> dict[str, Any]:
    binding = _mapping(value, context="submission model_closure")
    _closed(
        binding,
        {"path", "sha256", "file_count", "total_bytes"},
        context="submission model_closure",
    )
    expected_path = source_snapshot / MODEL_CLOSURE_RELPATH
    manifest_value, raw, resolved = _strict_json_file(
        Path(str(binding["path"])),
        context="submission model closure manifest",
    )
    if (
        resolved != expected_path
        or binding["sha256"] != FROZEN_MODEL_CLOSURE_SHA256
        or _sha256_bytes(raw) != FROZEN_MODEL_CLOSURE_SHA256
    ):
        raise GokuActionV13RunArtifactError(
            "submission model-closure trust binding differs"
        )
    manifest = _mapping(
        manifest_value,
        context="submission model closure manifest",
    )
    _closed(
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
        context="submission model closure manifest",
    )
    if (
        manifest["schema_version"] != MODEL_CLOSURE_SCHEMA
        or manifest["model_id"] != FROZEN_MODEL_ID
        or manifest["revision"] != FROZEN_MODEL_REVISION
        or model_path.name != FROZEN_MODEL_REVISION
        or manifest["model_path"] != str(model_path)
        or manifest["hash_algorithm"] != "sha256"
        or binding["file_count"] != manifest["file_count"]
        or binding["total_bytes"] != manifest["total_bytes"]
    ):
        raise GokuActionV13RunArtifactError(
            "submission model-closure public binding differs"
        )
    return dict(binding)


def _validate_model(
    model_path: Path,
    *,
    expected_config_sha256: str,
    model_closure: Mapping[str, Any],
) -> dict[str, Any]:
    expected_config_sha256 = _sha256(
        expected_config_sha256,
        context="expected model config SHA-256",
    )
    model = _directory(model_path, context="Qwen model")
    if str(model) != FROZEN_MODEL_PATH:
        raise GokuActionV13RunArtifactError(
            "Qwen model path differs from frozen production"
        )
    if expected_config_sha256 != FROZEN_MODEL_CONFIG_SHA256:
        raise GokuActionV13RunArtifactError(
            "expected Qwen model config SHA-256 differs from frozen production"
        )
    if model_closure["_model_path"] != str(model):
        raise GokuActionV13RunArtifactError(
            "validated model closure root differs from Qwen model"
        )
    closure_files = _mapping(
        model_closure["_files_by_relative"],
        context="validated model closure files",
    )
    config_logical = model / "config.json"
    if not config_logical.exists():
        raise FileNotFoundError(config_logical)
    config = config_logical.resolve(strict=True)
    if not config.is_file():
        raise GokuActionV13RunArtifactError(
            "Qwen model config is not a regular file"
        )
    config_binding = _mapping(
        closure_files.get("config.json"),
        context="model closure config binding",
    )
    config_sha256 = str(config_binding["sha256"])
    if config_sha256 != expected_config_sha256:
        raise GokuActionV13RunArtifactError(
            "Qwen model config SHA-256 differs"
        )
    config_value = _strict_json_bytes(
        config.read_bytes(),
        context=f"Qwen model config {config}",
    )
    _mapping(config_value, context="Qwen model config")

    index_logical = model / "model.safetensors.index.json"
    index = index_logical.resolve(strict=True)
    if not index.is_file():
        raise GokuActionV13RunArtifactError(
            "Qwen model safetensors index is not a regular file"
        )
    index_value = _strict_json_bytes(
        index.read_bytes(),
        context=f"Qwen model safetensors index {index}",
    )
    index_object = _mapping(
        index_value,
        context="Qwen model safetensors index",
    )
    weight_map = _mapping(
        index_object.get("weight_map"),
        context="Qwen model safetensors weight_map",
    )
    if not weight_map:
        raise GokuActionV13RunArtifactError(
            "Qwen model safetensors weight_map is empty"
        )
    logical_names: list[str] = []
    for tensor_name, file_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise GokuActionV13RunArtifactError(
                "Qwen model weight_map tensor name is invalid"
            )
        if not isinstance(file_name, str):
            raise GokuActionV13RunArtifactError(
                "Qwen model weight_map filename is invalid"
            )
        pure = PurePosixPath(file_name)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != file_name
            or pure.suffix != ".safetensors"
        ):
            raise GokuActionV13RunArtifactError(
                f"Qwen model weight filename is unsafe: {file_name!r}"
            )
        logical_names.append(file_name)
    weight_files: list[dict[str, Any]] = []
    for file_name in sorted(set(logical_names)):
        logical = model / file_name
        resolved = logical.resolve(strict=True)
        binding = _mapping(
            closure_files.get(file_name),
            context=f"model closure weight {file_name}",
        )
        if (
            not resolved.is_file()
            or resolved.stat().st_size <= 0
            or binding["resolved_path"] != str(resolved)
        ):
            raise GokuActionV13RunArtifactError(
                f"Qwen model weight is missing/empty: {file_name}"
            )
        weight_files.append(
            {
                "name": file_name,
                "resolved_path": str(resolved),
                "bytes": binding["bytes"],
                "sha256": binding["sha256"],
            }
        )
    return {
        "path": str(model),
        "revision": model.name,
        "config": {
            "logical_path": str(config_logical),
            "resolved_path": str(config),
            "sha256": config_sha256,
            "bytes": config.stat().st_size,
        },
        "weight_index": {
            "logical_path": str(index_logical),
            "resolved_path": str(index),
            "sha256": closure_files[
                "model.safetensors.index.json"
            ]["sha256"],
            "bytes": closure_files[
                "model.safetensors.index.json"
            ]["bytes"],
        },
        "weight_files": weight_files,
        "weight_file_count": len(weight_files),
        "weight_total_bytes": sum(
            int(item["bytes"]) for item in weight_files
        ),
    }


def _public_snapshot_binding(
    snapshot: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    implementations = snapshot["implementations"]
    return {
        "path": snapshot["path"],
        "tree_sha256": snapshot["tree_sha256"],
        "manifest_path": snapshot["manifest"]["path"],
        "manifest_sha256": snapshot["manifest"]["sha256"],
        "archive_path": archive["path"],
        "archive_sha256": archive["sha256"],
        "qwen_relpath": implementations["qwen"]["relative_path"],
        "qwen_implementation_sha256": implementations["qwen"]["sha256"],
        "finalizer_relpath": implementations["finalizer"]["relative_path"],
        "finalizer_implementation_sha256": implementations["finalizer"][
            "sha256"
        ],
        "verifier_relpath": implementations["verifier"]["relative_path"],
        "verifier_implementation_sha256": implementations["verifier"][
            "sha256"
        ],
        "sbatch_relpath": implementations["sbatch"]["relative_path"],
        "sbatch_sha256": implementations["sbatch"]["sha256"],
    }


def _build_subset_provenance(
    *,
    source_selected: Path,
    copied_selected: Path,
    selected_raw: bytes,
    selected_rows: Sequence[Mapping[str, Any]],
    smoke_gold: Mapping[str, Any],
) -> dict[str, Any]:
    selected_binding = _mapping(
        smoke_gold["selected"],
        context="smoke gold selected binding",
    )
    num_shards = int(selected_binding["num_shards"])
    row_bindings: list[dict[str, Any]] = []
    for index, (raw_line, row) in enumerate(
        zip(selected_raw.splitlines(), selected_rows),
        start=1,
    ):
        iid = str(row["iid"])
        row_bindings.append(
            {
                "line_number": index,
                "iid": iid,
                "raw_line_sha256": _sha256_bytes(raw_line),
                "canonical_row_sha256": _object_digest(row),
                "assigned_shard": _iid_shard(
                    iid,
                    num_shards=num_shards,
                ),
            }
        )
    provenance: dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA,
        "selection_rule": (
            "exact byte copy of the selected_smoke JSONL bound by the "
            "source-anchored v16 semantic visual-audit gold; no selection, rewrite, "
            "normalization, or reordering"
        ),
        "source": {
            "path": str(source_selected),
            "sha256": _sha256_bytes(selected_raw),
            "bytes": len(selected_raw),
            "rows": len(selected_rows),
        },
        "subset": {
            "path": str(copied_selected),
            "sha256": _sha256_bytes(selected_raw),
            "bytes": len(selected_raw),
            "rows": len(selected_rows),
        },
        "parent_selected_sha256": smoke_gold["value"][
            "parent_selected"
        ]["sha256"],
        "ordered_iids": [str(row["iid"]) for row in selected_rows],
        "ordered_iids_sha256": selected_binding[
            "ordered_iids_sha256"
        ],
        "iid_set_sha256": selected_binding["iid_set_sha256"],
        "qwen_shard_row_counts": list(
            selected_binding["expected_shard_rows"]
        ),
        "row_bindings": row_bindings,
    }
    return _bind_self_digest(
        provenance,
        digest_field="provenance_digest",
    )


def prepare_run(
    *,
    run_root: str | Path,
    frozen_selected: str | Path,
    smoke_gold: str | Path,
    model_closure: str | Path,
    source_snapshot: str | Path,
    source_archive: str | Path,
    source_tree_sha256: str,
    source_manifest_sha256: str,
    source_archive_sha256: str,
    model_path: str | Path,
    model_config_sha256: str,
) -> dict[str, Any]:
    """Atomically create a fresh v16 smoke run root and submission contract."""

    root_candidate = Path(run_root).expanduser()
    if not root_candidate.is_absolute():
        raise GokuActionV13RunArtifactError(
            "run_root must be an absolute path"
        )
    root = root_candidate.resolve(strict=False)
    if root == Path(root.anchor):
        raise GokuActionV13RunArtifactError(
            "run_root must not be a filesystem root"
        )
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    parent = _directory(root.parent, context="run root parent")
    root = parent / root.name
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)

    snapshot = _validate_source_snapshot(
        Path(source_snapshot),
        expected_tree_sha256=source_tree_sha256,
        expected_manifest_sha256=source_manifest_sha256,
    )
    gold = _validate_smoke_gold(
        Path(smoke_gold),
        source_snapshot=Path(str(snapshot["path"])),
    )
    selected_rows, selected_raw, selected_source = _validated_selected(
        Path(frozen_selected),
        smoke_gold=gold,
    )
    archive = _validate_source_archive(
        Path(source_archive),
        expected_archive_sha256=source_archive_sha256,
        snapshot=snapshot,
    )
    validated_model_closure = _validate_model_closure(
        Path(model_closure),
        source_snapshot=Path(str(snapshot["path"])),
        model_path=Path(model_path),
    )
    model = _validate_model(
        Path(model_path),
        expected_config_sha256=model_config_sha256,
        model_closure=validated_model_closure,
    )

    executing_builder_sha256 = _sha256_file(
        Path(__file__).resolve(strict=True)
    )
    snapshotted_builder_sha256 = str(snapshot["_builder"]["sha256"])
    if executing_builder_sha256 != snapshotted_builder_sha256:
        raise GokuActionV13RunArtifactError(
            "executing run-artifact builder differs from source snapshot"
        )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.",
            suffix=".prepare.tmp",
            dir=parent,
        )
    )
    try:
        input_dir = staging / "input"
        logs_dir = staging / "logs"
        input_dir.mkdir(mode=0o700)
        logs_dir.mkdir(mode=0o700)
        selected_staging = input_dir / "selected_smoke.jsonl"
        selected_staging.write_bytes(selected_raw)
        if selected_staging.read_bytes() != selected_raw:
            raise GokuActionV13RunArtifactError(
                "staged selected bytes differ after copy"
            )

        selected_final = root / "input" / "selected_smoke.jsonl"
        qwen_final = root / "qwen8"
        finalizer_final = root / "final"
        forbidden = (
            staging / "qwen8",
            staging / "final",
            staging / "jobs.tsv",
            staging / "completion_receipt.json",
            staging / "acceptance_contract.json",
            staging / "acceptance_result.json",
        )
        if any(path.exists() or path.is_symlink() for path in forbidden):
            raise GokuActionV13RunArtifactError(
                "fresh staging unexpectedly contains terminal artifacts"
            )

        subset = _build_subset_provenance(
            source_selected=selected_source,
            copied_selected=selected_final,
            selected_raw=selected_raw,
            selected_rows=selected_rows,
            smoke_gold=gold,
        )
        subset_payload = _canonical_bytes(subset)
        (input_dir / "subset_provenance.json").write_bytes(subset_payload)

        source_binding = _public_snapshot_binding(snapshot, archive)
        model_binding = {
            "path": model["path"],
            "config_path": model["config"]["logical_path"],
            "config_sha256": model["config"]["sha256"],
        }
        submission: dict[str, Any] = {
            "schema_version": SUBMISSION_SCHEMA,
            "selected": {
                "path": str(selected_final),
                "sha256": gold["selected"]["sha256"],
                "rows": gold["selected"]["rows"],
            },
            "smoke_gold": {
                "path": gold["path"],
                "sha256": gold["sha256"],
            },
            "model_closure": _public_model_closure_binding(
                validated_model_closure
            ),
            "source_snapshot": source_binding,
            "model": model_binding,
            "runtime": {
                "num_shards": 8,
                "max_samples": None,
                "max_new_tokens": 1_536,
                "nframes": 12,
                "max_pixels": 589_824,
                "attn_implementation": "sdpa",
                "allow_download": False,
                "repair_attempts": 1,
                "final_seed": 260_730,
                "allow_partial": True,
            },
            "outputs": {
                "qwen_root": str(qwen_final),
                "final_output": str(finalizer_final),
            },
        }
        submission_payload = _canonical_bytes(submission)
        (staging / "submission_contract.json").write_bytes(
            submission_payload
        )

        # Ensure the entire staged publication has only the intended closure.
        actual = {
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
        }
        expected = {
            "input",
            "input/selected_smoke.jsonl",
            "input/subset_provenance.json",
            "logs",
            "submission_contract.json",
        }
        if actual != expected:
            raise GokuActionV13RunArtifactError(
                f"prepared run closure differs: {sorted(actual ^ expected)}"
            )
        for path in (
            selected_staging,
            input_dir / "subset_provenance.json",
            staging / "submission_contract.json",
        ):
            path.chmod(0o400)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        input_dir.chmod(0o500)
        for directory in (input_dir, logs_dir, staging):
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        os.rename(staging, root)
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Return the exact published object after a byte-level self-check.
    published_value, published_raw, _ = _strict_json_file(
        root / "submission_contract.json",
        context="published submission contract",
        require_canonical=True,
    )
    if published_raw != submission_payload or published_value != submission:
        raise GokuActionV13RunArtifactError(
            "published submission contract differs from staged bytes"
        )
    return dict(submission)


def _validate_subset_provenance(
    path: Path,
    *,
    expected_selected_path: str,
    smoke_gold: Mapping[str, Any],
) -> dict[str, Any]:
    value, _, resolved = _strict_json_file(
        path,
        context="subset provenance",
        require_canonical=True,
    )
    provenance = _mapping(value, context="subset provenance")
    _closed(
        provenance,
        {
            "schema_version",
            "selection_rule",
            "source",
            "subset",
            "parent_selected_sha256",
            "ordered_iids",
            "ordered_iids_sha256",
            "iid_set_sha256",
            "qwen_shard_row_counts",
            "row_bindings",
            "provenance_digest",
        },
        context="subset provenance",
    )
    if provenance["schema_version"] != SUBSET_SCHEMA:
        raise GokuActionV13RunArtifactError(
            "subset provenance schema differs"
        )
    if provenance["provenance_digest"] != _self_digest(
        provenance,
        digest_field="provenance_digest",
    ):
        raise GokuActionV13RunArtifactError(
            "subset provenance digest differs"
        )
    selected_binding = _mapping(
        smoke_gold["selected"],
        context="validated smoke gold selected binding",
    )
    selected_rows, selected_raw, selected_source = _validated_selected(
        Path(str(smoke_gold["selected_path"])),
        smoke_gold=smoke_gold,
    )
    selected_iids = [str(row["iid"]) for row in selected_rows]
    if provenance["parent_selected_sha256"] != smoke_gold["value"][
        "parent_selected"
    ]["sha256"]:
        raise GokuActionV13RunArtifactError(
            "subset provenance parent-selected binding differs"
        )
    if provenance["ordered_iids"] != selected_iids:
        raise GokuActionV13RunArtifactError(
            "subset provenance IID order differs"
        )
    if (
        provenance["ordered_iids_sha256"]
        != selected_binding["ordered_iids_sha256"]
        or provenance["iid_set_sha256"]
        != selected_binding["iid_set_sha256"]
        or provenance["qwen_shard_row_counts"]
        != list(selected_binding["expected_shard_rows"])
    ):
        raise GokuActionV13RunArtifactError(
            "subset provenance IID/shard digest differs"
        )
    subset = _mapping(provenance["subset"], context="provenance subset")
    _closed(
        subset,
        {"path", "sha256", "bytes", "rows"},
        context="provenance subset",
    )
    if subset != {
        "path": expected_selected_path,
        "sha256": selected_binding["sha256"],
        "bytes": selected_binding["bytes"],
        "rows": selected_binding["rows"],
    }:
        raise GokuActionV13RunArtifactError(
            "subset provenance selected binding differs"
        )
    row_bindings = _sequence(
        provenance["row_bindings"],
        context="subset provenance row_bindings",
    )
    if len(row_bindings) != selected_binding["rows"]:
        raise GokuActionV13RunArtifactError(
            "subset provenance row binding count differs"
        )
    raw_lines = selected_raw.splitlines()
    for index, (value_row, iid, selected_row, raw_line) in enumerate(
        zip(row_bindings, selected_iids, selected_rows, raw_lines),
        start=1,
    ):
        row = _mapping(
            value_row,
            context=f"subset provenance row binding {index}",
        )
        _closed(
            row,
            {
                "line_number",
                "iid",
                "raw_line_sha256",
                "canonical_row_sha256",
                "assigned_shard",
            },
            context=f"subset provenance row binding {index}",
        )
        if (
            row["line_number"] != index
            or row["iid"] != iid
            or row["assigned_shard"]
            != _iid_shard(
                iid,
                num_shards=int(selected_binding["num_shards"]),
            )
            or row["raw_line_sha256"] != _sha256_bytes(raw_line)
            or row["canonical_row_sha256"]
            != _object_digest(selected_row)
        ):
            raise GokuActionV13RunArtifactError(
                f"subset provenance row binding {index} differs"
            )
        _sha256(
            row["raw_line_sha256"],
            context=f"subset row {index} raw SHA-256",
        )
        _sha256(
            row["canonical_row_sha256"],
            context=f"subset row {index} canonical SHA-256",
        )
    source = _mapping(provenance["source"], context="provenance source")
    _closed(
        source,
        {"path", "sha256", "bytes", "rows"},
        context="provenance source",
    )
    if source != {
        "path": str(selected_source),
        "sha256": selected_binding["sha256"],
        "bytes": selected_binding["bytes"],
        "rows": selected_binding["rows"],
    }:
        raise GokuActionV13RunArtifactError(
            "subset provenance source binding differs"
        )
    if str(resolved) != str(path.expanduser().resolve(strict=True)):
        raise GokuActionV13RunArtifactError(
            "subset provenance path did not resolve canonically"
        )
    return dict(provenance)


def _validate_submission_contract(
    path: Path,
) -> tuple[dict[str, Any], bytes, Path]:
    value, raw, resolved = _strict_json_file(
        path,
        context="submission contract",
        require_canonical=True,
    )
    contract = _mapping(value, context="submission contract")
    _closed(
        contract,
        {
            "schema_version",
            "selected",
            "smoke_gold",
            "model_closure",
            "source_snapshot",
            "model",
            "runtime",
            "outputs",
        },
        context="submission contract",
    )
    if contract["schema_version"] != SUBMISSION_SCHEMA:
        raise GokuActionV13RunArtifactError(
            "submission contract schema differs"
        )
    root = _directory(resolved.parent, context="submitted run root")
    if resolved != root / "submission_contract.json":
        raise GokuActionV13RunArtifactError(
            "submission contract must use the canonical run-root filename"
        )
    _directory(root / "logs", context="submitted logs directory")

    snapshot = _mapping(
        contract["source_snapshot"],
        context="submission source snapshot",
    )
    snapshot_keys = {
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
    _closed(snapshot, snapshot_keys, context="submission source snapshot")
    for field in (
        "tree_sha256",
        "manifest_sha256",
        "archive_sha256",
        "qwen_implementation_sha256",
        "finalizer_implementation_sha256",
        "verifier_implementation_sha256",
        "sbatch_sha256",
    ):
        _sha256(snapshot[field], context=f"source snapshot {field}")
    if snapshot["tree_sha256"] != snapshot["manifest_sha256"]:
        raise GokuActionV13RunArtifactError(
            "submission source tree/manifest SHA-256 differ"
        )
    frozen_source_bindings = {
        "qwen_implementation_sha256": (
            FROZEN_QWEN_IMPLEMENTATION_SHA256
        ),
        "finalizer_implementation_sha256": (
            FROZEN_FINALIZER_IMPLEMENTATION_SHA256
        ),
        "verifier_implementation_sha256": (
            FROZEN_ACCEPTANCE_VERIFIER_SHA256
        ),
        "sbatch_sha256": FROZEN_SBATCH_SHA256,
    }
    for field, expected in frozen_source_bindings.items():
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(
            expected
        ):
            raise GokuActionV13RunArtifactError(
                f"submission source-level {field} trust anchor is not frozen"
            )
        if snapshot[field] != expected:
            raise GokuActionV13RunArtifactError(
                f"submission {field} differs from frozen production"
            )
    snapshot_root = _directory(
        Path(str(snapshot["path"])),
        context="submission source snapshot",
    )
    manifest_path = _regular_file(
        Path(str(snapshot["manifest_path"])),
        context="submission source manifest",
    )
    if (
        manifest_path != snapshot_root / "SOURCE_FILES.jsonl"
        or _sha256_file(manifest_path) != snapshot["manifest_sha256"]
    ):
        raise GokuActionV13RunArtifactError(
            "submission source manifest binding differs"
        )
    archive_path = _regular_file(
        Path(str(snapshot["archive_path"])),
        context="submission source archive",
    )
    if _sha256_file(archive_path) != snapshot["archive_sha256"]:
        raise GokuActionV13RunArtifactError(
            "submission source archive binding differs"
        )
    role_fields = {
        "qwen": ("qwen_relpath", "qwen_implementation_sha256"),
        "finalizer": (
            "finalizer_relpath",
            "finalizer_implementation_sha256",
        ),
        "verifier": ("verifier_relpath", "verifier_implementation_sha256"),
        "sbatch": ("sbatch_relpath", "sbatch_sha256"),
    }
    for role, (path_field, sha_field) in role_fields.items():
        relative = snapshot[path_field]
        if relative != CONTRACT_SOURCE_RELATIVE_PATHS[role]:
            raise GokuActionV13RunArtifactError(
                f"submission {role} relative path differs"
            )
        implementation = _regular_file(
            snapshot_root / str(relative),
            context=f"submission {role} implementation",
        )
        if _sha256_file(implementation) != snapshot[sha_field]:
            raise GokuActionV13RunArtifactError(
                f"submission {role} implementation SHA-256 differs"
            )

    smoke_gold_binding = _mapping(
        contract["smoke_gold"],
        context="submission smoke_gold",
    )
    _closed(
        smoke_gold_binding,
        {"path", "sha256"},
        context="submission smoke_gold",
    )
    gold = _validate_smoke_gold(
        Path(str(smoke_gold_binding["path"])),
        source_snapshot=snapshot_root,
    )
    if dict(smoke_gold_binding) != {
        "path": gold["path"],
        "sha256": gold["sha256"],
    }:
        raise GokuActionV13RunArtifactError(
            "submission smoke-gold binding differs"
        )

    selected = _mapping(
        contract["selected"],
        context="submission selected",
    )
    _closed(
        selected,
        {"path", "sha256", "rows"},
        context="submission selected",
    )
    selected_binding = _mapping(
        gold["selected"],
        context="submission gold selected binding",
    )
    if (
        selected.get("sha256") != selected_binding["sha256"]
        or selected.get("rows") != selected_binding["rows"]
    ):
        raise GokuActionV13RunArtifactError(
            "submission selected identity differs from smoke gold"
        )
    selected_path = Path(str(selected["path"]))
    selected_rows, _, selected_resolved = _validated_selected(
        selected_path,
        smoke_gold=gold,
        require_gold_path=False,
    )
    if selected_resolved != root / "input" / "selected_smoke.jsonl":
        raise GokuActionV13RunArtifactError(
            "submission selected path binding differs"
        )
    _validate_subset_provenance(
        _regular_file(
            root / "input" / "subset_provenance.json",
            context="submission subset provenance",
        ),
        expected_selected_path=str(selected_resolved),
        smoke_gold=gold,
    )

    model = _mapping(contract["model"], context="submission model")
    _closed(
        model,
        {"path", "config_path", "config_sha256"},
        context="submission model",
    )
    model_root = _directory(Path(str(model["path"])), context="Qwen model")
    if str(model_root) != FROZEN_MODEL_PATH:
        raise GokuActionV13RunArtifactError(
            "submission model path differs from frozen production"
        )
    model_closure_binding = _validate_model_closure_binding(
        contract["model_closure"],
        source_snapshot=snapshot_root,
        model_path=model_root,
    )
    if (
        _sha256(
            model["config_sha256"],
            context="submission model config SHA-256",
        )
        != FROZEN_MODEL_CONFIG_SHA256
    ):
        raise GokuActionV13RunArtifactError(
            "submission model config SHA-256 differs from frozen production"
        )
    config_logical = Path(str(model["config_path"]))
    if config_logical != model_root / "config.json":
        raise GokuActionV13RunArtifactError(
            "submission model config path differs"
        )
    config = config_logical.resolve(strict=True)
    if (
        not config.is_file()
        or _sha256_file(config) != model["config_sha256"]
    ):
        raise GokuActionV13RunArtifactError(
            "submission model config binding differs"
        )

    runtime = _mapping(contract["runtime"], context="submission runtime")
    runtime_keys = {
        "num_shards",
        "max_samples",
        "max_new_tokens",
        "nframes",
        "max_pixels",
        "attn_implementation",
        "allow_download",
        "repair_attempts",
        "final_seed",
        "allow_partial",
    }
    _closed(runtime, runtime_keys, context="submission runtime")
    expected_runtime = {
        "num_shards": 8,
        "max_samples": None,
        "max_new_tokens": 1_536,
        "nframes": 12,
        "max_pixels": 589_824,
        "attn_implementation": "sdpa",
        "allow_download": False,
        "repair_attempts": 1,
        "final_seed": 260_730,
        "allow_partial": True,
    }
    if dict(runtime) != expected_runtime:
        raise GokuActionV13RunArtifactError(
            "submission runtime differs from frozen smoke"
        )

    outputs = _mapping(contract["outputs"], context="submission outputs")
    _closed(
        outputs,
        {"qwen_root", "final_output"},
        context="submission outputs",
    )
    expected_outputs = {
        "qwen_root": str(root / "qwen8"),
        "final_output": str(root / "final"),
    }
    if dict(outputs) != expected_outputs:
        raise GokuActionV13RunArtifactError(
            "submission output paths differ"
        )

    # Keep selected rows only in memory; they are never serialized into the
    # verifier-facing closed schema.
    result = dict(contract)
    result["_selected_rows"] = selected_rows
    result["_smoke_gold"] = gold
    result["_model_closure"] = model_closure_binding
    return result, raw, resolved


def _validate_one_qwen_shard(
    *,
    shard_index: int,
    qwen_root: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
) -> dict[str, Any]:
    gold = _mapping(
        submission["_smoke_gold"],
        context="submission validated smoke gold",
    )
    selected_binding = _mapping(
        gold["selected"],
        context="submission gold selected binding",
    )
    num_shards = int(selected_binding["num_shards"])
    shard = _regular_file(
        qwen_root / f"qwen_shard_{shard_index:03d}.jsonl",
        context=f"Qwen shard {shard_index}",
    )
    receipt_path = _regular_file(
        qwen_root / f"qwen_shard_{shard_index:03d}.receipt.json",
        context=f"Qwen shard {shard_index} receipt",
    )
    rows, raw, _ = _strict_jsonl_file(
        shard,
        context=f"Qwen shard {shard_index}",
        allow_empty=True,
        require_canonical_rows=True,
    )
    expected_iids = [
        str(row["iid"])
        for row in selected_rows
        if _iid_shard(
            str(row["iid"]),
            num_shards=num_shards,
        )
        == shard_index
    ]
    actual_iids = [str(row.get("iid", "")) for row in rows]
    if actual_iids != expected_iids:
        raise GokuActionV13RunArtifactError(
            f"Qwen shard {shard_index} IID order/coverage differs"
        )
    if len(rows) != selected_binding["expected_shard_rows"][
        shard_index
    ]:
        raise GokuActionV13RunArtifactError(
            f"Qwen shard {shard_index} row count differs"
        )

    receipt_value, receipt_raw, _ = _strict_json_file(
        receipt_path,
        context=f"Qwen shard {shard_index} receipt",
        require_canonical=True,
    )
    receipt = _mapping(
        receipt_value,
        context=f"Qwen shard {shard_index} receipt",
    )
    receipt_keys = {
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
    _closed(
        receipt,
        receipt_keys,
        context=f"Qwen shard {shard_index} receipt",
    )
    selected = _mapping(
        submission["selected"],
        context="submission selected",
    )
    snapshot = _mapping(
        submission["source_snapshot"],
        context="submission source snapshot",
    )
    model = _mapping(submission["model"], context="submission model")
    expected_receipt_bindings = {
        "schema_version": SHARD_RECEIPT_SCHEMA,
        "status": "complete",
        "execution_manifest": selected["path"],
        "execution_manifest_sha256": selected["sha256"],
        "root": str(Path(str(selected["path"])).parent),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "assigned_iids": expected_iids,
        "implementation_digest": snapshot[
            "qwen_implementation_sha256"
        ],
        "model_path": model["path"],
    }
    for field, expected in expected_receipt_bindings.items():
        if receipt.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"Qwen shard {shard_index} receipt {field} differs"
            )
    for field in (
        "config_digest",
        "run_config_digest",
        "receipt_digest",
    ):
        _sha256(
            receipt[field],
            context=f"Qwen shard {shard_index} receipt {field}",
        )
    run_config = _mapping(
        receipt["run_config"],
        context=f"Qwen shard {shard_index} run_config",
    )
    if _object_digest(run_config) != receipt["run_config_digest"]:
        raise GokuActionV13RunArtifactError(
            f"Qwen shard {shard_index} run_config digest differs"
        )
    runtime = _mapping(
        submission["runtime"],
        context="submission runtime",
    )
    run_config_bindings = {
        "model_path": model["path"],
        "max_samples": runtime["max_samples"],
        "num_shards": runtime["num_shards"],
        "max_new_tokens": runtime["max_new_tokens"],
        "nframes": runtime["nframes"],
        "max_pixels": runtime["max_pixels"],
        "attn_implementation": runtime["attn_implementation"],
        "allow_download": runtime["allow_download"],
        "repair_attempts": runtime["repair_attempts"],
        "implementation_digest": snapshot[
            "qwen_implementation_sha256"
        ],
    }
    for field, expected in run_config_bindings.items():
        if run_config.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"Qwen shard {shard_index} run_config {field} differs"
            )

    output = _mapping(
        receipt["output"],
        context=f"Qwen shard {shard_index} receipt output",
    )
    _closed(
        output,
        {"path", "sha256", "bytes", "rows", "status_counts"},
        context=f"Qwen shard {shard_index} receipt output",
    )
    expected_status_counts = {"ok": len(rows)} if rows else {}
    expected_output = {
        "path": str(shard),
        "sha256": _sha256_bytes(raw),
        "bytes": len(raw),
        "rows": len(rows),
        "status_counts": expected_status_counts,
    }
    if dict(output) != expected_output:
        raise GokuActionV13RunArtifactError(
            f"Qwen shard {shard_index} receipt output differs"
        )
    if receipt["receipt_digest"] != _self_digest(
        receipt,
        digest_field="receipt_digest",
    ):
        raise GokuActionV13RunArtifactError(
            f"Qwen shard {shard_index} receipt self-digest differs"
        )

    for row_number, row in enumerate(rows, start=1):
        row_bindings = {
            "iid": expected_iids[row_number - 1],
            "status": "ok",
            "execution_manifest": selected["path"],
            "execution_manifest_sha256": selected["sha256"],
            "shard_index": shard_index,
            "num_shards": num_shards,
            "implementation_digest": snapshot[
                "qwen_implementation_sha256"
            ],
            "config_digest": receipt["config_digest"],
            "run_config_digest": receipt["run_config_digest"],
            "model_path": receipt["model_path"],
            "model_revision": receipt["model_revision"],
            "transformers_version": receipt["transformers_version"],
        }
        for field, expected in row_bindings.items():
            if row.get(field) != expected:
                raise GokuActionV13RunArtifactError(
                    f"Qwen shard {shard_index} row {row_number} "
                    f"{field} differs"
                )

    return {
        "index": shard_index,
        "path": str(shard),
        "sha256": _sha256_bytes(raw),
        "bytes": len(raw),
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_bytes(receipt_raw),
    }


def _validate_qwen_terminal(
    *,
    qwen_root_path: Path,
    selected_rows: Sequence[Mapping[str, Any]],
    submission: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gold = _mapping(
        submission["_smoke_gold"],
        context="submission validated smoke gold",
    )
    num_shards = int(gold["selected"]["num_shards"])
    qwen_root = _directory(qwen_root_path, context="Qwen terminal root")
    expected_names = {
        *(
            f"qwen_shard_{index:03d}.jsonl"
            for index in range(num_shards)
        ),
        *(
            f"qwen_shard_{index:03d}.receipt.json"
            for index in range(num_shards)
        ),
    }
    actual_names: set[str] = set()
    for entry in qwen_root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise GokuActionV13RunArtifactError(
                f"Qwen terminal root has non-regular entry: {entry.name}"
            )
        actual_names.add(entry.name)
    if actual_names != expected_names:
        raise GokuActionV13RunArtifactError(
            "Qwen terminal file closure differs: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    return [
        _validate_one_qwen_shard(
            shard_index=index,
            qwen_root=qwen_root,
            selected_rows=selected_rows,
            submission=submission,
        )
        for index in range(num_shards)
    ]


def _validate_final_terminal(
    *,
    final_dir_path: Path,
    submission: Mapping[str, Any],
    qwen_shards: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    gold = _mapping(
        submission["_smoke_gold"],
        context="submission validated smoke gold",
    )
    labels_by_iid = _mapping(
        gold["labels_by_iid"],
        context="validated smoke gold labels",
    )
    hard_pass_iids = {
        iid
        for iid, label_value in labels_by_iid.items()
        if _mapping(
            label_value,
            context=f"smoke gold label iid={iid}",
        )["label"]
        == "admissible"
    }
    hard_pass_rows = len(hard_pass_iids)
    rejected_rows = len(labels_by_iid) - hard_pass_rows
    selected_binding = _mapping(
        gold["selected"],
        context="submission gold selected binding",
    )
    num_shards = int(selected_binding["num_shards"])
    final_dir = _directory(final_dir_path, context="final terminal directory")
    actual_names: set[str] = set()
    for entry in final_dir.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise GokuActionV13RunArtifactError(
                f"final directory has non-regular entry: {entry.name}"
            )
        actual_names.add(entry.name)
    expected_names = set(FINAL_ARTIFACT_NAMES)
    if actual_names != expected_names:
        raise GokuActionV13RunArtifactError(
            "final artifact closure differs: "
            f"missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    row_counts = {
        "review_candidates.jsonl": hard_pass_rows,
        "proposed_128.jsonl": hard_pass_rows,
        "reserve_32.jsonl": 0,
        "generation_manifest.jsonl": hard_pass_rows,
    }
    parsed_rows: dict[str, list[dict[str, Any]]] = {}
    artifact_raw: dict[str, bytes] = {}
    for name, expected_rows in row_counts.items():
        path = _regular_file(
            final_dir / name,
            context=f"final artifact {name}",
        )
        rows, raw, _ = _strict_jsonl_file(
            path,
            context=f"final artifact {name}",
            allow_empty=(expected_rows == 0),
            require_canonical_rows=True,
        )
        if len(rows) != expected_rows:
            raise GokuActionV13RunArtifactError(
                f"final artifact {name} row count differs"
            )
        parsed_rows[name] = rows
        artifact_raw[name] = raw

    for name in (
        "review_candidates.jsonl",
        "proposed_128.jsonl",
        "generation_manifest.jsonl",
    ):
        iids = [str(row.get("iid", "")) for row in parsed_rows[name]]
        if len(iids) != len(set(iids)) or set(iids) != hard_pass_iids:
            raise GokuActionV13RunArtifactError(
                f"final artifact {name} IID set differs"
            )
    for name in ("review_candidates.jsonl", "proposed_128.jsonl"):
        for row in parsed_rows[name]:
            finalization = _mapping(
                row.get("action_anchor_finalization"),
                context=f"{name} action_anchor_finalization",
            )
            expected = {
                "hard_gate_passed": True,
                "hard_gate_failures": [],
                "human_review_status": "pending",
                "human_label": False,
                "generation_authorized": False,
                "manifest_role": "review_proposal",
                "production_eligible": False,
                "approval": None,
                "authorization_interface_available": False,
            }
            for field, value in expected.items():
                if finalization.get(field) != value:
                    raise GokuActionV13RunArtifactError(
                        f"{name} iid={row.get('iid')} {field} differs"
                    )
    for row in parsed_rows["generation_manifest.jsonl"]:
        expected = {
            "human_review_status": "pending",
            "generation_authorized": False,
            "manifest_role": "review_proposal",
            "production_eligible": False,
            "approval": None,
            "authorization_interface_available": False,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise GokuActionV13RunArtifactError(
                    "generation manifest is not review-only: "
                    f"iid={row.get('iid')} field={field}"
                )

    summary_value, summary_raw, _ = _strict_json_file(
        final_dir / "summary.json",
        context="final summary",
    )
    summary = _mapping(summary_value, context="final summary")
    artifact_raw["summary.json"] = summary_raw
    if summary.get("schema_version") != (
        "motive-goku-action-anchor-finalize-v8"
    ):
        raise GokuActionV13RunArtifactError(
            "final summary schema differs"
        )
    selected = _mapping(
        submission["selected"],
        context="submission selected",
    )
    summary_input = _mapping(summary.get("input"), context="summary input")
    expected_summary_input = {
        "selected_path": selected["path"],
        "selected_rows": selected["rows"],
        "selected_sha256": selected["sha256"],
        "qwen_num_shards": 8,
        "qwen_implementation_digest": submission["source_snapshot"][
            "qwen_implementation_sha256"
        ],
    }
    for field, expected in expected_summary_input.items():
        if summary_input.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"final summary input {field} differs"
            )
    summary_shards = _sequence(
        summary_input.get("qwen_shards"),
        context="summary qwen_shards",
    )
    if len(summary_shards) != num_shards:
        raise GokuActionV13RunArtifactError(
            "final summary Qwen shard count differs"
        )
    for expected_index, (summary_shard, completion_shard) in enumerate(
        zip(summary_shards, qwen_shards)
    ):
        item = _mapping(
            summary_shard,
            context=f"summary Qwen shard {expected_index}",
        )
        expected = {
            "index": expected_index,
            "path": completion_shard["path"],
            "sha256": completion_shard["sha256"],
            "bytes": completion_shard["bytes"],
            "rows": selected_binding["expected_shard_rows"][
                expected_index
            ],
            "receipt_path": completion_shard["receipt_path"],
            "receipt_sha256": completion_shard["receipt_sha256"],
        }
        for field, value in expected.items():
            if item.get(field) != value:
                raise GokuActionV13RunArtifactError(
                    f"summary Qwen shard {expected_index} {field} differs"
                )
    hard_gate = _mapping(
        summary.get("hard_gate"),
        context="final summary hard_gate",
    )
    if (
        hard_gate.get("passed_rows") != hard_pass_rows
        or hard_gate.get("rejected_rows") != rejected_rows
    ):
        raise GokuActionV13RunArtifactError(
            "final summary hard-gate counts differ from smoke gold"
        )
    selection = _mapping(
        summary.get("selection"),
        context="final summary selection",
    )
    expected_selection = {
        "allow_partial": True,
        "review_rows": hard_pass_rows,
        "proposed_rows": hard_pass_rows,
        "reserve_rows": 0,
        "generation_rows": hard_pass_rows,
    }
    for field, expected in expected_selection.items():
        if selection.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"final summary selection {field} differs"
            )
    semantics = _mapping(
        summary.get("semantics"),
        context="final summary semantics",
    )
    expected_semantics = {
        "manifest_role": "review_proposal",
        "human_review_status": "pending",
        "human_labels_asserted": False,
        "generation_authorized": False,
        "production_eligible": False,
        "approval": None,
        "authorization_interface_available": False,
    }
    for field, expected in expected_semantics.items():
        if semantics.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"final summary semantics {field} differs"
            )
    if summary.get("implementation_sha256") != submission[
        "source_snapshot"
    ]["finalizer_implementation_sha256"]:
        raise GokuActionV13RunArtifactError(
            "final summary implementation SHA-256 differs"
        )
    summary_outputs = _mapping(
        summary.get("output_sha256"),
        context="final summary output_sha256",
    )
    expected_summary_outputs = {
        name: _sha256_bytes(artifact_raw[name])
        for name in row_counts
    }
    if dict(summary_outputs) != dict(sorted(expected_summary_outputs.items())):
        raise GokuActionV13RunArtifactError(
            "final summary output SHA-256 bindings differ"
        )

    done_value, done_raw, _ = _strict_json_file(
        final_dir / "done.json",
        context="final done marker",
    )
    done = _mapping(done_value, context="final done marker")
    artifact_raw["done.json"] = done_raw
    _closed(
        done,
        {
            "schema_version",
            "status",
            "summary_sha256",
            "implementation_sha256",
            "output_sha256",
        },
        context="final done marker",
    )
    if (
        done["schema_version"]
        != "motive-goku-action-anchor-finalize-done-v8"
        or done["status"] != "complete"
        or done["summary_sha256"] != _sha256_bytes(summary_raw)
        or done["implementation_sha256"]
        != submission["source_snapshot"][
            "finalizer_implementation_sha256"
        ]
    ):
        raise GokuActionV13RunArtifactError(
            "final done marker terminal binding differs"
        )
    expected_done_outputs = {
        **expected_summary_outputs,
        "summary.json": _sha256_bytes(summary_raw),
    }
    if done["output_sha256"] != dict(sorted(expected_done_outputs.items())):
        raise GokuActionV13RunArtifactError(
            "final done output SHA-256 bindings differ"
        )

    return {
        name: {
            "path": str(final_dir / name),
            "sha256": _sha256_bytes(artifact_raw[name]),
            "bytes": len(artifact_raw[name]),
        }
        for name in FINAL_ARTIFACT_NAMES
    }


def complete_run(
    *,
    submission_contract: str | Path,
    job_id: str,
    output: str | Path,
    qwen_root: str | Path | None = None,
    final_output: str | Path | None = None,
) -> dict[str, Any]:
    """Publish a terminal receipt for one successful frozen smoke job."""

    if not isinstance(job_id, str) or not _SLURM_JOB_ID_RE.fullmatch(job_id):
        raise GokuActionV13RunArtifactError(
            "job_id must be a positive decimal Slurm job identifier"
        )
    output_path = Path(output).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    submission, submission_raw, submission_path = (
        _validate_submission_contract(Path(submission_contract))
    )
    selected_rows = submission.pop("_selected_rows")
    outputs = _mapping(submission["outputs"], context="submission outputs")
    expected_qwen_root = Path(str(outputs["qwen_root"])).resolve(
        strict=True
    )
    expected_final_output = Path(str(outputs["final_output"])).resolve(
        strict=True
    )
    if qwen_root is not None and (
        Path(qwen_root).expanduser().resolve(strict=True)
        != expected_qwen_root
    ):
        raise GokuActionV13RunArtifactError(
            "explicit qwen_root differs from submission contract"
        )
    if final_output is not None and (
        Path(final_output).expanduser().resolve(strict=True)
        != expected_final_output
    ):
        raise GokuActionV13RunArtifactError(
            "explicit final_output differs from submission contract"
        )
    qwen_shards = _validate_qwen_terminal(
        qwen_root_path=expected_qwen_root,
        selected_rows=selected_rows,
        submission=submission,
    )
    final_artifacts = _validate_final_terminal(
        final_dir_path=expected_final_output,
        submission=submission,
        qwen_shards=qwen_shards,
    )
    receipt: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "job_id": job_id,
        "submission_contract_path": str(submission_path),
        "submission_contract_sha256": _sha256_bytes(submission_raw),
        "selected_sha256": submission["selected"]["sha256"],
        "smoke_gold_sha256": submission["smoke_gold"]["sha256"],
        "model_closure": dict(submission["model_closure"]),
        "qwen_root": str(expected_qwen_root),
        "final_output": str(expected_final_output),
        "qwen_shards": qwen_shards,
        "final_artifacts": final_artifacts,
    }
    expected_output = submission_path.parent / "completion_receipt.json"
    if output_path.resolve(strict=False) != expected_output:
        raise GokuActionV13RunArtifactError(
            "completion receipt must use the canonical run-root path"
        )
    _write_new_json(expected_output, receipt)
    return receipt


def _validate_completion_receipt(
    path: Path,
    *,
    submission: Mapping[str, Any],
    submission_raw: bytes,
    submission_path: Path,
) -> tuple[dict[str, Any], bytes, Path]:
    value, raw, resolved = _strict_json_file(
        path,
        context="completion receipt",
        require_canonical=True,
    )
    receipt = _mapping(value, context="completion receipt")
    _closed(
        receipt,
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
        context="completion receipt",
    )
    if (
        receipt["schema_version"] != COMPLETION_SCHEMA
        or receipt["status"] != "complete"
    ):
        raise GokuActionV13RunArtifactError(
            "completion receipt is not terminal complete"
        )
    if (
        not isinstance(receipt["job_id"], str)
        or not _SLURM_JOB_ID_RE.fullmatch(receipt["job_id"])
    ):
        raise GokuActionV13RunArtifactError(
            "completion receipt job_id is invalid"
        )
    expected_top = {
        "submission_contract_path": str(submission_path),
        "submission_contract_sha256": _sha256_bytes(submission_raw),
        "selected_sha256": submission["selected"]["sha256"],
        "smoke_gold_sha256": submission["smoke_gold"]["sha256"],
        "model_closure": submission["model_closure"],
        "qwen_root": submission["outputs"]["qwen_root"],
        "final_output": submission["outputs"]["final_output"],
    }
    for field, expected in expected_top.items():
        if receipt.get(field) != expected:
            raise GokuActionV13RunArtifactError(
                f"completion receipt {field} differs"
            )
    qwen_shards = _sequence(
        receipt["qwen_shards"],
        context="completion qwen_shards",
    )
    num_shards = int(submission["runtime"]["num_shards"])
    if len(qwen_shards) != num_shards:
        raise GokuActionV13RunArtifactError(
            "completion receipt Qwen shard count differs"
        )
    for index, value_item in enumerate(qwen_shards):
        item = _mapping(
            value_item,
            context=f"completion Qwen shard {index}",
        )
        _closed(
            item,
            {
                "index",
                "path",
                "sha256",
                "bytes",
                "receipt_path",
                "receipt_sha256",
            },
            context=f"completion Qwen shard {index}",
        )
        if item["index"] != index:
            raise GokuActionV13RunArtifactError(
                f"completion Qwen shard {index} index differs"
            )
        expected_shard_path = (
            Path(str(receipt["qwen_root"]))
            / f"qwen_shard_{index:03d}.jsonl"
        ).resolve(strict=True)
        expected_receipt_path = (
            Path(str(receipt["qwen_root"]))
            / f"qwen_shard_{index:03d}.receipt.json"
        ).resolve(strict=True)
        if (
            item["path"] != str(expected_shard_path)
            or item["receipt_path"] != str(expected_receipt_path)
            or item["sha256"] != _sha256_file(expected_shard_path)
            or item["receipt_sha256"]
            != _sha256_file(expected_receipt_path)
            or item["bytes"] != expected_shard_path.stat().st_size
        ):
            raise GokuActionV13RunArtifactError(
                f"completion Qwen shard {index} file binding differs"
            )

    final_artifacts = _mapping(
        receipt["final_artifacts"],
        context="completion final_artifacts",
    )
    if set(final_artifacts) != set(FINAL_ARTIFACT_NAMES):
        raise GokuActionV13RunArtifactError(
            "completion final artifact name set differs"
        )
    for name in FINAL_ARTIFACT_NAMES:
        binding = _mapping(
            final_artifacts[name],
            context=f"completion final artifact {name}",
        )
        _closed(
            binding,
            {"path", "sha256", "bytes"},
            context=f"completion final artifact {name}",
        )
        expected_path = (
            Path(str(receipt["final_output"])) / name
        ).resolve(strict=True)
        if (
            binding["path"] != str(expected_path)
            or binding["sha256"] != _sha256_file(expected_path)
            or binding["bytes"] != expected_path.stat().st_size
        ):
            raise GokuActionV13RunArtifactError(
                f"completion final artifact {name} binding differs"
            )
    if resolved != submission_path.parent / "completion_receipt.json":
        raise GokuActionV13RunArtifactError(
            "completion receipt canonical path differs"
        )
    return dict(receipt), raw, resolved


def build_acceptance_contract(
    *,
    submission_contract: str | Path,
    completion_receipt: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Publish the exact immutable input to the independent verifier."""

    output_path = Path(output).expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    submission, submission_raw, submission_path = (
        _validate_submission_contract(Path(submission_contract))
    )
    submission.pop("_selected_rows")
    gold = submission.pop("_smoke_gold")
    submission.pop("_model_closure")
    completion, completion_raw, _ = _validate_completion_receipt(
        Path(completion_receipt),
        submission=submission,
        submission_raw=submission_raw,
        submission_path=submission_path,
    )
    runtime = _mapping(submission["runtime"], context="submission runtime")
    contract: dict[str, Any] = {
        "schema_version": ACCEPTANCE_CONTRACT_SCHEMA,
        "selected": {
            "rows": gold["selected"]["rows"],
            "sha256": gold["selected"]["sha256"],
            "ordered_iids_sha256": gold["selected"][
                "ordered_iids_sha256"
            ],
        },
        "smoke_gold": {
            "path": submission["smoke_gold"]["path"],
            "sha256": submission["smoke_gold"]["sha256"],
        },
        "model_closure": dict(submission["model_closure"]),
        "expected_shard_counts": list(
            gold["selected"]["expected_shard_rows"]
        ),
        "source_snapshot": dict(submission["source_snapshot"]),
        "model": dict(submission["model"]),
        "execution": {
            "num_shards": runtime["num_shards"],
            "max_samples": runtime["max_samples"],
            "max_new_tokens": runtime["max_new_tokens"],
            "nframes": runtime["nframes"],
            "max_pixels": runtime["max_pixels"],
            "attn_implementation": runtime["attn_implementation"],
            "allow_download": runtime["allow_download"],
            "repair_attempts": runtime["repair_attempts"],
        },
        "final": {
            "seed": runtime["final_seed"],
            "allow_partial": runtime["allow_partial"],
            "manifest_role": "review_proposal",
            "human_review_status": "pending",
            "generation_authorized": False,
            "production_eligible": False,
            "wan_generation_authorized": False,
        },
        "bindings": {
            "submission_contract_sha256": _sha256_bytes(submission_raw),
            "completion_receipt_sha256": _sha256_bytes(completion_raw),
        },
    }
    expected_output = submission_path.parent / "acceptance_contract.json"
    if output_path.resolve(strict=False) != expected_output:
        raise GokuActionV13RunArtifactError(
            "acceptance contract must use the canonical run-root path"
        )
    # Reassert completion is used, even though only its file hash is persisted.
    if completion["selected_sha256"] != submission["selected"]["sha256"]:
        raise GokuActionV13RunArtifactError(
            "completion selected binding differs before acceptance"
        )
    if (
        completion["smoke_gold_sha256"]
        != submission["smoke_gold"]["sha256"]
    ):
        raise GokuActionV13RunArtifactError(
            "completion smoke-gold binding differs before acceptance"
        )
    if completion["model_closure"] != submission["model_closure"]:
        raise GokuActionV13RunArtifactError(
            "completion model-closure binding differs before acceptance"
        )
    _write_new_json(expected_output, contract)
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build closed, no-overwrite control artifacts for the frozen "
            "Goku action-editing v16 smoke."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="atomically create a fresh run root and submission contract",
    )
    prepare.add_argument("--run-root", required=True, type=Path)
    prepare.add_argument("--frozen-selected", required=True, type=Path)
    prepare.add_argument("--smoke-gold", required=True, type=Path)
    prepare.add_argument("--model-closure", required=True, type=Path)
    prepare.add_argument("--source-snapshot", required=True, type=Path)
    prepare.add_argument("--source-archive", required=True, type=Path)
    prepare.add_argument("--source-tree-sha256", required=True)
    prepare.add_argument("--source-manifest-sha256", required=True)
    prepare.add_argument("--source-archive-sha256", required=True)
    prepare.add_argument("--model", required=True, type=Path)
    prepare.add_argument("--model-config-sha256", required=True)

    complete = subparsers.add_parser(
        "complete",
        help="bind one completed job to exact Qwen/final artifacts",
    )
    complete.add_argument(
        "--submission-contract",
        required=True,
        type=Path,
    )
    complete.add_argument("--job-id", required=True)
    complete.add_argument("--qwen-root", type=Path)
    complete.add_argument("--final-output", type=Path)
    complete.add_argument("--output", required=True, type=Path)

    acceptance = subparsers.add_parser(
        "acceptance-contract",
        help="build the immutable independent-verifier input contract",
    )
    acceptance.add_argument(
        "--submission-contract",
        required=True,
        type=Path,
    )
    acceptance.add_argument(
        "--completion-receipt",
        required=True,
        type=Path,
    )
    acceptance.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_run(
            run_root=args.run_root,
            frozen_selected=args.frozen_selected,
            smoke_gold=args.smoke_gold,
            model_closure=args.model_closure,
            source_snapshot=args.source_snapshot,
            source_archive=args.source_archive,
            source_tree_sha256=args.source_tree_sha256,
            source_manifest_sha256=args.source_manifest_sha256,
            source_archive_sha256=args.source_archive_sha256,
            model_path=args.model,
            model_config_sha256=args.model_config_sha256,
        )
    elif args.command == "complete":
        result = complete_run(
            submission_contract=args.submission_contract,
            job_id=args.job_id,
            qwen_root=args.qwen_root,
            final_output=args.final_output,
            output=args.output,
        )
    elif args.command == "acceptance-contract":
        result = build_acceptance_contract(
            submission_contract=args.submission_contract,
            completion_receipt=args.completion_receipt,
            output=args.output,
        )
    else:  # pragma: no cover - argparse enforces the subcommand.
        raise AssertionError(args.command)
    print(_canonical_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

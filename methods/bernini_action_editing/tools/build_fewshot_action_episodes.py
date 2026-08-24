#!/usr/bin/env python3
"""Build hash-bound, preview-only few-shot action episodes for Bernini CPMR.

The builder joins the authoritative OmniVideo2 preview manifest to Bernini's
finalized VAE index by exact IID.  It never reads ``family`` when assigning an
action: the only semantic source is
``target_plan.dynamic_subject_targets[0].target_action_signature``.

Programmatic construction is deliberately side-effect free.  It returns the
canonical JSONL and receipt bytes as a dry-run payload.  The CLI publishes the
two files only when ``--publish`` is supplied, using create-only hard links and
publishing the receipt last as the ready marker.

These episodes remain experimental previews.  A successful build is not a
human review, training authorization, production authorization, or scientific
quality claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence


PREVIEW_ROW_SCHEMA = "omnivideo2-action-preview-row-v1"
VAE_INDEX_ROW_SCHEMA = "bernini-r-action-vae-index-row-v2"
EPISODE_ROW_SCHEMA = "bernini-cpmr-fewshot-action-episode-v1"
EPISODE_RECEIPT_SCHEMA = "bernini-cpmr-fewshot-action-episode-receipt-v1"
ONTOLOGY_SCHEMA = "bernini-cpmr-atomic-action-ontology-v1"

DEFAULT_K_SHOT = 2
DEFAULT_SEED = 20260807
LATENT_PHASE_COUNT = 21

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SIGNATURE_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")

PREVIEW_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "group_id",
        "family",
        "source_video_path",
        "source_video_sha256",
        "target_video_path",
        "target_video_sha256",
        "edit_instruction",
        "edit_instruction_sha256",
        "instruction_source",
        "generation_instruction",
        "generation_instruction_sha256",
        "source_census",
        "target_plan",
        "selection_gates",
        "preview_only",
        "training_authorized",
        "training_use_forbidden",
        "production_eligible",
        "post_video_acceptance",
        "provenance",
        "row_digest",
    }
)

VAE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "iid",
        "parquet_path",
        "parquet_sha256",
        "materialized_row_digest",
        "bucket_hw",
        "posterior_parameters_shape",
        "sample_receipt_path",
        "sample_receipt_sha256",
        "preview_only",
        "production_claim_forbidden",
    }
)

REQUIRED_SELECTION_GATES = (
    "single_dynamic_actor",
    "source_camera_locked_off",
    "target_camera_locked_off",
    "target_camera_preserve_static",
    "source_census_high_confidence",
    "target_plan_high_confidence",
)

# Rules are deliberately small, explicit, and versioned.  An alias is matched
# as a whole underscore-delimited token sequence.  A signature is retained
# only when exactly one primitive matches and no coordination token is present.
_ACTION_ALIASES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("bend", (("bend",), ("bow",))),
    ("catch", (("catch",), ("receive",))),
    ("clap", (("clap",),)),
    ("crawl", (("crawl",),)),
    ("crouch", (("crouch",), ("squat",))),
    ("dance", (("dance",),)),
    ("drop", (("drop",), ("release",))),
    ("jump", (("jump",), ("hop",), ("leap",))),
    ("kick", (("kick",),)),
    ("kneel", (("kneel",),)),
    ("lower", (("lower",),)),
    ("nod", (("nod",),)),
    ("pick_up", (("pick", "up"),)),
    ("point", (("point",),)),
    ("pull", (("pull",),)),
    ("punch", (("punch",),)),
    ("push", (("push",),)),
    ("raise", (("raise",), ("lift",))),
    ("reach", (("reach",),)),
    ("roll", (("roll",),)),
    ("run", (("run",), ("jog",), ("sprint",))),
    ("shake_head", (("shake", "head"),)),
    ("sit", (("sit",),)),
    ("stand", (("stand",),)),
    ("swing", (("swing",),)),
    ("throw", (("throw",), ("toss",))),
    ("touch", (("touch",), ("tap",))),
    ("transfer", (("transfer",), ("hand", "over"))),
    ("turn", (("turn",), ("rotate",), ("spin",))),
    ("walk", (("walk",), ("step",), ("stroll",))),
    ("wave", (("wave",),)),
)

_COMPOSITION_TOKENS = frozenset(
    {"and", "then", "while", "plus", "followed", "before", "after"}
)


class FewShotEpisodeError(RuntimeError):
    """An input contract, split invariant, or publication invariant failed."""


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
        raise FewShotEpisodeError(f"value is not canonical JSON: {error}") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


ONTOLOGY_SPEC: dict[str, Any] = {
    "schema_version": ONTOLOGY_SCHEMA,
    "semantic_source": (
        "target_plan.dynamic_subject_targets[0].target_action_signature"
    ),
    "source_family_used": False,
    "composition_tokens_rejected": sorted(_COMPOSITION_TOKENS),
    "matching_policy": (
        "lower_snake_case_whole_token_alias_exactly_one_primitive"
    ),
    "primitive_aliases": {
        primitive: ["_".join(alias) for alias in aliases]
        for primitive, aliases in _ACTION_ALIASES
    },
}
ONTOLOGY_SHA256 = object_sha256(ONTOLOGY_SPEC)


@dataclass(frozen=True)
class AtomicActionMatch:
    primitive_id: str
    matched_aliases: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    iid: str
    group_id: str
    source_video_sha256: str
    target_video_sha256: str
    edit_instruction: str
    edit_instruction_sha256: str
    target_action_signature: str
    primitive_id: str
    matched_aliases: tuple[str, ...]
    subject_id: str
    stable_reference: str
    preview_row_digest: str
    vae_index_row: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodeBuildPayload:
    """Side-effect-free canonical bytes ready for optional CLI publication."""

    jsonl_bytes: bytes
    receipt_bytes: bytes
    receipt: Mapping[str, Any]
    output_jsonl: Path
    output_receipt: Path


def _required_sha256(value: Any, *, context: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise FewShotEpisodeError(f"{context} must be a lowercase SHA-256")
    return value


def _required_text(value: Any, *, context: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise FewShotEpisodeError(f"{context} must be non-empty text without NUL")
    return value


def _required_iid(value: Any, *, context: str) -> str:
    if type(value) is not str or _IID_RE.fullmatch(value) is None:
        raise FewShotEpisodeError(f"unsafe {context}: {value!r}")
    return value


def _mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FewShotEpisodeError(f"{context} must be an object")
    return dict(value)


def _sequence(value: Any, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise FewShotEpisodeError(f"{context} must be a list")
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FewShotEpisodeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise FewShotEpisodeError(f"non-finite JSON number: {value}")


def _decode_json_object(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except FewShotEpisodeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FewShotEpisodeError(f"invalid {context}: {error}") from error
    if not isinstance(value, dict):
        raise FewShotEpisodeError(f"{context} must contain one JSON object")
    return value


def _plain_file(path: Path, *, context: str) -> Path:
    requested = path.expanduser()
    try:
        mode = requested.lstat().st_mode
    except FileNotFoundError as error:
        raise FewShotEpisodeError(f"missing {context}: {requested}") from error
    if not stat.S_ISREG(mode) or requested.is_symlink():
        raise FewShotEpisodeError(f"{context} must be a plain non-symlink file")
    try:
        return requested.resolve(strict=True)
    except OSError as error:
        raise FewShotEpisodeError(f"cannot resolve {context}: {error}") from error


def _read_stable_bytes(path: Path, *, context: str) -> tuple[Path, bytes]:
    resolved = _plain_file(path, context=context)
    try:
        before = resolved.stat()
        with resolved.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                before.st_dev != opened.st_dev
                or before.st_ino != opened.st_ino
                or not stat.S_ISREG(opened.st_mode)
            ):
                raise FewShotEpisodeError(f"{context} changed while opening")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise FewShotEpisodeError(f"cannot read {context}: {error}") from error
    if (
        opened.st_dev != after.st_dev
        or opened.st_ino != after.st_ino
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
    ):
        raise FewShotEpisodeError(f"{context} changed while reading")
    return resolved, payload


def _pinned_jsonl(
    path: Path, *, expected_sha256: str, context: str
) -> tuple[Path, bytes, list[dict[str, Any]]]:
    expected = _required_sha256(
        expected_sha256, context=f"caller-pinned {context} SHA-256"
    )
    resolved, payload = _read_stable_bytes(path, context=context)
    if bytes_sha256(payload) != expected:
        raise FewShotEpisodeError(
            f"{context} differs from the caller-pinned SHA-256"
        )
    if not payload.endswith(b"\n"):
        raise FewShotEpisodeError(f"{context} must end with one newline")
    lines = payload.splitlines()
    if not lines or any(not line for line in lines):
        raise FewShotEpisodeError(f"{context} must contain non-blank JSONL rows")
    rows = [
        _decode_json_object(line, context=f"{context} row {line_number}")
        for line_number, line in enumerate(lines, 1)
    ]
    return resolved, payload, rows


def _contains_alias(tokens: tuple[str, ...], alias: tuple[str, ...]) -> bool:
    width = len(alias)
    return any(tokens[index : index + width] == alias for index in range(len(tokens) - width + 1))


def classify_atomic_action_signature(
    signature: Any,
) -> tuple[Optional[AtomicActionMatch], str]:
    """Conservatively map one target signature to one versioned primitive."""

    if type(signature) is not str or _SIGNATURE_RE.fullmatch(signature) is None:
        return None, "invalid_target_action_signature"
    tokens = tuple(signature.split("_"))
    if any(token in _COMPOSITION_TOKENS for token in tokens):
        return None, "composite_target_action_signature"
    matches: list[tuple[str, tuple[str, ...]]] = []
    for primitive, aliases in _ACTION_ALIASES:
        found = tuple(
            "_".join(alias) for alias in aliases if _contains_alias(tokens, alias)
        )
        if found:
            matches.append((primitive, found))
    if not matches:
        return None, "unsupported_target_action_signature"
    if len(matches) != 1:
        return None, "ambiguous_target_action_signature"
    primitive, aliases = matches[0]
    return AtomicActionMatch(primitive, aliases), "eligible"


def _validate_preview_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_iid: dict[str, dict[str, Any]] = {}
    for line_number, raw in enumerate(rows, 1):
        row = dict(raw)
        if set(row) != PREVIEW_FIELDS:
            raise FewShotEpisodeError(
                f"preview row {line_number} fields differ: "
                f"missing={sorted(PREVIEW_FIELDS - set(row))}, "
                f"unknown={sorted(set(row) - PREVIEW_FIELDS)}"
            )
        iid = _required_iid(row.get("iid"), context=f"preview IID at row {line_number}")
        if iid in by_iid:
            raise FewShotEpisodeError(f"duplicate preview IID: {iid}")
        if row.get("schema_version") != PREVIEW_ROW_SCHEMA:
            raise FewShotEpisodeError(f"preview row schema differs: {iid}")
        if (
            row.get("preview_only") is not True
            or row.get("training_authorized") is not False
            or row.get("training_use_forbidden") is not True
            or row.get("production_eligible") is not False
            or row.get("post_video_acceptance") != "pending"
        ):
            raise FewShotEpisodeError(f"preview safety state differs: {iid}")
        digest = _required_sha256(
            row.get("row_digest"), context=f"preview row digest for {iid}"
        )
        unsigned = dict(row)
        unsigned.pop("row_digest")
        if object_sha256(unsigned) != digest:
            raise FewShotEpisodeError(f"preview row digest differs: {iid}")

        _required_text(row.get("group_id"), context=f"group_id for {iid}")
        source_sha = _required_sha256(
            row.get("source_video_sha256"), context=f"source video hash for {iid}"
        )
        _required_sha256(
            row.get("target_video_sha256"), context=f"target video hash for {iid}"
        )
        edit = _required_text(
            row.get("edit_instruction"), context=f"edit instruction for {iid}"
        )
        if hashlib.sha256(edit.encode("utf-8")).hexdigest() != _required_sha256(
            row.get("edit_instruction_sha256"),
            context=f"edit instruction hash for {iid}",
        ):
            raise FewShotEpisodeError(f"edit instruction hash differs: {iid}")
        generation = _required_text(
            row.get("generation_instruction"),
            context=f"generation instruction for {iid}",
        )
        if hashlib.sha256(generation.encode("utf-8")).hexdigest() != _required_sha256(
            row.get("generation_instruction_sha256"),
            context=f"generation instruction hash for {iid}",
        ):
            raise FewShotEpisodeError(f"generation instruction hash differs: {iid}")
        if row.get("instruction_source") not in {"structured", "natural"}:
            raise FewShotEpisodeError(f"instruction source differs: {iid}")

        census = _mapping(row.get("source_census"), context=f"source census for {iid}")
        plan = _mapping(row.get("target_plan"), context=f"target plan for {iid}")
        if census.get("iid") != iid or plan.get("iid") != iid:
            raise FewShotEpisodeError(f"nested preview IID differs: {iid}")
        subjects = _sequence(
            census.get("dynamic_subjects"), context=f"dynamic subjects for {iid}"
        )
        targets = _sequence(
            plan.get("dynamic_subject_targets"),
            context=f"dynamic subject targets for {iid}",
        )
        if any(not isinstance(value, Mapping) for value in subjects + targets):
            raise FewShotEpisodeError(f"subject entries must be objects: {iid}")
        source_camera = _mapping(
            census.get("camera"), context=f"source camera for {iid}"
        )
        target_camera = _mapping(
            plan.get("camera_target"), context=f"target camera for {iid}"
        )
        computed_gates = {
            "single_dynamic_actor": (
                len(subjects) == 1
                and subjects[0].get("dynamic") is True
                and len(targets) == 1
                and targets[0].get("subject_id") == subjects[0].get("subject_id")
            ),
            "source_camera_locked_off": (
                source_camera.get("motion_class") == "locked_off"
            ),
            "target_camera_locked_off": (
                target_camera.get("motion_class") == "locked_off"
            ),
            "target_camera_preserve_static": (
                target_camera.get("relation") == "preserve_static"
            ),
            "source_census_high_confidence": census.get("confidence") == "high",
            "target_plan_high_confidence": plan.get("confidence") == "high",
        }
        gates = _mapping(
            row.get("selection_gates"), context=f"selection gates for {iid}"
        )
        if set(gates) != set(REQUIRED_SELECTION_GATES) or any(
            type(value) is not bool for value in gates.values()
        ):
            raise FewShotEpisodeError(f"selection gate schema differs: {iid}")
        if gates != computed_gates:
            raise FewShotEpisodeError(
                f"selection gates disagree with nested source/target semantics: {iid}"
            )
        # Bind the exact source hash now; it is the leakage identity later.
        if source_sha != row["source_video_sha256"]:
            raise AssertionError("source hash normalization changed")
        by_iid[iid] = row
    if not by_iid:
        raise FewShotEpisodeError("preview manifest is empty")
    return by_iid


def _positive_int_list(value: Any, *, context: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        type(item) is not int or item <= 0 for item in value
    ):
        raise FewShotEpisodeError(f"{context} must be a non-empty positive-int list")
    return list(value)


def _validate_vae_index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_iid: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    ordered_iids: list[str] = []
    for line_number, raw in enumerate(rows, 1):
        row = dict(raw)
        if set(row) != VAE_INDEX_FIELDS:
            raise FewShotEpisodeError(
                f"VAE index row {line_number} fields differ: "
                f"missing={sorted(VAE_INDEX_FIELDS - set(row))}, "
                f"unknown={sorted(set(row) - VAE_INDEX_FIELDS)}"
            )
        iid = _required_iid(row.get("iid"), context=f"VAE IID at row {line_number}")
        if iid in by_iid:
            raise FewShotEpisodeError(f"duplicate VAE index IID: {iid}")
        if row.get("schema_version") != VAE_INDEX_ROW_SCHEMA:
            raise FewShotEpisodeError(f"VAE index row schema differs: {iid}")
        if (
            row.get("preview_only") is not True
            or row.get("production_claim_forbidden") is not True
        ):
            raise FewShotEpisodeError(f"VAE index safety state differs: {iid}")
        parquet_path = _required_text(
            row.get("parquet_path"), context=f"VAE parquet path for {iid}"
        )
        receipt_path = _required_text(
            row.get("sample_receipt_path"), context=f"VAE receipt path for {iid}"
        )
        if parquet_path in paths or receipt_path in paths:
            raise FewShotEpisodeError(f"duplicate VAE index artifact path: {iid}")
        paths.update((parquet_path, receipt_path))
        for field in (
            "parquet_sha256",
            "materialized_row_digest",
            "sample_receipt_sha256",
        ):
            _required_sha256(row.get(field), context=f"{field} for {iid}")
        bucket = _positive_int_list(row.get("bucket_hw"), context=f"bucket_hw for {iid}")
        if len(bucket) != 2:
            raise FewShotEpisodeError(f"bucket_hw must have two dimensions: {iid}")
        _positive_int_list(
            row.get("posterior_parameters_shape"),
            context=f"posterior_parameters_shape for {iid}",
        )
        by_iid[iid] = row
        ordered_iids.append(iid)
    if not by_iid:
        raise FewShotEpisodeError("VAE index is empty")
    if ordered_iids != sorted(ordered_iids):
        raise FewShotEpisodeError("VAE index IIDs must be in canonical sorted order")
    return by_iid


def _primary_gate_exclusion(gates: Mapping[str, bool]) -> Optional[str]:
    checks = (
        ("single_dynamic_actor", "single_dynamic_actor_not_true"),
        ("source_camera_locked_off", "source_camera_not_locked_off"),
        ("target_camera_locked_off", "target_camera_not_locked_off"),
        ("target_camera_preserve_static", "target_camera_not_preserve_static"),
        ("source_census_high_confidence", "source_census_not_high_confidence"),
        ("target_plan_high_confidence", "target_plan_not_high_confidence"),
    )
    for gate, reason in checks:
        if gates[gate] is not True:
            return reason
    return None


def _candidate_from_row(
    row: Mapping[str, Any], vae_row: Mapping[str, Any]
) -> tuple[Optional[Candidate], str]:
    gate_reason = _primary_gate_exclusion(row["selection_gates"])
    if gate_reason is not None:
        return None, gate_reason
    iid = str(row["iid"])
    census = row["source_census"]
    plan = row["target_plan"]
    subjects = census["dynamic_subjects"]
    targets = plan["dynamic_subject_targets"]
    subject = subjects[0]
    target = targets[0]
    if (
        census.get("all_dynamic_subjects_enumerated") is not True
        or census.get("crowd_or_unresolved_motion") is not False
        or type(subject.get("stable_reference")) is not str
        or not subject["stable_reference"].strip()
        or "\x00" in subject["stable_reference"]
        or type(subject.get("subject_id")) is not str
        or not subject["subject_id"]
        or target.get("subject_id") != subject.get("subject_id")
        or target.get("substantive_change") is not True
        or type(target.get("target_motion")) is not str
        or not target["target_motion"].strip()
    ):
        return None, "subject_target_not_unambiguous"
    match, reason = classify_atomic_action_signature(
        target.get("target_action_signature")
    )
    if match is None:
        return None, reason
    return (
        Candidate(
            iid=iid,
            group_id=str(row["group_id"]),
            source_video_sha256=str(row["source_video_sha256"]),
            target_video_sha256=str(row["target_video_sha256"]),
            edit_instruction=str(row["edit_instruction"]),
            edit_instruction_sha256=str(row["edit_instruction_sha256"]),
            target_action_signature=str(target["target_action_signature"]),
            primitive_id=match.primitive_id,
            matched_aliases=match.matched_aliases,
            subject_id=str(subject["subject_id"]),
            stable_reference=" ".join(str(subject["stable_reference"]).split()),
            preview_row_digest=str(row["row_digest"]),
            vae_index_row=dict(vae_row),
        ),
        "eligible",
    )


def _rank(seed: int, namespace: str, *values: str) -> tuple[str, ...]:
    payload = "\0".join((str(seed), namespace, *values)).encode("utf-8")
    return (hashlib.sha256(payload).hexdigest(), *values)


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _deduplicate_leak_components(
    candidates: Sequence[Candidate], *, seed: int
) -> tuple[list[Candidate], int]:
    """Choose one representative per group/source connected component."""

    disjoint = _DisjointSet(len(candidates))
    by_group: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        for table, key in (
            (by_group, candidate.group_id),
            (by_source, candidate.source_video_sha256),
        ):
            previous = table.get(key)
            if previous is None:
                table[key] = index
            else:
                disjoint.union(index, previous)
    components: dict[int, list[Candidate]] = {}
    for index, candidate in enumerate(candidates):
        components.setdefault(disjoint.find(index), []).append(candidate)
    selected = [
        min(
            values,
            key=lambda item: _rank(
                seed,
                "leak-component-representative",
                item.primitive_id,
                item.iid,
                item.group_id,
                item.source_video_sha256,
            ),
        )
        for values in components.values()
    ]
    selected.sort(key=lambda item: (item.primitive_id, item.iid))
    return selected, len(candidates) - len(selected)


def _member(candidate: Candidate, *, role: str, ordinal: Optional[int] = None) -> dict[str, Any]:
    vae = candidate.vae_index_row
    result: dict[str, Any] = {
        "iid": candidate.iid,
        "group_id": candidate.group_id,
        "source_video_sha256": candidate.source_video_sha256,
        "target_video_sha256": candidate.target_video_sha256,
        "edit_instruction": candidate.edit_instruction,
        "edit_instruction_sha256": candidate.edit_instruction_sha256,
        "target_action_signature": candidate.target_action_signature,
        "primitive_id": candidate.primitive_id,
        "ontology_matched_aliases": list(candidate.matched_aliases),
        "subject_id": candidate.subject_id,
        "stable_reference": candidate.stable_reference,
        "preview_row_digest": candidate.preview_row_digest,
        "role": role,
        "target_post_video_acceptance": "pending",
        "vae": {
            "index_row_digest": object_sha256(vae),
            "parquet_path": vae["parquet_path"],
            "parquet_sha256": vae["parquet_sha256"],
            "materialized_row_digest": vae["materialized_row_digest"],
            "bucket_hw": vae["bucket_hw"],
            "posterior_parameters_shape": vae["posterior_parameters_shape"],
            "sample_receipt_path": vae["sample_receipt_path"],
            "sample_receipt_sha256": vae["sample_receipt_sha256"],
        },
    }
    if ordinal is not None:
        result["support_ordinal"] = ordinal
    return result


def _phase_permutation(seed: int, episode_id: str) -> list[int]:
    nonboundary = sorted(
        range(1, LATENT_PHASE_COUNT),
        key=lambda phase: _rank(
            seed, "phase-shuffle", episode_id, f"{phase:02d}"
        ),
    )
    identity = list(range(1, LATENT_PHASE_COUNT))
    reverse = list(reversed(identity))
    if nonboundary == identity or nonboundary == reverse:
        nonboundary = nonboundary[1:] + nonboundary[:1]
    return [0, *nonboundary]


def _phase_controls(seed: int, episode_id: str) -> list[dict[str, Any]]:
    identity = list(range(LATENT_PHASE_COUNT))
    reverse = [0, *range(LATENT_PHASE_COUNT - 1, 0, -1)]
    return [
        {
            "label": "correct",
            "support_set": "positive",
            "phase_permutation": identity,
            "carrier_sign": 1,
        },
        {
            "label": "reverse_nonboundary",
            "support_set": "positive",
            "phase_permutation": reverse,
            "carrier_sign": 1,
        },
        {
            "label": "shuffle_nonboundary",
            "support_set": "positive",
            "phase_permutation": _phase_permutation(seed, episode_id),
            "carrier_sign": 1,
        },
        {
            "label": "negate",
            "support_set": "positive",
            "phase_permutation": identity,
            "carrier_sign": -1,
        },
        {
            "label": "wrong_action_support",
            "support_set": "negative_action",
            "phase_permutation": identity,
            "carrier_sign": 1,
        },
    ]


def _identity_set(candidates: Iterable[Candidate], field: str) -> set[str]:
    return {str(getattr(candidate, field)) for candidate in candidates}


def _assert_disjoint_splits(
    supports: Sequence[Candidate],
    train_queries: Sequence[Candidate],
    heldout_queries: Sequence[Candidate],
) -> None:
    roles = {
        "support": supports,
        "train_query": train_queries,
        "heldout_query": heldout_queries,
    }
    for field in ("iid", "group_id", "source_video_sha256"):
        sets = {name: _identity_set(values, field) for name, values in roles.items()}
        pairs = (
            ("support", "train_query"),
            ("support", "heldout_query"),
            ("train_query", "heldout_query"),
        )
        for left, right in pairs:
            overlap = sets[left] & sets[right]
            if overlap:
                raise FewShotEpisodeError(
                    f"{field} leakage between {left} and {right}: {sorted(overlap)}"
                )


def _validate_k_seed(k_shot: Any, seed: Any) -> tuple[int, int]:
    if type(k_shot) is not int or k_shot <= 0:
        raise FewShotEpisodeError("k_shot must be a positive integer")
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise FewShotEpisodeError("seed must be an integer in [0, 2**63)")
    return k_shot, seed


def build_fewshot_episode_payloads(
    *,
    preview_manifest: Path,
    expected_preview_manifest_sha256: str,
    vae_index: Path,
    expected_vae_index_sha256: str,
    output_jsonl: Path,
    k_shot: int = DEFAULT_K_SHOT,
    seed: int = DEFAULT_SEED,
) -> EpisodeBuildPayload:
    """Validate inputs and return canonical output bytes without writing files."""

    k_shot, seed = _validate_k_seed(k_shot, seed)
    minimum_groups = max(4, k_shot + 2)
    preview_path, preview_payload, preview_rows = _pinned_jsonl(
        preview_manifest,
        expected_sha256=expected_preview_manifest_sha256,
        context="authoritative preview manifest",
    )
    index_path, index_payload, index_rows = _pinned_jsonl(
        vae_index,
        expected_sha256=expected_vae_index_sha256,
        context="VAE index",
    )
    preview_by_iid = _validate_preview_rows(preview_rows)
    index_by_iid = _validate_vae_index_rows(index_rows)
    if set(preview_by_iid) != set(index_by_iid):
        preview_only = sorted(set(preview_by_iid) - set(index_by_iid))
        index_only = sorted(set(index_by_iid) - set(preview_by_iid))
        raise FewShotEpisodeError(
            "preview/VAE IID membership differs: "
            f"preview_only={preview_only[:8]} index_only={index_only[:8]}"
        )

    exclusion_counts: dict[str, int] = {}
    candidates: list[Candidate] = []
    for iid in sorted(preview_by_iid):
        candidate, reason = _candidate_from_row(
            preview_by_iid[iid], index_by_iid[iid]
        )
        if candidate is None:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
        else:
            candidates.append(candidate)
    deduplicated, leak_duplicates = _deduplicate_leak_components(
        candidates, seed=seed
    )
    if leak_duplicates:
        exclusion_counts["group_or_source_leak_component_duplicate"] = leak_duplicates

    by_primitive: dict[str, list[Candidate]] = {}
    for candidate in deduplicated:
        by_primitive.setdefault(candidate.primitive_id, []).append(candidate)
    primitive_counts = {
        primitive: len(values) for primitive, values in sorted(by_primitive.items())
    }
    viable = {
        primitive: values
        for primitive, values in by_primitive.items()
        if len(values) >= minimum_groups
    }
    if len(viable) < 2:
        raise FewShotEpisodeError(
            "at least two target-action primitives need "
            f"{minimum_groups} independent group/source units for K={k_shot}; "
            f"observed={primitive_counts}"
        )

    assignments: dict[str, dict[str, Any]] = {}
    for primitive in sorted(viable):
        ordered = sorted(
            viable[primitive],
            key=lambda item: _rank(
                seed,
                "primitive-membership",
                primitive,
                item.iid,
                item.group_id,
                item.source_video_sha256,
            ),
        )
        assignments[primitive] = {
            "supports": ordered[:k_shot],
            "train_query": ordered[k_shot],
            "heldout_query": ordered[k_shot + 1],
            "unassigned_count": len(ordered) - (k_shot + 2),
        }

    support_pool = [
        candidate
        for primitive in sorted(assignments)
        for candidate in assignments[primitive]["supports"]
    ]
    train_queries = [
        assignments[primitive]["train_query"] for primitive in sorted(assignments)
    ]
    heldout_queries = [
        assignments[primitive]["heldout_query"] for primitive in sorted(assignments)
    ]
    _assert_disjoint_splits(support_pool, train_queries, heldout_queries)

    episode_rows: list[dict[str, Any]] = []
    primitives = sorted(assignments)
    for primitive in primitives:
        alternatives = [value for value in primitives if value != primitive]
        negative_primitive = min(
            alternatives,
            key=lambda value: _rank(seed, "negative-primitive", primitive, value),
        )
        assignment = assignments[primitive]
        negative_supports = assignments[negative_primitive]["supports"]
        identity = {
            "ontology_sha256": ONTOLOGY_SHA256,
            "primitive_id": primitive,
            "positive_support_iids": [item.iid for item in assignment["supports"]],
            "train_query_iid": assignment["train_query"].iid,
            "heldout_query_iid": assignment["heldout_query"].iid,
            "negative_primitive_id": negative_primitive,
            "negative_support_iids": [item.iid for item in negative_supports],
            "k_shot": k_shot,
            "seed": seed,
        }
        episode_id = f"{primitive}-{object_sha256(identity)[:16]}"
        row = {
            "schema_version": EPISODE_ROW_SCHEMA,
            "episode_id": episode_id,
            "primitive_id": primitive,
            "ontology": {
                "schema_version": ONTOLOGY_SCHEMA,
                "sha256": ONTOLOGY_SHA256,
                "semantic_source": ONTOLOGY_SPEC["semantic_source"],
                "source_family_used": False,
            },
            "k_shot": k_shot,
            "seed": seed,
            "positive_supports": [
                _member(item, role="positive_support", ordinal=index)
                for index, item in enumerate(assignment["supports"])
            ],
            "train_query": _member(
                assignment["train_query"], role="train_query"
            ),
            "heldout_query": _member(
                assignment["heldout_query"], role="heldout_query"
            ),
            "negative_action": {
                "primitive_id": negative_primitive,
                "supports": [
                    _member(item, role="negative_action_support", ordinal=index)
                    for index, item in enumerate(negative_supports)
                ],
            },
            "phase0_policy": "fixed_zero_residual_boundary",
            "phase_controls": _phase_controls(seed, episode_id),
            "experimental_only": True,
            "preview_only": True,
            "training_authorized": False,
            "production_claim": False,
            "manual_review_performed": False,
            "target_post_video_acceptance": "pending",
        }
        row["episode_digest"] = object_sha256(row)
        episode_rows.append(row)

    jsonl_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in episode_rows
    )
    output = output_jsonl.expanduser().absolute()
    receipt_path = Path(f"{output}.receipt.json")
    if len({preview_path, index_path, output, receipt_path}) != 4:
        raise FewShotEpisodeError("input/output paths must be distinct")

    split_value = {
        "support_iids": sorted(item.iid for item in support_pool),
        "train_query_iids": sorted(item.iid for item in train_queries),
        "heldout_query_iids": sorted(item.iid for item in heldout_queries),
        "support_group_ids": sorted(item.group_id for item in support_pool),
        "train_query_group_ids": sorted(item.group_id for item in train_queries),
        "heldout_query_group_ids": sorted(item.group_id for item in heldout_queries),
        "support_source_sha256": sorted(
            item.source_video_sha256 for item in support_pool
        ),
        "train_query_source_sha256": sorted(
            item.source_video_sha256 for item in train_queries
        ),
        "heldout_query_source_sha256": sorted(
            item.source_video_sha256 for item in heldout_queries
        ),
    }
    receipt: dict[str, Any] = {
        "schema_version": EPISODE_RECEIPT_SCHEMA,
        "complete": True,
        "experimental_only": True,
        "preview_only": True,
        "training_authorized": False,
        "training_use_forbidden": True,
        "production_claim": False,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "manual_review_performed": False,
        "human_review_status": "not_performed",
        "target_post_video_acceptance": "pending",
        "target_video_quality_verified": False,
        "episode_schema_version": EPISODE_ROW_SCHEMA,
        "ontology": {
            "schema_version": ONTOLOGY_SCHEMA,
            "sha256": ONTOLOGY_SHA256,
            "spec": ONTOLOGY_SPEC,
        },
        "source_family_field_used": False,
        "preview_manifest_path": str(preview_path),
        "preview_manifest_sha256": bytes_sha256(preview_payload),
        "preview_manifest_rows": len(preview_rows),
        "vae_index_path": str(index_path),
        "vae_index_sha256": bytes_sha256(index_payload),
        "vae_index_rows": len(index_rows),
        "join_policy": "exact_equal_iid_membership",
        "k_shot": k_shot,
        "seed": seed,
        "minimum_independent_groups_per_primitive": minimum_groups,
        "eligible_before_leak_dedup": len(candidates),
        "eligible_after_leak_dedup": len(deduplicated),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "primitive_independent_group_counts": primitive_counts,
        "retained_primitives": sorted(assignments),
        "episode_count": len(episode_rows),
        "unassigned_eligible_rows": sum(
            int(assignment["unassigned_count"])
            for assignment in assignments.values()
        ),
        "split_contract": {
            "positive_support_count_per_episode": k_shot,
            "negative_support_count_per_episode": k_shot,
            "train_query_count_per_episode": 1,
            "heldout_query_count_per_episode": 1,
            "iid_disjoint": True,
            "group_id_disjoint": True,
            "source_video_sha256_disjoint": True,
            "negative_support_policy": (
                "reuse_positive_support_pool_of_different_primitive_only"
            ),
            "heldout_never_used_as_support": True,
            "split_digest": object_sha256(split_value),
        },
        "phase_control_labels": [
            "correct",
            "reverse_nonboundary",
            "shuffle_nonboundary",
            "negate",
            "wrong_action_support",
        ],
        "latent_phase_count": LATENT_PHASE_COUNT,
        "output_jsonl_path": str(output),
        "output_jsonl_sha256": bytes_sha256(jsonl_bytes),
        "output_jsonl_bytes": len(jsonl_bytes),
        "output_jsonl_lines": len(episode_rows),
        "episode_rows_digest": object_sha256(episode_rows),
        "receipt_path": str(receipt_path),
        "publication_contract": (
            "programmatic_dry_run_cli_opt_in_create_only_receipt_ready_marker_last"
        ),
    }
    receipt["receipt_digest"] = object_sha256(receipt)
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    return EpisodeBuildPayload(
        jsonl_bytes=jsonl_bytes,
        receipt_bytes=receipt_bytes,
        receipt=receipt,
        output_jsonl=output,
        output_receipt=receipt_path,
    )


def build_fewshot_action_episodes(
    *,
    preview_manifest: Path,
    expected_preview_manifest_sha256: str,
    vae_index: Path,
    expected_vae_index_sha256: str,
    output_jsonl: Path,
    k_shot: int = DEFAULT_K_SHOT,
    seed: int = DEFAULT_SEED,
) -> EpisodeBuildPayload:
    """Named public entry point; like the payload builder, this never writes."""

    return build_fewshot_episode_payloads(
        preview_manifest=preview_manifest,
        expected_preview_manifest_sha256=expected_preview_manifest_sha256,
        vae_index=vae_index,
        expected_vae_index_sha256=expected_vae_index_sha256,
        output_jsonl=output_jsonl,
        k_shot=k_shot,
        seed=seed,
    )


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_create_only(payload: EpisodeBuildPayload) -> None:
    """Publish JSONL then its receipt without replacing existing paths."""

    output = payload.output_jsonl
    receipt = payload.output_receipt
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_mode = output.parent.lstat().st_mode
    except OSError as error:
        raise FewShotEpisodeError(f"cannot inspect output directory: {error}") from error
    if not stat.S_ISDIR(parent_mode) or output.parent.is_symlink():
        raise FewShotEpisodeError("output parent must be a plain non-symlink directory")
    for destination in (output, receipt):
        if destination.exists() or destination.is_symlink():
            raise FewShotEpisodeError(f"create-only output exists: {destination}")
    published: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.staging.", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        staged = (
            (staging / output.name, output, payload.jsonl_bytes),
            # Receipt is the ready marker and must appear last.
            (staging / receipt.name, receipt, payload.receipt_bytes),
        )
        try:
            for source, _destination, contents in staged:
                _write_fsynced(source, contents)
            for source, destination, _contents in staged:
                try:
                    os.link(source, destination)
                except FileExistsError as error:
                    raise FewShotEpisodeError(
                        f"create-only output appeared during publication: {destination}"
                    ) from error
                published.append(destination)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            for destination in reversed(published):
                destination.unlink(missing_ok=True)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-manifest", type=Path, required=True)
    parser.add_argument("--expected-preview-manifest-sha256", required=True)
    parser.add_argument("--vae-index", type=Path, required=True)
    parser.add_argument("--expected-vae-index-sha256", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--k-shot", type=int, default=DEFAULT_K_SHOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish create-only JSONL and receipt; otherwise perform a dry run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_fewshot_episode_payloads(
            preview_manifest=args.preview_manifest,
            expected_preview_manifest_sha256=args.expected_preview_manifest_sha256,
            vae_index=args.vae_index,
            expected_vae_index_sha256=args.expected_vae_index_sha256,
            output_jsonl=args.output_jsonl,
            k_shot=args.k_shot,
            seed=args.seed,
        )
        if args.publish:
            publish_create_only(payload)
    except FewShotEpisodeError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    status = {
        "dry_run": not args.publish,
        "published": bool(args.publish),
        "output_jsonl_path": str(payload.output_jsonl),
        "receipt_path": str(payload.output_receipt),
        "output_jsonl_sha256": payload.receipt["output_jsonl_sha256"],
        "receipt_digest": payload.receipt["receipt_digest"],
        "episode_count": payload.receipt["episode_count"],
        "experimental_only": True,
        "training_authorized": False,
        "production_claim": False,
    }
    print(canonical_json_bytes(status).decode("utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

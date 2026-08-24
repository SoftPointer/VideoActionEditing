"""Qwen planner for one atomic first-frame-conditioned motion-edit event.

This is a new lineage; it does not alter or overwrite the v16 planner.  It
reuses v16 source-video preparation, source-motion census, closed validators,
instruction compiler, and persistent strided-worker execution pattern.  Its
target call is stricter: Qwen must return one explicit causal-event graph plus
the per-subject v16 target plan used to condition Wan.

Multiple people, tools, manipulated objects, and physical effects may be
participants in the same event.  Two independently instructable actions are
not an atomic event, even if they are merely put in sequence or happen at the
same time.  The internal target plan may retain frame-level trajectories for
video generation.  That compiled v16 instruction is a *Wan generation prompt*,
never the training label.  ``goku_atomic_motion_instruction`` owns the final
single-sentence ``atomic_action_instruction`` sidecar; camera and preservation
remain separate metadata there.

Successful rows deliberately publish the unchanged
``motive-goku-full-motion-qwen-v16-passed-v1`` schema, so existing Wan adapters
can consume ``passed/<iid>.jsonl`` without modification.  Atomic prompt and
graph provenance live in a new ``rows/<iid>/result.json`` schema and a new
terminal receipt schema.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

from . import goku_full_motion_qwen_v16 as v16
from .goku_action_anchor_qwen import (
    _reject_backend_cpu_or_disk_offload,
    validate_input_row,
)
from .qwen_filter import LocalQwenBackend, _file_digest


ATOMIC_TARGET_RESPONSE_SCHEMA = "motive-goku-atomic-motion-qwen-target-response-v1"
ATOMIC_EVENT_SCHEMA = "motive-goku-atomic-motion-qwen-event-v1"
ATOMIC_PARTICIPANT_SCHEMA = "motive-goku-atomic-motion-qwen-participant-v1"
ATOMIC_CAUSAL_EDGE_SCHEMA = "motive-goku-atomic-motion-qwen-causal-edge-v1"
PLANNER_PROVENANCE_SCHEMA = "motive-goku-atomic-motion-qwen-provenance-v1"
RECORD_SCHEMA = "motive-goku-atomic-motion-qwen-record-v1"
ROW_RECEIPT_SCHEMA = "motive-goku-atomic-motion-qwen-row-receipt-v1"
SOURCE_CENSUS_CACHE_AUDIT_SCHEMA = (
    "motive-goku-atomic-motion-qwen-source-census-cache-audit-v1"
)

SOURCE_CENSUS_SCHEMA = v16.SOURCE_CENSUS_SCHEMA
TARGET_PLAN_SCHEMA = v16.TARGET_PLAN_SCHEMA
PASSED_SCHEMA = v16.PASSED_SCHEMA

DEFAULT_MAX_NEW_TOKENS = v16.DEFAULT_MAX_NEW_TOKENS
DEFAULT_NFRAMES = v16.DEFAULT_NFRAMES
DEFAULT_MAX_PIXELS = v16.DEFAULT_MAX_PIXELS
DEFAULT_TILE_WIDTH = v16.DEFAULT_TILE_WIDTH
DEFAULT_MOSAIC_COLUMNS = v16.DEFAULT_MOSAIC_COLUMNS

_ROLE_VALUES = frozenset({"agent", "co_agent", "tool", "patient", "effect"})
_EDGE_RELATIONS = frozenset(
    {
        "acts_on",
        "causes",
        "controls",
        "emits",
        "jointly_coordinates_with",
        "moves",
        "reacts_to",
        "receives",
        "supports",
        "uses",
    }
)
_NON_ATOMIC_STITCH_RE = re.compile(
    r"\b(?:then|next|afterward|afterwards|subsequently|finally|meanwhile|"
    r"followed\s+by|before\s+that|after\s+that|in\s+parallel|"
    r"separate(?:ly)?|independent(?:ly)?)\b",
    re.IGNORECASE,
)
_PRIVATE_TIMING_RE = re.compile(
    r"\b(?:frames?|frame\s+index|fps|seconds?|secs?|milliseconds?|ms|"
    r"phase\s*\d+|stage\s*\d+)\b",
    re.IGNORECASE,
)


class GokuAtomicMotionQwenError(v16.GokuFullMotionQwenV16Error):
    """Fail-closed atomic-planner contract error."""


class GokuAtomicMotionQwenStageError(GokuAtomicMotionQwenError):
    """One model stage failed after its allowed mechanical retry."""

    def __init__(self, stage: str, attempts: Sequence[Mapping[str, Any]]) -> None:
        self.stage = stage
        self.attempts = [dict(item) for item in attempts]
        message = self.attempts[-1].get("error") if self.attempts else "unknown"
        super().__init__(f"{stage} failed: {message}")


class AtomicEventRejected(GokuAtomicMotionQwenError):
    """The target response does not encode one connected causal event."""


# Public compatibility helpers used by launch/finalization code.
object_sha256 = v16.object_sha256
_iter_jsonl = v16._iter_jsonl
prepare_row = v16.prepare_row
validate_passed_row = v16.validate_passed_row


def _require_closed(
    value: Mapping[str, Any], required: set[str], *, context: str
) -> None:
    v16._require_closed(value, required, context=context)


def _text(value: Any, *, context: str, maximum: int = 1000) -> str:
    return v16._text(value, context=context, maximum=maximum)


def _prompt_digest() -> str:
    return v16._sha256_bytes(
        (ATOMIC_TARGET_SYSTEM + "\n" + ATOMIC_TARGET_RESPONSE_SCHEMA).encode("utf-8")
    )


def _provenance() -> dict[str, Any]:
    return {
        "schema_version": PLANNER_PROVENANCE_SCHEMA,
        "planner_module": "motive.goku_atomic_motion_qwen",
        "planner_version": "atomic-qwen-v1",
        "source_census_contract": SOURCE_CENSUS_SCHEMA,
        "internal_target_plan_contract": TARGET_PLAN_SCHEMA,
        "wan_passed_contract": PASSED_SCHEMA,
        "atomic_target_response_contract": ATOMIC_TARGET_RESPONSE_SCHEMA,
        "atomic_prompt_contract_sha256": _prompt_digest(),
        "wan_generation_prompt_field": "passed.edit_instruction",
        "wan_generation_prompt_is_training_label": False,
        "training_label_owner": (
            "motive.goku_atomic_motion_instruction.atomic_action_instruction"
        ),
        "camera_and_preservation_are_label_metadata": True,
        "optional_source_census_cache_is_hash_bound": True,
    }


def _target_example(source_census: Mapping[str, Any]) -> dict[str, Any]:
    source_ids = [item["subject_id"] for item in source_census["dynamic_subjects"]]
    targets = [
        {
            "schema_version": v16.TARGET_SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "target_action_signature": "event_specific_subject_motion",
            "target_motion": (
                "complete absolute I0-grounded trajectory for this participant "
                "within the single focal event"
            ),
            "substantive_change": True,
        }
        for subject_id in source_ids
    ]
    participants = [
        {
            "schema_version": ATOMIC_PARTICIPANT_SCHEMA,
            "subject_id": subject_id,
            "role": "agent" if index == 0 else "co_agent",
            "event_contribution": "contribution to the same focal event",
            "target_action_signature": targets[index]["target_action_signature"],
        }
        for index, subject_id in enumerate(source_ids)
    ]
    edges = [
        {
            "schema_version": ATOMIC_CAUSAL_EDGE_SCHEMA,
            "from_subject_id": source_ids[index - 1],
            "to_subject_id": source_ids[index],
            "relation": "jointly_coordinates_with",
        }
        for index in range(1, len(source_ids))
    ]
    source_camera_class = source_census["camera"]["motion_class"]
    camera_relation = (
        "preserve_static" if source_camera_class == "locked_off" else "replace_motion"
    )
    return {
        "schema_version": ATOMIC_TARGET_RESPONSE_SCHEMA,
        "iid": source_census["iid"],
        "atomic_event": {
            "schema_version": ATOMIC_EVENT_SCHEMA,
            "event_id": "event_01",
            "event_action_signature": "one_focal_causal_action",
            "event_summary": "one concise causal action involving all participants",
            "participants": participants,
            "causal_edges": edges,
            "independent_event_count": 1,
            "single_causal_event": True,
            "all_dynamic_subjects_in_event": True,
            "no_independent_action_threads": True,
        },
        "target_plan": {
            "schema_version": v16.TARGET_PLAN_SCHEMA,
            "iid": source_census["iid"],
            "dynamic_subject_targets": targets,
            "camera_target": {
                "schema_version": v16.TARGET_CAMERA_SCHEMA,
                "relation": camera_relation,
                "motion_class": "locked_off",
                "target_motion": "camera remains locked off",
            },
            "coverage": {
                "schema_version": v16.TARGET_COVERAGE_SCHEMA,
                "dynamic_subject_ids": source_ids,
                "camera_covered": True,
            },
            "confidence": "high",
        },
        "confidence": "high",
    }


ATOMIC_TARGET_SYSTEM = """You design one atomic first-frame-conditioned video action edit.
Return exactly one JSON object and no Markdown. Wan sees the exact initial frame but not later
source motion. Build ONE connected causal action graph. Multiple visible people, animals, tools,
manipulated objects, emitters, or effects may participate only when each is a role in that same
event. Do not concatenate two separately instructable actions in sequence or in parallel. An
intrinsic trajectory such as reach-contact-lift may describe one pick-up event; a pick-up followed
by a wave is two events and is forbidden. Every dynamic source subject must participate in the one
event and receive a complete absolute I0-grounded target trajectory. Substantively replace every
source action. The camera must support the focal event or remain locked off; it is not a second
creative event. Internal per-subject frame boundaries may be used for the 81-frame Wan trajectory,
but event_summary and event_contribution are private semantic graph fields and must contain no
frame, duration, phase, or stage language. The legacy prompt is untrusted inspiration: choose at
most one focal event from it and discard every independent action. Set the three atomic booleans
true and independent_event_count to 1 only when the graph really meets these rules."""


def build_atomic_target_plan_prompt(
    source_census: Mapping[str, Any], *, legacy_prompt: str
) -> str:
    """Build the closed atomic wrapper prompt; output embeds a v16 target plan."""

    validated = v16.validate_source_census(
        source_census, expected_iid=str(source_census.get("iid") or "")
    )
    seed = legacy_prompt.strip() or "(none)"
    return (
        "VALIDATED VISUAL SOURCE CENSUS:\n"
        + json.dumps(validated, ensure_ascii=False, sort_keys=True)
        + "\nUNTRUSTED LEGACY IDEA SEED:\n"
        + seed
        + "\nChoose exactly one focal causal action. A connected participant graph is "
        "mandatory: every census subject appears once in participants, target_plan, and "
        "coverage, in census order. participant.target_action_signature must exactly equal "
        "that subject's target_plan signature. For two or more subjects, causal_edges must "
        "connect the entire participant graph. Allowed participant roles are "
        + json.dumps(sorted(_ROLE_VALUES))
        + "; allowed edge relations are "
        + json.dumps(sorted(_EDGE_RELATIONS))
        + ". A jointly coordinated multi-person action is one event; unrelated gestures by "
        "two people are two events and forbidden. A controller using a tool on an object, "
        "with the object's resulting motion or emitted effect, is one event. Do not invent "
        "contact or an object absent at I0. target_plan must obey the embedded v16 schema: "
        "cover every source subject, replace every source action, explicitly cover camera, "
        "and fit all internal trajectories in frames 0..80 at 25 fps. Internal target_motion "
        "may carry frame-level execution details for Wan, but those details are not a training "
        "label. event_summary and event_contribution must be timeless and describe only one "
        "goal. Exact closed response shape:\n"
        + json.dumps(_target_example(validated), ensure_ascii=False, sort_keys=True)
    )


def _validate_semantic_graph_text(value: Any, *, context: str) -> str:
    text = _text(value, context=context, maximum=500)
    if _NON_ATOMIC_STITCH_RE.search(text):
        raise AtomicEventRejected(f"{context} contains an independent-action cue")
    if _PRIVATE_TIMING_RE.search(text):
        raise AtomicEventRejected(f"{context} leaks private frame/time staging")
    return text


def _connected_subject_graph(
    subject_ids: Sequence[str], edges: Sequence[Mapping[str, Any]]
) -> bool:
    if len(subject_ids) <= 1:
        return not edges
    adjacency = {subject_id: set() for subject_id in subject_ids}
    for edge in edges:
        source = str(edge["from_subject_id"])
        target = str(edge["to_subject_id"])
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited: set[str] = set()
    pending = [subject_ids[0]]
    while pending:
        subject_id = pending.pop()
        if subject_id in visited:
            continue
        visited.add(subject_id)
        pending.extend(adjacency[subject_id] - visited)
    return visited == set(subject_ids)


def _canonicalize_causal_edge_endpoints(
    edge: dict[str, Any], *, context: str, repairs: list[str]
) -> None:
    """Normalize only the two common, unambiguous endpoint aliases.

    Qwen occasionally emits JSON keys named ``from`` and ``to`` even though
    their values are the requested subject IDs.  This is presentation noise,
    not a semantic field to infer.  If a canonical key is also present, the
    alias is removable only when its value is exactly identical (including
    type); conflicting values remain a closed-schema failure.
    """

    for alias, canonical in (
        ("from", "from_subject_id"),
        ("to", "to_subject_id"),
    ):
        if alias not in edge:
            continue
        if canonical in edge:
            alias_value = edge[alias]
            canonical_value = edge[canonical]
            if (
                type(alias_value) is not type(canonical_value)
                or alias_value != canonical_value
            ):
                raise GokuAtomicMotionQwenError(
                    f"{context} supplies ambiguous {alias!r} and "
                    f"{canonical!r} endpoints"
                )
            del edge[alias]
            repairs.append(
                f"removed_redundant_{context}.{alias}_matching_{canonical}"
            )
            continue
        edge[canonical] = edge.pop(alias)
        repairs.append(f"renamed_{context}.{alias}_to_{canonical}")


def validate_atomic_target_response(
    value: Mapping[str, Any],
    *,
    source_census: Mapping[str, Any],
    repair_operations: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the event wrapper and its Wan-compatible v16 target plan."""

    if not isinstance(value, Mapping):
        raise GokuAtomicMotionQwenError("atomic target response must be an object")
    response = copy.deepcopy(dict(value))
    _require_closed(
        response,
        {"schema_version", "iid", "atomic_event", "target_plan", "confidence"},
        context="atomic_target_response",
    )
    iid = str(source_census["iid"])
    if (
        response["schema_version"] != ATOMIC_TARGET_RESPONSE_SCHEMA
        or response["iid"] != iid
    ):
        raise GokuAtomicMotionQwenError("atomic target response schema or IID differs")
    if response["confidence"] != "high":
        raise AtomicEventRejected("atomic target response confidence is not high")

    source_ids = [
        str(item["subject_id"]) for item in source_census["dynamic_subjects"]
    ]
    target_raw = response["target_plan"]
    if not isinstance(target_raw, Mapping):
        raise GokuAtomicMotionQwenError("atomic target_plan must be an object")
    target_canonical, repair = v16.canonicalize_target_plan(
        target_raw,
        expected_iid=iid,
        source_census=source_census,
    )
    if repair["operations"]:
        # The result should bind the exact canonical plan that is fed to Wan.
        response["target_plan"] = target_canonical
    target_plan = v16.validate_target_plan(
        target_canonical,
        expected_iid=iid,
        source_census=source_census,
    )
    response["target_plan"] = target_plan
    target_by_id = {
        item["subject_id"]: item for item in target_plan["dynamic_subject_targets"]
    }

    event_raw = response["atomic_event"]
    if not isinstance(event_raw, Mapping):
        raise GokuAtomicMotionQwenError("atomic_event must be an object")
    event = dict(event_raw)
    _require_closed(
        event,
        {
            "schema_version",
            "event_id",
            "event_action_signature",
            "event_summary",
            "participants",
            "causal_edges",
            "independent_event_count",
            "single_causal_event",
            "all_dynamic_subjects_in_event",
            "no_independent_action_threads",
        },
        context="atomic_event",
    )
    if event["schema_version"] != ATOMIC_EVENT_SCHEMA or event["event_id"] != "event_01":
        raise GokuAtomicMotionQwenError("atomic event schema or event_id differs")
    event["event_action_signature"] = v16._signature(
        event["event_action_signature"], context="atomic_event.event_action_signature"
    )
    event["event_summary"] = _validate_semantic_graph_text(
        event["event_summary"], context="atomic_event.event_summary"
    )
    if type(event["independent_event_count"]) is not int:
        raise GokuAtomicMotionQwenError("independent_event_count must be an integer")
    if event["independent_event_count"] != 1:
        raise AtomicEventRejected("independent_event_count is not one")
    for field in (
        "single_causal_event",
        "all_dynamic_subjects_in_event",
        "no_independent_action_threads",
    ):
        if type(event[field]) is not bool:
            raise GokuAtomicMotionQwenError(f"atomic_event.{field} must be boolean")
        if event[field] is not True:
            raise AtomicEventRejected(f"atomic_event.{field} must be true")

    participants_raw = event["participants"]
    if not isinstance(participants_raw, list) or len(participants_raw) != len(source_ids):
        raise AtomicEventRejected("atomic participant count differs from source census")
    participants: list[dict[str, Any]] = []
    for index, item_raw in enumerate(participants_raw):
        if not isinstance(item_raw, Mapping):
            raise GokuAtomicMotionQwenError("atomic participant must be an object")
        item = dict(item_raw)
        _require_closed(
            item,
            {
                "schema_version",
                "subject_id",
                "role",
                "event_contribution",
                "target_action_signature",
            },
            context=f"atomic_event.participants[{index}]",
        )
        subject_id = source_ids[index]
        if (
            item["schema_version"] != ATOMIC_PARTICIPANT_SCHEMA
            or item["subject_id"] != subject_id
        ):
            raise AtomicEventRejected("atomic participant schema, coverage, or order differs")
        if item["role"] not in _ROLE_VALUES:
            raise GokuAtomicMotionQwenError("atomic participant role is invalid")
        item["event_contribution"] = _validate_semantic_graph_text(
            item["event_contribution"],
            context=f"atomic participant {subject_id} contribution",
        )
        expected_signature = target_by_id[subject_id]["target_action_signature"]
        if item["target_action_signature"] != expected_signature:
            raise AtomicEventRejected(
                f"atomic participant {subject_id} target signature differs"
            )
        participants.append(item)

    edges_raw = event["causal_edges"]
    if not isinstance(edges_raw, list):
        raise GokuAtomicMotionQwenError("atomic causal_edges must be a list")
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    edge_repairs: list[str] = []
    for index, edge_raw in enumerate(edges_raw):
        if not isinstance(edge_raw, Mapping):
            raise GokuAtomicMotionQwenError("atomic causal edge must be an object")
        edge = dict(edge_raw)
        _canonicalize_causal_edge_endpoints(
            edge,
            context=f"atomic_event.causal_edges[{index}]",
            repairs=edge_repairs,
        )
        _require_closed(
            edge,
            {"schema_version", "from_subject_id", "to_subject_id", "relation"},
            context=f"atomic_event.causal_edges[{index}]",
        )
        source_id = edge["from_subject_id"]
        target_id = edge["to_subject_id"]
        if edge["schema_version"] != ATOMIC_CAUSAL_EDGE_SCHEMA:
            raise GokuAtomicMotionQwenError("atomic causal edge schema differs")
        if source_id not in source_ids or target_id not in source_ids or source_id == target_id:
            raise AtomicEventRejected("atomic causal edge endpoints are invalid")
        if edge["relation"] not in _EDGE_RELATIONS:
            raise GokuAtomicMotionQwenError("atomic causal edge relation is invalid")
        edge_key = (str(source_id), str(target_id), str(edge["relation"]))
        if edge_key in seen_edges:
            raise GokuAtomicMotionQwenError("duplicate atomic causal edge")
        seen_edges.add(edge_key)
        edges.append(edge)
    if not _connected_subject_graph(source_ids, edges):
        raise AtomicEventRejected("atomic participant causal graph is disconnected")

    event["participants"] = participants
    event["causal_edges"] = edges
    response["atomic_event"] = event
    if repair_operations is not None:
        repair_operations.extend(edge_repairs)
    return response


def _atomic_retryable(error: Exception) -> bool:
    """Allow one corrective model call for planner-contract failures only.

    Generation/runtime exceptions occur outside the validation ``try`` and
    are never retried here.  Every accepted second response still passes the
    unchanged closed validators, so this changes recovery, not semantics.
    """

    return isinstance(
        error,
        (AtomicEventRejected, v16.GokuFullMotionQwenV16Error),
    )


def _atomic_retry_prompt(base_prompt: str, *, error: Exception) -> str:
    """Return bounded feedback appropriate to schema or semantic rejection."""

    if not isinstance(error, AtomicEventRejected) and v16._is_retryable_schema_error(
        error
    ):
        return v16._schema_retry_prompt(
            base_prompt, stage="atomic target response", error=error
        )
    return (
        base_prompt
        + "\nSEMANTIC VALIDATION RETRY (one final attempt only): Your previous "
        "candidate was rejected with this exact validator error: "
        + str(error)[:700]
        + ". Re-inspect the same validated source census and visual evidence, "
        "then construct a corrected candidate. Substantively replace every "
        "source subject's action in both target_action_signature and "
        "target_motion, while keeping all participants in exactly one connected "
        "causal event. Do not weaken, omit, or set around any validator rule. "
        "Emit one bare JSON object matching the exact closed schema."
    )


def _source_census_cache_audit(
    *, cache_root: Path, iid: str, input_row_sha256: str
) -> dict[str, Any]:
    result_path = cache_root / "rows" / iid / "result.json"
    return {
        "schema_version": SOURCE_CENSUS_CACHE_AUDIT_SCHEMA,
        "status": "miss",
        "cache_root": str(cache_root),
        "cache_result_path": str(result_path),
        "cache_result_sha256": None,
        "cached_record_digest": None,
        "input_row_sha256": input_row_sha256,
        "source_video_sha256": None,
        "source_census_sha256": None,
        "cached_source_stage_sha256": None,
        "rejection": "cache result is absent",
    }


def _load_cached_source_census(
    row: Mapping[str, Any], *, cache_root: Path
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load one old v16 census only under complete row/media hash closure.

    Missing or rejected entries are ordinary cache misses: the caller records
    the audit and falls back to a fresh visual census call.  A cache hit must
    bind the exact current input row, IID, and source-video hash, as well as a
    valid self-digested v16 result and a freshly validated source census.
    """

    iid = str(row["iid"])
    input_row_sha256 = object_sha256(row)
    audit = _source_census_cache_audit(
        cache_root=cache_root,
        iid=iid,
        input_row_sha256=input_row_sha256,
    )
    result_path = cache_root / "rows" / iid / "result.json"
    if not result_path.exists():
        return None, audit
    try:
        if result_path.is_symlink() or not result_path.is_file():
            raise GokuAtomicMotionQwenError("cache result is not a regular file")
        record = v16._strict_read_object(result_path)
        audit["cache_result_sha256"] = _file_digest(result_path)
        if record.get("schema_version") != v16.RECORD_SCHEMA:
            raise GokuAtomicMotionQwenError("cache result schema differs from v16")
        if record.get("iid") != iid:
            raise GokuAtomicMotionQwenError("cache result IID differs")
        cached_input = record.get("input_row")
        if not isinstance(cached_input, Mapping):
            raise GokuAtomicMotionQwenError("cache input_row is absent")
        cached_input_sha256 = object_sha256(cached_input)
        if (
            record.get("input_digest") != cached_input_sha256
            or cached_input_sha256 != input_row_sha256
        ):
            raise GokuAtomicMotionQwenError("cache input row hash differs")
        source_video_sha256 = str(row["source_video_sha256"])
        if cached_input.get("source_video_sha256") != source_video_sha256:
            raise GokuAtomicMotionQwenError("cache source-video hash differs")
        audit["source_video_sha256"] = source_video_sha256
        record_digest = record.get("record_digest")
        if (
            not isinstance(record_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", record_digest) is None
            or record_digest
            != v16._digest_object_with_field(record, "record_digest")
        ):
            raise GokuAtomicMotionQwenError("cache record self-digest differs")
        audit["cached_record_digest"] = record_digest
        source_stage = record.get("source_stage")
        if (
            not isinstance(source_stage, Mapping)
            or source_stage.get("selected_attempt") not in {1, 2}
        ):
            raise GokuAtomicMotionQwenError("cache source stage is not successful")
        cached_census = record.get("source_census")
        if not isinstance(cached_census, Mapping):
            raise GokuAtomicMotionQwenError("cache source census is absent")
        census = v16.validate_source_census(cached_census, expected_iid=iid)
        if census != cached_census:
            raise GokuAtomicMotionQwenError("cache source census is not canonical")
        audit.update(
            {
                "status": "hit",
                "source_census_sha256": object_sha256(census),
                "cached_source_stage_sha256": object_sha256(source_stage),
                "rejection": None,
            }
        )
        return census, audit
    except (OSError, UnicodeError, v16.GokuFullMotionQwenV16Error) as error:
        audit["status"] = "rejected"
        audit["rejection"] = f"{type(error).__name__}: {str(error)[:500]}"
        return None, audit


def _annotate_source_stage(
    row: Mapping[str, Any],
    *,
    backend: Any,
    source_path: Path,
    anchor_path: Path,
    visuals: Sequence[Any],
    visual_input_digest: str,
    runtime: Mapping[str, int],
    trace: dict[str, Any],
    source_census_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run the unchanged v16 source-census stage."""

    iid = str(row["iid"])
    cache_audit: dict[str, Any] | None = None
    if source_census_cache_root is not None:
        cached_census, cache_audit = _load_cached_source_census(
            row, cache_root=source_census_cache_root
        )
        if cached_census is not None:
            trace["source_stage"] = {
                "attempts": [],
                "selected_attempt": "cache",
                "mechanical_repair": None,
                "cache": cache_audit,
            }
            trace["source_census"] = cached_census
            return cached_census
    base_prompt = v16.build_source_census_prompt(iid)
    prompt = base_prompt
    attempts: list[dict[str, Any]] = []
    for attempt in (1, 2):
        raw = v16._generate_visual_stage(
            backend,
            custom_method="generate_source_motion_census_v16",
            system=v16.SOURCE_CENSUS_SYSTEM,
            prompt=prompt,
            source_path=source_path,
            anchor_path=anchor_path,
            visuals=visuals,
            visual_input_digest=visual_input_digest,
            runtime=runtime,
        )
        try:
            value, parse_repairs = v16._loads_object(raw, stage="source census")
            canonical, repair = v16.canonicalize_source_census(
                value, expected_iid=iid
            )
            repair["operations"] = parse_repairs + repair["operations"]
            census = v16.validate_source_census(canonical, expected_iid=iid)
        except Exception as error:
            attempts.append(
                v16._attempt_transcript(
                    attempt=attempt, prompt=prompt, raw=raw, error=error
                )
            )
            trace["source_stage"] = {
                "attempts": attempts,
                "selected_attempt": None,
                "mechanical_repair": None,
            }
            if cache_audit is not None:
                trace["source_stage"]["cache"] = cache_audit
            if attempt == 1 and v16._is_retryable_schema_error(error):
                prompt = v16._schema_retry_prompt(
                    base_prompt, stage="source census", error=error
                )
                continue
            raise GokuAtomicMotionQwenStageError("source_census", attempts) from error
        attempts.append(
            v16._attempt_transcript(attempt=attempt, prompt=prompt, raw=raw, error=None)
        )
        trace["source_stage"] = {
            "attempts": attempts,
            "selected_attempt": attempt,
            "mechanical_repair": repair,
        }
        if cache_audit is not None:
            trace["source_stage"]["cache"] = cache_audit
        trace["source_census"] = census
        return census
    raise AssertionError("unreachable source census loop")


def annotate_prepared_row(
    row: Mapping[str, Any],
    *,
    backend: Any,
    source_path: Path,
    anchor_path: Path,
    media_verification: Mapping[str, Any],
    visuals: Sequence[Any],
    visual_input_digest: str,
    runtime: Mapping[str, int],
    trace: dict[str, Any] | None = None,
    source_census_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run the reused v16 census and one atomic-wrapper target call."""

    trace_out = trace if trace is not None else {}
    trace_out["media_verification"] = dict(media_verification)
    trace_out["visual_input_digest"] = visual_input_digest
    census = _annotate_source_stage(
        row,
        backend=backend,
        source_path=source_path,
        anchor_path=anchor_path,
        visuals=visuals,
        visual_input_digest=visual_input_digest,
        runtime=runtime,
        trace=trace_out,
        source_census_cache_root=source_census_cache_root,
    )

    iid = str(row["iid"])
    base_prompt = build_atomic_target_plan_prompt(
        census, legacy_prompt=str(row.get("prompt") or "")
    )
    prompt = base_prompt
    attempts: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    for attempt in (1, 2):
        raw = v16._generate_visual_stage(
            backend,
            custom_method="generate_atomic_target_plan_v1",
            system=ATOMIC_TARGET_SYSTEM,
            prompt=prompt,
            source_path=source_path,
            anchor_path=anchor_path,
            visuals=visuals,
            visual_input_digest=visual_input_digest,
            runtime=runtime,
        )
        try:
            validation_repairs: list[str] = []
            value, parse_repairs = v16._loads_object(
                raw, stage="atomic target response"
            )
            accepted = validate_atomic_target_response(
                value,
                source_census=census,
                repair_operations=validation_repairs,
            )
        except Exception as error:
            attempts.append(
                v16._attempt_transcript(
                    attempt=attempt, prompt=prompt, raw=raw, error=error
                )
            )
            trace_out["atomic_target_stage"] = {
                "prompt_contract_sha256": _prompt_digest(),
                "attempts": attempts,
                "selected_attempt": None,
                "parse_repairs": None,
            }
            if attempt == 1 and _atomic_retryable(error):
                prompt = _atomic_retry_prompt(base_prompt, error=error)
                continue
            raise GokuAtomicMotionQwenStageError(
                "atomic_target_plan", attempts
            ) from error
        attempts.append(
            v16._attempt_transcript(attempt=attempt, prompt=prompt, raw=raw, error=None)
        )
        trace_out["atomic_target_stage"] = {
            "prompt_contract_sha256": _prompt_digest(),
            "attempts": attempts,
            "selected_attempt": attempt,
            "parse_repairs": parse_repairs + validation_repairs,
        }
        break
    assert accepted is not None
    target_plan = accepted["target_plan"]
    compiled = v16.compile_instruction(census, target_plan)
    payload = {
        "media_verification": dict(media_verification),
        "visual_input_digest": visual_input_digest,
        "source_stage": trace_out["source_stage"],
        "atomic_target_stage": trace_out["atomic_target_stage"],
        "source_census": census,
        "atomic_target_response": accepted,
        "target_plan": target_plan,
        "compiled_instruction": compiled,
    }
    trace_out.update(payload)
    return payload


def _model_identity(backend: Any) -> dict[str, str]:
    return v16._model_identity(backend)


def _new_record(
    row: Mapping[str, Any], *, backend: Any, runtime: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA,
        "planner_provenance": _provenance(),
        "iid": str(row["iid"]),
        "status": "running",
        "input_digest": object_sha256(row),
        "input_row": copy.deepcopy(dict(row)),
        "model": _model_identity(backend),
        "runtime": dict(runtime),
        "media_verification": None,
        "visual_input_digest": None,
        "source_stage": None,
        "atomic_target_stage": None,
        "source_census": None,
        "atomic_target_response": None,
        "target_plan": None,
        "compiled_instruction": None,
        "error": None,
        "record_digest": None,
    }


def _passed_row(
    record: Mapping[str, Any], *, source_path: Path, anchor_path: Path
) -> dict[str, Any]:
    """Project an atomic record into the unchanged Wan-facing v16 schema."""

    return v16._passed_row(record, source_path=source_path, anchor_path=anchor_path)


def _validate_terminal_receipt(
    receipt_path: Path, *, output_root: Path, iid: str, input_digest: str
) -> dict[str, Any]:
    """Validate an immutable atomic-planner terminal receipt."""

    receipt = v16._strict_read_object(receipt_path)
    _require_closed(
        receipt,
        {
            "schema_version",
            "iid",
            "status",
            "input_digest",
            "result_path",
            "result_sha256",
            "passed_path",
            "passed_sha256",
            "receipt_digest",
        },
        context="terminal_receipt",
    )
    if (
        receipt["schema_version"] != ROW_RECEIPT_SCHEMA
        or receipt["iid"] != iid
        or receipt["input_digest"] != input_digest
        or receipt["status"] not in {"ok", "error"}
    ):
        raise GokuAtomicMotionQwenError("terminal receipt identity differs")
    result_path = output_root / "rows" / iid / "result.json"
    if receipt["result_path"] != str(result_path.resolve()) or not result_path.is_file():
        raise GokuAtomicMotionQwenError("terminal receipt result path differs")
    if receipt["result_sha256"] != _file_digest(result_path):
        raise GokuAtomicMotionQwenError("terminal receipt result hash differs")
    passed_path = output_root / "passed" / f"{iid}.jsonl"
    if receipt["status"] == "ok":
        if (
            receipt["passed_path"] != str(passed_path.resolve())
            or not passed_path.is_file()
            or receipt["passed_sha256"] != _file_digest(passed_path)
        ):
            raise GokuAtomicMotionQwenError("terminal receipt passed binding differs")
        passed_rows = _iter_jsonl(passed_path)
        if len(passed_rows) != 1:
            raise GokuAtomicMotionQwenError("passed fragment must contain one row")
        validate_passed_row(passed_rows[0])
    elif receipt["passed_path"] is not None or receipt["passed_sha256"] is not None:
        raise GokuAtomicMotionQwenError("error receipt unexpectedly binds passed row")
    if receipt["receipt_digest"] != v16._digest_object_with_field(
        receipt, "receipt_digest"
    ):
        raise GokuAtomicMotionQwenError("terminal receipt digest differs")
    return receipt


def run_one(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    backend: Any | None = None,
    prepare: Callable[..., tuple[Path, Path, dict[str, Any], tuple[Any, ...], str]] = prepare_row,
    loaded_rows: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    input_path = args.input.expanduser().resolve(strict=True)
    rows = _iter_jsonl(input_path) if loaded_rows is None else [dict(row) for row in loaded_rows]
    if args.num_rows != len(rows):
        raise GokuAtomicMotionQwenError(
            f"--num-rows={args.num_rows} differs from input rows={len(rows)}"
        )
    if args.row_index is None or not 0 <= args.row_index < args.num_rows:
        raise GokuAtomicMotionQwenError("row index is out of range")
    row = rows[args.row_index]
    validate_input_row(dict(row))
    iid = str(row["iid"])
    output_root = args.output_root.expanduser().resolve()
    v16._ensure_directory(output_root)
    result_path = output_root / "rows" / iid / "result.json"
    passed_path = output_root / "passed" / f"{iid}.jsonl"
    receipt_path = output_root / "terminal" / f"{iid}.receipt.json"
    input_digest = object_sha256(row)
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_terminal_receipt(
            receipt_path,
            output_root=output_root,
            iid=iid,
            input_digest=input_digest,
        )
        return 0 if receipt["status"] == "ok" or args.allow_errors else 2
    if result_path.exists() or passed_path.exists():
        raise GokuAtomicMotionQwenError(
            f"partial create-only output exists for iid={iid}; use a fresh output root"
        )
    runtime = {
        "nframes": args.nframes,
        "max_pixels": args.max_pixels,
        "tile_width": args.tile_width,
        "mosaic_columns": args.mosaic_columns,
    }
    if backend is None:
        factory = backend_factory or LocalQwenBackend
        backend = factory(
            model_path=args.model,
            mode="visual",
            attn_implementation=args.attn_implementation,
            allow_download=args.allow_download,
            max_new_tokens=args.max_new_tokens,
        )
        _reject_backend_cpu_or_disk_offload(backend)
    record = _new_record(row, backend=backend, runtime=runtime)
    source_path: Path | None = None
    anchor_path: Path | None = None
    trace: dict[str, Any] = {}
    try:
        source_path, anchor_path, media, visuals, visual_digest = prepare(
            row, root=args.root.expanduser().resolve(strict=True), runtime=runtime
        )
        cache_root_arg = getattr(args, "source_census_cache_root", None)
        cache_root = (
            None
            if cache_root_arg is None
            else cache_root_arg.expanduser().resolve(strict=True)
        )
        record.update(
            annotate_prepared_row(
                row,
                backend=backend,
                source_path=source_path,
                anchor_path=anchor_path,
                media_verification=media,
                visuals=visuals,
                visual_input_digest=visual_digest,
                runtime=runtime,
                trace=trace,
                source_census_cache_root=cache_root,
            )
        )
        record["status"] = "ok"
    except GokuAtomicMotionQwenStageError as error:
        for field in (
            "media_verification",
            "visual_input_digest",
            "source_stage",
            "atomic_target_stage",
            "source_census",
            "atomic_target_response",
            "target_plan",
            "compiled_instruction",
        ):
            if field in trace:
                record[field] = trace[field]
        record["status"] = "error"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    record["record_digest"] = v16._digest_object_with_field(record, "record_digest")
    result_bytes = v16._pretty_bytes(record)
    v16._publish_create_only(result_path, result_bytes)

    passed_sha: str | None = None
    if record["status"] == "ok":
        assert source_path is not None and anchor_path is not None
        passed = validate_passed_row(
            _passed_row(record, source_path=source_path, anchor_path=anchor_path)
        )
        passed_bytes = v16._canonical_bytes(passed) + b"\n"
        v16._publish_create_only(passed_path, passed_bytes)
        passed_sha = v16._sha256_bytes(passed_bytes)
    receipt: dict[str, Any] = {
        "schema_version": ROW_RECEIPT_SCHEMA,
        "iid": iid,
        "status": record["status"],
        "input_digest": input_digest,
        "result_path": str(result_path.resolve()),
        "result_sha256": v16._sha256_bytes(result_bytes),
        "passed_path": str(passed_path.resolve()) if passed_sha is not None else None,
        "passed_sha256": passed_sha,
        "receipt_digest": None,
    }
    receipt["receipt_digest"] = v16._digest_object_with_field(
        receipt, "receipt_digest"
    )
    v16._publish_create_only(receipt_path, v16._pretty_bytes(receipt))
    print(
        f"[goku-atomic-motion-qwen] iid={iid} status={record['status']} "
        f"result={result_path}",
        flush=True,
    )
    return 0 if record["status"] == "ok" or args.allow_errors else 2


def run_worker(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[..., Any] | None = None,
    prepare: Callable[..., tuple[Path, Path, dict[str, Any], tuple[Any, ...], str]] = prepare_row,
) -> int:
    """Process one deterministic strided shard with one persistent backend."""

    input_path = args.input.expanduser().resolve(strict=True)
    rows = _iter_jsonl(input_path)
    if args.num_rows != len(rows):
        raise GokuAtomicMotionQwenError(
            f"--num-rows={args.num_rows} differs from input rows={len(rows)}"
        )
    worker_index = getattr(args, "worker_index", None)
    num_workers = getattr(args, "num_workers", None)
    if type(worker_index) is not int or type(num_workers) is not int:
        raise GokuAtomicMotionQwenError(
            "worker mode requires integer --worker-index and --num-workers"
        )
    if not 1 <= num_workers <= args.num_rows:
        raise GokuAtomicMotionQwenError("num_workers must be in [1, num_rows]")
    if not 0 <= worker_index < num_workers:
        raise GokuAtomicMotionQwenError("worker_index must be in [0, num_workers)")
    assigned = list(range(worker_index, args.num_rows, num_workers))
    if not assigned:
        raise GokuAtomicMotionQwenError("worker owns no input rows")

    factory = backend_factory or LocalQwenBackend
    shared_backend: Any | None = None

    def persistent_factory(**kwargs: Any) -> Any:
        nonlocal shared_backend
        if shared_backend is None:
            shared_backend = factory(**kwargs)
            _reject_backend_cpu_or_disk_offload(shared_backend)
        return shared_backend

    processed = 0
    for row_index in assigned:
        row_args = argparse.Namespace(**vars(args))
        row_args.row_index = row_index
        status = run_one(
            row_args,
            backend_factory=persistent_factory,
            prepare=prepare,
            loaded_rows=rows,
        )
        processed += 1
        if status != 0:
            print(
                "[goku-atomic-motion-qwen-worker] "
                f"worker={worker_index}/{num_workers} stopped_at={row_index} "
                f"processed={processed}/{len(assigned)} status={status}",
                flush=True,
            )
            return status
    print(
        "[goku-atomic-motion-qwen-worker] "
        f"worker={worker_index}/{num_workers} indices={assigned} "
        f"processed={processed} backend_loaded={shared_backend is not None}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-census-cache-root",
        type=Path,
        help=(
            "optional prior v16 Qwen root containing rows/<iid>/result.json; "
            "only exact IID/input-row/source-video-hash matches are reused"
        ),
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--row-index", type=int)
    selector.add_argument("--worker-index", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-rows", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--nframes", type=int, default=DEFAULT_NFRAMES)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--tile-width", type=int, default=DEFAULT_TILE_WIDTH)
    parser.add_argument("--mosaic-columns", type=int, default=DEFAULT_MOSAIC_COLUMNS)
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker_index is not None:
        return run_worker(args)
    if args.num_workers is not None:
        raise GokuAtomicMotionQwenError(
            "--num-workers is only valid with --worker-index"
        )
    return run_one(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except GokuAtomicMotionQwenError as error:
        print(f"[goku-atomic-motion-qwen] ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)

#!/usr/bin/env python3
"""Audit-only catalog for real adjacent MEV target graph teachers.

This module deliberately does *not* create a trainer-ready manifest.  It
catalogs the 3,749 real source/target candidates, removes every video UUID
owned by the sealed development/confirmation evaluation set, and records the
remaining authority gaps.  The current dataset is Qwen-visually accepted but
pending human qualification, has ``formal_sft_authorized=false``, and contains
continuation-derived adjacent targets rather than strict counterfactual ground
truth.  Those facts are immutable input invariants, not labels that a caller
may override.

The only permitted future use is a frozen, stop-gradient target-video graph
teacher whose published output is relational graph metadata.  Target RGB,
latents, flow, hidden states, or values may not become generator conditions or
regression targets.  This catalog itself authorizes neither graph-teacher
pretraining nor generator training.

The exclusion registry is checked against both the source manual-action
contract and the candidate JSONL.  Exclusion is by the complete MEV video UUID,
not merely by the named pair, so other adjacent event pairs from the same
video are also removed.  UUID/actor/scene/action/media/perceptual groups are
checked for cross-split leakage whenever an authoritative identifier is
present.  Free-text captions are never hashed and relabelled as actor, scene,
or action identities.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, NoReturn, Sequence
import uuid as uuid_module


CANDIDATE_SCHEMA = "mev-action-edit-paired-training-candidate-v2"
EXCLUSION_SCHEMA = "bernini-real-target-graph-eval-exclusion-registry-v1"
CATALOG_ROW_SCHEMA = "bernini-real-target-graph-pretrain-catalog-row-v1"
RECEIPT_SCHEMA = "bernini-real-target-graph-pretrain-catalog-receipt-v1"
AUTHORIZED_SPLIT_SCHEMA = "bernini-real-target-graph-pretrain-authorized-split-v1"

EXPECTED_MODE = "paired_with_real_adjacent_target"
EXPECTED_TARGET_PROVENANCE = "real-adjacent-segment"
EXPECTED_QUALIFICATION = (
    "qwen-visual-accepted-annotation-instruction-pending-human"
)
EXPECTED_SEMANTIC_TRUTH = "continuation-derived"
EXPECTED_TRAINING_USE = "sft_candidate_pending_human_qualification"
EXPECTED_INSTRUCTION_SOURCE = "mev.json target event caption"

DEVELOPMENT_PREFIXES = frozenset(
    ("840b214afead", "8b05aaf463db", "40712e1341dc", "5e83a9279951")
)
SPLITS = ("train", "validation", "test")
GROUP_KINDS = (
    "video_uuid",
    "actor_group_id",
    "scene_group_id",
    "action_group_id",
    "source_media_sha256",
    "target_media_sha256",
    "perceptual_cluster_id",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PREFIX_RE = re.compile(r"^[0-9a-f]{12}$")
GROUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}$")


class RealTargetGraphCatalogError(RuntimeError):
    """Raised before ambiguous metadata gains catalog or split authority."""


def fail(message: str) -> NoReturn:
    raise RealTargetGraphCatalogError(message)


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
        raise RealTargetGraphCatalogError("value is not canonical JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    fail(f"non-finite JSON constant is forbidden: {value}")


def _decode_json(raw: str, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except RealTargetGraphCatalogError:
        raise
    except Exception as error:
        raise RealTargetGraphCatalogError(f"invalid {label}: {error}") from error
    if type(value) is not dict:
        fail(f"{label} root must be one JSON object")
    return value


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        return _decode_json(path.read_text(encoding="utf-8"), label=str(path))
    except OSError as error:
        raise RealTargetGraphCatalogError(f"cannot read {path}: {error}") from error


def load_jsonl(path: Path) -> tuple[list[Mapping[str, Any]], str]:
    rows: list[Mapping[str, Any]] = []
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    fail(f"candidate JSONL line {line_number} is blank")
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise RealTargetGraphCatalogError(
                        f"candidate JSONL line {line_number} is not UTF-8"
                    ) from error
                rows.append(
                    _decode_json(text, label=f"candidate JSONL line {line_number}")
                )
    except OSError as error:
        raise RealTargetGraphCatalogError(f"cannot read {path}: {error}") from error
    if not rows:
        fail("candidate JSONL is empty")
    return rows, digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        fail(f"{label} must be an object")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        fail(f"{label} must be {'a nonempty ' if nonempty else 'an '}array")
    return value


def _exact_keys(
    value: Any,
    required: Iterable[str],
    label: str,
    *,
    optional: Iterable[str] = (),
) -> Mapping[str, Any]:
    row = _mapping(value, label)
    required_set = set(required)
    optional_set = set(optional)
    missing = required_set - set(row)
    extra = set(row) - required_set - optional_set
    if missing or extra:
        fail(f"{label} field closure differs: missing={sorted(missing)} extra={sorted(extra)}")
    return row


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        fail(f"{label} must be nonempty boundary-trimmed text")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be lowercase SHA-256")
    return value


def _prefix(value: Any, label: str) -> str:
    if not isinstance(value, str) or PREFIX_RE.fullmatch(value) is None:
        fail(f"{label} must be 12 lowercase hexadecimal characters")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer")
    return value


def _canonical_uuid(value: Any, label: str) -> str:
    text = _text(value, label)
    try:
        parsed = uuid_module.UUID(text)
    except ValueError as error:
        raise RealTargetGraphCatalogError(f"{label} must be a UUID") from error
    if str(parsed) != text:
        fail(f"{label} must use canonical lowercase UUID spelling")
    return text


def _group_id(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or GROUP_ID_RE.fullmatch(value) is None:
        fail(f"{label} is not a safe explicit group identifier")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def validate_manual_contracts(value: Mapping[str, Any]) -> set[str]:
    if value.get("schema_version") != "mev-action-anchor-human-contracts-v2":
        fail("manual contracts schema_version is not v2")
    if value.get("evaluation_role") != "calibration_only_not_independent_test":
        fail("manual contracts evaluation_role changed")
    samples = _array(value.get("samples"), "manual contracts samples", nonempty=True)
    prefixes: list[str] = []
    for index, sample_value in enumerate(samples):
        sample = _mapping(sample_value, f"manual contracts sample {index}")
        prefixes.append(_prefix(sample.get("pair_prefix"), f"manual sample {index} prefix"))
    if len(prefixes) != 16 or len(set(prefixes)) != 16:
        fail("manual contracts must contain exactly 16 unique evaluation pair prefixes")
    if not DEVELOPMENT_PREFIXES.issubset(prefixes):
        fail("manual contracts do not contain all four sealed development prefixes")
    return set(prefixes)


def validate_exclusion_registry(
    value: Mapping[str, Any],
    *,
    manual_contracts: Mapping[str, Any],
    manual_contracts_sha256: str,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    registry = _exact_keys(
        value,
        ("schema_version", "candidate_manifest", "manual_contracts", "policy", "cases"),
        "exclusion registry",
    )
    if registry["schema_version"] != EXCLUSION_SCHEMA:
        fail("exclusion registry schema_version changed")

    candidate_pin = _exact_keys(
        registry["candidate_manifest"],
        ("path", "sha256", "row_count"),
        "candidate manifest pin",
    )
    _text(candidate_pin["path"], "candidate manifest pin path")
    _sha256(candidate_pin["sha256"], "candidate manifest pin sha256")
    _positive_int(candidate_pin["row_count"], "candidate manifest pin row_count")

    manual_pin = _exact_keys(
        registry["manual_contracts"],
        ("path", "sha256", "sample_count"),
        "manual contracts pin",
    )
    _text(manual_pin["path"], "manual contracts pin path")
    if _sha256(manual_pin["sha256"], "manual contracts pin sha256") != manual_contracts_sha256:
        fail("manual contracts bytes do not match the exclusion registry pin")
    if _positive_int(manual_pin["sample_count"], "manual contracts sample_count") != 16:
        fail("manual contracts pin must require 16 samples")
    manual_prefixes = validate_manual_contracts(manual_contracts)

    policy = _exact_keys(
        registry["policy"],
        (
            "development_count",
            "locked_confirmation_count",
            "total_case_count",
            "exclude_entire_video_uuid",
            "exclude_all_temporally_adjacent_pairs",
            "development_pair_prefixes",
        ),
        "exclusion policy",
    )
    if (
        policy["development_count"] != 4
        or policy["locked_confirmation_count"] != 12
        or policy["total_case_count"] != 16
        or policy["exclude_entire_video_uuid"] is not True
        or policy["exclude_all_temporally_adjacent_pairs"] is not True
    ):
        fail("exclusion policy is not the sealed 4+12 entire-UUID policy")
    development = _array(
        policy["development_pair_prefixes"],
        "development pair prefixes",
        nonempty=True,
    )
    if set(development) != DEVELOPMENT_PREFIXES or len(development) != 4:
        fail("development pair prefix registry differs from the sealed four")

    cases = _array(registry["cases"], "exclusion cases", nonempty=True)
    if len(cases) != 16:
        fail("exclusion registry must have exactly 16 cases")
    prefixes: set[str] = set()
    pair_ids: set[str] = set()
    uuids: set[str] = set()
    roles = Counter()
    validated: list[Mapping[str, Any]] = []
    case_fields = (
        "pair_prefix",
        "pair_id",
        "uuid",
        "evaluation_role",
        "source_event_id",
        "target_event_id",
        "source_filename",
        "target_filename",
        "adjacent_exclusion_scope",
    )
    for index, case_value in enumerate(cases):
        case = _exact_keys(case_value, case_fields, f"exclusion case {index}")
        prefix = _prefix(case["pair_prefix"], f"exclusion case {index} prefix")
        pair_id = _sha256(case["pair_id"], f"exclusion case {index} pair_id")
        video_uuid = _canonical_uuid(case["uuid"], f"exclusion case {index} uuid")
        if not pair_id.startswith(prefix):
            fail(f"exclusion case {index} pair_id does not match its prefix")
        role = case["evaluation_role"]
        expected_role = "development" if prefix in DEVELOPMENT_PREFIXES else "locked_confirmation"
        if role != expected_role:
            fail(f"exclusion case {prefix} has wrong evaluation_role")
        source_event_id = _positive_int(case["source_event_id"], f"case {prefix} source_event_id")
        target_event_id = _positive_int(case["target_event_id"], f"case {prefix} target_event_id")
        if target_event_id != source_event_id + 1:
            fail(f"exclusion case {prefix} is not an adjacent event pair")
        _text(case["source_filename"], f"case {prefix} source_filename")
        _text(case["target_filename"], f"case {prefix} target_filename")
        if case["adjacent_exclusion_scope"] != "entire_video_uuid":
            fail(f"exclusion case {prefix} does not exclude its entire video UUID")
        if prefix in prefixes or pair_id in pair_ids or video_uuid in uuids:
            fail("exclusion cases must have unique prefixes, pair IDs, and video UUIDs")
        prefixes.add(prefix)
        pair_ids.add(pair_id)
        uuids.add(video_uuid)
        roles[role] += 1
        validated.append(case)
    if prefixes != manual_prefixes:
        fail("exclusion registry does not exactly cover the 16 manual contracts")
    if roles != Counter({"locked_confirmation": 12, "development": 4}):
        fail("exclusion registry role counts are not 4 development + 12 locked")
    return validated, candidate_pin


def _required(row: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in row:
        fail(f"{label} is missing required field {key!r}")
    return row[key]


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _optional_explicit_groups(row: Mapping[str, Any], label: str) -> dict[str, str | None]:
    grouping_value = row.get("group_ids", {})
    grouping = _mapping(grouping_value, f"{label}.group_ids")
    result: dict[str, str | None] = {}
    for key in ("actor_group_id", "scene_group_id", "action_group_id"):
        top = row.get(key)
        nested = grouping.get(key)
        if top is not None and nested is not None and top != nested:
            fail(f"{label} has conflicting explicit {key} values")
        result[key] = _group_id(top if top is not None else nested, f"{label}.{key}")
    for key in ("source_media_sha256", "target_media_sha256"):
        value = row.get(key)
        result[key] = None if value is None else _sha256(value, f"{label}.{key}")
    result["perceptual_cluster_id"] = _group_id(
        row.get("perceptual_cluster_id"), f"{label}.perceptual_cluster_id"
    )
    return result


def validate_candidate(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    label = f"candidate row {index}"
    if _required(row, "schema_version", label) != CANDIDATE_SCHEMA:
        fail(f"{label} schema_version changed")
    pair_id = _sha256(_required(row, "pair_id", label), f"{label}.pair_id")
    _sha256(_required(row, "row_id", label), f"{label}.row_id")
    video_uuid = _canonical_uuid(_required(row, "uuid", label), f"{label}.uuid")
    if _required(row, "mode", label) != EXPECTED_MODE:
        fail(f"{label} is not a real adjacent source/target pair")
    if _required(row, "formal_sft_authorized", label) is not False:
        fail(f"{label}.formal_sft_authorized must remain false")
    if _required(row, "is_strict_counterfactual_ground_truth", label) is not False:
        fail(f"{label} cannot claim strict counterfactual ground truth")
    if _required(row, "training_use", label) != EXPECTED_TRAINING_USE:
        fail(f"{label}.training_use changed from pending human qualification")
    if _required(row, "instruction_source", label) != EXPECTED_INSTRUCTION_SOURCE:
        fail(f"{label}.instruction_source changed")
    if _required(row, "instruction_semantic_override_by_qwen_allowed", label) is not False:
        fail(f"{label} cannot give Qwen instruction authority")
    _text(_required(row, "instruction", label), f"{label}.instruction")
    _text(_required(row, "source_action_caption", label), f"{label}.source_action_caption")
    _text(_required(row, "target_action_caption", label), f"{label}.target_action_caption")
    if _required(row, "videos_copied", label) is not False:
        fail(f"{label}.videos_copied must remain false")

    split = _required(row, "split", label)
    if split not in SPLITS:
        fail(f"{label}.split is outside {SPLITS!r}")
    target = _mapping(_required(row, "target", label), f"{label}.target")
    if target.get("provenance") != EXPECTED_TARGET_PROVENANCE:
        fail(f"{label} target provenance changed")
    if target.get("qualification_status") != EXPECTED_QUALIFICATION:
        fail(f"{label} is not Qwen-accepted/pending-human")
    if target.get("semantic_truth_class") != EXPECTED_SEMANTIC_TRUTH:
        fail(f"{label} target is not continuation-derived")

    audit = _mapping(_required(row, "automatic_visual_audit", label), f"{label}.automatic_visual_audit")
    if audit.get("verdict") != "accept":
        fail(f"{label} automatic visual audit is not accepted")

    source_provenance = _mapping(
        _required(row, "source_annotation_provenance", label),
        f"{label}.source_annotation_provenance",
    )
    target_provenance = _mapping(
        _required(row, "target_annotation_provenance", label),
        f"{label}.target_annotation_provenance",
    )
    if source_provenance.get("uuid") != video_uuid or target_provenance.get("uuid") != video_uuid:
        fail(f"{label} source/target provenance UUIDs do not match the top-level UUID")

    source_event = _mapping(
        _required(row, "source_event_annotation", label), f"{label}.source_event_annotation"
    )
    target_event = _mapping(
        _required(row, "target_event_annotation", label), f"{label}.target_event_annotation"
    )
    source_event_id = _positive_int(source_event.get("event_id"), f"{label}.source event_id")
    target_event_id = _positive_int(target_event.get("event_id"), f"{label}.target event_id")
    if target_event_id != source_event_id + 1:
        fail(f"{label} source/target event IDs are not adjacent")
    source_end = _finite_number(source_event.get("end_time"), f"{label}.source end_time")
    target_start = _finite_number(target_event.get("start_time"), f"{label}.target start_time")
    if abs(source_end - target_start) > 1e-6:
        fail(f"{label} source end and target start are not temporally adjacent")
    source_filename = _text(source_event.get("filename"), f"{label}.source filename")
    target_filename = _text(target_event.get("filename"), f"{label}.target filename")
    source_path = _text(_required(row, "source_video_path", label), f"{label}.source_video_path")
    target_path = _text(_required(row, "target_video_path", label), f"{label}.target_video_path")
    if not source_path.startswith("/") or not target_path.startswith("/"):
        fail(f"{label} source and target paths must be absolute")
    if _basename(source_path) != source_filename or _basename(target_path) != target_filename:
        fail(f"{label} path basenames do not match event filenames")
    if source_path == target_path:
        fail(f"{label} source and target video paths are identical")

    groups = _optional_explicit_groups(row, label)
    return {
        "pair_id": pair_id,
        "row_id": row["row_id"],
        "uuid": video_uuid,
        "split": split,
        "source_event_id": source_event_id,
        "target_event_id": target_event_id,
        "source_filename": source_filename,
        "target_filename": target_filename,
        "source_video_path": source_path,
        "target_video_path": target_path,
        "instruction": row["instruction"],
        "source_action_caption": row["source_action_caption"],
        "target_action_caption": row["target_action_caption"],
        "groups": groups,
    }


def _group_split_audit(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values: dict[str, dict[str, set[str]]] = {
        kind: defaultdict(set) for kind in GROUP_KINDS
    }
    coverage = Counter()
    for row in rows:
        candidates = {"video_uuid": row["uuid"], **row["groups"]}
        for kind, value in candidates.items():
            if value is not None:
                coverage[kind] += 1
                values[kind][value].add(row["split"])
    collisions: dict[str, list[Mapping[str, Any]]] = {}
    for kind in GROUP_KINDS:
        collisions[kind] = [
            {"group_id": group_id, "splits": sorted(splits)}
            for group_id, splits in sorted(values[kind].items())
            if len(splits) > 1
        ]
    return {
        "row_coverage": {kind: coverage[kind] for kind in GROUP_KINDS},
        "unique_group_count": {kind: len(values[kind]) for kind in GROUP_KINDS},
        "cross_split_collisions": collisions,
        "all_available_groups_split_disjoint": not any(collisions.values()),
        "free_text_actor_scene_action_grouping_used": False,
    }


def build_catalog(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_manifest_sha256: str,
    exclusion_registry: Mapping[str, Any],
    exclusion_registry_sha256: str,
    manual_contracts: Mapping[str, Any],
    manual_contracts_sha256: str,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    cases, candidate_pin = validate_exclusion_registry(
        exclusion_registry,
        manual_contracts=manual_contracts,
        manual_contracts_sha256=manual_contracts_sha256,
    )
    if candidate_manifest_sha256 != candidate_pin["sha256"]:
        fail("candidate JSONL bytes do not match the exclusion registry pin")
    if len(candidate_rows) != candidate_pin["row_count"]:
        fail("candidate JSONL row count does not match the exclusion registry pin")

    validated: list[Mapping[str, Any]] = []
    pair_ids: set[str] = set()
    row_ids: set[str] = set()
    for index, row in enumerate(candidate_rows, 1):
        parsed = validate_candidate(row, index)
        if parsed["pair_id"] in pair_ids:
            fail(f"candidate pair_id is duplicated: {parsed['pair_id']}")
        if parsed["row_id"] in row_ids:
            fail(f"candidate row_id is duplicated: {parsed['row_id']}")
        pair_ids.add(parsed["pair_id"])
        row_ids.add(parsed["row_id"])
        validated.append(parsed)

    case_by_pair = {case["pair_id"]: case for case in cases}
    row_by_pair = {row["pair_id"]: row for row in validated}
    missing_pairs = sorted(set(case_by_pair) - set(row_by_pair))
    if missing_pairs:
        fail(f"candidate JSONL is missing {len(missing_pairs)} registered evaluation pairs")
    for pair_id, case in case_by_pair.items():
        row = row_by_pair[pair_id]
        exact_fields = (
            "uuid",
            "source_event_id",
            "target_event_id",
            "source_filename",
            "target_filename",
        )
        for field in exact_fields:
            if row[field] != case[field]:
                fail(f"registered evaluation pair {case['pair_prefix']} mismatches field {field}")

    excluded_uuids = {case["uuid"] for case in cases}
    primary_pairs = set(case_by_pair)
    catalog: list[Mapping[str, Any]] = []
    status_counts = Counter()
    source_split_counts = Counter()
    adjacent_excluded_count = 0
    for row in validated:
        if row["uuid"] in excluded_uuids:
            if row["pair_id"] in primary_pairs:
                status = "excluded_named_evaluation_pair"
            else:
                status = "excluded_temporally_adjacent_same_video_uuid"
                adjacent_excluded_count += 1
            reasons = ["sealed_evaluation_video_uuid"]
        else:
            status = "catalog_candidate_pending_authority"
            reasons = []
        status_counts[status] += 1
        source_split_counts[row["split"]] += 1
        catalog.append(
            {
                "schema_version": CATALOG_ROW_SCHEMA,
                "catalog_status": status,
                "catalog_partition": row["split"],
                "pair_id": row["pair_id"],
                "row_id": row["row_id"],
                "video_uuid": row["uuid"],
                "source_event_id": row["source_event_id"],
                "target_event_id": row["target_event_id"],
                "source_video_path": row["source_video_path"],
                "target_video_path": row["target_video_path"],
                "instruction": row["instruction"],
                "source_action_caption": row["source_action_caption"],
                "target_action_caption": row["target_action_caption"],
                "explicit_group_ids": row["groups"],
                "exclusion_reasons": reasons,
                "data_truth": {
                    "real_adjacent_target": True,
                    "semantic_truth_class": EXPECTED_SEMANTIC_TRUTH,
                    "strict_counterfactual_ground_truth": False,
                    "qualification_status": EXPECTED_QUALIFICATION,
                    "human_qualification_complete": False,
                    "formal_sft_authorized": False,
                    "qwen_instruction_authoritative": False,
                },
                "target_graph_teacher_contract": {
                    "role": "stop_gradient_graph_teacher_candidate_only",
                    "target_gradient_allowed": False,
                    "target_is_generator_input": False,
                    "target_rgb_latent_flow_regression_allowed": False,
                    "target_hidden_or_value_copy_allowed": False,
                    "published_output_scope": "object_nodes_typed_edges_lifecycle_only",
                    "graph_teacher_pretraining_authorized": False,
                    "generator_training_authorized": False,
                },
            }
        )

    eligible_rows = [row for row in validated if row["uuid"] not in excluded_uuids]
    group_audit = _group_split_audit(eligible_rows)
    blockers = [
        "ALL_ROWS_PENDING_HUMAN_QUALIFICATION",
        "FORMAL_SFT_AUTHORIZED_FALSE",
    ]
    for kind in ("actor_group_id", "scene_group_id", "action_group_id"):
        if group_audit["row_coverage"][kind] != len(eligible_rows):
            blockers.append(f"INCOMPLETE_{kind.upper()}_COVERAGE")
    for kind in ("source_media_sha256", "target_media_sha256", "perceptual_cluster_id"):
        if group_audit["row_coverage"][kind] != len(eligible_rows):
            blockers.append(f"INCOMPLETE_{kind.upper()}_DEDUP_COVERAGE")
    if not group_audit["all_available_groups_split_disjoint"]:
        blockers.append("AVAILABLE_GROUP_CROSSES_SOURCE_DATASET_SPLITS")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "CATALOG_AUDITED_TRAIN_BLOCKED",
        "candidate_manifest": {
            "path": candidate_pin["path"],
            "sha256": candidate_manifest_sha256,
            "row_count": len(validated),
        },
        "authority": {
            "exclusion_registry_sha256": exclusion_registry_sha256,
            "manual_contracts_sha256": manual_contracts_sha256,
            "registered_development_cases": 4,
            "registered_locked_confirmation_cases": 12,
            "registered_total_cases": 16,
            "all_registered_pairs_found_exactly": True,
        },
        "counts": {
            "catalog_rows": len(catalog),
            "catalog_candidate_pending_authority": status_counts[
                "catalog_candidate_pending_authority"
            ],
            "excluded_named_evaluation_pairs": status_counts[
                "excluded_named_evaluation_pair"
            ],
            "excluded_temporally_adjacent_same_video_uuid": adjacent_excluded_count,
            "excluded_total_by_evaluation_uuid": len(catalog) - len(eligible_rows),
            "source_dataset_splits": {key: source_split_counts[key] for key in SPLITS},
        },
        "leakage_audit": {
            "exclusion_granularity": "entire_video_uuid",
            "named_evaluation_pair_in_candidate_count": 0,
            "evaluation_uuid_in_candidate_count": 0,
            "group_split_audit": group_audit,
            "actor_scene_action_from_free_text_inferred": False,
        },
        "data_status": {
            "qualification_status": EXPECTED_QUALIFICATION,
            "human_qualification_complete": False,
            "formal_sft_authorized": False,
            "target_provenance": EXPECTED_TARGET_PROVENANCE,
            "semantic_truth_class": EXPECTED_SEMANTIC_TRUTH,
            "strict_counterfactual_ground_truth_claimed": False,
        },
        "usage_contract": {
            "catalog_only": True,
            "target_video_role": "future_frozen_stop_gradient_graph_teacher_only",
            "target_gradient_allowed": False,
            "target_is_generator_input": False,
            "target_rgb_latent_flow_regression_allowed": False,
            "target_hidden_or_value_copy_allowed": False,
            "published_teacher_output_scope": "object_nodes_typed_edges_lifecycle_only",
            "graph_teacher_pretraining_authorized": False,
            "generator_training_authorized": False,
            "trainer_readable_split_emitted": False,
        },
        "authorization_blockers": blockers,
        "claim_limits": {
            "catalog_is_pretraining_result": False,
            "catalog_is_action_representation_evidence": False,
            "stable_transferable_action_representation_claimed": False,
            "strict_counterfactual_dataset_claimed": False,
            "scientific_generalization_claim_authorized": False,
        },
    }
    receipt["catalog_content_sha256"] = hashlib.sha256(
        b"".join(canonical_json_bytes(row) + b"\n" for row in catalog)
    ).hexdigest()
    receipt["receipt_sha256"] = object_sha256(receipt)
    return catalog, receipt


def emit_authorized_split_manifest(
    catalog: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    blockers = receipt.get("authorization_blockers")
    if blockers:
        fail(
            "trainer-readable split emission refused; unresolved blockers: "
            + ", ".join(str(value) for value in blockers)
        )
    if receipt.get("status") != "CATALOG_AUDITED_TRAIN_AUTHORIZED":
        fail("trainer-readable split emission requires explicit graph-teacher authorization")
    rows = [
        {"pair_id": row["pair_id"], "partition": row["catalog_partition"]}
        for row in catalog
        if row["catalog_status"] == "catalog_candidate_authorized"
    ]
    result: dict[str, Any] = {
        "schema_version": AUTHORIZED_SPLIT_SCHEMA,
        "rows": rows,
        "target_video_role": "frozen_stop_gradient_graph_teacher_only",
        "target_is_generator_input": False,
        "target_rgb_latent_flow_regression_allowed": False,
    }
    result["manifest_sha256"] = object_sha256(result)
    return result


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_jsonl_create_only(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--exclusion-registry", type=Path, required=True)
    parser.add_argument("--manual-contracts", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--authorized-split-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_rows, candidate_sha = load_jsonl(args.candidates)
    registry = load_json(args.exclusion_registry)
    manual = load_json(args.manual_contracts)
    catalog, receipt = build_catalog(
        candidate_rows,
        candidate_manifest_sha256=candidate_sha,
        exclusion_registry=registry,
        exclusion_registry_sha256=file_sha256(args.exclusion_registry),
        manual_contracts=manual,
        manual_contracts_sha256=file_sha256(args.manual_contracts),
    )
    _write_jsonl_create_only(args.catalog_output, catalog)
    _write_json_create_only(args.receipt_output, receipt)
    if args.authorized_split_output is not None:
        split = emit_authorized_split_manifest(catalog, receipt)
        _write_json_create_only(args.authorized_split_output, split)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

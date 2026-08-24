#!/usr/bin/env python3
"""Fail-closed admission for the 20260817_box paired exact160 contract.

This module does not train a model.  It proves that a proposed launch has the
three frozen, disjoint data manifests and the independent action-encoder
qualification required by ``md/action_editing/20260817_box``.  The old
``full644`` source catalog and self-generated action anchors are deliberately
not accepted as optimizer targets.  Production admission also verifies every
media/annotation artifact against an absolute path plus SHA-256 pin and parses
the closed target, anchor, annotation, and representation-disjointness
receipts.  ``verify_files=False`` is test-only structural preflight and can
never return a GO status.  More importantly, the box has not frozen canonical
implementations for ``row_id``, ``semantic_key``, or
``composition_semantic_key``.  Consequently even full file/receipt replay is
only structural preflight: this module never grants data, R1, R2, or training
GO from caller-supplied identity keys and self-reported receipts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple


SCHEMA = "bernini-exact160-target-grounded-admission-v1"
PREFLIGHT_RECEIPT_SCHEMA = "bernini-exact160-structural-preflight-receipt-v2"
ROW_SCHEMA = "bernini-exact160-common-row-v1"
QUALIFICATION_SCHEMA = "bernini-elal3-action-encoder-qualification-v1"
TARGET_RECEIPT_SCHEMA = "bernini-exact160-clean-target-qualification-receipt-v1"
ANCHOR_RECEIPT_SCHEMA = "bernini-exact160-anchor-compatibility-receipt-v1"
ANNOTATION_RECEIPT_SCHEMA = "bernini-exact160-media-annotation-receipt-v1"
REPRESENTATION_DISJOINTNESS_SCHEMA = "bernini-elal3-representation-disjointness-receipt-v1"
CANONICAL_IDENTITY_AUTHORITY_REQUIREMENT_SCHEMA = (
    "bernini-exact160-canonical-identity-authority-requirement-v1"
)
TRAIN_SPLIT = "exact160_train"
CALIBRATION_SPLIT = "calibration32"
LOCKED_SPLIT = "unseen_locked32"
STRATA = (
    "nontrivial_single_actor",
    "actor_object",
    "two_subject",
    "multi_entity_long_horizon",
)
INTERACTION_STRATA = set(STRATA[1:])
CAUSAL_ROLES = {"agent", "co_agent", "patient", "object", "instrument", "goal"}
ENTITY_ROLES = {"designated_actor", "secondary_actor", "patient", "object", "distractor"}
TARGET_PROVENANCE = {"real", "licensed-paired", "simulator", "qualified-teacher-pseudo"}
ANCHOR_PROVENANCE = {"real", "licensed", "simulator", "qualified-independent-generator"}
NEGATIVE_KEYS = {
    "noop", "reverse", "incomplete", "wrong_actor", "wrong_object",
    "wrong_participant_or_role", "camera_only", "appearance_only",
}
FORBIDDEN_OPTIMIZER_FIELDS = {
    "teacher_unit", "frozen_source_action_velocity", "frozen_velocity",
    "frozen_relative_band", "psiout", "source_carrier_target",
    "anchor_clean_latent_target", "anchor_velocity_target",
    "self_generated_anchor_as_target", "fullfield_action_noop",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
ARTIFACT_KEYS = {
    "entity_soft_tracks", "signed_dense_motion", "visibility_confidence", "phase_windows",
}
TARGET_AXIS_KEYS = {
    "designated_actor_consistency",
    "action_direction_participant_effects",
    "ordered_transition_terminal_hold",
    "identity_preservation",
    "background_camera_non_target_preservation",
    "temporal_integrity",
    "visual_quality",
}
ANCHOR_AXIS_KEYS = {
    "causal_participants_roles_effects",
    "action_and_direction",
    "ordered_transition",
    "terminal_and_hold",
    "phase_alignment",
    "appearance_and_camera_excluded",
}
QUALIFICATION_AXIS_KEYS = {
    "same_action_different_actor_scene",
    "same_magnitude_reverse_direction",
    "action_vs_noop",
    "complete_vs_incomplete",
    "ordered_vs_shuffled",
    "correct_vs_wrong_actor_object_participant",
    "correct_vs_wrong_participant_role_effect",
    "object_motion_vs_camera_only",
    "occlusion_entity_continuity",
    "text_blind_pair_encoder",
    "false_validity_rejected",
}
QUALIFICATION_METRIC_THRESHOLDS = {
    "compatible_anchor_precision": 0.95,
    "compatible_anchor_recall": 0.80,
    "action_vs_noop_auroc": 0.90,
    "object_motion_vs_camera_only_auroc": 0.90,
    "participant_binding_auroc": 0.90,
    "participant_role_effect_auroc": 0.90,
    "forward_vs_reverse_auroc": 0.90,
    "complete_vs_incomplete_auroc": 0.90,
    "ordered_vs_shuffled_auroc": 0.90,
    "terminal_hold_macro_f1": 0.90,
    "occlusion_entity_association_idf1": 0.85,
}


class Exact160AdmissionError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise Exact160AdmissionError(message)


def _strict_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            fail("duplicate JSON key: " + key)
        value[key] = item
    return value


def load_json_strict(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = _decode_json_strict(raw, str(path))
    except Exact160AdmissionError:
        raise
    except Exception as exc:
        fail("cannot read strict JSON %s: %s" % (path, exc))
    return value


def _decode_json_strict(raw: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except Exact160AdmissionError:
        raise
    except Exception as exc:
        fail("cannot decode strict JSON %s: %s" % (label, exc))
    if type(value) is not dict:
        fail("JSON root must be an object: " + label)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Mapping[str, Any], digest_field: str) -> str:
    projected = dict(value)
    projected.pop(digest_field, None)
    raw = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_value_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _closed(value: Any, keys: Set[str], label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        fail(label + " must be an object")
    actual = set(value)
    if actual != keys:
        fail("%s keys differ: missing=%s extra=%s" % (label, sorted(keys - actual), sorted(actual - keys)))
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        fail(label + " must be a non-empty built-in string")
    return value


def _hex(value: Any, label: str) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        fail(label + " must be lowercase SHA-256")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        fail(label + " must be a built-in bool")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail("%s must be an int >= %d" % (label, minimum))
    return value


def _sequence(value: Any, label: str) -> List[Any]:
    if type(value) is not list:
        fail(label + " must be a list")
    return value


def _number(value: Any, label: str, minimum_exclusive: float = 0.0) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        fail(label + " must be a finite built-in number")
    number = float(value)
    if not math.isfinite(number) or number <= minimum_exclusive:
        fail("%s must be finite and > %s" % (label, minimum_exclusive))
    return number


def _utc_timestamp(value: Any, label: str) -> datetime:
    text = _text(value, label)
    if not text.endswith("Z"):
        fail(label + " must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        fail(label + " must be a valid RFC3339 UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        fail(label + " must be UTC")
    return parsed


def _stable_file(
    path_value: Any, sha_value: Any, label: str, verify_files: bool, capture: bool = False,
) -> Tuple[str, str, Any]:
    path = _text(path_value, label + ".path")
    expected = _hex(sha_value, label + ".sha256")
    resolved = Path(path)
    if not resolved.is_absolute():
        fail(label + " must reference an absolute path")
    if not verify_files:
        return path, expected, None

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail(label + " must reference a regular file")
        digest = hashlib.sha256()
        captured = bytearray() if capture else None
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if captured is not None:
                if len(captured) + len(block) > 8 * 1024 * 1024:
                    fail(label + " JSON receipt exceeds 8 MiB")
                captured.extend(block)
        after = os.fstat(descriptor)
    except Exact160AdmissionError:
        raise
    except OSError as exc:
        fail("%s cannot be opened as a stable non-symlink file: %s" % (label, exc))
    finally:
        if descriptor is not None:
            os.close(descriptor)

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    if identity_before != identity_after:
        fail(label + " changed while its bytes were being verified")
    if digest.hexdigest() != expected:
        fail(label + " bytes differ from SHA-256")
    return path, expected, bytes(captured) if captured is not None else None


def _stable_json_file(
    path_value: Any, sha_value: Any, label: str, verify_files: bool,
) -> Any:
    _, _, raw = _stable_file(path_value, sha_value, label, verify_files, capture=True)
    if not verify_files:
        return None
    return _decode_json_strict(raw, label)


def _artifact_pin(value: Any, label: str, verify_files: bool) -> Mapping[str, Any]:
    artifact = _closed(value, {"path", "sha256"}, label)
    _stable_file(artifact["path"], artifact["sha256"], label, verify_files)
    return artifact


def _forbidden_key_scan(value: Any, label: str = "root") -> None:
    if type(value) is dict:
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_OPTIMIZER_FIELDS:
                fail("forbidden optimizer field at %s.%s" % (label, key))
            _forbidden_key_scan(item, "%s.%s" % (label, key))
    elif type(value) is list:
        for index, item in enumerate(value):
            _forbidden_key_scan(item, "%s[%d]" % (label, index))


def _validate_source(value: Any, label: str, verify_files: bool) -> Mapping[str, Any]:
    source = _closed(value, {
        "path", "sha256", "source_id", "source_group_id", "scene_id",
        "camera_class", "entities", "entity_instance_group_ids",
    }, label)
    _stable_file(source["path"], source["sha256"], label, verify_files)
    for key in ("source_id", "source_group_id", "scene_id"):
        _text(source[key], label + "." + key)
    if source["camera_class"] not in {"static", "moving", "handheld", "other"}:
        fail(label + ".camera_class differs")
    entities = _sequence(source["entities"], label + ".entities")
    if not 1 <= len(entities) <= 3:
        fail(label + ".entities must contain 1..3 entities")
    entity_ids: List[str] = []
    groups: List[str] = []
    designated: List[str] = []
    for index, raw in enumerate(entities):
        entity = _closed(raw, {"entity_id", "instance_group_id", "taxonomy_id", "kind", "role"}, "%s.entities[%d]" % (label, index))
        entity_id = _text(entity["entity_id"], label + ".entity_id")
        entity_ids.append(entity_id)
        groups.append(_text(entity["instance_group_id"], label + ".instance_group_id"))
        _text(entity["taxonomy_id"], label + ".taxonomy_id")
        if entity["kind"] not in {"person", "animal", "object"} or entity["role"] not in ENTITY_ROLES:
            fail(label + " entity kind/role differs")
        if entity["role"] == "designated_actor":
            designated.append(entity_id)
    if len(set(entity_ids)) != len(entity_ids) or len(set(groups)) != len(groups):
        fail(label + " entity IDs/groups must be unique")
    if len(designated) != 1:
        fail(label + " must have exactly one designated actor")
    declared = _sequence(source["entity_instance_group_ids"], label + ".entity_instance_group_ids")
    if any(type(x) is not str or not x for x in declared) or declared != sorted(set(groups)):
        fail(label + ".entity_instance_group_ids differs from entities")
    return source


def _validate_instruction(value: Any, source: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    instruction = _closed(value, {
        "text", "sha256", "semantic_key", "composition_semantic_key",
        "designated_actor_id", "action_family", "direction", "participants", "initial_state",
        "ordered_transition", "terminal_state", "hold_requirement",
    }, label)
    text = _text(instruction["text"], label + ".text")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != _hex(instruction["sha256"], label + ".sha256"):
        fail(label + " text SHA differs")
    for key in ("semantic_key", "action_family", "direction", "initial_state", "ordered_transition", "terminal_state", "hold_requirement"):
        _text(instruction[key], label + "." + key)
    _hex(instruction["composition_semantic_key"], label + ".composition_semantic_key")
    entities = {x["entity_id"]: x for x in source["entities"]}
    actor = _text(instruction["designated_actor_id"], label + ".designated_actor_id")
    if actor not in entities or entities[actor]["role"] != "designated_actor":
        fail(label + " designated actor does not resolve")
    participants = _sequence(instruction["participants"], label + ".participants")
    causal = 0
    agents = 0
    participant_ids: Set[str] = set()
    for index, raw in enumerate(participants):
        participant = _closed(raw, {"entity_id", "semantic_role", "required_transition_or_effect"}, "%s.participants[%d]" % (label, index))
        entity_id = _text(participant["entity_id"], label + ".participant.entity_id")
        if entity_id not in entities or entity_id in participant_ids:
            fail(label + " participant does not uniquely resolve")
        participant_ids.add(entity_id)
        role = participant["semantic_role"]
        if role not in CAUSAL_ROLES | {"distractor"}:
            fail(label + " participant role differs")
        effect = participant["required_transition_or_effect"]
        if effect is not None:
            _text(effect, label + ".participant.effect")
        if role in CAUSAL_ROLES:
            causal += 1
        if role == "agent":
            agents += 1
            if entity_id != actor:
                fail(label + " agent must be designated actor")
    if agents != 1 or not 1 <= causal <= 3:
        fail(label + " must contain one agent and 1..3 causal participants")
    return instruction


def _validate_stratum_semantics(row: Mapping[str, Any], label: str) -> None:
    entities = {x["entity_id"]: x for x in row["source"]["entities"]}
    causal = [x for x in row["instruction"]["participants"] if x["semantic_role"] in CAUSAL_ROLES]
    non_agent = [x for x in causal if x["semantic_role"] != "agent"]
    stratum = row["stratum"]
    if stratum == "nontrivial_single_actor":
        if len(causal) != 1 or non_agent:
            fail(label + " single-actor stratum may contain only the designated agent participant")
    elif stratum == "actor_object":
        if len(causal) != 2 or len(non_agent) != 1 or non_agent[0]["semantic_role"] not in {"patient", "object", "instrument", "goal"}:
            fail(label + " actor-object stratum needs exactly one agent and one receiver/object role")
    elif stratum == "two_subject":
        if len(causal) != 2 or len(non_agent) != 1:
            fail(label + " two-subject stratum needs exactly two causal participants")
        kinds = [entities[x["entity_id"]]["kind"] for x in causal]
        if any(kind not in {"person", "animal"} for kind in kinds):
            fail(label + " two-subject stratum causal participants must both be subjects")
    elif stratum == "multi_entity_long_horizon":
        if not 2 <= len(causal) <= 3:
            fail(label + " multi-entity long-horizon stratum needs 2..3 causal participants")


def _causal_participants(instruction: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return [dict(x) for x in instruction["participants"] if x["semantic_role"] in CAUSAL_ROLES]


def _validate_review_authority(value: Any, label: str) -> Mapping[str, Any]:
    authority = _closed(value, {
        "kind", "version", "authority_sha256", "threshold_profile_sha256",
    }, label)
    if authority["kind"] not in {"human", "evaluator", "hybrid"}:
        fail(label + ".kind differs")
    _text(authority["version"], label + ".version")
    _hex(authority["authority_sha256"], label + ".authority_sha256")
    _hex(authority["threshold_profile_sha256"], label + ".threshold_profile_sha256")
    return authority


def _validate_axis_verdicts(value: Any, keys: Set[str], label: str, require_all_pass: bool) -> Mapping[str, Any]:
    axes = _closed(value, keys, label)
    for key, verdict in axes.items():
        if verdict not in {"PASS", "FAIL"}:
            fail(label + "." + key + " must be PASS or FAIL")
    if require_all_pass and set(axes.values()) != {"PASS"}:
        fail(label + " must be all PASS")
    if not require_all_pass and "FAIL" not in axes.values():
        fail(label + " must explain a non-accepted verdict with at least one FAIL")
    return axes


def _validate_full81_review(value: Any, label: str, verify_files: bool) -> Mapping[str, Any]:
    review = _closed(value, {"verdict", "evidence"}, label)
    if review["verdict"] != "accept":
        fail(label + " must be accept")
    _artifact_pin(review["evidence"], label + ".evidence", verify_files)
    return review


def _validate_target(value: Any, label: str, verify_files: bool) -> Tuple[Mapping[str, Any], Any]:
    target = _closed(value, {
        "path", "sha256", "provenance", "qualification_receipt_path",
        "qualification_receipt_sha256", "full_video_review",
    }, label)
    _stable_file(target["path"], target["sha256"], label, verify_files)
    receipt = _stable_json_file(
        target["qualification_receipt_path"], target["qualification_receipt_sha256"],
        label + ".qualification_receipt", verify_files,
    )
    if target["provenance"] not in TARGET_PROVENANCE or target["full_video_review"] != "accept":
        fail(label + " is not a qualified clean edited target")
    return target, receipt


def _validate_anchor(value: Any, label: str, verify_files: bool) -> Tuple[Mapping[str, Any], Any]:
    anchor = _closed(value, {
        "path", "sha256", "teacher_candidate_group_id", "provenance", "role",
        "compatibility", "participant_role_effects_and_action_direction",
        "ordered_transition", "terminal_state", "hold_requirement", "phase_windows",
        "full_81_frame_review", "compatibility_receipt_path", "compatibility_receipt_sha256",
    }, label)
    _stable_file(anchor["path"], anchor["sha256"], label, verify_files)
    receipt = _stable_json_file(
        anchor["compatibility_receipt_path"], anchor["compatibility_receipt_sha256"],
        label + ".compatibility_receipt", verify_files,
    )
    _text(anchor["teacher_candidate_group_id"], label + ".teacher_candidate_group_id")
    if anchor["provenance"] not in ANCHOR_PROVENANCE or anchor["role"] != "action-reference-only":
        fail(label + " provenance/role differs")
    if anchor["compatibility"] not in {"accept", "contrastive-only", "reject"}:
        fail(label + " compatibility differs")
    for key in ("participant_role_effects_and_action_direction", "ordered_transition", "terminal_state", "hold_requirement", "phase_windows"):
        _text(anchor[key], label + "." + key)
    if anchor["full_81_frame_review"] != "accept":
        fail(label + " lacks accepted full81 review")
    return anchor, receipt


def _validate_annotation(value: Any, label: str, verify_files: bool) -> Tuple[Mapping[str, Any], Any]:
    annotation = _closed(value, {
        "media_sha256", "media_role", "coordinate_space", "entity_soft_tracks",
        "signed_dense_motion", "visibility_confidence", "phase_windows",
        "annotation_receipt_path", "annotation_receipt_sha256",
    }, label)
    _hex(annotation["media_sha256"], label + ".media_sha256")
    if annotation["media_role"] not in {"source", "edited_target", "action_anchor"}:
        fail(label + ".media_role differs")
    _text(annotation["coordinate_space"], label + ".coordinate_space")
    artifact_pins = []
    for key in sorted(ARTIFACT_KEYS):
        artifact = _artifact_pin(annotation[key], label + "." + key, verify_files)
        artifact_pins.append((artifact["path"], artifact["sha256"]))
    if len(set(artifact_pins)) != len(artifact_pins):
        fail(label + " must use four distinct annotation artifacts")
    receipt = _stable_json_file(
        annotation["annotation_receipt_path"], annotation["annotation_receipt_sha256"],
        label + ".annotation_receipt", verify_files,
    )
    return annotation, receipt


def _validate_target_receipt(
    receipt: Mapping[str, Any], row: Mapping[str, Any], target: Mapping[str, Any],
    annotation: Mapping[str, Any], annotation_receipt: Mapping[str, Any],
    label: str, verify_files: bool,
) -> None:
    receipt = _closed(receipt, {
        "schema_version", "row_id", "target_sha256", "target_provenance", "fps",
        "frame_count", "designated_actor_id", "action_family", "direction",
        "causal_participants", "initial_state", "ordered_transition", "terminal_state",
        "hold_requirement", "annotation_receipt_sha256", "phase_windows_artifact_sha256",
        "axis_verdicts", "full_81_frame_review", "review_authority", "verdict",
        "receipt_digest",
    }, label)
    instruction = row["instruction"]
    expected = {
        "schema_version": TARGET_RECEIPT_SCHEMA,
        "row_id": row["row_id"],
        "target_sha256": target["sha256"],
        "target_provenance": target["provenance"],
        "designated_actor_id": instruction["designated_actor_id"],
        "action_family": instruction["action_family"],
        "direction": instruction["direction"],
        "initial_state": instruction["initial_state"],
        "ordered_transition": instruction["ordered_transition"],
        "terminal_state": instruction["terminal_state"],
        "hold_requirement": instruction["hold_requirement"],
        "annotation_receipt_sha256": annotation["annotation_receipt_sha256"],
        "phase_windows_artifact_sha256": annotation["phase_windows"]["sha256"],
        "verdict": "accept",
    }
    for key, wanted in expected.items():
        if receipt[key] != wanted:
            fail(label + "." + key + " does not join the row/target annotation")
    _number(receipt["fps"], label + ".fps")
    if _integer(receipt["frame_count"], label + ".frame_count", 1) != 81:
        fail(label + ".frame_count must be exact81")
    if receipt["fps"] != annotation_receipt["fps"] or receipt["frame_count"] != annotation_receipt["frame_count"]:
        fail(label + " fps/frame_count differs from target annotation receipt")
    if receipt["causal_participants"] != _causal_participants(instruction):
        fail(label + ".causal_participants does not join instruction participants")
    _validate_axis_verdicts(receipt["axis_verdicts"], TARGET_AXIS_KEYS, label + ".axis_verdicts", True)
    _validate_full81_review(receipt["full_81_frame_review"], label + ".full_81_frame_review", verify_files)
    _validate_review_authority(receipt["review_authority"], label + ".review_authority")
    _hex(receipt["receipt_digest"], label + ".receipt_digest")
    if canonical_digest(receipt, "receipt_digest") != receipt["receipt_digest"]:
        fail(label + ".receipt_digest differs")


def _validate_anchor_receipt(
    receipt: Mapping[str, Any], row: Mapping[str, Any], anchor: Mapping[str, Any],
    annotation: Any, label: str, verify_files: bool,
) -> None:
    receipt = _closed(receipt, {
        "schema_version", "row_id", "anchor_sha256", "anchor_provenance",
        "teacher_candidate_group_id", "causal_participants", "action_family", "direction",
        "ordered_transition", "terminal_state", "hold_requirement",
        "phase_windows_artifact_sha256", "compatibility_axes", "full_81_frame_review",
        "review_authority", "verdict", "receipt_digest",
    }, label)
    instruction = row["instruction"]
    expected = {
        "schema_version": ANCHOR_RECEIPT_SCHEMA,
        "row_id": row["row_id"],
        "anchor_sha256": anchor["sha256"],
        "anchor_provenance": anchor["provenance"],
        "teacher_candidate_group_id": anchor["teacher_candidate_group_id"],
        "action_family": instruction["action_family"],
        "direction": instruction["direction"],
        "ordered_transition": anchor["ordered_transition"],
        "terminal_state": anchor["terminal_state"],
        "hold_requirement": anchor["hold_requirement"],
        "verdict": anchor["compatibility"],
    }
    for key, wanted in expected.items():
        if receipt[key] != wanted:
            fail(label + "." + key + " does not join the row/anchor")
    if receipt["causal_participants"] != _causal_participants(instruction):
        fail(label + ".causal_participants does not join instruction participants")
    if annotation is not None:
        if receipt["phase_windows_artifact_sha256"] != annotation["phase_windows"]["sha256"]:
            fail(label + ".phase_windows_artifact_sha256 does not join anchor annotation")
    else:
        _hex(receipt["phase_windows_artifact_sha256"], label + ".phase_windows_artifact_sha256")
    accepted = anchor["compatibility"] == "accept"
    _validate_axis_verdicts(receipt["compatibility_axes"], ANCHOR_AXIS_KEYS, label + ".compatibility_axes", accepted)
    _validate_full81_review(receipt["full_81_frame_review"], label + ".full_81_frame_review", verify_files)
    _validate_review_authority(receipt["review_authority"], label + ".review_authority")
    _hex(receipt["receipt_digest"], label + ".receipt_digest")
    if canonical_digest(receipt, "receipt_digest") != receipt["receipt_digest"]:
        fail(label + ".receipt_digest differs")


def _validate_processing_authority(value: Any, label: str) -> Mapping[str, Any]:
    authority = _closed(value, {"name", "version", "implementation_sha256", "weights_sha256"}, label)
    _text(authority["name"], label + ".name")
    _text(authority["version"], label + ".version")
    _hex(authority["implementation_sha256"], label + ".implementation_sha256")
    _hex(authority["weights_sha256"], label + ".weights_sha256")
    return authority


def _validate_annotation_receipt(
    receipt: Mapping[str, Any], row: Mapping[str, Any], annotation: Mapping[str, Any],
    label: str, verify_files: bool,
) -> None:
    receipt = _closed(receipt, {
        "schema_version", "row_id", "input_media_sha256", "media_role",
        "extractor_authority", "tracker_authority", "fps", "frame_count",
        "native_coordinate_space", "entity_id_mapping", "artifacts",
        "visibility_confidence_abi", "phase_abi", "latent_grid_resize_abi",
        "verdict", "receipt_digest",
    }, label)
    expected = {
        "schema_version": ANNOTATION_RECEIPT_SCHEMA,
        "row_id": row["row_id"],
        "input_media_sha256": annotation["media_sha256"],
        "media_role": annotation["media_role"],
        "native_coordinate_space": annotation["coordinate_space"],
        "verdict": "accept",
    }
    for key, wanted in expected.items():
        if receipt[key] != wanted:
            fail(label + "." + key + " does not join the row/media annotation")
    _validate_processing_authority(receipt["extractor_authority"], label + ".extractor_authority")
    _validate_processing_authority(receipt["tracker_authority"], label + ".tracker_authority")
    _number(receipt["fps"], label + ".fps")
    if _integer(receipt["frame_count"], label + ".frame_count", 1) != 81:
        fail(label + ".frame_count must be exact81")

    mappings = _sequence(receipt["entity_id_mapping"], label + ".entity_id_mapping")
    normalized = []
    media_ids: Set[str] = set()
    for index, raw in enumerate(mappings):
        mapping = _closed(raw, {
            "row_entity_id", "media_entity_id", "semantic_role", "required_transition_or_effect",
        }, "%s.entity_id_mapping[%d]" % (label, index))
        _text(mapping["row_entity_id"], label + ".mapping.row_entity_id")
        media_id = _text(mapping["media_entity_id"], label + ".mapping.media_entity_id")
        if media_id in media_ids:
            fail(label + ".entity_id_mapping reuses media_entity_id")
        media_ids.add(media_id)
        if mapping["semantic_role"] not in CAUSAL_ROLES:
            fail(label + ".entity_id_mapping semantic_role differs")
        if mapping["required_transition_or_effect"] is not None:
            _text(mapping["required_transition_or_effect"], label + ".mapping.required_transition_or_effect")
        normalized.append({
            "entity_id": mapping["row_entity_id"],
            "semantic_role": mapping["semantic_role"],
            "required_transition_or_effect": mapping["required_transition_or_effect"],
        })
    if normalized != _causal_participants(row["instruction"]):
        fail(label + ".entity_id_mapping does not map every causal participant exactly once")

    artifacts = _closed(receipt["artifacts"], ARTIFACT_KEYS, label + ".artifacts")
    for key in sorted(ARTIFACT_KEYS):
        receipt_pin = _artifact_pin(artifacts[key], label + ".artifacts." + key, verify_files)
        if receipt_pin != annotation[key]:
            fail(label + ".artifacts." + key + " does not join annotation pin")
    for key in ("visibility_confidence_abi", "phase_abi", "latent_grid_resize_abi"):
        _artifact_pin(receipt[key], label + "." + key, verify_files)
    _hex(receipt["receipt_digest"], label + ".receipt_digest")
    if canonical_digest(receipt, "receipt_digest") != receipt["receipt_digest"]:
        fail(label + ".receipt_digest differs")


def validate_row(value: Any, split: str, index: int, verify_files: bool) -> Mapping[str, Any]:
    if type(verify_files) is not bool:
        fail("verify_files must be a built-in bool")
    label = "%s.rows[%d]" % (split, index)
    row = _closed(value, {
        "schema_version", "row_id", "stratum", "source", "instruction",
        "edited_target", "action_anchors", "matched_negatives", "media_annotations",
        "contract_tags", "row_digest",
    }, label)
    if row["schema_version"] != ROW_SCHEMA:
        fail(label + " schema differs")
    _hex(row["row_id"], label + ".row_id")
    if row["stratum"] not in STRATA:
        fail(label + ".stratum differs")
    source = _validate_source(row["source"], label + ".source", verify_files)
    _validate_instruction(row["instruction"], source, label + ".instruction")
    _validate_stratum_semantics(row, label)
    target, target_receipt = _validate_target(row["edited_target"], label + ".edited_target", verify_files)
    anchors = _sequence(row["action_anchors"], label + ".action_anchors")
    anchor_pairs = [_validate_anchor(x, "%s.action_anchors[%d]" % (label, i), verify_files) for i, x in enumerate(anchors)]
    parsed_anchors = [x[0] for x in anchor_pairs]
    if not any(x["compatibility"] == "accept" for x in parsed_anchors):
        fail(label + " needs at least one accepted action anchor")
    media_role_shas = [source["sha256"], target["sha256"]] + [x["sha256"] for x in parsed_anchors]
    if len(set(media_role_shas)) != len(media_role_shas):
        fail(label + " source/edited-target/action-anchor media must be byte-disjoint")
    negatives = _closed(row["matched_negatives"], NEGATIVE_KEYS, label + ".matched_negatives")
    for key, item in negatives.items():
        if item is not None:
            _text(item, label + ".matched_negatives." + key)
    annotation_pairs = [_validate_annotation(x, "%s.media_annotations[%d]" % (label, i), verify_files) for i, x in enumerate(_sequence(row["media_annotations"], label + ".media_annotations"))]
    annotations = [x[0] for x in annotation_pairs]
    required_annotations = {(source["sha256"], "source"), (target["sha256"], "edited_target")}
    required_annotations.update((x["sha256"], "action_anchor") for x in parsed_anchors if x["compatibility"] == "accept")
    observed_annotation_keys = [(x["media_sha256"], x["media_role"]) for x in annotations]
    if len(set(observed_annotation_keys)) != len(observed_annotation_keys):
        fail(label + " duplicates a per-media annotation")
    if set(observed_annotation_keys) != required_annotations:
        fail(label + " per-media annotations must equal source/target/accepted-anchor set")
    coordinate_spaces = [x["coordinate_space"] for x in annotations]
    if len(set(coordinate_spaces)) != len(coordinate_spaces):
        fail(label + " must name a distinct native coordinate space for each media")
    annotation_by_key = {key: pair for key, pair in zip(observed_annotation_keys, annotation_pairs)}

    all_annotation_pins = []
    all_receipt_pins = [
        (target["qualification_receipt_path"], target["qualification_receipt_sha256"]),
    ]
    for anchor in parsed_anchors:
        all_receipt_pins.append((anchor["compatibility_receipt_path"], anchor["compatibility_receipt_sha256"]))
    for annotation in annotations:
        all_receipt_pins.append((annotation["annotation_receipt_path"], annotation["annotation_receipt_sha256"]))
        all_annotation_pins.extend((annotation[key]["path"], annotation[key]["sha256"]) for key in ARTIFACT_KEYS)
    receipt_paths = [x[0] for x in all_receipt_pins]
    receipt_shas = [x[1] for x in all_receipt_pins]
    annotation_paths = [x[0] for x in all_annotation_pins]
    annotation_shas = [x[1] for x in all_annotation_pins]
    if len(set(receipt_paths)) != len(receipt_paths) or len(set(receipt_shas)) != len(receipt_shas):
        fail(label + " reuses a qualification/compatibility/annotation receipt")
    if len(set(annotation_paths)) != len(annotation_paths) or len(set(annotation_shas)) != len(annotation_shas):
        fail(label + " reuses annotation artifact bytes or paths across media/roles")

    if verify_files:
        for annotation_index, (annotation, annotation_receipt) in enumerate(annotation_pairs):
            _validate_annotation_receipt(
                annotation_receipt, row, annotation,
                "%s.media_annotations[%d].annotation_receipt" % (label, annotation_index), verify_files,
            )
        target_annotation, target_annotation_receipt = annotation_by_key[(target["sha256"], "edited_target")]
        _validate_target_receipt(
            target_receipt, row, target, target_annotation, target_annotation_receipt,
            label + ".edited_target.qualification_receipt", verify_files,
        )
        for anchor_index, (anchor, anchor_receipt) in enumerate(anchor_pairs):
            annotation = None
            if anchor["compatibility"] == "accept":
                annotation = annotation_by_key[(anchor["sha256"], "action_anchor")][0]
            _validate_anchor_receipt(
                anchor_receipt, row, anchor, annotation,
                "%s.action_anchors[%d].compatibility_receipt" % (label, anchor_index), verify_files,
            )
    tags = _closed(row["contract_tags"], {"occlusion_or_blocking", "long_horizon"}, label + ".contract_tags")
    _bool(tags["occlusion_or_blocking"], label + ".contract_tags.occlusion_or_blocking")
    _bool(tags["long_horizon"], label + ".contract_tags.long_horizon")
    if row["stratum"] == "multi_entity_long_horizon" and tags["long_horizon"] is not True:
        fail(label + " long-horizon stratum lacks tag")
    _hex(row["row_digest"], label + ".row_digest")
    if canonical_digest(row, "row_digest") != row["row_digest"]:
        fail(label + " row digest differs")
    _forbidden_key_scan(row, label)
    return row


def validate_manifest(value: Mapping[str, Any], expected_split: str, verify_files: bool = True) -> Mapping[str, Any]:
    if type(verify_files) is not bool:
        fail("verify_files must be a built-in bool")
    manifest = _closed(value, {"schema_version", "split", "optimizer_eligible", "rows", "manifest_digest"}, expected_split)
    if manifest["schema_version"] != SCHEMA or manifest["split"] != expected_split:
        fail(expected_split + " manifest identity differs")
    expected_count = 160 if expected_split == TRAIN_SPLIT else 32
    if manifest["optimizer_eligible"] is not (expected_split == TRAIN_SPLIT):
        fail(expected_split + " optimizer eligibility differs")
    rows = [validate_row(x, expected_split, i, verify_files) for i, x in enumerate(_sequence(manifest["rows"], expected_split + ".rows"))]
    if len(rows) != expected_count:
        fail("%s must contain exact%d rows" % (expected_split, expected_count))
    _hex(manifest["manifest_digest"], expected_split + ".manifest_digest")
    if canonical_digest(manifest, "manifest_digest") != manifest["manifest_digest"]:
        fail(expected_split + " manifest digest differs")

    unique_fields = {
        "row_id": [x["row_id"] for x in rows],
        "source.sha256": [x["source"]["sha256"] for x in rows],
        "source.source_group_id": [x["source"]["source_group_id"] for x in rows],
        "source.scene_id": [x["source"]["scene_id"] for x in rows],
        "edited_target.sha256": [x["edited_target"]["sha256"] for x in rows],
        "instruction.semantic_key": [x["instruction"]["semantic_key"] for x in rows],
        "instruction.composition_semantic_key": [x["instruction"]["composition_semantic_key"] for x in rows],
    }
    for key, items in unique_fields.items():
        if len(set(items)) != len(items):
            fail(expected_split + " duplicates " + key)
    entity_groups = [g for x in rows for g in x["source"]["entity_instance_group_ids"]]
    anchor_groups = [a["teacher_candidate_group_id"] for x in rows for a in x["action_anchors"] if a["compatibility"] == "accept"]
    accepted_anchor_shas = [a["sha256"] for x in rows for a in x["action_anchors"] if a["compatibility"] == "accept"]
    if (len(set(entity_groups)) != len(entity_groups)
            or len(set(anchor_groups)) != len(anchor_groups)
            or len(set(accepted_anchor_shas)) != len(accepted_anchor_shas)):
        fail(expected_split + " reuses entity or accepted-anchor groups")

    annotation_artifact_paths = [
        annotation[key]["path"] for row in rows for annotation in row["media_annotations"]
        for key in ARTIFACT_KEYS
    ]
    annotation_artifact_shas = [
        annotation[key]["sha256"] for row in rows for annotation in row["media_annotations"]
        for key in ARTIFACT_KEYS
    ]
    annotation_receipt_paths = [
        annotation["annotation_receipt_path"] for row in rows for annotation in row["media_annotations"]
    ]
    annotation_receipt_shas = [
        annotation["annotation_receipt_sha256"] for row in rows for annotation in row["media_annotations"]
    ]
    for name, values in (
        ("annotation artifact path", annotation_artifact_paths),
        ("annotation artifact SHA", annotation_artifact_shas),
        ("annotation receipt path", annotation_receipt_paths),
        ("annotation receipt SHA", annotation_receipt_shas),
    ):
        if len(set(values)) != len(values):
            fail(expected_split + " reuses " + name)

    strata = Counter(x["stratum"] for x in rows)
    families = Counter(x["instruction"]["action_family"] for x in rows)
    if expected_split == TRAIN_SPLIT:
        if strata != Counter({x: 40 for x in STRATA}):
            fail("train strata must be exact40 each")
        if len(families) < 20 or max(families.values()) > 8:
            fail("train action-family coverage differs")
    else:
        if strata != Counter({x: 8 for x in STRATA}):
            fail(expected_split + " strata must be exact8 each")
        if sum(strata[x] for x in INTERACTION_STRATA) != 24:
            fail(expected_split + " must contain exact24 interaction rows")
        if len(families) < 12 or max(families.values()) > 4:
            fail(expected_split + " action-family coverage differs")
        if sum(bool(x["contract_tags"]["occlusion_or_blocking"]) for x in rows) < 8:
            fail(expected_split + " needs at least8 occlusion/blocking rows")
        if sum(bool(x["contract_tags"]["long_horizon"]) for x in rows) != 8:
            fail(expected_split + " needs exact8 long-horizon rows")
    return manifest


def _membership_keys(manifest: Mapping[str, Any]) -> Dict[str, Set[str]]:
    rows = manifest["rows"]
    return {
        "source_sha": {x["source"]["sha256"] for x in rows},
        "source_group": {x["source"]["source_group_id"] for x in rows},
        "scene": {x["source"]["scene_id"] for x in rows},
        "entity_group": {g for x in rows for g in x["source"]["entity_instance_group_ids"]},
        "target_sha": {x["edited_target"]["sha256"] for x in rows},
        "anchor_group": {a["teacher_candidate_group_id"] for x in rows for a in x["action_anchors"]},
        "anchor_sha": {a["sha256"] for x in rows for a in x["action_anchors"]},
        "composition": {x["instruction"]["composition_semantic_key"] for x in rows},
        "annotation_artifact_sha": {
            annotation[key]["sha256"] for x in rows for annotation in x["media_annotations"]
            for key in ARTIFACT_KEYS
        },
    }


def require_disjoint(manifests: Sequence[Tuple[str, Mapping[str, Any]]]) -> None:
    projected = [(label, _membership_keys(value)) for label, value in manifests]
    for left_index, (left_label, left) in enumerate(projected):
        for right_label, right in projected[left_index + 1:]:
            for key in left:
                overlap = left[key] & right[key]
                if overlap:
                    fail("%s/%s overlap on %s" % (left_label, right_label, key))


def _validate_encoder_authority(value: Any, label: str, verify_files: bool) -> Mapping[str, Any]:
    authority = _closed(value, {
        "interface_family", "implementation_version", "implementation_sha256",
        "weights_path", "weights_sha256", "e_anchor_wrapper_sha256",
        "deterministic_initial_carrier_sha256", "canonicalizer_sha256",
        "participant_locator_schema_sha256",
    }, label)
    if authority["interface_family"] != "ELAL-3":
        fail(label + ".interface_family must be ELAL-3")
    _text(authority["implementation_version"], label + ".implementation_version")
    for key in (
        "implementation_sha256", "e_anchor_wrapper_sha256",
        "deterministic_initial_carrier_sha256", "canonicalizer_sha256",
        "participant_locator_schema_sha256",
    ):
        _hex(authority[key], label + "." + key)
    _stable_file(authority["weights_path"], authority["weights_sha256"], label + ".weights", verify_files)
    return authority


def _validate_qualification_row(value: Any, index: int, verify_files: bool) -> Mapping[str, Any]:
    label = "action_encoder_qualification.rows[%d]" % index
    row = _closed(value, {
        "row_id", "stratum", "source_sha", "source_group", "scene", "entity_group_ids",
        "target_sha", "anchor_group", "composition", "semantic_key", "covered_axes",
        "verdict", "evidence",
    }, label)
    for key in ("row_id", "source_sha", "target_sha", "composition"):
        _hex(row[key], label + "." + key)
    if row["stratum"] not in STRATA:
        fail(label + ".stratum differs")
    for key in ("source_group", "scene", "anchor_group", "semantic_key"):
        _text(row[key], label + "." + key)
    groups = _sequence(row["entity_group_ids"], label + ".entity_group_ids")
    if not 1 <= len(groups) <= 3 or any(type(x) is not str or not x for x in groups) or len(set(groups)) != len(groups):
        fail(label + ".entity_group_ids must contain 1..3 unique IDs")
    axes = _sequence(row["covered_axes"], label + ".covered_axes")
    if not axes or any(type(x) is not str or x not in QUALIFICATION_AXIS_KEYS for x in axes) or len(set(axes)) != len(axes):
        fail(label + ".covered_axes must be a nonempty unique subset of the frozen axes")
    if row["verdict"] != "PASS":
        fail(label + ".verdict must be PASS")
    _artifact_pin(row["evidence"], label + ".evidence", verify_files)
    return row


def _qualification_membership_keys(qualification: Mapping[str, Any]) -> Dict[str, Set[str]]:
    rows = qualification["rows"]
    return {
        "source_sha": {x["source_sha"] for x in rows},
        "source_group": {x["source_group"] for x in rows},
        "scene": {x["scene"] for x in rows},
        "entity_group": {g for x in rows for g in x["entity_group_ids"]},
        "target_sha": {x["target_sha"] for x in rows},
        "anchor_group": {x["anchor_group"] for x in rows},
        "composition": {x["composition"] for x in rows},
    }


def validate_action_encoder_qualification(
    value: Mapping[str, Any], verify_files: bool = True,
) -> Mapping[str, Any]:
    if type(verify_files) is not bool:
        fail("verify_files must be a built-in bool")
    qualification = _closed(value, {
        "schema_version", "row_count", "encoder_authority", "training_corpus",
        "weights_frozen_at_utc", "exact160_membership_revealed_at_utc",
        "frozen_before_exact160_membership_reveal", "student_joint_update_forbidden",
        "thresholds_frozen_at_utc", "qualification_labels_revealed_at_utc",
        "thresholds_frozen_before_qualification_labels_reveal", "rows", "metrics",
        "thresholds", "verdict", "qualification_digest",
    }, "action_encoder_qualification")
    if qualification["schema_version"] != QUALIFICATION_SCHEMA:
        fail("action encoder qualification schema differs")
    if type(qualification["row_count"]) is not int or qualification["row_count"] != 64:
        fail("action encoder qualification count must be built-in exact64")
    if (qualification["frozen_before_exact160_membership_reveal"] is not True
            or qualification["student_joint_update_forbidden"] is not True
            or qualification["thresholds_frozen_before_qualification_labels_reveal"] is not True
            or qualification["verdict"] != "GO"):
        fail("action encoder qualification is not GO/frozen/non-joint")
    frozen_at = _utc_timestamp(qualification["weights_frozen_at_utc"], "qualification.weights_frozen_at_utc")
    revealed_at = _utc_timestamp(qualification["exact160_membership_revealed_at_utc"], "qualification.exact160_membership_revealed_at_utc")
    if frozen_at >= revealed_at:
        fail("action encoder weights were not frozen before exact160 membership reveal")
    thresholds_frozen_at = _utc_timestamp(
        qualification["thresholds_frozen_at_utc"], "qualification.thresholds_frozen_at_utc",
    )
    labels_revealed_at = _utc_timestamp(
        qualification["qualification_labels_revealed_at_utc"],
        "qualification.qualification_labels_revealed_at_utc",
    )
    if thresholds_frozen_at >= labels_revealed_at:
        fail("action encoder thresholds were not frozen before qualification labels reveal")
    _validate_encoder_authority(qualification["encoder_authority"], "qualification.encoder_authority", verify_files)

    corpus = _closed(qualification["training_corpus"], {
        "manifest_path", "manifest_sha256", "provenance",
        "provenance_manifest_path", "provenance_manifest_sha256",
        "disjointness_receipt_path", "disjointness_receipt_sha256",
    }, "qualification.training_corpus")
    _stable_file(corpus["manifest_path"], corpus["manifest_sha256"], "qualification.training_corpus.manifest", verify_files)
    if corpus["provenance"] not in {"real", "licensed", "simulator", "mixed-qualified"}:
        fail("qualification.training_corpus.provenance differs")
    _stable_file(
        corpus["provenance_manifest_path"], corpus["provenance_manifest_sha256"],
        "qualification.training_corpus.provenance_manifest", verify_files,
    )
    _stable_json_file(
        corpus["disjointness_receipt_path"], corpus["disjointness_receipt_sha256"],
        "qualification.training_corpus.disjointness_receipt", verify_files,
    )

    rows = [_validate_qualification_row(x, index, verify_files) for index, x in enumerate(
        _sequence(qualification["rows"], "qualification.rows")
    )]
    if len(rows) != 64:
        fail("action encoder qualification must contain exact64 rows")
    if Counter(x["stratum"] for x in rows) != Counter({x: 16 for x in STRATA}):
        fail("action encoder qualification strata must be exact16 each")
    unique_fields = {
        "row_id": [x["row_id"] for x in rows],
        "source_sha": [x["source_sha"] for x in rows],
        "source_group": [x["source_group"] for x in rows],
        "scene": [x["scene"] for x in rows],
        "target_sha": [x["target_sha"] for x in rows],
        "anchor_group": [x["anchor_group"] for x in rows],
        "composition": [x["composition"] for x in rows],
        "semantic_key": [x["semantic_key"] for x in rows],
        "evidence.path": [x["evidence"]["path"] for x in rows],
        "evidence.sha256": [x["evidence"]["sha256"] for x in rows],
    }
    for key, values in unique_fields.items():
        if len(set(values)) != len(values):
            fail("action encoder qualification reuses " + key)
    entity_groups = [g for x in rows for g in x["entity_group_ids"]]
    if len(set(entity_groups)) != len(entity_groups):
        fail("action encoder qualification reuses entity groups")
    covered_axes = {axis for x in rows for axis in x["covered_axes"]}
    if covered_axes != QUALIFICATION_AXIS_KEYS:
        fail("action encoder qualification does not cover every frozen axis")

    metrics = _closed(qualification["metrics"], set(QUALIFICATION_METRIC_THRESHOLDS), "qualification.metrics")
    thresholds = _closed(qualification["thresholds"], set(QUALIFICATION_METRIC_THRESHOLDS), "qualification.thresholds")
    for key, minimum in QUALIFICATION_METRIC_THRESHOLDS.items():
        threshold = _number(thresholds[key], "qualification.thresholds." + key)
        metric = _number(metrics[key], "qualification.metrics." + key)
        if threshold > 1.0 or metric > 1.0:
            fail("qualification metric/threshold %s must be <= 1" % key)
        if threshold < minimum or metric < threshold:
            fail("qualification metric/threshold %s misses frozen gate %.2f" % (key, minimum))
    _hex(qualification["qualification_digest"], "qualification.qualification_digest")
    if canonical_digest(qualification, "qualification_digest") != qualification["qualification_digest"]:
        fail("qualification digest differs")
    return qualification


def require_qualification_disjoint(qualification: Mapping[str, Any], manifests: Sequence[Tuple[str, Mapping[str, Any]]]) -> None:
    q = _qualification_membership_keys(qualification)
    for label, manifest in manifests:
        m = _membership_keys(manifest)
        for key in q:
            if q[key] & m[key]:
                fail("action encoder qualification overlaps %s on %s" % (label, key))


def _validate_representation_disjointness_receipt(
    receipt: Mapping[str, Any], qualification: Mapping[str, Any],
    manifests: Sequence[Tuple[str, Mapping[str, Any]]], label: str,
) -> None:
    receipt = _closed(receipt, {
        "schema_version", "training_corpus_manifest_sha256",
        "training_corpus_provenance_manifest_sha256",
        "qualification_membership_digest", "compared_manifest_digests",
        "overlap_counts", "verifier_authority", "verdict", "receipt_digest",
    }, label)
    if receipt["schema_version"] != REPRESENTATION_DISJOINTNESS_SCHEMA:
        fail(label + ".schema_version differs")
    if receipt["training_corpus_manifest_sha256"] != qualification["training_corpus"]["manifest_sha256"]:
        fail(label + " does not join the representation training corpus manifest")
    if receipt["training_corpus_provenance_manifest_sha256"] != qualification["training_corpus"]["provenance_manifest_sha256"]:
        fail(label + " does not join the representation corpus provenance manifest")
    if receipt["qualification_membership_digest"] != canonical_value_digest(qualification["rows"]):
        fail(label + " does not join qualification membership")
    expected_digests = {name: manifest["manifest_digest"] for name, manifest in manifests}
    compared = _closed(receipt["compared_manifest_digests"], set(expected_digests), label + ".compared_manifest_digests")
    if compared != expected_digests:
        fail(label + " does not join the three exact160 manifests")
    dataset_keys = set(expected_digests) | {"action_encoder_qualification"}
    overlap_counts = _closed(receipt["overlap_counts"], dataset_keys, label + ".overlap_counts")
    membership_keys = {"source_sha", "source_group", "scene", "entity_group", "target_sha", "anchor_group", "composition"}
    for dataset, raw in overlap_counts.items():
        counts = _closed(raw, membership_keys, label + ".overlap_counts." + dataset)
        for key, count in counts.items():
            if type(count) is not int or count != 0:
                fail("%s overlap count %s.%s must be built-in zero" % (label, dataset, key))
    _validate_review_authority(receipt["verifier_authority"], label + ".verifier_authority")
    if receipt["verdict"] != "GO":
        fail(label + ".verdict must be GO")
    _hex(receipt["receipt_digest"], label + ".receipt_digest")
    if canonical_digest(receipt, "receipt_digest") != receipt["receipt_digest"]:
        fail(label + ".receipt_digest differs")


def validate_bundle(
    train: Mapping[str, Any], calibration: Mapping[str, Any], locked: Mapping[str, Any],
    qualification: Mapping[str, Any], verify_files: bool = True,
) -> Mapping[str, Any]:
    if type(verify_files) is not bool:
        fail("verify_files must be a built-in bool")
    train = validate_manifest(train, TRAIN_SPLIT, verify_files)
    calibration = validate_manifest(calibration, CALIBRATION_SPLIT, verify_files)
    locked = validate_manifest(locked, LOCKED_SPLIT, verify_files)
    qualification = validate_action_encoder_qualification(qualification, verify_files)
    manifests = [(TRAIN_SPLIT, train), (CALIBRATION_SPLIT, calibration), (LOCKED_SPLIT, locked)]
    require_disjoint(manifests)
    require_qualification_disjoint(qualification, manifests)
    if verify_files:
        corpus = qualification["training_corpus"]
        disjointness_receipt = _stable_json_file(
            corpus["disjointness_receipt_path"], corpus["disjointness_receipt_sha256"],
            "qualification.training_corpus.disjointness_receipt", True,
        )
        _validate_representation_disjointness_receipt(
            disjointness_receipt, qualification, manifests,
            "qualification.training_corpus.disjointness_receipt",
        )
    train_families = {x["instruction"]["action_family"] for x in train["rows"]}
    locked_unseen = [x for x in locked["rows"] if x["instruction"]["action_family"] not in train_families]
    if len(locked_unseen) < 8 or len({x["instruction"]["action_family"] for x in locked_unseen}) < 4:
        fail("locked32 lacks >=8 rows from >=4 train-unseen action families")
    receipt = {
        "schema_version": PREFLIGHT_RECEIPT_SCHEMA,
        "status": "STRUCTURAL_PREFLIGHT_ONLY",
        "blocking_reason": "CANONICAL_IDENTITY_NOT_FROZEN",
        "data_and_encoder_admission_go": False,
        "stable_file_bytes_verified": bool(verify_files),
        "closed_receipt_schema_and_internal_joins_verified": bool(verify_files),
        "canonical_identity_recomputed": False,
        "independent_receipt_claims_established": False,
        "r1_representation_go": False,
        "r2_action_encoder_go": False,
        "formal_training_authorized": False,
        "requires_user_approval_after_overfit_video": True,
        "train_rows": 160,
        "calibration_rows": 32,
        "locked_rows": 32,
        "qualification_rows": 64,
        "train_manifest_digest": train["manifest_digest"],
        "calibration_manifest_digest": calibration["manifest_digest"],
        "locked_manifest_digest": locked["manifest_digest"],
        "action_encoder_qualification_digest": qualification["qualification_digest"],
        "action_encoder_weights_sha256": qualification["encoder_authority"]["weights_sha256"],
        "representation_corpus_manifest_sha256": qualification["training_corpus"]["manifest_sha256"],
        "representation_corpus_provenance_manifest_sha256": qualification["training_corpus"]["provenance_manifest_sha256"],
        "representation_disjointness_receipt_sha256": qualification["training_corpus"]["disjointness_receipt_sha256"],
        "receipt_abi_versions": {
            "target": TARGET_RECEIPT_SCHEMA,
            "anchor": ANCHOR_RECEIPT_SCHEMA,
            "annotation": ANNOTATION_RECEIPT_SCHEMA,
            "representation_disjointness": REPRESENTATION_DISJOINTNESS_SCHEMA,
        },
        "canonical_identity_authority_requirement": {
            "schema_version": CANONICAL_IDENTITY_AUTHORITY_REQUIREMENT_SCHEMA,
            "status": "NOT_FROZEN",
            "required_identity_outputs": [
                "row_id", "instruction.semantic_key",
                "instruction.composition_semantic_key",
            ],
            "required_closed_authority_fields": [
                "schema_version",
                "row_identity_canonicalizer_path",
                "row_identity_canonicalizer_sha256",
                "semantic_key_canonicalizer_path",
                "semantic_key_canonicalizer_sha256",
                "composition_semantic_key_canonicalizer_path",
                "composition_semantic_key_canonicalizer_sha256",
                "qualification_receipt_path",
                "qualification_receipt_sha256",
                "authority_digest",
            ],
        },
        "declared_but_not_independently_established": [
            "action_encoder_qualification.verdict",
            "representation_training_corpus_disjointness.verdict",
            "canonical semantic uniqueness",
            "canonical composition uniqueness",
            "canonical row identity",
        ],
        "optimizer_target_role": "clean_edited_target_only",
        "action_anchor_role": "action-reference-only-canonical-prototype",
        "full644_role": "candidate_catalog_only",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, "receipt_digest")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--locked", required=True)
    parser.add_argument("--action-encoder-qualification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute() or output.exists():
        fail("output must be a fresh absolute path")
    values = [load_json_strict(Path(x)) for x in (args.train, args.calibration, args.locked, args.action_encoder_qualification)]
    receipt = validate_bundle(values[0], values[1], values[2], values[3], verify_files=True)
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = output.open("xb")
    try:
        descriptor.write(raw)
        descriptor.flush()
    finally:
        descriptor.close()
    output.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "output": str(output), "sha256": file_sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prospective same-video wrong-action-family prompt-swap pilot.

This module deliberately does not modify or relabel the sealed PAIR core4-v2
bank.  It copies the ten already sealed generation captions for two fresh-seed
confirmation cells, while keeping family A, family B, and one common neutral
query prompt as byte/hash-bound *critic queries*.  Query prompts never become
T2V generation captions.

The protocol has four fail-closed stages:

1. authenticate the registry and the old core4-v2 caption/geometry authority;
2. scan every JSON file in the caller-declared full-bank inventory and reject
   any occurrence of either fresh seed outside the authenticated registry;
3. after rendering, bind all ten branch receipts to one official-Gaussian raw
   and content digest per cell; and
4. require exactly 24 prompt-blind full-exact81 family judgments.  ``unknown``
   or ``ambiguous`` invalidates the whole prospective tuple.

Passing this pilot is never editor evidence, a scientific-critic result, or
optimizer authorization.  Generated media remain critic/audit evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import pair_v5_t2v_calibration_bank_spec as bank_contract  # noqa: E402


REGISTRY_SCHEMA = "bernini-starc-wrong-family-prompt-swap-registry-v1"
SEED_AUDIT_SCHEMA = "bernini-starc-wrong-family-seed-collision-audit-v1"
GENERATION_PLAN_SCHEMA = "bernini-starc-wrong-family-generation-plan-v1"
GAUSSIAN_BINDING_SCHEMA = "bernini-starc-wrong-family-gaussian-binding-v1"
AUDIT_PLAN_SCHEMA = "bernini-starc-wrong-family-detached-audit-plan-v1"
COMPLETED_AUDIT_SCHEMA = "bernini-starc-wrong-family-completed-audit-v1"
RESULT_RECEIPT_SCHEMA = "bernini-starc-wrong-family-result-receipt-v1"

REGISTRY_ASSET_BASENAME = "wrong_family_prompt_swap_pilot_registry_v1.json"
REGISTRY_RAW_SHA256 = "f8978311ee7db7f524e827b49747f74ec1a5e0d568e2bbada3fd212225f20cff"
SOURCE_BANK_BASENAME = "pair_v5_t2v_calibration_core4_bank_v2.json"
SOURCE_BANK_RAW_SHA256 = "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95"
PILOT_ID = "starc-family-swap-prospective-p1"

FRESH_SEEDS = {
    "841b5e0080a1441d": 2026081301,
    "a66e6818e4144928": 2026081302,
}
SOURCE_SEEDS = {
    "7b88a1ca1f804f41": 2026080825,
    "841b5e0080a1441d": 2026080826,
    "a35b590961d24694": 2026080827,
    "a66e6818e4144928": 2026080828,
}
SOURCE_ACTION_FAMILY_BY_IID = {
    "7b88a1ca1f804f41": "dog-sit-facing-camera",
    "841b5e0080a1441d": "dog-sit-facing-camera",
    "a35b590961d24694": "human-rise-to-stand",
    "a66e6818e4144928": "human-rise-to-stand",
}
CELL_SEAL = {
    "7b88a1ca1f804f41": {
        "cell_id": "dog-fit-7b88-retrospective",
        "actor_topology_id": "single-quadruped-dog",
        "role": "retrospective_discovery",
        "a_family_id": "dog-stand-to-sit-facing-camera",
        "b_family_id": "dog-sit-to-stand-facing-camera",
    },
    "a35b590961d24694": {
        "cell_id": "human-fit-a35b-retrospective",
        "actor_topology_id": "single-fullbody-human-one-knee",
        "role": "retrospective_discovery",
        "a_family_id": "human-one-knee-to-upright-stand",
        "b_family_id": "human-upright-stand-to-one-knee",
    },
    "841b5e0080a1441d": {
        "cell_id": "dog-confirm-841b-fresh",
        "actor_topology_id": "single-quadruped-dog",
        "role": "prospective_confirmation",
        "a_family_id": "dog-stand-to-sit-facing-camera",
        "b_family_id": "dog-sit-to-stand-facing-camera",
    },
    "a66e6818e4144928": {
        "cell_id": "human-confirm-a66e-fresh",
        "actor_topology_id": "single-fullbody-human-low-crouch",
        "role": "prospective_confirmation",
        "a_family_id": "human-low-crouch-to-upright-stand",
        "b_family_id": "human-upright-stand-to-low-crouch",
    },
}
RETROSPECTIVE_IIDS = ("7b88a1ca1f804f41", "a35b590961d24694")
PROSPECTIVE_IIDS = ("841b5e0080a1441d", "a66e6818e4144928")
BRANCH_ORDER = tuple(bank_contract.MACE_BRANCH_ORDER)
FAMILY_B_BRANCH_ORDER = ("action", "reverse")

INTERPRETATION_CONTRACT = {
    "editor": False,
    "scientific_critic": False,
    "optimizer_authorized": False,
    "fresh_confirmation_enters_optimizer": False,
    "retrospective_media_can_fit_or_confirm": False,
    "generated_media_role": "critic_query_and_detached_family_audit_only",
    "generated_media_as_editor_target_condition_reference_noise_or_donor": False,
    "passing_tuple_role": "pilot_evidence_only_cannot_authorize_editor_backward",
}
SEED_PREFLIGHT_CONTRACT = {
    "scope": "all_known_bernini_t2v_bank_specs_and_receipts",
    "scan_before_any_fresh_render": True,
    "scan_all_json_scalar_occurrences": True,
    "registry_file_is_the_only_allowed_occurrence": True,
    "collision_policy": "fail_entire_tuple_no_seed_replacement",
    "on_site_seed_swap": False,
}
GAUSSIAN_BINDING_CONTRACT = {
    "required_after_render_before_query_or_audit": True,
    "tensor_key": "official_initial_gaussian",
    "stored_dtype": "torch.float32",
    "all_ten_branches_share_raw_and_content_digest": True,
    "different_fresh_cells_must_have_different_raw_digest": True,
    "integer_seed_without_tensor_digest_is_sufficient": False,
}
FAMILY_A_REQUIRED = {branch: branch == "action" for branch in BRANCH_ORDER}
FAMILY_B_REQUIRED = {"action": False, "reverse": True}
AUDIT_CONTRACT = {
    "audit_key": ["candidate_id", "evaluated_family_id"],
    "family_a_branch_order": list(BRANCH_ORDER),
    "family_a_required_outcome_by_branch": FAMILY_A_REQUIRED,
    "family_b_branch_order": list(FAMILY_B_BRANCH_ORDER),
    "family_b_required_outcome_by_branch": FAMILY_B_REQUIRED,
    "judgments_per_fresh_cell": 12,
    "total_fresh_judgments": 24,
    "full_exact81_required": True,
    "generation_prompt_hidden_from_auditor": True,
    "unknown_or_ambiguous_policy": "fail_entire_prospective_tuple",
    "common_null_policy": "same_cell_same_utf8_bytes_for_a_and_b",
    "branch_name_is_not_observation": True,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_ROOT_FIELDS = {
    "schema_version",
    "pilot_id",
    "source_bank",
    "sampling_contract",
    "interpretation_contract",
    "seed_preflight_contract",
    "gaussian_binding_contract",
    "audit_contract",
    "cells",
}
_SOURCE_BANK_FIELDS = {"asset_basename", "raw_sha256", "schema_version"}
_CELL_FIELDS = {
    "cell_id",
    "iid",
    "actor_topology_id",
    "source_seed",
    "source_media_seen_before_seal",
    "role",
    "fresh_seed",
    "fresh_media_status",
    "source_result_use",
    "a_family_id",
    "b_family_id",
    "query_prompts",
    "family_rubrics",
}
_PROMPT_FIELDS = {"utf8_text", "utf8_sha256"}
_RUBRIC_FIELDS = {
    "rubric_id",
    "evaluated_family_id",
    "start_state",
    "ordered_milestones",
    "terminal_hold",
    "decision_protocol",
}
_PLAN_FIELDS = {
    "schema_version",
    "pilot_id",
    "registry_raw_sha256",
    "source_bank_raw_sha256",
    "sampling_contract",
    "interpretation_contract",
    "seed_collision_audit",
    "retrospective_discovery_exclusions",
    "prospective_cells",
    "candidate_count",
    "judgment_count",
    "query_prompts_are_generation_captions",
    "official_gaussian_binding_status",
    "generation_plan_digest",
}
_PLAN_CELL_FIELDS = {
    "cell_id",
    "iid",
    "fresh_seed",
    "source_cell_binding",
    "generation_candidates",
    "query_prompts",
    "family_rubrics",
    "judgment_requirements",
}
_JUDGMENT_REQUIREMENT_FIELDS = {
    "audit_key",
    "candidate_id",
    "semantic_branch",
    "evaluated_family_id",
    "required_outcome",
    "rubric_id",
    "rubric_sha256",
    "family_prompt_utf8_sha256",
    "common_null_utf8_sha256",
}
_GAUSSIAN_ROOT_FIELDS = {
    "schema_version",
    "generation_plan_digest",
    "bindings",
    "gaussian_binding_digest",
}
_GAUSSIAN_ROW_FIELDS = {
    "candidate_id",
    "seed",
    "candidate_receipt_path",
    "candidate_receipt_sha256",
    "mp4_path",
    "mp4_sha256",
    "official_gaussian_path",
    "official_gaussian_artifact_sha256",
    "raw_value_sha256",
    "content_sha256",
    "tensor_key",
    "shape",
    "dtype",
    "stored_dtype",
    "generator_initial_seed",
    "captured_from_native_sampler",
    "external_initial_noise_injection",
    "source_or_target_derived",
}
_AUDIT_PLAN_FIELDS = {
    "schema_version",
    "pilot_id",
    "generation_plan_digest",
    "gaussian_binding_digest",
    "interpretation_contract",
    "judgments",
    "judgment_count",
    "audit_plan_digest",
}
_AUDIT_JUDGMENT_FIELDS = _JUDGMENT_REQUIREMENT_FIELDS | {
    "mp4_path",
    "mp4_sha256",
    "candidate_receipt_sha256",
    "full_exact81_required",
    "generation_prompt_must_be_hidden",
}
_COMPLETED_ROOT_FIELDS = {
    "schema_version",
    "audit_plan_digest",
    "judgments",
}
_COMPLETED_ROW_FIELDS = {
    "candidate_id",
    "evaluated_family_id",
    "decision",
    "full_exact81_viewed",
    "generation_prompt_hidden",
    "reviewer_id",
    "review_notes",
    "mp4_sha256",
    "rubric_sha256",
}


class WrongFamilyPromptSwapError(RuntimeError):
    """Raised before a non-prospective or incomplete tuple can be used."""


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
        raise WrongFamilyPromptSwapError("value is not canonical finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise WrongFamilyPromptSwapError("non-finite JSON constant is forbidden: %s" % value)


def _reject_duplicate_pairs(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WrongFamilyPromptSwapError("duplicate JSON key: %r" % key)
        result[key] = value
    return result


def loads_strict(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WrongFamilyPromptSwapError("%s is not valid UTF-8 JSON" % label) from error


def _closed(value: Any, fields: Set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise WrongFamilyPromptSwapError(
            "%s keys differ: expected=%r actual=%r" % (label, sorted(fields), actual)
        )
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise WrongFamilyPromptSwapError("%s must be lowercase SHA-256" % label)
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise WrongFamilyPromptSwapError("%s must be a safe identifier" % label)
    return value


def _text(value: Any, label: str, *, minimum_words: int = 4) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise WrongFamilyPromptSwapError("%s must be text without NUL" % label)
    if value != value.strip() or len(value.split()) < minimum_words:
        raise WrongFamilyPromptSwapError("%s is incomplete or has edge whitespace" % label)
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise WrongFamilyPromptSwapError("%s must be boolean" % label)
    return value


def _plain_absolute_file(path_value: Any, label: str) -> Path:
    if not isinstance(path_value, (str, os.PathLike)):
        raise WrongFamilyPromptSwapError("%s must be a path" % label)
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise WrongFamilyPromptSwapError("%s must be an absolute plain file" % label)
    return path


def _validate_prompt(value: Any, label: str) -> Dict[str, str]:
    prompt = _closed(value, _PROMPT_FIELDS, label)
    text = _text(prompt["utf8_text"], label + ".utf8_text", minimum_words=12)
    digest = _sha(prompt["utf8_sha256"], label + ".utf8_sha256")
    if sha256_bytes(text.encode("utf-8")) != digest:
        raise WrongFamilyPromptSwapError("%s UTF-8 digest differs" % label)
    return {"utf8_text": text, "utf8_sha256": digest}


def _validate_rubric(value: Any, family_id: str, label: str) -> Dict[str, Any]:
    rubric = _closed(value, _RUBRIC_FIELDS, label)
    result = {
        "rubric_id": _safe_id(rubric["rubric_id"], label + ".rubric_id"),
        "evaluated_family_id": _safe_id(
            rubric["evaluated_family_id"], label + ".evaluated_family_id"
        ),
        "start_state": _text(rubric["start_state"], label + ".start_state", minimum_words=8),
        "ordered_milestones": rubric["ordered_milestones"],
        "terminal_hold": _text(
            rubric["terminal_hold"], label + ".terminal_hold", minimum_words=8
        ),
        "decision_protocol": _text(
            rubric["decision_protocol"], label + ".decision_protocol", minimum_words=20
        ),
    }
    if result["evaluated_family_id"] != family_id:
        raise WrongFamilyPromptSwapError("%s family binding differs" % label)
    milestones = result["ordered_milestones"]
    if not isinstance(milestones, list) or len(milestones) < 3:
        raise WrongFamilyPromptSwapError("%s requires at least three ordered milestones" % label)
    result["ordered_milestones"] = [
        _text(item, "%s.ordered_milestones[%d]" % (label, index), minimum_words=7)
        for index, item in enumerate(milestones)
    ]
    protocol = result["decision_protocol"].lower()
    for token in ("true only", "false only", "unknown", "ambiguous", "full exact81"):
        if token not in protocol:
            raise WrongFamilyPromptSwapError("%s decision protocol lost %r" % (label, token))
    return result


def validate_registry(value: Any) -> Dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, "registry")
    if root["schema_version"] != REGISTRY_SCHEMA or root["pilot_id"] != PILOT_ID:
        raise WrongFamilyPromptSwapError("registry schema or pilot ID differs")
    source_bank = _closed(root["source_bank"], _SOURCE_BANK_FIELDS, "source_bank")
    if source_bank != {
        "asset_basename": SOURCE_BANK_BASENAME,
        "raw_sha256": SOURCE_BANK_RAW_SHA256,
        "schema_version": bank_contract.SCHEMA_VERSION_V2,
    }:
        raise WrongFamilyPromptSwapError("source bank authority differs")
    if root["sampling_contract"] != bank_contract.SAMPLING_CONTRACT:
        raise WrongFamilyPromptSwapError("sampling is not exact81/40 official-Gaussian T2V")
    if root["interpretation_contract"] != INTERPRETATION_CONTRACT:
        raise WrongFamilyPromptSwapError("editor/scientific interpretation closure differs")
    if root["seed_preflight_contract"] != SEED_PREFLIGHT_CONTRACT:
        raise WrongFamilyPromptSwapError("seed preflight closure differs")
    if root["gaussian_binding_contract"] != GAUSSIAN_BINDING_CONTRACT:
        raise WrongFamilyPromptSwapError("official Gaussian closure differs")
    if root["audit_contract"] != AUDIT_CONTRACT:
        raise WrongFamilyPromptSwapError("24-judgment audit closure differs")
    cells = root["cells"]
    if not isinstance(cells, list) or len(cells) != 4:
        raise WrongFamilyPromptSwapError("registry must contain exactly four sealed cells")
    normalized_cells: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_iids: Set[str] = set()
    for index, raw_cell in enumerate(cells):
        label = "cell[%d]" % index
        cell = _closed(raw_cell, _CELL_FIELDS, label)
        cell_id = _safe_id(cell["cell_id"], label + ".cell_id")
        iid = _safe_id(cell["iid"], label + ".iid")
        if cell_id in seen_ids or iid in seen_iids:
            raise WrongFamilyPromptSwapError("cell IDs and IIDs must be unique")
        seen_ids.add(cell_id)
        seen_iids.add(iid)
        if iid not in SOURCE_SEEDS or cell["source_seed"] != SOURCE_SEEDS[iid]:
            raise WrongFamilyPromptSwapError("%s source seed differs" % label)
        seal = CELL_SEAL[iid]
        for field in (
            "cell_id",
            "actor_topology_id",
            "role",
            "a_family_id",
            "b_family_id",
        ):
            if cell[field] != seal[field]:
                raise WrongFamilyPromptSwapError("%s sealed %s differs" % (label, field))
        if cell["source_media_seen_before_seal"] is not True:
            raise WrongFamilyPromptSwapError("all old core4 media must be declared seen")
        role = cell["role"]
        if iid in RETROSPECTIVE_IIDS:
            if (
                role != "retrospective_discovery"
                or cell["fresh_seed"] is not None
                or cell["fresh_media_status"]
                != "not_applicable_source_media_previously_seen"
                or cell["source_result_use"]
                != "rubric_discovery_only_never_fit_confirmation_or_threshold_selection"
            ):
                raise WrongFamilyPromptSwapError("retrospective cell gained prospective use")
        elif iid in PROSPECTIVE_IIDS:
            if (
                role != "prospective_confirmation"
                or cell["fresh_seed"] != FRESH_SEEDS[iid]
                or cell["fresh_media_status"] != "unrendered_and_unseen_at_registry_seal"
                or cell["source_result_use"]
                != "caption_and_geometry_authority_only_old_seed_media_outcomes_forbidden"
            ):
                raise WrongFamilyPromptSwapError("prospective cell or fixed seed differs")
        else:
            raise WrongFamilyPromptSwapError("unregistered cell IID")
        a_family = _safe_id(cell["a_family_id"], label + ".a_family_id")
        b_family = _safe_id(cell["b_family_id"], label + ".b_family_id")
        if a_family == b_family:
            raise WrongFamilyPromptSwapError("A and B must be distinct action families")
        prompts = _closed(cell["query_prompts"], {"a", "b", "common_null"}, label + ".query_prompts")
        normalized_prompts = {
            key: _validate_prompt(prompts[key], "%s.query_prompts.%s" % (label, key))
            for key in ("a", "b", "common_null")
        }
        if len({item["utf8_sha256"] for item in normalized_prompts.values()}) != 3:
            raise WrongFamilyPromptSwapError("A/B/common-null prompt bytes must be distinct")
        rubrics = _closed(cell["family_rubrics"], {"a", "b"}, label + ".family_rubrics")
        normalized_rubrics = {
            "a": _validate_rubric(rubrics["a"], a_family, label + ".family_rubrics.a"),
            "b": _validate_rubric(rubrics["b"], b_family, label + ".family_rubrics.b"),
        }
        normalized_cells.append(
            {
                "cell_id": cell_id,
                "iid": iid,
                "actor_topology_id": _safe_id(
                    cell["actor_topology_id"], label + ".actor_topology_id"
                ),
                "source_seed": cell["source_seed"],
                "source_media_seen_before_seal": True,
                "role": role,
                "fresh_seed": cell["fresh_seed"],
                "fresh_media_status": cell["fresh_media_status"],
                "source_result_use": cell["source_result_use"],
                "a_family_id": a_family,
                "b_family_id": b_family,
                "query_prompts": normalized_prompts,
                "family_rubrics": normalized_rubrics,
            }
        )
    if tuple(cell["iid"] for cell in normalized_cells[:2]) != RETROSPECTIVE_IIDS:
        raise WrongFamilyPromptSwapError("retrospective cells must be sealed first in fixed order")
    if tuple(cell["iid"] for cell in normalized_cells[2:]) != PROSPECTIVE_IIDS:
        raise WrongFamilyPromptSwapError("fresh confirmation cells must follow fixed order")
    if len(set(FRESH_SEEDS.values())) != 2 or set(FRESH_SEEDS.values()) & set(SOURCE_SEEDS.values()):
        raise WrongFamilyPromptSwapError("fresh seeds alias source seeds")
    return {
        "schema_version": REGISTRY_SCHEMA,
        "pilot_id": PILOT_ID,
        "source_bank": dict(source_bank),
        "sampling_contract": dict(bank_contract.SAMPLING_CONTRACT),
        "interpretation_contract": dict(INTERPRETATION_CONTRACT),
        "seed_preflight_contract": dict(SEED_PREFLIGHT_CONTRACT),
        "gaussian_binding_contract": dict(GAUSSIAN_BINDING_CONTRACT),
        "audit_contract": dict(AUDIT_CONTRACT),
        "cells": normalized_cells,
    }


def load_registry(path_value: Any, expected_sha256: str) -> Tuple[Dict[str, Any], str]:
    path = _plain_absolute_file(path_value, "registry")
    expected = _sha(expected_sha256, "registry expected SHA-256")
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if expected != REGISTRY_RAW_SHA256 or actual != expected:
        raise WrongFamilyPromptSwapError("registry is not the pinned prospective authority")
    return validate_registry(loads_strict(raw, label="registry")), actual


def load_source_bank(path_value: Any, registry: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    path = _plain_absolute_file(path_value, "source bank")
    if path.name != registry["source_bank"]["asset_basename"]:
        raise WrongFamilyPromptSwapError("source bank basename differs")
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != registry["source_bank"]["raw_sha256"] or actual != SOURCE_BANK_RAW_SHA256:
        raise WrongFamilyPromptSwapError("source bank raw SHA-256 differs")
    try:
        bank = bank_contract.validate_root_spec(loads_strict(raw, label="source bank"))
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyPromptSwapError("source bank failed its sealed validator") from error
    return bank, actual


def _iid_from_candidate(candidate: Mapping[str, Any]) -> str:
    group = candidate["calibration_group_id"]
    suffix = "-s%d" % candidate["seed"]
    if not group.startswith("cell-") or not group.endswith(suffix):
        raise WrongFamilyPromptSwapError("source calibration group cannot be decoded")
    return group[len("cell-") : -len(suffix)]


def _source_rows_by_iid(bank: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for group in bank["groups"]:
        for candidate in group["candidates"]:
            iid = _iid_from_candidate(candidate)
            result.setdefault(iid, []).append(candidate)
    if set(result) != set(SOURCE_SEEDS):
        raise WrongFamilyPromptSwapError("source bank cell population differs")
    for iid, rows in result.items():
        if [row["semantic_branch"] for row in rows] != list(BRANCH_ORDER):
            raise WrongFamilyPromptSwapError("source cell %s lost branch order" % iid)
        if {row["seed"] for row in rows} != {SOURCE_SEEDS[iid]}:
            raise WrongFamilyPromptSwapError("source cell %s seed differs" % iid)
    return result


def _walk_json_files(root: Path) -> Iterator[Path]:
    if root.is_symlink():
        raise WrongFamilyPromptSwapError("seed scan root cannot be a symlink")
    if root.is_file():
        if root.suffix.lower() != ".json":
            raise WrongFamilyPromptSwapError("seed scan file root must be JSON")
        yield root
        return
    if not root.is_dir():
        raise WrongFamilyPromptSwapError("seed scan root must exist")
    for directory, names, filenames in os.walk(str(root), followlinks=False):
        current = Path(directory)
        names.sort()
        filenames.sort()
        for name in list(names):
            child = current / name
            if child.is_symlink():
                raise WrongFamilyPromptSwapError("seed scan encountered a directory symlink")
        for name in filenames:
            child = current / name
            if child.is_symlink():
                raise WrongFamilyPromptSwapError("seed scan encountered a file symlink")
            if child.suffix.lower() == ".json":
                mode = child.stat().st_mode
                if not stat.S_ISREG(mode):
                    raise WrongFamilyPromptSwapError("seed scan encountered non-regular JSON")
                yield child


def _fresh_seed_occurrences(value: Any, pointer: str = "") -> Iterator[Tuple[str, int]]:
    fresh = set(FRESH_SEEDS.values())
    if type(value) is int and value in fresh:
        yield pointer or "/", value
    elif isinstance(value, str) and value in {str(seed) for seed in fresh}:
        yield pointer or "/", int(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            for found in _fresh_seed_occurrences(item, pointer + "/" + escaped):
                yield found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            for found in _fresh_seed_occurrences(item, pointer + "/" + str(index)):
                yield found


def build_seed_collision_audit(
    *,
    registry_path: Path,
    registry_raw_sha256: str,
    scan_roots: Sequence[Path],
) -> Dict[str, Any]:
    registry_file = _plain_absolute_file(registry_path, "registry exclusion").resolve()
    if file_sha256(registry_file) != registry_raw_sha256:
        raise WrongFamilyPromptSwapError("registry exclusion digest differs")
    if not scan_roots:
        raise WrongFamilyPromptSwapError("at least one full-bank seed scan root is required")
    resolved_roots: List[Path] = []
    for raw_root in scan_roots:
        root = Path(raw_root)
        if not root.is_absolute() or root == Path("/"):
            raise WrongFamilyPromptSwapError("seed scan roots must be absolute and non-root")
        if root.is_symlink():
            raise WrongFamilyPromptSwapError("seed scan roots cannot be symlinks")
        resolved = root.resolve()
        if resolved in resolved_roots:
            raise WrongFamilyPromptSwapError("seed scan roots must be unique")
        resolved_roots.append(resolved)
    entries: List[Dict[str, Any]] = []
    observed_paths: Set[Path] = set()
    excluded = False
    collisions: List[Dict[str, Any]] = []
    for root in resolved_roots:
        for path in _walk_json_files(root):
            resolved = path.resolve()
            if resolved in observed_paths:
                continue
            observed_paths.add(resolved)
            raw = path.read_bytes()
            digest = sha256_bytes(raw)
            if resolved == registry_file:
                if digest != registry_raw_sha256:
                    raise WrongFamilyPromptSwapError("scanned registry digest differs")
                excluded = True
                registry_value = loads_strict(raw, label="seed inventory registry")
                registry_occurrences = [
                    {"json_pointer": pointer, "seed": seed}
                    for pointer, seed in _fresh_seed_occurrences(registry_value)
                ]
                entries.append(
                    {
                        "path": str(resolved),
                        "sha256": digest,
                        "size_bytes": len(raw),
                        "role": "authenticated_registry_only_allowed_fresh_seed_occurrence",
                        "fresh_seed_occurrences": registry_occurrences,
                    }
                )
                continue
            value = loads_strict(raw, label="seed inventory %s" % resolved)
            occurrences = [
                {"json_pointer": pointer, "seed": seed}
                for pointer, seed in _fresh_seed_occurrences(value)
            ]
            entries.append(
                {
                    "path": str(resolved),
                    "sha256": digest,
                    "size_bytes": len(raw),
                    "role": "scanned_json",
                    "fresh_seed_occurrences": occurrences,
                }
            )
            for occurrence in occurrences:
                collisions.append(
                    {
                        "path": str(resolved),
                        "json_pointer": occurrence["json_pointer"],
                        "seed": occurrence["seed"],
                    }
                )
    if not excluded:
        raise WrongFamilyPromptSwapError(
            "full-bank roots must include the authenticated registry occurrence"
        )
    entries.sort(key=lambda row: row["path"])
    audit = {
        "schema_version": SEED_AUDIT_SCHEMA,
        "scope": SEED_PREFLIGHT_CONTRACT["scope"],
        "scan_roots": [str(path) for path in resolved_roots],
        "registry_path": str(registry_file),
        "registry_raw_sha256": registry_raw_sha256,
        "fresh_seed_by_iid": dict(FRESH_SEEDS),
        "json_file_inventory": entries,
        "json_file_count": len(entries),
        "collisions": collisions,
        "collision_free": not collisions,
    }
    audit["seed_collision_audit_digest"] = sha256_bytes(canonical_json_bytes(audit))
    if collisions:
        raise WrongFamilyPromptSwapError(
            "fresh seed collision found; the tuple is null and on-site replacement is forbidden"
        )
    return audit


def validate_seed_collision_audit(value: Any) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "scope",
        "scan_roots",
        "registry_path",
        "registry_raw_sha256",
        "fresh_seed_by_iid",
        "json_file_inventory",
        "json_file_count",
        "collisions",
        "collision_free",
        "seed_collision_audit_digest",
    }
    audit = dict(_closed(value, fields, "seed collision audit"))
    digest = _sha(audit.pop("seed_collision_audit_digest"), "seed audit digest")
    if sha256_bytes(canonical_json_bytes(audit)) != digest:
        raise WrongFamilyPromptSwapError("seed collision audit digest differs")
    audit["seed_collision_audit_digest"] = digest
    if (
        audit["schema_version"] != SEED_AUDIT_SCHEMA
        or audit["scope"] != SEED_PREFLIGHT_CONTRACT["scope"]
        or audit["registry_raw_sha256"] != REGISTRY_RAW_SHA256
        or audit["fresh_seed_by_iid"] != FRESH_SEEDS
        or audit["collision_free"] is not True
        or audit["collisions"] != []
    ):
        raise WrongFamilyPromptSwapError("seed collision audit is not a passing fixed-seed audit")
    entries = audit["json_file_inventory"]
    if not isinstance(entries, list) or audit["json_file_count"] != len(entries) or not entries:
        raise WrongFamilyPromptSwapError("seed JSON inventory is incomplete")
    entry_fields = {
        "path",
        "sha256",
        "size_bytes",
        "role",
        "fresh_seed_occurrences",
    }
    registry_entries = []
    normalized_paths = []
    for index, raw_entry in enumerate(entries):
        entry = _closed(raw_entry, entry_fields, "seed inventory entry[%d]" % index)
        path = Path(str(entry["path"]))
        if not path.is_absolute() or path == Path("/"):
            raise WrongFamilyPromptSwapError("seed inventory paths must be absolute")
        normalized_paths.append(str(path))
        _sha(entry["sha256"], "seed inventory file SHA-256")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise WrongFamilyPromptSwapError("seed inventory file size differs")
        if entry["role"] == "authenticated_registry_only_allowed_fresh_seed_occurrence":
            registry_entries.append(entry)
            occurrences = entry["fresh_seed_occurrences"]
            if (
                not isinstance(occurrences, list)
                or sorted(item.get("seed") for item in occurrences if isinstance(item, Mapping))
                != sorted(FRESH_SEEDS.values())
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != {"json_pointer", "seed"}
                    or not isinstance(item["json_pointer"], str)
                    for item in occurrences
                )
            ):
                raise WrongFamilyPromptSwapError("registry seed occurrences differ")
        elif entry["role"] != "scanned_json":
            raise WrongFamilyPromptSwapError("seed inventory role differs")
        elif entry["fresh_seed_occurrences"] != []:
            raise WrongFamilyPromptSwapError("passing seed inventory contains a collision")
    if normalized_paths != sorted(normalized_paths):
        raise WrongFamilyPromptSwapError("seed inventory paths must be sorted")
    if len(set(normalized_paths)) != len(entries):
        raise WrongFamilyPromptSwapError("seed JSON inventory repeats a path")
    if (
        len(registry_entries) != 1
        or registry_entries[0]["path"] != audit["registry_path"]
        or registry_entries[0]["sha256"] != REGISTRY_RAW_SHA256
    ):
        raise WrongFamilyPromptSwapError("seed inventory lost the authenticated registry entry")
    roots = audit["scan_roots"]
    if (
        not isinstance(roots, list)
        or not roots
        or len(set(roots)) != len(roots)
        or any(not Path(str(root)).is_absolute() or Path(str(root)) == Path("/") for root in roots)
    ):
        raise WrongFamilyPromptSwapError("seed scan root closure differs")
    return audit


def _rewrite_fresh_candidate(
    source: Mapping[str, Any], *, iid: str, seed: int
) -> Dict[str, Any]:
    row = json.loads(json.dumps(source, ensure_ascii=False))
    branch = row["semantic_branch"]
    row["candidate_id"] = "%s-%s-%s" % (PILOT_ID, iid, branch)
    row["calibration_group_id"] = "%s-%s-s%d" % (PILOT_ID, iid, seed)
    row["seed"] = seed
    row["analysis_split"] = "confirmation"
    try:
        return bank_contract.validate_candidate(row)
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyPromptSwapError("fresh candidate is not renderer-compatible") from error


def _rubric_sha256(rubric: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(rubric))


def _judgment_requirements(
    cell: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_branch = {row["semantic_branch"]: row for row in candidates}
    result: List[Dict[str, Any]] = []
    for family_key, branches, required in (
        ("a", BRANCH_ORDER, FAMILY_A_REQUIRED),
        ("b", FAMILY_B_BRANCH_ORDER, FAMILY_B_REQUIRED),
    ):
        family_id = cell[family_key + "_family_id"]
        rubric = cell["family_rubrics"][family_key]
        for branch in branches:
            candidate = by_branch[branch]
            result.append(
                {
                    "audit_key": [candidate["candidate_id"], family_id],
                    "candidate_id": candidate["candidate_id"],
                    "semantic_branch": branch,
                    "evaluated_family_id": family_id,
                    "required_outcome": required[branch],
                    "rubric_id": rubric["rubric_id"],
                    "rubric_sha256": _rubric_sha256(rubric),
                    "family_prompt_utf8_sha256": cell["query_prompts"][family_key][
                        "utf8_sha256"
                    ],
                    "common_null_utf8_sha256": cell["query_prompts"]["common_null"][
                        "utf8_sha256"
                    ],
                }
            )
    return result


def build_generation_plan(
    *,
    registry_path: Path,
    expected_registry_sha256: str,
    source_bank_path: Path,
    seed_scan_roots: Sequence[Path],
) -> Dict[str, Any]:
    registry, registry_digest = load_registry(registry_path, expected_registry_sha256)
    bank, bank_digest = load_source_bank(source_bank_path, registry)
    source_rows = _source_rows_by_iid(bank)
    audit = build_seed_collision_audit(
        registry_path=Path(registry_path),
        registry_raw_sha256=registry_digest,
        scan_roots=seed_scan_roots,
    )
    cells_by_iid = {cell["iid"]: cell for cell in registry["cells"]}
    prospective_cells: List[Dict[str, Any]] = []
    all_judgment_keys: Set[Tuple[str, str]] = set()
    for iid in PROSPECTIVE_IIDS:
        cell = cells_by_iid[iid]
        source_cell = source_rows[iid]
        if {row["action_family_id"] for row in source_cell} != {
            "dog-sit-facing-camera" if iid.startswith("841b") else "human-rise-to-stand"
        }:
            raise WrongFamilyPromptSwapError("source A-family authority differs")
        candidates = [
            _rewrite_fresh_candidate(row, iid=iid, seed=FRESH_SEEDS[iid])
            for row in source_cell
        ]
        requirements = _judgment_requirements(cell, candidates)
        if len(requirements) != 12:
            raise WrongFamilyPromptSwapError("fresh cell must have exactly 12 judgments")
        for requirement in requirements:
            key = tuple(requirement["audit_key"])
            if key in all_judgment_keys:
                raise WrongFamilyPromptSwapError("family audit key collision")
            all_judgment_keys.add(key)
        first = source_cell[0]
        prospective_cells.append(
            {
                "cell_id": cell["cell_id"],
                "iid": iid,
                "fresh_seed": FRESH_SEEDS[iid],
                "source_cell_binding": {
                    "source_calibration_group_id": first["calibration_group_id"],
                    "source_seed": first["seed"],
                    "geometry_source_video": first["geometry_source_video"],
                    "geometry_source_video_sha256": first["geometry_source_video_sha256"],
                    "old_seed_media_seen_and_outcomes_forbidden": True,
                    "caption_and_geometry_only": True,
                },
                "generation_candidates": candidates,
                "query_prompts": cell["query_prompts"],
                "family_rubrics": cell["family_rubrics"],
                "judgment_requirements": requirements,
            }
        )
    retrospective = []
    for iid in RETROSPECTIVE_IIDS:
        cell = cells_by_iid[iid]
        retrospective.append(
            {
                "cell_id": cell["cell_id"],
                "iid": iid,
                "source_seed": cell["source_seed"],
                "source_candidate_ids": [row["candidate_id"] for row in source_rows[iid]],
                "media_seen_before_registry_seal": True,
                "allowed_role": "rubric_discovery_only",
                "fit_confirmation_threshold_or_optimizer_use": False,
            }
        )
    plan = {
        "schema_version": GENERATION_PLAN_SCHEMA,
        "pilot_id": PILOT_ID,
        "registry_raw_sha256": registry_digest,
        "source_bank_raw_sha256": bank_digest,
        "sampling_contract": bank_contract.SAMPLING_CONTRACT,
        "interpretation_contract": INTERPRETATION_CONTRACT,
        "seed_collision_audit": audit,
        "retrospective_discovery_exclusions": retrospective,
        "prospective_cells": prospective_cells,
        "candidate_count": sum(len(cell["generation_candidates"]) for cell in prospective_cells),
        "judgment_count": sum(len(cell["judgment_requirements"]) for cell in prospective_cells),
        "query_prompts_are_generation_captions": False,
        "official_gaussian_binding_status": "required_post_render_before_any_query_or_audit",
    }
    plan["generation_plan_digest"] = sha256_bytes(canonical_json_bytes(plan))
    return validate_generation_plan(plan)


def validate_generation_plan(value: Any) -> Dict[str, Any]:
    raw = dict(_closed(value, _PLAN_FIELDS, "generation plan"))
    digest = _sha(raw.pop("generation_plan_digest"), "generation plan digest")
    if sha256_bytes(canonical_json_bytes(raw)) != digest:
        raise WrongFamilyPromptSwapError("generation plan digest differs")
    raw["generation_plan_digest"] = digest
    if (
        raw["schema_version"] != GENERATION_PLAN_SCHEMA
        or raw["pilot_id"] != PILOT_ID
        or raw["registry_raw_sha256"] != REGISTRY_RAW_SHA256
        or raw["source_bank_raw_sha256"] != SOURCE_BANK_RAW_SHA256
        or raw["sampling_contract"] != bank_contract.SAMPLING_CONTRACT
        or raw["interpretation_contract"] != INTERPRETATION_CONTRACT
        or raw["query_prompts_are_generation_captions"] is not False
        or raw["official_gaussian_binding_status"]
        != "required_post_render_before_any_query_or_audit"
    ):
        raise WrongFamilyPromptSwapError("generation plan contract differs")
    validate_seed_collision_audit(raw["seed_collision_audit"])
    authority_registry, _ = load_registry(
        (METHOD_ROOT / "assets" / REGISTRY_ASSET_BASENAME).resolve(),
        REGISTRY_RAW_SHA256,
    )
    authority_bank, _ = load_source_bank(
        (METHOD_ROOT / "assets" / SOURCE_BANK_BASENAME).resolve(),
        authority_registry,
    )
    authority_cells = {cell["iid"]: cell for cell in authority_registry["cells"]}
    authority_rows = _source_rows_by_iid(authority_bank)
    if raw["candidate_count"] != 20 or raw["judgment_count"] != 24:
        raise WrongFamilyPromptSwapError("generation plan population differs")
    exclusions = raw["retrospective_discovery_exclusions"]
    if not isinstance(exclusions, list) or [row.get("iid") for row in exclusions] != list(
        RETROSPECTIVE_IIDS
    ):
        raise WrongFamilyPromptSwapError("retrospective discovery closure differs")
    for row, iid in zip(exclusions, RETROSPECTIVE_IIDS):
        if (
            row.get("media_seen_before_registry_seal") is not True
            or row.get("allowed_role") != "rubric_discovery_only"
            or row.get("fit_confirmation_threshold_or_optimizer_use") is not False
            or len(row.get("source_candidate_ids", [])) != 10
            or row.get("cell_id") != authority_cells[iid]["cell_id"]
            or row.get("source_seed") != SOURCE_SEEDS[iid]
            or row.get("source_candidate_ids")
            != [candidate["candidate_id"] for candidate in authority_rows[iid]]
        ):
            raise WrongFamilyPromptSwapError("retrospective media gained analytic use")
    cells = raw["prospective_cells"]
    if not isinstance(cells, list) or [cell.get("iid") for cell in cells] != list(
        PROSPECTIVE_IIDS
    ):
        raise WrongFamilyPromptSwapError("prospective cell order differs")
    candidate_ids: Set[str] = set()
    audit_keys: Set[Tuple[str, str]] = set()
    for raw_cell in cells:
        cell = _closed(raw_cell, _PLAN_CELL_FIELDS, "prospective cell")
        iid = cell["iid"]
        authority_cell = authority_cells[iid]
        if cell["fresh_seed"] != FRESH_SEEDS[iid]:
            raise WrongFamilyPromptSwapError("fresh seed changed after preregistration")
        if (
            cell["cell_id"] != authority_cell["cell_id"]
            or cell["query_prompts"] != authority_cell["query_prompts"]
            or cell["family_rubrics"] != authority_cell["family_rubrics"]
        ):
            raise WrongFamilyPromptSwapError("plan queries/rubrics differ from registry")
        source = cell["source_cell_binding"]
        authority_first = authority_rows[iid][0]
        expected_source_binding = {
            "source_calibration_group_id": authority_first["calibration_group_id"],
            "source_seed": authority_first["seed"],
            "geometry_source_video": authority_first["geometry_source_video"],
            "geometry_source_video_sha256": authority_first[
                "geometry_source_video_sha256"
            ],
            "old_seed_media_seen_and_outcomes_forbidden": True,
            "caption_and_geometry_only": True,
        }
        if source != expected_source_binding:
            raise WrongFamilyPromptSwapError("old confirmation media leaked into fresh result")
        candidates = cell["generation_candidates"]
        if not isinstance(candidates, list) or len(candidates) != 10:
            raise WrongFamilyPromptSwapError("fresh cell must contain ten generation branches")
        normalized_candidates = []
        for candidate in candidates:
            try:
                checked = bank_contract.validate_candidate(candidate)
            except bank_contract.PairT2VCalibrationSpecError as error:
                raise WrongFamilyPromptSwapError("fresh candidate failed closed validation") from error
            if (
                checked["seed"] != FRESH_SEEDS[iid]
                or checked["analysis_split"] != "confirmation"
                or not checked["candidate_id"].startswith(PILOT_ID + "-" + iid + "-")
            ):
                raise WrongFamilyPromptSwapError("fresh candidate identity/seed differs")
            if checked["candidate_id"] in candidate_ids:
                raise WrongFamilyPromptSwapError("fresh candidate IDs alias")
            candidate_ids.add(checked["candidate_id"])
            normalized_candidates.append(checked)
        if [row["semantic_branch"] for row in normalized_candidates] != list(BRANCH_ORDER):
            raise WrongFamilyPromptSwapError("fresh branch order differs")
        expected_candidates = [
            _rewrite_fresh_candidate(
                candidate, iid=iid, seed=FRESH_SEEDS[iid]
            )
            for candidate in authority_rows[iid]
        ]
        if normalized_candidates != expected_candidates:
            raise WrongFamilyPromptSwapError("fresh candidates differ from source authority")
        if len({row["calibration_group_id"] for row in normalized_candidates}) != 1:
            raise WrongFamilyPromptSwapError("fresh cell does not share one calibration group")
        prompts = _closed(cell["query_prompts"], {"a", "b", "common_null"}, "plan query prompts")
        checked_prompts = {
            role: _validate_prompt(prompts[role], "plan query prompt " + role)
            for role in ("a", "b", "common_null")
        }
        if len({prompt["utf8_sha256"] for prompt in checked_prompts.values()}) != 3:
            raise WrongFamilyPromptSwapError("plan prompt bytes collide")
        rubrics = _closed(cell["family_rubrics"], {"a", "b"}, "plan rubrics")
        requirements = cell["judgment_requirements"]
        if not isinstance(requirements, list) or len(requirements) != 12:
            raise WrongFamilyPromptSwapError("fresh cell lost its 12 judgments")
        by_id = {row["candidate_id"]: row for row in normalized_candidates}
        a_family = rubrics["a"]["evaluated_family_id"]
        b_family = rubrics["b"]["evaluated_family_id"]
        expected = []
        for family_key, family_id, branches, outcomes in (
            ("a", a_family, BRANCH_ORDER, FAMILY_A_REQUIRED),
            ("b", b_family, FAMILY_B_BRANCH_ORDER, FAMILY_B_REQUIRED),
        ):
            rubric_sha = _rubric_sha256(rubrics[family_key])
            for branch in branches:
                candidate = next(row for row in normalized_candidates if row["semantic_branch"] == branch)
                expected.append(
                    {
                        "audit_key": [candidate["candidate_id"], family_id],
                        "candidate_id": candidate["candidate_id"],
                        "semantic_branch": branch,
                        "evaluated_family_id": family_id,
                        "required_outcome": outcomes[branch],
                        "rubric_id": rubrics[family_key]["rubric_id"],
                        "rubric_sha256": rubric_sha,
                        "family_prompt_utf8_sha256": checked_prompts[family_key]["utf8_sha256"],
                        "common_null_utf8_sha256": checked_prompts["common_null"]["utf8_sha256"],
                    }
                )
        if requirements != expected:
            raise WrongFamilyPromptSwapError("family audit requirements differ from 10+2 seal")
        for requirement in requirements:
            _closed(requirement, _JUDGMENT_REQUIREMENT_FIELDS, "judgment requirement")
            key = tuple(requirement["audit_key"])
            if key in audit_keys or requirement["candidate_id"] not in by_id:
                raise WrongFamilyPromptSwapError("family audit key aliases or lacks media")
            audit_keys.add(key)
    if len(candidate_ids) != 20 or len(audit_keys) != 24:
        raise WrongFamilyPromptSwapError("prospective tuple is not 20 candidates / 24 judgments")
    return raw


def _verify_bound_file(path_value: Any, expected_sha256: Any, label: str) -> Tuple[str, str]:
    path = _plain_absolute_file(path_value, label)
    expected = _sha(expected_sha256, label + " SHA-256")
    if file_sha256(path) != expected:
        raise WrongFamilyPromptSwapError("%s bytes differ" % label)
    return str(path), expected


def _validate_renderer_receipt_binding(
    receipt_path: str,
    *,
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    raw = Path(receipt_path).read_bytes()
    receipt = loads_strict(raw, label="candidate renderer receipt")
    if not isinstance(receipt, Mapping):
        raise WrongFamilyPromptSwapError("candidate renderer receipt must be an object")
    declared_digest = _sha(receipt.get("receipt_digest"), "candidate receipt digest")
    unsigned = dict(receipt)
    unsigned.pop("receipt_digest")
    if sha256_bytes(canonical_json_bytes(unsigned)) != declared_digest:
        raise WrongFamilyPromptSwapError("candidate renderer receipt digest differs")
    try:
        receipt_candidate = bank_contract.validate_candidate(receipt.get("candidate"))
    except bank_contract.PairT2VCalibrationSpecError as error:
        raise WrongFamilyPromptSwapError("renderer receipt candidate differs") from error
    if receipt_candidate != candidate:
        raise WrongFamilyPromptSwapError("renderer receipt does not own the planned candidate")
    if receipt.get("sampling_contract") != bank_contract.SAMPLING_CONTRACT:
        raise WrongFamilyPromptSwapError("renderer receipt sampling differs")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise WrongFamilyPromptSwapError("renderer receipt artifacts differ")
    mp4 = artifacts.get("mp4")
    gaussian = artifacts.get("official_initial_gaussian")
    if not isinstance(mp4, Mapping) or not isinstance(gaussian, Mapping):
        raise WrongFamilyPromptSwapError("renderer receipt lacks MP4 or official Gaussian")
    if (
        mp4.get("path") != row["mp4_path"]
        or mp4.get("sha256") != row["mp4_sha256"]
        or mp4.get("frame_count") != 81
        or mp4.get("fps") != 25
    ):
        raise WrongFamilyPromptSwapError("renderer receipt is not bound to exact81 MP4")
    gaussian_expected = {
        "path": row["official_gaussian_path"],
        "sha256": row["official_gaussian_artifact_sha256"],
        "raw_value_sha256": row["raw_value_sha256"],
        "content_sha256": row["content_sha256"],
        "tensor_key": row["tensor_key"],
        "shape": row["shape"],
        "dtype": row["dtype"],
        "stored_dtype": row["stored_dtype"],
        "generator_initial_seed": row["generator_initial_seed"],
        "captured_from_native_sampler": row["captured_from_native_sampler"],
        "external_initial_noise_injection": row["external_initial_noise_injection"],
        "source_or_target_derived": row["source_or_target_derived"],
    }
    for field, expected in gaussian_expected.items():
        if gaussian.get(field) != expected:
            raise WrongFamilyPromptSwapError(
                "renderer receipt official Gaussian %s differs" % field
            )


def build_gaussian_binding_manifest(
    generation_plan: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Seal and validate the post-render native-Gaussian evidence rows."""

    plan = validate_generation_plan(generation_plan)
    result = {
        "schema_version": GAUSSIAN_BINDING_SCHEMA,
        "generation_plan_digest": plan["generation_plan_digest"],
        "bindings": [dict(row) for row in bindings],
    }
    result["gaussian_binding_digest"] = sha256_bytes(canonical_json_bytes(result))
    return validate_gaussian_binding(result, plan)


def validate_gaussian_binding(value: Any, generation_plan: Mapping[str, Any]) -> Dict[str, Any]:
    plan = validate_generation_plan(generation_plan)
    root = dict(_closed(value, _GAUSSIAN_ROOT_FIELDS, "Gaussian binding"))
    binding_digest = _sha(
        root.pop("gaussian_binding_digest"), "Gaussian binding digest"
    )
    if sha256_bytes(canonical_json_bytes(root)) != binding_digest:
        raise WrongFamilyPromptSwapError("Gaussian binding digest differs")
    if (
        root["schema_version"] != GAUSSIAN_BINDING_SCHEMA
        or root["generation_plan_digest"] != plan["generation_plan_digest"]
    ):
        raise WrongFamilyPromptSwapError("Gaussian binding plan authority differs")
    planned = {
        candidate["candidate_id"]: (cell, candidate)
        for cell in plan["prospective_cells"]
        for candidate in cell["generation_candidates"]
    }
    rows = root["bindings"]
    if not isinstance(rows, list) or len(rows) != 20:
        raise WrongFamilyPromptSwapError("Gaussian binding must contain all 20 candidates")
    normalized: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    cell_identities: Dict[str, Set[str]] = {}
    for index, raw_row in enumerate(rows):
        row = dict(_closed(raw_row, _GAUSSIAN_ROW_FIELDS, "Gaussian row[%d]" % index))
        candidate_id = _safe_id(row["candidate_id"], "Gaussian candidate_id")
        if candidate_id not in planned or candidate_id in seen:
            raise WrongFamilyPromptSwapError("Gaussian candidate is extra or repeated")
        seen.add(candidate_id)
        cell, candidate = planned[candidate_id]
        seed = candidate["seed"]
        if row["seed"] != seed or row["generator_initial_seed"] != seed:
            raise WrongFamilyPromptSwapError("official Gaussian seed binding differs")
        row["candidate_receipt_path"], row["candidate_receipt_sha256"] = _verify_bound_file(
            row["candidate_receipt_path"], row["candidate_receipt_sha256"], "candidate receipt"
        )
        row["mp4_path"], row["mp4_sha256"] = _verify_bound_file(
            row["mp4_path"], row["mp4_sha256"], "exact81 MP4"
        )
        row["official_gaussian_path"], row["official_gaussian_artifact_sha256"] = _verify_bound_file(
            row["official_gaussian_path"],
            row["official_gaussian_artifact_sha256"],
            "official Gaussian artifact",
        )
        raw_digest = _sha(row["raw_value_sha256"], "official Gaussian raw value")
        content_digest = _sha(row["content_sha256"], "official Gaussian content")
        shape = row["shape"]
        if (
            row["tensor_key"] != "official_initial_gaussian"
            or row["dtype"] != "torch.float32"
            or row["stored_dtype"] != "torch.float32"
            or not isinstance(shape, list)
            or len(shape) != 5
            or shape[0:3] != [1, 16, 21]
            or any(type(item) is not int or item <= 0 for item in shape)
            or row["captured_from_native_sampler"] is not True
            or row["external_initial_noise_injection"] is not False
            or row["source_or_target_derived"] is not False
        ):
            raise WrongFamilyPromptSwapError("official Gaussian native provenance differs")
        _validate_renderer_receipt_binding(
            row["candidate_receipt_path"], candidate=candidate, row=row
        )
        identity = sha256_bytes(
            canonical_json_bytes(
                {
                    "raw_value_sha256": raw_digest,
                    "content_sha256": content_digest,
                    "shape": shape,
                    "dtype": row["dtype"],
                    "stored_dtype": row["stored_dtype"],
                    "generator_initial_seed": seed,
                }
            )
        )
        cell_identities.setdefault(cell["iid"], set()).add(identity)
        normalized.append(row)
    if seen != set(planned):
        raise WrongFamilyPromptSwapError("Gaussian binding is missing planned candidates")
    if any(len(values) != 1 for values in cell_identities.values()):
        raise WrongFamilyPromptSwapError("ten branches did not reuse one exact Gaussian tensor")
    raw_by_iid = {
        iid: {
            row["raw_value_sha256"]
            for row in normalized
            if planned[row["candidate_id"]][0]["iid"] == iid
        }
        for iid in PROSPECTIVE_IIDS
    }
    if any(len(values) != 1 for values in raw_by_iid.values()) or len(
        {next(iter(values)) for values in raw_by_iid.values()}
    ) != 2:
        raise WrongFamilyPromptSwapError("fresh cells have missing or colliding Gaussian values")
    result = {
        "schema_version": GAUSSIAN_BINDING_SCHEMA,
        "generation_plan_digest": plan["generation_plan_digest"],
        "bindings": normalized,
        "gaussian_binding_digest": binding_digest,
    }
    return result


def build_audit_plan(
    generation_plan: Mapping[str, Any], gaussian_binding: Mapping[str, Any]
) -> Dict[str, Any]:
    plan = validate_generation_plan(generation_plan)
    binding = validate_gaussian_binding(gaussian_binding, plan)
    binding_rows = {row["candidate_id"]: row for row in binding["bindings"]}
    judgments: List[Dict[str, Any]] = []
    for cell in plan["prospective_cells"]:
        for requirement in cell["judgment_requirements"]:
            media = binding_rows[requirement["candidate_id"]]
            judgments.append(
                {
                    **dict(requirement),
                    "mp4_path": media["mp4_path"],
                    "mp4_sha256": media["mp4_sha256"],
                    "candidate_receipt_sha256": media["candidate_receipt_sha256"],
                    "full_exact81_required": True,
                    "generation_prompt_must_be_hidden": True,
                }
            )
    result = {
        "schema_version": AUDIT_PLAN_SCHEMA,
        "pilot_id": PILOT_ID,
        "generation_plan_digest": plan["generation_plan_digest"],
        "gaussian_binding_digest": binding["gaussian_binding_digest"],
        "interpretation_contract": INTERPRETATION_CONTRACT,
        "judgments": judgments,
        "judgment_count": len(judgments),
    }
    result["audit_plan_digest"] = sha256_bytes(canonical_json_bytes(result))
    return validate_audit_plan(result)


def validate_audit_plan(value: Any) -> Dict[str, Any]:
    raw = dict(_closed(value, _AUDIT_PLAN_FIELDS, "audit plan"))
    digest = _sha(raw.pop("audit_plan_digest"), "audit plan digest")
    if sha256_bytes(canonical_json_bytes(raw)) != digest:
        raise WrongFamilyPromptSwapError("audit plan digest differs")
    raw["audit_plan_digest"] = digest
    if (
        raw["schema_version"] != AUDIT_PLAN_SCHEMA
        or raw["pilot_id"] != PILOT_ID
        or raw["interpretation_contract"] != INTERPRETATION_CONTRACT
        or raw["judgment_count"] != 24
    ):
        raise WrongFamilyPromptSwapError("audit plan closure differs")
    rows = raw["judgments"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise WrongFamilyPromptSwapError("audit plan must have exactly 24 judgments")
    keys: Set[Tuple[str, str]] = set()
    for row in rows:
        checked = _closed(row, _AUDIT_JUDGMENT_FIELDS, "audit plan judgment")
        key = tuple(checked["audit_key"])
        if key != (checked["candidate_id"], checked["evaluated_family_id"]) or key in keys:
            raise WrongFamilyPromptSwapError("family audit key differs or aliases")
        keys.add(key)
        if (
            type(checked["required_outcome"]) is not bool
            or checked["full_exact81_required"] is not True
            or checked["generation_prompt_must_be_hidden"] is not True
        ):
            raise WrongFamilyPromptSwapError("audit hard gate was relaxed")
        _sha(checked["mp4_sha256"], "audit MP4 SHA-256")
        _sha(checked["rubric_sha256"], "audit rubric SHA-256")
    return raw


def validate_completed_audit(
    audit_plan: Mapping[str, Any], completed: Mapping[str, Any]
) -> Dict[str, Any]:
    plan = validate_audit_plan(audit_plan)
    root = _closed(completed, _COMPLETED_ROOT_FIELDS, "completed audit")
    if (
        root["schema_version"] != COMPLETED_AUDIT_SCHEMA
        or root["audit_plan_digest"] != plan["audit_plan_digest"]
    ):
        raise WrongFamilyPromptSwapError("completed audit plan binding differs")
    rows = root["judgments"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise WrongFamilyPromptSwapError("completed audit must contain exactly 24 judgments")
    expected = {
        (row["candidate_id"], row["evaluated_family_id"]): row
        for row in plan["judgments"]
    }
    seen: Set[Tuple[str, str]] = set()
    normalized: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = dict(_closed(raw_row, _COMPLETED_ROW_FIELDS, "completed judgment"))
        key = (row["candidate_id"], row["evaluated_family_id"])
        if key not in expected or key in seen:
            raise WrongFamilyPromptSwapError("completed family audit key is extra or repeated")
        seen.add(key)
        requirement = expected[key]
        decision = row["decision"]
        if decision in ("unknown", "ambiguous"):
            raise WrongFamilyPromptSwapError(
                "%s invalidates the entire prospective tuple" % decision
            )
        if decision not in ("true", "false"):
            raise WrongFamilyPromptSwapError("completed decision is outside sealed vocabulary")
        observed = decision == "true"
        if observed is not requirement["required_outcome"]:
            raise WrongFamilyPromptSwapError("observed family result violates preregistered outcome")
        if (
            row["full_exact81_viewed"] is not True
            or row["generation_prompt_hidden"] is not True
            or row["mp4_sha256"] != requirement["mp4_sha256"]
            or row["rubric_sha256"] != requirement["rubric_sha256"]
        ):
            raise WrongFamilyPromptSwapError("audit media/rubric/blinding evidence differs")
        _safe_id(row["reviewer_id"], "reviewer_id")
        if not isinstance(row["review_notes"], str) or "\x00" in row["review_notes"]:
            raise WrongFamilyPromptSwapError("review notes must be text without NUL")
        normalized.append(row)
    if seen != set(expected):
        raise WrongFamilyPromptSwapError("completed audit is missing a family key")
    receipt = {
        "schema_version": RESULT_RECEIPT_SCHEMA,
        "audit_plan_digest": plan["audit_plan_digest"],
        "completed_judgment_count": 24,
        "prospective_tuple_pass": True,
        "pilot_family_swap_evidence_usable": True,
        "editor": False,
        "scientific_critic": False,
        "optimizer_authorized": False,
        "fresh_confirmation_enters_optimizer": False,
        "retrospective_media_can_fit_or_confirm": False,
        "completed_audit_digest": sha256_bytes(canonical_json_bytes(dict(root))),
    }
    receipt["result_receipt_digest"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def _write_create_only(path_value: Any, value: Mapping[str, Any]) -> str:
    path = Path(path_value)
    if not path.is_absolute() or path == Path("/") or path.exists() or path.is_symlink():
        raise WrongFamilyPromptSwapError("output must be a fresh absolute plain-file path")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise WrongFamilyPromptSwapError("output parent must be an existing plain directory")
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    digest = sha256_bytes(payload)
    if file_sha256(path) != digest:
        raise WrongFamilyPromptSwapError("published output failed byte replay")
    return digest


def _load_hash_bound_json(path_value: Any, expected_sha256: str, label: str) -> Any:
    path = _plain_absolute_file(path_value, label)
    expected = _sha(expected_sha256, label + " expected SHA-256")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected:
        raise WrongFamilyPromptSwapError("%s raw SHA-256 differs" % label)
    return loads_strict(raw, label=label)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-plan")
    build.add_argument("--registry", required=True)
    build.add_argument("--expected-registry-sha256", required=True)
    build.add_argument("--source-bank", required=True)
    build.add_argument("--seed-scan-root", action="append", required=True)
    build.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate-completed-audit")
    validate.add_argument("--audit-plan", required=True)
    validate.add_argument("--expected-audit-plan-sha256", required=True)
    validate.add_argument("--completed-audit", required=True)
    validate.add_argument("--expected-completed-audit-sha256", required=True)
    validate.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-plan":
        plan = build_generation_plan(
            registry_path=Path(args.registry),
            expected_registry_sha256=args.expected_registry_sha256,
            source_bank_path=Path(args.source_bank),
            seed_scan_roots=[Path(item) for item in args.seed_scan_root],
        )
        digest = _write_create_only(args.output, plan)
        print(canonical_json_bytes({"output": args.output, "sha256": digest}).decode("utf-8"))
        return 0
    audit_plan = _load_hash_bound_json(
        args.audit_plan, args.expected_audit_plan_sha256, "audit plan"
    )
    completed = _load_hash_bound_json(
        args.completed_audit,
        args.expected_completed_audit_sha256,
        "completed audit",
    )
    receipt = validate_completed_audit(audit_plan, completed)
    digest = _write_create_only(args.output, receipt)
    print(canonical_json_bytes({"output": args.output, "sha256": digest}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_PLAN_SCHEMA",
    "BRANCH_ORDER",
    "COMPLETED_AUDIT_SCHEMA",
    "FRESH_SEEDS",
    "GAUSSIAN_BINDING_SCHEMA",
    "GENERATION_PLAN_SCHEMA",
    "INTERPRETATION_CONTRACT",
    "PILOT_ID",
    "PROSPECTIVE_IIDS",
    "REGISTRY_RAW_SHA256",
    "REGISTRY_SCHEMA",
    "RESULT_RECEIPT_SCHEMA",
    "SOURCE_BANK_RAW_SHA256",
    "WrongFamilyPromptSwapError",
    "build_audit_plan",
    "build_gaussian_binding_manifest",
    "build_generation_plan",
    "build_seed_collision_audit",
    "canonical_json_bytes",
    "file_sha256",
    "load_registry",
    "load_source_bank",
    "sha256_bytes",
    "validate_audit_plan",
    "validate_completed_audit",
    "validate_gaussian_binding",
    "validate_generation_plan",
    "validate_registry",
    "validate_seed_collision_audit",
]

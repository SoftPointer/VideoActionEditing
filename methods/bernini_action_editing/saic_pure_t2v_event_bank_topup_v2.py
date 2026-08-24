#!/usr/bin/env python3
"""Seal the SAIC pure-T2V hard-negative event-bank v2 top-up.

This is an additive companion to ``saic_pure_t2v_event_bank_v1``.  It does
not replace or rewrite the running v1 bank.  For every one of v1's twenty
``(iid, seed)`` cells it preregisters exactly three extra text-only Bernini-R
proposals:

* ``incomplete``: the requested action starts but never reaches its terminal
  state;
* ``camera_only``: the actor stays in q0 while the camera moves; and
* ``appearance_only``: the actor stays in q0 while identity-bearing appearance
  changes.

The three top-up branches reuse v1's identity/scene text, q0 text, geometry,
seed, and official Gaussian sampling contract.  As in v1, a launch-local
constant-black exact81 clip is only a legacy spatial-bucket argument.  No real
source RGB, source latent/noise, target, reference, or donor is admitted.

Rendered media remains an unaudited proposal.  Nothing in this module selects
a seed, verifies an event, creates a training target, or authorizes an update.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import build_saic_reversible_source_set_v1 as source_set  # noqa: E402
import saic_pure_t2v_event_bank_v1 as v1  # noqa: E402


SCHEMA_VERSION = "bernini-saic-pure-t2v-event-bank-topup-spec-v2"
CANDIDATE_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-topup-candidate-v2"
PLAN_SCHEMA_VERSION = "bernini-saic-pure-t2v-event-topup-plan-v2"
ASSET_PATH = METHOD_ROOT / "assets" / "saic_pure_t2v_event_bank_topup_v2.json"

# These two digests are deliberately duplicated from the immutable v1 tests.
# A changed v1 asset therefore fails before v2 can be authored or loaded.
BASE_V1_SPEC_RAW_SHA256 = (
    "623a7ed8a2ce2d327247c541b59aa2d39f1fbfe4a480f7351d042c7ef7a47927"
)
BASE_V1_SPEC_CONTENT_SHA256 = (
    "3920d5c121b75c6bbf984c24440c9773dfb49006778c61a671ae50963bb5456a"
)

BRANCH_ORDER = ("incomplete", "camera_only", "appearance_only")
MERGED_BRANCH_ORDER = (*v1.BRANCH_ORDER, *BRANCH_ORDER)
BANK_ID = "saic-text-only-hard-negative-topup-exact81-v2"
TOP_UP_RELATION = {
    "incomplete": "desired_action_progress_without_terminal_state",
    "camera_only": "camera_motion_without_desired_actor_transition",
    "appearance_only": "identity_appearance_change_without_desired_actor_transition",
}

# Reuse the complete, already-tested v1 runtime contracts byte-for-byte.
SOURCE_MANIFEST_CONTENT_SHA256 = v1.SOURCE_MANIFEST_CONTENT_SHA256
GROUP_LAYOUT = v1.GROUP_LAYOUT
FRAME_COUNT = v1.FRAME_COUNT
FPS = v1.FPS
SAMPLING_CONTRACT = v1.SAMPLING_CONTRACT
SEMANTIC_INPUT_CLOSURE = v1.SEMANTIC_INPUT_CLOSURE
GEOMETRY_PROXY_CONTRACT = v1.GEOMETRY_PROXY_CONTRACT
ARTIFACT_AUTHORITY = v1.ARTIFACT_AUTHORITY
PROXY_RECEIPT_SCHEMA_VERSION = v1.PROXY_RECEIPT_SCHEMA_VERSION


class SAICPureT2VEventBankTopupError(RuntimeError):
    """Raised before an ambiguous or privileged v2 artifact is accepted."""


# Per-row instructions avoid turning a typed hard negative into a vague generic
# prompt.  All instructions are ASCII because the sealed JSON is canonical
# ASCII and every prompt digest is over its UTF-8 bytes.
HARD_NEGATIVE_INSTRUCTIONS: Mapping[str, Mapping[str, str]] = {
    "7b88a1ca1f804f41": {
        "incomplete": (
            "Have the same French Bulldog begin bending both hind legs and lower "
            "its pelvis only partway, then stop in a sustained half-crouch without "
            "ever settling into a seated state. Keep its location and facing "
            "direction fixed and preserve identity, harness, leash, leaves, "
            "lighting, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same French Bulldog in "
            "its original stable four-legged stand for the entire clip and never "
            "let it sit, while the camera performs a conspicuous smooth rightward "
            "orbit and mild push-in. Preserve the dog, harness, leash, and scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the French Bulldog in "
            "its original stable four-legged stand and never let it sit, while its "
            "gray coat gradually becomes bright white and its black harness becomes "
            "red. Keep location, leaves, geometry, framing, and camera fixed."
        ),
    },
    "841b5e0080a1441d": {
        "incomplete": (
            "Have the same shepherd begin bending its hind legs and lower its hips "
            "only halfway toward the grass, then hold a clear half-crouch without "
            "ever reaching a seated state. Preserve identity, coat pattern, collar, "
            "leash, field, scale, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same shepherd standing "
            "on all four legs for the entire clip and never let it sit, while the "
            "camera makes a conspicuous smooth leftward orbit and mild zoom-in. "
            "Preserve the dog, collar, leash, and field."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the shepherd standing "
            "on all four legs and never let it sit, while its black-and-tan coat "
            "gradually becomes pale cream and its collar becomes blue. Keep the "
            "field, body geometry, framing, and camera fixed."
        ),
    },
    "99cde432839f4240": {
        "incomplete": (
            "Have the same dog begin bending its hind legs and lower its hips only "
            "partway, then hold a sustained half-crouch without ever reaching a "
            "seated state. Keep the tail above the ground and preserve identity, "
            "forest, monochrome appearance, scale, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same dog in its original "
            "four-legged stand for the entire clip and never let it sit, while the "
            "camera makes a conspicuous smooth lateral arc and push-in through the "
            "forest. Preserve the dog and monochrome scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the dog in its original "
            "four-legged stand and never let it sit, while the monochrome coat and "
            "forest gradually shift into vivid warm colors. Keep identity geometry, "
            "location, framing, and camera fixed."
        ),
    },
    "6ea45d35943742bb": {
        "incomplete": (
            "Have the same white dog begin bending its hind legs and lower its hips "
            "only halfway while continuing the slow rightward head turn, then hold "
            "a half-crouch without ever reaching a seated state. Preserve identity, "
            "collar, grass, shadows, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same white dog standing "
            "on all four legs for the entire clip, allow only its slow rightward "
            "head turn, and never let it sit, while the camera makes a conspicuous "
            "smooth leftward orbit and push-in. Preserve dog and scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the white dog standing "
            "on all four legs, allow only its slow head turn, and never let it sit, "
            "while its white coat gradually becomes dark brown and its collar "
            "becomes red. Keep geometry, grass, shadows, framing, and camera fixed."
        ),
    },
    "a35b590961d24694": {
        "incomplete": (
            "Have the same woman transfer weight toward both feet and lift her right "
            "knee only partway from the floor, then stop in a sustained low crouch "
            "without ever reaching an upright stand. Preserve identity, hair, "
            "clothing, studio, scale, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same woman in her original "
            "one-knee kneeling pose for the entire clip and never let her stand, "
            "while the camera performs a conspicuous smooth clockwise orbit and "
            "push-in. Preserve the woman, clothing, and studio scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the woman in her original "
            "one-knee kneeling pose and never let her stand, while her brown hair "
            "gradually becomes silver and her cream trousers become blue. Keep body "
            "geometry, studio, framing, and camera fixed."
        ),
    },
    "31c34509415745ca": {
        "incomplete": (
            "Have the same subject transfer weight toward both feet and lift the "
            "right knee only partway, then hold a sustained low crouch without ever "
            "reaching an upright stand, while retaining the smile and outward arm "
            "gesture. Preserve red braids, clothing, mall, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same subject in the "
            "original right-knee kneel with the natural arm gesture and never let "
            "the lower body stand, while the camera makes a conspicuous smooth "
            "rightward orbit and zoom-in. Preserve the subject and mall scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the subject in the "
            "original right-knee kneel and never let the lower body stand, while the "
            "red braids gradually become black and the bright socks become plain "
            "white. Keep geometry, mall, framing, and camera fixed."
        ),
    },
    "311c82f83eca4a7f": {
        "incomplete": (
            "Have the same subject shift weight toward both feet and lift the right "
            "knee only partway from the floor, then hold a low crouch without ever "
            "reaching an upright stand, while retaining the phone and right-arm "
            "flex. Preserve identity, tattoos, clothing, mirror, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same subject in the "
            "original right-knee kneeling selfie pose with the phone and flex, and "
            "never let the body stand, while the camera makes a conspicuous smooth "
            "lateral orbit and push-in. Preserve the subject and gym scene."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the subject in the "
            "original right-knee kneeling selfie pose and never let the body stand, "
            "while the blond hair gradually becomes black and the gym clothing "
            "changes to bright red. Keep tattoos, geometry, mirror, and camera fixed."
        ),
    },
    "6d346c38cf504493": {
        "incomplete": (
            "Have the same subject shift weight toward both feet and lift the right "
            "knee only partway, then hold a low crouch without ever reaching an "
            "upright stand, while retaining a safe grip on both white cosplay props. "
            "Preserve identity, costume, set, framing, and locked camera."
        ),
        "camera_only": (
            "Counterfactual camera-only negative: keep the same subject in the "
            "original right-knee kneel with a safe grip on both props and never let "
            "the body stand, while the camera performs a conspicuous smooth orbit "
            "and push-in. Preserve the subject, costume, props, and photo set."
        ),
        "appearance_only": (
            "Counterfactual appearance-only negative: keep the subject in the "
            "original right-knee kneel and never let the body stand, while the blue "
            "wig gradually becomes bright red and the dark costume becomes white. "
            "Keep props, body geometry, photo set, framing, and camera fixed."
        ),
    },
}


canonical_json_bytes = v1.canonical_json_bytes
object_sha256 = v1.object_sha256
file_sha256 = v1.file_sha256
text_sha256 = v1.text_sha256


def _error(message: str, error: Exception | None = None) -> SAICPureT2VEventBankTopupError:
    result = SAICPureT2VEventBankTopupError(message)
    if error is not None:
        result.__cause__ = error
    return result


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        return v1._load_json(path, label=label)
    except v1.SAICPureT2VEventBankError as error:
        raise _error(f"cannot load {label}", error)


def _closed(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    try:
        return v1._closed(value, fields, label=label)
    except v1.SAICPureT2VEventBankError as error:
        raise _error(f"{label} is not closed", error)


def _sha(value: Any, *, label: str) -> str:
    try:
        return v1._sha(value, label=label)
    except v1.SAICPureT2VEventBankError as error:
        raise _error(f"{label} differs", error)


def _write_create_only(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> None:
    try:
        v1._write_create_only(path, canonical_json_bytes(value) + b"\n", mode=mode)
    except v1.SAICPureT2VEventBankError as error:
        raise _error(f"refusing to overwrite {path}", error)


def load_base_v1_spec(
    path: str | Path = v1.ASSET_PATH,
    *,
    source_manifest_path: str | Path = source_set.ASSET_PATH,
) -> dict[str, Any]:
    source = Path(path)
    value = _load_json(source, label="immutable v1 event spec")
    if file_sha256(source) != BASE_V1_SPEC_RAW_SHA256:
        raise SAICPureT2VEventBankTopupError("immutable v1 raw SHA-256 changed")
    try:
        summary = v1.validate_spec(value, source_manifest_path=source_manifest_path)
    except v1.SAICPureT2VEventBankError as error:
        raise _error("immutable v1 validation failed", error)
    if summary["spec_content_sha256"] != BASE_V1_SPEC_CONTENT_SHA256:
        raise SAICPureT2VEventBankTopupError("immutable v1 content SHA-256 changed")
    return value


def _v1_cells(spec: Mapping[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for group in spec["groups"]:
        for candidate in group["candidates"]:
            cells.setdefault((candidate["iid"], candidate["seed"]), []).append(candidate)
    for key, rows in cells.items():
        if [row["branch"] for row in rows] != list(v1.BRANCH_ORDER):
            raise SAICPureT2VEventBankTopupError(f"v1 cell order differs: {key!r}")
    if len(cells) != 20:
        raise SAICPureT2VEventBankTopupError("v1 must expose exactly twenty cells")
    return cells


def _full_prompt(base: Mapping[str, Any], instruction: str) -> str:
    return " ".join(
        (
            "An exactly 81-frame realistic video at 25 fps is one continuous shot with no cut.",
            base["identity_scene_caption"],
            base["branch_start_state_caption"],
            instruction,
        )
    )


def author_spec(
    source_manifest_path: str | Path = source_set.ASSET_PATH,
    base_v1_spec_path: str | Path = v1.ASSET_PATH,
) -> dict[str, Any]:
    base_spec = load_base_v1_spec(
        base_v1_spec_path, source_manifest_path=source_manifest_path
    )
    base_cells = _v1_cells(base_spec)
    if set(HARD_NEGATIVE_INSTRUCTIONS) != {iid for iid, _ in base_cells}:
        raise SAICPureT2VEventBankTopupError("hard-negative text coverage differs")

    groups: list[dict[str, Any]] = []
    for base_group in base_spec["groups"]:
        candidates: list[dict[str, Any]] = []
        ordinal = 0
        group_cells: list[tuple[tuple[str, int], list[dict[str, Any]]]] = []
        for key, rows in base_cells.items():
            if rows[0]["actor_family"] == base_group["actor_family"]:
                group_cells.append((key, rows))
        for (iid, seed), base_rows in group_cells:
            forward, reverse, noop = base_rows
            if forward["branch"] != "forward" or noop["branch"] != "noop":
                raise SAICPureT2VEventBankTopupError("v1 q0 binding differs")
            v1_pair_ids = {
                row["branch"]: row["candidate_id"] for row in base_rows
            }
            v1_cell_digest = object_sha256(
                {
                    "iid": iid,
                    "seed": seed,
                    "candidate_ids": v1_pair_ids,
                    "prompt_sha256": {
                        row["branch"]: row["full_t2v_caption_utf8_sha256"]
                        for row in base_rows
                    },
                }
            )
            for branch in BRANCH_ORDER:
                instruction = HARD_NEGATIVE_INSTRUCTIONS[iid][branch]
                prompt = _full_prompt(forward, instruction)
                candidate = {
                    "candidate_id": f"saic-topup-v2-{iid}-{branch}-s{seed}",
                    "ordinal": ordinal,
                    "row_id": forward["row_id"],
                    "iid": iid,
                    "analysis_split": forward["analysis_split"],
                    "actor_family": forward["actor_family"],
                    "action_family_id": forward["action_family_id"],
                    "initial_state_type": forward["initial_state_type"],
                    "terminal_state_type": forward["terminal_state_type"],
                    "source_media_sha256_for_nonuse_audit": forward[
                        "source_media_sha256_for_nonuse_audit"
                    ],
                    "source_geometry_hw": forward["source_geometry_hw"],
                    "source_caption_utf8_sha256": forward[
                        "source_caption_utf8_sha256"
                    ],
                    "identity_scene_caption": forward["identity_scene_caption"],
                    "identity_scene_caption_utf8_sha256": forward[
                        "identity_scene_caption_utf8_sha256"
                    ],
                    "branch": branch,
                    "hard_negative_relation": TOP_UP_RELATION[branch],
                    "branch_start_state_caption": forward[
                        "branch_start_state_caption"
                    ],
                    "branch_start_state_caption_utf8_sha256": forward[
                        "branch_start_state_caption_utf8_sha256"
                    ],
                    "branch_instruction": instruction,
                    "branch_instruction_utf8_sha256": text_sha256(instruction),
                    "full_t2v_caption": prompt,
                    "full_t2v_caption_utf8_sha256": text_sha256(prompt),
                    "seed": seed,
                    "paired_v1_candidate_ids": v1_pair_ids,
                    "paired_v1_cell_digest": v1_cell_digest,
                    "event_audit_status": "pending_detached_full81_review",
                    "event_verified": False,
                    "identity_preservation_verified": False,
                    "seed_selection_authorized": False,
                    "training_target_authorized": False,
                    "optimizer_authorized": False,
                }
                candidates.append(candidate)
                ordinal += 1
        groups.append(
            {
                "group_id": base_group["group_id"],
                "actor_family": base_group["actor_family"],
                "visible_gpus": base_group["visible_gpus"],
                "candidates": candidates,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "bank_id": BANK_ID,
        "top_up_only": True,
        "base_v1_spec_raw_sha256": BASE_V1_SPEC_RAW_SHA256,
        "base_v1_spec_content_sha256": BASE_V1_SPEC_CONTENT_SHA256,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "source_manifest_file_sha256": v1.file_sha256(source_manifest_path),
        "sampling_contract": SAMPLING_CONTRACT,
        "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
        "geometry_proxy_contract": GEOMETRY_PROXY_CONTRACT,
        "artifact_authority": ARTIFACT_AUTHORITY,
        "branch_order": list(BRANCH_ORDER),
        "merged_branch_order": list(MERGED_BRANCH_ORDER),
        "groups": groups,
    }


_ROOT_FIELDS = {
    "schema_version",
    "bank_id",
    "top_up_only",
    "base_v1_spec_raw_sha256",
    "base_v1_spec_content_sha256",
    "source_manifest_content_sha256",
    "source_manifest_file_sha256",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "branch_order",
    "merged_branch_order",
    "groups",
}
_GROUP_FIELDS = {"group_id", "actor_family", "visible_gpus", "candidates"}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "ordinal",
    "row_id",
    "iid",
    "analysis_split",
    "actor_family",
    "action_family_id",
    "initial_state_type",
    "terminal_state_type",
    "source_media_sha256_for_nonuse_audit",
    "source_geometry_hw",
    "source_caption_utf8_sha256",
    "identity_scene_caption",
    "identity_scene_caption_utf8_sha256",
    "branch",
    "hard_negative_relation",
    "branch_start_state_caption",
    "branch_start_state_caption_utf8_sha256",
    "branch_instruction",
    "branch_instruction_utf8_sha256",
    "full_t2v_caption",
    "full_t2v_caption_utf8_sha256",
    "seed",
    "paired_v1_candidate_ids",
    "paired_v1_cell_digest",
    "event_audit_status",
    "event_verified",
    "identity_preservation_verified",
    "seed_selection_authorized",
    "training_target_authorized",
    "optimizer_authorized",
}
_ENVELOPE_FIELDS = {
    "schema_version",
    "root_spec_raw_sha256",
    "base_v1_spec_raw_sha256",
    "source_manifest_content_sha256",
    "group_id",
    "actor_family",
    "visible_gpus",
    "sampling_contract",
    "semantic_input_closure",
    "geometry_proxy_contract",
    "artifact_authority",
    "candidate",
    "geometry_proxy",
}


def validate_spec(
    value: Mapping[str, Any],
    *,
    source_manifest_path: str | Path = source_set.ASSET_PATH,
    base_v1_spec_path: str | Path = v1.ASSET_PATH,
) -> dict[str, Any]:
    root = _closed(value, _ROOT_FIELDS, label="v2 top-up event spec")
    expected = author_spec(source_manifest_path, base_v1_spec_path)
    if root != expected:
        raise SAICPureT2VEventBankTopupError("v2 top-up differs from sealed authoring")
    if (
        root["schema_version"] != SCHEMA_VERSION
        or root["bank_id"] != BANK_ID
        or root["top_up_only"] is not True
        or root["branch_order"] != list(BRANCH_ORDER)
        or root["merged_branch_order"] != list(MERGED_BRANCH_ORDER)
        or root["sampling_contract"] != SAMPLING_CONTRACT
        or root["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE
        or root["geometry_proxy_contract"] != GEOMETRY_PROXY_CONTRACT
        or root["artifact_authority"] != ARTIFACT_AUTHORITY
    ):
        raise SAICPureT2VEventBankTopupError("v2 top-up root contract differs")

    seen: set[str] = set()
    cells: dict[tuple[str, int], list[str]] = {}
    count = 0
    for group, (group_id, actor, gpus) in zip(root["groups"], GROUP_LAYOUT):
        _closed(group, _GROUP_FIELDS, label=f"v2 group {group_id}")
        if (
            group["group_id"] != group_id
            or group["actor_family"] != actor
            or group["visible_gpus"] != gpus
            or len(group["candidates"]) != 30
        ):
            raise SAICPureT2VEventBankTopupError(f"v2 group {group_id} differs")
        for ordinal, candidate in enumerate(group["candidates"]):
            _closed(candidate, _CANDIDATE_FIELDS, label="v2 candidate")
            candidate_id = candidate["candidate_id"]
            try:
                v1._safe_id(candidate_id, label="v2 candidate ID")
            except v1.SAICPureT2VEventBankError as error:
                raise _error("v2 candidate ID differs", error)
            if candidate_id in seen or candidate["ordinal"] != ordinal:
                raise SAICPureT2VEventBankTopupError("v2 identity/order differs")
            seen.add(candidate_id)
            branch = candidate["branch"]
            if (
                candidate["actor_family"] != actor
                or branch not in BRANCH_ORDER
                or candidate["hard_negative_relation"] != TOP_UP_RELATION[branch]
            ):
                raise SAICPureT2VEventBankTopupError("v2 branch relation differs")
            for text_field, digest_field in (
                ("identity_scene_caption", "identity_scene_caption_utf8_sha256"),
                ("branch_start_state_caption", "branch_start_state_caption_utf8_sha256"),
                ("branch_instruction", "branch_instruction_utf8_sha256"),
                ("full_t2v_caption", "full_t2v_caption_utf8_sha256"),
            ):
                try:
                    text = v1._ascii_text(candidate[text_field], label=text_field)
                except v1.SAICPureT2VEventBankError as error:
                    raise _error(f"v2 {text_field} differs", error)
                if text_sha256(text) != _sha(candidate[digest_field], label=digest_field):
                    raise SAICPureT2VEventBankTopupError(f"v2 {text_field} digest differs")
            geometry = candidate["source_geometry_hw"]
            if (
                not isinstance(geometry, list)
                or len(geometry) != 2
                or any(type(item) is not int or item <= 0 for item in geometry)
                or type(candidate["seed"]) is not int
                or not 0 <= candidate["seed"] < 2**63
                or candidate["event_audit_status"]
                != "pending_detached_full81_review"
                or candidate["event_verified"] is not False
                or candidate["identity_preservation_verified"] is not False
                or candidate["seed_selection_authorized"] is not False
                or candidate["training_target_authorized"] is not False
                or candidate["optimizer_authorized"] is not False
            ):
                raise SAICPureT2VEventBankTopupError("v2 candidate authority differs")
            cells.setdefault((candidate["iid"], candidate["seed"]), []).append(branch)
            count += 1
    if count != 60 or len(cells) != 20:
        raise SAICPureT2VEventBankTopupError("v2 must contain sixty top-up attempts")
    if any(branches != list(BRANCH_ORDER) for branches in cells.values()):
        raise SAICPureT2VEventBankTopupError("v2 cell branch order differs")

    source_manifest = source_set.load_manifest(source_manifest_path)
    serialized = canonical_json_bytes(root)
    for row in source_manifest["rows"]:
        if row["source_video"].encode("ascii") in serialized:
            raise SAICPureT2VEventBankTopupError("real source path leaked into v2")
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_content_sha256": object_sha256(root),
        "candidate_count": 60,
        "row_count": 8,
        "seed_cell_count": 20,
        "top_up_only": True,
        "event_verified": False,
        "optimizer_authorized": False,
    }


def merge_six_branch_cells(
    base_v1_spec: Mapping[str, Any], topup_v2_spec: Mapping[str, Any]
) -> dict[tuple[str, int], tuple[dict[str, Any], ...]]:
    """Validate and merge specs into twenty exact six-branch cells."""

    try:
        v1.validate_spec(base_v1_spec)
    except v1.SAICPureT2VEventBankError as error:
        raise _error("base v1 merge input differs", error)
    validate_spec(topup_v2_spec)
    v1_cells = _v1_cells(base_v1_spec)
    v2_cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for group in topup_v2_spec["groups"]:
        for candidate in group["candidates"]:
            v2_cells.setdefault((candidate["iid"], candidate["seed"]), []).append(
                candidate
            )
    if set(v1_cells) != set(v2_cells):
        raise SAICPureT2VEventBankTopupError("v1/v2 cell coverage differs")

    merged: dict[tuple[str, int], tuple[dict[str, Any], ...]] = {}
    shared_fields = (
        "row_id",
        "iid",
        "analysis_split",
        "actor_family",
        "action_family_id",
        "initial_state_type",
        "terminal_state_type",
        "source_media_sha256_for_nonuse_audit",
        "source_geometry_hw",
        "source_caption_utf8_sha256",
        "identity_scene_caption",
        "identity_scene_caption_utf8_sha256",
        "seed",
    )
    for key in v1_cells:
        base_rows = v1_cells[key]
        topup_rows = v2_cells[key]
        if [row["branch"] for row in topup_rows] != list(BRANCH_ORDER):
            raise SAICPureT2VEventBankTopupError("v2 merge order differs")
        anchor = base_rows[0]
        for candidate in topup_rows:
            if any(candidate[field] != anchor[field] for field in shared_fields):
                raise SAICPureT2VEventBankTopupError("v1/v2 cell metadata differs")
            if candidate["branch_start_state_caption"] != anchor[
                "branch_start_state_caption"
            ]:
                raise SAICPureT2VEventBankTopupError("v2 is not q0-aligned")
        rows = tuple((*base_rows, *topup_rows))
        if [row["branch"] for row in rows] != list(MERGED_BRANCH_ORDER):
            raise SAICPureT2VEventBankTopupError("six-branch merge order differs")
        merged[key] = rows
    return merged


def load_sealed_spec(
    path: str | Path,
    *,
    expected_raw_sha256: str,
    source_manifest_path: str | Path,
    base_v1_spec_path: str | Path,
) -> tuple[dict[str, Any], str]:
    expected = _sha(expected_raw_sha256, label="v2 expected raw SHA-256")
    source = Path(path)
    value = _load_json(source, label="sealed v2 top-up spec")
    actual = file_sha256(source)
    if actual != expected:
        raise SAICPureT2VEventBankTopupError("sealed v2 raw SHA-256 differs")
    validate_spec(
        value,
        source_manifest_path=source_manifest_path,
        base_v1_spec_path=base_v1_spec_path,
    )
    return value, actual


def build_asset(
    *,
    source_manifest_path: str | Path,
    base_v1_spec_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    value = author_spec(source_manifest_path, base_v1_spec_path)
    summary = validate_spec(
        value,
        source_manifest_path=source_manifest_path,
        base_v1_spec_path=base_v1_spec_path,
    )
    output = Path(output_path)
    _write_create_only(output, value)
    return {
        "output_path": str(output),
        "output_raw_sha256": file_sha256(output),
        **summary,
    }


def materialize_geometry_proxies(
    *, spec: Mapping[str, Any], output_dir: str | Path, ffmpeg_path: str | Path,
    ffprobe_path: str | Path,
) -> dict[str, Any]:
    """Reuse v1's launch-local exact81 black-proxy implementation."""

    try:
        return v1.materialize_geometry_proxies(
            spec=spec,
            output_dir=output_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )
    except v1.SAICPureT2VEventBankError as error:
        raise _error("v2 geometry proxy materialization failed", error)


def materialize_plan(
    *,
    spec_path: str | Path,
    expected_spec_raw_sha256: str,
    source_manifest_path: str | Path,
    base_v1_spec_path: str | Path,
    proxy_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    spec, raw_sha = load_sealed_spec(
        spec_path,
        expected_raw_sha256=expected_spec_raw_sha256,
        source_manifest_path=source_manifest_path,
        base_v1_spec_path=base_v1_spec_path,
    )
    try:
        proxy_receipt = v1.load_proxy_receipt(proxy_receipt_path)
    except v1.SAICPureT2VEventBankError as error:
        raise _error("v2 proxy receipt differs", error)
    proxies = {
        (record["height"], record["width"]): record
        for record in proxy_receipt["records"]
    }
    required = {
        tuple(candidate["source_geometry_hw"])
        for group in spec["groups"]
        for candidate in group["candidates"]
    }
    if set(proxies) != required:
        raise SAICPureT2VEventBankTopupError("v2 proxy geometry coverage differs")
    output = Path(output_dir)
    if (
        not output.is_absolute()
        or output == Path("/")
        or output.exists()
        or output.is_symlink()
    ):
        raise SAICPureT2VEventBankTopupError("v2 plan output must be fresh and absolute")
    output.mkdir(parents=False, exist_ok=False)

    records: list[dict[str, Any]] = []
    for group in spec["groups"]:
        group_dir = output / group["group_id"]
        group_dir.mkdir()
        for candidate in group["candidates"]:
            proxy = proxies[tuple(candidate["source_geometry_hw"])]
            if proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]:
                raise SAICPureT2VEventBankTopupError("v2 proxy aliases source bytes")
            envelope = {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "root_spec_raw_sha256": raw_sha,
                "base_v1_spec_raw_sha256": BASE_V1_SPEC_RAW_SHA256,
                "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
                "group_id": group["group_id"],
                "actor_family": group["actor_family"],
                "visible_gpus": group["visible_gpus"],
                "sampling_contract": SAMPLING_CONTRACT,
                "semantic_input_closure": SEMANTIC_INPUT_CLOSURE,
                "geometry_proxy_contract": GEOMETRY_PROXY_CONTRACT,
                "artifact_authority": ARTIFACT_AUTHORITY,
                "candidate": candidate,
                "geometry_proxy": {
                    "path": proxy["path"],
                    "sha256": proxy["sha256"],
                    "height": proxy["height"],
                    "width": proxy["width"],
                    "source_media_read": False,
                },
            }
            filename = f"{candidate['ordinal']:04d}-{candidate['candidate_id']}.json"
            path = group_dir / filename
            # The terminal rendezvous auditor requires an exact 0444 envelope.
            # Write that mode explicitly: the job-level umask is 077, so merely
            # passing 0444 to os.open would otherwise materialize 0400.
            _write_create_only(path, envelope, mode=0o444)
            records.append(
                {
                    "group_id": group["group_id"],
                    "candidate_id": candidate["candidate_id"],
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    unsigned = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "root_spec_raw_sha256": raw_sha,
        "base_v1_spec_raw_sha256": BASE_V1_SPEC_RAW_SHA256,
        "proxy_receipt_path": str(Path(proxy_receipt_path).resolve(strict=True)),
        "proxy_receipt_sha256": file_sha256(proxy_receipt_path),
        "candidate_count": len(records),
        "top_up_only": True,
        "records": records,
        "event_verified": False,
        "optimizer_authorized": False,
    }
    manifest = {**unsigned, "manifest_digest": object_sha256(unsigned)}
    _write_create_only(output / "manifest.json", manifest, mode=0o400)
    return manifest


def load_candidate_envelope(
    path: str | Path, *, expected_root_spec_sha256: str
) -> dict[str, Any]:
    envelope = _closed(
        _load_json(path, label="v2 candidate envelope"),
        _ENVELOPE_FIELDS,
        label="v2 candidate envelope",
    )
    if (
        envelope["schema_version"] != CANDIDATE_SCHEMA_VERSION
        or envelope["root_spec_raw_sha256"]
        != _sha(expected_root_spec_sha256, label="v2 root spec SHA-256")
        or envelope["base_v1_spec_raw_sha256"] != BASE_V1_SPEC_RAW_SHA256
        or envelope["source_manifest_content_sha256"]
        != SOURCE_MANIFEST_CONTENT_SHA256
        or envelope["sampling_contract"] != SAMPLING_CONTRACT
        or envelope["semantic_input_closure"] != SEMANTIC_INPUT_CLOSURE
        or envelope["geometry_proxy_contract"] != GEOMETRY_PROXY_CONTRACT
        or envelope["artifact_authority"] != ARTIFACT_AUTHORITY
    ):
        raise SAICPureT2VEventBankTopupError("v2 envelope root contract differs")
    layout = {group_id: (actor, gpus) for group_id, actor, gpus in GROUP_LAYOUT}
    if layout.get(envelope["group_id"]) != (
        envelope["actor_family"], envelope["visible_gpus"]
    ):
        raise SAICPureT2VEventBankTopupError("v2 envelope group differs")
    candidate = _closed(envelope["candidate"], _CANDIDATE_FIELDS, label="v2 candidate")
    if candidate["branch"] not in BRANCH_ORDER:
        raise SAICPureT2VEventBankTopupError("v2 envelope branch differs")
    proxy = _closed(
        envelope["geometry_proxy"],
        {"path", "sha256", "height", "width", "source_media_read"},
        label="v2 geometry proxy",
    )
    proxy_path = Path(proxy["path"])
    if (
        not proxy_path.is_absolute()
        or not proxy_path.is_file()
        or proxy_path.is_symlink()
        or file_sha256(proxy_path) != _sha(proxy["sha256"], label="proxy SHA-256")
        or proxy["source_media_read"] is not False
        or [proxy["height"], proxy["width"]] != candidate["source_geometry_hw"]
        or proxy["sha256"] == candidate["source_media_sha256_for_nonuse_audit"]
    ):
        raise SAICPureT2VEventBankTopupError("v2 geometry proxy differs")
    return dict(envelope)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-asset")
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--base-v1-spec", required=True)
    build.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--source-manifest", required=True)
    validate.add_argument("--base-v1-spec", required=True)
    validate.add_argument("--spec", required=True)
    validate.add_argument("--expected-spec-raw-sha256", required=True)
    proxy = sub.add_parser("materialize-proxies")
    proxy.add_argument("--source-manifest", required=True)
    proxy.add_argument("--base-v1-spec", required=True)
    proxy.add_argument("--spec", required=True)
    proxy.add_argument("--expected-spec-raw-sha256", required=True)
    proxy.add_argument("--output-dir", required=True)
    proxy.add_argument("--ffmpeg", required=True)
    proxy.add_argument("--ffprobe", required=True)
    plan = sub.add_parser("materialize-plan")
    plan.add_argument("--source-manifest", required=True)
    plan.add_argument("--base-v1-spec", required=True)
    plan.add_argument("--spec", required=True)
    plan.add_argument("--expected-spec-raw-sha256", required=True)
    plan.add_argument("--proxy-receipt", required=True)
    plan.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-asset":
        result = build_asset(
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
            output_path=args.output,
        )
    elif args.command == "validate":
        spec, raw_sha = load_sealed_spec(
            args.spec,
            expected_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
        )
        result = {
            **validate_spec(
                spec,
                source_manifest_path=args.source_manifest,
                base_v1_spec_path=args.base_v1_spec,
            ),
            "raw_sha256": raw_sha,
        }
    elif args.command == "materialize-proxies":
        spec, _ = load_sealed_spec(
            args.spec,
            expected_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
        )
        result = materialize_geometry_proxies(
            spec=spec,
            output_dir=args.output_dir,
            ffmpeg_path=args.ffmpeg,
            ffprobe_path=args.ffprobe,
        )
    else:
        result = materialize_plan(
            spec_path=args.spec,
            expected_spec_raw_sha256=args.expected_spec_raw_sha256,
            source_manifest_path=args.source_manifest,
            base_v1_spec_path=args.base_v1_spec,
            proxy_receipt_path=args.proxy_receipt,
            output_dir=args.output_dir,
        )
    print(canonical_json_bytes(result).decode("ascii"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

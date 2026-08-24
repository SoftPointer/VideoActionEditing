#!/usr/bin/env python3
"""Validate and canonically materialize the frozen SAIC v1 source set.

This builder deliberately cannot authorize training.  Version 1 binds eight
real exact81 source clips, typed reversible-arrow text, and a prospective
confirmation split, but it has no event-qualified forward/reverse/noop media.
Every accepted row must therefore remain ``terminal_event_verified=false``
and ``optimizer_eligible=false``.  Adding a receipt or flipping either flag is
a schema violation, not an upgrade path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-saic-reversible-source-set-v1"
DATASET_ID = "saic-reversible-source-set-exact81-v1"
BUILD_RECEIPT_SCHEMA = "bernini-saic-reversible-source-set-build-receipt-v1"
ASSET_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "saic_reversible_source_set_v1.json"
)

V17_ROOT = Path(
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/goku_action_wan22_20260730T043022Z/"
    "fullmotion_next1000_v17_20260803T133300Z"
)
V17_WAN_ROOT = V17_ROOT / "wan_next1000_v17"
V17_QWEN_ROOT = V17_ROOT / "qwen_next1000_v17" / "rows"

EXPECTED_ROW_ORDER = (
    ("fit", "dog", "7b88a1ca1f804f41"),
    ("fit", "dog", "841b5e0080a1441d"),
    ("fit", "human", "a35b590961d24694"),
    ("fit", "human", "31c34509415745ca"),
    ("confirmation", "dog", "99cde432839f4240"),
    ("confirmation", "dog", "6ea45d35943742bb"),
    ("confirmation", "human", "311c82f83eca4a7f"),
    ("confirmation", "human", "6d346c38cf504493"),
)
EXPECTED_COUNTS = {
    ("fit", "dog"): 2,
    ("fit", "human"): 2,
    ("confirmation", "dog"): 2,
    ("confirmation", "human"): 2,
}
EXPECTED_STATES = {
    "dog": (
        "dog-stand-sit-reversible-v1",
        "dog_four_leg_stand",
        "dog_stable_sit",
    ),
    "human": (
        "human-one-knee-stand-reversible-v1",
        "human_one_knee_kneel",
        "human_upright_stand",
    ),
}
EXPECTED_TIMELINE = (0, 20, 40, 60, 80)
EXPECTED_FORBIDDEN_INPUTS = (
    "target_video",
    "mask",
    "sam_track",
    "pose",
    "flow",
    "motion_donor",
    "pure_t2v_rgb_as_student_target",
    "pure_t2v_latent_as_student_target",
    "pure_t2v_noise_as_student_input",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IID = re.compile(r"[0-9a-f]{16}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{1,191}")


class SAICReversibleSourceSetError(RuntimeError):
    """Raised before an ambiguous or self-authorizing v1 manifest is used."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise SAICReversibleSourceSetError(
            f"manifest is not canonical-JSON encodable: {error}"
        ) from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SAICReversibleSourceSetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, *, label: str = "manifest") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SAICReversibleSourceSetError(f"{label} contains a non-finite float")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, label=f"{label}[{index}]")


def load_manifest(path: str | Path = ASSET_PATH) -> dict[str, Any]:
    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SAICReversibleSourceSetError(
            f"cannot read SAIC source manifest {source}: {error}"
        ) from error
    if type(raw) is not dict:
        raise SAICReversibleSourceSetError("manifest root must be one exact object")
    _reject_nonfinite(raw)
    return raw


def _exact_keys(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SAICReversibleSourceSetError(
            f"{label} keys differ: observed={observed!r}, expected={sorted(expected)!r}"
        )
    return value


def _exact_bool(value: Any, expected: bool, *, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise SAICReversibleSourceSetError(f"{label} must be exactly {expected}")


def _safe_text(value: Any, *, label: str, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise SAICReversibleSourceSetError(f"{label} is missing or too short")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise SAICReversibleSourceSetError(f"{label} must be ASCII") from error
    return value


def _safe_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SAICReversibleSourceSetError(f"{label} must be a safe identifier")
    return value


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SAICReversibleSourceSetError(f"{label} must be lowercase SHA-256")
    return value


def _ffprobe_video(path: Path) -> Mapping[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        decoded = json.loads(subprocess.check_output(command, text=True))
        streams = decoded["streams"]
        if len(streams) != 1:
            raise ValueError("expected one selected video stream")
        stream = dict(streams[0])
        if "duration" not in stream:
            stream["duration"] = decoded["format"]["duration"]
        return stream
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise SAICReversibleSourceSetError(
            f"ffprobe failed for {path}: {error}"
        ) from error


def _verify_bound_files(row: Mapping[str, Any], *, label: str) -> None:
    source = Path(row["source_video"])
    census = Path(row["source_census"]["path"])
    for path, expected_sha, media_label in (
        (source, row["source_video_sha256"], "source video"),
        (census, row["source_census"]["file_sha256"], "source census"),
    ):
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise SAICReversibleSourceSetError(
                f"{label} {media_label} must be an absolute plain file"
            )
        if file_sha256(path) != expected_sha:
            raise SAICReversibleSourceSetError(f"{label} {media_label} SHA differs")

    observed = _ffprobe_video(source)
    expected = row["media_probe"]
    for key in ("width", "height"):
        if int(observed[key]) != expected[key]:
            raise SAICReversibleSourceSetError(f"{label} ffprobe {key} differs")
    for key in ("r_frame_rate", "avg_frame_rate", "duration"):
        if str(observed[key]) != expected[key]:
            raise SAICReversibleSourceSetError(f"{label} ffprobe {key} differs")
    for key in ("nb_frames", "nb_read_frames"):
        if int(observed[key]) != expected[key]:
            raise SAICReversibleSourceSetError(f"{label} ffprobe {key} differs")

    qwen = load_manifest(census)
    census_binding = row["source_census"]
    if (
        qwen.get("iid") != row["iid"]
        or qwen.get("record_digest") != census_binding["record_digest"]
        or qwen.get("visual_input_digest") != census_binding["visual_input_digest"]
    ):
        raise SAICReversibleSourceSetError(f"{label} Qwen record binding differs")
    source_census = qwen.get("source_census")
    if type(source_census) is not dict or (
        source_census.get("confidence") != "high"
        or source_census.get("all_dynamic_subjects_enumerated") is not True
        or source_census.get("crowd_or_unresolved_motion") is not False
        or source_census.get("camera", {}).get("motion_class") != "locked_off"
    ):
        raise SAICReversibleSourceSetError(f"{label} Qwen source gate differs")


def validate_manifest(
    manifest: Mapping[str, Any], *, verify_bound_files: bool = False
) -> Mapping[str, Any]:
    """Validate the immutable v1 source set and return a non-authorizing summary."""

    root = _exact_keys(
        manifest,
        {
            "schema_version",
            "dataset_id",
            "created_at",
            "scientific_status",
            "contracts",
            "known_exclusions",
            "rows",
        },
        label="manifest",
    )
    if root["schema_version"] != SCHEMA_VERSION or root["dataset_id"] != DATASET_ID:
        raise SAICReversibleSourceSetError("manifest identity differs")
    _safe_text(root["created_at"], label="created_at")

    status = _exact_keys(
        root["scientific_status"],
        {
            "source_manifest_ready",
            "terminal_events_verified",
            "optimizer_updates_authorized",
            "reason",
        },
        label="scientific_status",
    )
    _exact_bool(status["source_manifest_ready"], True, label="source_manifest_ready")
    _exact_bool(
        status["terminal_events_verified"], False, label="terminal_events_verified"
    )
    _exact_bool(
        status["optimizer_updates_authorized"],
        False,
        label="optimizer_updates_authorized",
    )
    reason = _safe_text(status["reason"], label="scientific_status.reason", minimum=40)
    if "no row has" not in reason.lower():
        raise SAICReversibleSourceSetError("scientific status does not disclose missing events")

    contracts = _exact_keys(
        root["contracts"],
        {"media", "timeline_review", "split", "arrow", "forbidden_inputs"},
        label="contracts",
    )
    media_contract = _exact_keys(
        contracts["media"],
        {
            "num_frames",
            "fps_num",
            "fps_den",
            "duration_seconds",
            "source_role",
            "target_video_required",
        },
        label="contracts.media",
    )
    if (
        media_contract["num_frames"] != 81
        or media_contract["fps_num"] != 25
        or media_contract["fps_den"] != 1
        or media_contract["duration_seconds"] != "3.240000"
        or media_contract["source_role"] != "real_source_and_inverse_endpoint_only"
    ):
        raise SAICReversibleSourceSetError("exact81 media contract differs")
    _exact_bool(
        media_contract["target_video_required"], False, label="target_video_required"
    )
    timeline_contract = _exact_keys(
        contracts["timeline_review"],
        {
            "frame_indices",
            "review_method",
            "review_date",
            "full81_independent_human_audit_required_before_optimizer",
        },
        label="contracts.timeline_review",
    )
    if tuple(timeline_contract["frame_indices"]) != EXPECTED_TIMELINE:
        raise SAICReversibleSourceSetError("timeline review indices differ")
    _safe_text(timeline_contract["review_method"], label="timeline review method")
    _safe_text(timeline_contract["review_date"], label="timeline review date")
    _exact_bool(
        timeline_contract["full81_independent_human_audit_required_before_optimizer"],
        True,
        label="full81 independent audit requirement",
    )
    split_contract = _exact_keys(
        contracts["split"],
        {
            "fit_sources_per_actor_family",
            "confirmation_sources_per_actor_family",
            "fit_rollout_seeds_per_source",
            "confirmation_rollout_seeds_per_source",
            "confirmation_never_enters_optimizer",
            "confirmation_iids_have_zero_bernini_receipt_matches_at_selection",
        },
        label="contracts.split",
    )
    if (
        split_contract["fit_sources_per_actor_family"] != 2
        or split_contract["confirmation_sources_per_actor_family"] != 2
        or split_contract["fit_rollout_seeds_per_source"] != 2
        or split_contract["confirmation_rollout_seeds_per_source"] != 3
    ):
        raise SAICReversibleSourceSetError("split size/seed contract differs")
    _exact_bool(
        split_contract["confirmation_never_enters_optimizer"],
        True,
        label="confirmation_never_enters_optimizer",
    )
    _exact_bool(
        split_contract["confirmation_iids_have_zero_bernini_receipt_matches_at_selection"],
        True,
        label="confirmation exposure declaration",
    )
    arrow_contract = _exact_keys(
        contracts["arrow"],
        {
            "forward",
            "inverse",
            "noop",
            "forward_inverse_share_typed_operator",
            "inference_inputs",
        },
        label="contracts.arrow",
    )
    if (
        arrow_contract["forward"] != "q0_to_q1"
        or arrow_contract["inverse"] != "q1_to_q0"
        or arrow_contract["noop"] != "q0_to_q0"
        or arrow_contract["inference_inputs"]
        != ["source_video", "natural_language_instruction"]
    ):
        raise SAICReversibleSourceSetError("typed-arrow/inference contract differs")
    _exact_bool(
        arrow_contract["forward_inverse_share_typed_operator"],
        True,
        label="typed operator sharing",
    )
    if tuple(contracts["forbidden_inputs"]) != EXPECTED_FORBIDDEN_INPUTS:
        raise SAICReversibleSourceSetError("forbidden input closure differs")

    exclusions = root["known_exclusions"]
    if type(exclusions) is not list or [row.get("iid") for row in exclusions] != [
        "a66e6818e4144928",
        "6a7ebea80ba64f18",
    ]:
        raise SAICReversibleSourceSetError("known exclusion closure differs")
    for index, exclusion in enumerate(exclusions):
        _exact_keys(exclusion, {"iid", "reason"}, label=f"known_exclusions[{index}]")
        _safe_text(exclusion["reason"], label=f"known_exclusions[{index}].reason", minimum=40)

    rows = root["rows"]
    if type(rows) is not list or len(rows) != len(EXPECTED_ROW_ORDER):
        raise SAICReversibleSourceSetError("exactly eight ordered rows are required")

    common_keys = {
        "row_id",
        "iid",
        "analysis_split",
        "actor_family",
        "actor_group_id",
        "scene_group_id",
        "action_family_id",
        "initial_state_type",
        "terminal_state_type",
        "selection_origin",
        "source_video",
        "source_video_sha256",
        "media_probe",
        "source_census",
        "five_point_timeline",
        "source_caption",
        "forward_instruction",
        "inverse_instruction",
        "noop_instruction",
        "rollout_seeds",
        "terminal_state_contract",
        "optimizer_eligible",
    }
    row_ids: set[str] = set()
    iids: set[str] = set()
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    actor_groups: set[str] = set()
    scene_groups: set[str] = set()
    all_seeds: set[int] = set()
    observed_counts: Counter[tuple[str, str]] = Counter()

    for index, (row, expected_identity) in enumerate(zip(rows, EXPECTED_ROW_ORDER)):
        label = f"rows[{index}]"
        split, actor, iid = expected_identity
        expected_keys = set(common_keys)
        if split == "confirmation":
            expected_keys.add("confirmation_exposure_scan")
        row = _exact_keys(row, expected_keys, label=label)
        if (
            row["analysis_split"] != split
            or row["actor_family"] != actor
            or row["iid"] != iid
            or _IID.fullmatch(row["iid"]) is None
        ):
            raise SAICReversibleSourceSetError(f"{label} split/family/IID differs")
        observed_counts[(split, actor)] += 1

        row_id = _safe_id(row["row_id"], label=f"{label}.row_id")
        actor_group = _safe_id(row["actor_group_id"], label=f"{label}.actor_group_id")
        scene_group = _safe_id(row["scene_group_id"], label=f"{label}.scene_group_id")
        _safe_text(row["selection_origin"], label=f"{label}.selection_origin")
        if not row_id.endswith(iid):
            raise SAICReversibleSourceSetError(f"{label} row_id is not IID-bound")

        family, initial, terminal = EXPECTED_STATES[actor]
        if (
            row["action_family_id"] != family
            or row["initial_state_type"] != initial
            or row["terminal_state_type"] != terminal
        ):
            raise SAICReversibleSourceSetError(f"{label} typed state arrow differs")

        expected_source = V17_WAN_ROOT / "samples" / iid / "samples" / iid / "source_video.mp4"
        expected_census = V17_QWEN_ROOT / iid / "result.json"
        if row["source_video"] != str(expected_source):
            raise SAICReversibleSourceSetError(f"{label} source path is not canonical")
        source_sha = _sha(row["source_video_sha256"], label=f"{label}.source_video_sha256")

        probe = _exact_keys(
            row["media_probe"],
            {
                "width",
                "height",
                "r_frame_rate",
                "avg_frame_rate",
                "duration",
                "nb_frames",
                "nb_read_frames",
            },
            label=f"{label}.media_probe",
        )
        if (
            type(probe["width"]) is not int
            or probe["width"] <= 0
            or type(probe["height"]) is not int
            or probe["height"] <= 0
            or probe["r_frame_rate"] != "25/1"
            or probe["avg_frame_rate"] != "25/1"
            or probe["duration"] != "3.240000"
            or probe["nb_frames"] != 81
            or probe["nb_read_frames"] != 81
        ):
            raise SAICReversibleSourceSetError(f"{label} is not exact81/25fps")

        census = _exact_keys(
            row["source_census"],
            {
                "path",
                "file_sha256",
                "record_digest",
                "visual_input_digest",
                "confidence",
                "all_dynamic_subjects_enumerated",
                "crowd_or_unresolved_motion",
                "camera_motion_class",
                "i0_state",
                "source_action_signature",
                "source_motion",
            },
            label=f"{label}.source_census",
        )
        if census["path"] != str(expected_census):
            raise SAICReversibleSourceSetError(f"{label} census path is not canonical")
        for key in ("file_sha256", "record_digest", "visual_input_digest"):
            _sha(census[key], label=f"{label}.source_census.{key}")
        if (
            census["confidence"] != "high"
            or census["all_dynamic_subjects_enumerated"] is not True
            or census["crowd_or_unresolved_motion"] is not False
            or census["camera_motion_class"] != "locked_off"
        ):
            raise SAICReversibleSourceSetError(f"{label} source census gate differs")
        for key in ("i0_state", "source_action_signature", "source_motion"):
            _safe_text(census[key], label=f"{label}.source_census.{key}")

        timeline = _exact_keys(
            row["five_point_timeline"],
            {
                "frame_indices",
                "state_labels",
                "initial_state_observed",
                "body_state_held_at_all_reviewed_frames",
                "camera_reframing_observed",
                "full81_qwen_census_high_confidence",
                "full81_independent_human_audit",
            },
            label=f"{label}.five_point_timeline",
        )
        if (
            tuple(timeline["frame_indices"]) != EXPECTED_TIMELINE
            or timeline["state_labels"] != [initial] * len(EXPECTED_TIMELINE)
        ):
            raise SAICReversibleSourceSetError(f"{label} five-point state labels differ")
        _exact_bool(
            timeline["initial_state_observed"], True, label=f"{label} initial state"
        )
        _exact_bool(
            timeline["body_state_held_at_all_reviewed_frames"],
            True,
            label=f"{label} body state hold",
        )
        _exact_bool(
            timeline["camera_reframing_observed"],
            False,
            label=f"{label} camera review",
        )
        _exact_bool(
            timeline["full81_qwen_census_high_confidence"],
            True,
            label=f"{label} Qwen timeline review",
        )
        _exact_bool(
            timeline["full81_independent_human_audit"],
            False,
            label=f"{label} independent full81 audit",
        )

        captions = {
            key: _safe_text(row[key], label=f"{label}.{key}", minimum=100)
            for key in (
                "source_caption",
                "forward_instruction",
                "inverse_instruction",
                "noop_instruction",
            )
        }
        if len(set(captions.values())) != len(captions):
            raise SAICReversibleSourceSetError(f"{label} captions are not distinct")
        if "locked-off" not in captions["source_caption"].lower():
            raise SAICReversibleSourceSetError(f"{label} source caption omits locked camera")
        for key in ("forward_instruction", "inverse_instruction", "noop_instruction"):
            lowered = captions[key].lower()
            if "preserve" not in lowered or "locked camera" not in lowered:
                raise SAICReversibleSourceSetError(f"{label} {key} omits preservation")
        if "never" not in captions["noop_instruction"].lower():
            raise SAICReversibleSourceSetError(f"{label} noop lacks explicit exclusion")
        if actor == "dog":
            if (
                "sit" not in captions["forward_instruction"].lower()
                or "stand" not in captions["inverse_instruction"].lower()
                or "stand" not in captions["noop_instruction"].lower()
                or "sit" not in captions["noop_instruction"].lower()
            ):
                raise SAICReversibleSourceSetError(f"{label} dog arrow text differs")
        else:
            if (
                "stand" not in captions["forward_instruction"].lower()
                or "knee" not in captions["inverse_instruction"].lower()
                or "stand" not in captions["noop_instruction"].lower()
                or "knee" not in captions["noop_instruction"].lower()
            ):
                raise SAICReversibleSourceSetError(f"{label} human arrow text differs")

        seeds = row["rollout_seeds"]
        expected_seed_count = 2 if split == "fit" else 3
        if (
            type(seeds) is not list
            or len(seeds) != expected_seed_count
            or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise SAICReversibleSourceSetError(f"{label} rollout seeds differ")
        if all_seeds.intersection(seeds):
            raise SAICReversibleSourceSetError(f"{label} rollout seeds are reused")
        all_seeds.update(seeds)

        if split == "confirmation":
            exposure = _exact_keys(
                row["confirmation_exposure_scan"],
                {
                    "scan_date",
                    "local_bernini_code_doc_matches",
                    "auh_bernini_code_doc_matches",
                    "auh_bernini_experiment_receipt_matches",
                    "formal_seal_still_required",
                },
                label=f"{label}.confirmation_exposure_scan",
            )
            if (
                exposure["scan_date"] != "2026-08-09"
                or exposure["local_bernini_code_doc_matches"] != 0
                or exposure["auh_bernini_code_doc_matches"] != 0
                or exposure["auh_bernini_experiment_receipt_matches"] != 0
            ):
                raise SAICReversibleSourceSetError(f"{label} exposure scan differs")
            _exact_bool(
                exposure["formal_seal_still_required"],
                True,
                label=f"{label} formal seal requirement",
            )

        terminal_contract = _exact_keys(
            row["terminal_state_contract"],
            {
                "semantic_reachability_reviewed",
                "decoded_terminal_media_path",
                "decoded_terminal_media_sha256",
                "terminal_event_verified",
                "pure_t2v_event_receipt",
            },
            label=f"{label}.terminal_state_contract",
        )
        _exact_bool(
            terminal_contract["semantic_reachability_reviewed"],
            True,
            label=f"{label} semantic reachability",
        )
        if (
            terminal_contract["decoded_terminal_media_path"] is not None
            or terminal_contract["decoded_terminal_media_sha256"] is not None
            or terminal_contract["pure_t2v_event_receipt"] is not None
        ):
            raise SAICReversibleSourceSetError(
                f"{label} v1 must not claim or bind missing terminal media"
            )
        _exact_bool(
            terminal_contract["terminal_event_verified"],
            False,
            label=f"{label} terminal event",
        )
        _exact_bool(row["optimizer_eligible"], False, label=f"{label} optimizer")

        for value, seen, seen_label in (
            (row_id, row_ids, "row IDs"),
            (iid, iids, "IIDs"),
            (row["source_video"], source_paths, "source paths"),
            (source_sha, source_hashes, "source hashes"),
            (actor_group, actor_groups, "actor groups"),
            (scene_group, scene_groups, "scene groups"),
        ):
            if value in seen:
                raise SAICReversibleSourceSetError(f"{label} repeats {seen_label}")
            seen.add(value)

        if verify_bound_files:
            _verify_bound_files(row, label=label)

    if dict(observed_counts) != EXPECTED_COUNTS:
        raise SAICReversibleSourceSetError("fit/confirmation family balance differs")

    return {
        "schema_version": BUILD_RECEIPT_SCHEMA,
        "dataset_id": DATASET_ID,
        "manifest_content_sha256": object_sha256(root),
        "row_count": len(rows),
        "fit_row_count": 4,
        "confirmation_row_count": 4,
        "source_manifest_ready": True,
        "terminal_events_verified": False,
        "optimizer_updates_authorized": False,
        "bound_files_verified": bool(verify_bound_files),
    }


def build_manifest(
    *,
    source_path: str | Path = ASSET_PATH,
    output_path: str | Path,
    verify_bound_files: bool = False,
) -> Mapping[str, Any]:
    """Validate ``source_path`` and create one canonical, create-only copy."""

    manifest = load_manifest(source_path)
    summary = dict(validate_manifest(manifest, verify_bound_files=verify_bound_files))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(manifest) + b"\n"
    try:
        with output.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as error:
        raise SAICReversibleSourceSetError(
            f"refusing to overwrite existing output: {output}"
        ) from error
    summary["output_path"] = str(output)
    summary["output_file_sha256"] = hashlib.sha256(payload).hexdigest()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSET_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify-bound-files",
        action="store_true",
        help="rehash source/Qwen files and rerun ffprobe (intended for AUH)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output is None:
        result = validate_manifest(
            load_manifest(args.source), verify_bound_files=args.verify_bound_files
        )
    else:
        result = build_manifest(
            source_path=args.source,
            output_path=args.output,
            verify_bound_files=args.verify_bound_files,
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

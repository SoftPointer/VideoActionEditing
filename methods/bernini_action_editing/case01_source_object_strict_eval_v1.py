#!/usr/bin/env python3
"""Fail-closed case01 evaluator for identity, object reuse, and action order.

This evaluator is intentionally case-specific.  It replays the immutable
source SAM2/G0 authorities and the locally materialized exact5 media before
turning a structured, all-frame human audit into three conjunctive gates:

1. the rendered dog is the same source dog;
2. the manipulated patient is the original ``bone#1`` (no duplicate prop);
3. ``approach -> contact -> grip -> lift -> hold`` is observed for that track.

The visual observations are not claimed to be an automatic identity or
tracking metric.  Missing, unpinned, malformed, or ambiguous evidence cannot
pass.  A variant passes only when all three gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "bernini-case01-source-object-action-observations-v1"
REPORT_SCHEMA_VERSION = "bernini-case01-source-object-action-strict-report-v1"
CASE_ID = "case01"
IID = "288545b9c031491a"
INSTRUCTION = "Make the dog pick up the bone and hold it in its mouth."
VARIANT_ORDER = (
    "exact_original",
    "codec_only_present",
    "bone_removed",
    "bone_translated_up150",
    "sham_control_up150",
)
ACTION_STAGES = ("approach", "contact", "grip", "lift", "hold")
IDENTITY_CUES = (
    "body_build",
    "head_shape",
    "chest_marking",
    "collar",
    "coat_appearance",
)
SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
EXPECTED_OBSERVATIONS_PATH = (
    "md/action_editing/20260821_man/evidence/"
    "case01_exact5_strict_observations_v1.json"
)
EXPECTED_OBSERVATIONS_SHA256 = (
    "4079312ef815b623b9b6f7fa4659d02187f9e989583b8001e714b3a5e83bccac"
)
EVALUATOR_PATH = "methods/bernini_action_editing/case01_source_object_strict_eval_v1.py"

EXPECTED_AUTHORITIES: Mapping[str, Mapping[str, Any]] = {
    "sam2_receipt": {
        "path": "artifacts/object_grounded_case01_0821_sam2_masklets_r2/receipt.json",
        "sha256": "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50",
        "size": 22_160,
    },
    "sam2_geometry": {
        "path": "artifacts/object_grounded_case01_0821_sam2_masklets_r2/geometry.json",
        "sha256": "2a7daf54a86606002704e6436096a2f04c63260356b08fcd7a5d57d915876157",
        "size": 21_229,
    },
    "sam2_manual_review": {
        "path": "artifacts/object_grounded_case01_0821_sam2_masklets_r2/manual_review_v1.json",
        "sha256": "3f4e407925e4077827acad7499ac33536a7b855bdeeabb027af156b3a6961a4b",
        "size": 1_490,
    },
    "g0_manifest": {
        "path": "artifacts/object_grounded_case01_0821_sparse_g0_v1/manifest.json",
        "sha256": "08079e1b7c35e04c49adee16ce47c42194aba2feab708e71a5799dbb39f9812f",
        "size": 30_931,
    },
    "g0_receipt": {
        "path": "artifacts/object_grounded_case01_0821_sparse_g0_v1/receipt.json",
        "sha256": "266743f9e5c370a64f35b1acde72c29aa3956eec69bb3ffc464c43fe66b0e096",
        "size": 3_666,
    },
    "g0_independent_review": {
        "path": (
            "md/action_editing/20260821_man/evidence/"
            "case01_g0_sparse_independent_review_v1.json"
        ),
        "sha256": "1237f4f57bb1e01aedecd6de3f6f80d4223cdfe14d420c1a455549a7d68aed74",
        "size": 2_117,
    },
    "exact5_postflight": {
        "path": (
            "artifacts/case01_source_bone_exact5_r64_failed_postmortem_20260821/"
            "evidence/postflight-manifest.json"
        ),
        "sha256": "67f046f5439a5f783520708b9add876b4b9cfc0307038cfd3bc6ed17d25ee25c",
        "size": 36_780,
    },
    "exact5_visual_audit": {
        "path": (
            "artifacts/case01_source_bone_exact5_r64_failed_postmortem_20260821/"
            "evidence/visual-audit.md"
        ),
        "sha256": "27bdab041ac122d3dfe1f95ea79c2c1848147f07b7531ad92b57b3dfc454c33a",
        "size": 5_088,
    },
}

_BUNDLE = "artifacts/case01_source_bone_exact5_r64_failed_postmortem_20260821"
EXPECTED_VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "exact_original": {
        "output_path": f"{_BUNDLE}/assets/media/exact_original-partial-output.mp4",
        "output_sha256": "b75bf680815df3015c49f9772610cab4e9433a6bc4554f04f9468156fc3a9a2c",
        "output_size": 7_492_365,
        "sheet_path": f"{_BUNDLE}/assets/sheets/exact_original-all81.jpg",
        "sheet_sha256": "1deb1026b63e2c394eca5136a8025806706aee4eb49f162bbfbc10774cbc1a9f",
        "sheet_size": 860_805,
    },
    "codec_only_present": {
        "output_path": f"{_BUNDLE}/assets/media/codec_only_present-partial-output.mp4",
        "output_sha256": "1c78c4289fc9b9b5bca748e54d8002cb6f51d46fe157974fb265b9311dbc9820",
        "output_size": 7_527_281,
        "sheet_path": f"{_BUNDLE}/assets/sheets/codec_only_present-all81.jpg",
        "sheet_sha256": "c39deeb11c16b431617fb0f1bf71dfff3193a12cb54f9aa4fb3e4ae374083b6d",
        "sheet_size": 855_939,
    },
    "bone_removed": {
        "output_path": f"{_BUNDLE}/assets/media/bone_removed-partial-output.mp4",
        "output_sha256": "177af8fb6d897a17e23b78e9d16f41ef32452d33ea4f16dcb07794bf0374837d",
        "output_size": 7_560_395,
        "sheet_path": f"{_BUNDLE}/assets/sheets/bone_removed-all81.jpg",
        "sheet_sha256": "94ccaa01b8c617d7d568c6a08740ebd188e8c38210bbad43b3f9c86cc3afb864",
        "sheet_size": 836_081,
    },
    "bone_translated_up150": {
        "output_path": f"{_BUNDLE}/assets/media/bone_translated_up150-partial-output.mp4",
        "output_sha256": "e6ad3e896dbb0266f4fb18582c2c66568f26469cfd6e55f245d30c98d13d40e7",
        "output_size": 7_454_622,
        "sheet_path": f"{_BUNDLE}/assets/sheets/bone_translated_up150-all81.jpg",
        "sheet_sha256": "352a91c9003a82c1fd9db3ebc4f263ae3cd59c1b9ce96b23ea24a838131b5503",
        "sheet_size": 855_046,
    },
    "sham_control_up150": {
        "output_path": f"{_BUNDLE}/assets/media/sham_control_up150-partial-output.mp4",
        "output_sha256": "61f8b1907613f731b9c113bcf162f6674b338024f2295fc4ac7e917b4b21bdba",
        "output_size": 7_507_903,
        "sheet_path": f"{_BUNDLE}/assets/sheets/sham_control_up150-all81.jpg",
        "sheet_sha256": "93b9efeca76b682ddf4fbedb5692c23ae8a260958ed1b05a8a18fc3a68f7195e",
        "sheet_size": 854_397,
    },
}

EXPECTED_MEDIA_PROBE = {
    "codec": "h264",
    "width": 480,
    "height": 496,
    "frame_count": 81,
    "fps_num": 25,
    "fps_den": 1,
    "stream_count": 1,
}


class StrictEvalError(RuntimeError):
    """An authority or observation contract is invalid or incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise StrictEvalError("value is not canonical finite JSON") from error


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StrictEvalError(message)


def _require_exact_keys(value: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    expected = set(keys)
    observed = set(value)
    _require(
        observed == expected,
        f"{name} key closure differs: missing={sorted(expected - observed)} "
        f"extra={sorted(observed - expected)}",
    )


def _safe_repo_path(repo_root: Path, relative: str) -> Path:
    _require(isinstance(relative, str) and relative != "", "empty repository path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute(), f"repository path is absolute: {relative}")
    _require("\\" not in relative, f"repository path uses backslash: {relative}")
    _require(
        all(part not in ("", ".", "..") for part in pure.parts),
        f"repository path is not canonical: {relative}",
    )
    root = repo_root.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    resolved = path.resolve(strict=True)
    _require(resolved == path, f"repository path traverses a symlink: {relative}")
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise StrictEvalError(f"repository path escapes root: {relative}") from error
    return resolved


def _stable_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[bytes, str, int]:
    _require(path.is_absolute(), f"file path is not absolute: {path}")
    _require(not path.is_symlink(), f"file is a symlink: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(
            all(getattr(before, key) == getattr(after, key) for key in identity_fields),
            f"file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        _require(observed_sha256 == expected_sha256, f"SHA-256 differs: {path}")
    if expected_size is not None:
        _require(size == expected_size, f"size differs: {path}")
    return b"".join(chunks), observed_sha256, size


def _load_json_bytes(payload: bytes, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StrictEvalError(f"invalid JSON: {name}") from error
    _require(isinstance(value, dict), f"JSON root is not an object: {name}")
    return value


def _probe_video(path: Path) -> Mapping[str, Any]:
    ffprobe = shutil.which("ffprobe")
    _require(ffprobe is not None, "ffprobe is unavailable; media cannot pass")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type,width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _require(completed.returncode == 0, f"ffprobe failed for {path}: {completed.stderr}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise StrictEvalError(f"ffprobe returned invalid JSON for {path}") from error
    streams = payload.get("streams")
    _require(isinstance(streams, list) and len(streams) == 1, f"media stream closure differs: {path}")
    stream = streams[0]
    _require(stream.get("codec_type") == "video", f"sole stream is not video: {path}")
    rate = str(stream.get("avg_frame_rate", "")).split("/")
    _require(len(rate) == 2, f"invalid frame rate: {path}")
    try:
        observed = {
            "codec": str(stream["codec_name"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frame_count": int(stream["nb_frames"]),
            "fps_num": int(rate[0]),
            "fps_den": int(rate[1]),
            "stream_count": len(streams),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise StrictEvalError(f"incomplete ffprobe record: {path}") from error
    _require(observed == EXPECTED_MEDIA_PROBE, f"media probe differs: {path}: {observed}")
    return observed


def _replay_sam2_and_g0(
    repo_root: Path,
    documents: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    receipt = documents["sam2_receipt"]
    geometry = documents["sam2_geometry"]
    manual = documents["sam2_manual_review"]
    g0 = documents["g0_manifest"]
    g0_receipt = documents["g0_receipt"]
    independent = documents["g0_independent_review"]

    _require(
        receipt.get("schema_version") == "bernini-case01-oracle-sam2-masklets-receipt-v1",
        "SAM2 receipt schema differs",
    )
    _require(receipt.get("case_id") == CASE_ID and receipt.get("iid") == IID, "SAM2 case differs")
    _require(receipt.get("status") == "COMPLETE_STAGE0_MASKLET_DIAGNOSTIC", "SAM2 status differs")
    source = receipt.get("source", {})
    _require(source.get("sha256") == SOURCE_SHA256, "SAM2 source hash differs")
    _require(source.get("frame_count") == 81, "SAM2 source frame count differs")

    receipt_outputs = receipt.get("outputs")
    _require(isinstance(receipt_outputs, list), "SAM2 receipt outputs missing")
    output_by_path = {row.get("path"): row for row in receipt_outputs if isinstance(row, dict)}
    required_mask_paths = {
        f"masks/{object_name}/{frame_index:05d}.png"
        for object_name in ("dog", "bone")
        for frame_index in range(81)
    }
    _require(required_mask_paths.issubset(output_by_path), "SAM2 receipt lacks one or more 81-frame masks")
    mask_root = Path(EXPECTED_AUTHORITIES["sam2_receipt"]["path"]).parent
    for relative in sorted(required_mask_paths):
        row = output_by_path[relative]
        _require(set(row) == {"path", "sha256", "size"}, f"SAM2 mask row differs: {relative}")
        mask_path = _safe_repo_path(repo_root, str(mask_root / relative))
        _stable_file(
            mask_path,
            expected_sha256=str(row["sha256"]),
            expected_size=int(row["size"]),
        )

    _require(
        geometry.get("schema_version") == "bernini-case01-oracle-sam2-masklet-geometry-v1",
        "SAM2 geometry schema differs",
    )
    objects = geometry.get("objects")
    _require(isinstance(objects, dict) and set(objects) == {"dog", "bone"}, "SAM2 object closure differs")
    for object_name in ("dog", "bone"):
        rows = objects[object_name]
        _require(isinstance(rows, list) and len(rows) == 81, f"SAM2 {object_name} row count differs")
        for frame_index, row in enumerate(rows):
            _require(row.get("frame_index") == frame_index, f"SAM2 {object_name} frame order differs")
            _require(row.get("visible") is True, f"SAM2 {object_name} visibility differs")
            _require(int(row.get("area", 0)) > 0, f"SAM2 {object_name} area is empty")
    ious = geometry.get("dog_bone_iou")
    _require(isinstance(ious, list) and len(ious) == 81, "SAM2 joint IoU row count differs")
    _require(all(float(value) == 0.0 for value in ious), "source dog and bone masks overlap")

    _require(
        manual.get("schema_version") == "bernini-case01-sam2-masklet-manual-review-v1",
        "SAM2 review schema differs",
    )
    _require(manual.get("joint_tracking_subgate") == "PASS", "SAM2 joint tracking review did not pass")
    _require(manual.get("frame_count_reviewed") == 81, "SAM2 review is not all-frame")
    _require(manual.get("dog_track", {}).get("status") == "PASS", "SAM2 dog review did not pass")
    _require(manual.get("bone_track", {}).get("status") == "PASS", "SAM2 bone review did not pass")

    _require(
        g0.get("schema_version") == "bernini-case01-g0-sparse-annotation-manifest-v1",
        "G0 manifest schema differs",
    )
    _require(g0.get("case_id") == CASE_ID and g0.get("iid") == IID, "G0 case differs")
    frame_schedule = tuple(range(0, 81, 10))
    _require(tuple(g0.get("frame_schedule", ())) == frame_schedule, "G0 frame schedule differs")
    g0_frames = g0.get("frames")
    _require(isinstance(g0_frames, list) and len(g0_frames) == 9, "G0 sparse frame count differs")
    for frame_row, frame_index in zip(g0_frames, frame_schedule):
        _require(frame_row.get("frame_index") == frame_index, "G0 sparse frame order differs")
        annotations = frame_row.get("annotations", {})
        for annotation_name, object_name in (("dog#1", "dog"), ("bone#1", "bone")):
            annotation = annotations.get(annotation_name, {})
            mask_relative = f"masks/{object_name}/{frame_index:05d}.png"
            receipt_row = output_by_path[mask_relative]
            geometry_row = objects[object_name][frame_index]
            _require(annotation.get("sha256") == receipt_row["sha256"], "G0 mask hash differs")
            _require(annotation.get("bbox_xyxy") == geometry_row["bbox_xyxy"], "G0 bbox differs")
            _require(annotation.get("area_pixels") == geometry_row["area"], "G0 area differs")

    _require(
        g0_receipt.get("schema_version") == "bernini-case01-g0-sparse-annotation-receipt-v1",
        "G0 receipt schema differs",
    )
    _require(g0_receipt.get("case_id") == CASE_ID and g0_receipt.get("iid") == IID, "G0 receipt case differs")
    _require(g0_receipt.get("validation", {}).get("exact_sparse_frame_count") == 9, "G0 receipt count differs")

    _require(
        independent.get("schema_version") == "bernini-case01-g0-sparse-independent-review-v1",
        "G0 independent review schema differs",
    )
    _require(independent.get("case_id") == CASE_ID and independent.get("iid") == IID, "G0 review case differs")
    ballot = independent.get("ballot", {})
    required_passes = (
        "dog_identity_binding",
        "bone_identity_binding",
        "head_box_coverage",
        "mouth_box_coverage",
        "safe_background_disjointness",
        "bone_support_proxy_derivation",
        "full_g0_grounding_admission",
    )
    _require(all(ballot.get(key) == "PASS" for key in required_passes), "G0 independent ballot differs")
    _require(ballot.get("p0_count") == 0 and ballot.get("p1_count") == 0, "G0 review has open findings")
    independent_authority = independent.get("authority", {})
    _require(
        independent_authority.get("stage0_receipt_file_sha256")
        == EXPECTED_AUTHORITIES["sam2_receipt"]["sha256"],
        "G0 review does not bind the SAM2 receipt",
    )
    _require(
        independent_authority.get("sparse_manifest_file_sha256")
        == EXPECTED_AUTHORITIES["g0_manifest"]["sha256"],
        "G0 review does not bind the sparse manifest",
    )
    _require(
        independent_authority.get("sparse_receipt_file_sha256")
        == EXPECTED_AUTHORITIES["g0_receipt"]["sha256"],
        "G0 review does not bind the sparse receipt",
    )
    return {
        "source_frame_count": 81,
        "source_dog_mask_count": 81,
        "source_bone_mask_count": 81,
        "source_joint_iou_zero_frames": 81,
        "g0_sparse_frame_count": 9,
        "g0_independent_admission": "PASS",
    }


def _replay_exact5(
    repo_root: Path,
    postflight: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]:
    _require(
        postflight.get("schema_version") == "case01-source-bone-exact5-r64-failure-postflight-v1",
        "exact5 postflight schema differs",
    )
    _require(postflight.get("task_count") == 5, "exact5 task count differs")
    rows = postflight.get("task_rows")
    _require(isinstance(rows, list), "exact5 task rows missing")
    _require(tuple(row.get("variant") for row in rows) == VARIANT_ORDER, "exact5 variant order differs")
    replay_rows: list[Mapping[str, Any]] = []
    rows_by_variant: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        variant = str(row["variant"])
        expected = EXPECTED_VARIANTS[variant]
        output = row.get("output", {})
        _require(output.get("sha256") == expected["output_sha256"], f"postflight output hash differs: {variant}")
        _require(output.get("size") == expected["output_size"], f"postflight output size differs: {variant}")
        postflight_probe = row.get("media_probe", {})
        expected_postflight_probe = {key: value for key, value in EXPECTED_MEDIA_PROBE.items() if key != "stream_count"}
        _require(postflight_probe == expected_postflight_probe, f"postflight media probe differs: {variant}")
        output_path = _safe_repo_path(repo_root, str(expected["output_path"]))
        _, digest, size = _stable_file(
            output_path,
            expected_sha256=str(expected["output_sha256"]),
            expected_size=int(expected["output_size"]),
        )
        probe = _probe_video(output_path)
        sheet_path = _safe_repo_path(repo_root, str(expected["sheet_path"]))
        _stable_file(
            sheet_path,
            expected_sha256=str(expected["sheet_sha256"]),
            expected_size=int(expected["sheet_size"]),
        )
        replay_rows.append(
            {
                "variant": variant,
                "output_path": expected["output_path"],
                "output_sha256": digest,
                "output_size": size,
                "media_probe": probe,
                "all81_sheet_path": expected["sheet_path"],
                "all81_sheet_sha256": expected["sheet_sha256"],
            }
        )
        rows_by_variant[variant] = row
    return replay_rows, rows_by_variant


def _identity_gate(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        identity,
        ("subject_track_id", "identity_switch_observed", "first_mismatch_frame", "cues"),
        "dog_identity",
    )
    _require(identity["subject_track_id"] == "dog#1", "identity subject must be dog#1")
    cues = identity["cues"]
    _require(isinstance(cues, list) and len(cues) == len(IDENTITY_CUES), "identity cue count differs")
    cue_by_name: dict[str, Mapping[str, Any]] = {}
    for cue in cues:
        _require(isinstance(cue, dict), "identity cue is not an object")
        _require_exact_keys(cue, ("name", "source", "output", "preserved"), "identity cue")
        name = cue["name"]
        _require(name in IDENTITY_CUES and name not in cue_by_name, f"identity cue differs: {name}")
        _require(isinstance(cue["preserved"], bool), f"identity cue is ambiguous: {name}")
        cue_by_name[name] = cue
    _require(tuple(cue_by_name) == IDENTITY_CUES, "identity cue order differs")
    switch = identity["identity_switch_observed"]
    _require(isinstance(switch, bool), "identity switch observation is ambiguous")
    first_mismatch = identity["first_mismatch_frame"]
    _require(first_mismatch is None or isinstance(first_mismatch, int), "first mismatch frame differs")
    reasons: list[str] = []
    if switch:
        reasons.append(f"identity_switch_observed_at_frame_{first_mismatch}")
    if not switch and first_mismatch is not None:
        reasons.append("first_mismatch_frame_present_without_identity_switch")
    for name in IDENTITY_CUES:
        cue = cue_by_name[name]
        if not cue["preserved"]:
            reasons.append(
                f"identity_cue_not_preserved:{name}:{cue['source']} -> {cue['output']}"
            )
    return {
        "status": "PASS" if not reasons else "FAIL",
        "subject_track_id": "dog#1",
        "required_cue_count": len(IDENTITY_CUES),
        "preserved_cue_count": sum(bool(cue_by_name[name]["preserved"]) for name in IDENTITY_CUES),
        "reasons": reasons,
    }


def _bone_reuse_gate(bone: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        bone,
        (
            "patient_track_id",
            "input_patient_available",
            "same_instance_continuity",
            "left_initial_support",
            "entered_effector_region",
            "terminal_hold",
            "source_instance_remains_in_background",
            "duplicate_or_substitute_prop",
            "observed_state",
        ),
        "source_bone",
    )
    _require(bone["patient_track_id"] == "bone#1", "bone patient must be bone#1")
    bool_fields = (
        "input_patient_available",
        "left_initial_support",
        "entered_effector_region",
        "terminal_hold",
        "source_instance_remains_in_background",
    )
    _require(all(isinstance(bone[field], bool) for field in bool_fields), "bone facts are ambiguous")
    continuity = bone["same_instance_continuity"]
    _require(continuity in ("PROVEN", "NOT_PROVEN"), "bone continuity status differs")
    duplicate = bone["duplicate_or_substitute_prop"]
    _require(isinstance(duplicate, dict), "duplicate prop fact is not an object")
    _require_exact_keys(duplicate, ("observed", "frame_interval", "description"), "duplicate prop")
    _require(isinstance(duplicate["observed"], bool), "duplicate prop observation is ambiguous")
    interval = duplicate["frame_interval"]
    _require(
        interval is None
        or (
            isinstance(interval, list)
            and len(interval) == 2
            and all(isinstance(value, int) for value in interval)
            and 0 <= interval[0] <= interval[1] <= 80
        ),
        "duplicate prop interval differs",
    )
    _require((interval is not None) == duplicate["observed"], "duplicate prop interval/flag differs")

    reasons: list[str] = []
    if not bone["input_patient_available"]:
        reasons.append("source_bone_not_available_in_intervention_input")
    if continuity != "PROVEN":
        reasons.append("same_source_bone_continuity_not_proven")
    if not bone["left_initial_support"]:
        reasons.append("source_bone_never_left_initial_support")
    if not bone["entered_effector_region"]:
        reasons.append("source_bone_never_entered_dog_mouth_region")
    if not bone["terminal_hold"]:
        reasons.append("source_bone_not_held_at_terminal_frames")
    if bone["source_instance_remains_in_background"]:
        reasons.append("source_bone_remains_in_background")
    if duplicate["observed"]:
        reasons.append("duplicate_or_substitute_prop_observed")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "patient_track_id": "bone#1",
        "same_instance_continuity": continuity,
        "observed_state": bone["observed_state"],
        "duplicate_or_substitute_prop": duplicate,
        "reasons": reasons,
    }


def _action_gate(action: Mapping[str, Any]) -> Mapping[str, Any]:
    _require_exact_keys(
        action,
        ("patient_track_id", "effector_region_id", "minimum_hold_frames", "stages"),
        "action_trace",
    )
    _require(action["patient_track_id"] == "bone#1", "action patient must be bone#1")
    _require(action["effector_region_id"] == "dog#1.mouth", "action effector must be dog#1.mouth")
    minimum_hold = action["minimum_hold_frames"]
    _require(isinstance(minimum_hold, int) and minimum_hold >= 5, "minimum hold duration is too weak")
    stages = action["stages"]
    _require(isinstance(stages, list) and len(stages) == len(ACTION_STAGES), "action stage count differs")
    stage_by_name: dict[str, Mapping[str, Any]] = {}
    reasons: list[str] = []
    starts: list[int] = []
    for stage, expected_name in zip(stages, ACTION_STAGES):
        _require(isinstance(stage, dict), "action stage is not an object")
        _require_exact_keys(stage, ("name", "observed", "frame_interval", "evidence"), "action stage")
        _require(stage["name"] == expected_name, "action stage order differs")
        _require(isinstance(stage["observed"], bool), f"action stage is ambiguous: {expected_name}")
        interval = stage["frame_interval"]
        _require(
            interval is None
            or (
                isinstance(interval, list)
                and len(interval) == 2
                and all(isinstance(value, int) for value in interval)
                and 0 <= interval[0] <= interval[1] <= 80
            ),
            f"action interval differs: {expected_name}",
        )
        _require((interval is not None) == stage["observed"], f"action evidence/interval differs: {expected_name}")
        if not stage["observed"]:
            reasons.append(f"source_bone_stage_not_observed:{expected_name}")
        else:
            starts.append(interval[0])
        stage_by_name[expected_name] = stage
    if len(starts) == len(ACTION_STAGES):
        if any(left >= right for left, right in zip(starts, starts[1:])):
            reasons.append("source_bone_stage_onsets_not_strictly_ordered")
        hold_interval = stage_by_name["hold"]["frame_interval"]
        hold_frames = hold_interval[1] - hold_interval[0] + 1
        if hold_frames < minimum_hold:
            reasons.append(f"source_bone_hold_too_short:{hold_frames}<{minimum_hold}")
    else:
        hold_frames = 0
    return {
        "status": "PASS" if not reasons else "FAIL",
        "patient_track_id": "bone#1",
        "effector_region_id": "dog#1.mouth",
        "required_order": list(ACTION_STAGES),
        "observed_stage_count": sum(bool(stage_by_name[name]["observed"]) for name in ACTION_STAGES),
        "hold_frame_count": hold_frames,
        "reasons": reasons,
    }


def evaluate_variant(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    """Evaluate one already-validated variant observation.

    Kept public for focused tests and future evaluator integration.  This
    function is not tied to the known failing outputs; a complete synthetic
    observation can pass.
    """

    _require_exact_keys(
        variant,
        ("variant", "review_coverage", "dog_identity", "source_bone", "action_trace"),
        "variant",
    )
    variant_name = variant["variant"]
    _require(variant_name in VARIANT_ORDER, f"unknown variant: {variant_name}")
    coverage = variant["review_coverage"]
    _require(isinstance(coverage, dict), "review coverage is not an object")
    _require_exact_keys(
        coverage,
        ("all_81_decoded_frames_reviewed", "source_and_output_pair_reviewed", "frame_range", "frame_count"),
        "review coverage",
    )
    coverage_reasons: list[str] = []
    if coverage["all_81_decoded_frames_reviewed"] is not True:
        coverage_reasons.append("all_81_output_frames_not_reviewed")
    if coverage["source_and_output_pair_reviewed"] is not True:
        coverage_reasons.append("source_output_pair_not_reviewed")
    if coverage["frame_range"] != [0, 80] or coverage["frame_count"] != 81:
        coverage_reasons.append("review_frame_closure_differs")

    identity = _identity_gate(variant["dog_identity"])
    bone = _bone_reuse_gate(variant["source_bone"])
    action = _action_gate(variant["action_trace"])
    gate_statuses = {
        "review_coverage": "PASS" if not coverage_reasons else "FAIL",
        "dog_identity_retention": identity["status"],
        "same_source_bone_reuse": bone["status"],
        "ordered_source_bone_action": action["status"],
    }
    passed = all(status == "PASS" for status in gate_statuses.values())
    return {
        "variant": variant_name,
        "status": "PASS" if passed else "FAIL",
        "conjunction": (
            "review_coverage AND dog_identity_retention AND same_source_bone_reuse "
            "AND ordered_source_bone_action"
        ),
        "gate_statuses": gate_statuses,
        "gates": {
            "review_coverage": {
                "status": gate_statuses["review_coverage"],
                "reasons": coverage_reasons,
            },
            "dog_identity_retention": identity,
            "same_source_bone_reuse": bone,
            "ordered_source_bone_action": action,
        },
    }


def _validate_observation_root(observations: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    _require_exact_keys(
        observations,
        ("schema_version", "case_id", "iid", "instruction", "review_method", "claim_limits", "variants"),
        "observations",
    )
    _require(observations["schema_version"] == SCHEMA_VERSION, "observation schema differs")
    _require(observations["case_id"] == CASE_ID and observations["iid"] == IID, "observation case differs")
    _require(observations["instruction"] == INSTRUCTION, "observation instruction differs")
    review_method = observations["review_method"]
    _require(isinstance(review_method, dict), "review method is not an object")
    _require_exact_keys(
        review_method,
        ("reviewer_role", "all_frame_visual_audit_sha256", "structured_after_review", "automatic_output_tracking"),
        "review method",
    )
    _require(review_method["reviewer_role"] == "independent-all81-visual-auditor", "reviewer role differs")
    _require(
        review_method["all_frame_visual_audit_sha256"]
        == EXPECTED_AUTHORITIES["exact5_visual_audit"]["sha256"],
        "observations do not bind the all-frame audit",
    )
    _require(review_method["structured_after_review"] is True, "review provenance differs")
    _require(review_method["automatic_output_tracking"] is False, "automatic tracking claim is unsupported")
    claim_limits = observations["claim_limits"]
    _require(isinstance(claim_limits, dict), "claim limits are not an object")
    required_false = (
        "automatic_identity_metric_claimed",
        "automatic_source_object_correspondence_claimed",
        "formal_causal_claim_authorized",
        "scientific_claim_authorized",
    )
    _require(all(claim_limits.get(key) is False for key in required_false), "claim limits are too broad")
    variants = observations["variants"]
    _require(isinstance(variants, list) and len(variants) == 5, "observation variant count differs")
    _require(tuple(row.get("variant") for row in variants) == VARIANT_ORDER, "observation variant order differs")
    return variants


def evaluate_bundle(
    repo_root: str | Path,
    observations_path: str = EXPECTED_OBSERVATIONS_PATH,
) -> Mapping[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    _require(root.is_dir(), "repository root is not a directory")

    authority_replay: list[Mapping[str, Any]] = []
    documents: dict[str, Mapping[str, Any]] = {}
    for name, expected in EXPECTED_AUTHORITIES.items():
        path = _safe_repo_path(root, str(expected["path"]))
        payload, digest, size = _stable_file(
            path,
            expected_sha256=str(expected["sha256"]),
            expected_size=int(expected["size"]),
        )
        authority_replay.append(
            {"name": name, "path": expected["path"], "sha256": digest, "size": size, "status": "PASS"}
        )
        if path.suffix == ".json":
            documents[name] = _load_json_bytes(payload, name)

    source_replay = _replay_sam2_and_g0(root, documents)
    media_replay, _ = _replay_exact5(root, documents["exact5_postflight"])

    _require(observations_path == EXPECTED_OBSERVATIONS_PATH, "unapproved observation path")
    observation_file = _safe_repo_path(root, observations_path)
    observation_payload, observation_sha256, observation_size = _stable_file(
        observation_file,
        expected_sha256=EXPECTED_OBSERVATIONS_SHA256,
    )
    observations = _load_json_bytes(observation_payload, "strict observations")
    variant_observations = _validate_observation_root(observations)

    variant_reports = [evaluate_variant(row) for row in variant_observations]
    pass_count = sum(row["status"] == "PASS" for row in variant_reports)
    evaluator_path = _safe_repo_path(root, EVALUATOR_PATH)
    _require(
        evaluator_path == Path(__file__).resolve(strict=True),
        "loaded evaluator does not match repository evaluator path",
    )
    _, evaluator_sha256, evaluator_size = _stable_file(evaluator_path)
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": CASE_ID,
        "iid": IID,
        "instruction": INSTRUCTION,
        "status": "PASS" if pass_count == len(VARIANT_ORDER) else "FAIL",
        "evaluation_status": "COMPLETE_FAIL_CLOSED",
        "success_rule": (
            "Every arm is evaluated independently. An arm passes iff all-frame review, "
            "dog identity retention, same-source-bone reuse, and the ordered five-stage "
            "action all pass. No averaging or compensation is allowed."
        ),
        "counts": {
            "variant_count": len(VARIANT_ORDER),
            "pass_count": pass_count,
            "fail_count": len(VARIANT_ORDER) - pass_count,
        },
        "all_five_failed": pass_count == 0,
        "authority_replay": authority_replay,
        "source_grounding_replay": source_replay,
        "media_replay": media_replay,
        "evaluator_authority": {
            "path": EVALUATOR_PATH,
            "sha256": evaluator_sha256,
            "size": evaluator_size,
        },
        "observation_authority": {
            "path": observations_path,
            "sha256": observation_sha256,
            "size": observation_size,
            "status": "PASS",
        },
        "variants": variant_reports,
        "claim_limits": {
            "diagnostic_failure_report_only": True,
            "partial_exact5_outputs_are_not_formal_results": True,
            "automatic_output_identity_or_correspondence_claimed": False,
            "formal_causal_claim_authorized": False,
            "scientific_claim_authorized": False,
        },
        "limitations": [
            "Output identity and object-lineage facts are structured from the pinned independent all-81-frame visual audit; V1 does not run an automatic output identity model or output SAM2 tracker.",
            "Source SAM2 masks and sparse G0 annotations establish the identities and effector regions in the source only; they do not by themselves prove output correspondence.",
            "The exact5 run failed its historical deterministic parity gate, used fixed order, and has one seed per intervention; these media remain diagnostic partial outputs.",
        ],
    }
    report["report_digest"] = object_sha256(report)
    return report


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--observations", default=EXPECTED_OBSERVATIONS_PATH)
    parser.add_argument("--output")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = evaluate_bundle(arguments.repo_root, arguments.observations)
    except (OSError, StrictEvalError) as error:
        print(json.dumps({"status": "EVIDENCE_INVALID", "error": str(error)}, ensure_ascii=False))
        return 2
    if arguments.output and not arguments.verify_only:
        output = Path(arguments.output)
        if not output.is_absolute():
            output = Path(arguments.repo_root).resolve() / output
        _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass_count": report["counts"]["pass_count"],
                "fail_count": report["counts"]["fail_count"],
                "all_five_failed": report["all_five_failed"],
                "report_digest": report["report_digest"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

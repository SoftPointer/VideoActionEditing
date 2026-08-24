#!/usr/bin/env python3
"""Extract a selection-only MEV840 action summary with frozen SAM2.1.

SAM2 masks and decoded RGB exist only inside this observer process.  Before
publication they are reduced to 21 coordinate-free scalar relation channels.
The selector-facing JSON/NPZ contain no paths, hashes, pixels, masks, boxes,
coordinates, flow, embeddings, model features, latents, or Q/K/V.  A separate
provenance receipt and RGB overlay are quarantined audit artifacts and are
explicitly unauthorized as generator/renderer inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import mev840_coordinate_free_action_oracle_v1 as oracle  # noqa: E402


SPEC_SCHEMA = "mev840-frozen-sam2-action-observer-spec-v1"
RECEIPT_SCHEMA = "mev840-frozen-sam2-action-observer-receipt-v1"
PHASE_FRAMES = tuple(range(0, 81, 4))
ROLE_NAMES = ("human_agent", "moving_object", "recipient", "head")


class FrozenSAM2ActionObserverError(RuntimeError):
    """The observer-only extraction contract was violated."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FrozenSAM2ActionObserverError("spec must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FrozenSAM2ActionObserverError("cannot parse spec") from error
    if not isinstance(value, dict):
        raise FrozenSAM2ActionObserverError("spec must be one JSON object")
    return value


def _validate_spec(spec: Mapping[str, Any]) -> None:
    video = spec.get("video")
    reference = spec.get("source_initial_reference")
    sam2 = spec.get("sam2")
    roles = spec.get("roles")
    limits = spec.get("claim_limits")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("case_id") != "MEV840"
        or not isinstance(video, dict)
        or video.get("role") not in {"real_target_oracle", "generated_candidate", "source_null"}
        or video.get("frame_count") != 81
        or not isinstance(video.get("width"), int)
        or not isinstance(video.get("height"), int)
        or (video["width"], video["height"]) not in {(960, 540), (656, 368)}
        or float(video.get("fps", -1)) != 25.0
        or not isinstance(reference, dict)
        or not isinstance(sam2, dict)
        or sam2.get("checkpoint_sha256")
        != "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        or sam2.get("config_authority_sha256")
        != "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
        or sam2.get("model_cfg") != "configs/sam2.1/sam2.1_hiera_l.yaml"
        or sam2.get("frozen") is not True
        or sam2.get("separate_role_states") is not True
        or spec.get("phase_frames") != list(PHASE_FRAMES)
        or not isinstance(roles, list)
        or [item.get("name") for item in roles] != list(ROLE_NAMES)
        or any(item.get("reviewed_on_shared_frame0") is not True for item in roles)
        or limits
        != {
            "observer_only": True,
            "training_performed": False,
            "optimizer_updates": 0,
            "renderer_forward_calls": 0,
            "generator_read_authorized": False,
            "route_authorized": False,
            "decode_authorized": False,
            "selection_only": True,
            "raw_masks_exported": False,
            "absolute_coordinates_exported_to_representation": False,
            "appearance_features_exported": False,
        }
    ):
        raise FrozenSAM2ActionObserverError("spec semantics differ")
    for label, item in (("video", video), ("source reference", reference)):
        if not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise FrozenSAM2ActionObserverError(f"{label} authority differs")
        if len(item["sha256"]) != 64:
            raise FrozenSAM2ActionObserverError(f"{label} digest differs")
    if video.get("role") == "generated_candidate":
        derivation = reference.get("derivation")
        if (
            not isinstance(derivation, dict)
            or set(derivation)
            != {
                "original_path",
                "original_sha256",
                "original_frame_count",
                "original_fps",
                "original_width",
                "original_height",
                "target_width",
                "target_height",
                "algorithm",
                "ffmpeg_path",
                "ffmpeg_sha256",
            }
            or [
                derivation.get(key)
                for key in (
                    "original_frame_count",
                    "original_fps",
                    "original_width",
                    "original_height",
                    "target_width",
                    "target_height",
                )
            ]
            != [81, 25.0, 1280, 720, video["width"], video["height"]]
            or derivation.get("algorithm")
            != "ffmpeg_scale_bicubic_libx264_preset_veryslow_crf1_yuv420p_r25"
            or derivation.get("ffmpeg_path")
            != "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/mev840_candidate_action_observer_v1_20260822_control/ffmpeg_4.4.2_authority"
            or derivation.get("ffmpeg_sha256")
            != "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
        ):
            raise FrozenSAM2ActionObserverError(
                "generated-candidate source-reference derivation differs"
            )
    for item in roles:
        if set(item) != {"object_id", "name", "box_xyxy", "reviewed_on_shared_frame0"}:
            raise FrozenSAM2ActionObserverError("role prompt ABI differs")
        box = item["box_xyxy"]
        if (
            not isinstance(item["object_id"], int)
            or not isinstance(box, list)
            or len(box) != 4
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
        ):
            raise FrozenSAM2ActionObserverError("role prompt geometry differs")
        x1, y1, x2, y2 = map(float, box)
        if not (
            0 <= x1 < x2 <= int(video["width"])
            and 0 <= y1 < y2 <= int(video["height"])
        ):
            raise FrozenSAM2ActionObserverError("role prompt is outside frame")


def _read_video(path: Path, *, width: int, height: int) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrozenSAM2ActionObserverError("video cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 81 or abs(fps - 25.0) > 1.0e-6:
        raise FrozenSAM2ActionObserverError("decoded temporal contract differs")
    if any(tuple(frame.shape[:2]) != (height, width) for frame in frames):
        raise FrozenSAM2ActionObserverError("decoded spatial contract differs")
    return frames, fps


def _frame0_correspondence(video_frame: Any, reference_frame: Any) -> dict[str, float]:
    left = video_frame.astype(np.float32)
    right = reference_frame.astype(np.float32)
    rmse = float(np.sqrt(np.mean(np.square(left - right))) / 255.0)
    mae = float(np.mean(np.abs(left - right)) / 255.0)
    # This is provenance-only.  It establishes that the human-reviewed boxes
    # refer to the same scene/instances; it is never exported to action.json.
    if not (rmse <= 0.20 and mae <= 0.12):
        raise FrozenSAM2ActionObserverError("frame-0 instance correspondence gate failed")
    return {"normalized_rmse": rmse, "normalized_mae": mae, "gate_passed": True}


def _mask_geometry(mask: Any) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        raise FrozenSAM2ActionObserverError("role mask became empty")
    coordinates = np.stack((xs.astype(np.float64), ys.astype(np.float64)), axis=1)
    center = coordinates.mean(axis=0)
    covariance = np.cov(coordinates - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1.0e-12)
    axis = eigenvectors[:, order[0]]
    angle = math.atan2(float(axis[1]), float(axis[0]))
    width = int(xs.max()) - int(xs.min()) + 1
    height = int(ys.max()) - int(ys.min()) + 1
    return {
        "center": center,
        "area": float(len(xs)),
        "scale": math.sqrt(float(len(xs))),
        "axis_angle": angle,
        "aspect": float(width / max(height, 1)),
        "eccentricity": float(1.0 - eigenvalues[1] / eigenvalues[0]),
    }


def _mask_gap(left: Any, right: Any) -> float:
    import cv2

    if bool(np.logical_and(left, right).any()):
        return 0.0
    distance = cv2.distanceTransform((~left).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[right]
    if not len(values):
        raise FrozenSAM2ActionObserverError("cannot measure role-mask distance")
    return float(values.min())


def _first_persistent(values: Sequence[bool], *, state: bool, run: int = 2) -> int | None:
    for start in range(0, len(values) - run + 1):
        if all(bool(values[index]) is state for index in range(start, start + run)):
            return start
    return None


def _infer_contact_events(
    agent_contacts: Sequence[bool], recipient_gaps: Sequence[float]
) -> tuple[list[bool], int | None, int | None, int | None]:
    if len(agent_contacts) != oracle.PHASE_COUNT or len(recipient_gaps) != oracle.PHASE_COUNT:
        raise FrozenSAM2ActionObserverError("contact-event phase count differs")
    recipient_start = _first_persistent(
        [float(value) <= 0.04 for value in recipient_gaps], state=True, run=2
    )
    recipient_contacts = [
        recipient_start is not None and index >= recipient_start
        for index in range(oracle.PHASE_COUNT)
    ]
    contact_end = _first_persistent(agent_contacts[1:], state=False)
    if contact_end is not None:
        contact_end += 1
    elif not bool(agent_contacts[-1]):
        # A singleton is allowed only at the final observed phase: MEV840's
        # release/withdrawal completes at frame 80, so no future phase exists
        # with which to establish two-phase persistence.
        contact_end = oracle.PHASE_COUNT - 1
    release_candidates = [
        (not bool(agent_contacts[index])) and recipient_contacts[index]
        for index in range(oracle.PHASE_COUNT)
    ]
    release = _first_persistent(release_candidates, state=True)
    if release is None and release_candidates[-1]:
        release = oracle.PHASE_COUNT - 1
    return recipient_contacts, contact_end, recipient_start, release


def _relations_from_masks(masks: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    geometries = {
        name: [_mask_geometry(masks[name][frame]) for frame in PHASE_FRAMES]
        for name in ROLE_NAMES
    }
    human = geometries["human_agent"]
    moving = geometries["moving_object"]
    recipient = geometries["recipient"]
    head = geometries["head"]
    initial_object_recipient = float(
        np.linalg.norm(moving[0]["center"] - recipient[0]["center"])
    )
    initial_agent_object = float(
        np.linalg.norm(moving[0]["center"] - human[0]["center"])
    )
    if initial_object_recipient < 8 or initial_agent_object < 8:
        raise FrozenSAM2ActionObserverError("initial relational normalization is degenerate")
    object_scale0 = moving[0]["scale"]
    object_angle0 = moving[0]["axis_angle"]
    head_aspect0 = head[0]["aspect"]
    head_eccentricity0 = head[0]["eccentricity"]
    agent_contact_threshold = max(4.0, 0.10 * object_scale0)
    raw_rows: list[dict[str, float]] = []
    recipient_center_distances = []
    for index, frame in enumerate(PHASE_FRAMES):
        object_center = moving[index]["center"]
        recipient_center_distances.append(
            float(np.linalg.norm(object_center - recipient[index]["center"]))
        )
        movement = float(np.linalg.norm(object_center - moving[0]["center"]))
        incremental = (
            0.0
            if index == 0
            else float(np.linalg.norm(object_center - moving[index - 1]["center"]))
        )
        agent_gap = _mask_gap(masks["human_agent"][frame], masks["moving_object"][frame])
        angle_delta = moving[index]["axis_angle"] - object_angle0
        axis_change = 1.0 - abs(math.cos(angle_delta))
        head_profile = min(
            abs(math.log(max(head[index]["aspect"], 1.0e-6) / max(head_aspect0, 1.0e-6))) / 0.7
            + abs(head[index]["eccentricity"] - head_eccentricity0),
            1.0,
        )
        raw_rows.append(
            {
                "motion": min(movement / initial_object_recipient, 1.0),
                "incremental": min(incremental / (0.15 * initial_object_recipient), 1.0),
                "agent_gap": agent_gap / initial_agent_object,
                "scale": math.log(max(moving[index]["scale"], 1.0e-6) / object_scale0),
                "axis": min(max(axis_change, 0.0), 1.0),
                "head": head_profile,
                "agent_contact": float(agent_gap <= agent_contact_threshold),
            }
        )
    recipient_min = min(recipient_center_distances)
    denominator = max(recipient_center_distances[0] - recipient_min, 1.0)
    recipient_gaps = [
        max(0.0, (value - recipient_min) / denominator)
        for value in recipient_center_distances
    ]
    agent_contacts = [bool(row["agent_contact"]) for row in raw_rows]
    # Contact is a latched state, not generic 2-D proximity.  See the helper's
    # two-phase approach gate and terminal-only singleton release rule.
    recipient_contacts, contact_end, recipient_start, release = _infer_contact_events(
        agent_contacts, recipient_gaps
    )
    rows = []
    for index, raw in enumerate(raw_rows):
        rows.append(
            {
                "phase_index": index,
                "phase_time": index / 20,
                "object_motion_progress": raw["motion"],
                "object_incremental_motion": raw["incremental"],
                "agent_object_gap_ratio": raw["agent_gap"],
                "object_recipient_gap_ratio": recipient_gaps[index],
                "agent_object_contact": raw["agent_contact"],
                "object_recipient_contact": float(recipient_contacts[index]),
                "object_scale_log_ratio": raw["scale"],
                "object_axis_change": raw["axis"],
                "head_profile_change": raw["head"],
            }
        )
    turn_onset = _first_persistent([row["head"] >= 0.10 for row in raw_rows], state=True)
    turn_peak = int(np.argmax([row["head"] for row in raw_rows]))
    events = {
        "turn_onset": None if turn_onset is None else turn_onset / 20,
        "turn_peak": turn_peak / 20,
        "agent_object_contact_end": None if contact_end is None else contact_end / 20,
        "recipient_contact_start": None if recipient_start is None else recipient_start / 20,
        "release": None if release is None else release / 20,
    }
    representation = oracle.make_representation(
        rows,
        events,
        evidence_boundary="frozen_sam2_geometry_reduced_before_export",
    )
    matrix = np.asarray(
        [[float(row[channel]) for channel in oracle.CHANNELS] for row in rows],
        dtype=np.float32,
    )
    return representation, matrix


def _render_audit(frames: Sequence[Any], masks: Mapping[str, Any], output: Path, fps: float) -> tuple[Path, Path]:
    import cv2

    overlay = output / "AUDIT_ONLY_overlay.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise FrozenSAM2ActionObserverError("audit overlay writer failed")
    colors = {
        "human_agent": (40, 220, 40),
        "moving_object": (30, 80, 255),
        "recipient": (255, 190, 20),
        "head": (255, 30, 220),
    }
    key_frames = []
    for frame_index, frame in enumerate(frames):
        canvas = frame.copy()
        for name in ROLE_NAMES:
            contours, _ = cv2.findContours(
                masks[name][frame_index].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(canvas, contours, -1, colors[name], 2)
        cv2.putText(
            canvas,
            f"AUDIT ONLY | frame {frame_index:02d} | green human red object cyan recipient magenta head",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
        if frame_index in (0, 20, 40, 60, 80):
            review_width = 480
            review_height = max(1, int(round(height * review_width / width)))
            key_frames.append(
                cv2.resize(canvas, (review_width, review_height), interpolation=cv2.INTER_AREA)
            )
    writer.release()
    sheet = output / "AUDIT_ONLY_contact_sheet_f0_20_40_60_80.jpg"
    if not cv2.imwrite(str(sheet), np.concatenate(key_frames, axis=1)):
        raise FrozenSAM2ActionObserverError("audit contact sheet writer failed")
    return overlay, sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise FrozenSAM2ActionObserverError("frozen SAM2 extraction requires Linux")
    spec_path = args.spec.resolve(strict=True)
    spec = _read_json(spec_path)
    _validate_spec(spec)
    video = Path(spec["video"]["path"]).resolve(strict=True)
    reference = Path(spec["source_initial_reference"]["path"]).resolve(strict=True)
    checkpoint = Path(spec["sam2"]["checkpoint_path"]).resolve(strict=True)
    config_authority = Path(spec["sam2"]["config_authority_path"]).resolve(strict=True)
    for path, expected, label in (
        (video, spec["video"]["sha256"], "video"),
        (reference, spec["source_initial_reference"]["sha256"], "source reference"),
        (checkpoint, spec["sam2"]["checkpoint_sha256"], "SAM2 checkpoint"),
        (config_authority, spec["sam2"]["config_authority_sha256"], "SAM2 config"),
    ):
        if file_sha256(path) != expected:
            raise FrozenSAM2ActionObserverError(f"{label} SHA-256 differs")
    reference_derivation = spec["source_initial_reference"].get("derivation")
    if reference_derivation is not None:
        original_reference = Path(reference_derivation["original_path"]).resolve(strict=True)
        ffmpeg_authority = Path(reference_derivation["ffmpeg_path"]).resolve(strict=True)
        if (
            file_sha256(original_reference) != reference_derivation["original_sha256"]
            or file_sha256(ffmpeg_authority) != reference_derivation["ffmpeg_sha256"]
        ):
            raise FrozenSAM2ActionObserverError(
                "source-reference derivation authority bytes differ"
            )
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise FrozenSAM2ActionObserverError("refusing to reuse output directory")

    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise FrozenSAM2ActionObserverError("one visible ROCm/CUDA GPU is required")
    width = int(spec["video"]["width"])
    height = int(spec["video"]["height"])
    frames, fps = _read_video(video, width=width, height=height)
    reference_frames, _ = _read_video(reference, width=width, height=height)
    correspondence = _frame0_correspondence(frames[0], reference_frames[0])
    output.mkdir(mode=0o700, parents=True)
    frame_dir = output / "_ephemeral_jpeg_frames"
    frame_dir.mkdir(mode=0o700)
    for index, frame in enumerate(frames):
        if not cv2.imwrite(
            str(frame_dir / f"{index:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 98],
        ):
            raise FrozenSAM2ActionObserverError("cannot stage SAM2 JPEG frame")
    predictor = build_sam2_video_predictor(
        spec["sam2"]["model_cfg"],
        str(checkpoint),
        device="cuda",
        apply_postprocessing=True,
    )
    masks: dict[str, Any] = {}
    freeze_before = {
        "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad
        ),
    }
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    freeze_before["trainable_after_explicit_freeze"] = sum(
        parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for role in spec["roles"]:
            state = predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=int(role["object_id"]),
                box=np.asarray(role["box_xyxy"], dtype=np.float32),
            )
            role_masks = np.zeros((81, height, width), dtype=np.bool_)
            seen = []
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                frame_index = int(frame_index)
                seen.append(frame_index)
                positions = [int(value) for value in object_ids]
                position = positions.index(int(role["object_id"]))
                value = logits[position]
                if value.ndim == 3:
                    value = value[0]
                role_masks[frame_index] = value.detach().float().cpu().numpy() > 0.0
            if sorted(seen) != list(range(81)):
                raise FrozenSAM2ActionObserverError(f"SAM2 {role['name']} coverage differs")
            masks[role["name"]] = role_masks
            predictor.reset_state(state)
            del state
    representation, matrix = _relations_from_masks(masks)
    action_path = output / "action.json"
    action_path.write_bytes(canonical_bytes(representation))
    npz_path = output / "relations.npz"
    np.savez_compressed(
        npz_path,
        relations=matrix,
        events=np.asarray(
            [
                np.nan if representation["events"][name] is None else representation["events"][name]
                for name in oracle.EVENTS
            ],
            dtype=np.float32,
        ),
    )
    overlay, sheet = _render_audit(frames, masks, output, fps)
    shutil.rmtree(frame_dir)
    del masks, predictor
    torch.cuda.empty_cache()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "case_id": "MEV840",
        "input_authority": {
            "spec_path": str(spec_path),
            "spec_sha256": file_sha256(spec_path),
            "video_path": str(video),
            "video_sha256": file_sha256(video),
            "video_role": spec["video"]["role"],
            "source_initial_reference_path": str(reference),
            "source_initial_reference_sha256": file_sha256(reference),
            "source_initial_reference_derivation": reference_derivation,
            "frame0_correspondence": correspondence,
        },
        "sam2_authority": {
            "model_cfg": spec["sam2"]["model_cfg"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "config_authority_path": str(config_authority),
            "config_authority_sha256": file_sha256(config_authority),
            "separate_role_states": True,
            "freeze": freeze_before,
            "all_81_frames_per_role_observed": True,
        },
        "published": {
            "action_json": {"path": str(action_path), "sha256": file_sha256(action_path)},
            "relations_npz": {"path": str(npz_path), "sha256": file_sha256(npz_path)},
            "audit_overlay": {"path": str(overlay), "sha256": file_sha256(overlay)},
            "audit_contact_sheet": {"path": str(sheet), "sha256": file_sha256(sheet)},
            "representation_digest": representation["representation_digest"],
        },
        "leakage_boundary": {
            "raw_masks_exported": False,
            "decoded_frames_exported": False,
            "sam2_embeddings_exported": False,
            "selector_reads_only_action_json": True,
            "npz_contains_only_relation_matrix_and_event_vector": True,
            "audit_rgb_authorized_for_selector": False,
            "target_or_candidate_media_authorized_for_generator": False,
            "generator_read_authorized": False,
            "renderer_condition_authorized": False,
            "training_authorized": False,
            "optimizer_updates": 0,
        },
    }
    receipt["receipt_sha256"] = oracle.object_sha256(receipt)
    receipt_path = output / "provenance.receipt.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(json.dumps({"output": str(output), "representation_digest": representation["representation_digest"], "receipt_sha256": receipt["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

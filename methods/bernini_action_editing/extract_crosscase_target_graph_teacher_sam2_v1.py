#!/usr/bin/env python3
"""Frozen-SAM2 review-only source/target graph observation scaffold.

This program is deliberately *not* an OCEG implementation.  It observes a
small, human-declared node registry in source and real-target videos, reduces
the masks to coordinate-free node/edge trajectories, and publishes audit RGB
separately.  Every machine relation is a geometry/proximity candidate rather
than a physical-contact claim.  Real-target artifacts are unauthorized for
the generator, renderer, training objective, or model selection.
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


SPEC_SCHEMA = "crosscase-frozen-sam2-graph-teacher-spec-v1"
GRAPH_SCHEMA = "target-graph-teacher-observation-scaffold-v1"
EVENT_SCHEMA = "target-graph-teacher-event-scaffold-v1"
DELTA_SCHEMA = "target-graph-teacher-source-target-delta-v1"
RECEIPT_SCHEMA = "target-graph-teacher-observer-receipt-v1"
PHASE_FRAMES = tuple(range(0, 81, 4))
MEDIA_ROLES = ("source", "real_target_teacher")
ALLOWED_NODE_TYPES = {
    "actor_root", "effector", "object", "tool", "support", "patient",
    "body_part", "distractor",
}
ALLOWED_EDGE_TYPES = {
    "part_of", "near", "approaches", "supports", "contains",
    "manipulates", "appearance_binding",
}
AUTHORITY = {
    "observer_only": True,
    "review_only_graph_teacher": True,
    "teacher_observation_scaffold_not_oceg": True,
    "generator_read_authorized": False,
    "renderer_condition_authorized": False,
    "training_authorized": False,
    "selection_authorized": False,
    "optimizer_updates": 0,
    "renderer_forward_calls": 0,
}


class GraphTeacherObserverError(RuntimeError):
    """The sealed observer contract was violated."""


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


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GraphTeacherObserverError("spec must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise GraphTeacherObserverError("cannot parse spec") from error
    if not isinstance(value, dict):
        raise GraphTeacherObserverError("spec must be one JSON object")
    return value


def _validate_prompt(prompt: Any, *, node_id: str, media_role: str) -> None:
    if prompt is None:
        return
    if not isinstance(prompt, dict) or set(prompt) != {
        "box_xyxy", "frame_index", "review_reliable_start", "review_reliable_end"
    }:
        raise GraphTeacherObserverError(f"{node_id} {media_role} prompt ABI differs")
    box = prompt["box_xyxy"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
    ):
        raise GraphTeacherObserverError(f"{node_id} {media_role} box differs")
    x1, y1, x2, y2 = map(float, box)
    if not (0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 540):
        raise GraphTeacherObserverError(f"{node_id} {media_role} box outside frame")
    frame_index = prompt["frame_index"]
    start = prompt["review_reliable_start"]
    end = prompt["review_reliable_end"]
    if not all(isinstance(value, int) and 0 <= value <= 80 for value in (frame_index, start, end)):
        raise GraphTeacherObserverError(f"{node_id} {media_role} frame window differs")
    if not start <= frame_index <= end:
        raise GraphTeacherObserverError(f"{node_id} {media_role} prompt outside review window")


def _validate_spec(spec: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "case_id", "source", "real_target_teacher", "sam2",
        "phase_frames", "nodes", "edges", "manual_event_program", "claim_limits",
    }
    if set(spec) != required or spec.get("schema_version") != SPEC_SCHEMA:
        raise GraphTeacherObserverError("spec top-level ABI differs")
    case_id = spec.get("case_id")
    if not isinstance(case_id, str) or len(case_id) != 12:
        raise GraphTeacherObserverError("case id differs")
    if spec.get("phase_frames") != list(PHASE_FRAMES):
        raise GraphTeacherObserverError("phase frames differ")
    if spec.get("claim_limits") != AUTHORITY:
        raise GraphTeacherObserverError("claim limits differ")
    for media_role in MEDIA_ROLES:
        media = spec.get(media_role)
        if not isinstance(media, dict) or set(media) != {
            "path", "sha256", "frame_count", "fps", "width", "height"
        }:
            raise GraphTeacherObserverError(f"{media_role} ABI differs")
        if [media.get(key) for key in ("frame_count", "width", "height")] != [81, 960, 540]:
            raise GraphTeacherObserverError(f"{media_role} geometry differs")
        if float(media.get("fps", -1)) != 25.0:
            raise GraphTeacherObserverError(f"{media_role} fps differs")
        if not isinstance(media.get("path"), str) or not isinstance(media.get("sha256"), str):
            raise GraphTeacherObserverError(f"{media_role} authority differs")
        if len(media["sha256"]) != 64:
            raise GraphTeacherObserverError(f"{media_role} hash differs")
    sam2 = spec.get("sam2")
    if not isinstance(sam2, dict) or set(sam2) != {
        "model_cfg", "checkpoint_path", "checkpoint_sha256",
        "config_authority_path", "config_authority_sha256", "frozen",
        "separate_node_states",
    }:
        raise GraphTeacherObserverError("SAM2 ABI differs")
    if (
        sam2["model_cfg"] != "configs/sam2.1/sam2.1_hiera_l.yaml"
        or sam2["checkpoint_sha256"]
        != "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        or sam2["config_authority_sha256"]
        != "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
        or sam2["frozen"] is not True
        or sam2["separate_node_states"] is not True
    ):
        raise GraphTeacherObserverError("SAM2 authority differs")
    nodes = spec.get("nodes")
    if not isinstance(nodes, list) or not (2 <= len(nodes) <= 8):
        raise GraphTeacherObserverError("node registry size differs")
    node_ids = []
    object_ids = []
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {
            "node_id", "object_id", "semantic_type", "identity_authority",
            "source_prompt", "real_target_teacher_prompt", "color_bgr",
        }:
            raise GraphTeacherObserverError("node ABI differs")
        node_id = node["node_id"]
        if not isinstance(node_id, str) or not node_id or node_id in node_ids:
            raise GraphTeacherObserverError("node identity differs")
        if node["semantic_type"] not in ALLOWED_NODE_TYPES:
            raise GraphTeacherObserverError(f"{node_id} type differs")
        if not isinstance(node["identity_authority"], str) or not node["identity_authority"]:
            raise GraphTeacherObserverError(f"{node_id} identity authority differs")
        if not isinstance(node["object_id"], int) or node["object_id"] in object_ids:
            raise GraphTeacherObserverError(f"{node_id} object id differs")
        color = node["color_bgr"]
        if not isinstance(color, list) or len(color) != 3 or not all(
            isinstance(value, int) and 0 <= value <= 255 for value in color
        ):
            raise GraphTeacherObserverError(f"{node_id} color differs")
        _validate_prompt(node["source_prompt"], node_id=node_id, media_role="source")
        _validate_prompt(
            node["real_target_teacher_prompt"],
            node_id=node_id,
            media_role="real_target_teacher",
        )
        if node["source_prompt"] is None and node["real_target_teacher_prompt"] is None:
            raise GraphTeacherObserverError(f"{node_id} has no observation authority")
        node_ids.append(node_id)
        object_ids.append(node["object_id"])
    edges = spec.get("edges")
    if not isinstance(edges, list) or not edges:
        raise GraphTeacherObserverError("edge registry differs")
    edge_ids = []
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {
            "edge_id", "source_node", "target_node", "relation_type", "required_for_action"
        }:
            raise GraphTeacherObserverError("edge ABI differs")
        if edge["edge_id"] in edge_ids or edge["source_node"] not in node_ids or edge["target_node"] not in node_ids:
            raise GraphTeacherObserverError("edge identity differs")
        if edge["source_node"] == edge["target_node"] or edge["relation_type"] not in ALLOWED_EDGE_TYPES:
            raise GraphTeacherObserverError("edge semantics differ")
        if not isinstance(edge["required_for_action"], bool):
            raise GraphTeacherObserverError("edge requirement differs")
        edge_ids.append(edge["edge_id"])
    events = spec.get("manual_event_program")
    if not isinstance(events, list) or not events:
        raise GraphTeacherObserverError("manual event program differs")
    for event in events:
        if not isinstance(event, dict) or set(event) != {
            "event_id", "status", "frame_window", "participant_nodes",
            "evidence_edge", "review_note",
        }:
            raise GraphTeacherObserverError("manual event ABI differs")
        if event["status"] not in {"review_confirmed", "review_unresolved"}:
            raise GraphTeacherObserverError("manual event status differs")
        window = event["frame_window"]
        if not isinstance(window, list) or len(window) != 2 or not all(
            isinstance(value, int) and 0 <= value <= 80 for value in window
        ) or window[0] > window[1]:
            raise GraphTeacherObserverError("manual event window differs")
        if not isinstance(event["participant_nodes"], list) or any(
            value not in node_ids for value in event["participant_nodes"]
        ):
            raise GraphTeacherObserverError("manual event participants differ")
        if event["evidence_edge"] is not None and event["evidence_edge"] not in edge_ids:
            raise GraphTeacherObserverError("manual event edge differs")


def _read_video(path: Path) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise GraphTeacherObserverError("video cannot be decoded")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 81 or abs(fps - 25.0) > 1.0e-6:
        raise GraphTeacherObserverError("decoded temporal contract differs")
    if any(tuple(frame.shape[:2]) != (540, 960) for frame in frames):
        raise GraphTeacherObserverError("decoded spatial contract differs")
    return frames, fps


def _frame0_correspondence(source_frame: Any, target_frame: Any) -> dict[str, Any]:
    left = source_frame.astype(np.float32)
    right = target_frame.astype(np.float32)
    rmse = float(np.sqrt(np.mean(np.square(left - right))) / 255.0)
    mae = float(np.mean(np.abs(left - right)) / 255.0)
    # This is a coarse scene/role sanity gate, not an identity metric.  The
    # object-only grill pair has a small camera/illumination offset at frame 0;
    # its participant binding authority is the separately reviewed prompts.
    if not (rmse <= 0.28 and mae <= 0.14):
        raise GraphTeacherObserverError("source/target frame-0 correspondence failed")
    return {
        "gate_kind": "coarse_scene_role_sanity_not_identity_metric",
        "normalized_rmse": rmse,
        "normalized_mae": mae,
        "rmse_limit": 0.28,
        "mae_limit": 0.14,
        "gate_passed": True,
    }


def _geometry(mask: Any) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 16:
        return None
    coordinates = np.stack((xs.astype(np.float64), ys.astype(np.float64)), axis=1)
    center = coordinates.mean(axis=0)
    covariance = np.cov(coordinates - center, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1.0e-12)
    axis = eigenvectors[:, order[0]]
    return {
        "center": center,
        "area": float(len(xs)),
        "axis_angle": math.atan2(float(axis[1]), float(axis[0])),
        "eccentricity": float(1.0 - eigenvalues[1] / eigenvalues[0]),
    }


def _mask_gap(left: Any, right: Any) -> float:
    import cv2

    if bool(np.logical_and(left, right).any()):
        return 0.0
    distance = cv2.distanceTransform((~left).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[right]
    return float(values.min()) if len(values) else math.inf


def _phase_status(prompt: Any, frame: int, geometry: Any) -> str:
    if prompt is None:
        return "unresolved_not_observed_in_this_media"
    if frame < int(prompt["review_reliable_start"]):
        return "unresolved_before_reviewed_visibility"
    if frame > int(prompt["review_reliable_end"]):
        return "unresolved_after_reviewed_visibility"
    if geometry is None:
        return "unresolved_mask_empty_or_occluded"
    return "observed_in_reviewed_window"


def _track_media(
    predictor: Any,
    frames: Sequence[Any],
    frame_dir: Path,
    nodes: Sequence[Mapping[str, Any]],
    media_role: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch

    masks: dict[str, np.ndarray] = {}
    coverage: dict[str, Any] = {}
    prompt_key = f"{media_role}_prompt"
    for node in nodes:
        node_id = node["node_id"]
        prompt = node[prompt_key]
        role_masks = np.zeros((81, 540, 960), dtype=np.bool_)
        if prompt is None:
            masks[node_id] = role_masks
            coverage[node_id] = {
                "prompted": False,
                "sam2_frames_observed": 0,
                "review_reliable_start": None,
                "review_reliable_end": None,
            }
            continue
        state = predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        frame_index = int(prompt["frame_index"])
        object_id = int(node["object_id"])
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_index,
            obj_id=object_id,
            box=np.asarray(prompt["box_xyxy"], dtype=np.float32),
        )
        seen: set[int] = set()
        directions = (False,) if frame_index == 0 else (False, True)
        for reverse in directions:
            for observed_index, object_ids, logits in predictor.propagate_in_video(
                state,
                start_frame_idx=frame_index,
                reverse=reverse,
            ):
                observed_index = int(observed_index)
                positions = [int(value) for value in object_ids]
                position = positions.index(object_id)
                value = logits[position]
                if value.ndim == 3:
                    value = value[0]
                role_masks[observed_index] = value.detach().float().cpu().numpy() > 0.0
                seen.add(observed_index)
        predictor.reset_state(state)
        del state
        masks[node_id] = role_masks
        coverage[node_id] = {
            "prompted": True,
            "sam2_frames_observed": len(seen),
            "sam2_full_bidirectional_coverage": sorted(seen) == list(range(81)),
            "review_reliable_start": int(prompt["review_reliable_start"]),
            "review_reliable_end": int(prompt["review_reliable_end"]),
        }
    diagonal = math.hypot(960.0, 540.0)
    node_records = []
    geometry_by_node: dict[str, list[Any]] = {}
    for node in nodes:
        node_id = node["node_id"]
        prompt = node[prompt_key]
        geometries = [_geometry(masks[node_id][frame]) for frame in PHASE_FRAMES]
        statuses = [
            _phase_status(prompt, frame, geometry)
            for frame, geometry in zip(PHASE_FRAMES, geometries)
        ]
        geometry_by_node[node_id] = [
            geometry if status == "observed_in_reviewed_window" else None
            for status, geometry in zip(statuses, geometries)
        ]
        reference = next(
            (geometry for status, geometry in zip(statuses, geometries)
             if status == "observed_in_reviewed_window" and geometry is not None),
            None,
        )
        phases = []
        for phase_index, (frame, status, geometry) in enumerate(
            zip(PHASE_FRAMES, statuses, geometries)
        ):
            row: dict[str, Any] = {
                "phase_index": phase_index,
                "phase_time": phase_index / 20,
                "visibility_state": status,
                "observed": status == "observed_in_reviewed_window",
                "motion_from_first_observation_norm": None,
                "area_log_ratio": None,
                "axis_change": None,
                "shape_change": None,
            }
            if row["observed"] and geometry is not None and reference is not None:
                row["motion_from_first_observation_norm"] = float(
                    np.linalg.norm(geometry["center"] - reference["center"]) / diagonal
                )
                row["area_log_ratio"] = float(
                    math.log(max(geometry["area"], 1.0) / max(reference["area"], 1.0))
                )
                angle_delta = geometry["axis_angle"] - reference["axis_angle"]
                row["axis_change"] = float(1.0 - abs(math.cos(angle_delta)))
                row["shape_change"] = float(
                    min(abs(geometry["eccentricity"] - reference["eccentricity"]), 1.0)
                )
            phases.append(row)
        node_records.append({
            "node_id": node_id,
            "semantic_type": node["semantic_type"],
            "identity_authority": node["identity_authority"],
            "tracking_authority": coverage[node_id],
            "phases": phases,
        })
    edge_records = []
    for edge in spec_edges(nodes):
        left_id = edge["source_node"]
        right_id = edge["target_node"]
        phases = []
        baseline_gap = None
        for phase_index, frame in enumerate(PHASE_FRAMES):
            left_geometry = geometry_by_node[left_id][phase_index]
            right_geometry = geometry_by_node[right_id][phase_index]
            row: dict[str, Any] = {
                "phase_index": phase_index,
                "phase_time": phase_index / 20,
                "observation_state": "unresolved",
                "center_distance_norm": None,
                "boundary_gap_norm": None,
                "gap_delta_from_first_joint_observation": None,
                "overlap_iou": None,
                "near_candidate": None,
                "proximity_contact_candidate": None,
            }
            if left_geometry is not None and right_geometry is not None:
                left_mask = masks[left_id][frame]
                right_mask = masks[right_id][frame]
                gap = _mask_gap(left_mask, right_mask) / diagonal
                intersection = float(np.logical_and(left_mask, right_mask).sum())
                union = float(np.logical_or(left_mask, right_mask).sum())
                if baseline_gap is None:
                    baseline_gap = gap
                row.update({
                    "observation_state": "observed_geometry_only",
                    "center_distance_norm": float(
                        np.linalg.norm(left_geometry["center"] - right_geometry["center"])
                        / diagonal
                    ),
                    "boundary_gap_norm": float(gap),
                    "gap_delta_from_first_joint_observation": float(gap - baseline_gap),
                    "overlap_iou": float(intersection / max(union, 1.0)),
                    "near_candidate": bool(gap <= 0.04),
                    "proximity_contact_candidate": bool(gap <= 0.008),
                })
            phases.append(row)
        edge_records.append({
            **edge,
            "claim_boundary": "mask_geometry_proximity_only_not_physical_contact",
            "phases": phases,
        })
    graph = {
        "schema": GRAPH_SCHEMA,
        "case_id": CURRENT_SPEC["case_id"],
        "media_role": media_role,
        "phase_frames": list(PHASE_FRAMES),
        "nodes": node_records,
        "edges": edge_records,
        "authority": dict(AUTHORITY),
        "representation_digest": "",
    }
    graph["representation_digest"] = object_sha256({k: v for k, v in graph.items() if k != "representation_digest"})
    return graph, masks


# Set once after spec validation.  Keeping the tracker function signature small
# avoids ever passing target media into renderer-facing code (there is none).
CURRENT_SPEC: dict[str, Any] = {}


def spec_edges(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    del nodes
    return [dict(edge) for edge in CURRENT_SPEC["edges"]]


def _first_persistent(values: Sequence[bool], run: int = 2) -> int | None:
    for start in range(0, len(values) - run + 1):
        if all(bool(values[index]) for index in range(start, start + run)):
            return start
    return None


def _machine_event_candidates(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = []
    for node in graph["nodes"]:
        values = [phase["motion_from_first_observation_norm"] for phase in node["phases"]]
        observed = [value is not None for value in values]
        onset = _first_persistent([
            value is not None and float(value) >= 0.015 for value in values
        ])
        valid = [(index, float(value)) for index, value in enumerate(values) if value is not None]
        peak = max(valid, key=lambda item: item[1]) if valid else None
        events.append({
            "event_id": f"{node['node_id']}:motion",
            "evidence_kind": "coordinate_free_mask_geometry_candidate",
            "observation_complete": all(observed),
            "motion_onset_phase": onset,
            "motion_peak_phase": None if peak is None else peak[0],
            "motion_peak_norm": None if peak is None else peak[1],
        })
    for edge in graph["edges"]:
        phases = edge["phases"]
        observed = [row["observation_state"] == "observed_geometry_only" for row in phases]
        near = [row["near_candidate"] is True for row in phases]
        events.append({
            "event_id": f"{edge['edge_id']}:proximity",
            "evidence_kind": "mask_proximity_candidate_not_physical_contact",
            "observation_complete": all(observed),
            "first_persistent_near_phase": _first_persistent(near),
            "first_persistent_not_near_phase": _first_persistent([
                row["near_candidate"] is False for row in phases
            ]),
        })
    return events


def _make_delta(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    source_edges = {edge["edge_id"]: edge for edge in source["edges"]}
    rows = []
    for target_edge in target["edges"]:
        source_edge = source_edges[target_edge["edge_id"]]
        phases = []
        for left, right in zip(source_edge["phases"], target_edge["phases"]):
            observed = (
                left["observation_state"] == "observed_geometry_only"
                and right["observation_state"] == "observed_geometry_only"
            )
            phases.append({
                "phase_index": right["phase_index"],
                "observation_state": "paired_observed" if observed else "unresolved",
                "target_minus_source_boundary_gap_norm": (
                    float(right["boundary_gap_norm"] - left["boundary_gap_norm"])
                    if observed else None
                ),
                "target_minus_source_center_distance_norm": (
                    float(right["center_distance_norm"] - left["center_distance_norm"])
                    if observed else None
                ),
            })
        rows.append({
            "edge_id": target_edge["edge_id"],
            "source_node": target_edge["source_node"],
            "target_node": target_edge["target_node"],
            "relation_type": target_edge["relation_type"],
            "required_for_action": target_edge["required_for_action"],
            "phases": phases,
        })
    result = {
        "schema": DELTA_SCHEMA,
        "case_id": source["case_id"],
        "edges": rows,
        "authority": dict(AUTHORITY),
        "representation_digest": "",
    }
    result["representation_digest"] = object_sha256({k: v for k, v in result.items() if k != "representation_digest"})
    return result


def _render_audit(
    frames: Sequence[Any],
    masks: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    media_role: str,
    output: Path,
    fps: float,
) -> tuple[Path, Path]:
    import cv2

    overlay = output / f"AUDIT_ONLY_{media_role}_overlay.mp4"
    writer = cv2.VideoWriter(str(overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    if not writer.isOpened():
        raise GraphTeacherObserverError("audit overlay writer failed")
    key_frames = []
    prompt_key = f"{media_role}_prompt"
    for frame_index, frame in enumerate(frames):
        canvas = frame.copy()
        for node in nodes:
            prompt = node[prompt_key]
            if prompt is None:
                continue
            reliable = int(prompt["review_reliable_start"]) <= frame_index <= int(prompt["review_reliable_end"])
            if reliable:
                contours, _ = cv2.findContours(
                    masks[node["node_id"]][frame_index].astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(canvas, contours, -1, tuple(node["color_bgr"]), 2)
        cv2.putText(
            canvas,
            f"AUDIT ONLY | {media_role} | f{frame_index:02d} | outside reviewed windows = UNKNOWN",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(canvas)
        if frame_index in (0, 20, 40, 60, 80):
            key_frames.append(cv2.resize(canvas, (480, 270), interpolation=cv2.INTER_AREA))
    writer.release()
    sheet = output / f"AUDIT_ONLY_{media_role}_contact_sheet_f0_20_40_60_80.jpg"
    if not cv2.imwrite(str(sheet), np.concatenate(key_frames, axis=1)):
        raise GraphTeacherObserverError("audit sheet writer failed")
    return overlay, sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise GraphTeacherObserverError("frozen SAM2 extraction requires Linux")
    spec_path = args.spec.resolve(strict=True)
    spec = _read_json(spec_path)
    _validate_spec(spec)
    global CURRENT_SPEC
    CURRENT_SPEC = dict(spec)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise GraphTeacherObserverError("fresh output directory required")
    paths = {
        "source": Path(spec["source"]["path"]).resolve(strict=True),
        "real_target_teacher": Path(spec["real_target_teacher"]["path"]).resolve(strict=True),
        "checkpoint": Path(spec["sam2"]["checkpoint_path"]).resolve(strict=True),
        "config": Path(spec["sam2"]["config_authority_path"]).resolve(strict=True),
    }
    expected = {
        "source": spec["source"]["sha256"],
        "real_target_teacher": spec["real_target_teacher"]["sha256"],
        "checkpoint": spec["sam2"]["checkpoint_sha256"],
        "config": spec["sam2"]["config_authority_sha256"],
    }
    for label, path in paths.items():
        if file_sha256(path) != expected[label]:
            raise GraphTeacherObserverError(f"{label} SHA-256 differs")

    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise GraphTeacherObserverError("one visible ROCm/CUDA GPU is required")
    media_frames = {}
    media_fps = {}
    for media_role in MEDIA_ROLES:
        media_frames[media_role], media_fps[media_role] = _read_video(paths[media_role])
    correspondence = _frame0_correspondence(
        media_frames["source"][0], media_frames["real_target_teacher"][0]
    )
    output.mkdir(mode=0o700, parents=True)
    predictor = build_sam2_video_predictor(
        spec["sam2"]["model_cfg"],
        str(paths["checkpoint"]),
        device="cuda",
        apply_postprocessing=True,
    )
    freeze = {
        "parameter_count": sum(parameter.numel() for parameter in predictor.parameters()),
        "trainable_before_explicit_freeze": sum(
            parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad
        ),
    }
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    freeze["trainable_after_explicit_freeze"] = sum(
        parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad
    )
    published: dict[str, dict[str, str]] = {}
    graphs = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for media_role in MEDIA_ROLES:
            frame_dir = output / f"_ephemeral_{media_role}_jpeg_frames"
            frame_dir.mkdir(mode=0o700)
            for index, frame in enumerate(media_frames[media_role]):
                if not cv2.imwrite(
                    str(frame_dir / f"{index:05d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 98],
                ):
                    raise GraphTeacherObserverError("cannot stage SAM2 JPEG frame")
            graph, masks = _track_media(
                predictor, media_frames[media_role], frame_dir, spec["nodes"], media_role
            )
            graphs[media_role] = graph
            graph_path = output / f"{media_role}_graph.json"
            graph_path.write_bytes(canonical_bytes(graph))
            overlay, sheet = _render_audit(
                media_frames[media_role], masks, spec["nodes"], media_role,
                output, media_fps[media_role],
            )
            shutil.rmtree(frame_dir)
            published[f"{media_role}_graph"] = {
                "path": str(graph_path), "sha256": file_sha256(graph_path)
            }
            published[f"{media_role}_audit_overlay"] = {
                "path": str(overlay), "sha256": file_sha256(overlay)
            }
            published[f"{media_role}_audit_contact_sheet"] = {
                "path": str(sheet), "sha256": file_sha256(sheet)
            }
            del masks
    event_scaffold = {
        "schema": EVENT_SCHEMA,
        "case_id": spec["case_id"],
        "manual_review_program": spec["manual_event_program"],
        "machine_geometry_candidates": _machine_event_candidates(graphs["real_target_teacher"]),
        "claim_boundary": (
            "manual frame-window observations plus frozen-SAM2 geometry candidates; "
            "not a learned event model, not physical-contact certification, not OCEG"
        ),
        "authority": dict(AUTHORITY),
        "representation_digest": "",
    }
    event_scaffold["representation_digest"] = object_sha256({
        key: value for key, value in event_scaffold.items() if key != "representation_digest"
    })
    event_path = output / "target_event_scaffold.json"
    event_path.write_bytes(canonical_bytes(event_scaffold))
    published["target_event_scaffold"] = {
        "path": str(event_path), "sha256": file_sha256(event_path)
    }
    delta = _make_delta(graphs["source"], graphs["real_target_teacher"])
    delta_path = output / "source_target_graph_delta.json"
    delta_path.write_bytes(canonical_bytes(delta))
    published["source_target_graph_delta"] = {
        "path": str(delta_path), "sha256": file_sha256(delta_path)
    }
    del predictor
    torch.cuda.empty_cache()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "case_id": spec["case_id"],
        "observer_class": "teacher_observation_scaffold_not_oceg",
        "input_authority": {
            "spec_path": str(spec_path),
            "spec_sha256": file_sha256(spec_path),
            "source_path": str(paths["source"]),
            "source_sha256": file_sha256(paths["source"]),
            "real_target_teacher_path": str(paths["real_target_teacher"]),
            "real_target_teacher_sha256": file_sha256(paths["real_target_teacher"]),
            "source_target_frame0_correspondence": correspondence,
        },
        "sam2_authority": {
            "model_cfg": spec["sam2"]["model_cfg"],
            "checkpoint_path": str(paths["checkpoint"]),
            "checkpoint_sha256": file_sha256(paths["checkpoint"]),
            "config_authority_path": str(paths["config"]),
            "config_authority_sha256": file_sha256(paths["config"]),
            "separate_node_states": True,
            "freeze": freeze,
        },
        "published": published,
        "leakage_boundary": {
            "raw_masks_exported": False,
            "decoded_frames_exported": False,
            "sam2_embeddings_exported": False,
            "absolute_coordinates_exported_to_graph": False,
            "audit_rgb_authorized_for_generator": False,
            "target_graph_authorized_for_generator": False,
            "target_graph_authorized_for_renderer": False,
            "target_graph_authorized_for_training": False,
            "target_graph_authorized_for_selection": False,
            "optimizer_updates": 0,
            "renderer_forward_calls": 0,
        },
    }
    receipt["receipt_sha256"] = object_sha256(receipt)
    receipt_path = output / "provenance.receipt.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(json.dumps({
        "case_id": spec["case_id"],
        "output": str(output),
        "receipt_sha256": receipt["receipt_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

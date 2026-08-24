#!/usr/bin/env python3
"""Track the reviewed case01 dog and bone boxes with frozen SAM2.1.

This is a Stage-0 diagnostic, not an editor and not a training program.  It
materializes auditable 81-frame binary masklets, an overlay video/contact
sheet, per-frame geometry and a receipt.  A poor masklet is still published
as a diagnostic result; it never silently becomes renderer authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


SCHEMA = "bernini-case01-oracle-sam2-masklets-receipt-v1"
EXPECTED_SPEC_SCHEMA = "bernini-case01-oracle-sam2-boxes-v1"
KEY_FRAMES = (0, 20, 40, 60, 80)


class OracleMaskletError(RuntimeError):
    """The frozen Stage-0 masklet contract was violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def read_spec(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise OracleMaskletError("spec must be one regular named file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise OracleMaskletError("spec JSON differs") from error
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_SPEC_SCHEMA:
        raise OracleMaskletError("spec schema differs")
    return value


def validate_spec(value: dict[str, Any]) -> None:
    source = value.get("source")
    sam2 = value.get("sam2")
    objects = value.get("frame0_objects")
    limits = value.get("claim_limits")
    if (
        value.get("case_id") != "case01"
        or value.get("iid") != "288545b9c031491a"
        or value.get("split") != "test-heldout-canary-no-training"
        or not isinstance(source, dict)
        or source.get("sha256")
        != "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
        or source.get("frame_count") != 81
        or source.get("fps") != 25.0
        or source.get("width") != 704
        or source.get("height") != 736
        or not isinstance(sam2, dict)
        or sam2.get("checkpoint_sha256")
        != "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
        or sam2.get("config_authority_sha256")
        != "1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107"
        or not isinstance(objects, list)
        or len(objects) != 2
        or [row.get("object_id") for row in objects] != [1, 2]
        or [row.get("name") for row in objects] != ["dog", "bone"]
        or [row.get("role") for row in objects] != ["actor", "patient"]
        or any(row.get("reviewed_by_human") is not True for row in objects)
        or not isinstance(limits, dict)
        or limits.get("training_performed") is not False
        or limits.get("optimizer_updates") != 0
        or limits.get("renderer_inference_performed") is not False
        or limits.get("manual_masklet_review_required") is not True
        or limits.get("single_case_scientific_claim_authorized") is not False
    ):
        raise OracleMaskletError("case01 Stage-0 authority differs")
    for row in objects:
        box = row.get("box_xyxy")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(type(item) in (int, float) and math.isfinite(float(item)) for item in box)
        ):
            raise OracleMaskletError("reviewed object box differs")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 704 and 0 <= y1 < y2 <= 736):
            raise OracleMaskletError("reviewed object box is outside the source frame")


def geometry(mask: Any) -> dict[str, Any]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"visible": False, "area": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "visible": True,
        "area": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def mask_iou(left: Any, right: Any) -> float:
    import numpy as np

    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if sys.platform != "linux":
        raise OracleMaskletError("real SAM2 materialization requires Linux")
    spec_path = args.spec.resolve(strict=True)
    spec = read_spec(spec_path)
    validate_spec(spec)
    source = Path(spec["source"]["path"]).resolve(strict=True)
    checkpoint = Path(spec["sam2"]["checkpoint_path"]).resolve(strict=True)
    config_authority = Path(spec["sam2"]["config_authority_path"]).resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise OracleMaskletError("refusing to reuse a masklet output directory")
    if sha256(source) != spec["source"]["sha256"]:
        raise OracleMaskletError("source SHA-256 differs")
    if sha256(checkpoint) != spec["sam2"]["checkpoint_sha256"]:
        raise OracleMaskletError("SAM2 checkpoint SHA-256 differs")
    if sha256(config_authority) != spec["sam2"]["config_authority_sha256"]:
        raise OracleMaskletError("SAM2 config authority SHA-256 differs")

    import cv2
    import numpy as np
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise OracleMaskletError("one ROCm/CUDA-visible GPU is required")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OracleMaskletError("source video could not be opened")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[Any] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if (
        len(frames) != spec["source"]["frame_count"]
        or width != spec["source"]["width"]
        or height != spec["source"]["height"]
        or abs(fps - spec["source"]["fps"]) > 1.0e-6
    ):
        raise OracleMaskletError("decoded source media contract differs")

    output.mkdir(mode=0o700, parents=True)
    frame_dir = output / "_decoded_jpeg_frames"
    frame_dir.mkdir(mode=0o700)
    for index, frame in enumerate(frames):
        path = frame_dir / f"{index:05d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise OracleMaskletError("failed to materialize SAM2 JPEG frames")

    predictor = build_sam2_video_predictor(
        spec["sam2"]["model_cfg"],
        str(checkpoint),
        device="cuda",
        apply_postprocessing=True,
    )
    state = predictor.init_state(
        video_path=str(frame_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    object_ids = [int(row["object_id"]) for row in spec["frame0_objects"]]
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for row in spec["frame0_objects"]:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=0,
                obj_id=int(row["object_id"]),
                box=np.asarray(row["box_xyxy"], dtype=np.float32),
            )
        masks = {
            object_id: np.zeros((len(frames), height, width), dtype=np.bool_)
            for object_id in object_ids
        }
        seen_frames: list[int] = []
        for frame_index, out_ids, logits in predictor.propagate_in_video(state):
            frame_index = int(frame_index)
            seen_frames.append(frame_index)
            for position, object_id in enumerate(out_ids):
                object_id = int(object_id)
                value = logits[position]
                if value.ndim == 3:
                    value = value[0]
                masks[object_id][frame_index] = value.detach().to("cpu").numpy() > 0.0
    if sorted(seen_frames) != list(range(len(frames))):
        raise OracleMaskletError("SAM2 did not propagate exactly 81 ordered frames")

    mask_root = output / "masks"
    mask_root.mkdir(mode=0o700)
    name_by_id = {int(row["object_id"]): str(row["name"]) for row in spec["frame0_objects"]}
    frame_rows: dict[str, list[dict[str, Any]]] = {}
    output_files: list[Path] = []
    for object_id in object_ids:
        name = name_by_id[object_id]
        directory = mask_root / name
        directory.mkdir(mode=0o700)
        rows = []
        for frame_index in range(len(frames)):
            mask = masks[object_id][frame_index]
            path = directory / f"{frame_index:05d}.png"
            if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                raise OracleMaskletError("failed to publish a binary mask frame")
            output_files.append(path)
            rows.append({"frame_index": frame_index, **geometry(mask)})
        frame_rows[name] = rows

    overlay_path = output / "overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OracleMaskletError("overlay video writer could not be opened")
    overlay_frames: list[Any] = []
    colors = {1: np.asarray([255, 210, 20], dtype=np.float32), 2: np.asarray([30, 80, 255], dtype=np.float32)}
    for frame_index, frame in enumerate(frames):
        canvas = frame.astype(np.float32)
        for object_id in object_ids:
            mask = masks[object_id][frame_index]
            canvas[mask] = 0.62 * canvas[mask] + 0.38 * colors[object_id]
        rendered = np.clip(canvas, 0, 255).astype(np.uint8)
        cv2.putText(
            rendered,
            f"frame {frame_index:02d} | cyan=dog red=bone",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(rendered)
        if frame_index in KEY_FRAMES:
            overlay_frames.append(cv2.resize(rendered, (352, 368), interpolation=cv2.INTER_AREA))
    writer.release()
    output_files.append(overlay_path)
    sheet_path = output / "contact_sheet_0_20_40_60_80.jpg"
    if not cv2.imwrite(str(sheet_path), np.concatenate(overlay_frames, axis=1)):
        raise OracleMaskletError("contact sheet publication failed")
    output_files.append(sheet_path)

    geometry_path = output / "geometry.json"
    geometry_payload = {
        "schema_version": "bernini-case01-oracle-sam2-masklet-geometry-v1",
        "objects": frame_rows,
        "dog_bone_iou": [
            mask_iou(masks[1][index], masks[2][index]) for index in range(len(frames))
        ],
    }
    geometry_path.write_bytes(canonical_bytes(geometry_payload))
    output_files.append(geometry_path)

    visible_counts = {
        name: sum(1 for row in rows if row["visible"])
        for name, rows in frame_rows.items()
    }
    areas = {
        name: [int(row["area"]) for row in rows if row["visible"]]
        for name, rows in frame_rows.items()
    }
    max_iou = max(geometry_payload["dog_bone_iou"])
    diagnostic_gate = {
        "all_81_frames_visible": all(value == 81 for value in visible_counts.values()),
        "dog_bone_max_iou_below_0p10": max_iou < 0.10,
        "bone_minimum_area_at_least_16_pixels": min(areas["bone"], default=0) >= 16,
        "bone_area_max_to_min_ratio_at_most_20": (
            max(areas["bone"], default=0) / max(min(areas["bone"], default=0), 1)
            <= 20.0
        ),
        "manual_full_masklet_review_required": True,
    }
    diagnostic_gate["automatic_geometry_gate_pass"] = all(
        diagnostic_gate[key]
        for key in (
            "all_81_frames_visible",
            "dog_bone_max_iou_below_0p10",
            "bone_minimum_area_at_least_16_pixels",
            "bone_area_max_to_min_ratio_at_most_20",
        )
    )
    receipt = {
        "schema_version": SCHEMA,
        "status": "COMPLETE_STAGE0_MASKLET_DIAGNOSTIC",
        "case_id": spec["case_id"],
        "iid": spec["iid"],
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "frame_count": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
        },
        "spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
        "sam2": {
            "model_cfg": spec["sam2"]["model_cfg"],
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "config_authority_path": str(config_authority),
            "config_authority_sha256": sha256(config_authority),
            "torch_version": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "device_name": torch.cuda.get_device_name(0),
        },
        "reviewed_frame0_objects": spec["frame0_objects"],
        "visible_frame_counts": visible_counts,
        "area_summary": {
            name: {
                "minimum": min(values, default=0),
                "maximum": max(values, default=0),
                "median": float(np.median(values)) if values else 0.0,
            }
            for name, values in areas.items()
        },
        "dog_bone_max_iou": max_iou,
        "diagnostic_gate": diagnostic_gate,
        "outputs": [
            {
                "path": str(path.relative_to(output)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_files)
        ],
        "claim_limits": spec["claim_limits"],
    }
    receipt["receipt_digest"] = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    receipt_path = output / "receipt.json"
    receipt_path.write_bytes(canonical_bytes(receipt))
    print(canonical_bytes(receipt).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

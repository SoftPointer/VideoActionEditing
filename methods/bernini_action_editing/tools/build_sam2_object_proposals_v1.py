#!/usr/bin/env python3
"""Generate auditable source-object proposals with SAM2, without a VLM.

The script only reads a source video.  SAM2 automatic masks are ranked using
source-derived geometry and RGB statistics: a moving lower-frame mask is used
as the actor proposal, while small, low-saturation masks near its lower edge
become candidate manipulated objects.  It writes masks, JSON, and overlays for
human audit before any proposal is allowed into an editing run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_video(path: Path, *, width: int, height: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("source video could not be opened")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 81:
        raise RuntimeError(f"source video must contain 81 frames, got {len(frames)}")
    return frames


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 0.0


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("SAM2 returned an empty mask")
    return float(xs.mean()), float(ys.mean())


def actor_score(
    annotation: dict,
    motion: np.ndarray,
    *,
    image_area: int,
    height: int,
) -> float:
    mask = annotation["segmentation"]
    area_fraction = float(annotation["area"]) / float(image_area)
    if not 0.003 <= area_fraction <= 0.18:
        return -math.inf
    _, center_y = mask_center(mask)
    lower_prior = min(1.0, max(0.0, center_y / float(height)))
    motion_score = float(motion[mask].mean()) / 255.0
    quality = 0.5 * float(annotation["predicted_iou"]) + 0.5 * float(
        annotation["stability_score"]
    )
    size_prior = math.exp(-abs(math.log(area_fraction) - math.log(0.035)))
    return 0.52 * motion_score + 0.18 * lower_prior + 0.15 * quality + 0.15 * size_prior


def stone_score(
    annotation: dict,
    image: np.ndarray,
    actor_mask: np.ndarray,
    actor_bbox: tuple[float, float, float, float],
) -> tuple[float, dict[str, float]]:
    height, width = image.shape[:2]
    image_area = height * width
    mask = annotation["segmentation"]
    area_fraction = float(annotation["area"]) / float(image_area)
    if not 0.00008 <= area_fraction <= 0.025 or mask_iou(mask, actor_mask) > 0.02:
        return -math.inf, {}
    center_x, center_y = mask_center(mask)
    actor_x, actor_y, actor_w, actor_h = actor_bbox
    target_x = actor_x + 0.5 * actor_w
    target_y = actor_y + actor_h
    diagonal = math.hypot(width, height)
    distance = math.hypot(center_x - target_x, center_y - target_y) / diagonal
    near_actor = math.exp(-distance / 0.20)
    lower_edge = math.exp(-abs(center_y - target_y) / (0.20 * height))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(hsv[..., 1][mask].mean()) / 255.0
    grayness = 1.0 - mean_saturation
    size_prior = math.exp(
        -abs(math.log(area_fraction) - math.log(0.0015)) / 1.4
    )
    quality = 0.5 * float(annotation["predicted_iou"]) + 0.5 * float(
        annotation["stability_score"]
    )
    score = (
        0.29 * near_actor
        + 0.21 * lower_edge
        + 0.20 * grayness
        + 0.15 * size_prior
        + 0.15 * quality
    )
    return score, {
        "area_fraction": area_fraction,
        "center_x": center_x,
        "center_y": center_y,
        "distance_to_actor_lower_edge": distance,
        "grayness": grayness,
        "near_actor": near_actor,
        "lower_edge_prior": lower_edge,
        "size_prior": size_prior,
        "quality": quality,
    }


def overlay_proposals(
    image: np.ndarray,
    actor_mask: np.ndarray,
    proposals: list[dict],
) -> np.ndarray:
    canvas = image.astype(np.float32).copy()
    actor_color = np.array([20, 220, 255], dtype=np.float32)
    canvas[actor_mask] = 0.62 * canvas[actor_mask] + 0.38 * actor_color
    colors = [
        (255, 70, 70),
        (70, 255, 100),
        (255, 200, 40),
        (190, 80, 255),
        (50, 220, 255),
        (255, 110, 210),
        (140, 255, 50),
        (255, 145, 50),
    ]
    for rank, proposal in enumerate(proposals):
        mask = proposal["mask"]
        color = np.array(colors[rank % len(colors)], dtype=np.float32)
        canvas[mask] = 0.52 * canvas[mask] + 0.48 * color
        x, y, w, h = (int(round(item)) for item in proposal["bbox"])
        bgr = tuple(int(item) for item in colors[rank % len(colors)][::-1])
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr, 2)
        cv2.putText(
            canvas,
            f"P{rank} {proposal['score']:.3f}",
            (max(0, x), max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            bgr,
            2,
            cv2.LINE_AA,
        )
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--sam2-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=576)
    args = parser.parse_args()

    source = Path(args.source_video).resolve(strict=True)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    sam2_root = Path(args.sam2_root).resolve(strict=True)
    output = Path(args.output_dir)
    if output.exists():
        raise RuntimeError("refusing to overwrite an existing proposal directory")
    if sha256(source) != args.expected_source_sha256:
        raise RuntimeError("source SHA-256 differs")
    if sha256(checkpoint) != args.expected_checkpoint_sha256:
        raise RuntimeError("SAM2 checkpoint SHA-256 differs")
    if not (sam2_root / "sam2" / "automatic_mask_generator.py").is_file():
        raise RuntimeError("SAM2 source tree differs")

    import sys

    sys.path.insert(0, str(sam2_root))
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 proposal generation requires one GPU")
    frames = read_video(source, width=args.width, height=args.height)
    image = frames[0]
    sample_indices = tuple(range(10, 81, 10))
    motion = np.mean(
        [
            np.abs(frames[index].astype(np.float32) - image.astype(np.float32)).mean(
                axis=-1
            )
            for index in sample_indices
        ],
        axis=0,
    )
    model = build_sam2(
        "sam2_hiera_t.yaml",
        str(checkpoint),
        device="cuda",
        apply_postprocessing=True,
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.88,
        box_nms_thresh=0.70,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=12,
        output_mode="binary_mask",
    )
    annotations = generator.generate(image)
    if not annotations:
        raise RuntimeError("SAM2 returned no masks")
    image_area = args.width * args.height
    actor_index = max(
        range(len(annotations)),
        key=lambda index: actor_score(
            annotations[index], motion, image_area=image_area, height=args.height
        ),
    )
    actor = annotations[actor_index]
    actor_mask = actor["segmentation"].astype(bool)
    actor_bbox = tuple(float(item) for item in actor["bbox"])

    ranked = []
    for index, annotation in enumerate(annotations):
        if index == actor_index:
            continue
        score, diagnostics = stone_score(annotation, image, actor_mask, actor_bbox)
        if not math.isfinite(score):
            continue
        ranked.append(
            {
                "sam_index": index,
                "score": score,
                "bbox": [float(item) for item in annotation["bbox"]],
                "area": int(annotation["area"]),
                "predicted_iou": float(annotation["predicted_iou"]),
                "stability_score": float(annotation["stability_score"]),
                "diagnostics": diagnostics,
                "mask": annotation["segmentation"].astype(bool),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict] = []
    for proposal in ranked:
        if all(mask_iou(proposal["mask"], row["mask"]) < 0.65 for row in selected):
            selected.append(proposal)
        if len(selected) == 8:
            break
    if len(selected) < 4:
        raise RuntimeError("fewer than four distinct object proposals survived")

    output.mkdir(parents=True)
    mask_dir = output / "masks"
    mask_dir.mkdir()
    cv2.imwrite(str(output / "source_frame0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "motion_energy.png"), np.clip(motion, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output / "actor_mask.png"), actor_mask.astype(np.uint8) * 255)
    for rank, proposal in enumerate(selected):
        cv2.imwrite(
            str(mask_dir / f"proposal_{rank:02d}.png"),
            proposal["mask"].astype(np.uint8) * 255,
        )
    overlay = overlay_proposals(image, actor_mask, selected)
    cv2.imwrite(str(output / "ranked_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    receipt = {
        "schema_version": "sam2-source-object-proposals-v1",
        "complete": True,
        "source": {"path": str(source), "sha256": sha256(source)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "sam2_root": str(sam2_root),
        "bucket_hw": [args.height, args.width],
        "sampled_motion_frames": list(sample_indices),
        "sam_mask_count": len(annotations),
        "actor": {
            "sam_index": actor_index,
            "bbox": list(actor_bbox),
            "area": int(actor["area"]),
            "score": actor_score(
                actor, motion, image_area=image_area, height=args.height
            ),
        },
        "proposals": [
            {key: value for key, value in row.items() if key != "mask"}
            | {"mask_path": f"masks/proposal_{rank:02d}.png"}
            for rank, row in enumerate(selected)
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

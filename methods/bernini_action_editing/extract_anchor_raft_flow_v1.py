#!/usr/bin/env python3
"""Extract raw and camera-compensated RAFT flow for the Stage-0 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FRAME_COUNT = 81
PHASE_INDICES = tuple(range(0, FRAME_COUNT, 4))
MAX_PIXELS = 512 * 480


class AnchorFlowExtractionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video(path: Path) -> tuple[list[Any], float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AnchorFlowExtractionError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != FRAME_COUNT or not math.isfinite(fps) or fps <= 0:
        raise AnchorFlowExtractionError(
            f"expected 81-frame positive-fps video, got {len(frames)} @ {fps}"
        )
    return frames, fps


def _source_bucket(source: Path) -> tuple[int, int]:
    frames, _ = _read_video(source)
    height, width = frames[0].shape[:2]
    scale = math.sqrt(float(MAX_PIXELS) / float(height * width))
    bucket_height = max(16, int(height * scale) // 16 * 16)
    bucket_width = max(16, int(width * scale) // 16 * 16)
    return bucket_height, bucket_width


def _resize_phases(frames: list[Any], bucket: tuple[int, int]) -> list[Any]:
    import cv2

    height, width = bucket
    return [
        cv2.resize(frames[index], (width, height), interpolation=cv2.INTER_AREA)
        for index in PHASE_INDICES
    ]


def _raft_pair(model: Any, transform: Any, left: Any, right: Any, device: Any) -> Any:
    import torch

    left_tensor = torch.from_numpy(left.copy()).permute(2, 0, 1).unsqueeze(0)
    right_tensor = torch.from_numpy(right.copy()).permute(2, 0, 1).unsqueeze(0)
    left_tensor, right_tensor = transform(left_tensor, right_tensor)
    with torch.no_grad():
        result = model(
            left_tensor.to(device), right_tensor.to(device), num_flow_updates=12
        )[-1]
    return result[0].detach().to(device="cpu", dtype=torch.float32)


def _to_latent_flow(flow: Any, latent_hw: tuple[int, int]) -> Any:
    import torch
    import torch.nn.functional as F

    input_height, input_width = map(int, flow.shape[-2:])
    latent_height, latent_width = latent_hw
    resized = F.interpolate(
        flow.unsqueeze(0),
        size=latent_hw,
        mode="bilinear",
        align_corners=True,
    )[0]
    resized[0] *= float(latent_width) / float(input_width)
    resized[1] *= float(latent_height) / float(input_height)
    return resized.contiguous()


def _sample_flow(flow: Any, coordinates: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    height, width = map(int, flow.shape[-2:])
    x = coordinates[0]
    y = coordinates[1]
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack(
        (
            2.0 * x / max(width - 1, 1) - 1.0,
            2.0 * y / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0)
    sampled = F.grid_sample(
        flow.unsqueeze(0), grid, align_corners=True, padding_mode="zeros"
    )[0]
    return sampled, inside


def _consistency(forward: Any, backward: Any) -> Any:
    import torch

    height, width = map(int, backward.shape[-2:])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    previous = torch.stack((xx + backward[0], yy + backward[1]))
    sampled_forward, inside = _sample_flow(forward, previous)
    error = torch.linalg.vector_norm(backward + sampled_forward, dim=0)
    scale = torch.linalg.vector_norm(backward, dim=0) + torch.linalg.vector_norm(
        sampled_forward, dim=0
    )
    return (inside & (error <= 0.5 + 0.05 * scale)).float().unsqueeze(0)


def _camera_residual(flow: Any) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np
    import torch

    height, width = map(int, flow.shape[-2:])
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    source = np.stack((xx, yy), axis=-1).reshape(-1, 2)
    flow_np = flow.permute(1, 2, 0).numpy()
    destination = (np.stack((xx, yy), axis=-1) + flow_np).reshape(-1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        destination,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None:
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        inlier_fraction = 0.0
    else:
        matrix = matrix.astype(np.float32)
        inlier_fraction = float(inliers.mean()) if inliers is not None else 0.0
    homogeneous = np.concatenate(
        (np.stack((xx, yy), axis=-1), np.ones((height, width, 1), np.float32)),
        axis=-1,
    )
    global_destination = homogeneous @ matrix.T
    global_flow = global_destination - np.stack((xx, yy), axis=-1)
    residual = flow_np - global_flow
    return torch.from_numpy(residual).permute(2, 0, 1).contiguous(), {
        "matrix": matrix.tolist(),
        "ransac_inlier_fraction": inlier_fraction,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-height", type=int)
    parser.add_argument("--latent-width", type=int)
    args = parser.parse_args()
    if bool(args.latent_height) != bool(args.latent_width):
        raise AnchorFlowExtractionError(
            "latent-height and latent-width must be supplied together"
        )
    if args.latent_height is not None and (
        args.latent_height <= 0 or args.latent_width <= 0
    ):
        raise AnchorFlowExtractionError("explicit latent dimensions must be positive")
    source = Path(args.source).expanduser().resolve(strict=True)
    anchor = Path(args.anchor).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.suffix != ".safetensors":
        raise AnchorFlowExtractionError("output must be an absolute safetensors path")
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise AnchorFlowExtractionError("refusing to overwrite flow output")
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from safetensors.torch import save_file
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    if not torch.cuda.is_available():
        raise AnchorFlowExtractionError("RAFT extraction requires a visible GPU")
    device = torch.device("cuda", 0)
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device).eval()
    transform = weights.transforms()
    anchor_frames, anchor_fps = _read_video(anchor)
    bucket = (
        (int(args.latent_height) * 8, int(args.latent_width) * 8)
        if args.latent_height is not None
        else _source_bucket(source)
    )
    phases = _resize_phases(anchor_frames, bucket)
    latent_hw = (bucket[0] // 8, bucket[1] // 8)

    forwards = []
    backwards = []
    for left, right in zip(phases[:-1], phases[1:]):
        forwards.append(_to_latent_flow(_raft_pair(model, transform, left, right, device), latent_hw))
        backwards.append(_to_latent_flow(_raft_pair(model, transform, right, left, device), latent_hw))
    forward = torch.stack(forwards).float().contiguous()
    backward = torch.stack(backwards).float().contiguous()
    validity = torch.stack(
        [_consistency(f, b) for f, b in zip(forward, backward)]
    ).float().contiguous()
    camera_rows = []
    camera_backward = []
    for item in backward:
        residual, camera = _camera_residual(item)
        camera_backward.append(residual)
        camera_rows.append(camera)
    camera_backward_tensor = torch.stack(camera_backward).float().contiguous()
    save_file(
        {
            "backward_raw": backward,
            "backward_camera_residual": camera_backward_tensor,
            "validity": validity,
        },
        str(output),
    )
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    metadata = {
        "schema_version": "bernini-anchor-raft-flow-bundle-v1",
        "source": str(source),
        "source_sha256": _sha256(source),
        "anchor": str(anchor),
        "anchor_sha256": _sha256(anchor),
        "anchor_fps": anchor_fps,
        "sampled_frame_indices": list(PHASE_INDICES),
        "image_bucket_hw": list(bucket),
        "latent_hw": list(latent_hw),
        "latent_geometry_authority": (
            "explicit_target_clean_latent"
            if args.latent_height is not None
            else "source_aspect_bucket"
        ),
        "raft_weights": str(weights),
        "raft_weights_url": weights.url,
        "raft_weights_sha256": _sha256(weight_path),
        "num_flow_updates": 12,
        "valid_fraction_mean": float(validity.mean().item()),
        "raw_backward_rms": float(backward.square().mean().sqrt().item()),
        "camera_residual_backward_rms": float(
            camera_backward_tensor.square().mean().sqrt().item()
        ),
        "camera_fits": camera_rows,
    }
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Materialize source-appearance targets transported by anchor dense flow.

This is an auditable synthetic-target construction for action editing.  The
self-generated action video contributes only its backward optical flow.  RGB
and VAE latents come exclusively from the original source video.  For every
latent phase, cumulative backward flow samples the deterministic source phase
zero latent, yielding one full spatiotemporal tensor rather than a pooled
representation or a small-update constraint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dense_flow_token_adapter_v1 as flow_core
import infer_lora as inference
import infer_source_value_residual_oracle as value_audit
import materialize_same_video_motion_pairs_v1 as same_video
import train_lora as trainer
import train_self_generated_action_fullfield_v4 as v4
import train_self_generated_action_quotient_v1 as data
from tools import materialize_vae


SCHEMA_VERSION = "bernini-source-flow-fused-targets-v1"
ROW_SCHEMA_VERSION = "bernini-source-flow-fused-target-v1"


class SourceFlowFusedTargetError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def warp_source_phase0(
    source_clean: Any, backward_flow: Any
) -> tuple[Any, Any]:
    """Return source phase zero transported along cumulative backward flow.

    ``source_clean`` is normalized Bernini clean latent ``[1,C,21,H,W]`` and
    ``backward_flow`` is ``[20,2,H,W]`` in latent-pixel coordinates.  Phase
    zero is byte-identical to the source phase zero; later phases use border
    padding so the construction never imports anchor appearance.
    """

    import torch
    import torch.nn.functional as F

    if (
        not isinstance(source_clean, torch.Tensor)
        or source_clean.ndim != 5
        or tuple(map(int, source_clean.shape[:3]))
        != (1, 16, flow_core.LATENT_PHASES)
    ):
        raise SourceFlowFusedTargetError("source clean latent geometry differs")
    if (
        not isinstance(backward_flow, torch.Tensor)
        or tuple(map(int, backward_flow.shape[:2]))
        != (flow_core.LATENT_PHASES - 1, 2)
        or tuple(map(int, backward_flow.shape[-2:]))
        != tuple(map(int, source_clean.shape[-2:]))
    ):
        raise SourceFlowFusedTargetError("backward flow/source geometry differs")
    if not bool(torch.isfinite(source_clean).all().item()) or not bool(
        torch.isfinite(backward_flow).all().item()
    ):
        raise SourceFlowFusedTargetError("source/flow contains non-finite values")

    cumulative = flow_core._cumulative_backward(backward_flow.float()).contiguous()
    height, width = map(int, source_clean.shape[-2:])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=source_clean.device),
        torch.arange(width, dtype=torch.float32, device=source_clean.device),
        indexing="ij",
    )
    phase0 = source_clean[:, :, 0].float()
    phases = [phase0]
    for phase in range(1, flow_core.LATENT_PHASES):
        flow = cumulative[phase].to(device=source_clean.device, dtype=torch.float32)
        grid = torch.stack(
            (
                2.0 * (xx + flow[0]) / max(width - 1, 1) - 1.0,
                2.0 * (yy + flow[1]) / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        ).unsqueeze(0)
        phases.append(
            F.grid_sample(
                phase0,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    target = torch.stack(phases, dim=2).contiguous()
    if not torch.equal(target[:, :, 0], source_clean[:, :, 0].float()):
        raise SourceFlowFusedTargetError("fused target phase zero changed source identity")
    if tuple(target.shape) != tuple(source_clean.shape):
        raise SourceFlowFusedTargetError("fused target shape differs")
    return target, cumulative.cpu().contiguous()


def warp_source_rgb_phase0(
    source_pixels: Any, cumulative_latent_flow: Any
) -> Any:
    """Warp resized source RGB phase zero at 81-frame resolution.

    Spatial displacement is converted from latent pixels to RGB pixels and the
    21 latent phases are linearly interpolated to the native 81 video frames.
    VAE encoding happens only after this transport, avoiding the severe blur
    caused by treating VAE channels as translation-equivariant RGB features.
    """

    import torch
    import torch.nn.functional as F

    if (
        not isinstance(source_pixels, torch.Tensor)
        or source_pixels.ndim != 5
        or tuple(map(int, source_pixels.shape[:3])) != (1, 3, 81)
    ):
        raise SourceFlowFusedTargetError("source RGB tensor geometry differs")
    if (
        not isinstance(cumulative_latent_flow, torch.Tensor)
        or tuple(map(int, cumulative_latent_flow.shape[:2]))
        != (flow_core.LATENT_PHASES, 2)
    ):
        raise SourceFlowFusedTargetError("cumulative latent flow geometry differs")
    height, width = map(int, source_pixels.shape[-2:])
    latent_height, latent_width = map(int, cumulative_latent_flow.shape[-2:])
    if (height, width) != (latent_height * 8, latent_width * 8):
        raise SourceFlowFusedTargetError("RGB/latent flow scale differs")
    flow = F.interpolate(
        cumulative_latent_flow.float().permute(1, 0, 2, 3).unsqueeze(0),
        size=(81, height, width),
        mode="trilinear",
        align_corners=True,
    )[0].permute(1, 0, 2, 3).contiguous()
    flow[:, 0].mul_(width / float(latent_width))
    flow[:, 1].mul_(height / float(latent_height))
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=source_pixels.device),
        torch.arange(width, dtype=torch.float32, device=source_pixels.device),
        indexing="ij",
    )
    grid = torch.stack(
        (
            2.0 * (xx.unsqueeze(0) + flow[:, 0]) / max(width - 1, 1) - 1.0,
            2.0 * (yy.unsqueeze(0) + flow[:, 1]) / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    phase0 = source_pixels[:, :, 0].float()
    warped = F.grid_sample(
        phase0.expand(81, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    warped[0].copy_(phase0[0])
    return warped.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def _save_video(decoded: Any, path: Path, *, save_output: Any) -> Mapping[str, Any]:
    if path.exists() or path.is_symlink():
        raise SourceFlowFusedTargetError(f"refusing to overwrite video: {path}")
    value_audit.save_video_atomically(
        decoded, path, fps=int(inference.FPS), save_output_fn=save_output
    )
    frames, fps, hw = materialize_vae._decode_exact_video(path)
    inference.validate_exact_video_metadata(int(frames.shape[0]), fps)
    return {"path": str(path), "sha256": file_sha256(path), "hw": list(hw)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument(
        "--warp-coordinate",
        choices=("normalized_clean_latent", "source_rgb_then_vae"),
        default="normalized_clean_latent",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SourceFlowFusedTargetError("output must be fresh")

    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        trainer.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
        )
    )
    checkpoint, _ = trainer.validate_checkpoint(args.checkpoint)
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file, save_file
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode, _vae_encode

    if not torch.cuda.is_available():
        raise SourceFlowFusedTargetError("fused target materialization requires one GPU")
    pair_manifest_path = Path(args.pair_manifest).expanduser().resolve(strict=True)
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    pair_rows = None
    stored = pair_manifest.pop("manifest_digest", None)
    if (
        pair_manifest.get("schema_version") != same_video.SCHEMA_VERSION
        or same_video.object_sha256(pair_manifest) != stored
    ):
        raise SourceFlowFusedTargetError("same-video manifest digest differs")
    pair_rows = pair_manifest.get("rows")
    if not isinstance(pair_rows, list) or len(pair_rows) != 4:
        raise SourceFlowFusedTargetError("fused release requires four rows")
    source_manifest_path = Path(pair_manifest["training_manifest"]).resolve(strict=True)
    source_manifest, source_rows = v4.load_source_manifest(
        source_manifest_path, pair_manifest["training_manifest_sha256"]
    )
    del source_manifest
    source_by_iid = {row["iid"]: row for row in source_rows}
    if set(source_by_iid) != {row["iid"] for row in pair_rows}:
        raise SourceFlowFusedTargetError("source/pair IID closure differs")

    output.mkdir(parents=True, mode=0o700)
    device = torch.device("cuda", 0)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    mean, std, _ = trainer._vae_statistics(checkpoint)
    rows = []
    for pair_row in pair_rows:
        iid = pair_row["iid"]
        source_row = source_by_iid[iid]
        source_path = Path(source_row["source_posterior"]["path"]).resolve(strict=True)
        if file_sha256(source_path) != source_row["source_posterior"]["sha256"]:
            raise SourceFlowFusedTargetError(f"source posterior SHA differs: {iid}")
        source_clean = data.source_clean_from_posterior(
            source_path.read_bytes(), mean, std
        ).float().contiguous()
        flow_path = Path(pair_row["flow_bundle"]).resolve(strict=True)
        flow_tensors = load_file(str(flow_path), device="cpu")
        if set(flow_tensors) != {
            "backward_raw", "backward_camera_residual", "validity"
        }:
            raise SourceFlowFusedTargetError(f"flow tensor closure differs: {iid}")
        cumulative = flow_core._cumulative_backward(
            flow_tensors["backward_raw"].float()
        ).cpu().contiguous()
        if args.warp_coordinate == "normalized_clean_latent":
            target, observed_cumulative = warp_source_phase0(
                source_clean, flow_tensors["backward_raw"]
            )
            if not torch.equal(cumulative, observed_cumulative):
                raise SourceFlowFusedTargetError("cumulative flow construction differs")
            construction = (
                "source_phase0_warped_by_anchor_cumulative_backward_raw_flow"
            )
        else:
            source_video = Path(pair_row["source_video"]).resolve(strict=True)
            frames, fps, source_hw = materialize_vae._decode_exact_video(source_video)
            inference.validate_exact_video_metadata(int(frames.shape[0]), fps)
            bucket = materialize_vae.source_aspect_bucket(
                *source_hw,
                max_pixels=inference.MAX_PIXELS,
                stride=inference.SPATIAL_STRIDE,
            )
            source_pixels = materialize_vae._resize_video(
                frames, bucket, None
            ).unsqueeze(0)
            if tuple(map(int, source_pixels.shape[-2:])) != tuple(
                int(value) * 8 for value in source_clean.shape[-2:]
            ):
                raise SourceFlowFusedTargetError("source RGB/source latent bucket differs")
            warped_pixels = warp_source_rgb_phase0(
                source_pixels.to(device), cumulative.to(device)
            )
            with torch.no_grad():
                target = _vae_encode(vae, warped_pixels).float().cpu().contiguous()
            target[:, :, 0].copy_(source_clean[:, :, 0])
            construction = (
                "source_rgb_phase0_warped_by_anchor_cumulative_backward_raw_flow_then_vae"
            )
        row_dir = output / iid
        row_dir.mkdir(mode=0o700)
        latent_path = row_dir / "source_flow_fused.safetensors"
        save_file(
            {"source": source_clean, "target": target},
            str(latent_path),
            metadata={
                "schema_version": ROW_SCHEMA_VERSION,
                "iid": iid,
                "coordinate": "bernini_normalized_clean_vae_latent",
                "target_construction": construction,
            },
        )
        with torch.no_grad():
            decoded = _vae_decode(vae, target.to(device))
        video = _save_video(decoded, row_dir / "target.mp4", save_output=save_output)
        rows.append(
            {
                "schema_version": ROW_SCHEMA_VERSION,
                "iid": iid,
                "instruction": pair_row["instruction"],
                "source_posterior": dict(source_row["source_posterior"]),
                "flow_bundle": str(flow_path),
                "flow_bundle_sha256": file_sha256(flow_path),
                "latents": {
                    "path": str(latent_path),
                    "sha256": file_sha256(latent_path),
                    "shape": list(map(int, target.shape)),
                    "source_tensor_sha256": same_video.tensor_sha256(source_clean),
                    "target_tensor_sha256": same_video.tensor_sha256(target),
                    "phase0_exact_source": True,
                },
                "video": video,
                "cumulative_flow_sha256": same_video.tensor_sha256(cumulative),
                "anchor_rgb_or_vae_latent_used": False,
                "source_rgb_or_vae_latent_only": True,
                "self_generated_anchor_contribution": "backward_raw_flow_only",
                "warp_coordinate": args.warp_coordinate,
            }
        )
        del source_clean, target, cumulative, decoded
        if args.warp_coordinate == "source_rgb_then_vae":
            del source_pixels, warped_pixels, frames
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "pair_manifest": str(pair_manifest_path),
        "pair_manifest_sha256": file_sha256(pair_manifest_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
        "target_construction": (
            "source_phase0_warped_by_anchor_cumulative_backward_raw_flow"
            if args.warp_coordinate == "normalized_clean_latent"
            else "source_rgb_phase0_warped_by_anchor_cumulative_backward_raw_flow_then_vae"
        ),
        "warp_coordinate": args.warp_coordinate,
        "qwen_used": False,
        "anchor_rgb_or_vae_latent_used": False,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

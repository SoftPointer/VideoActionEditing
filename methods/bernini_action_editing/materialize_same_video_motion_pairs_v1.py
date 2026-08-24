#!/usr/bin/env python3
"""Materialize coherent same-video source variants for motion-adapter training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference
import infer_source_value_residual_oracle as value_audit
import train_lora as trainer
from tools import materialize_vae


SCHEMA_VERSION = "bernini-same-video-motion-pairs-v1"
PAIR_SCHEMA_VERSION = "bernini-same-video-motion-pair-v1"
INCOMPLETE_CUTOFF_FRAME = 40
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SameVideoPairError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    import torch

    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(map(int, tensor.shape))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_manifest(path: Path, expected_sha: str | None) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if expected_sha is not None and file_sha256(resolved) != expected_sha:
        raise SameVideoPairError(f"manifest SHA-256 differs: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get("rows"), list):
        raise SameVideoPairError(f"manifest row closure differs: {resolved}")
    return value


def _source_bucket_and_anchor_pixels(source: Path, anchor: Path) -> tuple[Any, tuple[int, int]]:
    source_frames, source_fps, source_hw = materialize_vae._decode_exact_video(source)
    anchor_frames, anchor_fps, anchor_hw = materialize_vae._decode_exact_video(anchor)
    inference.validate_exact_video_metadata(int(source_frames.shape[0]), source_fps)
    inference.validate_exact_video_metadata(int(anchor_frames.shape[0]), anchor_fps)
    bucket = materialize_vae.source_aspect_bucket(
        *source_hw,
        max_pixels=inference.MAX_PIXELS,
        stride=inference.SPATIAL_STRIDE,
    )
    crop, retention = materialize_vae.target_crop_to_source_aspect(
        anchor_hw[0], anchor_hw[1], source_hw[0], source_hw[1]
    )
    if retention < 0.90:
        raise SameVideoPairError("anchor/source aspect crop retention is below 90%")
    pixels = materialize_vae._resize_video(anchor_frames, bucket, crop).unsqueeze(0)
    if tuple(map(int, pixels.shape)) != (1, 3, 81, bucket[0], bucket[1]):
        raise SameVideoPairError("anchor pixel tensor geometry differs")
    return pixels, bucket


def _save_video(decoded: Any, path: Path, *, save_output: Any) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise SameVideoPairError(f"refusing to overwrite video: {path}")
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
    parser.add_argument("--flow-manifest", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if _SHA256.fullmatch(args.training_manifest_sha256) is None:
        raise SameVideoPairError("training manifest SHA-256 is invalid")
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise SameVideoPairError("output must be fresh")

    bernini_root, veomni_root, bernini_revision, veomni_revision = trainer.validate_source_trees(
        args.bernini_root,
        args.veomni_root,
        expected_bernini_commit=trainer.BERNINI_OFFICIAL_COMMIT,
        expected_veomni_commit=trainer.VEOMNI_TESTED_COMMIT,
    )
    checkpoint, _ = trainer.validate_checkpoint(args.checkpoint)
    trainer.activate_source_trees(bernini_root, veomni_root)

    import torch
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file, save_file
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode, _vae_encode

    if not torch.cuda.is_available():
        raise SameVideoPairError("same-video pair materialization requires one GPU")
    flow_manifest_path = Path(args.flow_manifest).expanduser().resolve(strict=True)
    flow_root = Path(args.flow_root).expanduser().resolve(strict=True)
    training_manifest_path = Path(args.training_manifest).expanduser().resolve(strict=True)
    flow_manifest = _load_manifest(flow_manifest_path, None)
    training_manifest = _load_manifest(
        training_manifest_path, args.training_manifest_sha256
    )
    train_by_iid = {row["iid"]: row for row in training_manifest["rows"]}
    if len(train_by_iid) != 4 or len(flow_manifest["rows"]) != 4:
        raise SameVideoPairError("same-video release requires exactly four rows")

    output.mkdir(parents=True, mode=0o700)
    device = torch.device("cuda", 0)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)
    media_root = Path(flow_manifest["remote_media_root"])
    rows = []
    for flow_row in flow_manifest["rows"]:
        iid = flow_row["iid"]
        if iid not in train_by_iid:
            raise SameVideoPairError(f"training target missing for {iid}")
        train_row = train_by_iid[iid]
        flow_bundle = (flow_root / f"{iid}.safetensors").resolve(strict=True)
        source = (media_root / iid / "source.mp4").resolve(strict=True)
        anchor = (media_root / iid / "anchor.mp4").resolve(strict=True)
        target_path = Path(train_row["action_anchor"]["latent_path"]).resolve(strict=True)
        if file_sha256(target_path) != train_row["action_anchor"]["latent_sha256"]:
            raise SameVideoPairError(f"target latent SHA differs for {iid}")
        target_file = load_file(str(target_path), device="cpu")
        if set(target_file) != {"normalized_clean_latent"}:
            raise SameVideoPairError(f"target latent keys differ for {iid}")
        target = target_file["normalized_clean_latent"].float().contiguous()
        pixels, bucket = _source_bucket_and_anchor_pixels(source, anchor)
        noop_pixels = pixels[:, :, 0:1].expand(-1, -1, 81, -1, -1).contiguous()
        incomplete_pixels = torch.cat(
            (
                pixels[:, :, : INCOMPLETE_CUTOFF_FRAME + 1],
                pixels[:, :, INCOMPLETE_CUTOFF_FRAME : INCOMPLETE_CUTOFF_FRAME + 1].expand(
                    -1, -1, 80 - INCOMPLETE_CUTOFF_FRAME, -1, -1
                ),
            ),
            dim=2,
        ).contiguous()
        with torch.no_grad():
            noop = _vae_encode(vae, noop_pixels.to(device=device, dtype=torch.float32)).float().cpu().contiguous()
            incomplete = _vae_encode(
                vae, incomplete_pixels.to(device=device, dtype=torch.float32)
            ).float().cpu().contiguous()
        expected_shape = (1, 16, 21, bucket[0] // 8, bucket[1] // 8)
        if tuple(map(int, target.shape)) != expected_shape or tuple(map(int, noop.shape)) != expected_shape or tuple(map(int, incomplete.shape)) != expected_shape:
            raise SameVideoPairError(
                f"same-video latent geometry differs for {iid}: "
                f"target={tuple(target.shape)} noop={tuple(noop.shape)} incomplete={tuple(incomplete.shape)} expected={expected_shape}"
            )
        row_dir = output / iid
        row_dir.mkdir(mode=0o700)
        latent_path = row_dir / "same_video_pair.safetensors"
        save_file(
            {"target": target, "source_noop": noop, "source_incomplete": incomplete},
            str(latent_path),
            metadata={
                "schema_version": PAIR_SCHEMA_VERSION,
                "iid": iid,
                "coordinate": "bernini_normalized_clean_vae_latent",
            },
        )
        vae.to(device)
        videos = {}
        with torch.no_grad():
            for role, latent in (
                ("target", target),
                ("source_noop", noop),
                ("source_incomplete", incomplete),
            ):
                decoded = _vae_decode(vae, latent.to(device))
                videos[role] = _save_video(
                    decoded, row_dir / f"{role}.mp4", save_output=save_output
                )
        rows.append(
            {
                "schema_version": PAIR_SCHEMA_VERSION,
                "iid": iid,
                "instruction": flow_row["instruction"],
                "source_video": str(source),
                "source_video_sha256": file_sha256(source),
                "anchor_video": str(anchor),
                "anchor_video_sha256": file_sha256(anchor),
                "flow_bundle": str(flow_bundle),
                "flow_bundle_sha256": file_sha256(flow_bundle),
                "latents": {
                    "path": str(latent_path),
                    "sha256": file_sha256(latent_path),
                    "shape": list(expected_shape),
                    "target_sha256": tensor_sha256(target),
                    "source_noop_sha256": tensor_sha256(noop),
                    "source_incomplete_sha256": tensor_sha256(incomplete),
                },
                "videos": videos,
                "incomplete_cutoff_frame": INCOMPLETE_CUTOFF_FRAME,
                "same_actor_world_target": True,
                "target_is_self_generated_action_video": True,
                "source_variant_is_deterministically_derived_from_target_rgb": True,
            }
        )
        del pixels, noop_pixels, incomplete_pixels, noop, incomplete, target
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "flow_manifest": str(flow_manifest_path),
        "flow_manifest_sha256": file_sha256(flow_manifest_path),
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": args.training_manifest_sha256,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
        "incomplete_cutoff_frame": INCOMPLETE_CUTOFF_FRAME,
        "qwen_used": False,
        "cross_actor_target_used": False,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

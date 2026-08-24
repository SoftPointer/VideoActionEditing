#!/usr/bin/env python3
"""Materialize an expanded bank of coherent self-generated action pairs."""

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


INPUT_SCHEMA = "bernini-expanded-self-generated-action-bank-input-v1"
PAIR_SCHEMA = "bernini-same-video-motion-pair-v1"
OUTPUT_SCHEMA = "bernini-same-video-motion-pairs-v1"
INCOMPLETE_CUTOFF_FRAME = 40
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExpandedActionBankError(RuntimeError):
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


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExpandedActionBankError(f"JSON root must be an object: {path}")
    return value


def load_input_manifest(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    value = load_json(path)
    rows = value.get("rows")
    if value.get("schema_version") != INPUT_SCHEMA or not isinstance(rows, list):
        raise ExpandedActionBankError("expanded-bank input manifest closure differs")
    if not 1 <= len(rows) <= 28:
        raise ExpandedActionBankError("expanded-bank input must contain 1 to 28 rows")
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExpandedActionBankError("expanded-bank row must be an object")
        iid = row.get("iid")
        if not isinstance(iid, str) or not iid or iid in seen:
            raise ExpandedActionBankError("expanded-bank IID closure differs")
        seen.add(iid)
        if not isinstance(row.get("instruction"), str) or not row["instruction"].strip():
            raise ExpandedActionBankError(f"instruction is empty: {iid}")
        if not isinstance(row.get("target_filename"), str) or not row["target_filename"]:
            raise ExpandedActionBankError(f"target filename is empty: {iid}")
        if _SHA256.fullmatch(str(row.get("target_sha256", ""))) is None:
            raise ExpandedActionBankError(f"target SHA-256 is invalid: {iid}")
        if row.get("human_temporal_audit") != "action_complete_5_frame_plus_full_video":
            raise ExpandedActionBankError(f"target lacks human temporal audit: {iid}")
    return value, rows


def load_base_rows(path: Path | None) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {}
    value = dict(load_json(path))
    stored = value.pop("manifest_digest", None)
    rows = value.get("rows")
    if (
        value.get("schema_version") != OUTPUT_SCHEMA
        or not isinstance(rows, list)
        or object_sha256(value) != stored
    ):
        raise ExpandedActionBankError("base pair manifest semantic digest differs")
    return list(rows), {
        "base_pair_manifest": str(path),
        "base_pair_manifest_sha256": file_sha256(path),
    }


def save_video(decoded: Any, path: Path, *, save_output: Any) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise ExpandedActionBankError(f"refusing to overwrite video: {path}")
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
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--flow-root", required=True)
    parser.add_argument("--base-pair-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise ExpandedActionBankError("output must be fresh")
    input_path = Path(args.input_manifest).expanduser().resolve(strict=True)
    video_root = Path(args.video_root).expanduser().resolve(strict=True)
    flow_root = Path(args.flow_root).expanduser().resolve(strict=True)
    base_path = (
        Path(args.base_pair_manifest).expanduser().resolve(strict=True)
        if args.base_pair_manifest else None
    )
    input_manifest, input_rows = load_input_manifest(input_path)
    base_rows, base_metadata = load_base_rows(base_path)
    base_iids = {row.get("iid") for row in base_rows}
    new_iids = {row["iid"] for row in input_rows}
    if None in base_iids or base_iids & new_iids:
        raise ExpandedActionBankError("base/new pair IID closure differs")
    if not 4 <= len(base_rows) + len(input_rows) <= 32:
        raise ExpandedActionBankError("combined action bank must contain 4 to 32 rows")

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
    from safetensors.torch import save_file
    from bernini.io_utils import save_output
    from bernini.pipeline import _vae_decode, _vae_encode

    if not torch.cuda.is_available():
        raise ExpandedActionBankError("expanded action bank materialization requires one GPU")
    output.mkdir(parents=True, mode=0o700)
    device = torch.device("cuda", 0)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint), subfolder="vae", torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)

    new_rows = []
    for row in input_rows:
        iid = row["iid"]
        target_video = (video_root / row["target_filename"]).resolve(strict=True)
        if target_video.parent != video_root or file_sha256(target_video) != row["target_sha256"]:
            raise ExpandedActionBankError(f"target video provenance differs: {iid}")
        flow_bundle = (flow_root / f"{iid}.safetensors").resolve(strict=True)
        if not flow_bundle.is_file():
            raise ExpandedActionBankError(f"flow bundle is unavailable: {iid}")

        frames, fps, target_hw = materialize_vae._decode_exact_video(target_video)
        inference.validate_exact_video_metadata(int(frames.shape[0]), fps)
        bucket = materialize_vae.source_aspect_bucket(
            *target_hw, max_pixels=inference.MAX_PIXELS,
            stride=inference.SPATIAL_STRIDE,
        )
        pixels = materialize_vae._resize_video(frames, bucket, None).unsqueeze(0)
        if tuple(map(int, pixels.shape)) != (1, 3, 81, bucket[0], bucket[1]):
            raise ExpandedActionBankError(f"target pixel geometry differs: {iid}")
        noop_pixels = pixels[:, :, 0:1].expand(-1, -1, 81, -1, -1).contiguous()
        incomplete_pixels = torch.cat(
            (
                pixels[:, :, : INCOMPLETE_CUTOFF_FRAME + 1],
                pixels[:, :, INCOMPLETE_CUTOFF_FRAME : INCOMPLETE_CUTOFF_FRAME + 1].expand(
                    -1, -1, 80 - INCOMPLETE_CUTOFF_FRAME, -1, -1
                ),
            ), dim=2,
        ).contiguous()
        with torch.no_grad():
            target = _vae_encode(vae, pixels.to(device=device, dtype=torch.float32)).float().cpu().contiguous()
            noop = _vae_encode(vae, noop_pixels.to(device=device, dtype=torch.float32)).float().cpu().contiguous()
            incomplete = _vae_encode(vae, incomplete_pixels.to(device=device, dtype=torch.float32)).float().cpu().contiguous()
        expected_shape = (1, 16, 21, bucket[0] // 8, bucket[1] // 8)
        if any(tuple(map(int, item.shape)) != expected_shape for item in (target, noop, incomplete)):
            raise ExpandedActionBankError(f"same-video latent geometry differs: {iid}")

        row_dir = output / iid
        row_dir.mkdir(mode=0o700)
        latent_path = row_dir / "same_video_pair.safetensors"
        save_file(
            {"target": target, "source_noop": noop, "source_incomplete": incomplete},
            str(latent_path),
            metadata={"schema_version": PAIR_SCHEMA, "iid": iid,
                      "coordinate": "bernini_normalized_clean_vae_latent"},
        )
        videos = {}
        with torch.no_grad():
            for role, latent in (
                ("target", target), ("source_noop", noop),
                ("source_incomplete", incomplete),
            ):
                decoded = _vae_decode(vae, latent.to(device))
                videos[role] = save_video(
                    decoded, row_dir / f"{role}.mp4", save_output=save_output
                )
        new_rows.append({
            "schema_version": PAIR_SCHEMA,
            "iid": iid,
            "instruction": row["instruction"],
            "source_video": str(target_video),
            "source_video_sha256": file_sha256(target_video),
            "anchor_video": str(target_video),
            "anchor_video_sha256": file_sha256(target_video),
            "flow_bundle": str(flow_bundle),
            "flow_bundle_sha256": file_sha256(flow_bundle),
            "latents": {
                "path": str(latent_path), "sha256": file_sha256(latent_path),
                "shape": list(expected_shape), "target_sha256": tensor_sha256(target),
                "source_noop_sha256": tensor_sha256(noop),
                "source_incomplete_sha256": tensor_sha256(incomplete),
            },
            "videos": videos,
            "incomplete_cutoff_frame": INCOMPLETE_CUTOFF_FRAME,
            "same_actor_world_target": True,
            "target_is_self_generated_action_video": True,
            "source_variant_is_deterministically_derived_from_target_rgb": True,
            "target_human_temporal_audit": row["human_temporal_audit"],
        })
        del pixels, noop_pixels, incomplete_pixels, target, noop, incomplete
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "rows": base_rows + new_rows,
        "input_manifest": str(input_path),
        "input_manifest_sha256": file_sha256(input_path),
        **base_metadata,
        "bernini_commit": bernini_revision,
        "veomni_commit": veomni_revision,
        "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
        "incomplete_cutoff_frame": INCOMPLETE_CUTOFF_FRAME,
        "qwen_used": False,
        "cross_actor_target_used": False,
        "expanded_bank_is_action_only": True,
    }
    manifest["manifest_digest"] = object_sha256(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "rows": len(manifest["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

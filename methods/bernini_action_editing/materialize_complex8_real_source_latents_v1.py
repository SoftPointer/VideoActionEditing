#!/usr/bin/env python3
"""Encode the eight Complex8 real sources into source-owned Wan latents.

This is data preparation, not optimization.  Every source is decoded with the
same exact81/bucket policy used by the DynaEdit evaluator, encoded as the
frozen Wan VAE posterior mode, normalized exactly once with the checkpoint's
official latent statistics, and saved independently.  No self-generated
anchor is read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Optional, Sequence


METHOD_ROOT = Path(__file__).resolve().parent
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_source_aligned_controller_oracle as source_audit
import infer_lora as inference
import train_lora as legacy


SCHEMA = "bernini-complex8-real-source-latents-v2"
SOURCE_SCHEMA = "bernini-interaction-complex8-real-source-fields-v1"
SHA = re.compile(r"[0-9a-f]{64}")


class MaterializationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise MaterializationError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").float().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def normalize_posterior_mode(raw_mode: Any, mean: Any, std: Any) -> Any:
    """Normalize an unnormalized VAE posterior mode exactly once."""

    import torch

    if raw_mode.ndim != 5 or mean.ndim != 4 or std.ndim != 4:
        fail("VAE posterior/statistic geometry differs")
    if int(raw_mode.shape[1]) != int(mean.shape[0]) or mean.shape != std.shape:
        fail("VAE posterior channel statistics differ")
    if not bool(torch.isfinite(raw_mode).all()) or not bool((std > 0).all()):
        fail("VAE posterior/statistics are not finite and positive")
    return (
        (raw_mode.float() - mean.unsqueeze(0).float())
        / std.unsqueeze(0).float()
    ).contiguous()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--bernini-root", required=True)
    value.add_argument("--veomni-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--source-fields", required=True)
    value.add_argument("--output", required=True)
    return value


def load_source_fields(path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if (
        value.get("schema_version") != SOURCE_SCHEMA
        or value.get("source_count") != 8
        or not isinstance(rows, list)
        or len(rows) != 8
    ):
        fail("real-source field authority differs")
    if [row.get("ordinal") for row in rows] != list(range(8)):
        fail("real-source ordinals differ")
    if len({row.get("event_id") for row in rows}) != 8:
        fail("real-source event registry is not unique")
    for row in rows:
        if (
            not isinstance(row.get("source_iid"), str)
            or not isinstance(row.get("source_video"), str)
            or SHA.fullmatch(str(row.get("source_video_sha256"))) is None
            or not isinstance(row.get("source_caption"), str)
            or not str(row["source_caption"]).strip()
            or not isinstance(row.get("target_caption"), str)
            or not str(row["target_caption"]).strip()
        ):
            fail("one real-source row is incomplete")
    return value, rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    bernini_root, veomni_root, bernini_revision, veomni_revision = (
        legacy.validate_source_trees(
            args.bernini_root,
            args.veomni_root,
            expected_bernini_commit=legacy.BERNINI_OFFICIAL_COMMIT,
            expected_veomni_commit=legacy.VEOMNI_TESTED_COMMIT,
        )
    )
    checkpoint, _ = legacy.validate_checkpoint(args.checkpoint)
    legacy.activate_source_trees(bernini_root, veomni_root)
    # ``prepare_hashed_source_snapshot`` deliberately receives its exact
    # decoder implementation by injection in the inference runtime.  Bind the
    # pinned exact81 decoder here before reading any source bytes.
    source_audit.legacy = inference

    source_fields = Path(args.source_fields).expanduser().resolve(strict=True)
    source_authority, rows = load_source_fields(source_fields)
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        fail("output must be a fresh absolute path")

    import torch
    from diffusers.models import AutoencoderKLWan
    from safetensors.torch import save_file

    if not torch.cuda.is_available() or getattr(torch.version, "hip", None) is None:
        fail("materialization requires one AUH ROCm GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    mean, std, _ = legacy._vae_statistics(checkpoint)
    mean = mean.to(device=device, dtype=torch.float32)
    std = std.to(device=device, dtype=torch.float32)

    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    vae.eval().requires_grad_(False)
    vae.to(device)

    output.mkdir(parents=True, exist_ok=False)
    encoded_rows: list[dict[str, Any]] = []
    for row in rows:
        requested = Path(str(row["source_video"])).expanduser()
        if not requested.is_absolute() or requested.is_symlink():
            fail("source video must be an absolute non-symlink")
        source_path = requested.resolve(strict=True)
        if source_path != requested or file_sha256(source_path) != row["source_video_sha256"]:
            fail("source video bytes differ")
        pixels, metadata, source_sha = source_audit.prepare_hashed_source_snapshot(
            source_path
        )
        if source_sha != row["source_video_sha256"] or metadata.get("frame_count") != 81:
            fail("source snapshot closure differs")
        with torch.inference_mode():
            # Do not call ``bernini.pipeline._vae_encode`` here: that helper
            # already applies the official mean/std normalization.  Calling
            # it and normalizing again silently corrupts the real-source state.
            raw_mode = (
                vae.encode(pixels.to(device=device, dtype=torch.float32))
                .latent_dist.mode()
                .float()
                .contiguous()
            )
            clean = normalize_posterior_mode(raw_mode, mean, std)
        if (
            clean.ndim != 5
            or tuple(map(int, clean.shape[:3])) != (1, 16, 21)
            or not bool(torch.isfinite(clean).all())
        ):
            fail("normalized real-source latent geometry differs")
        latent_path = output / f"e{int(row['ordinal']):02d}_{row['event_id']}.safetensors"
        cpu_clean = clean.to(device="cpu").contiguous()
        save_file({"clean": cpu_clean}, str(latent_path))
        encoded_rows.append(
            {
                "ordinal": int(row["ordinal"]),
                "event_id": str(row["event_id"]),
                "source_iid": str(row["source_iid"]),
                "source_video": str(source_path),
                "source_video_sha256": source_sha,
                "source_caption": str(row["source_caption"]),
                "target_caption": str(row["target_caption"]),
                "snapshot_metadata": metadata,
                "latent": str(latent_path),
                "latent_file_sha256": file_sha256(latent_path),
                "latent_tensor_sha256": tensor_sha256(cpu_clean),
                "latent_shape": list(map(int, cpu_clean.shape)),
                "latent_dtype": str(cpu_clean.dtype),
                "latent_role": "normalized_complete_real_source_trajectory_identity_content_and_existing_motion_authority",
                "posterior_statistic": "latent_dist.mode",
                "normalization_count": 1,
                "normalization_application_count": 1,
            }
        )
        del pixels, raw_mode, clean, cpu_clean
        torch.cuda.empty_cache()

    vae.to("cpu")
    manifest = {
        "schema_version": SCHEMA,
        "source_fields": str(source_fields),
        "source_fields_sha256": file_sha256(source_fields),
        "anchor_authoring_sha256": source_authority["anchor_authoring_sha256"],
        "bernini_revision": bernini_revision,
        "veomni_revision": veomni_revision,
        "checkpoint": str(checkpoint),
        "row_count": len(encoded_rows),
        "posterior_statistic": "latent_dist.mode",
        "normalization": "checkpoint_latents_mean_std_exactly_once",
        "normalization_count": 1,
        "normalization_application_count": 1,
        "bernini_private_vae_encode_used": False,
        "self_generated_anchor_read_during_materialization": False,
        "optimization_steps": 0,
        "rows": encoded_rows,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"output": str(output), "manifest_sha256": file_sha256(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

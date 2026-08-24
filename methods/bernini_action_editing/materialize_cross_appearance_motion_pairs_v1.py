#!/usr/bin/env python3
"""Build 32 source-consistent targets with matched and cross-appearance motion.

For every complex-interaction T2V variant, the target is its clean action
latent and noop/incomplete sources are deterministically derived from that
same video's RGB.  Two manifests reuse those exact target/source tensors:

* matched: RAFT motion comes from the same T2V variant (capacity upper bound);
* cross: RAFT motion comes from the next variant of the same action family.

The cross manifest therefore exposes motion from another person/scene without
ever using that person's RGB or VAE latent as the optimization target.
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

import infer_lora as inference
import train_lora as trainer
from tools import materialize_vae


SCHEMA_VERSION = "bernini-same-video-motion-pairs-v1"
PAIR_SCHEMA_VERSION = "bernini-cross-appearance-motion-pair-v1"
INCOMPLETE_CUTOFF_FRAME = 40


class CrossAppearancePairError(RuntimeError):
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


def event_directory(root: Path, ordinal: int, event_id: str) -> Path:
    value = root / f"e{ordinal:02d}_{event_id}"
    if not value.is_dir():
        raise CrossAppearancePairError(f"missing event directory: {value}")
    return value


def load_authoring(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    events = value.get("events")
    if (
        value.get("schema_version")
        != "bernini-interaction-complex8-multianchor-authoring-v2"
        or not isinstance(events, list)
        or len(events) != 8
        or any(len(event.get("variants", ())) != 4 for event in events)
    ):
        raise CrossAppearancePairError("complex8 authoring closure differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--authoring", required=True)
    parser.add_argument("--anchor-root", required=True)
    parser.add_argument("--matched-flow-root", required=True)
    parser.add_argument("--cross-flow-root", required=True)
    parser.add_argument(
        "--source-authoring-mode",
        choices=("deterministic", "generated_counterfactual"),
        default="deterministic",
        help=(
            "deterministic repeats/freezes frames from the action target; "
            "generated_counterfactual encodes the separately generated noop "
            "and incomplete videos from the same variant directory"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise CrossAppearancePairError("output must be fresh")
    authoring_path = Path(args.authoring).expanduser().resolve(strict=True)
    anchor_root = Path(args.anchor_root).expanduser().resolve(strict=True)
    matched_flow_root = Path(args.matched_flow_root).expanduser().resolve(strict=True)
    cross_flow_root = Path(args.cross_flow_root).expanduser().resolve(strict=True)
    authoring = load_authoring(authoring_path)

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
    from bernini.pipeline import _vae_encode

    if not torch.cuda.is_available():
        raise CrossAppearancePairError("pair materialization requires one GPU")
    output.mkdir(parents=True, mode=0o700)
    device = torch.device("cuda", 0)
    vae = AutoencoderKLWan.from_pretrained(
        str(checkpoint),
        subfolder="vae",
        torch_dtype=torch.float32,
        local_files_only=True,
    ).eval().requires_grad_(False).to(device)

    row_bases: list[dict[str, Any]] = []
    for event in authoring["events"]:
        ordinal = int(event["ordinal"])
        event_id = str(event["event_id"])
        event_root = event_directory(anchor_root, ordinal, event_id)
        instruction = f'{event["action"]} {event["constraints"]}'
        for variant_index, variant in enumerate(event["variants"]):
            variant_id = str(variant["variant_id"])
            iid = f"e{ordinal:02d}-{variant_id}"
            video = (event_root / variant_id / "t2v.mp4").resolve(strict=True)
            target_path = (
                event_root / variant_id / "t2v.normalized-clean-latent.safetensors"
            ).resolve(strict=True)
            values = load_file(str(target_path), device="cpu")
            if set(values) != {"normalized_clean_latent"}:
                raise CrossAppearancePairError(f"target latent keys differ: {iid}")
            target = values["normalized_clean_latent"].float().contiguous()
            frames, fps, source_hw = materialize_vae._decode_exact_video(video)
            inference.validate_exact_video_metadata(int(frames.shape[0]), fps)
            # The saved clean target is the optimization authority.  Some
            # narrow T2V buckets (notably jet-ski/plant) are 656 px high and
            # must not be silently rounded up to the generic 672 px bucket.
            bucket = (int(target.shape[-2]) * 8, int(target.shape[-1]) * 8)
            pixels = materialize_vae._resize_video(frames, bucket, None).unsqueeze(0)
            if tuple(map(int, pixels.shape)) != (1, 3, 81, bucket[0], bucket[1]):
                raise CrossAppearancePairError(f"target RGB geometry differs: {iid}")
            source_video_records: dict[str, dict[str, Any]] = {}
            if args.source_authoring_mode == "deterministic":
                noop_pixels = (
                    pixels[:, :, 0:1].expand(-1, -1, 81, -1, -1).contiguous()
                )
                incomplete_pixels = torch.cat(
                    (
                        pixels[:, :, : INCOMPLETE_CUTOFF_FRAME + 1],
                        pixels[
                            :,
                            :,
                            INCOMPLETE_CUTOFF_FRAME : INCOMPLETE_CUTOFF_FRAME + 1,
                        ].expand(-1, -1, 80 - INCOMPLETE_CUTOFF_FRAME, -1, -1),
                    ),
                    dim=2,
                ).contiguous()
            else:
                source_pixels = {}
                for source_name in ("noop", "incomplete"):
                    source_video = (event_root / variant_id / f"{source_name}.mp4").resolve(
                        strict=True
                    )
                    source_frames, source_fps, source_source_hw = (
                        materialize_vae._decode_exact_video(source_video)
                    )
                    inference.validate_exact_video_metadata(
                        int(source_frames.shape[0]), source_fps
                    )
                    source_pixels[source_name] = materialize_vae._resize_video(
                        source_frames, bucket, None
                    ).unsqueeze(0)
                    if tuple(map(int, source_pixels[source_name].shape)) != (
                        1,
                        3,
                        81,
                        bucket[0],
                        bucket[1],
                    ):
                        raise CrossAppearancePairError(
                            f"generated source RGB geometry differs: {iid}/{source_name}"
                        )
                    source_video_records[source_name] = {
                        "path": str(source_video),
                        "sha256": file_sha256(source_video),
                        "source_hw": list(map(int, source_source_hw)),
                    }
                    del source_frames
                noop_pixels = source_pixels["noop"].contiguous()
                incomplete_pixels = source_pixels["incomplete"].contiguous()
            with torch.no_grad():
                noop = _vae_encode(
                    vae, noop_pixels.to(device=device, dtype=torch.float32)
                ).float().cpu().contiguous()
                incomplete = _vae_encode(
                    vae, incomplete_pixels.to(device=device, dtype=torch.float32)
                ).float().cpu().contiguous()
            expected_shape = (1, 16, 21, bucket[0] // 8, bucket[1] // 8)
            if any(
                tuple(map(int, tensor.shape)) != expected_shape
                for tensor in (target, noop, incomplete)
            ):
                raise CrossAppearancePairError(f"latent geometry differs: {iid}")
            row_dir = output / "latents" / iid
            row_dir.mkdir(parents=True, mode=0o700)
            latent_path = row_dir / "motion_pair.safetensors"
            save_file(
                {
                    "target": target,
                    "source_noop": noop,
                    "source_incomplete": incomplete,
                },
                str(latent_path),
                metadata={"schema_version": PAIR_SCHEMA_VERSION, "iid": iid},
            )
            row_bases.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "iid": iid,
                    "event_ordinal": ordinal,
                    "event_id": event_id,
                    "variant_id": variant_id,
                    "variant_index": variant_index,
                    "instruction": instruction,
                    "source_video": str(video),
                    "source_video_sha256": file_sha256(video),
                    "target_video": str(video),
                    "target_video_sha256": file_sha256(video),
                    "latents": {
                        "path": str(latent_path),
                        "sha256": file_sha256(latent_path),
                        "shape": list(expected_shape),
                    },
                    "incomplete_cutoff_frame": INCOMPLETE_CUTOFF_FRAME,
                    "same_actor_world_target": True,
                    "target_is_self_generated_action_video": True,
                    "source_authoring_mode": args.source_authoring_mode,
                    "source_variant_is_deterministically_derived_from_target_rgb": (
                        args.source_authoring_mode == "deterministic"
                    ),
                    "source_variants_are_self_generated_counterfactual_videos": (
                        args.source_authoring_mode == "generated_counterfactual"
                    ),
                    "source_variant_videos": source_video_records,
                }
            )
            del frames, pixels, noop_pixels, incomplete_pixels, target, noop, incomplete
            torch.cuda.empty_cache()

    if len(row_bases) != 32:
        raise CrossAppearancePairError("materialized row count differs")
    for mode, flow_root, donor_offset in (
        ("matched", matched_flow_root, 0),
        ("cross", cross_flow_root, 1),
    ):
        rows = []
        for base in row_bases:
            donor_index = (int(base["variant_index"]) + donor_offset) % 4
            donor_id = f"v{donor_index}"
            flow = (flow_root / f'{base["iid"]}.safetensors').resolve(strict=True)
            correspondence_flow = (
                matched_flow_root / f'{base["iid"]}.safetensors'
            ).resolve(strict=True)
            row = dict(base)
            row.update(
                {
                    "flow_bundle": str(flow),
                    "flow_bundle_sha256": file_sha256(flow),
                    "source_correspondence_flow_bundle": str(correspondence_flow),
                    "source_correspondence_flow_bundle_sha256": file_sha256(
                        correspondence_flow
                    ),
                    "motion_anchor_variant_id": donor_id,
                    "motion_anchor_is_cross_appearance": mode == "cross",
                    "anchor_rgb_or_vae_latent_used_by_model": False,
                }
            )
            rows.append(row)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "rows": rows,
            "training_manifest": str(authoring_path),
            "training_manifest_sha256": file_sha256(authoring_path),
            "authoring": str(authoring_path),
            "authoring_sha256": file_sha256(authoring_path),
            "bernini_commit": bernini_revision,
            "veomni_commit": veomni_revision,
            "checkpoint_tree_sha256": trainer.CHECKPOINT_TREE_SHA256,
            "pairing_mode": mode,
            "motion_anchor_cross_appearance": mode == "cross",
            "anchor_rgb_or_vae_latent_used_by_model": False,
            "qwen_used": False,
            "cross_actor_target_used": False,
            "source_authoring_mode": args.source_authoring_mode,
        }
        manifest["manifest_digest"] = object_sha256(manifest)
        (output / f"manifest_{mode}.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "rows": len(row_bases),
                "matched_manifest": str(output / "manifest_matched.json"),
                "cross_manifest": str(output / "manifest_cross.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

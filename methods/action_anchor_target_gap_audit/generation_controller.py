#!/usr/bin/env python3
"""Run one modulo shard of paired Bernini T2V/RV2V generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .audit import (
    MANIFEST_SCHEMA,
    assert_not_protected_write,
    file_sha256,
    load_json,
    write_json,
)


def _probe_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video for probing: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"video probe geometry differs: {path}")
    return {
        "duration": frames / fps, "frame_count": frames, "fps": fps,
        "width": width, "height": height, "backend": "opencv",
    }


def _verify_normalized_source(sample: Mapping[str, Any]) -> bool:
    specification = sample["generation"]["normalized_source"]
    output = Path(specification["path"])
    receipt_path = Path(specification["receipt"])
    if not output.is_file() or output.is_symlink() or not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "mev-action-gap-exact81-source-receipt-v1"
        or receipt.get("original_source", {}).get("sha256") != sample["source"]["sha256"]
        or receipt.get("normalized_source", {}).get("sha256") != file_sha256(output)
    ):
        return False
    probe = _probe_video(output)
    return probe["frame_count"] == 81 and abs(probe["fps"] - 25.0) < 1.0e-6


def _prepare_normalized_source(
    sample: Mapping[str, Any], *, ffmpeg: str,
) -> tuple[Path, str]:
    specification = sample["generation"]["normalized_source"]
    output = Path(specification["path"])
    receipt_path = Path(specification["receipt"])
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        if not _verify_normalized_source(sample):
            raise ValueError(f"partial/unverified normalized source exists: {output.parent}")
        return output, file_sha256(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(sample["source"]["path"])
    source_probe = _probe_video(source)
    if source_probe["duration"] <= 0:
        raise ValueError("source duration differs")
    # Retiming spans the complete event over 80 output intervals.  tpad only
    # protects against the final timestamp rounding one frame short.
    pts_scale = 3.2 / source_probe["duration"]
    video_filter = (
        f"setpts={pts_scale:.12f}*PTS,fps=25,"
        "tpad=stop_mode=clone:stop_duration=1,pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    temporary = output.with_name(f".{output.stem}.tmp-{os.getpid()}.mp4")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"temporary normalized source exists: {temporary}")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-an", "-vf", video_filter,
        "-frames:v", "81", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        normalized_probe = _probe_video(temporary)
        if normalized_probe["frame_count"] != 81 or abs(normalized_probe["fps"] - 25.0) >= 1.0e-6:
            raise RuntimeError(f"exact81 transcode differs: {normalized_probe}")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": "mev-action-gap-exact81-source-receipt-v1",
        "pair_id": sample["pair_id"], "pair_prefix": sample["pair_prefix"],
        "original_source": {
            "path": str(source), "sha256": sample["source"]["sha256"], "probe": source_probe,
        },
        "normalized_source": {
            "path": str(output), "sha256": file_sha256(output), "probe": normalized_probe,
        },
        "contract": {
            "complete_original_event_retimed_to_80_intervals": True,
            "frame_count": 81, "fps": 25, "target_video_read": False,
            "protected_source_modified": False,
        },
        "ffmpeg_argv": command[:-1] + [str(output)],
        "ffmpeg_sha256": file_sha256(ffmpeg), "probe_backend": "opencv",
    }
    write_json(receipt_path, receipt)
    if not _verify_normalized_source(sample):
        raise RuntimeError("normalized source receipt verification failed")
    return output, receipt["normalized_source"]["sha256"]


def _verify_existing(sample: Mapping[str, Any]) -> bool:
    generation = sample["generation"]
    receipt_path = Path(generation["receipt"])
    anchor = Path(generation["anchor"]["path"])
    frozen_base = Path(generation["frozen_base"]["path"])
    if not all(path.is_file() and not path.is_symlink() for path in (receipt_path, anchor, frozen_base)):
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("arms") != ["t2v", "rv2v"]:
        return False
    if not _verify_normalized_source(sample):
        return False
    normalized_sha = file_sha256(sample["generation"]["normalized_source"]["path"])
    if receipt.get("input", {}).get("source_video_sha256") != normalized_sha:
        return False
    outputs = receipt.get("outputs", {})
    for role, path in (("t2v", anchor), ("rv2v", frozen_base)):
        artifact = outputs.get(role, {}).get("mp4") or outputs.get(role)
        if not isinstance(artifact, Mapping) or artifact.get("sha256") != file_sha256(path):
            return False
    return True


def _git_archive_digest(repo_root: Path) -> tuple[str, str]:
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    process = subprocess.Popen(
        ["git", "-C", str(repo_root), "archive", "--format=tar", revision],
        stdout=subprocess.PIPE,
    )
    digest = hashlib.sha256()
    assert process.stdout is not None
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    status = process.wait()
    if status != 0:
        raise RuntimeError("git archive failed")
    return revision, digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema differs")
    assert_not_protected_write(manifest["experiment_root"])
    for sample in manifest.get("samples", []):
        generation = sample["generation"]
        for output_path in (
            generation["output_dir"],
            generation["receipt"],
            generation["anchor"]["path"],
            generation["frozen_base"]["path"],
            generation["normalized_source"]["path"],
            generation["normalized_source"]["receipt"],
        ):
            assert_not_protected_write(output_path)
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker index differs")
    repo_root = Path(args.repo_root).resolve(strict=True)
    inference_root = (
        Path(args.inference_root).resolve(strict=True)
        if args.inference_root
        else repo_root / "methods/bernini_action_editing"
    )
    inference = inference_root / "infer_native_identity_generation_canary.py"
    if not inference.is_file():
        raise FileNotFoundError(inference)
    revision, archive_sha = _git_archive_digest(repo_root)
    selected = [
        sample for sample in manifest["samples"]
        if sample["ordinal"] % args.num_workers == args.worker_index
    ]
    for ordinal, sample in enumerate(selected):
        if _verify_existing(sample):
            print(json.dumps({"pair_prefix": sample["pair_prefix"], "status": "verified_skip"}), flush=True)
            continue
        normalized_source, normalized_source_sha = _prepare_normalized_source(
            sample, ffmpeg=args.ffmpeg
        )
        output_dir = Path(sample["generation"]["output_dir"])
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError(f"partial/unverified output exists: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        master_port = args.master_port_base + ordinal
        command = [
            args.python, "-B", "-m", "torch.distributed.run",
            "--nnodes=1", "--node_rank=0", "--nproc_per_node=4",
            "--master_addr=127.0.0.1", f"--master_port={master_port}",
            str(inference),
            "--bernini-root", args.bernini_root,
            "--veomni-root", args.veomni_root,
            "--checkpoint", args.checkpoint,
            "--checkpoint-content-manifest", args.checkpoint_content_manifest,
            "--source-video", str(normalized_source),
            "--expected-source-sha256", normalized_source_sha,
            "--action-prompt", sample["generation_caption"],
            "--expected-action-prompt-sha256", sample["generation_caption_sha256"],
            "--output-dir", str(output_dir),
            "--arms", "t2v", "rv2v",
            "--num-inference-steps", "40", "--seed", str(sample["seed"]),
            "--method-source-revision", revision,
            "--method-source-archive-sha256", archive_sha,
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(inference_root)
        print(json.dumps({
            "pair_prefix": sample["pair_prefix"], "status": "start",
            "seed": sample["seed"], "master_port": master_port,
        }), flush=True)
        subprocess.run(command, check=True, cwd=inference_root, env=environment)
        if not _verify_existing(sample):
            raise RuntimeError(f"generation receipt verification failed: {sample['pair_prefix']}")
        print(json.dumps({"pair_prefix": sample["pair_prefix"], "status": "complete"}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--inference-root",
        help="Closed bernini_action_editing source snapshot; defaults to REPO_ROOT/methods/bernini_action_editing",
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--bernini-root", required=True)
    parser.add_argument("--veomni-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-content-manifest", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--master-port-base", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

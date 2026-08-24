#!/usr/bin/env python3
"""Fail-closed validation for the 68-video full-field V4 review sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    streams = value.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        fail(f"video stream count differs: {path}")
    stream = streams[0]
    if (
        stream.get("codec_name") != "h264"
        or stream.get("r_frame_rate") != "25/1"
        or int(stream.get("nb_frames", 0)) != 81
        or int(stream.get("width", 0)) <= 0
        or int(stream.get("height", 0)) <= 0
        or abs(float(value["format"]["duration"]) - 3.24) > 0.01
    ):
        fail(f"video geometry/codec contract differs: {path}: {value}")
    return {
        "codec": "h264",
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": 25.0,
        "frame_count": 81,
        "duration": float(value["format"]["duration"]),
    }


def validate_generated(
    path: Path,
    *,
    expected_step: int | None,
    expected_source_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        fail(f"missing generated video: {path}")
    receipt_path = path.with_name(f"{path.name}.receipt.json")
    if not receipt_path.is_file():
        fail(f"missing generated receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    output = receipt.get("output", {})
    digest = sha256(path)
    if output.get("sha256") != digest:
        fail(f"video SHA-256 differs from receipt: {path}")
    probe = probe_video(path)
    if (
        output.get("frame_count") != 81
        or float(output.get("fps", 0.0)) != 25.0
        or output.get("width") != probe["width"]
        or output.get("height") != probe["height"]
    ):
        fail(f"decoded output receipt differs from ffprobe: {path}")
    sampling = receipt.get("sampling", {})
    if (
        sampling.get("seed") != 2026081601
        or sampling.get("num_inference_steps") != 40
        or sampling.get("source_onset_policy") != "hard1_every_step"
    ):
        fail(f"sampling contract differs: {path}")
    input_value = receipt.get("input", {})
    source_digest = input_value.get("source_video_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        fail(f"source digest is missing: {path}")
    if expected_source_sha256 is not None and source_digest != expected_source_sha256:
        fail(f"source digest differs inside one case: {path}")
    adapter = receipt.get("adapter", {})
    if expected_step is None:
        if adapter.get("enabled") is not False or adapter.get("tensor_count") != 0:
            fail(f"Frozen base unexpectedly contains an adapter: {path}")
    elif (
        adapter.get("enabled") is not True
        or adapter.get("strictly_reloaded") is not True
        or adapter.get("safe_merged_for_inference") is not True
        or adapter.get("tensor_count") != 480
        or adapter.get("training_global_step") != expected_step
    ):
        fail(f"adapter reload contract differs: {path}: {adapter}")
    return probe, source_digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    root = Path(args.review_root).resolve(strict=True)
    manifest = json.loads(
        Path(args.manifest).resolve(strict=True).read_text(encoding="utf-8")
    )
    steps = [int(item) for item in manifest["steps"]]
    arms = [item["key"] for item in manifest["arm_definitions"]]
    cases = manifest["cases"]
    if len(steps) != 4 or len(arms) != 4 or len(cases) != 4:
        fail("review manifest must contain 4 steps, 4 arms and 4 cases")

    generated_count = 0
    all_video_count = 0
    case_summary: dict[str, Any] = {}
    for case in cases:
        iid = case["iid"]
        summary: dict[str, Any] = {"videos": {}}
        for label in ("source", "anchor"):
            relative = Path(case[label])
            summary["videos"][label] = probe_video(root / relative)
            all_video_count += 1
        base_path = root / case["base"]
        base_probe, source_digest = validate_generated(
            base_path, expected_step=None, expected_source_sha256=None
        )
        summary["videos"]["frozen_base"] = base_probe
        generated_count += 1
        all_video_count += 1
        for arm in arms:
            for step in steps:
                relative = case["arms"][arm].replace("{step}", str(step))
                probe, _ = validate_generated(
                    root / relative,
                    expected_step=step,
                    expected_source_sha256=source_digest,
                )
                summary["videos"][f"{arm}@{step}"] = probe
                generated_count += 1
                all_video_count += 1
        case_summary[iid] = summary
    if generated_count != 68 or all_video_count != 76:
        fail(
            f"review cardinality differs: generated={generated_count}, all={all_video_count}"
        )
    result = {
        "schema_version": "action-fullfield-v4-review-validation-v1",
        "valid": True,
        "generated_video_count": generated_count,
        "all_review_video_count": all_video_count,
        "case_count": len(cases),
        "arm_count": len(arms),
        "checkpoint_steps": steps,
        "sampling": {
            "seed": 2026081601,
            "num_inference_steps": 40,
            "source_onset_policy": "hard1_every_step",
        },
        "cases": case_summary,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

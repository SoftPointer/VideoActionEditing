#!/usr/bin/env python3
"""Materialize and audit the case01 matched bone-intervention inputs.

This is a deterministic media preprocessor, not a renderer or a training entry
point.  It intentionally uses only the Python standard library plus ffmpeg and
ffprobe so that the intervention recipe remains easy to replay.

The four matched-codec arms are:

* codec_only_present: no pixel intervention;
* bone_removed: per-frame SAM2 bone support, dilated by three 8-neighbour passes,
  filled by bidirectional boundary interpolation;
* bone_translated_up150: bone_removed plus the sham-identical target support,
  then the exact source bone pixels shifted upward by 150 pixels using the
  undilated SAM2 mask;
* sham_control_up150: the same interpolation used by bone_removed, applied to
  the same support shifted upward by 150 pixels while the source bone remains.

The byte-exact source MP4 is also copied as a provenance control.  Canonical
changed-pixel statistics are computed before the common H.264 encoding step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_IID = "288545b9c031491a"
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
WIDTH = 704
HEIGHT = 736
FPS = 25
FRAME_COUNT = 81
FRAME_PIXELS = WIDTH * HEIGHT
RGB_FRAME_BYTES = FRAME_PIXELS * 3
TRANSLATE_DY = -150
DILATION_PASSES = 3
KEYFRAMES = (0, 20, 40, 60, 80)
MATCHED_ARMS = (
    "codec_only_present",
    "bone_removed",
    "bone_translated_up150",
    "sham_control_up150",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--masklet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--manual-visual-audit",
        choices=("PENDING", "PASS_INPUT_ASSET_QA_ONLY"),
        default="PENDING",
        help=(
            "Record a completed all-81-frame input-asset review. This never "
            "authorizes a renderer or scientific-result claim."
        ),
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(
    command: Sequence[str],
    *,
    input_bytes: bytes | bytearray | memoryview | None = None,
    capture_stdout: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command),
        input=input_bytes,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )


def ffmpeg_version(ffmpeg: str) -> str:
    result = run((ffmpeg, "-version"), capture_stdout=True)
    return result.stdout.decode("utf-8", errors="replace").splitlines()[0]


def ffprobe_media(ffprobe: str, path: Path) -> dict:
    result = run(
        (
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_name,width,height,pix_fmt,r_frame_rate,"
                "avg_frame_rate,nb_frames,nb_read_frames"
            ),
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ),
        capture_stdout=True,
    )
    value = json.loads(result.stdout)
    stream = value["streams"][0]
    if int(stream["width"]) != WIDTH or int(stream["height"]) != HEIGHT:
        raise RuntimeError(f"unexpected geometry for {path}: {stream}")
    if stream["r_frame_rate"] != f"{FPS}/1":
        raise RuntimeError(f"unexpected frame rate for {path}: {stream}")
    if int(stream.get("nb_read_frames") or stream.get("nb_frames")) != FRAME_COUNT:
        raise RuntimeError(f"unexpected frame count for {path}: {stream}")
    if abs(float(value["format"]["duration"]) - FRAME_COUNT / FPS) > 1e-6:
        raise RuntimeError(f"unexpected duration for {path}: {value['format']}")
    return value


def decode_video_rgb(ffmpeg: str, source_video: Path) -> bytes:
    result = run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        capture_stdout=True,
    )
    expected = RGB_FRAME_BYTES * FRAME_COUNT
    if len(result.stdout) != expected:
        raise RuntimeError(
            f"decoded source byte count mismatch: {len(result.stdout)} != {expected}"
        )
    return result.stdout


def decode_mask_sequence(ffmpeg: str, mask_dir: Path) -> list[bytes]:
    expected_paths = [mask_dir / f"{index:05d}.png" for index in range(FRAME_COUNT)]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mask frames: {missing[:3]}")
    result = run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-start_number",
            "0",
            "-i",
            str(mask_dir / "%05d.png"),
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        capture_stdout=True,
    )
    expected = FRAME_PIXELS * FRAME_COUNT
    if len(result.stdout) != expected:
        raise RuntimeError(
            f"decoded mask byte count mismatch: {len(result.stdout)} != {expected}"
        )
    frames = [
        result.stdout[index * FRAME_PIXELS : (index + 1) * FRAME_PIXELS]
        for index in range(FRAME_COUNT)
    ]
    for index, frame in enumerate(frames):
        values = set(frame)
        if not values.issubset({0, 255}) or 255 not in values:
            raise RuntimeError(f"mask {index:05d} is not nonempty binary: {values}")
    return frames


def active_indices(mask: bytes | bytearray) -> set[int]:
    return {index for index, value in enumerate(mask) if value}


def binary_mask(indices: Iterable[int]) -> bytes:
    result = bytearray(FRAME_PIXELS)
    for index in indices:
        result[index] = 255
    return bytes(result)


def dilate_indices(indices: set[int], passes: int) -> set[int]:
    active = set(indices)
    for _ in range(passes):
        expanded = set(active)
        for index in active:
            y, x = divmod(index, WIDTH)
            for ny in range(max(0, y - 1), min(HEIGHT, y + 2)):
                row = ny * WIDTH
                for nx in range(max(0, x - 1), min(WIDTH, x + 2)):
                    expanded.add(row + nx)
        active = expanded
    return active


def shift_indices(indices: set[int], dy: int) -> set[int]:
    shifted: set[int] = set()
    for index in indices:
        y, x = divmod(index, WIDTH)
        new_y = y + dy
        if 0 <= new_y < HEIGHT:
            shifted.add(new_y * WIDTH + x)
    return shifted


def contiguous_runs(values: list[int]) -> Iterable[tuple[int, int]]:
    if not values:
        return
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            yield start, previous
            start = value
        previous = value
    yield start, previous


def interpolate_support(source: bytes, support: set[int]) -> bytes:
    """Fill support from source boundary pixels, changing no pixel outside it.

    Horizontal and vertical linear interpolants are computed from the closest
    non-support pixels bracketing each contiguous support run.  Their rounded
    mean is used where both are available.
    """

    if not support:
        return source
    horizontal: dict[int, tuple[int, int, int]] = {}
    vertical: dict[int, tuple[int, int, int]] = {}

    rows: dict[int, list[int]] = {}
    columns: dict[int, list[int]] = {}
    for index in support:
        y, x = divmod(index, WIDTH)
        rows.setdefault(y, []).append(x)
        columns.setdefault(x, []).append(y)

    for y, xs in rows.items():
        xs.sort()
        for start, end in contiguous_runs(xs):
            left = start - 1
            right = end + 1
            if left < 0 or right >= WIDTH:
                continue
            denominator = right - left
            left_offset = (y * WIDTH + left) * 3
            right_offset = (y * WIDTH + right) * 3
            left_rgb = source[left_offset : left_offset + 3]
            right_rgb = source[right_offset : right_offset + 3]
            for x in range(start, end + 1):
                numerator = x - left
                rgb = tuple(
                    (
                        left_rgb[channel] * (denominator - numerator)
                        + right_rgb[channel] * numerator
                        + denominator // 2
                    )
                    // denominator
                    for channel in range(3)
                )
                horizontal[y * WIDTH + x] = rgb

    for x, ys in columns.items():
        ys.sort()
        for start, end in contiguous_runs(ys):
            top = start - 1
            bottom = end + 1
            if top < 0 or bottom >= HEIGHT:
                continue
            denominator = bottom - top
            top_offset = (top * WIDTH + x) * 3
            bottom_offset = (bottom * WIDTH + x) * 3
            top_rgb = source[top_offset : top_offset + 3]
            bottom_rgb = source[bottom_offset : bottom_offset + 3]
            for y in range(start, end + 1):
                numerator = y - top
                rgb = tuple(
                    (
                        top_rgb[channel] * (denominator - numerator)
                        + bottom_rgb[channel] * numerator
                        + denominator // 2
                    )
                    // denominator
                    for channel in range(3)
                )
                vertical[y * WIDTH + x] = rgb

    result = bytearray(source)
    for index in support:
        horizontal_rgb = horizontal.get(index)
        vertical_rgb = vertical.get(index)
        if horizontal_rgb is None and vertical_rgb is None:
            raise RuntimeError(f"support touches both image axes at pixel {index}")
        if horizontal_rgb is None:
            rgb = vertical_rgb
        elif vertical_rgb is None:
            rgb = horizontal_rgb
        else:
            rgb = tuple(
                (horizontal_rgb[channel] + vertical_rgb[channel] + 1) // 2
                for channel in range(3)
            )
        offset = index * 3
        result[offset : offset + 3] = bytes(rgb)
    return bytes(result)


def translate_bone_pixels(
    source: bytes,
    removed: bytes,
    source_bone_indices: set[int],
    dy: int,
) -> tuple[bytes, set[int]]:
    result = bytearray(removed)
    translated_indices: set[int] = set()
    for source_index in source_bone_indices:
        y, x = divmod(source_index, WIDTH)
        new_y = y + dy
        if not 0 <= new_y < HEIGHT:
            raise RuntimeError(f"translated bone clips at source pixel {source_index}")
        target_index = new_y * WIDTH + x
        translated_indices.add(target_index)
        source_offset = source_index * 3
        target_offset = target_index * 3
        result[target_offset : target_offset + 3] = source[
            source_offset : source_offset + 3
        ]
    if len(translated_indices) != len(source_bone_indices):
        raise RuntimeError("translation is not one-to-one")
    return bytes(result), translated_indices


def copy_support_pixels(
    base: bytes, donor: bytes, support: set[int]
) -> bytes:
    """Copy donor RGB bytes on support and preserve base everywhere else."""

    result = bytearray(base)
    for index in support:
        offset = index * 3
        result[offset : offset + 3] = donor[offset : offset + 3]
    return bytes(result)


def assert_translated_matched_relations(
    *,
    source: bytes,
    removed: bytes,
    translated: bytes,
    sham: bytes,
    source_bone: set[int],
    removal_support: set[int],
    sham_support: set[int],
    translated_bone: set[int],
    dy: int,
) -> dict:
    """Hard-assert the source- and target-tube matched relations."""

    if removal_support & sham_support:
        raise RuntimeError("original and target supports overlap")
    if not translated_bone.issubset(sham_support):
        raise RuntimeError("translated bone is not contained in target sham support")

    for index in removal_support:
        offset = index * 3
        if translated[offset : offset + 3] != removed[offset : offset + 3]:
            raise RuntimeError(
                f"translated original-support byte mismatch at pixel {index}"
            )

    target_nonbone = sham_support - translated_bone
    for index in target_nonbone:
        offset = index * 3
        if translated[offset : offset + 3] != sham[offset : offset + 3]:
            raise RuntimeError(
                f"translated target non-bone byte mismatch at pixel {index}"
            )

    for source_index in source_bone:
        y, x = divmod(source_index, WIDTH)
        target_index = (y + dy) * WIDTH + x
        if target_index not in translated_bone:
            raise RuntimeError("translated bone mapping mismatch")
        source_offset = source_index * 3
        target_offset = target_index * 3
        if (
            translated[target_offset : target_offset + 3]
            != source[source_offset : source_offset + 3]
        ):
            raise RuntimeError(
                f"translated target bone byte mismatch at pixel {target_index}"
            )

    return {
        "original_support_equals_removed": True,
        "original_support_pixels_checked": len(removal_support),
        "target_nonbone_support_equals_sham": True,
        "target_nonbone_support_pixels_checked": len(target_nonbone),
        "target_bone_equals_shifted_source": True,
        "target_bone_pixels_checked": len(translated_bone),
        "original_and_target_supports_disjoint": True,
    }


def sequence_digest(frames: Sequence[bytes]) -> tuple[str, list[str]]:
    sequence = hashlib.sha256()
    frame_hashes: list[str] = []
    for index, frame in enumerate(frames):
        frame_sha = hashlib.sha256(frame).hexdigest()
        frame_hashes.append(frame_sha)
        sequence.update(f"{index:05d} {frame_sha}\n".encode("ascii"))
    return sequence.hexdigest(), frame_hashes


def frame_diff_metrics(
    source: bytes, changed: bytes, expected_support: set[int]
) -> dict:
    changed_count = 0
    outside_changed_count = 0
    outside_changed_examples: list[int] = []
    rgb_l1 = 0
    for index in range(FRAME_PIXELS):
        offset = index * 3
        if (
            source[offset] != changed[offset]
            or source[offset + 1] != changed[offset + 1]
            or source[offset + 2] != changed[offset + 2]
        ):
            if index not in expected_support:
                outside_changed_count += 1
                if len(outside_changed_examples) < 8:
                    outside_changed_examples.append(index)
                continue
            changed_count += 1
            rgb_l1 += (
                abs(source[offset] - changed[offset])
                + abs(source[offset + 1] - changed[offset + 1])
                + abs(source[offset + 2] - changed[offset + 2])
            )
    if outside_changed_count:
        raise RuntimeError(
            "pixels changed outside declared support: "
            f"count={outside_changed_count}, examples={outside_changed_examples}"
        )
    return {
        "expected_tube_pixels": len(expected_support),
        "changed_pixels": changed_count,
        "unchanged_inside_tube_pixels": len(expected_support) - changed_count,
        "outside_expected_tube_changed_pixels": outside_changed_count,
        "outside_expected_tube_changed_pixel_examples": outside_changed_examples,
        "rgb_l1_sum_inside_tube": rgb_l1,
    }


def summarize_frame_metrics(per_frame: list[dict]) -> dict:
    changed = [row["changed_pixels"] for row in per_frame]
    tubes = [row["expected_tube_pixels"] for row in per_frame]
    return {
        "frame_count": len(per_frame),
        "space_time_changed_pixels": sum(changed),
        "space_time_expected_tube_pixels": sum(tubes),
        "space_time_frame_pixels": FRAME_PIXELS * len(per_frame),
        "changed_fraction_of_expected_tube": sum(changed) / sum(tubes),
        "changed_fraction_of_all_frame_pixels": sum(changed)
        / (FRAME_PIXELS * len(per_frame)),
        "outside_expected_tube_changed_pixels": sum(
            row["outside_expected_tube_changed_pixels"] for row in per_frame
        ),
        "changed_pixels_per_frame": {
            "minimum": min(changed),
            "median": statistics.median(changed),
            "maximum": max(changed),
        },
        "expected_tube_pixels_per_frame": {
            "minimum": min(tubes),
            "median": statistics.median(tubes),
            "maximum": max(tubes),
        },
        "rgb_l1_sum_inside_tube": sum(
            row["rgb_l1_sum_inside_tube"] for row in per_frame
        ),
        "per_frame": per_frame,
    }


def mask_geometry(indices: set[int]) -> dict:
    ys: list[int] = []
    xs: list[int] = []
    for index in indices:
        y, x = divmod(index, WIDTH)
        ys.append(y)
        xs.append(x)
    return {
        "area": len(indices),
        "bbox_xyxy_inclusive": [min(xs), min(ys), max(xs), max(ys)],
    }


def summarize_tube(index_frames: Sequence[set[int]]) -> dict:
    areas = [len(indices) for indices in index_frames]
    union = set().union(*index_frames)
    return {
        "frame_count": len(index_frames),
        "space_time_pixels": sum(areas),
        "spatial_union_pixels": len(union),
        "area_per_frame": {
            "minimum": min(areas),
            "median": statistics.median(areas),
            "maximum": max(areas),
        },
        "per_frame": [
            {"frame_index": index, **mask_geometry(indices)}
            for index, indices in enumerate(index_frames)
        ],
    }


def encode_rgb_video(
    ffmpeg: str, frames: Sequence[bytes], output_path: Path
) -> None:
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-frames:v",
        str(FRAME_COUNT),
        "-movflags",
        "+faststart",
        "-map_metadata",
        "-1",
        str(output_path),
    )
    run(command, input_bytes=b"".join(frames))


def encode_gray_ffv1(
    ffmpeg: str, masks: Sequence[bytes], output_path: Path
) -> str:
    payload = b"".join(masks)
    command = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s:v",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "gray",
        str(output_path),
    )
    run(command, input_bytes=payload)
    decoded = run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output_path),
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        capture_stdout=True,
    ).stdout
    if decoded != payload:
        raise RuntimeError(f"FFV1 mask round-trip mismatch: {output_path}")
    return hashlib.sha256(payload).hexdigest()


def make_all81_sheet(ffmpeg: str, video: Path, output: Path, label: str) -> None:
    escaped = label.replace(":", "_")
    filter_value = (
        "scale=176:184,"
        f"drawtext=text='{escaped}  f%{{n}}':x=4:y=4:fontsize=14:"
        "fontcolor=white:box=1:boxcolor=black@0.65,"
        "tile=9x9:nb_frames=81:padding=1:margin=0"
    )
    run(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            filter_value,
            "-frames:v",
            "1",
            str(output),
        )
    )


def make_keyframe_sheet(
    ffmpeg: str, videos: Sequence[tuple[str, Path]], output: Path
) -> None:
    command: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for _, path in videos:
        command.extend(("-i", str(path)))
    select = "+".join(f"eq(n\\,{index})" for index in KEYFRAMES)
    filters: list[str] = []
    labels: list[str] = []
    for input_index, (label, _) in enumerate(videos):
        output_label = f"column{input_index}"
        labels.append(f"[{output_label}]")
        filters.append(
            f"[{input_index}:v]select={select},scale=281:294,"
            f"drawtext=text='{label}':x=5:y=5:fontsize=16:"
            f"fontcolor=white:box=1:boxcolor=black@0.65,"
            f"tile=1x5:nb_frames=5[{output_label}]"
        )
    filters.append("".join(labels) + f"hstack=inputs={len(videos)}[sheet]")
    command.extend(
        (
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[sheet]",
            "-frames:v",
            "1",
            str(output),
        )
    )
    run(command)


def collect_outputs(root: Path) -> list[dict]:
    outputs: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        outputs.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return outputs


def main() -> int:
    args = parse_args()
    source_video = args.source_video.resolve()
    masklet_root = args.masklet_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    if not source_video.is_file():
        raise FileNotFoundError(source_video)
    if sha256_path(source_video) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("source video SHA-256 mismatch")

    masklet_receipt_path = masklet_root / "receipt.json"
    masklet_receipt = json.loads(masklet_receipt_path.read_text(encoding="utf-8"))
    if (
        masklet_receipt.get("iid") != EXPECTED_IID
        or masklet_receipt.get("source", {}).get("sha256")
        != EXPECTED_SOURCE_SHA256
    ):
        raise RuntimeError("masklet receipt authority mismatch")

    source_probe = ffprobe_media(args.ffprobe, source_video)
    bone_masks = decode_mask_sequence(args.ffmpeg, masklet_root / "masks" / "bone")
    dog_masks = decode_mask_sequence(args.ffmpeg, masklet_root / "masks" / "dog")
    bone_index_frames = [active_indices(mask) for mask in bone_masks]
    dog_index_frames = [active_indices(mask) for mask in dog_masks]
    removal_support_frames = [
        dilate_indices(indices, DILATION_PASSES) for indices in bone_index_frames
    ]
    translated_bone_frames = [
        shift_indices(indices, TRANSLATE_DY) for indices in bone_index_frames
    ]
    sham_support_frames = [
        shift_indices(indices, TRANSLATE_DY) for indices in removal_support_frames
    ]
    translated_union_frames = [
        removal | target
        for removal, target in zip(removal_support_frames, sham_support_frames)
    ]
    for index, (original, translated) in enumerate(
        zip(bone_index_frames, translated_bone_frames)
    ):
        if len(original) != len(translated):
            raise RuntimeError(f"translated bone clips in frame {index}")
    dog_overlap = {
        "removal_support": sum(
            len(dog & support)
            for dog, support in zip(dog_index_frames, removal_support_frames)
        ),
        "translated_bone": sum(
            len(dog & translated)
            for dog, translated in zip(dog_index_frames, translated_bone_frames)
        ),
        "sham_support": sum(
            len(dog & support)
            for dog, support in zip(dog_index_frames, sham_support_frames)
        ),
    }
    if any(dog_overlap.values()):
        raise RuntimeError(f"intervention support intersects dog mask: {dog_overlap}")

    output_root.mkdir(parents=True)
    videos_dir = output_root / "videos"
    tubes_dir = output_root / "tubes"
    qa_dir = output_root / "qa"
    videos_dir.mkdir()
    tubes_dir.mkdir()
    qa_dir.mkdir()

    exact_original = videos_dir / "exact_original.mp4"
    shutil.copyfile(source_video, exact_original)
    if sha256_path(exact_original) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("exact-original copy is not byte exact")

    source_raw = decode_video_rgb(args.ffmpeg, source_video)
    source_frames = [
        source_raw[index * RGB_FRAME_BYTES : (index + 1) * RGB_FRAME_BYTES]
        for index in range(FRAME_COUNT)
    ]
    present_digest, present_frame_hashes = sequence_digest(source_frames)

    arm_frames: dict[str, list[bytes]] = {"codec_only_present": source_frames}
    arm_metrics: dict[str, dict] = {
        "codec_only_present": {
            "canonical_rgb_sequence_digest": present_digest,
            "canonical_frame_sha256": present_frame_hashes,
            "frame_count": FRAME_COUNT,
            "space_time_changed_pixels": 0,
            "space_time_expected_tube_pixels": 0,
            "space_time_frame_pixels": FRAME_PIXELS * FRAME_COUNT,
            "changed_fraction_of_all_frame_pixels": 0.0,
            "outside_expected_tube_changed_pixels": 0,
            "per_frame": [
                {
                    "frame_index": index,
                    "expected_tube_pixels": 0,
                    "changed_pixels": 0,
                    "unchanged_inside_tube_pixels": 0,
                    "outside_expected_tube_changed_pixels": 0,
                    "rgb_l1_sum_inside_tube": 0,
                }
                for index in range(FRAME_COUNT)
            ],
        }
    }

    removed_frames: list[bytes] = []
    translated_frames: list[bytes] = []
    sham_frames: list[bytes] = []
    removed_rows: list[dict] = []
    translated_rows: list[dict] = []
    sham_rows: list[dict] = []
    translated_symmetry_rows: list[dict] = []
    for index, source_frame in enumerate(source_frames):
        removal_support = removal_support_frames[index]
        removed = interpolate_support(source_frame, removal_support)
        sham = interpolate_support(source_frame, sham_support_frames[index])
        matched_target_background = copy_support_pixels(
            removed, sham, sham_support_frames[index]
        )
        translated, translated_bone = translate_bone_pixels(
            source_frame,
            matched_target_background,
            bone_index_frames[index],
            TRANSLATE_DY,
        )
        if translated_bone != translated_bone_frames[index]:
            raise RuntimeError(f"translation mask mismatch at frame {index}")
        symmetry = assert_translated_matched_relations(
            source=source_frame,
            removed=removed,
            translated=translated,
            sham=sham,
            source_bone=bone_index_frames[index],
            removal_support=removal_support,
            sham_support=sham_support_frames[index],
            translated_bone=translated_bone,
            dy=TRANSLATE_DY,
        )

        removed_frames.append(removed)
        translated_frames.append(translated)
        sham_frames.append(sham)
        translated_symmetry_rows.append({"frame_index": index, **symmetry})
        removed_rows.append(
            {
                "frame_index": index,
                **frame_diff_metrics(source_frame, removed, removal_support),
            }
        )
        translated_rows.append(
            {
                "frame_index": index,
                **frame_diff_metrics(
                    source_frame, translated, translated_union_frames[index]
                ),
            }
        )
        sham_rows.append(
            {
                "frame_index": index,
                **frame_diff_metrics(
                    source_frame, sham, sham_support_frames[index]
                ),
            }
        )

    arm_frames["bone_removed"] = removed_frames
    arm_frames["bone_translated_up150"] = translated_frames
    arm_frames["sham_control_up150"] = sham_frames
    for arm, rows in (
        ("bone_removed", removed_rows),
        ("bone_translated_up150", translated_rows),
        ("sham_control_up150", sham_rows),
    ):
        digest, hashes = sequence_digest(arm_frames[arm])
        arm_metrics[arm] = {
            "canonical_rgb_sequence_digest": digest,
            "canonical_frame_sha256": hashes,
            **summarize_frame_metrics(rows),
        }
        if arm_metrics[arm]["outside_expected_tube_changed_pixels"] != 0:
            raise RuntimeError(f"out-of-tube pixel changed in {arm}")

    matched_video_paths: dict[str, Path] = {}
    for arm in MATCHED_ARMS:
        output_path = videos_dir / f"{arm}.mp4"
        encode_rgb_video(args.ffmpeg, arm_frames[arm], output_path)
        matched_video_paths[arm] = output_path

    tube_payloads = {
        "bone_sam2": bone_masks,
        "removal_support_dilate3": [
            binary_mask(indices) for indices in removal_support_frames
        ],
        "translated_bone_up150": [
            binary_mask(indices) for indices in translated_bone_frames
        ],
        "translated_union_up150": [
            binary_mask(indices) for indices in translated_union_frames
        ],
        "sham_support_up150": [
            binary_mask(indices) for indices in sham_support_frames
        ],
    }
    tube_raw_sha256: dict[str, str] = {}
    tube_video_paths: dict[str, Path] = {}
    for name, masks in tube_payloads.items():
        path = tubes_dir / f"{name}.mkv"
        tube_raw_sha256[name] = encode_gray_ffv1(args.ffmpeg, masks, path)
        tube_video_paths[name] = path

    all_video_paths = {"exact_original": exact_original, **matched_video_paths}
    media_probes: dict[str, dict] = {}
    for arm, path in all_video_paths.items():
        media_probes[arm] = ffprobe_media(args.ffprobe, path)
        make_all81_sheet(
            args.ffmpeg, path, qa_dir / f"all81_{arm}_9x9.jpg", arm
        )
    make_keyframe_sheet(
        args.ffmpeg,
        [(arm, path) for arm, path in all_video_paths.items()],
        qa_dir / "keyframes_0_20_40_60_80_all5.jpg",
    )

    tube_metrics = {
        "bone_sam2": summarize_tube(bone_index_frames),
        "removal_support_dilate3": summarize_tube(removal_support_frames),
        "translated_bone_up150": summarize_tube(translated_bone_frames),
        "translated_union_up150": summarize_tube(translated_union_frames),
        "sham_support_up150": summarize_tube(sham_support_frames),
    }
    for name, metrics in tube_metrics.items():
        metrics["raw_gray_sequence_sha256"] = tube_raw_sha256[name]
        metrics["lossless_ffv1_path"] = tube_video_paths[name].relative_to(
            output_root
        ).as_posix()

    matched_codec_contract = {
        "codec_name": "h264",
        "encoder": "libx264",
        "preset": "medium",
        "crf": 17,
        "pixel_format": "yuv420p",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "duration_seconds": FRAME_COUNT / FPS,
        "audio": False,
    }
    for arm in MATCHED_ARMS:
        stream = media_probes[arm]["streams"][0]
        if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
            raise RuntimeError(f"matched codec contract mismatch for {arm}: {stream}")

    rejected_candidates: list[dict] = []
    provisional_root = Path("/private/tmp/0821_object_canary_inputs_r1")
    for filename, reason in (
        (
            "case01-bone-removed-r4.mp4",
            "fixed union removelogo support leaves a conspicuous rectangular blur tube",
        ),
        (
            "case01-bone-translated-up150-r1.mp4",
            "inherits the fixed-union rectangular removal artifact",
        ),
        (
            "case01-sham-blur-bone-present-r2.mp4",
            "sham blur is not spatially matched to the translated target tube",
        ),
    ):
        path = provisional_root / filename
        if path.is_file():
            rejected_candidates.append(
                {
                    "path": str(path),
                    "sha256": sha256_path(path),
                    "reason_not_promoted": reason,
                }
            )

    manifest = {
        "schema_version": "bernini-case01-matched-bone-interventions-v1",
        "case_id": "case01",
        "iid": EXPECTED_IID,
        "instruction": "Make the dog pick up the bone and hold it in its mouth.",
        "authority": {
            "source_video_input_path": str(source_video),
            "source_video_sha256": EXPECTED_SOURCE_SHA256,
            "source_probe": source_probe,
            "masklet_root": str(masklet_root),
            "masklet_receipt_path": str(masklet_receipt_path),
            "masklet_receipt_sha256": sha256_path(masklet_receipt_path),
            "materializer_path": str(Path(__file__).resolve()),
            "materializer_sha256": sha256_path(Path(__file__).resolve()),
            "ffmpeg_version": ffmpeg_version(args.ffmpeg),
        },
        "intervention_recipe": {
            "bone_removed": {
                "input_mask": "per-frame binary SAM2 bone mask",
                "support_expansion": (
                    "three deterministic 8-neighbour dilation passes "
                    "(Chebyshev radius 3)"
                ),
                "fill": (
                    "rounded mean of horizontal and vertical linear "
                    "interpolants between nearest non-support boundary pixels"
                ),
                "generative_model_used": False,
            },
            "bone_translated_up150": {
                "background": (
                    "bone_removed at the original support plus byte-identical "
                    "sham interpolation across the full translated target support"
                ),
                "translation_xy_pixels": [0, TRANSLATE_DY],
                "foreground": (
                    "exact decoded RGB pixels from the source bone under the "
                    "undilated per-frame SAM2 mask"
                ),
                "matched_relations": (
                    "original-support bytes equal bone_removed; translated "
                    "target non-bone support bytes equal sham_control_up150"
                ),
                "clipped_pixels": 0,
                "generative_model_used": False,
            },
            "sham_control_up150": {
                "bone_state": "present at original source location",
                "support": (
                    "the exact per-frame removal support translated by [0,-150]"
                ),
                "fill": "identical boundary interpolation as bone_removed",
                "generative_model_used": False,
            },
        },
        "matched_codec_contract": matched_codec_contract,
        "media_probes": media_probes,
        "canonical_precodec_pixel_metrics": arm_metrics,
        "tube_metrics": tube_metrics,
        "translated_matched_symmetry_assertions": {
            "all_frames_passed": True,
            "frame_count": FRAME_COUNT,
            "original_support_equals_removed": True,
            "target_nonbone_support_equals_sham": True,
            "target_bone_equals_shifted_source": True,
            "original_and_target_supports_disjoint": True,
            "space_time_original_support_pixels_checked": sum(
                row["original_support_pixels_checked"]
                for row in translated_symmetry_rows
            ),
            "space_time_target_nonbone_support_pixels_checked": sum(
                row["target_nonbone_support_pixels_checked"]
                for row in translated_symmetry_rows
            ),
            "space_time_target_bone_pixels_checked": sum(
                row["target_bone_pixels_checked"]
                for row in translated_symmetry_rows
            ),
            "per_frame": translated_symmetry_rows,
        },
        "dog_sam2_intervention_overlap_space_time_pixels": dog_overlap,
        "provisional_candidates_not_promoted": rejected_candidates,
        "manual_visual_audit": {
            "status": args.manual_visual_audit,
            "scope": "input assets only; no renderer output was reviewed",
            "all_81_frame_sheets_present": True,
            "keyframe_comparison_sheet_present": True,
            "keyframe_row_frame_indices": list(KEYFRAMES),
            "observations_if_passed": (
                [
                    "source bone remains present in exact-original, codec-only, and sham arms",
                    "source bone is absent from its original support in removed and translated arms",
                    "translated arm contains the same source bone pixels shifted upward by 150 pixels",
                    "no translated bone pixel or intervention support intersects the SAM2 dog mask",
                    "the interpolated support is visually smooth but is not ground-truth background recovery",
                ]
                if args.manual_visual_audit == "PASS_INPUT_ASSET_QA_ONLY"
                else []
            ),
        },
        "claim_limits": {
            "renderer_inference_performed": False,
            "training_performed": False,
            "optimizer_updates": 0,
            "intervention_inputs_only": True,
            "scientific_result_claim_authorized": False,
            "manual_visual_review_required": True,
            "interpolation_is_not_ground_truth_background_recovery": True,
        },
    }
    manifest["outputs"] = collect_outputs(output_root)
    manifest["artifact_digest"] = canonical_json_sha256(manifest)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": (
                    "COMPLETE_INPUT_MATERIALIZATION_REVIEW_PENDING"
                    if args.manual_visual_audit == "PENDING"
                    else "COMPLETE_INPUT_MATERIALIZATION_AND_INPUT_ASSET_QA"
                ),
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "artifact_digest": manifest["artifact_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr.decode("utf-8", errors="replace"))
        raise

#!/usr/bin/env python3
"""Materialize and verify case01's preregistered nine-frame G0 annotations.

This program is deliberately local and non-generative.  It binds manually
reviewed head/mouth/safe-background boxes to the already reviewed Stage-0
dog/bone SAM2 masks, validates all geometry with the Python standard library,
and asks ffmpeg only to render review surfaces.  It never runs a renderer,
training loop, or optimizer.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "bernini-case01-g0-sparse-annotation-manifest-v1"
RECEIPT_SCHEMA_VERSION = "bernini-case01-g0-sparse-annotation-receipt-v1"
SPEC_SCHEMA_VERSION = "bernini-case01-g0-sparse-human-annotation-spec-v1"
EXPECTED_CASE_ID = "case01"
EXPECTED_IID = "288545b9c031491a"
EXPECTED_SOURCE_SHA256 = (
    "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
)
EXPECTED_STAGE0_RECEIPT_SHA256 = (
    "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
)
EXPECTED_STAGE0_MANUAL_REVIEW_SHA256 = (
    "3f4e407925e4077827acad7499ac33536a7b855bdeeabb027af156b3a6961a4b"
)
EXPECTED_SPEC_SHA256 = (
    "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020"
)
EXPECTED_FRAMES = tuple(range(0, 81, 10))
EXPECTED_IMAGE_SIZE = (704, 736)
OVERLAY_SHEET_NAME = "overlay_sparse_0_10_20_30_40_50_60_70_80_3x3.png"


class SparseAnnotationError(RuntimeError):
    """Fail-closed validation error."""


def _fail(message: str) -> None:
    raise SparseAnnotationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read JSON {path}: {exc}")


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _require_plain_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file() or path.is_symlink():
        _fail(f"{label} is not a plain regular file: {path}")


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        _fail(f"path is outside repository root: {path}")


def _closed(mapping: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    actual = set(mapping)
    if actual != expected:
        _fail(
            f"{label} key closure differs: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _read_png_8bit_gray(path: Path) -> tuple[int, int, list[bytes]]:
    """Decode the Stage-0 binary masks without optional image packages."""

    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        _fail(f"invalid PNG signature: {path}")
    pos = 8
    idat = bytearray()
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    saw_iend = False
    while pos < len(payload):
        if pos + 12 > len(payload):
            _fail(f"truncated PNG chunk header: {path}")
        length = struct.unpack(">I", payload[pos : pos + 4])[0]
        kind = payload[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(payload):
            _fail(f"truncated PNG chunk: {path}")
        data = payload[pos + 8 : pos + 8 + length]
        expected_crc = struct.unpack(">I", payload[pos + 8 + length : end])[0]
        actual_crc = binascii.crc32(kind + data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            _fail(f"PNG CRC mismatch in {kind!r}: {path}")
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            idat.extend(data)
        elif kind == b"IEND":
            saw_iend = True
        pos = end
    if ihdr is None or not saw_iend:
        _fail(f"PNG lacks IHDR/IEND: {path}")
    width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
    if (bit_depth, color_type, compression, filter_method, interlace) != (8, 0, 0, 0, 0):
        _fail(f"mask must be non-interlaced 8-bit grayscale PNG: {path}")
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        _fail(f"cannot decompress PNG {path}: {exc}")
    stride = width
    expected_length = height * (stride + 1)
    if len(raw) != expected_length:
        _fail(f"PNG decompressed length differs for {path}")
    rows: list[bytes] = []
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        current = bytearray(raw[offset : offset + stride])
        offset += stride
        for x in range(stride):
            left = current[x - 1] if x else 0
            above = previous[x]
            upper_left = previous[x - 1] if x else 0
            if filter_type == 1:
                current[x] = (current[x] + left) & 0xFF
            elif filter_type == 2:
                current[x] = (current[x] + above) & 0xFF
            elif filter_type == 3:
                current[x] = (current[x] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                current[x] = (
                    current[x] + _paeth(left, above, upper_left)
                ) & 0xFF
            elif filter_type != 0:
                _fail(f"unsupported PNG filter {filter_type}: {path}")
        if set(current) - {0, 255}:
            _fail(f"mask is not binary 0/255: {path}")
        rows.append(bytes(current))
        previous = current
    return width, height, rows


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()[:24]
    if len(payload) != 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        _fail(f"invalid rendered PNG: {path}")
    if payload[12:16] != b"IHDR":
        _fail(f"rendered PNG lacks leading IHDR: {path}")
    return struct.unpack(">II", payload[16:24])


def _mask_stats(rows: Sequence[bytes]) -> dict[str, Any]:
    xs: list[int] = []
    ys: list[int] = []
    area = 0
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            if value:
                xs.append(x)
                ys.append(y)
                area += 1
    if not area:
        return {"area": 0, "bbox_xyxy": None}
    return {
        "area": area,
        "bbox_xyxy": [min(xs), min(ys), max(xs) + 1, max(ys) + 1],
    }


def _validate_box(value: Any, width: int, height: int, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        _fail(f"{label} must be a four-integer half-open box")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        _fail(f"{label} is out of bounds: {value}")
    return list(value)


def _box_contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _mask_hits(rows: Sequence[bytes], box: Sequence[int]) -> int:
    x0, y0, x1, y1 = box
    return sum(value != 0 for row in rows[y0:y1] for value in row[x0:x1])


def _mask_fraction(rows: Sequence[bytes], box: Sequence[int]) -> float:
    x0, y0, x1, y1 = box
    return _mask_hits(rows, box) / ((x1 - x0) * (y1 - y0))


def _expanded_box(box: Sequence[int], pixels: int, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        max(0, x0 - pixels),
        max(0, y0 - pixels),
        min(width, x1 + pixels),
        min(height, y1 + pixels),
    ]


def _load_and_validate_inputs(
    repo_root: Path, spec_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_plain_file(spec_path, "annotation spec")
    if _sha256(spec_path) != EXPECTED_SPEC_SHA256:
        _fail("annotation spec SHA-256 differs from frozen v1 authority")
    spec = _read_json(spec_path)
    if not isinstance(spec, dict) or spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        _fail("annotation spec schema differs")
    if spec.get("case_id") != EXPECTED_CASE_ID or spec.get("iid") != EXPECTED_IID:
        _fail("annotation spec case identity differs")
    if spec.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        _fail("annotation spec source digest differs")
    if tuple(spec.get("image_size_wh", [])) != EXPECTED_IMAGE_SIZE:
        _fail("annotation spec image dimensions differ")
    if tuple(spec.get("frame_indices", [])) != EXPECTED_FRAMES:
        _fail("annotation spec sparse frame schedule differs")
    stage0 = spec.get("stage0_authority")
    if not isinstance(stage0, dict):
        _fail("stage0 authority is missing")
    stage0_root = repo_root / str(stage0.get("root", ""))
    receipt_path = repo_root / str(stage0.get("receipt", ""))
    review_path = repo_root / str(stage0.get("manual_review", ""))
    geometry_path = stage0_root / "geometry.json"
    for path, label in (
        (receipt_path, "Stage-0 receipt"),
        (review_path, "Stage-0 manual review"),
        (geometry_path, "Stage-0 geometry"),
        (stage0_root / "overlay.mp4", "Stage-0 overlay video"),
    ):
        _require_plain_file(path, label)
    if _sha256(receipt_path) != EXPECTED_STAGE0_RECEIPT_SHA256:
        _fail("Stage-0 receipt digest differs")
    if _sha256(review_path) != EXPECTED_STAGE0_MANUAL_REVIEW_SHA256:
        _fail("Stage-0 manual review digest differs")
    if stage0.get("receipt_sha256") != EXPECTED_STAGE0_RECEIPT_SHA256:
        _fail("spec Stage-0 receipt pin differs")
    if stage0.get("manual_review_sha256") != EXPECTED_STAGE0_MANUAL_REVIEW_SHA256:
        _fail("spec Stage-0 manual-review pin differs")
    receipt = _read_json(receipt_path)
    review = _read_json(review_path)
    geometry = _read_json(geometry_path)
    if receipt.get("iid") != EXPECTED_IID:
        _fail("Stage-0 receipt IID differs")
    if receipt.get("source", {}).get("sha256") != EXPECTED_SOURCE_SHA256:
        _fail("Stage-0 source digest differs")
    if receipt.get("status") != "COMPLETE_STAGE0_MASKLET_DIAGNOSTIC":
        _fail("Stage-0 receipt is not complete")
    if receipt.get("diagnostic_gate", {}).get("automatic_geometry_gate_pass") is not True:
        _fail("Stage-0 geometry gate did not pass")
    if review.get("joint_tracking_subgate") != "PASS":
        _fail("Stage-0 primary tracking review did not pass")
    if review.get("frame_count_reviewed") != 81:
        _fail("Stage-0 review did not cover all 81 frames")
    return spec, stage0_root, receipt, review, geometry


def _build_manifest(
    repo_root: Path,
    spec_path: Path,
    materializer_path: Path,
) -> dict[str, Any]:
    spec, stage0_root, receipt, review, geometry = _load_and_validate_inputs(
        repo_root, spec_path
    )
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        _fail("Stage-0 output ledger is missing")
    output_digests = {
        str(item["path"]): str(item["sha256"])
        for item in outputs
        if isinstance(item, dict) and "path" in item and "sha256" in item
    }
    annotations = spec.get("sparse_annotations")
    if not isinstance(annotations, list) or len(annotations) != len(EXPECTED_FRAMES):
        _fail("sparse annotation count differs")
    annotation_by_frame = {entry.get("frame_index"): entry for entry in annotations}
    if tuple(annotation_by_frame) != EXPECTED_FRAMES:
        _fail("sparse annotation order/schedule differs")
    width, height = EXPECTED_IMAGE_SIZE
    safe_box = _validate_box(
        spec.get("safe_background_policy", {}).get("bbox_xyxy"),
        width,
        height,
        "safe-background box",
    )
    expansion = spec.get("support_context_expansion_pixels")
    if isinstance(expansion, bool) or not isinstance(expansion, int) or expansion != 8:
        _fail("bone support context expansion must remain exactly 8 pixels")
    dog_geometry = geometry.get("objects", {}).get("dog")
    bone_geometry = geometry.get("objects", {}).get("bone")
    if not isinstance(dog_geometry, list) or len(dog_geometry) != 81:
        _fail("dog geometry must contain exactly 81 frames")
    if not isinstance(bone_geometry, list) or len(bone_geometry) != 81:
        _fail("bone geometry must contain exactly 81 frames")
    frame_records: list[dict[str, Any]] = []
    minimum_head_dog_fraction = 1.0
    minimum_mouth_dog_fraction = 1.0
    for frame_index in EXPECTED_FRAMES:
        annotation = annotation_by_frame[frame_index]
        if not isinstance(annotation, dict):
            _fail(f"frame {frame_index} annotation must be an object")
        _closed(
            annotation,
            (
                "frame_index",
                "head_bbox_xyxy",
                "mouth_bbox_xyxy",
                "head_confidence",
                "mouth_confidence",
                "ambiguity",
            ),
            f"frame {frame_index} annotation",
        )
        head_box = _validate_box(
            annotation["head_bbox_xyxy"], width, height, f"frame {frame_index} head"
        )
        mouth_box = _validate_box(
            annotation["mouth_bbox_xyxy"], width, height, f"frame {frame_index} mouth"
        )
        if not _box_contains(head_box, mouth_box):
            _fail(f"frame {frame_index} mouth box is not contained by head box")
        frame_path = stage0_root / "_decoded_jpeg_frames" / f"{frame_index:05d}.jpg"
        dog_path = stage0_root / "masks" / "dog" / f"{frame_index:05d}.png"
        bone_path = stage0_root / "masks" / "bone" / f"{frame_index:05d}.png"
        for path, label in (
            (frame_path, "source frame"),
            (dog_path, "dog mask"),
            (bone_path, "bone mask"),
        ):
            _require_plain_file(path, f"frame {frame_index} {label}")
        dog_rel = f"masks/dog/{frame_index:05d}.png"
        bone_rel = f"masks/bone/{frame_index:05d}.png"
        dog_sha = _sha256(dog_path)
        bone_sha = _sha256(bone_path)
        if output_digests.get(dog_rel) != dog_sha:
            _fail(f"frame {frame_index} dog mask digest differs from receipt")
        if output_digests.get(bone_rel) != bone_sha:
            _fail(f"frame {frame_index} bone mask digest differs from receipt")
        dog_width, dog_height, dog_rows = _read_png_8bit_gray(dog_path)
        bone_width, bone_height, bone_rows = _read_png_8bit_gray(bone_path)
        if (dog_width, dog_height) != EXPECTED_IMAGE_SIZE:
            _fail(f"frame {frame_index} dog mask dimensions differ")
        if (bone_width, bone_height) != EXPECTED_IMAGE_SIZE:
            _fail(f"frame {frame_index} bone mask dimensions differ")
        dog_stats = _mask_stats(dog_rows)
        bone_stats = _mask_stats(bone_rows)
        expected_dog = dog_geometry[frame_index]
        expected_bone = bone_geometry[frame_index]
        if dog_stats["area"] != expected_dog.get("area"):
            _fail(f"frame {frame_index} dog area differs from geometry authority")
        if dog_stats["bbox_xyxy"] != expected_dog.get("bbox_xyxy"):
            _fail(f"frame {frame_index} dog bbox differs from geometry authority")
        if bone_stats["area"] != expected_bone.get("area"):
            _fail(f"frame {frame_index} bone area differs from geometry authority")
        if bone_stats["bbox_xyxy"] != expected_bone.get("bbox_xyxy"):
            _fail(f"frame {frame_index} bone bbox differs from geometry authority")
        head_fraction = _mask_fraction(dog_rows, head_box)
        mouth_fraction = _mask_fraction(dog_rows, mouth_box)
        minimum_head_dog_fraction = min(minimum_head_dog_fraction, head_fraction)
        minimum_mouth_dog_fraction = min(minimum_mouth_dog_fraction, mouth_fraction)
        if head_fraction < 0.70:
            _fail(f"frame {frame_index} head box has insufficient dog-mask support")
        if mouth_fraction < 0.70:
            _fail(f"frame {frame_index} mouth box has insufficient dog-mask support")
        if _mask_hits(bone_rows, head_box) or _mask_hits(bone_rows, mouth_box):
            _fail(f"frame {frame_index} head/mouth overlaps source bone")
        if _mask_hits(dog_rows, safe_box) or _mask_hits(bone_rows, safe_box):
            _fail(f"frame {frame_index} safe-background box overlaps dog/bone")
        bone_bbox = bone_stats["bbox_xyxy"]
        if bone_bbox is None:
            _fail(f"frame {frame_index} bone is unexpectedly invisible")
        support_box = _expanded_box(bone_bbox, expansion, width, height)
        frame_records.append(
            {
                "frame_index": frame_index,
                "source_frame": {
                    "path": _repo_relative(frame_path, repo_root),
                    "sha256": _sha256(frame_path),
                },
                "annotations": {
                    "dog#1": {
                        "geometry_type": "binary_mask_reference",
                        "path": _repo_relative(dog_path, repo_root),
                        "sha256": dog_sha,
                        "area_pixels": dog_stats["area"],
                        "bbox_xyxy": dog_stats["bbox_xyxy"],
                        "visibility": "visible",
                        "authority": "stage0_sam2_r2_plus_all81_manual_tracking_review",
                    },
                    "dog#1.head": {
                        "geometry_type": "reviewed_box",
                        "bbox_xyxy": head_box,
                        "confidence": annotation["head_confidence"],
                        "dog_mask_coverage_fraction": round(head_fraction, 6),
                    },
                    "dog#1.mouth": {
                        "geometry_type": "reviewed_box",
                        "bbox_xyxy": mouth_box,
                        "confidence": annotation["mouth_confidence"],
                        "dog_mask_coverage_fraction": round(mouth_fraction, 6),
                        "ambiguity": annotation["ambiguity"],
                    },
                    "bone#1": {
                        "geometry_type": "binary_mask_reference",
                        "path": _repo_relative(bone_path, repo_root),
                        "sha256": bone_sha,
                        "area_pixels": bone_stats["area"],
                        "bbox_xyxy": bone_bbox,
                        "visibility": "visible",
                        "authority": "stage0_sam2_r2_plus_all81_manual_tracking_review",
                        "instruction_role": "patient",
                    },
                    "bone#1.support": {
                        "geometry_type": "derived_ground_context_box",
                        "bbox_xyxy": support_box,
                        "relation": "supported_by(bone#1,ground#1)",
                        "derivation": "bone_mask_bbox_expanded_by_8_pixels",
                        "confidence": "medium",
                        "ambiguity": (
                            "Overhead RGB does not reveal the occluded physical contact "
                            "surface; this is a reviewed ground-context proxy, not target RGB."
                        ),
                    },
                    "safe-background#1": {
                        "geometry_type": "reviewed_box",
                        "bbox_xyxy": safe_box,
                        "confidence": spec["safe_background_policy"]["confidence"],
                        "selection_reason": spec["safe_background_policy"][
                            "selection_reason"
                        ],
                    },
                },
                "primary_review_status": "PASS",
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": EXPECTED_CASE_ID,
        "iid": EXPECTED_IID,
        "source": {
            "sha256": EXPECTED_SOURCE_SHA256,
            "frame_count": 81,
            "fps": 25.0,
            "width": width,
            "height": height,
            "heldout_canary": True,
        },
        "coordinate_convention": spec["coordinate_convention"],
        "frame_schedule": list(EXPECTED_FRAMES),
        "authority": {
            "annotation_spec": {
                "path": _repo_relative(spec_path, repo_root),
                "sha256": _sha256(spec_path),
            },
            "materializer": {
                "path": _repo_relative(materializer_path, repo_root),
                "sha256": _sha256(materializer_path),
            },
            "stage0_receipt": {
                "path": spec["stage0_authority"]["receipt"],
                "sha256": EXPECTED_STAGE0_RECEIPT_SHA256,
                "canonical_receipt_digest": receipt["receipt_digest"],
            },
            "stage0_manual_review": {
                "path": spec["stage0_authority"]["manual_review"],
                "sha256": EXPECTED_STAGE0_MANUAL_REVIEW_SHA256,
                "joint_tracking_subgate": review["joint_tracking_subgate"],
            },
        },
        "annotation_definitions": spec["label_contract"],
        "overlay_legend": {
            "dog#1": "cyan fill inherited from Stage-0 overlay",
            "bone#1": "orange-red fill inherited from Stage-0 overlay",
            "dog#1.head": "lime box",
            "dog#1.mouth": "magenta box",
            "bone#1.support": "yellow box",
            "safe-background#1": "blue box",
        },
        "frames": frame_records,
        "validation": {
            "exact_sparse_frame_count": 9,
            "all_dog_masks_bound_to_stage0_receipt": True,
            "all_bone_masks_bound_to_stage0_receipt": True,
            "all_head_and_mouth_boxes_supported_by_dog_mask": True,
            "minimum_head_box_dog_mask_coverage_fraction": round(
                minimum_head_dog_fraction, 6
            ),
            "minimum_mouth_box_dog_mask_coverage_fraction": round(
                minimum_mouth_dog_fraction, 6
            ),
            "all_safe_background_boxes_disjoint_from_dog_and_bone_masks": True,
            "all_bones_visible": True,
            "patient_identity_primary_review": "PASS_bone#1",
            "annotation_half_primary_review": "PASS",
            "independent_second_review": "PENDING",
            "full_g0": "PENDING_INDEPENDENT_SECOND_REVIEW",
        },
        "primary_review": spec["primary_review"],
        "claim_limits": spec["claim_limits"],
    }
    manifest["manifest_digest"] = _canonical_digest(manifest)
    return manifest


def _drawbox(box: Sequence[int], color: str, thickness: int) -> str:
    x0, y0, x1, y1 = box
    return (
        f"drawbox=x={x0}:y={y0}:w={x1 - x0}:h={y1 - y0}:"
        f"color={color}:t={thickness}"
    )


def _run(command: Sequence[str], label: str) -> None:
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"{label} failed: {exc}")


def _render_overlays(
    manifest: Mapping[str, Any], stage0_root: Path, output_root: Path, ffmpeg: str
) -> list[Path]:
    overlay_video = stage0_root / "overlay.mp4"
    overlay_root = output_root / "overlays"
    overlay_root.mkdir(mode=0o755)
    rendered: list[Path] = []
    for record in manifest["frames"]:
        frame_index = record["frame_index"]
        labels = record["annotations"]
        filters = [
            f"select=eq(n\\,{frame_index})",
            _drawbox(labels["bone#1.support"]["bbox_xyxy"], "0xFFFF00@0.95", 3),
            _drawbox(labels["dog#1.head"]["bbox_xyxy"], "0x00FF00@0.95", 4),
            _drawbox(labels["dog#1.mouth"]["bbox_xyxy"], "0xFF00FF@0.95", 4),
            _drawbox(labels["safe-background#1"]["bbox_xyxy"], "0x0066FF@0.95", 4),
        ]
        destination = overlay_root / f"{frame_index:05d}.png"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(overlay_video),
            "-vf",
            ",".join(filters),
            "-frames:v",
            "1",
            "-y",
            str(destination),
        ]
        _run(command, f"render sparse overlay frame {frame_index}")
        if _png_dimensions(destination) != EXPECTED_IMAGE_SIZE:
            _fail(f"rendered overlay dimensions differ: {destination}")
        rendered.append(destination)
    sheet_path = output_root / OVERLAY_SHEET_NAME
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    for path in rendered:
        command.extend(["-i", str(path)])
    scale_filters = [
        f"[{index}:v]scale=352:368:flags=lanczos[s{index}]"
        for index in range(len(rendered))
    ]
    scale_filters.extend(
        [
            "[s0][s1][s2]hstack=inputs=3[row0]",
            "[s3][s4][s5]hstack=inputs=3[row1]",
            "[s6][s7][s8]hstack=inputs=3[row2]",
            "[row0][row1][row2]vstack=inputs=3[out]",
        ]
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(scale_filters),
            "-map",
            "[out]",
            "-frames:v",
            "1",
            "-y",
            str(sheet_path),
        ]
    )
    _run(command, "render sparse overlay sheet")
    if _png_dimensions(sheet_path) != (1056, 1104):
        _fail("overlay sheet dimensions differ")
    return rendered + [sheet_path]


def _receipt_without_digest(
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
    rendered: Sequence[Path],
    ffmpeg: str,
) -> dict[str, Any]:
    try:
        ffmpeg_version = subprocess.check_output(
            [ffmpeg, "-version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError) as exc:
        _fail(f"cannot identify ffmpeg: {exc}")
    manifest_path = output_root / "manifest.json"
    output_records = [
        {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in [manifest_path, *rendered]
    ]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETE_PRIMARY_SPARSE_ANNOTATION_HALF",
        "case_id": EXPECTED_CASE_ID,
        "iid": EXPECTED_IID,
        "manifest_digest": manifest["manifest_digest"],
        "inputs": {
            "annotation_spec": {
                "path": _repo_relative(spec_path, repo_root),
                "sha256": _sha256(spec_path),
            },
            "stage0_receipt_sha256": EXPECTED_STAGE0_RECEIPT_SHA256,
            "stage0_manual_review_sha256": EXPECTED_STAGE0_MANUAL_REVIEW_SHA256,
            "source_sha256": EXPECTED_SOURCE_SHA256,
        },
        "render_tool": ffmpeg_version,
        "outputs": output_records,
        "validation": manifest["validation"],
        "claim_limits": {
            "annotation_half_primary_review_pass": True,
            "independent_second_review_completed": False,
            "full_g0_authorized": False,
            "renderer_inference_performed": False,
            "training_performed": False,
            "optimizer_updates": 0,
            "scientific_result_claim_authorized": False,
        },
    }


def _verify_digest_field(value: Mapping[str, Any], field: str, label: str) -> None:
    if field not in value:
        _fail(f"{label} lacks {field}")
    digest = value[field]
    unsigned = dict(value)
    unsigned.pop(field)
    if digest != _canonical_digest(unsigned):
        _fail(f"{label} canonical digest differs")


def verify_output(
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    materializer_path: Path | None = None,
) -> dict[str, Any]:
    materializer_path = materializer_path or Path(__file__).resolve()
    manifest_path = output_root / "manifest.json"
    receipt_path = output_root / "receipt.json"
    _require_plain_file(manifest_path, "sparse annotation manifest")
    _require_plain_file(receipt_path, "sparse annotation receipt")
    manifest = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        _fail("manifest schema differs")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        _fail("receipt schema differs")
    _verify_digest_field(manifest, "manifest_digest", "manifest")
    _verify_digest_field(receipt, "receipt_digest", "receipt")
    expected_manifest = _build_manifest(repo_root, spec_path, materializer_path)
    if manifest != expected_manifest:
        _fail("materialized manifest differs from current frozen inputs")
    if receipt.get("manifest_digest") != manifest["manifest_digest"]:
        _fail("receipt/manifest digest link differs")
    output_records = receipt.get("outputs")
    if not isinstance(output_records, list):
        _fail("receipt output ledger is missing")
    expected_paths = {
        "manifest.json",
        *(f"overlays/{frame_index:05d}.png" for frame_index in EXPECTED_FRAMES),
        OVERLAY_SHEET_NAME,
    }
    actual_paths: set[str] = set()
    for item in output_records:
        if not isinstance(item, dict):
            _fail("receipt output record is not an object")
        _closed(item, ("path", "sha256", "size"), "receipt output record")
        relative = str(item["path"])
        if relative in actual_paths:
            _fail(f"duplicate output ledger path: {relative}")
        actual_paths.add(relative)
        candidate = output_root / relative
        _require_plain_file(candidate, f"receipt output {relative}")
        if _sha256(candidate) != item["sha256"]:
            _fail(f"output digest differs: {relative}")
        if candidate.stat().st_size != item["size"]:
            _fail(f"output size differs: {relative}")
    if actual_paths != expected_paths:
        _fail("receipt output path closure differs")
    if _png_dimensions(output_root / OVERLAY_SHEET_NAME) != (1056, 1104):
        _fail("overlay sheet dimensions differ during verification")
    if receipt.get("status") != "COMPLETE_PRIMARY_SPARSE_ANNOTATION_HALF":
        _fail("receipt status differs")
    limits = receipt.get("claim_limits", {})
    if limits.get("full_g0_authorized") is not False:
        _fail("receipt improperly authorizes full G0")
    if limits.get("renderer_inference_performed") is not False:
        _fail("receipt improperly records renderer inference")
    if limits.get("training_performed") is not False or limits.get("optimizer_updates") != 0:
        _fail("receipt improperly records training")
    return receipt


def materialize(
    repo_root: Path,
    spec_path: Path,
    output_root: Path,
    ffmpeg: str,
    materializer_path: Path | None = None,
) -> dict[str, Any]:
    materializer_path = materializer_path or Path(__file__).resolve()
    if output_root.exists():
        _fail(f"fresh output root required; path already exists: {output_root}")
    ffmpeg_path = shutil.which(ffmpeg)
    if ffmpeg_path is None:
        _fail(f"ffmpeg executable not found: {ffmpeg}")
    manifest = _build_manifest(repo_root, spec_path, materializer_path)
    output_root.mkdir(parents=True, mode=0o755)
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    stage0_root = repo_root / _read_json(spec_path)["stage0_authority"]["root"]
    rendered = _render_overlays(manifest, stage0_root, output_root, ffmpeg_path)
    receipt = _receipt_without_digest(
        repo_root, spec_path, output_root, manifest, rendered, ffmpeg_path
    )
    receipt["receipt_digest"] = _canonical_digest(receipt)
    _write_json(output_root / "receipt.json", receipt)
    return verify_output(
        repo_root, spec_path, output_root, materializer_path=materializer_path
    )


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--spec",
        type=Path,
        default=repo_root
        / "methods/bernini_action_editing/assets/"
        "case01_288545b9c031491a_g0_sparse_annotations_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts/object_grounded_case01_0821_sparse_g0_v1",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    spec_path = args.spec.resolve()
    output_root = args.output_root.resolve()
    if args.verify_only:
        receipt = verify_output(repo_root, spec_path, output_root)
    else:
        receipt = materialize(repo_root, spec_path, output_root, args.ffmpeg)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an *unsigned* case01 bone/contact-shadow/halo review packet.

This local-only utility does not infer a clean plate and does not authorize a
VACE/GPU run.  It starts from the byte-pinned, manually reviewed SAM2 bone and
dog tracks, builds a deliberately conservative support *candidate*, and emits
native-resolution review surfaces plus two blank external-review templates.

The contact-shadow component is only a directional geometric apron.  Whether
it actually covers every source shadow/halo pixel is a human visual question
and is therefore always left PENDING by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence


SCHEMA = "bernini-case01-bone-contact-support-review-packet-v1"
CASE_ID = "case01"
IID = "288545b9c031491a"
WIDTH = 704
HEIGHT = 736
FPS = 25
FRAME_COUNT = 81
FRAME_PIXELS = WIDTH * HEIGHT
RGB_FRAME_BYTES = FRAME_PIXELS * 3

SOURCE_SHA256 = "5ed911f66fea3ed2000f507412da75adecb8099b26b71089d0fd2c0ac2982b18"
MASKLET_RECEIPT_SHA256 = "c9dc8567d4f147f943171d2d7abb55e52aadc685fbfc2f75fff4e837e2ab0b50"
MASKLET_MANUAL_REVIEW_SHA256 = "3f4e407925e4077827acad7499ac33536a7b855bdeeabb027af156b3a6961a4b"
MASKLET_GEOMETRY_SHA256 = "2a7daf54a86606002704e6436096a2f04c63260356b08fcd7a5d57d915876157"
SPARSE_SPEC_SHA256 = "e5185a1edd72fa8a1f2ece15e98c67d66e3fa65a2a9eb724bf06031c4d0e2020"
SPARSE_MANIFEST_SHA256 = "08079e1b7c35e04c49adee16ce47c42194aba2feab708e71a5799dbb39f9812f"
SPARSE_RECEIPT_SHA256 = "266743f9e5c370a64f35b1acde72c29aa3956eec69bb3ffc464c43fe66b0e096"
REVOKED_R4_MANIFEST_SHA256 = "0a62b74056f4be1ab17ed632d31068964aed27c607212f58c2a7d17b74becf5e"
REVOKED_R4_SUPPORT_SHA256 = "83818847c61b506370edc9a4a6cae8b9fe2bb06681f5425e40d9fb6e850fd554"

SAFETY_HALO_RADIUS = 8
DOG_GUARD_RADIUS = 12
CONTACT_X_RADIUS = 12
CONTACT_Y_UP = 4
CONTACT_Y_DOWN = 16
REVOKED_R4_RADIUS = 3
CROP_PADDING = 32


class SupportPacketError(RuntimeError):
    """An input authority or unsigned-packet invariant was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def require_regular(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise SupportPacketError(f"{label} must be one regular non-symlink file")
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise SupportPacketError(f"{label} SHA-256 differs: {actual}")
    return path


def load_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require_regular(path, expected_sha256, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise SupportPacketError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise SupportPacketError(f"{label} must contain one JSON object")
    return value


def validate_authorities(
    *,
    source_video: Path,
    masklet_root: Path,
    sparse_spec_path: Path,
    sparse_manifest_path: Path,
    sparse_receipt_path: Path,
    revoked_r4_root: Path,
) -> dict[str, Any]:
    """Validate every named input and return parsed authority records.

    The revoked r4 files are negative evidence only.  Their pixels never seed
    the new candidate.
    """

    require_regular(source_video, SOURCE_SHA256, "source video")
    receipt_path = masklet_root / "receipt.json"
    manual_path = masklet_root / "manual_review_v1.json"
    geometry_path = masklet_root / "geometry.json"
    receipt = load_json(receipt_path, MASKLET_RECEIPT_SHA256, "masklet receipt")
    manual = load_json(
        manual_path, MASKLET_MANUAL_REVIEW_SHA256, "masklet manual review"
    )
    geometry = load_json(geometry_path, MASKLET_GEOMETRY_SHA256, "masklet geometry")
    sparse_spec = load_json(sparse_spec_path, SPARSE_SPEC_SHA256, "sparse spec")
    sparse_manifest = load_json(
        sparse_manifest_path, SPARSE_MANIFEST_SHA256, "sparse manifest"
    )
    sparse_receipt = load_json(
        sparse_receipt_path, SPARSE_RECEIPT_SHA256, "sparse receipt"
    )
    revoked_manifest_path = revoked_r4_root / "manifest.json"
    revoked_support_path = revoked_r4_root / "tubes/removal_support_dilate3.mkv"
    revoked_manifest = load_json(
        revoked_manifest_path, REVOKED_R4_MANIFEST_SHA256, "revoked r4 manifest"
    )
    require_regular(
        revoked_support_path, REVOKED_R4_SUPPORT_SHA256, "revoked r4 support tube"
    )

    source = receipt.get("source")
    limits = receipt.get("claim_limits")
    if (
        receipt.get("schema_version")
        != "bernini-case01-oracle-sam2-masklets-receipt-v1"
        or receipt.get("case_id") != CASE_ID
        or receipt.get("iid") != IID
        or receipt.get("status") != "COMPLETE_STAGE0_MASKLET_DIAGNOSTIC"
        or type(source) is not dict
        or source.get("sha256") != SOURCE_SHA256
        or source.get("frame_count") != FRAME_COUNT
        or source.get("width") != WIDTH
        or source.get("height") != HEIGHT
        or source.get("fps") != 25.0
        or type(limits) is not dict
        or limits.get("manual_masklet_review_required") is not True
        or limits.get("single_case_scientific_claim_authorized") is not False
    ):
        raise SupportPacketError("masklet receipt contract differs")
    if (
        manual.get("schema_version")
        != "bernini-case01-sam2-masklet-manual-review-v1"
        or manual.get("source_receipt_sha256") != MASKLET_RECEIPT_SHA256
        or manual.get("frame_count_reviewed") != FRAME_COUNT
        or manual.get("joint_tracking_subgate") != "PASS"
        or manual.get("bone_track", {}).get("status") != "PASS"
        or manual.get("dog_track", {}).get("status") != "PASS"
    ):
        raise SupportPacketError("masklet manual-review contract differs")
    if geometry.get("schema_version") != "bernini-case01-oracle-sam2-masklet-geometry-v1":
        raise SupportPacketError("masklet geometry schema differs")

    if (
        sparse_spec.get("schema_version")
        != "bernini-case01-g0-sparse-human-annotation-spec-v1"
        or sparse_spec.get("source_sha256") != SOURCE_SHA256
        or sparse_spec.get("image_size_wh") != [WIDTH, HEIGHT]
        or sparse_spec.get("support_context_expansion_pixels")
        != SAFETY_HALO_RADIUS
        or sparse_spec.get("primary_review", {}).get(
            "independent_second_review_completed"
        )
        is not False
        or sparse_spec.get("claim_limits", {}).get("full_g0_authorized") is not False
    ):
        raise SupportPacketError("sparse annotation contract differs")
    if (
        sparse_receipt.get("schema_version")
        != "bernini-case01-g0-sparse-annotation-receipt-v1"
        or sparse_receipt.get("status")
        != "COMPLETE_PRIMARY_SPARSE_ANNOTATION_HALF"
        or sparse_receipt.get("validation", {}).get("exact_sparse_frame_count") != 9
        or sparse_receipt.get("validation", {}).get("independent_second_review")
        != "PENDING"
        or sparse_receipt.get("claim_limits", {}).get("full_g0_authorized")
        is not False
    ):
        raise SupportPacketError("sparse receipt must remain second-review PENDING")
    if sparse_manifest.get("claim_limits", {}).get("full_g0_authorized") is not False:
        raise SupportPacketError("sparse manifest unexpectedly authorizes full G0")

    if (
        revoked_manifest.get("schema_version")
        != "bernini-case01-matched-bone-interventions-v1"
        or revoked_manifest.get("case_id") != CASE_ID
        or revoked_manifest.get("iid") != IID
        or revoked_manifest.get("intervention_recipe", {})
        .get("bone_removed", {})
        .get("support_expansion")
        != "three deterministic 8-neighbour dilation passes (Chebyshev radius 3)"
    ):
        raise SupportPacketError("revoked r4 negative-evidence contract differs")

    outputs = receipt.get("outputs")
    if type(outputs) is not list:
        raise SupportPacketError("masklet receipt outputs differ")
    output_by_path: dict[str, dict[str, Any]] = {}
    for row in outputs:
        if type(row) is not dict or type(row.get("path")) is not str:
            raise SupportPacketError("masklet receipt output row differs")
        if row["path"] in output_by_path:
            raise SupportPacketError("duplicate masklet receipt output path")
        output_by_path[row["path"]] = row
    for name in ("bone", "dog"):
        for frame_index in range(FRAME_COUNT):
            relative = f"masks/{name}/{frame_index:05d}.png"
            row = output_by_path.get(relative)
            if type(row) is not dict or type(row.get("sha256")) is not str:
                raise SupportPacketError(f"missing receipt pin for {relative}")
            path = masklet_root / relative
            require_regular(path, row["sha256"], relative)
            if path.stat().st_size != row.get("size"):
                raise SupportPacketError(f"receipt size differs for {relative}")

    return {
        "receipt": receipt,
        "manual": manual,
        "geometry": geometry,
        "sparse_spec": sparse_spec,
        "sparse_manifest": sparse_manifest,
        "sparse_receipt": sparse_receipt,
        "revoked_manifest": revoked_manifest,
        "revoked_support_path": revoked_support_path,
        "output_by_path": output_by_path,
        "paths": {
            "source_video": source_video,
            "masklet_receipt": receipt_path,
            "masklet_manual_review": manual_path,
            "masklet_geometry": geometry_path,
            "sparse_spec": sparse_spec_path,
            "sparse_manifest": sparse_manifest_path,
            "sparse_receipt": sparse_receipt_path,
            "revoked_r4_manifest": revoked_manifest_path,
            "revoked_r4_support": revoked_support_path,
        },
    }


def run_bytes(command: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace")[-2000:]
        raise SupportPacketError(f"media command failed: {detail}") from error
    return result.stdout


def ffmpeg_identity(ffmpeg: Path) -> dict[str, Any]:
    if not ffmpeg.is_file() or ffmpeg.is_symlink():
        raise SupportPacketError("ffmpeg must be one regular non-symlink executable")
    version = run_bytes((str(ffmpeg), "-version")).decode(
        "utf-8", errors="replace"
    ).splitlines()[0]
    return {
        "path": str(ffmpeg),
        "sha256": sha256_path(ffmpeg),
        "size": ffmpeg.stat().st_size,
        "version_line": version,
    }


def decode_source_rgb(ffmpeg: Path, source: Path) -> bytes:
    payload = run_bytes(
        (
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
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
        )
    )
    expected = FRAME_COUNT * RGB_FRAME_BYTES
    if len(payload) != expected:
        raise SupportPacketError(f"source decode length differs: {len(payload)} != {expected}")
    return payload


def decode_revoked_support(ffmpeg: Path, path: Path) -> bytes:
    payload = run_bytes(
        (
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        )
    )
    expected = FRAME_COUNT * FRAME_PIXELS
    if len(payload) != expected:
        raise SupportPacketError("revoked r4 support decode length differs")
    if not set(payload).issubset({0, 255}):
        raise SupportPacketError("revoked r4 support is not binary after decode")
    return payload


def dilate_square(mask: Any, radius: int) -> Any:
    import cv2
    import numpy as np

    if type(radius) is not int or radius < 0:
        raise SupportPacketError("dilation radius must be a nonnegative builtin int")
    value = np.asarray(mask, dtype=np.bool_)
    if value.shape != (HEIGHT, WIDTH):
        raise SupportPacketError("mask geometry differs")
    if radius == 0:
        return value.copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(value.astype(np.uint8), kernel, iterations=1).astype(bool)


def contact_shadow_apron(mask: Any) -> Any:
    """Return a directional ground apron, not an observed shadow mask."""

    import cv2
    import numpy as np

    value = np.asarray(mask, dtype=np.bool_)
    kernel = np.ones(
        (CONTACT_Y_UP + CONTACT_Y_DOWN + 1, 2 * CONTACT_X_RADIUS + 1),
        dtype=np.uint8,
    )
    # A source pixel affects destination offsets [anchor-(size-1), anchor].
    # The y anchor at 16 therefore gives offsets y=[-4,+16].
    anchor = (CONTACT_X_RADIUS, CONTACT_Y_DOWN)
    return cv2.dilate(
        value.astype(np.uint8), kernel, anchor=anchor, iterations=1
    ).astype(bool)


def build_support_components(bone: Any, dog: Any) -> dict[str, Any]:
    import numpy as np

    bone = np.asarray(bone, dtype=np.bool_)
    dog = np.asarray(dog, dtype=np.bool_)
    if bone.shape != (HEIGHT, WIDTH) or dog.shape != (HEIGHT, WIDTH):
        raise SupportPacketError("bone/dog geometry differs")
    if not bone.any() or not dog.any() or np.logical_and(bone, dog).any():
        raise SupportPacketError("bone/dog masks are empty or intersect")
    halo_inclusive = dilate_square(bone, SAFETY_HALO_RADIUS)
    halo_ring = np.logical_and(halo_inclusive, ~bone)
    apron_inclusive = contact_shadow_apron(bone)
    apron_only = np.logical_and(apron_inclusive, ~halo_inclusive)
    candidate = np.logical_or(halo_inclusive, apron_inclusive)
    dog_guard = dilate_square(dog, DOG_GUARD_RADIUS)
    if np.logical_and(candidate, dog).any():
        raise SupportPacketError("candidate intersects exact dog mask")
    if np.logical_and(candidate, dog_guard).any():
        raise SupportPacketError("candidate intersects protected dog guard")
    if not np.all(candidate[halo_inclusive]):
        raise SupportPacketError("candidate does not contain the full 8px safety halo")
    return {
        "bone_core": bone,
        "safety_halo8_ring": halo_ring,
        "contact_shadow_apron": apron_only,
        "candidate_support": candidate,
        "dog_guard12": dog_guard,
        "halo8_inclusive": halo_inclusive,
    }


def bbox_xyxy(mask: Any) -> list[int]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise SupportPacketError("cannot form bbox for empty mask")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def padded_bbox(box: Sequence[int], padding: int = CROP_PADDING) -> list[int]:
    x1, y1, x2, y2 = map(int, box)
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(WIDTH, x2 + padding),
        min(HEIGHT, y2 + padding),
    ]


def save_binary_png(path: Path, mask: Any) -> None:
    import cv2
    import numpy as np

    value = np.asarray(mask, dtype=np.bool_).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), value, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise SupportPacketError(f"could not write {path}")


def blend(canvas: Any, mask: Any, rgb: Sequence[int], alpha: float) -> None:
    import numpy as np

    selector = np.asarray(mask, dtype=np.bool_)
    color = np.asarray(rgb, dtype=np.float32)
    canvas[selector] = (1.0 - alpha) * canvas[selector] + alpha * color


def render_overlay(source_rgb: Any, parts: dict[str, Any], frame_index: int) -> Any:
    import cv2
    import numpy as np

    canvas = np.asarray(source_rgb, dtype=np.uint8).astype(np.float32)
    guard_ring = np.logical_and(parts["dog_guard12"], ~parts["candidate_support"])
    blend(canvas, guard_ring, (0, 210, 255), 0.08)
    blend(canvas, parts["contact_shadow_apron"], (90, 90, 255), 0.42)
    blend(canvas, parts["safety_halo8_ring"], (255, 215, 20), 0.48)
    blend(canvas, parts["bone_core"], (255, 45, 45), 0.52)
    rendered = np.clip(canvas, 0, 255).astype(np.uint8)
    box = bbox_xyxy(parts["candidate_support"])
    cv2.rectangle(rendered, (box[0], box[1]), (box[2] - 1, box[3] - 1), (255, 255, 255), 2)
    cv2.putText(
        rendered,
        f"UNSIGNED frame {frame_index:02d} | red bone | yellow halo8 | blue shadow-proxy",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        rendered,
        "CONTACT/SHADOW COVERAGE: PENDING TWO EXTERNAL REVIEWS",
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return rendered


def save_rgb_png(path: Path, rgb: Any) -> None:
    import cv2
    import numpy as np

    value = np.asarray(rgb, dtype=np.uint8)
    if value.ndim != 3 or value.shape[2] != 3:
        raise SupportPacketError("RGB image geometry differs")
    if not cv2.imwrite(
        str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 7]
    ):
        raise SupportPacketError(f"could not write {path}")


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_path(path),
        "size": path.stat().st_size,
    }


def output_tree_records(root: Path, excluded: Iterable[str] = ()) -> list[dict[str, Any]]:
    excluded_set = set(excluded)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SupportPacketError("output tree must not contain symlinks")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded_set:
            records.append(file_record(path, root))
    return records


def review_template(slot: int) -> dict[str, Any]:
    return {
        "schema_version": "bernini-case01-bone-contact-support-external-review-template-v1",
        "candidate_manifest_sha256": None,
        "reviewer_slot": slot,
        "reviewer_identity": None,
        "reviewer_affiliation_or_role": None,
        "reviewed_at_utc": None,
        "independence_attestation": None,
        "instructions": [
            "Copy this template outside the immutable candidate directory before filling it.",
            "Inspect the native 704x736 overlay and unscaled source/overlay crop for every frame.",
            "Reject or request edits if the original bone, contact shadow, halo, or adjacent ground is outside support.",
            "Reject if support touches the dog or would edit dog identity pixels.",
            "Do not infer PASS from geometry or from the other reviewer's decision.",
        ],
        "frames": [
            {
                "frame_index": index,
                "bone_coverage": "PENDING",
                "contact_shadow_coverage": "PENDING",
                "halo_and_adjacent_ground_coverage": "PENDING",
                "dog_and_guard_protection": "PENDING",
                "boundary_edit_requested": None,
                "notes": None,
                "decision": "PENDING",
            }
            for index in range(FRAME_COUNT)
        ],
        "all_81_native_frames_reviewed": False,
        "overall_decision": "PENDING",
        "signature_or_external_receipt": None,
        "claim_limits_acknowledged": False,
    }


def build_html(frame_rows: Sequence[dict[str, Any]]) -> str:
    cards = []
    for row in frame_rows:
        index = int(row["frame_index"])
        cards.append(
            f"""
<section class="card" id="frame-{index:05d}">
  <h2>Frame {index:02d} — PENDING</h2>
  <p>support area {row['areas']['candidate_support']} px; crop {html.escape(str(row['crop_bbox_xyxy']))}</p>
  <div class="pair">
    <figure><img src="crops/source_native/{index:05d}.png" loading="lazy"><figcaption>source crop, native pixels</figcaption></figure>
    <figure><img src="crops/overlay_native/{index:05d}.png" loading="lazy"><figcaption>candidate overlay crop, no scaling in file</figcaption></figure>
  </div>
  <p><a href="overlays/native/{index:05d}.png">open full native 704x736 overlay</a></p>
</section>"""
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>case01 unsigned support review</title>
<style>
body{{font:15px system-ui,sans-serif;background:#111;color:#eee;margin:24px}} a{{color:#8bd}}
.warning{{border:3px solid #f55;padding:16px;background:#311;position:sticky;top:0;z-index:2}}
.card{{border:1px solid #555;margin:18px 0;padding:12px;background:#1b1b1b}}
.pair{{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}} figure{{margin:0}}
img{{image-rendering:auto;max-width:none;border:1px solid #777}} figcaption{{color:#bbb}}
</style></head><body>
<div class="warning"><strong>UNSIGNED — HOLD.</strong> This packet has no support PASS. Contact-shadow coverage is a geometric proxy and remains PENDING until two independent external reviewers inspect all 81 native frames. It is not a clean plate and does not authorize GPU/VACE use.</div>
<h1>case01 bone + contact-shadow + halo support candidate</h1>
<p>Red: reviewed SAM2 bone. Yellow: exact 8px safety halo ring. Blue: directional contact-shadow proxy. Pale cyan: protected 12px dog guard. White: candidate bbox.</p>
<p>Reviewers must copy and fill <code>reviews/reviewer_1_receipt.template.json</code> and <code>reviews/reviewer_2_receipt.template.json</code> independently. Every field starts PENDING.</p>
{''.join(cards)}
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--masklet-root", required=True, type=Path)
    parser.add_argument("--sparse-spec", required=True, type=Path)
    parser.add_argument("--sparse-manifest", required=True, type=Path)
    parser.add_argument("--sparse-receipt", required=True, type=Path)
    parser.add_argument("--revoked-r4-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_video = args.source_video.resolve(strict=True)
    masklet_root = args.masklet_root.resolve(strict=True)
    sparse_spec_path = args.sparse_spec.resolve(strict=True)
    sparse_manifest_path = args.sparse_manifest.resolve(strict=True)
    sparse_receipt_path = args.sparse_receipt.resolve(strict=True)
    revoked_r4_root = args.revoked_r4_root.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise SupportPacketError("refusing to reuse an output path")
    output.parent.mkdir(parents=True, exist_ok=True)

    authorities = validate_authorities(
        source_video=source_video,
        masklet_root=masklet_root,
        sparse_spec_path=sparse_spec_path,
        sparse_manifest_path=sparse_manifest_path,
        sparse_receipt_path=sparse_receipt_path,
        revoked_r4_root=revoked_r4_root,
    )
    ffmpeg_record = ffmpeg_identity(ffmpeg)
    source_rgb = decode_source_rgb(ffmpeg, source_video)
    revoked_support = decode_revoked_support(ffmpeg, authorities["revoked_support_path"])

    import cv2
    import numpy as np

    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial.", dir=str(output.parent))
    )
    try:
        for relative in (
            "masks/bone_core",
            "masks/safety_halo8_ring",
            "masks/contact_shadow_apron",
            "masks/candidate_support",
            "masks/dog_guard12",
            "overlays/native",
            "crops/source_native",
            "crops/overlay_native",
            "reviews",
            "reference",
        ):
            (staging / relative).mkdir(parents=True)
        shutil.copyfile(source_video, staging / "reference/case01-source.authority.mp4")

        frame_rows: list[dict[str, Any]] = []
        for frame_index in range(FRAME_COUNT):
            bone_path = masklet_root / f"masks/bone/{frame_index:05d}.png"
            dog_path = masklet_root / f"masks/dog/{frame_index:05d}.png"
            bone_raw = cv2.imread(str(bone_path), cv2.IMREAD_GRAYSCALE)
            dog_raw = cv2.imread(str(dog_path), cv2.IMREAD_GRAYSCALE)
            if bone_raw is None or dog_raw is None:
                raise SupportPacketError("could not decode an authority mask")
            if set(np.unique(bone_raw).tolist()) != {0, 255} or set(
                np.unique(dog_raw).tolist()
            ) != {0, 255}:
                raise SupportPacketError("authority masks must be nonempty binary PNGs")
            bone = bone_raw != 0
            dog = dog_raw != 0
            parts = build_support_components(bone, dog)

            r4_start = frame_index * FRAME_PIXELS
            r4_frame = np.frombuffer(
                revoked_support[r4_start : r4_start + FRAME_PIXELS], dtype=np.uint8
            ).reshape(HEIGHT, WIDTH) != 0
            expected_r4 = dilate_square(bone, REVOKED_R4_RADIUS)
            if not np.array_equal(r4_frame, expected_r4):
                raise SupportPacketError("revoked r4 support is not exact bone dilate3")
            if not np.all(parts["candidate_support"][r4_frame]):
                raise SupportPacketError("new candidate does not contain revoked r4 support")

            frame_rgb_start = frame_index * RGB_FRAME_BYTES
            frame_rgb_bytes = source_rgb[
                frame_rgb_start : frame_rgb_start + RGB_FRAME_BYTES
            ]
            frame_rgb = np.frombuffer(frame_rgb_bytes, dtype=np.uint8).reshape(
                HEIGHT, WIDTH, 3
            )
            overlay = render_overlay(frame_rgb, parts, frame_index)
            crop_box = padded_bbox(bbox_xyxy(parts["candidate_support"]))
            x1, y1, x2, y2 = crop_box

            written: dict[str, dict[str, Any]] = {}
            for name in (
                "bone_core",
                "safety_halo8_ring",
                "contact_shadow_apron",
                "candidate_support",
                "dog_guard12",
            ):
                path = staging / f"masks/{name}/{frame_index:05d}.png"
                save_binary_png(path, parts[name])
                written[name] = file_record(path, staging)
            overlay_path = staging / f"overlays/native/{frame_index:05d}.png"
            source_crop_path = staging / f"crops/source_native/{frame_index:05d}.png"
            overlay_crop_path = staging / f"crops/overlay_native/{frame_index:05d}.png"
            save_rgb_png(overlay_path, overlay)
            save_rgb_png(source_crop_path, frame_rgb[y1:y2, x1:x2])
            save_rgb_png(overlay_crop_path, overlay[y1:y2, x1:x2])
            written["overlay_native"] = file_record(overlay_path, staging)
            written["source_crop_native"] = file_record(source_crop_path, staging)
            written["overlay_crop_native"] = file_record(overlay_crop_path, staging)

            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "source_decoded_rgb_sha256": hashlib.sha256(frame_rgb_bytes).hexdigest(),
                    "authority_inputs": {
                        "bone": {
                            "path": bone_path.relative_to(masklet_root).as_posix(),
                            "sha256": sha256_path(bone_path),
                        },
                        "dog": {
                            "path": dog_path.relative_to(masklet_root).as_posix(),
                            "sha256": sha256_path(dog_path),
                        },
                    },
                    "areas": {
                        "bone_core": int(parts["bone_core"].sum()),
                        "safety_halo8_ring": int(parts["safety_halo8_ring"].sum()),
                        "contact_shadow_apron_outside_halo8": int(
                            parts["contact_shadow_apron"].sum()
                        ),
                        "candidate_support": int(parts["candidate_support"].sum()),
                        "dog_guard12": int(parts["dog_guard12"].sum()),
                        "revoked_r4_dilate3": int(r4_frame.sum()),
                    },
                    "candidate_bbox_xyxy": bbox_xyxy(parts["candidate_support"]),
                    "crop_bbox_xyxy": crop_box,
                    "checks": {
                        "bone_exactly_in_candidate": bool(
                            np.all(parts["candidate_support"][bone])
                        ),
                        "full_chebyshev_halo8_in_candidate": bool(
                            np.all(
                                parts["candidate_support"][parts["halo8_inclusive"]]
                            )
                        ),
                        "candidate_dog_overlap_pixels": int(
                            np.logical_and(parts["candidate_support"], dog).sum()
                        ),
                        "candidate_dog_guard12_overlap_pixels": int(
                            np.logical_and(
                                parts["candidate_support"], parts["dog_guard12"]
                            ).sum()
                        ),
                        "revoked_r4_equals_bone_dilate3": True,
                        "revoked_r4_is_strictly_contained": bool(
                            parts["candidate_support"].sum() > r4_frame.sum()
                        ),
                        "contact_shadow_visual_coverage": "PENDING_TWO_EXTERNAL_REVIEWS",
                    },
                    "outputs": written,
                }
            )

        write_json(staging / "reviews/reviewer_1_receipt.template.json", review_template(1))
        write_json(staging / "reviews/reviewer_2_receipt.template.json", review_template(2))
        (staging / "index.html").write_text(build_html(frame_rows), encoding="utf-8")

        premanifest_records = output_tree_records(staging)
        tree_digest = hashlib.sha256(
            canonical_json_bytes(premanifest_records)
        ).hexdigest()
        authority_records = {
            key: {
                "path": str(path),
                "sha256": sha256_path(path),
                "size": path.stat().st_size,
            }
            for key, path in authorities["paths"].items()
        }
        manifest = {
            "schema_version": SCHEMA,
            "case_id": CASE_ID,
            "iid": IID,
            "status": "UNSIGNED_CANDIDATE_HOLD_PENDING_TWO_EXTERNAL_REVIEWS",
            "candidate_is_review_passed": False,
            "contact_shadow_visual_coverage": "PENDING_TWO_EXTERNAL_REVIEWS",
            "frame_count": FRAME_COUNT,
            "image_size_wh": [WIDTH, HEIGHT],
            "fps": FPS,
            "derivation": {
                "bone_core": "byte-pinned reviewed SAM2 r2 bone mask",
                "safety_halo": {
                    "metric": "Chebyshev/square dilation",
                    "radius_pixels": SAFETY_HALO_RADIUS,
                    "source_sparse_contract_minimum_pixels": SAFETY_HALO_RADIUS,
                },
                "contact_shadow_apron": {
                    "kind": "UNREVIEWED_DIRECTIONAL_GEOMETRIC_PROXY",
                    "x_offsets_inclusive": [-CONTACT_X_RADIUS, CONTACT_X_RADIUS],
                    "y_offsets_inclusive": [-CONTACT_Y_UP, CONTACT_Y_DOWN],
                    "visual_coverage_proven": False,
                },
                "candidate_support": "union(bone, halo8, directional shadow apron)",
                "dog_guard": {
                    "metric": "Chebyshev/square dilation",
                    "radius_pixels": DOG_GUARD_RADIUS,
                    "overlap_policy": "fail closed; never clip silently",
                },
            },
            "negative_evidence": {
                "revoked_r4_role": "comparison only; never a clean-plate or support authority",
                "revoked_r4_visual_status": "REVOKED_VISUAL_SCAR_FAILED_CLEANPLATE_AUTHORITY",
                "decoded_support_relation": "exact bone Chebyshev dilate3 in all 81 frames and strictly contained by this candidate",
                "revoked_pixels_used_to_seed_candidate": False,
            },
            "authority": {
                **authority_records,
                "ffmpeg_decoder": ffmpeg_record,
                "source_decoded_rgb_sequence_sha256": hashlib.sha256(source_rgb).hexdigest(),
            },
            "review_gate": {
                "required_external_reviewers": 2,
                "reviewers_must_be_independent": True,
                "all_81_native_frames_each": True,
                "candidate_manifest_sha256_must_be_bound_in_each_receipt": True,
                "both_receipts_must_pass_every_frame": True,
                "current_external_receipts": 0,
                "current_status": "HOLD",
                "templates_are_not_receipts": True,
            },
            "claim_limits": {
                "cleanplate_generated": False,
                "renderer_or_vace_run_authorized": False,
                "gpu_execution_performed": False,
                "training_performed": False,
                "optimizer_updates": 0,
                "scientific_claim_authorized": False,
                "support_pass_authorized": False,
                "automatic_contact_shadow_claim_authorized": False,
                "sparse_g0_independent_second_review_still_pending": True,
            },
            "frames": frame_rows,
            "premanifest_output_tree": premanifest_records,
            "premanifest_output_tree_digest": tree_digest,
        }
        write_json(staging / "manifest.json", manifest)
        all_records = output_tree_records(staging, excluded=("SHA256SUMS",))
        sums = "".join(
            f"{row['sha256']}  {row['path']}\n" for row in all_records
        )
        (staging / "SHA256SUMS").write_text(sums, encoding="ascii")

        # A final exact tree scan catches unlisted or late-created files.
        final_paths = [
            path.relative_to(staging).as_posix()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        expected_paths = sorted(
            [row["path"] for row in all_records] + ["SHA256SUMS"]
        )
        if final_paths != expected_paths:
            raise SupportPacketError("final output tree differs")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "UNSIGNED_CANDIDATE_HOLD_PENDING_TWO_EXTERNAL_REVIEWS",
                "output_dir": str(output),
                "manifest_sha256": sha256_path(output / "manifest.json"),
                "frame_count": FRAME_COUNT,
                "review_pass": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

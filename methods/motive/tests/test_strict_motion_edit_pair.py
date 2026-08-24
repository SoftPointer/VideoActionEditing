from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as torch_functional


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "motive" / "strict_motion_edit_pair.py"
_SPEC = importlib.util.spec_from_file_location(
    "_strict_motion_edit_pair_under_test",
    MODULE,
)
assert _SPEC is not None and _SPEC.loader is not None
_PAIR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PAIR
_SPEC.loader.exec_module(_PAIR)

FIRST_FRAME_POLICY = _PAIR.FIRST_FRAME_POLICY
GENERATED_MANIFEST_SCHEMA = _PAIR.GENERATED_MANIFEST_SCHEMA
SAMPLE_RESULT_SCHEMA = _PAIR.SAMPLE_RESULT_SCHEMA
StrictMotionEditDataset = _PAIR.StrictMotionEditDataset
StrictMotionEditPairError = _PAIR.StrictMotionEditPairError
load_strict_motion_edit_pair = _PAIR.load_strict_motion_edit_pair
read_generated_manifest = _PAIR.read_generated_manifest
validate_generated_row = _PAIR.validate_generated_row


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object_digest(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    payload = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(payload)


def _project(array: np.ndarray) -> np.ndarray:
    return (
        np.rint(
            (array.transpose(1, 2, 0) + np.float32(1.0))
            * np.float32(127.5)
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


def _fixture(root: Path) -> tuple[dict[str, object], np.ndarray, Path]:
    source = root / "source.avi"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"MJPG"),
        6.0,
        (6, 4),
    )
    if not writer.isOpened():  # pragma: no cover - environment failure
        raise RuntimeError("OpenCV cannot create the strict-pair test video")
    for frame_index in range(5):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = 30 + frame_index
        frame[:, :, 1] = np.arange(6, dtype=np.uint8)[None, :] * 20
        frame[:, :, 2] = np.arange(4, dtype=np.uint8)[:, None] * 35
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(source))
    ok, source_i0_bgr = capture.read()
    capture.release()
    if not ok:  # pragma: no cover - environment failure
        raise RuntimeError("OpenCV cannot decode the strict-pair test video")
    source_i0_rgb = cv2.cvtColor(source_i0_bgr, cv2.COLOR_BGR2RGB)

    sample = root / "samples" / "dog-0001"
    sample.mkdir(parents=True)
    target = sample / "preview.mp4"
    target.write_bytes(b"lossy-target-preview-not-native-frame0-equal")
    anchor = sample / "conditioning_anchor_original.png"
    Image.fromarray(source_i0_rgb, mode="RGB").save(anchor, format="PNG")

    anchor_tensor = (
        torch.tensor(source_i0_rgb.tolist(), dtype=torch.uint8)
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div(255.0)
        .sub(0.5)
        .div(0.5)
    )
    conditioning_tensor = torch_functional.interpolate(
        anchor_tensor[None],
        size=(2, 3),
        mode="bicubic",
        align_corners=False,
    )[0].contiguous()
    conditioning = np.asarray(
        conditioning_tensor.tolist(),
        dtype="<f4",
    )
    float32_path = sample / "conditioning_frame0_float32.npy"
    np.save(float32_path, conditioning, allow_pickle=False)
    png_pixels = _project(conditioning)
    png_path = sample / "conditioning_frame0.png"
    Image.fromarray(png_pixels, mode="RGB").save(png_path, format="PNG")
    pixel_digest = hashlib.sha256(png_pixels.tobytes(order="C")).hexdigest()
    approval = {
        "schema_version": _PAIR.APPROVAL_SCHEMA,
        "approval_digest": "1" * 64,
        "approval_file_sha256": "2" * 64,
        "proposal_sha256": "3" * 64,
        "reviewer_id": "human-reviewer",
        "reviewed_at_utc": "2026-07-30T15:00:00Z",
        "decision": "approved",
        "reason": "The requested target action is causally grounded in I0.",
    }
    temporal_policy = {
        "policy_version": _PAIR.TEMPORAL_POLICY,
        "model_sample_fps": _PAIR.MODEL_SAMPLE_FPS,
        "model_sample_fps_role": "diffusion_configuration_only",
        "output_container_rate_source": "source_video",
        "source": {
            "frame_count": 5,
            "frame_rate": "6/1",
            "duration_seconds": 5 / 6,
        },
        "target": {
            "frame_count": 5,
            "frame_rate": "6/1",
            "duration_seconds": 5 / 6,
        },
        "frame_count_equal": True,
        "frame_rate_equal": True,
        "duration_delta_seconds": 0.0,
        "duration_delta_frames": 0.0,
        "duration_match_tolerance_frames": 1,
        "duration_match_tolerance_seconds": 1 / 6,
        "duration_within_tolerance": True,
    }

    result: dict[str, object] = {
        "schema_version": SAMPLE_RESULT_SCHEMA,
        "iid": "dog-0001",
        "group_id": "group-dog-0001",
        "action_change_substantive": "yes",
        "seed": 260730,
        "authorization_mode": _PAIR.REQUIRED_AUTHORIZATION_MODE,
        "manifest_role": _PAIR.APPROVED_MANIFEST_ROLE,
        "production_eligible": True,
        "generation_authorized_in_manifest": True,
        "human_review_status_at_generation": "approved",
        "approval": approval,
        "temporal_policy": temporal_policy,
        "prompt": {
            "field": "absolute_target_prompt",
            "text": (
                "The same seated dog first picks up the visible bone and "
                "then stands."
            ),
            "edit_instruction": (
                "Have the dog pick up the visible bone, then stand."
            ),
        },
        "inputs": {
            "source_video_resolved_path": str(source),
            "source_video_sha256": _sha(source),
            "anchor_sha256": _sha(anchor),
            "anchor_rgb_sha256": hashlib.sha256(
                source_i0_rgb.tobytes(order="C")
            ).hexdigest(),
            "anchor_width": 6,
            "anchor_height": 4,
            "source_video_ffprobe": {
                "frames": 5,
                "frame_rate": "6/1",
                "duration_seconds": 5 / 6,
            },
        },
        "first_frame_policy": {
            "policy_version": FIRST_FRAME_POLICY,
            "tensor_frame0_overridden_before_encoding": True,
            "conditioning_tensor_shape": [3, 2, 3],
            "conditioning_tensor_dtype": "float32",
            "preencode_frame0_pixel_sha256": pixel_digest,
            "lossless_png_pixel_sha256": pixel_digest,
            "preencode_frame0_matches_png_pixels": True,
            "mp4_codec_is_lossy": True,
            "mp4_decode_pixel_equality_claimed": False,
        },
        "outputs": {
            "preview_mp4": target.name,
            "preview_mp4_sha256": _sha(target),
            "preview_mp4_ffprobe": {
                "frames": 5,
                "frame_rate": "6/1",
                "duration_seconds": 5 / 6,
            },
            "conditioning_anchor_original": anchor.name,
            "conditioning_anchor_original_sha256": _sha(anchor),
            "conditioning_frame0_float32": float32_path.name,
            "conditioning_frame0_float32_sha256": _sha(float32_path),
            "conditioning_frame0_png": png_path.name,
            "conditioning_frame0_png_sha256": _sha(png_path),
        },
    }
    result["result_digest"] = _object_digest(result)
    result_path = sample / "result.json"
    _write_json(result_path, result)
    row: dict[str, object] = {
        "schema_version": GENERATED_MANIFEST_SCHEMA,
        "iid": "dog-0001",
        "group_id": "group-dog-0001",
        "action_category": "interaction",
        "target_action_verb": "pick_up",
        "action_change_substantive": "yes",
        "absolute_target_prompt": result["prompt"]["text"],  # type: ignore[index]
        "edit_instruction": result["prompt"]["edit_instruction"],  # type: ignore[index]
        "source_video": source.name,
        "source_video_sha256": _sha(source),
        "conditioning_anchor_original": str(anchor.relative_to(root)),
        "conditioning_anchor_original_sha256": _sha(anchor),
        "conditioning_frame0_float32": str(float32_path.relative_to(root)),
        "conditioning_frame0_float32_sha256": _sha(float32_path),
        "conditioning_frame0_png": str(png_path.relative_to(root)),
        "conditioning_frame0_png_sha256": _sha(png_path),
        "target_preview_mp4": str(target.relative_to(root)),
        "target_preview_mp4_sha256": _sha(target),
        "result_json": str(result_path.relative_to(root)),
        "result_digest": result["result_digest"],
        "seed": 260730,
        "authorization_mode": _PAIR.REQUIRED_AUTHORIZATION_MODE,
        "manifest_role": _PAIR.APPROVED_MANIFEST_ROLE,
        "production_eligible": True,
        "human_review_status": "approved",
        "generation_authorized": True,
        "approval": approval,
        "temporal_policy": temporal_policy,
        "first_frame_policy": FIRST_FRAME_POLICY,
        "mp4_decode_pixel_equality_claimed": False,
    }
    return row, conditioning, result_path


def _rewrite_result(
    row: dict[str, object],
    result_path: Path,
    mutate,
) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutate(result)
    result.pop("result_digest", None)
    result["result_digest"] = _object_digest(result)
    _write_json(result_path, result)
    row["result_digest"] = result["result_digest"]
    return result


def _decoded_pair(
    *,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.full(
        (num_frames, 3, height, width),
        -0.75,
        dtype=torch.float32,
    )
    target = torch.full(
        (num_frames, 3, height, width),
        0.75,
        dtype=torch.float32,
    )
    return source, target


def _torch_array(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array.tolist(), dtype=torch.float32)


class StrictMotionEditPairTests(unittest.TestCase):
    def test_replaces_both_decoded_frame_zero_from_one_float32_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, conditioning, _ = _fixture(root)
            source, target = _decoded_pair(num_frames=3, height=2, width=3)
            with mock.patch.object(
                _PAIR,
                "_decode_video_tensor",
                side_effect=[source, target],
            ):
                sample = load_strict_motion_edit_pair(
                    row,
                    base_dir=root,
                    width=3,
                    height=2,
                    num_frames=3,
                )
            expected = _torch_array(conditioning)
            self.assertTrue(torch.equal(sample["source_video"][0], expected))
            self.assertTrue(torch.equal(sample["target_video"][0], expected))
            self.assertTrue(
                torch.equal(
                    sample["source_video"][0],
                    sample["target_video"][0],
                )
            )
            self.assertTrue(
                torch.equal(sample["source_video"][1], source[1])
            )
            self.assertTrue(
                torch.equal(sample["target_video"][1], target[1])
            )
            self.assertFalse(sample["mp4_decode_pixel_equality_claimed"])
            self.assertTrue(sample["strict_frame0_replacement_applied"])
            self.assertLessEqual(
                sample["conditioning_reconstruction_max_abs_error"],
                _PAIR.CONDITIONING_RECONSTRUCTION_ATOL,
            )
            self.assertEqual(tuple(sample["source_video"].shape), (3, 3, 2, 3))

    def test_float32_resize_is_shared_bicubic_without_clamping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, conditioning, _ = _fixture(root)
            source, target = _decoded_pair(num_frames=3, height=4, width=6)
            with mock.patch.object(
                _PAIR,
                "_decode_video_tensor",
                side_effect=[source, target],
            ):
                sample = load_strict_motion_edit_pair(
                    row,
                    base_dir=root,
                    width=6,
                    height=4,
                    num_frames=3,
                )
            expected = torch_functional.interpolate(
                _torch_array(conditioning)[None],
                size=(4, 6),
                mode="bicubic",
                align_corners=False,
            )[0]
            self.assertTrue(torch.equal(sample["source_video"][0], expected))
            self.assertTrue(torch.equal(sample["target_video"][0], expected))

    def test_png_sidecar_is_an_explicit_quantized_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, conditioning, _ = _fixture(root)
            source, target = _decoded_pair(num_frames=2, height=2, width=3)
            with mock.patch.object(
                _PAIR,
                "_decode_video_tensor",
                side_effect=[source, target],
            ):
                sample = load_strict_motion_edit_pair(
                    row,
                    base_dir=root,
                    width=3,
                    height=2,
                    num_frames=2,
                    sidecar="png",
                )
            expected = _torch_array(
                (_project(conditioning).astype(np.float32) / 127.5 - 1.0)
                .transpose(2, 0, 1)
                .copy()
            )
            self.assertTrue(torch.equal(sample["source_video"][0], expected))
            self.assertTrue(torch.equal(sample["target_video"][0], expected))
            self.assertEqual(sample["conditioning_frame0_kind"], "png")

    def test_tampered_target_preview_is_rejected_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            target = root / str(row["target_preview_mp4"])
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "hash mismatch",
            ):
                validate_generated_row(row, base_dir=root)

    def test_forbidden_mp4_native_equality_claim_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, result_path = _fixture(root)
            row["mp4_decode_pixel_equality_claimed"] = True
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "deny MP4 decoded pixel equality",
            ):
                validate_generated_row(row, base_dir=root)

            row["mp4_decode_pixel_equality_claimed"] = False
            _rewrite_result(
                row,
                result_path,
                lambda result: result["first_frame_policy"].update(  # type: ignore[index]
                    {"mp4_decode_pixel_equality_claimed": True}
                ),
            )
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "mp4_decode_pixel_equality_claimed",
            ):
                validate_generated_row(row, base_dir=root)

    def test_semantically_mismatched_png_and_npy_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, conditioning, result_path = _fixture(root)
            png_path = root / str(row["conditioning_frame0_png"])
            pixels = _project(conditioning)
            pixels[0, 0, 0] ^= np.uint8(1)
            Image.fromarray(pixels, mode="RGB").save(png_path, format="PNG")
            changed_file_hash = _sha(png_path)
            changed_pixel_hash = hashlib.sha256(
                pixels.tobytes(order="C")
            ).hexdigest()
            row["conditioning_frame0_png_sha256"] = changed_file_hash

            def mutate(result: dict[str, object]) -> None:
                outputs = result["outputs"]  # type: ignore[assignment]
                outputs["conditioning_frame0_png_sha256"] = changed_file_hash
                policy = result["first_frame_policy"]  # type: ignore[assignment]
                policy["preencode_frame0_pixel_sha256"] = changed_pixel_hash
                policy["lossless_png_pixel_sha256"] = changed_pixel_hash

            _rewrite_result(row, result_path, mutate)
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "display projection",
            ):
                validate_generated_row(row, base_dir=root)

    def test_anchor_pixels_must_equal_exact_source_frame_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, result_path = _fixture(root)
            anchor_path = root / str(
                row["conditioning_anchor_original"]
            )
            with Image.open(anchor_path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            pixels[0, 0, 0] ^= np.uint8(1)
            Image.fromarray(pixels, mode="RGB").save(
                anchor_path,
                format="PNG",
            )
            changed_file_hash = _sha(anchor_path)
            changed_pixel_hash = hashlib.sha256(
                pixels.tobytes(order="C")
            ).hexdigest()
            row[
                "conditioning_anchor_original_sha256"
            ] = changed_file_hash

            def mutate(result: dict[str, object]) -> None:
                outputs = result["outputs"]  # type: ignore[assignment]
                outputs[
                    "conditioning_anchor_original_sha256"
                ] = changed_file_hash
                inputs = result["inputs"]  # type: ignore[assignment]
                inputs["anchor_sha256"] = changed_file_hash
                inputs["anchor_rgb_sha256"] = changed_pixel_hash

            _rewrite_result(row, result_path, mutate)
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "do not equal.*source frame zero",
            ):
                validate_generated_row(row, base_dir=root)

    def test_float32_sidecar_must_be_bicubic_derived_from_source_i0(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, result_path = _fixture(root)
            float32_path = root / str(
                row["conditioning_frame0_float32"]
            )
            array = np.load(float32_path, allow_pickle=False)
            array[0, 0, 0] += np.float32(0.05)
            np.save(float32_path, array.astype("<f4"), allow_pickle=False)
            png_path = root / str(row["conditioning_frame0_png"])
            pixels = _project(array)
            Image.fromarray(pixels, mode="RGB").save(
                png_path,
                format="PNG",
            )
            float_hash = _sha(float32_path)
            png_hash = _sha(png_path)
            pixel_hash = hashlib.sha256(
                pixels.tobytes(order="C")
            ).hexdigest()
            row["conditioning_frame0_float32_sha256"] = float_hash
            row["conditioning_frame0_png_sha256"] = png_hash

            def mutate(result: dict[str, object]) -> None:
                outputs = result["outputs"]  # type: ignore[assignment]
                outputs[
                    "conditioning_frame0_float32_sha256"
                ] = float_hash
                outputs["conditioning_frame0_png_sha256"] = png_hash
                policy = result["first_frame_policy"]  # type: ignore[assignment]
                policy["preencode_frame0_pixel_sha256"] = pixel_hash
                policy["lossless_png_pixel_sha256"] = pixel_hash

            _rewrite_result(row, result_path, mutate)
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "not derived from the exact source-I0 anchor",
            ):
                validate_generated_row(row, base_dir=root)

    def test_generated_row_result_digest_must_match_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            row["result_digest"] = "0" * 64
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "differs from result_json",
            ):
                validate_generated_row(row, base_dir=root)

    def test_manifest_rejects_duplicate_iid_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            manifest = root / "generated_manifest.jsonl"
            _write_manifest(manifest, [row, row])
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "duplicate generated manifest iid",
            ):
                read_generated_manifest(manifest)
            payload = json.dumps(row, sort_keys=True).encode("utf-8")
            manifest.write_bytes(payload)
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "end with a newline",
            ):
                read_generated_manifest(manifest)

    def test_manifest_under_invalid_batch_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = root / "batch" / "final"
            final.mkdir(parents=True)
            manifest = final / "generated_manifest.jsonl"
            manifest.write_text('{"iid":"unused"}\n', encoding="utf-8")
            marker = root / "batch" / _PAIR.INVALID_BATCH_MARKER
            marker.write_text(
                '{"status":"invalid_do_not_train"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "quarantined.*INVALID_DO_NOT_TRAIN",
            ):
                read_generated_manifest(manifest)

    def test_copied_manifest_cannot_escape_sample_batch_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "invalid-batch"
            batch.mkdir()
            row, _, _ = _fixture(batch)
            for field in (
                "source_video",
                "target_preview_mp4",
                "conditioning_anchor_original",
                "conditioning_frame0_float32",
                "conditioning_frame0_png",
                "result_json",
            ):
                row[field] = str((batch / str(row[field])).resolve())
            (batch / _PAIR.INVALID_BATCH_MARKER).write_text(
                '{"status":"invalid_do_not_train"}\n',
                encoding="utf-8",
            )
            copied_manifest_parent = root / "copied-elsewhere"
            copied_manifest_parent.mkdir()
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "quarantined.*INVALID_DO_NOT_TRAIN",
            ):
                validate_generated_row(
                    row,
                    base_dir=copied_manifest_parent,
                )

    def test_pending_review_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            row["authorization_mode"] = "explicit_pending_review_override"
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "proposal-bound human approval",
            ):
                validate_generated_row(row, base_dir=root)

    def test_unbound_or_rejected_approval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, result_path = _fixture(root)
            approval = dict(row["approval"])  # type: ignore[arg-type]
            approval["decision"] = "rejected"
            row["approval"] = approval
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "decision must be exactly 'approved'",
            ):
                validate_generated_row(row, base_dir=root)

            second = root / "second"
            second.mkdir()
            row, _, result_path = _fixture(second)
            row_approval = dict(row["approval"])  # type: ignore[arg-type]
            row_approval["reason"] = "A different unbound review reason."
            row["approval"] = row_approval
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "approval differs",
            ):
                validate_generated_row(row, base_dir=second)

    def test_source_target_25_to_16_fps_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, result_path = _fixture(root)

            def mutate(result: dict[str, object]) -> None:
                inputs = result["inputs"]  # type: ignore[assignment]
                inputs["source_video_ffprobe"] = {
                    "frames": 81,
                    "frame_rate": "25/1",
                    "duration_seconds": 3.24,
                }
                outputs = result["outputs"]  # type: ignore[assignment]
                outputs["preview_mp4_ffprobe"] = {
                    "frames": 81,
                    "frame_rate": "16/1",
                    "duration_seconds": 5.0625,
                }

            _rewrite_result(row, result_path, mutate)
            with self.assertRaisesRegex(
                StrictMotionEditPairError,
                "temporal FPS mismatch.*source=25.*target=16",
            ):
                validate_generated_row(row, base_dir=root)

    def test_81_frame_25_fps_temporal_grid_is_accepted(self) -> None:
        source, target = _PAIR._validate_temporal_alignment(
            inputs={
                "source_video_ffprobe": {
                    "frames": 81,
                    "frame_rate": "25/1",
                    "duration_seconds": 3.24,
                }
            },
            outputs={
                "preview_mp4_ffprobe": {
                    "frames": 81,
                    "frame_rate": "25/1",
                    "duration_seconds": 3.24,
                }
            },
        )
        self.assertEqual(source, target)
        self.assertEqual(source[0], 81)
        self.assertEqual(source[1], _PAIR.Fraction(25, 1))
        self.assertEqual(source[2], 3.24)

    def test_dataset_validates_eagerly_and_yields_strict_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            manifest = root / "generated_manifest.jsonl"
            _write_manifest(manifest, [row])
            dataset = StrictMotionEditDataset(
                manifest,
                width=3,
                height=2,
                num_frames=2,
            )
            self.assertEqual(len(dataset), 1)
            source, target = _decoded_pair(num_frames=2, height=2, width=3)
            with mock.patch.object(
                _PAIR,
                "_decode_video_tensor",
                side_effect=[source, target],
            ):
                sample = dataset[0]
            self.assertTrue(
                torch.equal(
                    sample["source_video"][0],
                    sample["target_video"][0],
                )
            )

    def test_first_sampling_stops_decoding_after_needed_strided_frames(self) -> None:
        consumed: list[int] = []

        def frames():
            for index in range(100):
                consumed.append(index)
                yield np.full((2, 3, 3), index, dtype=np.uint8)

        with mock.patch.object(
            _PAIR,
            "_iter_decoded_frames",
            side_effect=lambda _path: frames(),
        ):
            tensor = _PAIR._decode_video_tensor(
                Path("/not/opened.mp4"),
                width=3,
                height=2,
                num_frames=3,
                frame_stride=2,
                sample_mode="first",
                short_video_mode="error",
            )
        self.assertEqual(consumed, [0, 1, 2, 3, 4])
        self.assertEqual(tuple(tensor.shape), (3, 3, 2, 3))
        expected_values = torch.tensor(
            [0.0, 2.0, 4.0],
            dtype=torch.float32,
        ) / 127.5 - 1.0
        self.assertTrue(
            torch.allclose(tensor[:, 0, 0, 0], expected_values)
        )

    def test_contract_and_cli_never_claim_native_mp4_equality(self) -> None:
        contract = _PAIR.loader_contract()
        self.assertFalse(
            contract["frame_zero"][
                "mp4_decoded_native_pixel_equality_claimed"
            ]
        )
        self.assertTrue(
            contract["temporal_alignment"][
                "source_target_frame_count_equal"
            ]
        )
        self.assertTrue(
            contract["temporal_alignment"]["source_target_fps_equal"]
        )
        output = io.StringIO()
        with redirect_stdout(output):
            status = _PAIR.main(["contract"])
        self.assertEqual(status, 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(
            payload["frame_zero"][
                "mp4_decoded_native_pixel_equality_claimed"
            ]
        )

    def test_audit_cli_validates_and_loads_without_native_equality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row, _, _ = _fixture(root)
            manifest = root / "generated_manifest.jsonl"
            _write_manifest(manifest, [row])
            source, target = _decoded_pair(num_frames=2, height=2, width=3)
            output = io.StringIO()
            with mock.patch.object(
                _PAIR,
                "_decode_video_tensor",
                side_effect=[source, target],
            ), redirect_stdout(output):
                status = _PAIR.main(
                    [
                        "audit",
                        "--manifest",
                        str(manifest),
                        "--width",
                        "3",
                        "--height",
                        "2",
                        "--num-frames",
                        "2",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["audited_rows"], 1)
            self.assertTrue(summary["strict_frame0_replacement_applied"])
            self.assertTrue(summary["source_i0_anchor_pixel_equal"])
            self.assertLessEqual(
                summary["conditioning_reconstruction_max_abs_error"],
                _PAIR.CONDITIONING_RECONSTRUCTION_ATOL,
            )
            self.assertFalse(
                summary["mp4_decoded_native_pixel_equality_claimed"]
            )


if __name__ == "__main__":
    unittest.main()

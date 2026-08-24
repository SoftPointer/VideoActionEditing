from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from motive.goku_full_motion_qwen_v16 import (
    MOTION_EVIDENCE_SCHEMA,
    SOURCE_CAMERA_SCHEMA,
    SOURCE_CENSUS_SCHEMA,
    SOURCE_SUBJECT_SCHEMA,
    TARGET_CAMERA_SCHEMA,
    TARGET_PLAN_SCHEMA,
    TARGET_SUBJECT_SCHEMA,
    run_one,
)
from motive.goku_full_motion_v16_audit import audit_bundle, main


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_row(iid: str, *, source: Path, anchor: Path) -> dict:
    return {
        "iid": iid,
        "group_id": f"group-{iid}",
        "family": "person",
        "src_video": source.name,
        "resolved_src_video": str(source),
        "source_caption": "A person raises one hand.",
        "edited_caption": "The person claps overhead.",
        "prompt": "Change the person's action to overhead clapping.",
        "anchor_image": anchor.name,
        "resolved_anchor_image": str(anchor),
        "anchor_sha256": _sha(anchor),
        "source_video_sha256": _sha(source),
        "prefilter_score": 8.0,
        "media": {},
        "motion": {},
    }


def _census(iid: str) -> dict:
    return {
        "schema_version": SOURCE_CENSUS_SCHEMA,
        "iid": iid,
        "dynamic_subjects": [
            {
                "schema_version": SOURCE_SUBJECT_SCHEMA,
                "subject_id": "subject_01",
                "entity_type": "person",
                "stable_reference": "the person in a blue shirt at center",
                "i0_bbox_xyxy_1000": [200, 100, 800, 980],
                "i0_state": "standing with both hands near the waist",
                "source_action_signature": "raise_right_hand",
                "source_motion": "raises the right hand from waist to shoulder",
                "motion_evidence": [
                    {
                        "schema_version": MOTION_EVIDENCE_SCHEMA,
                        "start_frame": 0,
                        "end_frame": 80,
                        "description": "the right hand rises across ordered frames",
                    }
                ],
                "dynamic": True,
            }
        ],
        "camera": {
            "schema_version": SOURCE_CAMERA_SCHEMA,
            "motion_class": "locked_off",
            "source_motion": "the camera remains locked off",
            "motion_evidence": [
                {
                    "schema_version": MOTION_EVIDENCE_SCHEMA,
                    "start_frame": 0,
                    "end_frame": 80,
                    "description": "the background framing remains fixed",
                }
            ],
        },
        "all_dynamic_subjects_enumerated": True,
        "crowd_or_unresolved_motion": False,
        "confidence": "high",
    }


def _plan(iid: str, *, fail: bool = False) -> dict:
    targets = [] if fail else [
        {
            "schema_version": TARGET_SUBJECT_SCHEMA,
            "subject_id": "subject_01",
            "target_action_signature": "clap_both_hands_overhead",
            "target_motion": (
                "immediately raises both hands overhead and repeatedly claps"
            ),
            "substantive_change": True,
        }
    ]
    return {
        "schema_version": TARGET_PLAN_SCHEMA,
        "iid": iid,
        "dynamic_subject_targets": targets,
        "camera_target": {
            "schema_version": TARGET_CAMERA_SCHEMA,
            "relation": "preserve_static",
            "motion_class": "locked_off",
            "target_motion": "the camera remains completely locked off",
        },
        "coverage": {
            "schema_version": "motive-goku-full-motion-v16-target-coverage-v1",
            "dynamic_subject_ids": ["subject_01"],
            "camera_covered": True,
        },
        "confidence": "high",
    }


class _Backend:
    model_path = "/fake/qwen"
    model_revision = "fixture"
    transformers_version = "fixture"

    def __init__(self, iid: str, *, fail: bool = False) -> None:
        self.iid = iid
        self.fail = fail

    def generate_source_motion_census_v16(self, **kwargs):
        return json.dumps(_census(self.iid)), kwargs["expected_visual_input_digest"]

    def generate_target_plan_v16(self, **kwargs):
        return (
            json.dumps(_plan(self.iid, fail=self.fail)),
            kwargs["expected_visual_input_digest"],
        )


def _run_args(
    input_manifest: Path, qwen_root: Path, root: Path, *, index: int, count: int
) -> argparse.Namespace:
    return argparse.Namespace(
        input=input_manifest,
        output_root=qwen_root,
        model="/fake/qwen",
        root=root,
        row_index=index,
        num_rows=count,
        max_new_tokens=4096,
        nframes=16,
        max_pixels=2_359_296,
        tile_width=512,
        mosaic_columns=4,
        attn_implementation="sdpa",
        allow_download=False,
        allow_errors=True,
    )


def _prepare(row, *, root, runtime):
    return (
        Path(row["resolved_src_video"]),
        Path(row["resolved_anchor_image"]),
        {
            "exact_i0": True,
            "temporal_geometry": {
                "frame_count": 81,
                "fps": "25/1",
                "timeline_span_seconds": 3.2,
                "width": 2,
                "height": 2,
            },
        },
        (object(), object(), object(), object(), object()),
        "a" * 64,
    )


def _load_passed(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_complete_wan_fixture(
    wan_root: Path, iid: str, passed: dict
) -> tuple[Path, Path]:
    output_root = wan_root / "samples" / iid
    output_root.mkdir(parents=True)
    (output_root / "run_contract.json").write_text("{}\n", encoding="utf-8")
    sample_dir = output_root / "samples" / iid
    sample_dir.mkdir(parents=True)

    source = sample_dir / "source_video.mp4"
    source.write_bytes(Path(passed["resolved_source_video"]).read_bytes())
    instruction = sample_dir / "edit_instruction.txt"
    instruction.write_text(passed["edit_instruction"], encoding="utf-8")
    anchor = sample_dir / "conditioning_anchor_original.png"
    anchor.write_bytes(Path(passed["resolved_anchor_image"]).read_bytes())

    uint8_frame = np.full((2, 2, 3), 128, dtype=np.uint8)
    conditioning = (
        (uint8_frame.astype(np.float32) / 127.5) - 1.0
    ).transpose(2, 0, 1)
    float32_path = sample_dir / "conditioning_frame0_float32.npy"
    np.save(float32_path, conditioning.astype(np.float32), allow_pickle=False)
    png_path = sample_dir / "conditioning_frame0.png"
    Image.fromarray(uint8_frame, mode="RGB").save(png_path, format="PNG")
    preview = sample_dir / "preview.mp4"
    preview.write_bytes(b"synthetic-preview-bytes")
    pixel_sha = hashlib.sha256(uint8_frame.tobytes(order="C")).hexdigest()
    result = {
        "iid": iid,
        "result_digest": "9" * 64,
        "prompt": {
            "field": "edit_instruction",
            "text": passed["edit_instruction"],
            "sha256": passed["edit_instruction_sha256"],
        },
        "first_frame_policy": {
            "preencode_frame0_matches_png_pixels": True,
            "preencode_frame0_pixel_sha256": pixel_sha,
            "lossless_png_pixel_sha256": pixel_sha,
        },
        "outputs": {
            "source_video": source.name,
            "source_video_sha256": _sha(source),
            "edit_instruction_file": instruction.name,
            "edit_instruction_file_sha256": _sha(instruction),
            "preview_mp4": preview.name,
            "preview_mp4_sha256": _sha(preview),
            "conditioning_anchor_original": anchor.name,
            "conditioning_anchor_original_sha256": _sha(anchor),
            "conditioning_frame0_float32": float32_path.name,
            "conditioning_frame0_float32_sha256": _sha(float32_path),
            "conditioning_frame0_png": png_path.name,
            "conditioning_frame0_png_sha256": _sha(png_path),
        },
    }
    result_path = sample_dir / "result.json"
    result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return sample_dir, result_path


def _fixture_commit_validator(**kwargs):
    return json.loads((kwargs["sample_dir"] / "result.json").read_text())


def _fixture_probe(path: Path, **kwargs):
    assert kwargs["expected_frames"] == 81
    assert kwargs["expected_fps"] == "25/1"
    assert kwargs["max_nominal_duration_error_frames"] == 0
    return {
        "probe_backend": "synthetic-ffprobe",
        "frames": 81,
        "frame_rate": "25/1",
        "duration_seconds": 3.24,
        "nominal_duration_seconds": 3.24,
    }


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


class GokuFullMotionV16AuditTests(unittest.TestCase):
    def test_synthetic_bundle_classifies_all_states_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.mp4"
            source.write_bytes(b"synthetic-source-video")
            anchor = root / "anchor.png"
            Image.fromarray(
                np.full((2, 2, 3), 128, dtype=np.uint8), mode="RGB"
            ).save(anchor, format="PNG")
            iids = ["pendingq", "qerror", "pendingw", "werror", "done"]
            rows = [_input_row(iid, source=source, anchor=anchor) for iid in iids]
            manifest = root / "input.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            qwen_root = root / "qwen"
            wan_root = root / "wan"
            qwen_root.mkdir()
            wan_root.mkdir()

            for index, iid in enumerate(iids[1:], start=1):
                run_one(
                    _run_args(
                        manifest,
                        qwen_root,
                        root,
                        index=index,
                        count=len(rows),
                    ),
                    backend_factory=lambda iid=iid, **kwargs: _Backend(
                        iid, fail=iid == "qerror"
                    ),
                    prepare=_prepare,
                )

            partial_output = wan_root / "samples" / "werror"
            partial_sample = partial_output / "samples" / "werror"
            partial_sample.mkdir(parents=True)
            (partial_output / "run_contract.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (partial_sample / "orphan.bin").write_bytes(b"partial")

            passed_path = qwen_root / "passed/done.jsonl"
            passed = _load_passed(passed_path)
            _, result_path = _write_complete_wan_fixture(
                wan_root, "done", passed
            )
            before = _tree_digest(root)
            summary = audit_bundle(
                manifest,
                qwen_root,
                wan_root,
                commit_validator=_fixture_commit_validator,
                video_probe=_fixture_probe,
            )
            after = _tree_digest(root)
            self.assertEqual(before, after)
            self.assertEqual(
                summary["counts"],
                {
                    "pending_qwen": 1,
                    "qwen_error": 1,
                    "pending_wan": 1,
                    "wan_error": 1,
                    "complete": 1,
                },
            )
            self.assertEqual(
                [row["status"] for row in summary["rows"]],
                [
                    "pending_qwen",
                    "qwen_error",
                    "pending_wan",
                    "wan_error",
                    "complete",
                ],
            )
            complete = summary["rows"][-1]
            self.assertEqual(complete["wan"]["fresh_ffprobe"]["frames"], 81)
            self.assertEqual(complete["wan"]["container_duration_seconds"], 3.24)
            self.assertEqual(complete["wan"]["timeline_span_seconds"], 3.2)

            sample_dir = result_path.parent
            (sample_dir / "edit_instruction.txt").write_text(
                "tampered", encoding="utf-8"
            )
            tampered = audit_bundle(
                manifest,
                qwen_root,
                wan_root,
                commit_validator=_fixture_commit_validator,
                video_probe=_fixture_probe,
            )
            self.assertEqual(tampered["rows"][-1]["status"], "wan_error")
            self.assertIn(
                "instruction",
                tampered["rows"][-1]["issues"][0]["message"].casefold(),
            )

    def test_cli_prints_json_for_pending_bundle_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            anchor = root / "anchor.png"
            Image.fromarray(
                np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB"
            ).save(anchor, format="PNG")
            manifest = root / "input.jsonl"
            manifest.write_text(
                json.dumps(_input_row("pending", source=source, anchor=anchor))
                + "\n",
                encoding="utf-8",
            )
            qwen_root = root / "qwen"
            wan_root = root / "wan"
            qwen_root.mkdir()
            wan_root.mkdir()
            before = _tree_digest(root)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "--input-manifest",
                        str(manifest),
                        "--qwen-root",
                        str(qwen_root),
                        "--wan-root",
                        str(wan_root),
                    ]
                )
            self.assertEqual(status, 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["counts"]["pending_qwen"], 1)
            self.assertFalse(value["all_complete"])
            self.assertEqual(before, _tree_digest(root))


if __name__ == "__main__":
    unittest.main()

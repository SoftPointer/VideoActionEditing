from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


# This merger is deliberately stdlib-only until strict-I0 replay is invoked.
# Load it directly so this unit test remains runnable even when optional Motive
# numerical dependencies are absent from a lightweight developer environment.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "motive" / "goku_atomic1000_merge.py"
_SPEC = importlib.util.spec_from_file_location("_goku_atomic1000_merge_tested", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
merge = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = merge
_SPEC.loader.exec_module(merge)


PIXEL_SHA = hashlib.sha256(b"conditioning-rgb-pixels").hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _pretty(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _object_digest(value: dict[str, object], field: str) -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))


class Atomic1000MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.ffprobe = self.root / "ffprobe"
        self.ffprobe.write_bytes(b"#!/bin/sh\nexit 1\n")
        self.ffprobe.chmod(0o700)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _one_row(self, epoch_root: Path, *, iid: str, group: str) -> dict[str, object]:
        sample = epoch_root / "sample" / iid
        sample.mkdir(parents=True)
        action = f"Have the visible subject perform action {iid}."
        camera = "Keep the camera fixed."
        preservation = "Preserve the visible appearance and scene."
        full = f"{action} {camera} {preservation}"
        prompt = f"Private trajectory generation prompt for {iid}."

        source = sample / "source_video.mp4"
        target = sample / "preview.mp4"
        anchor = sample / "conditioning_anchor_original.png"
        conditioning_npy = sample / "conditioning_frame0_float32.npy"
        conditioning_png = sample / "conditioning_frame0.png"
        for path, raw in (
            (source, f"source-{iid}".encode()),
            (target, f"target-{iid}".encode()),
            (anchor, f"anchor-{iid}".encode()),
            (conditioning_npy, f"npy-{iid}".encode()),
            (conditioning_png, f"png-{iid}".encode()),
        ):
            path.write_bytes(raw)

        planner = {
            "schema_version": merge.PLANNER_PASSED_SCHEMA,
            "iid": iid,
            "group_id": group,
            "family": "walking",
            "source_video": f"relative/{iid}.mp4",
            "resolved_source_video": str(source),
            "anchor_image": f"anchors/{iid}.png",
            "resolved_anchor_image": str(anchor),
            "source_video_sha256": _sha(source.read_bytes()),
            "anchor_sha256": _sha(anchor.read_bytes()),
            "strict_temporal_geometry": {
                "frame_count": 81,
                "fps": "25/1",
                "timeline_span_seconds": 3.2,
                "width": 1280,
                "height": 704,
            },
            "edit_instruction": prompt,
            "edit_instruction_sha256": _sha(prompt.encode()),
            "source_census": {"opaque": "validated upstream"},
            "target_plan": {"opaque": "validated upstream"},
            "compiled_instruction": {"opaque": "validated upstream"},
            "qwen_record_digest": "1" * 64,
            "action_change_substantive": True,
            "all_dynamic_subjects_covered": True,
            "camera_covered": True,
            "human_review_status": "pending",
            "generation_authorized": False,
            "production_eligible": False,
        }
        planner_path = sample / "planner_primary.jsonl"
        planner_raw = _canonical(planner) + b"\n"
        planner_path.write_bytes(planner_raw)

        atomic: dict[str, object] = {
            "schema_version": merge.ATOMIC_RESULT_SCHEMA,
            "iid": iid,
            "original_candidate_index": 0,
            "status": "ok",
            "input_row_digest": "2" * 64,
            "source_passed_path": str(planner_path),
            "source_passed_sha256": _sha(planner_raw),
            "source_frame_grid_generation_prompt": prompt,
            "source_frame_grid_generation_prompt_sha256": _sha(prompt.encode()),
            "target_plan_sha256": "3" * 64,
            "backend": {},
            "plan_audit_attempts": [],
            "plan_audit": {},
            "rewrite_attempts": [],
            "rewrite": {},
            "semantic_audit": {},
            "atomic_action_instruction": action,
            "atomic_action_instruction_sha256": _sha(action.encode()),
            "camera_instruction": camera,
            "camera_instruction_sha256": _sha(camera.encode()),
            "preservation_instruction": preservation,
            "preservation_instruction_sha256": _sha(preservation.encode()),
            "full_edit_instruction": full,
            "full_edit_instruction_sha256": _sha(full.encode()),
            "error": None,
            "record_digest": None,
        }
        atomic["record_digest"] = _object_digest(atomic, "record_digest")
        atomic_path = sample / "atomic_primary.json"
        atomic_path.write_bytes(_pretty(atomic))

        copied_sidecars = {
            "wan_generation_prompt.txt": prompt.encode(),
            "atomic_action_instruction.txt": action.encode(),
            "camera_instruction.txt": camera.encode(),
            "preservation_instruction.txt": preservation.encode(),
            "full_edit_instruction.txt": full.encode(),
            "planner_passed.jsonl": planner_raw,
            "atomic_result.json": atomic_path.read_bytes(),
            "atomic_admission.json": b"{}\n",
            "edit_instruction.txt": prompt.encode(),
        }
        for name, raw in copied_sidecars.items():
            (sample / name).write_bytes(raw)

        wan: dict[str, object] = {
            "schema_version": merge.WAN_RESULT_SCHEMA,
            "iid": iid,
            "prompt": {
                "field": "edit_instruction",
                "text": prompt,
                "sha256": _sha(prompt.encode()),
            },
            "inputs": {
                "source_video_committed_path": str(source),
                "source_video_sha256": _sha(source.read_bytes()),
            },
            "first_frame_policy": {
                "policy_version": merge.FIRST_FRAME_POLICY,
                "tensor_frame0_overridden_before_encoding": True,
                "conditioning_tensor_dtype": "float32",
                "preencode_frame0_pixel_sha256": PIXEL_SHA,
                "lossless_png_pixel_sha256": PIXEL_SHA,
                "preencode_frame0_matches_png_pixels": True,
                "mp4_codec_is_lossy": True,
                "mp4_decode_pixel_equality_claimed": False,
            },
            "outputs": {
                "preview_mp4": target.name,
                "preview_mp4_sha256": _sha(target.read_bytes()),
                "conditioning_anchor_original": anchor.name,
                "conditioning_anchor_original_sha256": _sha(anchor.read_bytes()),
                "conditioning_frame0_float32": conditioning_npy.name,
                "conditioning_frame0_float32_sha256": _sha(
                    conditioning_npy.read_bytes()
                ),
                "conditioning_frame0_png": conditioning_png.name,
                "conditioning_frame0_png_sha256": _sha(conditioning_png.read_bytes()),
            },
            "result_digest": None,
        }
        wan["result_digest"] = _object_digest(wan, "result_digest")
        wan_path = sample / "result.json"
        wan_path.write_bytes(_pretty(wan))

        artifact_names = set(copied_sidecars) | {
            source.name,
            target.name,
            wan_path.name,
        }
        artifacts = {
            name: {
                "path": str(sample / name),
                "sha256": _sha((sample / name).read_bytes()),
                "bytes": (sample / name).stat().st_size,
            }
            for name in artifact_names
        }
        metadata: dict[str, object] = {
            "schema_version": merge.SAMPLE_METADATA_SCHEMA,
            "iid": iid,
            "primary_training_label_field": "atomic_action_instruction",
            "wan_generation_prompt_field": "planner_passed.edit_instruction",
            "wan_generation_prompt_is_training_label": False,
            "edit_instruction_txt_role": "generation_only_not_training_label",
            "wan_generation_prompt_txt_role": "generation_only_not_training_label",
            "atomic_action_instruction": action,
            "atomic_action_instruction_sha256": _sha(action.encode()),
            "camera_instruction": camera,
            "camera_instruction_sha256": _sha(camera.encode()),
            "preservation_instruction": preservation,
            "preservation_instruction_sha256": _sha(preservation.encode()),
            "full_edit_instruction": full,
            "full_edit_instruction_sha256": _sha(full.encode()),
            "wan_generation_prompt": prompt,
            "wan_generation_prompt_sha256": _sha(prompt.encode()),
            "source_video_sha256": _sha(source.read_bytes()),
            "artifacts": artifacts,
            "metadata_digest": None,
        }
        metadata["metadata_digest"] = _object_digest(metadata, "metadata_digest")
        metadata_path = sample / "atomic_sample_metadata.json"
        metadata_path.write_bytes(_pretty(metadata))

        return {
            "schema_version": merge.ROW_SCHEMA,
            "iid": iid,
            "lineage": "atomic_new_wan",
            "primary_training_label_field": "atomic_action_instruction",
            "atomic_action_instruction": action,
            "atomic_action_instruction_sha256": _sha(action.encode()),
            "camera_instruction": camera,
            "camera_instruction_sha256": _sha(camera.encode()),
            "preservation_instruction": preservation,
            "preservation_instruction_sha256": _sha(preservation.encode()),
            "full_edit_instruction": full,
            "full_edit_instruction_sha256": _sha(full.encode()),
            "wan_generation_prompt": prompt,
            "wan_generation_prompt_sha256": _sha(prompt.encode()),
            "wan_edit_instruction_txt_role": (
                "generation_prompt_not_primary_training_label"
            ),
            "source_video": str(source),
            "source_video_sha256": _sha(source.read_bytes()),
            "target_video": str(target),
            "target_video_sha256": _sha(target.read_bytes()),
            "source_temporal_geometry": {"frame_count": 81, "frame_rate": "25/1"},
            "target_temporal_geometry": {"frame_count": 81, "frame_rate": "25/1"},
            "strict_target_frame0_float32_npy": str(conditioning_npy),
            "strict_target_frame0_float32_npy_sha256": _sha(
                conditioning_npy.read_bytes()
            ),
            "strict_target_frame0_png": str(conditioning_png),
            "strict_target_frame0_png_sha256": _sha(conditioning_png.read_bytes()),
            "strict_source_frame0_anchor_png": str(anchor),
            "strict_source_frame0_anchor_png_sha256": _sha(anchor.read_bytes()),
            "decoded_target_frame0_override_required": True,
            "target_mp4_decoded_frame0_pixel_equality_claimed": False,
            "atomic_result": str(atomic_path),
            "atomic_result_sha256": _sha(atomic_path.read_bytes()),
            "planner_passed": str(planner_path),
            "planner_passed_sha256": _sha(planner_path.read_bytes()),
            "wan_result": str(wan_path),
            "wan_result_sha256": _sha(wan_path.read_bytes()),
            "sample_metadata": str(metadata_path),
            "sample_metadata_sha256": _sha(metadata_path.read_bytes()),
        }

    def _epoch(self, name: str, rows: list[dict[str, object]]) -> Path:
        root = self.root / name
        root.mkdir()
        manifest = b"".join(_canonical(row) + b"\n" for row in rows)
        (root / merge.EPOCH_MANIFEST_NAME).write_bytes(manifest)
        summary = {
            "schema_version": merge.EPOCH_SUMMARY_SCHEMA,
            "status": "complete",
            "minimum_success": len(rows),
            "total_rows": len(rows),
            "new_wan_rows": len(rows),
            "legacy_reused_rows": 0,
            "manifest_sha256": _sha(manifest),
            "primary_training_label_field": "atomic_action_instruction",
            "wan_generation_prompt_is_separate": True,
        }
        (root / merge.EPOCH_SUMMARY_NAME).write_bytes(_pretty(summary))
        return root

    def _mocked_merge(
        self, specs: list[merge.EpochSpec], output: Path
    ) -> dict[str, object]:
        with patch.object(
            merge,
            "_probe_video",
            return_value={"frame_count": 81, "frame_rate": "25/1"},
        ), patch.object(
            merge,
            "_verify_strict_sidecars",
            return_value={
                "source_anchor_rgb_sha256": "a" * 64,
                "conditioning_rgb_sha256": PIXEL_SHA,
            },
        ):
            return merge.merge_epochs(
                specs,
                output_root=output,
                ffprobe=self.ffprobe,
                required_targets=(1, 1),
                expected_total=2,
            )

    def test_two_epochs_merge_in_declared_order_and_bind_outputs(self) -> None:
        epoch0 = self.root / "epoch0"
        epoch1 = self.root / "epoch1"
        row0 = self._one_row(epoch0, iid="iid-000", group="group-000")
        row1 = self._one_row(epoch1, iid="iid-001", group="group-001")
        roots = [self._epoch("run0", [row0]), self._epoch("run1", [row1])]
        output = self.root / "merged"
        done = self._mocked_merge(
            [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)], output
        )
        rows = [
            json.loads(line)
            for line in (output / merge.OUTPUT_MANIFEST_NAME).read_text().splitlines()
        ]
        self.assertEqual([row["iid"] for row in rows], ["iid-000", "iid-001"])
        self.assertEqual(done["total_rows"], 2)
        self.assertEqual(done["epoch_targets"], [1, 1])
        stored_done = json.loads((output / merge.OUTPUT_DONE_NAME).read_text())
        self.assertEqual(
            stored_done["done_digest"], _object_digest(stored_done, "done_digest")
        )
        summary = json.loads((output / merge.OUTPUT_SUMMARY_NAME).read_text())
        self.assertTrue(summary["iid_unique"])
        self.assertTrue(summary["group_id_unique"])
        self.assertEqual(summary["sample_metadata_artifacts_sha256_verified"], 24)

    def test_cross_epoch_duplicate_group_fails_closed(self) -> None:
        row0 = self._one_row(self.root / "a", iid="iid-a", group="same-group")
        row1 = self._one_row(self.root / "b", iid="iid-b", group="same-group")
        roots = [self._epoch("run-a", [row0]), self._epoch("run-b", [row1])]
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "duplicate planner group_id"):
            self._mocked_merge(
                [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)],
                self.root / "duplicate-group-output",
            )

    def test_cross_epoch_duplicate_iid_fails_closed(self) -> None:
        row0 = self._one_row(self.root / "iid-a", iid="same-iid", group="group-one")
        row1 = self._one_row(self.root / "iid-b", iid="same-iid", group="group-two")
        roots = [
            self._epoch("run-iid-a", [row0]),
            self._epoch("run-iid-b", [row1]),
        ]
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "duplicate IID same-iid"):
            self._mocked_merge(
                [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)],
                self.root / "duplicate-iid-output",
            )

    def test_mutated_referenced_artifact_fails_hash_replay(self) -> None:
        row0 = self._one_row(self.root / "c", iid="iid-c", group="group-c")
        row1 = self._one_row(self.root / "d", iid="iid-d", group="group-d")
        roots = [self._epoch("run-c", [row0]), self._epoch("run-d", [row1])]
        Path(str(row1["source_video"])).write_bytes(b"mutated-after-publication")
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "source_video SHA-256 differs"):
            self._mocked_merge(
                [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)],
                self.root / "mutated-output",
            )

    def test_dataset_row_extra_field_is_rejected(self) -> None:
        row0 = self._one_row(self.root / "e", iid="iid-e", group="group-e")
        row1 = self._one_row(self.root / "f", iid="iid-f", group="group-f")
        row1["unbound_extension"] = True
        roots = [self._epoch("run-e", [row0]), self._epoch("run-f", [row1])]
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "dataset schema is open"):
            self._mocked_merge(
                [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)],
                self.root / "open-row-output",
            )

    def test_create_only_output_refuses_second_publication(self) -> None:
        row0 = self._one_row(self.root / "g", iid="iid-g", group="group-g")
        row1 = self._one_row(self.root / "h", iid="iid-h", group="group-h")
        roots = [self._epoch("run-g", [row0]), self._epoch("run-h", [row1])]
        specs = [merge.EpochSpec(roots[0], 1), merge.EpochSpec(roots[1], 1)]
        output = self.root / "create-only"
        self._mocked_merge(specs, output)
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "already exists"):
            self._mocked_merge(specs, output)

    def test_production_target_shape_is_not_only_a_sum_check(self) -> None:
        with self.assertRaisesRegex(merge.Atomic1000MergeError, "epoch targets must be exactly"):
            merge.merge_epochs(
                [merge.EpochSpec(self.root, 500), merge.EpochSpec(self.root / "x", 500)],
                output_root=self.root / "never-created",
                ffprobe=self.ffprobe,
            )


if __name__ == "__main__":
    unittest.main()

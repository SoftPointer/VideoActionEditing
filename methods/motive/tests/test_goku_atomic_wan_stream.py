from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from motive import goku_atomic_wan_stream as stream


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "tmp" / "launch_goku_atomic1000_g8_pipeline.sh"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class AtomicWanStreamTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[argparse.Namespace, list[str]]:
        root = root.resolve()
        planner_root = root / "planner"
        (planner_root / "passed").mkdir(parents=True)
        wan_root = root / "wan"
        admission_root = wan_root / "admissions"
        input_root = wan_root / "inputs"
        receipt_root = wan_root / "admission_batches"
        batch_root = wan_root / "batches" / "batch_0000"
        for path in (admission_root, input_root, receipt_root, batch_root.parent):
            path.mkdir(parents=True, exist_ok=True)

        iids = ["pass-a", "pass-b", "not-yet-passed"]
        planner_rows = [{"iid": iid, "opaque": index} for index, iid in enumerate(iids)]
        planner_input = root / "planner_input.jsonl"
        planner_input.write_bytes(
            b"".join(canonical(row) + b"\n" for row in planner_rows)
        )

        atomic_rows = []
        for index, iid in enumerate(iids[:2]):
            source = root / f"{iid}.mp4"
            source.write_bytes(("source-" + iid).encode())
            prompt = f"Private frame-grid trajectory for {iid}."
            action = f"Have subject {iid} jump."
            camera = "Keep the camera fixed."
            preservation = "Preserve appearance and scene content."
            full = f"{action} {camera} {preservation}"
            atomic_result = root / f"{iid}.atomic.json"
            result_value = {
                "iid": iid,
                "status": "ok",
                "atomic_action_instruction": action,
                "atomic_action_instruction_sha256": sha(action.encode()),
                "camera_instruction": camera,
                "camera_instruction_sha256": sha(camera.encode()),
                "preservation_instruction": preservation,
                "preservation_instruction_sha256": sha(preservation.encode()),
                "full_edit_instruction": full,
                "full_edit_instruction_sha256": sha(full.encode()),
            }
            atomic_result.write_bytes(json.dumps(result_value).encode() + b"\n")
            passed = {
                "iid": iid,
                "resolved_source_video": str(source),
                "source_video_sha256": sha(source.read_bytes()),
                "edit_instruction": prompt,
                "edit_instruction_sha256": sha(prompt.encode()),
            }
            (planner_root / "passed" / f"{iid}.jsonl").write_bytes(
                canonical(passed) + b"\n"
            )
            atomic_rows.append(
                {
                    "schema_version": stream.ATOMIC_DATASET_SCHEMA,
                    "iid": iid,
                    "original_candidate_index": index,
                    "label_status": (
                        "atomic_plan_and_instruction_audits_passed_"
                        "video_audit_pending"
                    ),
                    "primary_training_label_field": "atomic_action_instruction",
                    "source_video": str(source),
                    "source_video_sha256": sha(source.read_bytes()),
                    "source_generation_provenance": {
                        "frame_gridded_prompt": prompt,
                        "frame_gridded_prompt_sha256": sha(prompt.encode()),
                    },
                    "atomic_action_instruction": action,
                    "atomic_action_instruction_sha256": sha(action.encode()),
                    "camera_instruction": camera,
                    "camera_instruction_sha256": sha(camera.encode()),
                    "preservation_instruction": preservation,
                    "preservation_instruction_sha256": sha(preservation.encode()),
                    "full_edit_instruction": full,
                    "full_edit_instruction_sha256": sha(full.encode()),
                    "result_path": str(atomic_result),
                    "result_sha256": sha(atomic_result.read_bytes()),
                }
            )
        atomic_manifest = root / "atomic.jsonl"
        atomic_manifest.write_bytes(
            b"".join(canonical(row) + b"\n" for row in atomic_rows)
        )
        progress = {
            "schema_version": stream.TOPUP_PROGRESS_SCHEMA,
            "status": "continue",
            "atomic_ok_rows": 2,
            "target_atomic_ok": 1000,
            "atomic_manifest": str(atomic_manifest),
            "atomic_manifest_sha256": sha(atomic_manifest.read_bytes()),
            "progress_digest": None,
        }
        progress["progress_digest"] = stream._object_digest(
            progress, omit="progress_digest"
        )
        progress_path = root / "progress.json"
        progress_path.write_bytes(stream._pretty(progress))
        args = argparse.Namespace(
            planner_input=planner_input,
            planner_input_sha256=sha(planner_input.read_bytes()),
            planner_root=planner_root,
            atomic_manifest=atomic_manifest,
            progress=progress_path,
            wan_root=wan_root,
            wan_batch_root=batch_root,
            admission_root=admission_root,
            output_input=input_root / "batch_0000.jsonl",
            output_receipt=receipt_root / "batch_0000.json",
            batch_tag="batch_0000",
            previous_admission_batch=None,
            resume=False,
        )
        return args, iids

    def test_batch_zero_two_passes_are_admitted_before_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, iids = self._fixture(Path(directory))
            with mock.patch(
                "motive.goku_atomic_motion_qwen.validate_passed_row",
                side_effect=lambda value: dict(value),
            ):
                self.assertEqual(stream.admit_batch(args), 0)
            receipt = json.loads(args.output_receipt.read_text())
            self.assertEqual(receipt["batch_iids"], iids[:2])
            self.assertEqual(receipt["batch_rows"], 2)
            self.assertEqual(receipt["cumulative_rows"], 2)
            self.assertEqual(
                [json.loads(line)["iid"] for line in args.output_input.read_bytes().splitlines()],
                iids[:2],
            )
            self.assertEqual(json.loads(args.progress.read_text())["status"], "continue")
            self.assertEqual(json.loads(args.progress.read_text())["target_atomic_ok"], 1000)
            before = {
                path: path.read_bytes()
                for path in (
                    args.output_input,
                    args.output_receipt,
                    *(args.admission_root / f"{iid}.json" for iid in iids[:2]),
                )
            }
            args.resume = True
            with mock.patch(
                "motive.goku_atomic_motion_qwen.validate_passed_row",
                side_effect=lambda value: dict(value),
            ):
                self.assertEqual(stream.admit_batch(args), 0)
            self.assertEqual(
                before,
                {path: path.read_bytes() for path in before},
            )

    def test_successful_sample_receives_all_instruction_and_role_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, iids = self._fixture(Path(directory))
            with mock.patch(
                "motive.goku_atomic_motion_qwen.validate_passed_row",
                side_effect=lambda value: dict(value),
            ):
                stream.admit_batch(args)
            batch = json.loads(args.output_receipt.read_text())
            batch_root = Path(batch["wan_batch_root"])
            batch_root.mkdir()
            contract = batch_root / "watch_contract.json"
            contract.write_text("{}\n")
            for iid in iids[:2]:
                admission = json.loads((args.admission_root / f"{iid}.json").read_text())
                sample = batch_root / "samples" / iid / "samples" / iid
                sample.mkdir(parents=True)
                (sample / "source_video.mp4").write_bytes(
                    Path(admission["source_video"]).read_bytes()
                )
                (sample / "edit_instruction.txt").write_text(
                    admission["wan_generation_prompt"]
                )
                (sample / "preview.mp4").write_bytes(b"target")
                (sample / "result.json").write_text("{}\n")
            terminal = {
                "schema_version": stream.V16_TERMINAL_SCHEMA,
                "status": "complete",
                "watch_contract_sha256": sha(contract.read_bytes()),
                "expected_iids": iids[:2],
                "qwen_ok_iids": iids[:2],
                "qwen_error_iids": [],
                "wan_success_iids": iids[:2],
                "wan_error_iids": [],
                "completed_at_utc": "2026-08-05T00:00:00+00:00",
                "terminal_digest": None,
            }
            terminal["terminal_digest"] = stream._object_digest(
                terminal, omit="terminal_digest"
            )
            (batch_root / "watcher_terminal.json").write_bytes(stream._pretty(terminal))
            metadata_args = argparse.Namespace(
                admission_batch=args.output_receipt,
                admission_root=args.admission_root,
                resume=False,
            )
            self.assertEqual(stream.materialize_metadata(metadata_args), 0)
            sample = batch_root / "samples" / iids[0] / "samples" / iids[0]
            for name in (
                "source_video.mp4",
                "edit_instruction.txt",
                "wan_generation_prompt.txt",
                "atomic_action_instruction.txt",
                "camera_instruction.txt",
                "preservation_instruction.txt",
                "full_edit_instruction.txt",
                "planner_passed.jsonl",
                "atomic_result.json",
                "atomic_admission.json",
                "atomic_sample_metadata.json",
            ):
                self.assertTrue((sample / name).is_file(), name)
            metadata = json.loads((sample / "atomic_sample_metadata.json").read_text())
            self.assertEqual(
                metadata["primary_training_label_field"],
                "atomic_action_instruction",
            )
            self.assertFalse(metadata["wan_generation_prompt_is_training_label"])
            self.assertEqual(
                metadata["edit_instruction_txt_role"],
                "generation_only_not_training_label",
            )
            self.assertEqual(
                (sample / "edit_instruction.txt").read_bytes(),
                (sample / "wan_generation_prompt.txt").read_bytes(),
            )
            metadata_args.resume = True
            self.assertEqual(stream.materialize_metadata(metadata_args), 0)
            terminal_path = args.wan_root / "stream_terminal.json"
            publish_args = argparse.Namespace(
                latest_admission_batch=args.output_receipt,
                atomic_manifest=args.atomic_manifest,
                output=terminal_path,
                resume=False,
            )
            self.assertEqual(stream.publish_terminal(publish_args), 0)
            closed = json.loads(terminal_path.read_text())
            self.assertEqual(closed["wan_success_iids"], iids[:2])
            self.assertEqual(closed["wan_error_iids"], [])

    def test_launcher_dispatches_inside_topup_before_smoke_or_target_gate(self) -> None:
        text = PIPELINE.read_text(encoding="utf-8")
        result = subprocess.run(
            ["bash", "-n", str(PIPELINE)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        controller = text.split("controller_main() {", 1)[1]
        progress = controller.index('publish-progress --candidates "${planner_input}"')
        dispatch = controller.index('dispatch_atomic_delta "${tag}"')
        smoke_gate = controller.index('if (( atomic_ok >= smoke_rows', dispatch)
        final_break = controller.index(
            "atomic_ok >= required_atomic_target && wan_success >= required_new",
            smoke_gate,
        )
        self.assertLess(progress, dispatch)
        self.assertLess(dispatch, smoke_gate)
        self.assertLess(smoke_gate, final_break)
        self.assertIn("MOTIVE_FULL_MOTION_WAN_EXPAND_AFTER_QWEN_TERMINAL=1", text)
        self.assertIn("immediate_wan_admissions", text)
        self.assertNotIn("build_wan_subset", text)


if __name__ == "__main__":
    unittest.main()

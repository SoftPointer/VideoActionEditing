from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE = REPO_ROOT / "tmp" / "launch_goku_atomic1000_g8_pipeline.sh"
SMOKE_EXISTING_JOB = REPO_ROOT / "tmp" / "launch_goku_atomic_smoke8_existing_job.sh"
WAN = REPO_ROOT / "tmp" / "launch_fullmotion_v16_wan_stream.sh"


def embedded_helper(text: str, marker: str) -> str:
    opening = f"<<'{marker}'"
    line_start = text.index(opening)
    start = text.index("\n", line_start) + 1
    end = text.index(f"\n{marker}\n", start)
    return text[start:end] + "\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class GokuAtomic1000G8PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = PIPELINE.read_text(encoding="utf-8")
        cls.smoke_existing_job = SMOKE_EXISTING_JOB.read_text(encoding="utf-8")
        cls.wan = WAN.read_text(encoding="utf-8")

    def test_shell_syntax_and_all_embedded_python_compile(self) -> None:
        for path in (PIPELINE, SMOKE_EXISTING_JOB, WAN):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        markers = (
            "PY_REUSE_PREFLIGHT",
            "PY_API",
            "PY_INPUT",
            "PY_HOLDER_READY",
            "PY_PLANNER_INPUT",
            "PY_CONTRACT",
            "PY_VERIFY_PLANNER",
            "PY_TOPUP_MATERIALIZED",
            "PY_TOPUP_SELECTION",
            "PY_TOPUP_PROGRESS",
            "PY_WAN_WATCHER_PID",
            "PY_FINAL",
            "PY_RELEASE",
            "PY_LAUNCH_INDEX",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                compile(embedded_helper(self.pipeline, marker), marker, "exec")
        for marker in ("PY_API", "PY_PLANNER", "PY_SELECTION", "PY_PROGRESS"):
            with self.subTest(smoke_marker=marker):
                compile(
                    embedded_helper(self.smoke_existing_job, marker), marker, "exec"
                )

    def test_frozen_geometry_and_sixteen_worker_topology(self) -> None:
        for marker in (
            "MOTIVE_ATOMIC_EXPECTED_ROWS:-1235",
            "MOTIVE_ATOMIC_SMOKE_ROWS:-8",
            "MOTIVE_ATOMIC_PLANNER_WORKERS:-16",
            "MOTIVE_ATOMIC_LABEL_WORKERS:-16",
            "MOTIVE_ATOMIC_SMOKE_BATCH_ROWS:-16",
            "MOTIVE_ATOMIC_FULL_BATCH_ROWS:-128",
            "MOTIVE_ATOMIC_MINIMUM_FINAL_SUCCESS:-1000",
            "planner_workers == 16 && atomic_workers == 16",
            "workers == nodes * 2",
            '--ntasks="${tasks_per_node}" --ntasks-per-node="${tasks_per_node}"',
            "--gpus-per-task=4 --gpu-bind=none",
            "worker=$((ATOMIC_WORKER_BASE + SLURM_LOCALID))",
            "visible=0,1,2,3",
            "visible=4,5,6,7",
        ):
            self.assertIn(marker, self.pipeline)

    def test_dual4_rocr_visibility_and_cache_contract(self) -> None:
        for text in (self.pipeline, self.smoke_existing_job):
            with self.subTest(path="pipeline" if text is self.pipeline else "smoke"):
                self.assertIn("unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES", text)
                self.assertIn('export ROCR_VISIBLE_DEVICES="${visible}"', text)
                self.assertNotIn(
                    'export HIP_VISIBLE_DEVICES="${visible}" ROCR_VISIBLE_DEVICES="${visible}"',
                    text,
                )
                self.assertIn("dual4 ROCm probe failed", text)
                self.assertIn("/bin/bash --noprofile --norc -c", text)
                self.assertIn("PYTORCH_KERNEL_CACHE_PATH", text)
                self.assertIn("MIOPEN_USER_DB_PATH", text)
                self.assertIn("MIOPEN_CUSTOM_CACHE_DIR", text)
                self.assertIn("HF_HUB_OFFLINE=1", text)

    def test_phase_order_is_dynamic_topup_then_wan(self) -> None:
        controller = self.pipeline.split("controller_main() {", 1)[1]
        order = [
            controller.index('select-batch --candidates "${planner_input}"'),
            controller.index('run_qwen_grid planner "${batch_input}"'),
            controller.index('materialize_atomic_iteration "${tag}"'),
            controller.index("run_qwen_grid atomic \"${atomic_full_input}\""),
            controller.index('publish-progress --candidates "${planner_input}"'),
            controller.index('dispatch_atomic_delta "${tag}"'),
            controller.index('publish-gate --candidates "${planner_input}"'),
            controller.index('publish-terminal'),
            controller.index("publish_final_dataset"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertIn("MOTIVE_FULL_MOTION_WAN_EXPAND_AFTER_QWEN_TERMINAL=1", self.pipeline)
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_ADAPTER_MODULE", self.pipeline)
        self.assertIn("immediate_wan_admissions", self.pipeline)
        self.assertIn("while (( cursor < planner_expected_rows ))", controller)
        self.assertIn("candidate pool exhausted", controller)
        self.assertNotIn("planner smoke8 requires all eight rows to pass", self.pipeline)
        self.assertNotIn("lines[:8]", self.pipeline)
        self.assertNotIn("requires 8/8 pass", self.smoke_existing_job)
        self.assertIn("while (( cursor < 1235 ))", self.smoke_existing_job)
        self.assertIn("selected 8 final atomic passes", self.smoke_existing_job.lower())
        self.assertIn("pool exhausted", self.smoke_existing_job)

    def test_wan_stream_accepts_explicit_atomic_receipt_adapter(self) -> None:
        self.assertIn("MOTIVE_FULL_MOTION_QWEN_ADAPTER_MODULE", self.wan)
        self.assertIn('importlib.import_module(sys.argv[4])', self.wan)
        self.assertIn('"terminal_adapter_module": qwen_adapter_module', self.wan)
        self.assertIn("load_non_production_preview_manifest", self.wan)

    def test_full_target_is_cumulative_final_atomic_passes_not_input_rows(self) -> None:
        controller = self.pipeline.split("controller_main() {", 1)[1]
        self.assertIn('--target-atomic-ok "${required_atomic_target}"', controller)
        self.assertIn(
            'atomic_ok >= required_atomic_target && wan_success >= required_new',
            controller,
        )
        self.assertIn('[[ "${progress_status}" != pool_exhausted ]]', controller)
        self.assertNotIn('[[ "${progress_status}" == target_reached ]]', controller)
        self.assertIn('atomic_ok >= smoke_rows', controller)
        self.assertIn('stage=smoke', controller)
        self.assertIn('stage=full', controller)
        self.assertIn('batch_size=${smoke_batch_rows}', controller)
        self.assertIn('batch_size=${full_batch_rows}', controller)
        self.assertNotIn('--num-rows 1000', controller)

    def test_final_join_separates_primary_label_from_generation_prompt(self) -> None:
        helper = embedded_helper(self.pipeline, "PY_FINAL")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_root = root / "atomic"
            planner_root = root / "planner"
            wan_root = root / "wan"
            iid = "sample001"
            action = "Have the dog lift the ball with its mouth."
            camera = "Keep the camera fixed."
            preservation = "Preserve appearance and scene content."
            generation_prompt = (
                "From frame 0 to frame 40 the dog lowers its head; "
                "from frame 40 to frame 80 it lifts the ball."
            )
            atomic_result = atomic_root / "rows" / iid / "result.json"
            planner_passed = planner_root / "passed" / f"{iid}.jsonl"
            sample = wan_root / "samples" / iid / "samples" / iid
            for path in (atomic_result.parent, planner_passed.parent, sample):
                path.mkdir(parents=True, exist_ok=True)
            composite = f"{action} {camera} {preservation}"
            atomic_result.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "rewrite": {},
                        "atomic_action_instruction": action,
                        "atomic_action_instruction_sha256": sha(action.encode()),
                        "camera_instruction": camera,
                        "camera_instruction_sha256": sha(camera.encode()),
                        "preservation_instruction": preservation,
                        "preservation_instruction_sha256": sha(preservation.encode()),
                        "full_edit_instruction": composite,
                        "full_edit_instruction_sha256": sha(composite.encode()),
                    }
                ),
                encoding="utf-8",
            )
            planner_passed.write_text(
                json.dumps(
                    {
                        "edit_instruction": generation_prompt,
                        "edit_instruction_sha256": sha(generation_prompt.encode()),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (sample / "source_video.mp4").write_bytes(b"source")
            (sample / "preview.mp4").write_bytes(b"target")
            (sample / "result.json").write_text("{}\n", encoding="utf-8")
            atomic_manifest = root / "atomic.jsonl"
            atomic_manifest.write_text(
                json.dumps(
                    {
                        "iid": iid,
                        "result_path": str(atomic_result.resolve()),
                        "result_sha256": sha(atomic_result.read_bytes()),
                        "atomic_action_instruction": action,
                        "atomic_action_instruction_sha256": sha(action.encode()),
                        "camera_instruction": camera,
                        "preservation_instruction": preservation,
                        "full_edit_instruction": composite,
                        "source_generation_provenance": {
                            "frame_gridded_prompt": generation_prompt,
                            "frame_gridded_prompt_sha256": sha(
                                generation_prompt.encode()
                            ),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metadata = sample / "atomic_sample_metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "motive-goku-atomic-wan-sample-metadata-v1"
                        ),
                        "iid": iid,
                        "primary_training_label_field": (
                            "atomic_action_instruction"
                        ),
                        "wan_generation_prompt_is_training_label": False,
                        "edit_instruction_txt_role": (
                            "generation_only_not_training_label"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            terminal = {
                "schema_version": "motive-goku-atomic-wan-stream-terminal-v1",
                "status": "complete",
                "latest_admission_batch": str(root / "batch_0000.json"),
                "latest_admission_batch_sha256": "a" * 64,
                "atomic_manifest": str(atomic_manifest),
                "atomic_manifest_sha256": sha(atomic_manifest.read_bytes()),
                "expected_iids": [iid],
                "wan_success_iids": [iid],
                "wan_error_iids": [],
                "records": [
                    {
                        "iid": iid,
                        "status": "success",
                        "sample_dir": str(sample),
                        "sample_metadata": str(metadata),
                        "sample_metadata_sha256": sha(metadata.read_bytes()),
                    }
                ],
                "terminal_digest": None,
            }
            terminal["terminal_digest"] = sha(
                json.dumps(
                    {key: value for key, value in terminal.items() if key != "terminal_digest"},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            )
            (wan_root / "stream_terminal.json").write_text(
                json.dumps(terminal) + "\n", encoding="utf-8"
            )
            output = root / "dataset.jsonl"
            summary = root / "summary.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    helper,
                    str(atomic_manifest),
                    str(atomic_root),
                    str(planner_root),
                    str(wan_root),
                    str(output),
                    str(summary),
                    "1",
                    "",
                    "",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["atomic_action_instruction"], action)
            self.assertEqual(row["primary_training_label_field"], "atomic_action_instruction")
            self.assertEqual(row["full_edit_instruction"], composite)
            self.assertEqual(row["wan_generation_prompt"], generation_prompt)
            self.assertNotEqual(
                row["atomic_action_instruction_sha256"],
                row["wan_generation_prompt_sha256"],
            )
            self.assertEqual(
                row["wan_edit_instruction_txt_role"],
                "generation_prompt_not_primary_training_label",
            )
            self.assertTrue(json.loads(summary.read_text())["wan_generation_prompt_is_separate"])

    def test_final_join_rejects_frame_grid_in_primary_action(self) -> None:
        helper = embedded_helper(self.pipeline, "PY_FINAL")
        self.assertIn("primary atomic label contains timing/stitching", helper)
        self.assertIn(r"frames?", helper)
        self.assertIn("atomic_action_instruction", helper)

    def test_reuse_requires_visual_video_audit_and_excludes_its_iids(self) -> None:
        reuse = embedded_helper(self.pipeline, "PY_REUSE_PREFLIGHT")
        selector = embedded_helper(self.pipeline, "PY_PLANNER_INPUT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            target = root / "target.mp4"
            audit = root / "audit.json"
            source.write_bytes(b"source")
            target.write_bytes(b"target")
            audit.write_text(
                json.dumps(
                    {
                        "schema_version": "motive-goku-atomic-video-audit-v1",
                        "iid": "reuse",
                        "single_causal_event_visible": True,
                        "matches_atomic_action": True,
                        "no_additional_independent_action": True,
                        "camera_behavior_match": True,
                        "preserves_initial_frame_identity": True,
                        "no_unrequested_appearance_edit": True,
                        "overall_verdict": "pass",
                        "confidence": "high",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selected = root / "selected.jsonl"
            selected.write_text(
                '{"iid":"keep"}\n{"iid":"reuse"}\n', encoding="utf-8"
            )
            action = "Have the dog lift the ball."
            prompt = "From frame 0 to frame 80, lift the ball."
            row = {
                "schema_version": "motive-goku-atomic-reuse-approved-v1",
                "iid": "reuse",
                "lineage": "legacy_v17_reused",
                "visual_video_audit_pass": True,
                "visual_video_audit": str(audit),
                "visual_video_audit_sha256": sha(audit.read_bytes()),
                "atomic_action_instruction": action,
                "atomic_action_instruction_sha256": sha(action.encode()),
                "camera_instruction": "Keep the camera fixed.",
                "preservation_instruction": "Preserve appearance and scene content.",
                "source_video": str(source),
                "source_video_sha256": sha(source.read_bytes()),
                "target_video": str(target),
                "target_video_sha256": sha(target.read_bytes()),
                "wan_generation_prompt": prompt,
                "wan_generation_prompt_sha256": sha(prompt.encode()),
            }
            manifest = root / "reuse.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            checked = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    reuse,
                    str(manifest),
                    sha(manifest.read_bytes()),
                    str(selected),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout.strip(), "1")
            planner_input = root / "planner.jsonl"
            selected_result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    selector,
                    str(selected),
                    str(manifest),
                    str(planner_input),
                    "0",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(selected_result.returncode, 0, selected_result.stderr)
            self.assertEqual(json.loads(planner_input.read_text())["iid"], "keep")

            row["visual_video_audit_pass"] = False
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    reuse,
                    str(manifest),
                    sha(manifest.read_bytes()),
                    str(selected),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_create_only_resume_hup_and_release_contracts(self) -> None:
        for marker in (
            "MOTIVE_ATOMIC_RESUME",
            "create-only run root exists",
            "pipeline_contract.json",
            'flock -n "${controller_lock_fd}"',
            "another atomic1000 controller is active",
            "trap '' HUP",
            "existing final artifact differs",
            "release_holder_${job_id}.json",
            "motive-goku-atomic1000-allocation-release-v1",
        ):
            self.assertIn(marker, self.pipeline)

    def test_launcher_never_submits_cancels_or_contacts_remote_hosts(self) -> None:
        for forbidden in ("ssh ", "scp ", "sbatch", "scancel", "scontrol cancel"):
            self.assertNotIn(forbidden, self.pipeline)

    def test_missing_bindings_fail_before_slurm_queries(self) -> None:
        result = subprocess.run(
            ["/bin/bash", str(PIPELINE)],
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing explicit binding", result.stderr)
        self.assertNotIn("scontrol", result.stderr)


if __name__ == "__main__":
    unittest.main()

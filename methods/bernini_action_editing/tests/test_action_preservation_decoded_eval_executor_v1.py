from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import action_preservation_decoded_eval_executor_v1 as executor
import action_preservation_decoded_eval_launcher_v1 as launcher
import action_preservation_decoded_eval_plan_v1 as plan


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def action_contract(iid: str) -> dict:
    description = f"Complete the fitted action for source {iid}, then hold the terminal pose."
    row = {
        "schema_version": plan.ACTION_REVIEW_CONTRACT_SCHEMA,
        "action_order_description": description,
        "action_order_description_sha256": plan.text_sha256(description),
        "expected_onset_frame_min": 4,
        "expected_onset_frame_max": 20,
        "terminal_hold_start_frame_min": 65,
        "terminal_hold_end_frame": 80,
        "full_video_frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
    }
    row["contract_digest"] = plan.object_sha256(row)
    return row


def sources() -> list[dict]:
    rows = []
    for index, iid in enumerate(plan.FITTED_IIDS):
        instruction = f"Perform the fitted action for source {iid}."
        rows.append(
            {
                "iid": iid,
                "source_video_sha256": digest(f"source:{iid}"),
                "source_receipt_sha256": digest(f"source-receipt:{iid}"),
                "instruction": instruction,
                "instruction_sha256": plan.text_sha256(instruction),
                "action_review_contract": action_contract(iid),
                "seed": 2026081801 + index,
            }
        )
    return rows


def checkpoints() -> list[dict]:
    return [
        {
            "arm": arm,
            "checkpoint_step": step,
            "checkpoint_receipt_sha256": digest(f"checkpoint:{arm}:{step}"),
            "adapter_sha256": digest(f"adapter:{arm}:{step}"),
        }
        for arm in plan.ARMS
        for step in plan.CHECKPOINT_STEPS
    ]


def pins() -> dict[str, str]:
    return {key: digest(key) for key in plan.PIN_FIELDS}


def read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_only_bytes(path: pathlib.Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def exact_probe() -> dict:
    return {
        "video_stream_count": 1,
        "frame_count": 81,
        "fps_num": 25,
        "fps_den": 1,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "frame_timestamp_times": [f"{index / 25:.6f}" for index in range(81)],
    }


class BundleTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = pathlib.Path(self.temporary.name).resolve()
        self.evaluation_root = self.parent / "evaluation"
        input_spec = plan.build_input_spec(
            evaluation_id="preservation-v2-local-executor-stub",
            evaluation_root=self.evaluation_root,
            pins=pins(),
            sources=sources(),
            checkpoints=checkpoints(),
        )
        bundle = plan.build_bundle(input_spec)
        plan.publish_bundle(bundle)
        self.bundle = executor.load_published_bundle(self.evaluation_root)
        self.decoder_identity = {
            "path": "/stub/pinned-decoder-adapter",
            "sha256": digest("decoder-adapter"),
        }
        self.ffprobe_identity = {
            "path": "/stub/pinned-ffprobe",
            "sha256": digest("ffprobe"),
        }
        self.physical_bindings_identity = {
            "path": "/stub/physical-bindings.json",
            "sha256": digest("physical-bindings"),
        }

    def successful_decoder(self, request_path: pathlib.Path, output_path: pathlib.Path):
        request = read_json(request_path)
        payload = (
            "stub-exact81@25fps:" + request["task_id"] + ":" + request["input_digest"]
        ).encode("utf-8")
        create_only_bytes(output_path, payload)
        return {"return_code": 0, "stdout": b"stub ok\n", "stderr": b""}

    @staticmethod
    def successful_probe(_video_path: pathlib.Path):
        return exact_probe()

    def run_holder(self, job_id: str, *, decoder=None, prober=None):
        return executor.execute_shard(
            bundle=self.bundle,
            holder_job_id=job_id,
            decoder_identity=self.decoder_identity,
            ffprobe_identity=self.ffprobe_identity,
            physical_bindings_identity=self.physical_bindings_identity,
            run_decoder=decoder or self.successful_decoder,
            probe_video=prober or self.successful_probe,
            verify_tools=False,
        )


class ExecutorTests(BundleTestCase):
    def test_production_holder_job_and_hostname_are_physically_bound(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "136719"}), mock.patch.object(
            executor.socket, "gethostname", return_value="auh7-1b-gpu-306"
        ):
            authority = executor.local_holder_execution_authority("136719")
        self.assertTrue(authority["exact_holder_match"])
        self.assertEqual(authority["observed_hostname"], "auh7-1b-gpu-306")

        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "136719"}), mock.patch.object(
            executor.socket, "gethostname", return_value="auh7-1b-gpu-299"
        ), self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError, "planned holder"
        ):
            executor.local_holder_execution_authority("136719")

    def test_stub_end_to_end_executes_exact_264_once_with_receipts(self):
        summaries = [self.run_holder(row["job_id"]) for row in plan.HOLDER_ROWS]
        self.assertEqual(sum(row["planned_task_count"] for row in summaries), 264)
        self.assertEqual(sum(row["attempted_task_count"] for row in summaries), 264)
        self.assertEqual(sum(row["success_count"] for row in summaries), 264)
        self.assertEqual(sum(row["failure_count"] for row in summaries), 0)
        self.assertTrue(all(row["automatic_retry_count"] == 0 for row in summaries))
        self.assertTrue(all(row["training_loss_read_or_used"] is False for row in summaries))
        self.assertTrue(all(row["execution_backend"] == "injected_stub" for row in summaries))
        self.assertTrue(all(row["tool_files_verified"] is False for row in summaries))
        self.assertTrue(
            all(row["scientific_promotion_authorized"] is False for row in summaries)
        )

        candidate_outputs = list((self.evaluation_root / "candidates").rglob("*.mp4"))
        control_outputs = list(
            (self.evaluation_root / "frozen_base_controls").rglob("*.mp4")
        )
        self.assertEqual(len(candidate_outputs), 256)
        self.assertEqual(len(control_outputs), 8)
        self.assertTrue(
            all((path.stat().st_mode & 0o222) == 0 for path in candidate_outputs)
        )
        self.assertTrue(
            all((path.stat().st_mode & 0o222) == 0 for path in control_outputs)
        )

        execution_root = self.evaluation_root / executor.EXECUTION_DIRECTORY
        task_roots = []
        for holder in plan.HOLDER_ROWS:
            shard_root = execution_root / holder["job_id"]
            summary = read_json(shard_root / executor.SUMMARY_FILENAME)
            self.assertEqual(summary["success_count"], 66)
            task_roots.extend((shard_root / "tasks").iterdir())
        self.assertEqual(len(task_roots), 264)
        self.assertTrue(
            all((root / executor.INPUT_RECEIPT_FILENAME).is_file() for root in task_roots)
        )
        self.assertTrue(
            all((root / executor.OUTPUT_RECEIPT_FILENAME).is_file() for root in task_roots)
        )
        self.assertTrue(
            all(not (root / executor.FAILURE_RECEIPT_FILENAME).exists() for root in task_roots)
        )
        input_receipts = [
            read_json(root / executor.INPUT_RECEIPT_FILENAME) for root in task_roots
        ]
        policy_counts = {}
        for receipt in input_receipts:
            policy = receipt["task_record"]["onset_policy"]["name"]
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            self.assertEqual(
                receipt["task_record"]["seed"],
                next(
                    source["seed"]
                    for source in self.bundle["input_spec"]["sources"]
                    if source["iid"] == receipt["task_record"]["iid"]
                ),
            )
        self.assertEqual(policy_counts, {"none": 132, "hard1_every_step": 132})

        sample_root = task_roots[0]
        input_receipt = read_json(sample_root / executor.INPUT_RECEIPT_FILENAME)
        process_receipt = read_json(sample_root / executor.PROCESS_RECEIPT_FILENAME)
        output_receipt = read_json(sample_root / executor.OUTPUT_RECEIPT_FILENAME)
        output_path = self.evaluation_root / output_receipt["output_relpath"]
        validated = executor.validate_output_receipt(
            output_receipt,
            input_receipt=input_receipt,
            process_receipt=process_receipt,
            output_path=output_path,
        )
        self.assertEqual(validated["probe"]["frame_count"], 81)
        self.assertEqual(
            (validated["probe"]["fps_num"], validated["probe"]["fps_den"]),
            (25, 1),
        )
        self.assertFalse(validated["training_loss_read_or_used"])
        self.assertFalse(validated["retry_allowed"])
        self.assertFalse(input_receipt["direct_exec_shell"])
        self.assertIn("BASH_ENV", input_receipt["subprocess_environment_denylist"])
        self.assertEqual(input_receipt["execution_backend"], "injected_stub")
        self.assertFalse(input_receipt["tool_files_verified"])
        self.assertEqual(validated["execution_backend"], "injected_stub")

    def test_failure_is_retained_and_holder_cannot_retry(self):
        calls = []

        def decoder(request_path: pathlib.Path, output_path: pathlib.Path):
            request = read_json(request_path)
            calls.append(request["task_id"])
            create_only_bytes(output_path, ("partial:" + request["task_id"]).encode())
            if len(calls) == 1:
                return {"return_code": 19, "stdout": b"", "stderr": b"stub failure"}
            return {"return_code": 0, "stdout": b"ok", "stderr": b""}

        summary = self.run_holder("136719", decoder=decoder)
        self.assertEqual(summary["attempted_task_count"], 66)
        self.assertEqual(summary["success_count"], 65)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(len(calls), 66)
        failed = next(row for row in summary["results"] if row["status"] == "failure")
        failed_root = (
            self.evaluation_root
            / executor.EXECUTION_DIRECTORY
            / "136719"
            / "tasks"
            / failed["task_id"]
        )
        self.assertTrue((failed_root / executor.STAGING_VIDEO_FILENAME).is_file())
        self.assertTrue((failed_root / executor.STDERR_FILENAME).is_file())
        failure = read_json(failed_root / executor.FAILURE_RECEIPT_FILENAME)
        input_receipt = read_json(failed_root / executor.INPUT_RECEIPT_FILENAME)
        failure = executor.validate_failure_receipt(
            failure, input_receipt=input_receipt
        )
        self.assertEqual(failure["failure_kind"], "decoder_nonzero")
        self.assertTrue(failure["failure_artifacts_retained"])
        self.assertFalse(failure["retry_allowed"])
        with self.assertRaisesRegex(executor.DecodedEvaluationExecutorError, "retries"):
            self.run_holder("136719", decoder=decoder)
        self.assertEqual(len(calls), 66)

    def test_non_exact_media_fails_closed_without_final_output(self):
        probe_calls = 0

        def prober(_video_path: pathlib.Path):
            nonlocal probe_calls
            probe_calls += 1
            value = exact_probe()
            if probe_calls == 1:
                value["frame_count"] = 80
            return value

        summary = self.run_holder("136141", prober=prober)
        self.assertEqual(summary["success_count"], 65)
        self.assertEqual(summary["failure_count"], 1)
        failure_row = next(row for row in summary["results"] if row["status"] == "failure")
        task_root = (
            self.evaluation_root
            / executor.EXECUTION_DIRECTORY
            / "136141"
            / "tasks"
            / failure_row["task_id"]
        )
        failure = read_json(task_root / executor.FAILURE_RECEIPT_FILENAME)
        self.assertEqual(failure["failure_kind"], "media_validation_failed")
        self.assertTrue(failure["staging_artifact"]["exists"])
        self.assertFalse(failure["final_artifact"]["exists"])
        self.assertFalse(
            (self.evaluation_root / failure_row["output_relpath"]).exists()
        )

    def test_decoder_cannot_mutate_sealed_input_receipt(self):
        calls = 0

        def decoder(request_path: pathlib.Path, output_path: pathlib.Path):
            nonlocal calls
            calls += 1
            request = read_json(request_path)
            create_only_bytes(output_path, ("stub:" + request["task_id"]).encode())
            if calls == 1:
                os.chmod(request_path, 0o600)
                request_path.write_text("{}\n", encoding="utf-8")
            return {"return_code": 0, "stdout": b"ok", "stderr": b""}

        summary = self.run_holder("136140", decoder=decoder)
        self.assertEqual(summary["success_count"], 65)
        self.assertEqual(summary["failure_count"], 1)
        failed = next(row for row in summary["results"] if row["status"] == "failure")
        failure = read_json(
            self.evaluation_root
            / executor.EXECUTION_DIRECTORY
            / "136140"
            / "tasks"
            / failed["task_id"]
            / executor.FAILURE_RECEIPT_FILENAME
        )
        self.assertEqual(failure["failure_kind"], "input_receipt_mutated")

    def test_tampered_shard_or_unknown_holder_is_rejected_before_attempt(self):
        with self.assertRaisesRegex(executor.DecodedEvaluationExecutorError, "outside"):
            self.run_holder("999999")
        hostile = copy.deepcopy(self.bundle)
        hostile["shards"]["136309"]["tasks"].pop()
        hostile["shards"]["136309"]["shard_digest"] = plan.object_sha256(
            {
                key: value
                for key, value in hostile["shards"]["136309"].items()
                if key != "shard_digest"
            }
        )
        with self.assertRaisesRegex(executor.DecodedEvaluationExecutorError, "closure"):
            executor.execute_shard(
                bundle=hostile,
                holder_job_id="136309",
                decoder_identity=self.decoder_identity,
                ffprobe_identity=self.ffprobe_identity,
                physical_bindings_identity=self.physical_bindings_identity,
                run_decoder=self.successful_decoder,
                probe_video=self.successful_probe,
                verify_tools=False,
            )

    def test_ffprobe_parser_and_exact_gate(self):
        observed = executor.parse_ffprobe_json(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "nb_read_frames": "81",
                        "avg_frame_rate": "25/1",
                    }
                ],
                "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
                "frames": [
                    {
                        "media_type": "video",
                        "best_effort_timestamp_time": f"{index / 25:.6f}",
                    }
                    for index in range(81)
                ],
            }
        )
        self.assertEqual(executor.validate_probe_result(observed), observed)
        observed["fps_num"] = 24
        with self.assertRaisesRegex(executor.DecodedEvaluationExecutorError, "full81"):
            executor.validate_probe_result(observed)

    def test_variable_rate_pts_is_rejected_even_with_average_25fps(self):
        observed = exact_probe()
        observed["frame_timestamp_times"][40] = "1.610000"
        with self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError, "variable-rate"
        ):
            executor.validate_probe_result(observed)

    def test_subprocess_environment_drops_shell_startup_injection(self):
        with mock.patch.dict(
            os.environ,
            {
                "BASH_ENV": "/hostile/bash-env",
                "ENV": "/hostile/sh-env",
                "ZDOTDIR": "/hostile/zsh",
                "PYTHONSTARTUP": "/hostile/python-startup",
                "PYTHONINSPECT": "1",
            },
        ):
            environment = executor.sanitized_subprocess_environment()
        for key in executor.SUBPROCESS_ENV_DENYLIST:
            self.assertNotIn(key, environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_production_mode_rejects_unbound_stub_tools(self):
        decoder_path = self.parent / "stub-decoder"
        ffprobe_path = self.parent / "stub-ffprobe"
        hostile_bash_env = self.parent / "hostile-bash-env"
        create_only_bytes(
            decoder_path,
            b"#!/bin/bash\nset -euo pipefail\n"
            b"request=\noutput=\nwhile (($#)); do case \"$1\" in "
            b"--request) request=$2; shift 2;; --output) output=$2; shift 2;; "
            b"*) exit 64;; esac; done\n"
            b"[[ -f \"$request\" && ! -e \"$output\" ]]\n"
            b"set -C\nprintf '%s' 'stub-mp4-exact81-at-25fps' > \"$output\"\n",
        )
        create_only_bytes(
            ffprobe_path,
            b"#!/bin/bash\nset -euo pipefail\n"
            b"printf '%s\\n' '{\"streams\":[{\"codec_type\":\"video\","
            b"\"nb_read_frames\":\"81\",\"avg_frame_rate\":\"25/1\"}],"
            b"\"format\":{\"format_name\":\"mov,mp4,m4a,3gp,3g2,mj2\"}}'\n",
        )
        create_only_bytes(hostile_bash_env, b"exit 77\n")
        os.chmod(decoder_path, 0o555)
        os.chmod(ffprobe_path, 0o555)
        decoder_identity = {
            "path": str(decoder_path),
            "sha256": plan.file_sha256(decoder_path),
        }
        ffprobe_identity = {
            "path": str(ffprobe_path),
            "sha256": plan.file_sha256(ffprobe_path),
        }
        with mock.patch.dict(
            os.environ, {"BASH_ENV": str(hostile_bash_env), "ENV": str(hostile_bash_env)}
        ), self.assertRaisesRegex(
            executor.DecodedEvaluationExecutorError, "physical bindings.*does not exist"
        ):
            executor.execute_shard(
                bundle=self.bundle,
                holder_job_id="136309",
                decoder_identity=decoder_identity,
                ffprobe_identity=ffprobe_identity,
                physical_bindings_identity=self.physical_bindings_identity,
                run_decoder=executor.subprocess_decoder_runner(decoder_path),
                probe_video=executor.ffprobe_video_prober(ffprobe_path),
                verify_tools=True,
            )


class LauncherTests(BundleTestCase):
    def test_launcher_only_publishes_four_manual_local_commands(self):
        identities = {
            "python_identity": {"path": "/stub/python", "sha256": digest("python")},
            "executor_identity": {
                "path": "/stub/executor.py",
                "sha256": digest("executor"),
            },
            "decoder_identity": self.decoder_identity,
            "ffprobe_identity": self.ffprobe_identity,
        }
        value = launcher.build_launch_manifest(
            bundle=self.bundle,
            launch_root=self.parent / "local-launch-plan",
            physical_bindings_identity=self.physical_bindings_identity,
            verify_tools=False,
            **identities,
        )
        self.assertEqual(len(value["commands"]), 4)
        self.assertFalse(value["command_execution_performed"])
        self.assertFalse(value["subprocess_spawned"])
        self.assertFalse(value["network_used"])
        self.assertFalse(value["remote_launch_performed"])
        self.assertFalse(value["automatic_retry"])
        self.assertFalse(value["training_loss_read_or_used"])
        self.assertFalse(value["tool_files_verified"])
        self.assertEqual(value["execution_backend"], "injected_stub_plan")
        self.assertIn("BASH_ENV", value["subprocess_environment_denylist"])
        flattened = [token for row in value["commands"] for token in row["argv"]]
        self.assertTrue(all(token not in flattened for token in ("ssh", "srun", "sbatch")))
        output = launcher.publish_launch_manifest(value, bundle=self.bundle)
        self.assertTrue(output.is_file())
        with self.assertRaisesRegex(launcher.DecodedEvaluationLauncherError, "not fresh"):
            launcher.publish_launch_manifest(value, bundle=self.bundle)

    def test_launcher_command_tamper_is_rejected_even_when_resigned(self):
        value = launcher.build_launch_manifest(
            bundle=self.bundle,
            launch_root=self.parent / "tampered-launch",
            python_identity={"path": "/stub/python", "sha256": digest("python")},
            executor_identity={"path": "/stub/executor.py", "sha256": digest("executor")},
            decoder_identity=self.decoder_identity,
            ffprobe_identity=self.ffprobe_identity,
            physical_bindings_identity=self.physical_bindings_identity,
            verify_tools=False,
        )
        value["commands"][0]["argv"][0] = "ssh"
        unsigned = {key: item for key, item in value.items() if key != "launch_manifest_digest"}
        value["launch_manifest_digest"] = launcher.object_sha256(unsigned)
        with self.assertRaisesRegex(launcher.DecodedEvaluationLauncherError, "command"):
            launcher.validate_launch_manifest(value, bundle=self.bundle)


if __name__ == "__main__":
    unittest.main()

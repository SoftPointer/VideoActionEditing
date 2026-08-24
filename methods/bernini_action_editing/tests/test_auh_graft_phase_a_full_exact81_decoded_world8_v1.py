#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_run_graft_phase_a_full_exact81_decoded_world8_v1.sbatch"
)
WRAPPER = (
    METHOD_ROOT
    / "scripts/auh_submit_graft_phase_a_full_exact81_decoded_world8_v1.sh"
)
PLAN = METHOD_ROOT / "assets/graft_phase_a_full_exact81_decoded_world8_plan_v1.json"
FIELD_PLAN = METHOD_ROOT / "assets/graft_phase_a_field14_exact40_world8_plan_v1.json"
ACTIVE_PLAN = METHOD_ROOT / "assets/graft_phase_a_active14_transaction_world8_plan_v1.json"
FULL_CORE = METHOD_ROOT / "graft_phase_a_full_exact81_decoded_v1.py"
FULL_RUNNER = METHOD_ROOT / "run_graft_phase_a_full_exact81_decoded_gpu_v1.py"
FIELD_CORE = METHOD_ROOT / "graft_phase_a_field14_exact40_v1.py"
FIELD_RUNNER = METHOD_ROOT / "run_graft_phase_a_field14_exact40_gpu_v1.py"
ACTIVE_CORE = METHOD_ROOT / "train_graft_phase_a_active14_transaction_v1.py"
ACTIVE_RUNNER = METHOD_ROOT / "run_graft_phase_a_active14_transaction_gpu_v1.py"
SHORT_RUNNER = METHOD_ROOT / "run_graft_phase_a_a_lite_short_gpu_v1.py"
REPO_ROOT = METHOD_ROOT.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readonly(text: str, name: str) -> str:
    match = re.search(rf"^readonly {re.escape(name)}=([^\n]+)$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing readonly {name}")
    return match.group(1)


class FullExact81AUHClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.plan_raw = PLAN.read_bytes()
        cls.plan = json.loads(cls.plan_raw.decode("ascii"))

    def test_exact_resources_and_runnable_afterok_submission(self) -> None:
        for path in (LAUNCHER, WRAPPER):
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        directives = {
            line.strip()
            for line in self.launcher.splitlines()
            if line.startswith("#SBATCH")
        }
        for required in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --time=72:00:00",
        ):
            self.assertIn(required, directives)
        self.assertEqual(
            self.plan["resources"],
            {
                "nodes": 1,
                "ntasks": 1,
                "gpus": 8,
                "cpus_per_task": 64,
                "memory_gib": 256,
                "time_limit_hours": 72,
            },
        )
        self.assertIn('f"--dependency=afterok:{active14_job_id}"', self.wrapper)
        self.assertEqual(self.wrapper.count("completed=subprocess.run"), 1)
        self.assertNotIn("--hold", self.wrapper)
        self.assertNotIn("scontrol release", self.wrapper)
        self.assertNotIn("--export=ALL", self.wrapper)
        self.assertIn('"user_hold":False', self.wrapper)
        self.assertIn('"state_expected":"PD_dependency_or_resource"', self.wrapper)

    def test_all_embedded_python_is_syntactically_valid(self) -> None:
        for path, source in ((LAUNCHER, self.launcher), (WRAPPER, self.wrapper)):
            programs = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", source, re.DOTALL)
            self.assertGreaterEqual(len(programs), 1, path.name)
            for index, program in enumerate(programs):
                try:
                    ast.parse(program, filename=f"{path.name}:heredoc{index}")
                except SyntaxError as error:
                    self.fail(str(error))

    def test_plan_binds_same_process_replay_and_operational_only_decode(self) -> None:
        self.assertEqual(
            self.plan_raw,
            json.dumps(
                self.plan,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n",
        )
        self.assertEqual(
            self.plan["state_continuity"],
            {
                "checkpoint_available_from_dependency": False,
                "dependency_transports_weights": False,
                "same_process_from_base_required": True,
                "short_29_38_replayed": True,
                "field14_0_39_replayed": True,
                "active14_26_39_replayed": True,
                "decode_continuation_before_restore": True,
            },
        )
        self.assertEqual(
            self.plan["replay_order"],
            [
                "sealed-source-and-base-checkpoint-admission",
                "short-updates-29-38-confirmation-parity",
                "field14-no-grad-indices-0-39",
                "active14-transaction-indices-26-39",
                "full-action-exact40-rollout",
                "vae-exact81-decode",
                "world8-receipt-and-atomic-publish",
            ],
        )
        sampling = self.plan["sampling"]
        self.assertEqual(
            (sampling["frame_count"], sampling["fps_fraction"], sampling["num_inference_steps"]),
            (81, "25/1", 40),
        )
        self.assertEqual(sampling["initial_state"], "fresh-source-keyed-standard-gaussian")
        self.assertFalse(sampling["target_video_used"])
        self.assertFalse(sampling["clean_source_initial_latent_used"])
        self.assertFalse(sampling["best_of_n"])
        decoder = self.plan["decoder"]
        self.assertEqual(decoder["output_frame_count"], 81)
        self.assertEqual(decoder["output_fps_fraction"], "25/1")
        self.assertTrue(decoder["latent_artifact_required"])
        self.assertTrue(decoder["mp4_artifact_required"])
        self.assertFalse(decoder["semantic_evaluator_present"])
        self.assertEqual(
            [
                (
                    row["family"],
                    row["source_input_hw"],
                    row["derived_bucket_hw"],
                )
                for row in self.plan["families"]
            ],
            [
                ("dog", [704, 736], [480, 496]),
                ("human", [896, 704], [544, 432]),
            ],
        )
        self.assertTrue(all(value is False for value in self.plan["authority"].values()))
        self.assertFalse(self.plan["checkpoint_policy"]["dependency_checkpoint_consumed"])
        self.assertFalse(self.plan["checkpoint_policy"]["checkpoint_written"])
        dependency = self.plan["dependency"]
        self.assertEqual(dependency["job_id"], "133534")
        self.assertEqual(dependency["kind"], "afterok")
        self.assertEqual(
            dependency["receipt_sha256_policy"],
            "derive-from-stable-sealed-file-after-afterok",
        )
        self.assertEqual(
            dependency["receipt_path"],
            "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
            "VideoEditing/VideoEdit_experiments/bernini_graft_v1_20260810/"
            "phase_a_active14_transaction_world8_v1/runs/"
            "source-037f7a8-launcher-8b6d08b-r2/receipt.json",
        )

    def test_runtime_pins_and_exact29_git_source_closure(self) -> None:
        runtime = self.plan["runtime"]
        self.assertEqual(runtime["full_exact81_core_sha256"], sha256(FULL_CORE))
        self.assertEqual(runtime["full_exact81_runner_sha256"], sha256(FULL_RUNNER))
        self.assertEqual(runtime["active14_core_sha256"], sha256(ACTIVE_CORE))
        self.assertEqual(runtime["active14_runner_sha256"], sha256(ACTIVE_RUNNER))
        expected_direct = {
            "required_full_core_sha": FULL_CORE,
            "required_full_runner_sha": FULL_RUNNER,
            "required_active_core_sha": ACTIVE_CORE,
            "required_active_runner_sha": ACTIVE_RUNNER,
            "required_field_core_sha": FIELD_CORE,
            "required_field_runner_sha": FIELD_RUNNER,
            "required_short_runner_sha": SHORT_RUNNER,
        }
        for name, path in expected_direct.items():
            self.assertEqual(readonly(self.launcher, name), sha256(path), name)

        source_commit = readonly(self.launcher, "required_full_source_commit")
        self.assertRegex(source_commit, r"^[0-9a-f]{40}$")
        manifest_block = self.launcher.split("expected={", 1)[1].split("}\nfiles=", 1)[0]
        closure_files = dict(re.findall(r'^"([^"]+\.py)":"([0-9a-f]{64})",?$', manifest_block, re.MULTILINE))
        self.assertEqual(len(closure_files), 29)
        self.assertEqual(
            set(closure_files),
            set(re.findall(r'"((?:tools/)?[A-Za-z0-9_]+\.py)"', manifest_block)),
        )
        for relative, expected_sha in closure_files.items():
            completed = subprocess.run(
                [
                    "git",
                    "show",
                    f"{source_commit}:methods/bernini_action_editing/{relative}",
                ],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, relative)
            self.assertEqual(hashlib.sha256(completed.stdout).hexdigest(), expected_sha, relative)
        for evidence in (
            '"source_git_commit"',
            'manifest.get("source_git_commit")!=source_commit',
            "len(members)!=29",
            "member.issym()",
            "member.islnk()",
            '".." in p.parts',
            "member.name!=p.as_posix()",
            "os.O_EXCL",
            "compile(data,member.name,\"exec\")",
        ):
            self.assertIn(evidence, self.launcher)

    def test_two_phase_transaction_fail_close_and_deep_postflight(self) -> None:
        for evidence in (
            "replay_active14_for_downstream",
            "prepare=prepare, finalize=finalize",
            "active14_updates_complete_downstream_prepare_pending",
            '"published": False',
            '"finalize_completed": True',
            '"active14_commit_receipt_digest"',
            '"preparation_receipt_digest"',
            "failed-postcommit-job",
            "failed-staging-job",
            "failed-launcher-{kind}-job",
            "publish_after_outer_close",
            "publish_deferred_until_active14_outer_close",
            "checkpoint_content_pre_sha256",
            "checkpoint_content_post_sha256",
            "WORLD8 deep recomputation differs",
            "full81.assemble_world8_result",
            "full81.canonical_json_bytes(recomputed)",
            "full81.validate_exact40_trace",
            "full81.validate_artifact_record",
            "rank_local_upstream_receipt_bindings",
            "rank_local_upstream_receipt_bindings_digest",
            "output exact artifact inventory differs",
            "latent endpoint identity differs",
            "_probe_with_frozen_ffprobe",
            '!=(81,25,1,25,1,expected_height,expected_width)',
            "codec_content_interpreted_for_semantics",
            "plan/media bucket geometry differs",
            "tensor_content_sha",
            "full81.assert_no_elevated_authority",
        ):
            self.assertIn(evidence, self.launcher + FULL_RUNNER.read_text(encoding="utf-8"))
        for forbidden in ("torch.save", "target_video_path", "selected_candidate"):
            self.assertNotIn(forbidden, FULL_RUNNER.read_text(encoding="utf-8"))

    def test_secure_wrapper_hash_chain_and_numeric_placeholders_fail_closed(self) -> None:
        self.assertIn("pass_fds=(launcher_fd,sbatch_fd)", self.wrapper)
        self.assertIn("reserve_receipt(parent_fd,receipt_name)", self.wrapper)
        self.assertIn("os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW", self.wrapper)
        self.assertIn("submission receipt already exists", self.wrapper)
        self.assertIn("plan canonical bytes differ", self.wrapper)
        self.assertIn('active_plan.get("field14_dependency"', self.wrapper)
        self.assertIn("dependency job ID pins differ", self.wrapper)
        self.assertIn("output and log roots alias", self.wrapper)
        self.assertNotIn(
            "GRAFT_FULL81_UPSTREAM_FIELD14_RECEIPT_SHA256", self.wrapper
        )
        self.assertIn(
            'field14_receipt_sha="$(file_sha256 "${field14_receipt}")"',
            self.launcher,
        )
        self.assertIn(
            'active14_receipt_sha="$(file_sha256 "${active14_receipt}")"',
            self.launcher,
        )
        self.assertIn(
            'value.get("field14_receipt_file_sha256")!=field_receipt_sha',
            self.launcher,
        )
        self.assertIn("active14 parent receipt digest differs", self.launcher)
        self.assertIn("active14 parent runtime differs", self.launcher)
        self.assertIn("active14 parent runner evidence differs", self.launcher)
        self.assertIn("GRAFT_FULL81_UPSTREAM_ACTIVE14_RECEIPT", self.wrapper)
        self.assertIn("active_anchor_fd,active_anchor_identity", self.wrapper)
        self.assertIn("upstream_active14_receipt_may_be_absent_at_submission", self.wrapper)
        self.assertIn("upstream_active14_receipt_sha256_policy", self.wrapper)
        self.assertIn("upstream_active14_parent_receipt_sha256", self.launcher)
        self.assertIn("upstream_active14_parent_receipt_digest", self.launcher)
        self.assertIn(
            "commit-579e84c-recursive-runtime-import-closure-v1",
            self.launcher,
        )
        self.assertIn("upstream_field14_receipt_may_be_absent_at_submission", self.wrapper)
        self.assertIn("open_stable_directory(field_anchor", self.wrapper)
        self.assertLess(
            self.wrapper.index("os.unlink(receipt_name,dir_fd=parent_fd)"),
            self.wrapper.rindex("os.close(parent_fd); os.close(log_parent_fd)"),
        )
        self.assertEqual(readonly(self.launcher, "required_field14_plan_sha"), sha256(FIELD_PLAN))
        active_plan_pin = readonly(self.launcher, "required_active14_plan_sha")
        full_plan_pin = readonly(self.launcher, "required_full81_plan_sha")
        launcher_pin = readonly(self.wrapper, "required_launcher_sha256")
        if not active_plan_pin.startswith("__"):
            self.assertEqual(active_plan_pin, sha256(ACTIVE_PLAN))
            self.assertEqual(readonly(self.wrapper, "required_active14_plan_sha256"), active_plan_pin)
        if not full_plan_pin.startswith("__"):
            self.assertEqual(full_plan_pin, sha256(PLAN))
            self.assertEqual(readonly(self.wrapper, "required_full81_plan_sha256"), full_plan_pin)
        if not launcher_pin.startswith("__"):
            self.assertEqual(launcher_pin, sha256(LAUNCHER))
        active_job = readonly(self.launcher, "required_active14_job_id")
        field_job = readonly(self.launcher, "required_field14_job_id")
        if active_job.startswith("__") or field_job.startswith("__"):
            self.assertIn("dependency job ID pins differ", self.launcher)
            self.assertIn("dependency job ID pins differ", self.wrapper)
        else:
            self.assertEqual(self.plan["dependency"]["job_id"], active_job)
            self.assertEqual(readonly(self.wrapper, "required_active14_job_id"), active_job)
            self.assertEqual(readonly(self.wrapper, "required_field14_job_id"), field_job)


if __name__ == "__main__":
    unittest.main()

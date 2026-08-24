#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts/auh_run_graft_phase_a_field14_exact40_world8_v1.sbatch"
WRAPPER = METHOD_ROOT / "scripts/auh_submit_graft_phase_a_field14_exact40_world8_v1.sh"
CORE = METHOD_ROOT / "graft_phase_a_field14_exact40_v1.py"
RUNNER = METHOD_ROOT / "run_graft_phase_a_field14_exact40_gpu_v1.py"
PLAN = METHOD_ROOT / "assets/graft_phase_a_field14_exact40_world8_plan_v1.json"
SHORT_RUNNER = METHOD_ROOT / "run_graft_phase_a_a_lite_short_gpu_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Field14AUHLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.plan = json.loads(PLAN.read_text(encoding="ascii"))

    def test_shell_syntax_and_exact_resources(self) -> None:
        for path in (LAUNCHER, WRAPPER):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
        directives = {
            line.strip() for line in self.launcher.splitlines() if line.startswith("#SBATCH")
        }
        for required in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=64",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --time=48:00:00",
        ):
            self.assertIn(required, directives)
        self.assertIn("--nproc-per-node=8", self.launcher)
        self.assertEqual(
            self.plan["resources"],
            {
                "cpus_per_task": 64,
                "gpus": 8,
                "memory_gib": 256,
                "nodes": 1,
                "ntasks": 1,
                "time_limit_hours": 48,
            },
        )

    def test_hash_chain_is_exact_and_has_no_placeholders(self) -> None:
        combined = "\n".join(
            (
                self.launcher,
                self.wrapper,
                RUNNER.read_text(encoding="utf-8"),
                PLAN.read_text(encoding="ascii"),
            )
        )
        self.assertIsNone(re.search(r"__[A-Z][A-Z0-9_]*__", combined))
        dependency_job_id = self.plan["afterok_dependency"]["job_id"]
        self.assertEqual(dependency_job_id, "133524")
        self.assertEqual(combined.count(dependency_job_id), 3)
        self.assertIn(
            f"readonly dependency_job_id={dependency_job_id}", self.wrapper
        )
        expected = {
            "required_plan_sha": _sha(PLAN),
            "required_core_sha": _sha(CORE),
            "required_field_runner_sha": _sha(RUNNER),
        }
        for name, digest in expected.items():
            match = re.search(rf"readonly {name}=([0-9a-f]{{64}})", self.launcher)
            self.assertIsNotNone(match, name)
            self.assertEqual(match.group(1), digest)
        launcher_match = re.search(
            r"readonly required_launcher_sha256=([0-9a-f]{64})", self.wrapper
        )
        self.assertIsNotNone(launcher_match)
        self.assertEqual(launcher_match.group(1), _sha(LAUNCHER))

    def test_exact25_closure_and_safe_extraction(self) -> None:
        names = set(
            re.findall(
                r'"((?:tools/)?[A-Za-z0-9_]+\.py)"',
                self.launcher.split("expected_names={", 1)[1].split("}", 1)[0],
            )
        )
        self.assertEqual(len(names), 25)
        self.assertIn("graft_phase_a_field14_exact40_v1.py", names)
        self.assertIn("run_graft_phase_a_field14_exact40_gpu_v1.py", names)
        self.assertIn("run_graft_phase_a_a_lite_short_gpu_v1.py", names)
        self.assertIn(
            "readonly required_field_source_commit="
            "f9ef982e976ad19ed81ed075d33c9221952945e4",
            self.launcher,
        )
        self.assertIn(
            'manifest.get("selection")!="commit-f9ef982-recursive-runtime-import-closure-v1"',
            self.launcher,
        )
        for required in (
            "len(members)!=25",
            "member.issym()",
            "member.islnk()",
            '".." in path.parts',
            "os.O_EXCL",
            '"O_NOFOLLOW"',
            "compile(payload",
        ):
            self.assertIn(required, self.launcher)

    def test_runner_cli_and_result_postflight_cover_field14(self) -> None:
        for flag in (
            "--expected-runner-sha256",
            "--expected-field14-core-sha256",
            "--expected-field14-runner-sha256",
            "--expected-field14-source-commit",
            "--plan-path",
            "--expected-plan-sha256",
            "--ack-two-update-diagnostic-no-checkpoint-no-scientific-claim",
            "--ack-exact40-no-grad-diagnostic-no-checkpoint-no-scientific-claim",
        ):
            self.assertIn(flag, self.launcher)
        for evidence in (
            "all_eight_field14_results",
            "both_sp4_arms_exact_field_hash_and_metric_consensus",
            "schedule_indices\")!=list(range(40))",
            "one_index_admitted_hashed_and_released_before_next",
            "cross_index_compensation_used",
            "cross_index_selection_used",
            "trainable_bytes_unchanged_during_sweep",
            "all_eight_full_field14_receipts_deeply_validated",
            "both_sp4_per_index_field_hash_and_metric_consensus_recomputed",
            "checkpoint_content_full_rehash_pre_and_post",
            "checkpoint_payload_returned",
        ):
            self.assertIn(evidence, self.launcher)
        self.assertNotIn("torch.save", self.launcher)

    def test_wrapper_submits_once_with_afterok_and_distinct_output(self) -> None:
        self.assertEqual(self.wrapper.count("completed=subprocess.run"), 1)
        self.assertIn('readonly dependency_job_id=133524', self.wrapper)
        self.assertIn('f"--dependency=afterok:{dependency_job_id}"', self.wrapper)
        self.assertIn("queue_gate_only\":True", self.wrapper)
        self.assertIn("inherits_weights\":False", self.wrapper)
        self.assertIn("submission receipt already exists", self.wrapper)
        self.assertIn("os.O_EXCL", self.wrapper)
        self.assertIn("reserve_receipt(parent_fd,receipt_name)", self.wrapper)
        self.assertIn("mode_0444_is_terminal_success_transition", self.wrapper)
        self.assertIn("pass_fds=(launcher_fd,sbatch_fd)", self.wrapper)
        self.assertIn("python_wrapper_sbatch_launcher_retained_and_revalidated", self.wrapper)
        self.assertIn("/usr/bin/env -i", self.wrapper)
        self.assertNotIn("--export=ALL", self.wrapper)
        self.assertNotIn("scontrol release", self.wrapper)
        self.assertNotIn("131339", self.wrapper)
        self.assertNotIn("131358", self.wrapper)

    def test_retained_roots_deep_postflight_and_terminal_publish(self) -> None:
        for evidence in (
            'exec 8<"${output_root}"',
            'exec 9<"${rank_log_root}"',
            "8<&- 9<&- &",
            "root_fd=os.dup(inherited_root_fd)",
            "log_fd=os.dup(inherited_log_fd)",
            "os.O_NOFOLLOW,dir_fd=current",
            "checkpoint_content_pre_sha256",
            "checkpoint_content_post_sha256",
            'set(world)!=world_keys',
            'set(sweep)!=sweep_keys',
            'set(representative)!=arm_keys',
            'set(result)!=result_keys',
            "provisional receipt transaction differs",
            "terminal receipt transition differs",
        ):
            self.assertIn(evidence, self.launcher)
        self.assertIn("set(item) not in (set(AUTHORITY),set(CONSUMER_AUTHORITY))", self.launcher)
        self.assertIn("false_authority(plan.get(\"authority\"),AUTHORITY12", self.wrapper)
        self.assertIn("false_authority(terminal.get(\"authority\"),AUTHORITY9", self.wrapper)

    def test_frozen_wrapper_is_fail_closed_without_exact_environment(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-p", str(WRAPPER)],
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(b"exactly twenty Field14 interface variables", completed.stderr)

    def test_short_r3_sources_are_exactly_pinned(self) -> None:
        self.assertEqual(
            _sha(SHORT_RUNNER),
            "4b98bc520c7b90f71a3fe1d58e5e2e2f96d05465611f4c4bb4143e6cc51a62c4",
        )
        old_launcher = METHOD_ROOT / "scripts/auh_run_graft_phase_a_a_lite_short_world8_v1.sbatch"
        old_wrapper = METHOD_ROOT / "scripts/auh_submit_graft_phase_a_a_lite_short_world8_v1.sh"
        self.assertEqual(
            _sha(old_launcher),
            "c62ee713e0309b6e0441b12375573d9e4cd7dc5ce94e5db652b8319dee2357a9",
        )
        self.assertEqual(
            _sha(old_wrapper),
            "aae2acd8a9a0ae8413d9a571134ab4067fbf9860401c5150763daf23b35b658d",
        )


if __name__ == "__main__":
    unittest.main()

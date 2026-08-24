from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = METHOD_ROOT / "scripts" / "auh_infer_source_noised_carrier_stage_b_two_holder_v1.sh"
RELEASE_ROOT = METHOD_ROOT / "releases" / "source_noised_carrier_stage_b_inference_r1"


class StageBInferenceTwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CONTROLLER.read_text(encoding="utf-8")

    def test_bash_syntax_and_usage_fail_closed(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        result = subprocess.run(
            [str(CONTROLLER)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_unique_release_authority_and_all_fourteen_members_are_pinned(self) -> None:
        archive = RELEASE_ROOT / "source.tar"
        manifest_path = RELEASE_ROOT / "source.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        revision = manifest["content_closure_sha1"]
        self.assertEqual(archive_sha, "646d5a9c73364db689b2592d3b1a4a486c2e9e032031f09e5be3a003845f3698")
        self.assertEqual(manifest_sha, "50f87120b08e4a576acd3ff44efdda699848db9c2d9d13336f5006431f418639")
        self.assertEqual(revision, "63eafa1b10f083eedf6bec316ad92fb3bedea17b")
        self.assertEqual(manifest["file_count"], 14)
        for value in (archive_sha, manifest_sha, revision):
            self.assertIn(value, self.source)
        self.assertIn(
            '[[ "${source_archive_sha}" == "${expected_release_archive_sha}" && "${source_manifest_sha}" == "${expected_release_manifest_sha}" && "${source_revision}" == "${expected_release_revision}" ]]',
            self.source,
        )
        for row in manifest["files"]:
            self.assertIn(row["sha256"], self.source)

    def test_four_world4_branches_are_serial_and_pair_verification_is_in_order(self) -> None:
        statements = [
            "run_branch registered_probes_frozen_base registered-probes frozen_base",
            "run_branch registered_probes_trained registered-probes trained",
            'verify_pair "${run_root}/registered_probes_frozen_base"',
            "run_branch full40_evolved_target_frozen_base full40-evolved-target-all40-route-extrapolation frozen_base",
            "run_branch full40_evolved_target_trained full40-evolved-target-all40-route-extrapolation trained",
            'verify_pair "${run_root}/full40_evolved_target_frozen_base"',
        ]
        positions = [self.source.index(value) for value in statements]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.source.count("run_branch registered_probes_"), 2)
        self.assertEqual(self.source.count("run_branch full40_evolved_target_"), 2)
        self.assertIn("--nnodes=2 --nproc_per_node=2", self.source)
        self.assertIn('--seed "${seed}"', self.source)
        self.assertNotIn("--anchor-action-video", self.source)
        self.assertNotIn("--expected-anchor-action-sha256", self.source)

    def test_child_signals_are_pid_identity_bound_and_parent_actions_forbidden(self) -> None:
        self.assertIn('ppid="$(proc_field "${pid}" 4)"', self.source)
        self.assertIn('[[ "${ppid}" == "$$"', self.source)
        self.assertIn('"$(basename -- "${exe}")" == srun', self.source)
        self.assertIn("pid_cmd_sha", self.source)
        self.assertIn("signal_owned_pid", self.source)
        forbidden = ("s" + "cancel", "scontrol " + "release", "scontrol " + "requeue", "p" + "kill", "kill" + "all")
        for value in forbidden:
            self.assertNotIn(value, self.source)
        direct_signals = [
            line.strip()
            for line in self.source.splitlines()
            if "kill -" in line and "kill -0" not in line
        ]
        self.assertEqual(
            direct_signals,
            ['signal_owned_pid() { if pid_identity_matches "$1"; then kill -"$2" "$1" 2>/dev/null || true; elif [[ -e "/proc/$1" ]]; then echo "REFUSE_SIGNAL pid=$1" >&2; fi; }'],
        )

    def test_resources_retained_parent_and_final_state_are_hard_gated(self) -> None:
        self.assertIn("readonly memory_peak_limit_bytes=55834574848", self.source)
        self.assertIn("--ntasks=1 --cpus-per-task=16 --mem=56G --gres=gpu:mi210:2", self.source)
        self.assertIn('sampled<int(limit)', self.source)
        self.assertIn('sacct<int(limit)', self.source)
        self.assertIn('assert_parent_running "${work_job0}"', self.source)
        self.assertIn('assert_parent_running "${work_job1}"', self.source)
        self.assertIn('assert_parent_running "${retained_job}"', self.source)
        self.assertNotIn('launch_child "${retained_job}"', self.source)
        self.assertNotIn('assert_remote_idle_once "${retained_job}"', self.source)
        self.assertLess(self.source.rindex("assert_idle_twice final"), self.source.rindex("assert_all_parents_running"))
        self.assertIn("parents_retained=135407,135411,135412", self.source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = METHOD_ROOT / "scripts" / "auh_train_source_self_role_repaint_two_holder_v1.sh"


class SourceSelfTwoHolderControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = CONTROLLER.read_text(encoding="utf-8")

    def test_shell_and_safe_usage_path(self) -> None:
        subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)
        result = subprocess.run(["bash", str(CONTROLLER)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)

    def test_exact_holder_phase_resources(self) -> None:
        for value in (
            "materialize_job=135412", "materialize_node=auh7-1b-gpu-293",
            "train_job0=135407", "train_node0=auh7-1b-gpu-260",
            "train_job1=135411", "train_node1=auh7-1b-gpu-214",
            "--cpus-per-task=16 --mem=24G --gres=gpu:mi210:1",
            "--cpus-per-task=16 --mem=56G --gres=gpu:mi210:2",
        ):
            self.assertIn(value, self.text)
        self.assertLess(self.text.index("pre-materialize"), self.text.index("pre-training"))

    def test_world4_is_explicit_and_rank_cache_pinned(self) -> None:
        for value in (
            "--nnodes=2", "--nproc_per_node=2", '--node_rank="${node_rank}"',
            "--parallel-topology world4-dp1-sp4", "--no_python", "BERNINI_HELDOUT_RANK_CACHE_TOKEN",
            "f416f8f687a61788d4f687e500dae04af677072ec205fb560016cb5249b4f9c5",
        ):
            self.assertIn(value, self.text)

    def test_release_is_archive_manifest_bound_without_git_clean_gate(self) -> None:
        for value in (
            "BERNINI_SSR_SOURCE_ARCHIVE", "BERNINI_SSR_SOURCE_MANIFEST",
            "bernini-source-self-training-release-v1", "content-closure-sha1",
            '--method-source-manifest-sha256 "${source_manifest_sha}"',
        ):
            self.assertIn(value, self.text)
        self.assertNotIn("BERNINI_ACTION_SOURCE_REPOSITORY", self.text)
        self.assertNotIn("git -C", self.text)
        self.assertNotIn("status --porcelain", self.text)

    def test_only_registered_child_srun_pids_are_signalled(self) -> None:
        lowered = self.text.lower()
        for forbidden in ("scan" + "cel", "scontrol " + "release", "scontrol " + "requeue", "p" + "kill", "kill" + "all"):
            self.assertNotIn(forbidden, lowered)
        self.assertIn('register_child_pid "${pid}" "${job}" "${node}"', self.text)
        self.assertEqual(self.text.count('kill -"$2" "$1"'), 1)
        self.assertIn("REFUSE_SIGNAL identity mismatch", self.text)
        self.assertIn("child_starttime", self.text)
        self.assertIn("child_cmdline_sha", self.text)

    def test_preflights_order_terminal_materializer_before_training(self) -> None:
        materializer_wait = self.text.index('wait "${materialize_pid}"')
        receipt_verify = self.text.index('assert {p.name for p in root.iterdir()} == {"dataset.parquet","receipt.json"}')
        train_launch = self.text.index('launch_train_node "${train_job0}"')
        self.assertLess(materializer_wait, receipt_verify)
        self.assertLess(receipt_verify, train_launch)
        self.assertIn("child_preflight 1", self.text)
        self.assertIn("child_preflight 2", self.text)
        self.assertIn("assert_idle_twice", self.text)
        self.assertIn("assert_master_port_free", self.text)
        self.assertIn("BERNINI_SSR_EXISTING_MATERIALIZED", self.text)
        self.assertIn('if [[ -z "${existing_materialized}" ]]', self.text)
        self.assertIn('r["dataset"]["rows"] == 2', self.text)

    def test_memory_peak_and_engineering_only_postflight(self) -> None:
        for value in (
            "memory.current", "resolve_cgroup2_memory_current", "/proc/self/mountinfo", "55834574848",
            "bernini-ssr-child-memory-crosscheck-v1", "sacct_max_rss_bytes",
            "COMPLETE_ENGINEERING_CANARY_ONLY", 'r["optimizer_steps"]==1',
            'r["distributed"]["profile"]=="world4-dp1-sp4"',
            '"scientific_claim_authorized"', '"long_training_automatically_submitted"',
        ):
            self.assertIn(value, self.text)
        self.assertIn('"${status}" == available', self.text)
        self.assertIn('"${peak}" -lt "${memory_peak_limit_bytes}"', self.text)
        self.assertIn('sampled < bound and sacct_peak < bound', self.text)
        self.assertNotIn("memory.usage_in_bytes", self.text)
        self.assertLess(
            self.text.index('cgroup_counter="$(resolve_cgroup2_memory_current'),
            self.text.index('"$@" &'),
        )
        self.assertIn('fields["node_rank"] == expected_node_rank', self.text)
        self.assertGreater(
            self.text.rindex("assert_all_parents_running"),
            self.text.index("verify_step_memory"),
        )

    def test_existing_materialized_mode_has_no_materializer_step_dependency(self) -> None:
        self.assertIn('jobs=("${train_job0}" "${train_job1}")', self.text)
        self.assertIn('[[ -n "${existing_materialized}" ]] || jobs=', self.text)
        self.assertIn("parquet_pin, materialization_receipt_pin, materialization_digest_pin", self.text)
        self.assertIn('r["dataset"]["parquet_sha256"]==parquet_pin', self.text)
        self.assertIn("materialization_mode=reused_sealed_existing", self.text)


if __name__ == "__main__":
    unittest.main()

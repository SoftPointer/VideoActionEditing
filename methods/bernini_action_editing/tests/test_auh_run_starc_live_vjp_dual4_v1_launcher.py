from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "auh_run_starc_live_vjp_dual4_v1.sbatch"


class AUHSTARCLiveVJPDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_bash_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_one_node_all_eight_is_exactly_two_concurrent_sp4(self) -> None:
        for token in (
            "#SBATCH --nodes=1",
            "#SBATCH --gres=gpu:mi210:8",
            'run_candidate sp4-a "0,1,2,3"',
            'run_candidate sp4-b "4,5,6,7"',
            "--nproc_per_node=4",
            '>"${output_root}/logs/sp4-a.log"',
            '>"${output_root}/logs/sp4-b.log"',
            'wait "${sp4_a_pid}"',
            'wait "${sp4_b_pid}"',
        ):
            self.assertIn(token, self.source)
        self.assertEqual(self.source.count("--nproc_per_node=4"), 1)

    def test_extracted_method_owned_loader_executes_live_run(self) -> None:
        for token in (
            "run_starc_live_vjp_sp4_v1.py",
            '"${loader}" "${common_args[@]}"',
            "common_args=(\n  run",
            "--candidate-manifest",
            "--current-clean-latent",
            "--native-noise",
            "--source-video",
            "--instruction-file",
            "--noop-caption-file",
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("compose", self.source)
        self.assertNotIn("mechanism receipt", self.source.lower())

    def test_archive_is_scanned_before_restricted_extraction(self) -> None:
        scan = self.source.index('with tarfile.open(sys.argv[1], "r:*") as handle:')
        extract = self.source.index("tar --no-same-owner --no-same-permissions -xf")
        self.assertLess(scan, extract)
        for token in (
            'or ".." in path.parts',
            "or member.issym()",
            "or member.islnk()",
            "or member.isdev()",
            "or member.isfifo()",
            "source archive contains an unsafe method member",
            "source archive lacks STARC live-VJP closure",
        ):
            self.assertIn(token, self.source)

    def test_source_loader_candidate_model_critic_and_master_are_hash_bound(self) -> None:
        for token in (
            "git get-tar-commit-id",
            "running launcher differs from source archive",
            "loader_sha256=",
            "--expected-loader-source-sha256",
            "--expected-source-archive-sha256",
            "--source-git-revision",
            "--expected-candidate-manifest-sha256",
            "--expected-materializer-master-sha256",
            "--expected-critic-checkpoint-sha256",
            "--expected-critic-checkpoint-receipt-sha256",
            "--expected-critic-config-receipt-sha256",
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
        ):
            self.assertIn(token, self.source)

    def test_both_failures_propagate_and_receipts_are_postflight_checked(self) -> None:
        for token in (
            "one or more live VJP SP4 groups failed",
            "composite receipt missing",
            "composite receipt mode differs",
            "bernini-starc-current-rv2v-live-vjp-composite-binding-v2",
            "durable source archive changed",
            "gradient audit failed",
        ):
            self.assertIn(token, self.source)

    def test_authority_is_explicitly_false_and_no_submission_is_performed(self) -> None:
        for token in (
            "--ack-mechanism-probe-only",
            "--ack-no-editor-parameter-or-update",
            "--ack-no-scientific-or-action-editing-claim",
            "optimizer=false authority=false",
            'row.get("editor_optimizer_authorized") is not False',
            'row.get("scientific_critic_claim_authorized") is not False',
        ):
            self.assertIn(token, self.source)
        self.assertNotIn("sbatch ", self.source)
        self.assertNotIn("srun ", self.source)


if __name__ == "__main__":
    unittest.main()

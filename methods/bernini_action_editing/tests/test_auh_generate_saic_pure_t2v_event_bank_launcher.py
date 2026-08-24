from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_generate_saic_pure_t2v_event_bank_all8_v1.sbatch"
)


class AUHSAICPureT2VEventBankLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_syntax_is_valid(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_launcher_uses_all8_as_two_concurrent_world4_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.source)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.source)
        self.assertGreaterEqual(self.source.count("--nproc_per_node=4"), 1)
        self.assertIn("dog_pid=$!", self.source)
        self.assertIn("human_pid=$!", self.source)
        self.assertIn('wait "${dog_pid}"', self.source)
        self.assertIn('wait "${human_pid}"', self.source)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn('value.get("num_attention_heads") != 12', self.source)
        self.assertIn("12 % 4", self.source)

    def test_launcher_is_exact81_text_only_and_never_passes_real_source(self) -> None:
        self.assertIn("attempts=60", self.source)
        self.assertIn("branches=forward,reverse,noop", self.source)
        self.assertIn("real_source_rgb=false", self.source)
        self.assertIn("source_latent=false", self.source)
        self.assertIn("source_noise=false", self.source)
        self.assertIn("materialize-proxies", self.source)
        self.assertIn("geometry-proxy-receipt.json", self.source)
        self.assertNotIn("--source-video", self.source)
        self.assertNotIn("source_video.mp4", self.source)

    def test_every_attempt_and_master_receipt_are_create_only_audited(self) -> None:
        self.assertIn('[[ ! -e "${candidate_output}"', self.source)
        self.assertIn("generate-attempt", self.source)
        self.assertIn("audit-bank", self.source)
        self.assertIn("event_audit=pending", self.source)
        self.assertIn("optimizer=false", self.source)
        self.assertIn("source_archive_sha256", self.source)
        self.assertIn("tools/ffprobe_pyav_saic.py", self.source)
        self.assertNotIn("SAIC_T2V_FFPROBE:?", self.source)
        self.assertIn("git get-tar-commit-id", self.source)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "scripts/auh_generate_self_imagined_owner_core2_dual4.sbatch"
)


class SelfImaginedOwnerLauncherTests(unittest.TestCase):
    def test_launcher_uses_all8_as_two_concurrent_sp4_groups(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn("launch_group dog 0,1,2,3", text)
        self.assertIn("launch_group human 4,5,6,7", text)
        self.assertIn("--nproc_per_node=4", text)
        self.assertIn("pure-T2V exact81 owners=2", text)
        self.assertNotIn("exact41", text)
        self.assertIn("audit-master", text)
        self.assertIn("semantic_audit=pending", text)

    def test_launcher_binds_running_spool_copy_to_repo_bytes(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('running_launcher="$(realpath -e -- "$0")"', text)
        self.assertIn("running launcher differs from repository launcher", text)
        self.assertIn("source_closure_sha256", text)
        self.assertIn("SIM_OWNER_SOURCE_ARCHIVE", text)
        self.assertIn("SIM_OWNER_SOURCE_ARCHIVE_SHA256", text)


if __name__ == "__main__":
    unittest.main()

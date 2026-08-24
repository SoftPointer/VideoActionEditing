from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_materialize_self_imagined_owner_core2_dual4_v1.sbatch"
)


class OwnerQuotientDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")

    def test_all8_is_two_concurrent_world4_groups(self) -> None:
        text = self.text
        self.assertIn("#SBATCH --gres=gpu:mi210:8", text)
        self.assertIn('run_group dog "0,1,2,3"', text)
        self.assertIn('run_group human "4,5,6,7"', text)
        self.assertGreaterEqual(text.count("--nproc_per_node=4"), 1)
        self.assertIn("WORLD4/SP4", text)
        self.assertIn("dog_pid=$!", text)
        self.assertIn("human_pid=$!", text)

    def test_signed_full81_preflight_happens_before_any_torchrun(self) -> None:
        text = self.text
        preflight = text.index('"${runtime}" preflight')
        torchrun = text.index("-m torch.distributed.run")
        dog_launch = text.index('run_group dog "0,1,2,3"')
        self.assertLess(preflight, torchrun)
        self.assertLess(preflight, dog_launch)
        self.assertIn("SIM_OWNER_FULL81_AUDIT_SIDECAR_SHA256", text)
        self.assertIn("SIM_OWNER_FULL81_AUDIT_PUBLIC_KEY_SHA256", text)
        self.assertIn("SIM_OWNER_JOB131524_MASTER_SHA256", text)
        self.assertNotIn("author-sidecar", text)

    def test_runtime_executes_only_from_authenticated_safe_archive(self) -> None:
        text = self.text
        self.assertIn("git get-tar-commit-id", text)
        self.assertIn("source archive contains an unsafe selected member", text)
        self.assertIn("member.issym()", text)
        self.assertIn("member.islnk()", text)
        self.assertIn('source_tree="${task_scratch}/source-tree"', text)
        self.assertIn(
            'runtime="${method_root}/materialize_self_imagined_owner_core2_v1.py"',
            text,
        )
        self.assertIn("running launcher differs from source archive", text)
        self.assertIn('find "${source_tree}" -type f -exec chmod a-w', text)
        self.assertIn(
            "score_pair_v5_t2v_energy_bank_frozen_d541801.py", text
        )

    def test_publication_is_atomic_and_only_allows_quotients_and_receipts(self) -> None:
        text = self.text
        self.assertIn("staging_root=", text)
        self.assertIn('mv -T -- "${staging_root}" "${output_root}"', text)
        self.assertIn("persistent output channel contains extra files", text)
        self.assertIn("owner-motion-quotients.safetensors", text)
        self.assertIn("owner-motion-specificity-receipt.json", text)
        self.assertIn("owner-core2-specificity-receipt.json", text)
        self.assertIn('>"${task_scratch}/logs/dog.log"', text)
        self.assertNotIn('>"${output_root}/logs', text)

    def test_no_seed_selection_or_averaging_and_no_training(self) -> None:
        text = self.text
        self.assertIn("seed_selection=false", text)
        self.assertIn("seed_averaging=false", text)
        self.assertNotIn("--optimizer", text)
        self.assertNotIn("--train", text)
        self.assertNotIn("mask_path", text)
        self.assertNotIn("swept_tube", text)


if __name__ == "__main__":
    unittest.main()

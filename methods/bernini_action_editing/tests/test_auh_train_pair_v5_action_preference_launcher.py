from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_train_pair_v5_action_preference_one_step.sbatch"
)
TRAINER = METHOD_ROOT / "train_pair_v5_action_preference.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))


class PairV5OneStepLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.trainer = TRAINER.read_text(encoding="utf-8")

    def test_uses_all_eight_gpus_as_world8_dp2_sp4(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertIn("--nproc_per_node=8", self.launcher)
        self.assertIn("topology=WORLD8/DP2xSP4", self.launcher)
        self.assertIn("init_parallel_state(ulysses_size=SP_SIZE)", self.trainer)
        self.assertEqual(8, __import__("train_pair_v5_action_preference").WORLD_SIZE)

    def test_canary_is_exact81_and_one_optimizer_step(self) -> None:
        self.assertIn("--max-steps 1", self.launcher)
        self.assertIn("--num-frames 81", self.launcher)
        self.assertIn("choices=(1,)", self.trainer)
        self.assertIn("REFERENCE_INDICES = (0, 27, 53, 80)", self.trainer)

    def test_launcher_exposes_no_proposal_or_privileged_input(self) -> None:
        expected = (
            "target=false proposal=false donor=false mask=false flow=false "
            "pose=false track=false trajectory=false"
        )
        self.assertIn(expected, self.launcher)
        self.assertNotIn("--proposal", self.launcher)
        self.assertNotIn("--mask", self.launcher)
        self.assertNotIn("--flow", self.launcher)

    def test_optional_frozen_cio_is_independent(self) -> None:
        self.assertIn("PAIR_V5_FROZEN_CIO_ADAPTER:-", self.launcher)
        self.assertIn("cio_args=()", self.launcher)
        self.assertIn("--frozen-cio-adapter", self.launcher)
        self.assertIn("active_in_student_and_reference", self.trainer)
        self.assertIn("optimized\": False", self.trainer)
        self.assertIn("frozen CIO checkpoint metadata/contract differs", self.trainer)
        self.assertIn("frozen CIO adapter after training", self.trainer)

    def test_preflight_covers_optimizer_authorization_cores(self) -> None:
        for name in (
            "test_pair_v5_candidate_evaluator_packet.py",
            "test_pair_v5_action_energy_calibration.py",
            "test_pair_v5_safe_pareto.py",
        ):
            self.assertIn(name, self.launcher)
        self.assertIn("pair_v5_candidate_evaluator_packet.py", self.launcher)

    def test_source_archive_and_manifest_are_hash_pinned(self) -> None:
        self.assertIn("PAIR_V5_SOURCE_ARCHIVE_SHA256", self.launcher)
        self.assertIn("PAIR_V5_PREFERENCE_MANIFEST_SHA256", self.launcher)
        self.assertIn("sha256sum \"${manifest}\"", self.launcher)
        self.assertIn("sha256sum \"${source_archive}\"", self.launcher)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.launcher)
        self.assertIn('git get-tar-commit-id <"${archive_copy}"', self.launcher)
        self.assertIn("checkpoint content verification failed", self.launcher)

    def test_authorization_precedes_source_activation_model_and_optimizer(self) -> None:
        manifest_gate = self.trainer.index("manifest = load_preference_manifest(")
        source_activation = self.trainer.index("legacy.activate_source_trees(")
        model_construction = self.trainer.index("renderer = BerniniRendererModel(config)")
        optimizer_construction = self.trainer.index("optimizer = torch.optim.AdamW(")
        self.assertLess(manifest_gate, source_activation)
        self.assertLess(source_activation, model_construction)
        self.assertLess(model_construction, optimizer_construction)

    def test_post_audit_does_not_promote_engineering_canary(self) -> None:
        self.assertIn('receipt.get("semantic_action_editing_success") is False', self.launcher)
        self.assertIn(
            'receipt.get("scientific_generalization_claim_authorized") is False',
            self.launcher,
        )
        self.assertIn("action_success_claim=false", self.launcher)


if __name__ == "__main__":
    unittest.main()

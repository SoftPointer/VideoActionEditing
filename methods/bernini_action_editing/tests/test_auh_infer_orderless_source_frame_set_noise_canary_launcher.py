from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts/auh_infer_orderless_source_frame_set_noise_canary_dual4.sbatch"
)
SOURCE = LAUNCHER.read_text(encoding="utf-8")


class AuhOrderlessSourceFrameSetNoiseCanaryLauncherTests(unittest.TestCase):
    def test_launcher_is_valid_bash_but_never_submits_itself(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        executable_lines = [
            line.strip()
            for line in SOURCE.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(
            any(re.match(r"(?:command\s+)?sbatch(?:\s|$)", line) for line in executable_lines)
        )

    def test_one_node_is_split_into_two_concurrent_world4_sp4_groups(self) -> None:
        self.assertIn("#SBATCH --nodes=1", SOURCE)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", SOURCE)
        self.assertIn("--nproc_per_node=4", SOURCE)
        self.assertIn("distributed.world_size != WORLD_SIZE", (
            METHOD_ROOT / "infer_orderless_source_frame_set_noise_canary.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("init_parallel_state(ulysses_size=SP_SIZE)", (
            METHOD_ROOT / "infer_orderless_source_frame_set_noise_canary.py"
        ).read_text(encoding="utf-8"))
        self.assertIn('launch_group dog 0,1,2,3 "${dog_port}"', SOURCE)
        self.assertIn('launch_group human 4,5,6,7 "${human_port}"', SOURCE)
        dog_launch = SOURCE.index('launch_group dog 0,1,2,3 "${dog_port}"')
        human_launch = SOURCE.index('launch_group human 4,5,6,7 "${human_port}"')
        first_wait = SOURCE.index('wait "${dog_pid}"', human_launch)
        self.assertLess(dog_launch, human_launch)
        self.assertLess(human_launch, first_wait)
        between = SOURCE[dog_launch:first_wait]
        self.assertGreaterEqual(between.count("&"), 2)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible_gpus}"', SOURCE)
        self.assertIn('unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL', SOURCE)

    def test_launcher_closes_source_archive_and_runs_three_cpu_preflights(self) -> None:
        required = (
            "infer_orderless_source_frame_set_noise_canary.py",
            "orderless_source_frame_set_noise.py",
            "infer_native_identity_generation_canary.py",
            "tools/build_renderer_dataset.py",
            "assets/orderless_source_frame_set_noise_core2_v1.json",
            "assets/pair_v5_t2v_calibration_first8_authoring_v1.json",
            "scripts/auh_infer_orderless_source_frame_set_noise_canary_dual4.sbatch",
            "tests/test_orderless_source_frame_set_noise.py",
            "tests/test_infer_orderless_source_frame_set_noise_canary.py",
            "tests/test_auh_infer_orderless_source_frame_set_noise_canary_launcher.py",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertIn(f'methods/bernini_action_editing/{relative}', SOURCE)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', SOURCE)
        self.assertIn('sha256sum "${source_archive}"', SOURCE)
        self.assertIn('sha256sum "${cell_spec}"', SOURCE)
        self.assertIn('sha256sum "${staged_authoring_spec}"', SOURCE)
        self.assertIn(
            'authoring_spec_sha256="204f7de92fde95a89ab5750ec226dea58fb71edba6c071c76a7c8c56f91bb89c"',
            SOURCE,
        )
        self.assertIn('--authoring-spec "${output_root}/sealed-first8-authoring-spec.json"', SOURCE)
        self.assertIn('--expected-authoring-spec-sha256 "${authoring_spec_sha256}"', SOURCE)
        for name in (
            "test_orderless_source_frame_set_noise.py",
            "test_infer_orderless_source_frame_set_noise_canary.py",
            "test_auh_infer_orderless_source_frame_set_noise_canary_launcher.py",
        ):
            self.assertIn(name, SOURCE)
        self.assertIn('PYTHONPATH="${method_root}" "${python_bin}" -B', SOURCE)

    def test_five_exact40_exact81_arms_and_all_ten_outputs_are_fail_closed(self) -> None:
        arms = (
            "official_gaussian",
            "correct_source_rho005",
            "wrong_source_rho005",
            "correct_source_rho010",
            "wrong_source_rho010",
        )
        for arm in arms:
            with self.subTest(arm=arm):
                self.assertGreaterEqual(SOURCE.count(f'"{arm}"'), 1)
        self.assertIn('--num-inference-steps 40', SOURCE)
        self.assertIn('sampling.get("exact40") is True', SOURCE)
        self.assertIn('sampling.get("exact81") is True', SOURCE)
        self.assertIn('sampling.get("frame_count") == 81', SOURCE)
        self.assertIn('len(arms) == 5', SOURCE)
        self.assertIn('set(outputs) == set(expected_arms)', SOURCE)
        self.assertIn('"total_candidate_count": 10', SOURCE)
        self.assertIn('for label in ("dog", "human"):', SOURCE)
        self.assertIn('"groups_ran_concurrently": True', SOURCE)
        self.assertIn('"upstream_authoring_spec_sha256": authoring_sha', SOURCE)
        self.assertIn(
            'source_conditioning.get("caller_selection_indices_consumed") is True',
            SOURCE,
        )
        self.assertIn(
            'source_conditioning.get("operator_received_frame_indices") is False',
            SOURCE,
        )
        self.assertIn(
            'source_conditioning.get("operator_set_sequence_order_consumed") is False',
            SOURCE,
        )
        self.assertIn(
            'source_conditioning.get("wrong_source_is_pure_identity_control") is False',
            SOURCE,
        )
        self.assertIn(
            'human source-specificity control confound is not sealed', SOURCE
        )
        self.assertIn(
            'gate_contract.get("identity_adapter_used_as_baseline") is False', SOURCE
        )
        self.assertIn(
            'gate_contract.get("identity_adapter_used_as_prior") is False', SOURCE
        )
        self.assertIn(
            '"same_rho_correct_vs_wrong_carrier_source_specificity"', SOURCE
        )
        self.assertIn(
            '"old_motion_direction_and_order_leakage_nonincrease"', SOURCE
        )

    def test_receipts_deny_training_reward_ranking_and_selection(self) -> None:
        denied_child = (
            'interpretation.get("training_performed") is False',
            'interpretation.get("training_harness_executed") is False',
            'interpretation.get("validation_only_train_lora_module_imported") is True',
            'interpretation.get("trainer_instantiated") is False',
            'interpretation.get("critic_loaded") is False',
            'interpretation.get("reward_computed") is False',
            'interpretation.get("best_arm_selected") is False',
            'interpretation.get("ranking_performed") is False',
        )
        for fragment in denied_child:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, SOURCE)
        denied_parent = (
            '"training_performed": False',
            '"training_harness_executed": False',
            '"validation_only_train_lora_module_imported": True',
            '"trainer_instantiated": False',
            '"critic_loaded": False',
            '"reward_computed": False',
            '"ranking_performed": False',
            '"best_arm_selected": False',
            '"scientific_claim_authorized": False',
        )
        for fragment in denied_parent:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, SOURCE)
        self.assertNotIn("argmax", SOURCE)
        self.assertNotIn("sort(key=", SOURCE)
        self.assertNotIn("optimizer.step", SOURCE)
        self.assertNotIn("backward()", SOURCE)

    def test_output_is_fresh_transactional_and_children_are_verified(self) -> None:
        self.assertIn('[[ ! -e "${output_root}" && ! -L "${output_root}" ]]', SOURCE)
        self.assertIn('mkdir -- "${output_root}"', SOURCE)
        self.assertIn('path = root / label / "receipt.json"', SOURCE)
        self.assertIn('not video.is_file()', SOURCE)
        self.assertIn('video.is_symlink()', SOURCE)
        self.assertIn('output.get("frame_count") != 81', SOURCE)
        self.assertIn('output.get("fps") != 25', SOURCE)
        self.assertIn("def canonical_object_digest(value):", SOURCE)
        self.assertIn('unsigned_child.pop("receipt_digest", None)', SOURCE)
        self.assertIn(
            'canonical_object_digest(unsigned_child) != claimed_receipt_digest',
            SOURCE,
        )
        self.assertIn("def verify_child_artifacts(", SOURCE)
        self.assertIn('if len(rows) != 26:', SOURCE)
        self.assertIn('resolved.relative_to(child_root)', SOURCE)
        self.assertIn('if observed_sha != expected_sha:', SOURCE)
        self.assertIn('"verified_artifact_count": len(verified)', SOURCE)
        self.assertIn('"verified_artifact_set_digest": canonical_object_digest(verified)', SOURCE)
        self.assertIn('os.O_WRONLY | os.O_CREAT | os.O_EXCL', SOURCE)
        self.assertIn('target = root / "all8-receipt.json"', SOURCE)
        self.assertIn('"${output_root}/sealed-core2-spec.json"', SOURCE)
        self.assertIn('"${output_root}/sealed-first8-authoring-spec.json"', SOURCE)


if __name__ == "__main__":
    unittest.main()

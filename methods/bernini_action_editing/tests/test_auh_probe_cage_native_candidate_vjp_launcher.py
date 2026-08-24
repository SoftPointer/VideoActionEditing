from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_probe_cage_native_candidate_vjp_dual4.sbatch"
)
PROBE = METHOD_ROOT / "probe_cage_native_candidate_vjp.py"


class AUHCAGENativeCandidateVJPDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = LAUNCHER.read_text(encoding="utf-8")
        cls.probe_text = PROBE.read_text(encoding="utf-8")

    def test_launcher_is_valid_bash_and_uses_all_eight_gpus_as_two_sp4_arms(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.text)
        self.assertIn("#SBATCH --nodes=1", self.text)
        self.assertIn("#SBATCH --qos=gtqos", self.text)
        self.assertIn('run_arm dog-fit "0,1,2,3"', self.text)
        self.assertIn('run_arm human-fit "4,5,6,7"', self.text)
        self.assertEqual(self.text.count("--nproc_per_node=4"), 1)
        self.assertIn("WORLD4/Ulysses-SP4", self.text)
        self.assertIn("all8=true", self.text)

    def test_launcher_pins_source_models_checkpoint_and_exact40_81f_banks(self) -> None:
        for name in (
            "CAGE_PROBE_SOURCE_ARCHIVE",
            "CAGE_PROBE_SOURCE_ARCHIVE_SHA256",
            "CAGE_PROBE_SOURCE_REVISION",
            "CAGE_PROBE_NATIVE_POPULATION_SPEC_SHA256",
            "CAGE_PROBE_T2V_CORE4_V2_SPEC_SHA256",
            "CAGE_PROBE_NATIVE_ROLLOUT_ROOT",
            "BERNINI_OFFICIAL_ROOT",
            "BERNINI_OFFICIAL_COMMIT",
            "BERNINI_VEOMNI_ROOT",
            "BERNINI_VEOMNI_COMMIT",
            "BERNINI_ACTION_CHECKPOINT",
            "BERNINI_CHECKPOINT_TREE_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
        ):
            self.assertIn(name, self.text)
        self.assertIn(
            "525d727951ee05d7aac27f47d294e3604996781106dfc710087d4029a1bbd8f0",
            self.text,
        )
        self.assertIn(
            "a18387b383fb11f19279c67694089754ff84b51e939e7a92b51a7e35a0743a95",
            self.text,
        )
        self.assertIn('sampling.get("num_frames") != 81', self.text)
        self.assertIn('sampling.get("latent_frames") != 21', self.text)
        self.assertIn('sampling.get("num_inference_steps") != 40', self.text)
        self.assertIn('native_sampling.get("num_frames") != 81', self.text)
        self.assertIn(
            'native_sampling.get("source_reference_indices") != [0, 27, 53, 80]',
            self.text,
        )
        self.assertIn("len(t2v_rows) != 40", self.text)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.text)
        self.assertIn('git get-tar-commit-id <"${archive_copy}"', self.text)
        self.assertGreaterEqual(
            self.text.count('sha256sum "${source_archive}"'), 1
        )

    def test_archive_preflight_requires_the_audited_exact26_python_closure(self) -> None:
        expected = {
            "methods/bernini_action_editing/cage_candidate_action_energy_vjp.py",
            "methods/bernini_action_editing/dclr_runtime_contract.py",
            "methods/bernini_action_editing/infer_lora.py",
            "methods/bernini_action_editing/infer_native_identity_generation_canary.py",
            "methods/bernini_action_editing/infer_pair_v5_t2v_calibration_bank.py",
            "methods/bernini_action_editing/infer_source_kv_carrier_oracle.py",
            "methods/bernini_action_editing/infer_source_value_residual_oracle.py",
            "methods/bernini_action_editing/inference_sigma_strata.py",
            "methods/bernini_action_editing/mace_candidate_action_energy.py",
            "methods/bernini_action_editing/pair_v5_action_adapter.py",
            "methods/bernini_action_editing/pair_v5_native_bridge.py",
            "methods/bernini_action_editing/pair_v5_native_rollout_spec.py",
            "methods/bernini_action_editing/pair_v5_native_rv2v_action_score_v3.py",
            "methods/bernini_action_editing/pair_v5_phase_conjunctive_energy.py",
            "methods/bernini_action_editing/pair_v5_t2v_calibration_bank_spec.py",
            "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
            "methods/bernini_action_editing/probe_cage_native_candidate_vjp.py",
            "methods/bernini_action_editing/score_pair_v5_t2v_energy_bank_v3.py",
            "methods/bernini_action_editing/source_kv_replay.py",
            "methods/bernini_action_editing/source_kv_route_batches.py",
            "methods/bernini_action_editing/source_self_native_ref_contrastive_v3.py",
            "methods/bernini_action_editing/source_self_native_rv2v_guidance.py",
            "methods/bernini_action_editing/source_self_native_target_adapter.py",
            "methods/bernini_action_editing/source_value_residual.py",
            "methods/bernini_action_editing/train_lora.py",
            "methods/bernini_action_editing/validate_pair_v5_t2v_calibration_mainline_v3.py",
        }
        start = self.text.index("required = {") + len("required = ")
        end = self.text.index("\n}\nif len(required) != 26:", start) + 2
        observed = ast.literal_eval(self.text[start:end])
        self.assertEqual(observed, expected)
        self.assertEqual(len(observed), 26)
        module_by_name = {
            path.stem: path for path in METHOD_ROOT.glob("*.py")
        }
        pending = [PROBE.stem]
        recursively_imported = set()
        while pending:
            name = pending.pop()
            path = module_by_name.get(name)
            if path is None or path in recursively_imported:
                continue
            recursively_imported.add(path)
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = [item.name.split(".")[0] for item in node.names]
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module
                ):
                    imported = [node.module.split(".")[0]]
                else:
                    imported = []
                pending.extend(
                    imported_name
                    for imported_name in imported
                    if imported_name in module_by_name
                )
        audited_from_source = {
            path.relative_to(METHOD_ROOT.parents[1]).as_posix()
            for path in recursively_imported
        }
        self.assertEqual(observed, audited_from_source)
        self.assertIn(
            'raise SystemExit("native CAGE audited Python import closure is not exact26")',
            self.text,
        )

    def test_launcher_selects_one_preregistered_fit_row_per_action_family(self) -> None:
        self.assertIn(
            "pair5-native-core4-v1-7b88a1ca1f804f41-action-s2026080901",
            self.text,
        )
        self.assertIn(
            "pair5-native-core4-v1-a35b590961d24694-action-s2026080901",
            self.text,
        )
        self.assertIn('"dog-sit-facing-camera", "fit"', self.text)
        self.assertIn('"human-rise-to-stand", "fit"', self.text)
        self.assertIn("native/prompt family binding differs", self.text)

    def test_launcher_invokes_only_the_real_no_update_probe_contract(self) -> None:
        self.assertIn("probe_cage_native_candidate_vjp.py", self.text)
        self.assertIn("--ack-probe-not-training", self.text)
        self.assertIn("--target-margin 0.0 --temperature 1.0", self.text)
        for prohibited in (
            "--target-video",
            "--proposal-video",
            "--proposal-latent",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "optimizer.step",
        ):
            self.assertNotIn(prohibited, self.text)
        self.assertIn('receipt.get("training_performed") is not False', self.text)
        self.assertIn('receipt.get("optimizer_created") is not False', self.text)
        self.assertIn('receipt.get("optimizer_steps") != 0', self.text)
        self.assertIn('"scientific_action_editing_success_claim": False', self.text)

    def test_strong_postflight_checks_both_receipts_cotangents_and_freeze(self) -> None:
        for token in (
            "CAGE_NATIVE_CANDIDATE_VJP_DUAL4_STRONG_POSTFLIGHT_OK",
            "cage-native-candidate-vjp-receipt.json",
            "cage-native-candidate-cotangent.safetensors",
            "verify_embedded",
            "freeze_certificates_identical",
            "manifest_sha256_computed",
            "scan_shared_step_calls",
            "scan_patch_vae_latent_calls",
            "replay_shared_step_calls",
            "gradient_nonzero",
            "safe_open",
            "torch.isfinite",
            "candidate_action_cotangent",
            "os.O_EXCL",
        ):
            self.assertIn(token, self.text)
        self.assertIn('execution.get("scan_shared_step_calls") != 30', self.text)
        self.assertIn('execution.get("replay_shared_step_calls") != 2', self.text)
        self.assertIn('artifact.get("shape", [])[:3] != [1, 16, 21]', self.text)
        self.assertIn('artifact.get("stored_dtype") != "torch.float32"', self.text)

    def test_probe_has_no_block_scan_cli_and_launcher_discloses_the_blocker(self) -> None:
        tree = ast.parse(self.probe_text)
        option_strings = {
            constant.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args[:1]
            if isinstance(argument, ast.Constant)
            for constant in (argument,)
            if isinstance(constant.value, str)
        }
        self.assertNotIn("--registered-block-indices", option_strings)
        self.assertNotIn("--sealed-band-receipt", option_strings)
        self.assertNotIn("--block-scan", option_strings)
        self.assertIn(
            '"transformer_30_block_parameter_scan_performed": False',
            self.text,
        )
        self.assertIn(
            '"sealed_trainable_band_receipt_produced": False', self.text
        )
        self.assertIn('"one_step_training_authorized": False', self.text)
        self.assertIn(
            '"thirty_branch_sigma_calls_are_transformer_block_scan": False',
            self.text,
        )
        self.assertIn("30 branch/sigma model calls are not \"30 blocks\"", self.text)


if __name__ == "__main__":
    unittest.main()

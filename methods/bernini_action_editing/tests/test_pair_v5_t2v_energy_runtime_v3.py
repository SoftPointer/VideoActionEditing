from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
SCORER = METHOD_ROOT / "score_pair_v5_t2v_energy_bank_v3.py"
FINALIZER = METHOD_ROOT / "finalize_pair_v5_t2v_cagd_v3.py"
VALIDATOR = METHOD_ROOT / "validate_pair_v5_cagd_evidence_v3.py"
LAUNCHER = METHOD_ROOT / "scripts/auh_score_pair_v5_t2v_energy_bank_v3_dual4.sbatch"
LEGACY_COMPAT = METHOD_ROOT / "pair_v5_t2v_score_v3_compat.py"


class PairV5EnergyRuntimeStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = SCORER.read_text(encoding="utf-8")
        cls.finalizer = FINALIZER.read_text(encoding="utf-8")
        cls.validator = VALIDATOR.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.legacy_compat = LEGACY_COMPAT.read_text(encoding="utf-8")
        ast.parse(cls.scorer)
        ast.parse(cls.finalizer)
        ast.parse(cls.validator)
        ast.parse(cls.legacy_compat)

    def test_legacy_v3_compat_packets_are_explicitly_non_authorizing(self) -> None:
        self.assertIn('"diagnostic_non_authorizing": True', self.legacy_compat)
        self.assertIn('"optimizer_authorized": False', self.legacy_compat)
        self.assertIn('"scientific_action_editing_claim": False', self.legacy_compat)
        self.assertIn("DIAGNOSTIC_NON_AUTHORIZING = True", self.legacy_compat)

    def test_single_real_exact40_mid_coordinate_is_preregistered(self) -> None:
        self.assertIn("PILOT_SCHEDULE_INDEX = 33", self.scorer)
        self.assertIn("PILOT_SIGMA = 0.5161304473876953", self.scorer)
        self.assertIn("PILOT_NATIVE_SCHEDULER_TIMESTEP = 516", self.scorer)
        self.assertIn(
            "PILOT_SCORER_PHYSICAL_TIMESTEP = float(PILOT_NATIVE_SCHEDULER_TIMESTEP)",
            self.scorer,
        )
        self.assertNotIn("PILOT_SCORER_PHYSICAL_TIMESTEP = 516.1304321289062", self.scorer)
        self.assertIn("native_schedule.NATIVE_UNIPC40_SIGMAS[PILOT_SCHEDULE_INDEX]", self.scorer)
        self.assertIn("action_adapter.sigma_gate(PILOT_SCHEDULE_INDEX)", self.scorer)

    def test_global_reward_is_primary_and_phase_is_diagnostic_only(self) -> None:
        self.assertIn("_canonical_action_energy_packet", self.scorer)
        self.assertIn('reward = canonical["score"]', self.scorer)
        self.assertIn(
            '"live_origin_global_action_energy_score_diagnostic": live_reward',
            self.scorer,
        )
        self.assertRegex(self.scorer, r"phase_reward\s*=\s*float\(score\.phase_energy\.reward\.item\(\)\)")
        self.assertIn('"raw_global_action_energy_score": reward', self.scorer)
        self.assertIn('"raw_phase_conjunctive_score_diagnostic": phase_reward', self.scorer)
        self.assertIn('"phase_diagnostic_used_for_calibration": False', self.scorer)
        self.assertIn("CANONICAL_ACTION_ENERGY_DECIMAL_PRECISION = 80", self.scorer)
        self.assertNotIn("MACE_CROSS_DEVICE_REPLAY_", self.scorer)
        self.assertNotIn('raw_phase_conjunctive_score=score[', self.finalizer)

    def test_scorer_uses_candidate_clean_latent_and_same_cell_gaussian_only(self) -> None:
        self.assertIn('key="normalized_clean_latent"', self.scorer)
        self.assertIn('key="official_initial_gaussian"', self.scorer)
        self.assertIn("same-cell official Gaussian tensor value drifted", self.scorer)
        self.assertIn("native_bridge.FrozenBerniniT2VScorer", self.scorer)
        self.assertIn("native_bridge.score_frozen_t2v_action_energy", self.scorer)
        self.assertIn('"generated_mp4_consumed_by_scorer": False', self.scorer)
        self.assertNotRegex(
            self.scorer,
            r"score_frozen_t2v_action_energy\([^\)]*(source|rv2v|target|mask|flow|pose|track)",
        )

    def test_every_branch_closes_generation_prompt_source_and_checkpoint(self) -> None:
        for token in (
            "generation_runtime_binding_from_native_receipt",
            "validate_generation_runtime_registry",
            '"generation_runtime_binding_by_branch"',
            '"generation_runtime_registry_digest"',
            'row["full_prompt_utf8_sha256"]',
            'hashlib.sha256(prompt.encode("utf-8")).hexdigest()',
            "CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "CHECKPOINT_TREE_SHA256",
            "BERNINI_OFFICIAL_COMMIT",
            "VEOMNI_TESTED_COMMIT",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.scorer)

    def test_event_audit_and_confirmation_never_enter_model_or_optimizer(self) -> None:
        self.assertIn('"event_audit_label_consumed_by_model": False', self.scorer)
        self.assertIn("complete_target_transition_observed", self.finalizer)
        self.assertIn("terminal_hold_observed", self.finalizer)
        self.assertIn("full_target_action_observed", self.finalizer)
        self.assertIn("full_target_action_false_confirmed", self.finalizer)
        self.assertIn('"confirmation_eligibility_count": 0', self.finalizer)
        self.assertIn('"confirmation_samples_consumed_by_optimizer": False', self.finalizer)
        self.assertIn('analysis_split="fit"', self.finalizer)

    def test_validator_recomputes_instead_of_trusting_eligibility(self) -> None:
        self.assertIn("legacy_eligibility_self_declared_booleans_are_insufficient", self.validator)
        self.assertIn("scorer.load_group_bank", self.validator)
        self.assertIn("calibration.calibrate_global_action_energy", self.validator)
        self.assertIn("guidance.seal_eligibility", self.validator)
        self.assertIn("guidance_trainer.load_manifest", self.validator)
        self.assertIn("guidance_trainer.load_event_tensors", self.validator)
        self.assertIn("raw_global_action_energy_score", self.validator)
        self.assertNotIn("legacy_eligibility_self_declaration_trusted\": True", self.validator)

    def test_launcher_is_two_concurrent_sp4_on_all_eight_gpus(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.launcher)
        self.assertIn('run_group sp4-a "0,1,2,3"', self.launcher)
        self.assertIn('run_group sp4-b "4,5,6,7"', self.launcher)
        self.assertGreaterEqual(self.launcher.count("--nproc_per_node=4"), 1)
        self.assertIn("& sp4_a_pid=$!", self.launcher)
        self.assertIn("& sp4_b_pid=$!", self.launcher)
        self.assertIn("PAIR_V5_T2V_GLOBAL_ENERGY_V4_DUAL4_OK", self.launcher)

    def test_launcher_fail_closed_archive_and_hash_contract(self) -> None:
        for token in (
            "git get-tar-commit-id",
            "sha256sum",
            "PAIR_V5_T2V_BANK_RECEIPT_SHA256",
            "PAIR_V5_T2V_BANK_SPEC_SHA256",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST",
            "BERNINI_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "--ack-no-action-success-claim",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.launcher)
        self.assertRegex(self.launcher, re.compile(r"\[\[ ! -e \"\$\{output_dir\}\""))
        self.assertIn("scratch source archive hash differs after copy", self.launcher)
        self.assertIn(
            "scratch source archive revision differs after copy", self.launcher
        )
        self.assertGreaterEqual(self.launcher.count('sha256sum "${archive_copy}"'), 1)
        self.assertGreaterEqual(
            self.launcher.count('git get-tar-commit-id <"${archive_copy}"'), 1
        )

    def test_launcher_accepts_only_hash_bound_core4_v2_bank(self) -> None:
        for token in (
            "pair-v5-frozen-bernini-t2v-calibration-bank-spec-v2",
            "pair-v5-frozen-bernini-t2v-calibration-bank-receipt-v2",
            'value.startswith("pair5-t2v-core4-v2-")',
            'bank.get("root_spec_raw_sha256") != spec_sha',
            'bank.get("candidate_count") != 40',
            'bank.get("cell_count") != 4',
            "core4-v2 embedded bank receipt digest differs",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.launcher)

    def test_launcher_archive_closure_has_only_actual_scorer_dependencies(self) -> None:
        match = re.search(r"required = (\{.*?\})\nseen=set\(\)", self.launcher, re.DOTALL)
        self.assertIsNotNone(match)
        actual = ast.literal_eval(match.group(1))
        dependencies = {
            "score_pair_v5_t2v_energy_bank_v3.py",
            "pair_v5_t2v_calibration_bank_spec.py",
            "infer_pair_v5_t2v_calibration_bank.py",
            "infer_native_identity_generation_canary.py",
            "infer_lora.py",
            "train_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
            "source_value_residual.py",
            "pair_v5_native_bridge.py",
            "pair_v5_phase_conjunctive_energy.py",
            "mace_candidate_action_energy.py",
            "pair_v5_action_adapter.py",
            "inference_sigma_strata.py",
            "source_self_native_ref_contrastive_v3.py",
            "source_self_native_rv2v_guidance.py",
            "source_self_native_target_adapter.py",
            "dclr_runtime_contract.py",
        }
        expected = {
            f"methods/bernini_action_editing/{dependency}"
            for dependency in dependencies
        }
        self.assertEqual(actual, expected)
        self.assertNotIn(
            "methods/bernini_action_editing/pair_v5_t2v_energy_calibration_v3.py",
            actual,
        )

    def test_launcher_postflight_binds_both_groups_to_native_timestep_516(self) -> None:
        for token in (
            "import score_pair_v5_t2v_energy_bank_v3 as scorer",
            "scorer.load_group_bank(",
            "scorer.validate_score_receipt(raw_score)",
            'receipt.get("bank_receipt_digest") != bank_digest',
            'receipt.get("method_source_revision") != revision',
            'receipt.get("method_source_archive_sha256") != archive_sha',
            'checkpoint.get("manifest_sha256") != checkpoint_manifest_sha',
            'coordinate.get("schedule_index") != 33',
            'coordinate.get("native_scheduler_timestep") != 516',
            'coordinate.get("frozen_t2v_scorer_timestep") != 516.0',
            "direct_native_unipc40_discrete_timestep_same_schedule_index",
            'coordinate.get("legacy_1000_sigma_timestep_rejected") is not True',
            'score.get("generation_runtime_binding_by_branch")',
            "expected_generation_registry_by_cell",
            "actual_candidate_receipt_digests != receipt.get(",
            "ordered candidate receipt digest registry differs",
            "native_model_timestep=516",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.launcher)


if __name__ == "__main__":
    unittest.main()

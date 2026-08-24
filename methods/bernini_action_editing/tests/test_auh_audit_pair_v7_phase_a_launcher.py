from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_audit_pair_v7_phase_a_dp2sp4.sbatch"


class PairV7PhaseALauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_one_node_all_eight_mi210s_and_world8_dp2sp4(self) -> None:
        self.assertIn("#SBATCH --nodes=1", self.source)
        self.assertIn("#SBATCH --ntasks=1", self.source)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("--nproc_per_node=8", self.source)
        self.assertIn("topology=WORLD8/DP2xSP4", self.source)
        self.assertIn('"topology") == "WORLD8-DP2xUlysses-SP4"', self.source)
        self.assertNotIn("--nproc_per_node=4", self.source)

    def test_first_measurement_cell_is_literal_33_and_not_environment_selected(self) -> None:
        self.assertEqual(self.source.count("--schedule-index 33"), 1)
        self.assertNotIn("PAIR_V7_PHASE_A_SCHEDULE_INDEX", self.source)
        self.assertNotIn("schedule index must be 0..39", self.source)
        self.assertNotIn("low38_39", self.source)
        self.assertNotIn("if index in (38, 39)", self.source)
        self.assertIn('schedule.get("schedule_index") == 33', self.source)
        self.assertIn('schedule.get("first_phase_a_schedule_index") == 33', self.source)

    def test_root_review_and_measurement_only_acknowledgements_fail_closed(self) -> None:
        self.assertIn("PAIR_V7_ROOT_REVIEWED_PHASE_A", self.source)
        self.assertIn("PAIR_V7_ACK_NO_MUTATION_NO_SUCCESS", self.source)
        self.assertIn("--ack-root-reviewed-phase-a-launch", self.source)
        self.assertIn("--ack-no-parameter-mutation-no-success-claim", self.source)

    def test_new_fit_only_authority_replaces_v6_source_and_optimizer_authority(self) -> None:
        for token in (
            "PAIR_V7_FIT_ONLY_MANIFEST",
            "PAIR_V7_FIT_ONLY_MANIFEST_SHA256",
            "PAIR_V7_FIT_ONLY_EVIDENCE",
            "PAIR_V7_FIT_ONLY_EVIDENCE_SHA256",
            "PAIR_V7_CAST_V4_METHOD_ARCHIVE",
            "PAIR_V7_CAST_V4_GROUP_A_RECEIPT",
            "PAIR_V7_CAST_V4_GROUP_B_RECEIPT",
        ):
            self.assertIn(token, self.source)
        for obsolete in (
            "PAIR_V7_SOURCE_BINDING_MANIFEST",
            "PAIR_V7_ACTION_EVENT_MANIFEST",
            "PAIR_V7_CAGD_V3_EVIDENCE",
            "--source-binding-manifest",
            "--expected-source-binding-manifest-sha256",
        ):
            self.assertNotIn(obsolete, self.source)
        # Compatibility CLI names map only to the new sealed fit/no-update files.
        self.assertIn('--action-event-manifest "${fit_manifest}"', self.source)
        self.assertIn('--cagd-validator-evidence "${fit_evidence}"', self.source)
        self.assertEqual(self.source.count('--scorer-group-receipt "'), 2)
        self.assertEqual(
            self.source.count('--expected-scorer-group-receipt-sha256 "'), 2
        )

    def test_scientific_pins_and_checkpoint_closure_are_fail_closed(self) -> None:
        for token in (
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "d5e87caf8f58a63c2ff902386d6e19100f7fcf34",
            "a15a95ea9bcee5cbe331ef975ce2cef38e6c164df644d8d053a87b5fe71c7470",
            "bc20af168fdb29f72854d6c4c0f978eb29ba6642c1a5cbbeab69f33bf068ea1a",
            "4a95e306c560b37b52f204c8d818774d43b648ff6fc79e5c704c9e69cdc2ce75",
        ):
            self.assertIn(token, self.source)
        self.assertEqual(self.source.count("sha256sum --strict --status -c"), 2)
        self.assertIn("checkpoint content verification failed before Phase-A", self.source)
        self.assertIn("checkpoint content verification failed after Phase-A", self.source)

    def test_runtime_and_cast_archives_are_revision_and_byte_bound(self) -> None:
        self.assertIn("PAIR_V7_PHASE_A_RUNTIME_ARCHIVE_SHA256", self.source)
        self.assertIn("PAIR_V7_PHASE_A_RUNTIME_REVISION", self.source)
        self.assertIn("PAIR_V7_CAST_V4_METHOD_ARCHIVE_SHA256", self.source)
        self.assertIn("PAIR_V7_CAST_V4_METHOD_REVISION", self.source)
        self.assertEqual(self.source.count("git get-tar-commit-id"), 2)
        self.assertIn("unsafe or duplicate runtime archive member", self.source)
        self.assertIn("runtime archive lacks fit-only Phase-A closure", self.source)
        self.assertIn("chmod a-w", self.source)
        self.assertIn("read-only extracted runtime tree changed", self.source)
        self.assertIn("sealed input changed during Phase-A", self.source)

    def test_runtime_archive_contains_actual_import_closure_and_focused_tests(self) -> None:
        for relative in (
            "pair_v7_fit_only_geometry_authority.py",
            "pair_v7_dual_coordinate_nullspace_transport.py",
            "pair_v5_t2v_guidance_distill.py",
            "score_pair_v5_t2v_energy_bank_v3.py",
            "infer_pair_v5_t2v_calibration_bank.py",
            "source_self_native_rv2v_guidance.py",
            "infer_native_identity_generation_canary.py",
            "test_pair_v7_dual_coordinate_nullspace_transport.py",
            "test_pair_v7_fit_only_geometry_authority.py",
            "test_audit_pair_v7_phase_a_geometry.py",
            "test_auh_audit_pair_v7_phase_a_launcher.py",
            "test_pair_v5_action_adapter.py",
        ):
            self.assertIn(relative, self.source)
        for obsolete in (
            "train_pair_v5_t2v_guidance_distill.py",
            "validate_pair_v5_cagd_evidence_v3.py",
            "finalize_pair_v5_t2v_cagd_v3.py",
        ):
            self.assertNotIn(obsolete, self.source)

    def test_focused_tests_precede_preflight_and_world8_runtime(self) -> None:
        unit = self.source.index("for test_name in")
        preflight = self.source.index("preflight: fit-only geometry measurement")
        runtime = self.source.index("-m torch.distributed.run")
        postflight = self.source.index('raw_root = Path(output_value)')
        self.assertLess(unit, preflight)
        self.assertLess(preflight, runtime)
        self.assertLess(runtime, postflight)

    def test_output_is_one_measurement_receipt_and_no_update_artifact(self) -> None:
        for forbidden in (
            "--learning-rate",
            "--max-schedule-steps",
            "torch.optim",
            "optimizer.step(",
            "parameter.add_(",
            "adapter.safetensors",
            "optimizer.pt",
            "ACTION_EDITING_SUCCESS",
        ):
            self.assertNotIn(forbidden, self.source)
        self.assertIn('entries[0].name == "receipt.json"', self.source)
        self.assertIn("Phase-A artifact closure must be receipt.json only", self.source)
        for field in (
            '"optimizer_constructed"',
            '"optimizer_step_called"',
            '"candidate_delta_constructed"',
            '"parameter_add_called"',
            '"parameter_mutation_performed"',
            '"parameter_update_authorized"',
            '"scientific_action_editing_success_claim"',
        ):
            self.assertIn(field, self.source)
        self.assertIn("MEASUREMENT_COMPLETE_ONLY", self.source)

    def test_postflight_requires_full_correct_source_and_sp4_receipts(self) -> None:
        for token in (
            'source.get("source_frame_count") == 81',
            'source.get("source_fps") == 25.0',
            'source.get("deployment_visual_condition") == "source_video_only_V"',
            'source.get("image_reference_count") == 0',
            'source.get("reference_indices") == []',
            'source.get("sp4_source_receipt_consensus_per_arm") is True',
            'source.get("wrong_source_fields_present") is False',
            'len(decoded) == 2',
            'len(ranks) == 8',
            'set(range(4))',
            'bundle.get("all_four_rank_local_vjps_bound") is True',
            'bundle.get("sp4_arithmetic_average_bound") is True',
            'len(identities) == 16',
            '"deploy_noop_identity"',
            '"deploy_camera_delta"',
        ):
            self.assertIn(token, self.source)

    def test_postflight_pins_deployed_v_only_apg_protocol_and_k4(self) -> None:
        for token in (
            'receipt.get("identity_deployment_protocol")',
            '"bernini-pair-v7-phase-a-geometry-audit-v3"',
            '"bernini-pair-v7-phase-a-identity-vjp-v2"',
            '"bernini-pair-v7-phase-a-dp2-union-projection-v2"',
            '"bernini-pair-v7-identity-deployment-protocol-v1"',
            '"VideoEdit_infer_lora_frozen_deployment_contract"',
            '"infer_lora.build_training_prompt"',
            '"infer_lora.DEFAULT_NEGATIVE_PROMPT"',
            'identity_protocol.get("guidance_mode") == "v2v_apg"',
            'identity_protocol.get("visual_condition") == "source_video_only_V"',
            'identity_protocol.get("image_reference_count") == 0',
            '["V_negative", "V_positive"]',
            'identity_protocol.get("omega_txt") == 4.0',
            'identity_protocol.get("eta") == 0.5',
            'identity_protocol.get("norm_threshold") == 50.0',
            'identity_protocol.get("momentum") == 0.0',
            'identity_protocol.get("schedule_index") == 33',
            'identity_protocol.get("timestep") == 516',
            '"scheduler.sigmas[33]_cpu_fp32"',
            'identity_protocol.get("sigma_float32_be_hex") == "3f042120"',
            'identity_protocol.get("fresh_zero_momentum_history_equivalent") is True',
            'identity_protocol.get("old_diff_vjp_coefficient") == 0.0',
            'identity_protocol.get("single_cell_local_field_geometry") is True',
            'identity_protocol.get("full_sampler_trajectory_equivalent") is False',
            'identity_protocol.get("vendor_apg_helper_used") is True',
            '"visual_pack_sampler_parameters_and_post_APG_operator"',
            '"action_lora_scope_is_method_specific_not_infer_lora_peft_scope"',
            'identity_protocol.get("forwarded_visual_branches") == ["V"]',
            '{row.get("sketch_index") for row in group} == set(range(4))',
            'len({row.get("feature_sketch_sha256") for row in group}) == 4',
            'row.get("identity_deployment_protocol_digest")',
            'expected_source_coordinate_digest=row[',
            'expected_deployment_v_pack_digest=row[',
            'expected_negative_prompt_embedding_sha256=row[',
            '"positive_prompt_embedding_sha256_by_branch"',
            'pack.get("patch_call_source_ids") == [1.0, 0.0]',
            'row.get("prompt_receipt_digest") == prompt_digest',
            'prompt_receipt.get("task_prefix_applied_exactly_once") is True',
            'identity_coordinate_by_arm',
            'receipt.get("world_union_solver_authority")',
            '"bernini-pair-v7-phase-a-world-union-authority-v1"',
            'solver.get("authoritative_world_rank") == 0',
            'solver.get("solver_execution_count") == 1',
            'solver.get("solver_device") == "cpu"',
            'solver.get("replicated_gpu_solver_used") is False',
            'solver.get("world_input_digest_consensus") is True',
            'solver.get("world_result_digest_consensus") is True',
            '"bernini-pair-v7-phase-a-world-union-input-v1"',
            'len(solver_input["identity_rows"]) == 16',
            'union.get("identity_probe_union_count") == 16',
            'union.get("identity_sketches_per_source_family") == 4',
            'union.get("identity_source_family_rank_gate")',
            'row.get("minimum_required_rank") == 3',
            'union.get("identity_minimum_global_effective_rank") == 8',
            'union.get("identity_global_effective_rank")',
            'receipt.get("geometry_audit_passed") is True',
            'check_seal(union, "union projection receipt")',
            'receipt.get("nullspace_transport_receipt")',
            'union.get("transport_receipt_digest") == transport_digest',
            'solver.get("union_projection_receipt_digest") == union_digest',
            'solver.get("nullspace_transport_receipt_digest") == transport_digest',
            'solver.get("world_result_digest")',
            '"input_receipt_digest": solver_input_digest',
        ):
            self.assertIn(token, self.source)

    def test_postflight_requires_authority_and_model_source_tree_rehashes(self) -> None:
        for token in (
            'authority = action_manifest',
            'authority.get("fit_only_geometry_authority_validation_receipt")',
            'validation.get("authority_scope") == "fit_only_read_only_gradient_geometry"',
            'validation.get("cast_v4_candidate_receipt_count") == 40',
            'validation.get("external_evidence_file_count") == 49',
            'authority.get("external_evidence_files_rehashed_post_audit") is True',
            'boundary_by_id.get("d541801_v3_confirmation_no_optimizer_go")',
            'boundary_by_id.get("sail_prior_frozen_intervention_no_success")',
            '{row.get("role") for row in sail_children} == {"dog", "human"}',
            'receipt.get("model_source_trees")',
            'trees.get("preflight_receipt") == trees.get("postflight_receipt")',
            'tree_receipt.get("schema_version") == "bernini-pair-v7-source-tree-binding-v2"',
            'tree_receipt.get("tracked_or_extracted_source_bytes_verified") is True',
            'value.get("tracked_relative_symlink_link_and_target_bytes_verified")',
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()

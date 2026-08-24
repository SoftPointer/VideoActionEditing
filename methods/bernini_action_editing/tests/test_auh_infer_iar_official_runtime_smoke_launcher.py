from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_iar_official_runtime_smoke.sbatch"


class AUHIAROfficialRuntimeSmokeLauncherTests(unittest.TestCase):
    """Static fail-closed checks for the dual-WORLD4 official IAR launcher."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL
        )
        cls.runner_region = cls.source.split("runner_args=(", 1)[1].split(
            "audit_receipt()", 1
        )[0]
        cls.audit_region = cls.source.split("audit_receipt()", 1)[1]

    def test_bash_and_every_embedded_python_block_parse(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 7)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_is_one_eight_gpu_allocation_split_into_two_world4_groups(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
        ):
            self.assertIn(directive, self.source)
        self.assertEqual(self.runner_region.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn(
            "topology=DP2xWORLD4/Ulysses4 allocated_gpus=8 groups=2",
            self.source,
        )
        self.assertEqual(self.runner_region.count("launch_group seed-"), 2)
        self.assertRegex(
            self.runner_region,
            r'launch_group seed-a [^\n]+ "\$\{task_scratch\}/group-a" &',
        )
        self.assertRegex(
            self.runner_region,
            r'launch_group seed-b [^\n]+ "\$\{task_scratch\}/group-b" &',
        )
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_launches_only_official_frozen_iar_runtime(self) -> None:
        target = '"${method_root}/infer_iar_official_runtime_smoke.py"'
        self.assertEqual(self.runner_region.count(target), 1)
        self.assertNotRegex(
            self.runner_region,
            r'"\$\{method_root\}/train_[^"\n]+\.py"',
        )
        for forbidden in (
            "--max-steps",
            "--learning-rate",
            "--optimizer",
            "--lora",
            "--adapter",
            "--save-checkpoint",
            "--positive-control-paired-target",
        ):
            self.assertNotIn(forbidden, self.runner_region)
        for declaration in (
            "training=false",
            "optimizer=false",
            "backward=false",
            "checkpoint_save=false",
        ):
            self.assertIn(declaration, self.source)

    def test_runner_gets_complete_native_only_cli(self) -> None:
        fragments = (
            '--bernini-root "${bernini_root}"',
            '--veomni-root "${veomni_root}"',
            '--checkpoint "${checkpoint}"',
            '--checkpoint-content-manifest "${checkpoint_manifest}"',
            '--preprocessed-parquet-dir "${preprocessed_parquet_dir}"',
            '--dataset-summary "${dataset_summary}"',
            '--wrong-source-row-index "${wrong_source_row_index}"',
            '--proposal-source-iid "${proposal_source_iid}"',
            '--expected-wrong-source-iid "${wrong_source_iid}"',
            '--candidate-clean-latent "${candidate_clean_latent}"',
            '--expected-candidate-clean-latent-sha256 "${candidate_clean_latent_sha256}"',
            '--correct-source-clean-latent "${correct_source_clean_latent}"',
            '--expected-correct-source-clean-latent-sha256 "${correct_source_clean_latent_sha256}"',
            '--candidate-arm "${candidate_arm}"',
            '--candidate-provenance-receipt "${candidate_provenance_receipt}"',
            '--expected-candidate-provenance-receipt-sha256 "${candidate_provenance_receipt_sha256}"',
            '--source-provenance-receipt "${source_provenance_receipt}"',
            '--expected-source-provenance-receipt-sha256 "${source_provenance_receipt_sha256}"',
            '--expected-proposal-source-video-sha256 "${proposal_source_video_sha256}"',
            '--wrong-source-clean-latent "${wrong_source_clean_latent}"',
            '--expected-wrong-source-clean-latent-sha256 "${wrong_source_clean_latent_sha256}"',
            '--wrong-source-provenance-receipt "${wrong_source_provenance_receipt}"',
            '--expected-wrong-source-provenance-receipt-sha256 "${wrong_source_provenance_receipt_sha256}"',
            '--expected-wrong-source-video-sha256 "${wrong_source_video_sha256}"',
            '--wrong-source-match-json "${wrong_source_match}"',
            '--expected-wrong-source-match-sha256 "${wrong_source_match_sha256}"',
            '--action-instruction "${action_instruction}"',
            '--expected-action-instruction-sha256 "${action_instruction_sha256}"',
            '--noop-instruction "${noop_instruction}"',
            '--expected-noop-instruction-sha256 "${noop_instruction_sha256}"',
            '--hard-negative-manifest "${hard_negative_manifest}"',
            '--expected-hard-negative-manifest-sha256 "${hard_negative_manifest_sha256}"',
            '--sigmas "${sigma_values[@]}"',
            '--bridge-fractions "${bridge_fraction_values[@]}"',
            "--num-frames 81",
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
            '--launcher-source-sha256 "${launcher_source_sha256}"',
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.runner_region)
        self.assertIn('--noise-seed "${group_seed}"', self.runner_region)
        self.assertIn('--output-receipt "${receipt_path}"', self.runner_region)
        for removed in (
            "IAR_RUNTIME_CANDIDATE_ROW_INDEX",
            "IAR_RUNTIME_EXPECTED_CANDIDATE_IID",
            "--candidate-row-index",
            "--expected-candidate-iid",
            "candidate_row_index",
        ):
            self.assertNotIn(removed, self.source)

    def test_exact_s4_k2_m1_l3_contract_is_fail_closed(self) -> None:
        for fragment in (
            'sigmas_text="${IAR_RUNTIME_SIGMAS:-0.80 0.60 0.35 0.15}"',
            'bridge_fractions_text="${IAR_RUNTIME_BRIDGE_FRACTIONS:-1 0.5 0}"',
            'expected = [0.80, 0.60, 0.35, 0.15]',
            'expected = [1.0, 0.5, 0.0]',
            'len(rows) != 2',
            'no-op instruction must occur exactly once among K=2 negatives',
            "exact81=true S=4 K=2 M=1 L=3 branches=7 forwards_per_rank=84",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("--num-frames 41", self.runner_region)

    def test_negative_manifest_is_content_bound_and_has_no_external_energies(self) -> None:
        for environment_name in (
            "IAR_RUNTIME_HARD_NEGATIVE_MANIFEST",
            "IAR_RUNTIME_HARD_NEGATIVE_MANIFEST_SHA256",
            "IAR_RUNTIME_NOOP_INSTRUCTION",
            "IAR_RUNTIME_NOOP_INSTRUCTION_SHA256",
        ):
            self.assertIn(environment_name, self.source)
        manifest_block = next(
            block
            for block in self.python_blocks
            if "hard-negative manifest must be an object" in block
        )
        for fragment in (
            '"bernini-iar-hard-negative-manifest-v1"',
            '"runtime_fp32_mse_to_epsilon_minus_bridge_clean_lower_is_harder"',
            '"engineering_plumbing_only"',
            '"condition_id", "instruction", "instruction_sha256"',
            'digest != hashlib.sha256(instruction.strip().encode("utf-8")).hexdigest()',
            "digest == action_sha",
            "len(seen_ids) != 2",
            "noop_sha not in seen_hashes",
        ):
            self.assertIn(fragment, manifest_block)
        self.assertNotIn("energies_by_sigma", manifest_block)
        self.assertNotIn("external_content_bound_lower_is_harder", manifest_block)

    def test_source_archive_is_revision_bound_and_has_full_import_closure(self) -> None:
        for fragment in (
            'git get-tar-commit-id <"${source_archive}"',
            '[[ "${actual_archive_revision}" == "${source_revision}" ]]',
            'sha256sum "${source_archive}"',
            'sha256sum "${archive_copy}"',
            "archive member escaped repository-relative closure",
            "member.issym() or member.islnk() or member.isfifo() or member.isdev()",
            "archive retains forbidden replaceable callback scaffold",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'find "${method_root}" -type f -exec chmod a-w',
        ):
            self.assertIn(fragment, self.source)
        for required in (
            "infer_iar_official_runtime_smoke.py",
            "identity_anchored_action_residual.py",
            "infer_dclr_reward_runtime_smoke.py",
            "dual_conditional_ratio_core.py",
            "dclr_runtime_contract.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_lora.py",
            "train_lora.py",
            "motion_residual.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
        ):
            self.assertGreaterEqual(self.source.count(required), 2)
        self.assertEqual(self.source.count("infer_iar_field_feasibility.py"), 1)

    def test_executing_launcher_is_exactly_bound_to_archive_and_receipt(self) -> None:
        for fragment in (
            'launcher_source="$(realpath -e -- "${BASH_SOURCE[0]}")"',
            'require_plain_file "${launcher_source}" "executing launcher"',
            'launcher_source_sha256="$(sha256sum "${launcher_source}"',
            '"${launcher_source}|${launcher_source_sha256}|executing launcher"',
            '"methods/bernini_action_editing/scripts/auh_infer_iar_official_runtime_smoke.sbatch"',
            'archived_launcher="${method_root}/scripts/auh_infer_iar_official_runtime_smoke.sbatch"',
            '[[ "${archived_launcher_sha256}" == "${launcher_source_sha256}" ]]',
            '--launcher-source-sha256 "${launcher_source_sha256}"',
            'local.get("launcher_source_sha256") == launcher_source_sha256',
        ):
            self.assertIn(fragment, self.source)

    def test_all_external_inputs_are_pre_and_post_hash_bound(self) -> None:
        for name in (
            "checkpoint_manifest_sha256",
            "dataset_summary_sha256",
            "candidate_clean_latent_sha256",
            "correct_source_clean_latent_sha256",
            "candidate_provenance_receipt_sha256",
            "source_provenance_receipt_sha256",
            "wrong_source_clean_latent_sha256",
            "wrong_source_provenance_receipt_sha256",
            "wrong_source_match_sha256",
            "hard_negative_manifest_sha256",
        ):
            self.assertGreaterEqual(self.source.count(name), 3, name)
        self.assertIn('declare -a immutable_files=(', self.source)
        self.assertIn('for entry in "${immutable_files[@]}"', self.source)
        self.assertIn("immutable input changed after runtime", self.source)

    def test_rocm_device_and_runtime_caches_are_isolated(self) -> None:
        for fragment in (
            'allocated_rocr_devices="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"',
            "DP2 x WORLD4 IAR smoke requires exactly eight numeric ROCr devices",
            'printf -v group_a_devices \'%s,\' "${allocated_devices[@]:0:4}"',
            'printf -v group_b_devices \'%s,\' "${allocated_devices[@]:4:4}"',
            "unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL",
            'export ROCR_VISIBLE_DEVICES="${visible_devices}"',
            'export MIOPEN_USER_DB_PATH="${scratch_root}/cache/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${scratch_root}/cache/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${scratch_root}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${scratch_root}/cache/triton"',
            'export XDG_CACHE_HOME="${scratch_root}/cache/xdg"',
            'export PYTHONPYCACHEPREFIX="${scratch_root}/cache/pycache"',
            'export TORCHELASTIC_ERROR_FILE="${scratch_root}/torch-elastic-error.json"',
            '"${task_scratch}/group-a/cache/miopen-user"',
            '"${task_scratch}/group-b/cache/miopen-user"',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn(
            'export HIP_VISIBLE_DEVICES="${allocated_rocr_devices}"', self.source
        )
        self.assertNotIn(
            'export CUDA_VISIBLE_DEVICES="${allocated_rocr_devices}"', self.source
        )

    def test_receipt_audit_binds_provenance_grid_and_non_authorization(self) -> None:
        for fragment in (
            "BERNINI_IAR_OFFICIAL_RUNTIME_SP4_STRONG_AUDIT_OK",
            'receipt.get("schema_version")\n    == "bernini-iar-official-runtime-smoke-receipt-v1"',
            'receipt.get("scientific_claim_authorized") is False',
            'receipt.get("donor_plumbing_only") is True',
            'receipt.get("training_authorized") is False',
            'receipt.get("training_pair_authorized") is False',
            'receipt.get("source_reward_calibration_authorized") is False',
            'receipt.get("source_action_invariance_calibration_authorized") is False',
            'receipt.get("paired_target_accessed") is False',
            'receipt.get("forward_callback_present") is False',
            'receipt.get("custom_core_present") is False',
            'distributed.get("world_size") == 4',
            'distributed.get("ulysses_size") == 4',
            'distributed.get("all_gather_full_evidence_digest_only") is True',
            'row.get("local_evidence_digest") == local_digest',
            'local.get("launcher_source_sha256") == launcher_source_sha256',
            'checkpoint_identity.get("verified_file_count") == 23',
            'checkpoint_identity.get("every_file_sha256_verified") is True',
            'native_provenance.get("proposal_latent_sha256")',
            'wrong_provenance.get("target_columns_accessed") == []',
            'wrong_provenance.get("target_media_accessed") is False',
            'match.get("declared_use") == "runtime_plumbing_only"',
            'manifest.get("declared_use") == "engineering_plumbing_only"',
            'manifest.get("energies_supplied_externally") is False',
            'message_schema.get("candidate_dataset_row_accessed") is False',
            'message_schema.get("candidate_dataset_index_consumed") is False',
            'message_schema.get("candidate_dataset_iid_consumed") is False',
            'message_schema.get("encode_renderer_messages_exact_input_guard") is True',
            'canonical_sample_identity(action_instruction, action_sha256)',
            'local.get("forwards_per_rank") == 84',
            'len(cells) == 12',
            '"frozen_identity_noop_wrong_source[0]"',
            '"frozen_identity_action_wrong_source[0]"',
            'independent.get("projection_uses_noop_source_swaps_only") is True',
            'independent.get("action_source_swaps_diagnostic_only") is True',
            'independent.get("gauge_invariant_cap_recomputed") is True',
            'independent.get("low_sigma_exact_zero_verified") is True',
            'cell.get("action_energy_EA")',
            'cell.get("ordering_margins_Ek_minus_EA")',
            'negative_energy - action_energy[0]',
            '"mean_over_all_nonbatch_target_token_and_patch_feature_dimensions"',
            'cell.get("rf_squared_error_proxy_not_likelihood_or_free_energy") is True',
            'cell.get("ordering_is_diagnostic_not_training_authorization") is True',
            'continuity.get("comparison_count") == 8',
            'iar_core.get("training_authorized_by_diagnostic") is False',
            '"forward_only": True',
            '"backward_performed": False',
            '"optimizer_present": False',
            '"checkpoint_saved": False',
            '"adapter_present": False',
        ):
            self.assertIn(fragment, self.audit_region)
        self.assertEqual(self.source.count('audit_receipt "${receipt_path_'), 4)
        self.assertIn("receipt hash closure differs", self.source)
        self.assertIn("immutable input changed after runtime", self.source)

    def test_dual_seed_parity_audit_is_fail_closed(self) -> None:
        for fragment in (
            'noise_seed_a="${IAR_RUNTIME_NOISE_SEED_A:-20260808}"',
            'noise_seed_b="${IAR_RUNTIME_NOISE_SEED_B:-20260809}"',
            '[[ "${noise_seed_a}" != "${noise_seed_b}" ]]',
            'receipt_path_a="${output_dir}/seed-a/receipt.json"',
            'receipt_path_b="${output_dir}/seed-b/receipt.json"',
            'seed_dependent_local = {"homotopy", "cell_records", "continuity", "iar_core"}',
            'stable_a == stable_b',
            'homotopy_a == homotopy_b',
            'epsilon_a.get("raw_storage_sha256") != epsilon_b.get("raw_storage_sha256")',
            'actual_files == set(expected_files)',
            'post_checkpoint_identity == local_a.get("checkpoint_content_identity")',
            "BERNINI_IAR_DP2_SP4_CROSS_SEED_INPUT_AUDIT_OK",
            "cross_seed_numeric_stability_authorized=false",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("BERNINI_IAR_DP2_SP4_DUAL_SEED_STRONG_AUDIT_OK", self.source)
        self.assertNotRegex(self.source, r"(?i)stability[_ -]?pass")
        self.assertEqual(self.source.count("forwards_per_rank=84"), 1)
        self.assertGreaterEqual(self.source.count('local.get("forwards_per_rank") == 84'), 2)

    def test_launcher_neither_submits_jobs_nor_mutates_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive|checkout|switch)\b",
        )


if __name__ == "__main__":
    unittest.main()

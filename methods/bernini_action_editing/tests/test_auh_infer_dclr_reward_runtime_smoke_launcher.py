from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_dclr_reward_runtime_smoke.sbatch"


class AUHDCLRRewardRuntimeSmokeLauncherTests(unittest.TestCase):
    """Static checks for the single-group AUH raw-reward launcher."""

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

    def test_bash_syntax_and_embedded_python_are_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 6)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_requests_one_world4_ulysses4_calibration_group(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=24",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:4",
        ):
            self.assertIn(directive, self.source)
        self.assertNotIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertEqual(self.runner_region.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn("topology=WORLD4/Ulysses4 calibration_group=1", self.source)
        self.assertIn("allocated_gpus=4 training=false optimizer=false", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")
        self.assertNotRegex(
            self.runner_region,
            r"(?m)^\s*[^#\n]*torch\.distributed\.run[^\n]*&\s*$",
        )

    def test_launches_only_frozen_raw_reward_runner_not_training(self) -> None:
        self.assertIn(
            '"${method_root}/infer_dclr_reward_runtime_smoke.py"',
            self.runner_region,
        )
        self.assertEqual(
            self.runner_region.count(
                '"${method_root}/infer_dclr_reward_runtime_smoke.py"'
            ),
            1,
        )
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
        ):
            self.assertNotIn(forbidden, self.runner_region)

    def test_runner_receives_complete_current_cli_contract(self) -> None:
        fragments = (
            '--bernini-root "${bernini_root}"',
            '--veomni-root "${veomni_root}"',
            '--checkpoint "${checkpoint}"',
            '--checkpoint-content-manifest "${checkpoint_manifest}"',
            '--preprocessed-parquet-dir "${preprocessed_parquet_dir}"',
            '--dataset-summary "${dataset_summary}"',
            '--candidate-row-index "${candidate_row_index}"',
            '--wrong-source-row-index "${wrong_source_row_index}"',
            '--expected-candidate-iid "${candidate_iid}"',
            '--proposal-source-iid "${proposal_source_iid}"',
            '--expected-wrong-source-iid "${wrong_source_iid}"',
            '--wrong-source-clean-latent "${wrong_source_clean_latent}"',
            '--expected-wrong-source-clean-latent-sha256 "${wrong_source_clean_latent_sha256}"',
            '--wrong-source-provenance-receipt "${wrong_source_provenance_receipt}"',
            '--expected-wrong-source-provenance-receipt-sha256 "${wrong_source_provenance_receipt_sha256}"',
            '--expected-wrong-source-video-sha256 "${wrong_source_video_sha256}"',
            '--wrong-source-match-json "${wrong_source_match}"',
            '--expected-wrong-source-match-sha256 "${wrong_source_match_sha256}"',
            '--action-instruction "${action_instruction}"',
            '--expected-action-instruction-sha256 "${action_instruction_sha256}"',
            '--hard-negative-instruction "${hard_negative_instruction}"',
            '--expected-hard-negative-instruction-sha256 "${hard_negative_instruction_sha256}"',
            '--sigmas "${sigma_values[@]}"',
            '--sigma-weights "${sigma_weight_values[@]}"',
            '--noise-seed "${noise_seed}"',
            "--num-frames 81",
            '--expected-bernini-commit "${pinned_bernini_commit}"',
            '--expected-veomni-commit "${pinned_veomni_commit}"',
            '--expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"',
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
            '--output-receipt "${receipt_path}"',
        )
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.runner_region)

    def test_default_native_proposal_is_latent_and_receipt_provenance_bound(self) -> None:
        self.assertIn(
            'positive_control="${DCLR_REWARD_POSITIVE_CONTROL_PAIRED_TARGET:-0}"',
            self.source,
        )
        for environment_name in (
            "DCLR_REWARD_CANDIDATE_CLEAN_LATENT",
            "DCLR_REWARD_CANDIDATE_CLEAN_LATENT_SHA256",
            "DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT",
            "DCLR_REWARD_CORRECT_SOURCE_CLEAN_LATENT_SHA256",
            "DCLR_REWARD_CANDIDATE_ARM",
            "DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT",
            "DCLR_REWARD_CANDIDATE_PROVENANCE_RECEIPT_SHA256",
            "DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT",
            "DCLR_REWARD_SOURCE_PROVENANCE_RECEIPT_SHA256",
            "DCLR_REWARD_PROPOSAL_SOURCE_VIDEO_SHA256",
        ):
            self.assertGreaterEqual(self.source.count(environment_name), 2)
        for fragment in (
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
            'proposal_origin="native_rollout_predecode_latent"',
        ):
            self.assertIn(fragment, self.source)
        self.assertIn(
            '[[ "${candidate_arm}" == t2v || "${candidate_arm}" == r2v || "${candidate_arm}" == rv2v ]]',
            self.source,
        )

    def test_paired_target_requires_explicit_opt_in_and_rejects_native_inputs(self) -> None:
        native_branch = self.source.split(
            'if [[ "${positive_control}" == 0 ]]; then', 1
        )[1].split("read -r -a sigma_values", 1)[0]
        self.assertIn("--positive-control-paired-target", self.runner_region)
        self.assertEqual(
            self.runner_region.count("--positive-control-paired-target"), 1
        )
        self.assertIn(
            '[[ "${proposal_source_iid}" == "${candidate_iid}" ]]',
            native_branch,
        )
        self.assertIn(
            '[[ -z "${!forbidden_name:-}" ]] || fail', native_branch
        )
        self.assertIn(
            "positive-control launcher unexpectedly supplied native provenance",
            self.audit_region,
        )
        self.assertIn(
            'receipt.get("paired_target_positive_control") is paired',
            self.audit_region,
        )
        self.assertIn(
            'receipt.get("training_pair_authorized") is False',
            self.audit_region,
        )
        self.assertIn(
            'positive-control paired target authorized source calibration',
            self.audit_region,
        )

    def test_wrong_source_is_always_source_only_artifact_bound(self) -> None:
        for environment_name in (
            "DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT",
            "DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT",
            "DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_VIDEO_SHA256",
        ):
            self.assertGreaterEqual(self.source.count(environment_name), 1)
        for fragment in (
            'wrong.get("message_template_columns_loaded") == [',
            'wrong.get("paired_target_accessed") is False',
            'wrong.get("source_video_sha256") == wrong_source_video_sha256',
            'audit_artifact(\n    wrong_artifact,',
            'wrong_provenance.get("source_iid") == wrong_iid',
            'wrong_provenance.get("target_columns_accessed") == []',
            'wrong_provenance.get("target_media_accessed") is False',
            'wrong_provenance.get("paired_target_accessed") is False',
            '"bernini-source-only-vae-materialization-v1"',
            'receipt.get("wrong_source_paired_target_accessed") is False',
        ):
            self.assertIn(fragment, self.audit_region)
        positive_control_branch = self.source.split(
            'else\n  proposal_origin="paired_target_positive_control"', 1
        )[1].split("read -r -a sigma_values", 1)[0]
        self.assertNotIn("DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT", positive_control_branch)
        self.assertNotIn("DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT", positive_control_branch)

    def test_wrong_source_v2_separates_plumbing_from_calibration(self) -> None:
        for fragment in (
            'match.get("declared_use") in ("runtime_plumbing_only", "reward_calibration")',
            'match.get("scientific_eligibility") is all(criteria.values())',
            'match.get("source_reward_calibration_authorized") is (',
            'match.get("candidate_source_video_sha256") != match.get("wrong_source_video_sha256")',
            'match.get("candidate_source_video_sha256") == proposal_source_video_sha256',
            'candidate.get("message_template_columns_loaded") == [',
            'wrong.get("paired_target_accessed") is False',
            'if match["declared_use"] == "runtime_plumbing_only":',
            'receipt.get("source_reward_calibration_authorized") is False',
            '"same_camera_class"',
            '"same_composition_class"',
        ):
            self.assertIn(fragment, self.audit_region)
        for required_runtime_criterion in (
            "distinct_identity",
            "same_actor_class",
            "same_actor_count",
            "same_spatial_bucket",
            "same_length",
            "manual_reviewed",
        ):
            self.assertIn(required_runtime_criterion, self.audit_region)
        self.assertNotIn(
            "and all(criteria.get(key) is True for key in expected_criteria)",
            self.audit_region,
        )

    def test_method_archive_has_revision_hash_and_import_closure(self) -> None:
        for fragment in (
            'git get-tar-commit-id <"${source_archive}"',
            '[[ "${actual_archive_revision}" == "${source_revision}" ]]',
            'sha256sum "${source_archive}"',
            'sha256sum "${archive_copy}"',
            'member.issym() or member.islnk() or member.isfifo() or member.isdev()',
            "archive member escaped repository-relative closure",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'find "${method_root}" -type f -exec chmod a-w',
        ):
            self.assertIn(fragment, self.source)
        for required in (
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

    def test_checkpoint_dataset_text_and_wrong_source_are_content_bound(self) -> None:
        for environment_name in (
            "DCLR_REWARD_CHECKPOINT_TREE_SHA256",
            "DCLR_REWARD_CHECKPOINT_CONTENT_MANIFEST_SHA256",
            "DCLR_REWARD_DATASET_SUMMARY_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_MATCH_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_CLEAN_LATENT_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_PROVENANCE_RECEIPT_SHA256",
            "DCLR_REWARD_WRONG_SOURCE_VIDEO_SHA256",
            "DCLR_REWARD_ACTION_INSTRUCTION_SHA256",
            "DCLR_REWARD_HARD_NEGATIVE_INSTRUCTION_SHA256",
        ):
            self.assertIn(environment_name, self.source)
        for fragment in (
            'checkpoint_identity.get("verified_file_count") == 23',
            'checkpoint_identity.get("every_file_sha256_verified") is True',
            'local.get("dataset_summary_sha256") == dataset_summary_sha256',
            'match.get("file_sha256") == wrong_match_sha256',
            'candidate.get("action_instruction_sha256") == action_sha256',
            'text.get("hard_negative_instruction_sha256") == hard_sha256',
            'provenance.get("checkpoint_content_identity") == checkpoint_identity',
            'provenance.get("source_video_sha256") == proposal_source_video_sha256',
            'wrong_provenance.get("source_video_sha256") == wrong_source_video_sha256',
            'wrong_provenance.get("checkpoint_content_identity") == checkpoint_identity',
            'wrong_provenance.get("source_condition_latent_sha256") == wrong_latent_sha256',
        ):
            self.assertIn(fragment, self.audit_region)

    def test_rocm_devices_and_runtime_caches_are_isolated(self) -> None:
        for fragment in (
            'allocated_rocr_devices="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"',
            "WORLD4 calibration requires exactly four numeric ROCr devices",
            'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL',
            'export ROCR_VISIBLE_DEVICES="${allocated_rocr_devices}"',
            'export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${task_scratch}/cache/triton"',
            'export XDG_CACHE_HOME="${task_scratch}/cache/xdg"',
            'export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"',
            'export TORCHELASTIC_ERROR_FILE="${task_scratch}/torch-elastic-error.json"',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn('export HIP_VISIBLE_DEVICES="${allocated_rocr_devices}"', self.source)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES="${allocated_rocr_devices}"', self.source)

    def test_strong_audit_requires_identical_sp4_raw_evidence(self) -> None:
        for fragment in (
            "BERNINI_DCLR_RAW_REWARD_SP4_STRONG_AUDIT_OK",
            'distributed.get("world_size") == 4',
            'distributed.get("ulysses_size") == 4',
            'distributed.get("reward_reduction") == "none"',
            'distributed.get("all_gather_evidence_only") is True',
            'row.get("local_evidence_digest") == local_digest',
            'local.get("forward_implementation") == "renderer.get_t5_text_embeddings+renderer.diff_dec.shared_step"',
            'local.get("adapter_state") == "absent_frozen_base"',
            'local.get("raw_positive_conditional_only") is True',
            'local.get("cfg") is False',
            'local.get("apg") is False',
            'geometry.get("correct_wrong_full_rotary_exact_equal") is True',
            'geometry.get("rotary_dtype") == "torch.complex128"',
            'record.get("mode_shared_sigma_and_timestep") is True',
            'local.get("forwards_per_rank") == 4 * len(sigmas)',
            'audit_diagnostics(local.get("action_target_vs_hard_negative")',
            'audit_diagnostics(local.get("source_correct_vs_matched_wrong")',
        ):
            self.assertIn(fragment, self.audit_region)
        self.assertEqual(self.source.count("audit_receipt || fail"), 2)
        self.assertIn("receipt hash closure differs", self.source)

    def test_artifact_binding_uses_physical_tensor_identity_not_diagnostic_label(
        self,
    ) -> None:
        for fragment in (
            "def tensor_payload_identity(value):",
            '"raw_storage_sha256",',
            'artifact_identity.get("label") == role',
            "tensor_payload_identity(artifact_identity) == tensor_payload_identity(branch_identity)",
        ):
            self.assertIn(fragment, self.audit_region)
        self.assertNotIn(
            'artifact.get("tensor_identity") == identity',
            self.audit_region,
        )

    def test_launcher_neither_submits_jobs_nor_mutates_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive|checkout|switch)\b",
        )


if __name__ == "__main__":
    unittest.main()

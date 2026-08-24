from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_train_source_self_identity_orbit_v4.sbatch"


class AUHIdentityOrbitV4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)

    def test_bash_and_embedded_python_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 1)
        ast.parse(self.python_blocks[0])

    def test_archive_selftest_uses_discovery_not_a_tests_package_import(self) -> None:
        for fragment in (
            "-m unittest discover -v",
            '-s "${method_root}/tests"',
            "-p 'test_train_source_self_identity_orbit_v4_contract.py'",
            "-p 'test_source_self_native_rv2v_guidance.py'",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn(
            "tests.test_train_source_self_identity_orbit_v4_contract",
            self.source,
        )

    def test_one_node_uses_all_eight_cards_as_dp2_sp4(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --qos=bgqos",
            "#SBATCH --time=24:00:00",
        ):
            self.assertIn(directive, self.source)
        self.assertIn("--nproc_per_node=8", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_rocm_and_compiler_caches_are_job_private_and_writable(self) -> None:
        for fragment in (
            '"${task_scratch}/miopen-user"',
            '"${task_scratch}/miopen-custom"',
            'export MIOPEN_USER_DB_PATH="${task_scratch}/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${task_scratch}/torch-extensions"',
            'export TRITON_CACHE_DIR="${task_scratch}/triton"',
            'export TMPDIR="${task_scratch}/tmp"',
        ):
            self.assertIn(fragment, self.source)

    def test_mode_contract_is_prefix_sealed_or_complete_36_cycle(self) -> None:
        for fragment in (
            'mode="${BERNINI_CIO_MODE:?set BERNINI_CIO_MODE=sealed-prefix-canary or complete-cycle}"',
            "sealed-prefix-canary)",
            "max_steps >= 1 && max_steps <= 35",
            "--expected-prefix-digest",
            "--ack-incomplete-cycle-no-scientific-claim",
            "complete-cycle)",
            "max_steps % 36 == 0",
            "complete-cycle mode forbids a prefix digest",
            "prefix_seal_body(int(sys.argv[1]))",
            "prefix digest is not the extracted trainer's exact cycle prefix",
        ):
            self.assertIn(fragment, self.source)

    def test_dataset_is_externally_hash_bound_read_only_and_rechecked(self) -> None:
        for fragment in (
            "BERNINI_CIO_DATASET_ROOT",
            "BERNINI_CIO_DATASET_RECEIPT_SHA256",
            "BERNINI_CIO_SPEC_SHA256",
            "$'dataset.parquet\\nreceipt.json'",
            "dataset ${artifact} must be durable read-only",
            "dataset receipt hash differs",
            'dataset_parquet_before="$(sha256sum',
            'dataset_receipt_before="$(sha256sum',
            "dataset parquet changed",
            "dataset receipt changed",
        ):
            self.assertIn(fragment, self.source)

    def test_commit_archive_and_transitive_module_closure_are_bound(self) -> None:
        for fragment in (
            "BERNINI_ACTION_SOURCE_ARCHIVE",
            "BERNINI_ACTION_SOURCE_ARCHIVE_SHA256",
            "BERNINI_ACTION_SOURCE_REVISION",
            "BERNINI_ACTION_SOURCE_REPOSITORY",
            'git -C "${source_repository}" archive --format=tar',
            '"${source_revision}" methods/bernini_action_editing',
            "source archive is not the declared commit subtree",
            "method_tree_before=",
            "method_tree_after=",
            "extracted method tree changed",
        ):
            self.assertIn(fragment, self.source)
        required = (
            "appearance_counterfactual_identity_orbit.py",
            "source_self_identity_orbit_v4.py",
            "source_self_native_ref_contrastive_v3.py",
            "source_self_native_rv2v_guidance.py",
            "source_self_native_target_adapter.py",
            "source_self_runtime.py",
            "train_lora.py",
            "train_source_self_identity_orbit_v4.py",
            "tools/materialize_appearance_counterfactual_identity_orbit.py",
            "tests/test_train_source_self_identity_orbit_v4_contract.py",
            "tests/test_source_self_native_rv2v_guidance.py",
        )
        for relative in required:
            with self.subTest(relative=relative):
                self.assertRegex(
                    self.source,
                    rf"(?m)^  {re.escape(relative)}(?: \\|; do)\s*$",
                )

    def test_checkpoint_and_vendor_sources_are_fully_pinned(self) -> None:
        for assignment in (
            'expected_checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"',
            'expected_checkpoint_tree_sha256="6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca"',
            'expected_bernini_commit="2d2b4591ac053ec25c6371b01a5a6746679e5793"',
            'expected_veomni_commit="f90b3dc6fbb0ce693745223cc7a94064123dbf4d"',
        ):
            self.assertIn(assignment, self.source)
        for fragment in (
            "checkpoint content verification failed",
            'if [[ -d "${bernini_root}/.git" && ! -L "${bernini_root}/.git" ]]; then',
            "Bernini revision mismatch",
            "VeOmni revision mismatch",
            "Bernini tracked source is dirty",
            "VeOmni tracked source is dirty",
            "Bernini pinned file is invalid",
            "Bernini pinned file hash mismatch",
        ):
            self.assertIn(fragment, self.source)
        for binding in (
            "bernini/models/renderer.py:fec319f3ede3482b28873dc55622208f1242ecba0caedea8e710093748dc7159",
            "bernini/models/wan_diffusion.py:59e860ba3490a83f06bd4be75697490f49a118ee5ca969e85eea4dd7fa122512",
            "bernini/models/transformer_wan.py:9fb579611e79e0f534d5d6ccdcd956c35e57b4513c15267e8533ff3832a1f223",
            "bernini/models/scheduler.py:b6d729187fd784bf66831d5260a5c9482d89c452881d2f700c8887278f52ef97",
            "bernini/training/data.py:29aa4f89579c7771cb9f78706fde4f0dca0de954fdb2f5e2de1abacd8a0d6c65",
            "bernini/attention.py:e3986d1e5ba2e70f5244f53e77adbec705720be5cd2e9dbbde92f5aec1f99055",
            "bernini/parallel/state.py:32d784e7193297a599569da07c091b8d0a51ab08ad319ee2cfc0e495921db3aa",
            "bernini/parallel/ops.py:c264f28b7b011ce01204ec5b0f11acd08adb6568a9855108b866fb9ce1a2ce30",
            "configs/bernini_renderer_wan21_1p3b/config.json:4659e97bbb09f6c9baa3528dcdbb23064998e2f92aace8e8fd4b02776c529496",
        ):
            self.assertIn(binding, self.source)

    def test_torchrun_is_exact81_rho0_and_acknowledges_pretext_scope(self) -> None:
        for fragment in (
            "train_source_self_identity_orbit_v4.py",
            "--expected-dataset-receipt-sha256",
            "--expected-materialization-spec-sha256",
            "--rho 0",
            "--num-frames 81",
            "--ack-pretext-not-action-editing",
        ):
            self.assertIn(fragment, self.source)
        for forbidden in (
            "train_ramp_c0.py",
            "build_latent_locality_routing.py",
            "--num-frames 41",
            "gpu:mi210:4",
        ):
            self.assertNotIn(forbidden, self.source)

    def test_postcheck_requires_native_rv2v4_data_pack_and_adapter_binding(self) -> None:
        for fragment in (
            'reference_encoding.get("reference_count") != 4',
            'reference_encoding.get("reference_rgb_indices") != [0, 27, 53, 80]',
            'reference_encoding.get("independent_vae_encode_calls_per_row") != 15',
            'native_refs.get("native_mode") != "RV2V-4"',
            'native_refs.get("total_visual_condition_count") != 5',
            'native_refs.get("vi_image_source_ids") != [2.0, 3.0, 4.0, 5.0]',
            'native_refs.get("i_image_source_ids") != [1.0, 2.0, 3.0, 4.0]',
            '!= [1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 0.0]',
            'native_refs.get("patch_call_roles")',
            'native_refs.get("branch_concat_order")',
            'native_refs.get("rotary_concat_dim") != 2',
            '"bernini-native-rv2v-guidance-training-v2"',
            '"bernini-source-self-native-ref-contrastive-v4"',
            'native_guidance.get("native_rv2v4_reference_contract_digest")',
            'metadata.get("reference_rgb_indices_json") != "[0,27,53,80]"',
            'metadata.get("native_rv2v4_reference_contract_digest")',
            '"bernini-native-target-row-qo-lora-checkpoint-v2"',
        ):
            self.assertIn(fragment, self.source)

    def test_final_audit_checks_roundtrip_claims_and_no_action_claim(self) -> None:
        for fragment in (
            '"bernini-counterfactual-identity-orbit-training-receipt-v5"',
            'receipt.get("action_editing_claim_authorized") is not False',
            'receipt.get("scientific_claim_authorized") is not False',
            'receipt.get("long_training_automatically_submitted") is not False',
            'roundtrip.get("file_loaded_into_live_adapter") is not True',
            'roundtrip.get("strict_key_shape_dtype_value_roundtrip") is not True',
            'causal.get("all_dp_gates_passed") is not True',
            'causal.get("wrong_scene_iids_may_be_other_training_cohort_rows") is not True',
            'causal.get("cross_iid_generalization_claim_authorized") is not False',
            'metadata.get("gradient_checkpointing_enabled") != "false"',
            'metadata.get("rho_hex") != float(0.0).hex()',
        ):
            self.assertIn(fragment, self.source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT / "scripts" / "auh_infer_full644_camera_beta_dual4.sbatch"
)


class AUHFull644CameraBetaDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        cls.launch_region = cls.source.split("launch_arm() (", 1)[1].split(
            "beta050_status=0", 1
        )[0]

    def test_bash_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedded_python_blocks_are_valid_ast(self) -> None:
        self.assertGreaterEqual(len(self.python_blocks), 6)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_one_node_eight_mi210_runs_two_ulysses_four_arms(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
        ):
            self.assertIn(directive, self.source)
        self.assertEqual(self.launch_region.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn("topology=dual-4-Ulysses", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_default_betas_are_overridable_bounded_and_distinct(self) -> None:
        for fragment in (
            'beta050="${BERNINI_CAMERA_BETA050:-0.5}"',
            'beta100="${BERNINI_CAMERA_BETA100:-1.0}"',
            '"${python_bin}" -B - "${beta050}" "${beta100}"',
            'if not math.isfinite(value) or not 0.0 <= value <= 1.0:',
            'raise SystemExit("fixed-beta arms must use distinct beta values")',
            'print(*(f"beta{format(value, \'.17g\').replace(\'.\', \'p\')}" for value in values))',
            'read -r beta050_slug beta100_slug <<<"${beta_slug_pair}"',
            '[[ "${beta050_slug}" != "${beta100_slug}" ]]',
            "beta_override_env=BERNINI_CAMERA_BETA050,BERNINI_CAMERA_BETA100",
            '--beta "${beta}"',
        ):
            self.assertIn(fragment, self.source)

    def test_camera_estimator_defaults_to_global_and_is_strictly_selected(self) -> None:
        for fragment in (
            'camera_estimator="${BERNINI_CAMERA_ESTIMATOR:-global_svd}"',
            'case "${camera_estimator}" in',
            "global_svd|grid_consensus) ;;",
            'fail "BERNINI_CAMERA_ESTIMATOR must be exactly global_svd or grid_consensus"',
            "camera_estimator=${camera_estimator} estimator_override_env=BERNINI_CAMERA_ESTIMATOR",
            '--camera-estimator "${camera_estimator}"',
            'require(args.camera_estimator in ("global_svd", "grid_consensus"), "audit camera estimator is unsupported")',
        ):
            self.assertIn(fragment, self.source)
        self.assertEqual(
            self.source.count('--camera-estimator "${camera_estimator}"'),
            2,
        )

    def test_gpu_groups_rendezvous_and_caches_are_isolated(self) -> None:
        for fragment in (
            'beta050_visible_gpus="0,1,2,3"',
            'beta100_visible_gpus="4,5,6,7"',
            'export ROCR_VISIBLE_DEVICES="${visible_gpus}"',
            'beta050_master_port=$((40000 + 2 * job_mod))',
            'beta100_master_port=$((beta050_master_port + 1))',
            '[[ "${beta050_master_port}" -ne "${beta100_master_port}" ]]',
            '--master_port="${master_port}"',
            'local arm_root="${task_scratch}/arms/${arm}"',
            'export MIOPEN_USER_DB_PATH="${arm_root}/cache/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${arm_root}/cache/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${arm_root}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${arm_root}/cache/triton"',
            'export XDG_CACHE_HOME="${arm_root}/cache/xdg"',
            'export PYTHONPYCACHEPREFIX="${arm_root}/cache/pycache"',
            'export TMPDIR="${arm_root}/tmp"',
            'export TORCHELASTIC_ERROR_FILE="${arm_root}/torch-elastic-error.json"',
        ):
            self.assertIn(fragment, self.source)

    def test_both_arms_launch_concurrently_and_any_failure_fails_job(self) -> None:
        for fragment in (
            'launch_arm beta050 "${beta050}" "${beta050_visible_gpus}"',
            'launch_arm beta100 "${beta100}" "${beta100_visible_gpus}"',
            "beta050_pid=$!",
            "beta100_pid=$!",
            'wait "${beta050_pid}" || beta050_status=$?',
            'wait "${beta100_pid}" || beta100_status=$?',
            '[[ "${beta050_status}" -ne 0 || "${beta100_status}" -ne 0 ]]',
            'fail "one or more camera-stabilizer arms failed"',
        ):
            self.assertIn(fragment, self.source)
        self.assertRegex(
            self.source,
            r"launch_arm beta050 .* >\"\$\{beta050_log\}\" 2>&1 &\nbeta050_pid=\$!",
        )
        self.assertRegex(
            self.source,
            r"launch_arm beta100 .* >\"\$\{beta100_log\}\" 2>&1 &\nbeta100_pid=\$!",
        )

    def test_exact_input_adapter_and_sampling_identity_is_hard_bound(self) -> None:
        for fragment in (
            'expected_source_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"',
            'expected_instruction_sha256="105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"',
            'expected_adapter_config_sha256="b91c3a236b0e0e893e7c992be043ec28cfa05c73b7792c0b93b4013db15aef39"',
            'expected_adapter_model_sha256="9217ff653e47f915105fe8fa64856037d63811562cec1e9fd53ae9e4613a9774"',
            'expected_adapter_receipt_sha256="5931f7544d1bd185adf3fc07edb046e6bf27811b0835de8446f91c8a25c782c6"',
            'expected_training_receipt_digest="6b5f2a053be048881b1426d9b7c4c380dc8b82f6098bfbda9c80034b26df17d1"',
            "expected_frames=81",
            "expected_fps=25",
            "expected_steps=40",
            "expected_seed=2027",
            '[[ "${training_global_step}" == 644 ]]',
            '--expected-source-sha256 "${expected_source_sha256}"',
            '--expected-instruction-sha256 "${expected_instruction_sha256}"',
        ):
            self.assertIn(fragment, self.source)

    def test_checkpoint_and_upstream_are_strictly_pinned(self) -> None:
        for digest in (
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670",
        ):
            self.assertIn(digest, self.source)
        self.assertIn("checkpoint_manifest_file_count=23", self.source)
        self.assertGreaterEqual(
            self.source.count('validate_checkpoint_content "${checkpoint_manifest}"'),
            2,
        )
        self.assertGreaterEqual(self.source.count("train_lora.validate_source_trees"), 2)
        self.assertIn("sha256sum --strict --status -c", self.source)
        self.assertIn("VeOmni tracked source is dirty", self.source)

    def test_archive_revision_is_self_authenticating_without_method_repo(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_CAMERA_DUAL4_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_CAMERA_DUAL4_SOURCE_REVISION:',
            'git get-tar-commit-id <"${source_archive}"',
            'git get-tar-commit-id <"${archive_copy}"',
            "staged method archive revision differs",
            "archive member escaped method subtree",
            "member.issym() or member.islnk() or member.isfifo() or member.isdev()",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'find "${method_root}" -type f -exec chmod a-w',
            'method_tree_digest_pre="$(method_tree_digest "${method_root}")"',
            'method_tree_digest_post="$(method_tree_digest "${method_root}")"',
            "extracted method tree changed during inference",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("BERNINI_CAMERA_DUAL4_SOURCE_REPOSITORY", self.source)
        self.assertNotIn("method-from-revision.tar", self.source)
        self.assertNotIn('git -C "${source_repository}"', self.source)

    def test_runner_cli_has_only_source_and_instruction_external_conditions(self) -> None:
        for fragment in (
            '"${method_root}/infer_full644_camera_stabilizer.py"',
            '--adapter-checkpoint "${arm_root}/input/adapter"',
            '--source-video "${arm_root}/input/source/source.mp4"',
            '--instruction "${instruction}"',
            '--output "${arm_output}"',
            '--beta "${beta}"',
            '--camera-estimator "${camera_estimator}"',
            '--num-inference-steps "${expected_steps}"',
            '--seed "${expected_seed}"',
        ):
            self.assertIn(fragment, self.launch_region)
        for forbidden in (
            "--target",
            "--support",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--reference",
            "--first-frame",
            "--swept-tube",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.launch_region)
        self.assertIn(
            "semantic_inputs=source_video,edit_instruction "
            "target=false support=false mask=false flow=false pose=false "
            "track=false trajectory=false reference=false first_frame=false",
            self.source,
        )

    def test_outputs_receipts_and_logs_are_distinct_and_nonoverwriting(self) -> None:
        for fragment in (
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse camera dual4 output directory"',
            'beta050_output="${output_dir}/${beta050_slug}.mp4"',
            'beta100_output="${output_dir}/${beta100_slug}.mp4"',
            'beta050_receipt="${beta050_output}.receipt.json"',
            'beta100_receipt="${beta100_output}.receipt.json"',
            'beta050_log="${output_dir}/${beta050_slug}.torchrun.log"',
            'beta100_log="${output_dir}/${beta100_slug}.torchrun.log"',
            '[[ ! -e "${candidate}" && ! -L "${candidate}" ]]',
            "preflight_nonoverwrite_not_atomic_create_only",
            "camera dual4 output directory must contain two mp4, two receipts, and two logs",
        ):
            self.assertIn(fragment, self.source)

    def test_strong_audit_covers_camera_execution_receipt(self) -> None:
        for fragment in (
            "FULL644_CAMERA_BETA_DUAL4_STRONG_AUDIT_OK",
            'receipt.get("schema_version") == "bernini-full644-camera-tangent-inference-v1"',
            'receipt.get("method") == "full644-generator-native-camera-tangent-stabilizer-v1"',
            'input_contract.get("accepted_external_conditions") == ["source_video", "edit_instruction"]',
            'adapter_identity.get("adapter_tensor_count") == 480',
            '"strict_480_tensor_reload",',
            'math.isclose(float(camera.get("beta")), beta',
            'camera.get("estimator") == args.camera_estimator',
            'camera.get("source_and_instruction_only") is True',
            'certificate.get("step_count") == 40',
            'certificate.get("transformer_forwards") == 120',
            'certificate.get("official_action_apg_exact_steps") == 40',
            'certificate.get("camera_callback_calls") == 40',
            'certificate.get("original_unipc_calls") == 40',
            'certificate.get("camera_estimator") == args.camera_estimator',
            'certificate.get("beta_zero_exact_full644_passthrough") is (beta == 0.0)',
            'expected_basis_builds = int(active and args.camera_estimator == "global_svd")',
            'expected_basis_reuses = 39 if active and args.camera_estimator == "global_svd" else 0',
            'expected_geometry_builds = int(active and args.camera_estimator == "grid_consensus")',
            'expected_geometry_reuses = 39 if active and args.camera_estimator == "grid_consensus" else 0',
            'certificate.get("camera_basis_build_count") == expected_basis_builds',
            'certificate.get("camera_basis_reuse_count") == expected_basis_reuses',
            'certificate.get("camera_geometry_build_count") == expected_geometry_builds',
            'certificate.get("camera_geometry_reuse_count") == expected_geometry_reuses',
            'camera_trace.get("estimator") == args.camera_estimator',
            'rank_authority.get("world_size") == 4',
            'rank_authority.get("receipt_group_rank") == 0',
            'rank_authority.get("rank0_receipt_aggregated_steps") == 40',
            'rank_authority.get("source_clean_cross_rank_exact") is True',
            'rank_authority.get("source_clean_exact_certification_steps") == 1',
            'rank_authority.get("action_clean_cross_rank_exact_steps") == 40',
            'rank_authority.get("all_rank_candidate_success_steps") == 40',
            'rank_authority.get("rank0_authoritative_broadcast_steps") == 40',
            'rank_authority.get("post_broadcast_exact_steps") == 40',
            'rank_authority.get("zero_beta_no_collective_steps") == 40',
            'rank_authority.get("large_object_all_gather") is False',
            'row.get("grid_rank_authority") is None',
            'row.get("grid_rank_authority", {}).get("post_broadcast_exact") is True',
            'row.get("basis_built_this_step")',
            'row.get("basis_reused_from_prior_step")',
            'row.get("geometry_built_this_step")',
            'row.get("geometry_reused_from_prior_step")',
            'certificate.get("custom_integrator") is False',
            'sampling.get("num_frames") == 81',
            'sampling.get("num_inference_steps") == 40',
            'sampling.get("seed") == 2027',
            'sampling.get("ulysses_size") == 4',
        ):
            self.assertIn(fragment, self.source)

    def test_archive_closure_and_preflight_tests_include_camera_runner(self) -> None:
        required = (
            "generator_native_camera_stabilizer.py",
            "fixed_grid_camera_consensus.py",
            "fixed_grid_camera_consensus_stabilizer.py",
            "infer_full644_camera_stabilizer.py",
            "tri_branch_unipc.py",
            "infer_lora.py",
            "motion_residual.py",
            "train_lora.py",
            "tools/materialize_vae.py",
            "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
            "scripts/auh_infer_full644_camera_beta_dual4.sbatch",
            "tests/test_generator_native_camera_stabilizer.py",
            "tests/test_fixed_grid_camera_consensus.py",
            "tests/test_fixed_grid_camera_consensus_stabilizer.py",
            "tests/test_infer_full644_camera_stabilizer_contract.py",
            "tests/test_auh_infer_full644_camera_beta_dual4_launcher.py",
        )
        for path in required:
            with self.subTest(path=path):
                self.assertRegex(self.source, rf"(?m)^  {re.escape(path)}(?: \\)?$")
        self.assertIn(
            '[[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]]',
            self.source,
        )
        for test_name in (
            "test_generator_native_camera_stabilizer.py",
            "test_fixed_grid_camera_consensus.py",
            "test_fixed_grid_camera_consensus_stabilizer.py",
            "test_infer_full644_camera_stabilizer_contract.py",
            "test_auh_infer_full644_camera_beta_dual4_launcher.py",
        ):
            self.assertGreaterEqual(self.source.count(test_name), 2)
        self.assertEqual(self.source.count("test_infer_lora_contract.py"), 1)

    def test_launcher_does_not_submit_or_mutate_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive)\b",
        )


if __name__ == "__main__":
    unittest.main()

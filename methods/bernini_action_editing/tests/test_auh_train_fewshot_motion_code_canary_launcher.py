from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_train_fewshot_motion_code_canary.sbatch"
FFPROBE_HELPER = METHOD_ROOT / "tools" / "ffprobe_pyav_compat.py"
TRAINER = METHOD_ROOT / "train_fewshot_motion_code.py"
TRAINER_CONTRACT_TEST = METHOD_ROOT / "tests" / "test_train_fewshot_motion_code_contract.py"
EPISODE_PARALLEL = METHOD_ROOT / "fewshot_episode_parallel.py"
EPISODE_PARALLEL_TEST = METHOD_ROOT / "tests" / "test_fewshot_episode_parallel.py"


class AUHFewShotMotionCodeCanaryLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)

    def test_bash_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_embedded_python_audit_is_valid_ast(self) -> None:
        self.assertGreaterEqual(len(self.python_blocks), 4)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_slurm_contract_is_one_node_eight_mi210(self) -> None:
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
        self.assertNotIn("#SBATCH --qos=stqos", self.source)
        self.assertNotIn("#SBATCH --qos=gtqos", self.source)
        self.assertIn("--nproc_per_node=8", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_mode_is_explicit_and_maps_to_exact_artifact_sets(self) -> None:
        self.assertIn(
            'mode="${BERNINI_EPMC_MODE:?set BERNINI_EPMC_MODE=smoke or formal}"',
            self.source,
        )
        self.assertIn("smoke)\n    mode_args=(--engineering-smoke)", self.source)
        self.assertIn('receipt_name="smoke.receipt.json"', self.source)
        self.assertIn("expected_output_count=2", self.source)
        self.assertIn("formal)\n    mode_args=()", self.source)
        self.assertIn('receipt_name="receipt.json"', self.source)
        self.assertIn("expected_output_count=5", self.source)
        self.assertIn(
            "BERNINI_EPMC_MODE must be exactly smoke or formal", self.source
        )
        for fragment in (
            "for name in diagnostics.pt smoke.receipt.json",
            "for name in diagnostics.pt training_go_receipt.json prototype.safetensors prototype.receipt.json receipt.json",
            'expected_files = (\n    {"diagnostics.pt", "smoke.receipt.json"}',
            '"training_go_receipt.json",\n        "prototype.safetensors",\n        "prototype.receipt.json",\n        "receipt.json",',
        ):
            self.assertIn(fragment, self.source)

    def test_exact_config_and_dataset_hashes_are_frozen(self) -> None:
        for assignment in (
            'expected_config_sha256="a46d18fce025b0cd3b30a6505514b817cd5d96c43d305d6202405a952eed2446"',
            'expected_preview_manifest_sha256="49506e003f86f319ebe8a5e843d19c88cef75e84cd4250968da283bb19252e47"',
            'expected_vae_index_sha256="d36fb5de3487ba5bf494589948430a60e214851d29776cc4f439e4e2d54ee52b"',
            'expected_trainer_sha256="53424538b0685e5e9a226d5b9729bf61546116fb4d853298fe9475ce06cb4851"',
            'expected_trainer_contract_test_sha256="e2f7e6c1b4b7b1795b75240eabe3acf443ace27b6622e6897c3e2cda0f7bc49f"',
            'expected_episode_parallel_sha256="220fc85c24fe9cf781e5908cfc548e5ca4acf54ad87a5c00bab7b2cdbefa70ec"',
            'expected_episode_parallel_test_sha256="a37d2242516bcb911db311c5c96660410c1cb66163bb1e7515b20a67bbc75415"',
        ):
            self.assertIn(assignment, self.source)
        for name in (
            "BERNINI_EPMC_PREVIEW_MANIFEST",
            "BERNINI_EPMC_VAE_INDEX",
        ):
            self.assertIn(f"${{{name}:?set {name}}}", self.source)
        for fragment in (
            "preview manifest hash mismatch",
            "VAE index hash mismatch",
            "staged preview manifest hash differs",
            "staged VAE index hash differs",
            "preview manifest changed",
            "VAE index changed",
            "selected_artifact_hashes_verified",
            "selected_media_probed",
            "final-stable trainer hash differs",
            "final-stable trainer contract test hash differs",
            "final-stable episode-parallel module hash differs",
            "final-stable episode-parallel test hash differs",
            "final-stable trainer changed",
            "final-stable trainer contract test changed",
            "final-stable episode-parallel module changed",
            "final-stable episode-parallel test changed",
            'episode_parallel_path="${method_root}/fewshot_episode_parallel.py"',
            'episode_parallel_test_path="${method_root}/tests/test_fewshot_episode_parallel.py"',
        ):
            self.assertIn(fragment, self.source)

    def test_embedded_source_hashes_match_current_bytes(self) -> None:
        pinned_files = {
            "expected_trainer_sha256": TRAINER,
            "expected_trainer_contract_test_sha256": TRAINER_CONTRACT_TEST,
            "expected_episode_parallel_sha256": EPISODE_PARALLEL,
            "expected_episode_parallel_test_sha256": EPISODE_PARALLEL_TEST,
        }
        for variable, path in pinned_files.items():
            with self.subTest(variable=variable, path=path):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertIn(f'{variable}="{digest}"', self.source)

    def test_ffprobe_backend_is_task_private_pinned_pyav_shim(self) -> None:
        helper_sha256 = hashlib.sha256(FFPROBE_HELPER.read_bytes()).hexdigest()
        self.assertIn(
            'python_bin="${BERNINI_EPMC_PYTHON_BIN:?set BERNINI_EPMC_PYTHON_BIN to the vace Python with PyAV 13.1.0}"',
            self.source,
        )
        for assignment in (
            f'expected_ffprobe_helper_sha256="{helper_sha256}"',
            'expected_pyav_version="13.1.0"',
            'episode_probe_iid="841b5e0080a1441d"',
            'episode_probe_source_sha256="5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a"',
        ):
            self.assertIn(assignment, self.source)
        for fragment in (
            'ffprobe_helper="${method_root}/tools/ffprobe_pyav_compat.py"',
            'ffprobe_bin="${task_scratch}/bin/ffprobe"',
            'cp -- "${ffprobe_helper}" "${ffprobe_bin}"',
            'chmod 0555 -- "${ffprobe_bin}"',
            '[[ -f "${ffprobe_bin}" && ! -L "${ffprobe_bin}" && -x "${ffprobe_bin}" ]]',
            '[[ "$(stat -c \'%a\' "${ffprobe_bin}")" == 555 ]]',
            'actual_sha256="$(sha256sum "${ffprobe_bin}"',
            'export BERNINI_EPMC_PYTHON_BIN="${python_bin}"',
            'import av; print(av.__version__)',
            'export PATH="${task_scratch}/bin${PATH:+:${PATH}}"',
            "hash -r",
            'resolved="$(command -v ffprobe || true)"',
            '[[ "${resolved}" == "${ffprobe_bin}" ]]',
            "media_probe_backend=pyav_ffprobe_shim",
            "ffprobe_helper_sha256=${actual_sha256}",
            "pyav_version=${actual_pyav_version}",
        ):
            self.assertIn(fragment, self.source)
        for phase in ("installed", "pre_torchrun", "post_torchrun", "final"):
            self.assertIn(f"validate_ffprobe_shim {phase}", self.source)
        self.assertEqual(self.source.count("validate_ffprobe_shim "), 4)
        self.assertNotIn("/usr/bin/ffprobe", self.source)
        self.assertNotIn("BERNINI_EPMC_FFPROBE_BIN", self.source)
        self.assertNotIn("expected_ffprobe_version", self.source)

    def test_real_81_frame_episode_probe_precedes_torchrun(self) -> None:
        for fragment in (
            'episode_source_path="$("${python_bin}" -B -',
            '"${k2_config}"',
            '"${staged_preview_manifest}"',
            '"${episode_probe_iid}"',
            '"${episode_probe_source_sha256}"',
            "probe IID is not unique in the K2 config",
            "probe IID is not unique in the preview manifest",
            "pinned episode source hash differs before real media probe",
            "from fewshot_episode_io import _probe_video_metadata",
            "metadata = _probe_video_metadata(source)",
            "metadata.frame_count != expected_frames",
            "metadata.fps != Fraction(25, 1)",
            "episode_source_probe=true frame_count={metadata.frame_count}",
        ):
            self.assertIn(fragment, self.source)
        probe_index = self.source.index('episode_source_path="$(')
        torchrun_index = self.source.index(
            'PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run'
        )
        self.assertLess(
            self.source.index("validate_ffprobe_shim pre_torchrun"), probe_index
        )
        self.assertLess(probe_index, torchrun_index)

    def test_commit_archive_is_byte_bound_safely_extracted_and_immutable(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_ACTION_SOURCE_REVISION:',
            'source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:',
            'git -C "${source_repository}" archive --format=tar',
            '"${source_revision}" methods/bernini_action_editing',
            "method archive bytes are not the declared commit",
            "member.issym() or member.islnk() or member.isfifo() or member.isdev()",
            "archive member escaped method subtree",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'method_root="${task_scratch}/source/methods/bernini_action_editing"',
            'find "${method_root}" -type f -exec chmod a-w',
            'method_tree_digest_pre="$(method_tree_digest "${method_root}")"',
            'method_tree_digest_post="$(method_tree_digest "${method_root}")"',
            "extracted method tree changed during training",
            "durable method archive changed",
        ):
            self.assertIn(fragment, self.source)

    def test_required_archive_files_cover_full_transitive_import_closure(self) -> None:
        required_modules = {
            "train_fewshot_motion_code.py",
            "fewshot_episode_io.py",
            "fewshot_episode_parallel.py",
            "inference_sigma_strata.py",
            "motion_residual.py",
            "train_lora.py",
            "train_prior_tangent_lora.py",
            "prior_guided_tangent.py",
            "train_delta_lora.py",
            "tri_branch_unipc.py",
            "counterfactual_proposal_motion_rebinding.py",
            "counterfactual_proposal_motion_branch.py",
            "counterfactual_proposal_motion_runtime.py",
            "fewshot_proposal_motion_carrier.py",
            "fewshot_motion_branch.py",
            "fewshot_privileged_motion_code.py",
            "fewshot_teacher_objective.py",
            "infer_fewshot_motion_code.py",
            "infer_counterfactual_proposal_motion_oracle.py",
            "infer_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "source_kv_route_batches.py",
            "source_kv_replay.py",
            "source_value_residual.py",
            "tools/ffprobe_pyav_compat.py",
        }
        for path in required_modules:
            with self.subTest(path=path):
                self.assertRegex(self.source, rf"(?m)^  {re.escape(path)} \\\s*$")
        for path in (
            "configs/epmc_sit_turn_head_k2_v1.json",
            "scripts/auh_train_fewshot_motion_code_canary.sbatch",
            "tests/test_train_fewshot_motion_code_contract.py",
            "tests/test_fewshot_episode_parallel.py",
            "tests/test_fewshot_proposal_motion_carrier.py",
            "tests/test_ffprobe_pyav_compat.py",
            "tests/test_auh_train_fewshot_motion_code_canary_launcher.py",
        ):
            self.assertIn(f"  {path}", self.source)
        self.assertIn(
            '[[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]]',
            self.source,
        )

    def test_checkpoint_and_upstream_sources_are_fail_closed(self) -> None:
        for value in (
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670",
        ):
            self.assertIn(value, self.source)
        self.assertIn("checkpoint_content_file_count=23", self.source)
        self.assertGreaterEqual(
            self.source.count('validate_checkpoint_content "${checkpoint_manifest}"'), 2
        )
        self.assertIn("sha256sum --strict --status -c", self.source)
        self.assertIn("VeOmni tracked source is dirty", self.source)
        self.assertIn("Bernini archive hash mismatch", self.source)

    def test_output_is_fresh_canonical_and_mode_labeled(self) -> None:
        self.assertIn(
            'output_dir="${BERNINI_EPMC_TRAIN_OUTPUT_DIR:?set BERNINI_EPMC_TRAIN_OUTPUT_DIR}"',
            self.source,
        )
        for fragment in (
            '[[ "${output_name}" =~ (^|[-_])${mode}($|[-_]) ]]',
            "output basename must explicitly contain the smoke/formal mode label",
            'canonical_output_dir="$(realpath -m -- "${output_dir}")"',
            '[[ "${canonical_output_dir}" == "${output_dir}" ]]',
            "output must be canonical and contain no symlink traversal",
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]]',
            "refusing to reuse training output",
        ):
            self.assertIn(fragment, self.source)

    def test_torchrun_uses_exact_fixed_81_frame_training_cli(self) -> None:
        for assignment in (
            "num_frames=81",
            "k_shot=2",
            "steps_per_support=50",
            'learning_rate="0.05"',
            'max_grad_norm="1.0"',
            "training_seed=20260808",
            "proposal_seed=2027",
            "fixed_sigma_index=20",
            "held_sigma_index=32",
            'full_target_fm_weight="0.0"',
        ):
            self.assertIn(assignment, self.source)
        command_region = self.source.split(
            'PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run', 1
        )[1].split('[[ -d "${output_dir}"', 1)[0]
        for fragment in (
            "--standalone",
            "--nproc_per_node=8",
            '"${method_root}/train_fewshot_motion_code.py"',
            '--num-frames "${num_frames}"',
            '--k-shot "${k_shot}"',
            '--steps-per-support "${steps_per_support}"',
            '--learning-rate "${learning_rate}"',
            '--max-grad-norm "${max_grad_norm}"',
            '--seed "${training_seed}"',
            '--proposal-seed "${proposal_seed}"',
            '--fixed-sigma-index "${fixed_sigma_index}"',
            '--held-sigma-index "${held_sigma_index}"',
            '--full-target-fm-weight "${full_target_fm_weight}"',
            "--ack-preview-experimental-only",
            '--expected-k2-config-sha256 "${expected_config_sha256}"',
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
            '"${mode_args[@]}"',
        ):
            self.assertIn(fragment, command_region)
        self.assertNotIn("--posthoc-heldout-eval", command_region)

    def test_receipt_is_strictly_rehashed_and_audited_twice(self) -> None:
        for fragment in (
            'receipt.get("schema_version") == "bernini-epmc-k2-code-inversion-receipt-v2"',
            'go_receipt.get("schema_version") == "bernini-epmc-k2-representability-gate-v1"',
            'receipt.get("posthoc_heldout") is None',
            'receipt.get("prototype_frozen_before_heldout_deserialization") is True',
            'receipt.get("heldout_target_latent_deserialized") is False',
            "args.world_size == 8",
            "args.ulysses_size == 4",
            "args.data_parallel_size == 2",
            '"sp_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]',
            '"dp_groups": [[0, 4], [1, 5], [2, 6], [3, 7]]',
            '"support_parallel": True',
            '"support_index": 1,\n        "iid": expected_support_iids[0],\n        "dp_rank": 0,\n        "sp_ranks": [0, 1, 2, 3]',
            '"support_index": 2,\n        "iid": expected_support_iids[1],\n        "dp_rank": 1,\n        "sp_ranks": [4, 5, 6, 7]',
            '"cross_support_gradient_sync": False',
            'receipt.get("distributed") == expected_distributed',
            'receipt.get("support_count") == 2',
            'receipt.get("optimizer_steps") == 2',
            'receipt.get("optimizer_steps_per_support") == 1',
            'observed.get("steps") == expected_support_steps',
            'observed.get("gradient_sync") == args.within_support_gradient_sync',
            'support_code_sha256 = [item["tied_36d_code_sha256"] for item in support]',
            'go_receipt.get("support_tied_36d_code_sha256") == support_code_sha256',
            'go_receipt.get("reference_gradient_probe_support_index") == 1',
            'len(probes) != 2',
            '"phase_only", True',
            '"block_only", True',
            "support-1-only reference gradient probes differ",
            'len(held_controls) != 2',
            '[item.get("iid") for item in held_controls] != expected_support_iids',
            "exact two-support held-noise controls differ",
            "training gate/run consensus evidence differs",
            'receipt.get("model", {}).get("trainable_dimension_per_support") == 36',
            "inference.load_prototype_bundle(",
            "audit_receipt || fail \"result receipt audit round 1 failed\"",
            "audit_receipt || fail \"result receipt audit round 2 failed\"",
            "receipt changed after audit round 1",
            "receipt changed after audit round 2",
            "output tree changed after audit round 1",
            "output tree changed after audit round 2",
        ):
            self.assertIn(fragment, self.source)
        self.assertEqual(self.source.count("audit_receipt || fail"), 2)
        for iid, source_sha, target_sha, parquet_sha in (
            (
                "841b5e0080a1441d",
                "5f354b6b0f5cf49bf14d57a359bad03e90263d1a3965a57b1b89ce1a707f492a",
                "c26970c7d263587db5e5f96b98b30072d7f4b30bc242f9ef581e5780d7c9f070",
                "51b30d41032fa443e667261af2ffeb8e9e1875338c41655c8f5e8e9ecd37fdc1",
            ),
            (
                "7262dd490cbf42c5",
                "f2807db8eda17c9657fe33552d1c58b208af08f88394396a5b4acedbec4f3548",
                "1bc20ee874314a60914a71554f109d38dfc4b02c706f9eabb0dc46e92b2879bc",
                "23924dea163e45ed1aaae7ebce36eb689304bf630200c9662766437a2141c69a",
            ),
            (
                "7b88a1ca1f804f41",
                "4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed",
                "8234f5f35f7001134cf074263c481e3a8079c10f799370090d30e054aef02015",
                "a84fc017ce59de301a9b31208a358a1559f24c50c4f14c2e8d7990a8f2ad9e44",
            ),
        ):
            for value in (iid, source_sha, target_sha, parquet_sha):
                self.assertIn(value, self.source)

    def test_launcher_neither_submits_nor_commits(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(self.source, r"(?m)^\s*git\s+(?:add|commit|push)\b")
        self.assertIn("test_train_fewshot_motion_code_contract.py", self.source)
        self.assertIn("test_fewshot_episode_parallel.py", self.source)
        self.assertIn("test_fewshot_proposal_motion_carrier.py", self.source)
        self.assertIn("test_ffprobe_pyav_compat.py", self.source)
        self.assertIn(
            "test_auh_train_fewshot_motion_code_canary_launcher.py", self.source
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_train_generator_native_trajectory_controller_canary.sbatch"
)


class EGNTCAUHLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def test_requests_one_complete_eight_gpu_node(self) -> None:
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.source)
        self.assertIn("#SBATCH --qos=bgqos", self.source)
        self.assertIn("--nproc_per_node=8", self.source)
        self.assertIn("world=8 DP=2 Ulysses=4", self.source)

    def test_locks_exact_81_frame_forty_step_program(self) -> None:
        self.assertIn("--num-frames 81", self.source)
        self.assertIn("--num-inference-steps 40", self.source)
        self.assertIn("--k-shot 2", self.source)
        self.assertNotIn("--num-frames 41", self.source)
        self.assertIn("forwards_per_support_rollout=120", self.source)
        self.assertIn("full_resolution_cache=1x16x21x60x62", self.source)

    def test_source_data_and_output_are_bound_and_non_overwriting(self) -> None:
        for variable in (
            "BERNINI_EGNTC_SOURCE_ARCHIVE",
            "BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256",
            "BERNINI_EGNTC_SOURCE_REVISION",
            "BERNINI_EGNTC_PREVIEW_MANIFEST",
            "BERNINI_EGNTC_VAE_INDEX",
            "BERNINI_EGNTC_OUTPUT_DIR",
        ):
            self.assertIn(variable, self.source)
        self.assertIn('[[ ! -e "${output_dir}"', self.source)
        self.assertIn('sha256sum "${source_archive}"', self.source)
        self.assertIn('source archive must be read-only', self.source)
        self.assertIn('git get-tar-commit-id <"${source_archive}"', self.source)
        self.assertIn('sha256sum "${task_scratch}/source.tar"', self.source)
        self.assertIn('git get-tar-commit-id <"${task_scratch}/source.tar"', self.source)
        self.assertIn('sha256sum "${preview_manifest}"', self.source)
        self.assertIn('sha256sum "${vae_index}"', self.source)

    def test_ffprobe_shim_is_bound_to_the_pinned_pyav_runtime(self) -> None:
        self.assertIn('expected_pyav_version="13.1.0"', self.source)
        self.assertIn('export BERNINI_EPMC_PYTHON_BIN="${python_bin}"', self.source)
        self.assertIn("import av; print(av.__version__)", self.source)
        self.assertIn("stat -c '%a'", self.source)
        self.assertIn(
            '[[ "${BERNINI_EPMC_PYTHON_BIN}" == "${python_bin}" ]]',
            self.source,
        )

    def test_checkpoint_content_is_verified_from_the_pinned_manifest(self) -> None:
        self.assertIn(
            'checkpoint_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"',
            self.source,
        )
        self.assertIn("checkpoint_manifest_file_count=23", self.source)
        self.assertIn("audits/bernini_r13_ff4c5d4_checkpoint.sha256", self.source)
        self.assertIn('sha256sum --strict --status -c "${manifest}"', self.source)
        self.assertIn('validate_checkpoint_content "${checkpoint_manifest}"', self.source)

    def test_hash_bound_release_tree_does_not_require_a_git_directory(self) -> None:
        self.assertIn(
            '[[ -d "${bernini_root}" && ! -L "${bernini_root}" ]]',
            self.source,
        )
        self.assertNotIn('-d "${bernini_root}/.git"', self.source)
        self.assertNotIn('git -C "${bernini_root}" rev-parse HEAD', self.source)
        self.assertIn("train_lora.validate_source_trees(", self.source)
        self.assertIn("SOURCE_TREES_OK", self.source)

    def test_runs_all_three_contract_suites_before_torchrun(self) -> None:
        names = (
            "test_generator_native_trajectory_controller.py",
            "test_train_generator_native_trajectory_controller_contract.py",
            "test_infer_generator_native_trajectory_controller_contract.py",
        )
        torchrun = self.source.index("-m torch.distributed.run")
        for name in names:
            self.assertLess(self.source.index(name), torchrun)

    def test_smoke_and_formal_are_explicit_and_never_deployable(self) -> None:
        self.assertIn('smoke) mode_args=(--engineering-smoke)', self.source)
        self.assertIn('formal) mode_args=()', self.source)
        self.assertIn('formal_postfit_prototype_rollout=', self.source)
        self.assertIn('post_refit_prototype_evaluation_rollout', self.source)
        self.assertIn('run.get("deployable") is False', self.source)
        self.assertIn('run.get("diagnostic_only") is True', self.source)

    def test_postrun_strong_audit_authenticates_all_four_artifacts(self) -> None:
        for required in (
            'run receipt self digest differs',
            'diagnostics hash differs',
            'inference.load_controller_bundle(',
            'training_run_receipt_sha256',
            'validate_controller_receipt(',
            'WORLD8 DP2xSP4 receipt differs',
            'executed source file hashes differ',
            'STRONG_AUDIT_OK',
        ):
            self.assertIn(required, self.source)

    def test_private_scratch_cleanup_is_prefix_guarded(self) -> None:
        self.assertIn('task_scratch="$(mktemp -d', self.source)
        self.assertIn('unsafe scratch path', self.source)
        self.assertIn('rm -rf -- "${task_scratch}"', self.source)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    METHOD_ROOT
    / "scripts"
    / "auh_infer_generator_native_trajectory_controller.sbatch"
)


class AUHEGNTCHeldoutInferenceLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        marker = (
            'PYTHONPATH="${method_root}" "${python_bin}" -B '
            "-m torch.distributed.run"
        )
        cls.torchrun_region = cls.source.split(marker, 1)[1].split(
            '[[ -f "${output_path}"', 1
        )[0]

    def test_bash_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_embedded_python_block_is_valid_ast(self) -> None:
        self.assertGreaterEqual(len(self.python_blocks), 5)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_slurm_contract_is_one_node_four_mi210_and_ulysses_four(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:4",
        ):
            self.assertIn(directive, self.source)
        self.assertIn("--nproc_per_node=4", self.torchrun_region)
        self.assertIn("world=4 Ulysses=4 GPUs=4xMI210", self.source)
        self.assertIn(
            'export BERNINI_EGNTC_RANK_CACHE_ROOT="${task_scratch}/rank-caches"',
            self.source,
        )
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_heldout_source_instruction_and_controller_are_required_and_hash_bound(self) -> None:
        required = (
            "BERNINI_EGNTC_HELDOUT_SOURCE_VIDEO",
            "BERNINI_EGNTC_HELDOUT_SOURCE_SHA256",
            "BERNINI_EGNTC_HELDOUT_INSTRUCTION",
            "BERNINI_EGNTC_HELDOUT_INSTRUCTION_SHA256",
            "BERNINI_EGNTC_CONTROLLER_STATE",
            "BERNINI_EGNTC_CONTROLLER_RECEIPT",
            "BERNINI_EGNTC_CONTROLLER_STATE_SHA256",
            "BERNINI_EGNTC_CONTROLLER_RECEIPT_SHA256",
        )
        for name in required:
            with self.subTest(name=name):
                self.assertIn(f"${{{name}:?set {name}}}", self.source)
        for fragment in (
            "held-out source video hash differs",
            "held-out instruction hash differs",
            "controller state hash differs",
            "controller receipt hash differs",
            "staged source hash differs",
            "staged controller state hash differs",
            "staged controller receipt hash differs",
            "held-out source video changed",
            "controller state changed",
            "controller receipt changed",
        ):
            self.assertIn(fragment, self.source)
        self.assertIn(
            '--expected-source-sha256 "${source_video_sha256}"',
            self.torchrun_region,
        )
        self.assertIn(
            '--expected-instruction-sha256 "${instruction_sha256}"',
            self.torchrun_region,
        )
        self.assertIn(
            '--expected-controller-state-sha256 "${controller_state_sha256}"',
            self.torchrun_region,
        )
        self.assertIn(
            '--expected-controller-receipt-sha256 "${controller_receipt_sha256}"',
            self.torchrun_region,
        )

    def test_torchrun_has_only_source_instruction_controller_semantic_inputs(self) -> None:
        for fragment in (
            '"${method_root}/infer_generator_native_trajectory_controller.py"',
            '--source-video "${staged_source_video}"',
            '--instruction "${instruction}"',
            '--controller-state "${staged_controller_state}"',
            '--controller-receipt "${staged_controller_receipt}"',
        ):
            self.assertIn(fragment, self.torchrun_region)
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
                self.assertNotIn(forbidden, self.torchrun_region)
        self.assertNotRegex(self.source, r"\$\{BERNINI_EGNTC_(?:TARGET|SUPPORT|MASK)")
        self.assertIn(
            "semantic_inputs=source_video,edit_instruction,controller "
            "target=false support=false mask=false flow=false pose=false "
            "track=false trajectory=false reference=false",
            self.source,
        )

    def test_current_v1_override_is_explicit_and_evaluation_only(self) -> None:
        self.assertEqual(self.source.count("--allow-diagnostic-no-go"), 1)
        self.assertIn("--allow-diagnostic-no-go", self.torchrun_region)
        self.assertIn(
            'bundle.representability_gate not in ("GO", "NO_GO")', self.source
        )
        self.assertIn(
            "GO and\n# NO_GO are both accepted here; the scalar diagnostic does not veto decoding",
            self.source,
        )
        self.assertIn(
            "held-out evaluation requires a formal controller artifact", self.source
        )
        self.assertIn(
            "evaluation_only=true diagnostic_override=true deployment=false "
            "scientific_claim=false production_claim=false",
            self.source,
        )
        self.assertIn('"evaluation_only": True', self.source)
        self.assertIn('receipt.get("diagnostic_only") is True', self.source)
        self.assertIn('receipt.get("deployable_output") is False', self.source)
        self.assertIn(
            'receipt.get("scientific_claim_authorized") is False', self.source
        )
        self.assertIn(
            'receipt.get("production_claim_forbidden") is True', self.source
        )

    def test_durable_archive_hash_and_embedded_revision_are_safely_extracted(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_EGNTC_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_EGNTC_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_EGNTC_SOURCE_REVISION:',
            'git get-tar-commit-id <"${source_archive}"',
            'git get-tar-commit-id <"${archive_copy}"',
            "staged method archive hash differs",
            "staged method archive revision differs",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'member.issym() or member.islnk() or member.isfifo() or member.isdev()',
            "archive member escaped method subtree",
            'find "${method_root}" -type f -exec chmod a-w',
            'method_tree_digest_pre="$(method_tree_digest "${method_root}")"',
            'method_tree_digest_post="$(method_tree_digest "${method_root}")"',
            "extracted method tree changed during inference",
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("BERNINI_EGNTC_SOURCE_REPOSITORY", self.source)
        self.assertNotIn("git -C", self.source)

    def test_archive_closure_contains_runtime_and_relevant_tests(self) -> None:
        required = {
            "generator_native_trajectory_controller.py",
            "infer_generator_native_trajectory_controller.py",
            "infer_lora.py",
            "inference_sigma_strata.py",
            "motion_residual.py",
            "train_lora.py",
            "tri_branch_unipc.py",
            "tools/ffprobe_pyav_compat.py",
            "tools/materialize_vae.py",
            "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
            "scripts/auh_infer_generator_native_trajectory_controller.sbatch",
            "tests/test_generator_native_trajectory_controller.py",
            "tests/test_tri_branch_unipc.py",
            "tests/test_infer_generator_native_trajectory_controller_contract.py",
            "tests/test_auh_infer_generator_native_trajectory_controller_launcher.py",
            "tests/test_ffprobe_pyav_compat.py",
        }
        for path in required:
            with self.subTest(path=path):
                self.assertRegex(self.source, rf"(?m)^  {re.escape(path)}(?: \\)?$")
        self.assertIn(
            '[[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]]',
            self.source,
        )

    def test_bernini_veomni_and_checkpoint_are_strictly_pinned(self) -> None:
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
        self.assertIn("sha256sum --strict --status -c", self.source)
        self.assertIn("train_lora.validate_source_trees", self.source)
        self.assertIn('receipt.get("bernini_commit") == args.bernini_commit', self.source)
        self.assertIn('receipt.get("veomni_commit") == args.veomni_commit', self.source)

    def test_private_pyav_ffprobe_shim_is_pinned_and_used_for_real_media(self) -> None:
        for fragment in (
            'ffprobe_helper_sha256="c2739db02e1e3abedebd5b7d84dc253432884dc797d4e16102df8362f158c23f"',
            'expected_pyav_version="13.1.0"',
            'export BERNINI_EPMC_PYTHON_BIN="${python_bin}"',
            'ffprobe_bin="${task_scratch}/bin/ffprobe"',
            'chmod 0555 -- "${ffprobe_bin}"',
            '[[ "$(command -v ffprobe)" == "${ffprobe_bin}" ]]',
            "held-out source is not exact 81 frames at 25 fps",
            "probe_video(ffprobe, output)",
            "backend=pyav_ffprobe_shim",
        ):
            self.assertIn(fragment, self.source)
        self.assertGreaterEqual(self.source.count('"-count_frames"'), 2)
        self.assertGreaterEqual(
            self.source.count('sha256sum "${ffprobe_bin}"'), 2
        )

    def test_exact_81_frame_40_step_runtime_is_fixed(self) -> None:
        for fragment in (
            "expected_frames=81",
            "expected_fps=25",
            "expected_steps=40",
            '--num-inference-steps "${expected_steps}"',
            'sampling.get("num_frames") == 81',
            'sampling.get("num_inference_steps") == 40',
            'sampling.get("ulysses_size") == 4',
            'certificate.get("step_count") == 40',
            'certificate.get("original_unipc_calls") == 40',
            'certificate.get("transformer_forwards") == 120',
            'sum(step.get("transformer_forwards", -1) for step in tri_steps) == 120',
            'output_contract.get("frame_count") == media["frame_count"]',
            'output_contract.get("fps") == float(media["fps"])',
        ):
            self.assertIn(fragment, self.source)

    def test_output_uses_preflight_nonoverwrite_and_exact_mp4_receipt_closure(self) -> None:
        for fragment in (
            "preflight non-overwrite isolation, not an atomic create-only transaction",
            "publication_policy=preflight_nonoverwrite_not_atomic_create_only",
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse inference output directory"',
            '[[ ! -e "${output_path}" && ! -L "${output_path}" ]] || fail "refusing to overwrite output video"',
            '[[ ! -e "${output_receipt}" && ! -L "${output_receipt}" ]] || fail "refusing to overwrite output receipt"',
            'output_path="${output_dir}/prototype.mp4"',
            'output_receipt="${output_path}.receipt.json"',
            'mkdir -- "${output_dir}"',
            '--output "${output_path}"',
            "output directory must contain exactly mp4 plus receipt",
            '{"prototype.mp4", "prototype.mp4.receipt.json"}',
        ):
            self.assertIn(fragment, self.source)

    def test_strong_audit_authenticates_receipt_media_inputs_and_trace(self) -> None:
        for fragment in (
            'stored_receipt_digest == hashlib.sha256(canonical_bytes(unsigned_receipt)).hexdigest()',
            'receipt.get("schema_version") == "bernini-egntc-source-instruction-inference-v1"',
            'receipt.get("method_files_sha256") == expected_method_files',
            'controller.get("state_file_sha256") == args.controller_state_sha256',
            'controller.get("receipt_file_sha256") == args.controller_receipt_sha256',
            'controller.get("diagnostic_override") is True',
            'input_contract.get("source_video_sha256") == args.source_sha256',
            'input_contract.get("instruction_utf8_sha256") == args.instruction_sha256',
            'input_contract.get("accepted_external_conditions") == ["source_video", "edit_instruction"]',
            'stored_trace_digest == hashlib.sha256(canonical_bytes(unsigned_trace)).hexdigest()',
            'output_contract.get("sha256") == file_sha256(output)',
            '"event": "EGNTC_HELDOUT_STRONG_AUDIT_OK"',
            '"source_instruction_controller_only": True',
        ):
            self.assertIn(fragment, self.source)

    def test_relevant_core_infer_shim_and_launcher_tests_run_before_torchrun(self) -> None:
        start = self.source.index("for pattern in \\\n")
        end = self.source.index(
            'PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run',
            start,
        )
        tests = self.source[start:end]
        for name in (
            "test_generator_native_trajectory_controller.py",
            "test_tri_branch_unipc.py",
            "test_infer_generator_native_trajectory_controller_contract.py",
            "test_ffprobe_pyav_compat.py",
            "test_auh_infer_generator_native_trajectory_controller_launcher.py",
        ):
            self.assertIn(name, tests)

    def test_scratch_cleanup_is_prefix_guarded_and_does_not_touch_output(self) -> None:
        for fragment in (
            'task_scratch="$(mktemp -d "${scratch_parent%/}/bernini-egntc-eval-${SLURM_JOB_ID}.XXXXXX")"',
            '"${scratch_parent%/}/bernini-egntc-eval-${SLURM_JOB_ID}."*)',
            "refusing unsafe scratch cleanup",
            'chmod -R u+w -- "${task_scratch}"',
            'rm -rf -- "${task_scratch}"',
            '[[ ! -e "${task_scratch}" && ! -L "${task_scratch}" ]]',
            "trap cleanup EXIT",
            "trap 'exit 143' TERM",
            "trap 'exit 130' INT",
        ):
            self.assertIn(fragment, self.source)
        cleanup_region = self.source.split("cleanup() {", 1)[1].split(
            "trap cleanup EXIT", 1
        )[0]
        self.assertNotIn('rm -rf -- "${output_dir}"', cleanup_region)

    def test_launcher_neither_submits_nor_mutates_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(self.source, r"(?m)^\s*git\s+(?:add|commit|push|reset|clean)\b")


if __name__ == "__main__":
    unittest.main()

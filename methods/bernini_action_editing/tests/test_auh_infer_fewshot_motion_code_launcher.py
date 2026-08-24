from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_fewshot_motion_code.sbatch"


class AUHFewShotMotionCodeLauncherTests(unittest.TestCase):
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

    def test_slurm_contract_is_one_node_four_mi210(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:4",
            "#SBATCH --time=24:00:00",
        ):
            self.assertIn(directive, self.source)
        self.assertIn("--nproc_per_node=4", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_prototype_artifacts_and_hashes_are_required_env(self) -> None:
        for name in (
            "BERNINI_EPMC_PROTOTYPE_STATE",
            "BERNINI_EPMC_PROTOTYPE_RECEIPT",
            "BERNINI_EPMC_PROTOTYPE_STATE_SHA256",
            "BERNINI_EPMC_PROTOTYPE_RECEIPT_SHA256",
        ):
            self.assertIn(f"${{{name}:?set {name}}}", self.source)
        self.assertIn('[[ "${prototype_state}" == *.safetensors ]]', self.source)
        self.assertIn("prototype state hash mismatch", self.source)
        self.assertIn("prototype receipt hash mismatch", self.source)
        self.assertIn("staged prototype state hash differs", self.source)
        self.assertIn("staged prototype receipt hash differs", self.source)
        self.assertIn("prototype state changed", self.source)
        self.assertIn("prototype receipt changed", self.source)

    def test_no_go_environment_maps_to_one_exact_cli_flag(self) -> None:
        self.assertIn(
            'allow_no_go_diagnostic="${BERNINI_EPMC_ALLOW_NO_GO_DIAGNOSTIC:-0}"',
            self.source,
        )
        self.assertIn(
            '[[ "${allow_no_go_diagnostic}" == 0 || "${allow_no_go_diagnostic}" == 1 ]]',
            self.source,
        )
        self.assertIn("diagnostic_args=()", self.source)
        self.assertIn("1) diagnostic_args+=(--allow-no-go-diagnostic)", self.source)
        self.assertIn('"${diagnostic_args[@]}"', self.source)
        self.assertIn(
            'expected_gate = "NO_GO" if args.allow_no_go_diagnostic == "1" else "GO"',
            self.source,
        )

    def test_heldout_source_and_instruction_are_hash_frozen(self) -> None:
        self.assertIn(
            'expected_source_sha256="4d0c5cdfa9e0aae394af34a5bdda7de82ac770cd62cddbf3173ad2378458f3ed"',
            self.source,
        )
        self.assertIn(
            'expected_instruction_sha256="105ee8052a0f65d700736a8a25fdf02eb56f1b60d581403c328a8db3d500558c"',
            self.source,
        )
        self.assertIn("held-out source hash mismatch", self.source)
        self.assertIn("held-out instruction hash mismatch", self.source)
        self.assertIn('--expected-source-sha256 "${expected_source_sha256}"', self.source)
        self.assertIn(
            '--expected-instruction-sha256 "${expected_instruction_sha256}"',
            self.source,
        )
        self.assertIn(
            "target=false support=false mask=false flow=false pose=false track=false trajectory=false reference=false heldout_oracle=false",
            self.source,
        )

    def test_commit_archive_is_byte_bound_and_safely_extracted(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_ACTION_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_ACTION_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_ACTION_SOURCE_REVISION:',
            'source_repository="${BERNINI_ACTION_SOURCE_REPOSITORY:',
            'git -C "${source_repository}" archive --format=tar',
            '"${source_revision}" methods/bernini_action_editing',
            '[[ "$(sha256sum "${archive_from_revision}"',
            "method archive bytes are not the declared commit",
            "member.issym() or member.islnk() or member.isfifo() or member.isdev()",
            "archive member escaped method subtree",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'method_root="${task_scratch}/source/methods/bernini_action_editing"',
            'find "${method_root}" -type f -exec chmod a-w',
            'method_tree_digest_pre="$(method_tree_digest "${method_root}")"',
            'method_tree_digest_post="$(method_tree_digest "${method_root}")"',
            "extracted method tree changed during inference",
        ):
            self.assertIn(fragment, self.source)
        self.assertIn('"${method_root}/infer_fewshot_motion_code.py"', self.source)
        self.assertNotIn(
            '/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/methods/bernini_action_editing/infer_fewshot_motion_code.py',
            self.source,
        )

    def test_required_archive_files_cover_complete_import_closure(self) -> None:
        required = {
            "infer_fewshot_motion_code.py",
            "counterfactual_proposal_motion_runtime.py",
            "counterfactual_proposal_motion_branch.py",
            "counterfactual_proposal_motion_rebinding.py",
            "fewshot_episode_io.py",
            "fewshot_motion_branch.py",
            "fewshot_privileged_motion_code.py",
            "fewshot_proposal_motion_carrier.py",
            "infer_counterfactual_proposal_motion_oracle.py",
            "infer_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "source_kv_route_batches.py",
            "source_kv_replay.py",
            "source_value_residual.py",
            "train_lora.py",
            "tools/materialize_vae.py",
            "scripts/auh_infer_fewshot_motion_code.sbatch",
            "tests/test_infer_fewshot_motion_code_contract.py",
            "tests/test_auh_infer_fewshot_motion_code_launcher.py",
        }
        for path in required:
            with self.subTest(path=path):
                self.assertRegex(
                    self.source,
                    rf"(?m)^  {re.escape(path)}(?: \\)?$",
                )
        self.assertIn(
            '[[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]]',
            self.source,
        )

    def test_checkpoint_and_upstream_source_are_strictly_pinned(self) -> None:
        for value in (
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "08a331958560544efb5e842666c236d819bfdc36d34b6eb9c1cdcee1546ae670",
        ):
            self.assertIn(value, self.source)
        self.assertIn("checkpoint_content_file_count=23", self.source)
        self.assertGreaterEqual(self.source.count('validate_checkpoint_content "${checkpoint_manifest}"'), 2)
        self.assertIn("sha256sum --strict --status -c", self.source)
        self.assertIn("VeOmni tracked source is dirty", self.source)

    def test_torchrun_executes_fixed_40_step_five_arm_runner(self) -> None:
        for fragment in (
            '"${python_bin}" -B -m torch.distributed.run',
            '--nproc_per_node=4',
            '"${method_root}/infer_fewshot_motion_code.py"',
            '--num-inference-steps "${expected_steps}"',
            '--proposal-seed "${proposal_seed}"',
            '--render-seed "${render_seed}"',
            'expected_frames=81',
            'expected_fps=25',
            'expected_steps=40',
            'proposal_seed=2027',
            'render_seed=2028',
            'arms=B0,Z0,PROTO,REVERSE,SHUFFLE videos=7',
        ):
            self.assertIn(fragment, self.source)
        command_region = self.source.split(
            'PYTHONPATH="${method_root}" "${python_bin}" -B -m torch.distributed.run', 1
        )[1].split('[[ -d "${output_dir}"', 1)[0]
        for forbidden in (
            "--target",
            "--support",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--reference",
        ):
            self.assertNotIn(forbidden, command_region)

    def test_output_directory_is_never_reused_and_has_exact_seven_videos(self) -> None:
        self.assertIn(
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse output directory"',
            self.source,
        )
        self.assertIn('canonical_output_dir="$(realpath -m -- "${output_dir}")"', self.source)
        self.assertIn("output directory must be canonical and contain no symlink traversal", self.source)
        self.assertIn('--output-dir "${output_dir}"', self.source)
        self.assertIn(
            "for name in proposal_action proposal_noop B0 Z0 PROTO REVERSE SHUFFLE",
            self.source,
        )
        self.assertIn(
            "output directory must contain exactly seven videos and one receipt",
            self.source,
        )
        self.assertIn(
            'arms.get("order") == ["B0", "Z0", "PROTO", "REVERSE", "SHUFFLE"]',
            self.source,
        )
        self.assertIn(
            'set(runtime) == {"Z0", "PROTO", "REVERSE", "SHUFFLE"}',
            self.source,
        )
        self.assertIn("all(item.get(\"completed_steps\") == 40", self.source)
        self.assertIn("every_output_is_81_frames_25fps", self.source)

    def test_receipt_audit_binds_prototype_parity_outputs_and_diagnostic_label(self) -> None:
        for fragment in (
            'receipt.get("schema_version") == "bernini-epmc-v12-source-instruction-inference-v1"',
            'receipt.get("scientific_claim") is False',
            'receipt.get("source_instruction_only_inference") is True',
            'receipt.get("heldout_oracle_arm_exists") is False',
            'receipt.get("heldout_oracle_used") is False',
            'receipt.get("diagnostic_only")',
            'prototype.get("state_file_sha256")',
            'prototype.get("receipt_file_sha256")',
            'prototype.get("all_12_heads_byte_exact_tied_per_block") is True',
            'claims.get("z0_full_latent_byte_exact_b0") is True',
            'row.get("frames") == 81',
            'row.get("fps") == 25',
            'row.get("bucket_hw") == [480, 496]',
            "ensure_ascii=True",
            'stored != hashlib.sha256(payload).hexdigest()',
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_only_validates_and_runs_but_does_not_submit_or_commit(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*git\s+(commit|push|add)\b")
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch\b")
        self.assertIn(
            "test_infer_fewshot_motion_code_contract.py", self.source
        )
        self.assertIn(
            "test_auh_infer_fewshot_motion_code_launcher.py", self.source
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_infer_full644_directed_dual4.sbatch"


class AUHFull644DirectedDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        cls.launch_region = cls.source.split("launch_arm() (", 1)[1].split(
            "late_status=0", 1
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
        self.assertGreaterEqual(len(self.python_blocks), 5)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_slurm_contract_is_one_node_eight_mi210_dual_four(self) -> None:
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

    def test_two_arms_are_concurrent_and_failure_is_joined(self) -> None:
        for fragment in (
            'late_visible_gpus="0,1,2,3"',
            'all_visible_gpus="4,5,6,7"',
            'export ROCR_VISIBLE_DEVICES="${visible_gpus}"',
            'launch_arm late late "${late_visible_gpus}"',
            'launch_arm all all "${all_visible_gpus}"',
            'late_pid=$!',
            'all_pid=$!',
            'wait "${late_pid}" || late_status=$?',
            'wait "${all_pid}" || all_status=$?',
            '[[ "${late_status}" -ne 0 || "${all_status}" -ne 0 ]]',
            'fail "one or more directed-attention arms failed"',
        ):
            self.assertIn(fragment, self.source)
        self.assertRegex(
            self.source,
            r"launch_arm late late .* >\"\$\{late_log\}\" 2>&1 &\nlate_pid=\$!",
        )
        self.assertRegex(
            self.source,
            r"launch_arm all all .* >\"\$\{all_log\}\" 2>&1 &\nall_pid=\$!",
        )

    def test_arms_have_distinct_rendezvous_and_cache_roots(self) -> None:
        for fragment in (
            'late_master_port=$((20000 + 2 * job_mod))',
            'all_master_port=$((late_master_port + 1))',
            '[[ "${late_master_port}" -ne "${all_master_port}" ]]',
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

    def test_exact_7b88_inputs_adapter_and_runtime_are_frozen(self) -> None:
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
        ):
            self.assertIn(fragment, self.source)
        for label in (
            "7b88 source video hash differs",
            "7b88 instruction hash differs",
            "full644 adapter config hash differs",
            "full644 adapter model hash differs",
            "full644 adapter receipt hash differs",
        ):
            self.assertIn(label, self.source)

    def test_checkpoint_and_upstream_are_content_pinned(self) -> None:
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
        self.assertGreaterEqual(self.source.count("train_lora.validate_source_trees"), 2)
        self.assertIn("VeOmni tracked source is dirty", self.source)

    def test_method_archive_is_self_authenticating_without_source_repository(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_DUAL4_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_DUAL4_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_DUAL4_SOURCE_REVISION:',
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
        self.assertNotIn("BERNINI_DUAL4_SOURCE_REPOSITORY", self.source)
        self.assertNotIn("method-from-revision.tar", self.source)
        self.assertNotIn('git -C "${source_repository}"', self.source)

    def test_torchrun_semantic_inputs_are_source_and_instruction_only(self) -> None:
        for fragment in (
            '"${method_root}/infer_directed_attn_oracle.py"',
            '--directed-attn-blocks "${block_scope}"',
            '--adapter-checkpoint "${arm_root}/input/adapter"',
            '--source-video "${arm_root}/input/source/source.mp4"',
            '--instruction "${instruction}"',
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

    def test_outputs_and_receipts_are_distinct_and_nonoverwriting(self) -> None:
        for fragment in (
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]] || fail "refusing to reuse dual4 output directory"',
            'late_output="${output_dir}/late.mp4"',
            'all_output="${output_dir}/all.mp4"',
            'late_receipt="${late_output}.receipt.json"',
            'all_receipt="${all_output}.receipt.json"',
            '[[ ! -e "${candidate}" && ! -L "${candidate}" ]]',
            "preflight_nonoverwrite_not_atomic_create_only",
            "dual4 output directory must contain two mp4, two receipts, and two logs",
        ):
            self.assertIn(fragment, self.source)
        self.assertIn("FULL644_DIRECTED_DUAL4_STRONG_AUDIT_OK", self.source)
        self.assertIn('attention.get("block_selection") == scope', self.source)
        self.assertIn('sampling.get("num_frames") == 81', self.source)
        self.assertIn('sampling.get("num_inference_steps") == 40', self.source)
        self.assertIn('sampling.get("seed") == 2027', self.source)
        self.assertIn('sampling.get("ulysses_size") == 4', self.source)

    def test_archive_closure_contains_launcher_and_contract_test(self) -> None:
        for required in (
            "directed_source_attention.py",
            "infer_directed_attn_oracle.py",
            "infer_lora.py",
            "train_lora.py",
            "tools/materialize_vae.py",
            "audits/bernini_r13_ff4c5d4_checkpoint.sha256",
            "scripts/auh_infer_full644_directed_dual4.sbatch",
            "tests/test_directed_source_attention.py",
            "tests/test_infer_directed_attn_oracle.py",
            "tests/test_infer_lora_contract.py",
            "tests/test_auh_infer_full644_directed_dual4_launcher.py",
        ):
            with self.subTest(required=required):
                self.assertRegex(
                    self.source,
                    rf"(?m)^  {re.escape(required)}(?: \\)?$",
                )
        self.assertIn(
            '[[ -f "${method_root}/${required}" && ! -L "${method_root}/${required}" ]]',
            self.source,
        )

    def test_launcher_does_not_submit_or_mutate_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive)\b",
        )


if __name__ == "__main__":
    unittest.main()

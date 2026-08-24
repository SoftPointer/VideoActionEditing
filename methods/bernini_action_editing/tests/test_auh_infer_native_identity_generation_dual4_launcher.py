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
    / "auh_infer_native_identity_generation_dual4.sbatch"
)


class AUHNativeIdentityGenerationDual4LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        cls.launch_region = cls.source.split("launch_group() (", 1)[1].split(
            "t2v_r2v_status=0", 1
        )[0]

    def test_bash_syntax_and_embedded_python_are_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(len(self.python_blocks), 3)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_one_node_eight_mi210s_are_two_legal_ulysses_four_groups(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=32",
            "#SBATCH --mem=256G",
            "#SBATCH --gres=gpu:mi210:8",
        ):
            self.assertIn(directive, self.source)
        self.assertIn('t2v_r2v_visible_gpus="0,1,2,3"', self.source)
        self.assertIn('rv2v_visible_gpus="4,5,6,7"', self.source)
        self.assertEqual(self.launch_region.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn("topology=dual-4-Ulysses", self.source)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")

    def test_groups_partition_three_arms_and_launch_concurrently(self) -> None:
        for fragment in (
            'launch_group t2v-r2v "${t2v_r2v_visible_gpus}" "${t2v_r2v_master_port}" "${t2v_r2v_output}" t2v r2v',
            'launch_group rv2v "${rv2v_visible_gpus}" "${rv2v_master_port}" "${rv2v_output}" rv2v',
            "t2v_r2v_pid=$!",
            "rv2v_pid=$!",
            'wait "${t2v_r2v_pid}" || t2v_r2v_status=$?',
            'wait "${rv2v_pid}" || rv2v_status=$?',
            'fail "one or more native canary groups failed"',
        ):
            self.assertIn(fragment, self.source)
        self.assertRegex(
            self.source,
            r"launch_group t2v-r2v .* >\"\$\{t2v_r2v_log\}\" 2>&1 &\nt2v_r2v_pid=\$!",
        )
        self.assertRegex(
            self.source,
            r"launch_group rv2v .* >\"\$\{rv2v_log\}\" 2>&1 &\nrv2v_pid=\$!",
        )

    def test_rendezvous_and_runtime_caches_are_group_isolated(self) -> None:
        for fragment in (
            'rv2v_master_port=$((t2v_r2v_master_port + 1))',
            '[[ "${t2v_r2v_master_port}" -ne "${rv2v_master_port}" ]]',
            '--master_port="${master_port}"',
            'local group_root="${task_scratch}/groups/${group}"',
            'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL',
            'export ROCR_VISIBLE_DEVICES="${visible_gpus}"',
            'export MIOPEN_USER_DB_PATH="${group_root}/cache/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${group_root}/cache/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${group_root}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${group_root}/cache/triton"',
            'export XDG_CACHE_HOME="${group_root}/cache/xdg"',
            'export PYTHONPYCACHEPREFIX="${group_root}/cache/pycache"',
            'export TORCHELASTIC_ERROR_FILE="${group_root}/torch-elastic-error.json"',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn('export HIP_VISIBLE_DEVICES="${visible_gpus}"', self.source)
        self.assertNotIn('export CUDA_VISIBLE_DEVICES="${visible_gpus}"', self.source)

    def test_runner_receives_exact81_gaussian_comparison_inputs_only(self) -> None:
        for fragment in (
            '"${method_root}/infer_native_identity_generation_canary.py"',
            '--source-video "${source_video}"',
            '--expected-source-sha256 "${expected_source_sha256}"',
            '--action-prompt "${action_prompt}"',
            '--expected-action-prompt-sha256 "${expected_action_prompt_sha256}"',
            '--arms "${arms[@]}"',
            '--num-inference-steps "${expected_steps}"',
            '--seed "${expected_seed}"',
            "expected_frames=81",
            "expected_fps=25",
            "expected_steps=40",
            "expected_seed=2027",
        ):
            self.assertIn(fragment, self.source)
        for forbidden in (
            "--target",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--reference",
            "--first-frame",
            "--initial-latent",
            "--initial-noise",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.launch_region)
        self.assertIn(
            "semantic_inputs=source_video,action_prompt target=false "
            "external_reference=false mask=false flow=false pose=false "
            "track=false trajectory=false first_frame=false",
            self.source,
        )

    def test_method_archive_and_checkpoint_are_content_bound(self) -> None:
        for fragment in (
            'source_archive="${BERNINI_NATIVE_SOURCE_ARCHIVE:',
            'source_archive_sha256="${BERNINI_NATIVE_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${BERNINI_NATIVE_SOURCE_REVISION:',
            'checkpoint_manifest="${BERNINI_CHECKPOINT_CONTENT_MANIFEST:',
            'sha256sum "${source_archive}"',
            'sha256sum "${archive_copy}"',
            'git get-tar-commit-id <"${source_archive}"',
            'git get-tar-commit-id <"${archive_copy}"',
            '[[ "${actual_archive_revision}" == "${source_revision}" ]]',
            'member.issym() or member.islnk() or member.isfifo() or member.isdev()',
            "archive member escaped method subtree",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'find "${method_root}" -type f -exec chmod a-w',
            '--checkpoint-content-manifest "${checkpoint_manifest}"',
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
        ):
            self.assertIn(fragment, self.source)
        for required in (
            "infer_native_identity_generation_canary.py",
            "infer_lora.py",
            "train_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
            "source_value_residual.py",
            "tools/materialize_vae.py",
        ):
            self.assertGreaterEqual(self.source.count(required), 2)

    def test_post_run_audit_requires_all_three_native_outputs(self) -> None:
        for fragment in (
            "BERNINI_NATIVE_IDENTITY_DUAL4_STRONG_AUDIT_OK",
            'left.get("arms") != ["t2v", "r2v"]',
            'right.get("arms") != ["rv2v"]',
            '"t2v": {"mode": "t2v_apg", "refs": [], "videos": 0}',
            '"r2v": {"mode": "r2v_apg", "refs": [0, 20, 40, 60, 80], "videos": 0}',
            '"rv2v": {"mode": "rv2v", "refs": [0, 27, 53, 80], "videos": 1}',
            'sampling.get("target_initialization") != "official_gen_wanx22_fresh_gaussian"',
            'sampling.get("target_mixed_with_source_latent") is not False',
            'condition.get("reference_from_temporal_video_latent_slice") is not False',
            'set(outputs) != {"t2v", "r2v", "rv2v"}',
            'output.get("frame_count") != 81 or output.get("fps") != 25',
            'output.get("normalized_clean_latent")',
            'clean.get("stored_dtype") != "torch.float32"',
            'clean.get("native_sampler_before_vae_decode") is not True',
            'clean.get("mp4_decode_reencode_used") is not False',
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_does_not_submit_or_mutate_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive)\b",
        )


if __name__ == "__main__":
    unittest.main()

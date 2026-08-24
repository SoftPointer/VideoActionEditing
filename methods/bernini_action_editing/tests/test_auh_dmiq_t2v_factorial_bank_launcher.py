from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_dmiq_t2v_factorial_bank_dual4.sbatch"


class AUHDMIQT2VFactorialBankLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL)
        cls.launch_region = cls.source.split("launch_group() (", 1)[1].split(
            'sp4_a_log="', 1
        )[0]

    def test_bash_and_embedded_python_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(len(self.python_blocks), 1)
        for block in self.python_blocks:
            ast.parse(block)

    def test_one_eight_gpu_node_is_split_into_two_isolated_sp4_groups(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:mi210:8",
        ):
            self.assertIn(directive, self.source)
        self.assertIn('sp4_a_visible_gpus="0,1,2,3"', self.source)
        self.assertIn('sp4_b_visible_gpus="4,5,6,7"', self.source)
        self.assertEqual(self.launch_region.count("--nproc_per_node=4"), 1)
        self.assertNotIn("--nproc_per_node=8", self.source)
        self.assertIn("topology=two-isolated-SP4", self.source)
        self.assertIn('unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL', self.source)
        self.assertIn('export ROCR_VISIBLE_DEVICES="${visible_gpus}"', self.source)

    def test_groups_run_concurrently_with_separate_ports_caches_and_logs(self) -> None:
        for fragment in (
            'sp4_b_master_port=$((sp4_a_master_port + 1))',
            'local group_root="${task_scratch}/groups/${group}"',
            'export MIOPEN_USER_DB_PATH="${group_root}/cache/miopen-user"',
            'export TORCH_EXTENSIONS_DIR="${group_root}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${group_root}/cache/triton"',
            'launch_group sp4-a "${sp4_a_visible_gpus}" "${sp4_a_master_port}"',
            'launch_group sp4-b "${sp4_b_visible_gpus}" "${sp4_b_master_port}"',
            "sp4_a_pid=$!",
            "sp4_b_pid=$!",
            'wait "${sp4_a_pid}" || sp4_a_status=$?',
            'wait "${sp4_b_pid}" || sp4_b_status=$?',
        ):
            self.assertIn(fragment, self.source)

    def test_every_cell_uses_existing_native_t2v_renderer_via_manifest_wrapper(self) -> None:
        for fragment in (
            "dmiq_t2v_factorial_bank.py",
            "infer_native_identity_generation_canary.py",
            "list-entry-ids",
            "render-entry",
            '--entry-id "${entry_id}"',
            '--nproc_per_node=4',
            "finalize",
            "bank.receipt.json",
            "DMIQ_T2V_FACTORIAL_BANK_PROVENANCE_OK_BLIND_SPLIT_EVENT_AUDIT_PENDING",
        ):
            self.assertIn(fragment, self.source)
        self.assertIn(
            "semantic_inputs=prompt,source_geometry_bucket target=false "
            "source_latent=false reference=false mask=false flow=false pose=false "
            "track=false trajectory=false first_frame=false",
            self.source,
        )
        self.assertIn("full_action plus all nine negatives on one SP4 group", self.source)
        self.assertIn("non-[1,16,21,62,60] geometry", self.source)
        self.assertIn(">=8 discovery and >=4 confirmation", self.source)
        self.assertIn("restricted to engineering_micro timing runs", self.source)
        self.assertIn("NOT scale-ready", self.source)
        self.assertIn(
            '[[ "${manifest_profile_and_rung}" == "engineering_micro:0" ]]',
            self.source,
        )
        self.assertIn("cumulative topup reuse are not implemented", self.source)
        for forbidden in (
            "--target-video",
            "--mask",
            "--flow",
            "--pose",
            "--track",
            "--trajectory",
            "--reference-image",
            "--reference-video",
            "--initial-latent",
            "--initial-noise",
        ):
            self.assertNotIn(forbidden, self.launch_region)

    def test_manifest_and_method_archive_are_content_bound(self) -> None:
        for fragment in (
            'manifest_file_sha256="${DMIQ_T2V_BANK_MANIFEST_SHA256:',
            'source_archive_sha256="${DMIQ_T2V_SOURCE_ARCHIVE_SHA256:',
            'source_revision="${DMIQ_T2V_SOURCE_REVISION:',
            'sha256sum "${manifest}"',
            'sha256sum "${persistent_manifest}"',
            'git get-tar-commit-id <"${source_archive}"',
            'git get-tar-commit-id <"${archive_copy}"',
            "archive member escaped method subtree",
            "archive contains a link or device",
            'find "${method_root}" -type f -exec chmod a-w',
            '--expected-file-sha256 "${manifest_file_sha256}"',
            '--method-source-revision "${source_revision}"',
            '--method-source-archive-sha256 "${source_archive_sha256}"',
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_neither_submits_nor_trains_nor_mutates_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(self.source, r"(?m)^\s*git\s+(?:add|commit|push|reset|clean)\b")
        for forbidden in (
            "optimizer.step",
            "loss.backward",
            "torchrun.*train",
            "deepspeed",
        ):
            self.assertNotRegex(self.source, forbidden)


if __name__ == "__main__":
    unittest.main()

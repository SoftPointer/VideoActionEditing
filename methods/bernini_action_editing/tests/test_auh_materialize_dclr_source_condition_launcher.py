from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_materialize_dclr_source_condition.sbatch"


class AUHSourceConditionMaterializerLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = LAUNCHER.read_text(encoding="utf-8")
        cls.python_blocks = re.findall(
            r"<<'PY'\n(.*?)\nPY", cls.source, re.DOTALL
        )
        cls.run_region = cls.source.split(
            'runtime_log="${task_scratch}/runtime.log"', 1
        )[1].split("audit_outputs()", 1)[0]
        cls.audit_region = cls.source.split("audit_outputs()", 1)[1]

    def test_bash_and_embedded_python_are_static_syntax_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.python_blocks), 2)
        for index, block in enumerate(self.python_blocks):
            with self.subTest(index=index):
                ast.parse(block)

    def test_requests_exactly_one_mi210_and_is_not_distributed(self) -> None:
        for directive in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=128G",
            "#SBATCH --gres=gpu:mi210:1",
        ):
            self.assertIn(directive, self.source)
        self.assertNotIn("#SBATCH --gres=gpu:mi210:4", self.source)
        self.assertNotIn("torch.distributed", self.run_region)
        self.assertNotIn("torchrun", self.run_region)
        self.assertNotIn("--nproc_per_node", self.run_region)
        self.assertNotRegex(self.source, r"(?m)^\s*srun(?:\s|$)")
        self.assertIn("gpu_count=1", self.run_region)

    def test_launcher_input_surface_is_raw_source_only(self) -> None:
        for environment_name in (
            "DCLR_SOURCE_ONLY_IID",
            "DCLR_SOURCE_ONLY_SOURCE_VIDEO",
            "DCLR_SOURCE_ONLY_SOURCE_VIDEO_SHA256",
            "DCLR_SOURCE_ONLY_BUCKET_HEIGHT",
            "DCLR_SOURCE_ONLY_BUCKET_WIDTH",
            "DCLR_SOURCE_ONLY_CHECKPOINT",
            "DCLR_SOURCE_ONLY_CHECKPOINT_CONTENT_MANIFEST",
        ):
            self.assertIn(environment_name, self.source)
        for fragment in (
            '--iid "${iid}"',
            '--source-video "${source_video}"',
            '--expected-source-sha256 "${source_video_sha256}"',
            '--expected-bucket-hw "${bucket_height}" "${bucket_width}"',
            'source_only=true target_path=false target_columns=false target_posterior=false target_media=false',
        ):
            self.assertIn(fragment, self.run_region)
        for forbidden_environment in (
            "DCLR_SOURCE_ONLY_TARGET",
            "DCLR_SOURCE_ONLY_EDITED",
            "DCLR_SOURCE_ONLY_PARQUET",
            "DCLR_SOURCE_ONLY_POSTERIOR",
        ):
            self.assertNotIn(forbidden_environment, self.source)
        for forbidden_flag in (
            "--target",
            "--edited",
            "--parquet",
            "--posterior",
        ):
            self.assertNotIn(forbidden_flag, self.run_region)

    def test_checkpoint_and_vae_bytes_are_pinned(self) -> None:
        for value in (
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527",
        ):
            self.assertIn(value, self.source)
        for fragment in (
            'sha256sum "${checkpoint_manifest}"',
            'sha256sum "${checkpoint}/vae/config.json"',
            '--expected-checkpoint-tree-sha256 "${checkpoint_tree_sha256}"',
            '--expected-checkpoint-content-manifest-sha256 "${checkpoint_manifest_sha256}"',
            '--expected-vae-config-sha256 "${vae_config_sha256}"',
            'content.get("verified_file_count") == 23',
            'content.get("every_file_sha256_verified") is True',
        ):
            self.assertIn(fragment, self.source)

    def test_archive_revision_hash_and_import_closure_are_audited(self) -> None:
        for fragment in (
            'git get-tar-commit-id <"${source_archive}"',
            'sha256sum "${source_archive}"',
            'sha256sum "${archive_copy}"',
            'member.issym() or member.islnk() or member.isfifo() or member.isdev()',
            "archive member escaped repository-relative closure",
            'tar --no-same-owner --no-same-permissions -xf "${archive_copy}"',
            'find "${method_root}" -type f -exec chmod a-w',
        ):
            self.assertIn(fragment, self.source)
        for required in (
            "materialize_dclr_source_condition.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_lora.py",
            "train_lora.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
            "tools/materialize_vae.py",
        ):
            self.assertGreaterEqual(self.source.count(required), 2)

    def test_rocr_device_and_caches_are_isolated(self) -> None:
        for fragment in (
            'allocated_rocr_device="${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"',
            '[[ "${allocated_rocr_device}" =~ ^[0-9]+$ ]]',
            'unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL',
            'export ROCR_VISIBLE_DEVICES="${allocated_rocr_device}"',
            'export MIOPEN_USER_DB_PATH="${task_scratch}/cache/miopen-user"',
            'export MIOPEN_CUSTOM_CACHE_DIR="${task_scratch}/cache/miopen-custom"',
            'export TORCH_EXTENSIONS_DIR="${task_scratch}/cache/torch-extensions"',
            'export TRITON_CACHE_DIR="${task_scratch}/cache/triton"',
            'export XDG_CACHE_HOME="${task_scratch}/cache/xdg"',
            'export PYTHONPYCACHEPREFIX="${task_scratch}/cache/pycache"',
        ):
            self.assertIn(fragment, self.source)

    def test_audit_matches_runtime_source_only_receipt_schema(self) -> None:
        for fragment in (
            'receipt.get("schema_version") == "bernini-source-only-vae-materialization-v1"',
            'receipt.get("source_only") is True',
            'source.get("source_iid") == iid',
            'source.get("source_video_sha256") == source_sha256',
            'closed = receipt.get("access_audit")',
            '"source_columns_accessed": ["iid", "source_video", "source_video_sha256"]',
            '"target_columns_accessed": []',
            '"target_media_accessed": False',
            '"paired_target_accessed": False',
            'artifact = receipt.get("source_condition_artifact")',
            'artifact.get("artifact_role") == "source_video_condition"',
            'artifact.get("coordinate") == "bernini_normalized_clean_vae_latent"',
            'artifact.get("frame_contract") == "exact81_latent21"',
            'artifact.get("stored_dtype") == "torch.float32"',
            'artifact.get("source_video_vae_encode_before_any_decode") is True',
            'artifact.get("mp4_decode_reencode_used") is False',
        ):
            self.assertIn(fragment, self.audit_region)

    def test_output_is_fresh_atomic_and_double_audited(self) -> None:
        for fragment in (
            '[[ ! -e "${output_dir}" && ! -L "${output_dir}" ]]',
            'source.normalized-clean-latent.safetensors',
            "BERNINI_DCLR_SOURCE_ONLY_MATERIALIZATION_AUDIT_OK",
            'receipt_sha_round1="$(sha256sum',
            'artifact_sha_round1="$(sha256sum',
            'audit_outputs || fail "output audit round 1 failed"',
            'audit_outputs || fail "output audit round 2 failed"',
            "receipt hash closure differs",
            "artifact hash closure differs",
        ):
            self.assertIn(fragment, self.source)

    def test_launcher_does_not_submit_or_mutate_git(self) -> None:
        self.assertNotRegex(self.source, r"(?m)^\s*sbatch(?:\s|$)")
        self.assertNotRegex(
            self.source,
            r"(?m)^\s*git\s+(?:add|commit|push|reset|clean|archive|checkout|switch)\b",
        )


if __name__ == "__main__":
    unittest.main()

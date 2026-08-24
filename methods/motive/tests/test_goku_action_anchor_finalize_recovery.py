from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STANDALONE_FINALIZE_SBATCH = (
    ROOT / "scripts" / "auh_goku_action_anchor_finalize.sbatch"
)
RECOVERY_SUBMIT_SCRIPT = (
    ROOT
    / "scripts"
    / "auh_submit_goku_action_anchor_finalize_recovery.sh"
)


class GokuActionAnchorFinalizeRecoveryTests(unittest.TestCase):
    def test_scripts_have_valid_bash_syntax(self) -> None:
        for script in (
            STANDALONE_FINALIZE_SBATCH,
            RECOVERY_SUBMIT_SCRIPT,
        ):
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"{script.name}: {completed.stderr}",
            )

    def test_standalone_recovery_uses_one_gpu_and_only_finalizes(self) -> None:
        text = STANDALONE_FINALIZE_SBATCH.read_text(encoding="utf-8")
        for marker in (
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:mi210:1",
            "shard_count=8",
            'actual_shards=("${qwen_root}"/qwen_shard_*.jsonl)',
            "-m motive.goku_action_anchor_finalize",
            '--input "${selected}"',
            '--qwen-root "${qwen_root}"',
            '--output-dir "${final_output}"',
            'finalizer_args+=(--allow-partial)',
            'finalizer_args+=(--approval "${approval_path}")',
            "generation_manifest.jsonl",
        ):
            self.assertIn(marker, text)
        self.assertNotIn("-m motive.goku_action_anchor_qwen", text)
        self.assertNotIn("MOTIVE_GOKU_ACTION_QWEN_MODEL", text)
        self.assertNotIn("#SBATCH --gres=gpu:mi210:8", text)

    def test_recovery_submitter_records_job_and_builds_wan_helper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            scripts = snapshot / "methods" / "motive" / "scripts"
            modules = snapshot / "methods" / "motive" / "motive"
            scripts.mkdir(parents=True)
            modules.mkdir(parents=True)
            (snapshot / "SOURCE_FILES.jsonl").write_text(
                '{"fixture":true}\n',
                encoding="utf-8",
            )
            for path in (
                modules / "goku_action_anchor_qwen.py",
                modules / "goku_action_anchor_finalize.py",
                scripts / "auh_goku_action_anchor_finalize.sbatch",
                scripts / "auh_submit_wan22_i2v_chain.sh",
            ):
                path.write_text("# fixture\n", encoding="utf-8")

            selected = root / "selected.jsonl"
            selected.write_text('{"iid":"fixture"}\n', encoding="utf-8")
            approval = root / "approval.json"
            approval.write_text('{"fixture":"approval"}\n', encoding="utf-8")
            qwen_root = root / "qwen8"
            qwen_root.mkdir()
            for shard_index in range(8):
                (
                    qwen_root / f"qwen_shard_{shard_index:03d}.jsonl"
                ).write_bytes(b"")

            python_bin = root / "python"
            python_bin.write_text(
                "#!/usr/bin/env bash\nexit 0\n",
                encoding="utf-8",
            )
            python_bin.chmod(0o700)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' '12345;fixture-cluster'\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o700)

            run_root = root / "recovery"
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SBATCH_")
            }
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
                    "MOTIVE_GOKU_ACTION_SOURCE_SNAPSHOT": str(snapshot),
                    "MOTIVE_GOKU_ACTION_SELECTED": str(selected),
                    "MOTIVE_GOKU_ACTION_QWEN_OUTPUT": str(qwen_root),
                    "MOTIVE_GOKU_ACTION_RECOVERY_RUN_ROOT": str(run_root),
                    "MOTIVE_GOKU_ACTION_PYTHON_BIN": str(python_bin),
                    "MOTIVE_GOKU_ACTION_APPROVAL_PATH": str(approval),
                    "MOTIVE_GOKU_ACTION_FINAL_SEED": "260730",
                    "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL": "1",
                }
            )
            completed = subprocess.run(
                ["bash", str(RECOVERY_SUBMIT_SCRIPT)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "standalone_finalize_job_id=12345",
                completed.stdout,
            )
            self.assertEqual(
                (run_root / "finalize_submission.raw").read_text(
                    encoding="utf-8"
                ),
                "12345;fixture-cluster\n",
            )
            jobs = (run_root / "jobs.tsv").read_text(encoding="utf-8")
            self.assertIn(
                f"standalone_finalize\t12345\tnone\t{run_root}/final",
                jobs,
            )
            retry = run_root / "retry_finalize.sh"
            wan_helper = run_root / "submit_wan_chain.sh"
            self.assertTrue(os.access(retry, os.X_OK))
            self.assertTrue(os.access(wan_helper, os.X_OK))
            helper_text = wan_helper.read_text(encoding="utf-8")
            self.assertIn(
                "MOTIVE_GOKU_ACTION_ALLOW_PARTIAL=1",
                retry.read_text(encoding="utf-8"),
            )
            self.assertIn(
                f"MOTIVE_GOKU_ACTION_APPROVAL_PATH={approval}",
                retry.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "export MOTIVE_GOKU_ACTION_CURATION_JOB_ID=12345",
                helper_text,
            )
            self.assertIn(
                (
                    "export MOTIVE_GOKU_ACTION_GENERATION_MANIFEST="
                    f"{run_root}/final/generation_manifest.jsonl"
                ),
                helper_text,
            )
            self.assertIn(
                f"exec bash {snapshot}/methods/motive/scripts/"
                "auh_submit_wan22_i2v_chain.sh",
                helper_text,
            )
            self.assertFalse((run_root / "final").exists())


if __name__ == "__main__":
    unittest.main()

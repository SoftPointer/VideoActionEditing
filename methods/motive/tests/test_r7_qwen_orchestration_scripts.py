from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKED = ROOT / "scripts" / "auh_r7_qwen_packed.sbatch"
FINALIZER = ROOT / "scripts" / "auh_r7_finalize_qwen_expansion.sbatch"


class R7QwenOrchestrationScriptTests(unittest.TestCase):
    def test_packed_parent_merge_is_bound_to_frozen_snapshot(self) -> None:
        text = PACKED.read_text(encoding="utf-8")
        export_index = text.index(
            'export PYTHONPATH="${code_root}:${source_snapshot}"'
        )
        chdir_index = text.index('cd "${code_root}"')
        srun_index = text.index("\nsrun \\")
        merge_index = text.index(
            '"${python_bin}" -m motive.r7_qwen_merge'
        )

        self.assertIn("export PYTHONDONTWRITEBYTECODE=1", text)
        self.assertLess(export_index, srun_index)
        self.assertLess(chdir_index, srun_index)
        self.assertLess(srun_index, merge_index)

    def test_finalizer_is_bound_and_both_stages_use_strict_resume(self) -> None:
        text = FINALIZER.read_text(encoding="utf-8")
        export_index = text.index(
            'export PYTHONPATH="${code_root}:${source_snapshot}"'
        )
        chdir_index = text.index('cd "${code_root}"')
        merge_index = text.index("-m motive.r7_qwen_merge")
        manifest_index = text.index(
            "-m motive.r7_build_expansion_manifest"
        )

        self.assertLess(export_index, merge_index)
        self.assertLess(chdir_index, merge_index)
        self.assertLess(merge_index, manifest_index)
        self.assertIn("--resume", text[merge_index:manifest_index])
        self.assertGreaterEqual(
            text.count("-m motive.r7_build_expansion_manifest"),
            2,
        )
        self.assertIn("resume_verified=True", text)
        self.assertIn("resume_verified=False", text)

    def test_both_scripts_parse_as_bash(self) -> None:
        for script in (PACKED, FINALIZER):
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


if __name__ == "__main__":
    unittest.main()

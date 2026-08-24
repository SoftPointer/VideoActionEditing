#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (
    METHOD_ROOT
    / "scripts/auh_run_braid_stage0_reference_4f_a_only_all8_v1.sbatch"
)
SHARED = METHOD_ROOT / "scripts/auh_run_braid_stage0_canaries_all8_v1.sbatch"


class BraidReference4FAOnlyAll8LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.shared = SHARED.read_text(encoding="utf-8")

    def test_named_wrapper_requests_one_all8_node_and_selects_only_partial_scope(self) -> None:
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
        subprocess.run(["bash", "-n", str(SHARED)], check=True)
        self.assertIn("#SBATCH --nodes=1", self.wrapper)
        self.assertIn("#SBATCH --gres=gpu:mi210:8", self.wrapper)
        self.assertNotRegex(self.wrapper, r"(?m)^#SBATCH\s+--qos(?:=|\s)")
        self.assertIn(
            "export BRAID_STAGE0_SCOPE=reference-4f-a-only", self.wrapper
        )
        self.assertIn('exec /bin/bash "${launcher_path}"', self.wrapper)

    def test_wrapper_is_hash_authenticated_by_the_immutable_archive(self) -> None:
        name = "auh_run_braid_stage0_reference_4f_a_only_all8_v1.sbatch"
        self.assertIn(name, self.shared)
        self.assertIn("BRAID_STAGE0_SCOPE_LAUNCHER", self.wrapper)
        self.assertIn("BRAID_STAGE0_SCOPE_LAUNCHER", self.shared)
        self.assertIn(
            "partial scope launcher differs from authenticated source archive",
            self.shared,
        )
        self.assertIn(
            "test_auh_run_braid_stage0_reference_4f_a_only_all8_v1.py",
            self.shared,
        )

    def test_partial_scope_runs_only_dog_and_human_reference4fa_world4(self) -> None:
        self.assertIn('launch_cell dog "${dog_seed}" 0,1,2,3', self.shared)
        self.assertIn('launch_cell human "${human_seed}" 4,5,6,7', self.shared)
        self.assertIn("readonly partial_arms=(parity-reset-off-reference-4f-a)", self.shared)
        scope_block = self.shared[
            self.shared.index("if [[ \"${stage0_scope}\" == full-six-arm ]]") :
            self.shared.index("launch_cell() {")
        ]
        self.assertIn('arms=("${partial_arms[@]}")', scope_block)
        self.assertNotIn("capacity-source-bias", scope_block)
        self.assertNotIn("reset-on-reference", scope_block)
        self.assertNotIn("shared-negative", scope_block)
        self.assertIn("--nproc_per_node=4", self.shared)
        self.assertNotIn("--nproc_per_node=8", self.shared)

    def test_partial_publication_is_explicit_and_never_calls_full_aggregate(self) -> None:
        self.assertIn(
            "BRAID_STAGE0_ACK_PARTIAL_STAGE0_NO_STAGE_A_AUTHORITY", self.shared
        )
        partial = self.shared[
            self.shared.index('else\n  publication="${output_root}/reference-4f-a-only') :
            self.shared.index('chmod a-w -- "${publication}"')
        ]
        self.assertIn("aggregate-reference4f-a-only-all8", partial)
        self.assertIn("reference-4f-a-only-all8.receipt.json", partial)
        self.assertNotIn("aggregate-all8", partial)
        self.assertNotIn("all8.manifest.json", partial)
        self.assertIn("partial_stage0=true", self.wrapper)
        self.assertIn("no Stage-A", self.wrapper)

    def test_full_scope_still_fails_before_gpu_when_any_arm_is_unimplemented(self) -> None:
        blocker = self.shared.index(
            "full all8 launcher is intentionally blocked before GPU model load"
        )
        output = self.shared.index('mkdir -- "${output_root}"')
        torchrun = self.shared.index('"${python_bin}" -B -m torch.distributed.run')
        self.assertLess(blocker, output)
        self.assertLess(output, torchrun)
        self.assertIn('scope == "full-six-arm" and implemented != expected', self.shared)

    def test_no_training_decode_or_stage_a_command_is_added(self) -> None:
        for text in (self.wrapper, self.shared):
            self.assertNotRegex(
                text,
                re.compile(r'"\$\{python_bin\}"[^\n]*(?:\.backward\(|optimizer\.step)'),
            )
        runner_call = self.shared[
            self.shared.index('"${runner}" run-world4') : self.shared.index("run_pair() {")
        ]
        self.assertNotRegex(runner_call, r"--(?:train|optimizer|update|decode)(?:\s|=)")


if __name__ == "__main__":
    unittest.main()

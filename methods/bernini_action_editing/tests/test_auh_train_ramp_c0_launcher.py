from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD_ROOT / "scripts" / "auh_train_ramp_c0.sbatch"
CHAIN = METHOD_ROOT / "scripts" / "auh_submit_ramp_c0_chain.sh"


class AUHRAMPLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.chain = CHAIN.read_text(encoding="utf-8")

    def test_bash_syntax_is_valid(self) -> None:
        for path in (LAUNCHER, CHAIN):
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True, check=False
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_embedded_python_blocks_are_valid_ast(self) -> None:
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", self.launcher, re.DOTALL)
        self.assertGreaterEqual(len(blocks), 2)
        for block in blocks:
            ast.parse(block)

    def test_launcher_requests_one_node_all_eight_mi210s(self) -> None:
        for fragment in (
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gres=gpu:mi210:8",
            "#SBATCH --mem=256G",
            "#SBATCH --qos=bgqos",
            "--nproc_per_node=8",
        ):
            self.assertIn(fragment, self.launcher)
        self.assertNotRegex(self.launcher, r"(?m)^\s*srun(?:\s|$)")

    def test_modes_are_exact_one_step_then_afterok_sixteen(self) -> None:
        self.assertIn("engineering-canary) expected_steps=1", self.launcher)
        self.assertIn("afterok-c0) expected_steps=16", self.launcher)
        self.assertIn("--dependency=\"afterok:${canary_job}\"", self.chain)
        self.assertIn("BERNINI_RAMP_C0_MODE=engineering-canary", self.chain)
        self.assertIn("BERNINI_RAMP_C0_MODE=afterok-c0", self.chain)

    def test_durable_code_data_checkpoint_and_outputs_are_hash_bound(self) -> None:
        for fragment in (
            "method archive bytes are not the declared commit",
            "staged pair config differs",
            "durable pair config changed",
            "durable method archive changed",
            "durable pair config must be read-only",
            "durable method archive must be read-only",
            "checkpoint content manifest verification failed",
            "output artifact set differs",
            "output contains a non-plain artifact",
            "run receipt digest differs",
            "artifact digest differs",
            "receipt source binding differs",
        ):
            self.assertIn(fragment, self.launcher)
        self.assertIn("trap 'exit 143' TERM", self.launcher)
        self.assertIn("trap 'exit 130' INT", self.launcher)
        for digest in (
            "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831",
            "6be0d0db0dd483daf1a843efa2b5aafc20090ad11dc0fc6ee8859bdf150635ca",
            "2d2b4591ac053ec25c6371b01a5a6746679e5793",
            "f90b3dc6fbb0ce693745223cc7a94064123dbf4d",
        ):
            self.assertIn(digest, self.launcher)

    def test_launcher_uses_extracted_immutable_trainer_and_acknowledges_pretext(self) -> None:
        for fragment in (
            'method_root="${task_scratch}/source/methods/bernini_action_editing"',
            'find "${method_root}" -type f -exec chmod a-w',
            '"${method_root}/train_ramp_c0.py"',
            "--ack-upstream-training-use-forbidden",
            "--expected-pair-config-sha256",
            "--method-source-revision",
            "--method-source-archive-sha256",
        ):
            self.assertIn(fragment, self.launcher)


if __name__ == "__main__":
    unittest.main()

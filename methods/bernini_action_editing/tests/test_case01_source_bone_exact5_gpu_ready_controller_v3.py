#!/usr/bin/env python3
"""Regression tests for the corrected exact5 GPU launch-receipt contract."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess
import unittest


METHOD = Path(__file__).resolve().parents[1]
LAUNCHER = METHOD / "case01_source_bone_exact5_spooled_launcher_auh_v1.py"
V2 = (
    METHOD / "scripts/"
    "auh_launch_case01_source_bone_exact5_r64_gpu_job143808_node292_once_v2.READY.sh"
)
V3 = (
    METHOD / "scripts/"
    "auh_launch_case01_source_bone_exact5_r64_gpu_job143808_node292_once_v3.READY.sh"
)
V2_SHA256 = "39c3a99c24f1988d408ed864b8815accb72453bf2f9a5ddb98f4ab0b01fff308"
V2_SIZE = 42_375
V3_SHA256 = "5b29f3f18ae1cdb1954325d7ef406b867d1ee3fee2bd1389fad9cb3dc82bd9dd"
V3_SIZE = 42_376
OLD_CONTRACT = (
    'or launch.get("remote_execution_authorized_by_this_receipt") is not True'
)
CORRECT_CONTRACT = (
    'or launch.get("remote_execution_authorized_by_this_receipt") is not False'
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def heredocs(source: str) -> list[str]:
    return [part.split("\nPY", 1)[0] for part in source.split("<<'PY'\n")[1:]]


def literal_assignment(source: str, name: str):
    tree = ast.parse(heredocs(source)[0])
    rows = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(rows) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return ast.literal_eval(rows[0])


def authorization_rejection_expression(source: str):
    tree = ast.parse(heredocs(source)[0])
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        call = node.left
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "launch"
            and call.func.attr == "get"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "remote_execution_authorized_by_this_receipt"
        ):
            matches.append(node)
    if len(matches) != 1:
        raise AssertionError("launch authorization comparison closure differs")
    return matches[0]


class ReadyV3ControllerTests(unittest.TestCase):
    def test_v2_is_preserved_and_v3_has_exactly_one_semantic_diff(self) -> None:
        v2_raw = V2.read_bytes()
        v3_raw = V3.read_bytes()
        self.assertEqual((sha256(v2_raw), len(v2_raw)), (V2_SHA256, V2_SIZE))
        self.assertEqual((sha256(v3_raw), len(v3_raw)), (V3_SHA256, V3_SIZE))
        v2 = v2_raw.decode("utf-8")
        v3 = v3_raw.decode("utf-8")
        self.assertEqual(v2.count(OLD_CONTRACT), 1)
        self.assertNotIn(CORRECT_CONTRACT, v2)
        self.assertNotIn(OLD_CONTRACT, v3)
        self.assertEqual(v3.count(CORRECT_CONTRACT), 1)
        self.assertEqual(v3, v2.replace(OLD_CONTRACT, CORRECT_CONTRACT))
        self.assertTrue(v3_raw.endswith(b"\n"))
        self.assertFalse(v3_raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", v3_raw)

    def test_real_false_contract_is_accepted_and_forged_true_is_rejected(self) -> None:
        source = V3.read_text("utf-8")
        expression = authorization_rejection_expression(source)
        self.assertIsInstance(expression.ops[0], ast.IsNot)
        self.assertIsInstance(expression.comparators[0], ast.Constant)
        self.assertIs(expression.comparators[0].value, False)
        code = compile(ast.fix_missing_locations(ast.Expression(expression)), "auth", "eval")

        def rejected(value):
            launch = {"remote_execution_authorized_by_this_receipt": value}
            return eval(code, {"launch": launch})

        self.assertFalse(rejected(False))
        self.assertTrue(rejected(True))
        self.assertTrue(rejected(None))
        launcher = LAUNCHER.read_text("utf-8")
        self.assertEqual(
            launcher.count('"remote_execution_authorized_by_this_receipt": False'),
            1,
        )
        self.assertNotIn(
            '"remote_execution_authorized_by_this_receipt": True', launcher,
        )

    def test_state_pins_freshness_and_single_srun_are_unchanged(self) -> None:
        v2 = V2.read_text("utf-8")
        v3 = V3.read_text("utf-8")
        self.assertEqual(literal_assignment(v3, "PINS"), literal_assignment(v2, "PINS"))
        self.assertEqual(sum(len(row) for row in literal_assignment(v3, "PINS").values()), 14)
        self.assertIn("readonly CONTROLLER_STATE=READY\n", v3)
        self.assertEqual(re.findall(r'"HOLD_[A-Z0-9_]+"', v3), [])
        self.assertEqual(v3.count("/usr/bin/srun"), 1)
        srun = v3.index("/usr/bin/srun")
        shell_fresh = '[[ ! -e "$ATTEMPT" && ! -L "$ATTEMPT"'
        embedded_fresh = "if os.path.lexists(cache) or os.path.lexists(attempt_path):"
        attempt_claim = '"status": "ATTEMPT_CLAIMED_BEFORE_SRUN"'
        self.assertLess(v3.index(shell_fresh), srun)
        self.assertLess(v3.index(embedded_fresh), v3.index(attempt_claim))
        self.assertLess(v3.index(attempt_claim), srun)
        self.assertIn("print(max(gate_rows))", v3)
        self.assertIn('(( 10#$SLURM_STEP_ID > 10#$1 ))', v3)

    def test_bash_and_both_embedded_programs_compile_normal_and_optimized(self) -> None:
        completed = subprocess.run(
            ["/bin/bash", "-n", str(V3)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        blocks = heredocs(V3.read_text("utf-8"))
        self.assertEqual(len(blocks), 2)
        for optimize in (0, 2):
            for index, block in enumerate(blocks):
                compile(block, f"{V3.name}:heredoc{index}", "exec", optimize=optimize)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Static exact-diff freeze for the AUHv2 receipt-gated CPU controller."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
HOLD_PATH = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.HOLD.py"
)
READY_PATH = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_world4_cpu_auh_controller_v2.READY.py"
)

HOLD_SHA256 = "348f8a86efd3fa664b45909d2b775ab5a565d8ac258634c8ed1bd3879c7f9eb8"
HOLD_SIZE = 87_014
READY_SHA256 = "9d5aebcdf4b7938848e0763b839010fbd58df196f8a0155515b05a032cc99cbd"
READY_SIZE = 86_998
HOLD_STATE = "HOLD_PENDING_AUHV2_STAGING_RECEIPT_REVIEW_AND_ACTIVATION"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_CPU_ADMISSION"
HOLD_ASSIGNMENT = f'CONTROLLER_STATE = "{HOLD_STATE}"\n'.encode("ascii")
READY_ASSIGNMENT = f'CONTROLLER_STATE = "{READY_STATE}"\n'.encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assignment(tree: ast.Module, name: str) -> ast.expr:
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return matches[0]


def literal(tree: ast.Module, name: str):
    return ast.literal_eval(assignment(tree, name))


class ReceiptGatedConsumerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hold_raw = HOLD_PATH.read_bytes()
        cls.ready_raw = READY_PATH.read_bytes()
        cls.source = cls.ready_raw.decode("utf-8")
        cls.hold_tree = ast.parse(cls.hold_raw, filename=str(HOLD_PATH))
        cls.tree = ast.parse(cls.ready_raw, filename=str(READY_PATH))

    def test_frozen_hashes_and_exactly_one_state_line_difference(self) -> None:
        self.assertEqual(
            (sha256(self.hold_raw), len(self.hold_raw)),
            (HOLD_SHA256, HOLD_SIZE),
        )
        self.assertEqual(
            (sha256(self.ready_raw), len(self.ready_raw)),
            (READY_SHA256, READY_SIZE),
        )
        self.assertEqual(self.hold_raw.count(HOLD_ASSIGNMENT), 1)
        self.assertNotIn(READY_ASSIGNMENT, self.hold_raw)
        self.assertNotIn(HOLD_ASSIGNMENT, self.ready_raw)
        self.assertEqual(self.ready_raw.count(READY_ASSIGNMENT), 1)
        self.assertEqual(
            self.ready_raw,
            self.hold_raw.replace(HOLD_ASSIGNMENT, READY_ASSIGNMENT, 1),
        )
        differing = [
            (before, after)
            for before, after in zip(
                self.hold_raw.splitlines(keepends=True),
                self.ready_raw.splitlines(keepends=True),
            )
            if before != after
        ]
        self.assertEqual(differing, [(HOLD_ASSIGNMENT, READY_ASSIGNMENT)])
        self.assertEqual(
            len(self.hold_raw.splitlines()), len(self.ready_raw.splitlines()),
        )
        self.assertTrue(self.ready_raw.endswith(b"\n"))
        self.assertFalse(self.ready_raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", self.ready_raw)

    def test_hold_and_ready_states_are_exact(self) -> None:
        self.assertEqual(
            literal(self.hold_tree, "CONTROLLER_STATE"), HOLD_STATE,
        )
        self.assertEqual(literal(self.tree, "CONTROLLER_STATE"), READY_STATE)
        self.assertEqual(
            literal(self.tree, "CONTROLLER_STATE"),
            literal(self.tree, "READY_STATE"),
        )

    def test_final_auh_v2_receipt_and_exact15_contract(self) -> None:
        self.assertEqual(
            literal(self.tree, "SOURCE_STAGE_SCHEMA") + "-receipt",
            "case01-object-trajectory-exact5-source-stager-auh-v2-receipt",
        )
        self.assertEqual(
            literal(self.tree, "SOURCE_STAGE_PUBLICATION_PROTOCOL"),
            "posix_rename_same_parent_under_held_O_EXCL_receipt_reservation",
        )
        rows = literal(self.tree, "SOURCE_STAGE_AUTHORITIES")
        self.assertEqual(len(rows), 15)
        self.assertEqual(
            [row["relative"] for row in rows],
            sorted(row["relative"] for row in rows),
        )
        for row in rows:
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(row["size"], 0)
        for required in (
            '"cooperative_writer_exclusion"',
            '"receipt_is_consumption_gate"',
            '"receipt_is_admission"',
            '"uncooperative_same_uid_race_out_of_scope"',
            '"STAGED_RECEIPT_GATED"',
            '"RECOVERED_RECEIPT_ONLY"',
        ):
            self.assertIn(required, self.source)

    def test_gate_precedes_every_login_mutation_and_srun(self) -> None:
        controller = self.source[self.source.index("def controller()") :]
        gate = controller.index("source_stage_gate = open_source_stage_gate()")
        for mutation in (
            "os.mkdir(TARGET_ROOT",
            "attempt_raw = _create_json(ATTEMPT_PATH, attempt)",
            "returncode, stdout_raw, stderr_raw = _run_single_srun(",
        ):
            self.assertLess(gate, controller.index(mutation), mutation)
        compute = self.source[
            self.source.index("def compute(plan_b64: str)") :
            self.source.index("def _seal_log(")
        ]
        self.assertLess(
            compute.index("source_stage_gate = open_source_stage_gate()"),
            compute.index("staged, compute_source_rows, receipt_raw, result = _execute_world4("),
        )
        self.assertEqual(self.source.count('"/usr/bin/srun"'), 1)
        self.assertEqual(self.source.count("subprocess.Popen("), 1)
        self.assertNotIn("subprocess.run(", self.source)
        self.assertNotIn('"retry_allowed": True', self.source)

    def test_static_compile_succeeds_without_execution(self) -> None:
        for path, raw in (
            (HOLD_PATH, self.hold_raw), (READY_PATH, self.ready_raw),
        ):
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(raw, str(path), "exec", optimize=optimize),
                )
        guards = [
            node for node in self.tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Name) and child.id == "__name__"
                for child in ast.walk(node.test)
            )
        ]
        self.assertEqual(len(guards), 1)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Static-only exact-diff freeze for the CPU deployer READY overlay."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
HOLD_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_deploy_case01_object_trajectory_world4_cpu_controller_v2_once_v1.HOLD.py"
)
READY_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_deploy_case01_object_trajectory_world4_cpu_controller_v2_once_v2.READY.py"
)
BOOTSTRAP_PATH = (
    METHOD_ROOT / "tools"
    / "case01_object_trajectory_world4_cpu_controller_deploy_bootstrap_v1.py"
)
HOLD_SHA256 = "48081947adf883b1c944ea56750f21171a74fa3527426bffbe0055ead683440b"
HOLD_SIZE = 83_263
READY_SHA256 = "5a65b78e14b21ec6c2860ccc30f815a108169404d349c720aabf492b1ae76b70"
READY_SIZE = 83_261
BOOTSTRAP_SHA256 = "2dca27942a3faae8fb0db019784682e288029f6794f573ae848c6292d0a15216"
BOOTSTRAP_SIZE = 118_408
HOLD_ASSIGNMENT = (
    b'CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"\n'
)
READY_ASSIGNMENT = (
    b'CONTROLLER_STATE = "READY_EXPLICIT_CREATE_ONLY_CONTROLLER_DEPLOY"\n'
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assignment(tree: ast.Module, name: str) -> ast.expr:
    rows = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(rows) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return rows[0]


def literal(tree: ast.Module, name: str):
    return ast.literal_eval(assignment(tree, name))


class CpuControllerDeployerReadyFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Reading/parsing/compiling bytes is the entire test boundary.  The
        # READY source is deliberately never imported, exec'ed, or run.
        cls.hold_raw = HOLD_PATH.read_bytes()
        cls.ready_raw = READY_PATH.read_bytes()
        cls.hold_tree = ast.parse(cls.hold_raw, filename=str(HOLD_PATH))
        cls.ready_tree = ast.parse(cls.ready_raw, filename=str(READY_PATH))
        cls.bootstrap_raw = BOOTSTRAP_PATH.read_bytes()

    def test_frozen_tuples_and_exactly_one_state_line_difference(self) -> None:
        self.assertEqual(
            (sha256(self.hold_raw), len(self.hold_raw)),
            (HOLD_SHA256, HOLD_SIZE),
        )
        self.assertEqual(
            (sha256(self.ready_raw), len(self.ready_raw)),
            (READY_SHA256, READY_SIZE),
        )
        self.assertEqual(
            (sha256(self.bootstrap_raw), len(self.bootstrap_raw)),
            (BOOTSTRAP_SHA256, BOOTSTRAP_SIZE),
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
            (left, right)
            for left, right in zip(
                self.hold_raw.splitlines(keepends=True),
                self.ready_raw.splitlines(keepends=True),
            )
            if left != right
        ]
        self.assertEqual(differing, [(HOLD_ASSIGNMENT, READY_ASSIGNMENT)])
        self.assertEqual(
            len(self.hold_raw.splitlines()), len(self.ready_raw.splitlines()),
        )

    def test_ready_state_equals_frozen_ready_authority(self) -> None:
        self.assertEqual(
            literal(self.hold_tree, "CONTROLLER_STATE"),
            "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY",
        )
        self.assertEqual(
            literal(self.ready_tree, "CONTROLLER_STATE"),
            literal(self.ready_tree, "READY_STATE"),
        )
        for name in (
            "SCHEMA", "MANIFEST_SCHEMA", "PAYLOAD_SCHEMA", "RECEIPT_SCHEMA",
            "RECEIPT_RESERVATION_SCHEMA", "TERMINAL_SCHEMA",
            "LOCAL_TERMINAL_SCHEMA", "AUDIT_SCHEMA", "READY_STATE", "LOCAL_ROOT",
            "LOCAL_COMMIT_TERMINAL_PATH",
            "LOCAL_CONTROLLER", "LOCAL_CONTROLLER_SHA256",
            "LOCAL_CONTROLLER_SIZE", "LOCAL_BOOTSTRAP",
            "LOCAL_BOOTSTRAP_SHA256", "LOCAL_BOOTSTRAP_SIZE", "REMOTE_PARENT",
            "REMOTE_TARGET_ROOT", "REMOTE_CONTROLLER_PATH",
            "REMOTE_RECEIPT_PATH", "REMOTE_PYTHON", "REMOTE_PYTHON_SHA256",
            "REMOTE_PYTHON_SIZE", "SSH_PATH", "SSH_SHA256", "SSH_IDENTITY",
            "SSH_IDENTITY_SHA256", "SSH_KNOWN_HOSTS",
            "SSH_KNOWN_HOSTS_SHA256", "SSH_DESTINATION",
            "TRANSPORT_TIMEOUT_SECONDS", "REMOTE_LOADER_SOURCE",
            "STAGE_OPERATION", "RECOVER_RECEIPT_OPERATION",
        ):
            self.assertEqual(
                ast.dump(assignment(self.hold_tree, name), include_attributes=False),
                ast.dump(assignment(self.ready_tree, name), include_attributes=False),
                name,
            )
        self.assertEqual(
            literal(self.ready_tree, "LOCAL_BOOTSTRAP_SHA256"),
            BOOTSTRAP_SHA256,
        )
        self.assertEqual(
            literal(self.ready_tree, "LOCAL_BOOTSTRAP_SIZE"), BOOTSTRAP_SIZE,
        )

    def test_main_state_gate_remains_first_and_execute_requires_token(self) -> None:
        functions = {
            node.name: node
            for node in self.ready_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        main = functions.get("main")
        self.assertIsInstance(main, ast.FunctionDef)
        self.assertGreater(len(main.body), 1)
        first = main.body[0]
        self.assertIsInstance(first, ast.If)
        first_names = {
            child.id for child in ast.walk(first.test)
            if isinstance(child, ast.Name)
        }
        self.assertEqual(first_names, {"CONTROLLER_STATE", "READY_STATE"})
        source = self.ready_raw.decode("utf-8")
        self.assertIn('values[0] == "--execute"', source)
        self.assertIn('values[1] == authorization_token()', source)
        self.assertIn('values == ["--audit-local"]', source)
        self.assertIn('values[0] == RECOVER_RECEIPT_OPERATION', source)
        self.assertIn('_write_local_commit_terminal(', source)
        self.assertIn('pass_fds=()', source)
        self.assertIn('close_fds=True', source)
        self.assertNotIn('f"/dev/fd/{transport[', source)
        self.assertNotIn('IdentityFile=/dev/fd/', source)
        self.assertIn(
            'posix_rename_same_parent_under_held_O_EXCL_receipt_reservation',
            source,
        )
        self.assertIn('"rename_noreplace": False', source)
        self.assertIn('"automatic_remote_retry": False', source)
        self.assertNotIn("subprocess.run(", source)
        self.assertEqual(source.count("subprocess.Popen("), 1)

    def test_normal_and_optimized_compile_only(self) -> None:
        for path, raw in (
            (HOLD_PATH, self.hold_raw), (READY_PATH, self.ready_raw),
            (BOOTSTRAP_PATH, self.bootstrap_raw),
        ):
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(raw, str(path), "exec", optimize=optimize),
                )
        self.assertTrue(self.ready_raw.endswith(b"\n"))
        self.assertFalse(self.ready_raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", self.ready_raw)


if __name__ == "__main__":
    unittest.main()

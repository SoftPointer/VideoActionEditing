#!/usr/bin/env python3
"""Static-only freeze for the physical15 source-stager READY state copy.

The READY source is never imported, executed, or passed to a subprocess.  The
test boundary is limited to byte reads, AST parsing, hashing, and compile-only
checks of the independently approved HOLD source and its one-line state copy.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
HOLD_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_stage_case01_object_trajectory_exact5_source_once_v1.HOLD.py"
)
READY_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_stage_case01_object_trajectory_exact5_source_once_v2.READY.py"
)
BOOTSTRAP_PATH = (
    METHOD_ROOT
    / "case01_object_trajectory_exact5_source_stager_remote_bootstrap_v1.py"
)
HOLD_SHA256 = "9823481b4913cd06d4e8b26fccfae3b5b59b7ec741c6a8f9efee1d52d600d246"
HOLD_SIZE = 171_334
READY_SHA256 = "5782e4fc2885e27ca88af015ada88c2dc50a70f4b0fc342357c86305d2e98d19"
READY_SIZE = 171_325
BOOTSTRAP_SHA256 = (
    "33c63bb114d6008bd32c67819cd86fb4acce7b796696c7ed34f41a431836e08a"
)
BOOTSTRAP_SIZE = 116_778
HOLD_ASSIGNMENT = (
    b'CONTROLLER_STATE = "HOLD_FINAL_PINS_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"\n'
)
READY_ASSIGNMENT = (
    b'CONTROLLER_STATE = "READY_EXPLICIT_RECEIPT_RESERVED_PHYSICAL15_STAGE"\n'
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assignment(tree: ast.Module, name: str) -> ast.expr:
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            matches.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            matches.append(node.value)
    if len(matches) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return matches[0]


def literal(tree: ast.Module, name: str):
    return ast.literal_eval(assignment(tree, name))


def resolved_literal(tree: ast.Module, name: str):
    """Evaluate constants while resolving references to frozen assignments."""
    resolving: set[str] = set()

    def convert(node: ast.AST):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                convert(key): convert(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Name):
            if node.id in resolving:
                raise AssertionError("constant assignment cycle differs")
            resolving.add(node.id)
            try:
                return convert(assignment(tree, node.id))
            finally:
                resolving.remove(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return convert(node.left) + convert(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub),
        ):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise AssertionError(
            "nonliteral frozen assignment differs: " + ast.dump(node)
        )

    return convert(assignment(tree, name))


class Physical15SourceStagerReadyFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hold_raw = HOLD_PATH.read_bytes()
        cls.ready_raw = READY_PATH.read_bytes()
        cls.bootstrap_raw = BOOTSTRAP_PATH.read_bytes()
        cls.hold_tree = ast.parse(cls.hold_raw, filename=str(HOLD_PATH))
        cls.ready_tree = ast.parse(cls.ready_raw, filename=str(READY_PATH))

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
        differences = [
            (before, after)
            for before, after in zip(
                self.hold_raw.splitlines(keepends=True),
                self.ready_raw.splitlines(keepends=True),
            )
            if before != after
        ]
        self.assertEqual(differences, [(HOLD_ASSIGNMENT, READY_ASSIGNMENT)])
        self.assertEqual(
            len(self.hold_raw.splitlines()), len(self.ready_raw.splitlines()),
        )
        self.assertTrue(self.ready_raw.endswith(b"\n"))
        self.assertFalse(self.ready_raw.endswith(b"\n\n"))
        self.assertNotIn(b"\r", self.ready_raw)

    def test_ready_state_and_all_authority_literals_remain_frozen(self) -> None:
        self.assertEqual(
            literal(self.hold_tree, "CONTROLLER_STATE"),
            "HOLD_FINAL_PINS_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY",
        )
        self.assertEqual(
            literal(self.ready_tree, "CONTROLLER_STATE"),
            literal(self.ready_tree, "READY_STATE"),
        )
        for name in (
            "SCHEMA", "MANIFEST_SCHEMA", "PAYLOAD_SCHEMA", "RECEIPT_SCHEMA",
            "TERMINAL_SCHEMA", "LOCAL_TERMINAL_SCHEMA",
            "TRANSPORT_DIAGNOSTIC_SCHEMA",
            "TRANSPORT_DIAGNOSTIC_PREFIX_LIMIT", "READY_STATE",
            "SOURCE_AUTHORITIES", "REMOTE_BOOTSTRAP_PATH",
            "REMOTE_BOOTSTRAP_SHA256", "REMOTE_BOOTSTRAP_SIZE",
            "LOCAL_SOURCE_ROOT", "REMOTE_PARENT", "REMOTE_TARGET_ROOT",
            "REMOTE_RECEIPT_PATH", "REMOTE_PYTHON", "REMOTE_PYTHON_SHA256",
            "REMOTE_PYTHON_SIZE", "SSH_PATH", "SSH_SHA256", "SSH_SIZE",
            "LOCAL_UID", "LOCAL_GID",
            "DARWIN_SF_RESTRICTED", "SSH_FAT_ARCHITECTURES",
            "SSH_CODE_DIRECTORIES",
            "SSH_IDENTITY", "SSH_IDENTITY_SHA256", "SSH_IDENTITY_SIZE",
            "SSH_KNOWN_HOSTS", "SSH_KNOWN_HOSTS_SHA256",
            "SSH_KNOWN_HOSTS_SIZE", "SSH_DESTINATION",
        ):
            self.assertEqual(
                ast.dump(
                    assignment(self.hold_tree, name), include_attributes=False,
                ),
                ast.dump(
                    assignment(self.ready_tree, name), include_attributes=False,
                ),
                name,
            )
        self.assertEqual(
            literal(self.ready_tree, "REMOTE_BOOTSTRAP_SHA256"),
            BOOTSTRAP_SHA256,
        )
        self.assertEqual(
            literal(self.ready_tree, "REMOTE_BOOTSTRAP_SIZE"),
            BOOTSTRAP_SIZE,
        )
        sources = resolved_literal(self.ready_tree, "SOURCE_AUTHORITIES")
        self.assertEqual(len(sources), 15)
        self.assertEqual(len({row["relative"] for row in sources}), 15)
        for row in sources:
            self.assertRegex(row["sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertGreater(row["size"], 0)

    def test_main_gate_and_transport_shape_are_byte_identical(self) -> None:
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
            node.id for node in ast.walk(first.test)
            if isinstance(node, ast.Name)
        }
        self.assertEqual(first_names, {"CONTROLLER_STATE", "READY_STATE"})
        source = self.ready_raw.decode("utf-8")
        self.assertEqual(source.count("subprocess.Popen("), 1)
        self.assertNotIn("subprocess.run(", source)
        self.assertIn('start_new_session=True', source)
        self.assertIn('str(SSH_PATH), "-F", "/dev/null"', source)
        self.assertNotIn('f"/dev/fd/{transport[', source)
        self.assertIn('f"IdentityFile={SSH_IDENTITY}"', source)
        self.assertIn('f"UserKnownHostsFile={SSH_KNOWN_HOSTS}"', source)
        self.assertIn('close_fds=True', source)
        self.assertIn('pass_fds=()', source)
        self.assertIn('stat.S_IMODE(opened.st_mode) != 0o600', source)
        self.assertIn('stat.S_IMODE(opened.st_mode) != 0o755', source)
        self.assertIn('_validate_named_transport_authorities(transport)', source)
        self.assertIn('def _transport_terminal_diagnostic(', source)
        self.assertIn('"prefix_may_contain_remote_echo_of_input": True', source)
        self.assertIn('opened_flags & DARWIN_SF_RESTRICTED', source)
        self.assertIn('os.fstatvfs(descriptor)', source)
        self.assertIn('os.statvfs(path)', source)
        self.assertIn('magic != 0xFADE0CC0', source)
        self.assertIn('"-I", "-S", "-B"', source)

    def test_normal_and_optimized_compile_only(self) -> None:
        for path, raw in (
            (HOLD_PATH, self.hold_raw),
            (READY_PATH, self.ready_raw),
            (BOOTSTRAP_PATH, self.bootstrap_raw),
        ):
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(raw, str(path), "exec", optimize=optimize),
                )


if __name__ == "__main__":
    unittest.main()

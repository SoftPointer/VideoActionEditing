#!/usr/bin/env python3
"""Static exact-diff freeze for the final-pinned exact5 GPU READY overlay."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
HOLD_PATH = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v2.HOLD.py"
)
READY_PATH = METHOD_ROOT / (
    "scripts/auh_launch_case01_object_trajectory_exact5_r64_gpu_once_v2.READY.py"
)

HOLD_SHA256 = "2138eaacee1c43cccb2e10ee8e56e4de7a89c8853ba6523b0a5a8571ff04bd07"
HOLD_SIZE = 185_372
READY_SHA256 = "651f533f564e73455dec78b218fc4fada80e44071af88a8b096f496041f8b2ae"
READY_SIZE = 185_369
HOLD_STATE = "HOLD_PENDING_STATIC_ROOTFAKE_WORLD4_AND_ROOT_PINS"
READY_STATE = "READY_EXPLICIT_SINGLE_SRUN_EXACT_FIVE_NO_RETRY"
READY_AUTHORIZATION_TOKEN = (
    "c321fd5aadf3ed0d75edaa0defdcdb6102b63da7c855d5d5c741d6a4d2ea6ba8"
)
HOLD_ASSIGNMENT = f'CONTROLLER_STATE = "{HOLD_STATE}"\n'.encode("ascii")
READY_ASSIGNMENT = f'CONTROLLER_STATE = "{READY_STATE}"\n'.encode("ascii")

FINAL_PINS = {
    "STATIC_RECEIPT_SHA256":
        "3e65f4342f33a0d4264fa7f09759bad3aa2f4c4622a6965db675f2c551fb07b8",
    "STATIC_RECEIPT_SIZE": 1_035,
    "STATIC_RECEIPT_DIGEST":
        "7ed16825624ca99dc7f2cbbea3c9a5a991122108aff4867796a3ac01456ab6be",
    "ROOT_FAKE_RECEIPT_SHA256":
        "af4cb28c23bc9e7a8355133f2068d02af5f97eda16083fa8b591e5131062f619",
    "ROOT_FAKE_RECEIPT_SIZE": 1_975,
    "ROOT_FAKE_RECEIPT_DIGEST":
        "4a65b5dab48904fced093fd0bff0c16a50c13b5caa30b6a70dc4e4ae9c6b170a",
    "WORLD4_RECEIPT_SHA256":
        "872a54dd93cd75f9dada706740d90bf56842d1fc524c97745792698add194e86",
    "WORLD4_RECEIPT_SIZE": 49_351,
    "WORLD4_RECEIPT_DIGEST":
        "3bfd87493da8f8813e26ff2db074d3d61d6cfecae45e046c76e987e4f48edf6c",
    "WORLD4_EVIDENCE_SHA256":
        "d6c1f71db64983ce0be0b8eba6524b1e0b36e14ea80948ccfce43ec201746295",
    "WORLD4_EVIDENCE_SIZE": 14_866,
    "WORLD4_EVIDENCE_DIGEST":
        "8759174e6c610f3fa2ad3e686a2c6ec16de89db46ff7632933c525d4971eb77c",
}


def load(path: Path) -> types.ModuleType:
    name = "_test_exact5_gpu_ready_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def literal(tree: ast.Module, name: str):
    matches = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            isinstance(getattr(node, "target", None), ast.Name)
            and node.target.id == name
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
    ]
    if len(matches) != 1:
        raise AssertionError(f"{name} assignment closure differs")
    return ast.literal_eval(matches[0])


class FinalPinnedGPUReadyStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hold_raw = HOLD_PATH.read_bytes()
        cls.ready_raw = READY_PATH.read_bytes()
        cls.hold_tree = ast.parse(cls.hold_raw, filename=str(HOLD_PATH))
        cls.ready_tree = ast.parse(cls.ready_raw, filename=str(READY_PATH))
        cls.ready = load(READY_PATH)

    def test_frozen_tuples_and_exactly_one_state_line_difference(self):
        self.assertEqual(
            (hashlib.sha256(self.hold_raw).hexdigest(), len(self.hold_raw)),
            (HOLD_SHA256, HOLD_SIZE),
        )
        self.assertEqual(
            (hashlib.sha256(self.ready_raw).hexdigest(), len(self.ready_raw)),
            (READY_SHA256, READY_SIZE),
        )
        self.assertEqual(self.hold_raw.count(HOLD_ASSIGNMENT), 1)
        self.assertEqual(self.ready_raw.count(READY_ASSIGNMENT), 1)
        self.assertNotIn(READY_ASSIGNMENT, self.hold_raw)
        self.assertNotIn(HOLD_ASSIGNMENT, self.ready_raw)
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

    def test_states_final_pins_paths_and_token_are_exact(self):
        self.assertEqual(literal(self.hold_tree, "CONTROLLER_STATE"), HOLD_STATE)
        self.assertEqual(literal(self.ready_tree, "CONTROLLER_STATE"), READY_STATE)
        self.assertEqual(literal(self.ready_tree, "READY_STATE"), READY_STATE)
        for name, expected in FINAL_PINS.items():
            self.assertEqual(literal(self.hold_tree, name), expected, name)
            self.assertEqual(literal(self.ready_tree, name), expected, name)
        self.assertEqual(
            self.ready.ROOT_FAKE_RECEIPT_PATH.name,
            "exact5_root_fake_runner_probe_receipt_v1.json",
        )
        self.assertEqual(
            self.ready.WORLD4_RECEIPT_PATH.name, "exact5_world4_receipt_v3.json",
        )
        self.assertEqual(
            self.ready.WORLD4_EVIDENCE_PATH.name,
            "exact5_world4_controller_receipt_v3.json",
        )
        self.assertEqual(self.ready.blocked_dynamic_pins(), ())
        self.assertEqual(self.ready.authorization_token(), READY_AUTHORIZATION_TOKEN)

    def test_ready_without_token_is_inert_before_io_in_normal_and_optimized(self):
        forbidden = mock.Mock(side_effect=AssertionError("READY crossed token gate"))
        with mock.patch.object(self.ready, "open_package_gate", forbidden), \
                mock.patch.object(self.ready.os, "lstat", forbidden), \
                mock.patch.object(self.ready.subprocess, "Popen", forbidden), \
                mock.patch.object(
                    sys, "stderr", new_callable=io.StringIO,
                ) as stderr:
            self.assertEqual(self.ready.main([]), 88)
        self.assertIn("authorization token required", stderr.getvalue())
        forbidden.assert_not_called()
        result = subprocess.run(
            [sys.executable, "-O", str(READY_PATH)], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 88)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"authorization token required", result.stderr)

    def test_hold_and_ready_static_compile_without_execution(self):
        for path, raw in (
            (HOLD_PATH, self.hold_raw), (READY_PATH, self.ready_raw),
        ):
            for optimize in (0, 1, 2):
                self.assertIsNotNone(
                    compile(raw, str(path), "exec", optimize=optimize),
                )
        self.assertEqual(self.ready.build_srun_argv(0).count("/usr/bin/srun"), 1)
        self.assertEqual(self.ready.build_srun_argv(0).count("--ntasks=1"), 1)
        self.assertEqual(self.ready.build_srun_argv(0).count("--gpus-per-node=8"), 1)


if __name__ == "__main__":
    unittest.main()

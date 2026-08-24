#!/usr/bin/env python3
"""Static HOLD/READY tests for the fresh dual-source package controller."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = METHOD_ROOT / (
    "scripts/auh_materialize_case01_object_trajectory_exact5_r64_overlay_package_once_v2.HOLD.py"
)
READY_PATH = METHOD_ROOT / (
    "scripts/auh_materialize_case01_object_trajectory_exact5_r64_overlay_package_once_v2.READY.py"
)
MATERIALIZER_PATH = METHOD_ROOT / (
    "tools/materialize_case01_object_trajectory_exact5_r64_overlay_package_v2.py"
)


def load(path: Path) -> types.ModuleType:
    name = "_test_case01_overlay_controller_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OverlayPackageControllerTest(unittest.TestCase):
    def test_hold_state_is_the_first_gate(self) -> None:
        controller = load(CONTROLLER_PATH)
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("crossed")
            raise AssertionError("HOLD state crossed")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(["--execute", "malicious"])
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_dual_source_remote_overlay_tuple_is_exact_and_unblocked(self) -> None:
        controller = load(CONTROLLER_PATH)
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        pins = controller.dynamic_pin_values()
        self.assertEqual(
            {
                "overlay_receipt_sha256": pins["overlay_receipt_sha256"],
                "overlay_receipt_size": pins["overlay_receipt_size"],
                "overlay_receipt_digest": pins["overlay_receipt_digest"],
                "overlay_root_identity": pins["overlay_root_identity"],
            },
            {
                "overlay_receipt_sha256":
                    "4ea56107b4c1171a22d76267e05bf315bf9130cbe9cfc6d80fa97bbcc30622b0",
                "overlay_receipt_size": 3_568,
                "overlay_receipt_digest":
                    "b90c920338963f2882fe4a3d669d5464f05e19eaafa75bb606db02ea216ab97a",
                "overlay_root_identity": [
                    48, 7093358014530864437, 2012, 2000, 16749, 2, 0,
                    4096, 0, 1787369767167114564, 1787369814456054102,
                ],
            },
        )
        self.assertEqual(
            (controller.MATERIALIZER_SHA256, controller.MATERIALIZER_SIZE),
            (
                "9df88457d593d5d105f305da4362ed226d12402ecd8f1704ab726cf20099e698",
                119_400,
            ),
        )
        self.assertEqual(controller.PACKAGE_ROOT.name, (
            "bernini_case01_object_trajectory_exact5_r64_canary_v2"
        ))
        self.assertEqual(
            controller.PACKAGE_RECEIPT_PATH.name,
            "bernini_case01_object_trajectory_exact5_r64_canary_v2.publication_receipt_v3.json",
        )
        self.assertEqual(
            controller.INTERNAL_RECEIPT_RELATIVE,
            "authority/package_materialization_receipt_v3.json",
        )
        self.assertEqual(len(controller.PRODUCTION_IDENTITY_ROLES), 26)
        self.assertIn("object_wrapper_inner", controller.PRODUCTION_IDENTITY_ROLES)

    def test_ready_is_exact_state_only_copy_and_is_never_invoked(self) -> None:
        hold_lines = CONTROLLER_PATH.read_bytes().splitlines(keepends=True)
        ready_lines = READY_PATH.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(hold_lines), len(ready_lines))
        differences = [
            index for index, (hold, ready) in enumerate(
                zip(hold_lines, ready_lines)
            )
            if hold != ready
        ]
        self.assertEqual(len(differences), 1)
        changed = differences[0]
        self.assertEqual(
            hold_lines[changed],
            b'CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"\n',
        )
        self.assertEqual(
            ready_lines[changed],
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_R64_HOLD_PACKAGE"\n',
        )
        hold = load(CONTROLLER_PATH)
        ready = load(READY_PATH)
        self.assertNotEqual(hold.CONTROLLER_STATE, hold.READY_STATE)
        self.assertEqual(ready.CONTROLLER_STATE, ready.READY_STATE)
        self.assertEqual(hold.dynamic_pin_values(), ready.dynamic_pin_values())
        self.assertEqual(hold.blocked_dynamic_pins(), ())
        self.assertEqual(ready.blocked_dynamic_pins(), ())
        self.assertEqual(hold.authorization_token(), ready.authorization_token())

    def test_controller_loads_only_the_overlay_materializer_contract(self) -> None:
        controller = load(CONTROLLER_PATH)
        materializer = controller.load_materializer(MATERIALIZER_PATH.read_bytes())
        self.assertEqual(materializer.TARGET_ROOT, controller.PACKAGE_ROOT)
        self.assertEqual(materializer.OVERLAY_ROOT, controller.OVERLAY_ROOT)
        self.assertEqual(
            materializer.OVERLAY_RECEIPT_PATH, controller.OVERLAY_RECEIPT_PATH,
        )
        self.assertEqual(
            materializer.OVERLAY_MATERIALIZER_RELATIVE,
            controller.MATERIALIZER_RELATIVE,
        )
        self.assertEqual(
            materializer.BASE_MATERIALIZER_SHA256,
            controller.BASE_MATERIALIZER_SHA256,
        )
        self.assertEqual(len(materializer.RELEASE_FILES), 25)
        self.assertEqual(len(materializer.OVERLAY_RELEASE_FILES), 4)

    def test_exact_materializer_argv_and_no_scheduler_surface(self) -> None:
        controller = load(CONTROLLER_PATH)
        argv = controller.materializer_argv(71, 72)
        self.assertEqual(argv[:6], [
            "/proc/self/fd/71", "-I", "-S", "-B", "/proc/self/fd/72",
            "materialize",
        ])
        self.assertIn("--overlay-root", argv)
        self.assertIn("--overlay-receipt", argv)
        self.assertIn("--snapshot-materializer-sha256", argv)
        self.assertIn("--overlay-materializer-sha256", argv)
        self.assertIn("--overlay-receipt-sha256", argv)
        self.assertIn("--overlay-receipt-size", argv)
        self.assertIn("--overlay-receipt-digest", argv)
        self.assertIn("--overlay-root-identity-json", argv)
        self.assertEqual(
            argv[argv.index("--overlay-receipt-sha256") + 1],
            "4ea56107b4c1171a22d76267e05bf315bf9130cbe9cfc6d80fa97bbcc30622b0",
        )
        self.assertEqual(
            argv[argv.index("--overlay-receipt-size") + 1], "3568",
        )
        self.assertEqual(
            argv[argv.index("--overlay-receipt-digest") + 1],
            "b90c920338963f2882fe4a3d669d5464f05e19eaafa75bb606db02ea216ab97a",
        )
        self.assertEqual(
            argv[argv.index("--overlay-root-identity-json") + 1],
            controller.canonical(controller.OVERLAY_ROOT_IDENTITY).decode("utf-8"),
        )
        self.assertNotIn("srun", argv)
        self.assertNotIn("sbatch", argv)
        self.assertNotIn("ssh", argv)

        tree = ast.parse(CONTROLLER_PATH.read_bytes(), filename=str(CONTROLLER_PATH))
        popen_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "Popen"
        ]
        self.assertEqual(len(popen_calls), 1)

    def test_run_one_uses_one_exact_spawn_and_saved_process_group(self) -> None:
        controller = load(CONTROLLER_PATH)
        runtime = types.SimpleNamespace(descriptor=71)
        materializer = types.SimpleNamespace(descriptor=72)

        class Process:
            pid = 4242
            returncode = 0
            stdin = None
            stdout = None
            stderr = None

            def communicate(self, timeout):
                self.timeout = timeout
                return b'{"status":"PASS"}\n', b""

        process = Process()
        with mock.patch.object(
            controller.subprocess, "Popen", return_value=process,
        ) as popen, mock.patch.object(
            controller, "_process_group_present", return_value=False,
        ), mock.patch.object(
            controller, "_seal_process_group",
        ) as seal:
            stdout, stderr, returncode, argv = controller.run_one_materializer(
                runtime, materializer,
            )
        self.assertEqual((stdout, stderr, returncode), (
            b'{"status":"PASS"}\n', b"", 0,
        ))
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args, (argv,))
        self.assertEqual(popen.call_args.kwargs["executable"], "/proc/self/fd/71")
        self.assertEqual(popen.call_args.kwargs["pass_fds"], (71, 72))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        seal.assert_called_once_with(process, 4242)

    def test_early_leader_and_term_ignoring_descendant_reach_esrch(self) -> None:
        controller = load(CONTROLLER_PATH)
        source = (
            "import os,signal,sys,time\n"
            "child=os.fork()\n"
            "if child == 0:\n"
            " signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            " sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            " while True: time.sleep(1)\n"
            "time.sleep(0.05)\n"
            "os._exit(0)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.35)
            # poll() may already report the leader terminal here.  Cleanup
            # must still use the PGID saved at spawn and kill the descendant.
            process.poll()
            with mock.patch.object(
                controller, "PROCESS_TERM_GRACE_SECONDS", 0.10,
            ), mock.patch.object(
                controller, "PROCESS_KILL_GRACE_SECONDS", 1.50,
            ):
                controller._seal_process_group(process, process.pid)
            self.assertTrue(controller._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)

    def test_pipe_close_error_does_not_bypass_process_group_zero_gate(self) -> None:
        controller = load(CONTROLLER_PATH)
        source = (
            "import signal,sys,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", source], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        original_stdout = process.stdout

        class BrokenClose:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.closed = False

            def close(self):
                if not self.closed:
                    self.closed = True
                    self.wrapped.close()
                raise OSError("synthetic pipe close failure")

        try:
            self.assertIsNotNone(original_stdout)
            self.assertEqual(original_stdout.readline(), b"READY\n")
            process.stdout = BrokenClose(original_stdout)
            with mock.patch.object(
                controller, "PROCESS_TERM_GRACE_SECONDS", 0.10,
            ), mock.patch.object(
                controller, "PROCESS_KILL_GRACE_SECONDS", 1.50,
            ), self.assertRaisesRegex(
                controller.PackageControllerError, "terminal pipe seal",
            ):
                controller._seal_process_group(process, process.pid)
            self.assertTrue(controller._process_group_absent(process.pid, 0.5))
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=1)

    def test_child_failure_keeps_one_0400_attempt_no_evidence_or_retry(self) -> None:
        controller = load(CONTROLLER_PATH)
        events: list[str] = []

        class FakeHeld:
            def __init__(self, path: Path, raw: bytes, descriptor: int):
                self.path = path
                self.raw = raw
                self.descriptor = descriptor
                self.held_identity = tuple(range(11))

            def replay(self):
                events.append("replay:" + self.path.name)

            def close(self):
                events.append("close:" + self.path.name)

            def row(self):
                return {
                    "path": str(self.path),
                    "sha256": hashlib.sha256(self.raw).hexdigest(),
                    "size": len(self.raw),
                    "identity": list(self.held_identity),
                }

        class FakeDirectory:
            def __init__(self, path: Path, descriptor: int):
                self.path = path
                self.descriptor = descriptor
                self.held_identity = tuple(range(11))

            def replay(self):
                events.append("replay-dir:" + self.path.name)

            def close(self):
                events.append("close-dir:" + self.path.name)

            def row(self):
                return {
                    "path": str(self.path),
                    "identity": list(self.held_identity),
                }

        overlay_receipt = FakeHeld(
            controller.OVERLAY_RECEIPT_PATH, b"overlay-receipt", 21,
        )
        materializer_held = FakeHeld(
            controller.MATERIALIZER_PATH, b"materializer", 22,
        )
        snapshot_receipt = FakeHeld(
            controller.SNAPSHOT_RECEIPT_PATH, b"snapshot-receipt", 23,
        )
        snapshot_manifest = FakeHeld(
            controller.SNAPSHOT_MANIFEST_PATH, b"snapshot-manifest", 24,
        )
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", 25)
        self_held = FakeHeld(CONTROLLER_PATH, b"controller", 26)
        overlay_root = FakeDirectory(controller.OVERLAY_ROOT, 31)
        snapshot_root = FakeDirectory(controller.SNAPSHOT_ROOT, 32)

        snapshot_bytes = {controller.MATERIALIZER_RELATIVE: b"base-materializer"}
        snapshot_evidence = {
            "sha256": controller.SNAPSHOT_MANIFEST_SHA256,
            "size": controller.SNAPSHOT_MANIFEST_SIZE,
            "manifest_digest": controller.SNAPSHOT_MANIFEST_DIGEST,
            "snapshot_publication_receipt": {
                "sha256": controller.SNAPSHOT_RECEIPT_SHA256,
                "size": controller.SNAPSHOT_RECEIPT_SIZE,
                "receipt_digest": controller.SNAPSHOT_RECEIPT_DIGEST,
            },
            "staging_receipt_authority": {
                "sha256": "0" * 64, "receipt_digest": "1" * 64,
            },
        }
        overlay_bytes = {
            controller.MATERIALIZER_RELATIVE: materializer_held.raw,
        }
        overlay_evidence = {
            "receipt": {
                "sha256": controller.OVERLAY_RECEIPT_SHA256,
                "size": controller.OVERLAY_RECEIPT_SIZE,
                "receipt_digest": controller.OVERLAY_RECEIPT_DIGEST,
            },
            "root_identity": controller.OVERLAY_ROOT_IDENTITY,
        }

        class FakeMaterializer:
            def preflight_snapshot(self, *_args, **_kwargs):
                events.append("preflight-snapshot")
                return snapshot_bytes, snapshot_evidence

            def preflight_overlay(self, *_args, **_kwargs):
                events.append("preflight-overlay")
                return overlay_bytes, overlay_evidence

        materializer = FakeMaterializer()

        def open_authority(path, **_kwargs):
            return {
                controller.OVERLAY_RECEIPT_PATH: overlay_receipt,
                controller.SNAPSHOT_RECEIPT_PATH: snapshot_receipt,
                controller.SNAPSHOT_MANIFEST_PATH: snapshot_manifest,
            }[path]

        def open_directory(path, **_kwargs):
            return {
                controller.OVERLAY_ROOT: overlay_root,
                controller.SNAPSHOT_ROOT: snapshot_root,
            }[path]

        canonical_tmp = Path(tempfile.gettempdir()).resolve(strict=True)
        with tempfile.TemporaryDirectory(
            prefix="case01-controller-failure-", dir=canonical_tmp,
        ) as temporary:
            experiment_root = Path(temporary).resolve()
            attempt_path = experiment_root / "attempt.json"
            evidence_path = experiment_root / "evidence.json"
            owner = os.lstat(experiment_root)
            real_create = controller.create_immutable_json

            def tracked_create(path, value):
                events.append("create:" + path.name)
                return real_create(path, value)

            def child_failure(*_args):
                events.append("child")
                raise RuntimeError("hostile child failure")

            with mock.patch.object(
                controller, "EXPERIMENTS", experiment_root,
            ), mock.patch.object(
                controller, "ATTEMPT_PATH", attempt_path,
            ), mock.patch.object(
                controller, "EVIDENCE_PATH", evidence_path,
            ), mock.patch.object(
                controller, "REMOTE_UID", owner.st_uid,
            ), mock.patch.object(
                controller, "REMOTE_GID", owner.st_gid,
            ), mock.patch.object(
                controller, "open_authority", side_effect=open_authority,
            ), mock.patch.object(
                controller, "open_directory", side_effect=open_directory,
            ), mock.patch.object(
                controller, "validate_overlay_receipt",
                return_value=(
                    {"receipt_digest": controller.OVERLAY_RECEIPT_DIGEST},
                    [materializer_held],
                ),
            ), mock.patch.object(
                controller, "load_materializer", return_value=materializer,
            ), mock.patch.object(
                controller, "validate_snapshot_receipt", return_value={},
            ), mock.patch.object(
                controller, "validate_snapshot_manifest", return_value={},
            ), mock.patch.object(
                controller, "open_runtime_authority", return_value=runtime,
            ), mock.patch.object(
                controller, "open_self_authority", return_value=self_held,
            ), mock.patch.object(
                controller, "require_fresh_outputs",
            ), mock.patch.object(
                controller, "create_immutable_json", side_effect=tracked_create,
            ), mock.patch.object(
                controller, "run_one_materializer", side_effect=child_failure,
            ) as child, mock.patch.object(
                controller, "validate_package",
            ) as validate_package, self.assertRaisesRegex(
                controller.PackageControllerError,
                "failed after the durable attempt claim",
            ):
                controller.controller()

            child.assert_called_once()
            validate_package.assert_not_called()
            self.assertTrue(attempt_path.is_file())
            self.assertEqual(stat.S_IMODE(attempt_path.stat().st_mode), 0o400)
            self.assertFalse(evidence_path.exists())
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            self.assertTrue(attempt["single_attempt"])
            self.assertFalse(attempt["retry_allowed"])
            self.assertFalse(attempt["launch_allowed"])
            self.assertEqual(events.count("child"), 1)
            self.assertLess(events.index("create:attempt.json"), events.index("child"))
            self.assertNotIn("create:evidence.json", events)

    def test_fresh_v3_receipt_and_crosslink_field_closures(self) -> None:
        controller = load(CONTROLLER_PATH)
        self.assertEqual(
            controller.PACKAGE_RECEIPT_SCHEMA,
            "case01-object-trajectory-exact5-r64-package-publication-v3-receipt",
        )
        self.assertEqual(
            controller.MATERIALIZATION_SCHEMA,
            "case01-object-trajectory-exact5-r64-hold-materialization-v3",
        )
        self.assertTrue({
            "source_overlay_receipt_sha256",
            "source_overlay_receipt_digest",
            "source_overlay_root_identity",
        } <= controller.PACKAGE_RECEIPT_FIELDS)
        self.assertTrue({
            "source_overlay", "source_provenance", "release", "production",
        } <= controller.REPORT_FIELDS)
        self.assertEqual(
            controller.EVIDENCE_SCHEMA,
            "case01-object-trajectory-exact5-r64-overlay-package-controller-v2-evidence",
        )


if __name__ == "__main__":
    unittest.main()

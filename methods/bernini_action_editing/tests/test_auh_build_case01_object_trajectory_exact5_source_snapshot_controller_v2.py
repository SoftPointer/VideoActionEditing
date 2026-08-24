#!/usr/bin/env python3
"""Hostile/static tests for the receipt-first exact35 HOLD controller."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_build_case01_object_trajectory_exact5_source_snapshot_once_v2.HOLD.py"
)
READY_PATH = (
    METHOD_ROOT / "scripts"
    / "auh_build_case01_object_trajectory_exact5_source_snapshot_once_v2.READY.py"
)
BUILDER_PATH = (
    METHOD_ROOT / "tools"
    / "build_case01_object_trajectory_exact5_source_snapshot_v1.py"
)
BUILDER_SHA256 = (
    "8ece3b3310b4065ceb8b7b8331f61d0ab6897f35e25febabd0f705f202a31432"
)
BUILDER_SIZE = 66_981


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> types.ModuleType:
    name = "_test_exact35_hold_controller_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controller = load(CONTROLLER_PATH)


class FakeHeld:
    def __init__(self, path: Path, raw: bytes, events: list[str]) -> None:
        self.path = path
        self.raw = raw
        self.events = events
        self.descriptor = 101
        self.held_identity = tuple(range(11))

    def replay(self) -> None:
        self.events.append("replay:" + self.path.name)

    def close(self) -> None:
        self.events.append("close:" + self.path.name)

    def row(self) -> dict[str, object]:
        return {
            "path": str(self.path), "sha256": sha256(self.raw),
            "size": len(self.raw), "identity": list(self.held_identity),
        }


class FakeGate:
    def __init__(
        self, receipt: FakeHeld, receipt_value: dict[str, object],
        events: list[str],
    ) -> None:
        self.receipt = receipt
        self.receipt_value = receipt_value
        self.events = events

    def replay(self) -> None:
        self.events.append("replay:source-gate")

    def close(self) -> None:
        self.events.append("close:source-gate")


class Exact35SnapshotControllerTest(unittest.TestCase):
    def test_hold_gate_is_first_and_performs_no_controller_io(self) -> None:
        touched: list[str] = []

        def forbidden(*_args, **_kwargs):
            touched.append("called")
            raise AssertionError("HOLD crossed the first state gate")

        stderr = io.StringIO()
        with mock.patch.object(controller, "blocked_dynamic_pins", forbidden), \
             mock.patch.object(controller, "controller", forbidden), \
             redirect_stderr(stderr):
            result = controller.main(["--execute", "malicious-token"])
        self.assertEqual(result, 88)
        self.assertEqual(touched, [])
        self.assertIn("HOLD", stderr.getvalue())

    def test_final_dynamic_receipt_pins_are_exact_and_complete(self) -> None:
        self.assertEqual(
            controller.dynamic_pin_values(),
            {
                "source_receipt_sha256":
                    "d91b18336ab56c72f95891da842e8ae57261f68c9a340b0bafbf9f0beeca8c5f",
                "source_receipt_size": 5_347,
                "source_receipt_digest":
                    "b13fc3ba5e9f61bfd244492da66570bf91db8d4fba373ccab8522ab256429091",
                "cpu_world4_receipt_sha256":
                    "61d72a7e37fc197fdab24f7173e74b289ee53e92379f5089ab89d5cdfb348083",
                "cpu_world4_receipt_size": 49_335,
                "cpu_world4_receipt_digest":
                    "bcf618ad9eeafeebf6dcbc794a9d4bf5fbd27fa13274ec5264d4672a9944ad28",
                "cpu_controller_evidence_sha256":
                    "0e138b349688028ad7bed82602e01e1b441e190857e87937d2d96cbb556879a2",
                "cpu_controller_evidence_size": 12_395,
                "cpu_controller_evidence_digest":
                    "b7195233777db70fa5ad068f0e88de7828cf62826bb80f0c950b9b01366209ee",
            },
        )
        self.assertEqual(controller.blocked_dynamic_pins(), ())
        token = controller.authorization_token()
        self.assertRegex(token, r"[0-9a-f]{64}\Z")
        with mock.patch.object(controller, "SOURCE_RECEIPT_SHA256", "0" * 64):
            self.assertNotEqual(controller.authorization_token(), token)

    def test_builder_pin_state_gate_and_single_call_are_static(self) -> None:
        raw = CONTROLLER_PATH.read_bytes()
        ready_raw = READY_PATH.read_bytes()
        builder_raw = BUILDER_PATH.read_bytes()
        tree = ast.parse(raw, filename=str(CONTROLLER_PATH))
        self.assertEqual((sha256(builder_raw), len(builder_raw)), (
            BUILDER_SHA256, BUILDER_SIZE,
        ))
        self.assertEqual(controller.BUILDER_SHA256, BUILDER_SHA256)
        self.assertEqual(controller.BUILDER_SIZE, BUILDER_SIZE)
        self.assertEqual(
            controller.CONTROLLER_STATE,
            "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY",
        )
        hold_assignment = (
            b'CONTROLLER_STATE = "HOLD_PENDING_INDEPENDENT_REVIEW_AND_STATE_COPY"\n'
        )
        ready_assignment = (
            b'CONTROLLER_STATE = "READY_EXPLICIT_SINGLE_ATTEMPT_EXACT35_SNAPSHOT"\n'
        )
        self.assertEqual(raw.count(hold_assignment), 1)
        self.assertNotIn(ready_assignment, raw)
        self.assertEqual(ready_raw.count(ready_assignment), 1)
        self.assertNotIn(hold_assignment, ready_raw)
        self.assertEqual(
            ready_raw, raw.replace(hold_assignment, ready_assignment, 1),
        )
        differing = [
            (left, right)
            for left, right in zip(
                raw.splitlines(keepends=True), ready_raw.splitlines(keepends=True),
            )
            if left != right
        ]
        self.assertEqual(differing, [(hold_assignment, ready_assignment)])
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        main = functions["main"]
        self.assertIsInstance(main.body[0], ast.If)
        first_names = {
            node.id for node in ast.walk(main.body[0].test)
            if isinstance(node, ast.Name)
        }
        self.assertEqual(first_names, {"CONTROLLER_STATE", "READY_STATE"})
        imports = {
            alias.name.split(".")[0]
            for node in tree.body if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertTrue({"subprocess", "socket", "tempfile"}.isdisjoint(imports))
        build_calls = [
            node for node in ast.walk(functions["controller"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "build"
        ]
        self.assertEqual(len(build_calls), 1)
        source = raw.decode("utf-8")
        self.assertLess(
            source.index("source_receipt = open_authority("),
            source.index("builder_authority = open_authority("),
        )
        self.assertLess(
            source.index("create_immutable_json(ATTEMPT_PATH, attempt)"),
            source.index("returned_manifest = builder.build("),
        )
        self.assertNotIn("os.rename(", source)
        self.assertNotIn("os.replace(", source)
        self.assertNotIn("subprocess.", source)
        for path, source_raw in ((CONTROLLER_PATH, raw), (READY_PATH, ready_raw)):
            for optimize in (0, 2):
                self.assertIsNotNone(
                    compile(source_raw, str(path), "exec", optimize=optimize)
                )
            self.assertTrue(source_raw.endswith(b"\n"))
            self.assertFalse(source_raw.endswith(b"\n\n"))
            self.assertNotIn(b"\r", source_raw)

    def test_strict_json_rejects_duplicate_and_noncanonical_bytes(self) -> None:
        self.assertEqual(
            controller.strict_json(b'{"a":1,"b":2}\n', label="valid"),
            {"a": 1, "b": 2},
        )
        for raw in (
            b'{"a":1,"a":2}\n', b'{"b":2, "a":1}\n',
            b'{"a":NaN}\n', b'{"a":1}',
        ):
            with self.subTest(raw=raw), self.assertRaises(
                controller.SnapshotControllerError
            ):
                controller.strict_json(raw, label="hostile")

    def test_snapshot_rows_require_exact34_unique_lexical_paths(self) -> None:
        duplicate = {
            "path": "duplicate", "sha256": "0" * 64, "size": 1,
            "mode": 0o444, "provenance": "hostile",
        }
        with self.assertRaisesRegex(
            controller.SnapshotControllerError, "manifest rows differ"
        ):
            controller.validate_snapshot_tree({"files": [duplicate] * 34})
        malformed = [dict(duplicate, path=f"row-{index:02d}") for index in range(34)]
        malformed[-1]["path"] = None
        with self.assertRaisesRegex(
            controller.SnapshotControllerError, "manifest rows differ"
        ):
            controller.validate_snapshot_tree({"files": malformed})

    def test_held_authority_rejects_links_and_named_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            path = base / "authority.json"
            raw = b'{"authority":true}\n'
            path.write_bytes(raw); path.chmod(0o400)
            authority_info = path.stat()
            held = controller.open_authority(
                path, expected_sha256=sha256(raw), expected_size=len(raw),
                expected_mode=0o400, expected_uid=authority_info.st_uid,
                expected_gid=authority_info.st_gid,
            )
            held.replay()
            replacement = base / "replacement.json"
            replacement.write_bytes(raw); replacement.chmod(0o400)
            os.replace(replacement, path)
            with self.assertRaisesRegex(
                controller.SnapshotControllerError, "held authority changed"
            ):
                held.replay()
            held.close()

            link = base / "link.json"
            link.symlink_to(path)
            with self.assertRaises(controller.SnapshotControllerError):
                controller.open_authority(
                    link, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=authority_info.st_uid,
                    expected_gid=authority_info.st_gid,
                )
            hardlink = base / "hardlink.json"
            os.link(path, hardlink)
            with self.assertRaises(controller.SnapshotControllerError):
                controller.open_authority(
                    path, expected_sha256=sha256(raw), expected_size=len(raw),
                    expected_mode=0o400, expected_uid=authority_info.st_uid,
                    expected_gid=authority_info.st_gid,
                )

    def test_create_only_json_seals_once_and_never_replaces(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            target = base / "attempt.json"
            parent_info = base.stat()
            with mock.patch.object(controller, "EXPERIMENTS", base), \
                 mock.patch.object(controller, "REMOTE_UID", parent_info.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", parent_info.st_gid):
                raw, anchor = controller.create_immutable_json(
                    target, {"schema_version": "test", "status": "CLAIMED"},
                )
                self.assertEqual(target.read_bytes(), raw)
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
                self.assertEqual(anchor, controller.inode_anchor(target.stat()))
                with self.assertRaises(FileExistsError):
                    controller.create_immutable_json(target, {"replacement": True})
            self.assertEqual(target.read_bytes(), raw)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)

    def test_postseal_fsync_error_never_demotes_or_unlinks_0400(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value).resolve()
            target = base / "attempt.json"
            parent_info = base.stat()
            real_fsync = os.fsync
            calls = 0

            def fail_parent_fsync(descriptor: int) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("hostile parent fsync error after seal")
                real_fsync(descriptor)

            with mock.patch.object(controller, "EXPERIMENTS", base), \
                 mock.patch.object(controller, "REMOTE_UID", parent_info.st_uid), \
                 mock.patch.object(controller, "REMOTE_GID", parent_info.st_gid), \
                 mock.patch.object(controller.os, "fsync", fail_parent_fsync), \
                 self.assertRaisesRegex(OSError, "hostile parent fsync"):
                controller.create_immutable_json(
                    target, {"schema_version": "test", "status": "CLAIMED"},
                )
            self.assertTrue(target.exists())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o400)
            self.assertEqual(
                target.read_bytes(),
                controller.canonical({
                    "schema_version": "test", "status": "CLAIMED",
                }) + b"\n",
            )

    def _run_mocked_controller(self, *, builder_error: Exception | None = None):
        events: list[str] = []
        source_value = {"manifest_digest": "1" * 64}
        world4 = FakeHeld(controller.CPU_WORLD4_RECEIPT_PATH, b"world4", events)
        cpu = FakeHeld(controller.CPU_CONTROLLER_EVIDENCE_PATH, b"cpu", events)
        source = FakeHeld(controller.SOURCE_RECEIPT_PATH, b"source", events)
        runtime = FakeHeld(controller.VACE_PYTHON, b"python", events)
        self_held = FakeHeld(Path(str(CONTROLLER_PATH)), b"self", events)
        builder_held = FakeHeld(controller.BUILDER_PATH, b"builder", events)
        gate = FakeGate(source, source_value, events)

        class FakeBuilder:
            def open_staging_gate(self, supplied_sha: str):
                events.append("open:source-gate")
                self.assert_sha = supplied_sha
                return gate

            def build(self, old, staging, target, *, builder_sha256):
                events.append("builder:build")
                self.build_args = (old, staging, target, builder_sha256)
                if builder_error is not None:
                    raise builder_error
                return {"manifest": "returned"}

        builder = FakeBuilder()
        held_by_path = {
            controller.CPU_WORLD4_RECEIPT_PATH: world4,
            controller.CPU_CONTROLLER_EVIDENCE_PATH: cpu,
            controller.SOURCE_RECEIPT_PATH: source,
            controller.BUILDER_PATH: builder_held,
        }

        def open_side_effect(path, **_kwargs):
            events.append("open:" + Path(path).name)
            return held_by_path[path]

        created: list[tuple[Path, dict[str, object]]] = []

        def create_side_effect(path, value):
            events.append("create:" + path.name)
            copied = dict(value)
            created.append((path, copied))
            return controller.canonical(copied) + b"\n", [1, 2, 3, 4, 5]

        def publication_side_effect(returned, held_source):
            events.append("validate:publication")
            self.assertEqual(returned, {"manifest": "returned"})
            self.assertIs(held_source, source)
            return {
                "manifest": {"sha256": "2" * 64},
                "publication_receipt": {"sha256": "3" * 64},
                "target_root_identity": list(range(11)),
                "file_count": 35, "directory_count": 8,
            }

        patches = (
            mock.patch.object(controller, "open_authority", side_effect=open_side_effect),
            mock.patch.object(controller, "validate_cpu_authorities", return_value=(
                {}, {"source_staging_receipt": {}},
            )),
            mock.patch.object(controller, "validate_source_receipt_prefix", return_value=source_value),
            mock.patch.object(controller, "validate_cpu_source_crosslink"),
            mock.patch.object(controller, "open_runtime_authority", return_value=runtime),
            mock.patch.object(controller, "open_self_authority", return_value=self_held),
            mock.patch.object(controller, "load_builder", return_value=builder),
            mock.patch.object(controller, "require_fresh_outputs"),
            mock.patch.object(controller, "create_immutable_json", side_effect=create_side_effect),
            mock.patch.object(controller, "validate_publication", side_effect=publication_side_effect),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9]:
            if builder_error is None:
                result = controller.controller()
                return result, events, created, builder
            with self.assertRaisesRegex(
                controller.SnapshotControllerError,
                "failed after the durable attempt claim",
            ):
                controller.controller()
        return None, events, created, builder

    def test_receipts_are_opened_before_builder_and_attempt_before_build(self) -> None:
        result, events, created, builder = self._run_mocked_controller()
        self.assertEqual(builder.assert_sha, BUILDER_SHA256)
        self.assertEqual(builder.build_args, (
            controller.OLD_ROOT, controller.SOURCE_ROOT, controller.TARGET_ROOT,
            BUILDER_SHA256,
        ))
        self.assertLess(
            events.index("open:" + controller.CPU_WORLD4_RECEIPT_PATH.name),
            events.index("open:" + controller.SOURCE_RECEIPT_PATH.name),
        )
        self.assertLess(
            events.index("open:" + controller.SOURCE_RECEIPT_PATH.name),
            events.index("open:" + controller.BUILDER_PATH.name),
        )
        self.assertLess(
            events.index("create:" + controller.ATTEMPT_PATH.name),
            events.index("builder:build"),
        )
        self.assertLess(
            events.index("builder:build"), events.index("validate:publication")
        )
        self.assertEqual(events.count("builder:build"), 1)
        self.assertEqual([path for path, _value in created], [
            controller.ATTEMPT_PATH, controller.EVIDENCE_PATH,
        ])
        self.assertEqual(created[0][1]["status"], "ATTEMPT_CLAIMED_BEFORE_BUILDER")
        self.assertEqual(
            created[1][1]["status"], "PASS_EXACT35_PUBLISHED_RECEIPT_GATED"
        )
        self.assertFalse(result["launch_allowed"])
        self.assertFalse(result["retry_allowed"])

    def test_builder_failure_keeps_single_attempt_and_no_evidence(self) -> None:
        result, events, created, _builder = self._run_mocked_controller(
            builder_error=RuntimeError("hostile builder refusal")
        )
        self.assertIsNone(result)
        self.assertEqual(events.count("builder:build"), 1)
        self.assertEqual([path for path, _value in created], [controller.ATTEMPT_PATH])
        self.assertNotIn("validate:publication", events)


if __name__ == "__main__":
    unittest.main()

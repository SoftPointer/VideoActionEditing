#!/usr/bin/env python3
"""Local regression tests for the root-fake v2 visibility barrier."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import threading
import time
import types
import unittest


METHOD = Path(__file__).resolve().parents[1]
GATE_V1 = (
    METHOD / "scripts/"
    "auh_gate_case01_source_bone_exact5_root_fake_job143808_node292_once_v1.sh"
)
GATE_V2 = (
    METHOD / "scripts/"
    "auh_gate_case01_source_bone_exact5_root_fake_job143808_node292_once_v2.sh"
)
V1_SHA256 = "e040b28c76898c5b21eaeeba15e89e1fbeb614d171d9fdd99f637bf3d3f29988"
V1_SIZE = 26_516


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode() + b"\n"


def heredocs(source: str) -> list[str]:
    return [part.split("\nPY", 1)[0] for part in source.split("<<'PY'\n")[1:]]


def barrier_namespace() -> dict:
    block = heredocs(GATE_V2.read_text("utf-8"))[1]
    tree = ast.parse(block)
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    namespace: dict = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "visibility-barrier", "exec"),
        namespace,
    )
    return namespace


class VisibilityFixture:
    def __init__(self, base: Path):
        self.parent = (base / "evidence").resolve()
        self.parent.mkdir()
        self.parent.chmod(0o755)
        self.receipt = self.parent / "receipt.json"
        self.held_fd = os.open(
            self.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        parent_stat = os.fstat(self.held_fd)
        self.uid = parent_stat.st_uid
        self.gid = parent_stat.st_gid

    def close(self) -> None:
        os.close(self.held_fd)

    def publish(self, delay: float = 0.0, commit_delay: float = 0.0) -> None:
        if delay:
            time.sleep(delay)
        raw = canonical({"schema_version": "probe-v1", "status": "PASS"})
        descriptor = os.open(
            self.receipt,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0,
        )
        try:
            if commit_delay:
                time.sleep(commit_delay)
            offset = 0
            while offset < len(raw):
                count = os.write(descriptor, raw[offset:])
                if count <= 0:
                    raise RuntimeError("test publisher made no progress")
                offset += count
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RootFakeGateV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.namespace = barrier_namespace()

    def wait(self, fixture: VisibilityFixture, timeout: float, poll: float):
        return self.namespace["wait_for_receipt"](
            fixture.held_fd,
            str(fixture.parent),
            str(fixture.receipt),
            timeout,
            poll,
            fixture.uid,
            fixture.gid,
        )

    def test_v1_preserved_v2_single_lf_unique_srun_and_barrier_order(self) -> None:
        v1 = GATE_V1.read_bytes()
        v2 = GATE_V2.read_bytes()
        source = v2.decode("utf-8")
        self.assertEqual((sha256(v1), len(v1)), (V1_SHA256, V1_SIZE))
        self.assertTrue(v2.endswith(b"\n"))
        self.assertFalse(v2.endswith(b"\n\n"))
        self.assertEqual(source.count("/usr/bin/srun"), 1)
        srun = source.index("/usr/bin/srun")
        barrier = source.index("root-fake visibility argv differs")
        postflight = source.index("root-fake postflight argv differs")
        self.assertLess(srun, barrier)
        self.assertLess(barrier, postflight)
        self.assertIn(
            "os.O_RDONLY|os.O_CLOEXEC|os.O_DIRECTORY|os.O_NOFOLLOW", source,
        )
        self.assertIn("post_evidencefd=os.open", source)
        self.assertIn("stable_at(post_evidencefd", source)
        self.assertIn("create(post_evidencefd", source)
        self.assertIn(
            "while True:\n  if time.monotonic()>deadline: "
            "raise RuntimeError(\"root-fake receipt visibility timeout\")\n"
            "  freshfd=acquire_parent(parent)",
            source,
        )
        self.assertIn("root-fake receipt visibility timeout", source)
        self.assertIn("[[ \"$SRUN_RC\" -eq 0 ]] || exit \"$SRUN_RC\"", source)
        self.assertIn("no retry or second srun", source)

    def test_bash_n_and_all_embedded_python_normal_and_optimized(self) -> None:
        checked = subprocess.run(
            ["/bin/bash", "-n", str(GATE_V2)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())
        blocks = heredocs(GATE_V2.read_text("utf-8"))
        self.assertEqual(len(blocks), 3)
        for optimize in (0, 2):
            for index, block in enumerate(blocks):
                compile(
                    block,
                    f"{GATE_V2.name}:heredoc{index}",
                    "exec",
                    dont_inherit=True,
                    optimize=optimize,
                )

    def test_delayed_mode_zero_then_canonical_0400_receipt_passes(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            fixture = VisibilityFixture(Path(value))
            failures: list[BaseException] = []

            def publish() -> None:
                try:
                    fixture.publish(delay=0.03, commit_delay=0.04)
                except BaseException as error:  # surfaced in the test thread
                    failures.append(error)

            thread = threading.Thread(target=publish)
            thread.start()
            try:
                raw, identity = self.wait(fixture, 1.0, 0.005)
                thread.join(timeout=1.0)
                self.assertFalse(thread.is_alive())
                self.assertEqual(failures, [])
                self.assertEqual(raw, canonical({"schema_version": "probe-v1", "status": "PASS"}))
                self.assertEqual(stat.S_IMODE(identity[4]), 0o400)
            finally:
                fixture.close()

    def test_missing_receipt_times_out_bounded(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            fixture = VisibilityFixture(Path(value))
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(RuntimeError, "visibility timeout"):
                    self.wait(fixture, 0.04, 0.005)
            finally:
                fixture.close()
            elapsed = time.monotonic() - started
            self.assertGreaterEqual(elapsed, 0.03)
            self.assertLess(elapsed, 0.5)

    def test_first_fresh_view_permanently_negative_second_view_visible(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            fixture = VisibilityFixture(Path(value))
            fixture.publish()
            real_acquire = self.namespace["acquire_parent"]
            real_receipt_once = self.namespace["receipt_once"]
            acquisitions: list[int] = []

            def counted_acquire(parent):
                descriptor = real_acquire(parent)
                acquisitions.append(descriptor)
                return descriptor

            def first_view_negative(parentfd, name, uid, gid):
                if len(acquisitions) == 1:
                    return None
                return real_receipt_once(parentfd, name, uid, gid)

            self.namespace["acquire_parent"] = counted_acquire
            self.namespace["receipt_once"] = first_view_negative
            try:
                raw, _ = self.wait(fixture, 0.5, 0.005)
                self.assertEqual(raw, fixture.receipt.read_bytes())
                self.assertGreaterEqual(len(acquisitions), 2)
            finally:
                fixture.close()

    def test_receipt_symlink_and_parent_symlink_swap_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value)
            fixture = VisibilityFixture(base)
            outside = base / "outside.json"
            outside.write_bytes(canonical({"status": "PASS"}))
            outside.chmod(0o400)
            fixture.receipt.symlink_to(outside)
            try:
                with self.assertRaisesRegex(RuntimeError, "type/owner/link"):
                    self.wait(fixture, 0.1, 0.005)
            finally:
                fixture.close()

        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value)
            fixture = VisibilityFixture(base)
            held_name = base / "evidence-held"
            swapped = False
            original_parent_contract = self.namespace["parent_contract"]

            def observed_parent_contract(*args):
                nonlocal swapped
                result = original_parent_contract(*args)
                if not swapped:
                    fixture.parent.rename(held_name)
                    fixture.parent.symlink_to(held_name, target_is_directory=True)
                    swapped = True
                return result

            self.namespace["parent_contract"] = observed_parent_contract
            try:
                with self.assertRaisesRegex(RuntimeError, "fresh visibility parent"):
                    self.wait(fixture, 1.0, 0.005)
                self.assertTrue(swapped)
            finally:
                fixture.close()

    def test_ordinary_directory_parent_replacement_fails_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            base = Path(value)
            fixture = VisibilityFixture(base)
            held_name = base / "evidence-held"
            replaced = False
            original_parent_contract = self.namespace["parent_contract"]

            def replace_after_first_contract(*args):
                nonlocal replaced
                result = original_parent_contract(*args)
                if not replaced:
                    fixture.parent.rename(held_name)
                    fixture.parent.mkdir()
                    fixture.parent.chmod(0o755)
                    replaced = True
                return result

            self.namespace["parent_contract"] = replace_after_first_contract
            try:
                with self.assertRaisesRegex(RuntimeError, "parent identity differs"):
                    self.wait(fixture, 1.0, 0.005)
                self.assertTrue(replaced)
            finally:
                fixture.close()


if __name__ == "__main__":
    unittest.main()

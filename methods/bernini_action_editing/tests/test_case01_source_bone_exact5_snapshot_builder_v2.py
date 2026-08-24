#!/usr/bin/env python3
"""Real-temp regression tests for the exact5 snapshot mode correction."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid


METHOD = Path(__file__).resolve().parents[1]
BUILDER_V1 = METHOD / "tools/build_case01_source_bone_exact5_source_snapshot_v1.py"
BUILDER_V2 = METHOD / "tools/build_case01_source_bone_exact5_source_snapshot_v2.py"
CONTROLLER_V1 = (
    METHOD / "scripts/auh_build_case01_source_bone_exact5_source_snapshot_once_v1.sh"
)
CONTROLLER_V2 = (
    METHOD / "scripts/auh_build_case01_source_bone_exact5_source_snapshot_once_v2.sh"
)
CONTROLLER_V3 = (
    METHOD / "scripts/auh_build_case01_source_bone_exact5_source_snapshot_once_v3.sh"
)
CONTROLLER_V4 = (
    METHOD / "scripts/auh_build_case01_source_bone_exact5_source_snapshot_once_v4.sh"
)
V1_BUILDER_SHA = "906db79519f8689f4ec3a2ceee626f788cd8d9f032178ba2b61346a5108d9a69"
V1_CONTROLLER_SHA = "6f1c552d074be008e5312cc1fb87de201bceb6215c2e7436353402a7285ad1d7"
V2_CONTROLLER_SHA = "acd0be06675a1059074a8c8993d11e9c2c9a51ae2a6d31d8276f45b74bf84489"
V3_CONTROLLER_SHA = "d74bb408ce2182791734dfefa610d78fa2e7f5d572ebc4c66da47e25f8d5e6fc"
V4_CONTROLLER_SHA = "4bff3a5ba53af9aa8e2a2541794eddde945fd500e7f9c4572e996e1953dbcde1"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(path: Path) -> types.ModuleType:
    name = "_test_snapshot_builder_v2_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load(BUILDER_V2)


class OwnedStat:
    """Present a real temp object's owner as uid2012/gid2000."""

    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        if name == "st_uid":
            return 2012
        if name == "st_gid":
            return 2000
        return getattr(self._value, name)


@contextmanager
def auh_authority_view():
    real_fstat, real_stat, real_lstat = os.fstat, os.stat, os.lstat
    real_path_lstat = Path.lstat
    real_ident = builder.ident

    def owned_fstat(*args, **kwargs):
        return OwnedStat(real_fstat(*args, **kwargs))

    def owned_stat(*args, **kwargs):
        return OwnedStat(real_stat(*args, **kwargs))

    def owned_lstat(*args, **kwargs):
        return OwnedStat(real_lstat(*args, **kwargs))

    def owned_path_lstat(path):
        return OwnedStat(real_path_lstat(path))

    def owned_ident(value):
        return real_ident(OwnedStat(value))

    with mock.patch.object(builder.os, "fstat", owned_fstat), \
         mock.patch.object(builder.os, "stat", owned_stat), \
         mock.patch.object(builder.os, "lstat", owned_lstat), \
         mock.patch.object(Path, "lstat", owned_path_lstat), \
         mock.patch.object(builder, "ident", owned_ident), \
         mock.patch.object(builder.os, "geteuid", return_value=2012), \
         mock.patch.object(builder.os, "getegid", return_value=2000):
        yield


class BuildFixture:
    def __init__(self, base: Path):
        self.old = (base / "old-r5f").resolve()
        self.staging = (base / "staging").resolve()
        self.target = (base / "target").resolve()
        self.old.mkdir()
        self.staging.mkdir()

        self.old_files = {}
        for index, relative in enumerate(builder.OLD_REUSED_FILES):
            raw = f"old-{index:02d}:{relative}\n".encode()
            self._write(self.old / relative, raw)
            self.old_files[relative] = sha(raw)

        self.new_files = {}
        for index, relative in enumerate(builder.NEW_STAGED_FILES):
            raw = f"new-{index:02d}:{relative}\n".encode()
            self._write(self.staging / relative, raw)
            self.new_files[relative] = sha(raw)

        self.failed_raw = BUILDER_V1.read_bytes()
        self.active_raw = BUILDER_V2.read_bytes()
        self._write(self.staging / builder.FAILED_BUILDER_RELATIVE, self.failed_raw)
        self._write(self.staging / builder.BUILDER_RELATIVE, self.active_raw)

    @staticmethod
    def _write(path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o644)

    @contextmanager
    def patched(self):
        with ExitStack() as stack:
            for name, value in (
                ("OLD_R5F_SNAPSHOT", self.old),
                ("STAGING_ROOT", self.staging),
                ("TARGET_ROOT", self.target),
                ("OLD_REUSED_FILES", self.old_files),
                ("NEW_STAGED_FILES", self.new_files),
                ("FAILED_BUILDER_SHA256", sha(self.failed_raw)),
                ("FAILED_BUILDER_SIZE", len(self.failed_raw)),
            ):
                stack.enter_context(mock.patch.object(builder, name, value))
            stack.enter_context(auh_authority_view())
            yield

    def unseal_target(self) -> None:
        if not self.target.exists():
            return
        directories = [
            path for path in self.target.rglob("*")
            if path.is_dir() and not path.is_symlink()
        ]
        for path in sorted(directories, key=lambda item: len(item.parts)):
            path.chmod(0o755)
        self.target.chmod(0o755)


class SnapshotBuilderV2Tests(unittest.TestCase):
    def test_real_temp_old_0644_builds_exact24_sealed_target(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as value:
            fixture = BuildFixture(Path(value))
            try:
                with fixture.patched():
                    manifest = builder.build_snapshot()
                self.assertEqual(manifest["file_count"], 23)
                self.assertEqual(manifest["physical_file_count_including_manifest"], 24)
                leaves = [
                    path for path in fixture.target.rglob("*") if path.is_file()
                ]
                directories = [
                    fixture.target,
                    *(path for path in fixture.target.rglob("*") if path.is_dir()),
                ]
                self.assertEqual(len(leaves), 24)
                self.assertTrue(all(
                    stat.S_IMODE(path.lstat().st_mode) == 0o444 for path in leaves
                ))
                self.assertTrue(all(
                    stat.S_IMODE(path.lstat().st_mode) == 0o555
                    for path in directories
                ))
                manifest_path = fixture.target / builder.SNAPSHOT_MANIFEST
                self.assertEqual(manifest_path.read_bytes()[-1:], b"\n")
            finally:
                fixture.unseal_target()

    def test_old_0444_and_0600_reject_before_target_mkdir(self) -> None:
        for mode in (0o444, 0o600):
            with self.subTest(mode=oct(mode)), \
                 tempfile.TemporaryDirectory(dir="/tmp") as value:
                fixture = BuildFixture(Path(value))
                (fixture.old / next(iter(fixture.old_files))).chmod(mode)
                with fixture.patched(), self.assertRaisesRegex(
                    builder.Exact5SnapshotError, "source authority differs"
                ):
                    builder.build_snapshot()
                self.assertFalse(fixture.target.exists())

    def test_staging_exact9_requires_exact_0644(self) -> None:
        for role in ("payload", "failed-builder", "active-builder"):
            for mode in (0o444, 0o600):
                with self.subTest(role=role, mode=oct(mode)), \
                     tempfile.TemporaryDirectory(dir="/tmp") as value:
                    fixture = BuildFixture(Path(value))
                    if role == "payload":
                        relative = next(iter(fixture.new_files))
                    elif role == "failed-builder":
                        relative = builder.FAILED_BUILDER_RELATIVE
                    else:
                        relative = builder.BUILDER_RELATIVE
                    (fixture.staging / relative).chmod(mode)
                    with fixture.patched(), self.assertRaisesRegex(
                        builder.Exact5SnapshotError, "source authority differs"
                    ):
                        builder.build_snapshot()
                    self.assertFalse(fixture.target.exists())

    def test_failed_lineage_and_old_controllers_are_byte_preserved(self) -> None:
        self.assertEqual(sha(BUILDER_V1.read_bytes()), V1_BUILDER_SHA)
        self.assertEqual(BUILDER_V1.stat().st_size, 19690)
        self.assertEqual(sha(CONTROLLER_V1.read_bytes()), V1_CONTROLLER_SHA)
        self.assertEqual(sha(CONTROLLER_V2.read_bytes()), V2_CONTROLLER_SHA)
        self.assertEqual(len(builder.NEW_STAGED_FILES) + 2, 9)
        self.assertEqual(
            builder.FAILED_BUILDER_RELATIVE,
            "methods/bernini_action_editing/tools/"
            "build_case01_source_bone_exact5_source_snapshot_v1.py",
        )
        self.assertEqual(
            builder.BUILDER_RELATIVE,
            "methods/bernini_action_editing/tools/"
            "build_case01_source_bone_exact5_source_snapshot_v2.py",
        )

    def test_controller_v3_pins_compile_bash_n_and_real_argv(self) -> None:
        builder_raw = BUILDER_V2.read_bytes()
        builder_sha = sha(builder_raw)
        controller = CONTROLLER_V3.read_text(encoding="utf-8")
        self.assertEqual(controller.count(builder_sha), 2)
        self.assertIn(
            f'"{builder_sha}",{len(builder_raw)},0o644,2012,2000', controller
        )
        self.assertIn("if len(sys.argv)!=8:", controller)
        self.assertNotIn(
            "stat.S_IMODE(os.fstat(srcfd).st_mode),2012,2000", controller
        )
        self.assertEqual(
            controller.count(
                "build_case01_source_bone_exact5_source_snapshot_v2.py"
            ),
            2,
        )
        self.assertIn(
            '"$ROOT_PYTHON_FD" "$BUILDER_FD" "$ROOT_PYTHON" "$BUILDER" \\\n'
            '  "$STAGING_ROOT" "$OLD_ROOT" "$TARGET_ROOT"',
            controller,
        )

        checked = subprocess.run(
            ["/bin/bash", "-n", str(CONTROLLER_V3)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())

        marker = "SNAPSHOT_BOOTSTRAP='"
        start = controller.index(marker) + len(marker)
        end = controller.index("'\nreadonly SNAPSHOT_BOOTSTRAP", start)
        bootstrap = controller[start:end]
        for optimize in (0, 2):
            compile(
                builder_raw.decode("utf-8"), BUILDER_V2.name, "exec",
                dont_inherit=True, optimize=optimize,
            )
            compile(
                bootstrap, CONTROLLER_V3.name + ":bootstrap", "exec",
                dont_inherit=True, optimize=optimize,
            )

        guard = 'if len(sys.argv)!=8: fail("snapshot bootstrap argv differs")'
        argv_probe = bootstrap.replace(
            guard,
            guard + '\nprint("\\n".join(sys.argv))\nraise SystemExit(0)',
            1,
        )
        actual = subprocess.run(
            [
                sys.executable, "-I", "-S", "-B", "-c", argv_probe,
                "101", "102", "/usr/bin/python3.10", "/staging/builder-v2.py",
                "/staging", "/old", "/target",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(actual.returncode, 0, actual.stderr.decode())
        self.assertEqual(
            actual.stdout.decode().splitlines(),
            [
                "-c", "101", "102", "/usr/bin/python3.10",
                "/staging/builder-v2.py", "/staging", "/old", "/target",
            ],
        )

    def test_controller_v4_is_exact_single_lf_runtime_eof_fix(self) -> None:
        v3_raw = CONTROLLER_V3.read_bytes()
        v4_raw = CONTROLLER_V4.read_bytes()
        self.assertEqual(sha(v3_raw), V3_CONTROLLER_SHA)
        self.assertEqual(sha(v4_raw), V4_CONTROLLER_SHA)
        self.assertEqual(len(v3_raw), 6837)
        self.assertEqual(len(v4_raw), 6836)
        self.assertEqual(v3_raw, v4_raw + b"\n")
        self.assertTrue(v4_raw.endswith(b"\n"))
        self.assertFalse(v4_raw.endswith(b"\n\n"))

        checked = subprocess.run(
            ["/bin/bash", "-n", str(CONTROLLER_V4)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr.decode())

        probe = (
            b'exec "$1" -I -S -B -c \'import sys; '
            b'value=sys.stdin.buffer.read(1); '
            b'print("EOF" if value==b"" else value.hex()); '
            b'raise SystemExit(0 if value==b"" else 41)\''
        )
        v3_tail = v3_raw[len(v3_raw.rstrip(b"\n")):]
        v4_tail = v4_raw[len(v4_raw.rstrip(b"\n")):]
        v3_actual = subprocess.run(
            ["/bin/bash", "-s", "--", sys.executable],
            input=probe + v3_tail,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        v4_actual = subprocess.run(
            ["/bin/bash", "-s", "--", sys.executable],
            input=probe + v4_tail,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(v3_actual.returncode, 41, v3_actual.stderr.decode())
        self.assertEqual(v3_actual.stdout, b"0a\n")
        self.assertEqual(v4_actual.returncode, 0, v4_actual.stderr.decode())
        self.assertEqual(v4_actual.stdout, b"EOF\n")


if __name__ == "__main__":
    unittest.main()

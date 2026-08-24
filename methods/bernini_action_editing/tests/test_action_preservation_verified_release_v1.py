from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import action_preservation_verified_release_v1 as runtime


TARGET = "train_self_generated_action_quotient_v1.py"
SHELL_TARGET = "scripts/auh_run_self_generated_action_preservation_v2.sh"
HELPER = "presv2_verified_test_helper.py"
STDLIB_CHILD_MODULE = "presv2_verified_test_stdlib_child"
FROZEN_DEPENDENCY_MODULE = "presv2_verified_test_frozen_dependency"
FIXTURE_PYTHON_MODULES = {
    Path(TARGET).stem,
    Path(HELPER).stem,
    STDLIB_CHILD_MODULE,
    FROZEN_DEPENDENCY_MODULE,
    "audit_self_generated_action_preservation_v2",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_writable(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    for current, directories, _ in os.walk(root, topdown=False):
        for name in directories:
            try:
                os.chmod(Path(current) / name, 0o700, follow_symlinks=False)
            except OSError:
                pass
        try:
            os.chmod(current, 0o700, follow_symlinks=False)
        except OSError:
            pass


class ReleaseFixture:
    def __init__(self, root: Path, *, target_source: bytes | None = None) -> None:
        self.root = root
        self.archive = root / "source.tar"
        self.manifest = root / "source.manifest.json"
        if target_source is None:
            target_source = (
                b"import os\nimport sys\nfrom pathlib import Path\n"
                b"import presv2_verified_test_helper as helper\n"
                b"Path(os.environ['PRESV2_VERIFIED_TEST_OUTPUT']).write_text("
                b"helper.VALUE + '|' + ','.join(sys.argv[1:]), encoding='utf-8')\n"
            )
        self.payloads = {
            "audit_self_generated_action_preservation_v2.py": b"VALUE = 'audit'\n",
            HELPER: b"VALUE = 'captured-helper'\n",
            SHELL_TARGET: b"printf '%s\\n' ORIGINAL_CAPTURED_SHELL\n",
            TARGET: target_source,
        }
        self.modes = {
            relative: 0o555 if relative.endswith(".sh") else 0o444
            for relative in self.payloads
        }
        self.rows = [
            {
                "path": relative,
                "mode": self.modes[relative],
                "size": len(self.payloads[relative]),
                "sha256": sha256(self.payloads[relative]),
            }
            for relative in sorted(self.payloads)
        ]
        self.revision = runtime.content_revision(self.rows)
        archive_raw = runtime.fixed_ustar_archive(self.rows, self.payloads)
        self.archive_sha = sha256(archive_raw)
        unsigned = {
            "schema_version": runtime.SCHEMA_VERSION,
            "release_generation": runtime.RELEASE_GENERATION,
            "member_root": runtime.MEMBER_ROOT,
            "archive_format": runtime.ARCHIVE_FORMAT,
            "file_count": len(self.rows),
            "exact_member_closure": True,
            "files": self.rows,
            "content_revision": self.revision,
            "allowed_entrypoints": [
                SHELL_TARGET,
                "audit_self_generated_action_preservation_v2.py",
                "auh_launch_self_generated_action_preservation_v2_four_holder_v1.sh",
            ],
            "authority": {
                "seed": 20260818,
                "experimental_training": True,
                "scientific_claim_authorized": False,
            },
            "component_sha256": {
                "trainer": sha256(self.payloads[TARGET]),
                "auditor": sha256(
                    self.payloads["audit_self_generated_action_preservation_v2.py"]
                ),
                "node_runner": sha256(self.payloads[SHELL_TARGET]),
            },
        }
        value = {**unsigned, "manifest_digest": runtime.object_sha256(unsigned)}
        manifest_raw = runtime.canonical_json_bytes(value) + b"\n"
        self.manifest_sha = sha256(manifest_raw)
        self.archive.write_bytes(archive_raw)
        self.manifest.write_bytes(manifest_raw)
        self.output = root / "materialized"

    def extract(self) -> None:
        runtime.extract_verified_release(
            archive=self.archive,
            expected_archive_sha256=self.archive_sha,
            manifest=self.manifest,
            expected_manifest_sha256=self.manifest_sha,
            expected_content_revision=self.revision,
            output_root=self.output,
        )


class ActionPreservationVerifiedReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.preexisting_fixture_modules = {
            name: sys.modules.pop(name)
            for name in FIXTURE_PYTHON_MODULES
            if name in sys.modules
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        try:
            make_writable(self.root)
            self.temporary.cleanup()
        finally:
            for name in FIXTURE_PYTHON_MODULES:
                sys.modules.pop(name, None)
            sys.modules.update(self.preexisting_fixture_modules)

    def test_contract_matches_self_generated_preservation_builder(self):
        self.assertEqual(
            runtime.SCHEMA_VERSION,
            "bernini-self-generated-action-preservation-v2-release-v1",
        )
        self.assertEqual(
            runtime.ARCHIVE_FORMAT,
            "fixed-ustar-ascii-zero-dev-sorted-owner0-mtime0-record10240-v1",
        )
        self.assertEqual(runtime.MEMBER_ROOT, "methods/bernini_action_editing")
        self.assertEqual(
            runtime.ALLOWED_PYTHON_TARGETS,
            {
                "train_self_generated_action_quotient_v1.py",
                "audit_self_generated_action_preservation_v2.py",
                "action_preservation_completion_publisher_v1.py",
            },
        )

    def test_fixed_ustar_header_has_ascii_zero_devices_and_exact_checksum(self):
        fixture = ReleaseFixture(self.root)
        row = fixture.rows[0]
        name = f"{runtime.MEMBER_ROOT}/{row['path']}"
        archive_raw = fixture.archive.read_bytes()
        header = archive_raw[: runtime.FIXED_USTAR_BLOCK_SIZE]
        self.assertEqual(
            header,
            runtime.fixed_ustar_header(
                name, size=row["size"], mode=row["mode"]
            ),
        )
        self.assertEqual(header[329:345], b"0000000\0" * 2)
        self.assertEqual(header[148:156][-2:], b"\0 ")
        checksum_header = bytearray(header)
        declared_checksum = int(checksum_header[148:154], 8)
        checksum_header[148:156] = b" " * 8
        self.assertEqual(declared_checksum, sum(checksum_header))
        self.assertEqual(len(archive_raw) % runtime.FIXED_USTAR_RECORD_SIZE, 0)

    def test_null_device_fields_from_host_tarinfo_are_rejected(self):
        fixture = ReleaseFixture(self.root)
        hostile = bytearray(fixture.archive.read_bytes())
        offset = 0
        for row in fixture.rows:
            header = bytearray(
                hostile[offset : offset + runtime.FIXED_USTAR_BLOCK_SIZE]
            )
            header[329:345] = b"\0" * 16
            header[148:156] = b" " * 8
            checksum = sum(header)
            header[148:156] = f"{checksum:06o}\0 ".encode("ascii")
            hostile[offset : offset + runtime.FIXED_USTAR_BLOCK_SIZE] = header
            blocks = (
                row["size"] + runtime.FIXED_USTAR_BLOCK_SIZE - 1
            ) // runtime.FIXED_USTAR_BLOCK_SIZE
            offset += runtime.FIXED_USTAR_BLOCK_SIZE * (1 + blocks)

        with tarfile.open(fileobj=io.BytesIO(hostile), mode="r:") as bundle:
            self.assertEqual(len(bundle.getmembers()), len(fixture.rows))
        manifest = runtime.capture_manifest(
            fixture.manifest,
            expected_manifest_sha256=fixture.manifest_sha,
            expected_content_revision=fixture.revision,
        )[0]
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "canonical USTAR"
        ):
            runtime.verify_archive_snapshot(bytes(hostile), manifest)

    def test_extract_seals_exact_tree_and_python_runs_captured_modules(self):
        fixture = ReleaseFixture(self.root)
        fixture.extract()
        expected_files = {
            f"{runtime.MEMBER_ROOT}/{relative}" for relative in fixture.payloads
        }
        actual_files = {
            path.relative_to(fixture.output).as_posix()
            for path in fixture.output.rglob("*")
            if path.is_file()
        }
        self.assertEqual(actual_files, expected_files)
        for path in fixture.output.rglob("*"):
            expected_mode = 0o555 if path.is_dir() else fixture.modes[
                path.relative_to(fixture.output / runtime.MEMBER_ROOT).as_posix()
            ]
            self.assertEqual(stat.S_IMODE(path.lstat().st_mode), expected_mode)
        self.assertEqual(stat.S_IMODE(fixture.output.lstat().st_mode), 0o555)
        output = self.root / "python-result.txt"
        old_argv = list(sys.argv)
        with mock.patch.dict(
            os.environ, {"PRESV2_VERIFIED_TEST_OUTPUT": str(output)}, clear=False
        ):
            result = runtime.verified_python_run(
                release_root=fixture.output,
                manifest=fixture.manifest,
                expected_manifest_sha256=fixture.manifest_sha,
                expected_content_revision=fixture.revision,
                target=TARGET,
                target_arguments=["--", "alpha", "beta"],
            )
        self.assertEqual(result, 0)
        self.assertEqual(output.read_text(encoding="utf-8"), "captured-helper|alpha,beta")
        self.assertEqual(sys.argv, old_argv)
        self.assertNotIn("presv2_verified_test_helper", sys.modules)

    def test_sys_path_filter_keeps_cwd_descendants_outside_release(self):
        home = self.root / "home"
        stdlib = home / "conda" / "lib" / "python3.12"
        dynload = stdlib / "lib-dynload"
        site_packages = stdlib / "site-packages"
        release = home / "release"
        release_child = release / "methods" / "bernini_action_editing"
        outside = self.root / "outside"
        for directory in (dynload, site_packages, release_child, outside):
            directory.mkdir(parents=True, exist_ok=True)

        original = [
            "",
            str(home),
            ".",
            str(stdlib),
            str(dynload),
            str(site_packages),
            "conda/lib/python3.12/lib-dynload",
            str(release),
            str(release_child),
            str(outside),
        ]
        expected = [
            str(stdlib),
            str(dynload),
            str(site_packages),
            "conda/lib/python3.12/lib-dynload",
            str(outside),
        ]
        with mock.patch.object(runtime.Path, "cwd", return_value=home), mock.patch.object(
            runtime.sys, "path", list(original)
        ):
            runtime._strip_forbidden_sys_path([release])
            self.assertEqual(runtime.sys.path, expected)

    def test_verified_run_from_home_keeps_descendant_stdlib_path(self):
        home = self.root / "home"
        fixture_root = home / "fixture"
        dynload = home / "conda" / "lib" / "python3.12" / "lib-dynload"
        fixture_root.mkdir(parents=True)
        dynload.mkdir(parents=True)
        (dynload / f"{STDLIB_CHILD_MODULE}.py").write_text(
            "VALUE = 'preserved-stdlib-child'\n", encoding="utf-8"
        )
        target_source = (
            b"import os\n"
            + f"import {STDLIB_CHILD_MODULE} as stdlib_child\n".encode("ascii")
            + b"with open(os.environ['PRESV2_VERIFIED_TEST_OUTPUT'], 'w', encoding='utf-8') as handle:\n"
            + b" handle.write(stdlib_child.VALUE)\n"
        )
        fixture = ReleaseFixture(fixture_root, target_source=target_source)
        fixture.extract()
        output = home / "integration-result.txt"
        test_path = [
            "",
            str(home),
            str(dynload),
            str(fixture.output),
            str(fixture.output / runtime.MEMBER_ROOT),
        ]
        old_cwd = Path.cwd()
        try:
            os.chdir(home)
            with mock.patch.object(runtime.sys, "path", list(test_path)), mock.patch.dict(
                os.environ,
                {"PRESV2_VERIFIED_TEST_OUTPUT": str(output)},
                clear=False,
            ):
                result = runtime.verified_python_run(
                    release_root=fixture.output,
                    manifest=fixture.manifest,
                    expected_manifest_sha256=fixture.manifest_sha,
                    expected_content_revision=fixture.revision,
                    target=TARGET,
                    target_arguments=[],
                )
                self.assertEqual(runtime.sys.path, test_path)
        finally:
            os.chdir(old_cwd)

        self.assertEqual(result, 0)
        self.assertEqual(
            output.read_text(encoding="utf-8"), "preserved-stdlib-child"
        )

    def test_frozen_site_is_added_only_after_capture_without_site_processing(self):
        site_packages = self.root / "frozen-site-packages"
        site_packages.mkdir()
        dependency = site_packages / f"{FROZEN_DEPENDENCY_MODULE}.py"
        dependency.write_text("VALUE = 'explicit-site-import'\n", encoding="utf-8")
        pth_sentinel = self.root / "pth-ran.txt"
        custom_sentinel = self.root / "sitecustomize-ran.txt"
        (site_packages / "hostile.pth").write_text(
            "import pathlib; pathlib.Path(" + repr(str(pth_sentinel))
            + ").write_text('PTH_RAN', encoding='utf-8')\n",
            encoding="utf-8",
        )
        (site_packages / "sitecustomize.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(custom_sentinel)!r}).write_text('CUSTOM_RAN', encoding='utf-8')\n",
            encoding="utf-8",
        )
        target_source = (
            b"import os\nfrom pathlib import Path\n"
            + f"import {FROZEN_DEPENDENCY_MODULE} as dependency\n".encode("ascii")
            + b"Path(os.environ['PRESV2_VERIFIED_TEST_OUTPUT']).write_text("
            + b"dependency.VALUE, encoding='utf-8')\n"
        )
        fixture_root = self.root / "post-capture-site-fixture"
        fixture_root.mkdir()
        fixture = ReleaseFixture(fixture_root, target_source=target_source)
        fixture.extract()
        output = self.root / "post-capture-site-result.txt"
        original_capture = runtime.capture_materialized_release
        capture_observed = False

        def capture_before_site(*args, **kwargs):
            nonlocal capture_observed
            self.assertNotIn(str(site_packages), runtime.sys.path)
            captured = original_capture(*args, **kwargs)
            self.assertNotIn(str(site_packages), runtime.sys.path)
            capture_observed = True
            return captured

        try:
            with mock.patch.object(
                runtime, "FROZEN_SITE_PACKAGES", site_packages
            ), mock.patch.object(
                runtime, "FROZEN_SITE_PACKAGES_LITERAL", site_packages.as_posix()
            ), mock.patch.object(
                runtime,
                "capture_materialized_release",
                side_effect=capture_before_site,
            ), mock.patch.dict(
                os.environ,
                {"PRESV2_VERIFIED_TEST_OUTPUT": str(output)},
                clear=False,
            ):
                runtime.verified_python_run(
                    release_root=fixture.output,
                    manifest=fixture.manifest,
                    expected_manifest_sha256=fixture.manifest_sha,
                    expected_content_revision=fixture.revision,
                    target=TARGET,
                    target_arguments=[],
                )
        finally:
            sys.modules.pop(FROZEN_DEPENDENCY_MODULE, None)

        self.assertTrue(capture_observed)
        self.assertEqual(output.read_text(encoding="utf-8"), "explicit-site-import")
        self.assertFalse(pth_sentinel.exists())
        self.assertFalse(custom_sentinel.exists())
        self.assertNotIn(str(site_packages), sys.path)

    def test_archive_and_manifest_tampering_fail_before_output_creation(self):
        fixture = ReleaseFixture(self.root)
        archive_raw = bytearray(fixture.archive.read_bytes())
        archive_raw[700] ^= 1
        fixture.archive.write_bytes(archive_raw)
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "archive SHA-256 differs"
        ):
            fixture.extract()
        self.assertFalse(fixture.output.exists())

        second_root = self.root / "second"
        second_root.mkdir()
        second = ReleaseFixture(second_root)
        second.manifest.write_bytes(second.manifest.read_bytes() + b" ")
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "manifest SHA-256 differs"
        ):
            second.extract()
        self.assertFalse(second.output.exists())

    def test_extra_and_symlink_materializations_are_rejected(self):
        fixture = ReleaseFixture(self.root)
        fixture.extract()
        method_root = fixture.output / runtime.MEMBER_ROOT
        os.chmod(method_root, 0o755)
        (method_root / "unsigned_extra.py").write_text("VALUE = 1\n", encoding="utf-8")
        os.chmod(method_root, 0o555)
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "extras"
        ):
            runtime.capture_materialized_release(
                fixture.output,
                runtime.capture_manifest(
                    fixture.manifest,
                    expected_manifest_sha256=fixture.manifest_sha,
                    expected_content_revision=fixture.revision,
                )[0],
            )

        second_root = self.root / "second"
        second_root.mkdir()
        second = ReleaseFixture(second_root)
        second.extract()
        second_method = second.output / runtime.MEMBER_ROOT
        helper = second_method / HELPER
        os.chmod(second_method, 0o755)
        helper.unlink()
        helper.symlink_to(TARGET)
        os.chmod(second_method, 0o555)
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "symlink"
        ):
            runtime.capture_materialized_release(
                second.output,
                runtime.capture_manifest(
                    second.manifest,
                    expected_manifest_sha256=second.manifest_sha,
                    expected_content_revision=second.revision,
                )[0],
            )

    def test_same_fd_double_read_rejects_named_path_swap(self):
        victim = self.root / "victim.bin"
        victim.write_bytes(b"trusted-bytes")
        expected = sha256(victim.read_bytes())
        original_read = runtime.os.read
        swapped = False

        def hostile_read(descriptor: int, count: int) -> bytes:
            nonlocal swapped
            raw = original_read(descriptor, count)
            if raw == b"" and not swapped:
                swapped = True
                victim.rename(self.root / "retained-old-inode.bin")
                victim.write_bytes(b"attacker-replacement")
            return raw

        with mock.patch.object(runtime.os, "read", side_effect=hostile_read):
            with self.assertRaisesRegex(
                runtime.ActionPreservationVerifiedReleaseError, "identity changed"
            ):
                runtime._stable_capture(
                    victim, label="path-swap hostile", expected_sha256=expected
                )

    def test_unknown_release_root_module_injected_after_capture_cannot_import(self):
        target_source = (
            b"import os\nimport sys\nfrom pathlib import Path\n"
            b"sys.path.insert(0, str(Path(__file__).parent))\n"
            b"try:\n import unsigned_release_injection\n"
            b"except ModuleNotFoundError:\n"
            b" Path(os.environ['PRESV2_VERIFIED_TEST_OUTPUT']).write_text('blocked', encoding='utf-8')\n"
            b"else:\n raise RuntimeError('unsigned injection imported')\n"
        )
        fixture = ReleaseFixture(self.root, target_source=target_source)
        fixture.extract()
        output = self.root / "injection-result.txt"
        sentinel = self.root / "unsigned-sentinel.txt"
        method_root = fixture.output / runtime.MEMBER_ROOT
        original_compile = builtins.compile
        injected = False

        def hostile_compile(*args, **kwargs):
            nonlocal injected
            if not injected:
                injected = True
                os.chmod(method_root, 0o755)
                (method_root / "unsigned_release_injection.py").write_text(
                    "from pathlib import Path\n"
                    f"Path({str(sentinel)!r}).write_text('IMPORTED', encoding='utf-8')\n",
                    encoding="utf-8",
                )
                os.chmod(method_root, 0o555)
            return original_compile(*args, **kwargs)

        with mock.patch.dict(
            os.environ, {"PRESV2_VERIFIED_TEST_OUTPUT": str(output)}, clear=False
        ), mock.patch.object(builtins, "compile", side_effect=hostile_compile):
            runtime.verified_python_run(
                release_root=fixture.output,
                manifest=fixture.manifest,
                expected_manifest_sha256=fixture.manifest_sha,
                expected_content_revision=fixture.revision,
                target=TARGET,
                target_arguments=[],
            )
        self.assertEqual(output.read_text(encoding="utf-8"), "blocked")
        self.assertFalse(sentinel.exists())

    def test_verified_shell_uses_captured_source_after_target_path_swap(self):
        fixture = ReleaseFixture(self.root)
        fixture.extract()
        target_path = fixture.output / runtime.MEMBER_ROOT / SHELL_TARGET
        original_source = fixture.payloads[SHELL_TARGET].decode("utf-8")

        def swap_then_hold(*args, **kwargs):
            scripts = target_path.parent
            os.chmod(scripts, 0o755)
            target_path.unlink()
            target_path.write_text("echo SWAPPED_PATH_BYTES\n", encoding="utf-8")
            os.chmod(target_path, 0o555)
            os.chmod(scripts, 0o555)
            return os.open("/dev/null", os.O_RDONLY)

        class ExecObserved(Exception):
            pass

        def observe_exec(descriptor, argv, environment):
            self.assertIsInstance(descriptor, int)
            self.assertEqual(argv[:5], ["/usr/bin/bash", "--noprofile", "--norc", "-p", "-c"])
            self.assertEqual(argv[5], original_source)
            self.assertNotIn("SWAPPED_PATH_BYTES", argv[5])
            self.assertEqual(argv[7:], ["one", "two"])
            self.assertNotIn("BASH_ENV", environment)
            raise ExecObserved

        with mock.patch.object(runtime, "EXECVE_SUPPORTS_FD", True), mock.patch.object(
            runtime, "_stable_executable_fd", side_effect=swap_then_hold
        ), mock.patch.object(runtime.os, "execve", side_effect=observe_exec):
            with self.assertRaises(ExecObserved):
                runtime.verified_shell_run(
                    release_root=fixture.output,
                    manifest=fixture.manifest,
                    expected_manifest_sha256=fixture.manifest_sha,
                    expected_content_revision=fixture.revision,
                    target=SHELL_TARGET,
                    target_arguments=["--", "one", "two"],
                    expected_bash_sha256="a" * 64,
                    expected_bash_size=1,
                )

    def test_unknown_shell_target_is_rejected(self):
        fixture = ReleaseFixture(self.root)
        fixture.extract()
        with self.assertRaisesRegex(
            runtime.ActionPreservationVerifiedReleaseError, "shell target is not allowed"
        ):
            runtime.verified_shell_run(
                release_root=fixture.output,
                manifest=fixture.manifest,
                expected_manifest_sha256=fixture.manifest_sha,
                expected_content_revision=fixture.revision,
                target="scripts/unsigned.sh",
                target_arguments=[],
                expected_bash_sha256="a" * 64,
                expected_bash_size=1,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import full644_exploratory_matched_r5d_cpu_consumption_probe_v1 as probe


class _Authority:
    object_sha256 = staticmethod(probe.object_sha256)


class R5DCPUConsumptionProbeUnitTests(unittest.TestCase):
    def test_frozen_eight_source_pins_match_repository_bytes(self) -> None:
        self.assertEqual(len(probe.SOURCE_SPECS), 8)
        for role, (relative, expected_sha, expected_size) in probe.SOURCE_SPECS.items():
            with self.subTest(role=role):
                raw = (ROOT / relative).read_bytes()
                self.assertEqual(len(raw), expected_size)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_sha)

    def test_source_closure_digest_is_exact(self) -> None:
        rows = [
            {
                "role": role,
                "relative_path": relative,
                "sha256": digest,
                "size": size,
                "mode": 0o444,
                "nlink": 1,
            }
            for role, (relative, digest, size) in sorted(probe.SOURCE_SPECS.items())
        ]
        self.assertEqual(probe.object_sha256(rows), probe.SOURCE_CLOSURE_DIGEST)

    def test_receipt_is_canonical_0400_nlink1_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "receipt.json"
            payload = {"schema_version": probe.SCHEMA, "status": "PASS"}
            row = probe.write_receipt(path, payload)
            raw = path.read_bytes()
            value = json.loads(raw)
            unsigned = dict(value)
            claim = unsigned.pop("receipt_digest")
            self.assertEqual(claim, probe.object_sha256(unsigned))
            self.assertEqual(raw, probe.canonical_bytes(value) + b"\n")
            self.assertEqual(row["sha256"], hashlib.sha256(raw).hexdigest())
            info = path.stat()
            self.assertEqual(stat.S_IMODE(info.st_mode), 0o400)
            self.assertEqual(info.st_nlink, 1)
            with self.assertRaises((FileExistsError, probe.R5DCPUConsumptionProbeError)):
                probe.write_receipt(path, payload)

    def test_production_receipt_commit_exits_zero_at_0400(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "committed.json"
            payload = {"schema_version": probe.SCHEMA, "status": "PASS"}
            receipt, reference = probe.prepare_receipt(path, payload)
            environment = dict(os.environ)
            environment["PROBE_RECEIPT"] = probe.canonical_bytes(receipt).decode()
            environment["PROBE_REFERENCE"] = probe.canonical_bytes(reference).decode()
            body = (
                "import json,os,sys\n"
                f"sys.path.insert(0,{str(ROOT)!r})\n"
                "import full644_exploratory_matched_r5d_cpu_consumption_probe_v1 as probe\n"
                f"probe.commit_receipt_and_exit({str(path)!r},json.loads(os.environ['PROBE_RECEIPT']),json.loads(os.environ['PROBE_REFERENCE']))\n"
            )
            completed = subprocess.run(
                [sys.executable, *(["-O"] if sys.flags.optimize else []), "-c", body],
                check=False,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stderr, b"")
            self.assertEqual(completed.stdout, probe.canonical_bytes(reference) + b"\n")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(path.read_bytes(), probe.canonical_bytes(receipt) + b"\n")

    def test_binding_row_replacement_rehashes_and_preserves_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            self.addCleanup(os.close, descriptor)
            binding = {
                "fd_rows": [
                    {
                        "fd": 9,
                        "scope": "adapter",
                        "role": "namespace_root",
                        "relative_path": ".",
                        "source_path": "/old",
                        "identity": probe.stat_identity(os.fstat(descriptor)),
                    }
                ],
                "fd_count": 1,
                "fd_rows_digest": "x",
                "fd_binding_digest": "y",
            }
            result = probe._replace_binding_row(
                _Authority,
                binding,
                scope="adapter",
                role="namespace_root",
                relative_path=".",
                descriptor=descriptor,
                source_path=root,
            )
            self.assertEqual(result["fd_rows"][0]["fd"], descriptor)
            self.assertEqual(result["fd_rows"][0]["source_path"], str(root))
            self.assertEqual(
                result["fd_rows_digest"], probe.object_sha256(result["fd_rows"])
            )
            unsigned = dict(result)
            claim = unsigned.pop("fd_binding_digest")
            self.assertEqual(claim, probe.object_sha256(unsigned))

    def test_public_cli_requires_site_and_production_roots(self) -> None:
        with self.assertRaises(SystemExit):
            probe.build_parser().parse_args(
                [
                    "--methods-root", "/methods",
                    "--work-root", "/work",
                    "--receipt", "/work/receipt.json",
                    "--probe-sha256", "0" * 64,
                ]
            )
        parsed = probe.build_parser().parse_args(
            [
                "--methods-root", "/methods",
                "--site-packages-root", "/site",
                "--work-root", "/work",
                "--receipt", "/work/receipt.json",
                "--probe-sha256", "0" * 64,
            ]
        )
        self.assertEqual(parsed.site_packages_root, "/site")

    def test_retained_probe_fd_is_byte_stable_and_named_replacement_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "probe.py"
            raw = Path(probe.__file__).read_bytes()
            path.write_bytes(raw)
            path.chmod(0o444)
            _, authority = probe.open_probe_code_authority(
                path, hashlib.sha256(raw).hexdigest()
            )
            try:
                probe._replay_parent_probe_code(authority)
                path.unlink()
                path.write_bytes(b"hostile replacement\n")
                path.chmod(0o444)
                self.assertEqual(
                    probe._pread_sha256(authority["fd"], authority["size"]),
                    authority["sha256"],
                )
                with self.assertRaises(probe.R5DCPUConsumptionProbeError):
                    probe._replay_parent_probe_code(authority)
            finally:
                os.close(authority["fd"])

    def test_child_command_executes_exact_parent_captured_source(self) -> None:
        raw = Path(probe.__file__).read_bytes()
        source = raw.decode("utf-8", "strict")
        digest = hashlib.sha256(raw).hexdigest()
        for optimize in (0, 1):
            with self.subTest(optimize=optimize):
                command = probe.build_child_command(source, digest, optimize)
                self.assertEqual(command[0], "/proc/self/exe")
                self.assertEqual(command[-3], "-c")
                self.assertEqual(command[-1], probe.CHILD_TOKEN)
                self.assertNotIn("/proc/self/fd/", command)
                self.assertFalse(
                    any(value.startswith("/proc/self/fd/") for value in command)
                )
                self.assertEqual(
                    hashlib.sha256(command[-2].encode("utf-8")).hexdigest(),
                    digest,
                )

    def test_shared_private_parent_number_is_role_bound_and_allowed(self) -> None:
        class Holder:
            def __init__(self, descriptor: int) -> None:
                self.capture_receipt = {
                    "private_parent": {"authority_fd": descriptor}
                }

        roles = probe._private_parent_roles(Holder(45), Holder(45))
        self.assertEqual(roles, {"model": 45, "adapter": 45})
        self.assertEqual(sorted(set(roles.values())), [45])
        full_binding = {
            "adapter_capture_digest": "1" * 64,
            "fd_rows": [{"scope": "model"}, {"scope": "adapter"}],
        }
        self.assertEqual(
            probe._validate_private_parent_roles(
                full_binding, {"model": 45, "adapter": 45}, "full"
            ),
            {"model": 45, "adapter": 45},
        )
        with self.assertRaises(probe.R5DCPUConsumptionProbeError):
            probe._validate_private_parent_roles(
                full_binding, {"model": 45}, "full"
            )
        with self.assertRaises(probe.R5DCPUConsumptionProbeError):
            probe._validate_private_parent_roles(
                {"adapter_capture_digest": None, "fd_rows": [{"scope": "model"}]},
                {"model": 45, "adapter": 45},
                "base",
            )

    def test_stderr_classifier_is_closed(self) -> None:
        self.assertEqual(probe.classify_child_stderr(b"")["kind"], "empty")
        allowed = (
            b"Unbundle Objects Error: '/tmp/comgr-a1/output/"
            b"hipfatbin-hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-.o': "
            b"Invalid argument\n"
        )
        self.assertEqual(
            probe.classify_child_stderr(allowed)["kind"],
            "rocm-comgr-unbundle-invalid-argument",
        )
        for hostile in (
            b"Traceback (most recent call last):\n",
            allowed + b"RuntimeError: failed\n",
            b"arbitrary warning\n",
        ):
            with self.subTest(hostile=hostile), self.assertRaises(
                probe.R5DCPUConsumptionProbeError
            ):
                probe.classify_child_stderr(hostile)

    def test_isolated_eight_source_module_gate_keeps_torch_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            methods = root / "methods"
            methods.mkdir(mode=0o700)
            for relative, _, _ in probe.SOURCE_SPECS.values():
                target = methods / relative
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
                target.chmod(0o444)
            for directory in sorted(
                {methods, *(path.parent for path in methods.rglob("*.py"))},
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            site_candidates = [
                Path(value).resolve(strict=True)
                for value in sys.path
                if value and Path(value).name in {"site-packages", "dist-packages"}
                and Path(value).is_dir()
            ]
            self.assertTrue(site_candidates)
            source = Path(probe.__file__).resolve(strict=True)
            body = (
                "import importlib.util,json,pathlib,sys\n"
                f"spec=importlib.util.spec_from_file_location('probe_under_test',{str(source)!r})\n"
                "module=importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                f"r5d,authority,row=module.load_production_sources(pathlib.Path({str(methods)!r}),pathlib.Path({str(site_candidates[0])!r}))\n"
                "print(json.dumps({'closure':row['source_closure_digest'],'torch':('torch' in sys.modules),'module':r5d.__name__},sort_keys=True,separators=(',',':')))\n"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    *(["-O"] if sys.flags.optimize else []),
                    "-c",
                    body,
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stderr, b"")
            observed = json.loads(completed.stdout)
            self.assertEqual(
                observed,
                {
                    "closure": probe.SOURCE_CLOSURE_DIGEST,
                    "module": probe.R5D_MODULE_NAME,
                    "torch": False,
                },
            )


@unittest.skipUnless(
    sys.platform == "linux" and Path("/proc/self/fd").is_dir(),
    "real production-mode FD consumption requires Linux /proc/self/fd",
)
class R5DCPUConsumptionProbeLinuxIntegrationTests(unittest.TestCase):
    def test_base_full_and_four_hostiles_through_real_exec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            methods = root / "production-methods"
            diagnostic = root / "diagnostic"
            work = root / "work"
            methods.mkdir(mode=0o700)
            diagnostic.mkdir(mode=0o700)
            work.mkdir(mode=0o700)
            for relative, _, _ in probe.SOURCE_SPECS.values():
                source = ROOT / relative
                target = methods / relative
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                target.chmod(0o444)
            for directory in sorted(
                {methods, *(path.parent for path in methods.rglob("*.py"))},
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            probe_source = Path(probe.__file__).resolve(strict=True)
            probe_copy = diagnostic / probe_source.name
            shutil.copyfile(probe_source, probe_copy)
            probe_copy.chmod(0o444)
            diagnostic.chmod(0o555)
            site_candidates = [
                Path(value).resolve(strict=True)
                for value in sys.path
                if value and Path(value).name in {"site-packages", "dist-packages"}
                and Path(value).is_dir()
            ]
            self.assertTrue(site_candidates)
            receipt = work / "receipt.json"
            command = [
                sys.executable,
                "-I",
                "-S",
                "-B",
                *(["-O"] if sys.flags.optimize else []),
                str(probe_copy),
                "--methods-root", str(methods),
                "--site-packages-root", str(site_candidates[0]),
                "--work-root", str(work),
                "--receipt", str(receipt),
                "--probe-sha256", hashlib.sha256(probe_copy.read_bytes()).hexdigest(),
            ]
            completed = subprocess.run(
                command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertEqual(completed.stderr, b"")
            reference = json.loads(completed.stdout)
            self.assertEqual(completed.stdout, probe.canonical_bytes(reference) + b"\n")
            value = json.loads(receipt.read_bytes())
            self.assertEqual(value["schema_version"], probe.SCHEMA)
            self.assertEqual(value["status"], "PASS")
            self.assertEqual(value["summary"]["success_count"], 2)
            self.assertEqual(value["summary"]["hostile_rejection_count"], 4)
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o400)


if __name__ == "__main__":
    unittest.main()

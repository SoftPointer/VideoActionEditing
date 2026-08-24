from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    METHOD_ROOT
    / "scripts"
    / "auh_launch_self_generated_action_preservation_v2_four_holder_v1.sh"
)
NODE_RUNNER = (
    METHOD_ROOT / "scripts" / "auh_run_self_generated_action_preservation_v2.sh"
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def heredoc(source: str, variable: str) -> str:
    pattern = re.compile(
        rf"(?ms)^{re.escape(variable)}=\"\"\n"
        rf"if ! IFS= read -r -d '' {re.escape(variable)} <<'PY'\n"
        r"(?P<body>.*?)\nPY\nthen\n",
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(f"missing {variable} bootstrap heredoc")
    return match.group("body")


def array_body(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}=\(\n(?P<body>.*?)^\)\n", source
    )
    if match is None:
        raise AssertionError(f"missing {name} array")
    return match.group("body")


def function_body(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(name)}\(\) \{{\n(?P<body>.*?)^\}}\n", source
    )
    if match is None:
        raise AssertionError(f"missing {name} function")
    return match.group("body")


class ActionPreservationV2VerifiedRuntimeWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = CONTROLLER.read_text(encoding="utf-8")
        cls.node = NODE_RUNNER.read_text(encoding="utf-8")

    def test_shells_parse_and_never_use_shell_tar(self) -> None:
        for script in (CONTROLLER, NODE_RUNNER):
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    ["/bin/bash", "-n", str(script)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                source = script.read_text(encoding="utf-8")
                self.assertNotIn("/usr/bin/tar", source)
                self.assertNotRegex(
                    source,
                    r"(?m)^[ \t]*(?:exec[ \t]+)?(?:/[^ \t]+/)?tar[ \t]+",
                )

    def test_controller_pins_exact_ssh_binary_identity(self) -> None:
        expected = "3a9c5d143150f0b2816ab1a5a7c58a9f970280b061f617abee54d2834a498b53"
        match = re.search(r"(?m)^readonly ssh_sha=([0-9a-f]+)$", self.controller)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), expected)
        self.assertEqual(len(match.group(1)), 64)
        self.assertIn('"/usr/bin/ssh:${ssh_sha}:846888"', self.controller)

    def test_controller_srun_dispatches_only_the_captured_node_shell(self) -> None:
        command = array_body(self.controller, "node_shell_command")
        self.assertIn('"${root_python}" -I -S -B -c "${runtime_bootstrap_source}"', command)
        self.assertIn("verified-shell-run", command)
        self.assertIn('--release-root "${materialized}"', command)
        self.assertIn('--manifest "${release_manifest}"', command)
        self.assertIn(
            "--expected-manifest-sha256 \"${release_manifest_sha}\"", command
        )
        self.assertIn(
            "--target scripts/auh_run_self_generated_action_preservation_v2.sh",
            command,
        )
        self.assertIn('--expected-bash-sha256 "${bash_sha}"', command)

        # There are two syntactic srun sites: the full cache and the arm loop.
        # Both end at the same captured node-shell argv array.
        srun_regions = self.controller.split("/usr/bin/srun ")[1:]
        self.assertEqual(len(srun_regions), 2)
        for region in srun_regions:
            self.assertIn('"${node_shell_command[@]}"', region)

        # These materialized path variables are topology probes only.  An
        # additional expansion would reintroduce path-based execution.
        topology_probe = self.controller[
            self.controller.index('[[ -x "${node_runner}"') :
            self.controller.index(
                '|| fail "materialized entrypoint differs"'
            )
        ]
        for variable in ("node_runner", "auditor", "completion_publisher"):
            token = f'"${{{variable}}}"'
            self.assertEqual(self.controller.count(token), 1, variable)
            self.assertIn(token, topology_probe)

    def test_node_torchrun_ranks_bootstrap_the_captured_trainer(self) -> None:
        expected_chain = re.compile(
            r"(?s)--standalone --nproc_per_node=4\s*\\\n"
            r"\s*--no-python \"\$\{root_python\}\" -I -S -B -c "
            r"\"\$\{runtime_bootstrap_source\}\"\s*\\\n"
            r".*?\"\$\{source_revision\}\" frozen \"\$\{python_bin\}\" "
            r"\"\$\{python_sha\}\" 31490256\s*\\\n"
            r"\s*verified-run --release-root \"\$\{scratch\}/source\" "
            r"--manifest \"\$\{release_manifest\}\"\s*\\\n"
            r".*?--target train_self_generated_action_quotient_v1\.py --",
        )
        self.assertRegex(self.node, expected_chain)
        self.assertEqual(self.node.count("--no-python"), 1)
        self.assertNotIn("-m torch.distributed.run", self.node)
        self.assertIn(
            'os.execve(fd,[python_path,"-I","-S","-B","-c",'
            "torchrun_bootstrap,torchrun_source",
            self.node,
        )
        self.assertIn(
            "torchrun_fd,torchrun_raw=stable(torchrun_path,torchrun_sha,"
            "torchrun_size,0o644,2012,2000,False)",
            self.node,
        )
        self.assertIn(
            "fd,_=stable(python_path,python_sha,python_size,0o755,2012,2000,True)",
            self.node,
        )

        # The trainer pathname is inspected before launch, but no command ever
        # consumes ${runner}; rank execution names only the verified target.
        runner_lines = [
            line for line in self.node.splitlines() if '"${runner}"' in line
        ]
        self.assertEqual(len(runner_lines), 1)
        self.assertIn("[[", runner_lines[0])
        self.assertNotIn('exec "${python_bin}"', self.node)
        self.assertNotIn('"${python_bin}" -I -B "${runner}"', self.node)

    def test_every_frozen_runtime_bootstrap_disables_automatic_site(self) -> None:
        for label, source in (("controller", self.controller), ("node", self.node)):
            with self.subTest(script=label):
                bootstrap = heredoc(source, "runtime_bootstrap_source")
                self.assertIn(
                    'os.execve(fd,[frozen,"-I","-S","-B","-c",'
                    "source_text,*runtime_args]",
                    bootstrap,
                )
                self.assertNotIn(
                    'os.execve(fd,[frozen,"-I","-B","-c"', bootstrap
                )
        self.assertNotIn('"${python_bin}" -I -B -m', self.node)

    def test_torchrun_capture_has_exact_literal_and_full_same_fd_identity(self) -> None:
        expected_path = (
            "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
            "site-packages/torch/distributed/run.py"
        )
        expected_sha = (
            "1aed399471b08b12c536def56553a6dfe"
            "53be234a52e0df48df325c6477f7e8c"
        )
        for value in (expected_path, expected_sha):
            self.assertIn(value, self.controller)
            self.assertIn(value, self.node)
        self.assertIn("readonly torchrun_size=31587", self.controller)
        self.assertIn('"${torchrun_size}" == 31587', self.node)
        self.assertIn(
            'flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|'
            'getattr(os,"O_CLOEXEC",0)',
            self.node,
        )
        for field in (
            "st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
            "st_rdev", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns",
        ):
            self.assertIn(field, self.node)
        self.assertIn("a=os.fstat(fd); first=b\"\"", self.node)
        self.assertIn("b=os.fstat(fd); os.lseek(fd,0,os.SEEK_SET)", self.node)
        self.assertIn("c=os.fstat(fd); named=os.lstat(path)", self.node)
        self.assertIn("ident(a)!=ident(b) or ident(a)!=ident(c)", self.node)
        self.assertIn("or ident(a)!=ident(named)", self.node)
        self.assertIn("first!=second", self.node)
        self.assertIn("a.st_nlink,a.st_size", self.node)

    def test_captured_torchrun_never_processes_pth_or_reopens_named_path(self) -> None:
        bootstrap = heredoc(self.node, "torchrun_bootstrap_source")
        remote_site = (
            "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
            "site-packages"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            site_packages = root / "site-packages"
            origin = site_packages / "torch" / "distributed" / "run.py"
            origin.parent.mkdir(parents=True)
            output = root / "captured-result.json"
            named_reopen_sentinel = root / "named-path-reopened.txt"
            pth_sentinel = root / "pth-ran.txt"
            custom_sentinel = root / "sitecustomize-ran.txt"
            origin.write_text(
                "from pathlib import Path\n"
                f"Path({str(named_reopen_sentinel)!r}).write_text('REOPENED', encoding='utf-8')\n",
                encoding="utf-8",
            )
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
            trusted_source = (
                "import json,sys\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps({"
                "'origin':__file__,'sitecustomize':('sitecustomize' in sys.modules),"
                "'usercustomize':('usercustomize' in sys.modules)},sort_keys=True),"
                "encoding='utf-8')\n"
            )
            local_bootstrap = bootstrap.replace(remote_site, str(site_packages))
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    local_bootstrap,
                    trusted_source,
                    str(origin),
                    sha256(trusted_source.encode("utf-8")),
                    str(site_packages),
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            observed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(observed["origin"], str(origin))
            self.assertFalse(observed["sitecustomize"])
            self.assertFalse(observed["usercustomize"])
            self.assertFalse(named_reopen_sentinel.exists())
            self.assertFalse(pth_sentinel.exists())
            self.assertFalse(custom_sentinel.exists())

    def test_controller_audits_and_completion_use_captured_targets(self) -> None:
        auditor = function_body(self.controller, "run_release_auditor")
        self.assertIn("run_release_runtime frozen verified-run", auditor)
        self.assertIn('--release-root "${materialized}"', auditor)
        self.assertIn('--manifest "${release_manifest}"', auditor)
        self.assertIn(
            "--target audit_self_generated_action_preservation_v2.py --", auditor
        )
        self.assertNotIn("run_frozen_python_clean", self.controller)

        completion = self.controller[
            self.controller.index("# The held-FD publisher captures") :
        ]
        self.assertRegex(
            completion,
            r'(?m)^exec "\$\{root_python\}" -I -S -B -c '
            r'"\$\{runtime_bootstrap_source\}"',
        )
        self.assertIn(
            '"${source_revision}" frozen "${frozen_python}" '
            '"${frozen_python_sha}" 31490256',
            completion,
        )
        self.assertIn("verified-run", completion)
        self.assertIn(
            "--target action_preservation_completion_publisher_v1.py --",
            completion,
        )
        self.assertNotIn('"${completion_publisher}"', completion)

    def test_release_manifest_environment_is_mandatory_and_forwarded(self) -> None:
        common = array_body(self.controller, "common_env")
        self.assertIn(
            'ACTION_PRESERVATION_RELEASE_MANIFEST="${release_manifest}"', common
        )
        self.assertIn(
            "ACTION_PRESERVATION_RELEASE_MANIFEST_SHA256="
            '"${release_manifest_sha}"',
            common,
        )
        for binding in (
            'ACTION_PRESERVATION_FROZEN_SITE_PACKAGES="${frozen_site_packages}"',
            'ACTION_PRESERVATION_TORCHRUN_PATH="${torchrun_path}"',
            'ACTION_PRESERVATION_TORCHRUN_SHA256="${torchrun_sha}"',
            'ACTION_PRESERVATION_TORCHRUN_SIZE="${torchrun_size}"',
        ):
            self.assertIn(binding, common)
        self.assertIn(
            'readonly release_manifest="${ACTION_PRESERVATION_RELEASE_MANIFEST:'
            '?set release manifest}"',
            self.node,
        )
        self.assertIn(
            'readonly release_manifest_sha="${ACTION_PRESERVATION_RELEASE_MANIFEST_SHA256:'
            '?pin release manifest SHA-256}"',
            self.node,
        )

        base_node_env = {
            "PATH": "/usr/bin:/bin",
            "ACTION_PRESERVATION_NODE_CONFIRM": (
                "run-approved-action-preservation-v2-seed20260818-r1"
            ),
            "ACTION_PRESERVATION_MODE": "cache",
            "ACTION_PRESERVATION_SOURCE_ARCHIVE": "/tmp/not-opened-source.tar",
            "ACTION_PRESERVATION_SOURCE_ARCHIVE_SHA256": "a" * 64,
        }
        cases = (
            ({}, "ACTION_PRESERVATION_RELEASE_MANIFEST"),
            (
                {
                    "ACTION_PRESERVATION_RELEASE_MANIFEST": (
                        "/tmp/not-opened-source.manifest.json"
                    )
                },
                "ACTION_PRESERVATION_RELEASE_MANIFEST_SHA256",
            ),
        )
        for additions, missing_name in cases:
            with self.subTest(missing=missing_name):
                completed = subprocess.run(
                    ["/bin/bash", "--noprofile", "--norc", "-p", str(NODE_RUNNER)],
                    env={**base_node_env, **additions},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(missing_name, completed.stderr)

        controller_env = {
            "PATH": "/usr/bin:/bin",
            "APV2_CONFIRM": "not-yet-checked",
            "APV2_RELEASE_ROOT": "/vast/users/guangyi.chen/not-opened-release",
            "APV2_EXPERIMENT_ROOT": "/vast/users/guangyi.chen/not-created-experiment",
            "APV2_ARCHIVE_SHA256": "a" * 64,
        }
        completed = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-p", str(CONTROLLER)],
            env=controller_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("APV2_RELEASE_MANIFEST_SHA256", completed.stderr)

    def test_embedded_bootstrap_runs_runtime_captured_from_archive(self) -> None:
        controller_bootstrap = heredoc(self.controller, "runtime_bootstrap_source")
        node_bootstrap = heredoc(self.node, "runtime_bootstrap_source")
        self.assertEqual(controller_bootstrap, node_bootstrap)

        runtime_source = (
            b"import json,sys\n"
            b"print(json.dumps({'origin':'captured-archive-runtime',"
            b"'argv':sys.argv[1:]},sort_keys=True))\n"
        )
        runtime_relative = "action_preservation_verified_release_v1.py"
        member_root = "methods/bernini_action_editing"
        revision = "b" * 40
        row = {
            "path": runtime_relative,
            "mode": 0o444,
            "size": len(runtime_source),
            "sha256": sha256(runtime_source),
        }
        manifest_value = {
            "content_revision": revision,
            "member_root": member_root,
            "files": [row],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "source.tar"
            manifest = root / "source.manifest.json"
            stream = io.BytesIO()
            with tarfile.open(
                fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT
            ) as handle:
                info = tarfile.TarInfo(f"{member_root}/{runtime_relative}")
                info.size = len(runtime_source)
                info.mode = 0o444
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                handle.addfile(info, io.BytesIO(runtime_source))
            archive_raw = stream.getvalue()
            manifest_raw = canonical_json_bytes(manifest_value)
            archive.write_bytes(archive_raw)
            manifest.write_bytes(manifest_raw)
            archive.chmod(0o444)
            manifest.chmod(0o444)

            # A hostile same-named cwd file proves the bootstrap source comes
            # from the captured archive member, not import/path lookup.
            (root / runtime_relative).write_text(
                "raise RuntimeError('cwd runtime was reopened')\n", encoding="utf-8"
            )
            argument_sets = (
                [
                    "verified-shell-run",
                    "--release-root",
                    "/sealed/release",
                    "--manifest",
                    str(manifest),
                    "--target",
                    "scripts/auh_run_self_generated_action_preservation_v2.sh",
                    "--",
                ],
                [
                    "verified-run",
                    "--release-root",
                    "/scratch/source",
                    "--manifest",
                    str(manifest),
                    "--target",
                    "train_self_generated_action_quotient_v1.py",
                    "--",
                    "--mode",
                    "train",
                ],
            )
            for runtime_arguments in argument_sets:
                with self.subTest(command=runtime_arguments[0]):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            "-S",
                            "-B",
                            "-c",
                            controller_bootstrap,
                            str(archive),
                            sha256(archive_raw),
                            str(manifest),
                            sha256(manifest_raw),
                            revision,
                            "root",
                            sys.executable,
                            "0" * 64,
                            str(os.stat(sys.executable).st_size),
                            *runtime_arguments,
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    observed = json.loads(completed.stdout)
                    self.assertEqual(observed["origin"], "captured-archive-runtime")
                    self.assertEqual(observed["argv"], runtime_arguments)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
for entry in (str(METHOD_ROOT), str(TOOLS_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import action_preservation_decoded_eval_verified_release_v1 as runtime
import action_preservation_decoded_eval_deployment_controller_v1 as deployment_controller
import build_action_preservation_decoded_eval_release_v4 as builder


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def make_writable(root: Path) -> None:
    if not root.exists():
        return
    for current, directories, _ in os.walk(root):
        try:
            Path(current).chmod(0o700)
        except OSError:
            pass
        for name in directories:
            try:
                (Path(current) / name).chmod(0o700)
            except OSError:
                pass


class DecodedEvalVerifiedReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.site_packages = self.root / "site-packages"
        self.site_packages.mkdir()
        self.site_packages.chmod(0o555)
        self.invocation_index = 0
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        make_writable(self.root)
        self.temporary.cleanup()

    def _completion_anchor(self) -> dict:
        value = {
            "schema_version": runtime.HOLDER_COMPLETION_ANCHOR_SCHEMA,
            "holder_job_id": "136719",
            "completion_path": (
                "/tmp/apv2-anchor-fixture/execution_shards/"
                "136719.holder-directory-completion.json"
            ),
            "initial_inode_identity": {
                "device": 1, "inode": 2, "uid": 3, "gid": 4,
                "rdev": 0,
            },
            "completion_sha256": sha256(b"completion bytes"),
            "completion_size": len(b"completion bytes"),
            "completion_mode": 0o444,
            "completion_digest": sha256(b"completion object"),
            "holder_summary_digest": sha256(b"summary object"),
        }
        value["anchor_digest"] = runtime.object_sha256(value)
        return value

    def _aggregate_completion_anchor(self) -> dict:
        def identity(*, directory: bool, mode: int, inode: int) -> dict:
            return {
                "device": 1,
                "inode": inode,
                "uid": 3,
                "gid": 4,
                "mode": (
                    (stat.S_IFDIR if directory else stat.S_IFREG) | mode
                ),
                "nlink": 1,
                "rdev": 0,
                "size": 0 if directory else 17,
                "blocks": 1,
                "mtime_ns": 10,
                "ctime_ns": 11,
            }

        def file_binding(
            relative_path: str, *, mode: int, inode: int,
        ) -> dict:
            return {
                "relative_path": relative_path,
                "sha256": sha256(relative_path.encode("utf-8")),
                "size": 17,
                "mode": mode,
                "identity": identity(
                    directory=False, mode=mode, inode=inode
                ),
                "object_digest": sha256(
                    (relative_path + ":object").encode("utf-8")
                ),
            }

        media_identity = identity(directory=True, mode=0o555, inode=6)
        media_rows_digest = sha256(b"media rows")
        value = {
            "schema_version": runtime.AGGREGATE_COMPLETION_ANCHOR_SCHEMA,
            "evaluation_id": "evaluation-r4",
            "aggregate_root": "/tmp/apv2-anchor-fixture/aggregate-r4",
            "aggregate_root_identity": identity(
                directory=True, mode=0o555, inode=5
            ),
            "aggregate_file": file_binding(
                "evaluation_complete.json", mode=0o444, inode=7
            ),
            "private_file": file_binding(
                "private_blind_mapping.json", mode=0o400, inode=8
            ),
            "public_file": file_binding(
                "blind_review_packet.json", mode=0o444, inode=9
            ),
            "media_directory_identity": media_identity,
            "media_file_count": 264,
            "media_rows_digest": media_rows_digest,
            "media_tree_digest": runtime.object_sha256(
                {
                    "media_directory_identity": media_identity,
                    "media_file_count": 264,
                    "media_rows_digest": media_rows_digest,
                }
            ),
        }
        value["anchor_digest"] = runtime.object_sha256(value)
        return value

    def test_completion_anchor_channel_is_exactly_once_and_cloexec(self) -> None:
        try:
            parent, child = socket.socketpair(
                socket.AF_UNIX, socket.SOCK_SEQPACKET
            )
        except OSError as error:
            self.skipTest(f"AF_UNIX SOCK_SEQPACKET unavailable: {error}")
        old_channel = os.environ.get(runtime.COMPLETION_ANCHOR_CHANNEL_ENV)
        old_sent = os.environ.get(runtime.COMPLETION_ANCHOR_SENT_ENV)
        try:
            binding = {
                "schema_version": runtime.COMPLETION_ANCHOR_CHANNEL_SCHEMA,
                "descriptor": child.fileno(),
                "controller_pid": os.getppid(),
                "target_pid": os.getpid(),
                "expected_target": (
                    "action_preservation_decoded_eval_executor_v2.py"
                ),
            }
            binding["binding_digest"] = runtime.object_sha256(binding)
            os.environ[runtime.COMPLETION_ANCHOR_CHANNEL_ENV] = (
                runtime.canonical_json_bytes(binding).decode("utf-8")
            )
            os.environ.pop(runtime.COMPLETION_ANCHOR_SENT_ENV, None)
            child.set_inheritable(True)
            sealed = runtime.seal_completion_anchor_channel(
                expected_target=(
                    "action_preservation_decoded_eval_executor_v2.py"
                )
            )
            self.assertFalse(os.get_inheritable(sealed["descriptor"]))
            anchor = self._completion_anchor()
            self.assertEqual(
                runtime.publish_holder_completion_anchor(anchor), anchor
            )
            self.assertEqual(
                parent.recv(16384), runtime.canonical_json_bytes(anchor) + b"\n"
            )
            with self.assertRaisesRegex(
                runtime.DecodedEvalVerifiedReleaseError, "already sent"
            ):
                runtime.publish_holder_completion_anchor(anchor)
        finally:
            if old_channel is None:
                os.environ.pop(runtime.COMPLETION_ANCHOR_CHANNEL_ENV, None)
            else:
                os.environ[runtime.COMPLETION_ANCHOR_CHANNEL_ENV] = old_channel
            if old_sent is None:
                os.environ.pop(runtime.COMPLETION_ANCHOR_SENT_ENV, None)
            else:
                os.environ[runtime.COMPLETION_ANCHOR_SENT_ENV] = old_sent
            parent.close()
            child.close()

    def test_completion_anchor_rejects_resigned_shape_path_and_pid(self) -> None:
        anchor = self._completion_anchor()
        hostile = copy.deepcopy(anchor)
        hostile["extra"] = True
        hostile["anchor_digest"] = runtime.object_sha256(
            {key: item for key, item in hostile.items()
             if key != "anchor_digest"}
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_holder_completion_anchor(hostile)

    def test_aggregate_completion_anchor_exact12_schema_is_closed(self) -> None:
        anchor = self._aggregate_completion_anchor()
        self.assertEqual(
            runtime.validate_aggregate_completion_anchor(anchor), anchor
        )
        hostile = copy.deepcopy(anchor)
        hostile["extra"] = True
        hostile["anchor_digest"] = runtime.object_sha256(
            {
                key: value for key, value in hostile.items()
                if key != "anchor_digest"
            }
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_aggregate_completion_anchor(hostile)
        hostile = copy.deepcopy(anchor)
        hostile["media_directory_identity"]["mode"] = (
            stat.S_IFREG | 0o555
        )
        hostile["anchor_digest"] = runtime.object_sha256(
            {
                key: value for key, value in hostile.items()
                if key != "anchor_digest"
            }
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_aggregate_completion_anchor(hostile)
        hostile = copy.deepcopy(anchor)
        hostile["completion_path"] = "/tmp/wrong.json"
        hostile["anchor_digest"] = runtime.object_sha256(
            {key: item for key, item in hostile.items()
             if key != "anchor_digest"}
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_holder_completion_anchor(hostile)

    def build_release(self, name: str = "release") -> tuple[Path, dict, dict, dict]:
        release = self.root / name
        receipt = builder.build(release)
        manifest = json.loads((release / "source.manifest.json").read_text("utf-8"))
        envelope = json.loads((release / "deployment-envelope.json").read_text("utf-8"))
        return release, receipt, manifest, envelope

    def extract(
        self, release: Path, receipt: dict, manifest: dict, envelope: dict,
        name: str = "materialized",
    ) -> Path:
        destination = self.root / name
        runtime.extract_verified_release(
            archive=release / "source.tar",
            expected_archive_sha256=receipt["archive_sha256"],
            manifest=release / "source.manifest.json",
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_content_revision=manifest["content_revision"],
            envelope=release / "deployment-envelope.json",
            expected_envelope_sha256=receipt["envelope_sha256"],
            output_root=destination,
        )
        return destination

    def interpreter_binding(self) -> dict:
        path = Path(os.path.realpath(sys.executable))
        return runtime.capture_executable_binding(path, label="test Python")

    def test_retained_parent_create_only_write_rejects_root_replacement(
        self,
    ) -> None:
        work_root = self.root / "retained-write-root"
        work_root.mkdir(mode=0o700)
        flags = (
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        root_fd = os.open(work_root, flags)
        os.set_inheritable(root_fd, False)
        try:
            positive = runtime._write_create_only(
                work_root / "positive.json",
                b'{"positive":true}\n',
                retained_parent_fd=root_fd,
                label="retained positive",
            )
            self.assertEqual(positive["mode"], 0o444)

            displaced = self.root / "retained-write-root-displaced"
            original_read = runtime._read_fd
            triggered = False

            def hostile_read(descriptor: int) -> bytes:
                nonlocal triggered
                raw = original_read(descriptor)
                if not triggered:
                    triggered = True
                    os.rename(work_root, displaced)
                    os.mkdir(work_root, 0o700)
                return raw

            with mock.patch.object(runtime, "_read_fd", side_effect=hostile_read):
                with self.assertRaisesRegex(
                    runtime.DecodedEvalVerifiedReleaseError,
                    "parent|retained post-close replay",
                ):
                    runtime._write_create_only(
                        work_root / "hostile.json",
                        b'{"hostile":true}\n',
                        retained_parent_fd=root_fd,
                        label="retained hostile",
                    )
            self.assertEqual(list(work_root.iterdir()), [])
            self.assertTrue((displaced / "hostile.json").is_file())
        finally:
            os.close(root_fd)

    def test_inherited_work_root_v2_binds_exact_authority_receipt_files(self) -> None:
        work_root = self.root / "authority-work"
        work_root.mkdir(mode=0o700)
        work_root.chmod(0o700)
        creation = deployment_controller._identity_row(work_root.stat())
        parent = deployment_controller._identity_row(work_root.parent.stat())
        immutable = ("device", "inode", "uid", "gid", "mode", "rdev")
        authority = {
            "schema_version": deployment_controller.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(work_root),
            "parent_path": str(work_root.parent),
            "creation_identity": creation,
            "immutable_identity": {key: creation[key] for key in immutable},
            "parent_immutable_identity": {
                key: parent[key] for key in immutable
            },
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        authority["authority_digest"] = deployment_controller.object_sha256(
            authority
        )
        deployment_digest = sha256(b"deployment object")
        source_digest = sha256(b"source authority object")
        deployment_value = {
            "work_root_authority": authority,
            "receipt_digest": deployment_digest,
        }
        source_value = {
            "work_root_authority": authority,
            "deployment_receipt_digest": deployment_digest,
            "receipt_digest": source_digest,
        }
        deployment_path = work_root / "deployment.json"
        source_path = work_root / "source-authority.json"
        deployment_raw = runtime.canonical_json_bytes(deployment_value) + b"\n"
        source_raw = runtime.canonical_json_bytes(source_value) + b"\n"
        deployment_path.write_bytes(deployment_raw)
        source_path.write_bytes(source_raw)
        deployment_path.chmod(0o444)
        source_path.chmod(0o444)
        held = deployment_controller._HeldWorkRoot.open(authority)
        self.addCleanup(held.close)
        binding = held.inherited_binding(
            deployment_receipt={
                "path": str(deployment_path),
                "sha256": sha256(deployment_raw),
            },
            source_spec_authority={
                "path": str(source_path),
                "sha256": sha256(source_raw),
            },
            deployment_receipt_digest=deployment_digest,
            source_spec_authority_digest=source_digest,
            target="action_preservation_decoded_eval_bridge_v1.py",
            capture_receipt_path=work_root / "runtime-capture.json",
        )
        self.assertEqual(
            runtime.validate_inherited_work_root_binding(
                binding,
                verify_open_fds=True,
                expected_inheritable=False,
            ),
            binding,
        )

        def resign(value: dict) -> dict:
            value.pop("binding_digest", None)
            value["binding_digest"] = runtime.object_sha256(value)
            return value

        swapped = copy.deepcopy(binding)
        swapped["deployment_receipt"], swapped["source_spec_authority"] = (
            swapped["source_spec_authority"],
            swapped["deployment_receipt"],
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_inherited_work_root_binding(
                resign(swapped), verify_open_fds=True,
                expected_inheritable=False,
            )
        wrong_sha = copy.deepcopy(binding)
        wrong_sha["deployment_receipt"]["sha256"] = sha256(b"wrong")
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_inherited_work_root_binding(
                resign(wrong_sha), verify_open_fds=True,
                expected_inheritable=False,
            )
        resigned = copy.deepcopy(binding)
        resigned["deployment_receipt_digest"] = sha256(b"resigned")
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_inherited_work_root_binding(
                resign(resigned), verify_open_fds=True,
                expected_inheritable=False,
            )

    def release_binding(
        self, release: Path, receipt: dict, manifest: dict, envelope: dict,
        materialized: Path,
    ) -> dict:
        return runtime.capture_release_binding(
            release_root=materialized, archive=release / "source.tar",
            expected_archive_sha256=receipt["archive_sha256"],
            manifest=release / "source.manifest.json",
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_content_revision=manifest["content_revision"],
            envelope=release / "deployment-envelope.json",
            expected_envelope_sha256=receipt["envelope_sha256"],
        )

    def invocation_bindings(
        self, release: Path, receipt: dict, manifest: dict, envelope: dict,
        materialized: Path, *, site_packages: Path | None = None,
        torchrun_binding: dict | None = None,
    ) -> dict:
        self.invocation_index += 1
        interpreter = self.interpreter_binding()
        release_value = self.release_binding(
            release, receipt, manifest, envelope, materialized
        )
        site_value = runtime.capture_directory_binding(
            self.site_packages if site_packages is None else site_packages,
            label="test site-packages",
        )
        controller = runtime.capture_file_binding(
            Path(runtime.__file__).resolve(), label="test detached controller"
        )
        authority_path = self.root / f"controller-authority-{self.invocation_index}.json"
        authority = runtime.publish_controller_authority_receipt(
            authority_path, controller_binding=controller,
            root_python_binding=interpreter, frozen_python_binding=interpreter,
            site_packages_binding=site_value, release_binding=release_value,
            torchrun_binding=torchrun_binding,
        )
        return {
            "root_python_binding": interpreter,
            "frozen_python_binding": interpreter,
            "site_packages_binding": site_value,
            "release_binding": release_value,
            "controller_binding": controller,
            "controller_authority_binding": authority,
        }

    def verified_run_kwargs(
        self, release: Path, receipt: dict, manifest: dict, envelope: dict,
        materialized: Path, *, target: str, arguments: list[str],
    ) -> dict:
        bindings = self.invocation_bindings(
            release, receipt, manifest, envelope, materialized
        )
        self.invocation_index += 1
        return {
            **bindings,
            "target": target, "target_arguments": arguments,
            "capture_receipt_path": self.root / f"capture-{self.invocation_index}.json",
        }

    def test_packager_is_deterministic_create_only_and_seals_exact_three(self) -> None:
        first, receipt1, manifest, envelope = self.build_release("first")
        second, receipt2, _, _ = self.build_release("second")
        self.assertEqual(
            {key: receipt1[key] for key in (
                "archive_sha256", "manifest_sha256", "manifest_digest",
                "content_revision", "envelope_sha256", "envelope_digest",
            )},
            {key: receipt2[key] for key in (
                "archive_sha256", "manifest_sha256", "manifest_digest",
                "content_revision", "envelope_sha256", "envelope_digest",
            )},
        )
        self.assertEqual(tuple(row["path"] for row in manifest["files"]), runtime.EVAL_RELEASE_MEMBERS)
        self.assertEqual(
            list(runtime.EVAL_RELEASE_MEMBERS), sorted(runtime.EVAL_RELEASE_MEMBERS)
        )
        self.assertEqual(
            [row["path"] for row in manifest["files"]],
            sorted(row["path"] for row in manifest["files"]),
        )
        self.assertIs(
            envelope["detached_controller_authority_receipt_required"], True
        )
        self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o555)
        self.assertEqual({item.name for item in first.iterdir()}, {
            "source.tar", "source.manifest.json", "deployment-envelope.json"
        })
        for item in first.iterdir():
            self.assertEqual(item.stat().st_nlink, 1)
            self.assertEqual(stat.S_IMODE(item.stat().st_mode), 0o444)
        with self.assertRaises(builder.Exact15ReleaseBuildError):
            builder.build(first)

    def test_materialized_tree_rejects_hardlink_symlink_and_extra(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        for index, topology in enumerate(("hardlink", "symlink", "extra")):
            with self.subTest(topology=topology):
                materialized = self.extract(
                    release, receipt, manifest, envelope, f"tree-{index}"
                )
                member_root = materialized / runtime.MEMBER_ROOT
                member_root.chmod(0o755)
                extra = member_root / "unexpected.py"
                target = member_root / "train_lora.py"
                if topology == "hardlink":
                    os.link(target, extra)
                elif topology == "symlink":
                    extra.symlink_to(target)
                else:
                    extra.write_bytes(b"unexpected\n")
                member_root.chmod(0o555)
                with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
                    runtime.capture_materialized_release(materialized, manifest)

    def test_release_artifact_hardlink_is_rejected(self) -> None:
        release, receipt, manifest, _ = self.build_release()
        release.chmod(0o755)
        os.link(release / "source.tar", release / "source-copy.tar")
        release.chmod(0o555)
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.capture_release_artifacts(
                archive=release / "source.tar",
                expected_archive_sha256=receipt["archive_sha256"],
                manifest=release / "source.manifest.json",
                expected_manifest_sha256=receipt["manifest_sha256"],
                expected_content_revision=manifest["content_revision"],
                envelope=release / "deployment-envelope.json",
                expected_envelope_sha256=receipt["envelope_sha256"],
            )

    def test_same_inode_ctime_drift_is_rejected(self) -> None:
        path = self.root / "stable.bin"
        path.write_bytes(b"stable bytes")
        path.chmod(0o444)
        original = runtime._read_fd
        calls = 0

        def drifting_read(descriptor: int) -> bytes:
            nonlocal calls
            raw = original(descriptor)
            calls += 1
            if calls == 1:
                os.fchmod(descriptor, 0o400)
                os.fchmod(descriptor, 0o444)
            return raw

        with mock.patch.object(runtime, "_read_fd", side_effect=drifting_read):
            with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
                runtime._stable_capture(
                    path, label="ctime drift", expected_sha256=sha256(b"stable bytes"),
                    expected_mode=0o444,
                )

    def test_full_binding_rejects_same_inode_post_capture_ctime_drift(self) -> None:
        path = self.root / "identity.bin"
        path.write_bytes(b"same inode and content")
        path.chmod(0o444)
        binding = runtime.capture_file_binding(path, label="identity fixture")
        before = path.stat()
        path.chmod(0o400)
        path.chmod(0o444)
        after = path.stat()
        self.assertEqual(before.st_ino, after.st_ino)
        self.assertEqual(binding["sha256"], sha256(path.read_bytes()))
        self.assertEqual(binding["size"], after.st_size)
        self.assertNotEqual(binding["ctime_ns"], after.st_ctime_ns)
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.replay_file_binding(binding, label="ctime-drifted fixture")

    def test_controller_authority_receipt_is_create_only_and_replayed(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        materialized = self.extract(release, receipt, manifest, envelope)
        bindings = self.invocation_bindings(
            release, receipt, manifest, envelope, materialized
        )
        authority = bindings["controller_authority_binding"]
        runtime.validate_controller_authority_binding(
            authority,
            controller_binding=bindings["controller_binding"],
            root_python_binding=bindings["root_python_binding"],
            frozen_python_binding=bindings["frozen_python_binding"],
            site_packages_binding=bindings["site_packages_binding"],
            release_binding=bindings["release_binding"], verify_file=True,
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.publish_controller_authority_receipt(
                Path(authority["receipt"]["path"]),
                controller_binding=bindings["controller_binding"],
                root_python_binding=bindings["root_python_binding"],
                frozen_python_binding=bindings["frozen_python_binding"],
                site_packages_binding=bindings["site_packages_binding"],
                release_binding=bindings["release_binding"], torchrun_binding=None,
            )
        authority_path = Path(authority["receipt"]["path"])
        authority_path.chmod(0o400)
        authority_path.chmod(0o444)
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.validate_controller_authority_binding(
                authority,
                controller_binding=bindings["controller_binding"],
                root_python_binding=bindings["root_python_binding"],
                frozen_python_binding=bindings["frozen_python_binding"],
                site_packages_binding=bindings["site_packages_binding"],
                release_binding=bindings["release_binding"],
                torchrun_binding=None, require_torchrun_continuity=True,
                verify_file=True,
            )

    def test_tools_namespace_cannot_fall_through_to_pathfinder(self) -> None:
        source = self.root / "tools" / "known.py"
        finder = runtime._CapturedReleaseFinder(
            modules={"tools.known": (source, b"VALUE = 1\n")},
            namespace_roots={"tools": source.parent},
            forbidden_roots=(),
        )
        package_spec = finder.find_spec("tools")
        self.assertIsNotNone(package_spec)
        assert package_spec is not None and package_spec.loader is not None
        module = package_spec.loader.create_module(package_spec)
        if module is None:
            import types
            module = types.ModuleType("tools")
        package_spec.loader.exec_module(module)
        self.assertEqual(module.__path__, [])
        with self.assertRaises(ImportError):
            finder.find_spec("tools.hostile")

    def test_root_bootstrap_executes_captured_tools_target_and_writes_receipt(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        materialized = self.extract(release, receipt, manifest, envelope)
        bindings = self.invocation_bindings(
            release, receipt, manifest, envelope, materialized
        )
        capture_path = self.root / "capture.json"
        argv = runtime.verified_target_argv(
            **bindings, target="tools/materialize_vae.py", args=["--help"],
            capture_receipt_path=str(capture_path),
        )
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage: materialize_vae.py", completed.stdout)
        value = json.loads(capture_path.read_text("utf-8"))
        runtime.validate_capture_receipt(
            value, verify_file=True, receipt_path=capture_path
        )
        self.assertEqual(value["target"], "tools/materialize_vae.py")
        self.assertEqual(value["member_count"], 15)

    def test_path_replacement_after_capture_still_uses_captured_import(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        materialized = self.extract(release, receipt, manifest, envelope)
        original_capture = runtime.capture_materialized_release
        replaced = False

        def capture_then_replace(root: Path, value: dict) -> dict:
            nonlocal replaced
            payloads = dict(original_capture(root, value))
            member_root = root / runtime.MEMBER_ROOT
            tools_root = member_root / "tools"
            target = tools_root / "materialize_vae.py"
            tools_root.chmod(0o755)
            target.unlink()
            target.write_bytes(b"raise RuntimeError('path replacement was reopened')\n")
            target.chmod(0o444)
            tools_root.chmod(0o555)
            replaced = True
            return payloads

        kwargs = self.verified_run_kwargs(
            release, receipt, manifest, envelope, materialized,
            target="tools/materialize_vae.py", arguments=["--help"],
        )
        guarded_names = {
            relative[:-3].replace("/", ".")
            for relative in runtime.EVAL_RELEASE_MEMBERS
            if relative.endswith(".py")
        } | {"tools"}
        loaded = {
            name: sys.modules.pop(name)
            for name in guarded_names
            if name in sys.modules
        }
        try:
            with mock.patch.object(
                runtime, "capture_materialized_release", side_effect=capture_then_replace
            ):
                with self.assertRaises(SystemExit) as exit_context:
                    runtime.verified_python_run(**kwargs)
            self.assertEqual(exit_context.exception.code, 0)
        finally:
            sys.modules.update(loaded)
        self.assertTrue(replaced)

    def test_fully_resigned_hostile_member_is_rejected_by_exact15_anchor(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        _, _, payloads, _ = runtime.capture_release_artifacts(
            archive=release / "source.tar", expected_archive_sha256=receipt["archive_sha256"],
            manifest=release / "source.manifest.json",
            expected_manifest_sha256=receipt["manifest_sha256"],
            expected_content_revision=manifest["content_revision"],
            envelope=release / "deployment-envelope.json",
            expected_envelope_sha256=receipt["envelope_sha256"],
        )
        hostile_manifest = copy.deepcopy(manifest)
        hostile_payloads = dict(payloads)
        hostile_payloads["train_lora.py"] += b"\n# hostile fully resigned change\n"
        for row in hostile_manifest["files"]:
            raw = hostile_payloads[row["path"]]
            row["size"], row["sha256"] = len(raw), sha256(raw)
        hostile_manifest["component_sha256"] = {
            row["path"]: row["sha256"] for row in hostile_manifest["files"]
        }
        hostile_manifest["content_revision"] = runtime.content_revision(
            hostile_manifest["files"]
        )
        hostile_manifest.pop("manifest_digest")
        hostile_manifest["manifest_digest"] = runtime.object_sha256(hostile_manifest)
        hostile_manifest_raw = runtime.canonical_json_bytes(hostile_manifest) + b"\n"
        hostile_archive = runtime.fixed_ustar_archive(
            hostile_manifest["files"], hostile_payloads
        )
        hostile_envelope = copy.deepcopy(envelope)
        hostile_envelope["source_archive"]["sha256"] = sha256(hostile_archive)
        hostile_envelope["source_manifest"].update({
            "sha256": sha256(hostile_manifest_raw),
            "manifest_digest": hostile_manifest["manifest_digest"],
            "content_revision": hostile_manifest["content_revision"],
        })
        hostile_envelope.pop("envelope_digest")
        hostile_envelope["envelope_digest"] = runtime.object_sha256(hostile_envelope)
        hostile_envelope_raw = runtime.canonical_json_bytes(hostile_envelope) + b"\n"
        hostile = self.root / "hostile"
        hostile.mkdir()
        for name, raw in (
            ("source.tar", hostile_archive),
            ("source.manifest.json", hostile_manifest_raw),
            ("deployment-envelope.json", hostile_envelope_raw),
        ):
            path = hostile / name
            path.write_bytes(raw)
            path.chmod(0o444)
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.capture_release_artifacts(
                archive=hostile / "source.tar",
                expected_archive_sha256=sha256(hostile_archive),
                manifest=hostile / "source.manifest.json",
                expected_manifest_sha256=sha256(hostile_manifest_raw),
                expected_content_revision=hostile_manifest["content_revision"],
                envelope=hostile / "deployment-envelope.json",
                expected_envelope_sha256=sha256(hostile_envelope_raw),
            )

    def test_extra_zero_record_and_nonzero_trailer_are_rejected(self) -> None:
        release, receipt, manifest, _ = self.build_release()
        raw = (release / "source.tar").read_bytes()
        for label, hostile in (
            ("extra-zero-record", raw + b"\0" * runtime.FIXED_USTAR_RECORD_SIZE),
            ("nonzero-trailer", raw[:-1] + b"X"),
        ):
            with self.subTest(label=label):
                with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
                    runtime.verify_archive_snapshot(hostile, manifest)

    def test_unpinned_torchrun_is_rejected_without_site_customization(self) -> None:
        site_root = self.root / "torch-site"
        run_path = site_root / "torch" / "distributed" / "run.py"
        run_path.parent.mkdir(parents=True)
        sentinel = self.root / "sitecustomize-ran"
        (site_root / "sitecustomize.py").write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
            encoding="utf-8",
        )
        (site_root / "hostile_pth.py").write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad-pth')\n",
            encoding="utf-8",
        )
        (site_root / "hostile.pth").write_text(
            "import hostile_pth\n", encoding="utf-8"
        )
        run_path.write_text(
            "import json,os,sys\n"
            "print('CAPTURED_TORCHRUN:'+json.dumps({"
            "'sitecustomize': 'sitecustomize' in sys.modules,"
            "'usercustomize': 'usercustomize' in sys.modules,"
            "'arguments': sys.argv[1:]},sort_keys=True),flush=True)\n"
            "index=sys.argv.index('--no-python')\n"
            "rank=sys.argv[index+1:]\n"
            "os.execve(rank[0],rank,dict(os.environ))\n",
            encoding="utf-8",
        )
        run_path.chmod(0o444)
        handler_path = (
            site_root / runtime.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        )
        handler_path.parent.mkdir(parents=True)
        handler_path.write_text(
            "class SubprocessHandler:\n    pass\n", encoding="utf-8"
        )
        handler_path.chmod(0o444)
        site_root.chmod(0o555)
        with self.assertRaisesRegex(
            runtime.DecodedEvalVerifiedReleaseError,
            "torchrun source bytes differ|subprocess handler",
        ):
            runtime.capture_torchrun_binding(
                site_root, label="fake captured torchrun"
            )
        self.assertFalse(sentinel.exists())

    def test_isolated_torchrun_rejects_replaced_unpinned_run_path(self) -> None:
        site_root = self.root / "swap-site"
        run_path = site_root / "torch" / "distributed" / "run.py"
        run_path.parent.mkdir(parents=True)
        sentinel = self.root / "replaced-run-path-ran"
        benign = b"print('CAPTURED_BENIGN_TORCHRUN')\n"
        run_path.write_bytes(benign)
        run_path.chmod(0o444)
        handler_path = (
            site_root / runtime.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        )
        handler_path.parent.mkdir(parents=True)
        handler_raw = b"class SubprocessHandler:\n    pass\n"
        handler_path.write_bytes(handler_raw)
        handler_path.chmod(0o444)
        site_root.chmod(0o555)
        site_binding = runtime.capture_directory_binding(
            site_root, label="swap test site-packages"
        )
        run_path.unlink()
        run_path.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
            encoding="utf-8",
        )
        run_path.chmod(0o444)
        argv = [
            self.interpreter_binding()["path"], "-I", "-S", "-B", "-c",
            runtime.ISOLATED_TORCHRUN_BOOTSTRAP,
            benign.decode("utf-8"), str(run_path), sha256(benign),
            handler_raw.decode("utf-8"), str(handler_path),
            sha256(handler_raw),
            runtime.canonical_json_bytes(site_binding).decode("utf-8"),
        ]
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(sentinel.exists())

    def test_replaced_torchrun_cannot_self_capture_against_prior_authority(self) -> None:
        site_root = self.root / "reauthority-site"
        run_path = site_root / "torch" / "distributed" / "run.py"
        run_path.parent.mkdir(parents=True)
        run_path.write_text("print('AUTHORIZED_TORCHRUN')\n", encoding="utf-8")
        run_path.chmod(0o444)
        handler_path = (
            site_root / runtime.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        )
        handler_path.parent.mkdir(parents=True)
        handler_path.write_text(
            "class SubprocessHandler:\n    pass\n", encoding="utf-8"
        )
        handler_path.chmod(0o444)
        site_root.chmod(0o555)
        run_sha = sha256(run_path.read_bytes())
        handler_sha = sha256(handler_path.read_bytes())
        patches = (
            mock.patch.object(runtime, "TORCHRUN_SOURCE_SHA256", run_sha),
            mock.patch.object(
                runtime, "TORCHRUN_SOURCE_SIZE", run_path.stat().st_size
            ),
            mock.patch.object(
                runtime, "TORCHRUN_SUBPROCESS_HANDLER_SHA256", handler_sha
            ),
            mock.patch.object(
                runtime,
                "TORCHRUN_SUBPROCESS_HANDLER_SIZE",
                handler_path.stat().st_size,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        authorized = runtime.capture_torchrun_binding(
            site_root, label="initial authorized torchrun"
        )
        sentinel = self.root / "reauthorized-hostile-ran"
        run_path.unlink()
        run_path.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('bad')\n",
            encoding="utf-8",
        )
        run_path.chmod(0o444)
        with self.assertRaisesRegex(
            runtime.DecodedEvalVerifiedReleaseError,
            "torchrun source bytes differ",
        ):
            runtime.capture_torchrun_binding(
                site_root, label="hostile recaptured torchrun"
            )
        self.assertEqual(authorized["source"]["sha256"], run_sha)
        self.assertFalse(sentinel.exists())

    def test_frozen_exec_and_verified_runtime_argv_contract(self) -> None:
        release, receipt, manifest, envelope = self.build_release()
        materialized = self.extract(release, receipt, manifest, envelope)
        bindings = self.invocation_bindings(
            release, receipt, manifest, envelope, materialized
        )
        capture_path = self.root / "argv-capture.json"
        authority_arguments = [
            "--model-consumption-input", str(self.root / "consumption-input.json"),
            "--model-consumption-input-sha256", "1" * 64,
            "--model-consumption-input-digest", "2" * 64,
            "--task-input-digest", "3" * 64,
        ]
        runtime_args = runtime.verified_runtime_arguments(
            **bindings, target="infer_lora.py", args=authority_arguments,
            capture_receipt_path=str(capture_path),
        )
        self.assertEqual(runtime_args[0], "verified-run")
        self.assertEqual(
            runtime_args[-(len(authority_arguments) + 1):],
            ["--", *authority_arguments],
        )
        with self.assertRaisesRegex(
            runtime.DecodedEvalVerifiedReleaseError,
            "exact15 inference authority option differs",
        ):
            runtime.verified_runtime_arguments(
                **bindings, target="infer_lora.py", args=["--help"],
                capture_receipt_path=str(capture_path),
            )
        frozen = runtime.frozen_exec_argv(
            **bindings, frozen_args=["-I", "-S", "-B", "-c", "pass"],
        )
        self.assertEqual(
            frozen[:5],
            [bindings["root_python_binding"]["path"], "-I", "-S", "-B", "-c"],
        )
        with self.assertRaises(runtime.DecodedEvalVerifiedReleaseError):
            runtime.frozen_exec_argv(
                **bindings,
                frozen_args=["-I", "-S", "-B", "-m", "torch.distributed.run"],
            )


if __name__ == "__main__":
    unittest.main()

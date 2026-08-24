from __future__ import annotations

import copy
import hashlib
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import action_preservation_decoded_eval_deployment_controller_v1 as controller
import action_preservation_decoded_eval_verified_release_v1 as runtime
import build_action_preservation_decoded_eval_release_v4 as builder


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DetachedDeploymentControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        detached = self.root / "detached-release"
        audit = builder.build(detached)
        work = self.root / "work"
        work.mkdir(mode=0o700)
        materialized = work / "materialized"
        runtime_source = self.root / "verified-runtime.py"
        shutil.copyfile(pathlib.Path(runtime.__file__), runtime_source)
        runtime_source.chmod(0o444)
        controller_copy = self.root / "detached-controller.py"
        shutil.copyfile(pathlib.Path(controller.__file__), controller_copy)
        controller_copy.chmod(0o444)
        root_python = self.root / "root-python"
        frozen_python = self.root / "frozen-python"
        for path, payload in (
            (root_python, b"fixture root python\n"),
            (frozen_python, b"fixture frozen python\n"),
        ):
            path.write_bytes(payload)
            path.chmod(0o755)
        site = self.root / "site-packages"
        torchrun = site / "torch" / "distributed" / "run.py"
        torchrun.parent.mkdir(parents=True)
        torchrun.write_bytes(b"# fixture captured torchrun\n")
        handler = site / runtime.TORCHRUN_SUBPROCESS_HANDLER_RELATIVE_PATH
        handler.parent.mkdir(parents=True)
        handler.write_bytes(b"class SubprocessHandler:\n    pass\n")
        handler.chmod(0o444)
        original_load_runtime = controller._load_runtime

        def load_fixture_runtime(raw: bytes, *, origin: pathlib.Path) -> dict:
            namespace = original_load_runtime(raw, origin=origin)
            self.assertIsInstance(namespace, dict)
            namespace["TORCHRUN_SOURCE_SHA256"] = sha(torchrun)
            namespace["TORCHRUN_SOURCE_SIZE"] = torchrun.stat().st_size
            namespace["TORCHRUN_SUBPROCESS_HANDLER_SHA256"] = sha(handler)
            namespace["TORCHRUN_SUBPROCESS_HANDLER_SIZE"] = handler.stat().st_size
            return namespace

        runtime_loader = mock.patch.object(
            controller, "_load_runtime", side_effect=load_fixture_runtime
        )
        runtime_loader.start()
        self.addCleanup(runtime_loader.stop)
        authority_path = work / "controller-authority.json"
        deployment_receipt_path = work / "deployment-receipt.json"
        source_runtime_spec = work / "source-runtime-spec.json"
        source_spec_authority_path = work / "source-spec-authority.json"
        work_info = work.lstat()
        parent_info = work.parent.lstat()
        work_root_authority = {
            "schema_version": controller.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(work),
            "parent_path": str(work.parent),
            "creation_identity": controller._identity_row(work_info),
            "immutable_identity": controller._immutable_directory_identity(
                work_info
            ),
            "parent_immutable_identity": (
                controller._immutable_directory_identity(parent_info)
            ),
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        work_root_authority["authority_digest"] = controller.object_sha256(
            work_root_authority
        )
        request = {
            "schema_version": controller.REQUEST_SCHEMA,
            "release_generation": runtime.RELEASE_GENERATION,
            "work_root_authority": work_root_authority,
            "controller": {
                "path": str(controller_copy), "sha256": sha(controller_copy),
            },
            "root_python": {
                "path": str(root_python), "sha256": sha(root_python),
            },
            "frozen_python": {
                "path": str(frozen_python), "sha256": sha(frozen_python),
            },
            "site_packages_path": str(site),
            "torchrun": {"path": str(torchrun), "sha256": sha(torchrun)},
            "release_root": str(materialized),
            "archive": {
                "path": str(detached / "source.tar"),
                "sha256": audit["archive_sha256"],
            },
            "manifest": {
                "path": str(detached / "source.manifest.json"),
                "sha256": audit["manifest_sha256"],
            },
            "manifest_digest": audit["manifest_digest"],
            "content_revision": audit["content_revision"],
            "envelope": {
                "path": str(detached / "deployment-envelope.json"),
                "sha256": audit["envelope_sha256"],
            },
            "envelope_digest": audit["envelope_digest"],
            "verified_runtime_source": {
                "path": str(runtime_source), "sha256": sha(runtime_source),
            },
            "source_runtime_spec_path": str(source_runtime_spec),
            "source_spec_authority_receipt_path": str(
                source_spec_authority_path
            ),
            "controller_authority_receipt_path": str(authority_path),
            "deployment_receipt_path": str(deployment_receipt_path),
            "automatic_retry": False,
            "network_allowed": False,
            "scientific_promotion_authorized": False,
        }
        request["request_digest"] = controller.object_sha256(request)
        request_path = work / "deployment-request.json"
        request_path.write_bytes(controller.canonical_json_bytes(request) + b"\n")
        request_path.chmod(0o444)
        self.request = request
        self.work = work
        self.request_path = request_path
        self.controller_copy = controller_copy
        self.root_python = root_python
        self.receipt_path = deployment_receipt_path
        self.torchrun = torchrun
        self.source_runtime_spec = source_runtime_spec
        self.source_spec_authority_path = source_spec_authority_path

    def _capture(self) -> tuple[dict, dict]:
        with mock.patch.object(controller, "__file__", str(self.controller_copy)), \
             mock.patch.object(controller, "ROOT_PYTHON_PATH", self.root_python), \
             mock.patch.object(controller, "ROOT_PYTHON_UID", self.root_python.stat().st_uid), \
             mock.patch.object(controller, "ROOT_PYTHON_GID", self.root_python.stat().st_gid):
            return controller.capture_authority(
                request_path=self.request_path,
                expected_request_sha256=sha(self.request_path),
            )

    def _write_source_spec(self, deployment: dict) -> dict:
        release = deployment["release"]
        authority = deployment["controller_authority"]

        def pair(value: dict) -> dict:
            return {"path": value["path"], "sha256": value["sha256"]}

        value = {
            "schema_version": controller.SOURCE_RUNTIME_SCHEMA,
            "pins": {},
            "pin_files": {
                "inference_release_manifest": pair(release["manifest"])
            },
            "sources": [],
            "runtime": {
                "root_python": pair(deployment["root_python"]),
                "python": pair(deployment["frozen_python"]),
                "site_packages": deployment["site_packages"]["path"],
                "torchrun": pair(deployment["torchrun"]["source"]),
                "deployment_controller": pair(deployment["controller"]),
                "controller_authority": {
                    "receipt": pair(authority["receipt"]),
                    "authority_digest": authority["authority_digest"],
                },
                "eval_release_root": release["release_root"]["path"],
                "eval_release_archive": pair(release["archive"]),
                "eval_release_envelope": pair(release["envelope"]),
                "eval_release_manifest_digest": release["manifest_digest"],
                "eval_release_content_revision": release["content_revision"],
                "eval_release_envelope_digest": release["envelope_digest"],
            },
        }
        value["spec_digest"] = controller.object_sha256(value)
        self.source_runtime_spec.write_bytes(
            controller.canonical_json_bytes(value) + b"\n"
        )
        self.source_runtime_spec.chmod(0o444)
        return value

    def _publish_source_spec_authority(self, deployment: dict) -> dict:
        self._write_source_spec(deployment)
        return controller.publish_source_spec_authority(
            deployment_receipt_path=self.receipt_path,
            expected_deployment_receipt_sha256=sha(self.receipt_path),
            source_runtime_spec_path=self.source_runtime_spec,
            expected_source_runtime_spec_sha256=sha(self.source_runtime_spec),
        )

    def test_real_release_authority_and_verified_target_argv(self) -> None:
        receipt, captured_runtime = self._capture()
        source_authority = self._publish_source_spec_authority(receipt)
        loaded, replayed_runtime = controller.load_deployment_receipt(
            self.receipt_path, expected_sha256=sha(self.receipt_path)
        )
        self.assertEqual(loaded, receipt)
        loaded_source_authority = controller.load_source_spec_authority(
            self.source_spec_authority_path,
            expected_sha256=sha(self.source_spec_authority_path),
            deployment=loaded,
            deployment_receipt_path=self.receipt_path,
            expected_deployment_receipt_sha256=sha(self.receipt_path),
        )
        self.assertEqual(loaded_source_authority, source_authority)
        argv = controller.build_target_argv(
            loaded, replayed_runtime,
            target="action_preservation_decoded_eval_bridge_v1.py",
            arguments=[
                "--source-runtime-spec", str(self.source_runtime_spec),
                "--source-runtime-spec-sha256", sha(self.source_runtime_spec),
                "--help",
            ],
            capture_receipt_path=self.root / "bridge-capture.json",
            source_spec_authority=loaded_source_authority,
        )
        self.assertEqual(argv[1:4], ["-I", "-S", "-B"])
        self.assertNotIn("-m", argv)
        self.assertEqual(
            source_authority["source_runtime_spec"]["sha256"],
            sha(self.source_runtime_spec),
        )
        self.assertEqual(
            receipt["controller_authority"]["authority_digest"],
            captured_runtime["validate_controller_authority_binding"](
                receipt["controller_authority"],
                controller_binding=receipt["controller"],
                root_python_binding=receipt["root_python"],
                frozen_python_binding=receipt["frozen_python"],
                site_packages_binding=receipt["site_packages"],
                release_binding=receipt["release"],
                torchrun_binding=receipt["torchrun"],
                require_torchrun_continuity=True,
                verify_file=True,
            )[0]["authority_digest"],
        )

    def test_external_request_sha_and_create_only_are_mandatory(self) -> None:
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError, "SHA|bytes"
        ):
            with mock.patch.object(controller, "__file__", str(self.controller_copy)):
                controller.capture_authority(
                    request_path=self.request_path,
                    expected_request_sha256="0" * 64,
                )
        self.source_runtime_spec.write_bytes(b"{}\n")
        self.source_runtime_spec.chmod(0o444)
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError,
            "held work root.*closure|source_runtime_spec_path.*not fresh",
        ):
            self._capture()
        self.source_runtime_spec.unlink()
        self._capture()
        deployment, _ = controller.load_deployment_receipt(
            self.receipt_path, expected_sha256=sha(self.receipt_path)
        )
        self._publish_source_spec_authority(deployment)
        with self.assertRaisesRegex(RuntimeError, "fresh|overwrite|exist|collision"):
            controller.publish_source_spec_authority(
                deployment_receipt_path=self.receipt_path,
                expected_deployment_receipt_sha256=sha(self.receipt_path),
                source_runtime_spec_path=self.source_runtime_spec,
                expected_source_runtime_spec_sha256=sha(
                    self.source_runtime_spec
                ),
            )
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError,
            "not fresh|held work root.*closure",
        ):
            self._capture()

    def test_authorized_torchrun_cannot_be_replaced_and_self_recaptured(self) -> None:
        self._capture()
        self.torchrun.chmod(0o644)
        self.torchrun.write_bytes(b"# hostile replacement torchrun\n")
        with self.assertRaisesRegex(
            RuntimeError,
            "torchrun|identity|SHA",
        ):
            controller.load_deployment_receipt(
                self.receipt_path, expected_sha256=sha(self.receipt_path)
            )

    def test_source_runtime_spec_cannot_be_replaced_or_self_resigned(self) -> None:
        receipt, captured_runtime = self._capture()
        source_authority = self._publish_source_spec_authority(receipt)
        self.source_runtime_spec.chmod(0o644)
        self.source_runtime_spec.write_bytes(
            b'{"fixture":"same-uid-self-resigned-replacement"}\n'
        )
        hostile_sha = sha(self.source_runtime_spec)
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError,
            "source/runtime spec.*(identity|bytes|SHA)",
        ):
            controller.load_source_spec_authority(
                self.source_spec_authority_path,
                expected_sha256=sha(self.source_spec_authority_path),
                deployment=receipt,
                deployment_receipt_path=self.receipt_path,
                expected_deployment_receipt_sha256=sha(self.receipt_path),
            )
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError,
            "source-runtime-spec-sha256.*detached authority",
        ):
            controller.build_target_argv(
                receipt, captured_runtime,
                target="action_preservation_decoded_eval_bridge_v1.py",
                arguments=[
                    "--source-runtime-spec", str(self.source_runtime_spec),
                    "--source-runtime-spec-sha256", hostile_sha,
                ],
                capture_receipt_path=self.root / "hostile-bridge-capture.json",
                source_spec_authority=source_authority,
            )

    def test_self_resigned_spec_cannot_change_runtime_authority(self) -> None:
        receipt, _ = self._capture()
        value = self._write_source_spec(receipt)
        value["runtime"]["controller_authority"]["authority_digest"] = (
            "0" * 64
        )
        value["spec_digest"] = controller.object_sha256(
            {key: item for key, item in value.items() if key != "spec_digest"}
        )
        self.source_runtime_spec.chmod(0o644)
        self.source_runtime_spec.write_bytes(
            controller.canonical_json_bytes(value) + b"\n"
        )
        self.source_runtime_spec.chmod(0o444)
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError,
            "differs from detached runtime authority",
        ):
            controller.publish_source_spec_authority(
                deployment_receipt_path=self.receipt_path,
                expected_deployment_receipt_sha256=sha(self.receipt_path),
                source_runtime_spec_path=self.source_runtime_spec,
                expected_source_runtime_spec_sha256=sha(
                    self.source_runtime_spec
                ),
            )

    def test_work_root_rename_and_replacement_aborts_before_receipt(self) -> None:
        original = controller._controller_binding
        moved = self.work.with_name("work-renamed-by-hostile")

        def replace_root(runtime_namespace: dict, request: dict) -> dict:
            self.work.rename(moved)
            self.work.mkdir(mode=0o700)
            return original(runtime_namespace, request)

        with mock.patch.object(
            controller, "_controller_binding", side_effect=replace_root
        ):
            with self.assertRaisesRegex(
                controller.DecodedEvalDeploymentControllerError,
                "held work root",
            ):
                self._capture()
        self.assertFalse(self.receipt_path.exists())

    def test_phase_b_root_replacement_after_spec_validation_aborts(self) -> None:
        deployment, _ = self._capture()
        self._write_source_spec(deployment)
        moved = self.work.with_name("phase-b-work-renamed-by-hostile")
        original = controller._validate_source_spec_authority_continuity

        def replace_root(value: dict, receipt: dict) -> dict:
            result = original(value, receipt)
            self.work.rename(moved)
            self.work.mkdir(mode=0o700)
            return result

        with mock.patch.object(
            controller,
            "_validate_source_spec_authority_continuity",
            side_effect=replace_root,
        ):
            with self.assertRaisesRegex(
                controller.DecodedEvalDeploymentControllerError,
                "held work root",
            ):
                controller.publish_source_spec_authority(
                    deployment_receipt_path=self.receipt_path,
                    expected_deployment_receipt_sha256=sha(self.receipt_path),
                    source_runtime_spec_path=self.source_runtime_spec,
                    expected_source_runtime_spec_sha256=sha(
                        self.source_runtime_spec
                    ),
                )
        self.assertFalse(self.source_spec_authority_path.exists())
        self.assertTrue((moved / self.source_runtime_spec.name).exists())

    def test_phase_b_root_replacement_after_held_receipt_write_aborts(self) -> None:
        deployment, _ = self._capture()
        self._write_source_spec(deployment)
        moved = self.work.with_name("phase-b-published-root-renamed-by-hostile")
        original = controller._HeldWorkRoot.publish_member

        def replace_after_write(held, *args, **kwargs):
            result = original(held, *args, **kwargs)
            self.work.rename(moved)
            self.work.mkdir(mode=0o700)
            return result

        with mock.patch.object(
            controller._HeldWorkRoot,
            "publish_member",
            autospec=True,
            side_effect=replace_after_write,
        ):
            with self.assertRaisesRegex(
                controller.DecodedEvalDeploymentControllerError,
                "held work root",
            ):
                controller.publish_source_spec_authority(
                    deployment_receipt_path=self.receipt_path,
                    expected_deployment_receipt_sha256=sha(self.receipt_path),
                    source_runtime_spec_path=self.source_runtime_spec,
                    expected_source_runtime_spec_sha256=sha(
                        self.source_runtime_spec
                    ),
                )
        self.assertFalse(self.source_spec_authority_path.exists())
        self.assertTrue((moved / self.source_spec_authority_path.name).exists())

    def test_bootstrap_is_isolated_and_requires_mode_0444(self) -> None:
        command = controller.controller_bootstrap_argv(
            controller_path=self.controller_copy,
            expected_controller_sha256=sha(self.controller_copy),
            arguments=["--help"],
            root_python_path=pathlib.Path(sys.executable),
        )
        self.assertEqual(command[1:4], ["-I", "-S", "-B"])
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Detached trust root", completed.stdout)
        hostile = copy.deepcopy(command)
        hostile[7] = "0" * 64
        self.assertNotEqual(hostile, command)
        self.controller_copy.chmod(0o644)
        with self.assertRaisesRegex(
            controller.DecodedEvalDeploymentControllerError, "mode|identity"
        ):
            controller.stable_file(
                self.controller_copy, label="controller",
                expected_sha256=sha(self.controller_copy), expected_mode=0o444,
            )


class HolderCompletionAnchorContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)
        work_info = self.work.stat()
        parent_info = self.work.parent.stat()
        authority = {
            "schema_version": controller.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(self.work),
            "parent_path": str(self.work.parent),
            "creation_identity": controller._identity_row(work_info),
            "immutable_identity": controller._immutable_directory_identity(
                work_info
            ),
            "parent_immutable_identity": (
                controller._immutable_directory_identity(parent_info)
            ),
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        authority["authority_digest"] = controller.object_sha256(authority)
        self.authority = authority
        self.evaluation = self.work / "evaluation-r4"
        execution = self.evaluation / controller.EXECUTION_SHARD_DIRECTORY
        holder_root = execution / "136719"
        holder_root.mkdir(parents=True)
        summary = {
            "schema_version": "fixture-holder-summary-v1",
            "holder_job_id": "136719",
        }
        summary["summary_digest"] = controller.object_sha256(summary)
        summary_path = holder_root / controller.SHARD_SUMMARY_FILENAME
        summary_path.write_bytes(controller.canonical_json_bytes(summary) + b"\n")
        summary_path.chmod(0o444)
        completion_path = (
            execution / f"136719{controller.HOLDER_COMPLETION_SUFFIX}"
        )
        completion_path.touch(mode=0o600)
        initial = completion_path.stat()
        completion = {
            "schema_version": "fixture-holder-completion-v1",
            "evaluation_root": str(self.evaluation),
            "holder_job_id": "136719",
            "holder_summary_digest": summary["summary_digest"],
        }
        completion["completion_digest"] = controller.object_sha256(completion)
        raw = controller.canonical_json_bytes(completion) + b"\n"
        completion_path.write_bytes(raw)
        completion_path.chmod(0o444)
        self.completion_path = completion_path
        self.completion_raw = raw
        self.anchor = {
            "schema_version": controller.HOLDER_COMPLETION_ANCHOR_SCHEMA,
            "holder_job_id": "136719",
            "completion_path": str(completion_path),
            "initial_inode_identity": {
                "device": initial.st_dev, "inode": initial.st_ino,
                "uid": initial.st_uid, "gid": initial.st_gid,
                "rdev": initial.st_rdev,
            },
            "completion_sha256": hashlib.sha256(raw).hexdigest(),
            "completion_size": len(raw),
            "completion_mode": 0o444,
            "completion_digest": completion["completion_digest"],
            "holder_summary_digest": summary["summary_digest"],
        }
        self.anchor["anchor_digest"] = controller.object_sha256(self.anchor)

    def _arguments(self) -> list[str]:
        return [
            "--evaluation-root", str(self.evaluation),
            "--holder-job-id", "136719",
        ]

    def test_controller_replays_anchor_from_held_work_root(self) -> None:
        held = controller._HeldWorkRoot.open(self.authority)
        try:
            self.assertEqual(
                controller._verify_completion_anchor_from_work_root(
                    self.anchor,
                    target_arguments=self._arguments(),
                    work_root=held,
                ),
                self.anchor,
            )
        finally:
            held.close()

    def test_same_inode_rewrite_after_child_anchor_is_rejected(self) -> None:
        original_inode = self.completion_path.stat().st_ino
        self.completion_path.chmod(0o600)
        self.completion_path.write_bytes(b'{"hostile":"same-inode"}\n')
        self.completion_path.chmod(0o444)
        self.assertEqual(self.completion_path.stat().st_ino, original_inode)
        held = controller._HeldWorkRoot.open(self.authority)
        try:
            with self.assertRaisesRegex(
                controller.DecodedEvalDeploymentControllerError,
                "physical file",
            ):
                controller._verify_completion_anchor_from_work_root(
                    self.anchor,
                    target_arguments=self._arguments(),
                    work_root=held,
                )
        finally:
            held.close()

    def test_anchor_extra_field_and_wrong_holder_are_rejected(self) -> None:
        hostile = copy.deepcopy(self.anchor)
        hostile["extra"] = True
        hostile["anchor_digest"] = controller.object_sha256(
            {key: value for key, value in hostile.items()
             if key != "anchor_digest"}
        )
        with self.assertRaises(controller.DecodedEvalDeploymentControllerError):
            controller._validate_holder_completion_anchor(hostile)
        hostile = copy.deepcopy(self.anchor)
        hostile["holder_job_id"] = "136141"
        hostile["anchor_digest"] = controller.object_sha256(
            {key: value for key, value in hostile.items()
             if key != "anchor_digest"}
        )
        with self.assertRaises(controller.DecodedEvalDeploymentControllerError):
            controller._validate_holder_completion_anchor(hostile)


class AggregateCompletionAnchorContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir(mode=0o700)
        work_info = self.work.stat()
        parent_info = self.work.parent.stat()
        authority = {
            "schema_version": controller.WORK_ROOT_AUTHORITY_SCHEMA,
            "path": str(self.work),
            "parent_path": str(self.work.parent),
            "creation_identity": controller._identity_row(work_info),
            "immutable_identity": controller._immutable_directory_identity(
                work_info
            ),
            "parent_immutable_identity": (
                controller._immutable_directory_identity(parent_info)
            ),
            "initial_entries": [],
            "retained_parent_fd_through_request_publication": True,
            "retained_root_fd_through_request_publication": True,
        }
        authority["authority_digest"] = controller.object_sha256(authority)
        self.authority = authority
        self.aggregate_root = self.work / "aggregate-r4"
        media_root = self.aggregate_root / "media"
        media_root.mkdir(parents=True)

        media_raw = b"fixture mp4 bytes"
        media_sha = hashlib.sha256(media_raw).hexdigest()
        media_path = media_root / f"{media_sha}.mp4"
        media_path.write_bytes(media_raw)
        media_path.chmod(0o444)

        private = {
            "schema_version": "fixture-private-v1",
            "evaluation_id": "evaluation-r4",
        }
        private["private_mapping_digest"] = controller.object_sha256(private)
        public = {
            "schema_version": "fixture-public-v1",
            "evaluation_id": "evaluation-r4",
            "private_mapping_digest": private["private_mapping_digest"],
        }
        public["public_packet_digest"] = controller.object_sha256(public)
        aggregate = {
            "schema_version": "fixture-aggregate-v1",
            "evaluation_id": "evaluation-r4",
            "private_mapping_digest": private["private_mapping_digest"],
            "public_packet_digest": public["public_packet_digest"],
        }
        aggregate["aggregate_digest"] = controller.object_sha256(aggregate)

        document_specs = (
            ("private_blind_mapping.json", private, 0o400),
            ("blind_review_packet.json", public, 0o444),
            ("evaluation_complete.json", aggregate, 0o444),
        )
        documents: dict[str, tuple[dict, bytes, pathlib.Path]] = {}
        for name, value, mode in document_specs:
            raw = controller.canonical_json_bytes(value) + b"\n"
            path = self.aggregate_root / name
            path.write_bytes(raw)
            path.chmod(mode)
            documents[name] = (value, raw, path)

        media_root.chmod(0o555)
        self.aggregate_root.chmod(0o555)
        media_info = media_root.stat()
        media_file_info = media_path.stat()
        media_rows = [
            {
                "relative_path": f"media/{media_path.name}",
                "sha256": media_sha,
                "size": len(media_raw),
                "mode": 0o444,
                "identity": controller._identity_row(media_file_info),
            }
        ]
        media_rows_digest = controller.object_sha256(media_rows)

        def file_binding(
            name: str, digest_field: str, mode: int,
        ) -> dict:
            value, raw, path = documents[name]
            return {
                "relative_path": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "mode": mode,
                "identity": controller._identity_row(path.stat()),
                "object_digest": value[digest_field],
            }

        self.anchor = {
            "schema_version": controller.AGGREGATE_COMPLETION_ANCHOR_SCHEMA,
            "evaluation_id": "evaluation-r4",
            "aggregate_root": str(self.aggregate_root),
            "aggregate_root_identity": controller._identity_row(
                self.aggregate_root.stat()
            ),
            "aggregate_file": file_binding(
                "evaluation_complete.json", "aggregate_digest", 0o444
            ),
            "private_file": file_binding(
                "private_blind_mapping.json", "private_mapping_digest", 0o400
            ),
            "public_file": file_binding(
                "blind_review_packet.json", "public_packet_digest", 0o444
            ),
            "media_directory_identity": controller._identity_row(media_info),
            "media_file_count": 1,
            "media_rows_digest": media_rows_digest,
            "media_tree_digest": controller.object_sha256(
                {
                    "media_directory_identity": controller._identity_row(
                        media_info
                    ),
                    "media_file_count": 1,
                    "media_rows_digest": media_rows_digest,
                }
            ),
        }
        self.anchor["anchor_digest"] = controller.object_sha256(self.anchor)
        self.media_path = media_path

    def _arguments(self) -> list[str]:
        return ["--aggregate-root", str(self.aggregate_root)]

    def test_controller_replays_exact_aggregate_tree_from_held_work_root(
        self,
    ) -> None:
        held = controller._HeldWorkRoot.open(self.authority)
        try:
            self.assertEqual(
                controller._verify_aggregate_anchor_from_work_root(
                    self.anchor,
                    target_arguments=self._arguments(),
                    work_root=held,
                ),
                self.anchor,
            )
        finally:
            held.close()

    def test_media_same_inode_rewrite_after_anchor_is_rejected(self) -> None:
        original_inode = self.media_path.stat().st_ino
        self.media_path.chmod(0o600)
        self.media_path.write_bytes(b"hostile replacement bytes")
        self.media_path.chmod(0o444)
        self.assertEqual(self.media_path.stat().st_ino, original_inode)
        held = controller._HeldWorkRoot.open(self.authority)
        try:
            with self.assertRaises(
                controller.DecodedEvalDeploymentControllerError
            ):
                controller._verify_aggregate_anchor_from_work_root(
                    self.anchor,
                    target_arguments=self._arguments(),
                    work_root=held,
                )
        finally:
            held.close()

    def test_resigned_media_tree_digest_without_rows_is_rejected(self) -> None:
        hostile = copy.deepcopy(self.anchor)
        hostile["media_rows_digest"] = hashlib.sha256(
            b"forged media rows"
        ).hexdigest()
        hostile["media_tree_digest"] = controller.object_sha256(
            {
                "media_directory_identity": hostile[
                    "media_directory_identity"
                ],
                "media_file_count": hostile["media_file_count"],
                "media_rows_digest": hostile["media_rows_digest"],
            }
        )
        hostile["anchor_digest"] = controller.object_sha256(
            {
                key: value for key, value in hostile.items()
                if key != "anchor_digest"
            }
        )
        held = controller._HeldWorkRoot.open(self.authority)
        try:
            with self.assertRaisesRegex(
                controller.DecodedEvalDeploymentControllerError,
                "media tree digest",
            ):
                controller._verify_aggregate_anchor_from_work_root(
                    hostile,
                    target_arguments=self._arguments(),
                    work_root=held,
                )
        finally:
            held.close()


if __name__ == "__main__":
    unittest.main()

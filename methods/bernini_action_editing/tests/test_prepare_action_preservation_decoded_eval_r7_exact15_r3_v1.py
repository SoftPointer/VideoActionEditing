from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for import_root in (TOOLS, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import action_preservation_decoded_eval_deployment_controller_v1 as controller
import prepare_action_preservation_decoded_eval_r7_exact15_r3_v1 as prepare


def binding(path: Path, digest: str = "1" * 64) -> dict:
    return {"path": str(path), "sha256": digest}


def work_root_authority(work: Path) -> dict:
    info = work.lstat()
    parent = work.parent.lstat()
    value = {
        "schema_version": (
            "bernini-action-preservation-decoded-eval-work-root-authority-v1"
        ),
        "path": str(work),
        "parent_path": str(work.parent),
        "creation_identity": prepare._identity_row(info),
        "immutable_identity": prepare._immutable_directory_identity(info),
        "parent_immutable_identity": prepare._immutable_directory_identity(
            parent
        ),
        "initial_entries": [],
        "retained_parent_fd_through_request_publication": True,
        "retained_root_fd_through_request_publication": True,
    }
    value["authority_digest"] = prepare.object_sha256(value)
    return value


class R7Exact15R3DeploymentPreparationTests(unittest.TestCase):
    def test_phase_a_request_is_fresh_closed_and_r3_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "fresh-work"
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized",
                "DEPLOYMENT_REQUEST_PATH": work / "request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "spec-authority.json",
            }
            namespace = {
                "validate_request": controller.validate_request,
                "ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap",
            }
            rows = {
                key: binding(work / key)
                for key in (
                    "archive", "manifest", "envelope", "runtime_source",
                    "controller", "root_python", "frozen_python", "torchrun",
                )
            }
            with mock.patch.multiple(prepare, **paths), mock.patch.object(
                prepare, "_validate_bundle_and_static_inputs", return_value=rows
            ), mock.patch.object(
                prepare, "_controller_namespace", return_value=namespace
            ):
                authority = prepare._PhaseAWorkAuthority.create()
                try:
                    request, observed = prepare.build_phase_a_request(
                        work_authority=authority
                    )
                finally:
                    authority.close()

        self.assertIs(observed, namespace)
        self.assertEqual(controller.validate_request(request), request)
        self.assertEqual(request["release_generation"], prepare.RELEASE_GENERATION)
        self.assertEqual(request["archive"]["sha256"], "1" * 64)
        self.assertEqual(request["manifest_digest"], prepare.MANIFEST_DIGEST)
        self.assertEqual(request["content_revision"], prepare.CONTENT_REVISION)
        unsigned = dict(request)
        claimed = unsigned.pop("request_digest")
        self.assertEqual(claimed, prepare.object_sha256(unsigned))

    def test_phase_a_fresh_creates_work_root_and_exact_request_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "fresh-work"
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized-release",
                "DEPLOYMENT_REQUEST_PATH": work / "deployment-request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "source-runtime-spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "source-spec-authority.json",
            }
            controller_copy = parent / "captured-controller.py"
            shutil.copyfile(Path(controller.__file__), controller_copy)
            controller_copy.chmod(0o444)
            controller_sha = hashlib.sha256(
                controller_copy.read_bytes()
            ).hexdigest()
            rows = {
                key: binding(parent / key)
                for key in (
                    "archive", "manifest", "envelope", "runtime_source",
                    "controller", "root_python", "frozen_python", "torchrun",
                )
            }
            rows["controller"] = binding(controller_copy, controller_sha)
            with mock.patch.multiple(prepare, **paths), mock.patch.object(
                prepare, "_validate_bundle_and_static_inputs", return_value=rows
            ), mock.patch.object(
                prepare, "CONTROLLER_PATH", controller_copy
            ), mock.patch.object(
                prepare, "CONTROLLER_SHA256", controller_sha
            ):
                result = prepare.publish_phase_a_request()

            self.assertEqual(
                sorted(item.name for item in work.iterdir()),
                ["deployment-request.json"],
            )
            self.assertEqual(result["work_root_initial"]["entries"], [])
            self.assertEqual(
                result["work_root_after_request"]["entries"],
                ["deployment-request.json"],
            )
            self.assertEqual(
                set(result["deployment_request"]),
                {"path", "sha256", "size", "mode"},
            )
            self.assertEqual(result["deployment_request"]["mode"], 0o444)

    def test_phase_a_rejects_preexisting_or_nonempty_work_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "poisoned-work"
            work.mkdir(mode=0o700)
            (work / "foreign-entry").write_bytes(b"hostile")
            with mock.patch.object(prepare, "WORK_ROOT", work):
                with self.assertRaisesRegex(
                    prepare.R7DeploymentPreparationError,
                    "work root is not fresh",
                ):
                    prepare.publish_phase_a_request()

    def test_phase_a_root_rename_replacement_cannot_change_request_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "fresh-work"
            moved = parent / "renamed-held-work"
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized-release",
                "DEPLOYMENT_REQUEST_PATH": work / "deployment-request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "source-runtime-spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "source-spec-authority.json",
            }
            rows = {
                key: binding(parent / key)
                for key in (
                    "archive", "manifest", "envelope", "runtime_source",
                    "controller", "root_python", "frozen_python", "torchrun",
                )
            }
            namespace = {
                "validate_request": controller.validate_request,
                "ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap",
            }
            original = prepare._write_create_only_at

            def replace_after_write(authority, path, value, **kwargs):
                result = original(authority, path, value, **kwargs)
                work.rename(moved)
                work.mkdir(mode=0o700)
                return result

            with mock.patch.multiple(prepare, **paths), mock.patch.object(
                prepare, "_validate_bundle_and_static_inputs", return_value=rows
            ), mock.patch.object(
                prepare, "_controller_namespace", return_value=namespace
            ), mock.patch.object(
                prepare, "_write_create_only_at", side_effect=replace_after_write
            ):
                with self.assertRaisesRegex(
                    prepare.R7DeploymentPreparationError,
                    "held work root",
                ):
                    prepare.publish_phase_a_request()
            self.assertFalse((work / "deployment-request.json").exists())
            self.assertTrue((moved / "deployment-request.json").exists())

    def test_phase_a_completion_requires_exact_four_entry_closure(self) -> None:
        work_root_authority = {"authority_digest": "9" * 64}
        request = {
            "request_digest": "8" * 64,
            "work_root_authority": work_root_authority,
        }
        request_raw = prepare.canonical_json_bytes(request) + b"\n"
        request_file = {
            "path": str(prepare.DEPLOYMENT_REQUEST_PATH),
            "sha256": "7" * 64,
        }
        deployment = {
            "release_generation": prepare.RELEASE_GENERATION,
            "work_root_authority": work_root_authority,
            "work_root_expected_phase_a_entries": sorted({
                prepare.DEPLOYMENT_REQUEST_PATH.name,
                prepare.MATERIALIZED_RELEASE_ROOT.name,
                prepare.CONTROLLER_AUTHORITY_PATH.name,
                prepare.DEPLOYMENT_RECEIPT_PATH.name,
            }),
            "work_root_held_fd_through_controller_publication": True,
            "deployment_request": request_file,
            "deployment_request_digest": request["request_digest"],
            "release": {
                "release_root": {"path": str(prepare.MATERIALIZED_RELEASE_ROOT)}
            },
            "source_runtime_spec_path": str(prepare.SOURCE_SPEC_PATH),
            "source_spec_authority_receipt_path": str(
                prepare.SOURCE_SPEC_AUTHORITY_PATH
            ),
            "controller_authority": {
                "receipt": {"path": str(prepare.CONTROLLER_AUTHORITY_PATH)}
            },
        }
        namespace = {
            "load_deployment_receipt": lambda *args, **kwargs: (deployment, {})
        }
        with mock.patch.object(
            prepare, "_directory_entries",
            return_value={"entries": sorted({
                prepare.DEPLOYMENT_REQUEST_PATH.name,
                prepare.MATERIALIZED_RELEASE_ROOT.name,
                prepare.CONTROLLER_AUTHORITY_PATH.name,
                prepare.DEPLOYMENT_RECEIPT_PATH.name,
            })},
        ) as closure, mock.patch.object(
            prepare, "_controller_namespace", return_value=namespace
        ), mock.patch.object(
            prepare, "stable_file", return_value=(request_raw, request_file)
        ):
            result = prepare.validate_phase_a_completion(
                deployment_receipt_sha256="6" * 64
            )
        expected = {
            prepare.DEPLOYMENT_REQUEST_PATH.name,
            prepare.MATERIALIZED_RELEASE_ROOT.name,
            prepare.CONTROLLER_AUTHORITY_PATH.name,
            prepare.DEPLOYMENT_RECEIPT_PATH.name,
        }
        closure.assert_called_once_with(prepare.WORK_ROOT, expected=expected)
        self.assertEqual(
            result["status"], "R7_EXACT15_R3_PHASE_A_COMPLETION_VERIFIED"
        )

    def test_phase_b_spec_binds_new_preprocessing_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            spec_path = root / "source-runtime-spec.json"
            release_root = root / "materialized"
            method_root = release_root / "methods/bernini_action_editing"
            authority_receipt = root / "controller-authority.json"
            deployment = {
                "release_generation": prepare.RELEASE_GENERATION,
                "source_runtime_spec_path": str(spec_path),
                "source_spec_authority_receipt_path": str(root / "source-authority.json"),
                "root_python": binding(root / "root-python"),
                "frozen_python": binding(root / "python"),
                "site_packages": {"path": str(root / "site-packages")},
                "torchrun": {"source": binding(root / "run.py")},
                "controller": binding(root / "controller.py"),
                "controller_authority": {
                    "receipt": binding(authority_receipt),
                    "authority_digest": "2" * 64,
                },
                "release": {
                    "release_root": {"path": str(release_root)},
                    "archive": binding(root / "source.tar", prepare.ARCHIVE_SHA256),
                    "manifest": binding(root / "source.manifest.json", prepare.MANIFEST_SHA256),
                    "envelope": binding(root / "envelope.json", prepare.ENVELOPE_SHA256),
                    "manifest_digest": prepare.MANIFEST_DIGEST,
                    "content_revision": prepare.CONTENT_REVISION,
                    "envelope_digest": prepare.ENVELOPE_DIGEST,
                },
            }
            namespace = {"ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap"}
            source = {
                "iid": "7b88a1ca1f804f41",
                "source_video_path": str(root / "video.mp4"),
                "source_video_sha256": "3" * 64,
                "source_receipt_path": str(root / "receipt.json"),
                "source_receipt_sha256": "4" * 64,
                "instruction": "Perform the fixed action.",
                "instruction_sha256": "5" * 64,
                "action_review_contract": {},
                "seed": 2026081801,
            }
            preprocessing = {"sources": [source]}
            preprocessing_binding = binding(
                root / "preprocessing.json", prepare.SOURCE_PREPROCESSING_SHA256
            )

            def stable(path: Path, *, label: str, expected_sha256: str,
                       expected_mode: int | None = None):
                return b"fixture", binding(Path(path), expected_sha256)

            with mock.patch.multiple(
                prepare,
                SOURCE_SPEC_PATH=spec_path,
                SOURCE_SPEC_AUTHORITY_PATH=root / "source-authority.json",
                DEPLOYMENT_RECEIPT_PATH=root / "deployment-receipt.json",
                MATERIALIZED_RELEASE_ROOT=release_root,
            ), mock.patch.object(
                prepare, "_load_source_preprocessing",
                return_value=(preprocessing, preprocessing_binding),
            ), mock.patch.object(prepare, "stable_file", side_effect=stable):
                authority = mock.Mock()
                spec, _ = prepare._build_phase_b_spec_from_context(
                    deployment=deployment,
                    namespace=namespace,
                    work_authority=authority,
                )

        self.assertEqual(
            spec["pins"]["source_preprocessing_sha256"],
            prepare.SOURCE_PREPROCESSING_SHA256,
        )
        self.assertEqual(
            spec["pin_files"]["source_preprocessing"], preprocessing_binding
        )
        self.assertEqual(spec["sources"], [source])
        self.assertEqual(spec["runtime"]["method_source_revision"],
                         prepare.SOURCE_REVISION)
        unsigned = dict(spec)
        claimed = unsigned.pop("spec_digest")
        self.assertEqual(claimed, prepare.object_sha256(unsigned))

    def test_phase_b_publishes_relative_to_held_exact_four_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "work"
            work.mkdir(mode=0o700)
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized-release",
                "DEPLOYMENT_REQUEST_PATH": work / "deployment-request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "source-runtime-spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "source-spec-authority.json",
            }
            authority_value = work_root_authority(work)
            paths["MATERIALIZED_RELEASE_ROOT"].mkdir(mode=0o700)
            for path in (
                paths["DEPLOYMENT_REQUEST_PATH"],
                paths["CONTROLLER_AUTHORITY_PATH"],
                paths["DEPLOYMENT_RECEIPT_PATH"],
            ):
                path.write_bytes(b"fixture\n")
                path.chmod(0o444)
            expected_four = {
                path.name
                for path in (
                    paths["MATERIALIZED_RELEASE_ROOT"],
                    paths["DEPLOYMENT_REQUEST_PATH"],
                    paths["CONTROLLER_AUTHORITY_PATH"],
                    paths["DEPLOYMENT_RECEIPT_PATH"],
                )
            }
            deployment = {
                "release_generation": prepare.RELEASE_GENERATION,
                "work_root_authority": authority_value,
                "work_root_expected_phase_a_entries": sorted(expected_four),
                "source_runtime_spec_path": str(paths["SOURCE_SPEC_PATH"]),
                "source_spec_authority_receipt_path": str(
                    paths["SOURCE_SPEC_AUTHORITY_PATH"]
                ),
            }
            namespace = {
                "load_deployment_receipt": lambda *args, **kwargs: (
                    deployment, {}
                ),
                "ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap",
            }
            spec = {"schema_version": "fixture", "spec_digest": "7" * 64}
            with mock.patch.multiple(prepare, **paths), mock.patch.object(
                prepare, "_controller_namespace", return_value=namespace
            ), mock.patch.object(
                prepare, "_build_phase_b_spec_from_context",
                return_value=(spec, namespace),
            ):
                result = prepare.publish_phase_b_spec(
                    deployment_receipt_sha256="6" * 64
                )
            self.assertEqual(
                sorted(item.name for item in work.iterdir()),
                sorted(expected_four | {paths["SOURCE_SPEC_PATH"].name}),
            )
            self.assertEqual(
                result["work_root_after_source_spec"]["entries"],
                sorted(expected_four | {paths["SOURCE_SPEC_PATH"].name}),
            )
            self.assertEqual(
                set(result["source_runtime_spec"]),
                {"path", "sha256", "size", "mode"},
            )
            self.assertEqual(result["source_runtime_spec"]["mode"], 0o444)

    def test_phase_b_root_rename_replacement_aborts_after_held_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            work = parent / "work"
            moved = parent / "renamed-held-work"
            work.mkdir(mode=0o700)
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized-release",
                "DEPLOYMENT_REQUEST_PATH": work / "deployment-request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "source-runtime-spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "source-spec-authority.json",
            }
            authority_value = work_root_authority(work)
            paths["MATERIALIZED_RELEASE_ROOT"].mkdir(mode=0o700)
            for path in (
                paths["DEPLOYMENT_REQUEST_PATH"],
                paths["CONTROLLER_AUTHORITY_PATH"],
                paths["DEPLOYMENT_RECEIPT_PATH"],
            ):
                path.write_bytes(b"fixture\n")
                path.chmod(0o444)
            expected_four = {
                path.name
                for path in (
                    paths["MATERIALIZED_RELEASE_ROOT"],
                    paths["DEPLOYMENT_REQUEST_PATH"],
                    paths["CONTROLLER_AUTHORITY_PATH"],
                    paths["DEPLOYMENT_RECEIPT_PATH"],
                )
            }
            deployment = {
                "release_generation": prepare.RELEASE_GENERATION,
                "work_root_authority": authority_value,
                "work_root_expected_phase_a_entries": sorted(expected_four),
                "source_runtime_spec_path": str(paths["SOURCE_SPEC_PATH"]),
                "source_spec_authority_receipt_path": str(
                    paths["SOURCE_SPEC_AUTHORITY_PATH"]
                ),
            }
            namespace = {
                "load_deployment_receipt": lambda *args, **kwargs: (
                    deployment, {}
                ),
                "ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap",
            }
            original = prepare._write_create_only_at

            def replace_after_write(authority, path, value, **kwargs):
                result = original(authority, path, value, **kwargs)
                work.rename(moved)
                work.mkdir(mode=0o700)
                return result

            with mock.patch.multiple(prepare, **paths), mock.patch.object(
                prepare, "_controller_namespace", return_value=namespace
            ), mock.patch.object(
                prepare, "_build_phase_b_spec_from_context",
                return_value=(
                    {"schema_version": "fixture", "spec_digest": "7" * 64},
                    namespace,
                ),
            ), mock.patch.object(
                prepare, "_write_create_only_at", side_effect=replace_after_write
            ):
                with self.assertRaisesRegex(
                    prepare.R7DeploymentPreparationError, "held work root"
                ):
                    prepare.publish_phase_b_spec(
                        deployment_receipt_sha256="6" * 64
                    )
            self.assertFalse(paths["SOURCE_SPEC_PATH"].exists())
            self.assertTrue((moved / paths["SOURCE_SPEC_PATH"].name).exists())

    def test_interfaces_use_successor_targets_and_literal_paths(self) -> None:
        namespace = {"ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap"}
        with mock.patch.object(prepare, "_controller_namespace", return_value=namespace):
            bridge = prepare.bridge_interface(
                deployment_receipt_sha256="1" * 64,
                source_spec_authority_sha256="2" * 64,
                source_runtime_spec_sha256="3" * 64,
            )
            launcher = prepare.launcher_interface(
                deployment_receipt_sha256="1" * 64,
                source_spec_authority_sha256="2" * 64,
                physical_bindings_sha256="4" * 64,
            )
            aggregate = prepare.aggregate_interface(
                deployment_receipt_sha256="1" * 64,
                source_spec_authority_sha256="2" * 64,
                physical_bindings_sha256="4" * 64,
            )

        self.assertIn("action_preservation_decoded_eval_bridge_v1.py", bridge["argv"])
        self.assertIn(prepare.TRAINING_COMPLETE_SHA256, bridge["argv"])
        self.assertTrue(
            any(
                item.endswith("/action_preservation_decoded_eval_executor_v2.py")
                for item in launcher["argv"]
            )
        )
        self.assertIn(prepare.EXECUTOR_SHA256, launcher["argv"])
        self.assertTrue(
            any(item.endswith("/bridge/physical_bindings.json")
                for item in launcher["argv"])
        )
        self.assertIn("action_preservation_decoded_eval_aggregate_v2.py",
                      aggregate["argv"])
        for result in (bridge, launcher, aggregate):
            self.assertEqual(result["argv"][:2], ["/usr/bin/env", "-i"])


if __name__ == "__main__":
    unittest.main()

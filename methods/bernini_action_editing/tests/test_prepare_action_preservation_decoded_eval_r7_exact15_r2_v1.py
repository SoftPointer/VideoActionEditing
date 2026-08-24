from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import prepare_action_preservation_decoded_eval_r7_exact15_r2_v1 as prepare


def binding(path: Path, digest: str = "1" * 64) -> dict:
    return {"path": str(path), "sha256": digest}


class R7Exact15R2DeploymentPreparationTests(unittest.TestCase):
    def test_phase_a_request_is_fresh_closed_and_r2_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve()
            paths = {
                "WORK_ROOT": work,
                "MATERIALIZED_RELEASE_ROOT": work / "materialized",
                "DEPLOYMENT_REQUEST_PATH": work / "request.json",
                "CONTROLLER_AUTHORITY_PATH": work / "controller-authority.json",
                "DEPLOYMENT_RECEIPT_PATH": work / "deployment-receipt.json",
                "SOURCE_SPEC_PATH": work / "spec.json",
                "SOURCE_SPEC_AUTHORITY_PATH": work / "spec-authority.json",
            }
            captured = {}

            def validate_request(value: dict) -> dict:
                captured.update(value)
                return value

            namespace = {
                "validate_request": validate_request,
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
                request, observed = prepare.build_phase_a_request()

        self.assertIs(observed, namespace)
        self.assertEqual(request, captured)
        self.assertEqual(request["release_generation"], prepare.RELEASE_GENERATION)
        self.assertEqual(request["archive"]["sha256"], "1" * 64)
        self.assertEqual(request["manifest_digest"], prepare.MANIFEST_DIGEST)
        self.assertEqual(request["content_revision"], prepare.CONTENT_REVISION)
        unsigned = dict(request)
        claimed = unsigned.pop("request_digest")
        self.assertEqual(claimed, prepare.object_sha256(unsigned))

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
            namespace = {
                "load_deployment_receipt": lambda *args, **kwargs: (deployment, {}),
                "ROOT_CONTROLLER_BOOTSTRAP_SOURCE": "bootstrap",
            }
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
                prepare, "_controller_namespace", return_value=namespace
            ), mock.patch.object(
                prepare, "_load_source_preprocessing",
                return_value=(preprocessing, preprocessing_binding),
            ), mock.patch.object(prepare, "stable_file", side_effect=stable):
                spec, _ = prepare.build_phase_b_spec(
                    deployment_receipt_sha256="6" * 64
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
                physical_bindings_sha256="4" * 64,
            )
            aggregate = prepare.aggregate_interface(
                deployment_receipt_sha256="1" * 64,
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

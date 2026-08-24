from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = METHOD_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_full30_action_training_release_v1 as builder


class Full30ActionTrainingReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="full30-action-release-test-")).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def _paths(self, label: str) -> tuple[Path, Path]:
        root = self.temporary / label
        root.mkdir()
        return root / "source.tar", root / "source.manifest.json"

    def _build(self, label: str):
        archive, manifest = self._paths(label)
        result = builder.build(METHOD_ROOT, archive, manifest)
        return result, archive, manifest

    def test_builder_members_equal_trainer_required_release_files(self) -> None:
        tree = ast.parse((METHOD_ROOT / "train_full30_action_lora_v1.py").read_text())
        observed = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "REQUIRED_RELEASE_FILES" for target in node.targets):
                continue
            self.assertIsInstance(node.value, ast.Call)
            observed = frozenset(ast.literal_eval(node.value.args[0]))
        self.assertIsNotNone(observed)
        self.assertEqual(observed, builder.RELEASE_FILE_SET)
        self.assertEqual(tuple(sorted(builder.RELEASE_FILES)), builder.RELEASE_FILES)

    def test_two_builds_are_byte_identical_and_audit_passes(self) -> None:
        first, archive1, manifest1 = self._build("one")
        second, archive2, manifest2 = self._build("two")
        self.assertEqual(archive1.read_bytes(), archive2.read_bytes())
        self.assertEqual(manifest1.read_bytes(), manifest2.read_bytes())
        self.assertEqual(first["release_sha256"], second["release_sha256"])
        audit = builder.audit(
            archive1,
            manifest1,
            expected_archive_sha256=first["archive_sha256"],
            expected_manifest_sha256=first["manifest_sha256"],
        )
        self.assertTrue(audit["audit_passed"])
        self.assertFalse(audit["launch_authorized"])
        self.assertEqual(audit["file_count"], 26)

    def test_extracted_tree_is_accepted_by_the_actual_trainer_validator(self) -> None:
        result, archive, manifest = self._build("trainer")
        extracted = self.temporary / "extracted"
        extracted.mkdir()
        with tarfile.open(archive, "r:") as handle:
            handle.extractall(extracted)
        executed_root = extracted / builder.MEMBER_ROOT
        module_name = "_full30_release_validator_test"
        spec = importlib.util.spec_from_file_location(
            module_name, executed_root / "train_full30_action_lora_v1.py"
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
            receipt = module.validate_executed_release_v1(
                method_root=executed_root,
                manifest=manifest,
                expected_manifest_sha256=result["manifest_sha256"],
                expected_release_sha256=result["release_sha256"],
                test_only_require_current_entrypoint=False,
            )
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            sys.modules.pop(module_name, None)
        self.assertEqual(receipt["file_count"], 26)
        self.assertTrue(receipt["exact_member_closure_verified"])

    def test_resigned_manifest_cannot_hide_archive_payload_tamper(self) -> None:
        result, archive, manifest = self._build("tamper")
        value = json.loads(manifest.read_text())
        value["files"][0]["sha256"] = "0" * 64
        release_payload = {
            "schema_version": builder.SCHEMA_VERSION,
            "files": value["files"],
        }
        value["release_sha256"] = builder.object_sha256(release_payload)
        unsigned = dict(value)
        unsigned.pop("manifest_digest")
        value["manifest_digest"] = builder.object_sha256(unsigned)
        resigned = self.temporary / "resigned.json"
        resigned.write_bytes(builder.canonical_json_bytes(value) + b"\n")
        with self.assertRaisesRegex(builder.Full30ActionReleaseError, "archive member differs"):
            builder.audit(
                archive,
                resigned,
                expected_archive_sha256=result["archive_sha256"],
                expected_manifest_sha256=hashlib.sha256(resigned.read_bytes()).hexdigest(),
            )

    def test_archive_extra_member_and_noncanonical_source_are_rejected(self) -> None:
        result, archive, manifest = self._build("extra")
        raw = io.BytesIO()
        with tarfile.open(archive, "r:") as source, tarfile.open(
            fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT
        ) as target:
            for member in source.getmembers():
                target.addfile(member, source.extractfile(member))
            payload = b"forbidden\n"
            info = builder._tar_info(f"{builder.MEMBER_ROOT}/extra.py", len(payload))
            target.addfile(info, io.BytesIO(payload))
        with self.assertRaisesRegex(builder.Full30ActionReleaseError, "archive member closure"):
            builder.validate_archive_bytes(raw.getvalue(), json.loads(manifest.read_text()))

        copied = self.temporary / "method"
        copied.mkdir()
        for relative in builder.RELEASE_FILES:
            shutil.copy2(METHOD_ROOT / relative, copied / relative)
        (copied / builder.RELEASE_FILES[0]).unlink()
        (copied / builder.RELEASE_FILES[0]).symlink_to(METHOD_ROOT / builder.RELEASE_FILES[0])
        with self.assertRaisesRegex(builder.Full30ActionReleaseError, "non-symlink"):
            builder.build_manifest(copied)

        with self.assertRaises(builder.Full30ActionReleaseError):
            builder.build(METHOD_ROOT, archive, manifest)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), result["archive_sha256"])

    def test_trainer_declared_member_drift_is_rejected_by_the_builder(self) -> None:
        copied = self.temporary / "drifted-method"
        copied.mkdir()
        for relative in builder.RELEASE_FILES:
            shutil.copy2(METHOD_ROOT / relative, copied / relative)
        trainer = copied / "train_full30_action_lora_v1.py"
        source = trainer.read_text()
        needle = '        "motion_residual.py",\n'
        self.assertIn(needle, source)
        trainer.write_text(source.replace(needle, "", 1))
        with self.assertRaisesRegex(
            builder.Full30ActionReleaseError,
            "builder/trainer required-release closure differs",
        ):
            builder.build_manifest(copied)


if __name__ == "__main__":
    unittest.main()

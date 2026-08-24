from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "methods/bernini_action_editing/tools/build_elal3_c2_exact16_materializer_release_v1.py"
LAUNCHER_PATH = REPO_ROOT / "methods/bernini_action_editing/scripts/auh_run_elal3_c2_exact16_materializer_release_v1.sh"
CONTROLLER_PATH = REPO_ROOT / "methods/bernini_action_editing/scripts/auh_control_elal3_c2_exact16_materializer_v1.sh"
spec = importlib.util.spec_from_file_location("c2_release_builder", MODULE_PATH)
assert spec and spec.loader
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ELAL3C2MaterializerReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        local = REPO_ROOT / "methods/bernini_action_editing/train_lora.py"
        raw = local.read_bytes()
        # The reviewed remote C1 file is deliberately not the local eae8 file.
        cls.remote_source = Path("/tmp/elal3-c2-train_lora-630c.py")
        if not cls.remote_source.exists():
            raise unittest.SkipTest("reviewed remote train_lora fixture is unavailable")
        if hashlib.sha256(raw).hexdigest() == subject.RUNTIME_PINS["methods/bernini_action_editing/train_lora.py"][0]:
            raise AssertionError("test must distinguish local eae8 from reviewed remote 630c")

    def test_fresh_two_payloads_are_byte_identical_and_exact9(self) -> None:
        first_archive, first_manifest = subject.build_payload(REPO_ROOT, self.remote_source)
        second_archive, second_manifest = subject.build_payload(REPO_ROOT, self.remote_source)
        self.assertEqual(first_archive, second_archive)
        self.assertEqual(subject.canonical(first_manifest), subject.canonical(second_manifest))
        self.assertEqual(first_manifest["file_count"], 9)
        self.assertEqual(first_manifest["archive_sha256"], hashlib.sha256(first_archive).hexdigest())
        unsigned = dict(first_manifest); stored = unsigned.pop("manifest_digest")
        self.assertEqual(stored, subject.digest(unsigned))
        with tarfile.open(fileobj=io.BytesIO(first_archive), mode="r:") as source:
            members = source.getmembers()
        self.assertEqual([row.name for row in members], sorted((row.name for row in members), key=lambda value: value.encode("ascii")))
        self.assertTrue(all(row.mode == 0o444 and row.uid == 0 and row.gid == 0 and row.mtime == 0 for row in members))
        train_row = next(row for row in first_manifest["files"] if row["path"].endswith("/train_lora.py"))
        self.assertEqual(train_row["sha256"], "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5")
        materializer_row = next(row for row in first_manifest["files"] if row["path"].endswith("/materialize_elal3_simulator_c2_vae_v1.py"))
        self.assertEqual(materializer_row["sha256"], subject.MATERIALIZER_SHA256)

    def test_wrong_remote_train_lora_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "train_lora.py"
            wrong.write_bytes(self.remote_source.read_bytes() + b"\n")
            with self.assertRaisesRegex(subject.ReleaseError, "SHA/size differs"):
                subject.build_payload(REPO_ROOT, wrong.resolve())

    def test_publish_is_create_only_and_root_is_0555(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "release"
            result = subject.publish(REPO_ROOT, self.remote_source, output.resolve())
            self.assertEqual(result["archive_sha256"], hashlib.sha256((output / "source.tar").read_bytes()).hexdigest())
            self.assertEqual((output.stat().st_mode & 0o777), 0o555)
            self.assertEqual(((output / "source.tar").stat().st_mode & 0o777), 0o444)
            with self.assertRaisesRegex(subject.ReleaseError, "fresh absolute"):
                subject.publish(REPO_ROOT, self.remote_source, output.resolve())

    def test_launcher_pins_current_release_and_outer_completion(self) -> None:
        archive, manifest = subject.build_payload(REPO_ROOT, self.remote_source)
        manifest_raw = subject.canonical(manifest) + b"\n"
        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        for literal in (
            f'archive_sha="{hashlib.sha256(archive).hexdigest()}"',
            f'manifest_sha="{hashlib.sha256(manifest_raw).hexdigest()}"',
            f'materializer_sha="{subject.MATERIALIZER_SHA256}"',
            'train_lora_sha="630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"',
            '"fresh_runtime_extract_file_mode": "0644_required_by_consumer"',
            '"materializer_internal_pre_post_final_replay_passed": True',
            'if tensor_sha(tensor) != row.get("tensor_sha256")',
            'if receipt_raw != canonical(receipt) + b"\\n"',
            'if source_binding.get("source_count") != 6',
            '"RUN_COMPLETE.json"',
        ):
            self.assertIn(literal, launcher)

    def test_external_controller_literal_pins_launcher_and_release(self) -> None:
        launcher_sha = hashlib.sha256(LAUNCHER_PATH.read_bytes()).hexdigest()
        controller = CONTROLLER_PATH.read_text(encoding="utf-8")
        self.assertIn(f'expected_launcher_sha="{launcher_sha}"', controller)
        self.assertIn('expected_archive_sha="143e99cfbbafe470f008a3be6cf3a23412ddc0fe3d7e5b41f161c7faa097fce6"', controller)
        self.assertIn('expected_manifest_sha="47a8c1ef2dd1805da91af4eed65868ff668dbec7950449cbeec0bf6814e3f687"', controller)
        self.assertIn('ELAL3_C2_CONTROLLER_SHA256', controller)
        self.assertIn('exec bash "${launcher}"', controller)
        self.assertIn('[[ -d "${release_root}" && ! -L "${release_root}"', controller)
        self.assertNotIn("stat -c '%a:%h' \"${release_root}\"", controller)


if __name__ == "__main__":
    unittest.main()

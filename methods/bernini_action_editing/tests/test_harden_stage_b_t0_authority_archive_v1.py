from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "harden_stage_b_t0_authority_archive_v1.py"
)
SPEC = importlib.util.spec_from_file_location("harden_authority_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HardenAuthorityArchiveTest(unittest.TestCase):
    def test_apply_preserves_gzip_bytes_but_replaces_raw_json_with_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive = root / "authority_revocation"
            originals = archive / "originals" / "source_retry5"
            originals.mkdir(parents=True)
            raw = originals / "stage_b_t0_single_update_retry5_authority_addendum.json"
            original = json.dumps(
                {"activation": {"state": MODULE.ACTIVE_STATE}, "revision": "retry5"},
                sort_keys=True,
            ).encode("ascii")
            raw.write_bytes(original)
            os.chmod(raw, 0o400)
            runtime = root / "source_retry5" / raw.name
            runtime.parent.mkdir()
            runtime.write_text(
                json.dumps(
                    {"activation": {"state": "REVOKED_SUPERSEDED_BY_RETRY6"}}
                ),
                encoding="ascii",
            )
            manifest = {
                "schema_version": "bernini-stage-b-t0-authority-revocation-v1",
                "mode": "applied",
                "experiment_root": str(root),
                "archive_dir": str(archive),
                "kept_active_authority_sha256": "6" * 64,
                "superseded_active_count": 1,
                "applied": [
                    {
                        "path": str(runtime),
                        "archive_path": str(raw),
                        "original_sha256": hashlib.sha256(original).hexdigest(),
                    }
                ],
            }
            manifest_path = archive / "manifest.json"
            manifest_payload = json.dumps(manifest).encode("ascii")
            manifest_path.write_bytes(manifest_payload)
            os.chmod(manifest_path, 0o400)
            manifest_sha = hashlib.sha256(manifest_payload).hexdigest()

            dry = MODULE.harden(
                experiment_root=root,
                archive_dir=archive,
                revocation_manifest_sha256=manifest_sha,
                apply=False,
            )
            self.assertEqual(dry["member_count"], 1)
            self.assertEqual(raw.read_bytes(), original)

            receipt = MODULE.harden(
                experiment_root=root,
                archive_dir=archive,
                revocation_manifest_sha256=manifest_sha,
                apply=True,
            )
            self.assertEqual(receipt["member_count"], 1)
            pointer = json.loads(raw.read_text(encoding="ascii"))
            self.assertEqual(pointer["activation"]["state"], MODULE.POINTER_STATE)
            self.assertFalse(pointer["activation"]["optimizer_creation_authorized"])
            gzip_path = Path(pointer["original_authority"]["gzip_path"])
            self.assertEqual(gzip.decompress(gzip_path.read_bytes()), original)
            self.assertEqual(
                hashlib.sha256(gzip.decompress(gzip_path.read_bytes())).hexdigest(),
                pointer["original_authority"]["sha256"],
            )
            self.assertTrue((archive / "hardening-manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

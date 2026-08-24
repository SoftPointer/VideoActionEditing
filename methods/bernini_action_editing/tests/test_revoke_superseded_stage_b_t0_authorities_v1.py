from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revoke_superseded_stage_b_t0_authorities_v1.py"
)
SPEC = importlib.util.spec_from_file_location("revoke_authorities", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def authority(state: str, revision: str) -> bytes:
    return json.dumps(
        {"activation": {"state": state}, "revision": revision},
        sort_keys=True,
        indent=2,
    ).encode("utf-8")


class RevokeSupersededAuthoritiesTest(unittest.TestCase):
    def test_dry_run_then_apply_archives_and_revokes_only_old_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            old_paths = []
            for revision in ("retry4", "retry5"):
                source = root / f"source_{revision}"
                source.mkdir()
                path = source / f"stage_b_t0_single_update_{revision}_authority_addendum.json"
                path.write_bytes(authority(MODULE.ACTIVE_STATE, revision))
                old_paths.append(path)
            current_bytes = authority(MODULE.ACTIVE_STATE, "retry6")
            current_sha = hashlib.sha256(current_bytes).hexdigest()
            current_paths = []
            for suffix in ("root", "md"):
                source = root / f"source_retry6_{suffix}"
                source.mkdir()
                path = source / "stage_b_t0_single_update_retry6_authority_addendum.json"
                path.write_bytes(current_bytes)
                current_paths.append(path)
            draft = root / "stage_b_t0_single_update_retry5_authority_addendum.template.json"
            draft_bytes = authority("DRAFT", "template")
            draft.write_bytes(draft_bytes)
            archive = root / "authority_revocation_retry6"

            plan = MODULE.revoke_authorities(
                experiment_root=root,
                archive_dir=archive,
                keep_active_sha256=current_sha,
                reason="unit test",
                apply=False,
            )
            self.assertEqual(plan["superseded_active_count"], 2)
            self.assertFalse(archive.exists())
            self.assertTrue(all(path.read_bytes() == current_bytes for path in current_paths))

            receipt = MODULE.revoke_authorities(
                experiment_root=root,
                archive_dir=archive,
                keep_active_sha256=current_sha,
                reason="unit test",
                apply=True,
            )
            self.assertEqual(len(receipt["applied"]), 2)
            self.assertTrue((archive / "manifest.json").is_file())
            for path in old_paths:
                value = json.loads(path.read_text(encoding="ascii"))
                self.assertEqual(value["activation"]["state"], MODULE.REVOKED_STATE)
                self.assertFalse(value["activation"]["optimizer_creation_authorized"])
                archived = Path(value["original"]["archive_path"])
                self.assertEqual(
                    hashlib.sha256(archived.read_bytes()).hexdigest(),
                    value["original"]["sha256"],
                )
            self.assertTrue(all(path.read_bytes() == current_bytes for path in current_paths))
            self.assertEqual(draft.read_bytes(), draft_bytes)

    def test_fails_without_matching_current_active_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stage_b_t0_single_update_retry5_authority_addendum.json"
            path.write_bytes(authority(MODULE.ACTIVE_STATE, "retry5"))
            with self.assertRaisesRegex(MODULE.RevocationError, "no ACTIVE authority matches"):
                MODULE.revoke_authorities(
                    experiment_root=root,
                    archive_dir=root / "archive",
                    keep_active_sha256="0" * 64,
                    reason="unit test",
                    apply=False,
                )

    def test_active_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "stage_b_t0_single_update_retry6_authority_addendum.json"
            current_bytes = authority(MODULE.ACTIVE_STATE, "retry6")
            current.write_bytes(current_bytes)
            target = root / "real_stage_b_t0_single_update_retry5_authority_addendum.json"
            target.write_bytes(authority(MODULE.ACTIVE_STATE, "retry5"))
            link = root / "stage_b_t0_single_update_retry4_authority_addendum.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(MODULE.RevocationError, "non-symlink"):
                MODULE.revoke_authorities(
                    experiment_root=root,
                    archive_dir=root / "archive",
                    keep_active_sha256=hashlib.sha256(current_bytes).hexdigest(),
                    reason="unit test",
                    apply=False,
                )


if __name__ == "__main__":
    unittest.main()

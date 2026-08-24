#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import unittest

from methods.bernini_action_editing.tools import (
    build_clean_source_visual_context_checkpoint_review_release_v1 as release,
)
from methods.bernini_action_editing.tools import (
    build_clean_source_visual_context_stage_b_release_v1 as stage_b_release,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]


def _copy_release_tree(destination: Path) -> Path:
    root = destination / "methods" / "bernini_action_editing"
    for relative in release.RELEASE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(METHOD_ROOT / relative, target)
    return root.resolve()


class CheckpointReviewReleaseTests(unittest.TestCase):
    def test_member_categories_are_exact_and_stage_b_base_is_unchanged(self) -> None:
        self.assertEqual(
            release.STAGE_B_RELEASE_FILES,
            stage_b_release.RELEASE_FILES,
        )
        self.assertEqual(len(release.STAGE_B_RELEASE_FILES), 15)
        self.assertEqual(len(release.CHECKPOINT_REVIEW_FILES), 14)
        self.assertEqual(len(release.RECURSIVE_IMPORT_FILES), 13)
        self.assertEqual(len(release.RELEASE_FILES), 42)
        self.assertEqual(len(set(release.RELEASE_FILES)), 42)
        self.assertIn(
            "infer_clean_source_visual_context_checkpoint_review_v1.py",
            release.CHECKPOINT_REVIEW_FILES,
        )
        for required in (
            "infer_native_identity_generation_canary.py",
            "infer_native_v_axis_exact81_probe_v1.py",
            "infer_orderless_source_frame_set_noise_canary.py",
            "tri_branch_unipc.py",
            "infer_lora.py",
            "infer_source_kv_carrier_oracle.py",
            "infer_source_value_residual_oracle.py",
            "native_v_axis_guidance_v1.py",
            "orderless_source_frame_set_noise.py",
            "source_kv_replay.py",
            "source_kv_route_batches.py",
            "source_value_residual.py",
        ):
            self.assertIn(required, release.RECURSIVE_IMPORT_FILES)

    def test_recursive_import_graph_is_complete_and_minimal(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        discovered, edges, vendor_importers = release.discover_import_closure(
            METHOD_ROOT, payloads
        )
        self.assertEqual(discovered, release.RELEASE_FILES)
        self.assertEqual(
            vendor_importers, tuple(sorted(release.VENDOR_IMPORT_EXCEPTIONS))
        )
        edge_pairs = {
            (row["importer"], row["imported"])
            for row in edges
        }
        for edge in (
            (
                "infer_clean_source_visual_context_checkpoint_review_v1.py",
                "infer_native_identity_generation_canary.py",
            ),
            (
                "infer_clean_source_visual_context_checkpoint_review_v1.py",
                "infer_native_v_axis_exact81_probe_v1.py",
            ),
            (
                "infer_clean_source_visual_context_checkpoint_review_v1.py",
                "infer_orderless_source_frame_set_noise_canary.py",
            ),
            (
                "infer_native_identity_generation_canary.py",
                "infer_source_value_residual_oracle.py",
            ),
            ("infer_source_value_residual_oracle.py", "source_value_residual.py"),
            ("native_v_axis_guidance_v1.py", "native_i_axis_guidance.py"),
            ("native_i_axis_guidance.py", "tri_branch_unipc.py"),
        ):
            self.assertIn(edge, edge_pairs)
        external = manifest["external_runtime_imports"]
        self.assertEqual(len(external), 1)
        self.assertEqual(external[0]["module"], "tools.materialize_vae")
        self.assertFalse(external[0]["packaged_as_local_member"])
        self.assertEqual(
            external[0]["provider"],
            "pinned-bernini-root-prepended-by-activate_source_trees",
        )

    def test_archive_is_byte_deterministic_exact_ustar(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        first = release.build_archive(manifest, payloads)
        second = release.build_archive(manifest, payloads)
        self.assertEqual(first, second)
        release.verify_archive(first, manifest)
        self.assertEqual(
            manifest["manifest_digest"],
            release.object_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_digest"
                }
            ),
        )
        with tarfile.open(fileobj=io.BytesIO(first), mode="r:") as archive:
            members = archive.getmembers()
            self.assertEqual(
                [member.name for member in members],
                [
                    f"{release.MEMBER_ROOT}/{relative}"
                    for relative in release.RELEASE_FILES
                ],
            )
            for member in members:
                self.assertTrue(member.isfile())
                self.assertFalse(member.issym())
                self.assertFalse(member.islnk())
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.uname, "")
                self.assertEqual(member.gname, "")
                self.assertEqual(member.mtime, 0)
                self.assertEqual(stat.S_IMODE(member.mode), 0o444)
                self.assertFalse(member.pax_headers)

    def test_create_only_bundle_and_executed_root_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive_path = root / "checkpoint-review-runtime.tar"
            manifest_path = root / "checkpoint-review-runtime.manifest.json"
            built = release.build(METHOD_ROOT, archive_path, manifest_path)
            self.assertEqual(
                built["archive_sha256"],
                hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                built["manifest_sha256"],
                hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(stat.S_IMODE(archive_path.stat().st_mode), 0o444)
            self.assertEqual(stat.S_IMODE(manifest_path.stat().st_mode), 0o444)
            extracted = root / "executed"
            extracted.mkdir()
            with tarfile.open(archive_path, mode="r:") as archive:
                archive.extractall(extracted)
            method_root = extracted / release.MEMBER_ROOT
            receipt = release.verify_executed_root(
                method_root=method_root,
                archive_path=archive_path,
                archive_sha256=built["archive_sha256"],
                manifest_path=manifest_path,
                manifest_sha256=built["manifest_sha256"],
                method_revision=built["content_closure_sha1"],
            )
            self.assertEqual(receipt["exact_member_count"], 42)
            self.assertTrue(receipt["archive_members_verified"])
            self.assertTrue(receipt["executed_tree_exact_member_closure"])
            self.assertTrue(receipt["recursive_import_closure_verified"])
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError, "fresh absolute"
            ):
                release.build(METHOD_ROOT, archive_path, root / "second.json")

    def test_new_local_import_is_rejected_until_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = _copy_release_tree(Path(temporary).resolve())
            dependency = copied / "undeclared_review_dependency.py"
            dependency.write_text("VALUE = 1\n", encoding="utf-8")
            importer = copied / "infer_clean_source_visual_context_checkpoint_review_v1.py"
            importer.write_text(
                importer.read_text(encoding="utf-8")
                + "\nimport undeclared_review_dependency\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError,
                "recursive local import closure differs.*undeclared_review_dependency",
            ):
                release.build_manifest(copied)

    def test_selected_symlink_and_special_file_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = _copy_release_tree(Path(temporary).resolve())
            selected = copied / "tri_branch_unipc.py"
            payload = copied / "ordinary-payload.py"
            payload.write_text("VALUE = 1\n", encoding="utf-8")
            selected.unlink()
            selected.symlink_to(payload)
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError, "symlink"
            ):
                release.build_manifest(copied)
        with tempfile.TemporaryDirectory() as temporary:
            copied = _copy_release_tree(Path(temporary).resolve())
            selected = copied / "tri_branch_unipc.py"
            selected.unlink()
            os.mkfifo(selected)
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError, "canonical plain file"
            ):
                release.build_manifest(copied)

    def test_executed_root_rejects_extra_file_and_mode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            archive_path = root / "runtime.tar"
            manifest_path = root / "runtime.json"
            built = release.build(METHOD_ROOT, archive_path, manifest_path)
            extracted = root / "executed"
            extracted.mkdir()
            with tarfile.open(archive_path, mode="r:") as archive:
                archive.extractall(extracted)
            method_root = extracted / release.MEMBER_ROOT
            extra = method_root / "unexpected.py"
            extra.write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError, "exact file set"
            ):
                release.verify_executed_root(
                    method_root=method_root,
                    archive_path=archive_path,
                    archive_sha256=built["archive_sha256"],
                    manifest_path=manifest_path,
                    manifest_sha256=built["manifest_sha256"],
                    method_revision=built["content_closure_sha1"],
                )
            extra.unlink()
            drift = method_root / "tri_branch_unipc.py"
            drift.chmod(0o644)
            with self.assertRaisesRegex(
                release.CheckpointReviewReleaseError, "executed method member"
            ):
                release.verify_executed_root(
                    method_root=method_root,
                    archive_path=archive_path,
                    archive_sha256=built["archive_sha256"],
                    manifest_path=manifest_path,
                    manifest_sha256=built["manifest_sha256"],
                    method_revision=built["content_closure_sha1"],
                )

    def test_manifest_is_canonical_and_hash_bound(self) -> None:
        manifest, _ = release.build_manifest(METHOD_ROOT)
        encoded = release.canonical_json_bytes(manifest) + b"\n"
        decoded = json.loads(encoded.decode("ascii"))
        self.assertEqual(decoded, manifest)
        self.assertEqual(manifest["file_count"], 42)
        self.assertTrue(manifest["exact_member_closure"])
        self.assertTrue(manifest["recursive_import_closure"]["complete"])
        self.assertFalse(manifest["git_commit_claimed"])

    def test_resigned_import_boundary_or_graph_tamper_is_rejected(self) -> None:
        manifest, payloads = release.build_manifest(METHOD_ROOT)
        archive = release.build_archive(manifest, payloads)
        tampered = json.loads(json.dumps(manifest))
        tampered["external_runtime_imports"][0]["provider"] = "ambient-python-path"
        unsigned = dict(tampered)
        unsigned.pop("manifest_digest")
        tampered["manifest_digest"] = release.object_sha256(unsigned)
        with self.assertRaisesRegex(
            release.CheckpointReviewReleaseError, "external runtime import"
        ):
            release.verify_archive(archive, tampered)
        tampered = json.loads(json.dumps(manifest))
        tampered["recursive_import_closure"]["edges"] = []
        unsigned = dict(tampered)
        unsigned.pop("manifest_digest")
        tampered["manifest_digest"] = release.object_sha256(unsigned)
        with self.assertRaisesRegex(
            release.CheckpointReviewReleaseError, "does not reach exact closure"
        ):
            release.verify_archive(archive, tampered)


if __name__ == "__main__":
    unittest.main()

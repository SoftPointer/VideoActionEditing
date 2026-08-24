#!/usr/bin/env python3
"""Tests for the read-only authority-tree observation materializer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = (
    ROOT
    / "methods"
    / "bernini_action_editing"
    / "tools"
    / "materialize_case01_bone_removed_v2_authority_tree_v1.py"
)
SPEC = importlib.util.spec_from_file_location("authority_tree_v1", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)

GENERATOR_PATH = (
    ROOT
    / "methods"
    / "bernini_action_editing"
    / "generate_case01_bone_removed_v2_vace_v1.py"
)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "bone_removed_v2_generator", GENERATOR_PATH
)
assert GENERATOR_SPEC is not None and GENERATOR_SPEC.loader is not None
generator = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(generator)

ACCEPTANCE_PATH = (
    ROOT
    / "methods"
    / "bernini_action_editing"
    / "tools"
    / "case01_bone_removed_v2_acceptance_v1.py"
)
ACCEPTANCE_SPEC = importlib.util.spec_from_file_location(
    "bone_removed_v2_acceptance", ACCEPTANCE_PATH
)
assert ACCEPTANCE_SPEC is not None and ACCEPTANCE_SPEC.loader is not None
acceptance = importlib.util.module_from_spec(ACCEPTANCE_SPEC)
ACCEPTANCE_SPEC.loader.exec_module(acceptance)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class AuthorityTreeMaterializerTests(unittest.TestCase):
    def _tree(self, parent: Path) -> Path:
        root = parent / "tree"
        (root / "nested").mkdir(parents=True)
        (root / "alpha.bin").write_bytes(b"alpha")
        (root / "nested" / "beta.bin").write_bytes(b"beta\x00")
        return root

    def test_build_manifest_exact_schema_and_digests(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = self._tree(Path(text).resolve())
            manifest = tool.build_manifest("vace_source_tree", str(root))
            self.assertEqual(
                set(manifest),
                {
                    "schema_version",
                    "authority_role",
                    "inventory_policy",
                    "tree_root",
                    "entries",
                    "file_count",
                    "total_bytes",
                    "tree_digest",
                    "manifest_digest",
                },
            )
            self.assertEqual(
                [row["relative_path"] for row in manifest["entries"]],
                ["alpha.bin", "nested/beta.bin"],
            )
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(manifest["total_bytes"], 10)
            self.assertEqual(
                manifest["tree_digest"],
                hashlib.sha256(_canonical(manifest["entries"])).hexdigest(),
            )
            payload = dict(manifest)
            digest = payload.pop("manifest_digest")
            self.assertEqual(digest, hashlib.sha256(_canonical(payload)).hexdigest())

    def test_materialize_is_create_only_canonical_and_mode0400(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            root = self._tree(parent)
            output = parent / "manifest.json"
            observed = tool.materialize(
                "python_runtime_tree", str(root), str(output)
            )
            payload = output.read_bytes()
            self.assertEqual(payload, _canonical(observed) + b"\n")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o400)
            self.assertEqual(output.stat().st_nlink, 1)
            with self.assertRaisesRegex(tool.AuthorityTreeError, "already exists"):
                tool.materialize("python_runtime_tree", str(root), str(output))

    def test_frozen_generator_replays_materialized_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            root = self._tree(parent)
            output = parent / "manifest.json"
            manifest = tool.materialize(
                "vace_checkpoint_tree", str(root), str(output)
            )
            replay = generator.replay_tree_manifest(
                output, "vace_checkpoint_tree"
            )
            self.assertEqual(replay["role"], "vace_checkpoint_tree")
            self.assertEqual(replay["tree_digest"], manifest["tree_digest"])
            self.assertEqual(set(replay["entries"]), {"alpha.bin", "nested/beta.bin"})

    def test_non_ascii_tree_replays_in_both_frozen_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            root = parent / "tree_树"
            root.mkdir()
            (root / "β.bin").write_bytes(b"beta")
            output = parent / "manifest_树.json"
            manifest = tool.materialize(
                "vace_source_tree", str(root), str(output)
            )
            self.assertEqual(output.read_bytes(), _canonical(manifest) + b"\n")
            generator_replay = generator.replay_tree_manifest(
                output, "vace_source_tree"
            )
            acceptance_replay = acceptance._replay_authority_tree_manifest(
                manifest,
                manifest_path=output,
                expected_role="vace_source_tree",
            )
            self.assertEqual(generator_replay["tree_digest"], manifest["tree_digest"])
            self.assertEqual(acceptance_replay["tree_digest"], manifest["tree_digest"])

    def test_walk_enumeration_error_is_fail_closed_in_scan_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = self._tree(Path(text).resolve())

            def denied_walk(*args, **kwargs):
                kwargs["onerror"](PermissionError("synthetic unreadable subtree"))
                return iter(())

            with mock.patch.object(tool.os, "walk", side_effect=denied_walk):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "enumeration failed"
                ):
                    tool.build_manifest("vace_source_tree", str(root))

            _entries, files, directories = tool._scan_tree(root)
            with mock.patch.object(tool.os, "walk", side_effect=denied_walk):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "enumeration failed"
                ):
                    tool._replay_inventory(root, files, directories)

    def test_unknown_role_empty_tree_and_output_inside_tree_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            empty = parent / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(tool.AuthorityTreeError, "role differs"):
                tool.build_manifest("forged", str(empty))
            with self.assertRaisesRegex(tool.AuthorityTreeError, "empty"):
                tool.build_manifest("vace_source_tree", str(empty))
            root = self._tree(parent)
            with self.assertRaisesRegex(tool.AuthorityTreeError, "outside"):
                tool.materialize(
                    "vace_source_tree", str(root), str(root / "manifest.json")
                )

    def test_symlink_hardlink_and_special_leaf_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            root = self._tree(parent)
            link = root / "link.bin"
            link.symlink_to(root / "alpha.bin")
            with self.assertRaisesRegex(tool.AuthorityTreeError, "nonsymlink"):
                tool.build_manifest("vace_source_tree", str(root))
            link.unlink()
            os.link(root / "alpha.bin", root / "hard.bin")
            with self.assertRaisesRegex(tool.AuthorityTreeError, "nlink1"):
                tool.build_manifest("vace_source_tree", str(root))

    def test_noncanonical_backslash_relative_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = self._tree(Path(text).resolve())
            (root / "bad\\name.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(tool.AuthorityTreeError, "relative path"):
                tool.build_manifest("vace_source_tree", str(root))

    def test_leaf_change_during_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = self._tree(Path(text).resolve())
            original_fstat = os.fstat
            calls = 0

            def changed(descriptor):
                nonlocal calls
                calls += 1
                row = original_fstat(descriptor)
                if calls == 2:
                    (root / "alpha.bin").write_bytes(b"changed")
                return row

            with mock.patch.object(tool.os, "fstat", side_effect=changed):
                with self.assertRaisesRegex(tool.AuthorityTreeError, "changed"):
                    tool.build_manifest("vace_source_tree", str(root))

    def test_inventory_addition_after_hash_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            root = self._tree(Path(text).resolve())
            original = tool._replay_inventory

            def add_then_replay(tree_root, files, directories):
                (tree_root / "late.bin").write_bytes(b"late")
                return original(tree_root, files, directories)

            with mock.patch.object(tool, "_replay_inventory", side_effect=add_then_replay):
                with self.assertRaisesRegex(tool.AuthorityTreeError, "inventory"):
                    tool.build_manifest("vace_source_tree", str(root))

    def test_write_failure_cleans_only_owned_output(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            with mock.patch.object(tool.os, "write", return_value=0):
                with self.assertRaisesRegex(tool.AuthorityTreeError, "no progress"):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertFalse(output.exists())

    def test_data_close_failure_removes_owned_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            original_close = tool.os.close
            calls = 0

            def fail_first_close(descriptor):
                nonlocal calls
                calls += 1
                original_close(descriptor)
                if calls == 1:
                    raise OSError("synthetic data close failure")

            with mock.patch.object(tool.os, "close", side_effect=fail_first_close):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "publication failed"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertFalse(output.exists())

    def test_first_data_fstat_failure_reauthenticates_and_removes_output(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            original_fstat = tool.os.fstat
            calls = 0

            def fail_first_data_fstat(descriptor):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("synthetic first data fstat failure")
                return original_fstat(descriptor)

            with mock.patch.object(
                tool.os, "fstat", side_effect=fail_first_data_fstat
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "publication failed"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertFalse(output.exists())

    def test_persistent_data_fstat_failure_reports_cleanup_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            original_fstat = tool.os.fstat

            def reject_regular_fstat(descriptor):
                row = original_fstat(descriptor)
                if stat.S_ISREG(row.st_mode):
                    raise OSError("synthetic persistent data fstat failure")
                return row

            with mock.patch.object(
                tool.os, "fstat", side_effect=reject_regular_fstat
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "cleanup differs"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertTrue(output.exists())

    def test_parent_fsync_failure_removes_owned_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            original_fsync = tool.os.fsync
            failed = False

            def fail_first_directory_fsync(descriptor):
                nonlocal failed
                row = tool.os.fstat(descriptor)
                if stat.S_ISDIR(row.st_mode) and not failed:
                    failed = True
                    raise OSError("synthetic parent fsync failure")
                return original_fsync(descriptor)

            with mock.patch.object(
                tool.os, "fsync", side_effect=fail_first_directory_fsync
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "publication failed"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertTrue(failed)
            self.assertFalse(output.exists())

    def test_parent_open_failure_occurs_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            with mock.patch.object(
                tool.os, "open", side_effect=OSError("synthetic parent open failure")
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "publication failed"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertFalse(output.exists())

    def test_parent_close_failure_removes_owned_complete_output(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            original_close = tool.os.close
            failed = False

            def fail_first_directory_close(descriptor):
                nonlocal failed
                row = tool.os.fstat(descriptor)
                is_directory = stat.S_ISDIR(row.st_mode)
                original_close(descriptor)
                if is_directory and not failed:
                    failed = True
                    raise OSError("synthetic parent close failure")

            with mock.patch.object(
                tool.os, "close", side_effect=fail_first_directory_close
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "publication failed"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertTrue(failed)
            self.assertFalse(output.exists())

    def test_cleanup_failure_is_reported_and_never_silently_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            parent = Path(text).resolve()
            output = parent / "manifest.json"
            with mock.patch.object(
                tool.os, "write", return_value=0
            ), mock.patch.object(
                tool.os, "unlink", side_effect=OSError("synthetic unlink failure")
            ):
                with self.assertRaisesRegex(
                    tool.AuthorityTreeError, "cleanup differs"
                ):
                    tool._write_create_only(str(output), b"payload\n")
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

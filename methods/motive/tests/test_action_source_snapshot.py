from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "action_source_snapshot.py"
SPEC = importlib.util.spec_from_file_location("action_source_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
snapshot_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_module)


def _repo(root: Path) -> Path:
    repo = root / "repo"
    (repo / "lucy").mkdir(parents=True)
    (repo / "methods" / "motive").mkdir(parents=True)
    (repo / "lucy" / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (repo / "methods" / "motive" / "README.md").write_text(
        "motive\n",
        encoding="utf-8",
    )
    (repo / "lucy" / "__pycache__").mkdir()
    (repo / "lucy" / "__pycache__" / "ignored.pyc").write_bytes(b"x")
    return repo


class ActionSourceSnapshotTests(unittest.TestCase):
    def test_snapshot_build_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                provenance = snapshot_module.build_snapshot(repo, output)
            verified = snapshot_module.verify_snapshot(
                output,
                expected_tree_sha256=provenance["source_tree_sha256"],
            )
            self.assertEqual(verified["source_file_count"], 2)
            self.assertFalse((output / "lucy" / "__pycache__").exists())
            self.assertEqual(output.stat().st_mode & 0o222, 0)
            self.assertEqual(
                (output / "SOURCE_FILES.jsonl").stat().st_mode & 0o222,
                0,
            )
            self.assertEqual(
                (output / "SOURCE_PROVENANCE.json").stat().st_mode & 0o222,
                0,
            )

    def test_snapshot_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                snapshot_module.build_snapshot(repo, output)
            path = output / "lucy" / "train.py"
            path.chmod(0o644)
            path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "(size|SHA-256) mismatch",
            ):
                snapshot_module.verify_snapshot(output)

    def test_snapshot_rejects_extra_file_outside_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                snapshot_module.build_snapshot(repo, output)
            motive = output / "methods" / "motive"
            motive.chmod(0o755)
            (motive / "injected.py").write_text(
                "print('unbound')\n",
                encoding="utf-8",
            )
            (motive / "injected.py").chmod(0o444)
            motive.chmod(0o555)
            with self.assertRaisesRegex(
                ValueError,
                "snapshot file closure mismatch",
            ):
                snapshot_module.verify_snapshot(output)

    def test_snapshot_rejects_extra_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                snapshot_module.build_snapshot(repo, output)
            output.chmod(0o755)
            (output / "injected-link").symlink_to(
                output / "lucy" / "train.py"
            )
            output.chmod(0o555)
            with self.assertRaisesRegex(
                ValueError,
                "snapshot contains symlink",
            ):
                snapshot_module.verify_snapshot(output)

    def test_snapshot_rejects_extra_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                snapshot_module.build_snapshot(repo, output)
            output.chmod(0o755)
            (output / "unbound-directory").mkdir(mode=0o555)
            output.chmod(0o555)
            with self.assertRaisesRegex(
                ValueError,
                "snapshot directory closure mismatch",
            ):
                snapshot_module.verify_snapshot(output)

    def test_snapshot_rejects_writable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                snapshot_module.build_snapshot(repo, output)
            (output / "lucy").chmod(0o755)
            with self.assertRaisesRegex(
                ValueError,
                "snapshot directory remains writable",
            ):
                snapshot_module.verify_snapshot(output)

    def test_snapshot_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = _repo(root)
            output = root / "snapshot"
            output.mkdir()
            with mock.patch.object(
                snapshot_module,
                "_git",
                return_value="test",
            ):
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to reuse",
                ):
                    snapshot_module.build_snapshot(repo, output)


if __name__ == "__main__":
    unittest.main()

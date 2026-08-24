from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = METHOD_ROOT / "tools" / "build_elal3_c0_release_v1.py"
SPEC = importlib.util.spec_from_file_location("build_elal3_c0_release_v1", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ELAL3C0ReleaseTests(unittest.TestCase):
    def _source_root(self, parent: Path) -> Path:
        root = parent / "method"
        (root / "tests").mkdir(parents=True)
        for index, relative in enumerate(builder.RELEASE_FILES):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"VALUE_{index} = {index}\n".encode("ascii"))
        return root.resolve()

    def test_build_is_deterministic_sorted_and_non_authorizing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = self._source_root(parent)
            first = parent / "release-a"
            second = parent / "release-b"
            a = builder.publish(root, first)
            b = builder.publish(root, second)
            self.assertEqual((first / "source.tar").read_bytes(), (second / "source.tar").read_bytes())
            self.assertEqual(
                (first / "source.manifest.json").read_bytes(),
                (second / "source.manifest.json").read_bytes(),
            )
            self.assertEqual(a["archive_sha256"], b["archive_sha256"])
            manifest_raw = (first / "source.manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
            self.assertEqual(manifest_raw, builder.canonical_json_bytes(manifest) + b"\n")
            self.assertFalse(manifest["training_authorized"])
            self.assertFalse(manifest["exact160_authorized"])
            self.assertFalse(manifest["representation_semantics_qualified"])
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o555)
            self.assertEqual(stat.S_IMODE((first / "source.tar").stat().st_mode), 0o444)
            with tarfile.open(first / "source.tar", "r:") as archive:
                members = archive.getmembers()
                names = [member.name for member in members]
                self.assertEqual(names, sorted(names, key=lambda value: value.encode("ascii")))
                self.assertEqual(len(names), 3)
                for member in members:
                    self.assertTrue(member.isreg())
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual(member.mode, 0o444)

    def test_publish_refuses_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            root = self._source_root(parent)
            output = parent / "release"
            builder.publish(root, output)
            with self.assertRaises(builder.ELAL3ReleaseError):
                builder.publish(root, output)


if __name__ == "__main__":
    unittest.main()

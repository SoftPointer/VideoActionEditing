from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from methods.bernini_action_editing import validate_v14r2_deployment_marker as validator


MIN_TEST_COUNT = 121


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V14R2DeploymentMarkerTests(unittest.TestCase):
    def _deployment(
        self,
        root: Path,
        role: str,
        payloads: dict[str, bytes],
        *,
        training: tuple[Path, dict[str, str]] | None = None,
    ) -> tuple[dict[str, object], dict[str, str]]:
        source = root / f"{role}-source"
        source.mkdir()
        hashes: dict[str, str] = {}
        for relative, payload in payloads.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            hashes[relative] = _sha(path)
        archive = root / f"{role}.tar"
        archive.write_bytes(f"{role}-archive".encode("ascii"))
        revision = root / f"{role}.revision"
        revision.write_text(f"{role}-revision\n", encoding="ascii")
        content_path = root / f"{role}.content.json"
        content = {
            "schema_version": validator.CONTENT_SCHEMA,
            "complete": True,
            "source_tree": str(source),
            "files": hashes,
        }
        content_path.write_text(json.dumps(content), encoding="ascii")
        marker_path = root / f"{role}.marker.json"
        marker: dict[str, object] = {
            "schema_version": validator.SCHEMA,
            "complete": True,
            "role": role,
            "source_tree": str(source),
            "archive": {"path": str(archive), "sha256": _sha(archive)},
            "revision": {
                "path": str(revision),
                "sha256": _sha(revision),
                "value": f"{role}-revision",
            },
            "content_manifest": {
                "path": str(content_path),
                "sha256": _sha(content_path),
            },
            "required_files": hashes,
            "tests": {"passed": True, "total_passed": MIN_TEST_COUNT},
        }
        if training is not None:
            training_marker, training_hashes = training
            marker["training_compatibility"] = {
                "training_marker_path": str(training_marker),
                "training_marker_sha256": _sha(training_marker),
                "shared_core": training_hashes,
            }
        marker_path.write_text(json.dumps(marker), encoding="ascii")
        paths = {
            "marker": str(marker_path),
            "source": str(source),
            "archive": str(archive),
            "revision": str(revision),
            "content": str(content_path),
        }
        return paths, hashes

    def _validate(
        self,
        paths: dict[str, str],
        role: str,
        required: list[str],
        *,
        training_marker: str | None = None,
        shared: list[str] | None = None,
    ) -> None:
        validator.validate(
            marker_path=Path(paths["marker"]),
            role=role,
            source_tree=Path(paths["source"]),
            archive=Path(paths["archive"]),
            revision=Path(paths["revision"]),
            content_manifest=Path(paths["content"]),
            min_test_count=MIN_TEST_COUNT,
            required_files=required,
            training_marker_path=(Path(training_marker) if training_marker else None),
            shared_core=shared or [],
        )

    def test_training_closure_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self._deployment(
                Path(directory), "training", {"method.py": b"strict"}
            )
            self._validate(paths, "training", ["method.py"])

    def test_archive_and_required_file_mutations_are_rejected(self) -> None:
        for target in ("archive", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                paths, _ = self._deployment(
                    Path(directory), "training", {"method.py": b"strict"}
                )
                if target == "archive":
                    Path(paths["archive"]).write_bytes(b"changed")
                else:
                    (Path(paths["source"]) / "method.py").write_bytes(b"changed")
                with self.assertRaises(validator.V14R2DeploymentValidationError):
                    self._validate(paths, "training", ["method.py"])

    def test_marker_must_cover_every_content_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self._deployment(
                Path(directory),
                "training",
                {"method.py": b"strict", "dependency.py": b"runtime"},
            )
            marker_path = Path(paths["marker"])
            marker = json.loads(marker_path.read_text(encoding="ascii"))
            del marker["required_files"]["dependency.py"]
            marker_path.write_text(json.dumps(marker), encoding="ascii")
            with self.assertRaises(validator.V14R2DeploymentValidationError):
                self._validate(paths, "training", ["method.py"])

    def test_unlisted_source_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self._deployment(
                Path(directory), "training", {"method.py": b"strict"}
            )
            (Path(paths["source"]) / "shadow.py").write_bytes(b"shadow")
            with self.assertRaises(validator.V14R2DeploymentValidationError):
                self._validate(paths, "training", ["method.py"])

    def test_decode_must_pin_hash_equivalent_training_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_paths, train_hashes = self._deployment(
                root, "training", {"shared.py": b"same", "train.py": b"train"}
            )
            decode_paths, _ = self._deployment(
                root,
                "decode",
                {"shared.py": b"same", "decode.py": b"decode"},
                training=(Path(train_paths["marker"]), {"shared.py": train_hashes["shared.py"]}),
            )
            self._validate(
                decode_paths,
                "decode",
                ["decode.py", "shared.py"],
                training_marker=train_paths["marker"],
                shared=["shared.py"],
            )

    def test_decode_shared_core_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_paths, train_hashes = self._deployment(
                root, "training", {"shared.py": b"same"}
            )
            decode_paths, _ = self._deployment(
                root,
                "decode",
                {"shared.py": b"different"},
                training=(Path(train_paths["marker"]), {"shared.py": train_hashes["shared.py"]}),
            )
            with self.assertRaises(validator.V14R2DeploymentValidationError):
                self._validate(
                    decode_paths,
                    "decode",
                    ["shared.py"],
                    training_marker=train_paths["marker"],
                    shared=["shared.py"],
                )

    def test_decode_rehashes_active_training_shared_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_paths, train_hashes = self._deployment(
                root, "training", {"shared.py": b"same", "train.py": b"train"}
            )
            decode_paths, _ = self._deployment(
                root,
                "decode",
                {"shared.py": b"same", "decode.py": b"decode"},
                training=(Path(train_paths["marker"]), {"shared.py": train_hashes["shared.py"]}),
            )
            (Path(train_paths["source"]) / "shared.py").write_bytes(b"drifted")
            with self.assertRaises(validator.V14R2DeploymentValidationError):
                self._validate(
                    decode_paths,
                    "decode",
                    ["decode.py", "shared.py"],
                    training_marker=train_paths["marker"],
                    shared=["shared.py"],
                )

    def test_decode_rehashes_active_decode_shared_core(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_paths, train_hashes = self._deployment(
                root, "training", {"shared.py": b"same", "train.py": b"train"}
            )
            decode_paths, _ = self._deployment(
                root,
                "decode",
                {"shared.py": b"same", "decode.py": b"decode"},
                training=(Path(train_paths["marker"]), {"shared.py": train_hashes["shared.py"]}),
            )
            (Path(decode_paths["source"]) / "shared.py").write_bytes(b"drifted")
            with self.assertRaises(validator.V14R2DeploymentValidationError):
                self._validate(
                    decode_paths,
                    "decode",
                    ["decode.py", "shared.py"],
                    training_marker=train_paths["marker"],
                    shared=["shared.py"],
                )


if __name__ == "__main__":
    unittest.main()

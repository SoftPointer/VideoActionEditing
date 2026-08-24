import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


from methods.bernini_action_editing import action_preservation_completion_publisher_v1 as publisher


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_file(path: Path, raw: bytes, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return digest(raw)


class CompletionPublisherTests(unittest.TestCase):
    def fixture(self, directory: str):
        base = Path(directory).resolve()
        release = base / "release"
        root = base / "experiment"
        source = base / "source-data.manifest.json"
        release.mkdir(mode=0o700)
        root.mkdir(mode=0o700)
        (root / "logs").mkdir(mode=0o700)
        (root / "materialized" / "methods" / "bernini_action_editing").mkdir(
            mode=0o700, parents=True
        )
        (root / "runs").mkdir(mode=0o700)

        materialized_raw = b"VALUE = 1\n"
        member_path = "member.py"
        materialized = (
            root
            / "materialized"
            / "methods"
            / "bernini_action_editing"
            / member_path
        )
        write_file(materialized, materialized_raw)
        manifest = {
            "member_root": "methods/bernini_action_editing",
            "files": [
                {
                    "path": member_path,
                    "mode": 0o444,
                    "size": len(materialized_raw),
                    "sha256": digest(materialized_raw),
                }
            ],
        }
        manifest["file_count"] = len(manifest["files"])
        revision = hashlib.sha1(
            publisher.canonical(manifest["files"])
        ).hexdigest()
        manifest["content_revision"] = revision
        manifest["manifest_digest"] = digest(publisher.canonical(manifest))
        manifest_raw = publisher.canonical(manifest) + b"\n"
        manifest_sha = write_file(release / "source.manifest.json", manifest_raw)

        cache_sha = write_file(root / publisher.CACHE_BASENAME, b"cache\n")
        receipt_sha = write_file(
            root / (publisher.CACHE_BASENAME + ".receipt.json"), b"receipt\n"
        )
        cache_audit = {
            "cache_audit_go": True,
            "cache_sha256": cache_sha,
            "cache_receipt_sha256": receipt_sha,
        }
        cache_audit_raw = json.dumps(cache_audit, indent=2).encode() + b"\n"
        cache_audit_sha = write_file(root / "logs" / "cache-audit.json", cache_audit_raw)
        materialization = {
            "release_root": str(root / "materialized"),
            "method_root": str(
                root / "materialized" / "methods" / "bernini_action_editing"
            ),
            "archive_sha256": "0" * 64,
            "manifest_sha256": manifest_sha,
            "content_revision": revision,
            "file_count": 1,
            "exact_tree_verified": True,
            "directories_sealed_mode": "0555",
        }
        materialization["receipt_digest"] = digest(
            publisher.canonical(materialization)
        )
        write_file(
            root / "logs" / "materialization.json",
            publisher.canonical(materialization) + b"\n",
        )
        write_file(root / "logs" / "cache-full.log", b"cache log\n")

        receipt_rows = []
        for arm in publisher.ARMS:
            write_file(root / "logs" / f"train-{arm}.log", f"{arm}\n".encode())
            for step in publisher.CHECKPOINT_STEPS:
                checkpoint = root / "runs" / arm / f"checkpoint-{step:08d}"
                adapter = checkpoint / "adapter"
                adapter.mkdir(mode=0o700, parents=True)
                receipt = f"receipt:{arm}:{step}\n".encode()
                optimizer = f"optimizer:{arm}:{step}\n".encode()
                model = f"model:{arm}:{step}\n".encode()
                config = f"config:{arm}:{step}\n".encode()
                receipt_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "receipt_sha256": write_file(checkpoint / "receipt.json", receipt),
                        "adapter_sha256": write_file(adapter / "adapter_model.safetensors", model),
                        "adapter_config_sha256": write_file(adapter / "adapter_config.json", config),
                        "optimizer_sha256": write_file(checkpoint / "optimizer.pt", optimizer),
                        "loss": 0.0 if step == 0 else 1.0,
                        "preclip_gradient_norm": 0.0 if step == 0 else 1.0,
                    }
                )
        training_audit = {
            "training_audit_go": True,
            "arm_count": 8,
            "checkpoint_count": 32,
            "checkpoint_steps": [0, 5, 10, 20],
            "receipt_rows": receipt_rows,
            "decoded_evaluation_complete": False,
            "scientific_promotion_authorized": False,
        }
        training_raw = json.dumps(training_audit, indent=2).encode() + b"\n"
        training_sha = write_file(root / "logs" / "training-audit.json", training_raw)

        for current, directories, _files in os.walk(root, topdown=False):
            for name in directories:
                (Path(current) / name).chmod(0o555)
        (root / "logs").chmod(0o555)
        (root / "materialized").chmod(0o555)
        (root / "runs").chmod(0o555)
        root.chmod(0o700)

        archive_sha = write_file(release / "source.tar", b"archive\n")
        controller_sha = write_file(release / "controller.sh", b"#!/bin/bash\n", 0o555)
        envelope_sha = write_file(release / "deployment-envelope.json", b"{}\n")
        source_value = {"schema_version": "test-source-manifest-v1"}
        source_digest = digest(publisher.canonical(source_value))
        source_value["manifest_digest"] = source_digest
        source_raw = publisher.canonical(source_value) + b"\n"
        source_sha = write_file(source, source_raw, 0o444)
        materialization_path = root / "logs" / "materialization.json"
        materialization = json.loads(materialization_path.read_text())
        materialization["archive_sha256"] = archive_sha
        unsigned_materialization = dict(materialization)
        unsigned_materialization.pop("receipt_digest")
        materialization["receipt_digest"] = digest(
            publisher.canonical(unsigned_materialization)
        )
        materialization_path.chmod(0o644)
        write_file(
            materialization_path,
            publisher.canonical(materialization) + b"\n",
        )
        args = argparse.Namespace(
            experiment_root=str(root),
            cache_sha256=cache_sha,
            cache_receipt_sha256=receipt_sha,
            cache_audit_sha256=cache_audit_sha,
            training_audit_sha256=training_sha,
            source_archive=str(release / "source.tar"),
            source_archive_sha256=archive_sha,
            release_manifest=str(release / "source.manifest.json"),
            release_manifest_sha256=manifest_sha,
            controller=str(release / "controller.sh"),
            controller_sha256=controller_sha,
            deployment_envelope=str(release / "deployment-envelope.json"),
            deployment_envelope_sha256=envelope_sha,
            source_data_manifest=str(source),
            source_data_manifest_sha256=source_sha,
            source_data_manifest_digest=source_digest,
            source_revision=revision,
        )
        return args, root

    def test_success_is_last_commit_and_seals_exact_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            completion_sha = publisher.publish(args)
            marker = root / publisher.MARKER_BASENAME
            self.assertEqual(digest(marker.read_bytes()), completion_sha)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o555)
            value = json.loads(marker.read_text())
            self.assertEqual(value["schema_version"], "bernini-action-preservation-v2-training-complete-v3")
            self.assertTrue(value["retained_tree_held_fd_identity_replay"])
            self.assertTrue(value["retained_tree_stable_double_read_before_commit"])
            self.assertEqual(value["retained_tree_file_count"], 143)
            self.assertFalse(value["scientific_promotion_authorized"])

    def test_root_seal_failure_rolls_back_exact_marker_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            real_fchmod = publisher.os.fchmod

            def fail_root_seal(descriptor, mode):
                if mode == 0o555 and stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    raise OSError("injected root seal failure")
                return real_fchmod(descriptor, mode)

            with mock.patch.object(publisher.os, "fchmod", side_effect=fail_root_seal):
                with self.assertRaises(OSError):
                    publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_partial_marker_write_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            real_write = publisher.os.write
            state = {"calls": 0}

            def short_then_fail(descriptor, raw):
                state["calls"] += 1
                if state["calls"] == 1:
                    return real_write(descriptor, bytes(raw[:1]))
                raise OSError("injected short-write failure")

            with mock.patch.object(publisher.os, "write", side_effect=short_then_fail):
                with self.assertRaises(OSError):
                    publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_alarm_after_canonical_link_rolls_back_known_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            real_link = publisher.os.link

            def link_then_alarm(*link_args, **link_kwargs):
                result = real_link(*link_args, **link_kwargs)
                signal.raise_signal(signal.SIGALRM)
                return result

            import signal

            with mock.patch.object(publisher.os, "link", side_effect=link_then_alarm):
                with self.assertRaises(TimeoutError):
                    publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_same_name_cache_replacement_before_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            real_revalidate = publisher.revalidate_file
            state = {"mutated": False}

            def mutate(item):
                if not state["mutated"] and item.label == publisher.CACHE_BASENAME:
                    state["mutated"] = True
                    cache = root / publisher.CACHE_BASENAME
                    cache.chmod(0o644)
                    cache.rename(root / "displaced-cache")
                    write_file(cache, b"hostile\n")
                return real_revalidate(item)

            with mock.patch.object(publisher, "revalidate_file", side_effect=mutate):
                with self.assertRaises(publisher.CompletionPublicationError):
                    publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_false_source_manifest_digest_is_rejected_before_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            args.source_data_manifest_digest = "3" * 64
            with self.assertRaisesRegex(
                publisher.CompletionPublicationError,
                "source data manifest digest differs",
            ):
                publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_fully_resigned_false_release_member_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            manifest_path = Path(args.release_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][0]["size"] += 17
            manifest["content_revision"] = hashlib.sha1(
                publisher.canonical(manifest["files"])
            ).hexdigest()
            unsigned = dict(manifest)
            unsigned.pop("manifest_digest")
            manifest["manifest_digest"] = digest(publisher.canonical(unsigned))
            raw = publisher.canonical(manifest) + b"\n"
            manifest_path.chmod(0o644)
            args.release_manifest_sha256 = write_file(manifest_path, raw)
            args.source_revision = manifest["content_revision"]
            materialization_path = root / "logs" / "materialization.json"
            materialization = json.loads(materialization_path.read_text())
            materialization["manifest_sha256"] = args.release_manifest_sha256
            materialization["content_revision"] = args.source_revision
            materialization_unsigned = dict(materialization)
            materialization_unsigned.pop("receipt_digest")
            materialization["receipt_digest"] = digest(
                publisher.canonical(materialization_unsigned)
            )
            materialization_path.chmod(0o644)
            write_file(
                materialization_path,
                publisher.canonical(materialization) + b"\n",
            )
            with self.assertRaisesRegex(
                publisher.CompletionPublicationError,
                "tree file size differs",
            ):
                publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())

    def test_fully_resigned_invented_release_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            manifest_path = Path(args.release_manifest)
            manifest = json.loads(manifest_path.read_text())
            manifest["content_revision"] = "f" * 40
            unsigned = dict(manifest)
            unsigned.pop("manifest_digest")
            manifest["manifest_digest"] = digest(publisher.canonical(unsigned))
            raw = publisher.canonical(manifest) + b"\n"
            manifest_path.chmod(0o644)
            args.release_manifest_sha256 = write_file(manifest_path, raw)
            args.source_revision = manifest["content_revision"]
            with self.assertRaisesRegex(
                publisher.CompletionPublicationError,
                "release content revision differs",
            ):
                publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())

    def test_fully_resigned_false_materialization_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            receipt_path = root / "logs" / "materialization.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["archive_sha256"] = "e" * 64
            unsigned = dict(receipt)
            unsigned.pop("receipt_digest")
            receipt["receipt_digest"] = digest(publisher.canonical(unsigned))
            receipt_path.chmod(0o644)
            write_file(
                receipt_path,
                publisher.canonical(receipt) + b"\n",
            )
            with self.assertRaisesRegex(
                publisher.CompletionPublicationError,
                "materialization receipt authority differs",
            ):
                publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())

    def test_same_name_directory_replacement_before_commit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, root = self.fixture(directory)
            real_revalidate = publisher.revalidate_directory
            state = {"mutated": False}

            def mutate(item, **kwargs):
                if not state["mutated"] and item.label == "logs":
                    state["mutated"] = True
                    logs = root / "logs"
                    root.chmod(0o700)
                    logs.chmod(0o700)
                    logs.rename(root / "displaced-logs")
                    logs.mkdir(mode=0o555)
                return real_revalidate(item, **kwargs)

            with mock.patch.object(
                publisher, "revalidate_directory", side_effect=mutate
            ):
                with self.assertRaises(publisher.CompletionPublicationError):
                    publisher.publish(args)
            self.assertFalse((root / publisher.MARKER_BASENAME).exists())
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)


if __name__ == "__main__":
    unittest.main()

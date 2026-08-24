from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
import types
from unittest import mock


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_eval_v1 as v1
import full644_exploratory_matched_eval_v2 as v2


def source_fixture(root: Path) -> tuple[dict, dict]:
    source = (root / "source.mp4").resolve()
    source.write_bytes(b"source-video-bytes")
    source.chmod(0o444)
    info = source.lstat()
    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    authority = v2._source_stat_projection(source, info, sha256)
    authority_digest = v1.object_sha256(authority)
    task = {
        "source_video": str(source),
        "source_video_sha256": sha256,
    }
    receipt = {
        "input": {
            "source_video_physical_authority": authority,
            "source_video_physical_authority_digest": authority_digest,
            "retained_source_fd_consumed": True,
            "source_video_pre_and_post_decode_rehashed": True,
        },
        "model_consumption": {
            "source_video_physical_authority_digest": authority_digest,
            "all_ranks_use_retained_source_fd": True,
        },
    }
    return task, receipt


def reseal_authority(receipt: dict) -> None:
    authority = receipt["input"]["source_video_physical_authority"]
    digest = v1.object_sha256(authority)
    receipt["input"]["source_video_physical_authority_digest"] = digest
    receipt["model_consumption"][
        "source_video_physical_authority_digest"
    ] = digest


def ffprobe_fixture(root: Path) -> tuple[dict, dict, int, Path]:
    executable = root / "ffprobe"
    executable.write_bytes(b"fixture-ffprobe-executable\n")
    executable.chmod(0o555)
    descriptor = os.open(executable, os.O_RDONLY)
    os.set_inheritable(descriptor, False)
    row = {
        "schema_version": v2.FFPROBE_AUTHORITY_SCHEMA,
        "fd": descriptor,
        "source_path": str(executable),
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "identity": v2._exec_identity(os.fstat(descriptor)),
    }
    row["authority_digest"] = v1.object_sha256(row)
    producer = {
        "ffprobe_path": str(executable),
        "ffprobe_sha256": row["sha256"],
    }
    return row, producer, descriptor, executable


def publication_fixture(
    root: Path, *, task_id: str = "shared8-00-base"
) -> tuple[dict, dict, tuple[int, int]]:
    publication = root / "publication"
    publication.mkdir(exist_ok=True)
    output = publication / "case00-base.mp4"
    receipt = publication / "case00-base.mp4.receipt.json"
    output.write_bytes(b"fixture-mp4")
    output.chmod(0o444)
    receipt.write_bytes(b'{"fixture":true}\n')
    receipt.chmod(0o400)
    output_fd = os.open(output, os.O_RDONLY)
    receipt_fd = os.open(receipt, os.O_RDONLY)
    os.set_inheritable(output_fd, False)
    os.set_inheritable(receipt_fd, False)
    task = {
        "task_id": task_id,
        "output": {
            "video_path": str(output),
            "receipt_path": str(receipt),
        },
    }
    row = {
        "schema_version": v2.PUBLICATION_AUTHORITY_SCHEMA,
        "task_id": task_id,
        "output_path": str(output),
        "output_fd": output_fd,
        "output_identity": v2._exec_identity(os.fstat(output_fd)),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_size": output.stat().st_size,
        "receipt_path": str(receipt),
        "receipt_fd": receipt_fd,
        "receipt_identity": v2._exec_identity(os.fstat(receipt_fd)),
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "receipt_size": receipt.stat().st_size,
    }
    row["authority_digest"] = v1.object_sha256(row)
    return row, task, (output_fd, receipt_fd)


class MatchedEvalV2Tests(unittest.TestCase):
    def test_ffprobe_exec_uses_retained_inode_and_exact_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            authority, producer, ffprobe_fd, _ = ffprobe_fixture(root)
            publication_authority, task, publication_fds = publication_fixture(
                root
            )
            publication = root / "publication"
            publication_fd = os.open(publication, os.O_RDONLY)
            os.set_inheritable(publication_fd, False)
            logical_output = Path(task["output"]["video_path"])
            launched: list[tuple[list[str], dict]] = []

            def fake_run(arguments, **kwargs):
                launched.append((list(arguments), dict(kwargs)))
                return object()

            def fake_probe(path, producer_value):
                result = v1.subprocess.run(
                    [
                        producer_value["ffprobe_path"],
                        "-v",
                        "error",
                        "-count_frames",
                        "-show_entries",
                        "stream=codec_type,width,height,avg_frame_rate,nb_read_frames",
                        "-of",
                        "json",
                        str(path),
                    ],
                    check=False,
                    stdout=v1.subprocess.PIPE,
                    stderr=v1.subprocess.PIPE,
                    timeout=60,
                    env={"LC_ALL": "C", "LANG": "C"},
                )
                self.assertIsNotNone(result)
                return {"fixture": True}

            try:
                with mock.patch.object(
                    v1.subprocess, "run", side_effect=fake_run
                ), mock.patch.object(v1, "_probe_mp4", side_effect=fake_probe):
                    with v2._v1_output_fd_compatibility(
                        logical_output,
                        publication,
                        publication_fd,
                        producer,
                        authority,
                        publication_authority,
                        task,
                    ):
                        observed = v1._probe_mp4(logical_output, producer)
                self.assertEqual(observed, {"fixture": True})
                self.assertEqual(len(launched), 1)
                kwargs = launched[0][1]
                self.assertEqual(
                    kwargs["executable"], f"/proc/self/fd/{ffprobe_fd}"
                )
                self.assertEqual(
                    kwargs["pass_fds"],
                    tuple(
                        sorted((publication_authority["output_fd"], ffprobe_fd))
                    ),
                )
                self.assertEqual(
                    kwargs["env"], {"LC_ALL": "C", "LANG": "C"}
                )
            finally:
                os.close(publication_fd)
                os.close(ffprobe_fd)
                for descriptor in publication_fds:
                    os.close(descriptor)

    def test_retained_publication_rejects_leaf_swap_after_capture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            authority, task, descriptors = publication_fixture(root)
            output = Path(task["output"]["video_path"])
            try:
                self.assertEqual(
                    v2.validate_retained_publication_authority(authority, task),
                    authority,
                )
                held = output.with_suffix(".held")
                output.rename(held)
                output.write_bytes(b"hostile-valid-mp4")
                output.chmod(0o444)
                with self.assertRaises(v2.MatchedEvalV2Error):
                    v2.validate_retained_publication_authority(authority, task)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_ffprobe_retained_authority_rejects_named_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            authority, producer, descriptor, executable = ffprobe_fixture(root)
            try:
                self.assertEqual(
                    v2.validate_retained_ffprobe_authority(authority, producer),
                    authority,
                )
                held = root / "ffprobe-held"
                executable.rename(held)
                executable.write_bytes(b"hostile-ffprobe\n")
                executable.chmod(0o555)
                with self.assertRaises(v2.MatchedEvalV2Error):
                    v2.validate_retained_ffprobe_authority(authority, producer)
            finally:
                os.close(descriptor)

    def test_retained_publication_root_rejects_named_symlink_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw).resolve(strict=True)
            root = parent / "publication"
            moved = parent / "publication-held"
            root.mkdir()
            descriptor = os.open(root, os.O_RDONLY)
            try:
                self.assertFalse(os.get_inheritable(descriptor))
                v2._validate_publication_root_fd(root, descriptor)
                root.rename(moved)
                root.symlink_to(moved, target_is_directory=True)
                with self.assertRaises(v2.MatchedEvalV2Error):
                    v2._validate_publication_root_fd(root, descriptor)
            finally:
                os.close(descriptor)

    def test_authority_cli_delegates_exact_argv_to_v1(self) -> None:
        command = [
            "authority-check",
            "--input-manifest",
            "input.jsonl",
            "--exposure-audit",
            "exposure.json",
        ]
        with mock.patch.object(v2.v1, "main", return_value=17) as delegated:
            self.assertEqual(v2.main(command), 17)
            delegated.assert_called_once_with(command)

    def test_terminal_cp644_config_compatibility_is_scoped_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = Path(raw).resolve(strict=True) / "checkpoint_manifest.json"
            manifest.write_text("fixture\n", encoding="utf-8")
            expected = v2._expected_terminal_checkpoint_identity(manifest)
            original = v1._TRAINING_CHECKPOINT_CONFIG_FIELDS

            def validate(path, sha256):
                self.assertEqual(Path(path), manifest)
                self.assertEqual(sha256, v2.TERMINAL_CP644_MANIFEST_SHA256)
                self.assertEqual(
                    v1._TRAINING_CHECKPOINT_CONFIG_FIELDS,
                    v2._TERMINAL_CHECKPOINT_CONFIG_FIELDS,
                )
                return dict(expected)

            with mock.patch.object(
                v1, "validate_terminal_checkpoint_manifest", side_effect=validate
            ) as delegated:
                self.assertEqual(
                    v2.validate_terminal_checkpoint_manifest(
                        manifest, v2.TERMINAL_CP644_MANIFEST_SHA256
                    ),
                    expected,
                )
                delegated.assert_called_once()
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)
            self.assertFalse(v2._CHECKPOINT_COMPAT_ACTIVE)

            with mock.patch.object(
                v1,
                "validate_terminal_checkpoint_manifest",
                side_effect=RuntimeError("injected"),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    v2.validate_terminal_checkpoint_manifest(
                        manifest, v2.TERMINAL_CP644_MANIFEST_SHA256
                    )
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)
            self.assertFalse(v2._CHECKPOINT_COMPAT_ACTIVE)

            with mock.patch.object(
                v1, "validate_terminal_checkpoint_manifest"
            ) as delegated:
                with self.assertRaisesRegex(
                    v2.MatchedEvalV2Error, "manifest SHA differs"
                ):
                    v2.validate_terminal_checkpoint_manifest(manifest, "0" * 64)
                delegated.assert_not_called()

            hostile = dict(expected)
            hostile["training_receipt_sha256"] = "0" * 64
            with mock.patch.object(
                v1,
                "validate_terminal_checkpoint_manifest",
                return_value=hostile,
            ):
                with self.assertRaisesRegex(
                    v2.MatchedEvalV2Error, "cp644 identity differs"
                ):
                    v2.validate_terminal_checkpoint_manifest(
                        manifest, v2.TERMINAL_CP644_MANIFEST_SHA256
                    )
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)

    def test_terminal_cp644_compatibility_restores_all_hostile_mutations(self) -> None:
        original = v1._TRAINING_CHECKPOINT_CONFIG_FIELDS
        original_contents = set(original)

        def assert_restored() -> None:
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)
            self.assertEqual(original, original_contents)
            self.assertFalse(v2._CHECKPOINT_COMPAT_ACTIVE)

        with self.subTest("deleted attribute"):
            with self.assertRaisesRegex(
                v2.MatchedEvalV2Error, "was not restored"
            ):
                with v2._v1_terminal_cp644_config_compatibility():
                    del v1._TRAINING_CHECKPOINT_CONFIG_FIELDS
            assert_restored()

        with self.subTest("equal-content rebind"):
            with self.assertRaisesRegex(
                v2.MatchedEvalV2Error, "was not restored"
            ):
                with v2._v1_terminal_cp644_config_compatibility():
                    v1._TRAINING_CHECKPOINT_CONFIG_FIELDS = set(
                        v2._TERMINAL_CHECKPOINT_CONFIG_FIELDS
                    )
            assert_restored()

        with self.subTest("original alias mutation"):
            with self.assertRaisesRegex(
                v2.MatchedEvalV2Error, "was not restored"
            ):
                with v2._v1_terminal_cp644_config_compatibility():
                    original.add("hostile/config.json")
            assert_restored()

        with self.subTest("nested scope"):
            with v2._v1_terminal_cp644_config_compatibility():
                with self.assertRaisesRegex(
                    v2.MatchedEvalV2Error, "origin differs"
                ):
                    with v2._v1_terminal_cp644_config_compatibility():
                        self.fail("nested compatibility scope entered")
            assert_restored()

    def test_build_plan_cli_uses_exact_cp644_validator_and_revalidates(self) -> None:
        args = types.SimpleNamespace(
            input_manifest="/authority/input.jsonl",
            exposure_audit="/authority/exposure.json",
            source_root="/authority/source",
            checkpoint_manifest="/checkpoint/checkpoint_manifest.json",
            checkpoint_manifest_sha256=v2.TERMINAL_CP644_MANIFEST_SHA256,
            infer_lora_source="/release/infer_lora.py",
            infer_lora_source_sha256="a" * 64,
            method_source_revision="b" * 40,
            method_source_archive_sha256="c" * 64,
            ffprobe="/usr/bin/ffprobe",
            ffprobe_sha256="d" * 64,
            output_root="/fresh/media",
            output_plan="/fresh/plan.json",
        )
        parser = mock.Mock()
        parser.parse_args.return_value = args
        authority = {"rows": [], "source_bytes_verified": True}
        checkpoint = {"global_step": 644}
        plan = {"fixture": True}
        command = ["build-plan", "--opaque-fixture"]
        with mock.patch.object(v1, "build_parser", return_value=parser), mock.patch.object(
            v1, "validate_shared8_authority", return_value=authority
        ) as validate_authority, mock.patch.object(
            v2, "validate_terminal_checkpoint_manifest", return_value=checkpoint
        ) as validate_checkpoint, mock.patch.object(
            v1, "build_plan", return_value=plan
        ) as build, mock.patch.object(
            v2, "validate_plan"
        ) as revalidate, mock.patch.object(
            v1, "write_create_only", return_value="e" * 64
        ) as write, mock.patch("builtins.print") as output:
            self.assertEqual(v2.main(command), 0)
        parser.parse_args.assert_called_once_with(command)
        validate_authority.assert_called_once_with(
            args.input_manifest,
            args.exposure_audit,
            require_source_bytes=True,
            source_root=args.source_root,
        )
        validate_checkpoint.assert_called_once_with(
            args.checkpoint_manifest, args.checkpoint_manifest_sha256
        )
        build.assert_called_once()
        producer = build.call_args.kwargs["producer"]
        self.assertEqual(
            producer,
            {
                "inference_receipt_schema": v1.INFERENCE_RECEIPT_SCHEMA,
                "infer_lora_path": args.infer_lora_source,
                "infer_lora_sha256": args.infer_lora_source_sha256,
                "method_source_revision": args.method_source_revision,
                "method_source_archive_sha256": (
                    args.method_source_archive_sha256
                ),
                "ffprobe_path": args.ffprobe,
                "ffprobe_sha256": args.ffprobe_sha256,
            },
        )
        revalidate.assert_called_once_with(plan)
        write.assert_called_once_with(args.output_plan, plan)
        output.assert_called_once_with("e" * 64)

    def test_plan_validation_and_load_use_scoped_cp644_authority(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve(strict=True)
            manifest = root / "checkpoint_manifest.json"
            manifest.write_text("fixture\n", encoding="utf-8")
            checkpoint = v2._expected_terminal_checkpoint_identity(manifest)
            plan = {"checkpoint_manifest": checkpoint}
            original = v1._TRAINING_CHECKPOINT_CONFIG_FIELDS

            def validate_plan(value):
                self.assertIs(value, plan)
                self.assertEqual(
                    v1._TRAINING_CHECKPOINT_CONFIG_FIELDS,
                    v2._TERMINAL_CHECKPOINT_CONFIG_FIELDS,
                )

            with mock.patch.object(
                v1, "validate_plan", side_effect=validate_plan
            ) as delegated:
                v2.validate_plan(plan)
                delegated.assert_called_once_with(plan)
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)

            plan_path = root / "plan.json"
            plan_path.write_text("fixture\n", encoding="utf-8")

            def load_plan(path, sha256):
                self.assertEqual(Path(path), plan_path)
                self.assertEqual(sha256, "f" * 64)
                self.assertEqual(
                    v1._TRAINING_CHECKPOINT_CONFIG_FIELDS,
                    v2._TERMINAL_CHECKPOINT_CONFIG_FIELDS,
                )
                return plan

            with mock.patch.object(
                v1, "_load_plan", side_effect=load_plan
            ) as delegated:
                self.assertIs(v2.load_plan(plan_path, "f" * 64), plan)
                delegated.assert_called_once_with(plan_path, "f" * 64)
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)
            self.assertFalse(v2._CHECKPOINT_COMPAT_ACTIVE)

            hostile = copy.deepcopy(plan)
            hostile["checkpoint_manifest"]["receipt_digest"] = "0" * 64
            with mock.patch.object(v1, "validate_plan"):
                with self.assertRaisesRegex(
                    v2.MatchedEvalV2Error, "plan terminal cp644 identity differs"
                ):
                    v2.validate_plan(hostile)
            self.assertIs(v1._TRAINING_CHECKPOINT_CONFIG_FIELDS, original)

    def test_real_infer_permission_mode_0444_passes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            task, receipt = source_fixture(Path(raw))
            observed = v2.validate_real_source_authority(task, receipt)
            self.assertEqual(observed["mode"], 0o444)
            self.assertLessEqual(observed["mode"], 0o7777)

    def test_mode_type_range_and_tamper_fail_closed(self) -> None:
        hostile = (True, 0.0, -1, 0o10000, 0o400)
        for value in hostile:
            with self.subTest(mode=value), tempfile.TemporaryDirectory() as raw:
                task, receipt = source_fixture(Path(raw))
                receipt["input"]["source_video_physical_authority"]["mode"] = value
                reseal_authority(receipt)
                with self.assertRaises(v2.MatchedEvalV2Error):
                    v2.validate_real_source_authority(task, receipt)

    def test_regular_file_authority_comes_from_stable_named_inode(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task, receipt = source_fixture(root)
            source = Path(task["source_video"])
            held = root / "held.mp4"
            source.rename(held)
            source.write_bytes(b"replacement")
            source.chmod(0o444)
            with self.assertRaises(v2.MatchedEvalV2Error):
                v2.validate_real_source_authority(task, receipt)

    def test_authority_digest_and_retained_replay_are_required(self) -> None:
        fields = (
            ("input", "source_video_physical_authority_digest", "0" * 64),
            ("model_consumption", "source_video_physical_authority_digest", "0" * 64),
            ("input", "retained_source_fd_consumed", False),
            ("input", "source_video_pre_and_post_decode_rehashed", False),
            ("model_consumption", "all_ranks_use_retained_source_fd", False),
        )
        for section, field, value in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                task, receipt = source_fixture(Path(raw))
                receipt[section][field] = value
                with self.assertRaises(v2.MatchedEvalV2Error):
                    v2.validate_real_source_authority(task, receipt)

    def test_v1_stat_facade_is_task_local_and_restored(self) -> None:
        original = v1.stat
        with v2._v1_permission_mode_compatibility():
            self.assertIsNot(v1.stat, original)
            self.assertTrue(v1.stat.S_ISREG(0o444))
            self.assertTrue(v1.stat.S_ISREG(stat.S_IFREG | 0o444))
            self.assertFalse(v1.stat.S_ISREG(stat.S_IFDIR | 0o755))
        self.assertIs(v1.stat, original)
        try:
            with v2._v1_permission_mode_compatibility():
                raise RuntimeError("injected")
        except RuntimeError:
            pass
        self.assertIs(v1.stat, original)

    def test_v1_receipt_replay_is_exact_once_and_restored(self) -> None:
        original = v1._load_receipt
        receipt = {"receipt_digest": "a" * 64}
        with v2._v1_exact_receipt_replay(
            "/tmp/exact.receipt.json", receipt, "b" * 64
        ) as calls:
            observed = v1._load_receipt("/tmp/exact.receipt.json")
            self.assertIs(observed[0], receipt)
            self.assertEqual(observed[1], "b" * 64)
            with self.assertRaises(v2.MatchedEvalV2Error):
                v1._load_receipt("/tmp/exact.receipt.json")
        self.assertEqual(calls, [1])
        self.assertIs(v1._load_receipt, original)
        try:
            with v2._v1_exact_receipt_replay(
                "/tmp/exact.receipt.json", receipt, "b" * 64
            ):
                raise RuntimeError("injected")
        except RuntimeError:
            pass
        self.assertIs(v1._load_receipt, original)
        self.assertFalse(v2._RECEIPT_PATCH_ACTIVE)


if __name__ == "__main__":
    unittest.main()

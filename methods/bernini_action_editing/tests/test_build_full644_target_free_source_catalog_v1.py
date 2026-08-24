from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import full644_target_free_preference_v1 as target_free
from tools import build_full644_target_free_source_catalog_v1 as extractor


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixtures(root: Path):
    preview_rows = []
    natural_rows = []
    raw_rows = []
    source_payloads = {}
    for index in range(target_free.SOURCE_COUNT):
        iid = f"row-{index:04d}"
        instruction = f"Move the visible actor using action family {index % 28:02d}."
        instruction_sha = sha_bytes(instruction.encode("utf-8"))
        source_path = root / "source" / f"{iid}.mp4"
        target_path = root / "poison-target-index1" / f"{iid}.mp4"
        source_payload = f"source-{index}".encode("ascii")
        source_sha = sha_bytes(source_payload)
        preview = {
            "schema_version": extractor.PREVIEW_ROW_SCHEMA,
            "iid": iid,
            "group_id": f"group-{index:04d}",
            "family": f"family-{index % 28:02d}",
            "source_video_path": str(source_path),
            "source_video_sha256": source_sha,
            "target_video_path": str(target_path),
            "target_video_sha256": sha_bytes(f"target-{index}".encode("ascii")),
            "edit_instruction": instruction,
            "edit_instruction_sha256": instruction_sha,
            "preview_only": True,
            "training_authorized": False,
            "training_use_forbidden": True,
            "production_eligible": False,
            "post_video_acceptance": "pending",
        }
        preview["row_digest"] = sha_bytes(
            extractor._upstream_canonical_json_bytes(preview)
        )
        preview_rows.append(preview)
        natural_rows.append(
            {
                "schema_version": extractor.NATURAL_ROW_SCHEMA,
                "iid": iid,
                "natural_edit_instruction": instruction,
                "natural_edit_instruction_sha256": instruction_sha,
            }
        )
        source_payloads[str(source_path)] = source_payload

    preview_raw = b"".join(
        extractor._upstream_canonical_json_bytes(row) + b"\n"
        for row in preview_rows
    )
    natural_raw = b"".join(
        extractor._upstream_canonical_json_bytes(row) + b"\n"
        for row in natural_rows
    )
    preview_sha = sha_bytes(preview_raw)
    for index, preview in enumerate(preview_rows):
        iid = str(preview["iid"])
        messages = [
            {"type": "video", "has_loss": 0},
            {"type": "text", "text": preview["edit_instruction"], "has_loss": 0},
            {"type": "video_gen", "has_loss": 1},
        ]
        raw_rows.append(
            {
                "schema_version": extractor.RAW_ROW_SCHEMA,
                "iid": iid,
                "group_id": preview["group_id"],
                "family": preview["family"],
                "inputs": json.dumps(messages, sort_keys=True, separators=(",", ":")),
                "videos": [
                    {"video_path": preview["source_video_path"]},
                    {"video_path": preview["target_video_path"]},
                ],
                "source_video_path": preview["source_video_path"],
                "source_video_declared_path": preview["source_video_path"],
                "source_video_sha256": preview["source_video_sha256"],
                "edit_instruction_sha256": preview["edit_instruction_sha256"],
                "preview_manifest_sha256": preview_sha,
                "preview_row_digest": preview["row_digest"],
                "preview_row_file_sha256": sha_bytes(
                    extractor._upstream_canonical_json_bytes(preview) + b"\n"
                ),
                "strict_selection_gates_all_true": index
                < extractor.STRICT_SOURCE_COUNT,
                "preview_only": True,
                "training_authorized": False,
                "training_use_forbidden": True,
                "production_eligible": False,
                "post_video_acceptance": "pending",
            }
        )
    iid_set_sha = extractor._iid_set_sha256(
        [str(row["iid"]) for row in preview_rows]
    )
    return preview_raw, natural_raw, raw_rows, source_payloads, iid_set_sha


class Full644TargetFreeSourceExtractorTests(unittest.TestCase):
    def build(self):
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        opened = []

        def source_loader(path: Path, expected_sha: str, label: str):
            self.assertIn("source video", label)
            self.assertNotIn("poison-target-index1", str(path))
            opened.append(str(path))
            payload = payloads[str(path)]
            self.assertEqual(sha_bytes(payload), expected_sha)
            return payload, {"frame_count": 81, "fps": 25.0}

        authority = {
            "preview_manifest_sha256": sha_bytes(preview_raw),
            "natural_manifest_sha256": sha_bytes(natural_raw),
            "raw_parquet_sha256": sha_bytes(b"raw-parquet-fixture"),
            "sorted_iid_set_sha256": iid_set_sha,
        }
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY, authority, clear=False
        ):
            catalog, evidence = extractor._build_catalog_value_with_source_loader_for_tests_v1(
                preview_raw=preview_raw,
                natural_raw=natural_raw,
                raw_rows=raw_rows,
                source_loader=source_loader,
            )
        return catalog, evidence, opened, (preview_raw, natural_raw, raw_rows, payloads)

    def test_exact644_source_only_catalog_never_opens_target_media(self) -> None:
        catalog, evidence, opened, _ = self.build()
        self.assertEqual(len(catalog["rows"]), 644)
        self.assertEqual(len(opened), 644)
        self.assertEqual(evidence["source_file_open_count"], 644)
        self.assertEqual(evidence["strict_source_count"], 359)
        self.assertEqual(evidence["broad_source_count"], 285)
        self.assertFalse(evidence["access_ledger_authoritative"])
        self.assertEqual(
            evidence["source_loader_contract"],
            "INJECTED_TEST_JOIN_HELPER_NOT_FOR_PUBLICATION",
        )
        self.assertTrue(evidence["raw_videos_role_column_read"])
        self.assertTrue(evidence["raw_target_path_metadata_read_for_role_rejection"])
        self.assertNotIn("extractor_owned_target_media_open_count", evidence)
        self.assertNotIn("target_media_payload_read", evidence)
        self.assertRegex(evidence["source_probe_inventory_digest"], r"^[0-9a-f]{64}$")
        serialized = target_free.canonical_json_bytes(catalog).decode("ascii")
        self.assertNotIn("poison-target-index1", serialized)
        self.assertNotIn("target_video", serialized)
        self.assertNotIn("video_vae_latents", serialized)

    def test_raw_source_role_cannot_be_replaced_by_target_index1(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        poisoned = copy.deepcopy(raw_rows)
        poisoned[0]["source_video_path"] = str(
            root / "poison-target-index1" / "row-0000.mp4"
        )
        poisoned[0]["source_video_declared_path"] = poisoned[0]["source_video_path"]
        authority = {
            "preview_manifest_sha256": sha_bytes(preview_raw),
            "natural_manifest_sha256": sha_bytes(natural_raw),
            "sorted_iid_set_sha256": iid_set_sha,
        }
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY, authority, clear=False
        ):
            with self.assertRaisesRegex(
                extractor.SourceCatalogExtractionError, "source-role join"
            ):
                extractor._build_catalog_value_with_source_loader_for_tests_v1(
                    preview_raw=preview_raw,
                    natural_raw=natural_raw,
                    raw_rows=poisoned,
                    source_loader=lambda path, expected, label: (
                        payloads[str(path)],
                        {"frame_count": 81, "fps": 25.0},
                    ),
                )

    def test_raw_videos_index0_index1_swap_is_rejected(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        poisoned = copy.deepcopy(raw_rows)
        poisoned[0]["videos"] = list(reversed(poisoned[0]["videos"]))
        authority = {
            "preview_manifest_sha256": sha_bytes(preview_raw),
            "natural_manifest_sha256": sha_bytes(natural_raw),
            "sorted_iid_set_sha256": iid_set_sha,
        }
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY, authority, clear=False
        ):
            with self.assertRaisesRegex(
                extractor.SourceCatalogExtractionError, "source-role join"
            ):
                extractor._build_catalog_value_with_source_loader_for_tests_v1(
                    preview_raw=preview_raw,
                    natural_raw=natural_raw,
                    raw_rows=poisoned,
                    source_loader=lambda path, expected, label: (
                        payloads[str(path)],
                        {"frame_count": 81, "fps": 25.0},
                    ),
                )

    def test_raw_inputs_reject_duplicate_keys_and_boolean_loss_flags(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        authority = {
            "preview_manifest_sha256": sha_bytes(preview_raw),
            "natural_manifest_sha256": sha_bytes(natural_raw),
            "sorted_iid_set_sha256": iid_set_sha,
        }
        variants = (
            '[{"type":"video","has_loss":0,"has_loss":0},'
            '{"type":"text","text":"x","has_loss":0},'
            '{"type":"video_gen","has_loss":1}]',
            '[{"type":"video","has_loss":false},'
            '{"type":"text","text":"x","has_loss":0},'
            '{"type":"video_gen","has_loss":1}]',
        )
        for value in variants:
            poisoned = copy.deepcopy(raw_rows)
            poisoned[0]["inputs"] = value
            with self.subTest(value=value), mock.patch.dict(
                target_free.PINNED_FULL644_SOURCE_AUTHORITY, authority, clear=False
            ):
                with self.assertRaises(extractor.SourceCatalogExtractionError):
                    extractor._build_catalog_value_with_source_loader_for_tests_v1(
                        preview_raw=preview_raw,
                        natural_raw=natural_raw,
                        raw_rows=poisoned,
                        source_loader=lambda path, expected, label: (
                            payloads[str(path)],
                            {"frame_count": 81, "fps": 25.0},
                        ),
                    )

    def test_natural_instruction_mismatch_is_rejected(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        rows = extractor._parse_jsonl(natural_raw, label="fixture natural")
        rows[0]["natural_edit_instruction"] = "Perform a different action now."
        tampered = b"".join(
            extractor._upstream_canonical_json_bytes(row) + b"\n" for row in rows
        )
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY,
            {
                "preview_manifest_sha256": sha_bytes(preview_raw),
                "natural_manifest_sha256": sha_bytes(tampered),
                "sorted_iid_set_sha256": iid_set_sha,
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                extractor.SourceCatalogExtractionError, "natural row differs"
            ):
                extractor._build_catalog_value_with_source_loader_for_tests_v1(
                    preview_raw=preview_raw,
                    natural_raw=tampered,
                    raw_rows=raw_rows,
                    source_loader=lambda path, expected, label: (
                        payloads[str(path)],
                        {"frame_count": 81, "fps": 25.0},
                    ),
                )

    def test_media_probe_requires_exact81_25(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, payloads, iid_set_sha = fixtures(root)
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY,
            {
                "preview_manifest_sha256": sha_bytes(preview_raw),
                "natural_manifest_sha256": sha_bytes(natural_raw),
                "sorted_iid_set_sha256": iid_set_sha,
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                extractor.SourceCatalogExtractionError, "exact81/25"
            ):
                extractor._build_catalog_value_with_source_loader_for_tests_v1(
                    preview_raw=preview_raw,
                    natural_raw=natural_raw,
                    raw_rows=raw_rows,
                    source_loader=lambda path, expected, label: (
                        payloads[str(path)],
                        {"frame_count": 80, "fps": 25.0},
                    ),
                )

    def test_create_only_output_rejects_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "existing.json"
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(directory, flags)
            try:
                binding = extractor._create_only_file_at(
                    descriptor, path.name, b"first"
                )
                with self.assertRaises(extractor.SourceCatalogExtractionError):
                    extractor._create_only_file_at(descriptor, path.name, b"second")
            finally:
                os.close(descriptor)
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(binding["mode"], 0o444)
            self.assertTrue(binding["same_fd_replay_verified"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)

    def test_injected_loader_payload_is_rehashed_and_never_authoritative(self) -> None:
        root = Path("/fixture/full644")
        preview_raw, natural_raw, raw_rows, _, iid_set_sha = fixtures(root)
        with mock.patch.dict(
            target_free.PINNED_FULL644_SOURCE_AUTHORITY,
            {
                "preview_manifest_sha256": sha_bytes(preview_raw),
                "natural_manifest_sha256": sha_bytes(natural_raw),
                "sorted_iid_set_sha256": iid_set_sha,
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                extractor.SourceCatalogExtractionError, "unbound bytes"
            ):
                extractor._build_catalog_value_with_source_loader_for_tests_v1(
                    preview_raw=preview_raw,
                    natural_raw=natural_raw,
                    raw_rows=raw_rows,
                    source_loader=lambda path, expected, label: (
                        b"wrong-source-bytes",
                        {"frame_count": 81, "fps": 25.0},
                    ),
                )

    def test_executable_binding_replays_same_held_inode_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "ffprobe-fixture"
            path.write_bytes(b"executable-fixture-A")
            path.chmod(0o755)
            expected = sha_bytes(path.read_bytes())
            descriptor, binding, identity = extractor._open_verified_executable(
                path, expected_sha256=expected, label="fixture executable"
            )
            try:
                self.assertEqual(binding["sha256"], expected)
                extractor._replay_verified_executable(
                    descriptor,
                    path=path,
                    expected_sha256=expected,
                    expected_identity=identity,
                    label="fixture executable",
                )
                path.write_bytes(b"executable-fixture-B")
                with self.assertRaisesRegex(
                    extractor.SourceCatalogExtractionError,
                    "changed across exact644 extraction",
                ):
                    extractor._replay_verified_executable(
                        descriptor,
                        path=path,
                        expected_sha256=expected,
                        expected_identity=identity,
                        label="fixture executable",
                    )
            finally:
                os.close(descriptor)

    def test_output_tree_is_created_and_sealed_through_held_dirfds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            parent_fd, parent_identity = extractor._open_canonical_parent(parent)
            output_fd = -1
            try:
                output_fd, output_identity = extractor._create_output_directory_at(
                    parent_fd, "published"
                )
                catalog_binding = extractor._create_only_file_at(
                    output_fd, extractor.CATALOG_FILENAME, b"catalog"
                )
                receipt_binding = extractor._create_only_file_at(
                    output_fd, extractor.RECEIPT_FILENAME, b"receipt"
                )
                extractor._seal_output_directory(
                    parent_fd=parent_fd,
                    parent_path=parent,
                    expected_parent_identity=parent_identity,
                    output_fd=output_fd,
                    output_name="published",
                    expected_output_identity=output_identity,
                    expected_file_bindings=(catalog_binding, receipt_binding),
                )
                self.assertEqual(
                    stat.S_IMODE(os.fstat(output_fd).st_mode), 0o555
                )
                self.assertEqual(
                    set(os.listdir(output_fd)),
                    {extractor.CATALOG_FILENAME, extractor.RECEIPT_FILENAME},
                )
            finally:
                if output_fd >= 0:
                    os.fchmod(output_fd, 0o700)
                    os.close(output_fd)
                os.close(parent_fd)

    def test_output_seal_rejects_symlink_and_regular_file_replacement(self) -> None:
        for replacement in ("symlink", "regular"):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory).resolve()
                parent_fd, parent_identity = extractor._open_canonical_parent(parent)
                output_fd = -1
                try:
                    output_fd, output_identity = extractor._create_output_directory_at(
                        parent_fd, "published"
                    )
                    catalog_binding = extractor._create_only_file_at(
                        output_fd, extractor.CATALOG_FILENAME, b"catalog"
                    )
                    receipt_binding = extractor._create_only_file_at(
                        output_fd, extractor.RECEIPT_FILENAME, b"receipt"
                    )
                    os.unlink(extractor.CATALOG_FILENAME, dir_fd=output_fd)
                    if replacement == "symlink":
                        os.symlink(
                            "/etc/passwd",
                            extractor.CATALOG_FILENAME,
                            dir_fd=output_fd,
                        )
                    else:
                        extractor._create_only_file_at(
                            output_fd,
                            extractor.CATALOG_FILENAME,
                            b"replacement",
                        )
                    with self.assertRaises(extractor.SourceCatalogExtractionError):
                        extractor._seal_output_directory(
                            parent_fd=parent_fd,
                            parent_path=parent,
                            expected_parent_identity=parent_identity,
                            output_fd=output_fd,
                            output_name="published",
                            expected_output_identity=output_identity,
                            expected_file_bindings=(
                                catalog_binding,
                                receipt_binding,
                            ),
                        )
                finally:
                    if output_fd >= 0:
                        os.fchmod(output_fd, 0o700)
                        os.close(output_fd)
                    os.close(parent_fd)

    def test_output_seal_rechecks_exact_entries_after_directory_fchmod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            parent_fd, parent_identity = extractor._open_canonical_parent(parent)
            output_fd = -1
            try:
                output_fd, output_identity = extractor._create_output_directory_at(
                    parent_fd, "published"
                )
                catalog_binding = extractor._create_only_file_at(
                    output_fd, extractor.CATALOG_FILENAME, b"catalog"
                )
                receipt_binding = extractor._create_only_file_at(
                    output_fd, extractor.RECEIPT_FILENAME, b"receipt"
                )
                exact = [extractor.CATALOG_FILENAME, extractor.RECEIPT_FILENAME]
                with mock.patch.object(
                    extractor.os,
                    "listdir",
                    side_effect=[exact, [*exact, "late-intruder"]],
                ):
                    with self.assertRaisesRegex(
                        extractor.SourceCatalogExtractionError,
                        "post-seal output directory entries differ",
                    ):
                        extractor._seal_output_directory(
                            parent_fd=parent_fd,
                            parent_path=parent,
                            expected_parent_identity=parent_identity,
                            output_fd=output_fd,
                            output_name="published",
                            expected_output_identity=output_identity,
                            expected_file_bindings=(
                                catalog_binding,
                                receipt_binding,
                            ),
                        )
            finally:
                if output_fd >= 0:
                    os.fchmod(output_fd, 0o700)
                    os.close(output_fd)
                os.close(parent_fd)

    def test_ffprobe_contract_parses_exact81_25(self) -> None:
        output = json.dumps(
            {
                "streams": [
                    {
                        "avg_frame_rate": "25/1",
                        "r_frame_rate": "25/1",
                        "nb_read_frames": "81",
                        "nb_frames": "81",
                    }
                ]
            }
        ).encode("utf-8")
        completed = mock.Mock(stdout=output, stderr=b"")
        with mock.patch.object(extractor.subprocess, "run", return_value=completed):
            self.assertEqual(
                extractor._probe_exact81_25(Path("/source.mp4"), ffprobe="ffprobe"),
                {"frame_count": 81, "fps": 25.0},
            )

    def test_ffprobe_rejects_multiple_video_streams_or_rate_mismatch(self) -> None:
        base = {
            "avg_frame_rate": "25/1",
            "r_frame_rate": "25/1",
            "nb_read_frames": "81",
            "nb_frames": "81",
        }
        outputs = (
            {"streams": [base, dict(base)]},
            {"streams": [{**base, "r_frame_rate": "24/1"}]},
        )
        for output in outputs:
            completed = mock.Mock(
                stdout=json.dumps(output).encode("utf-8"), stderr=b""
            )
            with self.subTest(output=output), mock.patch.object(
                extractor.subprocess, "run", return_value=completed
            ):
                with self.assertRaises(extractor.SourceCatalogExtractionError):
                    extractor._probe_exact81_25(
                        Path("/source.mp4"), ffprobe="ffprobe"
                    )

    def test_held_source_detects_same_size_in_place_mutation_during_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "source.mp4"
            path.write_bytes(b"source-A")
            expected = sha_bytes(b"source-A")

            def mutate_during_probe(*args, **kwargs):
                path.write_bytes(b"source-B")
                return {"frame_count": 81, "fps": 25.0}

            with mock.patch.object(
                extractor, "_probe_exact81_25", side_effect=mutate_during_probe
            ):
                with self.assertRaisesRegex(
                    extractor.SourceCatalogExtractionError,
                    "changed across held-FD decode",
                ):
                    extractor._read_and_probe_stable_source(
                        path,
                        expected_sha256=expected,
                        label="mutation fixture",
                        ffprobe="ffprobe",
                    )


if __name__ == "__main__":
    unittest.main()

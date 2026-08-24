#!/usr/bin/env python3
"""Hostile unit tests for the fresh case01 VACE cleanplate generator."""

from __future__ import annotations

from argparse import Namespace
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "methods" / "bernini_action_editing" / "generate_case01_bone_removed_v2_vace_v1.py"
SPEC = importlib.util.spec_from_file_location("bone_removed_v2_vace_producer", SOURCE)
assert SPEC is not None and SPEC.loader is not None
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(producer.canonical_json_bytes(value) + b"\n")


def manifest_for(root: Path, path: Path, role: str) -> dict:
    entries = []
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        row = producer.file_row(child)
        entries.append(
            {
                "relative_path": child.relative_to(root).as_posix(),
                "sha256": row["sha256"],
                "size": row["size"],
            }
        )
    value = {
        "schema_version": producer.TREE_MANIFEST_SCHEMA,
        "authority_role": role,
        "inventory_policy": "exact_recursive_regular_nonsymlink_nlink1",
        "tree_root": str(root),
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(row["size"] for row in entries),
        "tree_digest": producer.object_sha256(entries),
    }
    value["manifest_digest"] = producer.object_sha256(value)
    write_json(path, value)
    return value


class GeneratorContractTests(unittest.TestCase):
    def test_transform_contract_is_frozen_fitpad_inverse(self) -> None:
        sha = "a" * 64
        file_row = {"path": "/authority/file", "sha256": sha, "size": 1}
        trace = {
            "frame_indices": list(range(81)),
            "resize_crop_applied": False,
            "digest_definition": "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
            "source_tensor": {
                "shape": [3, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
            "mask_tensor": {
                "shape": [1, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
        }
        rows = {name: file_row for name in (
            "precanvas_source_video", "precanvas_mask_video",
            "processed_source_video", "processed_mask_video",
        )}
        value = producer.transform_contract(trace=trace, media_rows=rows)
        self.assertEqual(value["fit_width"], 612)
        self.assertEqual(value["fit_height"], 640)
        self.assertEqual((value["pad_left"], value["pad_right"]), (6, 6))
        self.assertEqual(value["inverse_crop_xyxy"], [6, 0, 618, 640])
        self.assertEqual(value["frame_indices"], list(range(81)))
        self.assertEqual(value["precanvas_authority_scope"], "lossless_vace_input_authority")
        self.assertEqual(value["processed_cache_authority_scope"], "nonauthoritative_codec_diagnostic_only")

    def test_prepare_source_trace_accepts_exact_contract(self) -> None:
        sha = "b" * 64
        trace = {
            "frame_indices": list(range(81)),
            "resize_crop_applied": False,
            "digest_definition": "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
            "source_tensor": {
                "shape": [3, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
            "mask_tensor": {
                "shape": [1, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
        }
        producer.validate_prepare_source_trace(trace)

    def test_prepare_source_trace_rejects_forged_frame_ids(self) -> None:
        sha = "b" * 64
        trace = {
            "frame_indices": [0] * 81,
            "resize_crop_applied": False,
            "digest_definition": "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
            "source_tensor": {
                "shape": [3, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
            "mask_tensor": {
                "shape": [1, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
        }
        with self.assertRaises(producer.ProducerHold):
            producer.validate_prepare_source_trace(trace)

    def test_prepare_source_trace_rejects_mutation(self) -> None:
        sha = "b" * 64
        other = "c" * 64
        trace = {
            "frame_indices": list(range(81)),
            "resize_crop_applied": False,
            "digest_definition": "sha256(torch.float32 contiguous CPU little-endian C-order bytes)",
            "source_tensor": {
                "shape": [3, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": other, "unchanged": True,
            },
            "mask_tensor": {
                "shape": [1, 81, 640, 624], "dtype": "float32",
                "pre_generate_sha256": sha, "post_generate_sha256": sha, "unchanged": True,
            },
        }
        with self.assertRaises(producer.ProducerHold):
            producer.validate_prepare_source_trace(trace)

    def test_hard_composite_is_exact_and_confined(self) -> None:
        source = bytes([10, 20, 30]) * producer.FRAME_PIXELS
        donor = bytes([200, 210, 220]) * producer.FRAME_PIXELS
        support = bytearray(producer.FRAME_PIXELS)
        support[7] = 255
        support[-9] = 255
        result, outside, mismatch = producer.hard_composite_frame(source, donor, bytes(support))
        self.assertEqual(outside, 0)
        self.assertEqual(mismatch, 0)
        self.assertEqual(result[7 * 3 : 7 * 3 + 3], donor[7 * 3 : 7 * 3 + 3])
        self.assertEqual(result[0:3], source[0:3])

    def test_hard_composite_rejects_nonbinary_support(self) -> None:
        source = bytes([0]) * producer.RGB_FRAME_BYTES
        support = bytearray(producer.FRAME_PIXELS)
        support[0] = 1
        with self.assertRaises(producer.ProducerHold):
            producer.hard_composite_frame(source, source, bytes(support))

    def test_deterministic_environment_pins_hash_seed_and_ffmpeg(self) -> None:
        args = Namespace(
            ffmpeg=Path("/pinned/ffmpeg"),
            vace_root=Path("/frozen/VACE"),
            gpu_visible_device="0",
        )
        os.environ["WORLD_SIZE"] = "999"
        try:
            env = producer.deterministic_environment(args, Path("/runtime/environment"))
        finally:
            os.environ.pop("WORLD_SIZE", None)
        self.assertEqual(env["PYTHONHASHSEED"], "20260822")
        self.assertEqual(env["IMAGEIO_FFMPEG_EXE"], "/pinned/ffmpeg")
        self.assertEqual(env["PYTHONPATH"], "/frozen/VACE")
        self.assertEqual((env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"]), ("0", "1", "0"))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")

    def test_deterministic_environment_rejects_multiple_gpus(self) -> None:
        args = Namespace(
            ffmpeg=Path("/pinned/ffmpeg"),
            vace_root=Path("/frozen/VACE"),
            gpu_visible_device="0,1",
        )
        with self.assertRaises(producer.ProducerHold):
            producer.deterministic_environment(args, Path("/runtime/environment"))

    def test_ffv1_rgb_storage_is_bgr0_not_unsupported_gbrp8(self) -> None:
        source_text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"-pix_fmt", "bgr0"', source_text)
        self.assertNotIn('"-pix_fmt", "gbrp"', source_text)

    def test_official_main_traces_same_wanvace_and_restores_methods(self) -> None:
        class FakeWanVace:
            def prepare_source(self, *_args: object, **_kwargs: object) -> tuple:
                return (["source-tensor"], ["mask-tensor"], [None])

            def generate(self, *_args: object, **_kwargs: object) -> str:
                return "generated"

        original_prepare = FakeWanVace.prepare_source
        original_generate = FakeWanVace.generate

        class FakeModule:
            WanVace = FakeWanVace

            @staticmethod
            def main(call: dict) -> None:
                self = FakeWanVace()
                source, mask, refs = self.prepare_source(call)
                self.generate("prompt", source, mask, refs)

        captured = producer.run_traced_vace_entry(
            FakeModule,
            {"seed": 7},
            tensor_digest=lambda value: "digest:" + value,
        )
        self.assertEqual(captured["source"], "source-tensor")
        self.assertEqual(captured["mask"], "mask-tensor")
        self.assertEqual(captured["prepare_calls"], 1)
        self.assertEqual(captured["generate_calls"], 1)
        self.assertIs(FakeWanVace.prepare_source, original_prepare)
        self.assertIs(FakeWanVace.generate, original_generate)

    def test_vace_argv_is_explicit_and_uses_child_wrapper(self) -> None:
        args = Namespace(
            python_bin=Path("/runtime/python3.12"),
            vace_root=Path("/frozen/VACE"),
            vace_checkpoint_root=Path("/model/VACE-1.3B"),
            seed=2026082201,
        )
        value = producer.vace_argv(
            args,
            Path("/stage/precanvas.mkv"),
            Path("/stage/mask.mkv"),
            Path("/stage/out.mp4"),
        )
        self.assertEqual(value[0], "/runtime/python3.12")
        self.assertIn("_vace_child", value)
        self.assertIn("2026082201", value)
        self.assertNotIn("torchrun", value)


class AuthorityManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name).resolve()
        self.tree = self.base / "tree"
        self.tree.mkdir()
        (self.tree / "a.txt").write_bytes(b"alpha")
        nested = self.tree / "nested"
        nested.mkdir()
        (nested / "b.bin").write_bytes(b"beta")
        self.manifest = self.base / "manifest.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_replays_exact_tree(self) -> None:
        manifest_for(self.tree, self.manifest, "vace_source_tree")
        row = producer.replay_tree_manifest(self.manifest, "vace_source_tree")
        self.assertEqual(row["tree_root"], str(self.tree))
        self.assertEqual(set(row["entries"]), {"a.txt", "nested/b.bin"})

    def test_rejects_extra_file(self) -> None:
        manifest_for(self.tree, self.manifest, "vace_source_tree")
        (self.tree / "late.txt").write_bytes(b"late")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(self.manifest, "vace_source_tree")

    def test_rejects_changed_file(self) -> None:
        manifest_for(self.tree, self.manifest, "vace_source_tree")
        (self.tree / "a.txt").write_bytes(b"changed")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(self.manifest, "vace_source_tree")

    def test_rejects_wrong_role(self) -> None:
        manifest_for(self.tree, self.manifest, "vace_source_tree")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(self.manifest, "vace_checkpoint_tree")

    def test_rejects_manifest_inside_tree(self) -> None:
        inside = self.tree / "manifest.json"
        manifest_for(self.tree, inside, "vace_source_tree")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(inside, "vace_source_tree")

    def test_rejects_noncanonical_json(self) -> None:
        value = manifest_for(self.tree, self.manifest, "vace_source_tree")
        self.manifest.write_text(json.dumps(value, indent=2), encoding="utf-8")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(self.manifest, "vace_source_tree")

    def test_rejects_symlink(self) -> None:
        link = self.tree / "link.txt"
        link.symlink_to(self.tree / "a.txt")
        # Build a manifest which omits the symlink; replay must still reject it.
        link.unlink()
        manifest_for(self.tree, self.manifest, "vace_source_tree")
        link.symlink_to(self.tree / "a.txt")
        with self.assertRaises(producer.ProducerHold):
            producer.replay_tree_manifest(self.manifest, "vace_source_tree")

    def test_held_file_keeps_authenticated_inode_across_name_swap(self) -> None:
        target = self.base / "held.bin"
        replacement = self.base / "replacement.bin"
        moved = self.base / "moved.bin"
        target.write_bytes(b"authenticated")
        replacement.write_bytes(b"replacement")
        row = producer.file_row(target)
        with producer.HeldFile(row, "held test") as held:
            target.rename(moved)
            replacement.rename(target)
            self.assertEqual(held.read_bytes(), b"authenticated")
            held.verify_unchanged(require_named_identity=False)
            with self.assertRaises(producer.ProducerHold):
                held.verify_unchanged()


class SupportReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/private/tmp")
        self.base = Path(self.temp.name).resolve()
        self.mask_dir = self.base / "formal-masks"
        self.mask_dir.mkdir()
        self.source = self.base / "source.mp4"
        self.sam2 = self.base / "sam2.json"
        self.source.write_bytes(b"source")
        self.sam2.write_bytes(b"sam2")
        self.source_row = producer.file_row(self.source)
        self.sam2_row = producer.file_row(self.sam2)

        frame_masks = []
        packet_records = []
        packet_frames = []
        for index in range(producer.FRAME_COUNT):
            path = self.mask_dir / ("%05d.png" % index)
            path.write_bytes(("mask-%d" % index).encode("ascii"))
            file_fields = producer.file_row(path)
            frame_masks.append(
                {
                    "frame_index": index,
                    **file_fields,
                    "bone_and_cast_shadow_covered": True,
                    "native_resolution_reviewed": True,
                }
            )
            candidate_support = {
                "path": "masks/candidate_support/%05d.png" % index,
                "sha256": file_fields["sha256"],
                "size": file_fields["size"],
            }
            packet_records.append(candidate_support)
            packet_frames.append(
                {
                    "frame_index": index,
                    "outputs": {"candidate_support": candidate_support},
                }
            )
        packet_records.sort(key=lambda row: row["path"])
        packet_tree_digest = producer.hashlib.sha256(
            producer.canonical_json_bytes(packet_records) + b"\n"
        ).hexdigest()
        packet_manifest = {
            "schema_version": producer.SUPPORT_PACKET_SCHEMA,
            "status": producer.SUPPORT_PACKET_STATUS,
            "case_id": producer.CASE_ID,
            "iid": producer.IID,
            "fps": producer.FPS,
            "frame_count": producer.FRAME_COUNT,
            "image_size_wh": [producer.WIDTH, producer.HEIGHT],
            "candidate_is_review_passed": False,
            "contact_shadow_visual_coverage": "PENDING_TWO_EXTERNAL_REVIEWS",
            "derivation": {},
            "negative_evidence": {},
            "authority": {
                "source_video": {
                    "path": "packet/reference/source.mp4",
                    "sha256": self.source_row["sha256"],
                    "size": self.source_row["size"],
                },
                "masklet_receipt": {
                    "path": "packet/reference/sam2.json",
                    "sha256": self.sam2_row["sha256"],
                    "size": self.sam2_row["size"],
                },
            },
            "review_gate": {},
            "claim_limits": {},
            "frames": packet_frames,
            "premanifest_output_tree": packet_records,
            "premanifest_output_tree_digest": packet_tree_digest,
        }
        self.packet_manifest_path = self.base / "candidate-manifest.json"
        write_json(self.packet_manifest_path, packet_manifest)
        packet_manifest_row = producer.file_row(self.packet_manifest_path)
        self.sha256sums_path = self.base / "candidate-SHA256SUMS"
        sums = {row["path"]: row["sha256"] for row in packet_records}
        sums["manifest.json"] = packet_manifest_row["sha256"]
        self.sha256sums_path.write_bytes(
            "".join(
                "%s  %s\n" % (sums[name], name) for name in sorted(sums)
            ).encode("utf-8")
        )
        candidate_packet = {
            "manifest": packet_manifest_row,
            "sha256sums": producer.file_row(self.sha256sums_path),
            "premanifest_output_tree_digest": packet_tree_digest,
        }
        for name, value in (
            ("SUPPORT_PACKET_MANIFEST_SHA256", packet_manifest_row["sha256"]),
            ("SUPPORT_PACKET_MANIFEST_SIZE", packet_manifest_row["size"]),
            ("SUPPORT_PACKET_PREMANIFEST_DIGEST", packet_tree_digest),
        ):
            patcher = mock.patch.object(producer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.receipt_paths = []
        self.evidence_rows = []
        self.receipts = []
        external_reviews = []
        independence = {
            name: True for name in producer.EXTERNAL_INDEPENDENCE_KEYS
        }
        for slot in (1, 2):
            evidence_path = self.base / ("reviewer-%d.evidence" % slot)
            evidence_path.write_bytes(("opaque-evidence-%d" % slot).encode("ascii"))
            evidence_row = producer.file_row(evidence_path)
            receipt = {
                "schema_version": producer.EXTERNAL_REVIEW_SCHEMA,
                "reviewer_slot": slot,
                "reviewer_identity": "external-reviewer-%d" % slot,
                "reviewer_affiliation_or_role": "independent visual reviewer %d" % slot,
                "candidate_manifest_sha256": packet_manifest_row["sha256"],
                "reviewed_at_utc": "2026-08-22T%02d:00:00Z" % slot,
                "independence_attestation": copy.deepcopy(independence),
                "all_81_native_frames_reviewed": True,
                "instructions": list(producer.EXTERNAL_REVIEW_INSTRUCTIONS),
                "frames": [
                    {
                        "frame_index": index,
                        "bone_coverage": "PASS",
                        "contact_shadow_coverage": "PASS",
                        "halo_and_adjacent_ground_coverage": "PASS",
                        "dog_and_guard_protection": "PASS",
                        "boundary_edit_requested": False,
                        "notes": "native frame %d reviewed" % index,
                        "decision": "PASS",
                    }
                    for index in range(producer.FRAME_COUNT)
                ],
                "overall_decision": "PASS",
                "signature_or_external_receipt": None,
                "claim_limits_acknowledged": True,
            }
            receipt["signature_or_external_receipt"] = {
                "kind": producer.EXTERNAL_SIGNATURE_KIND,
                "review_projection_sha256": producer.object_sha256(receipt),
                "evidence_sha256": evidence_row["sha256"],
                "evidence_size": evidence_row["size"],
            }
            receipt_path = self.base / ("reviewer-%d.json" % slot)
            write_json(receipt_path, receipt)
            self.receipt_paths.append(receipt_path)
            self.evidence_rows.append(evidence_row)
            self.receipts.append(receipt)
            external_reviews.append(
                {
                    "reviewer_slot": slot,
                    "receipt": producer.file_row(receipt_path),
                    "reviewer_identity": receipt["reviewer_identity"],
                    "reviewer_affiliation_or_role": receipt["reviewer_affiliation_or_role"],
                    "reviewed_at_utc": receipt["reviewed_at_utc"],
                    "independence_attestation": copy.deepcopy(independence),
                    "signature": copy.deepcopy(receipt["signature_or_external_receipt"]),
                    "evidence": evidence_row,
                }
            )

        self.review = {
            "schema_version": producer.SUPPORT_REVIEW_SCHEMA,
            "status": producer.SUPPORT_REVIEW_STATUS,
            "case_id": producer.CASE_ID,
            "iid": producer.IID,
            "candidate_packet": candidate_packet,
            "source": self.source_row,
            "sam2_receipt": self.sam2_row,
            "external_reviews": external_reviews,
            "protocol": {
                "native_resolution_704x736": True,
                "all_81_frames_reviewed_by_each": True,
                "required_external_reviewers": 2,
                "bone_covered_all_frames_by_each": True,
                "cast_shadow_and_halo_covered_all_frames_by_each": True,
                "minimum_bone_dilation_pixels": 8,
                "old_dilate3_reused": False,
                "dog_guard_excluded_all_frames": True,
            },
            "frame_masks": frame_masks,
            "claim_limits": copy.deepcopy(producer.SUPPORT_REVIEW_CLAIM_LIMITS),
        }
        self.review_path = self.base / "review.json"
        self._write_review()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_review(self) -> None:
        self.review.pop("review_digest", None)
        self.review["review_digest"] = producer.object_sha256(self.review)
        write_json(self.review_path, self.review)

    def _store_receipt(self, index: int) -> None:
        receipt = self.receipts[index]
        write_json(self.receipt_paths[index], receipt)
        formal = self.review["external_reviews"][index]
        formal["receipt"] = producer.file_row(self.receipt_paths[index])
        formal["reviewer_identity"] = receipt["reviewer_identity"]
        formal["reviewer_affiliation_or_role"] = receipt[
            "reviewer_affiliation_or_role"
        ]
        formal["reviewed_at_utc"] = receipt["reviewed_at_utc"]
        formal["independence_attestation"] = copy.deepcopy(
            receipt["independence_attestation"]
        )
        formal["signature"] = copy.deepcopy(
            receipt["signature_or_external_receipt"]
        )
        self._write_review()

    def _resign_receipt(
        self, index: int, *, evidence_row: dict | None = None
    ) -> None:
        receipt = self.receipts[index]
        if evidence_row is None:
            evidence_row = self.review["external_reviews"][index]["evidence"]
        receipt["signature_or_external_receipt"] = None
        projection_sha256 = producer.object_sha256(receipt)
        receipt["signature_or_external_receipt"] = {
            "kind": producer.EXTERNAL_SIGNATURE_KIND,
            "review_projection_sha256": projection_sha256,
            "evidence_sha256": evidence_row["sha256"],
            "evidence_size": evidence_row["size"],
        }
        self.review["external_reviews"][index]["evidence"] = evidence_row
        self._store_receipt(index)

    def _assert_held(self) -> None:
        with self.assertRaises(producer.ProducerHold):
            producer.validate_support_review(
                self.review_path,
                source_row=self.source_row,
                sam2_row=self.sam2_row,
            )

    def test_accepts_closed_review_and_compacts_rows(self) -> None:
        review_row, rows = producer.validate_support_review(
            self.review_path,
            source_row=self.source_row,
            sam2_row=self.sam2_row,
        )
        self.assertEqual(review_row["path"], str(self.review_path))
        self.assertEqual(len(rows), producer.FRAME_COUNT)
        self.assertEqual(set(rows[0]), {"frame_index", "path", "sha256", "size"})
        self.assertNotEqual(
            rows[0]["path"],
            "masks/candidate_support/00000.png",
        )

    def test_rejects_one_reviewer(self) -> None:
        self.review["external_reviews"].pop()
        self._write_review()
        self._assert_held()

    def test_rejects_duplicated_identity(self) -> None:
        self.receipts[1]["reviewer_identity"] = self.receipts[0][
            "reviewer_identity"
        ]
        self._resign_receipt(1)
        self._assert_held()

    def test_rejects_duplicated_receipt(self) -> None:
        self.review["external_reviews"][1]["receipt"] = copy.deepcopy(
            self.review["external_reviews"][0]["receipt"]
        )
        self._write_review()
        self._assert_held()

    def test_rejects_duplicated_evidence(self) -> None:
        duplicate = copy.deepcopy(self.review["external_reviews"][0]["evidence"])
        self._resign_receipt(1, evidence_row=duplicate)
        self._assert_held()

    def test_rejects_manifest_binding_mismatch(self) -> None:
        self.receipts[0]["candidate_manifest_sha256"] = "f" * 64
        self._resign_receipt(0)
        self._assert_held()

    def test_rejects_noncanonical_external_receipt(self) -> None:
        self.receipt_paths[0].write_text(
            json.dumps(self.receipts[0], indent=2), encoding="utf-8"
        )
        self.review["external_reviews"][0]["receipt"] = producer.file_row(
            self.receipt_paths[0]
        )
        self._write_review()
        self._assert_held()

    def test_rejects_projection_mismatch(self) -> None:
        self.receipts[0]["signature_or_external_receipt"][
            "review_projection_sha256"
        ] = "e" * 64
        self._store_receipt(0)
        self._assert_held()

    def test_rejects_evidence_mismatch(self) -> None:
        self.receipts[0]["signature_or_external_receipt"][
            "evidence_sha256"
        ] = "d" * 64
        self._store_receipt(0)
        self._assert_held()

    def test_rejects_one_frame_fail(self) -> None:
        frame = self.receipts[0]["frames"][17]
        frame["contact_shadow_coverage"] = "FAIL"
        frame["decision"] = "FAIL"
        self._resign_receipt(0)
        self._assert_held()

    def test_rejects_one_frame_boundary_edit(self) -> None:
        self.receipts[0]["frames"][18]["boundary_edit_requested"] = True
        self._resign_receipt(0)
        self._assert_held()

    def test_rejects_one_frame_pending(self) -> None:
        frame = self.receipts[1]["frames"][19]
        frame["halo_and_adjacent_ground_coverage"] = "PENDING"
        frame["decision"] = "PENDING"
        self._resign_receipt(1)
        self._assert_held()

    def test_rejects_extra_packet_inventory_file(self) -> None:
        self.sha256sums_path.write_bytes(
            self.sha256sums_path.read_bytes() + ("a" * 64 + "  extra.bin\n").encode("ascii")
        )
        self.review["candidate_packet"]["sha256sums"] = producer.file_row(
            self.sha256sums_path
        )
        self._write_review()
        self._assert_held()

    def test_rejects_bad_claim(self) -> None:
        self.review["claim_limits"]["reviewer_identity_verified_by_generator"] = True
        self._write_review()
        self._assert_held()

    def test_rejects_old_dilate3_claim(self) -> None:
        self.review["protocol"]["old_dilate3_reused"] = True
        self._write_review()
        self._assert_held()

    def test_rejects_unreviewed_frame(self) -> None:
        self.review["frame_masks"][40]["native_resolution_reviewed"] = False
        self._write_review()
        self._assert_held()

    def test_rejects_bool_reviewer_slot(self) -> None:
        self.review["external_reviews"][0]["reviewer_slot"] = True
        self._write_review()
        self._assert_held()

    def test_rejects_bool_receipt_reviewer_slot(self) -> None:
        self.receipts[0]["reviewer_slot"] = True
        self._resign_receipt(0)
        self._assert_held()

    def test_rejects_bool_external_frame_index(self) -> None:
        self.receipts[0]["frames"][0]["frame_index"] = False
        self._resign_receipt(0)
        self._assert_held()

    def test_rejects_bool_formal_mask_frame_index(self) -> None:
        self.review["frame_masks"][0]["frame_index"] = False
        self._write_review()
        self._assert_held()

    def test_rejects_invalid_calendar_timestamp(self) -> None:
        self.receipts[0]["reviewed_at_utc"] = "2026-13-22T01:00:00Z"
        self._resign_receipt(0)
        self._assert_held()


if __name__ == "__main__":
    unittest.main()

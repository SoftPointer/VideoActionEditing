#!/usr/bin/env python3
"""Hostile tests for the case01 dual support-review promotion boundary."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import py_compile
import struct
import tempfile
import unittest
from unittest import mock
import zlib


ROOT = Path(__file__).resolve().parents[3]
PROMOTER_SOURCE = (
    ROOT
    / "methods/bernini_action_editing/tools/"
    / "promote_case01_bone_contact_support_dual_review_v1.py"
)
GENERATOR_SOURCE = (
    ROOT
    / "methods/bernini_action_editing/"
    / "generate_case01_bone_removed_v2_vace_v1.py"
)
ACCEPTANCE_SOURCE = (
    ROOT
    / "methods/bernini_action_editing/tools/"
    / "case01_bone_removed_v2_acceptance_v1.py"
)
CANONICAL_TMP = Path(tempfile.gettempdir()).resolve()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promoter = load_module("case01_support_dual_review_promoter", PROMOTER_SOURCE)
generator = load_module("case01_support_dual_review_test_generator", GENERATOR_SOURCE)
acceptance = load_module("case01_support_dual_review_test_acceptance", ACCEPTANCE_SOURCE)


def write_json(path: Path, value: dict) -> None:
    path.write_bytes(generator.canonical_json_bytes(value) + b"\n")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def grayscale_png(value: int) -> bytes:
    scanline = b"\x00" + bytes((value,)) * generator.WIDTH
    pixels = scanline * generator.HEIGHT
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                generator.WIDTH,
                generator.HEIGHT,
                8,
                0,
                0,
                0,
                0,
            ),
        )
        + png_chunk(b"IDAT", zlib.compress(pixels))
        + png_chunk(b"IEND", b"")
    )


def file_row(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


class DualReviewPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=str(CANONICAL_TMP))
        self.base = Path(self.temporary.name).resolve()
        self.packet = self.base / "packet"
        self.packet.mkdir()
        (self.packet / "masks").mkdir()
        self.packet_masks = self.packet / "masks/candidate_support"
        self.packet_masks.mkdir()
        self.inputs = self.base / "external-inputs"
        self.inputs.mkdir()
        self.output = self.base / "promoted"

        self.source = self.base / "deployment-source.mp4"
        self.sam2 = self.base / "deployment-sam2.json"
        self.source.write_bytes(b"synthetic-source-authority")
        self.sam2.write_bytes(b"synthetic-sam2-authority")
        self.source_row = file_row(self.source)
        self.sam2_row = file_row(self.sam2)

        records = []
        frames = []
        for index in range(generator.FRAME_COUNT):
            path = self.packet_masks / ("%05d.png" % index)
            path.write_bytes(grayscale_png(index))
            row = file_row(path)
            relative = "masks/candidate_support/%05d.png" % index
            support = {
                "path": relative,
                "sha256": row["sha256"],
                "size": row["size"],
            }
            records.append(support)
            frames.append(
                {
                    "frame_index": index,
                    "outputs": {"candidate_support": support},
                }
            )
        records.sort(key=lambda row: row["path"])
        tree_digest = hashlib.sha256(
            generator.canonical_json_bytes(records) + b"\n"
        ).hexdigest()
        self.packet_manifest = {
            "schema_version": generator.SUPPORT_PACKET_SCHEMA,
            "status": generator.SUPPORT_PACKET_STATUS,
            "case_id": generator.CASE_ID,
            "iid": generator.IID,
            "fps": generator.FPS,
            "frame_count": generator.FRAME_COUNT,
            "image_size_wh": [generator.WIDTH, generator.HEIGHT],
            "candidate_is_review_passed": False,
            "contact_shadow_visual_coverage": "PENDING_TWO_EXTERNAL_REVIEWS",
            "derivation": {},
            "negative_evidence": {},
            "authority": {
                "source_video": {
                    "path": "/review-host/source.mp4",
                    "sha256": self.source_row["sha256"],
                    "size": self.source_row["size"],
                },
                "masklet_receipt": {
                    "path": "/review-host/sam2.json",
                    "sha256": self.sam2_row["sha256"],
                    "size": self.sam2_row["size"],
                },
            },
            "review_gate": {},
            "claim_limits": {},
            "frames": frames,
            "premanifest_output_tree": records,
            "premanifest_output_tree_digest": tree_digest,
        }
        self.packet_manifest_path = self.packet / "manifest.json"
        write_json(self.packet_manifest_path, self.packet_manifest)
        self.packet_manifest_row = file_row(self.packet_manifest_path)
        sums = {row["path"]: row["sha256"] for row in records}
        sums["manifest.json"] = self.packet_manifest_row["sha256"]
        (self.packet / "SHA256SUMS").write_bytes(
            "".join(
                "%s  %s\n" % (sums[name], name) for name in sorted(sums)
            ).encode("utf-8")
        )
        self.packet_tree_digest = tree_digest

        for module in (promoter, generator, acceptance):
            for name, value in (
                ("SUPPORT_PACKET_MANIFEST_SHA256", self.packet_manifest_row["sha256"]),
                ("SUPPORT_PACKET_MANIFEST_SIZE", self.packet_manifest_row["size"]),
                ("SUPPORT_PACKET_PREMANIFEST_DIGEST", tree_digest),
            ):
                if hasattr(module, name):
                    patcher = mock.patch.object(module, name, value)
                    patcher.start()
                    self.addCleanup(patcher.stop)
        for name, value in (
            ("PACKET_MANIFEST_SHA256", self.packet_manifest_row["sha256"]),
            ("PACKET_MANIFEST_SIZE", self.packet_manifest_row["size"]),
            ("PACKET_PREMANIFEST_DIGEST", tree_digest),
        ):
            patcher = mock.patch.object(promoter, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (
            ("SOURCE_SHA256", self.source_row["sha256"]),
            ("SOURCE_SIZE", self.source_row["size"]),
            ("SAM2_RECEIPT_SHA256", self.sam2_row["sha256"]),
            ("SAM2_RECEIPT_SIZE", self.sam2_row["size"]),
        ):
            patcher = mock.patch.object(generator, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        load_patcher = mock.patch.object(promoter, "_load_generator", return_value=generator)
        load_patcher.start()
        self.addCleanup(load_patcher.stop)

        self.evidence_paths = []
        self.receipt_paths = []
        self.receipts = []
        for slot in (1, 2):
            evidence = self.inputs / ("reviewer-%d.evidence" % slot)
            evidence.write_bytes(("opaque-detached-evidence-%d" % slot).encode("ascii"))
            receipt = self._base_receipt(slot)
            self.evidence_paths.append(evidence)
            self.receipt_paths.append(self.inputs / ("reviewer-%d.json" % slot))
            self.receipts.append(receipt)
            self._resign(slot - 1)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _base_receipt(self, slot: int) -> dict:
        return {
            "schema_version": generator.EXTERNAL_REVIEW_SCHEMA,
            "reviewer_slot": slot,
            "reviewer_identity": "Independent Human Reviewer %d" % slot,
            "reviewer_affiliation_or_role": "external visual auditor %d" % slot,
            "candidate_manifest_sha256": self.packet_manifest_row["sha256"],
            "reviewed_at_utc": "2026-08-23T%02d:00:00Z" % slot,
            "independence_attestation": {
                key: True for key in generator.EXTERNAL_INDEPENDENCE_KEYS
            },
            "all_81_native_frames_reviewed": True,
            "instructions": list(generator.EXTERNAL_REVIEW_INSTRUCTIONS),
            "frames": [
                {
                    "frame_index": index,
                    "bone_coverage": "PASS",
                    "contact_shadow_coverage": "PASS",
                    "halo_and_adjacent_ground_coverage": "PASS",
                    "dog_and_guard_protection": "PASS",
                    "boundary_edit_requested": False,
                    "notes": "native frame %d inspected" % index,
                    "decision": "PASS",
                }
                for index in range(generator.FRAME_COUNT)
            ],
            "overall_decision": "PASS",
            "signature_or_external_receipt": None,
            "claim_limits_acknowledged": True,
        }

    def _resign(self, index: int) -> None:
        receipt = self.receipts[index]
        evidence = self.evidence_paths[index].read_bytes()
        receipt["signature_or_external_receipt"] = None
        projection = generator.object_sha256(receipt)
        receipt["signature_or_external_receipt"] = {
            "kind": generator.EXTERNAL_SIGNATURE_KIND,
            "review_projection_sha256": projection,
            "evidence_sha256": hashlib.sha256(evidence).hexdigest(),
            "evidence_size": len(evidence),
        }
        write_json(self.receipt_paths[index], receipt)

    def _completed_draft(self, index: int) -> tuple[Path, dict]:
        draft = json.loads(json.dumps(self.receipts[index]))
        draft["candidate_manifest_sha256"] = None
        draft["signature_or_external_receipt"] = None
        path = self.inputs / ("reviewer-%d.completed-draft.json" % (index + 1))
        path.write_text(
            json.dumps(draft, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path, draft

    def _seal_review(self, index: int, draft: Path, output: Path):
        return promoter.seal_review(
            packet_root=self.packet,
            reviewer_slot=index + 1,
            completed_draft=draft,
            detached_evidence=self.evidence_paths[index],
            output_root=output,
        )

    def _promote(self, **overrides):
        arguments = {
            "packet_root": self.packet,
            "reviewer_receipts": tuple(self.receipt_paths),
            "reviewer_evidence": tuple(self.evidence_paths),
            "output_root": self.output,
            "source_path": self.source,
            "sam2_receipt_path": self.sam2,
        }
        arguments.update(overrides)
        return promoter.promote(**arguments)

    def _validate_only(self, **overrides):
        arguments = {
            "packet_root": self.packet,
            "reviewer_receipts": tuple(self.receipt_paths),
            "reviewer_evidence": tuple(self.evidence_paths),
            "source_path": self.source,
            "sam2_receipt_path": self.sam2,
        }
        arguments.update(overrides)
        return promoter.validate_only(**arguments)

    def _write_traps(self):
        stack = contextlib.ExitStack()
        for owner, name in (
            (promoter.tempfile, "mkdtemp"),
            (promoter, "_write_stage_file"),
            (promoter, "_copy_stage_file"),
            (promoter, "_fsync_directory"),
            (promoter, "_fsync_tree"),
            (promoter, "_rename_noreplace"),
            (promoter.os, "write"),
            (promoter.os, "fsync"),
        ):
            stack.enter_context(
                mock.patch.object(
                    owner,
                    name,
                    side_effect=AssertionError("validate-only attempted a write"),
                )
            )
        return stack

    def _tree_snapshot(self) -> dict:
        snapshot = {}
        for path in sorted((self.base, *self.base.rglob("*")), key=str):
            named = path.lstat()
            digest = None
            if path.is_file() and not path.is_symlink():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshot[path.relative_to(self.base).as_posix()] = (
                int(named.st_mode),
                int(named.st_nlink),
                int(named.st_ino),
                int(named.st_size),
                digest,
            )
        return snapshot

    def _packet_case_alias(self) -> Path:
        alias = self.packet.with_name(self.packet.name.upper())
        if not os.path.lexists(alias) or not os.path.samefile(alias, self.packet):
            self.skipTest("filesystem is case-sensitive for the packet path")
        self.assertNotEqual(str(alias), str(self.packet))
        return alias

    def _assert_hold(self, **overrides) -> None:
        with self.assertRaises(promoter.SupportPromotionHold):
            self._promote(**overrides)
        self.assertFalse(os.path.lexists(self.output))

    def _assert_preflight_hold(self, **overrides) -> None:
        with self._write_traps(), self.assertRaises(promoter.SupportPromotionHold):
            self._validate_only(**overrides)
        self.assertFalse(os.path.lexists(self.output))

    def _rewrite_promotion_and_checksums(self, promotion: dict) -> None:
        promotion_path = self.output / "promotion_receipt.json"
        promotion_path.chmod(0o600)
        write_json(promotion_path, promotion)
        promotion_path.chmod(0o400)
        records = promoter._bundle_records(self.output, exclude_sums=True)
        sums = self.output / "SHA256SUMS"
        sums.chmod(0o600)
        sums.write_bytes(
            "".join(
                "%s  %s\n" % (row["sha256"], row["path"])
                for row in records
            ).encode("utf-8")
        )
        sums.chmod(0o400)

    def test_success_is_path_bound_create_only_and_generator_replays_it(self) -> None:
        result = self._promote()
        self.assertEqual(result["root"], str(self.output))
        self.assertTrue((self.output / "COMPLETE").is_file())
        promotion = json.loads(
            (self.output / "promotion_receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(promotion["schema_version"], promoter.PROMOTION_SCHEMA)
        self.assertEqual(promotion["status"], promoter.PROMOTION_STATUS)
        self.assertIn("PATH_BOUND", promotion["status"])
        self.assertNotIn("PORTABLE", promotion["status"])
        self.assertEqual(
            {path.name for path in (self.output / "masks/candidate_support").iterdir()},
            {"%05d.png" % index for index in range(generator.FRAME_COUNT)},
        )
        formal_path, formal = generator.load_canonical_json(
            self.output / "support_review_receipt.json", "formal support"
        )
        self.assertEqual(formal_path.parent, self.output)
        self.assertEqual(len(formal["external_reviews"]), 2)
        self.assertTrue(
            all(
                self.output in Path(row["receipt"]["path"]).parents
                for row in formal["external_reviews"]
            )
        )
        _, rows = generator.validate_support_review(
            formal_path,
            source_row=self.source_row,
            sam2_row=self.sam2_row,
        )
        self.assertEqual(len(rows), generator.FRAME_COUNT)
        payloads, replay = acceptance._replay_support_review(
            formal,
            producer={
                "source": self.source_row,
                "mask_authority": {"receipt": self.sam2_row},
                "support": {"frame_masks": rows},
            },
        )
        self.assertEqual(len(payloads), generator.FRAME_COUNT)
        self.assertEqual(replay["frame_count"], generator.FRAME_COUNT)
        self.assertEqual(replay["review_digest"], formal["review_digest"])
        self.assertEqual(
            replay["reviewer_identities"],
            [row["reviewer_identity"] for row in formal["external_reviews"]],
        )
        self.assertEqual(
            replay["external_review_receipt_sha256s"],
            [row["receipt"]["sha256"] for row in formal["external_reviews"]],
        )
        self.assertFalse(any(self.base.glob(".promoted.partial.*")))

    def test_validate_only_replays_twice_without_filesystem_output(self) -> None:
        before = self._tree_snapshot()
        with self._write_traps():
            result = self._validate_only()
        after = self._tree_snapshot()
        self.assertEqual(after, before)
        self.assertEqual(result["schema_version"], promoter.PREFLIGHT_SCHEMA)
        self.assertEqual(result["status"], promoter.PREFLIGHT_STATUS)
        self.assertEqual(result["packet"]["support_frame_count"], generator.FRAME_COUNT)
        self.assertEqual(len(result["external_reviews"]), 2)
        self.assertFalse(result["claim_limits"]["formal_support_authority_created"])
        self.assertFalse(result["validation_scope"]["publication_attempted"])
        self.assertNotIn("review_digest", result)
        self.assertNotIn("promotion_digest", result)
        self.assertFalse(any(self.base.glob(".*.partial.*")))

    def test_seal_two_pretty_completed_pass_drafts_then_validate_pair(self) -> None:
        sealed_roots = []
        drafts = []
        for index in range(2):
            draft_path, draft = self._completed_draft(index)
            output = self.base / ("sealed-review-%d" % (index + 1))
            evidence_before = self.evidence_paths[index].read_bytes()
            draft_before = draft_path.read_bytes()
            result = self._seal_review(index, draft_path, output)
            self.assertEqual(
                result["submission"]["status"],
                promoter.REVIEW_SUBMISSION_PASS_STATUS,
            )
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                {
                    "completed_draft.input.json",
                    "receipt.json",
                    "evidence.bin",
                    "submission.json",
                    "SHA256SUMS",
                    "COMPLETE",
                },
            )
            self.assertEqual(self.evidence_paths[index].read_bytes(), evidence_before)
            self.assertEqual(draft_path.read_bytes(), draft_before)
            sealed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
            expected = json.loads(json.dumps(draft))
            expected["candidate_manifest_sha256"] = self.packet_manifest_row["sha256"]
            expected["signature_or_external_receipt"] = sealed[
                "signature_or_external_receipt"
            ]
            self.assertEqual(sealed, expected)
            payload = (output / "receipt.json").read_bytes()
            self.assertEqual(payload, generator.canonical_json_bytes(sealed) + b"\n")
            self.assertFalse(
                result["submission"]["promotion_eligible_by_this_ballot_alone"]
            )
            self.assertEqual(
                result["submission"]["claim_limits"],
                promoter.REVIEW_SUBMISSION_CLAIM_LIMITS,
            )
            sealed_roots.append(output)
            drafts.append(draft)
        validated = promoter.validate_only(
            packet_root=self.packet,
            reviewer_receipts=tuple(root / "receipt.json" for root in sealed_roots),
            reviewer_evidence=tuple(root / "evidence.bin" for root in sealed_roots),
            source_path=self.source,
            sam2_receipt_path=self.sam2,
        )
        self.assertEqual(validated["status"], promoter.PREFLIGHT_STATUS)

    def test_seal_fail_submission_is_preserved_but_never_promotable(self) -> None:
        draft_path, draft = self._completed_draft(0)
        draft["frames"][17]["contact_shadow_coverage"] = "FAIL"
        draft["frames"][17]["decision"] = "FAIL"
        draft["frames"][17]["notes"] = "shadow apron misses the contact region"
        draft["overall_decision"] = "FAIL"
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        sealed = self.base / "sealed-fail-review-1"
        result = self._seal_review(0, draft_path, sealed)
        self.assertEqual(
            result["submission"]["status"],
            promoter.REVIEW_SUBMISSION_FAIL_STATUS,
        )
        self.assertEqual(result["submission"]["overall_decision"], "FAIL")
        self._assert_preflight_hold(
            reviewer_receipts=(sealed / "receipt.json", self.receipt_paths[1]),
            reviewer_evidence=(sealed / "evidence.bin", self.evidence_paths[1]),
        )
        self._assert_hold(
            reviewer_receipts=(sealed / "receipt.json", self.receipt_paths[1]),
            reviewer_evidence=(sealed / "evidence.bin", self.evidence_paths[1]),
        )
        self.assertTrue((sealed / "COMPLETE").is_file())

    def test_sealer_rejects_pending_human_fields_before_staging(self) -> None:
        draft_path, draft = self._completed_draft(0)
        draft["reviewer_identity"] = None
        draft["frames"][0]["bone_coverage"] = "PENDING"
        draft["frames"][0]["decision"] = "PENDING"
        draft["overall_decision"] = "PENDING"
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        output = self.base / "invalid-review-submission"
        with self._write_traps(), self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, output)
        self.assertFalse(os.path.lexists(output))
        self.assertFalse(any(self.base.glob(".invalid-review-submission.partial.*")))

    def test_sealer_requires_null_machine_fields_and_consistent_decisions(self) -> None:
        draft_path, draft = self._completed_draft(0)
        draft["candidate_manifest_sha256"] = self.packet_manifest_row["sha256"]
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        output = self.base / "wrong-machine-field"
        with self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, output)
        self.assertFalse(os.path.lexists(output))

        draft["candidate_manifest_sha256"] = None
        draft["frames"][3]["boundary_edit_requested"] = True
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, output)
        self.assertFalse(os.path.lexists(output))

    def test_sealer_rejects_empty_evidence_before_staging(self) -> None:
        draft_path, _ = self._completed_draft(0)
        self.evidence_paths[0].write_bytes(b"")
        output = self.base / "empty-evidence-submission"
        with self._write_traps(), self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, output)
        self.assertFalse(os.path.lexists(output))

    def test_case_alias_cannot_receive_sealer_or_promotion_output(self) -> None:
        alias = self._packet_case_alias()
        draft_path, _ = self._completed_draft(0)
        sealed = alias / "alias-sealed-review"
        with self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, sealed)
        self.assertFalse(os.path.lexists(sealed))
        self.assertFalse(any(self.packet.glob(".alias-sealed-review.partial.*")))

        promoted = alias / "alias-promoted-review"
        self._assert_hold(output_root=promoted)
        self.assertFalse(os.path.lexists(promoted))
        self.assertFalse(any(self.packet.glob(".alias-promoted-review.partial.*")))

    def test_case_alias_rejects_pair_receipt_and_evidence_before_open(self) -> None:
        alias = self._packet_case_alias()
        packet = promoter.replay_packet(self.packet)
        with mock.patch.object(
            promoter,
            "_load_canonical_json",
            side_effect=AssertionError("aliased packet receipt must not be opened"),
        ), self.assertRaises(promoter.SupportPromotionHold):
            promoter._review_inputs(
                generator,
                packet,
                (alias / "manifest.json", self.receipt_paths[1]),
                tuple(self.evidence_paths),
            )
        with mock.patch.object(
            promoter,
            "_load_canonical_json",
            side_effect=AssertionError("aliased packet evidence must be rejected first"),
        ), self.assertRaises(promoter.SupportPromotionHold):
            promoter._review_inputs(
                generator,
                packet,
                tuple(self.receipt_paths),
                (alias / "SHA256SUMS", self.evidence_paths[1]),
            )

    def test_case_alias_rejects_sealer_draft_and_evidence_before_open(self) -> None:
        alias = self._packet_case_alias()
        output = self.base / "outside-review-submission"
        with mock.patch.object(
            promoter,
            "_load_json_draft",
            side_effect=AssertionError("aliased packet draft must not be opened"),
        ), self.assertRaises(promoter.SupportPromotionHold):
            promoter.seal_review(
                packet_root=self.packet,
                reviewer_slot=1,
                completed_draft=alias / "manifest.json",
                detached_evidence=self.evidence_paths[0],
                output_root=output,
            )
        draft_path, _ = self._completed_draft(0)
        packet = promoter.replay_packet(self.packet)
        with mock.patch.object(
            promoter,
            "_load_json_draft",
            wraps=promoter._load_json_draft,
        ), mock.patch.object(
            promoter,
            "_stable_bytes",
            side_effect=AssertionError("aliased packet evidence must not be opened"),
        ), self.assertRaises(promoter.SupportPromotionHold):
            promoter._review_draft_input(
                generator,
                packet,
                expected_slot=1,
                draft_path_value=draft_path,
                evidence_path_value=alias / "SHA256SUMS",
            )
        self.assertFalse(os.path.lexists(output))

    def test_sealer_refuses_existing_output_without_mutation(self) -> None:
        draft_path, _ = self._completed_draft(0)
        output = self.base / "existing-review-submission"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_bytes(b"untouched")
        with self.assertRaises(promoter.SupportPromotionHold):
            self._seal_review(0, draft_path, output)
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_sealer_postpublication_fault_reports_final_quarantine(self) -> None:
        draft_path, _ = self._completed_draft(0)
        output = self.base / "late-fault-review-submission"
        original = promoter._replay_review_submission

        def fail_final(root, *args, **kwargs):
            if Path(root) == output:
                raise promoter.SupportPromotionHold("synthetic final replay fault")
            return original(root, *args, **kwargs)

        with mock.patch.object(
            promoter,
            "_replay_review_submission",
            side_effect=fail_final,
        ), self.assertRaisesRegex(
            promoter.SupportPromotionHold,
            "treat bundle as quarantined",
        ):
            self._seal_review(0, draft_path, output)
        self.assertTrue(output.is_dir())
        self.assertFalse(any(self.base.glob(".late-fault-review-submission.partial.*")))

    def test_sealer_reloads_generator_after_publication(self) -> None:
        draft_path, _ = self._completed_draft(0)
        output = self.base / "published-generator-drift-review-submission"
        calls = {"count": 0}

        def reload_generator():
            calls["count"] += 1
            if calls["count"] == 3:
                raise promoter.SupportPromotionHold(
                    "synthetic generator drift after review publication"
                )
            return generator

        with mock.patch.object(
            promoter,
            "_load_generator",
            side_effect=reload_generator,
        ), self.assertRaisesRegex(
            promoter.SupportPromotionHold,
            "treat bundle as quarantined",
        ):
            self._seal_review(0, draft_path, output)
        self.assertEqual(calls["count"], 3)
        self.assertTrue(output.is_dir())
        self.assertFalse(
            any(self.base.glob(".published-generator-drift-review-submission.partial.*"))
        )

    def test_seal_review_cli_emits_canonical_pass_summary(self) -> None:
        class BinaryCapture:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        draft_path, _ = self._completed_draft(0)
        output = self.base / "cli-sealed-review"
        captured = BinaryCapture()
        arguments = [
            "seal-review",
            "--packet-root",
            str(self.packet),
            "--reviewer-slot",
            "1",
            "--completed-draft",
            str(draft_path),
            "--detached-evidence",
            str(self.evidence_paths[0]),
            "--output-root",
            str(output),
        ]
        with mock.patch.object(promoter.sys, "stdout", captured), mock.patch.object(
            promoter,
            "_promote_once",
            side_effect=AssertionError("seal-review entered promotion"),
        ), mock.patch.object(
            promoter,
            "_validate_only_once",
            side_effect=AssertionError("seal-review entered validate-only"),
        ):
            self.assertEqual(promoter.main(arguments), 0)
        payload = captured.buffer.getvalue()
        parsed = json.loads(payload.decode("utf-8"))
        self.assertEqual(payload, promoter.canonical_json_bytes(parsed) + b"\n")
        self.assertEqual(
            parsed["submission"]["status"],
            promoter.REVIEW_SUBMISSION_PASS_STATUS,
        )
        self.assertTrue((output / "COMPLETE").is_file())

    def test_validate_only_cli_routes_without_promotion_and_emits_one_lf(self) -> None:
        class BinaryCapture:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        captured = BinaryCapture()
        arguments = [
            "validate-only",
            "--packet-root",
            str(self.packet),
            "--reviewer-1-receipt",
            str(self.receipt_paths[0]),
            "--reviewer-1-evidence",
            str(self.evidence_paths[0]),
            "--reviewer-2-receipt",
            str(self.receipt_paths[1]),
            "--reviewer-2-evidence",
            str(self.evidence_paths[1]),
            "--source",
            str(self.source),
            "--sam2-receipt",
            str(self.sam2),
        ]
        with mock.patch.object(promoter.sys, "stdout", captured), mock.patch.object(
            promoter,
            "_promote_once",
            side_effect=AssertionError("validate-only entered promotion"),
        ), self._write_traps():
            self.assertEqual(promoter.main(arguments), 0)
        payload = captured.buffer.getvalue()
        self.assertTrue(payload.endswith(b"\n"))
        self.assertFalse(payload.endswith(b"\n\n"))
        parsed = json.loads(payload.decode("utf-8"))
        self.assertEqual(payload, promoter.canonical_json_bytes(parsed) + b"\n")
        self.assertEqual(parsed["status"], promoter.PREFLIGHT_STATUS)

    def test_validate_only_parser_does_not_accept_output_root(self) -> None:
        arguments = [
            "validate-only",
            "--packet-root",
            str(self.packet),
            "--reviewer-1-receipt",
            str(self.receipt_paths[0]),
            "--reviewer-1-evidence",
            str(self.evidence_paths[0]),
            "--reviewer-2-receipt",
            str(self.receipt_paths[1]),
            "--reviewer-2-evidence",
            str(self.evidence_paths[1]),
            "--output-root",
            str(self.output),
        ]
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
            SystemExit
        ) as caught:
            promoter._parser().parse_args(arguments)
        self.assertEqual(caught.exception.code, 2)

    def test_verify_packet_cli_double_replays_without_writes(self) -> None:
        class BinaryCapture:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        captured = BinaryCapture()
        before = self._tree_snapshot()
        original = promoter.replay_packet
        calls = {"count": 0}

        def counted_replay(root):
            calls["count"] += 1
            return original(root)

        with self._write_traps(), mock.patch.object(
            promoter,
            "replay_packet",
            side_effect=counted_replay,
        ), mock.patch.object(
            promoter,
            "_review_inputs",
            side_effect=AssertionError("verify-packet read review inputs"),
        ), mock.patch.object(
            promoter.sys,
            "stdout",
            captured,
        ):
            self.assertEqual(
                promoter.main(
                    ["verify-packet", "--packet-root", str(self.packet)]
                ),
                0,
            )
        self.assertEqual(calls["count"], 2)
        self.assertEqual(self._tree_snapshot(), before)
        payload = captured.buffer.getvalue()
        parsed = json.loads(payload.decode("utf-8"))
        self.assertEqual(payload, promoter.canonical_json_bytes(parsed) + b"\n")
        self.assertEqual(parsed["schema_version"], promoter.PACKET_PREFLIGHT_SCHEMA)
        self.assertEqual(parsed["status"], promoter.PACKET_PREFLIGHT_STATUS)
        self.assertEqual(parsed["packet"]["file_count"], generator.FRAME_COUNT + 2)
        self.assertEqual(
            parsed["packet"]["support_frame_count"], generator.FRAME_COUNT
        )
        self.assertFalse(
            parsed["claim_limits"]["external_review_receipt_created"]
        )
        self.assertFalse(any(self.base.glob(".*.partial.*")))

    def test_verify_packet_rejects_extra_file_without_writes(self) -> None:
        extra = self.packet / "unexpected.bin"
        extra.write_bytes(b"not in the frozen inventory")
        before = self._tree_snapshot()
        with self._write_traps(), self.assertRaises(promoter.SupportPromotionHold):
            promoter.verify_packet_only(packet_root=self.packet)
        self.assertEqual(self._tree_snapshot(), before)
        self.assertFalse(any(self.base.glob(".*.partial.*")))

    def test_verify_submission_cli_replays_full_pass_lineage_without_writes(self) -> None:
        class BinaryCapture:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        draft_path, _ = self._completed_draft(0)
        sealed = self.base / "verify-pass-submission"
        self._seal_review(0, draft_path, sealed)
        captured = BinaryCapture()
        before = self._tree_snapshot()
        original = promoter._replay_review_submission
        calls = {"count": 0}

        def counted_replay(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        arguments = [
            "verify-submission",
            "--packet-root",
            str(self.packet),
            "--reviewer-slot",
            "1",
            "--submission-root",
            str(sealed),
        ]
        with self._write_traps(), mock.patch.object(
            promoter,
            "_replay_review_submission",
            side_effect=counted_replay,
        ), mock.patch.object(
            promoter,
            "_review_inputs",
            side_effect=AssertionError("verify-submission entered pair validation"),
        ), mock.patch.object(promoter.sys, "stdout", captured):
            self.assertEqual(promoter.main(arguments), 0)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(self._tree_snapshot(), before)
        payload = captured.buffer.getvalue()
        parsed = json.loads(payload.decode("utf-8"))
        self.assertEqual(payload, promoter.canonical_json_bytes(parsed) + b"\n")
        self.assertEqual(parsed["root"], str(sealed))
        self.assertEqual(parsed["file_count"], 6)
        self.assertEqual(
            parsed["submission"]["status"],
            promoter.REVIEW_SUBMISSION_PASS_STATUS,
        )

        copied_draft = sealed / "completed_draft.input.json"
        copied_draft.chmod(0o600)
        copied_draft.write_bytes(copied_draft.read_bytes() + b" ")
        copied_draft.chmod(0o400)
        tampered = self._tree_snapshot()
        with self._write_traps(), self.assertRaises(promoter.SupportPromotionHold):
            promoter.verify_submission_only(
                packet_root=self.packet,
                reviewer_slot=1,
                submission_root=sealed,
            )
        self.assertEqual(self._tree_snapshot(), tampered)

    def test_verify_submission_preserves_and_reports_fail_submission(self) -> None:
        draft_path, draft = self._completed_draft(0)
        draft["frames"][9]["halo_and_adjacent_ground_coverage"] = "FAIL"
        draft["frames"][9]["decision"] = "FAIL"
        draft["frames"][9]["notes"] = "adjacent-ground apron is incomplete"
        draft["overall_decision"] = "FAIL"
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        sealed = self.base / "verify-fail-submission"
        self._seal_review(0, draft_path, sealed)
        before = self._tree_snapshot()
        with self._write_traps():
            result = promoter.verify_submission_only(
                packet_root=self.packet,
                reviewer_slot=1,
                submission_root=sealed,
            )
        self.assertEqual(self._tree_snapshot(), before)
        self.assertEqual(
            result["submission"]["status"],
            promoter.REVIEW_SUBMISSION_FAIL_STATUS,
        )
        self.assertEqual(result["submission"]["overall_decision"], "FAIL")
        self.assertFalse(
            result["submission"]["promotion_eligible_by_this_ballot_alone"]
        )

    def test_verify_submission_rechecks_generator_before_return(self) -> None:
        draft_path, _ = self._completed_draft(0)
        sealed = self.base / "verify-generator-drift-submission"
        self._seal_review(0, draft_path, sealed)
        calls = {"count": 0}

        def reload_generator():
            calls["count"] += 1
            if calls["count"] == 3:
                raise promoter.SupportPromotionHold(
                    "synthetic generator drift before verification return"
                )
            return generator

        with self._write_traps(), mock.patch.object(
            promoter,
            "_load_generator",
            side_effect=reload_generator,
        ), self.assertRaises(promoter.SupportPromotionHold):
            promoter.verify_submission_only(
                packet_root=self.packet,
                reviewer_slot=1,
                submission_root=sealed,
            )
        self.assertEqual(calls["count"], 3)

    def test_validate_only_rejects_distinct_files_with_same_evidence_bytes(self) -> None:
        self.evidence_paths[1].write_bytes(self.evidence_paths[0].read_bytes())
        self._resign(1)
        self._assert_preflight_hold()
        self._assert_hold()

    def test_validate_only_rejects_drift_between_complete_replays(self) -> None:
        original = promoter.replay_packet
        calls = {"count": 0}

        def drifting_replay(root):
            calls["count"] += 1
            packet = original(root)
            if calls["count"] == 2:
                packet = dict(packet)
                packet["file_count"] += 1
            return packet

        with mock.patch.object(promoter, "replay_packet", side_effect=drifting_replay):
            self._assert_preflight_hold()
        self.assertEqual(calls["count"], 2)

    def test_validate_only_rechecks_generator_before_return(self) -> None:
        calls = {"count": 0}

        def reload_generator():
            calls["count"] += 1
            if calls["count"] == 3:
                raise promoter.SupportPromotionHold(
                    "synthetic generator drift before preflight return"
                )
            return generator

        with mock.patch.object(
            promoter,
            "_load_generator",
            side_effect=reload_generator,
        ):
            self._assert_preflight_hold()
        self.assertEqual(calls["count"], 3)

    def test_promote_reloads_generator_immediately_before_publication(self) -> None:
        calls = {"count": 0}

        def reload_generator():
            calls["count"] += 1
            if calls["count"] == 2:
                raise promoter.SupportPromotionHold(
                    "synthetic frozen generator drift before publication"
                )
            return generator

        with mock.patch.object(
            promoter,
            "_load_generator",
            side_effect=reload_generator,
        ):
            self._assert_hold()
        self.assertEqual(calls["count"], 2)
        self.assertTrue(any(self.base.glob(".promoted.partial.*")))

    def test_promote_reloads_generator_after_publication(self) -> None:
        calls = {"count": 0}

        def reload_generator():
            calls["count"] += 1
            if calls["count"] == 3:
                raise promoter.SupportPromotionHold(
                    "synthetic frozen generator drift after publication"
                )
            return generator

        with mock.patch.object(
            promoter,
            "_load_generator",
            side_effect=reload_generator,
        ), self.assertRaisesRegex(
            promoter.SupportPromotionHold,
            "treat bundle as quarantined",
        ):
            self._promote()
        self.assertEqual(calls["count"], 3)
        self.assertTrue(self.output.is_dir())
        self.assertFalse(any(self.base.glob(".promoted.partial.*")))

    def test_promote_pins_running_program_from_entry_to_publication(self) -> None:
        initial = promoter._program_identity()
        changed = dict(initial)
        changed["sha256"] = "a" * 64
        with mock.patch.object(
            promoter,
            "_program_identity",
            side_effect=(initial, changed),
        ) as identity:
            self._assert_hold()
        self.assertEqual(identity.call_count, 2)
        self.assertTrue(any(self.base.glob(".promoted.partial.*")))

    def test_validate_only_rejects_failed_frame_without_writes(self) -> None:
        frame = self.receipts[1]["frames"][8]
        frame["bone_coverage"] = "FAIL"
        frame["decision"] = "FAIL"
        self._resign(1)
        self._assert_preflight_hold()

    def test_refuses_existing_output_without_mutation(self) -> None:
        self.output.mkdir()
        sentinel = self.output / "sentinel"
        sentinel.write_bytes(b"untouched")
        with self.assertRaises(promoter.SupportPromotionHold):
            self._promote()
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_refuses_output_inside_immutable_packet_before_staging(self) -> None:
        inside = self.packet / "promotion"
        self._assert_hold(output_root=inside)
        self.assertFalse(os.path.lexists(inside))
        self.assertFalse(any(self.packet.glob(".promotion.partial.*")))

    def test_requires_exactly_two_reviewers(self) -> None:
        self._assert_hold(
            reviewer_receipts=(self.receipt_paths[0],),
            reviewer_evidence=(self.evidence_paths[0],),
        )

    def test_rejects_pending_frame(self) -> None:
        frame = self.receipts[0]["frames"][7]
        frame["contact_shadow_coverage"] = "PENDING"
        frame["decision"] = "PENDING"
        self._resign(0)
        self._assert_hold()

    def test_rejects_failed_frame(self) -> None:
        frame = self.receipts[1]["frames"][8]
        frame["bone_coverage"] = "FAIL"
        frame["decision"] = "FAIL"
        self._resign(1)
        self._assert_hold()

    def test_rejects_boundary_edit(self) -> None:
        self.receipts[0]["frames"][9]["boundary_edit_requested"] = True
        self._resign(0)
        self._assert_hold()

    def test_rejects_empty_frame_note(self) -> None:
        self.receipts[0]["frames"][10]["notes"] = "   "
        self._resign(0)
        self._assert_hold()

    def test_rejects_casefold_duplicate_identity(self) -> None:
        self.receipts[1]["reviewer_identity"] = self.receipts[0][
            "reviewer_identity"
        ].swapcase()
        self._resign(1)
        self._assert_hold()

    def test_rejects_bad_projection(self) -> None:
        self.receipts[0]["signature_or_external_receipt"][
            "review_projection_sha256"
        ] = "e" * 64
        write_json(self.receipt_paths[0], self.receipts[0])
        self._assert_hold()

    def test_rejects_changed_evidence(self) -> None:
        self.evidence_paths[0].write_bytes(b"changed after receipt")
        self._assert_hold()

    def test_rejects_noncanonical_receipt(self) -> None:
        self.receipt_paths[0].write_text(
            json.dumps(self.receipts[0], indent=2), encoding="utf-8"
        )
        self._assert_hold()

    def test_rejects_bool_reviewer_slot(self) -> None:
        self.receipts[0]["reviewer_slot"] = True
        self._resign(0)
        self._assert_hold()

    def test_rejects_extra_packet_file(self) -> None:
        (self.packet / "extra.bin").write_bytes(b"extra")
        self._assert_hold()

    def test_rejects_missing_packet_mask(self) -> None:
        (self.packet_masks / "00017.png").unlink()
        self._assert_hold()

    def test_rejects_tampered_packet_sha256sums(self) -> None:
        with (self.packet / "SHA256SUMS").open("ab") as handle:
            handle.write(("f" * 64 + "  extra.bin\n").encode("ascii"))
        self._assert_hold()

    def test_rejects_packet_symlink(self) -> None:
        original = self.packet_masks / "00018.png"
        moved = self.base / "moved-mask.png"
        original.rename(moved)
        original.symlink_to(moved)
        self._assert_hold()

    def test_rejects_review_hardlink(self) -> None:
        duplicate = self.inputs / "reviewer-1-hardlink.json"
        os.link(self.receipt_paths[0], duplicate)
        self._assert_hold(
            reviewer_receipts=(duplicate, self.receipt_paths[1]),
        )

    def test_special_review_input_is_rejected_before_open(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        special = self.inputs / "reviewer-special.fifo"
        os.mkfifo(special)
        with mock.patch.object(
            promoter.os,
            "open",
            side_effect=AssertionError("special file must be rejected before open"),
        ):
            with self.assertRaises(promoter.SupportPromotionHold):
                promoter._stable_file(
                    special,
                    "special review input",
                    require_nlink1=True,
                    maximum_bytes=1024,
                )

    def test_rejects_source_bytes_not_bound_to_packet(self) -> None:
        wrong = self.base / "wrong-source.mp4"
        wrong.write_bytes(b"wrong")
        self._assert_hold(source_path=wrong)

    def test_unavailable_publication_primitive_leaves_final_absent(self) -> None:
        with mock.patch.object(
            promoter,
            "_rename_noreplace",
            side_effect=promoter.SupportPromotionHold("unavailable"),
        ):
            self._assert_hold()
        self.assertTrue(any(self.base.glob(".promoted.partial.*")))

    def test_committed_publication_fault_reports_final_quarantine(self) -> None:
        original = promoter._rename_noreplace

        def commit_then_fail(source, destination, state):
            original(source, destination, state)
            raise OSError("synthetic post-commit fault")

        with mock.patch.object(
            promoter,
            "_rename_noreplace",
            side_effect=commit_then_fail,
        ):
            with self.assertRaisesRegex(
                promoter.SupportPromotionHold,
                "treat bundle as quarantined",
            ):
                self._promote()
        self.assertTrue(self.output.is_dir())
        self.assertFalse(any(self.base.glob(".promoted.partial.*")))

    def test_postpublication_hold_reports_final_quarantine(self) -> None:
        with mock.patch.object(
            promoter,
            "_replay_bundle",
            side_effect=promoter.SupportPromotionHold("synthetic late HOLD"),
        ):
            with self.assertRaisesRegex(
                promoter.SupportPromotionHold,
                "treat bundle as quarantined",
            ):
                self._promote()
        self.assertTrue(self.output.is_dir())
        self.assertFalse(any(self.base.glob(".promoted.partial.*")))

    def test_prepublication_seal_rejects_checksum_bytes_drift(self) -> None:
        self._promote()
        expected_records = promoter._bundle_records(
            self.output,
            exclude_sums=True,
        )
        sums = self.output / "SHA256SUMS"
        expected_sums = sums.read_bytes()
        formal_payload = (self.output / "support_review_receipt.json").read_bytes()
        promotion_payload = (self.output / "promotion_receipt.json").read_bytes()
        sums.chmod(0o600)
        sums.write_bytes(b"corrupt\n")
        sums.chmod(0o400)
        with self.assertRaises(promoter.SupportPromotionHold):
            promoter._verify_prepublication_seal(
                self.output,
                expected_records,
                expected_sums,
                formal_payload,
                promotion_payload,
            )

    def test_final_bundle_replay_rejects_extra_empty_directory(self) -> None:
        self._promote()
        extra = self.output / "extra-empty"
        extra.mkdir(mode=0o700)
        with self.assertRaises(promoter.SupportPromotionHold):
            promoter._replay_bundle(
                self.output,
                generator,
                self.source_row,
                self.sam2_row,
            )

    def test_final_bundle_replay_rejects_public_root_mode(self) -> None:
        self._promote()
        self.output.chmod(0o755)
        with self.assertRaises(promoter.SupportPromotionHold):
            promoter._replay_bundle(
                self.output,
                generator,
                self.source_row,
                self.sam2_row,
            )

    def test_final_bundle_replay_binds_program_identity(self) -> None:
        self._promote()
        promotion = json.loads(
            (self.output / "promotion_receipt.json").read_text(encoding="utf-8")
        )
        promotion["program"]["sha256"] = "f" * 64
        unsigned = dict(promotion)
        unsigned.pop("promotion_digest")
        promotion["promotion_digest"] = promoter.object_sha256(unsigned)
        self._rewrite_promotion_and_checksums(promotion)
        with self.assertRaises(promoter.SupportPromotionHold):
            promoter._replay_bundle(
                self.output,
                generator,
                self.source_row,
                self.sam2_row,
            )

    def test_final_bundle_replay_binds_generator_identity(self) -> None:
        self._promote()
        promotion = json.loads(
            (self.output / "promotion_receipt.json").read_text(encoding="utf-8")
        )
        promotion["generator"]["size"] += 1
        unsigned = dict(promotion)
        unsigned.pop("promotion_digest")
        promotion["promotion_digest"] = promoter.object_sha256(unsigned)
        self._rewrite_promotion_and_checksums(promotion)
        with self.assertRaises(promoter.SupportPromotionHold):
            promoter._replay_bundle(
                self.output,
                generator,
                self.source_row,
                self.sam2_row,
            )

    def test_source_has_no_ordinary_rename_or_replace_fallback(self) -> None:
        source = PROMOTER_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("os.replace", source)
        self.assertNotIn("os.rename(", source)
        self.assertIn("renameat2", source)
        self.assertIn("renameatx_np", source)
        self.assertNotIn('getattr(library, "renamex_np"', source)


class FrozenGeneratorSourceLoadingTests(unittest.TestCase):
    def test_stale_timestamp_pyc_cannot_override_hashed_source_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(CANONICAL_TMP)) as directory:
            source = Path(directory) / "frozen_generator.py"
            cached = b"VALUE = 'cached'\n"
            current = b"VALUE = 'source'\n"
            self.assertEqual(len(cached), len(current))
            source.write_bytes(cached)
            timestamp = 1_700_000_000
            os.utime(source, (timestamp, timestamp))
            py_compile.compile(str(source), doraise=True)
            source.write_bytes(current)
            os.utime(source, (timestamp, timestamp))
            with mock.patch.object(promoter, "GENERATOR_PATH", source), mock.patch.object(
                promoter,
                "GENERATOR_SHA256",
                hashlib.sha256(current).hexdigest(),
            ), mock.patch.object(promoter, "GENERATOR_SIZE", len(current)):
                loaded = promoter._load_generator()
            self.assertEqual(loaded.VALUE, "source")


if __name__ == "__main__":
    unittest.main()

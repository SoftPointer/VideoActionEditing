#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = METHOD_ROOT / "action_edit_sft_manifest_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "action_edit_sft_manifest_v2_test_subject", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
manifest_v2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_v2
SPEC.loader.exec_module(manifest_v2)


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ActionEditSFTManifestV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def _file(self, relative: str, payload: bytes) -> dict:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": path.as_posix(),
            "sha256": bytes_sha256(payload),
            "size_bytes": len(payload),
        }

    def _row(
        self,
        index: int,
        *,
        source_index: Optional[int] = None,
        actor_scene_group_id: Optional[str] = None,
        semantic_tag: Optional[str] = None,
        instruction_text: Optional[str] = None,
        generation_seed: Optional[int] = None,
        copy_of: bool = False,
        transcode: bool = False,
    ) -> dict:
        source_number = index if source_index is None else source_index
        semantic = semantic_tag or "semantic-{}".format(index)
        source_payload = "source-{}".format(source_number).encode("utf-8")
        target_payload = "target-{}-{}".format(index, semantic).encode("utf-8")
        source = {
            **self._file("source/{}.mp4".format(source_number), source_payload),
            "source_id": "source-{}".format(source_number),
            "canonical_source_id": "canonical-source-{}".format(source_number),
            "actor_ids": ["actor-{}".format(source_number)],
            "scene_id": "scene-{}".format(source_number),
            "camera_class": "static",
            "initial_state": "initial-state-{}".format(source_number),
        }
        text = instruction_text or "Perform {} now".format(semantic)
        text_bytes = text.encode("utf-8")
        instruction = {
            "text": text,
            "sha256": bytes_sha256(text_bytes),
            "size_bytes": len(text_bytes),
            "encoding": "utf-8",
            "semantic_id": "0" * 64,
            "template_family": "imperative-v1",
            "actor": "primary-actor",
            "action": semantic,
            "object": "object-1",
            "direction": "forward",
            "speed": "normal",
            "amplitude": "full",
            "onset": "early",
            "outcome": "completed",
            "terminal_state": "completed",
            "preserve": ["background", "camera", "identity"],
        }
        instruction["semantic_id"] = manifest_v2.expected_instruction_semantic_id(
            instruction
        )
        target = {
            **self._file("target/{}.mp4".format(index), target_payload),
            "provenance": "teacher-pseudo",
            "semantic_truth_class": "teacher-pseudo",
            "teacher_id": "teacher-v1",
            "qualification_status": "pending",
            "qualification_receipt": None,
            "human_review": None,
            "human_review_receipt_sha256": None,
            "action_feature_encoder_sha256": None,
            "q_y_sha256": None,
            "compatibility_receipt_sha256": None,
        }
        row = {
            "schema_version": manifest_v2.ROW_SCHEMA,
            "row_id": "0" * 64,
            "semantic_edit_id": "0" * 64,
            "action_family": "family-main",
            "upstream_group_id": "upstream-{}".format(source_number),
            "actor_scene_group_id": actor_scene_group_id
            or "actor-scene-{}".format(source_number),
            "source": source,
            "instruction": instruction,
            "target": target,
            "action_anchors": [],
            "annotations": {"fixture_id": "row-{}".format(index)},
            "row_tier": "train",
            "training_subset": "action_motion",
            "calibration_kind": None,
            "evaluation_stratum": None,
            "generation_seed": generation_seed,
            "copy_of_row_id": None,
            "transcode_of_sha256": None,
        }
        row["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(row)
        row["row_id"] = manifest_v2.expected_row_id(row)
        if copy_of:
            row["copy_of_row_id"] = row["row_id"]
        if transcode:
            row["transcode_of_sha256"] = bytes_sha256(b"pre-transcode")
        return row

    def _smoke(self, rows, minimum=None, verify_files=True):
        if minimum is None:
            minimum = len(rows)
        authority = self._equivalence_authority(rows)
        self.equivalence_authority = authority
        self.equivalence_authority_digest = authority["authority_digest"]
        return manifest_v2.build_train_manifest(
            rows,
            build_mode=manifest_v2.BUILD_MODE_ENGINEERING_SMOKE,
            equivalence_authority=authority,
            expected_equivalence_authority_digest=authority["authority_digest"],
            engineering_smoke_minimum_count=minimum,
            verify_files=verify_files,
        )

    def _equivalence_authority(self, rows):
        source_groups = {}
        instruction_groups = {}
        for row in rows:
            canonical_id = row["source"]["canonical_source_id"]
            source = source_groups.setdefault(
                canonical_id,
                {
                    "canonical_source_id": canonical_id,
                    "source_ids": set(),
                    "upstream_group_id": row["upstream_group_id"],
                    "actor_scene_group_id": row["actor_scene_group_id"],
                    "file_sha256s": set(),
                },
            )
            source["source_ids"].add(row["source"]["source_id"])
            source["file_sha256s"].add(row["source"]["sha256"])
            semantic_id = row["instruction"]["semantic_id"]
            instruction = instruction_groups.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "identity": manifest_v2.instruction_semantic_identity(
                        row["instruction"]
                    ),
                    "text_sha256s": set(),
                },
            )
            instruction["text_sha256s"].add(row["instruction"]["sha256"])
        sources = []
        for key in sorted(source_groups):
            item = source_groups[key]
            sources.append(
                {
                    **item,
                    "source_ids": sorted(item["source_ids"]),
                    "file_sha256s": sorted(item["file_sha256s"]),
                }
            )
        instructions = []
        for key in sorted(instruction_groups):
            item = instruction_groups[key]
            instructions.append(
                {
                    **item,
                    "text_sha256s": sorted(item["text_sha256s"]),
                }
            )
        unsigned = {
            "schema_version": manifest_v2.EQUIVALENCE_AUTHORITY_SCHEMA,
            "exact_member_closure": True,
            "sources": sources,
            "instructions": instructions,
        }
        return {
            **unsigned,
            "authority_digest": manifest_v2.object_sha256(unsigned),
        }

    def _validate_manifest(self, manifest, verify_files=True):
        return manifest_v2.validate_train_manifest(
            manifest,
            equivalence_authority=self.equivalence_authority,
            expected_equivalence_authority_digest=self.equivalence_authority_digest,
            expected_manifest_digest=manifest["manifest_digest"],
            verify_files=verify_files,
        )

    def _qualification_authority(self, rows):
        receipts = sorted(
            {
                row["target"]["qualification_receipt"]["sha256"]
                for row in rows
            }
        )
        unsigned = {
            "schema_version": manifest_v2.QUALIFICATION_AUTHORITY_SCHEMA,
            "exact_member_closure": True,
            "qualification_receipt_sha256s": receipts,
        }
        return {
            **unsigned,
            "authority_digest": manifest_v2.object_sha256(unsigned),
        }

    def _build_sampler(self, manifest):
        return manifest_v2.build_exact_sampler(
            manifest,
            equivalence_authority=self.equivalence_authority,
            expected_equivalence_authority_digest=self.equivalence_authority_digest,
            expected_manifest_digest=manifest["manifest_digest"],
        )

    def _validate_sampler(self, manifest, sampler):
        return manifest_v2.validate_exact_sampler(
            manifest,
            sampler,
            equivalence_authority=self.equivalence_authority,
            expected_equivalence_authority_digest=self.equivalence_authority_digest,
            expected_manifest_digest=manifest["manifest_digest"],
            expected_sampler_digest=sampler["sampler_digest"],
        )

    def _accept(self, row):
        accepted = copy.deepcopy(row)
        authorities = {
            "qualification_status": "accepted",
            "human_review": "accepted",
            "human_review_receipt_sha256": bytes_sha256(
                b"fixture-human-review-authority-v2-20260817"
            ),
            "action_feature_encoder_sha256": bytes_sha256(
                b"fixture-action-feature-encoder-v2-20260817"
            ),
            "q_y_sha256": bytes_sha256(
                b"fixture-qualified-q-y-authority-v2-20260817"
            ),
            "compatibility_receipt_sha256": bytes_sha256(
                b"fixture-compatibility-authority-v2-20260817"
            ),
        }
        accepted["target"].update(authorities)
        receipt_unsigned = {
            "schema_version": manifest_v2.QUALIFICATION_RECEIPT_SCHEMA,
            "qualification_status": "accepted",
            "row_id": accepted["row_id"],
            "semantic_edit_id": accepted["semantic_edit_id"],
            "canonical_source_id": accepted["source"]["canonical_source_id"],
            "source_sha256": accepted["source"]["sha256"],
            "instruction_sha256": accepted["instruction"]["sha256"],
            "instruction_semantic_id": accepted["instruction"]["semantic_id"],
            "instruction_identity": manifest_v2.instruction_semantic_identity(
                accepted["instruction"]
            ),
            "target_sha256": accepted["target"]["sha256"],
            "action_family": accepted["action_family"],
            "training_subset": accepted["training_subset"],
            "target_provenance": accepted["target"]["provenance"],
            "target_semantic_truth_class": accepted["target"][
                "semantic_truth_class"
            ],
            "target_teacher_id": accepted["target"]["teacher_id"],
            "human_review": "accepted",
            "human_review_receipt_sha256": authorities[
                "human_review_receipt_sha256"
            ],
            "action_feature_encoder_sha256": authorities[
                "action_feature_encoder_sha256"
            ],
            "q_y_sha256": authorities["q_y_sha256"],
            "compatibility_receipt_sha256": authorities[
                "compatibility_receipt_sha256"
            ],
        }
        receipt = {
            **receipt_unsigned,
            "receipt_digest": manifest_v2.object_sha256(receipt_unsigned),
        }
        receipt_bytes = manifest_v2.canonical_json_bytes(receipt)
        accepted["target"]["qualification_receipt"] = self._file(
            "qualification/{}.json".format(accepted["row_id"]), receipt_bytes
        )
        return accepted

    @staticmethod
    def _reidentify(row):
        row["instruction"][
            "semantic_id"
        ] = manifest_v2.expected_instruction_semantic_id(row["instruction"])
        row["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(row)
        row["row_id"] = manifest_v2.expected_row_id(row)
        return row

    def _anchor(self, index=0):
        return {
            **self._file(
                "anchors/{}.mp4".format(index),
                "action-reference-{}".format(index).encode("utf-8"),
            ),
            "generation_seed": 9000 + index,
            "role": "action-reference-only",
            "q_anchor_sha256": bytes_sha256(
                "fixture-q-anchor-authority-{}".format(index).encode("utf-8")
            ),
            "compatibility_score": 0.875,
            "compatibility_verdict": "accept",
            "training_use": "point-distill",
            "compatibility_receipt_sha256": bytes_sha256(
                "fixture-anchor-compatibility-receipt-{}".format(index).encode(
                    "utf-8"
                )
            ),
        }

    @staticmethod
    def _reseal_manifest(manifest):
        unsigned = dict(manifest)
        unsigned.pop("manifest_digest", None)
        manifest["manifest_digest"] = manifest_v2.object_sha256(unsigned)

    @staticmethod
    def _reseal_sampler(sampler):
        sampler["members_digest"] = manifest_v2.object_sha256(sampler["members"])
        unsigned = dict(sampler)
        unsigned.pop("sampler_digest", None)
        sampler["sampler_digest"] = manifest_v2.object_sha256(unsigned)

    def test_valid_smoke_binds_three_envelopes_and_exact_sampler(self):
        rows = [self._row(index) for index in range(3)]
        manifest = self._smoke(rows, minimum=3)
        self.assertEqual(manifest["engineering_effective_N"], 3)
        self.assertEqual(manifest["D2_train_eligible_effective_N"], 0)
        self.assertEqual(manifest["row_tier"], "train")
        self.assertTrue(manifest["exact_member_closure"])
        self.assertTrue(manifest["engineering_smoke_only"])
        self.assertFalse(manifest["d0_claimed"])
        self.assertFalse(manifest["d2_claimed"])
        self.assertFalse(manifest["formal_training_authorized"])
        self.assertFalse(manifest["raw_accounting_authoritative"])
        self.assertEqual(
            manifest["claim_scope"], "engineering-smoke-only-not-d0-or-d2"
        )
        self.assertEqual(
            manifest["qualification_scope"],
            "pending-or-unqualified-engineering-only",
        )
        self.assertEqual(
            self._validate_manifest(manifest), manifest
        )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "manifest differs from pinned digest",
        ):
            manifest_v2.validate_train_manifest(
                manifest,
                equivalence_authority=self.equivalence_authority,
                expected_equivalence_authority_digest=self.equivalence_authority_digest,
                expected_manifest_digest=bytes_sha256(b"different-manifest"),
            )

        sampler = self._build_sampler(manifest)
        self.assertEqual(sampler["member_count"], 3)
        self.assertEqual(
            [member["ordinal"] for member in sampler["members"]], [0, 1, 2]
        )
        self.assertEqual(
            self._validate_sampler(manifest, sampler), sampler
        )

    def test_source_target_and_instruction_byte_envelopes_are_replayed(self):
        valid = self._row(0)
        for mutation, message in (
            (lambda row: row["source"].__setitem__("sha256", "f" * 64), "source SHA-256 differs"),
            (lambda row: row["target"].__setitem__("size_bytes", 999), "target size differs"),
            (lambda row: row["instruction"].__setitem__("sha256", "e" * 64), "instruction UTF-8 SHA-256 differs"),
            (lambda row: row["instruction"].__setitem__("size_bytes", 999), "instruction UTF-8 size differs"),
        ):
            hostile = copy.deepcopy(valid)
            mutation(hostile)
            with self.assertRaisesRegex(
                manifest_v2.ActionEditSFTManifestError, message
            ):
                manifest_v2.validate_train_row(hostile)

    def test_symlink_source_is_not_a_plain_file_envelope(self):
        row = self._row(0)
        link = self.root / "source-link.mp4"
        os.symlink(Path(row["source"]["path"]), link)
        row["source"]["path"] = link.as_posix()
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "canonical path|symlink"
        ):
            manifest_v2.validate_train_row(row)

    def test_intermediate_symlink_component_is_rejected(self):
        row = self._row(0)
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        real_file = real_parent / "source.mp4"
        real_file.write_bytes(b"intermediate-symlink-source")
        alias_parent = self.root / "alias-parent"
        os.symlink(real_parent, alias_parent)
        row["source"].update(
            {
                "path": (alias_parent / "source.mp4").as_posix(),
                "sha256": bytes_sha256(b"intermediate-symlink-source"),
                "size_bytes": len(b"intermediate-symlink-source"),
            }
        )
        row["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(row)
        row["row_id"] = manifest_v2.expected_row_id(row)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "canonical path|symlink"
        ):
            manifest_v2.validate_train_row(row)

    def test_action_anchor_is_closed_reference_only_and_never_target(self):
        row = self._row(0)
        row["action_anchors"] = [self._anchor()]
        normalized = manifest_v2.validate_train_row(row)
        self.assertEqual(
            normalized["action_anchors"][0]["role"], "action-reference-only"
        )

        for mutate, message in (
            (
                lambda anchor: anchor.__setitem__("role", "supervised-target"),
                "role must be action-reference-only",
            ),
            (
                lambda anchor: anchor.__setitem__("unexpected", "field"),
                "field closure differs",
            ),
        ):
            hostile = copy.deepcopy(row)
            mutate(hostile["action_anchors"][0])
            with self.assertRaisesRegex(
                manifest_v2.ActionEditSFTManifestError, message
            ):
                manifest_v2.validate_train_row(hostile)

        target_alias = copy.deepcopy(row)
        target_alias["action_anchors"][0].update(
            {
                key: target_alias["target"][key]
                for key in ("path", "sha256", "size_bytes")
            }
        )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "must not be the supervised target",
        ):
            manifest_v2.validate_train_row(target_alias)

    def test_every_nontrain_tier_is_forbidden_even_when_named_explicitly(self):
        for tier in manifest_v2.NONTRAIN_TIERS:
            hostile = self._row(0)
            hostile["row_tier"] = tier
            hostile["training_subset"] = None
            if tier == "calibration":
                hostile["calibration_kind"] = "action_feature_tuning"
            else:
                hostile["evaluation_stratum"] = "seen_action_unseen_source"
            with self.assertRaisesRegex(
                manifest_v2.ActionEditSFTManifestError,
                "forbidden from the train manifest and sampler",
            ):
                manifest_v2.validate_train_row(hostile)

    def test_train_tier_pairing_is_fail_closed(self):
        hostile = self._row(0)
        hostile["calibration_kind"] = "action_feature_tuning"
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "must not set calibration_kind",
        ):
            manifest_v2.validate_train_row(hostile)
        hostile = self._row(1)
        hostile["evaluation_stratum"] = "unseen_scene_camera"
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "must not set evaluation_stratum",
        ):
            manifest_v2.validate_train_row(hostile)

    def test_seed_copy_transcode_and_paraphrase_do_not_inflate_effective_n(self):
        original = self._row(0, semantic_tag="jump", generation_seed=7)
        paraphrase = copy.deepcopy(original)
        paraphrase_text = "Please perform jump immediately"
        paraphrase["instruction"]["text"] = paraphrase_text
        paraphrase["instruction"]["sha256"] = bytes_sha256(
            paraphrase_text.encode("utf-8")
        )
        paraphrase["instruction"]["size_bytes"] = len(
            paraphrase_text.encode("utf-8")
        )
        paraphrase["instruction"]["template_family"] = "polite-v2"
        paraphrase["generation_seed"] = 11
        # Structured semantics, semantic_edit_id, and row_id remain identical.

        copied = copy.deepcopy(original)
        copied["copy_of_row_id"] = copied["row_id"]
        copied["generation_seed"] = 13
        transcoded = copy.deepcopy(original)
        transcoded_source = self._file(
            "source/transcoded-source-0.mp4", b"source-0-transcoded-container"
        )
        transcoded["source"].update(transcoded_source)
        transcoded["transcode_of_sha256"] = bytes_sha256(b"original-container")
        transcoded["generation_seed"] = 17
        transcoded["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(
            transcoded
        )
        transcoded["row_id"] = manifest_v2.expected_row_id(transcoded)

        manifest = self._smoke(
            [transcoded, copied, paraphrase, original], minimum=1
        )
        self.assertEqual(manifest["raw_row_count"], 4)
        self.assertEqual(manifest["engineering_effective_N"], 1)
        self.assertEqual(manifest["D2_train_eligible_effective_N"], 0)
        self.assertEqual(
            sum(manifest["deduplication"].values()), 3
        )
        self.assertIsNone(manifest["rows"][0]["copy_of_row_id"])
        self.assertIsNone(manifest["rows"][0]["transcode_of_sha256"])

    def test_derivative_only_pool_cannot_satisfy_even_one_row_smoke(self):
        copied = self._row(0, copy_of=True)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "below the explicit minimum"
        ):
            self._smoke([copied], minimum=1)

    def test_endpoint_bytes_cannot_declare_conflicting_classification(self):
        original = self._row(0)
        hostile = copy.deepcopy(original)
        hostile["action_family"] = "different-family"
        hostile["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(hostile)
        hostile["row_id"] = manifest_v2.expected_row_id(hostile)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "conflicting classifications",
        ):
            self._smoke([original, hostile], minimum=1)

    def test_source_sha_aliases_cannot_evade_source_or_group_caps(self):
        original = self._row(0, semantic_tag="jump")
        alias = self._row(1, semantic_tag="turn")
        alias["source"] = copy.deepcopy(original["source"])
        alias["source"]["source_id"] = "fabricated-source-alias"
        alias["upstream_group_id"] = "fabricated-upstream-alias"
        alias["actor_scene_group_id"] = "fabricated-actor-scene-alias"
        alias["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(alias)
        alias["row_id"] = manifest_v2.expected_row_id(alias)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "not exactly bound by frozen equivalence authority",
        ):
            self._smoke([original, alias], minimum=1)

        reused_id = self._row(2, semantic_tag="land")
        reused_id["source"]["source_id"] = original["source"]["source_id"]
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "disjoint exact partition|multiple canonical source identities",
        ):
            self._smoke([original, reused_id], minimum=1)

    def test_pinned_equivalence_authority_rejects_unmarked_transcode_alias(self):
        original = self._row(0, semantic_tag="jump")
        alias = copy.deepcopy(original)
        alias["source"].update(
            self._file("source/unmarked-transcode.mp4", b"source-0-other-container")
        )
        alias["source"]["source_id"] = "source-0-alias"
        alias["source"]["canonical_source_id"] = "canonical-source-0-alias"
        alias["target"].update(
            self._file("target/unmarked-transcode-target.mp4", b"other-target-seed")
        )
        alias["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(alias)
        alias["row_id"] = manifest_v2.expected_row_id(alias)
        authority = self._equivalence_authority([original])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "absent from frozen equivalence authority",
        ):
            manifest_v2.build_train_manifest(
                [original, alias],
                build_mode=manifest_v2.BUILD_MODE_ENGINEERING_SMOKE,
                equivalence_authority=authority,
                expected_equivalence_authority_digest=authority["authority_digest"],
                engineering_smoke_minimum_count=1,
            )

    def test_pinned_equivalence_authority_rejects_synonym_paraphrase_alias(self):
        original = self._row(0, semantic_tag="jump")
        alias = copy.deepcopy(original)
        text = "Please leap now"
        alias["instruction"]["text"] = text
        alias["instruction"]["sha256"] = bytes_sha256(text.encode("utf-8"))
        alias["instruction"]["size_bytes"] = len(text.encode("utf-8"))
        alias["instruction"]["action"] = "leap"
        alias["instruction"][
            "semantic_id"
        ] = manifest_v2.expected_instruction_semantic_id(alias["instruction"])
        alias["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(alias)
        alias["row_id"] = manifest_v2.expected_row_id(alias)
        authority = self._equivalence_authority([original])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "semantics are absent from frozen equivalence authority",
        ):
            manifest_v2.build_train_manifest(
                [original, alias],
                build_mode=manifest_v2.BUILD_MODE_ENGINEERING_SMOKE,
                equivalence_authority=authority,
                expected_equivalence_authority_digest=authority["authority_digest"],
                engineering_smoke_minimum_count=1,
            )

    def test_source_cap_is_applied_after_semantic_deduplication(self):
        rows = [
            self._row(
                index,
                source_index=0,
                actor_scene_group_id="actor-scene-shared",
                semantic_tag="source-edit-{}".format(index),
            )
            for index in range(10)
        ]
        manifest = self._smoke(rows, minimum=8)
        self.assertEqual(manifest["engineering_effective_N"], 8)
        self.assertEqual(manifest["D2_train_eligible_effective_N"], 0)
        self.assertEqual(
            manifest["deduplication"]["source_cap_rows_dropped"], 2
        )

    def test_actor_scene_cap_is_applied_across_distinct_sources(self):
        rows = [
            self._row(
                index,
                actor_scene_group_id="actor-scene-shared",
                semantic_tag="group-edit-{}".format(index),
            )
            for index in range(18)
        ]
        manifest = self._smoke(rows, minimum=16)
        self.assertEqual(manifest["engineering_effective_N"], 16)
        self.assertEqual(manifest["D2_train_eligible_effective_N"], 0)
        self.assertEqual(
            manifest["deduplication"]["actor_scene_cap_rows_dropped"], 2
        )

    def test_engineering_minimum_is_explicit_and_never_becomes_d0_or_d2(self):
        row = self._row(0)
        authority = self._equivalence_authority([row])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "explicit minimum count"
        ):
            manifest_v2.build_train_manifest(
                [row],
                build_mode=manifest_v2.BUILD_MODE_ENGINEERING_SMOKE,
                equivalence_authority=authority,
                expected_equivalence_authority_digest=authority["authority_digest"],
            )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "below the explicit minimum"
        ):
            self._smoke([row], minimum=2)
        artifact = self._smoke([row], minimum=1)
        self.assertLess(
            artifact["minimum_effective_count"], manifest_v2.D0_MINIMUM_COUNT
        )
        self.assertFalse(artifact["d0_claimed"])
        self.assertFalse(artifact["d2_claimed"])
        self.assertFalse(artifact["formal_training_authorized"])

    def test_formal_d2_floor_cannot_be_lowered(self):
        row = self._accept(self._row(0))
        authority = self._equivalence_authority([row])
        qualification_authority = self._qualification_authority([row])
        authority_kwargs = {
            "equivalence_authority": authority,
            "expected_equivalence_authority_digest": authority["authority_digest"],
            "qualification_authority": qualification_authority,
            "expected_qualification_authority_digest": qualification_authority[
                "authority_digest"
            ],
        }
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "must not accept"
        ):
            manifest_v2.build_train_manifest(
                [row],
                build_mode=manifest_v2.BUILD_MODE_FORMAL_D2,
                engineering_smoke_minimum_count=1,
                **authority_kwargs,
            )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "minimum=100000",
        ):
            manifest_v2.build_train_manifest(
                [row], build_mode=manifest_v2.BUILD_MODE_FORMAL_D2, **authority_kwargs
            )

        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "requires physical file verification",
        ):
            manifest_v2.build_train_manifest(
                [row],
                build_mode=manifest_v2.BUILD_MODE_FORMAL_D2,
                verify_files=False,
                **authority_kwargs,
            )

    def test_engineering_qualification_is_explicitly_unqualified(self):
        row = self._row(0)
        normalized = manifest_v2.validate_train_row(row)
        self.assertEqual(normalized["target"]["qualification_status"], "pending")
        self.assertIsNone(normalized["target"]["human_review"])
        self.assertIsNone(normalized["target"]["q_y_sha256"])
        self.assertIsNone(normalized["target"]["qualification_receipt"])

        accepted = self._accept(row)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "engineering smoke accepts only pending/unqualified",
        ):
            self._smoke([accepted], minimum=1)

    def test_engineering_row_cannot_enter_formal_candidate_sampler(self):
        pending = self._row(0)
        authority = self._equivalence_authority([pending])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "requires qualification_status=accepted",
        ):
            manifest_v2.build_train_manifest(
                [pending],
                build_mode=manifest_v2.BUILD_MODE_FORMAL_D2,
                equivalence_authority=authority,
                expected_equivalence_authority_digest=authority["authority_digest"],
            )

    def test_formal_qualification_receipt_is_exactly_endpoint_bound(self):
        accepted = self._accept(self._row(0))
        normalized = manifest_v2.validate_train_row(accepted)
        self.assertEqual(normalized["target"]["qualification_status"], "accepted")

        hostile = copy.deepcopy(accepted)
        hostile["target"]["q_y_sha256"] = bytes_sha256(b"different-q-y")
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "not exactly bound",
        ):
            manifest_v2.validate_train_row(hostile)

        placeholder = copy.deepcopy(accepted)
        placeholder["target"]["human_review_receipt_sha256"] = "a" * 64
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "placeholder"
        ):
            manifest_v2.validate_train_row(placeholder)

        conventional_placeholder = copy.deepcopy(accepted)
        conventional_placeholder["target"][
            "human_review_receipt_sha256"
        ] = bytes_sha256(b"human-review")
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "placeholder"
        ):
            manifest_v2.validate_train_row(conventional_placeholder)

    def test_one_accepted_receipt_cannot_survive_any_semantic_relabel(self):
        accepted = self._accept(self._row(0, semantic_tag="jump"))

        mutations = []
        semantic = copy.deepcopy(accepted)
        semantic["instruction"]["speed"] = "slow"
        self._reidentify(semantic)
        mutations.append(("semantic_id", semantic))

        for label, mutate in (
            (
                "action_family",
                lambda row: row.__setitem__("action_family", "family-relabelled"),
            ),
            (
                "training_subset",
                lambda row: row.__setitem__("training_subset", "long_horizon"),
            ),
            (
                "provenance",
                lambda row: row["target"].__setitem__("provenance", "simulator"),
            ),
            (
                "truth",
                lambda row: row["target"].__setitem__(
                    "semantic_truth_class", "simulator-gt"
                ),
            ),
            (
                "teacher",
                lambda row: row["target"].__setitem__(
                    "teacher_id", "teacher-relabelled"
                ),
            ),
        ):
            hostile = copy.deepcopy(accepted)
            mutate(hostile)
            mutations.append((label, hostile))

        for label, hostile in mutations:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    manifest_v2.ActionEditSFTManifestError,
                    "full row semantics, endpoints, and authorities",
                ):
                    manifest_v2.validate_train_row(hostile)

    def test_qualification_authority_is_external_exact_and_caller_pinned(self):
        accepted = self._accept(self._row(0))
        authority = self._qualification_authority([accepted])
        self.assertEqual(
            manifest_v2.validate_qualification_authority(
                authority, expected_authority_digest=authority["authority_digest"]
            ),
            authority,
        )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "differs from pinned digest",
        ):
            manifest_v2.validate_qualification_authority(
                authority,
                expected_authority_digest=bytes_sha256(
                    b"different-external-qualification-authority"
                ),
            )

        equivalence = self._equivalence_authority([accepted])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "requires pinned qualification authority",
        ):
            manifest_v2.build_train_manifest(
                [accepted],
                build_mode=manifest_v2.BUILD_MODE_FORMAL_D2,
                equivalence_authority=equivalence,
                expected_equivalence_authority_digest=equivalence[
                    "authority_digest"
                ],
            )

        wrong_unsigned = {
            "schema_version": manifest_v2.QUALIFICATION_AUTHORITY_SCHEMA,
            "exact_member_closure": True,
            "qualification_receipt_sha256s": sorted(
                authority["qualification_receipt_sha256s"]
                + [bytes_sha256(b"unrelated-qualified-receipt")]
            ),
        }
        wrong_authority = {
            **wrong_unsigned,
            "authority_digest": manifest_v2.object_sha256(wrong_unsigned),
        }
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "exact qualified receipt authority closure",
        ):
            manifest_v2.build_train_manifest(
                [accepted],
                build_mode=manifest_v2.BUILD_MODE_FORMAL_D2,
                equivalence_authority=equivalence,
                expected_equivalence_authority_digest=equivalence[
                    "authority_digest"
                ],
                qualification_authority=wrong_authority,
                expected_qualification_authority_digest=wrong_authority[
                    "authority_digest"
                ],
            )

    def test_target_bytes_cannot_count_for_two_semantic_rows(self):
        first = self._row(0, semantic_tag="jump")
        second = self._row(1, semantic_tag="turn")
        second["target"].update(
            {
                key: first["target"][key]
                for key in ("path", "sha256", "size_bytes")
            }
        )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "target byte identity is reused across distinct semantic rows",
        ):
            self._smoke([first, second], minimum=2)

    def test_source_equals_target_is_only_the_exact_noop_contract(self):
        action = self._row(0, semantic_tag="jump")
        action["target"].update(
            {
                key: action["source"][key]
                for key in ("path", "sha256", "size_bytes")
            }
        )
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "only for the explicit noop preservation contract",
        ):
            manifest_v2.validate_train_row(action)

        noop = self._row(1, semantic_tag="noop")
        noop["target"].update(
            {
                key: noop["source"][key]
                for key in ("path", "sha256", "size_bytes")
            }
        )
        noop["target"]["semantic_truth_class"] = "noop"
        noop["training_subset"] = "noop_preservation"
        noop["instruction"]["action"] = "noop"
        self._reidentify(noop)
        self.assertEqual(manifest_v2.validate_train_row(noop), noop)

    def test_physical_reverification_and_strict_count_types_are_mandatory(self):
        row = self._row(0)
        authority = self._equivalence_authority([row])
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "requires physical file verification",
        ):
            manifest_v2.build_train_manifest(
                [row],
                build_mode=manifest_v2.BUILD_MODE_ENGINEERING_SMOKE,
                equivalence_authority=authority,
                expected_equivalence_authority_digest=authority[
                    "authority_digest"
                ],
                engineering_smoke_minimum_count=1,
                verify_files=False,
            )

        manifest = self._smoke([row], minimum=1)
        for field, hostile_value in (
            ("engineering_effective_N", True),
            ("minimum_effective_count", 1.0),
            ("source_semantic_edit_cap", 8.0),
            ("actor_scene_row_cap", 16.0),
        ):
            hostile = copy.deepcopy(manifest)
            hostile[field] = hostile_value
            self._reseal_manifest(hostile)
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    manifest_v2.ActionEditSFTManifestError,
                    "integer",
                ):
                    self._validate_manifest(hostile)

        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "replay requires physical file verification",
        ):
            self._validate_manifest(manifest, verify_files=False)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "replay requires physical file verification",
        ):
            manifest_v2.build_exact_sampler(
                manifest,
                equivalence_authority=self.equivalence_authority,
                expected_equivalence_authority_digest=self.equivalence_authority_digest,
                expected_manifest_digest=manifest["manifest_digest"],
                verify_files=False,
            )

    def test_sampler_rejects_missing_extra_and_reordered_members_after_reseal(self):
        manifest = self._smoke([self._row(0), self._row(1)], minimum=2)
        sampler = self._build_sampler(manifest)

        missing = copy.deepcopy(sampler)
        missing["members"].pop()
        missing["member_count"] = 1
        self._reseal_sampler(missing)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "exact ordered"
        ):
            self._validate_sampler(manifest, missing)

        extra = copy.deepcopy(sampler)
        extra["members"].append(copy.deepcopy(extra["members"][0]))
        extra["members"][-1]["ordinal"] = 2
        extra["member_count"] = 3
        self._reseal_sampler(extra)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "exact ordered"
        ):
            self._validate_sampler(manifest, extra)

        reordered = copy.deepcopy(sampler)
        reordered["members"][0]["row_id"], reordered["members"][1]["row_id"] = (
            reordered["members"][1]["row_id"],
            reordered["members"][0]["row_id"],
        )
        self._reseal_sampler(reordered)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError, "exact ordered"
        ):
            self._validate_sampler(manifest, reordered)

        boolean_alias = copy.deepcopy(sampler)
        boolean_alias["member_count"] = True
        boolean_alias["members"][0]["ordinal"] = False
        self._reseal_sampler(boolean_alias)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "non-negative integer",
        ):
            self._validate_sampler(manifest, boolean_alias)

    def test_resealed_manifest_still_rejects_nontrain_member(self):
        manifest = self._smoke([self._row(0)], minimum=1)
        manifest["rows"][0]["row_tier"] = "locked_final"
        manifest["rows_digest"] = manifest_v2.object_sha256(
            [
                {
                    "row_id": manifest["rows"][0]["row_id"],
                    "row_sha256": manifest_v2.object_sha256(manifest["rows"][0]),
                }
            ]
        )
        self._reseal_manifest(manifest)
        with self.assertRaisesRegex(
            manifest_v2.ActionEditSFTManifestError,
            "forbidden from the train manifest and sampler",
        ):
            self._validate_manifest(manifest)

    def test_manifest_is_deterministic_under_raw_input_permutation(self):
        rows = [self._row(index) for index in range(4)]
        forward = self._smoke(rows, minimum=4)
        reverse = self._smoke(list(reversed(rows)), minimum=4)
        self.assertEqual(forward, reverse)
        self.assertEqual(
            self._build_sampler(forward),
            self._build_sampler(reverse),
        )


if __name__ == "__main__":
    unittest.main()

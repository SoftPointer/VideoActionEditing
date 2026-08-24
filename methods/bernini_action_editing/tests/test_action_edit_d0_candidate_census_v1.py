#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Optional
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = METHOD_ROOT.parents[1]
MANIFEST_PATH = METHOD_ROOT / "action_edit_sft_manifest_v2.py"
CENSUS_PATH = METHOD_ROOT / "action_edit_d0_candidate_census_v1.py"
INSVIE_CSV = Path("/private/tmp/insvie_12efa8d_train_insvie_align.csv")

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

SPEC = importlib.util.spec_from_file_location("action_edit_d0_candidate_census_v1_test_subject", CENSUS_PATH)
assert SPEC is not None and SPEC.loader is not None
census_v1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = census_v1
SPEC.loader.exec_module(census_v1)

MANIFEST_SPEC = importlib.util.spec_from_file_location("action_edit_sft_manifest_v2_test_helper", MANIFEST_PATH)
assert MANIFEST_SPEC is not None and MANIFEST_SPEC.loader is not None
manifest_v2 = importlib.util.module_from_spec(MANIFEST_SPEC)
sys.modules[MANIFEST_SPEC.name] = manifest_v2
MANIFEST_SPEC.loader.exec_module(manifest_v2)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def redigest(value: dict, digest_field: str) -> None:
    unsigned = dict(value)
    unsigned.pop(digest_field)
    value[digest_field] = census_v1.object_sha256(unsigned)


class ActionEditD0CandidateCensusV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not INSVIE_CSV.is_file():
            raise unittest.SkipTest("pinned InsViE metadata CSV is not present")
        cls.census = census_v1.build_current_census(
            REPO_ROOT, insvie_metadata_csv=INSVIE_CSV
        )
        cls.plan = census_v1.build_d0_plan(cls.census)
        cls.pools = {
            item["pool_id"]: item
            for item in cls.census["target_anchor_candidate_pools"]
        }
        cls.source_pools = {
            item["pool_id"]: item
            for item in cls.census["source_candidate_pools"]
        }

    def test_01_census_replays_and_is_non_authoritative(self):
        replay = census_v1.validate_current_census(self.census)
        self.assertEqual(replay["status"], "DATA_NOT_READY")
        self.assertEqual(replay["train_ready_N"], 0)
        self.assertEqual(replay["D0_train_eligible_effective_N"], 0)
        self.assertFalse(replay["optimizer_launch_authorized"])

    def test_02_five_candidate_pools_have_distinct_roles(self):
        self.assertEqual(len(self.pools), 5)
        self.assertEqual(
            len({item["role"] for item in self.pools.values()}), 5
        )
        self.assertTrue(
            all(item["train_ready_contribution"] == 0 for item in self.pools.values())
        )

    def test_03_full644_is_declared_not_semantically_closed(self):
        pool = self.pools["legacy_full644_preview_pairs"]
        self.assertEqual(pool["candidate_artifact_count"], 644)
        self.assertIsNone(pool["semantic_candidate_count"])
        self.assertFalse(pool["source_group_closure"])
        self.assertEqual(pool["source_exposure"], "ALL_HISTORICAL_OPTIMIZER_EXPOSED")
        self.assertEqual(pool["target_eligible_count"], 0)

    def test_04_history18_scout_verdict_does_not_grant_targets(self):
        pool = self.pools["historical_factorial_forward_target_comparison18"]
        self.assertEqual(pool["candidate_artifact_count"], 18)
        self.assertEqual(pool["semantic_candidate_count"], 18)
        self.assertEqual(pool["target_candidate_count"], 18)
        self.assertEqual(pool["target_eligible_count"], 0)
        self.assertIn(
            "treat_scout_strict_eligible_as_manifest_accepted",
            pool["forbidden_uses"],
        )

    def test_05_native8_collapses_seed_to_four_semantic_rows(self):
        pool = self.pools["native_core4_rv2v_proposals"]
        self.assertEqual(pool["candidate_artifact_count"], 8)
        self.assertEqual(pool["semantic_candidate_count"], 4)
        self.assertEqual(pool["known_source_group_count"], 4)
        self.assertFalse(pool["seed_is_semantic_identity"])
        self.assertEqual(pool["target_eligible_count"], 0)

    def test_06_quotient8_is_detached_teacher_reference(self):
        pool = self.pools["quotient_fitted_unseen_anchor8"]
        self.assertEqual(pool["candidate_artifact_count"], 8)
        self.assertEqual(pool["anchor_candidate_count"], 8)
        self.assertEqual(pool["target_candidate_count"], 0)
        self.assertEqual(pool["role"], "detached_action_teacher_reference")

    def test_07_outcome40_is_two_iids_twenty_seed_collapsed_semantics(self):
        pool = self.pools["outcome5_confirmation40"]
        self.assertEqual(pool["candidate_artifact_count"], 40)
        self.assertEqual(pool["known_source_group_count"], 2)
        self.assertEqual(pool["semantic_candidate_count"], 20)
        self.assertEqual(pool["anchor_candidate_count"], 4)
        self.assertEqual(pool["preference_candidate_count"], 36)
        self.assertEqual(pool["target_candidate_count"], 0)

    def test_08_known_cross_pool_overlap_is_visible(self):
        overlaps = {
            item["source_group_id"]: item["pool_ids"]
            for item in self.census["known_cross_pool_source_group_overlaps"]
        }
        for iid in (
            "7b88a1ca1f804f41",
            "841b5e0080a1441d",
            "a35b590961d24694",
        ):
            self.assertIn(iid, overlaps)
            self.assertIn("native_core4_rv2v_proposals", overlaps[iid])
            self.assertIn("quotient_fitted_unseen_anchor8", overlaps[iid])

    def test_09_insvie_metadata_exact_mechanical_census(self):
        row = self.census["insvie_external_metadata_candidate"]
        self.assertEqual(row["metadata_row_count"], 1_019_570)
        self.assertEqual(row["unique_video_id_count"], 1_019_570)
        self.assertEqual(row["derived_source_root_count"], 371_451)
        self.assertEqual(row["normalized_lower_strip_instruction_count"], 85_399)
        self.assertEqual(row["evidence"]["sha256"], census_v1.INSVIE_METADATA_SHA256)

    def test_10_insvie_action_screen_is_reproducible_but_non_authoritative(self):
        row = self.census["insvie_external_metadata_candidate"]
        self.assertEqual(row["preliminary_action_candidate_count"], 1_243)
        self.assertEqual(
            row["preliminary_action_prefix_counts"],
            {
                "pexel_dynamic": 41,
                "pexel_static": 6,
                "openvid_static": 1,
                "magicbrush": 578,
                "instructp2p": 617,
            },
        )
        self.assertFalse(row["preliminary_action_screen_authoritative"])
        self.assertEqual(row["target_eligible_count"], 0)

    def test_11_census_digest_tamper_fails(self):
        bad = copy.deepcopy(self.census)
        bad["d0_gap"] = 1999
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.validate_current_census(bad)

    def test_12_even_redigested_pool_authority_tamper_fails(self):
        bad = copy.deepcopy(self.census)
        bad["target_anchor_candidate_pools"][0]["train_ready_contribution"] = 1
        redigest(bad, "census_digest")
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.validate_current_census(bad)

    def test_12a_frozen_census_pin_rejects_redigested_critical_semantics(self):
        mutations = [
            ("authority", lambda row: row.__setitem__("authority_scope", "TARGET_AUTHORITY")),
            ("download", lambda row: row.__setitem__("media_download_performed", True)),
            (
                "target-pool-role",
                lambda row: row["target_anchor_candidate_pools"][2].__setitem__(
                    "role", "qualified_target"
                ),
            ),
            (
                "target-count",
                lambda row: row["target_anchor_candidate_pools"][1].__setitem__(
                    "target_eligible_count", 6
                ),
            ),
            (
                "source-pool-role",
                lambda row: row["source_candidate_pools"][0].__setitem__(
                    "role", "TRAIN_READY_LICENSED_PAIRS"
                ),
            ),
            (
                "source-pool-production",
                lambda row: row["source_candidate_pools"][0].__setitem__(
                    "production_eligible", True
                ),
            ),
            (
                "source-pool-target",
                lambda row: row["source_candidate_pools"][0].__setitem__(
                    "target_eligible_N", 16_000
                ),
            ),
            (
                "goku-card-sha",
                lambda row: row["source_candidate_pools"][0][
                    "official_dataset_card_authority"
                ].__setitem__("sha256", "0" * 64),
            ),
            (
                "insvie-prefix-count",
                lambda row: row["insvie_external_metadata_candidate"][
                    "preliminary_action_prefix_counts"
                ].__setitem__("pexel_dynamic", 47),
            ),
            (
                "insvie-screen-authority",
                lambda row: row["insvie_external_metadata_candidate"].__setitem__(
                    "preliminary_action_screen_authoritative", True
                ),
            ),
            (
                "partition-merge",
                lambda row: row["candidate_pool_partition_contract"].__setitem__(
                    "pool_classes_must_not_be_merged_for_counting", False
                ),
            ),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                bad = copy.deepcopy(self.census)
                mutate(bad)
                redigest(bad, "census_digest")
                with self.assertRaises(census_v1.CandidateCensusError):
                    census_v1.validate_current_census(bad)

    def test_12b_goku_is_separate_licensed_paired_candidate_not_target(self):
        self.assertEqual(len(self.source_pools), 1)
        pool = self.source_pools["goku_fullmotion_source_census_high_recall_16000"]
        self.assertEqual(
            pool["role"],
            "LICENSED_PAIRED_SOURCE_CANDIDATE_ENDPOINTS_UNQUALIFIED",
        )
        self.assertEqual(pool["candidate_class_after_full_qualification"], "licensed-paired")
        self.assertEqual(pool["candidate_source_N"], 16_000)
        self.assertEqual(pool["candidate_paired_endpoint_presence_N"], 16_000)
        self.assertEqual(pool["source_catalog_v2_eligible_N"], 0)
        self.assertEqual(pool["target_eligible_N"], 0)
        self.assertEqual(pool["train_ready_contribution"], 0)
        self.assertFalse(pool["production_eligible"])
        self.assertFalse(pool["formal_training_authorized"])

    def test_12c_goku_remote_pins_card_and_unsealed_stat_are_explicit(self):
        pool = self.source_pools["goku_fullmotion_source_census_high_recall_16000"]
        card = pool["official_dataset_card_authority"]
        self.assertEqual(
            card["sha256"],
            "c8fb7f1a024c0c83d72e46ac76dfca590b95da69862b14f7dca6a15c910a4e49",
        )
        self.assertEqual(card["size_bytes"], 7_823)
        self.assertEqual(card["repository"], "Goku-2M/GOKU-2M")
        self.assertEqual(card["license_label"], "CC-BY-NC-4.0")
        self.assertEqual(card["declared_configuration"], "subject_movement")
        self.assertEqual(card["hf_cache_metadata_opaque_lines"][2], "1784845015.9175262")
        self.assertFalse(card["cache_timestamp_has_qualification_semantics"])
        self.assertEqual(len(pool["remote_authority_members"]), 6)
        self.assertFalse(pool["user_reported_readonly_endpoint_stat"]["sealed_authority"])
        self.assertFalse(
            pool["paper_method_context_not_row_qualification"][
                "provider_filtering_is_0817_per_row_human_review"
            ]
        )

    def test_12d_pinned_insvie_metadata_is_mandatory(self):
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.build_current_census(REPO_ROOT, insvie_metadata_csv=None)

    def test_13_d0_plan_is_future_exact_2k_matrix(self):
        replay = census_v1.validate_d0_plan(
            self.plan, expected_census_digest=self.census["census_digest"]
        )
        self.assertTrue(replay["counts_are_future_requirements_not_existing_assets"])
        self.assertEqual(replay["current_train_ready_N"], 0)
        self.assertEqual(replay["target_D0_train_eligible_effective_N"], 2000)
        self.assertEqual(sum(replay["future_training_subset_counts"].values()), 2000)

    def test_14_d0_provenance_and_truth_caps_are_exact(self):
        provenance = self.plan["future_target_provenance_counts"]
        truth = self.plan["future_target_truth_class_counts"]
        self.assertEqual(provenance["teacher-pseudo"], 850)
        self.assertEqual(truth["teacher-pseudo"], 450)
        self.assertEqual(truth["continuation"], 400)
        self.assertEqual(truth["noop"], 300)
        self.assertEqual(
            self.plan["provenance_gates"]["high_confidence_real_simulator_licensed_non_noop_planned_rows"],
            850,
        )

    def test_15_source_plan_does_not_count_current_anchors(self):
        source = self.plan["source_and_group_plan"]
        self.assertEqual(source["future_unique_source_goal"], 1000)
        self.assertEqual(source["minimum_new_unexposed_sources_if_all_644_are_verified_unique"], 356)
        self.assertTrue(source["all_full644_sources_must_be_marked_exposed"])
        self.assertFalse(source["seed_transcode_copy_paraphrase_increase_effective_N"])

    def test_16_teacher_disjoint_is_a_hard_future_requirement(self):
        teacher = self.plan["teacher_disjoint_requirements"]
        self.assertTrue(teacher["train_teacher_outputs_byte_disjoint_across_splits"])
        self.assertTrue(teacher["reserve_at_least_one_teacher_family_absent_from_train_for_promotion_diagnostic"])
        self.assertTrue(teacher["reserve_separate_teacher_family_absent_from_train_calibration_promotion_for_locked_final_subset"])

    def test_17_insvie_card_license_is_not_row_rights(self):
        insvie = self.plan["external_candidate_priority"][0]
        self.assertEqual(insvie["dataset_card_license_label"], "CC-BY-4.0")
        self.assertFalse(insvie["rights_layers"]["card_license_proves_each_upstream_asset_rights"])
        self.assertIn("UNKNOWN", insvie["rights_layers"]["per_source_upstream_license_and_redistribution_rights"])
        self.assertFalse(insvie["download_authorized"])
        self.assertEqual(insvie["target_eligible_count"], 0)

    def test_18_insvie_plan_uses_archive_index_before_media_transfer(self):
        insvie = self.plan["external_candidate_priority"][0]
        self.assertEqual(insvie["archive_byte_count"], 765_117_294_333)
        self.assertEqual(insvie["source_zip_count"], 18)
        self.assertEqual(insvie["edited_zip_count"], 19)
        joined = " ".join(insvie["retrieval_plan"])
        self.assertIn("central directories", joined)
        self.assertIn("range-fetch only selected", joined)

    def test_19_easyv2v_and_dynaedit_roles_are_not_assets(self):
        candidates = {
            item["candidate_id"]: item
            for item in self.plan["external_candidate_priority"]
        }
        easy = candidates["easyv2v-human-action-route"]
        dyna = candidates["dynaedit-source-conditioned-teacher"]
        self.assertEqual(easy["role"], "pipeline_inspiration_only")
        self.assertFalse(easy["downloadable_paired_asset_asserted"])
        self.assertEqual(dyna["role"], "teacher_candidate_only")
        self.assertEqual(dyna["target_eligible_count"], 0)

    def test_20_plan_digest_and_census_pin_tamper_fail(self):
        bad = copy.deepcopy(self.plan)
        bad["input_census_digest"] = "0" * 64
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.validate_d0_plan(
                bad, expected_census_digest=self.census["census_digest"]
            )

    def test_21_redigested_teacher_overcap_plan_fails(self):
        bad = copy.deepcopy(self.plan)
        bad["future_target_provenance_counts"]["teacher-pseudo"] = 1001
        bad["future_target_provenance_counts"]["real"] -= 151
        bad["future_target_provenance_counts"]["licensed-dataset"] += 150
        redigest(bad, "plan_digest")
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.validate_d0_plan(
                bad, expected_census_digest=self.census["census_digest"]
            )

    def test_21a_frozen_plan_pin_rejects_redigested_critical_semantics(self):
        mutations = [
            (
                "future-labelled-current",
                lambda row: row.__setitem__(
                    "counts_are_future_requirements_not_existing_assets", False
                ),
            ),
            (
                "matrix-truth",
                lambda row: row["future_row_matrix"][0].__setitem__(
                    "semantic_truth_class", "teacher-pseudo"
                ),
            ),
            (
                "provenance-cap",
                lambda row: row["provenance_gates"].__setitem__(
                    "teacher_pseudo_max_fraction", 1.0
                ),
            ),
            (
                "source-cap",
                lambda row: row["source_and_group_plan"].__setitem__(
                    "hard_source_semantic_edit_cap", 100
                ),
            ),
            (
                "source-candidate-role",
                lambda row: row["source_candidate_pool_inputs"][0].__setitem__(
                    "role", "train_ready_pairs"
                ),
            ),
            (
                "goku-current-materialized",
                lambda row: row["licensed_goku_candidate_materialization"].__setitem__(
                    "current_materialized_and_qualified_N", 500
                ),
            ),
            (
                "split-overlap",
                lambda row: row["split_and_leakage_requirements"].__setitem__(
                    "source_actor_scene_upstream_target_bytes_cross_split_overlap_allowed",
                    True,
                ),
            ),
            (
                "group-key",
                lambda row: row["split_and_leakage_requirements"][
                    "group_key_fields"
                ].pop(),
            ),
            (
                "teacher-disjoint",
                lambda row: row["teacher_disjoint_requirements"].__setitem__(
                    "train_teacher_outputs_byte_disjoint_across_splits", False
                ),
            ),
            (
                "insvie-download",
                lambda row: row["external_candidate_priority"][0].__setitem__(
                    "download_authorized", True
                ),
            ),
            (
                "dyna-role",
                lambda row: row["external_candidate_priority"][2].__setitem__(
                    "role", "qualified_target_source"
                ),
            ),
            ("formal", lambda row: row.__setitem__("formal_training_authorized", True)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                bad = copy.deepcopy(self.plan)
                mutate(bad)
                redigest(bad, "plan_digest")
                with self.assertRaises(census_v1.CandidateCensusError):
                    census_v1.validate_d0_plan(
                        bad,
                        expected_census_digest=self.census["census_digest"],
                    )

    def test_21b_goku_licensed_500_is_future_plan_not_current_data(self):
        route = self.plan["licensed_goku_candidate_materialization"]
        self.assertTrue(route["counts_are_future_requirements_not_existing_assets"])
        self.assertEqual(route["current_materialized_and_qualified_N"], 0)
        self.assertEqual(route["future_required_total_N"], 500)
        self.assertEqual(
            [item["future_required_rows"] for item in route["future_rows"]],
            [300, 100, 100],
        )
        self.assertEqual(
            [item["semantic_truth_class"] for item in route["future_rows"]],
            ["licensed-paired", "licensed-paired", "noop"],
        )
        self.assertFalse(route["provider_or_census_selection_self_qualifies_row"])
        self.assertEqual(route["target_eligible_N"], 0)
        self.assertEqual(route["train_ready_contribution"], 0)

    def test_22_candidate_row_shape_never_self_grants_train_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source_path = root / "source.mp4"
            target_path = root / "target.mp4"
            source_path.write_bytes(b"source")
            target_path.write_bytes(b"target")
            instruction_text = "Make the actor sit and preserve the scene"
            instruction = {
                "text": instruction_text,
                "sha256": sha(instruction_text.encode("utf-8")),
                "size_bytes": len(instruction_text.encode("utf-8")),
                "encoding": "utf-8",
                "semantic_id": "0" * 64,
                "template_family": "action-imperative-v1",
                "actor": "actor-1",
                "action": "sit",
                "object": "ground",
                "direction": "down",
                "speed": "normal",
                "amplitude": "full",
                "onset": "early",
                "outcome": "completed",
                "terminal_state": "seated",
                "preserve": ["background", "camera", "identity"],
            }
            instruction["semantic_id"] = manifest_v2.expected_instruction_semantic_id(instruction)
            row = {
                "schema_version": manifest_v2.ROW_SCHEMA,
                "row_id": "0" * 64,
                "semantic_edit_id": "0" * 64,
                "action_family": "sit-v1",
                "upstream_group_id": "upstream-1",
                "actor_scene_group_id": "actor-scene-1",
                "source": {
                    "path": source_path.as_posix(), "sha256": sha(b"source"), "size_bytes": 6,
                    "source_id": "source-1", "canonical_source_id": "canonical-source-1",
                    "actor_ids": ["actor-1"], "scene_id": "scene-1", "camera_class": "static",
                    "initial_state": "standing",
                },
                "instruction": instruction,
                "target": {
                    "path": target_path.as_posix(), "sha256": sha(b"target"), "size_bytes": 6,
                    "provenance": "teacher-pseudo", "semantic_truth_class": "teacher-pseudo",
                    "teacher_id": "teacher-candidate-v1", "qualification_status": "pending",
                    "qualification_receipt": None, "human_review": None,
                    "human_review_receipt_sha256": None, "action_feature_encoder_sha256": None,
                    "q_y_sha256": None, "compatibility_receipt_sha256": None,
                },
                "action_anchors": [], "annotations": {}, "row_tier": "train",
                "training_subset": "action_motion", "calibration_kind": None,
                "evaluation_stratum": None, "generation_seed": 7,
                "copy_of_row_id": None, "transcode_of_sha256": None,
            }
            row["semantic_edit_id"] = manifest_v2.expected_semantic_edit_id(row)
            row["row_id"] = manifest_v2.expected_row_id(row)
            report = census_v1.classify_candidate_rows([row])
            self.assertEqual(report["qualification_shape_counts"]["pending"], 1)
            self.assertEqual(report["train_ready_N"], 0)
            self.assertTrue(report["formal_manifest_v2_required"])

    def test_23_no_network_or_launcher_surface_exists(self):
        source = CENSUS_PATH.read_text(encoding="utf-8")
        for forbidden in ("requests.", "urllib.request", "subprocess", "ssh ", "torchrun", "sbatch"):
            self.assertNotIn(forbidden, source)

    def test_24_redigested_matrix_summary_mismatch_fails(self):
        bad = copy.deepcopy(self.plan)
        bad["future_row_matrix"][0]["future_required_rows"] -= 1
        bad["future_row_matrix"][1]["future_required_rows"] += 1
        redigest(bad, "plan_digest")
        with self.assertRaises(census_v1.CandidateCensusError):
            census_v1.validate_d0_plan(
                bad, expected_census_digest=self.census["census_digest"]
            )

    def test_25_frozen_object_digests_match_code_pins(self):
        self.assertEqual(self.census["census_digest"], census_v1.FROZEN_CENSUS_DIGEST)
        self.assertEqual(self.plan["plan_digest"], census_v1.FROZEN_D0_PLAN_DIGEST)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import v10a2_p0_target_blind_perceptual_frozen_observer_qualification_v1 as qualification


ASSET_ROOT = METHOD_ROOT / "assets"
PREREG_PATH = ASSET_ROOT / "v10a2_p0_target_blind_perceptual_frozen_observer_qualification_prereg_v1.json"
SCHEMA_PATH = ASSET_ROOT / "v10a2_p0_target_blind_perceptual_frozen_observer_qualification_receipt_schema_v1.json"
REGISTRY_PATH = ASSET_ROOT / "v10a2_p0_source_only_64_provisional_v1.json"
ACTUAL_PATH = ASSET_ROOT / "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def rehash(value: dict, key: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = qualification.object_sha256(result)
    return result


def model_closure(prefix: str) -> list[dict]:
    rows = []
    for name in qualification.FOUNDATION_MODELS:
        state = digest(f"{prefix}-{name}-state")
        rows.append(
            {
                "name": name,
                "weight_closure_sha256": digest(f"{prefix}-{name}-weight"),
                "config_preprocess_source_closure_sha256": digest(f"{prefix}-{name}-config"),
                "environment_closure_sha256": digest(f"{prefix}-{name}-env"),
                "state_before_sha256": state,
                "state_after_sha256": state,
                "eval_mode": True,
                "requires_grad_true_count": 0,
                "gradient_tensor_count": 0,
                "real_model": True,
            }
        )
    return rows


MODEL_BINDING = digest("model-binding")
ANCESTRY = digest("ancestry")
ACCESS_LEDGER = digest("access-ledger")
PERCEPTUAL_SEAL = digest("perceptual-seal")
OBSERVER_SEAL = digest("observer-seal")
PERCEPTUAL_FILE = digest("perceptual-file")
OBSERVER_FILE = digest("observer-file")


def valid_perceptual_receipt() -> dict:
    registry = load(REGISTRY_PATH)
    value = {
        "schema_version": qualification.PERCEPTUAL_SCHEMA,
        "qualification_id": qualification.QUALIFICATION_ID,
        "decision": "PASS",
        "pins": {
            "qualification_prereg_self_sha256": qualification.EXPECTED_PREREG_SELF_SHA256,
            "provisional_registry_file_sha256": qualification.provisional.EXPECTED_REGISTRY_FILE_SHA256,
            "provisional_registry_self_sha256": qualification.provisional.EXPECTED_REGISTRY_SELF_SHA256,
            "actual_manifest_file_sha256": qualification.provisional.EXPECTED_ACTUAL_MANIFEST_FILE_SHA256,
            "actual_manifest_self_sha256": qualification.provisional.EXPECTED_ACTUAL_MANIFEST_SELF_SHA256,
        },
        "provenance": {
            "runner_source_sha256": digest("perceptual-runner"),
            "model_binding_receipt_sha256": MODEL_BINDING,
            "ancestor_provenance_receipt_sha256": ANCESTRY,
            "target_blind_access_ledger_sha256": ACCESS_LEDGER,
            "external_completion_seal_sha256": PERCEPTUAL_SEAL,
            "real_models_executed": True,
            "synthetic_or_mock_features": False,
        },
        "target_blind": {
            "auditor_process_disjoint_from_training": True,
            "training_process_target_media_reads": 0,
            "training_process_target_feature_reads": 0,
            "released_target_feature_bytes": 0,
            "released_target_neighbor_identities": 0,
            "released_target_neighbor_scores": 0,
            "released_candidate_cluster_commitments_to_training": 0,
            "protected_feature_files_released": False,
            "access_ledger_complete": True,
        },
        "sampling": {
            "candidate_media_count": 64,
            "actual_media_count": 32,
            "frame_grid_count_per_media": 32,
            "candidate_decoded_frame_count": 2048,
            "actual_decoded_frame_count": 1024,
            "candidate_media_sha256_revalidated_count": 64,
            "actual_media_sha256_revalidated_count": 32,
            "decode_failure_count": 0,
            "missing_transform_count": 0,
            "nonfinite_measure_count": 0,
        },
        "thresholds": {
            "phash_hamming_max_inclusive": 6,
            "phash_close_aligned_frame_min": 8,
            "dinov2_aligned_median_cosine_min_inclusive": 0.95,
            "vjepa2_video_cosine_min_inclusive": 0.97,
            "threshold_equality_is_duplicate": True,
            "temporal_directions": ["forward", "reverse"],
            "nuisance_alignments": ["trim", "time_shift", "reencode", "resize", "crop", "horizontal_flip"],
        },
        "comparisons": {
            "candidate_candidate_pair_count": 2016,
            "candidate_actual_pair_count": 2048,
            "total_pair_count": 4064,
            "candidate_candidate_near_duplicate_edge_count": 0,
            "candidate_actual_near_duplicate_edge_count": 0,
            "candidate_unique_perceptual_cluster_count": 64,
            "candidate_ancestor_unique_count": 64,
            "candidate_candidate_ancestor_collision_count": 0,
            "candidate_actual_ancestor_overlap_count": 0,
            "missing_pair_comparison_count": 0,
        },
        "model_closure": model_closure("perceptual"),
        "safety": {
            "autograd_enabled": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "parameter_updates": 0,
            "generator_loaded": False,
            "generator_forward_calls": 0,
            "b0_executed": False,
            "b0_replaced": False,
        },
        "raw_release": {
            "raw_actual_frames_released": False,
            "raw_actual_embeddings_released": False,
            "raw_actual_neighbor_scores_released": False,
            "raw_candidate_embeddings_released": False,
            "zeroization_verified": True,
        },
        "rows": [
            {
                "ordinal": ordinal,
                "uuid": row["uuid"],
                "source_media_sha256": row["source_media_sha256"],
                "opaque_cluster_commitment": digest(f"cluster-{ordinal}"),
                "candidate_near_duplicate_count": 0,
                "actual_near_duplicate_count": 0,
                "ancestor_overlap": False,
                "status": "PASS",
            }
            for ordinal, row in enumerate(registry["rows"])
        ],
    }
    return rehash(value, "receipt_sha256")


def valid_observer_receipt() -> dict:
    registry = load(REGISTRY_PATH)
    rows = []
    for ordinal, registered in enumerate(registry["rows"]):
        rich = ordinal % 16 < 4
        rows.append(
            {
                "ordinal": ordinal,
                "uuid": registered["uuid"],
                "source_media_sha256": registered["source_media_sha256"],
                "stratum": registered["stratum"],
                "parent_track_count": 2,
                "min_parent_visible_tau": 6,
                "min_parent_soft_mass_fraction": 0.001,
                "min_parent_alive_posterior": 0.5,
                "parent_track_idf1": 0.75,
                "parent_mask_iou_median": 0.55,
                "parent_slot_alias_count": 0,
                "identity_switches_max": 1,
                "mean_dustbin_mass": 0.2,
                "single_tau_dustbin_mass_max": 0.35,
                "part_rich": rich,
                "qualifying_part_count": 2 if rich else 0,
                "min_part_visible_tau": 4 if rich else None,
                "min_part_parent_relative_mass": 0.02 if rich else None,
                "max_part_outside_parent_mass": 0.0 if rich else None,
                "min_part_mask_iou": 0.35 if rich else None,
                "pair_inventory_complete": True,
                "dynamic_edge_lifecycle": rich,
                "dynamic_edge_distinct_tau": 3 if rich else 0,
                "dynamic_edge_same_endpoints": True,
                "phase0_only_edge": False,
                "later_persist_or_deactivate": rich,
                "relative_velocity_all_finite": True,
                "dynamic_edge_f1": 0.8 if rich else None,
                "event_phase_error_tau": 1 if rich else None,
                "three_plus_member_group_positive": rich,
                "group_member_count": 3 if rich else 0,
                "group_active_tau": 2 if rich else 0,
                "group_members_unique": True,
                "group_permutation_equivariant": True,
                "nonfinite_count": 0,
                "forced_match_count": 0,
                "status": "PASS",
            }
        )
    per_stratum = {stratum: 4 for stratum in qualification.STRATA}
    value = {
        "schema_version": qualification.OBSERVER_SCHEMA,
        "qualification_id": qualification.QUALIFICATION_ID,
        "decision": "PASS",
        "pins": {
            "qualification_prereg_self_sha256": qualification.EXPECTED_PREREG_SELF_SHA256,
            "provisional_registry_file_sha256": qualification.provisional.EXPECTED_REGISTRY_FILE_SHA256,
            "provisional_registry_self_sha256": qualification.provisional.EXPECTED_REGISTRY_SELF_SHA256,
        },
        "provenance": {
            "runner_source_sha256": digest("observer-runner"),
            "model_binding_receipt_sha256": MODEL_BINDING,
            "external_completion_seal_sha256": OBSERVER_SEAL,
            "real_models_executed": True,
            "synthetic_or_mock_outputs": False,
        },
        "source_only": {
            "candidate_source_media_count": 64,
            "target_media_read_count": 0,
            "actual_manifest_read_count": 0,
            "prompt_caption_instruction_action_or_family_read_count": 0,
            "metadata_to_model_input_count": 0,
        },
        "sampling": {
            "view_ids": ["reference", "evaluation"],
            "tau_per_view": 8,
            "temporal_index_overlap_count": 0,
            "geometry_inverse_map_missing_count": 0,
            "decoded_candidate_videos": 64,
            "decoded_view_sequences": 128,
            "decode_failure_count": 0,
        },
        "call_counts": {
            "sam2_keyframe_calls": 1024,
            "dinov2_keyframe_calls": 1024,
            "cotracker_video_calls": 128,
            "vjepa2_video_calls": 128,
        },
        "model_closure": model_closure("observer"),
        "safety": {
            "autograd_enabled": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "parameter_updates": 0,
            "generator_import_count": 0,
            "generator_loaded": False,
            "generator_forward_calls": 0,
            "generator_capture_calls": 0,
            "route_or_lora_loaded": False,
            "binder_loaded": False,
            "slot_model_loaded": False,
            "b0_executed": False,
            "b0_replaced": False,
        },
        "aggregate": {
            "row_count": 64,
            "parent_qualified_row_count": 64,
            "part_rich_row_count": 16,
            "dynamic_edge_lifecycle_row_count": 16,
            "three_plus_member_group_positive_row_count": 16,
            "part_rich_rows_by_stratum": per_stratum,
            "dynamic_edge_rows_by_stratum": per_stratum,
            "group_positive_rows_by_stratum": per_stratum,
            "pair_inventory_complete_row_count": 64,
            "nonfinite_count": 0,
            "observer_abstention_count": 0,
            "forced_match_count": 0,
            "raw_masks_tracks_descriptors_or_hidden_released": False,
        },
        "rows": rows,
    }
    return rehash(value, "receipt_sha256")


def validate_perceptual(value: dict) -> None:
    qualification.validate_perceptual_evidence_receipt(
        value,
        registry=load(REGISTRY_PATH),
        observed_file_sha256=PERCEPTUAL_FILE,
        expected_file_sha256=PERCEPTUAL_FILE,
        expected_model_binding_sha256=MODEL_BINDING,
        expected_ancestor_receipt_sha256=ANCESTRY,
        expected_access_ledger_sha256=ACCESS_LEDGER,
        expected_completion_seal_sha256=PERCEPTUAL_SEAL,
    )


def validate_observer(value: dict) -> None:
    qualification.validate_observer_evidence_receipt(
        value,
        registry=load(REGISTRY_PATH),
        observed_file_sha256=OBSERVER_FILE,
        expected_file_sha256=OBSERVER_FILE,
        expected_model_binding_sha256=MODEL_BINDING,
        expected_completion_seal_sha256=OBSERVER_SEAL,
    )


class V10A2P0QualificationV1Tests(unittest.TestCase):
    def assert_perceptual_rejected(self, value: dict, message: str) -> None:
        with self.assertRaisesRegex(qualification.V10A2P0QualificationError, message):
            validate_perceptual(rehash(value, "receipt_sha256"))

    def assert_observer_rejected(self, value: dict, message: str) -> None:
        with self.assertRaisesRegex(qualification.V10A2P0QualificationError, message):
            validate_observer(rehash(value, "receipt_sha256"))

    def test_checked_in_authorities_have_canonical_self_and_file_hashes(self) -> None:
        prereg = load(PREREG_PATH)
        prereg_self = prereg.pop("qualification_prereg_sha256")
        self.assertEqual(prereg_self, qualification.EXPECTED_PREREG_SELF_SHA256)
        self.assertEqual(qualification.object_sha256(prereg), prereg_self)
        self.assertEqual(qualification.file_sha256(PREREG_PATH), qualification.EXPECTED_PREREG_FILE_SHA256)
        schema = load(SCHEMA_PATH)
        schema_self = schema.pop("receipt_schema_sha256")
        self.assertEqual(schema_self, qualification.EXPECTED_RECEIPT_SCHEMA_SELF_SHA256)
        self.assertEqual(qualification.object_sha256(schema), schema_self)
        self.assertEqual(qualification.file_sha256(SCHEMA_PATH), qualification.EXPECTED_RECEIPT_SCHEMA_FILE_SHA256)

    def test_current_runner_is_explicit_no_abstain_and_preserves_b0(self) -> None:
        receipt = dict(qualification.run_qualification())
        self.assertEqual(receipt["status"], qualification.ONLY_STATUS)
        self.assertEqual(receipt["decision"], qualification.ONLY_DECISION)
        self.assertEqual(receipt["blockers"], list(qualification.REQUIRED_ARTIFACT_IDS))
        self.assertEqual(receipt["frozen_base_b0"]["arm_id"], "B0_FROZEN_BASE")
        self.assertTrue(receipt["frozen_base_b0"]["first_class_future_arm_preserved"])
        self.assertFalse(receipt["frozen_base_b0"]["replaced_by_historical_or_metadata_row"])
        self.assertFalse(receipt["execution"]["generator_loaded"])
        self.assertEqual(receipt["execution"]["generator_forward_calls"], 0)
        self.assertEqual(receipt["execution"]["parameter_updates"], 0)
        self.assertFalse(any(receipt["permissions"].values()))
        receipt_hash = receipt.pop("receipt_sha256")
        self.assertEqual(qualification.object_sha256(receipt), receipt_hash)

    def test_unpinned_candidate_receipts_are_not_opened_or_admitted(self) -> None:
        receipt = qualification.run_qualification(
            perceptual_receipt_path=Path("/does/not/exist/perceptual.json"),
            observer_receipt_path=Path("/does/not/exist/observer.json"),
        )
        self.assertEqual(receipt["decision"], "ABSTAIN")
        self.assertIn("UNPREREGISTERED_PERCEPTUAL_RECEIPT_IGNORED", receipt["blockers"])
        self.assertIn("UNPREREGISTERED_OBSERVER_RECEIPT_IGNORED", receipt["blockers"])
        self.assertFalse(receipt["perceptual_qualification"]["evidence_admitted"])
        self.assertFalse(receipt["observer_qualification"]["evidence_admitted"])

    def test_rehashed_contract_cannot_enable_training_or_replace_b0(self) -> None:
        prereg = load(PREREG_PATH)
        prereg["current_authorization"]["p0_slot_pretraining_authorized"] = True
        prereg["frozen_base_b0"]["replaced_by_historical_p0_or_metadata_row"] = True
        sealed = rehash(prereg, "qualification_prereg_sha256")
        with mock.patch.object(qualification, "EXPECTED_PREREG_SELF_SHA256", sealed["qualification_prereg_sha256"]):
            with self.assertRaisesRegex(qualification.V10A2P0QualificationError, "Frozen Base|authorization"):
                qualification.validate_qualification_prereg(
                    sealed, observed_file_sha256=qualification.EXPECTED_PREREG_FILE_SHA256
                )

    def test_rehashed_contract_rejects_p3_target_teacher_reward(self) -> None:
        prereg = load(PREREG_PATH)
        prereg["target_teacher_boundary"]["p3_route_process_target_teacher_read_count"] = 1
        prereg["target_teacher_boundary"]["p3_reward_source"] = "target_pair_graph"
        sealed = rehash(prereg, "qualification_prereg_sha256")
        with mock.patch.object(qualification, "EXPECTED_PREREG_SELF_SHA256", sealed["qualification_prereg_sha256"]):
            with self.assertRaisesRegex(qualification.V10A2P0QualificationError, "target-teacher/P3 route firewall"):
                qualification.validate_qualification_prereg(
                    sealed, observed_file_sha256=qualification.EXPECTED_PREREG_FILE_SHA256
                )

    def test_semantically_valid_perceptual_fixture_is_accepted_only_by_low_level_validator(self) -> None:
        validate_perceptual(valid_perceptual_receipt())

    def test_perceptual_target_feature_or_neighbor_leak_is_rejected(self) -> None:
        receipt = valid_perceptual_receipt()
        receipt["target_blind"]["released_target_feature_bytes"] = 1
        self.assert_perceptual_rejected(receipt, "target-blind")

    def test_perceptual_missing_comparison_and_near_duplicate_are_rejected(self) -> None:
        receipt = valid_perceptual_receipt()
        receipt["comparisons"]["candidate_actual_pair_count"] = 2047
        receipt["comparisons"]["candidate_actual_near_duplicate_edge_count"] = 1
        self.assert_perceptual_rejected(receipt, "comparison/cluster")

    def test_perceptual_transitive_duplicate_cluster_is_rejected(self) -> None:
        receipt = valid_perceptual_receipt()
        receipt["rows"][1]["opaque_cluster_commitment"] = receipt["rows"][0]["opaque_cluster_commitment"]
        self.assert_perceptual_rejected(receipt, "clusters are not unique")

    def test_perceptual_decode_or_transform_failure_is_rejected(self) -> None:
        receipt = valid_perceptual_receipt()
        receipt["sampling"]["decode_failure_count"] = 1
        self.assert_perceptual_rejected(receipt, "sampling/decode")

    def test_perceptual_model_state_drift_is_rejected(self) -> None:
        receipt = valid_perceptual_receipt()
        receipt["model_closure"][0]["state_after_sha256"] = digest("changed")
        self.assert_perceptual_rejected(receipt, "state changed")

    def test_semantically_valid_observer_fixture_is_accepted_only_by_low_level_validator(self) -> None:
        validate_observer(valid_observer_receipt())

    def test_observer_target_or_metadata_read_is_rejected(self) -> None:
        receipt = valid_observer_receipt()
        receipt["source_only"]["target_media_read_count"] = 1
        self.assert_observer_rejected(receipt, "not source-only")

    def test_observer_generator_or_update_is_rejected(self) -> None:
        receipt = valid_observer_receipt()
        receipt["safety"]["generator_loaded"] = True
        receipt["safety"]["parameter_updates"] = 1
        self.assert_observer_rejected(receipt, "training/generator/B0")

    def test_observer_sampling_views_must_be_disjoint(self) -> None:
        receipt = valid_observer_receipt()
        receipt["sampling"]["temporal_index_overlap_count"] = 1
        self.assert_observer_rejected(receipt, "dual-view sampling")

    def test_observer_parent_and_dustbin_thresholds_are_inclusive(self) -> None:
        validate_observer(valid_observer_receipt())
        for key, bad, message in (
            ("parent_track_idf1", 0.749, "IDF1"),
            ("parent_mask_iou_median", 0.549, "mask IoU"),
            ("mean_dustbin_mass", 0.201, "mean dustbin"),
            ("single_tau_dustbin_mass_max", 0.351, "single-tau dustbin"),
        ):
            receipt = valid_observer_receipt()
            receipt["rows"][0][key] = bad
            self.assert_observer_rejected(receipt, message)

    def test_observer_part_containment_and_iou_are_hard(self) -> None:
        for key, bad, message in (
            ("max_part_outside_parent_mass", 0.0000001, "leaves parent"),
            ("min_part_mask_iou", 0.349, "part IoU"),
        ):
            receipt = valid_observer_receipt()
            receipt["rows"][0][key] = bad
            self.assert_observer_rejected(receipt, message)

    def test_observer_phase0_edge_f1_and_phase_error_are_hard(self) -> None:
        for key, bad, message in (
            ("dynamic_edge_same_endpoints", False, "changes endpoints"),
            ("phase0_only_edge", True, "phase0-only"),
            ("dynamic_edge_f1", 0.799, "edge F1"),
            ("event_phase_error_tau", 2, "phase error"),
        ):
            receipt = valid_observer_receipt()
            receipt["rows"][0][key] = bad
            self.assert_observer_rejected(receipt, message)

    def test_observer_parent_alias_and_one_tau_group_are_rejected(self) -> None:
        receipt = valid_observer_receipt()
        receipt["rows"][0]["parent_slot_alias_count"] = 1
        self.assert_observer_rejected(receipt, "parent slots alias")
        receipt = valid_observer_receipt()
        receipt["rows"][0]["group_active_tau"] = 1
        self.assert_observer_rejected(receipt, "group lacks temporal support")

    def test_observer_duplicate_group_member_and_row_replay_are_rejected(self) -> None:
        receipt = valid_observer_receipt()
        receipt["rows"][0]["group_members_unique"] = False
        self.assert_observer_rejected(receipt, "repeats a member")
        receipt = valid_observer_receipt()
        receipt["rows"][1]["uuid"] = receipt["rows"][0]["uuid"]
        self.assert_observer_rejected(receipt, "identity/replay binding")

    def test_observer_pool_coverage_cannot_be_faked_by_aggregate(self) -> None:
        receipt = valid_observer_receipt()
        receipt["rows"][0]["part_rich"] = False
        receipt["rows"][0]["qualifying_part_count"] = 0
        receipt["rows"][0]["min_part_visible_tau"] = None
        receipt["rows"][0]["min_part_parent_relative_mass"] = None
        receipt["rows"][0]["max_part_outside_parent_mass"] = None
        receipt["rows"][0]["min_part_mask_iou"] = None
        self.assert_observer_rejected(receipt, "aggregate does not match rows")

    def test_completion_seal_cannot_use_file_presence_as_authority(self) -> None:
        evidence = valid_perceptual_receipt()
        seal = {
            "schema_version": qualification.COMPLETION_SEAL_SCHEMA,
            "qualification_id": qualification.QUALIFICATION_ID,
            "stage": "perceptual",
            "decision": "PASS",
            "external_controller": True,
            "producer_process_disjoint": True,
            "candidate_file_presence_is_completion_authority": True,
            "evidence_receipt_file_sha256": PERCEPTUAL_FILE,
            "evidence_receipt_self_sha256": evidence["receipt_sha256"],
        }
        seal = rehash(seal, "seal_sha256")
        with self.assertRaisesRegex(qualification.V10A2P0QualificationError, "presence cannot complete"):
            qualification.validate_external_completion_seal(
                seal,
                stage="perceptual",
                evidence_receipt_file_sha256=PERCEPTUAL_FILE,
                evidence_receipt_self_sha256=evidence["receipt_sha256"],
            )


if __name__ == "__main__":
    unittest.main()

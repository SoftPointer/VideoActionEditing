from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import v10a2_hierarchical_object_event_graph_preflight_v1 as preflight


PREREGISTRATION_PATH = (
    METHOD_ROOT
    / "assets"
    / "v10a2_hierarchical_object_event_graph_prereg_v1.json"
)
SOURCE_MANIFEST_PATH = (
    METHOD_ROOT
    / "assets"
    / "target_factorized_soft_ot_graph_teacher_manifest_v5_r1b.json"
)
PROVISIONAL_REGISTRY_PATH = (
    METHOD_ROOT / "assets" / "v10a2_p0_source_only_64_provisional_v1.json"
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rehash_preregistration(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("preregistration_sha256", None)
    result["preregistration_sha256"] = preflight.object_sha256(result)
    return result


def build(
    *,
    preregistration: dict | None = None,
    provisional_registry: dict | None = None,
) -> dict:
    return dict(
        preflight.build_preflight_receipt(
            preregistration=(
                load(PREREGISTRATION_PATH)
                if preregistration is None
                else preregistration
            ),
            source_manifest=load(SOURCE_MANIFEST_PATH),
            source_manifest_file_sha256=preflight.file_sha256(
                SOURCE_MANIFEST_PATH
            ),
            provisional_registry=(
                load(PROVISIONAL_REGISTRY_PATH)
                if provisional_registry is None
                else provisional_registry
            ),
            provisional_registry_file_sha256=preflight.file_sha256(
                PROVISIONAL_REGISTRY_PATH
            ),
        )
    )


class V10A2HierarchicalObjectEventGraphPreflightV1Tests(unittest.TestCase):
    def assert_semantic_tamper_rejected(
        self, preregistration: dict, message: str
    ) -> None:
        sealed = rehash_preregistration(preregistration)
        # Exercise the semantic invariant independently of the immutable digest.
        with mock.patch.object(
            preflight,
            "EXPECTED_PREREGISTRATION_SHA256",
            sealed["preregistration_sha256"],
        ):
            with self.assertRaisesRegex(preflight.V10A2PreflightError, message):
                build(preregistration=sealed)

    def test_checked_in_preregistration_has_canonical_self_hash(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        expected = preregistration.pop("preregistration_sha256")
        self.assertEqual(expected, preflight.EXPECTED_PREREGISTRATION_SHA256)
        self.assertEqual(preflight.object_sha256(preregistration), expected)

    def test_current_preflight_is_hard_pre_run_no(self) -> None:
        receipt = build()
        self.assertEqual(receipt["status"], preflight.ONLY_STATUS)
        self.assertEqual(
            receipt["authorization_blockers"], list(preflight.BLOCKERS)
        )
        self.assertEqual(receipt["missing_dependency_count"], 10)
        for key in (
            "launch_authorized",
            "gpu_launch_authorized",
            "slot_pretraining_authorized",
            "binder_training_authorized",
            "generator_training_authorized",
            "official_runner_authorized",
            "training_executed",
            "optimizer_created",
            "can_emit_ready_status",
            "representation_admitted",
            "stable_action_representation_supported",
            "transferable_action_representation_supported",
            "scientific_claim_authorized",
        ):
            self.assertFalse(receipt[key], key)
        self.assertEqual(receipt["parameter_updates"], 0)
        self.assertEqual(receipt["generator_forward_calls"], 0)
        self.assertEqual(
            receipt["fixed_split"],
            {
                "binder_train_ordinals": [0, 1, 3, 4, 8, 9, 10, 14],
                "calibration_ordinals": [5, 12, 13, 15],
                "locked_actual_ordinals": [2, 6, 7, 11],
            },
        )
        self.assertEqual(receipt["design_counts"]["mev_action_cells"], 40)
        self.assertEqual(
            receipt["design_counts"]["source_only_factorial_action_cells"], 36
        )
        self.assertEqual(receipt["design_counts"]["total_action_cells"], 76)
        self.assertEqual(receipt["design_counts"]["total_trajectories"], 320)
        self.assertEqual(
            receipt["design_counts"]["total_transformer_calls"], 28016
        )
        self.assertEqual(
            receipt["design_counts"]["selected_conditional_capture_calls"],
            1596,
        )
        self.assertEqual(receipt["design_counts"]["projected_block_rows"], 6384)
        self.assertTrue(receipt["frozen_base_contract_verified_not_executed"])
        self.assertTrue(
            receipt["source_only_3x3x2x2_factorial_registered_not_executed"]
        )
        self.assertTrue(
            receipt["dual_noun_matched_noop_mean_and_null_registered_not_executed"]
        )
        self.assertTrue(
            receipt["p3_target_teacher_zero_read_firewall_registered_not_executed"]
        )
        p0 = receipt["p0_provisional_evidence"]
        self.assertTrue(p0["integrity_verified"])
        self.assertEqual(p0["candidate_count"], 64)
        self.assertEqual(p0["eligible_count_recorded"], 283)
        self.assertEqual(p0["exact_uuid_path_media_overlap_with_actual"], 0)
        self.assertFalse(p0["official_source_only_registry"])
        self.assertFalse(p0["perceptual_exclusion_complete"])
        self.assertFalse(p0["frozen_observer_qualification_complete"])
        self.assertFalse(p0["p0_slot_pretraining_authorized"])
        self.assertEqual(p0["blocker"], preflight.BLOCKERS[0])
        receipt_hash = receipt.pop("receipt_sha256")
        self.assertEqual(preflight.object_sha256(receipt), receipt_hash)

    def test_run_preflight_verifies_byte_pinned_real_source(self) -> None:
        receipt = preflight.run_preflight()
        self.assertEqual(receipt["status"], preflight.ONLY_STATUS)
        self.assertEqual(
            receipt["source_manifest"]["file_sha256"],
            preflight.EXPECTED_SOURCE_FILE_SHA256,
        )
        self.assertEqual(receipt["source_manifest"]["row_count"], 16)
        self.assertTrue(receipt["source_manifest"]["fixed_split_verified"])

    def test_direct_tamper_fails_canonical_self_hash(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["trajectory_matrix"]["total_trajectories"] = 321
        with self.assertRaisesRegex(
            preflight.V10A2PreflightError, "self hash differs"
        ):
            build(preregistration=preregistration)

    def test_rehashed_split_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["fixed_split"]["binder_train_ordinals"][0] = 2
        self.assert_semantic_tamper_rejected(
            preregistration, "binder-train split differs"
        )

    def test_rehashed_count_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["trajectory_matrix"]["total_transformer_calls"] += 1
        self.assert_semantic_tamper_rejected(
            preregistration, "trajectory count differs"
        )

    def test_rehashed_frozen_base_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["frozen_base"]["capture_calls"] = 1
        self.assert_semantic_tamper_rejected(
            preregistration, "Frozen Base contract differs"
        )

    def test_rehashed_stage_permission_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["training_stages"]["P1"][
            "generator_base_updates_allowed"
        ] = True
        self.assert_semantic_tamper_rejected(
            preregistration, "stage freeze/current authorization differs: P1"
        )

    def test_rehashed_p3_target_read_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["training_stages"]["P3"][
            "target_teacher_read_count_required"
        ] = 1
        self.assert_semantic_tamper_rejected(
            preregistration, "P3 target teacher read count must be zero"
        )

    def test_rehashed_absolute_graph_leakage_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["dynamic_graph"][
            "absolute_centroid_layout_or_scale_allowed"
        ] = True
        self.assert_semantic_tamper_rejected(
            preregistration, "absolute centroid/layout/scale leakage is forbidden"
        )

    def test_rehashed_single_noop_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["trajectory_matrix"]["same_state_null_companions"].pop()
        self.assert_semantic_tamper_rejected(
            preregistration, "dual noun-matched NOOP registry differs"
        )

    def test_rehashed_launch_permission_tamper_fails_semantic_guard(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["current_authorization"]["gpu_launch_authorized"] = True
        self.assert_semantic_tamper_rejected(
            preregistration, "current authorization must remain hard PRE_RUN_NO"
        )

    def test_rehashed_provisional_p0_authority_tamper_fails(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        preregistration["source_only_slot_pretraining"][
            "p0_slot_pretraining_authorized"
        ] = True
        self.assert_semantic_tamper_rejected(
            preregistration,
            "64-source slot-pretraining fail-closed contract differs",
        )

    def test_missing_dependency_cannot_be_rehashed_into_readiness(self) -> None:
        preregistration = load(PREREGISTRATION_PATH)
        dependency = preregistration["dependency_requirements"][0]
        dependency["present"] = True
        dependency["artifact_path"] = "/untrusted/fabricated-registry.json"
        dependency["artifact_sha256"] = "a" * 64
        self.assert_semantic_tamper_rejected(
            preregistration, "dependency must remain explicitly absent"
        )

    def test_source_manifest_whitespace_change_breaks_byte_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            changed_path = Path(temporary_directory) / "source.json"
            changed_path.write_bytes(SOURCE_MANIFEST_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                preflight.V10A2PreflightError, "source manifest bytes differ"
            ):
                preflight.run_preflight(
                    PREREGISTRATION_PATH,
                    changed_path,
                )

    def test_provisional_registry_tamper_blocks_integrated_preflight(self) -> None:
        provisional = load(PROVISIONAL_REGISTRY_PATH)
        provisional["rows"][0]["size_bytes"] += 1
        with self.assertRaisesRegex(
            preflight.V10A2PreflightError,
            "provisional P0 evidence differs: provisional registry self hash differs",
        ):
            build(provisional_registry=provisional)

    def test_provisional_registry_whitespace_change_breaks_byte_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            changed_path = Path(temporary_directory) / "provisional.json"
            changed_path.write_bytes(PROVISIONAL_REGISTRY_PATH.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                preflight.V10A2PreflightError,
                "provisional P0 evidence differs: provisional registry bytes differ",
            ):
                preflight.run_preflight(
                    PREREGISTRATION_PATH,
                    SOURCE_MANIFEST_PATH,
                    changed_path,
                )


if __name__ == "__main__":
    unittest.main()

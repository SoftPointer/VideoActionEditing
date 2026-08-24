from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import target_graph_binder_development_preflight_v4e as preflight


PREREG_PATH = (
    METHOD_ROOT / "assets" / "target_graph_binder_development_prereg_v4e.json"
)
EXCLUSION_PATH = (
    METHOD_ROOT / "assets" / "real_target_graph_pretrain_eval_exclusions_v1.json"
)
MANUAL_PATH = (
    METHOD_ROOT.parents[0]
    / "action_anchor_target_gap_audit"
    / "manual_action_contracts_v2.json"
)
REVIEW_ROOT = (
    METHOD_ROOT.parents[1]
    / "md"
    / "action_editing"
    / "20260822_object_centric_interaction_graph_reward"
)
CATALOG_RECEIPT_PATH = REVIEW_ROOT / "real_target_graph_pretrain_catalog_receipt_v1.json"
TEACHER_ROOT = (
    METHOD_ROOT.parents[1]
    / "md"
    / "action_editing"
    / "20260822_crosscase_target_graph_teacher_sam2_r3_review"
)
TEACHER_CASES = ("8b05aaf463db", "40712e1341dc", "5e83a9279951")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def rehash(value: dict, key: str) -> dict:
    result = copy.deepcopy(value)
    result.pop(key, None)
    result[key] = preflight.object_sha256(result)
    return result


def authorized_catalog_receipt() -> dict:
    receipt = load(CATALOG_RECEIPT_PATH)
    receipt["status"] = preflight.EXPECTED_CATALOG_STATUS
    receipt["authorization_blockers"] = []
    receipt["usage_contract"]["catalog_only"] = False
    receipt["usage_contract"]["graph_teacher_pretraining_authorized"] = True
    receipt["usage_contract"]["trainer_readable_split_emitted"] = True
    receipt["data_status"]["human_qualification_complete"] = True
    receipt["data_status"]["formal_sft_authorized"] = True
    count = receipt["counts"]["catalog_candidate_pending_authority"]
    for kind in preflight.GROUP_KINDS:
        receipt["leakage_audit"]["group_split_audit"]["row_coverage"][kind] = count
    return rehash(receipt, "receipt_sha256")


def qualified_row(index: int, partition: str) -> dict:
    video_uuid = f"00000000-0000-4000-8000-{index + 1000:012d}"
    return {
        "pair_id": sha(f"pair:{index}"),
        "video_uuid": video_uuid,
        "partition": partition,
        "actor_group_id": f"actor-{partition}-{index}",
        "scene_group_id": f"scene-{partition}-{index}",
        "action_group_id": f"action-{partition}-{index}",
        "source_media_sha256": sha(f"source:{index}"),
        "target_media_sha256": sha(f"target:{index}"),
        "perceptual_cluster_id": f"perceptual-{partition}-{index}",
        "source_video_path": f"/qualified/source/{index}.mp4",
        "target_video_path": f"/qualified/target/{index}.mp4",
        "teacher_graph_path": f"/qualified/teacher/{index}.json",
        "teacher_graph_sha256": sha(f"teacher:{index}"),
        "anchor_middle_path": f"/qualified/anchor_middle/{index}.json",
        "anchor_middle_sha256": sha(f"anchor:{index}"),
        "teacher_contract": {
            "teacher_frozen": True,
            "training_authorized": True,
            "canonical_graph_metadata_only": True,
            "raw_masks_exported": False,
            "rgb_latent_flow_exported": False,
            "teacher_embeddings_exported": False,
            "physical_contact_inferred_from_proximity_only": False,
            "uncertainty_and_visibility_exported": True,
            "graph_schema": "qualified-object-interaction-graph-v4e",
        },
        "anchor_middle_contract": {
            "bernini_frozen": True,
            "target_video_accessed": False,
            "target_graph_accessed_during_extraction": False,
            "self_generated_intermediate_only": True,
            "decoded_final_video_used": False,
            "generator_parameter_updates": False,
            "raw_middle_tensor_persisted": False,
            "blocks": [6, 12, 18, 24],
            "published_scope": "reduced_role_slot_edge_statistics_only",
        },
    }


def qualified_manifest() -> dict:
    rows = [qualified_row(index, "train") for index in range(128)]
    rows += [qualified_row(1000 + index, "validation") for index in range(32)]
    manifest = {
        "schema_version": preflight.QUALIFIED_SPLIT_SCHEMA,
        "authorization": {
            "status": preflight.EXPECTED_SPLIT_STATUS,
            "human_qualification_complete": True,
            "formal_sft_authorized": True,
            "graph_teacher_pretraining_authorized": True,
            "generator_training_authorized": False,
            "authority_id": "synthetic-test-authority",
            "sealed_at_utc": "2026-08-23T00:00:00Z",
        },
        "rows": rows,
    }
    return rehash(manifest, "manifest_sha256")


def build(
    *, catalog: dict | None = None, split: dict | None = None, teachers: bool = False
) -> dict:
    teacher_rows = []
    if teachers:
        for case_id in TEACHER_CASES:
            path = TEACHER_ROOT / case_id / "provenance.receipt.json"
            teacher_rows.append((path, load(path)))
    return preflight.build_preflight_receipt(
        preregistration=load(PREREG_PATH),
        catalog_receipt=load(CATALOG_RECEIPT_PATH) if catalog is None else catalog,
        exclusion_registry=load(EXCLUSION_PATH),
        exclusion_registry_sha256=preflight.file_sha256(EXCLUSION_PATH),
        manual_contracts=load(MANUAL_PATH),
        manual_contracts_sha256=preflight.file_sha256(MANUAL_PATH),
        development_teacher_receipts=teacher_rows,
        qualified_split_manifest=split,
    )


class TargetGraphBinderDevelopmentPreflightV4ETests(unittest.TestCase):
    def test_current_real_catalog_fails_closed_without_training(self) -> None:
        receipt = build()
        self.assertEqual(receipt["status"], preflight.BLOCKED_STATUS)
        self.assertFalse(receipt["launch_authorized"])
        self.assertFalse(receipt["training_executed"])
        self.assertFalse(receipt["optimizer_created"])
        self.assertEqual(receipt["parameter_updates"], 0)
        self.assertFalse(receipt["generator"]["loaded"])
        self.assertEqual(receipt["generator"]["forward_calls"], 0)
        self.assertIn(
            "CATALOG_ALL_ROWS_PENDING_HUMAN_QUALIFICATION",
            receipt["authorization_blockers"],
        )
        self.assertIn(
            "CATALOG_INCOMPLETE_ACTOR_GROUP_ID_COVERAGE",
            receipt["authorization_blockers"],
        )
        self.assertIn("QUALIFIED_SPLIT_MANIFEST_MISSING", receipt["authorization_blockers"])

    def test_registered_r3_teachers_are_review_only_warnings(self) -> None:
        receipt = build(teachers=True)
        self.assertEqual(len(receipt["development_teacher_evidence"]), 3)
        self.assertEqual(len(receipt["warnings_not_training_authority"]), 6)
        for row in receipt["development_teacher_evidence"]:
            self.assertEqual(row["optimizer_updates"], 0)
            self.assertEqual(row["renderer_forward_calls"], 0)
            self.assertFalse(row["target_graph_authorized_for_training"])
            self.assertEqual(row["observer_class"], "teacher_observation_scaffold_not_oceg")

    def test_tampered_catalog_receipt_refuses_authority(self) -> None:
        catalog = load(CATALOG_RECEIPT_PATH)
        catalog["counts"]["catalog_rows"] += 1
        with self.assertRaises(preflight.TargetGraphBinderPreflightError):
            build(catalog=catalog)

    def test_fully_qualified_synthetic_authority_can_only_ready_binder_launch(self) -> None:
        receipt = build(catalog=authorized_catalog_receipt(), split=qualified_manifest())
        self.assertEqual(receipt["status"], preflight.READY_STATUS)
        self.assertTrue(receipt["launch_authorized"])
        self.assertEqual(receipt["authorization_blockers"], [])
        self.assertFalse(receipt["training_executed"])
        self.assertFalse(receipt["generator"]["connected_to_binder"])
        self.assertEqual(
            receipt["qualified_split_audit"]["partition_counts"],
            {"train": 128, "validation": 32},
        )

    def test_registered_evaluation_uuid_in_training_is_rejected(self) -> None:
        manifest = qualified_manifest()
        exclusion = load(EXCLUSION_PATH)
        manifest["rows"][0]["video_uuid"] = exclusion["cases"][0]["uuid"]
        manifest = rehash(manifest, "manifest_sha256")
        with self.assertRaisesRegex(
            preflight.TargetGraphBinderPreflightError, "registered evaluation"
        ):
            build(catalog=authorized_catalog_receipt(), split=manifest)

    def test_cross_partition_actor_group_is_rejected(self) -> None:
        manifest = qualified_manifest()
        manifest["rows"][-1]["actor_group_id"] = manifest["rows"][0]["actor_group_id"]
        manifest = rehash(manifest, "manifest_sha256")
        with self.assertRaisesRegex(
            preflight.TargetGraphBinderPreflightError, "cross-partition group leakage"
        ):
            build(catalog=authorized_catalog_receipt(), split=manifest)

    def test_target_video_access_by_anchor_middle_is_rejected(self) -> None:
        manifest = qualified_manifest()
        manifest["rows"][0]["anchor_middle_contract"]["target_video_accessed"] = True
        manifest = rehash(manifest, "manifest_sha256")
        with self.assertRaisesRegex(
            preflight.TargetGraphBinderPreflightError, "target_video_accessed"
        ):
            build(catalog=authorized_catalog_receipt(), split=manifest)

    def test_decoded_anchor_video_feature_is_rejected(self) -> None:
        manifest = qualified_manifest()
        manifest["rows"][0]["anchor_middle_contract"]["decoded_final_video_used"] = True
        manifest = rehash(manifest, "manifest_sha256")
        with self.assertRaisesRegex(
            preflight.TargetGraphBinderPreflightError, "decoded_final_video_used"
        ):
            build(catalog=authorized_catalog_receipt(), split=manifest)

    def test_preregistration_threshold_drift_is_rejected(self) -> None:
        prereg = load(PREREG_PATH)
        prereg["representation_admission"]["cross_appearance_cosine_min"] = 0.5
        prereg = rehash(prereg, "preregistration_sha256")
        with self.assertRaisesRegex(
            preflight.TargetGraphBinderPreflightError, "cosine gate changed"
        ):
            preflight.validate_preregistration(prereg)

    def test_preflight_receipt_output_is_create_only(self) -> None:
        receipt = build()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            preflight._write_json_create_only(path, receipt)
            with self.assertRaisesRegex(
                preflight.TargetGraphBinderPreflightError, "refusing to overwrite"
            ):
                preflight._write_json_create_only(path, receipt)


if __name__ == "__main__":
    unittest.main()

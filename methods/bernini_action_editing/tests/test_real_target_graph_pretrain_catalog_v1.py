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

import real_target_graph_pretrain_catalog_v1 as cataloger


REGISTRY_PATH = (
    METHOD_ROOT / "assets" / "real_target_graph_pretrain_eval_exclusions_v1.json"
)
MANUAL_PATH = (
    METHOD_ROOT.parents[0]
    / "action_anchor_target_gap_audit"
    / "manual_action_contracts_v2.json"
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate(
    *,
    pair_id: str,
    video_uuid: str,
    source_event_id: int,
    target_event_id: int,
    source_filename: str,
    target_filename: str,
    split: str = "test",
    extra: dict | None = None,
) -> dict:
    row = {
        "schema_version": cataloger.CANDIDATE_SCHEMA,
        "pair_id": pair_id,
        "row_id": sha("row:" + pair_id),
        "uuid": video_uuid,
        "mode": cataloger.EXPECTED_MODE,
        "formal_sft_authorized": False,
        "is_strict_counterfactual_ground_truth": False,
        "training_use": cataloger.EXPECTED_TRAINING_USE,
        "instruction_source": cataloger.EXPECTED_INSTRUCTION_SOURCE,
        "instruction_semantic_override_by_qwen_allowed": False,
        "instruction": "Edit the action using the annotation caption.",
        "source_action_caption": "A source event occurs.",
        "target_action_caption": "A target event occurs next.",
        "videos_copied": False,
        "split": split,
        "target": {
            "provenance": cataloger.EXPECTED_TARGET_PROVENANCE,
            "qualification_status": cataloger.EXPECTED_QUALIFICATION,
            "semantic_truth_class": cataloger.EXPECTED_SEMANTIC_TRUTH,
        },
        "automatic_visual_audit": {"verdict": "accept"},
        "source_annotation_provenance": {"uuid": video_uuid},
        "target_annotation_provenance": {"uuid": video_uuid},
        "source_event_annotation": {
            "event_id": source_event_id,
            "end_time": float(source_event_id),
            "filename": source_filename,
        },
        "target_event_annotation": {
            "event_id": target_event_id,
            "start_time": float(source_event_id),
            "filename": target_filename,
        },
        "source_video_path": "/dataset/" + source_filename,
        "target_video_path": "/dataset/" + target_filename,
    }
    if extra:
        row.update(extra)
    return row


def encode_jsonl(rows: list[dict]) -> bytes:
    return b"".join(cataloger.canonical_json_bytes(row) + b"\n" for row in rows)


def fixture(*, adjacent: bool = True, eligible: int = 1) -> tuple[list[dict], dict, dict]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
    rows = [
        candidate(
            pair_id=case["pair_id"],
            video_uuid=case["uuid"],
            source_event_id=case["source_event_id"],
            target_event_id=case["target_event_id"],
            source_filename=case["source_filename"],
            target_filename=case["target_filename"],
        )
        for case in registry["cases"]
    ]
    if adjacent:
        case = next(value for value in registry["cases"] if value["pair_prefix"] == "7077cfb160cf")
        rows.append(
            candidate(
                pair_id=sha("adjacent same evaluation UUID"),
                video_uuid=case["uuid"],
                source_event_id=1,
                target_event_id=2,
                source_filename=case["uuid"] + "-adjacent-source.mp4",
                target_filename=case["uuid"] + "-adjacent-target.mp4",
            )
        )
    for index in range(eligible):
        video_uuid = f"00000000-0000-4000-8000-{index + 1:012d}"
        rows.append(
            candidate(
                pair_id=sha(f"eligible:{index}"),
                video_uuid=video_uuid,
                source_event_id=1,
                target_event_id=2,
                source_filename=video_uuid + "-source.mp4",
                target_filename=video_uuid + "-target.mp4",
                split="train",
            )
        )
    raw = encode_jsonl(rows)
    registry["candidate_manifest"] = {
        "path": "/synthetic/paired_training_candidates.jsonl",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
    }
    return rows, registry, manual


def build(rows: list[dict], registry: dict, manual: dict):
    return cataloger.build_catalog(
        rows,
        candidate_manifest_sha256=registry["candidate_manifest"]["sha256"],
        exclusion_registry=registry,
        exclusion_registry_sha256=cataloger.object_sha256(registry),
        manual_contracts=manual,
        manual_contracts_sha256=cataloger.file_sha256(MANUAL_PATH),
    )


class RealTargetGraphPretrainCatalogTests(unittest.TestCase):
    def test_checked_registry_closes_four_development_plus_twelve_locked(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        manual = json.loads(MANUAL_PATH.read_text(encoding="utf-8"))
        cases, pin = cataloger.validate_exclusion_registry(
            registry,
            manual_contracts=manual,
            manual_contracts_sha256=cataloger.file_sha256(MANUAL_PATH),
        )
        self.assertEqual(len(cases), 16)
        self.assertEqual(pin["row_count"], 3749)
        self.assertEqual(
            sum(case["evaluation_role"] == "development" for case in cases), 4
        )
        self.assertEqual(
            sum(case["evaluation_role"] == "locked_confirmation" for case in cases),
            12,
        )

    def test_entire_uuid_exclusion_removes_named_and_other_adjacent_pairs(self) -> None:
        rows, registry, manual = fixture(adjacent=True, eligible=1)
        catalog, receipt = build(rows, registry, manual)
        statuses = [row["catalog_status"] for row in catalog]
        self.assertEqual(statuses.count("excluded_named_evaluation_pair"), 16)
        self.assertEqual(
            statuses.count("excluded_temporally_adjacent_same_video_uuid"), 1
        )
        self.assertEqual(statuses.count("catalog_candidate_pending_authority"), 1)
        self.assertEqual(receipt["counts"]["excluded_total_by_evaluation_uuid"], 17)
        self.assertEqual(
            receipt["leakage_audit"]["evaluation_uuid_in_candidate_count"], 0
        )

    def test_catalog_is_not_training_or_representation_evidence(self) -> None:
        rows, registry, manual = fixture()
        catalog, receipt = build(rows, registry, manual)
        self.assertEqual(receipt["status"], "CATALOG_AUDITED_TRAIN_BLOCKED")
        self.assertFalse(receipt["usage_contract"]["graph_teacher_pretraining_authorized"])
        self.assertFalse(receipt["usage_contract"]["target_is_generator_input"])
        self.assertFalse(receipt["usage_contract"]["target_gradient_allowed"])
        self.assertFalse(
            receipt["usage_contract"]["target_rgb_latent_flow_regression_allowed"]
        )
        self.assertFalse(
            receipt["claim_limits"]["stable_transferable_action_representation_claimed"]
        )
        for row in catalog:
            contract = row["target_graph_teacher_contract"]
            self.assertFalse(contract["graph_teacher_pretraining_authorized"])
            self.assertFalse(contract["generator_training_authorized"])

    def test_pending_human_and_missing_group_dedup_authority_block_split(self) -> None:
        rows, registry, manual = fixture()
        catalog, receipt = build(rows, registry, manual)
        self.assertIn("ALL_ROWS_PENDING_HUMAN_QUALIFICATION", receipt["authorization_blockers"])
        self.assertIn("FORMAL_SFT_AUTHORIZED_FALSE", receipt["authorization_blockers"])
        self.assertIn(
            "INCOMPLETE_ACTOR_GROUP_ID_COVERAGE", receipt["authorization_blockers"]
        )
        self.assertIn(
            "INCOMPLETE_PERCEPTUAL_CLUSTER_ID_DEDUP_COVERAGE",
            receipt["authorization_blockers"],
        )
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            cataloger.emit_authorized_split_manifest(catalog, receipt)

    def test_missing_locked_case_refuses_catalog(self) -> None:
        rows, registry, manual = fixture()
        rows.pop(0)
        registry["candidate_manifest"]["row_count"] = len(rows)
        registry["candidate_manifest"]["sha256"] = hashlib.sha256(
            encode_jsonl(rows)
        ).hexdigest()
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            build(rows, registry, manual)

    def test_incomplete_exclusion_registry_refuses_catalog(self) -> None:
        rows, registry, manual = fixture()
        registry["cases"].pop()
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            build(rows, registry, manual)

    def test_manual_contract_byte_drift_refuses_catalog(self) -> None:
        rows, registry, manual = fixture()
        changed = copy.deepcopy(manual)
        changed["samples"][0]["human_note"] += " changed"
        changed_sha = cataloger.object_sha256(changed)
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            cataloger.build_catalog(
                rows,
                candidate_manifest_sha256=registry["candidate_manifest"]["sha256"],
                exclusion_registry=registry,
                exclusion_registry_sha256=cataloger.object_sha256(registry),
                manual_contracts=changed,
                manual_contracts_sha256=changed_sha,
            )

    def test_strict_counterfactual_relabelling_is_rejected(self) -> None:
        rows, registry, manual = fixture()
        rows[-1]["is_strict_counterfactual_ground_truth"] = True
        registry["candidate_manifest"]["sha256"] = hashlib.sha256(
            encode_jsonl(rows)
        ).hexdigest()
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            build(rows, registry, manual)

    def test_target_provenance_relabelling_is_rejected(self) -> None:
        rows, registry, manual = fixture()
        rows[-1]["target"]["semantic_truth_class"] = "real-counterfactual"
        registry["candidate_manifest"]["sha256"] = hashlib.sha256(
            encode_jsonl(rows)
        ).hexdigest()
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            build(rows, registry, manual)

    def test_explicit_actor_group_cross_split_is_detected_without_text_inference(self) -> None:
        rows, registry, manual = fixture(eligible=2)
        rows[-2]["actor_group_id"] = "actor-shared"
        rows[-1]["actor_group_id"] = "actor-shared"
        rows[-1]["split"] = "validation"
        registry["candidate_manifest"]["sha256"] = hashlib.sha256(
            encode_jsonl(rows)
        ).hexdigest()
        catalog, receipt = build(rows, registry, manual)
        audit = receipt["leakage_audit"]["group_split_audit"]
        self.assertFalse(audit["all_available_groups_split_disjoint"])
        self.assertEqual(len(audit["cross_split_collisions"]["actor_group_id"]), 1)
        self.assertFalse(audit["free_text_actor_scene_action_grouping_used"])
        with self.assertRaises(cataloger.RealTargetGraphCatalogError):
            cataloger.emit_authorized_split_manifest(catalog, receipt)

    def test_cli_create_only_catalog_and_receipt_then_refuses_split(self) -> None:
        rows, registry, _manual = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            registry_path = root / "registry.json"
            catalog_path = root / "catalog.jsonl"
            receipt_path = root / "receipt.json"
            split_path = root / "train.json"
            candidates.write_bytes(encode_jsonl(rows))
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaises(cataloger.RealTargetGraphCatalogError):
                cataloger.main(
                    [
                        "--candidates",
                        str(candidates),
                        "--exclusion-registry",
                        str(registry_path),
                        "--manual-contracts",
                        str(MANUAL_PATH),
                        "--catalog-output",
                        str(catalog_path),
                        "--receipt-output",
                        str(receipt_path),
                        "--authorized-split-output",
                        str(split_path),
                    ]
                )
            self.assertTrue(catalog_path.is_file())
            self.assertTrue(receipt_path.is_file())
            self.assertFalse(split_path.exists())


if __name__ == "__main__":
    unittest.main()

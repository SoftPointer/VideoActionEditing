#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import train_saic_stage_b_v1 as subject


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> subject.FileSnapshot:
    path.write_bytes(raw)
    return subject.FileSnapshot.capture(path, _sha(raw), label=path.name)


def _stage_a_receipt(adapter_sha: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": subject.source_anchor_trainer.RUN_RECEIPT_SCHEMA,
        "method": subject.source_anchor_trainer.METHOD_NAME,
        "complete": True,
        "run_contract": {
            "mode": "formal",
            "world_size": 8,
            "data_parallel_size": 2,
            "sequence_parallel_size": 4,
            "frame_count": 81,
            "optimizer_updates": subject.source_anchor_trainer.FORMAL_UPDATES,
            "all_train_rows_used_once_as_clean_endpoint": True,
        },
        "manifest": {},
        "native_runtime": {},
        "objective": {},
        "adapter": {
            "checkpoint_published": True,
            "safetensors_roundtrip": {
                "schema_version": subject.source_anchor_trainer.SAFETENSORS_SCHEMA,
                "file_sha256": adapter_sha,
                "state_tensor_sha256": "a" * 64,
                "metadata_closed": True,
                "roundtrip_byte_exact_tensors": True,
            },
        },
        "heldout_gate": {
            "noncompensating_all_pass": True,
            "digest": "b" * 64,
        },
        "scientific_limitations": {
            "future_action_stage_requires_fresh_rollout_nonregression": True,
            "future_action_stage_must_test_action_and_identity_camera_background_separately": True,
        },
        "artifacts": {"adapter.safetensors": adapter_sha},
        "model": {},
        "runtime": {},
        "method_source_revision": "c" * 40,
        "method_source_archive_sha256": "d" * 64,
        "source_anchor_pretext_only": True,
        "action_training": False,
        "semantic_action_editing_success": False,
        "decoded_rgb_appearance_preservation_success": False,
        "source_anchor_checkpoint_publication_authorized": True,
        "action_stage_authorized": False,
        "smoke_incomplete_row_coverage": False,
    }
    value["receipt_digest"] = subject.object_sha256(value)
    return value


def _published_receipt(*, motion_sha: str, anchor_sha: str) -> dict[str, object]:
    contracts = subject.primitive_contracts()
    confirmation = {
        "dog_each_source_two_of_three_four_stage_event": True,
        "human_each_source_two_of_three_four_stage_event": True,
        "all_seven_axes_noninferior_to_frozen_base": True,
        "noop_exact": True,
        "correct_source_beats_wrong_and_drop": True,
        "camera_or_appearance_shortcut_rejected": True,
        "a1_inverse_ranking_beats_a0_on_multiple_unseen_sources": True,
        "all_seeds_reported": True,
    }
    inference = {
        "online_motion_field_recomputed_each_step": True,
        "source_video_and_natural_language_only": True,
        "action_id_used": False,
        "mask_pose_flow_track_trajectory_used": False,
        "training_and_inference_route_identical": True,
    }
    value: dict[str, object] = {
        "schema_version": subject.PUBLISHED_CHECKPOINT_SCHEMA_VERSION,
        "method": subject.METHOD_NAME,
        "complete": True,
        "publication_authorized": True,
        "world_size": 8,
        "data_parallel_size": 2,
        "sequence_parallel_size": 4,
        "frame_count": 81,
        "latent_phases": 21,
        "exact40_steps": 40,
        "rollout_k": 4,
        "outer_cycles_completed": 2,
        "optimizer_update_count": 8,
        "source_anchor_adapter_sha256": anchor_sha,
        "critic_checkpoint_sha256": "1" * 64,
        "critic_qualification_receipt_digest": "2" * 64,
        "motion_adapter_sha256": motion_sha,
        "motion_adapter_state_tensor_sha256": "3" * 64,
        "action_operator_contract_digest": contracts[
            "temporal_operator_contract_digest"
        ],
        "online_motion_contract_digest": contracts["online_motion_contract_digest"],
        "confirmation_gate": confirmation,
        "inference_contract": inference,
        "method_source_revision": "6" * 40,
        "method_source_archive_sha256": "7" * 64,
    }
    value["receipt_digest"] = subject.object_sha256(value)
    return value


class TestSAICStageBTraining(unittest.TestCase):
    def test_parser_has_no_privileged_visual_or_action_id_argument(self) -> None:
        destinations = {
            action.dest for action in subject.build_parser()._actions if action.dest != "help"
        }
        self.assertFalse(destinations & subject.FORBIDDEN_PUBLIC_ARGUMENTS)

    def test_execution_plan_is_exact81_k4_noncompensating(self) -> None:
        plan = subject.execution_plan()
        self.assertEqual(plan["frame_count"], 81)
        self.assertEqual(plan["latent_phases"], 21)
        self.assertEqual(plan["exact40_steps"], 40)
        self.assertEqual(plan["rollout_k_per_source_per_round"], 4)
        self.assertEqual(plan["registered_update_indices"], list(subject.UPDATE_INDICES))
        self.assertEqual(plan["forbidden_exact_base_indices"], [38, 39])
        self.assertEqual(plan["pair_gate"]["preservation_axes"], list(subject.PRESERVATION_AXES))
        self.assertFalse(plan["pair_gate"]["weighted_compensation"])
        self.assertEqual(
            plan["inference_inputs"], ["source_video", "natural_language_instruction"]
        )

    def test_world8_maps_dp2_sp4_and_rejects_other_topology(self) -> None:
        for rank in range(8):
            receipt = subject.validate_world8_environment(
                {"WORLD_SIZE": "8", "RANK": str(rank), "LOCAL_RANK": str(rank)}
            )
            self.assertEqual(receipt["arm_index"], rank // 4)
            self.assertEqual(receipt["sequence_parallel_rank"], rank % 4)
        with self.assertRaises(subject.SAICStageBTrainingError):
            subject.validate_world8_environment(
                {"WORLD_SIZE": "4", "RANK": "0", "LOCAL_RANK": "0"}
            )

    def test_runtime_capability_audit_is_honestly_blocked(self) -> None:
        capabilities = subject.runtime_capability_audit()
        blockers = subject.runtime_blockers(capabilities)
        self.assertTrue(capabilities["online_motion_field_primitive"])
        self.assertTrue(capabilities["hard_seven_axis_pair_gate_primitive"])
        self.assertFalse(capabilities["native_sampler_pre_forward_raw_state_hook"])
        self.assertGreaterEqual(len(blockers), 7)
        self.assertFalse(any("random" in item.lower() for item in blockers))

    def test_stage_a_requires_published_formal_pass_and_exact_adapter_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            adapter = _write(root / "adapter.safetensors", b"stage-a-adapter")
            receipt_value = _stage_a_receipt(adapter.sha256)
            receipt_raw = subject.canonical_json_bytes(receipt_value)
            receipt = _write(root / "receipt.json", receipt_raw)
            summary = subject.validate_stage_a_bundle(adapter=adapter, receipt=receipt)
            self.assertEqual(summary["adapter_sha256"], adapter.sha256)
            self.assertTrue(summary["fresh_stage_b_nonregression_still_required"])

            broken = _stage_a_receipt("0" * 64)
            broken_raw = subject.canonical_json_bytes(broken)
            broken_receipt = _write(root / "broken.json", broken_raw)
            with self.assertRaises(subject.SAICStageBTrainingError):
                subject.validate_stage_a_bundle(adapter=adapter, receipt=broken_receipt)

    def test_published_checkpoint_receipt_is_closed_and_all_gates_are_required(self) -> None:
        motion_sha, anchor_sha = "8" * 64, "9" * 64
        receipt = _published_receipt(motion_sha=motion_sha, anchor_sha=anchor_sha)
        validated = subject.validate_published_checkpoint_receipt(
            receipt,
            motion_adapter_sha256=motion_sha,
            stage_a_adapter_sha256=anchor_sha,
        )
        self.assertTrue(validated["publication_authorized"])
        broken = dict(receipt)
        broken_confirmation = dict(receipt["confirmation_gate"])
        broken_confirmation["noop_exact"] = False
        broken["confirmation_gate"] = broken_confirmation
        broken.pop("receipt_digest")
        broken["receipt_digest"] = subject.object_sha256(broken)
        with self.assertRaises(subject.SAICStageBTrainingError):
            subject.validate_published_checkpoint_receipt(
                broken,
                motion_adapter_sha256=motion_sha,
                stage_a_adapter_sha256=anchor_sha,
            )

    def test_preflight_receipt_never_claims_training_or_creates_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            snapshots = [
                _write(root / f"artifact-{index}.bin", f"value-{index}".encode())
                for index in range(6)
            ]
            artifacts = subject.StageBArtifacts(
                source_manifest=snapshots[0],
                stage_a_adapter=snapshots[1],
                stage_a_receipt=snapshots[2],
                critic_checkpoint=snapshots[3],
                critic_qualification=snapshots[4],
                checkpoint_content_manifest=snapshots[5],
                source_summary={"row_count": 8},
                stage_a_summary={"qualified": True},
                critic_boundary=SimpleNamespace(
                    qualification_receipt_digest="a" * 64
                ),
                checkpoint_identity={"qualified": True},
            )
            receipt = subject.build_preflight_receipt(
                artifacts=artifacts,
                topology={
                    "world_size": 8,
                    "rank": 0,
                    "local_rank": 0,
                    "data_parallel_size": 2,
                    "sequence_parallel_size": 4,
                    "arm_index": 0,
                    "sequence_parallel_rank": 0,
                },
            )
            self.assertTrue(receipt["artifacts_qualified"])
            self.assertFalse(receipt["runtime_complete"])
            self.assertFalse(receipt["optimizer_created"])
            self.assertEqual(receipt["optimizer_updates"], 0)
            self.assertFalse(receipt["training_started"])
            body = dict(receipt)
            digest = body.pop("preflight_digest")
            self.assertEqual(digest, subject.object_sha256(body))

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "fresh-run"
            self.assertEqual(subject.resolve_create_only_output(output), output)
            output.mkdir()
            with self.assertRaises(subject.SAICStageBTrainingError):
                subject.resolve_create_only_output(output)

    def test_source_contains_no_optimizer_or_training_body(self) -> None:
        source = inspect.getsource(subject.main)
        self.assertNotIn("AdamW", source)
        self.assertNotIn("optimizer.step", source)
        self.assertIn("fail-closed", source)


if __name__ == "__main__":
    unittest.main()

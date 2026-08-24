#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_saic_stage_b_v1 as subject
import train_saic_stage_b_v1 as stage_b


def _snapshot(path: Path, raw: bytes) -> stage_b.FileSnapshot:
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return stage_b.FileSnapshot.capture(path, digest, label=path.name)


class TestSAICStageBInference(unittest.TestCase):
    def test_public_parser_has_no_action_id_or_privileged_visual_input(self) -> None:
        destinations = {
            action.dest for action in subject.build_parser()._actions if action.dest != "help"
        }
        self.assertFalse(destinations & subject.FORBIDDEN_ARGUMENTS)
        self.assertIn("source_video", destinations)
        self.assertIn("instruction", destinations)
        self.assertNotIn("noop_instruction", destinations)
        self.assertNotIn("motion_code", destinations)

    def test_instruction_is_natural_language_not_registry_id(self) -> None:
        instruction = (
            "Have the same dog lower its pelvis into a stable sit and hold it."
        )
        self.assertEqual(subject.validate_natural_instruction(instruction), instruction)
        for invalid in ("dog_sit_v1", "sit", " leading text", "text\x00bad"):
            with self.assertRaises(subject.SAICStageBInferenceError):
                subject.validate_natural_instruction(invalid)

    def test_method_owned_noop_is_natural_and_not_a_public_argument(self) -> None:
        self.assertEqual(
            subject.validate_natural_instruction(subject.DEPLOYMENT_NOOP_PROMPT),
            subject.DEPLOYMENT_NOOP_PROMPT,
        )
        parser_destinations = {action.dest for action in subject.build_parser()._actions}
        self.assertNotIn("noop_prompt", parser_destinations)

    def test_inference_requires_world4_sp4(self) -> None:
        for rank in range(4):
            topology = subject.validate_inference_environment(
                {"WORLD_SIZE": "4", "RANK": str(rank), "LOCAL_RANK": str(rank)}
            )
            self.assertEqual(topology["sequence_parallel_rank"], rank)
        with self.assertRaises(subject.SAICStageBInferenceError):
            subject.validate_inference_environment(
                {"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}
            )

    def test_runtime_is_blocked_instead_of_using_offline_motion_code(self) -> None:
        capabilities = subject.runtime_capability_audit()
        blockers = subject.runtime_blockers(capabilities)
        self.assertTrue(capabilities["online_motion_field_primitive"])
        self.assertFalse(capabilities["native_unipc_pre_forward_raw_state_hook"])
        self.assertFalse(
            capabilities[
                "same_step_frozen_t2v_action_noop_then_source_editor_executor"
            ]
        )
        self.assertGreaterEqual(len(blockers), 4)
        self.assertFalse(any("offline" in key for key in capabilities))

    def test_preflight_receipt_is_closed_and_never_claims_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            snapshots = [
                _snapshot(root / f"artifact-{index}.bin", f"data-{index}".encode())
                for index in range(6)
            ]
            instruction = (
                "Have the same person rise to a stable upright stand and hold it."
            )
            receipt = subject.build_preflight_receipt(
                source=snapshots[0],
                source_probe={
                    "frame_count": 81,
                    "fps_numerator": 25,
                    "fps_denominator": 1,
                    "width": 736,
                    "height": 704,
                },
                instruction=instruction,
                stage_a_adapter=snapshots[1],
                stage_a_receipt=snapshots[2],
                stage_b_adapter=snapshots[3],
                stage_b_receipt=snapshots[4],
                published={"receipt_digest": "a" * 64},
                checkpoint_manifest=snapshots[5],
                topology={
                    "world_size": 4,
                    "rank": 0,
                    "local_rank": 0,
                    "sequence_parallel_size": 4,
                    "sequence_parallel_rank": 0,
                },
            )
            self.assertEqual(receipt["forbidden_argument_names_present"], [])
            self.assertFalse(receipt["runtime_complete"])
            self.assertFalse(receipt["model_loaded"])
            self.assertFalse(receipt["sampling_started"])
            self.assertFalse(receipt["output_created"])
            self.assertTrue(
                receipt["sampling_contract"]["online_motion_field_recomputed_each_step"]
            )
            body = dict(receipt)
            digest = body.pop("preflight_digest")
            self.assertEqual(digest, subject.object_sha256(body))

    def test_output_and_sidecar_are_both_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "edit.mp4"
            resolved, receipt = subject.resolve_create_only_media_output(output)
            self.assertEqual(resolved, output)
            self.assertEqual(receipt, root / "edit.mp4.receipt.json")
            receipt.write_text("occupied", encoding="ascii")
            with self.assertRaises(subject.SAICStageBInferenceError):
                subject.resolve_create_only_media_output(output)

    def test_main_contains_no_fallback_sampling_path(self) -> None:
        source = inspect.getsource(subject.main)
        self.assertNotIn("model.sample", source)
        self.assertNotIn("torch.randn", source)
        self.assertIn("fail-closed", source)


if __name__ == "__main__":
    unittest.main()

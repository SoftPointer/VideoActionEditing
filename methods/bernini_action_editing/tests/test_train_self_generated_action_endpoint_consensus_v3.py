from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference
import self_generated_action_endpoint_consensus_v3 as endpoint
import train_lora as legacy
import train_self_generated_action_endpoint_consensus_v3 as trainer


class EndpointConsensusTrainingContractTests(unittest.TestCase):
    def test_checkpoint_receipt_is_inference_compatible_and_truthful(self) -> None:
        targets = inference.expected_lora_target_modules()
        args = argparse.Namespace(
            arm="endpoint_consensus_trust_010",
            max_steps=80,
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            seed=20260817,
            source_manifest_sha256="3" * 64,
        )
        authority = {
            (row, slot): endpoint.EndpointAuthority(
                cell_unit=None,
                consensus_unit=None,
                robust_amplitude=0.01 if row < 2 else 0.02,
                cell_amplitude=0.015,
                peer_consensus_cosine=0.25,
            )
            for row in range(4)
            for slot in range(4)
        }
        receipt = trainer.checkpoint_receipt(
            args=args,
            manifest={"manifest_digest": "4" * 64},
            step=80,
            loss=0.25,
            grad_norm=0.5,
            target_modules=targets,
            trainable_count=123,
            bernini_revision=legacy.BERNINI_OFFICIAL_COMMIT,
            veomni_revision=legacy.VEOMNI_TESTED_COMMIT,
            transformers_version="5.5.4",
            initial_digest="5" * 64,
            teacher_cache_seed=20260817,
            teacher_cache_sha256="6" * 64,
            authority=authority,
        )
        config = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(inference.PEFT_COMPACT_TARGET_MODULES),
            "modules_to_save": None,
            "use_dora": False,
            "use_rslora": False,
        }
        identity = inference.validate_adapter_contract(config, receipt)
        self.assertEqual(identity["global_step"], 80)
        contract = receipt["training_contract"]
        self.assertEqual(contract["objective"], endpoint.SCHEMA)
        self.assertEqual(contract["action_constraint"], "two_sided_endpoint_gain_band")
        self.assertEqual(contract["teacher_representation"]["mode"], "consensus")
        self.assertTrue(contract["full_post_head_velocity_trust"])
        self.assertEqual(contract["rv2v_supervision_target"], "source_video_only")
        self.assertFalse(contract["historical_selected_target_reachable"])


if __name__ == "__main__":
    unittest.main()

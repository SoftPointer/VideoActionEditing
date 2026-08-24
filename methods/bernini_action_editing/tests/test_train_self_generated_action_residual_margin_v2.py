from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as inference
import train_lora as legacy
import train_self_generated_action_residual_margin_v2 as trainer


class ResidualMarginTrainingContractTests(unittest.TestCase):
    def test_sealed_v1_cache_contract_is_validated_locally(self) -> None:
        cells = []
        for row_index in range(4):
            for slot in range(4):
                cells.append(
                    {
                        "row_index": row_index,
                        "slot": slot,
                        "seed": legacy.step_seed(20260817, slot, row_index),
                        "teacher_amplitude": 0.02,
                    }
                )
        cache = {"slots": 4, "seed": 20260817, "cells": cells}
        self.assertEqual(trainer.validate_teacher_cache_seed(cache, 20260817), 20260817)
        observed, by_key = trainer.validate_teacher_cache_cells(
            cache, slots=4, expected_seed=20260817
        )
        self.assertEqual(len(observed), 16)
        self.assertEqual(set(by_key), {(r, s) for r in range(4) for s in range(4)})
        with self.assertRaises(trainer.ResidualTrainingError):
            trainer.validate_teacher_cache_seed(cache, 20260816)

    def test_file_sha_validation_is_self_contained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_bytes(b"manifest")
            expected = "05b3abf2579a5eb66403cd78be557fd860633a1fe2103c7642030defe32c657f"
            self.assertEqual(trainer.validate_file_sha(path, expected, "manifest"), expected)

    def test_cache_requires_positive_teacher_amplitude(self) -> None:
        trainer.validate_residual_cache_cells([{"teacher_amplitude": 0.02}])
        for value in (0.0, -1.0, float("nan"), None, True):
            with self.subTest(value=value), self.assertRaises(
                trainer.ResidualTrainingError
            ):
                trainer.validate_residual_cache_cells(
                    [{"teacher_amplitude": value}]
                )

    def test_checkpoint_receipt_is_accepted_by_strict_inference(self) -> None:
        targets = inference.expected_lora_target_modules()
        args = argparse.Namespace(
            arm="margin_010_perp_100_onset_400",
            max_steps=160,
            method_source_revision="1" * 40,
            method_source_archive_sha256="2" * 64,
            seed=20260817,
            source_manifest_sha256="3" * 64,
        )
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
        self.assertEqual(
            contract["optimized_quantity"],
            "Psi(v_lora_action)-Psi(v_frozen_action)",
        )
        self.assertIs(contract["historical_selected_target_reachable"], False)


if __name__ == "__main__":
    unittest.main()

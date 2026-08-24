from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_delta_lora as inference


def _targets() -> list[str]:
    available = sorted(
        f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
        for block in range(30)
        for attention in (1, 2)
        for projection in ("to_q", "to_k", "to_v", "to_out.0")
    )
    return inference.motion.select_lora_scope(available, "q_out")


def _receipt() -> dict:
    targets = _targets()
    immutable_value = {
        "target_modules": targets,
        "lora_scope": "q_out",
        "checkpoint_tree_sha256": inference.legacy_train.CHECKPOINT_TREE_SHA256,
        "noop_instruction_sha256": "a" * 64,
        "method_source_revision": "b" * 40,
        "method_source_archive_sha256": "c" * 64,
    }
    receipt = {
        "schema_version": inference.LEGACY_TRAINING_RECEIPT_SCHEMA,
        "method": inference.LEGACY_TRAINING_METHOD,
        "global_step": 2,
        "bernini_commit": inference.legacy_train.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": inference.legacy_train.VEOMNI_TESTED_COMMIT,
        "checkpoint": {
            "tree_sha256": inference.legacy_train.CHECKPOINT_TREE_SHA256,
        },
        "adapter": {
            "scope": "q_out",
            "target_modules": targets,
            "target_module_count": len(targets),
            "target_modules_sha256": inference.legacy_train.object_sha256(targets),
        },
        "immutable_contract": {
            "value": immutable_value,
            "digest": inference.legacy_train.object_sha256(immutable_value),
        },
        "supervision": {
            "target_used_as_condition": False,
            "external_mask_track_pose_trajectory": False,
            "unreviewed_full_target_weight": 0.0,
            "shared_source_posterior_mode": True,
            "shared_sigma": True,
            "shared_diffusion_noise": True,
        },
        "distributed": {"ulysses_size": 4},
        "transformers_version": "test",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = inference.legacy_train.object_sha256(receipt)
    return receipt


def _adapter_config() -> dict:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "target_modules": ["to_q", "to_out.0"],
    }


class ReceiptTests(unittest.TestCase):
    class _RuntimeAttentionModel:
        def named_modules(self):
            yield "", self
            for name in sorted(
                f"diff_dec.transformer.blocks.{block}.attn{attention}.{projection}"
                for block in range(30)
                for attention in (1, 2)
                for projection in ("to_q", "to_k", "to_v", "to_out.0")
            ):
                yield name, SimpleNamespace(weight=object())

    def test_q_out_receipt_and_compact_peft_scope_are_accepted(self) -> None:
        identity = inference.validate_training_adapter_contract(
            _adapter_config(), _receipt()
        )
        self.assertEqual(identity["scope"], "q_out")
        self.assertEqual(len(identity["targets"]), 120)

    def test_receipt_tamper_or_target_condition_fails_closed(self) -> None:
        tampered = _receipt()
        tampered["global_step"] = 3
        with self.assertRaises(inference.DeltaInferenceError):
            inference.validate_training_adapter_contract(_adapter_config(), tampered)

        leaked = _receipt()
        leaked["supervision"] = dict(leaked["supervision"])
        leaked["supervision"]["target_used_as_condition"] = True
        leaked.pop("receipt_digest")
        leaked["receipt_digest"] = inference.legacy_train.object_sha256(leaked)
        with self.assertRaises(inference.DeltaInferenceError):
            inference.validate_training_adapter_contract(_adapter_config(), leaked)

    def test_dynamic_expected_state_keys_follow_scope(self) -> None:
        targets = _targets()
        keys = inference.expected_adapter_state_keys(targets)
        self.assertEqual(len(keys), 240)
        self.assertTrue(all(".to_k." not in key and ".to_v." not in key for key in keys))

    def test_legacy_named_scope_cannot_accept_an_arbitrary_exact_subset(self) -> None:
        model = self._RuntimeAttentionModel()
        targets = _targets()
        self.assertEqual(
            inference.validate_runtime_exact_lora_targets(
                model, targets, expected_named_scope="q_out"
            ),
            targets,
        )
        with self.assertRaisesRegex(
            inference.DeltaInferenceError, "differ from.*named LoRA scope"
        ):
            inference.validate_runtime_exact_lora_targets(
                model,
                targets[:-1],
                expected_named_scope="q_out",
            )

    def test_legacy_two_state_sampler_rejects_same_state_v2_adapter(self) -> None:
        receipt = _receipt()
        receipt["schema_version"] = inference.delta_train.RECEIPT_SCHEMA
        receipt["method"] = inference.delta_train.METHOD_NAME
        receipt.pop("receipt_digest")
        receipt["receipt_digest"] = inference.legacy_train.object_sha256(receipt)
        with self.assertRaisesRegex(
            inference.DeltaInferenceError, "receipt schema"
        ):
            inference.validate_training_adapter_contract(
                _adapter_config(), receipt
            )


class _AdapterSlot:
    def __contains__(self, key):
        return key == "default"


class _LoraLayer:
    def __init__(self, scaling=1.0):
        self.scaling = {"default": scaling}
        self.lora_A = _AdapterSlot()
        self.lora_B = _AdapterSlot()


class _FakeModel:
    def __init__(self):
        self.layers = [_LoraLayer(1.0), _LoraLayer(0.5)]

    def named_modules(self):
        yield "", self
        for index, layer in enumerate(self.layers):
            yield str(index), layer


class StrengthTests(unittest.TestCase):
    def test_adapter_scaling_is_linear_and_scope_complete(self) -> None:
        model = _FakeModel()
        self.assertEqual(inference.apply_adapter_strength(model, 1.5), 2)
        self.assertEqual(model.layers[0].scaling["default"], 1.5)
        self.assertEqual(model.layers[1].scaling["default"], 0.75)

    def test_negative_or_nonfinite_strength_fails(self) -> None:
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(
                inference.DeltaInferenceError
            ):
                inference.apply_adapter_strength(_FakeModel(), value)


if __name__ == "__main__":
    unittest.main()

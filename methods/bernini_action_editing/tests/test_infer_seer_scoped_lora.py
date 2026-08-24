from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_lora as original  # noqa: E402
import infer_seer_scoped_lora as seer  # noqa: E402
import train_seer_event_erasure_fm as same_train  # noqa: E402


SHA1 = "1" * 40
SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_fixture_safetensors(path: Path, keys: list[str]) -> None:
    """Write a valid safetensors container without optional Python packages."""

    header = {}
    offset = 0
    for key in keys:
        header[key] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    path.write_bytes(
        len(encoded).to_bytes(8, "little")
        + encoded
        + b"".join(struct.pack("<f", index / 1000.0) for index in range(len(keys)))
    )


def _adapter_config() -> dict[str, object]:
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        # PEFT may compact all exact Q/O module names to these two suffixes.
        "target_modules": ["to_q", "to_out.0"],
    }


def _training_receipt() -> dict[str, object]:
    targets = seer.expected_lora_target_modules()
    receipt: dict[str, object] = {
        "schema_version": seer.TRAINING_RECEIPT_SCHEMA,
        "method": seer.TRAINING_METHOD,
        "global_step": 4,
        "last_preclip_gradient_norm": 0.25,
        "bernini_commit": original.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": original.trainer.VEOMNI_TESTED_COMMIT,
        "bernini_training_files_index_sha256": original.object_sha256(
            original.trainer.BERNINI_PINNED_FILE_HASHES
        ),
        "checkpoint_tree_sha256": original.trainer.CHECKPOINT_TREE_SHA256,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA_B,
        "target_module_count": 60,
        "target_modules_sha256": original.object_sha256(targets),
        "training_contract": {
            "lora_scope": seer.TRAINING_SCOPE,
            "lora_rank": 8,
            "lora_alpha": 8,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "num_frames": 81,
            "latent_frames": 21,
            "transformers_version": "fixture-transformers",
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "explicit_lora_gradient_all_reduce": True,
        },
        "seer": {
            "owner_spec_sha256": SHA_A,
            "self_generated_target_supervision": True,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        },
        "parameter_update_evidence": {
            "initial_trainable_parameter_digest": SHA_A,
            "final_trainable_parameter_digest": SHA_B,
            "exact_parameter_bytes_changed": True,
            "method_success_claimed": False,
        },
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = original.object_sha256(receipt)
    return receipt


def _same_state_training_receipt() -> dict[str, object]:
    targets = seer.expected_lora_target_modules()
    value: dict[str, object] = {
        "method": same_train.METHOD_NAME,
        "branch_state_mode": "shared_noisy_clean_field",
        "exact_same_noisy_query": True,
        "lora_scope": "cross_q_out",
        "target_modules": targets,
        "full_pair_flow_matching_weight": 1.0,
        "same_state_causal_motion_weight": 0.5,
        "same_state_noop_copy_weight": 0.5,
        "training_completion_is_method_success": False,
        "heldout_decoded_review_required": True,
        "expected_seer_manifest_sha256": SHA_A,
        "expected_seer_owner_spec_sha256": "c" * 64,
        "method_source_revision": SHA1,
        "method_source_archive_sha256": SHA_B,
        "seer_row_count": 2,
        "seer_authority": dict(same_train.AUTHORITY),
        "same_generated_video_coordinate": True,
        "event_erasure_source_excludes_transition_and_terminal": True,
        "rejected_cmsg_cross_identity_gate_reused": False,
    }
    immutable = {
        "value": value,
        "digest": original.object_sha256(value),
        "expected_seer_manifest_sha256": SHA_A,
        "expected_seer_owner_spec_sha256": "c" * 64,
        "method_source_archive_sha256": SHA_B,
    }
    receipt: dict[str, object] = {
        "schema_version": same_train.RECEIPT_SCHEMA,
        "method": same_train.METHOD_NAME,
        "global_step": 4,
        "bernini_commit": original.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": original.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint": {
            "path": "/checkpoint",
            "tree_sha256": original.trainer.CHECKPOINT_TREE_SHA256,
        },
        "adapter": {
            "rank": 8,
            "alpha": 8,
            "scope": "cross_q_out",
            "target_module_count": 60,
            "target_modules": targets,
            "target_modules_sha256": original.object_sha256(targets),
            "initialization_digest": SHA_A,
            "checkpoint_parameter_digest": SHA_B,
        },
        "immutable_contract": immutable,
        "supervision": {
            "exact_same_noisy_query": True,
            "self_generated_target_supervision": True,
            "event_erased_source_supervision": True,
            "full_pair_flow_matching_enabled": True,
            "full_pair_flow_matching_weight": 1.0,
            "same_state_causal_motion_weight": 0.5,
            "same_state_noop_copy_weight": 0.5,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "same_pair_all_ranks": True,
            "explicit_lora_gradient_all_reduce": True,
        },
        "last_metrics": {"preclip_gradient_norm": 0.125},
        "parameter_update_evidence": {
            "initial_trainable_parameter_digest": SHA_A,
            "final_trainable_parameter_digest": SHA_B,
            "exact_parameter_bytes_changed": True,
            "engineering_execution_success": True,
            "method_success_claimed": False,
            "final_preclip_gradient_norm": 0.125,
        },
        "seer": {
            "owner_spec_sha256": "c" * 64,
            "dataset_manifest_sha256": SHA_A,
            "row_count": 2,
            "self_generated_target_supervision": True,
            "event_erased_source_supervision": True,
            "training_completion_is_method_success": False,
            "heldout_decoded_review_required": True,
        },
        "transformers_version": "fixture-transformers",
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = original.object_sha256(receipt)
    return receipt


class SeerScopedInferenceContractTests(unittest.TestCase):
    def test_scope_is_exact_60_cross_attention_q_out_modules(self) -> None:
        targets = seer.expected_lora_target_modules()
        self.assertEqual(len(targets), 60)
        self.assertEqual(len(set(targets)), 60)
        self.assertTrue(all(seer._CROSS_Q_OUT.fullmatch(name) for name in targets))
        self.assertFalse(any("attn1" in name or ".to_k" in name for name in targets))
        self.assertEqual(len(seer.expected_adapter_state_keys()), 120)
        # Importing the wrapper must not weaken the historical 240-module CLI.
        self.assertEqual(len(original.expected_lora_target_modules()), 240)
        self.assertEqual(original.EXPECTED_ADAPTER_TENSOR_COUNT, 480)

    def test_valid_real_fixture_safetensors_has_exact_120_factor_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            _write_fixture_safetensors(path, seer.expected_adapter_state_keys())
            checked = seer.validate_scoped_safetensors(path)
        self.assertEqual(checked, seer.expected_adapter_state_keys())
        self.assertEqual(len(checked), 120)

    def test_fixture_with_one_foreign_factor_fails_closed(self) -> None:
        keys = seer.expected_adapter_state_keys()
        keys[-1] = "base_model.model.diff_dec.transformer.blocks.0.attn1.to_q.lora_A.weight"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            _write_fixture_safetensors(path, keys)
            with self.assertRaisesRegex(seer.SeerInferenceError, "120-factor"):
                seer.validate_scoped_safetensors(path)

    def test_b0_receipt_and_compact_peft_scope_are_admitted(self) -> None:
        result = seer.validate_adapter_contract(
            _adapter_config(), _training_receipt()
        )
        self.assertEqual(result["global_step"], 4)
        self.assertEqual(result["seer_owner_spec_sha256"], SHA_A)
        self.assertEqual(
            result["target_modules_sha256"],
            original.object_sha256(seer.expected_lora_target_modules()),
        )

    def test_full_same_state_receipt_uses_same_exact_60_120_loader(self) -> None:
        result = seer.validate_adapter_contract(
            _adapter_config(), _same_state_training_receipt()
        )
        self.assertEqual(result["global_step"], 4)
        self.assertEqual(result["seer_manifest_sha256"], SHA_A)
        self.assertEqual(
            result["target_modules_sha256"],
            original.object_sha256(seer.expected_lora_target_modules()),
        )

    def test_scoped_script_canonical_alias_prevents_duplicate_loader_module(self) -> None:
        source = Path(seer.__file__).read_text(encoding="utf-8")
        self.assertIn(
            'sys.modules.setdefault("infer_seer_scoped_lora", sys.modules[__name__])',
            source,
        )

    def test_full_same_state_manifest_or_delta_mutation_fails_closed(self) -> None:
        receipt = _same_state_training_receipt()
        immutable = dict(receipt["immutable_contract"])
        immutable["expected_seer_manifest_sha256"] = "d" * 64
        receipt["immutable_contract"] = immutable
        receipt["receipt_digest"] = original.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(seer.SeerInferenceError, "immutable"):
            seer.validate_adapter_contract(_adapter_config(), receipt)

        receipt = _same_state_training_receipt()
        adapter = dict(receipt["adapter"])
        adapter["checkpoint_parameter_digest"] = SHA_A
        receipt["adapter"] = adapter
        receipt["receipt_digest"] = original.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(seer.SeerInferenceError, "equal initialization"):
            seer.validate_adapter_contract(_adapter_config(), receipt)

    def test_full_same_state_authority_or_update_crossbind_mutation_fails_closed(self) -> None:
        receipt = _same_state_training_receipt()
        immutable = dict(receipt["immutable_contract"])
        value = dict(immutable["value"])
        value["seer_authority"] = dict(value["seer_authority"])
        value["seer_authority"]["training_completion_is_method_success"] = True
        immutable["value"] = value
        immutable["digest"] = original.object_sha256(value)
        receipt["immutable_contract"] = immutable
        receipt["receipt_digest"] = original.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(seer.SeerInferenceError, "immutable"):
            seer.validate_adapter_contract(_adapter_config(), receipt)

        receipt = _same_state_training_receipt()
        update = dict(receipt["parameter_update_evidence"])
        update["final_trainable_parameter_digest"] = "d" * 64
        receipt["parameter_update_evidence"] = update
        receipt["receipt_digest"] = original.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(seer.SeerInferenceError, "cross-bind"):
            seer.validate_adapter_contract(_adapter_config(), receipt)

    def test_no_update_or_wrong_scope_is_rejected(self) -> None:
        receipt = _training_receipt()
        update = dict(receipt["parameter_update_evidence"])
        update["final_trainable_parameter_digest"] = SHA_A
        receipt["parameter_update_evidence"] = update
        receipt["receipt_digest"] = original.object_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        with self.assertRaisesRegex(seer.SeerInferenceError, "equal initialization"):
            seer.validate_adapter_contract(_adapter_config(), receipt)

        config = _adapter_config()
        config["target_modules"] = ["to_q", "to_k", "to_out.0"]
        with self.assertRaisesRegex(seer.SeerInferenceError, "exceed"):
            seer.validate_adapter_contract(config, _training_receipt())


if __name__ == "__main__":
    unittest.main()

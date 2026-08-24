from __future__ import annotations

import copy

import infer_lora as infer


def adapter_config() -> dict:
    return {
        "target_modules": ["to_k", "to_out.0", "to_q", "to_v"],
        "peft_type": "LORA",
        "r": 256,
        "lora_alpha": 256,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
    }


def receipt() -> dict:
    targets = infer.expected_lora_target_modules()
    value = {
        "schema_version": infer.V4_FULLFIELD_TRAINING_RECEIPT_SCHEMA,
        "global_step": 10,
        "max_steps": 40,
        "last_loss": 0.5,
        "last_preclip_gradient_norm": 1.0,
        "bernini_commit": infer.trainer.BERNINI_OFFICIAL_COMMIT,
        "veomni_commit": infer.trainer.VEOMNI_TESTED_COMMIT,
        "checkpoint_tree_sha256": infer.trainer.CHECKPOINT_TREE_SHA256,
        "bernini_training_files_index_sha256": infer.object_sha256(
            infer.trainer.BERNINI_PINNED_FILE_HASHES
        ),
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": "b" * 64,
        "training_contract": {
            "method": infer.V4_FULLFIELD_METHOD,
            "arm": "fullfield_action_noop",
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "mv2v_flow_shift": infer.FLOW_SHIFT,
            "num_frames": infer.FRAME_COUNT,
            "latent_frames": infer.LATENT_FRAME_COUNT,
            "task_source_name": infer.trainer.TASK_SOURCE_NAME,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "target_embedding_or_caption_conditioning": False,
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "lora_rank": 256,
            "lora_alpha": 256,
            "lora_scope": "all_30_blocks_attn1_attn2_qkvo",
            "gradient_checkpointing": "selective_nonreentrant_stride4",
            "selective_checkpoint_blocks": [0, 4, 8, 12, 16, 20, 24, 28],
            "full_field_shape": "[B,16,21,H,W]",
            "frozen_rv2v_action_target": False,
            "frozen_relative_band_or_trust_radius": False,
            "pooled_or_32d_representation": False,
            "phase0_action_teacher_exact_zero": True,
            "transformers_version": "5.5.4",
        },
        "target_module_count": len(targets),
        "target_modules": targets,
        "target_modules_sha256": infer.object_sha256(targets),
        "trainable_parameter_count": 188_743_680,
        "distributed": {"world_size": 4, "ulysses_size": 4},
        "memory_gate": {
            "passed": True,
            "dummy_or_padding_allocations": False,
            "minimum_reserved_fraction": 0.937,
        },
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    value["receipt_digest"] = infer.object_sha256(value)
    return value


def test_v4_rank256_fullfield_contract_is_accepted() -> None:
    result = infer.validate_adapter_contract(adapter_config(), receipt())
    assert result["v4_fullfield"] is True
    assert len(result["target_modules"]) == 240


def test_v4_rank8_or_low_memory_is_rejected() -> None:
    bad_config = adapter_config()
    bad_config["r"] = 8
    try:
        infer.validate_adapter_contract(bad_config, receipt())
    except infer.InferenceContractError:
        pass
    else:
        raise AssertionError("rank-8 V4 adapter was accepted")

    bad_receipt = copy.deepcopy(receipt())
    bad_receipt["memory_gate"]["minimum_reserved_fraction"] = 0.49
    bad_receipt.pop("receipt_digest")
    bad_receipt["receipt_digest"] = infer.object_sha256(bad_receipt)
    try:
        infer.validate_adapter_contract(adapter_config(), bad_receipt)
    except infer.InferenceContractError:
        pass
    else:
        raise AssertionError("low-memory V4 receipt was accepted")

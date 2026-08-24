#!/usr/bin/env python3
from __future__ import annotations

import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_schedule_block_causal_localization_v1 as subject
import schedule_block_target_row_prompt_swap_v1 as core


def binding(owner: str, schedule: int, *, epsilon: str = "a") -> core.OwnerInputBinding:
    correct = owner == "correct_owner"
    return core.OwnerInputBinding(
        owner=owner,
        schedule_index=schedule,
        timestep=subject.exact40.PINNED_TIMESTEPS[schedule],
        sigma_float32_be_hex=subject.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[schedule],
        orbit_row_digest=core.ORBIT_ROW_DIGEST,
        target_source_full_blob_sha256=core.OWNER_FULL_BLOB_SHA256["correct_owner"],
        owner_full_blob_sha256=core.OWNER_FULL_BLOB_SHA256[owner],
        owner_reference_blob_sha256=core.OWNER_REFERENCE_BLOB_SHA256[owner],
        decoded_target_tensor_sha256="1" * 64,
        decoded_owner_full_tensor_sha256=("1" if correct else "2") * 64,
        decoded_owner_reference_tensor_sha256=(
            tuple(character * 64 for character in ("3", "4", "5", "6"))
            if correct else tuple(character * 64 for character in ("7", "8", "9", "b"))
        ),
        epsilon_sha256=epsilon * 64,
        target_x_s_sha256=hashlib.sha256(f"x-s-{schedule}".encode()).hexdigest(),
        prepared_visual_prefix_sha256=("c" if correct else "d") * 64,
        prepared_prefix_rotary_sha256="e" * 64,
        total_tokens=100,
        condition_tokens=60,
    )


def pair_row(schedule: int, *, epsilon: str = "a") -> dict[str, object]:
    return {
        "correct_owner_binding": dict(binding("correct_owner", schedule, epsilon=epsilon).receipt()),
        "wrong_owner_binding": dict(binding("wrong_owner", schedule, epsilon=epsilon).receipt()),
    }


def digest_record(value: dict[str, object]) -> dict[str, object]:
    return {**value, "digest": subject.object_sha256(value)}


def resign(value: dict[str, object], key: str = "digest") -> None:
    unsigned = dict(value)
    unsigned.pop(key, None)
    value[key] = subject.object_sha256(unsigned)


def text_runtime_fixture() -> dict[str, object]:
    branches: dict[str, object] = {}
    for ordinal, branch in enumerate(core.TEXT_BRANCHES):
        base: dict[str, object] = {
            "branch": branch,
            "prompt_sha256": subject.PROMPT_SHA256[branch],
            "input_ids_sha256": hashlib.sha256(f"ids-{branch}".encode()).hexdigest(),
            "attention_mask_sha256": hashlib.sha256(f"mask-{branch}".encode()).hexdigest(),
            "t5_input_lens_sha256": hashlib.sha256(f"lens-{branch}".encode()).hexdigest(),
            "text_lens": [512],
            "embedding_sha256": hashlib.sha256(f"embed-{branch}".encode()).hexdigest(),
            "input_ids_shape": [1, 10 + ordinal],
            "attention_mask_shape": [1, 10 + ordinal],
            "t5_input_lens_shape": [1, 1],
            "embedding_shape": [1, 512, 4096],
            "embedding_dtype": "torch.bfloat16",
            "embedding_device_type": "cuda",
            "encoded_call_ordinal": ordinal,
        }
        runtime_digest = subject.object_sha256(base)
        branches[branch] = digest_record({
            **base,
            "runtime_tensor_binding_digest": runtime_digest,
            "world4_rank_binding_digests": [runtime_digest] * 4,
            "world4_consensus": True,
        })
    return digest_record({
        "branches": branches,
        "all_encoded_once_before_forward": True,
    })


def expanded_pair_row(schedule: int, *, epsilon: str = "a") -> dict[str, object]:
    correct = binding("correct_owner", schedule, epsilon=epsilon)
    wrong = binding("wrong_owner", schedule, epsilon=epsilon)
    pair_validation = dict(core.validate_owner_pair_bindings(correct, wrong))
    tensor_bundle = {
        "decoded_target_tensor_sha256": correct.decoded_target_tensor_sha256,
        "decoded_correct_owner_full_tensor_sha256": correct.decoded_owner_full_tensor_sha256,
        "decoded_correct_owner_reference_tensor_sha256": list(correct.decoded_owner_reference_tensor_sha256),
        "decoded_wrong_owner_full_tensor_sha256": wrong.decoded_owner_full_tensor_sha256,
        "decoded_wrong_owner_reference_tensor_sha256": list(wrong.decoded_owner_reference_tensor_sha256),
        "epsilon_sha256": correct.epsilon_sha256,
        "target_x_s_sha256": correct.target_x_s_sha256,
        "correct_prepared_visual_prefix_sha256": correct.prepared_visual_prefix_sha256,
        "wrong_prepared_visual_prefix_sha256": wrong.prepared_visual_prefix_sha256,
        "prepared_prefix_rotary_sha256": correct.prepared_prefix_rotary_sha256,
    }
    geometry = {
        "orbit_member_order": ["V0", "V1", "V2"],
        "owner_aliases": {"correct_owner": "V0/source", "wrong_owner": "V1/variant_a"},
        "full_latent_shape": list(subject.EXPECTED_LATENT_SHAPE),
        "reference_latent_shape": [1, 16, 1, 74, 50],
        "reference_rgb_indices": [0, 27, 53, 80],
        "condition_components": [
            "owner_full_21", "owner_ref0_1", "owner_ref27_1",
            "owner_ref53_1", "owner_ref80_1",
        ],
        "target_component": "unchanged_source_x_s_21",
        "source_ids": list(core.NATIVE_SOURCE_IDS),
        "concat_order": ["video", "ref0", "ref1", "ref2", "ref3", "target"],
        "total_tokens": correct.total_tokens,
        "condition_tokens": correct.condition_tokens,
        "target_tokens": correct.total_tokens - correct.condition_tokens,
        "target_is_strict_suffix": True,
        "sp4_layout_receipts": [
            dict(core.NativeTargetSuffixLayout(
                correct.total_tokens, correct.condition_tokens, rank, 4
            ).receipt())
            for rank in range(4)
        ],
        "append_false_then_contiguous_rank_chunks": True,
    }
    return digest_record({
        "schedule_index": schedule,
        "correct_owner_binding": dict(correct.receipt()),
        "wrong_owner_binding": dict(wrong.receipt()),
        "pair_validation": pair_validation,
        "actual_object_hashes_recomputed": True,
        "world4_consensus_before_forward": True,
        "actual_tensor_bundle": tensor_bundle,
        "actual_tensor_bundle_digest": subject.object_sha256(tensor_bundle),
        "world4_actual_tensor_bundle_rank_digests": [subject.object_sha256(tensor_bundle)] * 4,
        "pack_geometry": geometry,
        "pack_geometry_digest": subject.object_sha256(geometry),
        "x_s_construction": {
            "function": "source_noised_ladder_v1.shared_noise_source_state",
            "formula": "(1-sigma_float32_authority)*decoded_V0+sigma_float32_authority*epsilon",
            "schedule_index": schedule,
            "sigma_float32_be_hex": correct.sigma_float32_be_hex,
            "clean_sha256": correct.decoded_target_tensor_sha256,
            "epsilon_sha256": correct.epsilon_sha256,
            "x_s_sha256": correct.target_x_s_sha256,
            "actual_recomputed": True,
        },
    })


def c0_processor_audits(pair: dict[str, object]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for descriptor in subject._expected_c0_audit_descriptors():
        owner = str(descriptor["owner"])
        owner_binding = pair[f"{owner}_binding"]
        assert isinstance(owner_binding, dict)
        core_binding = subject._binding_from_receipt(owner_binding, label="fixture binding")
        results.append(digest_record({
            "execution_id": descriptor["execution_id"],
            "owner": owner,
            "schedule_index": descriptor["schedule"],
            "branch": descriptor["branch"],
            "phase": descriptor["phase"],
            "band_name": descriptor["band"],
            "selected_blocks": list(core.band_blocks(str(descriptor["band"]))),
            "installed_block_indices": list(range(core.TOTAL_BLOCKS)),
            "owner_input_binding_digest": core_binding.digest,
            "per_block_counter_deltas": [
                dict(item) for item in subject.expected_processor_deltas(
                    str(descriptor["phase"]), str(descriptor["band"])
                )
            ],
            "exact_hook_counts": True,
            "all30_processor_inventory": True,
        }))
    return results


def full_processor_audits(
    pairs: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for descriptor in subject._expected_full_audit_descriptors():
        owner = str(descriptor["owner"])
        schedule = int(descriptor["schedule"])
        owner_binding = pairs[schedule][f"{owner}_binding"]
        assert isinstance(owner_binding, dict)
        core_binding = subject._binding_from_receipt(
            owner_binding, label="full fixture binding"
        )
        results.append(digest_record({
            "execution_id": descriptor["execution_id"],
            "owner": owner,
            "schedule_index": schedule,
            "branch": descriptor["branch"],
            "phase": descriptor["phase"],
            "band_name": descriptor["band"],
            "selected_blocks": list(core.band_blocks(str(descriptor["band"]))),
            "installed_block_indices": list(range(core.TOTAL_BLOCKS)),
            "owner_input_binding_digest": core_binding.digest,
            "per_block_counter_deltas": [
                dict(item) for item in subject.expected_processor_deltas(
                    str(descriptor["phase"]), str(descriptor["band"])
                )
            ],
            "exact_hook_counts": True,
            "all30_processor_inventory": True,
        }))
    return results


def cache_audits_fixture(
    *, profile: str, pairs: dict[int, dict[str, object]],
    audits: list[dict[str, object]],
    text_runtime: dict[str, object],
) -> list[dict[str, object]]:
    processor = {str(row["execution_id"]): row for row in audits}
    branches = text_runtime["branches"]
    assert isinstance(branches, dict)
    results: list[dict[str, object]] = []
    for spec in subject._expected_cache_specs(profile):
        owner = str(spec["owner"])
        schedule = int(spec["schedule_index"])
        pair = pairs[schedule]
        owner_binding = pair[f"{owner}_binding"]
        assert isinstance(owner_binding, dict)
        binding_value = subject._binding_from_receipt(owner_binding, label="cache fixture")
        prompt = branches[str(spec["branch"])]
        assert isinstance(prompt, dict)
        shapes = {str(block): [1, 512, 4096] for block in range(core.TOTAL_BLOCKS)}
        identities: dict[str, object] = {}
        for block in range(core.TOTAL_BLOCKS):
            identities[str(block)] = digest_record({
                "shape": [1, 512, 4096],
                "dtype": "torch.bfloat16",
                "raw_sha256": hashlib.sha256(
                    f"{spec['execution_id']}-{block}".encode()
                ).hexdigest(),
                "tensor_version": 0,
            })
        cache_receipt = digest_record({
            "branch": spec["branch"],
            "expected_block_indices": list(range(core.TOTAL_BLOCKS)),
            "captured_block_indices": list(range(core.TOTAL_BLOCKS)),
            "captured_shapes": shapes,
            "captured_content_identity_by_block": identities,
            "sealed": True,
            "capturing": False,
            "capture_aborted": False,
            "reuse_count": spec["expected_reuse_count"],
            "captured_hidden_or_output_reused": False,
            "captured_text_encoder_state_only": True,
        })
        terminal = digest_record({
            "branch": spec["branch"],
            "block_identity_digest": subject.object_sha256(identities),
            "all_30_content_and_versions_unchanged": True,
        })
        prediction = hashlib.sha256(str(spec["execution_id"]).encode()).hexdigest()
        rng_rank = [
            hashlib.sha256(
                f"rng-{spec['execution_id']}-rank-{rank}".encode()
            ).hexdigest()
            for rank in range(4)
        ]
        rng = subject._rng_world4_vector_sha256(
            rng_rank, label="fixture capture RNG"
        )
        results.append(digest_record({
            "cache_instance_ordinal": spec["cache_instance_ordinal"],
            "execution_id": spec["execution_id"],
            "owner": owner,
            "schedule_index": spec["schedule_index"],
            "branch": spec["branch"],
            "owner_input_binding_digest": binding_value.digest,
            "capture_processor_audit_digest": processor[str(spec["execution_id"])]["digest"],
            "capture_prediction_sha256": prediction,
            "capture_prediction_world4_consensus": True,
            "capture_prediction_rank_sha256": [prediction] * 4,
            "capture_prompt_runtime_binding_digest": prompt["digest"],
            "capture_prompt_embedding_sha256": prompt["embedding_sha256"],
            "capture_prediction_discarded": True,
            "capture_decode_performed": False,
            "capture_scheduler_steps": 0,
            "rng_state_before_sha256": rng,
            "rng_state_after_sha256": rng,
            "rng_state_before_world4_rank_sha256": rng_rank,
            "rng_state_after_world4_rank_sha256": list(rng_rank),
            "rng_state_unchanged": True,
            "mixed_execution_ids": spec["mixed_execution_ids"],
            "expected_reuse_count": spec["expected_reuse_count"],
            "cache_receipt": cache_receipt,
            "terminal_cache_audit": terminal,
            "fresh_for_exact_owner_schedule_branch": True,
            "cache_object_identity_unique_within_phase": True,
        }))
    return results


def c0_cache_audits(
    pair: dict[str, object], audits: list[dict[str, object]],
    text_runtime: dict[str, object],
) -> list[dict[str, object]]:
    return cache_audits_fixture(
        profile="c0", pairs={core.C0_SCHEDULE_INDEX: pair},
        audits=audits, text_runtime=text_runtime,
    )


def c0_noop_parity(
    pair: dict[str, object], audits: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["execution_id"]): row for row in audits}
    rows: list[dict[str, object]] = []
    cells = core.c0_smoke_cells()
    for owner in core.OWNERS:
        plain_id = next(
            cell.cell_id for cell in cells
            if cell.owner == owner and cell.role == "noop_baseline"
        )
        swap_id = subject._noop_swap_id(owner)
        owner_binding = pair[f"{owner}_binding"]
        assert isinstance(owner_binding, dict)
        binding_value = subject._binding_from_receipt(owner_binding, label="parity fixture")
        velocity = hashlib.sha256(f"velocity-{owner}".encode()).hexdigest()
        predecode = hashlib.sha256(f"predecode-{owner}".encode()).hexdigest()
        rows.append(digest_record({
            "owner": owner,
            "schedule_index": core.C0_SCHEDULE_INDEX,
            "band_name": core.C0_BAND,
            "plain_execution_id": plain_id,
            "swap_execution_id": swap_id,
            "owner_input_binding_digest": binding_value.digest,
            "plain_processor_audit_digest": by_id[plain_id]["digest"],
            "swap_processor_audit_digest": by_id[swap_id]["digest"],
            "plain_vs_swap_velocity_raw_bytes_equal": True,
            "plain_vs_swap_predecode_raw_bytes_equal": True,
            "plain_velocity_sha256": velocity,
            "swap_velocity_sha256": velocity,
            "plain_predecode_sha256": predecode,
            "swap_predecode_sha256": predecode,
            "internal_noop_swap_decoded": False,
            "full_velocity_compared": True,
            "full_predecode_compared": True,
        }))
    return rows


def processor_patch_fixture(audits: list[dict[str, object]]) -> dict[str, object]:
    owner_digests = sorted({str(row["owner_input_binding_digest"]) for row in audits})
    statistics: list[dict[str, object]] = []
    for block in range(core.TOTAL_BLOCKS):
        counter_values = {
            counter: sum(
                int(row["per_block_counter_deltas"][block][counter])  # type: ignore[index]
                for row in audits
            )
            for counter in subject._COUNTERS
        }
        statistics.append({
            "block_index": block,
            **counter_values,
            "owner_input_binding_digests": owner_digests,
        })
    return digest_record({
        "schema_version": core.SCHEMA_VERSION,
        "installed_block_indices": list(range(core.TOTAL_BLOCKS)),
        "installed_projection": "blocks.{0..29}.attn2.processor",
        "base_processor_reused_for_noop_and_branch": True,
        "same_current_hidden_states_for_both_selected_calls": True,
        "non_target_rows_select_noop_output": True,
        "unselected_blocks_single_official_call": True,
        "capture_text_condition_only": True,
        "capture_hidden_or_output_reused": False,
        "owner_binding_required_by_every_context": True,
        "installation_and_restore_transactional": True,
        "inference_no_grad_required": True,
        "attention_parameters_frozen_and_grad_free_required": True,
        "optimizer_present": False,
        "parameter_update_authorized": False,
        "restored": True,
        "statistics": statistics,
    })


def input_snapshot_fixture(
    *, pairs: dict[int, dict[str, object]], text_runtime: dict[str, object],
) -> dict[str, object]:
    branches = text_runtime["branches"]
    assert isinstance(branches, dict)
    prompt_hashes = {
        branch: branches[branch]["embedding_sha256"]  # type: ignore[index]
        for branch in core.TEXT_BRANCHES
    }
    schedule_inputs: dict[str, object] = {}
    epsilon_sha256: str | None = None
    for schedule in core.policy.REGISTERED_SCHEDULE_INDICES:
        if schedule not in pairs:
            continue
        pair = pairs[schedule]
        owners: dict[str, object] = {}
        x_s_sha256: str | None = None
        for owner in core.OWNERS:
            raw_binding = pair[f"{owner}_binding"]
            assert isinstance(raw_binding, dict)
            value = subject._binding_from_receipt(
                raw_binding, label=f"input snapshot fixture s{schedule} {owner}"
            )
            if epsilon_sha256 is None:
                epsilon_sha256 = value.epsilon_sha256
            if x_s_sha256 is None:
                x_s_sha256 = value.target_x_s_sha256
            owners[owner] = {
                "owner_input_binding_digest": value.digest,
                "packed_latents_sha256": hashlib.sha256(
                    f"packed-latents-s{schedule}-{owner}".encode()
                ).hexdigest(),
                "packed_rotary_sha256": hashlib.sha256(
                    f"packed-rotary-s{schedule}-{owner}".encode()
                ).hexdigest(),
                "source_ids": list(value.source_ids),
                "prepared_visual_prefix_sha256": value.prepared_visual_prefix_sha256,
                "prepared_prefix_rotary_sha256": value.prepared_prefix_rotary_sha256,
            }
        assert x_s_sha256 is not None
        schedule_inputs[str(schedule)] = {
            "x_s_sha256": x_s_sha256, "owners": owners,
        }
    assert epsilon_sha256 is not None
    return digest_record({
        "epsilon_sha256": epsilon_sha256,
        "prompt_embedding_sha256_by_branch": prompt_hashes,
        "schedule_inputs": schedule_inputs,
    })


def smoke_receipt_fixture() -> dict[str, object]:
    pair = expanded_pair_row(core.C0_SCHEDULE_INDEX)
    text_runtime = text_runtime_fixture()
    audits = c0_processor_audits(pair)
    caches = c0_cache_audits(pair, audits, text_runtime)
    parity = c0_noop_parity(pair, audits)
    branches = text_runtime["branches"]
    assert isinstance(branches, dict)
    noop = branches["noop"]
    assert isinstance(noop, dict)
    audit_by_id = {str(row["execution_id"]): row for row in audits}
    cache_by_id = {str(row["execution_id"]): row for row in caches}
    outputs: list[dict[str, object]] = []
    artifacts: dict[str, str] = {
        "c0-plan.json": hashlib.sha256(b"c0-plan").hexdigest(),
    }
    pair_validation = pair["pair_validation"]
    assert isinstance(pair_validation, dict)
    for cell in core.c0_smoke_cells():
        name = f"c0/{cell.output_name}"
        owner_binding = pair[f"{cell.owner}_binding"]
        assert isinstance(owner_binding, dict)
        binding_value = subject._binding_from_receipt(owner_binding, label="output fixture")
        velocity = hashlib.sha256(f"velocity-{cell.cell_id}".encode()).hexdigest()
        predecode = hashlib.sha256(f"x0-{cell.cell_id}".encode()).hexdigest()
        media = hashlib.sha256(f"media-{cell.cell_id}".encode()).hexdigest()
        artifacts[name] = media
        cache_digest = None
        if cell.role != "noop_baseline":
            cache_digest = cache_by_id[
                subject._capture_id("c0", cell.owner, cell.schedule_index, cell.branch)
            ]["digest"]
        outputs.append({
            "name": name,
            "phase": "c0",
            "cell": dict(cell.receipt()),
            "cell_digest": subject.object_sha256(cell.receipt()),
            "owner_input_binding": owner_binding,
            "owner_input_binding_digest": binding_value.digest,
            "owner_pair_validation_digest": pair_validation["digest"],
            "velocity_sha256": velocity,
            "predecode_x0_hat_sha256": predecode,
            "sha256": media,
            "frames": subject.EXPECTED_FRAMES,
            "fps": float(subject.FPS),
            "hw": list(subject.EXPECTED_HW),
            "decode_input_latent_sha256": predecode,
            "decode_input_latent_shape": list(subject.EXPECTED_LATENT_SHAPE),
            "vae_frozen_eval": True,
            "actual_object_binding_used_for_forward": True,
            "global_prompt_branch": "noop",
            "global_noop_prompt_runtime_binding_digest": noop["digest"],
            "global_noop_embedding_sha256": noop["embedding_sha256"],
            "processor_audit_digest": audit_by_id[cell.cell_id]["digest"],
            "branch_capture_cache_audit_digest": cache_digest,
            "result_rank_sha256": [velocity] * 4,
            "result_world4_consensus": True,
            "decode_input_dtype": "torch.float32",
            "decode_input_device_type": "cuda",
            "decode_input_contiguous": True,
            "decode_input_finite": True,
        })
    integrity = {
        "certificate_schema": "torch-module-parameters-buffers-raw-sha256-v1",
        "pre_sha256": "a" * 64,
        "post_c0_sha256": "a" * 64,
        "post_sha256": "a" * 64,
        "parameter_tensors": 1,
        "buffer_tensors": 0,
        "bytes_unchanged": True,
        "all_parameters_frozen": True,
        "all_parameter_gradients_absent": True,
    }
    input_pre_c0 = input_snapshot_fixture(
        pairs={core.C0_SCHEDULE_INDEX: pair}, text_runtime=text_runtime,
    )
    input_invariants = dict(subject.build_input_invariant_receipt(
        profile="smoke-only", pre_c0=input_pre_c0,
        post_c0=dict(input_pre_c0), post_full=None,
        terminal=dict(input_pre_c0),
    ))
    input_bundle = {
        "model_sha256": integrity["pre_sha256"],
        "text_runtime_digest": text_runtime["digest"],
        "epsilon_sha256": binding("correct_owner", 29).epsilon_sha256,
        "orbit_row_digest": core.ORBIT_ROW_DIGEST,
        "owner_pair_digests": [pair["digest"]],
        "c0_plan_digest": core.build_plan("c0-smoke")["plan_digest"],
        "full_plan_digest": None,
        "actual_input_pre_c0_snapshot_digest": input_pre_c0["digest"],
    }
    first_cache = caches[0]
    first_unsigned = {
        "world_size": 4,
        "passed": True,
        "before_any_real_forward": True,
        "actual_model_text_input_tensor_hashes": True,
        "input_bundle_digest": subject.object_sha256(input_bundle),
        "input_rank_digests": [subject.object_sha256(input_bundle)] * 4,
        "first_execution_id": first_cache["execution_id"],
        "first_output_sha256": first_cache["capture_prediction_sha256"],
        "first_output_rank_digests": first_cache["capture_prediction_rank_sha256"],
        "first_output_world4_consensus": True,
    }
    first = digest_record(first_unsigned)
    c0_no_update = {
        "gradient_enabled": False,
        "optimizer_present": False,
        "scheduler_present": False,
        "scheduler_steps": 0,
        "parameter_gradients_present": False,
        "parameter_updates": 0,
    }
    c0 = digest_record({
        "schema_version": subject.C0_GATE_SCHEMA,
        "engineering_pass": True,
        "scientific_pass_claimed": False,
        "visual_selection_performed": False,
        "decoded_output_count": 6,
        "internal_noop_parity_decoded_output_count": 0,
        "noop_parity": parity,
        "processor_audits": audits,
        "cache_audits": caches,
        "first_forward_consensus": first,
        "model_integrity": {
            "pre_c0_sha256": "a" * 64,
            "post_c0_sha256": "a" * 64,
            "bytes_unchanged": True,
        },
        "no_update": c0_no_update,
        "decoded_output_names": [row["name"] for row in outputs],
        "decoded_output_record_digests": [subject.object_sha256(row) for row in outputs],
        "media_complete": True,
    })
    source_identity = {
        "root": "/sealed/source",
        "parquet_sha256": subject.SOURCE_DATASET_PARQUET_SHA256,
        "receipt_sha256": subject.SOURCE_DATASET_RECEIPT_SHA256,
        "receipt_digest": subject.SOURCE_DATASET_RECEIPT_DIGEST,
        "materialization_spec_sha256": subject.SOURCE_DATASET_SPEC_SHA256,
        "materialization_spec_digest": subject.SOURCE_DATASET_SPEC_DIGEST,
        "iid": subject.IID,
        "source_video_sha256": subject.SOURCE_VIDEO_SHA256,
        "row_digest": "1" * 64,
        "projected_columns": ["iid", "source_video_sha256", "row_digest"],
        "posterior_blob_columns_read": [],
        "latents_consumed": False,
    }
    tensor_map = {
        "V0.video": binding("correct_owner", 29).decoded_owner_full_tensor_sha256,
        "V1.video": binding("wrong_owner", 29).decoded_owner_full_tensor_sha256,
        "V2.video": "f" * 64,
    }
    for owner_name, prefix in (("correct_owner", "V0"), ("wrong_owner", "V1")):
        for index, value in zip(
            (0, 27, 53, 80), binding(owner_name, 29).decoded_owner_reference_tensor_sha256
        ):
            tensor_map[f"{prefix}.ref{index}"] = value
    for index, character in zip((0, 27, 53, 80), ("0", "a", "b", "c")):
        tensor_map[f"V2.ref{index}"] = character * 64
    vae_validation = digest_record({
        "dataset_vae_identity_digest": "e" * 64,
        "training_checkpoint_root": "/sealed/checkpoint",
        "vae_files": {"vae/config.json": "f" * 64},
        "all_offline_encoder_files_rehashed_before_training": True,
    })
    orbit_base = {
        "root": "/sealed/orbit",
        "parquet_sha256": subject.ORBIT_DATASET_PARQUET_SHA256,
        "receipt_sha256": subject.ORBIT_DATASET_RECEIPT_SHA256,
        "receipt_digest": subject.ORBIT_DATASET_RECEIPT_DIGEST,
        "materialization_spec_sha256": subject.ORBIT_DATASET_SPEC_SHA256,
        "materialization_spec_digest": subject.ORBIT_DATASET_SPEC_DIGEST,
        "reference_encoding_contract_digest": subject.ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST,
        "iid": subject.IID,
        "row_digest": core.ORBIT_ROW_DIGEST,
        "pinned_vae_identity_digest": "e" * 64,
        "all_target_and_owner_latents_from_orbit_row": True,
    }
    orbit_identity = {
        **orbit_base,
        "vae_runtime_validation": vae_validation,
        "orbit_tensor_broadcast": {
            "source_rank": 0,
            "tensor_digest": subject.object_sha256(tensor_map),
            "tensor_sha256_by_member": tensor_map,
            "world4_rank_digests": [subject.object_sha256(tensor_map)] * 4,
            "world4_consensus": True,
        },
    }
    topology = digest_record({
        "path": "/sealed/topology",
        "world_size": 4,
        "empty_on_every_rank": True,
        "collective_before_output_reservation": True,
    })
    distributed = digest_record({
        "world_size": 4,
        "local_world_size": 2,
        "nodes": 2,
        "ranks_per_node": 2,
        "ulysses_sp_size": 4,
        "sp4_crosses_nodes": True,
        "rank_hostname_local_rank": [
            {"rank": 0, "local_rank": 0, "hostname": "host-a"},
            {"rank": 1, "local_rank": 1, "hostname": "host-a"},
            {"rank": 2, "local_rank": 0, "hostname": "host-b"},
            {"rank": 3, "local_rank": 1, "hostname": "host-b"},
        ],
        "topology_admission": topology,
        "topology_admitted_collectively_before_output": True,
    })
    checkpoint_audit = {
        "manifest_path": "/sealed/checkpoint-manifest.json",
        "manifest_sha256": subject.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
        "verified_file_count": 23,
        "every_non_cache_file_sha256_verified": True,
        "verified_entries_digest": "b" * 64,
    }
    terminal = digest_record({
        "source_self_identity_digest": subject.object_sha256(source_identity),
        "orbit_identity_digest": subject.object_sha256(orbit_base),
        "source_video_sha256": subject.SOURCE_VIDEO_SHA256,
        "checkpoint_content_audit_digest": subject.object_sha256(checkpoint_audit),
        "all_live_authorities_reopened_and_stable": True,
    })
    unsigned: dict[str, object] = {
        "schema_version": subject.RECEIPT_SCHEMA,
        "complete": True,
        "profile": "smoke-only",
        "model_load_count": 1,
        "one_process_one_model_load": True,
        "same_load_c0_then_full": False,
        "seed": subject.SEED,
        "datasets": {
            "source_self_cross_authority": source_identity,
            "orbit_model_inputs": orbit_identity,
        },
        "source_video": {
            "path": "/sealed/source.mp4",
            "sha256": subject.SOURCE_VIDEO_SHA256,
            "cross_authority_only": True,
            "model_condition_consumed": False,
        },
        "prompt_authority": dict(subject.load_prompt_authority()),
        "text_runtime": text_runtime,
        "orbit_review_authority": dict(subject.load_orbit_review_authority()),
        "model": {
            "bernini_root": "/sealed/bernini",
            "veomni_root": "/sealed/veomni",
            "checkpoint": "/sealed/checkpoint",
            "checkpoint_content_manifest": "/sealed/checkpoint-manifest.json",
            "bernini_commit": subject.EXPECTED_BERNINI_COMMIT,
            "veomni_commit": subject.EXPECTED_VEOMNI_COMMIT,
            "checkpoint_tree_sha256": subject.EXPECTED_CHECKPOINT_TREE_SHA256,
            "checkpoint_manifest_sha256": subject.EXPECTED_CHECKPOINT_MANIFEST_SHA256,
            "checkpoint_verified_file_count": 23,
            "checkpoint_content_audit": checkpoint_audit,
            "renderer": "Bernini-R-1.3B-transformer_1",
            "transformer_count": 1,
            "transformer_block_count": core.TOTAL_BLOCKS,
        },
        "distributed": distributed,
        "first_forward_consensus": first,
        "owner_pairs": {"c0": [pair], "full": None, "cross_schedule_closure": None},
        "input_invariants": input_invariants,
        "method_source": {"revision": "c" * 40, "archive_sha256": "d" * 64},
        "execution": {
            "distributed_invocation_count": 1,
            "model_load_count": 1,
            "vae_load_count": 1,
            "c0_model_forward_count": 10,
            "c0_capture_forward_count": 2,
            "c0_internal_parity_forward_count": 2,
            "c0_decoded_output_count": 6,
            "full_model_forward_count": 0,
            "full_capture_forward_count": 0,
            "full_decoded_output_count": 0,
            "total_model_forward_count": 10,
            "scheduler_instance_count": 0,
            "scheduler_step_count": 0,
            "optimizer_instance_count": 0,
        },
        "c0": c0,
        "full": None,
        "outputs": outputs,
        "artifacts": artifacts,
        "model_integrity": integrity,
        "no_update": {**c0_no_update, "torch_inference_mode_all_forwards": True},
        "processor_patch": processor_patch_fixture(audits),
        "calibration": dict(subject._calibration_receipt()),
        "terminal_authority_audit": terminal,
        "interpretation": dict(subject._interpretation_receipt()),
    }
    return dict(subject.finalize_receipt(unsigned))


def formal_receipt_fixture() -> dict[str, object]:
    receipt = smoke_receipt_fixture()
    receipt.pop("receipt_digest")
    owner_pairs = receipt["owner_pairs"]
    assert isinstance(owner_pairs, dict)
    c0_rows = owner_pairs["c0"]
    assert isinstance(c0_rows, list) and len(c0_rows) == 1
    c0_pair = c0_rows[0]
    assert isinstance(c0_pair, dict)
    pairs: dict[int, dict[str, object]] = {}
    for schedule in core.policy.REGISTERED_SCHEDULE_INDICES:
        pairs[schedule] = (
            c0_pair if schedule == core.C0_SCHEDULE_INDEX
            else expanded_pair_row(schedule)
        )

    text_runtime = receipt["text_runtime"]
    assert isinstance(text_runtime, dict)
    branches = text_runtime["branches"]
    assert isinstance(branches, dict)
    noop = branches["noop"]
    assert isinstance(noop, dict)
    full_audits = full_processor_audits(pairs)
    full_caches = cache_audits_fixture(
        profile="full", pairs=pairs, audits=full_audits,
        text_runtime=text_runtime,
    )
    audit_by_id = {str(row["execution_id"]): row for row in full_audits}
    cache_by_id = {str(row["execution_id"]): row for row in full_caches}
    outputs = receipt["outputs"]
    artifacts = receipt["artifacts"]
    assert isinstance(outputs, list) and isinstance(artifacts, dict)
    artifacts["full-plan.json"] = hashlib.sha256(b"full-plan").hexdigest()
    for cell in core.full_grid_cells():
        pair = pairs[cell.schedule_index]
        owner_binding = pair[f"{cell.owner}_binding"]
        pair_validation = pair["pair_validation"]
        assert isinstance(owner_binding, dict) and isinstance(pair_validation, dict)
        binding_value = subject._binding_from_receipt(
            owner_binding, label="formal output fixture"
        )
        velocity = hashlib.sha256(f"velocity-{cell.cell_id}".encode()).hexdigest()
        predecode = hashlib.sha256(f"x0-{cell.cell_id}".encode()).hexdigest()
        media = hashlib.sha256(f"media-{cell.cell_id}".encode()).hexdigest()
        name = f"full/{cell.output_name}"
        artifacts[name] = media
        cache_digest = None
        if cell.role != "noop_baseline":
            capture_id = subject._capture_id(
                "full", cell.owner, cell.schedule_index, cell.branch
            )
            cache_digest = cache_by_id[capture_id]["digest"]
        outputs.append({
            "name": name,
            "phase": "full",
            "cell": dict(cell.receipt()),
            "cell_digest": subject.object_sha256(cell.receipt()),
            "owner_input_binding": owner_binding,
            "owner_input_binding_digest": binding_value.digest,
            "owner_pair_validation_digest": pair_validation["digest"],
            "velocity_sha256": velocity,
            "predecode_x0_hat_sha256": predecode,
            "sha256": media,
            "frames": subject.EXPECTED_FRAMES,
            "fps": float(subject.FPS),
            "hw": list(subject.EXPECTED_HW),
            "decode_input_latent_sha256": predecode,
            "decode_input_latent_shape": list(subject.EXPECTED_LATENT_SHAPE),
            "vae_frozen_eval": True,
            "actual_object_binding_used_for_forward": True,
            "global_prompt_branch": "noop",
            "global_noop_prompt_runtime_binding_digest": noop["digest"],
            "global_noop_embedding_sha256": noop["embedding_sha256"],
            "processor_audit_digest": audit_by_id[cell.cell_id]["digest"],
            "branch_capture_cache_audit_digest": cache_digest,
            "result_rank_sha256": [velocity] * 4,
            "result_world4_consensus": True,
            "decode_input_dtype": "torch.float32",
            "decode_input_device_type": "cuda",
            "decode_input_contiguous": True,
            "decode_input_finite": True,
        })

    input_pre_c0 = input_snapshot_fixture(pairs=pairs, text_runtime=text_runtime)
    receipt["input_invariants"] = dict(subject.build_input_invariant_receipt(
        profile=subject.FORMAL_PROFILE, pre_c0=input_pre_c0,
        post_c0=dict(input_pre_c0), post_full=dict(input_pre_c0),
        terminal=dict(input_pre_c0),
    ))
    integrity = receipt["model_integrity"]
    c0 = receipt["c0"]
    assert isinstance(integrity, dict) and isinstance(c0, dict)
    c0_caches = c0["cache_audits"]
    assert isinstance(c0_caches, list) and c0_caches
    first_cache = c0_caches[0]
    assert isinstance(first_cache, dict)
    input_bundle = {
        "model_sha256": integrity["pre_sha256"],
        "text_runtime_digest": text_runtime["digest"],
        "epsilon_sha256": binding("correct_owner", 29).epsilon_sha256,
        "orbit_row_digest": core.ORBIT_ROW_DIGEST,
        "owner_pair_digests": [
            pairs[schedule]["digest"]
            for schedule in core.policy.REGISTERED_SCHEDULE_INDICES
        ],
        "c0_plan_digest": core.build_plan("c0-smoke")["plan_digest"],
        "full_plan_digest": core.build_plan("full-grid")["plan_digest"],
        "actual_input_pre_c0_snapshot_digest": input_pre_c0["digest"],
    }
    input_digest = subject.object_sha256(input_bundle)
    first = digest_record({
        "world_size": 4,
        "passed": True,
        "before_any_real_forward": True,
        "actual_model_text_input_tensor_hashes": True,
        "input_bundle_digest": input_digest,
        "input_rank_digests": [input_digest] * 4,
        "first_execution_id": first_cache["execution_id"],
        "first_output_sha256": first_cache["capture_prediction_sha256"],
        "first_output_rank_digests": first_cache["capture_prediction_rank_sha256"],
        "first_output_world4_consensus": True,
    })
    receipt["first_forward_consensus"] = first
    c0["first_forward_consensus"] = dict(first)
    resign(c0)

    owner_pairs["full"] = [
        pairs[schedule] for schedule in core.policy.REGISTERED_SCHEDULE_INDICES
    ]
    owner_pairs["cross_schedule_closure"] = dict(
        subject.build_cross_schedule_owner_closure(pairs, c0_pair=c0_pair)
    )
    receipt["profile"] = subject.FORMAL_PROFILE
    receipt["same_load_c0_then_full"] = True
    receipt["full"] = digest_record({
        "started_after_c0_pass": True,
        "same_model_load": True,
        "fixed_plan_no_adaptation": True,
        "decoded_output_count": 112,
        "completed": True,
        "processor_audits": full_audits,
        "cache_audits": full_caches,
        "plan_digest": core.build_plan("full-grid")["plan_digest"],
    })
    receipt["execution"] = {
        "distributed_invocation_count": 1,
        "model_load_count": 1,
        "vae_load_count": 1,
        "c0_model_forward_count": 10,
        "c0_capture_forward_count": 2,
        "c0_internal_parity_forward_count": 2,
        "c0_decoded_output_count": 6,
        "full_model_forward_count": 136,
        "full_capture_forward_count": 24,
        "full_decoded_output_count": 112,
        "total_model_forward_count": 146,
        "scheduler_instance_count": 0,
        "scheduler_step_count": 0,
        "optimizer_instance_count": 0,
    }
    c0_audits = c0["processor_audits"]
    assert isinstance(c0_audits, list)
    receipt["processor_patch"] = processor_patch_fixture(c0_audits + full_audits)
    return dict(subject.finalize_receipt(receipt))


class RuntimeContractTests(unittest.TestCase):
    @staticmethod
    def _minimal_c0_gate(cache_audit: dict[str, object]) -> dict[str, object]:
        parity = []
        for owner in core.OWNERS:
            parity.append({
                "owner": owner,
                "plain_vs_swap_velocity_raw_bytes_equal": True,
                "plain_vs_swap_predecode_raw_bytes_equal": True,
                "plain_velocity_sha256": "a" * 64,
                "swap_velocity_sha256": "a" * 64,
                "plain_predecode_sha256": "b" * 64,
                "swap_predecode_sha256": "b" * 64,
                "internal_noop_swap_decoded": False,
            })
        outputs = [
            {
                "name": f"c0/output-{index}.mp4",
                "frames": subject.EXPECTED_FRAMES,
                "fps": float(subject.FPS),
                "hw": list(subject.EXPECTED_HW),
            }
            for index in range(6)
        ]
        return dict(subject.build_c0_engineering_gate(
            noop_parity=parity,
            processor_audits=[{
                "exact_hook_counts": True,
                "all30_processor_inventory": True,
            }],
            cache_audits=[cache_audit], output_records=outputs,
            first_forward_consensus={"world_size": 4, "passed": True},
            model_integrity={
                "pre_c0_sha256": "c" * 64,
                "post_c0_sha256": "c" * 64,
                "bytes_unchanged": True,
            },
            no_update={
                "gradient_enabled": False, "optimizer_present": False,
                "scheduler_present": False, "scheduler_steps": 0,
                "parameter_gradients_present": False, "parameter_updates": 0,
            },
        ))

    def test_world4_rng_evidence_accepts_distinct_unchanged_rank_states(self) -> None:
        rank_states = [
            hashlib.sha256(f"rank-{rank}-rng".encode()).hexdigest()
            for rank in range(4)
        ]
        rows = [
            {"rank": rank, "before_sha256": value, "after_sha256": value}
            for rank, value in enumerate(rank_states)
        ]
        evidence = subject._world4_rng_state_evidence(rows)
        self.assertEqual(
            evidence["rng_state_before_world4_rank_sha256"], rank_states
        )
        self.assertEqual(
            evidence["rng_state_after_world4_rank_sha256"], rank_states
        )
        self.assertEqual(
            evidence["rng_state_before_sha256"],
            subject._rng_world4_vector_sha256(
                rank_states, label="expected WORLD4 RNG"
            ),
        )
        self.assertEqual(
            evidence["rng_state_before_sha256"],
            evidence["rng_state_after_sha256"],
        )

    def test_world4_rng_evidence_rejects_one_changed_rank(self) -> None:
        rows = []
        for rank in range(4):
            value = hashlib.sha256(f"rank-{rank}-rng".encode()).hexdigest()
            rows.append({
                "rank": rank,
                "before_sha256": value,
                "after_sha256": (
                    hashlib.sha256(b"hostile-rank-2-after").hexdigest()
                    if rank == 2 else value
                ),
            })
        with self.assertRaisesRegex(
            subject.StageARuntimeError, "changed on rank 2"
        ):
            subject._world4_rng_state_evidence(rows)

    def test_world4_rng_evidence_rejects_noncanonical_rank_order(self) -> None:
        rows = []
        for rank in range(4):
            value = hashlib.sha256(f"rank-{rank}-rng".encode()).hexdigest()
            rows.append({
                "rank": rank, "before_sha256": value, "after_sha256": value,
            })
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaisesRegex(
            subject.StageARuntimeError, "rank order differs"
        ):
            subject._world4_rng_state_evidence(rows)

    def test_distinct_rank_rng_local_views_have_one_cache_and_c0_digest(self) -> None:
        rank_states = [
            hashlib.sha256(f"local-view-rank-{rank}".encode()).hexdigest()
            for rank in range(4)
        ]
        gathered = [
            {"rank": rank, "before_sha256": value, "after_sha256": value}
            for rank, value in enumerate(rank_states)
        ]
        local_views = [
            {
                "local_rank": rank,
                "local_rng_sha256": rank_states[rank],
                "gathered": [dict(row) for row in gathered],
            }
            for rank in range(4)
        ]
        self.assertEqual(
            len({str(view["local_rng_sha256"]) for view in local_views}), 4
        )
        cache_digests: list[str] = []
        gate_digests: list[str] = []
        for view in local_views:
            evidence = subject._world4_rng_state_evidence(view["gathered"])
            cache = digest_record({
                "schema_version": "test-capture-cache-rng-evidence-v1",
                **dict(evidence),
            })
            cache_digests.append(str(cache["digest"]))
            gate_digests.append(str(self._minimal_c0_gate(cache)["digest"]))
        self.assertEqual(len(set(cache_digests)), 1)
        self.assertEqual(len(set(gate_digests)), 1)

    def test_cache_verifier_rejects_hostile_world4_rng_evidence(self) -> None:
        cases = ("short_vector", "changed_rank", "forged_aggregate")
        for case in cases:
            with self.subTest(case=case):
                pair = expanded_pair_row(core.C0_SCHEDULE_INDEX)
                text_runtime = text_runtime_fixture()
                audits = c0_processor_audits(pair)
                caches = c0_cache_audits(pair, audits, text_runtime)
                cache = caches[0]
                before = cache["rng_state_before_world4_rank_sha256"]
                after = cache["rng_state_after_world4_rank_sha256"]
                assert isinstance(before, list) and isinstance(after, list)
                if case == "short_vector":
                    before.pop()
                elif case == "changed_rank":
                    after[2] = hashlib.sha256(b"hostile-rank-2-after").hexdigest()
                    cache["rng_state_after_sha256"] = (
                        subject._rng_world4_vector_sha256(
                            after, label="hostile RNG after"
                        )
                    )
                else:
                    cache["rng_state_before_sha256"] = "f" * 64
                resign(cache)
                with self.assertRaises(subject.StageARuntimeError):
                    subject._validate_cache_audits(
                        caches, profile="c0",
                        pairs={core.C0_SCHEDULE_INDEX: pair},
                        processor_audits=audits,
                        text_bindings=text_runtime["branches"],  # type: ignore[arg-type]
                    )

    def test_live_cache_identity_rejects_same_object_for_c0_and_full(self) -> None:
        for label in ("C0 owner axis", "full owner/schedule/branch"):
            with self.subTest(label=label):
                live_references: dict[int, object] = {}
                cache = core.PostConditionBranchCache("forward")
                self.assertTrue(subject._admit_unique_cache_object(
                    cache, live_references, label=label
                ))
                with self.assertRaisesRegex(
                    subject.StageARuntimeError, "cache object was reused"
                ):
                    subject._admit_unique_cache_object(
                        cache, live_references, label=label
                    )

    def test_dead_cache_address_tombstone_is_reusable_for_c0_and_full(self) -> None:
        class Cache:
            pass

        stale_cache = Cache()
        stale_reference = weakref.ref(stale_cache)
        del stale_cache
        gc.collect()
        self.assertIsNone(stale_reference())
        for label in ("C0 owner axis", "full owner/schedule/branch"):
            with self.subTest(label=label):
                current = Cache()
                simulated_reused_address = id(current)
                live_references: dict[int, object] = {
                    simulated_reused_address: stale_reference,
                }
                self.assertTrue(subject._admit_unique_cache_object(
                    current, live_references, label=label
                ))
                current_reference = live_references[simulated_reused_address]
                self.assertIsInstance(current_reference, weakref.ReferenceType)
                self.assertIs(current_reference(), current)  # type: ignore[operator]

    def test_cache_identity_registry_prunes_dead_entries_on_next_admission(self) -> None:
        class Cache:
            pass

        stale_cache = Cache()
        stale_reference = weakref.ref(stale_cache)
        del stale_cache
        gc.collect()
        stale_key = -1
        live_references: dict[int, object] = {stale_key: stale_reference}
        current = Cache()
        self.assertTrue(subject._admit_unique_cache_object(
            current, live_references, label="full owner/schedule/branch"
        ))
        self.assertNotIn(stale_key, live_references)
        self.assertEqual(set(live_references), {id(current)})

    def test_cache_identity_registry_corruption_fails_closed(self) -> None:
        class Cache:
            pass

        with self.assertRaisesRegex(
            subject.StageARuntimeError, "identity registry differs"
        ):
            subject._admit_unique_cache_object(
                Cache(), {0: object()}, label="hostile registry"
            )

    def test_nonweakrefable_cache_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            subject.StageARuntimeError, "does not support weak identity tracking"
        ):
            subject._admit_unique_cache_object(
                object(), {}, label="hostile cache"
            )

    def test_cache_identity_registry_does_not_retain_cache_or_payload(self) -> None:
        class GpuPayloadStandIn:
            pass

        live_references: dict[int, object] = {}
        cache = core.PostConditionBranchCache("forward")
        payload = GpuPayloadStandIn()
        cache._values[0] = payload
        cache_reference = weakref.ref(cache)
        payload_reference = weakref.ref(payload)
        self.assertTrue(subject._admit_unique_cache_object(
            cache, live_references, label="full owner/schedule/branch"
        ))
        self.assertTrue(all(
            isinstance(reference, weakref.ReferenceType)
            for reference in live_references.values()
        ))
        del cache
        del payload
        gc.collect()
        self.assertIsNone(cache_reference())
        self.assertIsNone(payload_reference())
        self.assertTrue(all(
            reference() is None  # type: ignore[operator]
            for reference in live_references.values()
        ))

    def test_fixed_plan_forward_and_cache_counts(self) -> None:
        self.assertEqual(len(core.c0_smoke_cells()), 6)
        self.assertEqual(len(core.full_grid_cells()), 112)
        self.assertEqual(len(subject._expected_c0_audit_descriptors()), 10)
        self.assertEqual(len(subject._expected_full_audit_descriptors()), 136)
        c0 = subject._expected_cache_specs("c0")
        full = subject._expected_cache_specs("full")
        self.assertEqual(len(c0), 2)
        self.assertEqual([item["expected_reuse_count"] for item in c0], [37, 37])
        self.assertEqual(len(full), 24)
        self.assertEqual(
            [item["expected_reuse_count"] for item in full[:6]],
            [60, 30, 30, 30, 30, 60],
        )
        self.assertTrue(all(item["band"] == core.ALL30_BAND for item in subject._expected_c0_audit_descriptors() if item["phase"] == "capture"))

    def test_exact_output_names_are_safe_unique_fixed_plan(self) -> None:
        names = subject.expected_artifact_names(subject.FORMAL_PROFILE)
        self.assertEqual(len(names), 120)
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            path = Path(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertEqual(path.as_posix(), name)

    def test_cross_schedule_closure_rejects_epsilon_change(self) -> None:
        full = {schedule: pair_row(schedule) for schedule in core.policy.REGISTERED_SCHEDULE_INDICES}
        result = subject.build_cross_schedule_owner_closure(full, c0_pair=full[29])
        self.assertTrue(result["single_epsilon_reused_across_all_schedules_and_owners"])
        hostile = dict(full)
        hostile[35] = pair_row(35, epsilon="f")
        with self.assertRaises(subject.StageARuntimeError):
            subject.build_cross_schedule_owner_closure(hostile, c0_pair=hostile[29])

    def test_cross_phase_s29_must_be_raw_equal(self) -> None:
        full = {schedule: pair_row(schedule) for schedule in core.policy.REGISTERED_SCHEDULE_INDICES}
        hostile_c0 = pair_row(29)
        hostile_c0["extra"] = False
        with self.assertRaises(subject.StageARuntimeError):
            subject.build_cross_schedule_owner_closure(full, c0_pair=hostile_c0)

    def test_single_binding_cannot_claim_pair_switch(self) -> None:
        row = dict(binding("correct_owner", 29).receipt())
        row["owner_pair_switch_audited_by_single_binding"] = True
        with self.assertRaises(subject.StageARuntimeError):
            subject._binding_from_receipt(row, label="hostile")

    def test_v2_substitution_rejected_before_pack(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        tensor = torch.zeros(subject.EXPECTED_LATENT_SHAPE, dtype=torch.float32)
        refs = tuple(torch.zeros((1, 16, 1, 74, 50), dtype=torch.float32) for _ in range(4))
        row = SimpleNamespace(
            iid=subject.IID, row_digest=core.ORBIT_ROW_DIGEST,
            variant_native_arms=("rv2v", "r2v"),
            full_shape=(1, 32, 21, 74, 50),
            reference_shape=(1, 32, 1, 74, 50), posterior_blobs={},
        )
        source = SimpleNamespace(name="V0", video_latent=tensor, image_references=refs)
        wrong_v2 = SimpleNamespace(name="V2", video_latent=tensor.clone(), image_references=refs)
        with self.assertRaisesRegex(subject.StageARuntimeError, "member alias"):
            subject.build_actual_owner_pair(
                transformer=object(), native=object(), row=row,
                source_member=source, wrong_member=wrong_v2,
                epsilon=tensor.clone(), schedule_index=29, sp_rank=0,
            )

    def test_module_state_certificate_is_raw_and_frozen(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        module = torch.nn.Linear(3, 2, bias=False).requires_grad_(False)
        first = subject._module_state_certificate(module)
        second = subject._module_state_certificate(module)
        self.assertEqual(first, second)
        self.assertTrue(first["all_parameters_frozen"])
        module.weight.requires_grad_(True)
        with self.assertRaises(subject.StageARuntimeError):
            subject._module_state_certificate(module)

    def test_actual_pack_phase_snapshots_reject_native_inplace_mutation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        total_tokens, condition_tokens = 100, 60
        epsilon = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).contiguous()
        x_s = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).add(10).contiguous()
        rotary = torch.arange(
            total_tokens, dtype=torch.float32
        ).reshape(1, 1, total_tokens, 1).contiguous()
        correct_latents = torch.zeros(
            (1, total_tokens, 2), dtype=torch.float32
        ).contiguous()
        wrong_latents = correct_latents.clone()
        wrong_latents[:, :condition_tokens].fill_(2.0)
        decoded_target = torch.zeros((1, 2), dtype=torch.float32)
        decoded_wrong = torch.ones((1, 2), dtype=torch.float32)
        decoded_target_sha = subject.runtime.tensor_sha256(decoded_target)
        decoded_wrong_sha = subject.runtime.tensor_sha256(decoded_wrong)
        epsilon_sha = subject.runtime.tensor_sha256(epsilon)
        x_s_sha = subject.runtime.tensor_sha256(x_s)
        rotary_prefix_sha = subject.runtime.tensor_sha256(
            rotary.narrow(2, 0, condition_tokens).contiguous()
        )

        def make_pack(owner: str, latents: object) -> subject.ActualOwnerPack:
            assert isinstance(latents, torch.Tensor)
            correct = owner == "correct_owner"
            branch = SimpleNamespace(
                latents=latents,
                rotary=rotary.clone(),
                source_ids=core.NATIVE_SOURCE_IDS,
                total_tokens=total_tokens,
                condition_tokens=condition_tokens,
            )
            owner_binding = core.OwnerInputBinding(
                owner=owner,
                schedule_index=core.C0_SCHEDULE_INDEX,
                timestep=subject.exact40.PINNED_TIMESTEPS[core.C0_SCHEDULE_INDEX],
                sigma_float32_be_hex=subject.exact40.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[
                    core.C0_SCHEDULE_INDEX
                ],
                orbit_row_digest=core.ORBIT_ROW_DIGEST,
                target_source_full_blob_sha256=core.OWNER_FULL_BLOB_SHA256[
                    "correct_owner"
                ],
                owner_full_blob_sha256=core.OWNER_FULL_BLOB_SHA256[owner],
                owner_reference_blob_sha256=core.OWNER_REFERENCE_BLOB_SHA256[owner],
                decoded_target_tensor_sha256=decoded_target_sha,
                decoded_owner_full_tensor_sha256=(
                    decoded_target_sha if correct else decoded_wrong_sha
                ),
                decoded_owner_reference_tensor_sha256=tuple(
                    hashlib.sha256(f"decoded-ref-{owner}-{index}".encode()).hexdigest()
                    for index in range(4)
                ),
                epsilon_sha256=epsilon_sha,
                target_x_s_sha256=x_s_sha,
                prepared_visual_prefix_sha256=subject.runtime.tensor_sha256(
                    branch.latents[:, :condition_tokens].contiguous()
                ),
                prepared_prefix_rotary_sha256=rotary_prefix_sha,
                total_tokens=total_tokens,
                condition_tokens=condition_tokens,
                source_ids=core.NATIVE_SOURCE_IDS,
            )
            return subject.ActualOwnerPack(
                owner=owner,
                schedule_index=core.C0_SCHEDULE_INDEX,
                branch=branch,
                binding=owner_binding,
                layout=core.NativeTargetSuffixLayout(
                    total_tokens, condition_tokens, 0, 4
                ),
            )

        correct_pack = make_pack("correct_owner", correct_latents)
        wrong_pack = make_pack("wrong_owner", wrong_latents)
        packs = {
            core.C0_SCHEDULE_INDEX: {
                "correct_owner": correct_pack, "wrong_owner": wrong_pack,
            }
        }
        prompts = {
            branch: torch.full((1, 2), float(index), dtype=torch.float32)
            for index, branch in enumerate(core.TEXT_BRANCHES)
        }
        text_bindings = {
            branch: {"embedding_sha256": subject.runtime.tensor_sha256(value)}
            for branch, value in prompts.items()
        }
        pre = subject.snapshot_actual_model_inputs(
            packs=packs, x_s_by_schedule={core.C0_SCHEDULE_INDEX: x_s},
            epsilon=epsilon, prompt_embeddings=prompts,
            text_bindings=text_bindings, schedules=(core.C0_SCHEDULE_INDEX,),
        )
        correct_pack.branch.latents.data[:, condition_tokens:].add_(1.0)
        post = subject.snapshot_actual_model_inputs(
            packs=packs, x_s_by_schedule={core.C0_SCHEDULE_INDEX: x_s},
            epsilon=epsilon, prompt_embeddings=prompts,
            text_bindings=text_bindings, schedules=(core.C0_SCHEDULE_INDEX,),
        )
        self.assertNotEqual(pre["digest"], post["digest"])
        with self.assertRaisesRegex(subject.StageARuntimeError, "changed during C0"):
            subject.build_input_invariant_receipt(
                profile="smoke-only", pre_c0=pre, post_c0=post,
                post_full=None, terminal=post,
            )
        x_s.data.add_(1.0)
        with self.assertRaisesRegex(subject.StageARuntimeError, "sealed binding"):
            subject.snapshot_actual_model_inputs(
                packs=packs, x_s_by_schedule={core.C0_SCHEDULE_INDEX: x_s},
                epsilon=epsilon, prompt_embeddings=prompts,
                text_bindings=text_bindings, schedules=(core.C0_SCHEDULE_INDEX,),
            )

    def test_stable_snapshot_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.json"
            target.write_bytes(b"{}\n")
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(subject.StageARuntimeError):
                subject.stable_file_snapshot(link, label="hostile link")

    def test_dataset_record_requires_raw_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parquet = root / "dataset.parquet"
            parquet.write_bytes(b"sealed")
            snapshot = subject.stable_file_snapshot(parquet, label="parquet")
            alias = root / "alias.parquet"
            alias.symlink_to(parquet)
            record = {
                "path": str(alias), "sha256": snapshot.sha256, "rows": 1,
                "iids": [subject.IID],
            }
            with self.assertRaises(subject.StageARuntimeError):
                subject._dataset_record_gate(
                    record, parquet=snapshot, root=root, label="hostile dataset"
                )

    def test_rank_zero_callback_failure_is_symmetric_error(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("torch distributed unavailable")
        with mock.patch.object(dist, "broadcast_object_list", return_value=None):
            with self.assertRaisesRegex(subject.StageARuntimeError, "rank zero"):
                subject._rank_zero_call(
                    lambda: (_ for _ in ()).throw(ValueError("boom")),
                    rank=0, group=object(), label="hostile callback",
                )

    def test_authority_digests_come_from_assets(self) -> None:
        spec = json.loads(
            (METHOD_ROOT / "assets/source_self_role_repaint_canary_spec_v2.json").read_text()
        )
        unsigned = dict(spec)
        declared = unsigned.pop("spec_digest")
        self.assertEqual(declared, subject.SOURCE_DATASET_SPEC_DIGEST)
        self.assertEqual(subject.object_sha256(unsigned), declared)
        import appearance_counterfactual_identity_orbit as appearance
        self.assertEqual(
            subject.ORBIT_REFERENCE_ENCODING_CONTRACT_DIGEST,
            appearance.PINNED_REFERENCE_ENCODING_CONTRACT_DIGEST,
        )

    def test_parser_freezes_formal_and_debug_profiles(self) -> None:
        parser = subject.build_parser()
        verify = parser.parse_args(["verify", "--output-dir", "/tmp/example"])
        self.assertEqual(verify.command, "verify")
        with mock.patch("sys.stderr") , self.assertRaises(SystemExit):
            parser.parse_args([
                "run", "--profile", "adaptive", "--output-dir", "/tmp/out",
            ])

    def test_complete_smoke_receipt_canonical_roundtrip(self) -> None:
        receipt = smoke_receipt_fixture()
        subject.validate_receipt(receipt, expected_profile="smoke-only")
        raw = subject.canonical_json_bytes(receipt) + b"\n"
        parsed = subject._strict_json_bytes(raw, label="roundtrip receipt")
        self.assertEqual(
            subject.validate_receipt(parsed, expected_profile="smoke-only"), parsed
        )

    def test_complete_formal_112_receipt_canonical_roundtrip(self) -> None:
        receipt = formal_receipt_fixture()
        self.assertEqual(len(receipt["outputs"]), 118)  # type: ignore[arg-type]
        full = receipt["full"]
        assert isinstance(full, dict)
        self.assertEqual(len(full["processor_audits"]), 136)  # type: ignore[arg-type]
        self.assertEqual(len(full["cache_audits"]), 24)  # type: ignore[arg-type]
        subject.validate_receipt(receipt, expected_profile=subject.FORMAL_PROFILE)
        raw = subject.canonical_json_bytes(receipt) + b"\n"
        parsed = subject._strict_json_bytes(raw, label="formal roundtrip receipt")
        self.assertEqual(
            subject.validate_receipt(
                parsed, expected_profile=subject.FORMAL_PROFILE
            ),
            parsed,
        )

    def test_resigned_formal_112_hostile_matrix_fails_closed(self) -> None:
        cases = (
            "missing_output", "extra_output", "reordered_output",
            "started_before_c0", "different_model_load", "schedule_epsilon_drift",
            "full_media_geometry", "input_phase_digest",
        )
        for case in cases:
            with self.subTest(case=case):
                receipt = formal_receipt_fixture()
                outputs = receipt["outputs"]
                full = receipt["full"]
                assert isinstance(outputs, list) and isinstance(full, dict)
                if case == "missing_output":
                    outputs.pop()
                elif case == "extra_output":
                    outputs.append(dict(outputs[-1]))
                elif case == "reordered_output":
                    outputs[-1], outputs[-2] = outputs[-2], outputs[-1]
                elif case == "started_before_c0":
                    full["started_after_c0_pass"] = False
                    resign(full)
                elif case == "different_model_load":
                    full["same_model_load"] = False
                    resign(full)
                elif case == "schedule_epsilon_drift":
                    owner_pairs = receipt["owner_pairs"]
                    assert isinstance(owner_pairs, dict)
                    pairs = owner_pairs["full"]
                    assert isinstance(pairs, list)
                    schedule_35 = list(core.policy.REGISTERED_SCHEDULE_INDICES).index(35)
                    pairs[schedule_35] = expanded_pair_row(35, epsilon="f")
                elif case == "full_media_geometry":
                    last = outputs[-1]
                    assert isinstance(last, dict)
                    last["frames"] = subject.EXPECTED_FRAMES - 1
                elif case == "input_phase_digest":
                    invariants = receipt["input_invariants"]
                    assert isinstance(invariants, dict)
                    stage_digests = invariants["stage_snapshot_digests"]
                    assert isinstance(stage_digests, dict)
                    stage_digests["post_full"] = "f" * 64
                    rank_digests = invariants["stage_world4_rank_digests"]
                    assert isinstance(rank_digests, dict)
                    rank_digests["post_full"] = ["f" * 64] * 4
                    resign(invariants)
                resign(receipt, "receipt_digest")
                with self.assertRaises(subject.StageARuntimeError):
                    subject.validate_receipt(
                        receipt, expected_profile=subject.FORMAL_PROFILE
                    )

    def test_resigned_nested_hostile_receipt_matrix_fails_closed(self) -> None:
        cases = (
            "text_type", "text_inner_digest", "cache_shape", "processor_execution",
            "patch_statistics", "calibration_extra", "topology_rank",
            "first_input", "output_traversal",
        )
        for case in cases:
            with self.subTest(case=case):
                receipt = smoke_receipt_fixture()
                c0 = receipt["c0"]
                assert isinstance(c0, dict)
                if case.startswith("text_"):
                    text = receipt["text_runtime"]
                    assert isinstance(text, dict)
                    branches = text["branches"]
                    assert isinstance(branches, dict)
                    forward = branches["forward"]
                    assert isinstance(forward, dict)
                    if case == "text_type":
                        forward["embedding_shape"] = None
                    else:
                        forward["runtime_tensor_binding_digest"] = "f" * 64
                    resign(forward)
                    resign(text)
                elif case == "cache_shape":
                    caches = c0["cache_audits"]
                    assert isinstance(caches, list)
                    cache = caches[0]
                    assert isinstance(cache, dict)
                    cache_receipt = cache["cache_receipt"]
                    assert isinstance(cache_receipt, dict)
                    shapes = cache_receipt["captured_shapes"]
                    assert isinstance(shapes, dict)
                    shapes["0"] = [1, 511, 4096]
                    resign(cache_receipt)
                    resign(cache)
                    resign(c0)
                elif case == "processor_execution":
                    audits = c0["processor_audits"]
                    assert isinstance(audits, list)
                    audit = audits[0]
                    assert isinstance(audit, dict)
                    audit["execution_id"] = "forged-capture"
                    resign(audit)
                    resign(c0)
                elif case == "patch_statistics":
                    patch = receipt["processor_patch"]
                    assert isinstance(patch, dict)
                    statistics = patch["statistics"]
                    assert isinstance(statistics, list)
                    row = statistics[0]
                    assert isinstance(row, dict)
                    row["base_calls"] = int(row["base_calls"]) + 1
                    resign(patch)
                elif case == "calibration_extra":
                    calibration = receipt["calibration"]
                    assert isinstance(calibration, dict)
                    calibration["scientific_pass"] = True
                    resign(calibration)
                elif case == "topology_rank":
                    distributed = receipt["distributed"]
                    assert isinstance(distributed, dict)
                    placements = distributed["rank_hostname_local_rank"]
                    assert isinstance(placements, list)
                    row = placements[2]
                    assert isinstance(row, dict)
                    row["local_rank"] = 1
                    resign(distributed)
                elif case == "first_input":
                    first = receipt["first_forward_consensus"]
                    assert isinstance(first, dict)
                    first["input_bundle_digest"] = "f" * 64
                    first["input_rank_digests"] = ["f" * 64] * 4
                    resign(first)
                    c0["first_forward_consensus"] = dict(first)
                    resign(c0)
                elif case == "output_traversal":
                    outputs = receipt["outputs"]
                    artifacts = receipt["artifacts"]
                    assert isinstance(outputs, list) and isinstance(artifacts, dict)
                    output = outputs[0]
                    assert isinstance(output, dict)
                    old_name = str(output["name"])
                    output["name"] = "../escape.mp4"
                    artifacts["../escape.mp4"] = artifacts.pop(old_name)
                    c0["decoded_output_names"][0] = "../escape.mp4"  # type: ignore[index]
                    c0["decoded_output_record_digests"][0] = subject.object_sha256(output)  # type: ignore[index]
                    resign(c0)
                resign(receipt, "receipt_digest")
                with self.assertRaises(subject.StageARuntimeError):
                    subject.validate_receipt(receipt, expected_profile="smoke-only")

    def test_publication_rejects_stage_replacement_after_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "output"
            stage = root / ".output.staging"
            old = root / "old-stage"
            stage.mkdir(mode=0o750)
            reserved = subject._directory_reservation_identity(stage)
            subject.runtime._OUTPUT_STAGE_IDENTITIES[str(stage)] = reserved[:3]
            calls = 0

            def hostile_verify(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    stage.rename(old)
                    stage.mkdir(mode=0o750)
                return {}

            try:
                with mock.patch.object(
                    subject, "verify_bundle", side_effect=hostile_verify
                ), mock.patch.object(
                    subject.runtime, "fsync_directory", return_value=None
                ), mock.patch.object(
                    subject, "_rank_zero_call",
                    side_effect=lambda callback, **kwargs: callback(),
                ):
                    with self.assertRaisesRegex(
                        subject.StageARuntimeError, "changed after verification"
                    ):
                        subject._publish_bundle(
                            output=output, stage=stage,
                            receipt={"profile": "smoke-only"},
                            reserved_stage_identity=reserved,
                            rank=0, group=object(),
                        )
            finally:
                subject.runtime._OUTPUT_STAGE_IDENTITIES.pop(str(stage), None)

    def test_no_public_mutable_module_authority_literal(self) -> None:
        tree = ast.parse(Path(subject.__file__).read_text())
        offenders: list[str] = []
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            names = [target.id for target in node.targets if isinstance(target, ast.Name)] if isinstance(node, ast.Assign) else ([node.target.id] if isinstance(node.target, ast.Name) else [])
            if names and any(not name.startswith("_") for name in names) and isinstance(value, (ast.Dict, ast.List, ast.Set)):
                offenders.extend(names)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

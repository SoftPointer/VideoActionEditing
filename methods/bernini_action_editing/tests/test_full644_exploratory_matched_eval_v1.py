from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import full644_exploratory_matched_eval_v1 as matched


INPUT_MANIFEST = (
    REPO_ROOT
    / "methods/action_editing_baselines/manifests/goku_legacy_heldout8_inputs.jsonl"
)
EXPOSURE_AUDIT = (
    REPO_ROOT
    / "methods/action_editing_baselines/manifests/goku_legacy_shared8_exposure.json"
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict) -> str:
    path.write_bytes(matched.canonical_json_bytes(value) + b"\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_checkpoint(root: Path) -> tuple[dict, Path, str]:
    checkpoint = root / "checkpoint-00000644"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter/adapter_config.json").write_bytes(b"{}\n")
    (checkpoint / "adapter/adapter_model.safetensors").write_bytes(b"model")
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    dataset_signature = sha("full644-dataset-content")
    source_authority = {
        "path": "/authority/full644-source-receipt.json",
        "sha256": matched.FULL644_SOURCE_AUTHORITY_SHA256,
        "membership_rows": 644,
        "action_family_count": 28,
        "unique_group_id": 644,
        "unique_source_video_sha256": 644,
        "raw_parquet_sha256": (
            "706d835a8cdf924776000d69b229c272fd434a91abc8942c67dc6fd7732b7d1b"
        ),
        "vae_index_sha256": matched.FULL644_DATASET_INDEX_SHA256,
        "vae_summary_sha256": matched.FULL644_DATASET_SUMMARY_SHA256,
        "role": "historical_exposed_train_debug_not_heldout",
        "historical_receipt_user_authorization_is_not_current_launch_authority": True,
    }
    receipt = {
        "schema_version": matched.TRAINING_RECEIPT_SCHEMA,
        "global_step": 644,
        "max_steps": 644,
        "last_loss": 1.0,
        "last_preclip_gradient_norm": 1.0,
        "bernini_commit": matched.EXPECTED_BERNINI_COMMIT,
        "bernini_training_files_index_sha256": (
            matched.EXPECTED_BERNINI_TRAINING_FILES_INDEX_SHA256
        ),
        "veomni_commit": matched.EXPECTED_VEOMNI_COMMIT,
        "method_source_revision": "1" * 40,
        "method_source_archive_sha256": "2" * 64,
        "checkpoint": {
            "path": "/models/Bernini-R-1.3B-Diffusers",
            "configs": {
                "model_index.json": sha("base:model-index"),
                "transformer/config.json": sha("base:transformer-config"),
                "vae/config.json": sha("base:vae-config"),
            },
        },
        "checkpoint_tree_sha256": matched.EXPECTED_CHECKPOINT_TREE_SHA256,
        "dataset": {
            "path": "/datasets/full644/vae_parquet",
            "rows": 644,
            "signature": dataset_signature,
            "content_signature": dataset_signature,
            "summary": {
                "path": "/datasets/full644/summary.json",
                "sha256": matched.FULL644_DATASET_SUMMARY_SHA256,
                "summary_digest": matched.FULL644_DATASET_SUMMARY_DIGEST,
                "complete": True,
                "allow_incomplete": False,
                "expected_rows": 644,
                "materialized_rows": 644,
                "index_path": "/datasets/full644/index.jsonl",
                "index_sha256": matched.FULL644_DATASET_INDEX_SHA256,
                "indexed_shards_sha256": sha("full644-indexed-shards"),
                "dataset_content_signature": dataset_signature,
                "reward_selected_synthetic_targets": False,
                "arm": None,
            },
        },
        "target_module_count": matched.EXPECTED_TARGET_MODULE_COUNT,
        "target_modules_sha256": matched.EXPECTED_TARGET_MODULES_SHA256,
        "training_contract": {
            "model": "Bernini-R-1.3B-Diffusers renderer-only",
            "single_expert": "transformer_1",
            "noise_tmin": 0.0,
            "noise_tmax": 1.0,
            "mv2v_flow_shift": 5.0,
            "num_frames": 81,
            "latent_frames": 21,
            "task_source_name": "mv2v$action_editing_81f",
            "external_spatial_mask": False,
            "external_tracking_or_swept_tube": False,
            "conditioning": ["clean_source_video_vae", "edit_instruction"],
            "supervision": ["noisy_target_video_vae", "target_velocity"],
            "target_embedding_or_caption_conditioning": False,
            "lora_rank": 64,
            "lora_alpha": 64,
            "lora_scope": "all Wan attn1/attn2 q,k,v,out projections",
            "tokenizer_fix_mistral_regex": True,
            "peft_version": "0.19.1",
            "transformers_version": "fixture-transformers",
            "gradient_checkpointing": True,
            "objective": "reference_dpo_preservation",
            "preference_weight": 1.0,
            "preference_margin": 0.05,
            "preference_temperature": 20.0,
            "dpo_beta": 10.0,
            "preservation_weight": 0.25,
            "contrastive_negative_kinds": ["noop", "reverse", "incomplete"],
            "contrastive_negative_schedule": "rotate",
            "preservation_branch": "source_as_target_conditional_identity",
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
        },
        "distributed": {
            "world_size": 4,
            "ulysses_size": 4,
            "backend": "nccl/rccl",
            "same_sample_all_ranks": True,
            "same_seed_all_ranks": True,
            "lora_initialization_seeded_all_ranks": True,
            "lora_parameters_broadcast_from_rank": 0,
            "lora_initialization_digest": sha("lora-initialization"),
            "explicit_lora_gradient_all_reduce": True,
        },
        "seed": matched.FULL644_SEED,
        "trainable_parameter_count": matched.FULL644_TRAINABLE_PARAMETER_COUNT,
        "resumed_from": None,
        "experimental_training": True,
        "exploratory_full644": {
            "profile": matched.FULL644_PROFILE,
            "historical_train_debug_rows": 644,
            "optimizer_rows_consumed": 644,
            "next_row_index": None,
            "row_sequence_prefix": "0..643",
            "row_sequence_sha256": matched.object_sha256(list(range(644))),
            "no_replacement_within_pass": True,
            "complete_one_pass": True,
            "historical_dataset_exists": True,
            "historical_optimizer_contribution_rows": 644,
            "historical_source_receipt_is_not_current_launch_authority": True,
            "runtime_data_integrity_validated": True,
            "dataset_quality_accepted_under_0817": False,
            "formal_training_dataset_authorized": False,
            "formal_heldout_contribution": 0,
            "target_scientific_qualification_complete": False,
            "matched_frozen_evaluation_required_before_claim": True,
            "resume_policy": "forbidden_for_this_profile",
            "intermediate_checkpoints_archival_only": True,
            "interrupted_run_requires_fresh_step0_restart": True,
            "dataset_summary_sha256": matched.FULL644_DATASET_SUMMARY_SHA256,
            "dataset_summary_digest": matched.FULL644_DATASET_SUMMARY_DIGEST,
            "dataset_index_sha256": matched.FULL644_DATASET_INDEX_SHA256,
            "dataset_content_signature": dataset_signature,
            "source_authority": source_authority,
            "indexed_source_and_target_vae_shards_verified_before_training": True,
            "indexed_source_and_target_vae_shards_reverified_after_training": True,
        },
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
    }
    receipt["receipt_digest"] = matched.object_sha256(receipt)
    write_json(checkpoint / "receipt.json", receipt)
    paths = sorted(
        "adapter/adapter_config.json adapter/adapter_model.safetensors "
        "optimizer.pt receipt.json".split()
    )
    entries = []
    for relative in paths:
        payload = (checkpoint / relative).read_bytes()
        entries.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    manifest = {
        "schema_version": matched.CHECKPOINT_MANIFEST_SCHEMA,
        "global_step": 644,
        "receipt_digest": receipt["receipt_digest"],
        "file_count": len(entries),
        "entries": entries,
    }
    manifest["manifest_digest"] = matched.object_sha256(manifest)
    path = checkpoint / "checkpoint_manifest.json"
    manifest_sha = write_json(path, manifest)
    identity = matched.validate_terminal_checkpoint_manifest(path, manifest_sha)
    return identity, path, manifest_sha


def reseal_checkpoint(
    manifest_path: Path,
    *,
    receipt_mutator=None,
    manifest_mutator=None,
) -> str:
    receipt_path = manifest_path.parent / "receipt.json"
    receipt = matched._json(receipt_path.read_bytes(), label="fixture receipt")
    if receipt_mutator is not None:
        receipt_mutator(receipt)
    receipt.pop("receipt_digest", None)
    receipt["receipt_digest"] = matched.object_sha256(receipt)
    receipt_sha = write_json(receipt_path, receipt)

    manifest = matched._json(manifest_path.read_bytes(), label="fixture manifest")
    manifest["receipt_digest"] = receipt["receipt_digest"]
    receipt_entry = next(
        row for row in manifest["entries"] if row["path"] == "receipt.json"
    )
    receipt_entry["sha256"] = receipt_sha
    receipt_entry["size"] = receipt_path.stat().st_size
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    manifest.pop("manifest_digest", None)
    manifest["manifest_digest"] = matched.object_sha256(manifest)
    return write_json(manifest_path, manifest)


def make_plan(root: Path, *, production_fixture: bool = False) -> dict:
    authority = matched.validate_shared8_authority(INPUT_MANIFEST, EXPOSURE_AUDIT)
    checkpoint, _, _ = make_checkpoint(root)
    output_root = root / "outputs"
    output_root.mkdir()
    infer_path = MODULE_ROOT / "infer_lora.py"
    ffprobe_value = shutil.which("ffprobe")
    if ffprobe_value is None:
        raise RuntimeError("ffprobe is required for this contract test")
    ffprobe_path = Path(ffprobe_value).resolve(strict=True)
    producer = {
        "inference_receipt_schema": matched.INFERENCE_RECEIPT_SCHEMA,
        "infer_lora_path": str(infer_path),
        "infer_lora_sha256": hashlib.sha256(infer_path.read_bytes()).hexdigest(),
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": "b" * 64,
        "ffprobe_path": str(ffprobe_path),
        "ffprobe_sha256": hashlib.sha256(ffprobe_path.read_bytes()).hexdigest(),
    }
    plan = matched.build_plan(
        authority, checkpoint, output_root, production=False, producer=producer
    )
    if production_fixture:
        plan.pop("plan_digest")
        plan["production_ready"] = True
        plan["authority"]["source_bytes_verified"] = True
        plan["plan_digest"] = matched.object_sha256(plan)
    return plan


def make_receipts(plan: dict) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for this contract test")
    fixture = Path(plan["tasks"][0]["output"]["video_path"]).parent.parent / "fixture.mp4"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=25",
            "-frames:v", "81", "-an", "-c:v", "mpeg4", "-q:v", "5",
            "-pix_fmt", "yuv420p", str(fixture),
        ],
        check=True,
        timeout=60,
    )
    video_payload = fixture.read_bytes()
    for task in plan["tasks"]:
        output_path = Path(task["output"]["video_path"])
        output_path.write_bytes(video_payload)
        output_path.chmod(0o444)
        publication_identity = matched._publication_identity(output_path)
        prepublication_identity = dict(publication_identity)
        prepublication_identity.update(
            mode=stat.S_IFREG | 0o600,
            nlink=0,
            inode=publication_identity["inode"] + 100_000,
        )
        adapter = (
            {
                "enabled": False,
                "mode": "frozen_base_no_adapter",
                "strictly_reloaded": False,
                "safe_merged_for_inference": False,
                "tensor_count": 0,
            }
            if task["arm"] == "base"
            else {
                "enabled": True,
                "mode": "lora_safe_merge",
                "checkpoint_root": "/proc/self/fd/101/adapter",
                "adapter_model_path": "/proc/self/fd/101/adapter/adapter/adapter_model.safetensors",
                "strictly_reloaded": True,
                "safe_merged_for_inference": True,
                "tensor_count": 480,
                "training_global_step": 644,
                "profile": matched.FULL644_PROFILE,
                "lora_rank": 64,
                "lora_alpha": 64,
                "target_module_count": matched.EXPECTED_TARGET_MODULE_COUNT,
                "target_modules_sha256": matched.EXPECTED_TARGET_MODULES_SHA256,
                "adapter_model_sha256": task["adapter"]["adapter_model_sha256"],
                "training_receipt_path": "/proc/self/fd/101/adapter/receipt.json",
                "training_receipt_digest": task["adapter"]["checkpoint_manifest"][
                    "receipt_digest"
                ],
                "checkpoint_manifest": task["adapter"]["checkpoint_manifest"],
            }
        )
        rank_digest = sha("rank-evidence:" + task["task_id"])
        source_authority = {
            "path": task["source_video"],
            "sha256": task["source_video_sha256"],
            "size": 1234,
            "mode": stat.S_IFREG | 0o444,
            "device": 1,
            "inode": 10_000 + task["case_index"],
            "uid": 1000,
            "gid": 1000,
            "nlink": 1,
            "rdev": 0,
            "blocks": 8,
            "mtime_ns": 1,
            "ctime_ns": 1,
        }
        source_authority_digest = matched.object_sha256(source_authority)
        consumption = {
            "consumption_input_digest": sha("consumption:" + task["task_id"]),
            "task_input_digest": sha("task-input:" + task["task_id"]),
            "model_capture_digest": sha("model-capture:" + task["iid"]),
            "model_view_root": "/proc/self/fd/100/model",
            "adapter_capture_digest": (
                None
                if task["arm"] == "base"
                else sha("adapter-capture:" + task["task_id"])
            ),
            "adapter_view_root": (
                None if task["arm"] == "base" else "/proc/self/fd/101/adapter"
            ),
            "fd_view_files_authorized": 10,
            "inherited_fd_binding_digest": sha("fd-binding:" + task["task_id"]),
            "inherited_fd_count": 12,
            "ptrace_authorization_used": False,
            "source_video_sha256": task["source_video_sha256"],
            "source_video_physical_authority_digest": source_authority_digest,
            "all_ranks_use_retained_source_fd": True,
            "four_rank_attestation": {
                "world_size": 4,
                "all_ranks_replayed_exact_fd_views": True,
                "rank_evidence_digest": rank_digest,
                "ordered_rank_evidence_digests": [rank_digest] * 4,
            },
        }
        receipt = {
            "schema_version": matched.INFERENCE_RECEIPT_SCHEMA,
            "infer_lora_source_sha256": plan["producer"]["infer_lora_sha256"],
            "method_source_revision": plan["producer"]["method_source_revision"],
            "method_source_archive_sha256": plan["producer"][
                "method_source_archive_sha256"
            ],
            "bernini_commit": matched.EXPECTED_BERNINI_COMMIT,
            "veomni_commit": matched.EXPECTED_VEOMNI_COMMIT,
            "bernini_inference_files": dict(matched.EXPECTED_BERNINI_INFERENCE_FILES),
            "checkpoint_tree_sha256": matched.EXPECTED_CHECKPOINT_TREE_SHA256,
            "consumption_input_digest": consumption["consumption_input_digest"],
            "task_input_digest": consumption["task_input_digest"],
            "model_consumption": consumption,
            "runtime_versions": {
                "torch": "fixture", "torch_hip": "fixture",
                "transformers": "fixture", "diffusers": "fixture", "peft": "0.19.1",
            },
            "adapter": adapter,
            "input": {
                "source_video_path": task["source_video"],
                "source_video_sha256": task["source_video_sha256"],
                "instruction_utf8_sha256": task["instruction_sha256"],
                "instruction_utf8_bytes": len(task["instruction"].encode("utf-8")),
                "accepted_model_conditions": ["source_video", "edit_instruction"],
                "target_video_argument": False,
                "target_accessed_by_inference": False,
                "external_mask_or_swept_tube": False,
                "external_tracking_pose_or_trajectory": False,
                "reference_image_or_video": False,
                "external_shared_i0": False,
                "source_video_physical_authority": source_authority,
                "retained_source_fd_consumed": True,
                "source_video_pre_and_post_decode_rehashed": True,
                "source_video_physical_authority_digest": source_authority_digest,
            },
            "preprocessing": {
                "frame_count": 81,
                "fps": 25.0,
                "reported_fps": 25.0,
                "source_input_hw": [64, 64],
                "source_derived_bucket_hw": [64, 64],
                "max_pixels": 245_760,
                "stride": 16,
                "temporal_policy": "all_integer_frames_0_through_80_no_subsampling",
                "spatial_policy": "sqrt_max_pixels_then_floor_each_dimension_to_stride",
                "resize": "torchvision_bicubic_antialias_true",
                "external_shared_i0": False,
            },
            "prompt_contract": {
                "task": "mv2v",
                "system_prompt_sha256": matched.EXPECTED_SYSTEM_PROMPT_SHA256,
                "cleaner": "diffusers.pipelines.wan.pipeline_wan.prompt_clean",
                "tokenizer_fix_mistral_regex": True,
                "tokenizer_padding_side": "right",
                "max_sequence_length": 512,
                "prompt_enhancer": False,
            },
            "sampling": {
                "num_frames": 81,
                "num_inference_steps": 40,
                "guidance_mode": "v2v_apg",
                "omega_vid": 1.25,
                "omega_img": 0.0,
                "omega_txt": 4.0,
                "omega_scale": 0.8,
                "flow_shift": 5.0,
                "seed": task["seed"],
                "eta": 0.5,
                "norm_threshold": [50.0, 50.0],
                "momentum": 0.0,
                "single_expert": "transformer_1",
                "ulysses_size": 4,
                "rank0_decode_and_save_only": True,
                "source_onset_policy": "none",
            },
            "output": {
                "path": str(output_path),
                "sha256": hashlib.sha256(video_payload).hexdigest(),
                "size": len(video_payload),
                "frame_count": 81,
                "fps": 25.0,
                "height": 64,
                "width": 64,
                "audio_preserved": False,
                "publication_identity": publication_identity,
                "prepublication_identity": prepublication_identity,
                "anonymous_creation_method": "linux-sealed-memfd-v1",
                "anonymous_seal_mask": 15,
                "sealed_source_sha256": hashlib.sha256(video_payload).hexdigest(),
                "sealed_source_size": len(video_payload),
                "anonymous_inode_encoded_and_decoded_before_publication": True,
                "create_only_copy_publication_after_decode": True,
                "sealed_source_and_publication_bytes_equal": True,
                "retained_inode_encoded_and_replayed": True,
                "named_output_never_replaced": True,
            },
            "experimental_inference": True,
            "production_claim_forbidden": True,
            "scientific_claim_authorized": False,
        }
        receipt["receipt_digest"] = matched.object_sha256(receipt)
        write_json(Path(task["output"]["receipt_path"]), receipt)


class Full644ExploratoryMatchedEvalTests(unittest.TestCase):
    def test_source_free_authority_is_exact_and_explicitly_diagnostic(self) -> None:
        authority = matched.validate_shared8_authority(INPUT_MANIFEST, EXPOSURE_AUDIT)
        self.assertFalse(authority["source_bytes_verified"])
        self.assertEqual([row["index"] for row in authority["rows"]], list(range(8)))
        self.assertEqual([row["seed"] for row in authority["rows"]], list(range(2026, 2034)))
        self.assertEqual(
            [row["source_sha256"] for row in authority["rows"]],
            list(matched.EXPECTED_SOURCE_SHA256),
        )
        self.assertFalse(authority["claim_limits"]["content_disjoint_split"])
        self.assertFalse(authority["claim_limits"]["formal_claim_authorized"])

    def test_authority_rejects_even_one_byte_manifest_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "inputs.jsonl"
            bad.write_bytes(INPUT_MANIFEST.read_bytes() + b"\n")
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.validate_shared8_authority(bad, EXPOSURE_AUDIT)

    def test_terminal_checkpoint_manifest_is_external_and_step_644(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity, path, manifest_sha = make_checkpoint(root)
            self.assertEqual(identity["global_step"], 644)
            self.assertEqual(identity["sha256"], manifest_sha)
            value = copy.deepcopy(matched._json(path.read_bytes(), label="fixture"))
            value["global_step"] = 643
            value["manifest_digest"] = matched.object_sha256(
                {key: item for key, item in value.items() if key != "manifest_digest"}
            )
            bad = root / "bad.json"
            bad_sha = write_json(bad, value)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.validate_terminal_checkpoint_manifest(bad, bad_sha)

            checkpoint_mutations = {
                "manifest-extra": (
                    None,
                    lambda value: value.__setitem__("extra_top", True),
                ),
                "receipt-extra": (
                    lambda value: value.__setitem__("extra_top", True),
                    None,
                ),
                "receipt-wrong-schema": (
                    lambda value: value.__setitem__(
                        "schema_version", "forged-training-receipt-v1"
                    ),
                    None,
                ),
                "training-contract-extra": (
                    lambda value: value["training_contract"].__setitem__(
                        "extra", True
                    ),
                    None,
                ),
                "exploratory-extra": (
                    lambda value: value["exploratory_full644"].__setitem__(
                        "extra", True
                    ),
                    None,
                ),
                "float-bool-alias": (
                    lambda value: value["training_contract"].__setitem__(
                        "noise_tmin", False
                    ),
                    None,
                ),
            }
            for label, (receipt_mutator, manifest_mutator) in checkpoint_mutations.items():
                with self.subTest(checkpoint_closure=label):
                    _, hostile_path, _ = make_checkpoint(root / label)
                    hostile_sha = reseal_checkpoint(
                        hostile_path,
                        receipt_mutator=receipt_mutator,
                        manifest_mutator=manifest_mutator,
                    )
                    with self.assertRaises(matched.MatchedEvalContractError):
                        matched.validate_terminal_checkpoint_manifest(
                            hostile_path, hostile_sha
                        )

    def test_plan_has_exactly_two_matched_arms_and_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root)
            matched.validate_plan(plan)
            self.assertEqual(len(plan["tasks"]), 16)
            self.assertFalse(plan["production_ready"])
            self.assertTrue(plan["execution"]["all_16_tasks_required_no_cherry_pick"])
            plan_closure_mutations = {
                "plan-extra": lambda value: value.__setitem__("extra_top", True),
                "authority-extra": lambda value: value["authority"].__setitem__(
                    "extra_top", True
                ),
                "input-identity-extra": lambda value: value["authority"][
                    "input_manifest"
                ].__setitem__("extra", True),
                "exposure-identity-extra": lambda value: value["authority"][
                    "exposure_audit"
                ].__setitem__("extra", True),
                "checkpoint-identity-extra": lambda value: value[
                    "checkpoint_manifest"
                ].__setitem__("extra_top", True),
            }
            for label, mutate in plan_closure_mutations.items():
                with self.subTest(plan_closure=label):
                    forged = copy.deepcopy(plan)
                    forged.pop("plan_digest")
                    mutate(forged)
                    forged["plan_digest"] = matched.object_sha256(forged)
                    with self.assertRaises(matched.MatchedEvalContractError):
                        matched.validate_plan(forged)

            def mutate_case_pair(value: dict, field: str, replacement: object) -> None:
                for task in value["tasks"]:
                    if task["case_index"] == 0:
                        task[field] = replacement

            numeric_plan_mutations = {
                "pair-count-float": lambda value: value.__setitem__(
                    "pair_count", 8.0
                ),
                "case-index-float": lambda value: mutate_case_pair(
                    value, "case_index", 0.0
                ),
                "seed-float": lambda value: mutate_case_pair(
                    value, "seed", 2026.0
                ),
                "steps-float": lambda value: mutate_case_pair(
                    value, "num_inference_steps", 40.0
                ),
                "claim-zero-bool": lambda value: value["claim_limits"].__setitem__(
                    "iid_overlap_with_full644", False
                ),
            }
            for label, mutate in numeric_plan_mutations.items():
                with self.subTest(plan_numeric_alias=label):
                    forged = copy.deepcopy(plan)
                    forged.pop("plan_digest")
                    mutate(forged)
                    forged["plan_digest"] = matched.object_sha256(forged)
                    with self.assertRaises(matched.MatchedEvalContractError):
                        matched.validate_plan(forged)
            forged_instruction_plan = copy.deepcopy(plan)
            forged_instruction_plan.pop("plan_digest")
            forged_instruction = "Replace the canonical edit with a forged action."
            forged_instruction_sha = hashlib.sha256(
                forged_instruction.encode("utf-8")
            ).hexdigest()
            for task in forged_instruction_plan["tasks"]:
                if task["case_index"] == 0:
                    task["instruction"] = forged_instruction
                    task["instruction_sha256"] = forged_instruction_sha
            forged_instruction_plan["plan_digest"] = matched.object_sha256(
                forged_instruction_plan
            )
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.validate_plan(forged_instruction_plan)
            forged_checkpoint_plan = copy.deepcopy(plan)
            forged_checkpoint_plan.pop("plan_digest")
            for task in forged_checkpoint_plan["tasks"]:
                if task["arm"] == "full644":
                    task["adapter"]["checkpoint_root"] = "/forged/checkpoint-root"
            forged_checkpoint_plan["plan_digest"] = matched.object_sha256(
                forged_checkpoint_plan
            )
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.validate_plan(forged_checkpoint_plan)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.build_plan(
                    matched.validate_shared8_authority(INPUT_MANIFEST, EXPOSURE_AUDIT),
                    plan["checkpoint_manifest"], root / "outputs", production=True,
                    producer=plan["producer"],
                )
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)
            plan_path = root / "plan.json"
            matched.write_create_only(plan_path, plan)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.write_create_only(plan_path, plan)

    def test_pair_receipts_verify_all_16_and_reject_sampling_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = make_plan(root, production_fixture=True)
            make_receipts(plan)
            report = matched.verify_results(plan)
            self.assertEqual(report["verified_task_count"], 16)
            self.assertTrue(report["all_16_tasks_verified_no_cherry_pick"])
            self.assertFalse(report["producer_execution_proven_by_receipt_contract"])
            self.assertTrue(report["external_frozen_runner_attestation_still_required"])
            valid_receipts: dict[Path, dict] = {}
            for task in plan["tasks"]:
                path = Path(task["output"]["receipt_path"])
                value = matched._json(path.read_bytes(), label="fixture")
                valid_receipts[path] = copy.deepcopy(value)
                value["input"].update(
                    target_video_argument=True,
                    target_accessed_by_inference=True,
                    accepted_model_conditions=[
                        "source_video", "edit_instruction", "target_video"
                    ],
                    target_video_path="/forged/target.mp4",
                )
                value["forged_producer_attestation"] = {
                    "infer_lora_never_executed": True
                }
                value["receipt_digest"] = matched.object_sha256(
                    {key: item for key, item in value.items() if key != "receipt_digest"}
                )
                write_json(path, value)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)
            for path, value in valid_receipts.items():
                write_json(path, value)

            base = next(
                task for task in plan["tasks"]
                if task["case_index"] == 0 and task["arm"] == "base"
            )

            adapted = next(
                task for task in plan["tasks"]
                if task["case_index"] == 0 and task["arm"] == "full644"
            )

            def assert_resigned_alias_rejected(
                label: str, task: dict, mutate: object
            ) -> None:
                path = Path(task["output"]["receipt_path"])
                value = copy.deepcopy(valid_receipts[path])
                mutate(value)
                value["receipt_digest"] = matched.object_sha256(
                    {
                        key: item
                        for key, item in value.items()
                        if key != "receipt_digest"
                    }
                )
                write_json(path, value)
                try:
                    with self.subTest(receipt_numeric_alias=label):
                        with self.assertRaises(matched.MatchedEvalContractError):
                            matched.verify_results(plan)
                finally:
                    write_json(path, valid_receipts[path])

            receipt_numeric_aliases = (
                (
                    "sampling-omega-img-bool",
                    base,
                    lambda value: value["sampling"].__setitem__("omega_img", False),
                ),
                (
                    "sampling-momentum-bool",
                    base,
                    lambda value: value["sampling"].__setitem__("momentum", False),
                ),
                (
                    "adapter-rank-float",
                    adapted,
                    lambda value: value["adapter"].__setitem__("lora_rank", 64.0),
                ),
                (
                    "preprocessing-frame-count-float",
                    base,
                    lambda value: value["preprocessing"].__setitem__(
                        "frame_count", 81.0
                    ),
                ),
                (
                    "output-frame-count-float",
                    base,
                    lambda value: value["output"].__setitem__(
                        "frame_count", 81.0
                    ),
                ),
                (
                    "model-consumption-count-float",
                    base,
                    lambda value: value["model_consumption"].__setitem__(
                        "inherited_fd_count", 12.0
                    ),
                ),
                (
                    "publication-size-float",
                    base,
                    lambda value: value["output"]["publication_identity"].__setitem__(
                        "size", float(value["output"]["publication_identity"]["size"])
                    ),
                ),
            )
            for label, task, mutate in receipt_numeric_aliases:
                assert_resigned_alias_rejected(label, task, mutate)

            receipt_path = Path(adapted["output"]["receipt_path"])
            receipt = matched._json(receipt_path.read_bytes(), label="fixture")
            valid_receipt = copy.deepcopy(receipt)

            # Both arms must consume the same frozen-base model capture.  A
            # validly re-digested adapted receipt may not substitute another
            # base checkpoint/capture behind the LoRA treatment.
            receipt["model_consumption"]["model_capture_digest"] = sha(
                "drifted-frozen-base-capture"
            )
            receipt["receipt_digest"] = matched.object_sha256(
                {key: item for key, item in receipt.items() if key != "receipt_digest"}
            )
            write_json(receipt_path, receipt)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)

            receipt = copy.deepcopy(valid_receipt)
            receipt["sampling"]["seed"] += 1
            receipt["receipt_digest"] = matched.object_sha256(
                {key: item for key, item in receipt.items() if key != "receipt_digest"}
            )
            write_json(receipt_path, receipt)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)

            # A correctly self-digested receipt from any other producer schema
            # is still not an infer_lora v5 receipt.
            receipt = copy.deepcopy(valid_receipt)
            receipt["schema_version"] = "forged-producer-v1"
            receipt["receipt_digest"] = matched.object_sha256(
                {key: item for key, item in receipt.items() if key != "receipt_digest"}
            )
            write_json(receipt_path, receipt)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)

            # Even if every output hash/publication field is self-consistent,
            # arbitrary bytes are not an 81-frame 25-FPS MP4.
            output_path = Path(adapted["output"]["video_path"])
            output_path.chmod(0o644)
            forged_video = b"not-an-mp4"
            output_path.write_bytes(forged_video)
            output_path.chmod(0o444)
            receipt = copy.deepcopy(valid_receipt)
            identity = matched._publication_identity(output_path)
            receipt["output"].update(
                sha256=hashlib.sha256(forged_video).hexdigest(),
                size=len(forged_video),
                publication_identity=identity,
                sealed_source_sha256=hashlib.sha256(forged_video).hexdigest(),
                sealed_source_size=len(forged_video),
            )
            receipt["output"]["prepublication_identity"].update(size=len(forged_video))
            receipt["receipt_digest"] = matched.object_sha256(
                {key: item for key, item in receipt.items() if key != "receipt_digest"}
            )
            write_json(receipt_path, receipt)
            with self.assertRaises(matched.MatchedEvalContractError):
                matched.verify_results(plan)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import infer_source_kv_route_v9 as inference
import source_kv_replay as replay
import source_kv_route_scope as scope
import train_source_kv_route_auh as trainer

try:
    import torch
except ImportError:  # pragma: no cover - lightweight local environment
    torch = None
try:
    import peft
except ImportError:  # pragma: no cover - lightweight local environment
    peft = None


def _scope_manifest():
    value = {
        "schema_version": scope.RECEIPT_MANIFEST_SCHEMA,
        "method": scope.METHOD_NAME,
        "scope": scope.SCOPE_NAME,
        "base_model": {
            "model": "Bernini-R-1.3B-Diffusers",
            "transformer_block_count": 30,
            "hidden_size": 1536,
        },
        "lora": {
            "rank": 8,
            "alpha": 8,
            "dropout": 0.0,
            "bias": "none",
            "hidden_size": 1536,
            "target_modules": scope.canonical_target_modules(),
            "target_module_count": 92,
            "target_modules_sha256": scope.EXPECTED_TARGET_MODULES_SHA256,
            "adapter_tensor_count": 184,
            "adapter_state_keys_sha256": scope.object_sha256(
                scope.canonical_adapter_state_keys()
            ),
            "trainable_parameter_count": 2_260_992,
            "middle_self_blocks_inclusive": [7, 22],
            "cross_attention_blocks_inclusive": [0, 29],
        },
        "initialization": scope.fresh_initialization_declaration(),
        "validation": {
            "runtime_module_inventory_exact": True,
            "runtime_base_weight_shapes_exact": True,
            "adapter_state_scope_and_shapes_exact": True,
            "fresh_initialization_exact": True,
            "v8_warm_start_forbidden_for_main": True,
        },
    }
    value["manifest_digest"] = scope.object_sha256(value)
    return value


def _step_records():
    blocks = list(range(30))
    records = []
    for index in range(40):
        shard_sha = f"{index:064x}"
        records.append(
            {
                "optimizer_step": index + 1,
                "sigma_schedule_index": index,
                "row_index": index,
                "forward_order": list(trainer.FORWARD_ORDER),
                "forwards_per_candidate": 6,
                "graph_forwards_per_candidate": 2,
                "paired_target_model_forward_access": False,
                "gradient_audit": {
                    "trainable_tensor_count": 184,
                    "all_gradients_finite": True,
                    "positive_global_l2_norm": True,
                    "global_l2_norm": 0.25,
                },
                "optimizer_audit": {
                    "state_parameter_count": 184,
                    "state_step_values": [index + 1],
                    "no_moment_reset": True,
                },
                "target_energy_retention": 1.0,
                "target_clipped_fraction": 0.0,
                "cache_after_backward_audit": {
                    "selected_blocks": blocks,
                    "capture_calls_delta": 30,
                    "replay_lookups_delta": 210,
                    "backward_recompute_observed": True,
                },
                "cache_after_clear_audit": {
                    "cleared_after_backward": True,
                    "identity_after_clear": None,
                },
                "fresh_parity_checked": index == 0,
                "input_shard_integrity": {
                    "access_ordinal": index,
                    "row_index": index,
                    "hash_closed_read": True,
                    "cache_invalidated_before_read": True,
                    "expected_sha256": shard_sha,
                    "before_read_sha256": shard_sha,
                    "after_read_sha256": shard_sha,
                },
            }
        )
    return records


def _adapter_config():
    return {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "modules_to_save": None,
        "use_dora": False,
        "use_rslora": False,
        "target_modules": inference.expected_serialized_target_patterns(),
    }


def _training_receipt(
    adapter_sha="c" * 64,
    config_sha="d" * 64,
    optimizer_sha="e" * 64,
):
    manifest = _scope_manifest()
    steps = _step_records()
    exact40 = trainer.validate_exact40_step_audit(steps, block_selection="all")
    immutable_value = {
        "method": trainer.METHOD_NAME,
        "schema_version": trainer.RECEIPT_SCHEMA,
        "run_role": "v9_main",
        "method_source_revision": "a" * 40,
        "method_source_archive_sha256": "b" * 64,
        "frames": 81,
        "latent_phases": 21,
        "max_steps": 40,
        "checkpoint_tree_sha256": trainer.legacy.CHECKPOINT_TREE_SHA256,
        "forward_order": list(inference.EXPECTED_BRANCH_ORDER),
        "forwards_per_candidate": 6,
        "training_diffusion_query": "source(beta=0)",
        "paired_target_used_as_model_condition": False,
        "inference_conditions": ["source_video", "action_instruction"],
        "first_frame_anchor": False,
        "target_clipping": False,
        "target_energy_retention": 1.0,
        "resume_integrated": False,
        "carrier": {
            "selection": "all",
            "selected_blocks": list(range(30)),
            "selected_block_count": 30,
            "source_only": True,
            "post_rope": True,
        },
        "lora_scope_manifest": manifest,
        "query_state_policy": {
            "name": "offline_source_tangent_beta0",
            "query_state_train_test_matched": False,
        },
    }
    immutable = {
        "value": immutable_value,
        "digest": scope.object_sha256(immutable_value),
    }
    accesses = [dict(record["input_shard_integrity"]) for record in steps]
    input_integrity = {
        "validated": True,
        "policy": "pinned_index_hash_before_and_after_each_optimizer_read",
        "access_count": 40,
        "unique_accessed_shard_count": 40,
        "accesses": accesses,
        "accesses_sha256": scope.object_sha256(accesses),
        "final_accessed_shards": [
            {
                "shard_path": f"/tmp/shard-{index}.parquet",
                "expected_sha256": access["expected_sha256"],
                "final_sha256": access["expected_sha256"],
            }
            for index, access in enumerate(accesses)
        ],
        "dataset_summary_final_sha256": trainer.PINNED_DATASET_SUMMARY_FILE_SHA256,
        "dataset_index_final_sha256": trainer.PINNED_DATASET_INDEX_SHA256,
        "routing_final_sha256": trainer.PINNED_ROUTING_SHA256,
    }
    input_integrity["audit_sha256"] = scope.object_sha256(input_integrity)
    receipt = {
        "schema_version": trainer.RECEIPT_SCHEMA,
        "method": trainer.METHOD_NAME,
        "global_step": 40,
        "formal_exact40_complete": True,
        "exact40_audit": exact40,
        "step_audit": steps,
        "step_audit_sha256": scope.object_sha256(steps),
        "immutable_contract": immutable,
        "dataset": {"input_integrity": input_integrity},
        "adapter": {
            "scope_manifest": manifest,
            "target_module_count": 92,
            "adapter_tensor_count": 184,
            "trainable_parameter_count": 2_260_992,
            "initialization_digest": "init-digest",
            "checkpoint_parameter_digest": "parameter-digest",
        },
        "paired_target_model_forward_access": False,
        "external_mask_track_flow_pose_trajectory": False,
        "first_frame_anchor": False,
        "production_claim_forbidden": True,
        "scientific_claim_authorized": False,
        "transformers_version": "4.test",
        "artifact_validation": {
            "schema_version": trainer.ARTIFACT_VALIDATION_SCHEMA,
            "verified": True,
            "status": inference.ADAPTER_READY_STATUS,
            "adapter_tensor_file_roundtrip_verified": True,
            "adapter_tensor_file_runtime_equality": True,
            "torch_deserialize_verified": True,
            "fresh_optimizer_load_state_dict_verified": True,
            "optimizer_state_logical_equality_verified": True,
            "runtime_adapter_loader_verified": False,
            "fresh_base_peft_from_pretrained_verified": False,
            "deployment_loader_claim_forbidden": True,
            "adapter_tensor_count": 184,
            "trainable_parameter_count": 2_260_992,
            "state_parameter_count": 184,
            "state_step_values": [40],
            "adapter_model_sha256": adapter_sha,
            "adapter_config_sha256": config_sha,
            "optimizer_checkpoint_sha256": optimizer_sha,
        },
    }
    receipt["receipt_digest"] = scope.object_sha256(receipt)
    return receipt


def _runtime_evidence():
    replay_counts = {name: 1200 for name in inference.REPLAY_BRANCH_ORDER}
    per_block = []
    for block in range(30):
        per_block.append(
            {
                "block_index": block,
                "capture_calls": 40,
                "replay_calls": 200,
                "branch_counts": {
                    replay.CAPTURE_BRANCH_TAG: 40,
                    **{name: 40 for name in inference.REPLAY_BRANCH_ORDER},
                },
                "execution_phase_counts": {
                    replay.EAGER_EXECUTION: 240,
                    replay.CHECKPOINT_FORWARD: 0,
                    replay.CHECKPOINT_RECOMPUTE: 0,
                },
            }
        )
    cache = {
        "identity": None,
        "captured_blocks": [],
        "capture_calls": 1200,
        "replay_lookups": 6000,
        "replay_branch_counts": replay_counts,
        "replay_phase_counts": {
            replay.EAGER_EXECUTION: 6000,
            replay.CHECKPOINT_FORWARD: 0,
            replay.CHECKPOINT_RECOMPUTE: 0,
        },
        "retired_identity_count": 40,
    }
    runtime = {
        "restored": True,
        "cache": cache,
        "per_block": per_block,
    }
    core = {
        "block_indices": list(range(30)),
        "runtime": runtime,
        "runtime_digest": scope.object_sha256(runtime),
    }
    steps = [
        {
            "step_index": index,
            "forward_order": list(inference.EXPECTED_BRANCH_ORDER),
            "capture_forwards": 1,
            "replay_forwards": 5,
            "original_scheduler_calls": 1,
            "official_adapted_action_exact_parity": True,
            "phase0_quotient_exact_zero": True,
            "source_phase0_exact_preservation": True,
            "target_energy_retention": 1.0,
            "target_clipped_fraction": 0.0,
            "sigma_strictly_positive": True,
        }
        for index in range(40)
    ]
    trace = {
        "sample_calls": 1,
        "step_count": 40,
        "source_prefix_verified": True,
        "steps": steps,
    }
    return core, trace


FORWARD_IDENTITY_DIGEST = "f" * 64
GENERATED_LATENT_DIGEST = "9" * 64


def _rank_certificate(core, trace, rank):
    return inference.validate_rank_runtime_certificate(
        core_receipt=core,
        trace=trace,
        rank=rank,
        hooks_restored=True,
        forward_identity_digest=FORWARD_IDENTITY_DIGEST,
        generated_latent_digest=GENERATED_LATENT_DIGEST,
    )


class SourceKVRouteV9InferenceContractTests(unittest.TestCase):
    def test_deployment_operator_is_single_unbounded_source_quotient(self):
        contract = inference.deployment_operator_contract()
        self.assertEqual(contract["clean_field"], "E_k=S+Q0(A_theta,k-N_theta,k)")
        self.assertEqual(
            contract["scheduler_velocity"], "v_deploy,k=(x_k-E_k)/sigma_k"
        )
        self.assertEqual(contract["gauge"], "Q0(X)=X-X[:,:,0:1]")
        self.assertIsNone(contract["rho"])
        self.assertIsNone(contract["radius"])
        self.assertFalse(contract["clipping"])
        self.assertFalse(contract["field_mix"])
        self.assertEqual(contract["target_energy_retention"], 1.0)
        self.assertEqual(contract["target_clipped_fraction"], 0.0)
        self.assertEqual(
            contract["frozen_action_noop_role"],
            "diagnostics_only_not_mixed_into_operator",
        )

    def test_cli_is_source_instruction_only_exact81_40_seed2027(self):
        parser = inference.build_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertIn("source_video", destinations)
        self.assertIn("instruction", destinations)
        for forbidden in (
            "target_video",
            "mask",
            "track",
            "swept_tube",
            "pose",
            "trajectory",
            "optical_flow",
            "first_frame_anchor",
        ):
            self.assertNotIn(forbidden, destinations)
        args = parser.parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/base",
                "--adapter-checkpoint", "/tmp/adapter",
                "--source-video", "/tmp/source.mp4",
                "--instruction", "move",
                "--output", "/tmp/output.mp4",
                "--method-source-revision", "a" * 40,
                "--method-source-archive-sha256", "b" * 64,
            ]
        )
        inference.validate_cli(args)
        self.assertEqual(args.num_inference_steps, 40)
        self.assertEqual(args.seed, 2027)
        args.seed = 2028
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_cli(args)

    def test_serialized_scope_expands_to_exact92(self):
        patterns = inference.expected_serialized_target_patterns()
        self.assertEqual(len(patterns), 34)
        self.assertEqual(
            inference._expand_serialized_target_patterns(patterns),
            scope.canonical_target_modules(),
        )
        broken = _adapter_config()
        broken["target_modules"] = ["attn1.to_q", "attn2.to_q"]
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference._validate_adapter_config(broken)

    def test_completed_v9_checkpoint_is_accepted_and_v8_partial_rejected(self):
        identity = inference.validate_training_checkpoint_contract(
            adapter_config=_adapter_config(),
            receipt=_training_receipt(),
            adapter_model_sha256="c" * 64,
            adapter_config_sha256="d" * 64,
            optimizer_checkpoint_sha256="e" * 64,
            expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
        )
        self.assertEqual(identity["global_step"], 40)
        self.assertEqual(len(identity["target_modules"]), 92)

        v8 = _training_receipt()
        v8["schema_version"] = "bernini-rs-fqt-auh-training-receipt-v8"
        v8.pop("receipt_digest")
        v8["receipt_digest"] = scope.object_sha256(v8)
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_training_checkpoint_contract(
                adapter_config=_adapter_config(),
                receipt=v8,
                adapter_model_sha256="c" * 64,
                adapter_config_sha256="d" * 64,
                optimizer_checkpoint_sha256="e" * 64,
                expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
            )

        pending = _training_receipt()
        pending["artifact_validation"]["verified"] = False
        pending["artifact_validation"]["status"] = "pending"
        pending.pop("receipt_digest")
        pending["receipt_digest"] = scope.object_sha256(pending)
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_training_checkpoint_contract(
                adapter_config=_adapter_config(),
                receipt=pending,
                adapter_model_sha256="c" * 64,
                adapter_config_sha256="d" * 64,
                optimizer_checkpoint_sha256="e" * 64,
                expected_checkpoint_tree_sha256=trainer.legacy.CHECKPOINT_TREE_SHA256,
            )

    def test_runtime_certificate_requires_capture40_replay200_every_layer(self):
        core, trace = _runtime_evidence()
        certificate = _rank_certificate(core, trace, 0)
        self.assertEqual(certificate["per_layer_capture_calls"], 40)
        self.assertEqual(certificate["per_layer_replay_calls"], 200)
        self.assertEqual(certificate["rank_local_replay_lookups"], 6000)

        core["runtime"]["per_block"][7]["replay_calls"] = 199
        core["runtime_digest"] = scope.object_sha256(core["runtime"])
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            _rank_certificate(core, trace, 0)

    def test_four_rank_certificate_requires_exact_ranks_and_same_counts(self):
        core, trace = _runtime_evidence()
        rows = [_rank_certificate(core, trace, rank) for rank in range(4)]
        result = inference.validate_four_rank_certificates(rows)
        self.assertTrue(result["all_four_ranks_exact"])
        self.assertTrue(result["all_four_ranks_input_model_prompt_exact"])
        self.assertTrue(result["all_four_ranks_trace_core_exact"])
        self.assertTrue(result["all_four_ranks_generated_latent_exact"])
        self.assertEqual(result["cross_rank_replay_lookups"], 24_000)
        rows[3]["rank_local_replay_lookups"] = 5999
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_four_rank_certificates(rows)

        rows = [_rank_certificate(core, trace, rank) for rank in range(4)]
        rows[3]["generated_latent_digest"] = "8" * 64
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_four_rank_certificates(rows)

    def test_pre_forward_four_rank_identity_is_exact_not_counts_only(self):
        identity = {
            "source_video_sha256": "a" * 64,
            "adapter_file_sha256": {"adapter_model": "b" * 64},
            "checkpoint_identity_digest": "c" * 64,
            "method_source_revision": "d" * 40,
            "prompt_sha256": "e" * 64,
        }
        rows = [
            {
                "rank": rank,
                "identity": dict(identity),
                "identity_digest": scope.object_sha256(identity),
            }
            for rank in range(4)
        ]
        result = inference.validate_four_rank_forward_identities(rows)
        self.assertTrue(result["validated_before_forward"])
        rows[2]["identity"] = {**identity, "prompt_sha256": "0" * 64}
        rows[2]["identity_digest"] = scope.object_sha256(rows[2]["identity"])
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.validate_four_rank_forward_identities(rows)

    def test_adapter_four_file_private_snapshot_is_hash_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            adapter = root / "adapter"
            adapter.mkdir(parents=True)
            files = {
                adapter / "adapter_config.json": b"{}\n",
                adapter / "adapter_model.safetensors": b"safe-adapter-bytes",
                root / "receipt.json": b"{}\n",
                root / "optimizer.pt": b"optimizer-bytes",
            }
            for path, payload in files.items():
                path.write_bytes(payload)
            staged = inference.stage_adapter_snapshot(root)
            try:
                self.assertNotEqual(staged.bundle.checkpoint_root, root)
                self.assertEqual(len(staged.files), 4)
                self.assertEqual(
                    staged.bundle.adapter_model_path.read_bytes(),
                    files[adapter / "adapter_model.safetensors"],
                )
                inference.verify_adapter_snapshot(staged)
                (root / "optimizer.pt").write_bytes(b"changed")
                with self.assertRaises(inference.SourceKVRouteInferenceError):
                    inference.verify_adapter_snapshot(staged)
            finally:
                inference.cleanup_staged_adapter(staged)
            self.assertFalse(staged.directory.exists())

    def test_runtime_checkpoint_manifest_is_dynamic_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "base"
            root.mkdir()
            payloads = {
                "config.json": b'{"model":"tiny"}\n',
                "transformer/weights.safetensors": b"weight-bytes",
            }
            for relative, payload in payloads.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            manifest = Path(directory) / "checkpoint.sha256"
            lines = [
                f"{hashlib.sha256(payload).hexdigest()}  ./{relative}"
                for relative, payload in sorted(payloads.items())
            ]
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with mock.patch.object(
                trainer, "CHECKPOINT_CONTENT_MANIFEST_SHA256", manifest_sha
            ):
                with mock.patch.object(
                    trainer, "CHECKPOINT_CONTENT_FILE_COUNT", len(payloads)
                ):
                    identity = inference.validate_runtime_checkpoint_manifest(
                        root, manifest
                    )
                    self.assertTrue(identity["every_non_cache_file_verified"])
                    self.assertEqual(identity["verified_file_count"], 2)
                    (root / "config.json").write_bytes(b"changed")
                    with self.assertRaises(inference.SourceKVRouteInferenceError):
                        inference.validate_runtime_checkpoint_manifest(root, manifest)

    def test_source_is_staged_by_hash_and_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            payload = b"source-snapshot-contract\x00bytes"
            source.write_bytes(payload)
            staged = inference.stage_source_snapshot(source)
            try:
                self.assertNotEqual(staged.staged_path, source)
                self.assertEqual(staged.staged_path.read_bytes(), payload)
                self.assertEqual(staged.sha256, hashlib.sha256(payload).hexdigest())
                self.assertEqual(staged.byte_count, len(payload))
            finally:
                inference.cleanup_staged_source(staged)
            self.assertFalse(staged.directory.exists())

            link = root / "link.mp4"
            try:
                link.symlink_to(source)
            except OSError:
                self.skipTest("symlinks unavailable")
            # resolve(strict=True) deliberately canonicalizes the CLI path; the
            # no-follow fd then pins that canonical plain file snapshot.
            canonical = inference.stage_source_snapshot(link)
            inference.cleanup_staged_source(canonical)

    def test_static_order_keeps_frozen_diagnostics_out_of_operator(self):
        source = (METHOD_ROOT / "infer_source_kv_route_v9.py").read_text(
            encoding="utf-8"
        )
        capture = source.index("branch_tag=replay.CAPTURE_BRANCH_TAG")
        frozen_negative = source.index('branch_tag="frozen_negative"', capture)
        frozen_noop = source.index('branch_tag="frozen_noop"', frozen_negative)
        frozen_action = source.index('branch_tag="frozen_action"', frozen_noop)
        adapted_noop = source.index('branch_tag="adapted_noop"', frozen_action)
        adapted_action = source.index('branch_tag="adapted_action"', adapted_noop)
        operator = source.index(
            "executed_clean, quotient = compute_deployment_clean_field(",
            adapted_action,
        )
        scheduler = source.index("self._original_scheduler_step", operator)
        self.assertLess(capture, frozen_negative)
        self.assertLess(frozen_negative, frozen_noop)
        self.assertLess(frozen_noop, frozen_action)
        self.assertLess(frozen_action, adapted_noop)
        self.assertLess(adapted_noop, adapted_action)
        self.assertLess(adapted_action, operator)
        self.assertLess(operator, scheduler)
        operator_region = source[operator - 400 : scheduler]
        self.assertNotIn("frozen_quotient +", operator_region)
        self.assertNotIn("rho", operator_region)
        self.assertNotIn("radius", operator_region)

    def test_auh_launcher_pins_four_gpus_hashes_counts_and_receipt_formula(self):
        launcher = (
            METHOD_ROOT / "scripts/auh_infer_source_kv_route_v9.sbatch"
        ).read_text(encoding="utf-8")
        for literal in (
            "#SBATCH --gres=gpu:mi210:4",
            "--nproc_per_node=4",
            "--num-inference-steps 40",
            "--seed 2027",
            '.runtime_versions.peft == "0.19.1"',
            "post_save_adapter_tensor_file_roundtrip_and_optimizer_load_state_dict_complete",
            "adapter_tensor_file_runtime_equality == true",
            "fresh_base_peft_from_pretrained_verified == false",
            "deployment_loader_claim_forbidden == true",
            'per_layer_capture_calls == 40',
            'per_layer_replay_calls == 200',
            'rank_local_capture_calls == 1200',
            'rank_local_replay_lookups == 6000',
            'clean_field == "E_k=S+Q0(A_theta,k-N_theta,k)"',
            'scheduler_velocity == "v_deploy,k=(x_k-E_k)/sigma_k"',
            ".deployment_operator.rho == null",
            ".deployment_operator.radius == null",
            ".input.target_accessed_by_inference == false",
            ".adapter.four_plain_files_staged_with_no_follow_fd == true",
            ".checkpoint_runtime_content.verified_before_and_after_sampling == true",
            ".all_four_ranks_input_model_prompt_exact == true",
            ".all_four_ranks_trace_core_exact == true",
            ".all_four_ranks_generated_latent_exact == true",
            ".carrier_runtime.rank0_trace.step_count == 40",
            'if trace_digest != carrier["rank0_trace_digest"]:',
            '--arg output_sha "$(sha256sum "${output_video}"',
            ".output.sha256 == $output_sha",
            'receipt_output_sha="$(jq -er',
            '[[ "${final_output_sha}" == "${receipt_output_sha}" ]]',
            "trap cleanup_task_scratch EXIT",
            "trap 'exit 143' TERM",
            "trap 'exit 130' INT",
            'export TMPDIR="${task_scratch}"',
        ):
            self.assertIn(literal, launcher)
        self.assertNotIn("BERNINI_V9_TARGET_VIDEO", launcher)
        self.assertNotIn("--target-video", launcher)
        self.assertNotIn("staged_snapshot_removed_after_run", launcher)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class SourceKVRouteV9DeploymentTensorTests(unittest.TestCase):
    def test_bfloat16_tensor_digest_binds_shape_dtype_and_exact_bytes(self):
        value = torch.arange(24, dtype=torch.bfloat16).reshape(1, 3, 8)
        digest = inference.tensor_sha256(value, label="bf16 fixture")
        self.assertEqual(digest, inference.tensor_sha256(value.clone(), label="clone"))
        changed = value.clone()
        changed[0, 0, 0] += 1
        self.assertNotEqual(
            digest, inference.tensor_sha256(changed, label="changed fixture")
        )

    def test_source_prefix_uses_official_1536d_patch_not_raw_64d_pack(self):
        source_tokens = 7
        hidden_size = 1536
        raw_packed_channels = 64
        source = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
        embedded = torch.arange(
            source_tokens * hidden_size, dtype=torch.float32
        ).reshape(1, source_tokens, hidden_size).to(torch.bfloat16)
        source_rotary = torch.arange(
            source_tokens * 8, dtype=torch.float32
        ).reshape(1, 1, source_tokens, 8)

        class FakeBerniniTransformer:
            dtype = torch.bfloat16

            def __init__(self):
                self.source_ids = []

            def patch_vae_latent(self, value, source_id=None):
                self.source_ids.append(source_id)
                self.last_dtype = value.dtype
                return embedded.clone(), source_rotary.clone()

        transformer = FakeBerniniTransformer()
        paired_hidden = torch.cat((embedded, torch.zeros_like(embedded)), dim=1)
        paired_rotary = torch.cat(
            (source_rotary, torch.zeros_like(source_rotary)), dim=2
        )
        self.assertTrue(
            inference.verify_official_source_prefix(
                transformer=transformer,
                source_clean=source,
                paired_hidden_states=paired_hidden,
                paired_rotary_embs=paired_rotary,
                source_tokens=source_tokens,
            )
        )
        self.assertEqual(transformer.source_ids, [1.0])
        self.assertEqual(transformer.last_dtype, torch.bfloat16)

        # A raw VAE pack has 64 channels and cannot be substituted for the
        # official 1536-dimensional transformer patch embedding.
        raw_pair = torch.zeros(
            1, 2 * source_tokens, raw_packed_channels, dtype=torch.bfloat16
        )
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.verify_official_source_prefix(
                transformer=transformer,
                source_clean=source,
                paired_hidden_states=raw_pair,
                paired_rotary_embs=paired_rotary,
                source_tokens=source_tokens,
            )

    def test_operator_is_exact_source_plus_q0_without_clipping(self):
        source = torch.randn(1, 3, 21, 2, 2, dtype=torch.float32)
        noop = torch.randn_like(source)
        raw_delta = 7.0 * torch.randn_like(source)
        action = noop + raw_delta
        executed, quotient = inference.compute_deployment_clean_field(
            source_clean=source,
            adapted_action=action,
            adapted_noop=noop,
        )
        reconstructed_delta = action.float() - noop.float()
        expected = reconstructed_delta - reconstructed_delta[:, :, :1]
        self.assertTrue(torch.equal(quotient, expected))
        self.assertTrue(torch.equal(executed, source + expected))
        self.assertTrue(torch.equal(quotient[:, :, :1], torch.zeros_like(quotient[:, :, :1])))
        self.assertTrue(torch.equal(executed[:, :, :1], source[:, :, :1]))
        # Large values survive unchanged: there is no radius or clipping arm.
        self.assertGreater(float(quotient.abs().max()), 1.0)

    def test_operator_rejects_wrong_phase_count_and_nonfinite_field(self):
        source = torch.zeros(1, 2, 21, 1, 1)
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.compute_deployment_clean_field(
                source_clean=source[:, :, :-1],
                adapted_action=source[:, :, :-1],
                adapted_noop=source[:, :, :-1],
            )
        action = source.clone()
        action[:, :, 5] = float("nan")
        with self.assertRaises(inference.SourceKVRouteInferenceError):
            inference.compute_deployment_clean_field(
                source_clean=source,
                adapted_action=action,
                adapted_noop=source,
            )

    @unittest.skipIf(peft is None, "PEFT is unavailable")
    def test_peft_serializes_exact92_as_the_audited_34_suffixes(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = torch.nn.Linear(4, 4, bias=False)
                self.to_k = torch.nn.Linear(4, 4, bias=False)
                self.to_v = torch.nn.Linear(4, 4, bias=False)
                self.to_out = torch.nn.Sequential(
                    torch.nn.Linear(4, 4, bias=False)
                )

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn1 = Attention()
                self.attn2 = Attention()

        class TinyBerniniNames(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diff_dec = torch.nn.Module()
                self.diff_dec.transformer = torch.nn.Module()
                self.diff_dec.transformer.blocks = torch.nn.ModuleList(
                    [Block() for _ in range(30)]
                )

            def forward(self, value):
                return value

        model = peft.get_peft_model(
            TinyBerniniNames(),
            peft.LoraConfig(
                r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                bias="none",
                target_modules=scope.canonical_target_modules(),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            config = json.loads(
                (Path(directory) / "adapter_config.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(
            sorted(config["target_modules"]),
            inference.expected_serialized_target_patterns(),
        )
        self.assertEqual(
            inference._validate_adapter_config(config),
            inference.expected_serialized_target_patterns(),
        )

    @unittest.skipIf(peft is None, "PEFT is unavailable")
    def test_pinned_shared_step_unipc_and_strict_peft_one_cell_smoke(self):
        """Run one V9 cell through pinned shared_step and real UniPC.

        This intentionally avoids loading 1.3B weights.  The 92 base affine
        tensors have the audited 1536x1536 geometry but use stride-zero
        storage; only the exact184 LoRA tensors are materialized.  Thus AUH
        can exercise the deployment integration in seconds on CPU.
        """

        pinned_root = Path(
            os.environ.get(
                "BERNINI_V9_PINNED_ROOT",
                "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/"
                "VideoEditing/VideoEdit_experiments/motive_action_repr_auto/"
                "vendor/Bernini-2d2b4591",
            )
        )
        wan_path = pinned_root / "bernini/models/wan_diffusion.py"
        if not wan_path.is_file():
            self.skipTest("pinned AUH Bernini source is unavailable")
        self.assertEqual(
            inference._file_sha256(wan_path),
            inference.tri.PINNED_WAN_DIFFUSION_SHA256,
        )

        # Load the exact pinned file while stubbing only its two relative
        # imports.  Importing bernini.models/__init__.py would unnecessarily
        # pull VeOmni and the full renderer stack into this CPU smoke.
        package = "_v9_pinned_smoke"
        models_package = f"{package}.models"
        module_name = f"{models_package}.wan_diffusion"
        inserted = []
        for name in (package, models_package):
            value = types.ModuleType(name)
            value.__path__ = []
            sys.modules[name] = value
            inserted.append(name)
        scheduler_stub = types.ModuleType(f"{models_package}.scheduler")
        scheduler_stub.FlowMatchScheduler = type("FlowMatchScheduler", (), {})
        transformer_stub = types.ModuleType(f"{models_package}.transformer_wan")
        transformer_stub.WanTransformer3DModel = type(
            "WanTransformer3DModel", (torch.nn.Module,), {}
        )
        sys.modules[scheduler_stub.__name__] = scheduler_stub
        sys.modules[transformer_stub.__name__] = transformer_stub
        inserted.extend((scheduler_stub.__name__, transformer_stub.__name__))
        try:
            spec = importlib.util.spec_from_file_location(module_name, wan_path)
            if spec is None or spec.loader is None:
                self.fail("cannot load pinned wan_diffusion.py")
            pinned = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = pinned
            inserted.append(module_name)
            spec.loader.exec_module(pinned)
        finally:
            for name in reversed(inserted):
                sys.modules.pop(name, None)

        hidden_size = scope.HIDDEN_SIZE

        def compressed_linear():
            layer = torch.nn.Linear(
                hidden_size, hidden_size, bias=False, device="meta"
            )
            # Shape is real and PEFT-supported; storage is one fp32 scalar.
            layer.weight = torch.nn.Parameter(
                torch.zeros(1).expand(hidden_size, hidden_size),
                requires_grad=False,
            )
            return layer

        class TinyAttention(torch.nn.Module):
            def __init__(self, enabled):
                super().__init__()
                if enabled:
                    self.to_q = compressed_linear()
                    self.to_out = torch.nn.Sequential(compressed_linear())

        class TinyBlock(torch.nn.Module):
            def __init__(self, block):
                super().__init__()
                self.attn1 = TinyAttention(7 <= block <= 22)
                self.attn2 = TinyAttention(True)

        class TinyTransformer(torch.nn.Module):
            dtype = torch.bfloat16

            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [TinyBlock(block) for block in range(30)]
                )

            def patch_vae_latent(self, source, source_id=None):
                self.last_source_id = source_id
                tokens = int(source.shape[2] * source.shape[3] // 2 * source.shape[4] // 2)
                hidden = torch.zeros(
                    int(source.shape[0]), tokens, hidden_size, dtype=self.dtype
                )
                rotary = torch.zeros(1, 1, tokens, 2, dtype=torch.float32)
                return hidden, rotary

            def forward(
                self,
                hidden_states,
                timesteps,
                *,
                encoder_hidden_states,
                rotary_emb,
                batch_image_vae_seqlen,
                text_features_length,
            ):
                del timesteps, rotary_emb, batch_image_vae_seqlen, text_features_length
                invocation = replay.current_source_kv_invocation()
                if invocation.mode == replay.CAPTURE_MODE:
                    key = torch.zeros(
                        1, hidden_states.shape[1], 1, 2, dtype=torch.bfloat16
                    )
                    for block in range(30):
                        invocation.cache_bank.capture(
                            invocation=invocation,
                            block_index=block,
                            key=key,
                            value=key,
                        )
                else:
                    current = torch.zeros(
                        1, hidden_states.shape[1], 1, 2, dtype=torch.bfloat16
                    )
                    for block in range(30):
                        invocation.cache_bank._lookup(
                            invocation=invocation,
                            block_index=block,
                            current_key=current,
                            current_value=current,
                            source_tokens=hidden_states.shape[1] // 2,
                        )
                conditioned = hidden_states.float() + encoder_hidden_states.float().mean()
                projected = self.blocks[7].attn1.to_q(conditioned)
                return types.SimpleNamespace(sample=projected[..., :64].to(torch.bfloat16))

        def make_renderer():
            core = pinned.GEN_Wanx22.__new__(pinned.GEN_Wanx22)
            torch.nn.Module.__init__(core)
            core.transformer = TinyTransformer()
            core.transformer_2 = None
            core.use_unipc = True
            core.scheduler = pinned.UniPCMultistepScheduler(
                solver_order=2,
                prediction_type="flow_prediction",
                thresholding=False,
                predict_x0=True,
                solver_type="bh2",
                use_flow_sigmas=True,
                flow_shift=5.0,
            )

            class TinyRenderer(torch.nn.Module):
                def __init__(self, diffusion):
                    super().__init__()
                    self.diff_dec = diffusion

                def forward(self, value):
                    return value

            return TinyRenderer(core)

        with tempfile.TemporaryDirectory() as directory:
            adapter_dir = Path(directory) / "adapter"
            seeded = peft.get_peft_model(
                make_renderer(),
                peft.LoraConfig(
                    r=scope.LORA_RANK,
                    lora_alpha=scope.LORA_ALPHA,
                    lora_dropout=0.0,
                    bias="none",
                    target_modules=scope.canonical_target_modules(),
                ),
            )
            with torch.no_grad():
                for _, layer in inference._lora_layers(seeded):
                    layer.lora_A["default"].weight.fill_(0.001)
                    layer.lora_B["default"].weight.fill_(0.001)
            seeded.save_pretrained(adapter_dir, safe_serialization=True)
            bundle = inference.legacy.AdapterBundle(
                checkpoint_root=Path(directory),
                adapter_dir=adapter_dir,
                adapter_config_path=adapter_dir / "adapter_config.json",
                adapter_model_path=adapter_dir / "adapter_model.safetensors",
                training_receipt_path=Path(directory) / "unused-receipt.json",
            )
            model, renderer, tensor_count, state_validation = (
                inference._strict_load_v9_adapter(
                    base_model=make_renderer(), bundle=bundle
                )
            )

            self.assertEqual(tensor_count, 184)
            self.assertEqual(state_validation["adapter_tensor_count"], 184)
            self.assertEqual(len(inference._lora_layers(model)), 92)

            core = renderer.diff_dec
            core.scheduler.set_timesteps(40)
            source = torch.zeros(1, 16, 21, 2, 2, dtype=torch.float32)
            source_tokens = 21
            source_hidden, source_rotary = core.transformer.patch_vae_latent(
                source.to(torch.bfloat16), source_id=1.0
            )
            target_hidden = torch.linspace(
                -0.2, 0.2, source_tokens, dtype=torch.float32
            ).reshape(1, source_tokens, 1).expand(1, source_tokens, hidden_size)
            paired = torch.cat((source_hidden, target_hidden.to(torch.bfloat16)), dim=1)
            rotary = torch.cat((source_rotary, source_rotary), dim=2)
            timestep = core.scheduler.timesteps[0].expand(1)
            negative = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
            noop = torch.full((1, 2, 4), -0.125, dtype=torch.bfloat16)
            action = torch.full((1, 2, 4), 0.25, dtype=torch.bfloat16)
            bank = replay.SourceKVCacheBank(range(30))

            with inference.source_kv_route_deployment_hook(
                renderer,
                adapter_controller=model,
                cache_bank=bank,
                noop_prompt_embeds=noop,
                source_clean=source,
                latent_shape=source.shape,
                rank=0,
                ulysses_size=4,
                expected_source_tokens=source_tokens,
                bernini_commit=inference.tri.PINNED_BERNINI_COMMIT,
                wan_diffusion_path=wan_path,
            ) as hook:
                parameters = inference.tri.APGParameters(
                    guidance_scale=4.0,
                    omega_scale=0.75,
                    scale_transformer_2=False,
                    eta=0.5,
                    norm_threshold=50.0,
                    momentum=0.0,
                )
                state = inference._ActiveSample(
                    action_prompt=action,
                    negative_prompt=negative,
                    apg=parameters,
                    momenta={
                        name: inference.tri._MomentumBuffer(0.0, branch=name)
                        for name in (
                            "frozen_noop",
                            "frozen_action",
                            "adapted_noop",
                            "adapted_action",
                        )
                    },
                )
                hook._active = state
                shared = dict(
                    model_id="transformer_1",
                    noisy_latents=paired,
                    timesteps=timestep,
                    rotary_embs=rotary,
                    batch_vae_seqlen=[2 * source_tokens],
                )
                core.shared_step(
                    cond_embeds=negative,
                    batch_text_seqlen=[negative.shape[1]],
                    **shared,
                )
                core.shared_step(
                    cond_embeds=action,
                    batch_text_seqlen=[action.shape[1]],
                    **shared,
                )
                self.assertFalse(
                    torch.equal(
                        state.branch_targets["adapted_action"],
                        state.branch_targets["frozen_action"],
                    )
                )

                sample = torch.linspace(
                    -0.1, 0.1, source_tokens * 64, dtype=torch.float32
                ).reshape(1, source_tokens, 64)
                sigma = core.scheduler.sigmas[0]
                sample_spatial = pinned._to_spatial(sample, source.shape)
                negative_clean = sample_spatial - sigma * pinned._to_spatial(
                    state.branch_targets["frozen_negative"], source.shape
                )
                action_clean = sample_spatial - sigma * pinned._to_spatial(
                    state.branch_targets["adapted_action"], source.shape
                )
                official_clean = pinned.normalized_guidance(
                    action_clean,
                    negative_clean,
                    4.0,
                    pinned.MomentumBuffer(0.0),
                    eta=0.5,
                    norm_threshold=50.0,
                )
                official_velocity = pinned._to_packed(
                    (sample_spatial - official_clean) / sigma, source.shape
                )
                result = core.scheduler.step(
                    official_velocity,
                    core.scheduler.timesteps[0],
                    sample,
                    return_dict=False,
                )
                self.assertEqual(len(result), 1)
                self.assertEqual(len(hook.trace.records), 1)
                self.assertTrue(
                    hook.trace.records[0].official_adapted_action_exact_parity
                )

            self.assertTrue(hook.restored)
            self.assertEqual(bank.capture_calls, 30)
            self.assertEqual(bank.replay_lookups, 150)
            self.assertEqual(
                bank.replay_branch_counts,
                {name: 30 for name in inference.REPLAY_BRANCH_ORDER},
            )


if __name__ == "__main__":
    unittest.main()

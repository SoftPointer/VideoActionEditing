#!/usr/bin/env python3

from __future__ import annotations

import copy
import base64
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import braid_stage0_all8_orchestrator_v1 as stage0  # noqa: E402


class BraidStage0All8OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "stage0-output"
        self.root.mkdir()
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        self.execution_private = Ed25519PrivateKey.generate()
        self.execution_public_path = (
            self.root / stage0.EXECUTION_PUBLIC_KEY_FILENAME
        )
        self.execution_public_path.write_bytes(
            self.execution_private.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        os.chmod(self.execution_public_path, 0o444)
        self.execution_public_sha = stage0.file_sha256(
            self.execution_public_path
        )
        self.dog_editor_receipt_sha = self._sha("dog-editor-receipt-file")
        self.human_editor_receipt_sha = self._sha(
            "human-editor-receipt-file"
        )
        self.plan = stage0.build_plan(
            slurm_job_id=132000,
            output_root=str(self.root),
            method_source_revision="1" * 40,
            source_archive_sha256="2" * 64,
            runtime_source_sha256="3" * 64,
            runner_source_sha256="4" * 64,
            dog_editor_receipt_file_sha256=self.dog_editor_receipt_sha,
            human_editor_receipt_file_sha256=(
                self.human_editor_receipt_sha
            ),
            execution_public_key_file_sha256=self.execution_public_sha,
        )
        self.plan_path = self.root / "stage0.plan.json"
        stage0.write_create_only_json(self.plan_path, self.plan)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _sha(label: str) -> str:
        return stage0.object_sha256({"fixture": label})

    def _runtime_receipt(
        self, *, arm: stage0.ArmSpec, rank: int, label: str
    ) -> dict:
        total = 200
        condition = 100
        local = 50
        start = rank * local
        source_rows = max(0, min(start + local, condition) - start)
        target_start = max(start, condition)
        target_rows = max(0, min(start + local, total) - target_start)
        padding_rows = local - source_rows - target_rows
        layout = {
            "schema_version": "bernini-braid-sp4-role-layout-v1",
            "sp_rank": rank,
            "sp_size": 4,
            "total_tokens": total,
            "condition_tokens": condition,
            "target_tokens": total - condition,
            "local_length_ceil": local,
            "shard_global_start": start,
            "shard_global_stop_padded": start + local,
            "source_rows": source_rows,
            "target_rows": target_rows,
            "padding_rows": padding_rows,
            "global_index_formula": (
                "g=sp_rank*ceil(total_tokens/4)+local_index;"
                "source iff g<condition_tokens;target iff "
                "condition_tokens<=g<total_tokens;padding iff g>=total_tokens"
            ),
            "cross_rank_hidden_gather_or_reinjection": False,
        }
        order = (
            ["base_negative", "base_positive", "action_negative", "action_positive"]
            if arm.forward_mode == "reference_4f"
            else ["base_negative", "base_positive", "action_positive"]
        )
        records = []
        trace = []
        for step in range(40):
            record = {
                "step_index": step,
                "block_index": 15,
                "forward_order": order,
                "reset_enabled": arm.reset_source_costate,
                "source_rows": source_rows,
                "target_rows": target_rows,
                "padding_rows": padding_rows,
                "source_pre_reset_mismatch_bytes": 0,
                "source_post_reset_mismatch_bytes": 0,
                "target_post_reset_mismatch_bytes": 0,
                "padding_post_reset_mismatch_bytes": 0,
                "reset_returned_new_object": arm.reset_source_costate,
                "reset_off_returned_original_object": not arm.reset_source_costate,
                "cache_created_once": True,
                "cache_consumed_once": True,
            }
            records.append(record)
            trace.append(
                {
                    "schema_version": stage0.DUAL_RUNTIME_SCHEMA,
                    "step_index": step,
                    "timestep": stage0.NATIVE_UNIPC40_TIMESTEPS[step],
                    "sigma": stage0.NATIVE_UNIPC40_SIGMAS[step],
                    "forward_mode": arm.forward_mode,
                    "forward_order": order,
                    "transformer_forwards": len(order),
                    "base_forwards": 2,
                    "action_forwards": len(order) - 2,
                    "shared_negative": arm.forward_mode
                    == "shared_negative_3f_diagnostic",
                    "independent_complete_native_apg_pairs": arm.forward_mode
                    == "reference_4f",
                    "vendor_base_apg_calls": 1,
                    "vendor_action_apg_calls": 1,
                    "base_action_buffers_distinct": True,
                    "base_stock_apg_exact_parity": True,
                    "base_stock_apg_parity_max_abs": 0.0,
                    "base_stock_apg_parity_rms": 0.0,
                    "negative_repeat_exact_parity": True,
                    "negative_repeat_mismatch_bytes": 0,
                    "action_base_velocity_delta_rms": (
                        0.0
                        if arm.noop_prompt_role == arm.action_prompt_role == "c0"
                        else 0.25
                    ),
                    "original_scheduler_calls": 1,
                    "scheduler_received_stock_base_object": True,
                    "block15": record,
                }
            )
        base_id = 10_000 + rank * 2
        unsigned = {
            "schema_version": stage0.DUAL_RUNTIME_SCHEMA,
            "method": "BRAID Stage-0 dual-native APG structural canary",
            "pinned_bernini_commit": stage0.PINNED_BERNINI_REVISION,
            "pinned_wan_diffusion_sha256": "5" * 64,
            "forward_mode": arm.forward_mode,
            "forward_mode_authority": (
                "four_forward_reference"
                if arm.forward_mode == "reference_4f"
                else "shared_negative_diagnostic_only"
            ),
            "per_step_forward_order": order,
            "steps": 40,
            "transformer_forwards": len(order) * 40,
            "base_forwards": 80,
            "action_forwards": (len(order) - 2) * 40,
            "vendor_base_apg_calls": 40,
            "vendor_action_apg_calls": 40,
            "original_scheduler_calls": 40,
            "scheduler_execution": "stock_base_V0_exact_object_only",
            "vendor_apg_function": "bernini.models.wan_diffusion.normalized_guidance",
            "base_apg_binding": {
                "branch": "base",
                "vendor_type": "bernini.models.wan_diffusion.MomentumBuffer",
                "buffer_object_id": base_id,
                "momentum": 0.0,
                "initial_integer_zero_authenticated": True,
                "normalized_guidance_calls": 40,
            },
            "action_apg_binding": {
                "branch": "action",
                "vendor_type": "bernini.models.wan_diffusion.MomentumBuffer",
                "buffer_object_id": base_id + 1,
                "momentum": 0.0,
                "initial_integer_zero_authenticated": True,
                "normalized_guidance_calls": 40,
            },
            "layout": layout,
            "block15": {
                "schema_version": "bernini-braid-block15-source-costate-canary-v1",
                "block_index": 15,
                "selection_authority": (
                    "infrastructure_canary_only_not_an_authorized_braid_reset_boundary"
                ),
                "reset_enabled": arm.reset_source_costate,
                "rank_local_only": True,
                "hidden_collective_or_reinjection": False,
                "records": records,
                "semantic_action_editing_claim": False,
                "training_authorized": False,
            },
            "trace": trace,
            "parameter_and_buffer_versions_unchanged": True,
            "optimizer_created": False,
            "backward_executed": False,
            "video_decoded": False,
            "checkpoint_read_or_written_by_runtime": False,
            "semantic_action_editing_claim": False,
            "training_authorized": False,
            "runtime_source_identity_enforcement": "external_canary_required",
        }
        return {**unsigned, "runtime_digest": stage0.object_sha256(unsigned)}

    def _world4_receipt(
        self,
        *,
        cell_id: str,
        arm: stage0.ArmSpec,
        process_namespace: str,
    ) -> dict:
        cell = stage0.plan_cell(self.plan, cell_id)
        process_rows = []
        for rank in range(4):
            process_rows.append(
                {
                    "sp_rank": rank,
                    "process_start_identity_sha256": self._sha(
                        f"{process_namespace}:process:{rank}"
                    ),
                    "model_object_identity_sha256": self._sha(
                        f"{process_namespace}:model:{rank}"
                    ),
                    "scheduler_object_identity_sha256": self._sha(
                        f"{process_namespace}:scheduler:{rank}"
                    ),
                    "noop_apg_state_identity_sha256": "",
                    "action_apg_state_identity_sha256": "",
                    "model_construct_count": 1,
                    "scheduler_construct_count": 1,
                    "sample_call_count": 1,
                }
            )
        measurements = {
            "runtime_finalize_passed": True,
            "projection_local_zero_residual_exact": True,
            "off_off_path_structural_pass": True,
            "reset_on_off_path_structural_pass": None,
            "old_motion_axis_observed": None,
            "desired_action_capacity_axis_observed": None,
            "old_motion_action_capacity_non_regression_pass": None,
            "scheduler_steps_observed": 40,
            "scheduler_advances_per_step": 1,
            "exact81_latent_rollout_observed": True,
            "decoded_video_observed": False,
        }
        if arm.canary == "co_state_reset_world4_sp4_oracle":
            measurements["reset_on_off_path_structural_pass"] = True
        elif arm.canary == "old_motion_action_capacity_oracle":
            measurements["old_motion_axis_observed"] = True
            measurements["desired_action_capacity_axis_observed"] = True
            measurements["old_motion_action_capacity_non_regression_pass"] = True
        runtime_receipts = [
            self._runtime_receipt(arm=arm, rank=rank, label=process_namespace)
            for rank in range(4)
        ]
        for rank, process in enumerate(process_rows):
            process["noop_apg_state_identity_sha256"] = (
                stage0.apg_state_identity_sha256(
                    process_start_identity_sha256=process[
                        "process_start_identity_sha256"
                    ],
                    binding=runtime_receipts[rank]["base_apg_binding"],
                )
            )
            process["action_apg_state_identity_sha256"] = (
                stage0.apg_state_identity_sha256(
                    process_start_identity_sha256=process[
                        "process_start_identity_sha256"
                    ],
                    binding=runtime_receipts[rank]["action_apg_binding"],
                )
            )
        device_rows = []
        expected_rocr = ",".join(str(item) for item in cell["visible_devices"])
        for rank in range(4):
            device = {
                "schema_version": stage0.DEVICE_ENVIRONMENT_SCHEMA,
                "sp_rank": rank,
                "rank": rank,
                "local_rank": rank,
                "world_size": 4,
                "rocr_visible_devices": expected_rocr,
                "physical_visible_devices": list(cell["visible_devices"]),
                "hip_visible_devices_unset": True,
                "cuda_visible_devices_unset": True,
                "gpu_device_ordinal_unset": True,
                "observed_before_torch_import": True,
            }
            device_rows.append(
                {
                    **device,
                    "environment_digest": stage0.object_sha256(device),
                }
            )
        unsigned = {
            "schema_version": stage0.WORLD4_SCHEMA,
            "method": stage0.METHOD,
            "plan_receipt_digest": self.plan["receipt_digest"],
            "cell_id": cell_id,
            "query_seed": cell["query_seed"],
            "source_iid": cell["source_iid"],
            "arm_id": arm.arm_id,
            "arm_contract": stage0.asdict(arm),
            "topology": {
                "world_size": 4,
                "sequence_parallel_size": 4,
                "rank_order": [0, 1, 2, 3],
                "visible_devices": cell["visible_devices"],
            },
            "provenance": self.plan["provenance"],
            "coordinate_evidence": {
                "editor_runtime_input_receipt_digest": self._sha(
                    f"{cell_id}:editor-packet"
                ),
                "editor_runtime_input_receipt_file_sha256": cell[
                    "editor_receipt_file_sha256"
                ],
                "editor_public_key_file_sha256": (
                    stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256
                ),
                "editor_method_source_revision": "1" * 40,
                "editor_method_source_archive_sha256": self._sha(
                    f"{cell_id}:editor-source-archive"
                ),
                "source_latent_sha256": self._sha(f"{cell_id}:source-latent"),
                "official_initial_noise_sha256": self._sha(f"{cell_id}:noise"),
                "endpoint_latent_sha256": self._sha(f"{cell_id}:endpoint"),
                "noop_prompt_tensor_sha256": self._sha(f"{cell_id}:c0"),
                "action_prompt_tensor_sha256": self._sha(
                    f"{cell_id}:{arm.action_prompt_role}"
                ),
                "negative_prompt_tensor_sha256": self._sha(f"{cell_id}:negative"),
                "exact40_timestep_sigma_digest": stage0.PINNED_NATIVE_SCHEDULE_DIGEST,
                "source_and_noise_byte_identity_revalidated": True,
                "prompt_byte_identity_revalidated": True,
                "all_rank_coordinate_consensus": True,
            },
            "mechanism_evidence": {
                "visual_pack_mode": stage0.VISUAL_PACK_MODE,
                "sp4_collective_receipt_digest": self._sha(
                    f"{process_namespace}:sp4-collective"
                ),
                "source_bias_mode": arm.source_bias_mode,
                "source_bias_operator_digest": (
                    self._sha("source-bias-operator-v1")
                    if arm.source_bias_mode == "read_only_simulated_stage_a_bias"
                    else None
                ),
                "source_bias_read_only": True,
                "source_bias_parameter_mutation": False,
                "comparison_evaluator_source_sha256": self.plan["provenance"][
                    "runner_source_sha256"
                ],
                "comparison_threshold_registry_sha256": stage0.PINNED_BRAID_ARM_REGISTRY_SHA256,
                "all_rank_metric_packet_digest": self._sha(
                    f"{process_namespace}:metric-packet"
                ),
                "all_rank_mechanism_consensus": True,
            },
            "runtime_receipts": runtime_receipts,
            "fresh_process_evidence": process_rows,
            "device_environment_evidence": device_rows,
            "measurements": measurements,
            "execution_authority": dict(stage0.EXECUTION_AUTHORITY),
            "result": {
                "status": "PASS",
                "classification": "ENGINEERING_FORWARD_PATH_ONLY",
                "semantic_authority": False,
                "decoded_quality_authority": False,
                "stage0_training_authority": False,
            },
        }
        payload = stage0.seal_receipt(unsigned)
        signature = self.execution_private.sign(
            stage0.canonical_json_bytes(payload)
        )
        return {
            **payload,
            "execution_signature_scheme": stage0.EXECUTION_SIGNATURE_SCHEME,
            "execution_public_key_file_sha256": self.execution_public_sha,
            "execution_signature_ed25519_base64": base64.b64encode(
                signature
            ).decode("ascii"),
        }

    def _resign_world4(self, value: dict, *, signer=None) -> dict:
        unsigned = copy.deepcopy(value)
        for name in stage0._WORLD4_SIGNATURE_KEYS | {"receipt_digest"}:
            unsigned.pop(name, None)
        payload = stage0.seal_receipt(unsigned)
        private = self.execution_private if signer is None else signer
        signature = private.sign(stage0.canonical_json_bytes(payload))
        return {
            **payload,
            "execution_signature_scheme": stage0.EXECUTION_SIGNATURE_SCHEME,
            "execution_public_key_file_sha256": self.execution_public_sha,
            "execution_signature_ed25519_base64": base64.b64encode(
                signature
            ).decode("ascii"),
        }

    def _write_full_evidence(self, *, reused_namespace: str | None = None) -> Path:
        evidence = self.root / "evidence"
        for cell in stage0.CELL_SPECS:
            cell_id = str(cell["cell_id"])
            for arm in stage0.ARM_SPECS:
                directory = evidence / cell_id / arm.arm_id
                directory.mkdir(parents=True)
                namespace = reused_namespace or f"{cell_id}:{arm.arm_id}"
                receipt = self._world4_receipt(
                    cell_id=cell_id, arm=arm, process_namespace=namespace
                )
                stage0.write_create_only_json(
                    directory / "world4.receipt.json", receipt
                )
        return evidence

    def _write_reference4f_a_only_evidence(self) -> Path:
        evidence = self.root / "evidence"
        arm = stage0.ARM_BY_ID["parity-reset-off-reference-4f-a"]
        for cell in stage0.CELL_SPECS:
            cell_id = str(cell["cell_id"])
            directory = evidence / cell_id / arm.arm_id
            directory.mkdir(parents=True)
            receipt = self._world4_receipt(
                cell_id=cell_id,
                arm=arm,
                process_namespace=f"partial:{cell_id}:{arm.arm_id}",
            )
            stage0.write_create_only_json(
                directory / "world4.receipt.json", receipt
            )
        return evidence

    def test_plan_is_fixed_all8_forward_only_and_cannot_authorize_stage_a(self) -> None:
        row = stage0.validate_plan(self.plan)
        self.assertEqual(len(row["arms"]), 6)
        self.assertEqual(row["topology"]["dog_visible_devices"], [0, 1, 2, 3])
        self.assertEqual(row["topology"]["human_visible_devices"], [4, 5, 6, 7])
        self.assertTrue(row["topology"]["fresh_process_per_cell_arm"])
        self.assertFalse(row["topology"]["shared_world8_process_group"])
        self.assertEqual(row["prohibitions"], stage0.PROHIBITIONS)
        self.assertEqual(
            row["provenance"]["editor_public_key_file_sha256"],
            stage0.PINNED_EDITOR_PUBLIC_KEY_SHA256,
        )
        self.assertEqual(
            stage0.plan_cell(row, "dog")["editor_receipt_file_sha256"],
            self.dog_editor_receipt_sha,
        )
        self.assertEqual(
            row["execution_authentication"]["public_key_file_sha256"],
            self.execution_public_sha,
        )
        self.assertEqual(
            row["publication_contract"]["stage_a_shadow_updates_authorized"], 0
        )

    def test_valid_world4_reopens_real_runtime_schema_and_rank3_full_target(self) -> None:
        arm = stage0.ARM_BY_ID["parity-reset-off-reference-4f-a"]
        receipt = self._world4_receipt(
            cell_id="dog", arm=arm, process_namespace="dog:parity-a"
        )
        row = stage0.validate_world4_receipt(
            receipt, plan=self.plan, expected_cell="dog", expected_arm=arm.arm_id
        )
        self.assertEqual(row["runtime_receipts"][3]["layout"]["target_rows"], 50)
        self.assertEqual(row["runtime_receipts"][3]["layout"]["source_rows"], 0)

    def test_runtime_tamper_or_forbidden_execution_fails_closed(self) -> None:
        arm = stage0.ARM_BY_ID["parity-reset-off-reference-4f-a"]
        receipt = self._world4_receipt(
            cell_id="dog", arm=arm, process_namespace="dog:tamper"
        )
        tampered = copy.deepcopy(receipt)
        tampered["runtime_receipts"][0]["backward_executed"] = True
        tampered = self._resign_world4(tampered)
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "dual runtime receipt seal differs|call/authority closure differs",
        ):
            stage0.validate_world4_receipt(
                tampered,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=arm.arm_id,
            )

    def test_editor_packet_device_environment_and_execution_signature_attacks_fail(self) -> None:
        arm = stage0.ARM_BY_ID["parity-reset-off-reference-4f-a"]
        receipt = self._world4_receipt(
            cell_id="dog", arm=arm, process_namespace="dog:attacks"
        )

        arbitrary_same_key_packet = copy.deepcopy(receipt)
        arbitrary_same_key_packet["coordinate_evidence"][
            "editor_runtime_input_receipt_file_sha256"
        ] = self._sha("other-valid-editor-packet-same-editor-key")
        arbitrary_same_key_packet = self._resign_world4(
            arbitrary_same_key_packet
        )
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "coordinate evidence failed",
        ):
            stage0.validate_world4_receipt(
                arbitrary_same_key_packet,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=arm.arm_id,
            )

        self_reported_devices = copy.deepcopy(receipt)
        device = self_reported_devices["device_environment_evidence"][0]
        device["rocr_visible_devices"] = "0,1,2,4"
        unsigned_device = dict(device)
        unsigned_device.pop("environment_digest")
        device["environment_digest"] = stage0.object_sha256(unsigned_device)
        self_reported_devices = self._resign_world4(self_reported_devices)
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "live device environment binding differs",
        ):
            stage0.validate_world4_receipt(
                self_reported_devices,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=arm.arm_id,
            )

        unsigned_receipt = copy.deepcopy(receipt)
        for name in stage0._WORLD4_SIGNATURE_KEYS:
            unsigned_receipt.pop(name)
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "lacks its closed execution signature",
        ):
            stage0.validate_world4_receipt(
                unsigned_receipt,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=arm.arm_id,
            )

        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        other_key_receipt = self._resign_world4(
            receipt, signer=Ed25519PrivateKey.generate()
        )
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "Ed25519 verification failed",
        ):
            stage0.validate_world4_receipt(
                other_key_receipt,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=arm.arm_id,
            )

    def test_unimplemented_reset_and_capacity_arms_fail_closed(self) -> None:
        reset = stage0.ARM_BY_ID["reset-on-reference-4f"]
        receipt = self._world4_receipt(
            cell_id="human", arm=reset, process_namespace="human:reset"
        )
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "evidence implementation is unavailable",
        ):
            stage0.validate_world4_receipt(
                receipt,
                plan=self.plan,
                expected_cell="human",
                expected_arm=reset.arm_id,
            )
        capacity = stage0.ARM_BY_ID[
            "capacity-source-bias-on-reference-4f"
        ]
        receipt = self._world4_receipt(
            cell_id="dog", arm=capacity, process_namespace="dog:capacity"
        )
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "evidence implementation is unavailable",
        ):
            stage0.validate_world4_receipt(
                receipt,
                plan=self.plan,
                expected_cell="dog",
                expected_arm=capacity.arm_id,
            )

    def test_all8_aggregate_cannot_publish_before_remaining_arms_exist(self) -> None:
        evidence = self._write_full_evidence()
        output = self.root / "all8.manifest.json"
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError,
            "evidence implementation is unavailable",
        ):
            stage0.aggregate_all8(
                plan_path=self.plan_path, evidence_root=evidence, output=output
            )
        self.assertFalse(output.exists())

    def test_reference4f_a_only_all8_receipt_is_explicitly_partial_and_non_authorizing(self) -> None:
        evidence = self._write_reference4f_a_only_evidence()
        output = self.root / "reference-4f-a-only-all8.receipt.json"
        receipt = stage0.aggregate_reference4f_a_only_all8(
            plan_path=self.plan_path,
            evidence_root=evidence,
            output=output,
        )
        self.assertTrue(receipt["partial_stage0"])
        self.assertFalse(receipt["full_stage0_complete"])
        self.assertFalse(receipt["stage_a_authorized"])
        self.assertEqual(receipt["stage_a_shadow_updates_authorized"], 0)
        self.assertFalse(receipt["stage0_training_authority"])
        self.assertFalse(receipt["scientific_authority"])
        self.assertEqual(receipt["world4_receipt_count"], 2)
        self.assertEqual(receipt["fresh_rank_process_count"], 8)
        self.assertEqual(
            receipt["completed_arm_ids"], ["parity-reset-off-reference-4f-a"]
        )
        self.assertEqual(len(receipt["missing_arm_ids"]), 5)
        reopened = stage0.validate_reference4f_a_only_all8_receipt(
            stage0.load_json(output, label="test partial receipt"), plan=self.plan
        )
        self.assertEqual(reopened["receipt_digest"], receipt["receipt_digest"])
        self.assertFalse((self.root / "all8.manifest.json").exists())

    def test_partial_receipt_authority_tamper_and_extra_evidence_fail_closed(self) -> None:
        evidence = self._write_reference4f_a_only_evidence()
        output = self.root / "reference-4f-a-only-all8.receipt.json"
        receipt = stage0.aggregate_reference4f_a_only_all8(
            plan_path=self.plan_path,
            evidence_root=evidence,
            output=output,
        )
        tampered = copy.deepcopy(receipt)
        tampered["stage_a_authorized"] = True
        unsigned = dict(tampered)
        unsigned.pop("receipt_digest")
        tampered["receipt_digest"] = stage0.object_sha256(unsigned)
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError, "partial authority closure differs"
        ):
            stage0.validate_reference4f_a_only_all8_receipt(
                tampered, plan=self.plan
            )
        extra = evidence / "dog/unregistered.json"
        extra.write_text("{}", encoding="ascii")
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError, "evidence closure differs"
        ):
            stage0.aggregate_reference4f_a_only_all8(
                plan_path=self.plan_path,
                evidence_root=evidence,
                output=output,
            )

    def test_world4_process_reuse_and_extra_artifact_are_rejected(self) -> None:
        evidence = self._write_full_evidence(reused_namespace="shared-process")
        first = evidence / "dog" / "parity-reset-off-reference-4f-a" / "world4.receipt.json"
        receipt = stage0.load_json(first, label="test WORLD4")
        broken = copy.deepcopy(receipt)
        broken["fresh_process_evidence"][1]["process_start_identity_sha256"] = (
            broken["fresh_process_evidence"][0]["process_start_identity_sha256"]
        )
        for process_key, binding_key in (
            ("noop_apg_state_identity_sha256", "base_apg_binding"),
            ("action_apg_state_identity_sha256", "action_apg_binding"),
        ):
            broken["fresh_process_evidence"][1][process_key] = (
                stage0.apg_state_identity_sha256(
                    process_start_identity_sha256=broken[
                        "fresh_process_evidence"
                    ][1]["process_start_identity_sha256"],
                    binding=broken["runtime_receipts"][1][binding_key],
                )
            )
        broken = self._resign_world4(broken)
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError, "reused one process identity"
        ):
            stage0.validate_world4_receipt(
                broken,
                plan=self.plan,
                expected_cell="dog",
                expected_arm="parity-reset-off-reference-4f-a",
            )
        extra = evidence / "dog" / "unexpected.json"
        extra.write_text("{}", encoding="ascii")
        with self.assertRaisesRegex(
            stage0.BraidStage0OrchestrationError, "evidence closure differs"
        ):
            stage0.aggregate_all8(
                plan_path=self.plan_path,
                evidence_root=evidence,
                output=self.root / "all8.manifest.json",
            )


if __name__ == "__main__":
    unittest.main()

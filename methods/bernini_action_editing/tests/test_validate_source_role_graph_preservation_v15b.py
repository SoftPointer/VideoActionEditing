import unittest
from dataclasses import replace
from pathlib import Path
import tempfile

import torch

from methods.bernini_action_editing import source_role_graph_preservation_v15b as core
from methods.bernini_action_editing import validate_source_role_graph_preservation_v15b as validator
from methods.bernini_action_editing.tests.test_source_role_graph_preservation_v15b import (
    EXTRACTOR_CODE_SHA,
    EXTRACTOR_CONFIG_SHA,
    HEADS,
    HEAD_DIM,
    HEIGHT,
    HIDDEN_WIDTH,
    SHA_A,
    SHA_B,
    SOURCE_LATENT_SHA,
    WIDTH,
    binding,
    carrier_and_memory,
    graph_fixture,
    mask_set,
    persistent_target_fixture,
)


def _reseal(row):
    row["receipt_digest"] = validator.receipt_digest_v15b(row)
    return row


def _clone_receipts(receipts):
    return [dict(row) for row in receipts]


class FourArmReceiptValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bound = binding(); cls.masks = mask_set(cls.bound)
        cls.fixture = graph_fixture()
        core_pin, validator_pin = validator.current_code_pins_v15b()
        cls.contract = core.FourArmContractV15B.create(
            action_id="pour", source_video_sha256=SHA_A,
            instruction_sha256=SHA_B, binding_digest=cls.bound.digest,
            mask_digest=cls.masks.digest,
            track_authority_digest=cls.masks.track_authority.digest,
            source_graph_digest=cls.fixture["source_graph"].digest,
            graph_a_slot="v0", graph_b_slot="v1",
            signed_graph_a_digest=cls.fixture["signed_a"].digest,
            signed_graph_b_digest=cls.fixture["signed_b"].digest,
            aligned_swap_report_digest=cls.fixture["report"].digest,
            four_anchor_consensus_digest=cls.fixture["consensus"].digest,
            canonical_trace_digest=cls.fixture["canonical"].digest,
            trace_extractor_code_sha256=EXTRACTOR_CODE_SHA,
            trace_extractor_config_sha256=EXTRACTOR_CONFIG_SHA,
            anchor_asset_sha256_by_slot=cls.fixture["consensus"].asset_sha256_by_slot,
            height=HEIGHT, width=WIDTH, batch_size=1, heads=HEADS,
            head_dim=HEAD_DIM, hidden_width=HIDDEN_WIDTH,
            self_reported_model_checkpoint_sha256="e" * 64,
            self_reported_model_code_sha256="f" * 64,
            core_code_sha256=core_pin, validator_code_sha256=validator_pin,
            route_strength=0.5, memory_strength=0.5,
        )
        cls.receipts = [cls._receipt(arm) for arm in cls.contract.arms]

    @classmethod
    def _audit(cls, arm, step, block, branch, stage):
        graph = None if arm.graph_slot is None else (
            cls.contract.signed_graph_a_digest if arm.graph_slot == "A"
            else cls.contract.signed_graph_b_digest
        )
        pre = stage == "pre"
        restore = arm.restore_background_pre if pre else arm.restore_background_post
        route = pre and arm.route_enabled
        memory = pre and arm.source_content_memory
        spatial = HEIGHT * WIDTH
        role_union = None
        for role_mask in cls.masks.role_masks.values():
            current = role_mask[:, :spatial]
            role_union = current.clone() if role_union is None else role_union | current
        corridor_counts = tuple(
            int(cls.masks.editable_corridor_mask[
                :, phase * spatial:(phase + 1) * spatial
            ].sum())
            for phase in range(core.LATENT_PHASES)
        )
        assigned_counts = (
            (int(role_union.sum()),) + (1,) * 20 if memory else (0,) * 21
        )
        unassigned_counts = (
            (0,) + tuple(count - 1 for count in corridor_counts[1:])
            if memory else (0,) * 21
        )
        row = {
            "schema_version": core.BLOCK_AUDIT_SCHEMA,
            "stage": stage, "step_index": step, "block_index": block,
            "branch": branch, "signed_graph_digest": graph,
            "mask_digest": cls.masks.digest,
            "raw_source_material_digest": "d" * 64,
            "source_latent_sha256": SOURCE_LATENT_SHA,
            "canonical_extraction_config_sha256": "e" * 64,
            "raw_source_material_reopened": True,
            "memory_builder_receipt_digest": "f" * 64 if memory else None,
            "slot_provenance_digest": "0" * 64 if memory else None,
            "slot_uuid_mask_provenance_verified": memory,
            "target_write_ownership_sha256": "1" * 64 if memory else None,
            "target_write_ownership_verified": memory,
            "cross_role_zero_proof_sha256": "2" * 64 if memory else None,
            "target_role_state_digest": "a" * 64 if memory else None,
            "target_transport_digest": "b" * 64 if memory else None,
            "position_projector_sha256": "c" * 64 if memory else None,
            "scrubbed_target_key_sha256": "d" * 64 if memory else None,
            "persistent_support_sha256": "e" * 64 if memory else None,
            "target_role_assigned_token_count_by_phase": assigned_counts,
            "target_role_unassigned_corridor_count_by_phase": unassigned_counts,
            "routed_roles": tuple(sorted(core.SIGNED_ROLES)) if route else (),
            "role_memory_read_count": 4 if memory else 0,
            "route_strength": arm.route_strength if pre else 0.0,
            "memory_strength": arm.memory_strength if pre else 0.0,
            "relation_operator": (
                "position_scrubbed_target_key_persistent_role_pool_query_scatter"
                if route else "none"
            ),
            "target_key_sha256": "e" * 64 if route else None,
            "tensor_batch_size": 1, "tensor_temporal_phases": 21,
            "tensor_height": HEIGHT, "tensor_width": WIDTH,
            "tensor_heads": HEADS, "tensor_head_dim": HEAD_DIM,
            "tensor_hidden_width": HIDDEN_WIDTH,
            "tensor_dtype": "torch.float32", "tensor_device": "cpu",
            "background_hidden_max_abs": 0.0 if restore else None,
            "background_key_max_abs": 0.0 if restore else None,
            "background_value_max_abs": 0.0 if restore else None,
            "route_delta_outside_corridor_max_abs": 0.0,
            "memory_residual_outside_corridor_max_abs": 0.0,
            "phase0_route_max_abs": 0.0, "phase0_memory_max_abs": 0.0,
            "disallowed_add_edge_max_abs": 0.0,
            "disallowed_remove_edge_max_abs": 0.0,
            "memory_hidden_mutation_max_abs": 0.0,
            "memory_key_mutation_max_abs": 0.0,
            "memory_convex_violation_max_abs": 0.0,
            "cross_role_memory_write_max_abs": 0.0,
            "target_cross_role_rename_count": 0,
            "target_corridor_escape_count": 0,
            "target_dual_position_component_count": 0,
            "transition_background_overlap_count": 0,
            "source_coordinate_target_write_count": 0,
            "phase0_full_source_restore_call_count": 1,
            "phase0_full_source_restore_token_count": HEIGHT * WIDTH,
            "phase0_hidden_source_max_abs": 0.0,
            "phase0_key_source_max_abs": 0.0,
            "phase0_value_source_max_abs": 0.0,
            "same_coordinate_object_kv_copy_count": 0,
            "object_hidden_hard_restore_count": 0,
            "phase_indexed_source_kv_access_count": 0,
            "post_rope_source_kv_access_count": 0,
            "anchor_forbidden_access_count": 0,
            "input_hidden_sha256": "0" * 64,
            "input_query_sha256": "1" * 64 if pre else None,
            "input_key_sha256": "2" * 64,
            "input_value_sha256": "3" * 64,
            "carrier_hidden_sha256": "4" * 64,
            "carrier_key_sha256": "5" * 64,
            "carrier_value_sha256": "6" * 64,
            "output_hidden_sha256": "7" * 64,
            "output_query_sha256": "8" * 64 if pre else None,
            "output_key_sha256": "9" * 64,
            "output_value_sha256": "a" * 64,
            "route_delta_sha256": "b" * 64 if pre else None,
            "appearance_residual_sha256": "c" * 64 if pre else None,
            "cell_tensor_abi_digest": "0" * 64,
        }
        abi = {
            "schema_version": row["schema_version"], "stage": row["stage"],
            "step_index": row["step_index"], "block_index": row["block_index"],
            "branch": row["branch"], "mask_digest": row["mask_digest"],
            "raw_source_material_digest": row["raw_source_material_digest"],
            "source_latent_sha256": row["source_latent_sha256"],
            "canonical_extraction_config_sha256": row[
                "canonical_extraction_config_sha256"
            ],
            "raw_source_material_reopened": row["raw_source_material_reopened"],
            "slot_provenance_digest": row["slot_provenance_digest"],
            "slot_uuid_mask_provenance_verified": row[
                "slot_uuid_mask_provenance_verified"
            ],
            "target_write_ownership_sha256": row[
                "target_write_ownership_sha256"
            ],
            "target_write_ownership_verified": row[
                "target_write_ownership_verified"
            ],
            "cross_role_zero_proof_sha256": row["cross_role_zero_proof_sha256"],
            "target_role_state_digest": row["target_role_state_digest"],
            "target_transport_digest": row["target_transport_digest"],
            "position_projector_sha256": row["position_projector_sha256"],
            "scrubbed_target_key_sha256": row["scrubbed_target_key_sha256"],
            "persistent_support_sha256": row["persistent_support_sha256"],
            "input_hidden_sha256": row["input_hidden_sha256"],
            "input_query_sha256": row["input_query_sha256"],
            "input_key_sha256": row["input_key_sha256"],
            "input_value_sha256": row["input_value_sha256"],
            "carrier_hidden_sha256": row["carrier_hidden_sha256"],
            "carrier_key_sha256": row["carrier_key_sha256"],
            "carrier_value_sha256": row["carrier_value_sha256"],
            "output_hidden_sha256": row["output_hidden_sha256"],
            "output_query_sha256": row["output_query_sha256"],
            "output_key_sha256": row["output_key_sha256"],
            "output_value_sha256": row["output_value_sha256"],
            "route_delta_sha256": row["route_delta_sha256"],
            "appearance_residual_sha256": row["appearance_residual_sha256"],
        }
        row["cell_tensor_abi_digest"] = core.object_sha256(abi)
        return row

    @classmethod
    def _receipt(cls, arm):
        slot = None if arm.graph_slot is None else (
            cls.contract.graph_a_slot if arm.graph_slot == "A" else cls.contract.graph_b_slot
        )
        graph = None if arm.graph_slot is None else (
            cls.contract.signed_graph_a_digest if arm.graph_slot == "A"
            else cls.contract.signed_graph_b_digest
        )
        audits = []
        for step in range(core.DENOISE_STEPS):
            for block in range(core.TRANSFORMER_BLOCKS):
                for branch in core.CFG_BRANCHES:
                    audits.append(cls._audit(arm, step, block, branch, "pre"))
                    audits.append(cls._audit(arm, step, block, branch, "post"))
        cells = core.EXPECTED_EXECUTION_CELLS
        row = {
            "schema_version": validator.ARM_RECEIPT_SCHEMA, "complete": True,
            "arm_id": arm.arm_id, "contract_digest": cls.contract.digest,
            "source_video_sha256": SHA_A, "instruction_sha256": SHA_B,
            "binding_digest": cls.bound.digest, "mask_digest": cls.masks.digest,
            "track_authority_digest": cls.masks.track_authority.digest,
            "source_graph_digest": cls.fixture["source_graph"].digest,
            "canonical_trace_digest": cls.fixture["canonical"].digest,
            "trace_extractor_code_sha256": EXTRACTOR_CODE_SHA,
            "trace_extractor_config_sha256": EXTRACTOR_CONFIG_SHA,
            "self_reported_model_checkpoint_sha256": (
                cls.contract.self_reported_model_checkpoint_sha256
            ),
            "self_reported_model_code_sha256": cls.contract.self_reported_model_code_sha256,
            "core_code_sha256": cls.contract.core_code_sha256,
            "validator_code_sha256": cls.contract.validator_code_sha256,
            "self_reported_noise_sha256": "c" * 64,
            "self_reported_candidate_schedule_sha256": "d" * 64,
            "initial_noise_mode": "keyed_only", "graph_anchor_slot": slot,
            "signed_edit_graph_digest": graph,
            "anchor_cached_fields": ["canonical_role_relation_graph"] if arm.route_enabled else [],
            "self_reported_anchor_forbidden_access_count": 0,
            "temporal_phases": 21, "height": HEIGHT, "width": WIDTH,
            "batch_size": 1, "heads": HEADS, "head_dim": HEAD_DIM,
            "hidden_width": HIDDEN_WIDTH,
            "tensor_dtype": "torch.float32", "tensor_device": "cpu",
            "denoise_steps": 40, "transformer_blocks": 22,
            "cfg_branches": list(core.CFG_BRANCHES),
            "route_strength": arm.route_strength, "memory_strength": arm.memory_strength,
            "self_reported_model_block_call_count": cells,
            "self_reported_pre_block_call_count": cells,
            "self_reported_post_block_call_count": cells,
            "self_reported_route_call_count": cells if arm.route_enabled else 0,
            "self_reported_source_content_memory_call_count": (
                cells if arm.source_content_memory else 0
            ),
            "self_reported_background_carrier_call_count": cells * (
                int(arm.restore_background_pre) + int(arm.restore_background_post)
            ),
            "self_reported_phase0_full_source_restore_call_count": 2 * cells,
            "self_reported_optimizer_update_count": 0,
            "self_reported_parameter_before_sha256": "0" * 64,
            "self_reported_parameter_after_sha256": "0" * 64,
            "self_reported_buffer_before_sha256": "1" * 64,
            "self_reported_buffer_after_sha256": "1" * 64,
            "self_reported_model_state_before_sha256": "2" * 64,
            "self_reported_model_state_after_sha256": "2" * 64,
            "self_reported_rng_state_before_sha256": "3" * 64,
            "self_reported_rng_state_after_sha256": "4" * 64,
            "self_reported_rng_replay_before_sha256": "3" * 64,
            "self_reported_rng_replay_after_sha256": "4" * 64,
            "self_reported_frame0_source_tensor_sha256": "5" * 64,
            "self_reported_frame0_output_tensor_sha256": "5" * 64,
            "self_reported_output_tensor_sha256": "6" * 64,
            "self_reported_output_video_sha256": "7" * 64,
            "runner_integration_present": False, "route_authorized": False,
            "self_reported_block_audits": audits,
        }
        return _reseal(row)

    @classmethod
    def _kwargs(cls, receipts=None, **overrides):
        kwargs = {
            "contract": cls.contract,
            "receipts": cls.receipts if receipts is None else receipts,
            "binding": cls.bound, "masks": cls.masks,
            "source_graph": cls.fixture["source_graph"],
            "source_warp": cls.fixture["source_warp"],
            "banks": list(cls.fixture["banks"].values()),
            "warps": list(cls.fixture["warps"].values()),
            "traces": list(cls.fixture["traces"].values()),
            "canonical_trace": cls.fixture["canonical"],
            "signed_graph_a": cls.fixture["signed_a"],
            "signed_graph_b": cls.fixture["signed_b"],
            "aligned_swap_report": cls.fixture["report"],
            "four_anchor_consensus": cls.fixture["consensus"],
        }
        kwargs.update(overrides)
        return kwargs

    def test_clean_reference_suite_is_explicit_no_go(self):
        result = validator.validate_four_arm_receipts_v15b(**self._kwargs())
        self.assertTrue(result["reference_validation_complete"])
        self.assertFalse(result["externally_authenticated"])
        self.assertFalse(result["source_video_latent_externally_authenticated"])
        self.assertEqual(
            result["cpu_mechanical_scope"],
            "internal_phase0_raw_extraction_consistency_only",
        )
        self.assertFalse(result["position_removed_claimed"])
        self.assertFalse(result["native_flow_claimed"])
        self.assertFalse(result["route_authorized"])
        self.assertFalse(result["scientific_claim_authorized"])
        self.assertEqual(result["decision"], "NO_GO_RUNNER_INTEGRATION_UNPROVEN")
        self.assertTrue(all(
            arm["block_audit_count"] == 2 * core.EXPECTED_EXECUTION_CELLS
            for arm in result["arms"]
        ))
        self.assertTrue(all(
            arm["externally_authenticated"] is False and
            arm["position_removed_claimed"] is False and
            arm["native_flow_claimed"] is False and
            arm["scientific_claim_authorized"] is False and
            arm["route_authorized"] is False
            for arm in result["arms"]
        ))

    def test_route_authorization_request_is_refused(self):
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(
                **self._kwargs(), require_route_authorization=True
            )

    def test_receipt_cannot_self_claim_runner_or_route_authority(self):
        for field in ("runner_integration_present", "route_authorized"):
            bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
            bad[1][field] = True; _reseal(bad[1])
            with self.subTest(field=field), self.assertRaises(
                validator.V15BReceiptValidationError
            ):
                validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_exact_call_count_and_audit_cartesian_product_are_enforced(self):
        bad = _clone_receipts(self.receipts); bad[2] = dict(bad[2])
        bad[2]["self_reported_route_call_count"] -= 1; _reseal(bad[2])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))
        bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
        audits = list(bad[1]["self_reported_block_audits"])
        audits[0] = dict(audits[0])
        counts = list(audits[0]["target_role_unassigned_corridor_count_by_phase"])
        counts[1] -= 1
        audits[0]["target_role_unassigned_corridor_count_by_phase"] = counts
        bad[1]["self_reported_block_audits"] = audits; _reseal(bad[1])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))
        bad = _clone_receipts(self.receipts); bad[0] = dict(bad[0])
        bad[0]["self_reported_phase0_full_source_restore_call_count"] -= 1
        _reseal(bad[0])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))
        bad = _clone_receipts(self.receipts); bad[2] = dict(bad[2])
        audits = list(bad[2]["self_reported_block_audits"]); audits[-1] = audits[0]
        bad[2]["self_reported_block_audits"] = audits; _reseal(bad[2])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_strength_target_k_and_builder_closure_are_enforced(self):
        mutations = (
            ("route_strength", 0.4),
            ("target_key_sha256", None),
            ("memory_builder_receipt_digest", None),
            ("target_role_state_digest", None),
            ("target_transport_digest", None),
            ("position_projector_sha256", None),
            ("scrubbed_target_key_sha256", None),
            ("persistent_support_sha256", None),
            ("slot_provenance_digest", None),
            ("slot_uuid_mask_provenance_verified", False),
            ("target_write_ownership_sha256", None),
            ("target_write_ownership_verified", False),
            ("cross_role_zero_proof_sha256", None),
        )
        for field, value in mutations:
            bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
            if field == "route_strength":
                bad[1][field] = value
            else:
                audits = list(bad[1]["self_reported_block_audits"])
                audits[0] = dict(audits[0]); audits[0][field] = value
                bad[1]["self_reported_block_audits"] = audits
            _reseal(bad[1])
            with self.subTest(field=field), self.assertRaises(
                validator.V15BReceiptValidationError
            ):
                validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_resealed_cell_tensor_hash_mutation_fails_abi_digest(self):
        bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
        audits = list(bad[1]["self_reported_block_audits"])
        audits[0] = dict(audits[0])
        audits[0]["output_value_sha256"] = "9" * 64
        bad[1]["self_reported_block_audits"] = audits; _reseal(bad[1])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_phase0_background_and_forbidden_copy_mutations_are_rejected(self):
        for field in (
            "phase0_route_max_abs", "phase0_memory_max_abs",
            "transition_background_overlap_count", "same_coordinate_object_kv_copy_count",
            "cross_role_memory_write_max_abs", "target_cross_role_rename_count",
            "target_corridor_escape_count", "target_dual_position_component_count",
            "object_hidden_hard_restore_count", "phase_indexed_source_kv_access_count",
            "post_rope_source_kv_access_count", "source_coordinate_target_write_count",
            "phase0_hidden_source_max_abs", "phase0_key_source_max_abs",
            "phase0_value_source_max_abs",
        ):
            bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
            audits = list(bad[1]["self_reported_block_audits"])
            audits[0] = dict(audits[0]); audits[0][field] = 1
            bad[1]["self_reported_block_audits"] = audits; _reseal(bad[1])
            with self.subTest(field=field), self.assertRaises(
                validator.V15BReceiptValidationError
            ):
                validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_material_cell_tensor_abi_is_recomputed_not_trusted_from_receipt(self):
        carrier, memory, _, _, _ = carrier_and_memory(self.bound, self.masks)
        key, transport, _ = persistent_target_fixture(self.bound, self.masks, memory)
        tokens = core.LATENT_PHASES * HEIGHT * WIDTH
        hidden = torch.zeros(1, tokens, HIDDEN_WIDTH)
        query = torch.zeros(1, tokens, HEADS, HEAD_DIM)
        value = torch.randn_like(key)
        observed = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key,
            target_value=value, carrier=carrier, binding=self.bound,
            signed_graph=self.fixture["signed_a"], content_memory=memory,
            target_native_transport=transport, route_strength=0.5,
            memory_strength=0.5, restore_background=True,
        )
        material = validator.serialize_pre_block_material_fixture_v15b(
            observed=observed, target_hidden=hidden, target_query=query,
            target_key=key, target_value=value, carrier=carrier,
            binding=self.bound, signed_graph=self.fixture["signed_a"],
            content_memory=memory, target_native_transport=transport,
            route_strength=0.5, memory_strength=0.5,
            restore_background=True,
        )
        result = validator.validate_pre_block_tensor_recompute_v15b(
            material_fixture=material
        )
        self.assertTrue(result["tensor_recompute_exact"])
        self.assertTrue(result["fresh_material_reinstantiated"])
        self.assertTrue(result["raw_source_material_fresh_reopen"])
        self.assertTrue(result["carrier_phase0_raw_hkv_exact"])
        self.assertTrue(result["source_memory_reextracted_from_raw"])
        self.assertTrue(result["per_role_key_value_exact"])
        self.assertTrue(result["slot_uuid_mask_provenance_verified"])
        self.assertTrue(result["target_write_ownership_verified"])
        self.assertFalse(result["source_video_latent_externally_authenticated"])
        self.assertEqual(
            result["cpu_mechanical_scope"],
            "internal_phase0_raw_extraction_consistency_only",
        )
        self.assertEqual(result["cross_role_memory_write_max_abs"], 0.0)
        self.assertFalse(result["externally_authenticated"])
        self.assertFalse(result["position_removed_claimed"])
        self.assertFalse(result["native_flow_claimed"])
        self.assertFalse(result["route_authorized"])
        self.assertFalse(result["scientific_claim_authorized"])
        poisoned_value = value.clone(); poisoned_value[:, HEIGHT * WIDTH] += 1
        poisoned_material = validator.serialize_pre_block_material_fixture_v15b(
            observed=observed, target_hidden=hidden, target_query=query,
            target_key=key, target_value=poisoned_value, carrier=carrier,
            binding=self.bound, signed_graph=self.fixture["signed_a"],
            content_memory=memory, target_native_transport=transport,
            route_strength=0.5, memory_strength=0.5,
            restore_background=True,
        )
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_pre_block_tensor_recompute_v15b(
                material_fixture=poisoned_material,
            )
        mutated_output = observed.value.clone(); mutated_output[:, HEIGHT * WIDTH] += 1
        with self.assertRaises(core.V15BContractError):
            replace(observed, value=mutated_output)

        post = core.apply_post_block_v15b(
            target_hidden=observed.hidden, target_key=observed.key,
            target_value=observed.value, carrier=carrier, binding=self.bound,
            signed_graph_digest=self.fixture["signed_a"].digest,
            restore_background=True,
        )
        post_material = validator.serialize_post_block_material_fixture_v15b(
            observed=post, target_hidden=observed.hidden,
            target_key=observed.key, target_value=observed.value,
            carrier=carrier, binding=self.bound,
            signed_graph_digest=self.fixture["signed_a"].digest,
            restore_background=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "post-cell.v15b-r8"
            path.write_bytes(post_material)
            post_result = validator.validate_post_block_tensor_recompute_v15b(
                material_fixture=path
            )
        self.assertTrue(post_result["tensor_recompute_exact"])
        self.assertTrue(post_result["fresh_material_reinstantiated"])
        self.assertTrue(post_result["raw_source_material_fresh_reopen"])
        self.assertTrue(post_result["carrier_phase0_raw_hkv_exact"])
        self.assertFalse(post_result["source_memory_reextracted_from_raw"])
        self.assertFalse(post_result["per_role_key_value_exact"])
        self.assertFalse(post_result["slot_uuid_mask_provenance_verified"])
        self.assertFalse(post_result["externally_authenticated"])
        self.assertFalse(post_result["position_removed_claimed"])
        self.assertFalse(post_result["native_flow_claimed"])
        self.assertFalse(post_result["route_authorized"])
        self.assertFalse(post_result["scientific_claim_authorized"])

    def test_resealed_fake_carrier_phase0_hkv_bytes_and_path_are_rejected(self):
        carrier, memory, _, _, _ = carrier_and_memory(self.bound, self.masks)
        key, transport, _ = persistent_target_fixture(
            self.bound, self.masks, memory
        )
        tokens = core.LATENT_PHASES * HEIGHT * WIDTH
        hidden = torch.zeros(1, tokens, HIDDEN_WIDTH)
        query = torch.zeros(1, tokens, HEADS, HEAD_DIM)
        value = torch.zeros_like(key)
        observed = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key,
            target_value=value, carrier=carrier, binding=self.bound,
            signed_graph=self.fixture["signed_a"], content_memory=memory,
            target_native_transport=transport, route_strength=0.5,
            memory_strength=0.5, restore_background=True,
        )
        spatial = HEIGHT * WIDTH
        forged_hidden = carrier.hidden.clone()
        forged_key = carrier.key.clone()
        forged_value = carrier.value.clone()
        forged_hidden[:, :spatial] += 31.0
        forged_key[:, :spatial] += 123.0
        forged_value[:, :spatial] -= 77.0
        object.__setattr__(carrier, "hidden", forged_hidden)
        object.__setattr__(carrier, "key", forged_key)
        object.__setattr__(carrier, "value", forged_value)
        object.__setattr__(
            carrier, "digest", core.object_sha256(carrier._payload())
        )
        material = validator.serialize_pre_block_material_fixture_v15b(
            observed=observed, target_hidden=hidden, target_query=query,
            target_key=key, target_value=value, carrier=carrier,
            binding=self.bound, signed_graph=self.fixture["signed_a"],
            content_memory=memory, target_native_transport=transport,
            route_strength=0.5, memory_strength=0.5,
            restore_background=True,
        )
        with self.assertRaisesRegex(
            validator.V15BReceiptValidationError, "fresh constructor"
        ):
            validator.validate_pre_block_tensor_recompute_v15b(
                material_fixture=material
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forged-carrier-phase0.v15b-r8"
            path.write_bytes(material)
            with self.assertRaisesRegex(
                validator.V15BReceiptValidationError, "fresh constructor"
            ):
                validator.validate_pre_block_tensor_recompute_v15b(
                    material_fixture=path
                )

    def test_serialized_builder_receipt_source_and_replay_reseal_is_rejected(self):
        fields_to_mutate = (
            "source_shape",
            "source_pre_rope_key_sha256",
            "source_scrubbed_pre_rope_key_sha256",
            "source_value_sha256",
            "permutation_probe_sha256",
            "position_scrub_projection_residual_max_abs",
        )
        for field in fields_to_mutate:
            carrier, memory, _, _, _ = carrier_and_memory(self.bound, self.masks)
            key, transport, _ = persistent_target_fixture(
                self.bound, self.masks, memory
            )
            tokens = core.LATENT_PHASES * HEIGHT * WIDTH
            hidden = torch.zeros(1, tokens, HIDDEN_WIDTH)
            query = torch.zeros(1, tokens, HEADS, HEAD_DIM)
            value = torch.zeros_like(key)
            observed = core.apply_pre_block_v15b(
                target_hidden=hidden, target_query=query, target_key=key,
                target_value=value, carrier=carrier, binding=self.bound,
                signed_graph=self.fixture["signed_a"], content_memory=memory,
                target_native_transport=transport, route_strength=0.5,
                memory_strength=0.5, restore_background=True,
            )
            receipt = memory.builder_receipt
            if field == "source_shape":
                forged = (
                    receipt.source_shape[0], receipt.source_shape[1] + 1,
                    receipt.source_shape[2], receipt.source_shape[3],
                )
            elif field == "position_scrub_projection_residual_max_abs":
                forged = core.POSITION_PROJECTOR_TOLERANCE / 2.0
                if forged == getattr(receipt, field):
                    forged = 0.0
            else:
                forged = "0" * 64
            object.__setattr__(receipt, field, forged)
            object.__setattr__(
                receipt, "digest", core.object_sha256(receipt._payload())
            )
            object.__setattr__(
                memory, "digest", core.object_sha256(memory._payload())
            )
            material = validator.serialize_pre_block_material_fixture_v15b(
                observed=observed, target_hidden=hidden, target_query=query,
                target_key=key, target_value=value, carrier=carrier,
                binding=self.bound, signed_graph=self.fixture["signed_a"],
                content_memory=memory, target_native_transport=transport,
                route_strength=0.5, memory_strength=0.5,
                restore_background=True,
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                validator.V15BReceiptValidationError, "fresh constructor"
            ):
                validator.validate_pre_block_tensor_recompute_v15b(
                    material_fixture=material
                )

    def test_serialized_fixture_breaks_aliases_and_rejects_tamper_or_nested_mutation(self):
        carrier, memory, _, _, _ = carrier_and_memory(self.bound, self.masks)
        key, transport, _ = persistent_target_fixture(self.bound, self.masks, memory)
        tokens = core.LATENT_PHASES * HEIGHT * WIDTH
        hidden = torch.zeros(1, tokens, HIDDEN_WIDTH)
        query = torch.zeros(1, tokens, HEADS, HEAD_DIM)
        value = torch.zeros_like(key)
        observed = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key,
            target_value=value, carrier=carrier, binding=self.bound,
            signed_graph=self.fixture["signed_a"], content_memory=memory,
            target_native_transport=transport, route_strength=0.5,
            memory_strength=0.5, restore_background=False,
        )
        material = validator.serialize_pre_block_material_fixture_v15b(
            observed=observed, target_hidden=hidden, target_query=query,
            target_key=key, target_value=value, carrier=carrier,
            binding=self.bound, signed_graph=self.fixture["signed_a"],
            content_memory=memory, target_native_transport=transport,
            route_strength=0.5, memory_strength=0.5,
            restore_background=False,
        )
        memory.value_content[0, 0, 0, 0] += 9
        valid = int(torch.nonzero(
            transport.motion_reference.backward_token_index[0, 0] >= 0
        ).flatten()[0])
        transport.motion_reference.backward_token_index[0, 0, valid] = -1
        result = validator.validate_pre_block_tensor_recompute_v15b(
            material_fixture=material
        )
        self.assertTrue(result["fresh_material_reinstantiated"])
        tampered = bytearray(material); tampered[-1] ^= 1
        with self.assertRaisesRegex(
            validator.V15BReceiptValidationError, "byte digest"
        ):
            validator.validate_pre_block_tensor_recompute_v15b(
                material_fixture=bytes(tampered)
            )
        with self.assertRaisesRegex(
            validator.V15BReceiptValidationError, "bytes or a path-like"
        ):
            validator.validate_pre_block_tensor_recompute_v15b(
                material_fixture=observed
            )

        carrier2, memory2, _, _, _ = carrier_and_memory(self.bound, self.masks)
        key2, transport2, _ = persistent_target_fixture(
            self.bound, self.masks, memory2
        )
        observed2 = core.apply_pre_block_v15b(
            target_hidden=hidden, target_query=query, target_key=key2,
            target_value=value, carrier=carrier2, binding=self.bound,
            signed_graph=self.fixture["signed_a"], content_memory=memory2,
            target_native_transport=transport2, route_strength=0.5,
            memory_strength=0.5, restore_background=False,
        )
        memory2.key_content[0, 0, 0, 0] += 3
        poisoned_material = validator.serialize_pre_block_material_fixture_v15b(
            observed=observed2, target_hidden=hidden, target_query=query,
            target_key=key2, target_value=value, carrier=carrier2,
            binding=self.bound, signed_graph=self.fixture["signed_a"],
            content_memory=memory2, target_native_transport=transport2,
            route_strength=0.5, memory_strength=0.5,
            restore_background=False,
        )
        with self.assertRaisesRegex(
            validator.V15BReceiptValidationError, "fresh constructor"
        ):
            validator.validate_pre_block_tensor_recompute_v15b(
                material_fixture=poisoned_material
            )

    def test_zero_update_frame0_rng_and_code_pins_are_enforced(self):
        mutations = {
            "self_reported_optimizer_update_count": 1,
            "self_reported_parameter_after_sha256": "9" * 64,
            "self_reported_frame0_output_tensor_sha256": "9" * 64,
            "self_reported_rng_replay_after_sha256": "9" * 64,
            "core_code_sha256": "9" * 64,
        }
        for field, value in mutations.items():
            bad = _clone_receipts(self.receipts); bad[0] = dict(bad[0])
            bad[0][field] = value; _reseal(bad[0])
            with self.subTest(field=field), self.assertRaises(
                validator.V15BReceiptValidationError
            ):
                validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_every_arm_frame0_identity_and_audit_geometry_are_enforced(self):
        for arm_index in range(len(self.receipts)):
            bad = _clone_receipts(self.receipts); bad[arm_index] = dict(bad[arm_index])
            bad[arm_index]["self_reported_frame0_output_tensor_sha256"] = "9" * 64
            _reseal(bad[arm_index])
            with self.subTest(arm_index=arm_index), self.assertRaises(
                validator.V15BReceiptValidationError
            ):
                validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))
        bad = _clone_receipts(self.receipts); bad[1] = dict(bad[1])
        audits = list(bad[1]["self_reported_block_audits"])
        audits[0] = dict(audits[0]); audits[0]["tensor_head_dim"] += 1
        bad[1]["self_reported_block_audits"] = audits; _reseal(bad[1])
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))

    def test_unreopenable_execution_fields_are_explicitly_self_reported(self):
        forbidden_unqualified = {
            "model_checkpoint_sha256", "model_code_sha256", "noise_sha256",
            "candidate_schedule_sha256", "optimizer_update_count",
            "parameter_before_sha256", "frame0_output_tensor_sha256",
            "output_video_sha256", "block_audits",
        }
        self.assertFalse(forbidden_unqualified & validator.ARM_RECEIPT_FIELDS)
        self.assertIn("self_reported_model_checkpoint_sha256",
                      validator.ARM_RECEIPT_FIELDS)
        self.assertIn("self_reported_block_audits", validator.ARM_RECEIPT_FIELDS)

    def test_validator_recomputes_pair_and_consensus_not_boolean_claims(self):
        failed_edge = replace(
            self.fixture["report"].edge_metrics[-1], aligned_cosine=0.0,
            aligned_normalized_frobenius_distance=1.0,
        )
        forged_report = replace(
            self.fixture["report"],
            edge_metrics=self.fixture["report"].edge_metrics[:-1] + (failed_edge,),
        )
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(
                **self._kwargs(aligned_swap_report=forged_report)
            )
        forged_consensus = replace(
            self.fixture["consensus"],
            consensus_cosines=(1.0, 1.0, 0.0, 0.0),
            consensus_distances=(0.0, 0.0, 1.0, 1.0),
        )
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(
                **self._kwargs(four_anchor_consensus=forged_consensus)
            )

    def test_resealed_trace_asset_mutation_is_rejected_by_recomputation(self):
        traces = list(self.fixture["traces"].values())
        mutated = core.RoleContactTraceV15B.create(
            anchor_slot="v3", asset_sha256="2" * 64,
            extractor_code_sha256=EXTRACTOR_CODE_SHA,
            extractor_config_sha256=EXTRACTOR_CONFIG_SHA,
            energy=traces[-1].energy,
        )
        traces[-1] = mutated
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(traces=traces))

    def test_unsealed_receipt_mutation_is_rejected_before_semantics(self):
        bad = _clone_receipts(self.receipts); bad[0] = dict(bad[0])
        bad[0]["self_reported_output_video_sha256"] = "9" * 64
        with self.assertRaises(validator.V15BReceiptValidationError):
            validator.validate_four_arm_receipts_v15b(**self._kwargs(bad))


if __name__ == "__main__":
    unittest.main()

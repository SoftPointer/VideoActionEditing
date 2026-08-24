#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve()
METHOD_ROOT = HERE.parents[1]
TEST_ROOT = HERE.parent
for value in (str(METHOD_ROOT), str(TEST_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import oracle_regeneration_canary_v1 as oracle  # noqa: E402
import self_guided_action_field_v1 as sgaf  # noqa: E402


SOURCE_SHA = "1" * 64
ANCHOR_SHA = "4" * 64
ACTION_CAPTION_SHA = "5" * 64
ACTION_PROGRAM_SHA = "6" * 64


class ManualOracleManifestTests(unittest.TestCase):
    def _qualified_artifacts(self, root: Path, *, geometry=None):
        gate_path = root / "e02.gate.json"
        receipt_path = root / "e02.review.json"
        delete = [[] for _ in range(oracle.PHASE_COUNT)]
        create = [[] for _ in range(oracle.PHASE_COUNT)]
        delete[1] = [[0, 1]]
        create[1] = [[3, 1]]
        geometry = geometry or [1, 1, oracle.PHASE_COUNT, 2, 2]
        payload = {
            "latent_geometry": geometry,
            "flattening": "per_phase_row_major_yx",
            "delete_rle": delete,
            "create_rle": create,
            "dtype": "bool",
        }
        mask_sha = __import__("hashlib").sha256(
            oracle.canonical_json_bytes_v1(payload)
        ).hexdigest()
        leaf_payload = {
            "schema_version": oracle.ANNOTATION_LEAF_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": SOURCE_SHA,
            "anchor_sha256": ANCHOR_SHA,
            "action_caption_sha256": ACTION_CAPTION_SHA,
            "structured_action_program_sha256": ACTION_PROGRAM_SHA,
            "mask_sha256": mask_sha,
            "annotator": "annotator-a",
            "reviewer": "reviewer-b",
        }
        leaf_sha = oracle.annotation_authority_leaf_sha256_v1(leaf_payload)
        authority_root_sha = leaf_sha
        gate = {
            "schema_version": oracle.GATE_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": SOURCE_SHA,
            "anchor_sha256": ANCHOR_SHA,
            "action_caption_sha256": ACTION_CAPTION_SHA,
            "structured_action_program_sha256": ACTION_PROGRAM_SHA,
            "latent_geometry": geometry,
            "flattening": "per_phase_row_major_yx",
            "dtype": "bool",
            "hard_support": True,
            "phase_zero_empty": True,
            "delete_rle": delete,
            "create_rle": create,
            "mask_sha256": mask_sha,
            "annotation_authority": {
                "tree_shape": oracle.ANNOTATION_TREE_SHAPE,
                "ledger_root_sha256": authority_root_sha,
                "leaf_sha256": leaf_sha,
                "leaf_index": 0,
                "tree_size": 1,
                "inclusion_proof": [],
            },
            "authority": {
                "role": "manual_source_coordinate_diagnostic_intervention_only",
                "training_target_authorized": False,
                "action_representation_claimed": False,
                "forbidden_inputs_absent": {
                    "failed_active_video_or_latent": True,
                    "raw_anchor_source_pixel_or_latent_difference": True,
                    "predicted_soft_gate": True,
                },
            },
            "qualification": {
                "status": "qualified_manual_diagnostic_oracle",
                "annotator": "annotator-a",
                "reviewer": "reviewer-b",
                "review_receipt_path": str(receipt_path),
            },
        }
        gate_path.write_bytes(oracle.canonical_json_bytes_v1(gate))
        gate_sha = oracle.file_sha256_v1(gate_path)
        receipt = {
            "schema_version": oracle.RECEIPT_SCHEMA_VERSION,
            "case_id": "e02",
            "source_sha256": SOURCE_SHA,
            "anchor_sha256": ANCHOR_SHA,
            "action_caption_sha256": ACTION_CAPTION_SHA,
            "structured_action_program_sha256": ACTION_PROGRAM_SHA,
            "gate_manifest_sha256": gate_sha,
            "mask_sha256": mask_sha,
            "annotation_authority_root_sha256": authority_root_sha,
            "annotation_authority_leaf_sha256": leaf_sha,
            "reviewer": "reviewer-b",
            "accepted": True,
            "phase_zero_source_authority_checked": True,
            "delete_create_semantics_checked": True,
            "failed_active_used_to_author_mask": False,
            "anchor_difference_used_to_author_mask": False,
            "predicted_soft_gate_used_to_author_mask": False,
        }
        receipt_path.write_bytes(oracle.canonical_json_bytes_v1(receipt))
        return (
            gate_path,
            gate_sha,
            receipt_path,
            oracle.file_sha256_v1(receipt_path),
            authority_root_sha,
        )

    def test_manual_gate_requires_independent_hash_bound_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gate_path, gate_sha, _, receipt_sha, authority_root = self._qualified_artifacts(
                Path(temporary)
            )
            result = oracle.validate_oracle_gate_manifest_v1(
                gate_path,
                expected_file_sha256=gate_sha,
                expected_review_receipt_sha256=receipt_sha,
                expected_case_id="e02",
                expected_source_sha256=SOURCE_SHA,
                expected_anchor_sha256=ANCHOR_SHA,
                expected_action_caption_sha256=ACTION_CAPTION_SHA,
                expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                expected_annotation_authority_root_sha256=authority_root,
                expected_latent_geometry=(1, 1, oracle.PHASE_COUNT, 2, 2),
            )
            self.assertEqual(result.mask_sha256, json.loads(gate_path.read_text())["mask_sha256"])
            self.assertNotEqual(result.annotator, result.reviewer)

    def test_flattening_and_soft_gate_review_evidence_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, _, receipt_path, receipt_sha, authority_root = (
                self._qualified_artifacts(root)
            )
            gate = json.loads(gate_path.read_text())
            gate["flattening"] = "per_phase_column_major_xy"
            gate_path.write_bytes(oracle.canonical_json_bytes_v1(gate))
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "flattening"
            ):
                oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    expected_file_sha256=oracle.file_sha256_v1(gate_path),
                    expected_review_receipt_sha256=receipt_sha,
                    expected_case_id="e02",
                    expected_source_sha256=SOURCE_SHA,
                    expected_anchor_sha256=ANCHOR_SHA,
                    expected_action_caption_sha256=ACTION_CAPTION_SHA,
                    expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                    expected_annotation_authority_root_sha256=authority_root,
                )
            gate["flattening"] = "per_phase_row_major_yx"
            gate_path.write_bytes(oracle.canonical_json_bytes_v1(gate))
            gate_sha = oracle.file_sha256_v1(gate_path)
            receipt = json.loads(receipt_path.read_text())
            receipt["gate_manifest_sha256"] = gate_sha
            receipt.pop("predicted_soft_gate_used_to_author_mask")
            receipt_path.write_bytes(oracle.canonical_json_bytes_v1(receipt))
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "independent oracle review"
            ):
                oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    expected_file_sha256=gate_sha,
                    expected_review_receipt_sha256=oracle.file_sha256_v1(
                        receipt_path
                    ),
                    expected_case_id="e02",
                    expected_source_sha256=SOURCE_SHA,
                    expected_anchor_sha256=ANCHOR_SHA,
                    expected_action_caption_sha256=ACTION_CAPTION_SHA,
                    expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                    expected_annotation_authority_root_sha256=authority_root,
                )

    def test_pending_or_self_reviewed_gate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path, _, _, receipt_sha, authority_root = self._qualified_artifacts(root)
            gate = json.loads(gate_path.read_text())
            gate["qualification"]["reviewer"] = gate["qualification"]["annotator"]
            gate_path.write_bytes(oracle.canonical_json_bytes_v1(gate))
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "distinct"
            ):
                oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    expected_file_sha256=oracle.file_sha256_v1(gate_path),
                    expected_review_receipt_sha256=receipt_sha,
                    expected_case_id="e02",
                    expected_source_sha256=SOURCE_SHA,
                    expected_anchor_sha256=ANCHOR_SHA,
                    expected_action_caption_sha256=ACTION_CAPTION_SHA,
                    expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                    expected_annotation_authority_root_sha256=authority_root,
                )

    def test_spec_is_explicitly_fail_closed_and_has_e03_abstention(self) -> None:
        spec_path = METHOD_ROOT / "assets/oracle_regeneration_e02_e03_canary_v1.json"
        spec = json.loads(spec_path.read_text())
        self.assertFalse(spec["gpu_launch_authorized"])
        self.assertFalse(spec["activation_implemented_in_this_version"])
        self.assertFalse(spec["training_authorized"])
        self.assertFalse(
            spec["selection_contract"]["successful_base_may_be_replaced_automatically"]
        )
        e03 = next(row for row in spec["cases"] if row["case_id"] == "e03")
        self.assertIn("abstain", e03["candidate_policy"])
        self.assertIsNone(e03["manual_gate_manifest"])
        self.assertFalse(oracle.contract_v1()["native_and_flowedit_outer_samplers_proven_connected"])
        for raw_surface in (
            "HardStateChangeGateV1",
            "materialize_hard_gate_v1",
            "scheduled_local_velocity_v1",
            "flowedit_step0_target_noise_v1",
        ):
            self.assertNotIn(raw_surface, oracle.__all__)
            self.assertFalse(hasattr(oracle, raw_surface))
        self.assertFalse(oracle.contract_v1()["compiled_annotation_roots_present"])
        future_blockers = oracle.contract_v1()["future_activation_blockers"]
        self.assertGreaterEqual(len(future_blockers), 5)
        self.assertTrue(any("VAE-encoded provenance" in row for row in future_blockers))
        self.assertTrue(any("domain/seed/generator" in row for row in future_blockers))
        self.assertTrue(any("public-key signature" in row for row in future_blockers))

    def test_preflight_launcher_contains_no_compute_dispatch(self) -> None:
        launcher = METHOD_ROOT / "scripts/auh_preflight_oracle_regeneration_e02_e03_canary_v1.sh"
        source = launcher.read_text()
        self.assertNotIn("srun ", source)
        self.assertNotIn("torch.distributed", source)
        self.assertNotIn("ROCR_VISIBLE_DEVICES", source)
        self.assertIn("--require-launch-ready", source)

    def test_domain_separated_seed_is_case_and_candidate_specific(self) -> None:
        first = oracle.derive_regeneration_seed_v1(
            master_seed=42, case_id="e02", candidate_index=0
        )
        self.assertEqual(
            first,
            oracle.derive_regeneration_seed_v1(
                master_seed=42, case_id="e02", candidate_index=0
            ),
        )
        self.assertNotEqual(
            first,
            oracle.derive_regeneration_seed_v1(
                master_seed=42, case_id="e03", candidate_index=0
            ),
        )
        self.assertNotEqual(
            first,
            oracle.derive_regeneration_seed_v1(
                master_seed=42, case_id="e02", candidate_index=1
            ),
        )

    def test_wrong_instruction_or_unpinned_authority_root_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gate_path, gate_sha, _, receipt_sha, authority_root = (
                self._qualified_artifacts(Path(temporary))
            )
            common = {
                "expected_file_sha256": gate_sha,
                "expected_review_receipt_sha256": receipt_sha,
                "expected_case_id": "e02",
                "expected_source_sha256": SOURCE_SHA,
                "expected_anchor_sha256": ANCHOR_SHA,
                "expected_action_caption_sha256": ACTION_CAPTION_SHA,
                "expected_structured_action_program_sha256": ACTION_PROGRAM_SHA,
                "expected_annotation_authority_root_sha256": authority_root,
            }
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "action-caption"
            ):
                oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    **{**common, "expected_action_caption_sha256": "7" * 64},
                )
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "authority"
            ):
                oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    **{
                        **common,
                        "expected_annotation_authority_root_sha256": "8" * 64,
                    },
                )

    def test_torch112_storage_pointer_fallback_keeps_exact_pointer(self) -> None:
        class Storage:
            def data_ptr(self):
                return 123456

        class LegacyTensorLike:
            def storage(self):
                return Storage()

        self.assertEqual(
            oracle._storage_data_ptr_compat_v1(LegacyTensorLike()), 123456
        )

    def test_duplicate_json_keys_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"schema_version":"a","schema_version":"b"}')
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "duplicate JSON key"
            ):
                oracle.strict_json_load_path_v1(path, label="unit")


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class HardGateTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def _gate(self, *, null: bool = False) -> "oracle._OwnedHardStateChangeGateV1":
        torch = self.torch
        delete = torch.zeros(1, 1, oracle.PHASE_COUNT, 2, 2, dtype=torch.bool)
        create = torch.zeros_like(delete)
        if not null:
            delete[:, :, 1, 0, 0] = True
            create[:, :, 2, 1, 1] = True
        support = torch.logical_or(delete, create).contiguous()
        preserve = torch.logical_not(support).contiguous()
        variant = "null" if null else "union"
        realized_sha = oracle.realized_gate_sha256_v1(
            delete=delete,
            create=create,
            support=support,
            preserve=preserve,
            source_mask_sha256="2" * 64,
            variant=variant,
        )
        return oracle._OwnedHardStateChangeGateV1(
            delete=delete,
            create=create,
            support=support,
            preserve=preserve,
            provenance="unit-manual-reviewed",
            source_mask_sha256="2" * 64,
            realized_gate_sha256=realized_sha,
            variant=variant,
            source_delete_count=1,
            source_create_count=1,
            realized_delete_count=0 if null else 1,
            realized_create_count=0 if null else 1,
            permutation_mass_preserved=None if null else True,
        )

    def test_null_gate_returns_original_signed_zero_object(self) -> None:
        torch = self.torch
        shape = (1, 16, oracle.PHASE_COUNT, 2, 2)
        packed_shape = (1, oracle.PHASE_COUNT, 64)
        official = torch.full(packed_shape, -0.0, dtype=torch.float32)
        sample = torch.zeros_like(official)
        executed, trace = oracle._scheduled_local_velocity_v1(
            sample=sample,
            high_r2v4_velocity=object(),
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=self._gate(null=True),
            target_latent_shape=shape,
        )
        self.assertIs(executed, official)
        self.assertTrue(trace["scheduler_received_original_official_object"])
        self.assertFalse(trace["scheduled_expert_evaluated"])
        self.assertFalse(trace["high_velocity_aggregated"])
        self.assertTrue(
            torch.equal(
                executed.contiguous().view(torch.uint8),
                official.contiguous().view(torch.uint8),
            )
        )

    def test_owned_gate_mutation_and_overlapping_flow_storage_fail_closed(self) -> None:
        torch = self.torch
        gate = self._gate()
        gate.support[:, :, 3, 0, 0] = True
        with self.assertRaisesRegex(
            oracle.OracleRegenerationCanaryError, "support|SHA-256"
        ):
            oracle._validate_owned_hard_gate_v1(gate)
        base = torch.zeros(400, dtype=torch.float32)
        shape = (1, 2, oracle.PHASE_COUNT, 2, 2)
        numel = 1 * 2 * oracle.PHASE_COUNT * 2 * 2
        source = base[:numel].view(shape)
        overlapping = base[1 : numel + 1].view(shape)
        other_a = torch.zeros(shape, dtype=torch.float32)
        other_b = torch.ones(shape, dtype=torch.float32)
        with self.assertRaisesRegex(
            oracle.OracleRegenerationCanaryError, "overlap"
        ):
            oracle._validate_flowedit_tensor_set_v1(
                (source, overlapping, other_a, other_b)
            )

    def test_exact_timestep_alias_certifier_supports_zero_stride_only(self) -> None:
        torch = self.torch
        scalar = torch.tensor(900, dtype=torch.int64)
        oracle._certify_expanded_timestep_compat_v1(scalar.expand(1), scalar)
        with self.assertRaisesRegex(
            oracle.OracleRegenerationCanaryError, "zero-stride"
        ):
            oracle._certify_expanded_timestep_compat_v1(scalar.clone().reshape(1), scalar)

    def test_spatial_control_is_same_mass_roll_with_distinct_realized_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ManualOracleManifestTests()._qualified_artifacts(
                Path(temporary)
            )
            gate_path, gate_sha, _, receipt_sha, authority_root = artifacts
            manifest = oracle.validate_oracle_gate_manifest_v1(
                gate_path,
                expected_file_sha256=gate_sha,
                expected_review_receipt_sha256=receipt_sha,
                expected_case_id="e02",
                expected_source_sha256=SOURCE_SHA,
                expected_anchor_sha256=ANCHOR_SHA,
                expected_action_caption_sha256=ACTION_CAPTION_SHA,
                expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                expected_annotation_authority_root_sha256=authority_root,
            )
            union = oracle._materialize_owned_hard_gate_v1(manifest, variant="union")
            shifted = oracle._materialize_owned_hard_gate_v1(
                manifest, variant="spatial_shift"
            )
            self.assertEqual(union.realized_delete_count, shifted.realized_delete_count)
            self.assertEqual(union.realized_create_count, shifted.realized_create_count)
            self.assertTrue(shifted.permutation_mass_preserved)
            self.assertNotEqual(
                union.realized_gate_sha256, shifted.realized_gate_sha256
            )

    def test_checked_in_execution_trust_anchors_are_empty_and_block_minting(self) -> None:
        self.assertEqual(dict(oracle.COMPILED_ANNOTATION_AUTHORITY_ROOTS), {})
        self.assertEqual(dict(oracle.COMPILED_NATIVE_BINDING_RECEIPT_SHA256), {})
        self.assertEqual(dict(oracle.COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256), {})
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = ManualOracleManifestTests()._qualified_artifacts(
                Path(temporary)
            )
            gate_path, gate_sha, _, receipt_sha, authority_root = artifacts
            manifest = oracle.validate_oracle_gate_manifest_v1(
                gate_path,
                expected_file_sha256=gate_sha,
                expected_review_receipt_sha256=receipt_sha,
                expected_case_id="e02",
                expected_source_sha256=SOURCE_SHA,
                expected_anchor_sha256=ANCHOR_SHA,
                expected_action_caption_sha256=ACTION_CAPTION_SHA,
                expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                expected_annotation_authority_root_sha256=authority_root,
            )
            with self.assertRaisesRegex(
                oracle.OracleRegenerationCanaryError, "not compiled"
            ):
                oracle.validate_native_execution_binding_receipt_v1(
                    gate_path,
                    expected_receipt_sha256=gate_sha,
                    validated_gate_manifest=manifest,
                    gate_variant="union",
                    sample_id="must-block",
                    source_video_latent=None,
                    source_reference_latents=(),
                    source_reference_rgb_indices=(),
                    r2v_action_prompt_embeds=None,
                )

    def test_local_scheduled_route_is_byte_exact_outside_bool_support(self) -> None:
        torch = self.torch
        shape = (1, 16, oracle.PHASE_COUNT, 2, 2)
        packed_shape = (1, oracle.PHASE_COUNT, 64)
        official = torch.full(packed_shape, -0.0, dtype=torch.float32)
        high = torch.ones_like(official)
        sample = torch.zeros_like(official)
        gate = self._gate()
        executed, trace = oracle._scheduled_local_velocity_v1(
            sample=sample,
            high_r2v4_velocity=high,
            official_v2v_velocity=official,
            sigma=torch.tensor(1.0, dtype=torch.float32),
            gate=gate,
            target_latent_shape=shape,
        )
        packed_support = sgaf._spatial_to_packed(
            gate.support.expand(shape), shape
        )
        outside = torch.logical_not(packed_support)
        self.assertTrue(
            torch.equal(
                executed[outside].contiguous().view(torch.uint8),
                official[outside].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(executed[packed_support], high[packed_support]))
        self.assertEqual(trace["endpoint"], "high_r2v4_apg")
        late, late_trace = oracle._scheduled_local_velocity_v1(
            sample=sample,
            high_r2v4_velocity=high,
            official_v2v_velocity=official,
            sigma=torch.tensor(0.1, dtype=torch.float32),
            gate=gate,
            target_latent_shape=shape,
        )
        self.assertIs(late, official)
        self.assertEqual(late_trace["endpoint"], "low_official_v2v_apg")

    def test_soft_dense_gate_is_rejected(self) -> None:
        torch = self.torch
        gate = self._gate()
        invalid = oracle._OwnedHardStateChangeGateV1(
            delete=gate.delete.float(),
            create=gate.create.float(),
            support=gate.support.float() * 0.01,
            preserve=1.0 - gate.support.float() * 0.01,
            provenance=gate.provenance,
            source_mask_sha256=gate.source_mask_sha256,
            realized_gate_sha256=gate.realized_gate_sha256,
        )
        with self.assertRaisesRegex(
            oracle.OracleRegenerationCanaryError, "exact bool"
        ):
            oracle._validate_owned_hard_gate_v1(invalid)

    def test_flowedit_step0_changes_target_only_inside_G(self) -> None:
        torch = self.torch

        def constructor(source, edit, noise, *, sigma):
            source_state = (1.0 - sigma) * source + sigma * noise
            target_state = edit + source_state - source
            return source_state, target_state

        shape = (1, 2, oracle.PHASE_COUNT, 2, 2)
        source = torch.zeros(shape, dtype=torch.float32)
        edit = torch.full(shape, -0.0, dtype=torch.float32)
        correlated = torch.full(shape, -0.0, dtype=torch.float32)
        independent = torch.ones(shape, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = ManualOracleManifestTests()._qualified_artifacts(root)
            gate_path, gate_sha, _, gate_receipt_sha, authority_root = artifacts
            manifest = oracle.validate_oracle_gate_manifest_v1(
                gate_path,
                expected_file_sha256=gate_sha,
                expected_review_receipt_sha256=gate_receipt_sha,
                expected_case_id="e02",
                expected_source_sha256=SOURCE_SHA,
                expected_anchor_sha256=ANCHOR_SHA,
                expected_action_caption_sha256=ACTION_CAPTION_SHA,
                expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                expected_annotation_authority_root_sha256=authority_root,
            )
            gate = oracle._materialize_owned_hard_gate_v1(
                manifest, variant="union"
            )
            receipt_path = Path(temporary) / "flowedit.review.json"
            constructor_path = Path(__file__).resolve()
            receipt = {
                "schema_version": oracle.FLOWEDIT_RECEIPT_SCHEMA_VERSION,
                "constructor_module": constructor.__module__,
                "constructor_qualname": constructor.__qualname__,
                "constructor_file_path": str(constructor_path),
                "constructor_file_sha256": oracle.file_sha256_v1(constructor_path),
                "case_id": "e02",
                "sample_id": "unit-flowedit-step0",
                "source_sha256": SOURCE_SHA,
                "anchor_sha256": ANCHOR_SHA,
                "action_caption_sha256": ACTION_CAPTION_SHA,
                "structured_action_program_sha256": ACTION_PROGRAM_SHA,
                "gate_manifest_sha256": manifest.file_sha256,
                "gate_review_receipt_sha256": manifest.review_receipt_sha256,
                "annotation_authority_root_sha256": authority_root,
                "gate_variant": "union",
                "source_tensor_sha256": oracle.tensor_content_sha256_v1(source),
                "edit_tensor_sha256": oracle.tensor_content_sha256_v1(edit),
                "source_correlated_noise_sha256": oracle.tensor_content_sha256_v1(
                    correlated
                ),
                "independent_target_noise_sha256": oracle.tensor_content_sha256_v1(
                    independent
                ),
                "realized_gate_sha256": gate.realized_gate_sha256,
                "sigma_float64_hex": float(1.0).hex(),
                "step_index": 0,
                "source_tensor_role": "bound_source_latent_not_target_derived",
                "edit_tensor_role": "bound_current_edit_state_not_teacher_target",
                "target_video_or_latent_used": False,
                "diagnostic_only": True,
                "training_target_authorized": False,
                "accepted": True,
            }
            receipt_path.write_bytes(oracle.canonical_json_bytes_v1(receipt))
            receipt_sha = oracle.file_sha256_v1(receipt_path)
            with mock.patch.object(
                oracle,
                "COMPILED_ANNOTATION_AUTHORITY_ROOTS",
                {"e02": authority_root},
            ), mock.patch.object(
                oracle,
                "COMPILED_FLOWEDIT_BINDING_RECEIPT_SHA256",
                {"e02": receipt_sha},
            ):
                execution = oracle.validate_flowedit_execution_receipt_v1(
                    receipt_path,
                    expected_receipt_sha256=receipt_sha,
                    flowedit_constructor=constructor,
                    expected_constructor_file_sha256=oracle.file_sha256_v1(
                        constructor_path
                    ),
                    validated_gate_manifest=manifest,
                    gate_variant="union",
                    sample_id="unit-flowedit-step0",
                    source=source,
                    edit=edit,
                    source_correlated_noise=correlated,
                    independent_target_noise=independent,
                    sigma=1.0,
                    step_index=0,
                )
                source_state, target, trace = oracle._flowedit_step0_target_noise_v1(
                    source=source,
                    edit=edit,
                    source_correlated_noise=correlated,
                    independent_target_noise=independent,
                    sigma=1.0,
                    step_index=0,
                    gate=gate,
                    execution=execution,
                )
        matched_source, matched_target = constructor(
            source, edit, correlated, sigma=1.0
        )
        support = gate.support.expand_as(source)
        self.assertTrue(torch.equal(source_state, matched_source))
        self.assertTrue(torch.equal(target[support], torch.ones_like(target[support])))
        self.assertTrue(
            torch.equal(
                target[~support].contiguous().view(torch.uint8),
                matched_target[~support].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                target[:, :, 0].contiguous().view(torch.uint8),
                matched_target[:, :, 0].contiguous().view(torch.uint8),
            )
        )
        self.assertTrue(trace["outside_hard_support_byte_exact"])
        self.assertEqual(trace["flowedit_constructor_calls"], 2)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class FiveForwardLocalRuntimeTests(unittest.TestCase):
    def _write_native_binding_receipt(
        self,
        root: Path,
        *,
        manifest,
        gate_variant,
        sample_kwargs,
        action_prompt,
    ):
        source = sample_kwargs["multi_video_vae_latents"][0]
        references = sample_kwargs["multi_image_vae_latents"]
        gate = oracle._materialize_owned_hard_gate_v1(
            manifest, variant=gate_variant
        )
        path = root / f"native-{gate_variant}.binding.json"
        payload = {
            "schema_version": oracle.NATIVE_BINDING_RECEIPT_SCHEMA_VERSION,
            "case_id": manifest.case_id,
            "sample_id": f"unit-{gate_variant}",
            "source_sha256": manifest.source_sha256,
            "anchor_sha256": manifest.anchor_sha256,
            "action_caption_sha256": manifest.action_caption_sha256,
            "structured_action_program_sha256": (
                manifest.structured_action_program_sha256
            ),
            "gate_manifest_sha256": manifest.file_sha256,
            "gate_review_receipt_sha256": manifest.review_receipt_sha256,
            "annotation_authority_root_sha256": (
                manifest.annotation_authority_root_sha256
            ),
            "annotation_authority_leaf_sha256": (
                manifest.annotation_authority_leaf_sha256
            ),
            "gate_variant": gate_variant,
            "realized_gate_sha256": gate.realized_gate_sha256,
            "source_latent_sha256": oracle.tensor_content_sha256_v1(source),
            "source_reference_latent_sha256": [
                oracle.tensor_content_sha256_v1(value) for value in references
            ],
            "source_reference_rgb_indices": [0, 27, 53, 80],
            "r2v_action_prompt_sha256": oracle.tensor_content_sha256_v1(
                action_prompt
            ),
            "source_latent_role": "official_vae_encode_of_bound_source_media",
            "r2v_action_prompt_role": "bound_action_caption_not_target_derived",
            "target_video_or_latent_used": False,
            "diagnostic_only": True,
            "training_target_authorized": False,
            "accepted": True,
        }
        path.write_bytes(oracle.canonical_json_bytes_v1(payload))
        return path, oracle.file_sha256_v1(path)

    def test_fake_native_runtime_keeps_official_outside_G_exact40(self) -> None:
        import torch
        import test_native_branch_homotopy_runtime_v1 as frozen_test

        fixture = frozen_test.RuntimePatchTests(methodName="runTest")
        fixture.torch = torch
        fixture.setUp()
        try:
            diffusion = fixture._diffusion()
            config = fixture._config()
            sample_kwargs = fixture._sample_kwargs(diffusion)
            with tempfile.TemporaryDirectory() as temporary:
                artifacts = ManualOracleManifestTests()._qualified_artifacts(
                    Path(temporary), geometry=list(config.target_latent_shape[:1])
                    + [1]
                    + list(config.target_latent_shape[2:])
                )
                gate_path, gate_sha, _, receipt_sha, authority_root = artifacts
                manifest = oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    expected_file_sha256=gate_sha,
                    expected_review_receipt_sha256=receipt_sha,
                    expected_case_id="e02",
                    expected_source_sha256=SOURCE_SHA,
                    expected_anchor_sha256=ANCHOR_SHA,
                    expected_action_caption_sha256=ACTION_CAPTION_SHA,
                    expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                    expected_annotation_authority_root_sha256=authority_root,
                    expected_latent_geometry=(1, 1, oracle.PHASE_COUNT, 2, 2),
                )
                binding_path, binding_sha = self._write_native_binding_receipt(
                    Path(temporary),
                    manifest=manifest,
                    gate_variant="union",
                    sample_kwargs=sample_kwargs,
                    action_prompt=fixture.high_action,
                )
                with mock.patch.object(
                    oracle,
                    "COMPILED_ANNOTATION_AUTHORITY_ROOTS",
                    {"e02": authority_root},
                ), mock.patch.object(
                    oracle,
                    "COMPILED_NATIVE_BINDING_RECEIPT_SHA256",
                    {"e02": binding_sha},
                ):
                    binding = oracle.validate_native_execution_binding_receipt_v1(
                        binding_path,
                        expected_receipt_sha256=binding_sha,
                        validated_gate_manifest=manifest,
                        gate_variant="union",
                        sample_id="unit-union",
                        source_video_latent=sample_kwargs[
                            "multi_video_vae_latents"
                        ][0],
                        source_reference_latents=sample_kwargs[
                            "multi_image_vae_latents"
                        ],
                        source_reference_rgb_indices=(0, 27, 53, 80),
                        r2v_action_prompt_embeds=fixture.high_action,
                    )
                    patch = oracle.LocalOracleNativeBranchRuntimePatchV1(
                        diffusion,
                        config=config,
                        native_execution_binding=binding,
                    )
                    support = patch._owned_hard_gate.support
                    patch.install()
                    try:
                        diffusion.sample(**sample_kwargs)
                    finally:
                        patch.restore()
                    live_source = sample_kwargs["multi_video_vae_latents"][0]
                    saved_source = live_source.clone()
                    with torch.no_grad():
                        live_source.add_(1.0)
                    with self.assertRaisesRegex(
                        Exception, "live source/reference/action tensors"
                    ):
                        patch.finalize()
                    with torch.no_grad():
                        live_source.copy_(saved_source)
                    receipt = patch.finalize()
            self.assertEqual(receipt["transformer_forwards"], 200)
            self.assertEqual(receipt["original_scheduler_calls"], 40)
            self.assertTrue(receipt["outside_G_official_bytes_exact_all_steps"])
            self.assertFalse(receipt["all40_raw_high_mode_available"])
            self.assertIsNone(receipt["selection_authority"])
            self.assertFalse(receipt["realized_gate_is_G_zero"])
            self.assertTrue(all(row["high_forwards_executed"] for row in receipt["trace"]))
            self.assertIsNone(
                receipt["realized_G_zero_direct_official_object_all_steps"]
            )
            packed_support = sgaf._spatial_to_packed(
                support.expand(config.target_latent_shape),
                config.target_latent_shape,
            )
            outside = torch.logical_not(packed_support)
            for received, official in zip(
                diffusion.scheduler.received_objects, diffusion.official_outputs
            ):
                self.assertTrue(
                    torch.equal(
                        received[outside].contiguous().view(torch.uint8),
                        official[outside].contiguous().view(torch.uint8),
                    )
                )
            for index in range(31, 40):
                self.assertIs(
                    diffusion.scheduler.received_objects[index],
                    diffusion.official_outputs[index],
                )
        finally:
            fixture.tearDown()

    def test_fake_native_null_gate_preserves_signed_zero_objects_exact40(self) -> None:
        import torch
        import test_native_branch_homotopy_runtime_v1 as frozen_test

        fixture = frozen_test.RuntimePatchTests(methodName="runTest")
        fixture.torch = torch
        fixture.setUp()
        try:
            diffusion = fixture._diffusion()
            config = fixture._config()
            sample_kwargs = fixture._sample_kwargs(diffusion)
            with tempfile.TemporaryDirectory() as temporary:
                artifacts = ManualOracleManifestTests()._qualified_artifacts(
                    Path(temporary),
                    geometry=[1, 1, oracle.PHASE_COUNT, 2, 2],
                )
                gate_path, gate_sha, _, receipt_sha, authority_root = artifacts
                manifest = oracle.validate_oracle_gate_manifest_v1(
                    gate_path,
                    expected_file_sha256=gate_sha,
                    expected_review_receipt_sha256=receipt_sha,
                    expected_case_id="e02",
                    expected_source_sha256=SOURCE_SHA,
                    expected_anchor_sha256=ANCHOR_SHA,
                    expected_action_caption_sha256=ACTION_CAPTION_SHA,
                    expected_structured_action_program_sha256=ACTION_PROGRAM_SHA,
                    expected_annotation_authority_root_sha256=authority_root,
                    expected_latent_geometry=(1, 1, oracle.PHASE_COUNT, 2, 2),
                )
                binding_path, binding_sha = self._write_native_binding_receipt(
                    Path(temporary),
                    manifest=manifest,
                    gate_variant="null",
                    sample_kwargs=sample_kwargs,
                    action_prompt=fixture.high_action,
                )

                def signed_zero_guided(*args, **kwargs):
                    output = torch.full_like(kwargs["output_like"], -0.0)
                    kwargs["momentum_buffer"].update(torch.zeros_like(output))
                    return output

                with mock.patch.object(
                    oracle,
                    "COMPILED_ANNOTATION_AUTHORITY_ROOTS",
                    {"e02": authority_root},
                ), mock.patch.object(
                    oracle,
                    "COMPILED_NATIVE_BINDING_RECEIPT_SHA256",
                    {"e02": binding_sha},
                ):
                    binding = oracle.validate_native_execution_binding_receipt_v1(
                        binding_path,
                        expected_receipt_sha256=binding_sha,
                        validated_gate_manifest=manifest,
                        gate_variant="null",
                        sample_id="unit-null",
                        source_video_latent=sample_kwargs[
                            "multi_video_vae_latents"
                        ][0],
                        source_reference_latents=sample_kwargs[
                            "multi_image_vae_latents"
                        ],
                        source_reference_rgb_indices=(0, 27, 53, 80),
                        r2v_action_prompt_embeds=fixture.high_action,
                    )
                    patch = oracle.LocalOracleNativeBranchRuntimePatchV1(
                        diffusion,
                        config=config,
                        native_execution_binding=binding,
                    )
                    patch.install()
                    try:
                        with mock.patch.object(
                            sgaf, "_guided_velocity", side_effect=signed_zero_guided
                        ):
                            diffusion.sample(**sample_kwargs)
                    finally:
                        patch.restore()
                    receipt = patch.finalize()
            self.assertTrue(receipt["realized_gate_is_G_zero"])
            self.assertTrue(
                receipt["realized_G_zero_direct_official_object_all_steps"]
            )
            self.assertTrue(receipt["G_zero_direct_official_object_capability"])
            self.assertTrue(all(row["high_forwards_executed"] for row in receipt["trace"]))
            self.assertTrue(
                all(not row["high_velocity_aggregated"] for row in receipt["trace"])
            )
            for received, official in zip(
                diffusion.scheduler.received_objects, diffusion.official_outputs
            ):
                self.assertIs(received, official)
                self.assertEqual(
                    oracle._tensor_raw_bytes_v1(received),
                    struct.pack("=f", -0.0) * int(received.numel()),
                )
                self.assertTrue(oracle._tensor_bytes_equal_v1(received, official))
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()

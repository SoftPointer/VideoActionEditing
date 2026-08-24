#!/usr/bin/env python3
"""Hostile tests for the pure-tensor counterfactual patient carrier."""

from __future__ import annotations

from dataclasses import replace
import json
import mmap
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "methods" / "bernini_action_editing"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import object_instance_counterfactual_carrier_v1 as carrier  # noqa: E402


def _phases() -> tuple[carrier.PhaseCorrespondence, ...]:
    return (
        carrier.PhaseCorrespondence(
            phase_index=0,
            phase_id="phase0",
            regime="pre_lift",
            phase_tokens=(0, 1),
            correspondence=((0, 0),),
            target_complement=(1,),
        ),
        carrier.PhaseCorrespondence(
            phase_index=1,
            phase_id="phase1",
            regime="lift",
            phase_tokens=(2, 3),
            correspondence=((2, 3),),
            target_complement=(2,),
        ),
        carrier.PhaseCorrespondence(
            phase_index=2,
            phase_id="phase2",
            regime="hold",
            phase_tokens=(4, 5),
            correspondence=((4, 5),),
            target_complement=(4,),
        ),
    )


def _inputs(
    *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    origin = (
        torch.arange(6 * carrier.PACKED_CHANNELS, dtype=torch.float32)
        .reshape(1, 6, carrier.PACKED_CHANNELS)
        .to(dtype=dtype)
        .clone()
        .contiguous()
    )
    source = origin.clone().contiguous()
    source[:, 0, :] += 1
    source[:, 2, :] += 2
    source[:, 4, :] += 3
    # A non-patient difference proves that the mask, rather than a whole-state
    # copy, controls the carrier.
    source[:, 1, :] += 17
    mask = torch.tensor((True, False, True, False, True, False))
    return source, origin, mask


def _seal(
    source: torch.Tensor,
    origin: torch.Tensor,
    mask: torch.Tensor,
    phases: tuple[carrier.PhaseCorrespondence, ...] | None = None,
) -> carrier.PatientCarrierAuthority:
    return carrier.seal_patient_carrier_authority(
        source_packed=source,
        bone_removed_packed=origin,
        source_mask=mask,
        phases=_phases() if phases is None else phases,
    )


def _build(
    source: torch.Tensor,
    origin: torch.Tensor,
    mask: torch.Tensor,
    authority: carrier.PatientCarrierAuthority,
    phases: tuple[carrier.PhaseCorrespondence, ...] | None = None,
) -> carrier.PatientCarrierResult:
    return carrier.build_counterfactual_patient_carrier(
        source_packed=source,
        bone_removed_packed=origin,
        source_mask=mask,
        phases=_phases() if phases is None else phases,
        authority=authority,
    )


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(key) or _contains_tensor(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


class CounterfactualPatientCarrierTests(unittest.TestCase):
    def test_string_subclasses_are_rejected_at_authority_replay(self) -> None:
        class StringSubclass(str):
            pass

        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        forged_authorities = (
            replace(
                authority,
                schema_version=StringSubclass(authority.schema_version),
            ),
            replace(
                authority,
                source_role=StringSubclass(authority.source_role),
            ),
            replace(
                authority,
                origin_role=StringSubclass(authority.origin_role),
            ),
            replace(
                authority,
                source=replace(
                    authority.source,
                    device_type=StringSubclass(authority.source.device_type),
                ),
            ),
        )
        for forged in forged_authorities:
            with self.subTest(forged=forged), self.assertRaises(
                carrier.ObjectInstanceCounterfactualCarrierError
            ):
                _build(source, origin, mask, forged)

    def test_exact_formula_origin_target_and_complement(self) -> None:
        source, origin, mask = _inputs()
        source_before = source.clone()
        origin_before = origin.clone()
        mask_before = mask.clone()
        versions_before = (source._version, origin._version, mask._version)
        authority = _seal(source, origin, mask)
        result = _build(source, origin, mask, authority)

        expected_residual = torch.zeros_like(origin)
        expected_residual[:, 0, :] = source[:, 0, :] - origin[:, 0, :]
        expected_residual[:, 3, :] = source[:, 2, :] - origin[:, 2, :]
        expected_residual[:, 5, :] = source[:, 4, :] - origin[:, 4, :]
        expected = origin.clone()
        expected[:, 0, :] = origin[:, 0, :] + expected_residual[:, 0, :]
        expected[:, 3, :] = origin[:, 3, :] + expected_residual[:, 3, :]
        expected[:, 5, :] = origin[:, 5, :] + expected_residual[:, 5, :]

        self.assertTrue(_byte_equal(result.transported_residual, expected_residual))
        self.assertTrue(_byte_equal(result.counterfactual, expected))
        self.assertTrue(
            _byte_equal(
                result.counterfactual[:, (1, 2, 4), :],
                origin[:, (1, 2, 4), :],
            )
        )
        # The unmasked source-only difference at token 1 is not transported.
        self.assertTrue(_byte_equal(result.counterfactual[:, 1, :], origin[:, 1, :]))
        self.assertTrue(_byte_equal(source, source_before))
        self.assertTrue(_byte_equal(origin, origin_before))
        self.assertTrue(_byte_equal(mask, mask_before))
        self.assertEqual((source._version, origin._version, mask._version), versions_before)

        receipt = result.audit_receipt()
        self.assertFalse(_contains_tensor(receipt))
        json.dumps(receipt, sort_keys=True, allow_nan=False)
        self.assertTrue(receipt["source_is_not_aux"])
        self.assertTrue(receipt["caller_inputs_copied_to_private_snapshots"])
        self.assertFalse(receipt["caller_backing_independence_authenticated"])
        self.assertTrue(receipt["working_snapshot_storages_pairwise_distinct"])
        self.assertTrue(receipt["working_snapshots_unmutated"])
        self.assertTrue(receipt["all_target_residuals_byte_exact"])
        self.assertTrue(receipt["all_complements_byte_exact_z0"])
        self.assertTrue(receipt["all_lift_hold_origins_byte_exact_z0"])
        self.assertTrue(receipt["single_target_occupancy"])
        self.assertFalse(receipt["renderer_integration"])
        self.assertFalse(receipt["visual_success_claimed"])
        self.assertRegex(receipt["trace_digest"], r"^[0-9a-f]{64}$")

    def test_deterministic_authority_trace_and_cloned_replay(self) -> None:
        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        first = _build(source, origin, mask, authority)
        second = _build(
            source.clone().contiguous(),
            origin.clone().contiguous(),
            mask.clone().contiguous(),
            authority,
        )
        self.assertEqual(first.trace, second.trace)
        self.assertTrue(_byte_equal(first.counterfactual, second.counterfactual))
        self.assertEqual(
            authority.authority_digest,
            _seal(source, origin, mask).authority_digest,
        )

    def test_supported_dtypes_preserve_exact_transport(self) -> None:
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            with self.subTest(dtype=str(dtype)):
                source, origin, mask = _inputs(dtype=dtype)
                result = _build(source, origin, mask, _seal(source, origin, mask))
                donor_residual = source[:, 2, :] - origin[:, 2, :]
                self.assertTrue(
                    _byte_equal(result.transported_residual[:, 3, :], donor_residual)
                )
                self.assertEqual(result.counterfactual.dtype, dtype)

    def test_inference_mode_tensors_use_exact_byte_mutation_witness(self) -> None:
        with torch.inference_mode():
            source, origin, mask = _inputs()
            authority = _seal(source, origin, mask)
            result = _build(source, origin, mask, authority)
            self.assertTrue(result.trace.working_snapshots_unmutated)
            self.assertTrue(result.trace.all_target_residuals_byte_exact)

    def test_pre_lift_output_source_equality_is_observed_not_claimed(self) -> None:
        source, origin, mask = _inputs()
        origin[:, 0, :] = 1.0e20
        source[:, 0, :] = 1.0
        result = _build(source, origin, mask, _seal(source, origin, mask))
        self.assertFalse(_byte_equal(result.counterfactual[:, 0, :], source[:, 0, :]))
        self.assertFalse(result.trace.phases[0].pre_lift_output_byte_equal_source)
        self.assertTrue(result.trace.phases[0].target_residual_byte_exact)
        self.assertIsNone(result.trace.phases[1].pre_lift_output_byte_equal_source)

    def test_source_equal_to_auxiliary_is_rejected_even_without_alias(self) -> None:
        _, origin, mask = _inputs()
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "equals the bone-removed auxiliary",
        ):
            _seal(origin.clone().contiguous(), origin, mask)

    def test_wrong_source_donor_is_rejected_by_sealed_authority(self) -> None:
        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        wrong = source.clone().contiguous()
        wrong[:, 0, 0] += 9
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "differ from sealed authority",
        ):
            _build(wrong, origin, mask, authority)

    def test_auxiliary_substituted_for_source_is_rejected_at_replay(self) -> None:
        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        with self.assertRaises(carrier.ObjectInstanceCounterfactualCarrierError):
            _build(origin.clone().contiguous(), origin, mask, authority)

    def test_wrong_or_mutated_mask_is_rejected(self) -> None:
        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        for index in (0, 1):
            with self.subTest(index=index):
                wrong = mask.clone().contiguous()
                wrong[index] = ~wrong[index]
                with self.assertRaisesRegex(
                    carrier.ObjectInstanceCounterfactualCarrierError,
                    "source mask differs",
                ):
                    _build(source, origin, wrong, authority)

        mutated = mask.clone().contiguous()
        mutated_authority = _seal(source, origin, mutated)
        mutated[0] = False
        with self.assertRaises(carrier.ObjectInstanceCounterfactualCarrierError):
            _build(source, origin, mutated, mutated_authority)

    def test_post_seal_source_or_origin_mutation_is_rejected(self) -> None:
        for which in ("source", "origin"):
            with self.subTest(which=which):
                source, origin, mask = _inputs()
                authority = _seal(source, origin, mask)
                value = source if which == "source" else origin
                value[:, 0, 0] += 1
                with self.assertRaises(carrier.ObjectInstanceCounterfactualCarrierError):
                    _build(source, origin, mask, authority)

    def test_wrong_semantic_roles_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        for field, kwargs in (
            ("source", {"source_role": "aux_bone_removed_source"}),
            ("origin", {"origin_role": "exact_source_patient_zs"}),
        ):
            with self.subTest(field=field), self.assertRaises(
                carrier.ObjectInstanceCounterfactualCarrierError
            ):
                carrier.seal_patient_carrier_authority(
                    source_packed=source,
                    bone_removed_packed=origin,
                    source_mask=mask,
                    phases=_phases(),
                    **kwargs,
                )

    def test_duplicate_correspondence_rows_and_endpoints_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        bad_rows = []
        duplicate_pair = list(_phases())
        duplicate_pair[1] = replace(
            duplicate_pair[1], correspondence=((2, 3), (2, 3))
        )
        bad_rows.append(tuple(duplicate_pair))
        duplicate_target = list(_phases())
        duplicate_target[1] = replace(
            duplicate_target[1], correspondence=((2, 3), (3, 3))
        )
        bad_rows.append(tuple(duplicate_target))
        for phases in bad_rows:
            with self.subTest(phases=phases), self.assertRaisesRegex(
                carrier.ObjectInstanceCounterfactualCarrierError,
                "duplicate or non-bijective",
            ):
                _seal(source, origin, mask, phases)

    def test_phase_partition_gap_and_overlap_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        overlap = list(_phases())
        overlap[0] = replace(overlap[0], target_complement=(0, 1))
        gap = list(_phases())
        gap[0] = replace(gap[0], phase_tokens=(0, 1, 2))
        for name, phases, message in (
            ("overlap", tuple(overlap), "partition overlaps"),
            ("gap", tuple(gap), "partition has a gap"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                carrier.ObjectInstanceCounterfactualCarrierError, message
            ):
                _seal(source, origin, mask, phases)

    def test_global_phase_gap_and_overlap_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        overlap = list(_phases())
        overlap[1] = replace(
            overlap[1],
            phase_tokens=(1, 2, 3),
            target_complement=(1, 2),
        )
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "global phase partition overlaps",
        ):
            _seal(source, origin, mask, tuple(overlap))

        # Every phase remains locally complete, but a seventh packed token is
        # absent from the global phase union.
        source_gap = torch.cat(
            (source, torch.zeros((1, 1, carrier.PACKED_CHANNELS))), dim=1
        ).contiguous()
        origin_gap = torch.cat(
            (origin, torch.zeros((1, 1, carrier.PACKED_CHANNELS))), dim=1
        ).contiguous()
        mask_gap = torch.cat((mask, torch.tensor((False,))), dim=0).contiguous()
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "global phase partition has a gap",
        ):
            _seal(source_gap, origin_gap, mask_gap)

    def test_pre_lift_nonidentity_and_lift_without_origin_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        nonidentity = list(_phases())
        nonidentity[0] = replace(
            nonidentity[0],
            correspondence=((0, 1),),
            target_complement=(0,),
        )
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "pre_lift transport is not identity",
        ):
            _seal(source, origin, mask, tuple(nonidentity))

        no_origin = list(_phases())
        no_origin[1] = replace(
            no_origin[1],
            correspondence=((2, 2),),
            target_complement=(3,),
        )
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "source and target supports overlap",
        ):
            _seal(source, origin, mask, tuple(no_origin))

    def test_all_three_regimes_are_mandatory(self) -> None:
        source, origin, mask = _inputs()
        missing_hold = list(_phases())
        missing_hold[2] = replace(missing_hold[2], regime="lift")
        missing_lift = list(_phases())
        missing_lift[1] = replace(missing_lift[1], regime="hold")
        for name, phases in (
            ("missing_hold", tuple(missing_hold)),
            ("missing_lift", tuple(missing_lift)),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(
                carrier.ObjectInstanceCounterfactualCarrierError,
                "must contain pre_lift, lift, and hold",
            ):
                _seal(source, origin, mask, phases)

    def test_nonfinite_inputs_and_arithmetic_overflow_are_rejected(self) -> None:
        for which in ("source", "origin"):
            with self.subTest(which=which):
                source, origin, mask = _inputs()
                value = source if which == "source" else origin
                value[:, 0, 0] = float("nan")
                with self.assertRaisesRegex(
                    carrier.ObjectInstanceCounterfactualCarrierError,
                    "finite floating",
                ):
                    _seal(source, origin, mask)

        source, origin, mask = _inputs(dtype=torch.float16)
        source[:, 0, :] = torch.finfo(torch.float16).max
        origin[:, 0, :] = -torch.finfo(torch.float16).max
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "residual is non-finite",
        ):
            _seal(source, origin, mask)

    def test_caller_storage_is_snapshotted_without_false_backing_claim(self) -> None:
        _, origin, mask = _inputs()
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "equals the bone-removed auxiliary",
        ):
            _seal(origin, origin, mask)

        base = torch.arange(
            12 * carrier.PACKED_CHANNELS, dtype=torch.float32
        ).reshape(1, 12, carrier.PACKED_CHANNELS)
        source_view = base[:, :6, :]
        origin_view = base[:, 6:, :]
        self.assertTrue(source_view.is_contiguous())
        self.assertTrue(origin_view.is_contiguous())
        authority = _seal(source_view, origin_view, mask)
        result = _build(source_view, origin_view, mask, authority)
        self.assertTrue(result.trace.caller_inputs_copied_to_private_snapshots)
        self.assertFalse(result.trace.caller_backing_independence_authenticated)
        self.assertTrue(result.trace.working_snapshot_storages_pairwise_distinct)

    def test_distinct_frombuffer_alias_isolated_by_private_snapshots(self) -> None:
        values = 6 * carrier.PACKED_CHANNELS
        backing = bytearray((values + 1) * torch.tensor(0.0).element_size())
        whole = torch.frombuffer(backing, dtype=torch.float32, count=values + 1)
        whole.copy_(torch.arange(values + 1, dtype=torch.float32))
        origin = torch.frombuffer(
            backing, dtype=torch.float32, count=values, offset=0
        ).reshape(1, 6, carrier.PACKED_CHANNELS)
        source = torch.frombuffer(
            backing,
            dtype=torch.float32,
            count=values,
            offset=torch.tensor(0.0).element_size(),
        ).reshape(1, 6, carrier.PACKED_CHANNELS)
        _, _, mask = _inputs()
        self.assertNotEqual(
            int(source.untyped_storage().data_ptr()),
            int(origin.untyped_storage().data_ptr()),
        )
        source_start = int(source.data_ptr())
        origin_start = int(origin.data_ptr())
        source_end = source_start + source.numel() * source.element_size()
        origin_end = origin_start + origin.numel() * origin.element_size()
        self.assertTrue(source_start < origin_end and origin_start < source_end)
        aliased_origin_before = float(origin.reshape(-1)[1].item())
        source.reshape(-1)[0] += 0.5
        self.assertNotEqual(
            float(origin.reshape(-1)[1].item()), aliased_origin_before
        )
        authority = _seal(source, origin, mask)
        result = _build(source, origin, mask, authority)
        self.assertTrue(result.trace.caller_inputs_copied_to_private_snapshots)
        self.assertFalse(result.trace.caller_backing_independence_authenticated)
        self.assertTrue(result.trace.working_snapshot_storages_pairwise_distinct)

    def test_distinct_mmaps_of_same_file_are_isolated_by_private_snapshots(self) -> None:
        values = 6 * carrier.PACKED_CHANNELS
        byte_count = values * torch.tensor(0.0).element_size()
        with tempfile.NamedTemporaryFile() as backing_file:
            backing_file.truncate(byte_count + torch.tensor(0.0).element_size())
            backing_file.flush()
            first_fd = os.open(backing_file.name, os.O_RDWR)
            second_fd = os.open(backing_file.name, os.O_RDWR)
            first_map = mmap.mmap(first_fd, byte_count + 4)
            second_map = mmap.mmap(second_fd, byte_count + 4)
            try:
                origin = torch.frombuffer(
                    first_map, dtype=torch.float32, count=values, offset=0
                ).reshape(1, 6, carrier.PACKED_CHANNELS)
                source = torch.frombuffer(
                    second_map, dtype=torch.float32, count=values, offset=4
                ).reshape(1, 6, carrier.PACKED_CHANNELS)
                _, _, mask = _inputs()
                self.assertNotEqual(
                    int(source.untyped_storage().data_ptr()),
                    int(origin.untyped_storage().data_ptr()),
                )
                source_start = int(source.data_ptr())
                origin_start = int(origin.data_ptr())
                source_end = source_start + source.numel() * source.element_size()
                origin_end = origin_start + origin.numel() * origin.element_size()
                self.assertFalse(
                    source_start < origin_end and origin_start < source_end
                )
                origin_before = float(origin.reshape(-1)[1].item())
                source.reshape(-1)[0] += 1.0
                self.assertNotEqual(
                    float(origin.reshape(-1)[1].item()), origin_before
                )
                authority = _seal(source, origin, mask)
                result = _build(source, origin, mask, authority)
                self.assertTrue(
                    result.trace.caller_inputs_copied_to_private_snapshots
                )
                self.assertFalse(
                    result.trace.caller_backing_independence_authenticated
                )
                self.assertTrue(
                    result.trace.working_snapshot_storages_pairwise_distinct
                )
                del source, origin
            finally:
                first_map.close()
                second_map.close()
                os.close(first_fd)
                os.close(second_fd)

    def test_forged_authority_and_phase_plan_replay_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        authority = _seal(source, origin, mask)
        forged = replace(authority, source_role="aux_bone_removed_source")
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "authority fields or digest differ",
        ):
            _build(source, origin, mask, forged)

        changed = list(_phases())
        changed[2] = replace(changed[2], phase_id="phase2_changed")
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "differ from sealed authority",
        ):
            _build(source, origin, mask, authority, tuple(changed))

    def test_mutable_or_noncanonical_plan_containers_are_rejected(self) -> None:
        source, origin, mask = _inputs()
        with self.assertRaisesRegex(
            carrier.ObjectInstanceCounterfactualCarrierError,
            "phases must be one nonempty exact tuple",
        ):
            carrier.seal_patient_carrier_authority(
                source_packed=source,
                bone_removed_packed=origin,
                source_mask=mask,
                phases=list(_phases()),  # type: ignore[arg-type]
            )
        bad = list(_phases())
        bad[0] = replace(
            bad[0], correspondence=((0, 0), (0, 1))
        )
        with self.assertRaises(carrier.ObjectInstanceCounterfactualCarrierError):
            _seal(source, origin, mask, tuple(bad))

    def test_tensor_abi_rejections(self) -> None:
        source, origin, mask = _inputs()
        cases = (
            (source.requires_grad_(True), origin, mask),
            (source.detach().transpose(0, 1), origin, mask),
            (source.detach().to(torch.int32), origin.to(torch.int32), mask),
            (source.detach(), origin, mask.to(torch.uint8)),
        )
        for source_bad, origin_bad, mask_bad in cases:
            with self.subTest(
                source_shape=tuple(source_bad.shape),
                source_dtype=str(source_bad.dtype),
                mask_dtype=str(mask_bad.dtype),
            ), self.assertRaises(carrier.ObjectInstanceCounterfactualCarrierError):
                _seal(source_bad, origin_bad, mask_bad)

    def test_result_allocations_do_not_alias_any_input_or_each_other(self) -> None:
        source, origin, mask = _inputs()
        result = _build(source, origin, mask, _seal(source, origin, mask))
        tensors = (
            source,
            origin,
            mask,
            result.counterfactual,
            result.transported_residual,
        )
        pointers = [int(value.untyped_storage().data_ptr()) for value in tensors]
        self.assertEqual(len(set(pointers)), len(pointers))
        self.assertTrue(result.trace.output_storages_fresh_and_pairwise_distinct)

    def test_contract_explicitly_disclaims_renderer_and_visual_success(self) -> None:
        contract = carrier.tensor_core_contract()
        self.assertEqual(contract["equation"], "z_cf=z0+T(mask*(zs-z0))")
        self.assertEqual(contract["pre_lift_transport"], "identity")
        self.assertEqual(
            contract["pre_lift_output_identity"],
            "formula_observation_only_not_required_byte_equal_to_zs",
        )
        self.assertEqual(contract["lift_hold_source_target_overlap"], "forbidden")
        self.assertEqual(contract["complement"], "byte_exact_z0")
        self.assertTrue(contract["caller_inputs_copied_to_private_snapshots"])
        self.assertFalse(contract["caller_backing_independence_authenticated"])
        self.assertFalse(contract["working_snapshot_storage_alias_allowed"])
        self.assertFalse(contract["renderer_integration"])
        self.assertFalse(contract["model_integration"])
        self.assertFalse(contract["visual_success_claimed"])
        self.assertFalse(contract["upstream_provenance_authenticated"])


if __name__ == "__main__":
    unittest.main()

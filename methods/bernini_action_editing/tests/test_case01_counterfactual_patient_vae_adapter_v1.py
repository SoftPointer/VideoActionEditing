#!/usr/bin/env python3
"""Focused tests for the bounded Case01 patient VAE adapter."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = REPO_ROOT / "methods" / "bernini_action_editing"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import case01_counterfactual_patient_vae_adapter_v1 as adapter  # noqa: E402


SCAFFOLD_PATH = (
    REPO_ROOT / "artifacts" / "case01_oracle_object_trajectory_v1" / "scaffold.json"
)


def _scaffold():
    with SCAFFOLD_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _byte_equal(left, right):
    return bool(
        torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _videos():
    source = torch.ones((1, 1, 1, 1, 1), dtype=torch.float16).expand(
        1, 3, 81, 496, 480
    )
    clean = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float16).expand(
        1, 3, 81, 496, 480
    )
    return source, clean


def _decoded(*, frames=81, height=496, width=480):
    return torch.zeros((1, 1, 1, 1, 1), dtype=torch.float16).expand(
        1, 3, frames, height, width
    )


def _latents():
    clean = torch.zeros(
        (1, 16, 21, 62, 60), dtype=torch.float16
    ).contiguous()
    source = torch.ones_like(clean).contiguous()
    return source, clean


def _successful_result(*, source_video=None, clean_video=None, decode=None):
    if source_video is None or clean_video is None:
        source_video, clean_video = _videos()
    source_latent, clean_latent = _latents()
    encode_calls = 0

    def encode(_vae, _video):
        nonlocal encode_calls
        encode_calls += 1
        latent = source_latent if encode_calls == 1 else clean_latent
        return latent.clone().contiguous()

    return adapter.run_vae_only_carrier_feasibility(
        vae=object(),
        encode=encode,
        decode=decode or (lambda _vae, _latent: _decoded()),
        source_video=source_video,
        bone_removed_v2_video=clean_video,
        scaffold=_scaffold(),
    )


class Case01PlanCompilerTests(unittest.TestCase):
    def test_wrong_carrier_module_path_fails_before_plan_or_callbacks(self):
        with mock.patch.object(
            adapter.carrier, "__file__", "/tmp/not-the-frozen-carrier.py"
        ), self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "path is absent|exact sibling",
        ):
            adapter.compile_case01_carrier_plan(_scaffold())

    def test_contract_never_claims_visual_or_scientific_pass(self):
        contract = adapter.adapter_contract()
        self.assertEqual(contract["only_return_status"], adapter.OUTPUT_STATUS)
        self.assertEqual(contract["rgb_shape"], ["B", 3, 81, 496, 480])
        self.assertEqual(contract["phase_regimes"]["pre_lift"], [0, 9])
        self.assertEqual(contract["phase_regimes"]["lift"], [10, 15])
        self.assertEqual(contract["phase_regimes"]["hold"], [16, 20])
        self.assertEqual(contract["phase10_replacement_shift_xy"], [-3, 0])
        self.assertFalse(contract["legacy_aux_consumed"])
        self.assertFalse(contract["vae_model_identity_authenticated"])
        self.assertFalse(contract["source_video_values_authenticated"])
        self.assertFalse(contract["decoded_video_values_authenticated"])
        self.assertFalse(contract["carrier_runtime_semantics_authenticated"])
        self.assertFalse(contract["caller_backing_independence_authenticated"])
        self.assertTrue(contract["caller_videos_copied_to_private_snapshots"])
        self.assertTrue(contract["decoded_output_copied_to_private_snapshot"])
        self.assertFalse(contract["all81_review_complete"])
        self.assertFalse(contract["visual_success_claimed"])
        self.assertFalse(contract["scientific_claim_authorized"])

    def test_exact_real_scaffold_compilation_and_frozen_digests(self):
        compiled = adapter.compile_case01_carrier_plan(_scaffold())
        self.assertEqual(len(compiled.phases), 21)
        self.assertEqual(len(compiled.phase_audits), 21)
        self.assertEqual(tuple(compiled.source_mask.shape), (19_530,))
        self.assertEqual(
            tuple(compiled.target_responsibility_mask.shape), (19_530,)
        )
        self.assertEqual(int(compiled.source_mask.sum().item()), 377)
        self.assertEqual(
            int(compiled.target_responsibility_mask.sum().item()), 2_776
        )
        self.assertEqual(
            compiled.scaffold_geometry_sha256,
            "2037982a36519301f962d041f55dcad847d0ed39b9d02e4c9c4b1b45995e130c",
        )
        self.assertEqual(
            compiled.source_mask_raw_sha256,
            "eba47751bdee4827f85dd0127913846dbdc9611dd0e87fa25788b0f4fd1a89ca",
        )
        self.assertEqual(
            compiled.target_responsibility_mask_raw_sha256,
            "0dbcc0ed0cbf76ccde988c56a8f64ebe7fd344a2f02f84d8305efbdd9ff00727",
        )
        self.assertEqual(
            compiled.plan_digest,
            "30a70d48b774658d05c10c4bb65a91b41d3fffe553eb7893a111da6a3ef01ece",
        )
        receipt = compiled.audit_receipt()
        self.assertEqual(receipt["source_patient_token_count"], 377)
        self.assertEqual(receipt["target_responsibility_token_count"], 2_776)
        self.assertFalse(receipt["legacy_aux_consumed"])
        json.dumps(receipt, sort_keys=True, allow_nan=False)

    def test_live_mask_mutation_invalidates_compiler_receipt(self):
        source_mutated = adapter.compile_case01_carrier_plan(_scaffold())
        source_mutated.source_mask[0] = ~source_mutated.source_mask[0]
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            source_mutated.audit_receipt()

        responsibility_mutated = adapter.compile_case01_carrier_plan(_scaffold())
        responsibility_mutated.target_responsibility_mask[0] = (
            ~responsibility_mutated.target_responsibility_mask[0]
        )
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            responsibility_mutated.audit_receipt()

    def test_regimes_and_phase_ownership_are_exact(self):
        compiled = adapter.compile_case01_carrier_plan(_scaffold())
        regimes = tuple(row.regime for row in compiled.phase_audits)
        self.assertEqual(regimes[:10], ("pre_lift",) * 10)
        self.assertEqual(regimes[10:16], ("lift",) * 6)
        self.assertEqual(regimes[16:], ("hold",) * 5)
        owned = []
        for phase_index, phase in enumerate(compiled.phases):
            expected = tuple(range(phase_index * 930, (phase_index + 1) * 930))
            self.assertEqual(phase.phase_tokens, expected)
            self.assertEqual(
                set(pair[1] for pair in phase.correspondence)
                | set(phase.target_complement),
                set(expected),
            )
            self.assertFalse(
                set(pair[1] for pair in phase.correspondence)
                & set(phase.target_complement)
            )
            owned.extend(phase.phase_tokens)
        self.assertEqual(tuple(owned), tuple(range(19_530)))

    def test_phase10_is_disjoint_x_minus_three_and_exact_union_only(self):
        scaffold = _scaffold()
        compiled = adapter.compile_case01_carrier_plan(scaffold)
        row = compiled.phase_audits[10]
        self.assertEqual(row.declared_shift_xy, (-1, 0))
        self.assertEqual(row.compiled_shift_xy, (-3, 0))
        self.assertTrue(row.replacement_applied)
        self.assertEqual(len(row.local_source_tokens), 19)
        self.assertEqual(len(row.local_target_tokens), 19)
        old_targets = set(
            scaffold["latent_phases"][10]["target_bone_tokens"]
        )
        self.assertEqual(len(set(row.local_source_tokens) & old_targets), 11)
        self.assertFalse(
            set(row.local_source_tokens) & set(row.local_target_tokens)
        )
        for source, target in zip(
            row.local_source_tokens, row.local_target_tokens
        ):
            source_y, source_x = divmod(source, 30)
            target_y, target_x = divmod(target, 30)
            self.assertEqual((target_x, target_y), (source_x - 3, source_y))
        old_responsibility = set(
            scaffold["latent_phases"][10]["target_responsibility_tokens"]
        )
        expected = old_responsibility | set(row.local_target_tokens)
        self.assertEqual(set(row.local_responsibility_tokens), expected)
        self.assertEqual(len(row.local_responsibility_tokens), 153)
        self.assertEqual(
            len(expected - old_responsibility), 16
        )
        self.assertEqual(
            set(row.global_source_tokens),
            {10 * 930 + token for token in row.local_source_tokens},
        )
        self.assertEqual(
            set(row.global_target_tokens),
            {10 * 930 + token for token in row.local_target_tokens},
        )

    def test_other_responsibilities_are_not_expanded(self):
        scaffold = _scaffold()
        compiled = adapter.compile_case01_carrier_plan(scaffold)
        for phase_index, row in enumerate(compiled.phase_audits):
            old = tuple(
                scaffold["latent_phases"][phase_index][
                    "target_responsibility_tokens"
                ]
            )
            if phase_index == 10:
                self.assertNotEqual(row.local_responsibility_tokens, old)
            else:
                self.assertEqual(row.local_responsibility_tokens, old)
            self.assertTrue(
                set(row.local_target_tokens).issubset(
                    row.local_responsibility_tokens
                )
            )

    def test_legacy_auxiliary_and_nongeometry_fields_are_not_consumed(self):
        first_scaffold = _scaffold()
        second_scaffold = copy.deepcopy(first_scaffold)
        second_scaffold["authority"] = {
            "bone_removed_auxiliary_video": object(),
            "anything_else": object(),
        }
        second_scaffold["artifact_digest"] = object()
        second_scaffold["claim_limits"] = object()
        second_scaffold["latent_phases"][10]["dog_identity_core_tokens"] = object()
        first = adapter.compile_case01_carrier_plan(first_scaffold)
        second = adapter.compile_case01_carrier_plan(second_scaffold)
        self.assertEqual(first.plan_digest, second.plan_digest)
        self.assertEqual(
            first.scaffold_geometry_sha256, second.scaffold_geometry_sha256
        )
        self.assertTrue(_byte_equal(first.source_mask, second.source_mask))
        self.assertFalse(second.legacy_aux_consumed)

    def test_any_consumed_geometry_drift_fails_closed(self):
        mutations = []
        changed_source = _scaffold()
        changed_source["latent_phases"][10]["source_bone_tokens"][0] -= 1
        mutations.append(changed_source)
        changed_shift = _scaffold()
        changed_shift["latent_phases"][10]["bone_shift_patch_xy"] = [-3, 0]
        mutations.append(changed_shift)
        changed_responsibility = _scaffold()
        changed_responsibility["latent_phases"][10][
            "target_responsibility_tokens"
        ].pop()
        mutations.append(changed_responsibility)
        changed_layout = _scaffold()
        changed_layout["latent_layout"]["tokens_per_phase"] = 929
        mutations.append(changed_layout)
        changed_bucket = _scaffold()
        changed_bucket["geometry"]["renderer_bucket_wh"] = [496, 480]
        mutations.append(changed_bucket)
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(
                adapter.Case01CounterfactualPatientVaeAdapterError
            ):
                adapter.compile_case01_carrier_plan(changed)

    def test_mutable_subclasses_and_noncanonical_geometry_fail_closed(self):
        class DictSubclass(dict):
            pass

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "exact dict",
        ):
            adapter.compile_case01_carrier_plan(DictSubclass(_scaffold()))

        changed = _scaffold()
        changed["latent_phases"] = tuple(changed["latent_phases"])
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "exact 21 latent phases",
        ):
            adapter.compile_case01_carrier_plan(changed)

        changed = _scaffold()
        changed["latent_phases"][0]["source_bone_tokens"] = tuple(
            changed["latent_phases"][0]["source_bone_tokens"]
        )
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            adapter.compile_case01_carrier_plan(changed)


class Case01PackAbiTests(unittest.TestCase):
    def test_overlapping_independent_frombuffer_storages_are_rejected(self):
        backing = bytearray(128)
        first = torch.frombuffer(backing, dtype=torch.uint8, count=64, offset=0)
        second = torch.frombuffer(backing, dtype=torch.uint8, count=64, offset=32)
        self.assertNotEqual(
            first.untyped_storage().data_ptr(),
            second.untyped_storage().data_ptr(),
        )
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "intervals overlap",
        ):
            adapter._require_distinct_allocations(
                (("first", first), ("second", second)),
                label="test buffers",
            )

    def test_pack_unpack_is_byte_exact_for_every_carrier_dtype(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            with self.subTest(dtype=str(dtype)):
                latent = torch.arange(
                    16 * 21 * 62 * 60, dtype=torch.float32
                ).remainder(1_024).reshape(1, 16, 21, 62, 60).to(
                    dtype=dtype
                ).contiguous()
                packed = adapter.pack_vae_latent(latent)
                replay = adapter.unpack_vae_latent(packed)
                self.assertEqual(tuple(packed.shape), (1, 19_530, 64))
                self.assertEqual(packed.dtype, dtype)
                self.assertTrue(_byte_equal(replay, latent))
                self.assertTrue(
                    _byte_equal(adapter.pack_vae_latent(replay), packed)
                )

    def test_pack_channel_order_matches_wan_2x2_patch_layout(self):
        latent = torch.zeros((1, 16, 21, 62, 60), dtype=torch.float32)
        phase = 3
        patch_y = 4
        patch_x = 5
        for inner_y in range(2):
            for inner_x in range(2):
                for channel in range(16):
                    latent[
                        0,
                        channel,
                        phase,
                        patch_y * 2 + inner_y,
                        patch_x * 2 + inner_x,
                    ] = 100 * inner_y + 20 * inner_x + channel
        packed = adapter.pack_vae_latent(latent.contiguous())
        token = phase * 930 + patch_y * 30 + patch_x
        for inner_y in range(2):
            for inner_x in range(2):
                for channel in range(16):
                    packed_channel = (inner_y * 2 + inner_x) * 16 + channel
                    self.assertEqual(
                        float(packed[0, token, packed_channel].item()),
                        float(100 * inner_y + 20 * inner_x + channel),
                    )

    def test_batch_dimension_is_preserved(self):
        latent = torch.arange(
            2 * 16 * 21 * 62 * 60, dtype=torch.float32
        ).reshape(2, 16, 21, 62, 60).contiguous()
        packed = adapter.pack_vae_latent(latent)
        self.assertEqual(tuple(packed.shape), (2, 19_530, 64))
        self.assertTrue(_byte_equal(adapter.unpack_vae_latent(packed), latent))

    def test_latent_and_packed_abi_rejections(self):
        good = torch.zeros((1, 16, 21, 62, 60), dtype=torch.float32)
        bad_latents = (
            good[:, :, :, :, :-1].contiguous(),
            good.transpose(3, 4),
            good.to(torch.int32),
            good.clone().requires_grad_(True),
        )
        nonfinite = good.clone()
        nonfinite[0, 0, 0, 0, 0] = float("nan")
        bad_latents = (*bad_latents, nonfinite)
        for value in bad_latents:
            with self.subTest(shape=tuple(value.shape), dtype=str(value.dtype)):
                with self.assertRaises(
                    adapter.Case01CounterfactualPatientVaeAdapterError
                ):
                    adapter.pack_vae_latent(value)

        packed = adapter.pack_vae_latent(good)
        bad_packed = (
            packed[:, :-1, :].contiguous(),
            packed.transpose(1, 2),
            packed.to(torch.int64),
            packed.clone().requires_grad_(True),
        )
        for value in bad_packed:
            with self.subTest(shape=tuple(value.shape), dtype=str(value.dtype)):
                with self.assertRaises(
                    adapter.Case01CounterfactualPatientVaeAdapterError
                ):
                    adapter.unpack_vae_latent(value)


class Case01SameVaeIntegrationTests(unittest.TestCase):
    def test_same_vae_routes_through_frozen_carrier_and_only_pending_status(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        vae = object()
        calls = []
        encode_inputs = []

        def encode(received_vae, video):
            calls.append(("encode", received_vae))
            encode_inputs.append(
                (
                    video is source_video,
                    video is clean_video,
                    video.is_contiguous(),
                )
            )
            latent = source_latent if len(encode_inputs) == 1 else clean_latent
            return latent.clone().contiguous()

        def decode(received_vae, latent):
            calls.append(("decode", received_vae))
            return _decoded()

        with mock.patch.object(
            adapter.carrier,
            "seal_patient_carrier_authority",
            wraps=adapter.carrier.seal_patient_carrier_authority,
        ) as seal, mock.patch.object(
            adapter.carrier,
            "build_counterfactual_patient_carrier",
            wraps=adapter.carrier.build_counterfactual_patient_carrier,
        ) as build:
            result = adapter.run_vae_only_carrier_feasibility(
                vae=vae,
                encode=encode,
                decode=decode,
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )
        self.assertEqual(seal.call_count, 1)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(
            [row[0] for row in calls], ["encode", "encode", "decode"]
        )
        self.assertTrue(all(row[1] is vae for row in calls))
        self.assertEqual(
            encode_inputs,
            [(False, False, True), (False, False, True)],
        )
        self.assertEqual(result.status, adapter.OUTPUT_STATUS)
        self.assertEqual(tuple(result.counterfactual_latent.shape), (1, 16, 21, 62, 60))
        self.assertEqual(
            tuple(result.decoded_video.shape), (1, 3, 81, 496, 480)
        )

        packed = result.carrier_result.counterfactual
        phase0 = result.compiled_plan.phase_audits[0]
        phase10 = result.compiled_plan.phase_audits[10]
        self.assertTrue(
            torch.equal(
                packed[:, phase0.global_target_tokens, :],
                torch.ones(
                    (1, len(phase0.global_target_tokens), 64),
                    dtype=packed.dtype,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                packed[:, phase10.global_source_tokens, :],
                torch.zeros(
                    (1, len(phase10.global_source_tokens), 64),
                    dtype=packed.dtype,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                packed[:, phase10.global_target_tokens, :],
                torch.ones(
                    (1, len(phase10.global_target_tokens), 64),
                    dtype=packed.dtype,
                ),
            )
        )
        receipt = result.audit_receipt()
        self.assertEqual(
            set(row["status"] for row in (receipt,)),
            {"VAE_ONLY_CARRIER_FEASIBILITY_PENDING_ALL81_REVIEW"},
        )
        self.assertTrue(receipt["source_pack_roundtrip_byte_exact"])
        self.assertTrue(receipt["origin_pack_roundtrip_byte_exact"])
        self.assertTrue(receipt["counterfactual_pack_roundtrip_byte_exact"])
        self.assertTrue(receipt["same_vae_object_argument_routed"])
        self.assertFalse(receipt["vae_model_identity_authenticated"])
        self.assertFalse(receipt["source_video_values_authenticated"])
        self.assertFalse(receipt["decoded_video_values_authenticated"])
        self.assertFalse(receipt["carrier_runtime_semantics_authenticated"])
        self.assertFalse(receipt["caller_backing_independence_authenticated"])
        self.assertTrue(receipt["caller_videos_copied_to_private_snapshots"])
        self.assertTrue(receipt["decoded_output_copied_to_private_snapshot"])
        self.assertEqual(receipt["source_video_dtype"], "torch.float16")
        self.assertEqual(receipt["source_video_device_type"], "cpu")
        self.assertEqual(receipt["decoded_dtype"], "torch.float16")
        self.assertEqual(receipt["decoded_device_type"], "cpu")
        self.assertEqual(
            receipt["carrier_program_sha256"],
            adapter.EXPECTED_CARRIER_PROGRAM_SHA256,
        )
        self.assertEqual(
            receipt["carrier_program_size"],
            adapter.EXPECTED_CARRIER_PROGRAM_SIZE,
        )
        self.assertEqual(
            receipt["source_packed_raw_sha256"],
            result.carrier_result.trace.source_raw_sha256,
        )
        self.assertEqual(
            receipt["bone_removed_v2_packed_raw_sha256"],
            result.carrier_result.trace.bone_removed_origin_raw_sha256,
        )
        self.assertEqual(
            receipt["patient_residual_packed_raw_sha256"],
            result.carrier_result.trace.patient_residual_raw_sha256,
        )
        self.assertFalse(receipt["legacy_aux_consumed"])
        self.assertFalse(receipt["all81_review_complete"])
        self.assertFalse(receipt["visual_success_claimed"])
        self.assertFalse(receipt["scientific_claim_authorized"])
        self.assertRegex(receipt["receipt_digest"], r"^[0-9a-f]{64}$")
        json.dumps(receipt, sort_keys=True, allow_nan=False)
        for forged in (
            replace(result.receipt, status="PASS"),
            replace(result.receipt, legacy_aux_consumed=True),
            replace(result.receipt, visual_success_claimed=True),
            replace(result.receipt, decoded_video_values_authenticated=True),
        ):
            with self.subTest(forged=forged), self.assertRaises(
                adapter.Case01CounterfactualPatientVaeAdapterError
            ):
                forged.as_dict()

    def test_caller_video_visible_alias_and_equal_bytes_fail_before_encode(self):
        source_video, _clean_video = _videos()
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "caller videos allocation alias",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=lambda _vae, _video: self.fail("encode must not run"),
                decode=lambda _vae, _latent: self.fail("decode must not run"),
                source_video=source_video,
                bone_removed_v2_video=source_video,
                scaffold=_scaffold(),
            )

        equal_clean = torch.ones(
            (1, 1, 1, 1, 1), dtype=torch.float16
        ).expand(1, 3, 81, 496, 480)
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "video bytes are identical",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=lambda _vae, _video: self.fail("encode must not run"),
                decode=lambda _vae, _latent: self.fail("decode must not run"),
                source_video=source_video,
                bone_removed_v2_video=equal_clean,
                scaffold=_scaffold(),
            )

    def test_independent_version_counter_backing_mutation_is_rejected(self):
        one_bytes = bytearray(
            torch.tensor([1.0], dtype=torch.float16)
            .view(torch.uint8)
            .tolist()
        )
        source_base = torch.frombuffer(one_bytes, dtype=torch.float16, count=1)
        hidden_alias = torch.frombuffer(
            one_bytes, dtype=torch.float16, count=1
        )
        source_video = source_base.expand(1, 3, 81, 496, 480)
        _unused_source, clean_video = _videos()
        source_latent, _clean_latent = _latents()
        source_version_before = source_video._version

        def encode(_vae, _video):
            hidden_alias.fill_(7)
            return source_latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "mutated caller source video bytes",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )
        self.assertEqual(source_video._version, source_version_before)

    def test_decode_closure_caller_mutation_is_rejected(self):
        source_video, clean_video = _videos()

        def decode(_vae, _latent):
            source_video[0, 0, 0, 0, 0] = 9
            return _decoded()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "decode callback mutated caller source video bytes",
        ):
            _successful_result(
                source_video=source_video,
                clean_video=clean_video,
                decode=decode,
            )

    def test_decode_expanded_counterfactual_alias_is_rejected(self):
        def decode(_vae, latent):
            return latent.reshape(-1)[:1].expand(1, 3, 81, 496, 480)

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "allocation alias|intervals overlap",
        ):
            _successful_result(decode=decode)

    def test_returned_live_tampering_and_equal_copy_are_rejected(self):
        tampered = _successful_result()
        self.assertTrue(tampered.decoded_video.is_contiguous())
        for label, value in (
            ("decoded", tampered.decoded_video),
            ("counterfactual", tampered.counterfactual_latent),
            ("transported", tampered.transported_residual_packed),
        ):
            with self.subTest(label=label):
                flat = value.reshape(-1)
                original = flat[0].item()
                version_before = value._version
                flat[0] = original + 1
                flat[0] = original
                self.assertGreater(value._version, version_before)
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            tampered.audit_receipt()

        result = _successful_result()

        copied = replace(
            result,
            transported_residual_packed=(
                result.transported_residual_packed.clone().contiguous()
            ),
        )
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "transported residual .*pin differs|exact carrier output",
        ):
            copied.audit_receipt()

        reinterpreted_decoded = replace(
            result,
            decoded_video=result.decoded_video.view(torch.bfloat16),
        )
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            reinterpreted_decoded.audit_receipt()

        transported_bfloat16 = result.transported_residual_packed.view(
            torch.bfloat16
        )
        reinterpreted_transport = replace(
            result,
            transported_residual_packed=transported_bfloat16,
            carrier_result=replace(
                result.carrier_result,
                transported_residual=transported_bfloat16,
            ),
        )
        with self.assertRaises(
            adapter.Case01CounterfactualPatientVaeAdapterError
        ):
            reinterpreted_transport.audit_receipt()

        for trace_field in (
            "source_raw_sha256",
            "bone_removed_origin_raw_sha256",
            "patient_residual_raw_sha256",
        ):
            forged_without_digest = replace(
                result.carrier_result.trace,
                **{trace_field: "f" * 64, "trace_digest": ""},
            )
            forged_trace = replace(
                forged_without_digest,
                trace_digest=adapter._object_sha256(
                    forged_without_digest.payload()
                ),
            )
            forged_result = replace(
                result,
                carrier_result=replace(
                    result.carrier_result,
                    trace=forged_trace,
                ),
            )
            with self.subTest(trace_field=trace_field), self.assertRaises(
                adapter.Case01CounterfactualPatientVaeAdapterError
            ):
                forged_result.audit_receipt()

    def test_encode_input_mutation_is_rejected_before_carrier(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        calls = 0

        def encode(_vae, video):
            nonlocal calls
            calls += 1
            if calls == 1:
                video[0, 0, 0, 0, 0] += 1
                return source_latent.clone().contiguous()
            return clean_latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "encode callback mutated",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_same_byte_private_video_abi_rebind_is_rejected(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        calls = 0

        def encode(_vae, video):
            nonlocal calls
            calls += 1
            if calls == 1:
                video.data = video.view(torch.bfloat16)
                return source_latent.clone().contiguous()
            return clean_latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "mutated private source video snapshot bytes/ABI/backing",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_private_video_transient_mutate_restore_is_rejected(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        calls = 0

        def encode(_vae, video):
            nonlocal calls
            calls += 1
            if calls == 1:
                original = video[0, 0, 0, 0, 0].item()
                video[0, 0, 0, 0, 0] = 7
                video[0, 0, 0, 0, 0] = original
                return source_latent.clone().contiguous()
            return clean_latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "mutated private source video snapshot bytes/ABI/backing",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_same_byte_source_latent_backing_rebind_is_rejected(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        live_source = source_latent.clone().contiguous()
        calls = 0

        def encode(_vae, _video):
            nonlocal calls
            calls += 1
            if calls == 1:
                return live_source
            live_source.data = live_source.clone().contiguous()
            return clean_latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "mutated the source latent bytes/ABI/backing",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_encode_latent_aliases_are_rejected(self):
        source_video, clean_video = _videos()
        latent_elements = 16 * 21 * 62 * 60

        def alias_private_video(_vae, video):
            return video.reshape(-1)[:latent_elements].reshape(
                1, 16, 21, 62, 60
            )

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "allocation alias|intervals overlap",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=alias_private_video,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

        shared_latent, _clean_latent = _latents()
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "allocation alias",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=lambda _vae, _video: shared_latent,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_decode_latent_mutation_is_rejected(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()
        encode_calls = 0

        def encode(_vae, video):
            nonlocal encode_calls
            encode_calls += 1
            latent = source_latent if encode_calls == 1 else clean_latent
            return latent.clone().contiguous()

        def decode(_vae, latent):
            latent[0, 0, 0, 0, 0] += 1
            return _decoded()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "decode mutated",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=decode,
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_wrong_latent_and_decode_geometries_fail_closed(self):
        source_video, clean_video = _videos()
        source_latent, clean_latent = _latents()

        wrong_source = torch.ones(
            (1, 1, 1, 1, 1), dtype=torch.float32
        ).expand(1, 3, 81, 495, 480)
        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "source_video",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=lambda _vae, _video: self.fail("encode must not run"),
                decode=lambda _vae, _latent: self.fail("decode must not run"),
                source_video=wrong_source,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

        wrong_encode_calls = 0

        def wrong_encode(_vae, video):
            nonlocal wrong_encode_calls
            wrong_encode_calls += 1
            latent = source_latent if wrong_encode_calls == 1 else clean_latent
            return latent[:, :, :, :, :-1].contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "source VAE latent",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=wrong_encode,
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

        encode_calls = 0

        def encode(_vae, video):
            nonlocal encode_calls
            encode_calls += 1
            latent = source_latent if encode_calls == 1 else clean_latent
            return latent.clone().contiguous()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "decoded carrier video",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=encode,
                decode=lambda _vae, _latent: _decoded(frames=80),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )

    def test_equal_source_and_origin_latents_are_rejected_by_frozen_carrier(self):
        source_video, clean_video = _videos()
        _, clean_latent = _latents()

        with self.assertRaisesRegex(
            adapter.Case01CounterfactualPatientVaeAdapterError,
            "frozen patient carrier rejected",
        ):
            adapter.run_vae_only_carrier_feasibility(
                vae=object(),
                encode=lambda _vae, _video: clean_latent.clone().contiguous(),
                decode=lambda _vae, _latent: _decoded(),
                source_video=source_video,
                bone_removed_v2_video=clean_video,
                scaffold=_scaffold(),
            )


if __name__ == "__main__":
    unittest.main()

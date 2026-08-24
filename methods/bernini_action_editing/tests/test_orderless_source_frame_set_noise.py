from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

operator = importlib.import_module("orderless_source_frame_set_noise")


def _fixture(
    *,
    gaussian_seed: int = 2301,
    frame_seed: int = 4103,
    frame_count: int = 5,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    height, width = 6, 8
    gaussian_generator = torch.Generator(device="cpu").manual_seed(gaussian_seed)
    frame_generator = torch.Generator(device="cpu").manual_seed(frame_seed)
    gaussian = torch.randn(
        (1, operator.LATENT_CHANNELS, operator.LATENT_PHASES, height, width),
        generator=gaussian_generator,
        dtype=dtype,
    ).contiguous()

    # A shared, spatially located identity signature plus frame-specific static
    # pose perturbations.  The operator should retain the signature's phase but
    # cannot infer the caller's temporal order.
    shared = torch.zeros(
        (1, operator.LATENT_CHANNELS, 1, height, width), dtype=dtype
    )
    shared[:, :8, :, 1:4, 2:5] = 1.4
    shared[:, 8:, :, 3:5, 5:7] = -0.9
    channel_bias = torch.linspace(
        -0.35, 0.45, operator.LATENT_CHANNELS, dtype=dtype
    ).reshape(1, operator.LATENT_CHANNELS, 1, 1, 1)
    frames = []
    for index in range(frame_count):
        noise = 0.08 * torch.randn(
            shared.shape, generator=frame_generator, dtype=dtype
        )
        pose = torch.zeros_like(shared)
        pose[:, index % operator.LATENT_CHANNELS, :, 4:6, index % width] = (
            0.2 * float(index + 1)
        )
        frames.append((shared + channel_bias + noise + pose).contiguous().clone())
    return gaussian, tuple(frames)


def _build(
    gaussian: torch.Tensor,
    frames: tuple[torch.Tensor, ...],
    *,
    rho: float = 0.05,
):
    return operator.build_orderless_source_frame_set_noise(
        canonical_gaussian=gaussian,
        independent_frame_latents=frames,
        rho=rho,
    )


def _temporal_residual(value: torch.Tensor) -> torch.Tensor:
    work = value.double()
    return work - work.mean(dim=2, keepdim=True)


def _centered_temporal_dc(value: torch.Tensor) -> torch.Tensor:
    work = value.double()
    dc = work.mean(dim=2, keepdim=True)
    scalar = dc.mean(dim=(1, 2, 3, 4), keepdim=True)
    return (dc - scalar).expand_as(work)


class OrderlessSourceFrameSetNoiseTests(unittest.TestCase):
    def test_public_api_is_closed_and_factorial_only(self) -> None:
        signature = inspect.signature(
            operator.build_orderless_source_frame_set_noise
        )
        self.assertEqual(
            list(signature.parameters),
            ["canonical_gaussian", "independent_frame_latents", "rho"],
        )
        self.assertTrue(
            all(
                value.kind is inspect.Parameter.KEYWORD_ONLY
                for value in signature.parameters.values()
            )
        )
        gaussian, frames = _fixture()
        for rho in operator.FACTORIAL_RHOS:
            with self.subTest(rho=rho):
                self.assertEqual(_build(gaussian, frames, rho=rho).diagnostics.rho, rho)
        for rho in (-0.1, 0.01, 0.051, 0.5, 1.0, float("nan"), True):
            with self.subTest(rejected_rho=rho):
                with self.assertRaisesRegex(
                    operator.OrderlessSourceFrameSetNoiseError, "factorial|real scalar"
                ):
                    _build(gaussian, frames, rho=rho)

        base = {
            "canonical_gaussian": gaussian,
            "independent_frame_latents": frames,
            "rho": 0.0,
        }
        for forbidden in (
            "source_frame_indices",
            "full_source_video_latent",
            "ordered_source_trajectory",
            "target",
            "paired_target",
            "action_proposal",
            "mask",
            "flow",
            "pose",
            "trainer_state",
            "inference_stage",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    operator.build_orderless_source_frame_set_noise(
                        **base, **{forbidden: None}
                    )

    def test_rho_zero_is_exact_native_object_and_receipt_denies_authority(self) -> None:
        gaussian, frames = _fixture()
        before = gaussian.view(torch.uint8).clone()
        result = _build(gaussian, frames, rho=0.0)
        self.assertIs(result.initial_noise, gaussian)
        self.assertEqual(result.initial_noise.data_ptr(), gaussian.data_ptr())
        self.assertTrue(torch.equal(result.initial_noise.view(torch.uint8), before))
        self.assertIsNone(result.source_set_prototype)
        self.assertIsNone(result.temporal_dc_carrier)
        self.assertTrue(result.diagnostics.rho_zero_exact_object_alias)
        self.assertFalse(result.diagnostics.source_conditioned_non_gaussian)
        receipt = result.receipt
        self.assertTrue(receipt["ablation_only"])
        self.assertFalse(receipt["editor_optimizer_authorized"])
        self.assertFalse(receipt["editor_training_authorized"])
        self.assertFalse(receipt["critic_reward_authorized"])
        self.assertFalse(receipt["scientific_success_claim_authorized"])
        self.assertFalse(receipt["operator_self_registers_sampler_hook"])
        self.assertEqual(receipt["allowed_factorial_rhos"], [0.0, 0.05, 0.10])

    def test_frame_permutation_and_reversal_are_bit_exact_invariant(self) -> None:
        gaussian, frames = _fixture()
        permutation = (3, 0, 4, 1, 2)
        permuted = tuple(frames[index] for index in permutation)
        original = _build(gaussian, frames, rho=0.10)
        reordered = _build(gaussian, permuted, rho=0.10)
        reversed_result = _build(gaussian, tuple(reversed(frames)), rho=0.10)
        for candidate in (reordered, reversed_result):
            self.assertTrue(torch.equal(original.initial_noise, candidate.initial_noise))
            self.assertTrue(
                torch.equal(original.source_set_prototype, candidate.source_set_prototype)
            )
            self.assertTrue(
                torch.equal(original.temporal_dc_carrier, candidate.temporal_dc_carrier)
            )
            self.assertEqual(
                original.diagnostics.source_frame_multiset_sha256,
                candidate.diagnostics.source_frame_multiset_sha256,
            )
            self.assertEqual(
                original.diagnostics.selected_medoid_value_sha256,
                candidate.diagnostics.selected_medoid_value_sha256,
            )
            self.assertEqual(original.receipt_sha256, candidate.receipt_sha256)

    def test_selected_exemplar_is_the_true_pairwise_set_medoid(self) -> None:
        gaussian, frames = _fixture(frame_count=5)
        result = _build(gaussian, frames, rho=0.10)
        rows = sorted(
            (
                operator._tensor_sha256(frame),
                frame.detach().cpu().double()[0, :, 0].contiguous(),
            )
            for frame in frames
        )
        values = torch.stack([row[1] for row in rows]).flatten(1)
        distances = torch.cdist(values, values, p=2).square().sum(dim=1)
        expected = min(
            (float(distances[index].item()), rows[index][0])
            for index in range(len(rows))
        )[1]
        self.assertEqual(result.diagnostics.selected_medoid_value_sha256, expected)
        self.assertEqual(
            result.receipt["prototype"]["set_medoid_objective"],
            "minimum_sum_pairwise_mean_squared_distance_to_all_multiset_members",
        )

    def test_spatial_phase_and_low_frequency_layout_are_really_consumed(self) -> None:
        gaussian, frames = _fixture(dtype=torch.float64)
        shifted = tuple(
            torch.roll(frame, shifts=(1, 2), dims=(-2, -1)).contiguous().clone()
            for frame in frames
        )
        original = _build(gaussian, frames, rho=0.10)
        translated = _build(gaussian, shifted, rho=0.10)
        self.assertFalse(
            torch.equal(original.source_set_prototype, translated.source_set_prototype)
        )
        self.assertFalse(torch.equal(original.initial_noise, translated.initial_noise))
        self.assertNotEqual(
            original.diagnostics.source_set_prototype_sha256,
            translated.diagnostics.source_set_prototype_sha256,
        )
        self.assertTrue(original.diagnostics.source_spatial_phase_consumed)
        self.assertTrue(original.diagnostics.source_low_frequency_layout_consumed)
        self.assertTrue(
            original.receipt["prototype"]["source_spatial_phase_retained"]
        )
        self.assertTrue(
            original.receipt["prototype"]["source_low_frequency_layout_retained"]
        )

    def test_active_carrier_is_temporal_dc_and_preserves_gaussian_residual(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                gaussian, frames = _fixture(dtype=dtype)
                result = _build(gaussian, frames, rho=0.10)
                carrier = result.temporal_dc_carrier
                self.assertIsNotNone(carrier)
                self.assertTrue(torch.equal(carrier, carrier[:, :, :1].expand_as(carrier)))
                if dtype == torch.float64:
                    rtol, atol = 2.0e-10, 2.0e-10
                else:
                    rtol, atol = 6.0e-5, 8.0e-6
                torch.testing.assert_close(
                    _temporal_residual(result.initial_noise),
                    _temporal_residual(gaussian),
                    rtol=rtol,
                    atol=atol,
                )
                self.assertFalse(result.diagnostics.source_frame_order_consumed)
                self.assertFalse(result.diagnostics.source_temporal_phase_consumed)
                self.assertTrue(result.diagnostics.carrier_strict_temporal_dc)

    def test_active_transport_preserves_realized_mean_and_norm_geometry(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                gaussian, frames = _fixture(dtype=dtype)
                result = _build(gaussian, frames, rho=0.05)
                output = result.initial_noise.double()
                before = gaussian.double()
                carrier = result.temporal_dc_carrier.double()
                gaussian_dc = _centered_temporal_dc(gaussian)
                if dtype == torch.float64:
                    rtol, atol = 2.0e-10, 2.0e-10
                else:
                    rtol, atol = 6.0e-5, 8.0e-6
                torch.testing.assert_close(
                    output.mean(), before.mean(), rtol=rtol, atol=atol
                )
                torch.testing.assert_close(
                    output.norm(), before.norm(), rtol=rtol, atol=atol
                )
                torch.testing.assert_close(
                    _centered_temporal_dc(result.initial_noise).norm(),
                    gaussian_dc.norm(),
                    rtol=rtol,
                    atol=atol,
                )
                torch.testing.assert_close(
                    carrier.norm(), gaussian_dc.norm(), rtol=rtol, atol=atol
                )
                self.assertLess(
                    result.diagnostics.carrier_gaussian_dc_normalized_dot_max,
                    2.0e-10,
                )
                self.assertGreater(
                    result.diagnostics.transported_to_raw_source_carrier_normalized_dot_min,
                    0.9,
                )
                self.assertTrue(result.diagnostics.numerical_audit_passed)

    def test_source_counterfactual_changes_only_active_endpoint(self) -> None:
        gaussian, frames = _fixture(frame_seed=11)
        _, wrong_frames = _fixture(frame_seed=97)
        correct = _build(gaussian, frames, rho=0.10)
        wrong = _build(gaussian, wrong_frames, rho=0.10)
        self.assertNotEqual(
            correct.diagnostics.source_frame_multiset_sha256,
            wrong.diagnostics.source_frame_multiset_sha256,
        )
        self.assertFalse(torch.equal(correct.initial_noise, wrong.initial_noise))
        correct_off = _build(gaussian, frames, rho=0.0)
        wrong_off = _build(gaussian, wrong_frames, rho=0.0)
        self.assertIs(correct_off.initial_noise, gaussian)
        self.assertIs(wrong_off.initial_noise, gaussian)

    def test_independently_stored_duplicate_frame_values_form_a_valid_multiset(self) -> None:
        gaussian, frames = _fixture()
        duplicates = (
            frames[0].clone(),
            frames[0].clone(),
            frames[1].clone(),
            frames[2].clone(),
        )
        result = _build(gaussian, duplicates, rho=0.05)
        reversed_result = _build(gaussian, tuple(reversed(duplicates)), rho=0.05)
        self.assertTrue(torch.equal(result.initial_noise, reversed_result.initial_noise))
        self.assertEqual(
            result.diagnostics.source_frame_multiset_sha256,
            reversed_result.diagnostics.source_frame_multiset_sha256,
        )
        self.assertTrue(result.receipt["source_contract"]["multiplicity_retained"])

    def test_receipt_is_canonical_hash_closed_and_honest_about_pose_risk(self) -> None:
        gaussian, frames = _fixture()
        result = _build(gaussian, frames, rho=0.05)
        payload = json.dumps(
            result.receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        self.assertEqual(hashlib.sha256(payload).hexdigest(), result.receipt_sha256)
        self.assertFalse(result.receipt["semantic_old_action_absence_claimed"])
        self.assertTrue(result.receipt["carrier"]["static_pose_can_be_retained"])
        self.assertTrue(
            result.receipt["carrier"]["unordered_pose_occupancy_can_be_retained"]
        )
        self.assertFalse(
            result.receipt["source_contract"][
                "independent_encoder_invocation_proven_by_operator"
            ]
        )
        self.assertFalse(
            result.receipt["source_contract"][
                "operator_received_member_selection_indices"
            ]
        )
        self.assertTrue(
            result.receipt["source_contract"][
                "external_member_selection_may_have_used_indices"
            ]
        )
        self.assertFalse(
            result.receipt["source_contract"][
                "operator_proves_index_free_member_selection"
            ]
        )
        self.assertIn(
            "wrong_source_counterfactual_required",
            result.receipt["required_external_factorial_controls"],
        )

    def test_invalid_shape_storage_and_full_video_inputs_fail_closed(self) -> None:
        gaussian, frames = _fixture()
        error = operator.OrderlessSourceFrameSetNoiseError
        with self.assertRaisesRegex(error, "canonical_gaussian"):
            _build(gaussian[:, :, :-1].contiguous(), frames)
        with self.assertRaisesRegex(error, "tuple/list"):
            operator.build_orderless_source_frame_set_noise(
                canonical_gaussian=gaussian,
                independent_frame_latents=torch.stack(frames),
                rho=0.0,
            )
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, (gaussian.clone(),) + frames[1:])
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, (frames[0].double(),) + frames[1:])
        shared = torch.randn((2, 16, 1, 6, 8))
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, (shared[0:1], shared[1:2]))
        with self.assertRaisesRegex(error, "2..81"):
            _build(gaussian, (frames[0],))

    def test_native_480x832_exact81_geometry_with_four_references(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(812104)
        gaussian = torch.randn(
            (1, 16, 21, 60, 104), generator=generator, dtype=torch.float32
        ).contiguous()
        frames = tuple(
            torch.randn(
                (1, 16, 1, 60, 104), generator=generator, dtype=torch.float32
            ).contiguous().clone()
            for _ in range(4)
        )
        result = _build(gaussian, frames, rho=0.05)
        self.assertEqual(tuple(result.initial_noise.shape), (1, 16, 21, 60, 104))
        self.assertEqual(tuple(result.source_set_prototype.shape), (1, 16, 1, 60, 104))
        self.assertEqual(tuple(result.temporal_dc_carrier.shape), (1, 16, 21, 60, 104))
        self.assertEqual(result.diagnostics.source_frame_count, 4)
        self.assertTrue(result.diagnostics.numerical_audit_passed)

    def test_operator_is_not_wired_into_existing_oasis_or_training_mainline(self) -> None:
        integration_candidates = (
            METHOD_ROOT / "infer_oasis_phase_a_noise_bank.py",
            METHOD_ROOT / "train_source_self_identity_orbit_v4.py",
            METHOD_ROOT / "train_lora.py",
        )
        for candidate in integration_candidates:
            if candidate.exists():
                self.assertNotIn(
                    "orderless_source_frame_set_noise",
                    candidate.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()

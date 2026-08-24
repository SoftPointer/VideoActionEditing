from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
import unittest

import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

masc = importlib.import_module("motion_null_appearance_noise")


def _fixture(
    *,
    gaussian_seed: int = 17,
    frame_seed: int = 29,
    frame_count: int = 5,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    height, width = 8, 10
    gaussian_generator = torch.Generator(device="cpu").manual_seed(gaussian_seed)
    frame_generator = torch.Generator(device="cpu").manual_seed(frame_seed)
    gaussian = torch.randn(
        (1, masc.LATENT_CHANNELS, masc.LATENT_PHASES, height, width),
        generator=gaussian_generator,
        dtype=dtype,
    ).contiguous()
    channel_offset = torch.linspace(
        -0.45, 0.45, masc.LATENT_CHANNELS, dtype=dtype
    ).reshape(1, masc.LATENT_CHANNELS, 1, 1, 1)
    y = torch.linspace(-1.0, 1.0, height, dtype=dtype).reshape(1, 1, 1, height, 1)
    x = torch.linspace(-1.0, 1.0, width, dtype=dtype).reshape(1, 1, 1, 1, width)
    frames = []
    for index in range(frame_count):
        value = torch.randn(
            (1, masc.LATENT_CHANNELS, 1, height, width),
            generator=frame_generator,
            dtype=dtype,
        )
        value = value + channel_offset + (0.03 * float(index + 1)) * (x - y)
        # clone() makes each T=1 input an independently allocated, base-free
        # tensor rather than a view cut out of a full-video allocation.
        frames.append(value.contiguous().clone())
    return gaussian, tuple(frames)


def _build(
    gaussian: torch.Tensor,
    frames: tuple[torch.Tensor, ...],
    *,
    rho: float = 0.25,
    seed: int = 101,
):
    return masc.build_motion_null_appearance_noise(
        canonical_gaussian=gaussian,
        independent_frame_latents=frames,
        rho=rho,
        carrier_seed=seed,
    )


def _temporal_residual(value: torch.Tensor) -> torch.Tensor:
    work = value.double()
    return work - work.mean(dim=2, keepdim=True)


def _centered_temporal_dc(value: torch.Tensor) -> torch.Tensor:
    work = value.double()
    dc = work.mean(dim=2, keepdim=True)
    scalar = dc.mean(dim=(1, 2, 3, 4), keepdim=True)
    return (dc - scalar).expand_as(work)


class MotionNullAppearanceNoiseTests(unittest.TestCase):
    def test_public_api_is_closed_and_target_like_keywords_are_rejected(self) -> None:
        signature = inspect.signature(masc.build_motion_null_appearance_noise)
        self.assertEqual(
            list(signature.parameters),
            [
                "canonical_gaussian",
                "independent_frame_latents",
                "rho",
                "carrier_seed",
            ],
        )
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in signature.parameters.values()
            )
        )
        base = {
            "canonical_gaussian": None,
            "independent_frame_latents": (),
            "rho": 0.0,
            "carrier_seed": 0,
        }
        for forbidden in (
            "target",
            "target_latent",
            "paired_target",
            "action_proposal",
            "proposal_latent",
            "full_video_latent",
            "source_video_latent",
            "motion_reference",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
                    masc.build_motion_null_appearance_noise(
                        **base, **{forbidden: None}
                    )

    def test_module_is_not_wired_into_existing_mainline(self) -> None:
        integration_candidates = (
            METHOD_ROOT / "train_source_self_identity_orbit_v4.py",
            METHOD_ROOT / "source_self_identity_orbit_v4.py",
            METHOD_ROOT / "infer_native_self_imagined_phase_noise_canary.py",
            METHOD_ROOT / "scripts" / "auh_train_source_self_identity_orbit_v4.sbatch",
        )
        for candidate in integration_candidates:
            if candidate.exists():
                self.assertNotIn(
                    "motion_null_appearance_noise",
                    candidate.read_text(encoding="utf-8"),
                )

    def test_rho_zero_returns_same_object_and_same_bytes_without_a_carrier(self) -> None:
        gaussian, frames = _fixture()
        before = gaussian.view(torch.uint8).clone()
        result = _build(gaussian, frames, rho=0.0, seed=999)
        self.assertIs(result.initial_noise, gaussian)
        self.assertEqual(result.initial_noise.data_ptr(), gaussian.data_ptr())
        self.assertTrue(torch.equal(result.initial_noise.view(torch.uint8), before))
        self.assertIsNone(result.temporal_dc_carrier)
        self.assertTrue(result.diagnostics.rho_zero_exact_object_alias)
        self.assertIsNone(result.diagnostics.descriptor_sha256)
        self.assertEqual(
            result.receipt["schema_version"],
            "bernini-motion-null-appearance-noise-v2",
        )
        self.assertFalse(result.receipt["trainer_integration_executed"])
        self.assertFalse(result.receipt["operator_self_registers_sampler_hook"])
        self.assertFalse(result.receipt["operator_self_registers_launcher"])
        self.assertNotIn("inference_integration", result.receipt)
        self.assertNotIn("launcher_registration", result.receipt)
        self.assertFalse(result.receipt["scientific_claim_authorized"])

    def test_frame_permutation_and_reversal_are_bit_exact_invariant(self) -> None:
        gaussian, frames = _fixture()
        permutation = (3, 0, 4, 1, 2)
        permuted = tuple(frames[index] for index in permutation)
        original = _build(gaussian, frames, seed=123)
        reordered = _build(gaussian, permuted, seed=123)
        reversed_result = _build(gaussian, tuple(reversed(frames)), seed=123)
        self.assertTrue(torch.equal(original.initial_noise, reordered.initial_noise))
        self.assertTrue(
            torch.equal(original.initial_noise, reversed_result.initial_noise)
        )
        self.assertTrue(
            torch.equal(original.temporal_dc_carrier, reordered.temporal_dc_carrier)
        )
        self.assertTrue(
            torch.equal(
                original.temporal_dc_carrier,
                reversed_result.temporal_dc_carrier,
            )
        )
        self.assertEqual(
            original.diagnostics.descriptor_sha256,
            reordered.diagnostics.descriptor_sha256,
        )
        self.assertEqual(
            original.diagnostics.descriptor_sha256,
            reversed_result.diagnostics.descriptor_sha256,
        )

    def test_source_common_spatial_translation_does_not_change_descriptor(self) -> None:
        gaussian, frames = _fixture(dtype=torch.float64)
        shifted = tuple(
            torch.roll(frame, shifts=(2, -3), dims=(-2, -1)).contiguous().clone()
            for frame in frames
        )
        original = _build(gaussian, frames, rho=0.2, seed=300)
        translated = _build(gaussian, shifted, rho=0.2, seed=300)
        self.assertEqual(
            original.diagnostics.descriptor_sha256,
            translated.diagnostics.descriptor_sha256,
        )
        torch.testing.assert_close(
            original.temporal_dc_carrier,
            translated.temporal_dc_carrier,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        torch.testing.assert_close(
            original.initial_noise,
            translated.initial_noise,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_active_carrier_is_strict_temporal_dc_and_residual_is_preserved(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                gaussian, frames = _fixture(dtype=dtype)
                result = _build(gaussian, frames, rho=0.3, seed=808)
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
                self.assertTrue(result.diagnostics.carrier_strict_temporal_dc)
                self.assertLessEqual(
                    result.diagnostics.gaussian_temporal_residual_max_abs_error,
                    atol,
                )

    def test_active_rotation_preserves_mean_norm_and_dc_geometry(self) -> None:
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                gaussian, frames = _fixture(dtype=dtype)
                result = _build(gaussian, frames, rho=0.4, seed=909)
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
                normalized_dot = float(
                    (carrier * gaussian_dc).sum().abs()
                    / (carrier.norm() * gaussian_dc.norm())
                )
                self.assertLess(
                    normalized_dot,
                    2.0e-10 if dtype == torch.float64 else 2.0e-7,
                )
                self.assertTrue(result.diagnostics.numerical_audit_passed)

    def test_carrier_seed_is_deterministic_and_domain_separates_variants(self) -> None:
        gaussian, frames = _fixture()
        first = _build(gaussian, frames, seed=41)
        repeated = _build(gaussian, frames, seed=41)
        changed = _build(gaussian, frames, seed=42)
        self.assertTrue(torch.equal(first.initial_noise, repeated.initial_noise))
        self.assertTrue(
            torch.equal(first.temporal_dc_carrier, repeated.temporal_dc_carrier)
        )
        self.assertEqual(
            first.diagnostics.carrier_sha256,
            repeated.diagnostics.carrier_sha256,
        )
        self.assertEqual(
            first.diagnostics.descriptor_sha256,
            changed.diagnostics.descriptor_sha256,
        )
        self.assertNotEqual(
            first.diagnostics.carrier_sha256,
            changed.diagnostics.carrier_sha256,
        )
        self.assertFalse(torch.equal(first.initial_noise, changed.initial_noise))

    def test_invalid_shapes_storage_and_scalar_contracts_fail_closed(self) -> None:
        gaussian, frames = _fixture()
        error = masc.MotionNullAppearanceNoiseError
        with self.assertRaisesRegex(error, "canonical_gaussian"):
            _build(gaussian[:, :, :-1].contiguous(), frames)
        with self.assertRaisesRegex(error, "standalone T=1"):
            masc.build_motion_null_appearance_noise(
                canonical_gaussian=gaussian,
                independent_frame_latents=torch.stack(frames),
                rho=0.0,
                carrier_seed=0,
            )
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, (gaussian.clone(),) + frames[1:])
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            wrong_temporal = torch.randn((1, 16, 2, 8, 10))
            _build(gaussian, (wrong_temporal,) + frames[1:])
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, (frames[0].double(),) + frames[1:])

        shared = torch.randn((2, 16, 1, 8, 10))
        shared_views = (shared[0:1], shared[1:2])
        with self.assertRaisesRegex(error, "standalone detached finite T=1"):
            _build(gaussian, shared_views)
        with self.assertRaisesRegex(error, "2..81"):
            _build(gaussian, (frames[0],))
        with self.assertRaisesRegex(error, "rho"):
            _build(gaussian, frames, rho=1.01)
        with self.assertRaisesRegex(error, "rho"):
            _build(gaussian, frames, rho=float("nan"))
        with self.assertRaisesRegex(error, "carrier_seed"):
            _build(gaussian, frames, seed=-1)
        with self.assertRaisesRegex(error, "carrier_seed"):
            _build(gaussian, frames, seed=True)


if __name__ == "__main__":
    unittest.main()

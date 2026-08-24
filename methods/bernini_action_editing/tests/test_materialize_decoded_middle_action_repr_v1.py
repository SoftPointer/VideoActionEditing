#!/usr/bin/env python3

from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import inspect
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import materialize_decoded_middle_action_repr_v1 as subject
import self_generated_intermediate_action_anchor_v1 as anchor_core


class _Block(nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((width,), float(index + 1) * 1.0e-3))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias.reshape(1, 1, -1).to(hidden.dtype)


class _Transformer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width, index) for index in range(30)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _batch(
    clean: torch.Tensor,
    gaussian: torch.Tensor,
    *,
    sigma: float,
) -> dict[str, torch.Tensor]:
    velocity = gaussian - clean
    # Match Bernini's float32 packing order exactly; cancellation in the
    # inverse is the behavior under test, not an algebraically rewritten fake.
    state = (1.0 - sigma) * clean + sigma * gaussian
    count = int(clean.shape[0])
    return {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "attention_mask": torch.ones(1, 3, dtype=torch.int64),
        "t5_input_lens": torch.tensor([3], dtype=torch.int64),
        "input_vae_rope": torch.arange(count * 4, dtype=torch.float32).reshape(
            count, 1, 2, 2
        ),
        "vae_latents_mask": torch.ones(1, count, dtype=torch.bool),
        "vae_seqlen": torch.tensor([[count]], dtype=torch.int64),
        "input_vae_latents": state.clone(),
        "target_velocity": velocity.clone(),
        "timesteps": torch.tensor([[sigma * 1000.0]], dtype=torch.float32),
    }


def _patches_to_spatial(
    patches: torch.Tensor, *, spatial_shape: tuple[int, ...]
) -> torch.Tensor:
    # Test patch geometry is one scalar patch per latent phase and channel.
    batch, channels, phases, height, width = spatial_shape
    if (batch, height, width) != (1, 1, 1):
        raise AssertionError("test spatial geometry differs")
    return patches.reshape(phases, channels).transpose(0, 1).reshape(
        batch, channels, phases, height, width
    )


def _toy_pack_vae_latents(
    vae_rope_func,
    vae_type_list,
    image_inputs,
    video_inputs,
    noise_sigma,
    max_vae_frames=None,
):
    del vae_rope_func, vae_type_list, image_inputs, max_vae_frames
    packed = video_inputs.pop("video_vae_latents")[0]
    noise = torch.randn_like(packed, dtype=torch.float32)
    state = (1 - noise_sigma) * packed + noise_sigma * noise
    return {
        "input_vae_latents": state,
        "input_vae_rope": torch.zeros(
            packed.shape[0], 1, 2, 2, dtype=torch.float32
        ),
        "vae_latents_mask": torch.ones(packed.shape[0], dtype=torch.bool),
        "vae_seqlen": torch.tensor([packed.shape[0]], dtype=torch.long),
        "target_velocity": noise - packed.float(),
        "target_lens": torch.tensor([packed.shape[0]], dtype=torch.long),
    }


def _toy_double_randn_pack(
    vae_rope_func,
    vae_type_list,
    image_inputs,
    video_inputs,
    noise_sigma,
    max_vae_frames=None,
):
    del vae_rope_func, vae_type_list, image_inputs, max_vae_frames
    packed = video_inputs.pop("video_vae_latents")[0]
    first = torch.randn_like(packed, dtype=torch.float32)
    torch.randn_like(packed, dtype=torch.float32)
    return {
        "input_vae_latents": (1 - noise_sigma) * packed + noise_sigma * first,
        "input_vae_rope": torch.zeros(packed.shape[0], 1, 2, 2),
        "vae_latents_mask": torch.ones(packed.shape[0], dtype=torch.bool),
        "vae_seqlen": torch.tensor([packed.shape[0]], dtype=torch.long),
        "target_velocity": first - packed.float(),
        "target_lens": torch.tensor([packed.shape[0]], dtype=torch.long),
    }


# The template deliberately resolves the vendor global name.  Subject code
# clones this function's private globals and replaces only that one binding.
pack_vae_latents = _toy_pack_vae_latents


def _toy_process_renderer_sample(
    sample,
    tokenizer=None,
    vae_rope_func=None,
    vae_latent_mean=None,
    vae_latent_std=None,
    noise_scheduler=None,
    **kwargs,
):
    del tokenizer, vae_latent_mean, vae_latent_std, noise_scheduler, kwargs
    packed = pack_vae_latents(
        vae_rope_func,
        torch.tensor([1], dtype=torch.long),
        {},
        {"video_vae_latents": [sample["clean"].clone()]},
        sample["sigma"].clone(),
        max_vae_frames=21,
    )
    packed.update(
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "t5_input_lens": torch.tensor([3], dtype=torch.long),
            "timesteps": sample["timestep"].clone(),
        }
    )
    return [packed]


def _toy_explicit_batch(
    clean: torch.Tensor,
    gaussian: torch.Tensor,
    *,
    sigma: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], subject.ExplicitGaussianPackCapture]:
    transformed, capture = (
        subject._run_cloned_renderer_process_with_explicit_gaussian(
            process_renderer_sample=_toy_process_renderer_sample,
            pack_vae_latents=_toy_pack_vae_latents,
            sample={
                "clean": clean,
                "sigma": sigma,
                "timestep": sigma * sigma.new_tensor(1000),
            },
            gaussian=gaussian,
            process_kwargs={},
        )
    )
    row = transformed[0]
    batch = dict(row)
    for key in (
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "vae_latents_mask",
        "vae_seqlen",
        "timesteps",
        "target_lens",
    ):
        batch[key] = row[key].unsqueeze(0)
    return batch, capture


def _wan_test_patches_to_spatial(
    patches: torch.Tensor, *, spatial_shape: tuple[int, ...]
) -> torch.Tensor:
    if tuple(spatial_shape) != (1, 16, 21, 2, 2):
        raise AssertionError("test Wan spatial geometry differs")
    if tuple(patches.shape) != (21, 16, 1, 2, 2):
        raise AssertionError("test Wan patch geometry differs")
    return patches.permute(1, 0, 2, 3, 4).squeeze(2).unsqueeze(0).contiguous()


def _explicit_authority_fixture(gaussian_sha: str = "5" * 64) -> dict:
    return {
        "authority_kind": "rank0_domain_seeded_explicit_prepack_fp32_gaussian",
        "domain": subject.EXPLICIT_GAUSSIAN_DOMAIN,
        "producer_rank": 0,
        "base_seed": 17,
        "derived_seed": 23,
        "dtype": "torch.float32",
        "shape": [21, 16, 1, 2, 2],
        "canonical_gaussian_sha256": gaussian_sha,
        "broadcast_transport": "torch_distributed_nccl_fp32_tensor_broadcast",
        "world_size": 4,
        "world4_raw_sha256_consensus": True,
        "action_injection_count": 1,
        "noop_injection_count": 1,
        "action_gaussian_sha256": gaussian_sha,
        "noop_gaussian_sha256": gaussian_sha,
        "raw_noise_sigma_dtype": "torch.bfloat16",
        "raw_noise_sigma_shape": [1],
        "action_raw_noise_sigma_sha256": "6" * 64,
        "noop_raw_noise_sigma_sha256": "6" * 64,
        "clean_capture_stage": "inside_cloned_pack_before_fm_interpolation",
        "packed_state_original_op_order_bit_exact": True,
        "target_velocity_bit_exact": True,
        "recovered_from_x_or_velocity": False,
        "vendor_data_file_sha256": subject.PINNED_BERNINI_DATA_SHA256,
        "pack_vae_latents_source_sha256": (
            subject.PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
        ),
        "process_renderer_sample_source_sha256": (
            subject.PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
        ),
        "vendor_module_mutated": False,
        "original_function_globals_mutated": False,
        "trainer_received_authority": False,
    }


def _deterministic_vae_authority_fixture(
    phase0_sha: str = "7" * 64,
) -> dict:
    before = {
        "deterministic_algorithms_enabled": False,
        "deterministic_algorithms_warn_only": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
    }
    return {
        "authority_kind": "rank0_local_strict_deterministic_vae_encode",
        "policy": subject.DETERMINISTIC_VAE_POLICY,
        "producer_rank": 0,
        "encode_call_count": 2,
        "scope": "action_and_first_frame_repeat_encode_calls_only",
        "before_flags": dict(before),
        "during_flags": {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "restored_flags": dict(before),
        "flags_restored_exact": True,
        "posterior_phase0_max_abs_error": 0.0,
        "posterior_phase0_bit_exact": True,
        "action_phase0_posterior_sha256": phase0_sha,
        "noop_phase0_posterior_sha256": phase0_sha,
        "posterior_modified_after_encode": False,
        "posterior_copy_or_splice_used": False,
        "trainer_received_posterior": False,
    }


class DecodedMiddleRepresentationTests(unittest.TestCase):
    def test_posterior_identity_is_tensor_semantic_not_torch_archive_bytes(self) -> None:
        posterior = torch.linspace(
            -1.0, 1.0, 1 * 32 * 21 * 2 * 3, dtype=torch.float32
        ).reshape(1, 32, 21, 2, 3)
        modern = io.BytesIO()
        legacy = io.BytesIO()
        torch.save(posterior, modern)
        torch.save(
            posterior,
            legacy,
            _use_new_zipfile_serialization=False,
        )
        modern_blob = modern.getvalue()
        legacy_blob = legacy.getvalue()
        self.assertNotEqual(
            hashlib.sha256(modern_blob).hexdigest(),
            hashlib.sha256(legacy_blob).hexdigest(),
        )
        metadata = {
            "posterior_parameters_shape": list(posterior.shape),
            "posterior_parameters_dtype": str(posterior.dtype),
            "posterior_parameters_tensor_sha256": subject.tensor_sha256(
                posterior
            ),
        }
        modern_tensor, modern_identity = (
            subject.load_validated_materializer_posterior(
                modern_blob, metadata, label="modern"
            )
        )
        legacy_tensor, legacy_identity = (
            subject.load_validated_materializer_posterior(
                legacy_blob, metadata, label="legacy"
            )
        )
        self.assertTrue(torch.equal(modern_tensor, posterior))
        self.assertTrue(torch.equal(legacy_tensor, posterior))
        self.assertEqual(modern_identity, legacy_identity)
        self.assertEqual(
            modern_identity["identity_kind"],
            "sha256_dtype_shape_raw_tensor_bytes",
        )
        self.assertNotIn("blob", modern_identity)

    def test_rank0_canonical_posterior_envelope_roundtrip_and_tamper_rejection(
        self,
    ) -> None:
        action = torch.arange(
            1 * 32 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(1, 32, 21, 2, 2)
        noop = torch.flip(action, dims=(2,)).contiguous()
        envelope = subject.build_rank0_posterior_envelope(
            action=action,
            noop=noop,
            fps=25.0,
            input_hw=(480, 640),
            bucket_hw=(16, 16),
            action_rgb_sha256="1" * 64,
            noop_rgb_sha256="2" * 64,
        )
        pair = subject.unpack_rank0_posterior_envelope(envelope)
        self.assertTrue(torch.equal(pair.action, action))
        self.assertTrue(torch.equal(pair.noop, noop))
        self.assertEqual(pair.action_identity["shape"], list(action.shape))
        self.assertEqual(pair.action_identity["tensor_sha256"], subject.tensor_sha256(action))
        self.assertEqual(
            envelope["transport"],
            "world4_nccl_object_broadcast_of_canonical_raw_tensor_bytes",
        )
        self.assertNotIn("path", " ".join(envelope.keys()).casefold())

        action_payload = dict(envelope["action_posterior"])
        corrupted = bytearray(action_payload["raw_tensor_bytes"])
        corrupted[-1] ^= 1
        action_payload["raw_tensor_bytes"] = bytes(corrupted)
        tampered = dict(envelope)
        tampered["action_posterior"] = action_payload
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject.unpack_rank0_posterior_envelope(tampered)

    def test_strict_deterministic_vae_scope_restores_flags_on_success_and_error(
        self,
    ) -> None:
        before = dict(subject._deterministic_backend_flags())
        with subject.strict_deterministic_vae_encode_scope() as success:
            self.assertEqual(
                success["during_flags"],
                {
                    "deterministic_algorithms_enabled": True,
                    "deterministic_algorithms_warn_only": False,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                },
            )
        self.assertEqual(subject._deterministic_backend_flags(), before)
        self.assertEqual(success["restored_flags"], before)
        self.assertTrue(success["flags_restored_exact"])

        with self.assertRaisesRegex(RuntimeError, "synthetic encode failure"):
            with subject.strict_deterministic_vae_encode_scope() as failed:
                raise RuntimeError("synthetic encode failure")
        self.assertEqual(subject._deterministic_backend_flags(), before)
        self.assertEqual(failed["restored_flags"], before)
        self.assertTrue(failed["flags_restored_exact"])

    def test_deterministic_vae_authority_is_exact_and_never_splices_posterior(
        self,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260824)
        action = torch.randn(1, 32, 21, 2, 2, generator=generator)
        noop = torch.randn(1, 32, 21, 2, 2, generator=generator)
        noop[:, :, 0] = action[:, :, 0]
        action_before = action.clone()
        noop_before = noop.clone()
        phase0_sha = subject.tensor_sha256(action[:, :, 0])
        fixture = _deterministic_vae_authority_fixture(phase0_sha)
        authority = subject.validate_deterministic_vae_authority(
            fixture, action=action, noop=noop
        )
        envelope = subject.build_rank0_posterior_envelope(
            action=action,
            noop=noop,
            fps=25.0,
            input_hw=(16, 16),
            bucket_hw=(16, 16),
            action_rgb_sha256="1" * 64,
            noop_rgb_sha256="2" * 64,
            deterministic_vae_authority=authority,
        )
        pair = subject.unpack_rank0_posterior_envelope(envelope)
        self.assertTrue(torch.equal(action, action_before))
        self.assertTrue(torch.equal(noop, noop_before))
        self.assertTrue(torch.equal(pair.action, action_before))
        self.assertTrue(torch.equal(pair.noop, noop_before))
        self.assertEqual(pair.deterministic_vae_authority, authority)
        self.assertFalse(authority["posterior_modified_after_encode"])
        self.assertFalse(authority["posterior_copy_or_splice_used"])

        mismatched = noop.clone()
        mismatched[:, :, 0, 0, 0] += 1.0e-3
        with self.assertRaisesRegex(
            subject.DecodedMiddleRepresentationError,
            "deterministic VAE posterior phase0 differs",
        ):
            subject.validate_deterministic_vae_authority(
                authority, action=action, noop=mismatched
            )

        scope_state = {
            "before_flags": authority["before_flags"],
            "during_flags": authority["during_flags"],
            "restored_flags": authority["restored_flags"],
            "flags_restored_exact": True,
        }
        with self.assertRaisesRegex(
            subject.DecodedMiddleRepresentationError,
            "posterior phase0 differs",
        ):
            subject.build_deterministic_vae_authority(
                scope_state, action=action, noop=mismatched
            )

    def test_world4_broadcast_is_rank0_only_and_device_explicit(self) -> None:
        envelope = {"schema_version": subject.POSTERIOR_ENVELOPE_SCHEMA}

        def receive(values, *, src, device):
            self.assertEqual(src, 0)
            self.assertEqual(device, torch.device("cuda", 1))
            self.assertEqual(values, [None])
            values[0] = envelope

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    torch.distributed, "is_available", return_value=True
                )
            )
            stack.enter_context(
                mock.patch.object(
                    torch.distributed, "is_initialized", return_value=True
                )
            )
            stack.enter_context(
                mock.patch.object(
                    torch.distributed, "get_world_size", return_value=4
                )
            )
            stack.enter_context(
                mock.patch.object(
                    torch.distributed, "get_rank", return_value=1
                )
            )
            broadcast = stack.enter_context(
                mock.patch.object(
                torch.distributed,
                "broadcast_object_list",
                side_effect=receive,
                )
            )
            received = subject.broadcast_rank0_posterior_envelope(
                None,
                rank=1,
                device=torch.device("cuda", 1),
            )
            with self.assertRaises(subject.DecodedMiddleRepresentationError):
                subject.broadcast_rank0_posterior_envelope(
                    envelope,
                    rank=1,
                    device=torch.device("cuda", 1),
                )
        self.assertIs(received, envelope)
        broadcast.assert_called_once()

    def test_explicit_gaussian_tensor_broadcast_is_rank0_owned_and_exact(self) -> None:
        shape = (21, 16, 1, 2, 2)
        authority = torch.arange(
            21 * 16 * 4, dtype=torch.float32
        ).reshape(shape)
        original_empty = torch.empty

        def cpu_empty(dimensions, *, dtype, device):
            self.assertEqual(device, torch.device("cuda", 1))
            return original_empty(dimensions, dtype=dtype, device="cpu")

        def receive(value, *, src):
            self.assertEqual(src, 0)
            value.copy_(authority)

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(torch.distributed, "is_available", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(torch.distributed, "is_initialized", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(torch.distributed, "get_world_size", return_value=4)
            )
            stack.enter_context(
                mock.patch.object(torch.distributed, "get_rank", return_value=1)
            )
            stack.enter_context(mock.patch.object(torch, "empty", side_effect=cpu_empty))
            broadcast = stack.enter_context(
                mock.patch.object(torch.distributed, "broadcast", side_effect=receive)
            )
            observed = subject.broadcast_rank0_explicit_gaussian(
                None,
                expected_shape=shape,
                rank=1,
                device=torch.device("cuda", 1),
            )
            with self.assertRaises(subject.DecodedMiddleRepresentationError):
                subject.broadcast_rank0_explicit_gaussian(
                    authority,
                    expected_shape=shape,
                    rank=1,
                    device=torch.device("cuda", 1),
                )
        self.assertTrue(torch.equal(observed, authority))
        broadcast.assert_called_once()

    def test_materializer_digest_tamper_fails_closed(self) -> None:
        posterior = torch.zeros(1, 32, 21, 1, 1, dtype=torch.float32)
        blob = subject.posterior_tensor_to_transport_blob(posterior)
        metadata = {
            "posterior_parameters_shape": list(posterior.shape),
            "posterior_parameters_dtype": str(posterior.dtype),
            "posterior_parameters_tensor_sha256": "f" * 64,
        }
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject.load_validated_materializer_posterior(
                blob, metadata, label="tampered"
            )

    def test_registered_roles_and_shuffle_alias(self) -> None:
        self.assertEqual(
            subject.canonical_input_role("shuffle"), "temporal_shuffle"
        )
        self.assertEqual(
            tuple(subject.CONTROL_ROLES),
            (
                "real_forward",
                "temporal_shuffle",
                "reverse",
                "self_generated",
                "self_generated_temporal_shuffle",
                "self_generated_reverse",
            ),
        )
        self.assertFalse(subject.is_self_generated_role("real_forward"))
        self.assertFalse(subject.is_self_generated_role("temporal_shuffle"))
        self.assertTrue(subject.is_self_generated_role("self_generated"))
        self.assertTrue(
            subject.is_self_generated_role("self_generated_temporal_shuffle")
        )
        self.assertTrue(subject.is_self_generated_role("self_generated_reverse"))
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject.canonical_input_role("real_target")

    def test_projection_is_case_independent_and_deterministic(self) -> None:
        first = subject.deterministic_projection(
            12, 5, seed=91, device=torch.device("cpu")
        )
        second = subject.deterministic_projection(
            12, 5, seed=91, device=torch.device("cpu")
        )
        other = subject.deterministic_projection(
            12, 5, seed=92, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertFalse(first.requires_grad)

    def test_middle_preprocessing_removes_nuisance_and_only_returns_projection(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260824)
        shape = (1, 21, 3, 4, 12)
        appearance = torch.linspace(-1.0, 1.0, 12)
        noop = appearance.reshape(1, 1, 1, 1, 12).expand(shape).clone()
        delta = torch.zeros(shape)
        # Static appearance term: temporal DC must remove it.
        delta += torch.linspace(-0.3, 0.3, 12).reshape(1, 1, 1, 1, 12)
        # Per-phase global motion: camera common-mode must remove it.
        phase = torch.linspace(-0.4, 0.5, 21).reshape(1, 21, 1, 1, 1)
        delta += phase * torch.randn((1, 1, 1, 1, 12), generator=generator)
        # Local ordered action that survives both nuisance removals.
        for time in range(1, 21):
            y = time % 3
            x = (time // 2) % 4
            delta[0, time, y, x, 2:7] += 0.4 + time / 30.0
            delta[0, time, (y + 1) % 3, (x + 2) % 4, 7:11] -= 0.3
        action = (noop + delta).detach()
        noop = noop.detach()
        projection = subject.deterministic_projection(
            12, 5, seed=7, device=torch.device("cpu")
        )
        projected, metrics = subject.preprocess_middle_delta(
            action_hidden=action,
            noop_hidden=noop,
            appearance_direction=noop.mean(dim=(0, 1, 2, 3)),
            projection=projection,
        )
        self.assertEqual(tuple(projected.shape), (21, 12, 5))
        self.assertTrue(torch.equal(projected[0], torch.zeros_like(projected[0])))
        self.assertFalse(projected.requires_grad)
        self.assertIsNone(projected.grad_fn)
        self.assertGreater(metrics["raw_action_minus_noop_rms"], 0.0)
        self.assertLess(metrics["spatial_common_mode_max_abs_after"], 1.0e-5)
        self.assertLess(metrics["appearance_direction_max_abs_after"], 1.0e-5)

    def test_explicit_prepack_authority_uses_raw_bfloat16_sigma_bit_exactly(
        self,
    ) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(18)
        gaussian = torch.randn(21, 16, 1, 2, 2, generator=generator)
        noop_clean = torch.randn(21, 16, 1, 2, 2, generator=generator)
        action_clean = noop_clean.clone()
        action_clean[1:] += torch.linspace(0.0, 0.8, 20).reshape(20, 1, 1, 1, 1)
        sigma = torch.tensor([0.82421875], dtype=torch.bfloat16)
        original_pack_torch = _toy_pack_vae_latents.__globals__["torch"]
        original_process_pack = _toy_process_renderer_sample.__globals__[
            "pack_vae_latents"
        ]
        action, action_capture = _toy_explicit_batch(
            action_clean, gaussian, sigma=sigma
        )
        noop, noop_capture = _toy_explicit_batch(
            noop_clean, gaussian, sigma=sigma
        )
        self.assertIs(_toy_pack_vae_latents.__globals__["torch"], original_pack_torch)
        self.assertIs(
            _toy_process_renderer_sample.__globals__["pack_vae_latents"],
            original_process_pack,
        )
        self.assertEqual(action_capture.randn_like_injection_count, 1)
        self.assertEqual(action_capture.raw_noise_sigma.dtype, torch.bfloat16)
        expected_state = (1 - sigma) * action_clean + sigma * gaussian
        self.assertTrue(torch.equal(action_capture.packed_state, expected_state))
        self.assertTrue(
            torch.equal(action_capture.target_velocity, gaussian - action_clean.float())
        )

        vendor = {
            "vendor_data_file_sha256": subject.PINNED_BERNINI_DATA_SHA256,
            "pack_vae_latents_source_sha256": (
                subject.PINNED_PACK_VAE_LATENTS_SOURCE_SHA256
            ),
            "process_renderer_sample_source_sha256": (
                subject.PINNED_PROCESS_RENDERER_SAMPLE_SOURCE_SHA256
            ),
            "vendor_module_mutated": False,
            "original_function_globals_mutated": False,
        }
        matched = subject.matched_patch_pair_from_explicit_captures(
            action,
            noop,
            action_capture=action_capture,
            noop_capture=noop_capture,
            spatial_shape=(1, 16, 21, 2, 2),
            patches_to_spatial=_wan_test_patches_to_spatial,
            base_seed=17,
            derived_seed=23,
            vendor_identity=vendor,
        )
        self.assertFalse(matched.gaussian_authority["recovered_from_x_or_velocity"])
        self.assertEqual(matched.gaussian_authority["action_injection_count"], 1)
        self.assertEqual(matched.gaussian_authority["noop_injection_count"], 1)
        self.assertEqual(
            matched.gaussian_authority["canonical_gaussian_sha256"],
            subject.tensor_sha256(gaussian),
        )
        retimed = subject.retime_fm_batch(
            action,
            clean=matched.action_clean,
            gaussian=matched.gaussian,
            selector=matched.selector,
            sigma=0.73,
        )
        self.assertTrue(
            torch.equal(retimed["input_vae_rope"], action["input_vae_rope"])
        )
        expected_timestep = float(
            torch.full_like(action["timesteps"], 730.0).item()
        )
        self.assertEqual(float(retimed["timesteps"].item()), expected_timestep)

    def test_explicit_gaussian_seed_is_control_video_independent(self) -> None:
        self.assertNotIn(
            "input_video_sha256",
            inspect.signature(subject.derive_explicit_gaussian_seed).parameters,
        )
        values = {
            subject.derive_explicit_gaussian_seed(
                base_seed=91,
                case_id="case01",
                instruction_sha256="2" * 64,
            )
            for _control_video_sha in ("3" * 64, "4" * 64, "5" * 64)
        }
        self.assertEqual(len(values), 1)
        derived = values.pop()
        shape = subject.explicit_gaussian_packed_shape((1, 32, 21, 4, 6))
        first = subject.generate_rank0_explicit_gaussian(shape, derived_seed=derived)
        second = subject.generate_rank0_explicit_gaussian(shape, derived_seed=derived)
        self.assertTrue(torch.equal(first, second))

    def test_legacy_recovery_accepts_only_bit_identical_self_pair(self) -> None:
        clean = torch.zeros(21, 2)
        gaussian = torch.ones(21, 2)
        source = _batch(clean, gaussian, sigma=0.5)
        matched = subject.recover_matched_patch_pair(
            source,
            source,
            spatial_shape=(1, 2, 21, 1, 1),
            patches_to_spatial=_patches_to_spatial,
        )
        self.assertTrue(matched.gaussian_authority["recovered_from_x_or_velocity"])

    def test_recover_rejects_unmatched_gaussian(self) -> None:
        clean = torch.zeros(21, 2)
        gaussian = torch.ones(21, 2)
        action = _batch(clean, gaussian, sigma=0.5)
        noop = _batch(clean, gaussian + 0.1, sigma=0.5)
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject.recover_matched_patch_pair(
                action,
                noop,
                spatial_shape=(1, 2, 21, 1, 1),
                patches_to_spatial=_patches_to_spatial,
            )

    def test_explicit_injection_fails_closed_on_second_randn_like(self) -> None:
        gaussian = torch.zeros(21, 16, 1, 2, 2)
        clean = torch.ones_like(gaussian)
        original_pack_torch = _toy_double_randn_pack.__globals__["torch"]
        private_process = subject._clone_function_with_private_globals(
            _toy_process_renderer_sample,
            replacements={"pack_vae_latents": _toy_double_randn_pack},
        )
        original_process_pack = private_process.__globals__["pack_vae_latents"]
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject._run_cloned_renderer_process_with_explicit_gaussian(
                process_renderer_sample=private_process,
                pack_vae_latents=_toy_double_randn_pack,
                sample={
                    "clean": clean,
                    "sigma": torch.tensor([0.5], dtype=torch.bfloat16),
                    "timestep": torch.tensor([500.0], dtype=torch.bfloat16),
                },
                gaussian=gaussian,
                process_kwargs={},
            )
        self.assertIs(
            _toy_double_randn_pack.__globals__["torch"], original_pack_torch
        )
        self.assertIs(
            private_process.__globals__["pack_vae_latents"],
            original_process_pack,
        )
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject._run_cloned_renderer_process_with_explicit_gaussian(
                process_renderer_sample=_toy_process_renderer_sample,
                pack_vae_latents=_toy_pack_vae_latents,
                sample={
                    "clean": clean,
                    "sigma": torch.tensor([0.5], dtype=torch.bfloat16),
                    "timestep": torch.tensor([500.0], dtype=torch.bfloat16),
                },
                gaussian=gaussian.to(torch.float16),
                process_kwargs={},
            )

    def test_four_block_hook_is_read_only_and_target_only(self) -> None:
        width = 8
        transformer = _Transformer(width).eval().requires_grad_(False)
        layout = anchor_core.LocalTokenLayout.build(
            condition_tokens=0,
            patch_height=1,
            patch_width=2,
            phases=21,
        )
        hidden = torch.randn(1, layout.local_length, width)
        baseline = transformer(hidden.clone()).detach()
        bank = subject.MiddleBlockCaptureBank(transformer, hidden_width=width)
        bank.install()
        with bank.capture(layout):
            observed = transformer(hidden.clone()).detach()
        captures = bank.pop()
        bank.remove()
        self.assertTrue(torch.equal(observed, baseline))
        self.assertEqual(set(captures), set(subject.BLOCK_INDICES))
        for value in captures.values():
            self.assertEqual(tuple(value.shape), (1, 42, width))
            self.assertFalse(value.requires_grad)
        for index in subject.BLOCK_INDICES:
            self.assertFalse(transformer.blocks[index]._forward_hooks)

    def test_cache_and_receipt_are_firewall_closed_for_all_g1_roles(self) -> None:
        tensors = {
            f"middle_block_{index:02d}": torch.randn(3, 21, 6, 4).half()
            for index in subject.BLOCK_INDICES
        }
        for value in tensors.values():
            value[:, 0].zero_()
        rows = subject.validate_cache_tensors(
            tensors, sigma_count=3, projection_width=4
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "middle_repr.safetensors"
            cache.write_bytes(b"test-cache")
            for role in subject.CONTROL_ROLES:
                receipt = subject.build_receipt(
                    case_id="case01",
                    input_role=role,
                    input_video_sha256="1" * 64,
                    instruction_sha256="2" * 64,
                    cache_path=cache,
                    cache_sha256=hashlib.sha256(b"test-cache").hexdigest(),
                    cache_tensors=rows,
                    sigmas=(0.85, 0.55, 0.20),
                    projection_width=4,
                    projection_seed=7,
                    projection_sha256="3" * 64,
                    patch_grid=(21, 2, 3),
                    noise_max_abs_error=0.0,
                    noise_max_abs_forward_error_bound=0.0,
                    noise_max_error_to_bound_ratio=0.0,
                    noise_original_dtype="torch.float32",
                    noise_dtype_epsilon=float(torch.finfo(torch.float32).eps),
                    canonical_gaussian_sha256="5" * 64,
                    gaussian_authority=_explicit_authority_fixture(),
                    deterministic_vae_authority=(
                        _deterministic_vae_authority_fixture()
                    ),
                    phase0_clean_max_abs_error=0.0,
                    block_metrics={},
                    model_identity={"base_frozen": True},
                    runtime_identity={"world_size": 4},
                    method_source_sha256="4" * 64,
                )
                subject.validate_receipt(receipt)
                firewall = receipt["information_firewall"]
                self.assertFalse(firewall["target_video_accessed_by_trainer"])
                self.assertFalse(firewall["input_vae_or_clean_latent_persisted"])
                self.assertFalse(firewall["absolute_action_hidden_persisted"])
                self.assertEqual(
                    firewall["target_video_accessed_by_extractor"],
                    not subject.is_self_generated_role(role),
                )
                self.assertEqual(
                    firewall["target_rgb_or_vae_used_by_frozen_extractor"],
                    not subject.is_self_generated_role(role),
                )
                self.assertFalse(
                    firewall["target_rgb_or_vae_target_used_by_trainer"]
                )
                self.assertTrue(
                    firewall[
                        "ephemeral_posterior_broadcast_inside_frozen_extractor_only"
                    ]
                )
                self.assertFalse(
                    firewall["broadcast_posterior_payload_persisted"]
                )
                gaussian_match = receipt["representation"]["gaussian_match"]
                self.assertEqual(
                    gaussian_match["canonical_gaussian_sha256"], "5" * 64
                )
                self.assertEqual(
                    gaussian_match["comparison_stage"],
                    "before_fm_interpolation",
                )
                self.assertFalse(
                    gaussian_match["inverse_recovery_numerical_fields_applicable"]
                )
                authority = gaussian_match["authority"]
                self.assertFalse(authority["recovered_from_x_or_velocity"])
                self.assertFalse(authority["vendor_module_mutated"])
                self.assertFalse(authority["trainer_received_authority"])
                self.assertFalse(
                    gaussian_match["fixed_absolute_tolerance_is_authority"]
                )
                deterministic_vae = receipt["representation"][
                    "deterministic_vae_authority"
                ]
                self.assertTrue(
                    deterministic_vae["posterior_phase0_bit_exact"]
                )
                self.assertFalse(
                    deterministic_vae["posterior_modified_after_encode"]
                )
                self.assertFalse(
                    deterministic_vae["posterior_copy_or_splice_used"]
                )
                self.assertEqual(
                    receipt["representation"]["phase0_match_atol"], 0.0
                )
                serialized = subject.canonical_json_bytes(receipt)
                self.assertNotIn(b'"target_video_path"', serialized)

                tampered = copy.deepcopy(receipt)
                tampered["representation"]["deterministic_vae_authority"][
                    "posterior_copy_or_splice_used"
                ] = True
                tampered.pop("receipt_digest")
                tampered["receipt_digest"] = subject.object_sha256(tampered)
                with self.assertRaises(
                    subject.DecodedMiddleRepresentationError
                ):
                    subject.validate_receipt(tampered)

    def test_cache_rejects_forbidden_or_absolute_tensor_key(self) -> None:
        tensors = {
            f"middle_block_{index:02d}": torch.zeros(1, 21, 2, 3).half()
            for index in subject.BLOCK_INDICES
        }
        tensors["target_latent"] = torch.zeros(1)
        with self.assertRaises(subject.DecodedMiddleRepresentationError):
            subject.validate_cache_tensors(
                tensors, sigma_count=1, projection_width=3
            )

    @unittest.skipIf(
        importlib.util.find_spec("safetensors") is None,
        "local test environment does not provide safetensors",
    )
    def test_real_safetensors_publication_and_g1_loader_round_trip(self) -> None:
        sigmas = (0.85, 0.55, 0.20)
        tensors = {
            f"middle_block_{index:02d}": torch.randn(3, 21, 6, 4).half()
            for index in subject.BLOCK_INDICES
        }
        for value in tensors.values():
            value[:, 0].zero_()
        rows = subject.validate_cache_tensors(
            tensors, sigma_count=len(sigmas), projection_width=4
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "middle_repr.safetensors"
            receipt_path = Path(directory) / "receipt.json"
            subject._atomic_safetensors(
                cache,
                tensors,
                metadata={
                    "schema_version": subject.CACHE_SCHEMA,
                    "method": subject.METHOD,
                    "representation_origin": "decoded_video_reencode",
                    "anchor_source_role": "real_forward",
                    "blocks": ",".join(map(str, subject.BLOCK_INDICES)),
                    "sigmas": ",".join(f"{value:.9g}" for value in sigmas),
                    "projection_width": "4",
                    "contains_detached_projected_residuals_only": "true",
                    "contains_rgb_latent_absolute_hidden_qkv_or_endpoint": "false",
                },
            )
            receipt = subject.build_receipt(
                case_id="case01",
                input_role="real_forward",
                input_video_sha256="1" * 64,
                instruction_sha256="2" * 64,
                cache_path=cache,
                cache_sha256=subject.file_sha256(cache),
                cache_tensors=rows,
                sigmas=sigmas,
                projection_width=4,
                projection_seed=7,
                projection_sha256="3" * 64,
                patch_grid=(21, 2, 3),
                noise_max_abs_error=0.0,
                noise_max_abs_forward_error_bound=0.0,
                noise_max_error_to_bound_ratio=0.0,
                noise_original_dtype="torch.float32",
                noise_dtype_epsilon=float(torch.finfo(torch.float32).eps),
                canonical_gaussian_sha256="5" * 64,
                gaussian_authority=_explicit_authority_fixture(),
                deterministic_vae_authority=(
                    _deterministic_vae_authority_fixture()
                ),
                phase0_clean_max_abs_error=0.0,
                block_metrics={},
                model_identity={"base_frozen": True},
                runtime_identity={"world_size": 4},
                method_source_sha256="4" * 64,
            )
            subject._atomic_json(receipt_path, receipt)
            loaded, loaded_receipt = subject.load_middle_representation_cache(
                cache, receipt_path, expected_role="real_forward"
            )
            self.assertEqual(loaded_receipt["receipt_digest"], receipt["receipt_digest"])
            self.assertEqual(set(loaded), set(tensors))
            for key in tensors:
                self.assertTrue(torch.equal(loaded[key], tensors[key]))


if __name__ == "__main__":
    unittest.main()

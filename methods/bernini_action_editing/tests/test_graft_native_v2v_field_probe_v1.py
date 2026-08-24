#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import graft_native_v2v_field_probe_v1 as probe  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch required")
class NativeV2VFieldProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
        cls.torch = torch

    @staticmethod
    def _vendor_symbols():
        import torch
        import torch.nn.functional as functional

        class MomentumBuffer:
            def __init__(self, momentum):
                self.momentum = momentum
                self.running_average = 0

            def update(self, update_value):
                self.running_average = update_value + self.momentum * self.running_average

        def normalized_guidance(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            diff = pred_cond - pred_uncond
            if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
            if norm_threshold > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
                diff = diff * torch.minimum(ones, norm_threshold / diff_norm)
            projected, base = diff.double(), pred_cond.double()
            base = functional.normalize(base, dim=[-1, -2, -4])
            parallel = (projected * base).sum(
                dim=[-1, -2, -4], keepdim=True
            ) * base
            orthogonal = projected - parallel
            normalized = orthogonal.to(diff.dtype) + eta * parallel.to(diff.dtype)
            return pred_uncond + guidance_scale * normalized

        return normalized_guidance, MomentumBuffer

    def _apg_inputs(self):
        torch = self.torch
        cond = torch.linspace(-0.8, 1.1, 2 * 3 * 2 * 2 * 2, dtype=torch.float32)
        cond = cond.reshape(2, 3, 2, 2, 2).requires_grad_(True)
        uncond = torch.linspace(0.7, -1.3, cond.numel(), dtype=torch.float32)
        uncond = uncond.reshape_as(cond).requires_grad_(True)
        cotangent = torch.linspace(0.2, 1.2, cond.numel(), dtype=torch.float32)
        return cond, uncond, cotangent.reshape_as(cond)

    def test_reuses_authenticated_native_vonly_contract(self) -> None:
        receipt = probe.native_field_wiring_receipt()
        self.assertEqual(receipt["patch_source_ids"], [1.0, 0.0])
        self.assertEqual(receipt["forward_order"], ["negative", "action"])
        self.assertEqual(receipt["guidance_mode"], "v2v_apg")
        self.assertTrue(receipt["negative_positive_same_visual_pack"])
        self.assertEqual(receipt["raw_output_target_selection"], "last_target_tokens")
        self.assertFalse(receipt["existing_observers_training_usable"])
        self.assertFalse(receipt["training_claim_authorized"])

    def test_connected_target_suffix_preserves_both_graphs(self) -> None:
        torch = self.torch
        source_tokens = target_tokens = 4
        visual = torch.zeros(1, 8, probe.EXPECTED_HIDDEN_DIM)
        rotary = torch.zeros(1, 1, 8, 8)
        timestep = torch.tensor([999.0])
        negative_leaf = torch.arange(8 * 64, dtype=torch.float32, requires_grad=True)
        action_leaf = torch.arange(8 * 64, dtype=torch.float32, requires_grad=True) + 2
        negative = (negative_leaf * 2).reshape(1, 8, 64)
        action = (action_leaf * 3).reshape(1, 8, 64)
        pair = probe.connected_target_tail_pair(
            negative_visual_pack=visual,
            action_visual_pack=visual,
            negative_rotary=rotary,
            action_rotary=rotary,
            negative_timestep=timestep,
            action_timestep=timestep,
            negative_raw_output=negative,
            action_raw_output=action,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
        )
        self.assertEqual(tuple(pair.negative.shape), (1, 4, 64))
        self.assertEqual(tuple(pair.action.shape), (1, 4, 64))
        self.assertTrue(pair.negative.requires_grad)
        self.assertTrue(pair.action.requires_grad)
        pair.negative.sum().backward()
        self.assertIsNotNone(negative_leaf.grad)
        self.assertEqual(int(torch.count_nonzero(negative_leaf.grad).item()), 4 * 64)

    def test_negative_action_must_share_exact_visual_objects(self) -> None:
        torch = self.torch
        visual = torch.zeros(1, 2, probe.EXPECTED_HIDDEN_DIM)
        rotary = torch.zeros(1, 1, 2, 8)
        timestep = torch.tensor([999.0])
        raw = (torch.zeros(1, 2, 64, requires_grad=True) + 1)
        with self.assertRaisesRegex(
            probe.GraftNativeV2VFieldProbeError, "same object"
        ):
            probe.connected_target_tail_pair(
                negative_visual_pack=visual,
                action_visual_pack=visual.clone(),
                negative_rotary=rotary,
                action_rotary=rotary,
                negative_timestep=timestep,
                action_timestep=timestep,
                negative_raw_output=raw,
                action_raw_output=raw + 1,
                source_tokens=1,
                target_tokens=1,
            )

    def test_detached_old_style_raw_output_is_rejected(self) -> None:
        torch = self.torch
        visual = torch.zeros(1, 2, probe.EXPECTED_HIDDEN_DIM)
        rotary = torch.zeros(1, 1, 2, 8)
        timestep = torch.tensor([999.0])
        detached = torch.zeros(1, 2, 64)
        with self.assertRaisesRegex(
            probe.GraftNativeV2VFieldProbeError, "old inference observer"
        ):
            probe.connected_target_tail_pair(
                negative_visual_pack=visual,
                action_visual_pack=visual,
                negative_rotary=rotary,
                action_rotary=rotary,
                negative_timestep=timestep,
                action_timestep=timestep,
                negative_raw_output=detached,
                action_raw_output=detached,
                source_tokens=1,
                target_tokens=1,
            )

    def test_vendor_fp32_leaves_forward_and_vjp_are_exact(self) -> None:
        vendor, momentum = self._vendor_symbols()
        cond, uncond, cotangent = self._apg_inputs()
        receipt = probe.normalized_guidance_vjp_parity(
            vendor_normalized_guidance=vendor,
            momentum_buffer_factory=momentum,
            pred_cond=cond,
            pred_uncond=uncond,
            cotangent=cotangent,
        )
        self.assertTrue(receipt["vendor_output_connected"])
        self.assertTrue(receipt["vendor_forward_independent_parity"])
        self.assertTrue(receipt["vendor_vjp_independent_parity"])
        self.assertEqual(receipt["forward_max_abs_error"], 0.0)
        self.assertEqual(receipt["conditional_vjp_max_abs_error"], 0.0)
        self.assertEqual(receipt["negative_vjp_max_abs_error"], 0.0)

    def test_vendor_detach_fails_closed(self) -> None:
        vendor, momentum = self._vendor_symbols()

        def detached_vendor(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            return vendor(
                pred_cond,
                pred_uncond,
                guidance_scale,
                momentum_buffer,
                eta,
                norm_threshold,
            ).detach()

        cond, uncond, cotangent = self._apg_inputs()
        with self.assertRaisesRegex(
            probe.GraftNativeV2VFieldProbeError, "detached FP32 leaves"
        ):
            probe.normalized_guidance_vjp_parity(
                vendor_normalized_guidance=detached_vendor,
                momentum_buffer_factory=momentum,
                pred_cond=cond,
                pred_uncond=uncond,
                cotangent=cotangent,
            )

    def test_wrong_vendor_algebra_fails_parity(self) -> None:
        vendor, momentum = self._vendor_symbols()

        def wrong_vendor(
            pred_cond,
            pred_uncond,
            guidance_scale,
            momentum_buffer=None,
            eta=1.0,
            norm_threshold=0.0,
        ):
            return vendor(
                pred_cond,
                pred_uncond,
                guidance_scale + 0.125,
                momentum_buffer,
                eta,
                norm_threshold,
            )

        cond, uncond, cotangent = self._apg_inputs()
        with self.assertRaisesRegex(
            probe.GraftNativeV2VFieldProbeError, "forward/VJP parity failed"
        ):
            probe.normalized_guidance_vjp_parity(
                vendor_normalized_guidance=wrong_vendor,
                momentum_buffer_factory=momentum,
                pred_cond=cond,
                pred_uncond=uncond,
                cotangent=cotangent,
            )

    def test_nonleaf_or_nonfp32_input_fails_closed(self) -> None:
        torch = self.torch
        vendor, momentum = self._vendor_symbols()
        cond, uncond, cotangent = self._apg_inputs()
        for bad in (cond + 0.0, cond.detach().to(torch.float64).requires_grad_(True)):
            with self.subTest(dtype=str(bad.dtype), leaf=bad.is_leaf):
                with self.assertRaisesRegex(
                    probe.GraftNativeV2VFieldProbeError, "FP32 five-dimensional leaf"
                ):
                    probe.normalized_guidance_vjp_parity(
                        vendor_normalized_guidance=vendor,
                        momentum_buffer_factory=momentum,
                        pred_cond=bad,
                        pred_uncond=uncond,
                        cotangent=cotangent,
                    )


if __name__ == "__main__":
    unittest.main()

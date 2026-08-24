from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType, SimpleNamespace
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import torch
from torch import nn

import self_imagined_native_rv2v_hidden_vjp_v1 as core
import mosaic_starc_stateless_jacobian_qp as qp

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover
    serialization = None
    Ed25519PrivateKey = None


class _IdentityBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _HookBernini(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_IdentityBlock() for _ in range(core.TOTAL_BLOCKS_1P3B)]
        )


class _NativePatchFake:
    dtype = torch.float32

    def patch_vae_latent(self, value: torch.Tensor, *, source_id: float):
        del source_id
        batch, _channels, phases, height, width = value.shape
        tokens = phases * (height // 2) * (width // 2)
        latent = torch.zeros(batch, tokens, 2, dtype=value.dtype)
        rotary = torch.zeros(batch, 1, tokens, 2, dtype=value.dtype)
        return latent, rotary


class _Attention(nn.Module):
    def __init__(self, query: nn.Linear, output: nn.Linear) -> None:
        super().__init__()
        self.to_q = query
        self.to_out = nn.ModuleList([output, nn.Identity()])


class _InstallBlock(nn.Module):
    def __init__(self, query: nn.Linear, output: nn.Linear) -> None:
        super().__init__()
        self.attn2 = _Attention(query, output)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _InstallBernini(nn.Module):
    """Low-memory structural fake: frozen Q/O bases are shared across blocks."""

    def __init__(self) -> None:
        super().__init__()
        query = nn.Linear(core.HIDDEN_SIZE, core.HIDDEN_SIZE, bias=False)
        output = nn.Linear(core.HIDDEN_SIZE, core.HIDDEN_SIZE, bias=False)
        self.patch_embedding = nn.Conv3d(
            16, core.HIDDEN_SIZE, kernel_size=(1, 2, 2), bias=False
        )
        self.blocks = nn.ModuleList(
            [
                _InstallBlock(query, output)
                for _ in range(core.TOTAL_BLOCKS_1P3B)
            ]
        )

    def patch_vae_latent(self, value: torch.Tensor, source_id: float):
        del source_id
        return value, value


class _RuntimeBernini(_InstallBernini):
    def patch_vae_latent(self, value: torch.Tensor, source_id: float):
        del source_id
        pooled = torch.nn.functional.avg_pool3d(
            value.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2)
        ).mean(dim=1)
        latent = pooled.reshape(value.shape[0], -1, 1).expand(
            -1, -1, core.HIDDEN_SIZE
        )
        rotary = torch.zeros(
            value.shape[0],
            1,
            latent.shape[1],
            2,
            dtype=latent.dtype,
            device=latent.device,
        )
        return latent, rotary


class _RuntimeDiffusion(nn.Module):
    def __init__(self, transformer: _RuntimeBernini, *, sp_rank: int) -> None:
        super().__init__()
        self.transformer = transformer
        self.sp_rank = sp_rank
        self.calls = []

    def shared_step(self, **kwargs):
        hidden = kwargs["noisy_latents"]
        condition = kwargs["cond_embeds"]
        local_length = (hidden.shape[1] + core.SP_SIZE - 1) // core.SP_SIZE
        start = self.sp_rank * local_length
        stop = min(start + local_length, hidden.shape[1])
        local = hidden[:, start:stop]
        if local.shape[1] < local_length:
            local = torch.cat(
                (
                    local,
                    torch.zeros(
                        1,
                        local_length - local.shape[1],
                        core.HIDDEN_SIZE,
                        dtype=local.dtype,
                        device=local.device,
                    ),
                ),
                dim=1,
            )
        ramp = torch.linspace(
            -1.0, 1.0, local_length, dtype=local.dtype, device=local.device
        ).reshape(1, local_length, 1)
        observed = self.transformer.blocks[core.HOOK_BLOCK_INDEX](
            local + condition.float().mean() * ramp
        )
        self.calls.append(kwargs["model_id"])
        return hidden + observed.sum() * 0.0


class _FakeSP4Dist:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    @staticmethod
    def get_world_size() -> int:
        return 4

    def get_rank(self) -> int:
        return self.rank

    @staticmethod
    def all_reduce(value: torch.Tensor) -> None:
        # One-process canary: geometry and call contract are exercised; the
        # four-rank SUM itself is covered by the existing aggregation tests.
        value.add_(0.0)


class _QuadraticScorer(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.query_seed = seed
        self.template_digest = f"toy-query-{seed}"

    def forward_sketched_residual(
        self, residual: torch.Tensor, *, require_input_grad: bool
    ) -> SimpleNamespace:
        if require_input_grad and not residual.requires_grad:
            raise RuntimeError("test scorer lost graph")
        weights = torch.linspace(
            0.5,
            1.5,
            residual.numel(),
            dtype=residual.dtype,
            device=residual.device,
        ).reshape_as(residual)
        return SimpleNamespace(score=(residual.square() * weights).mean())


def _measurement_pair(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (
        1,
        core.LATENT_PHASES,
        core.SPATIAL_SKETCH_COORDINATES,
        core.HIDDEN_SIZE,
    )
    action = torch.randn(shape, generator=generator, dtype=torch.float32)
    noop = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
    return action.detach().contiguous(), noop.detach().contiguous()


def _native_layouts(branch: str, *, patch_height: int = 4, patch_width: int = 5):
    return [
        core.build_native_target_suffix_layout(
            branch_name=branch,
            patch_height=patch_height,
            patch_width=patch_width,
            sp_rank=rank,
        )
        for rank in range(core.SP_SIZE)
    ]


def _tiny_action_route(*, total_tokens: int, condition_tokens: int):
    return core.pair_adapter.PairV5ActionRoute(
        total_tokens=total_tokens,
        condition_tokens=condition_tokens,
        sequence_parallel_rank=0,
        sequence_parallel_size=1,
        branch_name="I",
        sigma_schedule_index=core.NATIVE_SCHEDULE_INDEX,
        enabled=True,
    )


def _core16_proof_branch() -> core.native.NativeRV2VBranch:
    total_tokens = 8
    condition_tokens = 4
    target_mask = torch.cat(
        (
            torch.zeros(condition_tokens, dtype=torch.bool),
            torch.ones(total_tokens - condition_tokens, dtype=torch.bool),
        )
    )
    return core.native.NativeRV2VBranch(
        name="VI",
        latents=torch.zeros(1, total_tokens, core.HIDDEN_SIZE),
        rotary=torch.zeros(1, 1, total_tokens, 2),
        target_mask=target_mask,
        total_tokens=total_tokens,
        condition_tokens=condition_tokens,
        source_ids=(0.0,),
        concat_order=("unit-test",),
    )


class NativeTargetSuffixLayoutTests(unittest.TestCase):
    def test_layout_authenticates_real_native_pack_branch_objects(self) -> None:
        donor = torch.zeros(1, 16, 21, 8, 10)
        target = torch.ones_like(donor)
        refs = [torch.zeros(1, 16, 1, 8, 10) for _ in range(4)]
        pack = core.native.build_native_rv2v_pack(
            _NativePatchFake(),
            donor_video=donor,
            image_references=refs,
            noisy_target=target,
        )
        branches = (pack.none, pack.video, pack.image, pack.video_image)
        self.assertEqual(tuple(row.name for row in branches), core.BRANCH_NAMES)
        for branch in branches:
            layouts = [
                core.layout_from_native_branch(
                    branch,
                    patch_height=4,
                    patch_width=5,
                    sp_rank=rank,
                )
                for rank in range(4)
            ]
            self.assertEqual(
                sum(int(row.local_target_indices.numel()) for row in layouts),
                branch.total_tokens - branch.condition_tokens,
            )

    def test_all_native_condition_prefixes_cover_target_exactly_once(self) -> None:
        patch_positions = 20
        for branch, condition_phases in core.CONDITION_PHASES_BY_BRANCH.items():
            layouts = _native_layouts(branch)
            self.assertEqual(layouts[0].condition_tokens, condition_phases * patch_positions)
            self.assertEqual(layouts[0].target_tokens, 21 * patch_positions)
            target_flat = torch.cat([row.target_flat_indices for row in layouts])
            self.assertTrue(
                torch.equal(target_flat.sort().values, torch.arange(21 * patch_positions))
            )
            full_global = torch.cat([row.global_target_indices for row in layouts])
            self.assertTrue(bool((full_global >= layouts[0].condition_tokens).all()))
            counts = torch.stack([row.phase_patch_count for row in layouts]).sum(0)
            self.assertTrue(
                torch.equal(counts, torch.full((21,), patch_positions, dtype=torch.int64))
            )
            receipt = layouts[0].receipt()
            self.assertIn("g=sp_rank*ceil", receipt["global_index_formula"])
            self.assertEqual(
                receipt["digest"],
                core.object_sha256({k: v for k, v in receipt.items() if k != "digest"}),
            )

    def test_observed_prefix_or_total_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "condition-prefix"):
            core.build_native_target_suffix_layout(
                branch_name="VI",
                patch_height=4,
                patch_width=5,
                sp_rank=0,
                observed_condition_tokens=499,
            )
        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "total token"):
            core.build_native_target_suffix_layout(
                branch_name="V",
                patch_height=4,
                patch_width=5,
                sp_rank=0,
                observed_total_tokens=1,
            )

    def test_condition_and_padding_hidden_rows_have_zero_sketch_gradient(self) -> None:
        layouts = [
            core.build_native_target_suffix_layout(
                branch_name="I",
                patch_height=1,
                patch_width=17,
                sp_rank=rank,
            )
            for rank in range(4)
        ]
        self.assertGreater(layouts[0].condition_rows_excluded, 0)
        self.assertGreater(layouts[3].padding_rows_excluded, 0)
        sketch = core.make_fixed_spatial_sketch(layouts[0].patch_positions)
        for layout in (layouts[0], layouts[3]):
            hidden = torch.randn(
                1,
                layout.local_length,
                core.HIDDEN_SIZE,
                dtype=torch.float32,
                requires_grad=True,
            )
            capture = core.sketch_local_block15_target_suffix(
                hidden,
                layout=layout,
                spatial_sketch=sketch,
                role="action",
                detach=False,
            )
            capture.tensor.square().sum().backward()
            self.assertIsNotNone(hidden.grad)
            target = torch.zeros(layout.local_length, dtype=torch.bool)
            target[layout.local_target_indices] = True
            self.assertTrue(torch.equal(hidden.grad[:, ~target], torch.zeros_like(hidden.grad[:, ~target])))
            self.assertGreater(float(hidden.grad[:, target].abs().sum()), 0.0)

    def test_fake_bernini_block15_hook_is_detached_or_graph_connected_by_mode(self) -> None:
        model = _HookBernini()
        layout = _native_layouts("none")[0]
        sketch = core.make_fixed_spatial_sketch(layout.patch_positions)
        observer = core.Block15TargetSuffixObserver(model, spatial_sketch=sketch)
        observer.install()
        try:
            hidden = torch.randn(
                1,
                layout.local_length,
                core.HIDDEN_SIZE,
                requires_grad=True,
            )
            with observer.capture(role="measure-action", layout=layout, detach=True) as measured:
                model.blocks[core.HOOK_BLOCK_INDEX](hidden)
            with observer.capture(role="replay-action", layout=layout, detach=False) as replayed:
                model.blocks[core.HOOK_BLOCK_INDEX](hidden)
        finally:
            observer.remove()
        self.assertFalse(measured[0].graph_connected)
        self.assertTrue(replayed[0].graph_connected)
        self.assertTrue(torch.equal(measured[0].tensor, replayed[0].tensor.detach()))
        gradient = torch.autograd.grad(replayed[0].tensor.square().sum(), hidden)[0]
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)

    def test_sp4_detached_sketch_sum_matches_direct_full_target_reference(self) -> None:
        layouts = _native_layouts("I")
        sketch = core.make_fixed_spatial_sketch(layouts[0].patch_positions)
        captures = []
        for layout in layouts:
            hidden = torch.zeros(1, layout.local_length, core.HIDDEN_SIZE)
            local = layout.local_target_indices
            hidden[0, local, 0] = layout.target_flat_indices.float() + 1.0
            captures.append(
                core.sketch_local_block15_target_suffix(
                    hidden,
                    layout=layout,
                    spatial_sketch=sketch,
                    role="action",
                    detach=True,
                )
            )
        observed = core.sum_detached_sp4_sketches(captures)
        target = torch.arange(1, 21 * layouts[0].patch_positions + 1, dtype=torch.float32)
        target = target.reshape(21, layouts[0].patch_positions)
        expected = torch.einsum("tp,kp->tk", target, sketch)
        self.assertTrue(torch.allclose(observed[0, :, :, 0], expected, atol=2.0e-4, rtol=1.0e-6))
        self.assertTrue(torch.equal(observed[..., 1:], torch.zeros_like(observed[..., 1:])))


class ExactZeroActionLoRARouteTests(unittest.TestCase):
    @staticmethod
    def _wrapper(
        *, dtype: torch.dtype = torch.float32
    ) -> core.Core16ExactZeroTargetRowActionLoRA:
        base = nn.Linear(4, 4, bias=False, dtype=dtype)
        wrapper = core.Core16ExactZeroTargetRowActionLoRA(
            base,
            projection="to_q",
            canonical_b_name=core.CANONICAL_B_PARAMETER_NAMES[0],
        )
        wrapper.action_lora_a.weight.requires_grad_(False)
        return wrapper

    def test_zero_b_forward_preserves_fp32_and_bf16_signed_zero_bytes(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                wrapper = self._wrapper(dtype=dtype)
                pattern = torch.tensor(
                    [-0.0, 0.0, -0.0, 0.0], dtype=dtype
                ).reshape(1, 1, 4)

                def signed_zero_output(
                    _module: nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    _output: torch.Tensor,
                ) -> torch.Tensor:
                    return pattern.expand(
                        int(inputs[0].shape[0]), int(inputs[0].shape[1]), 4
                    ).clone()

                hook = wrapper.base.register_forward_hook(signed_zero_output)
                hidden = torch.zeros(1, 3, 4, dtype=dtype)
                route = _tiny_action_route(total_tokens=3, condition_tokens=1)
                try:
                    expected = wrapper.base(hidden)
                    with core.pair_adapter.activate_route(route):
                        observed = wrapper(hidden)
                finally:
                    hook.remove()
                expected_sha256 = core.tensor_sha256(
                    expected, label=f"{dtype} signed-zero expected"
                )
                self.assertNotEqual(
                    expected_sha256,
                    core.tensor_sha256(
                        torch.zeros_like(expected),
                        label=f"{dtype} positive-zero control",
                    ),
                )
                self.assertEqual(
                    core.tensor_sha256(
                        observed, label=f"{dtype} signed-zero observed"
                    ),
                    expected_sha256,
                )

    def test_zero_b_custom_backward_reaches_base_and_b_but_not_fixed_a(self) -> None:
        wrapper = self._wrapper()
        with torch.no_grad():
            wrapper.base.weight.copy_(torch.eye(4))
            wrapper.action_lora_a.weight.fill_(0.25)
            wrapper.action_lora_b.weight.zero_()
        hidden = torch.tensor(
            [[[9.0, 8.0, 7.0, 6.0], [1.0, 2.0, 3.0, 4.0]]],
            requires_grad=True,
        )
        route = _tiny_action_route(total_tokens=2, condition_tokens=1)
        with core.pair_adapter.activate_route(route):
            result = wrapper(hidden)
        base_gradient, b_gradient, hidden_gradient = torch.autograd.grad(
            result.sum(),
            (wrapper.base.weight, wrapper.action_lora_b.weight, hidden),
        )
        self.assertTrue(bool(torch.isfinite(base_gradient).all().item()))
        self.assertTrue(bool(torch.isfinite(b_gradient).all().item()))
        self.assertTrue(bool(torch.isfinite(hidden_gradient).all().item()))
        self.assertGreater(float(torch.linalg.vector_norm(base_gradient)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(b_gradient)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(hidden_gradient)), 0.0)
        self.assertFalse(wrapper.action_lora_a.weight.requires_grad)
        self.assertIsNone(wrapper.action_lora_a.weight.grad)
        self.assertEqual(int(torch.count_nonzero(wrapper.action_lora_b.weight)), 0)

    def test_nonzero_b_keeps_pair_v5_target_only_residual_semantics(self) -> None:
        wrapper = self._wrapper()
        with torch.no_grad():
            wrapper.base.weight.copy_(torch.eye(4))
            wrapper.action_lora_a.weight.fill_(0.25)
            wrapper.action_lora_b.weight.fill_(0.125)
        hidden = torch.tensor(
            [[[8.0, 7.0, 6.0, 5.0], [1.0, 2.0, 3.0, 4.0]]]
        )
        route = _tiny_action_route(total_tokens=2, condition_tokens=1)
        selector = route.local_target_selector(device=hidden.device)
        base = wrapper.base(hidden)
        delta = wrapper._selected_delta(
            hidden, selector, route.gate_weight
        ).to(base.dtype)
        with core.pair_adapter.activate_route(route):
            result = wrapper(hidden)
        self.assertEqual(
            core.tensor_sha256(
                result[:, ~selector, :], label="nonzero route source rows"
            ),
            core.tensor_sha256(
                base[:, ~selector, :], label="nonzero route base source rows"
            ),
        )
        self.assertTrue(
            torch.equal(result[:, selector, :], base[:, selector, :] + delta)
        )
        self.assertGreater(float(torch.linalg.vector_norm(delta)), 0.0)


class FixedGaugeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(19)
        cls.model = _InstallBernini().requires_grad_(False)
        cls.original_late = tuple(
            (
                cls.model.blocks[index].attn2.to_q,
                cls.model.blocks[index].attn2.to_out[0],
            )
            for index in range(16, core.TOTAL_BLOCKS_1P3B)
        )
        cls.handle = core.install_core16_fixed_a_b_only_action_lora(cls.model)

    @classmethod
    def tearDownClass(cls) -> None:
        if not cls.handle.restored:
            cls.handle.restore()

    def test_canonical_b_order_is_q_o_interleaved_per_block(self) -> None:
        names = tuple(name for name, _ in self.handle.canonical_b_named_parameters())
        self.assertEqual(names, core.CANONICAL_B_PARAMETER_NAMES)
        self.assertEqual(names, qp.CANONICAL_PARAMETER_NAMES)
        self.assertEqual(core.CANONICAL_B_SHAPE, qp.CANONICAL_B_SHAPE)
        self.assertEqual(core.CANONICAL_A_SHAPE, qp.CANONICAL_A_SHAPE)
        self.assertEqual(core.CANONICAL_B_PARAMETER_COUNT, qp.CANONICAL_PARAMETER_COUNT)
        for block in core.ACTION_BLOCK_INDICES:
            self.assertEqual(
                names[2 * block : 2 * block + 2],
                (
                    f"blocks.{block}.attn2.to_q.action_lora_b.weight",
                    f"blocks.{block}.attn2.to_out.0.action_lora_b.weight",
                ),
            )
        legacy_style = tuple(
            [name for name in names if ".to_q." in name]
            + [name for name in names if ".to_out.0." in name]
        )
        self.assertNotEqual(names, legacy_style)

    def test_fixed_seed_fp32_a_frozen_and_zero_b_only(self) -> None:
        self.handle.assert_fixed_gauge()
        a = self.handle.canonical_a_named_parameters()
        b = self.handle.canonical_b_named_parameters()
        self.assertEqual(len(a), 32)
        self.assertEqual(len(b), 32)
        self.assertEqual(sum(parameter.numel() for _, parameter in b), 393216)
        self.assertTrue(all(parameter.dtype == torch.float32 for _, parameter in (*a, *b)))
        self.assertTrue(all(not parameter.requires_grad for _, parameter in a))
        self.assertTrue(all(parameter.requires_grad for _, parameter in b))
        self.assertTrue(all(torch.count_nonzero(parameter) == 0 for _, parameter in b))
        for name, parameter in a:
            expected = core._fixed_a_cpu(name, core.CANONICAL_A_SHAPE)
            self.assertTrue(torch.equal(parameter.detach().cpu(), expected))
            self.assertEqual(int(torch.linalg.matrix_rank(parameter.detach().cpu())), 8)
        layout = qp.FixedParameterLayout.from_ordered_parameters(b)
        self.assertEqual(
            self.handle.b_parameter_state_sha256(),
            layout.parameter_state_sha256,
        )

    def test_only_blocks_0_through_15_are_wrapped(self) -> None:
        for index in core.ACTION_BLOCK_INDICES:
            self.assertIsInstance(
                self.model.blocks[index].attn2.to_q,
                core.pair_adapter.PairV5TargetRowActionLoRA,
            )
            self.assertIsInstance(
                self.model.blocks[index].attn2.to_q,
                core.Core16ExactZeroTargetRowActionLoRA,
            )
            self.assertIsInstance(
                self.model.blocks[index].attn2.to_out[0],
                core.pair_adapter.PairV5TargetRowActionLoRA,
            )
            self.assertIsInstance(
                self.model.blocks[index].attn2.to_out[0],
                core.Core16ExactZeroTargetRowActionLoRA,
            )
        for offset, index in enumerate(range(16, core.TOTAL_BLOCKS_1P3B)):
            self.assertIs(self.model.blocks[index].attn2.to_q, self.original_late[offset][0])
            self.assertIs(self.model.blocks[index].attn2.to_out[0], self.original_late[offset][1])
        self.assertTrue(self.handle.late_blocks_untouched())
        self.assertFalse(self.handle.gauge_receipt["optimizer_or_parameter_update"])
        self.assertFalse(self.handle.gauge_receipt["real_auh_runtime_validated"])

    def test_zero_route_proof_closes_all_wrappers_once_with_plain_receipt(self) -> None:
        branch = _core16_proof_branch()
        hidden = torch.zeros(1, 2, core.HIDDEN_SIZE)
        with self.handle.capture_zero_route_proof(
            role="action", sp_rank=3
        ) as holder:
            with self.handle.route(
                branch, sp_rank=3, adapter_enabled=True
            ):
                for name in core.CANONICAL_B_PARAMETER_NAMES:
                    self.handle.wrappers_by_b_name[name](hidden)
        receipt = holder.require_receipt()
        self.assertEqual(
            receipt["schema_version"], core.ZERO_ROUTE_PROOF_SCHEMA_VERSION
        )
        self.assertEqual(receipt["wrapper_count"], 32)
        self.assertEqual(receipt["missing_wrapper_count"], 0)
        self.assertEqual(receipt["repeated_wrapper_count"], 0)
        self.assertTrue(receipt["all_base_result_raw_bytes_equal"])
        self.assertTrue(receipt["all_selected_deltas_numerically_exact_zero"])
        self.assertEqual(len(receipt["call_evidence"]), 32)
        self.assertTrue(
            all(
                row["selected_row_count"] == row["local_row_count"] == 2
                and row["b_raw_nonzero_byte_count"] == 0
                and row["selected_delta_nonzero_element_count"] == 0
                and row["base_result_raw_byte_mismatch_count"] == 0
                for row in receipt["call_evidence"]
            )
        )
        self.assertEqual(
            receipt["call_evidence_sha256"],
            core.object_sha256(receipt["call_evidence"]),
        )
        self.assertEqual(
            receipt["b_state_before_sha256"], receipt["b_state_after_sha256"]
        )
        self.assertEqual(
            receipt["digest"],
            core.object_sha256(
                {key: value for key, value in receipt.items() if key != "digest"}
            ),
        )
        json.dumps(receipt, allow_nan=False)

    def test_zero_route_proof_accepts_a_source_only_sp_rank(self) -> None:
        branch = _core16_proof_branch()
        hidden = torch.zeros(1, 2, core.HIDDEN_SIZE)
        with self.handle.capture_zero_route_proof(
            role="noop", sp_rank=0
        ) as holder:
            with self.handle.route(
                branch, sp_rank=0, adapter_enabled=True
            ):
                for name in core.CANONICAL_B_PARAMETER_NAMES:
                    self.handle.wrappers_by_b_name[name](hidden)
        receipt = holder.require_receipt()
        self.assertEqual(receipt["total_selected_row_count"], 0)
        self.assertTrue(
            all(
                row["selected_row_count"] == 0
                and row["selected_delta_nonzero_element_count"] == 0
                and row["base_result_raw_byte_mismatch_count"] == 0
                for row in receipt["call_evidence"]
            )
        )
        self.handle.assert_fixed_gauge()

    def test_zero_route_proof_rejects_repeated_and_missing_wrappers(self) -> None:
        branch = _core16_proof_branch()
        hidden = torch.zeros(1, 2, core.HIDDEN_SIZE)
        first = self.handle.wrappers_by_b_name[
            core.CANONICAL_B_PARAMETER_NAMES[0]
        ]
        with self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "wrapper repeated"
        ):
            with self.handle.capture_zero_route_proof(
                role="action", sp_rank=3
            ):
                with self.handle.route(
                    branch, sp_rank=3, adapter_enabled=True
                ):
                    first(hidden)
                    first(hidden)
        with self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "missing or reordered"
        ):
            with self.handle.capture_zero_route_proof(
                role="noop", sp_rank=3
            ):
                pass
        self.handle.assert_fixed_gauge()

    def test_clean_latent_vjp_is_finite_nonzero_and_leaves_b_unchanged(self) -> None:
        action, noop = _measurement_pair()
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=101,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(101),
        )
        clean = torch.randn(1, 16, 21, 2, 2, dtype=torch.float32, requires_grad=True)
        clean_anchor = clean.detach().clone()
        before = self.handle.state_digest()
        direction = packet.action_cotangent

        route_flags = []

        def replay(*, role: str, adapter_enabled: bool) -> torch.Tensor:
            route_flags.append((role, adapter_enabled))
            displacement = (clean - clean_anchor).sum()
            sign = 1.0 if role == "action" else -1.0
            expected = (
                packet.action_measurement if role == "action" else packet.noop_measurement
            )
            return expected + sign * displacement * direction

        row = core._replay_score_cotangent_unsafe_for_test(
            packet,
            vjp_target="clean_latent",
            sp_rank=0,
            clean_latent=clean,
            replay_graph=replay,
            action_handle=self.handle,
        )
        self.assertEqual(row.vjp_target, "clean_latent")
        self.assertEqual(route_flags, [("action", False), ("noop", False)])
        self.assertTrue(bool(torch.isfinite(row.values).all()))
        self.assertGreater(row.value_norm, 0.0)
        self.assertEqual(before, self.handle.state_digest())
        self.assertTrue(
            all(
                torch.count_nonzero(parameter) == 0 and parameter.grad is None
                for _, parameter in self.handle.canonical_b_named_parameters()
            )
        )

    def test_clean_latent_serial_branches_retain_one_shared_x_sigma_prefix(self) -> None:
        action, noop = _measurement_pair(seed=23)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=102,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(102),
        )
        clean = torch.randn(1, 16, 21, 2, 2, dtype=torch.float32, requires_grad=True)
        noise = torch.randn_like(clean)
        # This prefix deliberately mirrors the production runner: it is built
        # once and consumed by both action and no-op native forwards.
        shared_x_sigma = (
            (1.0 - core.NATIVE_SIGMA) * clean + core.NATIVE_SIGMA * noise
        )
        anchor = shared_x_sigma.detach().clone()

        def replay(*, role: str, adapter_enabled: bool) -> torch.Tensor:
            self.assertFalse(adapter_enabled)
            expected = packet.action_measurement if role == "action" else packet.noop_measurement
            sign = 1.0 if role == "action" else -1.0
            return expected + sign * (shared_x_sigma - anchor).sum() * packet.action_cotangent

        row = core._replay_score_cotangent_unsafe_for_test(
            packet,
            vjp_target="clean_latent",
            sp_rank=0,
            clean_latent=clean,
            replay_graph=replay,
            action_handle=self.handle,
        )
        self.assertGreater(row.value_norm, 0.0)
        self.assertTrue(bool(torch.isfinite(row.values).all()))
        self.handle.assert_fixed_gauge()

    def test_direction_gate_precedes_serial_lora_b_vjp_and_state_stays_zero(self) -> None:
        action, noop = _measurement_pair(seed=29)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=202,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(202),
        )
        calls = []

        def replay(*, role: str, adapter_enabled: bool) -> torch.Tensor:
            calls.append((role, adapter_enabled))
            parameters = self.handle.canonical_b_named_parameters()
            scalar = torch.zeros((), dtype=torch.float32)
            for ordinal, (_name, parameter) in enumerate(parameters):
                scalar = scalar + parameter.sum() * float(ordinal + 1)
            expected = (
                packet.action_measurement if role == "action" else packet.noop_measurement
            )
            sign = 1.0 if role == "action" else -1.0
            return expected + sign * scalar * packet.action_cotangent

        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "direction gate"):
            core._replay_score_cotangent_unsafe_for_test(
                packet,
                vjp_target="lora_b",
                sp_rank=0,
                replay_graph=replay,
                action_handle=self.handle,
                decoded_direction_gate_passed=False,
            )
        self.assertEqual(calls, [])
        before = self.handle.state_digest()
        row = core._replay_score_cotangent_unsafe_for_test(
            packet,
            vjp_target="lora_b",
            sp_rank=0,
            replay_graph=replay,
            action_handle=self.handle,
            decoded_direction_gate_passed=True,
        )
        self.assertEqual(calls, [("action", True), ("noop", True)])
        self.assertEqual(tuple(row.values), core.CANONICAL_B_PARAMETER_NAMES)
        self.assertEqual(row.vjp_target, "lora_b")
        self.assertGreater(row.value_norm, 0.0)
        self.assertEqual(before, self.handle.state_digest())
        self.assertTrue(all(parameter.grad is None for _, parameter in self.handle.canonical_b_named_parameters()))

    def test_unified_vjp_target_selector_rejects_every_other_target(self) -> None:
        action, noop = _measurement_pair(seed=31)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=211,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(211),
        )
        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "vjp_target"):
            core._replay_score_cotangent_unsafe_for_test(
                packet,
                vjp_target="optimizer",
                sp_rank=0,
                replay_graph=lambda **_kwargs: action,
                action_handle=self.handle,
            )

    def test_failed_clean_replay_rolls_back_any_illegal_b_mutation(self) -> None:
        action, noop = _measurement_pair(seed=41)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=212,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(212),
        )
        clean = torch.randn(1, 16, 21, 2, 2, requires_grad=True)
        anchor = clean.detach().clone()
        first_b = self.handle.canonical_b_named_parameters()[0][1]

        def illegal(*, role: str, adapter_enabled: bool) -> torch.Tensor:
            self.assertFalse(adapter_enabled)
            if role == "action":
                with torch.no_grad():
                    first_b.fill_(1.0)
            expected = (
                packet.action_measurement if role == "action" else packet.noop_measurement
            )
            sign = 1.0 if role == "action" else -1.0
            return expected + sign * (clean - anchor).sum() * packet.action_cotangent

        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "rolled back"):
            core._replay_score_cotangent_unsafe_for_test(
                packet,
                vjp_target="clean_latent",
                sp_rank=0,
                clean_latent=clean,
                replay_graph=illegal,
                action_handle=self.handle,
            )
        self.assertEqual(int(torch.count_nonzero(first_b)), 0)
        self.handle.assert_fixed_gauge()

    def test_native_bridge_accepts_only_the_closed_core16_registration(self) -> None:
        donor = torch.zeros(1, 16, 21, 8, 10)
        refs = [torch.zeros(1, 16, 1, 8, 10) for _ in range(4)]
        pack = core.native.build_native_rv2v_pack(
            _NativePatchFake(),
            donor_video=donor,
            image_references=refs,
            noisy_target=donor.clone(),
        )
        with core.native_bridge._route_context(
            self.handle,
            transformer=self.model,
            branch=pack.video_image,
            sequence_parallel_rank=0,
            sequence_parallel_size=4,
            sigma_schedule_index=33,
            enabled=False,
        ):
            self.assertIsNotNone(core.pair_adapter.active_route())
        self.assertIsNone(core.pair_adapter.active_route())
        for branch in (pack.none, pack.video, pack.video_image):
            with core.native_bridge._route_context(
                self.handle,
                transformer=self.model,
                branch=branch,
                sequence_parallel_rank=0,
                sequence_parallel_size=4,
                sigma_schedule_index=33,
                enabled=True,
            ):
                self.assertEqual(core.pair_adapter.active_route().branch_name, branch.name)
        self.assertEqual(
            core.native_bridge._action_adapter_gate(
                self.handle, sigma_schedule_index=33
            ),
            core.pair_adapter.sigma_gate(33),
        )

        class Duck:
            transformer = self.model

            @staticmethod
            def route(_route):
                raise AssertionError("duck route must not run")

        with self.assertRaisesRegex(
            core.native_bridge.PairV5NativeBridgeError, "closed route registry"
        ):
            core.native_bridge._route_context(
                Duck(),
                transformer=self.model,
                branch=pack.video_image,
                sequence_parallel_rank=0,
                sequence_parallel_size=4,
                sigma_schedule_index=33,
                enabled=False,
            )

    def test_base_model_tamper_poisoned_and_never_restored(self) -> None:
        model = _InstallBernini().requires_grad_(False)
        handle = core.install_core16_fixed_a_b_only_action_lora(model)
        action, noop = _measurement_pair(seed=43)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=213,
            action_measurement=action,
            noop_measurement=noop,
            scorer=_QuadraticScorer(213),
        )
        clean = torch.randn(1, 16, 21, 2, 2, requires_grad=True)
        anchor = clean.detach().clone()
        first_a = handle.canonical_a_named_parameters()[0][1]
        base = model.patch_embedding.weight

        def illegal(*, role: str, adapter_enabled: bool) -> torch.Tensor:
            self.assertFalse(adapter_enabled)
            if role == "action":
                first_a.requires_grad_(True)
                first_a.grad = torch.ones_like(first_a)
                base.requires_grad_(True)
                base.grad = torch.ones_like(base)
                model.blocks[0].attn2.to_q = nn.Identity()
                leaked_route = core.pair_adapter.PairV5ActionRoute(
                    total_tokens=100,
                    condition_tokens=16,
                    sequence_parallel_rank=0,
                    sequence_parallel_size=4,
                    branch_name="VI",
                    sigma_schedule_index=33,
                    enabled=False,
                )
                core.pair_adapter._ACTIVE_ROUTE.set(leaked_route)
            expected = packet.action_measurement if role == "action" else packet.noop_measurement
            return expected + (clean - anchor).sum() * packet.action_cotangent

        try:
            with self.assertRaisesRegex(
                core.NativeRuntimeSealChangedError, "no restoration attempted"
            ):
                core._replay_score_cotangent_unsafe_for_test(
                    packet,
                    vjp_target="clean_latent",
                    sp_rank=0,
                    clean_latent=clean,
                    replay_graph=illegal,
                    action_handle=handle,
                )
            self.assertTrue(first_a.requires_grad)
            self.assertIsNotNone(first_a.grad)
            self.assertTrue(base.requires_grad)
            self.assertIsNotNone(base.grad)
            self.assertIsInstance(model.blocks[0].attn2.to_q, nn.Identity)
            self.assertIsNotNone(core.pair_adapter.active_route())
        finally:
            # Isolate the process-poisoning canary from later unit tests.  The
            # production replay path itself deliberately performed no repair.
            core.pair_adapter._ACTIVE_ROUTE.set(None)


class ScoreAndAggregationTests(unittest.TestCase):
    def test_detached_score_cotangent_divides_by_four_once(self) -> None:
        action, noop = _measurement_pair(seed=37)
        scorer = _QuadraticScorer(303)
        packet = core._score_cotangent_from_detached_measurement_unsafe_for_test(
            query_seed=303,
            action_measurement=action,
            noop_measurement=noop,
            scorer=scorer,
        )
        action_leaf = action.clone().requires_grad_(True)
        noop_leaf = noop.clone().requires_grad_(True)
        reference_score = scorer.forward_sketched_residual(
            action_leaf - noop_leaf, require_input_grad=True
        ).score
        reference = torch.autograd.grad(reference_score, action_leaf)[0]
        self.assertTrue(torch.allclose(packet.action_cotangent * 4.0, reference))
        self.assertTrue(torch.equal(packet.action_cotangent, -packet.noop_cotangent))
        receipt = packet.receipt()
        self.assertEqual(receipt["score_divisor"], 4)
        self.assertIsNone(receipt["post_sum_divisor"])

    def test_sp4_sum_does_not_divide_score_normalized_rows_again(self) -> None:
        rows = []
        for rank in range(4):
            value = torch.full((1, 16, 21, 2, 2), 0.25, dtype=torch.float32)
            rows.append(
                core._seal_rank_local_vjp_row(core.RankLocalVJPRow(
                    query_seed=404,
                    sp_rank=rank,
                    vjp_target="clean_latent",
                    values=value,
                    score_cotangent_receipt_digest=f"{rank + 1:064x}",
                    editor_packet_receipt_digest=f"{rank + 11:064x}",
                    global_cotangent_identity_digest="a" * 64,
                    value_sha256=core.tensor_sha256(value, label="toy row"),
                    value_norm=float(torch.linalg.vector_norm(value.double())),
                    replay_max_abs=0.0,
                    parameter_state_sha256="b" * 64,
                ))
            )
        summed = core._sum_rank_local_vjp_rows_unsafe_for_test(rows)
        self.assertTrue(torch.equal(summed.values, torch.ones_like(rows[0].values)))
        self.assertEqual(summed.receipt()["aggregation"], "SUM")
        self.assertFalse(summed.receipt()["divide_after_sum"])
        self.assertEqual(summed.receipt()["normalization_count"], 1)
        self.assertEqual(
            summed.rank_editor_packet_receipt_digests,
            tuple(row.editor_packet_receipt_digest for row in rows),
        )
        changed = replace(rows[-1], global_cotangent_identity_digest="c" * 64)
        object.__setattr__(changed, "_token", core._RANK_VJP_ROW_TOKEN)
        with self.assertRaisesRegex(core.NativeRV2VHiddenVJPError, "provenance"):
            core._sum_rank_local_vjp_rows_unsafe_for_test((*rows[:-1], changed))

    @staticmethod
    def _summed_b_row(seed: int, scalar: float) -> core.SP4SummedVJPRow:
        rows = []
        for rank in range(core.SP_SIZE):
            mapping = MappingProxyType(
                {
                    name: torch.full(
                        core.CANONICAL_B_SHAPE,
                        scalar / float(core.SP_SIZE),
                        dtype=torch.float32,
                    )
                    for name in core.CANONICAL_B_PARAMETER_NAMES
                }
            )
            norm = float(
                torch.sqrt(
                    sum(value.double().square().sum() for value in mapping.values())
                )
            )
            rows.append(
                core._seal_rank_local_vjp_row(
                    core.RankLocalVJPRow(
                        query_seed=seed,
                        sp_rank=rank,
                        vjp_target="lora_b",
                        values=mapping,
                        score_cotangent_receipt_digest=f"{rank + 21:064x}",
                        editor_packet_receipt_digest=f"{rank + 31:064x}",
                        global_cotangent_identity_digest="c" * 64,
                        value_sha256=core._named_tensor_sha256(
                            tuple(mapping.items()), label="toy rank B"
                        ),
                        value_norm=norm,
                        replay_max_abs=0.0,
                        parameter_state_sha256="d" * 64,
                    )
                )
            )
        return core._sum_rank_local_vjp_rows_unsafe_for_test(rows)

    def test_two_fixed_query_seed_action_rows_are_not_averaged_or_selected(self) -> None:
        first = self._summed_b_row(505, 1.0)
        second = self._summed_b_row(506, -2.0)
        result = core.build_two_query_seed_action_rows(
            (second, first), ordered_query_seeds=(505, 506)
        )
        self.assertEqual(result.ordered_query_seeds, (505, 506))
        self.assertIs(result.rows[0], first)
        self.assertIs(result.rows[1], second)
        self.assertTrue(
            torch.equal(
                result.rows[0].values[core.CANONICAL_B_PARAMETER_NAMES[0]],
                torch.ones(core.CANONICAL_B_SHAPE),
            )
        )
        self.assertTrue(
            torch.equal(
                result.rows[1].values[core.CANONICAL_B_PARAMETER_NAMES[0]],
                torch.full(core.CANONICAL_B_SHAPE, -2.0),
            )
        )
        receipt = result.receipt()
        self.assertFalse(receipt["seed_averaging"])
        self.assertFalse(receipt["seed_ranking_or_selection"])

    def test_any_missing_seed_returns_null_not_a_partial_row(self) -> None:
        only = self._summed_b_row(607, 1.0)
        self.assertIsNone(
            core.try_build_two_query_seed_action_rows(
                (only,), ordered_query_seeds=(607, 608)
            )
        )
        self.assertIsNone(
            core.try_build_two_query_seed_action_rows(
                (only, only), ordered_query_seeds=(607, 608)
            )
        )


class AuthoritativeSurfaceTests(unittest.TestCase):
    def test_public_surface_has_no_raw_scorer_callback_or_boolean_gate(self) -> None:
        self.assertNotIn("score_cotangent_from_detached_measurement", core.__all__)
        self.assertNotIn("_score_cotangent_from_detached_measurement_unsafe_for_test", core.__all__)
        score_parameters = inspect.signature(
            core.score_cotangent_from_authenticated_packets
        ).parameters
        replay_parameters = inspect.signature(core.replay_score_cotangent).parameters
        self.assertEqual(tuple(score_parameters), ("owner", "editor"))
        self.assertNotIn("scorer", score_parameters)
        self.assertNotIn("replay_graph", replay_parameters)
        self.assertNotIn("decoded_direction_gate_passed", replay_parameters)
        self.assertIn("direction_gate", replay_parameters)
        with self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "sealed owner/editor packets"
        ):
            core.score_cotangent_from_authenticated_packets(object(), object())
        canary = core.NativeSharedStepSP4ReplayRunner.canary_contract()
        self.assertEqual(canary["required_frame_count"], 81)
        self.assertEqual(canary["required_world_size"], 4)
        self.assertEqual(canary["shared_step_model_id"], "transformer_1")
        self.assertFalse(canary["real_auh_runtime_validated"])

    def test_production_runner_rejects_duck_dist_and_raw_runtime_tensors(self) -> None:
        parameters = inspect.signature(
            core.NativeSharedStepSP4ReplayRunner
        ).parameters
        self.assertIn("sp4_collective", parameters)
        self.assertIn("runtime_inputs", parameters)
        self.assertIn("checkpoint_content", parameters)
        self.assertTrue(
            callable(core.NativeSharedStepSP4ReplayRunner.sum_rank_local_vjp)
        )
        self.assertNotIn("sum_score_normalized_sp4_vjp_rows", core.__all__)
        for forbidden in (
            "dist_module",
            "source_latent",
            "image_references",
            "clean_latent",
            "initial_noise",
            "x_sigma",
            "action_condition",
            "noop_condition",
            "prompt_condition_binding",
            "checkpoint_content_receipt_digest",
        ):
            self.assertNotIn(forbidden, parameters)
        with self.assertRaises(core.NativeRV2VHiddenVJPError):
            core.authenticate_live_bernini_sp4_collective(
                parallel_state=SimpleNamespace(
                    ulysses_size=4,
                    ulysses_rank=0,
                    ulysses_group=_FakeSP4Dist(0),
                )
            )

    def test_paired_functionals_are_fixed_independent_rademacher_rows(self) -> None:
        torch.manual_seed(89)
        shape = (1, 21 * 2 * 2, core.PACKED_PREDICTION_DIM)
        base = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        noop = base * 0.7
        action = base * 1.1 + torch.linspace(-1.0, 1.0, shape[1]).reshape(
            1, shape[1], 1
        )
        video = base * 0.9
        image = base * 0.8
        none = base * 0.5
        observed = {}
        for functional_id in (
            core.WEAK_I_AXIS_FUNCTIONAL_ID,
            *core.FUNCTIONAL_PRESERVATION_SPECS,
        ):
            feature = core.paired_functional_preservation_feature(
                functional_id,
                patch_height=2,
                patch_width=2,
                noop_predicted_clean=noop,
                action_predicted_clean=(
                    action
                    if functional_id
                    not in (
                        "noop_predicted_clean_invariance",
                        "source_video_v_none_sensitivity",
                        core.WEAK_I_AXIS_FUNCTIONAL_ID,
                    )
                    else None
                ),
                video_predicted_clean=(
                    video
                    if functional_id == "source_video_v_none_sensitivity"
                    else None
                ),
                image_predicted_clean=(
                    image
                    if functional_id == core.WEAK_I_AXIS_FUNCTIONAL_ID
                    else None
                ),
                none_predicted_clean=(
                    none
                    if functional_id
                    in (
                        "source_video_v_none_sensitivity",
                        core.WEAK_I_AXIS_FUNCTIONAL_ID,
                    )
                    else None
                ),
            )
            scalars = tuple(
                core.fixed_rademacher_functional_scalar(feature, rademacher_seed=seed)
                for seed in core.PRESERVATION_RADEMACHER_SEEDS
            )
            self.assertTrue(all(value.requires_grad for value in scalars))
            self.assertNotEqual(float(scalars[0].detach()), float(scalars[1].detach()))
            observed[functional_id] = scalars
        self.assertEqual(
            set(observed),
            {core.WEAK_I_AXIS_FUNCTIONAL_ID, *core.FUNCTIONAL_PRESERVATION_SPECS},
        )
        contract = core.NativeSharedStepSP4ReplayRunner.functional_preservation_contract()
        self.assertFalse(contract["row_averaging"])
        self.assertEqual(contract["i_axis_role"], "weak_slab_only")
        self.assertEqual(contract["qp_infeasible_policy"], "byte_exact_zero_no_update")
        collector_parameters = inspect.signature(
            core.NativeSharedStepSP4ReplayRunner.collect_functional_preservation_cone
        ).parameters
        self.assertNotIn("rows", collector_parameters)
        self.assertNotIn("values", collector_parameters)
        self.assertNotIn("seal_paired_functional_preservation_cone", core.__all__)

    @unittest.skipIf(Ed25519PrivateKey is None, "cryptography Ed25519 unavailable")
    def _legacy_v1_authenticated_owner_and_gate_fixture(self) -> None:
        """Retained only as non-discovered migration documentation for v1."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cell_root = root / "published_dog"
            cell_root.mkdir()
            query_seed = 701
            action, noop = _measurement_pair(seed=71)
            feature = core.motion_cotangent.temporal_motion_quotient(
                action - noop, require_input_grad=False
            )
            unit = (feature / torch.linalg.vector_norm(feature)).detach().contiguous()
            unit_digest = core.owner_materializer.tensor_sha256(
                unit, label="fake published owner quotient"
            )
            quotient_path = cell_root / core.owner_materializer.QUOTIENT_FILENAME
            quotient_path.write_bytes(b"sealed-fake-safetensors")
            receipt_path = cell_root / core.owner_materializer.CELL_RECEIPT_FILENAME
            receipt_value = {"fake_validator_input": True}
            receipt_path.write_bytes(core.canonical_json_bytes(receipt_value) + b"\n")
            receipt_file_sha = core.file_sha256(receipt_path)
            source_sha = hashlib.sha256(b"source-video").hexdigest()
            action_sha = hashlib.sha256(b"dog turns right").hexdigest()
            noop_sha = hashlib.sha256(b"dog remains still").hexdigest()
            cell = SimpleNamespace(
                cell_id="dog",
                query_seeds=(query_seed, query_seed + 1),
                source_iid="dog-source-001",
                source_video_sha256=source_sha,
                action_caption_utf8_sha256=action_sha,
                noop_caption_utf8_sha256=noop_sha,
                action_family_id="turn-head-right",
            )

            class Registry:
                @staticmethod
                def cell(cell_id: str):
                    if cell_id != "dog":
                        raise ValueError("unknown test cell")
                    return cell

            authority = core.owner_materializer.AuthorizedOwnerInputs(
                registry=Registry(),
                registry_path=receipt_path,
                registry_file_sha256="1" * 64,
                owner_root=root,
                master_path=receipt_path,
                master_file_sha256="2" * 64,
                master_receipt_digest="3" * 64,
                child_receipts=MappingProxyType({}),
                child_paths=MappingProxyType({}),
                child_file_sha256=MappingProxyType({}),
                audit_sidecar_path=receipt_path,
                audit_sidecar_file_sha256="4" * 64,
                audit_sidecar_receipt_digest="5" * 64,
                audit_evidence_file_sha256="6" * 64,
                audit_public_key_file_sha256="7" * 64,
            )
            checked = {
                "cell_id": "dog",
                "receipt_digest": "8" * 64,
                "query_rows": [
                    {
                        "template": {
                            "query_seed": query_seed,
                            "unit_feature_digest": unit_digest,
                        }
                    },
                    {
                        "template": {
                            "query_seed": query_seed + 1,
                            "unit_feature_digest": unit_digest,
                        }
                    },
                ],
                "quotient_artifact": {
                    "file_sha256": core.file_sha256(quotient_path)
                },
                "model_binding": {"checkpoint": "test"},
                "owner_child_receipt_digest": "9" * 64,
                "external_full81_audit_sidecar_receipt_digest": "a" * 64,
            }

            class FakeOpen:
                def __init__(self, _path: str, **_kwargs: object) -> None:
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                @staticmethod
                def keys():
                    return [
                        f"{core.owner_materializer.TENSOR_KEY_PREFIX}{query_seed}",
                        f"{core.owner_materializer.TENSOR_KEY_PREFIX}{query_seed + 1}",
                    ]

                @staticmethod
                def get_tensor(_key: str):
                    return unit.clone()

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = FakeOpen
            modules = {"safetensors": fake_safetensors}
            validator = mock.patch.object(
                core.owner_materializer,
                "validate_published_cell_packet",
                return_value=checked,
            )
            with validator, mock.patch.dict(sys.modules, modules):
                owner = core.load_authenticated_owner_quotient_packet(
                    authority=authority,
                    cell_root=cell_root,
                    receipt_path=receipt_path,
                    expected_receipt_file_sha256=receipt_file_sha,
                    query_seed=query_seed,
                )
                runtime = torch.tensor([1.0], dtype=torch.float32)
                runtime_tensors = MappingProxyType({"x_sigma": runtime})
                runtime_bindings = MappingProxyType(
                    {
                        "x_sigma": core._tensor_runtime_binding(
                            runtime, label="test runtime x_sigma"
                        )
                    }
                )
                editor = core.EditorSameStatePromptPacket(
                    cell_id="dog",
                    query_seed=query_seed,
                    sp_rank=0,
                    branch_name="VI",
                    source_iid=cell.source_iid,
                    source_video_sha256=source_sha,
                    action_prompt_sha256=action_sha,
                    noop_prompt_sha256=noop_sha,
                    action_measurement=action,
                    noop_measurement=noop,
                    local_action_measurement=action,
                    local_noop_measurement=noop,
                    bindings=MappingProxyType(
                        {
                            "native_schedule_index": 33,
                            "native_timestep": 516,
                            "native_sigma": core.NATIVE_SIGMA,
                            "sp_size": 4,
                            "sp_rank": 0,
                            "model_proof_runtime_owned": True,
                            "editor_query_seed": query_seed,
                            "official_cpu_generator_gaussian": True,
                            "prompt_condition_binding_digest": "a" * 64,
                            "sealed_action_measurement_sha256": core.tensor_sha256(
                                action, label="test sealed action"
                            ),
                            "sealed_noop_measurement_sha256": core.tensor_sha256(
                                noop, label="test sealed no-op"
                            ),
                            "sealed_local_action_measurement_sha256": core.tensor_sha256(
                                action, label="test sealed local action"
                            ),
                            "sealed_local_noop_measurement_sha256": core.tensor_sha256(
                                noop, label="test sealed local no-op"
                            ),
                        }
                    ),
                    _runtime_tensors=runtime_tensors,
                    _runtime_tensor_bindings=runtime_bindings,
                    _runtime_owner_digest=owner.receipt()["digest"],
                )
                object.__setattr__(editor, "_token", core._EDITOR_PACKET_TOKEN)
                score = core.score_cotangent_from_authenticated_packets(owner, editor)
                self.assertEqual(score.score_divisor, 4)
                self.assertTrue(score.scorer_id.startswith("authenticated-owner:"))
                self.assertTrue(torch.equal(score.action_cotangent, -score.noop_cotangent))

                clean_receipt = {
                    "schema_version": core.SP4_ROW_SCHEMA_VERSION,
                    "query_seed": query_seed,
                    "vjp_target": "clean_latent",
                    "score_cotangent_receipt_digest": score.receipt()["digest"],
                    "value_sha256": "c" * 64,
                    "digest": "b" * 64,
                }
                zero_parameters = tuple(
                    (
                        name,
                        torch.zeros(core.CANONICAL_B_SHAPE, dtype=torch.float32),
                    )
                    for name in core.CANONICAL_B_PARAMETER_NAMES
                )
                layout = qp.FixedParameterLayout.from_ordered_parameters(
                    zero_parameters
                )
                required_functionals = (
                    core.WEAK_I_AXIS_FUNCTIONAL_ID,
                    *core.FUNCTIONAL_PRESERVATION_SPECS.keys(),
                )
                functional_rows = []
                ordinal = 0
                for functional_id in required_functionals:
                    for rademacher_seed in core.PRESERVATION_RADEMACHER_SEEDS:
                        ordinal += 1
                        if functional_id == core.WEAK_I_AXIS_FUNCTIONAL_ID:
                            family, contrast, weak, bound = (
                                "identity",
                                "I_axis_weak_slab",
                                True,
                                0.4,
                            )
                        else:
                            family, contrast = core.FUNCTIONAL_PRESERVATION_SPECS[
                                functional_id
                            ]
                            weak, bound = False, 0.1
                        tensor = torch.full(
                            core.CANONICAL_B_SHAPE,
                            float(ordinal) / 1000.0,
                            dtype=torch.float32,
                        )
                        mapping = MappingProxyType(
                            {name: tensor for name in core.CANONICAL_B_PARAMETER_NAMES}
                        )
                        norm = float(
                            torch.sqrt(
                                sum(
                                    value.double().square().sum()
                                    for value in mapping.values()
                                )
                            )
                        )
                        row = core.FunctionalPreservationVJPRow(
                            row_id=f"dog:{functional_id}:r{rademacher_seed}",
                            functional_id=functional_id,
                            qp_family=family,
                            native_contrast=contrast,
                            rademacher_seed=rademacher_seed,
                            weak_i_axis_slab=weak,
                            values=mapping,
                            maximum_absolute_dot=bound,
                            same_state_binding_digest="2" * 64,
                            clean_vjp_receipt_digest=clean_receipt["digest"],
                            checkpoint_content_receipt_digest="1" * 64,
                            parameter_state_sha256=layout.parameter_state_sha256,
                            sp4_editor_packet_receipt_digests=(
                                editor.receipt()["digest"],
                                "3" * 64,
                                "4" * 64,
                                "5" * 64,
                            ),
                            sp4_rank_vjp_receipt_digests=(
                                "6" * 64,
                                "7" * 64,
                                "8" * 64,
                                "9" * 64,
                            ),
                            value_sha256=core._named_tensor_sha256(
                                tuple(mapping.items()), label="test functional row"
                            ),
                            value_norm=norm,
                        )
                        object.__setattr__(row, "_token", core._FUNCTIONAL_ROW_TOKEN)
                        functional_rows.append(row)
                cone = core.seal_paired_functional_preservation_cone(
                    owner=owner,
                    editor=editor,
                    score_packet=score,
                    clean_vjp_receipt=clean_receipt,
                    rows=functional_rows,
                )
                qp_rows = cone.to_qp_rows(layout)
                self.assertEqual(len(qp_rows), len(functional_rows))
                self.assertEqual(
                    {row.family for row in qp_rows}, set(qp.PRESERVATION_FAMILIES)
                )
                self.assertFalse(cone.receipt()["row_averaging"])
                self.assertTrue(cone.receipt()["qp_infeasible_returns_exact_zero"])
                with self.assertRaisesRegex(
                    core.NativeRV2VHiddenVJPError, "every fixed Rademacher row"
                ):
                    core.seal_paired_functional_preservation_cone(
                        owner=owner,
                        editor=editor,
                        score_packet=score,
                        clean_vjp_receipt=clean_receipt,
                        rows=functional_rows[:-1],
                    )
                weakened = replace(functional_rows[0], maximum_absolute_dot=0.01)
                object.__setattr__(weakened, "_token", core._FUNCTIONAL_ROW_TOKEN)
                bad_rows = [weakened, *functional_rows[1:]]
                with self.assertRaisesRegex(
                    core.NativeRV2VHiddenVJPError, "weak-I policy"
                ):
                    core.seal_paired_functional_preservation_cone(
                        owner=owner,
                        editor=editor,
                        score_packet=score,
                        clean_vjp_receipt=clean_receipt,
                        rows=bad_rows,
                    )

                artifacts = []
                for role in (
                    "base_exact81_mp4",
                    "plus_exact81_mp4",
                    "minus_exact81_mp4",
                ):
                    path = root / f"{role}.mp4"
                    path.write_bytes(role.encode("ascii"))
                    artifacts.append(
                        {
                            "role": role,
                            "path": str(path),
                            "file_sha256": core.file_sha256(path),
                            "frame_count": 81,
                        }
                    )
                private = Ed25519PrivateKey.generate()
                public_bytes = private.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                key_path = root / "direction-gate-ed25519.pem"
                key_path.write_bytes(public_bytes)
                gate_unsigned = {
                    "schema_version": core.DIRECTION_GATE_SCHEMA_VERSION,
                    "cell_id": "dog",
                    "query_seed": query_seed,
                    "decode_seed": 9001,
                    "source_iid": cell.source_iid,
                    "source_video_sha256": source_sha,
                    "action_prompt_sha256": action_sha,
                    "noop_prompt_sha256": noop_sha,
                    "owner_packet_receipt_digest": owner.receipt()["digest"],
                    "editor_packet_receipt_digest": editor.receipt()["digest"],
                    "clean_vjp_binding": {
                        "receipt_digest": clean_receipt["digest"],
                        "value_sha256": clean_receipt["value_sha256"],
                        "score_cotangent_receipt_digest": score.receipt()["digest"],
                    },
                    "symmetric_pair": {
                        "arm_order": ["plus", "minus"],
                        "relative_l2_dose": 0.01,
                        "clean_latent_sha256": "d" * 64,
                        "direction_tensor_sha256": "e" * 64,
                        "plus_latent_sha256": "f" * 64,
                        "minus_latent_sha256": "0" * 64,
                        "same_query_and_decode_seed": True,
                        "same_decode_configuration": True,
                        "passed": True,
                    },
                    "artifacts": artifacts,
                    "decision": {
                        "decoded_direction_gate_passed": True,
                        "plus_improves_requested_action": True,
                        "minus_does_not_match_or_exceed_plus": True,
                        "identity_and_unedited_content_acceptable": True,
                    },
                    "authority_public_key_sha256": core.file_sha256(key_path),
                    "authority_signature_scheme": core.DIRECTION_GATE_SIGNATURE_SCHEME,
                }
                signed = {
                    **gate_unsigned,
                    "receipt_digest": core.object_sha256(gate_unsigned),
                }
                signature = private.sign(core.canonical_json_bytes(signed))
                gate_value = {
                    **signed,
                    "authority_signature_ed25519_base64": base64.b64encode(signature).decode("ascii"),
                }
                gate_path = root / "signed-exact81-direction-gate.json"
                gate_path.write_bytes(core.canonical_json_bytes(gate_value) + b"\n")
                gate = core.load_validated_exact81_direction_gate(
                    gate_path=gate_path,
                    expected_gate_file_sha256=core.file_sha256(gate_path),
                    public_key_path=key_path,
                    expected_public_key_file_sha256=core.file_sha256(key_path),
                    artifact_root=root,
                    owner=owner,
                    editor=editor,
                    score_packet=score,
                    clean_vjp_receipt=clean_receipt,
                )
                self.assertFalse(gate.receipt()["naked_boolean_gate"])
                Path(artifacts[1]["path"]).write_bytes(b"tampered-plus")
                with self.assertRaisesRegex(
                    core.NativeRV2VHiddenVJPError, "artifact bytes changed"
                ):
                    gate.assert_live(
                        owner=owner,
                        editor=editor,
                        score_packet=score,
                        clean_vjp_receipt=clean_receipt,
                    )
                Path(artifacts[1]["path"]).write_bytes(
                    artifacts[1]["role"].encode("ascii")
                )
                changed_unsigned = dict(gate_unsigned)
                changed_decision = dict(changed_unsigned["decision"])
                changed_decision["decoded_direction_gate_passed"] = False
                changed_unsigned["decision"] = changed_decision
                changed_signed = {
                    **changed_unsigned,
                    "receipt_digest": core.object_sha256(changed_unsigned),
                }
                changed_gate = {
                    **changed_signed,
                    # Deliberately retain the signature over the original
                    # canonical signed payload.
                    "authority_signature_ed25519_base64": gate_value[
                        "authority_signature_ed25519_base64"
                    ],
                }
                gate_path.write_bytes(core.canonical_json_bytes(changed_gate) + b"\n")
                with self.assertRaisesRegex(
                    core.NativeRV2VHiddenVJPError, "signature verification failed"
                ):
                    core.load_validated_exact81_direction_gate(
                        gate_path=gate_path,
                        expected_gate_file_sha256=core.file_sha256(gate_path),
                        public_key_path=key_path,
                        expected_public_key_file_sha256=core.file_sha256(key_path),
                        artifact_root=root,
                        owner=owner,
                        editor=editor,
                        score_packet=score,
                        clean_vjp_receipt=clean_receipt,
                    )


class LiveAuthorityAndRuntimeSealTests(unittest.TestCase):
    @unittest.skipIf(
        Ed25519PrivateKey is None, "cryptography Ed25519 unavailable"
    )
    def test_owner_assert_live_reloads_evidence_and_every_child(self) -> None:
        test_root = Path(__file__).resolve().parent
        if str(test_root) not in sys.path:
            sys.path.insert(0, str(test_root))
        from test_materialize_self_imagined_owner_core2_v1 import (  # noqa: PLC0415
            SignedOwnerFixture,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture = SignedOwnerFixture(root)
            cell = fixture.registry_value.cell("dog")
            query_seed = cell.query_seeds[0]
            unit = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)
            unit = (unit / torch.linalg.vector_norm(unit)).contiguous()
            unit_digest = core.owner_materializer.tensor_sha256(
                unit, label="test owner quotient"
            )
            cell_root = root / "published-dog"
            cell_root.mkdir()
            quotient_path = cell_root / core.owner_materializer.QUOTIENT_FILENAME
            quotient_path.write_bytes(b"test-safetensors-container")
            receipt_path = cell_root / core.owner_materializer.CELL_RECEIPT_FILENAME
            receipt_path.write_bytes(core.canonical_json_bytes({"test": True}) + b"\n")
            checked = {
                "cell_id": "dog",
                "receipt_digest": "8" * 64,
                "query_rows": [
                    {
                        "template": {
                            "query_seed": seed,
                            "unit_feature_digest": unit_digest,
                        }
                    }
                    for seed in cell.query_seeds
                ],
                "quotient_artifact": {
                    "file_sha256": core.file_sha256(quotient_path)
                },
                "model_binding": {"checkpoint": "test"},
                "owner_child_receipt_digest": "9" * 64,
                "external_full81_audit_sidecar_receipt_digest": "a" * 64,
            }

            class FakeOpen:
                def __init__(self, _path: str, **_kwargs: object) -> None:
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                @staticmethod
                def keys():
                    return [
                        f"{core.owner_materializer.TENSOR_KEY_PREFIX}{seed}"
                        for seed in cell.query_seeds
                    ]

                @staticmethod
                def get_tensor(_key: str):
                    return unit.clone()

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = FakeOpen
            with mock.patch.object(
                core.owner_materializer,
                "validate_published_cell_packet",
                return_value=checked,
            ), mock.patch.dict(sys.modules, {"safetensors": fake_safetensors}):
                owner = core.load_authenticated_owner_quotient_packet(
                    **fixture.kwargs(),
                    cell_root=cell_root,
                    receipt_path=receipt_path,
                    expected_receipt_file_sha256=core.file_sha256(receipt_path),
                    query_seed=query_seed,
                )
                owner.assert_live()

                evidence_bytes = fixture.evidence.read_bytes()
                fixture.evidence.write_bytes(evidence_bytes + b"tampered\n")
                with self.assertRaises(core.NativeRV2VHiddenVJPError):
                    owner.assert_live()
                fixture.evidence.write_bytes(evidence_bytes)
                owner.assert_live()

                child = fixture.child_paths["human"]
                child_bytes = child.read_bytes()
                child.write_bytes(child_bytes + b"tampered\n")
                with self.assertRaises(core.NativeRV2VHiddenVJPError):
                    owner.assert_live()

    def test_checkpoint_manifest_rehashes_bytes_and_file_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            first = checkpoint / "config.json"
            second = checkpoint / "weights.bin"
            first.write_bytes(b"config-v1")
            second.write_bytes(b"weights-v1")
            manifest = root / "checkpoint.sha256"
            manifest.write_text(
                "\n".join(
                    (
                        f"{core.file_sha256(first)}  ./config.json",
                        f"{core.file_sha256(second)}  ./weights.bin",
                    )
                )
                + "\n",
                encoding="ascii",
            )
            packet = core.load_validated_checkpoint_content_manifest(
                checkpoint_root=checkpoint,
                content_manifest_path=manifest,
                expected_manifest_sha256=core.file_sha256(manifest),
                expected_file_count=2,
            )
            packet.assert_live()
            second.write_bytes(b"weights-v2")
            with self.assertRaisesRegex(
                core.NativeRV2VHiddenVJPError, "content hash differs"
            ):
                packet.assert_live()
            second.write_bytes(b"weights-v1")
            packet.assert_live()
            (checkpoint / "unlisted.bin").write_bytes(b"unlisted")
            with self.assertRaisesRegex(
                core.NativeRV2VHiddenVJPError, "file closure differs"
            ):
                packet.assert_live()

    @staticmethod
    def _minimal_live_runner():
        transformer = nn.Sequential(nn.Linear(3, 3), nn.ReLU())

        class TinyDiffusion(nn.Module):
            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.transformer = model

            def shared_step(self, **_kwargs: object) -> torch.Tensor:
                return torch.zeros(1)

        diffusion = TinyDiffusion(transformer)
        observer_handle = object()
        spatial_sketch = torch.ones(2, dtype=torch.float32)
        observer = SimpleNamespace(
            transformer=transformer,
            block=transformer[0],
            _handle=observer_handle,
            _pending=None,
            spatial_sketch=spatial_sketch,
        )
        action_handle = SimpleNamespace(transformer=transformer)
        collective = SimpleNamespace(sp_rank=0, assert_live=lambda: None)
        checkpoint = SimpleNamespace(assert_live=lambda: None)
        owner = object()
        runtime_inputs = SimpleNamespace(
            assert_live=lambda _owner, _checkpoint: None
        )
        runner = object.__new__(core.NativeSharedStepSP4ReplayRunner)
        runner.diffusion = diffusion
        runner.transformer = transformer
        runner.action_handle = action_handle
        runner.observer = observer
        runner.collective = collective
        runner.checkpoint_content = checkpoint
        runner.owner = owner
        runner.runtime_inputs = runtime_inputs
        runner.sp_rank = 0
        runner._diffusion_object = diffusion
        runner._transformer_object = transformer
        runner._action_handle_object = action_handle
        runner._observer_object = observer
        runner._collective_object = collective
        runner._checkpoint_content_object = checkpoint
        runner._owner_object = owner
        runner._runtime_inputs_object = runtime_inputs
        runner._observer_runtime_binding = MappingProxyType(
            {
                "observer_object_id": id(observer),
                "observer_transformer_object_id": id(transformer),
                "observer_block_object_id": id(observer.block),
                "observer_handle_object_id": id(observer_handle),
                "observer_spatial_sketch": dict(
                    core._tensor_runtime_binding(
                        spatial_sketch, label="minimal runner spatial sketch"
                    )
                ),
                "observer_spatial_sketch_metadata": dict(
                    core._tensor_runtime_metadata_binding(
                        spatial_sketch,
                        label="minimal runner spatial sketch metadata",
                    )
                ),
            }
        )
        runner._runtime_tensors = MappingProxyType({})
        runner._runtime_bindings = MappingProxyType({})
        runner._runtime_metadata_bindings = MappingProxyType({})
        runner._adapter_b_ids = frozenset()
        runner._model_runtime_seal = core._complete_model_runtime_receipt(
            diffusion=diffusion,
            transformer=transformer,
            adapter_b_ids=frozenset(),
        )
        runner._model_runtime_metadata_seal = core._complete_model_runtime_receipt(
            diffusion=diffusion,
            transformer=transformer,
            adapter_b_ids=frozenset(),
            _hash_tensor_bytes=False,
        )
        return runner

    def test_runner_full_model_seal_fails_without_restoring_tamper(self) -> None:
        runner = self._minimal_live_runner()
        runner._assert_runtime_live()
        parameter = next(runner.transformer.parameters())
        with torch.no_grad():
            parameter.add_(1.0)
        tampered = parameter.detach().clone()
        with self.assertRaises(core.NativeRuntimeSealChangedError):
            runner._assert_runtime_live()
        self.assertTrue(torch.equal(parameter.detach(), tampered))

        replaced_runner = self._minimal_live_runner()
        replacement = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        replaced_runner.diffusion = replacement
        with self.assertRaises(core.NativeRuntimeSealChangedError):
            replaced_runner._assert_runtime_live()
        self.assertIs(replaced_runner.diffusion, replacement)

        priority_runner = self._minimal_live_runner()
        priority_runner.checkpoint_content.assert_live = lambda: (_ for _ in ()).throw(
            core.NativeRV2VHiddenVJPError("simultaneous checkpoint failure")
        )
        priority_parameter = next(priority_runner.transformer.parameters())
        with torch.no_grad():
            priority_parameter.mul_(2.0)
        with self.assertRaises(core.NativeRuntimeSealChangedError):
            priority_runner._assert_runtime_live()

    def test_runner_full_model_seal_rejects_live_forward_hook_tamper(self) -> None:
        runner = self._minimal_live_runner()
        block = runner.transformer[0]
        runner._assert_runtime_live()

        handle = block.register_forward_hook(lambda _module, _inputs, output: output)
        hook_id = handle.id
        try:
            with self.assertRaises(core.NativeRuntimeSealChangedError):
                runner._assert_runtime_live()
            self.assertIn(hook_id, block._forward_hooks)
        finally:
            handle.remove()

    def test_runner_cheap_seal_uses_metadata_and_terminal_uses_full_bytes(self) -> None:
        runner = self._minimal_live_runner()
        with mock.patch.object(
            core,
            "_complete_model_runtime_receipt",
            wraps=core._complete_model_runtime_receipt,
        ) as sealed:
            runner._assert_runtime_live(deep=False)
            self.assertEqual(sealed.call_count, 1)
            self.assertFalse(sealed.call_args.kwargs["_hash_tensor_bytes"])

            runner._assert_runtime_live(deep=True)
            self.assertEqual(sealed.call_count, 3)
            modes = [
                call.kwargs.get("_hash_tensor_bytes", True)
                for call in sealed.call_args_list
            ]
            self.assertEqual(modes, [False, False, True])

    def test_vendor_condition_casts_only_call_edge_and_keeps_signed_fp32(self) -> None:
        runner = object.__new__(core.NativeSharedStepSP4ReplayRunner)
        runner.transformer = SimpleNamespace(
            patch_embedding=nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        )
        runner.x_sigma = torch.zeros(1, dtype=torch.float32)
        runner.action_condition = torch.randn(1, 3, 4, dtype=torch.float32)
        runner.noop_condition = torch.randn(1, 3, 4, dtype=torch.float32)
        action_before = runner.action_condition.clone()
        noop_before = runner.noop_condition.clone()

        action_vendor = runner._vendor_condition("action")
        noop_vendor = runner._vendor_condition("noop")

        self.assertEqual(action_vendor.dtype, torch.bfloat16)
        self.assertEqual(noop_vendor.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(runner.action_condition, action_before))
        self.assertTrue(torch.equal(runner.noop_condition, noop_before))
        self.assertEqual(runner.action_condition.dtype, torch.float32)
        self.assertEqual(runner.noop_condition.dtype, torch.float32)

    def test_editor_measurement_matches_replay_grad_mode_then_detaches(self) -> None:
        runner = object.__new__(core.NativeSharedStepSP4ReplayRunner)
        source = torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=True)
        observed = []

        def forward_local(*, role: str, adapter_enabled: bool, detach: bool):
            observed.append(
                (
                    role,
                    adapter_enabled,
                    detach,
                    torch.is_grad_enabled(),
                    torch.is_inference_mode_enabled(),
                )
            )
            return source * 3.0

        class Collective:
            @staticmethod
            def all_reduce_sum(value: torch.Tensor) -> None:
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)

        runner._forward_local = forward_local
        runner.collective = Collective()
        with torch.no_grad():
            local, global_value = runner._measure_role("action")

        self.assertEqual(observed, [("action", False, False, True, False)])
        self.assertTrue(torch.equal(local, source.detach() * 3.0))
        self.assertTrue(torch.equal(global_value, local))
        self.assertFalse(local.requires_grad)
        self.assertIsNone(local.grad_fn)
        self.assertFalse(global_value.requires_grad)
        self.assertIsNone(global_value.grad_fn)

    def test_clean_freeze_skips_adapter_rollback_after_poison(self) -> None:
        model = _InstallBernini().requires_grad_(False)
        handle = core.install_core16_fixed_a_b_only_action_lora(model)
        adapter_ids = frozenset(
            id(parameter) for _, parameter in handle.trainable_named_parameters()
        )
        sealed = core._base_transformer_runtime_receipt(
            transformer=model, adapter_parameter_ids=adapter_ids
        )

        def poison_check() -> None:
            observed = core._base_transformer_runtime_receipt(
                transformer=model, adapter_parameter_ids=adapter_ids
            )
            if dict(observed) != dict(sealed):
                raise core.NativeRuntimeSealChangedError("test process poisoned")

        first_b = handle.canonical_b_named_parameters()[0][1]
        base = model.patch_embedding.weight
        base_before = base.detach().clone()
        with self.assertRaises(core.NativeRuntimeSealChangedError):
            with handle.frozen_b_for_clean_vjp(poison_check=poison_check):
                self.assertTrue(first_b.requires_grad)
                with torch.no_grad():
                    base.add_(1.0)
                raise RuntimeError("forward failed after mutating base")
        self.assertTrue(first_b.requires_grad)
        self.assertTrue(torch.equal(base.detach(), base_before + 1.0))

    def test_clean_vjp_scope_preserves_authenticated_b_runtime_metadata(self) -> None:
        model = _InstallBernini().requires_grad_(False)
        handle = core.install_core16_fixed_a_b_only_action_lora(model)

        class TinyDiffusion(nn.Module):
            def __init__(self, transformer: nn.Module) -> None:
                super().__init__()
                self.transformer = transformer

            def shared_step(self, **_kwargs: object) -> torch.Tensor:
                return torch.zeros(1)

        diffusion = TinyDiffusion(model)
        b_ids = frozenset(
            id(parameter) for _, parameter in handle.canonical_b_named_parameters()
        )
        before = core._complete_model_runtime_receipt(
            diffusion=diffusion,
            transformer=model,
            adapter_b_ids=b_ids,
            _hash_tensor_bytes=False,
        )

        def assert_metadata_live() -> None:
            observed = core._complete_model_runtime_receipt(
                diffusion=diffusion,
                transformer=model,
                adapter_b_ids=b_ids,
                _hash_tensor_bytes=False,
            )
            self.assertEqual(dict(observed), dict(before))

        with handle.frozen_b_for_clean_vjp(
            poison_check=assert_metadata_live
        ):
            assert_metadata_live()
            self.assertTrue(
                all(
                    parameter.requires_grad and parameter.grad is None
                    for _, parameter in handle.canonical_b_named_parameters()
                )
            )

        assert_metadata_live()
        handle.assert_fixed_gauge()


class Exact81AndFunctionalExploitTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("av") and importlib.util.find_spec("imageio_ffmpeg"),
        "pinned PyAV/imageio_ffmpeg unavailable",
    )
    def test_exact81_probe_decodes_real_mp4_and_rejects_json_disguise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            video = root / "exact81.mp4"
            ffmpeg, _, _, _, _, _, _, _ = core._resolve_imageio_bundled_ffmpeg()
            subprocess.run(
                (
                    str(ffmpeg),
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=16x16:r=25",
                    "-frames:v",
                    "81",
                    "-c:v",
                    "mpeg4",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            receipt = core._probe_decode_exact81(video)
            self.assertEqual(set(receipt), core.EXACT81_MEDIA_PROBE_FIELDS)
            self.assertEqual(receipt["pyav_decoded_frame_count"], 81)
            self.assertEqual(
                receipt["bundled_ffmpeg_framemd5_frame_count"], 81
            )
            self.assertTrue(receipt["pyav_exact_25fps_pts_cadence"])
            self.assertEqual(
                receipt["bundled_ffmpeg_executable_sha256"],
                core.PINNED_BUNDLED_FFMPEG_SHA256,
            )
            disguised = root / "disguised.mp4"
            disguised.write_text(
                json.dumps({"frame_count": 81, "decoded": True}),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                core.NativeRV2VHiddenVJPError, "portable media probe failed"
            ):
                core._probe_decode_exact81(disguised)

    def test_bundled_ffmpeg_rejects_caller_environment_injection(self) -> None:
        with mock.patch.dict(
            os.environ, {"IMAGEIO_FFMPEG_EXE": "/bin/true"}, clear=False
        ), self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "caller injection is forbidden"
        ):
            core._resolve_imageio_bundled_ffmpeg()

    def test_symmetric_latent_proof_recomputes_q_plus_and_minus(self) -> None:
        clean = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        base = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        direction = (clean / torch.linalg.vector_norm(clean)).contiguous()
        dose = 0.125
        scale = torch.tensor(dose, dtype=torch.float32) * torch.linalg.vector_norm(base)
        plus = (base + scale * direction).contiguous()
        minus = (base - scale * direction).contiguous()
        core._validate_symmetric_gate_latents(
            clean_vjp=clean,
            runtime_clean_latent=base,
            direction=direction,
            base=base,
            plus=plus,
            minus=minus,
            relative_l2_dose=dose,
        )
        forged_plus = plus.clone()
        forged_plus[0, 0] += 1.0e-3
        with self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "numeric proof differs"
        ):
            core._validate_symmetric_gate_latents(
                clean_vjp=clean,
                runtime_clean_latent=base,
                direction=direction,
                base=base,
                plus=forged_plus,
                minus=minus,
                relative_l2_dose=dose,
            )

    def test_stolen_functional_token_cannot_relabel_a_row(self) -> None:
        value = torch.ones(core.CANONICAL_B_SHAPE, dtype=torch.float32)
        mapping = MappingProxyType(
            {name: value for name in core.CANONICAL_B_PARAMETER_NAMES}
        )
        value_sha = core._named_tensor_sha256(
            tuple(mapping.items()), label="functional relabel exploit"
        )
        value_norm = float(
            torch.sqrt(
                sum(item.double().square().sum() for item in mapping.values())
            ).item()
        )
        seed = core.PRESERVATION_RADEMACHER_SEEDS[0]
        functional_id = "noop_predicted_clean_invariance"
        row = core.FunctionalPreservationVJPRow(
            row_id=f"cell:{functional_id}:r{seed}",
            functional_id=functional_id,
            qp_family="noop",
            native_contrast="VI_noop",
            rademacher_seed=seed,
            weak_i_axis_slab=False,
            values=mapping,
            maximum_absolute_dot=0.0,
            same_state_binding_digest="1" * 64,
            clean_vjp_receipt_digest="2" * 64,
            checkpoint_content_receipt_digest="3" * 64,
            parameter_state_sha256="4" * 64,
            sp4_editor_packet_receipt_digests=tuple(
                f"{index + 5:064x}" for index in range(core.SP_SIZE)
            ),
            sp4_rank_vjp_receipt_digests=tuple(
                f"{index + 9:064x}" for index in range(core.SP_SIZE)
            ),
            value_sha256=value_sha,
            value_norm=value_norm,
        )
        object.__setattr__(row, "_token", core._FUNCTIONAL_ROW_TOKEN)
        row.assert_live()
        relabeled_id = "action_noop_spatial_dc"
        relabeled = replace(
            row,
            row_id=f"cell:{relabeled_id}:r{seed}",
            functional_id=relabeled_id,
        )
        object.__setattr__(relabeled, "_token", core._FUNCTIONAL_ROW_TOKEN)
        with self.assertRaisesRegex(
            core.NativeRV2VHiddenVJPError, "live binding changed"
        ):
            relabeled.assert_live()


if __name__ == "__main__":
    unittest.main()

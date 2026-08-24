from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
from torch import nn


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import ramp_target_row_lora as route  # noqa: E402


class _Attention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.to_q = nn.Linear(width, width, bias=True)


class _Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attn1 = _Attention(width)


class _TinyTransformer(nn.Module):
    def __init__(self, width: int = 8, blocks: int = 2):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            16, width, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.blocks = nn.ModuleList([_Block(width) for _ in range(blocks)])


def _identity_rgb_coordinates() -> torch.Tensor:
    return torch.arange(81, dtype=torch.float64).contiguous()


def _program_patches() -> torch.Tensor:
    transport = route.latent_phase_transport(_identity_rgb_coordinates())
    return route.oracle_program_patches(transport)


def _packed_patches(layout: route.TokenRoleLayout) -> torch.Tensor:
    source = torch.randn(layout.source_tokens, 16, 1, 2, 2, dtype=torch.float32)
    target = torch.randn(layout.target_tokens, 16, 1, 2, 2, dtype=torch.float32)
    return torch.cat([source, _program_patches(), target], dim=0).contiguous()


class RAMPProgramEncodingTests(unittest.TestCase):
    def test_phase_transport_and_patch_roundtrip_are_exact_for_identity(self):
        matrix = route.latent_phase_transport(_identity_rgb_coordinates())
        self.assertTrue(torch.equal(matrix, torch.eye(21, dtype=torch.float32)))
        patches = route.oracle_program_patches(matrix)
        self.assertEqual(tuple(patches.shape), (21, 16, 1, 2, 2))
        self.assertTrue(torch.equal(route.recover_oracle_transport(patches), matrix))

    def test_fractional_phase_rows_are_row_stochastic(self):
        coordinate = torch.linspace(0.0, 80.0, 81, dtype=torch.float64).pow(0.9)
        coordinate = (coordinate / coordinate[-1] * 80.0).contiguous()
        matrix = route.latent_phase_transport(coordinate)
        self.assertTrue(torch.allclose(matrix.sum(-1), torch.ones(21), atol=1e-6, rtol=0))
        self.assertTrue(bool((matrix >= 0).all().item()))

    def test_program_patch_tail_is_fail_closed(self):
        patches = _program_patches()
        patches[0].reshape(-1)[22] = 1.0
        with self.assertRaisesRegex(route.RAMPRouteError, "tail"):
            route.recover_oracle_transport(patches)


class RAMPRouteLayoutTests(unittest.TestCase):
    def test_sp4_selector_matches_append_padding_and_contiguous_chunks(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=7, target_tokens=6)
        self.assertEqual(layout.total_tokens, 34)
        chunks = [
            route.RouteInvocation(layout, rank, 4).local_roles(device=torch.device("cpu"))
            for rank in range(4)
        ]
        self.assertTrue(all(int(chunk.numel()) == 9 for chunk in chunks))
        joined = torch.cat(chunks)
        self.assertTrue(torch.equal(joined[:34], torch.tensor(layout.roles)))
        self.assertTrue(torch.equal(joined[34:], torch.zeros(2, dtype=torch.int64)))

    def test_nested_context_is_rejected_and_context_restores(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=1, target_tokens=1)
        invocation = route.RouteInvocation(layout, 0, 1)
        self.assertIsNone(route.active_route())
        with route.activate_route(invocation):
            self.assertIs(route.active_route(), invocation)
            with self.assertRaisesRegex(route.RAMPRouteError, "nested"):
                with route.activate_route(invocation):
                    pass
        self.assertIsNone(route.active_route())


class RAMPAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.transformer = _TinyTransformer()
        self.transformer.requires_grad_(False)
        self.original_patch = self.transformer.patch_embedding
        self.original_queries = tuple(block.attn1.to_q for block in self.transformer.blocks)
        self.handle = route.install_ramp_adapter(self.transformer, rank=2, alpha=2.0)

    def tearDown(self):
        if not self.handle.restored:
            self.handle.restore()

    def test_installation_freezes_base_and_receipt_is_explicit(self):
        self.assertTrue(self.handle.base_parameters_frozen())
        receipt = self.handle.receipt()
        self.assertTrue(receipt["target_query_only"])
        self.assertEqual(receipt["raw_donor_tokens"], 0)
        self.assertEqual(receipt["motion_tokens"], 21)
        self.assertEqual(len(receipt["digest"]), 64)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.original_patch.parameters()))
        self.assertTrue(all(not parameter.requires_grad for query in self.original_queries for parameter in query.parameters()))

    def test_absent_route_is_byte_exact_base_path(self):
        patches = torch.randn(5, 16, 1, 2, 2)
        expected_patch = self.original_patch(patches)
        actual_patch = self.transformer.patch_embedding(patches)
        self.assertTrue(torch.equal(actual_patch, expected_patch))
        hidden = torch.randn(1, 5, 8)
        for original, block in zip(self.original_queries, self.transformer.blocks):
            self.assertTrue(torch.equal(block.attn1.to_q(hidden), original(hidden)))

    def test_role_patch_path_preserves_source_and_target_conv_rows(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=2, target_tokens=3)
        invocation = route.RouteInvocation(layout, 0, 1)
        patches = _packed_patches(layout)
        expected = self.original_patch(patches).flatten(1)
        with self.handle.route(invocation):
            actual = self.transformer.patch_embedding(patches).flatten(1)
        source_end = layout.source_tokens
        target_start = source_end + layout.motion_tokens
        self.assertTrue(torch.equal(actual[:source_end], expected[:source_end]))
        self.assertTrue(torch.equal(actual[target_start:], expected[target_start:]))
        self.assertFalse(torch.equal(actual[source_end:target_start], expected[source_end:target_start]))

    def test_zero_lora_is_exact_then_nonzero_delta_is_target_only_sp1(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=3, target_tokens=4)
        invocation = route.RouteInvocation(layout, 0, 1)
        hidden = torch.ones(1, layout.total_tokens, 8)
        wrapper = self.handle.query_wrappers[0]
        expected = self.original_queries[0](hidden)
        with self.handle.route(invocation):
            zero = wrapper(hidden)
        self.assertTrue(torch.equal(zero, expected))
        with torch.no_grad():
            wrapper.lora_a.weight.fill_(0.25)
            wrapper.lora_b.weight.fill_(0.5)
        with self.handle.route(invocation):
            changed = wrapper(hidden)
        delta = changed - expected
        roles = torch.tensor(layout.roles)
        self.assertTrue(torch.equal(delta[:, roles != route.ROLE_TARGET], torch.zeros_like(delta[:, roles != route.ROLE_TARGET])))
        self.assertTrue(bool((delta[:, roles == route.ROLE_TARGET].abs() > 0).all().item()))

    def test_sp4_query_selector_changes_only_target_rows_on_every_rank(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=7, target_tokens=6)
        wrapper = self.handle.query_wrappers[0]
        with torch.no_grad():
            wrapper.lora_a.weight.fill_(0.25)
            wrapper.lora_b.weight.fill_(0.5)
        for rank in range(4):
            invocation = route.RouteInvocation(layout, rank, 4)
            hidden = torch.ones(1, invocation.local_length, 8)
            expected = self.original_queries[0](hidden)
            with self.handle.route(invocation):
                changed = wrapper(hidden)
            delta = changed - expected
            roles = invocation.local_roles(device=torch.device("cpu"))
            self.assertTrue(torch.equal(delta[:, roles != route.ROLE_TARGET], torch.zeros_like(delta[:, roles != route.ROLE_TARGET])))
            if bool((roles == route.ROLE_TARGET).any().item()):
                self.assertTrue(bool((delta[:, roles == route.ROLE_TARGET].abs() > 0).all().item()))

    def test_gradients_reach_program_projector_and_target_lora_only(self):
        layout = route.TokenRoleLayout.contiguous(source_tokens=2, target_tokens=3)
        invocation = route.RouteInvocation(layout, 0, 1)
        patches = _packed_patches(layout)
        wrapper = self.handle.query_wrappers[0]
        with torch.no_grad():
            wrapper.lora_b.weight.fill_(0.1)
        hidden = torch.randn(1, layout.total_tokens, 8, requires_grad=True)
        roles = torch.tensor(layout.roles)
        with self.handle.route(invocation):
            embedded = self.transformer.patch_embedding(patches).flatten(1)
            query = wrapper(hidden)
            loss = embedded[:, 0].sum() + query[:, roles == route.ROLE_TARGET].sum()
            loss.backward()
        self.assertIsNotNone(self.handle.patch_wrapper.program_projector.weight.grad)
        self.assertGreater(float(self.handle.patch_wrapper.program_projector.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(wrapper.lora_a.weight.grad)
        self.assertIsNotNone(wrapper.lora_b.weight.grad)
        self.assertTrue(torch.equal(hidden.grad[:, roles != route.ROLE_TARGET], torch.zeros_like(hidden.grad[:, roles != route.ROLE_TARGET])))
        self.assertGreater(float(hidden.grad[:, roles == route.ROLE_TARGET].abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in self.original_patch.parameters()))

    def test_restore_recovers_original_modules(self):
        self.handle.restore()
        self.assertIs(self.transformer.patch_embedding, self.original_patch)
        for block, original in zip(self.transformer.blocks, self.original_queries):
            self.assertIs(block.attn1.to_q, original)


if __name__ == "__main__":
    unittest.main()

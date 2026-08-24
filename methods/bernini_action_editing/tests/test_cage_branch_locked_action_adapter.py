from __future__ import annotations

from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    import cage_branch_locked_action_adapter as cage
    TEST_REGISTERED_BLOCKS = (2, 3)
    TEST_REGISTERED_PROJECTIONS = cage.CAGE_PROJECTIONS
else:
    cage = None  # type: ignore[assignment]


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(cage.TOTAL_BLOCKS_1P3B)]
            )

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


def _route(
    guidance_row: str,
    *,
    sigma_index: int = 0,
    rank: int = 0,
    size: int = 1,
    total: int = 13,
    condition: int | None = None,
) -> "cage.CAGEBranchLockedRoute":
    if condition is None:
        condition = 0 if guidance_row == "empty_uncond" else 5
    return cage.CAGEBranchLockedRoute(
        total_tokens=total,
        condition_tokens=condition,
        sequence_parallel_rank=rank,
        sequence_parallel_size=size,
        guidance_row=guidance_row,
        sigma_schedule_index=sigma_index,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAGEBranchRouteTests(unittest.TestCase):
    def test_route_is_exact_four_row_coordinate_and_replays_receipt(self) -> None:
        expected_native = {
            "empty_uncond": "none",
            "V_uncond": "V",
            "VI_uncond": "VI",
            "VI_cond": "VI",
        }
        self.assertEqual(tuple(expected_native), cage.GUIDANCE_ROWS)
        for guidance_row, native_branch in expected_native.items():
            with self.subTest(guidance_row=guidance_row):
                route = _route(guidance_row)
                self.assertEqual(route.native_branch, native_branch)
                receipt = route.receipt()
                self.assertEqual(cage.validate_route_receipt(receipt), receipt)
                self.assertEqual(receipt["derived_native_branch"], native_branch)
        for forbidden in ("none_uncond", "I", "I_uncond", "VI", ""):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(
                    cage.CAGEBranchLockError, "guidance_row"
                ):
                    _route(forbidden)
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "empty_uncond"):
            _route("empty_uncond", condition=1)
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "conditioned"):
            _route("VI_cond", condition=0)

    def test_sigma_band_is_unit_nonlow_and_low_is_base_only(self) -> None:
        for index in (0, 32, 33, 37):
            self.assertEqual(cage.sigma_gate(index), ("allowed_unit", 1.0))
            self.assertEqual(_route("VI_cond", sigma_index=index).gate_weight, 1.0)
        for index in (38, 39):
            self.assertEqual(cage.sigma_gate(index), ("low_base_only", 0.0))
            self.assertEqual(_route("VI_cond", sigma_index=index).gate_weight, 0.0)
        with self.assertRaisesRegex(cage.CAGEBranchLockError, r"\[0,39\]"):
            cage.sigma_gate(True)
        with self.assertRaisesRegex(cage.CAGEBranchLockError, r"\[0,39\]"):
            cage.sigma_gate(40)

    def test_authorization_truth_table_and_audit_receipt_are_closed(self) -> None:
        for guidance_row in cage.GUIDANCE_ROWS:
            for sigma_index in (0, 37, 38, 39):
                for block_index in (1, 2, 23):
                    expected = (
                        guidance_row == "VI_cond"
                        and sigma_index <= 37
                        and block_index in TEST_REGISTERED_BLOCKS
                    )
                    route = _route(guidance_row, sigma_index=sigma_index)
                    self.assertEqual(
                        route.authorizes_module(
                            block_index,
                            "attn2.to_q",
                            registered_block_indices=TEST_REGISTERED_BLOCKS,
                            registered_projections=TEST_REGISTERED_PROJECTIONS,
                        ),
                        expected,
                    )
                    receipt = cage.make_branch_lock_audit_receipt(
                        route,
                        block_index=block_index,
                        projection="attn2.to_q",
                        registered_block_indices=TEST_REGISTERED_BLOCKS,
                        registered_projections=TEST_REGISTERED_PROJECTIONS,
                    )
                    self.assertEqual(
                        cage.validate_branch_lock_audit_receipt(receipt), receipt
                    )
                    self.assertEqual(
                        receipt["target_rows_delta_authorized"], expected
                    )
                    self.assertFalse(receipt["source_rows_delta_authorized"])
                    self.assertFalse(receipt["padding_rows_delta_authorized"])
                    self.assertEqual(
                        receipt["full_projection_direct_base_required"],
                        not expected,
                    )
        self.assertFalse(
            cage.module_in_delta_band(
                2,
                "attn2.to_k",
                registered_block_indices=TEST_REGISTERED_BLOCKS,
                registered_projections=TEST_REGISTERED_PROJECTIONS,
            )
        )
        receipt = cage.make_branch_lock_audit_receipt(
            _route("VI_cond"),
            block_index=2,
            projection="attn2.to_q",
            registered_block_indices=TEST_REGISTERED_BLOCKS,
            registered_projections=TEST_REGISTERED_PROJECTIONS,
        )
        tampered = dict(receipt)
        tampered["source_rows_delta_authorized"] = True
        unsigned = dict(tampered)
        unsigned.pop("digest")
        tampered["digest"] = cage.object_sha256(unsigned)
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "replay"):
            cage.validate_branch_lock_audit_receipt(tampered)

    def test_registered_module_band_is_explicit_continuous_and_closed(self) -> None:
        receipt = cage.registered_module_band_receipt(
            TEST_REGISTERED_BLOCKS, TEST_REGISTERED_PROJECTIONS
        )
        self.assertEqual(
            cage.validate_registered_module_band_receipt(receipt), receipt
        )
        invalid = (
            ((), TEST_REGISTERED_PROJECTIONS, "empty"),
            ((2, 4), TEST_REGISTERED_PROJECTIONS, "continuous"),
            ((3, 2), TEST_REGISTERED_PROJECTIONS, "ascending"),
            ((2, 2), TEST_REGISTERED_PROJECTIONS, "unique"),
            ((29, 30), TEST_REGISTERED_PROJECTIONS, r"\[0,29\]"),
            (TEST_REGISTERED_BLOCKS, (), "projection.*empty"),
            (TEST_REGISTERED_BLOCKS, ("attn2.to_k",), "Q/O"),
            (
                TEST_REGISTERED_BLOCKS,
                ("attn2.to_out.0", "attn2.to_q"),
                "canonical",
            ),
        )
        for blocks, projections, message in invalid:
            with self.subTest(blocks=blocks, projections=projections):
                with self.assertRaisesRegex(cage.CAGEBranchLockError, message):
                    cage.validate_registered_module_band(blocks, projections)

    def test_sp4_selector_is_global_suffix_then_append_false_and_slice(self) -> None:
        routes = [
            _route("VI_cond", rank=rank, size=4) for rank in range(4)
        ]
        selectors = [
            route.local_target_selector(device=torch.device("cpu"))
            for route in routes
        ]
        joined = torch.cat(selectors)
        self.assertEqual(joined.numel(), 16)
        self.assertTrue(
            torch.equal(joined[:13], torch.tensor([False] * 5 + [True] * 8))
        )
        self.assertFalse(bool(joined[13:].any()))

    def test_context_is_not_nestable_and_resets_after_exception(self) -> None:
        route = _route("VI_cond")
        self.assertIsNone(cage.active_route())
        with cage.activate_route(route):
            self.assertIs(cage.active_route(), route)
            with self.assertRaisesRegex(cage.CAGEBranchLockError, "nested"):
                with cage.activate_route(route):
                    pass
        self.assertIsNone(cage.active_route())
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with cage.activate_route(route):
                raise RuntimeError("sentinel")
        self.assertIsNone(cage.active_route())

    def test_byte_comparator_distinguishes_signed_zero(self) -> None:
        negative_zero = torch.tensor([-0.0], dtype=torch.float32)
        positive_zero = torch.tensor([0.0], dtype=torch.float32)
        self.assertTrue(torch.equal(negative_zero, positive_zero))
        self.assertFalse(cage.tensors_byte_exact(negative_zero, positive_zero))
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "byte-exact"):
            cage.assert_tensors_byte_exact(
                negative_zero, positive_zero, label="signed-zero probe"
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class CAGEBranchLockedAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(812)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_patch = self.model.patch_embedding
        self.original_attn1 = tuple(
            (
                block.attn1,
                block.attn1.to_q,
                block.attn1.to_out[0],
            )
            for block in self.model.blocks
        )
        self.original_q = tuple(block.attn2.to_q for block in self.model.blocks)
        self.original_o = tuple(block.attn2.to_out[0] for block in self.model.blocks)
        self.handle = cage.install_cage_branch_locked_action_adapter(
            self.model,
            registered_block_indices=TEST_REGISTERED_BLOCKS,
            registered_projections=TEST_REGISTERED_PROJECTIONS,
        )

    def tearDown(self) -> None:
        if not self.handle.restored:
            self.handle.restore()

    @staticmethod
    def _make_nonzero(wrapper: "cage.CAGEBranchLockedActionLoRA") -> None:
        with torch.no_grad():
            wrapper.cage_lora_a.weight.fill_(0.25)
            wrapper.cage_lora_b.weight.fill_(0.5)

    @staticmethod
    def _factor_call_counter(
        wrapper: "cage.CAGEBranchLockedActionLoRA",
    ) -> tuple[dict[str, int], tuple[object, object]]:
        calls = {"a": 0, "b": 0}

        def count_a(module: nn.Module, inputs: object) -> None:
            del module, inputs
            calls["a"] += 1

        def count_b(module: nn.Module, inputs: object) -> None:
            del module, inputs
            calls["b"] += 1

        return calls, (
            wrapper.cage_lora_a.register_forward_pre_hook(count_a),
            wrapper.cage_lora_b.register_forward_pre_hook(count_b),
        )

    def test_three_locked_guidance_rows_are_full_byte_exact_and_skip_lora(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.randn(1, 13, 8)
        calls, hooks = self._factor_call_counter(wrapper)
        try:
            for guidance_row in cage.LOCKED_GUIDANCE_ROWS:
                with self.subTest(guidance_row=guidance_row):
                    expected = self.original_q[2](hidden)
                    with self.handle.route(_route(guidance_row)):
                        actual = wrapper(hidden)
                        delta = wrapper.adapter_delta(hidden)
                    cage.assert_tensors_byte_exact(
                        actual, expected, label=guidance_row
                    )
                    cage.assert_tensors_byte_exact(
                        delta, torch.zeros_like(delta), label=f"{guidance_row} delta"
                    )
            self.assertEqual(calls, {"a": 0, "b": 0})
        finally:
            for hook in hooks:
                hook.remove()

    def test_low_sigma_is_full_byte_exact_and_skips_lora(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.randn(1, 13, 8)
        calls, hooks = self._factor_call_counter(wrapper)
        try:
            for sigma_index in (38, 39):
                expected = self.original_q[2](hidden)
                with self.handle.route(
                    _route("VI_cond", sigma_index=sigma_index)
                ):
                    actual = wrapper(hidden)
                    delta = wrapper.adapter_delta(hidden)
                cage.assert_tensors_byte_exact(
                    actual, expected, label=f"low sigma {sigma_index}"
                )
                cage.assert_tensors_byte_exact(
                    delta,
                    torch.zeros_like(delta),
                    label=f"low sigma {sigma_index} delta",
                )
            self.assertEqual(calls, {"a": 0, "b": 0})
        finally:
            for hook in hooks:
                hook.remove()

    def test_vi_cond_nonlow_changes_only_target_rows_at_unit_weight(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 13, 8)
        route0 = _route("VI_cond", sigma_index=0)
        route37 = _route("VI_cond", sigma_index=37)
        selector = route0.local_target_selector(device=torch.device("cpu"))
        expected = self.original_q[2](hidden)
        with self.handle.route(route0):
            high = wrapper(hidden)
            high_delta = wrapper.adapter_delta(hidden)
        with self.handle.route(route37):
            boundary = wrapper(hidden)
            boundary_delta = wrapper.adapter_delta(hidden)
        cage.assert_tensors_byte_exact(
            high[:, ~selector], expected[:, ~selector], label="VI_cond source"
        )
        cage.assert_tensors_byte_exact(
            boundary[:, ~selector],
            expected[:, ~selector],
            label="VI_cond boundary source",
        )
        self.assertTrue(cage.tensors_byte_exact(high_delta, boundary_delta))
        self.assertGreater(
            float((high[:, selector] - expected[:, selector]).abs().sum()), 0.0
        )
        self.assertGreater(
            float((boundary[:, selector] - expected[:, selector]).abs().sum()),
            0.0,
        )

    def test_sp4_source_and_padding_rows_are_byte_exact_on_every_rank(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        for rank in range(4):
            route = _route("VI_cond", rank=rank, size=4)
            hidden = torch.ones(1, route.local_length, 8)
            selector = route.local_target_selector(device=torch.device("cpu"))
            expected = self.original_q[2](hidden)
            with self.handle.route(route):
                actual = wrapper(hidden)
            cage.assert_tensors_byte_exact(
                actual[:, ~selector],
                expected[:, ~selector],
                label=f"SP4 rank {rank} source/padding",
            )
            if bool(selector.any().item()):
                self.assertGreater(
                    float(
                        (actual[:, selector] - expected[:, selector]).abs().sum()
                    ),
                    0.0,
                )
            else:
                cage.assert_tensors_byte_exact(
                    actual, expected, label=f"SP4 rank {rank} no-target"
                )

    def test_outside_module_band_is_direct_base_and_skips_lora(self) -> None:
        base = self.original_q[23]
        wrapper = cage.CAGEBranchLockedActionLoRA(
            base,
            block_index=23,
            projection="attn2.to_q",
            registered_block_indices=TEST_REGISTERED_BLOCKS,
            registered_projections=TEST_REGISTERED_PROJECTIONS,
        )
        self._make_nonzero(wrapper)
        hidden = torch.randn(1, 13, 8)
        calls, hooks = self._factor_call_counter(wrapper)
        try:
            expected = base(hidden)
            with cage.activate_route(_route("VI_cond", sigma_index=0)):
                actual = wrapper(hidden)
                delta = wrapper.adapter_delta(hidden)
            cage.assert_tensors_byte_exact(
                actual, expected, label="outside module band"
            )
            cage.assert_tensors_byte_exact(
                delta, torch.zeros_like(delta), label="outside module delta"
            )
            self.assertEqual(calls, {"a": 0, "b": 0})
        finally:
            for hook in hooks:
                hook.remove()

    def test_single_wrapper_can_probe_each_of_all_30_blocks(self) -> None:
        self.assertEqual(cage.PROBE_BLOCK_INDICES, tuple(range(30)))
        hidden = torch.ones(1, 13, 8)
        for block_index in cage.PROBE_BLOCK_INDICES:
            with self.subTest(block_index=block_index):
                base = nn.Linear(8, 8, bias=False)
                base.requires_grad_(False)
                wrapper = cage.CAGEBranchLockedActionLoRA(
                    base,
                    block_index=block_index,
                    projection="attn2.to_q",
                    registered_block_indices=(block_index,),
                    registered_projections=("attn2.to_q",),
                )
                self._make_nonzero(wrapper)
                with cage.activate_route(_route("VI_cond")):
                    delta = wrapper.adapter_delta(hidden)
                self.assertGreater(float(delta.abs().sum()), 0.0)

    def test_gradient_graph_exists_only_for_authorized_branch_and_rows(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        locked_hidden = torch.ones(1, 13, 8, requires_grad=True)
        with self.handle.route(_route("VI_uncond")):
            wrapper(locked_hidden).sum().backward()
        self.assertIsNone(wrapper.cage_lora_a.weight.grad)
        self.assertIsNone(wrapper.cage_lora_b.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.original_q[2].parameters())
        )

        source_hidden = torch.ones(1, 13, 8, requires_grad=True)
        source_selector = ~_route("VI_cond").local_target_selector(
            device=torch.device("cpu")
        )
        with self.handle.route(_route("VI_cond")):
            wrapper(source_hidden)[:, source_selector].sum().backward()
        for parameter in (
            wrapper.cage_lora_a.weight,
            wrapper.cage_lora_b.weight,
        ):
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
                )
            parameter.grad = None

        active_hidden = torch.ones(1, 13, 8, requires_grad=True)
        selector = _route("VI_cond").local_target_selector(
            device=torch.device("cpu")
        )
        with self.handle.route(_route("VI_cond")):
            wrapper(active_hidden)[:, selector].sum().backward()
        self.assertGreater(float(wrapper.cage_lora_a.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(wrapper.cage_lora_b.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.original_q[2].parameters())
        )

    def test_install_scope_receipt_and_restore_are_exact(self) -> None:
        self.handle.assert_scope()
        self.assertIs(self.model.patch_embedding, self.original_patch)
        self.assertTrue(self.handle.base_parameters_frozen())
        self.assertTrue(self.handle.self_attention_untouched())
        for index, block in enumerate(self.model.blocks):
            attn1, attn1_q, attn1_o = self.original_attn1[index]
            self.assertIs(block.attn1, attn1)
            self.assertIs(block.attn1.to_q, attn1_q)
            self.assertIs(block.attn1.to_out[0], attn1_o)
            if index in TEST_REGISTERED_BLOCKS:
                self.assertIsInstance(
                    block.attn2.to_q, cage.CAGEBranchLockedActionLoRA
                )
                self.assertIsInstance(
                    block.attn2.to_out[0], cage.CAGEBranchLockedActionLoRA
                )
            else:
                self.assertIs(block.attn2.to_q, self.original_q[index])
                self.assertIs(block.attn2.to_out[0], self.original_o[index])
        trainable = self.handle.trainable_named_parameters()
        self.assertEqual(len(trainable), 2 * 2 * 2)
        receipt = self.handle.receipt()
        self.assertEqual(cage.validate_adapter_receipt(receipt), receipt)
        self.assertEqual(
            receipt["module_band"]["block_indices"],
            list(TEST_REGISTERED_BLOCKS),
        )
        self.assertEqual(
            receipt["module_band"]["projections"],
            list(TEST_REGISTERED_PROJECTIONS),
        )
        self.assertTrue(receipt["module_band"]["explicit_no_default"])
        self.assertTrue(receipt["locked_branches_direct_base_return"])
        self.assertTrue(
            receipt[
                "wrapped_projection_source_and_padding_rows_byte_exact_base"
            ]
        )
        self.assertFalse(receipt["full_source_memory_reinjection_implemented"])
        self.assertFalse(receipt["global_source_memory_lock_claim"])
        self.assertFalse(receipt["optimizer_authorized"])
        self.assertFalse(receipt["semantic_action_claim"])

        unregistered_hidden = torch.randn(1, 13, 8)
        expected_unregistered = self.original_q[1](unregistered_hidden)
        with self.handle.route(_route("VI_cond")):
            actual_unregistered = self.model.blocks[1].attn2.to_q(
                unregistered_hidden
            )
        cage.assert_tensors_byte_exact(
            actual_unregistered,
            expected_unregistered,
            label="unregistered block under active VI_cond",
        )

        with self.assertRaisesRegex(cage.CAGEBranchLockError, "cannot be restored"):
            with self.handle.route(_route("VI_cond")):
                self.handle.restore()
        self.handle.restore()
        for index, block in enumerate(self.model.blocks):
            self.assertIs(block.attn2.to_q, self.original_q[index])
            self.assertIs(block.attn2.to_out[0], self.original_o[index])

    def test_projection_band_can_register_q_without_silently_wrapping_o(self) -> None:
        model = _Transformer()
        model.requires_grad_(False)
        original_q = tuple(block.attn2.to_q for block in model.blocks)
        original_o = tuple(block.attn2.to_out[0] for block in model.blocks)
        handle = cage.install_cage_branch_locked_action_adapter(
            model,
            registered_block_indices=TEST_REGISTERED_BLOCKS,
            registered_projections=("attn2.to_q",),
        )
        try:
            self.assertEqual(len(handle.q_wrappers), 2)
            self.assertEqual(len(handle.o_wrappers), 0)
            for index, block in enumerate(model.blocks):
                if index in TEST_REGISTERED_BLOCKS:
                    self.assertIsInstance(
                        block.attn2.to_q, cage.CAGEBranchLockedActionLoRA
                    )
                else:
                    self.assertIs(block.attn2.to_q, original_q[index])
                self.assertIs(block.attn2.to_out[0], original_o[index])
            receipt = cage.validate_adapter_receipt(handle.receipt())
            self.assertEqual(
                receipt["module_band"]["projections"], ["attn2.to_q"]
            )
            self.assertEqual(len(receipt["trainable"]), 4)
        finally:
            if not handle.restored:
                handle.restore()

    def test_installer_has_no_implicit_or_empty_band(self) -> None:
        frozen = _Transformer()
        frozen.requires_grad_(False)
        with self.assertRaises(TypeError):
            cage.install_cage_branch_locked_action_adapter(frozen)
        for blocks, projections, message in (
            ((), TEST_REGISTERED_PROJECTIONS, "empty"),
            ((2, 4), TEST_REGISTERED_PROJECTIONS, "continuous"),
            ((29, 30), TEST_REGISTERED_PROJECTIONS, r"\[0,29\]"),
            (TEST_REGISTERED_BLOCKS, (), "projection.*empty"),
        ):
            with self.subTest(blocks=blocks, projections=projections):
                candidate = _Transformer()
                candidate.requires_grad_(False)
                with self.assertRaisesRegex(cage.CAGEBranchLockError, message):
                    cage.install_cage_branch_locked_action_adapter(
                        candidate,
                        registered_block_indices=blocks,
                        registered_projections=projections,
                    )

    def test_install_rejects_unfrozen_or_wrong_structure(self) -> None:
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "freeze"):
            cage.install_cage_branch_locked_action_adapter(
                _Transformer(),
                registered_block_indices=TEST_REGISTERED_BLOCKS,
                registered_projections=TEST_REGISTERED_PROJECTIONS,
            )
        wrong = _Transformer()
        wrong.requires_grad_(False)
        wrong.blocks = nn.ModuleList(list(wrong.blocks)[:-1])
        with self.assertRaisesRegex(cage.CAGEBranchLockError, "structure"):
            cage.install_cage_branch_locked_action_adapter(
                wrong,
                registered_block_indices=TEST_REGISTERED_BLOCKS,
                registered_projections=TEST_REGISTERED_PROJECTIONS,
            )


if __name__ == "__main__":
    unittest.main()

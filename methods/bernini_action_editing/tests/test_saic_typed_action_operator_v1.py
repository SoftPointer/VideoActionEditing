from __future__ import annotations

from pathlib import Path
import importlib
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import saic_typed_action_operator_v1 as saic

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    saic = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.to_q = nn.Linear(hidden, hidden, bias=False)
            self.to_k = nn.Linear(hidden, hidden, bias=False)
            self.to_v = nn.Linear(hidden, hidden, bias=False)
            self.to_out = nn.ModuleList(
                [nn.Linear(hidden, hidden, bias=False), nn.Identity()]
            )


    class _Block(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.attn1 = _Attention(hidden)
            self.attn2 = _Attention(hidden)


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8, blocks: int = 30) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList([_Block(hidden) for _ in range(blocks)])
            self.gradient_checkpointing = False

        @property
        def is_gradient_checkpointing(self) -> bool:
            return bool(self.gradient_checkpointing)

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


    class _NativeBranch:
        def __init__(
            self, *, name: str, total_tokens: int, condition_tokens: int
        ) -> None:
            descriptor = saic.NATIVE_BRANCH_ORDERED_DESCRIPTORS.get(
                name, (("target", "00000000"),)
            )
            self.name = name
            self.total_tokens = total_tokens
            self.condition_tokens = condition_tokens
            self.concat_order = tuple(role for role, _ in descriptor)
            self.source_ids = tuple(
                float.fromhex("0x0p+0")
                if encoded == "00000000"
                else float(
                    torch.tensor(
                        int(encoded, 16), dtype=torch.int32
                    ).view(torch.float32).item()
                )
                for _, encoded in descriptor
            )
            self.latents = torch.zeros(1, max(total_tokens, 1), 8)
            self.target_mask = torch.cat(
                (
                    torch.zeros(max(condition_tokens, 0), dtype=torch.bool),
                    torch.ones(
                        max(total_tokens - condition_tokens, 0), dtype=torch.bool
                    ),
                )
            )[: max(total_tokens, 0)]


def _action_arrow() -> "saic.SAICArrowCode":
    return saic.SAICArrowCode.quantized(
        "standing", "sitting", [0.25] * saic.ARROW_CODE_DIM
    )


def _route(
    *,
    handle: "saic.SAICTypedActionOperatorHandle | None" = None,
    sigma_index: int = 0,
    arrow: "saic.SAICArrowCode | None" = None,
    rank: int = 0,
    size: int = 1,
    total: int = 13,
    condition: int = 5,
    branch_name: str = "VI",
) -> "saic.SAICTypedActionRoute":
    owned_handle = handle
    if owned_handle is None:
        model = _Transformer(hidden=8)
        model.requires_grad_(False)
        owned_handle = saic.install_saic_typed_action_operator(model)
    branch = _NativeBranch(
        name=branch_name,
        total_tokens=total,
        condition_tokens=condition,
    )
    local_length = (total + size - 1) // size if size > 0 else 0
    global_mask = branch.target_mask
    padded = torch.cat(
        (
            global_mask,
            torch.zeros(
                max(local_length * size - total, 0), dtype=torch.bool
            ),
        )
    )
    local_mask = padded[
        rank * local_length : (rank + 1) * local_length
    ].contiguous()
    if 0 <= sigma_index < saic.sigma_strata.NUM_INFERENCE_STEPS:
        sigma = saic.sigma_strata.PINNED_POSITIVE_SIGMAS[sigma_index]
    else:
        sigma = 0.0
    with mock.patch.object(
        saic, "_query_sequence_parallel_coordinate", return_value=(rank, size)
    ):
        return owned_handle.bind_runtime_route(
            native_branch=branch,
            actual_local_target_mask=local_mask,
            actual_sigma=torch.tensor(sigma, dtype=torch.float32),
            arrow=_action_arrow() if arrow is None else arrow,
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICArrowAndRouteTests(unittest.TestCase):
    def test_exact_fp32_typed_arrow_and_sign_reversal(self) -> None:
        arrow = _action_arrow()
        reversed_arrow = arrow.sign_reversed()
        self.assertEqual(reversed_arrow.initial_state_type, "sitting")
        self.assertEqual(reversed_arrow.terminal_state_type, "standing")
        self.assertTrue(
            torch.equal(
                reversed_arrow.tensor(device=torch.device("cpu")),
                -arrow.tensor(device=torch.device("cpu")),
            )
        )
        self.assertEqual(reversed_arrow.sign_reversed(), arrow)
        receipt = dict(arrow.receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, saic._object_sha256(receipt))
        self.assertEqual(len(receipt["float32_be_hex"]), saic.ARROW_CODE_DIM)

    def test_endpoint_difference_and_noop_are_typed(self) -> None:
        initial = torch.zeros(saic.ARROW_CODE_DIM, dtype=torch.float32)
        terminal = torch.full(
            (saic.ARROW_CODE_DIM,), 0.5, dtype=torch.float32
        )
        arrow = saic.SAICArrowCode.between(
            "standing", initial, "sitting", terminal
        )
        self.assertTrue(
            torch.equal(
                arrow.tensor(device=torch.device("cpu")), terminal - initial
            )
        )
        noop = saic.SAICArrowCode.between(
            "standing", initial, "standing", initial.clone()
        )
        self.assertTrue(noop.is_noop)
        self.assertEqual(noop, saic.SAICArrowCode.noop("standing"))

    def test_malformed_arrows_fail_closed(self) -> None:
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "length 32"):
            saic.SAICArrowCode.quantized("standing", "sitting", [1.0] * 31)
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "exact FP32"):
            saic.SAICArrowCode(
                "standing", "sitting", tuple([0.1] * saic.ARROW_CODE_DIM)
            )
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "zero arrow iff"):
            saic.SAICArrowCode.noop("standing").__class__(
                "standing", "sitting", (0.0,) * saic.ARROW_CODE_DIM
            )
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "zero arrow iff"):
            saic.SAICArrowCode.quantized(
                "standing", "standing", [0.25] * saic.ARROW_CODE_DIM
            )
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "initial_state_type"):
            saic.SAICArrowCode.quantized(
                "Standing Pose", "sitting", [0.25] * saic.ARROW_CODE_DIM
            )

    def test_exact40_phase_and_sp4_target_suffix(self) -> None:
        self.assertEqual(saic.HIGH_SIGMA_INDICES, tuple(range(33)))
        self.assertEqual(saic.MID_SIGMA_INDICES, tuple(range(33, 38)))
        self.assertEqual(saic.LOW_SIGMA_INDICES, (38, 39))
        self.assertEqual(saic.sigma_gate(0), ("high", 1.0))
        self.assertEqual(saic.sigma_gate(33), ("mid", 0.5))
        self.assertEqual(saic.sigma_gate(38), ("low_base_only", 0.0))
        selectors = [
            _route(rank=rank, size=4).local_target_selector(
                device=torch.device("cpu")
            )
            for rank in range(4)
        ]
        joined = torch.cat(selectors)
        self.assertEqual(joined.numel(), 16)
        self.assertTrue(
            torch.equal(joined[:13], torch.tensor([False] * 5 + [True] * 8))
        )
        self.assertFalse(bool(joined[13:].any().item()))

    def test_noop_route_is_base_only_at_every_sigma(self) -> None:
        route = _route(
            sigma_index=0, arrow=saic.SAICArrowCode.noop("standing")
        )
        self.assertFalse(route.operator_active)
        self.assertEqual(route.gate_name, "noop_base_only")
        self.assertEqual(route.gate_weight, 0.0)

    def test_malformed_routes_and_nested_context_fail_closed(self) -> None:
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "descriptor name"):
            _route(branch_name="source")
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "prefix disagree"):
            _route(branch_name="none", condition=1)
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "condition geometry"):
            _route(total=8, condition=8)
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "SP1/SP4"):
            _route(size=2)
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "pinned"):
            _route(sigma_index=40)
        route = _route()
        with saic.activate_route(route):
            with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "nested"):
                with saic.activate_route(route):
                    pass
        self.assertIsNone(saic.active_route())

    def test_direct_route_constructor_is_non_executable(self) -> None:
        direct = saic.SAICTypedActionRoute(
            total_tokens=13,
            condition_tokens=5,
            sequence_parallel_rank=0,
            sequence_parallel_size=1,
            branch_name="VI",
            sigma_schedule_index=0,
            arrow=_action_arrow(),
        )
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "factory-bound"
        ):
            with saic.activate_route(direct):
                pass
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "factory-bound"
        ):
            direct.receipt()

    def test_factory_receipt_binds_actual_mask_parallel_sigma_and_order(self) -> None:
        route = _route(rank=3, size=4, sigma_index=33, branch_name="VI")
        receipt = route.receipt()
        runtime = receipt["runtime_descriptor"]
        self.assertTrue(receipt["route_factory_bound"])
        self.assertEqual(runtime["parallel_rank"], 3)
        self.assertEqual(runtime["parallel_size"], 4)
        self.assertEqual(
            runtime["actual_sigma_float32_be_hex"],
            saic.sigma_strata.PINNED_POSITIVE_SIGMA_FLOAT32_HEX[33],
        )
        self.assertEqual(
            runtime["pinned_schedule_sha256"], saic.sigma_strata.SCHEDULE_SHA256
        )
        self.assertEqual(
            [row["role"] for row in runtime["branch_ordered_descriptor"]],
            ["video", "ref0", "ref1", "ref2", "ref3", "target"],
        )
        self.assertRegex(runtime["local_target_mask_sha256"], r"^[0-9a-f]{64}$")

    def test_package_relative_import_resolves_sibling_module(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        with mock.patch.object(sys, "path", [str(repository_root), *sys.path]):
            package_module = importlib.import_module(
                "methods.bernini_action_editing.saic_typed_action_operator_v1"
            )
        self.assertEqual(
            package_module.sigma_strata.__name__,
            "methods.bernini_action_editing.inference_sigma_strata",
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICTypedActionOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(809)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_patch = self.model.patch_embedding
        self.original_q = tuple(block.attn2.to_q for block in self.model.blocks)
        self.original_o = tuple(block.attn2.to_out[0] for block in self.model.blocks)
        self.original_kv = tuple(
            (block.attn2.to_k, block.attn2.to_v) for block in self.model.blocks
        )
        self.original_attn1 = tuple(block.attn1 for block in self.model.blocks)
        self.handle = saic.install_saic_typed_action_operator(self.model)

    def tearDown(self) -> None:
        if not self.handle.restored:
            self.handle.restore()

    @staticmethod
    def _make_nonzero(wrapper: "saic.SAICTargetRowTypedActionOperator") -> None:
        with torch.no_grad():
            for stratum in ("high", "mid"):
                getattr(wrapper, f"state_down_{stratum}").weight.fill_(0.125)
                getattr(wrapper, f"arrow_gate_{stratum}").weight.fill_(0.125)
                getattr(wrapper, f"output_up_{stratum}").weight.fill_(0.25)

    def test_install_is_exact_function_preserving_then_action_is_state_dependent(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        hidden = torch.ones(1, 13, 8)
        with self.handle.route(_route(handle=self.handle)):
            installed = wrapper(hidden)
        self.assertTrue(torch.equal(installed, self.original_q[0](hidden)))
        self._make_nonzero(wrapper)
        with self.handle.route(_route(handle=self.handle)):
            delta_one = wrapper.adapter_delta(hidden)
            delta_two = wrapper.adapter_delta(hidden * 2.0)
        self.assertGreater(float(delta_one.abs().sum().item()), 0.0)
        self.assertFalse(torch.equal(delta_one, delta_two))

    def test_noop_and_low_sigma_direct_base_even_with_nonfinite_operator(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        with torch.no_grad():
            wrapper.state_down.weight.fill_(float("nan"))
            wrapper.arrow_gate.weight.fill_(float("nan"))
            wrapper.output_up.weight.fill_(float("nan"))
        hidden = torch.randn(1, 13, 8)
        expected = self.original_q[0](hidden)
        noop = _route(
            handle=self.handle, arrow=saic.SAICArrowCode.noop("standing")
        )
        low = _route(handle=self.handle, sigma_index=38)
        for route in (noop, low):
            with self.handle.route(route):
                actual = wrapper(hidden)
                delta = wrapper.adapter_delta(hidden)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_sign_reversed_arrow_is_same_hidden_exactly_odd_before_up(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 13, 8)
        forward_route = _route(handle=self.handle)
        reversed_route = _route(
            handle=self.handle, arrow=forward_route.arrow.sign_reversed()
        )
        forward = wrapper.selected_pre_up(hidden, forward_route)
        reversed_feature = wrapper.selected_pre_up(hidden, reversed_route)
        self.assertTrue(torch.equal(reversed_feature, -forward))

    def test_source_rows_and_padding_rows_are_exact_for_q_and_o(self) -> None:
        for rank in range(4):
            route = _route(handle=self.handle, rank=rank, size=4)
            selector = route.local_target_selector(device=torch.device("cpu"))
            hidden = torch.ones(1, route.local_length, 8)
            for wrapper, original in (
                (self.handle.q_wrappers[0][1], self.original_q[0]),
                (self.handle.o_wrappers[0][1], self.original_o[0]),
            ):
                self._make_nonzero(wrapper)
                expected = original(hidden)
                with self.handle.route(route):
                    actual = wrapper(hidden)
                self.assertTrue(
                    torch.equal(actual[:, ~selector, :], expected[:, ~selector, :])
                )
                if bool(selector.any().item()):
                    self.assertGreater(
                        float(
                            (actual[:, selector, :] - expected[:, selector, :])
                            .abs()
                            .sum()
                            .item()
                        ),
                        0.0,
                    )

    def test_mid_sigma_is_exactly_half_the_high_delta(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 13, 8)
        with self.handle.route(_route(handle=self.handle, sigma_index=0)):
            high = wrapper.adapter_delta(hidden)
        with self.handle.route(_route(handle=self.handle, sigma_index=33)):
            mid = wrapper.adapter_delta(hidden)
        self.assertTrue(torch.equal(mid * 2.0, high))

    def test_high_and_mid_sigma_use_disjoint_complete_parameter_sets(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        with torch.no_grad():
            wrapper.state_down_high.weight.fill_(0.125)
            wrapper.arrow_gate_high.weight.fill_(0.125)
            wrapper.output_up_high.weight.fill_(0.25)
            wrapper.state_down_mid.weight.fill_(0.125)
            wrapper.arrow_gate_mid.weight.fill_(0.125)
            wrapper.output_up_mid.weight.zero_()
        hidden = torch.ones(1, 13, 8)
        with self.handle.route(_route(handle=self.handle, sigma_index=0)):
            high = wrapper.adapter_delta(hidden)
        with self.handle.route(_route(handle=self.handle, sigma_index=33)):
            mid = wrapper.adapter_delta(hidden)
        self.assertGreater(float(high.abs().sum().item()), 0.0)
        self.assertTrue(torch.equal(mid, torch.zeros_like(mid)))
        high_parameters = self.handle.trainable_named_parameters_for_sigma("high")
        mid_parameters = self.handle.trainable_named_parameters_for_sigma("mid")
        self.assertEqual(len(high_parameters), 23 * 2 * 3)
        self.assertEqual(len(mid_parameters), 23 * 2 * 3)
        self.assertFalse(
            {id(parameter) for _, parameter in high_parameters}
            & {id(parameter) for _, parameter in mid_parameters}
        )
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "optimizer sigma stratum"
        ):
            self.handle.trainable_named_parameters_for_sigma("low")

    def test_all_operator_parameters_receive_gradient_and_base_stays_frozen(self) -> None:
        route = _route(handle=self.handle, total=5, condition=2)
        hidden = torch.ones(1, 5, 8)
        selector = route.local_target_selector(device=torch.device("cpu"))
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            self._make_nonzero(wrapper)
        loss = torch.zeros((), dtype=torch.float32)
        with self.handle.route(route):
            for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
                loss = loss + wrapper(hidden)[:, selector, :].sum()
        loss.backward()
        trainable = self.handle.trainable_named_parameters()
        self.assertEqual(len(trainable), 23 * 2 * 6)
        for name, parameter in trainable:
            self.assertIsNotNone(parameter.grad, name)
            if "_high.weight" in name:
                self.assertGreater(float(parameter.grad.abs().sum().item()), 0.0, name)
            else:
                self.assertEqual(float(parameter.grad.abs().sum().item()), 0.0, name)
        for _, parameter in trainable:
            parameter.grad = None
        mid_route = _route(handle=self.handle, sigma_index=33, total=5, condition=2)
        loss = torch.zeros((), dtype=torch.float32)
        with self.handle.route(mid_route):
            for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
                loss = loss + wrapper(hidden)[:, selector, :].sum()
        loss.backward()
        for name, parameter in trainable:
            self.assertIsNotNone(parameter.grad, name)
            if "_mid.weight" in name:
                self.assertGreater(float(parameter.grad.abs().sum().item()), 0.0, name)
            else:
                self.assertEqual(float(parameter.grad.abs().sum().item()), 0.0, name)
        original_parameters = [
            parameter
            for module in self.original_q + self.original_o
            for parameter in module.parameters()
        ]
        self.assertTrue(all(parameter.grad is None for parameter in original_parameters))

    def test_zero_init_warmup_releases_state_and_arrow_gradients_on_second_step(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        with torch.no_grad():
            wrapper.state_down.weight.fill_(0.125)
            wrapper.arrow_gate.weight.fill_(0.125)
        self.assertEqual(int(torch.count_nonzero(wrapper.output_up.weight).item()), 0)
        parameters = (
            wrapper.state_down.weight,
            wrapper.arrow_gate.weight,
            wrapper.output_up.weight,
        )
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        hidden = torch.ones(1, 13, 8)
        route = _route(handle=self.handle)
        with self.handle.route(route):
            wrapper(hidden).sum().backward()
        self.assertEqual(
            int(torch.count_nonzero(wrapper.state_down.weight.grad).item()), 0
        )
        self.assertEqual(
            int(torch.count_nonzero(wrapper.arrow_gate.weight.grad).item()), 0
        )
        self.assertGreater(
            int(torch.count_nonzero(wrapper.output_up.weight.grad).item()), 0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with self.handle.route(route):
            wrapper(hidden).sum().backward()
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum().item()), 0.0)

    def test_empty_target_sp_shard_retains_operator_autograd_topology(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        route = _route(
            handle=self.handle,
            rank=0,
            size=4,
            total=8,
            condition=4,
            branch_name="V",
        )
        selector = route.local_target_selector(device=torch.device("cpu"))
        self.assertFalse(bool(selector.any().item()))
        hidden = torch.ones(1, route.local_length, 8)
        with self.handle.route(route):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, self.original_q[0](hidden)))
        self.assertTrue(actual.requires_grad)
        actual.sum().backward()
        for parameter in (
            wrapper.state_down.weight,
            wrapper.arrow_gate.weight,
            wrapper.output_up.weight,
            wrapper.state_down_mid.weight,
            wrapper.arrow_gate_mid.weight,
            wrapper.output_up_mid.weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(int(torch.count_nonzero(parameter.grad).item()), 0)

    def test_scope_is_only_attn2_q_o_blocks_zero_through_twenty_two(self) -> None:
        self.assertIs(self.model.patch_embedding, self.original_patch)
        self.assertTrue(self.handle.base_parameters_frozen())
        self.assertTrue(self.handle.protected_attention_untouched())
        for index, block in enumerate(self.model.blocks):
            self.assertIs(block.attn1, self.original_attn1[index])
            self.assertIs(block.attn2.to_k, self.original_kv[index][0])
            self.assertIs(block.attn2.to_v, self.original_kv[index][1])
            if index in saic.ACTION_BLOCK_INDICES:
                self.assertIsInstance(
                    block.attn2.to_q, saic.SAICTargetRowTypedActionOperator
                )
                self.assertIsInstance(
                    block.attn2.to_out[0], saic.SAICTargetRowTypedActionOperator
                )
            else:
                self.assertIs(block.attn2.to_q, self.original_q[index])
                self.assertIs(block.attn2.to_out[0], self.original_o[index])
        receipt = dict(self.handle.receipt())
        digest = receipt.pop("digest")
        self.assertEqual(digest, saic._object_sha256(receipt))
        self.assertEqual(receipt["rank"], 8)
        self.assertEqual(receipt["arrow_code_dimension"], 32)
        self.assertFalse(receipt["key_value_trainable"])
        self.assertFalse(receipt["self_attention_trainable"])
        self.assertFalse(receipt["late_blocks_trainable"])
        self.assertTrue(
            receipt["sigma_parameter_partition"][
                "complete_parameter_disjoint_heads"
            ]
        )
        self.assertFalse(
            receipt["sigma_parameter_partition"]["shared_trainable_parameters"]
        )

    def test_closed_state_digest_checkpoint_save_and_load(self) -> None:
        with torch.no_grad():
            for index, (_, parameter) in enumerate(
                self.handle.trainable_named_parameters()
            ):
                parameter.fill_(float(index + 1) / 256.0)
        expected_digest = self.handle.trainable_state_digest()
        expected_state = dict(self.handle.state_dict_for_save())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saic.pt"
            save_receipt = self.handle.save_checkpoint(path)
            self.assertTrue(path.is_file())
            self.assertEqual(save_receipt["state_tensor_sha256"], expected_digest)
            with torch.no_grad():
                for _, parameter in self.handle.trainable_named_parameters():
                    parameter.zero_()
            self.assertNotEqual(self.handle.trainable_state_digest(), expected_digest)
            load_receipt = self.handle.load_checkpoint(path)
            self.assertEqual(load_receipt["state_tensor_sha256"], expected_digest)
        observed = dict(self.handle.state_dict_for_save())
        self.assertEqual(set(observed), set(expected_state))
        self.assertTrue(
            all(torch.equal(observed[name], value) for name, value in expected_state.items())
        )
        missing = dict(expected_state)
        missing.pop(sorted(missing)[0])
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "key closure"):
            self.handle.load_trainable_state_dict(missing)

    def test_hidden_shape_mismatch_and_restore_fail_closed(self) -> None:
        wrapper = self.handle.q_wrappers[0][1]
        with self.handle.route(_route(handle=self.handle)):
            with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, r"\[1,N,D\]"):
                wrapper(torch.ones(2, 13, 8))
            with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "local hidden"):
                wrapper(torch.ones(1, 12, 8))
            with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "cannot be restored"):
                self.handle.restore()
        self.handle.restore()
        for index, block in enumerate(self.model.blocks):
            self.assertIs(block.attn2.to_q, self.original_q[index])
            self.assertIs(block.attn2.to_out[0], self.original_o[index])

    def test_factory_rejects_wrong_actual_source_target_mask(self) -> None:
        branch = _NativeBranch(name="VI", total_tokens=13, condition_tokens=5)
        wrong_local_mask = branch.target_mask.clone()
        wrong_local_mask[0] = True
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "actual local target/source mask"
        ):
            self.handle.bind_runtime_route(
                native_branch=branch,
                actual_local_target_mask=wrong_local_mask,
                actual_sigma=torch.tensor(
                    saic.sigma_strata.PINNED_POSITIVE_SIGMAS[0],
                    dtype=torch.float32,
                ),
                arrow=_action_arrow(),
            )

    def test_route_bound_to_another_install_is_rejected(self) -> None:
        route = _route(handle=self.handle)
        other_model = _Transformer(hidden=8)
        other_model.requires_grad_(False)
        other_handle = saic.install_saic_typed_action_operator(other_model)
        try:
            with self.assertRaisesRegex(
                saic.SAICTypedActionOperatorError, "different installed"
            ):
                with other_handle.route(route):
                    pass
        finally:
            other_handle.restore()

    def test_gradient_checkpointing_fails_at_install_and_routed_runtime(self) -> None:
        enabled_model = _Transformer(hidden=8)
        enabled_model.requires_grad_(False)
        enabled_model.gradient_checkpointing = True
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "gradient checkpointing"
        ):
            saic.install_saic_typed_action_operator(enabled_model)

        route = _route(handle=self.handle)
        self.model.gradient_checkpointing = True
        with self.assertRaisesRegex(
            saic.SAICTypedActionOperatorError, "gradient checkpointing"
        ):
            with self.handle.route(route):
                pass
        # Even bypassing the handle context after entry cannot evade the
        # wrapper's before-forward runtime guard.
        self.model.gradient_checkpointing = False
        with self.handle.route(route):
            self.model.gradient_checkpointing = True
            with self.assertRaisesRegex(
                saic.SAICTypedActionOperatorError, "gradient checkpointing"
            ):
                self.handle.q_wrappers[0][1](torch.ones(1, 13, 8))
        self.model.gradient_checkpointing = False

    def test_install_rejects_unfrozen_or_wrong_structure(self) -> None:
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "freeze"):
            saic.install_saic_typed_action_operator(_Transformer())
        wrong = _Transformer(blocks=29)
        wrong.requires_grad_(False)
        with self.assertRaisesRegex(saic.SAICTypedActionOperatorError, "structure"):
            saic.install_saic_typed_action_operator(wrong)


if __name__ == "__main__":
    unittest.main()

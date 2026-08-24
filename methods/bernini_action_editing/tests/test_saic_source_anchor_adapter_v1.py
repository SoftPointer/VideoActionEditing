from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from torch import nn

    import inference_sigma_strata as strata
    import saic_source_anchor_adapter_v1 as anchor
    import source_self_native_ref_contrastive_v3 as native

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    strata = None  # type: ignore[assignment]
    anchor = None  # type: ignore[assignment]
    native = None  # type: ignore[assignment]
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
            self.gradient_checkpointing = False


    class _Transformer(nn.Module):
        def __init__(self, hidden: int = 8) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(16, hidden, kernel_size=(1, 2, 2))
            self.blocks = nn.ModuleList(
                [_Block(hidden) for _ in range(anchor.TOTAL_BLOCKS_1P3B)]
            )
            self.gradient_checkpointing = False
            self.is_gradient_checkpointing = False

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


    class UniPCMultistepScheduler:
        def __init__(self) -> None:
            self.config = {
                "_class_name": "UniPCMultistepScheduler",
                "num_train_timesteps": 1000,
                "flow_shift": 5.0,
                "prediction_type": "flow_prediction",
                "predict_x0": True,
                "use_flow_sigmas": True,
                "thresholding": False,
                "solver_order": 2,
                "solver_type": "bh2",
                "final_sigmas_type": "zero",
            }
            self.timesteps = torch.tensor(
                strata.PINNED_TIMESTEPS, dtype=torch.int64
            )
            self.sigmas = torch.tensor(
                (*strata.PINNED_POSITIVE_SIGMAS, 0.0), dtype=torch.float32
            )


    class _ParallelState:
        def __init__(self, rank: int = 0, size: int = 1) -> None:
            self.ulysses_rank = rank
            self.ulysses_size = size


def _branch(name: str = "V", *, hidden: int = 8):
    target_tokens = 21
    if name == "V":
        condition = 21
        source_ids = (1.0, 0.0)
    elif name == "VI":
        condition = 25
        source_ids = (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
    elif name == "I":
        condition = 4
        source_ids = (1.0, 2.0, 3.0, 4.0, 0.0)
    else:
        condition = 0
        source_ids = (0.0,)
    total = condition + target_tokens
    mask = torch.zeros(total, dtype=torch.bool)
    mask[condition:] = True
    return native.NativeRV2VBranch(
        name=name,
        latents=torch.zeros(1, total, hidden),
        rotary=torch.zeros(1, 1, total, 2),
        target_mask=mask,
        total_tokens=total,
        condition_tokens=condition,
        source_ids=source_ids,
        concat_order=native.BRANCH_CONCAT_ORDER[name],
    )


def _timestep(index: int) -> torch.Tensor:
    return torch.tensor(
        [float(strata.PINNED_TIMESTEPS[index])], dtype=torch.float32
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICSourceAnchorInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(809)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_q = tuple(block.attn1.to_q for block in self.model.blocks)
        self.original_o = tuple(block.attn1.to_out[0] for block in self.model.blocks)
        self.handle = anchor.install_saic_source_anchor_adapter(self.model)

    def tearDown(self) -> None:
        if not self.handle.restored and anchor.active_route() is None:
            self.handle.restore()

    @staticmethod
    def _make_nonzero(wrapper) -> None:
        with torch.no_grad():
            wrapper.state_down.weight.fill_(0.125)
            wrapper.output_up.weight.fill_(0.25)

    def test_scope_rank_fp32_zero_init_and_exact_install(self) -> None:
        self.assertEqual(anchor.SOURCE_ANCHOR_BLOCK_INDICES, tuple(range(23, 30)))
        self.assertEqual(anchor.SOURCE_ANCHOR_RANK, 8)
        for index in range(30):
            if index in anchor.SOURCE_ANCHOR_BLOCK_INDICES:
                self.assertIsInstance(
                    self.model.blocks[index].attn1.to_q,
                    anchor.SAICSourceAnchorResidual,
                )
                self.assertIsInstance(
                    self.model.blocks[index].attn1.to_out[0],
                    anchor.SAICSourceAnchorResidual,
                )
            else:
                self.assertIs(self.model.blocks[index].attn1.to_q, self.original_q[index])
                self.assertIs(self.model.blocks[index].attn1.to_out[0], self.original_o[index])
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            self.assertEqual(wrapper.rank, 8)
            self.assertIsNone(wrapper.state_down.bias)
            self.assertIsNone(wrapper.output_up.bias)
            self.assertEqual(wrapper.state_down.weight.dtype, torch.float32)
            self.assertEqual(wrapper.output_up.weight.dtype, torch.float32)
            self.assertEqual(int(torch.count_nonzero(wrapper.output_up.weight)), 0)
            hidden = torch.randn(1, 42, 8)
            self.assertTrue(torch.equal(wrapper(hidden), wrapper.base(hidden)))
        receipt = self.handle.receipt()
        self.assertEqual(receipt["blocks"], list(range(23, 30)))
        self.assertEqual(receipt["projections"], ["attn1.to_q", "attn1.to_out.0"])
        self.assertFalse(receipt["route_accepts_caller_rank_size_index_or_mask"])
        self.assertFalse(receipt["training_authorized"])
        self.assertFalse(receipt["semantic_action_success_claim"])

    def test_target_only_source_reference_and_padding_are_byte_exact(self) -> None:
        scheduler = UniPCMultistepScheduler()
        cases = (
            ("V", 1, 4),   # source rows plus first target row
            ("VI", 2, 4),  # final reference row plus target rows
            ("V", 3, 4),   # target rows plus two append-padding rows
        )
        for branch_name, rank, size in cases:
            with self.subTest(branch=branch_name, rank=rank):
                branch = _branch(branch_name)
                state = _ParallelState(rank, size)
                wrapper = self.handle.q_wrappers[0][1]
                self._make_nonzero(wrapper)
                local_length = (branch.total_tokens + size - 1) // size
                hidden = torch.ones(1, local_length, 8)
                expected = wrapper.base(hidden)
                with mock.patch.object(
                    anchor, "_get_live_parallel_state", return_value=state
                ):
                    with self.handle.route(
                        branch=branch,
                        scheduler=scheduler,
                        timestep=_timestep(35),
                    ) as route:
                        selector = route.local_target_mask.clone()
                        actual = wrapper(hidden)
                self.assertTrue(
                    torch.equal(actual[:, ~selector, :], expected[:, ~selector, :])
                )
                self.assertGreater(int(selector.sum()), 0)
                self.assertGreater(
                    float((actual[:, selector, :] - expected[:, selector, :]).abs().sum()),
                    0.0,
                )

    def test_actual_sigma_route_is_active_only_35_through_39(self) -> None:
        scheduler = UniPCMultistepScheduler()
        branch = _branch("V")
        state = _ParallelState()
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, branch.total_tokens, 8)
        base = wrapper.base(hidden)
        with mock.patch.object(anchor, "_get_live_parallel_state", return_value=state):
            for index in (0, 34):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(index)
                ) as route:
                    self.assertFalse(route.adapter_active)
                    self.assertTrue(torch.equal(wrapper(hidden), base))
            for index in anchor.ACTIVE_SIGMA_INDICES:
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(index)
                ) as route:
                    self.assertTrue(route.adapter_active)
                    self.assertFalse(torch.equal(wrapper(hidden), base))

    def test_wrong_source_mask_branch_and_live_route_fail_closed(self) -> None:
        scheduler = UniPCMultistepScheduler()
        state = _ParallelState()
        with mock.patch.object(anchor, "_get_live_parallel_state", return_value=state):
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "only full-source"):
                with self.handle.route(
                    branch=_branch("I"), scheduler=scheduler, timestep=_timestep(35)
                ):
                    pass

            bad = _branch("V")
            bad.target_mask[0] = True
            bad.target_mask[-1] = False
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "target mask"):
                with self.handle.route(
                    branch=bad, scheduler=scheduler, timestep=_timestep(35)
                ):
                    pass

            branch = _branch("V")
            hidden = torch.ones(1, branch.total_tokens, 8)
            wrapper = self.handle.q_wrappers[0][1]
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "target mask"):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(35)
                ):
                    branch.target_mask[0] = True
                    wrapper(hidden)
            branch.target_mask[0] = False

        first = _ParallelState()
        second = _ParallelState()
        branch = _branch("V")
        with mock.patch.object(
            anchor,
            "_get_live_parallel_state",
            side_effect=[first, first, first, second, second],
        ):
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "Ulysses route"):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(35)
                ):
                    self.handle.q_wrappers[0][1](
                        torch.ones(1, branch.total_tokens, 8)
                    )

    def test_scheduler_and_hidden_route_binding_reject_drift(self) -> None:
        scheduler = UniPCMultistepScheduler()
        branch = _branch("V")
        state = _ParallelState()
        wrapper = self.handle.q_wrappers[0][1]
        with mock.patch.object(anchor, "_get_live_parallel_state", return_value=state):
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "hidden sequence"):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(35)
                ):
                    wrapper(torch.ones(1, branch.total_tokens - 1, 8))
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "scheduler sigmas"):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(35)
                ):
                    scheduler.sigmas[35].add_(0.01)
                    wrapper(torch.ones(1, branch.total_tokens, 8))

    def test_gradient_checkpointing_is_detected_at_install_and_live_forward(self) -> None:
        bad_model = _Transformer()
        bad_model.requires_grad_(False)
        bad_model.gradient_checkpointing = True
        with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "checkpointing"):
            anchor.install_saic_source_anchor_adapter(bad_model)

        scheduler = UniPCMultistepScheduler()
        branch = _branch("V")
        state = _ParallelState()
        with mock.patch.object(anchor, "_get_live_parallel_state", return_value=state):
            with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "checkpointing"):
                with self.handle.route(
                    branch=branch, scheduler=scheduler, timestep=_timestep(35)
                ):
                    self.model.blocks[29].gradient_checkpointing = True
                    self.handle.q_wrappers[0][1](
                        torch.ones(1, branch.total_tokens, 8)
                    )
        self.model.blocks[29].gradient_checkpointing = False

    def test_all_adapter_parameters_receive_gradients_and_base_remains_frozen(self) -> None:
        scheduler = UniPCMultistepScheduler()
        branch = _branch("VI")
        state = _ParallelState()
        hidden = torch.ones(1, branch.total_tokens, 8)
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            self._make_nonzero(wrapper)
        loss = torch.zeros((), dtype=torch.float32)
        with mock.patch.object(anchor, "_get_live_parallel_state", return_value=state):
            with self.handle.route(
                branch=branch, scheduler=scheduler, timestep=_timestep(35)
            ):
                for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
                    loss = loss + wrapper(hidden).sum()
                loss.backward()
        named = self.handle.trainable_named_parameters()
        self.assertEqual(len(named), 28)
        for name, parameter in named:
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)
        base_parameters = [
            parameter
            for module in self.original_q + self.original_o
            for parameter in module.parameters()
        ]
        self.assertTrue(all(parameter.grad is None for parameter in base_parameters))

    def test_closed_state_save_load_and_extra_key_rejection(self) -> None:
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            self._make_nonzero(wrapper)
        expected = dict(self.handle.state_dict_for_save())
        digest = anchor.trainable_state_digest(expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_anchor.pt"
            saved = self.handle.save_checkpoint(path)
            self.assertEqual(saved["state_tensor_sha256"], digest)
            with torch.no_grad():
                for _, parameter in self.handle.trainable_named_parameters():
                    parameter.zero_()
            loaded = self.handle.load_checkpoint(path)
            self.assertEqual(loaded["state_tensor_sha256"], digest)
        actual = self.handle.state_dict_for_save()
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)
        extra = dict(expected)
        extra["unexpected.weight"] = torch.zeros(1, dtype=torch.float32)
        with self.assertRaisesRegex(anchor.SAICSourceAnchorError, "key closure"):
            self.handle.load_trainable_state_dict(extra)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICSourceAnchorImportTests(unittest.TestCase):
    def test_direct_and_namespace_package_imports_are_compatible(self) -> None:
        packaged = importlib.import_module(
            "methods.bernini_action_editing.saic_source_anchor_adapter_v1"
        )
        self.assertEqual(packaged.SCHEMA_VERSION, anchor.SCHEMA_VERSION)
        self.assertEqual(
            packaged.SOURCE_ANCHOR_BLOCK_INDICES,
            anchor.SOURCE_ANCHOR_BLOCK_INDICES,
        )
        self.assertEqual(packaged.ACTIVE_SIGMA_INDICES, tuple(range(35, 40)))


if __name__ == "__main__":
    unittest.main()

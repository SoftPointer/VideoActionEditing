from __future__ import annotations

import inspect
from pathlib import Path
import sys
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
    import saic_online_motion_field_v1 as motion
    import saic_temporal_action_operator_v2 as operator
    import source_self_native_ref_contrastive_v3 as native

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    strata = None  # type: ignore[assignment]
    motion = None  # type: ignore[assignment]
    operator = None  # type: ignore[assignment]
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
                [_Block(hidden) for _ in range(operator.TOTAL_BLOCKS_1P3B)]
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


def _timestep(index: int) -> "torch.Tensor":
    return torch.tensor(
        [float(strata.PINNED_TIMESTEPS[index])], dtype=torch.float32
    )


def _state(*, spatial: int = 2) -> "torch.Tensor":
    return torch.linspace(
        -1.0,
        1.0,
        16 * 21 * spatial * spatial,
        dtype=torch.float32,
    ).reshape(1, 16, 21, spatial, spatial)


def _branch(
    current_state: "torch.Tensor", *, name: str = "V", hidden: int = 8
) -> "native.NativeRV2VBranch":
    height, width = map(int, current_state.shape[-2:])
    patch_positions = (height // 2) * (width // 2)
    target_tokens = 21 * patch_positions
    if name == "V":
        condition = target_tokens
        source_ids = (1.0, 0.0)
    elif name == "VI":
        condition = target_tokens + 4 * patch_positions
        source_ids = (1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
    elif name == "I":
        condition = 4 * patch_positions
        source_ids = (1.0, 2.0, 3.0, 4.0, 0.0)
    else:
        condition = 0
        source_ids = (0.0,)
    total = condition + target_tokens
    target_mask = torch.zeros(total, dtype=torch.bool)
    target_mask[condition:] = True
    return native.NativeRV2VBranch(
        name=name,
        latents=torch.zeros(1, total, hidden),
        rotary=torch.zeros(1, 1, total, 2),
        target_mask=target_mask,
        total_tokens=total,
        condition_tokens=condition,
        source_ids=source_ids,
        concat_order=native.BRANCH_CONCAT_ORDER[name],
    )


def _captions() -> tuple[str, str]:
    return (
        "A brown dog smoothly lowers its body and sits on the floor.",
        "A brown dog remains standing still in the same place.",
    )


def _motion_field(
    current_state: "torch.Tensor",
    scheduler: "UniPCMultistepScheduler",
    timestep: "torch.Tensor",
    *,
    noop: bool = False,
) -> "motion.SAICOnlineMotionField":
    action_prompt, noop_prompt = _captions()
    teacher_noop = torch.zeros_like(current_state)
    teacher_action = torch.zeros_like(current_state)
    teacher_action[:, :, 0, 0, 0] = 4.0
    teacher_action[:, :, 7, -1, -1] = -2.0

    def callback(request):
        return teacher_action if request.branch == "action" else teacher_noop

    return motion.build_online_motion_field(
        current_noisy_target=current_state,
        action_prompt=noop_prompt if noop else action_prompt,
        noop_prompt=noop_prompt,
        scheduler=scheduler,
        timestep=timestep,
        frozen_t2v_velocity=callback,
    )


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class SAICTemporalActionOperatorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(810)
        self.model = _Transformer(hidden=8)
        self.model.requires_grad_(False)
        self.original_patch = self.model.patch_embedding
        self.original_q = tuple(block.attn2.to_q for block in self.model.blocks)
        self.original_o = tuple(block.attn2.to_out[0] for block in self.model.blocks)
        self.original_kv = tuple(
            (block.attn2.to_k, block.attn2.to_v) for block in self.model.blocks
        )
        self.original_attn1 = tuple(block.attn1 for block in self.model.blocks)
        self.handle = operator.install_saic_temporal_action_operator(self.model)

    def tearDown(self) -> None:
        if not self.handle.restored and operator.active_route() is None:
            self.handle.restore()

    @staticmethod
    def _make_nonzero(wrapper) -> None:
        with torch.no_grad():
            wrapper.state_down.weight.fill_(0.125)
            wrapper.phase_gate.weight.fill_(0.125)
            wrapper.output_up.weight.fill_(0.25)

    def test_exact_early_cross_qo_rank8_scope_and_zero_init(self) -> None:
        self.assertEqual(operator.ACTION_BLOCK_INDICES, tuple(range(23)))
        self.assertEqual(operator.ACTION_OPERATOR_RANK, 8)
        for index in range(30):
            if index < 23:
                self.assertIsInstance(
                    self.model.blocks[index].attn2.to_q,
                    operator.SAICTemporalActionResidual,
                )
                self.assertIsInstance(
                    self.model.blocks[index].attn2.to_out[0],
                    operator.SAICTemporalActionResidual,
                )
            else:
                self.assertIs(self.model.blocks[index].attn2.to_q, self.original_q[index])
                self.assertIs(self.model.blocks[index].attn2.to_out[0], self.original_o[index])
            self.assertIs(self.model.blocks[index].attn1, self.original_attn1[index])
            self.assertIs(self.model.blocks[index].attn2.to_k, self.original_kv[index][0])
            self.assertIs(self.model.blocks[index].attn2.to_v, self.original_kv[index][1])
        for _, wrapper in self.handle.q_wrappers + self.handle.o_wrappers:
            self.assertEqual(wrapper.rank, 8)
            self.assertIsNone(wrapper.state_down.bias)
            self.assertIsNone(wrapper.phase_gate.bias)
            self.assertIsNone(wrapper.output_up.bias)
            self.assertEqual(wrapper.state_down.weight.dtype, torch.float32)
            self.assertEqual(wrapper.phase_gate.weight.dtype, torch.float32)
            self.assertEqual(wrapper.output_up.weight.dtype, torch.float32)
            self.assertEqual(int(torch.count_nonzero(wrapper.output_up.weight)), 0)
        receipt = self.handle.receipt()
        self.assertEqual(receipt["blocks"], list(range(23)))
        self.assertEqual(receipt["projections"], ["attn2.to_q", "attn2.to_out.0"])
        self.assertFalse(receipt["route_accepts_mask_rank_size_index_sigma_phase_or_code"])
        self.assertTrue(receipt["training_and_inference_route_identical"])
        self.assertFalse(receipt["training_authorized"])

    def test_native_source_target_phase_and_append_padding_routing(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(0)
        field = _motion_field(state, scheduler, timestep)
        branch = _branch(state, name="V")  # 42 global rows -> 44 SP-padded rows.
        parallel = _ParallelState(rank=3, size=4)
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, 11, 8)
        expected = wrapper.base(hidden)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.handle.route(
                branch=branch,
                scheduler=scheduler,
                timestep=timestep,
                current_noisy_target=state,
                motion_field=field,
            ) as route:
                selector = route.local_target_mask.clone()
                phases = route.local_phase_indices.clone()
                actual = wrapper(hidden)
        self.assertEqual(selector.tolist(), [True] * 9 + [False] * 2)
        self.assertEqual(phases.tolist(), list(range(12, 21)) + [-1, -1])
        self.assertTrue(torch.equal(actual[:, ~selector, :], expected[:, ~selector, :]))
        self.assertGreater(
            float((actual[:, selector, :] - expected[:, selector, :]).abs().sum()),
            0.0,
        )

    def test_source_rows_are_exact_base_and_different_phases_get_different_delta(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(10)
        field = _motion_field(state, scheduler, timestep)
        branch = _branch(state, name="V")
        parallel = _ParallelState(rank=0, size=1)
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.ones(1, branch.total_tokens, 8)
        expected = wrapper.base(hidden)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.handle.route(
                branch=branch,
                scheduler=scheduler,
                timestep=timestep,
                current_noisy_target=state,
                motion_field=field,
            ) as route:
                actual = wrapper(hidden)
                delta = wrapper.adapter_delta(hidden)
        self.assertTrue(
            torch.equal(
                actual[:, : branch.condition_tokens, :],
                expected[:, : branch.condition_tokens, :],
            )
        )
        self.assertEqual(
            int(torch.count_nonzero(delta[:, : branch.condition_tokens, :])), 0
        )
        target_delta = delta[:, route.local_target_mask, :]
        self.assertGreater(float(target_delta.abs().sum()), 0.0)
        self.assertGreater(int(torch.unique(target_delta, dim=1).shape[1]), 1)

    def test_noop_and_indices_38_39_direct_base_even_with_nan_adapter(self) -> None:
        state = _state(spatial=2)
        branch = _branch(state)
        wrapper = self.handle.q_wrappers[0][1]
        hidden = torch.randn(1, branch.total_tokens, 8)
        expected = wrapper.base(hidden)
        with torch.no_grad():
            wrapper.state_down.weight.fill_(float("nan"))
            wrapper.phase_gate.weight.fill_(float("nan"))
            wrapper.output_up.weight.fill_(float("nan"))
        parallel = _ParallelState()
        for index, as_noop in ((0, True), (38, False), (39, False)):
            scheduler = UniPCMultistepScheduler()
            timestep = _timestep(index)
            field = _motion_field(state, scheduler, timestep, noop=as_noop)
            with self.subTest(index=index, noop=as_noop):
                with mock.patch.object(
                    operator, "_get_live_parallel_state", return_value=parallel
                ):
                    with self.handle.route(
                        branch=branch,
                        scheduler=scheduler,
                        timestep=timestep,
                        current_noisy_target=state,
                        motion_field=field,
                    ) as route:
                        actual = wrapper(hidden)
                        self.assertFalse(route.operator_active)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.isfinite(actual).all())

    def test_training_and_inference_share_route_and_only_operator_gets_gradients(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(33)
        field = _motion_field(state, scheduler, timestep)
        branch = _branch(state, name="VI")
        parallel = _ParallelState()
        wrapper = self.handle.q_wrappers[0][1]
        self._make_nonzero(wrapper)
        hidden = torch.randn(1, branch.total_tokens, 8)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.handle.route(
                branch=branch,
                scheduler=scheduler,
                timestep=timestep,
                current_noisy_target=state,
                motion_field=field,
            ):
                with torch.no_grad():
                    inference_output = wrapper(hidden)
                training_output = wrapper(hidden)
                training_output.float().square().mean().backward()
        self.assertTrue(torch.equal(inference_output, training_output.detach()))
        self.assertIsNotNone(wrapper.state_down.weight.grad)
        self.assertIsNotNone(wrapper.phase_gate.weight.grad)
        self.assertIsNotNone(wrapper.output_up.weight.grad)
        self.assertFalse(field.phase_code.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.original_q[0].parameters()))
        self.assertTrue(self.handle.base_parameters_frozen())

    def test_wrong_state_object_and_runtime_drift_fail_before_execution(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(0)
        field = _motion_field(state, scheduler, timestep)
        branch = _branch(state)
        parallel = _ParallelState()
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "target changed"):
                with self.handle.route(
                    branch=branch,
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state.clone(),
                    motion_field=field,
                ):
                    pass

        wrapper = self.handle.q_wrappers[0][1]
        hidden = torch.ones(1, branch.total_tokens, 8)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "Ulysses"):
                with self.handle.route(
                    branch=branch,
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state,
                    motion_field=field,
                ):
                    parallel.ulysses_size = 4
                    wrapper(hidden)

    def test_native_mask_and_scheduler_mutation_fail_closed(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(0)
        field = _motion_field(state, scheduler, timestep)
        branch = _branch(state)
        parallel = _ParallelState()
        wrapper = self.handle.q_wrappers[0][1]
        hidden = torch.ones(1, branch.total_tokens, 8)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "mask"):
                with self.handle.route(
                    branch=branch,
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state,
                    motion_field=field,
                ):
                    branch.target_mask.add_(False)
                    wrapper(hidden)

        branch = _branch(state)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(0)
        field = _motion_field(state, scheduler, timestep)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "sigmas changed"):
                with self.handle.route(
                    branch=branch,
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state,
                    motion_field=field,
                ):
                    scheduler.sigmas.add_(0.0)
                    wrapper(hidden)

    def test_non_source_branch_gradient_checkpointing_and_wrong_geometry_fail(self) -> None:
        state = _state(spatial=2)
        scheduler = UniPCMultistepScheduler()
        timestep = _timestep(0)
        field = _motion_field(state, scheduler, timestep)
        parallel = _ParallelState()
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "full-source"):
                with self.handle.route(
                    branch=_branch(state, name="I"),
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state,
                    motion_field=field,
                ):
                    pass
            self.model.gradient_checkpointing = True
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "checkpointing"):
                with self.handle.route(
                    branch=_branch(state),
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=state,
                    motion_field=field,
                ):
                    pass
            self.model.gradient_checkpointing = False

        larger = _state(spatial=4)
        wrong_branch = _branch(state)
        larger_field = _motion_field(larger, scheduler, timestep)
        with mock.patch.object(
            operator, "_get_live_parallel_state", return_value=parallel
        ):
            with self.assertRaisesRegex(operator.SAICTemporalActionOperatorError, "patch geometry"):
                with self.handle.route(
                    branch=wrong_branch,
                    scheduler=scheduler,
                    timestep=timestep,
                    current_noisy_target=larger,
                    motion_field=larger_field,
                ):
                    pass

    def test_route_surface_cannot_accept_self_reported_or_offline_guidance(self) -> None:
        forbidden = {
            "action_id",
            "mask",
            "target_mask",
            "source_mask",
            "pose",
            "flow",
            "track",
            "trajectory",
            "t2v_rgb",
            "t2v_latent",
            "t2v_noise",
            "proposal",
            "rank",
            "sequence_parallel_rank",
            "sequence_parallel_size",
            "schedule_index",
            "sigma",
            "phase_code",
            "phase_indices",
        }
        route_parameters = set(
            inspect.signature(operator.SAICTemporalActionOperatorHandle.route).parameters
        )
        self.assertTrue(forbidden.isdisjoint(route_parameters))
        self.assertEqual(
            route_parameters,
            {
                "self",
                "branch",
                "scheduler",
                "timestep",
                "current_noisy_target",
                "motion_field",
            },
        )

    def test_restore_recovers_exact_native_projection_objects(self) -> None:
        self.handle.restore()
        for index in range(30):
            self.assertIs(self.model.blocks[index].attn2.to_q, self.original_q[index])
            self.assertIs(self.model.blocks[index].attn2.to_out[0], self.original_o[index])
        self.assertIs(self.model.patch_embedding, self.original_patch)


if __name__ == "__main__":
    unittest.main()

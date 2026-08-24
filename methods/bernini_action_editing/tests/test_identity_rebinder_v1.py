from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = METHOD_ROOT / "identity_rebinder_v1.py"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

try:
    import torch
    from torch import nn

    import identity_rebinder_v1 as rebinder

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    rebinder = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class IdentityRebinderStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CORE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_pinned_bernini_insertion_and_closed_scope_are_explicit(self) -> None:
        for fragment in (
            "TOTAL_BLOCKS_1P3B = 30",
            "HIDDEN_SIZE_1P3B = 1536",
            "DEFAULT_BLOCK_INDICES = tuple(range(8, 23))",
            'PINNED_TRANSFORMER_CLASS_MODULE = "bernini.models.transformer_wan"',
            '"in_channels": 16',
            '"num_layers": TOTAL_BLOCKS_1P3B',
            'insertion": "blocks[i].attn1.to_out[0]"',
            "blocks[index].attn1.to_out[0] = wrapper",
            "not callable(getattr(transformer, \"patch_vae_latent\", None))",
            '"patch_embedding_untouched"',
            '"gpu_validated": False',
        ):
            self.assertIn(fragment, self.source)

    def test_identity_path_discloses_order_removed_but_content_leakage_remains(self) -> None:
        for fragment in (
            '"frame_position_embedding": False',
            '"spatial_position_embedding": False',
            '"frame_boundary_marker": False',
            '"temporal_convolution": False',
            '"frame_difference": False',
            '"flow_pose_track_mask": False',
            '"static_pose_leakage_information_theoretically_excluded": False',
            '"appearance_multiplicity_or_dwell_time_leakage_excluded": False',
            '"permutation_invariant_but_not_multiplicity_invariant": True',
        ):
            self.assertIn(fragment, self.source)
        self.assertNotIn("Conv3d(3,", self.source)
        self.assertNotIn("frame_position =", self.source)

    def test_memory_is_direct_target_query_write_and_V_VI_owned(self) -> None:
        for fragment in (
            'SOURCE_MEMORY_BRANCHES = {"V", "VI"}',
            'SOURCE_FREE_BRANCHES = {"none", "I"}',
            "target_queries = hidden_states[:, selector, :]",
            "key = self.key(memory.float())",
            "value = self.value(memory.float())",
            "result[:, selector, :] = delta.to(hidden_states.dtype)",
            '"condition_rows_written": False',
            '"direct_write_scope_only_later_layers_may_propagate": True',
            '"explicit_route_required": True',
        ):
            self.assertIn(fragment, self.source)

    def test_native_route_is_derived_from_objects_scheduler_and_parallel_state(self) -> None:
        for fragment in (
            "class NativeRV2VIdentityRouteBinder",
            "id(packet.latent_input)",
            "target_tokens = int(none_input.shape[1])",
            '"target_suffix_derived_from_none_branch_length": True',
            "_native_scheduler_sigma(self.scheduler, timestep)",
            "_native_parallel_coordinate()",
            'self.expected_calls = {"none": 1, "V": 1, "VI": 2}',
            'str(dist.get_backend(group)).lower() != "nccl"',
        ):
            self.assertIn(fragment, self.source)

    def test_raw_video_pretraining_terms_and_per_example_hinge_are_present(self) -> None:
        for fragment in (
            "correct_error_per_example",
            "wrong_error_per_example",
            "shuffled_atlas",
            "dropped_atlas",
            "resampled_atlas",
            "wrong_identity_ranking",
            "identity_contrast",
            '"memory_target_authority_clip_disjoint": True',
            '"action_labels": False',
            '"edited_targets": False',
        ):
            self.assertIn(fragment, self.source)


if _TORCH_AVAILABLE:
    class _Attention(nn.Module):
        def __init__(self, shared_projection: nn.Linear) -> None:
            super().__init__()
            self.to_out = nn.ModuleList([shared_projection, nn.Identity()])


    class _Block(nn.Module):
        def __init__(self, shared_projection: nn.Linear) -> None:
            super().__init__()
            self.attn1 = _Attention(shared_projection)


    class WanTransformer3DModel(nn.Module):
        """Memory-light structural double of the pinned 1.3B class."""

        def __init__(self) -> None:
            super().__init__()
            hidden = rebinder.HIDDEN_SIZE_1P3B
            self.config = dict(rebinder.PINNED_TRANSFORMER_CONFIG)
            self.patch_embedding = nn.Conv3d(
                16,
                hidden,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2),
                bias=False,
            )
            # The structural test does not need 30 independent 1536x1536 base
            # matrices.  Sharing one frozen projection keeps the double small;
            # every installed adapter remains independent.
            shared_projection = nn.Linear(hidden, hidden, bias=False)
            self.blocks = nn.ModuleList(
                [
                    _Block(shared_projection)
                    for _ in range(rebinder.TOTAL_BLOCKS_1P3B)
                ]
            )

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


    WanTransformer3DModel.__module__ = rebinder.PINNED_TRANSFORMER_CLASS_MODULE


    class _BadTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = dict(rebinder.PINNED_TRANSFORMER_CONFIG)
            self.patch_embedding = nn.Conv3d(16, 8, kernel_size=(1, 2, 2))
            projection = nn.Linear(8, 8, bias=False)
            self.blocks = nn.ModuleList(
                [_Block(projection) for _ in range(rebinder.TOTAL_BLOCKS_1P3B)]
            )

        def patch_vae_latent(self, value: torch.Tensor, source_id: float):
            del source_id
            return value, value


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class IdentityAtlasDynamicTests(unittest.TestCase):
    def test_atlas_is_frame_shuffle_invariant(self) -> None:
        torch.manual_seed(9)
        encoder = rebinder.OrderlessIdentityAtlasEncoder(
            hidden_size=8, atlas_width=8, atlas_tokens=4, patch_size=4
        )
        frames = (
            torch.randn((1, 5, 3, 8, 8), dtype=torch.float32)
            .clamp(-1, 1)
            .contiguous()
        )
        shuffled = frames[:, torch.tensor([3, 0, 4, 1, 2]), :, :, :].contiguous()
        first = encoder(frames)
        second = encoder(shuffled)
        self.assertTrue(torch.allclose(first, second, atol=2e-6, rtol=2e-6))
        receipt = encoder.architecture_receipt()
        self.assertFalse(receipt["frame_position_embedding"])
        self.assertTrue(receipt["permutation_invariant_but_not_multiplicity_invariant"])

    def test_atlas_can_leak_pose_frequency_or_dwell_time_through_multiplicity(self) -> None:
        torch.manual_seed(17)
        encoder = rebinder.OrderlessIdentityAtlasEncoder(
            hidden_size=8, atlas_width=8, atlas_tokens=4, patch_size=4
        )
        first_pose = torch.full((1, 1, 3, 8, 8), -0.75, dtype=torch.float32)
        second_pose = torch.full((1, 1, 3, 8, 8), 0.75, dtype=torch.float32)
        aab = torch.cat((first_pose, first_pose, second_pose), dim=1).contiguous()
        abb = torch.cat((first_pose, second_pose, second_pose), dim=1).contiguous()
        difference = (encoder(aab) - encoder(abb)).abs().max().item()
        self.assertGreater(float(difference), 1.0e-6)
        self.assertFalse(
            encoder.build_atlas(aab, source_video_sha256="a" * 64)
            .receipt()["appearance_multiplicity_or_dwell_time_leakage_excluded"]
        )

    def test_sigma_gate_is_high_off_mid_smooth_low_on(self) -> None:
        self.assertEqual(rebinder.mid_low_sigma_gate(1.0), 0.0)
        self.assertEqual(rebinder.mid_low_sigma_gate(0.75), 0.0)
        self.assertEqual(rebinder.mid_low_sigma_gate(0.25), 1.0)
        self.assertEqual(rebinder.mid_low_sigma_gate(0.0), 1.0)
        mid = rebinder.mid_low_sigma_gate(0.5)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)

    def test_source_free_branch_rejects_memory(self) -> None:
        tokens = torch.zeros((1, 4, 8), dtype=torch.float32).contiguous()
        atlas = rebinder.IdentityAtlas(tokens, "a" * 64, 81, "b" * 64)
        with self.assertRaisesRegex(rebinder.IdentityRebinderContractError, "must not"):
            rebinder.IdentityRebinderRoute(8, 0, 0, 1, "I", 0.2, atlas)


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class IdentityRebinderAdapterDynamicTests(unittest.TestCase):
    @staticmethod
    def _install(model: nn.Module):
        model.requires_grad_(False)
        return rebinder.install_identity_rebinder_v1(
            model,
            runtime_source_commit=rebinder.PINNED_BERNINI_SOURCE_COMMIT,
            model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
            checkpoint_manifest_sha256="c" * 64,
            rank=4,
            alpha=4.0,
            atlas_width=8,
            atlas_tokens=4,
            atlas_patch_size=4,
        )

    def _installed(self):
        model = WanTransformer3DModel()
        handle = self._install(model)
        frames = (
            torch.randn((1, 5, 3, 8, 8), dtype=torch.float32)
            .clamp(-1, 1)
            .contiguous()
        )
        atlas = handle.build_atlas(frames, source_video_sha256="a" * 64)
        return model, handle, atlas, frames

    def test_strict_installer_rejects_arbitrary_wan_like_model_and_wrong_pin(self) -> None:
        bad = _BadTransformer().requires_grad_(False)
        with self.assertRaisesRegex(rebinder.IdentityRebinderContractError, "class"):
            self._install(bad)
        pinned = WanTransformer3DModel().requires_grad_(False)
        with self.assertRaisesRegex(rebinder.IdentityRebinderContractError, "commit"):
            rebinder.install_identity_rebinder_v1(
                pinned,
                runtime_source_commit="0" * 40,
                model_revision=rebinder.PINNED_BERNINI_MODEL_REVISION,
                checkpoint_manifest_sha256="c" * 64,
            )

    def test_zero_init_is_exact_base_and_install_is_reversible(self) -> None:
        model, handle, atlas, _ = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]
        self.assertIsInstance(wrapper, rebinder.TargetQueryIdentityCrossAttention)
        hidden = torch.randn((1, 8, rebinder.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        route = rebinder.IdentityRebinderRoute(8, 3, 0, 1, "V", 0.2, atlas)
        expected = wrapper.base(hidden)
        with handle.route(route):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, expected))
        receipt = handle.receipt()
        self.assertEqual(
            receipt["runtime_source_commit"], rebinder.PINNED_BERNINI_SOURCE_COMMIT
        )
        self.assertTrue(receipt["base_parameters_frozen"])
        self.assertTrue(receipt["patch_embedding_untouched"])
        original = wrapper.base
        handle.restore()
        self.assertIs(model.blocks[8].attn1.to_out[0], original)

    def test_missing_explicit_route_fails_closed(self) -> None:
        model, handle, _, _ = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]
        hidden = torch.randn((1, 4, rebinder.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        with self.assertRaisesRegex(rebinder.IdentityRebinderContractError, "authenticated"):
            wrapper(hidden)
        handle.restore()

    def test_first_update_only_output_then_second_update_reaches_qkv_and_encoder(self) -> None:
        torch.manual_seed(23)
        model, handle, atlas, frames = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]
        hidden = torch.randn((1, 8, rebinder.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        route = rebinder.IdentityRebinderRoute(8, 3, 0, 1, "V", 0.2, atlas)
        optimizer = torch.optim.SGD(
            [parameter for _, parameter in handle.trainable_named_parameters()], lr=0.5
        )
        with handle.route(route):
            wrapper(hidden).square().mean().backward()

        def grad_sum(parameter: torch.Tensor) -> float:
            if parameter.grad is None:
                return 0.0
            return float(parameter.grad.abs().sum().item())

        self.assertGreater(grad_sum(wrapper.output.weight), 0.0)
        for projection in (wrapper.query, wrapper.key, wrapper.value):
            self.assertEqual(grad_sum(projection.weight), 0.0)
        self.assertEqual(
            sum(grad_sum(parameter) for parameter in handle.atlas_encoder.parameters()),
            0.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        second_atlas = handle.build_atlas(frames, source_video_sha256="a" * 64)
        second_route = rebinder.IdentityRebinderRoute(
            8, 3, 0, 1, "V", 0.2, second_atlas
        )
        with handle.route(second_route):
            wrapper(hidden).square().mean().backward()
        for projection in (wrapper.query, wrapper.key, wrapper.value):
            self.assertGreater(grad_sum(projection.weight), 0.0)
        self.assertGreater(
            sum(grad_sum(parameter) for parameter in handle.atlas_encoder.parameters()),
            0.0,
        )
        handle.restore()

    def test_trained_residual_directly_writes_only_target_rows(self) -> None:
        model, handle, atlas, _ = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]
        nn.init.ones_(wrapper.output.weight)
        hidden = torch.randn((1, 8, rebinder.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        route = rebinder.IdentityRebinderRoute(8, 3, 0, 1, "VI", 0.2, atlas)
        with handle.route(route):
            delta = wrapper.adapter_delta(hidden)
        self.assertTrue(torch.equal(delta[:, :3], torch.zeros_like(delta[:, :3])))
        self.assertGreater(float(delta[:, 3:].abs().sum().item()), 0.0)
        self.assertTrue(
            handle.receipt()["direct_write_scope_only_later_layers_may_propagate"]
        )
        handle.restore()

    def test_high_sigma_is_exact_base_after_nonzero_weights(self) -> None:
        model, handle, atlas, _ = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]
        nn.init.ones_(wrapper.output.weight)
        hidden = torch.randn((1, 8, rebinder.HIDDEN_SIZE_1P3B), dtype=torch.float32)
        route = rebinder.IdentityRebinderRoute(8, 3, 0, 1, "V", 0.9, atlas)
        expected = wrapper.base(hidden)
        with handle.route(route):
            actual = wrapper(hidden)
        self.assertTrue(torch.equal(actual, expected))
        handle.restore()

    def test_sp4_padding_never_becomes_a_target(self) -> None:
        selectors = []
        tokens = torch.zeros((1, 4, 8), dtype=torch.float32).contiguous()
        atlas = rebinder.IdentityAtlas(tokens, "a" * 64, 81, "b" * 64)
        for rank in range(4):
            route = rebinder.IdentityRebinderRoute(13, 5, rank, 4, "V", 0.2, atlas)
            selectors.append(route.local_target_selector(device=torch.device("cpu")))
        joined = torch.cat(selectors)
        self.assertTrue(
            torch.equal(joined[:13], torch.tensor([False] * 5 + [True] * 8))
        )
        self.assertFalse(bool(joined[13:].any().item()))

    def test_native_sp1_binder_authenticates_pack_inventory_and_routes(self) -> None:
        model, handle, atlas, _ = self._installed()
        wrapper = model.blocks[8].attn1.to_out[0]

        scheduler_type = type(rebinder.PINNED_SCHEDULER_CLASS_NAME, (), {})
        scheduler_type.__module__ = rebinder.PINNED_SCHEDULER_CLASS_MODULE
        scheduler = scheduler_type()
        scheduler.timesteps = torch.tensor([999, 500], dtype=torch.int64)
        scheduler.sigmas = torch.tensor([1.0, 0.2, 0.0], dtype=torch.float32)
        scheduler.step_index = 1

        parent_module = types.ModuleType("bernini")
        parent_module.__path__ = []
        parallel_module = types.ModuleType("bernini.parallel")
        state_type = type("ParallelState", (), {})
        state_type.__module__ = "bernini.parallel"
        state = state_type()
        state.ulysses_enabled = False
        state.ulysses_size = 1
        state.ulysses_rank = 0
        parallel_module.get_parallel_state = lambda: state
        parent_module.parallel = parallel_module
        previous_parent = sys.modules.get("bernini")
        previous_parallel = sys.modules.get("bernini.parallel")
        sys.modules["bernini"] = parent_module
        sys.modules["bernini.parallel"] = parallel_module
        try:
            none_input = torch.randn((1, 5, rebinder.HIDDEN_SIZE_1P3B)).contiguous()
            video_input = torch.randn((1, 7, rebinder.HIDDEN_SIZE_1P3B)).contiguous()
            vi_input = torch.randn((1, 6, rebinder.HIDDEN_SIZE_1P3B)).contiguous()
            binder = rebinder.NativeRV2VIdentityRouteBinder(
                handle=handle, scheduler=scheduler, atlas=atlas
            )
            kwargs = {
                "timestep": torch.tensor(500, dtype=torch.int64),
                "none_input": none_input,
                "video_input": video_input,
                "video_image_input": vi_input,
            }
            with binder.native_rv2v_step(**kwargs) as step:
                with self.assertRaisesRegex(
                    rebinder.IdentityRebinderContractError, "not minted"
                ):
                    with step.route_for_call(
                        latent_input=video_input.clone()
                    ):
                        pass
                for latent, expected_branch in (
                    (none_input, "none"),
                    (video_input, "V"),
                    (vi_input, "VI"),
                    (vi_input, "VI"),
                ):
                    with step.route_for_call(latent_input=latent) as route:
                        self.assertEqual(route.branch_name, expected_branch)
                        self.assertEqual(route.sequence_parallel_size, 1)
                        self.assertEqual(route.target_tokens, 5)
                        wrapper.adapter_delta(latent)
            self.assertTrue(step.closed)
            self.assertIs(step.receipt, binder.last_step_receipt)
            self.assertTrue(
                step.receipt["target_suffix_derived_from_none_branch_length"]
            )

            with self.assertRaisesRegex(
                rebinder.IdentityRebinderContractError, "inventory is incomplete"
            ):
                with binder.native_rv2v_step(**kwargs) as incomplete:
                    with incomplete.route_for_call(latent_input=none_input):
                        pass

            changed_video = video_input.clone().contiguous()
            changed_kwargs = dict(kwargs)
            changed_kwargs["video_input"] = changed_video
            with self.assertRaisesRegex(
                rebinder.IdentityRebinderContractError, "pack changed"
            ):
                with binder.native_rv2v_step(**changed_kwargs) as changed:
                    changed_video.add_(1.0)
                    with changed.route_for_call(latent_input=changed_video):
                        pass
        finally:
            if previous_parent is None:
                sys.modules.pop("bernini", None)
            else:
                sys.modules["bernini"] = previous_parent
            if previous_parallel is None:
                sys.modules.pop("bernini.parallel", None)
            else:
                sys.modules["bernini.parallel"] = previous_parallel
            handle.restore()


@unittest.skipUnless(_TORCH_AVAILABLE, "torch runtime is required")
class IdentityRebinderObjectiveDynamicTests(unittest.TestCase):
    def test_good_recovery_and_identity_views_have_zero_auxiliary_loss(self) -> None:
        target = torch.zeros((1, 3, 8), dtype=torch.float32)
        correct = target.clone().requires_grad_(True)
        wrong = torch.ones_like(target, requires_grad=True)
        atlas = torch.randn((1, 4, 8), dtype=torch.float32)
        wrong_atlas = -atlas
        loss = rebinder.identity_rebinder_pretrain_objective(
            correct_prediction=correct,
            wrong_prediction=wrong,
            target=target,
            canonical_atlas=atlas,
            shuffled_atlas=atlas.clone(),
            dropped_atlas=atlas.clone(),
            resampled_atlas=atlas.clone(),
            wrong_atlas=wrong_atlas,
        )
        self.assertEqual(float(loss.wrong_identity_ranking.item()), 0.0)
        self.assertEqual(float(loss.view_consistency.item()), 0.0)
        self.assertEqual(float(loss.identity_contrast.item()), 0.0)
        loss.total.backward()
        self.assertIsNotNone(correct.grad)

    def test_recovery_margin_is_hinged_per_example_before_batch_mean(self) -> None:
        target = torch.zeros((2, 1, 1), dtype=torch.float32)
        correct = torch.tensor([[[1.0]], [[0.0]]], dtype=torch.float32)
        wrong = torch.tensor([[[0.0]], [[2.0]]], dtype=torch.float32)
        atlas = torch.randn((2, 2, 3), dtype=torch.float32)
        loss = rebinder.identity_rebinder_pretrain_objective(
            correct_prediction=correct,
            wrong_prediction=wrong,
            target=target,
            canonical_atlas=atlas,
            shuffled_atlas=atlas.clone(),
            dropped_atlas=atlas.clone(),
            resampled_atlas=atlas.clone(),
            wrong_atlas=-atlas,
            recovery_rank_margin=0.05,
        )
        # Per-example hinges are [0.05 + 1 - 0, 0.05 + 0 - 4]_+.
        self.assertAlmostEqual(float(loss.wrong_identity_ranking.item()), 0.525, places=6)


if __name__ == "__main__":
    unittest.main()

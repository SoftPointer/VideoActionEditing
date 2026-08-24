from __future__ import annotations

from pathlib import Path
import sys
import unittest

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest("torch unavailable") from error


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import dense_flow_token_adapter_v1 as dense  # noqa: E402
import dense_flow_preservation_adapter_v1 as preservation  # noqa: E402
import dense_flow_source_copy_adapter_v1 as source_copy  # noqa: E402
import infer_same_video_dense_flow_adapter_v1 as infer_dense  # noqa: E402
import materialize_source_flow_fused_targets_v1 as fused  # noqa: E402
import train_same_video_dense_flow_adapter_v1 as train_dense  # noqa: E402


class _Block(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return hidden_states + 0.25


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(30)])


class DenseFlowFeatureTest(unittest.TestCase):
    def test_sga_anc_weights_are_stable_noncollapsing_softmin(self):
        weights = train_dense.sga_anc_weights(
            (0.10, 0.20, 0.30), temperature=0.02, uniform_mass=0.30
        )
        self.assertAlmostEqual(sum(weights), 1.0, places=12)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        self.assertGreaterEqual(min(weights), 0.10)
        uniform = train_dense.sga_anc_weights(
            (0.25, 0.25, 0.25), temperature=0.02, uniform_mass=0.25
        )
        for value in uniform:
            self.assertAlmostEqual(value, 1.0 / 3.0, places=12)

    def test_sga_anc_training_requires_large_real_training_scope(self):
        common = [
            "--bernini-root", "bernini",
            "--veomni-root", "veomni",
            "--checkpoint", "checkpoint",
            "--pair-manifest", "pairs.json",
            "--sga-anc-bank-manifest", "bank.json",
            "--output", "output",
            "--dense-flow-mode", "phase_attention_12x20",
            "--method-source-revision", "1" * 40,
            "--method-source-archive-sha256", "2" * 64,
        ]
        args = train_dense.parser().parse_args(common + ["--full-attention-lora"])
        train_dense.validate_args(args)
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(train_dense.parser().parse_args(common))

    def test_zero_motion_control_keeps_activity_but_removes_anchor_payload(self):
        record = {
            "features": torch.randn((1, 12, 12)),
            "activity": torch.ones((1, 12), dtype=torch.bool),
            "variant": "noop",
        }
        control = train_dense.zero_motion_control(record)
        self.assertEqual(torch.count_nonzero(control["features"]).item(), 0)
        self.assertTrue(torch.equal(control["activity"], record["activity"]))
        self.assertGreater(torch.count_nonzero(record["features"]).item(), 0)

    def test_fused_target_zero_flow_repeats_source_phase_zero(self):
        source = torch.randn((1, 16, 21, 4, 6), dtype=torch.float32)
        flow = torch.zeros((20, 2, 4, 6), dtype=torch.float32)
        target, cumulative = fused.warp_source_phase0(source, flow)
        expected = source[:, :, 0:1].expand_as(source).contiguous()
        torch.testing.assert_close(target, expected, rtol=0.0, atol=1.0e-5)
        self.assertEqual(tuple(cumulative.shape), (21, 2, 4, 6))
        self.assertEqual(torch.count_nonzero(cumulative).item(), 0)

    def test_fused_target_keeps_phase_zero_exact_under_nonzero_flow(self):
        source = torch.randn((1, 16, 21, 4, 6), dtype=torch.float32)
        flow = torch.randn((20, 2, 4, 6), dtype=torch.float32) * 0.25
        target, _ = fused.warp_source_phase0(source, flow)
        self.assertTrue(torch.equal(target[:, :, 0], source[:, :, 0]))
        self.assertFalse(torch.equal(target[:, :, -1], source[:, :, 0]))

    def test_rgb_warp_zero_flow_repeats_source_phase_zero(self):
        source = torch.randn((1, 3, 81, 32, 48), dtype=torch.float32)
        cumulative = torch.zeros((21, 2, 4, 6), dtype=torch.float32)
        target = fused.warp_source_rgb_phase0(source, cumulative)
        expected = source[:, :, 0:1].expand_as(source).contiguous()
        torch.testing.assert_close(target, expected, rtol=0.0, atol=1.0e-5)
        self.assertTrue(torch.equal(target[:, :, 0], source[:, :, 0]))

    def test_source_reconstruction_schedule_is_balanced_across_rows(self):
        observed = {row: [] for row in range(4)}
        for step in range(32):
            row = step % 4
            observed[row].append(
                train_dense.training_record_role(
                    global_step=step,
                    micro=0,
                    micro_records=1,
                    row_count=4,
                    source_reconstruction_every=4,
                )
            )
        expected = [
            "action",
            "action",
            "action",
            "source_reconstruction",
            "action",
            "action",
            "action",
            "source_reconstruction",
        ]
        self.assertEqual(observed, {row: expected for row in range(4)})

    def test_source_copy_allows_balanced_original_source_records(self):
        args = train_dense.parser().parse_args(
            [
                "--bernini-root", "bernini",
                "--veomni-root", "veomni",
                "--checkpoint", "base",
                "--pair-manifest", "pairs.json",
                "--output", "out",
                "--frozen-motion-checkpoint", "motion",
                "--source-copy-mode", "phase0_broadcast",
                "--source-reconstruction-every", "4",
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )
        train_dense.validate_args(args)

    def test_source_copy_block_subset_has_exact_parameter_closure(self):
        args = train_dense.parser().parse_args(
            [
                "--bernini-root", "bernini",
                "--veomni-root", "veomni",
                "--checkpoint", "checkpoint",
                "--pair-manifest", "pairs.json",
                "--output", "output",
                "--frozen-motion-checkpoint", "motion",
                "--source-copy-mode", "phase0_flowwarp_raw",
                "--source-copy-block-indices", "16,20,24,28",
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )
        train_dense.validate_args(args)
        self.assertEqual(
            train_dense.source_copy_block_indices(args), (16, 20, 24, 28)
        )
        self.assertEqual(
            source_copy.expected_trainable_parameters(
                "phase0_flowwarp_raw", block_count=4
            ),
            source_copy.EXPECTED_TRAINABLE_PARAMETERS // 2,
        )
        with self.assertRaises(train_dense.SameVideoTrainingError):
            args.source_copy_block_indices = "20,16"
            train_dense.source_copy_block_indices(args)

    def test_source_attention_can_train_over_a_frozen_joint_motion_checkpoint(self):
        args = train_dense.parser().parse_args(
            [
                "--bernini-root", "bernini",
                "--veomni-root", "veomni",
                "--checkpoint", "checkpoint",
                "--pair-manifest", "pairs.json",
                "--output", "output",
                "--frozen-motion-checkpoint", "joint-motion",
                "--dense-flow-mode", "phase_attention_12x20",
                "--source-copy-mode", "phase0_attention_12x20",
                "--source-copy-block-indices", ",".join(map(str, range(30))),
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )
        train_dense.validate_args(args)
        self.assertEqual(train_dense.source_copy_block_indices(args), tuple(range(30)))
        self.assertEqual(
            source_copy.expected_trainable_parameters(
                "phase0_attention_12x20", block_count=30
            ),
            source_copy.ATTENTION_TRAINABLE_PARAMETERS // 8 * 30,
        )
        args.source_attention_correspondence_weight = 0.02
        train_dense.validate_args(args)
        args.source_copy_initial_checkpoint = "source-attention-step10"
        args.source_attention_correspondence_only = True
        train_dense.validate_args(args)
        args.source_copy_initial_checkpoint = None
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(args)
        args.source_copy_initial_checkpoint = "source-attention-step10"
        args.source_copy_mode = "phase0_broadcast"
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(args)

    def test_motion_all_block_attention_capacity_is_explicit(self):
        args = train_dense.parser().parse_args(
            [
                "--bernini-root", "bernini",
                "--veomni-root", "veomni",
                "--checkpoint", "checkpoint",
                "--pair-manifest", "pairs.json",
                "--output", "output",
                "--dense-flow-mode", "phase_attention_12x20",
                "--adapter-block-indices", ",".join(map(str, range(30))),
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )
        train_dense.validate_args(args)
        self.assertEqual(train_dense.motion_block_indices(args), tuple(range(30)))
        self.assertEqual(
            dense.expected_trainable_parameters(
                "phase_attention_12x20", block_count=30
            ),
            dense.ATTENTION_TRAINABLE_PARAMETERS // 8 * 30,
        )
        args.adapter_block_indices = "0,2,1"
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.motion_block_indices(args)

    def test_sequence_parallel_materializes_only_dense_adapter_zeros(self):
        names = []
        for component in ("query", "key", "value", "output"):
            parameter = nn.Parameter(torch.ones((2, 2)))
            names.append(
                (f"blocks.0.dense_flow_adapter.{component}.weight", parameter)
            )
        names[-1][1].grad = torch.ones_like(names[-1][1])
        missing = train_dense.materialize_sequence_parallel_adapter_gradients(names)
        self.assertEqual(len(missing), 3)
        self.assertTrue(all(parameter.grad is not None for _, parameter in names))
        self.assertTrue(
            all(
                torch.count_nonzero(parameter.grad) == 0
                for _, parameter in names[:3]
            )
        )

        lora = nn.Parameter(torch.ones((2, 2)))
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.materialize_sequence_parallel_adapter_gradients(
                [("blocks.0.attn1.to_q.lora_A.default.weight", lora)]
            )

    def test_post_reduce_dense_component_coverage_enforces_warmup_then_all(self):
        names = []
        for component in ("query", "key", "value", "output"):
            parameter = nn.Parameter(torch.ones((2, 2)))
            parameter.grad = (
                torch.ones_like(parameter)
                if component == "output"
                else torch.zeros_like(parameter)
            )
            names.append(
                (f"blocks.0.dense_flow_adapter.{component}.weight", parameter)
            )
        warmup = train_dense.dense_flow_component_coverage(names)
        train_dense.assert_zero_init_output_only_active(
            warmup, expected_blocks=(0,)
        )
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.assert_dense_flow_components_active(
                warmup, expected_blocks=(0,)
            )

        for _, parameter in names:
            parameter.grad = torch.ones_like(parameter)
        active = train_dense.dense_flow_component_coverage(names)
        train_dense.assert_dense_flow_components_active(active, expected_blocks=(0,))
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.assert_zero_init_output_only_active(
                active, expected_blocks=(0,)
            )

    def test_training_timestep_limit_is_varying_and_bounded(self):
        args = train_dense.parser().parse_args(
            [
                "--bernini-root", "bernini",
                "--veomni-root", "veomni",
                "--checkpoint", "checkpoint",
                "--pair-manifest", "pairs.json",
                "--output", "output",
                "--training-max-timestep", "833",
                "--method-source-revision", "1" * 40,
                "--method-source-archive-sha256", "2" * 64,
            ]
        )
        train_dense.validate_args(args)
        args.training_noise_policy = "fixed"
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(args)
        args.training_noise_policy = "varying"
        args.training_max_timestep = 1001
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(args)
        args.training_max_timestep = 833
        args.training_min_timestep = 834
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.validate_args(args)

    def test_training_timestep_rejection_resamples_before_update(self):
        timesteps = {10: 1000.0, 11: 950.0, 12: 840.0, 13: 833.0}

        def builder(seed):
            return {"timestep": timesteps[seed], "seed": seed}

        seed, record, rejected = train_dense.select_training_record(
            base_seed=10,
            min_timestep=None,
            max_timestep=833,
            builder=builder,
        )
        self.assertEqual((seed, rejected), (13, 3))
        self.assertEqual(record, {"timestep": 833.0, "seed": 13})

    def test_training_timestep_band_rejects_both_ends(self):
        timesteps = {20: 1000.0, 21: 116.0, 22: 626.0, 23: 117.0}

        def builder(seed):
            return {"timestep": timesteps[seed], "seed": seed}

        seed, record, rejected = train_dense.select_training_record(
            base_seed=20,
            min_timestep=117,
            max_timestep=625,
            builder=builder,
        )
        self.assertEqual((seed, rejected), (23, 3))
        self.assertEqual(record, {"timestep": 117.0, "seed": 23})

    def test_mixed_source_schedule_covers_both_variants_for_every_row(self):
        observed = {row: set() for row in range(4)}
        for step in range(8):
            row = step % 4
            observed[row].add(
                train_dense.source_variant_for_step(
                    "mixed",
                    global_step=step,
                    micro=0,
                    micro_records=1,
                    row_count=4,
                )
            )
        self.assertEqual(
            observed,
            {row: {"noop", "incomplete"} for row in range(4)},
        )

    def test_complex32_micro2_covers_noop_and_incomplete_in_32_updates(self):
        observed = {row: set() for row in range(32)}
        for step in range(32):
            for micro in range(2):
                row = (step * 2 + micro) % 32
                observed[row].add(
                    train_dense.source_variant_for_step(
                        "mixed",
                        global_step=step,
                        micro=micro,
                        micro_records=2,
                        row_count=32,
                    )
                )
        self.assertEqual(
            observed,
            {row: {"noop", "incomplete"} for row in range(32)},
        )

    def test_row_repeat_schedule_oversamples_only_requested_row(self):
        schedule = train_dense.row_repeat_schedule("1,3,1,1")
        self.assertEqual(schedule, (0, 1, 1, 1, 2, 3))
        self.assertEqual({row: schedule.count(row) for row in range(4)}, {
            0: 1, 1: 3, 2: 1, 3: 1,
        })
        observed = {row: set() for row in range(4)}
        for step in range(2 * len(schedule)):
            row = schedule[step % len(schedule)]
            observed[row].add(
                train_dense.source_variant_for_step(
                    "mixed",
                    global_step=step,
                    micro=0,
                    micro_records=1,
                    row_count=len(schedule),
                )
            )
        self.assertEqual(
            observed,
            {row: {"noop", "incomplete"} for row in range(4)},
        )

    def test_row_repeat_schedule_rejects_unbounded_or_ambiguous_specs(self):
        for specification in ("1,1,1", "1,0,1,1", "1,9,1,1", "x,1,1,1"):
            with self.subTest(specification=specification):
                with self.assertRaises(train_dense.SameVideoTrainingError):
                    train_dense.row_repeat_schedule(specification)

    def test_row_repeat_schedule_supports_a_larger_balanced_bank(self):
        self.assertEqual(
            train_dense.row_repeat_schedule("auto", row_count=7),
            tuple(range(7)),
        )
        schedule = train_dense.row_repeat_schedule(
            "1,1,1,2,1,1,1", row_count=7
        )
        self.assertEqual(schedule, (0, 1, 2, 3, 3, 4, 5, 6))

    def test_row_repeat_schedule_requires_manifest_length_match(self):
        with self.assertRaises(train_dense.SameVideoTrainingError):
            train_dense.row_repeat_schedule("1,1,1,1", row_count=7)

    def test_full_patch_grid_and_exact_masks(self):
        raw = torch.zeros((20, 2, 4, 6), dtype=torch.float32)
        camera = torch.zeros_like(raw)
        validity = torch.ones((20, 1, 4, 6), dtype=torch.float32)
        raw[:, 0] = 1.0
        camera[:, 1] = -0.5
        features, activity = dense.dense_flow_features_from_tensors(
            raw, camera, validity
        )
        target_tokens = 21 * 2 * 3
        self.assertEqual(tuple(features.shape), (1, 2 * target_tokens, 12))
        self.assertEqual(tuple(activity.shape), (1, 2 * target_tokens, 1))
        self.assertEqual(torch.count_nonzero(features[:, :target_tokens]).item(), 0)
        self.assertFalse(bool(activity[:, :target_tokens].any().item()))
        self.assertFalse(bool(activity[:, target_tokens : target_tokens + 6].any().item()))
        self.assertTrue(bool(activity[:, target_tokens + 6 :].all().item()))

    def test_zero_validity_is_exact_no_condition(self):
        raw = torch.randn((20, 2, 4, 6), dtype=torch.float32)
        camera = torch.randn_like(raw)
        validity = torch.zeros((20, 1, 4, 6), dtype=torch.float32)
        _, activity = dense.dense_flow_features_from_tensors(raw, camera, validity)
        self.assertFalse(bool(activity.any().item()))


class DenseFlowPatchTest(unittest.TestCase):
    def setUp(self):
        raw = torch.zeros((20, 2, 4, 6), dtype=torch.float32)
        camera = torch.zeros_like(raw)
        validity = torch.ones((20, 1, 4, 6), dtype=torch.float32)
        raw[:, 0] = 1.0
        camera[:, 1] = 0.5
        self.features, self.activity = dense.dense_flow_features_from_tensors(
            raw, camera, validity
        )

    def test_zero_init_parity_then_target_only_effect(self):
        transformer = _Transformer()
        handle = dense.install_dense_flow_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        self.assertTrue(handle.base_is_frozen())
        self.assertTrue(handle.zero_effect())
        hidden = torch.randn((1, self.features.shape[1], 8))
        expected = hidden + 0.25
        invocation = dense.DenseFlowInvocation(self.features, self.activity)
        with dense.dense_flow_invocation(invocation):
            observed = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(observed, expected))

        with torch.no_grad():
            handle.adapters[0].output.weight.fill_(0.05)
        with dense.dense_flow_invocation(invocation):
            changed = transformer.blocks[0](hidden)
        target_tokens = self.features.shape[1] // 2
        phase0_end = target_tokens + 6
        self.assertTrue(torch.equal(changed[:, :phase0_end], expected[:, :phase0_end]))
        self.assertFalse(torch.equal(changed[:, phase0_end:], expected[:, phase0_end:]))

        handle.restore()
        restored = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(restored, expected))

    def test_zero_activity_remains_exact_after_training(self):
        transformer = _Transformer()
        handle = dense.install_dense_flow_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        with torch.no_grad():
            for _, parameter in handle.trainable_named_parameters():
                parameter.uniform_(-0.5, 0.5)
        hidden = torch.randn((1, self.features.shape[1], 8))
        invocation = dense.DenseFlowInvocation(
            self.features, torch.zeros_like(self.activity)
        )
        with dense.dense_flow_invocation(invocation):
            observed = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(observed, hidden + 0.25))

    def test_phase_attention_is_zero_init_then_query_dependent_target_only(self):
        transformer = _Transformer()
        handle = dense.install_dense_flow_adapter(
            transformer,
            mode="phase_attention_8x12",
            block_indices=(0,),
            hidden_width=8,
            bottleneck_width=4,
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        expected = hidden + 0.25
        invocation = dense.DenseFlowInvocation(
            self.features,
            self.activity,
            mode="phase_attention_8x12",
            spatial_shape=(2, 3),
        )
        with dense.dense_flow_invocation(invocation):
            zero = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(zero, expected))

        with torch.no_grad():
            handle.adapters[0].output.weight.fill_(0.05)
        with dense.dense_flow_invocation(invocation):
            changed = transformer.blocks[0](hidden)
        target_tokens = self.features.shape[1] // 2
        phase0_end = target_tokens + 6
        self.assertTrue(torch.equal(changed[:, :phase0_end], expected[:, :phase0_end]))
        self.assertFalse(torch.equal(changed[:, phase0_end:], expected[:, phase0_end:]))

        changed.float().sum().backward()
        names = dict(handle.trainable_named_parameters())
        for suffix in ("query.weight", "key.weight", "value.weight", "output.weight"):
            parameter = next(value for name, value in names.items() if name.endswith(suffix))
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_phase_attention_requires_exact_spatial_contract(self):
        with self.assertRaises(dense.DenseFlowAdapterError):
            dense.DenseFlowInvocation(
                self.features,
                self.activity,
                mode="phase_attention_12x20",
                spatial_shape=(3, 3),
            ).validate()
        self.assertEqual(
            dense.expected_trainable_parameters("phase_attention_8x12"),
            dense.ATTENTION_TRAINABLE_PARAMETERS,
        )

    def test_phase_attention_bfloat16_autocast_matches_attention_value_dtype(self):
        transformer = _Transformer()
        handle = dense.install_dense_flow_adapter(
            transformer,
            mode="phase_attention_8x12",
            block_indices=(0,),
            hidden_width=8,
            bottleneck_width=4,
        )
        with torch.no_grad():
            handle.adapters[0].output.weight.fill_(0.05)
        hidden = torch.randn((1, self.features.shape[1], 8))
        invocation = dense.DenseFlowInvocation(
            self.features,
            self.activity,
            mode="phase_attention_8x12",
            spatial_shape=(2, 3),
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with dense.dense_flow_invocation(invocation):
                observed = transformer.blocks[0](hidden)
        self.assertEqual(observed.dtype, hidden.dtype)
        self.assertTrue(torch.isfinite(observed).all())
        self.assertFalse(torch.equal(observed, hidden + 0.25))

    def test_strict_state_roundtrip(self):
        first = _Transformer()
        handle_first = dense.install_dense_flow_adapter(
            first, block_indices=(0, 4), hidden_width=8, bottleneck_width=4
        )
        with torch.no_grad():
            for _, parameter in handle_first.trainable_named_parameters():
                parameter.normal_()
        state = handle_first.state_dict_cpu()
        second = _Transformer()
        handle_second = dense.install_dense_flow_adapter(
            second, block_indices=(0, 4), hidden_width=8, bottleneck_width=4
        )
        handle_second.load_state_dict_strict(state)
        loaded = handle_second.state_dict_cpu()
        self.assertEqual(set(state), set(loaded))
        self.assertTrue(all(torch.equal(state[key], loaded[key]) for key in state))

    def test_independent_preservation_branch_is_zero_then_sequential(self):
        transformer = _Transformer()
        motion = dense.install_dense_flow_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        keep = preservation.install_preservation_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        expected = hidden + 0.25
        motion_context = dense.DenseFlowInvocation(self.features, self.activity)
        keep_context = preservation.PreservationInvocation(self.activity)
        with dense.dense_flow_invocation(motion_context), preservation.preservation_invocation(
            keep_context
        ):
            zero = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(zero, expected))

        with torch.no_grad():
            motion.adapters[0].output.weight.fill_(0.02)
        motion_state = motion.state_dict_cpu()
        keep.load_dense_flow_state_strict(motion_state, output_scale=0.5)
        with dense.dense_flow_invocation(motion_context):
            action_only = transformer.blocks[0](hidden)
        with dense.dense_flow_invocation(motion_context), preservation.preservation_invocation(
            keep_context
        ):
            combined = transformer.blocks[0](hidden)
        self.assertFalse(torch.equal(action_only, expected))
        self.assertFalse(torch.equal(combined, action_only))
        target_tokens = self.features.shape[1] // 2
        phase0_end = target_tokens + 6
        self.assertTrue(
            torch.equal(combined[:, :phase0_end], action_only[:, :phase0_end])
        )

    def test_source_copy_carrier_modes_are_explicit(self):
        hidden = torch.randn((1, self.features.shape[1], 8))
        target_tokens = self.features.shape[1] // 2
        spatial_tokens = target_tokens // dense.LATENT_PHASES
        aligned = source_copy._global_carrier(
            source_copy.SourceCopyInvocation(
                self.activity, mode="phase_aligned"
            ),
            hidden,
        )
        phase0 = source_copy._global_carrier(
            source_copy.SourceCopyInvocation(
                self.activity, mode="phase0_broadcast"
            ),
            hidden,
        )
        self.assertEqual(torch.count_nonzero(aligned[:, :target_tokens]).item(), 0)
        self.assertTrue(
            torch.equal(aligned[:, target_tokens:], hidden[:, :target_tokens])
        )
        expected = hidden[:, :spatial_tokens].repeat(
            1, dense.LATENT_PHASES, 1
        )
        self.assertTrue(torch.equal(phase0[:, target_tokens:], expected))

    def test_source_copy_branch_composes_after_frozen_motion(self):
        transformer = _Transformer()
        motion = dense.install_dense_flow_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        keep = source_copy.install_source_copy_adapter(
            transformer, block_indices=(0,), hidden_width=8, bottleneck_width=4
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        motion_context = dense.DenseFlowInvocation(self.features, self.activity)
        keep_context = source_copy.SourceCopyInvocation(
            self.activity, mode="phase_aligned"
        )
        with dense.dense_flow_invocation(motion_context), source_copy.source_copy_invocation(
            keep_context
        ):
            zero = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(zero, hidden + 0.25))
        with torch.no_grad():
            motion.adapters[0].output.weight.fill_(0.02)
            keep.adapters[0].output.weight.fill_(0.03)
        with dense.dense_flow_invocation(motion_context):
            action_only = transformer.blocks[0](hidden)
        with dense.dense_flow_invocation(motion_context), source_copy.source_copy_invocation(
            keep_context
        ):
            combined = transformer.blocks[0](hidden)
        self.assertFalse(torch.equal(combined, action_only))
        target_tokens = self.features.shape[1] // 2
        phase0_end = target_tokens + 6
        self.assertTrue(
            torch.equal(combined[:, :phase0_end], action_only[:, :phase0_end])
        )

        with dense.dense_flow_invocation(motion_context), source_copy.source_copy_invocation(
            keep_context
        ), source_copy.source_copy_denoise_weight(0.0):
            gated_off = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(gated_off, action_only))

        with dense.dense_flow_invocation(motion_context), source_copy.source_copy_invocation(
            keep_context
        ), source_copy.source_copy_denoise_weight(0.5):
            half = transformer.blocks[0](hidden)
        torch.testing.assert_close(
            half.float(),
            action_only.float() + 0.5 * (combined.float() - action_only.float()),
        )

    def test_source_copy_late_schedule_routes_cfg_pairs_together(self):
        observed = [
            infer_dense.source_copy_schedule_weight(
                shared_call_index=index,
                schedule_start=2,
                inference_steps=4,
            )
            for index in range(8)
        ]
        self.assertEqual(observed, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        with self.assertRaises(infer_dense.DenseFlowInferenceError):
            infer_dense.source_copy_schedule_weight(
                shared_call_index=8,
                schedule_start=2,
                inference_steps=4,
            )

    def test_hard_source_transport_is_full_width_target_only_and_schedulable(self):
        transformer = _Transformer()
        handle = source_copy.install_hard_source_transport(
            transformer,
            mode="phase0_broadcast",
            scale=1.0,
            block_indices=(0,),
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        original = hidden + 0.25
        invocation = source_copy.SourceCopyInvocation(
            self.activity, mode="phase0_broadcast"
        )
        with source_copy.source_copy_invocation(invocation):
            transported = transformer.blocks[0](hidden)
        target_tokens = self.features.shape[1] // 2
        spatial_tokens = target_tokens // dense.LATENT_PHASES
        self.assertTrue(
            torch.equal(
                transported[:, : target_tokens + spatial_tokens],
                original[:, : target_tokens + spatial_tokens],
            )
        )
        expected_target = original[:, :spatial_tokens].repeat(
            1, dense.LATENT_PHASES, 1
        )
        self.assertTrue(
            torch.equal(
                transported[:, target_tokens + spatial_tokens :],
                expected_target[:, spatial_tokens:],
            )
        )
        with source_copy.source_copy_invocation(
            invocation
        ), source_copy.source_copy_denoise_weight(0.0):
            gated_off = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(gated_off, original))
        handle.restore()
        self.assertTrue(torch.equal(transformer.blocks[0](hidden), original))

    def test_source_attention_requires_exact_spatial_shape(self):
        with self.assertRaises(source_copy.SourceCopyAdapterError):
            source_copy.SourceCopyInvocation(
                self.activity, mode="phase0_attention_8x12"
            ).validate()
        source_copy.SourceCopyInvocation(
            self.activity,
            mode="phase0_attention_8x12",
            spatial_shape=(2, 3),
        ).validate()

    def test_source_attention_zero_init_then_all_projections_receive_gradient(self):
        transformer = _Transformer()
        keep = source_copy.install_source_copy_adapter(
            transformer,
            mode="phase0_attention_8x12",
            block_indices=(0,),
            hidden_width=8,
            bottleneck_width=4,
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        expected = hidden + 0.25
        invocation = source_copy.SourceCopyInvocation(
            self.activity,
            mode="phase0_attention_8x12",
            spatial_shape=(2, 3),
        )
        with source_copy.source_copy_invocation(invocation):
            zero = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(zero, expected))

        with torch.no_grad():
            keep.adapters[0].output.weight.normal_(std=0.05)
        with source_copy.source_copy_invocation(invocation):
            changed = transformer.blocks[0](hidden)
        self.assertFalse(torch.equal(changed, expected))
        changed.float().square().mean().backward()
        gradients = {
            name: parameter.grad
            for name, parameter in keep.trainable_named_parameters()
        }
        for suffix in ("query.weight", "key.weight", "value.weight", "output.weight"):
            matches = [value for name, value in gradients.items() if name.endswith(suffix)]
            self.assertEqual(len(matches), 1)
            self.assertIsNotNone(matches[0])
            self.assertGreater(torch.count_nonzero(matches[0]).item(), 0)

    def test_dynamic_correspondence_labels_trace_to_source_memory(self):
        target_tokens = dense.LATENT_PHASES * 2 * 3
        features = torch.zeros((1, 2 * target_tokens, 12), dtype=torch.float32)
        activity = torch.zeros((1, 2 * target_tokens, 1), dtype=torch.bool)
        target = features[:, target_tokens:].reshape(1, dense.LATENT_PHASES, 2, 3, 12)
        target[..., 9] = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        activity[:, target_tokens + 6 :] = True
        labels = source_copy.source_attention_correspondence_labels(
            features,
            activity,
            spatial_shape=(2, 3),
            memory_shape=(2, 3),
        )
        self.assertEqual(tuple(labels.shape), (1, 2 * target_tokens))
        self.assertTrue(bool(labels[:, : target_tokens + 6].eq(-1).all().item()))
        for phase in range(1, dense.LATENT_PHASES):
            phase_labels = labels[
                :, target_tokens + phase * 6 : target_tokens + (phase + 1) * 6
            ]
            self.assertEqual(torch.count_nonzero(phase_labels.ge(0)).item(), 2)
            self.assertEqual(sorted(phase_labels[phase_labels.ge(0)].tolist()), [4, 5])

    def test_correspondence_loss_trains_query_and_key_before_output_unfreezes(self):
        transformer = _Transformer()
        keep = source_copy.install_source_copy_adapter(
            transformer,
            mode="phase0_attention_8x12",
            block_indices=(0,),
            hidden_width=8,
            bottleneck_width=4,
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        labels = source_copy.source_attention_correspondence_labels(
            self.features,
            self.activity,
            spatial_shape=(2, 3),
            memory_shape=(8, 12),
        )
        losses = []
        invocation = source_copy.SourceCopyInvocation(
            self.activity,
            mode="phase0_attention_8x12",
            spatial_shape=(2, 3),
            correspondence_labels=labels,
            correspondence_losses=losses,
            max_correspondence_queries=16,
        )
        with source_copy.source_copy_invocation(invocation):
            observed = transformer.blocks[0](hidden)
        self.assertTrue(torch.equal(observed, hidden + 0.25))
        self.assertEqual(len(losses), 1)
        losses[0].backward()
        gradients = {
            name: parameter.grad
            for name, parameter in keep.trainable_named_parameters()
        }
        for suffix in ("query.weight", "key.weight"):
            gradient = next(
                value for name, value in gradients.items() if name.endswith(suffix)
            )
            self.assertIsNotNone(gradient)
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)

    def test_correspondence_source_rank_executes_fixed_shape_zero_dummy(self):
        transformer = _Transformer()
        keep = source_copy.install_source_copy_adapter(
            transformer,
            mode="phase0_attention_8x12",
            block_indices=(0,),
            hidden_width=8,
            bottleneck_width=4,
        )
        hidden = torch.randn((1, self.features.shape[1], 8))
        labels = torch.full(
            self.activity.shape[:2], -1, dtype=torch.long
        )
        losses = []
        invocation = source_copy.SourceCopyInvocation(
            self.activity,
            mode="phase0_attention_8x12",
            spatial_shape=(2, 3),
            correspondence_labels=labels,
            correspondence_losses=losses,
            max_correspondence_queries=16,
        )
        with source_copy.source_copy_invocation(invocation):
            transformer.blocks[0](hidden)
        self.assertEqual(len(losses), 1)
        self.assertEqual(float(losses[0].item()), 0.0)
        losses[0].backward()
        gradients = {
            name: parameter.grad
            for name, parameter in keep.trainable_named_parameters()
        }
        for suffix in ("query.weight", "key.weight"):
            gradient = next(
                value for name, value in gradients.items() if name.endswith(suffix)
            )
            self.assertIsNotNone(gradient)
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_flow_warp_requires_motion_features_and_retrieves_mapped_source(self):
        with self.assertRaises(source_copy.SourceCopyAdapterError):
            source_copy.SourceCopyInvocation(
                self.activity,
                mode="phase0_flowwarp_raw",
                spatial_shape=(2, 3),
            ).validate()

        features = torch.zeros_like(self.features)
        target_tokens = features.shape[1] // 2
        spatial_tokens = target_tokens // dense.LATENT_PHASES
        # One target patch step to the right is two pixels on the latent grid.
        features[
            :, target_tokens + spatial_tokens : target_tokens + 2 * spatial_tokens, 2
        ] = torch.tanh(torch.tensor(2.0 / 8.0))
        hidden = torch.zeros((1, features.shape[1], 1), dtype=torch.float32)
        hidden[0, :spatial_tokens, 0] = torch.arange(spatial_tokens)
        invocation = source_copy.SourceCopyInvocation(
            self.activity,
            mode="phase0_flowwarp_raw",
            spatial_shape=(2, 3),
            motion_features=features,
        )
        invocation.validate()
        carrier = source_copy._global_flow_warped_carrier(invocation, hidden)
        phase1 = carrier[
            0,
            target_tokens + spatial_tokens : target_tokens + 2 * spatial_tokens,
            0,
        ]
        self.assertTrue(
            torch.allclose(
                phase1, torch.tensor([1.0, 2.0, 2.0, 4.0, 5.0, 5.0])
            )
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import sys
import unittest


try:
    import torch  # noqa: F401
except ModuleNotFoundError as error:  # lightweight local control environment
    raise unittest.SkipTest("online-anchor unit tests require torch") from error


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_online_anchor_attention_v1 as method


V14R2_LAUNCHER = (
    ROOT / "scripts/auh_launch_target_owned_routed_teacher_v14r2.sh"
)
FULLGRID_RUNNER = (
    ROOT / "scripts/auh_train_online_anchor_attention_fullgrid_existing_allocations_v2.sh"
)


class OnlineAnchorAttentionTrainingTest(unittest.TestCase):
    def rows(self):
        rows = []
        for event in method.NOOP_BY_EVENT:
            for index in range(4):
                rows.append(
                    {
                        "event_id": event,
                        "variant_id": f"v{index}",
                        "iid": f"{event}-v{index}",
                    }
                )
        return rows

    def packed_real_source_batch(
        self,
        *,
        spatial_shape=(1, 16, 21, 4, 6),
        seed=211,
    ):
        _, channels, phases, height, width = spatial_shape
        target_tokens = phases * (height // 2) * (width // 2)
        total_tokens = 2 * target_tokens
        generator = torch.Generator().manual_seed(seed)
        selector = torch.zeros((1, total_tokens), dtype=torch.bool)
        selector[:, target_tokens:] = True
        return {
            "input_vae_latents": torch.randn(
                (total_tokens, channels, 1, 2, 2), generator=generator
            ),
            "input_vae_rope": torch.randn(
                (total_tokens, 1, 64), generator=generator
            ).to(torch.complex128),
            "vae_latents_mask": selector,
            "vae_seqlen": torch.tensor([[total_tokens]], dtype=torch.int64),
            "timesteps": torch.tensor([[764.0]], dtype=torch.bfloat16),
            "target_velocity": torch.randn(
                (target_tokens, channels, 1, 2, 2), generator=generator
            ),
            "target_lens": torch.tensor([[target_tokens]], dtype=torch.int64),
        }

    def test_route_and_checkpoint_blocks_are_disjoint_and_cover_all_blocks(self):
        self.assertFalse(set(method.ROUTE_BLOCKS) & set(method.CHECKPOINT_BLOCKS))
        self.assertEqual(
            set(method.ROUTE_BLOCKS) | set(method.CHECKPOINT_BLOCKS), set(range(30))
        )
        self.assertEqual(len(method.ROUTE_BLOCKS), 22)

    def test_donor_schedule_is_cross_appearance_and_closes_three_variants(self):
        rows = self.rows()
        registry = method.row_registry(rows)
        target = rows[0]
        donors = [
            method.donor_row(target, registry, donor_index=index)["variant_id"]
            for index in range(3)
        ]
        self.assertEqual(donors, ["v1", "v2", "v3"])
        self.assertNotIn(target["variant_id"], donors)

    def test_output_must_be_fresh_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "fresh")
            args = method.build_parser().parse_args(
                [
                    "--bernini-root", "/tmp/bernini",
                    "--veomni-root", "/tmp/veomni",
                    "--checkpoint", "/tmp/checkpoint",
                    "--pair-manifest", "/tmp/pairs.json",
                    "--output", output,
                    "--profile", "hybrid",
                    "--method-source-revision", "0" * 40,
                    "--method-source-archive-sha256", "0" * 64,
                ]
            )
            method.validate_args(args)
            Path(output).mkdir()
            with self.assertRaises(method.OnlineAnchorTrainingError):
                method.validate_args(args)

    def test_content_safe_route_is_an_explicit_training_operator(self):
        self.assertIn("self_temporal_kernel", method.ROUTE_OPERATORS)
        self.assertIn("self_target_gated_kernel25", method.ROUTE_OPERATORS)
        self.assertIn("self_correspondence_kernel25", method.ROUTE_OPERATORS)
        args = method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/checkpoint",
                "--pair-manifest", "/tmp/pairs.json",
                "--output", "/tmp/fresh-online-anchor-kernel-test",
                "--profile", "action_noop",
                "--route-operator", "self_temporal_kernel",
                "--method-source-revision", "0" * 40,
                "--method-source-archive-sha256", "0" * 64,
            ]
        )
        self.assertEqual(args.route_operator, "self_temporal_kernel")

    def test_noop_prompt_is_an_explicit_preservation_replay_mode(self):
        args = method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/checkpoint",
                "--pair-manifest", "/tmp/pairs.json",
                "--output", "/tmp/fresh-online-anchor-noop-replay-test",
                "--profile", "action_noop",
                "--source-reconstruction-prompt", "noop",
                "--method-source-revision", "0" * 40,
                "--method-source-archive-sha256", "0" * 64,
            ]
        )
        self.assertEqual(args.source_reconstruction_prompt, "noop")

    def test_identity_prompt_and_four_counterfactual_schedule_are_explicit(self):
        args = method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/checkpoint",
                "--pair-manifest", "/tmp/pairs.json",
                "--output", "/tmp/fresh-online-anchor-identity-replay-test",
                "--profile", "action_noop",
                "--source-variant", "counterfactual4",
                "--source-reconstruction-prompt", "identity",
                "--method-source-revision", "0" * 40,
                "--method-source-archive-sha256", "0" * 64,
            ]
        )
        self.assertEqual(args.source_variant, "counterfactual4")
        self.assertEqual(args.source_reconstruction_prompt, "identity")
        self.assertEqual(
            method.source_variants_for_update(
                source_variant="counterfactual4", global_step=0, row_count=32
            ),
            ("noop", "incomplete"),
        )
        self.assertEqual(
            method.source_variants_for_update(
                source_variant="counterfactual4", global_step=32, row_count=32
            ),
            ("reverse", "shuffle"),
        )

    def test_reverse_and_shuffle_preserve_phase_zero_but_change_chronology(self):
        target = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
        noop = target[:, :, :1].expand_as(target).clone()
        incomplete = target.clone()
        reverse = method.source_clean_for_variant(
            target=target, noop=noop, incomplete=incomplete, variant="reverse"
        )
        shuffle = method.source_clean_for_variant(
            target=target, noop=noop, incomplete=incomplete, variant="shuffle"
        )
        self.assertTrue(torch.equal(reverse[:, :, 0], target[:, :, 0]))
        self.assertTrue(torch.equal(shuffle[:, :, 0], target[:, :, 0]))
        self.assertFalse(torch.equal(reverse, target))
        self.assertFalse(torch.equal(shuffle, target))
        self.assertEqual(
            sorted(reverse.reshape(-1).tolist()), sorted(target.reshape(-1).tolist())
        )
        self.assertEqual(
            sorted(shuffle.reshape(-1).tolist()), sorted(target.reshape(-1).tolist())
        )

    def test_paired_delta_objective_is_explicit_full_field_not_a_band(self):
        args = method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/checkpoint",
                "--pair-manifest", "/tmp/pairs.json",
                "--output", "/tmp/fresh-online-anchor-paired-delta-test",
                "--profile", "action_noop",
                "--training-objective", "paired_delta_fm",
                "--paired-target-fm-weight", "0.25",
                "--method-source-revision", "0" * 40,
                "--method-source-archive-sha256", "0" * 64,
            ]
        )
        self.assertEqual(args.training_objective, "paired_delta_fm")
        self.assertEqual(args.paired_target_fm_weight, 0.25)

    def test_sequential_paired_delta_backward_matches_two_sided_gradient(self):
        action_target = torch.tensor([[[[[2.0, -1.0]]]]])
        source_target = torch.tensor([[[[[0.5, 0.25]]]]])

        action_joint = torch.tensor([[[[[0.4, -0.2]]]]], requires_grad=True)
        source_joint = torch.tensor([[[[[0.1, 0.3]]]]], requires_grad=True)
        joint = method.paired_delta_loss(
            action_prediction=action_joint,
            source_prediction=source_joint,
            action_target=action_target,
            source_target=source_target,
            name="joint",
        )
        joint.backward()

        action_split = action_joint.detach().clone().requires_grad_(True)
        source_reference = source_joint.detach().clone()
        action_side = method.paired_delta_loss(
            action_prediction=action_split,
            source_prediction=source_reference,
            action_target=action_target,
            source_target=source_target,
            name="action side",
        )
        action_side.backward()
        source_split = source_joint.detach().clone().requires_grad_(True)
        source_side = method.paired_delta_loss(
            action_prediction=action_split.detach(),
            source_prediction=source_split,
            action_target=action_target,
            source_target=source_target,
            name="source side",
        )
        source_side.backward()

        self.assertTrue(torch.allclose(action_split.grad, action_joint.grad))
        self.assertTrue(torch.allclose(source_split.grad, source_joint.grad))

    def test_real_source_teacher_delta_is_full_field_and_phase0_relative(self):
        teacher_action = torch.arange(
            2 * 3 * 21 * 2 * 2, dtype=torch.float32
        ).reshape(2, 3, 21, 2, 2)
        teacher_noop = torch.zeros_like(teacher_action)
        relative = method.teacher_delta_tensor(
            teacher_action - teacher_noop, mode="phase0_relative"
        )
        self.assertEqual(tuple(relative.shape), tuple(teacher_action.shape))
        self.assertEqual(relative.numel(), teacher_action.numel())
        self.assertTrue(torch.equal(relative[:, :, 0], torch.zeros_like(relative[:, :, 0])))
        self.assertFalse(torch.equal(relative[:, :, 1:], torch.zeros_like(relative[:, :, 1:])))
        self.assertTrue(
            torch.equal(
                method.teacher_delta_tensor(teacher_action, mode="raw"),
                teacher_action,
            )
        )

    def test_real_source_teacher_delta_sequential_backward_is_two_sided(self):
        shape = (1, 2, 21, 1, 2)
        action_joint = torch.randn(shape, generator=torch.Generator().manual_seed(1), requires_grad=True)
        source_joint = torch.randn(shape, generator=torch.Generator().manual_seed(2), requires_grad=True)
        teacher_action = torch.randn(shape, generator=torch.Generator().manual_seed(3))
        teacher_noop = torch.randn(shape, generator=torch.Generator().manual_seed(4))
        joint = method.real_source_teacher_delta_loss(
            action_prediction=action_joint,
            source_prediction=source_joint,
            teacher_action=teacher_action,
            teacher_noop=teacher_noop,
            mode="phase0_relative",
            name="joint teacher delta",
        )
        joint.backward()

        action_split = action_joint.detach().clone().requires_grad_(True)
        action_side = method.real_source_teacher_delta_loss(
            action_prediction=action_split,
            source_prediction=source_joint.detach(),
            teacher_action=teacher_action,
            teacher_noop=teacher_noop,
            mode="phase0_relative",
            name="action teacher delta",
        )
        action_side.backward()
        source_split = source_joint.detach().clone().requires_grad_(True)
        source_side = method.real_source_teacher_delta_loss(
            action_prediction=action_split.detach(),
            source_prediction=source_split,
            teacher_action=teacher_action,
            teacher_noop=teacher_noop,
            mode="phase0_relative",
            name="source teacher delta",
        )
        source_side.backward()
        self.assertTrue(torch.allclose(action_split.grad, action_joint.grad))
        self.assertTrue(torch.allclose(source_split.grad, source_joint.grad))

    def test_real_source_objective_forbids_synthetic_target_and_missing_anchor(self):
        base = [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--pair-manifest", "/tmp/pairs.json",
            "--authoring", "/tmp/authoring.json",
            "--real-source-manifest", "/tmp/real-source.json",
            "--real-source-manifest-sha256", "1" * 64,
            "--output", "/tmp/fresh-real-source-teacher-test",
            "--training-objective", "real_source_teacher_delta",
            "--training-interface", "first_phase_caption_i2v",
            "--method-source-revision", "0" * 40,
            "--method-source-archive-sha256", "0" * 64,
        ]
        valid = method.build_parser().parse_args(
            base + ["--profile", "action_noop", "--paired-target-fm-weight", "0"]
        )
        method.validate_args(valid)
        self.assertEqual(valid.teacher_delta_mode, "phase0_relative")
        bad_target = method.build_parser().parse_args(
            base + ["--profile", "action_noop", "--paired-target-fm-weight", "0.25"]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(bad_target)
        bad_anchor = method.build_parser().parse_args(
            base + ["--profile", "no_anchor", "--paired-target-fm-weight", "0"]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(bad_anchor)

    def test_routed_teacher_requires_raw_target_owned_route_and_stronger_teacher(self):
        base = [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--pair-manifest", "/tmp/pairs.json",
            "--authoring", "/tmp/authoring.json",
            "--real-source-manifest", "/tmp/real-source.json",
            "--real-source-manifest-sha256", "2" * 64,
            "--output", "/tmp/fresh-real-source-routed-teacher-test",
            "--profile", "action_noop",
            "--training-objective", "real_source_routed_teacher_delta",
            "--training-interface", "first_phase_caption_i2v",
            "--paired-target-fm-weight", "0",
            "--teacher-delta-mode", "raw",
            "--route-strength", "0.25",
            "--teacher-route-strength", "1.0",
            "--method-source-revision", "0" * 40,
            "--method-source-archive-sha256", "0" * 64,
        ]
        valid = method.build_parser().parse_args(
            base + ["--route-operator", "self_correspondence_kernel25"]
        )
        method.validate_args(valid)
        self.assertEqual(valid.teacher_delta_mode, "raw")

        bad_cross = method.build_parser().parse_args(
            base + ["--route-operator", "cross_sparse"]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(bad_cross)

        bad_relative = method.build_parser().parse_args(
            base
            + [
                "--route-operator", "self_temporal_kernel",
                "--teacher-delta-mode", "phase0_relative",
            ]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(bad_relative)

        equal_strength = method.build_parser().parse_args(
            base
            + [
                "--route-operator", "self_temporal_kernel",
                "--teacher-route-strength", "0.25",
            ]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(equal_strength)

    def test_real_source_coordinate_contract_is_exact_and_complete(self):
        spatial_shape = (1, 16, 21, 4, 6)
        action = self.packed_real_source_batch(spatial_shape=spatial_shape, seed=21)
        source = {key: value.clone() for key, value in action.items()}
        expected_packed_shape = (252, 16, 1, 2, 2)
        self.assertEqual(
            method.require_same_real_source_noisy_state(
                action,
                source,
                spatial_shape=spatial_shape,
            ),
            expected_packed_shape,
        )
        diagnostic = method.real_source_prebind_state_diagnostic(
            action,
            source,
            spatial_shape=spatial_shape,
        )
        self.assertTrue(diagnostic["raw_same_seed_state_exact"])
        self.assertEqual(diagnostic["raw_same_seed_unequal_fields"], [])
        self.assertEqual(
            diagnostic["packed_geometry"]["latent_phases"],
            21,
        )
        self.assertEqual(
            diagnostic["packed_geometry"]["input_vae_latents_shape"],
            list(expected_packed_shape),
        )
        geometry = diagnostic["packed_geometry"]
        self.assertEqual(geometry["selector_transition_count"], 1)
        self.assertEqual(geometry["vae_seqlen"], 252)
        self.assertEqual(geometry["target_lens"], 126)
        self.assertEqual(geometry["timestep_value"], 764.0)
        self.assertEqual(geometry["timestep_range_inclusive"], [0.0, 1000.0])
        self.assertEqual(
            {
                key: value["dtype"]
                for key, value in geometry["state_fields"].items()
            },
            {
                "input_vae_latents": "float32",
                "input_vae_rope": "complex128",
                "vae_latents_mask": "bool",
                "vae_seqlen": "int64",
                "timesteps": "bfloat16",
                "target_velocity": "float32",
                "target_lens": "int64",
            },
        )

        changed = {key: value.clone() for key, value in source.items()}
        changed["input_vae_latents"][7, 1, 0, 1, 1] += 1
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                changed,
                spatial_shape=spatial_shape,
            )

        wrong_selector = {key: value.clone() for key, value in source.items()}
        wrong_selector["vae_latents_mask"][:, 0] = True
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                wrong_selector,
                spatial_shape=spatial_shape,
            )

        wrong_seqlen = {key: value.clone() for key, value in source.items()}
        wrong_seqlen["vae_seqlen"] = torch.tensor([[251]], dtype=torch.int64)
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                wrong_seqlen,
                spatial_shape=spatial_shape,
            )

        wrong_lens = {key: value.clone() for key, value in source.items()}
        wrong_lens["target_lens"] = torch.tensor([[125]], dtype=torch.int64)
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                wrong_lens,
                spatial_shape=spatial_shape,
            )

        wrong_rope = {key: value.clone() for key, value in source.items()}
        wrong_rope["input_vae_rope"] = torch.zeros(
            (252, 1, 63), dtype=torch.complex128
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                wrong_rope,
                spatial_shape=spatial_shape,
            )

        wrong_timestep = {key: value.clone() for key, value in source.items()}
        wrong_timestep["timesteps"] = torch.tensor(
            [[1024.0]], dtype=torch.bfloat16
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                wrong_timestep,
                spatial_shape=spatial_shape,
            )

        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.require_same_real_source_noisy_state(
                action,
                source,
                spatial_shape=(1, 16, 20, 4, 6),
            )

    def test_real_source_caption_is_bound_to_one_explicit_vae_state(self):
        spatial_shape = (1, 16, 21, 4, 6)
        action = self.packed_real_source_batch(
            spatial_shape=spatial_shape,
            seed=211,
        )
        source = {key: value.clone() for key, value in action.items()}
        source["input_ids"] = torch.tensor([[91, 92]])
        source["attention_mask"] = torch.tensor([[1, 1]])
        source["t5_input_lens"] = torch.tensor([2])
        bound, diagnostic = method.bind_real_source_caption_to_action_state(
            action,
            source,
            spatial_shape=spatial_shape,
        )
        self.assertTrue(diagnostic["raw_same_seed_state_exact"])
        for key in method.REAL_SOURCE_EXACT_STATE_KEYS:
            self.assertIs(bound[key], action[key])
        self.assertIs(bound["input_ids"], source["input_ids"])
        self.assertIs(bound["attention_mask"], source["attention_mask"])
        self.assertIs(bound["t5_input_lens"], source["t5_input_lens"])
        self.assertEqual(
            method.require_same_real_source_noisy_state(
                action,
                bound,
                spatial_shape=spatial_shape,
            ),
            (252, 16, 1, 2, 2),
        )
        bad = dict(source)
        bad["target_velocity"] = source["target_velocity"].clone()
        bad["target_velocity"][0, 0, 0, 0, 0] += 1
        raw_bad_target_velocity = bad["target_velocity"]
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.bind_real_source_caption_to_action_state(
                action,
                bad,
                spatial_shape=spatial_shape,
            )
        # A raw mismatch must fail before binding rather than being hidden by
        # replacing the source-caption tensor with the action tensor.
        self.assertIs(bad["target_velocity"], raw_bad_target_velocity)
        self.assertIsNot(bad["target_velocity"], action["target_velocity"])

    def test_routed_teacher_signal_is_full_tensor_and_strength_identifiable(self):
        shape = (1, 2, 21, 2, 2)
        generator = torch.Generator().manual_seed(22)
        base_action = torch.randn(shape, generator=generator)
        base_source = torch.randn(shape, generator=generator)
        route = torch.randn(shape, generator=generator)
        student_strength = 0.25

        equal = method.routed_teacher_supervision_residual(
            action_prediction=base_action + student_strength * route,
            source_prediction=base_source,
            teacher_action=base_action + student_strength * route,
            teacher_source=base_source,
            mode="raw",
        )
        self.assertEqual(tuple(equal.shape), shape)
        self.assertEqual(equal.numel(), base_action.numel())
        self.assertTrue(torch.equal(equal, torch.zeros_like(equal)))

        unequal = method.routed_teacher_supervision_residual(
            action_prediction=base_action + student_strength * route,
            source_prediction=base_source,
            teacher_action=base_action + 1.0 * route,
            teacher_source=base_source,
            mode="raw",
        )
        self.assertEqual(tuple(unequal.shape), shape)
        self.assertTrue(bool(unequal.ne(0).any().item()))

    def test_routed_teacher_route_capture_is_replayed_exactly_twice(self):
        self.assertEqual(
            method.anchor_route_replay_uses("real_source_routed_teacher_delta"), 2
        )
        self.assertEqual(
            method.anchor_route_replay_uses(
                "real_source_target_owned_routed_teacher_delta_v14r2"
            ),
            2,
        )
        self.assertEqual(method.anchor_route_replay_uses("real_source_teacher_delta"), 1)
        cache = method.qk.AnchorQKCacheBank(method.ROUTE_BLOCKS)
        invocation, _ = method.replay_invocation(
            cache=cache,
            invocation_index=0,
            candidate_index=0,
            rank=0,
            world_size=1,
            strength=0.5,
            route_operator="self_temporal_kernel",
            replay_uses=2,
        )
        self.assertEqual(invocation.replay_uses, 2)

    def test_same_action_route_only_caller_stops_routeoff_and_omits_source_backward(self):
        source = Path(method.__file__).read_text(encoding="utf-8")
        same_action = source.index("if same_action_route_only:", source.index("def main("))
        routeoff = source.index("student_route_off_prediction", same_action)
        routed_query = source.index("with invocation_context(invocation)", routeoff)
        self.assertIn("with torch.no_grad()", source[same_action:routed_query])
        self.assertIn(
            "if paired_action_loss is not None and not same_action_route_only:",
            source,
        )
        self.assertIn(
            '"action_objective_backpropagates_only_routed_student_query"',
            source,
        )

    def test_routed_teacher_sequential_backward_is_two_sided(self):
        shape = (1, 2, 21, 1, 2)
        generator = torch.Generator().manual_seed(23)
        action_joint = torch.randn(shape, generator=generator, requires_grad=True)
        source_joint = torch.randn(shape, generator=generator, requires_grad=True)
        teacher_action = torch.randn(shape, generator=generator)
        teacher_source = torch.randn(shape, generator=generator)
        joint = method.real_source_routed_teacher_delta_loss(
            action_prediction=action_joint,
            source_prediction=source_joint,
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="joint routed teacher delta",
        )
        joint.backward()

        action_split = action_joint.detach().clone().requires_grad_(True)
        action_side = method.real_source_routed_teacher_delta_loss(
            action_prediction=action_split,
            source_prediction=source_joint.detach(),
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="action-side routed teacher delta",
        )
        action_side.backward()
        source_split = source_joint.detach().clone().requires_grad_(True)
        source_side = method.real_source_routed_teacher_delta_loss(
            action_prediction=action_split.detach(),
            source_prediction=source_split,
            teacher_action=teacher_action,
            teacher_source=teacher_source,
            mode="raw",
            name="source-side routed teacher delta",
        )
        source_side.backward()
        self.assertTrue(torch.allclose(action_split.grad, action_joint.grad))
        self.assertTrue(torch.allclose(source_split.grad, source_joint.grad))

    def test_caption_i2v_interface_has_full_variant_captions_and_phase0_condition(self):
        authoring = ROOT / "assets" / "interaction_complex8_multianchor_authoring_v2.json"
        captions = method.load_caption_registry(authoring)
        self.assertEqual(len(captions), 32)
        row = captions[("pour-liquid-into-cup", "v0")]
        self.assertIn("young East Asian woman", row["target"])
        self.assertIn("pours a clearly visible continuous stream", row["target"])
        self.assertIn("No vessel is lifted or tilted", row["noop"])
        clean = torch.arange(21, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
        condition = method.repeated_phase_zero(clean)
        self.assertEqual(tuple(condition.shape), tuple(clean.shape))
        self.assertTrue(torch.equal(condition, clean[:, :, :1].expand_as(clean)))
        self.assertFalse(torch.equal(condition, clean))

    def test_caption_i2v_interface_rejects_uncaptionable_temporal_permutations(self):
        args = method.build_parser().parse_args(
            [
                "--bernini-root", "/tmp/bernini",
                "--veomni-root", "/tmp/veomni",
                "--checkpoint", "/tmp/checkpoint",
                "--pair-manifest", "/tmp/pairs.json",
                "--authoring", "/tmp/authoring.json",
                "--output", "/tmp/fresh-online-anchor-caption-i2v-test",
                "--profile", "action_noop",
                "--training-objective", "paired_delta_fm",
                "--training-interface", "first_phase_caption_i2v",
                "--source-variant", "counterfactual4",
                "--method-source-revision", "0" * 40,
                "--method-source-archive-sha256", "0" * 64,
            ]
        )
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_args(args)

    def test_v14r2_objective_accepts_only_unique_target_owned_qk_routes(self):
        route = "self_target_owned_temporal_kernel_v14r2"
        base = [
            "--bernini-root", "/tmp/bernini",
            "--veomni-root", "/tmp/veomni",
            "--checkpoint", "/tmp/checkpoint",
            "--pair-manifest", "/tmp/pairs.json",
            "--authoring", "/tmp/authoring.json",
            "--real-source-manifest", "/tmp/real-source.json",
            "--real-source-manifest-sha256", "3" * 64,
            "--output", "/tmp/fresh-target-owned-v14r2-test",
            "--profile", "action_noop",
            "--training-objective",
            "real_source_target_owned_routed_teacher_delta_v14r2",
            "--training-interface", "first_phase_caption_i2v",
            "--paired-target-fm-weight", "0",
            "--source-reconstruction-weight", "0.025",
            "--source-variant", "not_applicable",
            "--teacher-delta-mode", "raw",
            "--routed-teacher-mode", "same_action_route_only",
            "--replay-combine-mode", "first_order_safe",
            "--route-strength", "0.25",
            "--teacher-route-strength", "0.5",
            "--method-source-revision", "0" * 40,
            "--method-source-archive-sha256", "0" * 64,
        ]
        args = method.build_parser().parse_args(base + ["--route-operator", route])
        method.validate_args(args)
        self.assertEqual(
            method.qk_transport_for_route_operator(route),
            method.qk.TARGET_OWNED_TEMPORAL_KERNEL_ATTN_OUTPUT_V14R2,
        )
        self.assertEqual(method.anchor_route_replay_uses(args.training_objective), 2)

        for unsafe in (
            "self_temporal_kernel",
            "self_target_gated_kernel25",
            "self_correspondence_kernel25",
            "cross_sparse",
        ):
            rejected = method.build_parser().parse_args(
                base + ["--route-operator", unsafe]
            )
            with self.assertRaises(method.OnlineAnchorTrainingError):
                method.validate_args(rejected)
        self.assertNotIn("qk_only_temporal_kernel", method.ROUTE_OPERATORS)

    def test_component_gradient_probe_reports_exact_ab_sides(self):
        parameters = []
        for name, value in (
            ("block.lora_A.weight", 0.0),
            ("block.lora_B.weight", 2.0),
            ("other.lora_A.weight", 3.0),
            ("other.lora_B.weight", 0.0),
        ):
            parameter = torch.nn.Parameter(torch.zeros(1))
            parameter.grad = torch.tensor([value])
            parameters.append((name, parameter))
        probe = method.component_gradient_probe(parameters)
        self.assertEqual(probe["tensor_count"], 4)
        self.assertEqual(probe["nonzero_tensor_count"], 2)
        self.assertEqual(probe["epsilon_active_tensor_count"], 2)
        self.assertEqual(
            probe["adapter_sides"]["lora_A"]["nonzero_tensor_count"], 1
        )
        self.assertEqual(
            probe["adapter_sides"]["lora_B"]["nonzero_tensor_count"], 1
        )

    def test_v14r2_component_coverage_closes_s1_s2_and_raw_replay(self):
        def probe(total, side_a, side_b):
            return {
                "tensor_count": 480,
                "nonzero_tensor_count": total,
                "epsilon_active_tensor_count": total,
                "adapter_sides": {
                    "lora_A": {
                        "tensor_count": 240,
                        "nonzero_tensor_count": side_a,
                        "epsilon_active_tensor_count": side_a,
                    },
                    "lora_B": {
                        "tensor_count": 240,
                        "nonzero_tensor_count": side_b,
                        "epsilon_active_tensor_count": side_b,
                    },
                },
            }

        method.validate_v14r2_component_coverage(
            probe(240, 0, 240), step=1, component="action"
        )
        method.validate_v14r2_component_coverage(
            probe(480, 240, 240), step=2, component="action"
        )
        method.validate_v14r2_component_coverage(
            probe(480, 240, 240), step=2, component="raw_replay"
        )
        bad = probe(240, 0, 240)
        bad["epsilon_active_tensor_count"] = 1
        with self.assertRaises(method.OnlineAnchorTrainingError):
            method.validate_v14r2_component_coverage(
                bad, step=1, component="action"
            )

    def test_component_gradient_merge_fixed_and_first_order_safe_are_exact(self):
        fixed_parameter = torch.nn.Parameter(torch.zeros(1))
        fixed_parameter.grad = torch.tensor([1.0])
        fixed = method.merge_component_gradients(
            [("x.lora_B.weight", fixed_parameter)],
            (torch.tensor([1.0]),),
            replay_combine_mode="fixed_0025",
            base_replay_scale=0.025,
        )
        self.assertTrue(torch.allclose(fixed_parameter.grad, torch.tensor([1.025])))
        self.assertEqual(fixed["effective_replay_scale"], 0.025)
        self.assertLess(fixed["weighted_replay_gradient_fraction"], 0.25)

        safe_parameter = torch.nn.Parameter(torch.zeros(2))
        safe_parameter.grad = torch.tensor([-2.0, 0.0])
        safe = method.merge_component_gradients(
            [("x.lora_B.weight", safe_parameter)],
            (torch.tensor([1.0, 10.0]),),
            replay_combine_mode="first_order_safe",
            base_replay_scale=0.025,
        )
        self.assertAlmostEqual(safe["first_order_safe_lambda_min"], 0.5)
        self.assertAlmostEqual(safe["effective_replay_scale"], 0.55)
        self.assertTrue(safe["first_order_source_fm_preserved"])
        self.assertGreaterEqual(
            safe["raw_source_fm_gradient_dot_combined_gradient_fp64"], 0.0
        )
        self.assertTrue(torch.allclose(safe_parameter.grad, torch.tensor([-0.1, 10.0])))

        negligible_replay = torch.nn.Parameter(torch.zeros(1))
        negligible_replay.grad = torch.tensor([1.0])
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            r"weighted_fraction=.*action_l2=.*raw_replay_l2=.*effective_scale=",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", negligible_replay)],
                (torch.tensor([1.0e6]),),
                replay_combine_mode="fixed_0025",
                base_replay_scale=0.025,
            )

        diagnostic_parameter = torch.nn.Parameter(torch.zeros(2))
        diagnostic_parameter.grad = torch.tensor([-2.0, 0.0])
        raw_replay_before = diagnostic_parameter.grad.clone()
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            r"GRADIENT_DIAGNOSTIC_COMPLETE\|optimizer_steps=0\|.*"
            r"action_l2=.*raw_replay_l2=.*raw_replay_to_action_ratio=.*cosine=",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", diagnostic_parameter)],
                (torch.tensor([1.0, 10.0]),),
                replay_combine_mode="first_order_safe",
                base_replay_scale=0.025,
                diagnostic_only=True,
            )
        self.assertTrue(torch.equal(diagnostic_parameter.grad, raw_replay_before))

    def test_v14r3_action_and_norm_balanced_merges_are_scale_invariant(self):
        action = torch.tensor([1.0, 0.0])
        cosine = 0.48
        raw_norm = 463.0
        replay = torch.tensor(
            [
                raw_norm * cosine,
                raw_norm * math.sqrt(1.0 - cosine * cosine),
            ]
        )

        action_only_parameter = torch.nn.Parameter(torch.zeros(2))
        action_only_parameter.grad = replay.clone()
        action_only = method.merge_component_gradients(
            [("x.lora_B.weight", action_only_parameter)],
            (action,),
            replay_combine_mode="action_only",
            base_replay_scale=0.025,
        )
        self.assertTrue(torch.equal(action_only_parameter.grad, action))
        self.assertEqual(action_only["correction_ratio_q"], 0.0)
        self.assertEqual(action_only["weighted_replay_gradient_fraction"], 0.0)
        self.assertAlmostEqual(
            action_only["action_gradient_dot_combined_gradient_fp64"], 1.0
        )

        for mode, expected_q in (
            ("norm_balanced_005", 0.05),
            ("norm_balanced_025", 0.25),
        ):
            parameter = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = replay.clone()
            interaction = method.merge_component_gradients(
                [("x.lora_B.weight", parameter)],
                (action,),
                replay_combine_mode=mode,
                base_replay_scale=0.025,
            )
            expected = action + expected_q * replay / replay.norm()
            self.assertTrue(torch.allclose(parameter.grad, expected, atol=1.0e-6))
            self.assertAlmostEqual(
                interaction["correction_ratio_q"], expected_q, places=6
            )
            self.assertAlmostEqual(
                interaction["action_alignment_ratio"],
                1.0 + expected_q * cosine,
                places=6,
            )
            self.assertGreater(
                interaction[
                    "raw_replay_gradient_dot_combined_gradient_fp64"
                ],
                0.0,
            )
            self.assertAlmostEqual(
                interaction[
                    "raw_replay_combined_alignment_over_action_replay_norms"
                ],
                cosine + expected_q,
                places=6,
            )

    def test_v14r3_source_safe_cap_fails_closed_when_geometry_has_no_budget(self):
        action = torch.tensor([1.0, 0.0])
        raw_norm = 4.886
        cosine = -0.926
        replay = torch.tensor(
            [
                raw_norm * cosine,
                raw_norm * math.sqrt(1.0 - cosine * cosine),
            ]
        )
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        before = parameter.grad.clone()
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            r"mode=source_safe_cap025, reason=required_q_exceeds_0.25, q=",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", parameter)],
                (action,),
                replay_combine_mode="source_safe_cap025",
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(parameter.grad, before))

        feasible_replay = torch.tensor([-0.5, math.sqrt(24.75)])
        parameter.grad = feasible_replay.clone()
        interaction = method.merge_component_gradients(
            [("x.lora_B.weight", parameter)],
            (action,),
            replay_combine_mode="source_safe_cap025",
            base_replay_scale=0.025,
        )
        self.assertAlmostEqual(interaction["correction_ratio_q"], 0.11, places=6)
        self.assertGreater(
            interaction["raw_replay_gradient_dot_combined_gradient_fp64"],
            0.0,
        )

    def test_v14r3_source_halfspace_is_source_safe_and_action_gated(self):
        action = torch.tensor([1.0, 0.0])
        raw_norm = 4.886
        cosine = -0.926
        replay = torch.tensor(
            [
                raw_norm * cosine,
                raw_norm * math.sqrt(1.0 - cosine * cosine),
            ]
        )
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        interaction = method.merge_component_gradients(
            [("x.lora_B.weight", parameter)],
            (action,),
            replay_combine_mode="source_halfspace_001",
            base_replay_scale=0.025,
        )
        self.assertAlmostEqual(
            interaction["correction_ratio_q"], -cosine + 0.01, places=6
        )
        self.assertGreaterEqual(interaction["action_alignment_ratio"], 0.1)
        self.assertAlmostEqual(
            interaction["raw_replay_gradient_dot_combined_gradient_fp64"],
            0.01 * raw_norm,
            places=5,
        )
        self.assertAlmostEqual(
            interaction[
                "raw_replay_combined_alignment_over_action_replay_norms"
            ],
            0.01,
            places=5,
        )

        compatible_parameter = torch.nn.Parameter(torch.zeros(2))
        compatible_parameter.grad = torch.tensor([2.0, 0.0])
        compatible = method.merge_component_gradients(
            [("x.lora_B.weight", compatible_parameter)],
            (action,),
            replay_combine_mode="source_halfspace_001",
            base_replay_scale=0.025,
        )
        self.assertAlmostEqual(compatible["correction_ratio_q"], 0.01, places=6)
        self.assertTrue(
            torch.allclose(compatible_parameter.grad, torch.tensor([1.01, 0.0]))
        )

        severe_cosine = -0.99
        severe = torch.tensor(
            [severe_cosine, math.sqrt(1.0 - severe_cosine * severe_cosine)]
        )
        severe_parameter = torch.nn.Parameter(torch.zeros(2))
        severe_parameter.grad = severe.clone()
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            r"mode=source_halfspace_001, reason=action_alignment_below_0.1",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", severe_parameter)],
                (action,),
                replay_combine_mode="source_halfspace_001",
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(severe_parameter.grad, severe))

    def test_v14r3_action_priority_pcgrad_projects_conflict_then_normalizes(self):
        action = torch.tensor([1.0, 0.0])
        raw_norm = 4.886
        cosine = -0.926
        replay = torch.tensor(
            [
                raw_norm * cosine,
                raw_norm * math.sqrt(1.0 - cosine * cosine),
            ]
        )
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = replay.clone()
        interaction = method.merge_component_gradients(
            [("x.lora_B.weight", parameter)],
            (action,),
            replay_combine_mode="action_priority_pcgrad_010",
            base_replay_scale=0.025,
        )
        self.assertTrue(torch.allclose(parameter.grad, torch.tensor([1.0, 0.1])))
        self.assertTrue(interaction["replay_projection_applied"])
        self.assertTrue(
            interaction[
                "action_priority_conflict_control_not_source_preservation"
            ]
        )
        self.assertAlmostEqual(interaction["correction_ratio_q"], 0.10, places=6)
        self.assertAlmostEqual(interaction["action_alignment_ratio"], 1.0, places=6)
        self.assertAlmostEqual(
            interaction["processed_replay_action_cosine"], 0.0, places=6
        )
        self.assertGreaterEqual(
            interaction["processed_replay_retained_raw_norm_fraction"], 0.2
        )
        self.assertLess(
            interaction["raw_replay_gradient_dot_combined_gradient_fp64"],
            0.0,
        )

        compatible_parameter = torch.nn.Parameter(torch.zeros(2))
        compatible_parameter.grad = torch.tensor([0.6, 0.8])
        compatible = method.merge_component_gradients(
            [("x.lora_B.weight", compatible_parameter)],
            (action,),
            replay_combine_mode="action_priority_pcgrad_010",
            base_replay_scale=0.025,
        )
        self.assertFalse(compatible["replay_projection_applied"])
        self.assertGreaterEqual(compatible["action_replay_cosine"], 0.0)
        self.assertTrue(
            torch.allclose(
                compatible_parameter.grad, torch.tensor([1.06, 0.08])
            )
        )

    def test_v14r3_formal_geometry_gates_fail_before_gradient_mutation(self):
        for mode, cosine in (
            ("norm_balanced_005", -0.10),
            ("norm_balanced_025", -0.30),
        ):
            action = torch.tensor([1.0, 0.0])
            replay = torch.tensor(
                [cosine, math.sqrt(1.0 - cosine * cosine)]
            )
            parameter = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = replay.clone()
            with self.assertRaisesRegex(
                method.OnlineAnchorTrainingError,
                "norm-balanced replay left the source-descent half-space",
            ):
                method.merge_component_gradients(
                    [("x.lora_B.weight", parameter)],
                    (action,),
                    replay_combine_mode=mode,
                    base_replay_scale=0.025,
                )
            self.assertTrue(torch.equal(parameter.grad, replay))

        action = torch.tensor([1.0, 0.0])
        conflict = torch.tensor([-0.6, 0.8])
        action_only_parameter = torch.nn.Parameter(torch.zeros(2))
        action_only_parameter.grad = conflict.clone()
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            "action-only formal update has excessive action/replay conflict",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", action_only_parameter)],
                (action,),
                replay_combine_mode="action_only",
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(action_only_parameter.grad, conflict))

        nearly_collinear = torch.tensor([-0.99, math.sqrt(1.0 - 0.99**2)])
        pcgrad_parameter = torch.nn.Parameter(torch.zeros(2))
        pcgrad_parameter.grad = nearly_collinear.clone()
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            "action-priority PCGrad formal geometry gate failed",
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", pcgrad_parameter)],
                (action,),
                replay_combine_mode="action_priority_pcgrad_010",
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(pcgrad_parameter.grad, nearly_collinear))

    def test_actual_optimizer_update_probe_observes_real_adamw_displacement(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        named = [("x.lora_B.weight", parameter)]
        action = (torch.tensor([0.5, -0.25]),)
        replay = (torch.tensor([-0.2, 0.1]),)
        parameter.grad = action[0].clone()
        optimizer = torch.optim.AdamW(
            [parameter], lr=1.0e-2, weight_decay=0.0
        )
        before = method.clone_trainable_parameter_values(named)
        optimizer.step()
        probe = method.actual_optimizer_update_probe(
            named,
            before,
            action,
            replay,
            replay_combine_mode="action_only",
            step=1,
        )
        delta = parameter.detach().double() - before[0].double()
        expected_action_dot = float((action[0].double() * delta).sum().item())
        expected_replay_dot = float((replay[0].double() * delta).sum().item())
        self.assertEqual(
            probe["schema_version"],
            "bernini-actual-optimizer-update-probe-v1",
        )
        self.assertAlmostEqual(
            probe["action_gradient_dot_delta_theta_fp64"], expected_action_dot
        )
        self.assertAlmostEqual(
            probe["raw_replay_gradient_dot_delta_theta_fp64"], expected_replay_dot
        )
        self.assertGreater(probe["action_descent_fp64"], 0.0)
        self.assertTrue(probe["action_descent_passed"])
        self.assertFalse(probe["source_descent_required"])
        self.assertEqual(probe["changed_tensor_count"], 1)
        self.assertEqual(probe["changed_element_count"], 2)
        self.assertGreater(probe["delta_theta_relative_parameter_l2_norm"], 0.0)
        self.assertEqual(probe["parameter_snapshot_dtypes"], ["float32"])
        self.assertTrue(torch.equal(before[0], torch.tensor([1.0, -2.0])))

    def test_actual_optimizer_update_source_gate_is_mode_specific(self):
        action = (torch.tensor([1.0, 0.0]),)
        replay = (torch.tensor([-0.2, 1.0]),)

        def displaced_named():
            parameter = torch.nn.Parameter(torch.tensor([-1.0, 0.0]))
            return [("x.lora_B.weight", parameter)]

        before = (torch.zeros(2),)
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            "actual optimizer update left the required source-descent half-space",
        ):
            method.actual_optimizer_update_probe(
                displaced_named(),
                before,
                action,
                replay,
                replay_combine_mode="norm_balanced_025",
                step=1,
            )
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError,
            "actual optimizer update left the required source-descent half-space",
        ):
            method.actual_optimizer_update_probe(
                displaced_named(),
                before,
                action,
                replay,
                replay_combine_mode="source_halfspace_001",
                step=1,
            )

        action_only = method.actual_optimizer_update_probe(
            displaced_named(),
            before,
            action,
            replay,
            replay_combine_mode="action_only",
            step=1,
        )
        pcgrad = method.actual_optimizer_update_probe(
            displaced_named(),
            before,
            action,
            replay,
            replay_combine_mode="action_priority_pcgrad_010",
            step=1,
        )
        self.assertFalse(action_only["source_descent_passed"])
        self.assertFalse(action_only["source_descent_required"])
        self.assertFalse(pcgrad["source_descent_passed"])
        self.assertFalse(pcgrad["source_descent_required"])

        source_safe_parameter = torch.nn.Parameter(torch.tensor([-1.0, -1.0]))
        source_safe = method.actual_optimizer_update_probe(
            [("x.lora_B.weight", source_safe_parameter)],
            before,
            action,
            (torch.tensor([1.0, 1.0]),),
            replay_combine_mode="source_halfspace_001",
            step=2,
        )
        self.assertTrue(source_safe["source_descent_required"])
        self.assertTrue(source_safe["source_descent_passed"])
        self.assertGreater(source_safe["source_descent_fp64"], 0.0)

    def test_v14r3_merge_fails_before_mutation_on_nonfinite_component(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        parameter.grad = torch.tensor([1.0])
        with self.assertRaisesRegex(
            method.OnlineAnchorTrainingError, "action component gradient is non-finite"
        ):
            method.merge_component_gradients(
                [("x.lora_B.weight", parameter)],
                (torch.tensor([float("nan")]),),
                replay_combine_mode="action_only",
                base_replay_scale=0.025,
            )
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([1.0])))

    def test_source_absorption_diagnostic_sums_to_one(self):
        action = torch.tensor([[[[[3.0, 1.0]]]]])
        source = torch.tensor([[[[[1.0, 2.0]]]]])
        target = torch.tensor([[[[[0.5, -1.0]]]]])
        diagnostic = method.source_absorption_diagnostic(
            action_prediction=action,
            source_prediction=source,
            source_velocity_target=target,
        )
        self.assertTrue(diagnostic["defined"])
        self.assertAlmostEqual(diagnostic["q_sum"], 1.0, places=6)

    def test_v14r2_receipt_separates_base_and_effective_replay_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_manifest = root / "pairs.json"
            authoring = root / "authoring.json"
            real_source = root / "real-source.json"
            for path in (pair_manifest, authoring, real_source):
                path.write_text("{}\n", encoding="ascii")
            args = method.build_parser().parse_args(
                [
                    "--bernini-root", "/tmp/bernini",
                    "--veomni-root", "/tmp/veomni",
                    "--checkpoint", "/tmp/checkpoint",
                    "--pair-manifest", str(pair_manifest),
                    "--authoring", str(authoring),
                    "--real-source-manifest", str(real_source),
                    "--real-source-manifest-sha256", "3" * 64,
                    "--output", str(root / "fresh"),
                    "--profile", "action_noop",
                    "--training-objective",
                    "real_source_target_owned_routed_teacher_delta_v14r2",
                    "--training-interface", "first_phase_caption_i2v",
                    "--route-operator", "self_target_owned_temporal_kernel_v14r2",
                    "--route-strength", "0.25",
                    "--teacher-route-strength", "0.5",
                    "--teacher-delta-mode", "raw",
                    "--routed-teacher-mode", "same_action_route_only",
                    "--replay-combine-mode", "first_order_safe",
                    "--paired-target-fm-weight", "0",
                    "--source-reconstruction-weight", "0.025",
                    "--source-variant", "not_applicable",
                    "--max-steps", "2",
                    "--method-source-revision", "0" * 40,
                    "--method-source-archive-sha256", "0" * 64,
                ]
            )
            packed = self.packed_real_source_batch()
            prebind = method.real_source_prebind_state_diagnostic(
                packed,
                {key: value.clone() for key, value in packed.items()},
                spatial_shape=(1, 16, 21, 4, 6),
            )
            prebind_update = {
                "schema_version": "bernini-real-source-prebind-packed-update-v1",
                "micro_count": 2,
                "all_raw_same_seed_state_exact": True,
                "action_branches": [prebind, prebind],
                "replay_branches": [prebind, prebind],
            }
            receipt = method.checkpoint_receipt(
                args=args,
                step=2,
                loss=2.1,
                action_objective=1.0,
                source_reconstruction=2.0,
                effective_replay_scale=0.55,
                grad_norm=3.0,
                memory={
                    "passed": True,
                    "capture_phase": (
                        "after_two_real_component_backwards_before_actual_update_audit_clones"
                    ),
                    "actual_update_audit_allocations_excluded": True,
                },
                targets=("x",),
                initial_digest="initial",
                cache=method.qk.AnchorQKCacheBank(method.ROUTE_BLOCKS),
                bernini_revision="b",
                veomni_revision="v",
                pair_manifest=pair_manifest,
                gradient={"tensor_count": 480, "nonzero_tensor_count": 480},
                action_gradient={"tensor_count": 480},
                replay_gradient={"tensor_count": 480},
                gradient_interaction={"effective_replay_scale": 0.55},
                actual_optimizer_update={
                    "schema_version": "bernini-actual-optimizer-update-probe-v1",
                    "step": 2,
                    "action_descent_passed": True,
                },
                source_absorption={"applicable": False},
                real_source_prebind_state=prebind_update,
            )
        self.assertEqual(receipt["schema_version"], method.QK_ONLY_RECEIPT_SCHEMA)
        self.assertEqual(receipt["actual_optimizer_update_probe"]["step"], 2)
        self.assertIsNone(receipt["last_loss"])
        self.assertEqual(receipt["last_reporting_scalar"], 2.1)
        self.assertTrue(
            receipt["last_reporting_scalar_is_not_a_joint_backpropagated_objective"]
        )
        components = receipt["last_objective_components"]
        self.assertEqual(components["base_replay_scale"], 0.025)
        self.assertEqual(components["effective_replay_scale"], 0.55)
        self.assertAlmostEqual(components["base_source_replay_scalar_diagnostic"], 0.05)
        self.assertAlmostEqual(
            components["effective_source_replay_scalar_for_reporting"], 1.1
        )
        contract = receipt["training_contract"]
        self.assertIsNone(contract["source_reconstruction_weight"])
        self.assertEqual(contract["source_reconstruction_weight_argument"], 0.025)
        self.assertEqual(contract["base_replay_scale"], 0.025)
        self.assertEqual(contract["effective_replay_scale"], 0.55)
        self.assertEqual(
            contract["training_memory_gate_capture_phase"],
            "after_two_real_component_backwards_before_actual_update_audit_clones",
        )
        self.assertTrue(
            contract[
                "actual_update_audit_allocations_excluded_from_training_memory_gate"
            ]
        )
        self.assertEqual(contract["source_variant_argument"], "not_applicable")
        self.assertEqual(
            contract["micro_semantics"],
            "different_seed_and_cross_appearance_donor",
        )
        self.assertTrue(
            contract[
                "anchor_qk_time_constant_caption_offset_removed_before_support_and_kernel"
            ]
        )
        self.assertTrue(contract["anchor_qk_phase0_only_difference_produces_zero_route"])
        self.assertTrue(
            receipt["real_source_prebind_state"][
                "all_raw_same_seed_state_exact"
            ]
        )
        self.assertEqual(
            receipt["real_source_prebind_state"]["action_branches"][0][
                "packed_geometry"
            ]["state_fields"]["input_vae_rope"]["dtype"],
            "complex128",
        )

    def test_checkpoint_save_pins_adapter_model_and_config_hashes(self):
        class Renderer:
            @staticmethod
            def save_pretrained(path, *, safe_serialization):
                self.assertTrue(safe_serialization)
                path.mkdir()
                (path / "adapter_model.safetensors").write_bytes(b"weights")
                (path / "adapter_config.json").write_text(
                    '{"lora_alpha":256}\n', encoding="ascii"
                )

        class Dist:
            calls = 0

            @classmethod
            def barrier(cls):
                cls.calls += 1

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            method.save_checkpoint(
                output=output,
                step=2,
                renderer=Renderer(),
                receipt={"schema_version": method.QK_ONLY_RECEIPT_SCHEMA},
                rank=0,
                dist=Dist,
            )
            root = output / "checkpoint-00000002"
            receipt = __import__("json").loads(
                (root / "receipt.json").read_text(encoding="ascii")
            )
            self.assertEqual(
                receipt["adapter_model_sha256"],
                method.pairs.file_sha256(
                    root / "adapter/adapter_model.safetensors"
                ),
            )
            self.assertEqual(
                receipt["adapter_config_sha256"],
                method.pairs.file_sha256(root / "adapter/adapter_config.json"),
            )
            self.assertEqual(Dist.calls, 2)

    def test_v14r3_launcher_gates_four_arms_at_s2_then_stops_at_s8(self):
        launcher = V14R2_LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("scancel", launcher)
        self.assertNotIn("ONLINE_ANCHOR_SOURCE_VARIANT", launcher)
        self.assertEqual(launcher.count("launch_arm_background auh7-1b-gpu-"), 4)
        for node in ("233", "268", "292", "315"):
            self.assertEqual(launcher.count(f"auh7-1b-gpu-{node}"), 1)
        run_arm = launcher.split("run_arm() {", 1)[1].split(
            'if [ "${1:-}" = _arm ]', 1
        )[0]
        self.assertEqual(run_arm.count("gate_smoke"), 2)
        self.assertIn("checkpoint-00000001/receipt.json", run_arm)
        self.assertIn("checkpoint-00000002/receipt.json", run_arm)
        self.assertIn('local short="${arm}_s8_v14r3_gradgeom"', run_arm)
        self.assertIn(
            'launch "$node" "$short" "$route_operator" "$teacher_mode" "$combine_mode" 8',
            run_arm,
        )
        self.assertNotIn("_s32_", run_arm)
        self.assertIn("epsilon_active_tensor_count == 240", launcher)
        self.assertIn("epsilon_active_tensor_count == 480", launcher)
        self.assertIn("lora_A.nonzero_tensor_count == 0", launcher)
        self.assertIn("lora_B.nonzero_tensor_count == 240", launcher)
        self.assertIn(
            "raw_source_caption_trajectory_replay.nonzero_tensor_count == 480",
            launcher,
        )
        for arm in (
            "sameaction_global_actiononly",
            "sameaction_global_norm025",
            "sameaction_gate25_pcgrad010",
            "sameaction_gate25_halfspace001",
        ):
            self.assertIn(arm, launcher)
        for combine_mode in (
            "action_only",
            "norm_balanced_025",
            "action_priority_pcgrad_010",
            "source_halfspace_001",
        ):
            self.assertIn(combine_mode, launcher)
        self.assertIn(
            "source-online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom",
            launcher,
        )
        self.assertIn(
            "online-anchor-targetowned-qk-routed-teacher-v14r3-gradgeom.content.json",
            launcher,
        )
        self.assertIn('_s2_smoke_v14r3_gradgeom"', launcher)
        self.assertIn('_s8_v14r3_gradgeom"', launcher)
        self.assertNotIn('_s32_v14r3_gradgeom"', launcher)
        self.assertIn(
            "after_two_real_component_backwards_before_actual_update_audit_clones",
            launcher,
        )
        self.assertIn(
            ".memory_gate.actual_update_audit_allocations_excluded == true",
            launcher,
        )
        self.assertIn("validate_v14r2_deployment_marker.py", launcher)
        self.assertIn("--role training", launcher)
        self.assertIn("--min-test-count 129", launcher)
        self.assertIn(
            "action_transform_vae_state_alias_source_caption_text_only",
            launcher,
        )
        self.assertIn(
            '.real_source_prebind_state.schema_version == "bernini-real-source-prebind-packed-update-v1"',
            launcher,
        )
        self.assertIn(
            "all(.real_source_prebind_state.action_branches[]; packed_state_diagnostic_ok)",
            launcher,
        )
        self.assertIn(
            "all(.real_source_prebind_state.replay_branches[]; packed_state_diagnostic_ok)",
            launcher,
        )
        self.assertIn('"dtype": "complex128"', launcher)
        self.assertIn('"dtype": "bfloat16"', launcher)
        self.assertIn("$g.selector_transition_count == 1", launcher)
        self.assertIn("$g.tokens_per_video == (", launcher)
        self.assertIn("$g.unpacked_spatial_shape[2] == 21", launcher)
        self.assertIn("bernini-actual-optimizer-update-probe-v1", launcher)
        self.assertIn(".action_descent_passed == true", launcher)
        self.assertIn(".parameter_element_count == 188743680", launcher)
        self.assertIn("actual_optimizer_update_ok($combine_mode; $expected_step)", launcher)
        self.assertNotIn("ONLINE_ANCHOR_GRADIENT_DIAGNOSTIC_ONLY=1", launcher)
        self.assertNotIn(
            ".last_objective_components.real_source_action_prebind_state_diagnostic",
            launcher,
        )
        self.assertIn("--required-file methods/bernini_action_editing/anchor_qk_transport.py", launcher)
        self.assertNotIn("source-online-anchor-targetcoord-routed-teacher-v14\"", launcher)

    def test_v14r2_runner_cannot_import_from_legacy_fallback_trees(self):
        runner = FULLGRID_RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'gradient_diagnostic_only="${ONLINE_ANCHOR_GRADIENT_DIAGNOSTIC_ONLY:-0}"',
            runner,
        )
        self.assertIn('gradient_diagnostic_args+=(--gradient-diagnostic-only)', runner)
        branch = runner.rsplit(
            'if [ "$training_objective" = real_source_target_owned_routed_teacher_delta_v14r2 ]; then',
            1,
        )[1].split("else", 1)[0]
        self.assertIn(
            'export PYTHONPATH="$source_tree/methods/bernini_action_editing"',
            branch,
        )
        self.assertNotIn("$fallback_tree", branch)
        self.assertNotIn("$runtime_tree", branch)


if __name__ == "__main__":
    unittest.main()

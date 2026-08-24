from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

import latent_temporal_event_critic_dataset as dataset  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - dependency-light local host
    torch = None

if torch is not None:
    import latent_temporal_event_critic as critic  # noqa: E402
    import train_latent_temporal_event_critic as trainer  # noqa: E402
else:  # pragma: no cover
    critic = None
    trainer = None


class CriticStaticContractTests(unittest.TestCase):
    def test_model_is_hidden_residual_head_not_video_cnn(self) -> None:
        path = METHOD_ROOT / "latent_temporal_event_critic.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("Conv3d", called_attributes)
        self.assertNotIn("Conv2d", called_attributes)
        text = path.read_text(encoding="utf-8")
        self.assertIn("action - noop", text)
        self.assertIn("fixed_full-support_counter-rademacher_no_mask_v1", text)
        self.assertIn("terminal_hold_phases_17_20", text)

    def test_dataset_requires_every_requested_negative(self) -> None:
        self.assertEqual(
            dataset.DERIVED_NEGATIVE_ROLES,
            (
                "same_video_reverse",
                "same_video_freeze_first",
                "same_video_phase_shuffle",
            ),
        )
        for role in (
            "semantic_noop",
            "semantic_incomplete",
            "semantic_wrong_actor",
            "semantic_wrong_object",
            "semantic_camera_only",
            "semantic_appearance_only",
        ):
            self.assertIn(role, dataset.NEGATIVE_ROLES)


@unittest.skipIf(torch is None, "torch is unavailable")
class CriticTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = critic.CriticConfig(
            hidden_size=16,
            patch_positions=12,
            spatial_coordinates=4,
            spatial_sketch_seed=19,
            projected_size=8,
            model_size=16,
            attention_heads=4,
            transformer_layers=1,
            softmin_temperature=0.25,
            production_geometry=False,
        )
        self.sketch = critic.make_fixed_spatial_sketch(
            patch_positions=12, coordinates=4, seed=19
        )
        self.model = critic.FrozenHiddenTemporalEventCritic(
            self.sketch, config=self.config
        )

    @staticmethod
    def pair(scale: float):
        base = torch.randn(1, 21, 12, 16, dtype=torch.float32)
        noop = 0.05 * base
        action = noop + scale * torch.linspace(0.0, 1.0, 21)[None, :, None, None]
        action = action.expand(-1, -1, 12, 16).contiguous()
        noop = noop.contiguous()
        return action, noop

    def negative_pairs(self):
        return {
            role: self.pair(0.02 + 0.001 * index)
            for index, role in enumerate(dataset.NEGATIVE_ROLES)
        }

    def test_fixed_sketch_and_small_head_are_deterministic(self) -> None:
        repeated = critic.make_fixed_spatial_sketch(
            patch_positions=12, coordinates=4, seed=19
        )
        self.assertTrue(torch.equal(self.sketch, repeated))
        self.assertLess(self.model.trainable_parameter_count, 1_000_000)
        receipt = critic.critic_contract_receipt(self.model)
        self.assertFalse(receipt["generated_video_or_latent_is_editor_target"])
        self.assertTrue(receipt["score_is_differentiable_wrt_current_rv2v_clean_latent"])
        self.assertFalse(receipt["score_alone_authorizes_editor_optimizer"])

    def test_production_head_accepts_each_dynamic_p_but_rejects_wrong_family(self) -> None:
        for positions in (930, 928, 918):
            with self.subTest(patch_positions=positions):
                sketch = critic.make_fixed_spatial_sketch(
                    patch_positions=positions,
                    coordinates=16,
                    seed=20260808017,
                )
                model = critic.FrozenHiddenTemporalEventCritic(sketch)
                self.assertEqual(model.config.patch_positions, positions)
                self.assertEqual(
                    critic.critic_contract_receipt(model)["patch_positions"],
                    positions,
                )
        wrong = critic.make_fixed_spatial_sketch(
            patch_positions=928, coordinates=16, seed=20260808018
        )
        with self.assertRaisesRegex(
            critic.LatentTemporalEventCriticError, "dynamic-P family"
        ):
            critic.FrozenHiddenTemporalEventCritic(wrong)

    def test_forward_is_differentiable_through_action_noop_hidden(self) -> None:
        action, noop = self.pair(0.4)
        action.requires_grad_(True)
        noop.requires_grad_(True)
        output = self.model(action, noop, require_input_grad=False)
        self.assertEqual(tuple(output.milestone_scores.shape), (1, 4))
        gradient = torch.autograd.grad(output.score.sum(), (action, noop))
        self.assertTrue(all(bool(torch.isfinite(row).all().item()) for row in gradient))
        self.assertTrue(all(float(torch.linalg.vector_norm(row).item()) > 0.0 for row in gradient))

    def test_temporal_transforms_are_exact_and_keep_graph(self) -> None:
        latent = torch.arange(21.0, dtype=torch.float32).reshape(1, 1, 21, 1, 1)
        latent.requires_grad_(True)
        reverse = critic.apply_registered_temporal_transform(latent, "reverse")
        freeze = critic.apply_registered_temporal_transform(latent, "freeze_first")
        shuffle = critic.apply_registered_temporal_transform(latent, "phase_shuffle")
        self.assertEqual(reverse.flatten().tolist(), list(reversed(range(21))))
        self.assertEqual(freeze.flatten().tolist(), [0.0] * 21)
        self.assertEqual(
            shuffle.flatten().tolist(), [float((8 * index) % 21) for index in range(21)]
        )
        self.assertTrue(reverse.requires_grad)

    def test_group_ranking_requires_all_roles_and_trains_only_head(self) -> None:
        positive = self.pair(0.5)
        negatives = self.negative_pairs()
        result = critic.group_ranking_loss(self.model, positive, negatives)
        self.assertEqual(
            tuple(result.negative_scores.shape), (len(dataset.NEGATIVE_ROLES), 1)
        )
        missing = dict(negatives)
        missing.pop("semantic_wrong_actor")
        with self.assertRaisesRegex(critic.LatentTemporalEventCriticError, "closure"):
            critic.group_ranking_loss(self.model, positive, missing)

        optimizer = trainer.build_critic_optimizer(self.model)
        positive_residual = self.model.sketch_same_state_hidden_residual(
            *positive
        ).detach()
        negative_residuals = {
            role: self.model.sketch_same_state_hidden_residual(*pair).detach()
            for role, pair in negatives.items()
        }
        batch = trainer.CriticGroupBatch(
            "episode", positive_residual, negative_residuals
        )
        step = trainer.train_critic_groups_one_step(
            self.model, optimizer, [batch]
        )
        self.assertTrue(step.optimizer_step_performed)
        self.assertFalse(step.editor_parameter_present)
        self.assertGreater(step.gradient_norm, 0.0)

    def test_live_score_backpropagates_to_current_rv2v_clean_latent(self) -> None:
        trainer.freeze_fitted_critic_for_reward(self.model)
        # Native 4x12 latent -> 2x6 patch grid -> P=12, matching this instance.
        clean = torch.randn(1, 16, 21, 4, 12, dtype=torch.float32, requires_grad=True)
        epsilon = torch.randn_like(clean).detach()
        calls = []

        def frozen_bridge(request):
            calls.append((request.role, id(request.x_sigma), request.source_condition))
            phase_channels = request.x_sigma.mean(dim=(3, 4)).permute(0, 2, 1)
            owner = phase_channels[:, :, None, :].expand(-1, -1, 12, -1)
            multiplier = 1.15 if request.role == "action" else 0.85
            return multiplier * owner

        live = critic.score_current_rv2v_clean_latent(
            self.model,
            clean,
            epsilon,
            action_condition="action condition",
            noop_condition="no-op condition",
            frozen_hidden_callback=frozen_bridge,
        )
        self.assertEqual([row[0] for row in calls], ["action", "noop"])
        self.assertEqual(len({row[1] for row in calls}), 1)
        self.assertTrue(all(row[2] is None for row in calls))
        audit = trainer.audit_current_clean_latent_gradient(live, clean)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["generated_t2v_target_consumed"])
        self.assertFalse(audit["editor_optimizer_authorized"])


if __name__ == "__main__":
    unittest.main()

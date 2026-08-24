from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(METHOD_ROOT))

import self_imagined_motion_cotangent_v1 as method  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - dependency-light hosts
    torch = None


ASSET = METHOD_ROOT / "assets/self_imagined_motion_cotangent_core2_v1.json"
SMALL_RESIDUAL_SHAPE = (1, 21, 2, 3)


def _provenance(seed: int, *, cell_id: str = "dog") -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "owner_generation_seed": 2026081501,
        "query_seed": seed,
        "owner_mode": "frozen_bernini_pure_t2v",
        "owner_exact81_action_audit_passed": True,
        "owner_used_source_video_condition": False,
    }


class RegistryAndPlanTests(unittest.TestCase):
    def test_core2_registry_is_strict_and_directly_plans_all8(self) -> None:
        raw = ASSET.read_bytes()
        registry = method.load_probe_registry(
            ASSET.resolve(), expected_file_sha256=hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual([row.cell_id for row in registry.cells], ["dog", "human"])
        self.assertEqual(registry.cell("dog").latent_shape, (1, 16, 21, 60, 62))
        self.assertEqual(registry.cell("human").latent_shape, (1, 16, 21, 64, 58))
        self.assertFalse(registry.contract["seed_selection"])
        self.assertFalse(registry.contract["seed_averaging"])
        self.assertEqual(
            registry.contract["failure_policy"],
            "null_no_intervention_no_decode_no_update",
        )
        self.assertTrue(registry.contract["same_topology_reverse_query_required"])
        self.assertNotEqual(
            registry.cell("dog").action_family_id,
            registry.cell("dog").reverse_wrong_family_id,
        )
        self.assertEqual(
            registry.contract["owner_to_editor_allowed_channel"],
            method.ALLOWED_OWNER_TO_EDITOR_CHANNEL,
        )
        plan = method.build_auh_dual4_execution_plan(registry)
        self.assertEqual(plan["world_size"], 8)
        self.assertEqual(plan["groups"][0]["sp4_ranks"], [0, 1, 2, 3])
        self.assertEqual(plan["groups"][1]["sp4_ranks"], [4, 5, 6, 7])
        self.assertEqual(plan["symmetric_exact81_decode_count"], 8)
        self.assertEqual(plan["optimizer_steps"], 0)
        unsigned = dict(plan)
        digest = unsigned.pop("receipt_digest")
        self.assertEqual(digest, method.object_sha256(unsigned))

    def test_registry_rejects_seed_selection_collision_and_prompt_mutation(self) -> None:
        value = json.loads(ASSET.read_text(encoding="utf-8"))
        bad = copy.deepcopy(value)
        bad["contract"]["seed_selection"] = True
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "seed_selection"
        ):
            method.validate_probe_registry(bad)
        bad = copy.deepcopy(value)
        bad["cells"][1]["query_seeds"][0] = bad["cells"][0]["query_seeds"][0]
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "unique"
        ):
            method.validate_probe_registry(bad)
        bad = copy.deepcopy(value)
        bad["cells"][0]["action_caption"] += " changed"
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "action_caption bytes"
        ):
            method.validate_probe_registry(bad)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class TensorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260809)
        self.shape_patch = mock.patch.object(
            method, "RESIDUAL_SHAPE", SMALL_RESIDUAL_SHAPE
        )
        self.shape_patch.start()

    def tearDown(self) -> None:
        self.shape_patch.stop()

    def test_temporal_quotient_is_spatial_orderless_and_keeps_time_order(self) -> None:
        residual = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        static = torch.randn((1, 1, 2, 3), dtype=torch.float32)
        feature = method.temporal_motion_quotient(residual)
        shifted = method.temporal_motion_quotient(residual + static)
        spatial = torch.randn((1, 21, 7, 3), dtype=torch.float32)
        spatial_feature = method.temporal_motion_quotient(spatial)
        permuted = method.temporal_motion_quotient(
            spatial[:, :, torch.tensor([3, 0, 6, 2, 5, 1, 4]), :]
        )
        translated = method.temporal_motion_quotient(spatial.roll(3, dims=2))
        reversed_feature = method.temporal_motion_quotient(residual.flip(1))
        self.assertTrue(torch.allclose(feature, shifted, atol=2.0e-6, rtol=2.0e-6))
        self.assertTrue(
            torch.allclose(spatial_feature, permuted, atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(spatial_feature, translated, atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertLess(
            method.cosine_similarity(
                feature, reversed_feature, label="forward/reverse quotient"
            ),
            0.95,
        )

    def test_owner_template_is_detached_and_scorer_has_live_candidate_gradient(self) -> None:
        owner = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        template = method.build_frozen_owner_template(
            owner, query_seed=2026081502, owner_provenance=_provenance(2026081502)
        )
        self.assertFalse(template.unit_feature.requires_grad)
        self.assertIsNone(template.unit_feature.grad_fn)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(template.unit_feature).item()), 1.0, places=5
        )
        scorer = method.make_frozen_per_query_scorer(template)
        self.assertEqual(sum(parameter.numel() for parameter in scorer.parameters()), 0)
        # The editor may have a different spatial token/sketch count from the
        # pure-T2V owner; Phi is independent of K.
        candidate_leaf = torch.randn(
            (1, 21, 5, 3), dtype=torch.float32, requires_grad=True
        )
        candidate = candidate_leaf * 1.0
        output = scorer.forward_sketched_residual(
            candidate, require_input_grad=True
        )
        gradient = torch.autograd.grad(output.score, candidate_leaf)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(torch.linalg.vector_norm(gradient).item()), 0.0)

    def test_owner_template_rejects_primal_condition_or_unaudited_media(self) -> None:
        owner = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        provenance = _provenance(2026081502)
        provenance["owner_used_source_video_condition"] = True
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "provenance"
        ):
            method.build_frozen_owner_template(
                owner,
                query_seed=2026081502,
                owner_provenance=provenance,
            )
        provenance = _provenance(2026081502)
        provenance["owner_exact81_action_audit_passed"] = False
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "provenance"
        ):
            method.build_frozen_owner_template(
                owner,
                query_seed=2026081502,
                owner_provenance=provenance,
            )

    def test_two_seed_template_audit_never_selects_or_averages(self) -> None:
        owner = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        templates = tuple(
            method.build_frozen_owner_template(
                owner + index * 1.0e-3,
                query_seed=seed,
                owner_provenance=_provenance(seed),
            )
            for index, seed in enumerate((2026081502, 2026081503))
        )
        audit = method.audit_two_seed_templates(templates, minimum_cosine=0.05)
        self.assertTrue(audit.passed)
        receipt = audit.receipt()
        self.assertFalse(receipt["seed_ranking_or_selection"])
        self.assertFalse(receipt["seed_averaging"])

    def test_same_topology_reverse_margin_rejects_generic_motion(self) -> None:
        seed = 2026081502
        action = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        template = method.build_frozen_owner_template(
            action, query_seed=seed, owner_provenance=_provenance(seed)
        )
        null = torch.zeros_like(action)
        failed = method.audit_prompt_specificity(
            template,
            action_residual=action,
            reverse_wrong_family_residual=action.clone(),
            common_scene_null_residual=null,
            same_x_sigma_binding_digest="a" * 64,
            minimum_margin=0.1,
        )
        self.assertFalse(failed.passed)
        passed = method.audit_prompt_specificity(
            template,
            action_residual=action,
            reverse_wrong_family_residual=action.flip(1).contiguous(),
            common_scene_null_residual=null,
            same_x_sigma_binding_digest="a" * 64,
            minimum_margin=0.1,
        )
        self.assertTrue(passed.passed)
        self.assertGreaterEqual(passed.reverse_wrong_family_margin, 0.1)
        self.assertTrue(passed.receipt()["same_topology_reverse_query_required"])

    def test_mask_free_projection_and_symmetric_pair_contract(self) -> None:
        gradient = torch.randn((1, 16, 21, 6, 8), dtype=torch.float32)
        projected = method.project_mask_free_nuisance_cotangent(
            gradient, minimum_survival_cosine=0.0
        )
        self.assertEqual(float(projected.tensor[:, :, 0].abs().max().item()), 0.0)
        self.assertLess(
            float(projected.tensor[:, :, 1:].sum(dim=2).abs().max().item()),
            2.0e-5,
        )
        self.assertLess(projected.spatial_affine_max_abs_dot, 2.0e-5)
        base = torch.randn_like(gradient)
        pair = method.make_symmetric_intervention_pair(
            base,
            projected,
            query_seed=2026081502,
            relative_l2_dose=0.01,
        )
        self.assertTrue(
            torch.allclose((pair.plus + pair.minus) * 0.5, base, atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertTrue(
            torch.allclose(pair.plus - base, -(pair.minus - base), atol=2.0e-6, rtol=2.0e-6)
        )
        self.assertAlmostEqual(pair.delta_norm / pair.base_norm, 0.01, places=6)
        self.assertFalse(pair.receipt()["seed_or_arm_selection"])

    def test_two_seed_live_runtime_keeps_both_directional_pairs(self) -> None:
        owner = torch.randn(SMALL_RESIDUAL_SHAPE, dtype=torch.float32)
        seeds = (2026081502, 2026081503)
        templates = tuple(
            method.build_frozen_owner_template(
                owner + index * 1.0e-4,
                query_seed=seed,
                owner_provenance=_provenance(seed),
            )
            for index, seed in enumerate(seeds)
        )
        specificity = {
            seed: method.audit_prompt_specificity(
                template,
                action_residual=owner + index * 1.0e-4,
                reverse_wrong_family_residual=owner.flip(1).contiguous(),
                common_scene_null_residual=torch.zeros_like(owner),
                same_x_sigma_binding_digest=hashlib.sha256(
                    f"same-state-{seed}".encode("ascii")
                ).hexdigest(),
                minimum_margin=0.1,
            )
            for index, (seed, template) in enumerate(zip(seeds, templates))
        }
        self.assertTrue(all(row.passed for row in specificity.values()))
        base = torch.randn((1, 16, 21, 6, 8), dtype=torch.float32)
        noises = {
            seed: torch.randn_like(base).detach() for seed in seeds
        }
        calls: list[int] = []

        class FakeBridge:
            def __init__(self, seed: int, scorer: object) -> None:
                self.seed = seed
                self.scorer = scorer

            def prove_current_clean_latent_vjp(
                self, clean: torch.Tensor, noise: torch.Tensor, *, minimum_norm: float
            ) -> SimpleNamespace:
                calls.append(self.seed)
                # Shared signal plus a small seed-specific component ensures a
                # positive two-seed direction without hiding either result.
                generator = torch.Generator(device=clean.device)
                generator.manual_seed(self.seed)
                gradient = (
                    0.2 * clean.detach()
                    + 0.01
                    * torch.randn(
                        clean.shape,
                        generator=generator,
                        device=clean.device,
                        dtype=clean.dtype,
                    )
                ).contiguous()
                return SimpleNamespace(
                    gradient=gradient,
                    gradient_norm=float(torch.linalg.vector_norm(gradient).item()),
                    critic_score=0.25,
                    real_sp4_autograd_collective=True,
                    hook_call_order=("action", "noop"),
                )

        result = method.run_two_query_seed_live_probe(
            templates=templates,
            specificity_audits_by_seed=specificity,
            current_clean_latent=base,
            noises_by_seed=noises,
            bridge_factory=lambda seed, scorer: FakeBridge(seed, scorer),
            relative_l2_dose=0.01,
            minimum_template_cosine=0.05,
            minimum_projected_cotangent_cosine=0.0,
            minimum_projection_survival_cosine=0.0,
        )
        self.assertEqual(calls, list(seeds))
        self.assertEqual(
            [row.query_seed for row in result.seed_probes], list(seeds)
        )
        self.assertTrue(result.two_seed_direction_consistent)
        receipt = result.receipt()
        self.assertFalse(receipt["seed_averaging"])
        self.assertFalse(receipt["seed_ranking_or_selection"])
        self.assertEqual(len(receipt["per_seed"]), 2)
        self.assertFalse(receipt["optimizer_or_parameter_update"])


class DirectionGateTests(unittest.TestCase):
    @staticmethod
    def make_outcome(seed: int, *, action: bool = True) -> dict[str, object]:
        return {
            "query_seed": seed,
            "frame_count_plus": 81,
            "frame_count_minus": 81,
            "plus_action_better_than_minus": action,
            "plus_reverse_wrong_family_not_improved_vs_minus_or_base": True,
            "plus_source_identity_noninferior_to_base": True,
            "plus_camera_noninferior_to_base": True,
            "plus_background_noninferior_to_base": True,
            "plus_quality_noninferior_to_base": True,
            "plus_temporal_consistency_noninferior_to_base": True,
            "audited_without_seed_or_arm_selection": True,
        }

    def test_gate_requires_both_seeds_and_never_authorizes_update(self) -> None:
        seeds = (2026081502, 2026081503)
        gate = method.evaluate_two_seed_direction_gate(
            [self.make_outcome(seeds[0]), self.make_outcome(seeds[1])],
            ordered_query_seeds=seeds,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["observed_pass_count"], 2)
        self.assertFalse(gate["parameter_update_authorized"])
        self.assertFalse(gate["action_editing_success_claim_authorized"])
        failed = method.evaluate_two_seed_direction_gate(
            [
                self.make_outcome(seeds[0]),
                self.make_outcome(seeds[1], action=False),
            ],
            ordered_query_seeds=seeds,
        )
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["next_authority_if_passed"], "null_stop")

    def test_gate_rejects_non_exact81_or_seed_reordering(self) -> None:
        seeds = (2026081502, 2026081503)
        rows = [self.make_outcome(seeds[0]), self.make_outcome(seeds[1])]
        rows[0]["frame_count_plus"] = 41
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "exact81"
        ):
            method.evaluate_two_seed_direction_gate(rows, ordered_query_seeds=seeds)
        with self.assertRaisesRegex(
            method.SelfImaginedCotangentContractError, "exact81"
        ):
            method.evaluate_two_seed_direction_gate(
                [self.make_outcome(seeds[1]), self.make_outcome(seeds[0])],
                ordered_query_seeds=seeds,
            )


if __name__ == "__main__":
    unittest.main()
